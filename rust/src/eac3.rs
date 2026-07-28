//! E-AC-3 sync-frame and EMDF extraction.
//!
//! Audio coding remains delegated to `FFmpeg`. This module owns the parts that
//! `FFmpeg` does not expose: frame timing and the OAMD/JOC payloads transported
//! in E-AC-3 auxiliary data. EMDF sync words are byte-aligned inside an
//! auxiliary stream, but that stream can begin at any bit offset in the parent
//! sync frame, so all eight alignments are searched.

use std::io::{self, Read};

use crate::{error::AppError, stream_io::read_up_to};

const EAC3_SYNC_WORD: u16 = 0x0b77;
const EMDF_SYNC_WORD: u16 = 0x5838;
const MAX_SYNC_FRAME_BYTES: usize = 4096;
const MAX_EMDF_PAYLOAD_BYTES: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StreamType {
    Independent,
    Dependent,
    Ac3Converted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SyncFrameHeader {
    pub stream_type: StreamType,
    pub substream_id: u8,
    pub frame_bytes: usize,
    pub sample_rate: u32,
    pub blocks: u8,
    pub channel_mode: u8,
    pub lfe: bool,
    pub bitstream_id: u8,
}

impl SyncFrameHeader {
    /// Parses the fixed portion of an E-AC-3 sync-frame header.
    ///
    /// # Errors
    ///
    /// Rejects truncated input, an invalid sync word, reserved values, AC-3
    /// frames, and impossible frame lengths.
    pub fn parse(bytes: &[u8]) -> Result<Self, AppError> {
        let mut bits = BitReader::new(bytes);
        if bits.read(16)? != u64::from(EAC3_SYNC_WORD) {
            return Err(corrupt("missing E-AC-3 sync word"));
        }
        let stream_type = match bits.read_u8(2)? {
            0 => StreamType::Independent,
            1 => StreamType::Dependent,
            2 => StreamType::Ac3Converted,
            _ => return Err(corrupt("reserved E-AC-3 stream type")),
        };
        let substream_id = bits.read_u8(3)?;
        let frame_bytes = (bits.read_usize(11)? + 1)
            .checked_mul(2)
            .ok_or_else(|| corrupt("E-AC-3 frame size overflowed"))?;
        if !(7..=MAX_SYNC_FRAME_BYTES).contains(&frame_bytes) {
            return Err(corrupt(format!(
                "invalid E-AC-3 frame length: {frame_bytes} bytes"
            )));
        }

        let sample_rate_code = bits.read_u8(2)?;
        let (sample_rate, blocks) = if sample_rate_code == 3 {
            let reduced_rate_code = bits.read_u8(2)?;
            let sample_rate = [24_000, 22_050, 16_000]
                .get(usize::from(reduced_rate_code))
                .copied()
                .ok_or_else(|| corrupt("reserved reduced E-AC-3 sample rate"))?;
            (sample_rate, 6)
        } else {
            let sample_rate = [48_000, 44_100, 32_000][usize::from(sample_rate_code)];
            let blocks = [1, 2, 3, 6][bits.read_usize(2)?];
            (sample_rate, blocks)
        };
        let channel_mode = bits.read_u8(3)?;
        let lfe = bits.read_bit()?;
        let bitstream_id = bits.read_u8(5)?;
        if !(11..=16).contains(&bitstream_id) {
            return Err(corrupt(format!(
                "sync frame is not E-AC-3 (bitstream ID {bitstream_id})"
            )));
        }

        Ok(Self {
            stream_type,
            substream_id,
            frame_bytes,
            sample_rate,
            blocks,
            channel_mode,
            lfe,
            bitstream_id,
        })
    }

    #[must_use]
    pub const fn sample_count(self) -> usize {
        self.blocks as usize * 256
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MetadataPayload {
    pub id: u32,
    pub sample_offset: Option<u16>,
    pub data: Vec<u8>,
    pub bit_len: usize,
}

impl MetadataPayload {
    #[must_use]
    pub fn bits(&self) -> BitReader<'_> {
        BitReader::with_limit(&self.data, self.bit_len)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Eac3Frame {
    pub header: SyncFrameHeader,
    pub sample_start: u64,
    pub payloads: Vec<MetadataPayload>,
}

/// Streams raw E-AC-3 sync frames and extracts valid EMDF payloads.
pub struct FrameReader<R> {
    input: R,
    sample_start: u64,
    current_program_start: u64,
    frame_index: u64,
}

impl<R: Read> FrameReader<R> {
    #[must_use]
    pub const fn new(input: R) -> Self {
        Self {
            input,
            sample_start: 0,
            current_program_start: 0,
            frame_index: 0,
        }
    }

    /// Reads the next complete sync frame.
    ///
    /// # Errors
    ///
    /// Rejects truncated or malformed frames, sample-rate changes, and I/O
    /// failures. A clean end of stream returns `None`.
    pub fn next_frame(&mut self) -> Result<Option<Eac3Frame>, AppError> {
        let mut prefix = [0_u8; 7];
        let prefix_len = read_up_to(&mut self.input, &mut prefix)?;
        if prefix_len == 0 {
            return Ok(None);
        }
        if prefix_len != prefix.len() {
            return Err(corrupt(format!(
                "E-AC-3 stream ended inside a sync-frame header after {prefix_len} bytes"
            )));
        }
        let header = SyncFrameHeader::parse(&prefix)?;
        let mut frame = vec![0_u8; header.frame_bytes];
        frame[..prefix.len()].copy_from_slice(&prefix);
        self.input
            .read_exact(&mut frame[prefix.len()..])
            .map_err(|error| match error.kind() {
                io::ErrorKind::UnexpectedEof => corrupt(format!(
                    "E-AC-3 stream ended inside frame {} (expected {} bytes)",
                    self.frame_index, header.frame_bytes
                )),
                _ => AppError::Io(error),
            })?;

        if header.stream_type == StreamType::Independent && header.substream_id == 0 {
            self.current_program_start = self.sample_start;
            self.sample_start =
                self.sample_start
                    .checked_add(u64::try_from(header.sample_count()).map_err(|error| {
                        corrupt(format!("E-AC-3 sample count overflowed: {error}"))
                    })?)
                    .ok_or_else(|| corrupt("E-AC-3 sample timeline overflowed"))?;
        }
        let payloads = extract_emdf_payloads(&frame);
        self.frame_index = self
            .frame_index
            .checked_add(1)
            .ok_or_else(|| corrupt("E-AC-3 frame count overflowed"))?;
        Ok(Some(Eac3Frame {
            header,
            sample_start: self.current_program_start,
            payloads,
        }))
    }
}

/// Finds every structurally valid EMDF block in one sync frame.
///
/// Candidate sync words are searched at all possible bit alignments. Invalid
/// candidates are ignored because compressed audio can contain the same 16-bit
/// pattern by chance; the EMDF length, version, key, payload IDs, and payload
/// bounds collectively validate a real block.
#[must_use]
pub fn extract_emdf_payloads(frame: &[u8]) -> Vec<MetadataPayload> {
    let bit_len = frame.len().saturating_mul(8);
    let mut result = Vec::new();
    let mut accepted_ranges = Vec::<(usize, usize)>::new();
    for alignment in 0_usize..8 {
        let mut position = alignment;
        while position.saturating_add(32) <= bit_len {
            if read_u16_at(frame, position) == Some(EMDF_SYNC_WORD)
                && !accepted_ranges
                    .iter()
                    .any(|(start, end)| position >= *start && position < *end)
                && let Ok((end, payloads)) = parse_emdf_block(frame, position)
            {
                accepted_ranges.push((position, end));
                result.extend(payloads);
                position = end;
                continue;
            }
            position += 8;
        }
    }
    result
}

fn parse_emdf_block(frame: &[u8], start: usize) -> Result<(usize, Vec<MetadataPayload>), AppError> {
    let mut bits = BitReader::at(frame, start);
    if bits.read(16)? != u64::from(EMDF_SYNC_WORD) {
        return Err(corrupt("invalid EMDF candidate"));
    }
    let frame_bytes = bits.read_usize(16)?;
    let end = bits
        .position()
        .checked_add(
            frame_bytes
                .checked_mul(8)
                .ok_or_else(|| corrupt("EMDF length overflowed"))?,
        )
        .ok_or_else(|| corrupt("EMDF endpoint overflowed"))?;
    if frame_bytes == 0 || end > frame.len().saturating_mul(8) {
        return Err(corrupt("EMDF block extends past its E-AC-3 frame"));
    }
    bits.set_limit(end)?;

    let mut version = bits.read_u32(2)?;
    if version == 3 {
        version = version
            .checked_add(bits.variable(2, None)?)
            .ok_or_else(|| corrupt("EMDF version overflowed"))?;
    }
    let mut key = bits.read_u32(3)?;
    if key == 7 {
        key = key
            .checked_add(bits.variable(3, None)?)
            .ok_or_else(|| corrupt("EMDF key overflowed"))?;
    }
    if version != 0 || key != 0 {
        return Err(corrupt("unsupported EMDF version or key"));
    }

    let mut payloads = Vec::new();
    while bits.remaining() >= 5 {
        let mut payload_id = bits.read_u32(5)?;
        if payload_id == 0 {
            break;
        }
        if payload_id == 31 {
            payload_id = payload_id
                .checked_add(bits.variable(5, None)?)
                .ok_or_else(|| corrupt("EMDF payload ID overflowed"))?;
        }
        if payload_id > 14 {
            return Err(corrupt("unsupported EMDF payload ID"));
        }

        let has_sample_offset = bits.read_bit()?;
        let sample_offset = has_sample_offset
            .then(|| bits.read_u16(12).map(|value| value >> 1))
            .transpose()?;
        if bits.read_bit()? {
            let _ = bits.variable(11, None)?;
        }
        if bits.read_bit()? {
            let _ = bits.variable(2, None)?;
        }
        if bits.read_bit()? {
            bits.skip(8)?;
        }
        if !bits.read_bit()? {
            let mut frame_aligned = false;
            if !has_sample_offset {
                frame_aligned = bits.read_bit()?;
                if frame_aligned {
                    bits.skip(2)?;
                }
            }
            if has_sample_offset || frame_aligned {
                bits.skip(7)?;
            }
        }

        let payload_bytes = bits.variable(8, None)? as usize;
        if payload_bytes > MAX_EMDF_PAYLOAD_BYTES {
            return Err(corrupt("EMDF payload exceeds the safety limit"));
        }
        let payload_bits = payload_bytes
            .checked_mul(8)
            .ok_or_else(|| corrupt("EMDF payload length overflowed"))?;
        if payload_bits > bits.remaining() {
            return Err(corrupt("EMDF payload extends past its block"));
        }
        let data = bits.copy_bits(payload_bits)?;
        payloads.push(MetadataPayload {
            id: payload_id,
            sample_offset,
            data,
            bit_len: payload_bits,
        });
    }
    Ok((end, payloads))
}

#[derive(Clone, Debug)]
pub struct BitReader<'a> {
    bytes: &'a [u8],
    position: usize,
    limit: usize,
}

#[allow(clippy::missing_errors_doc)] // Low-level readers share one truncation/corruption contract.
impl<'a> BitReader<'a> {
    #[must_use]
    pub const fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            position: 0,
            limit: bytes.len() * 8,
        }
    }

    #[must_use]
    pub const fn with_limit(bytes: &'a [u8], bit_len: usize) -> Self {
        Self {
            bytes,
            position: 0,
            limit: bit_len,
        }
    }

    #[must_use]
    pub const fn at(bytes: &'a [u8], position: usize) -> Self {
        Self {
            bytes,
            position,
            limit: bytes.len() * 8,
        }
    }

    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    #[must_use]
    pub const fn limit(&self) -> usize {
        self.limit
    }

    #[must_use]
    pub const fn remaining(&self) -> usize {
        self.limit.saturating_sub(self.position)
    }

    pub fn set_limit(&mut self, limit: usize) -> Result<(), AppError> {
        if limit < self.position || limit > self.bytes.len().saturating_mul(8) {
            return Err(corrupt("invalid bit-reader limit"));
        }
        self.limit = limit;
        Ok(())
    }

    pub fn set_position(&mut self, position: usize) -> Result<(), AppError> {
        if position > self.limit {
            return Err(corrupt("bit-reader position exceeds its limit"));
        }
        self.position = position;
        Ok(())
    }

    pub fn read_bit(&mut self) -> Result<bool, AppError> {
        Ok(self.read(1)? != 0)
    }

    pub fn read_u8(&mut self, count: usize) -> Result<u8, AppError> {
        u8::try_from(self.read(count)?)
            .map_err(|error| corrupt(format!("bit field does not fit in u8: {error}")))
    }

    pub fn read_u16(&mut self, count: usize) -> Result<u16, AppError> {
        u16::try_from(self.read(count)?)
            .map_err(|error| corrupt(format!("bit field does not fit in u16: {error}")))
    }

    pub fn read_u32(&mut self, count: usize) -> Result<u32, AppError> {
        u32::try_from(self.read(count)?)
            .map_err(|error| corrupt(format!("bit field does not fit in u32: {error}")))
    }

    pub fn read_usize(&mut self, count: usize) -> Result<usize, AppError> {
        usize::try_from(self.read(count)?)
            .map_err(|error| corrupt(format!("bit field does not fit in usize: {error}")))
    }

    pub fn read_signed(&mut self, count: usize) -> Result<i32, AppError> {
        if count == 0 || count > i32::BITS as usize {
            return Err(corrupt("invalid signed bit-field width"));
        }
        let unsigned = self.read_u32(count)?;
        let sign = 1_u32 << (count - 1);
        let value = if unsigned & sign == 0 {
            i64::from(unsigned)
        } else {
            i64::from(unsigned) - (1_i64 << count)
        };
        i32::try_from(value)
            .map_err(|error| corrupt(format!("signed bit field does not fit in i32: {error}")))
    }

    pub fn read(&mut self, count: usize) -> Result<u64, AppError> {
        if count > u64::BITS as usize || count > self.remaining() {
            return Err(corrupt(format!(
                "truncated bitstream at bit {} of {}: requested {count} bits with {} remaining",
                self.position,
                self.limit,
                self.remaining(),
            )));
        }
        let mut value = 0_u64;
        for _ in 0..count {
            let byte = self.bytes[self.position >> 3];
            let bit = (byte >> (7 - (self.position & 7))) & 1;
            value = (value << 1) | u64::from(bit);
            self.position += 1;
        }
        Ok(value)
    }

    pub fn skip(&mut self, count: usize) -> Result<(), AppError> {
        if count > self.remaining() {
            return Err(corrupt("truncated bitstream while skipping a field"));
        }
        self.position += count;
        Ok(())
    }

    /// Reads the EMDF variable-length integer defined by ETSI TS 102 366.
    pub fn variable(&mut self, chunk_bits: usize, limit: Option<usize>) -> Result<u32, AppError> {
        if chunk_bits == 0 || chunk_bits > 31 {
            return Err(corrupt("invalid variable-field chunk width"));
        }
        let mut value = 0_u32;
        let mut extension = 0_usize;
        loop {
            value = value
                .checked_add(self.read_u32(chunk_bits)?)
                .ok_or_else(|| corrupt("variable-length value overflowed"))?;
            let read_more = self.read_bit()?;
            if !read_more || limit.is_some_and(|maximum| extension >= maximum) {
                return Ok(value);
            }
            value = value
                .checked_add(1)
                .and_then(|value| value.checked_shl(u32::try_from(chunk_bits).ok()?))
                .ok_or_else(|| corrupt("variable-length value overflowed"))?;
            extension += 1;
        }
    }

    fn copy_bits(&mut self, count: usize) -> Result<Vec<u8>, AppError> {
        let mut data = vec![0_u8; count.div_ceil(8)];
        for bit in 0..count {
            if self.read_bit()? {
                data[bit >> 3] |= 1 << (7 - (bit & 7));
            }
        }
        Ok(data)
    }
}

fn read_u16_at(bytes: &[u8], position: usize) -> Option<u16> {
    if position.checked_add(16)? > bytes.len().checked_mul(8)? {
        return None;
    }
    let byte = position >> 3;
    let shift = position & 7;
    if shift == 0 {
        return Some(u16::from_be_bytes([bytes[byte], bytes[byte + 1]]));
    }
    let window = (u32::from(bytes[byte]) << 16)
        | (u32::from(bytes[byte + 1]) << 8)
        | u32::from(*bytes.get(byte + 2).unwrap_or(&0));
    Some(((window >> (8 - shift)) & 0xffff) as u16)
}

fn corrupt(message: impl Into<String>) -> AppError {
    AppError::Render(format!("invalid E-AC-3 metadata: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{
        BitReader, FrameReader, MetadataPayload, StreamType, SyncFrameHeader, extract_emdf_payloads,
    };

    #[test]
    fn parses_full_rate_header() {
        let frame = make_sync_frame(0, 0, 3, 7, false, 11);
        let header = SyncFrameHeader::parse(&frame).unwrap();
        assert_eq!(header.stream_type, StreamType::Independent);
        assert_eq!(header.frame_bytes, 16);
        assert_eq!(header.sample_rate, 48_000);
        assert_eq!(header.blocks, 6);
        assert_eq!(header.channel_mode, 7);
        assert!(!header.lfe);
    }

    #[test]
    fn frame_reader_tracks_primary_program_samples() {
        let first = make_sync_frame(0, 0, 3, 7, false, 11);
        let dependent = make_sync_frame(1, 0, 3, 7, false, 11);
        let second = make_sync_frame(0, 0, 1, 7, false, 11);
        let bytes = [first, dependent, second].concat();
        let mut reader = FrameReader::new(Cursor::new(bytes));
        let first = reader.next_frame().unwrap().unwrap();
        let dependent = reader.next_frame().unwrap().unwrap();
        let second = reader.next_frame().unwrap().unwrap();
        assert_eq!(first.sample_start, 0);
        assert_eq!(dependent.sample_start, 0);
        assert_eq!(second.sample_start, 1536);
        assert_eq!(second.header.sample_count(), 512);
        assert!(reader.next_frame().unwrap().is_none());
    }

    #[test]
    fn truncated_sync_frame_is_rejected_without_partial_output() {
        let frame = make_sync_frame(0, 0, 3, 7, true, 11);
        for length in 1..frame.len() {
            let mut reader = FrameReader::new(Cursor::new(&frame[..length]));
            assert!(
                reader.next_frame().is_err(),
                "accepted frame truncated to {length} bytes"
            );
        }
    }

    #[test]
    fn extracts_unaligned_emdf_payload() {
        let payload = MetadataPayload {
            id: 11,
            sample_offset: Some(37),
            data: vec![0xa5],
            bit_len: 8,
        };
        let emdf = make_emdf(&payload);
        let mut frame = vec![0_u8; 4 + emdf.len()];
        copy_unaligned(&emdf, &mut frame, 3);
        let decoded = extract_emdf_payloads(&frame);
        assert_eq!(decoded, [payload]);
    }

    #[test]
    fn variable_length_integer_matches_emdf_rule() {
        // 0b00111 = chunk 3 + extension, then 0b0100 = chunk 2 + stop:
        // ((3 + 1) << 3) + 2 = 34.
        let mut bits = BitReader::with_limit(&[0b0111_0100], 8);
        assert_eq!(bits.variable(3, None).unwrap(), 34);
    }

    fn make_sync_frame(
        stream_type: u8,
        substream: u8,
        blocks_code: u8,
        channel_mode: u8,
        lfe: bool,
        bitstream_id: u8,
    ) -> Vec<u8> {
        let mut bits = BitWriter::default();
        bits.write(0x0b77, 16);
        bits.write(u64::from(stream_type), 2);
        bits.write(u64::from(substream), 3);
        bits.write(7, 11); // 8 words / 16 bytes
        bits.write(0, 2); // 48 kHz
        bits.write(u64::from(blocks_code), 2);
        bits.write(u64::from(channel_mode), 3);
        bits.write(u64::from(lfe), 1);
        bits.write(u64::from(bitstream_id), 5);
        bits.bytes.resize(16, 0);
        bits.bytes
    }

    fn make_emdf(payload: &MetadataPayload) -> Vec<u8> {
        let mut body = BitWriter::default();
        body.write(0, 2); // version
        body.write(0, 3); // key
        body.write(u64::from(payload.id), 5);
        body.write(1, 1); // sample offset present
        body.write(u64::from(payload.sample_offset.unwrap()) << 1, 12);
        body.write(0, 1); // duration
        body.write(0, 1); // group
        body.write(0, 1); // codec data
        body.write(1, 1); // discard unknown
        body.write(payload.data.len() as u64, 8);
        body.write(0, 1); // no size extension
        for byte in &payload.data {
            body.write(u64::from(*byte), 8);
        }
        body.write(0, 5); // end marker
        while body.bit_len % 8 != 0 {
            body.write(0, 1);
        }

        let mut result = BitWriter::default();
        result.write(0x5838, 16);
        result.write(body.bytes.len() as u64, 16);
        for byte in body.bytes {
            result.write(u64::from(byte), 8);
        }
        result.bytes
    }

    fn copy_unaligned(source: &[u8], target: &mut [u8], start: usize) {
        for bit in 0..source.len() * 8 {
            if source[bit >> 3] & (1 << (7 - (bit & 7))) != 0 {
                let target_bit = start + bit;
                target[target_bit >> 3] |= 1 << (7 - (target_bit & 7));
            }
        }
    }

    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        bit_len: usize,
    }

    impl BitWriter {
        fn write(&mut self, value: u64, count: usize) {
            for shift in (0..count).rev() {
                if self.bit_len.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                if (value >> shift) & 1 != 0 {
                    let offset = self.bit_len & 7;
                    *self.bytes.last_mut().unwrap() |= 1 << (7 - offset);
                }
                self.bit_len += 1;
            }
        }
    }
}
