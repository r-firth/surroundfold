//! Joint Object Coding side-information decoder.
//!
//! This implements clauses 6.2 through 6.6 of ETSI TS 103 420 V1.1.1.
//! The decoded matrix is ready to apply to complex 64-band QMF samples.

use crate::{
    eac3::{BitReader, MetadataPayload},
    error::AppError,
    joc_tables::{
        COARSE_GENERIC, COARSE_SPARSE, FINE_GENERIC, FINE_SPARSE, FIVE_CHANNEL_INDEX,
        SEVEN_CHANNEL_INDEX,
    },
};

pub const QMF_SUBBANDS: usize = 64;
const JOC_PAYLOAD_ID: u32 = 14;
const MAX_OBJECTS: usize = 64;
const BAND_COUNTS: [usize; 8] = [1, 3, 5, 7, 9, 12, 15, 23];
const BAND_BOUNDARIES: [&[u8]; 8] = [
    &[0],
    &[0, 3, 14],
    &[0, 1, 3, 9, 23],
    &[0, 1, 2, 4, 8, 14, 23],
    &[0, 1, 2, 3, 5, 7, 9, 14, 23],
    &[0, 1, 2, 3, 4, 6, 8, 11, 14, 18, 23, 35],
    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 18, 23, 35],
    &[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 23, 26, 30, 35, 41, 48,
    ],
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DownmixConfiguration {
    FiveChannel,
    SevenChannel,
    FiveChannelHeight,
    FiveChannelPhaseShifted,
    FiveChannelHeightPhaseShifted,
}

impl DownmixConfiguration {
    fn read(bits: &mut BitReader<'_>) -> Result<Self, AppError> {
        match bits.read_u8(3)? {
            0 => Ok(Self::FiveChannel),
            1 => Ok(Self::SevenChannel),
            2 => Ok(Self::FiveChannelHeight),
            3 => Ok(Self::FiveChannelPhaseShifted),
            4 => Ok(Self::FiveChannelHeightPhaseShifted),
            value => Err(unsupported(format!(
                "reserved JOC downmix configuration {value}"
            ))),
        }
    }

    #[must_use]
    pub const fn channel_count(self) -> usize {
        match self {
            Self::FiveChannel | Self::FiveChannelPhaseShifted => 5,
            Self::SevenChannel | Self::FiveChannelHeight | Self::FiveChannelHeightPhaseShifted => 7,
        }
    }
}

/// One frame of interpolated JOC reconstruction coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct JocFrame {
    pub downmix: DownmixConfiguration,
    pub object_count: usize,
    pub input_channels: usize,
    pub timeslots: usize,
    pub clip_gain: f32,
    pub sequence: u16,
    pub splice: bool,
    /// Object-major, then timeslot, input channel, and QMF subband.
    coefficients: Vec<f32>,
}

impl JocFrame {
    #[must_use]
    pub fn coefficient(
        &self,
        object: usize,
        timeslot: usize,
        channel: usize,
        subband: usize,
    ) -> Option<f32> {
        let index = self.index(object, timeslot, channel, subband)?;
        self.coefficients.get(index).copied()
    }

    #[must_use]
    pub fn channel_coefficients(
        &self,
        object: usize,
        timeslot: usize,
        channel: usize,
    ) -> Option<&[f32]> {
        let start = self.index(object, timeslot, channel, 0)?;
        self.coefficients.get(start..start + QMF_SUBBANDS)
    }

    #[must_use]
    pub fn coefficients(&self) -> &[f32] {
        &self.coefficients
    }

    pub(crate) fn hold_last(&self, timeslots: usize) -> Self {
        let mut coefficients =
            vec![0.0; self.object_count * timeslots * self.input_channels * QMF_SUBBANDS];
        for object in 0..self.object_count {
            for timeslot in 0..timeslots {
                for channel in 0..self.input_channels {
                    let source_start = ((object * self.timeslots + self.timeslots - 1)
                        * self.input_channels
                        + channel)
                        * QMF_SUBBANDS;
                    let source = &self.coefficients[source_start..source_start + QMF_SUBBANDS];
                    let start = ((object * timeslots + timeslot) * self.input_channels + channel)
                        * QMF_SUBBANDS;
                    coefficients[start..start + QMF_SUBBANDS].copy_from_slice(source);
                }
            }
        }
        Self {
            downmix: self.downmix,
            object_count: self.object_count,
            input_channels: self.input_channels,
            timeslots,
            clip_gain: self.clip_gain,
            sequence: self.sequence,
            splice: false,
            coefficients,
        }
    }

    fn index(
        &self,
        object: usize,
        timeslot: usize,
        channel: usize,
        subband: usize,
    ) -> Option<usize> {
        if object >= self.object_count
            || timeslot >= self.timeslots
            || channel >= self.input_channels
            || subband >= QMF_SUBBANDS
        {
            return None;
        }
        Some(
            (((object * self.timeslots + timeslot) * self.input_channels + channel) * QMF_SUBBANDS)
                + subband,
        )
    }
}

/// Stateful decoder retaining the previous reconstruction matrix between
/// consecutive JOC frames.
#[derive(Clone, Debug, Default)]
pub struct JocDecoder {
    previous: Vec<f32>,
    previous_objects: usize,
    previous_channels: usize,
    last_sequence: Option<u16>,
}

impl JocDecoder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            previous: Vec::new(),
            previous_objects: 0,
            previous_channels: 0,
            last_sequence: None,
        }
    }

    /// Decodes and temporally interpolates one JOC EMDF payload.
    ///
    /// `frame_samples` is the number of PCM samples decoded from the matching
    /// independent E-AC-3 frame.
    ///
    /// # Errors
    ///
    /// Rejects malformed Huffman data, reserved configurations, invalid object
    /// or channel indexes, non-zero padding, and frame sizes that cannot map to
    /// 64-sample QMF timeslots.
    pub fn decode(
        &mut self,
        payload: &MetadataPayload,
        frame_samples: usize,
    ) -> Result<JocFrame, AppError> {
        if payload.id != JOC_PAYLOAD_ID {
            return Err(invalid(format!(
                "expected JOC payload ID {JOC_PAYLOAD_ID}, found {}",
                payload.id
            )));
        }
        if frame_samples == 0 || !frame_samples.is_multiple_of(QMF_SUBBANDS) {
            return Err(invalid(format!(
                "JOC frame has {frame_samples} PCM samples; expected a positive multiple of {QMF_SUBBANDS}"
            )));
        }
        let timeslots = frame_samples / QMF_SUBBANDS;
        if timeslots > 32 {
            return Err(invalid(format!(
                "JOC frame spans {timeslots} QMF timeslots; the syntax can address at most 32"
            )));
        }
        let mut bits = payload.bits();
        let downmix = DownmixConfiguration::read(&mut bits)?;
        let input_channels = downmix.channel_count();
        let object_count = bits.read_usize(6)? + 1;
        if object_count > MAX_OBJECTS {
            return Err(unsupported(format!(
                "JOC object count {object_count}; supported maximum is {MAX_OBJECTS}"
            )));
        }
        let extension = bits.read_u8(3)?;
        if extension != 0 {
            return Err(unsupported(format!(
                "reserved JOC extension configuration {extension}"
            )));
        }

        let clip_gain_power = i32::from(bits.read_u8(3)?);
        let clip_gain_fraction = f32::from(bits.read_u8(5)?) / 32.0;
        let clip_gain = 1.0 + clip_gain_fraction * 2.0_f32.powi(clip_gain_power - 4);
        let sequence = bits.read_u16(10)?;
        let mut objects = Vec::with_capacity(object_count);
        for _ in 0..object_count {
            objects.push(ObjectInfo::read(&mut bits)?);
        }
        for object in objects.iter_mut().flatten() {
            object.read_data(&mut bits, input_channels)?;
        }
        if bits.remaining() > 7 {
            return Err(invalid(format!(
                "JOC payload has {} trailing bits; at most 7 padding bits are allowed",
                bits.remaining()
            )));
        }
        if bits.read(bits.remaining())? != 0 {
            return Err(invalid("JOC payload has non-zero padding"));
        }

        let dimensions_changed = self.previous_objects != object_count
            || self.previous_channels != input_channels
            || self.previous.len() != object_count * input_channels * QMF_SUBBANDS;
        let sequence_discontinuity = sequence == 0
            || self
                .last_sequence
                .is_some_and(|last| sequence != if last == 1023 { 1 } else { last + 1 });
        let splice = dimensions_changed || sequence_discontinuity;
        if splice {
            self.previous = vec![0.0; object_count * input_channels * QMF_SUBBANDS];
            self.previous_objects = object_count;
            self.previous_channels = input_channels;
        }

        let mut coefficients =
            vec![0.0_f32; object_count * timeslots * input_channels * QMF_SUBBANDS];
        for (object_index, object) in objects.iter().enumerate() {
            let Some(object) = object else {
                continue;
            };
            let dequantized = object.differential_decode(input_channels);
            interpolate_object(
                object_index,
                object,
                input_channels,
                timeslots,
                &dequantized,
                &mut self.previous,
                &mut coefficients,
            );
        }
        self.last_sequence = Some(sequence);

        Ok(JocFrame {
            downmix,
            object_count,
            input_channels,
            timeslots,
            clip_gain,
            sequence,
            splice,
            coefficients,
        })
    }
}

#[derive(Clone, Debug)]
struct ObjectInfo {
    band_index: usize,
    bands: usize,
    quantization: Quantization,
    interpolation: Interpolation,
    data_points: usize,
    coded: CodedData,
}

impl ObjectInfo {
    fn read(bits: &mut BitReader<'_>) -> Result<Option<Self>, AppError> {
        if !bits.read_bit()? {
            return Ok(None);
        }
        let band_index = bits.read_usize(3)?;
        let bands = BAND_COUNTS[band_index];
        let sparse = bits.read_bit()?;
        let quantization = if bits.read_bit()? {
            Quantization::Fine
        } else {
            Quantization::Coarse
        };
        let steep = bits.read_bit()?;
        let data_points = bits.read_usize(1)? + 1;
        let mut offsets = [0_usize; 2];
        if steep {
            for offset in &mut offsets[..data_points] {
                *offset = bits.read_usize(5)? + 1;
            }
            if data_points == 2 && offsets[1] < offsets[0] {
                return Err(invalid(format!(
                    "JOC steep-transition offsets are not ordered: {} then {}",
                    offsets[0], offsets[1]
                )));
            }
        }
        Ok(Some(Self {
            band_index,
            bands,
            quantization,
            interpolation: if steep {
                Interpolation::Steep(offsets)
            } else {
                Interpolation::Smooth
            },
            data_points,
            coded: if sparse {
                CodedData::Sparse(Vec::new())
            } else {
                CodedData::Dense(Vec::new())
            },
        }))
    }

    fn read_data(
        &mut self,
        bits: &mut BitReader<'_>,
        input_channels: usize,
    ) -> Result<(), AppError> {
        self.coded = match &self.coded {
            CodedData::Sparse(_) => {
                let index_tree: &[[i16; 2]] = if input_channels == 5 {
                    &FIVE_CHANNEL_INDEX
                } else {
                    &SEVEN_CHANNEL_INDEX
                };
                let vector_tree = self.quantization.sparse_tree();
                let mut points = Vec::with_capacity(self.data_points);
                for _ in 0..self.data_points {
                    let first_channel = bits.read_usize(3)?;
                    if first_channel >= input_channels {
                        return Err(invalid(format!(
                            "sparse JOC channel {first_channel} exceeds the {input_channels}-channel downmix"
                        )));
                    }
                    let mut channels = Vec::with_capacity(self.bands);
                    channels.push(first_channel);
                    for _ in 1..self.bands {
                        channels.push(usize::from(huffman_decode(bits, index_tree)?));
                    }
                    let mut vector = Vec::with_capacity(self.bands);
                    for _ in 0..self.bands {
                        vector.push(huffman_decode(bits, vector_tree)?);
                    }
                    points.push(SparsePoint { channels, vector });
                }
                CodedData::Sparse(points)
            }
            CodedData::Dense(_) => {
                let matrix_tree = self.quantization.generic_tree();
                let mut points = Vec::with_capacity(self.data_points);
                for _ in 0..self.data_points {
                    let mut matrix = Vec::with_capacity(input_channels * self.bands);
                    for _ in 0..input_channels * self.bands {
                        matrix.push(huffman_decode(bits, matrix_tree)?);
                    }
                    points.push(matrix);
                }
                CodedData::Dense(points)
            }
        };
        Ok(())
    }

    fn differential_decode(&self, input_channels: usize) -> Vec<f32> {
        let nquant = self.quantization.steps();
        let quantization_scale = self.quantization.scale();
        let mut result = vec![0.0; self.data_points * input_channels * self.bands];
        match &self.coded {
            CodedData::Sparse(points) => {
                for (point_index, point) in points.iter().enumerate() {
                    let point_start = point_index * input_channels * self.bands;
                    let offset = self.quantization.sparse_offset();
                    let mut quantized = vec![offset; input_channels * self.bands];
                    let mut selected_channel = point.channels[0];
                    for band in 0..self.bands {
                        if band != 0 {
                            selected_channel =
                                (point.channels[band - 1] + point.channels[band]) % input_channels;
                        }
                        let selected_index = selected_channel * self.bands + band;
                        quantized[selected_index] = if band == 0 {
                            (offset + point.vector[band]) % nquant
                        } else {
                            (quantized[selected_index - 1] + point.vector[band]) % nquant
                        };
                    }
                    for (index, value) in quantized.into_iter().enumerate() {
                        result[point_start + index] =
                            (f32::from(value) - f32::from(nquant) / 2.0) * quantization_scale;
                    }
                }
            }
            CodedData::Dense(points) => {
                for (point_index, matrix) in points.iter().enumerate() {
                    let point_start = point_index * input_channels * self.bands;
                    let offset = nquant / 2;
                    for channel in 0..input_channels {
                        let mut quantized = 0_u16;
                        for band in 0..self.bands {
                            let coded = matrix[channel * self.bands + band];
                            quantized = if band == 0 {
                                (offset + coded) % nquant
                            } else {
                                (quantized + coded) % nquant
                            };
                            result[point_start + channel * self.bands + band] =
                                (f32::from(quantized) - f32::from(nquant) / 2.0)
                                    * quantization_scale;
                        }
                    }
                }
            }
        }
        result
    }
}

#[derive(Clone, Copy, Debug)]
enum Quantization {
    Coarse,
    Fine,
}

impl Quantization {
    const fn steps(self) -> u16 {
        match self {
            Self::Coarse => 96,
            Self::Fine => 192,
        }
    }

    const fn sparse_offset(self) -> u16 {
        match self {
            Self::Coarse => 50,
            Self::Fine => 100,
        }
    }

    const fn scale(self) -> f32 {
        match self {
            Self::Coarse => 820.0 / 4096.0,
            Self::Fine => 820.0 / 8192.0,
        }
    }

    const fn generic_tree(self) -> &'static [[i16; 2]] {
        match self {
            Self::Coarse => &COARSE_GENERIC,
            Self::Fine => &FINE_GENERIC,
        }
    }

    const fn sparse_tree(self) -> &'static [[i16; 2]] {
        match self {
            Self::Coarse => &COARSE_SPARSE,
            Self::Fine => &FINE_SPARSE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Interpolation {
    Smooth,
    Steep([usize; 2]),
}

#[derive(Clone, Debug)]
enum CodedData {
    Sparse(Vec<SparsePoint>),
    Dense(Vec<Vec<u16>>),
}

#[derive(Clone, Debug)]
struct SparsePoint {
    channels: Vec<usize>,
    vector: Vec<u16>,
}

#[allow(clippy::too_many_arguments)]
fn interpolate_object(
    object_index: usize,
    object: &ObjectInfo,
    channels: usize,
    timeslots: usize,
    dequantized: &[f32],
    previous: &mut [f32],
    output: &mut [f32],
) {
    let previous_start = object_index * channels * QMF_SUBBANDS;
    let timeslots_u8 = u8::try_from(timeslots).expect("JOC timeslots were validated");
    for channel in 0..channels {
        for subband in 0..QMF_SUBBANDS {
            let parameter_band = subband_to_parameter_band(object.band_index, subband);
            let previous_value = previous[previous_start + channel * QMF_SUBBANDS + subband];
            let current = |point: usize| {
                dequantized[(point * channels + channel) * object.bands + parameter_band]
            };
            for timeslot_u8 in 0..timeslots_u8 {
                let timeslot = usize::from(timeslot_u8);
                let value = if let Interpolation::Steep(offsets) = object.interpolation {
                    if object.data_points == 1 {
                        if timeslot < offsets[0] {
                            previous_value
                        } else {
                            current(0)
                        }
                    } else if timeslot < offsets[0] {
                        previous_value
                    } else if timeslot < offsets[1] {
                        current(0)
                    } else {
                        current(1)
                    }
                } else if object.data_points == 1 {
                    let fraction = f32::from(timeslot_u8 + 1) / f32::from(timeslots_u8);
                    previous_value + (current(0) - previous_value) * fraction
                } else {
                    let midpoint = timeslots / 2;
                    if timeslot < midpoint {
                        let fraction = f32::from(timeslot_u8 + 1) / f32::from(timeslots_u8 / 2);
                        previous_value + (current(0) - previous_value) * fraction
                    } else {
                        let fraction = f32::from(timeslot_u8 - timeslots_u8 / 2 + 1)
                            / f32::from(timeslots_u8 - timeslots_u8 / 2);
                        current(0) + (current(1) - current(0)) * fraction
                    }
                };
                let output_index = (((object_index * timeslots + timeslot) * channels + channel)
                    * QMF_SUBBANDS)
                    + subband;
                output[output_index] = value;
            }
            previous[previous_start + channel * QMF_SUBBANDS + subband] =
                current(object.data_points - 1);
        }
    }
}

fn subband_to_parameter_band(band_index: usize, subband: usize) -> usize {
    BAND_BOUNDARIES[band_index].partition_point(|boundary| usize::from(*boundary) <= subband) - 1
}

fn huffman_decode(bits: &mut BitReader<'_>, tree: &[[i16; 2]]) -> Result<u16, AppError> {
    let mut node = 0_i16;
    for _ in 0..=tree.len() {
        let row = usize::try_from(node)
            .ok()
            .and_then(|node| tree.get(node))
            .ok_or_else(|| invalid("JOC Huffman tree selected an invalid node"))?;
        node = row[usize::from(bits.read_bit()?)];
        if node <= 0 {
            return u16::try_from(-i32::from(node) - 1)
                .map_err(|error| invalid(format!("JOC Huffman leaf overflowed: {error}")));
        }
    }
    Err(invalid("JOC Huffman code did not terminate"))
}

fn invalid(message: impl Into<String>) -> AppError {
    AppError::Render(format!("invalid JOC metadata: {}", message.into()))
}

fn unsupported(message: impl Into<String>) -> AppError {
    AppError::UnsupportedInput(format!("unsupported JOC feature: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::{
        BAND_COUNTS, COARSE_GENERIC, FINE_SPARSE, JocDecoder, QMF_SUBBANDS, huffman_decode,
        subband_to_parameter_band,
    };
    use crate::eac3::{BitReader, MetadataPayload};

    #[test]
    fn parameter_band_mapping_matches_standard_example() {
        let band_index = BAND_COUNTS.iter().position(|count| *count == 15).unwrap();
        assert_eq!(subband_to_parameter_band(band_index, 24), 13);
        assert_eq!(subband_to_parameter_band(0, 63), 0);
        assert_eq!(subband_to_parameter_band(7, 48), 22);
    }

    #[test]
    fn normative_huffman_trees_decode_root_leaves() {
        let mut zero = BitReader::new(&[0]);
        assert_eq!(huffman_decode(&mut zero, &COARSE_GENERIC).unwrap(), 0);
        let mut one = BitReader::new(&[0b1000_0000]);
        assert_eq!(huffman_decode(&mut one, &FINE_SPARSE).unwrap(), 0);
    }

    #[test]
    fn fresh_decoder_has_no_history() {
        let decoder = JocDecoder::new();
        assert!(decoder.previous.is_empty());
        assert_eq!(QMF_SUBBANDS, 64);
    }

    #[test]
    fn short_arbitrary_payloads_fail_without_panicking() {
        for length in 0_usize..64 {
            let data = (0..length)
                .map(|index| {
                    u8::try_from(index)
                        .unwrap()
                        .wrapping_mul(73)
                        .wrapping_add(19)
                })
                .collect::<Vec<_>>();
            let payload = MetadataPayload {
                id: 14,
                sample_offset: None,
                bit_len: data.len() * 8,
                data,
            };
            let _ = JocDecoder::new().decode(&payload, 1536);
        }
    }
}
