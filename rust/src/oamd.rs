//! Object Audio Metadata decoding for ETSI TS 103 420.

use crate::{
    eac3::{BitReader, MetadataPayload},
    error::AppError,
    hrir::Speaker,
    isf::IsfConfig,
    object::{IsfState, ObjectState, SpatialUpdate},
};

const OAMD_PAYLOAD_ID: u32 = 11;
const MAX_OBJECTS: usize = 64;
const SAMPLE_OFFSET_TABLE: [usize; 4] = [8, 16, 18, 24];
const RAMP_DURATION_TABLE: [usize; 16] = [
    32, 64, 128, 256, 320, 480, 1000, 1001, 1024, 1600, 1601, 1602, 1920, 2000, 2002, 2048,
];
const DISTANCE_FACTORS: [f32; 16] = [
    1.1, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 7.9, 10.0, 12.6, 15.8, 20.0, 25.1, 50.1,
];
const DEPTH_FACTORS: [f32; 4] = [0.25, 0.5, 1.0, 2.0];

#[derive(Clone, Debug, PartialEq)]
pub struct OamdFrame {
    pub object_count: usize,
    /// Number of object essences produced by JOC (all objects except LFE).
    pub joc_object_count: usize,
    pub dynamic_object_count: usize,
    /// Bed labels in the same order as the first objects in the OAMD program.
    pub bed_speakers: Vec<Speaker>,
    /// Full OAMD object indexes that bypass JOC and use the decoded LFE.
    pub lfe_object_indices: Vec<usize>,
    pub isf: Option<IsfConfig>,
    pub updates: Vec<SpatialUpdate>,
}

#[derive(Clone, Debug, Default)]
pub struct OamdDecoder {
    properties: Vec<ObjectProperties>,
}

impl OamdDecoder {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            properties: Vec::new(),
        }
    }

    /// Decodes one OAMD EMDF payload while retaining reuse state across frames.
    ///
    /// # Errors
    ///
    /// Rejects malformed or truncated metadata, reserved syntax, inconsistent
    /// object counts, and impossible element sizes.
    #[allow(clippy::too_many_lines)] // Mirrors the ordered top-level OAMD payload syntax.
    pub fn decode(&mut self, payload: &MetadataPayload) -> Result<OamdFrame, AppError> {
        if payload.id != OAMD_PAYLOAD_ID {
            return Err(invalid(format!(
                "expected OAMD payload ID {OAMD_PAYLOAD_ID}, found {}",
                payload.id
            )));
        }
        let mut bits = payload.bits();
        let mut version = bits.read_usize(2)?;
        if version == 3 {
            version += bits.read_usize(3)?;
        }
        if version != 0 {
            return Err(unsupported(format!("OAMD version {version}")));
        }

        let mut object_count = bits.read_usize(5)? + 1;
        if object_count == 32 {
            object_count += bits.read_usize(7)?;
        }
        if object_count == 0 || object_count > MAX_OBJECTS {
            return Err(unsupported(format!(
                "OAMD object count {object_count}; supported range is 1..={MAX_OBJECTS}"
            )));
        }
        let assignment = ProgramAssignment::read(&mut bits, object_count)?;
        let fixed_object_count = assignment
            .bed_speakers
            .len()
            .checked_add(assignment.isf_object_count)
            .ok_or_else(|| invalid("OAMD fixed-object count overflowed"))?;
        if fixed_object_count > object_count {
            return Err(invalid(
                "bed and ISF assignment exceeds the OAMD object count",
            ));
        }
        let lfe_object_indices = assignment
            .bed_speakers
            .iter()
            .enumerate()
            .filter_map(|(index, speaker)| (*speaker == Speaker::Lfe).then_some(index))
            .collect::<Vec<_>>();
        if lfe_object_indices.len() > 1 {
            return Err(unsupported(
                "OAMD programs with two independent LFE objects",
            ));
        }
        let joc_object_count = object_count - lfe_object_indices.len();
        let dynamic_object_count = object_count - fixed_object_count;
        let isf_start = assignment.bed_speakers.len() - lfe_object_indices.len();
        let isf = (assignment.isf_object_count != 0)
            .then(|| IsfConfig::new(isf_start, assignment.isf_object_count))
            .transpose()?;

        if self.properties.len() != object_count {
            self.properties = vec![ObjectProperties::default(); object_count];
        }

        let alternate_data = bits.read_bit()?;
        let mut element_count = bits.read_usize(4)?;
        if element_count == 15 {
            element_count += bits.read_usize(5)?;
        }
        let mut updates = Vec::new();
        let mut decoded_object_element = false;
        for _ in 0..element_count {
            let element_id = bits.read_u8(4)?;
            let element_bytes = usize::try_from(bits.variable(4, Some(3))?)
                .map_err(|error| invalid(format!("OAMD element size overflowed: {error}")))?
                .checked_add(1)
                .ok_or_else(|| invalid("OAMD element size overflowed"))?;
            let element_end = bits
                .position()
                .checked_add(
                    element_bytes
                        .checked_mul(8)
                        .ok_or_else(|| invalid("OAMD element size overflowed"))?,
                )
                .ok_or_else(|| invalid("OAMD element endpoint overflowed"))?;
            if element_end > bits.limit() {
                return Err(invalid("OAMD element extends past its payload"));
            }
            let outer_limit = bits.limit();
            bits.set_limit(element_end)?;

            if alternate_data {
                let alternate_id = bits.read_u8(4)?;
                if element_id == 1 && alternate_id != 0 {
                    return Err(unsupported(format!(
                        "OAMD alternate object-data type {alternate_id}"
                    )));
                }
            }
            let _discard_unknown = bits.read_bit()?;
            if element_id == 1 {
                if decoded_object_element {
                    return Err(unsupported("multiple OAMD object elements in one payload"));
                }
                updates = self.read_object_element(
                    &mut bits,
                    payload.sample_offset.map_or(0, usize::from),
                    &assignment.bed_speakers,
                    &lfe_object_indices,
                    assignment.isf_object_count,
                    object_count,
                )?;
                decoded_object_element = true;
            }
            bits.set_position(element_end)?;
            bits.set_limit(outer_limit)?;
        }

        Ok(OamdFrame {
            object_count,
            joc_object_count,
            dynamic_object_count,
            bed_speakers: assignment.bed_speakers,
            lfe_object_indices,
            isf,
            updates,
        })
    }

    #[allow(clippy::too_many_lines)] // Mirrors one object-element syntax pass.
    fn read_object_element(
        &mut self,
        bits: &mut BitReader<'_>,
        emdf_sample_offset: usize,
        bed_speakers: &[Speaker],
        lfe_object_indices: &[usize],
        isf_object_count: usize,
        object_count: usize,
    ) -> Result<Vec<SpatialUpdate>, AppError> {
        let sample_offset = match bits.read_u8(2)? {
            0 => 0,
            1 => SAMPLE_OFFSET_TABLE[bits.read_usize(2)?],
            2 => bits.read_usize(5)?,
            _ => return Err(unsupported("reserved OAMD sample-offset code")),
        };
        let block_count = bits.read_usize(3)? + 1;
        let mut timing = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let block_offset = bits.read_usize(6)?;
            let ramp_samples = match bits.read_u8(2)? {
                0 => 0,
                1 => 512,
                2 => 1536,
                _ if bits.read_bit()? => RAMP_DURATION_TABLE[bits.read_usize(4)?],
                _ => bits.read_usize(11)?,
            };
            timing.push(BlockTiming {
                sample_offset: emdf_sample_offset
                    .checked_add(sample_offset)
                    .and_then(|offset| offset.checked_add(block_offset * 32))
                    .ok_or_else(|| invalid("OAMD sample offset overflowed"))?,
                ramp_samples,
            });
        }

        let default_screen_ratio = bits.read_bit()?;
        let _reference_screen_ratio = if default_screen_ratio {
            1.0
        } else {
            (f32::from(bits.read_u8(5)?) + 1.0) / 33.0
        };

        let mut states = vec![Vec::with_capacity(object_count); block_count];
        let mut isf_states = vec![Vec::with_capacity(isf_object_count); block_count];
        let mut previous_gain_in_block = vec![1.0_f32; block_count];
        for object_index in 0..object_count {
            let in_bed = object_index < bed_speakers.len();
            let in_isf = object_index >= bed_speakers.len()
                && object_index < bed_speakers.len() + isf_object_count;
            for block_index in 0..block_count {
                let inactive = bits.read_bit()?;
                let basic_status = if inactive {
                    0
                } else if block_index == 0 {
                    1
                } else {
                    bits.read_u8(2)?
                };
                read_basic_info(
                    bits,
                    &mut self.properties[object_index],
                    basic_status,
                    previous_gain_in_block[block_index],
                )
                .map_err(|error| {
                    invalid(format!(
                        "object {object_index}, block {block_index}, basic info: {error}"
                    ))
                })?;
                previous_gain_in_block[block_index] = self.properties[object_index].gain;

                let render_status = if inactive || in_bed || in_isf {
                    0
                } else if block_index == 0 {
                    1
                } else {
                    bits.read_u8(2)?
                };
                read_render_info(
                    bits,
                    &mut self.properties[object_index],
                    render_status,
                    block_index,
                )
                .map_err(|error| {
                    invalid(format!(
                        "object {object_index}, block {block_index}, render info: {error}"
                    ))
                })?;
                if bits.read_bit()? {
                    let additional_bytes = bits.read_usize(4)? + 1;
                    bits.skip(additional_bytes * 8)?;
                }

                let properties = &self.properties[object_index];
                let position = if let Some(speaker) = bed_speakers.get(object_index) {
                    speaker.position()
                } else {
                    room_position(properties.position)
                };
                if !lfe_object_indices.contains(&object_index) {
                    let source_channel = object_index
                        - lfe_object_indices.partition_point(|lfe_index| *lfe_index < object_index);
                    if in_isf {
                        isf_states[block_index].push(IsfState {
                            source_channel,
                            active: !inactive,
                            gain: properties.gain,
                        });
                    } else {
                        states[block_index].push(ObjectState {
                            source_channel,
                            active: !inactive,
                            bed: in_bed,
                            position,
                            gain: properties.gain,
                            size: properties.diffusion(),
                        });
                    }
                }
            }
        }

        Ok(timing
            .into_iter()
            .zip(states)
            .zip(isf_states)
            .map(|((timing, objects), isf)| SpatialUpdate {
                sample_offset: timing.sample_offset,
                ramp_samples: timing.ramp_samples,
                bed_speakers: bed_speakers.to_vec(),
                isf,
                objects,
            })
            .collect())
    }
}

#[derive(Clone, Debug, Default)]
struct ProgramAssignment {
    bed_speakers: Vec<Speaker>,
    isf_object_count: usize,
}

impl ProgramAssignment {
    fn read(bits: &mut BitReader<'_>, object_count: usize) -> Result<Self, AppError> {
        if bits.read_bit()? {
            let bed_speakers = bits
                .read_bit()?
                .then_some(Speaker::Lfe)
                .into_iter()
                .collect();
            return Ok(Self {
                bed_speakers,
                isf_object_count: 0,
            });
        }

        let content_mask = bits.read_u8(4)?;
        let mut bed_speakers = Vec::new();
        if content_mask & 1 != 0 {
            let _distributable = bits.read_bit()?;
            let bed_instances = if bits.read_bit()? {
                bits.read_usize(3)? + 2
            } else {
                1
            };
            for _ in 0..bed_instances {
                let lfe_only = bits.read_bit()?;
                if lfe_only {
                    bed_speakers.push(Speaker::Lfe);
                } else {
                    let standard_assignment = bits.read_bit()?;
                    if standard_assignment {
                        append_standard_bed(bits.read_u16(10)?, &mut bed_speakers);
                    } else {
                        append_nonstandard_bed(bits.read_u32(17)?, &mut bed_speakers);
                    }
                }
            }
        }
        let isf_object_count = if content_mask & 2 != 0 {
            let index = bits.read_usize(3)?;
            *[4, 8, 10, 14, 15, 30]
                .get(index)
                .ok_or_else(|| unsupported(format!("reserved OAMD ISF index {index}")))?
        } else {
            0
        };
        if content_mask & 4 != 0 {
            let mut dynamic_objects = bits.read_usize(5)?;
            if dynamic_objects == 31 {
                dynamic_objects += bits.read_usize(7)?;
            }
            dynamic_objects += 1;
            if bed_speakers
                .len()
                .saturating_add(isf_object_count)
                .saturating_add(dynamic_objects)
                != object_count
            {
                return Err(invalid(format!(
                    "OAMD program assignment describes {} objects, header describes {object_count}",
                    bed_speakers.len() + isf_object_count + dynamic_objects
                )));
            }
        }
        if content_mask & 8 != 0 {
            let reserved_bytes = bits.read_usize(4)? + 1;
            bits.skip(reserved_bytes * 8)?;
        }
        Ok(Self {
            bed_speakers,
            isf_object_count,
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct BlockTiming {
    sample_offset: usize,
    ramp_samples: usize,
}

#[derive(Clone, Debug)]
struct ObjectProperties {
    position: [f32; 3],
    size: [f32; 3],
    gain: f32,
    screen_anchored: bool,
    screen_factor: f32,
    depth_factor: f32,
    distance_factor: Option<f32>,
}

impl Default for ObjectProperties {
    fn default() -> Self {
        Self {
            position: [0.5, 0.5, 0.0],
            size: [0.0; 3],
            gain: 0.0,
            screen_anchored: false,
            screen_factor: 0.0,
            depth_factor: 1.0,
            distance_factor: None,
        }
    }
}

impl ObjectProperties {
    fn reset_basic(&mut self) {
        self.gain = 0.0;
    }

    fn reset_render(&mut self) {
        self.position = [0.5, 0.5, 0.0];
        self.size = [0.0; 3];
        self.screen_anchored = false;
        self.screen_factor = 0.0;
        self.depth_factor = 1.0;
        self.distance_factor = None;
    }

    fn diffusion(&self) -> f32 {
        (self.size.iter().map(|value| value * value).sum::<f32>() / 3.0)
            .sqrt()
            .clamp(0.0, 1.0)
    }
}

fn read_basic_info(
    bits: &mut BitReader<'_>,
    properties: &mut ObjectProperties,
    status: u8,
    previous_object_gain: f32,
) -> Result<(), AppError> {
    match status {
        0 => properties.reset_basic(),
        1 | 3 => {
            let mask = if status == 1 { 3 } else { bits.read_u8(2)? };
            if mask & 1 != 0 {
                properties.gain = match bits.read_u8(2)? {
                    0 => 1.0,
                    1 => 0.0,
                    2 => {
                        let code = bits.read_u8(6)?;
                        let gain_db = if code < 15 {
                            15 - i16::from(code)
                        } else {
                            14 - i16::from(code)
                        };
                        10_f32.powf(f32::from(gain_db) / 20.0)
                    }
                    _ => previous_object_gain,
                };
            }
            if mask & 2 != 0 && !bits.read_bit()? {
                let _priority = bits.read_u8(5)?;
            }
        }
        2 => {}
        _ => return Err(invalid("invalid OAMD basic-info status")),
    }
    Ok(())
}

fn read_render_info(
    bits: &mut BitReader<'_>,
    properties: &mut ObjectProperties,
    status: u8,
    block_index: usize,
) -> Result<(), AppError> {
    match status {
        0 => {
            properties.reset_render();
            return Ok(());
        }
        2 => return Ok(()),
        1 | 3 => {}
        _ => return Err(invalid("invalid OAMD render-info status")),
    }
    let mask = if status == 1 { 15 } else { bits.read_u8(4)? };
    if mask & 1 != 0 {
        let differential = block_index != 0 && bits.read_bit()?;
        if differential {
            let delta_x = i16::try_from(bits.read_signed(3)?)
                .map_err(|error| invalid(format!("OAMD X delta overflowed: {error}")))?;
            let delta_y = i16::try_from(bits.read_signed(3)?)
                .map_err(|error| invalid(format!("OAMD Y delta overflowed: {error}")))?;
            let delta_z = i16::try_from(bits.read_signed(3)?)
                .map_err(|error| invalid(format!("OAMD Z delta overflowed: {error}")))?;
            properties.position[0] =
                (properties.position[0] + f32::from(delta_x) / 62.0).clamp(0.0, 1.0);
            properties.position[1] =
                (properties.position[1] + f32::from(delta_y) / 62.0).clamp(0.0, 1.0);
            properties.position[2] =
                (properties.position[2] + f32::from(delta_z) / 15.0).clamp(-1.0, 1.0);
        } else {
            properties.position[0] = (f32::from(bits.read_u8(6)?) / 62.0).min(1.0);
            properties.position[1] = (f32::from(bits.read_u8(6)?) / 62.0).min(1.0);
            let sign = if bits.read_bit()? { 1.0 } else { -1.0 };
            properties.position[2] = sign * f32::from(bits.read_u8(4)?) / 15.0;
        }
        properties.distance_factor = if bits.read_bit()? {
            Some(if bits.read_bit()? {
                f32::INFINITY
            } else {
                DISTANCE_FACTORS[bits.read_usize(4)?]
            })
        } else {
            None
        };
    }
    if mask & 2 != 0 {
        let _zone = bits.read_u8(3)?;
        let _elevation_enabled = bits.read_bit()?;
    }
    if mask & 4 != 0 {
        properties.size = match bits.read_u8(2)? {
            0 | 3 => [0.0; 3],
            1 => {
                let size = f32::from(bits.read_u8(5)?) / 31.0;
                [size; 3]
            }
            2 => [
                f32::from(bits.read_u8(5)?) / 31.0,
                f32::from(bits.read_u8(5)?) / 31.0,
                f32::from(bits.read_u8(5)?) / 31.0,
            ],
            _ => unreachable!("two-bit size index is exhaustive"),
        };
    }
    if mask & 8 != 0 {
        properties.screen_anchored = bits.read_bit()?;
        if properties.screen_anchored {
            properties.screen_factor = (f32::from(bits.read_u8(3)?) + 1.0) / 8.0;
            properties.depth_factor = DEPTH_FACTORS[bits.read_usize(2)?];
        } else {
            properties.screen_factor = 0.0;
            properties.depth_factor = 1.0;
        }
    }
    let _channel_lock = bits.read_bit()?;
    Ok(())
}

fn append_standard_bed(mask: u16, result: &mut Vec<Speaker>) {
    const LABELS: [&[Speaker]; 10] = [
        &[Speaker::FrontLeft, Speaker::FrontRight],
        &[Speaker::FrontCenter],
        &[Speaker::Lfe],
        &[Speaker::SideLeft, Speaker::SideRight],
        &[Speaker::RearLeft, Speaker::RearRight],
        &[Speaker::TopFrontLeft, Speaker::TopFrontRight],
        &[Speaker::TopSideLeft, Speaker::TopSideRight],
        &[Speaker::TopRearLeft, Speaker::TopRearRight],
        &[Speaker::WideLeft, Speaker::WideRight],
        &[Speaker::Lfe],
    ];
    for (bit, labels) in LABELS.iter().enumerate() {
        if mask & (1 << bit) != 0 {
            result.extend_from_slice(labels);
        }
    }
}

fn append_nonstandard_bed(mask: u32, result: &mut Vec<Speaker>) {
    const LABELS: [Speaker; 17] = [
        Speaker::FrontLeft,
        Speaker::FrontRight,
        Speaker::FrontCenter,
        Speaker::Lfe,
        Speaker::SideLeft,
        Speaker::SideRight,
        Speaker::RearLeft,
        Speaker::RearRight,
        Speaker::TopFrontLeft,
        Speaker::TopFrontRight,
        Speaker::TopSideLeft,
        Speaker::TopSideRight,
        Speaker::TopRearLeft,
        Speaker::TopRearRight,
        Speaker::WideLeft,
        Speaker::WideRight,
        Speaker::Lfe,
    ];
    for (bit, speaker) in LABELS.into_iter().enumerate() {
        if mask & (1 << bit) != 0 {
            result.push(speaker);
        }
    }
}

fn room_position(position: [f32; 3]) -> [f32; 3] {
    [
        position[0].mul_add(2.0, -1.0),
        (-position[1]).mul_add(2.0, 1.0),
        position[2],
    ]
}

fn invalid(message: impl Into<String>) -> AppError {
    AppError::Render(format!("invalid OAMD metadata: {}", message.into()))
}

fn unsupported(message: impl Into<String>) -> AppError {
    AppError::UnsupportedInput(format!("unsupported {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::{OamdDecoder, append_standard_bed, room_position};
    use crate::{eac3::MetadataPayload, hrir::Speaker};

    #[test]
    fn standard_bed_mask_uses_lsb_label_order() {
        let mut speakers = Vec::new();
        append_standard_bed(0b1_0000_1101, &mut speakers);
        assert_eq!(
            speakers,
            [
                Speaker::FrontLeft,
                Speaker::FrontRight,
                Speaker::Lfe,
                Speaker::SideLeft,
                Speaker::SideRight,
                Speaker::WideLeft,
                Speaker::WideRight,
            ]
        );
    }

    #[test]
    #[allow(clippy::float_cmp)] // Boundary values are exactly representable.
    fn room_coordinates_map_to_renderer_axes() {
        assert_eq!(room_position([0.0, 0.0, -1.0]), [-1.0, 1.0, -1.0]);
        assert_eq!(room_position([1.0, 1.0, 1.0]), [1.0, -1.0, 1.0]);
    }

    #[test]
    fn rejects_non_oamd_payloads() {
        let payload = MetadataPayload {
            id: 14,
            sample_offset: None,
            data: Vec::new(),
            bit_len: 0,
        };
        assert!(OamdDecoder::new().decode(&payload).is_err());
    }

    #[test]
    fn short_arbitrary_payloads_fail_without_panicking() {
        for length in 0_usize..64 {
            let data = (0..length)
                .map(|index| {
                    u8::try_from(index)
                        .unwrap()
                        .wrapping_mul(41)
                        .wrapping_add(7)
                })
                .collect::<Vec<_>>();
            let payload = MetadataPayload {
                id: 11,
                sample_offset: None,
                bit_len: data.len() * 8,
                data,
            };
            let _ = OamdDecoder::new().decode(&payload);
        }
    }
}
