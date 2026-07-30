//! Object Audio Metadata decoding for ETSI TS 103 420.

use crate::{
    eac3::{BitReader, MetadataPayload},
    error::AppError,
    hrir::Speaker,
    isf::IsfConfig,
    object::{
        IsfState, ObjectState, ObjectTrim, ObjectTrimMode, ObjectTrimSettings, ObjectZone,
        SpatialUpdate, interpolate_screen_geometry, project_room_distance,
    },
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
const DIVERGENCE_TABLE: [f32; 4] = [0.500_755, 0.608_529, 0.704_833, 1.0];
#[rustfmt::skip]
const DIVERGENCE_CODES: [Option<f32>; 64] = [
    None, Some(0.0), Some(0.004_026), Some(0.007_16),
    Some(0.012_731), Some(0.020_173), Some(0.028_485), Some(0.040_21),
    Some(0.050_582), Some(0.063_601), Some(0.079_914), Some(0.100_299),
    Some(0.125_666), Some(0.140_532), Some(0.157_027), Some(0.175_282),
    Some(0.195_417), Some(0.217_536), Some(0.241_718), Some(0.268_002),
    Some(0.296_377), Some(0.326_766), Some(0.359_017), Some(0.392_895),
    Some(0.428_081), Some(0.464_184), Some(0.500_755), Some(0.537_316),
    Some(0.573_389), Some(0.608_529), Some(0.642_346), Some(0.674_524),
    Some(0.704_833), Some(0.733_123), Some(0.759_32), Some(0.783_416),
    Some(0.805_451), Some(0.825_506), Some(0.843_686), Some(0.860_112),
    Some(0.874_914), Some(0.888_222), Some(0.900_168), Some(0.910_875),
    Some(0.920_461), Some(0.929_035), Some(0.936_698), Some(0.943_544),
    Some(0.949_656), Some(0.955_112), Some(0.959_98), Some(0.964_322),
    Some(0.968_195), Some(0.974_729), Some(0.979_923), Some(0.984_05),
    Some(0.987_33), Some(0.989_935), Some(0.992_874), Some(0.994_955),
    Some(0.996_817), Some(0.998_21), Some(0.998_993), Some(1.0),
];
const EXTENDED_POSITION_CODES: [f32; 4] = [1.0, 2.0, -1.0, -2.0];
const TRIM_LEVELS_DB: [f32; 16] = [
    6.0, 3.0, 1.5, 0.75, -0.75, -1.5, -3.0, -4.5, -6.0, -7.5, -9.0, -10.5, -12.0, -13.5, -16.0,
    -36.0,
];

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
                "OAMD programs with two LFE labels but only one bypass LFE essence",
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
        let mut object_blocks = None;
        let mut decoded_extended_element = false;
        let mut trim = None;
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
                let decoded = self.read_object_element(
                    &mut bits,
                    payload.sample_offset.map_or(0, usize::from),
                    &assignment.bed_speakers,
                    &lfe_object_indices,
                    assignment.isf_object_count,
                    object_count,
                )?;
                updates = decoded.updates;
                object_blocks = Some(decoded.blocks);
                decoded_object_element = true;
            } else if element_id == 2 {
                if trim.is_some() {
                    return Err(unsupported("multiple OAMD trim elements in one payload"));
                }
                trim = Some(read_trim_element(&mut bits, object_count)?);
            } else if element_id == 5 {
                if decoded_extended_element {
                    return Err(unsupported(
                        "multiple OAMD extended-object elements in one payload",
                    ));
                }
                let blocks = object_blocks.as_ref().ok_or_else(|| {
                    unsupported("OAMD extended-object element precedes its object element")
                })?;
                self.apply_extended_object_element(&mut bits, blocks, &mut updates)?;
                decoded_extended_element = true;
            }
            bits.set_position(element_end)?;
            bits.set_limit(outer_limit)?;
        }
        if let (Some(trim), Some(blocks)) = (&trim, &object_blocks) {
            apply_trim_element(trim, blocks, &mut updates)?;
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
    ) -> Result<DecodedObjectElement, AppError> {
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

        let reserved_data_not_present = bits.read_bit()?;
        if !reserved_data_not_present {
            bits.skip(5)?;
        }

        let mut states = vec![Vec::with_capacity(object_count); block_count];
        let mut isf_states = vec![Vec::with_capacity(isf_object_count); block_count];
        let mut blocks = vec![vec![ObjectBlockContext::default(); block_count]; object_count];
        let mut previous_gain_in_block = vec![1.0_f32; block_count];
        for (object_index, object_blocks) in blocks.iter_mut().enumerate() {
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
                let source_channel = if lfe_object_indices.contains(&object_index) {
                    Some(object_count - lfe_object_indices.len())
                } else {
                    Some(
                        object_index
                            - lfe_object_indices
                                .partition_point(|lfe_index| *lfe_index < object_index),
                    )
                };
                object_blocks[block_index] = ObjectBlockContext {
                    active: !inactive,
                    dynamic: !(in_bed || in_isf),
                    source_channel,
                    room_position: properties.position,
                    screen_anchored: properties.screen_anchored,
                    screen_factor: properties.screen_factor,
                    depth_factor: properties.depth_factor,
                    distance_factor: properties.distance_factor,
                };
                let (position, size) = if let Some(speaker) = bed_speakers.get(object_index) {
                    (speaker.position(), properties.size)
                } else {
                    let room_position = room_position(properties.position);
                    let (room_position, size) = if properties.screen_anchored {
                        interpolate_screen_geometry(
                            room_position,
                            properties.size,
                            properties.screen_factor,
                            properties.depth_factor,
                        )
                    } else {
                        (room_position, properties.size)
                    };
                    (
                        project_room_distance(room_position, properties.distance_factor),
                        size,
                    )
                };
                if let Some(source_channel) = source_channel {
                    if in_isf {
                        isf_states[block_index].push(IsfState {
                            source_channel,
                            active: !inactive,
                            gain: properties.gain,
                            trim: ObjectTrim::default_algorithm(),
                        });
                    } else {
                        states[block_index].push(ObjectState {
                            source_channel,
                            active: !inactive,
                            bed_speaker: if in_bed {
                                bed_speakers.get(object_index).copied()
                            } else {
                                None
                            },
                            position,
                            distance_factor: properties.distance_factor,
                            gain: properties.gain,
                            size,
                            snap: properties.snap && !properties.screen_anchored,
                            zone: properties.zone,
                            elevation: properties.elevation,
                            divergence: properties.divergence,
                            trim: ObjectTrim::default_algorithm(),
                        });
                    }
                }
            }
        }

        let updates = timing
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
            .collect();
        Ok(DecodedObjectElement { updates, blocks })
    }

    fn apply_extended_object_element(
        &mut self,
        bits: &mut BitReader<'_>,
        blocks: &[Vec<ObjectBlockContext>],
        updates: &mut [SpatialUpdate],
    ) -> Result<(), AppError> {
        if bits.read_bit()? {
            for (object_index, object_blocks) in blocks.iter().enumerate() {
                let mut previous = self.properties[object_index].divergence;
                for (block_index, context) in object_blocks.iter().enumerate() {
                    let divergence = if !context.active || !context.dynamic {
                        0.0
                    } else {
                        read_divergence(bits, previous)?
                    };
                    previous = divergence;
                    if let Some(source_channel) = context.source_channel
                        && let Some(state) = updates.get_mut(block_index).and_then(|update| {
                            update
                                .objects
                                .iter_mut()
                                .find(|state| state.source_channel == source_channel)
                        })
                    {
                        state.divergence = divergence;
                    }
                }
                self.properties[object_index].divergence = previous;
            }
        } else {
            for properties in &mut self.properties {
                properties.divergence = 0.0;
            }
        }

        if bits.read_bit()? {
            for object_blocks in blocks {
                for (block_index, context) in object_blocks.iter().enumerate() {
                    if !context.active || !context.dynamic || !bits.read_bit()? {
                        continue;
                    }
                    let presence = bits.read_u8(3)?;
                    let mut position = context.room_position;
                    if presence & 1 != 0 {
                        position[0] += EXTENDED_POSITION_CODES[bits.read_usize(2)?] / 310.0;
                    }
                    if presence & 2 != 0 {
                        position[1] += EXTENDED_POSITION_CODES[bits.read_usize(2)?] / 310.0;
                    }
                    if presence & 4 != 0 {
                        position[2] += EXTENDED_POSITION_CODES[bits.read_usize(2)?] / 75.0;
                    }
                    position[0] = position[0].clamp(0.0, 1.0);
                    position[1] = position[1].clamp(0.0, 1.0);
                    position[2] = position[2].clamp(-1.0, 1.0);
                    if let Some(source_channel) = context.source_channel
                        && let Some(state) = updates.get_mut(block_index).and_then(|update| {
                            update
                                .objects
                                .iter_mut()
                                .find(|state| state.source_channel == source_channel)
                        })
                    {
                        let room_position = room_position(position);
                        let room_position = if context.screen_anchored {
                            interpolate_screen_geometry(
                                room_position,
                                [0.0; 3],
                                context.screen_factor,
                                context.depth_factor,
                            )
                            .0
                        } else {
                            room_position
                        };
                        state.position =
                            project_room_distance(room_position, context.distance_factor);
                    }
                }
            }
        }
        Ok(())
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

struct DecodedObjectElement {
    updates: Vec<SpatialUpdate>,
    blocks: Vec<Vec<ObjectBlockContext>>,
}

#[derive(Clone, Copy, Debug, Default)]
struct ObjectBlockContext {
    active: bool,
    dynamic: bool,
    source_channel: Option<usize>,
    room_position: [f32; 3],
    screen_anchored: bool,
    screen_factor: f32,
    depth_factor: f32,
    distance_factor: Option<f32>,
}

#[derive(Clone, Debug)]
struct DecodedTrim {
    trim: ObjectTrim,
    disabled_objects: Vec<bool>,
}

impl DecodedTrim {
    fn for_object(&self, object_index: usize) -> ObjectTrim {
        let mut trim = self.trim;
        trim.disabled = self
            .disabled_objects
            .get(object_index)
            .copied()
            .unwrap_or(false);
        trim
    }
}

fn read_trim_element(
    bits: &mut BitReader<'_>,
    object_count: usize,
) -> Result<DecodedTrim, AppError> {
    let warp_y = match bits.read_u8(2)? {
        0 => false,
        1 => true,
        reserved => return Err(unsupported(format!("reserved OAMD warp mode {reserved}"))),
    };
    let _reserved = bits.read_u8(2)?;
    let global_mode = bits.read_u8(2)?;
    if global_mode == 3 {
        return Err(unsupported("reserved OAMD global trim mode 3"));
    }
    let trim = match global_mode {
        0 => ObjectTrim::uniform(
            warp_y,
            ObjectTrimSettings {
                mode: ObjectTrimMode::Default,
                ..ObjectTrimSettings::default()
            },
        ),
        1 => ObjectTrim::uniform(warp_y, ObjectTrimSettings::default()),
        2 => {
            let mut configurations = [ObjectTrimSettings::default(); 9];
            for configuration in &mut configurations {
                let default_trim = bits.read_bit()?;
                let configuration_trim = if default_trim {
                    ObjectTrimSettings {
                        mode: ObjectTrimMode::Default,
                        ..ObjectTrimSettings::default()
                    }
                } else if bits.read_bit()? {
                    ObjectTrimSettings::default()
                } else {
                    let presence = bits.read_u8(5)?;
                    let center_db = if presence & 1 != 0 {
                        TRIM_LEVELS_DB[bits.read_usize(4)?]
                    } else {
                        0.0
                    };
                    let surround_db = if presence & 2 != 0 {
                        read_reduction_trim(bits, "surround")?
                    } else {
                        0.0
                    };
                    let height_db = if presence & 4 != 0 {
                        read_reduction_trim(bits, "height")?
                    } else {
                        0.0
                    };
                    let top_bottom_balance = if presence & 8 != 0 {
                        read_balance(bits)?
                    } else {
                        0.0
                    };
                    let listener_balance = if presence & 16 != 0 {
                        read_balance(bits)?
                    } else {
                        0.0
                    };
                    ObjectTrimSettings {
                        mode: ObjectTrimMode::Custom,
                        center_db,
                        surround_db,
                        height_db,
                        top_bottom_balance,
                        listener_balance,
                    }
                };
                *configuration = configuration_trim;
            }
            ObjectTrim::from_configurations(warp_y, configurations)
        }
        _ => unreachable!("reserved global mode was rejected above"),
    };
    let disabled_objects = if bits.read_bit()? {
        (0..object_count)
            .map(|_| bits.read_bit())
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    Ok(DecodedTrim {
        trim,
        disabled_objects,
    })
}

fn read_reduction_trim(bits: &mut BitReader<'_>, name: &str) -> Result<f32, AppError> {
    let code = bits.read_usize(4)?;
    if code < 4 {
        return Err(unsupported(format!(
            "reserved OAMD {name} trim code {code}"
        )));
    }
    Ok(TRIM_LEVELS_DB[code])
}

fn read_balance(bits: &mut BitReader<'_>) -> Result<f32, AppError> {
    let sign = if bits.read_bit()? { 1.0 } else { -1.0 };
    Ok(sign * (f32::from(bits.read_u8(4)?) + 1.0) / 16.0)
}

fn apply_trim_element(
    trim: &DecodedTrim,
    blocks: &[Vec<ObjectBlockContext>],
    updates: &mut [SpatialUpdate],
) -> Result<(), AppError> {
    for (object_index, object_blocks) in blocks.iter().enumerate() {
        let object_trim = trim.for_object(object_index);
        for (block_index, context) in object_blocks.iter().enumerate() {
            let Some(source_channel) = context.source_channel else {
                continue;
            };
            let update = updates.get_mut(block_index).ok_or_else(|| {
                invalid(format!(
                    "trim metadata references missing object block {block_index}"
                ))
            })?;
            if let Some(state) = update
                .objects
                .iter_mut()
                .find(|state| state.source_channel == source_channel)
            {
                state.trim = object_trim;
            } else if let Some(state) = update
                .isf
                .iter_mut()
                .find(|state| state.source_channel == source_channel)
            {
                state.trim = object_trim;
            }
        }
    }
    Ok(())
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
    snap: bool,
    zone: ObjectZone,
    elevation: bool,
    divergence: f32,
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
            snap: false,
            zone: ObjectZone::All,
            elevation: true,
            divergence: 0.0,
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
        self.snap = false;
        self.zone = ObjectZone::All;
        self.elevation = true;
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
        let zone_index = bits.read_u8(3)?;
        properties.zone = ObjectZone::try_from(zone_index)
            .map_err(|reserved| unsupported(format!("reserved OAMD zone index {reserved}")))?;
        properties.elevation = bits.read_bit()?;
    }
    if mask & 4 != 0 {
        properties.size = match bits.read_u8(2)? {
            0 => [0.0; 3],
            1 => {
                let size = f32::from(bits.read_u8(5)?) / 31.0;
                [size; 3]
            }
            2 => [
                f32::from(bits.read_u8(5)?) / 31.0,
                f32::from(bits.read_u8(5)?) / 31.0,
                f32::from(bits.read_u8(5)?) / 31.0,
            ],
            3 => return Err(unsupported("reserved OAMD object size index 3")),
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
    properties.snap = bits.read_bit()?;
    Ok(())
}

fn read_divergence(bits: &mut BitReader<'_>, previous: f32) -> Result<f32, AppError> {
    if !bits.read_bit()? {
        return Ok(0.0);
    }
    match bits.read_u8(2)? {
        0 => Ok(DIVERGENCE_TABLE[bits.read_usize(2)?]),
        1 => Ok(previous),
        2 => {
            let code = bits.read_usize(6)?;
            DIVERGENCE_CODES[code].ok_or_else(|| unsupported("reserved OAMD divergence code 0"))
        }
        reserved => Err(unsupported(format!(
            "reserved OAMD divergence mode {reserved}"
        ))),
    }
}

fn append_standard_bed(mask: u16, result: &mut Vec<Speaker>) {
    // OAMD arrays are sent from index 0, so an MSB-first integer reader puts
    // the final array element (RC_L/RC_R) in the numeric least-significant bit.
    // Iterating numeric bits upward therefore also preserves bed-object order.
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
    // Numeric bit 0 is the final transmitted array element (RC_L), while
    // numeric bit 16 is the first one (RC_LFE2).
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
    use super::{
        OamdDecoder, ObjectProperties, ProgramAssignment, append_nonstandard_bed,
        append_standard_bed, read_divergence, read_render_info, read_trim_element, room_position,
    };
    use crate::{
        eac3::{BitReader, MetadataPayload},
        hrir::Speaker,
        object::ObjectTrimMode,
    };

    #[test]
    fn bed_masks_preserve_object_order_after_msb_first_reading() {
        let mut standard_edges = Vec::new();
        append_standard_bed(1 | (1 << 9), &mut standard_edges);
        assert_eq!(
            standard_edges,
            [Speaker::FrontLeft, Speaker::FrontRight, Speaker::Lfe,]
        );

        let mut nonstandard_edges = Vec::new();
        append_nonstandard_bed(1 | (1 << 16), &mut nonstandard_edges);
        assert_eq!(nonstandard_edges, [Speaker::FrontLeft, Speaker::Lfe]);

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
    fn reserved_object_size_index_is_rejected() {
        let mut properties = ObjectProperties::default();
        let error = read_render_info(&mut BitReader::new(&[0b0100_1100]), &mut properties, 3, 1)
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("reserved OAMD object size index 3")
        );
    }

    #[test]
    fn render_info_retains_finite_and_infinite_distance() {
        let mut finite = ObjectProperties::default();
        read_render_info(
            &mut BitReader::new(&[0x17, 0xc0, 0x84, 0xe0]),
            &mut finite,
            3,
            0,
        )
        .unwrap();
        assert_eq!(finite.distance_factor, Some(5.0));

        let mut infinite = ObjectProperties::default();
        read_render_info(
            &mut BitReader::new(&[0x17, 0xc0, 0x86]),
            &mut infinite,
            3,
            0,
        )
        .unwrap();
        assert!(infinite.distance_factor.is_some_and(f32::is_infinite));
    }

    #[test]
    #[allow(clippy::float_cmp)] // Boundary values are exactly representable.
    fn room_coordinates_map_to_renderer_axes() {
        assert_eq!(room_position([0.0, 0.0, -1.0]), [-1.0, 1.0, -1.0]);
        assert_eq!(room_position([1.0, 1.0, 1.0]), [1.0, -1.0, 1.0]);
    }

    #[test]
    fn divergence_decodes_table_code_and_reuse_modes() {
        let table = read_divergence(&mut BitReader::new(&[0b1001_0000]), 0.25).unwrap();
        assert!((table - 0.704_833).abs() < f32::EPSILON);

        let reuse = read_divergence(&mut BitReader::new(&[0b1010_0000]), 0.25).unwrap();
        assert!((reuse - 0.25).abs() < f32::EPSILON);

        let code = read_divergence(&mut BitReader::new(&[0b1101_1111, 0b1000_0000]), 0.0).unwrap();
        assert!((code - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn reserved_divergence_syntax_is_rejected() {
        assert!(read_divergence(&mut BitReader::new(&[0b1100_0000, 0]), 0.0).is_err());
        assert!(read_divergence(&mut BitReader::new(&[0b1110_0000]), 0.0).is_err());
    }

    #[test]
    fn program_assignment_accepts_every_defined_isf_size() {
        for (index, expected) in [4, 8, 10, 14, 15, 30].into_iter().enumerate() {
            let encoded = 0b0001_0000 | u8::try_from(index).unwrap();
            let assignment =
                ProgramAssignment::read(&mut BitReader::new(&[encoded]), expected).unwrap();
            assert_eq!(assignment.isf_object_count, expected);
        }
    }

    #[test]
    fn decodes_custom_trim_and_warp_metadata() {
        // warp Y, custom global mode, configuration 0 centre trim -6 dB,
        // configurations 1..8 default, no per-object disables.
        let trim = read_trim_element(&mut BitReader::new(&[0x48, 0x0c, 0x7f, 0x80]), 1).unwrap();
        let object = trim.for_object(0);
        assert!(object.warp_y);
        let settings = object.settings(0);
        assert_eq!(settings.mode, ObjectTrimMode::Custom);
        assert!((settings.center_db + 6.0).abs() < f32::EPSILON);
        assert!(settings.surround_db.abs() < f32::EPSILON);
        assert!(settings.height_db.abs() < f32::EPSILON);
        assert_eq!(object.settings(1).mode, ObjectTrimMode::Default);
    }

    #[test]
    fn decodes_global_and_per_object_trim_modes() {
        let default = read_trim_element(&mut BitReader::new(&[0]), 1).unwrap();
        assert_eq!(
            default.for_object(0).settings(0).mode,
            ObjectTrimMode::Default
        );

        let disabled = read_trim_element(&mut BitReader::new(&[0x04]), 1).unwrap();
        assert_eq!(
            disabled.for_object(0).settings(0).mode,
            ObjectTrimMode::Disabled
        );

        let object_disabled = read_trim_element(&mut BitReader::new(&[0x03]), 1).unwrap();
        assert_eq!(
            object_disabled.for_object(0).settings(0).mode,
            ObjectTrimMode::Disabled
        );
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
