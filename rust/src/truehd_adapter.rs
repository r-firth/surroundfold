//! Narrow compatibility layer around the experimental `truehd` decoder.
//!
//! Nothing outside this module depends on the decoder crate's public data
//! structures. That keeps upgrades localized and gives the renderer a stable,
//! product-owned representation of PCM and object metadata.

use std::{
    io::Read,
    panic::{AssertUnwindSafe, catch_unwind},
};

use log::Level;
use truehd::{
    process::{
        decode::{DecodedAccessUnit, Decoder},
        extract::Extractor,
        parse::Parser,
    },
    structs::{channel::ChannelLabel, oamd::ObjectAudioMetadataPayload},
    utils::errors::ExtractError,
};

use crate::{
    error::AppError,
    hrir::Speaker,
    isf::IsfConfig,
    object::{
        IsfState, ObjectState, ObjectTrim, ObjectTrimMode, ObjectTrimSettings, ObjectZone,
        SpatialUpdate, interpolate_screen_geometry, project_room_distance,
    },
    stream_io::read_up_to,
};

const INPUT_CHUNK_BYTES: usize = 64 * 1024;
const PCM_24BIT_SCALE: f32 = 8_388_608.0;
const DISTANCE_FACTORS: [f32; 16] = [
    1.1, 1.3, 1.6, 2.0, 2.5, 3.2, 4.0, 5.0, 6.3, 7.9, 10.0, 12.6, 15.8, 20.0, 25.1, 50.1,
];

#[derive(Clone, Debug, PartialEq)]
pub struct TrueHdFrame {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub channel_count: usize,
    pub samples: Vec<f32>,
    pub channel_speakers: Vec<Option<Speaker>>,
    pub isf: Option<IsfConfig>,
    pub spatial_updates: Vec<SpatialUpdate>,
}

/// Streams an elementary TrueHD/MLP bitstream through the embedded decoder.
///
/// The callback is invoked once per non-duplicate access unit. Validation is
/// warning-strict unless `relaxed_validation` is set.
///
/// # Errors
///
/// Returns an error for I/O, extraction, parsing, decoding, unsupported object
/// layouts, decoder panics, or callback failures.
pub fn decode_stream(
    mut input: impl Read,
    presentation: u8,
    relaxed_validation: bool,
    mut on_frame: impl FnMut(TrueHdFrame) -> Result<(), AppError>,
) -> Result<(), AppError> {
    let result = catch_unwind(AssertUnwindSafe(|| {
        decode_stream_inner(&mut input, presentation, relaxed_validation, &mut on_frame)
    }));
    match result {
        Ok(result) => result,
        Err(payload) => Err(AppError::Render(format!(
            "embedded TrueHD decoder panicked: {}",
            panic_message(&payload)
        ))),
    }
}

fn decode_stream_inner(
    input: &mut impl Read,
    presentation: u8,
    relaxed_validation: bool,
    on_frame: &mut impl FnMut(TrueHdFrame) -> Result<(), AppError>,
) -> Result<(), AppError> {
    let mut extractor = Extractor::default();
    let mut parser = Parser::default();
    let mut decoder = Decoder::default();
    if !relaxed_validation {
        parser.set_fail_level(Level::Warn);
        decoder.set_fail_level(Level::Warn);
    }

    let mut bytes = vec![0_u8; INPUT_CHUNK_BYTES];
    let mut decoded_frames = 0_u64;
    loop {
        let read = read_up_to(input, &mut bytes)?;
        if read == 0 {
            break;
        }
        extractor.push_bytes(&bytes[..read]);
        drain_frames(
            &mut extractor,
            &mut parser,
            &mut decoder,
            presentation,
            relaxed_validation,
            &mut decoded_frames,
            on_frame,
        )?;
    }
    drain_frames(
        &mut extractor,
        &mut parser,
        &mut decoder,
        presentation,
        relaxed_validation,
        &mut decoded_frames,
        on_frame,
    )?;
    if decoded_frames == 0 {
        return Err(AppError::Render(
            "embedded TrueHD decoder produced no audio frames".into(),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn drain_frames(
    extractor: &mut Extractor,
    parser: &mut Parser,
    decoder: &mut Decoder,
    presentation: u8,
    relaxed_validation: bool,
    decoded_frames: &mut u64,
    on_frame: &mut impl FnMut(TrueHdFrame) -> Result<(), AppError>,
) -> Result<(), AppError> {
    loop {
        let frame = match extractor.next() {
            Some(Ok(frame)) => frame,
            Some(Err(ExtractError::InsufficientData)) | None => break,
            Some(Err(error)) => {
                if relaxed_validation {
                    continue;
                }
                return Err(AppError::Render(format!(
                    "TrueHD frame extraction failed: {error}"
                )));
            }
        };
        let access_unit = match parser.parse(&frame) {
            Ok(access_unit) => access_unit,
            Err(error) => {
                if relaxed_validation {
                    continue;
                }
                return Err(AppError::Render(format!(
                    "TrueHD access-unit parsing failed: {error:#}"
                )));
            }
        };
        let decoded_au = match decoder.decode_presentation(&access_unit, usize::from(presentation))
        {
            Ok(decoded_au) => decoded_au,
            Err(error) => {
                if relaxed_validation {
                    continue;
                }
                return Err(AppError::Render(format!(
                    "TrueHD presentation {presentation} decoding failed: {error:#}"
                )));
            }
        };
        if decoded_au.is_duplicate {
            continue;
        }
        on_frame(convert_frame(&decoded_au)?)?;
        *decoded_frames = decoded_frames
            .checked_add(1)
            .ok_or_else(|| AppError::Render("TrueHD frame count overflowed".into()))?;
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)] // TrueHD's 24-bit integer PCM maps exactly enough for f32 DSP.
fn convert_frame(decoded: &DecodedAccessUnit) -> Result<TrueHdFrame, AppError> {
    if decoded.sample_length > decoded.pcm_data.len() || decoded.channel_count > 16 {
        return Err(AppError::Render(format!(
            "TrueHD decoder returned invalid dimensions: {} samples, {} channels",
            decoded.sample_length, decoded.channel_count
        )));
    }
    if decoded.sampling_frequency == 0 || decoded.channel_count == 0 {
        return Err(AppError::Render(
            "TrueHD decoder returned an invalid sample rate or channel count".into(),
        ));
    }

    let mut samples = Vec::with_capacity(decoded.sample_length * decoded.channel_count);
    for frame in &decoded.pcm_data[..decoded.sample_length] {
        samples.extend(
            frame[..decoded.channel_count]
                .iter()
                .map(|sample| *sample as f32 / PCM_24BIT_SCALE),
        );
    }
    let mut channel_speakers = (0..decoded.channel_count)
        .map(|channel| {
            decoded
                .channel_labels
                .get(channel)
                .copied()
                .map(channel_label_to_speaker)
        })
        .collect::<Vec<_>>();
    let mut isf = None;
    let mut spatial_updates = Vec::new();
    for payload in &decoded.oamd {
        let updates = metadata_updates(payload, decoded.channel_count, decoded.sampling_frequency)?;
        for (channel, speaker) in updates
            .first()
            .into_iter()
            .flat_map(|update| update.bed_speakers.iter().copied())
            .enumerate()
        {
            if channel < channel_speakers.len() {
                channel_speakers[channel] = Some(speaker);
            }
        }
        if payload.program_assignment.num_isf_objects != 0 {
            let config = IsfConfig::new(
                payload.program_assignment.num_bed_objects,
                payload.program_assignment.num_isf_objects,
            )?;
            if config
                .start_channel
                .checked_add(config.channel_count)
                .is_none_or(|end| end > decoded.channel_count)
            {
                return Err(AppError::Render(format!(
                    "TrueHD metadata describes an ISF range beyond {} decoded channels",
                    decoded.channel_count
                )));
            }
            if isf
                .replace(config)
                .is_some_and(|previous| previous != config)
            {
                return Err(AppError::UnsupportedInput(
                    "TrueHD access unit contains inconsistent ISF assignments".into(),
                ));
            }
        }
        spatial_updates.extend(updates);
    }

    Ok(TrueHdFrame {
        sample_rate: decoded.sampling_frequency,
        sample_count: decoded.sample_length,
        channel_count: decoded.channel_count,
        samples,
        channel_speakers,
        isf,
        spatial_updates,
    })
}

fn metadata_updates(
    payload: &ObjectAudioMetadataPayload,
    channel_count: usize,
    sample_rate: u32,
) -> Result<Vec<SpatialUpdate>, AppError> {
    let bed_indices = payload
        .program_assignment
        .bed_assignment
        .iter()
        .flat_map(truehd::structs::oamd::BedAssignment::to_index_vec)
        .collect::<Vec<_>>();
    let bed_speakers = bed_indices
        .iter()
        .copied()
        .map(bed_index_to_speaker)
        .collect::<Result<Vec<_>, _>>()?;
    if bed_speakers.len() != payload.program_assignment.num_bed_objects {
        return Err(AppError::Render(format!(
            "TrueHD metadata bed assignment contains {} channels but declares {}",
            bed_speakers.len(),
            payload.program_assignment.num_bed_objects
        )));
    }
    if bed_speakers.len() > channel_count {
        return Err(AppError::Render(format!(
            "TrueHD metadata describes {} bed channels but decoder returned {channel_count} channels",
            bed_speakers.len()
        )));
    }

    let Some(element) = &payload.object_element else {
        return Ok(vec![SpatialUpdate {
            sample_offset: 0,
            ramp_samples: 0,
            bed_speakers,
            isf: Vec::new(),
            objects: Vec::new(),
        }]);
    };
    let positions = payload.get_damf_pos();
    let dynamic_count = payload.program_assignment.num_dynamic_objects;
    let described_channels = bed_speakers
        .len()
        .saturating_add(payload.program_assignment.num_isf_objects)
        .saturating_add(dynamic_count);
    if described_channels > channel_count {
        return Err(AppError::Render(format!(
            "TrueHD metadata describes {} bed, {} ISF, and {dynamic_count} dynamic channels, but decoder returned {channel_count}",
            bed_speakers.len(),
            payload.program_assignment.num_isf_objects
        )));
    }

    let block_count = element.md_update_info.num_obj_info_blocks;
    if block_count == 0 {
        return Ok(Vec::new());
    }
    let block_metadata = BlockMetadata {
        element,
        extended: payload.extended_object_element.as_ref(),
        trim: payload.trim_element.as_ref(),
        positions: &positions,
        bed_speakers: &bed_speakers,
        isf_count: payload.program_assignment.num_isf_objects,
        dynamic_count,
    };
    (0..block_count)
        .map(|block_index| {
            let states = states_for_block(&block_metadata, block_index)?;
            let timing = element
                .md_update_info
                .block_update_info
                .get(block_index)
                .ok_or_else(|| {
                    AppError::Render(format!(
                        "TrueHD metadata is missing timing for object block {block_index}"
                    ))
                })?;
            let block_offset = usize::from(timing.block_offset_factor_bits)
                .checked_mul(32)
                .ok_or_else(|| AppError::Render("TrueHD OAMD block offset overflowed".into()))?;
            let oamd_offset = element
                .md_update_info
                .sample_offset
                .checked_add(block_offset)
                .ok_or_else(|| AppError::Render("TrueHD OAMD sample offset overflowed".into()))?;
            let evo_offset = usize::try_from(payload.evo_sample_offset).map_err(|error| {
                AppError::Render(format!(
                    "TrueHD evolution sample offset overflowed: {error}"
                ))
            })?;
            let scaled_oamd_offset = scale_oamd_samples(oamd_offset, sample_rate)?;
            let sample_offset = evo_offset.checked_add(scaled_oamd_offset).ok_or_else(|| {
                AppError::Render("TrueHD metadata sample offset overflowed".into())
            })?;

            Ok(SpatialUpdate {
                sample_offset,
                ramp_samples: scale_oamd_samples(usize::from(timing.ramp_duration), sample_rate)?,
                bed_speakers: bed_speakers.clone(),
                isf: states.isf,
                objects: states.objects,
            })
        })
        .collect()
}

struct BlockMetadata<'a> {
    element: &'a truehd::structs::oamd::ObjectElement,
    extended: Option<&'a truehd::structs::oamd::ExtendedObjectElement>,
    trim: Option<&'a truehd::structs::oamd::TrimElement>,
    positions: &'a [Vec<[f64; 3]>],
    bed_speakers: &'a [Speaker],
    isf_count: usize,
    dynamic_count: usize,
}

#[allow(clippy::cast_possible_truncation)] // OAMD coordinates are deliberately reduced to f32 DSP.
fn states_for_block(
    metadata: &BlockMetadata<'_>,
    block_index: usize,
) -> Result<BlockStates, AppError> {
    let bed_count = metadata.bed_speakers.len();
    let isf = (0..metadata.isf_count)
        .map(|isf_index| {
            let metadata_index = bed_count + isf_index;
            let info = metadata
                .element
                .object_data
                .get(metadata_index)
                .and_then(|blocks| blocks.get(block_index))
                .ok_or_else(|| {
                    AppError::Render(format!(
                        "TrueHD metadata is missing block {block_index} for ISF signal {isf_index}"
                    ))
                })?;
            Ok(IsfState {
                source_channel: metadata_index,
                active: !info.b_object_not_active,
                gain: object_gain(info.object_basic_info.object_gain),
                trim: object_trim(metadata.trim, metadata_index)?,
            })
        })
        .collect::<Result<Vec<_>, AppError>>()?;
    let beds = metadata
        .bed_speakers
        .iter()
        .copied()
        .enumerate()
        .map(|(metadata_index, speaker)| {
            let info = metadata
                .element
                .object_data
                .get(metadata_index)
                .and_then(|blocks| blocks.get(block_index))
                .ok_or_else(|| {
                    AppError::Render(format!(
                        "TrueHD metadata is missing block {block_index} for bed object {metadata_index}"
                    ))
                })?;
            Ok(ObjectState {
                source_channel: metadata_index,
                active: !info.b_object_not_active,
                bed_speaker: Some(speaker),
                position: speaker.position(),
                distance_factor: None,
                gain: object_gain(info.object_basic_info.object_gain),
                size: [0.0; 3],
                snap: false,
                zone: ObjectZone::All,
                elevation: true,
                divergence: 0.0,
                trim: object_trim(metadata.trim, metadata_index)?,
            })
        })
        .collect::<Result<Vec<_>, AppError>>()?;
    let dynamic = (0..metadata.dynamic_count)
        .map(|dynamic_index| dynamic_state(metadata, block_index, dynamic_index))
        .collect::<Result<Vec<_>, AppError>>()?;
    let objects = beds.into_iter().chain(dynamic).collect();
    Ok(BlockStates { isf, objects })
}

#[allow(clippy::cast_possible_truncation)] // OAMD coordinates are deliberately reduced to f32 DSP.
fn dynamic_state(
    metadata: &BlockMetadata<'_>,
    block_index: usize,
    dynamic_index: usize,
) -> Result<ObjectState, AppError> {
    let bed_count = metadata.bed_speakers.len();
    let metadata_index = bed_count + metadata.isf_count + dynamic_index;
    let info = metadata
        .element
        .object_data
        .get(metadata_index)
        .and_then(|blocks| blocks.get(block_index))
        .ok_or_else(|| {
            AppError::Render(format!(
                "TrueHD metadata is missing block {block_index} for dynamic object {dynamic_index}"
            ))
        })?;
    let position = metadata
        .positions
        .get(metadata_index)
        .and_then(|blocks| blocks.get(block_index))
        .copied()
        .unwrap_or([0.0, 0.0, 0.0])
        .map(|value| value as f32);
    let size = info
        .object_render_info
        .object_size
        .map(|value| value as f32);
    let screen_anchored = info.object_render_info.b_object_use_screen_ref;
    let (position, size) = if screen_anchored {
        interpolate_screen_geometry(
            position,
            size,
            info.object_render_info.screen_factor as f32,
            info.object_render_info.depth_factor as f32,
        )
    } else {
        (position, size)
    };
    let distance_factor = info
        .object_render_info
        .b_object_distance_specified
        .then(|| {
            if info.object_render_info.b_object_at_infinity {
                Ok(f32::INFINITY)
            } else {
                DISTANCE_FACTORS
                    .get(usize::from(info.object_render_info.distance_factor_idx))
                    .copied()
                    .ok_or_else(|| {
                        AppError::UnsupportedInput(format!(
                            "invalid TrueHD OAMD distance index {}",
                            info.object_render_info.distance_factor_idx
                        ))
                    })
            }
        })
        .transpose()?;
    let divergence = metadata
        .extended
        .and_then(|extended| extended.object_div_block.get(metadata_index))
        .and_then(|blocks| blocks.get(block_index))
        .filter(|block| block.b_object_divergence)
        .map_or(0.0, |block| block.object_divergence as f32);
    Ok(ObjectState {
        source_channel: metadata_index,
        active: !info.b_object_not_active,
        bed_speaker: None,
        position: project_room_distance(position, distance_factor),
        distance_factor,
        gain: object_gain(info.object_basic_info.object_gain),
        size,
        snap: info.object_render_info.b_object_snap && !screen_anchored,
        zone: ObjectZone::try_from(info.object_render_info.zone_constraints_idx).map_err(
            |reserved| {
                AppError::UnsupportedInput(format!("reserved TrueHD OAMD zone index {reserved}"))
            },
        )?,
        elevation: info.object_render_info.b_enable_elevation,
        divergence,
        trim: object_trim(metadata.trim, metadata_index)?,
    })
}

struct BlockStates {
    isf: Vec<IsfState>,
    objects: Vec<ObjectState>,
}

fn object_gain(gain_db: i8) -> f32 {
    if gain_db == truehd::structs::oamd::GAIN_MINUS_INFINITY {
        0.0
    } else {
        10_f32.powf(f32::from(gain_db) / 20.0)
    }
}

#[allow(clippy::cast_possible_truncation)] // OAMD trim values are bounded table entries.
fn object_trim(
    element: Option<&truehd::structs::oamd::TrimElement>,
    object_index: usize,
) -> Result<ObjectTrim, AppError> {
    let Some(element) = element else {
        return Ok(ObjectTrim::default_algorithm());
    };
    let warp_y = match element.warp_mode {
        0 => false,
        1 => true,
        reserved => {
            return Err(AppError::UnsupportedInput(format!(
                "reserved TrueHD OAMD warp mode {reserved}"
            )));
        }
    };
    if element.global_trim_mode == 3 {
        return Err(AppError::UnsupportedInput(
            "reserved TrueHD OAMD global trim mode 3".into(),
        ));
    }
    let object_disabled = if element.b_disable_trim_per_obj {
        *element.b_disable_trim.get(object_index).ok_or_else(|| {
            AppError::Render(format!(
                "TrueHD OAMD trim metadata is missing object {object_index}"
            ))
        })?
    } else {
        false
    };
    if object_disabled || element.global_trim_mode == 1 {
        return Ok(ObjectTrim::uniform(warp_y, ObjectTrimSettings::default()));
    }
    if element.global_trim_mode == 0 {
        return Ok(ObjectTrim::uniform(
            warp_y,
            ObjectTrimSettings {
                mode: ObjectTrimMode::Default,
                ..ObjectTrimSettings::default()
            },
        ));
    }

    let mut configurations = [ObjectTrimSettings::default(); 9];
    for (configuration, encoded) in element.trims.iter().enumerate() {
        let Some(encoded) = encoded else {
            configurations[configuration] = ObjectTrimSettings {
                mode: ObjectTrimMode::Default,
                ..ObjectTrimSettings::default()
            };
            continue;
        };
        configurations[configuration] = if encoded.b_default_trim {
            ObjectTrimSettings {
                mode: ObjectTrimMode::Default,
                ..ObjectTrimSettings::default()
            }
        } else if encoded.b_disable_trim {
            ObjectTrimSettings::default()
        } else {
            for (name, value) in [
                ("surround", encoded.trim_surround),
                ("height", encoded.trim_height),
            ] {
                if value.is_some_and(|db| db > -0.75) {
                    return Err(AppError::UnsupportedInput(format!(
                        "reserved TrueHD OAMD {name} trim value"
                    )));
                }
            }
            ObjectTrimSettings {
                mode: ObjectTrimMode::Custom,
                center_db: encoded.trim_centre.unwrap_or(0.0) as f32,
                surround_db: encoded.trim_surround.unwrap_or(0.0) as f32,
                height_db: encoded.trim_height.unwrap_or(0.0) as f32,
                top_bottom_balance: encoded.bal3d_y_tb.unwrap_or(0.0) as f32,
                listener_balance: encoded.bal3d_y_lis.unwrap_or(0.0) as f32,
            }
        };
    }
    Ok(ObjectTrim::from_configurations(warp_y, configurations))
}

fn scale_oamd_samples(samples: usize, sample_rate: u32) -> Result<usize, AppError> {
    let samples = u64::try_from(samples)
        .map_err(|error| AppError::Render(format!("OAMD sample offset overflowed: {error}")))?;
    let scaled = u128::from(samples)
        .checked_mul(u128::from(sample_rate))
        .ok_or_else(|| AppError::Render("scaled OAMD sample offset overflowed".into()))?
        / 48_000;
    usize::try_from(scaled)
        .map_err(|error| AppError::Render(format!("scaled OAMD sample offset overflowed: {error}")))
}

const fn channel_label_to_speaker(label: ChannelLabel) -> Speaker {
    match label {
        ChannelLabel::L => Speaker::FrontLeft,
        ChannelLabel::R => Speaker::FrontRight,
        ChannelLabel::C => Speaker::FrontCenter,
        ChannelLabel::LFE | ChannelLabel::LFE2 => Speaker::Lfe,
        ChannelLabel::Ls | ChannelLabel::Lsc => Speaker::SideLeft,
        ChannelLabel::Rs | ChannelLabel::Rsc => Speaker::SideRight,
        ChannelLabel::Lb | ChannelLabel::Lsd => Speaker::RearLeft,
        ChannelLabel::Rb | ChannelLabel::Rsd => Speaker::RearRight,
        ChannelLabel::Cb => Speaker::RearCenter,
        ChannelLabel::Tfl => Speaker::TopFrontLeft,
        ChannelLabel::Tfr => Speaker::TopFrontRight,
        ChannelLabel::Tsl => Speaker::TopSideLeft,
        ChannelLabel::Tsr => Speaker::TopSideRight,
        ChannelLabel::Tbl => Speaker::TopRearLeft,
        ChannelLabel::Tbr => Speaker::TopRearRight,
        ChannelLabel::Tc | ChannelLabel::Tfc => Speaker::TopFrontCenter,
        ChannelLabel::Lw => Speaker::WideLeft,
        ChannelLabel::Rw => Speaker::WideRight,
    }
}

fn bed_index_to_speaker(index: usize) -> Result<Speaker, AppError> {
    let speaker = match index {
        0 => Speaker::FrontLeft,
        1 => Speaker::FrontRight,
        2 => Speaker::FrontCenter,
        3 | 16 => Speaker::Lfe,
        4 => Speaker::SideLeft,
        5 => Speaker::SideRight,
        6 => Speaker::RearLeft,
        7 => Speaker::RearRight,
        8 => Speaker::TopFrontLeft,
        9 => Speaker::TopFrontRight,
        10 => Speaker::TopSideLeft,
        11 => Speaker::TopSideRight,
        12 => Speaker::TopRearLeft,
        13 => Speaker::TopRearRight,
        14 => Speaker::WideLeft,
        15 => Speaker::WideRight,
        _ => {
            return Err(AppError::UnsupportedInput(format!(
                "unsupported TrueHD bed speaker index {index}"
            )));
        }
    };
    Ok(speaker)
}

fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> &str {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic payload")
}

#[cfg(test)]
mod tests {
    use crate::object::ObjectTrimMode;
    use truehd::process::EXAMPLE_DATA;
    use truehd::structs::oamd::{
        BedAssignment, BlockUpdateInfo, ExtendedObjectElement, MDUpdateInfo,
        ObjectAudioMetadataPayload, ObjectDivergenceBlock, ObjectElement, ObjectInfoBlock,
        ObjectRenderInfo, ProgramAssignment, Trim, TrimElement,
    };

    use super::{Speaker, decode_stream, metadata_updates};

    #[test]
    fn embedded_example_decodes_without_leaking_decoder_types() {
        let mut frames = Vec::new();
        decode_stream(EXAMPLE_DATA, 0, false, |frame| {
            frames.push(frame);
            Ok(())
        })
        .unwrap();

        assert!(!frames.is_empty());
        assert!(frames.iter().all(|frame| frame.sample_rate > 0));
        assert!(frames.iter().all(|frame| {
            frame.samples.len() == frame.sample_count.saturating_mul(frame.channel_count)
        }));
    }

    #[test]
    fn every_oamd_block_becomes_a_scaled_timed_update() {
        let payload = ObjectAudioMetadataPayload {
            object_count: 1,
            program_assignment: ProgramAssignment {
                num_dynamic_objects: 1,
                ..ProgramAssignment::default()
            },
            object_element: Some(ObjectElement {
                md_update_info: MDUpdateInfo {
                    sample_offset: 8,
                    num_obj_info_blocks: 2,
                    block_update_info: vec![
                        BlockUpdateInfo {
                            block_offset_factor_bits: 0,
                            ramp_duration: 32,
                            ..BlockUpdateInfo::default()
                        },
                        BlockUpdateInfo {
                            block_offset_factor_bits: 2,
                            ramp_duration: 64,
                            ..BlockUpdateInfo::default()
                        },
                    ],
                },
                object_data: vec![vec![ObjectInfoBlock::default(), ObjectInfoBlock::default()]],
                ..ObjectElement::default()
            }),
            ..ObjectAudioMetadataPayload::default()
        };

        let updates = metadata_updates(&payload, 1, 96_000).unwrap();
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].sample_offset, 16);
        assert_eq!(updates[0].ramp_samples, 64);
        assert_eq!(updates[1].sample_offset, 144);
        assert_eq!(updates[1].ramp_samples, 128);
    }

    #[test]
    fn isf_signals_precede_dynamic_objects_in_decoder_channel_order() {
        let mut object_data = vec![vec![ObjectInfoBlock::default()]; 5];
        for blocks in &mut object_data {
            blocks[0].object_basic_info.object_gain = 0;
        }
        let payload = ObjectAudioMetadataPayload {
            object_count: 5,
            program_assignment: ProgramAssignment {
                num_isf_objects: 4,
                num_dynamic_objects: 1,
                ..ProgramAssignment::default()
            },
            object_element: Some(ObjectElement {
                md_update_info: MDUpdateInfo {
                    num_obj_info_blocks: 1,
                    block_update_info: vec![BlockUpdateInfo::default()],
                    ..MDUpdateInfo::default()
                },
                object_data,
                ..ObjectElement::default()
            }),
            ..ObjectAudioMetadataPayload::default()
        };

        let updates = metadata_updates(&payload, 5, 48_000).unwrap();
        assert_eq!(updates.len(), 1);
        assert_eq!(
            updates[0]
                .isf
                .iter()
                .map(|state| state.source_channel)
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
        assert_eq!(updates[0].objects[0].source_channel, 4);
        assert_eq!(
            updates[0].objects[0].trim.settings(8).mode,
            ObjectTrimMode::Default
        );
    }

    #[test]
    fn bed_gain_and_custom_trim_reach_the_renderer() {
        let mut trims = std::array::from_fn(|_| None);
        trims[2] = Some(Trim {
            b_default_trim: false,
            b_disable_trim: false,
            trim_centre: Some(-6.0),
            trim_surround: Some(-3.0),
            trim_height: None,
            bal3d_y_tb: None,
            bal3d_y_lis: Some(-0.5),
        });
        let mut info = ObjectInfoBlock::default();
        info.object_basic_info.object_gain = -6;
        let payload = ObjectAudioMetadataPayload {
            object_count: 1,
            program_assignment: ProgramAssignment {
                bed_assignment: vec![BedAssignment::from_non_std(1 << 2)],
                num_bed_objects: 1,
                ..ProgramAssignment::default()
            },
            object_element: Some(ObjectElement {
                md_update_info: MDUpdateInfo {
                    num_obj_info_blocks: 1,
                    block_update_info: vec![BlockUpdateInfo::default()],
                    ..MDUpdateInfo::default()
                },
                object_data: vec![vec![info]],
                ..ObjectElement::default()
            }),
            trim_element: Some(TrimElement {
                warp_mode: 1,
                global_trim_mode: 2,
                trims,
                ..TrimElement::default()
            }),
            ..ObjectAudioMetadataPayload::default()
        };

        let state = &metadata_updates(&payload, 1, 48_000).unwrap()[0].objects[0];
        assert_eq!(state.bed_speaker, Some(Speaker::FrontCenter));
        assert!((state.gain - 10_f32.powf(-6.0 / 20.0)).abs() < 1e-6);
        assert!(state.trim.warp_y);
        let trim = state.trim.settings(2);
        assert_eq!(trim.mode, ObjectTrimMode::Custom);
        assert!((trim.center_db + 6.0).abs() < f32::EPSILON);
        assert!((trim.surround_db + 3.0).abs() < f32::EPSILON);
        assert!((trim.listener_balance + 0.5).abs() < f32::EPSILON);
        assert_eq!(state.trim.settings(1).mode, ObjectTrimMode::Default);
    }

    #[test]
    fn screen_reference_distance_and_divergence_reach_the_render_state() {
        let payload = ObjectAudioMetadataPayload {
            object_count: 1,
            program_assignment: ProgramAssignment {
                num_dynamic_objects: 1,
                ..ProgramAssignment::default()
            },
            object_element: Some(ObjectElement {
                md_update_info: MDUpdateInfo {
                    num_obj_info_blocks: 1,
                    block_update_info: vec![BlockUpdateInfo::default()],
                    ..MDUpdateInfo::default()
                },
                b_reserved_data_not_present: false,
                reserved_data: 31,
                object_data: vec![vec![ObjectInfoBlock {
                    object_render_info: ObjectRenderInfo {
                        pos3d: [0.0, 0.0, 1.0],
                        b_object_use_screen_ref: true,
                        screen_factor: 1.0,
                        depth_factor: 1.0,
                        b_object_distance_specified: true,
                        distance_factor_idx: 7,
                        b_object_snap: true,
                        ..ObjectRenderInfo::default()
                    },
                    ..ObjectInfoBlock::default()
                }]],
            }),
            extended_object_element: Some(ExtendedObjectElement {
                b_obj_div_block: true,
                object_div_block: vec![vec![ObjectDivergenceBlock {
                    b_object_divergence: true,
                    object_divergence: 0.704_833,
                    ..ObjectDivergenceBlock::default()
                }]],
                ..ExtendedObjectElement::default()
            }),
            ..ObjectAudioMetadataPayload::default()
        };

        let object = &metadata_updates(&payload, 1, 48_000).unwrap()[0].objects[0];
        assert_eq!(object.distance_factor, Some(5.0));
        assert!((object.position[0] + 2.5).abs() < f32::EPSILON);
        assert!((object.position[1] - 5.0).abs() < f32::EPSILON);
        assert!((object.position[2] - 2.5 / 1.78).abs() < 1e-6);
        assert!((object.divergence - 0.704_833).abs() < f32::EPSILON);
        assert!(!object.snap);
    }
}
