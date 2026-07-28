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
    object::{IsfState, ObjectState, SpatialUpdate},
    stream_io::read_up_to,
};

const INPUT_CHUNK_BYTES: usize = 64 * 1024;
const PCM_24BIT_SCALE: f32 = 8_388_608.0;

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
    let dynamic_start =
        payload.program_assignment.num_bed_objects + payload.program_assignment.num_isf_objects;
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
    (0..block_count)
        .map(|block_index| {
            let states = states_for_block(
                element,
                &positions,
                bed_speakers.len(),
                payload.program_assignment.num_isf_objects,
                dynamic_start,
                dynamic_count,
                block_index,
            )?;
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

#[allow(clippy::cast_possible_truncation)] // OAMD coordinates are deliberately reduced to f32 DSP.
fn states_for_block(
    element: &truehd::structs::oamd::ObjectElement,
    positions: &[Vec<[f64; 3]>],
    bed_count: usize,
    isf_count: usize,
    dynamic_start: usize,
    dynamic_count: usize,
    block_index: usize,
) -> Result<BlockStates, AppError> {
    let isf = (0..isf_count)
        .map(|isf_index| {
            let metadata_index = bed_count + isf_index;
            let info = element
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
            })
        })
        .collect::<Result<Vec<_>, AppError>>()?;
    let objects = (0..dynamic_count)
        .map(|dynamic_index| {
            let metadata_index = dynamic_start + dynamic_index;
            let info = element
                .object_data
                .get(metadata_index)
                .and_then(|blocks| blocks.get(block_index))
                .ok_or_else(|| {
                    AppError::Render(format!(
                        "TrueHD metadata is missing block {block_index} for dynamic object {dynamic_index}"
                    ))
                })?;
            let position = positions
                .get(metadata_index)
                .and_then(|blocks| blocks.get(block_index))
                .copied()
                .unwrap_or([0.0, 0.0, 0.0])
                .map(|value| value as f32);
            Ok(ObjectState {
                source_channel: bed_count + isf_count + dynamic_index,
                active: !info.b_object_not_active,
                bed: false,
                position,
                gain: object_gain(info.object_basic_info.object_gain),
                size: info.object_render_info.object_size[0] as f32,
            })
        })
        .collect::<Result<Vec<_>, AppError>>()?;
    Ok(BlockStates { isf, objects })
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
    use truehd::process::EXAMPLE_DATA;
    use truehd::structs::oamd::{
        BlockUpdateInfo, MDUpdateInfo, ObjectAudioMetadataPayload, ObjectElement, ObjectInfoBlock,
        ProgramAssignment,
    };

    use super::{decode_stream, metadata_updates};

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
}
