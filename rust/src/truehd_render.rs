use std::{fs::File, io::BufReader, path::Path};

use crate::{
    binaural::BinauralWriter,
    error::AppError,
    hrir::HrirSet,
    object_render::{ObjectPcmFrame, ObjectRenderOptions, ObjectRenderer},
    process::ProcessRunner,
    render::{RenderResult, demux_copy},
    room::RoomCorrection,
    truehd_adapter::decode_stream,
};

#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)] // Independent render switches mirror the public CLI.
pub struct TrueHdRenderOptions {
    pub presentation: u8,
    pub relaxed_validation: bool,
    pub gain_db: f64,
    pub surround_swap: bool,
    pub mute_bed: bool,
    pub mute_ground: bool,
    pub speaker_virtualizer: bool,
}

/// Demuxes a selected `TrueHD` stream, decodes it in-process, and renders beds
/// and moving objects through the selected HRIR.
///
/// # Errors
///
/// Returns an error for demux, decode, metadata, routing, convolution, or WAV
/// output failures.
#[allow(clippy::too_many_arguments)]
pub fn render_truehd_track(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    input: &Path,
    stream_index: usize,
    hrir: &HrirSet,
    room_correction: Option<&RoomCorrection>,
    options: TrueHdRenderOptions,
    elementary_path: &Path,
    output_wave: &Path,
) -> Result<RenderResult, AppError> {
    demux_copy(
        runner,
        ffmpeg,
        input,
        stream_index,
        "truehd",
        "TrueHD",
        elementary_path,
    )?;
    let writer = BinauralWriter::new(
        output_wave,
        hrir,
        room_correction,
        options.gain_db,
        hrir.channels.iter().map(|channel| channel.speaker),
    )?;
    let mut renderer = ObjectRenderer::new(
        writer,
        hrir,
        ObjectRenderOptions {
            surround_swap: options.surround_swap,
            mute_bed: options.mute_bed,
            mute_ground: options.mute_ground,
            speaker_virtualizer: options.speaker_virtualizer,
        },
    )?;
    let source = File::open(elementary_path).map_err(|source| AppError::File {
        path: elementary_path.to_path_buf(),
        source,
    })?;
    decode_stream(
        BufReader::new(source),
        options.presentation,
        options.relaxed_validation,
        |frame| {
            runner.check_cancelled()?;
            renderer.push(ObjectPcmFrame {
                sample_rate: frame.sample_rate,
                sample_count: frame.sample_count,
                channel_count: frame.channel_count,
                samples: frame.samples,
                channel_speakers: frame.channel_speakers,
                isf: frame.isf,
                spatial_updates: frame.spatial_updates,
            })
        },
    )?;
    renderer.finish()
}
