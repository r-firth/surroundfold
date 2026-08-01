use std::{
    ffi::OsString,
    io::{BufReader, Read},
    path::Path,
    sync::mpsc::sync_channel,
    thread,
};

use crate::{
    binaural::BinauralWriter,
    cli::{DistanceRendererMode, ObjectRendererMode},
    error::AppError,
    hrir::HrirSet,
    object_render::{ObjectPcmFrame, ObjectRenderOptions, ObjectRenderer},
    process::ProcessRunner,
    render::RenderResult,
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
    pub object_renderer: ObjectRendererMode,
    pub distance_renderer: DistanceRendererMode,
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
    output_wave: &Path,
) -> Result<RenderResult, AppError> {
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
            object_renderer: options.object_renderer,
            distance_renderer: options.distance_renderer,
        },
    )?;
    let arguments = [
        OsString::from("-nostdin"),
        OsString::from("-hide_banner"),
        OsString::from("-loglevel"),
        OsString::from("error"),
        OsString::from("-i"),
        input.as_os_str().to_os_string(),
        OsString::from("-map"),
        OsString::from(format!("0:{stream_index}")),
        OsString::from("-vn"),
        OsString::from("-sn"),
        OsString::from("-dn"),
        OsString::from("-c:a"),
        OsString::from("copy"),
        OsString::from("-f"),
        OsString::from("truehd"),
        OsString::from("pipe:1"),
    ];
    let process = runner.run_with_stdout(ffmpeg, arguments, |stdout| {
        decode_and_render(
            BufReader::new(stdout),
            runner,
            options.presentation,
            options.relaxed_validation,
            &mut renderer,
        )
    })?;
    if !process.status.success() {
        return Err(AppError::Render(format!(
            "ffmpeg could not stream the selected TrueHD track ({}): {}",
            process.status,
            String::from_utf8_lossy(&process.stderr).trim()
        )));
    }
    renderer.finish()
}

fn decode_and_render(
    input: impl Read + Send,
    runner: &ProcessRunner,
    presentation: u8,
    relaxed_validation: bool,
    renderer: &mut ObjectRenderer<'_>,
) -> Result<(), AppError> {
    const PIPELINE_FRAMES: usize = 4;
    let (sender, receiver) = sync_channel(PIPELINE_FRAMES);
    thread::scope(|scope| {
        let decoder = scope.spawn(move || {
            decode_stream(input, presentation, relaxed_validation, |frame| {
                runner.check_cancelled()?;
                sender
                    .send(frame)
                    .map_err(|_| AppError::Render("TrueHD render pipeline stopped".into()))
            })
        });

        let mut render_result = Ok(());
        while let Ok(frame) = receiver.recv() {
            if let Err(error) = renderer.push(ObjectPcmFrame {
                sample_rate: frame.sample_rate,
                sample_count: frame.sample_count,
                channel_count: frame.channel_count,
                samples: frame.samples,
                channel_speakers: frame.channel_speakers,
                isf: frame.isf,
                spatial_updates: frame.spatial_updates,
            }) {
                render_result = Err(error);
                break;
            }
        }
        drop(receiver);
        let decode_result = decoder
            .join()
            .map_err(|_| AppError::Render("TrueHD decoder thread panicked".into()))?;
        render_result.and(decode_result)
    })
}
