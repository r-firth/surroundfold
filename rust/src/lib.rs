pub mod binaural;
pub mod cancel;
pub mod cli;
pub mod dsp;
pub mod eac3;
pub mod eac3_render;
pub mod error;
pub mod hrir;
mod isf;
mod isf_tables;
pub mod joc;
mod joc_tables;
pub mod media;
pub mod mux;
pub mod oamd;
pub mod object;
mod object_render;
pub mod paths;
pub mod process;
pub mod qmf;
mod qmf_tables;
pub mod render;
pub mod room;
pub mod selection;
pub(crate) mod spatial;
mod stream_io;
#[cfg(feature = "embedded-truehd")]
pub mod truehd_adapter;
#[cfg(feature = "embedded-truehd")]
pub mod truehd_render;
pub(crate) mod upmix;
pub mod workspace;

use std::io::{self, Write};

use cancel::Cancellation;
use cli::{Cli, ProgressMode, Toggle};
use eac3_render::{Eac3RenderOptions, render_eac3_track};
use error::AppError;
use hrir::HrirSet;
use media::MediaProbe;
use mux::{advanced_argument_pairs, build_mux_arguments, mux, verify_output};
use paths::ResolvedPaths;
use process::ProcessRunner;
use render::{ChannelRenderOptions, render_channel_track};
use room::RoomCorrection;
use selection::{AudioTrack, DecodeCapability, select_track};
#[cfg(feature = "embedded-truehd")]
use truehd_render::{TrueHdRenderOptions, render_truehd_track};
use workspace::{AtomicOutput, Workspace};

/// Executes one CLI invocation.
///
/// # Errors
///
/// Returns a categorized [`AppError`] when validation, probing, selection, or
/// rendering fails.
#[allow(clippy::too_many_lines)] // Keeping the ordered transactional pipeline in one place aids auditing.
pub fn run(cli: &Cli, cancellation: &Cancellation) -> Result<(), AppError> {
    let advanced = if cli.list_tracks {
        Vec::new()
    } else {
        advanced_argument_pairs(&cli.ffmpeg_arg)?
    };
    let paths = ResolvedPaths::from_cli(cli)?;
    let runner = ProcessRunner::new(cancellation.clone());
    let ffprobe = runner.locate_required("ffprobe", cli.ffprobe.as_deref())?;
    let ffprobe_version = runner.check_version(&ffprobe)?;

    let probe = MediaProbe::new(&runner, ffprobe);
    let manifest = probe.probe(&paths.input)?;
    let tracks = AudioTrack::from_manifest(&manifest);

    if cli.list_tracks {
        print_tracks(&tracks, cli.progress)?;
        return Ok(());
    }

    let selected = select_track(&tracks, cli.track, cli.language.as_deref())?;
    validate_implemented_options(cli, selected.capability)?;
    let selected_stream = manifest
        .streams
        .iter()
        .find(|stream| stream.index == selected.stream_index)
        .ok_or_else(|| {
            AppError::UnsupportedInput("selected stream disappeared from the manifest".into())
        })?;
    let ffmpeg = runner.locate_required("ffmpeg", cli.ffmpeg.as_deref())?;
    let ffmpeg_version = runner.check_version(&ffmpeg)?;
    let output_path = paths
        .output
        .as_deref()
        .ok_or_else(|| AppError::Usage("--output could not be resolved".into()))?;
    let hrir = paths
        .hrir
        .as_deref()
        .map_or_else(HrirSet::load_default, HrirSet::load_concatenated_wave)?;
    let room_correction = paths
        .room_correction
        .as_deref()
        .map(|path| RoomCorrection::load(path, hrir.sample_rate))
        .transpose()?;
    let workspace = Workspace::new(cli.keep_temp_files)?;
    let rendered = workspace.file("binaural.wav")?;

    report(
        cli.progress,
        &format!(
            "selected audio {} (stream {}, {}, {}, {})",
            selected.audio_index,
            selected.stream_index,
            selected.codec,
            selected.language.as_deref().unwrap_or("und"),
            selected.capability
        ),
    );
    report(cli.progress, "rendering binaural track");
    let render = match selected.capability {
        DecodeCapability::Channels => {
            let decoded = workspace.file("selected.f32le")?;
            render_channel_track(
                &runner,
                &ffmpeg,
                &paths.input,
                selected_stream,
                &hrir,
                room_correction.as_ref(),
                ChannelRenderOptions {
                    gain_db: cli.gain_db,
                    surround_swap: cli.surround_swap.enabled(),
                    matrix: cli.matrix.enabled(),
                    upconvert: cli.upconvert.enabled(),
                    effect: cli.effect * 0.01,
                    smoothness: cli.smoothness * 0.01,
                    mute_bed: cli.mute_bed.enabled(),
                    mute_ground: cli.mute_ground.enabled(),
                    speaker_virtualizer: cli.speaker_virtualizer.enabled(),
                },
                &decoded,
                &rendered,
            )?
        }
        DecodeCapability::TrueHdObjects => {
            #[cfg(feature = "embedded-truehd")]
            {
                let elementary = workspace.file("selected.thd")?;
                render_truehd_track(
                    &runner,
                    &ffmpeg,
                    &paths.input,
                    selected_stream.index,
                    &hrir,
                    room_correction.as_ref(),
                    TrueHdRenderOptions {
                        presentation: cli.mlp_presentation.unwrap_or(3),
                        relaxed_validation: cli.unsafe_parsing,
                        gain_db: cli.gain_db,
                        surround_swap: cli.surround_swap.enabled(),
                        mute_bed: cli.mute_bed.enabled(),
                        mute_ground: cli.mute_ground.enabled(),
                        speaker_virtualizer: cli.speaker_virtualizer.enabled(),
                    },
                    &elementary,
                    &rendered,
                )?
            }
            #[cfg(not(feature = "embedded-truehd"))]
            {
                return Err(AppError::Dependency(
                    "this build does not include embedded TrueHD decoding".into(),
                ));
            }
        }
        DecodeCapability::JocObjects => {
            let elementary = workspace.file("selected.ec3")?;
            let decoded = workspace.file("selected.f32le")?;
            render_eac3_track(
                &runner,
                &ffmpeg,
                &paths.input,
                selected_stream,
                &hrir,
                room_correction.as_ref(),
                Eac3RenderOptions {
                    gain_db: cli.gain_db,
                    surround_swap: cli.surround_swap.enabled(),
                    mute_bed: cli.mute_bed.enabled(),
                    mute_ground: cli.mute_ground.enabled(),
                    speaker_virtualizer: cli.speaker_virtualizer.enabled(),
                },
                &elementary,
                &decoded,
                &rendered,
            )?
        }
        DecodeCapability::Unsupported => {
            return Err(AppError::UnsupportedInput(
                "selected audio codec is not renderable".into(),
            ));
        }
    };

    let atomic = AtomicOutput::new(output_path, cli.overwrite)?;
    let selected_start = selected_stream.start_time.unwrap_or(0.0);
    let mux_arguments = build_mux_arguments(
        &paths.input,
        &rendered,
        atomic.partial_path(),
        &manifest,
        selected_stream,
        selected_start,
        &advanced,
    )?;
    report(cli.progress, "muxing preservation copy");
    mux(&runner, &ffmpeg, &mux_arguments)?;
    report(cli.progress, "verifying preservation and synchronization");
    let output_manifest = probe.probe(atomic.partial_path())?;
    verify_output(
        &manifest,
        &output_manifest,
        selected_stream,
        render.sample_rate,
        render.frames,
        selected_start,
    )?;
    atomic.commit()?;

    match cli.progress {
        ProgressMode::Json => {
            let result = serde_json::json!({
                "input": paths.input,
                "output": output_path,
                "selectedTrack": selected,
                "sampleRate": render.sample_rate,
                "renderedSamples": render.frames,
                "peakBeforeLimiting": render.peak_before_limiting,
                "tools": {
                    "ffmpeg": ffmpeg_version,
                    "ffprobe": ffprobe_version,
                },
                "temporaryDirectory": cli.keep_temp_files.then(|| workspace.path()),
            });
            serde_json::to_writer(io::stdout().lock(), &result)?;
            println!();
        }
        ProgressMode::Text | ProgressMode::Quiet => println!("{}", output_path.display()),
    }
    Ok(())
}

fn validate_implemented_options(cli: &Cli, capability: DecodeCapability) -> Result<(), AppError> {
    let mut unsupported = Vec::new();
    if matches!(
        capability,
        DecodeCapability::TrueHdObjects | DecodeCapability::JocObjects
    ) {
        unsupported.extend(
            [
                (cli.upconvert == Toggle::On, "--upconvert"),
                (cli.matrix == Toggle::On, "--matrix"),
            ]
            .into_iter()
            .filter_map(|(enabled, name)| enabled.then_some(name)),
        );
    }
    if capability != DecodeCapability::TrueHdObjects {
        unsupported.extend(
            [(cli.mlp_presentation.is_some(), "--mlp-presentation")]
                .into_iter()
                .filter_map(|(enabled, name)| enabled.then_some(name)),
        );
    }
    if unsupported.is_empty() {
        Ok(())
    } else {
        Err(AppError::Usage(format!(
            "these options are not implemented by the Rust renderer yet: {}",
            unsupported.join(", ")
        )))
    }
}

fn report(mode: ProgressMode, message: &str) {
    if mode == ProgressMode::Text {
        eprintln!("{message}");
    }
}

fn print_tracks(tracks: &[AudioTrack], mode: ProgressMode) -> Result<(), AppError> {
    let stdout = io::stdout();
    let mut output = stdout.lock();
    if mode == ProgressMode::Json {
        serde_json::to_writer_pretty(&mut output, tracks)?;
        writeln!(output)?;
        return Ok(());
    }

    if tracks.is_empty() {
        writeln!(output, "No audio tracks.")?;
        return Ok(());
    }

    writeln!(
        output,
        "audio  stream  codec       language  channels  capability"
    )?;
    for track in tracks {
        writeln!(
            output,
            "{:>5}  {:>6}  {:<10}  {:<8}  {:>8}  {}",
            track.audio_index,
            track.stream_index,
            track.codec,
            track.language.as_deref().unwrap_or("und"),
            track
                .channels
                .map_or_else(|| "?".into(), |value| value.to_string()),
            track.capability
        )?;
    }
    Ok(())
}
