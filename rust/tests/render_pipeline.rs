use std::{ffi::OsString, path::Path};

use clap::Parser;
use surroundfold::{
    cancel::Cancellation, cli::Cli, media::MediaProbe, mux::BINAURAL_TITLE, process::ProcessRunner,
};

mod common;

use common::{generate_height_discrimination_hrir, generate_height_hrir, generate_hrir};

#[test]
fn default_channel_render_produces_lossless_24_bit_flac() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping render test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping render test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let output = directory.path().join("output.mkv");
    generate_input(&runner, &ffmpeg, &input);

    let cli = Cli::try_parse_from([
        OsString::from("surroundfold"),
        input.as_os_str().to_os_string(),
        OsString::from("--output"),
        output.as_os_str().to_os_string(),
        OsString::from("--progress"),
        OsString::from("quiet"),
    ])
    .unwrap();
    surroundfold::run(&cli, &Cancellation::new()).unwrap();

    let manifest = MediaProbe::new(&runner, ffprobe).probe(&output).unwrap();
    assert_eq!(manifest.streams.len(), 2);
    let appended = &manifest.streams[1];
    assert_eq!(appended.codec_name, "flac");
    assert_eq!(appended.bits_per_raw_sample, Some(24));
    assert_eq!(appended.channels, Some(2));
    assert_eq!(appended.sample_rate, Some(48_000));
}

#[test]
fn explicit_aac_compatibility_render_produces_stereo_lc() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping render test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping render test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let output = directory.path().join("output.mkv");
    generate_input(&runner, &ffmpeg, &input);

    let cli = Cli::try_parse_from([
        OsString::from("surroundfold"),
        input.as_os_str().to_os_string(),
        OsString::from("--output"),
        output.as_os_str().to_os_string(),
        OsString::from("--output-codec"),
        OsString::from("aac"),
        OsString::from("--progress"),
        OsString::from("quiet"),
    ])
    .unwrap();
    surroundfold::run(&cli, &Cancellation::new()).unwrap();

    let manifest = MediaProbe::new(&runner, ffprobe).probe(&output).unwrap();
    let appended = &manifest.streams[1];
    assert_eq!(appended.codec_name, "aac");
    assert_eq!(appended.profile.as_deref(), Some("LC"));
    assert_eq!(appended.channels, Some(2));
    assert_eq!(appended.sample_rate, Some(48_000));
    assert_eq!(appended.disposition.get("default"), Some(&0));
}

#[test]
fn default_operation_replaces_input_and_appends_one_finished_track() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping in-place test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping in-place test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    generate_input(&runner, &ffmpeg, &input);
    let original_size = input.metadata().unwrap().len();
    let original_manifest = MediaProbe::new(&runner, ffprobe.clone())
        .probe(&input)
        .unwrap();
    let original_default = original_manifest.streams[0]
        .disposition
        .get("default")
        .copied()
        .unwrap_or(0);
    let cli = Cli::try_parse_from([
        OsString::from("surroundfold"),
        input.as_os_str().to_os_string(),
        OsString::from("--progress"),
        OsString::from("quiet"),
    ])
    .unwrap();

    surroundfold::run(&cli, &Cancellation::new()).unwrap();
    surroundfold::run(&cli, &Cancellation::new()).unwrap();

    let manifest = MediaProbe::new(&runner, ffprobe).probe(&input).unwrap();
    assert_eq!(manifest.streams.len(), 2);
    assert!(input.metadata().unwrap().len() > original_size);
    let original = &manifest.streams[0];
    assert_eq!(
        original.disposition.get("default").copied().unwrap_or(0),
        original_default
    );
    let appended = manifest
        .streams
        .iter()
        .filter(|stream| stream.tag("title") == Some(BINAURAL_TITLE))
        .collect::<Vec<_>>();
    assert_eq!(appended.len(), 1);
    assert_eq!(appended[0].codec_name, "flac");
    assert_eq!(appended[0].bits_per_raw_sample, Some(24));
    assert_eq!(appended[0].channels, Some(2));
    assert_eq!(appended[0].disposition.get("default"), Some(&0));
}

#[test]
fn matrix_and_height_controls_run_through_the_native_pipeline() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping render test because ffmpeg is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let hrir = directory.path().join("height-hrir.wav");
    let baseline = directory.path().join("baseline.mkv");
    let enhanced = directory.path().join("enhanced.mkv");
    generate_stereo_input(&runner, &ffmpeg, &input);
    generate_height_hrir(&hrir);

    run_with_options(&input, &hrir, &baseline, &[]);
    run_with_options(
        &input,
        &hrir,
        &enhanced,
        &[
            ("--matrix", "on"),
            ("--upconvert", "on"),
            ("--effect", "42"),
            ("--smoothness", "33"),
            ("--surround-swap", "on"),
            ("--speaker-virtualizer", "on"),
        ],
    );

    let baseline_pcm = extract_appended_pcm(&runner, &ffmpeg, &baseline);
    let enhanced_pcm = extract_appended_pcm(&runner, &ffmpeg, &enhanced);
    assert!(!baseline_pcm.is_empty());
    assert_eq!(baseline_pcm.len(), enhanced_pcm.len());
    assert_ne!(baseline_pcm, enhanced_pcm);
}

#[test]
fn named_height_bed_reaches_the_height_hrir_route() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping height-bed render test because ffmpeg is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("height-source.mkv");
    let hrir = directory.path().join("height-hrir.wav");
    let output = directory.path().join("height-output.mkv");
    generate_height_only_input(&runner, &ffmpeg, &input);
    generate_height_discrimination_hrir(&hrir);
    run_with_options(&input, &hrir, &output, &[]);

    let pcm = extract_appended_pcm(&runner, &ffmpeg, &output);
    let left_peak = pcm
        .chunks_exact(4)
        .map(|frame| i16::from_le_bytes([frame[0], frame[1]]).unsigned_abs())
        .max()
        .unwrap();
    let right_peak = pcm
        .chunks_exact(4)
        .map(|frame| i16::from_le_bytes([frame[2], frame[3]]).unsigned_abs())
        .max()
        .unwrap();
    assert!(
        left_peak > right_peak + right_peak / 6,
        "5.1.2 top-front-left did not retain the expected parametric left bias: left={left_peak}, right={right_peak}"
    );
}

#[test]
fn delayed_selected_audio_and_binaural_track_start_together() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping render test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping render test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("delayed-source.mkv");
    let hrir = directory.path().join("hrir.wav");
    let output = directory.path().join("delayed-output.mkv");
    generate_delayed_input(&runner, &ffmpeg, &input);
    generate_hrir(&hrir);
    run_with_options(&input, &hrir, &output, &[("--language", "spa")]);

    let manifest = MediaProbe::new(&runner, ffprobe).probe(&output).unwrap();
    let selected = manifest
        .streams
        .iter()
        .find(|stream| {
            stream.codec_type == "audio"
                && stream.codec_name == "pcm_s24le"
                && stream.tag("language") == Some("spa")
                && stream.tag("title") != Some(BINAURAL_TITLE)
        })
        .unwrap();
    let appended = manifest
        .streams
        .iter()
        .find(|stream| stream.tag("title") == Some(BINAURAL_TITLE))
        .unwrap();
    let selected_start = selected.start_time.unwrap();
    let appended_presentation_start =
        appended.start_time.unwrap() + f64::from(appended.initial_padding.unwrap_or(0)) / 48_000.0;
    assert!(
        (selected_start - appended_presentation_start).abs() <= 0.001,
        "selected audio starts at {selected_start}, appended presentation starts at {appended_presentation_start}"
    );
    assert_eq!(appended.tag("language"), Some("spa"));
}

fn generate_input(runner: &ProcessRunner, ffmpeg: &Path, output: &Path) {
    let mut arguments = [
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=997:sample_rate=48000:duration=0.1",
        "-c:a",
        "flac",
    ]
    .map(OsString::from)
    .to_vec();
    arguments.push(output.as_os_str().to_os_string());
    let result = runner.run(ffmpeg, &arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn generate_stereo_input(runner: &ProcessRunner, ffmpeg: &Path, output: &Path) {
    let mut arguments = [
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=997:sample_rate=48000:duration=0.1",
        "-filter_complex",
        "pan=stereo|FL=c0|FR=0.6*c0",
        "-c:a",
        "flac",
    ]
    .map(OsString::from)
    .to_vec();
    arguments.push(output.as_os_str().to_os_string());
    let result = runner.run(ffmpeg, &arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn generate_height_only_input(runner: &ProcessRunner, ffmpeg: &Path, output: &Path) {
    let mut arguments = [
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "aevalsrc=0|0|0|0|0|0|0.05*sin(2*PI*997*t)|0:s=48000:d=0.1:c=5.1.2",
        "-c:a",
        "flac",
    ]
    .map(OsString::from)
    .to_vec();
    arguments.push(output.as_os_str().to_os_string());
    let result = runner.run(ffmpeg, &arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn generate_delayed_input(runner: &ProcessRunner, ffmpeg: &Path, output: &Path) {
    let mut arguments = [
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x54:rate=10:duration=1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=330:sample_rate=48000:duration=1",
        "-itsoffset",
        "0.125",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=660:sample_rate=48000:duration=0.75",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:a",
        "-c:v",
        "ffv1",
        "-c:a:0",
        "eac3",
        "-b:a:0",
        "192k",
        "-c:a:1",
        "pcm_s24le",
        "-metadata:s:a:0",
        "language=eng",
        "-metadata:s:a:1",
        "language=spa",
        "-metadata:s:a:1",
        "title=Delayed PCM",
    ]
    .map(OsString::from)
    .to_vec();
    arguments.push(output.as_os_str().to_os_string());
    let result = runner.run(ffmpeg, arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn run_with_options(input: &Path, hrir: &Path, output: &Path, options: &[(&str, &str)]) {
    let mut arguments = vec![
        OsString::from("surroundfold"),
        input.as_os_str().to_os_string(),
        OsString::from("--hrir"),
        hrir.as_os_str().to_os_string(),
        OsString::from("--output"),
        output.as_os_str().to_os_string(),
        OsString::from("--progress"),
        OsString::from("quiet"),
    ];
    for (option, value) in options {
        arguments.push(OsString::from(option));
        arguments.push(OsString::from(value));
    }
    let cli = Cli::try_parse_from(arguments).unwrap();
    surroundfold::run(&cli, &Cancellation::new()).unwrap();
}

fn extract_appended_pcm(runner: &ProcessRunner, ffmpeg: &Path, input: &Path) -> Vec<u8> {
    let arguments = [
        OsString::from("-v"),
        OsString::from("error"),
        OsString::from("-i"),
        input.as_os_str().to_os_string(),
        OsString::from("-map"),
        OsString::from("0:a:1"),
        OsString::from("-f"),
        OsString::from("s16le"),
        OsString::from("-"),
    ];
    let result = runner.run(ffmpeg, arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}
