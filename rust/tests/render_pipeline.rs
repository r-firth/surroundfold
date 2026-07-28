use std::{ffi::OsString, path::Path};

use clap::Parser;
use surroundfold::{
    cancel::Cancellation, cli::Cli, media::MediaProbe, mux::BINAURAL_TITLE, process::ProcessRunner,
};

mod common;

use common::{generate_height_hrir, generate_hrir};

#[test]
fn default_channel_render_produces_a_verified_matroska() {
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
    assert_eq!(appended.codec_name, "pcm_s16le");
    assert_eq!(appended.channels, Some(2));
    assert_eq!(appended.sample_rate, Some(48_000));
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
                && stream.codec_name == "pcm_s16le"
                && stream.tag("language") == Some("spa")
                && stream.tag("title") != Some(BINAURAL_TITLE)
        })
        .unwrap();
    let appended = manifest
        .streams
        .iter()
        .find(|stream| stream.tag("title") == Some(BINAURAL_TITLE))
        .unwrap();
    assert_eq!(selected.start_time, appended.start_time);
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
        "pcm_s16le",
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
