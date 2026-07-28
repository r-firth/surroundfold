#![cfg(feature = "embedded-truehd")]

use std::{ffi::OsString, path::Path};

use clap::Parser;
use surroundfold::{cancel::Cancellation, cli::Cli, media::MediaProbe, process::ProcessRunner};

mod common;

use common::generate_hrir;

#[test]
fn generated_truehd_is_decoded_in_process_and_muxed() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping TrueHD render test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping TrueHD render test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let hrir = directory.path().join("hrir.wav");
    let output = directory.path().join("output.mkv");
    if !generate_truehd_input(&runner, &ffmpeg, &input) {
        eprintln!("skipping TrueHD render test because this FFmpeg lacks its experimental encoder");
        return;
    }
    generate_hrir(&hrir);

    let cli = Cli::try_parse_from([
        OsString::from("surroundfold"),
        input.as_os_str().to_os_string(),
        OsString::from("--hrir"),
        hrir.as_os_str().to_os_string(),
        OsString::from("--output"),
        output.as_os_str().to_os_string(),
        OsString::from("--progress"),
        OsString::from("quiet"),
    ])
    .unwrap();
    surroundfold::run(&cli, &Cancellation::new()).unwrap();

    let manifest = MediaProbe::new(&runner, ffprobe).probe(&output).unwrap();
    assert_eq!(manifest.streams.len(), 2);
    assert_eq!(manifest.streams[0].codec_name, "truehd");
    let appended = &manifest.streams[1];
    assert_eq!(appended.codec_name, "pcm_s16le");
    assert_eq!(appended.channels, Some(2));
    assert_eq!(appended.sample_rate, Some(48_000));
}

fn generate_truehd_input(runner: &ProcessRunner, ffmpeg: &Path, output: &Path) -> bool {
    let mut arguments = [
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=997:sample_rate=48000:duration=0.1",
        "-filter_complex",
        "pan=5.1|FL=c0|FR=0.8*c0|FC=0.6*c0|LFE=0*c0|BL=0.4*c0|BR=0.2*c0",
        "-c:a",
        "truehd",
        "-strict",
        "-2",
    ]
    .map(OsString::from)
    .to_vec();
    arguments.push(output.as_os_str().to_os_string());
    runner
        .run(ffmpeg, &arguments)
        .is_ok_and(|result| result.status.success())
}
