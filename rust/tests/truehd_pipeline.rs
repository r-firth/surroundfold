#![cfg(feature = "embedded-truehd")]

use std::{
    ffi::OsString,
    fs::{self, File},
    io::BufReader,
    path::Path,
};

use clap::Parser;
use surroundfold::{
    cancel::Cancellation, cli::Cli, media::MediaProbe, process::ProcessRunner,
    truehd_adapter::decode_stream,
};

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
    if !generate_truehd_input(&runner, &ffmpeg, &input, 48_000) {
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
    assert_eq!(appended.codec_name, "flac");
    assert_eq!(appended.bits_per_raw_sample, Some(24));
    assert_eq!(appended.channels, Some(2));
    assert_eq!(appended.sample_rate, Some(48_000));
}

#[test]
fn generated_truehd_pcm_matches_ffmpeg_decode() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping TrueHD PCM comparison because ffmpeg is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    for sample_rate in [48_000, 96_000] {
        let input = directory.path().join(format!("source-{sample_rate}.mkv"));
        let elementary = directory.path().join(format!("source-{sample_rate}.thd"));
        let ffmpeg_pcm = directory.path().join(format!("ffmpeg-{sample_rate}.f32le"));
        if !generate_truehd_input(&runner, &ffmpeg, &input, sample_rate) {
            eprintln!(
                "skipping TrueHD PCM comparison because this FFmpeg lacks its experimental encoder"
            );
            return;
        }

        run_ffmpeg(
            &runner,
            &ffmpeg,
            &[
                OsString::from("-v"),
                OsString::from("error"),
                OsString::from("-i"),
                input.as_os_str().to_os_string(),
                OsString::from("-map"),
                OsString::from("0:a:0"),
                OsString::from("-c:a"),
                OsString::from("copy"),
                OsString::from("-f"),
                OsString::from("truehd"),
                elementary.as_os_str().to_os_string(),
            ],
        );
        run_ffmpeg(
            &runner,
            &ffmpeg,
            &[
                OsString::from("-v"),
                OsString::from("error"),
                OsString::from("-i"),
                input.as_os_str().to_os_string(),
                OsString::from("-map"),
                OsString::from("0:a:0"),
                OsString::from("-c:a"),
                OsString::from("pcm_f32le"),
                OsString::from("-f"),
                OsString::from("f32le"),
                ffmpeg_pcm.as_os_str().to_os_string(),
            ],
        );

        let expected = fs::read(ffmpeg_pcm)
            .unwrap()
            .chunks_exact(4)
            .map(|sample| f32::from_le_bytes(sample.try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut actual = Vec::new();
        decode_stream(
            BufReader::new(File::open(elementary).unwrap()),
            3,
            false,
            |frame| {
                actual.extend(frame.samples);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(actual.len(), expected.len());
        let maximum_error = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        assert!(
            maximum_error <= 1.0 / 8_388_608.0,
            "embedded TrueHD PCM at {sample_rate} Hz differs from FFmpeg by {maximum_error}"
        );
    }
}

#[test]
fn generated_96khz_truehd_preserves_render_rate() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping 96 kHz TrueHD test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping 96 kHz TrueHD test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source-96khz.mkv");
    let hrir = directory.path().join("hrir.wav");
    let output = directory.path().join("output-96khz.mkv");
    if !generate_truehd_input(&runner, &ffmpeg, &input, 96_000) {
        eprintln!("skipping 96 kHz TrueHD test because this FFmpeg lacks its experimental encoder");
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
    assert_eq!(manifest.streams[0].codec_name, "truehd");
    assert_eq!(manifest.streams[0].sample_rate, Some(96_000));
    assert_eq!(manifest.streams[1].codec_name, "flac");
    assert_eq!(manifest.streams[1].bits_per_raw_sample, Some(24));
    assert_eq!(manifest.streams[1].sample_rate, Some(96_000));
}

fn run_ffmpeg(runner: &ProcessRunner, ffmpeg: &Path, arguments: &[OsString]) {
    let result = runner.run(ffmpeg, arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn generate_truehd_input(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    output: &Path,
    sample_rate: u32,
) -> bool {
    let mut arguments = vec![
        OsString::from("-v"),
        OsString::from("error"),
        OsString::from("-f"),
        OsString::from("lavfi"),
        OsString::from("-i"),
        OsString::from(format!(
            "sine=frequency=997:sample_rate={sample_rate}:duration=0.1"
        )),
    ];
    arguments.extend(
        [
            "-filter_complex",
            "pan=5.1|FL=c0|FR=0.8*c0|FC=0.6*c0|LFE=0*c0|BL=0.4*c0|BR=0.2*c0",
            "-c:a",
            "truehd",
            "-strict",
            "-2",
        ]
        .map(OsString::from),
    );
    arguments.push(output.as_os_str().to_os_string());
    runner
        .run(ffmpeg, &arguments)
        .is_ok_and(|result| result.status.success())
}
