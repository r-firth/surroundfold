#![cfg(feature = "embedded-truehd")]

use std::{env, ffi::OsString, fs::File, io::BufReader, path::Path};

use clap::Parser;
use surroundfold::{
    cancel::Cancellation, cli::Cli, media::MediaProbe, process::ProcessRunner,
    truehd_adapter::decode_stream,
};

/// Optional real-OAMD regression test. The sample remains outside the repo so
/// large or redistributability-restricted fixtures are never committed.
#[test]
fn external_atmos_fixture_renders_objects() {
    let Some(sample) = env::var_os("SURROUNDFOLD_ATMOS_SAMPLE") else {
        eprintln!("skipping real Atmos fixture; SURROUNDFOLD_ATMOS_SAMPLE is not set");
        return;
    };
    let sample = Path::new(&sample);
    assert!(sample.is_file(), "Atmos fixture does not exist");
    let mut object_updates = 0;
    decode_stream(
        BufReader::new(File::open(sample).unwrap()),
        3,
        false,
        |frame| {
            object_updates += frame
                .spatial_updates
                .iter()
                .map(|update| update.objects.len())
                .sum::<usize>();
            Ok(())
        },
    )
    .unwrap();
    assert!(
        object_updates > 0,
        "fixture contains no decoded OAMD objects"
    );

    let runner = ProcessRunner::new(Cancellation::new());
    let ffmpeg = runner.locate_required("ffmpeg", None).unwrap();
    let ffprobe = runner.locate_required("ffprobe", None).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let output = directory.path().join("output.mkv");
    wrap_truehd(&runner, &ffmpeg, sample, &input);

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
    assert_eq!(manifest.streams[0].codec_name, "truehd");
    assert_eq!(manifest.streams[1].codec_name, "pcm_s16le");
}

fn wrap_truehd(runner: &ProcessRunner, ffmpeg: &Path, sample: &Path, output: &Path) {
    let mut arguments = ["-v", "error", "-f", "truehd", "-i"]
        .map(OsString::from)
        .to_vec();
    arguments.push(sample.as_os_str().to_os_string());
    arguments.extend(["-map", "0:a:0", "-c:a", "copy"].map(OsString::from));
    arguments.push(output.as_os_str().to_os_string());
    let result = runner.run(ffmpeg, &arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
