use std::{
    env,
    ffi::OsString,
    fs::File,
    io::BufReader,
    path::{Path, PathBuf},
};

use clap::Parser;
use surroundfold::{
    cancel::Cancellation, cli::Cli, eac3::FrameReader, joc::JocDecoder, media::MediaProbe,
    oamd::OamdDecoder, process::ProcessRunner,
};

/// Optional public DD+/Atmos fixture. The media remains outside the repository.
#[test]
fn external_eac3_joc_fixture_exposes_oamd_and_joc() {
    let Some(sample) = env::var_os("SURROUNDFOLD_EAC3_JOC_SAMPLE") else {
        eprintln!("skipping DD+/Atmos fixture; SURROUNDFOLD_EAC3_JOC_SAMPLE is not set");
        return;
    };
    let input = File::open(&sample).expect("DD+/Atmos fixture does not exist");
    let mut frames = FrameReader::new(BufReader::new(input));
    let mut frame_count = 0;
    let mut oamd_count = 0;
    let mut joc_count = 0;
    let mut object_updates = 0;
    let mut assignment = None;
    let mut saw_dynamic_objects = false;
    let mut saw_lfe = false;
    let mut oamd = OamdDecoder::new();
    let mut joc = JocDecoder::new();
    while let Some(frame) = frames.next_frame().expect("fixture must be valid E-AC-3") {
        frame_count += 1;
        let mut oamd_objects = None;
        let mut joc_objects = None;
        for payload in frame.payloads {
            match payload.id {
                11 => {
                    oamd_count += 1;
                    let metadata = oamd
                        .decode(&payload)
                        .unwrap_or_else(|error| panic!("OAMD frame {frame_count} failed: {error}"));
                    assert!(metadata.object_count > 0);
                    assert_eq!(
                        metadata.object_count,
                        metadata.joc_object_count + metadata.lfe_object_indices.len()
                    );
                    assert!(metadata.dynamic_object_count <= metadata.joc_object_count);
                    let current_assignment = (
                        metadata.object_count,
                        metadata.joc_object_count,
                        metadata.dynamic_object_count,
                        metadata.lfe_object_indices.clone(),
                    );
                    if let Some(expected) = &assignment {
                        assert_eq!(&current_assignment, expected);
                    } else {
                        assignment = Some(current_assignment);
                    }
                    saw_dynamic_objects |= metadata.dynamic_object_count != 0;
                    saw_lfe |= !metadata.lfe_object_indices.is_empty();
                    object_updates += metadata.updates.len();
                    oamd_objects = Some(metadata.joc_object_count);
                }
                14 => {
                    joc_count += 1;
                    let matrix = joc
                        .decode(&payload, frame.header.sample_count())
                        .unwrap_or_else(|error| panic!("JOC frame {frame_count} failed: {error}"));
                    assert_eq!(matrix.timeslots, 24);
                    assert_eq!(matrix.input_channels, 5);
                    assert!((1.0..=8.75).contains(&matrix.clip_gain));
                    assert!(matrix.coefficients().iter().all(|value| value.is_finite()));
                    joc_objects = Some(matrix.object_count);
                }
                _ => {}
            }
        }
        assert_eq!(oamd_objects, joc_objects);
    }
    eprintln!("frames={frame_count}, OAMD={oamd_count}, JOC={joc_count}, updates={object_updates}");
    assert!(frame_count > 0, "fixture contains no E-AC-3 frames");
    assert!(oamd_count > 0, "fixture contains no OAMD payloads");
    assert!(joc_count > 0, "fixture contains no JOC payloads");
    assert!(object_updates > 0, "fixture contains no object updates");
    assert!(saw_dynamic_objects, "fixture contains no dynamic objects");
    assert!(saw_lfe, "fixture does not exercise the LFE bypass");
}

/// Full native JOC reconstruction, object rendering, and preservation muxing.
#[test]
fn external_eac3_joc_fixture_renders_end_to_end() {
    let Some(sample) = env::var_os("SURROUNDFOLD_EAC3_JOC_SAMPLE") else {
        eprintln!("skipping DD+/Atmos render fixture; SURROUNDFOLD_EAC3_JOC_SAMPLE is not set");
        return;
    };
    let sample = PathBuf::from(sample);
    assert!(sample.is_file(), "DD+/Atmos fixture does not exist");

    let runner = ProcessRunner::new(Cancellation::new());
    let ffmpeg = runner.locate_required("ffmpeg", None).unwrap();
    let ffprobe = runner.locate_required("ffprobe", None).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let output = directory.path().join("output.mkv");
    wrap_eac3(&runner, &ffmpeg, &sample, &input);

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
    assert_eq!(manifest.streams[0].codec_name, "eac3");
    assert_eq!(
        manifest.streams[0].profile.as_deref(),
        Some("Dolby Digital Plus + Dolby Atmos")
    );
    assert_eq!(manifest.streams[1].codec_name, "flac");
    assert_eq!(manifest.streams[1].bits_per_raw_sample, Some(24));
    assert_eq!(manifest.streams[1].channels, Some(2));
    assert_eq!(manifest.streams[1].sample_rate, Some(48_000));

    let pcm = extract_rendered_pcm(&runner, &ffmpeg, &output);
    assert!(!pcm.is_empty());
    assert!(
        pcm.chunks_exact(2)
            .any(|sample| i16::from_le_bytes([sample[0], sample[1]]) != 0),
        "rendered JOC track is silent"
    );
}

fn wrap_eac3(runner: &ProcessRunner, ffmpeg: &Path, sample: &Path, output: &Path) {
    let mut arguments = ["-v", "error", "-f", "eac3", "-i"]
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

fn extract_rendered_pcm(runner: &ProcessRunner, ffmpeg: &Path, input: &Path) -> Vec<u8> {
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
