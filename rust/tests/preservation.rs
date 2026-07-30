use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use surroundfold::{
    cancel::Cancellation,
    cli::{Cli, OutputCodec},
    media::MediaProbe,
    mux::{AppendedTrack, BINAURAL_TITLE, build_mux_arguments, mux, verify_output},
    process::ProcessRunner,
    selection::{AudioTrack, select_track},
    workspace::AtomicOutput,
};

mod common;

use common::generate_hrir;

#[test]
#[allow(clippy::too_many_lines)] // Keep the complete preservation workflow visible in one test.
fn generated_matroska_is_preserved_and_receives_sample_exact_flac() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping preservation test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping preservation test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let rendered = directory.path().join("render.wav");
    let final_output = directory.path().join("result.mkv");

    run_ffmpeg(
        &runner,
        &ffmpeg,
        [
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.1",
            "-c:a",
            "flac",
            "-metadata",
            "title=Rust preservation fixture",
            "-metadata:s:a:0",
            "language=eng",
        ],
        &input,
    );
    run_ffmpeg(
        &runner,
        &ffmpeg,
        [
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.05*sin(2*PI*60*t)+0.04*sin(2*PI*240*t)+0.03*sin(2*PI*8000*t)|0.04*sin(2*PI*60*t)+0.03*sin(2*PI*240*t)+0.05*sin(2*PI*8000*t):s=48000:d=1",
            "-c:a",
            "pcm_s24le",
        ],
        &rendered,
    );
    let probe = MediaProbe::new(&runner, ffprobe.clone());
    let source = probe.probe(&input).unwrap();
    let tracks = AudioTrack::from_manifest(&source);
    let selected = select_track(&tracks, None, None).unwrap();
    let selected_stream = source
        .streams
        .iter()
        .find(|stream| stream.index == selected.stream_index)
        .unwrap();
    let atomic = AtomicOutput::new(&final_output, false).unwrap();
    let appended = [AppendedTrack {
        path: &rendered,
        title: BINAURAL_TITLE,
        codec: OutputCodec::Flac,
        sample_rate: 48_000,
        frames: 48_000,
    }];
    let arguments = build_mux_arguments(
        &input,
        &appended,
        atomic.partial_path(),
        &source,
        selected_stream,
        0.0,
        &[],
    )
    .unwrap();
    mux(&runner, &ffmpeg, &arguments).unwrap();
    let output = probe.probe(atomic.partial_path()).unwrap();
    verify_output(&source, &output, selected_stream, &appended, 0.0).unwrap();
    assert_eq!(
        decoded_s24le(&runner, &ffmpeg, &rendered, "0:a:0"),
        decoded_s24le(&runner, &ffmpeg, atomic.partial_path(), "0:a:1"),
        "FLAC delivery did not decode to the exact finished 24-bit samples"
    );
    atomic.commit().unwrap();

    assert!(input.is_file());
    assert!(final_output.is_file());
}

#[test]
#[allow(clippy::too_many_lines)] // The reference encode makes compatibility settings auditable.
fn aac_compatibility_delivery_uses_the_approved_fast_encode() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping preservation test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping preservation test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let rendered = directory.path().join("render.wav");
    let approved = directory.path().join("approved.mka");
    let output = directory.path().join("result.mkv");

    run_ffmpeg(
        &runner,
        &ffmpeg,
        [
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=0.1",
            "-c:a",
            "flac",
            "-metadata:s:a:0",
            "language=eng",
        ],
        &input,
    );
    run_ffmpeg(
        &runner,
        &ffmpeg,
        [
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.05*sin(2*PI*997*t)|0.04*sin(2*PI*997*t):s=48000:d=1",
            "-c:a",
            "pcm_s24le",
        ],
        &rendered,
    );
    run_ffmpeg(
        &runner,
        &ffmpeg,
        [
            "-v",
            "error",
            "-i",
            rendered.to_str().unwrap(),
            "-c:a",
            "aac",
            "-b:a",
            "320k",
            "-aac_coder",
            "fast",
            "-f",
            "matroska",
        ],
        &approved,
    );

    let probe = MediaProbe::new(&runner, ffprobe.clone());
    let source = probe.probe(&input).unwrap();
    let selected_stream = source
        .streams
        .iter()
        .find(|stream| stream.codec_type == "audio")
        .unwrap();
    let appended = [AppendedTrack {
        path: &rendered,
        title: BINAURAL_TITLE,
        codec: OutputCodec::Aac,
        sample_rate: 48_000,
        frames: 48_000,
    }];
    let arguments = build_mux_arguments(
        &input,
        &appended,
        &output,
        &source,
        selected_stream,
        0.0,
        &[],
    )
    .unwrap();
    mux(&runner, &ffmpeg, &arguments).unwrap();
    let rendered_manifest = probe.probe(&output).unwrap();
    verify_output(&source, &rendered_manifest, selected_stream, &appended, 0.0).unwrap();
    assert_eq!(
        packet_hashes(&runner, &ffprobe, &approved).get(&0),
        packet_hashes(&runner, &ffprobe, &output).get(&1),
        "compatibility mux changed the approved 320 kb/s fast AAC payloads"
    );
}

#[test]
#[allow(clippy::too_many_lines)] // Keep the complete sparse-PGS reproduction visible in one test.
fn sparse_pgs_streams_do_not_starve_appended_audio_packets() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping PGS interleave test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping PGS interleave test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let subtitle = directory.path().join("late.sup");
    let input = directory.path().join("source.mkv");
    let rendered = directory.path().join("render.wav");
    let output = directory.path().join("result.mkv");
    generate_sparse_pgs(&subtitle);

    let mut source_arguments = [
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=24:duration=12",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=12",
        "-f",
        "sup",
        "-i",
    ]
    .map(OsString::from)
    .to_vec();
    source_arguments.push(subtitle.as_os_str().to_os_string());
    source_arguments.extend(["-map", "0:v", "-map", "1:a"].map(OsString::from));
    for _ in 0..16 {
        source_arguments.extend(["-map", "2:s:0"].map(OsString::from));
    }
    source_arguments.extend(
        [
            "-c:v",
            "mpeg2video",
            "-b:v",
            "8M",
            "-minrate",
            "8M",
            "-maxrate",
            "8M",
            "-bufsize",
            "8M",
            "-c:a",
            "flac",
            "-c:s",
            "copy",
        ]
        .map(OsString::from),
    );
    source_arguments.push(input.as_os_str().to_os_string());
    let result = runner.run(&ffmpeg, source_arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    run_ffmpeg(
        &runner,
        &ffmpeg,
        [
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=48000:cl=stereo:d=12",
            "-c:a",
            "pcm_s24le",
        ],
        &rendered,
    );

    let probe = MediaProbe::new(&runner, ffprobe.clone());
    let source = probe.probe(&input).unwrap();
    assert_eq!(
        source
            .streams
            .iter()
            .filter(|stream| stream.codec_name == "hdmv_pgs_subtitle")
            .count(),
        16
    );
    let selected_stream = source
        .streams
        .iter()
        .find(|stream| stream.codec_type == "audio")
        .unwrap();
    let appended = [AppendedTrack {
        path: &rendered,
        title: BINAURAL_TITLE,
        codec: OutputCodec::Flac,
        sample_rate: 48_000,
        frames: 576_000,
    }];
    let arguments = build_mux_arguments(
        &input,
        &appended,
        &output,
        &source,
        selected_stream,
        0.0,
        &[],
    )
    .unwrap();
    mux(&runner, &ffmpeg, &arguments).unwrap();
    let rendered = probe.probe(&output).unwrap();
    verify_output(&source, &rendered, selected_stream, &appended, 0.0).unwrap();

    let packets = audio_packet_facts(&runner, &ffprobe, &output, "a:1");
    assert!(packets.len() > 100);
    let first_position = packets.first().unwrap().position;
    assert!(
        first_position < 4 * 1024 * 1024,
        "first appended FLAC packet is physically {first_position} bytes into the file"
    );
    assert!(
        packets.iter().all(|packet| packet
            .duration
            .is_some_and(|duration| duration > 0.0 && duration <= 0.1)),
        "appended FLAC contains an unexpectedly long or invalid packet"
    );
    let largest_gap = packets
        .windows(2)
        .map(|pair| pair[0].position.abs_diff(pair[1].position))
        .max()
        .unwrap();
    assert!(
        largest_gap < 512 * 1024,
        "appended FLAC has a {largest_gap}-byte physical packet gap"
    );
}

#[test]
fn full_render_preserves_packets_chapters_subtitles_and_attachments() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping preservation test because ffmpeg is unavailable");
        return;
    };
    let Ok(ffprobe) = runner.locate_required("ffprobe", None) else {
        eprintln!("skipping preservation test because ffprobe is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.mkv");
    let hrir = directory.path().join("hrir.wav");
    let output = directory.path().join("output.mkv");
    let attachment = generate_complex_input(&runner, &ffmpeg, directory.path(), &input);
    let original_bytes = fs::read(&input).unwrap();
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

    assert_eq!(fs::read(&input).unwrap(), original_bytes);
    let probe = MediaProbe::new(&runner, ffprobe.clone());
    let source = probe.probe(&input).unwrap();
    let rendered = probe.probe(&output).unwrap();
    assert_eq!(rendered.streams.len(), source.streams.len() + 1);
    assert_eq!(rendered.chapters, source.chapters);
    assert_eq!(
        rendered
            .streams
            .iter()
            .filter(|stream| stream.codec_type == "attachment")
            .count(),
        1
    );

    let source_packets = packet_hashes(&runner, &ffprobe, &input);
    let output_packets = packet_hashes(&runner, &ffprobe, &output);
    let mut ordinals = BTreeMap::<String, usize>::new();
    for stream in &source.streams {
        let ordinal = ordinals.entry(stream.codec_type.clone()).or_default();
        let preserved = rendered
            .streams
            .iter()
            .filter(|candidate| candidate.codec_type == stream.codec_type)
            .nth(*ordinal)
            .unwrap();
        *ordinal += 1;
        if let Some(expected) = source_packets.get(&stream.index) {
            assert_eq!(
                output_packets.get(&preserved.index),
                Some(expected),
                "packet payload changed for source stream {}",
                stream.index
            );
        }
    }

    let extracted = directory.path().join("extracted-attachment.bin");
    extract_attachment(&runner, &ffmpeg, &output, &extracted);
    assert_eq!(fs::read(extracted).unwrap(), fs::read(attachment).unwrap());
}

#[allow(clippy::too_many_lines)] // Keep the complete FFmpeg fixture command readable in one place.
fn generate_complex_input(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    directory: &Path,
    output: &Path,
) -> PathBuf {
    let subtitle = directory.join("subtitle.srt");
    let chapters = directory.join("chapters.ffmeta");
    let attachment = directory.join("attachment.txt");
    fs::write(
        &subtitle,
        "1\n00:00:00,050 --> 00:00:00,150\nSynthetic subtitle.\n",
    )
    .unwrap();
    fs::write(
        &chapters,
        ";FFMETADATA1\ntitle=Preservation fixture\n\n\
         [CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=100\ntitle=First\n\n\
         [CHAPTER]\nTIMEBASE=1/1000\nSTART=100\nEND=200\ntitle=Second\n",
    )
    .unwrap();
    fs::write(&attachment, "attachment payload\n").unwrap();

    let mut arguments = [
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x54:rate=10:duration=0.2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=0.2",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=880:sample_rate=48000:duration=0.2",
        "-f",
        "srt",
        "-i",
    ]
    .map(OsString::from)
    .to_vec();
    arguments.push(subtitle.as_os_str().to_os_string());
    arguments.extend(["-f", "ffmetadata", "-i"].map(OsString::from));
    arguments.push(chapters.as_os_str().to_os_string());
    arguments.extend(
        [
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-map",
            "2:a",
            "-map",
            "3:s",
            "-c:v",
            "ffv1",
            "-filter:a:0",
            "pan=5.1(side)|FL=c0|FR=c0|FC=c0|LFE=0*c0|SL=c0|SR=c0",
            "-c:a:0",
            "eac3",
            "-b:a:0",
            "640k",
            "-c:a:1",
            "pcm_s24le",
            "-c:s",
            "srt",
            "-map_metadata",
            "4",
            "-map_chapters",
            "4",
            "-metadata",
            "comment=Generated preservation signals",
            "-metadata:s:v:0",
            "title=Synthetic video",
            "-metadata:s:a:0",
            "language=eng",
            "-metadata:s:a:0",
            "title=Preferred E-AC-3",
            "-metadata:s:a:1",
            "language=spa",
            "-metadata:s:a:1",
            "title=Original PCM",
            "-metadata:s:s:0",
            "language=eng",
            "-disposition:a:0",
            "default",
            "-disposition:a:1",
            "0",
            "-attach",
        ]
        .map(OsString::from),
    );
    arguments.push(attachment.as_os_str().to_os_string());
    arguments.extend(
        [
            "-metadata:s:t:0",
            "mimetype=text/plain",
            "-metadata:s:t:0",
            "filename=fixture-attachment.txt",
        ]
        .map(OsString::from),
    );
    arguments.push(output.as_os_str().to_os_string());
    let result = runner.run(ffmpeg, arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    attachment
}

fn packet_hashes(
    runner: &ProcessRunner,
    ffprobe: &Path,
    input: &Path,
) -> BTreeMap<usize, Vec<String>> {
    let arguments = [
        OsString::from("-v"),
        OsString::from("error"),
        OsString::from("-show_packets"),
        OsString::from("-show_data_hash"),
        OsString::from("sha256"),
        OsString::from("-show_entries"),
        OsString::from("packet=stream_index,data_hash"),
        OsString::from("-of"),
        OsString::from("json"),
        input.as_os_str().to_os_string(),
    ];
    let result = runner.run(ffprobe, arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    let mut hashes = BTreeMap::<usize, Vec<String>>::new();
    for packet in json["packets"].as_array().unwrap() {
        let Some(index) = packet["stream_index"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
        else {
            continue;
        };
        let Some(hash) = packet["data_hash"].as_str() else {
            continue;
        };
        hashes.entry(index).or_default().push(hash.into());
    }
    hashes
}

#[derive(Debug)]
struct AudioPacketFact {
    position: u64,
    duration: Option<f64>,
}

fn audio_packet_facts(
    runner: &ProcessRunner,
    ffprobe: &Path,
    input: &Path,
    selector: &str,
) -> Vec<AudioPacketFact> {
    let arguments = [
        OsString::from("-v"),
        OsString::from("error"),
        OsString::from("-select_streams"),
        OsString::from(selector),
        OsString::from("-show_packets"),
        OsString::from("-show_entries"),
        OsString::from("packet=pos,duration_time"),
        OsString::from("-of"),
        OsString::from("json"),
        input.as_os_str().to_os_string(),
    ];
    let result = runner.run(ffprobe, arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let json: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    json["packets"]
        .as_array()
        .unwrap()
        .iter()
        .map(|packet| AudioPacketFact {
            position: packet["pos"].as_str().unwrap().parse().unwrap(),
            duration: packet["duration_time"]
                .as_str()
                .and_then(|duration| duration.parse().ok()),
        })
        .collect()
}

fn decoded_s24le(runner: &ProcessRunner, ffmpeg: &Path, input: &Path, selector: &str) -> Vec<u8> {
    let arguments = [
        OsString::from("-v"),
        OsString::from("error"),
        OsString::from("-i"),
        input.as_os_str().to_os_string(),
        OsString::from("-map"),
        OsString::from(selector),
        OsString::from("-f"),
        OsString::from("s24le"),
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

fn generate_sparse_pgs(path: &Path) {
    fn append_segment(bytes: &mut Vec<u8>, timestamp: u32, kind: u8, payload: &[u8]) {
        bytes.extend_from_slice(b"PG");
        bytes.extend_from_slice(&timestamp.to_be_bytes());
        bytes.extend_from_slice(&timestamp.to_be_bytes());
        bytes.push(kind);
        bytes.extend_from_slice(&u16::try_from(payload.len()).unwrap().to_be_bytes());
        bytes.extend_from_slice(payload);
    }

    const SHOW: u32 = 11 * 90_000;
    const CLEAR: u32 = 11 * 90_000 + 45_000;
    let mut bytes = Vec::new();
    append_segment(
        &mut bytes,
        SHOW,
        0x16,
        &[
            0x07, 0x80, 0x04, 0x38, 0x10, 0, 1, 0x80, 0, 0, 1, 0, 0, 0, 0, 0, 100, 0, 100,
        ],
    );
    append_segment(
        &mut bytes,
        SHOW,
        0x17,
        &[1, 0, 0, 0, 0, 0, 0x07, 0x80, 0x04, 0x38],
    );
    append_segment(
        &mut bytes,
        SHOW,
        0x14,
        &[0, 0, 0, 16, 128, 128, 0, 1, 235, 128, 128, 255],
    );
    append_segment(
        &mut bytes,
        SHOW,
        0x15,
        &[0, 0, 0, 0xc0, 0, 0, 12, 0, 2, 0, 2, 1, 1, 0, 0, 1, 1, 0, 0],
    );
    append_segment(&mut bytes, SHOW, 0x80, &[]);
    append_segment(
        &mut bytes,
        CLEAR,
        0x16,
        &[0x07, 0x80, 0x04, 0x38, 0x10, 0, 2, 0, 0, 0, 0],
    );
    append_segment(&mut bytes, CLEAR, 0x80, &[]);
    fs::write(path, bytes).unwrap();
}

fn extract_attachment(runner: &ProcessRunner, ffmpeg: &Path, input: &Path, output: &Path) {
    let arguments = [
        OsString::from("-y"),
        OsString::from("-v"),
        OsString::from("error"),
        OsString::from("-dump_attachment:t:0"),
        output.as_os_str().to_os_string(),
        OsString::from("-i"),
        input.as_os_str().to_os_string(),
        OsString::from("-map"),
        OsString::from("0:v:0"),
        OsString::from("-frames:v"),
        OsString::from("1"),
        OsString::from("-f"),
        OsString::from("null"),
        OsString::from("-"),
    ];
    let result = runner.run(ffmpeg, arguments).unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn run_ffmpeg<const N: usize>(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    args: [&str; N],
    output: &Path,
) {
    let mut arguments = args.map(OsString::from).to_vec();
    arguments.push(output.as_os_str().to_os_string());
    let result = runner.run(ffmpeg, &arguments).unwrap();
    assert!(
        result.status.success(),
        "fixture generation failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
}
