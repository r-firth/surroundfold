#![cfg(unix)]

use std::{
    ffi::OsString,
    fs,
    io::{BufRead, BufReader, Read},
    path::Path,
    process::{Command, Stdio},
    sync::mpsc,
    thread,
    time::Duration,
};

use nix::{
    sys::signal::{Signal, kill},
    unistd::Pid,
};
use surroundfold::{cancel::Cancellation, process::ProcessRunner};

mod common;

use common::generate_hrir;

#[test]
fn cli_handles_leading_hyphen_unicode_spaces_and_apostrophes() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping CLI test because ffmpeg is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let special = directory.path().join("space ünicode's directory");
    fs::create_dir(&special).unwrap();
    let generated = directory.path().join("generated.mkv");
    generate_input(&runner, &ffmpeg, &generated, 0.1);
    let input_name = "-Mövie's source.mkv";
    fs::copy(&generated, special.join(input_name)).unwrap();
    let hrir = directory.path().join("hrir.wav");
    let output = special.join("Rendered ünicode's output.mkv");
    generate_hrir(&hrir);

    let result = Command::new(binary())
        .current_dir(&special)
        .args(["--hrir"])
        .arg(&hrir)
        .args(["--output"])
        .arg(&output)
        .args(["--progress", "json", "--", input_name])
        .output()
        .unwrap();

    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(output.is_file());
    let json: serde_json::Value = serde_json::from_slice(&result.stdout).unwrap();
    assert_eq!(
        json["output"].as_str(),
        fs::canonicalize(&output).unwrap().to_str()
    );
    let timings = &json["timingsSeconds"];
    for phase in [
        "preparation",
        "render",
        "encodeAndMux",
        "verificationAndPublication",
        "total",
    ] {
        assert!(
            timings[phase]
                .as_f64()
                .is_some_and(|seconds| seconds >= 0.0),
            "missing or invalid {phase} timing: {timings}"
        );
    }
    assert!(
        timings["total"].as_f64().unwrap()
            >= timings["render"].as_f64().unwrap() + timings["encodeAndMux"].as_f64().unwrap()
    );
}

#[test]
fn protected_mux_option_fails_before_missing_media_is_opened() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("must-not-exist.mkv");
    let result = Command::new(binary())
        .current_dir(directory.path())
        .args([
            "missing-input.mkv",
            "--hrir",
            "missing-hrir.wav",
            "--ffmpeg-arg",
            "-map",
            "0",
        ])
        .args(["--output"])
        .arg(&output)
        .output()
        .unwrap();

    assert_eq!(result.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(stderr.contains("protected mux setting"), "{stderr}");
    assert!(!output.exists());
}

#[test]
fn sigint_during_render_returns_130_and_leaves_no_output() {
    let runner = ProcessRunner::new(Cancellation::new());
    let Ok(ffmpeg) = runner.locate_required("ffmpeg", None) else {
        eprintln!("skipping cancellation test because ffmpeg is unavailable");
        return;
    };
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("long-source.mkv");
    let hrir = directory.path().join("hrir.wav");
    let output = directory.path().join("cancelled.mkv");
    generate_input(&runner, &ffmpeg, &input, 30.0);
    generate_hrir(&hrir);

    let mut child = Command::new(binary())
        .arg(&input)
        .args(["--hrir"])
        .arg(&hrir)
        .args(["--output"])
        .arg(&output)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = child.stderr.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let (stage_sender, stage_receiver) = mpsc::channel();
    let stderr_reader = thread::spawn(move || {
        let mut text = String::new();
        for line in BufReader::new(stderr).lines() {
            let line = line.unwrap();
            if line.contains("rendering binaural track") {
                let _ = stage_sender.send(());
            }
            text.push_str(&line);
            text.push('\n');
        }
        text
    });
    let stdout_reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        BufReader::new(stdout).read_to_end(&mut bytes).unwrap();
        bytes
    });

    stage_receiver
        .recv_timeout(Duration::from_secs(20))
        .expect("CLI exited before beginning its render");
    kill(
        Pid::from_raw(i32::try_from(child.id()).unwrap()),
        Signal::SIGINT,
    )
    .unwrap();
    let status = child.wait().unwrap();
    let stderr = stderr_reader.join().unwrap();
    let _stdout = stdout_reader.join().unwrap();

    assert_eq!(status.code(), Some(130), "{stderr}");
    assert!(stderr.contains("operation cancelled"), "{stderr}");
    assert!(!output.exists());
}

fn binary() -> &'static Path {
    Path::new(env!("CARGO_BIN_EXE_surroundfold"))
}

fn generate_input(runner: &ProcessRunner, ffmpeg: &Path, output: &Path, duration: f64) {
    let mut arguments = [
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        &format!("sine=frequency=997:sample_rate=48000:duration={duration}"),
        "-c:a",
        "flac",
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
