use std::{collections::BTreeMap, ffi::OsString, path::Path, time::Duration};

use crate::{
    error::AppError,
    media::{ChapterManifest, ContainerManifest, StreamManifest, tag},
    process::ProcessRunner,
};

pub const BINAURAL_TITLE: &str = "SurroundFold binaural";
const MUXER_TAGS: &[&str] = &["ENCODER"];

const RESERVED_ARGUMENTS: &[&str] = &[
    "-i",
    "-map",
    "-map_metadata",
    "-map_chapters",
    "-c",
    "-codec",
    "-acodec",
    "-vcodec",
    "-scodec",
    "-dcodec",
    "-f",
    "-y",
    "-n",
    "-shortest",
    "-copyts",
    "-start_at_zero",
    "-avoid_negative_ts",
    "-itsoffset",
    "-isync",
    "-itsscale",
    "-output_ts_offset",
    "-ss",
    "-sseof",
    "-t",
    "-to",
    "-fs",
    "-frames",
    "-vn",
    "-an",
    "-sn",
    "-dn",
    "-af",
    "-vf",
    "-lavfi",
    "-filter_complex",
    "-filter_complex_script",
    "-absf",
    "-vbsf",
    "-attach",
    "-dump_attachment",
    "-progress",
    "-report",
    "-stats",
    "-nostats",
    "-benchmark",
    "-benchmark_all",
    "-pass",
    "-passlogfile",
    "-copy_unknown",
    "-ignore_unknown",
];

/// Converts the flat clap representation into option/value pairs.
///
/// # Errors
///
/// Returns an error if a caller bypassed clap and supplied an incomplete pair,
/// or if an option can alter protected preservation behaviour.
pub fn advanced_argument_pairs(values: &[String]) -> Result<Vec<(String, String)>, AppError> {
    if values.len() % 2 != 0 {
        return Err(AppError::Usage(
            "--ffmpeg-arg requires an option and value".into(),
        ));
    }
    let pairs = values
        .chunks_exact(2)
        .map(|pair| (pair[0].clone(), pair[1].clone()))
        .collect::<Vec<_>>();
    validate_advanced_arguments(&pairs)?;
    Ok(pairs)
}

/// Validates that advanced options cannot replace inputs, mappings, codecs,
/// timing, filtering, reporting, or the output destination.
///
/// # Errors
///
/// Returns a usage error for malformed or protected options.
pub fn validate_advanced_arguments(arguments: &[(String, String)]) -> Result<(), AppError> {
    for (key, _) in arguments {
        if key.trim().is_empty() || !key.starts_with('-') {
            return Err(AppError::Usage(
                "--ffmpeg-arg OPTION must begin with '-'".into(),
            ));
        }
        let base = key.split('=').next().unwrap_or(key);
        let lower = base.to_ascii_lowercase();
        let protected_prefix = ["-map", "-c:", "-codec:", "-filter:", "-af:", "-vf:", "-bsf"]
            .iter()
            .any(|prefix| lower.starts_with(prefix));
        if protected_prefix || RESERVED_ARGUMENTS.iter().any(|reserved| lower == *reserved) {
            return Err(AppError::Usage(format!(
                "advanced ffmpeg option '{key}' would override a protected mux setting"
            )));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
/// Builds a preservation-first ffmpeg argument vector without invoking a shell.
///
/// # Errors
///
/// Returns a usage error if an advanced option attempts to override a protected
/// mux setting.
pub fn build_mux_arguments(
    input: &Path,
    rendered_wave: &Path,
    partial_output: &Path,
    source: &ContainerManifest,
    selected_source: &StreamManifest,
    selected_start: f64,
    advanced: &[(String, String)],
) -> Result<Vec<OsString>, AppError> {
    validate_advanced_arguments(advanced)?;
    let new_audio_index = source
        .streams
        .iter()
        .filter(|stream| stream.codec_type == "audio")
        .count();
    let mut args = [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-copyts",
        "-avoid_negative_ts",
        "disabled",
        "-i",
    ]
    .map(OsString::from)
    .to_vec();
    args.push(input.as_os_str().to_os_string());
    if selected_start.abs() > f64::EPSILON {
        args.push("-itsoffset".into());
        args.push(format_timestamp(selected_start).into());
    }
    args.push("-i".into());
    args.push(rendered_wave.as_os_str().to_os_string());
    args.extend(
        [
            "-map",
            "0",
            "-map",
            "1:a:0",
            "-map_metadata",
            "0",
            "-map_chapters",
            "0",
        ]
        .map(OsString::from),
    );
    if source
        .streams
        .iter()
        .any(|stream| matches!(stream.codec_type.as_str(), "data" | "unknown"))
    {
        args.push("-copy_unknown".into());
    }
    args.extend(["-c", "copy"].map(OsString::from));
    args.push(format!("-c:a:{new_audio_index}").into());
    args.push("pcm_s16le".into());

    for (key, value) in &source.format.tags {
        args.push("-metadata".into());
        args.push(format!("{key}={value}").into());
    }
    for (position, stream) in source.streams.iter().enumerate() {
        for (key, value) in &stream.tags {
            args.push(format!("-metadata:s:{position}").into());
            args.push(format!("{key}={value}").into());
        }
        args.push(format!("-disposition:{position}").into());
        args.push(disposition_value(&stream.disposition).into());
    }

    args.push(format!("-metadata:s:a:{new_audio_index}").into());
    args.push(format!("title={BINAURAL_TITLE}").into());
    if let Some(language) = selected_source
        .tag("language")
        .filter(|value| !value.trim().is_empty())
    {
        args.push(format!("-metadata:s:a:{new_audio_index}").into());
        args.push(format!("language={language}").into());
    }
    args.push(format!("-disposition:a:{new_audio_index}").into());
    args.push("0".into());
    for (key, value) in advanced {
        args.push(key.into());
        args.push(value.into());
    }
    args.push(partial_output.as_os_str().to_os_string());
    Ok(args)
}

/// Runs a previously constructed preservation mux.
///
/// # Errors
///
/// Returns an error when ffmpeg cannot execute or exits unsuccessfully.
pub fn mux(runner: &ProcessRunner, ffmpeg: &Path, arguments: &[OsString]) -> Result<(), AppError> {
    let output = runner.run(ffmpeg, arguments)?;
    if output.status.success() {
        return Ok(());
    }
    let message = String::from_utf8_lossy(&output.stderr);
    let detail = message
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("unknown ffmpeg error");
    Err(AppError::Mux(format!(
        "ffmpeg mux failed with {}: {detail}",
        output.status
    )))
}

/// Verifies preservation and the properties and timing of the appended stream.
///
/// # Errors
///
/// Returns a mux error for any preservation, format, metadata, disposition, or
/// synchronization mismatch.
pub fn verify_output(
    source: &ContainerManifest,
    output: &ContainerManifest,
    selected_source: &StreamManifest,
    expected_sample_rate: u32,
    rendered_samples: u64,
    expected_start: f64,
) -> Result<(), AppError> {
    let original_positions = verify_source_preservation(source, output)?;

    let source_audio_count = source
        .streams
        .iter()
        .filter(|stream| stream.codec_type == "audio")
        .count();
    let appended = output
        .streams
        .iter()
        .filter(|stream| stream.codec_type == "audio")
        .nth(source_audio_count)
        .ok_or_else(|| AppError::Mux("partial output has no appended audio stream".into()))?;
    if appended.codec_type != "audio"
        || !appended.codec_name.eq_ignore_ascii_case("pcm_s16le")
        || appended.channels != Some(2)
        || appended.sample_rate != Some(expected_sample_rate)
    {
        return Err(AppError::Mux(
            "appended stream is not stereo signed 16-bit PCM at the expected sample rate".into(),
        ));
    }
    if appended.tag("title") != Some(BINAURAL_TITLE) {
        return Err(AppError::Mux("appended binaural title is missing".into()));
    }
    if let Some(language) = selected_source.tag("language")
        && !appended
            .tag("language")
            .is_some_and(|actual| actual.eq_ignore_ascii_case(language))
    {
        return Err(AppError::Mux(
            "appended language does not match the selected source".into(),
        ));
    }
    if appended.disposition.get("default").copied().unwrap_or(0) != 0 {
        return Err(AppError::Mux(
            "appended binaural stream is marked default".into(),
        ));
    }

    let tolerance = (1.0 / f64::from(expected_sample_rate)).max(0.001);
    let selected_source_position = source
        .streams
        .iter()
        .position(|stream| stream.index == selected_source.index)
        .ok_or_else(|| AppError::Mux("selected source stream is absent from manifest".into()))?;
    let synchronized_start = output.streams[original_positions[selected_source_position]]
        .start_time
        .unwrap_or(expected_start);
    let actual_start = appended
        .start_time
        .ok_or_else(|| AppError::Mux("ffprobe did not report appended start time".into()))?;
    if (actual_start - synchronized_start).abs() > tolerance {
        return Err(AppError::Mux(format!(
            "appended stream starts at {actual_start}s; selected stream starts at {synchronized_start}s"
        )));
    }

    let whole_seconds = rendered_samples / u64::from(expected_sample_rate);
    let remaining_samples = u32::try_from(rendered_samples % u64::from(expected_sample_rate))
        .map_err(|error| {
            AppError::Mux(format!("could not calculate rendered duration: {error}"))
        })?;
    let expected_duration = Duration::from_secs(whole_seconds).as_secs_f64()
        + f64::from(remaining_samples) / f64::from(expected_sample_rate);
    let duration_from_tag = appended
        .tag("DURATION")
        .and_then(parse_duration)
        .map(|end| end - actual_start);
    let actual_duration = appended.duration.or(duration_from_tag).ok_or_else(|| {
        AppError::Mux("ffprobe reported neither appended duration nor DURATION tag".into())
    })?;
    if (actual_duration - expected_duration).abs() > tolerance {
        return Err(AppError::Mux(format!(
            "appended duration is {actual_duration}s; expected {expected_duration}s"
        )));
    }
    Ok(())
}

fn verify_source_preservation(
    source: &ContainerManifest,
    output: &ContainerManifest,
) -> Result<Vec<usize>, AppError> {
    if !output
        .format
        .format_name
        .to_ascii_lowercase()
        .contains("matroska")
    {
        return Err(AppError::Mux(format!(
            "partial output format is '{}', not Matroska",
            output.format.format_name
        )));
    }
    if output.streams.len() != source.streams.len() + 1 {
        return Err(AppError::Mux(format!(
            "stream preservation failed: expected {}, found {}",
            source.streams.len() + 1,
            output.streams.len()
        )));
    }

    let original_positions = original_output_positions(source, output)?;
    for (position, (original, output_position)) in
        source.streams.iter().zip(&original_positions).enumerate()
    {
        verify_original_stream(original, &output.streams[*output_position], position)?;
    }
    verify_chapters(&source.chapters, &output.chapters)?;
    verify_tags(
        &source.format.tags,
        &output.format.tags,
        "global metadata",
        MUXER_TAGS,
    )?;
    Ok(original_positions)
}

fn original_output_positions(
    source: &ContainerManifest,
    output: &ContainerManifest,
) -> Result<Vec<usize>, AppError> {
    let mut type_ordinals = BTreeMap::<String, usize>::new();
    source
        .streams
        .iter()
        .map(|stream| {
            let kind = stream.codec_type.to_ascii_lowercase();
            let ordinal = type_ordinals.entry(kind.clone()).or_default();
            let position = output
                .streams
                .iter()
                .enumerate()
                .filter(|(_, candidate)| candidate.codec_type.eq_ignore_ascii_case(&kind))
                .nth(*ordinal)
                .map(|(position, _)| position)
                .ok_or_else(|| {
                    AppError::Mux(format!(
                        "original {} stream {} is missing from the partial output",
                        stream.codec_type, *ordinal
                    ))
                })?;
            *ordinal += 1;
            Ok(position)
        })
        .collect()
}

fn verify_original_stream(
    source: &StreamManifest,
    output: &StreamManifest,
    position: usize,
) -> Result<(), AppError> {
    if !source.codec_type.eq_ignore_ascii_case(&output.codec_type)
        || !source.codec_name.eq_ignore_ascii_case(&output.codec_name)
    {
        return Err(AppError::Mux(format!(
            "original stream {position} changed from {}/{} to {}/{}",
            source.codec_type, source.codec_name, output.codec_type, output.codec_name
        )));
    }
    verify_tags(
        &source.tags,
        &output.tags,
        &format!("stream {position} metadata"),
        &[],
    )?;
    for (key, expected) in &source.disposition {
        if output.disposition.get(key).copied().unwrap_or(0) != *expected {
            return Err(AppError::Mux(format!(
                "original stream {position} disposition '{key}' changed"
            )));
        }
    }
    Ok(())
}

fn verify_chapters(source: &[ChapterManifest], output: &[ChapterManifest]) -> Result<(), AppError> {
    if source.len() != output.len() {
        return Err(AppError::Mux(format!(
            "chapter preservation failed: expected {}, found {}",
            source.len(),
            output.len()
        )));
    }
    for (position, (expected, actual)) in source.iter().zip(output).enumerate() {
        if optional_difference(expected.start_time, actual.start_time) > 0.001
            || optional_difference(expected.end_time, actual.end_time) > 0.001
        {
            return Err(AppError::Mux(format!("chapter {position} timing changed")));
        }
        verify_tags(
            &expected.tags,
            &actual.tags,
            &format!("chapter {position} metadata"),
            &[],
        )?;
    }
    Ok(())
}

fn verify_tags(
    expected: &BTreeMap<String, String>,
    actual: &BTreeMap<String, String>,
    description: &str,
    generated: &[&str],
) -> Result<(), AppError> {
    for (key, expected_value) in expected {
        if key.eq_ignore_ascii_case("DURATION")
            || generated
                .iter()
                .any(|generated| key.eq_ignore_ascii_case(generated))
        {
            continue;
        }
        let actual_value = tag(actual, key);
        if actual_value != Some(expected_value.as_str()) {
            return Err(AppError::Mux(format!(
                "{description} field '{key}' changed or is missing"
            )));
        }
    }
    Ok(())
}

fn optional_difference(first: Option<f64>, second: Option<f64>) -> f64 {
    match (first, second) {
        (None, None) => 0.0,
        (Some(first), Some(second)) => (first - second).abs(),
        _ => f64::INFINITY,
    }
}

fn parse_duration(value: &str) -> Option<f64> {
    let mut parts = value.split(':');
    let hours: f64 = parts.next()?.parse().ok()?;
    let minutes: f64 = parts.next()?.parse().ok()?;
    let seconds: f64 = parts.next()?.parse().ok()?;
    if parts.next().is_some() {
        return None;
    }
    Some(hours * 3600.0 + minutes * 60.0 + seconds)
}

fn disposition_value(disposition: &BTreeMap<String, i32>) -> String {
    let enabled = disposition
        .iter()
        .filter(|(_, value)| **value != 0)
        .map(|(key, _)| key.as_str())
        .collect::<Vec<_>>();
    if enabled.is_empty() {
        "0".into()
    } else {
        enabled.join("+")
    }
}

fn format_timestamp(value: f64) -> String {
    let formatted = format!("{value:.9}");
    let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
    if matches!(trimmed, "" | "-0") {
        "0".into()
    } else {
        trimmed.into()
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, ffi::OsString, path::Path};

    use super::{
        BINAURAL_TITLE, advanced_argument_pairs, build_mux_arguments, verify_output, verify_tags,
    };
    use crate::media::{ContainerManifest, FormatManifest, StreamManifest};

    fn stream(index: usize, kind: &str, codec: &str) -> StreamManifest {
        StreamManifest {
            index,
            codec_type: kind.into(),
            codec_name: codec.into(),
            profile: None,
            channels: None,
            sample_rate: None,
            start_time: Some(1.0),
            duration: Some(2.0),
            channel_layout: None,
            tags: BTreeMap::new(),
            disposition: BTreeMap::from([("default".into(), i32::from(index == 1))]),
        }
    }

    fn source_manifest() -> ContainerManifest {
        let mut video = stream(0, "video", "h264");
        video.tags.insert("title".into(), "Picture".into());
        let mut audio = stream(1, "audio", "flac");
        audio.channels = Some(6);
        audio.sample_rate = Some(48_000);
        audio.tags.insert("language".into(), "eng".into());
        ContainerManifest {
            streams: vec![video, audio],
            chapters: vec![],
            format: FormatManifest {
                format_name: "matroska,webm".into(),
                start_time: Some(1.0),
                duration: Some(2.0),
                tags: BTreeMap::from([("title".into(), "Fixture".into())]),
            },
        }
    }

    #[test]
    fn protected_advanced_arguments_are_rejected() {
        assert!(advanced_argument_pairs(&["-map".into(), "0".into()]).is_err());
        assert!(advanced_argument_pairs(&["-c:a:0".into(), "aac".into()]).is_err());
        assert!(advanced_argument_pairs(&["movflags".into(), "faststart".into()]).is_err());
    }

    #[test]
    fn mux_maps_every_source_stream_then_appends_pcm() {
        let source = source_manifest();
        let args = build_mux_arguments(
            Path::new("source.mkv"),
            Path::new("render.wav"),
            Path::new("partial.mkv"),
            &source,
            &source.streams[1],
            1.25,
            &[("-cluster_time_limit".into(), "5000".into())],
        )
        .unwrap();
        let expected_tail = [
            "-metadata:s:a:1",
            &format!("title={BINAURAL_TITLE}"),
            "-metadata:s:a:1",
            "language=eng",
            "-disposition:a:1",
            "0",
            "-cluster_time_limit",
            "5000",
            "partial.mkv",
        ]
        .map(OsString::from);
        assert!(
            args.windows(4)
                .any(|window| window == os_slice(["-map", "0", "-map", "1:a:0"]))
        );
        assert!(args.ends_with(&expected_tail));
    }

    #[test]
    fn verifier_accepts_preserved_streams_and_appended_pcm() {
        let source = source_manifest();
        let mut output = source.clone();
        output
            .format
            .tags
            .insert("ENCODER".into(), "Lavf62.3.100".into());
        let mut appended = stream(2, "audio", "pcm_s16le");
        appended.channels = Some(2);
        appended.sample_rate = Some(48_000);
        appended.duration = Some(2.0);
        appended.tags.insert("title".into(), BINAURAL_TITLE.into());
        appended.tags.insert("language".into(), "eng".into());
        appended.disposition.insert("default".into(), 0);
        output.streams.push(appended);
        verify_output(&source, &output, &source.streams[1], 48_000, 96_000, 1.0).unwrap();
    }

    #[test]
    fn only_declared_container_generated_tags_may_change() {
        let expected = BTreeMap::from([
            ("ENCODER".into(), "Lavf58.76.100".into()),
            ("title".into(), "Original title".into()),
        ]);
        let mut actual = BTreeMap::from([
            ("ENCODER".into(), "Lavf62.3.100".into()),
            ("title".into(), "Original title".into()),
        ]);
        verify_tags(&expected, &actual, "global metadata", &["ENCODER"]).unwrap();

        actual.insert("title".into(), "Changed title".into());
        assert!(verify_tags(&expected, &actual, "global metadata", &["ENCODER"]).is_err());
    }

    #[test]
    fn verifier_allows_matroska_to_place_attachments_after_appended_audio() {
        let mut source = source_manifest();
        let mut attachment = stream(2, "attachment", "unknown");
        attachment.tags.insert("filename".into(), "font.ttf".into());
        source.streams.push(attachment.clone());

        let mut appended = stream(2, "audio", "pcm_s16le");
        appended.channels = Some(2);
        appended.sample_rate = Some(48_000);
        appended.duration = Some(2.0);
        appended.tags.insert("title".into(), BINAURAL_TITLE.into());
        appended.tags.insert("language".into(), "eng".into());
        appended.disposition.insert("default".into(), 0);
        attachment.index = 3;
        let output = ContainerManifest {
            streams: vec![
                source.streams[0].clone(),
                source.streams[1].clone(),
                appended,
                attachment,
            ],
            chapters: source.chapters.clone(),
            format: source.format.clone(),
        };

        verify_output(&source, &output, &source.streams[1], 48_000, 96_000, 1.0).unwrap();
    }

    fn os_slice<const N: usize>(values: [&str; N]) -> [OsString; N] {
        values.map(OsString::from)
    }
}
