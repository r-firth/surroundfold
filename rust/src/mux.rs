use std::{collections::BTreeMap, ffi::OsString, path::Path, time::Duration};

use crate::{
    cli::OutputCodec,
    error::AppError,
    finishing::{AAC_BITRATE, AAC_CODER, FLAC_COMPRESSION_LEVEL},
    media::{ChapterManifest, ContainerManifest, StreamManifest, tag},
    process::ProcessRunner,
};

pub const BINAURAL_TITLE: &str = "SurroundFold binaural";
const BINAURAL_TITLE_PREFIX: &str = "surroundfold binaural";
const MUXER_TAGS: &[&str] = &["ENCODER"];
const OMITTABLE_STREAM_TAGS: &[&str] = &["creation_time"];

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
    "-strict",
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

#[derive(Clone, Copy, Debug)]
pub struct AppendedTrack<'a> {
    pub path: &'a Path,
    pub title: &'a str,
    pub codec: OutputCodec,
    pub sample_rate: u32,
    pub frames: u64,
}

#[derive(Debug)]
pub struct MuxArguments {
    pub media_stage: Vec<OsString>,
    pub preservation_stage: Vec<OsString>,
}

/// Returns the source manifest used for replacement muxing, excluding tracks
/// created by an earlier `SurroundFold` run.
#[must_use]
pub fn source_without_previous_outputs(source: &ContainerManifest) -> ContainerManifest {
    let mut replacement = source.clone();
    replacement.streams.retain(|stream| {
        stream.codec_type != "audio"
            || !stream.tag("title").is_some_and(|title| {
                title
                    .to_ascii_lowercase()
                    .starts_with(BINAURAL_TITLE_PREFIX)
            })
    });
    replacement
}

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

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
/// Builds a preservation-first ffmpeg argument vector without invoking a shell.
///
/// # Errors
///
/// Returns a usage error if an advanced option attempts to override a protected
/// mux setting.
pub fn build_mux_arguments(
    input: &Path,
    appended_tracks: &[AppendedTrack<'_>],
    partial_output: &Path,
    source: &ContainerManifest,
    selected_source: &StreamManifest,
    selected_start: f64,
    advanced: &[(String, String)],
) -> Result<MuxArguments, AppError> {
    validate_advanced_arguments(advanced)?;
    if appended_tracks.is_empty() {
        return Err(AppError::Usage(
            "at least one rendered track is required for muxing".into(),
        ));
    }
    let new_audio_index = source
        .streams
        .iter()
        .filter(|stream| stream.codec_type == "audio")
        .count();
    let mut media_stage = [
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
    media_stage.push(input.as_os_str().to_os_string());
    for track in appended_tracks {
        if selected_start.abs() > f64::EPSILON {
            media_stage.push("-itsoffset".into());
            media_stage.push(format_timestamp(selected_start).into());
        }
        media_stage.push("-i".into());
        media_stage.push(track.path.as_os_str().to_os_string());
    }
    let media_source_positions = source
        .streams
        .iter()
        .enumerate()
        .filter(|(_, stream)| matches!(stream.codec_type.as_str(), "video" | "audio"))
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    let preservation_source_positions = source
        .streams
        .iter()
        .enumerate()
        .filter(|(_, stream)| !matches!(stream.codec_type.as_str(), "video" | "audio"))
        .map(|(position, _)| position)
        .collect::<Vec<_>>();
    for position in &media_source_positions {
        media_stage.push("-map".into());
        media_stage.push(format!("0:{}", source.streams[*position].index).into());
    }
    for input_index in 1..=appended_tracks.len() {
        media_stage.push("-map".into());
        media_stage.push(format!("{input_index}:a:0").into());
    }
    media_stage
        .extend(["-map_metadata", "-1", "-map_chapters", "-1", "-c", "copy"].map(OsString::from));
    for (offset, track) in appended_tracks.iter().enumerate() {
        let output_audio_index = new_audio_index + offset;
        media_stage.push(format!("-c:a:{output_audio_index}").into());
        match track.codec {
            OutputCodec::Flac => {
                media_stage.push("flac".into());
                media_stage.push(format!("-compression_level:a:{output_audio_index}").into());
                media_stage.push(FLAC_COMPRESSION_LEVEL.into());
            }
            OutputCodec::Aac => {
                media_stage.push("aac".into());
                media_stage.push(format!("-b:a:{output_audio_index}").into());
                media_stage.push(AAC_BITRATE.into());
                media_stage.push(format!("-aac_coder:a:{output_audio_index}").into());
                media_stage.push(AAC_CODER.into());
            }
        }
    }
    media_stage.extend(["-f", "matroska", "pipe:1"].map(OsString::from));

    let mut preservation_stage = [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-copyts",
        "-avoid_negative_ts",
        "disabled",
        "-i",
        "pipe:0",
        "-i",
    ]
    .map(OsString::from)
    .to_vec();
    preservation_stage.push(input.as_os_str().to_os_string());
    preservation_stage.extend(["-map", "0"].map(OsString::from));
    for position in &preservation_source_positions {
        preservation_stage.push("-map".into());
        preservation_stage.push(format!("1:{}", source.streams[*position].index).into());
    }
    preservation_stage.extend(["-map_metadata", "1", "-map_chapters", "1"].map(OsString::from));
    if source
        .streams
        .iter()
        .any(|stream| matches!(stream.codec_type.as_str(), "data" | "unknown"))
    {
        preservation_stage.push("-copy_unknown".into());
    }
    preservation_stage.extend(["-c", "copy"].map(OsString::from));
    for (key, value) in &source.format.tags {
        preservation_stage.push("-metadata".into());
        preservation_stage.push(format!("{key}={value}").into());
    }
    let mut output_positions = vec![0; source.streams.len()];
    for (output_position, source_position) in media_source_positions.iter().enumerate() {
        output_positions[*source_position] = output_position;
    }
    let preservation_offset = media_source_positions.len() + appended_tracks.len();
    for (offset, source_position) in preservation_source_positions.iter().enumerate() {
        output_positions[*source_position] = preservation_offset + offset;
    }
    for (source_position, stream) in source.streams.iter().enumerate() {
        let output_position = output_positions[source_position];
        for (key, value) in &stream.tags {
            preservation_stage.push(format!("-metadata:s:{output_position}").into());
            preservation_stage.push(format!("{key}={value}").into());
        }
        preservation_stage.push(format!("-disposition:{output_position}").into());
        preservation_stage.push(disposition_value(&stream.disposition).into());
    }

    let language = selected_source
        .tag("language")
        .filter(|value| !value.trim().is_empty());
    for (offset, track) in appended_tracks.iter().enumerate() {
        let audio_index = new_audio_index + offset;
        preservation_stage.push(format!("-metadata:s:a:{audio_index}").into());
        preservation_stage.push(format!("title={}", track.title).into());
        if let Some(language) = language {
            preservation_stage.push(format!("-metadata:s:a:{audio_index}").into());
            preservation_stage.push(format!("language={language}").into());
        }
        preservation_stage.push(format!("-disposition:a:{audio_index}").into());
        preservation_stage.push("0".into());
    }
    for (key, value) in advanced {
        preservation_stage.push(key.into());
        preservation_stage.push(value.into());
    }
    preservation_stage.push(partial_output.as_os_str().to_os_string());
    Ok(MuxArguments {
        media_stage,
        preservation_stage,
    })
}

/// Runs a previously constructed preservation mux.
///
/// # Errors
///
/// Returns an error when ffmpeg cannot execute or exits unsuccessfully.
pub fn mux(
    runner: &ProcessRunner,
    ffmpeg: &Path,
    arguments: &MuxArguments,
) -> Result<(), AppError> {
    let output = runner.run_pipeline(
        ffmpeg,
        &arguments.media_stage,
        ffmpeg,
        &arguments.preservation_stage,
    )?;
    if output.producer_status.success() && output.consumer_status.success() {
        return Ok(());
    }
    let producer_detail = concise_ffmpeg_error(&output.producer_stderr);
    let consumer_detail = concise_ffmpeg_error(&output.consumer_stderr);
    Err(AppError::Mux(format!(
        "ffmpeg mux pipeline failed (media stage {}, preservation stage {}): media: {producer_detail}; preservation: {consumer_detail}",
        output.producer_status, output.consumer_status
    )))
}

fn concise_ffmpeg_error(stderr: &[u8]) -> String {
    let message = String::from_utf8_lossy(stderr);
    let mut lines = message
        .lines()
        .rev()
        .filter(|line| !line.trim().is_empty())
        .take(8)
        .map(str::to_owned)
        .collect::<Vec<_>>();
    lines.reverse();
    if lines.is_empty() {
        "no error output".into()
    } else {
        lines.join("\n")
    }
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
    appended_tracks: &[AppendedTrack<'_>],
    expected_start: f64,
) -> Result<(), AppError> {
    if appended_tracks.is_empty() {
        return Err(AppError::Mux(
            "no appended tracks were supplied for verification".into(),
        ));
    }
    let original_positions = verify_source_preservation(source, output, appended_tracks.len())?;

    let source_audio_count = source
        .streams
        .iter()
        .filter(|stream| stream.codec_type == "audio")
        .count();
    let selected_source_position = source
        .streams
        .iter()
        .position(|stream| stream.index == selected_source.index)
        .ok_or_else(|| AppError::Mux("selected source stream is absent from manifest".into()))?;
    let synchronized_start = output.streams[original_positions[selected_source_position]]
        .start_time
        .unwrap_or(expected_start);
    let audio_streams = output
        .streams
        .iter()
        .filter(|stream| stream.codec_type == "audio")
        .collect::<Vec<_>>();
    for (offset, expected) in appended_tracks.iter().enumerate() {
        let appended = audio_streams
            .get(source_audio_count + offset)
            .copied()
            .ok_or_else(|| {
                AppError::Mux(format!(
                    "partial output has no appended audio stream {offset}"
                ))
            })?;
        verify_appended_track(appended, expected, selected_source, synchronized_start)?;
    }
    Ok(())
}

fn verify_appended_track(
    appended: &StreamManifest,
    expected: &AppendedTrack<'_>,
    selected_source: &StreamManifest,
    synchronized_start: f64,
) -> Result<(), AppError> {
    let codec_matches = match expected.codec {
        OutputCodec::Flac => {
            appended.codec_name.eq_ignore_ascii_case("flac")
                && appended.bits_per_raw_sample == Some(24)
                && appended.initial_padding.unwrap_or(0) == 0
        }
        OutputCodec::Aac => {
            appended.codec_name.eq_ignore_ascii_case("aac")
                && appended
                    .profile
                    .as_deref()
                    .is_some_and(|profile| profile.eq_ignore_ascii_case("LC"))
        }
    };
    if !codec_matches
        || appended.channels != Some(2)
        || !appended
            .channel_layout
            .as_deref()
            .is_some_and(|layout| layout.eq_ignore_ascii_case("stereo"))
        || appended.sample_rate != Some(expected.sample_rate)
    {
        return Err(AppError::Mux(format!(
            "appended '{}' stream is not valid 24-bit FLAC or AAC-LC compatibility audio at the expected sample rate",
            expected.title,
        )));
    }
    if appended.tag("title") != Some(expected.title) {
        return Err(AppError::Mux(format!(
            "appended binaural title '{}' is missing",
            expected.title
        )));
    }
    if let Some(language) = selected_source.tag("language")
        && !(language.eq_ignore_ascii_case("und") && appended.tag("language").is_none())
        && !appended
            .tag("language")
            .is_some_and(|actual| actual.eq_ignore_ascii_case(language))
    {
        return Err(AppError::Mux(format!(
            "appended '{}' language does not match the selected source",
            expected.title
        )));
    }
    if appended.disposition.get("default").copied().unwrap_or(0) != 0 {
        return Err(AppError::Mux(format!(
            "appended '{}' stream is marked default",
            expected.title
        )));
    }

    let timestamp_tolerance = (1.0 / f64::from(expected.sample_rate)).max(0.001);
    let actual_start = appended
        .start_time
        .ok_or_else(|| AppError::Mux("ffprobe did not report appended start time".into()))?;
    let encoder_padding =
        f64::from(appended.initial_padding.unwrap_or(0)) / f64::from(expected.sample_rate);
    let presentation_start = actual_start + encoder_padding;
    if (presentation_start - synchronized_start).abs() > timestamp_tolerance {
        return Err(AppError::Mux(format!(
            "appended '{}' presentation starts at {presentation_start}s; selected stream starts at {synchronized_start}s",
            expected.title
        )));
    }

    let whole_seconds = expected.frames / u64::from(expected.sample_rate);
    let remaining_samples = u32::try_from(expected.frames % u64::from(expected.sample_rate))
        .map_err(|error| {
            AppError::Mux(format!("could not calculate rendered duration: {error}"))
        })?;
    let expected_duration = Duration::from_secs(whole_seconds).as_secs_f64()
        + f64::from(remaining_samples) / f64::from(expected.sample_rate);
    let duration_from_tag = appended
        .tag("DURATION")
        .and_then(parse_duration)
        .map(|end| end - actual_start);
    let actual_duration = appended
        .duration
        .or(duration_from_tag)
        .map(|duration| duration - encoder_padding)
        .ok_or_else(|| {
            AppError::Mux("ffprobe reported neither appended duration nor DURATION tag".into())
        })?;
    let duration_tolerance = match expected.codec {
        // Native AAC carries one 1,024-sample priming frame and rounds the tail
        // to a complete frame. Matroska reports the priming separately but its
        // stream end can still include both complete packets.
        OutputCodec::Aac => {
            (2048.0 / f64::from(expected.sample_rate) + 0.006).max(timestamp_tolerance)
        }
        OutputCodec::Flac => 0.002_f64.max(timestamp_tolerance),
    };
    if (actual_duration - expected_duration).abs() > duration_tolerance {
        return Err(AppError::Mux(format!(
            "appended '{}' duration is {actual_duration}s; expected {expected_duration}s",
            expected.title
        )));
    }
    Ok(())
}

fn verify_source_preservation(
    source: &ContainerManifest,
    output: &ContainerManifest,
    appended_count: usize,
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
    if output.streams.len() != source.streams.len() + appended_count {
        return Err(AppError::Mux(format!(
            "stream preservation failed: expected {}, found {}",
            source.streams.len() + appended_count,
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
        &[],
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
        OMITTABLE_STREAM_TAGS,
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
    omittable: &[&str],
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
        if actual_value.is_none()
            && omittable
                .iter()
                .any(|omittable| key.eq_ignore_ascii_case(omittable))
        {
            continue;
        }
        if key.eq_ignore_ascii_case("language")
            && expected_value.eq_ignore_ascii_case("und")
            && actual_value.is_none()
        {
            continue;
        }
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
        AppendedTrack, BINAURAL_TITLE, advanced_argument_pairs, build_mux_arguments,
        source_without_previous_outputs, verify_output, verify_tags,
    };
    use crate::{
        cli::OutputCodec,
        finishing::FLAC_COMPRESSION_LEVEL,
        media::{ContainerManifest, FormatManifest, StreamManifest},
    };

    const LEGACY_CLEAN_TITLE: &str = "SurroundFold binaural - clean";
    const LEGACY_TONAL_TITLE: &str = "SurroundFold binaural - sub lift + low-mid dip (no dynamics)";
    const SECOND_FIXTURE_TITLE: &str = "SurroundFold binaural - second fixture";

    fn stream(index: usize, kind: &str, codec: &str) -> StreamManifest {
        StreamManifest {
            index,
            codec_type: kind.into(),
            codec_name: codec.into(),
            profile: codec.eq_ignore_ascii_case("aac").then(|| "LC".into()),
            channels: None,
            sample_rate: None,
            bits_per_raw_sample: codec.eq_ignore_ascii_case("flac").then_some(24),
            initial_padding: None,
            start_time: Some(1.0),
            duration: Some(2.0),
            channel_layout: matches!(codec.to_ascii_lowercase().as_str(), "aac" | "flac")
                .then(|| "stereo".into()),
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
    fn replacement_manifest_removes_only_previous_surroundfold_audio() {
        let mut source = source_manifest();
        let mut clean = stream(2, "audio", "pcm_s24le");
        clean.tags.insert("title".into(), LEGACY_CLEAN_TITLE.into());
        let mut tonal = stream(3, "audio", "aac");
        tonal.tags.insert("title".into(), LEGACY_TONAL_TITLE.into());
        let mut unrelated = stream(4, "audio", "aac");
        unrelated
            .tags
            .insert("title".into(), "Other binaural mix".into());
        let mut subtitle = stream(5, "subtitle", "subrip");
        subtitle.tags.insert("title".into(), BINAURAL_TITLE.into());
        source.streams.extend([clean, tonal, unrelated, subtitle]);

        let replacement = source_without_previous_outputs(&source);

        assert_eq!(
            replacement
                .streams
                .iter()
                .map(|stream| stream.index)
                .collect::<Vec<_>>(),
            [0, 1, 4, 5]
        );
        assert_eq!(replacement.format, source.format);
        assert_eq!(replacement.chapters, source.chapters);
    }

    #[test]
    fn mux_interleaves_media_before_adding_sparse_streams() {
        let mut source = source_manifest();
        source.streams.push(stream(2, "subtitle", "subrip"));
        let rendered = [AppendedTrack {
            path: Path::new("render.wav"),
            title: BINAURAL_TITLE,
            codec: OutputCodec::Flac,
            sample_rate: 48_000,
            frames: 96_000,
        }];
        let args = build_mux_arguments(
            Path::new("source.mkv"),
            &rendered,
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
            args.media_stage
                .windows(4)
                .any(|window| window == os_slice(["-map", "0:1", "-map", "1:a:0"]))
        );
        assert!(
            args.media_stage.windows(4).any(|window| {
                window
                    == os_slice([
                        "-c:a:1",
                        "flac",
                        "-compression_level:a:1",
                        FLAC_COMPRESSION_LEVEL,
                    ])
            }),
            "media stage did not request the fast lossless FLAC delivery encode"
        );
        assert!(
            !args
                .media_stage
                .iter()
                .any(|argument| argument.to_string_lossy().starts_with("-filter")),
            "finishing must happen in the renderer before lossless delivery"
        );
        assert!(
            !args
                .media_stage
                .windows(2)
                .any(|window| window == os_slice(["-map", "0:2"]))
        );
        assert!(
            args.preservation_stage
                .windows(4)
                .any(|window| window == os_slice(["-map", "0", "-map", "1:2"]))
        );
        assert!(
            args.preservation_stage
                .windows(2)
                .any(|window| window == os_slice(["-c", "copy"])),
            "preservation stage must remain copy-only"
        );
        assert!(args.preservation_stage.ends_with(&expected_tail));
    }

    #[test]
    fn verifier_accepts_preserved_streams_and_appended_flac() {
        let source = source_manifest();
        let mut output = source.clone();
        output
            .format
            .tags
            .insert("ENCODER".into(), "Lavf62.3.100".into());
        let mut appended = stream(2, "audio", "flac");
        appended.channels = Some(2);
        appended.sample_rate = Some(48_000);
        appended.duration = Some(2.0);
        appended.tags.insert("title".into(), BINAURAL_TITLE.into());
        appended.tags.insert("language".into(), "eng".into());
        appended.disposition.insert("default".into(), 0);
        output.streams.push(appended);
        let expected = [AppendedTrack {
            path: Path::new("render.wav"),
            title: BINAURAL_TITLE,
            codec: OutputCodec::Flac,
            sample_rate: 48_000,
            frames: 96_000,
        }];
        verify_output(&source, &output, &source.streams[1], &expected, 1.0).unwrap();
    }

    #[test]
    fn verifier_rejects_experimental_truehd_and_pcm_delivery() {
        let source = source_manifest();
        let expected = [AppendedTrack {
            path: Path::new("render.wav"),
            title: BINAURAL_TITLE,
            codec: OutputCodec::Flac,
            sample_rate: 48_000,
            frames: 96_000,
        }];
        for codec in ["truehd", "pcm_s24le"] {
            let mut output = source.clone();
            let mut appended = stream(2, "audio", codec);
            appended.channels = Some(2);
            appended.sample_rate = Some(48_000);
            appended.duration = Some(2.0);
            appended.tags.insert("title".into(), BINAURAL_TITLE.into());
            appended.tags.insert("language".into(), "eng".into());
            appended.disposition.insert("default".into(), 0);
            output.streams.push(appended);
            assert!(
                verify_output(&source, &output, &source.streams[1], &expected, 1.0).is_err(),
                "{codec} delivery was accepted"
            );
        }
    }

    #[test]
    fn verifier_rejects_aac_without_a_confirmed_lc_profile() {
        let source = source_manifest();
        let expected = [AppendedTrack {
            path: Path::new("render.mka"),
            title: BINAURAL_TITLE,
            codec: OutputCodec::Aac,
            sample_rate: 48_000,
            frames: 96_000,
        }];
        let mut output = source.clone();
        let mut appended = stream(2, "audio", "aac");
        appended.profile = Some("unknown".into());
        appended.channels = Some(2);
        appended.sample_rate = Some(48_000);
        appended.duration = Some(2.0);
        appended.tags.insert("title".into(), BINAURAL_TITLE.into());
        appended.tags.insert("language".into(), "eng".into());
        appended.disposition.insert("default".into(), 0);
        output.streams.push(appended);

        assert!(
            verify_output(&source, &output, &source.streams[1], &expected, 1.0).is_err(),
            "AAC with an unidentified profile was accepted"
        );
    }

    #[test]
    fn verifier_accounts_for_aac_priming_in_presentation_start() {
        let source = source_manifest();
        let expected = [AppendedTrack {
            path: Path::new("render.mka"),
            title: BINAURAL_TITLE,
            codec: OutputCodec::Aac,
            sample_rate: 48_000,
            frames: 96_000,
        }];
        let mut output = source.clone();
        let mut appended = stream(2, "audio", "aac");
        appended.channels = Some(2);
        appended.sample_rate = Some(48_000);
        appended.initial_padding = Some(1024);
        appended.start_time = Some(1.0 - 1024.0 / 48_000.0);
        appended.duration = Some(2.0 + 1024.0 / 48_000.0);
        appended.tags.insert("title".into(), BINAURAL_TITLE.into());
        appended.tags.insert("language".into(), "eng".into());
        appended.disposition.insert("default".into(), 0);
        output.streams.push(appended);
        verify_output(&source, &output, &source.streams[1], &expected, 1.0).unwrap();

        output.streams[2].start_time = Some(1.0);
        assert!(
            verify_output(&source, &output, &source.streams[1], &expected, 1.0).is_err(),
            "AAC whose priming delays presentation start was accepted"
        );
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
        verify_tags(&expected, &actual, "global metadata", &["ENCODER"], &[]).unwrap();

        actual.insert("title".into(), "Changed title".into());
        assert!(verify_tags(&expected, &actual, "global metadata", &["ENCODER"], &[]).is_err());
    }

    #[test]
    fn verifier_only_allows_declared_unrepresentable_tags_to_be_absent() {
        let expected = BTreeMap::from([
            ("creation_time".into(), "2016-12-06T07:08:34.000000Z".into()),
            ("language".into(), "und".into()),
            ("title".into(), "Picture".into()),
        ]);
        let actual = BTreeMap::from([("title".into(), "Picture".into())]);

        verify_tags(
            &expected,
            &actual,
            "stream metadata",
            &[],
            &["creation_time"],
        )
        .unwrap();
        assert!(
            verify_tags(
                &expected,
                &BTreeMap::new(),
                "stream metadata",
                &[],
                &["creation_time"]
            )
            .is_err()
        );
    }

    #[test]
    fn verifier_allows_matroska_to_place_attachments_after_appended_audio() {
        let mut source = source_manifest();
        let mut attachment = stream(2, "attachment", "unknown");
        attachment.tags.insert("filename".into(), "font.ttf".into());
        source.streams.push(attachment.clone());

        let mut appended = stream(2, "audio", "flac");
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

        let expected = [AppendedTrack {
            path: Path::new("render.wav"),
            title: BINAURAL_TITLE,
            codec: OutputCodec::Flac,
            sample_rate: 48_000,
            frames: 96_000,
        }];
        verify_output(&source, &output, &source.streams[1], &expected, 1.0).unwrap();
    }

    #[test]
    fn verifier_accepts_two_distinct_non_default_tracks() {
        let source = source_manifest();
        let mut output = source.clone();
        for (index, title) in [BINAURAL_TITLE, SECOND_FIXTURE_TITLE]
            .into_iter()
            .enumerate()
        {
            let mut appended = stream(index + 2, "audio", "aac");
            appended.channels = Some(2);
            appended.sample_rate = Some(48_000);
            appended.duration = Some(2.0);
            appended.tags.insert("title".into(), title.into());
            appended.tags.insert("language".into(), "eng".into());
            appended.disposition.insert("default".into(), 0);
            output.streams.push(appended);
        }
        let expected = [
            AppendedTrack {
                path: Path::new("clean.wav"),
                title: BINAURAL_TITLE,
                codec: OutputCodec::Aac,
                sample_rate: 48_000,
                frames: 96_000,
            },
            AppendedTrack {
                path: Path::new("tonal.wav"),
                title: SECOND_FIXTURE_TITLE,
                codec: OutputCodec::Aac,
                sample_rate: 48_000,
                frames: 96_000,
            },
        ];

        verify_output(&source, &output, &source.streams[1], &expected, 1.0).unwrap();
    }

    fn os_slice<const N: usize>(values: [&str; N]) -> [OsString; N] {
        values.map(OsString::from)
    }
}
