use std::{
    collections::BTreeMap,
    ffi::OsString,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use crate::{
    error::AppError,
    process::{ProcessOutput, ProcessRunner},
};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct StreamManifest {
    pub index: usize,
    #[serde(default)]
    pub codec_type: String,
    #[serde(default)]
    pub codec_name: String,
    pub profile: Option<String>,
    pub channels: Option<u16>,
    #[serde(default, deserialize_with = "optional_number")]
    pub sample_rate: Option<u32>,
    #[serde(default, deserialize_with = "optional_number")]
    pub start_time: Option<f64>,
    #[serde(default, deserialize_with = "optional_number")]
    pub duration: Option<f64>,
    #[serde(default)]
    pub channel_layout: Option<String>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
    #[serde(default)]
    pub disposition: BTreeMap<String, i32>,
}

impl StreamManifest {
    #[must_use]
    pub fn tag(&self, name: &str) -> Option<&str> {
        tag(&self.tags, name)
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ChapterManifest {
    #[serde(default, deserialize_with = "optional_number")]
    pub start_time: Option<f64>,
    #[serde(default, deserialize_with = "optional_number")]
    pub end_time: Option<f64>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct FormatManifest {
    #[serde(default)]
    pub format_name: String,
    #[serde(default, deserialize_with = "optional_number")]
    pub start_time: Option<f64>,
    #[serde(default, deserialize_with = "optional_number")]
    pub duration: Option<f64>,
    #[serde(default)]
    pub tags: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct ContainerManifest {
    #[serde(default)]
    pub streams: Vec<StreamManifest>,
    #[serde(default)]
    pub chapters: Vec<ChapterManifest>,
    pub format: FormatManifest,
}

#[must_use]
pub fn tag<'a>(tags: &'a BTreeMap<String, String>, name: &str) -> Option<&'a str> {
    tags.iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

#[derive(Debug)]
pub struct MediaProbe<'a> {
    runner: &'a ProcessRunner,
    executable: PathBuf,
}

impl<'a> MediaProbe<'a> {
    #[must_use]
    pub fn new(runner: &'a ProcessRunner, executable: PathBuf) -> Self {
        Self { runner, executable }
    }

    /// Reads a complete stream, chapter, and container manifest with ffprobe.
    ///
    /// # Errors
    ///
    /// Returns an error when ffprobe cannot run, rejects the input, is
    /// cancelled, or emits malformed JSON.
    pub fn probe(&self, input: &Path) -> Result<ContainerManifest, AppError> {
        let args = [
            OsString::from("-v"),
            OsString::from("error"),
            OsString::from("-show_format"),
            OsString::from("-show_streams"),
            OsString::from("-show_chapters"),
            OsString::from("-of"),
            OsString::from("json"),
            input.as_os_str().to_os_string(),
        ];
        let output = self.runner.run(&self.executable, args)?;
        parse_probe_output(input, &output)
    }
}

fn parse_probe_output(input: &Path, output: &ProcessOutput) -> Result<ContainerManifest, AppError> {
    if !output.status.success() {
        return Err(AppError::UnsupportedInput(format!(
            "ffprobe could not inspect {} ({}): {}",
            input.display(),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    serde_json::from_slice(&output.stdout).map_err(|error| {
        AppError::UnsupportedInput(format!(
            "ffprobe returned invalid JSON for {}: {error}",
            input.display()
        ))
    })
}

fn optional_number<'de, D, T>(deserializer: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: std::str::FromStr + Deserialize<'de>,
    T::Err: std::fmt::Display,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Number<T> {
        Number(T),
        String(String),
    }

    match Option::<Number<T>>::deserialize(deserializer)? {
        Some(Number::Number(value)) => Ok(Some(value)),
        Some(Number::String(value)) if value == "N/A" => Ok(None),
        Some(Number::String(value)) => value.parse().map(Some).map_err(serde::de::Error::custom),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::ContainerManifest;

    #[test]
    fn parses_ffprobe_numbers_as_strings() {
        let json = r#"{
            "streams": [{
                "index": 2,
                "codec_type": "audio",
                "codec_name": "truehd",
                "channels": 8,
                "sample_rate": "48000",
                "start_time": "1.250000",
                "duration": "N/A",
                "tags": {"language": "eng"},
                "disposition": {"default": 1}
            }],
            "chapters": [],
            "format": {"format_name": "matroska,webm", "duration": "12.5"}
        }"#;
        let manifest: ContainerManifest = serde_json::from_str(json).unwrap();
        assert_eq!(manifest.streams[0].sample_rate, Some(48_000));
        assert_eq!(manifest.streams[0].start_time, Some(1.25));
        assert_eq!(manifest.streams[0].duration, None);
        assert_eq!(manifest.format.duration, Some(12.5));
    }
}
