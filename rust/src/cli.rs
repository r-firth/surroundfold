use std::path::PathBuf;

use clap::{Parser, ValueEnum};
use serde::Serialize;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum Toggle {
    #[default]
    Off,
    On,
}

impl Toggle {
    #[must_use]
    pub const fn enabled(self) -> bool {
        matches!(self, Self::On)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum ProgressMode {
    #[default]
    Text,
    Json,
    Quiet,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum OutputCodec {
    #[default]
    Flac,
    Aac,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum ObjectRendererMode {
    /// Approved virtual-speaker renderer.
    #[default]
    Baseline,
    /// Continuous object filters and source-relative early reflections.
    Continuous,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum DistanceRendererMode {
    /// Approved fixed sparse early field.
    #[default]
    Baseline,
    /// Constant-power direct sound plus source-relative image reflections.
    ImageSource,
}

#[derive(Debug, Parser)]
#[command(
    name = "surroundfold",
    version,
    about = "Offline object- and channel-audio binaural renderer",
    long_about = None
)]
#[allow(clippy::struct_excessive_bools)]
pub struct Cli {
    /// Input video or audio file.
    pub input: PathBuf,

    /// Custom SOFA or concatenated stereo HRIR WAV; defaults to the embedded profile.
    #[arg(long, value_name = "PATH")]
    pub hrir: Option<PathBuf>,

    /// Write a separate Matroska file instead of replacing the input in place.
    #[arg(long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Zero-based audio-track index; bypasses automatic selection.
    #[arg(long)]
    pub track: Option<usize>,

    /// Require this language during automatic selection.
    #[arg(long)]
    pub language: Option<String>,

    /// List audio tracks and exit without rendering.
    #[arg(long)]
    pub list_tracks: bool,

    /// Height generation.
    #[arg(long, value_enum, default_value_t)]
    pub upconvert: Toggle,

    /// Matrix expansion before height generation.
    #[arg(long, value_enum, default_value_t)]
    pub matrix: Toggle,

    /// Height-generation effect percentage.
    #[arg(long, default_value_t = 75.0)]
    pub effect: f32,

    /// Generated-object movement smoothing percentage.
    #[arg(long, default_value_t = 80.0)]
    pub smoothness: f32,

    /// Additional render gain in decibels.
    #[arg(long, default_value_t = -5.5)]
    pub gain_db: f64,

    /// Swap side and rear surrounds.
    #[arg(long, value_enum, default_value_t)]
    pub surround_swap: Toggle,

    /// Bypass HRIR convolution for ground-bed channels.
    #[arg(long, value_enum, default_value_t)]
    pub speaker_virtualizer: Toggle,

    /// Object spatialization architecture.
    #[arg(long, value_enum, default_value_t)]
    pub object_renderer: ObjectRendererMode,

    /// Authored-distance and early-reflection architecture.
    #[arg(long, value_enum, default_value_t)]
    pub distance_renderer: DistanceRendererMode,

    /// Mute reference bed-position sources.
    #[arg(long, value_enum, default_value_t)]
    pub mute_bed: Toggle,

    /// Mute sources on the ground plane.
    #[arg(long, value_enum, default_value_t)]
    pub mute_ground: Toggle,

    /// Room-correction root or explicit stereo FIR package.
    #[arg(long, value_name = "PATH")]
    pub room_correction: Option<PathBuf>,

    /// Appended-track codec; FLAC is lossless, AAC maximizes device compatibility.
    #[arg(long, value_enum, default_value_t)]
    pub output_codec: OutputCodec,

    /// Override the appended audio track's Matroska title.
    #[arg(long, value_name = "TITLE")]
    pub track_title: Option<String>,

    /// Preserve audio tracks created by earlier `SurroundFold` runs.
    #[arg(long)]
    pub keep_existing_surroundfold: bool,

    /// MLP/TrueHD presentation index.
    #[arg(long, value_parser = clap::value_parser!(u8).range(0..=3))]
    pub mlp_presentation: Option<u8>,

    /// Relax strict bitstream validation where supported.
    #[arg(long)]
    pub unsafe_parsing: bool,

    /// Explicit ffmpeg executable.
    #[arg(long, value_name = "PATH")]
    pub ffmpeg: Option<PathBuf>,

    /// Explicit ffprobe executable.
    #[arg(long, value_name = "PATH")]
    pub ffprobe: Option<PathBuf>,

    /// Keep the isolated per-run temporary directory.
    #[arg(long)]
    pub keep_temp_files: bool,

    /// Progress output format.
    #[arg(long, value_enum, default_value_t)]
    pub progress: ProgressMode,

    /// Replace an existing explicit --output after verification.
    #[arg(long)]
    pub overwrite: bool,

    /// Repeatable advanced ffmpeg option and value pair.
    #[arg(
        long,
        value_names = ["OPTION", "VALUE"],
        num_args = 2,
        action = clap::ArgAction::Append,
        allow_hyphen_values = true
    )]
    pub ffmpeg_arg: Vec<String>,
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Cli, DistanceRendererMode, ObjectRendererMode, OutputCodec};

    #[test]
    fn playback_calibrated_gain_is_the_default() {
        let cli = Cli::try_parse_from(["surroundfold", "input.mkv"]).unwrap();
        assert!((cli.gain_db + 5.5).abs() < f64::EPSILON);
        assert_eq!(cli.output_codec, OutputCodec::Flac);
        assert_eq!(cli.object_renderer, ObjectRendererMode::Baseline);
        assert_eq!(cli.distance_renderer, DistanceRendererMode::Baseline);
        assert_eq!(cli.track_title, None);
        assert!(!cli.keep_existing_surroundfold);
    }

    #[test]
    fn custom_title_and_existing_track_retention_are_explicit() {
        let cli = Cli::try_parse_from([
            "surroundfold",
            "input.mkv",
            "--track-title",
            "SurroundFold 04 - Continuous + Image Distance",
            "--keep-existing-surroundfold",
        ])
        .unwrap();
        assert_eq!(
            cli.track_title.as_deref(),
            Some("SurroundFold 04 - Continuous + Image Distance")
        );
        assert!(cli.keep_existing_surroundfold);
    }
}
