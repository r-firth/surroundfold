use std::path::PathBuf;

use clap::{Parser, ValueEnum};

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

    /// Custom stereo HRIR WAV; defaults to the embedded profile.
    #[arg(long, value_name = "PATH")]
    pub hrir: Option<PathBuf>,

    /// Final Matroska path. Defaults beside the input.
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
    #[arg(long, default_value_t = 0.0)]
    pub gain_db: f64,

    /// Swap side and rear surrounds.
    #[arg(long, value_enum, default_value_t)]
    pub surround_swap: Toggle,

    /// Render through a virtual loudspeaker layout.
    #[arg(long, value_enum, default_value_t)]
    pub speaker_virtualizer: Toggle,

    /// Mute reference bed-position sources.
    #[arg(long, value_enum, default_value_t)]
    pub mute_bed: Toggle,

    /// Mute sources on the ground plane.
    #[arg(long, value_enum, default_value_t)]
    pub mute_ground: Toggle,

    /// Room-correction root or explicit stereo FIR package.
    #[arg(long, value_name = "PATH")]
    pub room_correction: Option<PathBuf>,

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

    /// Replace an existing output after the replacement passes verification.
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
