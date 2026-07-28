use std::{
    io::{Cursor, Read},
    path::Path,
};

use hound::{SampleFormat, WavReader};
use serde::Serialize;

use crate::error::AppError;

const SILENCE: f32 = 1e-9;
const TRANSIENT_THRESHOLD: f32 = 0.5;
const MINIMUM_IMPULSE_SPACING: usize = 64;
const DEFAULT_HRIR_WAV: &[u8] = include_bytes!("../assets/default_hrir.wav");

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Speaker {
    FrontLeft,
    FrontRight,
    FrontCenter,
    Lfe,
    RearLeft,
    RearRight,
    RearCenter,
    SideLeft,
    SideRight,
    WideLeft,
    WideRight,
    TopFrontLeft,
    TopFrontCenter,
    TopFrontRight,
    TopSideLeft,
    TopSideRight,
    TopRearLeft,
    TopRearCenter,
    TopRearRight,
}

#[derive(Clone, Debug)]
pub struct HrirChannel {
    pub speaker: Speaker,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct HrirSet {
    pub sample_rate: u32,
    pub channels: Vec<HrirChannel>,
}

impl HrirSet {
    /// Loads the HRIR profile embedded in the executable.
    ///
    /// # Errors
    ///
    /// Returns an error if the bundled profile is malformed or unsupported.
    pub fn load_default() -> Result<Self, AppError> {
        let reader = WavReader::new(Cursor::new(DEFAULT_HRIR_WAV)).map_err(|error| {
            AppError::InvalidHrir(format!("embedded default HRIR is not a valid WAV: {error}"))
        })?;
        Self::from_wave(read_wave_reader(reader, "embedded default HRIR")?)
    }

    /// Loads the concatenated stereo HRIR representation used by the current
    /// application.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed WAV data, non-stereo input, non-finite
    /// samples, unidentifiable impulse boundaries, or unsupported layouts.
    pub fn load_concatenated_wave(path: &Path) -> Result<Self, AppError> {
        Self::from_wave(read_wave(path, "HRIR")?)
    }

    fn from_wave(wave: WaveData) -> Result<Self, AppError> {
        if wave.channels.len() != 2 {
            return Err(AppError::InvalidHrir(format!(
                "HRIR must contain exactly two WAV channels; found {}",
                wave.channels.len()
            )));
        }
        let mut channels = wave.channels;
        let mut right = channels
            .pop()
            .ok_or_else(|| AppError::InvalidHrir("HRIR right channel is missing".into()))?;
        let mut left = channels
            .pop()
            .ok_or_else(|| AppError::InvalidHrir("HRIR left channel is missing".into()))?;
        trim_leading_silence(&mut left, &mut right);
        normalize(&mut left, &mut right)?;
        let segment_length = detect_segment_length(&left, &right)?;
        let mut segments = split_segments(&left, &right, segment_length);
        while segments
            .last()
            .is_some_and(|(left, right)| is_silent(left) && is_silent(right))
        {
            segments.pop();
        }
        if segments.len() % 2 == 1 {
            if segments.len() < 3 {
                return Err(AppError::InvalidHrir(
                    "odd HRIR layouts require at least three positions".into(),
                ));
            }
            segments.insert(3, segments[2].clone());
        }
        let speakers = standard_layout(segments.len()).ok_or_else(|| {
            AppError::InvalidHrir(format!(
                "HRIR contains {} virtual channels; supported concatenated layouts contain 1 through 16",
                segments.len()
            ))
        })?;

        let channels = speakers
            .iter()
            .enumerate()
            .map(|(index, speaker)| {
                let impulse_index = if *speaker == Speaker::Lfe {
                    index.saturating_sub(1)
                } else {
                    index
                };
                let (mut left, mut right) = segments[impulse_index].clone();
                trim_trailing_silence(&mut left, &mut right);
                HrirChannel {
                    speaker: *speaker,
                    left,
                    right,
                }
            })
            .collect();
        Ok(Self {
            sample_rate: wave.sample_rate,
            channels,
        })
    }

    #[must_use]
    pub fn channel(&self, speaker: Speaker) -> Option<&HrirChannel> {
        self.channels
            .iter()
            .find(|channel| channel.speaker == speaker)
            .or_else(|| fallback_channel(&self.channels, speaker))
    }

    #[must_use]
    pub(crate) fn has_exact(&self, speaker: Speaker) -> bool {
        self.channels
            .iter()
            .any(|channel| channel.speaker == speaker)
    }

    #[must_use]
    pub fn resolved_speaker(&self, speaker: Speaker) -> Option<Speaker> {
        self.channel(speaker).map(|channel| channel.speaker)
    }
}

impl Speaker {
    /// Cartesian virtual-room position: left/right, rear/front, down/up.
    #[must_use]
    pub const fn position(self) -> [f32; 3] {
        match self {
            Self::FrontLeft => [-1.0, 1.0, 0.0],
            Self::FrontRight => [1.0, 1.0, 0.0],
            Self::FrontCenter | Self::Lfe => [0.0, 1.0, 0.0],
            Self::RearLeft => [-1.0, -1.0, 0.0],
            Self::RearRight => [1.0, -1.0, 0.0],
            Self::RearCenter => [0.0, -1.0, 0.0],
            Self::SideLeft => [-1.0, 0.0, 0.0],
            Self::SideRight => [1.0, 0.0, 0.0],
            Self::WideLeft => [-1.0, 0.68, 0.0],
            Self::WideRight => [1.0, 0.68, 0.0],
            Self::TopFrontLeft => [-1.0, 1.0, 1.0],
            Self::TopFrontCenter => [0.0, 1.0, 1.0],
            Self::TopFrontRight => [1.0, 1.0, 1.0],
            Self::TopSideLeft => [-1.0, 0.0, 1.0],
            Self::TopSideRight => [1.0, 0.0, 1.0],
            Self::TopRearLeft => [-1.0, -1.0, 1.0],
            Self::TopRearCenter => [0.0, -1.0, 1.0],
            Self::TopRearRight => [1.0, -1.0, 1.0],
        }
    }

    #[must_use]
    pub const fn surround_swapped(self) -> Self {
        match self {
            Self::RearLeft => Self::SideLeft,
            Self::RearRight => Self::SideRight,
            Self::SideLeft => Self::RearLeft,
            Self::SideRight => Self::RearRight,
            other => other,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct WaveData {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

pub(crate) fn read_wave(path: &Path, description: &str) -> Result<WaveData, AppError> {
    let reader = WavReader::open(path).map_err(|error| {
        AppError::InvalidHrir(format!(
            "could not open {description} {}: {error}",
            path.display()
        ))
    })?;
    read_wave_reader(reader, description)
}

fn read_wave_reader<R: Read>(
    reader: WavReader<R>,
    description: &str,
) -> Result<WaveData, AppError> {
    let spec = reader.spec();
    if spec.channels == 0 || spec.sample_rate == 0 {
        return Err(AppError::InvalidHrir(format!(
            "{description} has an invalid channel count or sample rate"
        )));
    }
    let interleaved = read_samples(reader, spec.sample_format, spec.bits_per_sample)?;
    let channel_count = usize::from(spec.channels);
    if interleaved.is_empty() || interleaved.len() % channel_count != 0 {
        return Err(AppError::InvalidHrir(format!(
            "{description} contains no complete audio frames"
        )));
    }
    if interleaved.iter().any(|sample| !sample.is_finite()) {
        return Err(AppError::InvalidHrir(format!(
            "{description} contains NaN or infinite samples"
        )));
    }
    let mut channels = vec![Vec::with_capacity(interleaved.len() / channel_count); channel_count];
    for frame in interleaved.chunks_exact(channel_count) {
        for (channel, sample) in frame.iter().enumerate() {
            channels[channel].push(*sample);
        }
    }
    Ok(WaveData {
        sample_rate: spec.sample_rate,
        channels,
    })
}

#[allow(clippy::cast_possible_truncation)] // Normalized integer PCM is intentionally represented as f32.
fn read_samples<R: Read>(
    mut reader: WavReader<R>,
    format: SampleFormat,
    bits: u16,
) -> Result<Vec<f32>, AppError> {
    match format {
        SampleFormat::Float => reader
            .samples::<f32>()
            .map(|sample| {
                sample.map_err(|error| AppError::InvalidHrir(format!("invalid float WAV: {error}")))
            })
            .collect(),
        SampleFormat::Int => {
            if bits == 0 || bits > 32 {
                return Err(AppError::InvalidHrir(format!(
                    "unsupported integer HRIR bit depth: {bits}"
                )));
            }
            let scale = 2_f64.powi(i32::from(bits) - 1);
            reader
                .samples::<i32>()
                .map(|sample| {
                    sample
                        .map(|value| (f64::from(value) / scale) as f32)
                        .map_err(|error| {
                            AppError::InvalidHrir(format!("invalid integer WAV: {error}"))
                        })
                })
                .collect()
        }
    }
}

fn trim_leading_silence(left: &mut Vec<f32>, right: &mut Vec<f32>) {
    let first_signal = left
        .iter()
        .zip(right.iter())
        .position(|(left, right)| left.abs() > SILENCE || right.abs() > SILENCE)
        .unwrap_or(left.len());
    left.drain(..first_signal);
    right.drain(..first_signal);
}

fn normalize(left: &mut [f32], right: &mut [f32]) -> Result<(), AppError> {
    let peak = left
        .iter()
        .chain(right.iter())
        .copied()
        .map(f32::abs)
        .fold(0.0, f32::max);
    if peak <= SILENCE {
        return Err(AppError::InvalidHrir("HRIR is silent".into()));
    }
    for sample in left.iter_mut().chain(right.iter_mut()) {
        *sample /= peak;
    }
    Ok(())
}

fn detect_segment_length(left: &[f32], right: &[f32]) -> Result<usize, AppError> {
    let mut peaks = Vec::with_capacity(2);
    let mut sample = 0;
    while sample < left.len() && peaks.len() < 2 {
        if left[sample].abs() > TRANSIENT_THRESHOLD || right[sample].abs() > TRANSIENT_THRESHOLD {
            peaks.push(sample);
            sample = sample.saturating_add(MINIMUM_IMPULSE_SPACING + 1);
        } else {
            sample += 1;
        }
    }
    if peaks.len() != 2 {
        return Err(AppError::InvalidHrir(
            "could not identify two concatenated HRIR boundaries".into(),
        ));
    }
    let length = peaks[1] - peaks[0];
    if length <= MINIMUM_IMPULSE_SPACING {
        return Err(AppError::InvalidHrir(format!(
            "detected HRIR segment is too short: {length} samples"
        )));
    }
    Ok(length)
}

fn split_segments(left: &[f32], right: &[f32], length: usize) -> Vec<(Vec<f32>, Vec<f32>)> {
    (0..left.len().div_ceil(length))
        .map(|index| {
            let start = index * length;
            let end = (start + length).min(left.len());
            (left[start..end].to_vec(), right[start..end].to_vec())
        })
        .collect()
}

fn trim_trailing_silence(left: &mut Vec<f32>, right: &mut Vec<f32>) {
    let length = left
        .iter()
        .zip(right.iter())
        .rposition(|(left, right)| left.abs() > SILENCE || right.abs() > SILENCE)
        .map_or(1, |index| index + 1);
    left.truncate(length);
    right.truncate(length);
}

fn is_silent(samples: &[f32]) -> bool {
    samples.iter().all(|sample| sample.abs() <= SILENCE)
}

fn fallback_channel(channels: &[HrirChannel], speaker: Speaker) -> Option<&HrirChannel> {
    let candidates: &[Speaker] = match speaker {
        Speaker::Lfe | Speaker::FrontCenter => &[
            Speaker::FrontCenter,
            Speaker::FrontLeft,
            Speaker::FrontRight,
        ],
        Speaker::RearLeft => &[Speaker::RearLeft, Speaker::SideLeft, Speaker::FrontLeft],
        Speaker::RearRight => &[Speaker::RearRight, Speaker::SideRight, Speaker::FrontRight],
        Speaker::RearCenter => &[
            Speaker::RearCenter,
            Speaker::RearLeft,
            Speaker::RearRight,
            Speaker::FrontCenter,
        ],
        Speaker::SideLeft => &[Speaker::SideLeft, Speaker::RearLeft, Speaker::FrontLeft],
        Speaker::SideRight => &[Speaker::SideRight, Speaker::RearRight, Speaker::FrontRight],
        Speaker::TopFrontLeft | Speaker::TopSideLeft | Speaker::TopRearLeft => &[
            Speaker::TopFrontLeft,
            Speaker::TopSideLeft,
            Speaker::TopRearLeft,
            Speaker::FrontLeft,
        ],
        Speaker::TopFrontRight | Speaker::TopSideRight | Speaker::TopRearRight => &[
            Speaker::TopFrontRight,
            Speaker::TopSideRight,
            Speaker::TopRearRight,
            Speaker::FrontRight,
        ],
        Speaker::TopFrontCenter | Speaker::TopRearCenter => &[
            Speaker::TopFrontCenter,
            Speaker::TopRearCenter,
            Speaker::FrontCenter,
        ],
        Speaker::WideLeft | Speaker::FrontLeft => &[Speaker::WideLeft, Speaker::FrontLeft],
        Speaker::WideRight | Speaker::FrontRight => &[Speaker::WideRight, Speaker::FrontRight],
    };
    candidates.iter().find_map(|candidate| {
        channels
            .iter()
            .find(|channel| channel.speaker == *candidate)
    })
}

#[rustfmt::skip]
fn standard_layout(count: usize) -> Option<&'static [Speaker]> {
    use Speaker::{
        FrontCenter as C, FrontLeft as L, FrontRight as R, Lfe,
        RearLeft as Rl, RearRight as Rr, SideLeft as Sl, SideRight as Sr,
        TopFrontCenter as Tfc, TopFrontLeft as Tfl, TopFrontRight as Tfr,
        TopRearCenter as Trc, TopRearLeft as Trl, TopRearRight as Trr,
        WideLeft as Wl, WideRight as Wr,
    };
    const LAYOUTS: &[&[Speaker]] = &[
        &[],
        &[C],
        &[L, R],
        &[L, R, C],
        &[L, R, Sl, Sr],
        &[L, R, C, Sl, Sr],
        &[L, R, C, Lfe, Sl, Sr],
        &[L, R, C, Rl, Rr, Sl, Sr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr],
        &[L, R, C, Rl, Rr, Sl, Sr, Tfl, Tfr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr, Tfl, Tfr],
        &[L, R, C, Rl, Rr, Sl, Sr, Tfl, Tfr, Trl, Trr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr, Tfl, Tfr, Trl, Trr],
        &[L, R, C, Rl, Rr, Sl, Sr, Tfl, Tfc, Tfr, Trl, Trc, Trr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr, Tfl, Tfc, Tfr, Trl, Trc, Trr],
        &[L, R, C, Rl, Rr, Sl, Sr, Wl, Wr, Tfl, Tfc, Tfr, Trl, Trc, Trr],
        &[L, R, C, Lfe, Rl, Rr, Sl, Sr, Wl, Wr, Tfl, Tfc, Tfr, Trl, Trc, Trr],
    ];
    LAYOUTS.get(count).copied()
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hound::{SampleFormat, WavSpec, WavWriter};

    use super::{HrirSet, Speaker};

    #[test]
    fn parses_concatenated_stereo_impulses() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hrir.wav");
        write_fixture(&path, 6, 128);
        let hrir = HrirSet::load_concatenated_wave(&path).unwrap();
        assert_eq!(hrir.sample_rate, 48_000);
        assert_eq!(hrir.channels.len(), 6);
        assert_eq!(hrir.channels[0].speaker, Speaker::FrontLeft);
        assert_eq!(hrir.channels[3].speaker, Speaker::Lfe);
        assert_eq!(
            hrir.channels[3].left,
            hrir.channel(Speaker::FrontCenter).unwrap().left
        );
    }

    #[test]
    fn parses_embedded_default() {
        let hrir = HrirSet::load_default().unwrap();
        assert_eq!(hrir.sample_rate, 48_000);
        assert_eq!(hrir.channels.len(), 8);
        assert_eq!(hrir.channels.last().unwrap().speaker, Speaker::SideRight);
        assert!(hrir.channels.iter().all(|channel| {
            !channel.left.is_empty()
                && !channel.right.is_empty()
                && channel
                    .left
                    .iter()
                    .chain(&channel.right)
                    .all(|sample| sample.is_finite())
        }));
    }

    #[test]
    fn odd_layout_duplicates_center_for_symmetry() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hrir.wav");
        write_fixture(&path, 5, 128);
        let hrir = HrirSet::load_concatenated_wave(&path).unwrap();
        assert_eq!(hrir.channels.len(), 6);
    }

    fn write_fixture(path: &Path, channels: usize, impulse_length: usize) {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for channel in 0..channels {
            for sample in 0..impulse_length {
                let left = if sample == 0 { 1.0 } else { 0.0 };
                let right = if sample == 1 {
                    0.8 - f32::from(u16::try_from(channel).unwrap()) * 0.01
                } else {
                    0.0
                };
                writer.write_sample(left).unwrap();
                writer.write_sample(right).unwrap();
            }
        }
        writer.finalize().unwrap();
    }
}
