use std::{
    io::{Cursor, Read},
    path::Path,
};

use audioadapter_buffers::direct::InterleavedSlice;
use hound::{SampleFormat, WavReader};
use rubato::{Fft, FixedSync, Resampler};
use rustfft::{FftPlanner, num_complex::Complex32};
use serde::Serialize;
use sofar::reader::{Filter as SofaFilter, OpenOptions as SofaOpenOptions, Sofar};

use crate::error::AppError;

const SILENCE: f32 = 1e-9;
const TRANSIENT_THRESHOLD: f32 = 0.5;
const MINIMUM_IMPULSE_SPACING: usize = 64;
const MAXIMUM_SOFA_DELAY_SECONDS: f32 = 1.0;
const FRACTIONAL_DELAY_RADIUS: usize = 8;
const FRACTIONAL_DELAY_TAPS: usize = FRACTIONAL_DELAY_RADIUS * 2 + 1;
const SOFA_DIRECTION_DEDUPLICATION_DOT: f32 = 0.999_5;
const DIFFUSE_FIELD_DIRECTIONS: usize = 192;
const DIFFUSE_FIELD_EQ_MAX_DB: f32 = 6.0;
const DEFAULT_HRIR_WAV: &[u8] = include_bytes!("../assets/default_hrir.wav");
const SOFA_SPEAKERS: [Speaker; 18] = [
    Speaker::FrontLeft,
    Speaker::FrontRight,
    Speaker::FrontCenter,
    Speaker::RearLeft,
    Speaker::RearRight,
    Speaker::RearCenter,
    Speaker::SideLeft,
    Speaker::SideRight,
    Speaker::WideLeft,
    Speaker::WideRight,
    Speaker::TopFrontLeft,
    Speaker::TopFrontCenter,
    Speaker::TopFrontRight,
    Speaker::TopSideLeft,
    Speaker::TopSideRight,
    Speaker::TopRearLeft,
    Speaker::TopRearCenter,
    Speaker::TopRearRight,
];

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
pub struct DirectionalHrir {
    /// Unit direction in renderer coordinates: right, forward, up.
    pub direction: [f32; 3],
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct HrirSet {
    pub sample_rate: u32,
    pub channels: Vec<HrirChannel>,
    /// Extra measured directions used for continuous object panning.
    ///
    /// Concatenated WAV profiles leave this empty. SOFA profiles populate a
    /// denser full-sphere array in addition to the exact speaker routes.
    pub directional: Vec<DirectionalHrir>,
}

impl HrirSet {
    /// Loads a concatenated WAV profile, or a SOFA profile resampled to the
    /// selected stream's sample rate.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile is malformed, unsupported, silent, or
    /// contains invalid timing data.
    pub fn load(path: &Path, sample_rate: u32) -> Result<Self, AppError> {
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("sofa"))
        {
            Self::load_sofa(path, sample_rate)
        } else {
            Self::load_concatenated_wave(path)?.resample_to(sample_rate)
        }
    }

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

    /// Loads the embedded profile and resamples it to the selected stream.
    ///
    /// # Errors
    ///
    /// Returns an error if the bundled profile is invalid or cannot be
    /// resampled to `sample_rate`.
    pub fn load_default_at(sample_rate: u32) -> Result<Self, AppError> {
        Self::load_default()?.resample_to(sample_rate)
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

    #[allow(clippy::cast_precision_loss)] // Audio sample rates are exactly representable at supported magnitudes.
    fn load_sofa(path: &Path, sample_rate: u32) -> Result<Self, AppError> {
        if sample_rate == 0 {
            return Err(AppError::InvalidHrir(
                "SOFA target sample rate must be non-zero".into(),
            ));
        }
        let mut options = SofaOpenOptions::new();
        options.sample_rate(sample_rate as f32).normalized(false);
        let sofa = options.open(path).map_err(|error| {
            AppError::InvalidHrir(format!(
                "could not load SOFA HRIR {}: {error}",
                path.display()
            ))
        })?;
        if sofa.filter_len() == 0 {
            return Err(AppError::InvalidHrir(
                "SOFA HRIR contains empty impulse responses".into(),
            ));
        }

        let mut filter = SofaFilter::new(sofa.filter_len());
        let equalizer = sofa_diffuse_field_equalizer(&sofa, &mut filter, sample_rate)?;
        let mut channels = SOFA_SPEAKERS
            .iter()
            .map(|speaker| {
                let direction = speaker.position();
                let (left, right) = sofa_impulse(
                    &sofa,
                    &mut filter,
                    direction,
                    sample_rate,
                    &format!("{speaker:?}"),
                )?;
                Ok(HrirChannel {
                    speaker: *speaker,
                    left,
                    right,
                })
            })
            .collect::<Result<Vec<_>, AppError>>()?;
        let mut directional = sofa_virtual_directions()
            .into_iter()
            .map(|direction| {
                let label = format!(
                    "direction [{:.4}, {:.4}, {:.4}]",
                    direction[0], direction[1], direction[2]
                );
                let (left, right) =
                    sofa_impulse(&sofa, &mut filter, direction, sample_rate, &label)?;
                Ok(DirectionalHrir {
                    direction,
                    left,
                    right,
                })
            })
            .collect::<Result<Vec<_>, AppError>>()?;
        for channel in &mut channels {
            channel.left = convolve_impulse(&channel.left, &equalizer);
            channel.right = convolve_impulse(&channel.right, &equalizer);
        }
        for channel in &mut directional {
            channel.left = convolve_impulse(&channel.left, &equalizer);
            channel.right = convolve_impulse(&channel.right, &equalizer);
        }
        Ok(Self {
            sample_rate,
            channels,
            directional,
        })
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
        let peak = signal_peak(&left, &right)?;
        let segment_length = detect_segment_length(&left, &right, peak * TRANSIENT_THRESHOLD)?;
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
            directional: Vec::new(),
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

    fn resample_to(mut self, sample_rate: u32) -> Result<Self, AppError> {
        if sample_rate == 0 {
            return Err(AppError::InvalidHrir(
                "HRIR target sample rate must be non-zero".into(),
            ));
        }
        if self.sample_rate == sample_rate {
            return Ok(self);
        }
        for channel in &mut self.channels {
            (channel.left, channel.right) = resample_stereo_impulse(
                &channel.left,
                &channel.right,
                self.sample_rate,
                sample_rate,
            )?;
        }
        for channel in &mut self.directional {
            (channel.left, channel.right) = resample_stereo_impulse(
                &channel.left,
                &channel.right,
                self.sample_rate,
                sample_rate,
            )?;
        }
        self.sample_rate = sample_rate;
        Ok(self)
    }
}

fn sofa_impulse(
    sofa: &Sofar,
    filter: &mut SofaFilter,
    direction: [f32; 3],
    sample_rate: u32,
    label: &str,
) -> Result<(Vec<f32>, Vec<f32>), AppError> {
    let [forward, left, up] = sofa_direction_coordinates(direction);
    sofa.filter(forward, left, up, filter);
    let mut left = delayed_filter(&filter.left, filter.ldelay, sample_rate)?;
    let mut right = delayed_filter(&filter.right, filter.rdelay, sample_rate)?;
    if left.iter().chain(&right).any(|sample| !sample.is_finite()) {
        return Err(AppError::InvalidHrir(format!(
            "SOFA HRIR contains NaN or infinite samples at {label}"
        )));
    }
    if signal_peak(&left, &right).is_err() {
        return Err(AppError::InvalidHrir(format!(
            "SOFA HRIR has no usable response at {label}"
        )));
    }
    let length = left.len().max(right.len());
    left.resize(length, 0.0);
    right.resize(length, 0.0);
    trim_trailing_silence(&mut left, &mut right);
    Ok((left, right))
}

fn sofa_diffuse_field_equalizer(
    sofa: &Sofar,
    filter: &mut SofaFilter,
    sample_rate: u32,
) -> Result<Vec<f32>, AppError> {
    let impulses = fibonacci_directions(DIFFUSE_FIELD_DIRECTIONS)
        .into_iter()
        .enumerate()
        .map(|(index, direction)| {
            sofa_impulse(
                sofa,
                filter,
                direction,
                sample_rate,
                &format!("diffuse-field sample {index}"),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    diffuse_field_equalizer(&impulses, sample_rate)
}

#[allow(clippy::cast_precision_loss)] // The fixed sampling grid is exactly represented closely enough.
fn fibonacci_directions(count: usize) -> Vec<[f32; 3]> {
    let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
    (0..count)
        .map(|index| {
            let up = 1.0 - 2.0 * (index as f32 + 0.5) / count as f32;
            let horizontal = (1.0 - up * up).sqrt();
            let azimuth = index as f32 * golden_angle;
            [azimuth.sin() * horizontal, azimuth.cos() * horizontal, up]
        })
        .collect()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // FFT sizes, positive bin indexes, and audio sample rates are tightly bounded.
fn diffuse_field_equalizer(
    impulses: &[(Vec<f32>, Vec<f32>)],
    sample_rate: u32,
) -> Result<Vec<f32>, AppError> {
    let maximum_impulse = impulses
        .iter()
        .flat_map(|(left, right)| [left.len(), right.len()])
        .max()
        .unwrap_or(0);
    if maximum_impulse == 0 || sample_rate == 0 {
        return Err(AppError::InvalidHrir(
            "cannot equalize an empty SOFA profile".into(),
        ));
    }
    let equalizer_taps = usize::try_from(sample_rate.div_ceil(188))
        .unwrap_or(usize::MAX)
        .clamp(128, 512);
    let fft_len = maximum_impulse
        .saturating_add(equalizer_taps)
        .next_power_of_two()
        .max(2_048);
    let half = fft_len / 2;
    let mut planner = FftPlanner::<f32>::new();
    let forward = planner.plan_fft_forward(fft_len);
    let mut spectrum = vec![Complex32::ZERO; fft_len];
    let mut power = vec![0.0_f64; half + 1];
    for samples in impulses
        .iter()
        .flat_map(|(left, right)| [left.as_slice(), right.as_slice()])
    {
        spectrum.fill(Complex32::ZERO);
        for (bin, sample) in spectrum.iter_mut().zip(samples) {
            bin.re = *sample;
        }
        forward.process(&mut spectrum);
        for (sum, bin) in power.iter_mut().zip(&spectrum[..=half]) {
            *sum += f64::from(bin.norm_sqr());
        }
    }
    let response_count = (impulses.len() * 2) as f64;
    let mut prefix = Vec::with_capacity(power.len() + 1);
    prefix.push(0.0);
    for value in power {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + value / response_count);
    }

    let octave_radius = 2.0_f32.powf(1.0 / 6.0);
    let mut log_correction = vec![0.0_f32; half + 1];
    for (bin, correction) in log_correction.iter_mut().enumerate().skip(1) {
        let lower = ((bin as f32 / octave_radius).floor() as usize).max(1);
        let upper = ((bin as f32 * octave_radius).ceil() as usize)
            .max(lower + 1)
            .min(half);
        let smoothed_power = (prefix[upper + 1] - prefix[lower]) / (upper - lower + 1) as f64;
        *correction = -0.5 * (smoothed_power.max(f64::EPSILON) as f32).ln();
    }
    log_correction[0] = log_correction[1];

    let bin_hz = sample_rate as f32 / fft_len as f32;
    let reference_start = (200.0 / bin_hz).ceil() as usize;
    let reference_end = (10_000.0 / bin_hz).floor() as usize;
    let reference = &log_correction[reference_start.min(half)..=reference_end.min(half)];
    let mean = reference.iter().sum::<f32>() / reference.len().max(1) as f32;
    let limit = DIFFUSE_FIELD_EQ_MAX_DB * std::f32::consts::LN_10 / 20.0;
    for (bin, correction) in log_correction.iter_mut().enumerate() {
        *correction = (*correction - mean).clamp(-limit, limit);
        let frequency = bin as f32 * bin_hz;
        let edge_weight = if frequency < 80.0 {
            (frequency / 80.0).clamp(0.0, 1.0)
        } else if frequency > 18_000.0 {
            ((sample_rate as f32 * 0.5 - frequency)
                / (sample_rate as f32 * 0.5 - 18_000.0).max(1.0))
            .clamp(0.0, 1.0)
        } else {
            1.0
        };
        *correction *= edge_weight * edge_weight * (3.0 - 2.0 * edge_weight);
    }
    Ok(minimum_phase_impulse(
        &log_correction,
        fft_len,
        equalizer_taps,
    ))
}

#[allow(clippy::cast_precision_loss)] // FFT and FIR lengths are at most a few thousand samples.
fn minimum_phase_impulse(log_magnitude: &[f32], fft_len: usize, taps: usize) -> Vec<f32> {
    let half = fft_len / 2;
    let mut log_spectrum = vec![Complex32::ZERO; fft_len];
    for bin in 0..=half {
        log_spectrum[bin].re = log_magnitude[bin];
    }
    for bin in half + 1..fft_len {
        log_spectrum[bin].re = log_magnitude[fft_len - bin];
    }

    let mut planner = FftPlanner::<f32>::new();
    let inverse = planner.plan_fft_inverse(fft_len);
    inverse.process(&mut log_spectrum);
    let scale = 1.0 / fft_len as f32;
    for value in &mut log_spectrum {
        *value *= scale;
    }
    for value in &mut log_spectrum[1..half] {
        *value *= 2.0;
    }
    log_spectrum[half + 1..].fill(Complex32::ZERO);

    let forward = planner.plan_fft_forward(fft_len);
    forward.process(&mut log_spectrum);
    for value in &mut log_spectrum {
        let magnitude = value.re.exp();
        *value = Complex32::new(magnitude * value.im.cos(), magnitude * value.im.sin());
    }
    inverse.process(&mut log_spectrum);

    let mut impulse = log_spectrum[..taps]
        .iter()
        .map(|value| value.re * scale)
        .collect::<Vec<_>>();
    let fade_start = taps * 3 / 4;
    for (index, sample) in impulse[fade_start..].iter_mut().enumerate() {
        let phase = std::f32::consts::PI * index as f32 / (taps - fade_start) as f32;
        *sample *= 0.5 * (1.0 + phase.cos());
    }
    impulse
}

fn convolve_impulse(samples: &[f32], filter: &[f32]) -> Vec<f32> {
    let mut output = vec![0.0; samples.len() + filter.len() - 1];
    for (input, sample) in samples.iter().copied().enumerate() {
        for (tap, coefficient) in filter.iter().copied().enumerate() {
            output[input + tap] += sample * coefficient;
        }
    }
    output
}

#[allow(clippy::cast_precision_loss)] // The fixed ring sizes are exactly represented.
fn sofa_virtual_directions() -> Vec<[f32; 3]> {
    let mut directions = Vec::new();
    for (elevation_degrees, count) in [
        (0.0_f32, 18_usize),
        (45.0, 12),
        (70.0, 8),
        (-45.0, 12),
        (-70.0, 8),
    ] {
        let elevation = elevation_degrees.to_radians();
        let horizontal = elevation.cos();
        let up = elevation.sin();
        for index in 0..count {
            let azimuth = std::f32::consts::TAU * index as f32 / count as f32;
            let direction = [azimuth.sin() * horizontal, azimuth.cos() * horizontal, up];
            if SOFA_SPEAKERS.iter().all(|speaker| {
                direction_dot(direction, speaker.position()) < SOFA_DIRECTION_DEDUPLICATION_DOT
            }) {
                directions.push(direction);
            }
        }
    }
    directions.push([0.0, 0.0, 1.0]);
    directions.push([0.0, 0.0, -1.0]);
    directions
}

const fn direction_dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn resample_stereo_impulse(
    left: &[f32],
    right: &[f32],
    source_rate: u32,
    target_rate: u32,
) -> Result<(Vec<f32>, Vec<f32>), AppError> {
    let frames = left.len().max(right.len());
    if frames == 0 {
        return Err(AppError::InvalidHrir(
            "cannot resample an empty HRIR impulse".into(),
        ));
    }
    let mut interleaved = Vec::with_capacity(frames * 2);
    for frame in 0..frames {
        interleaved.push(left.get(frame).copied().unwrap_or(0.0));
        interleaved.push(right.get(frame).copied().unwrap_or(0.0));
    }
    let input = InterleavedSlice::new(&interleaved, 2, frames)
        .map_err(|error| AppError::InvalidHrir(format!("could not buffer HRIR: {error}")))?;
    let source_rate = usize::try_from(source_rate)
        .map_err(|error| AppError::InvalidHrir(format!("invalid HRIR sample rate: {error}")))?;
    let target_rate = usize::try_from(target_rate).map_err(|error| {
        AppError::InvalidHrir(format!("invalid HRIR target sample rate: {error}"))
    })?;
    let mut resampler = Fft::<f32>::new(
        source_rate,
        target_rate,
        frames.clamp(64, 1_024),
        1,
        2,
        FixedSync::Both,
    )
    .map_err(|error| {
        AppError::InvalidHrir(format!("could not configure HRIR resampling: {error}"))
    })?;
    let output_capacity = resampler.process_all_needed_output_len(frames);
    let mut interleaved_output = vec![0.0; output_capacity * 2];
    let mut output = InterleavedSlice::new_mut(&mut interleaved_output, 2, output_capacity)
        .map_err(|error| {
            AppError::InvalidHrir(format!("could not buffer resampled HRIR: {error}"))
        })?;
    let (_, output_frames) = resampler
        .process_all_into_buffer(&input, &mut output, frames, None)
        .map_err(|error| AppError::InvalidHrir(format!("HRIR resampling failed: {error}")))?;
    let mut left_output = Vec::with_capacity(output_frames);
    let mut right_output = Vec::with_capacity(output_frames);
    for frame in interleaved_output[..output_frames * 2].chunks_exact(2) {
        left_output.push(frame[0]);
        right_output.push(frame[1]);
    }
    trim_trailing_silence(&mut left_output, &mut right_output);
    Ok((left_output, right_output))
}

#[cfg(test)]
fn sofa_coordinates(speaker: Speaker) -> [f32; 3] {
    sofa_direction_coordinates(speaker.position())
}

fn sofa_direction_coordinates(direction: [f32; 3]) -> [f32; 3] {
    let [right, forward, up] = direction;
    [forward, -right, up]
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn delayed_filter(
    samples: &[f32],
    delay_seconds: f32,
    sample_rate: u32,
) -> Result<Vec<f32>, AppError> {
    if !delay_seconds.is_finite() || !(0.0..=MAXIMUM_SOFA_DELAY_SECONDS).contains(&delay_seconds) {
        return Err(AppError::InvalidHrir(format!(
            "SOFA HRIR contains invalid per-ear delay {delay_seconds} seconds"
        )));
    }
    let delay_samples = delay_seconds * sample_rate as f32;
    let nearest_integer = delay_samples.round();
    let (whole_samples, fractional_sample) = if (delay_samples - nearest_integer).abs() <= 1e-5 {
        (nearest_integer as usize, 0.0)
    } else {
        (delay_samples.floor() as usize, delay_samples.fract())
    };
    let centre = FRACTIONAL_DELAY_RADIUS as f32 + fractional_sample;
    let mut coefficients = [0.0_f32; FRACTIONAL_DELAY_TAPS];
    for (tap, coefficient) in coefficients.iter_mut().enumerate() {
        let distance = tap as f32 - centre;
        let phase = 2.0 * std::f32::consts::PI * tap as f32 / (FRACTIONAL_DELAY_TAPS - 1) as f32;
        let blackman = 0.42 - 0.5 * phase.cos() + 0.08 * (2.0 * phase).cos();
        *coefficient = sinc(distance) * blackman;
    }
    let dc_gain = coefficients.iter().sum::<f32>();
    for coefficient in &mut coefficients {
        *coefficient /= dc_gain;
    }

    let output_len = whole_samples
        .checked_add(samples.len())
        .and_then(|length| length.checked_add(FRACTIONAL_DELAY_TAPS - 1))
        .ok_or_else(|| AppError::InvalidHrir("SOFA per-ear delay is too long".into()))?;
    let mut delayed = vec![0.0; output_len];
    for (input, sample) in samples.iter().copied().enumerate() {
        for (tap, coefficient) in coefficients.iter().copied().enumerate() {
            delayed[whole_samples + input + tap] += sample * coefficient;
        }
    }
    Ok(delayed)
}

fn sinc(value: f32) -> f32 {
    if value.abs() <= f32::EPSILON {
        1.0
    } else {
        let angle = std::f32::consts::PI * value;
        angle.sin() / angle
    }
}

impl Speaker {
    /// Reference-layout unit direction: left/right, rear/front, down/up.
    ///
    /// The listener-level angles use the centres of Dolby's recommended
    /// placement ranges. Overheads use 45 degrees elevation.
    #[must_use]
    pub const fn position(self) -> [f32; 3] {
        match self {
            Self::FrontLeft => [-0.5, 0.866_025_4, 0.0],
            Self::FrontRight => [0.5, 0.866_025_4, 0.0],
            Self::FrontCenter | Self::Lfe => [0.0, 1.0, 0.0],
            Self::RearLeft => [-0.573_576_45, -0.819_152, 0.0],
            Self::RearRight => [0.573_576_45, -0.819_152, 0.0],
            Self::RearCenter => [0.0, -1.0, 0.0],
            Self::SideLeft => [-0.984_807_7, -0.173_648_18, 0.0],
            Self::SideRight => [0.984_807_7, -0.173_648_18, 0.0],
            Self::WideLeft => [-0.866_025_4, 0.5, 0.0],
            Self::WideRight => [0.866_025_4, 0.5, 0.0],
            Self::TopFrontLeft => [-0.353_553_38, 0.612_372_46, 0.707_106_77],
            Self::TopFrontCenter => [0.0, 0.707_106_77, 0.707_106_77],
            Self::TopFrontRight => [0.353_553_38, 0.612_372_46, 0.707_106_77],
            Self::TopSideLeft => [-0.707_106_77, 0.0, 0.707_106_77],
            Self::TopSideRight => [0.707_106_77, 0.0, 0.707_106_77],
            Self::TopRearLeft => [-0.405_579_78, -0.579_228, 0.707_106_77],
            Self::TopRearCenter => [0.0, -0.707_106_77, 0.707_106_77],
            Self::TopRearRight => [0.405_579_78, -0.579_228, 0.707_106_77],
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

fn signal_peak(left: &[f32], right: &[f32]) -> Result<f32, AppError> {
    let peak = left
        .iter()
        .chain(right.iter())
        .copied()
        .map(f32::abs)
        .fold(0.0, f32::max);
    if peak <= SILENCE {
        return Err(AppError::InvalidHrir("HRIR is silent".into()));
    }
    Ok(peak)
}

fn detect_segment_length(
    left: &[f32],
    right: &[f32],
    transient_threshold: f32,
) -> Result<usize, AppError> {
    let mut peaks = Vec::with_capacity(2);
    let mut sample = 0;
    while sample < left.len() && peaks.len() < 2 {
        if left[sample].abs() > transient_threshold || right[sample].abs() > transient_threshold {
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

    use super::{
        FRACTIONAL_DELAY_RADIUS, HrirSet, SOFA_DIRECTION_DEDUPLICATION_DOT, SOFA_SPEAKERS, Speaker,
        convolve_impulse, delayed_filter, diffuse_field_equalizer, direction_dot, sofa_coordinates,
        sofa_virtual_directions,
    };

    #[test]
    fn sofa_virtual_grid_densifies_the_full_sphere_without_duplicate_routes() {
        let directions = sofa_virtual_directions();
        assert_eq!(directions.len(), 48);
        assert!(directions.iter().any(|direction| {
            direction[2].abs() < 1e-6 && direction[0].abs() > 0.1 && direction[1] > 0.9
        }));
        assert!(directions.iter().any(|direction| direction[2] > 0.9));
        assert!(directions.iter().any(|direction| direction[2] < -0.9));
        for direction in directions {
            let magnitude = direction_dot(direction, direction).sqrt();
            assert!((magnitude - 1.0).abs() < 1e-6);
            assert!(SOFA_SPEAKERS.iter().all(|speaker| {
                direction_dot(direction, speaker.position()) < SOFA_DIRECTION_DEDUPLICATION_DOT
            }));
        }
    }

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
    fn preserves_the_profile_gain() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("quiet-hrir.wav");
        write_fixture_with_gain(&path, 2, 128, 0.25);
        let hrir = HrirSet::load_concatenated_wave(&path).unwrap();
        assert!((hrir.channels[0].left[0] - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn concatenated_wave_resamples_to_the_selected_stream() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("hrir.wav");
        write_fixture(&path, 2, 128);
        let hrir = HrirSet::load(&path, 96_000).unwrap();
        assert_eq!(hrir.sample_rate, 96_000);
        let right_peak = hrir.channels[0]
            .right
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
            .unwrap()
            .0;
        assert!((2..=3).contains(&right_peak));
        assert!(hrir.channels.iter().all(|channel| {
            channel
                .left
                .iter()
                .chain(&channel.right)
                .all(|sample| sample.is_finite())
        }));
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

    #[test]
    fn maps_reference_layout_directions_to_sofa_axes() {
        let front_left = sofa_coordinates(Speaker::FrontLeft);
        assert!((front_left[0] - 0.866_025_4).abs() < 1e-6);
        assert!((front_left[1] - 0.5).abs() < 1e-6);
        assert!(front_left[2].abs() < f32::EPSILON);

        let top_rear_right = sofa_coordinates(Speaker::TopRearRight);
        assert!((top_rear_right[0] + 0.579_228).abs() < 1e-6);
        assert!((top_rear_right[1] + 0.405_579_78).abs() < 1e-6);
        assert!((top_rear_right[2] - 0.707_106_77).abs() < 1e-6);
    }

    #[test]
    fn applies_sofa_per_ear_delays() {
        let delayed = delayed_filter(&[1.0, 0.5], 2.0 / 48_000.0, 48_000).unwrap();
        assert!((delayed[FRACTIONAL_DELAY_RADIUS + 2] - 1.0).abs() < 1e-6);
        assert!((delayed[FRACTIONAL_DELAY_RADIUS + 3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn preserves_fractional_sofa_delay() {
        let delayed = delayed_filter(&[1.0], 0.5 / 48_000.0, 48_000).unwrap();
        let sum = delayed.iter().sum::<f32>();
        let centre = delayed
            .iter()
            .enumerate()
            .map(|(sample, value)| f32::from(u16::try_from(sample).unwrap()) * value)
            .sum::<f32>()
            / sum;

        let expected = f32::from(u16::try_from(FRACTIONAL_DELAY_RADIUS).unwrap()) + 0.5;
        assert!((centre - expected).abs() < 1e-4);
    }

    #[test]
    fn diffuse_field_equalizer_leaves_a_flat_profile_unchanged() {
        let impulses = vec![(vec![1.0], vec![1.0]); 8];
        let equalizer = diffuse_field_equalizer(&impulses, 48_000).unwrap();

        assert!((equalizer[0] - 1.0).abs() < 1e-5);
        assert!(
            equalizer[1..]
                .iter()
                .map(|sample| sample.abs())
                .sum::<f32>()
                < 1e-4
        );
    }

    #[test]
    fn diffuse_field_equalizer_reduces_common_tilt_without_changing_itd_or_ild() {
        let common_response = vec![0.75, 0.25];
        let impulses = vec![(common_response.clone(), common_response.clone()); 16];
        let equalizer = diffuse_field_equalizer(&impulses, 48_000).unwrap();
        let corrected_response = convolve_impulse(&common_response, &equalizer);
        let before = spectral_tilt(&common_response);
        let after = spectral_tilt(&corrected_response);
        assert!(
            after.abs() < before.abs(),
            "profile tilt changed from {before:.2} to {after:.2} dB"
        );

        let mut left = vec![0.0; 12];
        let mut right = vec![0.0; 12];
        left[2] = 1.0;
        right[8] = 0.25;
        let level_ratio_before = vector_energy(&left) / vector_energy(&right);
        let left = convolve_impulse(&left, &equalizer);
        let right = convolve_impulse(&right, &equalizer);
        assert_eq!(peak_index(&right) - peak_index(&left), 6);
        let level_ratio_after = vector_energy(&left) / vector_energy(&right);
        assert!((level_ratio_after - level_ratio_before).abs() < 1e-4);
    }

    fn spectral_tilt(samples: &[f32]) -> f32 {
        20.0 * (magnitude_at(samples, 12_000.0) / magnitude_at(samples, 1_000.0)).log10()
    }

    fn magnitude_at(samples: &[f32], frequency: f32) -> f32 {
        let radians = std::f32::consts::TAU * frequency / 48_000.0;
        let (real, imaginary) =
            samples
                .iter()
                .enumerate()
                .fold((0.0, 0.0), |(real, imaginary), (index, sample)| {
                    let phase = radians * f32::from(u16::try_from(index).unwrap());
                    (
                        sample.mul_add(phase.cos(), real),
                        (-sample).mul_add(phase.sin(), imaginary),
                    )
                });
        real.hypot(imaginary)
    }

    fn peak_index(samples: &[f32]) -> usize {
        samples
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
            .unwrap()
            .0
    }

    fn vector_energy(samples: &[f32]) -> f32 {
        samples.iter().map(|sample| sample * sample).sum()
    }

    fn write_fixture(path: &Path, channels: usize, impulse_length: usize) {
        write_fixture_with_gain(path, channels, impulse_length, 1.0);
    }

    fn write_fixture_with_gain(path: &Path, channels: usize, impulse_length: usize, gain: f32) {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for channel in 0..channels {
            for sample in 0..impulse_length {
                let left = if sample == 0 { gain } else { 0.0 };
                let right = if sample == 1 {
                    gain * (0.8 - f32::from(u16::try_from(channel).unwrap()) * 0.01)
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
