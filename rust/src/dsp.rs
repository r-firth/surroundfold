use std::{collections::VecDeque, sync::Arc};

use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use rustfft::num_complex::Complex32;

use crate::error::AppError;

pub const DEFAULT_CONVOLUTION_BLOCK: usize = 1_024;

pub struct StereoConvolver {
    block_size: usize,
    fft_size: usize,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    filters_left: Vec<Vec<Complex32>>,
    filters_right: Vec<Vec<Complex32>>,
    history: Vec<Vec<Complex32>>,
    history_head: usize,
    input: Vec<f32>,
    spectrum_left: Vec<Complex32>,
    spectrum_right: Vec<Complex32>,
    time_left: Vec<f32>,
    time_right: Vec<f32>,
    overlap_left: Vec<f32>,
    overlap_right: Vec<f32>,
    forward_scratch: Vec<Complex32>,
    inverse_scratch: Vec<Complex32>,
}

impl std::fmt::Debug for StereoConvolver {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("StereoConvolver")
            .field("block_size", &self.block_size)
            .field("fft_size", &self.fft_size)
            .field("partitions", &self.filters_left.len())
            .finish_non_exhaustive()
    }
}

impl StereoConvolver {
    /// Prepares a uniform partitioned stereo convolution filter.
    ///
    /// # Errors
    ///
    /// Returns an error for empty impulses or a block size that is not a
    /// non-zero power of two.
    pub fn new(
        left_impulse: &[f32],
        right_impulse: &[f32],
        block_size: usize,
    ) -> Result<Self, AppError> {
        if left_impulse.is_empty() || right_impulse.is_empty() {
            return Err(AppError::InvalidHrir(
                "HRIR impulses must not be empty".into(),
            ));
        }
        if !block_size.is_power_of_two() {
            return Err(AppError::InvalidHrir(
                "convolution block size must be a non-zero power of two".into(),
            ));
        }

        let fft_size = block_size * 2;
        let partitions = left_impulse
            .len()
            .max(right_impulse.len())
            .div_ceil(block_size);
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft_size);
        let inverse = planner.plan_fft_inverse(fft_size);
        let filters_left = partition_impulse(left_impulse, block_size, &forward)?;
        let filters_right = partition_impulse(right_impulse, block_size, &forward)?;
        let bins = fft_size / 2 + 1;

        Ok(Self {
            block_size,
            fft_size,
            forward_scratch: forward.make_scratch_vec(),
            inverse_scratch: inverse.make_scratch_vec(),
            forward,
            inverse,
            filters_left: pad_partitions(filters_left, partitions, bins),
            filters_right: pad_partitions(filters_right, partitions, bins),
            history: vec![vec![Complex32::default(); bins]; partitions],
            history_head: 0,
            input: vec![0.0; fft_size],
            spectrum_left: vec![Complex32::default(); bins],
            spectrum_right: vec![Complex32::default(); bins],
            time_left: vec![0.0; fft_size],
            time_right: vec![0.0; fft_size],
            overlap_left: vec![0.0; block_size],
            overlap_right: vec![0.0; block_size],
        })
    }

    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Processes one block and writes a full block for each ear.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is larger than the configured block, the
    /// output blocks have the wrong size, or the FFT implementation fails.
    #[allow(clippy::cast_precision_loss)] // FFT sizes are bounded by addressable buffers.
    pub fn process(
        &mut self,
        input: &[f32],
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> Result<(), AppError> {
        if input.len() > self.block_size {
            return Err(AppError::Render(format!(
                "convolution input has {} samples; block size is {}",
                input.len(),
                self.block_size
            )));
        }
        if output_left.len() != self.block_size || output_right.len() != self.block_size {
            return Err(AppError::Render(
                "convolution output buffers do not match the block size".into(),
            ));
        }

        self.input.fill(0.0);
        self.input[..input.len()].copy_from_slice(input);
        self.forward
            .process_with_scratch(
                &mut self.input,
                &mut self.history[self.history_head],
                &mut self.forward_scratch,
            )
            .map_err(|error| AppError::Render(format!("forward FFT failed: {error}")))?;

        self.spectrum_left.fill(Complex32::default());
        self.spectrum_right.fill(Complex32::default());
        for partition in 0..self.filters_left.len() {
            let history_index =
                (self.history_head + self.history.len() - partition) % self.history.len();
            for bin in 0..self.spectrum_left.len() {
                let input_bin = self.history[history_index][bin];
                self.spectrum_left[bin] += input_bin * self.filters_left[partition][bin];
                self.spectrum_right[bin] += input_bin * self.filters_right[partition][bin];
            }
        }
        self.history_head = (self.history_head + 1) % self.history.len();

        prepare_real_inverse_spectrum(&mut self.spectrum_left)?;
        prepare_real_inverse_spectrum(&mut self.spectrum_right)?;
        self.inverse
            .process_with_scratch(
                &mut self.spectrum_left,
                &mut self.time_left,
                &mut self.inverse_scratch,
            )
            .map_err(|error| AppError::Render(format!("left inverse FFT failed: {error}")))?;
        self.inverse
            .process_with_scratch(
                &mut self.spectrum_right,
                &mut self.time_right,
                &mut self.inverse_scratch,
            )
            .map_err(|error| AppError::Render(format!("right inverse FFT failed: {error}")))?;

        let scale = 1.0 / self.fft_size as f32;
        for sample in 0..self.block_size {
            output_left[sample] = self.time_left[sample] * scale + self.overlap_left[sample];
            output_right[sample] = self.time_right[sample] * scale + self.overlap_right[sample];
            self.overlap_left[sample] = self.time_left[sample + self.block_size] * scale;
            self.overlap_right[sample] = self.time_right[sample + self.block_size] * scale;
        }
        Ok(())
    }
}

struct ConvolverBus {
    filters_left: Vec<Vec<Complex32>>,
    filters_right: Vec<Vec<Complex32>>,
    history: Vec<Vec<Complex32>>,
    history_head: usize,
}

/// Convolves many independent virtual-speaker buses into one stereo output.
///
/// Each active bus needs one forward transform, but their filtered spectra are
/// summed before the two shared ear transforms. This avoids performing two
/// identical-shape inverse FFTs for every HRIR direction.
pub struct StereoConvolverBank {
    block_size: usize,
    fft_size: usize,
    forward: Arc<dyn RealToComplex<f32>>,
    inverse: Arc<dyn ComplexToReal<f32>>,
    buses: Vec<ConvolverBus>,
    input: Vec<f32>,
    spectrum_left: Vec<Complex32>,
    spectrum_right: Vec<Complex32>,
    time_left: Vec<f32>,
    time_right: Vec<f32>,
    overlap_left: Vec<f32>,
    overlap_right: Vec<f32>,
    forward_scratch: Vec<Complex32>,
    inverse_scratch: Vec<Complex32>,
}

impl StereoConvolverBank {
    /// Prepares one partitioned HRIR pair per input bus.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty filter, no filters, or an invalid block size.
    pub fn new<'a>(
        filters: impl IntoIterator<Item = (&'a [f32], &'a [f32])>,
        block_size: usize,
    ) -> Result<Self, AppError> {
        if !block_size.is_power_of_two() {
            return Err(AppError::InvalidHrir(
                "convolution block size must be a non-zero power of two".into(),
            ));
        }
        let fft_size = block_size * 2;
        let mut planner = RealFftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft_size);
        let inverse = planner.plan_fft_inverse(fft_size);
        let bins = fft_size / 2 + 1;
        let buses = filters
            .into_iter()
            .map(|(left, right)| {
                if left.is_empty() || right.is_empty() {
                    return Err(AppError::InvalidHrir(
                        "HRIR impulses must not be empty".into(),
                    ));
                }
                let partitions = left.len().max(right.len()).div_ceil(block_size);
                Ok(ConvolverBus {
                    filters_left: pad_partitions(
                        partition_impulse(left, block_size, &forward)?,
                        partitions,
                        bins,
                    ),
                    filters_right: pad_partitions(
                        partition_impulse(right, block_size, &forward)?,
                        partitions,
                        bins,
                    ),
                    history: vec![vec![Complex32::default(); bins]; partitions],
                    history_head: 0,
                })
            })
            .collect::<Result<Vec<_>, AppError>>()?;
        if buses.is_empty() {
            return Err(AppError::InvalidHrir(
                "binaural convolver bank has no filters".into(),
            ));
        }

        Ok(Self {
            block_size,
            fft_size,
            forward_scratch: forward.make_scratch_vec(),
            inverse_scratch: inverse.make_scratch_vec(),
            forward,
            inverse,
            buses,
            input: vec![0.0; fft_size],
            spectrum_left: vec![Complex32::default(); bins],
            spectrum_right: vec![Complex32::default(); bins],
            time_left: vec![0.0; fft_size],
            time_right: vec![0.0; fft_size],
            overlap_left: vec![0.0; block_size],
            overlap_right: vec![0.0; block_size],
        })
    }

    /// Processes one block from every enabled bus into the shared stereo output.
    ///
    /// # Errors
    ///
    /// Returns an error for inconsistent dimensions or an FFT failure.
    #[allow(clippy::cast_precision_loss)] // FFT sizes are bounded by addressable buffers.
    pub fn process(
        &mut self,
        inputs: &[Vec<f32>],
        enabled: &[bool],
        output_left: &mut [f32],
        output_right: &mut [f32],
    ) -> Result<(), AppError> {
        if inputs.len() != self.buses.len()
            || enabled.len() != self.buses.len()
            || inputs.iter().any(|input| input.len() != self.block_size)
            || output_left.len() != self.block_size
            || output_right.len() != self.block_size
        {
            return Err(AppError::Render(
                "binaural convolver bank dimensions do not match".into(),
            ));
        }

        self.spectrum_left.fill(Complex32::default());
        self.spectrum_right.fill(Complex32::default());
        for (bus_index, bus) in self.buses.iter_mut().enumerate() {
            if !enabled[bus_index] {
                continue;
            }
            self.input.fill(0.0);
            self.input[..self.block_size].copy_from_slice(&inputs[bus_index]);
            self.forward
                .process_with_scratch(
                    &mut self.input,
                    &mut bus.history[bus.history_head],
                    &mut self.forward_scratch,
                )
                .map_err(|error| AppError::Render(format!("forward FFT failed: {error}")))?;
            for partition in 0..bus.filters_left.len() {
                let history_index =
                    (bus.history_head + bus.history.len() - partition) % bus.history.len();
                for bin in 0..self.spectrum_left.len() {
                    let input_bin = bus.history[history_index][bin];
                    self.spectrum_left[bin] += input_bin * bus.filters_left[partition][bin];
                    self.spectrum_right[bin] += input_bin * bus.filters_right[partition][bin];
                }
            }
            bus.history_head = (bus.history_head + 1) % bus.history.len();
        }

        prepare_real_inverse_spectrum(&mut self.spectrum_left)?;
        prepare_real_inverse_spectrum(&mut self.spectrum_right)?;
        self.inverse
            .process_with_scratch(
                &mut self.spectrum_left,
                &mut self.time_left,
                &mut self.inverse_scratch,
            )
            .map_err(|error| AppError::Render(format!("left inverse FFT failed: {error}")))?;
        self.inverse
            .process_with_scratch(
                &mut self.spectrum_right,
                &mut self.time_right,
                &mut self.inverse_scratch,
            )
            .map_err(|error| AppError::Render(format!("right inverse FFT failed: {error}")))?;

        let scale = 1.0 / self.fft_size as f32;
        for sample in 0..self.block_size {
            output_left[sample] = self.time_left[sample] * scale + self.overlap_left[sample];
            output_right[sample] = self.time_right[sample] * scale + self.overlap_right[sample];
            self.overlap_left[sample] = self.time_left[sample + self.block_size] * scale;
            self.overlap_right[sample] = self.time_right[sample + self.block_size] * scale;
        }
        Ok(())
    }
}

fn prepare_real_inverse_spectrum(spectrum: &mut [Complex32]) -> Result<(), AppError> {
    let last = spectrum.len().checked_sub(1).ok_or_else(|| {
        AppError::Render("real inverse FFT received an empty frequency spectrum".into())
    })?;
    for index in [0, last] {
        let endpoint = &mut spectrum[index];
        if !endpoint.re.is_finite() || !endpoint.im.is_finite() {
            return Err(AppError::Render(
                "real inverse FFT spectrum has a non-finite endpoint".into(),
            ));
        }
        // DC and Nyquist are their own conjugates and therefore have no
        // imaginary component. Long accumulations can leave a rounding
        // residue here on some FFT backends, so restore the exact invariant
        // required by the real inverse transform.
        endpoint.im = 0.0;
    }
    Ok(())
}

fn partition_impulse(
    impulse: &[f32],
    block_size: usize,
    forward: &Arc<dyn RealToComplex<f32>>,
) -> Result<Vec<Vec<Complex32>>, AppError> {
    let fft_size = block_size * 2;
    let mut scratch = forward.make_scratch_vec();
    impulse
        .chunks(block_size)
        .map(|partition| {
            let mut time = vec![0.0; fft_size];
            time[..partition.len()].copy_from_slice(partition);
            let mut spectrum = forward.make_output_vec();
            forward
                .process_with_scratch(&mut time, &mut spectrum, &mut scratch)
                .map_err(|error| AppError::InvalidHrir(format!("HRIR FFT failed: {error}")))?;
            Ok(spectrum)
        })
        .collect()
}

fn pad_partitions(
    mut partitions: Vec<Vec<Complex32>>,
    count: usize,
    bins: usize,
) -> Vec<Vec<Complex32>> {
    partitions.resize_with(count, || vec![Complex32::default(); bins]);
    partitions
}

#[derive(Clone, Debug)]
pub struct PeakLimiter {
    gain: f32,
    ceiling: f32,
    lookahead_frames: usize,
    attack_coefficient: f32,
    release_coefficient: f32,
    true_peak: TruePeakDetector,
    delayed: VecDeque<[f32; 2]>,
    peak_window: VecDeque<(u64, f32)>,
    frame_index: u64,
}

impl Default for PeakLimiter {
    fn default() -> Self {
        Self::new(48_000)
    }
}

impl PeakLimiter {
    const RELEASE_SECONDS: f32 = 0.100;
    const CEILING_DBFS: f32 = -1.0;

    #[must_use]
    #[allow(clippy::cast_precision_loss)] // Audio sample rates fit exactly enough for time constants.
    pub fn new(sample_rate: u32) -> Self {
        let rate = sample_rate.max(1) as f32;
        let envelope_lookahead = usize::try_from(sample_rate.div_ceil(200))
            .unwrap_or(usize::MAX)
            .max(1);
        let true_peak = TruePeakDetector::new();
        let lookahead_frames =
            envelope_lookahead.saturating_add(TruePeakDetector::latency_frames());
        let attack_coefficient = 0.001_f32.powf(1.0 / envelope_lookahead as f32);
        let release_coefficient = (-1.0 / (Self::RELEASE_SECONDS * rate)).exp();
        Self {
            gain: 1.0,
            ceiling: 10_f32.powf(Self::CEILING_DBFS / 20.0),
            lookahead_frames,
            attack_coefficient,
            release_coefficient,
            true_peak,
            delayed: VecDeque::with_capacity(lookahead_frames + 1),
            peak_window: VecDeque::with_capacity(lookahead_frames + 1),
            frame_index: 0,
        }
    }

    /// Applies a linked-stereo lookahead envelope.
    ///
    /// The returned block can be shorter than the input while the lookahead
    /// fills. Feeding the same number of zero frames drains the delayed audio.
    pub fn process(&mut self, interleaved_stereo: &[f32], output: &mut Vec<f32>) {
        debug_assert_eq!(interleaved_stereo.len() % 2, 0);
        output.clear();
        output.reserve(interleaved_stereo.len());
        for frame in interleaved_stereo.chunks_exact(2) {
            let stereo_frame = [frame[0], frame[1]];
            let peak = frame[0]
                .abs()
                .max(frame[1].abs())
                .max(self.true_peak.push(stereo_frame));
            while self
                .peak_window
                .back()
                .is_some_and(|(_, previous)| *previous <= peak)
            {
                self.peak_window.pop_back();
            }
            self.peak_window.push_back((self.frame_index, peak));
            let oldest = self
                .frame_index
                .saturating_sub(u64::try_from(self.lookahead_frames).unwrap_or(u64::MAX));
            while self
                .peak_window
                .front()
                .is_some_and(|(index, _)| *index < oldest)
            {
                self.peak_window.pop_front();
            }

            let window_peak = self.peak_window.front().map_or(0.0, |(_, peak)| *peak);
            // The small margin covers the finite settling error of the
            // lookahead attack while retaining a -1 dBFS output ceiling.
            let target = if window_peak > self.ceiling {
                self.ceiling * 0.999 / window_peak
            } else {
                1.0
            };
            let coefficient = if target < self.gain {
                self.attack_coefficient
            } else {
                self.release_coefficient
            };
            self.gain = target + coefficient * (self.gain - target);

            self.delayed.push_back(stereo_frame);
            if self.delayed.len() > self.lookahead_frames
                && let Some(delayed) = self.delayed.pop_front()
            {
                output.push(delayed[0] * self.gain);
                output.push(delayed[1] * self.gain);
            }
            self.frame_index = self.frame_index.saturating_add(1);
        }
    }

    /// Emits every delayed input frame without adding silence to the result.
    pub fn drain(&mut self, output: &mut Vec<f32>) {
        let silence = vec![0.0; self.lookahead_frames * 2];
        self.process(&silence, output);
    }
}

/// Four-times oversampled stereo peak detector.
///
/// A 24-tap Lanczos-windowed sinc reconstructs the three intersample phases
/// between each pair of PCM samples. The 12-frame detector latency is absorbed
/// by the limiter's existing lookahead.
#[derive(Clone, Debug)]
struct TruePeakDetector {
    history: VecDeque<[f32; 2]>,
    coefficients: [[f32; Self::TAPS]; 3],
}

impl TruePeakDetector {
    const RADIUS: usize = 12;
    const TAPS: usize = Self::RADIUS * 2;
    const PHASES: [f32; 3] = [0.25, 0.5, 0.75];

    fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(Self::TAPS),
            coefficients: Self::PHASES.map(interpolation_coefficients),
        }
    }

    const fn latency_frames() -> usize {
        Self::RADIUS
    }

    fn push(&mut self, frame: [f32; 2]) -> f32 {
        self.history.push_back(frame);
        if self.history.len() < Self::TAPS {
            return 0.0;
        }
        if self.history.len() > Self::TAPS {
            self.history.pop_front();
        }

        let mut peak = 0.0_f32;
        for coefficients in &self.coefficients {
            for ear in 0..2 {
                let sample = self
                    .history
                    .iter()
                    .zip(coefficients)
                    .map(|(frame, coefficient)| frame[ear] * coefficient)
                    .sum::<f32>();
                peak = peak.max(sample.abs());
            }
        }
        peak
    }
}

#[allow(clippy::cast_precision_loss)] // The fixed 24-tap index is exactly represented.
fn interpolation_coefficients(phase: f32) -> [f32; TruePeakDetector::TAPS] {
    let mut coefficients = [0.0; TruePeakDetector::TAPS];
    for (tap, coefficient) in coefficients.iter_mut().enumerate() {
        let relative_sample = tap as f32 - (TruePeakDetector::RADIUS.saturating_sub(1)) as f32;
        let distance = phase - relative_sample;
        *coefficient = sinc(distance) * sinc(distance / TruePeakDetector::RADIUS as f32);
    }
    let dc_gain = coefficients.iter().sum::<f32>();
    for coefficient in &mut coefficients {
        *coefficient /= dc_gain;
    }
    coefficients
}

fn sinc(value: f32) -> f32 {
    if value.abs() <= f32::EPSILON {
        1.0
    } else {
        let angle = std::f32::consts::PI * value;
        angle.sin() / angle
    }
}

#[derive(Clone, Debug)]
pub struct TpdfDither {
    state: u64,
}

impl Default for TpdfDither {
    fn default() -> Self {
        Self {
            state: 0x6a09_e667_f3bc_c909,
        }
    }
}

impl TpdfDither {
    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // The value is rounded and clamped to i16 first.
    pub fn quantize_i16(&mut self, sample: f32) -> i16 {
        let noise = self.uniform() - self.uniform();
        let scaled = sample.clamp(-1.0, 1.0) * f32::from(i16::MAX) + noise;
        scaled
            .round()
            .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // Rounded and clamped to the signed 24-bit range.
    pub fn quantize_i24(&mut self, sample: f32) -> i32 {
        const MAX: f32 = 8_388_607.0;
        const MIN: f32 = -8_388_608.0;
        let noise = self.uniform() - self.uniform();
        (sample.clamp(-1.0, 1.0) * MAX + noise)
            .round()
            .clamp(MIN, MAX) as i32
    }

    #[allow(clippy::cast_precision_loss)] // Only the upper 24 bits are retained, exactly fitting f32.
    fn uniform(&mut self) -> f32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        let value = u32::try_from(self.state >> 40).unwrap_or_default();
        value as f32 / 16_777_216.0
    }
}

#[cfg(test)]
mod tests {
    use super::{PeakLimiter, StereoConvolver, StereoConvolverBank, TpdfDither, TruePeakDetector};

    #[test]
    fn identity_impulse_is_continuous_across_blocks() {
        let mut convolver = StereoConvolver::new(&[1.0], &[0.5], 4).unwrap();
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        convolver
            .process(&[1.0, 2.0, 3.0, 4.0], &mut left, &mut right)
            .unwrap();
        assert_samples(&left, &[1.0, 2.0, 3.0, 4.0]);
        assert_samples(&right, &[0.5, 1.0, 1.5, 2.0]);
        convolver
            .process(&[5.0, 6.0], &mut left, &mut right)
            .unwrap();
        assert_samples(&left, &[5.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn overlap_is_carried_to_the_next_block() {
        let mut convolver = StereoConvolver::new(&[0.0, 0.0, 0.0, 0.0, 1.0], &[1.0], 4).unwrap();
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        convolver.process(&[1.0], &mut left, &mut right).unwrap();
        assert_samples(&left, &[0.0; 4]);
        convolver.process(&[], &mut left, &mut right).unwrap();
        assert_samples(&left, &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn convolver_bank_matches_summed_independent_buses() {
        let filters = [
            (vec![1.0, 0.25], vec![0.5, -0.1]),
            (vec![-0.2, 0.4], vec![0.3, 0.15]),
        ];
        let inputs = vec![vec![1.0, 0.5, -0.25, 0.0], vec![0.2, -0.4, 0.8, 0.1]];
        let mut bank = StereoConvolverBank::new(
            filters
                .iter()
                .map(|(left, right)| (left.as_slice(), right.as_slice())),
            4,
        )
        .unwrap();
        let mut bank_left = [0.0; 4];
        let mut bank_right = [0.0; 4];
        bank.process(&inputs, &[true, true], &mut bank_left, &mut bank_right)
            .unwrap();

        let mut expected_left = [0.0; 4];
        let mut expected_right = [0.0; 4];
        for ((left_filter, right_filter), input) in filters.iter().zip(&inputs) {
            let mut convolver = StereoConvolver::new(left_filter, right_filter, 4).unwrap();
            let mut left = [0.0; 4];
            let mut right = [0.0; 4];
            convolver.process(input, &mut left, &mut right).unwrap();
            for sample in 0..4 {
                expected_left[sample] += left[sample];
                expected_right[sample] += right[sample];
            }
        }

        assert_samples(&bank_left, &expected_left);
        assert_samples(&bank_right, &expected_right);
    }

    #[test]
    fn inverse_fft_accepts_rounding_residue_at_real_spectrum_endpoints() {
        let filters = [(vec![1.0], vec![0.5])];
        let mut bank = StereoConvolverBank::new(
            filters
                .iter()
                .map(|(left, right)| (left.as_slice(), right.as_slice())),
            4,
        )
        .unwrap();
        let last = bank.buses[0].filters_left[0].len() - 1;
        let bus = &mut bank.buses[0];
        for spectrum in [&mut bus.filters_left[0], &mut bus.filters_right[0]] {
            spectrum[0].im = f32::EPSILON;
            spectrum[last].im = -f32::EPSILON;
        }

        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        bank.process(&[vec![1.0, 0.0, 0.0, 0.0]], &[true], &mut left, &mut right)
            .unwrap();
        assert_samples(&left, &[1.0, 0.0, 0.0, 0.0]);
        assert_samples(&right, &[0.5, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn limiter_is_block_independent_and_stays_below_its_ceiling() {
        let mut samples = vec![0.25; 80];
        samples[40] = 2.0;
        samples[41] = -2.0;

        let mut whole = PeakLimiter::new(1_000);
        let mut whole_output = Vec::new();
        whole.process(&samples, &mut whole_output);
        let mut drained = Vec::new();
        whole.drain(&mut drained);
        whole_output.extend(drained);

        let mut chunked = PeakLimiter::new(1_000);
        let mut chunked_output = Vec::new();
        for chunk in samples.chunks(14) {
            let mut output = Vec::new();
            chunked.process(chunk, &mut output);
            chunked_output.extend(output);
        }
        let mut chunked_tail = Vec::new();
        chunked.drain(&mut chunked_tail);
        chunked_output.extend(chunked_tail);

        assert_samples(&chunked_output, &whole_output);
        assert_eq!(whole_output.len(), samples.len());
        assert!(
            whole_output
                .iter()
                .all(|sample| sample.abs() <= 10_f32.powf(-1.0 / 20.0))
        );
    }

    #[test]
    fn detector_finds_intersample_peaks() {
        let mut detector = TruePeakDetector::new();
        let mut peak = 0.0_f32;
        for frame in 0..128 {
            let sample = if frame % 4 < 2 { 1.0 } else { -1.0 };
            peak = peak.max(detector.push([sample, sample]));
        }
        assert!(peak > 1.35, "detected peak was {peak}");
    }

    #[test]
    fn limiter_controls_intersample_peaks() {
        let samples = (0..2_048)
            .flat_map(|frame| {
                let sample = if frame % 4 < 2 { 1.0 } else { -1.0 };
                [sample, sample]
            })
            .collect::<Vec<_>>();
        let mut limiter = PeakLimiter::new(48_000);
        let mut output = Vec::new();
        limiter.process(&samples, &mut output);
        let mut tail = Vec::new();
        limiter.drain(&mut tail);
        output.extend(tail);

        let mut detector = TruePeakDetector::new();
        let peak = output
            .chunks_exact(2)
            .map(|frame| detector.push([frame[0], frame[1]]))
            .fold(0.0, f32::max);
        let ceiling = 10_f32.powf(-1.0 / 20.0);
        assert!(peak <= ceiling * 1.001, "limited true peak was {peak}");
    }

    #[test]
    fn dither_quantization_is_bounded() {
        let mut dither = TpdfDither::default();
        assert!(dither.quantize_i16(2.0) >= 32_766);
        assert!(dither.quantize_i16(-2.0) <= -32_766);
        assert!(dither.quantize_i24(2.0) >= 8_388_606);
        assert!(dither.quantize_i24(-2.0) <= -8_388_607);
    }

    fn assert_samples(actual: &[f32], expected: &[f32]) {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "expected {expected}, found {actual}"
            );
        }
    }
}
