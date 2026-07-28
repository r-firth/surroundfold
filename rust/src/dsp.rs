use std::sync::Arc;

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
    release_seconds: f32,
    ceiling: f32,
}

impl Default for PeakLimiter {
    fn default() -> Self {
        Self {
            gain: 1.0,
            release_seconds: 0.100,
            ceiling: 0.98,
        }
    }
}

impl PeakLimiter {
    #[allow(clippy::cast_precision_loss)] // Audio block sizes and sample rates are small integers.
    pub fn process(&mut self, interleaved_stereo: &mut [f32], sample_rate: u32) {
        let peak = interleaved_stereo
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0, f32::max);
        let target = if peak > self.ceiling {
            self.ceiling / peak
        } else {
            1.0
        };
        if target < self.gain {
            self.gain = target;
        }
        for sample in interleaved_stereo.iter_mut() {
            *sample *= self.gain;
        }

        let frames = interleaved_stereo.len() / 2;
        let release_frames = self.release_seconds * sample_rate as f32;
        if release_frames > 0.0 {
            let recovered = 1.0 - (-(frames as f32) / release_frames).exp();
            self.gain += (1.0 - self.gain) * recovered;
        }
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
    use super::{PeakLimiter, StereoConvolver, TpdfDither};

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
    fn limiter_recovers_without_exceeding_ceiling() {
        let mut limiter = PeakLimiter::default();
        let mut samples = [2.0, -2.0, 0.5, -0.5];
        limiter.process(&mut samples, 48_000);
        assert!(samples.iter().all(|sample| sample.abs() <= 0.98));
    }

    #[test]
    fn dither_quantization_is_bounded() {
        let mut dither = TpdfDither::default();
        assert!(dither.quantize_i16(2.0) >= 32_766);
        assert!(dither.quantize_i16(-2.0) <= -32_766);
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
