use std::sync::{Arc, OnceLock};

use rustfft::FftPlanner;
#[cfg(target_arch = "x86_64")]
use wide::f32x8;

use crate::parametric::{MAXIMUM_ITD_SECONDS, ParametricHrtfModel, minimum_phase_impulse};

const SYNTHESIS_FFT: usize = 512;
const FILTER_LENGTH: usize = 64;
const AZIMUTH_STEP_DEGREES: f32 = 5.0;
const AZIMUTH_COUNT: usize = 72;
const ELEVATION_STEP_DEGREES: f32 = 10.0;
const ELEVATION_COUNT: usize = 19;
const INTERPOLATION_GUARD_FRAMES: f32 = 1.0;
const FRACTIONAL_DELAY_TAPS: usize = 24;
const FRACTIONAL_DELAY_PRE_SAMPLES: usize = 11;
const FRACTIONAL_DELAY_POST_SAMPLES: usize =
    FRACTIONAL_DELAY_TAPS - FRACTIONAL_DELAY_PRE_SAMPLES - 1;
const FRACTIONAL_DELAY_PHASES: usize = 8_192;
const FRACTIONAL_DELAY_KAISER_BETA: f32 = 8.6;
const WOODWORTH_MAXIMUM_ANGLE_TERM: f32 = std::f32::consts::FRAC_PI_2 + 1.0;
pub(crate) const FRACTIONAL_DELAY_GUARD_FRAMES: usize = FRACTIONAL_DELAY_PRE_SAMPLES;

struct GridFilter {
    left: [f32; FILTER_LENGTH],
    right: [f32; FILTER_LENGTH],
}

pub(crate) struct ContinuousHrtfGrid {
    sample_rate: u32,
    common_delay: usize,
    filters: Vec<GridFilter>,
}

pub(crate) struct ContinuousTarget {
    left: [f32; FILTER_LENGTH],
    right: [f32; FILTER_LENGTH],
    left_delay: f32,
    right_delay: f32,
}

impl ContinuousHrtfGrid {
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn new(model: &ParametricHrtfModel) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(SYNTHESIS_FFT);
        let inverse = planner.plan_fft_inverse(SYNTHESIS_FFT);
        let mut filters = Vec::with_capacity(AZIMUTH_COUNT * ELEVATION_COUNT);
        for elevation_index in 0..ELEVATION_COUNT {
            let elevation = (-90.0 + elevation_index as f32 * ELEVATION_STEP_DEGREES).to_radians();
            for azimuth_index in 0..AZIMUTH_COUNT {
                let azimuth = (-180.0 + azimuth_index as f32 * AZIMUTH_STEP_DEGREES).to_radians();
                let horizontal = elevation.cos();
                let direction = [
                    azimuth.sin() * horizontal,
                    azimuth.cos() * horizontal,
                    elevation.sin(),
                ];
                let (left_magnitude, right_magnitude) = model.magnitudes(direction, SYNTHESIS_FFT);
                let synthesize = |magnitude| {
                    minimum_phase_impulse(
                        magnitude,
                        SYNTHESIS_FFT,
                        FILTER_LENGTH,
                        forward.as_ref(),
                        inverse.as_ref(),
                    )
                    .try_into()
                    .expect("continuous HRTF synthesis returned the requested filter length")
                };
                filters.push(GridFilter {
                    left: synthesize(&left_magnitude),
                    right: synthesize(&right_magnitude),
                });
            }
        }
        Self {
            sample_rate: model.sample_rate(),
            common_delay: model.common_delay(),
            filters,
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    pub(crate) fn target(&self, direction: [f32; 3]) -> ContinuousTarget {
        let direction = normalized(direction);
        let azimuth_degrees = direction[0].atan2(direction[1]).to_degrees();
        let azimuth_position = (azimuth_degrees + 180.0) / AZIMUTH_STEP_DEGREES;
        let azimuth_lower = azimuth_position.floor() as usize % AZIMUTH_COUNT;
        let azimuth_upper = (azimuth_lower + 1) % AZIMUTH_COUNT;
        let azimuth_fraction = azimuth_position - azimuth_position.floor();

        let elevation_degrees = direction[2].asin().to_degrees();
        let elevation_position = ((elevation_degrees + 90.0) / ELEVATION_STEP_DEGREES)
            .clamp(0.0, (ELEVATION_COUNT - 1) as f32);
        let elevation_lower = elevation_position.floor() as usize;
        let elevation_upper = elevation_lower.saturating_add(1).min(ELEVATION_COUNT - 1);
        let elevation_fraction = elevation_position - elevation_lower as f32;

        let lower_left = self.filter(elevation_lower, azimuth_lower);
        let lower_right = self.filter(elevation_lower, azimuth_upper);
        let upper_left = self.filter(elevation_upper, azimuth_lower);
        let upper_right = self.filter(elevation_upper, azimuth_upper);
        let interpolate = |lower_left: f32, lower_right: f32, upper_left: f32, upper_right: f32| {
            let lower = lower_left.mul_add(1.0 - azimuth_fraction, lower_right * azimuth_fraction);
            let upper = upper_left.mul_add(1.0 - azimuth_fraction, upper_right * azimuth_fraction);
            lower.mul_add(1.0 - elevation_fraction, upper * elevation_fraction)
        };
        let left = std::array::from_fn(|index| {
            interpolate(
                lower_left.left[index],
                lower_right.left[index],
                upper_left.left[index],
                upper_right.left[index],
            )
        });
        let right = std::array::from_fn(|index| {
            interpolate(
                lower_left.right[index],
                lower_right.right[index],
                upper_left.right[index],
                upper_right.right[index],
            )
        });

        // Projection onto the interaural axis naturally weakens lateral cues
        // with elevation and has no singularity at either vertical pole.
        let lateral = direction[0].clamp(-1.0, 1.0);
        let itd = woodworth_itd_seconds(lateral) * self.sample_rate as f32;
        let common = (self.common_delay as f32 + INTERPOLATION_GUARD_FRAMES)
            .max(FRACTIONAL_DELAY_GUARD_FRAMES as f32);
        ContinuousTarget {
            left,
            right,
            left_delay: common + itd.max(0.0),
            right_delay: common + (-itd).max(0.0),
        }
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // The bounded sub-millisecond delay is always a small positive frame count.
    pub(crate) fn maximum_tail_frames(&self) -> usize {
        let itd = (f64::from(MAXIMUM_ITD_SECONDS) * f64::from(self.sample_rate)).ceil() as usize;
        FILTER_LENGTH
            .saturating_add(self.common_delay)
            .saturating_add(itd)
            .saturating_add(FRACTIONAL_DELAY_TAPS)
            .saturating_add(4)
    }

    fn filter(&self, elevation: usize, azimuth: usize) -> &GridFilter {
        &self.filters[elevation * AZIMUTH_COUNT + azimuth]
    }
}

pub(crate) struct ContinuousBinaural {
    grid: Arc<ContinuousHrtfGrid>,
    history: Vec<f32>,
    cursor: usize,
    left: [f32; FILTER_LENGTH],
    right: [f32; FILTER_LENGTH],
    left_step: [f32; FILTER_LENGTH],
    right_step: [f32; FILTER_LENGTH],
    left_delay: f32,
    right_delay: f32,
    left_delay_step: f32,
    right_delay_step: f32,
    filter_moving: bool,
    delay_moving: bool,
    remaining: usize,
    active_tail: usize,
    tail_frames: usize,
    delay_line: StereoFractionalDelay,
}

impl ContinuousBinaural {
    #[allow(clippy::cast_possible_truncation)]
    pub(crate) fn new(grid: Arc<ContinuousHrtfGrid>, direction: [f32; 3]) -> Self {
        let target = grid.target(direction);
        let tail_frames = grid.maximum_tail_frames();
        let delay_capacity = tail_frames.max(8);
        Self {
            grid,
            history: vec![0.0; FILTER_LENGTH * 2],
            cursor: 0,
            left_step: [0.0; FILTER_LENGTH],
            right_step: [0.0; FILTER_LENGTH],
            left: target.left,
            right: target.right,
            left_delay: target.left_delay,
            right_delay: target.right_delay,
            left_delay_step: 0.0,
            right_delay_step: 0.0,
            filter_moving: false,
            delay_moving: false,
            remaining: 0,
            active_tail: 0,
            tail_frames,
            delay_line: StereoFractionalDelay::new(delay_capacity),
        }
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn set_direction(&mut self, direction: [f32; 3], frames: usize) {
        let target = self.grid.target(direction);
        if frames == 0 {
            self.left = target.left;
            self.right = target.right;
            self.left_step.fill(0.0);
            self.right_step.fill(0.0);
            self.left_delay = target.left_delay;
            self.right_delay = target.right_delay;
            self.left_delay_step = 0.0;
            self.right_delay_step = 0.0;
            self.filter_moving = false;
            self.delay_moving = false;
            self.remaining = 0;
            return;
        }
        let duration = frames as f32;
        for ((step, current), target) in self.left_step.iter_mut().zip(&self.left).zip(&target.left)
        {
            *step = (*target - *current) / duration;
        }
        for ((step, current), target) in self
            .right_step
            .iter_mut()
            .zip(&self.right)
            .zip(&target.right)
        {
            *step = (*target - *current) / duration;
        }
        self.left_delay_step = (target.left_delay - self.left_delay) / duration;
        self.right_delay_step = (target.right_delay - self.right_delay) / duration;
        self.filter_moving = self
            .left_step
            .iter()
            .chain(&self.right_step)
            .any(|step| *step != 0.0);
        self.delay_moving = self.left_delay_step != 0.0 || self.right_delay_step != 0.0;
        self.remaining = frames;
    }

    pub(crate) fn process(&mut self, input: f32) -> [f32; 2] {
        if input == 0.0 && self.active_tail == 0 {
            self.advance();
            return [0.0; 2];
        }
        if input == 0.0 {
            self.active_tail -= 1;
        } else {
            self.active_tail = self.tail_frames;
        }
        self.cursor = if self.cursor == 0 {
            FILTER_LENGTH - 1
        } else {
            self.cursor - 1
        };
        self.history[self.cursor] = input;
        self.history[self.cursor + FILTER_LENGTH] = input;
        let samples = &self.history[self.cursor..self.cursor + FILTER_LENGTH];
        let [left, right] = stereo_dot(samples, &self.left, &self.right);
        let output = self
            .delay_line
            .process([left, right], [self.left_delay, self.right_delay]);
        self.advance();
        output
    }

    #[must_use]
    pub(crate) fn tail_frames(&self) -> usize {
        self.tail_frames
    }

    fn advance(&mut self) {
        if self.remaining == 0 {
            return;
        }
        if self.filter_moving {
            for (value, step) in self.left.iter_mut().zip(&self.left_step) {
                *value += step;
            }
            for (value, step) in self.right.iter_mut().zip(&self.right_step) {
                *value += step;
            }
        }
        if self.delay_moving {
            self.left_delay += self.left_delay_step;
            self.right_delay += self.right_delay_step;
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            self.left_step.fill(0.0);
            self.right_step.fill(0.0);
            self.left_delay_step = 0.0;
            self.right_delay_step = 0.0;
            self.filter_moving = false;
            self.delay_moving = false;
        }
    }
}

fn woodworth_itd_seconds(lateral: f32) -> f32 {
    let lateral = lateral.clamp(-1.0, 1.0);
    MAXIMUM_ITD_SECONDS * (lateral.asin() + lateral) / WOODWORTH_MAXIMUM_ANGLE_TERM
}

#[cfg(not(target_arch = "x86_64"))]
fn stereo_dot(samples: &[f32], left: &[f32], right: &[f32]) -> [f32; 2] {
    let mut output = [0.0; 2];
    for ((sample, left), right) in samples.iter().zip(left).zip(right) {
        output[0] += sample * left;
        output[1] += sample * right;
    }
    output
}

#[cfg(target_arch = "x86_64")]
fn stereo_dot(samples: &[f32], left: &[f32], right: &[f32]) -> [f32; 2] {
    let mut left_sum = [f32x8::ZERO; 2];
    let mut right_sum = [f32x8::ZERO; 2];
    for tap in (0..FILTER_LENGTH).step_by(16) {
        for lane in 0..2 {
            let start = tap + lane * 8;
            let end = start + 8;
            let samples = f32x8::new(samples[start..end].try_into().expect("eight samples"));
            let left = f32x8::new(left[start..end].try_into().expect("eight left taps"));
            let right = f32x8::new(right[start..end].try_into().expect("eight right taps"));
            left_sum[lane] = samples.mul_add(left, left_sum[lane]);
            right_sum[lane] = samples.mul_add(right, right_sum[lane]);
        }
    }
    [
        (left_sum[0] + left_sum[1]).reduce_add(),
        (right_sum[0] + right_sum[1]).reduce_add(),
    ]
}

struct StereoFractionalDelay {
    left: Vec<f32>,
    right: Vec<f32>,
    capacity: usize,
    cursor: usize,
    table: &'static [[f32; FRACTIONAL_DELAY_TAPS]],
    cache: [FractionalDelayCache; 2],
}

struct FractionalDelayCache {
    delay: u32,
    whole: usize,
    kernel: &'static [f32; FRACTIONAL_DELAY_TAPS],
}

impl StereoFractionalDelay {
    fn new(capacity: usize) -> Self {
        let table = fractional_delay_table();
        Self {
            left: vec![0.0; capacity * 2],
            right: vec![0.0; capacity * 2],
            capacity,
            cursor: 0,
            table,
            cache: std::array::from_fn(|_| FractionalDelayCache {
                delay: u32::MAX,
                whole: 0,
                kernel: &table[0],
            }),
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn process(&mut self, input: [f32; 2], delay: [f32; 2]) -> [f32; 2] {
        self.left[self.cursor] = input[0];
        self.left[self.cursor + self.capacity] = input[0];
        self.right[self.cursor] = input[1];
        self.right[self.cursor + self.capacity] = input[1];
        let minimum = FRACTIONAL_DELAY_GUARD_FRAMES as f32;
        let maximum = self
            .capacity
            .saturating_sub(FRACTIONAL_DELAY_POST_SAMPLES + 2) as f32;
        for (cache, delay) in self.cache.iter_mut().zip(delay) {
            let delay = delay.clamp(minimum, maximum.max(minimum));
            let delay_bits = delay.to_bits();
            if cache.delay != delay_bits {
                cache.whole = delay.floor() as usize;
                let phase = fractional_delay_phase(delay - cache.whole as f32);
                cache.kernel = &self.table[phase];
                cache.delay = delay_bits;
            }
        }
        let left_first = self.cursor + self.cache[0].whole - FRACTIONAL_DELAY_PRE_SAMPLES;
        let right_first = self.cursor + self.cache[1].whole - FRACTIONAL_DELAY_PRE_SAMPLES;
        let left = &self.left[left_first..left_first + FRACTIONAL_DELAY_TAPS];
        let right = &self.right[right_first..right_first + FRACTIONAL_DELAY_TAPS];
        let output = stereo_fractional_dot(left, right, self.cache[0].kernel, self.cache[1].kernel);
        self.cursor = if self.cursor == 0 {
            self.capacity - 1
        } else {
            self.cursor - 1
        };
        output
    }
}

#[cfg(not(target_arch = "x86_64"))]
fn stereo_fractional_dot(
    left: &[f32],
    right: &[f32],
    left_kernel: &[f32],
    right_kernel: &[f32],
) -> [f32; 2] {
    let mut output = [0.0; 2];
    for tap in 0..FRACTIONAL_DELAY_TAPS {
        output[0] = left[tap].mul_add(left_kernel[tap], output[0]);
        output[1] = right[tap].mul_add(right_kernel[tap], output[1]);
    }
    output
}

#[cfg(target_arch = "x86_64")]
fn stereo_fractional_dot(
    left: &[f32],
    right: &[f32],
    left_kernel: &[f32],
    right_kernel: &[f32],
) -> [f32; 2] {
    let mut left_sum = [f32x8::ZERO; 3];
    let mut right_sum = [f32x8::ZERO; 3];
    for (chunk, tap) in (0..FRACTIONAL_DELAY_TAPS).step_by(8).enumerate() {
        let end = tap + 8;
        let left = f32x8::new(left[tap..end].try_into().expect("eight left samples"));
        let right = f32x8::new(right[tap..end].try_into().expect("eight right samples"));
        let left_kernel = f32x8::new(left_kernel[tap..end].try_into().expect("eight left taps"));
        let right_kernel = f32x8::new(right_kernel[tap..end].try_into().expect("eight right taps"));
        left_sum[chunk] = left * left_kernel;
        right_sum[chunk] = right * right_kernel;
    }
    [
        ((left_sum[0] + left_sum[1]) + left_sum[2]).reduce_add(),
        ((right_sum[0] + right_sum[1]) + right_sum[2]).reduce_add(),
    ]
}

#[cfg(test)]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn fractional_delay_read_duplicated(
    samples: &[f32],
    capacity: usize,
    cursor: usize,
    delay: f32,
    table: &[[f32; FRACTIONAL_DELAY_TAPS]],
) -> f32 {
    debug_assert_eq!(samples.len(), capacity * 2);
    let minimum = FRACTIONAL_DELAY_GUARD_FRAMES as f32;
    let maximum = capacity.saturating_sub(FRACTIONAL_DELAY_POST_SAMPLES + 2) as f32;
    let delay = delay.clamp(minimum, maximum.max(minimum));
    let whole = delay.floor() as usize;
    let fraction = delay - whole as f32;
    let coefficients = &table[fractional_delay_phase(fraction)];
    fractional_delay_read_prepared(samples, capacity, cursor, whole, coefficients)
}

#[cfg(test)]
fn fractional_delay_read_prepared(
    samples: &[f32],
    capacity: usize,
    cursor: usize,
    whole: usize,
    coefficients: &[f32; FRACTIONAL_DELAY_TAPS],
) -> f32 {
    let newest = cursor + capacity - (whole - FRACTIONAL_DELAY_PRE_SAMPLES);
    let ordered_samples = samples[newest + 1 - FRACTIONAL_DELAY_TAPS..=newest]
        .iter()
        .rev();
    let mut output = 0.0;
    for (sample, coefficient) in ordered_samples.zip(coefficients) {
        output = sample.mul_add(*coefficient, output);
    }
    output
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
pub(crate) fn fractional_delay_read(samples: &[f32], cursor: usize, delay: f32) -> f32 {
    let minimum = FRACTIONAL_DELAY_GUARD_FRAMES as f32;
    let maximum = samples
        .len()
        .saturating_sub(FRACTIONAL_DELAY_POST_SAMPLES + 2) as f32;
    let delay = delay.clamp(minimum, maximum.max(minimum));
    let whole = delay.floor() as usize;
    let fraction = delay - whole as f32;
    let coefficients = &fractional_delay_table()[fractional_delay_phase(fraction)];
    let mut output = 0.0;
    for (tap, coefficient) in coefficients.iter().copied().enumerate() {
        let age = if tap < FRACTIONAL_DELAY_PRE_SAMPLES {
            whole - (FRACTIONAL_DELAY_PRE_SAMPLES - tap)
        } else {
            whole + (tap - FRACTIONAL_DELAY_PRE_SAMPLES)
        };
        debug_assert!(age < samples.len());
        let index = if cursor >= age {
            cursor - age
        } else {
            samples.len() + cursor - age
        };
        let sample = samples[index];
        output = sample.mul_add(coefficient, output);
    }
    output
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn fractional_delay_phase(fraction: f32) -> usize {
    (fraction.clamp(0.0, 1.0) * FRACTIONAL_DELAY_PHASES as f32)
        .round()
        .min(FRACTIONAL_DELAY_PHASES as f32) as usize
}

fn fractional_delay_table() -> &'static [[f32; FRACTIONAL_DELAY_TAPS]] {
    static TABLE: OnceLock<Vec<[f32; FRACTIONAL_DELAY_TAPS]>> = OnceLock::new();
    TABLE.get_or_init(|| {
        (0..=FRACTIONAL_DELAY_PHASES)
            .map(fractional_delay_kernel)
            .collect()
    })
}

#[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
fn fractional_delay_kernel(phase: usize) -> [f32; FRACTIONAL_DELAY_TAPS] {
    let fraction = phase as f32 / FRACTIONAL_DELAY_PHASES as f32;
    let radius = FRACTIONAL_DELAY_TAPS as f32 / 2.0;
    let beta_normalization = bessel_i0(FRACTIONAL_DELAY_KAISER_BETA);
    let mut kernel = std::array::from_fn(|tap| {
        let offset = tap as f32 - FRACTIONAL_DELAY_PRE_SAMPLES as f32 - fraction;
        let position = offset / radius;
        if position.abs() >= 1.0 {
            return 0.0;
        }
        let window = bessel_i0(FRACTIONAL_DELAY_KAISER_BETA * (1.0 - position * position).sqrt())
            / beta_normalization;
        sinc(offset) * window
    });
    let sum = kernel.iter().sum::<f32>().max(f32::EPSILON);
    for coefficient in &mut kernel {
        *coefficient /= sum;
    }
    kernel
}

fn bessel_i0(value: f32) -> f32 {
    let scaled = value * value / 4.0;
    let mut sum = 1.0;
    let mut term = 1.0;
    for order in 1_u8..=16 {
        let order = f32::from(order);
        term *= scaled / (order * order);
        sum += term;
    }
    sum
}

fn sinc(value: f32) -> f32 {
    if value.abs() <= f32::EPSILON {
        1.0
    } else {
        let angle = std::f32::consts::PI * value;
        angle.sin() / angle
    }
}

fn normalized(direction: [f32; 3]) -> [f32; 3] {
    let length = direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    if length > f32::EPSILON {
        direction.map(|value| value / length)
    } else {
        [0.0, 1.0, 0.0]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rustfft::{FftPlanner, num_complex::Complex32};

    use super::{
        ContinuousBinaural, ContinuousHrtfGrid, FRACTIONAL_DELAY_GUARD_FRAMES,
        FRACTIONAL_DELAY_POST_SAMPLES, FRACTIONAL_DELAY_PRE_SAMPLES, FRACTIONAL_DELAY_TAPS,
        SYNTHESIS_FFT, StereoFractionalDelay, fractional_delay_phase, fractional_delay_read,
        fractional_delay_read_duplicated, fractional_delay_table,
    };
    use crate::hrir::{HrirSet, Speaker};
    use crate::{binaural::PanningRoute, parametric::ParametricHrtfModel};

    #[test]
    fn bandlimited_delay_preserves_integer_samples() {
        let mut delay = StereoFractionalDelay::new(64);
        let output = (0..32)
            .map(|index| delay.process([if index == 0 { 1.0 } else { 0.0 }, 0.0], [2.0; 2])[0])
            .collect::<Vec<_>>();
        assert!((output[FRACTIONAL_DELAY_GUARD_FRAMES] - 1.0).abs() < f32::EPSILON);
        assert!(
            output
                .iter()
                .enumerate()
                .all(|(index, sample)| index == FRACTIONAL_DELAY_GUARD_FRAMES
                    || sample.abs() < f32::EPSILON)
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn duplicated_delay_line_is_bit_exact_with_the_ring_reader() {
        let capacity = 64;
        let samples = (0..capacity)
            .map(|index| (index as f32 * 0.371).sin())
            .collect::<Vec<_>>();
        let mut duplicated = samples.clone();
        duplicated.extend_from_slice(&samples);
        for cursor in 0..capacity {
            for delay in [11.0, 11.37, 18.75, 31.125, 49.999] {
                assert_eq!(
                    fractional_delay_read_duplicated(
                        &duplicated,
                        capacity,
                        cursor,
                        delay,
                        fractional_delay_table(),
                    )
                    .to_bits(),
                    fractional_delay_read(&samples, cursor, delay).to_bits(),
                    "cursor {cursor}, delay {delay}"
                );
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn descending_delay_line_is_bit_exact_with_the_ring_reader() {
        let capacity = 64;
        let mut optimized = StereoFractionalDelay::new(capacity);
        let mut reference = vec![0.0; capacity];
        let mut cursor = 0;
        for frame in 0..4_096 {
            let input = (frame as f32 * 0.371).sin();
            let delay = 11.0 + (frame % 37) as f32 * 0.73;
            reference[cursor] = input;
            let expected = fractional_delay_read(&reference, cursor, delay);
            let actual = optimized.process([input, -input], [delay, delay])[0];
            assert_eq!(actual.to_bits(), expected.to_bits(), "frame {frame}");
            cursor += 1;
            if cursor == capacity {
                cursor = 0;
            }
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn bandlimited_delay_is_flat_and_phase_accurate_through_eighteen_kilohertz() {
        let table = fractional_delay_table();
        let mut maximum_magnitude_error_db = 0.0_f32;
        let mut maximum_phase_error = 0.0_f32;
        for fraction_index in 0..=512 {
            let fraction = (fraction_index as f32 + 0.37) / 513.0;
            let phase = fractional_delay_phase(fraction);
            for frequency in (0..=18_000).step_by(125) {
                let angle = std::f32::consts::TAU * frequency as f32 / 48_000.0;
                let mut real = 0.0;
                let mut imaginary = 0.0;
                for (tap, coefficient) in table[phase].iter().copied().enumerate() {
                    let offset = tap as f32 - FRACTIONAL_DELAY_PRE_SAMPLES as f32;
                    real = coefficient.mul_add((-angle * offset).cos(), real);
                    imaginary = coefficient.mul_add((-angle * offset).sin(), imaginary);
                }
                let magnitude_error_db = 20.0 * real.hypot(imaginary).max(1e-9).log10();
                maximum_magnitude_error_db =
                    maximum_magnitude_error_db.max(magnitude_error_db.abs());
                let mut phase_error = imaginary.atan2(real) + angle * fraction;
                phase_error = (phase_error + std::f32::consts::PI)
                    .rem_euclid(std::f32::consts::TAU)
                    - std::f32::consts::PI;
                maximum_phase_error = maximum_phase_error.max(phase_error.abs());
            }
        }
        assert!(
            maximum_magnitude_error_db < 0.002,
            "fractional-delay magnitude error reached {maximum_magnitude_error_db:.6} dB"
        );
        assert!(
            maximum_phase_error < 2.5e-4,
            "fractional-delay phase error reached {maximum_phase_error:.7} radians"
        );
        assert_eq!(
            FRACTIONAL_DELAY_PRE_SAMPLES + FRACTIONAL_DELAY_POST_SAMPLES + 1,
            FRACTIONAL_DELAY_TAPS
        );
    }

    #[test]
    fn exact_front_centre_is_strictly_mono_compatible() {
        let grid = ContinuousHrtfGrid::new(&flat_model());
        let target = grid.target([0.0, 1.0, 0.0]);
        assert!((target.left_delay - target.right_delay).abs() < f32::EPSILON);
        assert!(
            target
                .left
                .iter()
                .zip(&target.right)
                .all(|(left, right)| (*left - *right).abs() < 1e-6)
        );
    }

    #[test]
    fn rear_azimuth_seam_is_continuous() {
        let grid = ContinuousHrtfGrid::new(&flat_model());
        let offset = 0.1_f32.to_radians();
        let clockwise = grid.target([offset.sin(), -offset.cos(), 0.0]);
        let anticlockwise = grid.target([-offset.sin(), -offset.cos(), 0.0]);
        let maximum_filter_delta = clockwise
            .left
            .iter()
            .chain(&clockwise.right)
            .zip(anticlockwise.right.iter().chain(&anticlockwise.left))
            .map(|(clockwise, anticlockwise)| (clockwise - anticlockwise).abs())
            .fold(0.0_f32, f32::max);
        let maximum_delay_delta = (clockwise.left_delay - anticlockwise.right_delay)
            .abs()
            .max((clockwise.right_delay - anticlockwise.left_delay).abs());

        assert!(
            maximum_filter_delta < 1e-3,
            "rear-seam mirrored FIR delta reached {maximum_filter_delta}"
        );
        assert!(
            maximum_delay_delta < 1e-3,
            "rear-seam mirrored delay delta reached {maximum_delay_delta}"
        );
    }

    #[test]
    fn short_continuous_filter_tracks_the_analytic_magnitude() {
        let model = flat_model();
        let grid = ContinuousHrtfGrid::new(&model);
        let azimuth = 35.0_f32.to_radians();
        let elevation = 20.0_f32.to_radians();
        let direction = [
            azimuth.sin() * elevation.cos(),
            azimuth.cos() * elevation.cos(),
            elevation.sin(),
        ];
        let target = grid.target(direction);
        let desired = model.magnitudes(direction, SYNTHESIS_FFT);
        let rendered = magnitude_spectrum(&target.left);
        let mut maximum_error_db = 0.0_f32;
        for (rendered, desired) in rendered[3..=192].iter().zip(&desired.0[3..=192]) {
            let error_db = 20.0 * (rendered.max(1e-9) / desired.max(1e-9)).log10();
            maximum_error_db = maximum_error_db.max(error_db.abs());
        }
        assert!(
            maximum_error_db < 0.75,
            "continuous FIR magnitude error reached {maximum_error_db:.3} dB"
        );
    }

    #[test]
    fn bundled_profile_shape_survives_the_short_continuous_filter() {
        let model = bundled_model();
        let grid = ContinuousHrtfGrid::new(&model);
        let mut maximum_error_db = 0.0_f32;
        for (azimuth, elevation) in [
            (0.0_f32, 0.0_f32),
            (45.0, 0.0),
            (90.0, 0.0),
            (135.0, 0.0),
            (-90.0, 0.0),
            (45.0, 30.0),
            (135.0, 30.0),
        ] {
            let azimuth = azimuth.to_radians();
            let elevation = elevation.to_radians();
            let direction = [
                azimuth.sin() * elevation.cos(),
                azimuth.cos() * elevation.cos(),
                elevation.sin(),
            ];
            let target = grid.target(direction);
            let desired = model.magnitudes(direction, SYNTHESIS_FFT);
            for (rendered, desired) in magnitude_spectrum(&target.left)[3..=192]
                .iter()
                .zip(&desired.0[3..=192])
            {
                maximum_error_db = maximum_error_db
                    .max((20.0 * (rendered.max(1e-9) / desired.max(1e-9)).log10()).abs());
            }
        }
        assert!(
            maximum_error_db < 0.85,
            "bundled-profile continuous FIR error reached {maximum_error_db:.3} dB"
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn dense_off_grid_cues_track_the_separate_parametric_targets() {
        let model = bundled_model();
        let grid = ContinuousHrtfGrid::new(&model);
        let mut maximum_timbre_error_db = 0.0_f32;
        let mut maximum_ild_error_db = 0.0_f32;
        let mut maximum_itd_error_frames = 0.0_f32;
        for direction in fibonacci_sphere(257) {
            let target = grid.target(direction);
            let desired = model.magnitudes(direction, SYNTHESIS_FFT);
            let rendered_left = magnitude_spectrum(&target.left);
            let rendered_right = magnitude_spectrum(&target.right);
            for bin in 3..=192 {
                let left_error =
                    20.0 * (rendered_left[bin].max(1e-9) / desired.0[bin].max(1e-9)).log10();
                let right_error =
                    20.0 * (rendered_right[bin].max(1e-9) / desired.1[bin].max(1e-9)).log10();
                maximum_timbre_error_db = maximum_timbre_error_db
                    .max(left_error.abs())
                    .max(right_error.abs());
                let rendered_ild =
                    20.0 * (rendered_left[bin].max(1e-9) / rendered_right[bin].max(1e-9)).log10();
                let desired_ild =
                    20.0 * (desired.0[bin].max(1e-9) / desired.1[bin].max(1e-9)).log10();
                maximum_ild_error_db = maximum_ild_error_db.max((rendered_ild - desired_ild).abs());
            }
            let desired_itd =
                super::woodworth_itd_seconds(direction[0]) * model.sample_rate() as f32;
            let rendered_itd = target.left_delay - target.right_delay;
            maximum_itd_error_frames =
                maximum_itd_error_frames.max((rendered_itd - desired_itd).abs());
        }
        assert!(
            maximum_timbre_error_db < 0.5,
            "dense off-grid timbre error reached {maximum_timbre_error_db:.3} dB"
        );
        assert!(
            maximum_ild_error_db < 0.35,
            "dense off-grid ILD error reached {maximum_ild_error_db:.3} dB"
        );
        assert!(
            maximum_itd_error_frames < 5e-5,
            "dense off-grid ITD error reached {maximum_itd_error_frames:.6} frames"
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn woodworth_itd_is_symmetric_monotonic_and_preserves_the_ear_maximum() {
        let mut previous = 0.0;
        for step in 0..=1_000 {
            let lateral = step as f32 / 1_000.0;
            let delay = super::woodworth_itd_seconds(lateral);
            assert!(delay >= previous);
            assert!((delay + super::woodworth_itd_seconds(-lateral)).abs() < 1e-9);
            previous = delay;
        }
        assert!(super::woodworth_itd_seconds(0.0).abs() < f32::EPSILON);
        assert!((super::woodworth_itd_seconds(1.0) - super::MAXIMUM_ITD_SECONDS).abs() < 1e-9);

        // A sine-scaled maximum exaggerates intermediate delays. At 30
        // degrees the rigid-sphere path is about 79.6% of that value.
        let lateral = 0.5;
        let sine_scaled = lateral * super::MAXIMUM_ITD_SECONDS;
        let rigid_sphere = super::woodworth_itd_seconds(lateral);
        assert!((rigid_sphere / sine_scaled - 0.795_8).abs() < 1e-3);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn moving_filter_has_no_control_interval_discontinuity() {
        let grid = Arc::new(ContinuousHrtfGrid::new(&flat_model()));
        let mut renderer = ContinuousBinaural::new(grid, [-1.0, 1.0, 0.0]);
        renderer.set_direction([1.0, 1.0, 0.0], 4_800);
        let output = (0..5_200)
            .map(|frame| {
                let input = (std::f32::consts::TAU * 1_000.0 * frame as f32 / 48_000.0).sin() * 0.5;
                renderer.process(input)
            })
            .collect::<Vec<_>>();
        let (maximum_delta, maximum_frame) = output
            .windows(2)
            .enumerate()
            .skip(128)
            .map(|(frame, frames)| {
                (
                    (frames[1][0] - frames[0][0])
                        .abs()
                        .max((frames[1][1] - frames[0][1]).abs()),
                    frame + 1,
                )
            })
            .max_by(|left, right| left.0.total_cmp(&right.0))
            .unwrap();
        assert!(
            maximum_delta < 0.2,
            "moving continuous filter jumped by {maximum_delta} at frame {maximum_frame}"
        );
    }

    #[test]
    fn silent_fast_path_is_bit_exact_after_the_finite_tail() {
        let grid = Arc::new(ContinuousHrtfGrid::new(&flat_model()));
        let mut optimized = ContinuousBinaural::new(grid.clone(), [0.5, 1.0, 0.25]);
        let mut reference = ContinuousBinaural::new(grid, [0.5, 1.0, 0.25]);
        assert_eq!(
            optimized.process(0.75).map(f32::to_bits),
            reference.process(0.75).map(f32::to_bits)
        );
        let tail = optimized.tail_frames();
        for frame in 0..tail + 128 {
            if frame == tail / 2 {
                optimized.set_direction([-0.75, 1.0, 0.5], 64);
                reference.set_direction([-0.75, 1.0, 0.5], 64);
            }
            reference.active_tail = usize::MAX;
            assert_eq!(
                optimized.process(0.0).map(f32::to_bits),
                reference.process(0.0).map(f32::to_bits),
                "silent output changed at tail frame {frame}"
            );
        }
    }

    fn flat_model() -> ParametricHrtfModel {
        let directions = [
            [-1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
        ];
        let filters = vec![(vec![1.0], vec![1.0]); directions.len()];
        let routes = directions
            .into_iter()
            .enumerate()
            .map(|(index, direction)| PanningRoute {
                index,
                speaker: None,
                direction,
            })
            .collect::<Vec<_>>();
        ParametricHrtfModel::new(&filters, &routes, 48_000).unwrap()
    }

    fn bundled_model() -> ParametricHrtfModel {
        let hrir = HrirSet::load_default().unwrap();
        let mut filters = Vec::new();
        let mut routes = Vec::new();
        for channel in &hrir.channels {
            if channel.speaker == Speaker::Lfe {
                continue;
            }
            let index = filters.len();
            filters.push((channel.left.clone(), channel.right.clone()));
            routes.push(PanningRoute {
                index,
                speaker: Some(channel.speaker),
                direction: channel.speaker.position(),
            });
        }
        ParametricHrtfModel::new(&filters, &routes, hrir.sample_rate).unwrap()
    }

    #[allow(clippy::cast_precision_loss)] // Test grid sizes are tiny.
    fn fibonacci_sphere(count: usize) -> Vec<[f32; 3]> {
        let golden_angle = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
        (0..count)
            .map(|index| {
                let z = 1.0 - 2.0 * (index as f32 + 0.5) / count as f32;
                let radius = (1.0 - z * z).sqrt();
                let azimuth = golden_angle * index as f32;
                [azimuth.sin() * radius, azimuth.cos() * radius, z]
            })
            .collect()
    }

    fn magnitude_spectrum(impulse: &[f32]) -> Vec<f32> {
        let mut spectrum = vec![Complex32::ZERO; SYNTHESIS_FFT];
        for (bin, sample) in spectrum.iter_mut().zip(impulse) {
            bin.re = *sample;
        }
        FftPlanner::<f32>::new()
            .plan_fft_forward(SYNTHESIS_FFT)
            .process(&mut spectrum);
        spectrum[..=SYNTHESIS_FFT / 2]
            .iter()
            .map(|bin| bin.norm())
            .collect()
    }
}
