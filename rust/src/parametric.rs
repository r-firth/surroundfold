use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::binaural::PanningRoute;

const DIRECTION_SHAPE_LIMIT_DB: f32 = 2.5;
pub(crate) const MAXIMUM_ITD_SECONDS: f32 = 0.000_65;

pub(crate) struct ParametricHrtfModel {
    sample_rate: u32,
    fft_len: usize,
    output_len: usize,
    common_delay: usize,
    directions: Vec<[f32; 3]>,
    direction_shapes: Vec<Vec<f32>>,
}

impl ParametricHrtfModel {
    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn new(
        measured: &[(Vec<f32>, Vec<f32>)],
        routes: &[PanningRoute],
        sample_rate: u32,
    ) -> Option<Self> {
        if routes.is_empty() {
            return None;
        }
        let maximum_length = routes
            .iter()
            .map(|route| {
                measured[route.index]
                    .0
                    .len()
                    .max(measured[route.index].1.len())
            })
            .max()
            .unwrap_or(256);
        let fft_len = maximum_length
            .saturating_mul(4)
            .next_power_of_two()
            .max(2_048);
        let output_len = maximum_length.clamp(256, 512);
        let bins = fft_len / 2 + 1;
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(fft_len);
        let spectra = routes
            .iter()
            .map(|route| {
                let (left, right) = &measured[route.index];
                (
                    magnitude_spectrum(left, fft_len, forward.as_ref()),
                    magnitude_spectrum(right, fft_len, forward.as_ref()),
                )
            })
            .collect::<Vec<_>>();
        let mut common_reference = vec![0.0; bins];
        for (left, right) in &spectra {
            for bin in 0..bins {
                common_reference[bin] +=
                    0.5 * (left[bin].max(1e-9).ln() + right[bin].max(1e-9).ln());
            }
        }
        for value in &mut common_reference {
            *value /= spectra.len() as f32;
        }
        let direction_shapes = spectra
            .iter()
            .map(|(left, right)| {
                direction_shape(left, right, &common_reference, sample_rate, fft_len)
            })
            .collect();
        Some(Self {
            sample_rate,
            fft_len,
            output_len,
            common_delay: profile_delay(measured, routes),
            directions: routes.iter().map(|route| route.direction).collect(),
            direction_shapes,
        })
    }

    #[must_use]
    pub(crate) const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[must_use]
    pub(crate) const fn common_delay(&self) -> usize {
        self.common_delay
    }

    #[allow(clippy::cast_precision_loss)]
    pub(crate) fn magnitudes(&self, direction: [f32; 3], fft_len: usize) -> (Vec<f32>, Vec<f32>) {
        let lateral = interaural_axis_projection(direction);
        let (mut left, mut right) = geometric_magnitude(lateral, self.sample_rate, fft_len);
        for bin in 0..left.len() {
            let frequency = bin as f32 * self.sample_rate as f32 / fft_len as f32;
            let multiplier = self.interpolated_shape(direction, frequency).exp();
            left[bin] *= multiplier;
            right[bin] *= multiplier;
        }
        (left, right)
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn interpolated_shape(&self, direction: [f32; 3], frequency: f32) -> f32 {
        let direction = normalized(direction);
        let position = frequency * self.fft_len as f32 / self.sample_rate as f32;
        let lower = position.floor().max(0.0) as usize;
        let upper = lower
            .saturating_add(1)
            .min(self.direction_shapes[0].len() - 1);
        let fraction = (position - lower as f32).clamp(0.0, 1.0);
        let mut weighted = 0.0;
        let mut weight_sum = 0.0;
        for (route_direction, shape) in self.directions.iter().zip(&self.direction_shapes) {
            let dot = direction_dot(direction, normalized(*route_direction)).clamp(-1.0, 1.0);
            let angular_distance = 1.0 - dot;
            let weight = 1.0 / (0.002 + angular_distance).powi(2);
            let value = shape[lower].mul_add(1.0 - fraction, shape[upper] * fraction);
            weighted += value * weight;
            weight_sum += weight;
        }
        weighted / weight_sum.max(f32::EPSILON)
    }

    #[allow(clippy::cast_precision_loss)]
    fn apply(&self, filters: &mut [(Vec<f32>, Vec<f32>)], routes: &[PanningRoute]) {
        let mut planner = FftPlanner::<f32>::new();
        let forward = planner.plan_fft_forward(self.fft_len);
        let inverse = planner.plan_fft_inverse(self.fft_len);
        for (route_position, route) in routes.iter().enumerate() {
            // Preserve the approved virtual-speaker baseline. The continuous
            // renderer uses the physically smooth 3-D interaural projection,
            // but changing these established route filters would also change
            // the default renderer before listening approval.
            let lateral = baseline_horizontal_lateral_position(route.direction);
            let (mut left_magnitude, mut right_magnitude) =
                geometric_magnitude(lateral, self.sample_rate, self.fft_len);
            for bin in 0..left_magnitude.len() {
                let multiplier = self.direction_shapes[route_position][bin].exp();
                left_magnitude[bin] *= multiplier;
                right_magnitude[bin] *= multiplier;
            }
            let left = minimum_phase_impulse(
                &left_magnitude,
                self.fft_len,
                self.output_len,
                forward.as_ref(),
                inverse.as_ref(),
            );
            let right = minimum_phase_impulse(
                &right_magnitude,
                self.fft_len,
                self.output_len,
                forward.as_ref(),
                inverse.as_ref(),
            );
            let itd = lateral * MAXIMUM_ITD_SECONDS * self.sample_rate as f32;
            filters[route.index] = (
                delayed_impulse(&left, self.common_delay as f32 + itd.max(0.0)),
                delayed_impulse(&right, self.common_delay as f32 + (-itd).max(0.0)),
            );
        }
    }
}

/// Replaces profile-wide HRIR colour with a clean analytic ITD/ILD body while
/// retaining a small, regularised portion of each measured direction's upper
/// spectral shape. The same direction-shape gain is applied to both ears, so
/// it cannot alter the analytic interaural level difference.
pub(crate) fn apply_direction_shaped_parametric_hrtf(
    filters: &mut [(Vec<f32>, Vec<f32>)],
    routes: &[PanningRoute],
    sample_rate: u32,
) -> Option<ParametricHrtfModel> {
    let measured = filters.to_vec();
    let model = ParametricHrtfModel::new(&measured, routes, sample_rate)?;
    model.apply(filters, routes);
    Some(model)
}

fn baseline_horizontal_lateral_position(direction: [f32; 3]) -> f32 {
    let horizontal = direction[0].hypot(direction[1]);
    if horizontal > f32::EPSILON {
        (direction[0] / horizontal).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

fn interaural_axis_projection(direction: [f32; 3]) -> f32 {
    normalized(direction)[0].clamp(-1.0, 1.0)
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

fn direction_dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[allow(clippy::cast_precision_loss)]
fn geometric_magnitude(lateral: f32, sample_rate: u32, fft_len: usize) -> (Vec<f32>, Vec<f32>) {
    let mut left = vec![0.0; fft_len / 2 + 1];
    let mut right = vec![0.0; fft_len / 2 + 1];
    for bin in 0..left.len() {
        let frequency = bin as f32 * sample_rate as f32 / fft_len as f32;
        let progress = if frequency <= 200.0 {
            0.0
        } else {
            (frequency / 200.0).log2() / (8_000.0_f32 / 200.0).log2()
        }
        .clamp(0.0, 1.0);
        let progress = progress * progress * (3.0 - 2.0 * progress);
        let ild_db = 12.0 * progress * lateral;
        let unnormalized_left = 10_f32.powf(-ild_db / 40.0);
        let unnormalized_right = 10_f32.powf(ild_db / 40.0);
        let normalization = unnormalized_left
            .hypot(unnormalized_right)
            .max(f32::EPSILON);
        left[bin] = unnormalized_left / normalization;
        right[bin] = unnormalized_right / normalization;
    }
    (left, right)
}

#[allow(clippy::cast_precision_loss)]
fn direction_shape(
    measured_left: &[f32],
    measured_right: &[f32],
    common_reference: &[f32],
    sample_rate: u32,
    fft_len: usize,
) -> Vec<f32> {
    let mut shape = fractional_octave_smooth(
        &measured_left
            .iter()
            .zip(measured_right)
            .zip(common_reference)
            .map(|((left, right), reference)| {
                0.5 * (left.max(1e-9).ln() + right.max(1e-9).ln()) - reference
            })
            .collect::<Vec<_>>(),
    );

    let mut mean = 0.0;
    let mut weight_sum = 0.0;
    for (bin, value) in shape.iter().copied().enumerate().skip(1) {
        let frequency = bin as f32 * sample_rate as f32 / fft_len as f32;
        if (4_000.0..=14_000.0).contains(&frequency) {
            let weight = 1.0 / frequency;
            mean += value * weight;
            weight_sum += weight;
        }
    }
    mean /= weight_sum.max(f32::EPSILON);
    let limit = DIRECTION_SHAPE_LIMIT_DB * std::f32::consts::LN_10 / 20.0;
    for (bin, value) in shape.iter_mut().enumerate() {
        let frequency = bin as f32 * sample_rate as f32 / fft_len as f32;
        let low_weight = if frequency <= 2_500.0 {
            0.0
        } else if frequency >= 4_000.0 {
            1.0
        } else {
            let progress = (frequency / 2_500.0).ln() / (4_000.0_f32 / 2_500.0).ln();
            progress * progress * (3.0 - 2.0 * progress)
        };
        let high_weight = if frequency <= 14_000.0 {
            1.0
        } else if frequency >= 18_000.0 {
            0.0
        } else {
            let progress = (frequency - 14_000.0) / 4_000.0;
            1.0 - progress * progress * (3.0 - 2.0 * progress)
        };
        *value = ((*value - mean) * low_weight * high_weight).clamp(-limit, limit);
    }
    shape
}

fn magnitude_spectrum(samples: &[f32], fft_len: usize, forward: &dyn Fft<f32>) -> Vec<f32> {
    let mut spectrum = vec![Complex32::ZERO; fft_len];
    for (bin, sample) in spectrum.iter_mut().zip(samples) {
        bin.re = *sample;
    }
    forward.process(&mut spectrum);
    spectrum[..=fft_len / 2]
        .iter()
        .map(|bin| bin.norm())
        .collect()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn fractional_octave_smooth(values: &[f32]) -> Vec<f32> {
    let radius = 2.0_f32.powf(1.0 / 6.0);
    let mut prefix = Vec::with_capacity(values.len() + 1);
    prefix.push(0.0);
    for value in values {
        prefix.push(prefix.last().copied().unwrap_or(0.0) + value);
    }
    (0..values.len())
        .map(|bin| {
            if bin == 0 {
                return values[0];
            }
            let lower = ((bin as f32 / radius).floor() as usize).max(1);
            let upper = ((bin as f32 * radius).ceil() as usize)
                .max(lower)
                .min(values.len() - 1);
            (prefix[upper + 1] - prefix[lower]) / (upper - lower + 1) as f32
        })
        .collect()
}

#[allow(clippy::cast_precision_loss)]
pub(crate) fn minimum_phase_impulse(
    magnitude: &[f32],
    fft_len: usize,
    output_len: usize,
    forward: &dyn Fft<f32>,
    inverse: &dyn Fft<f32>,
) -> Vec<f32> {
    let half = fft_len / 2;
    let mut spectrum = vec![Complex32::ZERO; fft_len];
    for bin in 0..=half {
        spectrum[bin].re = magnitude[bin].max(1e-9).ln();
    }
    for bin in half + 1..fft_len {
        spectrum[bin].re = spectrum[fft_len - bin].re;
    }
    inverse.process(&mut spectrum);
    let inverse_scale = 1.0 / fft_len as f32;
    for value in &mut spectrum {
        value.re *= inverse_scale;
        value.im = 0.0;
    }
    for value in &mut spectrum[1..half] {
        value.re *= 2.0;
    }
    for value in &mut spectrum[half + 1..] {
        *value = Complex32::ZERO;
    }
    forward.process(&mut spectrum);
    for value in &mut spectrum {
        *value = Complex32::from_polar(value.re.exp(), value.im);
    }
    inverse.process(&mut spectrum);
    spectrum
        .into_iter()
        .take(output_len)
        .map(|value| value.re * inverse_scale)
        .collect()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
fn delayed_impulse(impulse: &[f32], delay: f32) -> Vec<f32> {
    let whole = delay.floor().max(0.0) as usize;
    let fraction = (delay - whole as f32).clamp(0.0, 1.0);
    let mut delayed = vec![0.0; impulse.len().saturating_add(whole).saturating_add(1)];
    for (index, sample) in impulse.iter().copied().enumerate() {
        delayed[index + whole] += sample * (1.0 - fraction);
        delayed[index + whole + 1] += sample * fraction;
    }
    delayed
}

fn profile_delay(filters: &[(Vec<f32>, Vec<f32>)], routes: &[PanningRoute]) -> usize {
    let mut peaks = routes
        .iter()
        .flat_map(|route| {
            let (left, right) = &filters[route.index];
            [peak_index(left), peak_index(right)]
        })
        .collect::<Vec<_>>();
    peaks.sort_unstable();
    peaks.get(peaks.len() / 2).copied().unwrap_or(0)
}

fn peak_index(samples: &[f32]) -> usize {
    samples
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
        .map_or(0, |(index, _)| index)
}

#[cfg(test)]
mod tests {
    use super::{
        DIRECTION_SHAPE_LIMIT_DB, direction_shape, geometric_magnitude, interaural_axis_projection,
    };

    #[test]
    fn geometric_ild_is_symmetric_and_energy_normalized() {
        let (left, right) = geometric_magnitude(0.8, 48_000, 2_048);
        let (mirrored_left, mirrored_right) = geometric_magnitude(-0.8, 48_000, 2_048);
        for bin in 0..left.len() {
            assert!((left[bin] - mirrored_right[bin]).abs() < 1e-6);
            assert!((right[bin] - mirrored_left[bin]).abs() < 1e-6);
            assert!((left[bin].hypot(right[bin]) - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn continuous_lateral_cues_fade_smoothly_toward_the_elevation_poles() {
        let horizontal = interaural_axis_projection([1.0, 0.0, 0.0]);
        let elevated = interaural_axis_projection([1.0, 0.0, 1.0]);
        let pole = interaural_axis_projection([0.0, 0.0, 1.0]);

        assert!((horizontal - 1.0).abs() < f32::EPSILON);
        assert!((elevated - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-6);
        assert!(pole.abs() < f32::EPSILON);
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::float_cmp)]
    fn direction_shape_is_bounded_and_absent_below_transition() {
        let bins = 1_025;
        let measured_left = (0..bins)
            .map(|bin| 0.5 + bin as f32 / bins as f32)
            .collect::<Vec<_>>();
        let measured_right = measured_left
            .iter()
            .map(|value| value * 0.9)
            .collect::<Vec<_>>();
        let common = vec![0.0; bins];
        let shape = direction_shape(&measured_left, &measured_right, &common, 48_000, 2_048);
        let limit = DIRECTION_SHAPE_LIMIT_DB * std::f32::consts::LN_10 / 20.0;
        for (bin, value) in shape.iter().enumerate() {
            let frequency = bin as f32 * 48_000.0 / 2_048.0;
            assert!(value.abs() <= limit + 1e-6);
            if frequency <= 2_500.0 {
                assert_eq!(*value, 0.0);
            }
        }
    }
}
