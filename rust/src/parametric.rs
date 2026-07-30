use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::binaural::PanningRoute;

const DIRECTION_SHAPE_LIMIT_DB: f32 = 2.5;
const MAXIMUM_ITD_SECONDS: f32 = 0.000_65;

/// Replaces profile-wide HRIR colour with a clean analytic ITD/ILD body while
/// retaining a small, regularised portion of each measured direction's upper
/// spectral shape. The same direction-shape gain is applied to both ears, so
/// it cannot alter the analytic interaural level difference.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn apply_direction_shaped_parametric_hrtf(
    filters: &mut [(Vec<f32>, Vec<f32>)],
    routes: &[PanningRoute],
    sample_rate: u32,
) {
    if routes.is_empty() {
        return;
    }

    let measured = filters.to_vec();
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
    let inverse = planner.plan_fft_inverse(fft_len);
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
            common_reference[bin] += 0.5 * (left[bin].max(1e-9).ln() + right[bin].max(1e-9).ln());
        }
    }
    for value in &mut common_reference {
        *value /= spectra.len() as f32;
    }
    let common_delay = profile_delay(&measured, routes);

    for (route_position, route) in routes.iter().enumerate() {
        let lateral = lateral_position(route.direction);
        let (mut left_magnitude, mut right_magnitude) =
            geometric_magnitude(lateral, sample_rate, fft_len);
        let shape = direction_shape(
            &spectra[route_position].0,
            &spectra[route_position].1,
            &common_reference,
            sample_rate,
            fft_len,
        );
        for bin in 0..bins {
            let multiplier = shape[bin].exp();
            left_magnitude[bin] *= multiplier;
            right_magnitude[bin] *= multiplier;
        }

        let left = minimum_phase_impulse(
            &left_magnitude,
            fft_len,
            output_len,
            forward.as_ref(),
            inverse.as_ref(),
        );
        let right = minimum_phase_impulse(
            &right_magnitude,
            fft_len,
            output_len,
            forward.as_ref(),
            inverse.as_ref(),
        );
        let itd = lateral * MAXIMUM_ITD_SECONDS * sample_rate as f32;
        filters[route.index] = (
            delayed_impulse(&left, common_delay as f32 + itd.max(0.0)),
            delayed_impulse(&right, common_delay as f32 + (-itd).max(0.0)),
        );
    }
}

fn lateral_position(direction: [f32; 3]) -> f32 {
    let horizontal = direction[0].hypot(direction[1]);
    if horizontal > f32::EPSILON {
        (direction[0] / horizontal).clamp(-1.0, 1.0)
    } else {
        0.0
    }
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
fn minimum_phase_impulse(
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
    use super::{DIRECTION_SHAPE_LIMIT_DB, direction_shape, geometric_magnitude};

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
