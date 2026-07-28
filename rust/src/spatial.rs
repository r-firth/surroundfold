use std::cmp::Ordering;

use crate::{binaural::BinauralWriter, error::AppError, hrir::Speaker};

/// Equal-power interpolation over the directional channels present in an HRIR.
pub(crate) struct SpatialPanner {
    buses: Vec<PanningBus>,
    output_bus_count: usize,
}

#[must_use]
pub(crate) fn direct_stereo_gains(speaker: Speaker) -> [f32; 2] {
    if speaker == Speaker::Lfe {
        return [0.5, 0.5];
    }
    let x = speaker.position()[0].clamp(-1.0, 1.0);
    [((1.0 - x) * 0.5).sqrt(), ((1.0 + x) * 0.5).sqrt()]
}

struct PanningBus {
    index: usize,
    direction: [f32; 3],
}

impl SpatialPanner {
    pub(crate) fn new(writer: &BinauralWriter) -> Result<Self, AppError> {
        let mut buses = writer
            .bus_by_speaker()
            .filter(|(speaker, _)| *speaker != Speaker::Lfe)
            .map(|(speaker, index)| PanningBus {
                index,
                direction: normalized(speaker.position()),
            })
            .collect::<Vec<_>>();
        buses.sort_by_key(|bus| bus.index);
        if buses.is_empty() {
            return Err(AppError::InvalidHrir(
                "HRIR has no directional channels for spatial rendering".into(),
            ));
        }
        Ok(Self {
            buses,
            output_bus_count: writer.bus_count(),
        })
    }

    #[must_use]
    pub(crate) const fn bus_count(&self) -> usize {
        self.output_bus_count
    }

    /// Pans one source position to the three nearest virtual speakers.
    ///
    /// `size` broadens a point source toward all directional buses while
    /// preserving equal power. `gain` is applied after panning.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata contains NaN or infinity.
    #[allow(clippy::cast_precision_loss)] // The tiny HRIR bus count is exactly represented.
    pub(crate) fn gains(
        &self,
        position: [f32; 3],
        size: f32,
        gain: f32,
    ) -> Result<Vec<f32>, AppError> {
        if position.iter().any(|value| !value.is_finite()) || !gain.is_finite() || !size.is_finite()
        {
            return Err(AppError::Render(
                "spatial metadata contains a non-finite value".into(),
            ));
        }
        let direction = if magnitude_squared(position) < 1e-8 {
            [0.0, 1.0, 0.0]
        } else {
            normalized(position)
        };
        let mut nearest = self
            .buses
            .iter()
            .map(|bus| {
                let dot = dot(direction, bus.direction).clamp(-1.0, 1.0);
                (bus.index, dot.acos())
            })
            .collect::<Vec<_>>();
        nearest.sort_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal));
        nearest.truncate(3.min(nearest.len()));

        let mut gains = vec![0.0; self.output_bus_count];
        for (bus, angle) in nearest {
            gains[bus] = 1.0 / (0.05 + angle).powi(2);
        }
        normalize_power(&mut gains);
        let spread = size.clamp(0.0, 1.0);
        if spread > 0.0 {
            let diffuse = 1.0 / (self.buses.len() as f32).sqrt();
            for bus in &self.buses {
                gains[bus.index] = gains[bus.index] * (1.0 - spread) + diffuse * spread;
            }
            normalize_power(&mut gains);
        }
        for output_gain in &mut gains {
            *output_gain *= gain;
        }
        Ok(gains)
    }
}

fn normalized(value: [f32; 3]) -> [f32; 3] {
    let inverse = magnitude_squared(value).sqrt().recip();
    [value[0] * inverse, value[1] * inverse, value[2] * inverse]
}

const fn magnitude_squared(value: [f32; 3]) -> f32 {
    dot(value, value)
}

const fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn normalize_power(gains: &mut [f32]) {
    let power = gains.iter().map(|gain| gain * gain).sum::<f32>().sqrt();
    if power > f32::EPSILON {
        for gain in gains {
            *gain /= power;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{direct_stereo_gains, normalize_power};
    use crate::hrir::Speaker;

    #[test]
    fn panning_gains_are_equal_power() {
        let mut gains = [1.0, 2.0, 3.0];
        normalize_power(&mut gains);
        let power: f32 = gains.iter().map(|gain| gain * gain).sum();
        assert!((power - 1.0).abs() < 1e-6);
    }

    #[test]
    fn direct_stereo_pan_preserves_power() {
        let left = direct_stereo_gains(Speaker::FrontLeft);
        assert!((left[0] - 1.0).abs() < f32::EPSILON);
        assert!(left[1].abs() < f32::EPSILON);
        let center = direct_stereo_gains(Speaker::FrontCenter);
        assert!((center[0].mul_add(center[0], center[1] * center[1]) - 1.0).abs() < 1e-6);
    }
}
