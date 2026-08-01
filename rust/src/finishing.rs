use std::f64::consts::PI;

pub(crate) const AAC_BITRATE: &str = "320k";
pub(crate) const AAC_CODER: &str = "fast";
pub(crate) const FLAC_COMPRESSION_LEVEL: &str = "0";

const BASS_PEAK_HZ: f64 = 55.0;
const BASS_PEAK_DB: f64 = 0.8;
const BASS_PEAK_Q: f64 = 0.8;
const LOW_MID_HZ: f64 = 240.0;
const LOW_MID_DB: f64 = -0.8;
const LOW_MID_Q: f64 = 0.65;
const HIGH_SHELF_HZ: f64 = 8_000.0;
const HIGH_SHELF_DB: f64 = 0.5;
const HIGH_SHELF_SLOPE: f64 = 0.7;

/// Applies the fixed finishing curve identically to both ears.
pub(crate) struct FinishingEq {
    left: FilterChain,
    right: FilterChain,
}

impl FinishingEq {
    pub(crate) fn new(sample_rate: u32) -> Self {
        Self {
            left: FilterChain::new(sample_rate),
            right: FilterChain::new(sample_rate),
        }
    }

    pub(crate) fn process(&mut self, interleaved_stereo: &mut [f32]) {
        debug_assert_eq!(interleaved_stereo.len() % 2, 0);
        for frame in interleaved_stereo.chunks_exact_mut(2) {
            frame[0] = self.left.process(frame[0]);
            frame[1] = self.right.process(frame[1]);
        }
    }
}

struct FilterChain {
    filters: [Biquad; 3],
}

impl FilterChain {
    fn new(sample_rate: u32) -> Self {
        Self {
            filters: [
                Biquad::peaking(sample_rate, BASS_PEAK_HZ, BASS_PEAK_Q, BASS_PEAK_DB),
                Biquad::peaking(sample_rate, LOW_MID_HZ, LOW_MID_Q, LOW_MID_DB),
                Biquad::high_shelf(sample_rate, HIGH_SHELF_HZ, HIGH_SHELF_SLOPE, HIGH_SHELF_DB),
            ],
        }
    }

    #[allow(clippy::cast_possible_truncation)] // The renderer is an f32 DSP pipeline.
    fn process(&mut self, sample: f32) -> f32 {
        self.filters
            .iter_mut()
            .fold(f64::from(sample), |value, filter| filter.process(value)) as f32
    }
}

struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    fn peaking(sample_rate: u32, frequency: f64, q: f64, gain_db: f64) -> Self {
        if frequency >= f64::from(sample_rate) * 0.5 {
            return Self::identity();
        }
        let amplitude = 10_f64.powf(gain_db / 40.0);
        let omega = 2.0 * PI * frequency / f64::from(sample_rate);
        let alpha = omega.sin() / (2.0 * q);
        Self::normalized(
            [
                1.0 + alpha * amplitude,
                -2.0 * omega.cos(),
                1.0 - alpha * amplitude,
            ],
            [
                1.0 + alpha / amplitude,
                -2.0 * omega.cos(),
                1.0 - alpha / amplitude,
            ],
        )
    }

    fn high_shelf(sample_rate: u32, frequency: f64, slope: f64, gain_db: f64) -> Self {
        if frequency >= f64::from(sample_rate) * 0.5 {
            return Self::identity();
        }
        let amplitude = 10_f64.powf(gain_db / 40.0);
        let omega = 2.0 * PI * frequency / f64::from(sample_rate);
        let cosine = omega.cos();
        let alpha = omega.sin()
            * 0.5
            * ((amplitude + amplitude.recip()) * (slope.recip() - 1.0) + 2.0).sqrt();
        let beta = 2.0 * amplitude.sqrt() * alpha;
        let numerator = [
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine + beta),
            -2.0 * amplitude * ((amplitude - 1.0) + (amplitude + 1.0) * cosine),
            amplitude * ((amplitude + 1.0) + (amplitude - 1.0) * cosine - beta),
        ];
        let denominator = [
            (amplitude + 1.0) - (amplitude - 1.0) * cosine + beta,
            2.0 * ((amplitude - 1.0) - (amplitude + 1.0) * cosine),
            (amplitude + 1.0) - (amplitude - 1.0) * cosine - beta,
        ];
        Self::normalized(numerator, denominator)
    }

    fn identity() -> Self {
        Self::normalized([1.0, 0.0, 0.0], [1.0, 0.0, 0.0])
    }

    fn normalized(numerator: [f64; 3], denominator: [f64; 3]) -> Self {
        let a0 = denominator[0];
        Self {
            b0: numerator[0] / a0,
            b1: numerator[1] / a0,
            b2: numerator[2] / a0,
            a1: denominator[1] / a0,
            a2: denominator[2] / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output
    }
}

#[cfg(test)]
mod tests {
    use super::{AAC_BITRATE, AAC_CODER, FinishingEq};

    #[test]
    fn finishing_is_common_left_right_and_block_independent() {
        let input = (0_u16..8_192)
            .flat_map(|index| {
                let index = f32::from(index);
                let sample = ((index * 0.017).sin() + (index * 0.071).cos()) * 0.2;
                [sample, sample]
            })
            .collect::<Vec<_>>();
        let mut whole = input.clone();
        FinishingEq::new(48_000).process(&mut whole);

        let mut chunked = input;
        let mut eq = FinishingEq::new(48_000);
        for block in chunked.chunks_mut(514) {
            eq.process(block);
        }

        assert_eq!(whole, chunked);
        assert!(
            whole
                .chunks_exact(2)
                .all(|frame| frame[0].to_bits() == frame[1].to_bits())
        );
    }

    #[test]
    fn finishing_curve_remains_subtle() {
        for (frequency, expected_db) in [(55.0, 0.8), (240.0, -0.8), (16_000.0, 0.5)] {
            let measured = measure_gain(frequency);
            assert!(
                (measured - expected_db).abs() < 0.15,
                "{frequency} Hz gain was {measured:.3} dB; expected {expected_db:.3} dB"
            );
        }
    }

    #[test]
    fn bass_contour_is_a_broad_peak_instead_of_a_shelf() {
        let low_edge = measure_gain(20.0);
        let center = measure_gain(55.0);
        let high_edge = measure_gain(80.0);

        assert!(low_edge > 0.05, "20 Hz gain was {low_edge:.3} dB");
        assert!(high_edge > 0.05, "80 Hz gain was {high_edge:.3} dB");
        assert!(center > low_edge + 0.2);
        assert!(center > high_edge + 0.2);
    }

    #[test]
    fn aac_compatibility_mode_uses_the_high_bitrate_fast_search() {
        assert_eq!(AAC_BITRATE, "320k");
        assert_eq!(AAC_CODER, "fast");
    }

    #[allow(clippy::cast_possible_truncation)] // The production DSP is intentionally f32.
    fn measure_gain(frequency: f64) -> f64 {
        const SAMPLE_RATE: u32 = 48_000;
        const FRAMES: usize = 96_000;
        let phase_step = std::f64::consts::TAU * frequency / f64::from(SAMPLE_RATE);
        let mut phase = 0.0_f64;
        let mut samples = (0..FRAMES)
            .flat_map(|_| {
                let sample = phase.sin() as f32;
                phase += phase_step;
                [sample, sample]
            })
            .collect::<Vec<_>>();
        FinishingEq::new(SAMPLE_RATE).process(&mut samples);
        let start = FRAMES;
        let output_power = samples[start..]
            .iter()
            .step_by(2)
            .map(|sample| f64::from(*sample).powi(2))
            .sum::<f64>();
        let frames = 48_000.0;
        10.0 * (output_power / (frames * 0.5)).log10()
    }
}
