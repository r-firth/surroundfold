//! Complex 64-band analysis/synthesis filter bank and JOC reconstruction.
//!
//! The transform follows clause 7 of ETSI TS 103 420 V1.1.1 directly. The
//! trigonometric modulation matrices are initialized once and shared; each
//! channel or object owns only its filter history.

use std::{array, f32::consts::PI, sync::OnceLock};

use rustfft::num_complex::Complex32;

use crate::{
    error::AppError,
    joc::{JocFrame, QMF_SUBBANDS},
    qmf_tables::PROTOTYPE_64,
};

const FILTER_LENGTH: usize = 640;
/// Causal delay of the normative analysis/synthesis pair, in PCM samples.
pub const RECONSTRUCTION_DELAY: usize = 577;
const DOUBLE_SUBBANDS: usize = QMF_SUBBANDS * 2;
const SYNTHESIS_HISTORY: usize = FILTER_LENGTH * 2;
const POLYPHASE_SECTIONS: usize = FILTER_LENGTH / DOUBLE_SUBBANDS;

struct ModulationKernels {
    analysis: Box<[Complex32]>,
    synthesis: Box<[Complex32]>,
}

impl ModulationKernels {
    fn get() -> &'static Self {
        static KERNELS: OnceLock<ModulationKernels> = OnceLock::new();
        KERNELS.get_or_init(Self::build)
    }

    #[allow(clippy::cast_precision_loss)] // Indices are bounded to 0..128.
    fn build() -> Self {
        let mut analysis = Vec::with_capacity(QMF_SUBBANDS * DOUBLE_SUBBANDS);
        let mut synthesis = Vec::with_capacity(QMF_SUBBANDS * DOUBLE_SUBBANDS);
        for subband in 0..QMF_SUBBANDS {
            for position in 0..DOUBLE_SUBBANDS {
                let analysis_phase =
                    PI * (subband as f32 + 0.5) * (position as f32 - 0.5) / QMF_SUBBANDS as f32;
                analysis.push(Complex32::from_polar(1.0, analysis_phase));

                // Clause 7.3's displayed matrix is used here. Its pseudocode
                // prints a different phase term; that term fails perfect
                // reconstruction, while this one produces the specified
                // analysis/synthesis pair.
                let synthesis_phase = PI
                    * (subband as f32 + 0.5)
                    * (position as f32 - 2.0 * QMF_SUBBANDS as f32 + 0.5)
                    / QMF_SUBBANDS as f32;
                synthesis.push(Complex32::from_polar(
                    1.0 / QMF_SUBBANDS as f32,
                    synthesis_phase,
                ));
            }
        }
        Self {
            analysis: analysis.into_boxed_slice(),
            synthesis: synthesis.into_boxed_slice(),
        }
    }
}

/// Stateful real-PCM to complex-QMF analysis transform.
#[derive(Clone, Debug)]
pub struct AnalysisFilter {
    history: [f32; FILTER_LENGTH],
}

impl AnalysisFilter {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            history: [0.0; FILTER_LENGTH],
        }
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
    }

    #[must_use]
    pub fn process(&mut self, pcm: &[f32; QMF_SUBBANDS]) -> [Complex32; QMF_SUBBANDS] {
        self.history
            .copy_within(0..FILTER_LENGTH - QMF_SUBBANDS, QMF_SUBBANDS);
        for (destination, source) in self.history[..QMF_SUBBANDS]
            .iter_mut()
            .zip(pcm.iter().rev())
        {
            *destination = *source;
        }

        let mut grouped = [0.0_f32; DOUBLE_SUBBANDS];
        for section in 0..POLYPHASE_SECTIONS {
            let start = section * DOUBLE_SUBBANDS;
            for position in 0..DOUBLE_SUBBANDS {
                grouped[position] +=
                    self.history[start + position] * PROTOTYPE_64[start + position];
            }
        }

        let kernels = &ModulationKernels::get().analysis;
        array::from_fn(|subband| {
            let kernel = &kernels[subband * DOUBLE_SUBBANDS..(subband + 1) * DOUBLE_SUBBANDS];
            grouped
                .iter()
                .zip(kernel)
                .fold(Complex32::ZERO, |sum, (sample, rotation)| {
                    sum + *rotation * *sample
                })
        })
    }
}

impl Default for AnalysisFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Stateful complex-QMF to real-PCM synthesis transform.
#[derive(Clone, Debug)]
pub struct SynthesisFilter {
    history: [f32; SYNTHESIS_HISTORY],
}

impl SynthesisFilter {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            history: [0.0; SYNTHESIS_HISTORY],
        }
    }

    pub fn reset(&mut self) {
        self.history.fill(0.0);
    }

    #[must_use]
    pub fn process(&mut self, qmf: &[Complex32; QMF_SUBBANDS]) -> [f32; QMF_SUBBANDS] {
        self.history
            .copy_within(0..SYNTHESIS_HISTORY - DOUBLE_SUBBANDS, DOUBLE_SUBBANDS);
        let kernels = &ModulationKernels::get().synthesis;
        for position in 0..DOUBLE_SUBBANDS {
            self.history[position] = qmf
                .iter()
                .enumerate()
                .map(|(subband, value)| {
                    let rotation = kernels[subband * DOUBLE_SUBBANDS + position];
                    value.re * rotation.re - value.im * rotation.im
                })
                .sum();
        }

        array::from_fn(|sample| {
            let mut output = 0.0_f32;
            for section in 0..POLYPHASE_SECTIONS {
                let history_start = section * 4 * QMF_SUBBANDS;
                let coefficient_start = section * DOUBLE_SUBBANDS;
                output +=
                    self.history[history_start + sample] * PROTOTYPE_64[coefficient_start + sample];
                output += self.history[history_start + 3 * QMF_SUBBANDS + sample]
                    * PROTOTYPE_64[coefficient_start + QMF_SUBBANDS + sample];
            }
            output
        })
    }
}

impl Default for SynthesisFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Applies JOC matrices to decoded downmix channels in the complex QMF domain.
#[derive(Clone, Debug, Default)]
pub struct JocReconstructor {
    analysis: Vec<AnalysisFilter>,
    synthesis: Vec<SynthesisFilter>,
}

impl JocReconstructor {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            analysis: Vec::new(),
            synthesis: Vec::new(),
        }
    }

    /// Reconstructs planar object PCM for one JOC frame.
    ///
    /// # Errors
    ///
    /// Rejects a channel-count mismatch or channels whose sample count differs
    /// from the matching JOC frame.
    pub fn reconstruct(
        &mut self,
        channels: &[&[f32]],
        joc: &JocFrame,
    ) -> Result<Vec<Vec<f32>>, AppError> {
        let sample_count = joc.timeslots * QMF_SUBBANDS;
        if channels.len() != joc.input_channels {
            return Err(invalid(format!(
                "JOC requires {} downmix channels, received {}",
                joc.input_channels,
                channels.len()
            )));
        }
        if let Some((index, channel)) = channels
            .iter()
            .enumerate()
            .find(|(_, channel)| channel.len() != sample_count)
        {
            return Err(invalid(format!(
                "JOC channel {index} has {} samples, expected {sample_count}",
                channel.len()
            )));
        }

        let dimensions_changed =
            self.analysis.len() != joc.input_channels || self.synthesis.len() != joc.object_count;
        if dimensions_changed {
            self.analysis = vec![AnalysisFilter::new(); joc.input_channels];
            self.synthesis = vec![SynthesisFilter::new(); joc.object_count];
        }

        let mut objects = vec![vec![0.0_f32; sample_count]; joc.object_count];
        let mut channel_qmf = vec![[Complex32::ZERO; QMF_SUBBANDS]; joc.input_channels];
        for timeslot in 0..joc.timeslots {
            let sample_start = timeslot * QMF_SUBBANDS;
            for (channel, (filter, transformed)) in
                self.analysis.iter_mut().zip(&mut channel_qmf).enumerate()
            {
                let pcm = channels[channel]
                    .get(sample_start..sample_start + QMF_SUBBANDS)
                    .and_then(|samples| <&[f32; QMF_SUBBANDS]>::try_from(samples).ok())
                    .ok_or_else(|| invalid("validated JOC channel block disappeared"))?;
                *transformed = filter.process(pcm);
            }

            for (object, (destination, filter)) in
                objects.iter_mut().zip(&mut self.synthesis).enumerate()
            {
                let mut object_qmf = [Complex32::ZERO; QMF_SUBBANDS];
                for (channel, transformed) in channel_qmf.iter().enumerate() {
                    let matrix = joc
                        .channel_coefficients(object, timeslot, channel)
                        .ok_or_else(|| invalid("JOC matrix dimensions are inconsistent"))?;
                    for subband in 0..QMF_SUBBANDS {
                        object_qmf[subband] += transformed[subband] * matrix[subband];
                    }
                }
                let pcm = filter.process(&object_qmf);
                let output_block = destination
                    .get_mut(sample_start..sample_start + QMF_SUBBANDS)
                    .ok_or_else(|| invalid("validated JOC output block disappeared"))?;
                for (destination, sample) in output_block.iter_mut().zip(pcm) {
                    *destination = sample * joc.clip_gain;
                }
            }
        }
        Ok(objects)
    }
}

fn invalid(message: impl Into<String>) -> AppError {
    AppError::Render(format!(
        "invalid JOC reconstruction input: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use rustfft::num_complex::Complex32;

    use super::{AnalysisFilter, QMF_SUBBANDS, SynthesisFilter};

    #[test]
    #[allow(clippy::float_cmp)] // A linear zero-state transform must preserve exact zero.
    fn zero_signal_stays_exactly_zero() {
        let mut analysis = AnalysisFilter::new();
        let mut synthesis = SynthesisFilter::new();
        for _ in 0..12 {
            let qmf = analysis.process(&[0.0; QMF_SUBBANDS]);
            assert_eq!(qmf, [Complex32::ZERO; QMF_SUBBANDS]);
            assert_eq!(synthesis.process(&qmf), [0.0; QMF_SUBBANDS]);
        }
    }

    #[test]
    fn analysis_synthesis_impulse_is_finite_and_energy_preserving() {
        let mut analysis = AnalysisFilter::new();
        let mut synthesis = SynthesisFilter::new();
        let mut output = Vec::new();
        for timeslot in 0..24 {
            let mut input = [0.0_f32; QMF_SUBBANDS];
            if timeslot == 0 {
                input[0] = 1.0;
            }
            output.extend(synthesis.process(&analysis.process(&input)));
        }
        assert!(output.iter().all(|sample| sample.is_finite()));
        let energy: f32 = output.iter().map(|sample| sample * sample).sum();
        let peak = output
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))
            .unwrap();
        assert!(
            (0.98..=1.02).contains(&energy),
            "impulse energy was {energy}"
        );
        assert_eq!(peak.0, super::RECONSTRUCTION_DELAY);
        assert!((peak.1 - 1.0).abs() < 1e-5);
    }
}
