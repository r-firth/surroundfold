//! Rendering for the stacked-ring Intermediate Spatial Format (ISF).

use crate::{
    binaural::BinauralWriter,
    error::AppError,
    hrir::{HrirSet, Speaker},
    isf_tables,
    object::ObjectTrim,
};

const ISF_CHANNEL_COUNTS: [usize; 6] = [4, 8, 10, 14, 15, 30];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IsfConfig {
    pub start_channel: usize,
    pub channel_count: usize,
}

impl IsfConfig {
    pub(crate) fn new(start_channel: usize, channel_count: usize) -> Result<Self, AppError> {
        if !ISF_CHANNEL_COUNTS.contains(&channel_count) {
            return Err(AppError::UnsupportedInput(format!(
                "unsupported ISF channel count {channel_count}"
            )));
        }
        Ok(Self {
            start_channel,
            channel_count,
        })
    }

    fn row(self, source_channel: usize) -> Option<usize> {
        source_channel
            .checked_sub(self.start_channel)
            .filter(|row| *row < self.channel_count)
    }
}

pub(crate) struct IsfRenderer {
    config: IsfConfig,
    matrix: Matrix,
    output_buses: Vec<Option<(usize, Speaker)>>,
    bus_count: usize,
    trim_configuration: usize,
    muted: bool,
}

impl IsfRenderer {
    pub(crate) fn new(
        config: IsfConfig,
        writer: &BinauralWriter,
        hrir: &HrirSet,
        surround_swap: bool,
        mute_bed: bool,
        mute_ground: bool,
    ) -> Result<Self, AppError> {
        let matrix = select_matrix(config.channel_count, hrir)?;
        let output_buses = matrix
            .speakers
            .iter()
            .map(|speaker| {
                if mute_ground && speaker.position()[2] <= 0.0 {
                    return Ok(None);
                }
                let routed = if surround_swap {
                    speaker.surround_swapped()
                } else {
                    *speaker
                };
                let resolved = hrir.resolved_speaker(routed).ok_or_else(|| {
                    AppError::InvalidHrir(format!(
                        "HRIR has no route for ISF output speaker {routed:?}"
                    ))
                })?;
                writer
                    .bus(resolved)
                    .map(|bus| Some((bus, routed)))
                    .ok_or_else(|| {
                        AppError::Render(format!(
                            "missing virtual-speaker bus for ISF output {resolved:?}"
                        ))
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let num_mid = matrix
            .speakers
            .iter()
            .filter(|speaker| {
                matches!(
                    speaker,
                    Speaker::SideLeft
                        | Speaker::SideRight
                        | Speaker::RearLeft
                        | Speaker::RearRight
                        | Speaker::RearCenter
                        | Speaker::WideLeft
                        | Speaker::WideRight
                )
            })
            .count();
        let num_top = matrix
            .speakers
            .iter()
            .filter(|speaker| speaker.position()[2] > 0.0)
            .count();
        let mid_category = match num_mid {
            0 => 0,
            1..=3 => 1,
            _ => 2,
        };
        let top_category = match num_top {
            0 => 0,
            1 | 2 => 1,
            _ => 2,
        };
        Ok(Self {
            config,
            matrix,
            output_buses,
            bus_count: writer.bus_count(),
            trim_configuration: mid_category + 3 * top_category,
            muted: mute_bed,
        })
    }

    #[must_use]
    pub(crate) const fn config(&self) -> IsfConfig {
        self.config
    }

    pub(crate) fn gains(
        &self,
        source_channel: usize,
        active: bool,
        gain: f32,
        trim: ObjectTrim,
    ) -> Result<Vec<f32>, AppError> {
        let row = self.config.row(source_channel).ok_or_else(|| {
            AppError::Render(format!(
                "channel {source_channel} is outside the configured ISF range"
            ))
        })?;
        if !gain.is_finite() {
            return Err(AppError::Render(
                "ISF metadata contains a non-finite gain".into(),
            ));
        }
        let mut gains = vec![0.0; self.bus_count];
        if !active || self.muted {
            return Ok(gains);
        }
        let start = row * self.matrix.speakers.len();
        let coefficients = &self.matrix.coefficients[start..start + self.matrix.speakers.len()];
        let trim = trim.settings(self.trim_configuration);
        for ((coefficient, output), speaker) in coefficients
            .iter()
            .zip(&self.output_buses)
            .zip(self.matrix.speakers)
        {
            if let Some((bus, routed)) = output {
                // ISF positions are speaker-anchored, so scalar trim is unity.
                // The separately signalled front/back balance remains active.
                gains[*bus] += coefficient
                    * gain
                    * trim.balance_gain(*routed, routed.position()[2].abs() > f32::EPSILON);
            } else {
                debug_assert!(speaker.position()[2] <= 0.0);
            }
        }
        Ok(gains)
    }
}

#[derive(Clone, Copy)]
struct Matrix {
    speakers: &'static [Speaker],
    coefficients: &'static [f32],
}

fn select_matrix(input_count: usize, hrir: &HrirSet) -> Result<Matrix, AppError> {
    for speakers in [
        &SPEAKERS_904[..],
        &SPEAKERS_704[..],
        &SPEAKERS_702[..],
        &SPEAKERS_7[..],
        &SPEAKERS_5[..],
        &SPEAKERS_2[..],
    ] {
        if speakers.iter().all(|speaker| hrir.has_exact(*speaker)) {
            return coefficients(input_count, speakers).map(|coefficients| Matrix {
                speakers,
                coefficients,
            });
        }
    }
    Err(AppError::InvalidHrir(
        "ISF rendering requires at least exact left and right HRIR responses".into(),
    ))
}

fn coefficients(
    input_count: usize,
    speakers: &'static [Speaker],
) -> Result<&'static [f32], AppError> {
    let output_count = speakers.len();
    let coefficients: Option<&'static [f32]> = match (input_count, output_count) {
        (4, 2) => Some(&isf_tables::SR3100_TO_2),
        (8, 2) => Some(&isf_tables::SR5300_TO_2),
        (10, 2) => Some(&isf_tables::SR7300_TO_2),
        (14, 2) => Some(&isf_tables::SR9500_TO_2),
        (15, 2) => Some(&isf_tables::SR7530_TO_2),
        (30, 2) => Some(&isf_tables::SR15951_TO_2),
        (4, 5) => Some(&isf_tables::SR3100_TO_5),
        (8, 5) => Some(&isf_tables::SR5300_TO_5),
        (10, 5) => Some(&isf_tables::SR7300_TO_5),
        (14, 5) => Some(&isf_tables::SR9500_TO_5),
        (15, 5) => Some(&isf_tables::SR7530_TO_5),
        (30, 5) => Some(&isf_tables::SR15951_TO_5),
        (4, 7) => Some(&isf_tables::SR3100_TO_7),
        (8, 7) => Some(&isf_tables::SR5300_TO_7),
        (10, 7) => Some(&isf_tables::SR7300_TO_7),
        (14, 7) => Some(&isf_tables::SR9500_TO_7),
        (15, 7) => Some(&isf_tables::SR7530_TO_7),
        (30, 7) => Some(&isf_tables::SR15951_TO_7),
        (4, 9) if speakers == SPEAKERS_702 => Some(&isf_tables::SR3100_TO_702),
        (8, 9) if speakers == SPEAKERS_702 => Some(&isf_tables::SR5300_TO_702),
        (10, 9) if speakers == SPEAKERS_702 => Some(&isf_tables::SR7300_TO_702),
        (14, 9) if speakers == SPEAKERS_702 => Some(&isf_tables::SR9500_TO_702),
        (15, 9) if speakers == SPEAKERS_702 => Some(&isf_tables::SR7530_TO_702),
        (30, 9) if speakers == SPEAKERS_702 => Some(&isf_tables::SR15951_TO_702),
        (4, 11) => Some(&isf_tables::SR3100_TO_704),
        (8, 11) => Some(&isf_tables::SR5300_TO_704),
        (10, 11) => Some(&isf_tables::SR7300_TO_704),
        (14, 11) => Some(&isf_tables::SR9500_TO_704),
        (15, 11) => Some(&isf_tables::SR7530_TO_704),
        (30, 11) => Some(&isf_tables::SR15951_TO_704),
        (4, 13) => Some(&isf_tables::SR3100_TO_904),
        (8, 13) => Some(&isf_tables::SR5300_TO_904),
        (10, 13) => Some(&isf_tables::SR7300_TO_904),
        (14, 13) => Some(&isf_tables::SR9500_TO_904),
        (15, 13) => Some(&isf_tables::SR7530_TO_904),
        (30, 13) => Some(&isf_tables::SR15951_TO_904),
        _ => None,
    };
    coefficients.ok_or_else(|| {
        AppError::UnsupportedInput(format!(
            "no ISF rendering matrix exists for {input_count} inputs and {output_count} speakers"
        ))
    })
}

use Speaker::{
    FrontCenter as C, FrontLeft as L, FrontRight as R, RearLeft as Lb, RearRight as Rb,
    SideLeft as Ls, SideRight as Rs, TopFrontLeft as Tfl, TopFrontRight as Tfr, TopRearLeft as Tbl,
    TopRearRight as Tbr, TopSideLeft as Tsl, TopSideRight as Tsr, WideLeft as Lscr,
    WideRight as Rscr,
};

const SPEAKERS_2: [Speaker; 2] = [L, R];
const SPEAKERS_5: [Speaker; 5] = [L, R, C, Ls, Rs];
const SPEAKERS_7: [Speaker; 7] = [L, R, C, Ls, Rs, Lb, Rb];
const SPEAKERS_702: [Speaker; 9] = [L, R, C, Ls, Rs, Lb, Rb, Tsl, Tsr];
const SPEAKERS_704: [Speaker; 11] = [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr];
const SPEAKERS_904: [Speaker; 13] = [L, R, C, Ls, Rs, Lb, Rb, Tfl, Tfr, Tbl, Tbr, Lscr, Rscr];

#[cfg(test)]
mod tests {
    use super::{IsfConfig, Matrix, SPEAKERS_2, SPEAKERS_7, SPEAKERS_904, select_matrix};
    use crate::hrir::{HrirChannel, HrirSet, Speaker};

    #[test]
    fn selects_the_richest_exact_hrir_layout() {
        let hrir = identity_hrir(&SPEAKERS_7);
        let matrix = select_matrix(4, &hrir).unwrap();
        assert_eq!(matrix.speakers, SPEAKERS_7);
        assert_eq!(matrix.coefficients.len(), 4 * 7);
    }

    #[test]
    fn stereo_matrix_preserves_normative_negative_coefficients() {
        let hrir = identity_hrir(&SPEAKERS_2);
        let Matrix { coefficients, .. } = select_matrix(4, &hrir).unwrap();
        assert!((coefficients[2] - 1.306_938_5).abs() < 1e-6);
        assert!((coefficients[3] + 0.174_692_74).abs() < 1e-6);
    }

    #[test]
    fn rejects_reserved_isf_sizes() {
        assert!(IsfConfig::new(0, 6).is_err());
    }

    #[test]
    fn nine_channel_height_layout_uses_standard_speaker_index_order() {
        assert_eq!(SPEAKERS_904[7], Speaker::TopFrontLeft);
        assert_eq!(SPEAKERS_904[10], Speaker::TopRearRight);
        assert_eq!(SPEAKERS_904[11], Speaker::WideLeft);
    }

    fn identity_hrir(speakers: &[crate::hrir::Speaker]) -> HrirSet {
        HrirSet {
            sample_rate: 48_000,
            channels: speakers
                .iter()
                .map(|speaker| HrirChannel {
                    speaker: *speaker,
                    left: vec![1.0],
                    right: vec![1.0],
                })
                .collect(),
            directional: Vec::new(),
        }
    }
}
