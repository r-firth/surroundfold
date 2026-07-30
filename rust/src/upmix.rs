use std::f64::consts::PI;

use crate::{
    binaural::BinauralWriter,
    error::AppError,
    hrir::{HrirSet, Speaker},
    object::{ObjectTrim, ObjectZone},
    spatial::{SpatialPanner, direct_stereo_gains},
};

const HEIGHT_CROSSOVER_HZ: f64 = 250.0;
const HEIGHT_ANALYSIS_FRAMES: usize = 64;

#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)] // Independent processing switches mirror the public CLI.
pub(crate) struct ChannelProcessingOptions {
    pub matrix: bool,
    pub upconvert: bool,
    pub effect: f32,
    pub smoothness: f32,
    pub surround_swap: bool,
    pub mute_bed: bool,
    pub mute_ground: bool,
    pub speaker_virtualizer: bool,
}

pub(crate) struct ChannelProcessor {
    source_channels: usize,
    channels: Vec<ProcessedChannel>,
    mixed: Vec<f32>,
    panner: SpatialPanner,
    options: ChannelProcessingOptions,
    sample_rate: u32,
}

struct ProcessedChannel {
    speaker: Speaker,
    terms: Vec<MixTerm>,
    ground_bus: usize,
    height: Option<HeightExtractor>,
    height_gains: Vec<f32>,
}

#[derive(Clone, Copy)]
struct MixTerm {
    source: usize,
    gain: f32,
}

impl ChannelProcessor {
    pub(crate) fn new(
        source_speakers: &[Speaker],
        hrir: &HrirSet,
        writer: &BinauralWriter,
        options: ChannelProcessingOptions,
    ) -> Result<Self, AppError> {
        let plan = MatrixPlan::new(source_speakers, hrir, options.matrix)?;
        let panner = SpatialPanner::new(writer)?;
        let mut channels = Vec::with_capacity(plan.channels.len());
        for planned in plan.channels {
            let speaker = if options.surround_swap {
                planned.speaker.surround_swapped()
            } else {
                planned.speaker
            };
            let ground_bus = if speaker == Speaker::Lfe {
                writer.bus(Speaker::Lfe)
            } else {
                hrir.resolved_speaker(speaker)
                    .and_then(|resolved| writer.bus(resolved))
            }
            .ok_or_else(|| {
                AppError::InvalidHrir(format!(
                    "HRIR has no route for processed channel {speaker:?}"
                ))
            })?;
            let eligible_for_height = options.upconvert
                && speaker.position()[2] == 0.0
                && !matches!(speaker, Speaker::Lfe | Speaker::FrontCenter);
            let height = eligible_for_height
                .then(|| HeightExtractor::new(hrir.sample_rate, HEIGHT_CROSSOVER_HZ));
            let height_gains = if height.is_some() {
                panner.gains(
                    speaker.position(),
                    [0.0; 3],
                    1.0,
                    false,
                    ObjectZone::All,
                    true,
                    0.0,
                    ObjectTrim::default(),
                    false,
                )?
            } else {
                Vec::new()
            };
            channels.push(ProcessedChannel {
                speaker,
                terms: planned.terms,
                ground_bus,
                height,
                height_gains,
            });
        }
        Ok(Self {
            source_channels: source_speakers.len(),
            mixed: vec![0.0; channels.len()],
            channels,
            panner,
            options,
            sample_rate: hrir.sample_rate,
        })
    }

    pub(crate) fn process_interleaved(
        &mut self,
        input: &[f32],
        writer: &mut BinauralWriter,
    ) -> Result<(), AppError> {
        if input.len() % self.source_channels != 0 {
            return Err(AppError::Render(
                "channel-processing input has an incomplete frame".into(),
            ));
        }
        for source_frame in input.chunks_exact(self.source_channels) {
            for (mixed, channel) in self.mixed.iter_mut().zip(&self.channels) {
                *mixed = channel
                    .terms
                    .iter()
                    .map(|term| source_frame[term.source] * term.gain)
                    .sum();
            }
            for (index, sample) in self.mixed.iter().copied().enumerate() {
                let channel = &mut self.channels[index];
                let speaker = channel.speaker;
                let ground_bus = channel.ground_bus;
                if let Some(extractor) = channel.height.as_mut() {
                    let split = extractor.process(
                        sample,
                        self.options.effect,
                        self.options.smoothness,
                        self.sample_rate,
                    );
                    if !(self.options.mute_bed || self.options.mute_ground) {
                        route_static(
                            writer,
                            speaker,
                            ground_bus,
                            split.ground,
                            self.options.speaker_virtualizer,
                        )?;
                    }
                    if split.height_changed {
                        let mut position = channel.speaker.position();
                        position[2] = extractor.height();
                        channel.height_gains = self.panner.gains(
                            position,
                            [0.0; 3],
                            1.0,
                            false,
                            ObjectZone::All,
                            true,
                            0.0,
                            ObjectTrim::default(),
                            false,
                        )?;
                    }
                    if !(self.options.mute_ground && extractor.height() <= 0.0) {
                        for (bus, gain) in channel.height_gains.iter().copied().enumerate() {
                            if gain != 0.0 {
                                writer.add(bus, split.height * gain)?;
                            }
                        }
                    }
                } else if !(self.options.mute_bed || self.options.mute_ground) {
                    route_static(
                        writer,
                        speaker,
                        ground_bus,
                        sample,
                        self.options.speaker_virtualizer,
                    )?;
                }
            }
            writer.end_frame()?;
        }
        Ok(())
    }
}

fn route_static(
    writer: &mut BinauralWriter,
    speaker: Speaker,
    ground_bus: usize,
    sample: f32,
    speaker_virtualizer: bool,
) -> Result<(), AppError> {
    if speaker == Speaker::Lfe {
        writer.add(ground_bus, sample)
    } else if speaker_virtualizer && speaker.position()[2] == 0.0 {
        let [left, right] = direct_stereo_gains(speaker);
        writer.add_direct(sample * left, sample * right);
        Ok(())
    } else {
        writer.add(ground_bus, sample)
    }
}

struct MatrixPlan {
    channels: Vec<PlannedChannel>,
}

struct PlannedChannel {
    speaker: Speaker,
    terms: Vec<MixTerm>,
}

impl MatrixPlan {
    fn new(source: &[Speaker], hrir: &HrirSet, enabled: bool) -> Result<Self, AppError> {
        if source.is_empty() {
            return Err(AppError::UnsupportedInput(
                "cannot process an empty channel layout".into(),
            ));
        }
        let mut channels = source
            .iter()
            .copied()
            .enumerate()
            .map(|(index, speaker)| PlannedChannel {
                speaker,
                terms: vec![MixTerm {
                    source: index,
                    gain: 1.0,
                }],
            })
            .collect::<Vec<_>>();
        if !enabled {
            return Ok(Self { channels });
        }

        let left = position(source, Speaker::FrontLeft);
        let right = position(source, Speaker::FrontRight);
        let (Some(left), Some(right)) = (left, right) else {
            return Ok(Self { channels });
        };
        if !contains(&channels, Speaker::FrontCenter) {
            channels.push(PlannedChannel {
                speaker: Speaker::FrontCenter,
                terms: vec![
                    MixTerm {
                        source: left,
                        gain: 0.5,
                    },
                    MixTerm {
                        source: right,
                        gain: 0.5,
                    },
                ],
            });
        }
        if !contains(&channels, Speaker::SideLeft) {
            channels.push(difference_channel(Speaker::SideLeft, left, right, 0.5));
        }
        if !contains(&channels, Speaker::SideRight) {
            channels.push(difference_channel(Speaker::SideRight, left, right, -0.5));
        }

        let has_rear_hrir = hrir
            .channels
            .iter()
            .any(|channel| channel.speaker == Speaker::RearLeft)
            && hrir
                .channels
                .iter()
                .any(|channel| channel.speaker == Speaker::RearRight);
        if has_rear_hrir
            && !contains(&channels, Speaker::RearLeft)
            && !contains(&channels, Speaker::RearRight)
        {
            split_side_to_rear(&mut channels, Speaker::SideLeft, Speaker::RearLeft)?;
            split_side_to_rear(&mut channels, Speaker::SideRight, Speaker::RearRight)?;
        }
        Ok(Self { channels })
    }
}

fn position(speakers: &[Speaker], wanted: Speaker) -> Option<usize> {
    speakers.iter().position(|speaker| *speaker == wanted)
}

fn contains(channels: &[PlannedChannel], wanted: Speaker) -> bool {
    channels.iter().any(|channel| channel.speaker == wanted)
}

fn difference_channel(speaker: Speaker, left: usize, right: usize, gain: f32) -> PlannedChannel {
    PlannedChannel {
        speaker,
        terms: vec![
            MixTerm { source: left, gain },
            MixTerm {
                source: right,
                gain: -gain,
            },
        ],
    }
}

fn split_side_to_rear(
    channels: &mut Vec<PlannedChannel>,
    side: Speaker,
    rear: Speaker,
) -> Result<(), AppError> {
    let side = channels
        .iter_mut()
        .find(|channel| channel.speaker == side)
        .ok_or_else(|| AppError::Render("matrix plan has no side channel to extend".into()))?;
    for term in &mut side.terms {
        term.gain *= 0.5;
    }
    let rear_terms = side.terms.clone();
    channels.push(PlannedChannel {
        speaker: rear,
        terms: rear_terms,
    });
    Ok(())
}

struct HeightExtractor {
    lowpass: CascadedLowpass,
    last_high: f32,
    last_depth: f32,
    last_sample: f32,
    maximum_high: f32,
    maximum_depth: f32,
    analysis_position: usize,
    height: f32,
}

struct HeightSplit {
    ground: f32,
    height: f32,
    height_changed: bool,
}

impl HeightExtractor {
    fn new(sample_rate: u32, crossover_hz: f64) -> Self {
        Self {
            lowpass: CascadedLowpass::new(sample_rate, crossover_hz),
            last_high: 0.0,
            last_depth: 0.0,
            last_sample: 0.0,
            maximum_high: 0.0001,
            maximum_depth: 0.0001,
            analysis_position: 0,
            height: 0.0,
        }
    }

    fn process(
        &mut self,
        sample: f32,
        effect: f32,
        smoothness: f32,
        sample_rate: u32,
    ) -> HeightSplit {
        let ground = self.lowpass.process(sample);
        let height = sample - ground;

        self.last_high = 0.9 * (self.last_high + sample - self.last_sample);
        self.maximum_high = self.maximum_high.max(self.last_high.abs());
        self.last_depth = self.last_depth * 0.99 + self.last_high * 0.01;
        self.maximum_depth = self.maximum_depth.max(self.last_depth.abs());
        self.last_sample = sample;
        self.analysis_position += 1;
        let height_changed = self.analysis_position == HEIGHT_ANALYSIS_FRAMES;
        if height_changed {
            let target =
                ((self.maximum_high - self.maximum_depth * 1.2) * 15.0 * effect).clamp(-0.2, 1.0);
            let smooth_factor = smoothing_factor(sample_rate, smoothness);
            self.height += (target - self.height) * smooth_factor;
            self.maximum_high = 0.0001;
            self.maximum_depth = 0.0001;
            self.analysis_position = 0;
        }
        HeightSplit {
            ground,
            height,
            height_changed,
        }
    }

    const fn height(&self) -> f32 {
        self.height
    }
}

#[allow(clippy::cast_precision_loss)] // Audio rates and block sizes fit exactly at supported values.
fn smoothing_factor(sample_rate: u32, smoothness: f32) -> f32 {
    let rate = sample_rate as f32;
    1.001
        - (HEIGHT_ANALYSIS_FRAMES as f32
            + (rate - HEIGHT_ANALYSIS_FRAMES as f32) * smoothness.powf(0.1))
            / rate
}

struct CascadedLowpass {
    first: Biquad,
    second: Biquad,
}

impl CascadedLowpass {
    fn new(sample_rate: u32, frequency: f64) -> Self {
        Self {
            first: Biquad::butterworth_lowpass(sample_rate, frequency),
            second: Biquad::butterworth_lowpass(sample_rate, frequency),
        }
    }

    fn process(&mut self, sample: f32) -> f32 {
        let sample = self.first.process(sample);
        self.second.process(sample)
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
    fn butterworth_lowpass(sample_rate: u32, frequency: f64) -> Self {
        let omega = 2.0 * PI * frequency / f64::from(sample_rate);
        let cosine = omega.cos();
        let alpha = omega.sin() / std::f64::consts::SQRT_2;
        let a0 = 1.0 + alpha;
        Self {
            b0: (1.0 - cosine) * 0.5 / a0,
            b1: (1.0 - cosine) / a0,
            b2: (1.0 - cosine) * 0.5 / a0,
            a1: -2.0 * cosine / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[allow(clippy::cast_possible_truncation)] // f64 state minimizes drift before returning to f32 DSP.
    fn process(&mut self, sample: f32) -> f32 {
        let input = f64::from(sample);
        let output = self.b0 * input + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = input;
        self.y2 = self.y1;
        self.y1 = output;
        output as f32
    }
}

#[cfg(test)]
mod tests {
    use super::{HeightExtractor, MatrixPlan};
    use crate::hrir::{HrirChannel, HrirSet, Speaker};

    #[test]
    fn stereo_matrix_creates_center_and_opposite_phase_sides() {
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: Vec::new(),
            directional: Vec::new(),
        };
        let plan =
            MatrixPlan::new(&[Speaker::FrontLeft, Speaker::FrontRight], &hrir, true).unwrap();
        let center = plan
            .channels
            .iter()
            .find(|channel| channel.speaker == Speaker::FrontCenter)
            .unwrap();
        assert_eq!(center.terms.len(), 2);
        let left_side = plan
            .channels
            .iter()
            .find(|channel| channel.speaker == Speaker::SideLeft)
            .unwrap();
        assert!((left_side.terms[0].gain - 0.5).abs() < f32::EPSILON);
        assert!((left_side.terms[1].gain + 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn crossover_outputs_reconstruct_each_input_sample() {
        let mut extractor = HeightExtractor::new(48_000, 250.0);
        for sample in [0.0, 1.0, -0.5, 0.25, 0.0] {
            let split = extractor.process(sample, 0.0, 0.8, 48_000);
            assert!((split.ground + split.height - sample).abs() < 1e-6);
        }
    }

    #[test]
    fn rear_matrix_is_only_added_for_a_real_rear_hrir_pair() {
        let impulse = |speaker| HrirChannel {
            speaker,
            left: vec![1.0],
            right: vec![1.0],
        };
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![impulse(Speaker::RearLeft), impulse(Speaker::RearRight)],
            directional: Vec::new(),
        };
        let plan =
            MatrixPlan::new(&[Speaker::FrontLeft, Speaker::FrontRight], &hrir, true).unwrap();
        assert!(
            plan.channels
                .iter()
                .any(|channel| channel.speaker == Speaker::RearLeft)
        );
        assert!(
            plan.channels
                .iter()
                .any(|channel| channel.speaker == Speaker::RearRight)
        );
    }
}
