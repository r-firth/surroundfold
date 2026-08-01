use std::{collections::HashMap, fs::File, io::BufWriter, path::Path, sync::Arc};

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::{
    dsp::{DEFAULT_CONVOLUTION_BLOCK, PeakLimiter, StereoConvolverBank, TpdfDither},
    error::AppError,
    finishing::FinishingEq,
    hrir::{HrirSet, Speaker},
    parametric::{ParametricHrtfModel, apply_direction_shaped_parametric_hrtf},
    render::RenderResult,
    room::{RoomCorrection, StereoRoomCorrector},
};

// Dolby's binaural renderer applies the calibrated +10 dB LFE playback gain,
// then splits it equally to both ears (-4.5 dB per branch).
const LFE_GAIN_PER_EAR: f32 = 1.883_649_1;
const EARLY_REFLECTIONS: [ReflectionSpec; 4] = [
    ReflectionSpec {
        delay_seconds: 0.007_3,
        azimuth_offset: -55.0,
        elevation_scale: -0.35,
        elevation_offset: -18.0,
        gain: 0.58,
        cutoff_hz: 11_000.0,
    },
    ReflectionSpec {
        delay_seconds: 0.011_7,
        azimuth_offset: 57.0,
        elevation_scale: -0.2,
        elevation_offset: 24.0,
        gain: 0.55,
        cutoff_hz: 9_000.0,
    },
    ReflectionSpec {
        delay_seconds: 0.021_1,
        azimuth_offset: -128.0,
        elevation_scale: 0.25,
        elevation_offset: -10.0,
        gain: 0.43,
        cutoff_hz: 7_000.0,
    },
    ReflectionSpec {
        delay_seconds: 0.033_9,
        azimuth_offset: 131.0,
        elevation_scale: 0.2,
        elevation_offset: 15.0,
        gain: 0.39,
        cutoff_hz: 5_500.0,
    },
];

#[derive(Clone, Copy)]
pub(crate) struct PanningRoute {
    pub index: usize,
    pub speaker: Option<Speaker>,
    pub direction: [f32; 3],
}

struct PreparedBuses {
    convolver: StereoConvolverBank,
    tail_blocks: Vec<usize>,
    panning_routes: Vec<PanningRoute>,
    maximum_impulse: usize,
    parametric_model: Option<Arc<ParametricHrtfModel>>,
}

#[derive(Clone, Copy)]
struct ReflectionSpec {
    delay_seconds: f32,
    azimuth_offset: f32,
    elevation_scale: f32,
    elevation_offset: f32,
    gain: f32,
    cutoff_hz: f32,
}

struct ReflectionTap {
    destination: usize,
    delay: usize,
    gain: f32,
    low_pass_coefficient: f32,
    filtered: f32,
}

struct ReflectionSource {
    delay_line: Vec<f32>,
    cursor: usize,
    taps: Vec<ReflectionTap>,
    remaining: usize,
}

struct EarlyReflectionField {
    inputs: Vec<f32>,
    sources: Vec<Option<ReflectionSource>>,
    active_sources: Vec<usize>,
    maximum_tail: usize,
}

impl EarlyReflectionField {
    fn new(sample_rate: u32, bus_count: usize, routes: &[PanningRoute]) -> Self {
        let taps = EARLY_REFLECTIONS
            .iter()
            .map(|spec| {
                let delay = reflection_delay_frames(spec, sample_rate);
                (spec, delay.max(1))
            })
            .collect::<Vec<_>>();
        let maximum_delay = taps.iter().map(|(_, delay)| *delay).max().unwrap_or(0);
        // This lets the slowest one-pole response decay below the 24-bit
        // output noise floor after its final delayed sample.
        let filter_tail = usize::try_from(sample_rate.div_ceil(1_000)).unwrap_or(usize::MAX);
        let maximum_tail = maximum_delay.saturating_add(filter_tail);
        let delay_line_len = maximum_delay + 1;
        let mut sources = (0..bus_count).map(|_| None).collect::<Vec<_>>();
        for route in routes {
            let source_taps = taps
                .iter()
                .map(|(spec, delay)| ReflectionTap {
                    destination: closest_route(routes, reflected_direction(route.direction, spec)),
                    delay: *delay,
                    gain: spec.gain,
                    low_pass_coefficient: one_pole_coefficient(spec.cutoff_hz, sample_rate),
                    filtered: 0.0,
                })
                .collect();
            sources[route.index] = Some(ReflectionSource {
                delay_line: vec![0.0; delay_line_len],
                cursor: 0,
                taps: source_taps,
                remaining: 0,
            });
        }
        Self {
            inputs: vec![0.0; bus_count],
            sources,
            active_sources: Vec::new(),
            maximum_tail,
        }
    }

    fn add(&mut self, bus: usize, sample: f32) -> Result<(), AppError> {
        let input = self.inputs.get_mut(bus).ok_or_else(|| {
            AppError::Render(format!("early-reflection source bus {bus} does not exist"))
        })?;
        if let Some(source) = self.sources[bus].as_mut() {
            *input += sample;
            if sample != 0.0 {
                if source.remaining == 0 {
                    self.active_sources.push(bus);
                }
                source.remaining = self.maximum_tail + 1;
            }
        }
        Ok(())
    }

    fn render_frame(&mut self, buses: &mut [Vec<f32>], bus_has_input: &mut [bool], frame: usize) {
        let active_sources = std::mem::take(&mut self.active_sources);
        for source_index in active_sources {
            let input = std::mem::take(&mut self.inputs[source_index]);
            let source = self.sources[source_index]
                .as_mut()
                .expect("active early-reflection source must exist");
            source.delay_line[source.cursor] = input;
            for tap in &mut source.taps {
                let index =
                    (source.cursor + source.delay_line.len() - tap.delay) % source.delay_line.len();
                let delayed = source.delay_line[index];
                tap.filtered += tap.low_pass_coefficient * (delayed - tap.filtered);
                let reflected = tap.filtered * tap.gain;
                buses[tap.destination][frame] += reflected;
                bus_has_input[tap.destination] |= reflected != 0.0;
            }
            source.cursor = (source.cursor + 1) % source.delay_line.len();
            source.remaining -= 1;
            if source.remaining == 0 {
                for tap in &mut source.taps {
                    tap.filtered = 0.0;
                }
            } else {
                self.active_sources.push(source_index);
            }
        }
    }

    fn tail_frames(&self) -> usize {
        self.active_sources
            .iter()
            .filter_map(|index| self.sources[*index].as_ref())
            .map(|source| source.remaining)
            .max()
            .unwrap_or(0)
    }
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)] // Positive, millisecond-scale tap delays are deliberately rounded to an audio frame.
fn reflection_delay_frames(spec: &ReflectionSpec, sample_rate: u32) -> usize {
    (spec.delay_seconds * sample_rate as f32).round() as usize
}

#[allow(clippy::cast_precision_loss)] // Audio rates are precise enough for this filter coefficient.
fn one_pole_coefficient(cutoff_hz: f32, sample_rate: u32) -> f32 {
    1.0 - (-std::f32::consts::TAU * cutoff_hz / sample_rate.max(1) as f32).exp()
}

/// Streaming virtual-speaker bus renderer shared by channel and object paths.
pub(crate) struct BinauralWriter {
    writer: WavWriter<BufWriter<File>>,
    sample_rate: u32,
    master_gain: f32,
    bus_by_speaker: HashMap<Speaker, usize>,
    panning_routes: Vec<PanningRoute>,
    parametric_model: Option<Arc<ParametricHrtfModel>>,
    convolver: StereoConvolverBank,
    buses: Vec<Vec<f32>>,
    bus_has_input: Vec<bool>,
    bus_enabled: Vec<bool>,
    bus_tail_blocks: Vec<usize>,
    bus_tail_remaining: Vec<usize>,
    filled: usize,
    convolved_left: Vec<f32>,
    convolved_right: Vec<f32>,
    stereo: Vec<f32>,
    limited: Vec<f32>,
    direct_stereo: Vec<f32>,
    early_reflections: EarlyReflectionField,
    room_correction: Option<StereoRoomCorrector>,
    finishing_eq: FinishingEq,
    limiter: PeakLimiter,
    dither: TpdfDither,
    input_frames: u64,
    processed_frames: u64,
    written_frames: u64,
    tail_frames: u64,
    peak_before_limiting: f32,
}

impl BinauralWriter {
    /// Creates one convolver per requested virtual speaker.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid gain, missing HRIR channels, convolution
    /// setup failures, or output creation failures.
    #[allow(clippy::cast_possible_truncation)] // The validated CLI gain feeds an f32 DSP pipeline.
    pub(crate) fn new(
        output: &Path,
        hrir: &HrirSet,
        room_correction: Option<&RoomCorrection>,
        gain_db: f64,
        speakers: impl IntoIterator<Item = Speaker>,
    ) -> Result<Self, AppError> {
        Self::new_with_parametric_hrtf(output, hrir, room_correction, gain_db, speakers, true)
    }

    #[cfg(test)]
    pub(crate) fn new_raw(
        output: &Path,
        hrir: &HrirSet,
        room_correction: Option<&RoomCorrection>,
        gain_db: f64,
        speakers: impl IntoIterator<Item = Speaker>,
    ) -> Result<Self, AppError> {
        Self::new_with_parametric_hrtf(output, hrir, room_correction, gain_db, speakers, false)
    }

    #[allow(clippy::cast_possible_truncation, clippy::too_many_lines)]
    fn new_with_parametric_hrtf(
        output: &Path,
        hrir: &HrirSet,
        room_correction: Option<&RoomCorrection>,
        gain_db: f64,
        speakers: impl IntoIterator<Item = Speaker>,
        parametric_hrtf: bool,
    ) -> Result<Self, AppError> {
        let master_gain = 10_f64.powf(gain_db / 20.0) as f32;
        if !master_gain.is_finite() {
            return Err(AppError::Usage(
                "--gain-db produces a non-finite linear gain".into(),
            ));
        }
        let mut unique_speakers = Vec::new();
        for speaker in speakers {
            if !unique_speakers.contains(&speaker) {
                unique_speakers.push(speaker);
            }
        }
        if unique_speakers.is_empty() {
            return Err(AppError::Render(
                "binaural renderer has no virtual speaker buses".into(),
            ));
        }
        if !unique_speakers.contains(&Speaker::Lfe) {
            unique_speakers.push(Speaker::Lfe);
        }

        let PreparedBuses {
            convolver,
            tail_blocks: bus_tail_blocks,
            panning_routes,
            maximum_impulse,
            parametric_model,
        } = prepare_buses(hrir, &unique_speakers, parametric_hrtf)?;
        let correction_tail = room_correction.map_or(0, |correction| {
            correction
                .left
                .len()
                .max(correction.right.len())
                .saturating_sub(1)
        });
        let tail_frames = maximum_impulse
            .saturating_sub(1)
            .saturating_add(correction_tail);
        let spec = WavSpec {
            channels: 2,
            sample_rate: hrir.sample_rate,
            bits_per_sample: 24,
            sample_format: SampleFormat::Int,
        };
        let writer = WavWriter::create(output, spec).map_err(|error| {
            AppError::Render(format!(
                "could not create rendered WAV {}: {error}",
                output.display()
            ))
        })?;
        let bus_by_speaker = unique_speakers
            .into_iter()
            .enumerate()
            .map(|(index, speaker)| (speaker, index))
            .collect();
        let bus_count = bus_tail_blocks.len();
        let early_reflections =
            EarlyReflectionField::new(hrir.sample_rate, bus_count, &panning_routes);

        Ok(Self {
            writer,
            sample_rate: hrir.sample_rate,
            master_gain,
            bus_by_speaker,
            panning_routes,
            parametric_model,
            buses: vec![vec![0.0; DEFAULT_CONVOLUTION_BLOCK]; bus_count],
            bus_has_input: vec![false; bus_count],
            bus_enabled: vec![false; bus_count],
            bus_tail_remaining: vec![0; bus_count],
            bus_tail_blocks,
            convolver,
            filled: 0,
            convolved_left: vec![0.0; DEFAULT_CONVOLUTION_BLOCK],
            convolved_right: vec![0.0; DEFAULT_CONVOLUTION_BLOCK],
            stereo: vec![0.0; DEFAULT_CONVOLUTION_BLOCK * 2],
            limited: Vec::with_capacity(DEFAULT_CONVOLUTION_BLOCK * 2),
            direct_stereo: vec![0.0; DEFAULT_CONVOLUTION_BLOCK * 2],
            early_reflections,
            room_correction: room_correction
                .map(|correction| StereoRoomCorrector::new(correction, DEFAULT_CONVOLUTION_BLOCK))
                .transpose()?,
            finishing_eq: FinishingEq::new(hrir.sample_rate),
            limiter: PeakLimiter::new(hrir.sample_rate),
            dither: TpdfDither::default(),
            input_frames: 0,
            processed_frames: 0,
            written_frames: 0,
            tail_frames: u64::try_from(tail_frames)
                .map_err(|error| AppError::Render(format!("FIR tail is too long: {error}")))?,
            peak_before_limiting: 0.0,
        })
    }

    #[must_use]
    pub(crate) fn bus(&self, speaker: Speaker) -> Option<usize> {
        self.bus_by_speaker.get(&speaker).copied()
    }

    pub(crate) fn panning_routes(&self) -> impl Iterator<Item = PanningRoute> + '_ {
        self.panning_routes.iter().copied()
    }

    #[must_use]
    pub(crate) fn parametric_model(&self) -> Option<Arc<ParametricHrtfModel>> {
        self.parametric_model.clone()
    }

    #[must_use]
    pub(crate) fn bus_count(&self) -> usize {
        self.buses.len()
    }

    pub(crate) fn add(&mut self, bus: usize, sample: f32) -> Result<(), AppError> {
        let target = self
            .buses
            .get_mut(bus)
            .ok_or_else(|| AppError::Render(format!("virtual-speaker bus {bus} does not exist")))?;
        let scaled = sample * self.master_gain;
        if !scaled.is_finite() {
            return Err(AppError::Render(format!(
                "non-finite sample routed to virtual-speaker bus {bus} at frame {}",
                self.input_frames
            )));
        }
        target[self.filled] += scaled;
        if !target[self.filled].is_finite() {
            return Err(AppError::Render(format!(
                "virtual-speaker bus {bus} overflowed at frame {}",
                self.input_frames
            )));
        }
        if scaled != 0.0 {
            self.bus_has_input[bus] = true;
        }
        Ok(())
    }

    pub(crate) fn add_early_reflection(&mut self, bus: usize, sample: f32) -> Result<(), AppError> {
        self.early_reflections.add(bus, sample * self.master_gain)
    }

    pub(crate) fn add_direct(&mut self, left: f32, right: f32) {
        self.direct_stereo[self.filled * 2] += left * self.master_gain;
        self.direct_stereo[self.filled * 2 + 1] += right * self.master_gain;
    }

    pub(crate) fn end_frame(&mut self) -> Result<(), AppError> {
        self.commit_frame();
        self.input_frames = self
            .input_frames
            .checked_add(1)
            .ok_or_else(|| AppError::Render("rendered frame count overflowed".into()))?;
        if self.filled == DEFAULT_CONVOLUTION_BLOCK {
            self.process_block(DEFAULT_CONVOLUTION_BLOCK)?;
        }
        Ok(())
    }

    /// Flushes the exact combined HRIR and room-FIR tail and finalizes the WAV.
    ///
    /// # Errors
    ///
    /// Returns an error for convolution or WAV output failures.
    pub(crate) fn finish(mut self) -> Result<RenderResult, AppError> {
        if self.input_frames == 0 {
            return Err(AppError::Render(
                "selected stream decoded to zero audio frames".into(),
            ));
        }
        let reflection_tail = self.early_reflections.tail_frames();
        let target_frames = self
            .input_frames
            .checked_add(u64::try_from(reflection_tail).map_err(|error| {
                AppError::Render(format!("early-reflection tail is too long: {error}"))
            })?)
            .ok_or_else(|| AppError::Render("rendered duration overflowed".into()))?
            .checked_add(self.tail_frames)
            .ok_or_else(|| AppError::Render("rendered duration overflowed".into()))?;
        for _ in 0..reflection_tail {
            self.commit_frame();
            if self.filled == DEFAULT_CONVOLUTION_BLOCK {
                self.process_block(DEFAULT_CONVOLUTION_BLOCK)?;
            }
        }
        while self.processed_frames < target_frames {
            let remaining = target_frames - self.processed_frames;
            let write_frames = usize::try_from(remaining)
                .unwrap_or(DEFAULT_CONVOLUTION_BLOCK)
                .min(DEFAULT_CONVOLUTION_BLOCK);
            self.process_block(write_frames)?;
        }
        self.limiter.drain(&mut self.limited);
        self.write_limited()?;
        if self.written_frames != target_frames {
            return Err(AppError::Render(format!(
                "output limiter produced {} frames; expected {target_frames}",
                self.written_frames
            )));
        }
        self.writer.finalize().map_err(|error| {
            AppError::Render(format!("could not finalize rendered WAV: {error}"))
        })?;
        Ok(RenderResult {
            sample_rate: self.sample_rate,
            frames: self.written_frames,
            peak_before_limiting: self.peak_before_limiting,
        })
    }

    fn commit_frame(&mut self) {
        self.early_reflections
            .render_frame(&mut self.buses, &mut self.bus_has_input, self.filled);
        self.filled += 1;
    }

    fn process_block(&mut self, write_frames: usize) -> Result<(), AppError> {
        self.stereo.copy_from_slice(&self.direct_stereo);
        self.direct_stereo.fill(0.0);
        for bus in 0..self.buses.len() {
            let has_input = self.bus_has_input[bus];
            self.bus_enabled[bus] = has_input || self.bus_tail_remaining[bus] != 0;
            if has_input {
                self.bus_tail_remaining[bus] = self.bus_tail_blocks[bus];
            } else if self.bus_tail_remaining[bus] != 0 {
                self.bus_tail_remaining[bus] -= 1;
            }
        }
        self.convolver.process(
            &self.buses,
            &self.bus_enabled,
            &mut self.convolved_left,
            &mut self.convolved_right,
        )?;
        for frame in 0..DEFAULT_CONVOLUTION_BLOCK {
            self.stereo[frame * 2] += self.convolved_left[frame];
            self.stereo[frame * 2 + 1] += self.convolved_right[frame];
        }
        for bus in 0..self.buses.len() {
            self.buses[bus].fill(0.0);
            self.bus_has_input[bus] = false;
        }
        if let Some(correction) = self.room_correction.as_mut() {
            correction.process(&mut self.stereo)?;
        }
        let output = &mut self.stereo[..write_frames * 2];
        self.finishing_eq.process(output);
        self.peak_before_limiting = output
            .iter()
            .copied()
            .map(f32::abs)
            .fold(self.peak_before_limiting, f32::max);
        self.limiter.process(output, &mut self.limited);
        self.write_limited()?;
        self.processed_frames = self
            .processed_frames
            .checked_add(u64::try_from(write_frames).map_err(|error| {
                AppError::Render(format!("rendered frame count overflowed: {error}"))
            })?)
            .ok_or_else(|| AppError::Render("rendered frame count overflowed".into()))?;
        self.filled = 0;
        Ok(())
    }

    fn write_limited(&mut self) -> Result<(), AppError> {
        for sample in &self.limited {
            self.writer
                .write_sample(self.dither.quantize_i24(*sample))
                .map_err(|error| AppError::Render(format!("WAV write failed: {error}")))?;
        }
        let frames = self.limited.len() / 2;
        self.written_frames = self
            .written_frames
            .checked_add(u64::try_from(frames).map_err(|error| {
                AppError::Render(format!("rendered frame count overflowed: {error}"))
            })?)
            .ok_or_else(|| AppError::Render("rendered frame count overflowed".into()))?;
        Ok(())
    }
}

fn closest_route(routes: &[PanningRoute], direction: [f32; 3]) -> usize {
    routes
        .iter()
        .max_by(|left, right| {
            direction_dot(left.direction, direction)
                .total_cmp(&direction_dot(right.direction, direction))
        })
        .map_or(0, |route| route.index)
}

fn reflected_direction(direction: [f32; 3], spec: &ReflectionSpec) -> [f32; 3] {
    let length = direction
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    let direction = if length > f32::EPSILON {
        direction.map(|value| value / length)
    } else {
        [0.0, 1.0, 0.0]
    };
    let azimuth = direction[0].atan2(direction[1]).to_degrees() + spec.azimuth_offset;
    let elevation = direction[2]
        .asin()
        .to_degrees()
        .mul_add(spec.elevation_scale, spec.elevation_offset);
    let azimuth = azimuth.to_radians();
    let elevation = elevation.clamp(-75.0, 75.0).to_radians();
    let horizontal = elevation.cos();
    [
        azimuth.sin() * horizontal,
        azimuth.cos() * horizontal,
        elevation.sin(),
    ]
}

fn direction_dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left.into_iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn prepare_buses(
    hrir: &HrirSet,
    speakers: &[Speaker],
    parametric_hrtf: bool,
) -> Result<PreparedBuses, AppError> {
    let capacity = speakers.len() + hrir.directional.len();
    let mut filters = Vec::with_capacity(capacity);
    let mut tail_blocks = Vec::with_capacity(capacity);
    let mut panning_routes = Vec::with_capacity(capacity);
    let mut maximum_impulse = 1;

    for speaker in speakers {
        let index = filters.len();
        if *speaker == Speaker::Lfe {
            filters.push((vec![LFE_GAIN_PER_EAR], vec![LFE_GAIN_PER_EAR]));
            tail_blocks.push(0);
            continue;
        }
        let channel = hrir.channel(*speaker).ok_or_else(|| {
            AppError::InvalidHrir(format!("HRIR has no usable response for {speaker:?}"))
        })?;
        let impulse_length = channel.left.len().max(channel.right.len());
        maximum_impulse = maximum_impulse.max(impulse_length);
        filters.push((channel.left.clone(), channel.right.clone()));
        tail_blocks.push(convolution_tail_blocks(impulse_length));
        panning_routes.push(PanningRoute {
            index,
            speaker: Some(*speaker),
            direction: speaker.position(),
        });
    }
    for channel in &hrir.directional {
        let index = filters.len();
        let impulse_length = channel.left.len().max(channel.right.len());
        maximum_impulse = maximum_impulse.max(impulse_length);
        filters.push((channel.left.clone(), channel.right.clone()));
        tail_blocks.push(convolution_tail_blocks(impulse_length));
        panning_routes.push(PanningRoute {
            index,
            speaker: None,
            direction: channel.direction,
        });
    }
    let parametric_model = if parametric_hrtf {
        apply_direction_shaped_parametric_hrtf(&mut filters, &panning_routes, hrir.sample_rate)
            .map(Arc::new)
    } else {
        None
    };
    if parametric_model.is_some() {
        for (index, (left, right)) in filters.iter().enumerate() {
            let impulse_length = left.len().max(right.len());
            maximum_impulse = maximum_impulse.max(impulse_length);
            tail_blocks[index] = convolution_tail_blocks(impulse_length);
        }
    }
    let convolver = StereoConvolverBank::new(
        filters
            .iter()
            .map(|(left, right)| (left.as_slice(), right.as_slice())),
        DEFAULT_CONVOLUTION_BLOCK,
    )?;
    Ok(PreparedBuses {
        convolver,
        tail_blocks,
        panning_routes,
        maximum_impulse,
        parametric_model,
    })
}

const fn convolution_tail_blocks(impulse_length: usize) -> usize {
    impulse_length
        .saturating_sub(1)
        .div_ceil(DEFAULT_CONVOLUTION_BLOCK)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

    use super::{BinauralWriter, EARLY_REFLECTIONS};
    use crate::{
        finishing::FinishingEq,
        hrir::{HrirChannel, HrirSet, Speaker},
    };

    #[test]
    fn flushes_the_exact_impulse_tail() {
        let directory = tempfile::tempdir().unwrap();
        let hrir_path = directory.path().join("hrir.wav");
        let output = directory.path().join("out.wav");
        write_hrir(&hrir_path);
        let hrir = HrirSet::load_concatenated_wave(&hrir_path).unwrap();
        let mut writer =
            BinauralWriter::new_raw(&output, &hrir, None, 0.0, [Speaker::FrontLeft]).unwrap();
        let bus = writer.bus(Speaker::FrontLeft).unwrap();
        writer.add(bus, 1.0).unwrap();
        writer.end_frame().unwrap();
        let result = writer.finish().unwrap();

        assert_eq!(result.frames, 128);
        let reader = WavReader::open(output).unwrap();
        assert_eq!(reader.duration(), 128);
        assert_eq!(reader.spec().bits_per_sample, 24);
    }

    #[test]
    fn sparse_bus_processing_retains_a_multi_block_impulse_tail() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("long-tail.wav");
        let mut impulse = vec![0.0; 1_025];
        impulse[0] = 0.5;
        impulse[1_024] = 0.25;
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![HrirChannel {
                speaker: Speaker::FrontCenter,
                left: impulse.clone(),
                right: impulse,
            }],
            directional: Vec::new(),
        };
        let mut writer =
            BinauralWriter::new_raw(&output, &hrir, None, 0.0, [Speaker::FrontCenter]).unwrap();
        let bus = writer.bus(Speaker::FrontCenter).unwrap();
        writer.add(bus, 0.5).unwrap();
        writer.end_frame().unwrap();
        let result = writer.finish().unwrap();

        assert_eq!(result.frames, 1_025);
        let samples = WavReader::open(output)
            .unwrap()
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(samples.len(), 2_050);
        let mut expected = vec![0.0_f32; 2_050];
        expected[0] = 0.25;
        expected[1] = 0.25;
        expected[2_048] = 0.125;
        expected[2_049] = 0.125;
        FinishingEq::new(48_000).process(&mut expected);
        let tail_left = f64::from(samples[2_048]) / 8_388_607.0;
        let tail_right = f64::from(samples[2_049]) / 8_388_607.0;
        assert!((tail_left - f64::from(expected[2_048])).abs() < 1e-5);
        assert!((tail_right - f64::from(expected[2_049])).abs() < 1e-5);
    }

    #[test]
    fn lfe_uses_the_calibrated_direct_binaural_gain() {
        let directory = tempfile::tempdir().unwrap();
        let hrir_path = directory.path().join("hrir.wav");
        let output = directory.path().join("lfe.wav");
        write_hrir(&hrir_path);
        let hrir = HrirSet::load_concatenated_wave(&hrir_path).unwrap();
        let mut writer =
            BinauralWriter::new_raw(&output, &hrir, None, 0.0, [Speaker::FrontLeft]).unwrap();
        let lfe = writer.bus(Speaker::Lfe).unwrap();
        writer.add(lfe, 0.1).unwrap();
        writer.end_frame().unwrap();
        writer.finish().unwrap();

        let samples = WavReader::open(output)
            .unwrap()
            .samples::<i32>()
            .take(2)
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let input = 0.1 * super::LFE_GAIN_PER_EAR;
        let mut expected = [input, input];
        FinishingEq::new(48_000).process(&mut expected);
        for (sample, expected) in samples.into_iter().zip(expected) {
            let normalized = f64::from(sample) / 8_388_607.0;
            assert!((normalized - f64::from(expected)).abs() < 1e-5);
        }
    }

    #[test]
    fn early_reflections_are_delayed_and_have_a_finite_tail() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("early-reflections.wav");
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![HrirChannel {
                speaker: Speaker::FrontCenter,
                left: vec![1.0],
                right: vec![1.0],
            }],
            directional: Vec::new(),
        };
        let mut writer =
            BinauralWriter::new_raw(&output, &hrir, None, 0.0, [Speaker::FrontCenter]).unwrap();
        let bus = writer.bus(Speaker::FrontCenter).unwrap();
        let first_delay = super::reflection_delay_frames(&EARLY_REFLECTIONS[0], hrir.sample_rate);
        let maximum_tail = writer.early_reflections.maximum_tail;

        writer.add_early_reflection(bus, 1.0).unwrap();
        writer.end_frame().unwrap();
        let result = writer.finish().unwrap();

        assert_eq!(result.frames, u64::try_from(maximum_tail + 1).unwrap());
        let samples = WavReader::open(output)
            .unwrap()
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(
            samples[..first_delay * 2]
                .iter()
                .all(|sample| sample.abs() <= 1)
        );
        let first_audible = samples
            .chunks_exact(2)
            .position(|frame| frame.iter().any(|sample| sample.abs() > 100_000));
        assert_eq!(first_audible, Some(first_delay));
    }

    fn write_hrir(path: &Path) {
        let spec = WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        for channel in 0..2 {
            for sample in 0..128 {
                let value = if sample == 0 {
                    if channel == 0 { 1.0 } else { 0.9 }
                } else if sample == 127 {
                    0.1
                } else {
                    0.0
                };
                writer.write_sample(value).unwrap();
                writer.write_sample(value).unwrap();
            }
        }
        writer.finalize().unwrap();
    }
}
