use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
};

use crate::{
    binaural::BinauralWriter,
    cli::{DistanceRendererMode, ObjectRendererMode},
    continuous::{
        ContinuousBinaural, ContinuousHrtfGrid, FRACTIONAL_DELAY_GUARD_FRAMES,
        fractional_delay_read,
    },
    error::AppError,
    hrir::{HrirSet, Speaker},
    isf::{IsfConfig, IsfRenderer},
    object::{ObjectState, ObjectZone, SpatialUpdate},
    render::RenderResult,
    spatial::{SpatialPanner, direct_stereo_gains},
};

const UNSPECIFIED_REFLECTION_RATIO: f32 = 0.063_095_73;

#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)] // Independent render switches mirror the public CLI.
pub(crate) struct ObjectRenderOptions {
    pub surround_swap: bool,
    pub mute_bed: bool,
    pub mute_ground: bool,
    pub speaker_virtualizer: bool,
    pub object_renderer: ObjectRendererMode,
    pub distance_renderer: DistanceRendererMode,
}

#[cfg(any(feature = "embedded-truehd", test))]
pub(crate) struct ObjectPcmFrame {
    pub sample_rate: u32,
    pub sample_count: usize,
    pub channel_count: usize,
    pub samples: Vec<f32>,
    pub channel_speakers: Vec<Option<Speaker>>,
    pub isf: Option<IsfConfig>,
    pub spatial_updates: Vec<SpatialUpdate>,
}

pub(crate) struct ObjectRenderer<'a> {
    writer: BinauralWriter,
    hrir: &'a HrirSet,
    options: ObjectRenderOptions,
    panner: SpatialPanner,
    continuous_grid: Option<Arc<ContinuousHrtfGrid>>,
    isf: Option<IsfRenderer>,
    isf_sources: HashMap<usize, MovingGains>,
    objects: HashMap<usize, MovingSpatialObject>,
    retired_objects: Vec<RetiredObject>,
    object_channels: HashSet<usize>,
    pending: VecDeque<ScheduledUpdate>,
    sample_position: u64,
    sample_rate: Option<u32>,
}

impl<'a> ObjectRenderer<'a> {
    pub(crate) fn new(
        writer: BinauralWriter,
        hrir: &'a HrirSet,
        options: ObjectRenderOptions,
    ) -> Result<Self, AppError> {
        let panner = SpatialPanner::new(&writer)?;
        let continuous_grid = if options.object_renderer == ObjectRendererMode::Continuous {
            let model = writer.parametric_model().ok_or_else(|| {
                AppError::Render(
                    "continuous object rendering requires the parametric HRTF model".into(),
                )
            })?;
            Some(Arc::new(ContinuousHrtfGrid::new(&model)))
        } else {
            None
        };
        Ok(Self {
            writer,
            hrir,
            options,
            panner,
            continuous_grid,
            isf: None,
            isf_sources: HashMap::new(),
            objects: HashMap::new(),
            retired_objects: Vec::new(),
            object_channels: HashSet::new(),
            pending: VecDeque::new(),
            sample_position: 0,
            sample_rate: None,
        })
    }

    #[cfg(any(feature = "embedded-truehd", test))]
    pub(crate) fn push(&mut self, frame: ObjectPcmFrame) -> Result<(), AppError> {
        self.validate_dimensions(
            frame.sample_rate,
            frame.sample_count,
            frame.channel_count,
            &frame.samples,
            &frame.channel_speakers,
        )?;
        self.schedule_at(self.sample_position, frame.spatial_updates)?;
        self.push_samples(
            frame.sample_rate,
            frame.sample_count,
            frame.channel_count,
            &frame.samples,
            &frame.channel_speakers,
            frame.isf,
        )
    }

    pub(crate) fn schedule_at(
        &mut self,
        frame_start: u64,
        updates: Vec<SpatialUpdate>,
    ) -> Result<(), AppError> {
        for update in updates {
            let at = frame_start
                .checked_add(u64::try_from(update.sample_offset).map_err(|error| {
                    AppError::Render(format!("object metadata offset overflowed: {error}"))
                })?)
                .ok_or_else(|| AppError::Render("object metadata timeline overflowed".into()))?;
            self.pending.push_back(ScheduledUpdate { at, update });
        }
        self.pending
            .make_contiguous()
            .sort_by_key(|update| update.at);
        Ok(())
    }

    pub(crate) fn push_samples(
        &mut self,
        sample_rate: u32,
        sample_count: usize,
        channel_count: usize,
        samples: &[f32],
        channel_speakers: &[Option<Speaker>],
        isf: Option<IsfConfig>,
    ) -> Result<(), AppError> {
        self.validate_dimensions(
            sample_rate,
            sample_count,
            channel_count,
            samples,
            channel_speakers,
        )?;
        self.configure_isf(isf, channel_count)?;
        for source in samples.chunks_exact(channel_count) {
            self.apply_due_updates()?;
            self.render_retired_objects()?;
            for (channel, sample) in source.iter().copied().enumerate() {
                if let Some(object) = self.objects.get_mut(&channel) {
                    object.render(sample, &mut self.writer)?;
                } else if let Some(source) = self.isf_sources.get(&channel) {
                    for (bus, gain) in source.current.iter().copied().enumerate() {
                        if gain != 0.0 {
                            let routed = sample * gain;
                            self.writer.add(bus, routed)?;
                            self.writer
                                .add_early_reflection(bus, routed * UNSPECIFIED_REFLECTION_RATIO)?;
                        }
                    }
                } else if !self.object_channels.contains(&channel)
                    && !self.options.mute_bed
                    && let Some(speaker) = channel_speakers[channel]
                {
                    self.render_bed_sample(speaker, sample)?;
                }
            }
            self.writer.end_frame()?;
            for source in self.isf_sources.values_mut() {
                source.advance();
            }
            for object in self.objects.values_mut() {
                object.advance(&self.panner)?;
            }
            self.sample_position = self
                .sample_position
                .checked_add(1)
                .ok_or_else(|| AppError::Render("object sample timeline overflowed".into()))?;
        }
        Ok(())
    }

    fn configure_isf(
        &mut self,
        config: Option<IsfConfig>,
        channel_count: usize,
    ) -> Result<(), AppError> {
        let Some(config) = config else {
            return Ok(());
        };
        let end = config
            .start_channel
            .checked_add(config.channel_count)
            .ok_or_else(|| AppError::Render("ISF channel range overflowed".into()))?;
        if end > channel_count {
            return Err(AppError::Render(format!(
                "ISF channel range {}..{end} exceeds {channel_count} decoded channels",
                config.start_channel
            )));
        }
        if let Some(renderer) = &self.isf {
            if renderer.config() != config {
                return Err(AppError::UnsupportedInput(
                    "ISF configuration changes within the stream".into(),
                ));
            }
            return Ok(());
        }

        let renderer = IsfRenderer::new(
            config,
            &self.writer,
            self.hrir,
            self.options.surround_swap,
            self.options.mute_bed,
            self.options.mute_ground,
        )?;
        for source_channel in config.start_channel..end {
            if let Some(object) = self.objects.remove(&source_channel) {
                self.retire_object(object);
            }
            self.object_channels.insert(source_channel);
            let target = renderer.gains(
                source_channel,
                true,
                1.0,
                crate::object::ObjectTrim::default(),
            )?;
            self.isf_sources
                .insert(source_channel, MovingGains::new(target));
        }
        self.isf = Some(renderer);
        Ok(())
    }

    fn render_bed_sample(&mut self, speaker: Speaker, sample: f32) -> Result<(), AppError> {
        let speaker = if self.options.surround_swap {
            speaker.surround_swapped()
        } else {
            speaker
        };
        if self.options.mute_ground && speaker.position()[2] <= 0.0 {
            return Ok(());
        }
        if speaker == Speaker::Lfe {
            let bus = self
                .writer
                .bus(Speaker::Lfe)
                .ok_or_else(|| AppError::Render("missing calibrated binaural LFE bus".into()))?;
            return self.writer.add(bus, sample);
        }
        if self.options.speaker_virtualizer && speaker.position()[2] == 0.0 {
            let [left, right] = direct_stereo_gains(speaker);
            self.writer.add_direct(sample * left, sample * right);
            return Ok(());
        }
        let resolved = self.hrir.resolved_speaker(speaker).ok_or_else(|| {
            AppError::InvalidHrir(format!("HRIR has no route for bed channel {speaker:?}"))
        })?;
        let bus = self.writer.bus(resolved).ok_or_else(|| {
            AppError::Render(format!("missing virtual-speaker bus for {resolved:?}"))
        })?;
        self.writer.add(bus, sample)?;
        self.writer
            .add_early_reflection(bus, sample * UNSPECIFIED_REFLECTION_RATIO)
    }

    fn validate_dimensions(
        &mut self,
        sample_rate: u32,
        sample_count: usize,
        channel_count: usize,
        samples: &[f32],
        channel_speakers: &[Option<Speaker>],
    ) -> Result<(), AppError> {
        if sample_rate != self.hrir.sample_rate {
            return Err(AppError::UnsupportedInput(format!(
                "object-audio sample rate {sample_rate} does not match the prepared HRIR rate {}",
                self.hrir.sample_rate
            )));
        }
        if self
            .sample_rate
            .replace(sample_rate)
            .is_some_and(|previous| previous != sample_rate)
        {
            return Err(AppError::UnsupportedInput(
                "object-audio sample rate changes within the stream".into(),
            ));
        }
        let expected_samples = sample_count.checked_mul(channel_count).ok_or_else(|| {
            AppError::Render("object-audio PCM frame dimensions overflowed".into())
        })?;
        if channel_count == 0
            || samples.len() != expected_samples
            || channel_speakers.len() != channel_count
        {
            return Err(AppError::Render(
                "object-audio PCM frame has inconsistent dimensions".into(),
            ));
        }
        Ok(())
    }

    fn apply_due_updates(&mut self) -> Result<(), AppError> {
        while self
            .pending
            .front()
            .is_some_and(|update| update.at <= self.sample_position)
        {
            if let Some(scheduled) = self.pending.pop_front() {
                self.apply_update(scheduled.update)?;
            }
        }
        Ok(())
    }

    fn apply_update(&mut self, update: SpatialUpdate) -> Result<(), AppError> {
        for source_channel in 0..update.bed_speakers.len() {
            let has_object_metadata = update
                .objects
                .iter()
                .any(|state| state.source_channel == source_channel);
            if !has_object_metadata {
                if let Some(object) = self.objects.remove(&source_channel) {
                    self.retire_object(object);
                }
                if !self.isf_sources.contains_key(&source_channel) {
                    self.object_channels.remove(&source_channel);
                }
            }
        }
        for state in update.isf {
            let renderer = self.isf.as_ref().ok_or_else(|| {
                AppError::Render("received ISF metadata before an ISF program assignment".into())
            })?;
            let target =
                renderer.gains(state.source_channel, state.active, state.gain, state.trim)?;
            self.isf_sources
                .get_mut(&state.source_channel)
                .ok_or_else(|| {
                    AppError::Render(format!(
                        "ISF metadata references unconfigured channel {}",
                        state.source_channel
                    ))
                })?
                .set_target(target, update.ramp_samples);
        }
        for state in update.objects {
            self.object_channels.insert(state.source_channel);
            let target = RenderableObject::new(&state, self.options);
            match self.objects.get_mut(&state.source_channel) {
                Some(object) => {
                    object.set_target(target, update.ramp_samples, &self.panner)?;
                }
                None => {
                    self.objects.insert(
                        state.source_channel,
                        MovingSpatialObject::new(
                            target,
                            &self.panner,
                            self.continuous_grid.clone(),
                            self.options.distance_renderer,
                            self.hrir.sample_rate,
                        )?,
                    );
                }
            }
        }
        Ok(())
    }

    fn retire_object(&mut self, object: MovingSpatialObject) {
        let remaining = object.tail_frames();
        if remaining != 0 {
            self.retired_objects
                .push(RetiredObject { object, remaining });
        }
    }

    fn render_retired_objects(&mut self) -> Result<(), AppError> {
        for retired in &mut self.retired_objects {
            retired.object.render_tail(&mut self.writer)?;
            retired.remaining = retired.remaining.saturating_sub(1);
        }
        self.retired_objects
            .retain(|retired| retired.remaining != 0);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<RenderResult, AppError> {
        let active_tail = self
            .objects
            .values()
            .map(MovingSpatialObject::tail_frames)
            .max()
            .unwrap_or(0);
        let retired_tail = self
            .retired_objects
            .iter()
            .map(|retired| retired.remaining)
            .max()
            .unwrap_or(0);
        let tail_frames = active_tail.max(retired_tail);
        for _ in 0..tail_frames {
            for object in self.objects.values_mut() {
                object.render_tail(&mut self.writer)?;
            }
            self.render_retired_objects()?;
            self.writer.end_frame()?;
        }
        self.writer.finish()
    }
}

struct ScheduledUpdate {
    at: u64,
    update: SpatialUpdate,
}

struct RetiredObject {
    object: MovingSpatialObject,
    remaining: usize,
}

struct MovingGains {
    current: Vec<f32>,
    linear: Vec<f32>,
    target: Vec<f32>,
    step: Vec<f32>,
    level: f32,
    level_step: f32,
    remaining: usize,
}

impl MovingGains {
    fn new(target: Vec<f32>) -> Self {
        let level = vector_level(&target);
        Self {
            current: target.clone(),
            linear: target.clone(),
            step: vec![0.0; target.len()],
            target,
            level,
            level_step: 0.0,
            remaining: 0,
        }
    }

    #[allow(clippy::cast_precision_loss)] // OAMD ramp durations are small bounded integers.
    fn set_target(&mut self, target: Vec<f32>, ramp_samples: usize) {
        self.target = target;
        if ramp_samples == 0 {
            self.current.clone_from(&self.target);
            self.linear.clone_from(&self.target);
            self.step.fill(0.0);
            self.level = vector_level(&self.target);
            self.level_step = 0.0;
            self.remaining = 0;
            return;
        }
        let duration = ramp_samples as f32;
        self.linear.clone_from(&self.current);
        for ((step, current), target) in self.step.iter_mut().zip(&self.linear).zip(&self.target) {
            *step = (*target - *current) / duration;
        }
        self.level = vector_level(&self.current);
        self.level_step = (vector_level(&self.target) - self.level) / duration;
        self.remaining = ramp_samples;
    }

    fn advance(&mut self) {
        if self.remaining == 0 {
            return;
        }
        for (linear, step) in self.linear.iter_mut().zip(&self.step) {
            *linear += *step;
        }
        self.current.clone_from(&self.linear);
        self.level += self.level_step;
        self.remaining -= 1;
        if self.remaining == 0 {
            self.current.clone_from(&self.target);
            self.linear.clone_from(&self.target);
            self.level = vector_level(&self.target);
        } else {
            let interpolated_level = vector_level(&self.current);
            if interpolated_level > f32::EPSILON {
                let scale = self.level / interpolated_level;
                for gain in &mut self.current {
                    *gain *= scale;
                }
            }
        }
    }
}

const SPATIAL_CONTROL_INTERVAL: usize = 64;

#[derive(Clone, Copy)]
struct RenderableObject {
    bed_speaker: Option<Speaker>,
    position: [f32; 3],
    size: [f32; 3],
    gain: f32,
    snap: bool,
    zone: ObjectZone,
    elevation: bool,
    divergence: f32,
    reflection_ratio: f32,
    continuous_weight: f32,
    distance_direct: f32,
    distance_early: f32,
    distance_progress: f32,
    trim: crate::object::ObjectTrim,
}

impl RenderableObject {
    fn new(state: &ObjectState, options: ObjectRenderOptions) -> Self {
        let audible = state.active
            && !(state.bed_speaker.is_some() && options.mute_bed)
            && !(options.mute_ground && state.position[2] <= 0.0);
        let mut position = state.position;
        if state.trim.warp_y {
            // OAMD warp mode doubles the original room Y coordinate, whose
            // range runs from 0 at the front wall to 1 at the back wall.
            // Renderer Y is listener-centred and points forward, so the
            // equivalent affine transform is y' = 2y - 1.
            position[1] = position[1].mul_add(2.0, -1.0);
        }
        let distance_progress = distance_progress(state.distance_factor);
        let use_image_distance = options.distance_renderer == DistanceRendererMode::ImageSource
            && state.distance_factor.is_some()
            && state.bed_speaker != Some(Speaker::Lfe);
        let (mut distance_direct, mut distance_early) = if use_image_distance {
            distance_mix(state.distance_factor)
        } else {
            (1.0, 0.0)
        };
        if state.bed_speaker == Some(Speaker::Lfe) {
            distance_direct = 1.0;
            distance_early = 0.0;
        }
        let spread = state
            .size
            .into_iter()
            .map(f32::abs)
            .fold(state.divergence.abs(), f32::max);
        let continuous_weight = if options.object_renderer == ObjectRendererMode::Continuous
            && state.bed_speaker.is_none()
        {
            let pointness = (1.0 - spread / 0.08).clamp(0.0, 1.0);
            pointness * pointness * (3.0 - 2.0 * pointness)
        } else {
            0.0
        };
        Self {
            bed_speaker: state.bed_speaker,
            position,
            size: state.size,
            gain: if audible { state.gain } else { 0.0 },
            snap: state.snap,
            zone: state.zone,
            elevation: state.elevation,
            divergence: state.divergence,
            reflection_ratio: if state.bed_speaker == Some(Speaker::Lfe) || use_image_distance {
                0.0
            } else {
                distance_reflection_ratio(state.distance_factor)
            },
            continuous_weight,
            distance_direct,
            distance_early,
            distance_progress,
            trim: state.trim,
        }
    }

    #[allow(clippy::cast_precision_loss)] // Control intervals are at most 64 audio frames.
    fn advance(&mut self, step: &ObjectStep, frames: usize) {
        let frames = frames as f32;
        for (value, step) in self.position.iter_mut().zip(step.position) {
            *value = step.mul_add(frames, *value);
        }
        for (value, step) in self.size.iter_mut().zip(step.size) {
            *value = step.mul_add(frames, *value);
        }
        self.gain = step.gain.mul_add(frames, self.gain);
        self.divergence = step.divergence.mul_add(frames, self.divergence);
        self.reflection_ratio = step.reflection_ratio.mul_add(frames, self.reflection_ratio);
        self.continuous_weight = step
            .continuous_weight
            .mul_add(frames, self.continuous_weight);
        self.distance_direct = step.distance_direct.mul_add(frames, self.distance_direct);
        self.distance_early = step.distance_early.mul_add(frames, self.distance_early);
        self.distance_progress = step
            .distance_progress
            .mul_add(frames, self.distance_progress);
    }
}

#[derive(Clone, Copy)]
struct ObjectStep {
    position: [f32; 3],
    size: [f32; 3],
    gain: f32,
    divergence: f32,
    reflection_ratio: f32,
    continuous_weight: f32,
    distance_direct: f32,
    distance_early: f32,
    distance_progress: f32,
}

impl ObjectStep {
    #[allow(clippy::cast_precision_loss)] // OAMD ramp durations are bounded integers.
    fn between(current: RenderableObject, target: RenderableObject, frames: usize) -> Self {
        let duration = frames as f32;
        Self {
            position: std::array::from_fn(|axis| {
                (target.position[axis] - current.position[axis]) / duration
            }),
            size: std::array::from_fn(|axis| (target.size[axis] - current.size[axis]) / duration),
            gain: (target.gain - current.gain) / duration,
            divergence: (target.divergence - current.divergence) / duration,
            reflection_ratio: (target.reflection_ratio - current.reflection_ratio) / duration,
            continuous_weight: (target.continuous_weight - current.continuous_weight) / duration,
            distance_direct: (target.distance_direct - current.distance_direct) / duration,
            distance_early: (target.distance_early - current.distance_early) / duration,
            distance_progress: (target.distance_progress - current.distance_progress) / duration,
        }
    }
}

const SPEED_OF_SOUND_METRES_PER_SECOND: f32 = 343.0;
const MAXIMUM_EARLY_FIELD_SECONDS: f32 = 0.08;
const EARLY_FIELD_SURFACES: [EarlySurface; 6] = [
    EarlySurface {
        axis: 0,
        plane: -4.5,
        reflection: 0.72,
        cutoff_hz: 10_500.0,
    },
    EarlySurface {
        axis: 0,
        plane: 4.5,
        reflection: 0.72,
        cutoff_hz: 10_500.0,
    },
    EarlySurface {
        axis: 1,
        plane: -5.5,
        reflection: 0.66,
        cutoff_hz: 9_500.0,
    },
    EarlySurface {
        axis: 1,
        plane: 5.5,
        reflection: 0.66,
        cutoff_hz: 9_500.0,
    },
    EarlySurface {
        axis: 2,
        plane: -1.7,
        reflection: 0.58,
        cutoff_hz: 8_000.0,
    },
    EarlySurface {
        axis: 2,
        plane: 2.7,
        reflection: 0.52,
        cutoff_hz: 7_000.0,
    },
];

#[derive(Clone, Copy)]
struct EarlySurface {
    axis: usize,
    plane: f32,
    reflection: f32,
    cutoff_hz: f32,
}

struct EarlyTapTarget {
    delay: f32,
    low_pass_coefficient: f32,
    gains: Vec<f32>,
}

struct MovingEarlyTap {
    delay: f32,
    delay_step: f32,
    low_pass_coefficient: f32,
    low_pass_step: f32,
    filtered: f32,
    gains: SparseMovingGains,
    remaining: usize,
}

impl MovingEarlyTap {
    fn new(target: EarlyTapTarget) -> Self {
        Self {
            delay: target.delay,
            delay_step: 0.0,
            low_pass_coefficient: target.low_pass_coefficient,
            low_pass_step: 0.0,
            filtered: 0.0,
            gains: SparseMovingGains::new(target.gains),
            remaining: 0,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn set_target(&mut self, target: EarlyTapTarget, frames: usize) {
        if frames == 0 {
            self.delay = target.delay;
            self.delay_step = 0.0;
            self.low_pass_coefficient = target.low_pass_coefficient;
            self.low_pass_step = 0.0;
            self.gains.set_target(target.gains, 0);
            self.remaining = 0;
            return;
        }
        let duration = frames as f32;
        self.delay_step = (target.delay - self.delay) / duration;
        self.low_pass_step = (target.low_pass_coefficient - self.low_pass_coefficient) / duration;
        self.gains.set_target(target.gains, frames);
        self.remaining = frames;
    }

    fn advance(&mut self) {
        self.gains.advance();
        if self.remaining == 0 {
            return;
        }
        self.delay += self.delay_step;
        self.low_pass_coefficient += self.low_pass_step;
        self.remaining -= 1;
        if self.remaining == 0 {
            self.delay_step = 0.0;
            self.low_pass_step = 0.0;
        }
    }
}

struct EarlyDistanceField {
    sample_rate: u32,
    delay_line: Vec<f32>,
    cursor: usize,
    taps: Vec<MovingEarlyTap>,
    filter_tail_frames: usize,
    remaining: usize,
}

impl EarlyDistanceField {
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_precision_loss,
        clippy::cast_sign_loss
    )]
    fn new(sample_rate: u32, direction: [f32; 3], progress: f32, panner: &SpatialPanner) -> Self {
        let capacity = (MAXIMUM_EARLY_FIELD_SECONDS * sample_rate as f32).ceil() as usize + 8;
        // One millisecond takes even the minimum 4.5 kHz absorption cutoff
        // well below the 24-bit floor. Retain the former 64-frame allowance
        // at lower rates so approved 48 kHz experimental renders stay exact.
        let filter_tail_frames = usize::try_from(sample_rate.div_ceil(1_000))
            .unwrap_or(usize::MAX)
            .max(64);
        let targets = early_field_targets(sample_rate, direction, progress, panner);
        Self {
            sample_rate,
            delay_line: vec![0.0; capacity],
            cursor: 0,
            taps: targets.into_iter().map(MovingEarlyTap::new).collect(),
            filter_tail_frames,
            remaining: 0,
        }
    }

    fn set_target(
        &mut self,
        direction: [f32; 3],
        progress: f32,
        frames: usize,
        panner: &SpatialPanner,
    ) {
        let targets = early_field_targets(self.sample_rate, direction, progress, panner);
        for (tap, target) in self.taps.iter_mut().zip(targets) {
            tap.set_target(target, frames);
        }
    }

    fn process(&mut self, input: f32, writer: &mut BinauralWriter) -> Result<(), AppError> {
        if !input.is_finite() {
            return Err(AppError::Render(
                "image-source early field received a non-finite input".into(),
            ));
        }
        if input == 0.0 && self.remaining == 0 {
            return Ok(());
        }
        if input == 0.0 {
            self.remaining -= 1;
        } else {
            self.remaining = self.maximum_tail_frames();
        }
        self.delay_line[self.cursor] = input;
        for (tap_index, tap) in self.taps.iter_mut().enumerate() {
            if !tap.delay.is_finite() || !tap.low_pass_coefficient.is_finite() {
                return Err(AppError::Render(format!(
                    "image-source early tap {tap_index} has non-finite geometry"
                )));
            }
            let delayed = fractional_delay_read(&self.delay_line, self.cursor, tap.delay);
            tap.filtered += tap.low_pass_coefficient * (delayed - tap.filtered);
            if !tap.filtered.is_finite() {
                return Err(AppError::Render(format!(
                    "image-source early tap {tap_index} became non-finite \
                     (input {input}, delayed {delayed}, coefficient {}, delay {})",
                    tap.low_pass_coefficient, tap.delay
                )));
            }
            for (bus, gain) in tap.gains.iter() {
                if !gain.is_finite() {
                    return Err(AppError::Render(format!(
                        "image-source early tap {tap_index} has a non-finite gain for bus {bus}"
                    )));
                }
                if gain != 0.0 {
                    writer.add(bus, tap.filtered * gain)?;
                }
            }
            tap.advance();
        }
        self.cursor = (self.cursor + 1) % self.delay_line.len();
        Ok(())
    }

    #[must_use]
    fn tail_frames(&self) -> usize {
        self.remaining
    }

    #[must_use]
    fn maximum_tail_frames(&self) -> usize {
        self.delay_line
            .len()
            .saturating_add(self.filter_tail_frames)
    }
}

struct SparseGain {
    index: usize,
    current: f32,
    linear: f32,
    target: f32,
    step: f32,
}

struct SparseMovingGains {
    entries: Vec<SparseGain>,
    level: f32,
    level_step: f32,
    remaining: usize,
}

impl SparseMovingGains {
    fn new(target: Vec<f32>) -> Self {
        let level = vector_level(&target);
        Self {
            entries: target
                .into_iter()
                .enumerate()
                .filter(|(_, gain)| *gain != 0.0)
                .map(|(index, gain)| SparseGain {
                    index,
                    current: gain,
                    linear: gain,
                    target: gain,
                    step: 0.0,
                })
                .collect(),
            level,
            level_step: 0.0,
            remaining: 0,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn set_target(&mut self, target: Vec<f32>, frames: usize) {
        for entry in &mut self.entries {
            entry.target = 0.0;
        }
        for (index, target) in target.into_iter().enumerate() {
            if target != 0.0 {
                if let Some(entry) = self.entries.iter_mut().find(|entry| entry.index == index) {
                    entry.target = target;
                } else {
                    self.entries.push(SparseGain {
                        index,
                        current: 0.0,
                        linear: 0.0,
                        target,
                        step: 0.0,
                    });
                }
            }
        }
        for entry in &mut self.entries {
            if frames == 0 {
                entry.current = entry.target;
                entry.linear = entry.target;
                entry.step = 0.0;
            } else {
                entry.linear = entry.current;
                entry.step = (entry.target - entry.current) / frames as f32;
            }
        }
        let target_level = self
            .entries
            .iter()
            .map(|entry| entry.target * entry.target)
            .sum::<f32>()
            .sqrt();
        if frames == 0 {
            self.level = target_level;
            self.level_step = 0.0;
        } else {
            self.level = self
                .entries
                .iter()
                .map(|entry| entry.current * entry.current)
                .sum::<f32>()
                .sqrt();
            self.level_step = (target_level - self.level) / frames as f32;
        }
        self.remaining = frames;
        if frames == 0 {
            self.entries.retain(|entry| entry.current != 0.0);
        }
    }

    fn advance(&mut self) {
        if self.remaining == 0 {
            return;
        }
        for entry in &mut self.entries {
            entry.linear += entry.step;
            entry.current = entry.linear;
        }
        self.level += self.level_step;
        self.remaining -= 1;
        if self.remaining == 0 {
            for entry in &mut self.entries {
                entry.current = entry.target;
                entry.linear = entry.target;
                entry.step = 0.0;
            }
            self.level = self
                .entries
                .iter()
                .map(|entry| entry.target * entry.target)
                .sum::<f32>()
                .sqrt();
            self.level_step = 0.0;
            self.entries.retain(|entry| entry.current != 0.0);
        } else {
            let interpolated_level = self
                .entries
                .iter()
                .map(|entry| entry.current * entry.current)
                .sum::<f32>()
                .sqrt();
            if interpolated_level > f32::EPSILON {
                let scale = self.level / interpolated_level;
                for entry in &mut self.entries {
                    entry.current *= scale;
                }
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = (usize, f32)> + '_ {
        self.entries
            .iter()
            .map(|entry| (entry.index, entry.current))
    }
}

#[allow(clippy::cast_precision_loss)]
fn early_field_targets(
    sample_rate: u32,
    direction: [f32; 3],
    progress: f32,
    panner: &SpatialPanner,
) -> Vec<EarlyTapTarget> {
    let direction = normalized_direction(direction);
    let boundary = ray_boundary_distance(direction);
    let radius = boundary * (0.35 + 0.45 * progress.clamp(0.0, 1.0));
    let source = direction.map(|axis| axis * radius);
    let mut geometry = EARLY_FIELD_SURFACES
        .iter()
        .map(|surface| {
            let mut image = source;
            image[surface.axis] = 2.0 * surface.plane - source[surface.axis];
            let path = vector_length(image).max(radius);
            let excess = (path - radius).max(0.0);
            let weight = surface.reflection * radius / path.max(f32::EPSILON);
            let cutoff = (surface.cutoff_hz * (-0.015 * excess).exp()).clamp(4_500.0, 14_000.0);
            (image, excess, weight, cutoff)
        })
        .collect::<Vec<_>>();
    let energy = geometry
        .iter()
        .map(|(_, _, weight, _)| weight * weight)
        .sum::<f32>()
        .sqrt()
        .max(f32::EPSILON);
    geometry
        .drain(..)
        .map(|(image, excess, weight, cutoff)| EarlyTapTarget {
            delay: (FRACTIONAL_DELAY_GUARD_FRAMES as f32
                + excess / SPEED_OF_SOUND_METRES_PER_SECOND * sample_rate as f32)
                .clamp(
                    FRACTIONAL_DELAY_GUARD_FRAMES as f32,
                    MAXIMUM_EARLY_FIELD_SECONDS * sample_rate as f32 - 32.0,
                ),
            low_pass_coefficient: 1.0
                - (-std::f32::consts::TAU * cutoff / sample_rate as f32).exp(),
            gains: panner.untrimmed_point_gains(normalized_direction(image), weight / energy),
        })
        .collect()
}

fn ray_boundary_distance(direction: [f32; 3]) -> f32 {
    let mut boundary = f32::INFINITY;
    for (component, negative, positive) in [
        (direction[0], 4.5, 4.5),
        (direction[1], 5.5, 5.5),
        (direction[2], 1.7, 2.7),
    ] {
        if component > f32::EPSILON {
            boundary = boundary.min(positive / component);
        } else if component < -f32::EPSILON {
            boundary = boundary.min(negative / -component);
        }
    }
    if boundary.is_finite() { boundary } else { 5.5 }
}

fn normalized_direction(direction: [f32; 3]) -> [f32; 3] {
    let length = vector_length(direction);
    if length > f32::EPSILON {
        direction.map(|value| value / length)
    } else {
        [0.0, 1.0, 0.0]
    }
}

fn vector_length(vector: [f32; 3]) -> f32 {
    vector.iter().map(|value| value * value).sum::<f32>().sqrt()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]
struct MovingSpatialObject {
    state: RenderableObject,
    target: RenderableObject,
    step: ObjectStep,
    gains: MovingGains,
    continuous: Option<ContinuousBinaural>,
    early_distance: Option<EarlyDistanceField>,
    sample_rate: u32,
    remaining: usize,
}

impl MovingSpatialObject {
    fn new(
        state: RenderableObject,
        panner: &SpatialPanner,
        continuous_grid: Option<Arc<ContinuousHrtfGrid>>,
        distance_renderer: DistanceRendererMode,
        sample_rate: u32,
    ) -> Result<Self, AppError> {
        let initial_gains = pan_object(panner, state)?;
        let direction = render_direction(panner, &initial_gains, state);
        let gains = MovingGains::new(initial_gains);
        let continuous = continuous_grid.map(|grid| ContinuousBinaural::new(grid, direction));
        let early_distance = (distance_renderer == DistanceRendererMode::ImageSource
            && state.distance_early != 0.0)
            .then(|| {
                EarlyDistanceField::new(sample_rate, direction, state.distance_progress, panner)
            });
        Ok(Self {
            state,
            target: state,
            step: ObjectStep {
                position: [0.0; 3],
                size: [0.0; 3],
                gain: 0.0,
                divergence: 0.0,
                reflection_ratio: 0.0,
                continuous_weight: 0.0,
                distance_direct: 0.0,
                distance_early: 0.0,
                distance_progress: 0.0,
            },
            gains,
            continuous,
            early_distance,
            sample_rate,
            remaining: 0,
        })
    }

    fn set_target(
        &mut self,
        target: RenderableObject,
        ramp_samples: usize,
        panner: &SpatialPanner,
    ) -> Result<(), AppError> {
        self.target = target;
        if ramp_samples == 0 {
            self.state = target;
            self.remaining = 0;
            let gains = pan_object(panner, target)?;
            let direction = render_direction(panner, &gains, target);
            if let Some(continuous) = self.continuous.as_mut() {
                continuous.set_direction(direction, 0);
            }
            self.ensure_early_distance(direction, target.distance_progress, panner);
            if let Some(early_distance) = self.early_distance.as_mut() {
                early_distance.set_target(direction, target.distance_progress, 0, panner);
            }
            self.gains.set_target(gains, 0);
            return Ok(());
        }
        self.step = ObjectStep::between(self.state, target, ramp_samples);
        self.state.snap = target.snap;
        self.state.zone = target.zone;
        self.state.elevation = target.elevation;
        self.state.trim = target.trim;
        self.remaining = ramp_samples;
        self.prepare_segment(panner)
    }

    fn advance(&mut self, panner: &SpatialPanner) -> Result<(), AppError> {
        if self.remaining == 0 {
            return Ok(());
        }
        self.gains.advance();
        self.state.advance(&self.step, 1);
        self.remaining -= 1;
        if self.remaining == 0 {
            self.state = self.target;
            let gains = pan_object(panner, self.target)?;
            let direction = render_direction(panner, &gains, self.target);
            if let Some(continuous) = self.continuous.as_mut() {
                continuous.set_direction(direction, 0);
            }
            self.ensure_early_distance(direction, self.target.distance_progress, panner);
            if let Some(early_distance) = self.early_distance.as_mut() {
                early_distance.set_target(direction, self.target.distance_progress, 0, panner);
            }
            self.gains.set_target(gains, 0);
        } else if self.gains.remaining == 0 {
            self.prepare_segment(panner)?;
        }
        Ok(())
    }

    fn prepare_segment(&mut self, panner: &SpatialPanner) -> Result<(), AppError> {
        let frames = self.remaining.min(SPATIAL_CONTROL_INTERVAL);
        let mut endpoint = self.state;
        endpoint.advance(&self.step, frames);
        let gains = pan_object(panner, endpoint)?;
        let direction = render_direction(panner, &gains, endpoint);
        if let Some(continuous) = self.continuous.as_mut() {
            continuous.set_direction(direction, frames);
        }
        self.ensure_early_distance(direction, endpoint.distance_progress, panner);
        if let Some(early_distance) = self.early_distance.as_mut() {
            early_distance.set_target(direction, endpoint.distance_progress, frames, panner);
        }
        self.gains.set_target(gains, frames);
        Ok(())
    }

    fn render(&mut self, sample: f32, writer: &mut BinauralWriter) -> Result<(), AppError> {
        let direct_gain = self.state.distance_direct;
        if let Some(continuous) = self.continuous.as_mut() {
            let continuous_weight = self.state.continuous_weight.clamp(0.0, 1.0);
            let baseline_weight = (1.0 - continuous_weight).sqrt();
            let continuous_weight = continuous_weight.sqrt();
            if baseline_weight > f32::EPSILON {
                for (bus, gain) in self.gains.current.iter().copied().enumerate() {
                    if gain != 0.0 {
                        writer.add(bus, sample * gain * baseline_weight * direct_gain)?;
                    }
                }
            }
            let [left, right] =
                continuous.process(sample * self.gains.level * continuous_weight * direct_gain);
            writer.add_direct(left, right);
        } else {
            for (bus, gain) in self.gains.current.iter().copied().enumerate() {
                if gain != 0.0 {
                    writer.add(bus, sample * gain * direct_gain)?;
                }
            }
        }
        let drop_early_distance = if let Some(early_distance) = self.early_distance.as_mut() {
            early_distance.process(
                sample * self.gains.level * self.state.distance_early,
                writer,
            )?;
            self.state.distance_early == 0.0
                && self.target.distance_early == 0.0
                && early_distance.tail_frames() == 0
        } else {
            false
        };
        if drop_early_distance {
            self.early_distance = None;
        }
        if self.state.reflection_ratio != 0.0 {
            for (bus, gain) in self.gains.current.iter().copied().enumerate() {
                if gain != 0.0 {
                    writer
                        .add_early_reflection(bus, sample * gain * self.state.reflection_ratio)?;
                }
            }
        }
        Ok(())
    }

    fn ensure_early_distance(
        &mut self,
        direction: [f32; 3],
        progress: f32,
        panner: &SpatialPanner,
    ) {
        if self.early_distance.is_none()
            && (self.state.distance_early != 0.0 || self.target.distance_early != 0.0)
        {
            self.early_distance = Some(EarlyDistanceField::new(
                self.sample_rate,
                direction,
                progress,
                panner,
            ));
        }
    }

    fn render_tail(&mut self, writer: &mut BinauralWriter) -> Result<(), AppError> {
        if let Some(continuous) = self.continuous.as_mut() {
            let [left, right] = continuous.process(0.0);
            writer.add_direct(left, right);
        }
        if let Some(early_distance) = self.early_distance.as_mut() {
            early_distance.process(0.0, writer)?;
        }
        Ok(())
    }

    #[must_use]
    fn tail_frames(&self) -> usize {
        self.continuous
            .as_ref()
            .map_or(0, ContinuousBinaural::tail_frames)
            .max(
                self.early_distance
                    .as_ref()
                    .map_or(0, EarlyDistanceField::tail_frames),
            )
    }
}

fn pan_object(panner: &SpatialPanner, object: RenderableObject) -> Result<Vec<f32>, AppError> {
    if object.gain == 0.0 {
        return Ok(vec![0.0; panner.bus_count()]);
    }
    if object.bed_speaker == Some(Speaker::Lfe) {
        return Ok(panner.lfe_gains(object.gain));
    }
    panner.gains(
        object.position,
        object.size,
        object.gain,
        object.snap,
        object.zone,
        object.elevation,
        object.divergence,
        object.trim,
        object.bed_speaker.is_some(),
    )
}

fn render_direction(panner: &SpatialPanner, gains: &[f32], object: RenderableObject) -> [f32; 3] {
    let mut authored = normalized_direction(object.position);
    if !object.elevation {
        authored[2] = 0.0;
        authored = normalized_direction(authored);
    }
    if !object.snap && object.zone == ObjectZone::All {
        return authored;
    }

    let routed = panner.resultant_direction(gains, authored);
    if object.snap || !object.elevation {
        return routed;
    }
    let horizontal = routed[0].hypot(routed[1]);
    if horizontal <= f32::EPSILON {
        return authored;
    }
    let authored_horizontal = (1.0 - authored[2] * authored[2]).max(0.0).sqrt();
    [
        routed[0] / horizontal * authored_horizontal,
        routed[1] / horizontal * authored_horizontal,
        authored[2],
    ]
}

fn vector_level(gains: &[f32]) -> f32 {
    gains.iter().map(|gain| gain * gain).sum::<f32>().sqrt()
}

fn distance_reflection_ratio(distance_factor: Option<f32>) -> f32 {
    const ROOM_BOUNDARY_DB: f32 = -20.0;
    const FAR_FIELD_DB: f32 = -10.0;
    const MAXIMUM_FINITE_DISTANCE: f32 = 50.0;

    let Some(distance_factor) = distance_factor else {
        return UNSPECIFIED_REFLECTION_RATIO;
    };
    let reflection_db = {
        let distance = if distance_factor.is_infinite() {
            1.0
        } else {
            (distance_factor.max(1.0).ln() / MAXIMUM_FINITE_DISTANCE.ln()).clamp(0.0, 1.0)
        };
        (FAR_FIELD_DB - ROOM_BOUNDARY_DB).mul_add(distance, ROOM_BOUNDARY_DB)
    };
    10_f32.powf(reflection_db / 20.0)
}

fn distance_mix(distance_factor: Option<f32>) -> (f32, f32) {
    const ROOM_BOUNDARY_DB: f32 = -22.0;
    const FAR_FIELD_DB: f32 = -14.0;
    const UNSPECIFIED_DB: f32 = -24.0;
    let early_db = distance_factor.map_or(UNSPECIFIED_DB, |factor| {
        let progress = distance_progress(Some(factor));
        (FAR_FIELD_DB - ROOM_BOUNDARY_DB).mul_add(progress, ROOM_BOUNDARY_DB)
    });
    let early = 10_f32.powf(early_db / 20.0);
    let normalization = 1.0 / early.hypot(1.0);
    (normalization, early * normalization)
}

fn distance_progress(distance_factor: Option<f32>) -> f32 {
    const MAXIMUM_FINITE_DISTANCE: f32 = 50.1;
    distance_factor.map_or(0.25, |factor| {
        if factor.is_infinite() {
            1.0
        } else {
            (factor.max(1.0).ln() / MAXIMUM_FINITE_DISTANCE.ln()).clamp(0.0, 1.0)
        }
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hound::WavReader;

    use super::{
        EarlyDistanceField, MovingGains, MovingSpatialObject, ObjectPcmFrame, ObjectRenderOptions,
        ObjectRenderer, RenderableObject, SparseMovingGains, distance_mix,
        distance_reflection_ratio, early_field_targets, pan_object, render_direction, vector_level,
    };
    use crate::{
        binaural::BinauralWriter,
        cli::{DistanceRendererMode, ObjectRendererMode},
        hrir::{HrirChannel, HrirSet, Speaker},
        isf::IsfConfig,
        object::{ObjectState, ObjectTrim, SpatialUpdate},
    };

    #[test]
    fn object_ramp_reaches_target_exactly() {
        let mut object = MovingGains::new(vec![0.0, 1.0]);
        object.set_target(vec![1.0, 0.0], 4);
        object.advance();
        object.advance();
        assert!((vector_level(&object.current) - 1.0).abs() < f32::EPSILON);
        for _ in 0..2 {
            object.advance();
        }
        assert_eq!(object.current, [1.0, 0.0]);
    }

    #[test]
    fn continuous_geometry_preserves_authored_elevation_without_height_routes() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("ground-only.wav");
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![
                HrirChannel {
                    speaker: Speaker::FrontLeft,
                    left: vec![1.0],
                    right: vec![0.0],
                },
                HrirChannel {
                    speaker: Speaker::FrontRight,
                    left: vec![0.0],
                    right: vec![1.0],
                },
            ],
            directional: Vec::new(),
        };
        let writer = BinauralWriter::new_raw(
            &output,
            &hrir,
            None,
            0.0,
            [Speaker::FrontLeft, Speaker::FrontRight],
        )
        .unwrap();
        let panner = crate::spatial::SpatialPanner::new(&writer).unwrap();
        let options = ObjectRenderOptions {
            surround_swap: false,
            mute_bed: false,
            mute_ground: false,
            speaker_virtualizer: false,
            object_renderer: ObjectRendererMode::Continuous,
            distance_renderer: DistanceRendererMode::ImageSource,
        };
        let state = ObjectState {
            source_channel: 0,
            active: true,
            bed_speaker: None,
            position: [0.3, 0.4, 0.866_025_4],
            distance_factor: None,
            gain: 1.0,
            size: [0.0; 3],
            snap: false,
            zone: crate::object::ObjectZone::All,
            elevation: true,
            divergence: 0.0,
            trim: ObjectTrim::default(),
        };
        let object = RenderableObject::new(&state, options);
        let gains = pan_object(&panner, object).unwrap();
        let routed = panner.resultant_direction(&gains, state.position);
        let continuous = render_direction(&panner, &gains, object);

        assert!(routed[2].abs() < f32::EPSILON);
        for (actual, expected) in continuous.into_iter().zip(state.position) {
            assert!((actual - expected).abs() < 1e-6);
        }

        let flattened_state = ObjectState {
            elevation: false,
            ..state
        };
        let flattened = RenderableObject::new(&flattened_state, options);
        let flattened_gains = pan_object(&panner, flattened).unwrap();
        assert!(render_direction(&panner, &flattened_gains, flattened)[2].abs() < f32::EPSILON);
    }

    #[test]
    fn sparse_reflection_ramp_matches_dense_constant_power_interpolation() {
        let initial = vec![0.0, 0.8, 0.6, 0.0, 0.0];
        let target = vec![0.0, 0.0, 0.3, 0.4, 0.5];
        let mut dense = MovingGains::new(initial.clone());
        let mut sparse = SparseMovingGains::new(initial);
        dense.set_target(target.clone(), 17);
        sparse.set_target(target, 17);
        for _ in 0..17 {
            dense.advance();
            sparse.advance();
            let mut expanded = vec![0.0; dense.current.len()];
            for (index, gain) in sparse.iter() {
                expanded[index] = gain;
            }
            for (dense, sparse) in dense.current.iter().zip(expanded) {
                assert!((*dense - sparse).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn authored_distance_only_changes_the_early_reflection_send() {
        let unspecified = distance_reflection_ratio(None);
        let boundary = distance_reflection_ratio(Some(1.0));
        let middle = distance_reflection_ratio(Some(10.0));
        let far = distance_reflection_ratio(Some(f32::INFINITY));

        assert!((20.0 * unspecified.log10() + 24.0).abs() < 1e-5);
        assert!((20.0 * boundary.log10() + 20.0).abs() < 1e-5);
        assert!(unspecified < boundary);
        assert!(boundary < middle);
        assert!(middle < far);
        assert!((20.0 * far.log10() + 10.0).abs() < 1e-5);
    }

    #[test]
    fn continuous_distance_is_constant_power_and_spans_one_clear_distance_step() {
        let unspecified = distance_mix(None);
        let boundary = distance_mix(Some(1.0));
        let far = distance_mix(Some(50.1));
        for (direct, early) in [unspecified, boundary, far] {
            assert!((direct.hypot(early) - 1.0).abs() < 1e-6);
        }
        assert!((20.0 * (boundary.1 / boundary.0).log10() + 22.0).abs() < 1e-5);
        assert!((20.0 * (far.1 / far.0).log10() + 14.0).abs() < 1e-5);
        assert!(far.0 > 0.98);
    }

    #[test]
    fn image_reflections_are_bounded_and_energy_normalized() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("early-field.wav");
        let speakers = [
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
            Speaker::RearLeft,
            Speaker::RearRight,
        ];
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: speakers
                .into_iter()
                .map(|speaker| HrirChannel {
                    speaker,
                    left: vec![1.0],
                    right: vec![1.0],
                })
                .collect(),
            directional: Vec::new(),
        };
        let writer = BinauralWriter::new_raw(&output, &hrir, None, 0.0, speakers).unwrap();
        let panner = crate::spatial::SpatialPanner::new(&writer).unwrap();
        let near = early_field_targets(48_000, [0.3, 1.0, 0.2], 0.0, &panner);
        let far = early_field_targets(48_000, [0.3, 1.0, 0.2], 1.0, &panner);
        let routed_energy = |taps: &[super::EarlyTapTarget]| {
            taps.iter()
                .flat_map(|tap| &tap.gains)
                .map(|gain| gain * gain)
                .sum::<f32>()
        };
        assert!((routed_energy(&near) - 1.0).abs() < 1e-5);
        assert!((routed_energy(&far) - 1.0).abs() < 1e-5);
        assert!(near.iter().all(|tap| {
            (1.0..48_000.0 * super::MAXIMUM_EARLY_FIELD_SECONDS).contains(&tap.delay)
                && (0.0..1.0).contains(&tap.low_pass_coefficient)
        }));
        assert!(
            near.iter()
                .zip(&far)
                .any(|(near, far)| (near.delay - far.delay).abs() > 1.0)
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn image_reflection_tail_tracks_sample_rate_and_reaches_the_24_bit_floor() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("early-tail.wav");
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![HrirChannel {
                speaker: Speaker::FrontCenter,
                left: vec![1.0],
                right: vec![1.0],
            }],
            directional: Vec::new(),
        };
        let writer =
            BinauralWriter::new_raw(&output, &hrir, None, 0.0, [Speaker::FrontCenter]).unwrap();
        let panner = crate::spatial::SpatialPanner::new(&writer).unwrap();

        for (sample_rate, expected_tail) in [(48_000, 64), (96_000, 96), (192_000, 192)] {
            let field = EarlyDistanceField::new(sample_rate, [0.0, 1.0, 0.0], 0.5, &panner);
            assert_eq!(field.filter_tail_frames, expected_tail);
            let slowest_residual =
                (-std::f32::consts::TAU * 4_500.0 * field.filter_tail_frames as f32
                    / sample_rate as f32)
                    .exp();
            assert!(
                slowest_residual < 2.0_f32.powi(-24),
                "{sample_rate} Hz reflection residual {slowest_residual} exceeded the 24-bit floor"
            );
        }
    }

    #[test]
    fn continuous_front_centre_render_remains_symmetric_through_its_early_field() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("continuous-centre.wav");
        let speakers = [
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
            Speaker::RearLeft,
            Speaker::RearRight,
        ];
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: speakers
                .into_iter()
                .map(|speaker| HrirChannel {
                    speaker,
                    left: vec![1.0],
                    right: vec![1.0],
                })
                .collect(),
            directional: Vec::new(),
        };
        let writer = BinauralWriter::new(&output, &hrir, None, 0.0, speakers).unwrap();
        let mut renderer = ObjectRenderer::new(
            writer,
            &hrir,
            ObjectRenderOptions {
                surround_swap: false,
                mute_bed: false,
                mute_ground: false,
                speaker_virtualizer: false,
                object_renderer: ObjectRendererMode::Continuous,
                distance_renderer: DistanceRendererMode::ImageSource,
            },
        )
        .unwrap();
        let mut samples = vec![0.0; 256];
        samples[0] = 0.05;
        renderer
            .push(ObjectPcmFrame {
                sample_rate: 48_000,
                sample_count: samples.len(),
                channel_count: 1,
                samples,
                channel_speakers: vec![None],
                isf: None,
                spatial_updates: vec![SpatialUpdate {
                    sample_offset: 0,
                    ramp_samples: 0,
                    bed_speakers: Vec::new(),
                    isf: Vec::new(),
                    objects: vec![ObjectState {
                        source_channel: 0,
                        active: true,
                        bed_speaker: None,
                        position: [0.0, 1.0, 0.0],
                        distance_factor: Some(1.0),
                        gain: 1.0,
                        size: [0.0; 3],
                        snap: false,
                        zone: crate::object::ObjectZone::All,
                        elevation: true,
                        divergence: 0.0,
                        trim: ObjectTrim::default(),
                    }],
                }],
            })
            .unwrap();
        renderer.finish().unwrap();

        let samples = WavReader::open(output)
            .unwrap()
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let maximum_difference = samples
            .chunks_exact(2)
            .map(|frame| frame[0].abs_diff(frame[1]))
            .max()
            .unwrap();
        assert!(
            maximum_difference <= 4,
            "front-centre L/R difference reached {maximum_difference} quantization steps"
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn continuous_renderer_is_frame_boundary_independent_and_peak_safe() {
        const PCM_I24_MINUS_ONE_DB_CEILING_WITH_QUANTIZATION_TOLERANCE: u32 = 7_476_356;

        let directory = tempfile::tempdir().unwrap();
        let samples = (0..2_500)
            .map(|frame| {
                let sine = (std::f32::consts::TAU * 317.0 * frame as f32 / 48_000.0).sin();
                let transient = if frame % 257 == 0 { 0.9 } else { 0.0 };
                0.82_f32.mul_add(sine, transient)
            })
            .collect::<Vec<_>>();
        let contiguous =
            render_continuous_regression(&directory.path().join("contiguous.wav"), &samples, None);
        let fragmented = render_continuous_regression(
            &directory.path().join("fragmented.wav"),
            &samples,
            Some(&[17, 1_007, 1, 475, 89]),
        );

        assert_eq!(contiguous.0, fragmented.0);
        assert_eq!(contiguous.1.frames, fragmented.1.frames);
        assert!(
            contiguous.1.peak_before_limiting > 1.0,
            "stress render never reached the limiter"
        );
        let sample_peak = contiguous
            .0
            .iter()
            .map(|sample| sample.unsigned_abs())
            .max()
            .unwrap();
        assert!(
            sample_peak <= PCM_I24_MINUS_ONE_DB_CEILING_WITH_QUANTIZATION_TOLERANCE,
            "limited sample peak {sample_peak} exceeded \
             {PCM_I24_MINUS_ONE_DB_CEILING_WITH_QUANTIZATION_TOLERANCE}"
        );
    }

    #[test]
    fn reference_beds_share_the_quiet_early_field_but_lfe_stays_dry() {
        let options = ObjectRenderOptions {
            surround_swap: false,
            mute_bed: false,
            mute_ground: false,
            speaker_virtualizer: false,
            object_renderer: ObjectRendererMode::Baseline,
            distance_renderer: DistanceRendererMode::Baseline,
        };
        let mut state = ObjectState {
            source_channel: 0,
            active: true,
            bed_speaker: Some(Speaker::FrontCenter),
            position: Speaker::FrontCenter.position(),
            distance_factor: None,
            gain: 1.0,
            size: [0.0; 3],
            snap: false,
            zone: crate::object::ObjectZone::All,
            elevation: true,
            divergence: 0.0,
            trim: ObjectTrim::default(),
        };
        let bed = RenderableObject::new(&state, options);
        assert!((bed.reflection_ratio - distance_reflection_ratio(None)).abs() < f32::EPSILON);

        state.bed_speaker = Some(Speaker::Lfe);
        state.position = Speaker::Lfe.position();
        let lfe = RenderableObject::new(&state, options);
        assert!(lfe.reflection_ratio.abs() < f32::EPSILON);
    }

    #[test]
    fn warp_mode_doubles_normative_room_depth_before_rendering() {
        let state = ObjectState {
            source_channel: 0,
            active: true,
            bed_speaker: None,
            // Renderer y=0.5 corresponds to normative room Y=0.25.
            position: [0.0, 0.5, 0.0],
            distance_factor: None,
            gain: 1.0,
            size: [0.0; 3],
            snap: false,
            zone: crate::object::ObjectZone::All,
            elevation: true,
            divergence: 0.0,
            trim: ObjectTrim {
                warp_y: true,
                ..ObjectTrim::default()
            },
        };
        let warped = RenderableObject::new(
            &state,
            ObjectRenderOptions {
                surround_swap: false,
                mute_bed: false,
                mute_ground: false,
                speaker_virtualizer: false,
                object_renderer: ObjectRendererMode::Baseline,
                distance_renderer: DistanceRendererMode::Baseline,
            },
        );

        // Doubling normative Y=0.25 gives Y=0.5, the listener centre.
        assert!(warped.position[1].abs() < f32::EPSILON);
    }

    #[test]
    fn spatial_ramp_passes_through_intermediate_hrir_positions() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("trajectory.wav");
        let speakers = [
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
        ];
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: speakers
                .into_iter()
                .map(|speaker| HrirChannel {
                    speaker,
                    left: vec![1.0],
                    right: vec![1.0],
                })
                .collect(),
            directional: Vec::new(),
        };
        let writer = BinauralWriter::new_raw(&output, &hrir, None, 0.0, speakers).unwrap();
        let panner = crate::spatial::SpatialPanner::new(&writer).unwrap();
        let mut object = MovingSpatialObject::new(
            RenderableObject {
                bed_speaker: None,
                position: [-1.0, 1.0, 0.0],
                size: [0.0; 3],
                gain: 1.0,
                snap: false,
                zone: crate::object::ObjectZone::All,
                elevation: true,
                divergence: 0.0,
                reflection_ratio: 0.0,
                continuous_weight: 0.0,
                distance_direct: 1.0,
                distance_early: 0.0,
                distance_progress: 0.0,
                trim: crate::object::ObjectTrim::default(),
            },
            &panner,
            None,
            DistanceRendererMode::Baseline,
            48_000,
        )
        .unwrap();
        object
            .set_target(
                RenderableObject {
                    position: [1.0, 1.0, 0.0],
                    ..object.state
                },
                256,
                &panner,
            )
            .unwrap();
        for _ in 0..128 {
            object.advance(&panner).unwrap();
        }

        assert!(
            object.gains.current[1] > 0.999,
            "centre gain was {}",
            object.gains.current[1]
        );
        assert!(object.gains.current[0].abs() < 1e-5);
        assert!(object.gains.current[2].abs() < 1e-5);
    }

    #[test]
    fn moving_object_is_routed_to_the_matching_hrir_direction() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("object.wav");
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![
                HrirChannel {
                    speaker: Speaker::FrontLeft,
                    left: vec![1.0],
                    right: vec![0.0],
                },
                HrirChannel {
                    speaker: Speaker::FrontRight,
                    left: vec![0.0],
                    right: vec![1.0],
                },
            ],
            directional: Vec::new(),
        };
        let writer = BinauralWriter::new_raw(
            &output,
            &hrir,
            None,
            0.0,
            [Speaker::FrontLeft, Speaker::FrontRight],
        )
        .unwrap();
        let mut renderer = ObjectRenderer::new(
            writer,
            &hrir,
            ObjectRenderOptions {
                surround_swap: false,
                mute_bed: false,
                mute_ground: false,
                speaker_virtualizer: false,
                object_renderer: ObjectRendererMode::Baseline,
                distance_renderer: DistanceRendererMode::Baseline,
            },
        )
        .unwrap();
        renderer
            .push(ObjectPcmFrame {
                sample_rate: 48_000,
                sample_count: 4,
                channel_count: 1,
                samples: vec![0.25; 4],
                channel_speakers: vec![None],
                isf: None,
                spatial_updates: vec![SpatialUpdate {
                    sample_offset: 0,
                    ramp_samples: 0,
                    bed_speakers: Vec::new(),
                    isf: Vec::new(),
                    objects: vec![ObjectState {
                        source_channel: 0,
                        active: true,
                        bed_speaker: None,
                        position: [-1.0, 1.0, 0.0],
                        distance_factor: None,
                        gain: 1.0,
                        size: [0.0; 3],
                        snap: false,
                        zone: crate::object::ObjectZone::All,
                        elevation: true,
                        divergence: 0.0,
                        trim: crate::object::ObjectTrim::default(),
                    }],
                }],
            })
            .unwrap();
        renderer.finish().unwrap();

        let samples = WavReader::open(output)
            .unwrap()
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let left_peak = samples.iter().step_by(2).copied().max().unwrap();
        let right_peak = samples.iter().skip(1).step_by(2).copied().max().unwrap();
        assert!(left_peak > right_peak.saturating_mul(10));
    }

    #[test]
    fn far_distance_strengthens_only_the_delayed_field() {
        let directory = tempfile::tempdir().unwrap();
        let near = render_distance_impulse(
            &directory.path().join("near.wav"),
            1.1,
            DistanceRendererMode::Baseline,
        );
        let far = render_distance_impulse(
            &directory.path().join("far.wav"),
            50.1,
            DistanceRendererMode::Baseline,
        );

        assert_eq!(near.len(), far.len());
        assert!(
            near[0].abs_diff(far[0]) <= 2,
            "direct sample changed from {} to {}",
            near[0],
            far[0]
        );
        let reflected_energy = |samples: &[i32]| {
            samples
                .iter()
                .step_by(2)
                .skip(1)
                .map(|sample| f64::from(*sample).powi(2))
                .sum::<f64>()
        };
        let near_energy = reflected_energy(&near);
        let far_energy = reflected_energy(&far);
        assert!(
            far_energy > near_energy * 5.0,
            "far/near reflected energy ratio was {}",
            far_energy / near_energy
        );
    }

    #[test]
    fn image_source_distance_preserves_direct_sound_and_strengthens_its_early_field() {
        let directory = tempfile::tempdir().unwrap();
        let near = render_distance_impulse(
            &directory.path().join("image-near.wav"),
            1.1,
            DistanceRendererMode::ImageSource,
        );
        let far = render_distance_impulse(
            &directory.path().join("image-far.wav"),
            50.1,
            DistanceRendererMode::ImageSource,
        );

        assert_eq!(near.len(), far.len());
        let direct_ratio = f64::from(far[0]) / f64::from(near[0]);
        assert!(
            (0.97..=1.0).contains(&direct_ratio),
            "far/near direct ratio was {direct_ratio}"
        );
        let delayed_energy = |samples: &[i32]| {
            samples
                .chunks_exact(2)
                .skip(1)
                .map(|frame| 0.5 * (f64::from(frame[0]).powi(2) + f64::from(frame[1]).powi(2)))
                .sum::<f64>()
        };
        let near_energy = delayed_energy(&near);
        let far_energy = delayed_energy(&far);
        assert!(
            far_energy > near_energy * 3.0,
            "far/near image-field energy ratio was {}",
            far_energy / near_energy
        );
    }

    #[test]
    fn bed_assignment_releases_channel_from_stale_object_routing() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("role-transition.wav");
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![
                HrirChannel {
                    speaker: Speaker::FrontLeft,
                    left: vec![1.0],
                    right: vec![0.0],
                },
                HrirChannel {
                    speaker: Speaker::FrontRight,
                    left: vec![0.0],
                    right: vec![1.0],
                },
            ],
            directional: Vec::new(),
        };
        let writer = BinauralWriter::new_raw(
            &output,
            &hrir,
            None,
            0.0,
            [Speaker::FrontLeft, Speaker::FrontRight],
        )
        .unwrap();
        let mut renderer = ObjectRenderer::new(
            writer,
            &hrir,
            ObjectRenderOptions {
                surround_swap: false,
                mute_bed: false,
                mute_ground: false,
                speaker_virtualizer: false,
                object_renderer: ObjectRendererMode::Baseline,
                distance_renderer: DistanceRendererMode::Baseline,
            },
        )
        .unwrap();
        renderer
            .apply_update(SpatialUpdate {
                sample_offset: 0,
                ramp_samples: 0,
                bed_speakers: Vec::new(),
                isf: Vec::new(),
                objects: vec![ObjectState {
                    source_channel: 0,
                    active: true,
                    bed_speaker: None,
                    position: [1.0, 1.0, 0.0],
                    distance_factor: None,
                    gain: 1.0,
                    size: [0.0; 3],
                    snap: false,
                    zone: crate::object::ObjectZone::All,
                    elevation: true,
                    divergence: 0.0,
                    trim: crate::object::ObjectTrim::default(),
                }],
            })
            .unwrap();
        assert!(renderer.objects.contains_key(&0));
        assert!(renderer.object_channels.contains(&0));

        renderer
            .apply_update(SpatialUpdate {
                sample_offset: 0,
                ramp_samples: 0,
                bed_speakers: vec![Speaker::FrontLeft],
                isf: Vec::new(),
                objects: Vec::new(),
            })
            .unwrap();

        assert!(!renderer.objects.contains_key(&0));
        assert!(!renderer.object_channels.contains(&0));
    }

    #[test]
    fn isf_keeps_normative_signed_coefficients_before_binaural_convolution() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("isf.wav");
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![
                HrirChannel {
                    speaker: Speaker::FrontLeft,
                    left: vec![1.0],
                    right: vec![0.0],
                },
                HrirChannel {
                    speaker: Speaker::FrontRight,
                    left: vec![0.0],
                    right: vec![1.0],
                },
            ],
            directional: Vec::new(),
        };
        let writer = BinauralWriter::new_raw(
            &output,
            &hrir,
            None,
            0.0,
            [Speaker::FrontLeft, Speaker::FrontRight],
        )
        .unwrap();
        let mut renderer = ObjectRenderer::new(
            writer,
            &hrir,
            ObjectRenderOptions {
                surround_swap: false,
                mute_bed: false,
                mute_ground: false,
                speaker_virtualizer: false,
                object_renderer: ObjectRendererMode::Baseline,
                distance_renderer: DistanceRendererMode::Baseline,
            },
        )
        .unwrap();
        renderer
            .push(ObjectPcmFrame {
                sample_rate: 48_000,
                sample_count: 1,
                channel_count: 4,
                samples: vec![0.0, 0.25, 0.0, 0.0],
                channel_speakers: vec![None; 4],
                isf: Some(IsfConfig::new(0, 4).unwrap()),
                spatial_updates: Vec::new(),
            })
            .unwrap();
        renderer.finish().unwrap();

        let samples = WavReader::open(output)
            .unwrap()
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(samples[0] > 2_560_000);
        assert!(samples[1] < -256_000);
    }

    fn render_distance_impulse(
        output: &Path,
        distance_factor: f32,
        distance_renderer: DistanceRendererMode,
    ) -> Vec<i32> {
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: vec![HrirChannel {
                speaker: Speaker::FrontCenter,
                left: vec![1.0],
                right: vec![1.0],
            }],
            directional: Vec::new(),
        };
        let writer =
            BinauralWriter::new_raw(output, &hrir, None, 0.0, [Speaker::FrontCenter]).unwrap();
        let mut renderer = ObjectRenderer::new(
            writer,
            &hrir,
            ObjectRenderOptions {
                surround_swap: false,
                mute_bed: false,
                mute_ground: false,
                speaker_virtualizer: false,
                object_renderer: ObjectRendererMode::Baseline,
                distance_renderer,
            },
        )
        .unwrap();
        let mut samples = vec![0.0; 2_048];
        samples[0] = 0.1;
        renderer
            .push(ObjectPcmFrame {
                sample_rate: 48_000,
                sample_count: samples.len(),
                channel_count: 1,
                samples,
                channel_speakers: vec![None],
                isf: None,
                spatial_updates: vec![SpatialUpdate {
                    sample_offset: 0,
                    ramp_samples: 0,
                    bed_speakers: Vec::new(),
                    isf: Vec::new(),
                    objects: vec![ObjectState {
                        source_channel: 0,
                        active: true,
                        bed_speaker: None,
                        position: [0.0, distance_factor, 0.0],
                        distance_factor: Some(distance_factor),
                        gain: 1.0,
                        size: [0.0; 3],
                        snap: false,
                        zone: crate::object::ObjectZone::All,
                        elevation: true,
                        divergence: 0.0,
                        trim: ObjectTrim::default(),
                    }],
                }],
            })
            .unwrap();
        renderer.finish().unwrap();
        WavReader::open(output)
            .unwrap()
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    // Keeping the complete fixture in one place makes the contiguous and fragmented
    // renders provably identical apart from their input frame boundaries.
    #[allow(clippy::too_many_lines)]
    fn render_continuous_regression(
        output: &Path,
        samples: &[f32],
        chunk_pattern: Option<&[usize]>,
    ) -> (Vec<i32>, crate::render::RenderResult) {
        let speakers = [
            Speaker::FrontLeft,
            Speaker::FrontCenter,
            Speaker::FrontRight,
            Speaker::RearLeft,
            Speaker::RearRight,
        ];
        let hrir = HrirSet {
            sample_rate: 48_000,
            channels: speakers
                .into_iter()
                .map(|speaker| HrirChannel {
                    speaker,
                    left: vec![1.0],
                    right: vec![1.0],
                })
                .collect(),
            directional: Vec::new(),
        };
        let writer = BinauralWriter::new(output, &hrir, None, 0.0, speakers).unwrap();
        let mut renderer = ObjectRenderer::new(
            writer,
            &hrir,
            ObjectRenderOptions {
                surround_swap: false,
                mute_bed: false,
                mute_ground: false,
                speaker_virtualizer: false,
                object_renderer: ObjectRendererMode::Continuous,
                distance_renderer: DistanceRendererMode::ImageSource,
            },
        )
        .unwrap();
        let updates = || {
            vec![
                SpatialUpdate {
                    sample_offset: 0,
                    ramp_samples: 0,
                    bed_speakers: Vec::new(),
                    isf: Vec::new(),
                    objects: vec![ObjectState {
                        source_channel: 0,
                        active: true,
                        bed_speaker: None,
                        position: [-1.0, 1.0, 0.2],
                        distance_factor: Some(1.1),
                        gain: 1.5,
                        size: [0.0; 3],
                        snap: false,
                        zone: crate::object::ObjectZone::All,
                        elevation: true,
                        divergence: 0.0,
                        trim: ObjectTrim::default(),
                    }],
                },
                SpatialUpdate {
                    sample_offset: 777,
                    ramp_samples: 913,
                    bed_speakers: Vec::new(),
                    isf: Vec::new(),
                    objects: vec![ObjectState {
                        source_channel: 0,
                        active: true,
                        bed_speaker: None,
                        position: [0.8, -1.0, 0.7],
                        distance_factor: Some(50.1),
                        gain: 1.5,
                        size: [0.0; 3],
                        snap: false,
                        zone: crate::object::ObjectZone::All,
                        elevation: true,
                        divergence: 0.0,
                        trim: ObjectTrim::default(),
                    }],
                },
            ]
        };
        if let Some(pattern) = chunk_pattern {
            let mut position = 0;
            let mut chunk = 0;
            while position < samples.len() {
                let frames = pattern[chunk % pattern.len()].min(samples.len() - position);
                renderer
                    .push(ObjectPcmFrame {
                        sample_rate: 48_000,
                        sample_count: frames,
                        channel_count: 1,
                        samples: samples[position..position + frames].to_vec(),
                        channel_speakers: vec![None],
                        isf: None,
                        spatial_updates: if position == 0 { updates() } else { Vec::new() },
                    })
                    .unwrap();
                position += frames;
                chunk += 1;
            }
        } else {
            renderer
                .push(ObjectPcmFrame {
                    sample_rate: 48_000,
                    sample_count: samples.len(),
                    channel_count: 1,
                    samples: samples.to_vec(),
                    channel_speakers: vec![None],
                    isf: None,
                    spatial_updates: updates(),
                })
                .unwrap();
        }
        let result = renderer.finish().unwrap();
        let output_samples = WavReader::open(output)
            .unwrap()
            .samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        (output_samples, result)
    }
}
