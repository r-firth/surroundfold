use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    binaural::BinauralWriter,
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
    isf: Option<IsfRenderer>,
    isf_sources: HashMap<usize, MovingGains>,
    objects: HashMap<usize, MovingSpatialObject>,
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
        Ok(Self {
            writer,
            hrir,
            options,
            panner,
            isf: None,
            isf_sources: HashMap::new(),
            objects: HashMap::new(),
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
            for (channel, sample) in source.iter().copied().enumerate() {
                if let Some(object) = self.objects.get(&channel) {
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
            self.objects.remove(&source_channel);
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
                self.objects.remove(&source_channel);
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
                        MovingSpatialObject::new(target, &self.panner)?,
                    );
                }
            }
        }
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<RenderResult, AppError> {
        self.writer.finish()
    }
}

struct ScheduledUpdate {
    at: u64,
    update: SpatialUpdate,
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
        Self {
            bed_speaker: state.bed_speaker,
            position,
            size: state.size,
            gain: if audible { state.gain } else { 0.0 },
            snap: state.snap,
            zone: state.zone,
            elevation: state.elevation,
            divergence: state.divergence,
            reflection_ratio: if state.bed_speaker == Some(Speaker::Lfe) {
                0.0
            } else {
                distance_reflection_ratio(state.distance_factor)
            },
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
    }
}

#[derive(Clone, Copy)]
struct ObjectStep {
    position: [f32; 3],
    size: [f32; 3],
    gain: f32,
    divergence: f32,
    reflection_ratio: f32,
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
        }
    }
}

struct MovingSpatialObject {
    state: RenderableObject,
    target: RenderableObject,
    step: ObjectStep,
    gains: MovingGains,
    remaining: usize,
}

impl MovingSpatialObject {
    fn new(state: RenderableObject, panner: &SpatialPanner) -> Result<Self, AppError> {
        let gains = MovingGains::new(pan_object(panner, state)?);
        Ok(Self {
            state,
            target: state,
            step: ObjectStep {
                position: [0.0; 3],
                size: [0.0; 3],
                gain: 0.0,
                divergence: 0.0,
                reflection_ratio: 0.0,
            },
            gains,
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
            self.gains.set_target(pan_object(panner, target)?, 0);
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
            self.gains.set_target(pan_object(panner, self.target)?, 0);
        } else if self.gains.remaining == 0 {
            self.prepare_segment(panner)?;
        }
        Ok(())
    }

    fn prepare_segment(&mut self, panner: &SpatialPanner) -> Result<(), AppError> {
        let frames = self.remaining.min(SPATIAL_CONTROL_INTERVAL);
        let mut endpoint = self.state;
        endpoint.advance(&self.step, frames);
        self.gains.set_target(pan_object(panner, endpoint)?, frames);
        Ok(())
    }

    fn render(&self, sample: f32, writer: &mut BinauralWriter) -> Result<(), AppError> {
        for (bus, gain) in self.gains.current.iter().copied().enumerate() {
            if gain == 0.0 {
                continue;
            }
            let routed = sample * gain;
            writer.add(bus, routed)?;
            if self.state.reflection_ratio != 0.0 {
                writer.add_early_reflection(bus, routed * self.state.reflection_ratio)?;
            }
        }
        Ok(())
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

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hound::WavReader;

    use super::{
        MovingGains, MovingSpatialObject, ObjectPcmFrame, ObjectRenderOptions, ObjectRenderer,
        RenderableObject, distance_reflection_ratio, vector_level,
    };
    use crate::{
        binaural::BinauralWriter,
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
    fn reference_beds_share_the_quiet_early_field_but_lfe_stays_dry() {
        let options = ObjectRenderOptions {
            surround_swap: false,
            mute_bed: false,
            mute_ground: false,
            speaker_virtualizer: false,
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
                trim: crate::object::ObjectTrim::default(),
            },
            &panner,
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
        let near = render_distance_impulse(&directory.path().join("near.wav"), 1.1);
        let far = render_distance_impulse(&directory.path().join("far.wav"), 50.1);

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

    fn render_distance_impulse(output: &Path, distance_factor: f32) -> Vec<i32> {
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
}
