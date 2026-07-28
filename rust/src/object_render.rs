use std::collections::{HashMap, HashSet, VecDeque};

use crate::{
    binaural::BinauralWriter,
    error::AppError,
    hrir::{HrirSet, Speaker},
    isf::{IsfConfig, IsfRenderer},
    object::SpatialUpdate,
    render::RenderResult,
    spatial::{SpatialPanner, direct_stereo_gains},
};

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
    objects: HashMap<usize, MovingObject>,
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
                    for (bus, gain) in object.current.iter().copied().enumerate() {
                        if gain != 0.0 {
                            self.writer.add(bus, sample * gain)?;
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
            for object in self.objects.values_mut() {
                object.advance();
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
            self.object_channels.insert(source_channel);
            let target = renderer.gains(source_channel, true, 1.0)?;
            self.objects
                .insert(source_channel, MovingObject::new(target));
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
        self.writer.add(bus, sample)
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
                "object-audio sample rate {sample_rate} does not match HRIR sample rate {}; resampling is not implemented yet",
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
        for state in update.isf {
            let renderer = self.isf.as_ref().ok_or_else(|| {
                AppError::Render("received ISF metadata before an ISF program assignment".into())
            })?;
            let target = renderer.gains(state.source_channel, state.active, state.gain)?;
            self.objects
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
            let target = if state.active
                && !(state.bed && self.options.mute_bed)
                && !(self.options.mute_ground && state.position[2] <= 0.0)
            {
                self.panner.gains(state.position, state.size, state.gain)?
            } else {
                vec![0.0; self.panner.bus_count()]
            };
            match self.objects.get_mut(&state.source_channel) {
                Some(object) => object.set_target(target, update.ramp_samples),
                None => {
                    self.objects
                        .insert(state.source_channel, MovingObject::new(target));
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

struct MovingObject {
    current: Vec<f32>,
    target: Vec<f32>,
    step: Vec<f32>,
    remaining: usize,
}

impl MovingObject {
    fn new(target: Vec<f32>) -> Self {
        Self {
            current: target.clone(),
            step: vec![0.0; target.len()],
            target,
            remaining: 0,
        }
    }

    #[allow(clippy::cast_precision_loss)] // OAMD ramp durations are small bounded integers.
    fn set_target(&mut self, target: Vec<f32>, ramp_samples: usize) {
        self.target = target;
        if ramp_samples == 0 {
            self.current.clone_from(&self.target);
            self.step.fill(0.0);
            self.remaining = 0;
            return;
        }
        let duration = ramp_samples as f32;
        for ((step, current), target) in self.step.iter_mut().zip(&self.current).zip(&self.target) {
            *step = (*target - *current) / duration;
        }
        self.remaining = ramp_samples;
    }

    fn advance(&mut self) {
        if self.remaining == 0 {
            return;
        }
        for (current, step) in self.current.iter_mut().zip(&self.step) {
            *current += *step;
        }
        self.remaining -= 1;
        if self.remaining == 0 {
            self.current.clone_from(&self.target);
        }
    }
}

#[cfg(test)]
mod tests {
    use hound::WavReader;

    use super::{MovingObject, ObjectPcmFrame, ObjectRenderOptions, ObjectRenderer};
    use crate::{
        binaural::BinauralWriter,
        hrir::{HrirChannel, HrirSet, Speaker},
        object::{ObjectState, SpatialUpdate},
    };

    #[test]
    fn object_ramp_reaches_target_exactly() {
        let mut object = MovingObject::new(vec![0.0, 1.0]);
        object.set_target(vec![1.0, 0.0], 4);
        for _ in 0..4 {
            object.advance();
        }
        assert_eq!(object.current, [1.0, 0.0]);
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
        };
        let writer = BinauralWriter::new(
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
                        bed: false,
                        position: [-1.0, 1.0, 0.0],
                        gain: 1.0,
                        size: 0.0,
                    }],
                }],
            })
            .unwrap();
        renderer.finish().unwrap();

        let samples = WavReader::open(output)
            .unwrap()
            .samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let left_peak = samples.iter().step_by(2).copied().max().unwrap();
        let right_peak = samples.iter().skip(1).step_by(2).copied().max().unwrap();
        assert!(left_peak > right_peak.saturating_mul(10));
    }
}
