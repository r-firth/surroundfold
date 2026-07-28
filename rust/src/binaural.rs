use std::{collections::HashMap, fs::File, io::BufWriter, path::Path};

use hound::{SampleFormat, WavSpec, WavWriter};

use crate::{
    dsp::{DEFAULT_CONVOLUTION_BLOCK, PeakLimiter, StereoConvolver, TpdfDither},
    error::AppError,
    hrir::{HrirSet, Speaker},
    render::RenderResult,
    room::{RoomCorrection, StereoRoomCorrector},
};

/// Streaming virtual-speaker bus renderer shared by channel and object paths.
pub(crate) struct BinauralWriter {
    writer: WavWriter<BufWriter<File>>,
    sample_rate: u32,
    master_gain: f32,
    bus_by_speaker: HashMap<Speaker, usize>,
    convolvers: Vec<StereoConvolver>,
    buses: Vec<Vec<f32>>,
    filled: usize,
    convolved_left: Vec<f32>,
    convolved_right: Vec<f32>,
    stereo: Vec<f32>,
    direct_stereo: Vec<f32>,
    room_correction: Option<StereoRoomCorrector>,
    limiter: PeakLimiter,
    dither: TpdfDither,
    input_frames: u64,
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

        let mut maximum_impulse = 1;
        let convolvers = unique_speakers
            .iter()
            .map(|speaker| {
                let channel = hrir.channel(*speaker).ok_or_else(|| {
                    AppError::InvalidHrir(format!("HRIR has no usable response for {speaker:?}"))
                })?;
                maximum_impulse = maximum_impulse.max(channel.left.len().max(channel.right.len()));
                StereoConvolver::new(&channel.left, &channel.right, DEFAULT_CONVOLUTION_BLOCK)
            })
            .collect::<Result<Vec<_>, _>>()?;
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
            bits_per_sample: 16,
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

        Ok(Self {
            writer,
            sample_rate: hrir.sample_rate,
            master_gain,
            bus_by_speaker,
            buses: vec![vec![0.0; DEFAULT_CONVOLUTION_BLOCK]; convolvers.len()],
            convolvers,
            filled: 0,
            convolved_left: vec![0.0; DEFAULT_CONVOLUTION_BLOCK],
            convolved_right: vec![0.0; DEFAULT_CONVOLUTION_BLOCK],
            stereo: vec![0.0; DEFAULT_CONVOLUTION_BLOCK * 2],
            direct_stereo: vec![0.0; DEFAULT_CONVOLUTION_BLOCK * 2],
            room_correction: room_correction
                .map(|correction| StereoRoomCorrector::new(correction, DEFAULT_CONVOLUTION_BLOCK))
                .transpose()?,
            limiter: PeakLimiter::default(),
            dither: TpdfDither::default(),
            input_frames: 0,
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

    pub(crate) fn bus_by_speaker(&self) -> impl Iterator<Item = (Speaker, usize)> + '_ {
        self.bus_by_speaker
            .iter()
            .map(|(speaker, index)| (*speaker, *index))
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
        target[self.filled] += sample * self.master_gain;
        Ok(())
    }

    pub(crate) fn add_direct(&mut self, left: f32, right: f32) {
        self.direct_stereo[self.filled * 2] += left * self.master_gain;
        self.direct_stereo[self.filled * 2 + 1] += right * self.master_gain;
    }

    pub(crate) fn end_frame(&mut self) -> Result<(), AppError> {
        self.filled += 1;
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
        let target_frames = self
            .input_frames
            .checked_add(self.tail_frames)
            .ok_or_else(|| AppError::Render("rendered duration overflowed".into()))?;
        while self.written_frames < target_frames {
            let remaining = target_frames - self.written_frames;
            let write_frames = usize::try_from(remaining)
                .unwrap_or(DEFAULT_CONVOLUTION_BLOCK)
                .min(DEFAULT_CONVOLUTION_BLOCK);
            self.process_block(write_frames)?;
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

    fn process_block(&mut self, write_frames: usize) -> Result<(), AppError> {
        self.stereo.copy_from_slice(&self.direct_stereo);
        self.direct_stereo.fill(0.0);
        for (bus, convolver) in self.convolvers.iter_mut().enumerate() {
            convolver.process(
                &self.buses[bus],
                &mut self.convolved_left,
                &mut self.convolved_right,
            )?;
            for frame in 0..DEFAULT_CONVOLUTION_BLOCK {
                self.stereo[frame * 2] += self.convolved_left[frame];
                self.stereo[frame * 2 + 1] += self.convolved_right[frame];
            }
            self.buses[bus].fill(0.0);
        }
        if let Some(correction) = self.room_correction.as_mut() {
            correction.process(&mut self.stereo)?;
        }
        let output = &mut self.stereo[..write_frames * 2];
        self.peak_before_limiting = output
            .iter()
            .copied()
            .map(f32::abs)
            .fold(self.peak_before_limiting, f32::max);
        self.limiter.process(output, self.sample_rate);
        for sample in output {
            self.writer
                .write_sample(self.dither.quantize_i16(*sample))
                .map_err(|error| AppError::Render(format!("WAV write failed: {error}")))?;
        }
        self.written_frames = self
            .written_frames
            .checked_add(u64::try_from(write_frames).map_err(|error| {
                AppError::Render(format!("rendered frame count overflowed: {error}"))
            })?)
            .ok_or_else(|| AppError::Render("rendered frame count overflowed".into()))?;
        self.filled = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use hound::{SampleFormat, WavReader, WavSpec, WavWriter};

    use super::BinauralWriter;
    use crate::hrir::{HrirSet, Speaker};

    #[test]
    fn flushes_the_exact_impulse_tail() {
        let directory = tempfile::tempdir().unwrap();
        let hrir_path = directory.path().join("hrir.wav");
        let output = directory.path().join("out.wav");
        write_hrir(&hrir_path);
        let hrir = HrirSet::load_concatenated_wave(&hrir_path).unwrap();
        let mut writer =
            BinauralWriter::new(&output, &hrir, None, 0.0, [Speaker::FrontLeft]).unwrap();
        let bus = writer.bus(Speaker::FrontLeft).unwrap();
        writer.add(bus, 1.0).unwrap();
        writer.end_frame().unwrap();
        let result = writer.finish().unwrap();

        assert_eq!(result.frames, 128);
        assert_eq!(WavReader::open(output).unwrap().duration(), 128);
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
