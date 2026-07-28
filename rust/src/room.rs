use std::{
    ffi::{OsStr, OsString},
    path::{Path, PathBuf},
};

use crate::{dsp::StereoConvolver, error::AppError, hrir::read_wave};

#[derive(Clone, Debug)]
pub struct RoomCorrection {
    pub sample_rate: u32,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
}

impl RoomCorrection {
    /// Loads the `SL`/`SR` or numbered mono FIR files beside a correction root.
    ///
    /// # Errors
    ///
    /// Returns an error when either impulse is missing, malformed, non-mono,
    /// empty, non-finite, or uses a different sample rate from the HRIR.
    pub fn load(root: &Path, expected_sample_rate: u32) -> Result<Self, AppError> {
        let left_path = correction_path(root, "SL", 1)?;
        let right_path = correction_path(root, "SR", 2)?;
        let left = read_wave(&left_path, "left room-correction impulse")?;
        let right = read_wave(&right_path, "right room-correction impulse")?;
        if left.channels.len() != 1 || right.channels.len() != 1 {
            return Err(AppError::InvalidHrir(
                "room-correction impulses must be mono WAV files".into(),
            ));
        }
        if left.sample_rate != right.sample_rate {
            return Err(AppError::InvalidHrir(
                "room-correction WAV sample rates do not match".into(),
            ));
        }
        if left.sample_rate != expected_sample_rate {
            return Err(AppError::InvalidHrir(format!(
                "room-correction sample rate {} does not match HRIR sample rate {expected_sample_rate}",
                left.sample_rate
            )));
        }
        Ok(Self {
            sample_rate: left.sample_rate,
            left: left.channels.into_iter().next().unwrap_or_default(),
            right: right.channels.into_iter().next().unwrap_or_default(),
        })
    }
}

#[derive(Debug)]
pub struct StereoRoomCorrector {
    left: StereoConvolver,
    right: StereoConvolver,
    input_left: Vec<f32>,
    input_right: Vec<f32>,
    output_left: Vec<f32>,
    output_right: Vec<f32>,
    discard: Vec<f32>,
}

impl StereoRoomCorrector {
    /// Prepares streaming stereo room-correction convolution.
    ///
    /// # Errors
    ///
    /// Returns an error if either correction impulse or the block size is
    /// invalid.
    pub fn new(correction: &RoomCorrection, block_size: usize) -> Result<Self, AppError> {
        Ok(Self {
            left: StereoConvolver::new(&correction.left, &[0.0], block_size)?,
            right: StereoConvolver::new(&[0.0], &correction.right, block_size)?,
            input_left: vec![0.0; block_size],
            input_right: vec![0.0; block_size],
            output_left: vec![0.0; block_size],
            output_right: vec![0.0; block_size],
            discard: vec![0.0; block_size],
        })
    }

    /// Applies correction to an interleaved stereo block in place.
    ///
    /// # Errors
    ///
    /// Returns an error for a non-stereo or oversized block, or if convolution
    /// fails.
    pub fn process(&mut self, stereo: &mut [f32]) -> Result<(), AppError> {
        if stereo.len() % 2 != 0 || stereo.len() / 2 > self.input_left.len() {
            return Err(AppError::Render(
                "room-correction block has an invalid length".into(),
            ));
        }
        let frames = stereo.len() / 2;
        for (frame, samples) in stereo.chunks_exact(2).enumerate() {
            self.input_left[frame] = samples[0];
            self.input_right[frame] = samples[1];
        }
        self.left.process(
            &self.input_left[..frames],
            &mut self.output_left,
            &mut self.discard,
        )?;
        self.right.process(
            &self.input_right[..frames],
            &mut self.discard,
            &mut self.output_right,
        )?;
        for (frame, samples) in stereo.chunks_exact_mut(2).enumerate() {
            samples[0] = self.output_left[frame];
            samples[1] = self.output_right[frame];
        }
        Ok(())
    }
}

fn correction_path(root: &Path, channel: &str, number: u8) -> Result<PathBuf, AppError> {
    let parent = root.parent().unwrap_or_else(|| Path::new("."));
    let stem = root.file_stem().ok_or_else(|| {
        AppError::InvalidHrir(format!(
            "room-correction root has no file stem: {}",
            root.display()
        ))
    })?;
    let named = parent.join(suffixed_name(stem, &format!(" {channel}.wav")));
    if named.is_file() {
        return Ok(named);
    }
    let numbered = parent.join(suffixed_name(stem, &format!(" {number}.wav")));
    if numbered.is_file() {
        return Ok(numbered);
    }
    Err(AppError::InvalidHrir(format!(
        "room-correction impulse for {channel} was not found beside {}",
        root.display()
    )))
}

fn suffixed_name(stem: &OsStr, suffix: &str) -> OsString {
    let mut result = stem.to_os_string();
    result.push(suffix);
    result
}

#[cfg(test)]
mod tests {
    use std::{fs, path::Path};

    use hound::{SampleFormat, WavSpec, WavWriter};

    use super::{RoomCorrection, StereoRoomCorrector};

    #[test]
    fn loads_named_stereo_filter_set() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("Cinema.txt");
        fs::write(&root, "").unwrap();
        write_impulse(&directory.path().join("Cinema SL.wav"), 0.5);
        write_impulse(&directory.path().join("Cinema SR.wav"), 0.25);
        let correction = RoomCorrection::load(&root, 48_000).unwrap();
        assert_eq!(correction.left, [0.5]);
        assert_eq!(correction.right, [0.25]);
    }

    #[test]
    fn applies_independent_left_and_right_filters() {
        let correction = RoomCorrection {
            sample_rate: 48_000,
            left: vec![0.5],
            right: vec![0.25],
        };
        let mut corrector = StereoRoomCorrector::new(&correction, 4).unwrap();
        let mut samples = [1.0, 1.0, 0.5, 0.5];
        corrector.process(&mut samples).unwrap();
        assert!((samples[0] - 0.5).abs() < 1e-5);
        assert!((samples[1] - 0.25).abs() < 1e-5);
        assert!((samples[2] - 0.25).abs() < 1e-5);
        assert!((samples[3] - 0.125).abs() < 1e-5);
    }

    fn write_impulse(path: &Path, value: f32) {
        let spec = WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: SampleFormat::Float,
        };
        let mut writer = WavWriter::create(path, spec).unwrap();
        writer.write_sample(value).unwrap();
        writer.finalize().unwrap();
    }
}
