use std::path::Path;

use hound::{SampleFormat, WavSpec, WavWriter};

pub fn generate_hrir(path: &Path) {
    generate_hrir_channels(path, 6);
}

#[allow(dead_code)] // This shared module is compiled independently by each integration-test crate.
pub fn generate_height_hrir(path: &Path) {
    generate_hrir_channels(path, 10);
}

#[allow(dead_code)] // This shared module is compiled independently by each integration-test crate.
pub fn generate_height_discrimination_hrir(path: &Path) {
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for channel in 0..10 {
        for sample in 0..128 {
            let (left, right) = if sample != 0 {
                (0.0, 0.0)
            } else if channel == 6 {
                // Side-left: deliberately opposite to top-front-left.
                (0.0, 0.8)
            } else if channel == 8 {
                // Top-front-left.
                (0.8, 0.0)
            } else {
                (0.7, 0.7)
            };
            writer.write_sample(left).unwrap();
            writer.write_sample(right).unwrap();
        }
    }
    writer.finalize().unwrap();
}

fn generate_hrir_channels(path: &Path, channels: usize) {
    let spec = WavSpec {
        channels: 2,
        sample_rate: 48_000,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut writer = WavWriter::create(path, spec).unwrap();
    for channel in 0..channels {
        for sample in 0..128 {
            let (left, right) = if sample == 0 {
                match channel {
                    0 | 4 => (1.0, 0.2),
                    1 | 5 => (0.2, 1.0),
                    _ => (0.7, 0.7),
                }
            } else {
                (0.0, 0.0)
            };
            writer.write_sample(left).unwrap();
            writer.write_sample(right).unwrap();
        }
    }
    writer.finalize().unwrap();
}
