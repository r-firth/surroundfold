use std::{
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

use crate::cli::ProgressMode;

const UPDATE_INTERVAL: Duration = Duration::from_secs(5);
const WAV_STEREO_I24_BYTES_PER_FRAME: u64 = 6;

#[derive(Debug)]
enum ProgressSource {
    StereoI24Wave { path: PathBuf, sample_rate: u32 },
    Ffmpeg { path: PathBuf },
}

impl ProgressSource {
    fn completed_seconds(&self) -> Option<f64> {
        match self {
            Self::StereoI24Wave { path, sample_rate } => {
                let bytes = fs::metadata(path).ok()?.len();
                let frames = bytes / WAV_STEREO_I24_BYTES_PER_FRAME;
                Some(frame_duration(frames, *sample_rate))
            }
            Self::Ffmpeg { path } => ffmpeg_completed_seconds(path),
        }
    }
}

#[derive(Debug)]
struct Worker {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

/// Periodically reports progress without entering or synchronizing with the audio path.
#[derive(Debug)]
pub(crate) struct ProgressMonitor {
    worker: Option<Worker>,
}

impl ProgressMonitor {
    pub(crate) fn render(
        mode: ProgressMode,
        wave: &Path,
        total_seconds: Option<f64>,
        sample_rate: u32,
    ) -> Self {
        Self::start(
            mode,
            "render",
            total_seconds,
            ProgressSource::StereoI24Wave {
                path: wave.to_path_buf(),
                sample_rate,
            },
        )
    }

    pub(crate) fn mux(
        mode: ProgressMode,
        ffmpeg_progress: &Path,
        total_seconds: Option<f64>,
    ) -> Self {
        Self::start(
            mode,
            "mux",
            total_seconds,
            ProgressSource::Ffmpeg {
                path: ffmpeg_progress.to_path_buf(),
            },
        )
    }

    fn start(
        mode: ProgressMode,
        label: &'static str,
        total_seconds: Option<f64>,
        source: ProgressSource,
    ) -> Self {
        let Some(total_seconds) = total_seconds.filter(|value| value.is_finite() && *value > 0.0)
        else {
            return Self { worker: None };
        };
        if mode != ProgressMode::Text {
            return Self { worker: None };
        }

        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            let started = Instant::now();
            let mut previous_seconds = 0.0;
            while !thread_stop.load(Ordering::Relaxed) {
                thread::park_timeout(UPDATE_INTERVAL);
                if thread_stop.load(Ordering::Relaxed) {
                    break;
                }
                let Some(completed_seconds) = source.completed_seconds() else {
                    continue;
                };
                if completed_seconds <= previous_seconds {
                    continue;
                }
                previous_seconds = completed_seconds;
                if let Some(line) = progress_line(
                    label,
                    completed_seconds,
                    total_seconds,
                    started.elapsed().as_secs_f64(),
                ) {
                    eprintln!("{line}");
                }
            }
        });
        Self {
            worker: Some(Worker {
                stop,
                thread: Some(thread),
            }),
        }
    }
}

impl Drop for ProgressMonitor {
    fn drop(&mut self) {
        let Some(worker) = self.worker.as_mut() else {
            return;
        };
        worker.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = worker.thread.take() {
            thread.thread().unpark();
            let _ = thread.join();
        }
    }
}

fn ffmpeg_completed_seconds(path: &Path) -> Option<f64> {
    let contents = fs::read_to_string(path).ok()?;
    contents.lines().rev().find_map(|line| {
        line.strip_prefix("out_time_us=")
            .and_then(|value| value.parse::<u64>().ok())
            .map(|microseconds| Duration::from_micros(microseconds).as_secs_f64())
    })
}

fn frame_duration(frames: u64, sample_rate: u32) -> f64 {
    let sample_rate_u64 = u64::from(sample_rate);
    let whole_seconds = frames / sample_rate_u64;
    let remainder = u32::try_from(frames % sample_rate_u64)
        .expect("a frame remainder is always smaller than the u32 sample rate");
    Duration::from_secs(whole_seconds).as_secs_f64() + f64::from(remainder) / f64::from(sample_rate)
}

fn progress_line(
    label: &str,
    completed_seconds: f64,
    total_seconds: f64,
    elapsed_seconds: f64,
) -> Option<String> {
    if !completed_seconds.is_finite()
        || !total_seconds.is_finite()
        || !elapsed_seconds.is_finite()
        || completed_seconds <= 0.0
        || total_seconds <= 0.0
        || elapsed_seconds <= 0.0
    {
        return None;
    }
    let progress = (completed_seconds / total_seconds).clamp(0.0, 0.999);
    let realtime = completed_seconds / elapsed_seconds;
    if !realtime.is_finite() || realtime <= 0.0 {
        return None;
    }
    let eta = (total_seconds - completed_seconds).max(0.0) / realtime;
    Some(format!(
        "{label}: {:5.1}% | {realtime:.2}x real-time | ETA {}",
        progress * 100.0,
        format_duration(eta)
    ))
}

fn format_duration(seconds: f64) -> String {
    let seconds = Duration::try_from_secs_f64(seconds.max(0.0).ceil())
        .map_or(u64::MAX, |duration| duration.as_secs());
    let hours = seconds / 3_600;
    let minutes = seconds % 3_600 / 60;
    let seconds = seconds % 60;
    if hours == 0 {
        format!("{minutes:02}:{seconds:02}")
    } else {
        format!("{hours}:{minutes:02}:{seconds:02}")
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use tempfile::tempdir;

    use super::{ffmpeg_completed_seconds, format_duration, progress_line};

    #[test]
    fn parses_the_latest_complete_ffmpeg_timestamp() {
        let directory = tempdir().unwrap();
        let progress = directory.path().join("progress.txt");
        fs::write(
            &progress,
            "out_time_us=1250000\nprogress=continue\nout_time_us=2750000\nprogress=continue\n",
        )
        .unwrap();
        assert_eq!(ffmpeg_completed_seconds(&progress), Some(2.75));
    }

    #[test]
    fn formats_progress_with_a_bounded_percentage_and_eta() {
        assert_eq!(format_duration(65.1), "01:06");
        assert_eq!(format_duration(3_661.0), "1:01:01");
        assert_eq!(
            progress_line("render", 900.0, 3_600.0, 300.0).as_deref(),
            Some("render:  25.0% | 3.00x real-time | ETA 15:00")
        );
        assert!(
            progress_line("mux", 3_601.0, 3_600.0, 300.0)
                .unwrap()
                .contains(" 99.9%")
        );
    }
}
