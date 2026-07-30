use std::{
    env,
    ffi::{OsStr, OsString},
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
use nix::{
    sys::signal::{Signal, kill},
    unistd::Pid,
};
#[cfg(unix)]
use std::os::unix::{fs::PermissionsExt, process::CommandExt};

use crate::{cancel::Cancellation, error::AppError};

#[derive(Debug)]
pub struct ProcessOutput {
    pub status: ExitStatus,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

#[derive(Debug)]
pub struct PipelineOutput {
    pub producer_status: ExitStatus,
    pub consumer_status: ExitStatus,
    pub producer_stderr: Vec<u8>,
    pub consumer_stderr: Vec<u8>,
}

#[derive(Debug)]
pub struct StreamProcessOutput<T> {
    pub status: ExitStatus,
    pub value: T,
    pub stderr: Vec<u8>,
}

#[derive(Debug)]
pub struct ProcessRunner {
    cancellation: Cancellation,
}

impl ProcessRunner {
    #[must_use]
    pub const fn new(cancellation: Cancellation) -> Self {
        Self { cancellation }
    }

    /// Resolves an explicit executable or searches `PATH`.
    ///
    /// # Errors
    ///
    /// Returns an error when the resolved path is missing, not a regular file,
    /// or not executable.
    pub fn locate_required(
        &self,
        name: &str,
        explicit: Option<&Path>,
    ) -> Result<PathBuf, AppError> {
        if let Some(path) = explicit {
            return executable(path).ok_or_else(|| {
                AppError::Dependency(format!(
                    "{name} is not an executable file: {}",
                    path.display()
                ))
            });
        }

        let path = env::var_os("PATH")
            .and_then(|value| {
                env::split_paths(&value)
                    .map(|directory| directory.join(name))
                    .find_map(|candidate| executable(&candidate))
            })
            .ok_or_else(|| AppError::Dependency(format!("{name} was not found on PATH")))?;
        Ok(path)
    }

    /// Runs `-version` and returns the first non-empty version line.
    ///
    /// # Errors
    ///
    /// Returns an error if the process cannot start, is cancelled, exits
    /// unsuccessfully, or does not report a version.
    pub fn check_version(&self, executable: &Path) -> Result<String, AppError> {
        let output = self.run(executable, [OsStr::new("-version")])?;
        if !output.status.success() {
            return Err(AppError::Dependency(format!(
                "{} -version failed with {}: {}",
                executable.display(),
                output.status,
                concise(&output.stderr)
            )));
        }
        let first_line = String::from_utf8_lossy(&output.stdout)
            .lines()
            .next()
            .unwrap_or_default()
            .trim()
            .to_owned();
        if first_line.is_empty() {
            return Err(AppError::Dependency(format!(
                "{} -version returned no version information",
                executable.display()
            )));
        }
        Ok(first_line)
    }

    /// Checks cancellation between in-process processing blocks.
    ///
    /// # Errors
    ///
    /// Returns [`AppError::Cancelled`] after an interrupt.
    pub fn check_cancelled(&self) -> Result<(), AppError> {
        self.cancellation.check()
    }

    /// Executes a program directly, captures both output streams, and observes
    /// cancellation without invoking a shell.
    ///
    /// # Errors
    ///
    /// Returns an error when the child cannot start, pipe collection fails, or
    /// cancellation is requested.
    pub fn run<I, S>(&self, executable: &Path, args: I) -> Result<ProcessOutput, AppError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.cancellation.check()?;
        let args: Vec<OsString> = args
            .into_iter()
            .map(|value| value.as_ref().to_os_string())
            .collect();
        let mut command = Command::new(executable);
        command
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command.process_group(0);

        let mut child = command.spawn().map_err(|source| AppError::File {
            path: executable.to_path_buf(),
            source,
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AppError::Dependency("could not capture child stdout".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AppError::Dependency("could not capture child stderr".into()))?;
        let stdout_reader = thread::spawn(move || read_all(stdout));
        let stderr_reader = thread::spawn(move || read_all(stderr));

        let status = loop {
            if self.cancellation.is_cancelled() {
                terminate(&mut child);
                let _ = stdout_reader.join();
                let _ = stderr_reader.join();
                return Err(AppError::Cancelled);
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
            thread::sleep(Duration::from_millis(20));
        };

        let stdout = stdout_reader
            .join()
            .map_err(|_| AppError::Dependency("child stdout reader panicked".into()))??;
        let stderr = stderr_reader
            .join()
            .map_err(|_| AppError::Dependency("child stderr reader panicked".into()))??;
        Ok(ProcessOutput {
            status,
            stdout,
            stderr,
        })
    }

    /// Executes a program and lets the caller consume stdout while the child
    /// is still running. Stderr is drained concurrently to avoid pipe
    /// backpressure.
    ///
    /// # Errors
    ///
    /// Returns an error when the child cannot start, the consumer fails,
    /// stderr collection fails, or cancellation is requested.
    pub fn run_with_stdout<I, S, T>(
        &self,
        executable: &Path,
        args: I,
        consume: impl FnOnce(std::process::ChildStdout) -> Result<T, AppError>,
    ) -> Result<StreamProcessOutput<T>, AppError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.cancellation.check()?;
        let mut command = Command::new(executable);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        command.process_group(0);

        let mut child = command.spawn().map_err(|source| AppError::File {
            path: executable.to_path_buf(),
            source,
        })?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| AppError::Dependency("could not capture child stdout".into()))?;
        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| AppError::Dependency("could not capture child stderr".into()))?;
        let stderr_reader = thread::spawn(move || read_all(stderr));

        let value = match consume(stdout) {
            Ok(value) => value,
            Err(error) => {
                terminate(&mut child);
                let _ = stderr_reader.join();
                return Err(error);
            }
        };
        let status = loop {
            if self.cancellation.is_cancelled() {
                terminate(&mut child);
                let _ = stderr_reader.join();
                return Err(AppError::Cancelled);
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
            thread::sleep(Duration::from_millis(20));
        };
        let stderr = stderr_reader
            .join()
            .map_err(|_| AppError::Dependency("child stderr reader panicked".into()))??;
        Ok(StreamProcessOutput {
            status,
            value,
            stderr,
        })
    }

    /// Pipes one process directly into another while draining both error
    /// streams and observing cancellation for both process groups.
    ///
    /// # Errors
    ///
    /// Returns an error when either child cannot start, stderr collection
    /// fails, or cancellation is requested.
    pub fn run_pipeline<I, S, J, T>(
        &self,
        producer_executable: &Path,
        producer_args: I,
        consumer_executable: &Path,
        consumer_args: J,
    ) -> Result<PipelineOutput, AppError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
        J: IntoIterator<Item = T>,
        T: AsRef<OsStr>,
    {
        self.cancellation.check()?;

        let mut producer_command = Command::new(producer_executable);
        producer_command
            .args(producer_args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        producer_command.process_group(0);
        let mut producer = producer_command.spawn().map_err(|source| AppError::File {
            path: producer_executable.to_path_buf(),
            source,
        })?;
        let producer_stdout = producer
            .stdout
            .take()
            .ok_or_else(|| AppError::Dependency("could not pipe producer stdout".into()))?;
        let producer_stderr = producer
            .stderr
            .take()
            .ok_or_else(|| AppError::Dependency("could not capture producer stderr".into()))?;

        let mut consumer_command = Command::new(consumer_executable);
        consumer_command
            .args(consumer_args)
            .stdin(Stdio::from(producer_stdout))
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        consumer_command.process_group(0);
        let mut consumer = match consumer_command.spawn() {
            Ok(child) => child,
            Err(source) => {
                terminate(&mut producer);
                let _ = producer.wait();
                return Err(AppError::File {
                    path: consumer_executable.to_path_buf(),
                    source,
                });
            }
        };
        let consumer_stderr = consumer
            .stderr
            .take()
            .ok_or_else(|| AppError::Dependency("could not capture consumer stderr".into()))?;
        let producer_stderr_reader = thread::spawn(move || read_all(producer_stderr));
        let consumer_stderr_reader = thread::spawn(move || read_all(consumer_stderr));

        let mut producer_status = None;
        let mut consumer_status = None;
        while producer_status.is_none() || consumer_status.is_none() {
            if self.cancellation.is_cancelled() {
                terminate(&mut producer);
                terminate(&mut consumer);
                let _ = producer.wait();
                let _ = consumer.wait();
                let _ = producer_stderr_reader.join();
                let _ = consumer_stderr_reader.join();
                return Err(AppError::Cancelled);
            }
            if producer_status.is_none() {
                producer_status = producer.try_wait()?;
            }
            if consumer_status.is_none() {
                consumer_status = consumer.try_wait()?;
            }
            if producer_status.is_none() || consumer_status.is_none() {
                thread::sleep(Duration::from_millis(20));
            }
        }

        let producer_stderr = producer_stderr_reader
            .join()
            .map_err(|_| AppError::Dependency("producer stderr reader panicked".into()))??;
        let consumer_stderr = consumer_stderr_reader
            .join()
            .map_err(|_| AppError::Dependency("consumer stderr reader panicked".into()))??;
        let Some(producer_status) = producer_status else {
            return Err(AppError::Dependency(
                "producer exited without a process status".into(),
            ));
        };
        let Some(consumer_status) = consumer_status else {
            return Err(AppError::Dependency(
                "consumer exited without a process status".into(),
            ));
        };
        Ok(PipelineOutput {
            producer_status,
            consumer_status,
            producer_stderr,
            consumer_stderr,
        })
    }
}

fn read_all(mut stream: impl Read) -> std::io::Result<Vec<u8>> {
    let mut result = Vec::new();
    stream.read_to_end(&mut result)?;
    Ok(result)
}

fn executable(path: &Path) -> Option<PathBuf> {
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    #[cfg(unix)]
    if metadata.permissions().mode() & 0o111 == 0 {
        return None;
    }
    Some(fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()))
}

fn concise(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let trimmed = text.trim();
    if trimmed.len() <= 500 {
        trimmed.to_owned()
    } else {
        format!("{}…", &trimmed[..500])
    }
}

fn terminate(child: &mut std::process::Child) {
    #[cfg(unix)]
    {
        let process_group = -(i32::try_from(child.id()).unwrap_or(i32::MAX));
        let _ = kill(Pid::from_raw(process_group), Signal::SIGINT);
        if wait_until(child, Duration::from_millis(500)) {
            return;
        }
        let _ = kill(Pid::from_raw(process_group), Signal::SIGTERM);
        if wait_until(child, Duration::from_millis(500)) {
            return;
        }
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn wait_until(child: &mut std::process::Child, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if child.try_wait().ok().flatten().is_some() {
            return true;
        }
        thread::sleep(Duration::from_millis(20));
    }
    false
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::ProcessRunner;
    use crate::cancel::Cancellation;

    #[test]
    fn captures_stdout_without_a_shell() {
        let cancellation = Cancellation::new();
        let runner = ProcessRunner::new(cancellation);
        let printf = runner.locate_required("printf", None).unwrap();
        let output = runner
            .run(&printf, [OsStr::new("%s"), OsStr::new("literal $HOME")])
            .unwrap();
        assert!(output.status.success());
        assert_eq!(output.stdout, b"literal $HOME");
    }

    #[test]
    fn streams_stdout_to_a_consumer() {
        let cancellation = Cancellation::new();
        let runner = ProcessRunner::new(cancellation);
        let printf = runner.locate_required("printf", None).unwrap();
        let output = runner
            .run_with_stdout(
                &printf,
                [OsStr::new("%s"), OsStr::new("streamed")],
                |mut stdout| {
                    let mut bytes = Vec::new();
                    std::io::Read::read_to_end(&mut stdout, &mut bytes)?;
                    Ok(bytes)
                },
            )
            .unwrap();
        assert!(output.status.success());
        assert_eq!(output.value, b"streamed");
    }

    #[test]
    fn pipes_one_process_into_another() {
        let runner = ProcessRunner::new(Cancellation::new());
        let printf = runner.locate_required("printf", None).unwrap();
        let tee = runner.locate_required("tee", None).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let output_path = directory.path().join("piped.txt");
        let output = runner
            .run_pipeline(
                &printf,
                [OsStr::new("%s"), OsStr::new("piped")],
                &tee,
                [output_path.as_os_str()],
            )
            .unwrap();
        assert!(output.producer_status.success());
        assert!(output.consumer_status.success());
        assert_eq!(std::fs::read(output_path).unwrap(), b"piped");
    }
}
