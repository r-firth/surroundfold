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
}
