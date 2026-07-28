use std::{
    path::{Component, Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

use tempfile::{Builder, TempDir};

use crate::error::AppError;

static RUN_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
pub struct Workspace {
    directory: Option<TempDir>,
    path: PathBuf,
    keep: bool,
}

impl Workspace {
    /// Creates an isolated workspace under the system temporary directory.
    ///
    /// # Errors
    ///
    /// Returns an error when the directory cannot be created.
    pub fn new(keep: bool) -> Result<Self, AppError> {
        let sequence = RUN_ID.fetch_add(1, Ordering::Relaxed);
        let prefix = format!("surroundfold-{}-{sequence}-", std::process::id());
        let directory = Builder::new().prefix(&prefix).tempdir()?;
        let path = directory.path().to_path_buf();
        Ok(Self {
            directory: Some(directory),
            path,
            keep,
        })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns a path for one direct child of the workspace.
    ///
    /// # Errors
    ///
    /// Rejects absolute paths, parent traversal, and nested paths.
    pub fn file(&self, name: &str) -> Result<PathBuf, AppError> {
        let mut components = Path::new(name).components();
        let valid =
            matches!(components.next(), Some(Component::Normal(_))) && components.next().is_none();
        if !valid {
            return Err(AppError::Usage(format!(
                "invalid workspace file name: {name}"
            )));
        }
        Ok(self.path().join(name))
    }
}

impl Drop for Workspace {
    fn drop(&mut self) {
        if self.keep {
            if let Some(directory) = self.directory.take() {
                let _retained_path = directory.keep();
            }
        }
    }
}

#[derive(Debug)]
pub struct AtomicOutput {
    final_path: PathBuf,
    partial_path: PathBuf,
    directory: Option<TempDir>,
    overwrite: bool,
}

impl AtomicOutput {
    /// Creates a same-filesystem directory for a partial output.
    ///
    /// # Errors
    ///
    /// Returns an error when the final path has no existing parent directory or
    /// the private partial directory cannot be created.
    pub fn new(final_path: &Path, overwrite: bool) -> Result<Self, AppError> {
        let parent = final_path.parent().ok_or_else(|| {
            AppError::Usage(format!(
                "output has no parent directory: {}",
                final_path.display()
            ))
        })?;
        let stem = final_path
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("output");
        let prefix = format!(".{stem}.");
        let directory = Builder::new().prefix(&prefix).tempdir_in(parent)?;
        let partial_path = directory.path().join("partial.mkv");
        Ok(Self {
            final_path: final_path.to_path_buf(),
            partial_path,
            directory: Some(directory),
            overwrite,
        })
    }

    #[must_use]
    pub fn partial_path(&self) -> &Path {
        &self.partial_path
    }

    /// Atomically publishes the completed partial file.
    ///
    /// # Errors
    ///
    /// Returns an error if the partial file is missing, overwrite policy has
    /// changed since validation, or the same-filesystem rename fails.
    pub fn commit(mut self) -> Result<(), AppError> {
        if !self.partial_path.is_file() {
            return Err(AppError::Mux(format!(
                "verified partial output is missing: {}",
                self.partial_path.display()
            )));
        }
        if self.final_path.exists() && !self.overwrite {
            return Err(AppError::Mux(format!(
                "output appeared during rendering and overwrite is disabled: {}",
                self.final_path.display()
            )));
        }
        std::fs::rename(&self.partial_path, &self.final_path).map_err(|error| {
            AppError::Mux(format!(
                "could not atomically publish {}: {error}",
                self.final_path.display()
            ))
        })?;
        self.directory.take();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{AtomicOutput, Workspace};

    #[test]
    fn workspace_rejects_path_traversal() {
        let workspace = Workspace::new(false).unwrap();
        assert!(workspace.file("audio.wav").is_ok());
        assert!(workspace.file("../audio.wav").is_err());
        assert!(workspace.file("nested/audio.wav").is_err());
    }

    #[test]
    fn atomic_output_publishes_only_on_commit() {
        let directory = tempfile::tempdir().unwrap();
        let final_path = directory.path().join("result.mkv");
        let output = AtomicOutput::new(&final_path, false).unwrap();
        fs::write(output.partial_path(), b"verified").unwrap();
        assert!(!final_path.exists());
        output.commit().unwrap();
        assert_eq!(fs::read(final_path).unwrap(), b"verified");
    }

    #[test]
    fn dropped_partial_output_is_removed() {
        let directory = tempfile::tempdir().unwrap();
        let final_path = directory.path().join("result.mkv");
        {
            let output = AtomicOutput::new(&final_path, false).unwrap();
            fs::write(output.partial_path(), b"unverified").unwrap();
        }
        assert!(!final_path.exists());
    }
}
