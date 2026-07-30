use std::{
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
};

use crate::{cli::Cli, error::AppError};

#[derive(Debug)]
pub struct ResolvedPaths {
    pub input: PathBuf,
    pub hrir: Option<PathBuf>,
    pub output: Option<PathBuf>,
    pub in_place: bool,
    pub room_correction: Option<PathBuf>,
}

impl ResolvedPaths {
    /// Resolves and validates all paths needed by the selected CLI operation.
    ///
    /// # Errors
    ///
    /// Returns an error for missing inputs, ambiguous or unsafe output paths,
    /// invalid numeric settings, and invalid custom HRIR paths.
    pub fn from_cli(cli: &Cli) -> Result<Self, AppError> {
        validate_numbers(cli)?;
        let input = existing_file(&cli.input, "input")?;

        if cli.list_tracks {
            return Ok(Self {
                input,
                hrir: None,
                output: None,
                in_place: false,
                room_correction: None,
            });
        }

        let hrir = cli
            .hrir
            .as_deref()
            .map(|path| existing_file(path, "HRIR"))
            .transpose()?;
        let room_correction = cli
            .room_correction
            .as_deref()
            .map(|path| existing_file(path, "room-correction"))
            .transpose()?;

        let requested_output = cli.output.clone().unwrap_or_else(|| input.clone());
        let output = resolve_output(&requested_output)?;

        if !output
            .extension()
            .and_then(OsStr::to_str)
            .is_some_and(|extension| extension.eq_ignore_ascii_case("mkv"))
        {
            return Err(AppError::Usage(
                "the output path must have an .mkv extension".into(),
            ));
        }
        let in_place = paths_equal(&input, &output)?;
        if output.exists() && !in_place && !cli.overwrite {
            return Err(AppError::Usage(format!(
                "the output already exists: {}; use --overwrite to replace it",
                output.display()
            )));
        }

        Ok(Self {
            input,
            hrir,
            output: Some(output),
            in_place,
            room_correction,
        })
    }
}

fn validate_numbers(cli: &Cli) -> Result<(), AppError> {
    if !cli.effect.is_finite() || !(0.0..=100.0).contains(&cli.effect) {
        return Err(AppError::Usage(
            "--effect must be a finite number between 0 and 100".into(),
        ));
    }
    if !cli.smoothness.is_finite() || !(0.0..=100.0).contains(&cli.smoothness) {
        return Err(AppError::Usage(
            "--smoothness must be a finite number between 0 and 100".into(),
        ));
    }
    if !cli.gain_db.is_finite() {
        return Err(AppError::Usage("--gain-db must be a finite number".into()));
    }
    Ok(())
}

fn existing_file(path: &Path, description: &str) -> Result<PathBuf, AppError> {
    let resolved = fs::canonicalize(path).map_err(|source| AppError::File {
        path: path.to_path_buf(),
        source,
    })?;
    if !resolved.is_file() {
        return Err(AppError::Usage(format!(
            "{description} is not a regular file: {}",
            resolved.display()
        )));
    }
    Ok(resolved)
}

fn resolve_output(path: &Path) -> Result<PathBuf, AppError> {
    let file_name = path
        .file_name()
        .ok_or_else(|| AppError::Usage(format!("output has no file name: {}", path.display())))?;
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let parent = fs::canonicalize(parent).map_err(|source| AppError::File {
        path: parent.to_path_buf(),
        source,
    })?;
    if !parent.is_dir() {
        return Err(AppError::Usage(format!(
            "output parent is not a directory: {}",
            parent.display()
        )));
    }
    Ok(parent.join(file_name))
}

fn paths_equal(first: &Path, second: &Path) -> Result<bool, AppError> {
    let first = fs::canonicalize(first).map_err(|source| AppError::File {
        path: first.to_path_buf(),
        source,
    })?;
    let second = if second.exists() {
        fs::canonicalize(second).map_err(|source| AppError::File {
            path: second.to_path_buf(),
            source,
        })?
    } else {
        second.to_path_buf()
    };

    #[cfg(target_os = "macos")]
    {
        Ok(first
            .to_string_lossy()
            .eq_ignore_ascii_case(&second.to_string_lossy()))
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(first == second)
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use clap::Parser;

    use super::ResolvedPaths;
    use crate::cli::Cli;

    #[test]
    fn default_output_is_the_input() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("Movie.mkv");
        fs::write(&input, b"fixture").unwrap();
        let cli = Cli::try_parse_from(["surroundfold", input.to_str().unwrap()]).unwrap();
        let paths = ResolvedPaths::from_cli(&cli).unwrap();

        assert_eq!(paths.output.as_deref(), Some(paths.input.as_path()));
        assert!(paths.in_place);
    }

    #[test]
    fn explicit_output_is_not_in_place() {
        let directory = tempfile::tempdir().unwrap();
        let input = directory.path().join("Movie.mkv");
        let output = directory.path().join("Other.mkv");
        fs::write(&input, b"fixture").unwrap();
        let cli = Cli::try_parse_from([
            "surroundfold",
            input.to_str().unwrap(),
            "--output",
            output.to_str().unwrap(),
        ])
        .unwrap();
        let paths = ResolvedPaths::from_cli(&cli).unwrap();
        let expected = fs::canonicalize(directory.path())
            .unwrap()
            .join("Other.mkv");

        assert_eq!(paths.output.as_deref(), Some(expected.as_path()));
        assert!(!paths.in_place);
    }
}
