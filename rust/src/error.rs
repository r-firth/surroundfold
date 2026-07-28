use std::{io, path::PathBuf};

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AppError {
    #[error("{0}")]
    Usage(String),

    #[error("{0}")]
    Dependency(String),

    #[error("{0}")]
    UnsupportedInput(String),

    #[error("{0}")]
    InvalidHrir(String),

    #[error("{0}")]
    Render(String),

    #[error("{0}")]
    Mux(String),

    #[error("operation cancelled")]
    Cancelled,

    #[error("could not access {path}: {source}")]
    File {
        path: PathBuf,
        #[source]
        source: io::Error,
    },

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Io(#[from] io::Error),
}

impl AppError {
    #[must_use]
    pub const fn exit_code(&self) -> u8 {
        match self {
            Self::Usage(_) => 2,
            Self::Dependency(_) => 3,
            Self::UnsupportedInput(_) => 4,
            Self::InvalidHrir(_) => 5,
            Self::Render(_) | Self::File { .. } | Self::Json(_) | Self::Io(_) => 6,
            Self::Mux(_) => 7,
            Self::Cancelled => 130,
        }
    }
}
