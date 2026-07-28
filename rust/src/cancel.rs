use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::error::AppError;

#[derive(Clone, Debug, Default)]
pub struct Cancellation(Arc<AtomicBool>);

impl Cancellation {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Installs the process-wide interrupt handler.
    ///
    /// # Errors
    ///
    /// Returns an error if another component already installed an incompatible
    /// Ctrl-C handler.
    pub fn install_handler(&self) -> Result<(), AppError> {
        let cancelled = Arc::clone(&self.0);
        ctrlc::set_handler(move || cancelled.store(true, Ordering::SeqCst)).map_err(|error| {
            AppError::Dependency(format!("could not install Ctrl-C handler: {error}"))
        })
    }

    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }

    /// Returns a cancellation error once an interrupt has been observed.
    ///
    /// # Errors
    ///
    /// Returns [`AppError::Cancelled`] after cancellation.
    pub fn check(&self) -> Result<(), AppError> {
        if self.is_cancelled() {
            Err(AppError::Cancelled)
        } else {
            Ok(())
        }
    }

    #[cfg(test)]
    pub(crate) fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::Cancellation;

    #[test]
    fn explicit_cancellation_is_observed() {
        let cancellation = Cancellation::new();
        cancellation.cancel();
        assert!(cancellation.check().is_err());
    }
}
