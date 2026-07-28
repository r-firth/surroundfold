use std::process::ExitCode;

use clap::Parser;
use surroundfold::{cancel::Cancellation, cli::Cli};

fn main() -> ExitCode {
    let cli = Cli::parse();
    let cancellation = Cancellation::new();
    if let Err(error) = cancellation.install_handler() {
        eprintln!("error: {error}");
        return ExitCode::from(3);
    }

    match surroundfold::run(&cli, &cancellation) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::from(error.exit_code())
        }
    }
}
