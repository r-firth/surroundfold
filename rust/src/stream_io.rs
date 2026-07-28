use std::io::{self, Read};

use crate::error::AppError;

/// Fills as much of `buffer` as possible, stopping only at EOF and retrying
/// interrupted reads.
pub(crate) fn read_up_to(reader: &mut impl Read, buffer: &mut [u8]) -> Result<usize, AppError> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(read) => filled += read,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(filled)
}
