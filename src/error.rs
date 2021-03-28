use std::io;

pub type Result<T> = std::result::Result<T, Error>;

pub enum Error {
    AcquiredExisting,
    DatabaseLocked,
    Io(io::Error),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}
