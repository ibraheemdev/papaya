use crate::Result;

use std::fs;
use std::io;
use std::path::Path;

pub trait File: io::Read + io::Write + io::Seek {
    fn metadata(&self) -> Result<fs::Metadata>;
    fn sync_all(&self) -> Result<()>;
    fn truncate(&self, size: i64) -> Result<()>;
}

pub trait FileSystem {
    type File: File;

    fn open(path: impl AsRef<Path>, flag: usize, mode: u32) -> Result<Self::File>;
    fn metadata(path: impl AsRef<Path>) -> Result<fs::Metadata>;
    fn remove(path: impl AsRef<Path>) -> Result<()>;
    fn rename(&mut self, from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<()>;
    fn read_dir(&self, path: impl AsRef<Path>) -> Result<fs::ReadDir>;
    fn create_lock_file(&self, path: impl AsRef<Path>, perm: usize) -> Result<()>;
}

pub enum AcquiredExisting {
    Yes,
    No,
}
