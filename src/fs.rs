use crate::Result;

use std::fs;
use std::io;
use std::path::Path;

pub trait File: io::Read + io::Write + io::Seek {
    fn metadata(&self) -> Result<fs::Metadata>;
    fn sync_all(&self) -> Result<()>;
    fn truncate(&self, size: i64) -> Result<()>;
}

pub trait LockFile {}

pub trait FileSystem {
    type File: File;
    type LockFile: LockFile;

    fn open(path: impl AsRef<Path>, flag: usize, mode: u32) -> Result<Self::File>;
    fn metadata(path: impl AsRef<Path>) -> Result<fs::Metadata>;
    fn remove(path: impl AsRef<Path>) -> Result<()>;
    fn rename(old: impl AsRef<Path>, new: &Path) -> Result<()>;
    fn read_dir(path: impl AsRef<Path>) -> Result<fs::Metadata>;
    fn create_lock_file(
        path: impl AsRef<Path>,
        perm: usize,
    ) -> Result<(Self::LockFile, AcquiredExisting)>;
}

pub enum AcquiredExisting {
    Yes,
    No,
}
