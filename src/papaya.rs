use crate::{recovery, Config, Error, FileSystem, Index, Log, Result, Stats};

use std::fs;
use std::path::Path;

pub struct Papaya<FileSystem>
where
    FileSystem: self::FileSystem,
{
    config: Config<FileSystem>,
    index: Index<FileSystem>,
    datalog: Log<FileSystem>,
    seed: u32,
    stats: Stats,
    sync_writes: bool,
    compaction_running: i32,
}

impl<FileSystem> Papaya<FileSystem>
where
    FileSystem: self::FileSystem,
{
    fn open(path: impl AsRef<Path>, config: Config<FileSystem>) -> Result<Self> {
        fs::create_dir_all(path.as_ref())?;

        match config.fs.create_lock_file(path.as_ref(), 0o644) {
            Err(Error::AcquiredExisting) => {
                // backup
                recovery::backup_non_segment_files(config.fs)?;
            }
            Err(e) => return Err(e),
            _ => (),
        };

        todo!()
    }
}
