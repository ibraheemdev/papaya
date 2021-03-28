#![allow(dead_code)] // tmp

mod error;
mod fs;
mod recovery;
mod segment;
pub(crate) mod utils;

#[doc(inline)]
pub use error::{Error, Result};

#[doc(inline)]
pub use segment::{Segment, SegmentMeta};

#[doc(inline)]
pub use fs::{AcquiredExisting, File, FileSystem};

use crate::utils::*;

use std::fs as sys;
use std::path::Path;
use std::time::Duration;

pub struct Papaya<FileSystem>
where
    FileSystem: self::FileSystem,
{
    config: Config<FileSystem>,
    index: Index<FileSystem>,
    datalog: DataLog<FileSystem>,
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
        sys::create_dir_all(path.as_ref())?;

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

struct Config<FileSystem>
where
    FileSystem: self::FileSystem,
{
    fs: FileSystem,
    create_new: bool,
    background_sync_interval: Duration,
    background_compaction_interval: Duration,
    max_segment_size: u32,
    compaction_min_segment_size: u32,
    compaction_min_fragmentation: f32,
}

struct Index<FileSystem>
where
    FileSystem: self::FileSystem,
{
    config: Config<FileSystem>,
    main: FileSystem::File,
    overflow: FileSystem::File,
    free_bucket_offs: Vec<i64>,
    level: u8,
    key_count: u32,
    bucket_count: u32,
    bucket_split_idx: u32,
}

struct DataLog<FileSystem>
where
    FileSystem: self::FileSystem,
{
    config: Config<FileSystem>,
    curr: Segment<FileSystem::File>,
    segments: [Segment<FileSystem::File>; MAX_SEGMENTS as _],
    max_sequence_id: u64,
}

struct Stats {
    puts: u64,
    dels: u64,
    gets: u64,
    collisions: u64,
}
