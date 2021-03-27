#![allow(dead_code)] // tmp

mod fs;

#[doc(inline)]
pub use fs::{AcquiredExisting, File, FileSystem, LockFile};

use std::io;
use std::path::PathBuf;
use std::time::Duration;

const MAX_KEY_LENGTH: u16 = u16::MAX;
const MAX_VALUE_LENGTH: u32 = 1 << 30 - 1;
const MAX_KEYS: u32 = u32::MAX;
const EXT: &str = ".papaya";
const DB_FILE: &str = "db.papaya";
const MAX_SEGMENTS: usize = u16::MAX as usize;
const LOCK_NAME: &str = "lock";

pub type Result<T> = std::result::Result<T, Error>;

pub enum Error {
    Io(io::Error),
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

pub struct Papaya<FileSystem>
where
    FileSystem: self::FileSystem,
{
    config: Config<FileSystem>,
    index: Index<FileSystem>,
    datalog: DataLog<FileSystem>,
    lock: FileSystem::LockFile,
    seed: u32,
    stats: Stats,
    sync_writes: bool,
    compaction_running: i32,
}

struct Config<FileSystem>
where
    FileSystem: self::FileSystem,
{
    path: PathBuf,
    file_system: FileSystem,
    create_new: bool,
    background_sync_interval: Duration,
    background_compaction_interval: Duration,
    max_segment_size: u32,
    compaction_min_segment_size: u32,
    compaction_min_fragmentation: f32,
}

impl<FileSystem> Config<FileSystem>
where
    FileSystem: self::FileSystem,
{
    fn open(self) -> Result<Papaya<FileSystem>> {
        // let mut builder = fs::DirBuilder::new();
        // builder.recursive(true);

        // #[cfg(unix)]
        // std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o755);

        // let mut acquired_existing = false;

        // if fs::metadata(&self.path).is_err() {
        //     acquired_existing = true;
        // }

        // let file = fs::File::open(&self.path)?;

        // let try_lock = fs2::FileExt::try_lock_exclusive(&file).map_err(|e| {
        //     io::Error::new(
        //         io::ErrorKind::Other,
        //         format!(
        //             "Could not acquire lock on {}: {}",
        //             self.path.to_string_lossy(),
        //             e
        //         ),
        //     )
        // })?;

        todo!()
    }
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
    segments: [Segment<FileSystem::File>; MAX_SEGMENTS],
    max_sequence_id: u64,
}

struct Segment<F>
where
    F: File,
{
    file: F,
    id: u16,
    sequence_id: u64,
    name: String,
    meta: SegmentMeta,
}

struct SegmentMeta {
    puts: u32,
    is_full: bool,
    deleted_records: u32,
    deleted_keys: u32,
    deleted_bytes: u32,
}

struct Stats {
    puts: u64,
    dels: u64,
    gets: u64,
    collisions: u64,
}
