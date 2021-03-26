use std::fs::File;
use std::time::Duration;

const MAX_KEY_LENGTH: u16 = u16::MAX;
const MAX_VALUE_LENGTH: u32 = 1 << 30 - 1;
const MAX_KEYS: u32 = u32::MAX;
const EXT: &str = ".papaya";
const DB_FILE: &str = "db.papaya";
const MAX_SEGMENTS: usize = u16::MAX as usize;

pub struct Papaya {
    config: Config,
    index: Index,
    datalog: DataLog,
    lock: LockFile,
    seed: u32,
    stats: Stats,
    sync_writes: bool,
    compaction_running: i32,
}

struct Config {
    background_sync_interval: Duration,
    background_compaction_interval: Duration,
    file_system: FileSystem,
    max_segment_size: u32,
    compaction_min_segment_size: u32,
    compaction_min_fragmentation: f32,
}

enum FileSystem {
    OS,
    MMap,
}

enum LockFile {
    OS,
    MMap,
}

struct Index {
    config: Config,
    main: File,
    overflow: File,
    free_bucket_offs: Vec<i64>,
    level: u8,
    key_count: u32,
    bucket_count: u32,
    bucket_split_idx: u32,
}

struct DataLog {
    config: Config,
    curr: Segment,
    segments: [Segment; MAX_SEGMENTS],
    max_sequence_id: u64,
}

struct Segment {
    file: File,
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
