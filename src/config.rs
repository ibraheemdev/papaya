use crate::FileSystem;

use std::time::Duration;

pub struct Config<FileSystem>
where
    FileSystem: self::FileSystem,
{
    pub(crate) fs: FileSystem,
    create_new: bool,
    background_sync_interval: Duration,
    background_compaction_interval: Duration,
    max_segment_size: u32,
    compaction_min_segment_size: u32,
    compaction_min_fragmentation: f32,
}
