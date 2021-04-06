use crate::{Config, FileSystem};

pub struct Index<FileSystem>
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
