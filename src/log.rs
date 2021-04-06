use crate::utils::MAX_SEGMENTS;
use crate::{Config, FileSystem, Segment};

pub struct Log<FileSystem>
where
    FileSystem: self::FileSystem,
{
    config: Config<FileSystem>,
    curr: Segment<FileSystem::File>,
    segments: [Segment<FileSystem::File>; MAX_SEGMENTS as _],
    max_sequence_id: u64,
}
