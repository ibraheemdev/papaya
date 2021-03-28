use crate::File;

pub struct Segment<F>
where
    F: File,
{
    file: F,
    id: u16,
    sequence_id: u64,
    name: String,
    meta: SegmentMeta,
}

pub struct SegmentMeta {
    puts: u32,
    is_full: bool,
    deleted_records: u32,
    deleted_keys: u32,
    deleted_bytes: u32,
}
