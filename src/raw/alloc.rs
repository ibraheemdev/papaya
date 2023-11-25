use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::{self, MaybeUninit};
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU8, AtomicUsize};
use std::sync::Mutex;
use std::{alloc, slice};

use crate::seize;

// A hash table layed out in a single allocation
#[repr(transparent)]
pub struct RawTable(u8);

// Safety: seize::Link is the first field
unsafe impl seize::AsLink for RawTable {}

// The table allocation's layout
struct TableLayout {
    link: seize::Link,
    capacity: usize,
    meta: [AtomicU8; 0],
    entries: [AtomicPtr<()>; 0],
}

pub struct ResizeState {
    pub next: AtomicPtr<RawTable>,
    pub allocating: Mutex<()>,
    pub copied: AtomicUsize,
    pub claim: AtomicUsize,
}

impl ResizeState {
    pub const PENDING: u32 = 0;
    pub const COMPLETE: u32 = 1;
}

// Manages a RawTable
#[repr(C)]
pub struct Table<T> {
    pub capacity: usize,
    pub raw: *mut RawTable,
    _t: PhantomData<T>,
}

impl<T> Copy for Table<T> {}

impl<T> Clone for Table<T> {
    fn clone(&self) -> Self {
        Table {
            capacity: self.capacity,
            raw: self.raw,
            _t: PhantomData,
        }
    }
}

impl<T> Table<T> {
    pub fn new(capacity: usize, link: seize::Link) -> Table<T> {
        unsafe {
            let layout = Self::layout(capacity);
            let ptr = alloc::alloc_zeroed(layout);

            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }

            ptr.cast::<seize::Link>().write(link);

            ptr.add(mem::size_of::<seize::Link>())
                .cast::<usize>()
                .write(capacity);

            ptr.add(Self::META_OFFSET)
                .cast::<u8>()
                .write_bytes(super::meta::EMPTY, capacity);

            let entry_offset = Self::META_OFFSET + (mem::size_of::<u8>() * capacity);
            ptr.add(entry_offset)
                .cast::<usize>()
                .write_bytes(0, capacity);

            Table {
                capacity,
                raw: ptr.cast::<RawTable>(),
                _t: PhantomData,
            }
        }
    }

    pub unsafe fn from_raw(raw: *mut RawTable) -> Table<T> {
        let capacity = unsafe {
            raw.add(mem::size_of::<seize::Link>())
                .cast::<usize>()
                .read()
        };

        Table {
            raw,
            capacity,
            _t: PhantomData,
        }
    }

    pub unsafe fn meta(&self, i: usize) -> &AtomicU8 {
        &*self
            .raw
            .add(Self::META_OFFSET)
            .add(i * mem::size_of::<u8>())
            .cast::<AtomicU8>()
    }

    pub unsafe fn entry(&self, i: usize) -> &AtomicPtr<T> {
        let offset = Self::META_OFFSET + (mem::size_of::<u8>() * self.capacity);

        &*self
            .raw
            .add(offset)
            .add(i * mem::size_of::<AtomicPtr<T>>())
            .cast::<AtomicPtr<T>>()
    }

    pub fn resize_state(&self) -> &ResizeState {
        let offset = Self::META_OFFSET
            + (mem::size_of::<u8>() * self.capacity)
            + (mem::size_of::<AtomicPtr<T>>() * self.capacity);

        unsafe { &*self.raw.add(offset).cast::<ResizeState>() }
    }

    pub unsafe fn dealloc(table: *mut Table<T>) {
        let layout = Self::layout((*table).capacity);
        unsafe { alloc::dealloc((*table).raw.cast::<u8>(), layout) }
    }

    const META_OFFSET: usize = mem::size_of::<seize::Link>() + mem::size_of::<usize>();

    fn layout(capacity: usize) -> Layout {
        let size = mem::size_of::<TableLayout>()
            + (mem::size_of::<u8>() * capacity) // meta
            + (mem::size_of::<usize>() * capacity); // entries
        let align = mem::align_of::<TableLayout>();
        Layout::from_size_align(size, align).unwrap()
    }
}

#[test]
fn layout() {
    unsafe {
        let collector = seize::Collector::new();
        let link = collector.link();
        let table: Table<u8> = Table::new(32, link);
        let table: Table<u8> = Table::from_raw(table.raw);
        assert_eq!(table.capacity, 32);
        dbg!(table.entry(0));
        dbg!(table.meta(0));
    }
}
