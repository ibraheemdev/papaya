use std::alloc;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::{self};
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU8, AtomicUsize};
use std::sync::Mutex;

use crate::seize;

// A hash table layed out in a single allocation
#[repr(transparent)]
pub struct RawTable(u8);

// Safety: seize::Link is the first field (see TableLayout)
unsafe impl seize::AsLink for RawTable {}

#[repr(align(16))]
struct AtomicU128(u128);

// The table allocation's layout
#[allow(unused)]
struct TableLayout {
    link: seize::Link,
    len: usize,
    capacity: usize,
    resize_state: ResizeState,
    meta: [AtomicU128; 0],
    entries: [AtomicPtr<()>; 0],
}

#[derive(Default)]
pub struct ResizeState {
    pub next: AtomicPtr<RawTable>,
    pub allocating: Mutex<()>,
    pub copied: AtomicUsize,
    pub claim: AtomicUsize,
    pub futex: AtomicU32,
}

impl ResizeState {
    pub const PENDING: u32 = 0;
    pub const COMPLETE: u32 = 1;
}

// Manages a table allocation.
#[repr(C)]
pub struct Table<T> {
    // the exposed length of the table
    pub len: usize,
    // the raw table pointer
    pub raw: *mut RawTable,
    // the true (padded) table capacity
    capacity: usize,
    _t: PhantomData<T>,
}

impl<T> Copy for Table<T> {}

impl<T> Clone for Table<T> {
    fn clone(&self) -> Self {
        Table {
            capacity: self.capacity,
            len: self.len,
            raw: self.raw,
            _t: PhantomData,
        }
    }
}

impl<T> Table<T> {
    pub fn new(len: usize, mut capacity: usize, link: seize::Link) -> Table<T> {
        assert!(mem::align_of::<seize::Link>() % mem::align_of::<*mut T>() == 0);

        // pad the meta table to allow one meta group of overflow
        capacity += ((capacity - len) + 15) & !15;
        // pad the meta table to fulfill the alignment requirement of an entry
        capacity = (capacity + mem::align_of::<*mut T>() - 1) & !(mem::align_of::<*mut T>() - 1);

        unsafe {
            let layout = Self::layout(capacity);
            let ptr = alloc::alloc(layout);

            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }

            // write the table layout state
            ptr.cast::<TableLayout>().write(TableLayout {
                link,
                len,
                capacity,
                resize_state: ResizeState::default(),
                meta: [],
                entries: [],
            });

            // initialize the meta table
            ptr.add(mem::size_of::<TableLayout>())
                .cast::<u8>()
                .write_bytes(super::meta::EMPTY, capacity);

            // zero the entries table
            let offset = mem::size_of::<TableLayout>() + mem::size_of::<u8>() * capacity;
            ptr.add(offset).cast::<usize>().write_bytes(0, capacity);

            Table {
                len,
                capacity,
                raw: ptr.cast::<RawTable>(),
                _t: PhantomData,
            }
        }
    }

    pub unsafe fn from_raw(raw: *mut RawTable) -> Table<T> {
        let layout = unsafe { &*raw.cast::<TableLayout>() };

        Table {
            raw,
            len: layout.len,
            capacity: layout.capacity,
            _t: PhantomData,
        }
    }

    pub unsafe fn meta(&self, i: usize) -> &AtomicU8 {
        &*self
            .raw
            .add(mem::size_of::<TableLayout>())
            .add(i * mem::size_of::<u8>())
            .cast::<AtomicU8>()
    }

    pub unsafe fn entry(&self, i: usize) -> &AtomicPtr<T> {
        let offset = mem::size_of::<TableLayout>()
            + mem::size_of::<u8>() * self.capacity
            + i * mem::size_of::<AtomicPtr<T>>();

        assert!(i < self.capacity);
        &*self.raw.add(offset).cast::<AtomicPtr<T>>()
    }

    pub fn resize_state(&self) -> &ResizeState {
        unsafe { &(*self.raw.cast::<TableLayout>()).resize_state }
    }

    pub unsafe fn dealloc(table: Table<T>) {
        let layout = Self::layout(table.capacity);
        unsafe { alloc::dealloc(table.raw.cast::<u8>(), layout) }
    }

    fn layout(capacity: usize) -> Layout {
        let size = mem::size_of::<TableLayout>()
            + (mem::size_of::<u8>() * capacity) // meta
            + (mem::size_of::<usize>() * capacity); // entries
        Layout::from_size_align(size, mem::align_of::<TableLayout>()).unwrap()
    }
}

#[test]
fn layout() {
    unsafe {
        let collector = seize::Collector::new();
        let link = collector.link();
        let table: Table<u8> = Table::new(30, 31, link);
        let table: Table<u8> = Table::from_raw(table.raw);
        assert_eq!(table.len, 30);
        assert_eq!(table.capacity, 48);
        Table::dealloc(table);
    }
}
