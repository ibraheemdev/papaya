use std::alloc::Layout;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU32, AtomicU8, AtomicUsize};
use std::sync::Mutex;
use std::{alloc, mem, ptr};

// A hash table layed out in a single allocation
#[repr(transparent)]
pub struct RawTable(u8);

// Safety: seize::Link is the first field (see TableLayout)
unsafe impl seize::AsLink for RawTable {}

#[repr(align(16))]
#[allow(dead_code)]
struct AtomicU128(u128);

// The table allocation's layout
#[repr(C)]
struct TableLayout {
    link: seize::Link,
    len: usize,
    capacity: usize,
    state: State,
    meta: [AtomicU128; 0],
    entries: [AtomicPtr<()>; 0],
}

#[derive(Default)]
pub struct State {
    pub next: AtomicPtr<RawTable>,
    pub allocating: Mutex<()>,
    pub copied: AtomicUsize,
    pub claim: AtomicUsize,
    pub status: AtomicU32,
    // todo: use seize linked lists here
    pub deferred: Mutex<Vec<*mut ()>>,
}

impl State {
    pub const PENDING: u32 = 0;
    pub const ABORTED: u32 = 1;
    pub const PROMOTED: u32 = 2;
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
        *self
    }
}

impl<T> Table<T> {
    pub fn new(len: usize, link: seize::Link) -> Table<T> {
        assert!(len.is_power_of_two());
        assert!(mem::align_of::<seize::Link>() % mem::align_of::<*mut T>() == 0);

        // pad the meta table to fulfill the alignment requirement of an entry
        let capacity = (len + mem::align_of::<*mut T>() - 1) & !(mem::align_of::<*mut T>() - 1);

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
                state: State::default(),
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

    #[inline(always)]
    pub unsafe fn from_raw(raw: *mut RawTable) -> Table<T> {
        if raw.is_null() {
            return Table {
                raw,
                len: 0,
                capacity: 0,
                _t: PhantomData,
            };
        }

        let layout = unsafe { &*raw.cast::<TableLayout>() };

        Table {
            raw,
            len: layout.len,
            capacity: layout.capacity,
            _t: PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn meta(&self, i: usize) -> &AtomicU8 {
        debug_assert!(i < self.capacity);
        &*self
            .raw
            .add(mem::size_of::<TableLayout>())
            .add(i * mem::size_of::<u8>())
            .cast::<AtomicU8>()
    }

    #[inline(always)]
    pub unsafe fn entry(&self, i: usize) -> &AtomicPtr<T> {
        let offset = mem::size_of::<TableLayout>()
            + mem::size_of::<u8>() * self.capacity
            + i * mem::size_of::<AtomicPtr<T>>();

        debug_assert!(i < self.capacity);
        &*self.raw.add(offset).cast::<AtomicPtr<T>>()
    }

    pub fn state(&self) -> &State {
        unsafe { &(*self.raw.cast::<TableLayout>()).state }
    }

    pub fn state_mut(&mut self) -> &mut State {
        unsafe { &mut (*self.raw.cast::<TableLayout>()).state }
    }

    pub unsafe fn dealloc(table: Table<T>) {
        let layout = Self::layout(table.capacity);
        ptr::drop_in_place(table.raw.cast::<TableLayout>());
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
        let table: Table<u8> = Table::new(4, link);
        let table: Table<u8> = Table::from_raw(table.raw);
        assert_eq!(table.len, 4);
        // padded for pointer alignment
        assert_eq!(table.capacity, 8);
        Table::dealloc(table);
    }
}
