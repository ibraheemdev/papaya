use std::alloc::Layout;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU8};
use std::{alloc, mem, ptr};

use seize::Collector;

use super::State;

// A hash-table laid out in a single allocation.
#[repr(transparent)]
pub struct RawTable(u8);

// Safety: `seize::Link` is the first field (see `TableLayout`).
unsafe impl seize::AsLink for RawTable {}

// The layout of the table allocation.
#[repr(C)]
struct TableLayout {
    link: seize::Link,
    len: usize,
    capacity: usize,
    state: State,
    meta: [AtomicU8; 0],
    entries: [AtomicPtr<()>; 0],
}

// Manages a table allocation.
#[repr(C)]
pub struct Table<T> {
    // The exposed length of the table.
    pub len: usize,
    // The raw table pointer.
    pub raw: *mut RawTable,
    // The true (padded) table capacity.
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
    // Allocate a table with the provided length.
    pub fn alloc(len: usize, collector: &Collector) -> Table<T> {
        assert!(len.is_power_of_two());
        assert!(mem::align_of::<seize::Link>() % mem::align_of::<*mut T>() == 0);

        // Pad the meta table to fulfill the alignment requirement of an entry.
        let capacity = (len + mem::align_of::<*mut T>() - 1) & !(mem::align_of::<*mut T>() - 1);

        unsafe {
            let layout = Self::layout(capacity);

            // Allocate the table, zeroing the entries.
            let ptr = alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }

            // Write the table state.
            ptr.cast::<TableLayout>().write(TableLayout {
                link: collector.link(),
                len,
                capacity,
                state: State {
                    collector,
                    ..State::default()
                },
                meta: [],
                entries: [],
            });

            // Initialize the meta table.
            ptr.add(mem::size_of::<TableLayout>())
                .cast::<u8>()
                .write_bytes(super::meta::EMPTY, capacity);

            Table {
                len,
                capacity,
                raw: ptr.cast::<RawTable>(),
                _t: PhantomData,
            }
        }
    }

    // Creates a `Table` from a raw pointer.
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

    // Returns the metadata entry at the given index.
    #[inline(always)]
    pub unsafe fn meta(&self, i: usize) -> &AtomicU8 {
        debug_assert!(i < self.capacity);
        &*self
            .raw
            .add(mem::size_of::<TableLayout>())
            .add(i * mem::size_of::<u8>())
            .cast::<AtomicU8>()
    }

    // Returns the entry at the given index.
    #[inline(always)]
    pub unsafe fn entry(&self, i: usize) -> &AtomicPtr<T> {
        let offset = mem::size_of::<TableLayout>()
            + mem::size_of::<u8>() * self.capacity
            + i * mem::size_of::<AtomicPtr<T>>();

        debug_assert!(i < self.capacity);
        &*self.raw.add(offset).cast::<AtomicPtr<T>>()
    }

    // Returns a reference to the table state.
    pub fn state(&self) -> &State {
        unsafe { &(*self.raw.cast::<TableLayout>()).state }
    }

    // Returns a mutable reference to the table state.
    pub fn state_mut(&mut self) -> &mut State {
        unsafe { &mut (*self.raw.cast::<TableLayout>()).state }
    }

    // Deallocate the table.
    pub unsafe fn dealloc(table: Table<T>) {
        let layout = Self::layout(table.capacity);
        ptr::drop_in_place(table.raw.cast::<TableLayout>());
        unsafe { alloc::dealloc(table.raw.cast::<u8>(), layout) }
    }

    // The table layout used for allocation.
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
        let table: Table<u8> = Table::alloc(4, &collector);
        let table: Table<u8> = Table::from_raw(table.raw);
        assert_eq!(table.len, 4);
        // The capacity is padded for pointer alignment.
        assert_eq!(table.capacity, 8);
        Table::dealloc(table);
    }
}
