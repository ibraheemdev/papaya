use std::alloc::Layout;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};
use std::{alloc, mem, ptr};

use seize::Collector;

use super::{probe, State};

// A hash-table laid out in a single allocation.
//
// Note that the PhantomData<T> ensures that the hash-table is invariant
// with respect to T, as RawTable is stored behind an AtomicPtr.
#[repr(transparent)]
pub struct RawTable<T>(u8, PhantomData<T>);

// Safety: `seize::Link` is the first field (see `TableLayout`).
unsafe impl<T> seize::AsLink for RawTable<T> {}

// The layout of the table allocation.
#[repr(C)]
struct TableLayout<T> {
    link: seize::Link,
    mask: usize,
    limit: usize,
    capacity: usize,
    state: State<T>,
    meta: [AtomicU8; 0],
    entries: [AtomicPtr<()>; 0],
}

// Manages a table allocation.
#[repr(C)]
pub struct Table<T> {
    // Mask for the table length.
    pub mask: usize,
    // The probe limit.
    pub limit: usize,
    // The raw table pointer.
    pub raw: *mut RawTable<T>,
    // The true (padded) table capacity.
    capacity: usize,
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
        let mask = len - 1;
        let limit = probe::limit(len);

        unsafe {
            let layout = Self::layout(capacity);

            // Allocate the table, zeroing the entries.
            let ptr = alloc::alloc_zeroed(layout);
            if ptr.is_null() {
                alloc::handle_alloc_error(layout);
            }

            // Write the table state.
            ptr.cast::<TableLayout<T>>().write(TableLayout {
                link: collector.link(),
                mask,
                limit,
                capacity,
                state: State {
                    collector,
                    ..State::default()
                },
                meta: [],
                entries: [],
            });

            // Initialize the meta table.
            ptr.add(mem::size_of::<TableLayout<T>>())
                .cast::<u8>()
                .write_bytes(super::meta::EMPTY, capacity);

            Table {
                mask,
                limit,
                capacity,
                raw: ptr.cast::<RawTable<T>>(),
            }
        }
    }

    // Creates a `Table` from a raw pointer.
    #[inline]
    pub unsafe fn from_raw(raw: *mut RawTable<T>) -> Table<T> {
        if raw.is_null() {
            return Table {
                raw,
                mask: 0,
                limit: 0,
                capacity: 0,
            };
        }

        let layout = unsafe { &*raw.cast::<TableLayout<T>>() };

        Table {
            raw,
            mask: layout.mask,
            limit: layout.limit,
            capacity: layout.capacity,
        }
    }

    // Returns the metadata entry at the given index.
    #[inline]
    pub unsafe fn meta(&self, i: usize) -> &AtomicU8 {
        debug_assert!(i < self.capacity);
        &*self
            .raw
            .add(mem::size_of::<TableLayout<T>>())
            .add(i)
            .cast::<AtomicU8>()
    }

    // Returns the entry at the given index.
    #[inline]
    pub unsafe fn entry(&self, i: usize) -> &AtomicPtr<T> {
        debug_assert!(i < self.capacity);

        &*self
            .raw
            .add(mem::size_of::<TableLayout<T>>())
            .add(self.capacity)
            .add(i * mem::size_of::<AtomicPtr<T>>())
            .cast::<AtomicPtr<T>>()
    }

    /// Returns the length of the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.mask + 1
    }

    // Returns a reference to the table state.
    #[inline]
    pub fn state(&self) -> &State<T> {
        unsafe { &(*self.raw.cast::<TableLayout<T>>()).state }
    }

    // Returns a mutable reference to the table state.
    #[inline]
    pub fn state_mut(&mut self) -> &mut State<T> {
        unsafe { &mut (*self.raw.cast::<TableLayout<T>>()).state }
    }

    // Returns a pointer to the next table, if it has already been created.
    #[inline]
    pub fn next_table(&self) -> Option<Self> {
        let next = self.state().next.load(Ordering::Acquire);

        if !next.is_null() {
            return unsafe { Some(Table::from_raw(next)) };
        }

        None
    }

    // Deallocate the table.
    pub unsafe fn dealloc(table: Table<T>) {
        let layout = Self::layout(table.capacity);
        ptr::drop_in_place(table.raw.cast::<TableLayout<T>>());
        unsafe { alloc::dealloc(table.raw.cast::<u8>(), layout) }
    }

    // The table layout used for allocation.
    fn layout(capacity: usize) -> Layout {
        let size = mem::size_of::<TableLayout<T>>()
            + (mem::size_of::<u8>() * capacity) // meta
            + (mem::size_of::<usize>() * capacity); // entries
        Layout::from_size_align(size, mem::align_of::<TableLayout<T>>()).unwrap()
    }
}

#[test]
fn layout() {
    unsafe {
        let collector = seize::Collector::new();
        let table: Table<u8> = Table::alloc(4, &collector);
        let table: Table<u8> = Table::from_raw(table.raw);
        assert_eq!(table.mask, 3);
        assert_eq!(table.len(), 4);
        // The capacity is padded for pointer alignment.
        assert_eq!(table.capacity, 8);
        Table::dealloc(table);
    }
}
