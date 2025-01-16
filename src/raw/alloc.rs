use std::alloc::Layout;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicPtr, AtomicU8, Ordering};
use std::{alloc, mem, ptr};

use seize::Collector;

use super::{probe, State};

// A hash-table laid out in a single allocation.
//
// Note that the `PhantomData<T>` ensures that the hash-table is invariant
// with respect to `T`, as this struct is stored behind an `AtomicPtr`.
#[repr(transparent)]
pub struct RawTable<T>(u8, PhantomData<T>);

// Safety: `seize::Link` is the first field (see `TableLayout`).
unsafe impl<T> seize::AsLink for RawTable<T> {}

// The layout of the table allocation.
#[repr(C)]
struct TableLayout<T> {
    /// A link to the `seize::Collector`, enabling garbage collection
    /// when the table resizes.
    link: seize::Link,

    /// A mask to get an index into the table from a hash.
    mask: usize,

    /// The maximum probe limit for this table.
    limit: usize,

    /// State for the table resize.
    state: State<T>,

    /// An array of metadata for each entry.
    meta: [AtomicU8; 0],

    /// An array of entries.
    entries: [AtomicPtr<T>; 0],
}

// Manages a table allocation.
#[repr(C)]
pub struct Table<T> {
    /// A mask to get an index into the table from a hash.
    pub mask: usize,

    /// The maximum probe limit for this table.
    pub limit: usize,

    // The raw table allocation.
    //
    // Invariant: This pointer is initialized and valid for reads and writes.
    pub raw: *mut RawTable<T>,
}

impl<T> Copy for Table<T> {}

impl<T> Clone for Table<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Table<T> {
    // Allocate a table with the provided length and collector.
    pub fn alloc(len: usize, collector: &Collector) -> Table<T> {
        assert!(len.is_power_of_two());

        // Pad the meta table to fulfill the alignment requirement of an entry.
        let len = len.max(mem::align_of::<AtomicPtr<T>>());
        let mask = len - 1;
        let limit = probe::limit(len);

        let layout = Table::<T>::layout(len);

        // Allocate the table, zeroing the entries.
        //
        // Safety: The layout for is guaranteed to be non-zero.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            alloc::handle_alloc_error(layout);
        }

        // Safety: We just allocated the pointer and ensured it is non-null above.
        unsafe {
            // Write the table state.
            ptr.cast::<TableLayout<T>>().write(TableLayout {
                link: collector.link(),
                mask,
                limit,
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
                .write_bytes(super::meta::EMPTY, len);
        }

        Table {
            mask,
            limit,
            // Invariant: We allocated and initialized the allocation above.
            raw: ptr.cast::<RawTable<T>>(),
        }
    }

    // Creates a `Table` from a raw pointer.
    //
    // # Safety
    //
    // The pointer must either be null, or a valid pointer created with `Table::alloc`.
    #[inline]
    pub unsafe fn from_raw(raw: *mut RawTable<T>) -> Table<T> {
        if raw.is_null() {
            return Table {
                raw,
                mask: 0,
                limit: 0,
            };
        }

        // Safety: The caller guarantees that the pointer is valid.
        let layout = unsafe { &*raw.cast::<TableLayout<T>>() };

        Table {
            raw,
            mask: layout.mask,
            limit: layout.limit,
        }
    }

    // Returns the metadata entry at the given index.
    //
    // # Safety
    //
    // The index must be in-bounds for the length of the table.
    #[inline]
    pub unsafe fn meta(&self, i: usize) -> &AtomicU8 {
        debug_assert!(i < self.len());

        // Safety: The caller guarantees the index is in-bounds.
        unsafe {
            let meta = self.raw.add(mem::size_of::<TableLayout<T>>());
            &*meta.cast::<AtomicU8>().add(i)
        }
    }

    // Returns the entry at the given index.
    //
    // # Safety
    //
    // The index must be in-bounds for the length of the table.
    #[inline]
    pub unsafe fn entry(&self, i: usize) -> &AtomicPtr<T> {
        debug_assert!(i < self.len());

        // Safety: The caller guarantees the index is in-bounds.
        unsafe {
            let meta = self.raw.add(mem::size_of::<TableLayout<T>>());
            let entries = meta.add(self.len()).cast::<AtomicPtr<T>>();
            &*entries.add(i)
        }
    }

    /// Returns the length of the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.mask + 1
    }

    // Returns a reference to the table state.
    #[inline]
    pub fn state(&self) -> &State<T> {
        // Safety: The raw table pointer is always valid for reads and writes.
        unsafe { &(*self.raw.cast::<TableLayout<T>>()).state }
    }

    // Returns a mutable reference to the table state.
    #[inline]
    pub fn state_mut(&mut self) -> &mut State<T> {
        // Safety: The raw table pointer is always valid for reads and writes.
        unsafe { &mut (*self.raw.cast::<TableLayout<T>>()).state }
    }

    // Returns a pointer to the next table, if it has already been created.
    #[inline]
    pub fn next_table(&self) -> Option<Self> {
        let next = self.state().next.load(Ordering::Acquire);

        if !next.is_null() {
            // Safety: We verified that the pointer is non-null, and the
            // next pointer is otherwise a valid pointer to a table allocation.
            return unsafe { Some(Table::from_raw(next)) };
        }

        None
    }

    // Deallocate the table.
    //
    // # Safety
    //
    // The table may not be accessed in any way after this method is
    // called.
    pub unsafe fn dealloc(table: Table<T>) {
        let layout = Self::layout(table.len());

        // Safety: The raw table pointer is valid and allocated with `alloc::alloc_zeroed`.
        // Additionally, the caller guarantees that the allocation will not be accessed after
        // this point.
        unsafe {
            ptr::drop_in_place(table.raw.cast::<TableLayout<T>>());
            alloc::dealloc(table.raw.cast::<u8>(), layout);
        };
    }

    // Returns the non-zero layout for a table allocation.
    fn layout(len: usize) -> Layout {
        let size = mem::size_of::<TableLayout<T>>()
            + (mem::size_of::<u8>() * len) // Metadata table.
            + (mem::size_of::<AtomicPtr<T>>() * len); // Entry pointers.
                                                      //
        Layout::from_size_align(size, mem::align_of::<TableLayout<T>>()).unwrap()
    }
}

#[test]
fn layout() {
    unsafe {
        let collector = seize::Collector::new();
        let table: Table<u8> = Table::alloc(4, &collector);
        let table: Table<u8> = Table::from_raw(table.raw);

        // The capacity is padded for pointer alignment.
        assert_eq!(table.mask, 7);
        assert_eq!(table.len(), 8);
        Table::dealloc(table);
    }
}
