use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::{hint, panic, ptr};

use super::alloc::{RawTable, Table};
use super::probe::{self, Probe};
use super::utils::{
    untagged, AtomicPtrFetchOps, Counter, MapGuard, Parker, Stack, StrictProvenance, Tagged,
    Unpack, VerifiedGuard,
};
use crate::map::ResizeMode;

use seize::{Collector, LocalGuard, OwnedGuard};

/// A lock-free hash-table.
pub struct HashTable<T, D: Dealloc<T>> {
    /// A pointer to the root table.
    table: AtomicPtr<RawTable<T>>,

    /// Collector for memory reclamation.
    pub(crate) collector: Collector,

    /// The resize mode, either blocking or incremental.
    resize: ResizeMode,

    /// An atomic counter of the number of keys in the table.
    count: Counter,

    /// The initial capacity provided to `HashMap::new`.
    ///
    /// The table is guaranteed to never shrink below this capacity.
    initial_capacity: usize,

    _dealloc: PhantomData<D>,
}

pub trait Dealloc<T> {
    unsafe fn dealloc(entry: *mut T);
}

/// Resize state for the hash-table.
pub struct State<T> {
    /// The next table used for resizing.
    pub next: AtomicPtr<RawTable<T>>,

    /// A lock acquired to allocate the next table.
    pub allocating: Mutex<()>,

    /// The number of entries that have been copied to the next table.
    pub copied: AtomicUsize,

    /// The number of entries that have been claimed by copiers,
    /// but not necessarily copied.
    pub claim: AtomicUsize,

    /// The status of the resize.
    pub status: AtomicU8,

    /// A thread parker for blocking on copy operations.
    pub parker: Parker,

    /// Entries whose retirement has been deferred by later tables.
    pub deferred: Stack<*mut T>,
}

impl<T> Default for State<T> {
    fn default() -> State<T> {
        State {
            next: AtomicPtr::new(ptr::null_mut()),
            allocating: Mutex::new(()),
            copied: AtomicUsize::new(0),
            claim: AtomicUsize::new(0),
            status: AtomicU8::new(State::PENDING),
            parker: Parker::default(),
            deferred: Stack::new(),
        }
    }
}

impl State<()> {
    /// A resize is in-progress.
    pub const PENDING: u8 = 0;

    /// The resize has been aborted, continue to the next table.
    pub const ABORTED: u8 = 1;

    /// The resize was complete and the table was promoted.
    pub const PROMOTED: u8 = 2;
}

// The raw result of an insert operation.
pub enum InsertResult<T> {
    /// Inserted the entry.
    Inserted,

    /// Replaced the given entry.
    Replaced(*mut T),

    /// Did not insert due to the given existing entry.
    Error(*mut T),
}

// An entry in the hash-table.
pub struct Entry;

impl Entry {
    /// The entry is being copied to the new table, no updates are allowed on the old table.
    ///
    /// This bit is put down to initiate a copy, forcing all writers to complete the resize
    /// before making progress.
    const COPYING: usize = 0b001;

    /// The entry has been copied to the new table.
    ///
    /// This bit is put down after a copy completes. Both readers and writers must go to
    /// the new table to see the new state of the entry.
    ///
    /// In blocking mode this is unused.
    const COPIED: usize = 0b010;

    /// The entry was copied from a previous table.
    ///
    /// This bit indicates that an entry may still be accessible from previous tables
    /// because the resize is still in progress, and so it is unsafe to reclaim.
    ///
    /// In blocking mode this is unused.
    const BORROWED: usize = 0b100;
}

impl Unpack for Entry {
    /// Mask for an entry pointer, ignoring any tag bits.
    const MASK: usize = !(Entry::COPYING | Entry::COPIED | Entry::BORROWED);
}

impl Entry {
    /// A sentinel pointer for a deleted entry.
    ///
    /// Null pointers are never copied to the new table, so this state is safe to use.
    /// Note that tombstone entries may still be marked as `COPYING`, so this state
    /// cannot be used for direct equality.
    const TOMBSTONE: *mut () = Entry::COPIED as _;

    unsafe fn reclaim<T, D: Dealloc<T>>(entry: *mut T, _collector: &Collector) {
        unsafe { D::dealloc(entry) };
    }
}

/// The status of an entry.
enum EntryStatus<T> {
    /// The entry is a tombstone or null (potentially a null copy).
    Null,

    /// The entry is being copied.
    Copied(Tagged<T, Entry>),

    /// A valid entry.
    Value(Tagged<T, Entry>),
}

impl<T> From<Tagged<T, Entry>> for EntryStatus<T> {
    /// Returns the status for this entry.
    #[inline]
    fn from(entry: Tagged<T, Entry>) -> Self {
        if entry.ptr.is_null() {
            EntryStatus::Null
        } else if entry.tag() & Entry::COPYING != 0 {
            EntryStatus::Copied(entry)
        } else {
            EntryStatus::Value(entry)
        }
    }
}

/// The state of an entry we attempted to update.
enum UpdateStatus<T> {
    /// Successfully replaced the given key and value.
    Replaced(Tagged<T, Entry>),

    /// A new entry was written before we could update.
    Found(EntryStatus<T>),
}

/// The state of an entry we attempted to insert into.
enum InsertStatus<T> {
    /// Successfully inserted the value.
    Inserted,

    /// A new entry was written before we could update.
    Found(EntryStatus<T>),
}

impl<T, D: Dealloc<T>> HashTable<T, D> {
    /// Creates new hash-table with the given options.
    #[inline]
    pub fn new(capacity: usize, collector: Collector, resize: ResizeMode) -> HashTable<T, D> {
        // The table is lazily allocated.
        if capacity == 0 {
            return HashTable {
                collector,
                resize,
                initial_capacity: 1,
                table: AtomicPtr::new(ptr::null_mut()),
                count: Counter::default(),
                _dealloc: PhantomData,
            };
        }

        // Initialize the table and mark it as the root.
        let mut table = Table::alloc(probe::entries_for(capacity));
        *table.state_mut().status.get_mut() = State::PROMOTED;

        HashTable {
            resize,
            collector,
            initial_capacity: capacity,
            table: AtomicPtr::new(table.raw),
            count: Counter::default(),
            _dealloc: PhantomData,
        }
    }

    /// Returns a guard for this collector
    pub fn guard(&self) -> MapGuard<LocalGuard<'_>> {
        // Safety: Created the guard from our collector.
        unsafe { MapGuard::new(self.collector().enter()) }
    }

    /// Returns an owned guard for this collector
    pub fn owned_guard(&self) -> MapGuard<OwnedGuard<'_>> {
        // Safety: Created the guard from our collector.
        unsafe { MapGuard::new(self.collector().enter_owned()) }
    }

    /// Verify a guard is valid to use with this map.
    #[inline]
    pub fn verify<'g, G>(&self, guard: &'g G) -> &'g MapGuard<G>
    where
        G: seize::Guard,
    {
        assert_eq!(
            *guard.collector(),
            self.collector,
            "Attempted to access map with incorrect guard"
        );

        // Safety: Verified the guard above.
        unsafe { MapGuard::from_ref(guard) }
    }

    /// Returns a reference to the root hash-table.
    #[inline]
    fn root(&self, guard: &impl VerifiedGuard) -> Table<T> {
        // Load the root table.
        let raw = guard.protect(&self.table, Ordering::Acquire);

        // Safety: The root table is either null or a valid table allocation.
        unsafe { Table::from_raw(raw) }
    }

    /// Returns a reference to the collector.
    #[inline]
    pub fn collector(&self) -> &Collector {
        &self.collector
    }

    /// Returns the number of entries in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.count.sum()
    }

    /// Returns `true` if the table is empty. Otherwise returns `false`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns true if incremental resizing is enabled.
    #[inline]
    fn is_incremental(&self) -> bool {
        matches!(self.resize, ResizeMode::Incremental(_))
    }
}

impl<T, D: Dealloc<T>> HashTable<T, D> {
    /// Returns a reference to the entry corresponding to the key.
    #[inline]
    pub fn find(
        &self,
        hash: u64,
        mut eq: impl FnMut(*mut T) -> bool,
        guard: &impl VerifiedGuard,
    ) -> Option<*mut T> {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not been initialized yet.
        if table.raw.is_null() {
            return None;
        }

        let (h1, h2) = meta::split(hash);

        loop {
            // Initialize the probe state.
            let mut probe = Probe::start(h1, table.mask);

            // Probe until we reach the limit.
            'probe: while probe.len <= table.limit {
                // Load the entry metadata first for cheap searches.
                //
                // Safety: `probe.i` is always in-bounds for the table length.
                let meta = unsafe { table.meta(probe.i) }.load(Ordering::Acquire);

                if meta == h2 {
                    // Load the full entry.
                    //
                    // Safety: `probe.i` is always in-bounds for the table length.
                    let entry = guard
                        .protect(unsafe { table.entry(probe.i) }, Ordering::Acquire)
                        .unpack::<Entry>();

                    // The entry was deleted, keep probing.
                    if entry.ptr.is_null() {
                        probe.next(table.mask);
                        continue 'probe;
                    }

                    // Check for a full match.
                    //
                    // Safety: We performed a protected load of the pointer using a verified guard with
                    // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                    // as we hold the guard.
                    if eq(entry.ptr) {
                        // The entry was copied to the new table.
                        //
                        // In blocking resize mode we do not need to perform self check as all writes block
                        // until any resizes are complete, making the root table the source of truth for readers.
                        if entry.tag() & Entry::COPIED != 0 {
                            break 'probe;
                        }

                        return Some(entry.ptr);
                    }
                }

                // The key is not in the table.
                //
                // It also cannot be in the next table because we have not went over the probe limit.
                if meta == meta::EMPTY {
                    return None;
                }

                probe.next(table.mask);
            }

            // In incremental resize mode, we have to check the next table if we found
            // a copied entry or went over the probe limit.
            if self.is_incremental() {
                if let Some(next) = table.next_table() {
                    table = next;
                    continue;
                }
            }

            // Otherwise, the key is not in the table.
            return None;
        }
    }

    /// Inserts an entry into the map.
    #[inline]
    pub fn insert(
        &self,
        hash: u64,
        entry: *mut T,
        mut eq: impl FnMut(*mut T) -> bool,
        hasher: impl Fn(*mut T) -> u64,
        should_replace: bool,
        guard: &impl VerifiedGuard,
    ) -> InsertResult<T> {
        // Allocate the entry to be inserted.
        let new_entry = untagged::<_, Entry>(entry);

        // Load the root table.
        let mut table = self.root(guard);

        // Allocate the table if it has not been initialized yet.
        if table.raw.is_null() {
            table = self.init(None);
        }

        let (h1, h2) = meta::split(hash);

        let mut help_copy = true;
        loop {
            // Initialize the probe state.
            let mut probe = Probe::start(h1, table.mask);

            // Probe until we reach the limit.
            let copying = 'probe: loop {
                if probe.len > table.limit {
                    break None;
                }

                // Load the entry metadata first for cheap searches.
                //
                // Safety: `probe.i` is always in-bounds for the table length.
                let meta = unsafe { table.meta(probe.i) }.load(Ordering::Acquire);

                // The entry is empty, try to insert.
                let entry = if meta == meta::EMPTY {
                    // Perform the insertion.
                    //
                    // Safety: `probe.i` is always in-bounds for the table length. Additionally,
                    // `new_entry` was allocated above and never shared.
                    match unsafe {
                        self.insert_at(probe.i, h2, new_entry.raw, table, &hasher, guard)
                    } {
                        // Successfully inserted.
                        InsertStatus::Inserted => {
                            // Increment the table length.
                            self.count.get(guard).fetch_add(1, Ordering::Relaxed);
                            return InsertResult::Inserted;
                        }

                        // Lost to a concurrent insert.
                        //
                        // If the key matches, we might be able to update the value.
                        InsertStatus::Found(EntryStatus::Value(found))
                        | InsertStatus::Found(EntryStatus::Copied(found)) => found,

                        // Otherwise, continue probing.
                        InsertStatus::Found(EntryStatus::Null) => {
                            probe.next(table.mask);
                            continue 'probe;
                        }
                    }
                }
                // Found a potential match.
                else if meta == h2 {
                    // Load the full entry.
                    //
                    // Safety: `probe.i` is always in-bounds for the table length.
                    let entry = guard
                        .protect(unsafe { table.entry(probe.i) }, Ordering::Acquire)
                        .unpack();

                    // The entry was deleted, keep probing.
                    if entry.ptr.is_null() {
                        probe.next(table.mask);
                        continue 'probe;
                    }

                    // If the key matches, we might be able to update the value.
                    entry
                }
                // Otherwise, continue probing.
                else {
                    probe.next(table.mask);
                    continue 'probe;
                };

                // Check for a full match.
                //
                // Safety: We performed a protected load of the pointer using a verified guard with
                // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                // as we hold the guard.
                if !eq(entry.ptr) {
                    probe.next(table.mask);
                    continue 'probe;
                }

                // The entry is being copied to the new table.
                if entry.tag() & Entry::COPYING != 0 {
                    break 'probe Some(probe.i);
                }

                // Return an error for calls to `try_insert`.
                if !should_replace {
                    return InsertResult::Error(entry.ptr);
                }

                // Try to update the value.
                //
                // Safety:
                // - `probe.i` is always in-bounds for the table length
                // - `entry` is a valid non-null entry that was inserted into the map.
                match unsafe { self.insert_slow(probe.i, entry, new_entry.raw, table, guard) } {
                    // Successfully performed the update.
                    UpdateStatus::Replaced(entry) => {
                        // Safety: `entry` is a valid non-null entry that we found in the map
                        // before replacing it.
                        return InsertResult::Replaced(entry.ptr);
                    }

                    // The entry is being copied.
                    UpdateStatus::Found(EntryStatus::Copied(_)) => break 'probe Some(probe.i),

                    // The entry was deleted before we could update it, continue probing.
                    UpdateStatus::Found(EntryStatus::Null) => {
                        probe.next(table.mask);
                        continue 'probe;
                    }

                    UpdateStatus::Found(EntryStatus::Value(_)) => {}
                }
            };

            // Prepare to retry in the next table.
            table = self.prepare_retry_insert(copying, &mut help_copy, table, &hasher, guard);
        }
    }

    /// The slow-path for `insert`, updating the value.
    ///
    /// The returned pointer is guaranteed to be non-null and valid for reads.
    ///
    /// # Safety
    ///
    /// The safety requirements of `HashMap::update_at` apply.
    #[cold]
    #[inline(never)]
    unsafe fn insert_slow(
        &self,
        i: usize,
        mut entry: Tagged<T, Entry>,
        new_entry: *mut T,
        table: Table<T>,
        guard: &impl VerifiedGuard,
    ) -> UpdateStatus<T> {
        loop {
            // Try to update the value.
            //
            // Safety: Guaranteed by caller.
            match unsafe { self.update_at(i, entry, new_entry, table, guard) } {
                // Someone else beat us to the update, retry.
                //
                // Note that the pointer we find here is a non-null entry that was inserted
                // into the map.
                UpdateStatus::Found(EntryStatus::Value(found)) => entry = found,

                status => return status,
            }
        }
    }

    /// Prepare to retry an insert operation in the next table.
    #[cold]
    #[inline(never)]
    fn prepare_retry_insert(
        &self,
        copying: Option<usize>,
        help_copy: &mut bool,
        table: Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Table<T> {
        // If went over the probe limit or found a copied entry, trigger a resize.
        let mut next_table = self.get_or_alloc_next(None, table);

        let next_table = match self.resize {
            // In blocking mode we must complete the resize before proceeding.
            ResizeMode::Blocking => self.help_copy(true, &table, hasher, guard),

            // In incremental mode we can perform more granular blocking.
            ResizeMode::Incremental(_) => {
                // Help out with the copy.
                if *help_copy {
                    next_table = self.help_copy(false, &table, hasher, guard);
                }

                // The entry we want to update is being copied.
                if let Some(i) = copying {
                    // Wait for the entry to be copied.
                    //
                    // We could race with the copy to insert into the table. However,
                    // this entire code path is very rare and likely to complete quickly,
                    // so blocking allows us to make copies faster.
                    self.wait_copied(i, &table);
                }

                next_table
            }
        };

        // Limit incremental copying to once per operation, for more consistent latency.
        *help_copy = false;

        // Continue in the new table.
        next_table
    }

    /// Removes a key from the map, returning the entry for the key if the key was previously in the map.
    #[inline]
    pub fn remove(
        &self,
        hash: u64,
        eq: impl FnMut(*mut T) -> bool,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Option<*mut T> {
        #[inline(always)]
        fn should_remove<T>(_entry: *mut T) -> bool {
            true
        }

        // Safety: `should_remove` unconditionally returns `true`.
        unsafe {
            self.remove_if(hash, eq, should_remove::<T>, hasher, guard)
                .unwrap_unchecked()
        }
    }

    /// Removes a key from the map, returning the entry for the key if the key was previously in the map
    /// and the provided closure returns `true`
    #[inline]
    pub fn remove_if(
        &self,
        hash: u64,
        mut eq: impl FnMut(*mut T) -> bool,
        mut should_remove: impl FnMut(*mut T) -> bool,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Result<Option<*mut T>, *mut T> {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not been initialized yet.
        if table.raw.is_null() {
            return Ok(None);
        }

        let (h1, h2) = meta::split(hash);

        let mut help_copy = true;
        loop {
            // Initialize the probe state.
            let mut probe = Probe::start(h1, table.mask);

            // Probe until we reach the limit.
            let copying = 'probe: loop {
                if probe.len > table.limit {
                    break None;
                }

                // Load the entry metadata first for cheap searches.
                //
                // Safety: `probe.i` is always in-bounds for the table length.
                let meta = unsafe { table.meta(probe.i).load(Ordering::Acquire) };

                // The key is not in the table.
                // It also cannot be in the next table because we have not went over the probe limit.
                if meta == meta::EMPTY {
                    return Ok(None);
                }

                // Check for a potential match.
                if meta != h2 {
                    probe.next(table.mask);
                    continue 'probe;
                }

                // Load the full entry.
                //
                // Safety: `probe.i` is always in-bounds for the table length.
                let mut entry = guard
                    .protect(unsafe { table.entry(probe.i) }, Ordering::Acquire)
                    .unpack();

                // The entry was deleted, keep probing.
                if entry.ptr.is_null() {
                    probe.next(table.mask);
                    continue 'probe;
                }

                // Check for a full match.
                //
                // Safety: We performed a protected load of the pointer using a verified guard with
                // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                // as we hold the guard.
                if !eq(entry.ptr) {
                    probe.next(table.mask);
                    continue 'probe;
                }

                // The entry is being copied to the new table, we have to complete the copy before
                // we can remove it.
                if entry.tag() & Entry::COPYING != 0 {
                    break 'probe Some(probe.i);
                }

                loop {
                    // Ensure that the entry should be removed.
                    //
                    // Safety: `entry` is a valid, non-null, protected entry that we found in the map.
                    if !should_remove(entry.ptr) {
                        return Err(entry.ptr);
                    }

                    // Safety:
                    // - `probe.i` is always in-bounds for the table length
                    // - `entry` is a valid non-null entry that we found in the map.
                    let status = unsafe {
                        self.update_at(probe.i, entry, Entry::TOMBSTONE.cast(), table, guard)
                    };

                    match status {
                        // Successfully removed the entry.
                        UpdateStatus::Replaced(entry) => {
                            // Mark the entry as a tombstone.
                            //
                            // Note that this might end up being overwritten by the metadata hash
                            // if the initial insertion is lagging behind, but we avoid the RMW
                            // and sacrifice reads in the extremely rare case.
                            //
                            // Safety: `probe.i` is always in-bounds for the table length.
                            unsafe {
                                table
                                    .meta(probe.i)
                                    .store(meta::TOMBSTONE, Ordering::Release)
                            };

                            // Decrement the table length.
                            self.count.get(guard).fetch_sub(1, Ordering::Relaxed);

                            // Safety: `entry` is the non-null entry that we just replaced.
                            return Ok(Some(entry.ptr));
                        }

                        // The entry is being copied to the new table, we have to complete the copy
                        // before we can remove.
                        UpdateStatus::Found(EntryStatus::Copied(_)) => break 'probe Some(probe.i),

                        // The entry was deleted.
                        //
                        // We know that at some point during our execution the key was not in the map.
                        UpdateStatus::Found(EntryStatus::Null) => return Ok(None),

                        // Lost to a concurrent update, retry.
                        UpdateStatus::Found(EntryStatus::Value(found)) => entry = found,
                    }
                }
            };

            // Prepare to retry in the next table.
            table = match self.prepare_retry(copying, &mut help_copy, table, &hasher, guard) {
                Some(table) => table,

                // The search was exhausted.
                None => return Ok(None),
            }
        }
    }

    /// Prepare to retry an operation on an existing key in the next table.
    ///
    /// Returns `None` if the recursive search has been exhausted.
    #[cold]
    fn prepare_retry(
        &self,
        copying: Option<usize>,
        help_copy: &mut bool,
        table: Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Option<Table<T>> {
        let next_table = match self.resize {
            ResizeMode::Blocking => match copying {
                // The entry we want to perform the operation on is being copied.
                //
                // In blocking mode we must complete the resize before proceeding.
                Some(_) => self.help_copy(true, &table, hasher, guard),

                // If we went over the probe limit, the key is not in the map.
                None => return None,
            },

            ResizeMode::Incremental(_) => {
                // In incremental resize mode, we always have to check the next table.
                let next_table = table.next_table()?;

                // Help out with the copy.
                if *help_copy {
                    self.help_copy(false, &table, hasher, guard);
                }

                if let Some(i) = copying {
                    // Wait for the entry to be copied.
                    //
                    // We could race with the copy to insert into the table. However,
                    // this entire code path is very rare and likely to complete quickly,
                    // so blocking allows us to make copies faster.
                    self.wait_copied(i, &table);
                }

                next_table
            }
        };

        // Limit incremental copying to once per operation, for more consistent latency.
        *help_copy = false;

        // Continue in the new table.
        Some(next_table)
    }

    /// Attempts to insert an entry at the given index.
    ///
    /// In the case of an error, the returned pointer is guaranteed to be
    /// protected and valid for reads as long as the guard is held.
    ///
    /// # Safety
    ///
    /// The index must be in-bounds for the table. Additionally, `new_entry` must be a
    /// valid owned pointer to insert into the map.
    #[inline]
    unsafe fn insert_at(
        &self,
        i: usize,
        meta: u8,
        new_entry: *mut T,
        table: Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> InsertStatus<T> {
        // Safety: The caller guarantees that `i` is in-bounds.
        let entry = unsafe { table.entry(i) };
        let meta_entry = unsafe { table.meta(i) };

        // Try to claim the empty entry.
        let found = match guard.compare_exchange(
            entry,
            ptr::null_mut(),
            new_entry,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            // Successfully claimed the entry.
            Ok(_) => {
                // Update the metadata table.
                meta_entry.store(meta, Ordering::Release);

                // Return the value we inserted.
                return InsertStatus::Inserted;
            }

            // Lost to a concurrent update.
            Err(found) => found.unpack(),
        };

        let (meta, status) = match EntryStatus::from(found) {
            EntryStatus::Value(_) | EntryStatus::Copied(_) => {
                // An entry was inserted, we have to hash it to get the metadata.
                //
                // The logic is the same for copied entries here as we have to
                // check if the key matches and continue the update in the new table.
                //
                // Safety: We performed a protected load of the pointer using a verified guard
                // with `Acquire` and ensured that it is non-null, meaning it is valid for reads
                // as long as we hold the guard.
                let hash = hasher(found.ptr);
                (meta::h2(hash), EntryStatus::Value(found))
            }

            // The entry was deleted or null copied.
            EntryStatus::Null => (meta::TOMBSTONE, EntryStatus::Null),
        };

        // Ensure the meta table is updated to keep the probe chain alive for readers.
        if meta_entry.load(Ordering::Relaxed) == meta::EMPTY {
            meta_entry.store(meta, Ordering::Release);
        }

        InsertStatus::Found(status)
    }

    /// Attempts to replace the value of an existing entry at the given index.
    ///
    /// In the case of an error, the returned pointer is guaranteed to be
    /// protected and valid for reads.
    ///
    /// # Safety
    ///
    /// - The index must be in-bounds for the table.
    /// - `current` must be a valid non-null entry that was inserted into the map.
    /// - `new_entry` must be a valid sentinel or owned pointer to insert into the map.
    #[inline]
    unsafe fn update_at(
        &self,
        i: usize,
        current: Tagged<T, Entry>,
        new_entry: *mut T,
        table: Table<T>,
        guard: &impl VerifiedGuard,
    ) -> UpdateStatus<T> {
        // Safety: The caller guarantees that `i` is in-bounds.
        let entry = unsafe { table.entry(i) };

        // Try to perform the update.
        let found = match guard.compare_exchange_weak(
            entry,
            current.raw,
            new_entry,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            // Successfully updated.
            Ok(_) => unsafe {
                // Safety: The caller guarantees that `current` is a valid non-null entry that was
                // inserted into the map. Additionally, it is now unreachable from this table due
                // to the CAS above.
                self.defer_retire(current, &table, guard);

                return UpdateStatus::Replaced(current);
            },

            // Lost to a concurrent update.
            Err(found) => found.unpack(),
        };

        UpdateStatus::Found(EntryStatus::from(found))
    }

    /// Reserve capacity for `additional` more elements.
    #[inline]
    pub fn reserve(
        &self,
        additional: usize,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) {
        let mut table = self.root(guard);

        // The table has not yet been allocated, initialize it.
        if table.raw.is_null() {
            table = self.init(Some(probe::entries_for(additional)));
        }

        loop {
            let capacity = probe::entries_for(self.count.sum().checked_add(additional).unwrap());

            // We have enough capacity.
            if table.len() >= capacity {
                return;
            }

            // Race to allocate the new table.
            self.get_or_alloc_next(Some(capacity), table);

            // Force the copy to complete.
            //
            // Note that this is not strictly necessary for a `reserve` operation.
            table = self.help_copy(true, &table, &hasher, guard);
        }
    }

    /// Remove all entries from this table.
    #[inline]
    pub fn clear(&self, hasher: impl Fn(*mut T) -> u64, guard: &impl VerifiedGuard) {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not been initialized yet.
        if table.raw.is_null() {
            return;
        }

        loop {
            // Get a clean copy of the table to delete from.
            table = self.linearize(table, &hasher, guard);

            // Note that this method is not implemented in terms of `retain(|_, _| true)` to avoid
            // loading entry metadata, as there is no need to provide consistency with `get`.
            let mut copying = false;

            'probe: for i in 0..table.len() {
                // Load the entry to delete.
                //
                // Safety: `i` is in bounds for the table length.
                let mut entry = guard
                    .protect(unsafe { table.entry(i) }, Ordering::Acquire)
                    .unpack();

                loop {
                    // The entry is empty or already deleted.
                    if entry.ptr.is_null() {
                        continue 'probe;
                    }

                    // Found a non-empty entry being copied.
                    if entry.tag() & Entry::COPYING != 0 {
                        // Clear every entry in this table that we can, then deal with the copy.
                        copying = true;
                        continue 'probe;
                    }

                    // Try to delete the entry.
                    //
                    // Safety: `i` is in bounds for the table length.
                    let result = unsafe {
                        table.entry(i).compare_exchange(
                            entry.raw,
                            Entry::TOMBSTONE.cast(),
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                    };

                    match result {
                        // Successfully deleted the entry.
                        Ok(_) => {
                            // Update the metadata table.
                            //
                            // Safety: `i` is in bounds for the table length.
                            unsafe { table.meta(i).store(meta::TOMBSTONE, Ordering::Release) };

                            // Decrement the table length.
                            self.count.get(guard).fetch_sub(1, Ordering::Relaxed);

                            // Safety: The caller guarantees that `current` is a valid non-null entry that was
                            // inserted into the map. Additionally, it is now unreachable from this table due
                            // to the CAS above.
                            unsafe { self.defer_retire(entry, &table, guard) };
                            continue 'probe;
                        }

                        // Lost to a concurrent update, retry.
                        Err(found) => entry = found.unpack(),
                    }
                }
            }

            // We cleared every entry in this table.
            if !copying {
                break;
            }

            // A resize prevented us from deleting all the entries in this table.
            //
            // Complete the resize and retry in the new table.
            table = self.help_copy(true, &table, &hasher, guard);
        }
    }

    /// Retains only the elements specified by the predicate.
    #[inline]
    pub fn retain<F>(&self, mut f: F, hasher: impl Fn(*mut T) -> u64, guard: &impl VerifiedGuard)
    where
        F: FnMut(*mut T) -> bool,
    {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not been initialized yet.
        if table.raw.is_null() {
            return;
        }

        loop {
            // Get a clean copy of the table to delete from.
            table = self.linearize(table, &hasher, guard);

            let mut copying = false;
            'probe: for i in 0..table.len() {
                // Load the entry metadata first to ensure consistency with calls to `get`
                // for entries that are retained.
                //
                // Safety: `i` is in bounds for the table length.
                let meta = unsafe { table.meta(i) }.load(Ordering::Acquire);

                // The entry is empty or deleted.
                if matches!(meta, meta::EMPTY | meta::TOMBSTONE) {
                    continue 'probe;
                }

                // Load the entry to delete.
                //
                // Safety: `i` is in bounds for the table length.
                let mut entry = guard
                    .protect(unsafe { table.entry(i) }, Ordering::Acquire)
                    .unpack();

                loop {
                    // The entry is empty or already deleted.
                    if entry.ptr.is_null() {
                        continue 'probe;
                    }

                    // Found a non-empty entry being copied.
                    if entry.tag() & Entry::COPYING != 0 {
                        // Clear every entry in this table that we can, then deal with the copy.
                        copying = true;
                        continue 'probe;
                    }

                    // Should we retain this entry?
                    //
                    // Safety: We performed a protected load of the pointer using a verified guard with
                    // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                    // as we hold the guard.
                    if f(entry.ptr) {
                        continue 'probe;
                    }

                    // Try to delete the entry.
                    //
                    // Safety: `i` is in bounds for the table length.
                    let result = unsafe {
                        table.entry(i).compare_exchange(
                            entry.raw,
                            Entry::TOMBSTONE.cast(),
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                    };

                    match result {
                        // Successfully deleted the entry.
                        Ok(_) => {
                            // Update the metadata table.
                            //
                            // Safety: `i` is in bounds for the table length.
                            unsafe { table.meta(i).store(meta::TOMBSTONE, Ordering::Release) };

                            // Decrement the table length.
                            self.count.get(guard).fetch_sub(1, Ordering::Relaxed);

                            // Safety: The caller guarantees that `current` is a valid non-null entry that was
                            // inserted into the map. Additionally, it is now unreachable from this table due
                            // to the CAS above.
                            unsafe { self.defer_retire(entry, &table, guard) };
                            continue 'probe;
                        }

                        // Lost to a concurrent update, retry.
                        Err(found) => entry = found.unpack(),
                    }
                }
            }

            // We cleared every entry in this table.
            if !copying {
                break;
            }

            // A resize prevented us from deleting all the entries in this table.
            //
            // Complete the resize and retry in the new table.
            table = self.help_copy(true, &table, &hasher, guard);
        }
    }

    /// Returns an iterator over the entries of this table.
    #[inline]
    pub fn iter<'g, G>(&self, hasher: impl Fn(*mut T) -> u64, guard: &'g G) -> Iter<'g, T, G>
    where
        G: VerifiedGuard,
    {
        // Load the root table.
        let root = self.root(guard);

        // The table has not been initialized yet, return a dummy iterator.
        if root.raw.is_null() {
            return Iter {
                i: 0,
                guard,
                table: root,
            };
        }

        // Get a clean copy of the table to iterate over.
        let table = self.linearize(root, &hasher, guard);

        Iter { i: 0, guard, table }
    }

    /// Returns a mutable iterator over the entries of this table.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<T> {
        // Safety: The root table is either null or a valid table allocation.
        let table = unsafe { Table::from_raw(*self.table.get_mut()) };

        // The table has not been initialized yet, return a dummy iterator.
        if table.raw.is_null() {
            return IterMut { i: 0, table };
        }

        IterMut { i: 0, table }
    }
}

impl<T, D: Dealloc<T>> IntoIterator for HashTable<T, D> {
    type Item = *mut T;
    type IntoIter = IntoIter<T, D>;

    /// Returns an owned iterator over the entries of this table.
    fn into_iter(self) -> Self::IntoIter {
        let mut map = ManuallyDrop::new(self);

        // Safety: The root table is either null or a valid table allocation.
        let table = unsafe { Table::from_raw(*map.table.get_mut()) };

        // The table has not been initialized yet, return a dummy iterator.
        if table.raw.is_null() {
            return IntoIter { i: 0, map, table };
        }

        IntoIter { i: 0, map, table }
    }
}

// An operation to perform on given entry in a [`HashTable`].
#[derive(Debug, PartialEq, Eq)]
pub enum Operation<T, A> {
    /// Insert the given value.
    Insert(*mut T),

    /// Remove the entry from the map.
    Remove,

    /// Abort the operation with the given value.
    Abort(A),
}

#[derive(Debug, PartialEq, Eq)]
pub enum Compute<T, A> {
    /// The given entry was inserted.
    Inserted(*mut T),

    /// The entry was updated.
    Updated {
        /// The entry that was replaced.
        old: *mut T,

        /// The entry that was inserted.
        new: *mut T,
    },

    /// The given entry was removed.
    Removed(*mut T),

    /// The operation was aborted with the given value.
    Aborted(A),
}

/// RMW operations.
impl<T, D: Dealloc<T>> HashTable<T, D> {
    /// Update an entry with a CAS function.
    ///
    /// Note that `compute` closure is guaranteed to be called for a `None` input only once, allowing the
    /// insertion of values that cannot be cloned or reconstructed.
    #[inline]
    pub fn compute<F, A>(
        &self,
        hash: u64,
        mut eq: impl FnMut(*mut T) -> bool,
        mut compute: F,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Compute<T, A>
    where
        F: FnMut(Option<*mut T>) -> Operation<T, A>,
    {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not yet been allocated.
        if table.raw.is_null() {
            // Compute the value to insert.
            match compute(None) {
                Operation::Insert(_) => {}
                Operation::Remove => panic!("Cannot remove `None` entry."),
                Operation::Abort(value) => return Compute::Aborted(value),
            }

            // Initialize the table.
            table = self.init(None);
        }

        let (h1, h2) = meta::split(hash);
        let mut help_copy = false;

        loop {
            // Initialize the probe state.
            let mut probe = Probe::start(h1, table.mask);

            // Probe until we reach the limit.
            let copying = 'probe: loop {
                if probe.len > table.limit {
                    break 'probe None;
                }

                // Load the entry metadata first for cheap searches.
                //
                // Safety: `probe.i` is always in-bounds for the table length.
                let meta = unsafe { table.meta(probe.i) }.load(Ordering::Acquire);

                // The entry is empty.
                let mut entry = if meta == meta::EMPTY {
                    // Compute the entry to insert.
                    let new_entry = match compute(None) {
                        Operation::Insert(new_entry) => new_entry,
                        Operation::Remove => panic!("Cannot remove `None` entry."),
                        Operation::Abort(value) => return Compute::Aborted(value),
                    };

                    // Attempt to insert.
                    //
                    // Safety: `probe.i` is always in-bounds for the table length.Additionally,
                    // `new_entry` was allocated above and never shared.
                    match unsafe { self.insert_at(probe.i, h2, new_entry, table, &hasher, guard) } {
                        // Successfully inserted.
                        InsertStatus::Inserted => {
                            // Increment the table length.
                            self.count.get(guard).fetch_add(1, Ordering::Relaxed);

                            // Safety: `new_entry` was initialized above.
                            return Compute::Inserted(new_entry);
                        }

                        // Lost to a concurrent insert.
                        //
                        // If the key matches, we might be able to update the value.
                        InsertStatus::Found(EntryStatus::Value(found))
                        | InsertStatus::Found(EntryStatus::Copied(found)) => found,

                        // The entry was removed or invalidated.
                        InsertStatus::Found(EntryStatus::Null) => {
                            // Continue probing.
                            probe.next(table.mask);
                            continue 'probe;
                        }
                    }
                }
                // Found a potential match.
                else if meta == h2 {
                    // Load the full entry.
                    //
                    // Safety: `probe.i` is always in-bounds for the table length.
                    let found = guard
                        .protect(unsafe { table.entry(probe.i) }, Ordering::Acquire)
                        .unpack();

                    // The entry was deleted, keep probing.
                    if found.ptr.is_null() {
                        probe.next(table.mask);
                        continue 'probe;
                    }

                    // If the key matches, we might be able to update the value.
                    found
                }
                // Otherwise, continue probing.
                else {
                    probe.next(table.mask);
                    continue 'probe;
                };

                // Check for a full match.
                //
                // Safety: We performed a protected load of the pointer using a verified guard with
                // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                // as we hold the guard.
                if !eq(entry.ptr) {
                    probe.next(table.mask);
                    continue 'probe;
                }

                // The entry is being copied to the new table.
                if entry.tag() & Entry::COPYING != 0 {
                    break 'probe Some(probe.i);
                }

                loop {
                    // Compute the value to insert.
                    //
                    // Safety: `entry` is valid for reads.
                    let failure = match compute(Some(entry.ptr)) {
                        // The operation was aborted.
                        Operation::Abort(value) => return Compute::Aborted(value),

                        // Update the value.
                        Operation::Insert(new_entry) => {
                            // Try to perform the update.
                            //
                            // Safety:
                            // - `probe.i` is always in-bounds for the table length
                            // - `entry` is a valid non-null entry that we found in the map.
                            // - `new_entry` was initialized above and never shared.
                            let status =
                                unsafe { self.update_at(probe.i, entry, new_entry, table, guard) };

                            match status {
                                // Successfully updated.
                                UpdateStatus::Replaced(entry) => {
                                    return Compute::Updated {
                                        // Safety: `entry` is a valid non-null entry that we found in the map
                                        // before replacing it.
                                        old: entry.ptr,
                                        // Safety: `new_entry` was initialized above.
                                        new: new_entry,
                                    };
                                }

                                // The update failed.
                                failure => failure,
                            }
                        }

                        // Remove the key from the map.
                        Operation::Remove => {
                            // Try to perform the removal.
                            //
                            // Safety:
                            // - `probe.i` is always in-bounds for the table length
                            // - `entry` is a valid non-null entry that we found in the map.
                            let status = unsafe {
                                self.update_at(
                                    probe.i,
                                    entry,
                                    Entry::TOMBSTONE.cast(),
                                    table,
                                    guard,
                                )
                            };

                            match status {
                                // Successfully removed the entry.
                                UpdateStatus::Replaced(entry) => {
                                    // Mark the entry as a tombstone.
                                    //
                                    // Note that this might end up being overwritten by the metadata hash
                                    // if the initial insertion is lagging behind, but we avoid the RMW
                                    // and sacrifice reads in the extremely rare case.
                                    unsafe {
                                        table
                                            .meta(probe.i)
                                            .store(meta::TOMBSTONE, Ordering::Release)
                                    };

                                    // Decrement the table length.
                                    self.count.get(guard).fetch_sub(1, Ordering::Relaxed);

                                    // Safety: `entry` is a valid non-null entry that we found in the map
                                    // before replacing it.
                                    return Compute::Removed(entry.ptr);
                                }

                                // The remove failed.
                                failure => failure,
                            }
                        }
                    };

                    match failure {
                        // The entry is being copied to the new table.
                        UpdateStatus::Found(EntryStatus::Copied(_)) => break 'probe Some(probe.i),

                        // The entry was deleted before we could update it.
                        //
                        // We know that at some point during our execution the key was not in the map.
                        UpdateStatus::Found(EntryStatus::Null) => {
                            // Compute the next operation.
                            match compute(None) {
                                Operation::Insert(_) => {
                                    // Continue probing to find an empty slot.
                                    probe.next(table.mask);
                                    continue 'probe;
                                }
                                Operation::Remove => panic!("Cannot remove `None` entry."),
                                Operation::Abort(value) => return Compute::Aborted(value),
                            }
                        }

                        // Someone else beat us to the update, retry.
                        UpdateStatus::Found(EntryStatus::Value(found)) => entry = found,

                        _ => unreachable!(),
                    }
                }
            };

            // Prepare to retry in the next table.
            if let Some(next_table) =
                self.prepare_retry(copying, &mut help_copy, table, &hasher, guard)
            {
                table = next_table;
                continue;
            }

            // Otherwise, we have exhausted our search, and the key is not in the map.
            //
            // Check if we should attempt to resize and insert into a new table.
            match compute(None) {
                // Need to insert into the new table.
                Operation::Insert(_) => {
                    table = self.prepare_retry_insert(None, &mut help_copy, table, &hasher, guard);
                }
                // The operation was aborted.
                Operation::Abort(value) => return Compute::Aborted(value),
                Operation::Remove => panic!("Cannot remove `None` entry."),
            }
        }
    }
}

/// Resize operations.
impl<T, D: Dealloc<T>> HashTable<T, D> {
    /// Allocate the initial table.
    #[cold]
    #[inline(never)]
    fn init(&self, capacity: Option<usize>) -> Table<T> {
        const CAPACITY: usize = 32;

        // Allocate the table and mark it as the root.
        let mut new = Table::alloc(capacity.unwrap_or(CAPACITY));
        *new.state_mut().status.get_mut() = State::PROMOTED;

        // Race to write the initial table.
        match self.table.compare_exchange(
            ptr::null_mut(),
            new.raw,
            Ordering::Release,
            Ordering::Acquire,
        ) {
            // Successfully initialized the table.
            Ok(_) => new,

            // Someone beat us, deallocate our table and use the table that was written.
            Err(found) => {
                // Safety: We allocated the table above and never shared it.
                unsafe { Table::dealloc(new) }

                // Safety: The table was just initialized.
                unsafe { Table::from_raw(found) }
            }
        }
    }

    /// Returns the next table, allocating it has not already been created.
    #[cold]
    #[inline(never)]
    fn get_or_alloc_next(&self, capacity: Option<usize>, table: Table<T>) -> Table<T> {
        // Avoid spinning in tests, which can hide race conditions.
        const SPIN_ALLOC: usize = if cfg!(any(test, debug_assertions)) {
            1
        } else {
            7
        };

        // The next table is already allocated.
        if let Some(next) = table.next_table() {
            return next;
        }

        let state = table.state();

        // Otherwise, try to acquire the allocation lock.
        //
        // Unlike in `init`, we do not race here to prevent unnecessary allocator pressure.
        let _allocating = match state.allocating.try_lock() {
            Ok(lock) => lock,
            // Someone else is currently allocating.
            Err(_) => {
                let mut spun = 0;

                // Spin for a bit, waiting for the table to be initialized.
                while spun <= SPIN_ALLOC {
                    for _ in 0..(spun * spun) {
                        hint::spin_loop();
                    }

                    // The table was initialized.
                    if let Some(next) = table.next_table() {
                        return next;
                    }

                    spun += 1;
                }

                // Otherwise, we have to block.
                state.allocating.lock().unwrap()
            }
        };

        // The table was allocated while we were waiting for the lock.
        if let Some(next) = table.next_table() {
            return next;
        }

        let current_capacity = table.len();

        // Loading the length here is quite expensive, we may want to consider
        // a probabilistic counter to detect high-deletion workloads.
        let active_entries = self.len();

        let next_capacity = match cfg!(papaya_stress) {
            // Never grow the table to stress the incremental resizing algorithm.
            true => current_capacity,

            // Double the table capacity if we are at least 50% full.
            false if active_entries >= (current_capacity >> 1) => current_capacity << 1,

            // Halve the table if we are at most 12.5% full.
            //
            // This heuristic is intentionally pessimistic as unnecessarily shrinking
            // is an expensive operation, but it may change in the future. We also respect
            // the initial capacity to give the user a way to retain a strict minimum table
            // size.
            false if active_entries <= (current_capacity >> 3) => {
                self.initial_capacity.max(current_capacity >> 1)
            }

            // Otherwise keep the capacity the same.
            //
            // This can occur due to poor hash distribution or frequent cycling of
            // insertions and deletions, in which case we want to avoid continuously
            // growing the table.
            false => current_capacity,
        };

        let next_capacity = capacity.unwrap_or(next_capacity);
        assert!(
            next_capacity <= isize::MAX as usize,
            "`HashMap` exceeded maximum capacity"
        );

        // Allocate the new table while holding the lock.
        let next = Table::alloc(next_capacity);
        state.next.store(next.raw, Ordering::Release);
        drop(_allocating);

        next
    }

    /// Help along with an existing resize operation, returning the new root table.
    ///
    /// If `copy_all` is `false` in incremental resize mode, this returns the current reference's next
    /// table, not necessarily the new root.
    #[cold]
    #[inline(never)]
    fn help_copy(
        &self,
        copy_all: bool,
        table: &Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Table<T> {
        match self.resize {
            ResizeMode::Blocking => self.help_copy_blocking(table, hasher, guard),
            ResizeMode::Incremental(chunk) => {
                let copied_to = self.help_copy_incremental(chunk, copy_all, hasher, guard);

                if !copy_all {
                    // If we weren't trying to linearize, we have to write to the next table
                    // even if the copy hasn't completed yet.
                    return table.next_table().unwrap();
                }

                copied_to
            }
        }
    }

    /// Help along the resize operation until it completes and the next table is promoted.
    ///
    /// Should only be called on the root table.
    fn help_copy_blocking(
        &self,
        table: &Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Table<T> {
        // Load the next table.
        let mut next = table.next_table().unwrap();

        'copy: loop {
            // Make sure we are copying to the correct table.
            while next.state().status.load(Ordering::Relaxed) == State::ABORTED {
                next = self.get_or_alloc_next(None, next);
            }

            // The copy already completed
            if self.try_promote(table, &next, 0, guard) {
                return next;
            }

            let copy_chunk = table.len().min(4096);

            loop {
                // Every entry has already been claimed.
                if next.state().claim.load(Ordering::Relaxed) >= table.len() {
                    break;
                }

                // Claim a chunk to copy.
                let copy_start = next.state().claim.fetch_add(copy_chunk, Ordering::Relaxed);

                // Copy our chunk of entries.
                let mut copied = 0;
                for i in 0..copy_chunk {
                    let i = copy_start + i;

                    if i >= table.len() {
                        break;
                    }

                    // Copy the entry.
                    //
                    // Safety: We verified that `i` is in-bounds above.
                    if unsafe { !self.copy_at_blocking(i, table, &next, &hasher, guard) } {
                        // This table doesn't have space for the next entry.
                        //
                        // Abort the current resize.
                        //
                        // Note that the `SeqCst` is necessary to make the store visible
                        // to threads that are unparked.
                        next.state().status.store(State::ABORTED, Ordering::SeqCst);

                        // Allocate the next table.
                        let allocated = self.get_or_alloc_next(None, next);

                        // Wake anyone waiting for us to finish.
                        let state = table.state();
                        state.parker.unpark(&state.status);

                        // Retry in a new table.
                        next = allocated;
                        continue 'copy;
                    }

                    copied += 1;
                }

                // Are we done?
                if self.try_promote(table, &next, copied, guard) {
                    return next;
                }

                // If the resize was aborted while we were copying, continue in the new table.
                if next.state().status.load(Ordering::Relaxed) == State::ABORTED {
                    continue 'copy;
                }
            }

            let state = next.state();
            // We copied all that we can, wait for the table to be promoted.
            for spun in 0.. {
                // Avoid spinning in tests, which can hide race conditions.
                const SPIN_WAIT: usize = if cfg!(any(test, debug_assertions)) {
                    1
                } else {
                    7
                };

                // Note that `Acquire` is necessary here to ensure we see the
                // relevant modifications to the root table if see the updated
                // state before parking.
                //
                // Otherwise, `Parker::park` will ensure the necessary synchronization
                // when we are unparked.
                let status = state.status.load(Ordering::Acquire);

                // If this copy was aborted, we have to retry in the new table.
                if status == State::ABORTED {
                    continue 'copy;
                }

                // The copy has completed.
                if status == State::PROMOTED {
                    return next;
                }

                // Copy chunks are relatively small and we expect to finish quickly,
                // so spin for a bit before resorting to parking.
                if spun <= SPIN_WAIT {
                    for _ in 0..(spun * spun) {
                        hint::spin_loop();
                    }

                    continue;
                }

                // Park until the table is promoted.
                state
                    .parker
                    .park(&state.status, |status| status == State::PENDING);
            }
        }
    }

    /// Copy the entry at the given index to the new table.
    ///
    /// Returns `true` if the entry was copied into the table or `false` if the table was full.
    ///
    /// # Safety
    ///
    /// The index must be in-bounds for the table.
    unsafe fn copy_at_blocking(
        &self,
        i: usize,
        table: &Table<T>,
        next_table: &Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> bool {
        // Mark the entry as copying.
        //
        // Safety: The caller guarantees that the index is in-bounds.
        //
        // Note that we don't need to protect the returned entry here, because
        // no one is allowed to retire the entry once we put the `COPYING` bit
        // down until it is inserted into the new table.
        let entry = unsafe { table.entry(i) }
            .fetch_or(Entry::COPYING, Ordering::AcqRel)
            .unpack::<Entry>();

        // The entry is a tombstone.
        if entry.raw.cast() == Entry::TOMBSTONE {
            return true;
        }

        // There is nothing to copy, we're done.
        if entry.ptr.is_null() {
            // Mark as a tombstone so readers avoid having to load the entry.
            //
            // Safety: The caller guarantees that the index is in-bounds.
            unsafe { table.meta(i) }.store(meta::TOMBSTONE, Ordering::Release);
            return true;
        }

        // Copy the value to the new table.
        //
        // Safety: We marked the entry as `COPYING`, ensuring that any updates
        // or removals wait until we complete the copy, and allowing us to get
        // away without a protected load. Additionally, we verified that the
        // entry is non-null, meaning that it is valid for reads.
        unsafe {
            self.insert_copy(entry.ptr.unpack(), false, next_table, hasher, guard)
                .is_some()
        }
    }

    /// Help along an in-progress resize incrementally by copying a chunk of entries.
    ///
    /// Returns the table that was copied to.
    fn help_copy_incremental(
        &self,
        chunk: usize,
        block: bool,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Table<T> {
        // Always help the highest priority root resize.
        let table = self.root(guard);

        // Load the next table.
        let Some(next) = table.next_table() else {
            // The copy we tried to help was already promoted.
            return table;
        };

        loop {
            // The copy already completed.
            if self.try_promote(&table, &next, 0, guard) {
                return next;
            }

            loop {
                // Every entry has already been claimed.
                if next.state().claim.load(Ordering::Relaxed) >= table.len() {
                    break;
                }

                // Claim a chunk to copy.
                let copy_start = next.state().claim.fetch_add(chunk, Ordering::Relaxed);

                // Copy our chunk of entries.
                let mut copied = 0;
                for i in 0..chunk {
                    let i = copy_start + i;

                    if i >= table.len() {
                        break;
                    }

                    // Copy the entry.
                    //
                    // Safety: We verified that `i` is in-bounds above.
                    unsafe { self.copy_at_incremental(i, &table, &next, &hasher, guard) };
                    copied += 1;
                }

                // Update the copy state, and try to promote the table.
                //
                // Only copy a single chunk if promotion fails, unless we are forced
                // to complete the resize.
                if self.try_promote(&table, &next, copied, guard) || !block {
                    return next;
                }
            }

            // There are no entries that we can copy, block if necessary.
            if !block {
                return next;
            }

            let state = next.state();
            for spun in 0.. {
                // Avoid spinning in tests, which can hide race conditions.
                const SPIN_WAIT: usize = if cfg!(any(test, debug_assertions)) {
                    1
                } else {
                    7
                };

                // The copy has completed.
                //
                // Note that `Acquire` is necessary here to ensure we see the
                // relevant modifications to the root table if see the updated
                // state before parking.
                //
                // Otherwise, `Parker::park` will ensure the necessary synchronization
                // when we are unparked.
                let status = state.status.load(Ordering::Acquire);
                if status == State::PROMOTED {
                    return next;
                }

                // Copy chunks are relatively small and we expect to finish quickly,
                // so spin for a bit before resorting to parking.
                if spun <= SPIN_WAIT {
                    for _ in 0..(spun * spun) {
                        hint::spin_loop();
                    }

                    continue;
                }

                // Park until the table is promoted.
                state
                    .parker
                    .park(&state.status, |status| status == State::PENDING);
            }
        }
    }

    /// Copy the entry at the given index to the new table.
    ///
    /// # Safety
    ///
    /// The index must be in-bounds for the table.
    unsafe fn copy_at_incremental(
        &self,
        i: usize,
        table: &Table<T>,
        next_table: &Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) {
        // Safety: The caller guarantees that the index is in-bounds.
        let entry = unsafe { table.entry(i) };

        // Mark the entry as copying.
        let found = entry.fetch_or(Entry::COPYING, Ordering::AcqRel).unpack();

        // The entry is a tombstone.
        if found.raw.cast() == Entry::TOMBSTONE {
            return;
        }

        // There is nothing to copy, we're done.
        if found.ptr.is_null() {
            // Mark as a tombstone so readers avoid having to load the entry.
            //
            // Safety: The caller guarantees that the index is in-bounds.
            unsafe { table.meta(i) }.store(meta::TOMBSTONE, Ordering::Release);
            return;
        }

        // Mark the entry as borrowed so writers in the new table know it was copied.
        let new_entry = found.map_tag(|addr| addr | Entry::BORROWED);

        // Copy the value to the new table.
        //
        // Safety: We marked the entry as `COPYING`, ensuring that any updates
        // or removals wait until we complete the copy, and allowing us to get
        // away without a protected load. Additionally, we verified that the
        // entry is non-null, meaning that it is valid for reads.
        unsafe {
            self.insert_copy(new_entry, true, next_table, hasher, guard)
                .unwrap();
        }

        // Mark the entry as copied.
        let copied = found
            .raw
            .map_addr(|addr| addr | Entry::COPYING | Entry::COPIED);

        // Note that we already wrote the COPYING bit, so no one is writing to the old
        // entry except us.
        //
        // Note that the `SeqCst` is necessary to make the store visible to threads
        // that are unparked.
        entry.store(copied, Ordering::SeqCst);

        // Notify any writers that the copy has completed.
        table.state().parker.unpark(entry);
    }

    // Copy an entry into the table, returning the index it was inserted into.
    //
    // This is an optimized version of `insert_entry` where the caller is the only writer
    // inserting the given key into the new table, as it has already been marked as copying.
    //
    // # Safety
    //
    // The new entry must be valid for reads.
    unsafe fn insert_copy(
        &self,
        new_entry: Tagged<T, Entry>,
        resize: bool,
        table: &Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Option<(Table<T>, usize)> {
        let mut table = *table;

        // Safety: The new entry is guaranteed to be valid for reads.
        let hash = hasher(new_entry.ptr);

        let (h1, h2) = meta::split(hash);

        loop {
            // Initialize the probe state.
            let mut probe = Probe::start(h1, table.mask);

            // Probe until we reach the limit.
            while probe.len <= table.limit {
                // Safety: `probe.i` is always in-bounds for the table length.
                let meta_entry = unsafe { table.meta(probe.i) };

                // Load the entry metadata first for cheap searches.
                let meta = meta_entry.load(Ordering::Acquire);

                // The entry is empty, try to insert.
                if meta == meta::EMPTY {
                    // Safety: `probe.i` is always in-bounds for the table length.
                    let entry = unsafe { table.entry(probe.i) };

                    // Try to claim the entry.
                    match guard.compare_exchange(
                        entry,
                        ptr::null_mut(),
                        new_entry.raw,
                        Ordering::Release,
                        Ordering::Acquire,
                    ) {
                        // Successfully inserted.
                        Ok(_) => {
                            // Update the metadata table.
                            meta_entry.store(h2, Ordering::Release);
                            return Some((table, probe.i));
                        }
                        Err(found) => {
                            let found = found.unpack::<Entry>();

                            // The entry was deleted or copied.
                            let meta = if found.ptr.is_null() {
                                meta::TOMBSTONE
                            } else {
                                // Ensure the meta table is updated to avoid breaking the probe chain.
                                //
                                // Safety: We performed a protected load of the pointer using a verified guard with
                                // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                                // as we hold the guard.
                                let hash = hasher(found.ptr);
                                meta::h2(hash)
                            };

                            if meta_entry.load(Ordering::Relaxed) == meta::EMPTY {
                                meta_entry.store(meta, Ordering::Release);
                            }
                        }
                    }
                }

                // Continue probing.
                probe.next(table.mask);
            }

            if !resize {
                return None;
            }

            // Insert into the next table.
            table = self.get_or_alloc_next(None, table);
        }
    }

    // Update the copy state and attempt to promote a table to the root.
    //
    // Returns `true` if the table was promoted.
    fn try_promote(
        &self,
        table: &Table<T>,
        next: &Table<T>,
        copied: usize,
        guard: &impl VerifiedGuard,
    ) -> bool {
        let state = next.state();

        // Update the copy count.
        let copied = if copied > 0 {
            state.copied.fetch_add(copied, Ordering::AcqRel) + copied
        } else {
            state.copied.load(Ordering::Acquire)
        };

        // If we copied all the entries in the table, we can try to promote.
        if copied == table.len() {
            let root = self.table.load(Ordering::Relaxed);

            // Only promote root copies.
            //
            // We can't promote a nested copy before it's parent has finished, as
            // it may not contain all the entries in the table.
            if table.raw == root {
                // Try to update the root.
                if self
                    .table
                    .compare_exchange(table.raw, next.raw, Ordering::Release, Ordering::Acquire)
                    .is_ok()
                {
                    // Successfully promoted the table.
                    //
                    // Note that the `SeqCst` is necessary to make the store visible to threads
                    // that are unparked.
                    state.status.store(State::PROMOTED, Ordering::SeqCst);

                    // Retire the old table.
                    //
                    // Safety: `table.raw` is a valid pointer to the table we just copied from.
                    // Additionally, the CAS above made the previous table unreachable from the
                    // root pointer, allowing it to be safely retired.
                    unsafe {
                        guard.defer_retire(table.raw, |table, collector| {
                            // Note that we do not drop entries because they have been copied to
                            // the new root.
                            drop_table::<_, D>(Table::from_raw(table), collector);
                        });
                    }
                }

                // Wake up any writers waiting for the resize to complete.
                state.parker.unpark(&state.status);
                return true;
            }
        }

        // Not ready to promote yet.
        false
    }

    // Completes all pending copies in incremental mode to get a clean copy of the table.
    //
    // This is necessary for operations like `iter` or `clear`, where entries in multiple tables
    // can cause lead to incomplete results.
    #[inline]
    fn linearize(
        &self,
        mut table: Table<T>,
        hasher: impl Fn(*mut T) -> u64,
        guard: &impl VerifiedGuard,
    ) -> Table<T> {
        if self.is_incremental() {
            // If we're in incremental resize mode, we need to complete any in-progress resizes to
            // ensure we don't miss any entries in the next table. We can't iterate over both because
            // we risk returning the same entry twice.
            while table.next_table().is_some() {
                table = self.help_copy(true, &table, &hasher, guard);
            }
        }

        table
    }

    // Wait for an incremental copy of a given entry to complete.
    #[cold]
    #[inline(never)]
    fn wait_copied(&self, i: usize, table: &Table<T>) {
        // Avoid spinning in tests, which can hide race conditions.
        const SPIN_WAIT: usize = if cfg!(any(test, debug_assertions)) {
            1
        } else {
            5
        };

        let entry = unsafe { table.entry(i) };

        // Spin for a short while, waiting for the entry to be copied.
        for spun in 0..SPIN_WAIT {
            // The entry was copied.
            let entry = entry.load(Ordering::Acquire).unpack::<Entry>();
            if entry.tag() & Entry::COPIED != 0 {
                return;
            }

            for _ in 0..(spun * spun) {
                hint::spin_loop();
            }
        }

        // Park until the copy completes.
        let parker = &table.state().parker;
        parker.park(entry, |entry| entry.addr() & Entry::COPIED == 0);
    }

    /// Retire an entry that was removed from the current table, but may still be reachable from
    /// previous tables.
    ///
    /// # Safety
    ///
    /// The entry must be a valid pointer that is unreachable from the current table. Additionally,
    /// it is *undefined behavior* to call this method multiple times for the same entry.
    #[inline]
    unsafe fn defer_retire(
        &self,
        entry: Tagged<T, Entry>,
        table: &Table<T>,
        guard: &impl VerifiedGuard,
    ) {
        match self.resize {
            // Safety: In blocking resize mode, we only ever write to the root table, so the entry
            // is inaccessible from all tables.
            ResizeMode::Blocking => unsafe {
                guard.defer_retire(entry.ptr, Entry::reclaim::<T, D>);
            },
            // In incremental resize mode, the entry may be accessible in previous tables.
            ResizeMode::Incremental(_) => {
                if entry.tag() & Entry::BORROWED == 0 {
                    // Safety: If the entry is not borrowed, meaning it is not in any previous tables,
                    // it is inaccessible even if the current table is not root. Thus we can safely retire.
                    unsafe { guard.defer_retire(entry.ptr, Entry::reclaim::<T, D>) };
                    return;
                }

                let root = self.root(guard);

                // Check if our table, or any subsequent table, is the root.
                let mut next = Some(*table);
                while let Some(table) = next {
                    if table.raw == root.raw {
                        // Safety: The root table is our table or a table that succeeds ours.
                        // Thus any previous tables are unreachable from the root, so we can safely retire.
                        unsafe { guard.defer_retire(entry.ptr, Entry::reclaim::<T, D>) };
                        return;
                    }

                    next = table.next_table();
                }

                // Otherwise, we have to wait for the table we are copying from to be reclaimed.
                //
                // Find the table we are copying from, searching from the root.
                let mut prev = root;

                loop {
                    let next = prev.next_table().unwrap();

                    // Defer the entry to be retired by the table we are copying from.
                    if next.raw == table.raw {
                        prev.state().deferred.push(entry.ptr);
                        return;
                    }

                    prev = next;
                }
            }
        }
    }
}

// An iterator over the keys and values of this table.
pub struct Iter<'g, T, G> {
    i: usize,
    table: Table<T>,
    guard: &'g G,
}

impl<'g, T, G> Iterator for Iter<'g, T, G>
where
    G: VerifiedGuard,
{
    type Item = *mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // The table has not yet been allocated.
        if self.table.raw.is_null() {
            return None;
        }

        loop {
            // Iterated over every entry in the table, we're done.
            if self.i >= self.table.len() {
                return None;
            }

            // Load the entry metadata first to ensure consistency with calls to `get`.
            //
            // Safety: We verified that `self.i` is in-bounds above.
            let meta = unsafe { self.table.meta(self.i) }.load(Ordering::Acquire);

            // The entry is empty or deleted.
            if matches!(meta, meta::EMPTY | meta::TOMBSTONE) {
                self.i += 1;
                continue;
            }

            // Load the entry.
            //
            // Safety: We verified that `self.i` is in-bounds above.
            let entry = self
                .guard
                .protect(unsafe { self.table.entry(self.i) }, Ordering::Acquire)
                .unpack::<Entry>();

            // The entry was deleted.
            if entry.ptr.is_null() {
                self.i += 1;
                continue;
            }

            self.i += 1;

            // Safety: We performed a protected load of the pointer using a verified guard with
            // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
            // as we hold the guard.
            return Some(entry.ptr);
        }
    }
}

// Safety: An iterator simply yields pointers to the keys and values.
// Dereferencing those pointers is up to the caller. Additionally,
// an iterator holds a shared reference to the guard, so the
// guard must be `Sync` for the iterator to be `Send` or `Sync`.
unsafe impl<T, G> Send for Iter<'_, T, G> where G: Sync {}
unsafe impl<T, G> Sync for Iter<'_, T, G> where G: Sync {}

impl<T, G> Clone for Iter<'_, T, G> {
    #[inline]
    fn clone(&self) -> Self {
        Iter {
            i: self.i,
            table: self.table,
            guard: self.guard,
        }
    }
}

// A mutable iterator over the keys and values of this table.
pub struct IterMut<T> {
    i: usize,
    table: Table<T>,
}

impl<T> Iterator for IterMut<T> {
    type Item = *mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.next_raw()
    }
}

impl<T> IterMut<T> {
    // Note that this method is guaranteed to only yield entry pointers that are valid for reads
    // and writes.
    #[inline]
    fn next_raw(&mut self) -> Option<*mut T> {
        // The table has not yet been allocated.
        if self.table.raw.is_null() {
            return None;
        }

        loop {
            // Iterated over every entry in the table, proceed to a nested resize if there is one.
            if self.i >= self.table.len() {
                if let Some(next_table) = self.table.next_table() {
                    self.i = 0;
                    self.table = next_table;
                    continue;
                }

                // Otherwise, we're done.
                return None;
            }

            // Load the entry.
            //
            // Safety: We verified that `self.i` is in-bounds above.
            let entry = unsafe { self.table.entry_mut(self.i) }.unpack::<Entry>();

            // The entry was deleted.
            if entry.ptr.is_null() {
                self.i += 1;
                continue;
            }

            // The entry was copied, we'll yield it when iterating over the table it was copied to.
            //
            // We have a mutable reference to the table, so there are no concurrent removals that
            // can lead to us yielding duplicate entries.
            if entry.tag() & Entry::COPIED != 0 {
                self.i += 1;
                continue;
            }

            self.i += 1;

            // Safety: We ensured the entry is non-null.
            return Some(entry.ptr);
        }
    }

    // Returns an immutable iterator over the remaining entries.
    pub(crate) fn iter(&self) -> impl Iterator<Item = *mut T> + '_ {
        // Note that we still hold on to the mutable reference to the table,
        // so we can iterate without synchronization.
        IterMut {
            i: self.i,
            table: self.table,
        }
    }
}

impl<T> Clone for IterMut<T> {
    #[inline]
    fn clone(&self) -> Self {
        IterMut {
            i: self.i,
            table: self.table,
        }
    }
}

// Safety: A mutable iterator does not perform any concurrent access and
// simply yields pointers to the keys and values. Dereferencing those
// pointers is up to the caller.
unsafe impl<T> Send for IterMut<T> {}
unsafe impl<T> Sync for IterMut<T> {}

// An owned iterator over the keys and values of this table.
pub struct IntoIter<T, D: Dealloc<T>> {
    i: usize,
    table: Table<T>,
    map: ManuallyDrop<HashTable<T, D>>,
}

impl<T, D: Dealloc<T>> Iterator for IntoIter<T, D> {
    type Item = *mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        // The table has not yet been allocated.
        if self.table.raw.is_null() {
            return None;
        }

        loop {
            // Iterated over every entry in the table, proceed to a nested resize if there is one.
            if self.i >= self.table.len() {
                if let Some(next_table) = self.table.next_table() {
                    // Drop the previous table.
                    //
                    // Safety: We have unique access to the table and do
                    // not access it after this call.
                    unsafe { drop_table::<_, D>(self.table, &self.map.collector) };

                    // Reset the iterator for the next table.
                    self.i = 0;
                    self.table = next_table;
                    continue;
                }

                // Otherwise, we're done.
                return None;
            }

            // Load the entry.
            //
            // Safety: We verified that `self.i` is in-bounds above.
            let entry = unsafe { self.table.entry_mut(self.i) }.unpack::<Entry>();

            // The entry was deleted.
            if entry.ptr.is_null() {
                self.i += 1;
                continue;
            }

            // The entry was copied, we'll yield it when iterating over the table it was copied to.
            //
            // We own the table, so there are no concurrent removals that can lead to us yielding
            // duplicate entries.
            if entry.tag() & Entry::COPIED != 0 {
                self.i += 1;
                continue;
            }

            self.i += 1;

            // Safety: We ensured the entry is non-null. Additionally, we own the map
            // and ensure not to drop already yielded entries.
            return Some(entry.ptr);
        }
    }
}

impl<T, D: Dealloc<T>> IntoIter<T, D> {
    // Returns an immutable iterator over the remaining entries.
    pub(crate) fn iter(&self) -> impl Iterator<Item = *mut T> + '_ {
        // `IntoIter` owns the `HashTable` and do not expose any access
        // except through `&mut self` so we can use a mutable iterator.
        IterMut {
            i: self.i,
            table: self.table,
        }
    }
}

impl<T, D: Dealloc<T>> Drop for IntoIter<T, D> {
    fn drop(&mut self) {
        // Drop the remaining elements that have not been yielded.
        drop_parts::<_, D>(self.i, self.table.raw, &mut self.map.collector);
    }
}

// Safety: An owned iterator does not perform any concurrent access and
// simply yields pointers to the keys and values. Dereferencing those
// pointers is up to the caller.
unsafe impl<T, D: Dealloc<T>> Send for IntoIter<T, D> {}
unsafe impl<T, D: Dealloc<T>> Sync for IntoIter<T, D> {}

impl<T, D: Dealloc<T>> Drop for HashTable<T, D> {
    fn drop(&mut self) {
        drop_parts::<_, D>(0, *self.table.get_mut(), &mut self.collector);
    }
}

// Drop the elements of a `HashMap`.
fn drop_parts<T, D: Dealloc<T>>(
    mut start: usize,
    mut raw: *mut RawTable<T>,
    collector: &mut Collector,
) {
    // Make sure all objects are reclaimed before the collector is dropped.
    //
    // Dropping a table depends on accessing the collector for deferred retirement,
    // using the shared collector pointer that is invalidated by drop.
    //
    // Safety: We have a unique reference to the collector.
    unsafe { collector.reclaim_all() };

    // Drop all nested tables and entries.
    while !raw.is_null() {
        // Safety: The root and next tables are always valid pointers to a
        // table allocation, or null.
        let mut table = unsafe { Table::from_raw(raw) };

        // Read the next table pointer before dropping the current one.
        let next = *table.state_mut().next.get_mut();

        // Safety: We have unique access to the table and do
        // not access the entries after this call.
        unsafe { drop_entries::<_, D>(start, table) };

        // Safety: We have unique access to the table and do
        // not access it after this call.
        unsafe { drop_table::<_, D>(table, collector) };

        // Continue for all nested tables.
        raw = next;
        start = 0;
    }
}

// Drop all entries in this table.
//
// # Safety
//
// The table entries must not be accessed after this call.
unsafe fn drop_entries<T, D: Dealloc<T>>(start: usize, table: Table<T>) {
    for i in start..table.len() {
        // Safety: `i` is in-bounds and we have unique access to the table.
        let entry = unsafe { (*table.entry(i).as_ptr()).unpack::<Entry>() };

        // The entry was copied, or there is nothing to deallocate.
        if entry.ptr.is_null() || entry.tag() & Entry::COPYING != 0 {
            continue;
        }

        // Drop the entry.
        //
        // Safety: We verified that the table is non-null and will
        // not be accessed after this call. Additionally, we ensured
        // that the entry is not copied to avoid double freeing entries
        // that may exist in multiple tables.
        unsafe { D::dealloc(entry.ptr) };
    }
}

// Drop the table allocation.
//
// # Safety
//
// The table must not be accessed after this call.
unsafe fn drop_table<T, D: Dealloc<T>>(mut table: Table<T>, collector: &Collector) {
    // Drop any entries that were deferred during an incremental resize.
    //
    // Safety: Entries are deferred after they are made unreachable from the
    // next table during a resize from this table. This table must have been accessible
    // from the root for any entry to have been deferred. Thus it is being retired now,
    // *after* the entry was made inaccessible from the next table. Additionally, for
    // this table to have been retired, it also must no longer be accessible from the root,
    // meaning that the entry has been totally removed from the map, and can be safely
    // retired.
    table
        .state_mut()
        .deferred
        .drain(|entry| unsafe { collector.retire(entry, Entry::reclaim::<T, D>) });

    // Deallocate the table.
    //
    // Safety: The caller guarantees that the table will not be accessed after this call.
    unsafe { Table::dealloc(table) };
}

// Entry metadata, inspired by `hashbrown`.
pub mod meta {
    use std::mem;

    // Indicates an empty entry.
    pub const EMPTY: u8 = 0x80;

    // Indicates an entry that has been deleted.
    pub const TOMBSTONE: u8 = u8::MAX;

    // Returns the primary hash and secondary for an entry.
    #[inline]
    pub fn split(hash: u64) -> (usize, u8) {
        (h1(hash), h2(hash))
    }

    // Returns the primary hash for an entry.
    #[inline]
    pub fn h1(hash: u64) -> usize {
        hash as usize
    }

    /// Return a byte of hash metadata, used for cheap searches.
    #[inline]
    pub fn h2(hash: u64) -> u8 {
        const MIN_HASH_LEN: usize = if mem::size_of::<usize>() < mem::size_of::<u64>() {
            mem::size_of::<usize>()
        } else {
            mem::size_of::<u64>()
        };

        // Grab the top 7 bits of the hash.
        //
        // While the hash is normally a full 64-bit value, some hash functions
        // (such as fxhash) produce a usize result instead, which means that the
        // top 32 bits are 0 on 32-bit platforms.
        let top7 = hash >> (MIN_HASH_LEN * 8 - 7);
        (top7 & 0x7f) as u8
    }
}
