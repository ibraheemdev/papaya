mod alloc;
mod probe;

pub(crate) mod utils;

use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::{hint, panic, ptr};

use self::alloc::{RawTable, Table};
use self::probe::Probe;
use self::utils::{untagged, AtomicPtrFetchOps, Counter, Parker, StrictProvenance, Tagged};
use crate::map::{Compute, Operation, ResizeMode};
use crate::Equivalent;

use seize::{Collector, LocalGuard, OwnedGuard};
use utils::{MapGuard, Stack, VerifiedGuard};

/// A lock-free hash-table.
pub struct HashMap<K, V, S> {
    /// A pointer to the root table.
    table: AtomicPtr<RawTable<Entry<K, V>>>,

    /// Collector for memory reclamation.
    collector: Collector,

    /// The resize mode, either blocking or incremental.
    resize: ResizeMode,

    /// An atomic counter of the number of keys in the table.
    count: Counter,

    /// The initial capacity provided to `HashMap::new`.
    ///
    /// The table is guaranteed to never shrink below this capacity.
    initial_capacity: usize,

    /// Hasher for keys.
    pub hasher: S,
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

// The result of an insert operation.
pub enum InsertResult<'g, V> {
    /// Inserted the given value.
    Inserted(&'g V),

    /// Replaced the given value.
    Replaced(&'g V),

    /// Error returned by `try_insert`.
    Error { current: &'g V, not_inserted: V },
}

// The raw result of an insert operation.
pub enum RawInsertResult<'g, K, V> {
    /// Inserted the given value.
    Inserted(&'g V),

    /// Replaced the given value.
    Replaced(&'g V),

    /// Error returned by `try_insert`.
    Error {
        current: &'g V,
        not_inserted: *mut Entry<K, V>,
    },
}

// An entry in the hash-table.
#[repr(C, align(8))] // Reserve the lower 3 bits for pointer tagging.
pub struct Entry<K, V> {
    /// The key for this entry.
    pub key: K,

    /// The value for this entry.
    pub value: V,
}

impl Entry<(), ()> {
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

impl<K, V> utils::Unpack for Entry<K, V> {
    /// Mask for an entry pointer, ignoring any tag bits.
    const MASK: usize = !(Entry::COPYING | Entry::COPIED | Entry::BORROWED);
}

impl<K, V> Entry<K, V> {
    /// A sentinel pointer for a deleted entry.
    ///
    /// Null pointers are never copied to the new table, so this state is safe to use.
    /// Note that tombstone entries may still be marked as `COPYING`, so this state
    /// cannot be used for direct equality.
    const TOMBSTONE: *mut Entry<K, V> = Entry::COPIED as _;
}

/// The status of an entry.
enum EntryStatus<K, V> {
    /// The entry is a tombstone or null (potentially a null copy).
    Null,

    /// The entry is being copied.
    Copied(Tagged<Entry<K, V>>),

    /// A valid entry.
    Value(Tagged<Entry<K, V>>),
}

impl<K, V> From<Tagged<Entry<K, V>>> for EntryStatus<K, V> {
    /// Returns the status for this entry.
    #[inline]
    fn from(entry: Tagged<Entry<K, V>>) -> Self {
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
enum UpdateStatus<K, V> {
    /// Successfully replaced the given key and value.
    Replaced(Tagged<Entry<K, V>>),

    /// A new entry was written before we could update.
    Found(EntryStatus<K, V>),
}

/// The state of an entry we attempted to insert into.
enum InsertStatus<K, V> {
    /// Successfully inserted the value.
    Inserted,

    /// A new entry was written before we could update.
    Found(EntryStatus<K, V>),
}

impl<K, V, S> HashMap<K, V, S> {
    /// Creates new hash-table with the given options.
    #[inline]
    pub fn new(
        capacity: usize,
        hasher: S,
        collector: Collector,
        resize: ResizeMode,
    ) -> HashMap<K, V, S> {
        // The table is lazily allocated.
        if capacity == 0 {
            return HashMap {
                collector,
                resize,
                hasher,
                initial_capacity: 1,
                table: AtomicPtr::new(ptr::null_mut()),
                count: Counter::default(),
            };
        }

        // Initialize the table and mark it as the root.
        let mut table = Table::alloc(probe::entries_for(capacity));
        *table.state_mut().status.get_mut() = State::PROMOTED;

        HashMap {
            hasher,
            resize,
            collector,
            initial_capacity: capacity,
            table: AtomicPtr::new(table.raw),
            count: Counter::default(),
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
    fn root(&self, guard: &impl VerifiedGuard) -> Table<Entry<K, V>> {
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

    /// Returns true if incremental resizing is enabled.
    #[inline]
    fn is_incremental(&self) -> bool {
        matches!(self.resize, ResizeMode::Incremental(_))
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Returns a reference to the entry corresponding to the key.
    #[inline]
    pub fn get<'g, Q>(&self, key: &Q, guard: &'g impl VerifiedGuard) -> Option<(&'g K, &'g V)>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not been initialized yet.
        if table.raw.is_null() {
            return None;
        }

        let (h1, h2) = self.hash(key);

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
                        .unpack();

                    // The entry was deleted, keep probing.
                    if entry.ptr.is_null() {
                        probe.next(table.mask);
                        continue 'probe;
                    }

                    // Safety: We performed a protected load of the pointer using a verified guard with
                    // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                    // as we hold the guard.
                    let entry_ref = unsafe { &(*entry.ptr) };

                    // Check for a full match.
                    if key.equivalent(&entry_ref.key) {
                        // The entry was copied to the new table.
                        //
                        // In blocking resize mode we do not need to perform self check as all writes block
                        // until any resizes are complete, making the root table the source of truth for readers.
                        if entry.tag() & Entry::COPIED != 0 {
                            break 'probe;
                        }

                        // Found the correct entry, return the key and value.
                        return Some((&entry_ref.key, &entry_ref.value));
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

    /// Inserts a key-value pair into the table.
    #[inline]
    pub fn insert<'g>(
        &self,
        key: K,
        value: V,
        replace: bool,
        guard: &'g impl VerifiedGuard,
    ) -> InsertResult<'g, V> {
        // Perform the insert.
        let raw_result = self.insert_inner(key, value, replace, guard);

        let result = match raw_result {
            // Updated an entry.
            RawInsertResult::Replaced(value) => InsertResult::Replaced(value),

            // Inserted a new entry.
            RawInsertResult::Inserted(value) => {
                // Increment the table length.
                self.count.get(guard).fetch_add(1, Ordering::Relaxed);

                InsertResult::Inserted(value)
            }

            // Failed to insert the entry.
            RawInsertResult::Error {
                current,
                not_inserted,
            } => {
                // Safety: We allocated this box above and it was not inserted into the table.
                let not_inserted = unsafe { Box::from_raw(not_inserted) };

                InsertResult::Error {
                    current,
                    not_inserted: not_inserted.value,
                }
            }
        };

        result
    }

    /// Inserts an entry into the map.
    #[inline]
    fn insert_inner<'g>(
        &self,
        key: K,
        value: V,
        should_replace: bool,
        guard: &'g impl VerifiedGuard,
    ) -> RawInsertResult<'g, K, V> {
        // Allocate the entry to be inserted.
        let new_entry = untagged(Box::into_raw(Box::new(Entry { key, value })));

        // Safety: We just allocated the entry above.
        let new_ref = unsafe { &(*new_entry.ptr) };

        // Load the root table.
        let mut table = self.root(guard);

        // Allocate the table if it has not been initialized yet.
        if table.raw.is_null() {
            table = self.init(None);
        }

        let (h1, h2) = self.hash(&new_ref.key);

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
                    match unsafe { self.insert_at(probe.i, h2, new_entry.raw, table, guard) } {
                        // Successfully inserted.
                        InsertStatus::Inserted => return RawInsertResult::Inserted(&new_ref.value),

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

                // Safety: We performed a protected load of the pointer using a verified guard with
                // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                // as we hold the guard.
                let entry_ref = unsafe { &(*entry.ptr) };

                // Check for a full match.
                if entry_ref.key != new_ref.key {
                    probe.next(table.mask);
                    continue 'probe;
                }

                // The entry is being copied to the new table.
                if entry.tag() & Entry::COPYING != 0 {
                    break 'probe Some(probe.i);
                }

                // Return an error for calls to `try_insert`.
                if !should_replace {
                    return RawInsertResult::Error {
                        current: &entry_ref.value,
                        not_inserted: new_entry.ptr,
                    };
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
                        let value = unsafe { &(*entry.ptr).value };
                        return RawInsertResult::Replaced(value);
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
            table = self.prepare_retry_insert(copying, &mut help_copy, table, guard);
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
        mut entry: Tagged<Entry<K, V>>,
        new_entry: *mut Entry<K, V>,
        table: Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> UpdateStatus<K, V> {
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
        table: Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> Table<Entry<K, V>> {
        // If went over the probe limit or found a copied entry, trigger a resize.
        let mut next_table = self.get_or_alloc_next(None, table);

        let next_table = match self.resize {
            // In blocking mode we must complete the resize before proceeding.
            ResizeMode::Blocking => self.help_copy(true, &table, guard),

            // In incremental mode we can perform more granular blocking.
            ResizeMode::Incremental(_) => {
                // Help out with the copy.
                if *help_copy {
                    next_table = self.help_copy(false, &table, guard);
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
    pub fn remove<'g, Q>(&self, key: &Q, guard: &'g impl VerifiedGuard) -> Option<(&'g K, &'g V)>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        #[inline(always)]
        fn should_remove<K, V>(_key: &K, _value: &V) -> bool {
            true
        }

        // Safety: `should_remove` unconditionally returns `true`.
        unsafe { self.remove_if(key, should_remove, guard).unwrap_unchecked() }
    }

    /// Removes a key from the map, returning the entry for the key if the key was previously in the map
    /// and the provided closure returns `true`
    #[inline]
    pub fn remove_if<'g, Q, F>(
        &self,
        key: &Q,
        mut should_remove: F,
        guard: &'g impl VerifiedGuard,
    ) -> Result<Option<(&'g K, &'g V)>, (&'g K, &'g V)>
    where
        Q: Equivalent<K> + Hash + ?Sized,
        F: FnMut(&K, &V) -> bool,
    {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not been initialized yet.
        if table.raw.is_null() {
            return Ok(None);
        }

        let (h1, h2) = self.hash(key);

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
                if !key.equivalent(unsafe { &(*entry.ptr).key }) {
                    probe.next(table.mask);
                    continue 'probe;
                }

                // The entry is being copied to the new table, we have to complete the copy before
                // we can remove it.
                if entry.tag() & Entry::COPYING != 0 {
                    break 'probe Some(probe.i);
                }

                loop {
                    // Safety: `entry` is a valid, non-null, protected entry that we found in the map.
                    let entry_ref = unsafe { &(*entry.ptr) };

                    // Ensure that the entry should be removed.
                    if !should_remove(&entry_ref.key, &entry_ref.value) {
                        return Err((&entry_ref.key, &entry_ref.value));
                    }

                    // Safety:
                    // - `probe.i` is always in-bounds for the table length
                    // - `entry` is a valid non-null entry that we found in the map.
                    let status =
                        unsafe { self.update_at(probe.i, entry, Entry::TOMBSTONE, table, guard) };

                    match status {
                        // Successfully removed the entry.
                        UpdateStatus::Replaced(_entry) => {
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

                            // Note that `entry_ref` here is the entry that we just replaced.
                            return Ok(Some((&entry_ref.key, &entry_ref.value)));
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
            table = match self.prepare_retry(copying, &mut help_copy, table, guard) {
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
        table: Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> Option<Table<Entry<K, V>>> {
        let next_table = match self.resize {
            ResizeMode::Blocking => match copying {
                // The entry we want to perform the operation on is being copied.
                //
                // In blocking mode we must complete the resize before proceeding.
                Some(_) => self.help_copy(true, &table, guard),

                // If we went over the probe limit, the key is not in the map.
                None => return None,
            },

            ResizeMode::Incremental(_) => {
                // In incremental resize mode, we always have to check the next table.
                let next_table = table.next_table()?;

                // Help out with the copy.
                if *help_copy {
                    self.help_copy(false, &table, guard);
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
        new_entry: *mut Entry<K, V>,
        table: Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> InsertStatus<K, V> {
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
                // Safety: We performed a protected load of the pointer using a verified guard
                // with `Acquire` and ensured that it is non-null, meaning it is valid for reads
                // as long as we hold the guard.
                let key = unsafe { &(*found.ptr).key };

                // An entry was inserted, we have to hash it to get the metadata.
                //
                // The logic is the same for copied entries here as we have to
                // check if the key matches and continue the update in the new table.
                let hash = self.hasher.hash_one(key);
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
        current: Tagged<Entry<K, V>>,
        new_entry: *mut Entry<K, V>,
        table: Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> UpdateStatus<K, V> {
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
    pub fn reserve(&self, additional: usize, guard: &impl VerifiedGuard) {
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
            table = self.help_copy(true, &table, guard);
        }
    }

    /// Remove all entries from this table.
    #[inline]
    pub fn clear(&self, guard: &impl VerifiedGuard) {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not been initialized yet.
        if table.raw.is_null() {
            return;
        }

        loop {
            // Get a clean copy of the table to delete from.
            table = self.linearize(table, guard);

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
                            Entry::TOMBSTONE,
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
            table = self.help_copy(true, &table, guard);
        }
    }

    /// Retains only the elements specified by the predicate.
    #[inline]
    pub fn retain<F>(&self, mut f: F, guard: &impl VerifiedGuard)
    where
        F: FnMut(&K, &V) -> bool,
    {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not been initialized yet.
        if table.raw.is_null() {
            return;
        }

        loop {
            // Get a clean copy of the table to delete from.
            table = self.linearize(table, guard);

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

                    // Safety: We performed a protected load of the pointer using a verified guard with
                    // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                    // as we hold the guard.
                    let entry_ref = unsafe { &*entry.ptr };

                    // Should we retain this entry?
                    if f(&entry_ref.key, &entry_ref.value) {
                        continue 'probe;
                    }

                    // Try to delete the entry.
                    //
                    // Safety: `i` is in bounds for the table length.
                    let result = unsafe {
                        table.entry(i).compare_exchange(
                            entry.raw,
                            Entry::TOMBSTONE,
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
            table = self.help_copy(true, &table, guard);
        }
    }

    /// Returns an iterator over the keys and values of this table.
    #[inline]
    pub fn iter<'g, G>(&self, guard: &'g G) -> Iter<'g, K, V, G>
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
                _entries: PhantomData,
            };
        }

        // Get a clean copy of the table to iterate over.
        let table = self.linearize(root, guard);

        Iter {
            i: 0,
            guard,
            table,
            _entries: PhantomData,
        }
    }

    /// Returns an iterator over the keys and values of this table.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        // Safety: The root table is either null or a valid table allocation.
        let table = unsafe { Table::from_raw(*self.table.get_mut()) };

        // The table has not been initialized yet, return a dummy iterator.
        if table.raw.is_null() {
            return IterMut {
                i: 0,
                table,
                _entries: PhantomData,
            };
        }

        IterMut {
            i: 0,
            table,
            _entries: PhantomData,
        }
    }

    /// Returns the h1 and h2 hash for the given key.
    #[inline]
    fn hash<Q>(&self, key: &Q) -> (usize, u8)
    where
        Q: Hash + ?Sized,
    {
        let hash = self.hasher.hash_one(key);
        (meta::h1(hash), meta::h2(hash))
    }
}

/// A wrapper around a CAS function that manages the computed state.
struct ComputeState<F, K, V, T> {
    /// The CAS function.
    compute: F,

    /// A cached insert transition.
    insert: Option<V>,

    /// A cached update transition.
    update: Option<CachedUpdate<K, V, T>>,
}

/// A cached update transition.
struct CachedUpdate<K, V, T> {
    /// The entry that the CAS function was called with.
    input: *mut Entry<K, V>,

    /// The cached result.
    output: Operation<V, T>,
}

impl<'g, F, K, V, T> ComputeState<F, K, V, T>
where
    F: FnMut(Option<(&'g K, &'g V)>) -> Operation<V, T>,
    K: 'g,
    V: 'g,
{
    /// Create a new `ComputeState` for the given function.
    #[inline]
    fn new(compute: F) -> ComputeState<F, K, V, T> {
        ComputeState {
            compute,
            insert: None,
            update: None,
        }
    }

    /// Performs a state transition.
    ///
    /// # Safety
    ///
    /// The entry pointer must be valid for reads if provided.
    #[inline]
    unsafe fn next(&mut self, entry: Option<*mut Entry<K, V>>) -> Operation<V, T> {
        let Some(entry) = entry else {
            // If there is no current entry, perform a transition for the insert.
            return match self.insert.take() {
                // Use the cached insert.
                Some(value) => Operation::Insert(value),

                // Otherwise, compute the value to insert.
                None => (self.compute)(None),
            };
        };

        // Otherwise, perform an update transition.
        match self.update.take() {
            // Used the cached update if the entry has not changed.
            Some(CachedUpdate { input, output }) if input == entry => output,

            // Otherwise, compute the value to update.
            _ => {
                // Safety: The caller guarantees that `entry` is valid for reads.
                let entry_ref = unsafe { &*entry };
                (self.compute)(Some((&entry_ref.key, &entry_ref.value)))
            }
        }
    }

    /// Restores the state if an operation fails.
    ///
    /// This allows the result of the compute closure with a given input to be cached.
    /// This is useful at it avoids calling the closure multiple times if an update needs
    /// to be retried in a new table.
    ///
    /// Additionally, update and insert operations are cached separately, although this
    /// is not guaranteed in the public API. This means that internal methods can rely on
    /// `compute(None)` being called at most once.
    #[inline]
    fn restore(&mut self, input: Option<*mut Entry<K, V>>, output: Operation<V, T>) {
        match input {
            Some(input) => self.update = Some(CachedUpdate { input, output }),
            None => match output {
                Operation::Insert(value) => self.insert = Some(value),
                _ => unreachable!(),
            },
        }
    }
}

/// A lazy initialized `Entry` allocation.
enum LazyEntry<K, V> {
    /// An uninitialized entry, containing just the owned key.
    Uninit(K),

    /// An allocated entry.
    Init(*mut Entry<K, MaybeUninit<V>>),
}

impl<K, V> LazyEntry<K, V> {
    /// Returns a reference to the entry's key.
    #[inline]
    fn key(&self) -> &K {
        match self {
            LazyEntry::Uninit(key) => key,
            LazyEntry::Init(entry) => unsafe { &(**entry).key },
        }
    }

    /// Initializes the entry if it has not already been initialized, returning the pointer
    /// to the entry allocation.
    #[inline]
    fn init(&mut self) -> *mut Entry<K, MaybeUninit<V>> {
        match self {
            LazyEntry::Init(entry) => *entry,
            LazyEntry::Uninit(key) => {
                // Safety: we read the current key with `ptr::read` and overwrite the
                // state with `ptr::write`. We also make sure to abort if the allocator
                // panics, ensuring the current value is not dropped twice.
                unsafe {
                    let key = ptr::read(key);
                    let entry = panic::catch_unwind(panic::AssertUnwindSafe(|| {
                        Box::into_raw(Box::new(Entry {
                            value: MaybeUninit::uninit(),
                            key,
                        }))
                    }))
                    .unwrap_or_else(|_| std::process::abort());
                    ptr::write(self, LazyEntry::Init(entry));
                    entry
                }
            }
        }
    }
}

/// RMW operations.
impl<K, V, S> HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Tries to insert a key and value computed from a closure into the map,
    /// and returns a reference to the value that was inserted.
    #[inline]
    pub fn try_insert_with<'g, F>(
        &self,
        key: K,
        f: F,
        guard: &'g impl VerifiedGuard,
    ) -> Result<&'g V, &'g V>
    where
        F: FnOnce() -> V,
        K: 'g,
    {
        let mut f = Some(f);
        let compute = |entry| match entry {
            // There is already an existing value.
            Some((_, current)) => Operation::Abort(current),

            // Insert the initial value.
            //
            // Note that this case is guaranteed to be executed at most
            // once as insert values are cached, so this can never panic.
            None => Operation::Insert((f.take().unwrap())()),
        };

        match self.compute(key, compute, guard) {
            // Failed to insert, return the existing value.
            Compute::Aborted(current) => Err(current),

            // Successfully inserted.
            Compute::Inserted(_, value) => Ok(value),

            _ => unreachable!(),
        }
    }

    /// Returns a reference to the value corresponding to the key, or inserts a default value
    /// computed from a closure.
    #[inline]
    pub fn get_or_insert_with<'g, F>(&self, key: K, f: F, guard: &'g impl VerifiedGuard) -> &'g V
    where
        F: FnOnce() -> V,
        K: 'g,
    {
        match self.try_insert_with(key, f, guard) {
            Ok(value) => value,
            Err(value) => value,
        }
    }

    /// Updates an existing entry atomically, returning the value that was inserted.
    #[inline]
    pub fn update<'g, F>(
        &self,
        key: K,
        mut update: F,
        guard: &'g impl VerifiedGuard,
    ) -> Option<&'g V>
    where
        F: FnMut(&V) -> V,
        K: 'g,
    {
        let compute = |entry| match entry {
            // There is nothing to update.
            None => Operation::Abort(()),
            // Perform the update.
            Some((_, value)) => Operation::Insert(update(value)),
        };

        match self.compute(key, compute, guard) {
            // Return the updated value.
            Compute::Updated {
                new: (_, value), ..
            } => Some(value),

            // There was nothing to update.
            Compute::Aborted(_) => None,

            _ => unreachable!(),
        }
    }

    /// Updates an existing entry or inserts a default value computed from a closure.
    #[inline]
    pub fn update_or_insert_with<'g, U, F>(
        &self,
        key: K,
        update: U,
        f: F,
        guard: &'g impl VerifiedGuard,
    ) -> &'g V
    where
        F: FnOnce() -> V,
        U: Fn(&V) -> V,
        K: 'g,
    {
        let mut f = Some(f);
        let compute = |entry| match entry {
            // Perform the update.
            Some((_, value)) => Operation::Insert::<_, ()>(update(value)),

            // Insert the initial value.
            //
            // Note that this case is guaranteed to be executed at most
            // once as insert values are cached, so this can never panic.
            None => Operation::Insert((f.take().unwrap())()),
        };

        match self.compute(key, compute, guard) {
            // Return the updated value.
            Compute::Updated {
                new: (_, value), ..
            } => value,

            // Return the value we inserted.
            Compute::Inserted(_, value) => value,

            _ => unreachable!(),
        }
    }

    /// Update an entry with a CAS function.
    ///
    /// Note that `compute` closure is guaranteed to be called for a `None` input only once, allowing the
    /// insertion of values that cannot be cloned or reconstructed.
    #[inline]
    pub fn compute<'g, F, T>(
        &self,
        key: K,
        compute: F,
        guard: &'g impl VerifiedGuard,
    ) -> Compute<'g, K, V, T>
    where
        F: FnMut(Option<(&'g K, &'g V)>) -> Operation<V, T>,
    {
        // Lazy initialize the entry allocation.
        let mut entry = LazyEntry::Uninit(key);

        // Perform the update.
        //
        // Safety: We just allocated the entry above.
        let result = unsafe { self.compute_with(&mut entry, ComputeState::new(compute), guard) };

        // Deallocate the entry if it was not inserted.
        if matches!(result, Compute::Removed(..) | Compute::Aborted(_)) {
            if let LazyEntry::Init(entry) = entry {
                // Safety: The entry was allocated but not inserted into the map.
                let _ = unsafe { Box::from_raw(entry) };
            }
        }

        result
    }

    /// Update an entry with a CAS function.
    ///
    /// # Safety
    ///
    /// The new entry must be a valid owned pointer to insert into the map.
    #[inline]
    unsafe fn compute_with<'g, F, T>(
        &self,
        new_entry: &mut LazyEntry<K, V>,
        mut state: ComputeState<F, K, V, T>,
        guard: &'g impl VerifiedGuard,
    ) -> Compute<'g, K, V, T>
    where
        F: FnMut(Option<(&'g K, &'g V)>) -> Operation<V, T>,
    {
        // Load the root table.
        let mut table = self.root(guard);

        // The table has not yet been allocated.
        if table.raw.is_null() {
            // Compute the value to insert.
            //
            // Safety: Insert transitions are always sound.
            match unsafe { state.next(None) } {
                op @ Operation::Insert(_) => state.restore(None, op),
                Operation::Remove => panic!("Cannot remove `None` entry."),
                Operation::Abort(value) => return Compute::Aborted(value),
            }

            // Initialize the table.
            table = self.init(None);
        }

        let (h1, h2) = self.hash(new_entry.key());
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
                    // Compute the value to insert.
                    //
                    // Safety: Insert transitions are always sound.
                    let value = match unsafe { state.next(None) } {
                        Operation::Insert(value) => value,
                        Operation::Remove => panic!("Cannot remove `None` entry."),
                        Operation::Abort(value) => return Compute::Aborted(value),
                    };

                    let new_entry = new_entry.init();
                    // Safety: `new_entry` was just allocated above and is valid for writes.
                    unsafe { (*new_entry).value = MaybeUninit::new(value) }

                    // Attempt to insert.
                    //
                    // Safety: `probe.i` is always in-bounds for the table length.Additionally,
                    // `new_entry` was allocated above and never shared.
                    match unsafe { self.insert_at(probe.i, h2, new_entry.cast(), table, guard) } {
                        // Successfully inserted.
                        InsertStatus::Inserted => {
                            // Increment the table length.
                            self.count.get(guard).fetch_add(1, Ordering::Relaxed);

                            // Safety: `new_entry` was initialized above.
                            let new_ref = unsafe { &*new_entry.cast::<Entry<K, V>>() };
                            return Compute::Inserted(&new_ref.key, &new_ref.value);
                        }

                        // Lost to a concurrent insert.
                        //
                        // If the key matches, we might be able to update the value.
                        InsertStatus::Found(EntryStatus::Value(found))
                        | InsertStatus::Found(EntryStatus::Copied(found)) => {
                            // Cache the previous value
                            //
                            // Safety: `new_entry` was initialized above and was not inserted
                            // into the map.
                            let value = unsafe { (*new_entry).value.assume_init_read() };
                            state.restore(None, Operation::Insert(value));

                            found
                        }

                        // The entry was removed or invalidated.
                        InsertStatus::Found(EntryStatus::Null) => {
                            // Cache the previous value.
                            //
                            // Safety: `new_entry` was initialized above and was not inserted
                            // into the map.
                            let value = unsafe { (*new_entry).value.assume_init_read() };
                            state.restore(None, Operation::Insert(value));

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
                if unsafe { (*entry.ptr).key != *new_entry.key() } {
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
                    let failure = match unsafe { state.next(Some(entry.ptr)) } {
                        // The operation was aborted.
                        Operation::Abort(value) => return Compute::Aborted(value),

                        // Update the value.
                        Operation::Insert(value) => {
                            let new_entry = new_entry.init();

                            // Safety: `new_entry` was just allocated above and is valid for writes.
                            unsafe { (*new_entry).value = MaybeUninit::new(value) }

                            // Try to perform the update.
                            //
                            // Safety:
                            // - `probe.i` is always in-bounds for the table length
                            // - `entry` is a valid non-null entry that we found in the map.
                            // - `new_entry` was initialized above and never shared.
                            let status = unsafe {
                                self.update_at(probe.i, entry, new_entry.cast(), table, guard)
                            };

                            match status {
                                // Successfully updated.
                                UpdateStatus::Replaced(entry) => {
                                    // Safety: `entry` is a valid non-null entry that we found in the map
                                    // before replacing it.
                                    let entry_ref = unsafe { &(*entry.ptr) };

                                    // Safety: `new_entry` was initialized above.
                                    let new_ref = unsafe { &*new_entry.cast::<Entry<K, V>>() };

                                    return Compute::Updated {
                                        old: (&entry_ref.key, &entry_ref.value),
                                        new: (&new_ref.key, &new_ref.value),
                                    };
                                }

                                // The update failed.
                                failure => {
                                    // Save the previous value.
                                    //
                                    // Safety: `new_entry` was initialized above and was not inserted
                                    // into the map.
                                    let value = unsafe { (*new_entry).value.assume_init_read() };
                                    state.restore(Some(entry.ptr), Operation::Insert(value));

                                    failure
                                }
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
                                self.update_at(probe.i, entry, Entry::TOMBSTONE, table, guard)
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
                                    let entry_ref = unsafe { &(*entry.ptr) };
                                    return Compute::Removed(&entry_ref.key, &entry_ref.value);
                                }

                                // The remove failed.
                                failure => {
                                    // Save the removal operation.
                                    state.restore(Some(entry.ptr), Operation::Remove);

                                    failure
                                }
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
                            //
                            // Safety: Insert transitions are always sound.
                            match unsafe { state.next(None) } {
                                Operation::Insert(value) => {
                                    // Save the computed value.
                                    state.restore(None, Operation::Insert(value));

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
            if let Some(next_table) = self.prepare_retry(copying, &mut help_copy, table, guard) {
                table = next_table;
                continue;
            }

            // Otherwise, the key is not in the map.
            //
            // Safety: Insert transitions are always sound.
            match unsafe { state.next(None) } {
                // Need to insert into the new table.
                op @ Operation::Insert(_) => {
                    table = self.prepare_retry_insert(None, &mut help_copy, table, guard);
                    state.restore(None, op);
                }
                // The operation was aborted.
                Operation::Abort(value) => return Compute::Aborted(value),
                Operation::Remove => panic!("Cannot remove `None` entry."),
            }
        }
    }
}

/// Resize operations.
impl<K, V, S> HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
    /// Allocate the initial table.
    #[cold]
    #[inline(never)]
    fn init(&self, capacity: Option<usize>) -> Table<Entry<K, V>> {
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
    fn get_or_alloc_next(
        &self,
        capacity: Option<usize>,
        table: Table<Entry<K, V>>,
    ) -> Table<Entry<K, V>> {
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
        table: &Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> Table<Entry<K, V>> {
        match self.resize {
            ResizeMode::Blocking => self.help_copy_blocking(table, guard),
            ResizeMode::Incremental(chunk) => {
                let copied_to = self.help_copy_incremental(chunk, copy_all, guard);

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
        table: &Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> Table<Entry<K, V>> {
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
                    if unsafe { !self.copy_at_blocking(i, table, &next, guard) } {
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
        table: &Table<Entry<K, V>>,
        next_table: &Table<Entry<K, V>>,
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
            .unpack();

        // The entry is a tombstone.
        if entry.raw == Entry::TOMBSTONE {
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
            self.insert_copy(entry.ptr.unpack(), false, next_table, guard)
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
        guard: &impl VerifiedGuard,
    ) -> Table<Entry<K, V>> {
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
                    unsafe { self.copy_at_incremental(i, &table, &next, guard) };
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
        table: &Table<Entry<K, V>>,
        next_table: &Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) {
        // Safety: The caller guarantees that the index is in-bounds.
        let entry = unsafe { table.entry(i) };

        // Mark the entry as copying.
        let found = entry.fetch_or(Entry::COPYING, Ordering::AcqRel).unpack();

        // The entry is a tombstone.
        if found.raw == Entry::TOMBSTONE {
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
            self.insert_copy(new_entry, true, next_table, guard)
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
        new_entry: Tagged<Entry<K, V>>,
        resize: bool,
        table: &Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> Option<(Table<Entry<K, V>>, usize)> {
        // Safety: The new entry is guaranteed to be valid for reads.
        let key = unsafe { &(*new_entry.ptr).key };

        let mut table = *table;
        let (h1, h2) = self.hash(key);

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
                            let found = found.unpack();

                            // The entry was deleted or copied.
                            let meta = if found.ptr.is_null() {
                                meta::TOMBSTONE
                            } else {
                                // Safety: We performed a protected load of the pointer using a verified guard with
                                // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
                                // as we hold the guard.
                                let found_ref = unsafe { &(*found.ptr) };

                                // Ensure the meta table is updated to avoid breaking the probe chain.
                                let hash = self.hasher.hash_one(&found_ref.key);
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
        table: &Table<Entry<K, V>>,
        next: &Table<Entry<K, V>>,
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
                            drop_table(Table::from_raw(table), collector);
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
        mut table: Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) -> Table<Entry<K, V>> {
        if self.is_incremental() {
            // If we're in incremental resize mode, we need to complete any in-progress resizes to
            // ensure we don't miss any entries in the next table. We can't iterate over both because
            // we risk returning the same entry twice.
            while table.next_table().is_some() {
                table = self.help_copy(true, &table, guard);
            }
        }

        table
    }

    // Wait for an incremental copy of a given entry to complete.
    #[cold]
    #[inline(never)]
    fn wait_copied(&self, i: usize, table: &Table<Entry<K, V>>) {
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
            let entry = entry.load(Ordering::Acquire).unpack();
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
        entry: Tagged<Entry<K, V>>,
        table: &Table<Entry<K, V>>,
        guard: &impl VerifiedGuard,
    ) {
        match self.resize {
            // Safety: In blocking resize mode, we only ever write to the root table, so the entry
            // is inaccessible from all tables.
            ResizeMode::Blocking => unsafe {
                guard.defer_retire(entry.ptr, seize::reclaim::boxed);
            },
            // In incremental resize mode, the entry may be accessible in previous tables.
            ResizeMode::Incremental(_) => {
                if entry.tag() & Entry::BORROWED == 0 {
                    // Safety: If the entry is not borrowed, meaning it is not in any previous tables,
                    // it is inaccessible even if the current table is not root. Thus we can safely retire.
                    unsafe { guard.defer_retire(entry.ptr, seize::reclaim::boxed) };
                    return;
                }

                let root = self.root(guard);

                // Check if our table, or any subsequent table, is the root.
                let mut next = Some(*table);
                while let Some(table) = next {
                    if table.raw == root.raw {
                        // Safety: The root table is our table or a table that succeeds ours.
                        // Thus any previous tables are unreachable from the root, so we can safely retire.
                        unsafe { guard.defer_retire(entry.ptr, seize::reclaim::boxed) };
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
pub struct Iter<'g, K, V, G> {
    i: usize,
    table: Table<Entry<K, V>>,
    guard: &'g G,
    _entries: PhantomData<(&'g K, &'g V)>,
}

impl<'g, K: 'g, V: 'g, G> Iterator for Iter<'g, K, V, G>
where
    G: VerifiedGuard,
{
    type Item = (&'g K, &'g V);

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
                .unpack();

            // The entry was deleted.
            if entry.ptr.is_null() {
                self.i += 1;
                continue;
            }

            // Safety: We performed a protected load of the pointer using a verified guard with
            // `Acquire` and ensured that it is non-null, meaning it is valid for reads as long
            // as we hold the guard.
            let entry_ref = unsafe { &(*entry.ptr) };

            self.i += 1;
            return Some((&entry_ref.key, &entry_ref.value));
        }
    }
}

// Safety: An iterator holds a shared reference to the `HashMap`
// and `Guard`, and outputs shared references to keys and values.
// Thus everything must be `Sync` for the iterator to be `Send`
// or `Sync`.
//
// It is not possible to obtain an owned key, value, or guard
// from an iterator, so `Send` is not a required bound.
unsafe impl<K, V, G> Send for Iter<'_, K, V, G>
where
    K: Sync,
    V: Sync,
    G: Sync,
{
}

unsafe impl<K, V, G> Sync for Iter<'_, K, V, G>
where
    K: Sync,
    V: Sync,
    G: Sync,
{
}

impl<K, V, G> Clone for Iter<'_, K, V, G> {
    #[inline]
    fn clone(&self) -> Self {
        Iter {
            i: self.i,
            table: self.table,
            guard: self.guard,
            _entries: PhantomData,
        }
    }
}

// A mutable iterator over the keys and values of this table.
pub struct IterMut<'map, K, V> {
    i: usize,
    table: Table<Entry<K, V>>,
    // Ensure invariance with respect to `V`.
    _entries: PhantomData<(&'map K, &'map mut V)>,
}

impl<'map, K, V> IterMut<'map, K, V> {
    #[inline]
    pub fn next(&mut self) -> Option<(&'map K, &'map mut V)> {
        // The table has not yet been allocated.
        if self.table.raw.is_null() {
            return None;
        }

        loop {
            // Iterated over every entry in the table, proceed to a nested resize if there is one.
            //
            // We have a mutable reference to the table, so there are no concurrent removals that
            // can lead to us yielding duplicate entries.
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
            let entry = unsafe { self.table.entry_mut(self.i) }.unpack();

            // The entry was deleted.
            if entry.ptr.is_null() {
                self.i += 1;
                continue;
            }

            // The entry was copied, we'll yield it when iterating over the table
            // it was copied to.
            if entry.tag() & Entry::COPIED != 0 {
                self.i += 1;
                continue;
            }

            // Safety: We have `&mut self` and ensured the entry is non-null.
            let entry_ref = unsafe { &mut (*entry.ptr) };

            self.i += 1;
            return Some((&entry_ref.key, &mut entry_ref.value));
        }
    }

    // Returns a clone of the iterator over the remaining entries.
    //
    // # Safety
    //
    // Creating multiple clones can lead to unsound mutable aliasing.
    pub(crate) unsafe fn clone(&self) -> IterMut<'_, K, V> {
        IterMut {
            i: self.i,
            table: self.table,
            _entries: PhantomData,
        }
    }
}

// Safety: An iterator holds a mutable reference to the `HashMap`
// and does not perform any concurrent access, so the normal `Send`
// and `Sync` rules apply.
unsafe impl<K, V> Send for IterMut<'_, K, V>
where
    K: Send,
    V: Send,
{
}

unsafe impl<K, V> Sync for IterMut<'_, K, V>
where
    K: Sync,
    V: Sync,
{
}

impl<K, V, S> Drop for HashMap<K, V, S> {
    fn drop(&mut self) {
        let mut raw = *self.table.get_mut();

        // Make sure all objects are reclaimed before the collector is dropped.
        //
        // Dropping a table depends on accessing the collector for deferred retirement,
        // using the shared collector pointer that is invalidated by drop.
        //
        // Safety: We have a unique reference to the collector.
        unsafe { self.collector.reclaim_all() };

        // Drop all nested tables and entries.
        while !raw.is_null() {
            // Safety: The root and next tables are always valid pointers to a
            // table allocation, or null.
            let mut table = unsafe { Table::from_raw(raw) };

            // Read the next table pointer before dropping the current one.
            let next = *table.state_mut().next.get_mut();

            // Safety: We have unique access to the table and do
            // not access the entries after this call.
            unsafe { drop_entries(table) };

            // Safety: We have unique access to the table and do
            // not access it after this call.
            unsafe { drop_table(table, &self.collector) };

            // Continue for all nested tables.
            raw = next;
        }
    }
}

// Drop all entries in this table.
//
// # Safety
//
// The table entries must not be accessed after this call.
unsafe fn drop_entries<K, V>(table: Table<Entry<K, V>>) {
    for i in 0..table.len() {
        // Safety: `i` is in-bounds and we have unique access to the table.
        let entry = unsafe { (*table.entry(i).as_ptr()).unpack() };

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
        unsafe { drop(Box::from_raw(entry.ptr)) }
    }
}

// Drop the table allocation.
//
// # Safety
//
// The table must not be accessed after this call.
unsafe fn drop_table<K, V>(mut table: Table<Entry<K, V>>, collector: &Collector) {
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
        .drain(|entry| unsafe { collector.retire(entry, seize::reclaim::boxed) });

    // Deallocate the table.
    //
    // Safety: The caller guarantees that the table will not be accessed after this call.
    unsafe { Table::dealloc(table) };
}

// Entry metadata, inspired by `hashbrown`.
mod meta {
    use std::mem;

    // Indicates an empty entry.
    pub const EMPTY: u8 = 0x80;

    // Indicates an entry that has been deleted.
    pub const TOMBSTONE: u8 = u8::MAX;

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
