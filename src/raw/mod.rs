mod alloc;
mod utils;

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{fence, AtomicPtr, Ordering};
use std::{hint, ptr};

use self::alloc::{RawTable, State};
use self::utils::{AtomicPtrFetchOps, Counter, StrictProvenance, Tagged};
use super::map::ResizeMode;

use seize::{AsLink, Collector, Guard, Link};

// A lock-free hash-map.
pub struct HashMap<K, V, S> {
    pub collector: Collector,
    pub hasher: S,
    pub resize: ResizeMode,
    table: AtomicPtr<RawTable>,
    count: Counter,
    _kv: PhantomData<(K, V)>,
}

// The hash-table allocation.
type Table<K, V> = self::alloc::Table<Entry<K, V>>;

// An entry in the hash-table.
#[repr(C)]
pub struct Entry<K, V> {
    pub link: Link,
    pub key: K,
    pub value: V,
}

// The state of an entry we attempted to update.
pub enum EntryStatus<'g, V> {
    Empty(&'g V),
    Tombstone(&'g V),
    Replaced(&'g V),
    Error { current: &'g V, not_inserted: V },
}

/// The state of an entry we attempted to replace.
enum ReplaceStatus<V> {
    HelpCopy,
    Removed,
    Replaced(V),
}

// Safety: repr(C) and seize::Link is the first field
unsafe impl<K, V> AsLink for Entry<K, V> {}

impl Entry<(), ()> {
    // The entry is being copied to the new table, no updates are allowed on the old table.
    //
    // In blocking mode, the COPYING bit is put down to initiate a copy, forcing all writers
    // to complete the resize before making progress. Readers can ignore this.
    //
    // In incremental mode, the COPYING bit is put down after a copy completes. Both readers and
    // writers must go to the new table to see the new state of the entry.
    const COPYING: usize = 0b001;

    // The entry was copied from a previous table.
    //
    // A borrowed entry may still be accessible from previous tables if the resize has not been
    // promoted, and so are unsafe to drop.
    //
    // This is only set in incremental mode.
    const BORROWED: usize = 0b010;

    // The entry was deleted.
    const TOMBSTONE: usize = 0b100;

    // Mask for entry pointer, ignoring tag bits.
    const MASK: usize = !(Entry::COPYING | Entry::BORROWED | Entry::TOMBSTONE);

    // Retires an entry.
    unsafe fn retire<K, V>(link: *mut Link) {
        let entry: *mut Entry<K, V> = link.cast();
        let _entry = unsafe { Box::from_raw(entry) };
    }
}

impl<K, V, S> HashMap<K, V, S> {
    // Creates a table with the given capacity and hasher.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> HashMap<K, V, S> {
        let collector = Collector::new().epoch_frequency(None);

        // the table is lazily allocated
        if capacity == 0 {
            return HashMap {
                collector,
                hasher: hash_builder,
                resize: ResizeMode::Blocking,
                table: AtomicPtr::new(ptr::null_mut()),
                count: Counter::default(),
                _kv: PhantomData,
            };
        }

        let mut table = Table::<K, V>::new(entries_for(capacity), collector.link());

        // mark this table as the root
        *table.state_mut().status.get_mut() = State::PROMOTED;

        HashMap {
            collector,
            hasher: hash_builder,
            resize: ResizeMode::Blocking,
            table: AtomicPtr::new(table.raw),
            count: Counter::default(),
            _kv: PhantomData,
        }
    }

    // Returns a reference to the root hash table.
    #[inline(always)]
    pub fn root<'g>(&'g self, guard: &'g impl Guard) -> HashMapRef<'g, K, V, S> {
        assert!(
            guard.belongs_to(&self.collector),
            "accessed map with incorrect guard"
        );

        let raw = guard.protect(&self.table, Ordering::Acquire);
        let table = unsafe { Table::<K, V>::from_raw(raw) };
        self.as_ref(table)
    }

    // Returns a pointer to the root table.
    pub fn root_ptr(&self) -> *mut u8 {
        self.table.load(Ordering::Relaxed).cast()
    }

    // Returns the number of entries in the table.
    pub fn len(&self) -> usize {
        self.count.active()
    }

    // Returns true if incremental resizing is enabled.
    #[inline(always)]
    fn is_incremental(&self) -> bool {
        matches!(self.resize, ResizeMode::Incremental(_))
    }

    // Returns true if blocking resizing is enabled.
    #[inline(always)]
    fn is_blocking(&self) -> bool {
        matches!(self.resize, ResizeMode::Blocking)
    }

    // Returns a reference to the given table.
    fn as_ref(&self, table: Table<K, V>) -> HashMapRef<'_, K, V, S> {
        HashMapRef { table, root: self }
    }
}

// A reference to the root table, or an arbitrarily nested table migration.
pub struct HashMapRef<'a, K, V, S> {
    table: Table<K, V>,
    root: &'a HashMap<K, V, S>,
}

impl<K, V, S> HashMapRef<'_, K, V, S>
where
    K: Sync + Send + Hash + Eq,
    V: Sync + Send,
    S: BuildHasher,
{
    // Returns a reference to the value corresponding to the key.
    pub fn get_entry<'g, Q>(&self, key: &Q, guard: &'g impl Guard) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.table.raw.is_null() {
            return None;
        }

        let hash = self.root.hasher.hash_one(key);
        let h2 = meta::h2(hash);

        let start = h1(hash) & (self.table.len - 1);
        let limit = probe_limit!(self.table.len);
        let mut probe = Probe::start();

        loop {
            let i = (start + probe.i) & (self.table.len - 1);
            if probe.length > limit {
                break;
            }

            let meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            // the entry is not in the table. it cannot be in the next table either
            // because we have not went over the probe limit
            if meta == meta::EMPTY {
                return None;
            }

            // potential match
            if meta == h2 {
                let entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) }
                    .unpack(Entry::MASK);

                // the entry was deleted
                if entry.addr & Entry::TOMBSTONE != 0 {
                    probe.next();
                    continue;
                }

                // check for a full match
                if unsafe { (*entry.ptr).key.borrow() } == key {
                    // the entry was copied to the new table
                    //
                    // in blocking mode we can ignore this, because writes block until resizes
                    // complete, so the root table is always the source of truth
                    if self.root.is_incremental() && entry.addr & Entry::COPYING != 0 {
                        break;
                    }

                    return unsafe { Some((&(*entry.ptr).key, &(*entry.ptr).value)) };
                }
            }

            probe.next();
        }

        // in incremental mode, we have to check the next table if we found a
        // copied entry or went over the probe limit
        if self.root.is_incremental() {
            if let Some(next) = self.next_table_ref() {
                return next.get_entry(key, guard);
            }
        }

        None
    }

    // Inserts a key-value pair into the map.
    pub fn insert<'g>(
        &mut self,
        key: K,
        value: V,
        replace: bool,
        guard: &'g impl Guard,
    ) -> EntryStatus<'g, V> {
        let entry = Box::into_raw(Box::new(Entry {
            key,
            value,
            link: self.root.collector.link(),
        }));

        let result = self.insert_entry(entry, replace, true, guard);

        // update the length if we inserted a new entry
        if matches!(result, EntryStatus::Empty(_)) {
            self.root
                .count
                .get(guard.thread_id())
                .fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    // Inserts an entry into the map.
    fn insert_entry<'g>(
        &mut self,
        new_entry: *mut Entry<K, V>,
        replace: bool,
        help_copy: bool,
        guard: &'g impl Guard,
    ) -> EntryStatus<'g, V> {
        if self.table.raw.is_null() {
            self.init(None);
        }

        let new_ref = unsafe { &*new_entry };

        let hash = self.root.hasher.hash_one(&new_ref.key);
        let h2 = meta::h2(hash);

        let start = h1(hash) & (self.table.len - 1);
        let limit = probe_limit!(self.table.len);
        let mut probe = Probe::start();

        loop {
            let i = (start + probe.i) & (self.table.len - 1);
            if probe.length > limit {
                break;
            }

            let meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            if meta == meta::EMPTY {
                let entry = unsafe { self.table.entry(i) };

                match entry.compare_exchange(
                    ptr::null_mut(),
                    new_entry,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    // successfully claimed this entry
                    Ok(_) => {
                        unsafe { self.table.meta(i).store(h2, Ordering::Release) };
                        return EntryStatus::Empty(&new_ref.value);
                    }
                    // lost to a concurrent insert
                    Err(found) => {
                        fence(Ordering::Acquire);

                        let found = found.unpack(Entry::MASK);

                        // ensure the meta table is updated to keep the probe chain alive
                        unsafe {
                            // the entry was deleted or copied
                            let meta = if found.ptr.is_null() {
                                // even if an empty entry was copied, we still mark it as a
                                // tombstone so readers know to continue
                                meta::TOMBSTONE
                            } else {
                                // make sure the meta is visible to readers
                                let hash = self.root.hasher.hash_one(&(*found.ptr).key);
                                meta::h2(hash)
                            };

                            if self.table.meta(i).load(Ordering::Relaxed) == meta::EMPTY {
                                self.table.meta(i).store(meta, Ordering::Release);
                            }
                        }

                        // the entry was deleted or copied
                        if found.ptr.is_null() {
                            probe.next();
                            continue;
                        }

                        // if the same key was just inserted, we might be able to update
                        if unsafe { (*found.ptr).key == new_ref.key } {
                            // we aren't replacing exist values, return an error
                            if !replace {
                                let new_entry = unsafe { Box::from_raw(new_entry) };
                                let current = unsafe { &(*found.ptr).value };

                                return EntryStatus::Error {
                                    current,
                                    not_inserted: new_entry.value,
                                };
                            }

                            match self.replace_entry(i, found, new_entry, guard) {
                                // the entry was deleted before we could update it, keep probing
                                ReplaceStatus::Removed => unsafe {
                                    self.table.meta(i).store(meta::TOMBSTONE, Ordering::Release);
                                },
                                // the entry is being copied
                                ReplaceStatus::HelpCopy => break,
                                // successful update
                                ReplaceStatus::Replaced(value) => {
                                    return EntryStatus::Replaced(value)
                                }
                            }
                        }

                        probe.next();
                        continue;
                    }
                }
            }

            // potential match
            if meta == h2 {
                let entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) }
                    .unpack(Entry::MASK);

                // the entry was deleted
                if entry.addr & Entry::TOMBSTONE != 0 {
                    probe.next();
                    continue;
                }

                // if the key matches, we might be able to update
                if unsafe { (*entry.ptr).key == new_ref.key } {
                    // we aren't replacing exist values, return an error
                    if !replace {
                        let new_entry = unsafe { Box::from_raw(new_entry) };
                        let current = unsafe { &(*entry.ptr).value };

                        return EntryStatus::Error {
                            current,
                            not_inserted: new_entry.value,
                        };
                    }

                    match self.replace_entry(i, entry, new_entry, guard) {
                        // the entry was deleted before we could update it, keep probing
                        ReplaceStatus::Removed => {}
                        // the entry is being copied
                        ReplaceStatus::HelpCopy => break,
                        // successful update
                        ReplaceStatus::Replaced(value) => return EntryStatus::Replaced(value),
                    }
                }
            }

            probe.next();
        }

        // went over the probe limit or found a copied entry, trigger a resize
        let mut next_table = self.get_or_alloc_next(None);

        // help out with the resize
        //
        // in blocking mode we always have to complete the copy before retrying
        // the update in the new root table
        //
        // in incremental mode we can insert into the next table because we went over
        // the probe limit or saw a copied entry, so no writers can ever insert this key into
        // the old table and reads will also check the next table
        if self.root.is_blocking() || help_copy {
            next_table = self.help_copy(guard, false);
        }

        // insert into the next table
        //
        // make sure not to help copy again in incremental mode to keep resizing costs constant
        self.as_ref(next_table)
            .insert_entry(new_entry, replace, false, guard)
    }

    // Replaces the value of an existing entry, returning the previous value if successful.
    //
    // Inserts into the new table if the entry is being copied.
    fn replace_entry<'g>(
        &self,
        i: usize,
        mut entry: Tagged<*mut Entry<K, V>>,
        new_entry: *mut Entry<K, V>,
        guard: &'g impl Guard,
    ) -> ReplaceStatus<&'g V> {
        loop {
            // the entry is being copied to a new table, we have to retry there
            if entry.addr & Entry::COPYING != 0 {
                return ReplaceStatus::HelpCopy;
            }

            match unsafe { self.table.entry(i) }.compare_exchange_weak(
                entry.raw,
                new_entry,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                // succesful update
                Ok(_) => unsafe {
                    // safety: the old value is now unreachable in this table
                    self.defer_retire(entry, guard);
                    return ReplaceStatus::Replaced(&(*entry.ptr).value);
                },

                // lost to a delete
                Err(found) if found.addr() & Entry::TOMBSTONE != 0 => {
                    return ReplaceStatus::Removed;
                }

                // lost to a concurrent update, retry
                Err(found) => entry = found.unpack(Entry::MASK),
            }
        }
    }

    unsafe fn defer_retire(&self, entry: Tagged<*mut Entry<K, V>>, guard: &impl Guard) {
        match self.root.resize {
            // safety: in blocking mode, we only ever write to the root table, so the entry
            // is already inaccessible. any threads that enter after we retire will load
            // the root table
            ResizeMode::Blocking => unsafe {
                guard.defer_retire(entry.ptr, Entry::retire::<K, V>);
            },
            // in incremental mode, the entry may be accessible in previous tables if the
            // current table is not the root
            ResizeMode::Incremental(_) => {
                if entry.addr & Entry::BORROWED == 0 {
                    // this entry is not borrowed, meaning it is not in any previous tables,
                    // so it is inaccessible even if we are not the root
                    unsafe { guard.defer_retire(entry.ptr, Entry::retire::<K, V>) };
                    return;
                }

                // todo: we can avoid reloading the root here if this was called on a
                // table loaded from the root, the common case
                let root = self.root.table.load(Ordering::Relaxed);

                let mut next = Some(self.clone());
                while let Some(map) = next {
                    if map.table.raw == root {
                        // safety: if the root is our table or any table that precedes ours,
                        // the entry is not accessible from the root
                        unsafe { guard.defer_retire(entry.ptr, Entry::retire::<K, V>) };
                        return;
                    }

                    next = map.next_table_ref();
                }

                // if not, we have to wait for the previous table to be retired before
                // dropping this entry
                let mut prev = self.as_ref(unsafe { Table::<K, V>::from_raw(root) });
                loop {
                    let next = prev.next_table_ref().unwrap();

                    if next.table.raw == self.table.raw {
                        // defer this entry
                        let mut deferred = prev.table.state().deferred.lock().unwrap();
                        deferred.push(entry.ptr.cast());
                        return;
                    }

                    prev = next;
                }
            }
        }
    }

    // Update an entry with a remapping function.
    pub fn update<'g, F>(&self, key: K, f: F, guard: &'g impl Guard) -> Option<&'g V>
    where
        F: Fn(&V) -> V,
    {
        if self.table.raw.is_null() {
            return None;
        }

        let update = Box::into_raw(Box::new(Entry {
            key,
            link: self.root.collector.link(),
            value: MaybeUninit::uninit(),
        }));

        self.update_with(update, f, true, guard)
    }

    // Update an entry with a remapping function.
    pub fn update_with<'g, F>(
        &self,
        new_entry: *mut Entry<K, MaybeUninit<V>>,
        update: F,
        help_copy: bool,
        guard: &'g impl Guard,
    ) -> Option<&'g V>
    where
        F: Fn(&V) -> V,
    {
        let hash = unsafe { self.root.hasher.hash_one(&(*new_entry).key) };
        let h2 = meta::h2(hash);

        let start = h1(hash) & (self.table.len - 1);
        let limit = probe_limit!(self.table.len);
        let mut probe = Probe::start();

        let copying = 'probe: loop {
            let i = (start + probe.i) & (self.table.len - 1);
            if probe.length > limit {
                break false;
            }

            let meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            // the entry is not in the table. it cannot be in the next table either
            // because we have not went over the probe limit
            if meta == meta::EMPTY {
                return None;
            }

            // potential match
            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) }
                    .unpack(Entry::MASK);

                // the entry was deleted
                if entry.addr & Entry::TOMBSTONE != 0 {
                    probe.next();
                    continue;
                }

                // the key matches, we might be able to perform an update
                if unsafe { (*entry.ptr).key == (*new_entry).key } {
                    loop {
                        // the entry is being copied to a new table, we have to copy it before we can update it
                        if entry.addr & Entry::COPYING != 0 {
                            break 'probe true;
                        }

                        // construct the new value
                        unsafe {
                            let value = update(&(*entry.ptr).value);
                            (*new_entry).value = MaybeUninit::new(value);
                        }

                        match unsafe { self.table.entry(i) }.compare_exchange_weak(
                            entry.raw,
                            new_entry as _,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            // succesful update
                            Ok(_) => unsafe {
                                // safety: the old value is now unreachable in this table
                                self.defer_retire(entry, guard);
                                return Some((*new_entry).value.assume_init_ref());
                            },

                            // the entry got deleted
                            //
                            // we can return directly here because we saw the entry was in
                            // this table at some point, and then got deleted
                            Err(found) if found.addr() & Entry::TOMBSTONE != 0 => {
                                return None;
                            }

                            // lost to a concurrent update or delete, retry
                            Err(found) => {
                                // drop the old value
                                unsafe { (*new_entry).value.assume_init_drop() }
                                entry = found.unpack(Entry::MASK);
                            }
                        }
                    }
                }
            }

            probe.next();
        };

        // the entry we want to update is being copied
        if copying {
            let mut next_table = self.next_table_ref().unwrap().table;

            // in blocking mode we always have to complete the copy before retrying
            // the update in the new root table
            if self.root.is_blocking() || help_copy {
                next_table = self.help_copy(guard, false);
            }

            // retry in the new table
            return self
                .as_ref(next_table)
                .update_with(new_entry, update, false, guard);
        }

        // in incremental mode, the entry might be in the next table if we went over the probe
        // limit
        if self.root.is_incremental() {
            if let Some(next_table) = self.next_table_ref() {
                if help_copy {
                    self.help_copy(guard, false);
                }

                return next_table.update_with(new_entry, update, false, guard);
            }
        }

        None
    }

    // Reserve capacity for `additional` more elements.
    pub fn reserve(&mut self, additional: usize, guard: &impl Guard) {
        if self.table.raw.is_null() && self.init(Some(entries_for(additional))) {
            return;
        }

        loop {
            let capacity = entries_for(self.root.count.active() + additional);

            // we have enough capacity
            if self.table.len >= capacity {
                return;
            }

            self.get_or_alloc_next(Some(capacity));
            self.table = self.help_copy(guard, true);
        }
    }

    // Allocate the inital table.
    fn init(&mut self, capacity: Option<usize>) -> bool {
        const CAPACITY: usize = 32;

        let mut table =
            Table::<K, V>::new(capacity.unwrap_or(CAPACITY), self.root.collector.link());
        // mark this table as the root
        *table.state_mut().status.get_mut() = State::PROMOTED;

        match self.root.table.compare_exchange(
            ptr::null_mut(),
            table.raw,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {
                self.table = table;
                true
            }

            // someone us allocated before us, deallocate our table
            Err(found) => {
                unsafe { Table::dealloc(table) }
                self.table = unsafe { Table::from_raw(found) };
                false
            }
        }
    }

    // Help along with an existing resize operation.
    fn help_copy(&self, guard: &impl Guard, copy_all: bool) -> Table<K, V> {
        match self.root.resize {
            ResizeMode::Blocking => self.help_copy_blocking(guard),
            ResizeMode::Incremental(chunk) => {
                self.help_copy_incremental(chunk, copy_all, guard);
                return self.next_table_ref().unwrap().table;
            }
        }
    }

    // Help along the resize operation until it completes and the next table is promoted.
    //
    // Should only be called on the root table.
    fn help_copy_blocking(&self, guard: &impl Guard) -> Table<K, V> {
        // make sure we are copying from the root (or ex-root) table
        debug_assert_eq!(
            self.table.state().status.load(Ordering::Relaxed),
            State::PROMOTED
        );

        let next = self.table.state().next.load(Ordering::Acquire);
        debug_assert!(!next.is_null());
        let mut next = unsafe { Table::<K, V>::from_raw(next) };

        'copy: loop {
            // make sure we are copying to the correct table
            while next.state().status.load(Ordering::Acquire) == State::ABORTED {
                next = self.as_ref(next).get_or_alloc_next(None);
            }

            // the copy already completed
            if self.try_promote(next, 0, guard) {
                return next;
            }

            let copy_chunk = self.table.len.min(4096);

            loop {
                // every entry has already been claimed
                if next.state().claim.load(Ordering::Acquire) >= self.table.len {
                    break;
                }

                // claim a range to copy
                let copy_start = next.state().claim.fetch_add(copy_chunk, Ordering::AcqRel);

                let mut copied = 0;
                for i in 0..copy_chunk {
                    let i = copy_start + i;

                    if i >= self.table.len {
                        break;
                    }

                    // if this table doesn't have space, we have to abort the resize and allocate a
                    // new table
                    if !self.copy_at_blocking(i, next) {
                        // allocate the next table
                        let allocated = self.as_ref(next).get_or_alloc_next(None);

                        // abort the current resize
                        next.state().status.store(State::ABORTED, Ordering::Release);
                        atomic_wait::wake_all(&next.state().status);

                        // retry in a new table
                        next = allocated;
                        continue 'copy;
                    }

                    copied += 1;
                }

                // are we done?
                if self.try_promote(next, copied, guard) {
                    return next;
                }

                // if the resize was aborted while we were copying, continue in the new table
                if next.state().status.load(Ordering::Acquire) == State::ABORTED {
                    continue 'copy;
                }
            }

            // we copied all that we can, wait for the table to be promoted
            for spun in 0.. {
                const SPIN_WAIT: usize = 7;

                let status = next.state().status.load(Ordering::Acquire);

                // if this copy was aborted, we have to retry in the new table
                if status == State::ABORTED {
                    continue 'copy;
                }

                // the copy is complete
                if status == State::PROMOTED {
                    return next;
                }

                // copy chunks are relatively small and we expect to finish quickly,
                // so spin for a bit before resorting to parking
                if spun <= SPIN_WAIT {
                    for _ in 0..(spun * spun) {
                        hint::spin_loop();
                    }

                    continue;
                }

                atomic_wait::wait(&next.state().status, State::PENDING);
            }
        }
    }

    // Copy the entry at the given index to the new table.
    //
    // Returns `true` if the entry was copied into the table, or `false` if
    // the table was full.
    fn copy_at_blocking(&self, i: usize, next_table: Table<K, V>) -> bool {
        // mark the entry as copying
        let entry = unsafe {
            self.table
                .entry(i)
                .fetch_or(Entry::COPYING, Ordering::AcqRel)
                .unpack(Entry::MASK)
        };

        // there is nothing to copy
        if entry.ptr.is_null() {
            unsafe { self.table.meta(i).store(meta::TOMBSTONE, Ordering::Release) };
            return true;
        }

        // otherwise, copy the value
        self.as_ref(next_table)
            .insert_copy(entry.ptr.unpack(Entry::MASK), true)
            .is_some()
    }

    // Help along an in-progress resize incrementally by copying a chunk of entries.
    fn help_copy_incremental(&self, chunk: usize, block: bool, guard: &impl Guard) {
        let root = self.root.root(guard);
        debug_assert!(!root.table.raw.is_null());

        // always help the highest priority root resize
        if self.table.raw != root.table.raw {
            return root.help_copy_incremental(chunk, block, guard);
        }

        let next = self.table.state().next.load(Ordering::Acquire);

        // the copy was already promoted to the root
        if next.is_null() {
            return;
        }

        let next = unsafe { Table::<K, V>::from_raw(next) };

        loop {
            // the copy already completed
            if self.try_promote(next, 0, guard) {
                return;
            }

            loop {
                // every entry has already been claimed
                if next.state().claim.load(Ordering::Acquire) >= self.table.len {
                    break;
                }

                // claim a range to copy
                let copy_start = next.state().claim.fetch_add(chunk, Ordering::AcqRel);

                let mut copied = 0;
                for i in 0..chunk {
                    let i = copy_start + i;

                    if i >= self.table.len {
                        break;
                    }

                    // copy the entry
                    self.copy_at(i, next);
                    copied += 1;
                }

                // are we done?
                if self.try_promote(next, copied, guard) || !block {
                    return;
                }
            }

            if !block {
                return;
            }

            // we need to block for the resize but all entries are claimed
            for spun in 0.. {
                const SPIN_WAIT: usize = 7;

                let status = next.state().status.load(Ordering::Acquire);

                // the copy is complete
                if status == State::PROMOTED {
                    return;
                }

                // copy chunks are relatively small and we expect to finish quickly,
                // so spin for a bit before resorting to parking
                if spun <= SPIN_WAIT {
                    for _ in 0..(spun * spun) {
                        hint::spin_loop();
                    }

                    continue;
                }

                atomic_wait::wait(&next.state().status, State::PENDING);
            }
        }
    }

    // Copy the entry at the given index to the new table.
    fn copy_at(&self, i: usize, next_table: Table<K, V>) {
        let mut entry = unsafe {
            self.table
                .entry(i)
                .load(Ordering::Acquire)
                .unpack(Entry::MASK)
        };

        // fast path, copying an empty entry
        if entry.raw.is_null() {
            match unsafe {
                self.table.entry(i).compare_exchange(
                    ptr::null_mut(),
                    Entry::COPYING as _,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
            } {
                Ok(_) => return,

                // we lost to an insert, we have to copy the entry
                Err(found) => entry = found.unpack(Entry::MASK),
            }
        }

        let mut next = None;

        loop {
            // the entry was deleted
            if entry.addr & Entry::TOMBSTONE != 0 {
                // if we already inserted into the new table, we have to delete the copy
                if let Some(next) = next {
                    unsafe {
                        // delete the entry we inserted
                        next_table
                            .entry(next)
                            .store(Entry::TOMBSTONE as _, Ordering::Release);

                        next_table
                            .meta(next)
                            .store(meta::TOMBSTONE, Ordering::Release);
                    }
                }

                unsafe {
                    // mark the entry as copied
                    //
                    // note that once a entry is deleted no one else can update it,
                    // so this always succeeds
                    self.table
                        .entry(i)
                        .store((Entry::TOMBSTONE | Entry::COPYING) as _, Ordering::Release);
                }

                return;
            }

            // copy the entry, if there is one
            match next {
                // we already inserted into the table but the entry changed,
                // so we just update the copy
                Some(next) => unsafe {
                    next_table.entry(next).store(
                        entry.ptr.map_addr(|addr| addr | Entry::BORROWED),
                        Ordering::Release,
                    )
                },
                // insert into the next table
                //
                // we are the unique copier and haven't marked the entry as
                // copied yet, so no one is trying to insert or access this
                // entry in the new table except us
                None => {
                    let ptr = entry
                        .ptr
                        .map_addr(|addr| addr | Entry::BORROWED)
                        .unpack(Entry::MASK);

                    next = Some(self.as_ref(next_table).insert_copy(ptr, false).unwrap());
                }
            }

            // try to mark the entry as copied
            match unsafe {
                self.table.entry(i).compare_exchange(
                    entry.raw,
                    entry.raw.map_addr(|addr| addr | Entry::COPYING),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
            } {
                Ok(_) => return,

                // we lost to an insert, retry
                Err(found) => {
                    entry = found.unpack(Entry::MASK);
                    continue;
                }
            }
        }
    }

    // Copy an entry into the table, returning the index it was inserted into.
    //
    // This is an optimized version of `insert_entry` where we are the only
    // writer inserting the given key into the new table
    //
    // If `abort` is `true`, this method returns `None` if the table is full.
    // Otherwise, it will recursively try to allocate and insert into a resize.
    fn insert_copy(&self, new_entry: Tagged<*mut Entry<K, V>>, abort: bool) -> Option<usize> {
        let key = unsafe { &(*new_entry.ptr).key };

        let hash = self.root.hasher.hash_one(key);

        let start = h1(hash) & (self.table.len - 1);
        let limit = probe_limit!(self.table.len);
        let mut probe = Probe::start();

        loop {
            let i = (start + probe.i) & (self.table.len - 1);
            if probe.length > limit {
                break;
            }

            let meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            if meta == meta::EMPTY {
                let entry = unsafe { self.table.entry(i) };

                // try to claim the entry
                match entry.compare_exchange(
                    ptr::null_mut(),
                    new_entry.raw,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        unsafe { self.table.meta(i).store(meta::h2(hash), Ordering::Release) };
                        return Some(i);
                    }
                    Err(found) => {
                        fence(Ordering::Acquire);
                        let found = found.unpack(Entry::MASK);

                        // ensure the meta table is updated to avoid breaking the probe chain
                        unsafe {
                            // the entry was deleted or copied (can only happen in incremental
                            // resize mode)
                            let meta = if found.ptr.is_null() {
                                meta::TOMBSTONE
                            } else {
                                let hash = self.root.hasher.hash_one(&(*found.ptr).key);
                                meta::h2(hash)
                            };

                            if self.table.meta(i).load(Ordering::Relaxed) == meta::EMPTY {
                                self.table.meta(i).store(meta, Ordering::Release);
                            }
                        }
                    }
                }
            }

            probe.next();
        }

        // the table is full, abort if necessary
        if abort {
            return None;
        }

        // otherwise we have to insert into the next table
        let next_table = self.get_or_alloc_next(None);
        self.as_ref(next_table).insert_copy(new_entry, false)
    }

    // Removes a key from the map, returning the entry for the key if the key was previously in the map.
    pub fn remove<'g, Q: ?Sized>(&self, key: &Q, guard: &'g impl Guard) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.remove_helper(key, true, guard)
    }

    // Removes a key from the map, returning the entry for the key if the key was previously in the map.
    pub fn remove_helper<'g, Q: ?Sized>(
        &self,
        key: &Q,
        help_copy: bool,
        guard: &'g impl Guard,
    ) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        if self.table.raw.is_null() {
            return None;
        }

        let hash = self.root.hasher.hash_one(key);
        let h2 = meta::h2(hash);

        let start = h1(hash) & (self.table.len - 1);
        let limit = probe_limit!(self.table.len);
        let mut probe = Probe::start();

        let copying = 'probe: loop {
            let i = (start + probe.i) & (self.table.len - 1);
            if probe.length > limit {
                break false;
            }

            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // the key is not in this table
            if meta == meta::EMPTY {
                return None;
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) }
                    .unpack(Entry::MASK);

                // the entry was deleted
                if entry.addr & Entry::TOMBSTONE != 0 {
                    probe.next();
                    continue;
                }

                // the key matches, we might be able to perform an update
                if unsafe { (*entry.ptr).key.borrow() == key } {
                    loop {
                        // the entry is being copied to a new table, we have to retry there
                        if entry.addr & Entry::COPYING != 0 {
                            break 'probe true;
                        }

                        // perform the deletion
                        match unsafe { self.table.entry(i) }.compare_exchange_weak(
                            entry.raw,
                            Entry::TOMBSTONE as _,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            // succesfully deleted
                            Ok(_) => unsafe {
                                // mark the key as a tombstone to avoid unnecessary reads
                                // note this might end up being overwritten by a slow h2 store,
                                // but we avoid the RMW and sacrifice extra reads in that extremely
                                // rare case
                                self.table.meta(i).store(meta::TOMBSTONE, Ordering::Release);

                                // safety: the old value is now unreachable in this table
                                self.defer_retire(entry, guard);

                                self.root
                                    .count
                                    .get(guard.thread_id())
                                    .fetch_sub(1, Ordering::Relaxed);

                                return Some((&(*entry.ptr).key, &(*entry.ptr).value));
                            },

                            // the entry was deleted
                            Err(found) if found.addr() & Entry::TOMBSTONE != 0 => {
                                return None;
                            }

                            // lost to a concurrent update, retry
                            Err(found) => entry = found.unpack(Entry::MASK),
                        }
                    }
                }
            }

            probe.next();
        };

        // the entry we want to update is being copied
        if copying {
            let mut next_table = self.next_table_ref().unwrap().table;

            // in blocking mode we always have to complete the copy before retrying
            // the removal in the new root table
            if self.root.is_blocking() || help_copy {
                next_table = self.help_copy(guard, false);
            }

            // retry in the new table
            return self.as_ref(next_table).remove_helper(key, false, guard);
        }

        // in incremental mode, the entry might be in the next table if we went over
        // the probe limit
        if self.root.is_incremental() {
            if let Some(next_table) = self.next_table_ref() {
                if help_copy {
                    self.help_copy(guard, false);
                }

                return next_table.remove_helper(key, false, guard);
            }
        }

        None
    }

    // Remove all entries from this table.
    pub fn clear(&self, guard: &impl Guard) {
        if self.table.raw.is_null() {
            return;
        }

        let mut copying = false;

        'probe: for i in 0..self.table.len {
            let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) }
                .unpack(Entry::MASK);

            loop {
                // a non-empty entry is being copied. clear every entry in this table that we can, then
                // deal with the copy
                if entry.addr & Entry::COPYING != 0 && !entry.ptr.is_null() {
                    copying = true;
                    continue 'probe;
                }

                // the entry is empty or already deleted
                if entry.ptr.is_null() {
                    continue 'probe;
                }

                // try to delete the entry
                let result = unsafe {
                    self.table.entry(i).compare_exchange(
                        entry.raw,
                        Entry::TOMBSTONE as _,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                };

                match result {
                    Ok(_) => unsafe {
                        self.table.meta(i).store(meta::TOMBSTONE, Ordering::Release);

                        self.root
                            .count
                            .get(guard.thread_id())
                            .fetch_sub(1, Ordering::Relaxed);

                        // safety: the old value is now unreachable in this table
                        self.defer_retire(entry, guard);
                        break;
                    },
                    // lost to a concurrent update, retry
                    Err(found) => {
                        entry = found.unpack(Entry::MASK);
                        continue;
                    }
                }
            }
        }

        // we can't delete the entries being copied until the copy is complete, so help the copy
        // and continue clearing. in incremental mode we have to check the new table anyways, but
        // we have to complete the copy to avoid deleting in-progress copies
        if copying || (self.root.is_incremental() && self.next_table_ref().is_some()) {
            let next_table = self.help_copy(guard, true);
            return self.as_ref(next_table).clear(guard);
        }
    }
}

// An iterator over the keys and values of this table.
pub struct Iter<'g, K, V, G> {
    i: usize,
    table: Table<K, V>,
    guard: &'g G,
}

impl<K, V, S> HashMapRef<'_, K, V, S>
where
    K: Sync + Send + Hash + Eq,
    V: Sync + Send,
    S: BuildHasher,
{
    // Returns an iterator over the keys and values of this table.
    pub fn iter<'g, G>(&mut self, guard: &'g G) -> Iter<'g, K, V, G>
    where
        G: Guard,
    {
        if self.table.raw.is_null() {
            return Iter {
                i: 0,
                guard,
                table: self.table,
            };
        }

        // if we're in incremental mode, we need to complete any in-progress resizes
        // to ensure we don't miss any entries in the next table. we can't iterate
        // over both because we risk returning the same entry twice.
        if self.root.is_incremental() {
            while self.next_table_ref().is_some() {
                self.table = self.help_copy(guard, true);
            }
        }

        Iter {
            i: 0,
            guard,
            table: self.table,
        }
    }
}

unsafe impl<K, V, G> Send for Iter<'_, K, V, G>
where
    K: Send,
    V: Send,
    G: Sync,
{
}

unsafe impl<K, V, G> Sync for Iter<'_, K, V, G>
where
    K: Send,
    V: Send,
    G: Sync,
{
}

impl<K, V, G> Clone for Iter<'_, K, V, G> {
    fn clone(&self) -> Self {
        Iter {
            i: self.i,
            table: self.table,
            guard: self.guard,
        }
    }
}

impl<'g, K: 'g, V: 'g, G> Iterator for Iter<'g, K, V, G>
where
    G: Guard,
{
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.table.raw.is_null() {
            return None;
        }

        loop {
            if self.i >= self.table.len {
                return None;
            }

            let meta = unsafe { self.table.meta(self.i) }.load(Ordering::Acquire);

            if matches!(meta, meta::EMPTY | meta::TOMBSTONE) {
                self.i += 1;
                continue;
            }

            let entry = unsafe {
                self.guard
                    .protect(self.table.entry(self.i), Ordering::Acquire)
                    .unpack(Entry::MASK)
            };

            if entry.addr & Entry::TOMBSTONE != 0 {
                self.i += 1;
                continue;
            }

            debug_assert!(!entry.ptr.is_null());

            self.i += 1;
            return unsafe { Some((&(*entry.ptr).key, &(*entry.ptr).value)) };
        }
    }
}

impl<'root, K, V, S> HashMapRef<'root, K, V, S> {
    // Returns a reference to the next table if it has already been created.
    fn next_table_ref(&self) -> Option<HashMapRef<'root, K, V, S>> {
        let state = self.table.state();
        let next = state.next.load(Ordering::Acquire);

        if !next.is_null() {
            return unsafe { Some(self.as_ref(Table::from_raw(next))) };
        }

        None
    }

    // Returns the next table, allocating it has not already been created.
    fn get_or_alloc_next(&self, capacity: Option<usize>) -> Table<K, V> {
        const SPIN_ALLOC: usize = 7;

        let state = self.table.state();
        let next = state.next.load(Ordering::Acquire);

        // the next table is already allocated
        if !next.is_null() {
            return unsafe { Table::from_raw(next) };
        }

        // otherwise try to acquire the lock
        let _allocating = match state.allocating.try_lock() {
            Ok(lock) => lock,
            // someone else is allocating
            Err(_) => {
                let mut spun = 0;

                // spin for a bit, waiting for the table
                while spun <= SPIN_ALLOC {
                    for _ in 0..(spun * spun) {
                        hint::spin_loop();
                    }

                    let next = state.next.load(Ordering::Acquire);
                    if !next.is_null() {
                        return unsafe { Table::from_raw(next) };
                    }

                    spun += 1;
                }

                // otherwise we wait
                state.allocating.lock().unwrap()
            }
        };

        // was the table allocated while we were acquiring the lock?
        let next = state.next.load(Ordering::Acquire);
        if !next.is_null() {
            return unsafe { Table::from_raw(next) };
        }

        // double the table's capacity
        let next_capacity = capacity.unwrap_or(self.table.len << 1);

        if next_capacity > isize::MAX as usize {
            panic!("Hash table exceeded maximum capacity");
        }

        // allocate the new table
        let next = Table::new(next_capacity, self.root.collector.link());
        state.next.store(next.raw, Ordering::Release);
        drop(_allocating);

        next
    }

    // Update the copy state and attempt to promote a copy to the root table.
    //
    // Returns true if the table was promoted.
    fn try_promote(&self, next: Table<K, V>, copied: usize, guard: &impl Guard) -> bool {
        // update the count
        let copied = if copied > 0 {
            next.state().copied.fetch_add(copied, Ordering::AcqRel) + copied
        } else {
            next.state().copied.load(Ordering::Acquire)
        };

        if copied == self.table.len {
            // mark this table as the root
            next.state()
                .status
                .store(State::PROMOTED, Ordering::Release);

            let root = guard.protect(&self.root.table, Ordering::Acquire);
            if self.table.raw == root {
                if self
                    .root
                    .table
                    .compare_exchange(
                        self.table.raw,
                        next.raw,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    unsafe {
                        // retire the old table. not we don't drop any entries because everything was copied
                        guard.defer_retire(self.table.raw, |link| {
                            let raw: *mut RawTable = link.cast();
                            let table = Table::<K, V>::from_raw(raw);
                            drop_table::<K, V>(table);
                        })
                    }
                }

                // wake up anyone waiting for the promotion
                atomic_wait::wake_all(&next.state().status);
                return true;
            }
        }

        false
    }

    // Creates a reference to the given table while maintaining the root table pointer.
    fn as_ref(&self, table: Table<K, V>) -> HashMapRef<'root, K, V, S> {
        HashMapRef {
            table,
            root: self.root,
        }
    }
}

impl<K, V, S> Clone for HashMapRef<'_, K, V, S> {
    fn clone(&self) -> Self {
        HashMapRef {
            table: self.table,
            root: self.root,
        }
    }
}

impl<K, V, S> Drop for HashMap<K, V, S> {
    fn drop(&mut self) {
        let mut raw = *self.table.get_mut();

        while !raw.is_null() {
            let mut table = unsafe { Table::<K, V>::from_raw(raw) };
            let next = *table.state_mut().next.get_mut();
            unsafe { drop_entries::<K, V>(table) };
            unsafe { drop_table::<K, V>(table) };
            raw = next;
        }
    }
}

unsafe fn drop_entries<K, V>(table: Table<K, V>) {
    // drop all the entries
    for i in 0..table.len {
        let entry = unsafe { (*table.entry(i).as_ptr()).unpack(Entry::MASK) };

        // the entry was copied, or there is nothing to deallocate
        if entry.ptr.is_null()
            || entry.addr & Entry::TOMBSTONE != 0
            || entry.addr & Entry::COPYING != 0
        {
            continue;
        }

        unsafe { Entry::retire::<K, V>(entry.ptr.cast()) }
    }
}

unsafe fn drop_table<K, V>(mut table: Table<K, V>) {
    // drop any entries deferred during an incremental resize
    let deferred = table.state_mut().deferred.get_mut().unwrap();
    for entry in deferred {
        // safety: a deferred entry was retired after it was made unreachable
        // from the next table during a resize. because our table was still accessible
        // for this entry to be deferred, our table must have been retired *after* the
        // entry was made accessible in the next table. additionally, any previous tables
        // must have been retired *before* our resize was promoted and we were retired.
        // thus that no threads that were active to access this entry in the next table,
        // nor previous tables, can be active now
        unsafe { Entry::retire::<K, V>((*entry).cast()) }
    }

    // deallocate the table
    unsafe { Table::dealloc(table) };
}

// The maximum probe length for table operations.
macro_rules! probe_limit {
    ($capacity:expr) => {
        // 5 * log2(capacity): testing shows this gives us a ~80% load factor
        5 * ((usize::BITS as usize) - ($capacity.leading_zeros() as usize) - 1)
    };
}

use probe_limit;

// Returns an esitmate of he number of entries needed to hold `capacity` elements.
fn entries_for(capacity: usize) -> usize {
    // we should rarely resize before 75%
    let capacity = capacity.checked_mul(8).expect("capacity overflow") / 6;
    capacity.next_power_of_two()
}

// Number of linear probes per triangular.
const GROUP: usize = 8;

// Triangular probe sequence.
//
// See https://fgiesen.wordpress.com/2015/02/22/triangular-numbers-mod-2n for details.
#[derive(Default)]
struct Probe {
    i: usize,
    length: usize,
    stride: usize,
}

impl Probe {
    fn start() -> Probe {
        Probe::default()
    }

    #[inline]
    fn next(&mut self) {
        self.length += 1;

        if self.length & (GROUP - 1) == 0 {
            self.stride += GROUP;
            self.i += self.stride;
            return;
        }

        self.i += 1;
    }
}

#[inline(always)]
fn h1(hash: u64) -> usize {
    hash as usize
}

// Entry metadata.
mod meta {
    use std::mem;

    // Marks an empty entry.
    pub const EMPTY: u8 = 0x80;

    // Marks an entry that has been deleted.
    pub const TOMBSTONE: u8 = u8::MAX;

    /// Returns the top bits of the hash, used as metadata.
    #[inline(always)]
    pub fn h2(hash: u64) -> u8 {
        const MIN_HASH_LEN: usize = if mem::size_of::<usize>() < mem::size_of::<u64>() {
            mem::size_of::<usize>()
        } else {
            mem::size_of::<u64>()
        };

        // grab the top 7 bits of the hash. while the hash is normally a full 64-bit
        // value, some hash functions (such as fxhash) produce a usize result
        // instead, which means that the top 32 bits are 0 on 32-bit platforms.
        // so we use min_hash_len constant to handle this.
        let top7 = hash >> (MIN_HASH_LEN * 8 - 7);
        (top7 & 0x7f) as u8 // truncation
    }
}
