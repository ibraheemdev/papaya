mod alloc;
mod utils;

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{self, fence, AtomicPtr, AtomicU64, AtomicUsize, Ordering};
use std::{hint, mem, ptr};

use self::alloc::{RawTable, ResizeState};
use self::utils::{log2, AtomicPtrFetchOps, StrictProvenance};
use crate::map::ResizeBehavior;
use crate::seize::{self, reclaim, AsLink, Collector, Guard, Link, Linked};

// A lock-free hash-map.
pub struct HashMap<K, V, S> {
    collector: Collector,
    table: AtomicPtr<RawTable>,
    build_hasher: S,
    pub resize_behavior: ResizeBehavior,
    _kv: PhantomData<(K, V)>,
}

// The hash-table allocation.
type Table<K, V> = self::alloc::Table<Entry<K, V>>;

// An entry in the hash-table.
pub struct Entry<K, V> {
    pub link: Link,
    pub key: K,
    pub value: MaybeUninit<V>,
}

enum EntryStatus<V> {
    Empty,
    Tombstone,
    Value(V),
}

// Safety: seize::Link is the first field
unsafe impl<K, V> AsLink for Entry<K, V> {}

impl Entry<(), ()> {
    // The entry is being copied to the new table, no updates are allowed on the old table.
    const COPYING: usize = 0b001;

    // The entry does not contain a value, i.e. it was deleted.
    const TOMBSTONE: usize = 0b010;

    // An entry with a value that has been copied to the new table.
    const COPIED: usize = 0b100;

    // A tombstone entry that has been 'copied' to the new table.
    const TOMBCOPIED: usize = Entry::TOMBSTONE | Entry::COPYING;

    // Mask for entry pointer, ignoring tag bits.
    const POINTER: usize = !(Entry::COPIED | Entry::COPYING | Entry::TOMBSTONE);

    // Retires the entry if it's reference count is 0.
    unsafe fn try_retire_value<K, V>(entry: *mut Entry<K, V>, guard: &Guard<'_>) -> bool {
        guard.retire(entry, |link| {
            let entry_addr: *mut Entry<K, V> = link.cast();
            let entry = unsafe { Box::from_raw(entry_addr) };

            // drop the value
            let _ = unsafe { entry.value.assume_init() };
        });

        true
    }
}

impl<K, V, S> HashMap<K, V, S> {
    pub fn with_capacity_and_hasher(capacity: usize, build_hasher: S) -> HashMap<K, V, S> {
        // allocate extra buffer capacity the same length as the probe limit. this allows us
        // to avoid overflow checks
        let capacity = capacity.next_power_of_two();
        let buffer = log2!(capacity);

        let collector = Collector::new().epoch_frequency(None).batch_size(2000);
        let table = alloc::Table::<Entry<K, V>>::new(capacity, capacity + buffer, collector.link());

        HashMap {
            collector,
            build_hasher,
            resize_behavior: ResizeBehavior::Blocking,
            table: AtomicPtr::new(table.raw),
            _kv: PhantomData,
        }
    }

    pub fn guard(&self) -> Guard<'_> {
        self.collector.enter()
    }

    fn as_ref<'a>(&'a self, table: Table<K, V>) -> HashMapRef<'a, K, V, S> {
        HashMapRef {
            table,
            root: &self.table,
            collector: &self.collector,
            build_hasher: &self.build_hasher,
            resize_behavior: self.resize_behavior.clone(),
        }
    }

    pub fn with_ref<'guard, F, T>(&self, f: F, guard: &'guard Guard<'_>) -> T
    where
        F: FnOnce(HashMapRef<'_, K, V, S>) -> T,
    {
        let raw = guard.protect(&self.table, Ordering::Acquire);
        let table = unsafe { Table::<K, V>::from_raw(raw) };
        f(self.as_ref(table))
    }
}

// A reference to the root hash table, or an arbitrarily nested table migration.
pub struct HashMapRef<'a, K, V, S> {
    table: Table<K, V>,
    root: &'a AtomicPtr<RawTable>,
    resize_behavior: ResizeBehavior,
    collector: &'a Collector,
    build_hasher: &'a S,
}

impl<K, V, S> HashMapRef<'_, K, V, S>
where
    K: Clone + Sync + Send + Hash + Eq,
    V: Sync + Send,
    S: BuildHasher,
{
    // Returns a reference to the value corresponding to the key.
    pub fn get<'guard, Q>(&self, key: &Q, guard: &'guard Guard<'_>) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let hash = self.hash(key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let probe_limit = i + log2!(self.table.len);

        while i <= probe_limit {
            let mut meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            // found an empty entry, the entry is not the table
            if meta == meta::EMPTY {
                return None;
            }

            // found an empty entry, the entry is not the table
            if meta == meta::TOMBSTONE {
                continue;
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                if entry.addr() == Entry::TOMBSTONE {
                    continue;
                }

                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // check for a full match
                if unsafe { (*entry_ptr).key.borrow() } == key {
                    // we don't care if this entry is currently being copied because until the copy is
                    // completed, this is the latest value
                    unsafe { return Some((*entry_ptr).value.assume_init_ref()) }
                }
            }

            i += 1;
        }

        None
    }

    // Inserts a key-value pair into the map.
    pub fn insert<'guard>(&self, key: K, value: V, guard: &'guard Guard<'_>) -> Option<&'guard V> {
        let entry = Box::into_raw(Box::new(Entry {
            key,
            value: MaybeUninit::new(value),
            link: self.collector.link(),
        }));

        match self.insert_entry(entry, guard) {
            EntryStatus::Value(value) => Some(value),
            _ => None,
        }
    }

    // Inserts an entry into the map.
    fn insert_entry<'guard>(
        &self,
        new_entry: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> EntryStatus<&'guard V> {
        let key = unsafe { &(*new_entry.map_addr(|addr| addr & Entry::POINTER)).key };

        let hash = self.hash(key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let probe_limit = i + log2!(self.table.len);

        while i <= probe_limit {
            let mut meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            if meta == meta::TOMBSTONE {
                continue;
            }

            if meta == meta::EMPTY {
                match unsafe { self.table.entry(i) }.compare_exchange(
                    ptr::null_mut(),
                    new_entry,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    // successfully claimed this entry
                    Ok(_) => {
                        // update the meta byte
                        unsafe { self.table.meta(i).store(h2, Ordering::Release) };
                        return EntryStatus::Empty;
                    }
                    Err(found) => {
                        if found.addr() == Entry::TOMBSTONE {
                            continue;
                        }

                        // found a phantom entry, switch to the new table.
                        if found.addr() == Entry::COPYING {
                            break;
                        }

                        fence(Ordering::Acquire);
                        let found_ptr = found.map_addr(|addr| addr & Entry::POINTER);

                        // the key matches, we might be able to perform an update
                        if unsafe { (*found_ptr).key == *key } {
                            match self.replace_entry(i, found, new_entry, guard) {
                                Ok(status) => return status,
                                Err(_) => break,
                            }
                        }

                        continue;
                    }
                }
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                if entry.addr() == Entry::TOMBSTONE {
                    continue;
                }

                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key == *key } {
                    match self.replace_entry(i, entry, new_entry, guard) {
                        Ok(status) => return status,
                        Err(_) => break,
                    }
                }
            }

            i += 1;
        }

        // went over the max probe count: trigger a resize.
        let next_table = self.get_or_alloc_next();

        // help along the highest priority (top-level) copy
        let root = guard.protect(&self.root, Ordering::Acquire);
        let root = unsafe { Table::<K, V>::from_raw(root) };
        self.as_ref(root).help_copy(guard);

        // insert into the next table
        self.as_ref(next_table).insert_entry(new_entry, guard)
    }

    // Replaces the value of an existing entry, returning the previous value if successful.
    //
    // Inserts into the new table if the entry is being copied.
    fn replace_entry<'guard>(
        &self,
        i: usize,
        mut entry: *mut Entry<K, V>,
        new_entry: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> Result<EntryStatus<&'guard V>, ()> {
        loop {
            // the entry is being copied to a new table, we have to finish the resize
            // before we insert
            if entry.addr() & Entry::COPYING != 0 {
                return Err(());
            }

            match unsafe { self.table.entry(i) }.compare_exchange_weak(
                entry,
                new_entry,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                // succesful update
                Ok(_) => unsafe {
                    let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                    Entry::try_retire_value(entry_ptr, guard);
                    return Ok(EntryStatus::Value((*entry_ptr).value.assume_init_ref()));
                },

                // lost to a concurrent update or delete, retry
                Err(found) if found.addr() == Entry::TOMBSTONE => {
                    return Ok(EntryStatus::Tombstone);
                }

                // lost to a concurrent update or delete, retry
                Err(found) => {
                    entry = found;
                }
            }
        }
    }

    // Update an entry with a remapping function.
    pub fn update<'guard, F>(&self, mut key: K, f: F, guard: &'guard Guard<'_>) -> Option<&'guard V>
    where
        F: Fn(&V) -> V,
    {
        let mut update: *mut Entry<K, V> = Box::into_raw(Box::new(Entry {
            key,
            value: MaybeUninit::uninit(),
            link: self.collector.link(),
        }));

        self.update_with(update, f, guard)
    }

    // Update an entry with a remapping function.
    pub fn update_with<'guard, F>(
        &self,
        mut update: *mut Entry<K, V>,
        f: F,
        guard: &'guard Guard<'_>,
    ) -> Option<&'guard V>
    where
        F: Fn(&V) -> V,
    {
        let hash = unsafe { self.hash(&(*update).key) };
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let probe_limit = i + log2!(self.table.len);

        while i <= probe_limit {
            let mut meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            // found an empty entry, the key is not in this table
            if meta == meta::EMPTY {
                return None;
            }

            if meta == meta::TOMBSTONE {
                continue;
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                if entry.addr() == Entry::TOMBSTONE {
                    continue;
                }

                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key == (*update).key } {
                    let entry = loop {
                        // the entry is being copied to a new table, we have to copy it before we can update it
                        if entry.addr() & Entry::COPYING != 0 {
                            break entry;
                        }

                        // perform the update
                        unsafe {
                            let value = f((*entry_ptr).value.assume_init_ref());
                            (*update).value = MaybeUninit::new(value);
                        }

                        match unsafe { self.table.entry(i) }.compare_exchange_weak(
                            entry,
                            update,
                            Ordering::Release,
                            Ordering::Relaxed,
                        ) {
                            // succesful update
                            Ok(_) => unsafe {
                                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                                // retire the only entry
                                Entry::try_retire_value(entry_ptr, guard);
                                return Some((*entry_ptr).value.assume_init_ref());
                            },

                            // the entry got deleted
                            Err(found) if found.addr() == Entry::TOMBSTONE => {
                                return None;
                            }

                            // lost to a concurrent update or delete, retry
                            Err(found) => {
                                unsafe {
                                    // drop the old value
                                    (*update).value.assume_init_drop()
                                }

                                entry = found;
                            }
                        }
                    };

                    let new_table = self.next_table_ref().unwrap();

                    // the entry has not been copied yet, we need to copy it
                    if entry.addr() & Entry::COPIED == 0 {
                        self.copy_entry(i, entry, new_table.table, guard);
                    }

                    // update the entry in the new table
                    return new_table.update_with(update, f, guard);
                }
            }

            i += 1;
        }

        None
    }

    // Removes a key from the map, returning the value at the key if the key was previously in the map.
    pub fn remove<'guard, Q: ?Sized>(&self, key: &Q, guard: &'guard Guard<'_>) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let hash = self.hash(key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let probe_limit = i + log2!(self.table.len);

        while i <= probe_limit {
            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // encountered an empty entry, the key is not in this table
            if meta == meta::EMPTY {
                return None;
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };
                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                if entry.addr() == Entry::TOMBSTONE {
                    continue;
                }

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key.borrow() == key } {
                    let owned_key = unsafe { (*entry_ptr).key.clone() };

                    // the entry is being copied to a new table, we have to go there and delete it
                    if entry.addr() & Entry::COPYING != 0 {
                        let next_table = self.next_table_ref().unwrap();
                        // TODO: can't do this, have to finish copy before deleting
                        self.copy_entry(i, found, next_table.table, guard);
                        return next_table.remove(key, guard);
                    }

                    loop {
                        // perform the deletion
                        match unsafe { self.table.entry(i) }.compare_exchange_weak(
                            entry,
                            Entry::TOMBSTONE as _,
                            Ordering::Release,
                            Ordering::Relaxed,
                        ) {
                            // succesfully deleted
                            Ok(_) => unsafe {
                                let entry = entry.map_addr(|addr| addr & Entry::POINTER);
                                Entry::try_retire_value(entry, guard);

                                return Some((*entry).value.assume_init_ref());
                            },

                            // the entry is being copied to the new table, retry there. note this might
                            // also be a tombcopied entry, but we still then have to ensure it's not in
                            // the new table
                            Err(found) if found.addr() & Entry::COPYING != 0 => {
                                break;
                            }

                            // the entry was deleted in this table and is not marked as copied,
                            // so it can't have been updated in the new table
                            Err(found) if found.addr() & Entry::TOMBSTONE != 0 => {
                                return None;
                            }

                            // lost to a concurrent update, retry
                            Err(found) => {
                                entry = found;
                            }
                        }
                    }
                }
            }

            i += 1;
        }

        // went over the max probe count: the key is not in this table, but it
        // might be in the new table
        if let Some(next_table) = self.next_table_ref() {
            next_table.help_copy(guard);
            return next_table.remove(key, guard);
        }

        return None;
    }

    // Help along the resize operation until the old table is fully copied from.
    //
    // Note this should only be called on the root table.
    fn help_copy(&self, guard: &Guard<'_>) {
        let state = self.table.resize_state();

        let next_table = state.next.load(Ordering::Acquire);

        // no copy in progress
        if next_table.is_null() {
            return;
        }

        let next_table = unsafe { Table::<K, V>::from_raw(next_table) };

        // is the copy already complete?
        if self.try_promote(next_table, 0, guard) {
            return;
        }

        // the true table capacity, we have to copy every entry including from the buffer
        let capacity = self.table.len + log2!(self.table.len);
        let copy_chunk = capacity.min(1024);

        'claim: loop {
            if state.claim.load(Ordering::Acquire) >= capacity {
                break;
            }

            // claim a range to copy
            let copy_start = state.claim.fetch_add(copy_chunk, Ordering::Relaxed);

            let mut copied = 0;
            for i in 0..copy_chunk {
                // note the capacity including the buffer is not a power of two
                let i = copy_start + i;

                if i >= capacity {
                    break 'claim;
                }

                // keep track of the entries we actually copy
                if self.copy_index(i, next_table, guard) {
                    copied += 1;
                }
            }

            // are we done?
            if self.try_promote(next_table, copied, guard) {
                return;
            }
        }

        while state.futex.load(Ordering::Acquire) == ResizeState::PENDING {
            atomic_wait::wait(&state.futex, ResizeState::PENDING);
        }
    }

    // Copy the entry at the given index to the new table.
    //
    // Returns true if this thread ended up doing the copy.
    fn copy_index(&self, i: usize, new_table: Table<K, V>, guard: &Guard<'_>) -> bool {
        let entry = unsafe { self.table.entry(i).load(Ordering::Acquire) };
        self.copy_entry(i, entry, new_table, guard)
    }

    // Copy the given entry to the new table.
    //
    // Returns true if this thread ended up doing the copy.
    fn copy_entry(
        &self,
        i: usize,
        mut entry: *mut Entry<K, V>,
        new_table: Table<K, V>,
        guard: &Guard<'_>,
    ) -> bool {
        loop {
            // the entry has already been copied
            if entry.addr() & Entry::COPIED != 0
                // the entry was empty
                || entry.addr() == Entry::COPYING
                // the entry was a tombstone
                || entry.addr() & Entry::TOMBCOPIED == Entry::TOMBCOPIED
            {
                return false;
            }

            // the entry is already marked as copying
            if entry.addr() & Entry::COPYING != 0 {
                break;
            }

            // TODO: fetch_or
            match unsafe {
                self.table.entry(i).compare_exchange_weak(
                    entry,
                    entry.map_addr(|addr| addr | Entry::COPYING),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
            } {
                // we created a phantom entry, there is nothing to copy
                Ok(_) if entry.is_null() => {
                    unsafe { self.table.meta(i).store(meta::TOMBSTONE, Ordering::Release) };
                    return true;
                }

                // the entry was a tombstone, there is nothing to copy
                Ok(_) if entry.is_null() || entry.addr() & Entry::TOMBSTONE != 0 => {
                    return true;
                }

                // otherwise we have to copy the value
                Ok(_) => break,

                // something changed, retry
                Err(found) => entry = found,
            }
        }

        entry = entry.map_addr(|addr| addr & Entry::POINTER);

        let copied = self.as_ref(new_table).insert_copy(entry, guard);

        // mark the entry as copied
        if copied {
            unsafe {
                self.table
                    .entry(i)
                    .fetch_or(Entry::COPIED, Ordering::Release);
            }
        } else {
            // otherwise we have to decrement the reference count because either an update won
            // the race and this entry was never in the new table, or the thread that did the copy
            // already incremented the count. in the unlikely case that this entry was already
            // deleted from the new table, we also have to retire.
            unsafe { Entry::try_retire_value(entry, guard) };
        }

        return copied;
    }

    // Copy an entry into the table.
    //
    // Returns whether or not the entry was inserted directly. Any matching key found in the table is
    // considered to overwrite the copy.
    fn insert_copy<'guard>(&self, new_entry: *mut Entry<K, V>, guard: &'guard Guard<'_>) -> bool {
        let key = unsafe { &(*new_entry.map_addr(|addr| addr & Entry::POINTER)).key };

        let hash = self.hash(key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let probe_limit = i + log2!(self.table.len);

        while i <= probe_limit {
            let mut meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            // found a phantom entry, switch to the nested resize
            if meta == meta::PHANTOM {
                break;
            }

            if meta == meta::EMPTY {
                match unsafe { self.table.entry(i) }.compare_exchange(
                    ptr::null_mut(),
                    new_entry,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    // successfully claimed this entry
                    Ok(_) => {
                        // update the meta byte
                        unsafe { self.table.meta(i).store(h2, Ordering::Release) };
                        return true;
                    }
                    Err(found) => {
                        // found a phantom entry, mark it for readers and switch to the nested resize
                        if found.addr() == Entry::COPYING {
                            unsafe { self.table.meta(i).store(meta::PHANTOM, Ordering::Release) };
                            break;
                        }

                        fence(Ordering::Acquire);
                        let found_ptr = found.map_addr(|addr| addr & Entry::POINTER);

                        // someone else copied the key or overwrote the old value, we're done
                        if unsafe { (*found_ptr).key == *key } {
                            return false;
                        }

                        continue;
                    }
                }
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };
                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // someone else copied the key or overwrote the old value, we're done
                if unsafe { (*entry_ptr).key == *key } {
                    return false;
                }
            }

            i += 1;
        }

        // went over the max probe count: trigger a nested resize
        let next_table = self.get_or_alloc_next();
        self.as_ref(next_table).insert_copy(new_entry, guard)
    }

    // Returns the hash of a key.
    #[inline]
    fn hash<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        let mut h = self.build_hasher.build_hasher();
        key.hash(&mut h);
        h.finish()
    }
}

impl<'root, K, V, S> HashMapRef<'root, K, V, S> {
    // Returns a reference to the next table if it has already been created.
    fn next_table_ref(&self) -> Option<HashMapRef<'root, K, V, S>> {
        let state = self.table.resize_state();
        let next = state.next.load(Ordering::Acquire);

        if !next.is_null() {
            return unsafe { Some(self.as_ref(Table::from_raw(next))) };
        }

        None
    }

    // Returns the next table, allocating it has not already been created.
    fn get_or_alloc_next(&self) -> Table<K, V> {
        const SPIN_ALLOC: usize = 7;

        let state = self.table.resize_state();
        let next = state.next.load(Ordering::Acquire);

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

        let next_capacity = self.table.len << 1;
        let buffer = log2!(next_capacity);

        if next_capacity > isize::MAX as usize {
            panic!("Hash table exceeded maximum capacity");
        }

        // allocate the new table
        let link = self.collector.link();
        let next = Table::new(next_capacity, next_capacity + buffer, link);

        // store it, and release the lock
        state.next.store(next.raw, Ordering::Release);
        drop(_allocating);

        next
    }

    // Update the copy state, and attempt to promote a copy to the root table.
    //
    // Returns true if the copy is complete, but not necessarily promoted.
    fn try_promote(&self, mut next: Table<K, V>, copied: usize, guard: &Guard<'_>) -> bool {
        let state = self.table.resize_state();

        // update the count
        let copied = if copied > 0 {
            state.copied.fetch_add(copied, Ordering::Relaxed) + copied
        } else {
            state.copied.load(Ordering::Relaxed)
        };

        if copied == self.table.len + log2!(self.table.len) {
            let root = self.root.load(Ordering::Acquire);
            state.futex.store(ResizeState::COMPLETE, Ordering::Release);

            // we only promote the top-level copy
            if root == self.table.raw {
                self.root
                    .compare_exchange(root, next.raw, Ordering::Release, Ordering::Relaxed)
                    .unwrap();

                atomic_wait::wake_all(&state.futex);

                // retire the old table
                unsafe {
                    guard.retire(root, |link| {
                        let raw: *mut RawTable = link.cast();
                        let mut table: Table<K, V> = Table::from_raw(raw);
                        drop_entries(&mut table);
                        Table::dealloc(table);
                    })
                };
            }

            return true;
        }

        false
    }

    // Creates a reference to the given table while maintaining the root table pointer.
    fn as_ref(&self, table: Table<K, V>) -> HashMapRef<'root, K, V, S> {
        HashMapRef {
            table,
            root: self.root,
            collector: self.collector,
            build_hasher: self.build_hasher,
            resize_behavior: self.resize_behavior.clone(),
        }
    }
}

fn drop_entries<K, V>(table: &mut Table<K, V>) {
    for i in 0..(table.len + log2!(table.len)) {
        let entry = unsafe { *(table.entry(i) as *const AtomicPtr<_> as *const *mut Entry<K, V>) };
        let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

        if entry_ptr.is_null() {
            continue;
        }

        // tombstone entries are allocated but never copied to another table
        if entry.addr() & Entry::TOMBSTONE != 0 {
            continue;
        }

        // any other non-null entry should be retired through reference counting
        // to ensure we don't double-free copied entries in nested tables
        unsafe {
            Entry::try_retire_value(entry_ptr, &seize::Guard::unprotected());
        }
    }
}

impl<K, V, S> Drop for HashMap<K, V, S> {
    fn drop(&mut self) {
        let table = unsafe { Table::<K, V>::from_raw(*self.table.get_mut()) };
        let mut next_map = Some(self.as_ref(table));

        while let Some(mut map) = next_map {
            drop_entries(&mut map.table);
            next_map = map.next_table_ref();
            unsafe { Table::dealloc(map.table) };
        }
    }
}

fn h1(hash: u64) -> usize {
    hash as usize
}

// Entry metadata.
mod meta {
    use std::mem;

    // Marks an empty entry.
    pub const EMPTY: u8 = 0x80;

    pub const TOMBSTONE: u8 = u8::MAX;

    /// Returns the top bits of the hash, used as metadata.
    #[inline]
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
