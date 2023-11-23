mod alloc;
mod utils;
// mod bit_lock;

// TODO: check for copied everywhere

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{self, AtomicPtr, Ordering};
use std::{hint, mem, ptr};

use self::alloc::{RawTable, ResizeState};
use self::utils::StrictProvenance;
use crate::seize::{reclaim, AsLink, Collector, Guard, Link, Linked};
use utils::log2;

// A lock-free hashmap.
pub struct HashMap<K, V, S> {
    collector: Collector,
    table: AtomicPtr<RawTable>,
    build_hasher: S,
    _kv: PhantomData<(K, V)>,
}

type Table<K, V> = self::alloc::Table<Entry<K, V>>;

// An entry in the hash table.
pub struct Entry<K, V> {
    pub link: Link,
    pub key: K,
    pub value: MaybeUninit<V>,
}

// Safety: seize::Link is the first field
unsafe impl<K, V> AsLink for Entry<K, V> {}

impl Entry<(), ()> {
    // the entry is being copied to a new table
    const COPYING: usize = 0b01;

    // the entry has been deleted from the table
    const TOMBSTONE: usize = 0b10;

    // the entry has been successfully copied to the new table
    const COPIED: usize = Entry::COPYING | Entry::TOMBSTONE;

    // mask for entry pointer, ignoring tag bits
    const POINTER: usize = !(Entry::COPYING | Entry::TOMBSTONE);
}

impl<K, V, S> HashMap<K, V, S> {
    pub fn with_capacity_and_hasher(capacity: usize, build_hasher: S) -> HashMap<K, V, S> {
        // allocate extra buffer capacity the same length as the probe limit.
        // this allows us to avoid overflow checks
        let max_probes = log2!(capacity);
        let capacity = capacity.next_power_of_two() + max_probes;

        let collector = Collector::new().epoch_frequency(None);
        let table = alloc::Table::<Entry<K, V>>::new(capacity, collector.link());

        HashMap {
            collector,
            build_hasher,
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
    collector: &'a Collector,
    build_hasher: &'a S,
}

impl<K, V, S> HashMapRef<'_, K, V, S>
where
    K: Clone + Hash + Eq + Sync + Send,
    V: Sync + Send,
    S: BuildHasher,
{
    pub fn get<'guard, Q>(&self, key: &Q, guard: &'guard Guard<'_>) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        let max_probes = log2!(self.table.capacity);
        let capacity = self.table.capacity - max_probes;

        let hash = self.hash(&key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (capacity - 1);
        let limit = i + max_probes;

        while i <= limit {
            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // encountered an empty meta in the probe sequence, the key cannot exist in the table
            if meta == meta::EMPTY {
                return None;
            }

            // metadata match
            if meta & meta::H2 == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry has been, or is being copied to the new table.
                // even if the copy is not complete yet, it is possible that the
                // latest value already exists in the new table courtesy of a concurrent
                // insert that may have happened-before us, so we cannot simply return
                // this value.
                if entry.addr() & Entry::COPYING != 0 {
                    return self.next_table_ref().unwrap().get(key, guard);
                }

                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // this is a true match
                if unsafe { (*entry_ptr).key.borrow() == key } {
                    // the entry existed in the table but was deleted
                    if entry.addr() & Entry::TOMBSTONE != 0 {
                        return None;
                    }

                    unsafe { return Some((*entry_ptr).value.assume_init_ref()) }
                }
            }

            // the slot contained a different key, keep searching
            i += 1;
        }

        // went over the max probe count: the key is not in this table.
        // any inserts would have triggered a resize at this point.
        // check if there is an active migration
        if let Some(next_table) = self.next_table_ref() {
            return next_table.get(key, guard);
        }

        // the key is not in the table and there is no active migration
        None
    }

    pub fn insert<'guard>(&self, key: K, value: V, guard: &'guard Guard<'_>) -> Option<&'guard V> {
        let entry = Box::into_raw(Box::new(Entry {
            key,
            value: MaybeUninit::new(value),
            link: self.collector.link(),
        }));

        self.insert_entry(entry, guard)
    }

    pub fn insert_entry<'guard>(
        &self,
        new_entry: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> Option<&'guard V> {
        let max_probes = log2!(self.table.capacity);
        let capacity = self.table.capacity - max_probes;

        let key = unsafe { &(*new_entry).key };

        let hash = self.hash(&key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (capacity - 1);
        let limit = i + max_probes;

        'probe: while i <= limit {
            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // possible empty entry
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
                        return None;
                    }

                    // lost the entry to someone else, keep searching, unless
                    // it was a concurrent insert of the same key
                    Err(found) => {
                        let found_ptr = found.map_addr(|addr| addr & Entry::POINTER);

                        // the key matches, we might be able to perform an update
                        if unsafe { (*found_ptr).key == *key } {
                            // the entry is being copied to a new table, we have to go there
                            // and join the race for the insertion
                            if found.addr() & Entry::COPYING != 0 {
                                return self
                                    .next_table_ref()
                                    .unwrap()
                                    .insert_entry(new_entry, guard);
                            }

                            // perform the update
                            match self.replace_entry(i, found, new_entry, guard) {
                                Ok(value) => return Some(value),
                                // lost to a concurrent copier, retry in the new table
                                Err(_) => {
                                    return self
                                        .next_table_ref()
                                        .unwrap()
                                        .insert_entry(new_entry, guard)
                                }
                            }
                        }

                        continue 'probe;
                    }
                }
            }

            if meta & meta::H2 == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };
                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key == *key } {
                    // the entry is being copied to a new table, we have to go there
                    // and join the race for the insertion
                    if entry.addr() & Entry::COPYING != 0 {
                        return self
                            .next_table_ref()
                            .unwrap()
                            .insert_entry(new_entry, guard);
                    }

                    // perform the update
                    match self.replace_entry(i, entry, new_entry, guard) {
                        Ok(value) => return Some(value),
                        // lost to a concurrent copier, retry in the new table
                        Err(_) => {
                            return self
                                .next_table_ref()
                                .unwrap()
                                .insert_entry(new_entry, guard)
                        }
                    }
                }
            }

            // this entry either contains another key, keep searching
            i += 1;
        }

        // went over the max probe count: trigger a resize.
        // the entry can be safely inserted directly into the new table
        // as any other concurrent insertions will also see the full
        // probe sequence.
        let next_table = self.next_table();

        // help along the highest priority (top-level) copy
        let root = guard.protect(&self.root, Ordering::Acquire);
        let root = unsafe { Table::<K, V>::from_raw(root) };
        self.as_ref(root).help_copy(guard);

        // insert into the next table
        self.as_ref(next_table).insert_entry(new_entry, guard)
    }

    // Performs `CAS(entry, new_entry)` on `entries[i]`, returning the previous value if successful.
    // Returns an error if the entry is copied.
    fn replace_entry<'guard>(
        &self,
        i: usize,
        mut entry: *mut Entry<K, V>,
        new_entry: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> Result<&'guard V, ()> {
        loop {
            match unsafe { self.table.entry(i) }.compare_exchange_weak(
                entry,
                new_entry,
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                // succesful update
                Ok(_) => unsafe {
                    let entry = entry.map_addr(|addr| addr & Entry::POINTER);

                    // retire the old entry
                    guard.retire(entry, reclaim::boxed::<Entry<K, V>>);

                    // return the previous value
                    return Ok((*entry).value.assume_init_ref());
                },

                // the entry is being copied to the new table, tell the caller to retry there
                Err(found) if found.addr() & Entry::COPIED != 0 => {
                    return Err(());
                }

                // lost to a concurrent update or delete, retry.
                Err(found) => {
                    entry = found;
                    continue;
                }
            }
        }
    }

    pub fn remove<'guard, Q: ?Sized>(&self, key: &Q, guard: &'guard Guard<'_>) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.remove_with(key, ptr::null_mut(), guard)
    }

    pub fn remove_with<'guard, Q: ?Sized>(
        &self,
        key: &Q,
        mut deletion: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let max_probes = log2!(self.table.capacity);
        let capacity = self.table.capacity - max_probes;

        let hash = self.hash(&key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (capacity - 1);
        let limit = i + max_probes;

        while i <= limit {
            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // encountered an empty meta in the probe sequence, the key cannot exist in the table
            if meta == meta::EMPTY {
                return None;
            }

            // possible match
            if meta & meta::H2 == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };
                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key.borrow() == key } {
                    let owned_key = unsafe { (*entry_ptr).key.clone() };

                    // the entry is being copied to a new table, we have to go there and
                    // delete it, or put down our deletion first
                    if entry.addr() & Entry::COPYING != 0 {
                        return self
                            .next_table_ref()
                            .unwrap()
                            .remove_with(key, deletion, guard);
                    }

                    // the entry is already deleted
                    if entry.addr() & Entry::TOMBSTONE != 0 {
                        return None;
                    }

                    if deletion.is_null() {
                        deletion = Box::into_raw(Box::new(Entry {
                            key: owned_key,
                            value: MaybeUninit::uninit(),
                            link: self.collector.link(),
                        }));
                    }

                    loop {
                        // perform the deletion
                        match unsafe { self.table.entry(i) }.compare_exchange_weak(
                            entry,
                            deletion,
                            Ordering::Release,
                            Ordering::Relaxed,
                        ) {
                            // succesfully deleted
                            Ok(_) => unsafe {
                                let entry = entry.map_addr(|addr| addr & Entry::POINTER);

                                // retire the old entry
                                guard.retire(entry, reclaim::boxed::<Entry<K, V>>);

                                // return the previous value
                                return Some((*entry).value.assume_init_ref());
                            },

                            // the entry is being copied to the new table, retry there
                            Err(found) if found.addr() & Entry::COPIED != 0 => {
                                return self
                                    .next_table_ref()
                                    .unwrap()
                                    .remove_with(key, deletion, guard);
                            }

                            // someone else deleted first
                            Err(found) if found.addr() & Entry::TOMBSTONE != 0 => {
                                return None;
                            }

                            // lost to a concurrent update, retry.
                            Err(found) => {
                                entry = found;
                                continue;
                            }
                        }
                    }
                }
            }

            // this entry either contains another key, keep searching
            i += 1;
        }

        // went over the max probe count: the key is not in this table.
        // any inserts would have triggered a resize at this point.
        // check if there is an active migration
        if let Some(next_table) = self.next_table_ref() {
            return next_table.remove(key, guard);
        }

        // the key is not in the table and there is no active migration
        None
    }

    // Returns a reference to the next table, if it has already been created.
    pub(crate) fn next_table_ref(&self) -> Option<HashMapRef<'_, K, V, S>> {
        let state = self.table.resize_state();
        let next = state.next.load(Ordering::Acquire);

        if !next.is_null() {
            return unsafe { Some(self.as_ref(Table::from_raw(next))) };
        }

        None
    }

    // Returns the next table, allocating it has not already been created.
    pub(crate) fn next_table(&self) -> Table<K, V> {
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

                // table still hasn't been allocated, wait for real now
                state.allocating.lock().unwrap()
            }
        };

        // was the table allocated while we were acquiring
        // the lock?
        let next = state.next.load(Ordering::Acquire);
        if !next.is_null() {
            return unsafe { Table::from_raw(next) };
        }

        // we have the lock, so we are the thread to create the table
        let cap = self.table.capacity;

        // calculate the new table size (TODO: store len?)
        let next_cap = cap << 1;

        if next_cap > isize::MAX as usize {
            panic!("Hash table exceeded maximum capacity");
        }

        // allocate the new table
        let link = self.collector.link();
        let next = Table::new(next_cap, link);

        // store it, and release the lock
        state.next.store(next.raw, Ordering::Release);
        drop(_allocating);

        next
    }

    // Help along the resize operation until the old table is fully copied from.
    fn help_copy(&self, guard: &Guard<'_>) {
        let state = self.table.resize_state();

        let next_table = state.next.load(Ordering::Acquire);

        // no-copy in progress
        assert!(!next_table.is_null());

        let next_table = unsafe { Table::<K, V>::from_raw(next_table) };

        // try to help, avoid parking
        if state.copied.load(Ordering::Relaxed) == self.table.capacity {
            self.try_promote(next_table);
            return;
        }

        let copy_chunk = self.table.capacity.min(1024);

        loop {
            let copy_start = state.claim.fetch_add(copy_chunk, Ordering::Relaxed);

            if copy_start > self.table.capacity {
                break;
            }

            let mut copy_chunk = copy_chunk;
            if copy_start + copy_chunk >= self.table.capacity {
                copy_chunk = self.table.capacity - copy_start;
            }

            for i in copy_start..(copy_start + copy_chunk) {
                self.copy_slot_to(i, next_table, guard);
            }

            let copied = state.copied.fetch_add(copy_chunk, Ordering::Relaxed);

            if copied + copy_chunk == self.table.capacity {
                state.futex.store(ResizeState::COMPLETE, Ordering::Release);
                self.try_promote(next_table);
                atomic_wait::wake_all(&state.futex);
                return;
            }
        }

        while state.futex.load(Ordering::Acquire) == ResizeState::PENDING {
            atomic_wait::wait(&state.futex, ResizeState::PENDING);
        }

        return;
    }

    fn try_promote(&self, next: Table<K, V>) {
        // only publish this table if we copied from the root table.
        // if we didn't, that means this is a nested copy operation,
        // and the previous copy has not been published yet.
        if self.root.load(Ordering::Acquire) == self.table.raw {
            let mut new_root = next;
            loop {
                let mut next = next.resize_state().next.load(Ordering::Acquire);

                if next.is_null() {
                    break;
                }

                let next = unsafe { Table::<K, V>::from_raw(next) };
                if next.resize_state().copied.load(Ordering::Acquire) != next.capacity {
                    break;
                }

                new_root = next;
            }

            self.root.compare_exchange(
                self.table.raw,
                new_root.raw,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
    }

    fn copy_slot_to(&self, i: usize, new_table: Table<K, V>, guard: &Guard<'_>) -> bool {
        // update metatable too?

        let mut old_entry = unsafe { self.table.entry(i).load(Ordering::Acquire) };
        loop {
            if old_entry.addr() & Entry::COPYING == 0 {
                return false;
            }

            // insert loop, if fail cas COPIED, rewrite to same index
        }

        let new_entry = old_entry.map_addr(|addr| addr | Entry::COPYING);
        match unsafe {
            self.table.entry(i).compare_exchange(
                old_entry,
                new_entry,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
        } {
            Ok(_) if new_entry == Entry::COPIED => return true,
            Ok(_) => {
                old_entry = new_entry;
            }
            Err(found) => {
                old_entry = found;
            }
        }

        let untagged = old_entry.map_addr(|addr| addr & !Entry::COPYING);
        self.as_ref(new_table).insert_copy(untagged, guard);
        // TODO: new_table.insert_if_overwrite_null

        while old_entry != Entry::COPIED {
            // make sure no one copies it now
            match unsafe {
                self.table.entry(i).compare_exchange(
                    old_entry,
                    Entry::COPIED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
            } {
                Ok(_) => {
                    return true;
                }
                Err(found) => {
                    old_entry = found;
                    continue;
                }
            }
        }

        return false;
    }

    fn as_ref<'a>(&'a self, table: Table<K, V>) -> HashMapRef<'a, K, V, S> {
        HashMapRef {
            table,
            root: self.root,
            collector: self.collector,
            build_hasher: self.build_hasher,
        }
    }

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

#[inline]
fn h1(hash: u64) -> usize {
    hash as usize
}

mod meta {
    use std::mem;

    // an empty slot
    pub const EMPTY: u8 = 0x80;

    // h2 mask, ignoring the first bit which is only
    // set for TOMBSTONE and EMPTY
    pub const H2: u8 = !EMPTY;

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
