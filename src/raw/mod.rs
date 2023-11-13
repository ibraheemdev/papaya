mod alloc;
mod utils;
// mod bit_lock;

// TODO: check for copied everywhere

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
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
    pub value: V,
}

// Safety: seize::Link is the first field
unsafe impl<K, V> AsLink for Entry<K, V> {}

impl<K, V> Entry<K, V> {
    pub const EMPTY: *mut Self = ptr::null_mut();
    pub const TOMBSTONE: *mut Self = !1 as _;
    pub const COPIED: *mut Self = !0 as _;
}

impl Entry<(), ()> {
    // tag bit reserved for copy marker
    const COPYING: usize = 1;
}

impl<K, V, S> HashMap<K, V, S> {
    pub fn with_capacity_and_hasher(capacity: usize, build_hasher: S) -> HashMap<K, V, S> {
        // add extra buffer capacity the same length as the probe limit.
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
    K: Hash + Eq + Sync + Send,
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

        'probe: loop {
            // went over the max probe count
            if i > limit {
                break;
            }

            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // encountered an empty meta in the probe sequence, the key cannot
            // exist in the table yet
            if meta == meta::EMPTY {
                return None;
            }

            // metadata match
            if meta & meta::H2 == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry was deleted before we could check it, but it's
                // possible that this was a false positive h2 match, so we
                // have to keep going
                if entry == Entry::TOMBSTONE {
                    i += 1;
                    continue 'probe;
                }

                // this is a true match
                if unsafe { (*entry).key.borrow() == key } {
                    unsafe { return Some(&(*entry).value) }
                }
            }

            // this slot either contains another key or a tombstone, keep searching
            i += 1;
        }

        // went over the max probe count: the key is not in the table
        // because any inserts would have triggered a resize at this point
        return None;
    }

    pub fn insert<'guard>(&self, key: K, value: V, guard: &'guard Guard<'_>) -> Option<&'guard V> {
        let entry = Box::into_raw(Box::new(Entry {
            key,
            value,
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

        'probe: loop {
            // went over the max probe count
            if i > limit {
                break;
            }

            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            if meta == meta::EMPTY {
                match unsafe { self.table.entry(i) }.compare_exchange(
                    Entry::EMPTY,
                    new_entry,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    // succesfully claimed this entry
                    Ok(_) => {
                        // update the meta byte
                        unsafe { self.table.meta(i).store(h2, Ordering::Release) };
                        return None;
                    }

                    // lost the entry to someone else, keep searching, unless
                    // it was a concurrent insert of the same key
                    Err(found) => {
                        // the key matches, perform an update
                        if unsafe { (*found).key == *key } {
                            match self.replace_entry(i, found, new_entry, guard) {
                                Ok(value) => return Some(value),
                                // lost to a concurrent delete, keep searching
                                Err(_) => continue 'probe,
                            }
                        }

                        continue 'probe;
                    }
                }
            }

            if meta & meta::H2 == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry was deleted before we could update it, we must now attempt
                // find a vacant slot or update a concurrent insert
                if entry == Entry::TOMBSTONE {
                    i += 1;
                    continue 'probe;
                }

                // 1/128 chance that we got a false positive from h2, so we must check full
                // equality to make sure
                if unsafe { (*entry).key == *key } {
                    match self.replace_entry(i, entry, new_entry, guard) {
                        Ok(value) => return Some(value),
                        // lost to a concurrent delete, keep searching
                        Err(_) => continue 'probe,
                    }
                }
            }

            // this slot either contains another key or a tombstone, keep searching
            i += 1;
        }

        // went over the max probe count.
        // the key is not in the table, trigger a resize
        let next_table = self.next_table();

        // help along the top-level copy
        let root = guard.protect(&self.root, Ordering::Acquire);
        let root = unsafe { Table::<K, V>::from_raw(root) };
        self.as_ref(root).help_copy(guard);

        // insert into the next table
        self.as_ref(next_table).insert_entry(new_entry, guard)
    }

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
                    // retire the old entry
                    guard.retire(entry, reclaim::boxed::<Entry<K, V>>);

                    // return the previous value
                    return Ok(&(*entry).value);
                },

                // lost to a concurrent delete, we must now attempt
                // find a vacant slot or update a concurrent insert
                //
                // we never reuse tombstones to maintain linearizability
                // against concurrent inserts, so it's as if this never happened.
                Err(found) if found == Entry::TOMBSTONE => return Err(()),

                // someone else updated before us
                Err(found) => {
                    // retry our update
                    entry = found;
                    continue;
                }
            }
        }
    }

    // only inserts if the entry does not already exist
    pub fn insert_copy<'guard>(
        &self,
        new_entry: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> Result<usize, usize> {
        let max_probes = log2!(self.table.capacity);
        let capacity = self.table.capacity - max_probes;

        let key = unsafe { &(*new_entry).key };

        let hash = self.hash(&key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (capacity - 1);
        let limit = i + max_probes;

        'probe: loop {
            // went over the max probe count
            if i > limit {
                break;
            }

            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            if meta == meta::EMPTY {
                match unsafe { self.table.entry(i) }.compare_exchange(
                    Entry::EMPTY,
                    new_entry,
                    Ordering::Release,
                    Ordering::Relaxed,
                ) {
                    // succesfully claimed this entry
                    Ok(_) => {
                        // update the meta byte
                        unsafe { self.table.meta(i).store(h2, Ordering::Release) };
                        return Ok(i);
                    }

                    // lost the entry to someone else
                    Err(found) => {
                        // the entry was already copied
                        if unsafe { (*found).key == *key } {
                            return Err(i);
                        }

                        // keep searching
                        continue 'probe;
                    }
                }
            }

            if meta & meta::H2 == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry was already copied
                if unsafe { (*entry).key == *key } {
                    return Err(i);
                }
            }

            // this slot either contains another key or a tombstone, keep searching
            i += 1;
        }

        // went over the max probe count, the key is not in the table
        let next_table = self.next_table();
        // don't abandon our copy to help along another one, just insert into
        // the next table and return to our current copy
        self.as_ref(next_table).insert_copy(new_entry, guard)
    }

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

        // we have the lock, so we are the thread to create
        // the table
        let cap = self.table.capacity;
        // let len = table.count.load(Ordering::Relaxed);

        // calculate the new table size
        let next_cap = match cap >> 20 {
            // // the table has capacity <1M and is >50% full
            // 0 if (len * 4) >= (cap * 2) => {
            //     // quadruple the capacity
            //     cap << 2
            // }
            // // or the table has capacity >1M and is >75% full
            // _ if (len * 4) >= (cap * 3) => {
            //     // quadruple the capacity
            //     cap << 2
            // }
            // otherwise double it
            _ => cap << 1,
        };

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

    pub fn remove<'guard, Q: ?Sized>(&self, key: &Q, guard: &'guard Guard<'_>) -> Option<&'guard V>
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

        'probe: loop {
            // went over the max probe count
            if i > limit {
                break;
            }

            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // encountered an empty meta in the probe sequence, the key
            // cannot exist in the table
            if meta == meta::EMPTY {
                return None;
            }

            if meta & meta::H2 == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry was deleted before we could check it, but it's
                // possible that this was a false positive h2 match, so we
                // have to keep going
                if entry == Entry::TOMBSTONE {
                    i += 1;
                    continue 'probe;
                }

                // this is a true match
                if unsafe { (*entry).key.borrow() == key } {
                    // replace it with a tombstone
                    let entry = unsafe {
                        self.table
                            .entry(i)
                            .swap(Entry::TOMBSTONE, Ordering::Release)
                    };

                    // retire the old entry if we were the one to delete
                    if entry != Entry::TOMBSTONE {
                        unsafe { guard.retire(entry, reclaim::boxed::<Entry<K, V>>) }
                    }

                    unsafe { return Some(&(*entry).value) }
                }
            }

            // this slot either contains another key or a tombstone, keep searching
            i += 1;
        }

        // went over the max probe count: the key is not in the table
        // because any inserts would have triggered a resize at this point
        return None;
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

            // - the table we copy from MUST be all set to COPIED before we perform updates in the
            //   new table to maintain consistency
            //      - every helper thread parks until that happens
            //
            // - if the new table is too small, we create a new new table
            //      - a nested table migration will always finish before it's parent, because `join_copy(); insert();` (1)
            //          - copied++ indicates that a table was copied from the old table to
            //            *somewhere*, not necessarily to our table. copied == capacity means the old
            //            table is dead for writers, which is what we need to allow new writes
            //            to the new table
            //         - we can't abandon the original table migration, we still have to use CAS
            //           copied to ensure we don't lose any entries. we don't know how many writers
            //           there are for them to acknowledge, without a more complex blocking mechanism.
            //      - we can't promote a table once it's parent is fully copied becase it might not have
            //        all the entries in the map, it's grandparent copy might still be in-progress
            //      - so we can only promote the root copy, even if it's full of COPIED, because
            //        only then are we sure the old root is fully copied from
            //      - so we wait till the root `copied == root.capacity`, and then we promote it,
            //        or it's most nested .next table (which is visible from copied.load(acquire)).
            //      - we know this most nested table is complete because (1), and we know it
            //        contains all entries because our copy is complete, which means every copy
            //        before the most nested one is full of COPIED, thus all the entries are in the
            //        most nested one and we are in a stable state.
            //        - any time we try to copy we have to check that said copy is not already
            //          completed, we don't have to try to promote it because copies will always be
            //          promoted by the last copier, or a parent copy. there is no race.
            //
            //  - we can't do the `loop { x.store(x.load); cas(x, COPIED) }` because a nested copy
            //    might pick up our in complete entry. we can only write up-to-date values to the
            //    new table.

            // we ..
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
}

#[inline]
fn h1(hash: u64) -> usize {
    hash as usize
}

mod meta {
    use std::mem;

    // an empty slot
    pub const EMPTY: u8 = 0b1111_1111;

    // a deleted slot
    pub const TOMBSTONE: u8 = 0b1000_0000;

    // h2 mask, ignoring the first bit which is only
    // set for TOMBSTONE and EMPTY
    pub const H2: u8 = !0x80;

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
