// TODO: if COPIED entry is still in old table, can't retire from new table after overwrite
// SOLUTION:
//       - refcount?
//       - no incremental migration, so retire after make new table visible to readers and writers (how does this work for writers)?
//       - delete COPIED entry from old table before copying to new table, make sure it's impossible for writers to go to the new table (how? mark everything in chain as COPIED, or see PHANTOM entry)
mod alloc;
mod utils;
mod x86_64;
// mod bit_lock;

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{self, AtomicPtr, Ordering};
use std::{hint, mem, ptr};

use self::alloc::{RawTable, ResizeState};
use self::utils::StrictProvenance;
use crate::seize::{self, reclaim, AsLink, Collector, Guard, Link, Linked};
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
    // the entry is being copied to the new table, no updates
    // are allowed on the old table
    const COPYING: usize = 0b001;

    // the entry does not currently contain a value, i.e. it was deleted
    // or it's value was copied to the new table
    const TOMBSTONE: usize = 0b010;

    // the (non-empty) entry has been copied to the new table
    const COPIED: usize = 0b100;

    // a tombstone entry has been copied to the new table
    const TOMBCOPIED: usize = Entry::TOMBSTONE | Entry::COPYING;

    // mask for entry pointer, ignoring tag bits
    const POINTER: usize = !(Entry::COPYING | Entry::TOMBSTONE | Entry::COPIED);
}

impl<K, V, S> HashMap<K, V, S> {
    pub fn with_capacity_and_hasher(capacity: usize, build_hasher: S) -> HashMap<K, V, S> {
        // allocate extra buffer capacity the same length as the probe limit.
        // this allows us to avoid overflow checks
        let capacity = capacity.next_power_of_two();
        let buffer = log2!(capacity);

        let collector = Collector::new().epoch_frequency(None);
        let table = alloc::Table::<Entry<K, V>>::new(capacity, capacity + buffer, collector.link());

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
        let max_probes = log2!(self.table.len);

        let hash = self.hash(&key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let before = i & 15;
        i &= !15;

        let start = i;
        let limit = i + max_probes + before;

        let before_mask: u128 = ((!0) << (before * 8) >> before * 8);
        let before_mask = unsafe { mem::transmute(before_mask.to_be_bytes()) };

        while i <= limit {
            let mut meta = unsafe { x86_64::load_128(self.table.meta_ptr(i).cast::<u128>()) };

            for bit in x86_64::match_byte(meta, h2) {
                let i = i + bit;

                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // this is a true match
                if unsafe { (*entry_ptr).key.borrow() == key } {
                    // the entry existed in the table but was deleted
                    if entry.addr() & Entry::TOMBSTONE != 0 {
                        return None;
                    }

                    // we don't care if the entry is currently being copied, because any
                    // concurrent inserts of the same key would have marked the entry as COPIED
                    // after they complete. thus this is still the latest value.
                    unsafe { return Some((*entry_ptr).value.assume_init_ref()) }
                }
            }

            // make sure not to search the bytes outside the probe chain
            if i == start {
                meta = unsafe { std::arch::x86_64::_mm_and_si128(before_mask, meta) };
            }

            if x86_64::match_byte(meta, meta::EMPTY).any_set() {
                return None;
            }

            // the slot contained a different key, keep searching
            i += 16;
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

        match self.insert_entry(entry, guard) {
            EntryStatus::Empty | EntryStatus::Tombstone => None,
            EntryStatus::Value(value) => Some(value),
        }
    }

    fn insert_entry<'guard>(
        &self,
        new_entry: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> EntryStatus<&'guard V> {
        let max_probes = log2!(self.table.len);

        let key = unsafe { &(*new_entry.map_addr(|addr| addr & Entry::POINTER)).key };

        let hash = self.hash(&key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let limit = i + max_probes;

        'probe: while i <= limit {
            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // migration in progress, switch to the new table
            if meta == meta::PHANTOM {
                break 'probe;
            }

            // possible empty entry
            if meta == meta::EMPTY {
                match unsafe { self.table.entry(i) }.compare_exchange(
                    ptr::null_mut(),
                    new_entry,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    // successfully claimed this entry
                    Ok(_) => {
                        // update the meta byte
                        unsafe { self.table.meta(i).store(h2, Ordering::Release) };
                        return EntryStatus::Empty;
                    }

                    // lost the entry to someone else, keep searching, unless
                    // it was a concurrent insert of the same key
                    Err(found) => {
                        // found phantom entry, we can safely move to the new table.
                        if found.addr() == Entry::COPYING {
                            // make sure readers know to go to the new table too
                            unsafe { self.table.meta(i).store(meta::PHANTOM, Ordering::Release) };
                            break 'probe;
                        }

                        let found_ptr = found.map_addr(|addr| addr & Entry::POINTER);

                        // the key matches, we might be able to perform an update
                        if unsafe { (*found_ptr).key == *key } {
                            return self.replace_entry(i, found, new_entry, guard);
                        }

                        continue 'probe;
                    }
                }
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };
                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key == *key } {
                    return self.replace_entry(i, entry, new_entry, guard);
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

    // Replaces the value of an existing entry, returning the previous value if successful.
    //
    // Inserts into the new table if the entry is being copied.
    fn replace_entry<'guard>(
        &self,
        i: usize,
        mut entry: *mut Entry<K, V>,
        new_entry: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> EntryStatus<&'guard V> {
        let found = loop {
            // the entry is being copied to a new table, we have to go there
            // and join the race for the insertion
            if entry.addr() & Entry::COPYING != 0 {
                break entry;
            }

            match unsafe { self.table.entry(i) }.compare_exchange_weak(
                entry,
                new_entry,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                // succesful update
                Ok(_) => unsafe {
                    let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                    // we replaced a tombstone
                    if entry.addr() & Entry::TOMBSTONE != 0 {
                        guard.retire(entry_ptr, reclaim_tombstone::<K, V>);
                        return EntryStatus::Tombstone;
                    }

                    // retire the old entry
                    guard.retire(entry_ptr, reclaim_entry::<K, V>);
                    return EntryStatus::Value((*entry_ptr).value.assume_init_ref());
                },

                // the entry is being copied to the new table, retry there
                Err(found) if found.addr() & Entry::COPYING != 0 => {
                    break found;
                }

                // lost to a concurrent update or delete, retry
                Err(found) => {
                    entry = found;
                    continue;
                }
            }
        };

        let next_table = self.next_table_ref().unwrap();

        // insert into the new table
        match next_table.insert_entry(new_entry, guard) {
            // if we claimed the slot before the copy, we have to update the copy count
            EntryStatus::Empty => {
                // the thread that put down COPYING will update the copy counts,
                // we overwrote a tombstone
                if found.addr() & Entry::TOMBSTONE != 0 {
                    return EntryStatus::Tombstone;
                }

                self.try_promote(next_table.table, 1, guard);

                // mark the entry as copied so readers know to look in the new table
                let copied = found.map_addr(|addr| addr | Entry::COPIED);

                // mark the entry as copied, or someone else will
                match unsafe { self.table.entry(i) }.compare_exchange(
                    found,
                    copied,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {}
                    Err(entry) => assert!(entry.addr() & Entry::COPIED == Entry::COPIED),
                }

                let found_ptr = found.map_addr(|addr| addr & Entry::POINTER);

                // our insertion didn't overwrite anything in the new table, but logically,
                // we did overwrite the value we found in the old table
                unsafe { EntryStatus::Value((*found_ptr).value.assume_init_ref()) }
            }
            status => status,
        }
    }

    pub fn remove<'guard, Q: ?Sized>(&self, key: &Q, guard: &'guard Guard<'_>) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.remove_with(key, ptr::null_mut(), guard)
    }

    fn remove_with<'guard, Q: ?Sized>(
        &self,
        key: &Q,
        mut deletion: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> Option<&'guard V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        let max_probes = log2!(self.table.len);

        let hash = self.hash(&key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let limit = i + max_probes;

        while i <= limit {
            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // migration in progress, switch to the new table
            if meta == meta::PHANTOM {
                break;
            }

            // encountered an empty entry in the probe sequence, the key cannot exist in the table
            if meta == meta::EMPTY {
                return None;
            }

            // possible match
            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };
                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key.borrow() == key } {
                    // the entry is already deleted
                    if entry.addr() & Entry::TOMBSTONE != 0 && entry.addr() & Entry::COPYING == 0 {
                        return None;
                    }

                    let owned_key = unsafe { (*entry_ptr).key.clone() };

                    if deletion.is_null() {
                        // allocate the entry that marks the key as deleted
                        deletion = Box::into_raw(Box::new(Entry {
                            key: owned_key,
                            value: MaybeUninit::uninit(),
                            link: self.collector.link(),
                        }));

                        deletion = deletion.map_addr(|addr| addr | Entry::TOMBSTONE);
                    }

                    // the entry is being copied to a new table, we have to go there and
                    // delete it, or put down our deletion first
                    if entry.addr() & Entry::COPYING != 0 {
                        return self.delete_copy(i, entry, deletion, guard);
                    }

                    loop {
                        // perform the deletion
                        match unsafe { self.table.entry(i) }.compare_exchange_weak(
                            entry,
                            deletion,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            // succesfully deleted
                            Ok(_) => unsafe {
                                let entry = entry.map_addr(|addr| addr & Entry::POINTER);

                                // retire the old entry
                                guard.retire(entry, reclaim_entry::<K, V>);

                                // return the previous value
                                return Some((*entry).value.assume_init_ref());
                            },

                            // the entry is being copied to the new table, retry there
                            Err(found) if found.addr() & Entry::COPYING != 0 => {
                                return self.delete_copy(i, entry, deletion, guard);
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

        // went over the max probe count: the key is not in this table as
        // any inserts would have triggered a resize at this point, but it
        // might be in the new table.
        if let Some(next_table) = self.next_table_ref() {
            return next_table.remove_with(key, deletion, guard);
        }

        // the key is not in the table and there is no active migration
        None
    }

    fn delete_copy<'guard>(
        &self,
        i: usize,
        found: *mut Entry<K, V>,
        deletion: *mut Entry<K, V>,
        guard: &'guard Guard<'_>,
    ) -> Option<&'guard V> {
        let next_table = self.next_table_ref().unwrap();

        // insert our deletion into the new table
        // TODO: this will unnecessarily overwrite existing tombstones, use a custom loop here
        match next_table.insert_entry(deletion, guard) {
            // if we claimed the slot before the copy, we have to update the copy count
            EntryStatus::Empty => {
                // the thread that put down COPYING will update the copy counts,
                // the entry was already deleted, but we made sure it wasn't in
                // the new table.
                if found.addr() & Entry::TOMBSTONE != 0 {
                    return None;
                }

                self.try_promote(next_table.table, 1, guard);

                // mark the entry as copied so readers know to look in the new table
                let copied = found.map_addr(|addr| addr | Entry::COPIED);

                // mark the entry as copied, or someone else will
                match unsafe { self.table.entry(i) }.compare_exchange(
                    found,
                    copied,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {}
                    Err(entry) => {
                        assert!(entry.addr() & Entry::COPIED == Entry::COPIED)
                    }
                }

                // our insertion didn't overwrite anything in the new table, but logically,
                // we did overwrite the value we found in the old table
                let found_ptr = found.map_addr(|addr| addr & Entry::POINTER);
                unsafe { Some((*found_ptr).value.assume_init_ref()) }
            }
            // the entry was already deleted in the new table
            EntryStatus::Tombstone => None,
            EntryStatus::Value(value) => Some(value),
        }
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
        // calculate the new table size (TODO: store len?)
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

        loop {
            // claim a range to copy
            let copy_start = state.claim.fetch_add(copy_chunk, Ordering::Relaxed);

            let mut copied = 0;
            for i in 0..copy_chunk {
                // copies wrap around. note the capacity including the buffer is not a
                // power of two
                let i = (copy_start + i) % capacity;

                // keep track of the entries we actually copy
                if self.copy_entry_to(i, next_table, guard) {
                    copied += 1;
                }
            }

            // are we done?
            if self.try_promote(next_table, copied, guard) {
                return;
            }
        }
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

        // are we done?
        let x = self.table.len + log2!(self.table.len);
        if copied == self.table.len + log2!(self.table.len) {
            let root = self.root.load(Ordering::Acquire);

            // we only promote the top-level copy
            if root == self.table.raw {
                if self
                    .root
                    .compare_exchange(root, next.raw, Ordering::Release, Ordering::Relaxed)
                    .is_ok()
                {
                    // retire the old table
                    unsafe {
                        guard.retire(root, |link| {
                            let raw: *mut RawTable = link.cast();
                            let table: Table<K, V> = Table::from_raw(raw);
                            Table::dealloc(table);
                        })
                    };
                }
            }

            return true;
        }

        return false;
    }

    // Copy the entry at the given index to the new table.
    //
    // Returns true if this thread ended up doing the copy.
    fn copy_entry_to(&self, i: usize, new_table: Table<K, V>, guard: &Guard<'_>) -> bool {
        let mut entry = unsafe { self.table.entry(i).load(Ordering::Acquire) };

        loop {
            // the entry has already been copied, or was empty
            if entry.addr() & Entry::COPIED != 0
                || entry.addr() == Entry::COPYING
                || entry.addr() & Entry::TOMBCOPIED == Entry::TOMBCOPIED
            {
                return false;
            }

            // the entry is already marked
            if entry.addr() & Entry::COPYING != 0 {
                break;
            }

            // mark the entry as being copied
            match unsafe {
                self.table.entry(i).compare_exchange_weak(
                    entry,
                    entry.map_addr(|addr| addr | Entry::COPYING),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
            } {
                // the entry was empty, so we're done
                Ok(_) if entry.is_null() => {
                    // mark this as a phantom entry so inserts can move to the new table early
                    unsafe { self.table.meta(i).store(meta::PHANTOM, Ordering::Release) };
                    return true;
                }

                // the entry was a tombstone, so we're done
                // TODO: don't leak
                Ok(_) if entry.addr() & Entry::TOMBSTONE != 0 => return true,

                // otherwise we have to copy the value
                Ok(_) => break,

                // something changed, retry
                Err(found) => entry = found,
            }
        }

        // copy the entry to the new table
        entry = entry.map_addr(|addr| addr & !Entry::COPYING);
        let copied = self.as_ref(new_table).insert_copy(entry, guard);

        // mark the entry as copied, so no one else tries to copy it
        while entry.addr() & Entry::COPIED == 0 {
            match unsafe {
                self.table.entry(i).compare_exchange_weak(
                    entry,
                    entry.map_addr(|addr| addr | Entry::COPIED),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
            } {
                Ok(_) => break,

                // retry
                Err(found) => entry = found,
            }
        }

        return copied;
    }

    // Copy an entry into the table.
    //
    // Any matching key found in the table is considered to overwrite the copy.
    pub fn insert_copy<'guard>(&self, copy: *mut Entry<K, V>, guard: &'guard Guard<'_>) -> bool {
        let max_probes = log2!(self.table.len);

        let key = unsafe { &(*copy.map_addr(|addr| addr & Entry::POINTER)).key };

        let hash = self.hash(&key);
        let h2 = meta::h2(hash);

        let mut i = h1(hash) & (self.table.len - 1);
        let limit = i + max_probes;

        'probe: while i <= limit {
            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // migration in progress, switch to the new table
            if meta == meta::PHANTOM {
                break 'probe;
            }

            // possible empty entry
            if meta == meta::EMPTY {
                match unsafe { self.table.entry(i) }.compare_exchange(
                    ptr::null_mut(),
                    copy,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    // successfully claimed this entry
                    Ok(_) => {
                        // update the meta byte
                        unsafe { self.table.meta(i).store(h2, Ordering::Release) };
                        return true;
                    }

                    // lost the entry to someone else, keep searching, unless
                    // it was a concurrent insert of the same key
                    Err(found) => {
                        // found phantom entry, we can move to the nested resize
                        if found.addr() == Entry::COPYING {
                            // make sure readers know to go to the new table too
                            unsafe { self.table.meta(i).store(meta::PHANTOM, Ordering::Release) };
                            break 'probe;
                        }

                        let found_ptr = found.map_addr(|addr| addr & Entry::POINTER);

                        // someone else copied the key, or overwrote the old value, we're done
                        if unsafe { (*found_ptr).key == *key } {
                            return false;
                        }

                        continue 'probe;
                    }
                }
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };
                let entry_ptr = entry.map_addr(|addr| addr & Entry::POINTER);

                // someone else copied the key, or overwrote the old value, we're done
                if unsafe { (*entry_ptr).key == *key } {
                    return false;
                }
            }

            // this entry either contains another key, keep searching
            i += 1;
        }

        // went over the max probe count: trigger a nested resize.
        // the entry can be safely inserted directly into the new table
        // as any other concurrent insertions will also see the full
        // probe sequence.
        let next_table = self.next_table();
        self.as_ref(next_table).insert_copy(copy, guard)
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

impl<K, V, S> Drop for HashMap<K, V, S> {
    fn drop(&mut self) {
        let table = unsafe { Table::<K, V>::from_raw(*self.table.get_mut()) };

        let capacity = table.len + log2!(table.len);
        for i in 0..capacity {
            let entry = unsafe { (table.entry(i) as *const _ as *const _ as *mut Entry<K, V>) };

            if entry.addr() == Entry::COPIED {
                continue;
            }

            if entry.addr() == Entry::TOMBCOPIED {
                continue;
            }
        }
    }
}

unsafe fn reclaim_entry<K, V>(link: *mut Link) {
    let entry_addr: *mut Entry<K, V> = link.cast();
    let entry = unsafe { Box::from_raw(entry_addr.map_addr(|addr| addr & Entry::POINTER)) };

    // drop the value
    let _ = unsafe { entry.value.assume_init() };
}

unsafe fn reclaim_tombstone<K, V>(link: *mut Link) {
    seize::reclaim::boxed::<Entry<K, V>>(link)
}

enum EntryStatus<V> {
    Empty,
    Tombstone,
    Value(V),
}

#[inline]
fn h1(hash: u64) -> usize {
    hash as usize
}

mod meta {
    use std::mem;

    // marks an empty entry
    pub const EMPTY: u8 = 0x80;

    // marks an empty entry that was copied to the new table.
    // [NULL -> COPIED] marks a linearization point for readers/writers
    // in the old table to move to the new table, as it is guaranteed to be
    // seen by all writers when they attempt to claim the entry.
    pub const PHANTOM: u8 = u8::MAX;

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
