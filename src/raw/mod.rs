mod alloc;
mod utils;

use std::borrow::Borrow;
use std::hash::{BuildHasher, Hash, Hasher};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::sync::atomic::{fence, AtomicPtr, Ordering};
use std::{hint, ptr};

use self::alloc::{RawTable, State};
use self::utils::{AtomicPtrFetchOps, Counter, StrictProvenance};

use seize::{AsLink, Collector, Guard, Link};

// A lock-free hash-map.
pub struct HashMap<K, V, S> {
    pub collector: Collector,
    pub hash_builder: S,
    table: AtomicPtr<RawTable>,
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

// The state of an entry that was just updated.
pub enum EntryStatus<'g, V> {
    Empty(&'g V),
    Tombstone(&'g V),
    Replaced(&'g V),
    Error { current: &'g V, not_inserted: V },
}

enum ReplaceStatus<V> {
    HelpCopy,
    Removed,
    Replaced(V),
}

// Safety: repr(C) and seize::Link is the first field
unsafe impl<K, V> AsLink for Entry<K, V> {}

impl Entry<(), ()> {
    // The entry is being copied to the new table, no updates are allowed on the old table.
    const COPYING: usize = 0b01;

    // The entry was deleted.
    const TOMBSTONE: usize = 0b10;

    // Mask for entry pointer, ignoring tag bits.
    const POINTER: usize = !(Entry::COPYING | Entry::TOMBSTONE);

    // Retires an entry.
    unsafe fn retire<K, V>(link: *mut Link) {
        let entry_addr: *mut Entry<K, V> = link.cast();
        let _entry = unsafe { Box::from_raw(entry_addr) };
    }
}

// The probe-limit for the table.
macro_rules! probe_limit {
    ($capacity:expr) => {
        // 6 * log2(capacity)
        6 * ((usize::BITS as usize) - ($capacity.leading_zeros() as usize) - 1)
    };
}

impl<K, V, S> HashMap<K, V, S> {
    // Creates a table with the given capacity and hasher.
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> HashMap<K, V, S> {
        let collector = Collector::new().epoch_frequency(None);

        // the table is lazily allocated
        if capacity == 0 {
            return HashMap {
                collector,
                hash_builder,
                table: AtomicPtr::new(ptr::null_mut()),
                _kv: PhantomData,
            };
        }

        // round the capacity to the next power of two based on an estimated load factor
        // to hold `capacity` elements
        let capacity = capacity.next_power_of_two();

        // allocate buffer capacity the same length as the probe limit. this allows us to avoid overflow checks
        let buffer = probe_limit!(capacity);

        let table = Table::<K, V>::new(capacity, capacity + buffer, collector.link());

        HashMap {
            collector,
            hash_builder,
            table: AtomicPtr::new(table.raw),
            _kv: PhantomData,
        }
    }

    // Returns the capacity of the table.
    pub fn capacity<'g>(&self, guard: &'g Guard<'_>) -> usize {
        self.root(guard).table.len
    }

    // Returns a reclamation guard.
    pub fn guard(&self) -> Guard<'_> {
        self.collector.enter()
    }

    // Returns a reference to the root hash table.
    #[inline(always)]
    pub fn root<'g>(&self, guard: &'g Guard<'_>) -> HashMapRef<'_, K, V, S> {
        if let Some(c) = guard.collector() {
            assert!(
                Collector::ptr_eq(c, &self.collector),
                "accessed map with incorrect guard"
            )
        }

        let raw = guard.protect(&self.table, Ordering::Acquire);
        let table = unsafe { Table::<K, V>::from_raw(raw) };
        self.as_ref(table)
    }

    // Returns a reference to the given table.
    fn as_ref<'a>(&'a self, table: Table<K, V>) -> HashMapRef<'a, K, V, S> {
        HashMapRef {
            table,
            root: &self.table,
            collector: &self.collector,
            hash_builder: &self.hash_builder,
        }
    }
}

// A reference to the root hash table, or an arbitrarily nested table migration.
pub struct HashMapRef<'a, K, V, S> {
    table: Table<K, V>,
    root: &'a AtomicPtr<RawTable>,
    collector: &'a Collector,
    hash_builder: &'a S,
}

impl<K, V, S> HashMapRef<'_, K, V, S>
where
    K: Sync + Send + Hash + Eq,
    V: Sync + Send,
    S: BuildHasher,
{
    // Returns a reference to the value corresponding to the key.
    pub fn get_entry<'g, Q>(&self, key: &Q, guard: &'g Guard<'_>) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        if self.table.raw.is_null() {
            return None;
        }

        let hash = self.hash(key);
        let h2 = meta::h2(hash);

        let i = h1(hash) & (self.table.len - 1);
        let limit = i + probe_limit!(self.table.len);

        for i in i..=limit {
            let meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            if meta == meta::EMPTY {
                return None;
            }

            // the entry is not the table
            if meta == meta::TOMBSTONE {
                continue;
            }

            // potential match
            if meta == h2 {
                let entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry was deleted
                if entry.addr() & Entry::TOMBSTONE != 0 {
                    continue;
                }

                let entry_ptr = unsafe { &*entry.mask(Entry::POINTER) };

                // check for a full match
                if entry_ptr.key.borrow() == key {
                    return Some((&entry_ptr.key, &entry_ptr.value));
                }
            }
        }

        None
    }

    // Returns an iterator over the keys and values of this table.
    pub fn iter<'g>(&self, guard: &'g Guard<'_>) -> Iter<'g, K, V> {
        Iter {
            i: 0,
            table: self.table,
            capacity: self.table.len + probe_limit!(self.table.len),
            _guard: guard,
        }
    }

    // Inserts a key-value pair into the map.
    pub fn insert<'g>(
        &mut self,
        key: K,
        value: V,
        replace: bool,
        guard: &'g Guard<'_>,
    ) -> EntryStatus<'g, V> {
        let entry = Box::into_raw(Box::new(Entry {
            key,
            value,
            link: self.collector.link(),
        }));

        self.insert_entry(entry, replace, guard)
    }

    // Inserts an entry into the map.
    fn insert_entry<'g>(
        &mut self,
        new_entry: *mut Entry<K, V>,
        replace: bool,
        guard: &'g Guard<'_>,
    ) -> EntryStatus<'g, V> {
        if self.table.raw.is_null() {
            self.init();
        }

        let new_ref = unsafe { &*new_entry.mask(Entry::POINTER) };

        let hash = self.hash(&new_ref.key);
        let h2 = meta::h2(hash);

        let i = h1(hash) & (self.table.len - 1);
        let limit = i + probe_limit!(self.table.len);

        for i in i..=limit {
            let meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            // we can't reuse tombstones
            if meta == meta::TOMBSTONE {
                continue;
            }

            if meta == meta::EMPTY {
                match unsafe { self.table.entry(i) }.compare_exchange(
                    ptr::null_mut(),
                    new_entry,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    // successfully claimed this entry
                    Ok(_) => {
                        unsafe { self.table.meta(i).store(h2, Ordering::Release) };

                        // we inserted a new entry, update the entry count
                        self.table.state().count.get(guard.thread.id).insert(1);
                        return EntryStatus::Empty(&new_ref.value);
                    }
                    Err(found) => {
                        fence(Ordering::Acquire);

                        let found_ptr = found.mask(Entry::POINTER);

                        // the entry was deleted or copied
                        if found_ptr.is_null() {
                            continue;
                        }

                        // ensure the meta table is updated to avoid breaking the probe chain
                        unsafe {
                            if self.table.meta(i).load(Ordering::Acquire) == meta::EMPTY {
                                let hash = self.hash(&(*found_ptr).key);
                                self.table.meta(i).store(meta::h2(hash), Ordering::Release);
                            }
                        }

                        // if the same key was just inserted, we might be able to update
                        if unsafe { (*found_ptr).key == new_ref.key } {
                            // don't replace the existing value, bail
                            if !replace {
                                let new_entry = unsafe { Box::from_raw(new_entry) };
                                let current = unsafe { &(*found_ptr).value };

                                return EntryStatus::Error {
                                    current,
                                    not_inserted: new_entry.value,
                                };
                            }

                            match self.replace_entry(i, found, new_entry, guard) {
                                // the entry was deleted before we could update it, keep probing
                                ReplaceStatus::Removed => {}
                                // the entry is being copied
                                ReplaceStatus::HelpCopy => break,
                                // successful update
                                ReplaceStatus::Replaced(value) => {
                                    return EntryStatus::Replaced(value)
                                }
                            }
                        }

                        continue;
                    }
                }
            }

            // potential match
            if meta == h2 {
                let entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry was deleted
                if entry.addr() & Entry::TOMBSTONE != 0 {
                    continue;
                }

                let entry_ptr = entry.mask(Entry::POINTER);

                // if the key matches, we might be able to update
                if unsafe { (*entry_ptr).key == new_ref.key } {
                    // don't replace the existing value, bail
                    if !replace {
                        let new_entry = unsafe { Box::from_raw(new_entry) };
                        let current = unsafe { &(*entry_ptr).value };

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
        }

        // went over the max probe count or found a copied entry: trigger a resize.
        self.get_or_alloc_next(None);
        let next_table = self.help_copy(guard);
        self.as_ref(next_table)
            .insert_entry(new_entry, replace, guard)
    }

    // Replaces the value of an existing entry, returning the previous value if successful.
    //
    // Inserts into the new table if the entry is being copied.
    fn replace_entry<'g>(
        &self,
        i: usize,
        mut entry: *mut Entry<K, V>,
        new_entry: *mut Entry<K, V>,
        guard: &'g Guard<'_>,
    ) -> ReplaceStatus<&'g V> {
        loop {
            // the entry is being copied to a new table, we have to finish the resize before we insert
            if entry.addr() & Entry::COPYING != 0 {
                return ReplaceStatus::HelpCopy;
            }

            match unsafe { self.table.entry(i) }.compare_exchange_weak(
                entry,
                new_entry,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                // succesful update
                Ok(_) => unsafe {
                    let entry_ptr = entry.mask(Entry::POINTER);

                    // retire the old value
                    guard.defer_retire(entry_ptr, Entry::retire::<K, V>);
                    return ReplaceStatus::Replaced(&(*entry_ptr).value);
                },

                // lost to a delete
                Err(found) if found.addr() & Entry::TOMBSTONE != 0 => {
                    return ReplaceStatus::Removed;
                }

                // lost to a concurrent update, retry
                Err(found) => {
                    entry = found;
                }
            }
        }
    }

    // Update an entry with a remapping function.
    pub fn update<'g, F>(&self, key: K, f: F, guard: &'g Guard<'_>) -> Option<&'g V>
    where
        F: Fn(&V) -> V,
    {
        if self.table.raw.is_null() {
            return None;
        }

        let update = Box::into_raw(Box::new(Entry {
            key,
            link: self.collector.link(),
            value: MaybeUninit::uninit(),
        }));

        self.update_with(update, false, f, guard)
    }

    // Update an entry with a remapping function.
    pub fn update_with<'g, F>(
        &self,
        update: *mut Entry<K, MaybeUninit<V>>,
        mut value_present: bool,
        f: F,
        guard: &'g Guard<'_>,
    ) -> Option<&'g V>
    where
        F: Fn(&V) -> V,
    {
        let hash = unsafe { self.hash(&(*update).key) };
        let h2 = meta::h2(hash);

        let i = h1(hash) & (self.table.len - 1);
        let limit = i + probe_limit!(self.table.len);
        let mut copying = false;

        'probe: for i in i..=limit {
            let meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);

            // the key is not in this table
            if meta == meta::EMPTY {
                return None;
            }

            // the entry was deleted
            if meta == meta::TOMBSTONE {
                continue;
            }

            // potential match
            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry was deleted
                if entry.addr() & Entry::TOMBSTONE != 0 {
                    continue;
                }

                let entry_ptr = entry.mask(Entry::POINTER);

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key == (*update).key } {
                    loop {
                        // the entry is being copied to a new table, we have to copy it before we can update it
                        if entry.addr() & Entry::COPYING != 0 {
                            copying = true;
                            break 'probe;
                        }

                        // construct the new value
                        unsafe {
                            let value = f(&(*entry_ptr).value);

                            // drop the old value, if `f` has already been called
                            if value_present {
                                (*update).value.assume_init_drop();
                            }

                            (*update).value = MaybeUninit::new(value);
                            value_present = true;
                        }

                        match unsafe { self.table.entry(i) }.compare_exchange_weak(
                            entry,
                            update as _,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            // succesful update
                            Ok(_) => unsafe {
                                let entry_ptr = entry.mask(Entry::POINTER);

                                // retire the old entry
                                guard.defer_retire(entry_ptr, Entry::retire::<K, V>);
                                return Some(&(*entry_ptr).value);
                            },

                            // the entry got deleted
                            Err(found) if found.addr() & Entry::TOMBSTONE != 0 => {
                                return None;
                            }

                            // lost to a concurrent update or delete, retry
                            Err(found) => {
                                // drop the old value
                                unsafe { (*update).value.assume_init_drop() }
                                entry = found;
                            }
                        }
                    }
                }
            }
        }

        if copying {
            // found a copied entry, finish the resize and update it in the new table
            self.next_table_ref().unwrap();
            let next_table = self.help_copy(guard);
            return self
                .as_ref(next_table)
                .update_with(update, value_present, f, guard);
        }

        None
    }

    pub fn reserve(&self, additional: usize, guard: &Guard<'_>) {
        loop {
            let capacity = (self.len() + additional).next_power_of_two();
            // we have enough capacity
            if self.table.len >= capacity {
                return;
            }

            self.get_or_alloc_next(Some(capacity));
            self.help_copy(guard);
        }
    }

    // Allocate the inital table.
    fn init<'g>(&mut self) {
        const CAPACITY: usize = 32;
        const BUFFER: usize = probe_limit!(CAPACITY);

        let table = Table::<K, V>::new(CAPACITY, CAPACITY + BUFFER, self.collector.link());
        match self.root.compare_exchange(
            ptr::null_mut(),
            table.raw,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => self.table = table,
            // someone us allocated before us, deallocate our table
            Err(found) => {
                unsafe { Table::dealloc(table) }
                self.table = unsafe { Table::from_raw(found) };
            }
        }
    }

    // Help along the resize operation until it completes.
    fn help_copy(&self, guard: &Guard<'_>) -> Table<K, V> {
        // while we only copy from the root table, if the new allocation runs out of space the
        // copy must be aborted, so we use the resize state of the aborted table to manage the
        // next resize attempt. `curr` represents the root, or last aborted table.
        let mut curr = self.table;

        let next = curr.state().next.load(Ordering::Acquire);
        assert!(!next.is_null());
        let mut next = unsafe { Table::<K, V>::from_raw(next) };

        'copy: loop {
            // make sure we are at the correct table
            while curr.state().status.load(Ordering::Acquire) == State::ABORTED {
                curr = next;
                next = self.as_ref(next).get_or_alloc_next(None);
            }

            // the copy already completed
            if self.try_promote(curr.state(), next, 0, guard) {
                return next;
            }

            // the true table capacity, we have to copy every entry including from the buffer
            let capacity = self.table.len + probe_limit!(self.table.len);
            let copy_chunk = capacity.min(1024);

            loop {
                // every entry has already been claimed
                if curr.state().claim.load(Ordering::Acquire) >= capacity {
                    break;
                }

                // claim a range to copy
                let copy_start = curr.state().claim.fetch_add(copy_chunk, Ordering::AcqRel);

                let mut copied = 0;
                for i in 0..copy_chunk {
                    let i = copy_start + i;

                    if i >= capacity {
                        break;
                    }

                    // if this table doesn't have space, we have to abort the copy and allocate a
                    // new table
                    if !self.copy_index(i, next) {
                        // abort the copy
                        curr.state().status.store(State::ABORTED, Ordering::Release);
                        let allocated = self.as_ref(next).get_or_alloc_next(None);
                        atomic_wait::wake_all(&curr.state().status);

                        // retry in a new table
                        curr = next;
                        next = allocated;
                        continue 'copy;
                    }

                    copied += 1;
                }

                // are we done?
                if self.try_promote(curr.state(), next, copied, guard) {
                    return next;
                }
            }

            // we copied all that we can, wait for the table to be promoted
            loop {
                let status = curr.state().status.load(Ordering::Acquire);

                // if this copy was aborted, we have to retry in the new table
                if status == State::ABORTED {
                    continue 'copy;
                }

                // the copy is complete
                if status == State::COMPLETE {
                    return next;
                }

                atomic_wait::wait(&curr.state().status, State::PENDING);
            }
        }
    }

    // Copy the entry at the given index to the new table.
    //
    // Returns true if the entry was copied by this thread,
    fn copy_index(&self, i: usize, new_table: Table<K, V>) -> bool {
        let mut entry = unsafe { self.table.entry(i) }.load(Ordering::Acquire);

        if entry.addr() & Entry::COPYING == 0 {
            // mark the entry as copying
            entry = unsafe {
                self.table
                    .entry(i)
                    .fetch_or(Entry::COPYING, Ordering::AcqRel)
            };
        }

        // there is nothing to copy
        if entry.mask(Entry::POINTER).is_null() {
            unsafe { self.table.meta(i).store(meta::TOMBSTONE, Ordering::Release) };
            return true;
        } else if entry.addr() & Entry::TOMBSTONE != 0 {
            return true;
        }

        // otherwise, copy the value
        let entry = entry.mask(Entry::POINTER);
        self.as_ref(new_table).insert_copy(entry)
    }

    // Copy an entry into the table.
    fn insert_copy<'g>(&self, new_entry: *mut Entry<K, V>) -> bool {
        let key = unsafe { &(*new_entry.mask(Entry::POINTER)).key };

        let hash = self.hash(key);

        let i = h1(hash) & (self.table.len - 1);
        let limit = i + probe_limit!(self.table.len);

        for i in i..=limit {
            let meta = unsafe { self.table.meta(i) }.load(Ordering::Acquire);
            assert_ne!(meta, meta::TOMBSTONE);

            if meta == meta::EMPTY {
                let entry = unsafe { self.table.entry(i) };

                // try to claim the entry
                match entry.compare_exchange(
                    ptr::null_mut(),
                    new_entry,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        unsafe { self.table.meta(i).store(meta::h2(hash), Ordering::Release) };
                        return true;
                    }
                    Err(found) => {
                        let found_ptr = found.mask(Entry::POINTER);
                        assert!(!found_ptr.is_null());

                        // ensure the meta table is updated to avoid breaking the probe chain
                        unsafe {
                            if self.table.meta(i).load(Ordering::Acquire) == meta::EMPTY {
                                let hash = self.hash(&(*found_ptr).key);
                                self.table.meta(i).store(meta::h2(hash), Ordering::Release);
                            }
                        }
                    }
                }
            }
        }

        false
    }

    // Removes a key from the map, returning the entry for the key if the key was previously in the map.
    pub fn remove<'g, Q: ?Sized>(&self, key: &Q, guard: &'g Guard<'_>) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        if self.table.raw.is_null() {
            return None;
        }

        let hash = self.hash(key);
        let h2 = meta::h2(hash);

        let i = h1(hash) & (self.table.len - 1);
        let limit = i + probe_limit!(self.table.len);
        let mut copying = false;

        'probe: for i in i..=limit {
            let meta = unsafe { self.table.meta(i).load(Ordering::Acquire) };

            // the key is not in this table
            if meta == meta::EMPTY {
                return None;
            }

            // the entry was deleted
            if meta == meta::TOMBSTONE {
                continue;
            }

            if meta == h2 {
                let mut entry = unsafe { guard.protect(self.table.entry(i), Ordering::Acquire) };

                // the entry was deleted
                if entry.addr() & Entry::TOMBSTONE != 0 {
                    continue;
                }

                let entry_ptr = entry.mask(Entry::POINTER);

                // the key matches, we might be able to perform an update
                if unsafe { (*entry_ptr).key.borrow() == key } {
                    // the entry is being copied to a new table, we have to finish the resize and delete it in the new table
                    if entry.addr() & Entry::COPYING != 0 {
                        copying = true;
                        break;
                    }

                    loop {
                        // perform the deletion
                        match unsafe { self.table.entry(i) }.compare_exchange_weak(
                            entry,
                            Entry::TOMBSTONE as _,
                            Ordering::AcqRel,
                            Ordering::Acquire,
                        ) {
                            // succesfully deleted
                            Ok(_) => unsafe {
                                let entry_ptr = entry.mask(Entry::POINTER);
                                self.table.meta(i).store(meta::TOMBSTONE, Ordering::Release);

                                // retire the old value
                                guard.defer_retire(entry_ptr, Entry::retire::<K, V>);

                                self.table.state().count.get(guard.thread.id).delete(1);
                                return Some((&(*entry_ptr).key, &(*entry_ptr).value));
                            },

                            // the entry is being copied to the new table
                            Err(found) if found.addr() & Entry::COPYING != 0 => {
                                copying = true;
                                break 'probe;
                            }

                            // the entry was deleted
                            Err(found) if found.addr() & Entry::TOMBSTONE != 0 => {
                                return None;
                            }

                            // lost to a concurrent update, retry
                            Err(found) => entry = found,
                        }
                    }
                }
            }
        }

        // found a copied entry, we have to finish the resize and delete it in the new table
        if copying {
            assert!(self.next_table_ref().is_some());
            let next_table = self.help_copy(guard);
            return self.as_ref(next_table).remove(key, guard);
        }

        return None;
    }

    pub fn clear(&self, guard: &Guard<'_>) {
        if self.table.raw.is_null() {
            return;
        }

        let mut copying = false;

        // drop all the entries
        for i in 0..(self.table.len + probe_limit!(self.table.len)) {
            let mut entry = unsafe { self.table.entry(i).load(Ordering::Acquire) };
            loop {
                // a non-empty entry is being copied. clear every entry in this table that we can, then
                // deal with the copy
                if entry.addr() & Entry::COPYING != 0
                    && entry.addr() != Entry::COPYING
                    && entry.addr() & Entry::TOMBSTONE == 0
                {
                    copying = true;
                    continue;
                }

                // the entry is empty or already deleted
                if entry.is_null() || entry.addr() & Entry::TOMBSTONE != 0 {
                    continue;
                }

                let result = unsafe {
                    self.table.entry(i).compare_exchange(
                        entry,
                        Entry::TOMBSTONE as _,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                };

                match result {
                    Ok(entry) => unsafe {
                        self.table.meta(i).store(meta::TOMBSTONE, Ordering::Release);

                        // retire the old value
                        let entry_ptr = entry.mask(Entry::POINTER);
                        guard.defer_retire(entry_ptr, Entry::retire::<K, V>);
                        break;
                    },
                    Err(found) => {
                        entry = found;
                        continue;
                    }
                }
            }
        }

        // we can't delete the entries being copied until the copy is complete, so help the copy
        // and continue the clear
        if copying {
            assert!(self.next_table_ref().is_some());
            let next_table = self.help_copy(guard);
            return self.as_ref(next_table).clear(guard);
        }
    }

    // Returns the hash of a key.
    #[inline(always)]
    fn hash<Q>(&self, key: &Q) -> u64
    where
        Q: Hash + ?Sized,
    {
        let mut h = self.hash_builder.build_hasher();
        key.hash(&mut h);
        h.finish()
    }
}

// An iterator over the keys and values of this table.
pub struct Iter<'g, K, V> {
    i: usize,
    capacity: usize,
    table: Table<K, V>,
    _guard: &'g Guard<'g>,
}

impl<'g, K, V> Clone for Iter<'g, K, V> {
    fn clone(&self) -> Self {
        Iter {
            table: self.table,
            capacity: self.capacity,
            _guard: self._guard,
            i: self.i,
        }
    }
}

impl<'g, K: 'g, V: 'g> Iterator for Iter<'g, K, V> {
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.table.raw.is_null() {
            return None;
        }

        loop {
            if self.i >= self.capacity {
                return None;
            }

            let meta = unsafe { self.table.meta(self.i) }.load(Ordering::Acquire);

            if matches!(meta, meta::EMPTY | meta::TOMBSTONE) {
                self.i += 1;
                continue;
            }

            let entry = unsafe { self.table.entry(self.i) }.load(Ordering::Acquire);

            if entry.addr() & Entry::TOMBSTONE != 0 {
                self.i += 1;
                continue;
            }

            let entry = unsafe { &*entry.mask(Entry::POINTER) };
            self.i += 1;
            return Some((&entry.key, &entry.value));
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
        let buffer = probe_limit!(next_capacity);

        if next_capacity > isize::MAX as usize {
            panic!("Hash table exceeded maximum capacity");
        }

        // allocate the new table
        let next = Table::new(next_capacity, next_capacity + buffer, self.collector.link());
        state.next.store(next.raw, Ordering::Release);
        drop(_allocating);

        next
    }

    // Update the copy state and attempt to promote a copy to the root table.
    //
    // Returns true if the table was promoted.
    fn try_promote(
        &self,
        state: &State,
        next: Table<K, V>,
        copied: usize,
        guard: &Guard<'_>,
    ) -> bool {
        let capacity = self.table.len + probe_limit!(self.table.len);

        // update the count
        let copied = if copied > 0 {
            state.copied.fetch_add(copied, Ordering::AcqRel) + copied
        } else {
            state.copied.load(Ordering::Acquire)
        };

        if copied == capacity {
            let root = self.root.load(Ordering::Acquire);
            if self.table.raw == root {
                let copied = self.len();

                // update the length of the new table
                let entries = &next.state().count.get(guard.thread.id).entries;
                entries
                    .compare_exchange(0, copied, Ordering::Relaxed, Ordering::Relaxed)
                    .ok();

                match self.root.compare_exchange(
                    self.table.raw,
                    next.raw,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => unsafe {
                        // retire the old table. not we don't drop any entries because everything was copied
                        guard.defer_retire(self.table.raw, |link| {
                            let raw: *mut RawTable = link.cast();
                            let table: Table<K, V> = Table::from_raw(raw);
                            Table::dealloc(table);
                        })
                    },
                    _ => {}
                }

                // wake up anyone waiting for the promotion
                state.status.store(State::COMPLETE, Ordering::Release);
                atomic_wait::wake_all(&state.status);
                return true;
            }
        }

        false
    }

    // Returns the number of entries in the table.
    pub fn len<'g>(&self) -> usize {
        self.table.state().count.sum(Counter::active)
    }

    // Creates a reference to the given table while maintaining the root table pointer.
    fn as_ref(&self, table: Table<K, V>) -> HashMapRef<'root, K, V, S> {
        HashMapRef {
            table,
            root: self.root,
            collector: self.collector,
            hash_builder: self.hash_builder,
        }
    }
}

impl<K, V, S> Drop for HashMap<K, V, S> {
    fn drop(&mut self) {
        let table = *self.table.get_mut();
        if table.is_null() {
            return;
        }

        let table = unsafe { Table::<K, V>::from_raw(table) };
        assert!(table.state().next.load(Ordering::Acquire).is_null());

        // drop all the entries
        for i in 0..(table.len + probe_limit!(table.len)) {
            let entry = unsafe { *table.entry(i).as_ptr() };
            let entry_ptr = entry.mask(Entry::POINTER);

            // nothing to copy
            if entry_ptr.is_null() || entry.addr() & Entry::TOMBSTONE != 0 {
                continue;
            }

            unsafe { self.collector.retire(entry_ptr, Entry::retire::<K, V>) }
        }

        // deallocate the table
        unsafe { Table::dealloc(table) };
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
