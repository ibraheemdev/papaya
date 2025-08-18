use std::cell::UnsafeCell;
use std::hash::{BuildHasher, Hash};
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::{panic, ptr};

use super::utils::{MapGuard, VerifiedGuard};
use crate::map::{Compute, Operation, ResizeMode};
use crate::raw::table::{self, Dealloc, HashTable};
use crate::Equivalent;

use seize::{Collector, LocalGuard, OwnedGuard};

/// A lock-free hash-table.
pub struct HashMap<K, V, S> {
    table: HashTable<Entry<K, V>, BoxDealloc<K, V>>,

    /// Hasher for keys.
    pub hasher: S,
}

struct BoxDealloc<K, V>(PhantomData<(K, V)>);

impl<K, V> Dealloc<Entry<K, V>> for BoxDealloc<K, V> {
    unsafe fn dealloc(entry: *mut Entry<K, V>) {
        let _: Box<Entry<K, V>> = unsafe { Box::from_raw(entry) };
    }
}

// An entry in the hash-table.
#[repr(C, align(8))] // Alignment requirement of `table::Entry`.
pub struct Entry<K, V> {
    /// The key for this entry.
    pub key: K,

    /// The value for this entry.
    pub value: V,
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

impl<K, V, S> HashMap<K, V, S> {
    /// Creates new hash-table with the given options.
    #[inline]
    pub fn new(
        capacity: usize,
        hasher: S,
        collector: Collector,
        resize: ResizeMode,
    ) -> HashMap<K, V, S> {
        HashMap {
            hasher,
            table: HashTable::new(capacity, collector, resize),
        }
    }

    /// Returns a guard for this collector
    pub fn guard(&self) -> MapGuard<LocalGuard<'_>> {
        self.table.guard()
    }

    /// Returns an owned guard for this collector
    pub fn owned_guard(&self) -> MapGuard<OwnedGuard<'_>> {
        self.table.owned_guard()
    }

    /// Verify a guard is valid to use with this map.
    #[inline]
    pub fn verify<'g, G>(&self, guard: &'g G) -> &'g MapGuard<G>
    where
        G: seize::Guard,
    {
        self.table.verify(guard)
    }

    /// Returns a reference to the collector.
    #[inline]
    pub fn collector(&self) -> &Collector {
        &self.table.collector
    }

    /// Returns the number of entries in the table.
    #[inline]
    pub fn len(&self) -> usize {
        self.table.len()
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
        let hash = self.hasher.hash_one(key);
        let eq = |entry: *mut Entry<K, V>| unsafe { key.equivalent(&(*entry).key) };

        self.table
            .find(hash, eq, guard)
            .map(|entry| unsafe { (&(*entry).key, &(*entry).value) })
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
        let hash = self.hasher.hash_one(&key);
        let new_entry = Box::into_raw(Box::new(Entry { key, value }));
        let eq = |entry: *mut Entry<K, V>| unsafe { (*new_entry).key.equivalent(&(*entry).key) };

        match self
            .table
            .insert(hash, new_entry, eq, self.hasher(), replace, guard)
        {
            table::InsertResult::Inserted => InsertResult::Inserted(unsafe { &(*new_entry).value }),

            table::InsertResult::Replaced(entry) => {
                InsertResult::Replaced(unsafe { &(*entry).value })
            }
            table::InsertResult::Error(entry) => {
                let new_entry = unsafe { Box::from_raw(new_entry) };

                InsertResult::Error {
                    current: unsafe { &(*entry).value },
                    not_inserted: new_entry.value,
                }
            }
        }
    }

    /// Removes a key from the map, returning the entry for the key if the key was previously in the map.
    #[inline]
    pub fn remove<'g, Q>(&self, key: &Q, guard: &'g impl VerifiedGuard) -> Option<(&'g K, &'g V)>
    where
        Q: Equivalent<K> + Hash + ?Sized,
    {
        let hash = self.hasher.hash_one(key);
        let eq = |entry: *mut Entry<K, V>| unsafe { key.equivalent(&(*entry).key) };

        self.table
            .remove(hash, eq, self.hasher(), guard)
            .map(|entry| unsafe { (&(*entry).key, &(*entry).value) })
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
        let hash = self.hasher.hash_one(key);
        let eq = |entry: *mut Entry<K, V>| unsafe { key.equivalent(&(*entry).key) };
        let should_remove =
            |entry: *mut Entry<K, V>| unsafe { should_remove(&(*entry).key, &(*entry).value) };

        self.table
            .remove_if(hash, eq, should_remove, self.hasher(), guard)
            .map(|option| option.map(|entry| unsafe { (&(*entry).key, &(*entry).value) }))
            .map_err(|entry| unsafe { (&(*entry).key, &(*entry).value) })
    }

    /// Reserve capacity for `additional` more elements.
    #[inline]
    pub fn reserve(&self, additional: usize, guard: &impl VerifiedGuard) {
        self.table.reserve(additional, self.hasher(), guard);
    }

    /// Remove all entries from this table.
    #[inline]
    pub fn clear(&self, guard: &impl VerifiedGuard) {
        self.table.clear(self.hasher(), guard);
    }

    /// Retains only the elements specified by the predicate.
    #[inline]
    pub fn retain<F>(&self, mut f: F, guard: &impl VerifiedGuard)
    where
        F: FnMut(&K, &V) -> bool,
    {
        let f = |entry: *mut Entry<K, V>| unsafe { f(&(*entry).key, &(*entry).value) };
        self.table.retain(f, self.hasher(), guard);
    }

    /// Returns an iterator over the keys and values of this table.
    #[inline]
    pub fn iter<'g, G>(&self, guard: &'g G) -> Iter<'g, K, V, G>
    where
        G: VerifiedGuard,
    {
        Iter {
            inner: self.table.iter(self.hasher(), guard),
            _entries: PhantomData,
        }
    }

    /// Returns a mutable iterator over the keys and values of this table.
    #[inline]
    pub fn iter_mut(&mut self) -> IterMut<'_, K, V> {
        IterMut {
            inner: self.table.iter_mut(),
            _entries: PhantomData,
        }
    }

    #[inline]
    fn hasher(&self) -> impl Fn(*mut Entry<K, V>) -> u64 + '_ {
        |entry: *mut Entry<K, V>| unsafe { self.hasher.hash_one(&(*entry).key) }
    }
}

impl<K, V, S> IntoIterator for HashMap<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            inner: self.table.into_iter(),
            _entries: PhantomData,
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
            //
            // TODO: This is no longer cached.
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
            //
            // TODO: This is no longer cached.
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
        mut compute: F,
        guard: &'g impl VerifiedGuard,
    ) -> Compute<'g, K, V, T>
    where
        F: FnMut(Option<(&'g K, &'g V)>) -> Operation<V, T>,
    {
        let hash = self.hasher.hash_one(&key);

        // Lazy initialize the entry allocation.
        let lazy_entry = UnsafeCell::new(LazyEntry::Uninit(key));

        let eq = |entry: *mut Entry<K, V>| unsafe {
            (*lazy_entry.get()).key().equivalent(&(*entry).key)
        };

        let compute = |entry: Option<*mut Entry<K, V>>| unsafe {
            let entry = entry.map(|entry| (&(*entry).key, &(*entry).value));

            match compute(entry) {
                Operation::Insert(value) => {
                    let entry = (*lazy_entry.get()).init();

                    (*entry).value.write(value);

                    table::Operation::Insert(entry.cast())
                }
                Operation::Remove => table::Operation::Remove,
                Operation::Abort(value) => table::Operation::Abort(value),
            }
        };

        let result = match self.table.compute(hash, eq, compute, self.hasher(), guard) {
            table::Compute::Inserted(entry) => unsafe {
                Compute::Inserted(&(*entry).key, &(*entry).value)
            },
            table::Compute::Updated { old, new } => unsafe {
                Compute::Updated {
                    old: (&(*old).key, &(*old).value),
                    new: (&(*new).key, &(*new).value),
                }
            },
            table::Compute::Removed(entry) => unsafe {
                Compute::Removed(&(*entry).key, &(*entry).value)
            },
            table::Compute::Aborted(value) => Compute::Aborted(value),
        };

        // Deallocate the entry if it was not inserted.
        if matches!(result, Compute::Removed(..) | Compute::Aborted(_)) {
            if let LazyEntry::Init(entry) = unsafe { &*lazy_entry.get() } {
                // Safety: The entry was allocated but not inserted into the map.
                let _ = unsafe { Box::from_raw(*entry) };
            }
        }

        result
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

// An iterator over the keys and values of this table.
pub struct Iter<'g, K, V, G> {
    inner: table::Iter<'g, Entry<K, V>, G>,
    _entries: PhantomData<(&'g K, &'g V)>,
}

impl<'g, K: 'g, V: 'g, G> Iterator for Iter<'g, K, V, G>
where
    G: VerifiedGuard,
{
    type Item = (&'g K, &'g V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|entry| unsafe { (&(*entry).key, &(*entry).value) })
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
            inner: self.inner.clone(),
            _entries: PhantomData,
        }
    }
}

// A mutable iterator over the keys and values of this table.
pub struct IterMut<'map, K, V> {
    inner: table::IterMut<Entry<K, V>>,
    // Ensure invariance with respect to `V`.
    _entries: PhantomData<(&'map K, &'map mut V)>,
}

impl<'map, K, V> Iterator for IterMut<'map, K, V> {
    type Item = (&'map K, &'map mut V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|entry| unsafe { (&(*entry).key, &mut (*entry).value) })
    }
}

impl<'map, K, V> IterMut<'map, K, V> {
    // Returns an immutable iterator over the remaining entries.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.inner
            .iter()
            .map(|entry| unsafe { (&(*entry).key, &(*entry).value) })
    }
}

// Safety: A mutable iterator does not perform any concurrent access,
// so the normal `Send` and `Sync` rules apply.
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

// An owned iterator over the keys and values of this table.
pub struct IntoIter<K, V> {
    inner: table::IntoIter<Entry<K, V>, BoxDealloc<K, V>>,
    _entries: PhantomData<(K, V)>,
}

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(|entry| {
            let entry = unsafe { ptr::read(entry) };
            (entry.key, entry.value)
        })
    }
}

impl<K, V> IntoIter<K, V> {
    // Returns an immutable iterator over the remaining entries.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (&K, &V)> + '_ {
        self.inner
            .iter()
            .map(|entry| unsafe { (&(*entry).key, &(*entry).value) })
    }
}

// Safety: An owned iterator does not perform any concurrent access,
// so the normal `Send` and `Sync` rules apply.
unsafe impl<K, V> Send for IntoIter<K, V>
where
    K: Send,
    V: Send,
{
}

unsafe impl<K, V> Sync for IntoIter<K, V>
where
    K: Sync,
    V: Sync,
{
}
