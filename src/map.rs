use crate::raw::{self, EntryStatus};
use seize::{Collector, Guard, LocalGuard, OwnedGuard};

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash};

/// A concurrent hash table.
///
/// Most hash table operations require a [`Guard`](crate::Guard), which can be acquired through
/// [`HashMap::guard`] or using the [`HashMap::pin`] API. See the [crate-level documentation](crate)
/// for details.
pub struct HashMap<K, V, S = RandomState> {
    pub raw: raw::HashMap<K, V, S>,
    pub initial: *mut u8,
}

unsafe impl<K, V, S: Send> Send for HashMap<K, V, S> {}
unsafe impl<K, V, S: Sync> Sync for HashMap<K, V, S> {}

/// Resize behavior for a [`HashMap`].
///
/// Hash maps must resize when the underlying table becomes full, migrating all key and value pairs
/// to a new allocation. This type allows you to configure the resizing behavior when passed to
/// [`HashMap::resize_mode`].
pub enum ResizeMode {
    /// Writers copy a constant number of key/value pairs to the new table before making
    /// progress.
    ///
    /// Incremental resizes avoids latency spikes that can occur when insert operations have
    /// to resize a large table. However, they reduce parallelism during the resize and so can reduce
    /// overall throughput. Incremental resizing also means all reads or write operations during an
    /// in-progress resize may have to search both the current and new table before succeeding, trading
    /// off median latency during a resize for tail latency.
    ///
    /// This is the default resize mode, with a chunk size of `1024`.
    Incremental(usize),
    /// All writes to the map must wait till the resize completes before making progress.
    ///
    /// Blocking resizes tend to be better in terms of throughput, especially in setups with
    /// multiple writers that can perform the resize in parallel. However, they can lead to latency
    /// spikes for insert operations that have to resize large tables.
    ///
    /// If insert latency is not a concern, such as if the keys in your map are stable, enabling blocking
    /// resizes may yield better performance.
    Blocking,
}

impl Default for ResizeMode {
    fn default() -> Self {
        // incremental resizing is a good default for most workloads as it avoids
        // unexpected latency spikes.
        ResizeMode::Incremental(1024)
    }
}

impl<K, V> HashMap<K, V> {
    /// Creates an empty `HashMap`.
    ///
    /// The hash map is initally crated with a capacity of 0, so it will not allocate
    /// until it is first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    /// let map: HashMap<&str, i32> = HashMap::new();
    /// ```
    pub fn new() -> HashMap<K, V> {
        HashMap::with_capacity_and_hasher(0, RandomState::new())
    }

    /// Creates an empty `HashMap` with the specified capacity.
    ///
    /// Note the table should be able to hold at least `capacity` elements before
    /// resizing, but may prematurely resize due to poor hash distribution. If `capacity`
    /// is 0, the hash map will not allocate.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    /// let map: HashMap<&str, i32> = HashMap::with_capacity(10);
    /// ```
    pub fn with_capacity(capacity: usize) -> HashMap<K, V> {
        HashMap::with_capacity_and_hasher(capacity, RandomState::new())
    }
}

impl<K, V, S> Default for HashMap<K, V, S>
where
    S: Default,
{
    fn default() -> Self {
        HashMap::with_hasher(S::default())
    }
}

impl<K, V, S> HashMap<K, V, S> {
    /// Creates an empty `HashMap` which will use the given hash builder to hash
    /// keys.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and is designed
    /// to allow HashMaps to be resistant to attacks that cause many collisions
    /// and very poor performance. Setting it manually using this function can
    /// expose a DoS attack vector.
    ///
    /// The `hash_builder` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    /// use std::hash::RandomState;
    ///
    /// let s = RandomState::new();
    /// let map = HashMap::with_hasher(s);
    /// map.pin().insert(1, 2);
    /// ```
    pub fn with_hasher(hash_builder: S) -> HashMap<K, V, S> {
        HashMap::with_capacity_and_hasher(0, hash_builder)
    }

    /// Creates an empty `HashMap` with at least the specified capacity, using
    /// `hash_builder` to hash the keys.
    ///
    /// Note the table should be able to hold at least `capacity` elements before
    /// resizing, but may prematurely resize due to poor hash distribution. If `capacity`
    /// is 0, the hash map will not allocate.
    ///
    /// Warning: `hash_builder` is normally randomly generated, and is designed
    /// to allow HashMaps to be resistant to attacks that cause many collisions
    /// and very poor performance. Setting it manually using this function can
    /// expose a DoS attack vector.
    ///
    /// The `hasher` passed should implement the [`BuildHasher`] trait for
    /// the HashMap to be useful, see its documentation for details.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    /// use std::hash::RandomState;
    ///
    /// let s = RandomState::new();
    /// let map = HashMap::with_capacity_and_hasher(10, s);
    /// map.pin().insert(1, 2);
    /// ```
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> HashMap<K, V, S> {
        let raw = raw::HashMap::with_capacity_and_hasher(capacity, hash_builder);
        HashMap {
            initial: raw.root_ptr(),
            raw,
        }
    }

    /// Associate a custom [`seize::Collector`] with this map.
    ///
    /// This method may be useful when you want more control over memory reclamation.
    /// See [`seize::Collector`] for details.
    ///
    /// Note that all `Guard` references used to access the map must be produced by
    /// `collector`.
    pub fn with_collector(mut self, collector: Collector) -> Self {
        self.raw.collector = collector;
        self
    }

    /// Configures the resizing mode for this map.
    ///
    /// See [`ResizeMode`] for details.
    pub fn resize_mode(mut self, resize: ResizeMode) -> Self {
        assert_eq!(
            self.raw.root_ptr(),
            self.initial,
            "cannot change resize mode after initialization"
        );

        self.raw.resize = resize;
        self
    }

    /// Returns a guard for use with this map.
    ///
    /// Note that holding on to a guard pins the current thread, preventing garbage
    /// collection. See the [crate-level documentation](crate) for details.
    #[inline]
    pub fn guard(&self) -> LocalGuard<'_> {
        self.raw.collector.enter()
    }

    /// Returns an owned guard for use with this map.
    ///
    /// Owned guards implement `Send` and `Sync`, allowing them to be held across
    /// `.await` points in multi-threaded schedulers. This is especially useful
    /// for iterators.
    ///
    /// Note that holding on to a guard pins the current thread, preventing garbage
    /// collection. See the [crate-level documentation](crate) for details.
    #[inline]
    pub fn owned_guard(&self) -> OwnedGuard<'_> {
        self.raw.collector.enter_owned()
    }

    /// Returns a pinned reference to the map.
    ///
    /// The returned reference manages a guard internally, preventing garbage collection
    /// for as long as it is held. See the [crate-level documentation](crate) for details.
    #[inline]
    pub fn pin(&self) -> HashMapRef<'_, K, V, S, LocalGuard<'_>> {
        HashMapRef {
            guard: self.guard(),
            map: self,
        }
    }

    /// Returns a pinned reference to the map.
    ///
    /// Unlike [`pin`](HashMap::pin), the retured reference implements `Send`
    /// and `Sync`, allowing it to be held across `.await` points in multi-threaded
    /// schedulers. This is especially useful for iterators.
    ///
    /// The returned reference manages a guard internally, preventing garbage collection
    /// for as long as it is held. See the [crate-level documentation](crate) for details.
    #[inline]
    pub fn pin_owned(&self) -> HashMapRef<'_, K, V, S, OwnedGuard<'_>> {
        HashMapRef {
            guard: self.owned_guard(),
            map: self,
        }
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    // note not all the methods below actually require thread-safety bounds, but
    // the map is not generally useful without them
    K: Send + Sync + Hash + Eq,
    V: Send + Sync,
    S: BuildHasher,
{
    /// Returns the number of entries in the map.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::new();
    ///
    /// map.pin().insert(1, "a");
    /// map.pin().insert(2, "b");
    /// assert!(map.len() == 2);
    /// ```
    pub fn len(&self) -> usize {
        self.raw.len()
    }

    /// Returns `true` if the map is empty. Otherwise returns `false`.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::new();
    /// assert!(map.is_empty());
    /// map.pin().insert("a", 1);
    /// assert!(!map.is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: std::cmp::Eq
    /// [`Hash`]: std::hash::Hash
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::new();
    /// let m = map.pin();
    /// m.insert(1, "a");
    /// assert_eq!(m.contains_key(&1), true);
    /// assert_eq!(m.contains_key(&2), false);
    /// ```
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q, guard: &impl Guard) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.get(key, guard).is_some()
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: std::cmp::Eq
    /// [`Hash`]: std::hash::Hash
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::new();
    /// let m = map.pin();
    /// m.insert(1, "a");
    /// assert_eq!(m.get(&1), Some(&"a"));
    /// assert_eq!(m.get(&2), None);
    /// ```
    #[inline]
    pub fn get<'g, Q>(&'g self, key: &Q, guard: &'g impl Guard) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw.root(guard).get_entry(key, guard).map(|(_, v)| v)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// The supplied key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// [`Eq`]: std::cmp::Eq
    /// [`Hash`]: std::hash::Hash
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::new();
    /// let m = map.pin();
    /// m.insert(1, "a");
    /// assert_eq!(m.get_key_value(&1), Some((&1, &"a")));
    /// assert_eq!(m.get_key_value(&2), None);
    /// ```
    #[inline]
    pub fn get_key_value<'g, Q>(&self, key: &Q, guard: &'g impl Guard) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw.root(guard).get_entry(key, guard)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, [`None`] is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old
    /// value is returned. The key is not updated, though; this matters for
    /// types that can be `==` without being identical. See the [standard library
    /// documentation] for more.
    ///
    /// [standard library documentation]: https://doc.rust-lang.org/std/collections/index.html#insert-and-complex-keys
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// assert_eq!(map.pin().insert(37, "a"), None);
    /// assert_eq!(map.pin().is_empty(), false);
    ///
    /// // note: you can also re-use a map pin like so:
    /// let m = map.pin();
    ///
    /// m.insert(37, "b");
    /// assert_eq!(m.insert(37, "c"), Some(&"b"));
    /// assert_eq!(m.get(&37), Some(&"c"));
    /// ```
    #[inline]
    pub fn insert<'g>(&'g self, key: K, value: V, guard: &'g impl Guard) -> Option<&'g V> {
        match self.raw.root(guard).insert(key, value, true, guard) {
            EntryStatus::Empty(_) | EntryStatus::Tombstone(_) => None,
            EntryStatus::Replaced(value) => Some(value),
            EntryStatus::Error { .. } => unreachable!(),
        }
    }

    /// Tries to insert a key-value pair into the map, and returns
    /// a reference to the value that was inserted.
    ///
    /// If the map already had this key present, nothing is updated, and
    /// an error containing the existing value is returned.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// let m = map.pin();
    /// assert_eq!(m.try_insert(37, "a").unwrap(), &"a");
    ///
    /// let err = m.try_insert(37, "b").unwrap_err();
    /// assert_eq!(err.current, &"a");
    /// assert_eq!(err.not_inserted, "b");
    /// ```
    #[inline]
    pub fn try_insert<'g>(
        &self,
        key: K,
        value: V,
        guard: &'g impl Guard,
    ) -> Result<&'g V, OccupiedError<'g, V>> {
        match self.raw.root(guard).insert(key, value, false, guard) {
            EntryStatus::Empty(value) | EntryStatus::Tombstone(value) => Ok(value),
            EntryStatus::Error {
                current,
                not_inserted,
            } => Err(OccupiedError {
                current,
                not_inserted,
            }),
            EntryStatus::Replaced(_) => unreachable!(),
        }
    }

    // Update an entry with a remapping function.
    //
    // If the value for the specified `key` is present, the new value is computed and stored the
    // using the provided update function, and the new value is returned. Otherwise, `None`
    // is returned.
    //
    // The update function should be pure, as it may be called multiple times if the current value
    // changes during the execution of this function. However, the update is performed atomically,
    // meaning the value is only updated from it's previous value using the call to `update` with that
    // value.
    //
    // # Examples
    //
    // ```
    // use papaya::HashMap;
    //
    // let mut map = HashMap::new();
    /// map.pin().insert("a", 1);
    /// assert_eq!(m.get(&"a"), Some(&1));
    ///
    /// map.pin().update("a", |v| v + 1);
    /// assert_eq!(m.get(&"a"), Some(&2));
    // ```
    pub fn update<'g, F>(&self, key: K, update: F, guard: &'g impl Guard) -> Option<&'g V>
    where
        F: Fn(&V) -> V,
    {
        self.raw.root(guard).update(key, update, guard)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.pin().insert(1, "a");
    /// assert_eq!(map.pin().remove(&1), Some(&"a"));
    /// assert_eq!(map.pin().remove(&1), None);
    /// ```
    #[inline]
    pub fn remove<'g, Q>(&self, key: &Q, guard: &'g impl Guard) -> Option<&'g V>
    where
        K: Borrow<Q> + 'g,
        Q: Hash + Eq + ?Sized,
    {
        match self.raw.root(guard).remove(key, guard) {
            Some((_, value)) => Some(value),
            None => None,
        }
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// The key may be any borrowed form of the map's key type, but
    /// [`Hash`] and [`Eq`] on the borrowed form *must* match those for
    /// the key type.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let mut map = HashMap::new();
    /// map.pin().insert(1, "a");
    /// assert_eq!(map.pin().get(&1), Some(&"a"));
    /// assert_eq!(map.pin().remove_entry(&1), Some((&1, &"a")));
    /// assert_eq!(map.pin().remove(&1), None);
    /// ```
    #[inline]
    pub fn remove_entry<'g, Q>(&'g self, key: &Q, guard: &'g impl Guard) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw.root(guard).remove(key, guard)
    }

    /// Tries to reserve capacity for `additional` more elements to be inserted
    /// in the `HashMap`.
    ///
    /// The collection may reserve more space to avoid frequent reallocations.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map: HashMap<&str, i32> = HashMap::new();
    ///
    /// map.pin().reserve(10);
    /// ```
    pub fn reserve(&self, additional: usize, guard: &impl Guard) {
        self.raw.root(guard).reserve(additional, guard);
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::new();
    ///
    /// map.pin().insert(1, "a");
    /// map.pin().clear();
    /// assert!(map.pin().is_empty());
    /// ```
    #[inline]
    pub fn clear(&self, guard: &impl Guard) {
        self.raw.root(guard).clear(guard)
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// for (key, val) in map.pin().iter() {
    ///     println!("key: {key} val: {val}");
    /// }
    #[inline]
    pub fn iter<'g, G>(&self, guard: &'g G) -> Iter<'g, K, V, G>
    where
        G: Guard,
    {
        Iter {
            raw: self.raw.root(guard).iter(guard),
        }
    }

    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// for key in map.pin().keys() {
    ///     println!("{key}");
    /// }
    /// ```
    #[inline]
    pub fn keys<'g, G>(&self, guard: &'g G) -> Keys<'g, K, V, G>
    where
        G: Guard,
    {
        Keys {
            iter: self.iter(guard),
        }
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    ///
    /// let map = HashMap::from([
    ///     ("a", 1),
    ///     ("b", 2),
    ///     ("c", 3),
    /// ]);
    ///
    /// for value in map.pin().values() {
    ///     println!("{value}");
    /// }
    /// ```
    #[inline]
    pub fn values<'g, G>(&self, guard: &'g G) -> Values<'g, K, V, G>
    where
        G: Guard,
    {
        Values {
            iter: self.iter(guard),
        }
    }
}

impl<K, V, S> PartialEq for HashMap<K, V, S>
where
    K: Hash + Eq + Send + Sync,
    V: PartialEq + Send + Sync,
    S: BuildHasher,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }

        let (guard1, guard2) = (&self.guard(), &other.guard());

        let mut iter = self.iter(guard1);
        iter.all(|(key, value)| other.get(key, guard2).map_or(false, |v| *value == *v))
    }
}

impl<K, V, S> Eq for HashMap<K, V, S>
where
    K: Hash + Eq + Send + Sync,
    V: Eq + Send + Sync,
    S: BuildHasher,
{
}

impl<K, V, S> fmt::Debug for HashMap<K, V, S>
where
    K: Hash + Eq + Send + Sync + fmt::Debug,
    V: Send + Sync + fmt::Debug,
    S: BuildHasher,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let guard = self.guard();
        f.debug_map().entries(self.iter(&guard)).finish()
    }
}

impl<K, V, S> Extend<(K, V)> for &HashMap<K, V, S>
where
    K: Sync + Send + Clone + Hash + Ord,
    V: Sync + Send,
    S: BuildHasher,
{
    fn extend<T: IntoIterator<Item = (K, V)>>(&mut self, iter: T) {
        // from `hashbrown::HashMap::extend`:
        // Keys may be already present or show multiple times in the iterator.
        // Reserve the entire hint lower bound if the map is empty.
        // Otherwise reserve half the hint (rounded up), so the map
        // will only resize twice in the worst case.
        let iter = iter.into_iter();
        let reserve = if self.is_empty() {
            iter.size_hint().0
        } else {
            (iter.size_hint().0 + 1) / 2
        };

        let guard = self.guard();
        self.reserve(reserve, &guard);

        for (key, value) in iter {
            self.insert(key, value, &guard);
        }
    }
}

impl<'a, K, V, S> Extend<(&'a K, &'a V)> for &HashMap<K, V, S>
where
    K: Sync + Send + Copy + Hash + Ord,
    V: Sync + Send + Copy,
    S: BuildHasher,
{
    fn extend<T: IntoIterator<Item = (&'a K, &'a V)>>(&mut self, iter: T) {
        self.extend(iter.into_iter().map(|(&key, &value)| (key, value)));
    }
}

impl<K, V, const N: usize> From<[(K, V); N]> for HashMap<K, V, RandomState>
where
    K: Sync + Send + Clone + Hash + Ord,
    V: Sync + Send,
{
    fn from(arr: [(K, V); N]) -> Self {
        HashMap::from_iter(arr)
    }
}

impl<K, V, S> FromIterator<(K, V)> for HashMap<K, V, S>
where
    K: Sync + Send + Clone + Hash + Ord,
    V: Sync + Send,
    S: BuildHasher + Default,
{
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let mut iter = iter.into_iter();

        if let Some((key, value)) = iter.next() {
            // safety: we own `map`
            let guard = unsafe { seize::unprotected() };

            let (lower, _) = iter.size_hint();
            let map = HashMap::with_capacity_and_hasher(lower.saturating_add(1), S::default());

            map.insert(key, value, &guard);

            for (key, value) in iter {
                map.insert(key, value, &guard);
            }

            map
        } else {
            Self::default()
        }
    }
}

impl<K, V, S> Clone for HashMap<K, V, S>
where
    K: Sync + Send + Clone + Hash + Ord,
    V: Sync + Send + Clone,
    S: BuildHasher + Clone,
{
    fn clone(&self) -> HashMap<K, V, S> {
        let other = Self::with_capacity_and_hasher(self.len(), self.raw.hasher.clone())
            .with_collector(self.raw.collector.clone());

        {
            let (guard1, guard2) = (&self.guard(), &other.guard());
            for (key, value) in self.iter(guard1) {
                other.insert(key.clone(), value.clone(), guard2);
            }
        }

        other
    }
}

/// The error returned by [`try_insert`](HashMap::try_insert) when the key already exists.
///
/// Contains the existing value, and the value that was not inserted.
#[derive(Debug, PartialEq, Eq)]
pub struct OccupiedError<'a, V: 'a> {
    /// The value in the map that was already present.
    pub current: &'a V,
    /// The value which was not inserted, because the entry was already occupied.
    pub not_inserted: V,
}

/// A pinned reference to a hash table.
///
/// See [`HashMap::pin`] for details.
pub struct HashMapRef<'map, K, V, S, G> {
    guard: G,
    map: &'map HashMap<K, V, S>,
}

impl<'map, K, V, S, G> HashMapRef<'map, K, V, S, G>
where
    K: Clone + Hash + Eq + Send + Sync,
    V: Send + Sync,
    S: BuildHasher,
    G: Guard,
{
    /// Returns a reference to the inner [`HashMap`].
    #[inline]
    pub fn map(&self) -> &'map HashMap<K, V, S> {
        self.map
    }

    /// Returns the number of entries in the map.
    ///
    /// See [`HashMap::len`] for details.
    #[inline]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the map is empty. Otherwise returns `false`.
    ///
    /// See [`HashMap::is_empty`] for details.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns `true` if the map contains a value for the specified key.
    ///
    /// See [`HashMap::contains_key`] for details.
    #[inline]
    pub fn contains_key<Q>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.contains_key(key, &self.guard)
    }

    /// Returns a reference to the value corresponding to the key.
    ///
    /// See [`HashMap::get`] for details.
    #[inline]
    pub fn get<'g, Q>(&'g self, key: &Q) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key, &self.guard)
    }

    /// Returns the key-value pair corresponding to the supplied key.
    ///
    /// See [`HashMap::get_key_value`] for details.
    #[inline]
    pub fn get_key_value<'g, Q>(&'g self, key: &Q) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get_key_value(key, &self.guard)
    }

    /// Inserts a key-value pair into the map.
    ///
    /// See [`HashMap::insert`] for details.
    #[inline]
    pub fn insert(&self, key: K, value: V) -> Option<&V> {
        self.map.insert(key, value, &self.guard)
    }

    /// Tries to insert a key-value pair into the map, and returns
    /// a reference to the value that was inserted.
    ///
    /// See [`HashMap::try_insert`] for details.
    #[inline]
    pub fn try_insert(&self, key: K, value: V) -> Result<&V, OccupiedError<'_, V>> {
        self.map.try_insert(key, value, &self.guard)
    }

    // Update an entry with a remapping function.
    //
    /// See [`HashMap::update`] for details.
    #[inline]
    pub fn update<F>(&self, key: K, update: F) -> Option<&V>
    where
        F: Fn(&V) -> V,
    {
        self.map.update(key, update, &self.guard)
    }

    /// Removes a key from the map, returning the value at the key if the key
    /// was previously in the map.
    ///
    /// See [`HashMap::remove`] for details.
    #[inline]
    pub fn remove<'g, Q>(&'g self, key: &Q) -> Option<&'g V>
    where
        K: Borrow<Q> + 'g,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove(key, &self.guard)
    }

    /// Removes a key from the map, returning the stored key and value if the
    /// key was previously in the map.
    ///
    /// See [`HashMap::remove_entry`] for details.
    #[inline]
    pub fn remove_entry<'g, Q>(&'g self, key: &Q) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.remove_entry(key, &self.guard)
    }

    /// Clears the map, removing all key-value pairs.
    ///
    /// See [`HashMap::clear`] for details.
    #[inline]
    pub fn clear(&self) {
        self.map.clear(&self.guard)
    }

    /// Tries to reserve capacity for `additional` more elements to be inserted
    /// in the map.
    ///
    /// See [`HashMap::reserve`] for details.
    #[inline]
    pub fn reserve(&self, additional: usize) {
        self.map.reserve(additional, &self.guard)
    }

    /// An iterator visiting all key-value pairs in arbitrary order.
    /// The iterator element type is `(&'a K, &'a V)`.
    ///
    /// See [`HashMap::iter`] for details.
    #[inline]
    pub fn iter(&self) -> Iter<'_, K, V, G> {
        self.map.iter(&self.guard)
    }

    /// An iterator visiting all keys in arbitrary order.
    /// The iterator element type is `&'a K`.
    ///
    /// See [`HashMap::keys`] for details.
    #[inline]
    pub fn keys(&self) -> Keys<'_, K, V, G> {
        self.map.keys(&self.guard)
    }

    /// An iterator visiting all values in arbitrary order.
    /// The iterator element type is `&'a V`.
    ///
    /// See [`HashMap::values`] for details.
    #[inline]
    pub fn values(&self) -> Values<'_, K, V, G> {
        self.map.values(&self.guard)
    }
}

/// An iterator over a map's entries.
///
/// See [`HashMap::iter`](crate::HashMap::iter) for details.
pub struct Iter<'g, K, V, G> {
    raw: raw::Iter<'g, K, V, G>,
}

impl<'g, K: 'g, V: 'g, G> Iterator for Iter<'g, K, V, G>
where
    G: Guard,
{
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next()
    }
}

impl<K, V, G> fmt::Debug for Iter<'_, K, V, G>
where
    K: fmt::Debug,
    V: fmt::Debug,
    G: Guard,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(Iter {
                raw: self.raw.clone(),
            })
            .finish()
    }
}

/// An iterator over a map's keys.
///
/// See [`HashMap::keys`](crate::HashMap::keys) for details.
pub struct Keys<'g, K, V, G> {
    iter: Iter<'g, K, V, G>,
}

impl<'g, K: 'g, V: 'g, G> Iterator for Keys<'g, K, V, G>
where
    G: Guard,
{
    type Item = &'g K;

    fn next(&mut self) -> Option<Self::Item> {
        let (key, _) = self.iter.next()?;
        Some(key)
    }
}

impl<K, V, G> fmt::Debug for Keys<'_, K, V, G>
where
    K: fmt::Debug,
    V: fmt::Debug,
    G: Guard,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Keys").field(&self.iter).finish()
    }
}

/// An iterator over a map's values.
///
/// See [`HashMap::values`](crate::HashMap::values) for details.
pub struct Values<'g, K, V, G> {
    iter: Iter<'g, K, V, G>,
}

impl<'g, K: 'g, V: 'g, G> Iterator for Values<'g, K, V, G>
where
    G: Guard,
{
    type Item = &'g V;

    fn next(&mut self) -> Option<Self::Item> {
        let (_, value) = self.iter.next()?;
        Some(value)
    }
}

impl<K, V, G> fmt::Debug for Values<'_, K, V, G>
where
    K: fmt::Debug,
    V: fmt::Debug,
    G: Guard,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Values").field(&self.iter).finish()
    }
}
