use crate::raw;
use crate::seize::{Collector, Guard};

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
    /// use std::collections::HashMap;
    /// use std::hash::RandomState;
    ///
    /// let s = RandomState::new();
    /// let map = HashMap::with_capacity_and_hasher(10, s);
    /// map.insert(1, 2);
    /// ```
    pub fn with_capacity_and_hasher(capacity: usize, hash_builder: S) -> HashMap<K, V, S> {
        HashMap {
            raw: raw::HashMap::with_capacity_and_hasher(capacity, hash_builder),
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

    /// Returns a `Guard` for use with this map.
    ///
    /// Note that holding on to a `Guard` pins the current thread, preventing garbage
    /// collection. See the [crate-level documentation](crate) for details.
    pub fn guard(&self) -> Guard<'_> {
        self.raw.collector.enter()
    }

    /// Returns a pinned reference to the map.
    ///
    /// The returned reference manages a `Guard` internally, preventing garbage collection
    /// for as long as it is held. See the [crate-level documentation](crate) for details.
    pub fn pin(&self) -> HashMapRef<'_, K, V, S> {
        HashMapRef {
            guard: self.raw.guard(),
            table: self,
        }
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
    pub fn iter<'g>(&self, guard: &'g Guard<'_>) -> Iter<'g, K, V> {
        Iter {
            raw: self.raw.root(&guard).iter(&guard),
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
    pub fn keys<'g>(&self, guard: &'g Guard<'_>) -> Keys<'g, K, V> {
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
    pub fn values<'g>(&self, guard: &'g Guard<'_>) -> Values<'g, K, V> {
        Values {
            iter: self.iter(guard),
        }
    }
}

impl<K, V, S> HashMap<K, V, S>
where
    K: Hash + Eq,
    S: BuildHasher,
{
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
    pub fn contains_key<Q>(&self, key: &Q, guard: &Guard<'_>) -> bool
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
    pub fn get<'g, Q>(&'g self, key: &Q, guard: &'g Guard<'_>) -> Option<&'g V>
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
    pub fn get_key_value<'g, Q>(&self, key: &Q, guard: &'g Guard<'_>) -> Option<(&'g K, &'g V)>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw.root(guard).get_entry(key, guard)
    }
}

pub struct HashMapRef<'table, K, V, S> {
    guard: Guard<'table>,
    table: &'table HashMap<K, V, S>,
}

impl<'table, K, V, S> HashMapRef<'table, K, V, S>
where
    K: Clone + Hash + Eq + Sync + Send,
    V: Sync + Send,
    S: BuildHasher,
{
    pub fn get<'g, Q>(&'g self, key: &Q) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.table.get(key, &self.guard)
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        self.table.iter(&self.guard)
    }

    pub fn insert(&self, key: K, value: V) -> Option<&V> {
        self.table
            .raw
            .root(&self.guard)
            .insert(key, value, &self.guard)
    }

    pub fn update<F>(&self, key: K, f: F) -> Option<&V>
    where
        F: Fn(&V) -> V,
    {
        self.table.raw.root(&self.guard).update(key, f, &self.guard)
    }

    pub fn remove<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.table.raw.root(&self.guard).remove(key, &self.guard)
    }
}

/// An iterator over a map's entries.
///
/// See [`HashMap::iter`](crate::HashMap::iter) for details.
pub struct Iter<'g, K, V> {
    raw: raw::Iter<'g, K, V>,
}

impl<'g, K: 'g, V: 'g> Iterator for Iter<'g, K, V> {
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next()
    }
}

impl<K, V> fmt::Debug for Iter<'_, K, V>
where
    K: fmt::Debug,
    V: fmt::Debug,
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
#[derive(Debug)]
pub struct Keys<'g, K, V> {
    iter: Iter<'g, K, V>,
}

impl<'g, K: 'g, V: 'g> Iterator for Keys<'g, K, V> {
    type Item = &'g K;

    fn next(&mut self) -> Option<Self::Item> {
        let (key, _) = self.iter.next()?;
        Some(key)
    }
}

/// An iterator over a map's values.
///
/// See [`HashMap::values`](crate::HashMap::values) for details.
#[derive(Debug)]
pub struct Values<'g, K, V> {
    iter: Iter<'g, K, V>,
}

impl<'g, K: 'g, V: 'g> Iterator for Values<'g, K, V> {
    type Item = &'g V;

    fn next(&mut self) -> Option<Self::Item> {
        let (_, value) = self.iter.next()?;
        Some(value)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct OccupiedError<'a, V> {
    pub current: &'a V,
    pub not_inserted: V,
}
