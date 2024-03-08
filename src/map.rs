use crate::raw;
use crate::seize::Guard;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

/// A concurrent hash table.
///
/// Most hash table operations require a [`Guard`](crate::Guard), which can be acquired through
/// [`HashMap::guard`] or using the [`HashMap::pin`] API. See the [crate-level
/// documentation](crate) for more details.
pub struct HashMap<K, V, S = RandomState> {
    pub raw: raw::HashMap<K, V, S>,
}

impl<K, V> Default for HashMap<K, V> {
    fn default() -> Self {
        HashMap::new()
    }
}

impl<K, V> HashMap<K, V> {
    /// Creates an empty `HashMap`.
    ///
    /// The hash map is initally crated with a capacity of 0, so it will not allocate until it is
    /// first inserted into.
    ///
    /// # Examples
    ///
    /// ```
    /// use papaya::HashMap;
    /// let map: HashMap<&str, i32> = HashMap::new();
    /// ```
    pub fn new() -> HashMap<K, V> {
        HashMap {
            raw: raw::HashMap::new(RandomState::new()),
        }
    }

    pub fn with_capacity(capacity: usize) -> HashMap<K, V> {
        HashMap {
            raw: raw::HashMap::with_capacity_and_hasher(capacity, RandomState::new()),
        }
    }
}

impl<K, V, S> HashMap<K, V, S> {
    pub fn with_capacity_and_hasher(capacity: usize, build_hasher: S) -> HashMap<K, V, S> {
        HashMap {
            raw: raw::HashMap::with_capacity_and_hasher(capacity, build_hasher),
        }
    }

    pub fn pin(&self) -> Pinned<'_, K, V, S> {
        Pinned {
            guard: self.raw.guard(),
            raw: &self.raw,
        }
    }
}

pub struct Pinned<'table, K, V, S> {
    guard: Guard<'table>,
    raw: &'table raw::HashMap<K, V, S>,
}

impl<'table, K, V, S> Pinned<'table, K, V, S>
where
    K: Clone + Hash + Eq + Sync + Send,
    V: Sync + Send,
    S: BuildHasher,
{
    pub fn capacity(&self) -> usize {
        self.raw.capacity(&self.guard)
    }

    pub fn iter(&self) -> Iter<'_, K, V> {
        Iter {
            raw: self.raw.root(&self.guard).iter(&self.guard),
        }
    }

    pub fn get<'g, Q>(&'g self, key: &Q) -> Option<&'g V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw.root(&self.guard).get(key, &self.guard)
    }

    pub fn insert(&self, key: K, value: V) -> Option<&V> {
        self.raw.root(&self.guard).insert(key, value, &self.guard)
    }

    pub fn update<F>(&self, key: K, f: F) -> Option<&V>
    where
        F: Fn(&V) -> V,
    {
        self.raw.root(&self.guard).update(key, f, &self.guard)
    }

    pub fn remove<Q: ?Sized>(&self, key: &Q) -> Option<&V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.raw.root(&self.guard).remove(key, &self.guard)
    }
}

pub struct Iter<'guard, K, V> {
    raw: raw::Iter<'guard, K, V>,
}

impl<'guard, K: 'guard, V: 'guard> Iterator for Iter<'guard, K, V> {
    type Item = (&'guard K, &'guard V);

    fn next(&mut self) -> Option<Self::Item> {
        self.raw.next()
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct OccupiedError<'a, V> {
    pub current: &'a V,
    pub not_inserted: V,
}
