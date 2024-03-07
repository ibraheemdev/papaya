use crate::raw;
use crate::seize::Guard;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

/// A lock-free hash table.
///
/// For more information, see the [crate-level documentation](crate);
pub struct HashMap<K, V, S = RandomState> {
    pub raw: raw::HashMap<K, V, S>,
}

impl<K, V> Default for HashMap<K, V> {
    fn default() -> Self {
        HashMap::new()
    }
}

impl<K, V> HashMap<K, V> {
    pub fn new() -> HashMap<K, V> {
        HashMap::with_capacity(32)
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

pub struct Pinned<'a, K, V, S> {
    guard: Guard<'a>,
    raw: &'a raw::HashMap<K, V, S>,
}

impl<'a, K, V, S> Pinned<'a, K, V, S>
where
    K: Clone + Hash + Eq + Sync + Send,
    V: Sync + Send,
    S: BuildHasher,
{
    pub fn capacity(&self) -> usize {
        self.raw.capacity(&self.guard)
    }

    pub fn get<Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw.root(&self.guard).get(key, &self.guard)
    }

    pub fn insert(&'a self, key: K, value: V) -> Option<&'a V> {
        self.raw.root(&self.guard).insert(key, value, &self.guard)
    }

    pub fn update<F>(&'a self, key: K, f: F) -> Option<&'a V>
    where
        F: Fn(&V) -> V,
    {
        self.raw.root(&self.guard).update(key, f, &self.guard)
    }

    pub fn remove<Q: ?Sized>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.raw.root(&self.guard).remove(key, &self.guard)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct OccupiedError<'a, V> {
    pub current: &'a V,
    pub not_inserted: V,
}
