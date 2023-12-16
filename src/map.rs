use crate::raw;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};

use crate::seize::Guard;

pub struct HashMap<K, V, S = RandomState> {
    raw: raw::HashMap<K, V, S>,
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
    pub fn get<Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw.with_ref(|m| m.get(key, &self.guard), &self.guard)
    }

    pub fn get_8<Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw
            .with_ref(|m| m.get_8(key, &self.guard), &self.guard)
    }

    pub fn get_nosimd<Q>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.raw
            .with_ref(|m| m.get_nosimd(key, &self.guard), &self.guard)
    }

    pub fn insert(&'a self, key: K, value: V) -> Option<&'a V> {
        self.raw
            .with_ref(|m| m.insert(key, value, &self.guard), &self.guard)
    }

    pub fn remove<Q: ?Sized>(&'a self, key: &Q) -> Option<&'a V>
    where
        K: Borrow<Q>,
        Q: Hash + Eq,
    {
        self.raw
            .with_ref(|m| m.remove(key, &self.guard), &self.guard)
    }
}

// try_insert
#[derive(Debug, PartialEq, Eq)]
pub struct OccupiedError<'a, V> {
    pub current: &'a V,
    pub not_inserted: V,
}
