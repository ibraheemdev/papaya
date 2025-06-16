//! Rayon parallel iterator implementations for papaya's HashMap.

use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::hash::{BuildHasher, Hash};

use crate::map::HashMapRef;
use crate::Guard;

/// Implementation of `IntoParallelIterator` for `HashMapRef`.
impl<'map, K, V, S, G> IntoParallelIterator for &'map HashMapRef<'map, K, V, S, G>
where
    K: Hash + Eq + Send + Sync,
    V: Send + Sync,
    S: BuildHasher + Send + Sync,
    G: Guard + Sync,
{
    type Iter = ParIter<'map, K, V, S, G>;
    type Item = (&'map K, &'map V);

    fn into_par_iter(self) -> Self::Iter {
        ParIter {
            map_ref: self,
        }
    }
}

/// Parallel iterator over entries in a HashMap.
/// 
/// This struct is created by calling `into_par_iter()` on a `&HashMapRef`.
pub struct ParIter<'map, K, V, S, G> {
    map_ref: &'map HashMapRef<'map, K, V, S, G>,
}

impl<K, V, S, G> std::fmt::Debug for ParIter<'_, K, V, S, G>
where
    K: std::fmt::Debug,
    V: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParIter").finish()
    }
}

impl<'map, K, V, S, G> ParallelIterator for ParIter<'map, K, V, S, G>
where
    K: Hash + Eq + Send + Sync + 'map,
    V: Send + Sync + 'map,
    S: BuildHasher + Send + Sync,
    G: Guard + Sync,
{
    type Item = (&'map K, &'map V);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: rayon::iter::plumbing::UnindexedConsumer<Self::Item>,
    {
        // Use the raw parallel iterator from the raw module
        self.map_ref.raw_par_iter().drive_unindexed(consumer)
    }
}