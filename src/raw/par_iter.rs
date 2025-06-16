// Parallel iteration support for rayon
use super::*;
use super::alloc::RawTable;
use rayon::iter::plumbing::{Producer, UnindexedConsumer};
use rayon::iter::ParallelIterator;

/// Parallel iterator over raw table entries
pub struct ParIter<'g, K, V, G> {
    map: &'g HashMap<K, V, std::collections::hash_map::RandomState>,
    guard: &'g G,
}

impl<'g, K, V, G> ParIter<'g, K, V, G> {
    pub fn new<S>(map: &'g HashMap<K, V, S>, guard: &'g G) -> ParIter<'g, K, V, G> 
    where
        G: VerifiedGuard,
        S: 'g,
    {
        // We need to cast away the S parameter since we don't use it for iteration
        let map_ptr = map as *const HashMap<K, V, S>;
        let map_ref = unsafe { &*(map_ptr as *const HashMap<K, V, std::collections::hash_map::RandomState>) };
        
        ParIter { 
            map: map_ref,
            guard 
        }
    }
}

impl<'g, K: 'g, V: 'g, G> ParallelIterator for ParIter<'g, K, V, G>
where
    K: Send + Sync,
    V: Send + Sync,
    G: VerifiedGuard + Sync,
{
    type Item = (&'g K, &'g V);

    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        let table = self.map.root(self.guard);
        
        // If table is null, return empty
        if table.raw.is_null() {
            // Create an empty producer and bridge it
            let empty_table = SendTable {
                ptr: table.raw,
                mask: 0,
                limit: 0,
            };
            let producer = TableProducer {
                table: empty_table,
                guard: self.guard,
                start: 0,
                end: 0,
            };
            return rayon::iter::plumbing::bridge_producer_consumer(0, producer, consumer);
        }

        // Create a wrapper that makes Table Send
        let table_ref = SendTable {
            ptr: table.raw,
            mask: table.mask,
            limit: table.limit,
        };
        
        // Create indexed producer
        let producer = TableProducer {
            table: table_ref,
            guard: self.guard,
            start: 0,
            end: table.len(),
        };

        // Bridge from producer to consumer
        rayon::iter::plumbing::bridge_producer_consumer(table.len(), producer, consumer)
    }
}

/// Wrapper to make Table Send-safe
struct SendTable<T> {
    ptr: *mut RawTable<T>,
    mask: usize,
    limit: usize,
}

impl<T> Clone for SendTable<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SendTable<T> {}

// Safety: We only access the table through atomic operations and guards
unsafe impl<T> Send for SendTable<T> {}
unsafe impl<T> Sync for SendTable<T> {}

impl<T> SendTable<T> {
    fn as_table(&self) -> Table<T> {
        Table {
            raw: self.ptr,
            mask: self.mask,
            limit: self.limit,
        }
    }
}

/// Producer for splitting table ranges
struct TableProducer<'g, K, V, G> {
    table: SendTable<Entry<K, V>>,
    guard: &'g G,
    start: usize,
    end: usize,
}

impl<'g, K: 'g, V: 'g, G> Producer for TableProducer<'g, K, V, G>
where
    K: Send + Sync,
    V: Send + Sync,
    G: VerifiedGuard + Sync,
{
    type Item = (&'g K, &'g V);
    type IntoIter = TableChunkIter<'g, K, V, G>;

    fn into_iter(self) -> Self::IntoIter {
        TableChunkIter {
            table: self.table,
            guard: self.guard,
            current: self.start,
            end: self.end,
        }
    }

    fn split_at(self, index: usize) -> (Self, Self) {
        let mid = self.start + index;
        (
            TableProducer {
                table: self.table,
                guard: self.guard,
                start: self.start,
                end: mid,
            },
            TableProducer {
                table: self.table,
                guard: self.guard,
                start: mid,
                end: self.end,
            },
        )
    }
}


/// Iterator over a chunk of table entries
struct TableChunkIter<'g, K, V, G> {
    table: SendTable<Entry<K, V>>,
    guard: &'g G,
    current: usize,
    end: usize,
}

impl<'g, K: 'g, V: 'g, G> Iterator for TableChunkIter<'g, K, V, G>
where
    G: VerifiedGuard,
{
    type Item = (&'g K, &'g V);

    fn next(&mut self) -> Option<Self::Item> {
        while self.current < self.end {
            let index = self.current;
            self.current += 1;

            let table = self.table.as_table();
            
            // Load the entry metadata first
            // Safety: index is within bounds
            let meta = unsafe { table.meta(index) }.load(Ordering::Acquire);

            // Skip empty entries
            if meta == meta::EMPTY {
                continue;
            }

            // Load the full entry
            // Safety: index is within bounds
            let entry = self.guard
                .protect(unsafe { table.entry(index) }, Ordering::Acquire)
                .unpack();

            // Skip null/tombstone entries
            if entry.ptr.is_null() {
                continue;
            }

            // Skip copied entries (in incremental resize mode)
            if entry.tag() & Entry::COPIED != 0 {
                continue;
            }

            // Safety: We have a valid, protected entry
            let entry_ref = unsafe { &*entry.ptr };
            return Some((&entry_ref.key, &entry_ref.value));
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.current);
        // We don't know exactly how many valid entries there are
        (0, Some(remaining))
    }
}

impl<'g, K: 'g, V: 'g, G> ExactSizeIterator for TableChunkIter<'g, K, V, G>
where
    G: VerifiedGuard,
{
    fn len(&self) -> usize {
        self.end.saturating_sub(self.current)
    }
}

impl<'g, K: 'g, V: 'g, G> DoubleEndedIterator for TableChunkIter<'g, K, V, G>
where
    G: VerifiedGuard,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        while self.current < self.end {
            self.end -= 1;
            let index = self.end;

            let table = self.table.as_table();
            
            let meta = unsafe { table.meta(index) }.load(Ordering::Acquire);
            if meta == meta::EMPTY {
                continue;
            }

            let entry = self.guard
                .protect(unsafe { table.entry(index) }, Ordering::Acquire)
                .unpack();

            if entry.ptr.is_null() {
                continue;
            }

            if entry.tag() & Entry::COPIED != 0 {
                continue;
            }

            let entry_ref = unsafe { &*entry.ptr };
            return Some((&entry_ref.key, &entry_ref.value));
        }
        None
    }
}