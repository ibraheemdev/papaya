use papaya::{Compute, HashMap, Operation};
use rand::prelude::*;

use std::hash::Hash;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Barrier;
use std::thread;

mod common;
use common::{threads, with_map};

// Call `contains_key` in parallel for a shared set of keys.
#[test]
#[ignore]
fn contains_key_stress() {
    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 64,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 12,
        _ => 1 << 19,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();
            let mut content = vec![0; ENTRIES];

            {
                let guard = map.guard();
                for k in 0..ENTRIES {
                    map.insert(k, k, &guard);
                    content[k] = k;
                }
            }

            let threads = threads();
            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        barrier.wait();
                        let guard = map.guard();
                        for i in 0..ENTRIES {
                            let key = content[i % content.len()];
                            assert!(map.contains_key(&key, &guard));
                        }
                    });
                }
            });
        }
    });
}

// Call `insert` in parallel with each thread inserting a distinct set of keys.
#[test]
#[ignore]
fn insert_stress() {
    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 64,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 12,
        _ => 1 << 17,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    #[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
    struct Random(usize);

    fn random() -> Random {
        Random(rand::thread_rng().gen())
    }

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();
            let threads = threads();
            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        barrier.wait();
                        for _ in 0..ENTRIES {
                            let key = random();
                            map.insert(key, key, &map.guard());
                            assert!(map.contains_key(&key, &map.guard()));
                        }
                    });
                }
            });
            assert_eq!(map.len(), ENTRIES * threads);
        }
    });
}

// Call `insert` in parallel on a small shared set of keys.
#[test]
#[ignore]
fn insert_overwrite_stress() {
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 256 };
    const OPERATIONS: usize = match () {
        _ if cfg!(miri) => 1,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 9,
        _ => 1 << 10,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    let entries = || {
        let mut entries = (0..(OPERATIONS))
            .flat_map(|_| (0..ENTRIES))
            .collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        entries.shuffle(&mut rng);
        entries
    };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();

            let counters = (0..ENTRIES)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>();

            let threads = threads();
            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                let mut handles = Vec::with_capacity(threads);
                for _ in 0..threads {
                    let h = s.spawn(|| {
                        let mut seen = (0..ENTRIES)
                            .map(|_| Vec::with_capacity(OPERATIONS))
                            .collect::<Vec<_>>();
                        let entries = entries();

                        barrier.wait();
                        for i in entries {
                            let value = counters[i].fetch_add(1, Ordering::Relaxed);
                            if let Some(&prev) = map.insert(i, value, &map.guard()) {
                                // Keep track of values we overwrite.
                                seen[i].push(prev);
                            }
                            assert!(map.contains_key(&i, &map.guard()));
                        }
                        seen
                    });
                    handles.push(h);
                }

                let mut seen = (0..ENTRIES)
                    .map(|_| Vec::<usize>::with_capacity(threads * OPERATIONS))
                    .collect::<Vec<_>>();

                for h in handles {
                    let values = h.join().unwrap();
                    for (i, values) in values.iter().enumerate() {
                        seen[i].extend(values);
                    }
                }
                for i in 0..seen.len() {
                    seen[i].push(*map.pin().get(&i).unwrap());
                }

                // Ensure every insert operation was consistent.
                let operations = (0..(threads * OPERATIONS)).collect::<Vec<_>>();
                for mut values in seen {
                    values.sort();
                    assert_eq!(values, operations);
                }
            });

            assert_eq!(map.len(), ENTRIES);
        }
    });
}

// Call `update` in parallel for a small shared set of keys.
#[test]
#[ignore]
fn update_stress() {
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 256 };
    const OPERATIONS: usize = match () {
        _ if cfg!(miri) => 1,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 9,
        _ => 1 << 10,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 48 };

    let entries = || {
        let mut entries = (0..(OPERATIONS))
            .flat_map(|_| (0..ENTRIES))
            .collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        entries.shuffle(&mut rng);
        entries
    };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();

            {
                let guard = map.guard();
                for i in 0..ENTRIES {
                    map.insert(i, 0, &guard);
                }
            }

            let threads = threads();
            let barrier = Barrier::new(threads);

            thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        let entries = entries();
                        barrier.wait();
                        for i in entries {
                            let guard = map.guard();
                            let new = *map.update(i, |v| v + 1, &guard).unwrap();
                            assert!((0..=(threads * OPERATIONS)).contains(&new));
                        }
                    });
                }
            });

            let guard = map.guard();
            for i in 0..ENTRIES {
                assert_eq!(*map.get(&i, &guard).unwrap(), threads * OPERATIONS);
            }
        }
    });
}

// Call `update` in parallel for a shared set of keys, with a single thread dedicated
// to inserting unrelated keys. This is likely to cause interference with incremental resizing.
#[test]
#[ignore]
fn update_insert_stress() {
    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 64,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 12,
        _ => 1 << 18,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 48 };

    with_map(|map| {
        let map = map();

        {
            let guard = map.guard();
            for i in 0..ENTRIES {
                map.insert(i, 0, &guard);
            }
        }

        for t in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let threads = threads();
            let barrier = Barrier::new(threads);

            let threads = &threads;
            thread::scope(|s| {
                for _ in 0..(threads - 1) {
                    s.spawn(|| {
                        barrier.wait();
                        let guard = map.guard();
                        for i in 0..ENTRIES {
                            let new = *map.update(i, |v| v + 1, &guard).unwrap();
                            assert!((0..=(threads * (t + 1))).contains(&new));
                        }
                    });
                }

                s.spawn(|| {
                    barrier.wait();
                    let guard = map.guard();
                    for i in ENTRIES..(ENTRIES * 2) {
                        map.insert(i, usize::MAX, &guard);
                    }
                });
            });

            let guard = map.guard();
            for i in 0..ENTRIES {
                assert_eq!(*map.get(&i, &guard).unwrap(), (threads - 1) * (t + 1));
            }

            for i in ENTRIES..(ENTRIES * 2) {
                assert_eq!(*map.get(&i, &guard).unwrap(), usize::MAX);
            }
        }
    });
}

// Call `update_or_insert` in parallel for a small shared set of keys.
// Stresses the `insert` -> `update` transition in `compute`.
#[test]
#[ignore]
fn update_or_insert_stress() {
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 256 };
    const OPERATIONS: usize = match () {
        _ if cfg!(miri) => 1,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 9,
        _ => 1 << 10,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 48 };

    let threads = threads();

    let entries = (0..(threads * OPERATIONS))
        .flat_map(|_| (0..ENTRIES))
        .collect::<Vec<_>>();

    let chunk = ENTRIES * OPERATIONS;

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();

            let mut entries = entries.clone();
            let mut rng = rand::thread_rng();
            entries.shuffle(&mut rng);

            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                for t in 0..threads {
                    let range = (chunk * t)..(chunk * (t + 1));

                    s.spawn(|| {
                        barrier.wait();
                        let guard = map.guard();
                        for i in &entries[range] {
                            map.update_or_insert(i, |v| v + 1, 1, &guard);
                        }
                    });
                }
            });

            let guard = map.guard();
            for i in 0..ENTRIES {
                assert_eq!(*map.get(&i, &guard).unwrap(), threads * OPERATIONS);
            }

            assert_eq!(map.len(), ENTRIES);
        }
    });
}

// Call `update_or_insert` in parallel for a small shared set of keys, with some
// threads removing and reinserting values.
//
// Stresses the `update` <-> `insert` transition in `compute`.
#[test]
#[ignore]
fn remove_update_or_insert_stress() {
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 256 };
    const OPERATIONS: usize = match () {
        _ if cfg!(miri) => 1,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 4,
        _ => 1 << 9,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    let threads = threads();

    let entries = || {
        let mut entries = (0..(OPERATIONS))
            .flat_map(|_| (0..ENTRIES))
            .collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        entries.shuffle(&mut rng);
        entries
    };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();

            let group = threads.checked_div(2).unwrap();
            let barrier = Barrier::new(threads + 1);
            thread::scope(|s| {
                for _ in 0..group {
                    s.spawn(|| {
                        let entries = entries();
                        barrier.wait();
                        let guard = map.guard();
                        for i in entries {
                            map.update_or_insert(i, |v| v + 1, 1, &guard);
                        }
                    });
                }

                for _ in 0..group {
                    s.spawn(|| {
                        let entries = entries();
                        barrier.wait();
                        let guard = map.guard();

                        for i in entries {
                            if let Some(&value) = map.remove(&i, &guard) {
                                map.update_or_insert(i, |v| v + value, value, &guard);
                            }
                        }
                    });
                }

                s.spawn(|| {
                    barrier.wait();
                    let guard = map.guard();
                    for i in ENTRIES..(ENTRIES * OPERATIONS) {
                        map.insert(i, usize::MAX, &guard);
                    }
                });
            });

            let guard = map.guard();
            assert_eq!(map.len(), ENTRIES * OPERATIONS);

            for i in 0..ENTRIES {
                assert_eq!(*map.get(&i, &guard).unwrap(), group * OPERATIONS);
            }

            for i in ENTRIES..(ENTRIES * OPERATIONS) {
                assert_eq!(*map.get(&i, &guard).unwrap(), usize::MAX);
            }
        }
    });
}

// Call `update_or_insert` in parallel for a small shared set of keys, with some
// threads conditionally removing and reinserting values.
//
// Stresses the `remove` <-> `update` transition in `compute`.
#[test]
#[ignore]
fn conditional_remove_update_or_insert_stress() {
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 256 };
    const OPERATIONS: usize = match () {
        _ if cfg!(miri) => 1,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 5,
        _ => 1 << 9,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    let threads = threads();

    let entries = || {
        let mut entries = (0..(OPERATIONS))
            .flat_map(|_| (0..ENTRIES))
            .collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        entries.shuffle(&mut rng);
        entries
    };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();

            let group = threads.checked_div(2).unwrap();
            let barrier = Barrier::new(threads + 1);
            thread::scope(|s| {
                for _ in 0..group {
                    s.spawn(|| {
                        let entries = entries();
                        barrier.wait();
                        let guard = map.guard();
                        for i in entries {
                            map.update_or_insert(i, |v| v + 1, 1, &guard);
                        }
                    });
                }

                for _ in 0..group {
                    s.spawn(|| {
                        let entries = entries();
                        barrier.wait();
                        let guard = map.guard();

                        for i in entries {
                            let compute = |entry| match entry {
                                Some((_, value)) if value % 2 == 0 => Operation::Remove,
                                _ => Operation::Abort(()),
                            };

                            if let Compute::Removed(_, &value) = map.compute(i, compute, &guard) {
                                map.update_or_insert(i, |v| v + value, value, &guard);
                            }
                        }
                    });
                }

                s.spawn(|| {
                    barrier.wait();
                    let guard = map.guard();
                    for i in ENTRIES..(ENTRIES * OPERATIONS) {
                        map.insert(i, usize::MAX, &guard);
                    }
                });
            });

            let guard = map.guard();
            assert_eq!(map.len(), ENTRIES * OPERATIONS);

            for i in 0..ENTRIES {
                assert_eq!(*map.get(&i, &guard).unwrap(), group * OPERATIONS);
            }

            for i in ENTRIES..(ENTRIES * OPERATIONS) {
                assert_eq!(*map.get(&i, &guard).unwrap(), usize::MAX);
            }
        }
    });
}

// Call `remove` and `insert` in parallel for a shared set of keys.
#[test]
#[ignore]
fn insert_remove_stress() {
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 256 };
    const OPERATIONS: usize = match () {
        _ if cfg!(miri) => 1,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 8,
        _ => 1 << 9,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 48 };

    let entries = || {
        let mut entries = (0..(OPERATIONS))
            .flat_map(|_| (0..ENTRIES))
            .collect::<Vec<_>>();
        let mut rng = rand::thread_rng();
        entries.shuffle(&mut rng);
        entries
    };

    let threads = threads();
    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();

            let group = threads.checked_div(2).unwrap();
            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                for _ in 0..group {
                    s.spawn(|| {
                        let entries = entries();
                        barrier.wait();

                        let guard = map.guard();
                        for i in entries {
                            map.insert(i, i, &guard);
                        }
                    });
                }

                for _ in 0..group {
                    s.spawn(|| {
                        let entries = entries();
                        barrier.wait();

                        let guard = map.guard();
                        for i in entries {
                            if map.remove(&i, &guard).is_some() {
                                map.insert(i, i, &guard);
                            }
                        }
                    });
                }
            });

            let guard = map.guard();
            assert_eq!(map.len(), ENTRIES);

            for i in 0..ENTRIES {
                assert_eq!(map.get(&i, &guard), Some(&i));
            }
        }
    });
}

// Call `remove` in parallel for a shared set of keys with other threads calling `update`,
// and a dedicated thread for inserting unrelated keys. This is likely to cause interference
// with incremental resizing.
#[test]
#[ignore]
fn remove_mixed_stress() {
    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 64,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 11,
        _ => 1 << 17,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 64 };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();

            {
                let guard = map.guard();
                for i in 0..ENTRIES {
                    map.insert(i, 0, &guard);
                }
            }

            let threads = threads().max(3);
            let barrier = Barrier::new(threads);

            thread::scope(|s| {
                for _ in 0..(threads - 2) {
                    s.spawn(|| {
                        let mut entries = (0..ENTRIES).collect::<Vec<_>>();
                        let mut rng = rand::thread_rng();
                        entries.shuffle(&mut rng);

                        barrier.wait();
                        let guard = map.guard();

                        loop {
                            let mut empty = true;
                            for &i in entries.iter() {
                                if map.update(i, |v| v + 1, &guard).is_some() {
                                    empty = false;
                                }
                            }
                            if empty {
                                break;
                            }
                        }
                    });
                }

                s.spawn(|| {
                    barrier.wait();
                    let guard = map.guard();
                    for i in 0..ENTRIES {
                        map.remove(&i, &guard);
                    }

                    for i in 0..ENTRIES {
                        assert_eq!(map.get(&i, &guard), None);
                    }
                });

                s.spawn(|| {
                    barrier.wait();
                    let guard = map.guard();
                    for i in ENTRIES..(ENTRIES * 2) {
                        map.insert(i, usize::MAX, &guard);
                    }
                });
            });

            let guard = map.guard();
            for i in 0..ENTRIES {
                assert_eq!(map.get(&i, &guard), None);
            }

            for i in ENTRIES..(ENTRIES * 2) {
                assert_eq!(*map.get(&i, &guard).unwrap(), usize::MAX);
            }

            assert_eq!(map.len(), ENTRIES);
        }
    });
}

// Performs insert and remove operations with each thread operating on a distinct set of keys.
#[test]
#[ignore]
fn insert_remove_chunk_stress() {
    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 48,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 11,
        _ => 1 << 17,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 48 };

    let run =
        |barrier: &Barrier, chunk: Range<usize>, map: &HashMap<usize, usize>, threads: usize| {
            barrier.wait();

            for i in chunk.clone() {
                assert_eq!(map.pin().insert(i, i), None);
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().get(&i), Some(&i));
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().remove(&i), Some(&i));
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().get(&i), None);
            }

            if !cfg!(papaya_stress) {
                for (&k, &v) in map.pin().iter() {
                    assert!(k < ENTRIES * threads);
                    assert!(v == k);
                }
            }
        };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();
            let threads = threads();
            let barrier = Barrier::new(threads);

            thread::scope(|s| {
                for i in 0..threads {
                    let map = &map;
                    let barrier = &barrier;

                    let chunk = (ENTRIES * i)..(ENTRIES * (i + 1));
                    s.spawn(move || run(barrier, chunk, map, threads));
                }
            });

            if !cfg!(papaya_stress) {
                let got: Vec<_> = map.pin().into_iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(got, []);
            }

            assert_eq!(map.len(), 0);
        }
    });
}

// Performs a mix of operations with each thread operating on a distinct set of keys.
#[test]
#[ignore]
fn mixed_chunk_stress() {
    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 48,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 11,
        _ => 1 << 16,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    let run =
        |barrier: &Barrier, chunk: Range<usize>, map: &HashMap<usize, usize>, threads: usize| {
            barrier.wait();

            for i in chunk.clone() {
                assert_eq!(map.pin().insert(i, i + 1), None);
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().get(&i), Some(&(i + 1)));
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().update(i, |i| i - 1), Some(&i));
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().remove(&i), Some(&i));
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().get(&i), None);
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().insert(i, i + 1), None);
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().get(&i), Some(&(i + 1)));
            }

            if !cfg!(papaya_stress) {
                for (&k, &v) in map.pin().iter() {
                    assert!(k < ENTRIES * threads);
                    assert!(v == k || v == k + 1);
                }
            }
        };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();
            let threads = threads();
            let barrier = Barrier::new(threads);

            thread::scope(|s| {
                for i in 0..threads {
                    let map = &map;
                    let barrier = &barrier;

                    let chunk = (ENTRIES * i)..(ENTRIES * (i + 1));
                    s.spawn(move || run(barrier, chunk, map, threads));
                }
            });

            if !cfg!(papaya_stress) {
                let v: Vec<_> = (0..ENTRIES * threads).map(|i| (i, i + 1)).collect();
                let mut got: Vec<_> = map.pin().iter().map(|(&k, &v)| (k, v)).collect();
                got.sort();
                assert_eq!(v, got);
            }

            assert_eq!(map.len(), ENTRIES * threads);
        }
    });
}

// Performs a mix of operations with each thread operating on a specific entry within
// a distinct set of keys. This is more likely to cause interference with incremental resizing.
#[test]
#[ignore]
fn mixed_entry_stress() {
    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 100,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 11,
        _ => 1 << 11,
    };
    const OPERATIONS: usize = if cfg!(miri) { 1 } else { 72 };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    let run =
        |barrier: &Barrier, chunk: Range<usize>, map: &HashMap<usize, usize>, threads: usize| {
            barrier.wait();

            for i in chunk.clone() {
                for _ in 0..OPERATIONS {
                    assert_eq!(map.pin().insert(i, i + 1), None);
                    assert_eq!(map.pin().get(&i), Some(&(i + 1)));
                    assert_eq!(map.pin().update(i, |i| i + 1), Some(&(i + 2)));
                    assert_eq!(map.pin().remove(&i), Some(&(i + 2)));
                    assert_eq!(map.pin().get(&i), None);
                    assert_eq!(map.pin().update(i, |i| i + 1), None);
                }
            }

            for i in chunk.clone() {
                assert_eq!(map.pin().get(&i), None);
            }

            if !cfg!(papaya_stress) {
                for (&k, &v) in map.pin().iter() {
                    assert!(k < ENTRIES * threads);
                    assert!(v == k + 1 || v == k + 2);
                }
            }
        };

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();
            let threads = threads();
            let barrier = Barrier::new(threads);

            thread::scope(|s| {
                for i in 0..threads {
                    let map = &map;
                    let barrier = &barrier;

                    let chunk = (ENTRIES * i)..(ENTRIES * (i + 1));
                    s.spawn(move || run(barrier, chunk, map, threads));
                }
            });

            if !cfg!(papaya_stress) {
                let got: Vec<_> = map.pin().iter().map(|(&k, &v)| (k, v)).collect();
                assert_eq!(got, []);
            }
            assert_eq!(map.len(), 0);
        }
    });
}

// Clears the map during concurrent insertion.
//
// This test is relatively vague as there are few guarantees observable from concurrent calls
// to `clear` but still useful for Miri.
#[test]
#[ignore]
fn clear_stress() {
    if cfg!(papaya_stress) {
        return;
    }

    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 64,
        _ if cfg!(papaya_asan) => 1 << 12,
        _ => 1 << 17,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    #[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
    struct Random(usize);

    fn random() -> Random {
        Random(rand::thread_rng().gen())
    }

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();
            let threads = threads();
            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                for _ in 0..(threads - 1) {
                    s.spawn(|| {
                        barrier.wait();
                        for _ in 0..ENTRIES {
                            let key = random();
                            map.pin().insert(key, key);
                        }
                    });
                }

                s.spawn(|| {
                    barrier.wait();
                    for _ in 0..(threads * 20) {
                        map.pin().clear();
                    }
                });
            });

            map.pin().clear();
            assert_eq!(map.len(), 0);
            assert_eq!(map.pin().iter().count(), 0);
        }
    });
}

// Retains the map during concurrent insertion.
#[test]
#[ignore]
fn retain_stress() {
    if cfg!(papaya_stress) {
        return;
    }

    const ENTRIES: usize = match () {
        _ if cfg!(miri) => 64,
        _ if cfg!(papaya_asan) => 1 << 12,
        _ => 1 << 17,
    };
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 32 };

    #[derive(Hash, PartialEq, Eq, Clone, Copy, Debug)]
    struct Random(usize);

    fn random() -> Random {
        Random(rand::thread_rng().gen())
    }

    with_map(|map| {
        for _ in (0..ITERATIONS).inspect(|e| debug!("{e}/{ITERATIONS}")) {
            let map = map();
            let threads = threads();
            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                for _ in 0..(threads - 1) {
                    s.spawn(|| {
                        barrier.wait();
                        for _ in 0..ENTRIES {
                            let key = random();
                            map.insert(key, key, &map.guard());
                            assert!(map.contains_key(&key, &map.guard()));
                        }
                    });
                }

                s.spawn(|| {
                    barrier.wait();
                    for _ in 0..(threads * 20) {
                        map.pin().retain(|_, _| true);
                    }
                });
            });

            assert_eq!(map.len(), ENTRIES * (threads - 1));
            assert_eq!(map.pin().iter().count(), ENTRIES * (threads - 1));
        }
    });
}

// Adapted from: https://github.com/jonhoo/flurry/tree/main/tests/jdk
#[test]
#[ignore]
fn everything() {
    const SIZE: usize = match () {
        _ if cfg!(miri) => 1 << 5,
        _ if cfg!(papaya_stress) || cfg!(papaya_asan) => 1 << 13,
        _ => 1 << 20,
    };
    // There must be more things absent than present!
    const ABSENT_SIZE: usize = SIZE << 1;
    const ABSENT_MASK: usize = ABSENT_SIZE - 1;

    let mut rng = rand::thread_rng();

    with_map(|map| {
        let map = map();
        let mut keys: Vec<_> = (0..ABSENT_SIZE + SIZE).collect();
        keys.shuffle(&mut rng);
        let absent_keys = &keys[0..ABSENT_SIZE];
        let keys = &keys[ABSENT_SIZE..];

        // put (absent)
        t3(&map, keys, SIZE);
        // put (present)
        t3(&map, keys, 0);
        // contains_key (present & absent)
        t7(&map, keys, absent_keys);
        // contains_key (present)
        t4(&map, keys, SIZE);
        // contains_key (absent)
        t4(&map, absent_keys, 0);
        // get
        t6(&map, keys, absent_keys, SIZE, ABSENT_MASK);
        // get (present)
        t1(&map, keys, SIZE);
        // get (absent)
        t1(&map, absent_keys, 0);
        // remove (absent)
        t2(&map, absent_keys, 0);
        // remove (present)
        t5(&map, keys, SIZE / 2);
        // put (half present)
        t3(&map, keys, SIZE / 2);

        // iter, keys, values (present)
        if !cfg!(papaya_stress) {
            ittest1(&map, SIZE);
            ittest2(&map, SIZE);
            ittest3(&map, SIZE);
        }
    });

    fn t1<K, V>(map: &HashMap<K, V>, keys: &[K], expect: usize)
    where
        K: Sync + Send + Clone + Hash + Ord,
        V: Sync + Send,
    {
        let mut sum = 0;
        let iters = 4;
        let guard = map.guard();
        for _ in 0..iters {
            for key in keys {
                if map.get(key, &guard).is_some() {
                    sum += 1;
                }
            }
        }
        assert_eq!(sum, expect * iters);
    }

    fn t2<K>(map: &HashMap<K, usize>, keys: &[K], expect: usize)
    where
        K: Sync + Send + Copy + Hash + Ord + std::fmt::Display,
    {
        let mut sum = 0;
        let guard = map.guard();
        for key in keys {
            if map.remove(key, &guard).is_some() {
                sum += 1;
            }
        }
        assert_eq!(sum, expect);
    }

    fn t3<K>(map: &HashMap<K, usize>, keys: &[K], expect: usize)
    where
        K: Sync + Send + Copy + Hash + Ord,
    {
        let mut sum = 0;
        let guard = map.guard();
        for i in 0..keys.len() {
            if map.insert(keys[i], 0, &guard).is_none() {
                sum += 1;
            }
        }
        assert_eq!(sum, expect);
    }

    fn t4<K>(map: &HashMap<K, usize>, keys: &[K], expect: usize)
    where
        K: Sync + Send + Copy + Hash + Ord,
    {
        let mut sum = 0;
        let guard = map.guard();
        for i in 0..keys.len() {
            if map.contains_key(&keys[i], &guard) {
                sum += 1;
            }
        }
        assert_eq!(sum, expect);
    }

    fn t5<K>(map: &HashMap<K, usize>, keys: &[K], expect: usize)
    where
        K: Sync + Send + Copy + Hash + Ord,
    {
        let mut sum = 0;
        let guard = map.guard();
        let mut i = keys.len() as isize - 2;
        while i >= 0 {
            if map.remove(&keys[i as usize], &guard).is_some() {
                sum += 1;
            }
            i -= 2;
        }
        assert_eq!(sum, expect);
    }

    fn t6<K, V>(map: &HashMap<K, V>, keys1: &[K], keys2: &[K], expect: usize, mask: usize)
    where
        K: Sync + Send + Clone + Hash + Ord,
        V: Sync + Send,
    {
        let mut sum = 0;
        let guard = map.guard();
        for i in 0..expect {
            if map.get(&keys1[i], &guard).is_some() {
                sum += 1;
            }
            if map.get(&keys2[i & mask], &guard).is_some() {
                sum += 1;
            }
        }
        assert_eq!(sum, expect);
    }

    fn t7<K>(map: &HashMap<K, usize>, k1: &[K], k2: &[K])
    where
        K: Sync + Send + Copy + Hash + Ord,
    {
        let mut sum = 0;
        let guard = map.guard();
        for i in 0..k1.len() {
            if map.contains_key(&k1[i], &guard) {
                sum += 1;
            }
            if map.contains_key(&k2[i], &guard) {
                sum += 1;
            }
        }
        assert_eq!(sum, k1.len());
    }

    fn ittest1<K>(map: &HashMap<K, usize>, expect: usize)
    where
        K: Sync + Send + Copy + Hash + Eq,
    {
        let mut sum = 0;
        let guard = map.guard();
        for _ in map.keys(&guard) {
            sum += 1;
        }
        assert_eq!(sum, expect);
    }

    fn ittest2<K>(map: &HashMap<K, usize>, expect: usize)
    where
        K: Sync + Send + Copy + Hash + Eq,
    {
        let mut sum = 0;
        let guard = map.guard();
        for _ in map.values(&guard) {
            sum += 1;
        }
        assert_eq!(sum, expect);
    }

    fn ittest3<K>(map: &HashMap<K, usize>, expect: usize)
    where
        K: Sync + Send + Copy + Hash + Eq,
    {
        let mut sum = 0;
        let guard = map.guard();
        for _ in map.iter(&guard) {
            sum += 1;
        }
        assert_eq!(sum, expect);
    }
}
