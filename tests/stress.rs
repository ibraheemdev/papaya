// adapted from: https://github.com/jonhoo/flurry/tree/main/tests/jdk

use papaya::{HashMap, ResizeMode};
use rand::prelude::*;

use std::hash::Hash;
use std::sync::Barrier;
use std::thread;

fn with_map<K, V>(mut test: impl FnMut(&dyn Fn() -> HashMap<K, V>)) {
    test(&(|| HashMap::new().resize_mode(ResizeMode::Blocking)));
    test(&(|| HashMap::new().resize_mode(ResizeMode::Incremental(1))));
    test(&(|| HashMap::new().resize_mode(ResizeMode::Incremental(128))));
}

#[test]
fn contains_key_stress() {
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 256 };
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 1 << 10 };
    const ROUNDS: usize = if cfg!(miri) { 1 } else { 32 };

    with_map(|map| {
        let map = map();
        let mut content = [0; ENTRIES];

        {
            let guard = map.guard();
            for k in 0..ENTRIES {
                map.insert(k, k, &guard);
                content[k] = k;
            }
        }

        for _ in 0..ITERATIONS {
            let threads = thread::available_parallelism().unwrap().get().min(8);
            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        barrier.wait();
                        let guard = map.guard();
                        for i in 0..ENTRIES * ROUNDS {
                            let key = content[i % content.len()];
                            assert!(map.contains_key(&key, &guard));
                        }
                    });
                }
            });
        }
    });
}

#[test]
fn update_stress() {
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 64 };
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 1 << 14 };

    with_map(|map| {
        let map = map();

        {
            let guard = map.guard();
            for i in 0..ENTRIES {
                map.insert(i, 0, &guard);
            }
        }

        for t in 0..ITERATIONS {
            let threads = thread::available_parallelism().unwrap().get().min(8);
            let barrier = std::sync::Barrier::new(threads);

            thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        barrier.wait();
                        let guard = map.guard();
                        for i in 0..ENTRIES {
                            let new = *map.update(i, |v| v + 1, &guard).unwrap();
                            assert!((0..=(threads * (t + 1))).contains(&new));
                        }
                    });
                }
            });

            let guard = map.guard();
            for i in 0..ENTRIES {
                assert_eq!(*map.get(&i, &guard).unwrap(), threads * (t + 1));
            }
        }
    });
}

#[test]
fn insert_stress<'g>() {
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 64 };
    const ENTRIES: usize = if cfg!(miri) { 64 } else { 1 << 12 };

    #[derive(Hash, PartialEq, Eq, Clone, Copy)]
    struct KeyVal {
        _data: usize,
    }

    impl KeyVal {
        pub fn new() -> Self {
            let mut rng = rand::thread_rng();
            Self { _data: rng.gen() }
        }
    }

    with_map(|map| {
        for _ in 0..ITERATIONS {
            let map = map();
            let threads = thread::available_parallelism().unwrap().get().min(8);
            let barrier = Barrier::new(threads);
            thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        barrier.wait();
                        for _ in 0..ENTRIES {
                            let key = KeyVal::new();
                            map.insert(key, key, &map.guard());
                            assert!(map.contains_key(&key, &map.guard()));
                        }
                    });
                }
            });
        }
    });
}

#[test]
fn mixed_stress() {
    const ITERATIONS: usize = if cfg!(miri) { 1 } else { 48 };
    const CHUNK: usize = if cfg!(miri) { 48 } else { 1 << 14 };

    let run = |barrier: &Barrier, t: usize, map: &HashMap<usize, usize>, threads: usize| {
        barrier.wait();

        let (start, end) = (CHUNK * t, CHUNK * (t + 1));

        for i in start..end {
            assert_eq!(map.pin().insert(i, i + 1), None);
        }

        for i in start..end {
            assert_eq!(map.pin().get(&i), Some(&(i + 1)));
        }

        for i in start..end {
            assert_eq!(map.pin().update(i, |i| i - 1), Some(&i));
        }

        for i in start..end {
            assert_eq!(map.pin().remove(&i), Some(&i));
        }

        for i in start..end {
            assert_eq!(map.pin().get(&i), None);
        }

        for i in start..end {
            assert_eq!(map.pin().insert(i, i + 1), None);
        }

        for i in start..end {
            assert_eq!(map.pin().get(&i), Some(&(i + 1)));
        }

        for (&k, &v) in map.pin().iter() {
            assert!(k < CHUNK * threads);
            assert!(v == k || v == k + 1);
        }
    };

    with_map(|map| {
        for _ in 0..ITERATIONS {
            let map = map();
            let threads = thread::available_parallelism().unwrap().get().min(8);
            let barrier = Barrier::new(threads);

            thread::scope(|s| {
                for t in 0..threads {
                    let map = &map;
                    let barrier = &barrier;

                    s.spawn(move || run(barrier, t, map, threads));
                }
            });

            let v: Vec<_> = (0..CHUNK * threads).map(|i| (i, i + 1)).collect();
            let mut got: Vec<_> = map.pin().iter().map(|(&k, &v)| (k, v)).collect();
            got.sort();
            assert_eq!(v, got);
        }
    });
}

const SIZE: usize = if cfg!(miri) { 12 } else { 50_000 };

// there must be more things absent than present!
const ABSENT_SIZE: usize = if cfg!(miri) { 1 << 5 } else { 1 << 17 };
const ABSENT_MASK: usize = ABSENT_SIZE - 1;

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

fn t6<K, V>(map: &HashMap<K, V>, keys1: &[K], keys2: &[K], expect: usize)
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
        if map.get(&keys2[i & ABSENT_MASK], &guard).is_some() {
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

#[test]
fn everything() {
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
        t6(&map, keys, absent_keys, SIZE);
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
        ittest1(&map, SIZE);
        ittest2(&map, SIZE);
        ittest3(&map, SIZE);
    });
}
