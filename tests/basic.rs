// adapted from: https://github.com/jonhoo/flurry/blob/main/tests/basic.rs

use papaya::{HashMap, ResizeMode};

use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
use std::sync::Arc;

fn with_map<K, V>(test: impl Fn(HashMap<K, V>)) {
    test(HashMap::new().resize_mode(ResizeMode::Blocking));
    test(HashMap::new().resize_mode(ResizeMode::Incremental(1)));
    test(HashMap::new().resize_mode(ResizeMode::Incremental(128)));
}

#[test]
fn new() {
    with_map::<usize, usize>(|map| drop(map));
}

#[test]
fn clear() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        {
            map.insert(0, 1, &guard);
            map.insert(1, 1, &guard);
            map.insert(2, 1, &guard);
            map.insert(3, 1, &guard);
            map.insert(4, 1, &guard);
        }
        map.clear(&guard);
        assert!(map.is_empty());
    });
}

#[test]
fn insert() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        let old = map.insert(42, 0, &guard);
        assert!(old.is_none());
    });
}

#[test]
fn get_empty() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        let e = map.get(&42, &guard);
        assert!(e.is_none());
    });
}

#[test]
fn get_key_value_empty() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        let e = map.get_key_value(&42, &guard);
        assert!(e.is_none());
    });
}

#[test]
fn remove_empty() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        let old = map.remove(&42, &guard);
        assert!(old.is_none());
    });
}

#[test]
fn insert_and_remove() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        map.insert(42, 0, &guard);
        let old = map.remove(&42, &guard).unwrap();
        assert_eq!(old, &0);
        assert!(map.get(&42, &guard).is_none());
    });
}

#[test]
fn insert_and_get() {
    with_map::<usize, usize>(|map| {
        map.insert(42, 0, &map.guard());

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &0);
        }
    });
}

#[test]
fn insert_and_get_key_value() {
    with_map::<usize, usize>(|map| {
        map.insert(42, 0, &map.guard());

        {
            let guard = map.guard();
            let e = map.get_key_value(&42, &guard).unwrap();
            assert_eq!(e, (&42, &0));
        }
    });
}

#[test]
fn reinsert() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        map.insert(42, 0, &guard);
        let old = map.insert(42, 1, &guard);
        assert_eq!(old, Some(&0));

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &1);
        }
    });
}

#[test]
fn update() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        map.insert(42, 0, &guard);
        let new = map.update(42, |v| v + 1, &guard);
        assert_eq!(new, Some(&1));

        {
            let guard = map.guard();
            let e = map.get(&42, &guard).unwrap();
            assert_eq!(e, &1);
        }
    });
}

#[test]
fn update_empty() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        let new = map.update(42, |v| v + 1, &guard);
        assert!(new.is_none());

        {
            let guard = map.guard();
            assert!(map.get(&42, &guard).is_none());
        }
    });
}

#[test]
fn concurrent_insert() {
    with_map::<usize, usize>(|map| {
        let map = Arc::new(map);

        let map1 = map.clone();
        let t1 = std::thread::spawn(move || {
            for i in 0..64 {
                map1.insert(i, 0, &map1.guard());
            }
        });
        let map2 = map.clone();
        let t2 = std::thread::spawn(move || {
            for i in 0..64 {
                map2.insert(i, 1, &map2.guard());
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let guard = map.guard();
        for i in 0..64 {
            let v = map.get(&i, &guard).unwrap();
            assert!(v == &0 || v == &1);

            let kv = map.get_key_value(&i, &guard).unwrap();
            assert!(kv == (&i, &0) || kv == (&i, &1));
        }
    });
}

#[test]
fn concurrent_remove() {
    with_map::<usize, usize>(|map| {
        let map = Arc::new(map);

        {
            let guard = map.guard();
            for i in 0..64 {
                map.insert(i, i, &guard);
            }
        }

        let map1 = map.clone();
        let t1 = std::thread::spawn(move || {
            let guard = map1.guard();
            for i in 0..64 {
                if let Some(v) = map1.remove(&i, &guard) {
                    assert_eq!(v, &i);
                }
            }
        });
        let map2 = map.clone();
        let t2 = std::thread::spawn(move || {
            let guard = map2.guard();
            for i in 0..64 {
                if let Some(v) = map2.remove(&i, &guard) {
                    assert_eq!(v, &i);
                }
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // after joining the threads, the map should be empty
        let guard = map.guard();
        for i in 0..64 {
            assert!(map.get(&i, &guard).is_none());
        }
    });
}

#[test]
fn concurrent_update() {
    with_map::<usize, usize>(|map| {
        let map = Arc::new(map);

        {
            let guard = map.guard();
            for i in 0..64 {
                map.insert(i, i, &guard);
            }
        }

        let map1 = map.clone();
        let t1 = std::thread::spawn(move || {
            let guard = map1.guard();
            for i in 0..64 {
                let new = *map1.update(i, |v| v + 1, &guard).unwrap();
                assert!(new == i + 1 || new == i + 2);
            }
        });
        let map2 = map.clone();
        let t2 = std::thread::spawn(move || {
            let guard = map2.guard();
            for i in 0..64 {
                let new = *map2.update(i, |v| v + 1, &guard).unwrap();
                assert!(new == i + 1 || new == i + 2);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // after joining the threads, the map should be empty
        let guard = map.guard();
        for i in 0..64 {
            assert_eq!(map.get(&i, &guard), Some(&(i + 2)));
        }
    });
}

#[test]
#[cfg_attr(miri, ignore)]
fn concurrent_resize_and_get() {
    with_map::<usize, usize>(|map| {
        let map = Arc::new(map);

        {
            let guard = map.guard();
            for i in 0..1024 {
                map.insert(i, i, &guard);
            }
        }

        let map1 = map.clone();
        // t1 is using reserve to trigger a bunch of resizes
        let t1 = std::thread::spawn(move || {
            let guard = map1.guard();
            // there should be 2 ** 10 capacity already, so trigger additional resizes
            for power in 11..16 {
                map1.reserve(1 << power, &guard);
            }
        });
        let map2 = map.clone();
        // t2 is retrieving existing keys a lot, attempting to encounter a BinEntry::Moved
        let t2 = std::thread::spawn(move || {
            let guard = map2.guard();
            for _ in 0..32 {
                for i in 0..1024 {
                    let v = map2.get(&i, &guard).unwrap();
                    assert_eq!(v, &i);
                }
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // make sure all the entries still exist after all the resizes
        {
            let guard = map.guard();

            for i in 0..1024 {
                let v = map.get(&i, &guard).unwrap();
                assert_eq!(v, &i);
            }
        }
    });
}

#[test]
fn current_kv_dropped() {
    let dropped1 = Arc::new(0);
    let dropped2 = Arc::new(0);

    with_map::<Arc<usize>, Arc<usize>>(|map| {
        map.insert(dropped1.clone(), dropped2.clone(), &map.guard());
        assert_eq!(Arc::strong_count(&dropped1), 2);
        assert_eq!(Arc::strong_count(&dropped2), 2);

        drop(map);

        // dropping the map should immediately drop (not deferred) all keys and values
        assert_eq!(Arc::strong_count(&dropped1), 1);
        assert_eq!(Arc::strong_count(&dropped2), 1);
    });
}

#[test]
fn empty_maps_equal() {
    with_map::<usize, usize>(|map1| {
        with_map::<usize, usize>(|map2| {
            assert_eq!(map1, map2);
            assert_eq!(map2, map1);
        });
    });
}

#[test]
fn different_size_maps_not_equal() {
    with_map::<usize, usize>(|map1| {
        with_map::<usize, usize>(|map2| {
            {
                let guard1 = map1.guard();
                let guard2 = map2.guard();

                map1.insert(1, 0, &guard1);
                map1.insert(2, 0, &guard1);
                map1.insert(3, 0, &guard1);

                map2.insert(1, 0, &guard2);
                map2.insert(2, 0, &guard2);
            }

            assert_ne!(map1, map2);
            assert_ne!(map2, map1);
        });
    });
}

#[test]
fn same_values_equal() {
    with_map::<usize, usize>(|map1| {
        with_map::<usize, usize>(|map2| {
            {
                map1.pin().insert(1, 0);
                map2.pin().insert(1, 0);
            }

            assert_eq!(map1, map2);
            assert_eq!(map2, map1);
        });
    });
}

#[test]
fn different_values_not_equal() {
    with_map::<usize, usize>(|map1| {
        with_map::<usize, usize>(|map2| {
            {
                map1.pin().insert(1, 0);
                map2.pin().insert(1, 1);
            }

            assert_ne!(map1, map2);
            assert_ne!(map2, map1);
        });
    });
}

#[test]
fn clone_map_empty() {
    with_map::<&'static str, u32>(|map| {
        let cloned_map = map.clone();
        assert_eq!(map.len(), cloned_map.len());
        assert_eq!(&map, &cloned_map);
        assert_eq!(cloned_map.len(), 0);
    });
}

#[test]
// Test that same values exists in both maps (original and cloned)
fn clone_map_filled() {
    with_map::<&'static str, u32>(|map| {
        map.insert("FooKey", 0, &map.guard());
        map.insert("BarKey", 10, &map.guard());
        let cloned_map = map.clone();
        assert_eq!(map.len(), cloned_map.len());
        assert_eq!(&map, &cloned_map);

        // test that we are not mapping the same tables
        map.insert("NewItem", 100, &map.guard());
        assert_ne!(&map, &cloned_map);
    });
}

#[test]
fn default() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        map.insert(42, 0, &guard);

        assert_eq!(map.get(&42, &guard), Some(&0));
    });
}

#[test]
fn debug() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();
        map.insert(42, 0, &guard);
        map.insert(16, 8, &guard);

        let formatted = format!("{:?}", map);

        assert!(formatted == "{42: 0, 16: 8}" || formatted == "{16: 8, 42: 0}");
    });
}

#[test]
fn extend() {
    with_map::<usize, usize>(|map| {
        let guard = map.guard();

        let mut entries: Vec<(usize, usize)> = vec![(42, 0), (16, 6), (38, 42)];
        entries.sort_unstable();

        (&map).extend(entries.clone().into_iter());

        let mut collected: Vec<(usize, usize)> = map
            .iter(&guard)
            .map(|(key, value)| (*key, *value))
            .collect();
        collected.sort_unstable();

        assert_eq!(entries, collected);
    });
}

#[test]
fn extend_ref() {
    with_map::<usize, usize>(|map| {
        let mut entries: Vec<(&usize, &usize)> = vec![(&42, &0), (&16, &6), (&38, &42)];
        entries.sort();

        (&map).extend(entries.clone().into_iter());

        let guard = map.guard();
        let mut collected: Vec<(&usize, &usize)> = map.iter(&guard).collect();
        collected.sort();

        assert_eq!(entries, collected);
    });
}

#[test]
fn from_iter_empty() {
    use std::iter::FromIterator;

    let entries: Vec<(usize, usize)> = Vec::new();
    let map: HashMap<usize, usize> = HashMap::from_iter(entries.into_iter());

    assert_eq!(map.len(), 0)
}

#[test]
fn len() {
    with_map::<usize, usize>(|map| {
        let len = if cfg!(miri) { 100 } else { 10_000 };
        for i in 0..len {
            map.pin().insert(i, i + 1);
        }
        assert_eq!(map.pin().len(), len);
    });
}

#[test]
fn iter() {
    with_map::<usize, usize>(|map| {
        let len = if cfg!(miri) { 100 } else { 10_000 };
        for i in 0..len {
            assert_eq!(map.pin().insert(i, i + 1), None);
        }

        let v: Vec<_> = (0..len).map(|i| (i, i + 1)).collect();
        let mut got: Vec<_> = map.pin().iter().map(|(&k, &v)| (k, v)).collect();
        got.sort();
        assert_eq!(v, got);
    });
}

#[test]
fn mixed() {
    const LEN: usize = if cfg!(miri) { 48 } else { 1024 };
    with_map::<usize, usize>(|map| {
        assert!(map.pin().get(&100).is_none());
        map.pin().insert(100, 101);
        assert_eq!(map.pin().get(&100), Some(&101));
        map.pin().update(100, |x| x + 2);
        assert_eq!(map.pin().get(&100), Some(&103));

        assert!(map.pin().get(&200).is_none());
        map.pin().insert(200, 202);
        assert_eq!(map.pin().get(&200), Some(&202));

        assert!(map.pin().get(&300).is_none());

        assert_eq!(map.pin().remove(&100), Some(&103));
        assert_eq!(map.pin().remove(&200), Some(&202));
        assert!(map.pin().remove(&300).is_none());

        assert!(map.pin().get(&100).is_none());
        assert!(map.pin().get(&200).is_none());
        assert!(map.pin().get(&300).is_none());

        for i in 0..LEN {
            assert_eq!(map.pin().insert(i, i + 1), None);
        }

        for i in 0..LEN {
            assert_eq!(map.pin().get(&i), Some(&(i + 1)));
        }

        for i in 0..LEN {
            assert_eq!(map.pin().update(i, |i| i - 1), Some(&i));
        }

        for i in 0..LEN {
            assert_eq!(map.pin().get(&i), Some(&i));
        }

        for i in 0..LEN {
            assert_eq!(map.pin().remove(&i), Some(&i));
        }

        for i in 0..LEN {
            assert_eq!(map.pin().get(&i), None);
        }

        for i in 0..(LEN * 2) {
            assert_eq!(map.pin().insert(i, i + 1), None);
        }

        for i in 0..(LEN * 2) {
            assert_eq!(map.pin().get(&i), Some(&(i + 1)));
        }
    });
}

// run tests with hashers that create unrealistically long probe sequences
mod hasher {
    use super::*;

    fn check<S: BuildHasher + Default>() {
        let range = if cfg!(miri) { 0..16 } else { 0..100 };

        with_map::<i32, i32>(|map| {
            let guard = map.guard();
            for i in range.clone() {
                map.insert(i, i, &guard);
            }

            assert!(!map.contains_key(&i32::min_value(), &guard));
            assert!(!map.contains_key(&(range.start - 1), &guard));
            for i in range.clone() {
                assert!(map.contains_key(&i, &guard));
            }
            assert!(!map.contains_key(&range.end, &guard));
            assert!(!map.contains_key(&i32::max_value(), &guard));
        });
    }

    #[test]
    fn test_zero_hasher() {
        #[derive(Default)]
        pub struct ZeroHasher;

        impl Hasher for ZeroHasher {
            fn finish(&self) -> u64 {
                0
            }

            fn write(&mut self, _: &[u8]) {}
        }

        check::<BuildHasherDefault<ZeroHasher>>();
    }

    #[test]
    fn test_max_hasher() {
        #[derive(Default)]
        struct MaxHasher;

        impl Hasher for MaxHasher {
            fn finish(&self) -> u64 {
                u64::max_value()
            }

            fn write(&mut self, _: &[u8]) {}
        }

        check::<BuildHasherDefault<MaxHasher>>();
    }
}
