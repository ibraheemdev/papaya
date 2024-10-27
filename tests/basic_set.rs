// Adapted from: https://github.com/jonhoo/flurry/blob/main/tests/basic.rs

use papaya::HashSet;

use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
use std::sync::Arc;

mod common;
use common::with_set;

#[test]
fn new() {
    with_set::<usize>(|set| drop(set()));
}

#[test]
fn clear() {
    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        {
            set.insert(0, &guard);
            set.insert(1, &guard);
            set.insert(2, &guard);
            set.insert(3, &guard);
            set.insert(4, &guard);
        }
        set.clear(&guard);
        assert!(set.is_empty());
    });
}

#[test]
fn insert() {
    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        assert_eq!(set.insert(42, &guard), true);
        assert_eq!(set.insert(42, &guard), false);
        assert_eq!(set.len(), 1);
    });
}

#[test]
fn get_empty() {
    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        let e = set.get(&42, &guard);
        assert!(e.is_none());
    });
}

#[test]
fn remove_empty() {
    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        assert_eq!(set.remove(&42, &guard), false);
    });
}

#[test]
fn insert_and_remove() {
    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        assert!(set.insert(42, &guard));
        assert!(set.remove(&42, &guard));
        assert!(set.get(&42, &guard).is_none());
    });
}

#[test]
fn insert_and_get() {
    with_set::<usize>(|set| {
        let set = set();
        set.insert(42, &set.guard());

        {
            let guard = set.guard();
            let e = set.get(&42, &guard).unwrap();
            assert_eq!(e, &42);
        }
    });
}

#[test]
fn reinsert() {
    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        assert!(set.insert(42, &guard));
        assert!(!set.insert(42, &guard));
        {
            let guard = set.guard();
            let e = set.get(&42, &guard).unwrap();
            assert_eq!(e, &42);
        }
    });
}

#[test]
fn concurrent_insert() {
    with_set::<usize>(|set| {
        let set = set();
        let set = Arc::new(set);

        let set1 = set.clone();
        let t1 = std::thread::spawn(move || {
            for i in 0..64 {
                set1.insert(i, &set1.guard());
            }
        });
        let set2 = set.clone();
        let t2 = std::thread::spawn(move || {
            for i in 0..64 {
                set2.insert(i, &set2.guard());
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        let guard = set.guard();
        for i in 0..64 {
            let v = set.get(&i, &guard).unwrap();
            assert!(v == &i);
        }
    });
}

#[test]
fn concurrent_remove() {
    with_set::<usize>(|set| {
        let set = set();
        let set = Arc::new(set);

        {
            let guard = set.guard();
            for i in 0..64 {
                set.insert(i, &guard);
            }
        }

        let set1 = set.clone();
        let t1 = std::thread::spawn(move || {
            let guard = set1.guard();
            for i in 0..64 {
                set1.remove(&i, &guard);
            }
        });
        let set2 = set.clone();
        let t2 = std::thread::spawn(move || {
            let guard = set2.guard();
            for i in 0..64 {
                set2.remove(&i, &guard);
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // after joining the threads, the set should be empty
        let guard = set.guard();
        for i in 0..64 {
            assert!(set.get(&i, &guard).is_none());
        }
    });
}

#[test]
#[cfg(not(miri))]
fn concurrent_resize_and_get() {
    if cfg!(papaya_stress) {
        return;
    }

    with_set::<usize>(|set| {
        let set = set();
        let set = Arc::new(set);

        {
            let guard = set.guard();
            for i in 0..1024 {
                set.insert(i, &guard);
            }
        }

        let set1 = set.clone();
        // t1 is using reserve to trigger a bunch of resizes
        let t1 = std::thread::spawn(move || {
            let guard = set1.guard();
            // there should be 2 ** 10 capacity already, so trigger additional resizes
            for power in 11..16 {
                set1.reserve(1 << power, &guard);
            }
        });
        let set2 = set.clone();
        // t2 is retrieving existing keys a lot, attempting to encounter a BinEntry::Moved
        let t2 = std::thread::spawn(move || {
            let guard = set2.guard();
            for _ in 0..32 {
                for i in 0..1024 {
                    let v = set2.get(&i, &guard).unwrap();
                    assert_eq!(v, &i);
                }
            }
        });

        t1.join().unwrap();
        t2.join().unwrap();

        // make sure all the entries still exist after all the resizes
        {
            let guard = set.guard();

            for i in 0..1024 {
                let v = set.get(&i, &guard).unwrap();
                assert_eq!(v, &i);
            }
        }
    });
}

#[test]
fn current_kv_dropped() {
    let dropped1 = Arc::new(0);

    with_set::<Arc<usize>>(|set| {
        let set = set();
        set.insert(dropped1.clone(), &set.guard());
        assert_eq!(Arc::strong_count(&dropped1), 2);

        drop(set);

        // dropping the set should immediately drop (not deferred) all keys and values
        assert_eq!(Arc::strong_count(&dropped1), 1);
    });
}

#[test]
fn empty_sets_equal() {
    with_set::<usize>(|set1| {
        let set1 = set1();
        with_set::<usize>(|set2| {
            let set2 = set2();
            assert_eq!(set1, set2);
            assert_eq!(set2, set1);
        });
    });
}

#[test]
fn different_size_sets_not_equal() {
    with_set::<usize>(|set1| {
        let set1 = set1();
        with_set::<usize>(|set2| {
            let set2 = set2();
            {
                let guard1 = set1.guard();
                let guard2 = set2.guard();

                set1.insert(1, &guard1);
                set1.insert(2, &guard1);
                set1.insert(3, &guard1);

                set2.insert(1, &guard2);
                set2.insert(2, &guard2);
            }

            assert_ne!(set1, set2);
            assert_ne!(set2, set1);
        });
    });
}

#[test]
fn same_values_equal() {
    with_set::<usize>(|set1| {
        let set1 = set1();
        with_set::<usize>(|set2| {
            let set2 = set2();
            {
                set1.pin().insert(1);
                set2.pin().insert(1);
            }

            assert_eq!(set1, set2);
            assert_eq!(set2, set1);
        });
    });
}

#[test]
fn different_values_not_equal() {
    with_set::<usize>(|set1| {
        let set1 = set1();
        with_set::<usize>(|set2| {
            let set2 = set2();
            {
                set1.pin().insert(1);
                set2.pin().insert(2);
            }

            assert_ne!(set1, set2);
            assert_ne!(set2, set1);
        });
    });
}

#[test]
fn clone_set_empty() {
    with_set::<&'static str>(|set| {
        let set = set();
        let cloned_set = set.clone();
        assert_eq!(set.len(), cloned_set.len());
        assert_eq!(&set, &cloned_set);
        assert_eq!(cloned_set.len(), 0);
    });
}

#[test]
// Test that same values exists in both sets (original and cloned)
fn clone_set_filled() {
    with_set::<&'static str>(|set| {
        let set = set();
        set.insert("FooKey", &set.guard());
        set.insert("BarKey", &set.guard());
        let cloned_set = set.clone();
        assert_eq!(set.len(), cloned_set.len());
        assert_eq!(&set, &cloned_set);

        // test that we are not setting the same tables
        set.insert("NewItem", &set.guard());
        assert_ne!(&set, &cloned_set);
    });
}

#[test]
fn default() {
    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        set.insert(42, &guard);

        assert_eq!(set.get(&42, &guard), Some(&42));
    });
}

#[test]
fn debug() {
    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();
        set.insert(42, &guard);
        set.insert(16, &guard);

        let formatted = format!("{:?}", set);

        assert!(formatted == "{42, 16}" || formatted == "{16, 42}");
    });
}

#[test]
fn extend() {
    if cfg!(papaya_stress) {
        return;
    }

    with_set::<usize>(|set| {
        let set = set();
        let guard = set.guard();

        let mut entries: Vec<usize> = vec![42, 16, 38];
        entries.sort_unstable();

        (&set).extend(entries.clone().into_iter());

        let mut collected: Vec<usize> = set.iter(&guard).map(|key| *key).collect();
        collected.sort_unstable();

        assert_eq!(entries, collected);
    });
}

#[test]
fn extend_ref() {
    if cfg!(papaya_stress) {
        return;
    }

    with_set::<usize>(|set| {
        let set = set();
        let mut entries: Vec<&usize> = vec![&42, &36, &18];
        entries.sort();

        (&set).extend(entries.clone().into_iter());

        let guard = set.guard();
        let mut collected: Vec<&usize> = set.iter(&guard).collect();
        collected.sort();

        assert_eq!(entries, collected);
    });
}

#[test]
fn from_iter_empty() {
    use std::iter::FromIterator;

    let entries: Vec<usize> = Vec::new();
    let set: HashSet<usize> = HashSet::from_iter(entries.into_iter());

    assert_eq!(set.len(), 0)
}

#[test]
fn from_iter_repeated() {
    use std::iter::FromIterator;

    let entries = vec![0, 0, 0];
    let set: HashSet<_> = HashSet::from_iter(entries.into_iter());
    let set = set.pin();
    assert_eq!(set.len(), 1);
    assert_eq!(set.iter().collect::<Vec<_>>(), vec![&0])
}

#[test]
fn len() {
    with_set::<usize>(|set| {
        let set = set();
        let len = if cfg!(miri) { 100 } else { 10_000 };
        for i in 0..len {
            set.pin().insert(i);
        }
        assert_eq!(set.len(), len);
    });
}

#[test]
fn iter() {
    if cfg!(papaya_stress) {
        return;
    }

    with_set::<usize>(|set| {
        let set = set();
        let len = if cfg!(miri) { 100 } else { 10_000 };
        for i in 0..len {
            assert_eq!(set.pin().insert(i), true);
        }

        let v: Vec<_> = (0..len).collect();
        let mut got: Vec<_> = set.pin().iter().map(|&k| k).collect();
        got.sort();
        assert_eq!(v, got);
    });
}

#[test]
fn retain_empty() {
    with_set::<usize>(|set| {
        let set = set();
        set.pin().retain(|_| false);
        assert_eq!(set.len(), 0);
    });
}

#[test]
fn retain_all_false() {
    with_set::<usize>(|set| {
        let set = set();
        for i in 0..10 {
            set.pin().insert(i);
        }
        set.pin().retain(|_| false);
        assert_eq!(set.len(), 0);
    });
}

#[test]
fn retain_all_true() {
    with_set::<usize>(|set| {
        let set = set();
        for i in 0..10 {
            set.pin().insert(i);
        }
        set.pin().retain(|_| true);
        assert_eq!(set.len(), 10);
    });
}

#[test]
fn retain_some() {
    with_set::<usize>(|set| {
        let set = set();
        for i in 0..10 {
            set.pin().insert(i);
        }
        set.pin().retain(|&k| k >= 5);
        assert_eq!(set.len(), 5);
        let mut got: Vec<_> = set.pin().iter().copied().collect();
        got.sort();
        assert_eq!(got, [5, 6, 7, 8, 9]);
    });
}

#[test]
fn mixed() {
    const LEN: usize = if cfg!(miri) { 48 } else { 1024 };
    with_set::<usize>(|set| {
        let set = set();
        assert!(set.pin().get(&100).is_none());
        set.pin().insert(100);
        assert_eq!(set.pin().get(&100), Some(&100));

        assert!(set.pin().get(&200).is_none());
        set.pin().insert(200);
        assert_eq!(set.pin().get(&200), Some(&200));

        assert!(set.pin().get(&300).is_none());

        assert_eq!(set.pin().remove(&100), true);
        assert_eq!(set.pin().remove(&200), true);
        assert_eq!(set.pin().remove(&300), false);

        assert!(set.pin().get(&100).is_none());
        assert!(set.pin().get(&200).is_none());
        assert!(set.pin().get(&300).is_none());

        for i in 0..LEN {
            assert_eq!(set.pin().insert(i), true);
        }

        for i in 0..LEN {
            assert_eq!(set.pin().get(&i), Some(&i));
        }

        for i in 0..LEN {
            assert_eq!(set.pin().remove(&i), true);
        }

        for i in 0..LEN {
            assert_eq!(set.pin().get(&i), None);
        }

        for i in 0..(LEN * 2) {
            assert_eq!(set.pin().insert(i), true);
        }

        for i in 0..(LEN * 2) {
            assert_eq!(set.pin().get(&i), Some(&i));
        }
    });
}

// run tests with hashers that create unrealistically long probe sequences
mod hasher {
    use super::*;

    fn check<S: BuildHasher + Default>() {
        let range = if cfg!(miri) { 0..16 } else { 0..100 };

        with_set::<i32>(|set| {
            let set = set();
            let guard = set.guard();
            for i in range.clone() {
                set.insert(i, &guard);
            }

            assert!(!set.contains(&i32::min_value(), &guard));
            assert!(!set.contains(&(range.start - 1), &guard));
            for i in range.clone() {
                assert!(set.contains(&i, &guard));
            }
            assert!(!set.contains(&range.end, &guard));
            assert!(!set.contains(&i32::max_value(), &guard));
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
