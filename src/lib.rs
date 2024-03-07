#![allow(unstable_name_collisions)]

mod map;
mod raw;
mod seize;

pub use map::HashMap;

#[test]
fn bench() {
    let map = HashMap::new();
    for i in 0..1000 {
        assert_eq!(map.pin().insert(i, i + 1), None);
    }

    let y = map.pin();
    for i in 0..1000 {
        assert_eq!(y.get(&i), Some(&(i + 1)));
    }

    let v: Vec<_> = (0..1000).map(|i| (i, i + 1)).collect();
    let mut got: Vec<_> = map.pin().iter().map(|(&k, &v)| (k, v)).collect();
    got.sort();
    assert_eq!(v, got);
}

#[test]
fn basic() {
    let map = HashMap::new();

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

    for i in 0..64 {
        assert_eq!(map.pin().insert(i, i + 1), None);
    }

    for i in 0..64 {
        assert_eq!(map.pin().get(&i), Some(&(i + 1)));
    }

    for i in 0..64 {
        assert_eq!(map.pin().update(i, |i| i - 1), Some(&(i + 1)));
    }

    for i in 0..64 {
        assert_eq!(map.pin().get(&i), Some(&i));
    }

    for i in 0..64 {
        assert_eq!(map.pin().remove(&i), Some(&i));
    }

    for i in 0..64 {
        assert_eq!(map.pin().get(&i), None);
    }

    for i in 0..256 {
        assert_eq!(map.pin().insert(i, i + 1), None);
    }

    for i in 0..256 {
        assert_eq!(map.pin().get(&i), Some(&(i + 1)));
    }

    dbg!(map.pin().capacity());
}

#[test]
fn stress() {
    let map = HashMap::<usize, usize>::new();
    let chunk = 1 << 8;

    std::thread::scope(|s| {
        for t in 0..16 {
            let map = &map;
            s.spawn(move || {
                let (start, end) = (chunk * t, chunk * (t + 1));

                for i in start..end {
                    assert_eq!(map.pin().insert(i, i + 1), None);
                }

                for i in start..end {
                    assert_eq!(map.pin().get(&i), Some(&(i + 1)));
                }

                for i in start..end {
                    assert_eq!(map.pin().remove(&i), Some(&(i + 1)));
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
                    assert!(k < chunk * 16);
                    assert_eq!(v, k + 1);
                }
            });
        }
    });

    let v: Vec<_> = (0..chunk * 16).map(|i| (i, i + 1)).collect();
    let mut got: Vec<_> = map.pin().iter().map(|(&k, &v)| (k, v)).collect();
    got.sort();
    assert_eq!(v, got);
}
