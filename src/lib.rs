#![allow(dead_code, unused, unstable_name_collisions)]

mod map;
mod raw;
mod seize;

pub use map::HashMap;

#[test]
fn bench() {
    let map = HashMap::new();
    for i in 0..1228 {
        assert_eq!(map.pin().insert(i, i + 1), None);
    }

    let y = map.pin();
    let x = std::time::Instant::now();
    for i in 0..1228 {
        assert_eq!(y.get(&i), Some(&(i + 1)));
    }

    println!("{}", x.elapsed().as_nanos());
}

#[test]
fn basic() {
    let map = HashMap::new();

    assert!(map.pin().get(&100).is_none());
    map.pin().insert(100, 101);
    assert_eq!(map.pin().get(&100), Some(&101));

    assert!(map.pin().get(&200).is_none());
    map.pin().insert(200, 202);
    assert_eq!(map.pin().get(&200), Some(&202));

    assert!(map.pin().get(&300).is_none());

    assert_eq!(map.pin().remove(&100), Some(&101));
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
        assert_eq!(map.pin().remove(&i), Some(&(i + 1)));
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
}

#[test]
fn foo() {
    let map = HashMap::new();

    std::thread::scope(|s| {
        for x in 0..16 {
            let map = &map;
            s.spawn(move || {
                let (start, end) = (8192 * x, 8192 * (x + 1));

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
            });
        }
    });
}
