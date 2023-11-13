#![allow(dead_code, unused, unstable_name_collisions)]

mod map;
mod raw;
mod seize;

pub use map::HashMap;

#[test]
fn foo() {
    let map = HashMap::new();

    assert!(map.pin().get(&100).is_none());
    map.pin().insert(100, "foo");
    assert_eq!(map.pin().get(&100), Some(&"foo"));

    assert!(map.pin().get(&200).is_none());
    map.pin().insert(200, "bar");
    assert_eq!(map.pin().get(&200), Some(&"bar"));

    assert!(map.pin().get(&300).is_none());

    assert_eq!(map.pin().remove(&100), Some(&"foo"));
    assert_eq!(map.pin().remove(&200), Some(&"bar"));
    assert!(map.pin().remove(&300).is_none());

    assert!(map.pin().get(&100).is_none());
    assert!(map.pin().get(&200).is_none());
    assert!(map.pin().get(&300).is_none());
}
