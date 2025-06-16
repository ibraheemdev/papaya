#![cfg(feature = "rayon")]

use rayon::prelude::*;

mod common;
use common::{with_map, with_set};

#[test]
fn hashmap_par_iter() {
    if cfg!(papaya_stress) {
        return;
    }
    with_map::<usize, usize>(|map| {
        let map = map();
        let len = if cfg!(miri) { 100 } else { 10_000 };
        for i in 0..len {
            map.pin_owned().insert(i, i + 1);
        }

        let mut expected: Vec<_> = (0..len).map(|i| (i, i + 1)).collect();

        let mut got: Vec<_> = map
            .pin_owned()
            .par_iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        got.sort();
        expected.sort();
        assert_eq!(expected, got);

        let mut via_trait: Vec<_> = (&map.pin_owned())
            .into_par_iter()
            .map(|(&k, &v)| (k, v))
            .collect();
        via_trait.sort();
        assert_eq!(expected, via_trait);
    });
}

#[test]
fn hashset_par_iter() {
    if cfg!(papaya_stress) {
        return;
    }
    with_set::<usize>(|set| {
        let set = set();
        let len = if cfg!(miri) { 100 } else { 10_000 };
        for i in 0..len {
            set.pin_owned().insert(i);
        }

        let mut expected: Vec<_> = (0..len).collect();

        let mut got: Vec<_> = set
            .pin_owned()
            .par_iter()
            .copied()
            .collect();
        got.sort();
        expected.sort();
        assert_eq!(expected, got);

        let mut via_trait: Vec<_> = (&set.pin_owned()).into_par_iter().copied().collect();
        via_trait.sort();
        assert_eq!(expected, via_trait);
    });
}

