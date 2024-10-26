#![allow(dead_code)]

use papaya::{HashMap, HashSet, ResizeMode};

// Run the test on different configurations of a `HashMap`.
pub fn with_map<K, V>(mut test: impl FnMut(&dyn Fn() -> HashMap<K, V>)) {
    // Blocking resize mode.
    if !cfg!(papaya_stress) {
        test(&(|| HashMap::builder().resize_mode(ResizeMode::Blocking).build()));
    }

    // Incremental resize mode with a small chunk to stress operations on nested tables.
    test(
        &(|| {
            HashMap::builder()
                .resize_mode(ResizeMode::Incremental(1))
                .build()
        }),
    );

    // Incremental resize mode with a medium-sized chunk to promote interference with incremental
    // resizing.
    test(
        &(|| {
            HashMap::builder()
                .resize_mode(ResizeMode::Incremental(128))
                .build()
        }),
    );
}

// Run the test on different configurations of a `HashSet`.
pub fn with_set<K>(mut test: impl FnMut(&dyn Fn() -> HashSet<K>)) {
    // Blocking resize mode.
    if !cfg!(papaya_stress) {
        test(&(|| HashSet::builder().resize_mode(ResizeMode::Blocking).build()));
    }

    // Incremental resize mode with a small chunk to stress operations on nested tables.
    test(
        &(|| {
            HashSet::builder()
                .resize_mode(ResizeMode::Incremental(1))
                .build()
        }),
    );

    // Incremental resize mode with a medium-sized chunk to promote interference with incremental
    // resizing.
    test(
        &(|| {
            HashSet::builder()
                .resize_mode(ResizeMode::Incremental(128))
                .build()
        }),
    );
}

// Prints a log message if `RUST_LOG=debug` is set.
#[macro_export]
macro_rules! debug {
    ($($x:tt)*) => {
        if std::env::var("RUST_LOG").as_deref() == Ok("debug") {
            println!($($x)*);
        }
    };
}

// Returns the number of threads to use for stress testing.
pub fn threads() -> usize {
    if cfg!(miri) {
        2
    } else {
        num_cpus::get_physical().next_power_of_two()
    }
}
