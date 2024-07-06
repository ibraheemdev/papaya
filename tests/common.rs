#![allow(dead_code)]

use papaya::{HashMap, ResizeMode};

// Run the test on different configurations of a `HashMap`.
pub fn with_map<K, V>(mut test: impl FnMut(&dyn Fn() -> HashMap<K, V>)) {
    // Blocking resize mode.
    if !crate::resize_stress!() {
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

// Prints a log message if `RUST_LOG=debug` is set.
#[macro_export]
macro_rules! debug {
    ($($x:tt)*) => {
        if std::env::var("RUST_LOG").as_deref() == Ok("debug") {
            println!($($x)*);
        }
    };
}

// Returns a `bool` indicating whether resize stress is enabled.
//
// If this is true, linearizable operations such as iteration cannot be
// performed and will block indefinitely.
#[macro_export]
macro_rules! resize_stress {
    () => {
        option_env!("PAPAYA_RESIZE_STRESS").is_some()
    };
}

// Returns the number of threads to use for stress testing.
pub fn threads() -> usize {
    if cfg!(miri) {
        2
    } else {
        std::thread::available_parallelism().unwrap().get() / 2
    }
}
