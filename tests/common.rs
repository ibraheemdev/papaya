use papaya::{HashMap, ResizeMode};

// Run the test on different configurations of a `HashMap`.
pub fn with_map<K, V>(mut test: impl FnMut(&dyn Fn() -> HashMap<K, V>)) {
    // Blocking resize mode.
    test(&(|| HashMap::builder().resize_mode(ResizeMode::Blocking).build()));

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
                .resize_mode(ResizeMode::Incremental(256))
                .build()
        }),
    );
}
