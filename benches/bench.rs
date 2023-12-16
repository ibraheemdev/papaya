// TODO: test u64 no simd
use std::collections::HashMap;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

const SIZE: usize = 10_000;

fn compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compare");

    // A random key iterator.
    #[derive(Clone, Copy)]
    struct RandomKeys {
        state: usize,
    }

    impl RandomKeys {
        fn new() -> Self {
            RandomKeys { state: 0 }
        }
    }

    impl Iterator for RandomKeys {
        type Item = usize;
        fn next(&mut self) -> Option<usize> {
            // Add 1 then multiply by some 32 bit prime.
            self.state = self.state.wrapping_add(1).wrapping_mul(3_787_392_781);
            Some(self.state)
        }
    }

    group.bench_function("wip-table-group-8", |b| {
        let m = the_one::HashMap::<usize, usize>::default();
        let m = m.pin();
        for i in RandomKeys::new().take(SIZE) {
            m.insert(i, i);
        }

        b.iter(|| {
            for i in RandomKeys::new().take(SIZE) {
                black_box(assert_eq!(m.get_8(&i), Some(&i)));
            }
        });
    });

    group.bench_function("std", |b| {
        let mut m = HashMap::<usize, usize>::default();
        for i in RandomKeys::new().take(SIZE) {
            m.insert(i, i);
        }

        b.iter(|| {
            for i in RandomKeys::new().take(SIZE) {
                black_box(assert_eq!(m.get(&i), Some(&i)));
            }
        });
    });

    group.bench_function("wip-table-simd-16", |b| {
        let m = the_one::HashMap::<usize, usize>::default();
        let m = m.pin();
        for i in RandomKeys::new().take(SIZE) {
            m.insert(i, i);
        }

        b.iter(|| {
            for i in RandomKeys::new().take(SIZE) {
                black_box(assert_eq!(m.get(&i), Some(&i)));
            }
        });
    });

    group.bench_function("wip-table-nosimd", |b| {
        let m = the_one::HashMap::<usize, usize>::default();
        let m = m.pin();
        for i in RandomKeys::new().take(SIZE) {
            m.insert(i, i);
        }

        b.iter(|| {
            for i in RandomKeys::new().take(SIZE) {
                black_box(assert_eq!(m.get_nosimd(&i), Some(&i)));
            }
        });
    });

    group.bench_function("dashmap", |b| {
        let m = dashmap::DashMap::<usize, usize>::default();
        for i in RandomKeys::new().take(SIZE) {
            m.insert(i, i);
        }

        b.iter(|| {
            for i in RandomKeys::new().take(SIZE) {
                black_box(assert_eq!(*m.get(&i).unwrap(), i));
            }
        });
    });

    group.finish();
}

criterion_group!(benches, compare);
criterion_main!(benches);
