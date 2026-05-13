use std::collections::HashMap;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use parking_lot::Mutex;

trait Atomic<T> {
    fn load(&self, ordering: Ordering) -> T;
}

impl<T> Atomic<*mut T> for AtomicPtr<T> {
    fn load(&self, ordering: Ordering) -> *mut T {
        self.load(ordering)
    }
}

impl Atomic<u8> for AtomicU8 {
    fn load(&self, ordering: Ordering) -> u8 {
        self.load(ordering)
    }
}

trait ParkApi: Default + Send + Sync + 'static {
    fn park<T>(&self, atomic: &impl Atomic<T>, should_park: impl Fn(T) -> bool);
    fn unpark<T>(&self, atomic: &impl Atomic<T>);
}

#[derive(Default)]
struct ThreadParker {
    pending: AtomicUsize,
    state: Mutex<ThreadState>,
}

#[derive(Default)]
struct ThreadState {
    count: u64,
    threads: HashMap<usize, HashMap<u64, Thread>>,
}

impl ParkApi for ThreadParker {
    fn park<T>(&self, atomic: &impl Atomic<T>, should_park: impl Fn(T) -> bool) {
        let key = atomic as *const _ as usize;

        loop {
            self.pending.fetch_add(1, Ordering::SeqCst);

            let id = {
                let mut state = self.state.lock();
                state.count += 1;

                let current_count = state.count;
                let threads = state.threads.entry(key).or_default();
                threads.insert(current_count, thread::current());

                state.count
            };

            if !should_park(atomic.load(Ordering::SeqCst)) {
                let thread = {
                    let mut state = self.state.lock();
                    state
                        .threads
                        .get_mut(&key)
                        .and_then(|threads| threads.remove(&id))
                };

                if thread.is_some() {
                    self.pending.fetch_sub(1, Ordering::Relaxed);
                }

                return;
            }

            loop {
                thread::park();

                let mut state = self.state.lock();
                if !state
                    .threads
                    .get_mut(&key)
                    .is_some_and(|threads: &mut HashMap<u64, Thread>| threads.contains_key(&id))
                {
                    break;
                }
            }

            if !should_park(atomic.load(Ordering::Acquire)) {
                return;
            }
        }
    }

    fn unpark<T>(&self, atomic: &impl Atomic<T>) {
        let key = atomic as *const _ as usize;

        if self.pending.load(Ordering::SeqCst) == 0 {
            return;
        }

        let threads = {
            let mut state = self.state.lock();
            state.threads.remove(&key)
        };

        if let Some(threads) = threads {
            self.pending.fetch_sub(threads.len(), Ordering::Relaxed);

            for (_, thread) in threads {
                thread.unpark();
            }
        }
    }
}

#[derive(Default)]
struct ParkingLotParker {
    pending: AtomicUsize,
}

#[derive(Default)]
struct PendingFastPath {
    pending: AtomicUsize,
}

impl PendingFastPath {
    #[inline]
    fn unpark<T>(&self, atomic: &impl Atomic<T>) {
        if self.pending.load(Ordering::SeqCst) == 0 {
            return;
        }

        black_box(atomic as *const _ as usize);
    }
}

impl ParkApi for ParkingLotParker {
    fn park<T>(&self, atomic: &impl Atomic<T>, should_park: impl Fn(T) -> bool) {
        let key = atomic as *const _ as usize;

        loop {
            self.pending.fetch_add(1, Ordering::SeqCst);

            let result = unsafe {
                parking_lot_core::park(
                    key,
                    || should_park(atomic.load(Ordering::SeqCst)),
                    || {},
                    |_, _| {},
                    parking_lot_core::DEFAULT_PARK_TOKEN,
                    None,
                )
            };

            match result {
                parking_lot_core::ParkResult::Invalid => {
                    self.pending.fetch_sub(1, Ordering::Relaxed);
                    return;
                }
                parking_lot_core::ParkResult::Unparked(_) => {
                    if !should_park(atomic.load(Ordering::Acquire)) {
                        return;
                    }
                }
                parking_lot_core::ParkResult::TimedOut => unreachable!(),
            }
        }
    }

    fn unpark<T>(&self, atomic: &impl Atomic<T>) {
        if self.pending.load(Ordering::SeqCst) == 0 {
            return;
        }

        self.unpark_slow(atomic);
    }
}

impl ParkingLotParker {
    #[cold]
    #[inline(never)]
    fn unpark_slow<T>(&self, atomic: &impl Atomic<T>) {
        let key = atomic as *const _ as usize;

        unsafe {
            let unparked =
                parking_lot_core::unpark_all(key, parking_lot_core::DEFAULT_UNPARK_TOKEN);
            self.pending.fetch_sub(unparked, Ordering::Relaxed);
        }
    }
}

fn ping_pong<P: ParkApi>(iters: u64) -> Duration {
    let parker = Arc::new(P::default());
    let state = Arc::new(AtomicU8::new(0));
    let ready = Arc::new(Barrier::new(2));

    let worker = {
        let parker = parker.clone();
        let state = state.clone();
        let ready = ready.clone();
        thread::spawn(move || {
            ready.wait();

            for _ in 0..iters {
                parker.park(&*state, |state| state == 0);
                state.store(0, Ordering::SeqCst);
                parker.unpark(&*state);
            }
        })
    };

    ready.wait();

    let start = Instant::now();
    for _ in 0..iters {
        state.store(1, Ordering::SeqCst);
        parker.unpark(&*state);
        parker.park(&*state, |state| state == 1);
    }
    let elapsed = start.elapsed();

    worker.join().unwrap();
    elapsed
}

fn compare(c: &mut Criterion) {
    let mut group = c.benchmark_group("parker");

    group.bench_function("old_thread_unpark_no_waiters", |b| {
        let parker = ThreadParker::default();
        let state = AtomicU8::new(0);

        b.iter(|| black_box(&parker).unpark(black_box(&state)));
    });

    group.bench_function("pending_fast_path_no_waiters", |b| {
        let parker = PendingFastPath::default();
        let state = AtomicU8::new(0);

        b.iter(|| black_box(&parker).unpark(black_box(&state)));
    });

    group.bench_function("parking_lot_unpark_no_waiters", |b| {
        let parker = ParkingLotParker::default();
        let state = AtomicU8::new(0);

        b.iter(|| black_box(&parker).unpark(black_box(&state)));
    });

    group.bench_function("old_thread_ping_pong", |b| {
        b.iter_custom(ping_pong::<ThreadParker>);
    });

    group.bench_function("parking_lot_ping_pong", |b| {
        b.iter_custom(ping_pong::<ParkingLotParker>);
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(2));
    targets = compare
}
criterion_main!(benches);
