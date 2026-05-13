use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};

// A simple thread parker.
//
#[derive(Default)]
pub struct Parker {
    pending: AtomicUsize,
}

impl Parker {
    // Block the current thread until the park condition is false.
    //
    // This method is guaranteed to establish happens-before with the unpark condition
    // before it returns.
    pub fn park<T>(&self, atomic: &impl Atomic<T>, should_park: impl Fn(T) -> bool) {
        let key = atomic as *const _ as usize;

        loop {
            // Announce our thread before validating the park condition so unparks
            // don't miss waiters racing to enter the parking lot queue.
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
                    // Ensure we were unparked for the correct reason.
                    //
                    // Establish happens-before with the unpark condition.
                    if !should_park(atomic.load(Ordering::Acquire)) {
                        return;
                    }
                }
                parking_lot_core::ParkResult::TimedOut => unreachable!(),
            }
        }
    }

    // Unpark all threads waiting on the given atomic.
    //
    // Note that any modifications must be `SeqCst` to be visible to unparked threads.
    pub fn unpark<T>(&self, atomic: &impl Atomic<T>) {
        // Fast-path, no one waiting to be unparked.
        //
        // Note that `SeqCst` is necessary here to participate in the total order
        // established between the increment of `self.pending` in `park` and the
        // `SeqCst` store of the unpark condition by the caller.
        if self.pending.load(Ordering::SeqCst) == 0 {
            return;
        }

        self.unpark_slow(atomic);
    }

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

/// A generic atomic variable.
pub trait Atomic<T> {
    /// Load the value using the given ordering.
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
