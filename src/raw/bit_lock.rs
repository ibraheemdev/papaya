use std::ptr;
use std::sync::atomic::{AtomicPtr, AtomicU8, AtomicUsize, Ordering};
use std::time::Duration;
use std::time::Instant;

use parking_lot_core::{
    self, deadlock, ParkResult, SpinWait, UnparkResult, UnparkToken, DEFAULT_PARK_TOKEN,
};

use super::utils::StrictProvenance;

// UnparkToken used to indicate that that the target thread should attempt to
// lock the mutex again as soon as it is unparked.
pub(crate) const TOKEN_NORMAL: UnparkToken = UnparkToken(0);

// UnparkToken used to indicate that the mutex is being handed off to the target
// thread directly without unlocking it.
pub(crate) const TOKEN_HANDOFF: UnparkToken = UnparkToken(1);

/// This bit is set in the `state` of a `RawMutex` when that mutex is locked by some thread.
const LOCKED_BIT: usize = 0b01;

/// This bit is set in the `state` of a `RawMutex` just before parking a thread. A thread is being
/// parked if it wants to lock the mutex, but it is currently being held by some other thread.
const PARKED_BIT: usize = 0b10;

const CLEAR: usize = 0b1 << 2;

/// Raw mutex type backed by the parking lot.
pub struct PtrMutex<T> {
    /// This atomic integer holds the current state of the mutex instance. Only the two lowest bits
    /// are used. See `LOCKED_BIT` and `PARKED_BIT` for the bitmask for these bits.
    ///
    /// # State table:
    ///
    /// PARKED_BIT | LOCKED_BIT | Description
    ///     0      |     0      | The mutex is not locked, nor is anyone waiting for it.
    /// -----------+------------+------------------------------------------------------------------
    ///     0      |     1      | The mutex is locked by exactly one thread. No other thread is
    ///            |            | waiting for it.
    /// -----------+------------+------------------------------------------------------------------
    ///     1      |     0      | The mutex is not locked. One or more thread is parked or about to
    ///            |            | park. At least one of the parked threads are just about to be
    ///            |            | unparked, or a thread heading for parking might abort the park.
    /// -----------+------------+------------------------------------------------------------------
    ///     1      |     1      | The mutex is locked by exactly one thread. One or more thread is
    ///            |            | parked or about to park, waiting for the lock to become available.
    ///            |            | In this state, PARKED_BIT is only ever cleared when a bucket lock
    ///            |            | is held (i.e. in a parking_lot_core callback). This ensures that
    ///            |            | we never end up in a situation where there are parked threads but
    ///            |            | PARKED_BIT is not set (which would result in those threads
    ///            |            | potentially never getting woken up).
    state: AtomicPtr<T>,
}

impl<T> PtrMutex<T> {
    pub fn new() -> PtrMutex<T> {
        PtrMutex {
            state: AtomicPtr::new(ptr::null_mut()),
        }
    }

    pub fn load(&self) -> *mut T {
        self.state.load(Ordering::Acquire)
    }

    #[inline]
    pub fn lock(&self) {
        let state = self.state.load(Ordering::Acquire);

        if state.addr() & (LOCKED_BIT | PARKED_BIT) == 0 {
            let locked = state.map_addr(|state| state | LOCKED_BIT);

            if self
                .state
                .compare_exchange_weak(state, locked, Ordering::Acquire, Ordering::Relaxed)
                .is_err()
            {
                self.lock_slow(None);
            }
        }

        unsafe { deadlock::acquire_resource(self as *const _ as usize) };
    }

    #[inline]
    pub fn try_lock(&self) -> bool {
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            if state.addr() & LOCKED_BIT != 0 {
                return false;
            }

            let locked = state.map_addr(|state| state | LOCKED_BIT);
            match self.state.compare_exchange_weak(
                state,
                locked,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    unsafe { deadlock::acquire_resource(self as *const _ as usize) };
                    return true;
                }
                Err(x) => state = x,
            }
        }
    }

    #[inline]
    pub unsafe fn unlock_with(&self, new_state: *mut T) {
        deadlock::release_resource(self as *const _ as usize);

        let state = self.state.load(Ordering::Acquire);
        if state.addr() & PARKED_BIT == 0 {
            let cleared = new_state.map_addr(|state| state & !(LOCKED_BIT | PARKED_BIT));

            if self
                .state
                .compare_exchange(state, cleared, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }

        self.unlock_slow(state);
    }

    #[inline]
    pub unsafe fn unlock(&self) {
        deadlock::release_resource(self as *const _ as usize);

        let state = self.state.load(Ordering::Acquire);
        if state.addr() & PARKED_BIT == 0 {
            let unlocked = state.map_addr(|state| state & !LOCKED_BIT);

            if self
                .state
                .compare_exchange(state, unlocked, Ordering::Release, Ordering::Relaxed)
                .is_ok()
            {
                return;
            }
        }

        self.unlock_slow(state);
    }

    #[cold]
    fn lock_slow(&self, timeout: Option<Instant>) -> bool {
        let mut spinwait = SpinWait::new();
        let mut state = self.state.load(Ordering::Relaxed);
        loop {
            // Grab the lock if it isn't locked, even if there is a queue on it
            if state.addr() & LOCKED_BIT == 0 {
                let locked = state.map_addr(|state| state | LOCKED_BIT);

                match self.state.compare_exchange_weak(
                    state,
                    locked,
                    Ordering::Acquire,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return true,
                    Err(x) => state = x,
                }
                continue;
            }

            // If there is no queue, try spinning a few times
            if state.addr() & PARKED_BIT == 0 && spinwait.spin() {
                state = self.state.load(Ordering::Relaxed);
                continue;
            }

            // Set the parked bit
            if state.addr() & PARKED_BIT == 0 {
                let parked = state.map_addr(|state| state | PARKED_BIT);

                if let Err(x) = self.state.compare_exchange_weak(
                    state,
                    parked,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    state = x;
                    continue;
                }
            }

            // Park our thread until we are woken up by an unlock
            let addr = self as *const _ as usize;
            let validate = || self.state.load(Ordering::Relaxed).addr() == LOCKED_BIT | PARKED_BIT;
            let before_sleep = || {};
            let timed_out = |_, was_last_thread| {
                // Clear the parked bit if we were the last parked thread
                if was_last_thread {
                    unsafe {
                        // TODO: strict_provenance_ptr
                        let atomic_addr = &self.state as *const AtomicPtr<T> as *const AtomicUsize;
                        (*atomic_addr).fetch_and(!PARKED_BIT, Ordering::Relaxed);
                    }
                }
            };
            // SAFETY:
            //   * `addr` is an address we control.
            //   * `validate`/`timed_out` does not panic or call into any function of `parking_lot`.
            //   * `before_sleep` does not call `park`, nor does it panic.
            match unsafe {
                parking_lot_core::park(
                    addr,
                    validate,
                    before_sleep,
                    timed_out,
                    DEFAULT_PARK_TOKEN,
                    timeout,
                )
            } {
                // The thread that unparked us passed the lock on to us
                // directly without unlocking it.
                ParkResult::Unparked(TOKEN_HANDOFF) => return true,

                // We were unparked normally, try acquiring the lock again
                ParkResult::Unparked(_) => (),

                // The validation function failed, try locking again
                ParkResult::Invalid => (),

                // Timeout expired
                ParkResult::TimedOut => return false,
            }

            // Loop back and try locking again
            spinwait.reset();
            state = self.state.load(Ordering::Relaxed);
        }
    }

    #[cold]
    fn unlock_slow(&self, new_state: *mut T) {
        // Unpark one thread and leave the parked bit set if there might
        // still be parked threads on this address.
        let addr = self as *const _ as usize;
        let callback = |result: UnparkResult| {
            // If we are using a fair unlock then we should keep the
            // mutex locked and hand it off to the unparked thread.
            if result.unparked_threads != 0 && result.be_fair {
                // Clear the parked bit if there are no more parked threads.
                if !result.have_more_threads {
                    let unparked = new_state.map_addr(|state| (state | LOCKED_BIT) & !PARKED_BIT);
                    self.state.store(unparked, Ordering::Relaxed);
                }

                return TOKEN_HANDOFF;
            }

            // Clear the locked bit, and the parked bit as well if there
            // are no more parked threads.
            if result.have_more_threads {
                let unlocked = new_state.map_addr(|state| (state | PARKED_BIT) & !LOCKED_BIT);
                self.state.store(unlocked, Ordering::Release);
            } else {
                let cleared = new_state.map_addr(|state| state & !(LOCKED_BIT | PARKED_BIT));
                self.state.store(cleared, Ordering::Release);
            }
            TOKEN_NORMAL
        };

        // SAFETY:
        //   * `addr` is an address we control.
        //   * `callback` does not panic or call into any function of `parking_lot`.
        unsafe {
            parking_lot_core::unpark_one(addr, callback);
        }
    }
}
