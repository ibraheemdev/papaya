use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

// Polyfill for the unstable strict-provenance APIs.
pub unsafe trait StrictProvenance: Sized {
    fn addr(self) -> usize;
    fn with_addr(self, addr: usize) -> Self;
    fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self;
    fn mask(self, mask: usize) -> Self;
}

unsafe impl<T> StrictProvenance for *mut T {
    fn addr(self) -> usize {
        self as usize
    }

    fn with_addr(self, addr: usize) -> Self {
        addr as Self
    }

    fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self {
        self.with_addr(f(self.addr()))
    }

    fn mask(self, mask: usize) -> Self {
        self.map_addr(|addr| addr & mask)
    }
}

pub trait AtomicPtrFetchOps<T> {
    fn fetch_or(&self, value: usize, ordering: Ordering) -> *mut T;
}

impl<T> AtomicPtrFetchOps<T> for AtomicPtr<T> {
    fn fetch_or(&self, value: usize, ordering: Ordering) -> *mut T {
        #[cfg(not(miri))]
        {
            // mark the entry as copied
            unsafe { &*(self as *const AtomicPtr<T> as *const AtomicUsize) }
                .fetch_or(value, ordering) as *mut T
        }

        #[cfg(miri)]
        {
            self.fetch_update(ordering, Ordering::Acquire, |ptr| {
                Some(ptr.map_addr(|addr| addr | value))
            })
            .unwrap()
        }
    }
}
