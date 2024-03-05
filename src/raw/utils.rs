use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};

// fast log2 on a power of two
macro_rules! log2 {
    ($x:expr) => {
        (usize::BITS as usize) - ($x.leading_zeros() as usize) - 1
    };
}

// Polyfill for the unstable strict-provenance APIs.
pub unsafe trait StrictProvenance: Sized {
    fn addr(self) -> usize;
    fn with_addr(self, addr: usize) -> Self;
    fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self;
    fn set(self, mask: usize) -> Self;
    fn mask(self, mask: usize) -> Self;
    fn unmask(self, mask: usize) -> usize;
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

    fn set(self, mask: usize) -> Self {
        self.map_addr(|addr| addr | mask)
    }

    fn unmask(self, mask: usize) -> usize {
        self.addr() & !mask
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
                .fetch_or(value, Ordering::Release) as *mut T
        }

        #[cfg(miri)]
        {
            self.fetch_update(Ordering::Release, Ordering::Relaxed, |ptr| {
                Some(ptr.map_addr(|addr| addr | value))
            })
            .unwrap()
        }
    }
}

pub(crate) use log2;
