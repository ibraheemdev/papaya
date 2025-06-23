use std::mem::align_of;
use std::sync::atomic::{AtomicPtr, Ordering};

// Polyfill for the unstable strict-provenance APIs.
#[allow(clippy::missing_safety_doc)]
#[allow(dead_code)] // `strict_provenance` has stabilized on nightly.
pub unsafe trait StrictProvenance<T>: Sized {
    fn addr(self) -> usize;
    fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self;
    fn unpack(self) -> Tagged<T>
    where
        T: Unpack;
}

// Unpack a tagged pointer.
pub trait Unpack {
    // A mask for the pointer tag bits.
    const MASK: usize;
}

// This function does nothing, but will fail to compile if T doesn't have an alignment
// that guarantees all valid pointers have zero in the bits excluded by T::MASK.
const fn static_assert_align_of<T: Unpack>() {
    struct Dummy<T>(T);
    impl<T: Unpack> Dummy<T> {
        const ASSERT: () = assert!(align_of::<T>() > !T::MASK);
    }
    Dummy::<T>::ASSERT
}

unsafe impl<T> StrictProvenance<T> for *mut T {
    #[inline(always)]
    fn addr(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self {
        f(self.addr()) as Self
    }

    #[inline(always)]
    fn unpack(self) -> Tagged<T>
    where
        T: Unpack,
    {
        static_assert_align_of::<T>();
        Tagged {
            raw: self,
            ptr: self.map_addr(|addr| addr & T::MASK),
        }
    }
}

// An unpacked tagged pointer.
pub struct Tagged<T> {
    // The raw tagged pointer.
    pub raw: *mut T,

    // The untagged pointer.
    pub ptr: *mut T,
}

// Creates a `Tagged` from an untagged pointer.
#[inline]
pub fn untagged<T>(value: *mut T) -> Tagged<T> {
    Tagged {
        raw: value,
        ptr: value,
    }
}

impl<T> Tagged<T>
where
    T: Unpack,
{
    // Returns the tag portion of this pointer.
    #[inline]
    pub fn tag(self) -> usize {
        self.raw.addr() & !T::MASK
    }

    // Maps the tag of this pointer.
    #[inline]
    pub fn map_tag(self, f: impl FnOnce(usize) -> usize) -> Self {
        Tagged {
            raw: self.raw.map_addr(f),
            ptr: self.ptr,
        }
    }
}

impl<T> Copy for Tagged<T> {}

impl<T> Clone for Tagged<T> {
    fn clone(&self) -> Self {
        *self
    }
}

// Polyfill for the unstable `atomic_ptr_strict_provenance` APIs.
pub trait AtomicPtrFetchOps<T> {
    fn fetch_or(&self, value: usize, ordering: Ordering) -> *mut T;
}

impl<T> AtomicPtrFetchOps<T> for AtomicPtr<T> {
    #[inline]
    fn fetch_or(&self, value: usize, ordering: Ordering) -> *mut T {
        #[cfg(not(miri))]
        {
            use std::sync::atomic::AtomicUsize;

            // Safety: `AtomicPtr` and `AtomicUsize` are identical in terms
            // of memory layout. This operation is technically invalid in that
            // it loses provenance, but there is no stable alternative.
            unsafe { &*(self as *const AtomicPtr<T> as *const AtomicUsize) }
                .fetch_or(value, ordering) as *mut T
        }

        // Avoid ptr2int under Miri.
        #[cfg(miri)]
        {
            // Returns the ordering for the read in an RMW operation.
            const fn read_ordering(ordering: Ordering) -> Ordering {
                match ordering {
                    Ordering::SeqCst => Ordering::SeqCst,
                    Ordering::AcqRel => Ordering::Acquire,
                    _ => Ordering::Relaxed,
                }
            }

            self.fetch_update(ordering, read_ordering(ordering), |ptr| {
                Some(ptr.map_addr(|addr| addr | value))
            })
            .unwrap()
        }
    }
}
