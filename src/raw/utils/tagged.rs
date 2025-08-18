use std::{
    marker::PhantomData,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::raw::table::Entry;

// Polyfill for the unstable strict-provenance APIs.
#[allow(clippy::missing_safety_doc)]
#[allow(dead_code)] // `strict_provenance` has stabilized on nightly.
pub unsafe trait StrictProvenance<T>: Sized {
    fn addr(self) -> usize;
    fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self;
    fn unpack<U>(self) -> Tagged<T, U>
    where
        U: Unpack;

    // This constant, if used, will fail to compile if `T` doesn't have an alignment
    // that guarantees all valid pointers have zero in the bits excluded by `T::MASK`.
    //
    // TODO: Make this generic over `T`.
    const ASSERT_ALIGNMENT: () = assert!(align_of::<Self>() > !Entry::MASK);
}

// Unpack a tagged pointer.
pub trait Unpack: Sized {
    // A mask for the pointer tag bits.
    const MASK: usize;
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
    fn unpack<U>(self) -> Tagged<T, U>
    where
        U: Unpack,
    {
        let () = Self::ASSERT_ALIGNMENT;

        Tagged {
            raw: self,
            ptr: self.map_addr(|addr| addr & Entry::MASK),
            _unpack: PhantomData,
        }
    }
}

// An unpacked tagged pointer.
pub struct Tagged<T, U> {
    // The raw tagged pointer.
    pub raw: *mut T,

    // The untagged pointer.
    pub ptr: *mut T,

    _unpack: PhantomData<U>,
}

// Creates a `Tagged` from an untagged pointer.
#[inline]
pub fn untagged<T, U>(value: *mut T) -> Tagged<T, U> {
    Tagged {
        raw: value,
        ptr: value,
        _unpack: PhantomData,
    }
}

impl<T, U> Tagged<T, U>
where
    U: Unpack,
{
    // Returns the tag portion of this pointer.
    #[inline]
    pub fn tag(self) -> usize {
        self.raw.addr() & !U::MASK
    }

    // Maps the tag of this pointer.
    #[inline]
    pub fn map_tag(self, f: impl FnOnce(usize) -> usize) -> Self {
        Tagged {
            raw: self.raw.map_addr(f),
            ptr: self.ptr,
            _unpack: PhantomData,
        }
    }
}

impl<T, U> Copy for Tagged<T, U> {}

impl<T, U> Clone for Tagged<T, U> {
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
