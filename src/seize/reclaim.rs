//! Common memory reclaimers.
//!
//! Functions in this module can be passed to [`retire`](crate::Collector::retire)
//! to free allocated memory or run drop glue. See [the guide](crate#3-reclaimers)
//! for details about memory reclamation, and writing custom reclaimers.

use std::ptr;

use super::Link;

/// Reclaims memory allocated with [`Box`].
///
/// This function calls [`Box::from_raw`] on the linked pointer.
///
/// # Safety
///
/// Ensure that the correct type annotations are used when
/// passing this function to [`retire`](crate::Collector::retire):
/// the link passed must have been created from a **valid**
/// `Linked<T>`.
pub unsafe fn boxed<T>(mut link: *mut Link) {
    unsafe {
        let _ = Box::from_raw(link.cast::<T>());
    }
}

/// Reclaims memory by dropping the value in place.
///
/// This function calls [`ptr::drop_in_place`] on the linked pointer.
///
/// # Safety
///
/// Ensure that the correct type annotations are used when
/// passing this function to [`retire`](crate::Collector::retire):
/// the link passed must have been created from a **valid**
/// `Linked<T>`.
pub unsafe fn in_place<T>(mut link: *mut Link) {
    unsafe {
        ptr::drop_in_place(link.cast::<T>());
    }
}
