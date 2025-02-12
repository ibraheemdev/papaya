mod counter;
mod parker;
mod stack;
mod tagged;

pub use counter::Counter;
pub use parker::Parker;
pub use stack::Stack;
pub use tagged::{untagged, AtomicPtrFetchOps, StrictProvenance, Tagged, Unpack};

/// A `seize::Guard` that has been verified to belong to a given map.
pub trait VerifiedGuard: seize::Guard {}

#[repr(transparent)]
pub struct MapGuard<G>(G);

impl<G> MapGuard<G> {
    /// Create a new `MapGuard`.
    ///
    /// # Safety
    ///
    /// The guard must be valid to use with the given map.
    pub unsafe fn new(guard: G) -> MapGuard<G> {
        MapGuard(guard)
    }

    /// Create a new `MapGuard` from a reference.
    ///
    /// # Safety
    ///
    /// The guard must be valid to use with the given map.
    pub unsafe fn from_ref(guard: &G) -> &MapGuard<G> {
        // Safety: `VerifiedGuard` is `repr(transparent)` over `G`.
        unsafe { &*(guard as *const G as *const MapGuard<G>) }
    }
}

impl<G> VerifiedGuard for MapGuard<G> where G: seize::Guard {}

impl<G> seize::Guard for MapGuard<G>
where
    G: seize::Guard,
{
    #[inline]
    fn refresh(&mut self) {
        self.0.refresh();
    }

    #[inline]
    fn flush(&self) {
        self.0.flush();
    }

    #[inline]
    fn collector(&self) -> &seize::Collector {
        self.0.collector()
    }

    #[inline]
    fn thread_id(&self) -> usize {
        self.0.thread_id()
    }

    #[inline]
    unsafe fn defer_retire<T>(&self, ptr: *mut T, reclaim: unsafe fn(*mut T, &seize::Collector)) {
        unsafe { self.0.defer_retire(ptr, reclaim) };
    }
}

/// Pads and aligns a value to the length of a cache line.
///
// Source: https://github.com/crossbeam-rs/crossbeam/blob/0f81a6957588ddca9973e32e92e7e94abdad801e/crossbeam-utils/src/cache_padded.rs#L63.
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm64ec",
        target_arch = "powerpc64",
    ),
    repr(align(128))
)]
#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips32r6",
        target_arch = "mips64",
        target_arch = "mips64r6",
        target_arch = "sparc",
        target_arch = "hexagon",
    ),
    repr(align(32))
)]
#[cfg_attr(target_arch = "m68k", repr(align(16)))]
#[cfg_attr(target_arch = "s390x", repr(align(256)))]
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "arm64ec",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips32r6",
        target_arch = "mips64",
        target_arch = "mips64r6",
        target_arch = "sparc",
        target_arch = "hexagon",
        target_arch = "m68k",
        target_arch = "s390x",
    )),
    repr(align(64))
)]
pub struct CachePadded<T> {
    value: T,
}
