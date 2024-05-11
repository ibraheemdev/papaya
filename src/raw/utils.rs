use std::{
    ops::Deref,
    ptr::NonNull,
    sync::atomic::{AtomicIsize, AtomicPtr, Ordering},
};

// Polyfill for the unstable strict-provenance APIs.
#[allow(clippy::missing_safety_doc)]
pub unsafe trait StrictProvenance: Sized {
    fn addr(self) -> usize;
    fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self;
    fn unpack(self, mask: usize) -> Tagged<Self>;
}

unsafe impl<T> StrictProvenance for *mut T {
    #[inline(always)]
    fn addr(self) -> usize {
        self as usize
    }

    #[inline(always)]
    fn map_addr(self, f: impl FnOnce(usize) -> usize) -> Self {
        f(self.addr()) as Self
    }

    #[inline(always)]
    fn unpack(self, mask: usize) -> Tagged<Self> {
        Tagged {
            raw: self,
            ptr: self.map_addr(|addr| addr & mask),
            addr: self.addr() & !mask,
        }
    }
}

#[derive(Copy, Clone)]
pub struct Tagged<T> {
    // The raw, tagged pointer.
    pub raw: T,
    // The untagged pointer.
    pub ptr: T,
    // The pointer address.
    pub addr: usize,
}

pub trait AtomicPtrFetchOps<T> {
    fn fetch_or(&self, value: usize, ordering: Ordering) -> *mut T;
}

impl<T> AtomicPtrFetchOps<T> for AtomicPtr<T> {
    fn fetch_or(&self, value: usize, ordering: Ordering) -> *mut T {
        #[cfg(not(miri))]
        {
            use std::sync::atomic::AtomicUsize;

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
/// Pads and aligns a value to the length of a cache line.
#[derive(Clone, Copy, Default, Hash, PartialEq, Eq)]
// Starting from Intel's Sandy Bridge, spatial prefetcher is now pulling pairs of 64-byte cache
// lines at a time, so we have to align to 128 bytes rather than 64.
//
// Sources:
// - https://www.intel.com/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-optimization-manual.pdf
// - https://github.com/facebook/folly/blob/1b5288e6eea6df074758f877c849b6e73bbb9fbb/folly/lang/Align.h#L107
//
// ARM's big.LITTLE architecture has asymmetric cores and "big" cores have 128-byte cache line size.
//
// Sources:
// - https://www.mono-project.com/news/2016/09/12/arm64-icache/
//
// powerpc64 has 128-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_ppc64x.go#L9
#[cfg_attr(
    any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
    ),
    repr(align(128))
)]
// arm, mips, mips64, and riscv64 have 32-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_arm.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mips.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mipsle.go#L7
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_mips64x.go#L9
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_riscv64.go#L7
#[cfg_attr(
    any(
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips32r6",
        target_arch = "mips64",
        target_arch = "mips64r6",
        target_arch = "riscv64",
    ),
    repr(align(32))
)]
// s390x has 256-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_s390x.go#L7
#[cfg_attr(target_arch = "s390x", repr(align(256)))]
// x86 and wasm have 64-byte cache line size.
//
// Sources:
// - https://github.com/golang/go/blob/dda2991c2ea0c5914714469c4defc2562a907230/src/internal/cpu/cpu_x86.go#L9
// - https://github.com/golang/go/blob/3dd58676054223962cd915bb0934d1f9f489d4d2/src/internal/cpu/cpu_wasm.go#L7
//
// All others are assumed to have 64-byte cache line size.
#[cfg_attr(
    not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "powerpc64",
        target_arch = "arm",
        target_arch = "mips",
        target_arch = "mips32r6",
        target_arch = "mips64",
        target_arch = "mips64r6",
        target_arch = "riscv64",
        target_arch = "s390x",
    )),
    repr(align(64))
)]
pub struct CachePadded<T> {
    value: T,
}

// A sharded counter.
pub struct Counter(Box<[CachePadded<AtomicIsize>]>);

impl Default for Counter {
    fn default() -> Counter {
        let num_cpus = std::thread::available_parallelism()
            .map(usize::from)
            .unwrap_or(1);
        let shards = (0..num_cpus.next_power_of_two())
            .map(|_| Default::default())
            .collect();
        Counter(shards)
    }
}

impl Counter {
    pub fn get(&self, tid: usize) -> &AtomicIsize {
        &self.0[tid & (self.0.len() - 1)].value
    }

    pub fn active(&self) -> usize {
        self.0
            .iter()
            .map(|x| x.value.load(Ordering::Relaxed))
            .sum::<isize>()
            .try_into()
            // depending on the order of deletion/insertions this might be negative, so assume the
            // map is empty
            .unwrap_or(0)
    }
}

// `Box<T>` but aliasable.
pub struct AliasableBox<T>(NonNull<T>);

impl<T> From<T> for AliasableBox<T> {
    fn from(value: T) -> AliasableBox<T> {
        AliasableBox(unsafe { NonNull::new_unchecked(Box::into_raw(Box::new(value))) })
    }
}

impl<T> Deref for AliasableBox<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0.as_ptr() }
    }
}
