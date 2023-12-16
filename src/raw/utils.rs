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
    fn split(self, mask: usize) -> (Self, usize);
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

    fn split(self, mask: usize) -> (Self, usize) {
        (self.mask(mask), self.unmask(mask))
    }

    fn mask(self, mask: usize) -> Self {
        self.map_addr(|addr| addr & mask)
    }

    fn unmask(self, mask: usize) -> usize {
        self.addr() & !mask
    }
}

pub(crate) use log2;
