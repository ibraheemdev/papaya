use std::arch::{asm, x86_64};
use std::mem;
use std::num::{NonZeroU16, NonZeroU64, NonZeroU8};

#[cfg(miri)]
pub unsafe fn load_128(src: *mut u128) -> x86_64::__m128i {
    mem::transmute((*src).to_ne_bytes())
}

#[cfg(not(miri))]
pub unsafe fn load_128(src: *mut u128) -> x86_64::__m128i {
    debug_assert!(src as usize % 16 == 0);

    unsafe {
        let out: x86_64::__m128i;
        asm!(
            concat!("vmovdqa {out}, xmmword ptr [{src}]"),
            src = in(reg) src,
            out = out(xmm_reg) out,
            options(nostack, preserves_flags),
        );
        out
    }
}

pub fn match_byte(group: x86_64::__m128i, byte: u8) -> BitIter {
    unsafe {
        let cmp = x86_64::_mm_cmpeq_epi8(group, x86_64::_mm_set1_epi8(byte as i8));
        BitIter(x86_64::_mm_movemask_epi8(cmp) as u16)
    }
}

fn repeat(b: u8) -> u64 {
    u64::from_ne_bytes([b; 8])
}

pub fn match_byte_8(group: u64, byte: u8) -> BitIter8 {
    let cmp = group ^ repeat(byte);
    let x = (cmp.wrapping_sub(repeat(0x01)) & !cmp & repeat(0x80)).to_le();
    let x = x & 0x8080_8080_8080_8080_u64;
    BitIter8(x)
}

pub fn match_empty_8(group: u64) -> BitIter8 {
    BitIter8((group & repeat(0x80)).to_le())
}

pub struct BitIter(u16);

impl BitIter {
    pub fn any_set(self) -> bool {
        self.0 != 0
    }
}

impl Iterator for BitIter {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let bit = (NonZeroU16::new(self.0)?.trailing_zeros());
        self.0 = self.0 & (self.0 - 1);
        Some(bit as usize)
    }
}

pub struct BitIter8(u64);

impl BitIter8 {
    pub fn any_set(self) -> bool {
        self.0 != 0
    }
}

impl Iterator for BitIter8 {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let bit = (NonZeroU64::new(self.0)?.trailing_zeros() as usize) / 8;
        self.0 = self.0 & (self.0 - 1);
        Some(bit)
    }
}
