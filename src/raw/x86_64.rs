use std::arch::{asm, x86_64};
use std::mem;
use std::num::NonZeroU16;

#[test]
fn test_128() {
    unsafe {
        dbg!(0x12_u8);
        let x: u128 = u128::from_be_bytes([7, 6, 5, 4, 3, 2, 1, 7, 6, 5, 4, 3, 2, 1, 7, 0]);
        let y = load_128(&x as *const _ as *mut _);
        assert_eq!(match_byte(y, 7).collect::<Vec<_>>(), [1, 8, 15]);
    }
}

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
        let bit = NonZeroU16::new(self.0)?.trailing_zeros();
        self.0 = self.0 & (self.0 - 1);
        Some(bit as usize)
    }
}
