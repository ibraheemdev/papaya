// The maximum probe length for table operations.
//
// Estimating a load factor for the hash-table based on probe lengths allows
// the hash-table to avoid loading the length every insert, which is a source
// of contention.
macro_rules! probe_limit {
    ($capacity:expr) => {
        // 5 * log2(capacity): Testing shows this gives us a ~80% load factor.
        5 * ((usize::BITS as usize) - ($capacity.leading_zeros() as usize) - 1)
    };
}

// Returns an estimate of the number of entries needed to hold `capacity` elements.
pub fn entries_for(capacity: usize) -> usize {
    // We should rarely resize before 75%.
    let capacity = capacity.checked_mul(8).expect("capacity overflow") / 6;
    capacity.next_power_of_two()
}

// A hybrid probe sequence.
//
// The probe sequencey walks a number of entries linearly before making
// a quadratic jump, balancing cache locality with probe lengths.
#[derive(Default)]
pub struct Probe {
    // The current index in the probe sequence.
    pub i: usize,
    // The current length of the probe sequence.
    pub len: usize,
    // Mask for the length of the table.
    mask: usize,
    // The current quadratic stride.
    stride: usize,
}

impl Probe {
    // Number of linear probes per quadratic jump.
    const GROUP: usize = 8;

    // Initialize the probe sequence, returning the maximum probe limit.
    #[inline]
    pub fn start(hash: usize, len: usize) -> (Probe, usize) {
        let i = hash & (len - 1);
        let probe = Probe {
            i,
            len: 0,
            stride: 0,
            mask: len - 1,
        };

        (probe, probe_limit!(len))
    }

    // Increment the probe sequence.
    #[inline]
    pub fn next(&mut self) {
        self.len += 1;

        if self.len & (Probe::GROUP - 1) == 0 {
            self.stride += Probe::GROUP;
            self.i += self.stride;
        } else {
            self.i += 1;
        }

        self.i &= self.mask;
    }
}
