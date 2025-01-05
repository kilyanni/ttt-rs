use std::marker::PhantomData;

use cubecl::prelude::*;

use super::dim::{Dim, DimOrOne};
use crate::{binary_ops::*, cube::swizzle, prelude::*, unary_ops::*};

#[derive(CubeType)]
pub struct St<F: Float, R: Dim, C: DimOrOne> {
    pub data: SharedMemory<Line<F>>,
    #[cube(comptime)]
    _phantom: PhantomData<(R, C)>,
    // This is just for ergonomics,
    // as we can't access Self::LEN due to CubeCL limitations
    #[cube(comptime)]
    len: usize,
}

pub type Sv<F, L> = St<F, L, D1>;

impl<F: Float, R: Dim, C: DimOrOne> St<F, R, C> {
    pub const ROWS: usize = R::VALUE;
    pub const COLS: usize = C::VALUE;
    pub const SIZE: usize = R::VALUE * C::VALUE;
    pub const LEN: usize = R::VALUE * C::VALUE / LINE_SIZE;

    pub fn len() -> usize {
        Self::LEN
    }

    pub fn size() -> usize {
        Self::SIZE
    }
}

/// General methods for all shared memory tiles (including vectors).
#[cube]
impl<F: Float, R: Dim, C: DimOrOne> St<F, R, C> {
    pub fn new() -> St<F, R, C> {
        St::<F, R, C> {
            data: SharedMemory::new_lined(comptime!(Self::LEN), LINE_SIZE),
            _phantom: PhantomData,
            len: Self::LEN,
        }
    }

    pub fn apply_unary_op<O: UnaryOp<F>>(&mut self, op: O) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        for i in range_stepped(tid, self.len, num_threads) {
            self.data[i] = op.apply(self.data[i]);
        }
    }

    pub fn apply_binary_op<O: BinaryOp<F>>(&mut self, op: O, other: &St<F, R, C>) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        for i in range_stepped(tid, self.len, num_threads) {
            self.data[i] = op.apply(self.data[i], other.data[i]);
        }
    }

    /// Copy contents from another St (cooperative, all threads participate)
    pub fn copy_from(&mut self, other: &St<F, R, C>) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        for i in range_stepped(tid, self.len, num_threads) {
            self.data[i] = other.data[i];
        }
    }
}

/// Matrix-specific methods (require C: Dim, i.e., C != D1).
#[cube]
impl<F: Float, R: Dim, C: Dim> St<F, R, C> {
    pub fn apply_row_broadcast<O: BinaryOp<F>>(&mut self, op: O, row: &Rv<F, C>) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        let vec_stride = C::LINES;
        let mask = vec_stride - 1;

        for i in range_stepped(tid, R::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;

            self.data[s_idx] = op.apply(self.data[s_idx], row.data[c_line]);
        }
    }

    pub fn apply_col_broadcast<O: BinaryOp<F>>(&mut self, op: O, col: &Rv<F, R>) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        let vec_stride = C::LINES;
        let mask = vec_stride - 1;

        for i in range_stepped(tid, R::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;

            // Get the scalar for this row from the column vector
            let r_line = r / LINE_SIZE;
            let r_idx = r % LINE_SIZE;
            let col_val = col.data[r_line][r_idx];
            let broadcast = Line::<F>::empty(LINE_SIZE).fill(col_val);

            self.data[s_idx] = op.apply(self.data[s_idx], broadcast);
        }
    }

    /// Zero elements above the diagonal (keep lower triangular).
    /// For element (r, c): keep if c <= r, zero if c > r.
    /// Cooperative: all threads participate.
    pub fn tril(&mut self) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        let vec_stride = C::LINES;
        let mask = vec_stride - 1;
        let zero = F::new(0.0);

        for i in range_stepped(tid, R::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;

            let mut line = self.data[s_idx];

            let c_base = c_line * LINE_SIZE;
            if c_base + 0 > r {
                line[0] = zero;
            }
            if c_base + 1 > r {
                line[1] = zero;
            }
            if c_base + 2 > r {
                line[2] = zero;
            }
            if c_base + 3 > r {
                line[3] = zero;
            }

            self.data[s_idx] = line;
        }
    }

    /// Zero elements below the diagonal (keep upper triangular).
    /// For element (r, c): keep if c >= r, zero if c < r.
    /// Cooperative: all threads participate.
    pub fn triu(&mut self) {
        let num_threads = CUBE_DIM as usize;
        let tid = UNIT_POS as usize;

        let vec_stride = C::LINES;
        let mask = vec_stride - 1;
        let zero = F::new(0.0);

        for i in range_stepped(tid, R::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;

            let mut line = self.data[s_idx];

            let c_base = c_line * LINE_SIZE;
            if c_base + 0 < r {
                line[0] = zero;
            }
            if c_base + 1 < r {
                line[1] = zero;
            }
            if c_base + 2 < r {
                line[2] = zero;
            }
            if c_base + 3 < r {
                line[3] = zero;
            }

            self.data[s_idx] = line;
        }
    }
}

impl<F: Float, R: Dim, C: DimOrOne> Default for St<F, R, C> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestFloat;

    const SIZE: usize = 8;

    #[cube(launch)]
    fn test_tril_kernel<F: Float + CubeElement>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut st = St::<F, D8, D8>::new();

        // Load input into St (with swizzle)
        let tid = UNIT_POS as usize;
        let num_threads = CUBE_DIM as usize;
        let vec_stride = D8::LINES;
        let mask = vec_stride - 1;

        for i in range_stepped(tid, D8::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;
            st.data[s_idx] = input[i];
        }

        sync_cube();

        st.tril();

        sync_cube();

        // Read back from St (with swizzle)
        for i in range_stepped(tid, D8::VALUE * vec_stride, num_threads) {
            let r = i / vec_stride;
            let c_line = i % vec_stride;
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;
            output[i] = st.data[s_idx];
        }
    }

    test_kernel! {
        #[test]
        fn test_tril() for F in all {
            let input: Array = [SIZE * SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [SIZE * SIZE];

            assert_eq!(
                test_tril_kernel(input(), output()) for (1, 1, 1) @ (32),
                {
                    for r in 0..SIZE {
                        for c in 0..SIZE {
                            let idx = r * SIZE + c;
                            if c <= r {
                                // Lower triangular: keep original value
                                output[idx] = input[idx];
                            } else {
                                // Upper triangular: zero
                                output[idx] = F::from_f64(0.0);
                            }
                        }
                    }
                }
            );
        }
    }
}
