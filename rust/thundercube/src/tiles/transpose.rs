//! Matrix transpose operations for register tiles.

use cubecl::prelude::*;

use super::{dim::Dim, rt::Rt};
use crate::{
    prelude::*,
    util::{LineBlock, write_into_line},
};

/// Op codes for scalar transpose operations
const OP_LOAD: u64 = 1; // load to temp (first of multi-element cycle)
const OP_SHIFT: u64 = 2; // move src to dst
const OP_STORE: u64 = 3; // store temp (last of multi-element cycle)

/// Encode (op, dst, src) into u64 for scalar operations
const fn encode_op(op: u64, dst: usize, src: usize) -> u64 {
    op | ((dst as u64) << 8) | ((src as u64) << 24)
}

/// Decode u64 into (op, dst, src)
const fn decode_op(v: u64) -> (u64, usize, usize) {
    (
        v & 0xFF,
        ((v >> 8) & 0xFFFF) as usize,
        ((v >> 24) & 0xFFFF) as usize,
    )
}

/// Compute scalar transpose operations for an R×C matrix.
/// Returns Vec<u64> with encoded ops working on linear indices.
pub fn transpose_ops(rows: usize, cols: usize) -> Vec<u64> {
    let total = rows * cols;
    let mut visited = vec![false; total];
    let mut ops = Vec::new();

    for start in 0..total {
        if !visited[start] {
            let mut cycle = Vec::new();
            let mut curr = start;

            loop {
                visited[curr] = true;
                cycle.push(curr);

                // Element at linear index curr is at (r, c) = (curr / cols, curr % cols)
                // After transpose, it goes to (c, r), linear index = c * rows + r
                let r = curr / cols;
                let c = curr % cols;
                let next = c * rows + r;

                if next == start {
                    break;
                }
                curr = next;
            }

            if cycle.len() == 1 {
                // Fixed point (diagonal element)
            } else {
                // Multi-element cycle: shift backwards, save last element
                // For cycle [a, b, c, d] meaning a→b→c→d→a:
                // - Save arr[d] (will be overwritten when we write to arr[a])
                // - arr[d] = arr[c], arr[c] = arr[b], arr[b] = arr[a]
                // - arr[a] = saved
                let last = cycle.len() - 1;
                ops.push(encode_op(OP_LOAD, cycle[last], 0));

                // Shift backwards: arr[cycle[i]] = arr[cycle[i-1]]
                for i in (1..=last).rev() {
                    ops.push(encode_op(OP_SHIFT, cycle[i], cycle[i - 1]));
                }

                // Store temp to first position
                ops.push(encode_op(OP_STORE, cycle[0], 0));
            }
        }
    }

    ops
}

// Wrapper for scalar temp to work around CubeCL loop issues
#[derive(CubeType, Clone, Copy)]
struct WrapScalar<F: Float> {
    val: F,
}

/// Matrix transpose methods.
#[cube]
impl<F: Float, R: Dim, C: Dim> Rt<F, R, C> {
    /// Transpose the matrix in-place using scalar cycle-following.
    ///
    /// Works for any dimensions. Uses compile-time computed cycles.
    pub fn transpose(self) -> Rt<F, C, R> {
        let mut selff = self;
        let mut temp = WrapScalar::<F> { val: F::new(0.0) };

        for encoded in comptime!(transpose_ops(R::VALUE, C::VALUE)) {
            let (op, dst, src) = comptime!(decode_op(encoded));
            let dst_line = dst / LINE_SIZE;
            let dst_idx = dst % LINE_SIZE;
            let src_line = src / LINE_SIZE;
            let src_idx = src % LINE_SIZE;

            if comptime!(op == OP_LOAD) {
                temp.val = selff.data[dst_line][dst_idx];
            } else if comptime!(op == OP_SHIFT) {
                let val = selff.data[src_line][src_idx];
                write_into_line(selff.data.slice_mut(dst_line, dst_line + 1), dst_idx, val);
            } else if comptime!(op == OP_STORE) {
                write_into_line(
                    selff.data.slice_mut(dst_line, dst_line + 1),
                    dst_idx,
                    temp.val,
                );
            }
        }

        Rt::<F, C, R>::from_data(selff.data)
    }

    /// Transpose using a copy with vectorized 4×4 block operations.
    pub fn transpose_copy(self) -> Rt<F, C, R> {
        let mut result = Rt::<F, C, R>::new();
        let src_stride = C::LINES;
        let dst_stride = R::LINES;
        let src_len = comptime!(Rt::<F, R, C>::LEN);
        let dst_len = comptime!(Rt::<F, C, R>::LEN);

        #[unroll(R::LINES <= UNROLL_LIMIT)]
        for br in 0..R::LINES {
            #[unroll(C::LINES <= UNROLL_LIMIT)]
            for bc in 0..C::LINES {
                let src_base_row = br * LINE_SIZE;
                let dst_base_row = bc * LINE_SIZE;
                LineBlock::load_new(self.data.slice(0, src_len), src_base_row, bc, src_stride)
                    .transpose()
                    .store(
                        result.data.slice_mut(0, dst_len),
                        dst_base_row,
                        br,
                        dst_stride,
                    );
            }
        }

        result
    }
}

/// Square matrix methods (R == C).
#[cube]
impl<F: Float, N: Dim> Rt<F, N, N> {
    /// Transpose the square matrix in-place (mutates self).
    ///
    /// More efficient than `transpose()` for square matrices as it uses
    /// the simpler diagonal/off-diagonal block swap pattern.
    pub fn transpose_square(&mut self) {
        let stride = N::LINES;
        let len = comptime!(Rt::<F, N, N>::LEN);

        #[unroll(N::LINES <= UNROLL_LIMIT)]
        for br in 0..N::LINES {
            #[unroll(N::LINES <= UNROLL_LIMIT)]
            for bc in br..N::LINES {
                if br == bc {
                    // Diagonal block: transpose in place
                    LineBlock::load_new(self.data.slice(0, len), br * LINE_SIZE, bc, stride)
                        .transpose()
                        .store(self.data.slice_mut(0, len), br * LINE_SIZE, bc, stride);
                } else {
                    // Off-diagonal: swap blocks A and B, transposing each
                    let block_a =
                        LineBlock::load_new(self.data.slice(0, len), br * LINE_SIZE, bc, stride)
                            .transpose();
                    let block_b =
                        LineBlock::load_new(self.data.slice(0, len), bc * LINE_SIZE, br, stride)
                            .transpose();
                    block_b.store(self.data.slice_mut(0, len), br * LINE_SIZE, bc, stride);
                    block_a.store(self.data.slice_mut(0, len), bc * LINE_SIZE, br, stride);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_kernel;

    /// Reference transpose: output[c * rows + r] = input[r * cols + c]
    fn transpose_ref<T: Copy>(input: &[T], output: &mut [T], rows: usize, cols: usize) {
        for r in 0..rows {
            for c in 0..cols {
                output[c * rows + r] = input[r * cols + c];
            }
        }
    }

    #[cube(launch)]
    fn transpose_kernel<F: Float, R: Dim, C: Dim>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut rt = Rt::<F, R, C>::new();
        rt.copy_from_array(input);
        rt.transpose().copy_to_array(output);
    }

    #[cube(launch)]
    fn transpose_copy_kernel<F: Float, R: Dim, C: Dim>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut rt = Rt::<F, R, C>::new();
        rt.copy_from_array(input);
        rt.transpose_copy().copy_to_array(output);
    }

    #[cube(launch)]
    fn transpose_square_kernel<F: Float, N: Dim>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut rt = Rt::<F, N, N>::new();
        rt.copy_from_array(input);
        rt.transpose_square();
        rt.copy_to_array(output);
    }

    test_kernel! {
        #[test]
        fn test_transpose() for F in all R in [D4, D8] C in [D4, D8, D16] {
            let input: Array = [R::VALUE * C::VALUE] as Range;
            let output: Array = [R::VALUE * C::VALUE];
            assert_eq!(
                transpose_kernel(input(), output()) for (1, 1, 1) @ (1),
                { transpose_ref(&input, &mut output, R::VALUE, C::VALUE) }
            );
        }

        #[test]
        fn test_transpose_copy() for F in all R in [D4, D8] C in [D4, D8, D16] {
            let input: Array = [R::VALUE * C::VALUE] as Range;
            let output: Array = [R::VALUE * C::VALUE];
            assert_eq!(
                transpose_copy_kernel(input(), output()) for (1, 1, 1) @ (1),
                { transpose_ref(&input, &mut output, R::VALUE, C::VALUE) }
            );
        }

        #[test]
        fn test_transpose_square() for F in all N in [D4, D8] {
            let input: Array = [N::VALUE * N::VALUE] as Range;
            let output: Array = [N::VALUE * N::VALUE];
            assert_eq!(
                transpose_square_kernel(input(), output()) for (1, 1, 1) @ (1),
                { transpose_ref(&input, &mut output, N::VALUE, N::VALUE) }
            );
        }
    }
}
