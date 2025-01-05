use cubecl::prelude::*;

use crate::{cube::swizzle, prelude::*, reduction_ops::*, tiles::Dim};

/// Max planes supported (1024 threads / 32 per plane = 32 planes, but 8 is typical max)
const MAX_PLANES: usize = 32;

/// Scratch buffer for cross-plane reductions.
/// Allocate once at kernel entry and pass to `reduce_*_cube` functions.
#[derive(CubeType)]
pub struct ReduceBuf<F: Float> {
    // Intentionally not vectorized, we only want scalar ops
    data: SharedMemory<F>,
}

#[cube]
impl<F: Float> ReduceBuf<F> {
    /// Create a new reduce buffer. Uses fixed max size.
    /// Call once at kernel entry.
    pub fn new() -> ReduceBuf<F> {
        ReduceBuf::<F> {
            data: SharedMemory::new(MAX_PLANES),
        }
    }
}

impl<F: Float> Default for ReduceBuf<F> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cooperatively reduces St across rows, producing one value per column.
/// All threads in the plane participate, each handling strided rows.
/// Result is broadcast to all threads.
///
/// St<FIn, R, C> -> Rv<FAcc, C>
#[cube]
pub fn reduce_cols_plane<FIn: Float, FAcc: Float, R: Dim, C: Dim, O: ReductionOp<FAcc>>(
    s_mem: &St<FIn, R, C>,
    result: &mut Rv<FAcc, C>,
) {
    let tid = UNIT_POS as usize;
    let num_threads = PLANE_DIM as usize;

    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll(C::LINES <= UNROLL_LIMIT)]
    for c_line in 0..C::LINES {
        // Each thread accumulates strided rows for this column
        let mut acc = O::identity();
        for r in range_stepped(tid, R::VALUE, num_threads) {
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;
            // Cast input line from FIn to FAcc for accumulation
            let in_line: Line<FAcc> = Line::cast_from(s_mem.data[s_idx]);
            acc = O::combine(acc, in_line);
        }
        // Combine across threads in plane, broadcast result
        result.data[c_line] = plane_reduce_line::<FAcc, O>(acc);
    }
}

/// Cooperatively reduces St across columns, producing one value per row.
/// All threads in the plane participate, each handling strided columns.
/// Result is broadcast to all threads.
///
/// St<FIn, R, C> -> Rv<FAcc, R>
#[cube]
pub fn reduce_rows_plane<FIn: Float, FAcc: Float, R: Dim, C: Dim, O: ReductionOp<FAcc>>(
    s_mem: &St<FIn, R, C>,
    result: &mut Rv<FAcc, R>,
) {
    let tid = UNIT_POS as usize;
    let num_threads = PLANE_DIM as usize;

    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll(R::LINES <= UNROLL_LIMIT)]
    for r_line in 0..R::LINES {
        let mut out_line = Line::<FAcc>::empty(LINE_SIZE);

        #[unroll]
        for i in 0..LINE_SIZE {
            let r = r_line * LINE_SIZE + i;

            // Each thread accumulates strided columns for this row
            let mut acc = O::identity();
            for c_line in range_stepped(tid, C::LINES, num_threads) {
                let phys_col = swizzle(r, c_line, mask);
                let s_idx = r * vec_stride + phys_col;
                let in_line: Line<FAcc> = Line::cast_from(s_mem.data[s_idx]);
                acc = O::combine(acc, in_line);
            }
            let local_scalar = O::finalize(acc);

            // Combine across threads in plane
            out_line[i] = O::plane_reduce(local_scalar);
        }
        result.data[r_line] = out_line;
    }
}

/// Convenience function: sum St across rows (reduce cols)
#[cube]
pub fn sum_cols_plane<FIn: Float, FAcc: Float, R: Dim, C: Dim>(
    s_mem: &St<FIn, R, C>,
    result: &mut Rv<FAcc, C>,
) {
    reduce_cols_plane::<FIn, FAcc, R, C, SumOp>(s_mem, result)
}

/// Convenience function: sum St across columns (reduce rows)
#[cube]
pub fn sum_rows_plane<FIn: Float, FAcc: Float, R: Dim, C: Dim>(
    s_mem: &St<FIn, R, C>,
    result: &mut Rv<FAcc, R>,
) {
    reduce_rows_plane::<FIn, FAcc, R, C, SumOp>(s_mem, result)
}

// =============================================================================
// Cube-level reductions (across all planes in the workgroup)
// =============================================================================

/// Helper: reduce a single scalar across all planes using ReduceBuf.
/// All threads must call this. Result is broadcast to all threads.
#[cube]
fn reduce_across_planes<F: Float, O: ReductionOp<F>>(val: F, buf: &mut ReduceBuf<F>) -> F {
    let plane_id = (UNIT_POS / PLANE_DIM) as usize;
    let lane_id = (UNIT_POS % PLANE_DIM) as usize;
    let num_planes = (CUBE_DIM / PLANE_DIM) as usize;

    // Lane 0 of each plane writes its value
    if lane_id == 0 {
        buf.data[plane_id] = val;
    }
    sync_cube();

    // Plane 0 reduces across all plane partials
    let mut final_val = O::identity()[0]; // scalar identity
    if plane_id == 0 {
        // Each thread in plane 0 handles strided partials
        let mut acc = O::identity()[0];
        for p in range_stepped(lane_id, num_planes, PLANE_DIM as usize) {
            acc = O::plane_combine(acc, buf.data[p]);
        }
        final_val = O::plane_reduce(acc);
    }

    // Broadcast result: plane 0 lane 0 writes, all read
    if UNIT_POS == 0 {
        buf.data[0] = final_val;
    }
    sync_cube();

    buf.data[0]
}

/// Cooperatively reduces St across rows (producing one value per column),
/// reducing across ALL planes in the cube.
///
/// St<FIn, R, C> -> Rv<FAcc, C>
#[cube]
pub fn reduce_cols<FIn: Float, FAcc: Float, R: Dim, C: Dim, O: ReductionOp<FAcc>>(
    s_mem: &St<FIn, R, C>,
    result: &mut Rv<FAcc, C>,
    buf: &mut ReduceBuf<FAcc>,
) {
    let tid = UNIT_POS as usize;
    let num_threads = PLANE_DIM as usize;

    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll(C::LINES <= UNROLL_LIMIT)]
    for c_line in 0..C::LINES {
        // Stage 1: Each thread accumulates strided rows for this column
        let mut acc = O::identity();
        for r in range_stepped(tid, R::VALUE, num_threads) {
            let phys_col = swizzle(r, c_line, mask);
            let s_idx = r * vec_stride + phys_col;
            // Cast input line from FIn to FAcc for accumulation
            let in_line: Line<FAcc> = Line::cast_from(s_mem.data[s_idx]);
            acc = O::combine(acc, in_line);
        }
        // Combine across threads in plane
        let plane_partial = plane_reduce_line::<FAcc, O>(acc);

        // Stage 2: Reduce across planes for each lane
        let mut out_line = Line::<FAcc>::empty(LINE_SIZE);
        #[unroll]
        for i in 0..LINE_SIZE {
            out_line[i] = reduce_across_planes::<FAcc, O>(plane_partial[i], buf);
        }
        result.data[c_line] = out_line;
    }
}

/// Cooperatively reduces St across columns (producing one value per row),
/// reducing across ALL planes in the cube.
///
/// St<FIn, R, C> -> Rv<FAcc, R>
#[cube]
pub fn reduce_rows<FIn: Float, FAcc: Float, R: Dim, C: Dim, O: ReductionOp<FAcc>>(
    s_mem: &St<FIn, R, C>,
    result: &mut Rv<FAcc, R>,
    buf: &mut ReduceBuf<FAcc>,
) {
    let tid = UNIT_POS as usize;
    let num_threads = PLANE_DIM as usize;

    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll(R::LINES <= UNROLL_LIMIT)]
    for r_line in 0..R::LINES {
        let mut out_line = Line::<FAcc>::empty(LINE_SIZE);

        #[unroll]
        for i in 0..LINE_SIZE {
            let r = r_line * LINE_SIZE + i;

            // Stage 1: Each thread accumulates strided columns for this row
            let mut acc = O::identity();
            for c_line in range_stepped(tid, C::LINES, num_threads) {
                let phys_col = swizzle(r, c_line, mask);
                let s_idx = r * vec_stride + phys_col;
                // Cast input line from FIn to FAcc for accumulation
                let in_line: Line<FAcc> = Line::cast_from(s_mem.data[s_idx]);
                acc = O::combine(acc, in_line);
            }
            let local_scalar = O::finalize(acc);

            // Combine across threads in plane
            let plane_partial = O::plane_reduce(local_scalar);

            // Stage 2: Reduce across planes
            out_line[i] = reduce_across_planes::<FAcc, O>(plane_partial, buf);
        }
        result.data[r_line] = out_line;
    }
}

/// Convenience function: sum St across rows (reduce cols) across full cube.
#[cube]
pub fn sum_cols<FIn: Float, FAcc: Float, R: Dim, C: Dim>(
    s_mem: &St<FIn, R, C>,
    result: &mut Rv<FAcc, C>,
    buf: &mut ReduceBuf<FAcc>,
) {
    reduce_cols::<FIn, FAcc, R, C, SumOp>(s_mem, result, buf)
}

/// Convenience function: sum St across columns (reduce rows) across full cube.
#[cube]
pub fn sum_rows<FIn: Float, FAcc: Float, R: Dim, C: Dim>(
    s_mem: &St<FIn, R, C>,
    result: &mut Rv<FAcc, R>,
    buf: &mut ReduceBuf<FAcc>,
) {
    reduce_rows::<FIn, FAcc, R, C, SumOp>(s_mem, result, buf)
}

#[cfg(test)]
mod tests {
    use test_case::test_matrix;

    use super::*;
    use crate::test_utils::TestFloat;

    const ROWS: usize = 8;
    const COLS: usize = 8;

    #[cube(launch)]
    fn test_sum_st_rows_kernel<F: Float + CubeElement>(
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

        let mut result = Rv::<F, D8>::new();
        sum_rows_plane::<F, F, D8, D8>(&st, &mut result);

        if UNIT_POS == 0 {
            result.copy_to_array(output);
        }
    }

    #[cube(launch)]
    fn test_sum_st_cols_kernel<F: Float + CubeElement>(
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

        let mut result = Rv::<F, D8>::new();
        sum_cols_plane::<F, F, D8, D8>(&st, &mut result);

        if UNIT_POS == 0 {
            result.copy_to_array(output);
        }
    }

    #[cube(launch)]
    fn test_sum_st_rows_cube_kernel<F: Float + CubeElement>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut st = St::<F, D8, D8>::new();
        let mut reduce_buf = ReduceBuf::<F>::new();

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

        let mut result = Rv::<F, D8>::new();
        sum_rows::<F, F, D8, D8>(&st, &mut result, &mut reduce_buf);

        if UNIT_POS == 0 {
            result.copy_to_array(output);
        }
    }

    #[cube(launch)]
    fn test_sum_st_cols_cube_kernel<F: Float + CubeElement>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut st = St::<F, D8, D8>::new();
        let mut reduce_buf = ReduceBuf::<F>::new();

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

        let mut result = Rv::<F, D8>::new();
        sum_cols::<F, F, D8, D8>(&st, &mut result, &mut reduce_buf);

        if UNIT_POS == 0 {
            result.copy_to_array(output);
        }
    }

    test_kernel! {
        // Plane-level reduce only works within a single plane
        #[test_matrix([16, 32])]
        fn test_sum_st_rows(threads: usize) for F in [::half::f16 as f16, f32, f64] {
            let input: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let output: Array = [ROWS];

            assert_eq!(
                test_sum_st_rows_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    for r in 0..ROWS {
                        let mut sum = 0.0;
                        for c in 0..COLS {
                            sum += input[r * COLS + c].into_f64();
                        }
                        output[r] = F::from_f64(sum);
                    }
                }
            );
        }

        #[test_matrix([16, 32])]
        fn test_sum_st_cols(threads: usize) for F in [::half::f16 as f16, f32, f64] {
            let input: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let output: Array = [COLS];

            assert_eq!(
                test_sum_st_cols_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    for c in 0..COLS {
                        let mut sum = 0.0;
                        for r in 0..ROWS {
                            sum += input[r * COLS + c].into_f64();
                        }
                        output[c] = F::from_f64(sum);
                    }
                }
            );
        }

        // Cube-level reductions - test with multiple planes
        #[test_matrix([32, 64, 128])]
        fn test_sum_st_rows_cube(threads: usize) for F in [::half::f16 as f16, f32, f64] {
            let input: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let output: Array = [ROWS];

            assert_eq!(
                test_sum_st_rows_cube_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    for r in 0..ROWS {
                        let mut sum = 0.0;
                        for c in 0..COLS {
                            sum += input[r * COLS + c].into_f64();
                        }
                        output[r] = F::from_f64(sum);
                    }
                }
            );
        }

        #[test_matrix([32, 64, 128])]
        fn test_sum_st_cols_cube(threads: usize) for F in [::half::f16 as f16, f32, f64] {
            let input: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let output: Array = [COLS];

            assert_eq!(
                test_sum_st_cols_cube_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    for c in 0..COLS {
                        let mut sum = 0.0;
                        for r in 0..ROWS {
                            sum += input[r * COLS + c].into_f64();
                        }
                        output[c] = F::from_f64(sum);
                    }
                }
            );
        }
    }
}
