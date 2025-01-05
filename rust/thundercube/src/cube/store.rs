use cubecl::prelude::*;

use crate::{
    cube::swizzle,
    prelude::*,
    tiles::Dim,
    util::{cast_line, transpose_4, write_into_line},
};

/// Cooperatively stores per-thread register tiles into shared memory.
///
/// Each thread stores its Rt to a sub-region of St determined by UNIT_POS.
/// Threads are mapped to sub-tiles in row-major order. The St uses a swizzled
/// layout for bank-conflict-free access.
///
/// Supports heterogeneous types: FRt (register) can differ from FSt (shared memory).
/// When types differ, values are cast element-wise. When equal, cast optimizes away.
///
/// Threads with UNIT_POS >= num_sub_tiles are safely skipped.
///
/// # Type Parameters
/// * `FRt` - Register tile element type
/// * `FSt` - Shared memory tile element type
/// * `R, C` - Register tile dimensions
/// * `SR, SC` - Shared memory tile dimensions (must be multiples of R, C)
#[cube]
pub fn store_rt_to_st<FRt: Float, FSt: Float, R: Dim, C: Dim, SR: Dim, SC: Dim>(
    rt_mem: &Rt<FRt, R, C>,
    s_mem: &mut St<FSt, SR, SC>,
) {
    let tiles_per_row = SC::VALUE / C::VALUE;
    let tiles_per_col = SR::VALUE / R::VALUE;
    let num_tiles = tiles_per_row * tiles_per_col;

    // Guard: only threads with valid tile indices participate
    if (UNIT_POS as usize) < num_tiles {
        let tile_idx = UNIT_POS as usize;
        let tile_row = tile_idx / tiles_per_row;
        let tile_col = tile_idx % tiles_per_row;

        let offset_row = tile_row * R::VALUE;
        let offset_col_vec = tile_col * C::LINES;

        let rt_rows = R::VALUE;
        let num_c_vecs = C::LINES;
        let s_stride = SC::LINES;
        let mask = s_stride - 1;

        #[unroll(R::VALUE <= UNROLL_LIMIT)]
        for row in 0..rt_rows {
            let s_row = offset_row + row;

            #[unroll(C::LINES <= UNROLL_LIMIT)]
            for cv in 0..num_c_vecs {
                let s_col_vec = offset_col_vec + cv;
                let phys_col = swizzle(s_row, s_col_vec, mask);
                let s_idx = s_row * s_stride + phys_col;

                let rt_idx = row * num_c_vecs + cv;
                s_mem.data[s_idx] = cast_line(rt_mem.data[rt_idx]);
            }
        }
    }
}

/// Cooperatively stores a shared memory tile to global memory without transposing.
///
/// All threads in the workgroup participate, each storing multiple `Line<F>` vectors
/// in a strided pattern to maximize memory bandwidth. Reads from shared memory use the
/// swizzled layout to avoid bank conflicts.
///
/// # Arguments
/// * `s_mem` - Source shared memory tile (`St`)
/// * `g_mem` - Destination tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for this tile
/// * `g_offset_col` - Column offset within the matrix for this tile
#[cube]
pub fn store_st_direct<F: Float, R: Dim, C: Dim>(
    s_mem: &St<F, R, C>,
    g_mem: &mut Tensor<Line<F>>,
    base_offset: usize,
    g_offset_row: usize,
    g_offset_col: usize,
) {
    let vec_stride = C::LINES;
    let total_vecs = R::VALUE * vec_stride;

    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    for i in range_stepped(tid, total_vecs, num_threads) {
        let r = i / vec_stride;
        let c_vec = i % vec_stride;

        let mask = vec_stride - 1;
        let phys_c = swizzle(r, c_vec, mask);
        let s_idx = (r * vec_stride) + phys_c;

        let val = s_mem.data[s_idx];

        let g_r = g_offset_row + r;
        let g_c = g_offset_col + (c_vec * LINE_SIZE);

        store_safe(g_mem, base_offset, g_r, g_c, val);
    }
}

/// Cooperatively stores a shared memory tile to global memory while transposing.
///
/// The transpose is performed in `LINE_SIZE × LINE_SIZE` patches (4×4 by default).
/// Each thread processes entire patches: loading 4 rows from shared memory (with swizzle),
/// transposing them via register shuffles, then storing as 4 rows to global memory at
/// swapped coordinates.
///
/// # Arguments
/// * `s_mem` - Source shared memory tile (`St`)
/// * `g_mem` - Destination tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for the destination tile
/// * `g_offset_col` - Column offset within the matrix for the destination tile
#[cube]
pub fn store_st_transpose<F: Float, R: Dim, C: Dim>(
    s_mem: &St<F, R, C>,
    g_mem: &mut Tensor<Line<F>>,
    base_offset: usize,
    g_offset_row: usize,
    g_offset_col: usize,
) {
    let s_stride = C::LINES;
    let patches_h = R::LINES;
    let patches_w = C::LINES;
    let total_patches = patches_h * patches_w;

    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    for i in range_stepped(tid, total_patches, num_threads) {
        let patch_r = i / patches_w;
        let patch_c = i % patches_w;

        let src_base_r = patch_r * LINE_SIZE;
        let src_vec_c = patch_c;
        let mask = s_stride - 1;

        let r0 = src_base_r + 0;
        let idx0 = r0 * s_stride + swizzle(r0, src_vec_c, mask);
        let v0 = s_mem.data[idx0];

        let r1 = src_base_r + 1;
        let idx1 = r1 * s_stride + swizzle(r1, src_vec_c, mask);
        let v1 = s_mem.data[idx1];

        let r2 = src_base_r + 2;
        let idx2 = r2 * s_stride + swizzle(r2, src_vec_c, mask);
        let v2 = s_mem.data[idx2];

        let r3 = src_base_r + 3;
        let idx3 = r3 * s_stride + swizzle(r3, src_vec_c, mask);
        let v3 = s_mem.data[idx3];

        let (t0, t1, t2, t3) = transpose_4(v0, v1, v2, v3);

        // We swap the patch coordinates to flip the block location
        let g_patch_r = patch_c;
        let g_patch_c = patch_r;

        let dst_r = g_offset_row + (g_patch_r * LINE_SIZE);
        let dst_c = g_offset_col + (g_patch_c * LINE_SIZE);

        store_safe(g_mem, base_offset, dst_r + 0, dst_c, t0);
        store_safe(g_mem, base_offset, dst_r + 1, dst_c, t1);
        store_safe(g_mem, base_offset, dst_r + 2, dst_c, t2);
        store_safe(g_mem, base_offset, dst_r + 3, dst_c, t3);
    }
}

/// Cooperatively stores per-thread register tiles to global memory without transposing.
///
/// Each thread stores its own register tile sub-region. Threads are mapped to sub-tiles
/// in row-major order based on UNIT_POS, matching `load_rt_from_st`. Threads with
/// UNIT_POS >= num_sub_tiles are safely skipped.
///
/// Out-of-bounds writes are also skipped, enabling safe stores at matrix boundaries.
///
/// # Type Parameters
/// * `R, C` - Register tile dimensions
/// * `SR, SC` - Full tile dimensions (determines thread mapping)
///
/// # Arguments
/// * `rt_mem` - Source register tile (`Rt`), stored row-major as `[rows, cols/LINE_SIZE]` Lines
/// * `g_mem` - Destination tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Additional row offset within the matrix (usually 0)
/// * `g_offset_col` - Additional column offset within the matrix (usually 0)
#[cube]
pub fn store_rt_direct<F: Float, R: Dim, C: Dim, SR: Dim, SC: Dim>(
    rt_mem: &Rt<F, R, C>,
    g_mem: &mut Tensor<Line<F>>,
    base_offset: usize,
    g_offset_row: usize,
    g_offset_col: usize,
) {
    let tiles_per_row = SC::VALUE / C::VALUE;
    let tiles_per_col = SR::VALUE / R::VALUE;
    let num_tiles = tiles_per_row * tiles_per_col;

    // Guard: only threads with valid tile indices participate
    if (UNIT_POS as usize) < num_tiles {
        let tile_idx = UNIT_POS as usize;
        let tile_row = tile_idx / tiles_per_row;
        let tile_col = tile_idx % tiles_per_row;

        // Compute this thread's offset within the full tile
        let offset_row = tile_row * R::VALUE;
        let offset_col = tile_col * C::VALUE;

        let rt_rows = R::VALUE;
        let num_n_vecs = C::LINES;

        let rank = g_mem.rank();
        let num_rows = g_mem.shape(rank - 2);
        let num_cols = g_mem.shape(rank - 1);
        let row_stride = g_mem.stride(rank - 2);

        #[unroll(R::VALUE <= UNROLL_LIMIT)]
        for row in 0..rt_rows {
            let g_r = g_offset_row + offset_row + row;
            if g_r < num_rows {
                #[unroll(C::LINES <= UNROLL_LIMIT)]
                for nl in 0..num_n_vecs {
                    let g_c = g_offset_col + offset_col + nl * LINE_SIZE;
                    if g_c < num_cols {
                        let rt_idx = row * num_n_vecs + nl;
                        let val = rt_mem.data[rt_idx];

                        let scalar_idx = base_offset + g_r * row_stride + g_c;
                        let line_idx = scalar_idx / LINE_SIZE;
                        g_mem[line_idx] = val;
                    }
                }
            }
        }
    }
}

/// Stores a single `Line<F>` (vector of `LINE_SIZE` elements) to a global tensor.
///
/// Handles both row-major and column-major layouts:
/// - **Row-major** (`col_stride == 1`): Direct contiguous store of `LINE_SIZE` consecutive columns
/// - **Column-major**: Scatters `LINE_SIZE` scalars to non-contiguous locations
///
/// Skips writes for out-of-bounds coordinates, enabling safe boundary handling.
///
/// # Arguments
/// * `g` - Destination tensor (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Pre-computed scalar offset from batch dimensions
/// * `r` - Row index within the matrix
/// * `c` - Starting column index (must be aligned to `LINE_SIZE` for row-major)
/// * `val` - The `Line<F>` to store
#[cube]
pub fn store_safe<F: Float>(
    g: &mut Tensor<Line<F>>,
    base_offset: usize,
    r: usize,
    c: usize,
    val: Line<F>,
) {
    let rank = g.rank();
    let row_dim = rank - 2;
    let col_dim = rank - 1;

    let num_rows = g.shape(row_dim);
    let num_cols = g.shape(col_dim);
    let row_stride = g.stride(row_dim);
    let col_stride = g.stride(col_dim);

    if r < num_rows && c < num_cols {
        let scalar_idx = base_offset + r * row_stride + c * col_stride;

        if col_stride == 1 {
            // Row-major: scalar index maps to Line index by dividing by LINE_SIZE
            let line_idx = scalar_idx / LINE_SIZE;
            g[line_idx] = val;
        } else {
            // Column-major: scatter 4 scalars to different Lines
            let line_idx0 = scalar_idx / LINE_SIZE;
            let line_idx1 = (scalar_idx + col_stride) / LINE_SIZE;
            let line_idx2 = (scalar_idx + col_stride * 2) / LINE_SIZE;
            let line_idx3 = (scalar_idx + col_stride * 3) / LINE_SIZE;

            let elem0 = scalar_idx % LINE_SIZE;
            let elem1 = (scalar_idx + col_stride) % LINE_SIZE;
            let elem2 = (scalar_idx + col_stride * 2) % LINE_SIZE;
            let elem3 = (scalar_idx + col_stride * 3) % LINE_SIZE;

            write_into_line(g.slice_mut(line_idx0, line_idx0 + 1), elem0, val[0]);
            write_into_line(g.slice_mut(line_idx1, line_idx1 + 1), elem1, val[1]);
            write_into_line(g.slice_mut(line_idx2, line_idx2 + 1), elem2, val[2]);
            write_into_line(g.slice_mut(line_idx3, line_idx3 + 1), elem3, val[3]);
        }
    }
}

/// Cooperatively stores a shared memory vector (Sv) to global memory.
///
/// All threads participate - each thread stores different Lines in a strided pattern.
///
/// # Arguments
/// * `s_mem` - Source shared memory vector
/// * `g_mem` - Destination 1D tensor of Lines
/// * `base_offset` - Scalar offset into the destination tensor
#[cube]
pub fn store_sv_direct<F: Float, L: Dim>(
    s_mem: &Sv<F, L>,
    g_mem: &mut Tensor<Line<F>>,
    base_offset: usize,
) {
    let num_lines = L::LINES;
    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    for i in range_stepped(tid, num_lines, num_threads) {
        let line_idx = base_offset / LINE_SIZE + i;
        g_mem[line_idx] = s_mem.data[i];
    }
}
