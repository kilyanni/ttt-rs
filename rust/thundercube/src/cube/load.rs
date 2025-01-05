use cubecl::prelude::*;

use crate::{
    cube::swizzle,
    prelude::*,
    tiles::Dim,
    util::{cast_line, transpose_4},
};

/// Cooperatively loads from shared memory into per-thread register tiles.
///
/// Each thread loads its Rt from a sub-region of St determined by UNIT_POS.
/// Threads are mapped to sub-tiles in row-major order. The St uses a swizzled
/// layout for bank-conflict-free access.
///
/// Supports heterogeneous types: FSt can differ from FRt.
/// When types differ, values are cast element-wise. When equal, cast optimizes away.
///
/// Threads with UNIT_POS >= num_sub_tiles are safely skipped (Rt unchanged).
///
/// # Type Parameters
/// * `FSt` - Shared memory tile element type
/// * `FRt` - Register tile element type
/// * `R, C` - Register tile dimensions
/// * `SR, SC` - Shared memory tile dimensions (must be multiples of R, C)
#[cube]
pub fn load_rt_from_st<FSt: Float, FRt: Float, R: Dim, C: Dim, SR: Dim, SC: Dim>(
    s_mem: &St<FSt, SR, SC>,
    rt_mem: &mut Rt<FRt, R, C>,
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
                rt_mem.data[rt_idx] = cast_line(s_mem.data[s_idx]);
            }
        }
    }
}

/// Cooperatively loads a tile from global memory into shared memory without transposing.
///
/// All threads in the workgroup participate, each loading multiple `Line<F>` vectors
/// in a strided pattern to maximize memory bandwidth. The shared memory uses a swizzled
/// layout to avoid bank conflicts during subsequent accesses.
///
/// # Arguments
/// * `g_mem` - Source tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `s_mem` - Destination shared memory tile (`St`)
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for this tile
/// * `g_offset_col` - Column offset within the matrix for this tile
#[cube]
pub fn load_st_direct<F: Float, R: Dim, C: Dim>(
    g_mem: &Tensor<Line<F>>,
    s_mem: &mut St<F, R, C>,
    base_offset: usize,
    g_offset_row: usize,
    g_offset_col: usize,
) {
    let vec_stride = C::LINES;
    let total_vecs = R::VALUE * vec_stride;
    let mask = vec_stride - 1;

    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    for i in range_stepped(tid, total_vecs, num_threads) {
        let r = i / vec_stride;
        let c_vec = i % vec_stride;

        // Global Coords
        let g_r = g_offset_row + r;
        let g_c = g_offset_col + (c_vec * LINE_SIZE);

        let val = load_safe(g_mem, base_offset, g_r, g_c);

        let phys_c = swizzle(r, c_vec, mask);
        let s_idx = (r * vec_stride) + phys_c;

        s_mem.data[s_idx] = val;
    }
}

/// Cooperatively loads a tile from global memory into shared memory while transposing.
///
/// The transpose is performed in `LINE_SIZE × LINE_SIZE` patches (4×4 by default).
/// Each thread processes entire patches: loading 4 consecutive rows from global memory,
/// transposing them via register shuffles, then storing as 4 columns to shared memory.
/// The shared memory uses a swizzled layout to avoid bank conflicts.
///
/// # Arguments
/// * `g_mem` - Source tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `s_mem` - Destination shared memory tile (`St`), will contain transposed data
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Row offset within the matrix for the source tile
/// * `g_offset_col` - Column offset within the matrix for the source tile
#[cube]
pub fn load_st_transpose<F: Float, R: Dim, C: Dim>(
    g_mem: &Tensor<Line<F>>,
    s_mem: &mut St<F, R, C>,
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
        let patch_r = i / patches_w; // Shared Row Index (in patches)
        let patch_c = i % patches_w; // Shared Col Index (in patches)

        let g_patch_r = patch_c;
        let g_patch_c = patch_r;

        let src_r = g_offset_row + (g_patch_r * LINE_SIZE);
        let src_c = g_offset_col + (g_patch_c * LINE_SIZE);

        let v0 = load_safe(g_mem, base_offset, src_r + 0, src_c);
        let v1 = load_safe(g_mem, base_offset, src_r + 1, src_c);
        let v2 = load_safe(g_mem, base_offset, src_r + 2, src_c);
        let v3 = load_safe(g_mem, base_offset, src_r + 3, src_c);

        let (t0, t1, t2, t3) = transpose_4(v0, v1, v2, v3);

        let dst_base_r = patch_r * LINE_SIZE;
        let dst_vec_c = patch_c;
        let mask = s_stride - 1;

        // We can't loop here because t0, t1, etc aren't
        // an array and CubeCL doesn't like arrays.

        let r0 = dst_base_r + 0;
        let phys_c0 = swizzle(r0, dst_vec_c, mask);
        s_mem.data[r0 * s_stride + phys_c0] = t0;

        let r1 = dst_base_r + 1;
        let phys_c1 = swizzle(r1, dst_vec_c, mask);
        s_mem.data[r1 * s_stride + phys_c1] = t1;

        let r2 = dst_base_r + 2;
        let phys_c2 = swizzle(r2, dst_vec_c, mask);
        s_mem.data[r2 * s_stride + phys_c2] = t2;

        let r3 = dst_base_r + 3;
        let phys_c3 = swizzle(r3, dst_vec_c, mask);
        s_mem.data[r3 * s_stride + phys_c3] = t3;
    }
}

/// Cooperatively loads per-thread register tiles from global memory without transposing.
///
/// Each thread loads its own register tile sub-region. Threads are mapped to sub-tiles
/// in row-major order based on UNIT_POS, matching `store_rt_direct`. Threads with
/// UNIT_POS >= num_sub_tiles are safely skipped (Rt unchanged).
///
/// Out-of-bounds reads return zeros, enabling safe loads at matrix boundaries.
///
/// # Type Parameters
/// * `R, C` - Register tile dimensions
/// * `SR, SC` - Full tile dimensions (determines thread mapping)
///
/// # Arguments
/// * `g_mem` - Source tensor in global memory (any rank, last 2 dims treated as matrix)
/// * `rt_mem` - Destination register tile (`Rt`), stored row-major as `[rows, cols/LINE_SIZE]` Lines
/// * `base_offset` - Scalar offset into `g_mem` from batch dimensions
/// * `g_offset_row` - Additional row offset within the matrix (usually 0)
/// * `g_offset_col` - Additional column offset within the matrix (usually 0)
#[cube]
pub fn load_rt_direct<F: Float, R: Dim, C: Dim, SR: Dim, SC: Dim>(
    g_mem: &Tensor<Line<F>>,
    rt_mem: &mut Rt<F, R, C>,
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
            #[unroll(C::LINES <= UNROLL_LIMIT)]
            for nl in 0..num_n_vecs {
                let g_c = g_offset_col + offset_col + nl * LINE_SIZE;
                let rt_idx = row * num_n_vecs + nl;

                if g_r < num_rows && g_c < num_cols {
                    let scalar_idx = base_offset + g_r * row_stride + g_c;
                    let line_idx = scalar_idx / LINE_SIZE;
                    rt_mem.data[rt_idx] = g_mem[line_idx];
                } else {
                    rt_mem.data[rt_idx] = Line::empty(LINE_SIZE).fill(F::from_int(0));
                }
            }
        }
    }
}

/// Loads a single `Line<F>` (vector of `LINE_SIZE` elements) from a global tensor.
///
/// Handles both row-major and column-major layouts:
/// - **Row-major** (`col_stride == 1`): Direct contiguous load of `LINE_SIZE` consecutive columns
/// - **Column-major**: Gathers `LINE_SIZE` scalars from non-contiguous locations
///
/// Returns zeros for out-of-bounds coordinates, enabling safe boundary handling.
///
/// # Arguments
/// * `g` - Source tensor (any rank, last 2 dims treated as matrix)
/// * `base_offset` - Pre-computed scalar offset from batch dimensions
/// * `r` - Row index within the matrix
/// * `c` - Starting column index (must be aligned to `LINE_SIZE` for row-major)
#[cube]
fn load_safe<F: Float>(g: &Tensor<Line<F>>, base_offset: usize, r: usize, c: usize) -> Line<F> {
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
            g[line_idx]
        } else {
            // Column-major: gather 4 scalars from different Lines
            let line_idx0 = scalar_idx / LINE_SIZE;
            let line_idx1 = (scalar_idx + col_stride) / LINE_SIZE;
            let line_idx2 = (scalar_idx + col_stride * 2) / LINE_SIZE;
            let line_idx3 = (scalar_idx + col_stride * 3) / LINE_SIZE;

            let elem0 = scalar_idx % LINE_SIZE;
            let elem1 = (scalar_idx + col_stride) % LINE_SIZE;
            let elem2 = (scalar_idx + col_stride * 2) % LINE_SIZE;
            let elem3 = (scalar_idx + col_stride * 3) % LINE_SIZE;

            let mut l = Line::empty(LINE_SIZE);
            l[0] = g[line_idx0][elem0];
            l[1] = g[line_idx1][elem1];
            l[2] = g[line_idx2][elem2];
            l[3] = g[line_idx3][elem3];
            l
        }
    } else {
        Line::empty(LINE_SIZE).fill(F::from_int(0))
    }
}

/// Loads a 1D vector from global memory into a shared memory vector (Sv).
///
/// For a 1D tensor with shape [L] and LINE_SIZE vectorization, this loads
/// L/LINE_SIZE Lines into the Sv<F, L> = St<F, L, D1>.
///
/// All threads participate cooperatively - each thread loads different Lines.
///
/// # Arguments
/// * `g_mem` - Source 1D tensor of Lines
/// * `s_mem` - Destination shared memory vector
/// * `base_offset` - Scalar offset into the source tensor
#[cube]
pub fn load_sv_direct<F: Float, L: Dim>(
    g_mem: &Tensor<Line<F>>,
    s_mem: &mut Sv<F, L>,
    base_offset: usize,
) {
    let num_lines = L::LINES;
    let num_threads = CUBE_DIM as usize;
    let tid = UNIT_POS as usize;

    for i in range_stepped(tid, num_lines, num_threads) {
        let line_idx = base_offset / LINE_SIZE + i;
        s_mem.data[i] = g_mem[line_idx];
    }
}
