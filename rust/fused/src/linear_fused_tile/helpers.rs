#![allow(type_alias_bounds)]

use cubecl::prelude::*;
use thundercube::prelude::*;

// Shared memory tiles (St) - use EVal (value type)
pub type StCsF<P: ParamsTrait> = St<P::EVal, P::CS, P::F>;
pub type StFCs<P: ParamsTrait> = St<P::EVal, P::F, P::CS>;
pub type StCsCs<P: ParamsTrait> = St<P::EVal, P::CS, P::CS>;
pub type StFF<P: ParamsTrait> = St<P::EVal, P::F, P::F>;

// Shared memory vectors (Sv) - use EVal
pub type SvCs<P: ParamsTrait> = Sv<P::EVal, P::CS>;
pub type SvF<P: ParamsTrait> = Sv<P::EVal, P::F>;

// Register tiles (Rt) - use EAcc (accumulator type)
pub type RtCsCs<P: ParamsTrait> = Rt<P::EAcc, P::CS_Reg, P::CS_Reg>;
pub type RtCsF<P: ParamsTrait> = Rt<P::EAcc, P::CS_Reg, P::F_Reg>;
pub type RtFF<P: ParamsTrait> = Rt<P::EAcc, P::F_Reg, P::F_Reg>;

// Register vectors (Rv) - thread-local size, EAcc
pub type RvCs<P: ParamsTrait> = Rv<P::EAcc, P::CS_Reg>;
pub type RvF<P: ParamsTrait> = Rv<P::EAcc, P::F_Reg>;

// Register vectors broadcast (Rvb) - full size, EAcc for computation
pub type RvbCsA<P: ParamsTrait> = Rv<P::EAcc, P::CS>;
pub type RvbFA<P: ParamsTrait> = Rv<P::EAcc, P::F>;

// Register vectors broadcast (Rvb) - full size, EVal for parameters
pub type RvbCsV<P: ParamsTrait> = Rv<P::EVal, P::CS>;
pub type RvbFV<P: ParamsTrait> = Rv<P::EVal, P::F>;

#[cube]
pub trait ParamsTrait: Send + Sync + 'static {
    /// Value type for shared memory and I/O (e.g., f16 for reduced memory)
    type EVal: Float;
    /// Accumulator type for registers (e.g., f32 for precision)
    type EAcc: Float;
    type CS: Dim;
    type F: Dim;

    type CS_Reg: Dim;
    type F_Reg: Dim;

    // CubeCL won't let us do default impls
    // Shared memory tiles
    fn st_cs_f() -> StCsF<Self>;
    fn st_f_cs() -> StFCs<Self>;
    fn st_cs_cs() -> StCsCs<Self>;
    fn st_ff() -> StFF<Self>;

    // Shared memory vectors
    fn sv_cs() -> SvCs<Self>;
    fn sv_f() -> SvF<Self>;

    // Register tiles
    fn rt_cs_cs() -> RtCsCs<Self>;
    fn rt_cs_f() -> RtCsF<Self>;
    fn rt_ff() -> RtFF<Self>;

    // Register vectors (thread-local)
    fn rv_cs() -> RvCs<Self>;
    fn rv_f() -> RvF<Self>;

    // Register vectors broadcast (full size)
    fn rvb_cs_a() -> RvbCsA<Self>;
    fn rvb_f_a() -> RvbFA<Self>;
    fn rvb_cs_v() -> RvbCsV<Self>;
    fn rvb_f_v() -> RvbFV<Self>;
}

pub struct Params<EVal: Float, EAcc: Float, CS: Dim, F: Dim, CS_Reg: Dim, F_Reg: Dim> {
    _phantom: std::marker::PhantomData<(EVal, EAcc, CS, F, CS_Reg, F_Reg)>,
}

#[cube]
impl<EVal: Float, EAcc: Float, CS: Dim, F: Dim, CS_Reg: Dim, F_Reg: Dim> ParamsTrait
    for Params<EVal, EAcc, CS, F, CS_Reg, F_Reg>
{
    type EVal = EVal;
    type EAcc = EAcc;
    type CS = CS;
    type F = F;
    type CS_Reg = CS_Reg;
    type F_Reg = F_Reg;

    fn st_cs_f() -> StCsF<Self> {
        St::new()
    }
    fn st_f_cs() -> StFCs<Self> {
        St::new()
    }
    fn st_cs_cs() -> StCsCs<Self> {
        St::new()
    }
    fn st_ff() -> StFF<Self> {
        St::new()
    }

    fn sv_cs() -> SvCs<Self> {
        Sv::new()
    }
    fn sv_f() -> SvF<Self> {
        Sv::new()
    }

    fn rt_cs_cs() -> RtCsCs<Self> {
        Rt::new()
    }
    fn rt_cs_f() -> RtCsF<Self> {
        Rt::new()
    }
    fn rt_ff() -> RtFF<Self> {
        Rt::new()
    }

    fn rv_cs() -> RvCs<Self> {
        Rv::new()
    }
    fn rv_f() -> RvF<Self> {
        Rv::new()
    }

    fn rvb_cs_a() -> RvbCsA<Self> {
        Rv::new()
    }
    fn rvb_f_a() -> RvbFA<Self> {
        Rv::new()
    }
    fn rvb_cs_v() -> RvbCsV<Self> {
        Rv::new()
    }
    fn rvb_f_v() -> RvbFV<Self> {
        Rv::new()
    }
}

/// Apply lower triangular mask to a CS×CS register tile.
/// Zeros elements where col > row (keeps lower triangle including diagonal).
/// Each thread masks its own sub-tile based on UNIT_POS.
#[cube]
pub fn rt_tril<P: ParamsTrait>(rt: &mut RtCsCs<P>) {
    let tiles_per_row = P::CS::VALUE / P::CS_Reg::VALUE;
    let tile_row = (UNIT_POS as usize) / tiles_per_row;
    let tile_col = (UNIT_POS as usize) % tiles_per_row;

    if tile_col > tile_row {
        // Entire sub-tile is above diagonal — zero all
        rt.zero();
    } else if tile_col == tile_row {
        // Diagonal block — per-element masking
        let num_c_vecs = P::CS_Reg::LINES;
        let zero = P::EAcc::new(0.0);

        #[unroll]
        for row in 0..P::CS_Reg::VALUE {
            let global_row = tile_row * P::CS_Reg::VALUE + row;

            #[unroll]
            for cv in 0..P::CS_Reg::LINES {
                let c_base = tile_col * P::CS_Reg::VALUE + cv * LINE_SIZE;
                let rt_idx = row * num_c_vecs + cv;
                let mut line = rt.data[rt_idx];

                if c_base + 0 > global_row {
                    line[0] = zero;
                }
                if c_base + 1 > global_row {
                    line[1] = zero;
                }
                if c_base + 2 > global_row {
                    line[2] = zero;
                }
                if c_base + 3 > global_row {
                    line[3] = zero;
                }

                rt.data[rt_idx] = line;
            }
        }
    }
    // else: tile_col < tile_row → below diagonal, keep all
}

/// Apply upper triangular mask to a CS×CS register tile.
/// Zeros elements where col < row (keeps upper triangle including diagonal).
/// Each thread masks its own sub-tile based on UNIT_POS.
#[cube]
pub fn rt_triu<P: ParamsTrait>(rt: &mut RtCsCs<P>) {
    let tiles_per_row = P::CS::VALUE / P::CS_Reg::VALUE;
    let tile_row = (UNIT_POS as usize) / tiles_per_row;
    let tile_col = (UNIT_POS as usize) % tiles_per_row;

    if tile_col < tile_row {
        // Entire sub-tile is below diagonal — zero all
        rt.zero();
    } else if tile_col == tile_row {
        // Diagonal block — per-element masking
        let num_c_vecs = P::CS_Reg::LINES;
        let zero = P::EAcc::new(0.0);

        #[unroll]
        for row in 0..P::CS_Reg::VALUE {
            let global_row = tile_row * P::CS_Reg::VALUE + row;

            #[unroll]
            for cv in 0..P::CS_Reg::LINES {
                let c_base = tile_col * P::CS_Reg::VALUE + cv * LINE_SIZE;
                let rt_idx = row * num_c_vecs + cv;
                let mut line = rt.data[rt_idx];

                if c_base + 0 < global_row {
                    line[0] = zero;
                }
                if c_base + 1 < global_row {
                    line[1] = zero;
                }
                if c_base + 2 < global_row {
                    line[2] = zero;
                }
                if c_base + 3 < global_row {
                    line[3] = zero;
                }

                rt.data[rt_idx] = line;
            }
        }
    }
    // else: tile_col > tile_row → above diagonal, keep all
}

/// Build eta matrix: η[i,j] = token_eta[i] * ttt_lr_eta[j], with triangular mask.
///
/// When `transposed` is false: builds lower triangular η (tril)
/// When `transposed` is true: builds upper triangular η^T (triu)
#[cube]
pub fn build_eta_matrix<P: ParamsTrait>(
    token_eta: &Tensor<Line<P::EVal>>,
    ttt_lr_eta: &Tensor<Line<P::EVal>>,
    output: &mut StCsCs<P>,
    ttt_lr_eta_idx: usize,
    #[comptime] transposed: bool,
) {
    let tiles_per_row = P::CS::VALUE / P::CS_Reg::VALUE;
    let tile_row = (UNIT_POS as usize) / tiles_per_row;
    let tile_col = (UNIT_POS as usize) % tiles_per_row;

    let mut eta_reg = P::rt_cs_cs();
    eta_reg.zero();

    let mut row_vec = P::rv_cs();
    let mut col_vec = P::rv_cs();

    if comptime!(transposed) {
        // η^T[i,j] = ttt_lr_eta[i] * token_eta[j]
        cube::broadcast::load_rv_direct(
            ttt_lr_eta,
            &mut row_vec,
            ttt_lr_eta_idx + tile_row * P::CS_Reg::VALUE,
        );
        cube::broadcast::load_rv_direct(token_eta, &mut col_vec, tile_col * P::CS_Reg::VALUE);
    } else {
        // η[i,j] = token_eta[i] * ttt_lr_eta[j]
        cube::broadcast::load_rv_direct(token_eta, &mut row_vec, tile_row * P::CS_Reg::VALUE);
        cube::broadcast::load_rv_direct(
            ttt_lr_eta,
            &mut col_vec,
            ttt_lr_eta_idx + tile_col * P::CS_Reg::VALUE,
        );
    }

    eta_reg.add_col(&row_vec);
    eta_reg.mul_row(&col_vec);

    // Apply triangular mask in registers before storing
    if comptime!(transposed) {
        rt_triu::<P>(&mut eta_reg);
    } else {
        rt_tril::<P>(&mut eta_reg);
    }

    cube::store_rt_to_st(&eta_reg, output);

    sync_cube();
}

/// Compute attention matrix: attn = XQ @ XK^T, with triangular mask.
///
/// When `transposed` is false: builds lower triangular attn (tril)
/// When `transposed` is true: builds upper triangular attn^T (triu)
#[cube]
pub fn build_attn_matrix<P: ParamsTrait>(
    q_smem: &StFCs<P>,
    k_smem: &StFCs<P>,
    output: &mut StCsCs<P>,
    #[comptime] transposed: bool,
) {
    let mut attn_reg = P::rt_cs_cs();
    attn_reg.zero();

    if comptime!(transposed) {
        // attn^T = XK @ XQ^T = k_smem^T @ q_smem
        cube::mma_AtB(&mut attn_reg, k_smem, q_smem);
    } else {
        // attn = XQ @ XK^T = q_smem^T @ k_smem
        cube::mma_AtB(&mut attn_reg, q_smem, k_smem);
    }

    if comptime!(transposed) {
        rt_triu::<P>(&mut attn_reg);
    } else {
        rt_tril::<P>(&mut attn_reg);
    }

    cube::store_rt_to_st(&attn_reg, output);

    sync_cube();
}

/// Compute fused (eta * attn) matrix directly in registers, avoiding separate attn tile.
///
/// Computes: output[i,j] = token_eta[i] * ttt_lr_eta[j] * (q[i] · k[j])
///
/// This fuses build_eta_matrix and build_attn_matrix, computing the element-wise
/// product in registers before storing to shared memory. Saves one CS×CS tile.
#[cube]
pub fn build_eta_attn_fused<P: ParamsTrait>(
    q_smem: &StFCs<P>,
    k_smem: &StFCs<P>,
    token_eta: &Tensor<Line<P::EVal>>,
    ttt_lr_eta: &Tensor<Line<P::EVal>>,
    output: &mut StCsCs<P>,
    ttt_lr_eta_idx: usize,
) {
    // Compute attn = q^T @ k in registers
    let mut attn_reg = P::rt_cs_cs();
    attn_reg.zero();
    cube::mma_AtB(&mut attn_reg, q_smem, k_smem);

    // MMA only reads q_smem/k_smem; subsequent ops are register-only until store to output
    // Compute eta values and multiply with attn in registers
    let tiles_per_row = P::CS::VALUE / P::CS_Reg::VALUE;
    let tile_row = (UNIT_POS as usize) / tiles_per_row;
    let tile_col = (UNIT_POS as usize) % tiles_per_row;

    // Load eta components: η[i,j] = token_eta[i] * ttt_lr_eta[j]
    let mut row_vec = P::rv_cs();
    let mut col_vec = P::rv_cs();
    cube::broadcast::load_rv_direct(token_eta, &mut row_vec, tile_row * P::CS_Reg::VALUE);
    cube::broadcast::load_rv_direct(
        ttt_lr_eta,
        &mut col_vec,
        ttt_lr_eta_idx + tile_col * P::CS_Reg::VALUE,
    );

    // Build eta in registers and multiply with attn
    let mut eta_reg = P::rt_cs_cs();
    eta_reg.zero();
    eta_reg.add_col(&row_vec);
    eta_reg.mul_row(&col_vec);

    // Element-wise multiply: result = eta * attn
    attn_reg.mul(&eta_reg);

    rt_tril::<P>(&mut attn_reg);

    cube::store_rt_to_st(&attn_reg, output);

    sync_cube();
}

// TODO: Move to thundercube and abstract?
/// Extract the last row of a shared memory tile into a register vector.
/// Casts from shared memory type (FVal) to register type (FAcc).
/// Result is broadcast to all.
#[cube]
#[must_use]
pub fn extract_last_row<FVal: Float, FAcc: Float, R: Dim, C: Dim>(
    st: &St<FVal, R, C>,
) -> Rv<FAcc, C> {
    let mut result = Rv::<FAcc, C>::new();
    let last_row = R::VALUE - 1;
    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll]
    for c_line in 0..C::LINES {
        let phys_col = cube::swizzle(last_row, c_line, mask);
        let s_idx = last_row * vec_stride + phys_col;
        result.data[c_line] = thundercube::util::cast_line(st.data[s_idx]);
    }
    result
}
