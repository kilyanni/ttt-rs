#![allow(non_snake_case)]

use cubecl::prelude::*;

use crate::{
    cube::swizzle,
    prelude::*,
    tiles::Dim,
    util::{cast_line, write_into_line},
};

/// Indexer for accessing matrices with optional swizzle and transpose.
#[derive(CubeType, Clone, Copy)]
struct Indexer {
    stride: usize,
    mask: usize,
    #[cube(comptime)]
    transposed: bool,
    #[cube(comptime)]
    swizzled: bool,
}

#[cube]
impl Indexer {
    pub fn new(stride: usize, #[comptime] transposed: bool, #[comptime] swizzled: bool) -> Self {
        Indexer {
            stride,
            mask: stride - 1,
            transposed,
            swizzled,
        }
    }

    pub fn vec_index(&self, row: usize, col_vec: usize) -> usize {
        if comptime!(self.swizzled) {
            row * self.stride + swizzle(row, col_vec, self.mask)
        } else {
            row * self.stride + col_vec
        }
    }

    pub fn scalar_index(&self, row: usize, col: usize) -> (usize, usize) {
        if comptime!(self.transposed) {
            let row_line = row / LINE_SIZE;
            let row_elem = row % LINE_SIZE;
            if comptime!(self.swizzled) {
                (
                    col * self.stride + swizzle(col, row_line, self.mask),
                    row_elem,
                )
            } else {
                (col * self.stride + row_line, row_elem)
            }
        } else {
            let col_line = col / LINE_SIZE;
            let col_elem = col % LINE_SIZE;
            if comptime!(self.swizzled) {
                (
                    row * self.stride + swizzle(row, col_line, self.mask),
                    col_elem,
                )
            } else {
                (row * self.stride + col_line, col_elem)
            }
        }
    }
}

/// Accumulate along N: B is vectorized, C is row-major.
#[cube]
fn accum_n<FIn: Float, FAcc: Float, RtR: Dim, RtC: Dim>(
    c: &mut Rt<FAcc, RtR, RtC>,
    a_data: &SharedMemory<Line<FIn>>,
    b_data: &SharedMemory<Line<FIn>>,
    a_idx: &Indexer,
    b_idx: &Indexer,
    c_idx: &Indexer,
    k: usize,
    offset_m: usize,
    offset_n: usize,
) {
    let m_lines = if comptime!(c_idx.transposed) {
        RtC::LINES
    } else {
        RtR::LINES
    };
    let n_lines = if comptime!(c_idx.transposed) {
        RtR::LINES
    } else {
        RtC::LINES
    };

    #[unroll(n_lines <= UNROLL_LIMIT_HOT)]
    for nl in 0..n_lines {
        let b_vec_in = b_data[b_idx.vec_index(k, offset_n + nl)];
        let b_vec = cast_line::<FIn, FAcc>(b_vec_in);

        if comptime!(a_idx.transposed) {
            #[unroll(m_lines <= UNROLL_LIMIT_HOT)]
            for ml in 0..m_lines {
                let a_vec_in = a_data[a_idx.vec_index(k, offset_m + ml)];
                #[unroll]
                for mi in 0..LINE_SIZE {
                    let a_val = FAcc::cast_from(a_vec_in[mi]);
                    let c_row = ml * LINE_SIZE + mi;
                    let c_line = c_idx.vec_index(c_row, nl);
                    c.data[c_line] += Line::empty(LINE_SIZE).fill(a_val) * b_vec;
                }
            }
        } else {
            #[unroll(m_lines <= UNROLL_LIMIT_HOT)]
            for ml in 0..m_lines {
                #[unroll]
                for mi in 0..LINE_SIZE {
                    let m = (offset_m + ml) * LINE_SIZE + mi;
                    let (a_line, a_elem) = a_idx.scalar_index(m, k);
                    let a_val = FAcc::cast_from(a_data[a_line][a_elem]);
                    let c_row = ml * LINE_SIZE + mi;
                    let c_line = c_idx.vec_index(c_row, nl);
                    c.data[c_line] += Line::empty(LINE_SIZE).fill(a_val) * b_vec;
                }
            }
        }
    }
}

/// Accumulate along M: A is vectorized, C is column-major.
#[cube]
fn accum_m<FIn: Float, FAcc: Float, RtR: Dim, RtC: Dim>(
    c: &mut Rt<FAcc, RtR, RtC>,
    a_data: &SharedMemory<Line<FIn>>,
    b_data: &SharedMemory<Line<FIn>>,
    a_idx: &Indexer,
    b_idx: &Indexer,
    c_idx: &Indexer,
    k: usize,
    offset_m: usize,
    offset_n: usize,
) {
    let m_lines = if comptime!(c_idx.transposed) {
        RtC::LINES
    } else {
        RtR::LINES
    };
    let n_lines = if comptime!(c_idx.transposed) {
        RtR::LINES
    } else {
        RtC::LINES
    };

    #[unroll(m_lines <= UNROLL_LIMIT_HOT)]
    for ml in 0..m_lines {
        let a_vec_in = a_data[a_idx.vec_index(k, offset_m + ml)];
        let a_vec = cast_line::<FIn, FAcc>(a_vec_in);

        if comptime!(b_idx.transposed) {
            #[unroll(n_lines <= UNROLL_LIMIT_HOT)]
            for nl in 0..n_lines {
                let b_vec_in = b_data[b_idx.vec_index(k, offset_n + nl)];
                #[unroll]
                for ni in 0..LINE_SIZE {
                    let b_val = FAcc::cast_from(b_vec_in[ni]);
                    let c_col = nl * LINE_SIZE + ni;
                    let c_line = c_idx.vec_index(c_col, ml);
                    c.data[c_line] += a_vec * Line::empty(LINE_SIZE).fill(b_val);
                }
            }
        } else {
            #[unroll(n_lines <= UNROLL_LIMIT_HOT)]
            for nl in 0..n_lines {
                #[unroll]
                for ni in 0..LINE_SIZE {
                    let n = (offset_n + nl) * LINE_SIZE + ni;
                    let (b_line, b_elem) = b_idx.scalar_index(n, k);
                    let b_val = FAcc::cast_from(b_data[b_line][b_elem]);
                    let c_col = nl * LINE_SIZE + ni;
                    let c_line = c_idx.vec_index(c_col, ml);
                    c.data[c_line] += a_vec * Line::empty(LINE_SIZE).fill(b_val);
                }
            }
        }
    }
}

/// Scalar accumulation fallback.
#[cube]
fn accum_scalar<FIn: Float, FAcc: Float, RtR: Dim, RtC: Dim>(
    c: &mut Rt<FAcc, RtR, RtC>,
    a_data: &SharedMemory<Line<FIn>>,
    b_data: &SharedMemory<Line<FIn>>,
    a_idx: &Indexer,
    b_idx: &Indexer,
    c_idx: &Indexer,
    k: usize,
    offset_m: usize,
    offset_n: usize,
) {
    let m_lines = if comptime!(c_idx.transposed) {
        RtC::LINES
    } else {
        RtR::LINES
    };
    let n_lines = if comptime!(c_idx.transposed) {
        RtR::LINES
    } else {
        RtC::LINES
    };

    #[unroll(m_lines <= UNROLL_LIMIT_HOT)]
    for ml in 0..m_lines {
        #[unroll]
        for mi in 0..LINE_SIZE {
            let a_val: FAcc = if comptime!(a_idx.transposed) {
                let a_vec = a_data[a_idx.vec_index(k, offset_m + ml)];
                FAcc::cast_from(a_vec[mi])
            } else {
                let m = (offset_m + ml) * LINE_SIZE + mi;
                let (a_line, a_elem) = a_idx.scalar_index(m, k);
                FAcc::cast_from(a_data[a_line][a_elem])
            };

            let c_row = ml * LINE_SIZE + mi;

            #[unroll(n_lines <= UNROLL_LIMIT_HOT)]
            for nl in 0..n_lines {
                #[unroll]
                for ni in 0..LINE_SIZE {
                    let b_val: FAcc = if comptime!(b_idx.transposed) {
                        let b_vec = b_data[b_idx.vec_index(k, offset_n + nl)];
                        FAcc::cast_from(b_vec[ni])
                    } else {
                        let n = (offset_n + nl) * LINE_SIZE + ni;
                        let (b_line, b_elem) = b_idx.scalar_index(n, k);
                        FAcc::cast_from(b_data[b_line][b_elem])
                    };

                    let c_col = nl * LINE_SIZE + ni;
                    let (c_line, c_elem) = c_idx.scalar_index(c_row, c_col);
                    let current = c.data[c_line][c_elem];
                    let new_val = current + a_val * b_val;
                    write_into_line(
                        c.data.to_slice_mut().slice_mut(c_line, c_line + 1),
                        c_elem,
                        new_val,
                    );
                }
            }
        }
    }
}

/// Generates mma_rt variants for all transpose combinations.
///
/// Naming: mma_rt_A{,t}B{,t}{,_t}
/// - At = A transposed, stored as [K, M]; A = not transposed, stored as [M, K]
/// - Bt = B transposed, stored as [K, N]; B = not transposed, stored as [N, K]
/// - _t suffix = C column-major (transposed); no suffix = C row-major
///
/// CM, CN: register tile dimensions (can be smaller than St tile)
/// TileM, TileK, TileN: shared tile dimensions
/// offset_m, offset_n: select which sub-tile of the St to operate on
///
/// FIn: input element type (A and B matrices)
/// FAcc: accumulator element type (C matrix)
macro_rules! define_mma_rt {
    ($name:ident, $a_trans:tt, $b_trans:tt, $c_trans:tt,
     [$a_d0:ident, $a_d1:ident], [$b_d0:ident, $b_d1:ident], $accum:ident,
     [$c_d0:ident, $c_d1:ident]) => {
        #[cube]
        pub fn $name<
            FIn: Float,
            FAcc: Float,
            CM: Dim,
            CN: Dim,
            TileM: Dim,
            TileK: Dim,
            TileN: Dim,
        >(
            c: &mut Rt<FAcc, $c_d0, $c_d1>,
            a: &St<FIn, $a_d0, $a_d1>,
            b: &St<FIn, $b_d0, $b_d1>,
            offset_m: usize,
            offset_n: usize,
        ) {
            let a_idx = Indexer::new($a_d1::LINES, $a_trans, true);
            let b_idx = Indexer::new($b_d1::LINES, $b_trans, true);
            let c_stride = if comptime!($c_trans) {
                CM::LINES
            } else {
                CN::LINES
            };
            let c_idx = Indexer::new(c_stride, $c_trans, false);

            for k in 0..TileK::VALUE {
                $accum::<FIn, FAcc, $c_d0, $c_d1>(
                    c, &a.data, &b.data, &a_idx, &b_idx, &c_idx, k, offset_m, offset_n,
                );
            }
        }
    };
}

// A: a_trans=false → [M,K], a_trans=true → [K,M]
// B: b_trans=false → [N,K], b_trans=true → [K,N]
// C: c_trans=false → Rt<CM,CN>, c_trans=true → Rt<CN,CM>
// Naming: A=[M,K], At=[K,M], B=[K,N], Bt=[N,K]
// Accum strategy: b_trans && !c_trans → accum_n, a_trans && c_trans → accum_m, else → accum_scalar

define_mma_rt!(
    mma_rt_ABt,
    false,
    false,
    false,
    [TileM, TileK],
    [TileN, TileK],
    accum_scalar,
    [CM, CN]
);
define_mma_rt!(
    mma_rt_ABt_t,
    false,
    false,
    true,
    [TileM, TileK],
    [TileN, TileK],
    accum_scalar,
    [CN, CM]
);
define_mma_rt!(
    mma_rt_AB,
    false,
    true,
    false,
    [TileM, TileK],
    [TileK, TileN],
    accum_n,
    [CM, CN]
);
define_mma_rt!(
    mma_rt_AB_t,
    false,
    true,
    true,
    [TileM, TileK],
    [TileK, TileN],
    accum_scalar,
    [CN, CM]
);
define_mma_rt!(
    mma_rt_AtBt,
    true,
    false,
    false,
    [TileK, TileM],
    [TileN, TileK],
    accum_scalar,
    [CM, CN]
);
define_mma_rt!(
    mma_rt_AtBt_t,
    true,
    false,
    true,
    [TileK, TileM],
    [TileN, TileK],
    accum_m,
    [CN, CM]
);
define_mma_rt!(
    mma_rt_AtB,
    true,
    true,
    false,
    [TileK, TileM],
    [TileK, TileN],
    accum_n,
    [CM, CN]
);
define_mma_rt!(
    mma_rt_AtB_t,
    true,
    true,
    true,
    [TileK, TileM],
    [TileK, TileN],
    accum_m,
    [CN, CM]
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cube::{load_st_direct, load_st_transpose},
        test_kernel,
    };

    fn reference_matmul<F: crate::test_utils::TestFloat>(
        in_a: &[F],
        in_b: &[F],
        output: &mut [F],
        m: usize,
        k: usize,
        n: usize,
    ) {
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = F::from_f64(0.0);
                for ki in 0..k {
                    sum = F::from_f64(
                        sum.into_f64()
                            + in_a[mi * k + ki].into_f64() * in_b[ki * n + ni].into_f64(),
                    );
                }
                output[mi * n + ni] = sum;
            }
        }
    }

    macro_rules! define_test_kernel {
        ($name:ident, $mma_fn:ident, $c_trans:tt,
         [$a_d0:ident, $a_d1:ident], $load_a:ident,
         [$b_d0:ident, $b_d1:ident], $load_b:ident) => {
            define_test_kernel!(@impl $name, $mma_fn, $c_trans,
                [$a_d0, $a_d1], $load_a, [$b_d0, $b_d1], $load_b);
        };

        (@impl $name:ident, $mma_fn:ident, false,
         [$a_d0:ident, $a_d1:ident], $load_a:ident,
         [$b_d0:ident, $b_d1:ident], $load_b:ident) => {
            #[cube(launch)]
            fn $name<F: Float, TileM: Dim, TileK: Dim, TileN: Dim>(
                in_a: &Tensor<Line<F>>,
                in_b: &Tensor<Line<F>>,
                output: &mut Array<Line<F>>,
            ) {
                let mut st_a = St::<F, $a_d0, $a_d1>::new();
                let mut st_b = St::<F, $b_d0, $b_d1>::new();
                $load_a(in_a, &mut st_a, 0, 0, 0);
                $load_b(in_b, &mut st_b, 0, 0, 0);

                let mut rt_c = Rt::<F, TileM, TileN>::new();
                rt_c.zero();

                // CM=TileM, CN=TileN (full tile per thread in this test)
                // Using F, F for homogeneous types (backward compatible)
                $mma_fn::<F, F, TileM, TileN, TileM, TileK, TileN>(&mut rt_c, &st_a, &st_b, 0, 0);
                rt_c.copy_to_array(output);
            }
        };

        (@impl $name:ident, $mma_fn:ident, true,
         [$a_d0:ident, $a_d1:ident], $load_a:ident,
         [$b_d0:ident, $b_d1:ident], $load_b:ident) => {
            #[cube(launch)]
            fn $name<F: Float, TileM: Dim, TileK: Dim, TileN: Dim>(
                in_a: &Tensor<Line<F>>,
                in_b: &Tensor<Line<F>>,
                output: &mut Array<Line<F>>,
            ) {
                let mut st_a = St::<F, $a_d0, $a_d1>::new();
                let mut st_b = St::<F, $b_d0, $b_d1>::new();
                $load_a(in_a, &mut st_a, 0, 0, 0);
                $load_b(in_b, &mut st_b, 0, 0, 0);

                        // c_trans=true: use Rt<N,M>, then transpose to Rt<M,N>
                let mut rt_c = Rt::<F, TileN, TileM>::new();
                rt_c.zero();

                // Using F, F for homogeneous types (backward compatible)
                $mma_fn::<F, F, TileM, TileN, TileM, TileK, TileN>(&mut rt_c, &st_a, &st_b, 0, 0);
                rt_c.transpose().copy_to_array(output);
            }
        };
    }

    // a_trans=false, b_trans=false: A=[M,K] direct, B=[K,N] transpose to [N,K]
    define_test_kernel!(
        test_kernel_ABt,
        mma_rt_ABt,
        false,
        [TileM, TileK],
        load_st_direct,
        [TileN, TileK],
        load_st_transpose
    );
    define_test_kernel!(
        test_kernel_ABt_t,
        mma_rt_ABt_t,
        true,
        [TileM, TileK],
        load_st_direct,
        [TileN, TileK],
        load_st_transpose
    );

    // a_trans=false, b_trans=true: A=[M,K] direct, B=[K,N] direct
    define_test_kernel!(
        test_kernel_AB,
        mma_rt_AB,
        false,
        [TileM, TileK],
        load_st_direct,
        [TileK, TileN],
        load_st_direct
    );
    define_test_kernel!(
        test_kernel_AB_t,
        mma_rt_AB_t,
        true,
        [TileM, TileK],
        load_st_direct,
        [TileK, TileN],
        load_st_direct
    );

    // a_trans=true, b_trans=false: A=[M,K] transpose to [K,M], B=[K,N] transpose to [N,K]
    define_test_kernel!(
        test_kernel_AtBt,
        mma_rt_AtBt,
        false,
        [TileK, TileM],
        load_st_transpose,
        [TileN, TileK],
        load_st_transpose
    );
    define_test_kernel!(
        test_kernel_AtBt_t,
        mma_rt_AtBt_t,
        true,
        [TileK, TileM],
        load_st_transpose,
        [TileN, TileK],
        load_st_transpose
    );

    // a_trans=true, b_trans=true: A=[M,K] transpose to [K,M], B=[K,N] direct
    define_test_kernel!(
        test_kernel_AtB,
        mma_rt_AtB,
        false,
        [TileK, TileM],
        load_st_transpose,
        [TileK, TileN],
        load_st_direct
    );
    define_test_kernel!(
        test_kernel_AtB_t,
        mma_rt_AtB_t,
        true,
        [TileK, TileM],
        load_st_transpose,
        [TileK, TileN],
        load_st_direct
    );

    macro_rules! define_test {
        ($test_name:ident, $kernel:ident) => {
            test_kernel! {
                #[test]
                fn $test_name() for F in [f32, f64] TileM in [D4, D8] TileK in [D4, D8] TileN in [D4, D8] {
                    let in_a: Tensor = [TileM::VALUE, TileK::VALUE] as Range;
                    let in_b: Tensor = [TileK::VALUE, TileN::VALUE] as Range;
                    let output: Array = [TileM::VALUE * TileN::VALUE];
                    assert_eq!(
                        $kernel(in_a(), in_b(), output()) for (1, 1, 1) @ (1),
                        { reference_matmul(&in_a, &in_b, &mut output, TileM::VALUE, TileK::VALUE, TileN::VALUE); }
                    );
                }
            }
        };
    }

    define_test!(test_mma_AB, test_kernel_AB);
    define_test!(test_mma_AB_t, test_kernel_AB_t);
    define_test!(test_mma_ABt, test_kernel_ABt);
    define_test!(test_mma_ABt_t, test_kernel_ABt_t);
    define_test!(test_mma_AtB, test_kernel_AtB);
    define_test!(test_mma_AtB_t, test_kernel_AtB_t);
    define_test!(test_mma_AtBt, test_kernel_AtBt);
    define_test!(test_mma_AtBt_t, test_kernel_AtBt_t);
}
