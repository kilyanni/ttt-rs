#![allow(non_snake_case)]

use cubecl::prelude::*;

use crate::{
    prelude::*,
    tiles::{Dim, mma::*},
};

/// Generates plane-level MMA functions for all transpose combinations.
///
/// Each thread computes a sub-tile of C based on its UNIT_POS.
/// The tile is divided into (TileM / ThreadTileM) × (TileN / ThreadTileN) sub-tiles.
///
/// FIn: input element type (A and B matrices)
/// FAcc: accumulator element type (C matrix)
macro_rules! define_plane_mma {
    ($name:ident, $mma_rt_fn:ident,
     [$a_d0:ident, $a_d1:ident], [$b_d0:ident, $b_d1:ident],
     [$rt_d0:ident, $rt_d1:ident]) => {
        #[cube]
        pub fn $name<
            FIn: Float,
            FAcc: Float,
            TileM: Dim,
            TileK: Dim,
            TileN: Dim,
            ThreadTileM: Dim,
            ThreadTileN: Dim,
        >(
            rt_c: &mut Rt<FAcc, $rt_d0, $rt_d1>,
            st_a: &St<FIn, $a_d0, $a_d1>,
            st_b: &St<FIn, $b_d0, $b_d1>,
        ) {
            let threads_m = TileM::VALUE / ThreadTileM::VALUE;
            let threads_n = TileN::VALUE / ThreadTileN::VALUE;
            let num_tiles = threads_m * threads_n;

            if (UNIT_POS as usize) < num_tiles {
                let tid = UNIT_POS as usize;
                let thread_m = tid / threads_n;
                let thread_n = tid % threads_n;

                let offset_m = ThreadTileM::LINES * thread_m;
                let offset_n = ThreadTileN::LINES * thread_n;

                $mma_rt_fn::<FIn, FAcc, ThreadTileM, ThreadTileN, TileM, TileK, TileN>(
                    rt_c, st_a, st_b, offset_m, offset_n,
                );
            }
        }
    };
}

// A: a_trans=false → [M,K], a_trans=true → [K,M]
// B: b_trans=false → [N,K], b_trans=true → [K,N]
// C: c_trans=false → Rt<M,N>, c_trans=true → Rt<N,M>
// Naming: A=[M,K], At=[K,M], B=[K,N], Bt=[N,K]

define_plane_mma!(
    mma_ABt,
    mma_rt_ABt,
    [TileM, TileK],
    [TileN, TileK],
    [ThreadTileM, ThreadTileN]
);
define_plane_mma!(
    mma_ABt_t,
    mma_rt_ABt_t,
    [TileM, TileK],
    [TileN, TileK],
    [ThreadTileN, ThreadTileM]
);
define_plane_mma!(
    mma_AB,
    mma_rt_AB,
    [TileM, TileK],
    [TileK, TileN],
    [ThreadTileM, ThreadTileN]
);
define_plane_mma!(
    mma_AB_t,
    mma_rt_AB_t,
    [TileM, TileK],
    [TileK, TileN],
    [ThreadTileN, ThreadTileM]
);
define_plane_mma!(
    mma_AtBt,
    mma_rt_AtBt,
    [TileK, TileM],
    [TileN, TileK],
    [ThreadTileM, ThreadTileN]
);
define_plane_mma!(
    mma_AtBt_t,
    mma_rt_AtBt_t,
    [TileK, TileM],
    [TileN, TileK],
    [ThreadTileN, ThreadTileM]
);
define_plane_mma!(
    mma_AtB,
    mma_rt_AtB,
    [TileK, TileM],
    [TileK, TileN],
    [ThreadTileM, ThreadTileN]
);
define_plane_mma!(
    mma_AtB_t,
    mma_rt_AtB_t,
    [TileK, TileM],
    [TileK, TileN],
    [ThreadTileN, ThreadTileM]
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        cube::{load_st_direct, load_st_transpose, store_rt_direct},
        test_kernel,
        tiles::Rt,
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

    macro_rules! define_plane_test_kernel {
        ($name:ident, $mma_fn:ident, $c_trans:tt,
         [$a_d0:ident, $a_d1:ident], $load_a:ident,
         [$b_d0:ident, $b_d1:ident], $load_b:ident) => {
            define_plane_test_kernel!(@impl $name, $mma_fn, $c_trans,
                [$a_d0, $a_d1], $load_a, [$b_d0, $b_d1], $load_b);
        };

        (@impl $name:ident, $mma_fn:ident, false,
         [$a_d0:ident, $a_d1:ident], $load_a:ident,
         [$b_d0:ident, $b_d1:ident], $load_b:ident) => {
            #[cube(launch)]
            fn $name<F: Float, TileM: Dim, TileK: Dim, TileN: Dim, ThreadTileM: Dim, ThreadTileN: Dim>(
                in_a: &Tensor<Line<F>>,
                in_b: &Tensor<Line<F>>,
                output: &mut Tensor<Line<F>>,
            ) {
                let mut st_a = St::<F, $a_d0, $a_d1>::new();
                let mut st_b = St::<F, $b_d0, $b_d1>::new();
                $load_a(in_a, &mut st_a, 0, 0, 0);
                $load_b(in_b, &mut st_b, 0, 0, 0);

                let mut rt_c = Rt::<F, ThreadTileM, ThreadTileN>::new();
                rt_c.zero();

                // Using F, F for homogeneous types (backward compatible)
                $mma_fn::<F, F, TileM, TileK, TileN, ThreadTileM, ThreadTileN>(
                    &mut rt_c, &st_a, &st_b,
                );

                store_rt_direct::<F, ThreadTileM, ThreadTileN, TileM, TileN>(&rt_c, output, 0, 0, 0);
            }
        };

        (@impl $name:ident, $mma_fn:ident, true,
         [$a_d0:ident, $a_d1:ident], $load_a:ident,
         [$b_d0:ident, $b_d1:ident], $load_b:ident) => {
            #[cube(launch)]
            fn $name<F: Float, TileM: Dim, TileK: Dim, TileN: Dim, ThreadTileM: Dim, ThreadTileN: Dim>(
                in_a: &Tensor<Line<F>>,
                in_b: &Tensor<Line<F>>,
                output: &mut Tensor<Line<F>>,
            ) {
                let mut st_a = St::<F, $a_d0, $a_d1>::new();
                let mut st_b = St::<F, $b_d0, $b_d1>::new();
                $load_a(in_a, &mut st_a, 0, 0, 0);
                $load_b(in_b, &mut st_b, 0, 0, 0);

                // c_trans=true: Rt<N, M>, then transpose to Rt<M, N>
                let mut rt_c = Rt::<F, ThreadTileN, ThreadTileM>::new();
                rt_c.zero();

                // Using F, F for homogeneous types (backward compatible)
                $mma_fn::<F, F, TileM, TileK, TileN, ThreadTileM, ThreadTileN>(
                    &mut rt_c, &st_a, &st_b,
                );

                let rt_result = rt_c.transpose();
                store_rt_direct::<F, ThreadTileM, ThreadTileN, TileM, TileN>(&rt_result, output, 0, 0, 0);
            }
        };
    }

    // a_trans=false, b_trans=false: A=[M,K] direct, B=[K,N] transpose to [N,K]
    define_plane_test_kernel!(
        test_kernel_ABt,
        mma_ABt,
        false,
        [TileM, TileK],
        load_st_direct,
        [TileN, TileK],
        load_st_transpose
    );
    define_plane_test_kernel!(
        test_kernel_ABt_t,
        mma_ABt_t,
        true,
        [TileM, TileK],
        load_st_direct,
        [TileN, TileK],
        load_st_transpose
    );

    // a_trans=false, b_trans=true: A=[M,K] direct, B=[K,N] direct
    define_plane_test_kernel!(
        test_kernel_AB,
        mma_AB,
        false,
        [TileM, TileK],
        load_st_direct,
        [TileK, TileN],
        load_st_direct
    );
    define_plane_test_kernel!(
        test_kernel_AB_t,
        mma_AB_t,
        true,
        [TileM, TileK],
        load_st_direct,
        [TileK, TileN],
        load_st_direct
    );

    // a_trans=true, b_trans=false: A=[M,K] transpose to [K,M], B=[K,N] transpose to [N,K]
    define_plane_test_kernel!(
        test_kernel_AtBt,
        mma_AtBt,
        false,
        [TileK, TileM],
        load_st_transpose,
        [TileN, TileK],
        load_st_transpose
    );
    define_plane_test_kernel!(
        test_kernel_AtBt_t,
        mma_AtBt_t,
        true,
        [TileK, TileM],
        load_st_transpose,
        [TileN, TileK],
        load_st_transpose
    );

    // a_trans=true, b_trans=true: A=[M,K] transpose to [K,M], B=[K,N] direct
    define_plane_test_kernel!(
        test_kernel_AtB,
        mma_AtB,
        false,
        [TileK, TileM],
        load_st_transpose,
        [TileK, TileN],
        load_st_direct
    );
    define_plane_test_kernel!(
        test_kernel_AtB_t,
        mma_AtB_t,
        true,
        [TileK, TileM],
        load_st_transpose,
        [TileK, TileN],
        load_st_direct
    );

    macro_rules! define_plane_test {
        ($test_name:ident, $kernel:ident) => {
            test_kernel! {
                #[test]
                fn $test_name() for F in [f32, f64] TileM in [D8] TileK in [D4, D8] TileN in [D8] ThreadTileM in [D4] ThreadTileN in [D4] {
                    let in_a: Tensor = [TileM::VALUE, TileK::VALUE] as Range;
                    let in_b: Tensor = [TileK::VALUE, TileN::VALUE] as Range;
                    let output: Tensor = [TileM::VALUE, TileN::VALUE];
                    {
                        let num_threads = (TileM::VALUE / ThreadTileM::VALUE) * (TileN::VALUE / ThreadTileN::VALUE);
                    }
                    assert_eq!(
                        $kernel(in_a(), in_b(), output()) for (num_threads, 1, 1) @ (num_threads),
                        { reference_matmul(&in_a, &in_b, &mut output, TileM::VALUE, TileK::VALUE, TileN::VALUE); }
                    );
                }
            }
        };
    }

    define_plane_test!(test_mma_ABt, test_kernel_ABt);
    define_plane_test!(test_mma_ABt_t, test_kernel_ABt_t);
    define_plane_test!(test_mma_AB, test_kernel_AB);
    define_plane_test!(test_mma_AB_t, test_kernel_AB_t);
    define_plane_test!(test_mma_AtBt, test_kernel_AtBt);
    define_plane_test!(test_mma_AtBt_t, test_kernel_AtBt_t);
    define_plane_test!(test_mma_AtB, test_kernel_AtB);
    define_plane_test!(test_mma_AtB_t, test_kernel_AtB_t);
}
