use cubecl::prelude::*;
use test_case::test_matrix;

use crate::{
    cube::{
        load_rt_direct, load_rt_from_st, load_st_direct, load_st_transpose, store_rt_direct,
        store_rt_to_st, store_st_direct, store_st_transpose,
    },
    prelude::*,
    test_kernel,
};

// =============================================================================
// St round-trip kernels (Global <-> St)
// =============================================================================

#[cube(launch)]
fn rw_direct<F: Float, TileM: Dim, TileN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, TileM, TileN>::new();

    let r_off = CUBE_POS_X as usize * TileM::VALUE;
    let c_off = CUBE_POS_Y as usize * TileN::VALUE;
    load_st_direct(input, &mut st, 0, r_off, c_off);
    store_st_direct(&st, output, 0, r_off, c_off);
}

#[cube(launch)]
fn rw_transpose<F: Float, TileM: Dim, TileN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, TileM, TileN>::new();

    let r_off = CUBE_POS_X as usize * TileM::VALUE;
    let c_off = CUBE_POS_Y as usize * TileN::VALUE;
    load_st_transpose(input, &mut st, 0, r_off, c_off);
    store_st_transpose(&st, output, 0, r_off, c_off);
}

// =============================================================================
// Rt <-> St round-trip kernels (Global -> St -> Rt -> St -> Global)
// =============================================================================

/// Tests round-trip: Global -> St -> Rt -> St -> Global
#[cube(launch)]
fn rw_rt_via_st<F: Float, TileM: Dim, TileN: Dim, RtM: Dim, RtN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, TileM, TileN>::new();
    let mut rt = Rt::<F, RtM, RtN>::new();

    // Global -> St
    load_st_direct(input, &mut st, 0, 0, 0);

    sync_cube();

    // St -> Rt (cooperative)
    load_rt_from_st::<F, F, RtM, RtN, TileM, TileN>(&st, &mut rt);

    sync_cube();

    // Rt -> St (cooperative)
    store_rt_to_st::<F, F, RtM, RtN, TileM, TileN>(&rt, &mut st);

    sync_cube();

    // St -> Global
    store_st_direct(&st, output, 0, 0, 0);
}

// =============================================================================
// Rt direct store kernels (Global -> St -> Rt -> Global direct)
// =============================================================================

/// Tests: Global -> St -> Rt -> Global (direct store, bypassing St on the way out)
#[cube(launch)]
fn rw_rt_store_direct<F: Float, TileM: Dim, TileN: Dim, RtM: Dim, RtN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, TileM, TileN>::new();
    let mut rt = Rt::<F, RtM, RtN>::new();

    // Global -> St
    load_st_direct(input, &mut st, 0, 0, 0);

    sync_cube();

    // St -> Rt (cooperative)
    load_rt_from_st::<F, F, RtM, RtN, TileM, TileN>(&st, &mut rt);

    sync_cube();

    // Rt -> Global directly (cooperative)
    store_rt_direct::<F, RtM, RtN, TileM, TileN>(&rt, output, 0, 0, 0);
}

/// Batched version with multiple tiles stored to different offsets
#[cube(launch)]
fn rw_rt_store_direct_batched<F: Float, TileM: Dim, TileN: Dim, RtM: Dim, RtN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let batch_idx = CUBE_POS_X as usize;

    let mut st = St::<F, TileM, TileN>::new();
    let mut rt = Rt::<F, RtM, RtN>::new();

    let base_offset = batch_idx * TileM::VALUE * TileN::VALUE;

    // Global -> St
    load_st_direct(input, &mut st, base_offset, 0, 0);

    sync_cube();

    // St -> Rt (cooperative)
    load_rt_from_st::<F, F, RtM, RtN, TileM, TileN>(&st, &mut rt);

    sync_cube();

    // Rt -> Global directly (cooperative)
    store_rt_direct::<F, RtM, RtN, TileM, TileN>(&rt, output, base_offset, 0, 0);
}

// =============================================================================
// Rt direct load kernels (Global direct -> Rt -> St -> Global)
// =============================================================================

/// Tests: Global -> Rt (direct load, bypassing St on the way in) -> St -> Global
#[cube(launch)]
fn rw_rt_load_direct<F: Float, TileM: Dim, TileN: Dim, RtM: Dim, RtN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut st = St::<F, TileM, TileN>::new();
    let mut rt = Rt::<F, RtM, RtN>::new();

    // Global -> Rt directly (cooperative)
    load_rt_direct::<F, RtM, RtN, TileM, TileN>(input, &mut rt, 0, 0, 0);

    sync_cube();

    // Rt -> St (cooperative)
    store_rt_to_st::<F, F, RtM, RtN, TileM, TileN>(&rt, &mut st);

    sync_cube();

    // St -> Global
    store_st_direct(&st, output, 0, 0, 0);
}

/// Batched version with multiple tiles loaded from different offsets
#[cube(launch)]
fn rw_rt_load_direct_batched<F: Float, TileM: Dim, TileN: Dim, RtM: Dim, RtN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let batch_idx = CUBE_POS_X as usize;

    let mut st = St::<F, TileM, TileN>::new();
    let mut rt = Rt::<F, RtM, RtN>::new();

    let base_offset = batch_idx * TileM::VALUE * TileN::VALUE;

    // Global -> Rt directly (cooperative)
    load_rt_direct::<F, RtM, RtN, TileM, TileN>(input, &mut rt, base_offset, 0, 0);

    sync_cube();

    // Rt -> St (cooperative)
    store_rt_to_st::<F, F, RtM, RtN, TileM, TileN>(&rt, &mut st);

    sync_cube();

    // St -> Global
    store_st_direct(&st, output, base_offset, 0, 0);
}

// =============================================================================
// Rt direct both directions (Global direct -> Rt -> Global direct)
// =============================================================================

/// Tests: Global -> Rt -> Global (bypassing St entirely)
#[cube(launch)]
fn rw_rt_direct<F: Float, TileM: Dim, TileN: Dim, RtM: Dim, RtN: Dim>(
    input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut rt = Rt::<F, RtM, RtN>::new();

    // Global -> Rt directly (cooperative)
    load_rt_direct::<F, RtM, RtN, TileM, TileN>(input, &mut rt, 0, 0, 0);

    sync_cube();

    // Rt -> Global directly (cooperative)
    store_rt_direct::<F, RtM, RtN, TileM, TileN>(&rt, output, 0, 0, 0);
}

// =============================================================================
// MMA then store kernels
// =============================================================================

/// MMA then store directly: load two tiles, compute C = A @ B, store C directly
#[cube(launch)]
fn mma_then_store_rt_direct<F: Float, M: Dim, K: Dim, N: Dim, RtM: Dim, RtN: Dim>(
    a_input: &Tensor<Line<F>>,
    b_input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut a_st = St::<F, M, K>::new();
    let mut b_st = St::<F, K, N>::new();
    let mut c_rt = Rt::<F, RtM, RtN>::new();

    // Load A and B
    load_st_direct(a_input, &mut a_st, 0, 0, 0);
    load_st_direct(b_input, &mut b_st, 0, 0, 0);

    sync_cube();

    // C = A @ B
    c_rt.zero();
    crate::cube::mma_AB::<F, F, M, K, N, RtM, RtN>(&mut c_rt, &a_st, &b_st);

    sync_cube();

    // Store C directly to global (cooperative)
    store_rt_direct::<F, RtM, RtN, M, N>(&c_rt, output, 0, 0, 0);
}

/// Control: MMA then store via St
#[cube(launch)]
fn mma_then_store_via_st<F: Float, M: Dim, K: Dim, N: Dim, RtM: Dim, RtN: Dim>(
    a_input: &Tensor<Line<F>>,
    b_input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let mut a_st = St::<F, M, K>::new();
    let mut b_st = St::<F, K, N>::new();
    let mut c_st = St::<F, M, N>::new();
    let mut c_rt = Rt::<F, RtM, RtN>::new();

    load_st_direct(a_input, &mut a_st, 0, 0, 0);
    load_st_direct(b_input, &mut b_st, 0, 0, 0);

    sync_cube();

    c_rt.zero();
    crate::cube::mma_AB::<F, F, M, K, N, RtM, RtN>(&mut c_rt, &a_st, &b_st);

    sync_cube();

    // Store via St
    store_rt_to_st::<F, F, RtM, RtN, M, N>(&c_rt, &mut c_st);

    sync_cube();

    store_st_direct(&c_st, output, 0, 0, 0);
}

/// Batched MMA then store directly
#[cube(launch)]
fn mma_then_store_rt_direct_batched<F: Float, M: Dim, K: Dim, N: Dim, RtM: Dim, RtN: Dim>(
    a_input: &Tensor<Line<F>>,
    b_input: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    let batch_idx = CUBE_POS_X as usize;

    let mut a_st = St::<F, M, K>::new();
    let mut b_st = St::<F, K, N>::new();
    let mut c_rt = Rt::<F, RtM, RtN>::new();

    let a_offset = batch_idx * M::VALUE * K::VALUE;
    let b_offset = batch_idx * K::VALUE * N::VALUE;
    let c_offset = batch_idx * M::VALUE * N::VALUE;

    load_st_direct(a_input, &mut a_st, a_offset, 0, 0);
    load_st_direct(b_input, &mut b_st, b_offset, 0, 0);

    sync_cube();

    c_rt.zero();
    crate::cube::mma_AB::<F, F, M, K, N, RtM, RtN>(&mut c_rt, &a_st, &b_st);

    sync_cube();

    store_rt_direct::<F, RtM, RtN, M, N>(&c_rt, output, c_offset, 0, 0);
}

// =============================================================================
// Tests
// =============================================================================

test_kernel! {
    // -------------------------------------------------------------------------
    // St round-trip tests
    // -------------------------------------------------------------------------

    #[test_matrix([4, 32], [4, 32], [1, 4, 32, 64])]
    fn test_rw_direct(rows: usize, cols: usize, threads: usize) for
        F in all
        TileM in [D4]
        TileN in [D4]
    {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_direct(input(), output()) for (rows/TileM::VALUE, cols/TileN::VALUE, 1) @ (threads),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([4, 32], [4, 32], [1, 4, 32, 64])]
    fn test_rw_transpose(rows: usize, cols: usize, threads: usize) for
        F in all
        TileM in [D4]
        TileN in [D4]
    {
        let input: Tensor = [rows, cols];
        let output: Tensor = [rows, cols];

        assert_eq!(
            rw_transpose(input(), output()) for (rows/TileM::VALUE, cols/TileN::VALUE, 1) @ (threads),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    // -------------------------------------------------------------------------
    // Rt <-> St round-trip tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rw_rt_via_st() for
        F in all
        TileM in [D16, D64]
        TileN in [D16, D64]
        RtM in [D4, D16]
        RtN in [D4, D16]
    {
        let input: Tensor = [TileM::VALUE, TileN::VALUE];
        let output: Tensor = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            rw_rt_via_st(input(), output()) for (1, 1, 1) @ ((TileM::VALUE / RtM::VALUE) * (TileN::VALUE / RtN::VALUE)),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    // -------------------------------------------------------------------------
    // Rt direct store tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rw_rt_store_direct() for
        F in all
        TileM in [D16, D64]
        TileN in [D16, D64]
        RtM in [D4, D16]
        RtN in [D4, D16]
    {
        let input: Tensor = [TileM::VALUE, TileN::VALUE];
        let output: Tensor = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            rw_rt_store_direct(input(), output()) for (1, 1, 1) @ ((TileM::VALUE / RtM::VALUE) * (TileN::VALUE / RtN::VALUE)),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([2, 4])]
    fn test_rw_rt_store_direct_batched(batches: usize) for
        F in all
        TileM in [D64]
        TileN in [D64]
        RtM in [D16]
        RtN in [D16]
    {
        let input: Tensor = [batches, TileM::VALUE, TileN::VALUE];
        let output: Tensor = [batches, TileM::VALUE, TileN::VALUE];

        assert_eq!(
            rw_rt_store_direct_batched(input(), output()) for (batches, 1, 1) @ ((TileM::VALUE / RtM::VALUE) * (TileN::VALUE / RtN::VALUE)),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    // -------------------------------------------------------------------------
    // Rt direct load tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_rw_rt_load_direct() for
        F in all
        TileM in [D16, D64]
        TileN in [D16, D64]
        RtM in [D4, D16]
        RtN in [D4, D16]
    {
        let input: Tensor = [TileM::VALUE, TileN::VALUE];
        let output: Tensor = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            rw_rt_load_direct(input(), output()) for (1, 1, 1) @ ((TileM::VALUE / RtM::VALUE) * (TileN::VALUE / RtN::VALUE)),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    #[test_matrix([2, 4])]
    fn test_rw_rt_load_direct_batched(batches: usize) for
        F in all
        TileM in [D64]
        TileN in [D64]
        RtM in [D16]
        RtN in [D16]
    {
        let input: Tensor = [batches, TileM::VALUE, TileN::VALUE];
        let output: Tensor = [batches, TileM::VALUE, TileN::VALUE];

        assert_eq!(
            rw_rt_load_direct_batched(input(), output()) for (batches, 1, 1) @ ((TileM::VALUE / RtM::VALUE) * (TileN::VALUE / RtN::VALUE)),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    // -------------------------------------------------------------------------
    // Rt direct both directions (bypassing St entirely)
    // -------------------------------------------------------------------------

    #[test]
    fn test_rw_rt_direct() for
        F in all
        TileM in [D16, D64]
        TileN in [D16, D64]
        RtM in [D4, D16]
        RtN in [D4, D16]
    {
        let input: Tensor = [TileM::VALUE, TileN::VALUE];
        let output: Tensor = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            rw_rt_direct(input(), output()) for (1, 1, 1) @ ((TileM::VALUE / RtM::VALUE) * (TileN::VALUE / RtN::VALUE)),
            {
                output.copy_from_slice(&input);
            }
        );
    }

    // -------------------------------------------------------------------------
    // MMA then store tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_mma_then_store_rt_direct() for
        F in [f32, f64]
        M in [D64]
        K in [D16]
        N in [D64]
        RtM in [D16]
        RtN in [D16]
    {
        let a: Tensor = [M::VALUE, K::VALUE] as Range;
        let b: Tensor = [K::VALUE, N::VALUE] as Range;
        let output: Tensor = [M::VALUE, N::VALUE];

        assert_eq!(
            mma_then_store_rt_direct(a(), b(), output()) for (1, 1, 1) @ ((M::VALUE / RtM::VALUE) * (N::VALUE / RtN::VALUE)),
            {
                for i in 0..M::VALUE {
                    for j in 0..N::VALUE {
                        let mut sum = F::from_int(0);
                        for k in 0..K::VALUE {
                            sum += a[i * K::VALUE + k] * b[k * N::VALUE + j];
                        }
                        output[i * N::VALUE + j] = sum;
                    }
                }
            }
        );
    }

    #[test]
    fn test_mma_then_store_via_st() for
        F in [f32, f64]
        M in [D64]
        K in [D16]
        N in [D64]
        RtM in [D16]
        RtN in [D16]
    {
        let a: Tensor = [M::VALUE, K::VALUE] as Range;
        let b: Tensor = [K::VALUE, N::VALUE] as Range;
        let output: Tensor = [M::VALUE, N::VALUE];

        assert_eq!(
            mma_then_store_via_st(a(), b(), output()) for (1, 1, 1) @ ((M::VALUE / RtM::VALUE) * (N::VALUE / RtN::VALUE)),
            {
                for i in 0..M::VALUE {
                    for j in 0..N::VALUE {
                        let mut sum = F::from_int(0);
                        for k in 0..K::VALUE {
                            sum += a[i * K::VALUE + k] * b[k * N::VALUE + j];
                        }
                        output[i * N::VALUE + j] = sum;
                    }
                }
            }
        );
    }

    #[test_matrix([2, 4])]
    fn test_mma_then_store_rt_direct_batched(batches: usize) for
        F in [f32, f64]
        M in [D64]
        K in [D16]
        N in [D64]
        RtM in [D16]
        RtN in [D16]
    {
        let a: Tensor = [batches, M::VALUE, K::VALUE] as Range;
        let b: Tensor = [batches, K::VALUE, N::VALUE] as Range;
        let output: Tensor = [batches, M::VALUE, N::VALUE];

        assert_eq!(
            mma_then_store_rt_direct_batched(a(), b(), output()) for (batches, 1, 1) @ ((M::VALUE / RtM::VALUE) * (N::VALUE / RtN::VALUE)),
            {
                for batch in 0..batches {
                    let a_off = batch * M::VALUE * K::VALUE;
                    let b_off = batch * K::VALUE * N::VALUE;
                    let c_off = batch * M::VALUE * N::VALUE;
                    for i in 0..M::VALUE {
                        for j in 0..N::VALUE {
                            let mut sum = F::from_int(0);
                            for k in 0..K::VALUE {
                                sum += a[a_off + i * K::VALUE + k] * b[b_off + k * N::VALUE + j];
                            }
                            output[c_off + i * N::VALUE + j] = sum;
                        }
                    }
                }
            }
        );
    }
}
