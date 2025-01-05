#![allow(non_snake_case)]

use cubecl::prelude::*;

use crate::{
    cube::{load_st_direct, load_st_transpose, mma_AB, mma_AtB, store_rt_direct},
    prelude::*,
    test_kernel,
    test_utils::TestFloat,
};

/// Heterogeneous test kernel for mma_AtB: FIn inputs, FAcc accumulator
#[cube(launch)]
fn test_mma_AtB_hetero<
    FIn: Float,
    FAcc: Float,
    TileK: Dim,
    TileM: Dim,
    TileN: Dim,
    ThreadTileM: Dim,
    ThreadTileN: Dim,
>(
    in_a: &Tensor<Line<FIn>>,
    in_b: &Tensor<Line<FIn>>,
    output: &mut Tensor<Line<FAcc>>,
) {
    let mut st_a = St::<FIn, TileK, TileM>::new();
    let mut st_b = St::<FIn, TileK, TileN>::new();

    let mut rt_c = Rt::<FAcc, ThreadTileM, ThreadTileN>::new();
    rt_c.zero();

    load_st_transpose(in_a, &mut st_a, 0, 0, 0);
    load_st_direct(in_b, &mut st_b, 0, 0, 0);

    mma_AtB::<FIn, FAcc, TileM, TileK, TileN, ThreadTileM, ThreadTileN>(&mut rt_c, &st_a, &st_b);

    store_rt_direct::<FAcc, ThreadTileM, ThreadTileN, TileM, TileN>(&rt_c, output, 0, 0, 0);
}

/// Test kernel that performs C = A * B using the mma_AtB function.
#[cube(launch)]
fn test_mma_AtB<
    F: Float,
    TileK: Dim,
    TileM: Dim,
    TileN: Dim,
    ThreadTileM: Dim,
    ThreadTileN: Dim,
>(
    in_a: &Tensor<Line<F>>,
    in_b: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    test_mma_AtB_hetero::<F, F, TileK, TileM, TileN, ThreadTileM, ThreadTileN>(in_a, in_b, output);
}

/// Heterogeneous test kernel for mma_AB: FIn inputs, FAcc accumulator
#[cube(launch)]
fn test_mma_AB_hetero<
    FIn: Float,
    FAcc: Float,
    TileM: Dim,
    TileK: Dim,
    TileN: Dim,
    ThreadTileM: Dim,
    ThreadTileN: Dim,
>(
    in_a: &Tensor<Line<FIn>>,
    in_b: &Tensor<Line<FIn>>,
    output: &mut Tensor<Line<FAcc>>,
) {
    let mut st_a = St::<FIn, TileM, TileK>::new();
    let mut st_b = St::<FIn, TileK, TileN>::new();

    let mut rt_c = Rt::<FAcc, ThreadTileM, ThreadTileN>::new();
    rt_c.zero();

    load_st_direct(in_a, &mut st_a, 0, 0, 0);
    load_st_direct(in_b, &mut st_b, 0, 0, 0);

    mma_AB::<FIn, FAcc, TileM, TileK, TileN, ThreadTileM, ThreadTileN>(&mut rt_c, &st_a, &st_b);

    store_rt_direct::<FAcc, ThreadTileM, ThreadTileN, TileM, TileN>(&rt_c, output, 0, 0, 0);
}

/// Test kernel that performs C = A * B using the mma_AB function.
#[cube(launch)]
fn test_mma_AB<F: Float, TileM: Dim, TileK: Dim, TileN: Dim, ThreadTileM: Dim, ThreadTileN: Dim>(
    in_a: &Tensor<Line<F>>,
    in_b: &Tensor<Line<F>>,
    output: &mut Tensor<Line<F>>,
) {
    test_mma_AB_hetero::<F, F, TileM, TileK, TileN, ThreadTileM, ThreadTileN>(in_a, in_b, output);
}

test_kernel! {
    #[test]
    fn test_mma_AtB() for
        F in [f32, f64]
        TileK in [D4, D8]
        TileM in [D4, D8, D16]
        TileN in [D4, D8, D16]
        ThreadTileM in [D4]
        ThreadTileN in [D4]
    {
        let in_a: Tensor = [TileM::VALUE, TileK::VALUE] as Range;
        let in_b: Tensor = [TileK::VALUE, TileN::VALUE] as Range;
        let output: Tensor = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            test_mma_AtB(in_a(), in_b(), output()) for (1, 1, 1) @ ((TileM::VALUE / ThreadTileM::VALUE) * (TileN::VALUE / ThreadTileN::VALUE)),
            {
                for mi in 0..TileM::VALUE {
                    for ni in 0..TileN::VALUE {
                        let mut sum = F::from_int(0);
                        for ki in 0..TileK::VALUE {
                            let a_val = in_a[mi * TileK::VALUE + ki];
                            let b_val = in_b[ki * TileN::VALUE + ni];
                            sum += a_val * b_val;
                        }
                        output[mi * TileN::VALUE + ni] = sum;
                    }
                }
            }
        );
    }

    #[test]
    fn test_mma_AB() for
        F in [f32, f64]
        TileM in [D4, D8, D16]
        TileK in [D4, D8]
        TileN in [D4, D8, D16]
        ThreadTileM in [D4]
        ThreadTileN in [D4]
    {
        let in_a: Tensor = [TileM::VALUE, TileK::VALUE] as Range;
        let in_b: Tensor = [TileK::VALUE, TileN::VALUE] as Range;
        let output: Tensor = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            test_mma_AB(in_a(), in_b(), output()) for (1, 1, 1) @ ((TileM::VALUE / ThreadTileM::VALUE) * (TileN::VALUE / ThreadTileN::VALUE)),
            {
                for mi in 0..TileM::VALUE {
                    for ni in 0..TileN::VALUE {
                        let mut sum = F::from_int(0);
                        for ki in 0..TileK::VALUE {
                            let a_val = in_a[mi * TileK::VALUE + ki];
                            let b_val = in_b[ki * TileN::VALUE + ni];
                            sum += a_val * b_val;
                        }
                        output[mi * TileN::VALUE + ni] = sum;
                    }
                }
            }
        );
    }
}

test_kernel! {
    #[test]
    fn test_mma_AB_hetero() for
        (FIn, FAcc) in [(bf16, f32), (f16, f32)]
        TileM in [D8]
        TileK in [D4, D8]
        TileN in [D8]
        ThreadTileM in [D4]
        ThreadTileN in [D4]
    {
        let in_a: Tensor<FIn> = [TileM::VALUE, TileK::VALUE] as Range;
        let in_b: Tensor<FIn> = [TileK::VALUE, TileN::VALUE] as Range;
        let output: Tensor<FAcc> = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            test_mma_AB_hetero(in_a(), in_b(), output()) for (1, 1, 1) @ ((TileM::VALUE / ThreadTileM::VALUE) * (TileN::VALUE / ThreadTileN::VALUE)),
            {
                for mi in 0..TileM::VALUE {
                    for ni in 0..TileN::VALUE {
                        let mut sum = FAcc::from_int(0);
                        for ki in 0..TileK::VALUE {
                            let a_val = FAcc::from_f64(in_a[mi * TileK::VALUE + ki].into_f64());
                            let b_val = FAcc::from_f64(in_b[ki * TileN::VALUE + ni].into_f64());
                            sum += a_val * b_val;
                        }
                        output[mi * TileN::VALUE + ni] = sum;
                    }
                }
            }
        );
    }

    #[test]
    fn test_mma_AtB_hetero() for
        (FIn, FAcc) in [(bf16, f32), (f16, f32)]
        TileK in [D4, D8]
        TileM in [D8]
        TileN in [D8]
        ThreadTileM in [D4]
        ThreadTileN in [D4]
    {
        let in_a: Tensor<FIn> = [TileM::VALUE, TileK::VALUE] as Range;
        let in_b: Tensor<FIn> = [TileK::VALUE, TileN::VALUE] as Range;
        let output: Tensor<FAcc> = [TileM::VALUE, TileN::VALUE];

        assert_eq!(
            test_mma_AtB_hetero(in_a(), in_b(), output()) for (1, 1, 1) @ ((TileM::VALUE / ThreadTileM::VALUE) * (TileN::VALUE / ThreadTileN::VALUE)),
            {
                for mi in 0..TileM::VALUE {
                    for ni in 0..TileN::VALUE {
                        let mut sum = FAcc::from_int(0);
                        for ki in 0..TileK::VALUE {
                            let a_val = FAcc::from_f64(in_a[mi * TileK::VALUE + ki].into_f64());
                            let b_val = FAcc::from_f64(in_b[ki * TileN::VALUE + ni].into_f64());
                            sum += a_val * b_val;
                        }
                        output[mi * TileN::VALUE + ni] = sum;
                    }
                }
            }
        );
    }
}
