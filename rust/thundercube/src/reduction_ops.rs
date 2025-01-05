use cubecl::prelude::*;

use crate::{prelude::*, tiles::Dim};

#[cube]
pub trait ReductionOp<F: Float> {
    /// Identity element as a Line (e.g., all zeros for sum)
    fn identity() -> Line<F>;
    /// Combine two Lines element-wise
    fn combine(a: Line<F>, b: Line<F>) -> Line<F>;
    /// Reduce a Line to a scalar
    fn finalize(line: Line<F>) -> F;
    /// Plane-level (warp) reduction of a scalar, broadcasts result to all threads
    fn plane_reduce(val: F) -> F;
    /// Combine the results of two plane reductions
    fn plane_combine(a: F, b: F) -> F;
}

/// Plane-level reduction of a Line (reduces each lane separately across threads)
#[cube]
pub fn plane_reduce_line<F: Float, O: ReductionOp<F>>(line: Line<F>) -> Line<F> {
    let mut result = Line::<F>::empty(LINE_SIZE);
    result[0] = O::plane_reduce(line[0]);
    result[1] = O::plane_reduce(line[1]);
    result[2] = O::plane_reduce(line[2]);
    result[3] = O::plane_reduce(line[3]);
    result
}

#[macro_export]
macro_rules! impl_reduction_ops {
    {
    $(
        $name:ident<$t:ident> {
            identity => $identity:expr;
            combine($a:ident, $b:ident) => $combine:expr;
            finalize($line:ident) => $finalize:expr;
            plane_reduce($val:ident) => $plane_reduce:expr;
            plane_combine($c:ident, $d:ident) => $plane_combine:expr;
        }
    )+
    } => {
    $(
        ::paste::paste! {
            #[derive(CubeType)]
            pub struct [<$name Op>];

            impl From<[<$name Op>]> for [<$name OpExpand>] {
                fn from(_: [<$name Op>]) -> Self {
                    [<$name OpExpand>] {}
                }
            }

            #[cube]
            impl<$t: Float> ReductionOp<$t> for [<$name Op>] {
                fn identity() -> Line<$t> {
                    $identity
                }

                fn combine($a: Line<$t>, $b: Line<$t>) -> Line<$t> {
                    $combine
                }

                fn finalize($line: Line<$t>) -> $t {
                    $finalize
                }

                fn plane_reduce($val: $t) -> $t {
                    $plane_reduce
                }

                fn plane_combine($c: $t, $d: $t) -> $t {
                    $plane_combine
                }
            }
        }
    )+
    };
}

macro_rules! impl_reduction_convenience_fns {
    {
        for $ty:ty;
        $(
            $name:ident<$t:ident> {
                identity => $identity:expr;
                combine($a:ident, $b:ident) => $combine:expr;
                finalize($line:ident) => $finalize:expr;
                plane_reduce($val:ident) => $plane_reduce:expr;
                plane_combine($c:ident, $d:ident) => $plane_combine:expr;
            }
        )+
    }
    => {
        ::paste::paste! {
            $(
                #[cube]
                impl<$t: Float, R: Dim, C: Dim> $ty<$t, R, C> {
                    pub fn [<$name:snake _rows>](&self, result: &mut Rv<$t, R>) {
                        self.reduce_rows::<[<$name Op>]>(result)
                    }

                    pub fn [<$name:snake _cols>](&self, result: &mut Rv<$t, C>) {
                        self.reduce_cols::<[<$name Op>]>(result)
                    }
                }
            )+
        }
    };
}

macro_rules! with_reduction_ops {
    ($callback:path ; $($($arg:tt)+)?) => {
        $callback! {
            $($($arg)+;)?

            Sum<F> {
                identity => Line::<F>::empty(LINE_SIZE).fill(F::new(0.0));
                combine(a, b) => a + b;
                finalize(line) => line[0] + line[1] + line[2] + line[3];
                plane_reduce(val) => plane_sum(val);
                plane_combine(a, b) => a + b;
            }
        }
    };
}

with_reduction_ops!(impl_reduction_ops;);
with_reduction_ops!(impl_reduction_convenience_fns; for Rt);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TestFloat;

    const ROWS: usize = 8;
    const COLS: usize = 8;

    #[cube(launch)]
    fn test_sum_rows_kernel<F: Float + CubeElement>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut rt = Rt::<F, D8, D8>::new();
        rt.copy_from_array(input);
        let mut result = Rv::<F, D8>::new();
        rt.sum_rows(&mut result);
        result.copy_to_array(output);
    }

    #[cube(launch)]
    fn test_sum_cols_kernel<F: Float + CubeElement>(
        input: &Array<Line<F>>,
        output: &mut Array<Line<F>>,
    ) {
        let mut rt = Rt::<F, D8, D8>::new();
        rt.copy_from_array(input);
        let mut result = Rv::<F, D8>::new();
        rt.sum_cols(&mut result);
        result.copy_to_array(output);
    }

    test_kernel! {
        #[test]
        fn test_sum_rows() for F in all {
            let input: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let output: Array = [ROWS];

            assert_eq!(
                test_sum_rows_kernel(input(), output()) for (1, 1, 1) @ (1),
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

        #[test]
        fn test_sum_cols() for F in [::half::bf16 as bf16, f32, f64] {
            let input: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let output: Array = [COLS];

            assert_eq!(
                test_sum_cols_kernel(input(), output()) for (1, 1, 1) @ (1),
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
