use cubecl::prelude::*;

use crate::{
    prelude::*,
    tiles::{Dim, DimOrOne},
};

#[cube]
pub trait BinaryOp<F: Float> {
    fn apply(&self, dst: Line<F>, src: Line<F>) -> Line<F>;
}

macro_rules! impl_binary_ops {
    {
    $(
        $name:ident<$t:ident>($dst:ident, $src:ident) => $body:expr;
    )+
    } => {
    $(
        ::paste::paste! {
            #[derive(CubeType)]
            pub struct [<$name Op>];

            // The CubeType derive doesn't handle unit structs too nicely,
            // so we have to hand-impl this
            impl From<[<$name Op>]> for [<$name OpExpand>] {
                fn from(_: [<$name Op>]) -> Self {
                    [<$name OpExpand>] {}
                }
            }

            #[cube]
            impl<$t: Float> BinaryOp<$t> for [<$name Op>] {
                fn apply(&self, $dst: Line<$t>, $src: Line<$t>) -> Line<$t> {
                    $body
                }
            }
        }
    )+
    };
}

macro_rules! impl_binary_convenience_fns {
    {
        for $ty:ty;
        $(
            $name:ident<$t:ident>($dst:ident, $src:ident) => $body:expr;
        )+
    }
    => {
        ::paste::paste! {
            $(
                #[cube]
                impl<$t: Float, R: Dim, C: DimOrOne> $ty<$t, R, C> {
                    pub fn [<$name:snake>](&mut self, other: &$ty<$t, R, C>) {
                        self.apply_binary_op::<[<$name Op>]>([<$name Op>], other);
                    }
                }
            )+
        }
    };
}

macro_rules! impl_broadcast_convenience_fns {
    {
        for $ty:ty;
        $(
            $name:ident<$t:ident>($dst:ident, $src:ident) => $body:expr;
        )+
    }
    => {
        ::paste::paste! {
            $(
                #[cube]
                impl<$t: Float, R: Dim, C: Dim> $ty<$t, R, C> {
                    pub fn [<$name:snake _row>](&mut self, row: &Rv<$t, C>) {
                        self.apply_row_broadcast::<[<$name Op>]>([<$name Op>], row);
                    }

                    pub fn [<$name:snake _col>](&mut self, col: &Rv<$t, R>) {
                        self.apply_col_broadcast::<[<$name Op>]>([<$name Op>], col);
                    }
                }
            )+
        }
    };
}

macro_rules! with_binary_ops {
    ($callback:path ; $($($arg:tt)+)?) => {
        $callback! {
            $($($arg)+;)?

            Set<F>(_dst, src) => src;

            Add<F>(dst, src) => dst + src;
            Sub<F>(dst, src) => dst - src;
            Mul<F>(dst, src) => dst * src;
            Div<F>(dst, src) => dst / src;

            Min<F>(dst, src) => Line::<F>::min(dst, src);
            Max<F>(dst, src) => Line::<F>::max(dst, src);

            Pow<F>(dst, src) => Line::<F>::powf(dst, src);
        }
    };
}

with_binary_ops!(impl_binary_ops;);
with_binary_ops!(impl_binary_convenience_fns; for Rt);
with_binary_ops!(impl_binary_convenience_fns; for St);
with_binary_ops!(impl_broadcast_convenience_fns; for Rt);
with_binary_ops!(impl_broadcast_convenience_fns; for St);

#[cfg(test)]
mod tests {
    use test_case::test_matrix;

    use crate::{binary_ops::*, test_utils::TestFloat};

    macro_rules! generate_binary_kernel {
        ($name:ident, $method:ident) => {
            #[cube(launch)]
            fn $name<F: Float + CubeElement>(
                a: &Array<Line<F>>,
                b: &Array<Line<F>>,
                output: &mut Array<Line<F>>,
            ) {
                let mut rt_a = Rt::<F, D4, D4>::new();
                let mut rt_b = Rt::<F, D4, D4>::new();
                rt_a.copy_from_array(a);
                rt_b.copy_from_array(b);
                rt_a.$method(&rt_b);
                rt_a.copy_to_array(output);
            }
        };
    }

    generate_binary_kernel!(test_add_kernel, add);
    generate_binary_kernel!(test_sub_kernel, sub);
    generate_binary_kernel!(test_mul_kernel, mul);
    generate_binary_kernel!(test_div_kernel, div);
    generate_binary_kernel!(test_min_kernel, min);
    generate_binary_kernel!(test_max_kernel, max);
    generate_binary_kernel!(test_pow_kernel, pow);

    // ==================== TESTS ====================

    test_kernel! {
        #[test]
        fn test_add() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_add_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64() + b[i].into_f64());
                    }
                }
            );
        }

        #[test]
        fn test_sub() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_sub_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64() - b[i].into_f64());
                    }
                }
            );
        }

        #[test]
        fn test_mul() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_mul_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64() * b[i].into_f64());
                    }
                }
            );
        }

        #[test]
        fn test_div() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            // Avoid division by zero by using values away from zero
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(0.5, 5.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_div_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64() / b[i].into_f64());
                    }
                }
            );
        }

        #[test]
        fn test_min() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_min_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64().min(b[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_max() for F in all {
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_max_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64().max(b[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_pow() for F in all {
            // Use positive base values for pow
            let a: Array = [LINE_SIZE * LINE_SIZE] as Uniform(0.1, 5.0);
            // Use reasonable exponent range
            let b: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-2.0, 2.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_pow_kernel(a(), b(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(a[i].into_f64().powf(b[i].into_f64()));
                    }
                }
            );
        }
    }

    const ROWS: usize = 8;
    const COLS: usize = 8;

    macro_rules! generate_row_broadcast_kernel {
        ($name:ident, $method:ident) => {
            #[cube(launch)]
            fn $name<F: Float + CubeElement>(
                a: &Array<Line<F>>,
                row: &Array<Line<F>>,
                output: &mut Array<Line<F>>,
            ) {
                let mut rt_a = Rt::<F, D8, D8>::new();
                let mut rv_row = Rv::<F, D8>::new();
                rt_a.copy_from_array(a);
                rv_row.copy_from_array(row);
                rt_a.$method(&rv_row);
                rt_a.copy_to_array(output);
            }
        };
    }

    macro_rules! generate_col_broadcast_kernel {
        ($name:ident, $method:ident) => {
            #[cube(launch)]
            fn $name<F: Float + CubeElement>(
                a: &Array<Line<F>>,
                col: &Array<Line<F>>,
                output: &mut Array<Line<F>>,
            ) {
                let mut rt_a = Rt::<F, D8, D8>::new();
                let mut rv_col = Rv::<F, D8>::new();
                rt_a.copy_from_array(a);
                rv_col.copy_from_array(col);
                rt_a.$method(&rv_col);
                rt_a.copy_to_array(output);
            }
        };
    }

    generate_row_broadcast_kernel!(test_add_row_kernel, add_row);
    generate_row_broadcast_kernel!(test_mul_row_kernel, mul_row);
    generate_col_broadcast_kernel!(test_add_col_kernel, add_col);
    generate_col_broadcast_kernel!(test_mul_col_kernel, mul_col);

    // ==================== BROADCAST TESTS ====================

    test_kernel! {
        #[test]
        fn test_add_row() for F in all {
            let a: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let row: Array = [COLS] as Uniform(-10.0, 10.0);
            let output: Array = [ROWS * COLS];

            assert_eq!(
                test_add_row_kernel(a(), row(), output()) for (1, 1, 1) @ (1),
                {
                    for r in 0..ROWS {
                        for c in 0..COLS {
                            let idx = r * COLS + c;
                            output[idx] = F::from_f64(a[idx].into_f64() + row[c].into_f64());
                        }
                    }
                }
            );
        }

        #[test]
        fn test_mul_row() for F in all {
            let a: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let row: Array = [COLS] as Uniform(-10.0, 10.0);
            let output: Array = [ROWS * COLS];

            assert_eq!(
                test_mul_row_kernel(a(), row(), output()) for (1, 1, 1) @ (1),
                {
                    for r in 0..ROWS {
                        for c in 0..COLS {
                            let idx = r * COLS + c;
                            output[idx] = F::from_f64(a[idx].into_f64() * row[c].into_f64());
                        }
                    }
                }
            );
        }

        #[test]
        fn test_add_col() for F in all {
            let a: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let col: Array = [ROWS] as Uniform(-10.0, 10.0);
            let output: Array = [ROWS * COLS];

            assert_eq!(
                test_add_col_kernel(a(), col(), output()) for (1, 1, 1) @ (1),
                {
                    for r in 0..ROWS {
                        for c in 0..COLS {
                            let idx = r * COLS + c;
                            output[idx] = F::from_f64(a[idx].into_f64() + col[r].into_f64());
                        }
                    }
                }
            );
        }

        #[test]
        fn test_mul_col() for F in all {
            let a: Array = [ROWS * COLS] as Uniform(-10.0, 10.0);
            let col: Array = [ROWS] as Uniform(-10.0, 10.0);
            let output: Array = [ROWS * COLS];

            assert_eq!(
                test_mul_col_kernel(a(), col(), output()) for (1, 1, 1) @ (1),
                {
                    for r in 0..ROWS {
                        for c in 0..COLS {
                            let idx = r * COLS + c;
                            output[idx] = F::from_f64(a[idx].into_f64() * col[r].into_f64());
                        }
                    }
                }
            );
        }
    }

    // ==================== ST BROADCAST TESTS ====================

    macro_rules! generate_st_row_broadcast_kernel {
        ($name:ident, $method:ident) => {
            #[cube(launch)]
            fn $name<F: Float + CubeElement>(
                a: &Tensor<Line<F>>,
                row: &Array<Line<F>>,
                output: &mut Tensor<Line<F>>,
            ) {
                let mut st = St::<F, D8, D8>::new();
                let mut rv_row = Rv::<F, D8>::new();

                rv_row.copy_from_array(row);
                crate::cube::load_st_direct(a, &mut st, 0, 0, 0);

                st.$method(&rv_row);

                crate::cube::store_st_direct(&st, output, 0, 0, 0);
            }
        };
    }

    macro_rules! generate_st_col_broadcast_kernel {
        ($name:ident, $method:ident) => {
            #[cube(launch)]
            fn $name<F: Float + CubeElement>(
                a: &Tensor<Line<F>>,
                col: &Array<Line<F>>,
                output: &mut Tensor<Line<F>>,
            ) {
                let mut st = St::<F, D8, D8>::new();
                let mut rv_col = Rv::<F, D8>::new();

                rv_col.copy_from_array(col);
                crate::cube::load_st_direct(a, &mut st, 0, 0, 0);

                st.$method(&rv_col);

                crate::cube::store_st_direct(&st, output, 0, 0, 0);
            }
        };
    }

    generate_st_row_broadcast_kernel!(test_st_add_row_kernel, add_row);
    generate_st_row_broadcast_kernel!(test_st_mul_row_kernel, mul_row);
    generate_st_col_broadcast_kernel!(test_st_add_col_kernel, add_col);
    generate_st_col_broadcast_kernel!(test_st_mul_col_kernel, mul_col);

    test_kernel! {
        #[test_matrix([1, 4, 32, 64])]
        fn test_st_add_row(threads: usize) for F in all {
            let a: Tensor = [ROWS, COLS] as Uniform(-10.0, 10.0);
            let row: Array = [COLS] as Uniform(-10.0, 10.0);
            let output: Tensor = [ROWS, COLS];

            assert_eq!(
                test_st_add_row_kernel(a(), row(), output()) for (1, 1, 1) @ (threads),
                {
                    for r in 0..ROWS {
                        for c in 0..COLS {
                            let idx = r * COLS + c;
                            output[idx] = F::from_f64(a[idx].into_f64() + row[c].into_f64());
                        }
                    }
                }
            );
        }

        #[test_matrix([1, 4, 32, 64])]
        fn test_st_mul_row(threads: usize) for F in all {
            let a: Tensor = [ROWS, COLS] as Uniform(-10.0, 10.0);
            let row: Array = [COLS] as Uniform(-10.0, 10.0);
            let output: Tensor = [ROWS, COLS];

            assert_eq!(
                test_st_mul_row_kernel(a(), row(), output()) for (1, 1, 1) @ (threads),
                {
                    for r in 0..ROWS {
                        for c in 0..COLS {
                            let idx = r * COLS + c;
                            output[idx] = F::from_f64(a[idx].into_f64() * row[c].into_f64());
                        }
                    }
                }
            );
        }

        #[test_matrix([1, 4, 32, 64])]
        fn test_st_add_col(threads: usize) for F in all {
            let a: Tensor = [ROWS, COLS] as Uniform(-10.0, 10.0);
            let col: Array = [ROWS] as Uniform(-10.0, 10.0);
            let output: Tensor = [ROWS, COLS];

            assert_eq!(
                test_st_add_col_kernel(a(), col(), output()) for (1, 1, 1) @ (threads),
                {
                    for r in 0..ROWS {
                        for c in 0..COLS {
                            let idx = r * COLS + c;
                            output[idx] = F::from_f64(a[idx].into_f64() + col[r].into_f64());
                        }
                    }
                }
            );
        }

        #[test_matrix([1, 4, 32, 64])]
        fn test_st_mul_col(threads: usize) for F in all {
            let a: Tensor = [ROWS, COLS] as Uniform(-10.0, 10.0);
            let col: Array = [ROWS] as Uniform(-10.0, 10.0);
            let output: Tensor = [ROWS, COLS];

            assert_eq!(
                test_st_mul_col_kernel(a(), col(), output()) for (1, 1, 1) @ (threads),
                {
                    for r in 0..ROWS {
                        for c in 0..COLS {
                            let idx = r * COLS + c;
                            output[idx] = F::from_f64(a[idx].into_f64() * col[r].into_f64());
                        }
                    }
                }
            );
        }
    }
}
