use cubecl::{
    frontend::{Exp, Log, Tanh},
    prelude::*,
};

use crate::{
    prelude::*,
    tiles::{Dim, DimOrOne},
};

macro_rules! impl_unary_ops {
    {
    $(
        $name:ident<$t:ident>($input:ident $(, $state:ident)?) => $body:expr;
    )+
    } => {
      $(
          ::paste::paste! {
              impl_unary_ops!([<$name Op>]<$t>($input $(, $state)?) => $body);
          }
      )+
    };

    ($name:ident<$t:ident>($input:ident, $state:ident) => $body:expr) => {
        #[derive(CubeType)]
        pub struct $name<$t: Float> { value: Line<$t> }

        #[cube]
        impl<$t: Float> $name<$t> {
            pub fn new(value: $t) -> Self {
                $name::<$t> { value: Line::empty(LINE_SIZE).fill(value) }
            }
        }

        #[cube]
        impl<$t: Float> UnaryOp<$t> for $name<$t> {
            fn apply(&self, $input: Line<$t>) -> Line<$t> {
                let $state = self.value;
                $body
            }
        }
    };
    ($name:ident<$t:ident>($input:ident) => $body:expr) => {
        #[derive(CubeType)]
        pub struct $name;

        // The CubeType derive doesn't handle unit structs too nicely,
        // so we have to hand-impl this
        ::paste::paste! {
            impl From<$name> for [<$name Expand>] {
                fn from(_: $name) -> Self {
                    [<$name Expand>] {}
                }
            }
        }

        #[cube]
        impl<$t: Float> UnaryOp<$t> for $name {
            fn apply(&self, $input: Line<$t>) -> Line<$t> {
                $body
            }
        }
    };
}

macro_rules! impl_convenience_fns {
    {
        for $ty:ty;
        $(
            $name:ident<$t:ident>($input:ident $(, $state:ident)?) => $body:expr;
        )+
    }
    => {
        ::paste::paste! {
            $(
                #[cube]
                impl<$t: Float, R: Dim, C: DimOrOne> $ty<$t, R, C> {
                    pub fn [<$name:snake>](&mut self $(, $state: $t)?) {
                        self.apply_unary_op::
                        <
                            impl_convenience_fns! { @generic [<$name Op>] <$t> $(, $state)? }
                        >
                        ([<$name Op>] $( ::new ($state))?);
                    }
                }
            )+
        }
    };

    { @generic $i:ident <$t:ty> , $state:ident } => { $i <$t> };
    { @generic $i:ident <$t:ty> } => { $i };
}

#[cube]
pub trait UnaryOp<F: Float> {
    fn apply(&self, x: Line<F>) -> Line<F>;
}

macro_rules! with_unary_ops {
    ($callback:path ; $($($arg:tt)+)?) => {
        $callback! {
            $($($arg)+;)?

            Zero<F>(_x) => Line::<F>::empty(LINE_SIZE).fill(F::from_int(0));
            Fill<F>(_x, f) => f;

            AddScalar<F>(x, f) => x + f;
            MulScalar<F>(x, f) => x * f;
            DivScalar<F>(x, f) => x / f;

            Neg<F>(x) => neg(x);

            Sqrt<F>(x) => Line::<F>::sqrt(x);
            Powf<F>(x, f) => Line::<F>::powf(x, f);

            Exp<F>(x) => Line::<F>::exp(x);
            Log<F>(x) => Line::<F>::ln(x);

            Gelu<F>(x) => gelu::<F>(x);
            GeluBwd<F>(x) => gelu_bwd::<F>(x);
            GeluBwdBwd<F>(x) => gelu_bwd_bwd::<F>(x);

            Sigmoid<F>(x) => sigmoid::<F>(x);
            SigmoidBwd<F>(x) => sigmoid_bwd::<F>(x);

            Tanh<F>(x) => Line::<F>::tanh(x);
            TanhBwd<F>(x) => tanh_bwd::<F>(x);
        }
    };
}

with_unary_ops!(impl_unary_ops;);
with_unary_ops!(impl_convenience_fns; for Rt);
with_unary_ops!(impl_convenience_fns; for St);

const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
const GELU_COEFF: f32 = 0.044715;
const MINUS_ONE: f32 = -1.0;

// just -x seems to cause issues
#[cube]
fn neg<F: Float>(x: Line<F>) -> Line<F> {
    Line::<F>::empty(LINE_SIZE).fill(F::cast_from(MINUS_ONE)) * x
}

#[cube]
pub fn gelu<F: Float>(x: Line<F>) -> Line<F> {
    let sqrt_2_over_pi = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(GELU_SQRT_2_OVER_PI));
    let coeff = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(GELU_COEFF));
    let half = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(0.5f32));
    let one = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(1.0f32));

    let x3 = x * x * x;
    let inner = sqrt_2_over_pi * (x + coeff * x3);
    x * half * (one + Line::<F>::tanh(inner))
}

#[cube]
pub fn gelu_bwd<F: Float>(x: Line<F>) -> Line<F> {
    let sqrt_2_over_pi = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(GELU_SQRT_2_OVER_PI));
    let coeff = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(GELU_COEFF));
    let half = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(0.5f32));
    let one = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(1.0f32));
    let three = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(3.0f32));

    let x2 = x * x;
    let x3 = x2 * x;
    let inner = sqrt_2_over_pi * (x + coeff * x3);
    let tanh_inner = Line::<F>::tanh(inner);
    let sech2_inner = one - tanh_inner * tanh_inner;
    let dinner_dx = sqrt_2_over_pi * (one + three * coeff * x2);
    half * (one + tanh_inner) + half * x * sech2_inner * dinner_dx
}

#[cube]
pub fn gelu_bwd_bwd<F: Float>(x: Line<F>) -> Line<F> {
    let sqrt_2_over_pi = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(GELU_SQRT_2_OVER_PI));
    let coeff = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(GELU_COEFF));
    let one = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(1.0f32));
    let three = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(3.0f32));
    let six = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(6.0f32));

    let x2 = x * x;

    let inner = sqrt_2_over_pi * x * (one + coeff * x2);
    let tanh_out = Line::<F>::tanh(inner);
    let sech2 = one - tanh_out * tanh_out;

    let term1 = sqrt_2_over_pi;
    let term2 = six * sqrt_2_over_pi * coeff * x2;
    let inner_deriv = sqrt_2_over_pi + three * sqrt_2_over_pi * coeff * x2;
    let term3 = x * tanh_out * inner_deriv * inner_deriv;

    sech2 * (term1 + term2 - term3)
}

#[cube]
fn sigmoid<F: Float>(x: Line<F>) -> Line<F> {
    let one = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(1.0f32));
    one / (one + Line::<F>::exp(neg(x)))
}

#[cube]
fn sigmoid_bwd<F: Float>(x: Line<F>) -> Line<F> {
    let one = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(1.0f32));
    let s = one / (one + Line::<F>::exp(neg(x)));
    s * (one - s)
}

#[cube]
fn tanh_bwd<F: Float>(x: Line<F>) -> Line<F> {
    let one = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(1.0f32));
    let t = Line::<F>::tanh(x);
    one - t * t
}

#[cfg(test)]
mod tests {
    use test_case::test_case;

    use crate::{test_utils::TestFloat, unary_ops::*};

    macro_rules! generate_kernel {
        ($name:ident, $method:ident $(, $arg:ident)?) => {
            #[cube(launch)]
            fn $name<F: Float + CubeElement>(
                input: &Array<Line<F>>,
                output: &mut Array<Line<F>>,
                $( $arg: F, )?
            ) {
                let mut rt = Rt::<F, D4, D4>::new();
                rt.copy_from_array(input);
                rt.$method($($arg)?);
                rt.copy_to_array(output);
            }
        };
    }

    generate_kernel!(test_zero_kernel, zero);
    generate_kernel!(test_fill_kernel, fill, fill_val);
    generate_kernel!(test_add_kernel, add_scalar, add_val);
    generate_kernel!(test_mul_kernel, mul_scalar, mul_val);
    generate_kernel!(test_div_kernel, div_scalar, div_val);
    generate_kernel!(test_sqrt_kernel, sqrt);
    generate_kernel!(test_powf_kernel, powf, powf_val);
    generate_kernel!(test_exp_kernel, exp);
    generate_kernel!(test_log_kernel, log);
    generate_kernel!(test_gelu_kernel, gelu);
    generate_kernel!(test_gelu_bwd_kernel, gelu_bwd);
    generate_kernel!(test_gelu_bwd_bwd_kernel, gelu_bwd_bwd);
    generate_kernel!(test_sigmoid_kernel, sigmoid);
    generate_kernel!(test_sigmoid_bwd_kernel, sigmoid_bwd);
    generate_kernel!(test_tanh_kernel, tanh);
    generate_kernel!(test_tanh_bwd_kernel, tanh_bwd);

    // ==================== REFERENCE IMPLEMENTATIONS ====================

    fn ref_gelu(x: f64) -> f64 {
        let sqrt_2_over_pi = GELU_SQRT_2_OVER_PI as f64;
        let coeff = GELU_COEFF as f64;
        let x3 = x * x * x;
        let inner = sqrt_2_over_pi * (x + coeff * x3);
        x * 0.5 * (1.0 + inner.tanh())
    }

    fn ref_gelu_bwd(x: f64) -> f64 {
        let sqrt_2_over_pi = GELU_SQRT_2_OVER_PI as f64;
        let coeff = GELU_COEFF as f64;
        let x2 = x * x;
        let x3 = x2 * x;
        let inner = sqrt_2_over_pi * (x + coeff * x3);
        let tanh_inner = inner.tanh();
        let sech2_inner = 1.0 - tanh_inner * tanh_inner;
        let dinner_dx = sqrt_2_over_pi * (1.0 + 3.0 * coeff * x2);
        0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * dinner_dx
    }

    fn ref_gelu_bwd_bwd(x: f64) -> f64 {
        let sqrt_2_over_pi = GELU_SQRT_2_OVER_PI as f64;
        let coeff = GELU_COEFF as f64;
        let x2 = x * x;
        let inner = sqrt_2_over_pi * x * (1.0 + coeff * x2);
        let tanh_out = inner.tanh();
        let sech2 = 1.0 - tanh_out * tanh_out;
        let term1 = sqrt_2_over_pi;
        let term2 = 6.0 * sqrt_2_over_pi * coeff * x2;
        let inner_deriv = sqrt_2_over_pi + 3.0 * sqrt_2_over_pi * coeff * x2;
        let term3 = x * tanh_out * inner_deriv * inner_deriv;
        sech2 * (term1 + term2 - term3)
    }

    fn ref_sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn ref_sigmoid_bwd(x: f64) -> f64 {
        let s = ref_sigmoid(x);
        s * (1.0 - s)
    }

    fn ref_tanh_bwd(x: f64) -> f64 {
        let t = x.tanh();
        1.0 - t * t
    }

    // ==================== TESTS ====================

    test_kernel! {
        #[test]
        fn test_zero() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_zero_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    output.fill(F::from_int(0));
                }
            );
        }

        #[test_case(0.0)]
        #[test_case(1.0)]
        #[test_case(-5.5)]
        #[test_case(42.0)]
        fn test_fill(val: f64) for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_fill_kernel(input(), output(), scalar(F::from_f64(val))) for (1, 1, 1) @ (1),
                {
                    output.fill(F::from_f64(val));
                }
            );
        }

        #[test_case(0.0)]
        #[test_case(1.0)]
        #[test_case(-3.5)]
        #[test_case(100.0)]
        fn test_add(val: f64) for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_add_kernel(input(), output(), scalar(F::from_f64(val))) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(input[i].into_f64() + val);
                    }
                }
            );
        }

        #[test_case(1.0)]
        #[test_case(2.0)]
        #[test_case(-0.5)]
        #[test_case(0.1)]
        fn test_mul(val: f64) for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_mul_kernel(input(), output(), scalar(F::from_f64(val))) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(input[i].into_f64() * val);
                    }
                }
            );
        }

        #[test_case(1.0)]
        #[test_case(2.0)]
        #[test_case(-0.5)]
        #[test_case(0.1)]
        fn test_div(val: f64) for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-10.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_div_kernel(input(), output(), scalar(F::from_f64(val))) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(input[i].into_f64() / val);
                    }
                }
            );
        }

        #[test]
        fn test_sqrt() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(0.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_sqrt_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(input[i].into_f64().sqrt());
                    }
                }
            );
        }

        #[test_case(1.0)]
        #[test_case(2.0)]
        #[test_case(1.5)]
        #[test_case(0.5)]
        #[test_case(-1.3)]
        #[test_case(0.0)]
        fn test_powf(exp: f64) for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(0.0, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_powf_kernel(input(), output(), scalar(F::from_f64(exp))) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(input[i].into_f64().powf(exp));
                    }
                }
            );
        }

        #[test]
        fn test_exp() for F in all {
            // Use smaller range to avoid overflow
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-2.0, 2.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_exp_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(input[i].into_f64().exp());
                    }
                }
            );
        }

        #[test]
        fn test_log() for F in all {
            // Use positive values for log
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(0.1, 10.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_log_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(input[i].into_f64().ln());
                    }
                }
            );
        }

        #[test]
        fn test_gelu() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-3.0, 3.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_gelu_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(ref_gelu(input[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_gelu_bwd() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-3.0, 3.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_gelu_bwd_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(ref_gelu_bwd(input[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_gelu_bwd_bwd() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-3.0, 3.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_gelu_bwd_bwd_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(ref_gelu_bwd_bwd(input[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_sigmoid() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-5.0, 5.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_sigmoid_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(ref_sigmoid(input[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_sigmoid_bwd() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-5.0, 5.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_sigmoid_bwd_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(ref_sigmoid_bwd(input[i].into_f64()));
                    }
                }
            );
        }

        #[test]
        fn test_tanh() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-3.0, 3.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_tanh_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(input[i].into_f64().tanh());
                    }
                }
            );
        }

        #[test]
        fn test_tanh_bwd() for F in all {
            let input: Array = [LINE_SIZE * LINE_SIZE] as Uniform(-3.0, 3.0);
            let output: Array = [LINE_SIZE * LINE_SIZE];

            assert_eq!(
                test_tanh_bwd_kernel(input(), output()) for (1, 1, 1) @ (1),
                {
                    for i in 0..output.len() {
                        output[i] = F::from_f64(ref_tanh_bwd(input[i].into_f64()));
                    }
                }
            );
        }
    }
}
