#![allow(dead_code)]

use std::{
    fmt::Display,
    hash::{DefaultHasher, Hasher},
};

use cubecl::{prelude::*, server::Handle};

#[cfg(all(
    any(test, feature = "test-utils"),
    not(any(feature = "cuda", feature = "rocm", feature = "wgpu", feature = "cpu"))
))]
pub type TestRuntime = compile_error!(
    "At least one backend must be enabled for test-utils, please run with `--features cuda/rocm/wgpu/cpu`"
);

#[cfg(feature = "rocm")]
pub type TestRuntime = cubecl::hip::HipRuntime;

#[cfg(feature = "cuda")]
pub type TestRuntime = cubecl::cuda::CudaRuntime;

#[cfg(feature = "wgpu")]
pub type TestRuntime = cubecl::wgpu::WgpuRuntime;

#[cfg(feature = "cpu")]
pub type TestRuntime = cubecl::cpu::CpuRuntime;

use half::{bf16, f16};
use rand::{RngExt, rngs::StdRng};

/// A macro for testing CubeCL GPU kernels by comparing their output against reference implementations.
///
/// # Syntax
///
/// ```ignore
/// test_kernel! {
///     #[test_attributes]
///     fn test_name(test_args) for TypeName in [type_list] or all {
///         seed(optional_seed);
///
///         let var_name: VarType = [dims] as Distribution;
///
///         // Optionally:
///         {
///             // Setup code, can modify variables.
///             // Despite looking scoped, declarations
///             // are available outside the block.
///         }
///
///         assert_eq!(
///             kernel_name(arg_spec, ...) for (cube_x, cube_y, cube_z) @ dim_spec,
///             { reference_code }
///         );
///     }
/// }
/// ```
///
/// # Components
///
/// - **Test attributes**: Standard Rust test attributes like `#[test]`, `#[test_case(...)]`, `#[test_matrix(...)]`
/// - **Type specification**: `for TypeName in [f32, f64]` or `for TypeName in all` (expands to bf16, f16, f32, f64).
///   Custom type aliases use `Type | alias` syntax (e.g., `::half::f16 | f16`).
/// - **Seed**: Optional `seed(u64)` for deterministic RNG. Defaults to hash of the test name.
/// - **Variables**: Declare test data with `let name: Type = [dims] as Distribution;`
///   - `Type`: `Tensor` or `Array`
///   - `dims`: Shape dimensions (e.g., `[4, 8]`)
///   - `Distribution`: `Range` (0..len) or `Uniform(start, end)`. Defaults to `Uniform(-10.0, 10.0)`.
/// - **Preamble**: Optional block of setup code that can modify or create variables.
/// - **Kernel launch**: `kernel_name(args) for (cube_count) @ dim_spec`
///   - `args`: `var_name()` passes the variable, `lit(expr)` passes a literal value
///   - `cube_count`: `(x, y, z)` cube count for launch
///   - `dim_spec`: `(x, y, z)` fixed dimensions or `max(n)` for max supported dimensions
/// - **Reference code**: A block that mutates declared variables to their expected values
///
/// # Example
///
/// ```ignore
/// test_kernel! {
///     #[test_case(4, 4)]
///     fn my_kernel_test(rows: usize, cols: usize) for F in all {
///         seed(123);
///         let input: Tensor = [rows, cols];
///         let output: Tensor = [rows, cols] as Range;
///         assert_eq!(
///             my_kernel(input(), output()) for (1, 1, 1) @ (32, 32),
///             {
///                 // Compute expected values in `output`
///                 for i in 0..output.len() {
///                     output[i] = input[i] * F::from_int(2);
///                 }
///             }
///         );
///     }
/// }
/// ```
#[macro_export]
macro_rules! test_kernel {
    // ==================== ENTRY POINT (HOMOGENEOUS) ====================
    {
    $(
        $(#[$attr:meta])*
        fn $name:ident($($args:tt)*)
            for $float_name:ident in $float_ty:tt
                $($gen_name:ident in [$($gen_opts:tt)+] )*
        {
            $(seed($seed:expr);)?

            $(
            let $var:ident: $vty:tt = [$($val:expr),*] $(as $distrib:ident $(($($distrib_param:expr),+))?)?;
            )*

            $({
                $($preamble:stmt)*
            })?

            assert_eq!(
                $kernel:ident ($($kernel_arg_name:ident($($kernel_arg:expr)?)),* $(,)?) for ($($count:expr),*) @ $($max:ident)? ($($dim:expr),*),
                $ref:expr $(,)?
            );
        }
    )*
    } => {
    $(
        $crate::test_kernel! {
            generics: [
                $float_name in $float_ty;
                $($gen_name in [$($gen_opts)+] ;)*
            ];
            ctx: {
                attrs: $(#[$attr])*;
                name: $name;
                args: ($($args)*);
                seed: ($($seed)?);
                vars: $(let $var: $vty < $float_name > = [$($val),*] $(as $distrib $(($($distrib_param),+))?)?;)*;
                preamble: {
                    $($($preamble)*)?
                };
                kernel: $kernel;
                kernel_args: ($($kernel_arg_name($($kernel_arg)?)),*);
                count: ($($count),*);
                dim: $($max)? ($($dim),*);
                ref: $ref;
            };
        }
    )*
    };

    // ==================== ENTRY POINT (HETEROGENEOUS) ====================
    {
    $(
        $(#[$attr:meta])*
        fn $name:ident($($args:tt)*)
            for ($fin_name:ident, $facc_name:ident) in [$(($fin_ty:tt, $facc_ty:tt)),+ $(,)?]
                $($gen_name:ident in [$($gen_opts:tt)+] )*
        {
            $(seed($seed:expr);)?

            $(
            let $var:ident: $vty:tt < $var_float:ident > = [$($val:expr),*] $(as $distrib:ident $(($($distrib_param:expr),+))?)?;
            )*

            $({
                $($preamble:stmt)*
            })?

            assert_eq!(
                $kernel:ident ($($kernel_arg_name:ident($($kernel_arg:expr)?)),* $(,)?) for ($($count:expr),*) @ $($max:ident)? ($($dim:expr),*),
                $ref:expr $(,)?
            );
        }
    )*
    } => {
    $(
        $crate::test_kernel! {
            generics_hetero: [
                ($fin_name, $facc_name) in [$(($fin_ty, $facc_ty)),+];
                $($gen_name in [$($gen_opts)+] ;)*
            ];
            ctx: {
                attrs: $(#[$attr])*;
                name: $name;
                args: ($($args)*);
                seed: ($($seed)?);
                vars: $(let $var: $vty < $var_float > = [$($val),*] $(as $distrib $(($($distrib_param),+))?)?;)*;
                preamble: {
                    $($($preamble)*)?
                };
                kernel: $kernel;
                kernel_args: ($($kernel_arg_name($($kernel_arg)?)),*);
                count: ($($count),*);
                dim: $($max)? ($($dim),*);
                ref: $ref;
            };
        }
    )*
    };

    {
        generics: [
            $float_name:ident in all;
            $($gen_name:ident in [$($gen_opts:tt)+] ;)*
        ];
        ctx: { $($ctx:tt)+ };
    } => {
        $crate::cartesian! {
            test_kernel!($($ctx)+);
            $float_name in [::half::f16 as f16, ::half::bf16 as bf16, f32, f64];
            $($gen_name in [$($gen_opts)+] ;)*
        }
    };

    {
        generics: [
            $float_name:ident in $float_ty:tt;
            $($gen_name:ident in [$($gen_opts:tt)+] ;)*
        ];
        ctx: { $($ctx:tt)+ };
    } => {
        $crate::cartesian! {
            test_kernel!($($ctx)+);
            $float_name in $float_ty;
            $($gen_name in [$($gen_opts)+] ;)*
        }
    };

    // ==================== GENERICS EXPANSION (HETEROGENEOUS) ====================
    {
        generics_hetero: [
            ($fin_name:ident, $facc_name:ident) in [$($float_pairs:tt)+];
            $($gen_rest:tt)*
        ];
        ctx: $ctx:tt;
    } => {
        $crate::test_kernel! {
            @hetero_expand_pairs
            fin_name: $fin_name;
            facc_name: $facc_name;
            pairs: [$($float_pairs)+];
            rest: [$($gen_rest)*];
            ctx: $ctx;
        }
    };

    {
        @hetero_expand_pairs
        fin_name: $fin_name:ident;
        facc_name: $facc_name:ident;
        pairs: [$(($fin_ty:tt, $facc_ty:tt)),+];
        rest: $rest:tt;
        ctx: $ctx:tt;
    } => {
        $(
            $crate::test_kernel! {
                @hetero_cartesian
                fin: $fin_name = $fin_ty;
                facc: $facc_name = $facc_ty;
                rest: $rest;
                ctx: $ctx;
            }
        )+
    };

    {
        @hetero_cartesian
        fin: $fin_name:ident = $fin_ty:tt;
        facc: $facc_name:ident = $facc_ty:tt;
        rest: [$($gen_name:ident in [$($gen_opts:tt)+] ;)*];
        ctx: { $($ctx:tt)+ };
    } => {
        $crate::cartesian! {
            test_kernel!(hetero: $fin_name = $fin_ty, $facc_name = $facc_ty; $($ctx)+);
            $($gen_name in [$($gen_opts)+] ;)*
        }
    };

    // ==================== FUNCTION GENERATION (HOMOGENEOUS) ====================
    {
        types: [
            $float_name:ident = $float_ty:ty as $float_str:ident;
            $( $gen_name:ident = $gen_t:ty as $gen_str:ident ; )*
        ];
        attrs: $(#[$attr:meta])*;
        name: $name:ident;
        args: ($($args:tt)*);
        $($rest:tt)*
    } => {
        ::paste::paste! {
            $(#[$attr])*
            #[allow(unused_mut)]
            fn [< $name _ $float_str $(_ $gen_str:snake)* >]($($args)*) {
                #[allow(dead_code)]
                type $float_name = $float_ty;
                $(
                    #[allow(dead_code)]
                    type $gen_name = $gen_t;
                )*
                test_kernel! {
                    @inner
                    name: $name;
                    launch_types: [$float_ty];
                    gen_names: [$($gen_name),*];
                    args: ($($args)*);
                    $($rest)*
                }
            }
        }
    };

    // ==================== FUNCTION GENERATION (HETEROGENEOUS) ====================
    {
        types: [
            $( $gen_name:ident = $gen_t:ty as $gen_str:ident ; )*
        ];
        hetero: $fin_name:ident = $fin_ty:tt, $facc_name:ident = $facc_ty:tt;
        attrs: $(#[$attr:meta])*;
        name: $name:ident;
        args: ($($args:tt)*);
        $($rest:tt)*
    } => {
        $crate::test_kernel! {
            @gen_hetero_fn
            fin: $fin_name = $fin_ty;
            facc: $facc_name = $facc_ty;
            types: [ $( $gen_name = $gen_t as $gen_str ; )* ];
            attrs: $(#[$attr])*;
            name: $name;
            args: ($($args)*);
            $($rest)*
        }
    };

    {
        @gen_hetero_fn
        fin: $fin_name:ident = $fin_ty:tt;
        facc: $facc_name:ident = $facc_ty:tt;
        types: [ $( $gen_name:ident = $gen_t:ty as $gen_str:ident ; )* ];
        attrs: $(#[$attr:meta])*;
        name: $name:ident;
        args: ($($args:tt)*);
        $($rest:tt)*
    } => {
        ::paste::paste! {
            $(#[$attr])*
            #[allow(unused_mut)]
            fn [< $name _ $fin_ty _ $facc_ty $(_ $gen_str:snake)* >]($($args)*) {
                #[allow(dead_code)]
                type $fin_name = test_kernel!{ @resolve_float_ty $fin_ty };
                #[allow(dead_code)]
                type $facc_name = test_kernel!{ @resolve_float_ty $facc_ty };
                $(
                    #[allow(dead_code)]
                    type $gen_name = $gen_t;
                )*
                test_kernel! {
                    @inner
                    name: $name;
                    launch_types: [test_kernel!{ @resolve_float_ty $fin_ty }, test_kernel!{ @resolve_float_ty $facc_ty }];
                    gen_names: [$($gen_name),*];
                    args: ($($args)*);
                    $($rest)*
                }
            }
        }
    };

    { @resolve_float_ty bf16 } => { ::half::bf16 };
    { @resolve_float_ty f16 } => { ::half::f16 };
    { @resolve_float_ty f32 } => { f32 };
    { @resolve_float_ty f64 } => { f64 };

    // ==================== TEST BODY (@inner) ====================
    // Generates the actual test implementation: setup, kernel launch, and verification
    {
        @inner
        name: $name:ident;
        launch_types: [$($launch_ty:ty),+];
        gen_names: [$($($gen_name:ident),+)?];
        args: ($($args:tt)*);
        seed: ($($seed:expr)?);
        vars: $(let $var:ident: $vty:tt < $var_float:ident > = [$($val:expr),*] $(as $distrib:ident $(($($distrib_param:expr),+))?)?;)*;
        preamble: { $($preamble:stmt)* };
        kernel: $kernel:ident;
        kernel_args: ($($kernel_arg_name:ident($($kernel_arg:expr)?)),*);
        count: ($($count:expr),*);
        dim: $($max:ident)? ($($dim:expr),*);
        ref: $ref:expr;
    } => {
        ::paste::paste! {
            // 1. Setup: get compute client and RNG
            let client = $crate::test_utils::client();

            use rand::SeedableRng;
            #[allow(unused_variables)]
            let mut rng = rand::rngs::StdRng::seed_from_u64(test_kernel!{ @seed($name) ($($seed)?) });

            // 2. Initialize variables: create data, shapes and strides.
            //    We're passing all the identifiers here because if they were
            //    generated by the macro, they would be unique and not accessible
            //    outside of that invocation due to hygiene.
            $(
            test_kernel!{ @val($var_float, rng, client) $vty = [$($val),*] $(as $distrib $(($($distrib_param),+))?)?;
                [< $var _shape >], [< $var _strides >], [< $var _len >], $var
            }
            )*

            // 2. Run the preamble, if given
            $($preamble)*

            // 2. Upload variables to GPU and create CubeCL args.
            $(
            test_kernel!{ @upload($var_float, rng, client) $vty = [$($val),*] ;
                [< $var _shape >], [< $var _strides >], [< $var _len >],
                $var, [< $var _handle >], [< $var _arg >]
            }
            )*

            // 3. Launch the kernel
            println!("Launching kernel");
            $kernel::launch::<$($launch_ty),+ $(, $($gen_name),*)?, $crate::test_utils::TestRuntime>(
                &client,
                CubeCount::Static($(($count) as u32),*),
                test_kernel!{ @dim(client) $($max)? ($($dim),*) },
                $(
                    test_kernel!{ @arg([<$kernel_arg_name _arg>]) $kernel_arg_name($($kernel_arg)?) }
                ),*
            ).expect("Kernel launch failed");

            // 4. Compute reference values (mutates the local `$var` vectors)
            println!("Computing reference");
            $ref;

            // 5. Download GPU results and compare against reference
            println!("Checking results");
            $(
            test_kernel!{ @check(client) $var == [< $var _handle >] }
            )*
        }
    };

    // ==================== HELPER: KERNEL ARGUMENTS (@arg) ====================
    // `var()` - pass the variable's kernel arg
    { @arg($arg_name:ident) $_:ident() } => { $arg_name };
    // `lit(expr)` - pass a literal value directly
    { @arg($arg_name:ident) lit($arg:expr) } => { $arg };
    // `scalar(expr)` - pass a scalar value arg
    { @arg($arg_name:ident) scalar($arg:expr) } => { ::cubecl::frontend::ScalarArg { elem: $arg } };

    // ==================== HELPER: SEED (@seed) ====================
    { @seed($name:ident) () } => { $crate::test_utils::string_to_seed(stringify!($name)) };  // Default seed
    { @seed($name:ident) ($seed:expr) } => { $seed };

    // ==================== HELPER: CUBE DIMENSIONS (@dim) ====================
    // `max(n)` - query device for max supported dimensions
    { @dim($client:ident) max($dim:expr) } => { CubeDim::new(&$client, $dim as usize) };
    // Fixed dimensions
    { @dim($client:ident) ($x:expr) } => { CubeDim::new_1d($x as u32) };
    { @dim($client:ident) ($x:expr, $y:expr) } => { CubeDim::new_2d($x as u32, $y as u32) };
    { @dim($client:ident) ($x:expr, $y:expr, $z:expr) } => { CubeDim::new_3d($x as u32, $y as u32, $z as u32) };

    // ==================== HELPER: VARIABLE INITIALIZATION (@val) ====================
    // Creates shape, strides, data vector, GPU handle, and kernel argument for a variable
    { @val($t:ty, $rng:ident, $client:ident) $vty:tt = [$($val:expr),*] $(as $distrib:tt $(($($distrib_param:expr),+))?)?;
        $shape:ident, $strides:ident, $len:ident, $data:ident
    } => {
        let $shape = vec![$($val),*];
        println!("Initializing {} with shape {:?}", stringify!($vty), $shape);
        let $strides = $crate::test_utils::get_strides(&$shape);
        let $len: usize = $shape.iter().product();
        let mut $data = test_kernel!{ @init_val($t, $rng, $len) $(as $distrib $(($($distrib_param),+))?)? };
    };

    { @upload($t:ty, $rng:ident, $client:ident) $vty:tt = [$($val:expr),*] ;
        $shape:ident, $strides:ident, $len:ident, $data:ident, $handle:ident, $arg:ident
    } => {
        let $handle = $crate::test_utils::upload(&$client, &$data);
        println!("Strides: {:?}", $strides);
        println!("Length: {}", $len);
        assert_eq!($len % $crate::LINE_SIZE, 0, "Length must be a multiple of LINE_SIZE");
        test_kernel!{ @make_arg($t) $vty; $handle, $strides, $shape, $len, $arg }
    };

    // ==================== HELPER: DATA INITIALIZATION (@init_val) ====================
    // No distribution specified: default to Uniform(-10, 10)
    { @init_val($t:ty, $rng:ident, $len:ident) } => {
        test_kernel!{ @init_val($t, $rng, $len) as Uniform(-10.0, 10.0) }
    };
    // `as Range` - sequential values 0, 1, 2, ...
    { @init_val($t:ty, $rng:ident, $len:ident) as Range } => {
        $crate::test_utils::range_vec::<$t>($len)
    };
    // `as Uniform(start, end)` - random values in [start, end)
    { @init_val($t:ty, $rng:ident, $len:ident) as Uniform($start:expr, $end:expr) } => {
        $crate::test_utils::random_vec::<$t>(&mut $rng, $len, $start, $end)
    };

    // ==================== HELPER: KERNEL ARG CONSTRUCTION (@make_arg) ====================
    // Build ArrayArg for `Array` type variables
    { @make_arg($t:ty) Array; $handle:ident, $strides:ident, $shape:ident, $len:ident, $arg:ident } => {
        let $arg: ArrayArg<'_, $crate::test_utils::TestRuntime> = unsafe {
            ArrayArg::from_raw_parts::<Line<$t>>(&$handle, $len, $crate::LINE_SIZE)
        };
    };
    // Build TensorArg for `Tensor` type variables
    { @make_arg($t:ty) Tensor; $handle:ident, $strides:ident, $shape:ident, $len:ident, $arg:ident } => {
        let $arg: TensorArg<'_, $crate::test_utils::TestRuntime> = unsafe {
            TensorArg::from_raw_parts::<Line<$t>>(&$handle, &$strides, &$shape, $crate::LINE_SIZE)
        };
    };

    // ==================== HELPER: RESULT VERIFICATION (@check) ====================
    // Downloads GPU data and compares against expected values
    { @check($client:ident) $var:ident == $handle:ident } => {
        paste::paste! {
            println!("Comparing {}", stringify!($var));
            let [< $var _kernel_data >] = $crate::test_utils::download(&$client, $handle);
            $crate::test_utils::slices_eq(&[< $var _kernel_data >], &$var, stringify!($var))
        }
    };
}

#[macro_export]
macro_rules! cartesian {
    // ==================== PUBLIC API ====================
    // Usage:
    // cartesian!(
    //     callback!(context);
    //     Name in [Type, Type as Alias];
    //     Name2 in [Type];
    // )
    //
    // Callback format:
    // callback!(
    //     types: [
    //         Name = Type as Alias;
    //         Name2 = Type;
    //     ];
    //     context
    // )
    {
        $callback:ident ! ( $($ctx:tt)* ) ;
        $( $name:ident in [ $($opts:tt)+ ] ; )+
    } => {
        $crate::cartesian! {
            @step_product
            ctx: { $callback ! ( $($ctx)* ) };     // Callback info
            stack: { };                            // Accumulated (Name, Type, Alias)
            rest: { $( $name in [ $($opts)+ ] ),+ }; // Remaining dimensions
        }
    };

    // ==================== STEP 1: PRODUCT (Depth) ====================
    // Branching: Are there more dimensions to process?

    // Case A: No more dimensions. We are at a leaf node. Run the callback.
    {
        @step_product
        ctx: { $callback:ident ! ( $($ctx:tt)* ) };
        stack: { $( { $n:ident, $t:ty, $a:ident } )* };
        rest: {};
    } => {
        $callback! {
            types:   [ $( $n = $t as $a ; )* ];
            $($ctx)*
        }
    };

    // Case B: More dimensions exist. Pop the first one and iterate over its options.
    {
        @step_product
        ctx: { $callback:ident ! ( $($ctx:tt)* ) };
        stack: { $($stack:tt)* };
        rest: { $name:ident in [ $($opts:tt)+ ] $(, $($tail:tt)*)? };
    } => {
        $crate::cartesian! {
            @step_iter
            ctx: { $callback ! ( $($ctx)* ) };
            stack: { $($stack)* };
            rest: { $($($tail)*)? };  // Save the tail for the next recursion
            current_dim: $name;
            opts: [ $($opts)+ ];  // The options to iterate horizontally
        }
    };

    // ==================== STEP 2: ITER (Breadth) ====================
    // Looping Logic: Iterate over options for the *current* dimension.

    // Base Case: No more options. Stop this branch.
    {
        @step_iter
        ctx: $ctx:tt;
        stack: $stack:tt;
        rest: $rest:tt;
        current_dim: $dim:ident;
        opts: [];
        $($resttt:tt)*
    } => {};

    // Option Type A: `Type as Alias` (Explicit naming)
    {
        @step_iter
        ctx: $ctx:tt;
        stack: { $($stack:tt)* };
        rest: $rest:tt;
        current_dim: $dim:ident;
        opts: [ $t:ty as $a:tt $(, $($next_opts:tt)*)? ]; // Match "Type as Alias"
    } => {
        // 1. Recurse DOWN (Product) with this option added to stack
        $crate::cartesian! {
            @step_product
            ctx: $ctx;
            stack: { $($stack)* { $dim, $t, $a } }; // Push
            rest: $rest;
        }
        // 2. Recurse SIDEWAYS (Iter) to handle the remaining options
        $crate::cartesian! {
            @step_iter
            ctx: $ctx;
            stack: { $($stack)* };
            rest: $rest;
            current_dim: $dim;
            opts: [ $($($next_opts)*)? ];
        }
    };

    // Option Type B: `Type` (Implicit naming: Alias = Type)
    // Matches if the option is just a single identifier or path
    {
        @step_iter
        ctx: $ctx:tt;
        stack: $stack:tt;
        rest: $rest:tt;
        current_dim: $dim:ident;
        opts: [ $t:ident $(, $($next_opts:tt)*)? ]; // Match single identifier
    } => {
        // Normalize to "Type as Type" and delegate to Type A
        $crate::cartesian! {
            @step_iter
            ctx: $ctx;
            stack: $stack;
            rest: $rest;
            current_dim: $dim;
            opts: [ $t as $t $(, $($next_opts)*)? ];
        }
    };
}

pub fn client() -> TestClient {
    TestRuntime::client(&<TestRuntime as cubecl::Runtime>::Device::default())
}

pub type TestClient = ComputeClient<TestRuntime>;

pub fn range_vec<F: TestFloat>(len: usize) -> Vec<F> {
    (0..len).map(|i| F::from_int(i as i64)).collect()
}

pub fn string_to_seed(s: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(s.as_bytes());
    hasher.finish()
}

pub fn random_vec<F: TestFloat>(rng: &mut StdRng, len: usize, start: f64, end: f64) -> Vec<F> {
    (0..len)
        .map(|_| F::from_f64(rng.random_range(start..end)))
        .collect()
}

pub fn upload<F: TestFloat>(client: &TestClient, data: &[F]) -> Handle {
    client.create_from_slice(F::as_bytes(data))
}

pub fn download<F: TestFloat>(client: &TestClient, handle: Handle) -> Vec<F> {
    F::from_bytes(&client.read_one(handle)).to_vec()
}

pub fn get_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

/// Assert two values are approximately equal.
pub fn approx_eq<F: TestFloat>(actual: F, expected: F) -> bool {
    let (a, e) = (actual.into_f64(), expected.into_f64());
    let diff = (a - e).abs();
    let tol = F::atol() + F::rtol() * e.abs();
    diff <= tol
}

/// Assert slices are approximately equal.
pub fn slices_eq<F: TestFloat>(actual: &[F], expected: &[F], ctx: &str) {
    assert_eq!(actual.len(), expected.len(), "{ctx}: length mismatch");

    let mut passed = true;

    let mut avg_magnitude_actual = 0.0;
    let mut avg_magnitude_expected = 0.0;

    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        if !approx_eq(a, e) {
            passed = false;
            println!("{ctx}[{i}] mismatch: expected {e}, got {a}");
        }
        avg_magnitude_actual += a.into_f64().abs();
        avg_magnitude_expected += e.into_f64().abs();
    }

    println!(
        "Average magnitude of actual values: {}",
        avg_magnitude_actual / actual.len() as f64
    );
    println!(
        "Average magnitude of expected values: {}",
        avg_magnitude_expected / expected.len() as f64
    );

    if !passed {
        panic!("{} mismatch", ctx);
    }
}

pub trait TestFloat: CubeElement + CubePrimitive + Float + Copy + Display {
    fn into_f64(self) -> f64;
    fn from_f64(v: f64) -> Self;
    fn rtol() -> f64;
    fn atol() -> f64;
}

impl TestFloat for f64 {
    fn into_f64(self) -> f64 {
        self
    }
    fn from_f64(v: f64) -> Self {
        v
    }
    fn rtol() -> f64 {
        1e-12
    }
    fn atol() -> f64 {
        1e-12
    }
}

impl TestFloat for f32 {
    fn into_f64(self) -> f64 {
        self as f64
    }
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn rtol() -> f64 {
        1e-4
    }
    fn atol() -> f64 {
        1e-4
    }
}

impl TestFloat for f16 {
    fn into_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        f16::from_f64(v)
    }
    fn rtol() -> f64 {
        1e-2
    }
    fn atol() -> f64 {
        1e-2
    }
}

impl TestFloat for bf16 {
    fn into_f64(self) -> f64 {
        self.to_f64()
    }
    fn from_f64(v: f64) -> Self {
        bf16::from_f64(v)
    }
    fn rtol() -> f64 {
        5e-2
    }
    fn atol() -> f64 {
        5e-2
    }
}

// These tests are for testing the macro, not any actual CubeCL code
#[cfg(test)]
mod test_macro_tests {
    use cubecl::{num_traits::Zero, prelude::*};
    use test_case::{test_case, test_matrix};

    use crate::{LINE_SIZE, test_utils::TestFloat};

    #[cube(launch)]
    fn noop<F: Float>(_x: Tensor<Line<F>>) {}

    #[cube(launch)]
    fn zero<F: Float>(x: &mut Tensor<Line<F>>) {
        for i in 0..x.len() {
            x[i] = Line::empty(LINE_SIZE).fill(F::from_int(0));
        }
    }

    test_kernel! {
        #[test_case(2, 2)]
        #[test_case(4, 4)]
        fn noop(a: usize, b: usize) for F in all {
            let x: Tensor = [a, b];
            assert_eq!(
                noop(x()) for (1, 1, 1) @ (1, 1),
                {}
            );
        }

        #[test_matrix([4, 8], [4, 8])]
        fn zero(a: usize, b: usize) for F in all {
            let x: Tensor = [a, b];
            assert_eq!(
                zero(x()) for (1, 1, 1) @ (1, 1),
                {
                    x.fill(F::zero());
                }
            );
        }

        #[test]
        #[should_panic = "x mismatch"]
        fn zero_panic() for F in all {
            let x: Tensor = [4];
            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {
                    x.fill(F::zero());
                }
            );
        }

        #[test]
        #[should_panic = "Length must be a multiple of LINE_SIZE"]
        fn not_line_aligned() for F in all {
            let x: Tensor = [3, 3];
            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {}
            );
        }

        #[test]
        fn consistent_seed_from_name() for F in all {
            let x: Tensor = [4];
            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {
                    x[0] = F::from_f64(-6.829865117115661);
                }
            );
        }

        #[test]
        fn consistent_seed_from_param() for F in all {
            seed(42*42);
            let x: Tensor = [4];

            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {
                    x[0] = F::from_f64(-2.9764995848432685);
                }
            );
        }

        #[test_matrix([4, 16])]
        fn range_vec(n: usize) for F in all {
            let x: Tensor = [n] as Range;

            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {
                    for (i, x) in x.iter_mut().enumerate() {
                        *x = F::from_int(i as i64);
                    }
                }
            );
        }

        #[test_matrix([4, 16])]
        fn preamble(n: usize) for F in all {
            let x: Tensor = [n] as Range;

            {
                x.fill(F::zero());
            }

            assert_eq!(
                noop(x()) for (1, 1, 1) @ max(1),
                {
                    x.iter().all(|x| *x == F::zero())
                }
            );
        }
    }
}
