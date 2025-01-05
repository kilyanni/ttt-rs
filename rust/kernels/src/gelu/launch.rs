use std::fmt::Debug;

use burn_cubecl::{
    CubeRuntime, FloatElement, kernel::into_contiguous, ops::numeric::empty_device,
    tensor::CubeTensor,
};

use super::types::{
    GeluBwdKernel, GeluInput, GeluOutput, GeluTanhBackwardBackwardKernel, GeluTanhBackwardKernel,
    GeluTanhKernel,
};
use crate::{
    gelu_tanh::{
        launch_gelu_bwd_forward, launch_gelu_tanh, launch_gelu_tanh_backward,
        launch_gelu_tanh_backward_backward,
    },
    kernel::FusedKernel,
};

fn empty_like<R: CubeRuntime, F: FloatElement>(template: &CubeTensor<R>) -> CubeTensor<R> {
    empty_device::<R, F>(
        template.client.clone(),
        template.device.clone(),
        template.shape.clone(),
    )
}

impl FusedKernel for GeluTanhKernel {
    type Inputs<T: Debug + Clone + Send> = GeluInput<T>;
    type Outputs<T: Debug + Clone + Send> = GeluOutput<T>;
    type SavedState<T: Debug + Clone + Send> = GeluInput<T>;
    type Config = ();

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
        _config: (),
    ) -> (GeluOutput<CubeTensor<R>>, GeluInput<CubeTensor<R>>) {
        let input = into_contiguous(inputs.input);
        let output = empty_like::<R, F>(&input);

        launch_gelu_tanh::<R, F>(&input.client, input.as_handle_ref(), output.as_handle_ref());

        (GeluOutput { output }, GeluInput { input })
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: GeluInput<CubeTensor<R>>,
        grad_outputs: GeluOutput<CubeTensor<R>>,
        _config: (),
    ) -> GeluInput<CubeTensor<R>> {
        let input = into_contiguous(saved.input);
        let grad_output = into_contiguous(grad_outputs.output);
        let grad_input = empty_like::<R, F>(&input);

        launch_gelu_tanh_backward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            grad_output.as_handle_ref(),
            grad_input.as_handle_ref(),
        );

        GeluInput { input: grad_input }
    }
}

impl FusedKernel for GeluBwdKernel {
    type Inputs<T: Debug + Clone + Send> = GeluInput<T>;
    type Outputs<T: Debug + Clone + Send> = GeluOutput<T>;
    type SavedState<T: Debug + Clone + Send> = GeluInput<T>;
    type Config = ();

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
        _config: (),
    ) -> (GeluOutput<CubeTensor<R>>, GeluInput<CubeTensor<R>>) {
        let input = into_contiguous(inputs.input);
        let output = empty_like::<R, F>(&input);

        launch_gelu_bwd_forward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
        );

        (GeluOutput { output }, GeluInput { input })
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: GeluInput<CubeTensor<R>>,
        grad_outputs: GeluOutput<CubeTensor<R>>,
        _config: (),
    ) -> GeluInput<CubeTensor<R>> {
        let input = into_contiguous(saved.input);
        let grad_output = into_contiguous(grad_outputs.output);
        let grad_input = empty_like::<R, F>(&input);

        launch_gelu_tanh_backward_backward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            grad_output.as_handle_ref(),
            grad_input.as_handle_ref(),
        );

        GeluInput { input: grad_input }
    }
}

impl FusedKernel for GeluTanhBackwardKernel {
    type Inputs<T: Debug + Clone + Send> = GeluInput<T>;
    type Outputs<T: Debug + Clone + Send> = GeluOutput<T>;
    type SavedState<T: Debug + Clone + Send> = GeluInput<T>;
    type Config = ();

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
        _config: (),
    ) -> (GeluOutput<CubeTensor<R>>, GeluInput<CubeTensor<R>>) {
        let input = into_contiguous(inputs.input);
        let output = empty_like::<R, F>(&input);

        launch_gelu_bwd_forward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
        );

        (GeluOutput { output }, GeluInput { input })
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        saved: GeluInput<CubeTensor<R>>,
        grad_outputs: GeluOutput<CubeTensor<R>>,
        _config: (),
    ) -> GeluInput<CubeTensor<R>> {
        let input = into_contiguous(saved.input);
        let grad_output = into_contiguous(grad_outputs.output);
        let grad_input = empty_like::<R, F>(&input);

        launch_gelu_tanh_backward_backward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            grad_output.as_handle_ref(),
            grad_input.as_handle_ref(),
        );

        GeluInput { input: grad_input }
    }
}

impl FusedKernel for GeluTanhBackwardBackwardKernel {
    type Inputs<T: Debug + Clone + Send> = GeluInput<T>;
    type Outputs<T: Debug + Clone + Send> = GeluOutput<T>;
    type SavedState<T: Debug + Clone + Send> = GeluInput<T>;
    type Config = ();

    fn forward_launch<R: CubeRuntime, F: FloatElement>(
        inputs: GeluInput<CubeTensor<R>>,
        _config: (),
    ) -> (GeluOutput<CubeTensor<R>>, GeluInput<CubeTensor<R>>) {
        let input = into_contiguous(inputs.input);
        let output = empty_like::<R, F>(&input);

        launch_gelu_tanh_backward_backward::<R, F>(
            &input.client,
            input.as_handle_ref(),
            output.as_handle_ref(),
            output.as_handle_ref(),
        );

        (GeluOutput { output }, GeluInput { input })
    }

    fn backward_launch<R: CubeRuntime, F: FloatElement>(
        _saved: GeluInput<CubeTensor<R>>,
        _grad_outputs: GeluOutput<CubeTensor<R>>,
        _config: (),
    ) -> GeluInput<CubeTensor<R>> {
        panic!("Third-order gradients through GELU are not supported")
    }
}
