use std::fmt::Debug;

use burn::{
    backend::autodiff::{
        Autodiff, NodeId,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    tensor::{FloatDType, ops::FloatTensor},
};
use burn_backend::{Shape, TensorMetadata};

use crate::{
    TensorBundle,
    kernel::{FusedKernel, FusedKernelBackend},
};

/// No-op backward used to wrap primitives into untracked `AutodiffTensors`.
/// With 0 parents, `prepare([])` always yields `UnTracked`, allowing us to
/// convert inner primitives to autodiff tensors without memory allocation.
#[derive(Debug)]
struct NoOpBackward;

impl<B: burn_backend::Backend> Backward<B, 0> for NoOpBackward {
    type State = ();

    fn backward(self, _ops: Ops<(), 0>, _grads: &mut Gradients, _checkpointer: &mut Checkpointer) {
        // Nothing to do - this is never called since we're always untracked
    }
}

/// Backward op for a specific output index.
/// Each output is tracked separately and runs the backward kernel independently.
/// Gradients to inputs accumulate via burn's gradient system.
#[derive(Debug)]
struct OutputBackwardOp<K, const N: usize, const M: usize, const S: usize> {
    output_idx: usize,
    _marker: std::marker::PhantomData<K>,
}

impl<K, const N: usize, const M: usize, const S: usize, B> Backward<B, N>
    for OutputBackwardOp<K, N, M, S>
where
    K: FusedKernel,
    B: FusedKernelBackend<K>,
    B::FloatTensorPrimitive: Send,
    K::SavedState<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; S]>,
    K::Outputs<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; M]>,
    K::Inputs<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; N]>,
{
    type State = (
        K::SavedState<B::FloatTensorPrimitive>, // Saved state from forward
        [(Vec<usize>, FloatDType); M],          // Output shapes for creating zeros
        [Option<NodeId>; N],                    // Input node IDs for gradient registration
        K::Config,                              // Saved config
    );

    fn backward(
        self,
        ops: Ops<Self::State, N>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<B>(&ops.node);
        let (saved_state, output_shapes, input_node_ids, config) = ops.state;

        // Get device from one of the saved tensors
        let saved_arr = saved_state.into_array();
        let device = B::float_device(&saved_arr[0]);

        // Build grad_outputs with this gradient at the appropriate index, zeros elsewhere
        let grad_outputs_arr: [B::FloatTensorPrimitive; M] = std::array::from_fn(|i| {
            if i == self.output_idx {
                grad_output.clone()
            } else {
                B::float_zeros(
                    Shape::from(output_shapes[i].0.clone()),
                    &device,
                    output_shapes[i].1,
                )
            }
        });
        let grad_outputs = K::Outputs::from_array(grad_outputs_arr);

        // Reconstruct saved state from the array
        let saved_state = K::SavedState::from_array(saved_arr);

        let grad_inputs = B::backward(saved_state, grad_outputs, config);

        // Register gradients for all tracked parents (accumulates if called multiple times)
        for (grad, node_id) in grad_inputs
            .into_array()
            .into_iter()
            .zip(input_node_ids.iter())
        {
            if let Some(id) = node_id {
                grads.register::<B>(*id, grad);
            }
        }
    }
}

impl<K, B, C, const N: usize, const M: usize, const S: usize> FusedKernelBackend<K>
    for Autodiff<B, C>
where
    K: FusedKernel,
    B: FusedKernelBackend<K>,
    C: CheckpointStrategy,
    B::FloatTensorPrimitive: Clone,
    // Extract N, M, S from the Array associated types
    K::Inputs<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; N]>,
    K::Outputs<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; M]>,
    K::SavedState<B::FloatTensorPrimitive>:
        TensorBundle<B::FloatTensorPrimitive, Array = [B::FloatTensorPrimitive; S]>,
    K::Inputs<FloatTensor<Self>>: TensorBundle<FloatTensor<Self>, Array = [FloatTensor<Self>; N]>,
    K::Outputs<FloatTensor<Self>>: TensorBundle<FloatTensor<Self>, Array = [FloatTensor<Self>; M]>,
    K::SavedState<FloatTensor<Self>>:
        TensorBundle<FloatTensor<Self>, Array = [FloatTensor<Self>; S]>,
{
    fn forward(
        inputs: K::Inputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> (
        K::Outputs<FloatTensor<Self>>,
        K::SavedState<FloatTensor<Self>>,
    ) {
        let input_arr = inputs.into_array();
        let nodes_array: [_; N] = input_arr.each_ref().map(|t| t.node.clone());
        let input_node_ids: [Option<NodeId>; N] = nodes_array.each_ref().map(|n| Some(n.id));
        let primitives: [_; N] = input_arr.map(|t| t.primitive.clone());

        let inner_inputs = K::Inputs::from_array(primitives);
        let (outputs, saved_state) = B::forward(inner_inputs, config.clone());
        let output_primitives: [_; M] = outputs.into_array();
        let saved_primitives: [_; S] = saved_state.into_array();

        // Save output shapes for creating zeros in backward
        let output_shapes: [_; M] = std::array::from_fn(|i| {
            (
                output_primitives[i].shape().dims.clone(),
                output_primitives[i].dtype().into(),
            )
        });

        // Save state for backward
        let saved_state_for_backward = K::SavedState::from_array(saved_primitives.clone());

        // Track each output separately
        let tracked_outputs: [FloatTensor<Self>; M] = std::array::from_fn(|idx| {
            let backward_op = OutputBackwardOp::<K, N, M, S> {
                output_idx: idx,
                _marker: std::marker::PhantomData,
            };

            match backward_op
                .prepare::<C>(nodes_array.clone())
                .compute_bound()
                .stateful()
            {
                OpsKind::Tracked(prep) => prep.finish(
                    (
                        saved_state_for_backward.clone(),
                        output_shapes.clone(),
                        input_node_ids,
                        config.clone(),
                    ),
                    output_primitives[idx].clone(),
                ),
                OpsKind::UnTracked(prep) => prep.finish(output_primitives[idx].clone()),
            }
        });

        // Wrap saved state primitives as untracked AutodiffTensors.
        let wrapped_saved: [FloatTensor<Self>; S] = std::array::from_fn(|i| {
            match NoOpBackward.prepare::<C>([]).compute_bound().stateful() {
                OpsKind::UnTracked(prep) => prep.finish(saved_primitives[i].clone()),
                OpsKind::Tracked(_) => unreachable!("0 parents always yields UnTracked"),
            }
        });

        (
            K::Outputs::from_array(tracked_outputs),
            K::SavedState::from_array(wrapped_saved),
        )
    }

    fn backward(
        _saved: K::SavedState<FloatTensor<Self>>,
        _grad_outputs: K::Outputs<FloatTensor<Self>>,
        _config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>> {
        panic!("Second-order gradients not supported")
    }
}
