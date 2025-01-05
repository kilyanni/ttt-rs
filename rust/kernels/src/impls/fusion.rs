use burn::{
    backend::ir::{InitOperationIr, OperationIr},
    tensor::ops::FloatTensor,
};
use burn_backend::TensorMetadata;
use burn_fusion::{
    Fusion, FusionBackend, NoOp, client::GlobalFusionClient, stream::OperationStreams,
};

use crate::{
    TensorBundle,
    kernel::{FusedKernel, FusedKernelBackend},
};

fn fusion_in<B: FusionBackend>(tensor: FloatTensor<Fusion<B>>) -> FloatTensor<B> {
    tensor.client.clone().resolve_tensor_float::<B>(tensor)
}

fn fusion_out<B: FusionBackend>(
    tensor: FloatTensor<B>,
    client: &GlobalFusionClient<B::FusionRuntime>,
) -> FloatTensor<Fusion<B>> {
    let shape = tensor.shape();
    let dtype = tensor.dtype();
    let handle = B::float_tensor_handle(tensor);
    let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));

    let mut new = client.register(
        OperationStreams::default(),
        OperationIr::Init(desc),
        NoOp::<B>::new(),
    );

    assert_eq!(new.len(), 1);
    new.pop().unwrap()
}

/// Helper trait to get the client from the first tensor in a bundle.
/// Generic over `B` to allow the compiler to infer the backend type.
pub trait HasClient<B: FusionBackend> {
    fn client(&self) -> &GlobalFusionClient<B::FusionRuntime>;
}

impl<K, B> FusedKernelBackend<K> for Fusion<B>
where
    K: FusedKernel,
    B: FusedKernelBackend<K> + FusionBackend,
    K::Inputs<FloatTensor<Self>>: HasClient<B>,
    K::SavedState<FloatTensor<Self>>: HasClient<B>,
    // Mapped<U> -> K::Inputs/Outputs/SavedState<U> conversions
    <K::Inputs<FloatTensor<Self>> as TensorBundle<FloatTensor<Self>>>::Mapped<FloatTensor<B>>:
        Into<K::Inputs<FloatTensor<B>>>,
    <K::Outputs<FloatTensor<B>> as TensorBundle<FloatTensor<B>>>::Mapped<FloatTensor<Self>>:
        Into<K::Outputs<FloatTensor<Self>>>,
    <K::Outputs<FloatTensor<Self>> as TensorBundle<FloatTensor<Self>>>::Mapped<FloatTensor<B>>:
        Into<K::Outputs<FloatTensor<B>>>,
    <K::SavedState<FloatTensor<B>> as TensorBundle<FloatTensor<B>>>::Mapped<FloatTensor<Self>>:
        Into<K::SavedState<FloatTensor<Self>>>,
    <K::SavedState<FloatTensor<Self>> as TensorBundle<FloatTensor<Self>>>::Mapped<FloatTensor<B>>:
        Into<K::SavedState<FloatTensor<B>>>,
    <K::Inputs<FloatTensor<B>> as TensorBundle<FloatTensor<B>>>::Mapped<FloatTensor<Self>>:
        Into<K::Inputs<FloatTensor<Self>>>,
{
    fn forward(
        inputs: K::Inputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> (
        K::Outputs<FloatTensor<Self>>,
        K::SavedState<FloatTensor<Self>>,
    ) {
        let client = inputs.client().clone();
        let inner_inputs = inputs.map(|t| fusion_in::<B>(t)).into();
        let (outputs, saved) = B::forward(inner_inputs, config);
        (
            outputs.map(|t| fusion_out::<B>(t, &client)).into(),
            saved.map(|t| fusion_out::<B>(t, &client)).into(),
        )
    }

    fn backward(
        saved: K::SavedState<FloatTensor<Self>>,
        grad_outputs: K::Outputs<FloatTensor<Self>>,
        config: K::Config,
    ) -> K::Inputs<FloatTensor<Self>> {
        let client = saved.client().clone();
        let inner_saved = saved.map(|t| fusion_in::<B>(t)).into();
        let inner_grad_outputs = grad_outputs.map(|t| fusion_in::<B>(t)).into();
        let grads = B::backward(inner_saved, inner_grad_outputs, config);
        grads.map(|t| fusion_out::<B>(t, &client)).into()
    }
}
