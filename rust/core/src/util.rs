use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Initializer, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Int, Tensor, activation::silu, module::conv1d, ops::ConvOptions},
};

pub fn causal_conv1d_fn<B: Backend>(
    x: Tensor<B, 3>,            // [batch_size, dim, seq_len]
    weight: Tensor<B, 3>,       // [channels_out, 1, kernel_size]
    bias: Option<Tensor<B, 1>>, // [channels_out]
) -> Tensor<B, 3> {
    let [batch_size, input_channels, seq_len] = x.shape().dims();
    let [channels_out, channels_in_per_group, kernel_size] = weight.shape().dims();

    debug_assert_eq!(
        input_channels,
        channels_out,
        "Input channels ({}) must match weight channels_out ({}). Input shape: {:?}, Weight shape: {:?}",
        input_channels,
        channels_out,
        x.shape(),
        weight.shape()
    );
    debug_assert_eq!(
        channels_in_per_group, 1,
        "Expected channels_in_per_group to be 1, got {channels_in_per_group}"
    );

    let out = conv1d(
        x,
        weight,
        bias,
        ConvOptions::new([1], [kernel_size - 1], [1], channels_out),
    );

    out.slice([0..batch_size, 0..channels_out, 0..seq_len])
}

#[derive(Config, Debug)]
pub struct CausalConvConfig {
    hidden_size: usize,
    kernel_size: usize,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct CausalConv<B: Backend> {
    pub weight: Param<Tensor<B, 3>>,
    pub bias: Param<Tensor<B, 1>>,
}

impl CausalConvConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> CausalConv<B> {
        let weight = self.initializer.init_with(
            [self.hidden_size, 1, self.kernel_size],
            Some(self.kernel_size),
            None,
            device,
        );
        let bias =
            self.initializer
                .init_with([self.hidden_size], Some(self.kernel_size), None, device);
        CausalConv { weight, bias }
    }
}

impl<B: Backend> CausalConv<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_channels_out, channels_in_per_group, _kernel_size] = self.weight.shape().dims();
        debug_assert_eq!(channels_in_per_group, 1);

        let x_transposed = x.permute([0, 2, 1]);
        let output = causal_conv1d_fn(x_transposed, self.weight.val(), Some(self.bias.val()));
        output.permute([0, 2, 1])
    }
}

#[derive(Config, Debug)]
pub struct MultiHeadLayerNormConfig {
    num_heads: usize,
    head_dim: usize,
    #[config(default = 1e-6)]
    epsilon: f64,
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

#[derive(Module, Debug)]
pub struct MultiHeadLayerNorm<B: Backend> {
    /// [num_heads, value_size]
    pub weight: Param<Tensor<B, 2>>,
    /// [num_heads, value_size]
    pub bias: Param<Tensor<B, 2>>,
    pub epsilon: f64,
}

impl MultiHeadLayerNormConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> MultiHeadLayerNorm<B> {
        let len = self.num_heads * self.head_dim;
        let weight = self.initializer.init_with(
            [self.num_heads, self.head_dim],
            Some(len),
            Some(len),
            device,
        );
        let bias = self.initializer.init_with(
            [self.num_heads, self.head_dim],
            Some(len),
            Some(len),
            device,
        );
        MultiHeadLayerNorm {
            weight,
            bias,
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> MultiHeadLayerNorm<B> {
    fn weight_and_bias(&self) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let w = self.weight.val();
        let b = self.bias.val();
        let [num_heads, head_dim] = w.shape().dims();
        debug_assert_eq!([num_heads, head_dim], b.shape().dims());
        (
            w.reshape([1, num_heads, 1, head_dim]),
            b.reshape([1, num_heads, 1, head_dim]),
        )
    }

    /// # Parameters
    /// - `x`: Input tensor of shape `[batch_size, num_heads, seq_len, value_size]`.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let (var, mean) = x.clone().var_mean_bias(3);
        let std = (var + self.epsilon).sqrt();

        let norm = (x - mean) / std.clone();

        let (weight, bias) = self.weight_and_bias();

        weight * norm + bias
    }

    /// # Parameters
    /// - `x`: Input tensor of shape `[batch_size, num_heads, seq_len, value_size]`.
    /// - `target`: Target tensor of shape `[batch_size, num_heads, seq_len, value_size]`.
    pub fn forward_and_l2_grad(
        &self,
        x: Tensor<B, 4>,
        target: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch_size, _num_heads, _seq_len, value_size] = x.shape().dims();

        let (var, mean) = x.clone().var_mean_bias(3);
        let std = (var + self.epsilon).sqrt();

        let norm = (x - mean) / std.clone();

        let (weight, bias) = self.weight_and_bias();

        let out = weight.clone() * norm.clone() + bias;

        let dl_dout = out.clone() - target;

        let dl_dnorm = dl_dout * weight;

        let dl_dx_term1 = dl_dnorm.clone() * (value_size as f32);
        let dl_dx_term2 = dl_dnorm.clone().sum_dim(3);
        let dl_dx_term3 = norm.clone() * (dl_dnorm * norm).sum_dim(3);

        let dl_dx = (dl_dx_term1 - dl_dx_term2 - dl_dx_term3) / (std * (value_size as f32));

        (out, dl_dx)
    }
}

// burn-rs's SwiGlu is not quite the same,
// theirs does
//  silu(linear(input)) * linear(input)
// this one does wraps the entire expression with another linear layer,
// and merges the other two linear layers into one projection that gets split

#[derive(Module, Debug)]
pub struct SwiGluMlp<B: Backend> {
    pub up_gate_proj: Linear<B>,
    pub down_proj: Linear<B>,
    pub intermediate_size: usize,
}

#[derive(Config, Debug)]
pub struct SwiGluMlpConfig {
    hidden_size: usize,
    intermediate_size: usize,
}

impl SwiGluMlpConfig {
    pub fn init<B: Backend>(self, device: &B::Device) -> SwiGluMlp<B> {
        SwiGluMlp {
            up_gate_proj: LinearConfig::new(self.hidden_size, 2 * self.intermediate_size)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            down_proj: LinearConfig::new(self.intermediate_size, self.hidden_size)
                .with_bias(false)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                })
                .init(device),
            intermediate_size: self.intermediate_size,
        }
    }
}

impl<B: Backend> SwiGluMlp<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [gate, up] = self
            .up_gate_proj
            .forward(x)
            .split(self.intermediate_size, 2)
            .try_into()
            .unwrap();

        self.down_proj.forward(silu(gate) * up)
    }
}

// The burn implementation uses a different data layout, so we handroll here.

/// Rotary Position Embedding that matches the PyTorch/EasyLM reference implementation.
/// This uses the "non-interleaved" or "GPT-NeoX style" rotary embedding with rotate_half.
#[derive(Module, Debug)]
pub struct RotaryEmbedding<B: Backend> {
    /// Precomputed inverse frequencies: [head_dim / 2]
    pub inv_freq: Tensor<B, 1>,
    /// Head dimension
    pub head_dim: usize,
}

/// Configuration for RotaryEmbedding
#[derive(Config, Debug)]
pub struct RotaryEmbeddingConfig {
    /// Head dimension (must be even)
    pub head_dim: usize,
    /// Base frequency for computing inverse frequencies (default: 10000.0)
    #[config(default = "10000.0")]
    pub base: f64,
}

impl RotaryEmbeddingConfig {
    /// Initialize RotaryEmbedding with the given configuration
    pub fn init<B: Backend>(&self, device: &B::Device) -> RotaryEmbedding<B> {
        assert!(
            self.head_dim.is_multiple_of(2),
            "head_dim must be even for rotary embedding"
        );

        // Compute inverse frequencies: 1 / (base^(2i/head_dim)) for i = 0, 1, ..., head_dim/2 - 1
        let half_dim = self.head_dim / 2;
        let inv_freq_data: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / self.base.powf((2 * i) as f64 / self.head_dim as f64) as f32)
            .collect();

        let inv_freq = Tensor::<B, 1>::from_floats(inv_freq_data.as_slice(), device);

        RotaryEmbedding {
            inv_freq,
            head_dim: self.head_dim,
        }
    }
}

impl<B: Backend> RotaryEmbedding<B> {
    /// Compute cos and sin for positions, optionally wrapping at mini_batch_size.
    /// If `mini_batch_size` is `Some`, positions wrap at that boundary.
    /// If `None`, positions are used as-is (global positions).
    /// Returns (cos, sin) each of shape [seq_len, head_dim]
    pub fn get_cos_sin(
        &self,
        seq_len: usize,
        offset: usize,
        mini_batch_size: Option<usize>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let device = self.inv_freq.device();

        // Position indices, optionally wrapping at mini_batch_size boundaries
        // e.g. for seq_len=32, offset=0, mini_batch_size=Some(16): [0,1,...,15, 0,1,...,15]
        // e.g. for seq_len=32, offset=0, mini_batch_size=None:     [0,1,...,31]
        // Compute entirely on device to avoid host-device transfer
        // Note: Do arithmetic in float space to avoid HIP backend bug with int remainder
        let pos_tensor =
            Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device).float() + (offset as f32);
        let pos_tensor = match mini_batch_size {
            Some(mbs) => pos_tensor.remainder_scalar(mbs as f32),
            None => pos_tensor,
        };

        // Compute frequencies: pos @ inv_freq -> [seq_len, head_dim/2]
        // pos: [seq_len, 1], inv_freq: [1, head_dim/2]
        let pos_2d = pos_tensor.unsqueeze_dim::<2>(1); // [seq_len, 1]
        let inv_freq_2d = self.inv_freq.clone().unsqueeze_dim::<2>(0); // [1, head_dim/2]
        let freqs = pos_2d.matmul(inv_freq_2d); // [seq_len, head_dim/2]

        // Double the frequencies to match head_dim: [seq_len, head_dim]
        let freqs_full = Tensor::cat(vec![freqs.clone(), freqs], 1); // [seq_len, head_dim]

        let cos = freqs_full.clone().cos();
        let sin = freqs_full.sin();

        (cos, sin)
    }

    /// Apply rotary position embedding to query and key tensors
    /// q, k: [batch_size, num_heads, seq_len, head_dim]
    /// If `mini_batch_size` is `Some`, positions wrap at that boundary.
    /// If `None`, positions are used as-is (global positions).
    /// Returns: (q_rotated, k_rotated) with same shape
    pub fn apply(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        offset: usize,
        mini_batch_size: Option<usize>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [_batch_size, _num_heads, seq_len, _head_dim] = q.shape().dims();

        let (cos, sin) = self.get_cos_sin(seq_len, offset, mini_batch_size);

        // Unsqueeze for broadcasting: [1, 1, seq_len, head_dim]
        let cos = cos.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);
        let sin = sin.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0);

        let q_rotated = apply_rotary_pos_emb_single(q, cos.clone(), sin.clone());
        let k_rotated = apply_rotary_pos_emb_single(k, cos, sin);

        (q_rotated, k_rotated)
    }
}

/// Rotate half of the hidden dims (GPT-NeoX style)
/// x: [..., head_dim] -> [-x2, x1] where x = [x1, x2] split at head_dim/2
fn rotate_half<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let shape = x.shape();
    let head_dim = shape.dims[D - 1];
    let half_dim = head_dim / 2;
    let dims = shape.dims;
    let device = x.device();

    let x1 = x.clone().slice(build_slice_ranges::<D>(&dims, 0, half_dim));
    let x2 = x.slice(build_slice_ranges::<D>(&dims, half_dim, head_dim));

    // Preallocate full-size output and use slice_assign for contiguous writes
    // output[..., :half_dim] = -x2, output[..., half_dim:] = x1
    let output: Tensor<B, D> = Tensor::zeros(&dims, &device);
    let output = output.slice_assign(build_slice_ranges::<D>(&dims, 0, half_dim), x2.neg());
    output.slice_assign(build_slice_ranges::<D>(&dims, half_dim, head_dim), x1)
}

/// Build slice ranges for extracting part of the last dimension
fn build_slice_ranges<const D: usize>(
    dims: &[usize],
    start: usize,
    end: usize,
) -> [std::ops::Range<usize>; D] {
    assert_eq!(dims.len(), D);
    let mut ranges: [std::ops::Range<usize>; D] = std::array::from_fn(|i| 0..dims[i]);

    ranges[D - 1] = start..end;
    ranges
}

/// Apply rotary position embedding to a single tensor
/// x: [batch_size, num_heads, seq_len, head_dim]
/// cos, sin: [1, 1, seq_len, head_dim]
fn apply_rotary_pos_emb_single<B: Backend>(
    x: Tensor<B, 4>,
    cos: Tensor<B, 4>,
    sin: Tensor<B, 4>,
) -> Tensor<B, 4> {
    x.clone() * cos + rotate_half(x) * sin
}
