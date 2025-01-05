//! Configuration types shared between TTT crates.

#[cfg(feature = "burn")]
pub use burn::config::Config;

mod train;
mod types;

pub use train::*;
pub use types::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_arch_from_size() {
        let arch = ModelArch::from_size(ModelSize::M60, 50257);
        assert_eq!(arch.hidden_size, 512);
        assert_eq!(arch.num_hidden_layers, 6);
        assert_eq!(arch.vocab_size, 50257);
    }

    #[test]
    fn test_train_params_default() {
        let params = TrainParams::default();
        assert_eq!(params.tokenizer, "gpt2");
        assert_eq!(params.train.epochs, 10);
        assert_eq!(params.train.batch, 32);
    }

    #[test]
    fn test_enum_serde() {
        assert_eq!(
            serde_json::from_str::<InnerModel>("\"linear\"").unwrap(),
            InnerModel::Linear
        );
        assert_eq!(
            serde_json::from_str::<ModelSize>("\"60m\"").unwrap(),
            ModelSize::M60
        );
        assert_eq!(
            serde_json::from_str::<PosEncoding>("\"rope\"").unwrap(),
            PosEncoding::Rope
        );
    }

    #[test]
    fn test_inner_model_from_str() {
        assert_eq!("linear".parse::<InnerModel>().unwrap(), InnerModel::Linear);
        assert_eq!(
            "fused-naive".parse::<InnerModel>().unwrap(),
            InnerModel::FusedNaiveLinear
        );
        assert_eq!(
            "fused".parse::<InnerModel>().unwrap(),
            InnerModel::FusedNaiveLinear
        );
        assert_eq!(
            "fused-linear".parse::<InnerModel>().unwrap(),
            InnerModel::FusedNaiveLinear
        );
        assert_eq!(
            "fused-tile".parse::<InnerModel>().unwrap(),
            InnerModel::FusedTileLinear
        );
        assert!("bogus".parse::<InnerModel>().is_err());
    }

    #[test]
    fn test_mix_pattern_uniform() {
        let mix: MixPattern = "linear".parse().unwrap();
        assert_eq!(mix.0, vec![InnerModel::Linear]);
        assert!(mix.is_uniform());
        assert_eq!(mix.to_string(), "linear");
    }

    #[test]
    fn test_mix_pattern_alternating() {
        let mix: MixPattern = "linear,mlp".parse().unwrap();
        assert_eq!(mix.0, vec![InnerModel::Linear, InnerModel::Mlp]);
        assert_eq!(mix.get(0), InnerModel::Linear);
        assert_eq!(mix.get(1), InnerModel::Mlp);
        assert_eq!(mix.get(2), InnerModel::Linear); // cycles
        assert_eq!(mix.to_string(), "linear,mlp");
    }

    #[test]
    fn test_mix_pattern_multiplier() {
        let mix: MixPattern = "4*linear,mlp".parse().unwrap();
        assert_eq!(mix.0.len(), 5);
        assert_eq!(mix.get(3), InnerModel::Linear);
        assert_eq!(mix.get(4), InnerModel::Mlp);
        assert_eq!(mix.to_string(), "4*linear,mlp");
    }

    #[test]
    fn test_mix_pattern_serde_roundtrip() {
        let mix: MixPattern = "4*linear,2*mlp".parse().unwrap();
        let json = serde_json::to_string(&mix).unwrap();
        assert_eq!(json, "\"4*linear,2*mlp\"");
        let decoded: MixPattern = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded, mix);
    }

    #[test]
    fn test_mix_pattern_serde_backward_compat() {
        // Old configs stored just "linear" as a string
        let decoded: MixPattern = serde_json::from_str("\"linear\"").unwrap();
        assert_eq!(decoded, MixPattern::uniform(InnerModel::Linear));
    }
}
