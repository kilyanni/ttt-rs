use tokenizers::tokenizer::Tokenizer as HfTokenizer;

pub trait TokenizerTrait: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize>;
    fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_token(&self) -> usize;
    fn eos_token(&self) -> usize;
    fn bos_token(&self) -> usize;
}

pub struct Tokenizer {
    inner: HfTokenizer,
    pad_token_id: usize,
    eos_token_id: usize,
    bos_token_id: usize,
}

impl Tokenizer {
    /// Creates a new Tokenizer with explicit token IDs.
    pub fn new(
        tokenizer: HfTokenizer,
        pad_token_id: usize,
        eos_token_id: usize,
        bos_token_id: usize,
    ) -> Self {
        Self {
            inner: tokenizer,
            pad_token_id,
            eos_token_id,
            bos_token_id,
        }
    }

    /// Load a tokenizer from either a HuggingFace model name or a local file path.
    ///
    /// - If the path exists on disk, loads from file
    /// - Otherwise, treats as a HuggingFace model name (e.g., "gpt2", "EleutherAI/gpt-neox-20b")
    ///
    /// Tries to automatically detect special tokens (specialized tokens first, generic fallbacks last).
    #[must_use]
    pub fn load(
        identifier: &str,
        pad_token: Option<&str>,
        eos_token: Option<&str>,
        bos_token: Option<&str>,
    ) -> Self {
        let path = std::path::Path::new(identifier);
        let tokenizer = if path.exists() {
            HfTokenizer::from_file(identifier).unwrap_or_else(|e| {
                panic!("Failed to load tokenizer from file '{identifier}': {e}")
            })
        } else {
            HfTokenizer::from_pretrained(identifier, None)
                .unwrap_or_else(|e| panic!("Failed to load tokenizer '{identifier}': {e}"))
        };

        Self::from_hf_tokenizer(tokenizer, pad_token, eos_token, bos_token)
    }

    /// Create from an already-loaded HuggingFace tokenizer with auto-detection of special tokens.
    /// Tries specialized tokens first, then falls back to generic ones.
    pub fn from_hf_tokenizer(
        tokenizer: HfTokenizer,
        pad_token: Option<&str>,
        eos_token: Option<&str>,
        bos_token: Option<&str>,
    ) -> Self {
        let eos_candidates = ["<eos>", "</s>", "[EOS]", "<|endoftext|>"];
        let bos_candidates = ["<bos>", "<s>", "[BOS]", "<|endoftext|>"];
        let pad_candidates = ["<pad>", "[PAD]", "</s>", "<|endoftext|>"];

        let eos_token_id = eos_token
            .and_then(|t| tokenizer.token_to_id(t))
            .or_else(|| eos_candidates.iter().find_map(|t| tokenizer.token_to_id(t)))
            .expect("Could not find EOS token") as usize;

        let bos_token_id = bos_token
            .and_then(|t| tokenizer.token_to_id(t))
            .or_else(|| bos_candidates.iter().find_map(|t| tokenizer.token_to_id(t)))
            .unwrap_or(eos_token_id as u32) as usize;

        let pad_token_id = pad_token
            .and_then(|t| tokenizer.token_to_id(t))
            .or_else(|| pad_candidates.iter().find_map(|t| tokenizer.token_to_id(t)))
            .unwrap_or(eos_token_id as u32) as usize;

        Self {
            inner: tokenizer,
            pad_token_id,
            eos_token_id,
            bos_token_id,
        }
    }

    /// GPT-2 tokenizer (vocab_size: 50257)
    #[must_use]
    pub fn gpt2() -> Self {
        Self::load("gpt2", None, None, None)
    }

    /// GPT-NeoX-20B tokenizer (vocab_size: 50432)
    #[must_use]
    pub fn gpt_neox_20b() -> Self {
        Self::load("EleutherAI/gpt-neox-20b", None, None, None)
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::gpt2()
    }
}

impl TokenizerTrait for Tokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        let encoding = self.inner.encode(text, add_special_tokens).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> String {
        let token_ids: Vec<u32> = token_ids
            .iter()
            .map(|&id| {
                id.try_into().expect(
                    "For some reason, burn mixes u32 and usize and during casting a value was OOB",
                )
            })
            .collect();
        self.inner.decode(&token_ids, skip_special_tokens).unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    fn pad_token(&self) -> usize {
        self.pad_token_id
    }

    fn eos_token(&self) -> usize {
        self.eos_token_id
    }

    fn bos_token(&self) -> usize {
        self.bos_token_id
    }
}
