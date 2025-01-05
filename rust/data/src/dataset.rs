// Taken from the burn-rs text-generation example

use burn::data::dataset::{Dataset, SqliteDataset, source::huggingface::HuggingfaceDatasetLoader};

#[derive(Clone, Debug)]
pub struct TextGenerationItem {
    pub text: String,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DatasetItem {
    pub text: String,
}

pub struct TextDataset {
    dataset: SqliteDataset<DatasetItem>,
}

impl Dataset<TextGenerationItem> for TextDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset
            .get(index)
            .map(|item| TextGenerationItem { text: item.text })
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl TextDataset {
    #[must_use]
    pub fn train() -> Self {
        Self::new("train")
    }

    #[must_use]
    pub fn test() -> Self {
        // Self::new("test")
        Self::new("validation")
    }

    #[must_use]
    pub fn new(split: &str) -> Self {
        // let dataset: SqliteDataset<DbPediaItem> = HuggingfaceDatasetLoader::new("dbpedia_14")
        let dataset: SqliteDataset<DatasetItem> =
            HuggingfaceDatasetLoader::new("roneneldan/TinyStories")
                .dataset(split)
                .unwrap();
        Self { dataset }
    }
}
