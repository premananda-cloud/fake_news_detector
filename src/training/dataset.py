"""
Loads the pre-unified train/val/test TSVs (text\tlabel) and tokenizes them
for a given model checkpoint. No re-scraping or re-unification needed —
datasets/datasets/processed/unified_{train,val,test}.tsv already exist.
"""

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from src.config import DATA_FILES, TRAIN_CONFIG


def load_raw_splits() -> DatasetDict:
    """Read the three unified TSVs into a HF DatasetDict."""
    splits = {}
    for split_name, path in DATA_FILES.items():
        df = pd.read_csv(path, sep='\t')
        # Defensive: drop any accidental blank rows, enforce int labels
        df = df.dropna(subset=['text', 'label'])
        df['label'] = df['label'].astype(int)
        splits[split_name] = Dataset.from_pandas(df[['text', 'label']], preserve_index=False)
    return DatasetDict(splits)


def tokenize_splits(raw: DatasetDict, model_name: str) -> tuple[DatasetDict, AutoTokenizer]:
    """Tokenize all splits with the given model's tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = TRAIN_CONFIG['max_length']

    def _tokenize(batch):
        return tokenizer(
            batch['text'],
            truncation=True,
            padding='max_length',
            max_length=max_length,
        )

    tokenized = raw.map(_tokenize, batched=True, remove_columns=['text'])
    tokenized = tokenized.rename_column('label', 'labels')
    tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized, tokenizer


def get_tokenized_dataset(model_name: str) -> tuple[DatasetDict, AutoTokenizer]:
    raw = load_raw_splits()
    return tokenize_splits(raw, model_name)
