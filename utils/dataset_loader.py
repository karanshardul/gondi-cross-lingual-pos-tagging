import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


POS_COLUMN = "pos"
WORD_COLUMN = "word"


@dataclass
class GondiExample:
    word: str
    label: int


class GondiPosDataset(Dataset):
    """
    Dataset treating each Gondi word as a single-token sequence for POS classification.
    """

    def __init__(
        self,
        examples: List[GondiExample],
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[idx]
        encoded = self.tokenizer(
            ex.word,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(ex.label, dtype=torch.long)
        # Keep raw word string for lexicon-guided losses / interpretability.
        # Note: XLM-R may split words into subwords; our classifier uses the first token,
        # so the POS label is effectively applied to the first token representation.
        item["word"] = ex.word
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_lexicon(
    path: str, train_ratio: float, dev_ratio: float, test_ratio: float
) -> Tuple[List[GondiExample], List[GondiExample], List[GondiExample], Dict[str, int], Dict[int, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lexicon CSV not found at {path}")

    df = pd.read_csv(path)
    if WORD_COLUMN not in df.columns or POS_COLUMN not in df.columns:
        raise ValueError(f"CSV must contain '{WORD_COLUMN}' and '{POS_COLUMN}' columns")

    # Normalize POS tags to strings, build label mapping
    unique_pos = sorted(df[POS_COLUMN].astype(str).unique().tolist())
    label2id = {pos: i for i, pos in enumerate(unique_pos)}
    id2label = {i: pos for pos, i in label2id.items()}

    # Shuffle rows reproducibly (caller should set global seed)
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)

    n = len(df)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    n_test = n - n_train - n_dev

    train_df = df.iloc[:n_train]
    dev_df = df.iloc[n_train : n_train + n_dev]
    test_df = df.iloc[n_train + n_dev :]

    def to_examples(sub_df: pd.DataFrame) -> List[GondiExample]:
        return [
            GondiExample(word=str(row[WORD_COLUMN]), label=label2id[str(row[POS_COLUMN])])
            for _, row in sub_df.iterrows()
        ]

    train_examples = to_examples(train_df)
    dev_examples = to_examples(dev_df)
    test_examples = to_examples(test_df)

    return train_examples, dev_examples, test_examples, label2id, id2label


def create_tokenizer(pretrained_name: str) -> AutoTokenizer:
    """
    Create an XLM-R tokenizer that correctly handles Telugu Unicode script.
    XLM-R operates over sentencepiece subword units and is fully Unicode-compatible,
    so we only need to ensure that we do not normalize or strip accents.
    """
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    # Ensure no lowercasing or accent stripping for multilingual Unicode
    if hasattr(tokenizer, "do_lower_case"):
        tokenizer.do_lower_case = False
    return tokenizer

