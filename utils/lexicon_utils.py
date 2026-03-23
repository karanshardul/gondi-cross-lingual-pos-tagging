from typing import Dict

import torch
import torch.nn.functional as F


def build_lexicon_lookup(df, word_column: str = "word", pos_column: str = "pos") -> Dict[str, str]:
    """
    Build a mapping from word -> POS tag string from a pandas DataFrame.
    """
    lex = {}
    for _, row in df.iterrows():
        word = str(row[word_column])
        pos = str(row[pos_column])
        lex[word] = pos
    return lex


def lexicon_guided_loss(
    words,
    logits: torch.Tensor,
    label2id: Dict[str, int],
    lexicon: Dict[str, str],
    lambda_weight: float,
) -> torch.Tensor:
    """
    Compute a lexicon-guided penalty.

    Since the dataset itself is a POS lexicon, we treat the lexicon label as
    a soft constraint and penalize the model when it assigns low probability
    to the lexicon POS:

        L_lex = mean(1 - p(y_lexicon | x))

    This is complementary to cross-entropy and makes the constraint explicit
    in the "proposed method" experiment.
    """
    if lambda_weight <= 0.0:
        return torch.tensor(0.0, device=logits.device)

    probs = F.softmax(logits, dim=-1)  # [batch, num_labels]
    penalties = []
    for i, word in enumerate(words):
        if word in lexicon:
            lex_pos = lexicon[word]
            if lex_pos not in label2id:
                continue
            idx = label2id[lex_pos]
            p_lex = probs[i, idx]
            penalties.append(1.0 - p_lex)

    if not penalties:
        return torch.tensor(0.0, device=logits.device)

    penalties_tensor = torch.stack(penalties)
    return lambda_weight * penalties_tensor.mean()

