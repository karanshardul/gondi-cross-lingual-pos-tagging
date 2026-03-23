import torch
import torch.nn as nn


class POSClassificationHead(nn.Module):
    """
    Simple POS classifier on top of XLM-R hidden states.

    For this project, each input word is treated as a single-token sequence.
    We use the representation of the first token (after XLM-R encoding)
    as the pooled representation.
    """

    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: [batch, seq_len, hidden] from encoder/adapter stack.
        Returns logits: [batch, num_labels]
        """
        # Use representation of first token (position 0)
        pooled = hidden_states[:, 0, :]
        x = self.dropout(pooled)
        logits = self.out_proj(x)
        return logits

