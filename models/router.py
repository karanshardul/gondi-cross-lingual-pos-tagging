import torch
import torch.nn as nn
from typing import Dict, Optional


class AdapterRouter(nn.Module):
    """
    Dynamic adapter router with optional similarity bias.

    Given a hidden representation h (e.g., pooled CLS),
    computes language weights:

        α = softmax(W_r h)                (no bias)
        α = softmax(W_r h + s)            (with similarity bias)

    where s is a trainable or fixed similarity prior (here fixed from config).
    """

    def __init__(
        self,
        hidden_size: int,
        languages=None,
        similarity_bias: Optional[torch.Tensor] = None,
        use_similarity_bias: bool = False,
    ):
        super().__init__()
        if languages is None:
            languages = ["hi", "mr", "te"]
        self.languages = languages
        self.num_langs = len(languages)
        self.router = nn.Linear(hidden_size, self.num_langs)

        if similarity_bias is not None:
            if similarity_bias.shape[-1] != self.num_langs:
                raise ValueError("similarity_bias length must match number of languages")
            # Register as buffer so it moves with the model but is not trainable
            self.register_buffer("similarity_bias", similarity_bias.view(1, -1))
        else:
            self.register_buffer("similarity_bias", torch.zeros(1, self.num_langs))

        self.use_similarity_bias = use_similarity_bias

    def forward(self, pooled_hidden: torch.Tensor) -> torch.Tensor:
        """
        pooled_hidden: [batch, hidden_size]
        returns router weights α: [batch, num_languages]
        """
        logits = self.router(pooled_hidden)  # [batch, num_langs]
        if self.use_similarity_bias:
            logits = logits + self.similarity_bias
        return torch.softmax(logits, dim=-1)

    def combine_adapters(self, adapter_outputs: Dict[str, torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """
        Linearly combine adapter outputs using router weights.

        adapter_outputs: dict(lang -> [batch, seq_len, hidden])
        weights: [batch, num_langs]
        returns: [batch, seq_len, hidden]
        """
        batch_size, seq_len, hidden = next(iter(adapter_outputs.values())).shape
        stacked = []
        for idx, lang in enumerate(self.languages):
            # [batch, 1, 1] for broadcasting over seq_len and hidden
            w = weights[:, idx].view(batch_size, 1, 1)
            stacked.append(w * adapter_outputs[lang])
        combined = torch.stack(stacked, dim=0).sum(dim=0)
        return combined

