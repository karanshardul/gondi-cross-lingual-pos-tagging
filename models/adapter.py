import torch
import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """
    Simple bottleneck adapter:

        A(h) = h + W_up σ(W_down h)

    where W_down ∈ R(d × r), W_up ∈ R(r × d) and r << d.
    """

    def __init__(self, hidden_size: int, bottleneck_size: int, activation: str = "relu"):
        super().__init__()
        self.down = nn.Linear(hidden_size, bottleneck_size)
        self.up = nn.Linear(bottleneck_size, hidden_size)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [batch, seq_len, hidden_size]
        z = self.down(hidden_states)
        z = self.activation(z)
        z = self.up(z)
        return hidden_states + z


class MultiLanguageAdapters(nn.Module):
    """
    Container for multiple language-specific adapters (Hindi, Marathi, Telugu).
    """

    def __init__(self, hidden_size: int, bottleneck_size: int, languages=None, activation: str = "relu"):
        super().__init__()
        if languages is None:
            languages = ["hi", "mr", "te"]
        self.languages = languages
        self.adapters = nn.ModuleDict(
            {
                lang: BottleneckAdapter(hidden_size, bottleneck_size, activation=activation)
                for lang in languages
            }
        )

    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        Returns a dict mapping language codes to adapted hidden states.
        """
        outputs = {}
        for lang, adapter in self.adapters.items():
            outputs[lang] = adapter(hidden_states)
        return outputs

