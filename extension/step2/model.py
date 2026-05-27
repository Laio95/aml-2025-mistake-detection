"""
model.py
========
Extension Step 2 — Task Verification Baseline.

TaskVerifier: Transformer-based binary classifier that operates on a
variable-length sequence of step-level EgoVLP embeddings (one per detected
step from ActionFormer) and predicts whether the whole recipe execution is
correct (0) or incorrect (1).

Architecture:
    Input  (N_steps, 256)
        └─ unsqueeze(0)  →  (1, N_steps, d_model)
        └─ PositionalEncoding
        └─ TransformerEncoder  (num_layers × EncoderLayer)
        └─ mean pooling over steps  →  (1, d_model)
        └─ Linear(d_model → 1)  →  logit scalar
    Loss: BCEWithLogitsLoss(pos_weight=...)   [set in training loop]
"""

import torch
import torch.nn as nn

from core.models.blocks import PositionalEncoding


class TaskVerifier(nn.Module):
    """
    Transformer-based video-level binary classifier for Task Verification.

    Designed for batch_size=1: the input x has no batch dimension;
    the batch axis is added and removed internally so the rest of the
    training loop stays simple.

    Args:
        d_model        : embedding dimension (must match EgoVLP output = 256)
        nhead          : number of self-attention heads
        num_layers     : number of stacked TransformerEncoder layers
        dim_feedforward: inner dimension of the pointwise feedforward block
        dropout        : dropout probability (applied in PE and Transformer)
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()

        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, N_steps, d_model) — batch_size=1 from DataLoader default collate

        Returns:
            logit: (1,) scalar — raw logit; apply sigmoid for probability
        """
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)

        x = self.pos_encoding(x)  # (1, N_steps, d_model)
        x = self.encoder(x)       # (1, N_steps, d_model)
        x = x.mean(dim=1)         # (1, d_model)  — mean pool over steps
        x = self.classifier(x)    # (1, 1)
        return x.squeeze(-1)      # (1,)
