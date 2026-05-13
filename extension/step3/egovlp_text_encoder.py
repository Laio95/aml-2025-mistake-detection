"""
Extension Step 3 — EgoVLP Text Encoder
========================================
Wraps FrozenInTime's text encoder to produce 256-dim L2-normalized embeddings
from lists of text strings (task graph node descriptions).

The same FrozenInTime model used for video feature extraction is reused here:
we call compute_text() instead of compute_video(). The output space is aligned
with EgoVLP video embeddings, so cosine similarity between text and video
embeddings is semantically meaningful — this is what makes Hungarian matching
possible in hungarian_matcher.py.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import List

from core.features_extraction.segment_feature_extractor import _build_egovlp_model


_DISTILBERT_MODEL = "distilbert-base-uncased"
_MAX_TOKEN_LENGTH = 64   # recipe steps are short sentences; 64 tokens is always enough


class EgoVLPTextEncoder:
    """
    Encodes text strings into 256-dim L2-normalized embeddings using the
    EgoVLP text encoder (DistilBERT + linear projection fine-tuned on Ego4D).

    The model is frozen: no gradients flow through it.
    The learnable fusion happens later in realization_builder.py.

    Usage:
        encoder = EgoVLPTextEncoder(
            egovlp_repo="/content/EgoVLP",
            ckpt_path=EGOVLP_CKPT,
            device=torch.device("cuda"),
        )
        T = encoder.encode(["Crack two eggs", "Whisk the eggs"])
        # T: Tensor(2, 256), L2-normalized, on CPU
    """

    def __init__(self, egovlp_repo: str, ckpt_path: str, device: torch.device):
        """
        Args:
            egovlp_repo: path to cloned showlab/EgoVLP repo
            ckpt_path:   path to egovlp.pth checkpoint
            device:      torch device for inference
        """
        self.device = device

        # Load FrozenInTime — identical call to the one in segment_feature_extractor.py.
        # Both video encoder (ViT) and text encoder (DistilBERT) are loaded together
        # because they share the same checkpoint; we just won't call compute_video() here.
        self.model = _build_egovlp_model(egovlp_repo, ckpt_path, device)
        self.model.eval()

        # Freeze all parameters: EgoVLP is a fixed feature extractor in this pipeline
        for p in self.model.parameters():
            p.requires_grad_(False)

        # DistilBERT tokenizer: converts raw strings into token IDs.
        # Has no trainable weights — it's just vocabulary + tokenization rules.
        self.tokenizer = AutoTokenizer.from_pretrained(_DISTILBERT_MODEL)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> torch.Tensor:
        """
        Encode a list of strings into L2-normalized 256-dim embeddings.

        Args:
            texts: N strings (task graph node descriptions)

        Returns:
            Tensor(N, 256), float32, L2-normalized, on CPU.
            Dot product of two rows == cosine similarity between the two texts
            (or between a text and an L2-normalized video embedding).
        """
        # Tokenize: pads shorter texts, truncates longer ones, returns PyTorch tensors
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=_MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # FrozenInTime.compute_text() expects {"input_ids": ..., "attention_mask": ...}
        # Internally: DistilBERT → CLS pooling → linear projection → (N, 256)
        embeddings = self.model.compute_text(tokens)  # (N, 256)

        # L2-normalize so hungarian_matcher can use raw dot product as cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)  # (N, 256)

        return embeddings.cpu()
