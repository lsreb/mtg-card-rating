"""Joint model: structured card features + LoRA-adapted text encoder -> rating.

Unlike model.CardRatingNet (a small head trained on frozen, precomputed MiniLM
embeddings), this model's text tower is itself trainable -- feature extraction and
the regression head are optimized together in the same backward pass. See
train_lora.py for the training loop.
"""

import torch
from torch import nn

from mtg_rating.lora_text_encoder import LORA_RANK, LoraTextEncoder
from mtg_rating.text_embeddings import EMBEDDING_DIM, MODEL_NAME


class CardRatingNetJoint(nn.Module):
    def __init__(
        self,
        structured_dim: int,
        hidden_dim: int = 16,
        lora_rank: int = LORA_RANK,
        lora_dropout: float = 0.0,
        head_dropout: float = 0.0,
        base_model_path=MODEL_NAME,
    ):
        super().__init__()
        self.text_encoder = LoraTextEncoder(rank=lora_rank, dropout=lora_dropout, base_model_path=base_model_path)
        self.head = nn.Sequential(
            nn.Linear(structured_dim + EMBEDDING_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, structured: torch.Tensor, texts: list) -> torch.Tensor:
        text_embedding = self.text_encoder(texts)
        combined = torch.cat([structured, text_embedding], dim=-1)
        return self.head(combined).squeeze(-1)

    def trainable_parameters(self):
        return list(self.text_encoder.trainable_parameters()) + list(self.head.parameters())
