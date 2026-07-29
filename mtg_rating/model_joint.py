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
        hidden_dims: list = None,
        lora_rank: int = LORA_RANK,
        lora_dropout: float = 0.0,
        head_dropout: float = 0.0,
        base_model_path=MODEL_NAME,
        set_context_dim: int = 0,
        output_dim: int = 1,
        layer_norm: bool = False,
    ):
        super().__init__()
        # [16] reproduces the original single-hidden-layer head exactly. A
        # deeper head (e.g. [64, 16]) is worth trying now that set_context
        # doubled the input width (398 -> 796) -- the single-layer head
        # compresses that in one big step; more capacity here wasn't as
        # clearly justified when the input was smaller and there was no set
        # context to help combine.
        hidden_dims = hidden_dims or [16]
        self.output_dim = output_dim
        self.text_encoder = LoraTextEncoder(rank=lora_rank, dropout=lora_dropout, base_model_path=base_model_path)

        # layer_norm=False (default) keeps hidden_dims=[16] byte-for-byte the
        # original head -- Pre-LN (LayerNorm immediately before each Linear,
        # same convention as CardNetBig/SharedTrunkActorCritic in the sibling
        # Coinche RL project) is opt-in, meant for deeper heads like
        # [256, 64, 16] where a deeper unnormalized MLP is more prone to
        # unstable training (the earlier [64, 16] attempt without LayerNorm
        # already showed ~8x higher seed-to-seed val variance).
        layers = []
        prev_dim = structured_dim + EMBEDDING_DIM + set_context_dim
        for dim in hidden_dims:
            if layer_norm:
                layers.append(nn.LayerNorm(prev_dim))
            layers += [nn.Linear(prev_dim, dim), nn.GELU(), nn.Dropout(head_dropout)]
            prev_dim = dim
        if layer_norm:
            layers.append(nn.LayerNorm(prev_dim))
        layers.append(nn.Linear(prev_dim, output_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, structured: torch.Tensor, texts: list, set_context: torch.Tensor = None) -> torch.Tensor:
        text_embedding = self.text_encoder(texts)
        parts = [structured, text_embedding]
        if set_context is not None:
            parts.append(set_context)
        combined = torch.cat(parts, dim=-1)
        out = self.head(combined)
        # output_dim=1 (the original, single-target case): squeeze to (batch,)
        # so nothing downstream needs to special-case it. output_dim>1 (e.g.
        # jointly predicting IIH and GP WR): keep (batch, output_dim) as-is.
        return out.squeeze(-1) if self.output_dim == 1 else out

    def trainable_parameters(self):
        return list(self.text_encoder.trainable_parameters()) + list(self.head.parameters())
