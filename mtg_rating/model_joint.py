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


class ColorAttentionContext(nn.Module):
    """Small cross-attention block used by train_color_context.py: a card's own
    (structured+text) vector as the query, the commons/uncommons sharing its
    primary color (see color_context.py) as keys/values. Kept deliberately
    small -- one linear projection down to context_dim, one attention layer, a
    residual -- both to stay "light" per the project's own finding that added
    capacity without added information tends not to help here, and because a
    smaller function class is less able to overfit the handful of distinct
    color-pool "environments" (~156 set x color buckets) it's trained on. See
    color_context.py's docstring for the generalization caveat this implies.
    """

    def __init__(self, input_dim: int, context_dim: int = 32, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(input_dim, context_dim)
        self.attn = nn.MultiheadAttention(context_dim, num_heads, dropout=dropout, batch_first=True)
        self.context_dim = context_dim

    def forward(self, combined: torch.Tensor, informant_local_idx: list) -> torch.Tensor:
        # combined: (n, input_dim), every card in one (set, color) bucket, in
        # query order. informant_local_idx: which rows of `combined` are
        # allowed as keys/values (the common/uncommon subset) -- falls back to
        # the whole bucket if it happens to have zero c/u members.
        proj = self.proj(combined)
        kv = proj[informant_local_idx] if informant_local_idx else proj
        attn_out, _ = self.attn(proj.unsqueeze(0), kv.unsqueeze(0), kv.unsqueeze(0))
        return attn_out.squeeze(0) + proj

    def trainable_parameters(self):
        return list(self.parameters())


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Identity in the forward pass; negates (and scales by `lambda_`) whatever
    gradient flows back through it. Used by train_color_context.py's DANN mode
    to push ColorAttentionContext's output towards not encoding which set a
    card came from -- see that module for why set-identity is a real shortcut
    risk given only ~18-26 distinct sets. `lambda_` is meant to be ramped up
    over training (0 -> 1), not held constant -- adversarial pressure against
    an undertrained encoder early on is a known destabilizer."""

    def __init__(self, lambda_: float = 1.0):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_)


class DomainClassifier(nn.Module):
    """Small MLP predicting which (training) set a context vector came from --
    paired with GradientReversalLayer ahead of it, its own weights train
    normally to classify well, while the *encoder* upstream of the GRL gets
    pushed the opposite direction (see GradientReversalLayer's docstring)."""

    def __init__(self, context_dim: int, num_domains: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, num_domains),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.net(context)
