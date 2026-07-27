"""Trainable, MTG-specific text encoder: the same MiniLM base as text_embeddings.py,
but wrapped with LoRA adapters (rank 4, on the attention query/value projections)
that get updated during training instead of staying frozen.

Contrast with text_embeddings.py: that module treats MiniLM as a fixed feature
extractor (no gradient ever reaches it). Here, peft injects a small number of extra
parameters per attention layer (two 384x4 + 4x384 matrices for query, same for
value) that are trained jointly with the rating head (see model_joint.py) on pooled
multi-set card data -- the base 22.7M pretrained weights stay frozen throughout.
"""

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModel, AutoTokenizer

from mtg_rating.text_embeddings import EMBEDDING_DIM, MODEL_NAME

LORA_RANK = 4
LORA_TARGET_MODULES = ["query", "value"]


class LoraTextEncoder(nn.Module):
    def __init__(self, rank: int = LORA_RANK, dropout: float = 0.0):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = AutoModel.from_pretrained(MODEL_NAME)
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=dropout,
            bias="none",
        )
        self.model = get_peft_model(base_model, lora_config)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, texts: list) -> torch.Tensor:
        device = next(self.model.parameters()).device
        encoded = self.tokenizer([t or "" for t in texts], padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        output = self.model(**encoded)

        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        summed = (token_embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


assert EMBEDDING_DIM == 384  # LoraTextEncoder output width must match the frozen encoder's
