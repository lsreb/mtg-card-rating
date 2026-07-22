"""Frozen pretrained-text embeddings for card oracle text.

Uses plain `transformers` (AutoTokenizer/AutoModel) rather than the
`sentence-transformers` wrapper, to avoid its extra scipy/scikit-learn/Pillow
dependencies -- mean-pooling over token embeddings is a few lines on its own. The
pretrained model's weights are never updated (frozen): it is only used as a fixed
feature extractor, with a small trainable head (see model.py) doing the actual
learning on top of these embeddings.
"""

import torch
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_tokenizer = None
_model = None


def _load():
    global _tokenizer, _model
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def embed_texts(texts: list) -> torch.Tensor:
    tokenizer, model = _load()
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoded)

    token_embeddings = output.last_hidden_state
    mask = encoded["attention_mask"].unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_text(text: str) -> list:
    return embed_texts([text or ""])[0].tolist()
