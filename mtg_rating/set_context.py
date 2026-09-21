"""Per-set context vector: a fixed (non-trainable) summary of "what's in this
set", concatenated to every card's own features so the rating head can adjust
for set-level effects (color distribution, curve, thematic keywords) without
needing per-card synergy detection.

Deliberately simple, per project discussion: a full cross-card attention
mechanism was considered and rejected -- with only ~26 sets total (26 when this was decided, 27 now), splitting
train/val/test at the *set* level (required if cards within a set must be
seen together) would leave ~18 training sets, far too few to learn a
generalizable attention mechanism. A plain mean over the set's own cards has
no such problem: it's a fixed aggregation (no parameters to overfit or that
need to generalize across sets), well-defined for any set including ones
never seen in training, and the head that consumes it is still trained
card-by-card on the existing ~5-6k row split -- no restructuring needed.

Uses the frozen MiniLM embedding (text_embeddings.embed_texts), not the
trainable LoRA encoder -- computing this from live LoRA embeddings would need
re-embedding every card of a set on every forward pass (expensive, and awkward
since a single training batch mixes cards from many different sets). Computed
once per set and cached; uses card text/structured features only (no
17Lands metrics), so including test-split cards in a set's own average is not
label leakage.
"""

import json
from pathlib import Path

from mtg_rating.color_context import primary_color
from mtg_rating.features import STRUCTURED_DIM, structured_features
from mtg_rating.text_embeddings import EMBEDDING_DIM, embed_texts

CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "set_context_vectors.json"
CACHE_PATH_PER_COLOR = Path(__file__).resolve().parent.parent / "data" / "raw" / "set_context_vectors_per_color.json"
CONTEXT_DIM = STRUCTURED_DIM + EMBEDDING_DIM


def _mean_vector(vectors: list) -> list:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def compute_set_context_vectors(records: list) -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())

    by_set = {}
    for r in records:
        by_set.setdefault(r["set_code"], []).append(r["scryfall_card"])

    result = {}
    for set_code, cards in by_set.items():
        structured = [structured_features(c) for c in cards]
        text_embeds = embed_texts([c.get("oracle_text", "") or "" for c in cards]).tolist()
        combined = [s + t for s, t in zip(structured, text_embeds)]
        result[set_code] = _mean_vector(combined)
        print(f"[set_context] {set_code}: {len(cards)} cards averaged")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(result))
    return result


def compute_per_color_context_vectors(records: list) -> dict:
    """Like compute_set_context_vectors, but grouped by (set, primary color)
    instead of by whole set -- a more targeted summary (e.g. "the average
    White card in this set" instead of "the average card in this set,
    blending all 5 colors together"), while remaining just as fixed/non-
    trainable as the whole-set mean -- same primary_color() grouping as
    color_context.py's (separately tried, trainable, and found to overfit by
    -set) ColorAttentionContext, but there's nothing here for a by-set split
    to catch, since a mean has no parameters to overfit in the first place.

    Returns {set_code: {color: vector}} rather than a flat dict, since the
    lookup key is now a pair, not a single set code.
    """
    if CACHE_PATH_PER_COLOR.exists():
        return json.loads(CACHE_PATH_PER_COLOR.read_text())

    by_set_color = {}
    for r in records:
        color = primary_color(r["scryfall_card"])
        by_set_color.setdefault(r["set_code"], {}).setdefault(color, []).append(r["scryfall_card"])

    result = {}
    for set_code, by_color in by_set_color.items():
        result[set_code] = {}
        for color, cards in by_color.items():
            structured = [structured_features(c) for c in cards]
            text_embeds = embed_texts([c.get("oracle_text", "") or "" for c in cards]).tolist()
            combined = [s + t for s, t in zip(structured, text_embeds)]
            result[set_code][color] = _mean_vector(combined)
        total = sum(len(v) for v in by_color.values())
        print(f"[set_context] {set_code}: per-color means over {total} cards ({len(by_color)} colors)")

    CACHE_PATH_PER_COLOR.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH_PER_COLOR.write_text(json.dumps(result))
    return result
