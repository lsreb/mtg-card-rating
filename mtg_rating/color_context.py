"""Per-card, per-color-group card pool used by the trainable cross-attention
context in train_color_context.py -- unlike set_context.py's fixed whole-set
mean, this groups a card with only the commons/uncommons that share its
color, so the model can pick up genuine archetype-support signal instead of a
flat "whole set vibe" average.

v1 scope, deliberately narrow (see project discussion, premier_jet.md
section 6): each card is assigned to exactly one bucket, its *primary* color
(colors[0] in Scryfall's WUBRG-ordered `colors` list, or "colorless" if
colorless) -- a two-color gold card is grouped with its first color only, not
pooled across both. Extending to genuine two-color archetype groups (the "4
relevant pairs" idea) is a deliberate v2 step, not done here -- this keeps
buckets a clean partition of each set (no card in two buckets at once), which
avoids the gradient-ordering headache of averaging a card's context across
buckets processed at different training steps.

Still a *trainable* mechanism (unlike set_context's fixed mean), so it
inherits the same generalization caveat that ruled out full cross-card
attention earlier in this project: attention weights are fit from the content
of only ~26 sets' worth of distinct card pools. Cards within one bucket share
a near-identical informant pool, so their prediction errors are correlated --
the effective sample size for validating whether the mechanism generalizes to
a genuinely unseen set is somewhere between the card count (~6500) and the
bucket count (~156), not the full card count. A name-based train/test split
doesn't probe this (held-out cards still come from already-seen sets'
buckets); train_color_context.py's split_by_set option does.
"""

COLORS = ["W", "U", "B", "R", "G"]
COMMON_UNCOMMON = {"common", "uncommon"}
COLORLESS = "colorless"


def primary_color(scryfall_card: dict) -> str:
    colors = scryfall_card.get("colors") or []
    for c in COLORS:
        if c in colors:
            return c
    return COLORLESS


def build_color_buckets(records: list) -> dict:
    """records -> {set_code: {color: {"query": [idx...], "informant": [idx subset]}}}.

    Indices are positions into the given `records` list (kept stable across
    train/eval since both must index into the same underlying list).
    "informant" is the "query" subset restricted to common/uncommon rarity --
    the pool of cards that inform the attention context; "query" is every
    card that needs its own contextualized output, regardless of rarity.
    """
    buckets = {}
    for i, r in enumerate(records):
        color = primary_color(r["scryfall_card"])
        by_color = buckets.setdefault(r["set_code"], {})
        entry = by_color.setdefault(color, {"query": [], "informant": []})
        entry["query"].append(i)
        if r["scryfall_card"].get("rarity") in COMMON_UNCOMMON:
            entry["informant"].append(i)
    return buckets


def flatten_buckets(color_buckets: dict) -> list:
    """{set_code: {color: bucket}} -> [(set_code, color, bucket), ...]."""
    return [
        (set_code, color, bucket)
        for set_code, by_color in color_buckets.items()
        for color, bucket in by_color.items()
    ]
