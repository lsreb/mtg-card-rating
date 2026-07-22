"""Turn a Scryfall card object into a fixed-size numeric feature vector.

Structured fields (mana value, colors, type, rarity, power/toughness) are kept
deliberately minimal for the v0 toy pipeline. Oracle text is folded in as a frozen
pretrained embedding (see text_embeddings.py) rather than ignored, so the model can
pick up on effects that the structured fields alone can't distinguish (e.g. two
3-mana white creatures with wildly different abilities).
"""

from mtg_rating.text_embeddings import EMBEDDING_DIM, embed_text

COLORS = ["W", "U", "B", "R", "G"]
TYPES = ["Creature", "Instant", "Sorcery", "Enchantment", "Artifact"]
RARITIES = {"common": 0, "uncommon": 1, "rare": 2, "mythic": 3}


def _parse_pt(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def card_to_features(card: dict) -> list:
    colors = card.get("colors", [])
    color_features = [1.0 if c in colors else 0.0 for c in COLORS]

    type_line = card.get("type_line", "")
    type_features = [1.0 if t in type_line else 0.0 for t in TYPES]

    rarity = RARITIES.get(card.get("rarity", "common"), 0)

    structured = [
        float(card.get("cmc", 0.0)),
        *color_features,
        *type_features,
        float(rarity),
        _parse_pt(card.get("power")),
        _parse_pt(card.get("toughness")),
    ]

    text_embedding = embed_text(card.get("oracle_text", ""))

    return [*structured, *text_embedding]


STRUCTURED_DIM = 1 + len(COLORS) + len(TYPES) + 1 + 2
FEATURE_DIM = STRUCTURED_DIM + EMBEDDING_DIM
