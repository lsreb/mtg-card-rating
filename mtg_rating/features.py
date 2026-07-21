"""Turn a Scryfall card object into a fixed-size numeric feature vector.

Deliberately minimal for the v0 toy pipeline: mana value, color one-hot, a coarse
type one-hot, rarity ordinal, power/toughness.
"""

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

    return [
        float(card.get("cmc", 0.0)),
        *color_features,
        *type_features,
        float(rarity),
        _parse_pt(card.get("power")),
        _parse_pt(card.get("toughness")),
    ]


FEATURE_DIM = 1 + len(COLORS) + len(TYPES) + 1 + 2
