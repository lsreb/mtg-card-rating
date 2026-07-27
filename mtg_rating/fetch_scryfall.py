"""Fetch card characteristics from the Scryfall API.

fetch_set_cards downloads (and caches) every card of a given set via the paginated
search endpoint: ~175 cards per page, so a ~300-card set costs about 2 HTTP requests
total, not one request per card. fetch_card looks up a single card by name on demand,
used only for the handful of cards outside the training set (generalization test).

fetch_bulk_oracle_cards pulls Scryfall's "oracle_cards" bulk file -- one object per
unique card (deduplicated by oracle_id across all reprints/printings), unlike
fetch_set_cards' per-set search. Used for the MLM pretraining corpus (see
mlm_pretrain.py): needs the broad historical card pool, not just the 26 sets with
17Lands data. Same disk-safety pattern as the rest of the project -- the raw bulk
file (~200MB) is deleted right after the fields actually needed are extracted, only
the small filtered/trimmed cache is kept.
"""

import json
import time
from pathlib import Path

import requests

USER_AGENT = "mtg-card-rating (research script)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
BULK_CACHE_PATH = DATA_DIR / "scryfall_bulk_oracle_filtered.json"
# Cards must be legal in at least one of these -- excludes Un-sets/joke cards and
# the handful of cards banned everywhere except Vintage/Commander (old power
# cards etc.), while keeping essentially the entire mainstream competitive pool
# (Legacy's banned list is short, so "legal in legacy" alone already covers most
# of Magic's history).
RELEVANT_FORMATS = ("legacy", "modern", "standard")
# Trimmed down from Scryfall's full card object (which also carries image_uris,
# prices, multiverse_ids, related/purchase URIs, etc.) to just what
# features.structured_features/card_to_features actually read -- cuts the cache
# from a few KB/card down to a few hundred bytes/card across ~20-25k cards.
BULK_FIELDS = ["name", "oracle_text", "cmc", "colors", "type_line", "rarity", "power", "toughness"]


def fetch_set_cards(set_code: str) -> list:
    cache_path = DATA_DIR / f"scryfall_set_{set_code.lower()}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())

    cards = []
    url = f"https://api.scryfall.com/cards/search?q=set%3A{set_code.lower()}"
    while url:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        payload = response.json()
        cards.extend(payload["data"])
        url = payload.get("next_page") if payload.get("has_more") else None
        if url:
            time.sleep(0.1)  # Scryfall etiquette: ~10 req/s max

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cards))
    return cards


def _is_relevant(card: dict) -> bool:
    legalities = card.get("legalities", {})
    return any(legalities.get(fmt) == "legal" for fmt in RELEVANT_FORMATS)


def fetch_bulk_oracle_cards() -> list:
    if BULK_CACHE_PATH.exists():
        return json.loads(BULK_CACHE_PATH.read_text())

    bulk_index = requests.get("https://api.scryfall.com/bulk-data", headers=HEADERS)
    bulk_index.raise_for_status()
    entry = next(d for d in bulk_index.json()["data"] if d["type"] == "oracle_cards")
    download_uri = entry["download_uri"]

    raw_path = DATA_DIR / "scryfall_bulk_oracle_raw.json"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with requests.get(download_uri, headers=HEADERS, stream=True) as response:
        response.raise_for_status()
        with open(raw_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    all_cards = json.loads(raw_path.read_text())
    filtered = [
        {field: card.get(field) for field in BULK_FIELDS}
        for card in all_cards
        if _is_relevant(card) and card.get("oracle_text")
    ]
    raw_path.unlink()

    BULK_CACHE_PATH.write_text(json.dumps(filtered))
    print(f"[scryfall] bulk oracle_cards: {len(all_cards)} total -> {len(filtered)} legacy/modern/standard-legal")
    return filtered


def fetch_card(name: str) -> dict:
    response = requests.get(
        "https://api.scryfall.com/cards/named",
        params={"fuzzy": name},
        headers=HEADERS,
    )
    response.raise_for_status()
    return response.json()
