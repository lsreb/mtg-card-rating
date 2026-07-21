"""Fetch card characteristics from the Scryfall API.

fetch_set_cards downloads (and caches) every card of a given set via the paginated
search endpoint: ~175 cards per page, so a ~300-card set costs about 2 HTTP requests
total, not one request per card. fetch_card looks up a single card by name on demand,
used only for the handful of cards outside the training set (generalization test).
"""

import json
import time
from pathlib import Path

import requests

USER_AGENT = "mtg-card-rating (research script)"
HEADERS = {"User-Agent": USER_AGENT, "Accept": "application/json"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


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


def fetch_card(name: str) -> dict:
    response = requests.get(
        "https://api.scryfall.com/cards/named",
        params={"fuzzy": name},
        headers=HEADERS,
    )
    response.raise_for_status()
    return response.json()
