"""Build a pooled multi-set training dataset from 17Lands game_data + Scryfall.

Set list verified directly against the public S3 dataset via HEAD requests (not the
17lands.com website API -- see project history for why that distinction matters).
Two Scryfall `set_type` categories are excluded on purpose, as a general rule, not
one-off cases: `alchemy` (digital-only Arena products, no real paper draft
environment) and `draft_innovation` (supplemental products designed as standalone
draft environments outside the normal Standard rotation) -- only `expansion`/`core`
sets remain.

Disk safety: this machine's disk margin is tight, so raw `game_data` files
(~60-90MB each) are never kept around for more than one set at a time -- each is
deleted immediately after its per-card metrics are extracted and cached (a few KB).
"""

import csv
from pathlib import Path

from mtg_rating.fetch_17lands import download_game_data
from mtg_rating.fetch_scryfall import fetch_set_cards
from mtg_rating.labels import compute_card_metrics

SET_CODES = [
    "FDN", "DSK", "BLB", "OTJ", "MKM", "LCI", "WOE", "MOM", "ONE", "BRO", "DMU",
    "SNC", "NEO", "VOW", "MID", "AFR", "STX", "KHM", "EOE", "TLA", "FIN", "TDM",
    "DFT", "SOS", "TMT", "ECL",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
METRIC_FIELDS = ["gih_wr", "gih_count", "gns_wr", "gns_count", "iih", "gp_wr", "play_rate", "pool_count"]


def _metrics_cache_path(set_code: str, event_type: str) -> Path:
    return DATA_DIR / f"metrics_{set_code}_{event_type}.csv"


def _load_cached_metrics(path: Path) -> dict:
    metrics = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            name = row.pop("name")
            metrics[name] = {k: (float(v) if v != "" else None) for k, v in row.items()}
    return metrics


def _round_metric(v):
    # Win rates carry sampling noise on the order of sqrt(p(1-p)/n) -- at
    # min_games=200 that's already ~3.5 points of %, and even the best-sampled
    # cards (tens of thousands of games) only get to ~1e-3/1e-4. Anything past
    # the 4th decimal in the raw float repr is binary noise, not signal, so
    # rounding here loses nothing real while cutting cache file size a lot.
    return round(v, 4) if isinstance(v, float) else v


def _save_metrics_cache(path: Path, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name"] + METRIC_FIELDS)
        writer.writeheader()
        for name, m in metrics.items():
            writer.writerow({"name": name, **{k: _round_metric(m.get(k)) for k in METRIC_FIELDS}})


def get_set_metrics(set_code: str, event_type: str = "PremierDraft") -> dict:
    cache_path = _metrics_cache_path(set_code, event_type)
    if cache_path.exists():
        return _load_cached_metrics(cache_path)

    game_data_path = download_game_data(set_code, event_type)
    # include_play_rate=True: measured overhead is modest (+10.2% wall time on
    # MSH's 377k-row game_data, 220.9s -> 243.4s) now that it's actually needed
    # for the "PR" (play_rate) target -- previously left off since nothing used
    # it and it wasn't free.
    metrics = compute_card_metrics(game_data_path, include_play_rate=True)
    _save_metrics_cache(cache_path, metrics)
    game_data_path.unlink()
    return metrics


def build_dataset(set_codes: list = None, event_type: str = "PremierDraft") -> list:
    set_codes = SET_CODES if set_codes is None else set_codes
    records = []
    for set_code in set_codes:
        try:
            print(f"[multiset] {set_code}: fetching metrics...")
            metrics = get_set_metrics(set_code, event_type)
            cards_by_name = {c["name"]: c for c in fetch_set_cards(set_code)}
        except Exception as exc:
            print(f"[multiset] {set_code}: SKIPPED ({exc})")
            continue

        count = 0
        for name, m in metrics.items():
            if m.get("iih") is None or name not in cards_by_name:
                continue
            records.append({
                **m,
                "name": name,
                "set_code": set_code,
                "scryfall_card": cards_by_name[name],
            })
            count += 1
        print(f"[multiset] {set_code}: {count} usable cards")

    print(f"[multiset] total: {len(records)} records across {len(set_codes)} sets")
    return records
