"""Real multi-set training + held-out evaluation.

Pools all 17Lands-tracked sets in multiset.SET_CODES, splits by unique card name
(not by row -- a card reprinted across sets must land entirely on one side, so the
test set never contains a feature vector the model has already seen at train time),
and reports the first genuine held-out generalization measurement in this project.

Run with: conda run -n env_coinche python -m mtg_rating.train_multiset
"""

import csv
import random
from pathlib import Path

import numpy as np
import torch

from mtg_rating.features import card_to_features
from mtg_rating.fetch_scryfall import fetch_card
from mtg_rating.model import train_toy
from mtg_rating.multiset import METRIC_FIELDS, build_dataset
from mtg_rating.ratings import RAW_SCORE_FORMULAS, apply_normalization, fit_normalization

POOL_SUMMARY_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "multiset_pool.csv"
DATASET_CACHE_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "multiset_dataset.npz"
CACHE_FIELDS = ["gih_wr", "gns_wr", "iih", "gp_wr", "play_rate"]

TEST_FRACTION = 0.2
SPLIT_SEED = 42
EPOCHS = 200

# Old, evergreen cards guaranteed to exist on Scryfall and outside every set in
# multiset.SET_CODES -- used only to sanity-check the model on genuinely novel input.
UNSEEN_TEST_CARDS = ["Lightning Bolt", "Llanowar Elves", "Counterspell"]


def split_by_name(records: list, test_fraction: float, seed: int):
    names = sorted({r["name"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = int(len(names) * test_fraction)
    return set(names[n_test:]), set(names[:n_test])


def pearson(a, b) -> float:
    return float(np.corrcoef(np.asarray(a), np.asarray(b))[0, 1])


def save_pool_summary(records: list, path: Path):
    """Dump the pooled dataset (minus the bulky Scryfall card blobs) to a small CSV,
    so it's inspectable on disk right after build_dataset() -- independent of
    whatever stdout buffering is or isn't visible for the rest of the run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "set_code"] + METRIC_FIELDS)
        writer.writeheader()
        for r in records:
            writer.writerow({"name": r["name"], "set_code": r["set_code"], **{k: r.get(k) for k in METRIC_FIELDS}})

    per_set = {}
    for r in records:
        per_set[r["set_code"]] = per_set.get(r["set_code"], 0) + 1
    print(f"[dataset] {len(records)} records total, saved to {path}")
    for set_code, count in sorted(per_set.items()):
        print(f"[dataset]   {set_code}: {count}")


def load_or_build_dataset() -> list:
    """Like build_dataset(), but also computes (and caches) each record's feature
    vector -- the MiniLM embedding pass over thousands of cards is the most
    expensive step once the per-set metrics/Scryfall caches already exist, and
    unlike those, it wasn't being cached at all before: every run recomputed every
    embedding from scratch. Cached as a single .npz (features are dense float
    arrays; CSV would work but be far bigger and slower to parse back)."""
    if DATASET_CACHE_PATH.exists():
        data = np.load(DATASET_CACHE_PATH, allow_pickle=False)
        records = [
            {
                "name": str(data["names"][i]),
                "set_code": str(data["set_codes"][i]),
                "features": data["features"][i].tolist(),
                **{field: float(data[field][i]) for field in CACHE_FIELDS},
            }
            for i in range(len(data["names"]))
        ]
        print(f"[cache] loaded {len(records)} records (with features) from {DATASET_CACHE_PATH}")
        return records

    records = build_dataset()
    save_pool_summary(records, POOL_SUMMARY_PATH)

    print("[features] computing feature vectors...")
    for r in records:
        r["features"] = card_to_features(r["scryfall_card"])

    DATASET_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        DATASET_CACHE_PATH,
        names=np.array([r["name"] for r in records]),
        set_codes=np.array([r["set_code"] for r in records]),
        features=np.array([r["features"] for r in records], dtype=np.float32),
        **{field: np.array([r[field] for r in records], dtype=np.float64) for field in CACHE_FIELDS},
    )
    print(f"[cache] saved {len(records)} records (with features) to {DATASET_CACHE_PATH}")
    return records


def main():
    records = load_or_build_dataset()

    train_names, test_names = split_by_name(records, TEST_FRACTION, SPLIT_SEED)
    assert train_names.isdisjoint(test_names)
    print(f"[split] {len(train_names)} train names, {len(test_names)} test names")

    train_records = [r for r in records if r["name"] in train_names]
    test_records = [r for r in records if r["name"] in test_names]
    print(f"[split] {len(train_records)} train rows, {len(test_records)} test rows")

    train_features = [r["features"] for r in train_records]
    test_features = [r["features"] for r in test_records]
    unseen_features = {name: card_to_features(fetch_card(name)) for name in UNSEEN_TEST_CARDS}

    for formula, fn in RAW_SCORE_FORMULAS.items():
        train_raw = {i: fn(r) for i, r in enumerate(train_records)}
        test_raw = {i: fn(r) for i, r in enumerate(test_records)}

        mu, sigma = fit_normalization(train_raw)
        train_ratings = apply_normalization(train_raw, mu, sigma)
        test_ratings = apply_normalization(test_raw, mu, sigma)
        train_targets = [train_ratings[i] for i in range(len(train_records))]
        test_targets = [test_ratings[i] for i in range(len(test_records))]

        print(f"\n[train:{formula}] {len(train_records)} rows, mu={mu:.4f} sigma={sigma:.4f}")
        model = train_toy(train_features, train_targets, epochs=EPOCHS)

        with torch.no_grad():
            preds = model(torch.tensor(test_features, dtype=torch.float32)).tolist()
        mse = sum((p - t) ** 2 for p, t in zip(preds, test_targets)) / len(test_targets)
        corr = pearson(preds, test_targets)
        print(f"[test:{formula}] n={len(test_targets)} MSE={mse:.3f} Pearson r={corr:.3f}")

        print(f"[inference:{formula}] unseen evergreen cards:")
        for name, vec in unseen_features.items():
            pred = model(torch.tensor([vec], dtype=torch.float32)).item()
            print(f"  {name!r} -> {pred:.2f} / 10")


if __name__ == "__main__":
    main()
