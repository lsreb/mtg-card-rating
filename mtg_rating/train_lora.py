"""LoRA joint fine-tuning: MiniLM (rank-4 LoRA adapters) + rating head, trained
together on pooled multi-set card data.

Unlike train_multiset.py (frozen MiniLM embeddings, precomputed once and cached in
multiset_dataset.npz), this script re-tokenizes and forward/backwards through the
text encoder on every batch, so it needs raw oracle_text straight from the Scryfall
cards -- multiset_dataset.npz can't be reused here, since it only stores the already
-frozen-embedded feature vectors, with no oracle_text left to re-embed. Calling
build_dataset() directly is still cheap: per-set metrics/Scryfall JSON caches are
already on disk (see multiset.py), so this is a re-join in memory, not a re-download.

Only one target formula is trained here (not all three like train_multiset.py) --
gih_iih, the empirically best-generalizing target so far -- since a full transformer
fine-tune is far more CPU-expensive per formula than the frozen-embedding toy MLP.

Run with: conda run -n env_coinche python -m mtg_rating.train_lora
"""

import random
from pathlib import Path

import torch
from torch import nn

from mtg_rating.features import structured_features
from mtg_rating.model_joint import CardRatingNetJoint
from mtg_rating.multiset import SET_CODES, build_dataset
from mtg_rating.ratings import RAW_SCORE_FORMULAS, apply_normalization, fit_normalization
from mtg_rating.train_multiset import pearson, split_by_name

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "lora_joint"
TEST_FRACTION = 0.2
SPLIT_SEED = 42
FORMULA = "gih_iih"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3


def iter_batches(pairs: list, batch_size: int):
    for start in range(0, len(pairs), batch_size):
        yield pairs[start : start + batch_size]


def save_checkpoint(model: CardRatingNetJoint, mu: float, sigma: float, formula: str):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    # LoRA adapter alone (a few 10s of KB) rather than the full 22.7M-param frozen
    # base -- that's the whole point of only training rank-4 adapters.
    model.text_encoder.model.save_pretrained(CHECKPOINT_DIR / "lora_adapter")
    torch.save(
        {"head_state_dict": model.head.state_dict(), "mu": mu, "sigma": sigma, "formula": formula},
        CHECKPOINT_DIR / "head.pt",
    )
    print(f"[checkpoint] saved to {CHECKPOINT_DIR}")


def main(set_codes: list = None, epochs: int = EPOCHS):
    records = build_dataset(set_codes)
    for r in records:
        r["structured"] = structured_features(r["scryfall_card"])
        r["oracle_text"] = r["scryfall_card"].get("oracle_text", "")

    train_names, test_names = split_by_name(records, TEST_FRACTION, SPLIT_SEED)
    train_records = [r for r in records if r["name"] in train_names]
    test_records = [r for r in records if r["name"] in test_names]
    print(f"[split] {len(train_records)} train rows, {len(test_records)} test rows")

    fn = RAW_SCORE_FORMULAS[FORMULA]
    train_raw = {i: fn(r) for i, r in enumerate(train_records)}
    test_raw = {i: fn(r) for i, r in enumerate(test_records)}
    mu, sigma = fit_normalization(train_raw)
    train_ratings = apply_normalization(train_raw, mu, sigma)
    test_ratings = apply_normalization(test_raw, mu, sigma)
    train_targets = [train_ratings[i] for i in range(len(train_records))]
    test_targets = [test_ratings[i] for i in range(len(test_records))]

    model = CardRatingNetJoint(structured_dim=len(train_records[0]["structured"]))
    model.text_encoder.print_trainable_parameters()
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    train_pairs = list(zip(train_records, train_targets))
    rng = random.Random(SPLIT_SEED)

    for epoch in range(epochs):
        rng.shuffle(train_pairs)
        model.train()
        total_loss = 0.0
        for batch in iter_batches(train_pairs, BATCH_SIZE):
            batch_records = [r for r, _ in batch]
            batch_targets = [t for _, t in batch]
            structured = torch.tensor([r["structured"] for r in batch_records], dtype=torch.float32)
            target = torch.tensor(batch_targets, dtype=torch.float32)

            optimizer.zero_grad()
            pred = model(structured, [r["oracle_text"] for r in batch_records])
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f"[epoch {epoch}] train MSE={total_loss / len(train_pairs):.3f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in iter_batches(list(zip(test_records, test_targets)), BATCH_SIZE):
            batch_records = [r for r, _ in batch]
            structured = torch.tensor([r["structured"] for r in batch_records], dtype=torch.float32)
            preds.extend(model(structured, [r["oracle_text"] for r in batch_records]).tolist())
    mse = sum((p - t) ** 2 for p, t in zip(preds, test_targets)) / len(test_targets)
    corr = pearson(preds, test_targets)
    print(f"[test:{FORMULA}] n={len(test_targets)} MSE={mse:.3f} Pearson r={corr:.3f}")

    save_checkpoint(model, mu, sigma, FORMULA)
    return model, preds, test_records, test_targets


if __name__ == "__main__":
    main(SET_CODES)
