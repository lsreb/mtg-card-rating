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

import copy
import random
from pathlib import Path

import torch
from torch import nn

from mtg_rating.features import structured_features
from mtg_rating.model_joint import CardRatingNetJoint
from mtg_rating.multiset import SET_CODES, build_dataset
from mtg_rating.ratings import RAW_SCORE_FORMULAS, apply_normalization, fit_normalization
from mtg_rating.train_multiset import pearson

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "lora_joint"
# Back to 70/10/20 -- the 80/10/10 experiment came out slightly worse (confounded
# by a smaller, noisier test set) and 70/10/20 is where the rank 2/4/8 comparison
# was established, so it stays the reference split for further hyperparameter
# tests like dropout. Val is used for early stopping and any future
# hyperparameter choice; test is only ever looked at once, at the very end --
# picking a checkpoint by its test-set score would bias that score optimistically
# (the whole point of not reusing it for selection).
TEST_FRACTION = 0.2
VAL_FRACTION = 0.1
SPLIT_SEED = 42
FORMULA = "gih_iih"
BATCH_SIZE = 32
EPOCHS = 40
# Chosen from the 40-epoch/3-seed sweep's per-epoch logs: the best val-equivalent
# epoch (13-20 depending on seed) was never beaten again in the remaining ~20-25
# epochs, and the closest near-miss came 8 epochs after the true best (seed 1:
# best 1.657 @ epoch 15, closest later value 1.668 @ epoch 23) -- 8 is enough
# patience to ride that out without waiting all the way to epoch 40.
PATIENCE = 8
MIN_DELTA = 0.0
# Differential LRs: the LoRA adapters are nudging already-pretrained attention
# weights (small, careful steps), while the head starts from random init and
# needs to learn faster -- one shared 1e-3 LR was likely too aggressive for the
# former (first real run showed train MSE still falling at epoch 5 while test
# MSE had already risen above it, a sign of overfitting/miscalibration rather
# than a clean win against the frozen-embedding baseline).
LORA_LR = 2e-4
HEAD_LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_batches(pairs: list, batch_size: int):
    for start in range(0, len(pairs), batch_size):
        yield pairs[start : start + batch_size]


def split_by_name_3way(records: list, val_fraction: float, test_fraction: float, seed: int):
    names = sorted({r["name"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(names)
    n_test = int(len(names) * test_fraction)
    n_val = int(len(names) * val_fraction)
    test_names = set(names[:n_test])
    val_names = set(names[n_test : n_test + n_val])
    train_names = set(names[n_test + n_val :])
    return train_names, val_names, test_names


def evaluate(model: CardRatingNetJoint, records: list, targets: list):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in iter_batches(list(zip(records, targets)), BATCH_SIZE):
            batch_records = [r for r, _ in batch]
            structured = torch.tensor([r["structured"] for r in batch_records], dtype=torch.float32, device=DEVICE)
            preds.extend(model(structured, [r["oracle_text"] for r in batch_records]).tolist())
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)
    corr = pearson(preds, targets)
    return mse, corr, preds


def save_checkpoint(model: CardRatingNetJoint, mu: float, sigma: float, formula: str, checkpoint_dir: Path = None):
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # LoRA adapter alone (a few 10s of KB) rather than the full 22.7M-param frozen
    # base -- that's the whole point of only training rank-4 adapters.
    model.text_encoder.model.save_pretrained(checkpoint_dir / "lora_adapter")
    torch.save(
        {"head_state_dict": model.head.state_dict(), "mu": mu, "sigma": sigma, "formula": formula},
        checkpoint_dir / "head.pt",
    )
    print(f"[checkpoint] saved to {checkpoint_dir}")


def main(
    set_codes: list = None,
    epochs: int = EPOCHS,
    seed: int = 0,
    save: bool = True,
    lora_rank: int = 4,
    lora_dropout: float = 0.0,
    head_dropout: float = 0.0,
    checkpoint_dir: Path = None,
):
    # `seed` controls only model init (LoRA adapter matrices, head) and epoch
    # shuffling -- the train/test partition always uses the fixed SPLIT_SEED, so
    # runs with different `seed` measure pure optimization variance on the same
    # split, not split variance.
    torch.manual_seed(seed)

    records = build_dataset(set_codes)
    for r in records:
        r["structured"] = structured_features(r["scryfall_card"])
        r["oracle_text"] = r["scryfall_card"].get("oracle_text", "")

    train_names, val_names, test_names = split_by_name_3way(records, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)
    train_records = [r for r in records if r["name"] in train_names]
    val_records = [r for r in records if r["name"] in val_names]
    test_records = [r for r in records if r["name"] in test_names]
    print(f"[split] {len(train_records)} train rows, {len(val_records)} val rows, {len(test_records)} test rows")

    fn = RAW_SCORE_FORMULAS[FORMULA]
    train_raw = {i: fn(r) for i, r in enumerate(train_records)}
    val_raw = {i: fn(r) for i, r in enumerate(val_records)}
    test_raw = {i: fn(r) for i, r in enumerate(test_records)}
    mu, sigma = fit_normalization(train_raw)
    train_ratings = apply_normalization(train_raw, mu, sigma)
    val_ratings = apply_normalization(val_raw, mu, sigma)
    test_ratings = apply_normalization(test_raw, mu, sigma)
    train_targets = [train_ratings[i] for i in range(len(train_records))]
    val_targets = [val_ratings[i] for i in range(len(val_records))]
    test_targets = [test_ratings[i] for i in range(len(test_records))]

    print(f"[device] {DEVICE}")
    model = CardRatingNetJoint(
        structured_dim=len(train_records[0]["structured"]),
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        head_dropout=head_dropout,
    ).to(DEVICE)
    model.text_encoder.print_trainable_parameters()
    optimizer = torch.optim.Adam(
        [
            {"params": model.text_encoder.trainable_parameters(), "lr": LORA_LR},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ]
    )
    loss_fn = nn.MSELoss()

    train_pairs = list(zip(train_records, train_targets))
    rng = random.Random(seed)

    best_val_mse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        rng.shuffle(train_pairs)
        model.train()
        total_loss = 0.0
        for batch in iter_batches(train_pairs, BATCH_SIZE):
            batch_records = [r for r, _ in batch]
            batch_targets = [t for _, t in batch]
            structured = torch.tensor([r["structured"] for r in batch_records], dtype=torch.float32, device=DEVICE)
            target = torch.tensor(batch_targets, dtype=torch.float32, device=DEVICE)

            optimizer.zero_grad()
            pred = model(structured, [r["oracle_text"] for r in batch_records])
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        train_mse = total_loss / len(train_pairs)

        # Watched every epoch on val (never test) so the run can early-stop at the
        # checkpoint that actually generalizes best, instead of whatever epoch
        # training happened to stop at -- the 40-epoch/3-seed sweep showed the
        # final epoch can be meaningfully worse than the best one reached mid-run.
        val_mse, val_corr, _ = evaluate(model, val_records, val_targets)
        print(f"[epoch {epoch}] train MSE={train_mse:.3f}  val MSE={val_mse:.3f} Pearson r={val_corr:.3f}")

        if val_mse < best_val_mse - MIN_DELTA:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[early stop] no val improvement for {PATIENCE} epochs, stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)

    # Test is touched exactly once, here, after every training decision (epoch
    # count included) has already been made from train+val alone -- otherwise
    # the reported number would be optimistically biased by having picked the
    # checkpoint that happens to look best on this specific test set.
    test_mse, test_corr, preds = evaluate(model, test_records, test_targets)
    print(f"[test:{FORMULA}] n={len(test_targets)} MSE={test_mse:.3f} Pearson r={test_corr:.3f}")

    if save:
        save_checkpoint(model, mu, sigma, FORMULA, checkpoint_dir=checkpoint_dir)
    # best_val_mse is returned so a multi-seed sweep can pick which checkpoint to
    # keep by val score, not test score -- selecting on test would reintroduce
    # the exact bias early stopping was written to avoid, just at the seed level
    # instead of the epoch level.
    return model, preds, test_records, test_targets, best_val_mse


if __name__ == "__main__":
    main(SET_CODES)
