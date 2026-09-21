"""Screen oracle-text patterns for correlation with the current best checkpoint's
GP WR prediction residual -- the automatic candidate-search half of the "validated
feature engineering" plan (see CLAUDE.md's under-dispersion discussion). Read-only
diagnostic, no training: loads a trained checkpoint (`checkpoint_dir`, default
rate_set.CHECKPOINT_DIR), rebuilds the dataset from the sets it was trained on
(recorded in the checkpoint; `set_codes` overrides, see
train_lora.resolve_training_sets) so the by-name split is the checkpoint's own,
and computes
residual = actual_gp_wr_rating - predicted_gp_wr_rating (both on the model's own
0-10 scale, via the checkpoint's own mu/sigma) for every train/val card, then
correlates every single-word/bigram pattern that appears often enough in
oracle_text against that residual -- scored on train, confirmed on val (never
test), so a pattern has to show a same-sign, non-trivial correlation on data it
wasn't scored on before being reported as a real candidate. This replaces guessing
3 restriction-flavored features by hand (tried once, explicitly rejected as too
approximate -- see project history) with a systematic scan; a human still has to
sanity-check the survivors before encoding any of them as real structured
features -- this script only ranks candidates, it doesn't add anything to the
model. Caveat: train rows are in-sample for the checkpoint, so their residuals
are deflated relative to genuinely held-out cards; the val column is the clean
one.

Run with: conda run -n env_coinche python -m mtg_rating.feature_search
"""

import re
import statistics
from collections import Counter

import torch

from mtg_rating.features import structured_features
from mtg_rating.multiset import build_dataset
from mtg_rating.rate_set import CHECKPOINT_DIR, load_model
from mtg_rating.ratings import RAW_SCORE_FORMULAS, apply_normalization
from mtg_rating.set_context import compute_set_context_vectors
from mtg_rating.train_lora import SPLIT_SEED, TEST_FRACTION, VAL_FRACTION, iter_batches, resolve_training_sets, split_by_name_3way
from mtg_rating.train_multiset import pearson

TARGET_FORMULA = "gp_wr_only"
BATCH_SIZE = 32
MIN_TRAIN_SUPPORT = 30
MIN_VAL_SUPPORT = 8
TOP_N = 40
CONFIRM_R = 0.05  # same-sign val correlation at least this large counts as "confirmed", not just screened

_PAREN_RE = re.compile(r"\([^)]*\)")  # strip reminder text -- restates the rule in plain words, not the rule itself
_WORD_RE = re.compile(r"[a-z']+")


def _tokenize(oracle_text: str) -> set:
    text = _PAREN_RE.sub(" ", (oracle_text or "").lower())
    words = _WORD_RE.findall(text)
    return set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])}


def _set_context_tensor(batch: list):
    if "set_context" not in batch[0]:
        return None
    return torch.tensor([r["set_context"] for r in batch], dtype=torch.float32)


@torch.no_grad()
def _residuals(model, target_idx: int, mu: float, sigma: float, records: list) -> list:
    preds = []
    for batch in iter_batches(records, BATCH_SIZE):
        structured = torch.tensor([r["structured"] for r in batch], dtype=torch.float32)
        out = model(structured, [r["oracle_text"] for r in batch], _set_context_tensor(batch))
        preds.extend((out[:, target_idx] if out.dim() > 1 else out).tolist())

    residuals = []
    for r, pred in zip(records, preds):
        actual_rating = apply_normalization({0: RAW_SCORE_FORMULAS[TARGET_FORMULA](r)}, mu, sigma)[0]
        residuals.append(actual_rating - pred)
    return residuals


def _score_patterns(train_records, train_residuals, val_records, val_residuals) -> list:
    train_tokens = [_tokenize(r["oracle_text"]) for r in train_records]
    val_tokens = [_tokenize(r["oracle_text"]) for r in val_records]

    train_support = Counter(tok for toks in train_tokens for tok in toks)
    val_support = Counter(tok for toks in val_tokens for tok in toks)
    candidates = [p for p, n in train_support.items() if n >= MIN_TRAIN_SUPPORT and val_support.get(p, 0) >= MIN_VAL_SUPPORT]

    rows = []
    for pattern in candidates:
        train_ind = [1.0 if pattern in toks else 0.0 for toks in train_tokens]
        val_ind = [1.0 if pattern in toks else 0.0 for toks in val_tokens]
        train_r = pearson(train_ind, train_residuals)
        val_r = pearson(val_ind, val_residuals)
        rows.append({
            "pattern": pattern, "train_r": train_r, "val_r": val_r,
            "train_support": train_support[pattern], "val_support": val_support[pattern],
        })
    rows.sort(key=lambda row: abs(row["train_r"]), reverse=True)
    return rows


def main(top_n: int = TOP_N, checkpoint_dir=CHECKPOINT_DIR, set_codes=None):
    model, formulas, use_set_context, norm_params = load_model(checkpoint_dir)
    if TARGET_FORMULA not in formulas:
        raise ValueError(f"checkpoint wasn't trained with {TARGET_FORMULA}: formulas={formulas}")
    target_idx = formulas.index(TARGET_FORMULA)
    mu, sigma = norm_params[target_idx]

    records = build_dataset(resolve_training_sets(set_codes, checkpoint_dir))
    for r in records:
        r["structured"] = structured_features(r["scryfall_card"])
        r["oracle_text"] = r["scryfall_card"].get("oracle_text", "") or ""
    if use_set_context:
        set_context_vectors = compute_set_context_vectors(records)
        for r in records:
            r["set_context"] = set_context_vectors[r["set_code"]]
    # Same filter as train_lora.main (every formula the checkpoint outputs), so the by-name split matches.
    records = [r for r in records if all(RAW_SCORE_FORMULAS[f](r) is not None for f in formulas)]

    train_sel, val_sel, _test_sel = split_by_name_3way(records, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)
    train_records = [r for r in records if r["name"] in train_sel]
    val_records = [r for r in records if r["name"] in val_sel]
    print(f"[feature_search] {len(train_records)} train, {len(val_records)} val cards (test untouched)")

    train_residuals = _residuals(model, target_idx, mu, sigma, train_records)
    val_residuals = _residuals(model, target_idx, mu, sigma, val_records)
    print(
        f"[feature_search] train residual mean={statistics.fmean(train_residuals):+.3f} "
        f"std={statistics.pstdev(train_residuals):.3f} | "
        f"val residual mean={statistics.fmean(val_residuals):+.3f} std={statistics.pstdev(val_residuals):.3f}"
    )

    rows = _score_patterns(train_records, train_residuals, val_records, val_residuals)
    print(f"[feature_search] {len(rows)} candidate patterns (train support>={MIN_TRAIN_SUPPORT}, val support>={MIN_VAL_SUPPORT})")
    print(f"{'pattern':<30} {'train_r':>8} {'val_r':>8} {'n_train':>8} {'n_val':>6}  confirmed")
    for row in rows[:top_n]:
        confirmed = (row["train_r"] * row["val_r"] > 0) and abs(row["val_r"]) >= CONFIRM_R
        print(
            f"{row['pattern']:<30} {row['train_r']:>8.3f} {row['val_r']:>8.3f} "
            f"{row['train_support']:>8} {row['val_support']:>6}  {'yes' if confirmed else ''}"
        )
    return rows


if __name__ == "__main__":
    main()
