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
from mtg_rating.set_context import CONTEXT_DIM, compute_set_context_vectors
from mtg_rating.text_embeddings import MODEL_NAME
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


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


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


def _set_context_tensor(batch_records: list):
    # Records only carry "set_context" when main() was called with
    # use_set_context=True -- None here means the model wasn't built with a
    # set_context_dim either, so forward() just ignores it.
    if "set_context" not in batch_records[0]:
        return None
    return torch.tensor([r["set_context"] for r in batch_records], dtype=torch.float32, device=DEVICE)


def evaluate(model: CardRatingNetJoint, records: list, targets: list):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in iter_batches(list(zip(records, targets)), BATCH_SIZE):
            batch_records = [r for r, _ in batch]
            structured = torch.tensor([r["structured"] for r in batch_records], dtype=torch.float32, device=DEVICE)
            set_context = _set_context_tensor(batch_records)
            preds.extend(model(structured, [r["oracle_text"] for r in batch_records], set_context).tolist())

    if isinstance(targets[0], tuple):
        # Dual-target mode (see `second_formula`): preds/targets are lists of
        # same-length tuples. `mse` (first return value) is the combined,
        # equally-weighted MSE across both components -- used for early
        # stopping/model selection exactly like the single-target case, so
        # that codepath doesn't need to know which mode it's in. `extra` is a
        # per-component (mse, pearson_r) breakdown instead of a single corr.
        n_targets = len(targets)
        n_outputs = len(targets[0])
        mse = sum(sum((p_i - t_i) ** 2 for p_i, t_i in zip(p, t)) for p, t in zip(preds, targets)) / (n_targets * n_outputs)
        extra = []
        for k in range(n_outputs):
            preds_k = [p[k] for p in preds]
            targets_k = [t[k] for t in targets]
            mse_k = sum((p - t) ** 2 for p, t in zip(preds_k, targets_k)) / n_targets
            extra.append((mse_k, pearson(preds_k, targets_k)))
        return mse, extra, preds

    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)
    corr = pearson(preds, targets)
    return mse, corr, preds


def save_checkpoint(
    model: CardRatingNetJoint,
    mu: float,
    sigma: float,
    formula: str,
    checkpoint_dir: Path = None,
    base_model_path=MODEL_NAME,
    use_set_context: bool = False,
    extra_formulas: list = None,
    extra_mus: list = None,
    extra_sigmas: list = None,
):
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # LoRA adapter alone (a few 10s of KB) rather than the full 22.7M-param frozen
    # base -- that's the whole point of only training rank-4 adapters.
    model.text_encoder.model.save_pretrained(checkpoint_dir / "lora_adapter")
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "mu": mu,
            "sigma": sigma,
            "formula": formula,
            # Recorded so a later reload knows which base encoder the adapter
            # was trained on top of, and whether the head expects a
            # concatenated set-context vector -- the adapter/head files alone
            # don't carry this, and guessing wrong would silently mismatch
            # dimensions or load the wrong base weights.
            "base_model_path": str(base_model_path),
            "use_set_context": use_set_context,
            # Empty unless training more than one output (see `extra_formulas`
            # on main()) -- each extra target has its own independent mu/sigma,
            # fit separately, in the same order as extra_formulas.
            "extra_formulas": extra_formulas or [],
            "extra_mus": extra_mus or [],
            "extra_sigmas": extra_sigmas or [],
        },
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
    lora_lr: float = LORA_LR,
    head_lr: float = HEAD_LR,
    formula: str = FORMULA,
    base_model_path=MODEL_NAME,
    use_set_context: bool = False,
    hidden_dims: list = None,
    extra_formulas: list = None,
    raw_formulas: list = None,
    checkpoint_dir: Path = None,
):
    extra_formulas = extra_formulas or []
    # Formulas in raw_formulas (must be a subset of extra_formulas) skip the
    # 0-10 rating normalization entirely and train on the raw score as-is --
    # e.g. play_rate is already a meaningful, interpretable 0-1 fraction, and
    # forcing it through the same z-score-to-rating mapping as IIH/GP would
    # make it just as opaque as them for no benefit. Note this means its
    # squared-error contribution to the shared loss is on a very different
    # scale (0-1 vs 0-10) than the normalized outputs -- effectively a much
    # smaller implicit weight in the unweighted mean the loss uses, not
    # rebalanced here.
    raw_formulas = set(raw_formulas or [])
    # `seed` controls only model init (LoRA adapter matrices, head) and epoch
    # shuffling -- the train/test partition always uses the fixed SPLIT_SEED, so
    # runs with different `seed` measure pure optimization variance on the same
    # split, not split variance.
    torch.manual_seed(seed)

    records = build_dataset(set_codes)
    for r in records:
        r["structured"] = structured_features(r["scryfall_card"])
        r["oracle_text"] = r["scryfall_card"].get("oracle_text", "")

    if use_set_context:
        # Fixed, non-trainable per-set mean (structured features + frozen MiniLM
        # embedding) -- see set_context.py for why this and not live/trainable
        # cross-card attention. No label information involved, so including
        # test-split cards' text in their set's own average isn't leakage.
        set_context_vectors = compute_set_context_vectors(records)
        for r in records:
            r["set_context"] = set_context_vectors[r["set_code"]]

    # build_dataset() already guarantees iih is present on every record, but
    # extra formulas (e.g. play_rate_only -- None when a card was never in any
    # final deck/sideboard build) aren't guaranteed -- drop any record missing a
    # value for a formula this run actually needs, before splitting.
    all_formulas = [formula] + extra_formulas
    records = [r for r in records if all(RAW_SCORE_FORMULAS[f](r) is not None for f in all_formulas)]

    train_names, val_names, test_names = split_by_name_3way(records, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)
    train_records = [r for r in records if r["name"] in train_names]
    val_records = [r for r in records if r["name"] in val_names]
    test_records = [r for r in records if r["name"] in test_names]
    print(f"[split] {len(train_records)} train rows, {len(val_records)} val rows, {len(test_records)} test rows")

    fn = RAW_SCORE_FORMULAS[formula]
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

    # Independent mu/sigma per extra target -- each raw score (IIH, GP WR, PR...)
    # lives on its own scale, each fit on train only, same as the primary. Once
    # there's more than one target, train/val/test_targets become tuples
    # (primary, extra_1, extra_2, ...) instead of bare floats -- evaluate() and
    # the training loop below are written generically over however many there are.
    extra_mus, extra_sigmas = [], []
    for extra_formula in extra_formulas:
        fn_extra = RAW_SCORE_FORMULAS[extra_formula]
        train_raw_extra = {i: fn_extra(r) for i, r in enumerate(train_records)}
        val_raw_extra = {i: fn_extra(r) for i, r in enumerate(val_records)}
        test_raw_extra = {i: fn_extra(r) for i, r in enumerate(test_records)}
        if extra_formula in raw_formulas:
            mu_extra = sigma_extra = None
            train_ratings_extra, val_ratings_extra, test_ratings_extra = train_raw_extra, val_raw_extra, test_raw_extra
            extra_mus.append(mu_extra)
            extra_sigmas.append(sigma_extra)
            train_targets = [(*_as_tuple(train_targets[i]), train_ratings_extra[i]) for i in range(len(train_records))]
            val_targets = [(*_as_tuple(val_targets[i]), val_ratings_extra[i]) for i in range(len(val_records))]
            test_targets = [(*_as_tuple(test_targets[i]), test_ratings_extra[i]) for i in range(len(test_records))]
            continue
        mu_extra, sigma_extra = fit_normalization(train_raw_extra)
        extra_mus.append(mu_extra)
        extra_sigmas.append(sigma_extra)
        train_ratings_extra = apply_normalization(train_raw_extra, mu_extra, sigma_extra)
        val_ratings_extra = apply_normalization(val_raw_extra, mu_extra, sigma_extra)
        test_ratings_extra = apply_normalization(test_raw_extra, mu_extra, sigma_extra)
        train_targets = [(*_as_tuple(train_targets[i]), train_ratings_extra[i]) for i in range(len(train_records))]
        val_targets = [(*_as_tuple(val_targets[i]), val_ratings_extra[i]) for i in range(len(val_records))]
        test_targets = [(*_as_tuple(test_targets[i]), test_ratings_extra[i]) for i in range(len(test_records))]

    print(f"[device] {DEVICE}")
    model = CardRatingNetJoint(
        structured_dim=len(train_records[0]["structured"]),
        hidden_dims=hidden_dims,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        head_dropout=head_dropout,
        base_model_path=base_model_path,
        set_context_dim=CONTEXT_DIM if use_set_context else 0,
        output_dim=len(all_formulas),
    ).to(DEVICE)
    model.text_encoder.print_trainable_parameters()
    optimizer = torch.optim.Adam(
        [
            {"params": model.text_encoder.trainable_parameters(), "lr": lora_lr},
            {"params": model.head.parameters(), "lr": head_lr},
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
            set_context = _set_context_tensor(batch_records)

            optimizer.zero_grad()
            pred = model(structured, [r["oracle_text"] for r in batch_records], set_context)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        train_mse = total_loss / len(train_pairs)

        # Watched every epoch on val (never test) so the run can early-stop at the
        # checkpoint that actually generalizes best, instead of whatever epoch
        # training happened to stop at -- the 40-epoch/3-seed sweep showed the
        # final epoch can be meaningfully worse than the best one reached mid-run.
        val_mse, val_extra, _ = evaluate(model, val_records, val_targets)
        if extra_formulas:
            detail = " | ".join(f"{name}: MSE={m:.3f} r={r:.3f}" for name, (m, r) in zip(all_formulas, val_extra))
            print(f"[epoch {epoch}] train MSE={train_mse:.3f}  val MSE={val_mse:.3f} ({detail})")
        else:
            print(f"[epoch {epoch}] train MSE={train_mse:.3f}  val MSE={val_mse:.3f} Pearson r={val_extra:.3f}")

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
    test_mse, test_extra, preds = evaluate(model, test_records, test_targets)
    if extra_formulas:
        detail = " | ".join(f"{name}: MSE={m:.3f} r={r:.3f}" for name, (m, r) in zip(all_formulas, test_extra))
        print(f"[test] n={len(test_targets)} combined MSE={test_mse:.3f} ({detail})")
    else:
        print(f"[test:{formula}] n={len(test_targets)} MSE={test_mse:.3f} Pearson r={test_extra:.3f}")

    if save:
        save_checkpoint(
            model, mu, sigma, formula, checkpoint_dir=checkpoint_dir,
            base_model_path=base_model_path, use_set_context=use_set_context,
            extra_formulas=extra_formulas, extra_mus=extra_mus, extra_sigmas=extra_sigmas,
        )
    # best_val_mse is returned so a multi-seed sweep can pick which checkpoint to
    # keep by val score, not test score -- selecting on test would reintroduce
    # the exact bias early stopping was written to avoid, just at the seed level
    # instead of the epoch level.
    return model, preds, test_records, test_targets, best_val_mse


if __name__ == "__main__":
    main(SET_CODES)
