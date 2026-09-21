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

from mtg_rating.color_context import primary_color
from mtg_rating.features import structured_features
from mtg_rating.model_joint import CardRatingNetJoint
from mtg_rating.multiset import SET_CODES, build_dataset
from mtg_rating.ratings import RAW_SCORE_FORMULAS, apply_normalization, fit_normalization
from mtg_rating.set_context import CONTEXT_DIM, compute_per_color_context_vectors, compute_set_context_vectors
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


def split_by_set_3way(records: list, val_fraction: float, test_fraction: float, seed: int):
    """Entire sets held out, not just card names within a shared pool -- used
    by train_color_context.py as the honest generalization check for its
    trainable cross-card mechanism (see color_context.py), and here so a
    plain/set_context run can be compared against it on the exact same split
    -- isolates whether a by-set split is just harder for any model, versus a
    specific problem with a trainable cross-card mechanism."""
    set_codes = sorted({r["set_code"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(set_codes)
    n_test = max(1, round(len(set_codes) * test_fraction))
    n_val = max(1, round(len(set_codes) * val_fraction))
    test_sets = set(set_codes[:n_test])
    val_sets = set(set_codes[n_test : n_test + n_val])
    train_sets = set(set_codes[n_test + n_val :])
    return train_sets, val_sets, test_sets


def _set_context_tensor(batch_records: list):
    # Records only carry "set_context" when main() was called with
    # use_set_context=True -- None here means the model wasn't built with a
    # set_context_dim either, so forward() just ignores it.
    if "set_context" not in batch_records[0]:
        return None
    return torch.tensor([r["set_context"] for r in batch_records], dtype=torch.float32, device=DEVICE)


def score_predictions(preds: list, targets: list):
    """Shared by evaluate() below and train_color_context.py's own evaluate()
    -- generic over preds/targets alone, no model/record dependency, so both
    training loops report results the same way."""
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
        return mse, extra

    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)
    corr = pearson(preds, targets)
    return mse, corr


def weighted_mse_loss(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
    threshold_penalty_weight: float = 0.0, threshold: float = 1.0,
    loss_shape: str = "mse", saturating_asymptote: float = 8.0, saturating_scale: float = 3.0,
) -> torch.Tensor:
    """Weighted mean squared error (or a saturating alternative, see
    `loss_shape` below) -- weight is per-card (batch,); broadcasts across
    output components for the dual-target case (batch, output_dim). With
    uniform weight=1 and loss_shape="mse" this is exactly
    nn.MSELoss()(pred, target), so the defaults leave training byte-for-byte
    unchanged from every prior run in this project.

    loss_shape="saturating": replaces the per-example squared error
    (pred-target)^2 with a Geman-McClure-style robust loss,
    `A * e^2 / (e^2 + B)` where e = pred-target -- convex and ~quadratic
    (matches plain MSE) for small errors, but bends over to a concave,
    flattening shape and saturates to a fixed asymptote A for large ones,
    instead of growing without bound. Designed together with the user from
    three explicit constraints: inflection exactly at |e|=1 (design_notes.md's
    "worse than 1 is a fail" threshold, in rating-scale units) -- which fixes
    B = 3*(inflection_x)^2 = 3 algebraically, independent of A -- and a slope
    of 3 *at* that inflection point, which (given B=3) fixes A = 8 (since
    slope_at_1 = 3*A/8). With A=8, B=3: loss value at the threshold is
    exactly 2, asymptote is 8. threshold_penalty_weight's hinge-squared
    add-on (below) is a separate, cruder attempt at the same "don't let
    ratings drift arbitrarily far apart" goal -- the two aren't meant to be
    combined, this saturating shape supersedes it as the more principled
    version.

    threshold_penalty_weight (default 0, disabled): per design_notes.md's own
    note ("if a rating prediction is worse than 1, it's a fail"), an extra
    hinge-squared term -- max(0, |pred-target| - threshold)^2 -- added on top
    of the base error term. Exactly zero contribution (and zero gradient) for
    any prediction already within `threshold` of its target, so it doesn't
    touch training for the already-fine majority; ramps up specifically for
    the worst misses. `pred`/`target` are already on the 0-10 rating scale at
    the call site, so threshold=1.0 matches "worse than 1" literally."""
    if pred.dim() > 1:
        weight = weight.unsqueeze(-1).expand_as(pred)
    sq_err = (pred - target) ** 2
    if loss_shape == "saturating":
        base_err = saturating_asymptote * sq_err / (sq_err + saturating_scale)
    else:
        base_err = sq_err
    mse = (base_err * weight).sum() / weight.sum()
    if not threshold_penalty_weight:
        return mse
    overage = (pred - target).abs() - threshold
    hinge = overage.clamp(min=0) ** 2
    penalty = (hinge * weight).sum() / weight.sum()
    return mse + threshold_penalty_weight * penalty


def contrastive_loss(text_embedding: torch.Tensor, target_component: torch.Tensor, tau: float) -> torch.Tensor:
    """Auxiliary embedding-space loss: pulls two cards' raw text embeddings
    together in proportion to how close their actual target values are, and
    apart otherwise -- regardless of surface text similarity. Motivated
    directly by the Annul embedding probe (see project memory): the card
    most similar to Annul in embedding space (a flexible, unrestricted
    counterspell) had one of the *largest* real GP WR gaps, i.e. the encoder
    currently organizes by surface "counter"/"target" vocabulary, not by the
    functional restrictiveness that actually drives outcomes. Rather than
    hand-picking which textual cues signal restrictiveness (tried, rejected
    as too approximate), this lets gradient descent find whatever cue
    produces the right embedding geometry on its own.

    target_sim = exp(-|target_i - target_j| / tau) -- close targets get a
    target_sim near 1 (pull together), far-apart targets get a target_sim
    near 0 (push apart), smoothly rather than a hard positive/negative
    split. Computed pairwise within the batch (batch, batch), matching the
    scale to actual cosine similarity of the (already L2-normalized)
    embeddings. tau=1.0 uses the same rating-scale unit as
    `threshold_penalty_weight`'s threshold elsewhere in this file."""
    normed = torch.nn.functional.normalize(text_embedding, dim=-1)
    sim = normed @ normed.T
    gap = (target_component.unsqueeze(0) - target_component.unsqueeze(1)).abs()
    target_sim = torch.exp(-gap / tau)
    mask = ~torch.eye(len(target_component), dtype=torch.bool, device=target_component.device)
    return ((sim - target_sim) ** 2)[mask].mean()


@torch.no_grad()
def mine_hard_triplets(model: CardRatingNetJoint, records: list, targets: list, threshold: float) -> list:
    """Find, for every training card, the single hardest negative for a
    triplet loss (see `triplet_loss` below) -- the card whose *text embedding*
    is most similar under the model's current encoder, among cards whose
    *real target value* is more than `threshold` apart (same rating-scale
    unit as `threshold_penalty_weight`/`contrastive_tau` elsewhere in this
    file). This is exactly the failure pattern the Annul embedding probe
    found (see `contrastive_loss`'s docstring): surface-similar, outcome-very-
    different pairs. The positive is just the closest *real target value* in
    the pool -- no embedding search needed for that half, since target values
    are already known upfront.

    A full forward pass over every training record (not just one batch), so
    this is meant to be called periodically (e.g. once per epoch, from
    `main`'s `triplet_mining_every`) rather than every step -- LoRA's
    differential, deliberately small LR means the embedding space doesn't
    shift much within an epoch, so triplets mined at its start stay
    informative through the epoch's batches, and re-embedding the whole pool
    on every mini-batch step would cost far more than training itself.

    Returns a list of (anchor_idx, positive_idx, negative_idx) into
    `records`/`targets` -- indices are only valid against that exact list
    (train_records itself is never reordered, only the record/target zip
    used for batching is, so these stay valid for the whole epoch).
    """
    was_training = model.training
    model.eval()
    embeddings = []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        embeddings.append(model.text_encoder([r["oracle_text"] for r in batch]))
    embeddings = torch.nn.functional.normalize(torch.cat(embeddings, dim=0), dim=-1)
    if was_training:
        model.train()

    target_values = torch.tensor(
        [_as_tuple(t)[-1] for t in targets], dtype=torch.float32, device=embeddings.device
    )
    sim = embeddings @ embeddings.T
    gap = (target_values.unsqueeze(0) - target_values.unsqueeze(1)).abs()
    n = len(records)
    eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)

    far_mask = (gap > threshold) & ~eye
    has_negative = far_mask.any(dim=1)
    negative_idx = sim.masked_fill(~far_mask, float("-inf")).argmax(dim=1)

    pos_gap = gap.masked_fill(eye, float("inf"))
    positive_idx = pos_gap.argmin(dim=1)

    anchors = torch.arange(n, device=embeddings.device)[has_negative]
    return list(zip(anchors.tolist(), positive_idx[has_negative].tolist(), negative_idx[has_negative].tolist()))


def triplet_loss(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
    """Standard margin triplet loss (cosine-distance version, matching the
    cosine similarity `mine_hard_triplets` used to pick `negative` in the
    first place): pushes the anchor closer to `positive` (closest real
    target value) than to `negative` (closest embedding, but a real target
    more than `threshold` away) by at least `margin`, zero loss/gradient
    once that's already true."""
    a = torch.nn.functional.normalize(anchor, dim=-1)
    p = torch.nn.functional.normalize(positive, dim=-1)
    n = torch.nn.functional.normalize(negative, dim=-1)
    d_ap = 1 - (a * p).sum(dim=-1)
    d_an = 1 - (a * n).sum(dim=-1)
    return (d_ap - d_an + margin).clamp(min=0).mean()


def evaluate(model: CardRatingNetJoint, records: list, targets: list):
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in iter_batches(list(zip(records, targets)), BATCH_SIZE):
            batch_records = [r for r, _ in batch]
            structured = torch.tensor([r["structured"] for r in batch_records], dtype=torch.float32, device=DEVICE)
            set_context = _set_context_tensor(batch_records)
            preds.extend(model(structured, [r["oracle_text"] for r in batch_records], set_context).tolist())

    mse, extra = score_predictions(preds, targets)
    return mse, extra, preds


def save_checkpoint(
    model: CardRatingNetJoint,
    mu: float,
    sigma: float,
    formula: str,
    checkpoint_dir: Path = None,
    base_model_path=MODEL_NAME,
    use_set_context: bool = False,
    use_color_context: bool = False,
    extra_formulas: list = None,
    extra_mus: list = None,
    extra_sigmas: list = None,
    set_codes: list = None,
    split_mode: str = None,
    split_seed: int = None,
    val_fraction: float = None,
    test_fraction: float = None,
):
    checkpoint_dir = Path(checkpoint_dir or CHECKPOINT_DIR)  # accept str or Path
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
            "use_color_context": use_color_context,
            # Empty unless training more than one output (see `extra_formulas`
            # on main()) -- each extra target has its own independent mu/sigma,
            # fit separately, in the same order as extra_formulas.
            "extra_formulas": extra_formulas or [],
            "extra_mus": extra_mus or [],
            "extra_sigmas": extra_sigmas or [],
            # The pool this checkpoint was trained on and how it was split. The
            # by-name split shuffles the *whole* list of card names, so adding a
            # set to SET_CODES re-deals every card: without this record, a later
            # diagnostic that rebuilds the dataset from the current SET_CODES sees
            # val/test cards this checkpoint trained on (see resolve_training_sets).
            # None = not recorded.
            "set_codes": set_codes,
            "split_mode": split_mode,
            "split_seed": split_seed,
            "val_fraction": val_fraction,
            "test_fraction": test_fraction,
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
    use_color_context: bool = False,
    hidden_dims: list = None,
    layer_norm: bool = False,
    extra_formulas: list = None,
    raw_formulas: list = None,
    checkpoint_dir: Path = None,
    split_mode: str = "name",  # "name" (default, every prior result in this project) or "set" (honesty check)
    sample_weight_power: float = 0.0,  # 0 = uniform (default, unchanged behavior); >0 upweights high-game-count cards
    threshold_penalty_weight: float = 0.0,  # 0 = disabled (default); >0 adds a hinge-squared penalty beyond `threshold`
    threshold: float = 1.0,  # rating-scale units (0-10); design_notes.md's "worse than 1 is a fail"
    loss_shape: str = "mse",  # "mse" (default, unchanged) or "saturating" (Geman-McClure style, see weighted_mse_loss)
    saturating_asymptote: float = 8.0,
    saturating_scale: float = 3.0,
    contrastive_weight: float = 0.0,  # 0 = disabled (default); >0 adds the embedding-space auxiliary loss
    contrastive_tau: float = 1.0,
    triplet_weight: float = 0.0,  # 0 = disabled (default); >0 adds the hard-mined triplet loss (see mine_hard_triplets)
    triplet_margin: float = 0.5,
    triplet_target_threshold: float = 1.0,  # rating-scale units; same "worse than 1 is a fail" threshold as above
    triplet_mining_every: int = 1,  # epochs between re-mining; 1 = every epoch
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
    assert not (use_set_context and use_color_context), "use_set_context and use_color_context are alternatives, not both"
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
    elif use_color_context:
        # Same fixed-mean idea, but grouped by (set, primary color) instead of
        # by whole set -- a more targeted "average card of this color in this
        # set" summary. Still zero trainable parameters (just a different
        # grouping for the same averaging operation), so it doesn't inherit
        # the by-set generalization problem the *trainable* attention version
        # (color_context.py's ColorAttentionContext, train_color_context.py)
        # was shown to have -- there's nothing here for a by-set split to
        # catch. Reuses the model's existing "set_context" input slot (same
        # dimensionality, CONTEXT_DIM either way), so no model_joint.py change
        # is needed -- only which vector gets computed and stored here.
        per_color_vectors = compute_per_color_context_vectors(records)
        for r in records:
            r["set_context"] = per_color_vectors[r["set_code"]][primary_color(r["scryfall_card"])]

    # build_dataset() already guarantees iih is present on every record, but
    # extra formulas (e.g. play_rate_only -- None when a card was never in any
    # final deck/sideboard build) aren't guaranteed -- drop any record missing a
    # value for a formula this run actually needs, before splitting.
    all_formulas = [formula] + extra_formulas
    records = [r for r in records if all(RAW_SCORE_FORMULAS[f](r) is not None for f in all_formulas)]
    trained_set_codes = sorted({r["set_code"] for r in records})  # recorded in the checkpoint, see save_checkpoint

    if split_mode == "set":
        train_sel, val_sel, test_sel = split_by_set_3way(records, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)
        train_records = [r for r in records if r["set_code"] in train_sel]
        val_records = [r for r in records if r["set_code"] in val_sel]
        test_records = [r for r in records if r["set_code"] in test_sel]
    else:
        train_sel, val_sel, test_sel = split_by_name_3way(records, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)
        train_records = [r for r in records if r["name"] in train_sel]
        val_records = [r for r in records if r["name"] in val_sel]
        test_records = [r for r in records if r["name"] in test_sel]
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
        layer_norm=layer_norm,
        lora_rank=lora_rank,
        lora_dropout=lora_dropout,
        head_dropout=head_dropout,
        base_model_path=base_model_path,
        set_context_dim=CONTEXT_DIM if (use_set_context or use_color_context) else 0,
        output_dim=len(all_formulas),
    ).to(DEVICE)
    model.text_encoder.print_trainable_parameters()
    optimizer = torch.optim.Adam(
        [
            {"params": model.text_encoder.trainable_parameters(), "lr": lora_lr},
            {"params": model.head.parameters(), "lr": head_lr},
        ]
    )
    # Sample-count-aware weighting: GP WR (and IIH) are themselves empirical win
    # rates, noisier for low-game-count cards -- a plain, unweighted MSE asks the
    # model to fit a 300-game card's label just as hard as a 100k-game card's,
    # even though the former carries much more sampling noise. gih_count +
    # gns_count is the true denominator behind gp_wr (see labels.py). Power=0
    # (default) keeps every weight at 1.0, identical to the unweighted loss used
    # everywhere else in this project.
    for r in train_records:
        n = (r.get("gih_count") or 0) + (r.get("gns_count") or 0)
        r["_loss_weight"] = max(n, 1) ** sample_weight_power if sample_weight_power else 1.0

    train_pairs = list(zip(train_records, train_targets))
    rng = random.Random(seed)

    best_val_mse = float("inf")
    best_state = None
    patience_counter = 0
    triplet_pairs = []  # (anchor_idx, positive_idx, negative_idx) into train_records/train_targets

    for epoch in range(epochs):
        if triplet_weight and epoch % triplet_mining_every == 0:
            triplet_pairs = mine_hard_triplets(model, train_records, train_targets, triplet_target_threshold)
            print(f"[triplet mining] epoch {epoch}: {len(triplet_pairs)} anchors with a hard negative")
        rng.shuffle(train_pairs)
        model.train()
        total_loss = 0.0
        for batch in iter_batches(train_pairs, BATCH_SIZE):
            batch_records = [r for r, _ in batch]
            batch_targets = [t for _, t in batch]
            structured = torch.tensor([r["structured"] for r in batch_records], dtype=torch.float32, device=DEVICE)
            target = torch.tensor(batch_targets, dtype=torch.float32, device=DEVICE)
            weight = torch.tensor([r["_loss_weight"] for r in batch_records], dtype=torch.float32, device=DEVICE)
            set_context = _set_context_tensor(batch_records)

            optimizer.zero_grad()
            pred = model(structured, [r["oracle_text"] for r in batch_records], set_context)
            loss = weighted_mse_loss(
                pred, target, weight, threshold_penalty_weight, threshold,
                loss_shape, saturating_asymptote, saturating_scale,
            )
            if contrastive_weight:
                # Separate forward through the same shared encoder -- duplicates
                # the text-encoder compute (model() already runs it internally)
                # rather than threading an extra return value through
                # CardRatingNetJoint.forward(), which train_color_context.py also
                # depends on -- acceptable overhead for a first test, same
                # tradeoff train_color_context.py already made.
                text_embedding = model.text_encoder([r["oracle_text"] for r in batch_records])
                target_component = target[:, -1] if target.dim() > 1 else target
                loss = loss + contrastive_weight * contrastive_loss(text_embedding, target_component, contrastive_tau)
            if triplet_weight and triplet_pairs:
                # One triplet per card in this batch (or fewer if there aren't
                # enough mined triplets yet) -- same batch-size-matching
                # convention as the contrastive_weight branch above. One
                # concatenated forward for anchor+positive+negative texts
                # instead of three, to only pay the tokenizer/encoder
                # overhead once per batch.
                sample = rng.sample(triplet_pairs, min(len(batch), len(triplet_pairs)))
                a_idx, p_idx, n_idx = zip(*sample)
                triplet_texts = (
                    [train_records[i]["oracle_text"] for i in a_idx]
                    + [train_records[i]["oracle_text"] for i in p_idx]
                    + [train_records[i]["oracle_text"] for i in n_idx]
                )
                triplet_embeds = model.text_encoder(triplet_texts)
                k = len(a_idx)
                loss = loss + triplet_weight * triplet_loss(
                    triplet_embeds[:k], triplet_embeds[k : 2 * k], triplet_embeds[2 * k :], triplet_margin
                )
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
            base_model_path=base_model_path, use_set_context=use_set_context, use_color_context=use_color_context,
            extra_formulas=extra_formulas, extra_mus=extra_mus, extra_sigmas=extra_sigmas,
            set_codes=trained_set_codes, split_mode=split_mode,
            split_seed=SPLIT_SEED, val_fraction=VAL_FRACTION, test_fraction=TEST_FRACTION,
        )
    # best_val_mse is returned so a multi-seed sweep can pick which checkpoint to
    # keep by val score, not test score -- selecting on test would reintroduce
    # the exact bias early stopping was written to avoid, just at the seed level
    # instead of the epoch level.
    return model, preds, test_records, test_targets, best_val_mse


def load_training_pool(checkpoint_dir) -> dict:
    """What `save_checkpoint` recorded about the pool a checkpoint was trained on
    ("set_codes", "split_mode", "split_seed", "val_fraction", "test_fraction").
    Empty for checkpoints saved before this was tracked (e.g. the original
    lora_joint_dual)."""
    ckpt = torch.load(Path(checkpoint_dir) / "head.pt", map_location="cpu", weights_only=False)
    keys = ("set_codes", "split_mode", "split_seed", "val_fraction", "test_fraction")
    return {k: ckpt[k] for k in keys if ckpt.get(k) is not None}


def resolve_training_sets(explicit, checkpoint_dir):
    """The set list a diagnostic should rebuild the dataset from, so that its
    by-name split is the checkpoint's own (see save_checkpoint for why this
    matters). Priority: `explicit` > the sets recorded in the checkpoint > None,
    i.e. the current SET_CODES, with a loud warning: for a checkpoint that
    predates the record, or was trained before the pool last changed, that split
    is NOT the checkpoint's own and val/test cards are partly in-sample."""
    name = Path(checkpoint_dir).name
    pool = load_training_pool(checkpoint_dir)
    current = {"split_seed": SPLIT_SEED, "val_fraction": VAL_FRACTION, "test_fraction": TEST_FRACTION}
    changed = {k: (pool[k], v) for k, v in current.items() if k in pool and pool[k] != v}
    if changed:
        print(f"[pool] WARNING: {name} was split with {{{', '.join(f'{k}={a}' for k, (a, _) in changed.items())}}} but the "
              f"current constants are {{{', '.join(f'{k}={b}' for k, (_, b) in changed.items())}}}: the rebuilt split will not match.")
    if pool.get("split_mode", "name") != "name":
        print(f"[pool] WARNING: {name} was trained with split_mode={pool['split_mode']!r}; diagnostics split by name, so the rebuilt split will not match.")
    if explicit is not None:
        return list(explicit)
    recorded = pool.get("set_codes")
    if recorded:
        if sorted(recorded) != sorted(SET_CODES):
            print(f"[pool] {name} was trained on {len(recorded)} sets but SET_CODES now has {len(SET_CODES)}: "
                  f"rebuilding from the checkpoint's own sets so the split matches.")
        return list(recorded)
    print(f"[pool] WARNING: {name} doesn't record its training sets (saved before that was tracked). Using the current "
          f"SET_CODES ({len(SET_CODES)} sets): if the pool changed since it was trained, this is NOT its own split and "
          f"val/test cards are partly in-sample. Pass set_codes=[...] explicitly to fix.")
    return None


if __name__ == "__main__":
    main(SET_CODES)
