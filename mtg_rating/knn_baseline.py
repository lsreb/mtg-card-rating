"""Nearest-neighbor ("find the closest comp") baseline for card ratings, as an
alternative to (or ensemble ingredient alongside) the parametric neural net.

Motivation: a human evaluating a brand-new card anchors on the closest known
comparable -- same cost, same power/toughness, same type, similar effect -- and
leans on that comp's known real-world performance. The current pipeline never does
this explicitly; CardRatingNetJoint learns one global parametric function instead.
This script tests whether an explicit k-NN retrieval, using the exact same input
space the trained model already uses (standardized structured features + the
model's own fine-tuned LoRA text embedding), predicts as well as or better than the
neural net's own head -- a cheap way to check the idea before building anything
more elaborate (e.g. blending KNN into the NN, or a real retrieval-augmented head).

Distance space: structured features are z-scored on train stats (so cost/power/
toughness/type/color all contribute on a comparable scale, instead of being
swamped by the 384-dim text embedding), then scaled by `structured_weight`
relative to the L2-normalized LoRA text embedding before concatenation and taking
Euclidean distance. Both `k` and `structured_weight` are swept and picked by val
score (never test), same discipline as every other hyperparameter in this project.

`checkpoint_dir` (default rate_set.CHECKPOINT_DIR) must be a checkpoint trained on the
same split this script rebuilds -- otherwise "val" cards may have been training cards
and the NN reference is contaminated. Even then the NN val number is mildly
optimistic (val picked its best epoch), as is the KNN one (val picked its best k).

Run with: conda run -n env_coinche python -m mtg_rating.knn_baseline
"""

import statistics

import torch

from mtg_rating.features import structured_features
from mtg_rating.multiset import build_dataset
from mtg_rating.rate_set import CHECKPOINT_DIR, load_model
from mtg_rating.ratings import RAW_SCORE_FORMULAS, apply_normalization
from mtg_rating.set_context import compute_set_context_vectors
from mtg_rating.train_lora import SPLIT_SEED, TEST_FRACTION, VAL_FRACTION, iter_batches, resolve_training_sets, split_by_name_3way
from mtg_rating.train_multiset import pearson

BATCH_SIZE = 32
K_VALUES = [3, 5, 10, 20, 50, 100, 200]
STRUCTURED_WEIGHTS = [1.0, 3.0, 10.0]
BLEND_WEIGHTS = [0.0, 0.1, 0.2, 0.3, 0.5]  # weight on the KNN prediction in an NN+KNN blend


@torch.no_grad()
def _text_embeddings(model, records: list) -> torch.Tensor:
    out = []
    for batch in iter_batches(records, BATCH_SIZE):
        emb = model.text_encoder([r["oracle_text"] for r in batch])
        out.append(torch.nn.functional.normalize(emb, dim=-1))
    return torch.cat(out, dim=0)


@torch.no_grad()
def _nn_predictions(model, formulas: list, use_set_context: bool, records: list) -> dict:
    preds = {f: [] for f in formulas}
    for batch in iter_batches(records, BATCH_SIZE):
        structured = torch.tensor([r["structured"] for r in batch], dtype=torch.float32)
        set_context = torch.tensor([r["set_context"] for r in batch], dtype=torch.float32) if use_set_context else None
        out = model(structured, [r["oracle_text"] for r in batch], set_context)
        for i, f in enumerate(formulas):
            preds[f].extend((out[:, i] if out.dim() > 1 else out).tolist())
    return preds


def _actual_ratings(records: list, formula: str, mu: float, sigma: float) -> list:
    return [apply_normalization({0: RAW_SCORE_FORMULAS[formula](r)}, mu, sigma)[0] for r in records]


def _knn_predict(train_structured, train_text, train_targets, query_structured, query_text, k, structured_weight):
    combined_train = torch.cat([train_structured * structured_weight, train_text], dim=1)
    combined_query = torch.cat([query_structured * structured_weight, query_text], dim=1)
    dist = torch.cdist(combined_query, combined_train)  # (n_query, n_train)
    knn_idx = dist.topk(k, largest=False).indices  # (n_query, k)
    knn_targets = train_targets[knn_idx]  # (n_query, k)
    return knn_targets.mean(dim=1)


def main(checkpoint_dir=CHECKPOINT_DIR, set_codes=None):
    model, formulas, use_set_context, norm_params = load_model(checkpoint_dir)
    gp_idx, iih_idx = formulas.index("gp_wr_only"), formulas.index("iih_only")
    gp_mu, gp_sigma = norm_params[gp_idx]
    iih_mu, iih_sigma = norm_params[iih_idx]

    records = build_dataset(resolve_training_sets(set_codes, checkpoint_dir))
    for r in records:
        r["structured"] = structured_features(r["scryfall_card"])
        r["oracle_text"] = r["scryfall_card"].get("oracle_text", "") or ""
    if use_set_context:
        scv = compute_set_context_vectors(records)
        for r in records:
            r["set_context"] = scv[r["set_code"]]
    # Same filter as train_lora.main (every formula the checkpoint outputs), so the by-name split matches.
    records = [r for r in records if all(RAW_SCORE_FORMULAS[f](r) is not None for f in formulas)]

    train_sel, val_sel, _test_sel = split_by_name_3way(records, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)
    train_records = [r for r in records if r["name"] in train_sel]
    val_records = [r for r in records if r["name"] in val_sel]
    print(f"[knn] {len(train_records)} train (neighbor pool), {len(val_records)} val queries")

    # Reference: the trained neural net's own val performance, same records, so the
    # comparison to KNN below is apples-to-apples (test stays untouched throughout).
    nn_preds = _nn_predictions(model, formulas, use_set_context, val_records)
    nn_actual_gp = _actual_ratings(val_records, "gp_wr_only", gp_mu, gp_sigma)
    nn_actual_iih = _actual_ratings(val_records, "iih_only", iih_mu, iih_sigma)
    nn_mse_gp = statistics.fmean((p - a) ** 2 for p, a in zip(nn_preds["gp_wr_only"], nn_actual_gp))
    nn_mse_iih = statistics.fmean((p - a) ** 2 for p, a in zip(nn_preds["iih_only"], nn_actual_iih))
    print(
        f"[reference: trained NN on val] "
        f"IIH MSE={nn_mse_iih:.3f} r={pearson(nn_preds['iih_only'], nn_actual_iih):.3f}  |  "
        f"GP MSE={nn_mse_gp:.3f} r={pearson(nn_preds['gp_wr_only'], nn_actual_gp):.3f}"
    )

    train_structured_raw = torch.tensor([r["structured"] for r in train_records], dtype=torch.float32)
    struct_mean, struct_std = train_structured_raw.mean(dim=0), train_structured_raw.std(dim=0).clamp(min=1e-6)
    train_structured = (train_structured_raw - struct_mean) / struct_std
    val_structured = (torch.tensor([r["structured"] for r in val_records], dtype=torch.float32) - struct_mean) / struct_std

    train_text = _text_embeddings(model, train_records)
    val_text = _text_embeddings(model, val_records)

    train_gp = torch.tensor(_actual_ratings(train_records, "gp_wr_only", gp_mu, gp_sigma))
    train_iih = torch.tensor(_actual_ratings(train_records, "iih_only", iih_mu, iih_sigma))

    print(f"\n{'k':>4} {'struct_w':>9}   {'IIH MSE':>8} {'IIH r':>7}   {'GP MSE':>8} {'GP r':>7}")
    best = None
    for sw in STRUCTURED_WEIGHTS:
        for k in K_VALUES:
            pred_iih = _knn_predict(train_structured, train_text, train_iih, val_structured, val_text, k, sw).tolist()
            pred_gp = _knn_predict(train_structured, train_text, train_gp, val_structured, val_text, k, sw).tolist()
            mse_iih = statistics.fmean((p - a) ** 2 for p, a in zip(pred_iih, nn_actual_iih))
            mse_gp = statistics.fmean((p - a) ** 2 for p, a in zip(pred_gp, nn_actual_gp))
            r_iih, r_gp = pearson(pred_iih, nn_actual_iih), pearson(pred_gp, nn_actual_gp)
            print(f"{k:>4} {sw:>9.1f}   {mse_iih:>8.3f} {r_iih:>7.3f}   {mse_gp:>8.3f} {r_gp:>7.3f}")
            combined = mse_iih + mse_gp
            if best is None or combined < best[0]:
                best = (combined, k, sw, mse_iih, r_iih, mse_gp, r_gp)

    print(f"\n[best by val] k={best[1]} structured_weight={best[2]}  IIH MSE={best[3]:.3f} r={best[4]:.3f}  GP MSE={best[5]:.3f} r={best[6]:.3f}")
    print(f"[reference NN] IIH MSE={nn_mse_iih:.3f}  GP MSE={nn_mse_gp:.3f}")

    # Does KNN add anything the NN doesn't already capture? Blend at the val-best KNN config.
    _, best_k, best_sw = best[:3]
    knn_iih = _knn_predict(train_structured, train_text, train_iih, val_structured, val_text, best_k, best_sw).tolist()
    knn_gp = _knn_predict(train_structured, train_text, train_gp, val_structured, val_text, best_k, best_sw).tolist()
    print(f"\n[NN+KNN blend, k={best_k} structured_weight={best_sw}]")
    print(f"{'knn_w':>6}   {'IIH MSE':>8} {'IIH r':>7}   {'GP MSE':>8} {'GP r':>7}")
    for w in BLEND_WEIGHTS:
        blend_iih = [(1 - w) * nn + w * knn for nn, knn in zip(nn_preds["iih_only"], knn_iih)]
        blend_gp = [(1 - w) * nn + w * knn for nn, knn in zip(nn_preds["gp_wr_only"], knn_gp)]
        mse_iih = statistics.fmean((p - a) ** 2 for p, a in zip(blend_iih, nn_actual_iih))
        mse_gp = statistics.fmean((p - a) ** 2 for p, a in zip(blend_gp, nn_actual_gp))
        print(f"{w:>6.1f}   {mse_iih:>8.3f} {pearson(blend_iih, nn_actual_iih):>7.3f}   {mse_gp:>8.3f} {pearson(blend_gp, nn_actual_gp):>7.3f}")


if __name__ == "__main__":
    main()
