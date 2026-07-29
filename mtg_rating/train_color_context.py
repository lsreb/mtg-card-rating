"""Trainable per-color cross-attention context (color_context.py) + rating head,
trained jointly with the rank-4 rating LoRA -- v1 of the "synergy" direction from
premier_jet.md section 6, an alternative to set_context.py's fixed whole-set mean.

Batching differs from train_lora.py's flat shuffled-32-card batches: a bucket (one
set's cards of one primary color, see color_context.py) is the natural unit here,
since the attention mechanism needs every member of a bucket present in the same
forward pass to attend over. Bucket sizes vary (roughly 5-50 cards), so each
optimizer step processes one shuffled bucket rather than a fixed-size batch.

Unlike set_context.py's fixed non-trainable mean, this *is* a trainable
mechanism -- see color_context.py's module docstring for why that changes what a
train/test split needs to prove. Two splits are supported here for exactly that
reason: split_by_name_3way (imported from train_lora.py, comparable to every other
result in this project) and split_by_set_3way (new, entire sets held out), the
latter used as an honesty check on whether by-name results are inflated by the
mechanism having trained on the same sets' bucket structure the "held-out" cards
come from.

Run with: conda run -n env_coinche python -m mtg_rating.train_color_context
"""

import copy
import math
import random
from pathlib import Path

import torch
from torch import nn

from mtg_rating.color_context import build_color_buckets, flatten_buckets
from mtg_rating.features import structured_features
from mtg_rating.model_joint import CardRatingNetJoint, ColorAttentionContext, DomainClassifier, GradientReversalLayer
from mtg_rating.multiset import SET_CODES, build_dataset
from mtg_rating.ratings import RAW_SCORE_FORMULAS, apply_normalization, fit_normalization
from mtg_rating.text_embeddings import EMBEDDING_DIM, MODEL_NAME
from mtg_rating.train_lora import score_predictions, split_by_name_3way, split_by_set_3way
from mtg_rating.features import STRUCTURED_DIM

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "color_context"
# Same reference split as train_lora.py so by-name results are directly comparable
# to every other number recorded for this project.
TEST_FRACTION = 0.2
VAL_FRACTION = 0.1
SPLIT_SEED = 42
FORMULA = "iih_only"
EPOCHS = 40
PATIENCE = 8
MIN_DELTA = 0.0
LORA_LR = 2e-4
HEAD_LR = 1e-3
CONTEXT_LR = 1e-3  # fresh/random-init component, same regime as the head
CONTEXT_DIM = 32
NUM_HEADS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bucket_forward(model, color_context, records, bucket, targets_by_index, device):
    """One (set, color) bucket -> raw per-target-card tensors (predictions,
    targets, and the context rows the domain classifier would use), or None if
    the bucket has no query card with a target in `targets_by_index`. No loss/
    backward here -- run_epoch pools this across several buckets before a
    single combined step (see run_epoch's docstring for why that matters for
    DANN specifically), and evaluate() calls it directly per-bucket since no
    gradient step is needed there either way.

    Runs the informant subset through attention regardless of targets -- other
    bucket members always contribute to context even if their own prediction
    isn't scored this call (e.g. val/test cards during a training pass)."""
    idx = bucket["query"]
    bucket_records = [records[i] for i in idx]
    structured = torch.tensor([r["structured"] for r in bucket_records], dtype=torch.float32, device=device)
    text_embed = model.text_encoder([r["oracle_text"] for r in bucket_records])
    combined = torch.cat([structured, text_embed], dim=-1)

    pos = {g: local for local, g in enumerate(idx)}
    informant_global = bucket["informant"] or idx  # fallback: no c/u members -> attend over everyone
    informant_local = [pos[g] for g in informant_global]
    context = color_context(combined, informant_local)

    head_input = torch.cat([structured, text_embed, context], dim=-1)
    out = model.head(head_input)
    out = out.squeeze(-1) if model.output_dim == 1 else out  # (n,) or (n, output_dim)

    target_positions = [local for local, g in enumerate(idx) if g in targets_by_index]
    if not target_positions:
        return None

    sel = torch.tensor(target_positions, device=device)
    target_vals = [targets_by_index[idx[local]] for local in target_positions]
    target_tensor = torch.tensor(target_vals, dtype=torch.float32, device=device)
    pred_sel = out[sel]

    if model.output_dim == 1:
        preds_out = {idx[local]: pred_sel[i].item() for i, local in enumerate(target_positions)}
    else:
        preds_out = {idx[local]: tuple(pred_sel[i].tolist()) for i, local in enumerate(target_positions)}

    return {
        "pred_sel": pred_sel,
        "target_tensor": target_tensor,
        "context_sel": context[sel],
        "n": len(target_positions),
        "preds_out": preds_out,
    }


def run_epoch(model, color_context, records, buckets, targets_by_index, device, optimizer, rng, dann=None, dann_chunk_size=1):
    """One optimizer step per `dann_chunk_size` buckets (default 1 -- exactly
    the original one-bucket-per-step behavior, unchanged for every non-DANN
    call site and bit-for-bit equivalent to the old per-bucket loop at
    chunk_size=1). DANN needs a bigger chunk: every card in a single bucket
    shares the *same* domain label (one set, one color), so a domain-
    classifier loss computed from one bucket alone is a degenerate single-
    class cross-entropy batch -- pushes the classifier to confidently predict
    whatever this step's one class is, then whiplashes it to a different
    class next step, never learning a real decision boundary (confirmed
    empirically: domain_acc sat at chance even at lambda=0, before any
    reversal existed to suppress it -- see project memory). Pooling several
    buckets (spanning multiple sets, since bucket order is globally shuffled)
    before the classifier's loss/backward gives it an actual multi-class
    batch to learn from."""
    model.train()
    color_context.train()
    order = list(range(len(buckets)))
    rng.shuffle(order)
    total_loss, total_n = 0.0, 0
    domain_correct, domain_total = 0, 0

    for start in range(0, len(order), dann_chunk_size):
        chunk = order[start : start + dann_chunk_size]
        pred_parts, target_parts, context_parts, domain_label_parts = [], [], [], []
        chunk_n = 0

        for bi in chunk:
            set_code, _, bucket = buckets[bi]
            result = _bucket_forward(model, color_context, records, bucket, targets_by_index, device)
            if result is None:
                continue
            pred_parts.append(result["pred_sel"])
            target_parts.append(result["target_tensor"])
            chunk_n += result["n"]
            if dann is not None:
                context_parts.append(result["context_sel"])
                domain_label = dann["domain_id_map"][set_code]
                domain_label_parts.append(torch.full((result["n"],), domain_label, dtype=torch.long, device=device))

        if not pred_parts:
            continue

        loss = nn.functional.mse_loss(torch.cat(pred_parts), torch.cat(target_parts))

        if dann is not None:
            domain_targets = torch.cat(domain_label_parts)
            domain_logits = dann["classifier"](dann["grl"](torch.cat(context_parts)))
            domain_loss = nn.functional.cross_entropy(domain_logits, domain_targets)
            loss = loss + dann["adv_weight"] * domain_loss
            # Diagnostic only (not used for optimization): how confidently the
            # classifier is actually distinguishing domains right now, so a
            # "DANN had no effect" result can be told apart from "the
            # classifier itself never got good enough to be a useful
            # adversary" -- see model_joint.py's DomainClassifier docstring.
            domain_correct += (domain_logits.argmax(dim=-1) == domain_targets).sum().item()
            domain_total += domain_targets.numel()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * chunk_n
        total_n += chunk_n

    domain_acc = domain_correct / domain_total if domain_total else None
    return total_loss / max(total_n, 1), domain_acc


def evaluate(model, color_context, records, buckets, targets_by_index, device):
    model.eval()
    color_context.eval()
    all_preds = {}
    with torch.no_grad():
        for _, _, bucket in buckets:
            result = _bucket_forward(model, color_context, records, bucket, targets_by_index, device)
            if result is not None:
                all_preds.update(result["preds_out"])
    indices = sorted(targets_by_index.keys())
    preds = [all_preds[i] for i in indices]
    targets = [targets_by_index[i] for i in indices]
    mse, extra = score_predictions(preds, targets)
    return mse, extra, preds


def save_checkpoint(model, color_context, mu, sigma, formula, checkpoint_dir=None, base_model_path=MODEL_NAME):
    checkpoint_dir = checkpoint_dir or CHECKPOINT_DIR
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.text_encoder.model.save_pretrained(checkpoint_dir / "lora_adapter")
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "color_context_state_dict": color_context.state_dict(),
            "context_dim": color_context.context_dim,
            "mu": mu,
            "sigma": sigma,
            "formula": formula,
            "base_model_path": str(base_model_path),
        },
        checkpoint_dir / "head.pt",
    )
    print(f"[checkpoint] saved to {checkpoint_dir}")


def main(
    set_codes: list = None,
    epochs: int = EPOCHS,
    seed: int = 0,
    save: bool = False,
    lora_rank: int = 4,
    formula: str = FORMULA,
    base_model_path=MODEL_NAME,
    context_dim: int = CONTEXT_DIM,
    num_heads: int = NUM_HEADS,
    attn_dropout: float = 0.0,
    weight_decay: float = 0.0,
    lora_lr: float = LORA_LR,
    head_lr: float = HEAD_LR,
    context_lr: float = CONTEXT_LR,
    split_mode: str = "name",  # "name" (default, comparable to train_lora.py) or "set" (honesty check)
    use_dann: bool = False,
    adv_weight: float = 0.1,
    classifier_lr: float = None,
    dann_chunk_size: int = 8,  # buckets pooled per domain-classifier step; see run_epoch's docstring
    checkpoint_dir: Path = None,
):
    torch.manual_seed(seed)

    records = build_dataset(set_codes)
    for r in records:
        r["structured"] = structured_features(r["scryfall_card"])
        r["oracle_text"] = r["scryfall_card"].get("oracle_text", "")
    records = [r for r in records if RAW_SCORE_FORMULAS[formula](r) is not None]

    if split_mode == "set":
        train_sel, val_sel, test_sel = split_by_set_3way(records, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)
        train_mask = lambda r: r["set_code"] in train_sel
        val_mask = lambda r: r["set_code"] in val_sel
        test_mask = lambda r: r["set_code"] in test_sel
    else:
        train_sel, val_sel, test_sel = split_by_name_3way(records, VAL_FRACTION, TEST_FRACTION, SPLIT_SEED)
        train_mask = lambda r: r["name"] in train_sel
        val_mask = lambda r: r["name"] in val_sel
        test_mask = lambda r: r["name"] in test_sel

    train_indices = {i for i, r in enumerate(records) if train_mask(r)}
    val_indices = {i for i, r in enumerate(records) if val_mask(r)}
    test_indices = {i for i, r in enumerate(records) if test_mask(r)}
    print(f"[split:{split_mode}] {len(train_indices)} train rows, {len(val_indices)} val rows, {len(test_indices)} test rows")

    fn = RAW_SCORE_FORMULAS[formula]
    train_raw = {i: fn(records[i]) for i in train_indices}
    mu, sigma = fit_normalization(train_raw)
    all_raw = {i: fn(r) for i, r in enumerate(records)}
    all_ratings = apply_normalization(all_raw, mu, sigma)
    train_targets = {i: all_ratings[i] for i in train_indices}
    val_targets = {i: all_ratings[i] for i in val_indices}
    test_targets = {i: all_ratings[i] for i in test_indices}

    color_buckets = flatten_buckets(build_color_buckets(records))
    print(f"[buckets] {len(color_buckets)} (set, color) groups")

    print(f"[device] {DEVICE}")
    model = CardRatingNetJoint(
        structured_dim=STRUCTURED_DIM,
        hidden_dims=[16],
        lora_rank=lora_rank,
        base_model_path=base_model_path,
        set_context_dim=context_dim,  # head is sized for the concatenated context vector either way
        output_dim=1,
    ).to(DEVICE)
    model.text_encoder.print_trainable_parameters()
    color_context = ColorAttentionContext(
        input_dim=STRUCTURED_DIM + EMBEDDING_DIM, context_dim=context_dim, num_heads=num_heads, dropout=attn_dropout
    ).to(DEVICE)

    param_groups = [
        {"params": model.text_encoder.trainable_parameters(), "lr": lora_lr},
        {"params": model.head.parameters(), "lr": head_lr, "weight_decay": weight_decay},
        {"params": color_context.trainable_parameters(), "lr": context_lr, "weight_decay": weight_decay},
    ]

    dann = None
    if use_dann:
        # Domains = whichever sets actually appear in train_indices -- 26 under
        # split_mode="name" (train rows are spread across every set), ~18 under
        # split_mode="set" (only the train-held sets appear at all). Built from
        # train_indices rather than train_sel directly so this is correct under
        # either split_mode without a special case.
        domain_codes = sorted({records[i]["set_code"] for i in train_indices})
        domain_id_map = {code: i for i, code in enumerate(domain_codes)}
        print(f"[dann] {len(domain_id_map)} domains (adv_weight={adv_weight})")
        domain_classifier = DomainClassifier(context_dim, len(domain_id_map)).to(DEVICE)
        grl = GradientReversalLayer(lambda_=0.0)  # ramped per-epoch below
        # Deliberately allowed to run faster than context_lr: a weak/undertrained
        # classifier produces a near-random reversed gradient, which is not
        # useful adversarial pressure -- the classifier should stay well ahead
        # of the encoder it's trying to be an adversary to.
        param_groups.append({"params": domain_classifier.parameters(), "lr": classifier_lr or context_lr})
        dann = {"classifier": domain_classifier, "grl": grl, "domain_id_map": domain_id_map, "adv_weight": adv_weight}

    optimizer = torch.optim.Adam(param_groups)

    rng = random.Random(seed)
    best_val_mse = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        if dann is not None:
            # Classic DANN ramp (Ganin & Lempitsky): 0 -> 1 over training so the
            # adversarial push only ramps up once the encoder/context has
            # something non-trivial to be pushed away from -- full-strength
            # reversal against a still-random encoder is a known destabilizer.
            progress = epoch / max(epochs - 1, 1)
            dann["grl"].lambda_ = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        train_mse, domain_acc = run_epoch(
            model, color_context, records, color_buckets, train_targets, DEVICE, optimizer, rng,
            dann=dann, dann_chunk_size=dann_chunk_size if dann is not None else 1,
        )
        val_mse, val_corr, _ = evaluate(model, color_context, records, color_buckets, val_targets, DEVICE)
        domain_detail = f"  domain_acc={domain_acc:.3f} (lambda={dann['grl'].lambda_:.2f})" if domain_acc is not None else ""
        print(f"[epoch {epoch}] train MSE={train_mse:.3f}  val MSE={val_mse:.3f} Pearson r={val_corr:.3f}{domain_detail}")

        if val_mse < best_val_mse - MIN_DELTA:
            best_val_mse = val_mse
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(color_context.state_dict()))
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[early stop] no val improvement for {PATIENCE} epochs, stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state[0])
    color_context.load_state_dict(best_state[1])

    test_mse, test_corr, preds = evaluate(model, color_context, records, color_buckets, test_targets, DEVICE)
    print(f"[test:{formula}] n={len(test_targets)} MSE={test_mse:.3f} Pearson r={test_corr:.3f}")

    if save:
        save_checkpoint(model, color_context, mu, sigma, formula, checkpoint_dir=checkpoint_dir, base_model_path=base_model_path)

    return model, color_context, best_val_mse, test_mse, test_corr


if __name__ == "__main__":
    main(SET_CODES)
