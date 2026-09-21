"""Targeted follow-up to feature_search.py's negative result -- instead of scanning
all words/bigrams blindly (which found nothing reliable), test one specific
hypothesis motivated by a concrete miss: The Sentry, Golden Guardian (MSH) reads as
strong text (flying, vigilance, indestructible, 5/5, legendary) but actually gives
the *opponent* a 5/5 flying indestructible token -- the model rated it #7/40 in its
color (predicted IIH 7.16) while its real IIH is 4.01, below average. The
hypothesis: the model doesn't reliably track *who* an effect benefits, and
overrates any card whose text contains a "grant" verb near the word "opponent"
regardless of the fact that granting something to an opponent is bad for the
caster.

Groups every pooled train+val card into one of four buckets by scanning for
"opponent"/"opponents" followed within 3 words by a benefit verb (creates, draws,
gains, ...) or a harm verb (loses, discards, sacrifices, ...), then compares mean
prediction residual (actual - predicted GP WR rating, from `checkpoint_dir`,
default rate_set.CHECKPOINT_DIR; the dataset is rebuilt from the sets that
checkpoint was trained on, so the split is its own -- see
train_lora.resolve_training_sets) across buckets. Read-only diagnostic, no training,
same residual computation as feature_search.py, so the same caveat applies: train
rows are in-sample for the checkpoint and their residuals are deflated.

Run with: conda run -n env_coinche python -m mtg_rating.opponent_benefit_probe
"""

import re
import statistics

from mtg_rating.feature_search import _residuals, _WORD_RE, _PAREN_RE
from mtg_rating.features import structured_features
from mtg_rating.multiset import build_dataset
from mtg_rating.rate_set import CHECKPOINT_DIR, load_model
from mtg_rating.ratings import RAW_SCORE_FORMULAS
from mtg_rating.set_context import compute_set_context_vectors
from mtg_rating.train_lora import SPLIT_SEED, TEST_FRACTION, VAL_FRACTION, resolve_training_sets, split_by_name_3way

TARGET_FORMULA = "gp_wr_only"

BENEFIT_VERBS = {"creates", "create", "draws", "draw", "gains", "gain", "untaps", "untap", "returns", "return", "puts", "put"}
HARM_VERBS = {"loses", "lose", "discards", "discard", "sacrifices", "sacrifice", "exiles", "exile", "mills", "mill"}


def _classify(oracle_text: str) -> str:
    text = _PAREN_RE.sub(" ", (oracle_text or "").lower())
    words = _WORD_RE.findall(text)
    tags = set()
    for i, w in enumerate(words):
        if w in ("opponent", "opponents"):
            window = words[i + 1 : i + 4]
            if any(v in BENEFIT_VERBS for v in window):
                tags.add("benefit")
            if any(v in HARM_VERBS for v in window):
                tags.add("harm")
    if "benefit" in tags:
        return "benefit"
    if "harm" in tags:
        return "harm"
    if any(w in ("opponent", "opponents") for w in words):
        return "opponent_other"
    return "none"


def main(checkpoint_dir=CHECKPOINT_DIR, set_codes=None):
    model, formulas, use_set_context, norm_params = load_model(checkpoint_dir)
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
    pool = [r for r in records if r["name"] in train_sel or r["name"] in val_sel]
    print(f"[opponent_probe] {len(pool)} train+val cards (test untouched)")

    residuals = _residuals(model, target_idx, mu, sigma, pool)
    buckets = {}
    for r, res in zip(pool, residuals):
        buckets.setdefault(_classify(r["oracle_text"]), []).append((r, res))

    print(f"{'bucket':<16} {'n':>5} {'mean residual':>14} {'std':>8}")
    for name in ["benefit", "harm", "opponent_other", "none"]:
        rows = buckets.get(name, [])
        if not rows:
            continue
        vals = [res for _, res in rows]
        print(f"{name:<16} {len(vals):>5} {statistics.fmean(vals):>+14.3f} {statistics.pstdev(vals):>8.3f}")

    print("\n[opponent_probe] 'benefit' bucket examples (name, predicted GP rating, actual GP rating, residual):")
    for r, res in sorted(buckets.get("benefit", []), key=lambda x: x[1])[:10]:
        pred = RAW_SCORE_FORMULAS[TARGET_FORMULA](r)
        print(f"  {r['name']:<35} residual={res:+.2f}  set={r['set_code']}")


if __name__ == "__main__":
    main()
