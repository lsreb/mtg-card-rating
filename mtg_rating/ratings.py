"""Convert raw per-card metrics (labels.compute_card_metrics) into a 0-10 rating.

Three candidate target formulas, all algebraically related via the identity
gih_wr = gp_wr + (1 - p) * iih (p = fraction of games the card was actually drawn):
- "gih": gih_wr alone -- the raw, most "contextual" signal (includes archetype/deck
  strength baked in).
- "gih_iih": gih_wr + iih -- pushes furthest towards an intrinsic, decontextualized
  signal (works out to (1+p)*gih_wr - p*gns_wr in terms of the two underlying win
  rates).
- "gp_iih": gp_wr + iih -- works out to gih_wr + p*iih, i.e. the *exact* p-weighted
  average of "gih" and "gih_iih" (not an approximation): a card seen in a higher
  fraction of its games sits closer to "gih_iih"; one seen rarely sits closer to
  "gih". See project discussion/memory for why this makes it a natural middle
  ground between the two other formulas, not just a visual coincidence.

0-10 normalization: 5 = mean across the card pool, 0 and 10 = mean +/- STD_MULTIPLIER
standard deviations of the raw score (not literal min/max, which is an unstable,
noisy estimate on a single set), clipped to [0, 10]. `fit_normalization` and
`apply_normalization` are split apart (rather than baked into one step) so a
train/test pipeline can fit mu/sigma on the training split only and apply that same
transform to held-out data, instead of leaking test-set statistics into the scale.
`normalize_to_10` is the fit+apply shortcut used when there's no split (single-set
smoke test).
"""

import statistics

STD_MULTIPLIER = 3.0

RAW_SCORE_FORMULAS = {
    "gih": lambda m: m["gih_wr"],
    "gih_iih": lambda m: m["gih_wr"] + m["iih"],
    "gp_iih": lambda m: m["gp_wr"] + m["iih"],
}


def raw_scores(metrics: dict, formula: str) -> dict:
    fn = RAW_SCORE_FORMULAS[formula]
    return {name: fn(m) for name, m in metrics.items() if m.get("iih") is not None}


def fit_normalization(raw: dict) -> tuple:
    values = list(raw.values())
    return statistics.fmean(values), statistics.stdev(values)


def apply_normalization(raw: dict, mu: float, sigma: float) -> dict:
    return {
        key: min(10.0, max(0.0, 5 + 5 * (v - mu) / (STD_MULTIPLIER * sigma)))
        for key, v in raw.items()
    }


def normalize_to_10(raw: dict) -> dict:
    return apply_normalization(raw, *fit_normalization(raw))


def compute_ratings(metrics: dict, formula: str) -> dict:
    return normalize_to_10(raw_scores(metrics, formula))
