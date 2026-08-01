# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A neural card-rating tool for Magic: The Gathering limited/draft. Given a card's Scryfall
data (mana cost, type, oracle text, rarity, P/T), it predicts a 0-10 rating meant to
generalize to cards from **not-yet-released** sets. Training labels come from 17Lands'
public Premier Draft `game_data` (win-rate metrics), features from Scryfall. See
`premier_jet.md` for the original, informal design notes (kept as-is, worth reading for
the "why" behind several decisions below) and `regles_coinche.md`-style project memory
(in the auto-memory system, not this repo) for the full experimental history — this file
is a condensed, current-state summary, not a replacement for that history.

**Environment**: shares the `env_coinche` conda env with the sibling `Coinche` project
(disk-space constraint on this machine — do not create a separate env). `pip install`,
never `conda install`, for anything added to this env (conda's solver doesn't know about
pip-installed packages and can silently swap them out, see project memory for the
incident that established this rule).

## Commands

Run from the repo root, e.g. `conda run -n env_coinche python -m mtg_rating.train_lora`.

- **Main training pipeline**: `python -m mtg_rating.train_lora` — trains the current
  best-known config end to end (see below). `main()` takes many keyword args (formula,
  base_model_path, use_set_context, lora_rank, split_mode, loss shape/weighting options,
  etc.) — no CLI flags, call it from a `python -c` one-liner or a scratch script for
  anything other than the hardcoded defaults.
- **Smoke test**: `python -m mtg_rating.smoke_test` — trains a toy model per formula on
  frozen embeddings, fast sanity check.
- **MLM pretraining stage**: `python -m mtg_rating.mlm_pretrain` — produces a new
  MTG-adapted base encoder checkpoint (see Architecture). Only needs re-running if
  changing the pretraining corpus/objective, not for normal rating-model iteration.
- **Frozen-embedding baseline**: `python -m mtg_rating.train_multiset` — the older,
  simpler pipeline (frozen MiniLM, no LoRA fine-tuning), kept for comparison.
- **Experimental trainable-attention pipeline**: `python -m mtg_rating.train_color_context`
  — not part of the adopted config (see Status below), kept for any future revisit.
- **Rate a whole set and render an HTML list**: `python -m mtg_rating.rate_set` (edit the
  `main("DFT")` call at the bottom, or `python -c 'from mtg_rating.rate_set import main; main("MKM")'`
  for another set) — loads the `lora_joint_dual` checkpoint, rates every card of the given
  set, and writes `data/ratings/ratings_<set>.html`: one table per primary color (WUBRG
  first-color grouping, see `color_context.primary_color`), sorted strongest to weakest by
  predicted GP WR (IIH shown alongside). premier_jet.md section 5's original display idea.
  Needs only Scryfall data for that set, no 17Lands labels — works on any set, including
  ones outside `multiset.SET_CODES` with no draft history yet.

No automated test suite. Verification throughout this project has been: build a
`train_lora.main()` (or similar) call, run it, inspect val/test MSE and Pearson r printed
per epoch, and (for anything touching the by-name/by-set distinction, or any *trainable*
cross-card mechanism) explicitly re-check under `split_mode="set"` before trusting a
result — see "By-name vs by-set" below, this has bitten the project multiple times.

## Architecture

- **`fetch_17lands.py`** — downloads a set's raw `game_data_public.<SET>.PremierDraft.csv.gz`
  (deleted immediately after `labels.py` extracts per-card metrics — disk is tight on
  this machine, never more than one raw file kept at a time).
- **`fetch_scryfall.py`** — per-set card data (`cards/search?q=set:X`) and the full bulk
  `oracle_cards` corpus (`fetch_bulk_oracle_cards`, ~30k cards, used only by
  `mlm_pretrain.py`). Both cached as JSON.
- **`labels.py`** — `compute_card_metrics(game_data_path)`: turns one set's raw
  `game_data` into per-card `gih_wr`, `gns_wr`, `iih` (= `gih_wr - gns_wr`, the
  "intrinsic"/decontextualized signal), `gp_wr` (win rate over all games, more
  context-diluted than `gih_wr`), `play_rate`. Basic lands excluded. Careful, non-obvious
  memory-safety work here (usecols + small dtypes) — `game_data`/`draft_data` have OOM'd
  this machine before on wider sets.
- **`features.py`** — `structured_features()`: 14-dim hand-crafted vector (mana cost,
  color one-hot, type one-hot, rarity, power, toughness) from a Scryfall card dict. No
  "set" feature by design (a categorical set feature is useless for rating a genuinely
  new set, and needing one would mean the target formula wasn't decontextualized enough).
- **`text_embeddings.py`** — frozen (never fine-tuned) `sentence-transformers/all-MiniLM-L6-v2`
  mean-pooled embedding (384-dim), `MODEL_NAME`/`EMBEDDING_DIM` constants reused
  throughout. Used only where a fixed, non-trainable embedding is required (`set_context.py`).
- **`lora_text_encoder.py`** — `LoraTextEncoder`: the *trainable* text tower for the main
  pipeline. Wraps a base MiniLM checkpoint (generic, or the MLM-pretrained one below)
  with a small LoRA adapter (`LORA_RANK=4` default, on attention query/value) that gets
  fine-tuned jointly with the rating head. Rank 4 beat rank 8 and (mildly) rank 2 in an
  early sweep, and rank 8 was re-confirmed worse again under the current full config —
  don't increase this without a fresh, matched-condition test.
- **`mlm_pretrain.py`** — a *separate*, upstream stage: adapts the generic MiniLM base to
  MTG vocabulary via a larger-rank LoRA (32 or 64) trained with a masked-LM objective on
  the full ~30k-card Scryfall corpus, then `merge_and_unload()`s into a standalone base
  checkpoint (`data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/` is the one
  actually used downstream). This rank (32/64) is a completely different axis from
  `LoraTextEncoder`'s rank — don't conflate the two "LoRA rank" numbers when discussing
  results.
- **`model.py`** — `CardRatingNet`: small MLP on top of *frozen* embeddings (used only by
  `train_multiset.py`, the older baseline).
- **`model_joint.py`** — the main model classes:
  - `CardRatingNetJoint`: `LoraTextEncoder` output + `structured_features` (+ optional
    concatenated context vector) → MLP head (`hidden_dims=[16]` default — deeper heads
    tested and rejected, see Status) → 1 or N outputs.
  - `ColorAttentionContext`, `DomainClassifier`, `GradientReversalLayer`: built for the
    experimental trainable per-card-color attention + DANN line of work. **Not part of
    the adopted config** — see Status.
- **`set_context.py`** — non-parametric (zero trainable parameters) context vectors,
  concatenated to a card's own features before the head:
  - `compute_set_context_vectors`: mean of (structured + frozen text embedding) over a
    whole set. **Adopted**, part of the best-known config.
  - `compute_per_color_context_vectors`: same idea, grouped by (set, primary color)
    instead of whole-set. Tried, not adopted (no clear win over the whole-set version,
    see Status).
  - Both cache to `data/raw/*.json`, keyed by file existence only (not content
    coverage) — **regenerating with a different/smaller `set_codes` subset silently
    creates a stale cache**; always delete the cache file if changing what it should
    cover.
- **`color_context.py`** — `primary_color()` (WUBRG-order first color, else
  `"colorless"`) and bucket-building for the experimental trainable attention
  (`train_color_context.py`) and the per-color context mean above.
- **`ratings.py`** — `RAW_SCORE_FORMULAS` (named formulas like `iih_only`, `gp_wr_only`,
  `gih_iih`, combinations) and 0-10 normalization (`5 = mean`, `0`/`10` = `mean ± 3·std`
  of that formula's own raw-score distribution, clipped).
- **`multiset.py`** — `build_dataset()`: pools per-card records across `SET_CODES` (26
  sets, Alchemy and draft-innovation products excluded by rule). Caches per-set metrics
  as small CSVs.
- **`train_lora.py`** — **the main production training script.** `main()` builds the
  pooled dataset, splits by card name (default) or by whole set (`split_mode="set"`,
  the honesty check), trains `CardRatingNetJoint` end to end with differential LRs
  (LoRA slower than the fresh head), early stopping on val MSE (never test), and several
  opt-in-only loss variants (`sample_weight_power`, `threshold_penalty_weight`,
  `loss_shape="saturating"`, `contrastive_weight`, `triplet_weight`) — **all default to
  plain unweighted MSE, none of them confirmed to help, see Status.** `triplet_weight`
  is the newest: `mine_hard_triplets` periodically (`triplet_mining_every`, default every
  epoch) re-embeds the whole train set to find each card's hardest same-text/different-
  outcome negative (the Annul-probe failure pattern), paired with its closest-real-target
  positive (no embedding search needed for that half); `triplet_loss` is a standard
  cosine-distance margin loss on those mined triples. Sharper-targeted successor to
  `contrastive_weight`'s random-in-batch-pairs version (which showed no reproducible
  effect across 12 runs) — **not yet run/confirmed itself**, only smoke-tested so far.
- **`train_color_context.py`** — the experimental trainable-attention pipeline (buckets
  cards by (set, primary color), cross-attention over commons/uncommons as the
  informant pool). Includes DANN support. **Not adopted** — kept for reference/reuse if
  this direction is revisited.
- **`train_multiset.py`** — older frozen-embedding trainer, `pearson()` (reused by
  `train_lora.py`), `save_pool_summary()`.
- **`smoke_test.py`** — quick end-to-end check across all three original formulas.
- **`rate_set.py`** — inference/display only, no training: loads a saved `CardRatingNetJoint`
  checkpoint (`load_model`, handles both the current `extra_formulas`-list checkpoint
  format and the older single-`second_formula` shape `lora_joint_dual/head.pt` was saved
  with), rates every non-basic-land card of a given set (`rate_cards`), and renders
  `render_html`'s per-color (strong-to-weak) HTML table. `_context_vector_for_set` reuses
  `set_context.py`'s cached whole-set vector for the 26 pooled sets, or computes the same
  mean fresh — without writing to that shared cache — for any other set.

## Current best-known config

Rank-4 rating LoRA on the rank-64/epoch-4 MLM-pretrained base
(`data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/`) + whole-set
`set_context` mean + dual `(iih_only, gp_wr_only)` output, plain MSE loss, 70/10/20
split by card name (`SPLIT_SEED=42`), head `[16]`, no dropout, differential LR
(LoRA 2e-4 / head 1e-3), early stopping (patience 8). Checkpoint at
`data/models/lora_joint_dual/`. Equivalent to:

```python
from mtg_rating.train_lora import main
main(
    formula="iih_only", extra_formulas=["gp_wr_only"],
    use_set_context=True,
    base_model_path="data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4",
)
```

## Key established facts (don't re-litigate without new evidence)

- **IIH vs GP WR**: IIH is the decontextualized/intrinsic signal (mostly a function of
  the card's own text — learnable well from card features alone, test r≈0.6). GP WR is
  the contextual signal (depends on the card's actual deck/archetype environment, which
  literally doesn't exist yet for an unreleased set) — harder, current best r≈0.48-0.5.
  Both are trained jointly (dual output) since predicting IIH alongside GP measurably
  improves GP, but only once there's enough adapter capacity to serve both objectives
  (true at MLM-pretraining rank 64, not at rank 32; the *rating*-LoRA's own rank 4 has
  never shown a capacity benefit from going higher, checked twice).
- **By-name vs by-set split**: by-name (the default, comparable to nearly every recorded
  result) is a safe, reliable evaluation **only for zero-trainable-parameter mechanisms**
  (the whole-set/per-color `set_context` mean). Any *trainable* mechanism that couples
  multiple cards together (the experimental attention, DANN) has shown large, sometimes
  fully-reversing inflation under by-name that evaporates under a genuine `split_mode="set"`
  check — always run that check before trusting a by-name "win" for anything trainable
  that spans more than one card.
- **Always confirm across multiple seeds before trusting a "win".** This project has
  repeatedly seen single-seed results look clean and consistent, then flip on
  confirmation (a threshold-penalty loss variant looked like the best result of an
  entire session at seed 0, then lost on 5 of 6 seeds once checked). Model/hyperparameter
  selection must use **val**, never test — test is touched exactly once, after every
  other decision is locked in.
- **GP WR is under-dispersed**: predicted GP WR values have roughly half the variance of
  actual GP WR (std ratio ≈0.5), despite a well-calibrated mean — the model is timid at
  the extremes (systematically over-predicts weak/narrow cards, under-predicts bombs).
  Correlates with per-card game count (low-sample cards get the worst predictions), but
  this isn't purely measurement noise — narrow/bad cards are *played* less, and at least
  some of the worst misses (e.g. Annul, "counter target artifact or enchantment") have
  textually explicit restrictions the model isn't pricing in. A direct embedding-space
  probe confirmed this concretely: Annul's *closest* embedding-space neighbor among
  comparable blue instants is a much more flexible, better-performing counterspell —
  the encoder currently organizes by surface "counter"/"target" vocabulary, not by
  functional restrictiveness. **Five different fixes were tried and none held up under
  multi-seed confirmation**: sample weighting by game count (made it worse — high-count
  cards are disproportionately unexciting average commons, not reliable extremes),
  a hinge-squared threshold penalty, a Geman-McClure-style saturating loss (replacing
  MSE outright softens gradient for normal cards too, not just outliers), a contrastive
  embedding-space auxiliary loss (12 runs across weight/temperature, no reproducible
  trend), and more rating-LoRA capacity (rank 8, worse again). This is an **open
  problem** — see below for untried directions.
- **Reprints across sets, basic lands, Alchemy/draft-innovation sets**: reprints are
  rare enough (~1% of the pooled dataset) not to be a serious train/test leakage risk,
  but the split is still by unique card name as a free hygiene practice. Basic lands are
  hard-excluded in `labels.py` (different draw/sample dynamics, shouldn't share a
  distribution with spells). Alchemy (digital-only) and draft-innovation (MH3, LTR-style
  supplemental products) sets are excluded from `SET_CODES` by rule, not case-by-case.

## Open / not yet tried

- **Validated feature engineering for GP WR's under-dispersion**: rather than
  hand-guessing which text patterns signal restrictiveness (tried once, explicitly
  rejected as too approximate — e.g. counting target-category keywords with no
  prevalence weighting, or an "unless/if" flag that can't distinguish restrictive
  clauses from bonus ones), empirically search a broad candidate keyword/pattern list
  for real correlation with the GP-vs-IIH residual *before* encoding anything.
- **Richer contrastive formulation**: the simple pairwise soft-target version tried
  (`exp(-|target_gap|/tau)`) showed no reproducible effect across 12 runs. **Now
  implemented** as `train_lora.py`'s `triplet_weight`/`mine_hard_triplets` (periodic
  hard-negative mining: same-text/different-outcome pairs like Annul/Protect-the-
  Negotiators) — smoke-tested only so far, not yet run for a real result or
  multi-seed-confirmed.
- **v2 richer color-pair grouping** for the (currently not-adopted) trainable attention
  line: group by 2-color archetype pairs instead of single primary color — never built,
  the single-color version was meant as the simpler stepping stone and was abandoned
  once it failed the by-set check.
- **Multi-seed confirmation of the trainable-attention by-set overfitting finding**
  itself — currently single-seed, directionally very likely real (consistent across
  every regularization variant tried) but never formally 3-seed-confirmed the way most
  other findings in this project have been.
- **Secondary objectives from the original design notes** (`premier_jet.md` §1.2),
  untouched so far: short natural-language card descriptions/characteristics alongside
  the numeric rating, set-dependent metagame prediction, draft-bot integration.
