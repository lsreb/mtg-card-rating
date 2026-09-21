# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A neural card-rating tool for Magic: The Gathering limited/draft. Given a card's Scryfall
data (mana cost, type, oracle text, rarity, P/T), it predicts a 0-10 rating meant to
generalize to cards from **not-yet-released** sets. Training labels come from 17Lands'
public Premier Draft `game_data` (win-rate metrics), features from Scryfall. See
`design_notes.md` for the original, informal design notes (kept as-is, worth reading for
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
  — not part of the adopted config (see Key established facts below), kept for any future revisit.
- **Rate a whole set and render an HTML list**: `python -m mtg_rating.rate_set` (edit the
  `main("DFT")` call at the bottom, or `python -c 'from mtg_rating.rate_set import main; main("MKM")'`
  for another set) — loads the `lora_joint_dual_msh` checkpoint, rates every card of the given
  set, and writes `data/ratings/ratings_<set>.html`: one table per color group (each of
  WUBRG, plus separate multicolor and colorless sections, see `rate_set.display_group`),
  sorted strongest to weakest by predicted GP WR (IIH shown alongside, the real 17Lands
  value in parentheses when one exists). design_notes.md section 5's original display idea.
  Needs only Scryfall data for that set, no 17Lands labels — works on any set, including
  ones outside `multiset.SET_CODES` with no draft history yet.
- **Diagnostics on a checkpoint** (read-only, no training): `python -m mtg_rating.feature_search`,
  `python -m mtg_rating.opponent_benefit_probe`, `python -m mtg_rating.knn_baseline`. Each has
  `main(checkpoint_dir=...)`, defaulting to `rate_set.CHECKPOINT_DIR` (`lora_joint_dual_msh`,
  the 27-set baseline). They rebuild the dataset from the sets recorded in the checkpoint (`train_lora.resolve_training_sets`;
  both existing checkpoints were backfilled 2026-09-21, pools verified against their recorded split row
  counts), so the split is the checkpoint's own. A checkpoint with no record falls back to the current
  `SET_CODES` with a warning; `set_codes=` overrides (split-drift note under Key established facts).

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
    tested and rejected) → 1 or N outputs.
  - `ColorAttentionContext`, `DomainClassifier`, `GradientReversalLayer`: built for the
    experimental trainable per-card-color attention + DANN line of work. **Not part of
    the adopted config** — see Key established facts.
- **`set_context.py`** — non-parametric (zero trainable parameters) context vectors,
  concatenated to a card's own features before the head:
  - `compute_set_context_vectors`: mean of (structured + frozen text embedding) over a
    whole set. **Adopted**, part of the best-known config.
  - `compute_per_color_context_vectors`: same idea, grouped by (set, primary color)
    instead of whole-set. Tried, not adopted (no clear win over the whole-set version).
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
- **`multiset.py`** — `build_dataset()`: pools per-card records across `SET_CODES` (27
  sets as of MSH's addition on 2026-08-20; Alchemy and draft-innovation products excluded by
  rule). Caches per-set metrics
  as small CSVs.
- **`train_lora.py`** — **the main production training script.** `main()` builds the
  pooled dataset, splits by card name (default) or by whole set (`split_mode="set"`,
  the honesty check), trains `CardRatingNetJoint` end to end with differential LRs
  (LoRA slower than the fresh head), early stopping on val MSE (never test), and several
  opt-in-only loss variants (`sample_weight_power`, `threshold_penalty_weight`,
  `loss_shape="saturating"`, `contrastive_weight`, `triplet_weight`) — **all default to
  plain unweighted MSE, none of them confirmed to help, see Key established facts.** `triplet_weight`
  is the newest: `mine_hard_triplets` periodically (`triplet_mining_every`, default every
  epoch) re-embeds the whole train set to find each card's hardest same-text/different-
  outcome negative (the Annul-probe failure pattern), paired with its closest-real-target
  positive (no embedding search needed for that half); `triplet_loss` is a standard
  cosine-distance margin loss on those mined triples. Sharper-targeted successor to
  `contrastive_weight`'s random-in-batch-pairs version (which showed no reproducible
  effect across 12 runs) — **tried, ruled out** (see Key established facts): no benefit at
  `weight=1.0/margin=0.5` (2 seeds), and monotonically *worse* (both IIH and GP WR) at
  `weight=5.0/margin=1.0`, consistent with fighting the main objective rather than
  complementing it on this little trainable capacity, not just an under-tuned weight.
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
  `set_context.py`'s cached whole-set vector for any set in `SET_CODES`, or computes the same
  mean fresh — without writing to that shared cache — for any other set.
- **`feature_search.py`, `opponent_benefit_probe.py`, `knn_baseline.py`** — read-only
  diagnostics on a trained checkpoint (`checkpoint_dir`; no training), written to investigate
  GP WR misses first noticed by eye on MSH (findings under Key established facts). All three
  call `rate_set.load_model` and rebuild the dataset through
  `multiset.build_dataset(train_lora.resolve_training_sets(set_codes, checkpoint_dir))`, keeping
  the records that have every formula the checkpoint outputs (train_lora's own filter);
  `opponent_benefit_probe` also imports helpers from `feature_search`.
  - `feature_search.py`: residual = actual − predicted `gp_wr_only` rating (0-10 scale, via
    the checkpoint's own mu/sigma). Every single word and bigram of `oracle_text` (reminder
    text stripped) with enough support (≥30 train, ≥8 val cards) is correlated against it on
    train; a pattern counts as "confirmed" only with the same sign and |r| ≥ 0.05 on val. It
    only ranks candidates, it adds nothing to the model. Train rows are in-sample for the
    checkpoint (deflated residuals), so the val column is the clean one.
  - `opponent_benefit_probe.py`: one targeted hypothesis, from The Sentry, Golden Guardian
    (reads as strong text but gives the *opponent* a 5/5 flying indestructible token). Buckets
    cards by whether "opponent(s)" is followed within 3 words by a benefit verb, a harm verb,
    neither, or does not appear at all, and compares the mean residual per bucket.
  - `knn_baseline.py`: "closest comparable card" retrieval — k-NN over z-scored structured
    features plus the checkpoint's own fine-tuned LoRA text embedding (grid k ∈ {3, …, 200}
    × structured weight ∈ {1, 3, 10}, plus an NN+kNN blend at the val-best config), compared
    on val against the network's own head.

## Current best-known config

Rank-4 rating LoRA on the rank-64/epoch-4 MLM-pretrained base
(`data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/`) + whole-set
`set_context` mean + dual `(iih_only, gp_wr_only)` output, plain MSE loss, 70/10/20
split by card name (`SPLIT_SEED=42`), head `[16]`, no dropout, differential LR
(LoRA 2e-4 / head 1e-3), early stopping (patience 8). Two checkpoints of this config exist,
trained on different pools: `data/models/lora_joint_dual/` (26 sets, before MSH) and
`data/models/lora_joint_dual_msh/` (27 sets — **the current baseline**, see below). Equivalent to:

```python
from mtg_rating.train_lora import main
main(
    formula="iih_only", extra_formulas=["gp_wr_only"],
    use_set_context=True,
    base_model_path="data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4",
)
```

**Current baseline: `lora_joint_dual_msh`** (trained 2026-09-21; seed 0; 27-set pool; 4729 /
679 / 1356 train / val / test rows, `SPLIT_SEED=42`; best val epoch 13, early stop at epoch 21,
~6 min on the GTX 1660 Super). Saved by adding `checkpoint_dir="data/models/lora_joint_dual_msh"`
to the call above (str or `Path`; coerced since 2026-09-21, before that a str crashed at the very end of training). Test
performance: combined MSE 1.820; `iih_only` MSE 1.597, r 0.599; `gp_wr_only` MSE 2.043,
r 0.469. Single seed. Compare new experiments against this, on this split; the numbers in
the next paragraph belong to the 26-set checkpoint's own, different split and are not
comparable to it.

Exact test-set performance of the 26-set `lora_joint_dual` checkpoint (1303 test rows, `SPLIT_SEED=42`,
deterministic -- re-measured directly by loading `lora_joint_dual/head.pt` and
evaluating, not a training-time log, since none was kept for the original run):
combined MSE 1.727; `iih_only` MSE 1.460, r 0.630; `gp_wr_only` MSE 1.994, r 0.482.
Matches the r range below (0.6 / 0.48-0.5) but is the first time the MSE side was
pinned down precisely.

**Which pool these numbers belong to**: this checkpoint was trained (2026-07-29) on the 26
sets that were in `SET_CODES` then — every current set except MSH, added 2026-08-20 — and the
1303 test rows are that 26-set pool's split (1271 test names). Re-measuring today by simply
calling `build_dataset()` yields a *different* split (see the split-drift note under Key
established facts), not this one; restrict the dataset to those 26 sets to reproduce it.

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
  functional restrictiveness. **Six different fixes were tried and none held up under
  multi-seed confirmation**: sample weighting by game count (made it worse — high-count
  cards are disproportionately unexciting average commons, not reliable extremes),
  a hinge-squared threshold penalty, a Geman-McClure-style saturating loss (replacing
  MSE outright softens gradient for normal cards too, not just outliers), a contrastive
  embedding-space auxiliary loss (12 runs across weight/temperature, no reproducible
  trend), more rating-LoRA capacity (rank 8, worse again), and a hard-negative-mined
  triplet loss (`train_lora.py`'s `triplet_weight`, see Architecture) — no benefit at
  `weight=1.0/margin=0.5` (2 seeds), monotonically *worse* at `weight=5.0/margin=1.0`
  (1 seed), i.e. more pressure made it worse rather than revealing an under-tuned
  weight, consistent with the auxiliary objective fighting the main one on the rating-
  LoRA's very small trainable capacity (36,864 params) rather than complementing it.
  This is an **open problem** — see below for untried directions.
- **Reprints across sets, basic lands, Alchemy/draft-innovation sets**: reprints are
  rare enough (~1% of the pooled dataset) not to be a serious train/test leakage risk,
  but the split is still by unique card name as a free hygiene practice. Basic lands are
  hard-excluded in `labels.py` (different draw/sample dynamics, shouldn't share a
  distribution with spells). Alchemy (digital-only) and draft-innovation (MH3, LTR-style
  supplemental products) sets are excluded from `SET_CODES` by rule, not case-by-case.
- **The by-name split depends on the whole pool of names — adding a set silently re-deals
  every card.** `split_by_name_3way` sorts *all* unique names, shuffles them with the fixed
  seed and slices, so any change to the pool (MSH: 26 → 27 sets, 1271 → 1324 test names)
  changes which cards are train/val/test, not just where the new set's cards land. Measured:
  about 70% of the 27-set val and test names were in the checkpoint's own *train* split —
  the chance rate for a random reshuffle, since train is 70% of the names. Consequence: an
  older checkpoint must be evaluated on the exact pool it was trained on, otherwise its
  "held-out" numbers are largely in-sample. Since 2026-09-21 `save_checkpoint` records
  `set_codes` and the split settings in `head.pt` (read back by `train_lora.load_training_pool` /
  `resolve_training_sets`); the two earlier checkpoints were backfilled the same day (`lora_joint_dual`: every set except MSH,
  `lora_joint_dual_msh`: all 27) after checking each pool reproduces its recorded split row counts.
- **Three diagnostics on the GP WR misses, run on `lora_joint_dual_msh` (its own split), found
  nothing that explains the big misses.** Trigger: several MSH cards flagged by eye as badly
  rated, all confirmed against the real 17Lands data (e.g. Avengers Assemble! predicted
  4.01/3.85 on the two outputs vs real 8.62/8.10; The Sentry, Golden Guardian predicted IIH
  7.16 vs real 4.01; both from the 26-set checkpoint, which had never seen MSH). Train rows are
  in-sample for the checkpoint, so val is the clean column. (1) `feature_search.py` — no
  word/bigram pattern clears a correction for the 628 candidates. The only coherent thing is a
  small land/mana cluster ("land", "tapped", "enters tapped", "add", "{T}: Add") with the same
  negative residual sign on train and val, i.e. lands and mana text are slightly overrated, by
  roughly 0.2 rating points against a residual std of ~1.25 — far too small to explain misses
  of ±3. (2) `opponent_benefit_probe.py` — flat aggregate effect (benefit bucket mean +0.03 vs
  +0.08 for no-"opponent" cards; n=61, std 1.68 vs 1.25). The same cards stay the worst
  (Eiganjo Uprising, Asinine Antics, Icebreaker Kraken, Acererak the Archlich, Flumph); of those,
  reading the real text, Flumph clearly and Eiganjo Uprising plausibly fit the "effect helps the
  opponent" mechanism and the other three are classifier false positives (the regex ignores
  grammatical subject and negation). (3) `knn_baseline.py` — clearly worse than the network on
  val (best k=50: IIH MSE 2.054 / GP MSE 2.355, network 1.633 / 1.865); an NN+kNN blend at
  weight 0.1-0.2 moves IIH MSE by about -0.02 and GP by about 0, within single-split noise —
  not worth adopting. All single seed, single split.

## Open / not yet tried

- **Validated feature engineering for GP WR's under-dispersion**: rather than
  hand-guessing which text patterns signal restrictiveness (tried once, explicitly
  rejected as too approximate — e.g. counting target-category keywords with no
  prevalence weighting, or an "unless/if" flag that can't distinguish restrictive
  clauses from bonus ones), empirically search a broad candidate keyword/pattern list
  for real correlation with the GP-vs-IIH residual *before* encoding anything. **The search
  itself has now been run** (`feature_search.py`, single words and bigrams, on
  `lora_joint_dual_msh`'s own split) and found nothing reliable beyond a small land/mana
  cluster (see Key established facts). Single words/bigrams cannot express grammatical
  structure (negation, who an effect applies to) — the opponent probe's false positives show
  that gap — so whether something structural (real parsing) would find more is the open
  question; the flat-word version does not justify building features.
- **Pin the training pool in checkpoints**: done 2026-09-21 — `save_checkpoint` records `set_codes` + split
  settings, the diagnostics read them through `resolve_training_sets` (tested with a real 1-epoch run),
  and both existing checkpoints were backfilled (originals backed up outside the repo). Also done:
  `save_checkpoint`, `rate_set.load_model`/`main`, `mlm_pretrain.main` and `train_color_context.save_checkpoint`
  coerce their directory args with `Path(...)`, so str paths work. (Retraining on all 27 sets and
  re-running the diagnostics against it was done the same day: `lora_joint_dual_msh`.)
- **v2 richer color-pair grouping** for the (currently not-adopted) trainable attention
  line: group by 2-color archetype pairs instead of single primary color — never built,
  the single-color version was meant as the simpler stepping stone and was abandoned
  once it failed the by-set check.
- **Multi-seed confirmation of the trainable-attention by-set overfitting finding**
  itself — currently single-seed, directionally very likely real (consistent across
  every regularization variant tried) but never formally 3-seed-confirmed the way most
  other findings in this project have been.
- **Secondary objectives from the original design notes** (`design_notes.md` §1.2),
  untouched so far: short natural-language card descriptions/characteristics alongside
  the numeric rating, set-dependent metagame prediction, draft-bot integration.
