# MTG Card Rating

A neural network that reads a Magic: The Gathering card (mana cost, colors, type, rarity,
power/toughness and oracle text) and predicts a **0-10 limited/draft rating**, trained on
public win-rate data from [17Lands](https://www.17lands.com/). The goal is to rate cards
from sets that **have not been released yet**, so the model only ever sees the card
itself: no set identifier, no draft context.

This is a research project in progress, not a polished tool. Trained checkpoints are not
included in the repository (see [Quickstart](#quickstart)).

## How it works

```
oracle text ──► MiniLM encoder (MTG-adapted, then LoRA fine-tuned) ─┐
14 structured features (cost, colors, types, rarity, P/T) ──────────┼──► MLP head ──► 2 ratings (0-10)
set context (mean vector of the card's set, no trainable params) ───┘        IIH-based  +  GP WR-based
```

**What it predicts.** Raw win rates mix a card's own power with the strength of the deck
and archetype it ends up in. A model that only sees the card cannot learn the deck part,
and for an unreleased set that context does not exist yet. So the main target is **IIH**
(improvement when in hand, `GIH WR - GNS WR`), which isolates the card's own effect. It is
trained jointly with **GP WR** (win rate over all games where the card was in the deck),
the more contextual signal. Predicting IIH alongside GP WR measurably improves GP WR. Each
target is normalized to a 0-10 scale (5 = mean, 0/10 = mean ± 3 standard deviations).

**Text model.** `all-MiniLM-L6-v2` is first adapted to MTG vocabulary with a rank-64 LoRA
and a masked-language-model objective over roughly 30,000 Scryfall oracle cards
(`mlm_pretrain.py`). It is then fine-tuned jointly with the rating head through a small
rank-4 LoRA (36,864 trainable parameters, `train_lora.py`).

**Set context.** Each card also receives the mean feature vector of every card in its set,
a cheap summary of "what kind of set is this" with no trainable parameters.

**Evaluation discipline.** Train/val/test (70/10/20) is split by unique card name.
Early stopping and every model or hyperparameter choice use the validation set only; the
test set is touched once at the end. Anything trainable that lets several cards interact
is additionally checked with whole sets held out (`split_mode="set"`), because gains that
looked real under the by-name split have collapsed under that check before.

## Results

Current best configuration, trained on the 27-set pool and evaluated on its held-out test
set (1,356 rows, one per card and set, split by card name, `SPLIT_SEED=42`). Errors are in
rating points on the 0-10 scale.

| target | test Pearson r | test MSE | RMSE |
|---|---|---|---|
| IIH (`iih_only`) | 0.599 | 1.597 | ≈ 1.3 |
| GP WR (`gp_wr_only`) | 0.469 | 2.043 | ≈ 1.4 |

Caveats worth knowing:

- One training run (one seed) on one split. Elsewhere in this project, single-seed results
  have repeatedly looked good and then not survived multi-seed confirmation.
- The by-name split shuffles the whole list of card names, so adding a set to the pool
  re-deals every card, and numbers from different pools are not comparable. An earlier
  checkpoint trained on the 26 sets that existed before MSH scored r = 0.630 (IIH) and
  0.482 (GP WR) on its own, different test split. Checkpoints record the
  sets they were trained on, so that split can always be rebuilt.
- GP WR predictions are under-dispersed: their spread is roughly half that of the real
  values. The model is timid at the extremes, over-rating weak or narrow cards and
  under-rating bombs. A likely cause is that the encoder groups cards by surface
  vocabulary ("counter", "target") rather than by how restrictive the effect is.

**What did not help**, each tried and dropped under multi-seed checks:

- Loss variants for the extremes: game-count sample weighting, a threshold penalty, a
  saturating (Geman-McClure-style) loss, a contrastive auxiliary loss, and a
  hard-negative triplet loss.
- More capacity: a rating-LoRA rank of 8 (rank 4 stays best), and deeper MLP heads.
- A trainable per-color attention mechanism, with and without a domain-adversarial (DANN)
  objective. Its by-name gains disappeared with whole sets held out.
- Per-color (instead of whole-set) context vectors: no clear win.
- Three diagnostic tools, run on the current checkpoint with its own split (validation
  cards are the clean column): an automatic word/bigram scan of oracle text against the
  model's residuals (no pattern survives a correction for the 628 candidates; the only
  coherent signal is a small land/mana cluster, about 0.2 rating points against a residual
  spread of about 1.25, far too small to explain misses of ±3), a targeted "effect benefits
  the opponent" hypothesis (flat in aggregate; it fits at most a couple of individual
  cards), and a k-nearest-neighbour "closest comparable card" baseline (clearly worse than
  the network; blending it in gains nothing beyond single-split noise). All single seed.

## Quickstart

Tested with Python 3.13, `torch` 2.12, `transformers` 5.14, `peft` 0.19, `pandas` 3.0,
`numpy` 2.5 and `requests` 2.34. A GPU is used automatically when available and is
strongly recommended for LoRA training. Loading a single 17Lands file takes a few GB of
RAM, and raw downloads are deleted right after the metrics are extracted.

```bash
pip install torch transformers peft pandas numpy requests
```

**Scripts have no command-line flags.** Each one is a `main()` function with keyword
arguments, run either as `python -m mtg_rating.<module>` (module defaults) or from a
`python -c` one-liner to change parameters. Everything is run from the repository root.

**1. Smoke test.** It downloads one set (ECL), trains a toy model per rating formula, and
checks inference on cards from other sets. It needs no checkpoint.

```bash
python -m mtg_rating.smoke_test
```

**2. Reproduce the full pipeline.** Data, caches and checkpoints are written under `data/`,
which is not tracked by git.

```bash
# (a) MTG vocabulary adaptation of MiniLM. Pass checkpoint_root explicitly: it does not
#     follow output_dir, and step (b) reads the epoch-4 snapshot from this folder.
python -c '
from mtg_rating.mlm_pretrain import main
main(
    lora_rank=64,
    output_dir="data/models/minilm_mtg_pretrained_rank64",
    checkpoint_every=4,
    checkpoint_root="data/models/minilm_mtg_pretrained_rank64_checkpoints",
)
'

# (b) Rating model: joint LoRA fine-tuning on the pooled multi-set dataset
python -c '
from mtg_rating.train_lora import main
main(
    formula="iih_only", extra_formulas=["gp_wr_only"],
    use_set_context=True,
    base_model_path="data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4",
    checkpoint_dir="data/models/lora_joint_dual_msh",  # where rate_set.py looks by default
)
'
```

**3. Rate a whole set.** Writes `data/ratings/ratings_<set>.html`: one table per color,
strongest to weakest, with the real 17Lands value alongside when one exists. It needs only
Scryfall data, so it works on a set with no draft history.

```bash
python -c 'from mtg_rating.rate_set import main; main("DFT")'
```

## Repository guide

| File | What it is |
|---|---|
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | Detailed walkthrough: every script and function, who calls what, and the pitfalls |
| [`CLAUDE.md`](CLAUDE.md) | Working notes for Claude Code, the AI coding assistant used on this project: a compact project-state cache, not user documentation |
| [`design_notes.md`](design_notes.md) | The original informal design notes and running observations |

Code layout under `mtg_rating/`, in the order data flows through it:

- **Data:** `fetch_17lands.py`, `fetch_scryfall.py`, `labels.py`, `multiset.py`
- **Representation:** `features.py`, `text_embeddings.py`, `ratings.py`, `set_context.py`, `color_context.py`
- **Models:** `lora_text_encoder.py`, `model_joint.py`, `model.py`, `mlm_pretrain.py`
- **Training and usage:** `train_lora.py` (main), `train_multiset.py` (frozen-embedding baseline), `train_color_context.py` (experimental), `rate_set.py`, `smoke_test.py`
- **Diagnostics:** `feature_search.py`, `opponent_benefit_probe.py`, `knn_baseline.py`

There is no automated test suite. Changes are verified by running an end-to-end training
and reading validation MSE and Pearson r per epoch.

## Open questions

- Fixing the GP WR under-dispersion, in particular teaching the encoder to price in
  restrictive effects rather than surface vocabulary.
- Rating cards *in the context of their set*: color and archetype strength, synergies.
- Secondary goals from the design notes: short natural-language card descriptions,
  metagame prediction, and using the ratings in a draft bot.

## Data sources and disclaimer

- [17Lands Public Datasets](https://www.17lands.com/public_datasets): draft game data
  (`game_data`, PremierDraft), used to compute the training targets. Please see 17Lands'
  own terms for use and attribution.
- [Scryfall API](https://scryfall.com/docs/api): card data. Requests are paced at
  100 ms and cached locally.

This is an unofficial fan project. It is not affiliated with or endorsed by 17Lands,
Scryfall or Wizards of the Coast. Magic: The Gathering is a trademark of Wizards of the
Coast LLC.
