# Getting started — MTG Card Rating

This document explains, very concretely, what each script in the repository does, how to run it, and how the functions/methods fit together. It is aimed at someone discovering the project who has read neither the code nor `CLAUDE.md`.

For the quick reference (architecture summary, current recommended config, established facts not to be questioned without new evidence), see `CLAUDE.md` at the repository root — this document goes further: one entry per function/method, with who calls it and what it calls. `design_notes.md` contains the original design notes (informal, kept as-is), referenced in places in the code ("design_notes.md section 5", etc.).

## Contents

1. [Repository overview](#1-repository-overview)
2. [Conventions to know before reading the code](#2-conventions-to-know-before-reading-the-code)
3. [Step-by-step walkthrough (commands)](#3-step-by-step-walkthrough-commands)
4. [Detailed reference, file by file](#4-detailed-reference-file-by-file)
   - [4.1 `fetch_17lands.py`](#41-fetch_17landspy)
   - [4.2 `fetch_scryfall.py`](#42-fetch_scryfallpy)
   - [4.3 `labels.py`](#43-labelspy)
   - [4.4 `features.py`](#44-featurespy)
   - [4.5 `text_embeddings.py`](#45-text_embeddingspy)
   - [4.6 `ratings.py`](#46-ratingspy)
   - [4.7 `multiset.py`](#47-multisetpy)
   - [4.8 `color_context.py`](#48-color_contextpy)
   - [4.9 `set_context.py`](#49-set_contextpy)
   - [4.10 `lora_text_encoder.py`](#410-lora_text_encoderpy)
   - [4.11 `model.py`](#411-modelpy)
   - [4.12 `model_joint.py`](#412-model_jointpy)
   - [4.13 `mlm_pretrain.py`](#413-mlm_pretrainpy)
   - [4.14 `train_multiset.py`](#414-train_multisetpy)
   - [4.15 `train_lora.py`](#415-train_lorapy)
   - [4.16 `train_color_context.py`](#416-train_color_contextpy)
   - [4.17 `rate_set.py`](#417-rate_setpy)
   - [4.18 `smoke_test.py`](#418-smoke_testpy)
   - [4.19 `feature_search.py`](#419-feature_searchpy)
   - [4.20 `opponent_benefit_probe.py`](#420-opponent_benefit_probepy)
   - [4.21 `knn_baseline.py`](#421-knn_baselinepy)

---

## 1. Repository overview

The project predicts a 0-10 rating for a Magic: The Gathering card (limited/draft) from its Scryfall characteristics (cost, type, text, rarity, power/toughness), trained on 17Lands' public win rates. The specific goal — and the constraint that shapes almost all of the code — is to **generalize to cards/sets that have not been released yet**: no feature may encode "which set" in a way that would make it unusable on a brand-new set.

The repository is organized in five layers, in the order the data flows through them:

1. **Data collection** (`fetch_17lands.py`, `fetch_scryfall.py`, `labels.py`): downloads and caches the 17Lands win rates and the Scryfall card characteristics, computes the per-card metrics.
2. **Representation** (`features.py`, `text_embeddings.py`, `ratings.py`, `color_context.py`): turns a card into a numeric vector, and a raw metric into a 0-10 rating.
3. **Models and multi-set aggregation** (`multiset.py`, `set_context.py`, `lora_text_encoder.py`, `model.py`, `model_joint.py`, `mlm_pretrain.py`): builds the pooled multi-set dataset, the context vectors, and the neural networks (including the LoRA-trainable text tower).
4. **Training and usage** (`train_multiset.py`, `train_lora.py`, `train_color_context.py`, `rate_set.py`, `smoke_test.py`): the scripts you actually run.
5. **Diagnostics** (`feature_search.py`, `opponent_benefit_probe.py`, `knn_baseline.py`): read-only analyses of the current checkpoint's errors — no training, nothing written.

There is no automated test suite (no `pytest`): verification is done by running a real training run and reading the per-epoch MSE/Pearson r on validation (never on test until the very end), and — for any mechanism that makes several cards interact with each other (trainable per-color context, DANN) — by explicitly re-checking with `split_mode="set"` (see §2).

**Warning**: unlike the `Coinche` repository, **no script here has CLI flags**. Every entry point is a `main()` function with many keyword arguments, whose defaults correspond either to a smoke test or to the currently recommended config. To change a parameter, call `main(...)` from a `python -c` one-liner or a small script, not from the command line.

### Dependency graph (who imports what)

```
fetch_17lands.py     (no internal dependency)
fetch_scryfall.py    (no internal dependency)
labels.py            (no internal dependency)
text_embeddings.py   (no internal dependency)
ratings.py           (no internal dependency)
color_context.py     (no internal dependency)
model.py             (no internal dependency)
        │
features.py  ────────────► text_embeddings.py
        │
multiset.py  ────────────► fetch_17lands.py, fetch_scryfall.py, labels.py
        │
set_context.py  ─────────► color_context.py, features.py, text_embeddings.py
        │
lora_text_encoder.py  ───► text_embeddings.py
        │
model_joint.py  ─────────► lora_text_encoder.py, text_embeddings.py
        │
mlm_pretrain.py  ────────► fetch_scryfall.py, text_embeddings.py
        │
train_multiset.py  ──────► features.py, fetch_scryfall.py, model.py, multiset.py, ratings.py
        │
train_lora.py  ──────────► color_context.py, features.py, model_joint.py, multiset.py,
        │                   ratings.py, set_context.py, text_embeddings.py, train_multiset.py (pearson)
        │
train_color_context.py ──► color_context.py, features.py, model_joint.py, multiset.py,
        │                   ratings.py, text_embeddings.py, train_lora.py (score_predictions,
        │                   split_by_name_3way, split_by_set_3way)
        │
rate_set.py  ────────────► features.py, fetch_scryfall.py, labels.py, model_joint.py,
        │                   multiset.py, ratings.py, set_context.py, text_embeddings.py
        │
smoke_test.py  ──────────► fetch_17lands.py, fetch_scryfall.py, features.py, labels.py,
                            model.py, ratings.py
        │
feature_search.py  ──────► features.py, multiset.py, rate_set.py (load_model), ratings.py,
        │                   set_context.py, train_lora.py (SPLIT_SEED, TEST_FRACTION,
        │                   VAL_FRACTION, iter_batches, split_by_name_3way), train_multiset.py (pearson)
        │
opponent_benefit_probe.py ► feature_search.py (_residuals, _WORD_RE, _PAREN_RE), features.py,
        │                   multiset.py, rate_set.py (load_model), ratings.py, set_context.py,
        │                   train_lora.py (SPLIT_SEED, TEST_FRACTION, VAL_FRACTION, split_by_name_3way)
        │
knn_baseline.py  ────────► features.py, multiset.py, rate_set.py (load_model), ratings.py,
                            set_context.py, train_lora.py (SPLIT_SEED, TEST_FRACTION,
                            VAL_FRACTION, iter_batches, split_by_name_3way), train_multiset.py (pearson)
```

---

## 2. Conventions to know before reading the code

- **17Lands metrics** (`labels.compute_card_metrics`): `gih_wr` (win rate when the card was seen in hand, drawn or in the opening hand), `gns_wr` (win rate when it was in the deck but never seen), `iih = gih_wr - gns_wr` (the card's intrinsic marginal impact, isolated from the strength of the rest of the deck — this is 17Lands' "IWD"), `gp_wr` (win rate over all games where the card was in the deck, seen or not — a more diluted/contextual signal than `gih_wr`), `play_rate` (fraction of decks where, when available in the pool, the card was actually maindecked).
- **Rating formulas** (`ratings.RAW_SCORE_FORMULAS`): combinations of these metrics (`gih_iih = gih_wr + iih`, `gp_iih = gp_wr + iih`, `iih_only`, `gp_wr_only`, etc.), all algebraically related via `gih_wr = gp_wr + (1-p)*iih`. They are then normalized to a 0-10 scale (`ratings.apply_normalization`: 5 = mean, 0/10 = mean ± 3 standard deviations, clamped).
- **Two completely different notions of "LoRA rank", never to be confused**: the rank 4 of `lora_text_encoder.LoraTextEncoder` (the LoRA of the rating task itself, trained in `train_lora.py`/`train_color_context.py`) and the rank 32/64 of `mlm_pretrain.py` (a different LoRA, on a vocabulary-adaptation step *upstream*, never used to predict a rating).
- **IIH vs GP WR**: IIH is the decontextualized signal (a function of the card's text alone, cleanly learnable, r≈0.6 on test); GP WR is the contextual signal (depends on the real draft environment, which does not exist yet for an unreleased set), harder (r≈0.48-0.5). Hence the dual objective (`extra_formulas=["gp_wr_only"]` in addition to `formula="iih_only"`) of the current config.
- **By-name vs by-set split** (`train_lora.split_by_name_3way`/`split_by_set_3way`): the split by card name (the default, comparable to almost all the results recorded in this project) is a reliable generalization test only for a mechanism with **no trainable parameters** (the fixed mean of `set_context.py`). Any **trainable** mechanism that makes several cards interact (`color_context.py`/`train_color_context.py`, DANN) must systematically be re-checked with `split_mode="set"` (whole sets held out) — gains that looked real by-name have already collapsed entirely under this test.
- **The by-name split depends on the whole pool of card names**: `split_by_name_3way` sorts every unique name, shuffles with the fixed seed and slices — so adding a set to `SET_CODES` re-deals *every* card between train/val/test, not just the new set's. Measured when MSH was added (26 → 27 sets): about 70% of the new val/test names had been in the existing checkpoint's *train* split. An older checkpoint must therefore be evaluated on exactly the pool it was trained on (for `lora_joint_dual`: every current set except MSH). Checkpoints record their `set_codes` and split settings in `head.pt` (since 2026-09-21; the two earlier ones were backfilled), and the diagnostics rebuild from them (`train_lora.resolve_training_sets`); a checkpoint without the record falls back to the current `SET_CODES` with a warning, and `set_codes=` overrides.
- **`records`, the pivot data structure**: `multiset.build_dataset()` returns a list of dicts `{**metrics_17lands, "name", "set_code", "scryfall_card"}`. `train_lora.py`/`train_color_context.py` enrich each record in place with `"structured"` (`features.structured_features`), `"oracle_text"`, and optionally `"set_context"` (fixed context vector) or `"_loss_weight"`. This dict, not a dedicated class, is what circulates everywhere.
- **Checkpoint format**: a folder with two elements — `lora_adapter/` (PEFT adapter saved by `model.save_pretrained`) and `head.pt` (a `torch.save` dict with `head_state_dict`, `mu`, `sigma`, `formula`, `base_model_path`, `use_set_context`, `extra_formulas`/`extra_mus`/`extra_sigmas`, and for `train_color_context.py` additionally `color_context_state_dict`/`context_dim`). `rate_set.load_model` can read both this current format (`extra_formulas` list) and the old single-`second_formula` format (the historical `lora_joint_dual` checkpoint).
- **Environment**: `env_coinche` (conda), shared with the `Coinche` repository — never `conda install`, always `pip install` so as not to break the conda solver (see project memory).

---

## 3. Step-by-step walkthrough (commands)

Everything is run from the repository root, with the `env_coinche` environment active (`conda run -n env_coinche python -m ...` or the environment already activated in the shell).

### 3.1 Smoke test (check that the whole chain runs)

```bash
python -m mtg_rating.smoke_test
```
Downloads one set (ECL), computes the metrics, trains a small toy MLP per rating formula, tests inference on 3 out-of-set cards. It does not aim for a good model — just to prove that fetch → features → train → predict does not crash. First reflex after a change touching `labels.py`/`features.py`/`ratings.py`.

### 3.2 MLM pretraining step (rarely needed — only if you change the corpus/objective)

```bash
python -m mtg_rating.mlm_pretrain
```
Adapts MiniLM to the MTG vocabulary via a rank-32 LoRA (careful: different from the rank 4 of the rating task) trained as a masked-LM on the ~30k Scryfall cards. Produces a standalone encoder base (`merge_and_unload()`), written to `output_dir` at the very end (best state according to the val loss, early stopping).

The checkpoint actually used downstream, `data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/`, is **not** that final output file (which exists only once, chosen by early stopping on the val loss) — it is a **periodic** snapshot taken at a fixed rhythm (`checkpoint_every`, unrelated to which epoch turns out to be the best), saved only if `checkpoint_every` is explicitly requested (default `0` = no intermediate snapshot). Reproducing this directory exactly therefore requires passing both `checkpoint_every` **and** `checkpoint_root` explicitly:
```bash
python -c '
from mtg_rating.mlm_pretrain import main
main(
    lora_rank=64,
    output_dir="data/models/minilm_mtg_pretrained_rank64",
    checkpoint_every=4,
    checkpoint_root="data/models/minilm_mtg_pretrained_rank64_checkpoints",
)
'
```
**Pitfall** (see §4.13): if `checkpoint_root` is *not* passed explicitly, it is **not** derived from the `output_dir` you just set on that same line — it falls back on the module constant `OUTPUT_DIR` (`data/models/minilm_mtg_pretrained`), hence on `data/models/minilm_mtg_pretrained_checkpoints`, a completely different folder. Omitting `checkpoint_root` here would silently send the intermediate snapshots to the wrong place.

### 3.3 Frozen-embedding baseline (for comparison, simpler than the main pipeline)

```bash
python -m mtg_rating.train_multiset
```
Pools the sets in `multiset.SET_CODES`, splits 80/20 by card name, trains a toy MLP (`model.CardRatingNet`) on MiniLM embeddings that are **never updated**, for each of the formulas in `ratings.RAW_SCORE_FORMULAS`. Caches the dataset with features already computed (`data/raw/multiset_dataset.npz`) — the most expensive part (the MiniLM pass) is therefore paid only once.

### 3.4 Main pipeline — joint LoRA fine-tuning (the production script)

```bash
python -m mtg_rating.train_lora
```
Runs the module's default config (`formula="gih_iih"`, no set context, no MLM-pretrained checkpoint) — and, since `save=True` by default, **silently overwrites** `data/models/lora_joint` (`CHECKPOINT_DIR`, `train_lora.py:35`) if that folder already contains a checkpoint you care about. To run the **currently recommended config** (`CLAUDE.md`), with the explicit `checkpoint_dir` needed to land in the right folder:
```bash
python -c '
from mtg_rating.train_lora import main
main(
    formula="iih_only", extra_formulas=["gp_wr_only"],
    use_set_context=True,
    base_model_path="data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4",
    checkpoint_dir="data/models/lora_joint_dual_msh",
)
'
```
**Pitfall**: without this explicit `checkpoint_dir`, `main()` saves by default into `data/models/lora_joint` (`CHECKPOINT_DIR`), **not** into `data/models/lora_joint_dual_msh` — the folder that `rate_set.py` loads by default (`rate_set.py:33`). Omitting this parameter would produce a correctly trained checkpoint that `python -m mtg_rating.rate_set` would never read (it would keep loading whatever is already in `lora_joint_dual_msh`, with no error or warning — just ratings that do not move after the new training).

Useful variants to know about (all disabled by default, none confirmed as a winner to date — see `CLAUDE.md` "Key established facts"):
```bash
# Honest check for any trainable mechanism that cuts across cards
python -c 'from mtg_rating.train_lora import main; main(split_mode="set")'

# Loss variants (none confirmed, to be tested one at a time, several seeds)
python -c 'from mtg_rating.train_lora import main; main(sample_weight_power=0.5)'
python -c 'from mtg_rating.train_lora import main; main(threshold_penalty_weight=1.0)'
python -c 'from mtg_rating.train_lora import main; main(loss_shape="saturating")'
python -c 'from mtg_rating.train_lora import main; main(contrastive_weight=0.1)'
python -c 'from mtg_rating.train_lora import main; main(triplet_weight=0.1)'  # the most recent, only smoke-tested
```

### 3.5 Experimental pipeline — trainable per-color context (not adopted, kept for reference)

```bash
python -c 'from mtg_rating.train_color_context import main; from mtg_rating.multiset import SET_CODES; main(SET_CODES)'
# honesty check (whole sets held out):
python -c 'from mtg_rating.train_color_context import main; from mtg_rating.multiset import SET_CODES; main(SET_CODES, split_mode="set")'
# with DANN (pushes the context not to encode the identity of the set):
python -c 'from mtg_rating.train_color_context import main; from mtg_rating.multiset import SET_CODES; main(SET_CODES, use_dann=True)'
```
Cross-card attention (`ColorAttentionContext`) over the commons/uncommons of the same primary color in the same set — this is precisely the mechanism that showed a by-name inflation that collapses by-set (`CLAUDE.md`), hence its "not adopted" status.

**Unlike `train_lora.main`, `save=False` by default here**: these three commands train and evaluate but persist nothing to disk — consistent with the "not adopted"/exploratory status of this pipeline, but worth knowing if you do want to keep a checkpoint: add `save=True` (and optionally `checkpoint_dir=...`, otherwise `data/models/color_context` by default), e.g. `main(SET_CODES, save=True)`.

### 3.6 Rate a whole set and generate an HTML page

```bash
python -m mtg_rating.rate_set                                              # rates DFT by default (main("DFT") at the bottom of the file)
python -c 'from mtg_rating.rate_set import main; main("MKM")'              # any other set
python -c 'from mtg_rating.rate_set import main; main("MKM", checkpoint_dir="data/models/lora_joint")'  # same set, single-formula checkpoint instead of the dual one
```
Loads the `lora_joint_dual_msh` checkpoint (the current baseline, trained on 27 sets) by default, rates every card of the given set, and writes `data/ratings/ratings_<set>.html` — one table per color group (each of WUBRG, plus separate multicolor and colorless sections), sorted from strongest to weakest by predicted GP WR. Requires **only** Scryfall data (no 17Lands labels) — so it works on a completely unreleased set.

### 3.7 Typical end-to-end pipeline

Simplified diagram (not copy-pastable commands — the complete commands, with the `checkpoint_every`/`checkpoint_root`/`checkpoint_dir` parameters needed to land in exactly these folders, are given in 3.2/3.4/3.6 above):

```
mlm_pretrain.py (rank 64)              → data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/
        │
        ▼
train_lora.py (iih_only + gp_wr_only,  → data/models/lora_joint_dual_msh/
  use_set_context=True, base_model_path=the checkpoint above)
        │
        ▼
rate_set.py main("<UPCOMING_SET>")     → data/ratings/ratings_<set>.html
```

### 3.8 Diagnostics on the current checkpoint (read-only)

```bash
python -m mtg_rating.feature_search            # which oracle-text words/bigrams correlate with the GP WR residual?
python -m mtg_rating.opponent_benefit_probe    # does the model overrate "gives something to the opponent" text?
python -m mtg_rating.knn_baseline              # does a nearest-comparable-card lookup beat the network?
```
Each loads a checkpoint (`checkpoint_dir`; default `data/models/lora_joint_dual_msh`), prints its tables and exits — no training, nothing written to disk. They rebuild the dataset from the sets the checkpoint recorded at training time, so their train/val/test split is the checkpoint's own. The two checkpoints that predate that record were backfilled after checking that each pool reproduces its recorded split row counts; a checkpoint with no record falls back to the current `SET_CODES` with a warning, and `set_codes=` overrides.

---

## 4. Detailed reference, file by file

### 4.1 `fetch_17lands.py`

Downloads the raw 17Lands files (`game_data`/`draft_data`), with a local disk cache — no internal dependency.

- `_download(kind, set_code, event_type, dest_dir)` (`fetch_17lands.py:11`): builds the S3 URL, downloads in streaming mode (1 MB chunks) if the file is not already cached, otherwise directly returns the existing path. Called by `download_game_data`/`download_draft_data`.
- `download_game_data(set_code, event_type="PremierDraft", dest_dir=None)` / `download_draft_data(...)` (`fetch_17lands.py:30`, `34`): public wrappers fixing `kind="game_data"`/`"draft_data"`. `download_game_data` is the only one of the two actually used elsewhere in the repository (`multiset.get_set_metrics`, `smoke_test.main`) — `download_draft_data` has no current caller in the code (planned for future use, cf. `design_notes.md`).

### 4.2 `fetch_scryfall.py`

Fetches card characteristics from the Scryfall API — two distinct uses: per set (for training/inference) and in bulk (for the MLM pretraining corpus).

- `fetch_set_cards(set_code)` (`fetch_scryfall.py:41`): paginates `cards/search?q=set:X` (~175 cards/page), respects Scryfall etiquette (`time.sleep(0.1)` between pages), caches the whole set as JSON. Called by `multiset.build_dataset`, `rate_set._usable_cards`, `smoke_test.main`.
- `_is_relevant(card)` (`fetch_scryfall.py:62`): true if the card is legal in at least one of the `RELEVANT_FORMATS` (`legacy`, `modern`, `standard`) — excludes Un-sets/cards banned everywhere except Vintage/Commander. Called by `fetch_bulk_oracle_cards`.
- `fetch_bulk_oracle_cards()` (`fetch_scryfall.py:67`): downloads Scryfall's `oracle_cards` bulk file (~200 MB, one entry per unique card across all printings), filters to the useful fields (`BULK_FIELDS`) and to relevant cards with text, deletes the raw file immediately after extraction (disk constraint), caches the filtered result. Called only by `mlm_pretrain.main` (MLM pretraining corpus — it is the only consumer of this function in the whole repository).
- `fetch_card(name)` (`fetch_scryfall.py:97`): one-off lookup of a card by fuzzy name (`fuzzy`), without caching. Called only by `train_multiset.main` and `smoke_test.main`, to test inference on "evergreen" cards guaranteed to be outside the training pool — `train_lora.py`/`train_color_context.py` do not have this built-in inference test and do not call it.

### 4.3 `labels.py`

Computes the per-card metrics from a raw `game_data` file — no internal dependency, all the work is done with `pandas`.

- `BASIC_LAND_NAMES` (`labels.py:49`): the 6 basic land names, excluded everywhere in the repository (draw dynamics not comparable to spells). Reused directly by `rate_set._usable_cards` (import of `labels.BASIC_LAND_NAMES`).
- `_is_basic_land_col(col)` (`labels.py:52`): recognizes a column such as `opening_hand_Plains`. Called by `compute_card_metrics` to exclude these columns right when reading the CSV (never even loaded into memory).
- **`compute_card_metrics(game_data_path, min_games=200, include_play_rate=False)`** (`labels.py:59`): reads only the necessary columns (`usecols`) with compact dtypes (nullable `Int8` — a `game_data` file loaded without these precautions caused an OOM on this machine on a larger set), computes per card `gih_wr`/`gns_wr`/`iih`/`gp_wr` and, if requested, `play_rate`/`pool_count` (more expensive: requires the `sideboard_*` columns and a deduplication by `(draft_id, build_index)`). Ignores any card with fewer than `min_games` games seen. Called by `multiset.get_set_metrics` (with `include_play_rate=True`) and by `smoke_test.main` (without, for the simplest smoke test).

### 4.4 `features.py`

Turns a Scryfall card dict into a fixed numeric vector. Depends on `text_embeddings.py`.

- `_parse_pt(value)` (`features.py:17`): converts a Scryfall power/toughness (which can be `"*"` or missing) into a float, `0.0` by default. Called by `structured_features`.
- **`structured_features(card)`** (`features.py:24`): 14-dim vector — mana cost, 5 one-hot colors, 5 one-hot types, rarity (0-3), power, toughness. Deliberately **without a "set" feature** (the whole point being to generalize to an unknown set). Called by `card_to_features`, `set_context.compute_set_context_vectors`/`compute_per_color_context_vectors`, `train_lora.main`/`train_color_context.main` (computed once per record and stored in `r["structured"]`), `rate_set.rate_cards`.
- **`card_to_features(card)`** (`features.py:43`): `structured_features(card)` + the frozen MiniLM embedding of the oracle text (`text_embeddings.embed_text`) — the complete vector used by the **frozen-embedding** pipeline (`model.py`/`train_multiset.py`/`smoke_test.py`). It is **not** used by `train_lora.py`/`train_color_context.py`, which re-encode the text on every batch through the trainable LoRA tower instead of this precomputed frozen embedding.
- `STRUCTURED_DIM`/`FEATURE_DIM` (`features.py:48-49`): derived sizes, reused as input dimensions in many places (`model_joint.CardRatingNetJoint`, `set_context.CONTEXT_DIM`, `train_color_context.py`).

### 4.5 `text_embeddings.py`

The **frozen** text encoder (never updated) — `sentence-transformers/all-MiniLM-L6-v2` through raw `transformers` (not the `sentence-transformers` wrapper, to avoid its scipy/scikit-learn/Pillow dependencies).

- `_load()` (`text_embeddings.py:21`): loads tokenizer + model only once (module-level cache `_tokenizer`/`_model`), puts the model in `eval()`. Called by `embed_texts` on every call (a no-op after the first thanks to the cache).
- **`embed_texts(texts)`** (`text_embeddings.py:30`): tokenizes a batch, runs it through the model without gradient (`torch.no_grad()`), does a mean-pooling weighted by the attention mask. Called by `embed_text`, `set_context.compute_set_context_vectors`/`compute_per_color_context_vectors`, `rate_set._context_vector_for_set`.
- `embed_text(text)` (`text_embeddings.py:43`): `embed_texts([text])[0]`, as a Python list. Called by `features.card_to_features`.
- `MODEL_NAME`/`EMBEDDING_DIM` (`text_embeddings.py:14-15`): constants reused throughout the repository — `MODEL_NAME` as the default value of `base_model_path` (`LoraTextEncoder`, `mlm_pretrain.py`), `EMBEDDING_DIM=384` as the expected dimension wherever a MiniLM embedding (frozen or LoRA) is concatenated with other features.

### 4.6 `ratings.py`

Converts a raw metric into a 0-10 rating — no internal dependency, pure statistical computation.

- `RAW_SCORE_FORMULAS` (`ratings.py:44`): dict `name → lambda(metrics) -> float`, the candidate formulas (`gih`, `gih_iih`, `gp_iih`, `gih_2iih`/`3iih`/`5iih`/`10iih`/`15iih`, `iih_only`, `gp_wr_only`, `play_rate_only`) — see §2 for their algebraic relationship. This is the dict that all the training scripts iterate over or index by formula name.
- `raw_scores(metrics, formula)` (`ratings.py:59`): applies a formula to each card of a `{name: metrics}` dict, skipping cards without `iih`. Called by `smoke_test.main` (the other scripts apply the formula directly on `records`, not through this function).
- `fit_normalization(raw)` / `apply_normalization(raw, mu, sigma)` (`ratings.py:64`, `69`): deliberately separate so that a train/test pipeline calibrates `(mu, sigma)` **only** on train and then applies the same transformation to the holdout, with no statistics leakage. Called together by `train_multiset.main`, `train_lora.main`, `train_color_context.main`, `rate_set.attach_actual_ratings`.
- `normalize_to_10(raw)` (`ratings.py:76`): fit+apply shortcut when there is no split (just `fit_normalization(raw)` then `apply_normalization`). Called by `compute_ratings`.
- `compute_ratings(metrics, formula)` (`ratings.py:80`): `raw_scores` + `normalize_to_10` chained. Called only by `smoke_test.main`.

### 4.7 `multiset.py`

Builds the pooled multi-set dataset — the junction point between 17Lands and Scryfall for the retained sets.

- `SET_CODES` (`multiset.py:23`): the list of the sets used for training (27 as of MSH's addition on 2026-08-20), verified directly against the public 17Lands S3 (not the site API); excludes as a matter of principle the Alchemy sets (digital-only) and "draft innovation" sets (extra products outside the Standard rotation). Reused by `train_lora.py`/`train_color_context.py` (default value of `set_codes`) and imported as-is in their `if __name__ == '__main__':` blocks.
- `_metrics_cache_path(set_code, event_type)` / `_load_cached_metrics(path)` / `_round_metric(v)` / `_save_metrics_cache(path, metrics)` (`multiset.py:33`, `37`, `46`, `55`): per-set CSV cache of the already-computed metrics — `_round_metric` rounds to 4 decimals (beyond that it is sampling noise, not signal). `_round_metric` is also reused directly by `train_multiset.save_pool_summary`.
- **`get_set_metrics(set_code, event_type="PremierDraft")`** (`multiset.py:64`): serves the CSV cache if it exists, otherwise downloads the `game_data` (`fetch_17lands.download_game_data`), computes the metrics (`labels.compute_card_metrics`, with `include_play_rate=True`), caches them, then **deletes the downloaded raw file** (disk constraint). Called by `build_dataset` and directly by `rate_set.attach_actual_ratings` (to compare a prediction with the real 17Lands result when it exists).
- **`build_dataset(set_codes=None, event_type="PremierDraft")`** (`multiset.py:80`): for each set, fetches the metrics (`get_set_metrics`) and the Scryfall cards (`fetch_scryfall.fetch_set_cards`), joins the two by name, silently skips (with a message) any set that fails. Returns the list of `records` (see §2) that then feeds all multi-set training. Called by `train_multiset.load_or_build_dataset`, `train_lora.main`, `train_color_context.main`.

### 4.8 `color_context.py`

Grouping of cards by "primary color" — used both by the fixed per-color mean (`set_context.compute_per_color_context_vectors`) and by the experimental trainable attention mechanism (`train_color_context.py`). No internal dependency.

- **`primary_color(scryfall_card)`** (`color_context.py:34`): first color in WUBRG order of `card["colors"]`, or `"colorless"`. A two-color card is therefore attached to a single group (its first color), not pooled over both — an accepted v1 limitation (`design_notes.md` §6 envisions a per-color-pair v2, never built). Called by `set_context.compute_per_color_context_vectors`, `build_color_buckets`; `rate_set.py` does **not** use it (its own `display_group` has different logic, see §4.17).
- **`build_color_buckets(records)`** (`color_context.py:42`): groups the indices of `records` by `(set_code, primary_color)`, with two sub-lists per bucket — `"query"` (all the cards of the bucket) and `"informant"` (the common/uncommon subset, the pool that informs the attention). Called only by `train_color_context.main`.
- **`flatten_buckets(color_buckets)`** (`color_context.py:62`): `{set_code: {color: bucket}}` → list of `(set_code, color, bucket)`, more convenient to iterate over/shuffle. Called only by `train_color_context.main`.

### 4.9 `set_context.py`

**Non-parametric** context vectors (no trainable weights) summarizing "what is in this set", concatenated to each card's own features. This is the **adopted** version (see `CLAUDE.md`), a simpler alternative to the trainable attention of `color_context.py`/`train_color_context.py`.

- `_mean_vector(vectors)` (`set_context.py:37`): component-wise mean of a list of vectors of the same dimension. Called by `compute_set_context_vectors`, `compute_per_color_context_vectors`, and reused directly by `rate_set._context_vector_for_set` (import of `set_context._mean_vector`).
- **`compute_set_context_vectors(records)`** (`set_context.py:43`): for each set present in `records`, the mean of `structured_features + frozen MiniLM embedding` over all its cards. Cached in `data/raw/set_context_vectors.json`, **keyed by file existence, not by content** — regenerating with a different subset of sets leaves a silently stale cache (the file has to be deleted by hand). Called by `train_lora.main` (if `use_set_context=True`) and read directly by `rate_set._context_vector_for_set` for the sets already pooled.
- **`compute_per_color_context_vectors(records)`** (`set_context.py:64`): same idea as above but grouped by `(set_code, primary_color)` (via `color_context.primary_color`) instead of by whole set — a more targeted summary ("the average white card of this set" rather than "the average card, all colors combined"), still without a trainable parameter and thus without the per-set generalization problem of trainable mechanisms. Cached in `set_context_vectors_per_color.json`, same cache pitfall. Tested but **not adopted** (no net gain vs the whole-set version, `CLAUDE.md`). Called only by `train_lora.main` (if `use_color_context=True` — mutually exclusive with `use_set_context`).
- `CACHE_PATH`/`CACHE_PATH_PER_COLOR`/`CONTEXT_DIM` (`set_context.py:32-34`): cache paths and dimension of the context vector (= `STRUCTURED_DIM + EMBEDDING_DIM`). `CACHE_PATH` is also imported directly by `rate_set.py` (under the alias `SET_CONTEXT_CACHE_PATH`), `CONTEXT_DIM` by `train_lora.py`/`rate_set.py`.

### 4.10 `lora_text_encoder.py`

The **trainable** text tower of the main pipeline — the same MiniLM base as `text_embeddings.py`, but wrapped with a LoRA adapter (rank 4, on the `query`/`value` projections) updated during training.

#### `LoraTextEncoder(nn.Module)` (`lora_text_encoder.py:23`)
- `__init__(rank=LORA_RANK, dropout=0.0, base_model_path=MODEL_NAME)` (`lora_text_encoder.py:24`): loads the base tokenizer/model (generic by default, or a locally MLM-pretrained base — see `mlm_pretrain.py`), wraps it with a `LoraConfig` (`peft.get_peft_model`) — this constructor is what decides, via `base_model_path`, whether to start from generic MiniLM or from the MTG-adapted version.
- `print_trainable_parameters()` / `trainable_parameters()` (`lora_text_encoder.py:41`, `44`): delegates to the built-in `peft` diagnostic / returns the list of parameters with `requires_grad=True` (the LoRA matrices only, never the base). `trainable_parameters()` is called by `model_joint.CardRatingNetJoint.trainable_parameters` and by the differentiated parameter groups of `train_lora.main`/`train_color_context.main` (lower LR than the head).
- `forward(texts)` (`lora_text_encoder.py:47`): tokenizes, runs through the model **with** gradient (unlike `text_embeddings.embed_texts`), identical mean-pooling. Called by `model_joint.CardRatingNetJoint.forward`/`ColorAttentionContext`-adjacent code in `train_color_context.py` (`model.text_encoder(...)`), and directly in `train_lora.py` for the auxiliary passes (`contrastive_loss`, `mine_hard_triplets`) that need the raw embedding, not just the final prediction.

### 4.11 `model.py`

The **simplest** pipeline in the repository: a small MLP on already-frozen embeddings — used only by `train_multiset.py`/`smoke_test.py`, never by the main LoRA pipeline.

#### `CardRatingNet(nn.Module)` (`model.py:7`)
`__init__(input_dim, hidden_dim=16)`: one ReLU hidden layer, one scalar output. `forward(x)`: forward pass + `squeeze(-1)`.

- **`train_toy(features, labels, epochs=50, lr=1e-2)`** (`model.py:20`): a complete training loop in one function — builds the model, `Adam`, `MSELoss`, full-batch loop (no mini-batches) over `epochs` iterations. Returns the trained model. Called by `smoke_test.main` and `train_multiset.main` — it is the only training function in the whole repository that has no built-in early stopping/validation logic (both callers handle train/test themselves around it).

### 4.12 `model_joint.py`

The model classes of the main pipeline (joint LoRA) and of the experimental pipeline (per-color attention + DANN).

#### `CardRatingNetJoint(nn.Module)` (`model_joint.py:16`) — the central model
- `__init__(structured_dim, hidden_dims=None, lora_rank=LORA_RANK, lora_dropout=0.0, head_dropout=0.0, base_model_path=MODEL_NAME, set_context_dim=0, output_dim=1, layer_norm=False)` (`model_joint.py:17`): builds a `LoraTextEncoder` (the text tower) then an MLP head (`hidden_dims` defaults to `[16]`, GELU + dropout per layer, optional Pre-LN LayerNorm — disabled by default to exactly reproduce the old one-layer head) taking `structured_dim + EMBEDDING_DIM + set_context_dim` as input. `output_dim > 1` serves the multi-target mode (`extra_formulas`).
- `forward(structured, texts, set_context=None)` (`model_joint.py:59`): encodes the texts (`self.text_encoder`), concatenates `[structured, text_embedding, set_context?]`, passes through the head. `squeeze(-1)` if `output_dim == 1` so that nothing downstream has to handle a special case. Called by `train_lora.evaluate`/the training loop of `train_lora.main`, and by `rate_set.rate_cards`.
- `trainable_parameters()` (`model_joint.py:71`): LoRA parameters of `text_encoder` + all the head's parameters, in a single list. No current caller in the repository: `train_lora.main`/`train_color_context.main` build their own differentiated LR groups by hand (`model.text_encoder.trainable_parameters()` and `model.head.parameters()` separately, to give them different LRs) rather than using this shortcut that merges them.

#### `ColorAttentionContext(nn.Module)` (`model_joint.py:75`) — experimental, not adopted
- `__init__(input_dim, context_dim=32, num_heads=4, dropout=0.0)` (`model_joint.py:87`): a linear projection to `context_dim` + an `nn.MultiheadAttention` layer. Deliberately small (little capacity), by a documented design choice (limiting overfitting on the ~156 set×color buckets available).
- `forward(combined, informant_local_idx)` (`model_joint.py:93`): `combined` = all the cards of a bucket (queries), `informant_local_idx` = indices of the commons/uncommons (keys/values). Attention + residual. Called only by `train_color_context._bucket_forward`.
- `trainable_parameters()` (`model_joint.py:103`): all the module's parameters (nothing is frozen here, unlike the text tower). Used by `train_color_context.main` for its own LR group.

#### `_GradientReversalFunction(torch.autograd.Function)` / `GradientReversalLayer(nn.Module)` / `DomainClassifier(nn.Module)` (`model_joint.py:107`, `118`, `135`) — DANN support, experimental
- `_GradientReversalFunction.forward`/`backward` (`model_joint.py:108`, `113`): identity on the forward pass, reversed gradient (× `-lambda_`) on the backward pass — the raw mechanism behind `GradientReversalLayer.forward` (`model_joint.py:131`), which wraps it in an `nn.Module` so it can be stacked in a model normally. `lambda_` is meant to be ramped from 0 to 1 during training (done by `train_color_context.main`, not by the class itself).
- `DomainClassifier.__init__(context_dim, num_domains)`/`forward(context)` (`model_joint.py:141`, `149`): a small MLP classifying which set a context vector comes from — placed right after a `GradientReversalLayer` in `train_color_context.py`, so that its own weights learn normally to classify well while the upstream encoder is pushed in the opposite direction (no longer encoding the identity of the set). Called only in `train_color_context.run_epoch`/`main` (`use_dann=True`).

### 4.13 `mlm_pretrain.py`

**Upstream** pretraining step, separate from the rating fine-tuning — adapts MiniLM to the MTG vocabulary by masked-LM over the whole Scryfall corpus (~30k cards), not over the ~5-6k rows labeled by 17Lands.

- `split_texts(texts, val_fraction, seed)` (`mlm_pretrain.py:59`): simple train/val split by shuffle+cut (not by card name as elsewhere — here the unit is the text, not a 17Lands metric that must not leak). Called by `main`.
- **`build_model(tokenizer, full_finetune=False, lora_rank=LORA_RANK)`** (`mlm_pretrain.py:66`): `full_finetune=True` returns the raw `AutoModelForMaskedLM` model, all parameters trainable (a comparison point only, never adopted by default — documented risk of overfitting/catastrophic forgetting); otherwise wraps it with a LoRA (`target_modules=["query","value"]`, `modules_to_save=["cls"]` — the MLM prediction head does not exist in the base `sentence-transformers` checkpoint and must therefore be trained in full, not just adapted). Called by `main`.
- `iter_batches(texts, batch_size)` (`mlm_pretrain.py:95`): slicing into chunks, identical in spirit to `train_lora.iter_batches` (duplicated, not shared). Called by `run_epoch`.
- **`run_epoch(model, tokenizer, collator, texts, batch_size, train, optimizer=None)`** (`mlm_pretrain.py:100`): one pass (train or eval) — randomly masks tokens (`DataCollatorForLanguageModeling`), computes the MLM loss, updates if `train=True`. Returns the mean loss weighted by number of masked tokens. Called twice per epoch in `main` (once train, once val).
- `save_pretrained_checkpoint(model, tokenizer, output_dir, full_finetune=False)` (`mlm_pretrain.py:122`): saves an intermediate checkpoint — works on a `deepcopy` so as not to disturb the model being trained, merges the LoRA (`merge_and_unload()`) except in `full_finetune` (no peft wrapper to merge). Called by `main` if `checkpoint_every` is positive, with `checkpoint_root / f"epoch{epoch+1}"` as `output_dir` — hence **not** the same folder as `main`'s final `output_dir`.
- **`main(seed=0, epochs=EPOCHS, checkpoint_every=0, checkpoint_root=None, full_finetune=False, output_dir=None, lora_rank=LORA_RANK)`** (`mlm_pretrain.py:135`): downloads the corpus (`fetch_scryfall.fetch_bulk_oracle_cards`), splits, builds the model (`build_model`), loops with early stopping on the val loss (patience `PATIENCE=5`), saves the best state, merges and saves the final encoder base in `output_dir`. This output file is what then becomes the `base_model_path` of a fresh `LoraTextEncoder` in `train_lora.py`.
  **Pitfall** (`mlm_pretrain.py:145-147`): `output_dir = output_dir or OUTPUT_DIR` does compute the local value from the passed parameter, but the line right after, `checkpoint_root = checkpoint_root or OUTPUT_DIR.parent / "..."`, falls back on the **module constant** `OUTPUT_DIR` (`data/models/minilm_mtg_pretrained`) if `checkpoint_root` is not provided — **not** on the value of `output_dir` you just set on the line above. Passing a custom `output_dir` without passing `checkpoint_root` at the same time therefore sends the intermediate snapshots (`checkpoint_every`) to a folder based on the old default name, not next to the new `output_dir`. The checkpoint actually used downstream (`data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/`, `checkpoint_every=4` judging by the snapshots present on disk — epoch4/8/12/16/20) could therefore only have been produced by passing `checkpoint_root` explicitly, see §3.2.

### 4.14 `train_multiset.py`

The "frozen embeddings" baseline with a real held-out evaluation (unlike the smoke test) — serves as a comparison point for the main LoRA pipeline.

- `split_by_name(records, test_fraction, seed)` (`train_multiset.py:37`): 2-way split (train/test) by unique card name. Called by `main`. (Note: `train_lora.py` has its own 3-way version, `split_by_name_3way` — not the same function, no code shared between the two.)
- **`pearson(a, b)`** (`train_multiset.py:45`): Pearson correlation via `numpy.corrcoef`. **Reused directly** (import) by `train_lora.score_predictions` — it is the only symbol that `train_lora.py` imports from this module, hence the `train_multiset.py (pearson)` entry in the dependency graph of §1.
- `save_pool_summary(records, path)` (`train_multiset.py:49`): lightweight CSV dump (without the full Scryfall blobs) of the pooled dataset, for on-disk inspection independent of stdout buffering. Called by `load_or_build_dataset`.
- **`load_or_build_dataset()`** (`train_multiset.py:72`): serves the `.npz` cache if it exists (features already computed — the most expensive step), otherwise calls `multiset.build_dataset()`, computes `card_to_features` for each record, and caches. Called only by `main`.
- **`main()`** (`train_multiset.py:112`): loads the dataset, splits 80/20 by name, then for **each formula** of `ratings.RAW_SCORE_FORMULAS`: calibrates `(mu, sigma)` on train only, trains (`model.train_toy`, 200 epochs, no early stopping), evaluates MSE + Pearson r on test, tests inference on 3 out-of-pool evergreen cards.

### 4.15 `train_lora.py`

**The main script of the repository.** Jointly fine-tunes the text tower (LoRA rank 4) and the rating head on the pooled multi-set dataset — the biggest file (632 lines), with several optional loss variants never confirmed as winners.

- `_as_tuple(value)` (`train_lora.py:67`): normalizes a target (single float or multi-formula tuple) into a tuple — an internal utility so the target-building code does not have to distinguish the two cases. Called in `main` (building `train_targets`/`val_targets`/`test_targets`) and by `mine_hard_triplets`.
- `iter_batches(pairs, batch_size)` (`train_lora.py:71`): slicing into chunks of a list of `(record, target)` pairs. Called by `evaluate` and `main`.
- **`split_by_name_3way(records, val_fraction, test_fraction, seed)`** (`train_lora.py:76`): train/val/test split by unique card name — **the reference split of the whole project** (`SPLIT_SEED=42` fixed everywhere). Called by `main` (`split_mode="name"`, the default) and imported by `train_color_context.py`.
- **`split_by_set_3way(records, val_fraction, test_fraction, seed)`** (`train_lora.py:88`): the same thing but at the level of the whole set (not the card name) — **the honesty test** for any trainable mechanism that cuts across cards (see §2). Called by `main` (`split_mode="set"`) and by `train_color_context.py`.
- `_set_context_tensor(batch_records)` (`train_lora.py:106`): returns `None` if the records do not carry a `"set_context"` key (no-context mode), otherwise stacks the vectors into a tensor. Called by `evaluate` and the training loop of `main`.
- **`score_predictions(preds, targets)`** (`train_lora.py:115`): computes MSE (+ Pearson r if single target, or a per-component breakdown if multi-target/`extra_formulas`) — logic **shared** with `train_color_context.evaluate` (direct import), so that the two training loops report their results identically. Called by `evaluate` (here) and `train_color_context.evaluate`.
- **`weighted_mse_loss(pred, target, weight, threshold_penalty_weight=0.0, threshold=1.0, loss_shape="mse", saturating_asymptote=8.0, saturating_scale=3.0)`** (`train_lora.py:142`): the loss function — with the default values, strictly equivalent to `nn.MSELoss()`. `loss_shape="saturating"` replaces the squared error with a Geman-McClure-type version that saturates at a fixed asymptote for large errors instead of growing without bound (parameters `A=8, B=3` derived algebraically from explicit constraints on the inflection point). `threshold_penalty_weight` adds a separate squared-hinge term beyond an error threshold — the two mechanisms are not meant to be combined (the second is a cruder attempt at the same objective that the first supersedes). Called in the training loop of `main`.
- **`contrastive_loss(text_embedding, target_component, tau)`** (`train_lora.py:195`): auxiliary loss in the embedding space — pulls two cards together in proportion to how close their real targets are (`exp(-|gap|/tau)`), independently of the surface similarity of the text. Directly motivated by the "Annul probe" (see `CLAUDE.md`). Called in the training loop of `main` if `contrastive_weight > 0`.
- **`mine_hard_triplets(model, records, targets, threshold)`** (`train_lora.py:224`, `@torch.no_grad()`): re-embeds the whole train set (not just a batch) to find, for each card, its hard negative (nearest embedding among the cards whose real target differs by more than `threshold`) and its positive (closest real target, no embedding search needed). Expensive (full pass) — called periodically (`triplet_mining_every`, every epoch by default), not on every batch. Called by `main`.
- **`triplet_loss(anchor, positive, negative, margin)`** (`train_lora.py:278`): standard triplet loss in cosine distance, consistent with the similarity metric used by `mine_hard_triplets` to choose the negative. Called in the training loop of `main` if `triplet_weight > 0` and triplets have already been mined.
- **`evaluate(model, records, targets)`** (`train_lora.py:293`): a complete (batched) eval pass, returns `(mse, extra, preds)` via `score_predictions`. Called by `main` at every epoch (on val) and once at the very end (on test).
- **`save_checkpoint(model, mu, sigma, formula, checkpoint_dir=None, base_model_path=MODEL_NAME, use_set_context=False, use_color_context=False, extra_formulas=None, extra_mus=None, extra_sigmas=None, set_codes=None, split_mode=None, split_seed=None, val_fraction=None, test_fraction=None)`** (`train_lora.py:307`): saves the LoRA adapter (`save_pretrained`, a few tens of KB) + a `head.pt` dict with all the context needed to reload the model correctly later (see §2, "Checkpoint format"). It also records the training pool and split (`set_codes`, `split_mode`, `split_seed`, `val_fraction`, `test_fraction`) so a later evaluation can rebuild the checkpoint's own split — see `resolve_training_sets`. `checkpoint_dir=None` falls back on the module constant `CHECKPOINT_DIR` = `data/models/lora_joint` (`train_lora.py:35`) — **not** `lora_joint_dual`, the folder that `rate_set.py` loads by default (`rate_set.CHECKPOINT_DIR`, different). Called by `main` if `save=True` (the default); see §3.4 for the recommended-config example with the explicit `checkpoint_dir` that avoids this pitfall.
- **`main(...)`** (`train_lora.py:367`): the core of the file — builds the pooled dataset, adds `structured`/`oracle_text` (and `set_context` if requested) to each record, filters out records without a value for the requested formulas, splits (`split_by_name_3way` or `split_by_set_3way`), calibrates `(mu, sigma)` per formula (primary + `extra_formulas`, each independently, on train only), builds `CardRatingNetJoint` with two LR groups (`lora_lr` for the text tower, `head_lr` for the head), loops over `epochs` with (optionally) weighting by number of games (`sample_weight_power`), saturating loss, threshold penalty, contrastive loss, triplet loss — evaluates on val at each epoch, early stops (patience `PATIENCE=8`), **touches the test set exactly once** at the very end. Returns `(model, preds, test_records, test_targets, best_val_mse)` — `best_val_mse` explicitly so that a multi-seed sweep selects the best run by **val** score, never by test score.
- `load_training_pool(checkpoint_dir)` (`train_lora.py:650`): reads what `save_checkpoint` recorded about the training pool (`set_codes`, `split_mode`, `split_seed`, `val_fraction`, `test_fraction`) out of `head.pt`; returns an empty dict for checkpoints saved before this was tracked. Called by `resolve_training_sets`.
- **`resolve_training_sets(explicit, checkpoint_dir)`** (`train_lora.py:660`): the set list a diagnostic should rebuild the dataset from, so that its by-name split is the checkpoint's own. Priority: the `explicit` list > the checkpoint's recorded `set_codes` (with a note if the pool has grown since) > `None`, meaning the current `SET_CODES`, with a loud warning that the split may not be the checkpoint's own. Also warns if the recorded seed/fractions differ from the current constants, or if the checkpoint was trained with `split_mode="set"`. Called by `feature_search.main`, `opponent_benefit_probe.main` and `knn_baseline.main`.

### 4.16 `train_color_context.py`

Experimental pipeline — trainable attention context per (set, primary color) bucket, with DANN support. **Not adopted** in the current config, kept for reference/possible resumption.

- **`_bucket_forward(model, color_context, records, bucket, targets_by_index, device)`** (`train_color_context.py:58`): one forward pass for a single bucket — encodes all the cards of the bucket (queries), runs the attention (`color_context`) with the common/uncommon subset as keys/values, computes the head's prediction for all the cards then keeps only those having a target in `targets_by_index`. Returns `None` if no card of the bucket has a target (bucket entirely val/test during a train pass, etc.). No backward here. Called by `run_epoch` and `evaluate`.
- **`run_epoch(model, color_context, records, buckets, targets_by_index, device, optimizer, rng, dann=None, dann_chunk_size=1)`** (`train_color_context.py:108`): shuffles the order of the buckets, groups `dann_chunk_size` buckets per optimization step (1 = historical behavior, bit-for-bit identical without DANN), aggregates the MSE loss over the group, adds the DANN adversarial loss if enabled (needs several buckets per step because a single bucket has only one domain class — degeneracy documented in the docstring), does a single `backward()`/`step()` per group. Called by `main` at every epoch.
- **`evaluate(model, color_context, records, buckets, targets_by_index, device)`** (`train_color_context.py:176`): a complete pass without gradient over all the buckets, aggregates all the predictions, calls `train_lora.score_predictions` for the final computation. Called by `main` (val at every epoch, test at the end).
- `save_checkpoint(model, color_context, mu, sigma, formula, checkpoint_dir=None, base_model_path=MODEL_NAME)` (`train_color_context.py:192`): like `train_lora.save_checkpoint`, plus `color_context_state_dict`/`context_dim` (the attention module does not exist in the main pipeline). Called by `main` if `save=True`.
- **`main(...)`** (`train_color_context.py:211`): builds the dataset, computes the buckets (`color_context.build_color_buckets` + `flatten_buckets`), splits (`split_by_name_3way` or `split_by_set_3way`, imported from `train_lora.py`), builds `CardRatingNetJoint` + `ColorAttentionContext` with 3 (or 4 with DANN) separate LR groups, loops with the classic DANN ramp (`2/(1+e^{-10·progress}) - 1`, Ganin & Lempitsky) if `use_dann=True`, early stopping identical to `train_lora.main`, single test at the end. Unlike `train_lora.main`, `save` is `False` by default here — a call without an explicit `save=True` therefore writes nothing to disk (consistent with the experimental/not-adopted status of this pipeline, see §3.5).

### 4.17 `rate_set.py`

Inference + display only, **no training**. Loads a saved checkpoint, rates all the cards of a set, produces an HTML page.

- `display_group(card)` (`rate_set.py:43`): **display** grouping (deliberately different from `color_context.primary_color`, which is not used here) — mono-color goes under its color, multicolor and colorless each have their own section (instead of being attached to a first color, more intuitive for a human reader). Called by `main`.
- `_usable_cards(set_code)` (`rate_set.py:71`): cards of the set (`fetch_scryfall.fetch_set_cards`) minus basic lands and cards without a recognized rarity. Called by `main`.
- **`_context_vector_for_set(set_code, cards)`** (`rate_set.py:76`): reuses the shared cache of `set_context.py` if `set_code` is one of the sets already pooled in `SET_CODES` (the same whole-set mean seen during training, for sets the checkpoint was trained on); otherwise computes the same mean on the fly, **without ever writing** to this shared cache (a partial write of a single set would silently corrupt it for all the other scripts that trust it). Called by `main`.
- **`load_model(checkpoint_dir=CHECKPOINT_DIR)`** (`rate_set.py:94`): loads `head.pt`, detects the format (current `extra_formulas` list, or old single `second_formula`), rebuilds `CardRatingNetJoint` with the right dimensions, reloads the LoRA adapter onto a fresh base (`PeftModel.from_pretrained` on a new `AutoModel`, rather than unwrapping the random adapter already built by `CardRatingNetJoint.__init__` — avoids depending on `peft` internals). Returns `(model, formulas, use_set_context, norm_params)`. Called by `main`.
- **`rate_cards(model, formulas, cards, context_vector=None)`** (`rate_set.py:133`, `@torch.no_grad()`): batches by `BATCH_SIZE=32`, calls `model(...)`, associates each formula with its prediction per card. Called by `main`.
- **`attach_actual_ratings(ratings, set_code, formulas, norm_params)`** (`rate_set.py:149`): mutates `ratings` in place — adds `f"{formula}_actual"` (the real 17Lands result, on the same 0-10 scale) for every card where `get_set_metrics` has data, silently leaves it absent otherwise (unreleased set or one never covered by 17Lands). Called by `main`.
- **`render_html(set_code, by_color, formulas, primary_metric)`** (`rate_set.py:178`): builds the complete HTML page (inline CSS, one section per color, sorted table, color average in the table footer, rarity color). Called by `main`.
- **`main(set_code, checkpoint_dir=CHECKPOINT_DIR, out_path=None)`** (`rate_set.py:263`): chains all the functions above, sorts each color group from strongest to weakest by the primary metric (`gp_wr_only` if present, otherwise the first formula), writes `data/ratings/ratings_<set>.html`. The `if __name__ == '__main__': main("DFT")` at the bottom of the file explains why `CLAUDE.md` says to edit that line to change the default set when launching with `-m`.

### 4.18 `smoke_test.py`

End-to-end smoke test — the fastest command to check that the whole chain (excluding LoRA, excluding multi-set) still works after a change.

- **`main()`** (`smoke_test.py:27`): downloads the `game_data` of a single set (`ECL`), computes the metrics (`labels.compute_card_metrics`), fetches the corresponding Scryfall cards, joins the two, computes the features (`features.card_to_features`) for the set's cards **and** for 3 out-of-set evergreen cards (`UNSEEN_TEST_CARDS`), then for **each formula** of `ratings.RAW_SCORE_FORMULAS`: computes the ratings (`ratings.compute_ratings`), trains a toy model (`model.train_toy`), prints the prediction on the 3 unseen cards. Does neither a train/test split nor early stopping — the goal is only to prove that the pipeline runs without an exception, not to measure real generalization (unlike `train_multiset.py`/`train_lora.py`).

### 4.19 `feature_search.py`

Read-only diagnostic: which words/bigrams of a card's oracle text are correlated with the checkpoint's GP WR prediction error? The automatic candidate-search half of the "validated feature engineering" idea in `CLAUDE.md` — it ranks candidates and adds nothing to the model.

- `_tokenize(oracle_text)` (`feature_search.py:51`): lowercases, strips parenthesized reminder text (`_PAREN_RE`), and returns the set of single words plus adjacent-word bigrams. Called by `_score_patterns`.
- `_set_context_tensor(batch)` (`feature_search.py:57`): stacks the batch's `"set_context"` vectors, or returns `None` if the records carry none. A local copy of `train_lora._set_context_tensor` (which also moves the tensor to the training device). Called by `_residuals`.
- **`_residuals(model, target_idx, mu, sigma, records)`** (`feature_search.py:63`, `@torch.no_grad()`): predicts every record and returns `actual − predicted` on the model's own 0-10 scale, the actual value being normalized with the checkpoint's own `mu`/`sigma`. Called by `main` and imported by `opponent_benefit_probe`.
- **`_score_patterns(train_records, train_residuals, val_records, val_residuals)`** (`feature_search.py:78`): candidate patterns are the tokens with at least `MIN_TRAIN_SUPPORT` (30) train cards and `MIN_VAL_SUPPORT` (8) val cards. For each, the Pearson r (`train_multiset.pearson`) between its 0/1 indicator and the residual, on train and on val. Rows come back sorted by `|train_r|`. Called by `main`.
- **`main(top_n=TOP_N, checkpoint_dir=CHECKPOINT_DIR, set_codes=None)`** (`feature_search.py:100`): loads the checkpoint (`rate_set.load_model(checkpoint_dir)`, which must contain `gp_wr_only`), builds the dataset from `resolve_training_sets(set_codes, checkpoint_dir)`, adds `structured`/`oracle_text`/`set_context`, keeps the records with a value for every formula the checkpoint outputs (the same filter as `train_lora.main`, so the split matches), splits by name (`split_by_name_3way`, `SPLIT_SEED`), computes residuals on train and val only (test untouched), and prints the top `top_n` patterns with a `confirmed` flag: same sign on train and val and `|val_r| >= CONFIRM_R` (0.05).

### 4.20 `opponent_benefit_probe.py`

Read-only diagnostic for one specific hypothesis, motivated by The Sentry, Golden Guardian (reads as strong text but gives the *opponent* a 5/5 flying indestructible token): the model may not track *who* an effect benefits, and so overrate cards where a "grant" verb follows the word "opponent".

- `BENEFIT_VERBS` / `HARM_VERBS` (`opponent_benefit_probe.py:38-39`): the verb lists used to classify text.
- `_classify(oracle_text)` (`opponent_benefit_probe.py:42`): looks at the 3 words following each "opponent"/"opponents" and returns `"benefit"` (a benefit verb, which takes priority), `"harm"`, `"opponent_other"` (the word appears with neither) or `"none"`. Purely regex-based: it ignores grammatical subject and negation, which is a known source of false positives. Called by `main`.
- **`main(checkpoint_dir=CHECKPOINT_DIR, set_codes=None)`** (`opponent_benefit_probe.py:62`): same checkpoint/dataset/split setup as `feature_search.main`, on the pooled train+val cards (test untouched). Computes residuals with `feature_search._residuals`, prints per-bucket count, mean residual and standard deviation, then lists the 10 cards of the `"benefit"` bucket with the most negative residual (name, residual, set).

### 4.21 `knn_baseline.py`

Read-only diagnostic: does an explicit "find the closest comparable card" lookup predict as well as the network's own head? It uses the same input space as the trained model (z-scored structured features + the checkpoint's own fine-tuned LoRA text embedding).

- `_text_embeddings(model, records)` (`knn_baseline.py:48`, `@torch.no_grad()`): the checkpoint's `text_encoder` output for every record, batched and L2-normalized. Called by `main`.
- `_nn_predictions(model, formulas, use_set_context, records)` (`knn_baseline.py:57`, `@torch.no_grad()`): the network's own predictions per formula — the reference the k-NN is compared against. Called by `main`.
- `_actual_ratings(records, formula, mu, sigma)` (`knn_baseline.py:69`): the real metric of each record on the 0-10 scale. Called by `main`.
- **`_knn_predict(train_structured, train_text, train_targets, query_structured, query_text, k, structured_weight)`** (`knn_baseline.py:73`): concatenates `structured * structured_weight` with the text embedding, takes Euclidean distances (`torch.cdist`) to every train card, and returns the plain (unweighted) mean target of the `k` nearest. Called by `main`.
- **`main(checkpoint_dir=CHECKPOINT_DIR, set_codes=None)`** (`knn_baseline.py:82`): loads the checkpoint (needs both `gp_wr_only` and `iih_only`), builds the dataset from `resolve_training_sets(set_codes, checkpoint_dir)`, splits by name, prints the network's own val MSE and r as the reference, sweeps `K_VALUES` (3, 5, 10, 20, 50, 100, 200) × `STRUCTURED_WEIGHTS` (1, 3, 10) on val and prints the best cell by combined IIH+GP val MSE, then blends the network's and the k-NN's predictions at that best cell for each weight in `BLEND_WEIGHTS` (0 to 0.5) to see whether the k-NN adds anything the network lacks. Test untouched.
