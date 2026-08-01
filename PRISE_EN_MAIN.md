# Prise en main — MTG Card Rating

Ce document explique, très concrètement, ce que fait chaque script du dépôt, comment les lancer, et comment les fonctions/méthodes s'articulent entre elles. Il s'adresse à quelqu'un qui découvre le projet et n'a lu ni le code ni `CLAUDE.md`.

Pour la référence rapide (résumé d'architecture, config actuelle recommandée, faits établis à ne pas remettre en cause sans preuve nouvelle), voir `CLAUDE.md` à la racine — ce document-ci va plus loin : une entrée par fonction/méthode, avec qui l'appelle et qui elle appelle. `premier_jet.md` contient les notes de conception originales (informelles, gardées telles quelles) référencées par endroits dans le code (« premier_jet.md section 5 », etc.).

## Sommaire

1. [Panorama du dépôt](#1-panorama-du-dépôt)
2. [Conventions à connaître avant de lire le code](#2-conventions-à-connaître-avant-de-lire-le-code)
3. [Parcours pas à pas (commandes)](#3-parcours-pas-à-pas-commandes)
4. [Référence détaillée, fichier par fichier](#4-référence-détaillée-fichier-par-fichier)
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

---

## 1. Panorama du dépôt

Le projet prédit une note 0-10 pour une carte Magic: The Gathering (limited/draft), à partir de ses caractéristiques Scryfall (coût, type, texte, rareté, force/endurance), entraîné sur les win rates publiques de 17Lands. L'objectif spécifique — et la contrainte qui façonne presque tout le code — est de **généraliser à des cartes/sets pas encore sortis** : aucune feature ne doit encoder « quel set » de façon à devenir inutilisable sur un set inédit.

Le dépôt s'organise en quatre couches, dans l'ordre où les données les traversent :

1. **Collecte de données** (`fetch_17lands.py`, `fetch_scryfall.py`, `labels.py`) : télécharge et met en cache les win rates 17Lands et les caractéristiques Scryfall, calcule les métriques par carte.
2. **Représentation** (`features.py`, `text_embeddings.py`, `ratings.py`, `color_context.py`) : transforme une carte en vecteur numérique, une métrique brute en note 0-10.
3. **Modèles et agrégation multi-set** (`multiset.py`, `set_context.py`, `lora_text_encoder.py`, `model.py`, `model_joint.py`, `mlm_pretrain.py`) : construit le dataset multi-set poolé, les vecteurs de contexte, les réseaux de neurones (dont le tower de texte entraînable par LoRA).
4. **Entraînement et usage** (`train_multiset.py`, `train_lora.py`, `train_color_context.py`, `rate_set.py`, `smoke_test.py`) : les scripts qu'on lance vraiment.

Aucun test automatisé n'existe (pas de `pytest`) : la vérification se fait en lançant un entraînement réel et en lisant MSE/Pearson r par epoch sur validation (jamais sur test avant la toute fin), et — pour tout mécanisme qui fait interagir plusieurs cartes entre elles (contexte par couleur entraînable, DANN) — en re-vérifiant explicitement avec `split_mode="set"` (voir §2).

**Attention** : contrairement au dépôt `Coinche`, **aucun script ici n'a de flags CLI**. Tous les points d'entrée sont des fonctions `main()` à arguments-mots-clés nombreux, avec des valeurs par défaut correspondant soit à un smoke test, soit à la config actuellement recommandée. Pour changer un paramètre, on appelle `main(...)` depuis un `python -c` one-liner ou un petit script, pas depuis la ligne de commande.

### Graphe de dépendances (qui importe quoi)

```
fetch_17lands.py     (aucune dépendance interne)
fetch_scryfall.py    (aucune dépendance interne)
labels.py            (aucune dépendance interne)
text_embeddings.py   (aucune dépendance interne)
ratings.py           (aucune dépendance interne)
color_context.py     (aucune dépendance interne)
model.py             (aucune dépendance interne)
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
```

---

## 2. Conventions à connaître avant de lire le code

- **Métriques 17Lands** (`labels.compute_card_metrics`) : `gih_wr` (win rate quand la carte a été vue en main, piochée ou en main d'ouverture), `gns_wr` (win rate quand elle était dans le deck mais jamais vue), `iih = gih_wr - gns_wr` (impact marginal intrinsèque de la carte, isolé de la force du reste du deck — c'est l'« IWD » de 17Lands), `gp_wr` (win rate sur toutes les parties où la carte était dans le deck, vue ou non — signal plus dilué/contextuel que `gih_wr`), `play_rate` (fraction des decks où, disponible dans le pool, la carte a effectivement été maindeckée).
- **Formules de note** (`ratings.RAW_SCORE_FORMULAS`) : combinaisons de ces métriques (`gih_iih = gih_wr + iih`, `gp_iih = gp_wr + iih`, `iih_only`, `gp_wr_only`, etc.), toutes reliées algébriquement via `gih_wr = gp_wr + (1-p)*iih`. Normalisées ensuite sur une échelle 0-10 (`ratings.apply_normalization` : 5 = moyenne, 0/10 = moyenne ± 3 écarts-types, clampé).
- **Deux notions de « rang LoRA » complètement différentes, à ne jamais confondre** : le rang 4 de `lora_text_encoder.LoraTextEncoder` (le LoRA de la tâche de rating elle-même, entraîné dans `train_lora.py`/`train_color_context.py`) et le rang 32/64 de `mlm_pretrain.py` (un LoRA différent, sur une étape d'adaptation de vocabulaire *en amont*, jamais utilisé pour prédire une note).
- **IIH vs GP WR** : IIH est le signal décontextualisé (fonction du texte de la carte seul, apprenable proprement, r≈0.6 en test) ; GP WR est le signal contextuel (dépend de l'environnement de draft réel, qui n'existe pas encore pour un set non sorti), plus dur (r≈0.48-0.5). D'où le double objectif (`extra_formulas=["gp_wr_only"]` en plus de `formula="iih_only"`) de la config actuelle.
- **Split by-name vs by-set** (`train_lora.split_by_name_3way`/`split_by_set_3way`) : le split par nom de carte (défaut, comparable à presque tous les résultats enregistrés dans ce projet) n'est un test de généralisation fiable que pour un mécanisme **sans paramètre entraînable** (la moyenne fixe de `set_context.py`). Tout mécanisme **entraînable** qui fait interagir plusieurs cartes entre elles (`color_context.py`/`train_color_context.py`, DANN) doit systématiquement être revérifié avec `split_mode="set"` (sets entiers tenus à l'écart) — des gains qui semblaient réels en by-name se sont déjà entièrement effondrés sous ce test.
- **`records`, la structure de données pivot** : `multiset.build_dataset()` renvoie une liste de dicts `{**metrics_17lands, "name", "set_code", "scryfall_card"}`. `train_lora.py`/`train_color_context.py` enrichissent chaque record en place avec `"structured"` (`features.structured_features`), `"oracle_text"`, et éventuellement `"set_context"` (vecteur de contexte fixe) ou `"_loss_weight"`. C'est ce dict, pas une classe dédiée, qui circule partout.
- **Format des checkpoints** : un dossier avec deux éléments — `lora_adapter/` (adaptateur PEFT sauvegardé par `model.save_pretrained`) et `head.pt` (dict `torch.save` avec `head_state_dict`, `mu`, `sigma`, `formula`, `base_model_path`, `use_set_context`, `extra_formulas`/`extra_mus`/`extra_sigmas`, et pour `train_color_context.py` en plus `color_context_state_dict`/`context_dim`). `rate_set.load_model` sait lire à la fois ce format courant (liste `extra_formulas`) et l'ancien format à un seul `second_formula` (checkpoint `lora_joint_dual` historique).
- **Environnement** : `env_coinche` (conda), partagé avec le dépôt `Coinche` — jamais de `conda install`, toujours `pip install` pour ne pas casser le solveur conda (voir mémoire projet).

---

## 3. Parcours pas à pas (commandes)

Tout se lance depuis la racine du dépôt, avec l'environnement `env_coinche` actif (`conda run -n env_coinche python -m ...` ou l'environnement déjà activé dans le shell).

### 3.1 Test de fumée (vérifier que toute la chaîne tourne)

```bash
python -m mtg_rating.smoke_test
```
Télécharge un set (ECL), calcule les métriques, entraîne un petit MLP jouet par formule de note, teste l'inférence sur 3 cartes hors set. Ne vise pas un bon modèle — juste prouver que fetch → features → train → predict ne plante pas. Premier réflexe après une modification touchant `labels.py`/`features.py`/`ratings.py`.

### 3.2 Étape de pré-entraînement MLM (rarement nécessaire — seulement si on change le corpus/objectif)

```bash
python -m mtg_rating.mlm_pretrain
```
Adapte MiniLM au vocabulaire MTG via un LoRA de rang 32 (attention !, différent du rang 4 de la tâche de rating) entraîné en masked-LM sur les ~30k cartes Scryfall. Produit une base encodeur autonome (`merge_and_unload()`), écrite dans `output_dir` à la toute fin (meilleur état selon la loss val, early stopping).

Le checkpoint effectivement utilisé en aval, `data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/`, n'est **pas** ce fichier de sortie final (celui-là n'existe qu'une fois, choisi par early stopping sur la loss val) — c'est un instantané **périodique** pris à un rythme fixe (`checkpoint_every`, sans lien avec quel epoch s'avère le meilleur), sauvegardé seulement si `checkpoint_every` est explicitement demandé (défaut `0` = aucun instantané intermédiaire). Reproduire ce répertoire précisément demande donc de passer `checkpoint_every` **et** `checkpoint_root` explicitement :
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
**Piège à connaître** (voir §4.13) : si `checkpoint_root` n'est *pas* passé explicitement, il ne se déduit **pas** de `output_dir` qu'on vient de fixer sur cette même ligne — il retombe sur la constante de module `OUTPUT_DIR` (`data/models/minilm_mtg_pretrained`), donc sur `data/models/minilm_mtg_pretrained_checkpoints`, un tout autre dossier. Omettre `checkpoint_root` ici enverrait silencieusement les instantanés intermédiaires au mauvais endroit.

### 3.3 Baseline à embeddings figés (pour comparaison, plus simple que le pipeline principal)

```bash
python -m mtg_rating.train_multiset
```
Pool les 26 sets, split 80/20 par nom de carte, entraîne un MLP jouet (`model.CardRatingNet`) sur des embeddings MiniLM **jamais mis à jour**, pour chacune des formules de `ratings.RAW_SCORE_FORMULAS`. Met en cache le dataset avec features déjà calculées (`data/raw/multiset_dataset.npz`) — le plus coûteux (le passage MiniLM) n'est donc payé qu'une fois.

### 3.4 Pipeline principal — fine-tuning LoRA joint (le script de production)

```bash
python -m mtg_rating.train_lora
```
Lance la config par défaut du module (`formula="gih_iih"`, pas de contexte de set, pas de checkpoint MLM-pretrained) — et, comme `save=True` par défaut, **écrase silencieusement** `data/models/lora_joint` (`CHECKPOINT_DIR`, `train_lora.py:35`) si ce dossier contient déjà un checkpoint auquel on tient. Pour lancer la **config actuellement recommandée** (`CLAUDE.md`), avec le `checkpoint_dir` explicite nécessaire pour atterrir dans le bon dossier :
```bash
python -c '
from mtg_rating.train_lora import main
main(
    formula="iih_only", extra_formulas=["gp_wr_only"],
    use_set_context=True,
    base_model_path="data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4",
    checkpoint_dir="data/models/lora_joint_dual",
)
'
```
**Piège à connaître** : sans ce `checkpoint_dir` explicite, `main()` sauvegarde par défaut dans `data/models/lora_joint` (`CHECKPOINT_DIR`), **pas** dans `data/models/lora_joint_dual` — le dossier que `rate_set.py` charge par défaut (`rate_set.py:32`). Omettre ce paramètre produirait un checkpoint entraîné correctement mais que `python -m mtg_rating.rate_set` n'irait jamais lire (il continuerait à charger l'ancien `lora_joint_dual`, sans erreur ni avertissement — juste des notes qui ne bougent pas après le nouvel entraînement).

Variantes utiles à connaître (toutes désactivées par défaut, aucune confirmée gagnante à ce jour — voir `CLAUDE.md` « Key established facts ») :
```bash
# Vérification honnête pour tout mécanisme entraînable transversal aux cartes
python -c 'from mtg_rating.train_lora import main; main(split_mode="set")'

# Variantes de loss (aucune confirmée, à tester une seule à la fois, plusieurs seeds)
python -c 'from mtg_rating.train_lora import main; main(sample_weight_power=0.5)'
python -c 'from mtg_rating.train_lora import main; main(threshold_penalty_weight=1.0)'
python -c 'from mtg_rating.train_lora import main; main(loss_shape="saturating")'
python -c 'from mtg_rating.train_lora import main; main(contrastive_weight=0.1)'
python -c 'from mtg_rating.train_lora import main; main(triplet_weight=0.1)'  # le plus récent, smoke-testé seulement
```

### 3.5 Pipeline expérimental — contexte par couleur entraînable (non adopté, gardé pour référence)

```bash
python -c 'from mtg_rating.train_color_context import main; from mtg_rating.multiset import SET_CODES; main(SET_CODES)'
# honnêteté (sets entiers tenus à l'écart) :
python -c 'from mtg_rating.train_color_context import main; from mtg_rating.multiset import SET_CODES; main(SET_CODES, split_mode="set")'
# avec DANN (pousse le contexte à ne pas encoder l'identité du set) :
python -c 'from mtg_rating.train_color_context import main; from mtg_rating.multiset import SET_CODES; main(SET_CODES, use_dann=True)'
```
Attention cross-carte (`ColorAttentionContext`) sur les communes/peu-communes de la même couleur primaire dans le même set — c'est ce mécanisme précisément qui a montré une inflation par-nom qui s'effondre par-set (`CLAUDE.md`), d'où son statut « non adopté ».

**Contrairement à `train_lora.main`, ici `save=False` par défaut** : ces trois commandes entraînent et évaluent mais ne persistent rien sur disque — cohérent avec le statut « non adopté »/exploratoire de ce pipeline, mais à savoir si on veut vraiment garder un checkpoint : ajouter `save=True` (et éventuellement `checkpoint_dir=...`, sinon `data/models/color_context` par défaut), ex. `main(SET_CODES, save=True)`.

### 3.6 Noter un set entier et générer une page HTML

```bash
python -m mtg_rating.rate_set                                              # note DFT par défaut (main("DFT") en bas du fichier)
python -c 'from mtg_rating.rate_set import main; main("MKM")'              # n'importe quel autre set
python -c 'from mtg_rating.rate_set import main; main("MKM", checkpoint_dir="data/models/lora_joint")'  # meme set, checkpoint single-formule au lieu du dual
```
Charge le checkpoint `lora_joint_dual` (config recommandée) par défaut, note toutes les cartes du set donné, écrit `data/ratings/ratings_<set>.html` — un tableau par couleur primaire, triées de la plus forte à la plus faible par GP WR prédit. Ne nécessite **que** des données Scryfall (pas de labels 17Lands) — fonctionne donc sur un set totalement inédit.

### 3.7 Pipeline bout-en-bout typique

Schéma simplifié (pas des commandes copiables telles quelles — les commandes complètes, avec les paramètres `checkpoint_every`/`checkpoint_root`/`checkpoint_dir` nécessaires pour atterrir exactement dans ces dossiers, sont données en 3.2/3.4/3.6 ci-dessus) :

```
mlm_pretrain.py (rank 64)              → data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/
        │
        ▼
train_lora.py (iih_only + gp_wr_only,  → data/models/lora_joint_dual/
  use_set_context=True, base_model_path=épreuve ci-dessus)
        │
        ▼
rate_set.py main("<SET_A_VENIR>")      → data/ratings/ratings_<set>.html
```

---

## 4. Référence détaillée, fichier par fichier

### 4.1 `fetch_17lands.py`

Télécharge les fichiers bruts 17Lands (`game_data`/`draft_data`), avec cache local disque — aucune dépendance interne.

- `_download(kind, set_code, event_type, dest_dir)` (`fetch_17lands.py:11`) : construit l'URL S3, télécharge en streaming (chunks de 1 Mo) si le fichier n'est pas déjà en cache, sinon retourne directement le chemin existant. Appelée par `download_game_data`/`download_draft_data`.
- `download_game_data(set_code, event_type="PremierDraft", dest_dir=None)` / `download_draft_data(...)` (`fetch_17lands.py:30`, `34`) : wrappers publics fixant `kind="game_data"`/`"draft_data"`. `download_game_data` est la seule des deux réellement utilisée ailleurs dans le dépôt (`multiset.get_set_metrics`, `smoke_test.main`) — `download_draft_data` n'a aucun appelant actuel dans le code (prévu pour un usage futur, cf. `premier_jet.md`).

### 4.2 `fetch_scryfall.py`

Récupère les caractéristiques de cartes depuis l'API Scryfall — deux usages distincts : par set (pour l'entraînement/l'inférence) et en bloc (pour le corpus de pré-entraînement MLM).

- `fetch_set_cards(set_code)` (`fetch_scryfall.py:41`) : pagine `cards/search?q=set:X` (~175 cartes/page), respecte l'étiquette Scryfall (`time.sleep(0.1)` entre pages), met en cache tout le set en JSON. Appelée par `multiset.build_dataset`, `rate_set._usable_cards`, `smoke_test.main`.
- `_is_relevant(card)` (`fetch_scryfall.py:62`) : vrai si la carte est légale dans au moins un des formats `RELEVANT_FORMATS` (`legacy`, `modern`, `standard`) — exclut Un-sets/cartes bannies partout sauf Vintage/Commander. Appelée par `fetch_bulk_oracle_cards`.
- `fetch_bulk_oracle_cards()` (`fetch_scryfall.py:67`) : télécharge le fichier bulk `oracle_cards` de Scryfall (~200 Mo, une entrée par carte unique toutes éditions confondues), filtre aux champs utiles (`BULK_FIELDS`) et aux cartes pertinentes/avec texte, supprime le fichier brut immédiatement après extraction (contrainte disque), met en cache le résultat filtré. Appelée uniquement par `mlm_pretrain.main` (corpus de pré-entraînement MLM — c'est le seul consommateur de cette fonction dans tout le dépôt).
- `fetch_card(name)` (`fetch_scryfall.py:97`) : lookup ponctuel d'une carte par nom flou (`fuzzy`), sans cache. Appelée uniquement par `train_multiset.main` et `smoke_test.main`, pour tester l'inférence sur des cartes « evergreen » garanties hors du pool d'entraînement — `train_lora.py`/`train_color_context.py` n'ont pas ce test d'inférence intégré et ne l'appellent pas.

### 4.3 `labels.py`

Calcule les métriques par carte à partir d'un fichier `game_data` brut — aucune dépendance interne, tout le travail se fait avec `pandas`.

- `BASIC_LAND_NAMES` (`labels.py:49`) : les 6 noms de terrain de base, exclus partout dans le dépôt (dynamiques de pioche non comparables à des sorts). Réutilisée directement par `rate_set._usable_cards` (import de `labels.BASIC_LAND_NAMES`).
- `_is_basic_land_col(col)` (`labels.py:52`) : reconnaît une colonne `opening_hand_Plains` etc. Appelée par `compute_card_metrics` pour exclure ces colonnes dès la lecture du CSV (jamais même chargées en mémoire).
- **`compute_card_metrics(game_data_path, min_games=200, include_play_rate=False)`** (`labels.py:59`) : lit uniquement les colonnes nécessaires (`usecols`) avec des dtypes compacts (`Int8` nullable — un `game_data` chargé sans ces précautions a fait OOM sur ce poste sur un set plus large), calcule par carte `gih_wr`/`gns_wr`/`iih`/`gp_wr` et, si demandé, `play_rate`/`pool_count` (plus coûteux : nécessite les colonnes `sideboard_*` et une déduplication par `(draft_id, build_index)`). Ignore toute carte avec moins de `min_games` parties vues. Appelée par `multiset.get_set_metrics` (avec `include_play_rate=True`) et par `smoke_test.main` (sans, pour le test de fumée le plus simple).

### 4.4 `features.py`

Transforme un dict de carte Scryfall en vecteur numérique fixe. Dépend de `text_embeddings.py`.

- `_parse_pt(value)` (`features.py:17`) : convertit une force/endurance Scryfall (qui peut être `"*"` ou absente) en float, `0.0` par défaut. Appelée par `structured_features`.
- **`structured_features(card)`** (`features.py:24`) : vecteur 14-dim — coût de mana, 5 couleurs one-hot, 5 types one-hot, rareté (0-3), force, endurance. Volontairement **sans feature « set »** (le but étant justement de généraliser à un set inconnu). Appelée par `card_to_features`, `set_context.compute_set_context_vectors`/`compute_per_color_context_vectors`, `train_lora.main`/`train_color_context.main` (calculée une fois par record et stockée dans `r["structured"]`), `rate_set.rate_cards`.
- **`card_to_features(card)`** (`features.py:43`) : `structured_features(card)` + l'embedding MiniLM figé du texte oracle (`text_embeddings.embed_text`) — le vecteur complet utilisé par le pipeline **frozen-embedding** (`model.py`/`train_multiset.py`/`smoke_test.py`). N'est **pas** utilisé par `train_lora.py`/`train_color_context.py`, qui ré-encodent le texte à chaque batch via le tower LoRA entraînable au lieu de cet embedding figé précalculé.
- `STRUCTURED_DIM`/`FEATURE_DIM` (`features.py:48-49`) : tailles dérivées, réutilisées comme dimensions d'entrée un peu partout (`model_joint.CardRatingNetJoint`, `set_context.CONTEXT_DIM`, `train_color_context.py`).

### 4.5 `text_embeddings.py`

L'encodeur de texte **figé** (jamais mis à jour) — `sentence-transformers/all-MiniLM-L6-v2` via `transformers` brut (pas le wrapper `sentence-transformers`, pour éviter ses dépendances scipy/scikit-learn/Pillow).

- `_load()` (`text_embeddings.py:21`) : charge tokenizer + modèle une seule fois (cache module-level `_tokenizer`/`_model`), met le modèle en `eval()`. Appelée par `embed_texts` à chaque appel (no-op après le premier grâce au cache).
- **`embed_texts(texts)`** (`text_embeddings.py:30`) : tokenize un batch, passe dans le modèle sans gradient (`torch.no_grad()`), fait un mean-pooling pondéré par le masque d'attention. Appelée par `embed_text`, `set_context.compute_set_context_vectors`/`compute_per_color_context_vectors`, `rate_set._context_vector_for_set`.
- `embed_text(text)` (`text_embeddings.py:43`) : `embed_texts([text])[0]`, en liste Python. Appelée par `features.card_to_features`.
- `MODEL_NAME`/`EMBEDDING_DIM` (`text_embeddings.py:14-15`) : constantes réutilisées dans tout le dépôt — `MODEL_NAME` comme valeur par défaut de `base_model_path` (`LoraTextEncoder`, `mlm_pretrain.py`), `EMBEDDING_DIM=384` comme dimension attendue partout où un embedding MiniLM (figé ou LoRA) est concaténé à d'autres features.

### 4.6 `ratings.py`

Convertit une métrique brute en note 0-10 — aucune dépendance interne, pur calcul statistique.

- `RAW_SCORE_FORMULAS` (`ratings.py:44`) : dict `nom → lambda(metrics) -> float`, les formules candidates (`gih`, `gih_iih`, `gp_iih`, `gih_2iih`/`3iih`/`5iih`/`10iih`/`15iih`, `iih_only`, `gp_wr_only`, `play_rate_only`) — voir §2 pour leur relation algébrique. C'est le dict que tous les scripts d'entraînement itèrent ou indexent par nom de formule.
- `raw_scores(metrics, formula)` (`ratings.py:59`) : applique une formule à chaque carte d'un dict `{nom: métriques}`, en sautant les cartes sans `iih`. Appelée par `smoke_test.main` (les autres scripts appliquent la formule directement sur des `records`, pas via cette fonction).
- `fit_normalization(raw)` / `apply_normalization(raw, mu, sigma)` (`ratings.py:64`, `69`) : séparées exprès pour qu'un pipeline train/test calibre `(mu, sigma)` **seulement** sur le train puis applique la même transformation au holdout, sans fuite de statistiques. Appelées ensemble par `train_multiset.main`, `train_lora.main`, `train_color_context.main`, `rate_set.attach_actual_ratings`.
- `normalize_to_10(raw)` (`ratings.py:76`) : raccourci fit+apply quand il n'y a pas de split (juste `fit_normalization(raw)` puis `apply_normalization`). Appelée par `compute_ratings`.
- `compute_ratings(metrics, formula)` (`ratings.py:80`) : `raw_scores` + `normalize_to_10` enchaînés. Appelée uniquement par `smoke_test.main`.

### 4.7 `multiset.py`

Construit le dataset multi-set poolé — le point de jonction entre 17Lands et Scryfall pour les 26 sets retenus.

- `SET_CODES` (`multiset.py:23`) : la liste des 26 sets utilisés à l'entraînement, vérifiée directement contre le S3 public 17Lands (pas l'API du site) ; exclut par principe les sets Alchemy (digital-only) et « draft innovation » (produits supplémentaires hors rotation Standard). Réutilisée par `train_lora.py`/`train_color_context.py` (valeur par défaut de `set_codes`) et importée telle quelle dans leurs blocs `if __name__ == '__main__':`.
- `_metrics_cache_path(set_code, event_type)` / `_load_cached_metrics(path)` / `_round_metric(v)` / `_save_metrics_cache(path, metrics)` (`multiset.py:33`, `37`, `46`, `55`) : cache CSV par set des métriques déjà calculées — `_round_metric` arrondit à 4 décimales (au-delà, c'est du bruit d'échantillonnage, pas du signal). `_round_metric` est aussi réutilisée directement par `train_multiset.save_pool_summary`.
- **`get_set_metrics(set_code, event_type="PremierDraft")`** (`multiset.py:64`) : sert le cache CSV s'il existe, sinon télécharge le `game_data` (`fetch_17lands.download_game_data`), calcule les métriques (`labels.compute_card_metrics`, avec `include_play_rate=True`), les met en cache, puis **supprime le fichier brut** téléchargé (contrainte disque). Appelée par `build_dataset` et directement par `rate_set.attach_actual_ratings` (pour comparer une prédiction au vrai résultat 17Lands quand il existe).
- **`build_dataset(set_codes=None, event_type="PremierDraft")`** (`multiset.py:80`) : pour chaque set, récupère les métriques (`get_set_metrics`) et les cartes Scryfall (`fetch_scryfall.fetch_set_cards`), joint les deux par nom, saute silencieusement (avec un message) tout set qui échoue. Retourne la liste de `records` (voir §2) qui alimente ensuite tout entraînement multi-set. Appelée par `train_multiset.load_or_build_dataset`, `train_lora.main`, `train_color_context.main`.

### 4.8 `color_context.py`

Groupement des cartes par « couleur primaire » — utilisé à la fois par la moyenne fixe par couleur (`set_context.compute_per_color_context_vectors`) et par le mécanisme d'attention entraînable expérimental (`train_color_context.py`). Aucune dépendance interne.

- **`primary_color(scryfall_card)`** (`color_context.py:34`) : première couleur dans l'ordre WUBRG de `card["colors"]`, ou `"colorless"`. Une carte bicolore est donc rattachée à un seul groupe (sa première couleur), pas pooled sur les deux — limitation v1 assumée (`premier_jet.md` §6 envisage un v2 par paire de couleurs, jamais construit). Appelée par `set_context.compute_per_color_context_vectors`, `build_color_buckets`, `rate_set.py` ne l'utilise **pas** (son propre `display_group` a une logique différente, voir §4.17).
- **`build_color_buckets(records)`** (`color_context.py:42`) : regroupe les indices de `records` par `(set_code, primary_color)`, avec deux sous-listes par bucket — `"query"` (toutes les cartes du bucket) et `"informant"` (le sous-ensemble commune/peu-commune, le pool qui informe l'attention). Appelée uniquement par `train_color_context.main`.
- **`flatten_buckets(color_buckets)`** (`color_context.py:62`) : `{set_code: {color: bucket}}` → liste de `(set_code, color, bucket)`, plus pratique à itérer/mélanger. Appelée uniquement par `train_color_context.main`.

### 4.9 `set_context.py`

Vecteurs de contexte **non-paramétriques** (aucun poids entraînable) résumant « ce qu'il y a dans ce set », concaténés aux features propres de chaque carte. C'est la version **adoptée** (voir `CLAUDE.md`), alternative plus simple que l'attention entraînable de `color_context.py`/`train_color_context.py`.

- `_mean_vector(vectors)` (`set_context.py:37`) : moyenne composante par composante d'une liste de vecteurs de même dimension. Appelée par `compute_set_context_vectors`, `compute_per_color_context_vectors`, et réutilisée directement par `rate_set._context_vector_for_set` (import de `set_context._mean_vector`).
- **`compute_set_context_vectors(records)`** (`set_context.py:43`) : pour chaque set présent dans `records`, moyenne de `structured_features + embedding MiniLM figé` sur toutes ses cartes. Mis en cache dans `data/raw/set_context_vectors.json`, **clé par existence de fichier, pas par contenu** — régénérer avec un sous-ensemble de sets différent laisse un cache silencieusement obsolète (il faut supprimer le fichier à la main). Appelée par `train_lora.main` (si `use_set_context=True`) et lue directement par `rate_set._context_vector_for_set` pour les 26 sets déjà poolés.
- **`compute_per_color_context_vectors(records)`** (`set_context.py:64`) : même idée que ci-dessus mais groupé par `(set_code, primary_color)` (via `color_context.primary_color`) au lieu du set entier — un résumé plus ciblé (« carte blanche moyenne de ce set » plutôt que « carte moyenne toutes couleurs confondues »), toujours sans paramètre entraînable donc sans le problème de généralisation par-set des mécanismes entraînables. Mis en cache dans `set_context_vectors_per_color.json`, même piège de cache. Testé mais **pas adopté** (pas de gain net vs la version whole-set, `CLAUDE.md`). Appelée uniquement par `train_lora.main` (si `use_color_context=True` — mutuellement exclusif avec `use_set_context`).
- `CACHE_PATH`/`CACHE_PATH_PER_COLOR`/`CONTEXT_DIM` (`set_context.py:32-34`) : chemins de cache et dimension du vecteur de contexte (= `STRUCTURED_DIM + EMBEDDING_DIM`). `CACHE_PATH` est aussi importé directement par `rate_set.py` (sous l'alias `SET_CONTEXT_CACHE_PATH`), `CONTEXT_DIM` par `train_lora.py`/`rate_set.py`.

### 4.10 `lora_text_encoder.py`

Le tower de texte **entraînable** du pipeline principal — même base MiniLM que `text_embeddings.py`, mais enveloppée d'un adaptateur LoRA (rang 4, sur les projections `query`/`value`) mis à jour pendant l'entraînement.

#### `LoraTextEncoder(nn.Module)` (`lora_text_encoder.py:23`)
- `__init__(rank=LORA_RANK, dropout=0.0, base_model_path=MODEL_NAME)` (`lora_text_encoder.py:24`) : charge le tokenizer/modèle de base (générique par défaut, ou une base MLM-pré-entraînée localement — voir `mlm_pretrain.py`), l'enveloppe d'un `LoraConfig` (`peft.get_peft_model`) — c'est ce constructeur qui décide, via `base_model_path`, si on part de MiniLM générique ou de la version adaptée MTG.
- `print_trainable_parameters()` / `trainable_parameters()` (`lora_text_encoder.py:41`, `44`) : délègue au diagnostic `peft` intégré / renvoie la liste des paramètres avec `requires_grad=True` (les matrices LoRA seulement, jamais la base). `trainable_parameters()` est appelée par `model_joint.CardRatingNetJoint.trainable_parameters` et par les groupes de paramètres différenciés de `train_lora.main`/`train_color_context.main` (LR plus faible que la tête).
- `forward(texts)` (`lora_text_encoder.py:47`) : tokenize, passe dans le modèle **avec** gradient (contrairement à `text_embeddings.embed_texts`), mean-pooling identique. Appelée par `model_joint.CardRatingNetJoint.forward`/`ColorAttentionContext`-adjacent code dans `train_color_context.py` (`model.text_encoder(...)`), et directement dans `train_lora.py` pour les passes auxiliaires (`contrastive_loss`, `mine_hard_triplets`) qui ont besoin de l'embedding brut, pas juste de la prédiction finale.

### 4.11 `model.py`

Le pipeline **le plus simple** du dépôt : un petit MLP sur des embeddings déjà figés — utilisé uniquement par `train_multiset.py`/`smoke_test.py`, jamais par le pipeline LoRA principal.

#### `CardRatingNet(nn.Module)` (`model.py:7`)
`__init__(input_dim, hidden_dim=16)` : une couche cachée ReLU, une sortie scalaire. `forward(x)` : passe avant + `squeeze(-1)`.

- **`train_toy(features, labels, epochs=50, lr=1e-2)`** (`model.py:20`) : boucle d'entraînement complète en une fonction — construit le modèle, `Adam`, `MSELoss`, boucle plein-batch (pas de mini-batchs) sur `epochs` itérations. Retourne le modèle entraîné. Appelée par `smoke_test.main` et `train_multiset.main` — c'est la seule fonction d'entraînement de tout le dépôt qui n'a pas de logique d'early stopping/validation intégrée (les deux appelants gèrent train/test eux-mêmes autour).

### 4.12 `model_joint.py`

Les classes de modèle du pipeline principal (LoRA joint) et du pipeline expérimental (attention par couleur + DANN).

#### `CardRatingNetJoint(nn.Module)` (`model_joint.py:16`) — le modèle central
- `__init__(structured_dim, hidden_dims=None, lora_rank=LORA_RANK, lora_dropout=0.0, head_dropout=0.0, base_model_path=MODEL_NAME, set_context_dim=0, output_dim=1, layer_norm=False)` (`model_joint.py:17`) : construit un `LoraTextEncoder` (le tower de texte) puis une tête MLP (`hidden_dims` par défaut `[16]`, GELU + dropout par couche, LayerNorm Pre-LN optionnelle — désactivée par défaut pour reproduire exactement l'ancienne tête à une couche) prenant en entrée `structured_dim + EMBEDDING_DIM + set_context_dim`. `output_dim > 1` sert au mode multi-cible (`extra_formulas`).
- `forward(structured, texts, set_context=None)` (`model_joint.py:59`) : encode les textes (`self.text_encoder`), concatène `[structured, text_embedding, set_context?]`, passe dans la tête. `squeeze(-1)` si `output_dim == 1` pour que rien en aval n'ait à gérer un cas spécial. Appelée par `train_lora.evaluate`/la boucle d'entraînement de `train_lora.main`, et par `rate_set.rate_cards`.
- `trainable_parameters()` (`model_joint.py:71`) : paramètres LoRA de `text_encoder` + tous les paramètres de la tête, en une seule liste. Sans appelant actuel dans le dépôt : `train_lora.main`/`train_color_context.main` construisent leurs propres groupes de LR différenciés à la main (`model.text_encoder.trainable_parameters()` et `model.head.parameters()` séparément, pour leur donner des LR différents) plutôt que d'utiliser ce raccourci qui les fusionne.

#### `ColorAttentionContext(nn.Module)` (`model_joint.py:75`) — expérimental, non adopté
- `__init__(input_dim, context_dim=32, num_heads=4, dropout=0.0)` (`model_joint.py:87`) : une projection linéaire vers `context_dim` + une couche `nn.MultiheadAttention`. Volontairement petit (peu de capacité), par choix de conception documenté (limiter l'overfitting sur les ~156 buckets set×couleur disponibles).
- `forward(combined, informant_local_idx)` (`model_joint.py:93`) : `combined` = toutes les cartes d'un bucket (requêtes), `informant_local_idx` = indices des communes/peu-communes (clés/valeurs). Attention + résiduel. Appelée uniquement par `train_color_context._bucket_forward`.
- `trainable_parameters()` (`model_joint.py:103`) : tous les paramètres du module (rien n'est figé ici, contrairement au tower de texte). Utilisée par `train_color_context.main` pour son propre groupe de LR.

#### `_GradientReversalFunction(torch.autograd.Function)` / `GradientReversalLayer(nn.Module)` / `DomainClassifier(nn.Module)` (`model_joint.py:107`, `118`, `135`) — support DANN, expérimental
- `_GradientReversalFunction.forward`/`backward` (`model_joint.py:108`, `113`) : identité en avant, gradient inversé (× `-lambda_`) en arrière — le mécanisme brut derrière `GradientReversalLayer.forward` (`model_joint.py:131`), qui l'enveloppe en `nn.Module` pour pouvoir être empilé dans un modèle normalement. `lambda_` est prévu pour être rampé de 0 à 1 pendant l'entraînement (fait par `train_color_context.main`, pas par la classe elle-même).
- `DomainClassifier.__init__(context_dim, num_domains)`/`forward(context)` (`model_joint.py:141`, `149`) : petit MLP classifiant de quel set provient un vecteur de contexte — placé juste après une `GradientReversalLayer` dans `train_color_context.py`, de sorte que ses propres poids apprennent normalement à bien classifier pendant que l'encodeur en amont est poussé dans la direction opposée (ne plus encoder l'identité du set). Appelées uniquement dans `train_color_context.run_epoch`/`main` (`use_dann=True`).

### 4.13 `mlm_pretrain.py`

Étape de pré-entraînement **en amont**, séparée du fine-tuning de rating — adapte MiniLM au vocabulaire MTG par masked-LM sur tout le corpus Scryfall (~30k cartes), pas sur les ~5-6k lignes labellisées 17Lands.

- `split_texts(texts, val_fraction, seed)` (`mlm_pretrain.py:59`) : split train/val simple par mélange+coupe (pas par nom de carte comme ailleurs — ici l'unité est le texte, pas une métrique 17Lands à ne pas fuiter). Appelée par `main`.
- **`build_model(tokenizer, full_finetune=False, lora_rank=LORA_RANK)`** (`mlm_pretrain.py:66`) : `full_finetune=True` renvoie le modèle `AutoModelForMaskedLM` brut, tous paramètres entraînables (point de comparaison seulement, jamais adopté par défaut — risque d'overfitting/oubli catastrophique documenté) ; sinon enveloppe d'un LoRA (`target_modules=["query","value"]`, `modules_to_save=["cls"]` — la tête de prédiction MLM n'existe pas dans le checkpoint `sentence-transformers` de base et doit donc être entraînée en entier, pas juste adaptée). Appelée par `main`.
- `iter_batches(texts, batch_size)` (`mlm_pretrain.py:95`) : découpage en tranches, identique en esprit à `train_lora.iter_batches` (dupliqué, pas partagé). Appelée par `run_epoch`.
- **`run_epoch(model, tokenizer, collator, texts, batch_size, train, optimizer=None)`** (`mlm_pretrain.py:100`) : une passe (train ou eval) — masque aléatoirement des tokens (`DataCollatorForLanguageModeling`), calcule la loss MLM, met à jour si `train=True`. Retourne la loss moyenne pondérée par nombre de tokens masqués. Appelée deux fois par epoch dans `main` (une fois train, une fois val).
- `save_pretrained_checkpoint(model, tokenizer, output_dir, full_finetune=False)` (`mlm_pretrain.py:122`) : sauvegarde un checkpoint intermédiaire — travaille sur une `deepcopy` pour ne pas perturber le modèle en cours d'entraînement, fusionne le LoRA (`merge_and_unload()`) sauf en `full_finetune` (pas de wrapper peft à fusionner). Appelée par `main` si `checkpoint_every` est positif, avec `checkpoint_root / f"epoch{epoch+1}"` comme `output_dir` — donc **pas** le même dossier que le `output_dir` final de `main`.
- **`main(seed=0, epochs=EPOCHS, checkpoint_every=0, checkpoint_root=None, full_finetune=False, output_dir=None, lora_rank=LORA_RANK)`** (`mlm_pretrain.py:135`) : télécharge le corpus (`fetch_scryfall.fetch_bulk_oracle_cards`), split, construit le modèle (`build_model`), boucle avec early stopping sur la loss val (patience `PATIENCE=5`), sauvegarde le meilleur état, fusionne et sauvegarde la base encodeur finale dans `output_dir`. C'est ce fichier de sortie qui devient ensuite `base_model_path` d'un `LoraTextEncoder` frais dans `train_lora.py`.
  **Piège** (`mlm_pretrain.py:145-147`) : `output_dir = output_dir or OUTPUT_DIR` calcule bien la valeur locale à partir du paramètre passé, mais la ligne juste après, `checkpoint_root = checkpoint_root or OUTPUT_DIR.parent / "..."`, retombe sur la **constante de module** `OUTPUT_DIR` (`data/models/minilm_mtg_pretrained`) si `checkpoint_root` n'est pas fourni — **pas** sur la valeur de `output_dir` qu'on vient de fixer sur la ligne d'au-dessus. Passer un `output_dir` personnalisé sans passer `checkpoint_root` en même temps envoie donc les instantanés intermédiaires (`checkpoint_every`) dans un dossier basé sur l'ancien nom par défaut, pas à côté du nouveau `output_dir`. Le checkpoint réellement utilisé en aval (`data/models/minilm_mtg_pretrained_rank64_checkpoints/epoch4/`, `checkpoint_every=4` d'après les instantanés présents sur disque — epoch4/8/12/16/20) n'a donc pu être produit qu'en passant `checkpoint_root` explicitement, voir §3.2.

### 4.14 `train_multiset.py`

La baseline « embeddings figés » avec vraie évaluation held-out (contrairement au smoke test) — sert de point de comparaison au pipeline LoRA principal.

- `split_by_name(records, test_fraction, seed)` (`train_multiset.py:37`) : split 2 voies (train/test) par nom de carte unique. Appelée par `main`. (Note : `train_lora.py` a sa propre version 3 voies, `split_by_name_3way` — pas la même fonction, pas de partage de code entre les deux.)
- **`pearson(a, b)`** (`train_multiset.py:45`) : corrélation de Pearson via `numpy.corrcoef`. **Réutilisée directement** (import) par `train_lora.score_predictions` — c'est le seul symbole que `train_lora.py` importe de ce module, d'où l'entrée `train_multiset.py (pearson)` dans le graphe de dépendances de la §1.
- `save_pool_summary(records, path)` (`train_multiset.py:49`) : dump CSV allégé (sans les blobs Scryfall complets) du dataset poolé, pour inspection sur disque indépendamment du buffering stdout. Appelée par `load_or_build_dataset`.
- **`load_or_build_dataset()`** (`train_multiset.py:72`) : sert le cache `.npz` s'il existe (features déjà calculées — l'étape la plus coûteuse), sinon appelle `multiset.build_dataset()`, calcule `card_to_features` pour chaque record, et met en cache. Appelée uniquement par `main`.
- **`main()`** (`train_multiset.py:112`) : charge le dataset, split 80/20 par nom, puis pour **chaque formule** de `ratings.RAW_SCORE_FORMULAS` : calibre `(mu, sigma)` sur train seul, entraîne (`model.train_toy`, 200 epochs, sans early stopping), évalue MSE + Pearson r sur test, teste l'inférence sur 3 cartes evergreen hors pool.

### 4.15 `train_lora.py`

**Le script principal du dépôt.** Fine-tune conjointement le tower de texte (LoRA rang 4) et la tête de rating sur le dataset multi-set poolé — le plus gros fichier (632 lignes), avec plusieurs variantes de loss optionnelles jamais confirmées gagnantes.

- `_as_tuple(value)` (`train_lora.py:67`) : normalise une cible (float unique ou tuple multi-formule) en tuple — utilitaire interne pour que le code de construction des cibles n'ait pas à distinguer les deux cas. Appelée dans `main` (construction de `train_targets`/`val_targets`/`test_targets`) et par `mine_hard_triplets`.
- `iter_batches(pairs, batch_size)` (`train_lora.py:71`) : découpage en tranches d'une liste de paires `(record, target)`. Appelée par `evaluate` et `main`.
- **`split_by_name_3way(records, val_fraction, test_fraction, seed)`** (`train_lora.py:76`) : split train/val/test par nom de carte unique — **le split de référence de tout le projet** (`SPLIT_SEED=42` fixe partout). Appelée par `main` (`split_mode="name"`, le défaut) et importée par `train_color_context.py`.
- **`split_by_set_3way(records, val_fraction, test_fraction, seed)`** (`train_lora.py:88`) : même chose mais au niveau du set entier (pas du nom de carte) — **le test d'honnêteté** pour tout mécanisme entraînable transversal aux cartes (voir §2). Appelée par `main` (`split_mode="set"`) et par `train_color_context.py`.
- `_set_context_tensor(batch_records)` (`train_lora.py:106`) : renvoie `None` si les records ne portent pas de clé `"set_context"` (mode sans contexte), sinon empile les vecteurs en tensor. Appelée par `evaluate` et la boucle d'entraînement de `main`.
- **`score_predictions(preds, targets)`** (`train_lora.py:115`) : calcule MSE (+ Pearson r si cible simple, ou un détail par composante si cible multiple/`extra_formulas`) — logique **partagée** avec `train_color_context.evaluate` (import direct), pour que les deux boucles d'entraînement rapportent leurs résultats de façon identique. Appelée par `evaluate` (ici) et `train_color_context.evaluate`.
- **`weighted_mse_loss(pred, target, weight, threshold_penalty_weight=0.0, threshold=1.0, loss_shape="mse", saturating_asymptote=8.0, saturating_scale=3.0)`** (`train_lora.py:142`) : la fonction de loss — avec les valeurs par défaut, strictement équivalente à `nn.MSELoss()`. `loss_shape="saturating"` remplace l'erreur quadratique par une version type Geman-McClure qui sature à une asymptote fixe pour les grosses erreurs au lieu de croître sans borne (paramètres `A=8, B=3` dérivés algébriquement de contraintes explicites sur le point d'inflexion). `threshold_penalty_weight` ajoute un terme charnière-au-carré séparé au-delà d'un seuil d'erreur — les deux mécanismes ne sont pas censés être combinés (le second est un essai plus grossier du même objectif que le premier supersede). Appelée dans la boucle d'entraînement de `main`.
- **`contrastive_loss(text_embedding, target_component, tau)`** (`train_lora.py:195`) : loss auxiliaire dans l'espace d'embedding — rapproche deux cartes proportionnellement à la proximité de leur cible réelle (`exp(-|gap|/tau)`), indépendamment de la similarité de surface du texte. Motivée directement par le « probe Annul » (voir `CLAUDE.md`). Appelée dans la boucle d'entraînement de `main` si `contrastive_weight > 0`.
- **`mine_hard_triplets(model, records, targets, threshold)`** (`train_lora.py:224`, `@torch.no_grad()`) : ré-embarque tout le train set (pas juste un batch) pour trouver, pour chaque carte, son négatif dur (embedding le plus proche parmi les cartes dont la cible réelle diffère de plus de `threshold`) et son positif (cible réelle la plus proche, sans recherche d'embedding nécessaire). Coûteux (passe complète) — appelé périodiquement (`triplet_mining_every`, par défaut chaque epoch), pas à chaque batch. Appelée par `main`.
- **`triplet_loss(anchor, positive, negative, margin)`** (`train_lora.py:278`) : loss triplet standard en distance cosinus, cohérente avec la métrique de similarité utilisée par `mine_hard_triplets` pour choisir le négatif. Appelée dans la boucle d'entraînement de `main` si `triplet_weight > 0` et que des triplets ont déjà été minés.
- **`evaluate(model, records, targets)`** (`train_lora.py:293`) : passe eval complète (batchée), retourne `(mse, extra, preds)` via `score_predictions`. Appelée par `main` à chaque epoch (sur val) et une fois à la toute fin (sur test).
- **`save_checkpoint(model, mu, sigma, formula, checkpoint_dir=None, base_model_path=MODEL_NAME, use_set_context=False, use_color_context=False, extra_formulas=None, extra_mus=None, extra_sigmas=None)`** (`train_lora.py:307`) : sauvegarde l'adaptateur LoRA (`save_pretrained`, quelques dizaines de Ko) + un dict `head.pt` avec tout le contexte nécessaire pour recharger le modèle correctement plus tard (voir §2, « format des checkpoints »). `checkpoint_dir=None` retombe sur la constante de module `CHECKPOINT_DIR` = `data/models/lora_joint` (`train_lora.py:35`) — **pas** `lora_joint_dual`, le dossier que `rate_set.py` charge par défaut (`rate_set.CHECKPOINT_DIR`, différent). Appelée par `main` si `save=True` (le défaut) ; voir §3.4 pour l'exemple de config recommandée avec le `checkpoint_dir` explicite qui évite ce piège.
- **`main(...)`** (`train_lora.py:351`) : le cœur du fichier — construit le dataset poolé, ajoute `structured`/`oracle_text` (et `set_context` si demandé) à chaque record, filtre les records sans valeur pour les formules demandées, split (`split_by_name_3way` ou `split_by_set_3way`), calibre `(mu, sigma)` par formule (primaire + `extra_formulas`, chacune indépendamment, sur train seul), construit `CardRatingNetJoint` avec deux groupes de LR (`lora_lr` pour le tower de texte, `head_lr` pour la tête), boucle sur `epochs` avec (au choix) pondération par nombre de parties (`sample_weight_power`), loss saturante, pénalité de seuil, perte contrastive, perte triplet — évalue sur val à chaque epoch, early stop (patience `PATIENCE=8`), **touche le test exactement une fois** à la toute fin. Retourne `(model, preds, test_records, test_targets, best_val_mse)` — `best_val_mse` explicitement pour qu'un balayage multi-seed sélectionne le meilleur run par score **val**, jamais par score test.

### 4.16 `train_color_context.py`

Pipeline expérimental — contexte d'attention entraînable par bucket (set, couleur primaire), avec support DANN. **Non adopté** dans la config actuelle, gardé pour référence/reprise éventuelle.

- **`_bucket_forward(model, color_context, records, bucket, targets_by_index, device)`** (`train_color_context.py:58`) : une passe avant pour un seul bucket — encode toutes les cartes du bucket (requêtes), fait tourner l'attention (`color_context`) avec le sous-ensemble commune/peu-commune comme clés/valeurs, calcule la prédiction de la tête pour toutes les cartes puis ne garde que celles ayant une cible dans `targets_by_index`. Retourne `None` si aucune carte du bucket n'a de cible (bucket entièrement val/test lors d'un passage train, etc.). Pas de backward ici. Appelée par `run_epoch` et `evaluate`.
- **`run_epoch(model, color_context, records, buckets, targets_by_index, device, optimizer, rng, dann=None, dann_chunk_size=1)`** (`train_color_context.py:108`) : mélange l'ordre des buckets, groupe `dann_chunk_size` buckets par step d'optimisation (1 = comportement historique, identique bit-à-bit sans DANN), agrège la loss MSE sur le groupe, ajoute la loss adversariale DANN si activée (nécessite plusieurs buckets par step car un bucket seul n'a qu'une seule classe de domaine — dégénérescence documentée dans le docstring), fait un seul `backward()`/`step()` par groupe. Appelée par `main` à chaque epoch.
- **`evaluate(model, color_context, records, buckets, targets_by_index, device)`** (`train_color_context.py:176`) : passe complète sans gradient sur tous les buckets, agrège toutes les prédictions, appelle `train_lora.score_predictions` pour le calcul final. Appelée par `main` (val à chaque epoch, test à la fin).
- `save_checkpoint(model, color_context, mu, sigma, formula, checkpoint_dir=None, base_model_path=MODEL_NAME)` (`train_color_context.py:192`) : comme `train_lora.save_checkpoint`, plus `color_context_state_dict`/`context_dim` (le module d'attention n'existe pas dans le pipeline principal). Appelée par `main` si `save=True`.
- **`main(...)`** (`train_color_context.py:211`) : construit le dataset, calcule les buckets (`color_context.build_color_buckets` + `flatten_buckets`), split (`split_by_name_3way` ou `split_by_set_3way`, importés de `train_lora.py`), construit `CardRatingNetJoint` + `ColorAttentionContext` avec 3 (ou 4 avec DANN) groupes de LR séparés, boucle avec ramp DANN classique (`2/(1+e^{-10·progress}) - 1`, Ganin & Lempitsky) si `use_dann=True`, early stopping identique à `train_lora.main`, test unique à la fin. Contrairement à `train_lora.main`, `save` vaut `False` par défaut ici — un appel sans `save=True` explicite n'écrit donc rien sur disque (cohérent avec le statut expérimental/non adopté de ce pipeline, voir §3.5).

### 4.17 `rate_set.py`

Inférence + affichage seulement, **aucun entraînement**. Charge un checkpoint sauvegardé, note toutes les cartes d'un set, produit une page HTML.

- `display_group(card)` (`rate_set.py:42`) : groupement **d'affichage** (délibérément différent de `color_context.primary_color`, qui n'est pas utilisée ici) — mono-couleur va sous sa couleur, multicolore et incolore ont chacun leur propre section (au lieu d'être rattachés à une première couleur, plus intuitif pour un lecteur humain). Appelée par `main`.
- `_usable_cards(set_code)` (`rate_set.py:70`) : cartes du set (`fetch_scryfall.fetch_set_cards`) moins les terrains de base et les cartes sans rareté reconnue. Appelée par `main`.
- **`_context_vector_for_set(set_code, cards)`** (`rate_set.py:75`) : réutilise le cache partagé de `set_context.py` si `set_code` fait partie des 26 sets déjà poolés (le vecteur exact vu à l'entraînement) ; sinon calcule la même moyenne à la volée, **sans jamais écrire** dans ce cache partagé (une écriture partielle d'un seul set le corromprait silencieusement pour tous les autres scripts qui lui font confiance). Appelée par `main`.
- **`load_model(checkpoint_dir=CHECKPOINT_DIR)`** (`rate_set.py:93`) : charge `head.pt`, détecte le format (liste `extra_formulas` actuelle, ou ancien `second_formula` unique), reconstruit `CardRatingNetJoint` avec les bonnes dimensions, recharge l'adaptateur LoRA sur une base fraîche (`PeftModel.from_pretrained` sur un `AutoModel` neuf, plutôt que de déballer l'adaptateur aléatoire déjà construit par `CardRatingNetJoint.__init__` — évite de dépendre des internals `peft`). Retourne `(model, formulas, use_set_context, norm_params)`. Appelée par `main`.
- **`rate_cards(model, formulas, cards, context_vector=None)`** (`rate_set.py:131`, `@torch.no_grad()`) : batch par `BATCH_SIZE=32`, appelle `model(...)`, associe chaque formule à sa prédiction par carte. Appelée par `main`.
- **`attach_actual_ratings(ratings, set_code, formulas, norm_params)`** (`rate_set.py:147`) : mute `ratings` en place — ajoute `f"{formula}_actual"` (le vrai résultat 17Lands, sur la même échelle 0-10) pour toute carte où `get_set_metrics` a des données, laisse silencieusement absent sinon (set inédit ou jamais couvert par 17Lands). Appelée par `main`.
- **`render_html(set_code, by_color, formulas, primary_metric)`** (`rate_set.py:176`) : construit la page HTML complète (CSS inline, une section par couleur, tableau trié, moyenne de couleur en pied de tableau, couleur de rareté). Appelée par `main`.
- **`main(set_code, checkpoint_dir=CHECKPOINT_DIR, out_path=None)`** (`rate_set.py:261`) : enchaîne toutes les fonctions ci-dessus, trie chaque groupe de couleur du plus fort au plus faible par la métrique primaire (`gp_wr_only` si présente, sinon la première formule), écrit `data/ratings/ratings_<set>.html`. Le `if __name__ == '__main__': main("DFT")` en bas de fichier explique pourquoi `CLAUDE.md` dit d'éditer cette ligne pour changer de set par défaut en lancement `-m`.

### 4.18 `smoke_test.py`

Test de fumée bout-en-bout — la commande la plus rapide pour vérifier que toute la chaîne (hors LoRA, hors multi-set) fonctionne encore après une modification.

- **`main()`** (`smoke_test.py:27`) : télécharge le `game_data` d'un seul set (`ECL`), calcule les métriques (`labels.compute_card_metrics`), récupère les cartes Scryfall correspondantes, joint les deux, calcule les features (`features.card_to_features`) pour les cartes du set **et** pour 3 cartes evergreen hors set (`UNSEEN_TEST_CARDS`), puis pour **chaque formule** de `ratings.RAW_SCORE_FORMULAS` : calcule les notes (`ratings.compute_ratings`), entraîne un modèle jouet (`model.train_toy`), affiche la prédiction sur les 3 cartes inconnues. Ne fait ni split train/test ni early stopping — l'objectif est seulement de prouver que le pipeline s'exécute sans exception, pas de mesurer une vraie généralisation (contrairement à `train_multiset.py`/`train_lora.py`).
