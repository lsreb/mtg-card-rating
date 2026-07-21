# MTG Card Rating

Réseau de neurones qui prend en entrée les caractéristiques d'une carte Magic: The
Gathering (coût, couleurs, type, rareté...) et renvoie une note de force, entraîné sur
les win rates de limited/draft publiés par 17Lands. Objectif : généraliser à des
cartes inconnues, y compris d'autres sets que celui utilisé à l'entraînement.

## Sources de données

- [17Lands Public Datasets](https://www.17lands.com/public_datasets) — win rates de
  parties de draft (`game_data`), utilisés comme cible d'entraînement (GIH WR : win
  rate parmi les parties où la carte a été piochée). Données publiées par 17Lands pour
  un usage recherche/communauté.
- [Scryfall API](https://scryfall.com/docs/api) — caractéristiques des cartes (coût,
  couleurs, type, rareté, texte...), utilisées comme features d'entrée du réseau.

## Environnement

Ce projet réutilise l'environnement conda `env_coinche` (déjà équipé de `numpy` et
`torch`), avec deux dépendances ajoutées :

```
conda run -n env_coinche pip install requests pandas
```

## Statut

v0 : pipeline de bout en bout volontairement minimal ("mauvais" réseau), sur les
données d'un seul set (ECL, format PremierDraft) — objectif : valider que le code
tourne (téléchargement, features, entraînement, inférence sur cartes inconnues d'un
autre set), pas d'obtenir un bon modèle.
