"""End-to-end pipeline smoke test: real data in, features out, toy training, then
inference on genuinely unseen cards from other sets.

Not meant to produce a good model -- only to prove the plumbing (fetch -> features ->
train -> predict on novel input) runs without errors, for each of the three candidate
target formulas in ratings.py. Run with:
    conda run -n env_coinche python -m mtg_rating.smoke_test
"""

import torch

from mtg_rating.fetch_17lands import download_game_data
from mtg_rating.fetch_scryfall import fetch_set_cards, fetch_card
from mtg_rating.features import card_to_features
from mtg_rating.labels import compute_card_metrics
from mtg_rating.model import train_toy
from mtg_rating.ratings import RAW_SCORE_FORMULAS, compute_ratings

SET_CODE = "ECL"
EVENT_TYPE = "PremierDraft"

# Old, evergreen cards guaranteed to exist on Scryfall regardless of the current
# Standard-legal set -- used only to check the model runs on unseen inputs.
UNSEEN_TEST_CARDS = ["Lightning Bolt", "Llanowar Elves", "Counterspell"]


def main():
    game_data_path = download_game_data(SET_CODE, EVENT_TYPE)
    metrics = compute_card_metrics(game_data_path)
    print(f"[labels] {len(metrics)} cards with enough games in {SET_CODE}/{EVENT_TYPE}")

    set_cards = fetch_set_cards(SET_CODE)
    cards_by_name = {card["name"]: card for card in set_cards}
    print(f"[scryfall] {len(cards_by_name)} cards fetched for set {SET_CODE}")

    names = [name for name in metrics if metrics[name].get("iih") is not None and name in cards_by_name]
    print(f"[join] {len(names)} cards with both a label and Scryfall features")

    features = [card_to_features(cards_by_name[name]) for name in names]
    unseen_cards = {name: fetch_card(name) for name in UNSEEN_TEST_CARDS}
    unseen_features = {name: card_to_features(card) for name, card in unseen_cards.items()}

    for formula in RAW_SCORE_FORMULAS:
        ratings = compute_ratings(metrics, formula)
        targets = [ratings[name] for name in names]

        print(f"\n[train:{formula}] training toy model on 0-10 rating...")
        model = train_toy(features, targets)
        print(f"[train:{formula}] done")

        print(f"[inference:{formula}] predicted rating for unseen cards from other sets:")
        for name, vec in unseen_features.items():
            pred = model(torch.tensor([vec], dtype=torch.float32)).item()
            set_name = unseen_cards[name].get("set_name")
            print(f"  {name!r} ({set_name}) -> {pred:.2f} / 10")


if __name__ == "__main__":
    main()
