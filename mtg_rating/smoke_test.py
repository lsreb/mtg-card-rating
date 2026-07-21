"""End-to-end pipeline smoke test: real data in, features out, toy training, then
inference on genuinely unseen cards from other sets.

Not meant to produce a good model -- only to prove the plumbing (fetch -> features ->
train -> predict on novel input) runs without errors. Run with:
    conda run -n env_coinche python -m mtg_rating.smoke_test
"""

import torch

from mtg_rating.fetch_17lands import download_game_data
from mtg_rating.fetch_scryfall import fetch_set_cards, fetch_card
from mtg_rating.features import card_to_features
from mtg_rating.labels import compute_gih_winrate
from mtg_rating.model import train_toy

SET_CODE = "ECL"
EVENT_TYPE = "PremierDraft"

# Old, evergreen cards guaranteed to exist on Scryfall regardless of the current
# Standard-legal set -- used only to check the model runs on unseen inputs.
UNSEEN_TEST_CARDS = ["Lightning Bolt", "Llanowar Elves", "Counterspell"]


def main():
    game_data_path = download_game_data(SET_CODE, EVENT_TYPE)
    labels = compute_gih_winrate(game_data_path)
    print(f"[labels] {len(labels)} cards with enough games in {SET_CODE}/{EVENT_TYPE}")

    set_cards = fetch_set_cards(SET_CODE)
    cards_by_name = {card["name"]: card for card in set_cards}
    print(f"[scryfall] {len(cards_by_name)} cards fetched for set {SET_CODE}")

    names = [name for name in labels if name in cards_by_name]
    print(f"[join] {len(names)} cards with both a label and Scryfall features")

    features = [card_to_features(cards_by_name[name]) for name in names]
    targets = [labels[name]["gih_wr"] for name in names]

    print("[train] training toy model...")
    model = train_toy(features, targets)
    print("[train] done")

    print("[inference] predicting rating for unseen cards from other sets:")
    for name in UNSEEN_TEST_CARDS:
        card = fetch_card(name)
        vec = card_to_features(card)
        pred = model(torch.tensor([vec], dtype=torch.float32)).item()
        print(f"  {name!r} ({card.get('set_name')}) -> predicted rating: {pred:.4f}")


if __name__ == "__main__":
    main()
