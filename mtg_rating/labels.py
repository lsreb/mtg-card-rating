"""Compute per-card metrics from a 17Lands game_data file: GIH win rate, IWD, and
play rate.

Schema confirmed empirically on game_data_public.ECL.PremierDraft.csv.gz: one `won`
column per game row, plus five per-card columns repeated for every card in the set:
`opening_hand_<CARD>`, `drawn_<CARD>`, `tutored_<CARD>`, `deck_<CARD>`,
`sideboard_<CARD>`. Also present: `draft_id` + `build_index`, which together identify
one deck build (a player may submit several builds across a run; deck/sideboard
membership is constant across the games played with a given build).

Metrics:
- gih_wr: win rate among games where the card was ever in hand (opening hand or
  drawn later).
- gnd_wr: win rate among games where the card was in the deck but never drawn.
- iih (a.k.a. IWD, "improvement when drawn" in 17Lands' own terminology):
  gih_wr - gnd_wr. Isolates the card's marginal impact from the deck's overall
  strength, since gnd_wr already reflects "how good is this deck without this card
  showing up".
- play_rate: among unique deck builds where the card was in the pool (in the deck OR
  the sideboard), the fraction where it was actually maindecked. A proxy for how
  often the card is considered worth playing when available.
"""

import pandas as pd

OPENING_HAND_PREFIX = "opening_hand_"
DRAWN_PREFIX = "drawn_"
DECK_PREFIX = "deck_"
SIDEBOARD_PREFIX = "sideboard_"


def compute_card_metrics(game_data_path, min_games: int = 200) -> dict:
    df = pd.read_csv(game_data_path)
    won = df["won"].astype(bool)

    card_names = sorted(
        col[len(OPENING_HAND_PREFIX):]
        for col in df.columns
        if col.startswith(OPENING_HAND_PREFIX)
    )

    # One row per unique deck build: deck/sideboard membership doesn't change across
    # the games played with the same build, so this avoids over-weighting play rate
    # by how many games a given build happened to play (e.g. Bo3 vs Bo1).
    builds = df.drop_duplicates(subset=["draft_id", "build_index"])

    results = {}
    for name in card_names:
        ever_drawn = (df[f"{OPENING_HAND_PREFIX}{name}"] > 0) | (df[f"{DRAWN_PREFIX}{name}"] > 0)
        in_deck = df[f"{DECK_PREFIX}{name}"] > 0
        never_drawn = in_deck & ~ever_drawn

        gih_count = int(ever_drawn.sum())
        if gih_count < min_games:
            continue
        gnd_count = int(never_drawn.sum())

        gih_wr = float(won[ever_drawn].mean())
        gnd_wr = float(won[never_drawn].mean()) if gnd_count > 0 else None

        build_in_deck = builds[f"{DECK_PREFIX}{name}"] > 0
        build_in_sideboard = builds[f"{SIDEBOARD_PREFIX}{name}"] > 0
        pool_count = int((build_in_deck | build_in_sideboard).sum())
        play_rate = float(build_in_deck.sum() / pool_count) if pool_count > 0 else None

        results[name] = {
            "gih_wr": gih_wr,
            "gih_count": gih_count,
            "gnd_wr": gnd_wr,
            "gnd_count": gnd_count,
            "iih": (gih_wr - gnd_wr) if gnd_wr is not None else None,
            "play_rate": play_rate,
            "pool_count": pool_count,
        }
    return results
