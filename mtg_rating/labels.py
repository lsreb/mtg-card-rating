"""Compute per-card GIH win rate (games-in-hand win rate) from a 17Lands game_data file.

Schema confirmed empirically on game_data_public.ECL.PremierDraft.csv.gz: one `won`
column per game row, plus five per-card columns (`opening_hand_<CARD>`,
`drawn_<CARD>`, `tutored_<CARD>`, `deck_<CARD>`, `sideboard_<CARD>`) repeated for every
card in the set. GIH WR = win rate among games where the card was ever in hand
(opening hand or drawn later).
"""

import pandas as pd

OPENING_HAND_PREFIX = "opening_hand_"
DRAWN_PREFIX = "drawn_"


def compute_gih_winrate(game_data_path, min_games: int = 200) -> dict:
    df = pd.read_csv(game_data_path)
    won = df["won"].astype(bool)

    card_names = sorted(
        col[len(OPENING_HAND_PREFIX):]
        for col in df.columns
        if col.startswith(OPENING_HAND_PREFIX)
    )

    results = {}
    for name in card_names:
        ever_drawn = (df[f"{OPENING_HAND_PREFIX}{name}"] > 0) | (df[f"{DRAWN_PREFIX}{name}"] > 0)
        game_count = int(ever_drawn.sum())
        if game_count < min_games:
            continue
        results[name] = {
            "gih_wr": float(won[ever_drawn].mean()),
            "game_count": game_count,
        }
    return results
