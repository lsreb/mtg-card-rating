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
- gns_wr ("games not seen"): win rate among games where the card was in the deck but
  never showed up in hand at all (neither opening hand nor drawn). Note: 17Lands'
  own public field for this is misleadingly named `never_drawn_win_rate` -- despite
  the name, it's defined as the complement of `ever_drawn` (which includes opening
  hand), i.e. it really means "never seen", not narrowly "never drawn from the
  library". Named gns_wr/gns_count here to avoid that ambiguity.
- iih (a.k.a. IWD, "improvement when drawn" in 17Lands' own terminology):
  gih_wr - gns_wr. Isolates the card's marginal impact from the deck's overall
  strength, since gns_wr already reflects "how good is this deck without this card
  showing up".
- gp_wr ("games played"): win rate among ALL games where the card was in the deck,
  drawn or not. With p = gih_count / (gih_count + gns_count) (fraction of games the
  card was actually seen), gp_wr = p*gih_wr + (1-p)*gns_wr, which gives the exact
  identity gih_wr = gp_wr + (1-p)*iih -- gp_wr, gih_wr and iih alone are enough to
  recover (1-p) (= (gih_wr - gp_wr) / iih) if ever needed, so it isn't stored
  separately here. gp_wr is dominated by the gns population for a typical card (only
  seen in ~40-50% of its games), so it's a *more* diluted/context-heavy signal than
  gih_wr, not a bias-reduced one -- see project memory for why gp_wr was considered
  and set aside as a worse target than gih_wr/iih for isolating card-level impact.
- play_rate: among unique deck builds where the card was in the pool (in the deck OR
  the sideboard), the fraction where it was actually maindecked. A proxy for how
  often the card is considered worth playing when available.
"""

import pandas as pd

OPENING_HAND_PREFIX = "opening_hand_"
DRAWN_PREFIX = "drawn_"
DECK_PREFIX = "deck_"
SIDEBOARD_PREFIX = "sideboard_"

# Basic land names are fixed across all Magic sets (no set has ever added a new one
# in decades of design) -- excluded everywhere, not just for visualization: their
# draw dynamics (~100% play rate, huge sample counts, no real "power level") don't
# belong in the same training/validation population as spells.
BASIC_LAND_NAMES = {"Plains", "Island", "Swamp", "Mountain", "Forest", "Wastes"}


def compute_card_metrics(game_data_path, min_games: int = 200) -> dict:
    # game_data files are very wide (hundreds to thousands of per-card columns).
    # Loading the whole thing with pandas' default dtypes OOM-killed a run on a
    # bigger set (FDN) even though smaller sets (ECL) had been fine -- read only the
    # columns this function actually uses (skip `tutored_*` and other metadata) and
    # downcast the per-card count columns to int8 (same fix applied to draft_data
    # loading earlier; game_data needed it too, as flagged but not yet done back then).
    header = pd.read_csv(game_data_path, nrows=0)
    card_cols = [
        col for col in header.columns
        if col.startswith((OPENING_HAND_PREFIX, DRAWN_PREFIX, DECK_PREFIX, SIDEBOARD_PREFIX))
    ]
    usecols = ["won", "draft_id", "build_index"] + card_cols
    dtype = {col: "int8" for col in card_cols}
    dtype["draft_id"] = "category"

    df = pd.read_csv(game_data_path, usecols=usecols, dtype=dtype)
    won = df["won"].astype(bool)

    card_names = sorted(
        col[len(OPENING_HAND_PREFIX):]
        for col in df.columns
        if col.startswith(OPENING_HAND_PREFIX) and col[len(OPENING_HAND_PREFIX):] not in BASIC_LAND_NAMES
    )

    # One row per unique deck build: deck/sideboard membership doesn't change across
    # the games played with the same build, so this avoids over-weighting play rate
    # by how many games a given build happened to play (e.g. Bo3 vs Bo1).
    builds = df.drop_duplicates(subset=["draft_id", "build_index"])

    results = {}
    for name in card_names:
        ever_seen = (df[f"{OPENING_HAND_PREFIX}{name}"] > 0) | (df[f"{DRAWN_PREFIX}{name}"] > 0)
        in_deck = df[f"{DECK_PREFIX}{name}"] > 0
        never_seen = in_deck & ~ever_seen

        gih_count = int(ever_seen.sum())
        if gih_count < min_games:
            continue
        gns_count = int(never_seen.sum())

        gih_wr = float(won[ever_seen].mean())
        gns_wr = float(won[never_seen].mean()) if gns_count > 0 else None
        gp_wr = float(won[in_deck].mean())

        build_in_deck = builds[f"{DECK_PREFIX}{name}"] > 0
        build_in_sideboard = builds[f"{SIDEBOARD_PREFIX}{name}"] > 0
        pool_count = int((build_in_deck | build_in_sideboard).sum())
        play_rate = float(build_in_deck.sum() / pool_count) if pool_count > 0 else None

        results[name] = {
            "gih_wr": gih_wr,
            "gih_count": gih_count,
            "gns_wr": gns_wr,
            "gns_count": gns_count,
            "iih": (gih_wr - gns_wr) if gns_wr is not None else None,
            "gp_wr": gp_wr,
            "play_rate": play_rate,
            "pool_count": pool_count,
        }
    return results
