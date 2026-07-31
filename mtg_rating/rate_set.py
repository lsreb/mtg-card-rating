"""Rate every card of one Magic set and render the result as a standalone HTML
page, grouped by primary color, strongest to weakest (premier_jet.md section 5's
original display idea).

Needs only Scryfall data for the target set -- no 17Lands labels -- so it works
on a set with no draft history yet, not just the 26 sets in multiset.SET_CODES
that train_lora.py trains on. Uses the dual iih_only/gp_wr_only checkpoint at
data/models/lora_joint_dual/ (CLAUDE.md's "Current best-known config") by
default; pass a different checkpoint_dir to rate with another one.

Run with: conda run -n env_coinche python -m mtg_rating.rate_set
"""

import html
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel

from mtg_rating.color_context import primary_color
from mtg_rating.features import RARITIES, STRUCTURED_DIM, structured_features
from mtg_rating.fetch_scryfall import fetch_set_cards
from mtg_rating.labels import BASIC_LAND_NAMES
from mtg_rating.model_joint import CardRatingNetJoint
from mtg_rating.set_context import CACHE_PATH as SET_CONTEXT_CACHE_PATH, CONTEXT_DIM, _mean_vector
from mtg_rating.text_embeddings import embed_texts

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "lora_joint_dual"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "ratings"
BATCH_SIZE = 32
COLOR_ORDER = ["W", "U", "B", "R", "G", "colorless"]
COLOR_NAMES = {
    "W": "White", "U": "Blue", "B": "Black", "R": "Red", "G": "Green",
    "colorless": "Colorless / multicolor (grouped by first color, see color_context.primary_color)",
}
METRIC_LABELS = {
    "iih_only": "IIH", "gp_wr_only": "GP WR", "gih_iih": "GIH+IIH",
    "gp_iih": "GP+IIH", "gih": "GIH", "play_rate_only": "Play Rate",
}


def _usable_cards(set_code: str) -> list:
    cards = fetch_set_cards(set_code)
    return [c for c in cards if c["name"] not in BASIC_LAND_NAMES and c.get("rarity") in RARITIES]


def _context_vector_for_set(set_code: str, cards: list) -> list:
    # Reuse set_context.py's shared cache when this set is already one of the
    # 26 pooled training sets -- the exact vector the checkpoint trained
    # against. For any other set (the actual not-yet-released-set use case),
    # compute the same whole-set mean fresh instead of writing into that
    # shared cache file: a partial, single-set write would silently poison it
    # for every other script that trusts it as "the full pool" (see
    # set_context.py's own cache-staleness warning).
    if SET_CONTEXT_CACHE_PATH.exists():
        cached = json.loads(SET_CONTEXT_CACHE_PATH.read_text())
        if set_code in cached:
            return cached[set_code]

    structured = [structured_features(c) for c in cards]
    text_embeds = embed_texts([c.get("oracle_text", "") or "" for c in cards]).tolist()
    return _mean_vector([s + t for s, t in zip(structured, text_embeds)])


def load_model(checkpoint_dir: Path = CHECKPOINT_DIR):
    ckpt = torch.load(checkpoint_dir / "head.pt", map_location="cpu", weights_only=False)
    if ckpt.get("extra_formulas"):
        extra_formulas = ckpt["extra_formulas"]
    elif "second_formula" in ckpt:
        # Older single-second-target layout (lora_joint_dual/head.pt was saved
        # before save_checkpoint() generalized to a list of extra_formulas) --
        # still the same trained dual-output config, just an earlier
        # serialization shape.
        extra_formulas = [ckpt["second_formula"]]
    else:
        extra_formulas = []
    formulas = [ckpt["formula"]] + extra_formulas
    use_set_context = bool(ckpt.get("use_set_context"))

    model = CardRatingNetJoint(
        structured_dim=STRUCTURED_DIM,
        set_context_dim=CONTEXT_DIM if use_set_context else 0,
        output_dim=len(formulas),
        base_model_path=ckpt["base_model_path"],
    )
    # Fresh base model rather than unwrapping the one CardRatingNetJoint just
    # built (which carries a randomly-initialized adapter of its own) --
    # avoids depending on peft's adapter-unwrap internals.
    fresh_base = AutoModel.from_pretrained(ckpt["base_model_path"])
    model.text_encoder.model = PeftModel.from_pretrained(fresh_base, str(checkpoint_dir / "lora_adapter"))
    model.head.load_state_dict(ckpt["head_state_dict"])
    model.eval()
    return model, formulas, use_set_context


@torch.no_grad()
def rate_cards(model, formulas: list, cards: list, context_vector: list = None) -> list:
    ratings = []
    for start in range(0, len(cards), BATCH_SIZE):
        batch = cards[start : start + BATCH_SIZE]
        structured = torch.tensor([structured_features(c) for c in batch], dtype=torch.float32)
        texts = [c.get("oracle_text", "") or "" for c in batch]
        set_context = torch.tensor([context_vector] * len(batch), dtype=torch.float32) if context_vector is not None else None
        preds = model(structured, texts, set_context)
        preds = preds.tolist()
        for card, pred in zip(batch, preds):
            values = pred if isinstance(pred, list) else [pred]
            ratings.append({"card": card, **dict(zip(formulas, values))})
    return ratings


def render_html(set_code: str, by_color: dict, formulas: list, primary_metric: str) -> str:
    metric_cols = "".join(f"<th>{html.escape(METRIC_LABELS.get(f, f))}</th>" for f in formulas)
    sections = []
    for color in COLOR_ORDER:
        if color not in by_color:
            continue
        rows = []
        for rank, entry in enumerate(by_color[color], start=1):
            card = entry["card"]
            value_cells = "".join(
                f'<td class="metric{" primary" if f == primary_metric else ""}">{entry[f]:.2f}</td>' for f in formulas
            )
            rows.append(
                "<tr>"
                f"<td class=\"rank\">{rank}</td>"
                f"<td class=\"name\">{html.escape(card['name'])}</td>"
                f"<td class=\"cost\">{html.escape(card.get('mana_cost') or '')}</td>"
                f"<td class=\"type\">{html.escape(card.get('type_line') or '')}</td>"
                f"<td class=\"rarity\">{html.escape(card.get('rarity') or '')}</td>"
                f"{value_cells}"
                "</tr>"
            )
        avg_primary = sum(e[primary_metric] for e in by_color[color]) / len(by_color[color])
        footer_cells = "".join(
            f'<td class="metric primary">{avg_primary:.2f}</td>' if f == primary_metric else '<td class="metric"></td>'
            for f in formulas
        )
        footer = (
            '<tfoot><tr class="avg-row">'
            '<td></td><td class="name">Moyenne couleur</td><td></td><td></td><td></td>'
            f"{footer_cells}</tr></tfoot>"
        )
        sections.append(
            f'<section><h2 id="{color}">{html.escape(COLOR_NAMES[color])} ({len(rows)})</h2>'
            f'<table><thead><tr><th>#</th><th>Name</th><th>Cost</th><th>Type</th><th>Rarity</th>{metric_cols}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody>{footer}</table></section>"
        )

    nav = " ".join(
        f'<a href="#{c}">{html.escape(COLOR_NAMES[c].split(" ")[0])}</a>' for c in COLOR_ORDER if c in by_color
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{html.escape(set_code)} card ratings</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2rem; background: #fafafa; color: #222; }}
h1 {{ margin-bottom: 0.2rem; }}
.subtitle {{ color: #666; margin-top: 0; }}
nav {{ margin: 1rem 0; }}
nav a {{ margin-right: 1rem; font-weight: 600; text-decoration: none; color: #2a4; }}
table {{ border-collapse: collapse; width: 100%; margin-bottom: 2rem; background: white; }}
th, td {{ padding: 0.35rem 0.6rem; border-bottom: 1px solid #ddd; text-align: left; font-size: 0.92rem; }}
th {{ background: #eee; }}
td.rank, td.cost, td.rarity {{ color: #666; white-space: nowrap; }}
td.metric {{ text-align: right; font-variant-numeric: tabular-nums; }}
td.metric.primary {{ font-weight: 700; }}
tr:nth-child(even) {{ background: #f5f5f5; }}
tfoot .avg-row td {{ border-top: 2px solid #999; border-bottom: none; font-style: italic; background: #eef4ee; }}
</style></head>
<body>
<h1>{html.escape(set_code)} card ratings</h1>
<p class="subtitle">0-10 scale, 5 = set average (see ratings.py). GP WR = predicted contextual
win-rate-based rating (bolded, the practical pick-order signal). IIH = predicted decontextualized
"raw power" rating (less noisy, ignores archetype/environment fit). Grouped by primary color,
strongest to weakest.</p>
<nav>{nav}</nav>
{''.join(sections)}
</body></html>"""


def main(set_code: str, checkpoint_dir: Path = CHECKPOINT_DIR, out_path: Path = None):
    cards = _usable_cards(set_code)
    print(f"[rate_set] {set_code}: {len(cards)} usable cards")
    context_vector = _context_vector_for_set(set_code, cards)

    model, formulas, use_set_context = load_model(checkpoint_dir)
    ratings = rate_cards(model, formulas, cards, context_vector if use_set_context else None)

    by_color = {}
    for entry in ratings:
        by_color.setdefault(primary_color(entry["card"]), []).append(entry)

    primary_metric = "gp_wr_only" if "gp_wr_only" in formulas else formulas[0]
    for color_ratings in by_color.values():
        color_ratings.sort(key=lambda e: e[primary_metric], reverse=True)

    out_path = out_path or (OUT_DIR / f"ratings_{set_code.lower()}.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(set_code, by_color, formulas, primary_metric))
    print(f"[rate_set] wrote {out_path}")
    return by_color


if __name__ == "__main__":
    main("DFT")
