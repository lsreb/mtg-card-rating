"""Download raw 17Lands Public Dataset files (game_data), with local caching."""

from pathlib import Path

import requests

BASE_URL = "https://17lands-public.s3.amazonaws.com/analysis_data/game_data"


def download_game_data(set_code: str, event_type: str = "PremierDraft", dest_dir: Path = None) -> Path:
    dest_dir = Path(dest_dir) if dest_dir else Path(__file__).resolve().parent.parent / "data" / "raw"
    dest_dir.mkdir(parents=True, exist_ok=True)

    filename = f"game_data_public.{set_code}.{event_type}.csv.gz"
    dest_path = dest_dir / filename
    if dest_path.exists():
        return dest_path

    url = f"{BASE_URL}/{filename}"
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)

    return dest_path
