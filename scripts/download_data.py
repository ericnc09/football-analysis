#!/usr/bin/env python3
"""
download_data.py
-----------------
Download free football datasets for the GNN pipeline.

Datasets:
  metrica  — Metrica Sports sample tracking data (Games 1-2, CSV format)
  statsbomb — StatsBomb 360 open data is fetched on-the-fly by build_statsbomb_graphs.py
              (no pre-download needed)

Usage:
    python scripts/download_data.py --metrica        # Download Metrica Games 1 & 2
    python scripts/download_data.py --metrica --game 2  # Download only Game 2
    python scripts/download_data.py --all            # Download everything
"""

import sys
import argparse
import requests
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent

METRICA_BASE = (
    "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data"
)

METRICA_FILES = {
    1: [
        "Sample_Game_1/Sample_Game_1_RawEventsData.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
    ],
    2: [
        "Sample_Game_2/Sample_Game_2_RawEventsData.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
    ],
}


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file if it doesn't exist (or force=True). Returns True if downloaded."""
    if dest.exists() and not force:
        size = dest.stat().st_size
        print(f"  [skip]  {dest.name}  ({size/1024:.0f} KB already present)")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [fetch] {dest.name} ... ", end="", flush=True)

    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=65536):
                fh.write(chunk)
        size = dest.stat().st_size
        print(f"OK  ({size/1024:.0f} KB)")
        return True
    except requests.exceptions.SSLError as e:
        print(f"FAILED (SSL) — {e}")
        raise  # cert failures must not be silently swallowed
    except (requests.exceptions.RequestException, OSError) as e:
        print(f"FAILED — {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_metrica(games: list[int], force: bool = False):
    """Download Metrica CSV tracking data for specified games."""
    base_dir = REPO_ROOT / "data" / "raw" / "metrica" / "data"
    print(f"\n{'='*60}")
    print(f"Metrica Sports — Games {games}")
    print(f"Destination: {base_dir}")
    print(f"{'='*60}")

    for game_num in games:
        if game_num not in METRICA_FILES:
            print(f"  WARNING: No download config for Game {game_num}")
            continue

        print(f"\n  Game {game_num}:")
        for rel_path in METRICA_FILES[game_num]:
            url = f"{METRICA_BASE}/{rel_path}"
            dest = base_dir / rel_path
            download_file(url, dest, force=force)

    print(f"\nDone. Run next:")
    for g in games:
        print(f"  python scripts/build_graphs.py --game {g}")


def main():
    parser = argparse.ArgumentParser(description="Download football datasets")
    parser.add_argument("--metrica", action="store_true", help="Download Metrica tracking data")
    parser.add_argument("--game", type=int, nargs="+", default=[1, 2],
                        help="Which Metrica games to download (default: 1 2)")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    args = parser.parse_args()

    if args.all or args.metrica:
        download_metrica(games=args.game, force=args.force)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
