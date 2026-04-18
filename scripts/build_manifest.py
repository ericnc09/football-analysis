#!/usr/bin/env python3
"""
build_manifest.py
-----------------
Generate `data/processed/MANIFEST.json` — the source of truth for shot-graph
data integrity.

Why this exists
---------------
Addresses reviews/04_ml_engineer_review.md §7.3 (no way to tell if a shot-graph
`.pt` on HF Hub matches the code that consumes it). The app's bootstrap block
(app.py:138) already knows how to read this file and will sha256-verify every
listed artefact on startup. All we have to do is produce it.

Output schema
-------------
    {
      "built_at":   "2026-04-17T12:03:41+00:00",
      "git_sha":    "a1b2c3d",
      "git_dirty":  false,
      "files": {
        "statsbomb_wc2022_shot_graphs.pt": {
          "sha256":     "c0ffee…",
          "size_bytes": 12345678,
          "mtime":      "2026-04-17T11:55:02+00:00"
        },
        ...
      }
    }

Which files get manifested
--------------------------
By default we glob `data/processed/*.pt` — shot-graph files and model
checkpoints both. Pass `--include-pattern` / `--exclude-pattern` to narrow.
Large-but-transient files (e.g. per-epoch checkpoints) should be excluded
via `.manifest-ignore` if needed.

Run
---
    python scripts/build_manifest.py                     # default
    python scripts/build_manifest.py --include "*.pt"    # just PyTorch files
    python scripts/build_manifest.py --dry-run           # print, don't write
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.reproducibility import git_dirty, git_sha  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "MANIFEST.json"

# Files whose bytes change every training run (temperature scalars fit in a
# handful of kB and get re-saved with every calibrate pass, so their hash
# drifting isn't a tamper signal — skip them to keep the manifest stable).
DEFAULT_EXCLUDE: list[str] = [
    "MANIFEST.json",          # never self-referential
    "*.meta.json",            # sidecars have their own sha256 field pointing at the .pt
    "pool_7comp_per_comp_T_*.pt",  # per-competition T dicts, re-fit frequently
]

DEFAULT_INCLUDE: list[str] = ["*.pt", "*.npz"]


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _iso(ts: float) -> str:
    return (
        datetime.fromtimestamp(ts, tz=timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def collect_files(
    root: Path,
    include: list[str],
    exclude: list[str],
) -> list[Path]:
    """Return sorted list of files under `root` matching include minus exclude."""
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        name = p.name
        if not _matches_any(name, include):
            continue
        if _matches_any(name, exclude):
            continue
        out.append(p)
    return out


def build_manifest(
    root: Path,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict:
    include = include or DEFAULT_INCLUDE
    exclude = exclude or DEFAULT_EXCLUDE

    files = collect_files(root, include, exclude)
    manifest_files: dict[str, dict] = {}
    for p in files:
        stat = p.stat()
        manifest_files[p.name] = {
            "sha256":     _sha256(p),
            "size_bytes": stat.st_size,
            "mtime":      _iso(stat.st_mtime),
        }

    return {
        "built_at":  datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_sha":   git_sha(short=True),
        "git_dirty": git_dirty(),
        "files":     manifest_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument(
        "--root", type=Path, default=PROCESSED_DIR,
        help=f"Directory to manifest. Default: {PROCESSED_DIR}",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write MANIFEST.json. Default: <root>/MANIFEST.json",
    )
    parser.add_argument(
        "--include", action="append", default=None,
        help=f"Glob pattern(s) to include. Repeatable. Default: {DEFAULT_INCLUDE}",
    )
    parser.add_argument(
        "--exclude", action="append", default=None,
        help=f"Glob pattern(s) to exclude. Repeatable. Default: {DEFAULT_EXCLUDE}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the manifest to stdout instead of writing to disk.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    manifest = build_manifest(root, include=args.include, exclude=args.exclude)
    out_path = args.output or (root / "MANIFEST.json")

    n = len(manifest["files"])
    if args.dry_run:
        json.dump(manifest, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        print(f"\n[dry-run] Would write manifest for {n} file(s) to {out_path}",
              file=sys.stderr)
        return 0

    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"Wrote MANIFEST for {n} file(s) → {out_path}")
    if manifest["git_dirty"]:
        print("  WARNING: working tree is dirty — manifest sha references an "
              "uncommitted state. Commit before publishing to HF Hub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
