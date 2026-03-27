#!/usr/bin/env python3
"""
upload_to_hub.py
----------------
Upload trained model weights and temperature scalars to a HuggingFace Hub repo.

The app downloads these at startup when running in a cloud environment where
data/processed/ is not present (e.g. Railway, Streamlit Cloud, Cloud Run).

Usage
-----
    # First time — create the repo and upload:
    python scripts/upload_to_hub.py --repo YOUR_HF_USERNAME/football-xg-models --create

    # Subsequent updates:
    python scripts/upload_to_hub.py --repo YOUR_HF_USERNAME/football-xg-models

    # Dry run — list files that would be uploaded:
    python scripts/upload_to_hub.py --repo YOUR_HF_USERNAME/football-xg-models --dry-run

Environment
-----------
    HF_TOKEN   HuggingFace write token (Settings → Access Tokens)
               Can also be passed with --token or stored via `huggingface-cli login`
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent
PROCESSED   = REPO_ROOT / "data" / "processed"

# Files to upload — model weights + calibration scalars
UPLOAD_FILES = [
    "pool_7comp_hybrid_xg.pt",          # HybridGCN weights
    "pool_7comp_hybrid_gat_xg.pt",      # HybridGAT weights
    "pool_7comp_T.pt",                  # GCN global temperature
    "pool_7comp_gat_T.pt",              # GAT global temperature
    "pool_7comp_per_comp_T_gcn.pt",     # GCN per-competition temperatures
    "pool_7comp_per_comp_T_gat.pt",     # GAT per-competition temperatures
    "feature_importance.json",          # Pre-computed feature importance (optional)
]


def parse_args():
    p = argparse.ArgumentParser(description="Upload xG model files to HuggingFace Hub")
    p.add_argument("--repo",    required=True, help="HF repo id, e.g. yourname/football-xg-models")
    p.add_argument("--token",   default=None,  help="HuggingFace write token (or set HF_TOKEN env var)")
    p.add_argument("--create",  action="store_true", help="Create the repo if it doesn't exist")
    p.add_argument("--dry-run", action="store_true", help="Print files that would be uploaded, don't upload")
    return p.parse_args()


def main():
    args = parse_args()
    token = args.token or os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("ERROR: huggingface-hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    # Check which files exist
    to_upload = []
    missing   = []
    for fname in UPLOAD_FILES:
        path = PROCESSED / fname
        if path.exists():
            to_upload.append(path)
        else:
            missing.append(fname)

    if missing:
        print(f"Warning: {len(missing)} file(s) not found and will be skipped:")
        for m in missing:
            print(f"  {m}")

    if not to_upload:
        print("No files to upload. Run train_xg_hybrid.py first.")
        sys.exit(1)

    print(f"\nFiles to upload to {args.repo}:")
    for p in to_upload:
        size_mb = p.stat().st_size / 1e6
        print(f"  {p.name:<45} {size_mb:6.1f} MB")
    total_mb = sum(p.stat().st_size for p in to_upload) / 1e6
    print(f"  {'TOTAL':<45} {total_mb:6.1f} MB")

    if args.dry_run:
        print("\n[dry-run] Not uploading.")
        return

    if not token:
        print("\nERROR: No HuggingFace token. Pass --token or set HF_TOKEN env var.")
        print("       Get a write token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    api = HfApi(token=token)

    if args.create:
        print(f"\nCreating repo {args.repo} ...")
        create_repo(args.repo, token=token, repo_type="model", exist_ok=True)
        print("  Created (or already exists).")

    print(f"\nUploading to {args.repo} ...")
    for path in to_upload:
        print(f"  Uploading {path.name} ...", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path.name,
            repo_id=args.repo,
            repo_type="model",
            commit_message=f"Upload {path.name}",
        )
        print("done")

    print(f"\nAll files uploaded to: https://huggingface.co/{args.repo}")
    print("\nTo use in app.py startup, add:")
    print(f"""
    from huggingface_hub import hf_hub_download
    HF_REPO = "{args.repo}"
    for fname in {[p.name for p in to_upload]}:
        dest = PROCESSED / fname
        if not dest.exists():
            hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir=PROCESSED)
    """)


if __name__ == "__main__":
    main()
