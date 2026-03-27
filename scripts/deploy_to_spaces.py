#!/usr/bin/env python3
"""
deploy_to_spaces.py
-------------------
One-shot deployment to HuggingFace Spaces (free tier, Streamlit SDK).

What it does
------------
1. Creates (or updates) a HF Hub model repo with model weights + shot graphs
2. Creates (or updates) a HF Space wired to the GitHub repo via README_SPACE.md
3. Sets the HF_REPO_ID Space secret so app.py knows where to download data

Usage
-----
    python scripts/deploy_to_spaces.py --hf-user YOUR_HF_USERNAME

    # With explicit token (alternative to hf auth login):
    python scripts/deploy_to_spaces.py --hf-user YOUR_HF_USERNAME --token hf_xxx

Prerequisites
-------------
    huggingface-cli login   (or pass --token)
    pip install huggingface-hub
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

MODEL_REPO_SUFFIX = "football-xg-models"   # <hf_user>/football-xg-models
SPACE_REPO_SUFFIX = "football-xg-dashboard" # <hf_user>/football-xg-dashboard

UPLOAD_FILES = [
    "pool_7comp_hybrid_xg.pt",
    "pool_7comp_hybrid_gat_xg.pt",
    "pool_7comp_T.pt",
    "pool_7comp_gat_T.pt",
    "pool_7comp_per_comp_T_gcn.pt",
    "pool_7comp_per_comp_T_gat.pt",
    "statsbomb_wc2022_shot_graphs.pt",
    "statsbomb_wwc2023_shot_graphs.pt",
    "statsbomb_euro2020_shot_graphs.pt",
    "statsbomb_euro2024_shot_graphs.pt",
    "statsbomb_bundesliga2324_shot_graphs.pt",
    "statsbomb_weuro2022_shot_graphs.pt",
    "statsbomb_weuro2025_shot_graphs.pt",
    "feature_importance.json",
]

# Files that every HF Space needs
SPACE_FILES = [
    ("README_SPACE.md", "README.md"),  # (local path, path in Space)
    ("app.py",          "app.py"),
    ("requirements.txt","requirements.txt"),
]
# Source dirs / files to sync into the Space
SPACE_DIRS = ["src"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hf-user", required=True, help="Your HuggingFace username")
    p.add_argument("--token",   default=None,  help="HF write token (or use hf auth login)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def get_token(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def main():
    args   = parse_args()
    token  = get_token(args.token)
    user   = args.hf_user

    if not token:
        print("ERROR: No HuggingFace token found.")
        print("  Run:  huggingface-cli login")
        print("  Or:   python deploy_to_spaces.py --token hf_xxx")
        sys.exit(1)

    try:
        from huggingface_hub import HfApi, create_repo
        from huggingface_hub.utils import RepositoryNotFoundError
    except ImportError:
        print("ERROR: pip install huggingface-hub")
        sys.exit(1)

    api         = HfApi(token=token)
    model_repo  = f"{user}/{MODEL_REPO_SUFFIX}"
    space_repo  = f"{user}/{SPACE_REPO_SUFFIX}"

    # ── 1. Create / verify model repo ────────────────────────────────────────
    print(f"\n── Model repo: {model_repo} ─────────────────────────────────────")
    if not args.dry_run:
        create_repo(model_repo, token=token, repo_type="model", exist_ok=True, private=False)
        print("  Repo ready.")

    # Upload each file
    to_upload = []
    missing   = []
    for fname in UPLOAD_FILES:
        p = PROCESSED / fname
        if p.exists():
            to_upload.append(p)
        else:
            missing.append(fname)

    if missing:
        print(f"  Warning — {len(missing)} file(s) not found (skipped):")
        for m in missing:
            print(f"    {m}")

    total_mb = sum(p.stat().st_size for p in to_upload) / 1e6
    print(f"  Uploading {len(to_upload)} files ({total_mb:.0f} MB) …")

    for path in to_upload:
        size_mb = path.stat().st_size / 1e6
        print(f"  → {path.name:<50} {size_mb:5.1f} MB", end="", flush=True)
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(path),
                path_in_repo=path.name,
                repo_id=model_repo,
                repo_type="model",
                commit_message=f"Upload {path.name}",
            )
            print(" done")
        else:
            print(" [dry-run]")

    # ── 2. Create / verify Space ──────────────────────────────────────────────
    print(f"\n── Space: {space_repo} ──────────────────────────────────────────")
    if not args.dry_run:
        create_repo(space_repo, token=token, repo_type="space",
                    space_sdk="streamlit", exist_ok=True, private=False)
        print("  Space ready.")

    # Upload Space files
    for local_name, space_name in SPACE_FILES:
        local_path = REPO_ROOT / local_name
        if not local_path.exists():
            print(f"  MISSING: {local_name} — skipping")
            continue
        print(f"  → {local_name} → {space_name}", end="", flush=True)
        if not args.dry_run:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=space_name,
                repo_id=space_repo,
                repo_type="space",
                commit_message=f"Deploy {space_name}",
            )
            print(" done")
        else:
            print(" [dry-run]")

    # Upload src/ directory
    src_dir = REPO_ROOT / "src"
    if src_dir.exists():
        print(f"  → src/ directory", end="", flush=True)
        if not args.dry_run:
            api.upload_folder(
                folder_path=str(src_dir),
                path_in_repo="src",
                repo_id=space_repo,
                repo_type="space",
                commit_message="Deploy src/",
            )
            print(" done")
        else:
            print(" [dry-run]")

    # ── 3. Set Space secrets ──────────────────────────────────────────────────
    print(f"\n── Space secrets ────────────────────────────────────────────────")
    if not args.dry_run:
        try:
            api.add_space_secret(space_repo, "HF_REPO_ID", model_repo)
            print(f"  HF_REPO_ID = {model_repo}")
        except Exception as e:
            print(f"  Could not set secret automatically: {e}")
            print(f"  → Set manually in Space settings: HF_REPO_ID = {model_repo}")
    else:
        print(f"  [dry-run] Would set HF_REPO_ID = {model_repo}")

    # ── 4. Summary ────────────────────────────────────────────────────────────
    print(f"""
{'=' * 60}
  Deployment complete!

  Model repo : https://huggingface.co/{model_repo}
  Space URL  : https://huggingface.co/spaces/{space_repo}
  Live app   : https://{user.lower()}-{SPACE_REPO_SUFFIX}.hf.space

  The Space will build for ~3-5 minutes on first deploy.
  Check build logs: https://huggingface.co/spaces/{space_repo}
{'=' * 60}
""")


if __name__ == "__main__":
    main()
