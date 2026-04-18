#!/usr/bin/env python3
"""
deploy_to_spaces.py
-------------------
One-shot deployment to HuggingFace Spaces (free tier, Streamlit SDK).

Hardened against `reviews/04_ml_engineer_review.md` §9 "Deployment script review":

- ``--yes`` flag for non-interactive runs; otherwise the script prompts for
  confirmation before overwriting production. CI calls with ``--yes``; humans
  get a typed-y/N prompt so a bare invocation can't ship by accident.
- Atomic uploads via ``HfApi.upload_folder(allow_patterns=...)`` instead of a
  per-file ``upload_file`` loop. Faster (parallel inside `huggingface_hub`),
  and the entire batch lands as a single commit so partial failures don't
  leave the repo in a half-deployed state.
- Version tagging via ``--tag vX.Y.Z`` (or auto-derived ``deploy-<git-sha>``)
  using ``HfApi.create_tag``. Gives a roll-back point per release.
- Pre-flight validation of ``HF_TOKEN`` (whoami round-trip) and existence of
  the model artefacts on disk. Fails loudly before touching the Hub.
- Post-deploy verification that ``HF_REPO_ID`` Space secret is actually set
  (re-read via ``get_space_variables``, since secret values aren't readable —
  but the *presence* of the secret name is, and that's what matters).
- Git SHA captured at deploy time and threaded through every commit message
  so an HF Hub commit can be traced back to a known repo state.

Usage
-----
    # Interactive (with confirmation prompt)
    python scripts/deploy_to_spaces.py --hf-user YOUR_HF_USERNAME

    # Non-interactive (CI)
    python scripts/deploy_to_spaces.py --hf-user YOUR_HF_USERNAME --yes

    # Tagged release
    python scripts/deploy_to_spaces.py --hf-user YOUR_HF_USERNAME --tag v0.2.0 --yes

    # Token via flag instead of `huggingface-cli login`
    python scripts/deploy_to_spaces.py --hf-user YOUR_HF_USERNAME --token hf_xxx --yes

Prerequisites
-------------
    huggingface-cli login   (or pass --token / set HF_TOKEN env var)
    pip install huggingface-hub
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
PROCESSED = REPO_ROOT / "data" / "processed"

MODEL_REPO_SUFFIX = "football-xg-models"     # <hf_user>/football-xg-models
SPACE_REPO_SUFFIX = "football-xg-dashboard"  # <hf_user>/football-xg-dashboard

# Model-repo upload patterns (passed to `upload_folder(allow_patterns=...)`).
# Keep in sync with app.py's HF Hub bootstrap REQUIRED + OPTIONAL lists.
MODEL_UPLOAD_PATTERNS = [
    # Required artefacts (model bootstrap fails without these)
    "pool_7comp_hybrid_xg.pt",
    "pool_7comp_hybrid_xg.meta.json",
    "pool_7comp_hybrid_gat_xg.pt",
    "pool_7comp_hybrid_gat_xg.meta.json",
    "pool_7comp_T.pt",
    "pool_7comp_gat_T.pt",
    "pool_7comp_per_comp_T_gcn.pt",
    "pool_7comp_per_comp_T_gat.pt",
    # Per-competition shot graphs (each enables one tab in the UI)
    "statsbomb_*_shot_graphs.pt",
    # Data manifest (sha256 over .pt files; consumed by app.py bootstrap)
    "MANIFEST.json",
    # Feature importance for the dashboard's Feature Importance view
    "feature_importance.json",
]

# Space-repo upload patterns. These are paths relative to REPO_ROOT.
# README_SPACE.md → README.md is the only renaming case; handled separately.
SPACE_UPLOAD_PATTERNS = [
    "app.py",
    "requirements.txt",
    "src/**",
]

# Files that need renaming on upload (local_name → name_in_space)
SPACE_RENAMES = {
    "README_SPACE.md": "README.md",
}


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deploy model artefacts + Streamlit app to HuggingFace Spaces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hf-user", required=True, help="Your HuggingFace username")
    p.add_argument("--token", default=None,
                   help="HF write token (default: HF_TOKEN env var or hf auth login)")
    p.add_argument(
        "--tag", default=None,
        help="Version tag to create on the model repo after upload "
             "(e.g. 'v0.2.0'). Defaults to 'deploy-<short-git-sha>'.",
    )
    p.add_argument(
        "--yes", "-y", action="store_true",
        help="Non-interactive: skip the confirmation prompt. Use in CI.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be uploaded; do not call the Hub.",
    )
    p.add_argument(
        "--skip-model-upload", action="store_true",
        help="Only update the Space (skip the model artefact repo).",
    )
    p.add_argument(
        "--skip-space-upload", action="store_true",
        help="Only update the model repo (skip the Streamlit Space).",
    )
    return p.parse_args()


# =============================================================================
# Token + git provenance
# =============================================================================

def get_token(explicit: str | None) -> str | None:
    """Resolve the HF write token from --token, env, or cached login."""
    if explicit:
        return explicit
    t = os.environ.get("HF_TOKEN")
    if t:
        return t
    token_file = Path.home() / ".cache" / "huggingface" / "token"
    if token_file.exists():
        return token_file.read_text().strip()
    return None


def git_sha(short: bool = True) -> str | None:
    """Best-effort short git SHA for provenance. None if not a git repo."""
    try:
        cmd = ["git", "-C", str(REPO_ROOT), "rev-parse"]
        if short:
            cmd.append("--short")
        cmd.append("HEAD")
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out or None
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def git_dirty() -> bool:
    """True if the working tree has uncommitted changes. False if no git."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return bool(out)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


# =============================================================================
# Pre-flight + confirmation
# =============================================================================

def validate_token(api) -> str:
    """Round-trip whoami() to confirm the token actually works.

    HF Hub will silently let you upload to a repo you don't have write access
    to, then 403 on commit. Better to fail at second 0 than minute 4.
    """
    try:
        info = api.whoami()
    except Exception as e:
        print(f"ERROR: HF token rejected by whoami(): {e}", file=sys.stderr)
        sys.exit(2)
    return info.get("name") or info.get("email") or "<unknown>"


def validate_artefacts(processed: Path) -> tuple[list[str], list[str]]:
    """Return (present, missing) artefact filenames.

    Missing optional files (per-comp shot graphs, manifest, feature importance)
    are merely warned about; missing REQUIRED artefacts hard-fail.
    """
    REQUIRED = [
        "pool_7comp_hybrid_xg.pt",
        "pool_7comp_hybrid_xg.meta.json",
        "pool_7comp_hybrid_gat_xg.pt",
        "pool_7comp_hybrid_gat_xg.meta.json",
        "pool_7comp_T.pt",
        "pool_7comp_gat_T.pt",
    ]
    OPTIONAL_GLOBS = [
        "pool_7comp_per_comp_T_gcn.pt",
        "pool_7comp_per_comp_T_gat.pt",
        "MANIFEST.json",
        "feature_importance.json",
    ]
    present: list[str] = []
    missing_required: list[str] = []
    for fname in REQUIRED:
        if (processed / fname).exists():
            present.append(fname)
        else:
            missing_required.append(fname)

    for fname in OPTIONAL_GLOBS:
        if (processed / fname).exists():
            present.append(fname)

    # Per-competition shot graphs — at least one must exist or the app shows
    # a blank dashboard. We surface the count so the operator can sanity-check.
    sg = sorted(processed.glob("statsbomb_*_shot_graphs.pt"))
    present.extend(p.name for p in sg)

    return present, missing_required


def confirm_or_abort(*, hf_user: str, model_repo: str, space_repo: str,
                     tag: str, present: list[str], dirty: bool, yes: bool) -> None:
    """Interactive y/N gate. Skipped under --yes."""
    print()
    print("=" * 64)
    print("  DEPLOYMENT PLAN")
    print("=" * 64)
    print(f"  HF user      : {hf_user}")
    print(f"  Model repo   : {model_repo}")
    print(f"  Space repo   : {space_repo}")
    print(f"  Version tag  : {tag}")
    print(f"  Artefacts    : {len(present)} files")
    print(f"  Git dirty?   : {'YES — uncommitted changes' if dirty else 'no'}")
    print("=" * 64)
    if yes:
        print("  --yes given; proceeding without prompt.")
        return
    try:
        ans = input("  Type 'yes' to deploy (anything else aborts): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n  Aborted (no input).")
        sys.exit(130)
    if ans != "yes":
        print("  Aborted.")
        sys.exit(1)


# =============================================================================
# HF Hub operations
# =============================================================================

def _commit_message(action: str, sha: str | None) -> str:
    """Stamp every commit with timestamp + git SHA for traceability."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    parts = [action, f"ts={ts}"]
    if sha:
        parts.append(f"git={sha}")
    return " | ".join(parts)


def upload_model_repo(api, *, model_repo: str, processed: Path,
                      sha: str | None, dry_run: bool) -> None:
    """Atomic upload of model artefacts via upload_folder.

    `upload_folder(allow_patterns=...)` ships every matching file as a single
    commit — partial failures roll back automatically. Much faster than the
    old per-file loop because internally HF Hub uploads with parallelism.
    """
    print(f"\n── Model repo: {model_repo} ─────────────────────────────────────")
    if dry_run:
        for pat in MODEL_UPLOAD_PATTERNS:
            for path in sorted(processed.glob(pat)):
                size_mb = path.stat().st_size / 1e6
                print(f"  [dry-run] would upload  {path.name:<48} {size_mb:6.1f} MB")
        return

    from huggingface_hub import create_repo
    create_repo(model_repo, token=api.token, repo_type="model",
                exist_ok=True, private=False)
    print("  Repo ready (created or already existed).")

    msg = _commit_message("Deploy model artefacts", sha)
    print(f"  Uploading via upload_folder (atomic) …")
    api.upload_folder(
        folder_path=str(processed),
        repo_id=model_repo,
        repo_type="model",
        allow_patterns=MODEL_UPLOAD_PATTERNS,
        commit_message=msg,
    )
    print("  Upload complete.")


def upload_space_repo(api, *, space_repo: str, repo_root: Path,
                      sha: str | None, dry_run: bool) -> None:
    """Atomic upload of the Streamlit app + src/ tree to the Space.

    README_SPACE.md must be renamed to README.md on upload (HF Spaces requires
    front-matter in README.md). We do that as a separate single-file upload
    after the folder upload because `upload_folder` doesn't support per-file
    renaming.
    """
    print(f"\n── Space: {space_repo} ──────────────────────────────────────────")
    if dry_run:
        print("  [dry-run] would create Space (streamlit SDK)")
        for pat in SPACE_UPLOAD_PATTERNS:
            print(f"  [dry-run] would upload pattern: {pat}")
        for local, remote in SPACE_RENAMES.items():
            print(f"  [dry-run] would upload: {local} → {remote}")
        return

    from huggingface_hub import create_repo
    create_repo(space_repo, token=api.token, repo_type="space",
                space_sdk="streamlit", exist_ok=True, private=False)
    print("  Space ready.")

    # Folder upload: app.py, requirements.txt, src/**.
    msg = _commit_message("Deploy app code", sha)
    api.upload_folder(
        folder_path=str(repo_root),
        repo_id=space_repo,
        repo_type="space",
        allow_patterns=SPACE_UPLOAD_PATTERNS,
        # Defensive ignore — HF Spaces should never see local caches / venvs / data.
        ignore_patterns=[
            "__pycache__/**", "*.pyc", ".pytest_cache/**",
            ".venv/**", "venv/**", "data/**", "results/**",
            ".git/**", "reviews/**", "paper/**", "tests/**",
        ],
        commit_message=msg,
    )
    print("  Folder upload complete.")

    # Renamed files (README_SPACE.md → README.md)
    for local_name, remote_name in SPACE_RENAMES.items():
        local_path = repo_root / local_name
        if not local_path.exists():
            print(f"  WARNING — rename source missing: {local_name} (skipped)")
            continue
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=remote_name,
            repo_id=space_repo,
            repo_type="space",
            commit_message=_commit_message(f"Deploy {remote_name}", sha),
        )
        print(f"  Uploaded {local_name} → {remote_name}")


def set_and_verify_space_secret(api, *, space_repo: str, model_repo: str,
                                dry_run: bool) -> None:
    """Set HF_REPO_ID on the Space and verify it actually landed.

    HF Hub silently swallows permission errors on `add_space_secret` for some
    edge cases (Space created moments ago, eventual consistency). Re-read the
    Space's variable list afterward to make sure HF_REPO_ID is registered.
    Secret *values* aren't readable (by design), but the *presence* of the
    secret name is — and that's what app.py's bootstrap actually checks.
    """
    print(f"\n── Space secrets ────────────────────────────────────────────────")
    if dry_run:
        print(f"  [dry-run] would set HF_REPO_ID = {model_repo}")
        return

    try:
        api.add_space_secret(space_repo, "HF_REPO_ID", model_repo)
        print(f"  Set HF_REPO_ID = {model_repo}")
    except Exception as e:
        print(f"  WARNING — add_space_secret raised: {e}")
        print(f"  → Set manually in Space settings: HF_REPO_ID = {model_repo}")
        return

    # Verify presence. get_space_secrets returns secret *names* only.
    try:
        secrets = api.get_space_secrets(space_repo)
        names = {s.get("key") if isinstance(s, dict) else getattr(s, "key", None)
                 for s in secrets}
        if "HF_REPO_ID" in names:
            print("  Verified: HF_REPO_ID secret is registered on the Space.")
        else:
            print("  WARNING — HF_REPO_ID not visible in Space secrets after set.")
            print(f"  → Set manually: https://huggingface.co/spaces/{space_repo}/settings")
    except Exception as e:
        # get_space_secrets isn't on every huggingface_hub version. Fall
        # back to a softer check: just trust the add_space_secret success.
        print(f"  Note: could not read back secret list ({type(e).__name__}); "
              "trusting add_space_secret return.")


def create_release_tag(api, *, model_repo: str, tag: str, sha: str | None,
                       dry_run: bool) -> None:
    """Create a roll-back point on the model repo.

    Without tagging, every upload commit is just `Deploy model artefacts |
    ts=…` — there's no clean way to say "go back to the AUC=0.760 release".
    A git-style tag on HF Hub fixes that.
    """
    if not tag:
        return
    print(f"\n── Tagging model repo as '{tag}' ────────────────────────────────")
    if dry_run:
        print(f"  [dry-run] would create tag '{tag}' on {model_repo}")
        return

    msg = f"Release {tag}" + (f" (git={sha})" if sha else "")
    try:
        api.create_tag(
            repo_id=model_repo,
            repo_type="model",
            tag=tag,
            tag_message=msg,
            exist_ok=False,
        )
        print(f"  Tag '{tag}' created.")
    except Exception as e:
        # Most common failure: tag already exists. Surface the error but
        # don't fail the deploy — the artefacts are already uploaded.
        print(f"  WARNING — could not create tag '{tag}': {e}")
        print(f"  → If you want to overwrite: delete the tag in the HF UI and re-run.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()
    token = get_token(args.token)
    user = args.hf_user

    if not token:
        print("ERROR: No HuggingFace token found.", file=sys.stderr)
        print("  Try one of:", file=sys.stderr)
        print("    huggingface-cli login", file=sys.stderr)
        print("    export HF_TOKEN=hf_xxx", file=sys.stderr)
        print("    python deploy_to_spaces.py --token hf_xxx ...", file=sys.stderr)
        sys.exit(2)

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: pip install huggingface-hub", file=sys.stderr)
        sys.exit(2)

    api = HfApi(token=token)

    # Pre-flight: token actually works, artefacts actually exist.
    who = validate_token(api)
    print(f"  Authenticated as: {who}")

    present, missing_required = validate_artefacts(PROCESSED)
    if missing_required and not args.skip_model_upload:
        print("ERROR: required model artefacts missing under "
              f"{PROCESSED}:", file=sys.stderr)
        for m in missing_required:
            print(f"  - {m}", file=sys.stderr)
        print("  Run training + calibration before deploying.", file=sys.stderr)
        sys.exit(2)

    sha = git_sha(short=True)
    dirty = git_dirty()

    # Tag default: deploy-<git-sha>. Skip tagging entirely if we have neither
    # an explicit --tag nor a git SHA.
    tag = args.tag or (f"deploy-{sha}" if sha else "")

    model_repo = f"{user}/{MODEL_REPO_SUFFIX}"
    space_repo = f"{user}/{SPACE_REPO_SUFFIX}"

    confirm_or_abort(
        hf_user=user,
        model_repo=model_repo,
        space_repo=space_repo,
        tag=tag or "(none)",
        present=present,
        dirty=dirty,
        yes=args.yes,
    )

    # ── 1. Model artefacts ───────────────────────────────────────────────────
    if not args.skip_model_upload:
        upload_model_repo(
            api,
            model_repo=model_repo,
            processed=PROCESSED,
            sha=sha,
            dry_run=args.dry_run,
        )
        if tag:
            create_release_tag(
                api,
                model_repo=model_repo,
                tag=tag,
                sha=sha,
                dry_run=args.dry_run,
            )
    else:
        print("\n  Skipping model upload (--skip-model-upload).")

    # ── 2. Space code ────────────────────────────────────────────────────────
    if not args.skip_space_upload:
        upload_space_repo(
            api,
            space_repo=space_repo,
            repo_root=REPO_ROOT,
            sha=sha,
            dry_run=args.dry_run,
        )
        set_and_verify_space_secret(
            api,
            space_repo=space_repo,
            model_repo=model_repo,
            dry_run=args.dry_run,
        )
    else:
        print("\n  Skipping Space upload (--skip-space-upload).")

    # ── 3. Summary ───────────────────────────────────────────────────────────
    print(f"""
{'=' * 64}
  Deployment complete{' (dry-run, nothing actually pushed)' if args.dry_run else ''}.

  Model repo : https://huggingface.co/{model_repo}
  Space URL  : https://huggingface.co/spaces/{space_repo}
  Live app   : https://{user.lower()}-{SPACE_REPO_SUFFIX}.hf.space
  Release    : {tag or '(no tag)'}
  Git SHA    : {sha or '(no git)'}{' (DIRTY)' if dirty else ''}

  The Space will rebuild for ~3-5 minutes on first push of new code.
  Build logs : https://huggingface.co/spaces/{space_repo}
{'=' * 64}
""")


if __name__ == "__main__":
    main()
