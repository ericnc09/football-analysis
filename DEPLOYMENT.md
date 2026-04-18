# DEPLOYMENT.md

End-to-end runbook for deploying the Football GNN xG dashboard to
HuggingFace Spaces. Satisfies `reviews/04_ml_engineer_review.md` §8
"Before launch" → DEPLOYMENT.md checklist item.

Audience: one person on their laptop pushing a new release. Read this
top-to-bottom once; after that the TL;DR box is enough.

---

## TL;DR

```bash
# 1. Train + calibrate (one-off per model release)
python scripts/train_xg_hybrid.py  --seed 42 --epochs 30
python scripts/calibrate_and_train_gat.py  --seed 42

# 2. Build the data manifest (sha256 over .pt files)
python scripts/build_manifest.py

# 3. Dry-run the deploy so you can see what will ship
python scripts/deploy_to_spaces.py --hf-user YOUR_HF_USER --dry-run

# 4. Deploy for real (interactive y/N prompt)
python scripts/deploy_to_spaces.py --hf-user YOUR_HF_USER --tag v0.2.0

# 5. Smoke test — open the Space, pick each view from the sidebar.
#    https://huggingface.co/spaces/YOUR_HF_USER/football-xg-dashboard
```

Stop reading here unless something fails.

---

## Prerequisites

### HuggingFace account

1. Create an account at `huggingface.co`.
2. Generate a write token: Settings → Access Tokens → *write* scope.
3. Either `huggingface-cli login` once, or export `HF_TOKEN=hf_...`
   in the shell where you run the deploy script.

The deploy script calls `HfApi.whoami()` before uploading anything. A
bad token fails in the first second of the script, not 4 minutes into
an upload.

### Local Python environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install huggingface-hub            # required by the deploy script
```

Python 3.10+. `requirements.txt` pins are lockfile-quality; reviewer §2
sign-off says CPU-only PyTorch + PyG 2.5.x is expected.

### Data + trained artefacts on disk

The deploy script uploads whatever is in `data/processed/`. Before
deploy, verify:

```bash
ls data/processed/pool_7comp_hybrid_xg.pt \
   data/processed/pool_7comp_hybrid_xg.meta.json \
   data/processed/pool_7comp_hybrid_gat_xg.pt \
   data/processed/pool_7comp_hybrid_gat_xg.meta.json \
   data/processed/pool_7comp_T.pt \
   data/processed/pool_7comp_gat_T.pt \
   data/processed/pool_7comp_per_comp_T_gcn.pt \
   data/processed/pool_7comp_per_comp_T_gat.pt \
   data/processed/MANIFEST.json \
   data/processed/statsbomb_*_shot_graphs.pt
```

Missing required files → the deploy script aborts pre-flight with a
list of what's missing. Don't bypass — retrain instead.

---

## Training pipeline (when you need a new model release)

Every training run records `git rev-parse --short HEAD` and appends a
line to `results/runs.jsonl` so two months from now you can tell which
commit shipped which AUC.

```bash
# HybridGCN baseline, 5-fold competition-pooled
python scripts/train_xg_hybrid.py \
    --seed 42 --epochs 30 --batch-size 64 \
    --lr 5e-4 --weight-decay 1e-4

# HybridGAT with per-competition temperature calibration
python scripts/calibrate_and_train_gat.py --seed 42 --epochs 30
```

Both scripts accept `--config path/to/yaml` for hyperparameter sweeps.
CLI flags always override YAML values. See `scripts/train_xg_hybrid.py`
docstring for the full argparse surface.

After training you should have two `.pt` checkpoints plus two
`.meta.json` sidecars (feature schema version, git SHA, seed, metrics).
The sidecars are validated at app startup; a schema mismatch between
the sidecar and `src/features.py` raises `FeatureSchemaMismatch` before
any view renders.

---

## Data manifest

```bash
python scripts/build_manifest.py
```

Writes `data/processed/MANIFEST.json` with the shape app.py's HF Hub
bootstrap expects:

```json
{
  "files": {
    "pool_7comp_hybrid_xg.pt": {
      "sha256": "…",
      "size_bytes": 1234567,
      "mtime": "2026-04-17T14:22:03Z"
    }
  }
}
```

Run it after training so the manifest covers the newly-written
checkpoints. The manifest is uploaded alongside the artefacts and
downstream `app.py` verifies sha256 on startup.

---

## Deploy command

```bash
python scripts/deploy_to_spaces.py \
    --hf-user YOUR_HF_USER \
    --tag v0.2.0 \
    [--yes] \
    [--dry-run]
```

What it does, in order:

1. `HfApi.whoami()` to verify the token works.
2. Checks that all required artefacts exist under `data/processed/`;
   aborts with a list of missing files if not.
3. Prints a deployment plan (repos, tag, file count, git cleanliness)
   and prompts for `yes`/no — unless `--yes` is given.
4. `upload_folder(allow_patterns=...)` to the model repo
   `YOUR_HF_USER/football-xg-models`. Atomic commit.
5. `create_tag()` on the model repo as a rollback point
   (`deploy-<git-sha>` if no `--tag`).
6. `upload_folder(...)` to the Space repo
   `YOUR_HF_USER/football-xg-dashboard`. Ignores `__pycache__/`,
   `.venv/`, `data/`, `.git/`, `reviews/`, etc. — so no
   accidental artefact leaks.
7. Uploads `README_SPACE.md` → `README.md` (required by HF Spaces
   for the front-matter block).
8. `add_space_secret("HF_REPO_ID", "<model-repo>")` then reads back
   the Space's secret-name list to verify the secret landed.

Commit messages on both repos embed timestamp + git SHA, so Hub
commits trace back to local repo state.

---

## Secret setup (first-time only)

The Space depends on one secret:

- `HF_REPO_ID` — where app.py downloads models from, e.g.
  `YOUR_HF_USER/football-xg-models`.

`deploy_to_spaces.py` sets this automatically. If it fails (most
common cause: token lacks Space-write scope), set it manually at:

```
https://huggingface.co/spaces/YOUR_HF_USER/football-xg-dashboard/settings
```

Sidebar → Variables and secrets → New secret → `HF_REPO_ID`.

Optional secrets (not required for the demo to boot):

- `HF_TOKEN` — only needed if the model repo is *private*.
- `LOG_LEVEL` — `DEBUG` | `INFO` | `WARNING`. Default `INFO`.

---

## Cold-start behaviour

First request after a Space rebuild takes ~30–45s:

1. HF Hub runtime boots the container (~15s).
2. `app.py` imports → `configure_logging()` → `st.session_state` init.
3. The HF Hub bootstrap block runs `snapshot_download(allow_patterns=...)`
   with 4 workers, pulling ~120 MB of `.pt` files + metadata sidecars.
   Cached on the Space's ephemeral disk; subsequent reboots still
   re-download (HF Spaces wipes /tmp between sleeps).
4. `st.cache_resource`-backed `load_gcn_model` / `load_gat_model`
   deserialise the checkpoints (~3s).
5. First user click renders.

Warm requests after that are <200ms for most views.

If cold-start feels stuck for >2 minutes, the most likely cause is the
Hub download silently failing. Check build logs at:

```
https://huggingface.co/spaces/YOUR_HF_USER/football-xg-dashboard
```

The `view_error` structured log lines include a session-id that maps a
specific user's crash to an otherwise-anonymous log feed.

---

## Rollback

Every `deploy_to_spaces.py` invocation creates a tag on the model repo
(`v0.2.0`, `deploy-abc1234`, etc.). To roll back:

### Option A — revert model artefacts only

```bash
# On HF Hub UI: YOUR_HF_USER/football-xg-models → branches/tags →
# right-click the previous tag → "Reset branch to this commit"
```

Within ~30 seconds (time for the Space to notice the update and
reload), the live app serves the older artefacts. No Space rebuild
required because the Space just downloads from the model repo at
startup.

### Option B — revert the Streamlit app

The Space is its own git-like repo. Browse commits at:

```
https://huggingface.co/spaces/YOUR_HF_USER/football-xg-dashboard/commits/main
```

HF Hub UI → pick a commit → "Revert". The Space rebuilds (~3–5 min).

### Option C — nuclear option

```bash
python scripts/deploy_to_spaces.py \
    --hf-user YOUR_HF_USER \
    --tag v0.1.0 \
    --yes
```

…from a git checkout of the previous release. Full re-deploy.

---

## Smoke test checklist

After deploy, open the Space and walk through these in order. Each
view is wrapped in a `view_boundary` so a bug in one view will NOT
blank the entire app — it'll render a `st.error` box with the session
id and keep the sidebar usable.

- [ ] Landing page loads; competition selector shows all 7 tournaments.
- [ ] **Shot Map** — dots render; hover works; per-shot probability
      rounds to the model's sidecar values for the selected comp.
- [ ] **Shot Inspector** — single-shot permutation importance shows
      non-zero bars for `shot_dist` and `shot_angle`.
- [ ] **xG Distributions** — histogram renders; model vs StatsBomb
      curves are visible.
- [ ] **Match Report** — defaults to a real match, shows per-team
      cumulative xG.
- [ ] **Surprise Goals** — at least one goal with high surprise index.
- [ ] **Player Profile** — autocomplete finds e.g. "Kylian Mbappé".
- [ ] **Feature Importance** — bar chart renders with the canonical
      feature labels (not raw keys like `shot_dist`).

If any view errors, the `view_error` JSON log line on the Space's log
stream includes `error_type`, `error_message`, and `session`. That's
enough to reproduce locally.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `HF token rejected by whoami()` | Token missing/expired | `huggingface-cli login` or `export HF_TOKEN=hf_...` |
| `required model artefacts missing` | Didn't train before deploying | Run §Training pipeline first |
| Space builds but shows `FeatureSchemaMismatch` | `src/features.py` changed without retraining | Retrain; sidecar schema must match runtime code |
| Space builds but shows `MANIFEST checksum mismatch` | Uploaded `.pt` doesn't match MANIFEST | Re-run `scripts/build_manifest.py` after training, then redeploy |
| `HF_REPO_ID` secret missing after deploy | Token lacks Space-write | Set manually in Space settings (see §Secret setup) |
| Space 500s on load, no useful log | Hub download timing out | Re-trigger build; check Hub status page |
| Cold start >2 min | First-time download of 120 MB | Wait. If >5 min, check Hub logs for network errors. |

---

## Related docs

- `MODEL_CARD.md` — what the model is, how it was trained, where it
  applies and where it doesn't. This is the file uploaded as
  `README.md` to the model repo.
- `README_SPACE.md` — uploaded as the Space's README; controls the
  front-page description on `huggingface.co/spaces/...`.
- `reviews/04_ml_engineer_review.md` §7–§9 — the ML engineering
  review that drove this runbook's design.
