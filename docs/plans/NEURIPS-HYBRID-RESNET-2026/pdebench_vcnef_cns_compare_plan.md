# VCNeF Capped CNS Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the official VCNeF source locally, run it on the capped PDEBench `2d_cfd_cns` slices, and compare it to `spectral_resnet_bottleneck_base` visually and via the existing CNS metric suite.

**Architecture:** Keep the VCNeF source external and pinned by commit; do not fork or rewrite the author repository. Add a thin local wrapper/profile that imports the official `VCNeFModel`, buffers the fixed CNS grid and one-step query time, and runs inside the existing `run_cfd_cns` runner so split logic, normalization, metrics, train/test eval emission, and comparison galleries stay identical. Use a `history_len=1` Markov contract for both VCNeF and hybrid-spectral; do not compare VCNeF directly against the existing `history_len=2` spectral artifacts.

**Tech Stack:** Python, PyTorch, HDF5, local PDEBench CNS runner, external VCNeF GitHub source, tmux for long runs.

---

## Context And Non-Negotiable Decisions

- Public VCNeF source inspected: `jhagnberger/vcnef` at commit `bd891e1bdbfa43f52949ceca050ae12efc8564be`.
- The author repo is not a PyPI package and has no `setup.py` or `pyproject.toml`; treat installation as `git clone` plus a pinned local import root, not `pip install vcnef`.
- The author PDEBench example (`examples_pde_bench.py`) is not an equal-footing comparison path for this repo because it:
  - uses its own PDEBench dataset loader and 90/10 split logic,
  - predicts a trajectory from a single initial state plus queried times,
  - uses `OneCycleLR`, W&B logging, and a 500-epoch recipe,
  - does not match the local CNS artifact/metric contract.
- Therefore the comparison should use the existing local CNS runner and reporting stack, with VCNeF inserted as one additional model profile.
- The first fair contract is:
  - task: `2d_cfd_cns`
  - `history_len=1`
  - one-step prediction `u[t-1] -> u[t]`
  - same capped trajectory splits already used locally
  - same local metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low/mid/high`
- Use `spectral_resnet_bottleneck_base` as the comparison row. If we want to compare against the stronger manual non-shared row later, treat that as a second tranche, not the default baseline.

## Planned Run Matrix

Primary execution tranche:

- small cap: `512 / 64 / 64` trajectories, `8` windows per trajectory
  - `10` epochs
  - `40` epochs
- larger cap: `1024 / 128 / 128` trajectories, `8` windows per trajectory
  - `40` epochs

Profiles per run:

- `spectral_resnet_bottleneck_base`
- `vcnef_cns_base`

Optional follow-up only if the first tranche is technically clean:

- larger cap `10`-epoch run for learning-curve alignment
- manual compare against `spectral_resnet_bottleneck_noshare`

## Output Contract

Each completed run must emit:

- standard `comparison_summary.json`
- per-profile `metrics_<profile>.json`
- train-split eval plus held-out test eval
- sample prediction gallery with ground truth plus both profiles
- sample residual gallery
- parameter counts
- source-provenance note containing the VCNeF Git commit and local clone path

The comparison is incomplete unless both the metric JSON and the PNG galleries exist.

### Task 1: Pin The VCNeF Source And Verify Importability

**Files:**
- Create: `.artifacts/external/vcnef/` (external clone, untracked)
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-vcnef-cns-compare/vcnef_source.json`
- Test: `python -c "import importlib.util; ..."`

- [ ] **Step 1: Clone the official VCNeF repo into an external artifact path**

Run:

```bash
git clone https://github.com/jhagnberger/vcnef .artifacts/external/vcnef
```

Expected: local clone exists and is not committed into the repo index.

- [ ] **Step 2: Record the pinned source revision**

Run:

```bash
python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path(".artifacts/external/vcnef")
sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
payload = {"repo": "https://github.com/jhagnberger/vcnef", "commit": sha, "path": str(root.resolve())}
path = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-vcnef-cns-compare/vcnef_source.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(path)
PY
```

Expected: `vcnef_source.json` exists with repo URL, commit SHA, and absolute path.

- [ ] **Step 3: Install only the dependencies the local wrapper actually needs**

Run:

```bash
python -m pip install einops==0.7.0
```

Expected: `einops` is present. Do not downgrade or replace the repo’s active `torch` unless the environment is already incompatible.

- [ ] **Step 4: Smoke-test direct import from the pinned clone**

Run:

```bash
PYTHONPATH="$(pwd)/.artifacts/external/vcnef:${PYTHONPATH}" python - <<'PY'
from vcnef.vcnef_2d import VCNeFModel
model = VCNeFModel(num_channels=4, condition_on_pde_param=False, d_model=256, n_heads=8, n_transformer_blocks=1, n_modulation_blocks=6)
print(type(model).__name__)
PY
```

Expected: prints `VCNeFModel` and exits `0`.

### Task 2: Add A Local VCNeF Wrapper That Fits The CNS Runner Contract

**Files:**
- Create: `scripts/studies/pdebench_image128/vcnef_adapter.py`
- Modify: `scripts/studies/pdebench_image128/models.py`
- Test: `tests/studies/test_pdebench_image128_models.py`

- [ ] **Step 1: Write failing model-builder tests for the VCNeF import boundary**

Cover:

```python
def test_vcnef_builder_blocks_cleanly_when_source_root_missing(): ...
def test_vcnef_wrapper_returns_bchw_output_for_cns_contract(): ...
def test_vcnef_wrapper_records_external_source_metadata(): ...
```

Expected: tests fail because `vcnef_cns_base` and the adapter do not exist yet.

- [ ] **Step 2: Add a small adapter module around the official model**

Implement `scripts/studies/pdebench_image128/vcnef_adapter.py` with:

- source-root resolution from `VCNEF_ROOT` first, then default `.artifacts/external/vcnef`
- clean `ModelBuildBlocker` when the source is absent or `einops`/imports fail
- `VCNeFCnsModel` wrapper that:
  - accepts normalized local input as `B,C,H,W`
  - converts to `B,H,W,C`
  - constructs the CNS mid-cell grid from shape and uniform spacing
  - uses a constant one-step normalized time query `t=[[1/(time_steps-1)]]`
  - calls the official `vcnef.vcnef_2d.VCNeFModel`
  - returns `B,C,H,W`
  - exposes source provenance for metrics payloads

- [ ] **Step 3: Extend the local model builder surface**

Modify `scripts/studies/pdebench_image128/models.py` to:

- accept optional `task_metadata` in `build_model_from_profile(...)`
- add a `vcnef_cns_net` builder branch
- pass through model provenance in `describe_model(...)` when available

- [ ] **Step 4: Run focused model tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -q
```

Expected: VCNeF-specific tests pass without breaking existing profiles.

### Task 3: Register A VCNeF CNS Profile Without Changing The Existing Baseline Defaults

**Files:**
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Test: `tests/studies/test_pdebench_image128_models.py`

- [ ] **Step 1: Add the model-profile fields needed for VCNeF**

Add only the knobs required now, for example:

- `vcnef_d_model`
- `vcnef_n_heads`
- `vcnef_transformer_blocks`
- `vcnef_modulation_blocks`
- `vcnef_patch_size_small`
- `vcnef_patch_size_large`

Do not create a large profile matrix.

- [ ] **Step 2: Register `vcnef_cns_base`**

Use the author defaults that are already shown in the official 2D example unless a local blocker forces a reduction:

- `d_model=256`
- `n_heads=8`
- `n_transformer_blocks=1`
- `n_modulation_blocks=6`
- patch sizes `4` and `16`
- `condition_on_pde_param=False`

Set `evidence_scope="readiness-only"` until a full comparison is complete.

- [ ] **Step 3: Re-run focused configuration tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -q
```

Expected: the new profile parses, builds, and does not disturb the existing profile IDs.

### Task 4: Keep The CNS Runner On The Existing Reporting Path

**Files:**
- Modify: `scripts/studies/pdebench_image128/cfd_cns.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Write a failing runner test for metadata pass-through**

Cover:

```python
def test_cfd_cns_passes_task_metadata_into_model_builder(): ...
```

Expected: fails because the builder does not yet receive task metadata.

- [ ] **Step 2: Pass task metadata into `build_model_from_profile(...)`**

Required metadata for VCNeF:

- `time_steps`
- `dx`
- `dy`
- `dt`
- `field_order`
- `history_len`

Do not change the training loop signature or the comparison-summary contract.

- [ ] **Step 3: Re-run focused runner tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_runner.py -q
```

Expected: runner tests pass and no existing profile path regresses.

### Task 5: Run The Small-Cap Matched Compare In The Existing Runner

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-vcnef-cns-compare/cap512-10ep-*`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-vcnef-cns-compare/cap512-40ep-*`
- Test: produced `comparison_summary.json`, `metrics_*.json`, gallery PNGs

- [ ] **Step 1: Launch the 512-cap 10-epoch compare in tmux**

Run:

```bash
VCNEF_ROOT="$(pwd)/.artifacts/external/vcnef" \
python scripts/studies/run_pdebench_image128_suite.py \
  --task-id 2d_cfd_cns \
  --mode readiness \
  --profile-ids spectral_resnet_bottleneck_base,vcnef_cns_base \
  --history-len 1 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-vcnef-cns-compare/cap512-10ep
```

Expected: one run root containing both model rows and shared comparison outputs.

- [ ] **Step 2: Launch the 512-cap 40-epoch compare in tmux**

Run the same command with `--epochs 40` and a distinct `--output-root`.

Expected: same artifact set for the longer budget.

- [ ] **Step 3: Verify the required artifacts**

Check for:

- `comparison_summary.json`
- `metrics_spectral_resnet_bottleneck_base.json`
- `metrics_vcnef_cns_base.json`
- sample prediction gallery
- sample residual gallery

### Task 6: Run The Larger-Cap 40-Epoch Compare

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-vcnef-cns-compare/cap1024-40ep-*`
- Test: produced `comparison_summary.json`, `metrics_*.json`, gallery PNGs

- [ ] **Step 1: Launch the 1024-cap 40-epoch compare**

Run:

```bash
VCNEF_ROOT="$(pwd)/.artifacts/external/vcnef" \
python scripts/studies/run_pdebench_image128_suite.py \
  --task-id 2d_cfd_cns \
  --mode readiness \
  --profile-ids spectral_resnet_bottleneck_base,vcnef_cns_base \
  --history-len 1 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-vcnef-cns-compare/cap1024-40ep
```

Expected: a larger-cap matched comparison using the same contract.

- [ ] **Step 2: Copy the final pairwise PNGs into `tmp/`**

Copy the prediction and residual galleries for quick inspection.

Expected: `tmp/` contains the latest `vcnef vs spectral` sample PNGs.

### Task 7: Write A Durable Summary And Caveat The Contract Correctly

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_vcnef_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/findings.md` (only if the result is actionable)

- [ ] **Step 1: Write the study summary**

Include:

- exact VCNeF source repo and commit
- local import path used
- why the comparison uses `history_len=1`
- all capped run roots
- side-by-side metrics for train and test
- direct statement of whether VCNeF beats or trails hybrid-spectral on:
  - `relative_l2`
  - `err_RMSE`
  - `fRMSE_high`
- whether visual artifacts tell a different story than the aggregate metrics

- [ ] **Step 2: Index the summary**

Add the summary to `docs/index.md` so later agents can find it without re-deriving the context.

- [ ] **Step 3: Add a finding only if there is a stable conclusion**

Examples of acceptable findings:

- VCNeF under the Markov one-step CNS contract clearly beats or trails `spectral_resnet_bottleneck_base`
- VCNeF is technically incompatible with the local equal-footing contract without invasive source changes

Do not add a finding for a noisy or obviously undertrained result.

### Task 8: Verification Before Claiming Completion

**Files:**
- Test: `tests/studies/test_pdebench_image128_models.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Test: run artifact roots from Tasks 5 and 6

- [ ] **Step 1: Run focused tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q
```

Expected: pass.

- [ ] **Step 2: Run compile verification**

Run:

```bash
python -m compileall -q scripts/studies/pdebench_image128
```

Expected: exit `0`.

- [ ] **Step 3: Confirm each claimed run is actually complete**

Required proof per run:

- tracked PID exited `0`
- `comparison_summary.json` exists
- both profile metrics files exist
- gallery PNGs exist and are fresh

Do not treat a tmux pane or partial log as completion proof.

## Stop Conditions

Stop and document the blocker instead of forcing the compare if any of these happen:

- the official VCNeF import path requires invasive edits inside the author source tree
- VRAM/runtime makes the 512-cap 10-epoch run infeasible on the local GPU
- the local runner cannot host VCNeF without changing the core metric/reporting contract
- the Markov `history_len=1` compare is rejected as scientifically insufficient by the user

## Acceptance Criteria

1. Official VCNeF source is cloned locally, pinned by commit, and importable.
2. `vcnef_cns_base` runs inside the local `2d_cfd_cns` runner without modifying author source.
3. A matched `history_len=1` comparison exists for the 512-cap slice at `10` and `40` epochs.
4. A matched `history_len=1` comparison exists for the 1024-cap slice at `40` epochs.
5. Standard metric JSON and PNG gallery artifacts exist for every claimed run.
6. A durable summary records the result and explicitly states the contract caveat: this is a Markov one-step compare, not a direct reuse of the existing `history_len=2` hybrid-spectral rows.
