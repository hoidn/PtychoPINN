# GNOT Capped CNS Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the official GNOT source locally, run it on the capped PDEBench `2d_cfd_cns` slices, and compare it to `spectral_resnet_bottleneck_base` visually and via the existing CNS metric suite.

**Architecture:** Keep the GNOT source external and pinned by commit; do not fork or rewrite the author repository. Use a dedicated conda environment for the compare if the active repo environment cannot supply a CUDA-capable DGL build. Add a thin local wrapper/profile that imports the official `models.mmgpt.GNOT`, converts each regular-grid CNS sample into the point-query plus input-function format that GNOT expects, and runs inside the existing `run_cfd_cns` runner so split logic, normalization, metrics, train/test eval emission, and comparison galleries stay identical. Keep the current local CNS contract `history_len=2`, `concat u[t-2:t] -> u[t]`; do not change the task contract just to suit the external code.

**Tech Stack:** Python, PyTorch, DGL, HDF5, local PDEBench CNS runner, external GNOT GitHub source, tmux for long runs.

---

## Context And Non-Negotiable Decisions

- Public GNOT source inspected: `HaoZhongkai/GNOT` at commit `5ee2e6925a43f9a340a6016bad4da2c82a452cbe`.
- The author repo is not a PyPI package and has no `setup.py` or `pyproject.toml`; treat installation as `git clone` plus a pinned local import root, not `pip install gnot`.
- The author training code (`train.py`) is not an equal-footing comparison path for this repo because it:
  - uses its own DGL dataset layer and pickle-based `MIODataset` contract,
  - expects query points plus a tuple of input functions rather than image tensors,
  - uses a different optimizer/scheduler recipe and default `rel2` training loss,
  - does not match the local CNS artifact/metric contract.
- Therefore the comparison should use the existing local CNS runner and reporting stack, with GNOT inserted as one additional model profile.
- Unlike VCNeF, GNOT does not require collapsing to a Markov-only setup. Its native operator interface can represent the existing local one-step CNS sample as:
  - query points `X`: flattened fixed `128x128` grid coordinates,
  - target field `Y`: next-step state on those points,
  - global parameters `Theta`: empty vector for v1,
  - one source-function branch input: `[x, y, u[t-2:t]]` at the same points.
- The first fair contract is the current local CNS one:
  - task: `2d_cfd_cns`
  - `history_len=2`
  - one-step prediction `concat u[t-2:t] -> u[t]`
  - same capped trajectory splits already used locally
  - same local metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low/mid/high`
  - same local training loss: `mse`
- Use `spectral_resnet_bottleneck_base` as the comparison row. Do not compare against the manually stronger `spectral_resnet_bottleneck_noshare` row in v1; keep the baseline canonical.

## Main Risks

1. GNOT depends on `dgl`, and the active repo environment may not have a CUDA-capable DGL build for its exact `torch` version. Selecting or preparing a dedicated compare environment is the first technical gate.
2. The author code is written around DGL graphs and `MultipleTensors`; the local wrapper must bridge image tensors to that interface without rewriting the author model.
3. Forward-time graph construction for `128x128 = 16384` query points may be expensive. If naive per-batch graph creation is too slow, the wrapper must cache a graph skeleton or stop and document the blocker.
4. The author recipe optimizes `rel2`, while the local CNS comparison contract currently optimizes `mse`. The v1 compare should keep `mse` for fairness to the local baselines and document that caveat explicitly.

## Planned Run Matrix

Primary execution tranche:

- small cap: `512 / 64 / 64` trajectories, `8` windows per trajectory
  - `10` epochs
  - `40` epochs
- larger cap: `1024 / 128 / 128` trajectories, `8` windows per trajectory
  - `40` epochs

Profiles per run:

- `spectral_resnet_bottleneck_base`
- `gnot_cns_base`

Optional follow-up only if the first tranche is technically clean:

- larger cap `10`-epoch run for learning-curve alignment
- local loss ablation `mse` vs `relative_l2` for GNOT only
- manual compare against `spectral_resnet_bottleneck_noshare`

## Output Contract

Each completed run must emit:

- standard `comparison_summary.json`
- per-profile `metrics_<profile>.json`
- train-split eval plus held-out test eval
- sample prediction gallery with ground truth plus both profiles
- sample residual gallery
- parameter counts
- source-provenance note containing the GNOT Git commit and local clone path

The comparison is incomplete unless both the metric JSON and the PNG galleries exist.

### Task 1: Choose A Dedicated Compare Environment Before Touching Model Code

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_blocker.md` only if the environment gate fails again
- Test: conda-env smoke commands

- [ ] **Step 1: Probe candidate conda environments**

Check at minimum:

- active `ptycho311`
- local fallback envs such as `ptycho-tf` and `ptycho311_2`

For each candidate, record:

- `torch` version
- CUDA availability
- whether `dgl` imports
- whether a DGL graph can move to CUDA
- whether the local spectral baseline can build and run one forward pass

- [ ] **Step 2: Select the environment only if it can run both stacks**

Acceptance gate for the chosen env:

- `import dgl` succeeds
- `dgl.graph(...).to("cuda")` succeeds
- `PYTHONPATH=<external gnot root> import models.mmgpt.GNOT` succeeds
- local `spectral_resnet_bottleneck_base` builds and runs one forward pass in the same env

Do not proceed if the env runs only GNOT or only the local baseline. The compare must stay equal-footing.

- [ ] **Step 3: If no existing env passes, create a dedicated compare env**

Preferred strategy:

- clone from the closest working torch env instead of mutating the active repo env
- pin the env name in the run notes, for example `gnot-compare-cu128`

The created env must still pass the Step 2 gate before adapter work begins.

### Task 2: Pin The GNOT Source And Clear The Dependency Gate

**Files:**
- Create: `.artifacts/external/gnot/` (external clone, untracked)
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot_source.json`
- Test: direct import smoke for `models.mmgpt.GNOT`

- [ ] **Step 1: Clone the official GNOT repo into an external artifact path**

Run:

```bash
git clone https://github.com/HaoZhongkai/GNOT .artifacts/external/gnot
```

Expected: local clone exists and is not committed into the repo index.

- [ ] **Step 2: Record the pinned source revision**

Run:

```bash
python - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path(".artifacts/external/gnot")
sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
payload = {"repo": "https://github.com/HaoZhongkai/GNOT", "commit": sha, "path": str(root.resolve())}
path = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot_source.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(path)
PY
```

Expected: `gnot_source.json` exists with repo URL, commit SHA, and absolute path.

- [ ] **Step 3: Install only the dependencies needed for model import**

Install:

- `einops`
- `dgl`

Do not downgrade or replace the repo’s active `torch` unless the environment is already incompatible.

- [ ] **Step 4: Smoke-test direct import from the pinned clone**

Run:

```bash
PYTHONPATH="$(pwd)/.artifacts/external/gnot:${PYTHONPATH}" python - <<'PY'
from models.mmgpt import GNOT
model = GNOT(trunk_size=2, branch_sizes=[10], output_size=4, n_layers=3, n_hidden=64, n_head=1, space_dim=2, n_experts=1, n_inner=4)
print(type(model).__name__)
PY
```

Expected: prints `GNOT` and exits `0`.

- [ ] **Step 5: Stop here if the chosen env still cannot supply CUDA DGL**

If `dgl` cannot be installed cleanly against the chosen compare env, stop and write the blocker instead of forcing a partial port.

### Task 3: Add Failing Tests For The GNOT Adapter Boundary

**Files:**
- Modify: `tests/studies/test_pdebench_image128_models.py`
- Modify: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Write failing model-builder tests for the GNOT import boundary**

Cover:

```python
def test_gnot_builder_blocks_cleanly_when_source_root_missing(): ...
def test_gnot_wrapper_returns_bchw_output_for_cns_contract(): ...
def test_gnot_wrapper_records_external_source_metadata(): ...
```

Expected: tests fail because `gnot_cns_base` and the adapter do not exist yet.

- [ ] **Step 2: Write a failing runner test for metadata pass-through**

Cover:

```python
def test_cfd_cns_passes_task_metadata_into_model_builder(): ...
```

Expected: fails because the builder does not yet receive task metadata.

- [ ] **Step 3: Run the focused red tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q
```

Expected: failures are the new missing GNOT path and metadata pass-through, not unrelated breakage.

### Task 4: Add A Local GNOT Adapter That Preserves The Image-Model Interface

**Files:**
- Create: `scripts/studies/pdebench_image128/gnot_adapter.py`
- Modify: `scripts/studies/pdebench_image128/models.py`

- [ ] **Step 1: Implement `gnot_adapter.py`**

Add:

- source-root resolution from `GNOT_ROOT` first, then default `.artifacts/external/gnot`
- clean `ModelBuildBlocker` when the source is absent or `dgl`/imports fail
- `GnotCnsModel` wrapper that:
  - accepts normalized local input as `B,C,H,W`
  - flattens spatial positions to `(N,2)` query points using the fixed CNS grid
  - flattens source-function values to one branch input of shape `(N, 2 + C_in)` where the first two columns are coordinates
  - uses empty `Theta` for v1
  - batches the query points through DGL the way the author model expects
  - returns `B,C_out,H,W`
  - exposes source provenance for metrics payloads

- [ ] **Step 2: Cache the regular-grid graph skeleton if needed**

The wrapper should not rebuild the same static query topology from scratch every forward unless profiling shows it is negligible.

- [ ] **Step 3: Extend the local model builder surface**

Modify `scripts/studies/pdebench_image128/models.py` to:

- accept optional `task_metadata` in `build_model_from_profile(...)`
- add a `gnot_cns_net` builder branch
- pass through model provenance in `describe_model(...)` when available

- [ ] **Step 4: Run focused model tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -q
```

Expected: GNOT-specific tests pass without breaking existing profiles.

### Task 5: Register A GNOT CNS Profile Without Changing The Existing Baseline Defaults

**Files:**
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Test: `tests/studies/test_pdebench_image128_models.py`

- [ ] **Step 1: Add the model-profile fields needed for GNOT**

Add only the knobs required now, for example:

- `gnot_hidden`
- `gnot_layers`
- `gnot_heads`
- `gnot_experts`
- `gnot_inner_multiplier`
- `gnot_mlp_layers`
- `gnot_attn_type`

Do not create a large profile matrix.

- [ ] **Step 2: Register `gnot_cns_base`**

Recommended starting point for the first local equal-footing row:

- `n_hidden=64`
- `n_layers=3`
- `n_head=1`
- `n_experts=1`
- `n_inner=4`
- `attn_type="linear"`
- `mlp_layers=3`

Reason: these are the author defaults in `args.py`, and they keep the first integration tranche close to the public source rather than inventing a repo-specific tuned profile.

Set `evidence_scope="readiness-only"` until a full comparison is complete.

- [ ] **Step 3: Re-run focused configuration tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -q
```

Expected: the new profile parses, builds, and does not disturb the existing profile IDs.

### Task 6: Keep The CNS Runner On The Existing Reporting Path

**Files:**
- Modify: `scripts/studies/pdebench_image128/cfd_cns.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Pass task metadata into `build_model_from_profile(...)`**

Required metadata for GNOT:

- `time_steps`
- `dx`
- `dy`
- `dt`
- `field_order`
- `history_len`

Do not change the training loop signature or the comparison-summary contract.

- [ ] **Step 2: Re-run focused runner tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_runner.py -q
```

Expected: runner tests pass and no existing profile path regresses.

### Task 7: Run The Small-Cap Matched Compare In The Existing Runner

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cap512-10ep-*`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cap512-40ep-*`
- Test: produced `comparison_summary.json`, `metrics_*.json`, gallery PNGs

- [ ] **Step 1: Launch the 512-cap 10-epoch compare in tmux**

Run:

```bash
GNOT_ROOT="$(pwd)/.artifacts/external/gnot" \
python scripts/studies/run_pdebench_image128_suite.py \
  --task-id 2d_cfd_cns \
  --mode readiness \
  --profile-ids spectral_resnet_bottleneck_base,gnot_cns_base \
  --history-len 2 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cap512-10ep
```

Expected: one run root containing both model rows and shared comparison outputs.

- [ ] **Step 2: Launch the 512-cap 40-epoch compare in tmux**

Run the same command with `--epochs 40` and a distinct `--output-root`.

Expected: same artifact set for the longer budget.

- [ ] **Step 3: Verify the required artifacts**

Check for:

- `comparison_summary.json`
- `metrics_spectral_resnet_bottleneck_base.json`
- `metrics_gnot_cns_base.json`
- sample prediction gallery
- sample residual gallery

### Task 8: Run The Larger-Cap 40-Epoch Compare

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cap1024-40ep-*`
- Test: produced `comparison_summary.json`, `metrics_*.json`, gallery PNGs

- [ ] **Step 1: Launch the 1024-cap 40-epoch compare**

Run:

```bash
GNOT_ROOT="$(pwd)/.artifacts/external/gnot" \
python scripts/studies/run_pdebench_image128_suite.py \
  --task-id 2d_cfd_cns \
  --mode readiness \
  --profile-ids spectral_resnet_bottleneck_base,gnot_cns_base \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cap1024-40ep
```

Expected: a larger-cap matched comparison using the same contract.

- [ ] **Step 2: Copy the final pairwise PNGs into `tmp/`**

Copy the prediction and residual galleries for quick inspection.

Expected: `tmp/` contains the latest `gnot vs spectral` sample PNGs.

### Task 9: Write A Durable Summary And Caveat The Contract Correctly

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/findings.md` (only if the result is actionable)

- [ ] **Step 1: Write the study summary**

Include:

- exact GNOT source repo and commit
- local import path used
- why the comparison kept `history_len=2`
- all capped run roots
- side-by-side metrics for train and test
- direct statement of whether GNOT beats or trails hybrid-spectral on:
  - `relative_l2`
  - `err_RMSE`
  - `fRMSE_high`
- whether visual artifacts tell a different story than the aggregate metrics
- explicit note that this is the official GNOT model under the local CNS `mse` contract, not a reproduction of the paper’s original training recipe

- [ ] **Step 2: Index the summary**

Add the summary to `docs/index.md` so later agents can find it without re-deriving the context.

- [ ] **Step 3: Add a finding only if there is a stable conclusion**

Examples of acceptable findings:

- GNOT under the local `history_len=2` CNS contract clearly beats or trails `spectral_resnet_bottleneck_base`
- GNOT is technically incompatible with the local equal-footing contract without invasive source changes

Do not add a finding for a noisy or obviously undertrained result.

### Task 10: Verification Before Claiming Completion

**Files:**
- Test: `tests/studies/test_pdebench_image128_models.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Test: run artifact roots from Tasks 6 and 7

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

- no candidate conda environment can provide CUDA-enabled DGL and still run the local spectral baseline
- the official GNOT import path requires invasive edits inside the author source tree
- regular-grid-to-DGL wrapping makes the 512-cap 10-epoch run infeasible on the local GPU/CPU budget
- the local runner cannot host GNOT without changing the core metric/reporting contract

## Acceptance Criteria

1. Official GNOT source is cloned locally, pinned by commit, and importable.
2. `gnot_cns_base` runs inside the local `2d_cfd_cns` runner without modifying author source.
3. A matched `history_len=2` comparison exists for the 512-cap slice at `10` and `40` epochs.
4. A matched `history_len=2` comparison exists for the 1024-cap slice at `40` epochs.
5. Standard metric JSON and PNG gallery artifacts exist for every claimed run.
6. A durable summary records the result and explicitly states the contract caveat: this is the official GNOT architecture under the local CNS `mse` training recipe, not a paper-faithful reproduction of the original GNOT training setup.
