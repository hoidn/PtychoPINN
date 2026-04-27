# PDEBench CNS Hybrid-Spectral Architecture Ablation Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the selected capped PDEBench `2d_cfd_cns` Hybrid-spectral architecture ablation using the already-implemented six-row pilot surface, identify per-family finalists under the fixed canonical CNS shell, confirm them on the larger-cap slice, and publish durable decision-support-only interpretation without widening scope or claiming benchmark completeness.

**Architecture:** Keep the work in three units. First, prove the current repo surface still matches the selected-item contract: the six Hybrid-spectral profiles, the task-local `pilot` mode, and the required emitted artifacts all work on the fixed CNS shell without reopening settled shell axes. Second, run a fresh `10`-epoch six-row pilot matrix, a bounded `40`-epoch follow-up, and a `1024 / 128 / 128` finalist confirmation, all under the same capped `history_len=2` CNS MSE contract with explicit artifact-audit gates and tmux/PID discipline. Third, write the ablation summary, sync the CNS summary and discoverability docs, and update the progress ledger while preserving the capped decision-support-only boundary.

**Tech Stack:** PATH `python`; conda env `ptycho311` for long CNS runs; PyTorch/Lightning; existing PDEBench CNS runner and gallery helper under `scripts/studies/pdebench_image128/`; pytest; compileall; tmux; Markdown/JSON/PNG artifacts under `.artifacts/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation`
- Status: pending
- Date: `2026-04-23`
- Scope owner: Roadmap Phase 2 selector-authorized capped CNS follow-up lane
- Selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/selected-item-context.md`
- Plan path authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- CNS summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/`

This document supersedes the prior revision at this same plan path and is the new execution authority for this backlog item.

## Inputs Read

- `AGENTS.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/0/items/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-cns-hybrid-spectral-architecture-ablation/execution_plan.md` (previous revision, background only)
- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

## Objective

- Run a focused PDEBench `2d_cfd_cns` Hybrid-spectral architecture ablation that isolates spectral weight sharing and spectral bottleneck depth under the already-settled canonical CNS shell.
- Use the existing six-profile matrix and `pilot` evidence classification to generate fresh bounded decision-support evidence, then confirm only the per-family finalists on the larger-cap slice.
- Publish a durable repo-local summary and ledger update that make the result discoverable without confusing this capped lane with the roadmap’s full-training benchmark gate.

## Scope

- Keep this item **CNS only**:
  - task: `2d_cfd_cns`
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - resolution: `128x128`
- Keep the current capped CNS contract fixed for the fresh `10`-epoch and `40`-epoch runs:
  - train / val / test trajectories: `512 / 64 / 64`
  - `max_windows_per_trajectory=8`
  - `history_len=2`
  - batch size `4`
  - training loss `mse`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Keep the current larger-cap confirmation fixed to finalists only:
  - train / val / test trajectories: `1024 / 128 / 128`
  - `max_windows_per_trajectory=8`
  - `history_len=2`
  - batch size `4`
  - training loss `mse`
  - same metric family as above
- Keep the fixed shell explicit on every ablation row:
  - `hidden_channels=32`
  - `fno_modes=12`
  - `fno_blocks=4`
  - `hybrid_downsample_steps=2`
  - `hybrid_resnet_blocks=6`
  - `hybrid_skip_connections=True`
  - `hybrid_skip_style="add"`
  - `hybrid_upsampler="pixelshuffle"`
  - `spectral_bottleneck_modes=12`
  - `spectral_bottleneck_gate_init=0.1`
  - `spectral_bottleneck_gate_mode="shared"`
- Cover only the Hybrid-spectral axes:
  - `spectral_bottleneck_share_weights`: `True` vs `False`
  - `spectral_bottleneck_blocks`: `6`, `8`, `10`
- Reuse the already-implemented six-profile matrix:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks8`
  - `spectral_resnet_bottleneck_shared_blocks10`
  - `spectral_resnet_bottleneck_noshare`
  - `spectral_resnet_bottleneck_noshare_blocks8`
  - `spectral_resnet_bottleneck_noshare_blocks10`
- Reuse the already-implemented task-local `pilot` mode as the only allowed ranked evidence surface for this backlog item.

## Explicit Non-Goals

- Do not widen this work into CDI, ptychography, Darcy, SWE, or `/home/ollie/Documents/neurips/` artifact assembly.
- Do not reopen skip routing, upsampler choice, physics regularization, `history_len=1`, higher-mode `32/32`, local-vs-spectral family compare, author FFNO, or GNOT as primary axes inside this item.
- Do not relabel capped runs as benchmark rows or paper-facing competitiveness evidence.
- Do not change the fixed CNS contract to MAE, a different batch size, different split caps, different history length, or different metric family just because other studies used them.
- Do not create worktrees.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not generalize this item into a broad runner refactor. Only fix a minimal blocking regression if the preflight checks fail.

## Steering And Roadmap Constraints

- Steering requires equal-footing comparisons. For this item, equal footing means the dataset file, split caps, `history_len`, `max_windows_per_trajectory`, training loss, batch size, metric family, and fixed shell stay identical across all rows inside a given run tranche.
- The roadmap and suite plan require smoke and pilot outputs to remain decision-support only. This item may rank rows only inside `pilot` mode for internal finalist selection.
- The roadmap explicitly allows this backlog item now as a bounded capped CNS follow-up lane. It is not the generic suite-wide focused-ablation gate, which remains downstream of full-training primary profiles.
- The approved design and current CNS summary already promoted `hybrid_skip_style=add` and `hybrid_upsampler=pixelshuffle` into the canonical CNS shell. This item must treat them as fixed controls.
- The steering concern about FFNO-before-GNOT is not part of this item’s scope. The internal Hybrid-spectral ablation must remain independent from the external-baseline queue.
- If a fresh read of the roadmap, selected-item context, or ledger no longer authorizes this capped lane, stop and mark the item blocked rather than widening or reclassifying the work.

## Roadmap Gate Position

- This backlog item is authorized by the roadmap text dated `2026-04-20` plus the `2026-04-21` CNS routing update that explicitly allows a “bounded CNS follow-up compare/ablation that stays explicitly capped and decision-support-only.”
- This backlog item is one of the separate capped CNS ablation lanes called out beside the `history_len=1` compare and the `modes32` compare. It is not serialized behind those lanes unless a new blocker appears.
- Full-training CNS benchmark completeness is still blocked because `hybrid_resnet_cns`, `fno_base`, and `unet_strong` have not yet been run on the full available training split for the official file.
- The plan must therefore preserve two truths at once:
  - execution is allowed now because this is a selector-authorized capped lane
  - benchmark completeness remains open because the roadmap’s full-training gate is not being attempted here

## Prerequisite Status

### Satisfied

- The official CNS data file is staged and verified:
  - bytes: `55,050,245,208`
  - MD5: `21969082d0e9524bcc4708e216148e60`
- The supervised CNS adapter, metrics, and artifact-writing path are implemented and already exercised on real data.
- The canonical CNS Hybrid shell is fixed at skip-add plus pixelshuffle.
- The current CNS loss contract is already corrected to `mse`, aligned with the official PDEBench FNO/U-Net forward baseline code.
- The six-row Hybrid-spectral profile matrix already exists in `scripts/studies/pdebench_image128/run_config.py`.
- The task-local `pilot` mode already exists in `scripts/studies/pdebench_image128/cfd_cns.py` and `scripts/studies/pdebench_image128/reporting.py`.
- Tests already cover:
  - six-profile presence and fixed-shell invariants
  - explicit-profile requirement for `pilot` mode
  - `pilot` comparison summary semantics
  - emitted `model_profile_*.json`, `metrics_*.json`, `comparison_summary.json`, and `comparison_summary.csv`

### Open But Not Blocking

- No fresh ablation run roots for this backlog item exist yet under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/`.
- The durable ablation summary, CNS summary update, docs index updates, and progress-ledger entry for this backlog item do not exist yet.
- Benchmark-complete CNS and suite-level claims remain blocked by the roadmap’s full-training gate; this item must not try to satisfy that gate.

## Prior Evidence To Reuse

- Corrected capped CNS MSE anchor:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- Shared-vs-non-shared capped spectral compare:
  - summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md`
  - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep`
- Shared-spectral `40`-epoch context from the FFNO-close bottleneck compare:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- Canonical shell promotion evidence:
  - skip-add: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-skip-study/cns-skipadd-vs-base-10ep`
  - pixelshuffle: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z`

These are provenance inputs and comparison context only. They are not substitutes for the fresh run roots required by this plan.

## Implementation Architecture

- **Preflight + Contract Validation Unit:** reuse the current runner surface in `scripts/studies/pdebench_image128/{run_config.py,cfd_cns.py,reporting.py}` and its coverage in `tests/studies/test_pdebench_image128_{models,runner}.py`. Default expectation: no code changes. Only patch a minimal blocking regression if the required preflight checks fail.
- **Execution + Artifact Audit Unit:** fresh run roots under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/` own the `10`-epoch matrix, the bounded `40`-epoch follow-up, and the `1024 / 128 / 128` finalist confirmation. Each run root must carry invocation, split, summary, metrics, model-profile, comparison, and gallery artifacts plus a generated `artifact_audit.json`.
- **Interpretation + State Unit:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, `docs/index.md`, `docs/studies/index.md`, `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, and `docs/findings.md` only if justified, own the durable interpretation and must preserve the capped-lane / benchmark-incomplete boundary.

## Concrete File And Artifact Targets

### Expected code and test changes

- Default expectation: no production code changes are required because the six-profile matrix and `pilot` mode already exist.
- Modify only if the preflight checks fail and a minimal blocker must be fixed:
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/reporting.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py`
  - `scripts/studies/run_pdebench_image128_suite.py` only if CLI passthrough is the blocker

### Expected durable documentation and state changes

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if a stable cross-study rule emerges: `docs/findings.md`

### Fresh artifacts required

- Create study root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/`
- Create fresh run roots under that study root:
  - `cns-hybrid-spectral-pilot-10ep-<timestamp>`
  - `cns-hybrid-spectral-pilot-40ep-<timestamp>`
  - `cns-hybrid-spectral-pilot-1024cap-40ep-<timestamp>`
- Require per run root:
  - `invocation.json`
  - `invocation.sh`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_<profile>.json`
  - `model_profile_<profile>.json`
  - `comparison_<profile>_sample0.png`
  - `comparison_<profile>_sample0.npz`
  - `artifact_audit.json`
- Require rendered family galleries:
  - `gallery_shared_sample0.png`
  - `gallery_shared_sample0_error.png`
  - `gallery_noshare_sample0.png`
  - `gallery_noshare_sample0_error.png`
  - a selected-row `40`-epoch gallery
  - a finalist-pair `1024 / 128 / 128` gallery

## Long-Run Execution Rules

- Launch every GPU run in tmux and activate `ptycho311` first.
- Track the exact launched PID for each long-running command. Do not use broad `pgrep -f` polling loops.
- Never launch a duplicate run against the same `--output-root`.
- Treat a run as complete only when:
  - the tracked PID exits with code `0`
  - the required per-run artifacts exist under the intended run root
  - those artifacts were freshly written by the launched run

## Required Deterministic Checks

These two commands come directly from the selected-item context and remain mandatory gates for this backlog item:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Use stronger focused checks in addition to these commands, not instead of them.

## Task 1: Preflight The Current Six-Row Pilot Surface

**Files:**
- Default expectation: no repo file changes
- Modify only if a blocking regression is exposed:
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/reporting.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py`

- [ ] **Step 1: Run the required deterministic checks before any GPU work**

Run exactly:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

- [ ] **Step 2: Run stronger focused preflight selectors that target this backlog item’s surfaces**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py -k "spectral_resnet_bottleneck"
pytest -q tests/studies/test_pdebench_image128_runner.py -k "pilot or cfd_cns"
```

These checks must confirm:

- the six profile IDs listed in Scope exist
- every profile pins the fixed shell fields
- `pilot` mode requires explicit `--profiles`
- `pilot` comparison summaries emit `evidence_scope == "capped_decision_support_only"`
- `pilot` runs emit `comparison_summary.csv`, `model_profile_*.json`, and `metrics_*.json`

- [ ] **Step 3: Fix only a minimal blocker if the preflight fails**

If any Step 1 or Step 2 check fails:

- patch only the smallest runner/config/reporting regression that blocks this backlog item
- keep the fix inside the PDEBench CNS path
- rerun Step 1 and Step 2 until green
- do not widen into unrelated refactors or adjacent backlog items

**Verification for Task 1**

- the two required selected-item commands pass
- the targeted spectral/pilot selectors pass
- if a regression fix was needed, it is confined to the minimal PDEBench CNS runner surface and the required checks are green again before long runs start

## Task 2: Run The Fresh `10`-Epoch Six-Row Pilot Matrix

**Artifacts:**
- Create: fresh `10`-epoch pilot run root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/`

- [ ] **Step 1: Launch the six-row `10`-epoch matrix in `pilot` mode**

Use tmux, activate `ptycho311`, and launch the run with a fresh output root:

```bash
RUN_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-pilot-10ep-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root "$RUN_ROOT" \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks8,spectral_resnet_bottleneck_shared_blocks10,spectral_resnet_bottleneck_noshare,spectral_resnet_bottleneck_noshare_blocks8,spectral_resnet_bottleneck_noshare_blocks10 \
  --history-len 2 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] **Step 2: Generate an explicit artifact audit for the fresh run root**

Write and execute this verification script against the fresh run root:

```bash
RUN_ROOT="<fresh_10ep_run_root>"
RUN_ROOT_ENV="$RUN_ROOT" python - <<'PY'
from pathlib import Path
import json
import os

run_root = Path(os.environ["RUN_ROOT_ENV"])
profiles = [
    "spectral_resnet_bottleneck_base",
    "spectral_resnet_bottleneck_shared_blocks8",
    "spectral_resnet_bottleneck_shared_blocks10",
    "spectral_resnet_bottleneck_noshare",
    "spectral_resnet_bottleneck_noshare_blocks8",
    "spectral_resnet_bottleneck_noshare_blocks10",
]
required_common = [
    "invocation.json",
    "invocation.sh",
    "dataset_manifest.json",
    "split_manifest.json",
    "comparison_summary.json",
    "comparison_summary.csv",
]
missing = [name for name in required_common if not (run_root / name).exists()]
for profile in profiles:
    for name in [
        f"metrics_{profile}.json",
        f"model_profile_{profile}.json",
        f"comparison_{profile}_sample0.png",
        f"comparison_{profile}_sample0.npz",
    ]:
        if not (run_root / name).exists():
            missing.append(name)
payload = {
    "run_root": str(run_root),
    "profiles": profiles,
    "required_common": required_common,
    "missing": missing,
    "passed": not missing,
}
(run_root / "artifact_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if missing:
    raise SystemExit(f"missing required artifacts: {missing}")
print("artifact audit passed")
PY
```

- [ ] **Step 3: Render family galleries with the existing helper**

Render the shared-family and non-shared-family galleries:

```bash
python scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py \
  --run-root "$RUN_ROOT" \
  --sample-index 0 \
  --baseline-profile spectral_resnet_bottleneck_base \
  --variant-profiles spectral_resnet_bottleneck_shared_blocks8,spectral_resnet_bottleneck_shared_blocks10 \
  --output-png "$RUN_ROOT/gallery_shared_sample0.png" \
  --output-error-png "$RUN_ROOT/gallery_shared_sample0_error.png"

python scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py \
  --run-root "$RUN_ROOT" \
  --sample-index 0 \
  --baseline-profile spectral_resnet_bottleneck_noshare \
  --variant-profiles spectral_resnet_bottleneck_noshare_blocks8,spectral_resnet_bottleneck_noshare_blocks10 \
  --output-png "$RUN_ROOT/gallery_noshare_sample0.png" \
  --output-error-png "$RUN_ROOT/gallery_noshare_sample0_error.png"
```

- [ ] **Step 4: Rank each family and choose the bounded `40`-epoch challengers**

Within each family, rank rows by:

1. lower `relative_l2`
2. if tied, lower `err_nRMSE`
3. if still tied, lower `fRMSE_high`

Record:

- the best shared-family row
- the best non-shared-family row
- whether `blocks8` or `blocks10` beat the corresponding family anchor enough to justify a `40`-epoch follow-up slot

This ranking is valid only for within-item `pilot` selection. It is not a suite or paper ranking.

**Verification for Task 2**

- the fresh `10`-epoch run exits `0`
- `comparison_summary.json` records `mode == "pilot"`
- `comparison_summary.json` records `evidence_scope == "capped_decision_support_only"`
- `artifact_audit.json` passes
- the shared-family and non-shared-family galleries render successfully
- the selected `40`-epoch challengers are traceable to the recorded metric ordering

## Task 3: Run The Bounded `40`-Epoch Pilot Follow-Up

**Artifacts:**
- Create: fresh `40`-epoch pilot run root under the same study artifact root

- [ ] **Step 1: Build the bounded `40`-epoch profile list**

Always include:

- `spectral_resnet_bottleneck_base`
- `spectral_resnet_bottleneck_noshare`

Then add:

- at most one extra shared-family row from `{spectral_resnet_bottleneck_shared_blocks8, spectral_resnet_bottleneck_shared_blocks10}`
- at most one extra non-shared-family row from `{spectral_resnet_bottleneck_noshare_blocks8, spectral_resnet_bottleneck_noshare_blocks10}`

Only add a challenger if it beat its family anchor under Task 2’s ordering. The resulting `40`-epoch profile list must contain two to four rows, never all six.

- [ ] **Step 2: Launch the bounded `40`-epoch pilot run**

Use the same capped contract and a fresh output root:

```bash
RUN_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-pilot-40ep-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root "$RUN_ROOT" \
  --profiles <selected_40ep_profile_csv> \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] **Step 3: Re-run the artifact audit and render the selected-row gallery**

Repeat Task 2’s artifact-audit script using only the selected `40`-epoch profiles, then render a single selected-row gallery with the existing helper.

- [ ] **Step 4: Choose the larger-cap finalists**

Using the same ordering as Task 2, pick:

- one shared-family finalist
- one non-shared-family finalist

If the family anchor remains best, keep the anchor as the finalist for that family.

**Verification for Task 3**

- the selected profile list obeys the bounded family rule
- the `40`-epoch run exits `0`
- `comparison_summary.json` still records `mode == "pilot"` and `evidence_scope == "capped_decision_support_only"`
- `artifact_audit.json` passes for the selected rows
- the larger-cap finalists are traceable back to the recorded `10`-epoch and `40`-epoch ordering

## Task 4: Run The `1024 / 128 / 128` Finalist Confirmation

**Artifacts:**
- Create: fresh larger-cap pilot run root under the same study artifact root

- [ ] **Step 1: Launch only the two family finalists on the larger-cap slice**

Keep:

- `mode=pilot`
- `history_len=2`
- `epochs=40`
- `batch_size=4`
- `max_windows_per_trajectory=8`
- training loss `mse`
- same dataset file and metric family

Change only the split caps to `1024 / 128 / 128`:

```bash
RUN_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-pilot-1024cap-40ep-$(date -u +%Y%m%dT%H%M%SZ)"
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root "$RUN_ROOT" \
  --profiles <best_shared_profile>,<best_noshare_profile> \
  --history-len 2 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 1024 \
  --max-val-trajectories 128 \
  --max-test-trajectories 128 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

- [ ] **Step 2: Audit the finalist run and compare it against the `40`-epoch capped slice**

Generate `artifact_audit.json` for the finalist pair, then record whether:

- aggregate held-out error improves or regresses
- `fRMSE_high` improves or regresses
- the train/test gap shrinks or widens
- the family ordering changes or stays stable

Render a final finalist-pair gallery for sample `0`.

**Verification for Task 4**

- the larger-cap finalist run exits `0`
- `comparison_summary.json` records `mode == "pilot"` and `evidence_scope == "capped_decision_support_only"`
- `artifact_audit.json` passes for both finalist rows
- the comparison against the `40`-epoch slice is captured in the summary draft
- the item still treats the result as capped decision-support evidence only

## Task 5: Publish The Durable Interpretation And State Updates

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify only if justified: `docs/findings.md`

- [ ] **Step 1: Write the durable ablation summary**

The summary must state:

- this item’s roadmap gate position as a selector-authorized capped CNS follow-up lane
- the fixed CNS contract and fixed shell invariants
- the exact six-row matrix
- the fresh `10`-epoch, `40`-epoch, and `1024 / 128 / 128` run roots
- the family-ranking rule and finalist-selection rule
- the chosen `40`-epoch rows and larger-cap finalists
- the key train/test metrics and gallery paths
- whether a current best local spectral CNS row was identified under this capped contract
- the claim boundary that the result remains capped decision-support evidence only
- the explicit reminder that the roadmap’s full-training CNS benchmark gate is still open

- [ ] **Step 2: Update the CNS summary with a concise architecture-ablation subsection**

Add a new subsection to `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md` that points to the ablation summary and records whether the current best bounded spectral row is:

- shared `blocks=6`
- non-shared `blocks=6`
- a deeper shared variant
- a deeper non-shared variant

This subsection must preserve the benchmark-incomplete wording.

- [ ] **Step 3: Promote a durable finding only if the result is stable enough**

Update `docs/findings.md` only if the evidence is strong enough to act as project guidance, for example:

- deeper shared or non-shared depth clearly beats the corresponding family anchor on both capped slices
- `blocks=6` remains the best bounded spectral depth and the deeper rows do not justify future carry-forward

If the result is ambiguous, keep the interpretation in the ablation summary only.

- [ ] **Step 4: Update discoverability and ledger state**

Update:

- `docs/index.md` so the new summary is discoverable from the initiative docs
- `docs/studies/index.md` so the CNS Hybrid-spectral ablation is discoverable inside the PDEBench suite section
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` with:
  - the completed backlog item
  - exact plan path and summary path
  - exact fresh run roots
  - verification evidence
  - `evidence_scope: capped_decision_support_only`
  - `metric_interpretation: decision_support_not_benchmark_performance`
  - `performance_assessment_complete: false`
  - wording that the roadmap full-training benchmark gate remains open

- [ ] **Step 5: Re-run the required deterministic checks after repo edits**

Run again:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

**Verification for Task 5**

- the durable summary, CNS summary, and progress ledger cite the same winning rows and run roots
- the summary and ledger preserve the capped-lane and benchmark-incomplete boundaries
- any new finding is narrower than the evidence and does not overclaim
- the required deterministic checks pass after repo edits

## Final Completion Criteria

- the current repo surface is preflighted successfully before any GPU run
- the fresh `10`-epoch six-row matrix completes on the fixed capped CNS contract under `pilot` mode
- the bounded `40`-epoch follow-up obeys the per-family selection rule and stays in `pilot` mode
- the `1024 / 128 / 128` confirmation runs only the two family finalists and stays in `pilot` mode
- every fresh run root has a passing `artifact_audit.json`
- the durable summary states clearly whether a current best local spectral CNS row was identified under this capped contract
- the repo records stay within the selected backlog scope, preserve roadmap order, and explicitly record that this item remains benchmark-incomplete capped decision-support evidence only
