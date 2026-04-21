# Phase 2 PDEBench SWE Primary Smoke Gate Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the selected PDEBench 2D Shallow Water Equations primary benchmark can proceed to longer benchmark-performance execution by pinning data/metric/split contracts and running bounded one-step smoke passes for Hybrid ResNet, FNO, and U-Net. This smoke gate is readiness evidence only, not a model-performance assessment.

**Architecture:** Keep this as a Roadmap Phase 2 prerequisite smoke/data-contract gate, not full PDE execution and not benchmark-performance evidence. Add a narrow supervised PDEBench SWE study harness that owns HDF5 introspection, deterministic trajectory splits, one-step datasets, local `err_nRMSE` metrics, tiny model smoke runs, and provenance; it must not route SWE through the ptychographic physics loss or stable CDI modules. Write bulky machine artifacts under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/` and the durable gate decision at `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`.

**Tech Stack:** Python 3.11 via PATH `python`, `h5py`, NumPy, PyTorch, Lightning optional but not required for this smoke harness, `neuralop.models.FNO` when available, existing `ptycho_torch` Hybrid ResNet components, Markdown/JSON/CSV artifacts, tmux with `ptycho311` for long-running smoke commands.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Title: Phase 2 PDEBench SWE Primary Smoke/Data-Contract Gate
- Status: pending
- Spec/Source:
  - Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
  - Roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - Tranche context: `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-pdebench-swe-primary-smoke-gate/tranche-context.md`
  - Phase 1 selection: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- Plan path: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-primary-smoke-gate/execution_plan.md`

## Compliance Matrix

- [ ] **Roadmap Phase Order:** Execute only the selected Roadmap Phase 2 smoke prerequisite slice. Do not start longer Phase 2 training, Phase 3 CDI anchor regeneration, Phase 4 `256x256` CDI scaling, or Phase 5 paper-facing artifact assembly.
- [ ] **Primary Benchmark Pin:** Use PDEBench SWE `2D_rdb_NA_NA.h5` from the official `swe` download path or DaRUS datafile `133021`; do not switch to `2d_cfd`, `ns_incom`, generated NS/CFD subsets, PDEArena, or OpenFWI without recording a pivot decision.
- [ ] **One-Step Gate:** The smoke gate is one-step next-state prediction only. Autoregressive rollout is a later optional extension and cannot replace this gate.
- [ ] **Shared Contract Across Models:** Hybrid ResNet-compatible, FNO, and U-Net smoke runs must use the same HDF5 file, split manifest, horizon, normalization, batch limits, and `err_nRMSE` implementation.
- [ ] **Smoke Evidence Boundary:** Treat all smoke metrics as data/contract/runtime sanity artifacts only. Smoke metrics cannot rank models, trigger a performance pivot, or satisfy the Phase 2 competitiveness gate.
- [ ] **Local Baseline Requirement:** Run tiny FNO and U-Net one-step smoke baselines, or write explicit baseline blocker records explaining why either could not be made runnable quickly.
- [ ] **Metric License Boundary:** Do not import upstream PDEBench metric code if license terms are unclear. Implement and document a local equivalent `err_nRMSE` formula, with hand-calculated unit tests.
- [ ] **Artifact Hygiene:** Store HDF5 manifests, split manifests, metrics, logs, and raw provenance under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/`. Do not commit the HDF5 file, large logs, checkpoints, or raw datasets.
- [ ] **Durable Summary:** Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md` with exactly one final decision: proceed with longer SWE execution, pivot to OpenFWI FlatVel-A, or block for human decision.
- [ ] **Discoverability:** Because this tranche adds durable study tooling and a roadmap gate summary, update `docs/studies/index.md` for the smoke runbook and `docs/index.md` for the durable smoke-gate summary.
- [ ] **Stable Modules:** Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- [ ] **No Worktrees:** Use the current checkout only.
- [ ] **Interpreter Policy:** Use PATH `python` in commands and subprocess examples. Do not introduce repository-specific interpreter wrappers.
- [ ] **Long-Run Guardrail:** Use tmux with `ptycho311` for the bounded smoke execution if it is not clearly sub-minute. Track the exact launched PID with `cmd ... & pid=$!; wait "$pid"` and do not launch duplicate runs writing the same `--output-root`.
- [ ] **Freshness Gate:** The official Phase F smoke run must record a run ID, start timestamp, tracked PID, and exit code under `logs/`; final verification must reject any required manifest/model artifact whose mtime predates the start marker or whose recorded run ID/provenance does not match the tracked run. Reusing `--output-root` is allowed only with this freshness check.
- [ ] **Dirty Worktree Safety:** Preserve unrelated dirty files. Only touch files named in this plan unless a focused implementation blocker requires adding a narrow companion file.

## Spec Alignment

- **Normative roadmap phase:** Phase 2 - Deep PDE Benchmark Execution.
- **Covered slice:** data access, data contract, split manifest, one-step metric, bounded data-load smoke, and tiny Hybrid ResNet/FNO/U-Net smoke runs for the selected primary benchmark.
- **Required durable output:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`.
- **Required ignored support root:** `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/`.
- **Required support artifacts:**
  - `dataset_manifest.json`
  - `hdf5_metadata.json`
  - `split_manifest.json`
  - `normalization_stats.json`
  - `runs/hybrid_resnet/metrics.json` or `runs/hybrid_resnet/blocker.json`
  - `runs/fno/metrics.json` or `runs/fno/blocker.json`
  - `runs/unet/metrics.json` or `runs/unet/blocker.json`
  - `runs/*/provenance.json`
  - `invocation.json` and `invocation.sh`
  - `logs/smoke.run_id`, `logs/smoke.started_at_ns`, `logs/smoke.pid`, and `logs/smoke.exit_code`
  - optional raw logs under `logs/`
- **Explicit non-goals:** no full PDE benchmark result, no production training, no spectral/local ablations beyond tiny smoke passes, no final Phase 2 execution summary unless the whole Phase 2 gate is unexpectedly satisfied, no OpenFWI fallback execution, no CDI anchor regeneration, no CDI baselines, no `256x256` scaling, no `/home/ollie/Documents/neurips/` artifacts, no manuscript prose, no worktree, and no stable core physics/model edits.

## Documents Read For This Plan Draft

- User-provided AGENTS/CLAUDE instructions for `/home/ollie/Documents/PtychoPINN`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-pdebench-swe-primary-smoke-gate/tranche-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-primary-smoke-gate/plan-phase/plan_path.txt`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate-plan-review.json`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/index.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/templates/` listing
- `docs/plans/templates/implementation_plan.md`
- `docs/findings.md`
- `docs/studies/index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_anchor_regeneration_plan.md`
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/execution_report.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/workflows/pytorch.md`
- `docs/TESTING_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/DATA_MANAGEMENT_GUIDE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-0-evidence-inventory/execution_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-1-pde-benchmark-selection/execution_plan.md`
- `pyproject.toml`
- `scripts/studies/invocation_logging.py`
- `ptycho_torch/generators/hybrid_resnet.py`
- `ptycho_torch/generators/fno_vanilla.py`
- `ptycho_torch/generators/cnn.py`
- `tests/studies/test_hybrid_resnet_mode_skip_sweep.py`
- `workflows/examples/neurips_hybrid_resnet_plan_impl_review.yaml`

## Plan Review Findings Addressed

- `PLAN-H1` (`Final smoke verification can pass stale run artifacts`): resolved by adding a Phase F freshness gate. The official smoke launch now records a unique `run_id`, start timestamp, tracked PID, and exit code; the smoke runner must propagate the run ID into root contract JSONs and per-model artifacts; and the verification script must reject artifacts older than the start marker or artifacts whose `run_id`/provenance does not match the tracked run. This preserves the approved design, roadmap phase order, and selected tranche scope.

## Implementation Architecture

This tranche needs an Implementation Architecture section because it crosses external data access, HDF5 contracts, deterministic split ownership, metric semantics, model adaptation, command/provenance IO, ignored machine artifacts, and durable roadmap-gate documentation. A single implementation unit would make it too easy for a broad helper or broad test file to absorb unrelated responsibilities and obscure the Phase 2 pivot gate.

### Material Architectural Decision To Pin

The design and roadmap require a Hybrid ResNet PDE smoke path, but they do not require routing PDEBench SWE through the ptychographic physics `PtychoPINN_Lightning` workflow. That exact physics workflow expects CDI diffraction/probe/object contracts and would be the wrong boundary for SWE state prediction. This plan therefore uses a supervised PDE smoke wrapper that reuses Hybrid ResNet architectural components and labels the result as `Hybrid ResNet-compatible`; it must not claim to be the same CDI physics training path. If implementation review requires the exact CDI Lightning module for SWE, block for human decision because that would require a new data/physics-contract design.

### Unit 1: Data Source and HDF5 Manifest

- **Owns:** locating/staging `2D_rdb_NA_NA.h5`, recording source URL or DaRUS ID `133021`, license/access notes, file size/mtime/checksum, disk feasibility, recursive HDF5 dataset inventory, and selected state-dataset/axis mapping.
- **Proposed files:**
  - Create: `scripts/studies/pdebench_swe/__init__.py`
  - Create: `scripts/studies/pdebench_swe/manifest.py`
  - Test: `tests/studies/test_pdebench_swe_manifest.py`
- **Stable interfaces/artifacts:**
  - `dataset_manifest.json`
  - `hdf5_metadata.json`
  - CLI arguments: `--data-file`, `--dataset-source`, `--dataset-source-url`, `--dataset-darus-id`, `--license-note`, `--state-dataset`, `--axis-order`
- **Must not own:** split construction, normalization, model training, metric computation, or final proceed/pivot/block decision.
- **Dependency direction:** provides pinned data identity and selected tensor layout for Units 2-6.
- **Compatibility boundary:** auto-detection may select the state dataset only when unambiguous; ambiguous HDF5 structure must exit before training and write a blocker rather than guessing.
- **Focused tests:** synthetic HDF5 recursion, checksum/size/mtime manifest, candidate dataset detection, ambiguous-dataset blocker, JSON schema for manifest outputs.

### Unit 2: Split Manifest and One-Step Dataset Contract

- **Owns:** deterministic trajectory-level 80/10/10 split with seed `20260420`, one-step `(t -> t+1)` pair enumeration, bounded subset selection, HDF5-backed lazy loading, channel/axis normalization to `torch.Tensor` shape `(B, C, H, W)`, and pad/crop policy needed by downsampling models.
- **Proposed files:**
  - Create: `scripts/studies/pdebench_swe/splits.py`
  - Create: `scripts/studies/pdebench_swe/data.py`
  - Test: `tests/studies/test_pdebench_swe_splits_data.py`
- **Stable interfaces/artifacts:**
  - `split_manifest.json` with `seed=20260420`, split ratios, trajectory IDs per split, source file identity, state dataset, axis order, and pair counts.
  - Dataset item contract: `{"input": FloatTensor[C,H,W], "target": FloatTensor[C,H,W], "trajectory_id": int, "time_index": int}`.
  - CLI arguments: `--split-seed 20260420`, `--train-fraction 0.8`, `--val-fraction 0.1`, `--test-fraction 0.1`, `--max-train-trajectories`, `--max-val-trajectories`, `--max-test-trajectories`, `--max-pairs-per-trajectory`, `--pad-multiple`.
- **Must not own:** metric formulas, model definitions, external data download, or final summary prose.
- **Dependency direction:** consumes Unit 1 HDF5 metadata and feeds Units 3-6.
- **Compatibility boundary:** all three model arms must consume the same split manifest and normalization stats; no model-specific split or silent shuffle is allowed.
- **Focused tests:** deterministic split identity, disjoint trajectory IDs, 80/10/10 ratio rounding for small and large trajectory counts, one-step pair count, lazy HDF5 reads, axis conversion, pad/crop round-trip.

### Unit 3: Normalization and Metric Contract

- **Owns:** normalization statistics computed only from the train split, transform/inverse-transform helpers, local `err_RMSE` and `err_nRMSE` equivalents, per-channel and aggregate metric JSON records, and license/protocol note saying upstream PDEBench code was not imported unless explicitly cleared.
- **Proposed files:**
  - Create: `scripts/studies/pdebench_swe/metrics.py`
  - Test: `tests/studies/test_pdebench_swe_metrics.py`
- **Stable interfaces/artifacts:**
  - `normalization_stats.json`
  - Metric payload keys: `err_RMSE`, `err_nRMSE`, `per_channel.err_RMSE`, `per_channel.err_nRMSE`, `num_eval_batches`, `num_eval_pairs`, `normalization`, `horizon`.
  - Formula: `err_nRMSE = sqrt(sum((prediction - target)^2) / max(sum(target^2), eps))` over the documented evaluation axes, with per-channel values computed before aggregate reporting.
- **Must not own:** model forward passes, split creation, or published-SOTA comparison rows.
- **Dependency direction:** consumes Unit 2 train/eval tensors and feeds Units 5-6 metrics.
- **Compatibility boundary:** record whether metrics are computed in normalized or physical units. The smoke-gate summary must state this; longer Phase 2 execution can revise only by plan update.
- **Focused tests:** hand-computed scalar RMSE/nRMSE, per-channel aggregation, epsilon behavior for zero target, normalization uses train split only, JSON serialization.

### Unit 4: Supervised SWE Model Adapters

- **Owns:** minimal supervised model factories for tiny smoke passes:
  - `hybrid_resnet`: a PDE head using existing Hybrid ResNet encoder/downsample/ResNet/CycleGAN building blocks where practical, outputting SWE state channels directly.
  - `fno`: a tiny FNO baseline using `neuralop.models.FNO` when available.
  - `unet`: a compact local 2D U-Net baseline.
- **Proposed files:**
  - Create: `scripts/studies/pdebench_swe/models.py`
  - Test: `tests/studies/test_pdebench_swe_models.py`
- **Stable interfaces:**
  - `build_model(model_name: str, in_channels: int, out_channels: int, spatial_shape: tuple[int, int], smoke_config: Mapping) -> torch.nn.Module`
  - Model names: `hybrid_resnet`, `fno`, `unet`
  - Forward contract: `model(x: FloatTensor[B,C,H,W]) -> FloatTensor[B,C,H,W]`
- **Must not own:** HDF5 IO, split policy, metric formulas, run orchestration, or CDI generator/Lightning behavior.
- **Dependency direction:** consumes channel/spatial metadata from Units 1-2 and feeds Unit 5.
- **Compatibility boundary:** do not change `ptycho_torch/generators/hybrid_resnet.py`, `ptycho_torch/generators/fno.py`, or generator registry behavior for this smoke gate. If a reusable PDE operator-learning module is desired later, that is a separate Phase 2 architecture plan.
- **Focused tests:** each model builds on CPU with tiny dimensions, preserves `(B,C,H,W)`, supports one backward pass, records parameter count, and fails with a clear blocker if `neuralop.models.FNO` is unavailable.

### Unit 5: Smoke Runner, CLI, and Provenance

- **Owns:** bounded one-step train/eval loop, per-model metrics/blocker files, runtime and peak GPU memory capture where available, command-line interface, invocation artifacts, package/runtime provenance, git commit or dirty-state note, duplicate-output-root guard, and exact PID-friendly execution behavior.
- **Proposed files:**
  - Create: `scripts/studies/pdebench_swe/smoke.py`
  - Create: `scripts/studies/run_pdebench_swe_smoke.py`
  - Test: `tests/studies/test_pdebench_swe_smoke_cli.py`
- **Stable interfaces/artifacts:**
  - `runs/<model>/metrics.json`
  - `runs/<model>/blocker.json`
  - `runs/<model>/provenance.json`
  - root `invocation.json` and `invocation.sh`
  - CLI arguments: `--output-root`, `--models hybrid_resnet,fno,unet`, `--epochs`, `--batch-size`, `--learning-rate`, `--device`, `--num-workers`, `--max-train-batches`, `--max-eval-batches`, `--run-id`, `--allow-existing-output-root`
- **Must not own:** final roadmap decision prose, docs index entries, or external fallback execution.
- **Dependency direction:** consumes Units 1-4 and writes raw evidence for Unit 6.
- **Compatibility boundary:** default behavior must refuse to write into an existing non-empty output root unless `--allow-existing-output-root` is passed. Long commands must be tmux-launched by the executor, not hidden behind broad process polling.
- **Focused tests:** CLI parser, invocation artifacts, duplicate-root rejection, run ID propagation, synthetic HDF5 smoke run on CPU for at least U-Net, per-model blocker path, provenance fields, metrics JSON schema, and stale-artifact/freshness validation with a reused output root.

### Unit 6: Durable Smoke-Gate Summary and Discoverability

- **Owns:** tracked `pdebench_swe_smoke_gate.md`, exact final decision, concise links to ignored support artifacts, and documentation index updates.
- **Proposed files:**
  - Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
  - Modify: `docs/studies/index.md`
  - Modify: `docs/index.md`
- **Required summary sections:** scope, dataset identity, license/access notes, HDF5 field/channel/axis metadata, split manifest, normalization/metric contract, smoke command, model results/blockers, runtime/memory/provenance, gate checks, decision, non-goals confirmed, raw artifact links, carry-forward notes.
- **Must not own:** raw logs, full Phase 2 execution summary, paper-facing evidence map, or fallback execution.
- **Dependency direction:** consumes all previous units and closes the tranche gate.
- **Focused tests:** structural summary check from tranche context plus index/discoverability checks. The final line of the decision section must contain exactly one of `proceed with longer SWE execution`, `pivot to OpenFWI FlatVel-A`, or `block for human decision`.

## Compatibility, Migration, and Boundary Notes

- No migrations are planned.
- No production dependency changes are planned. `h5py`, `torch`, and `neuralop` are expected local dependencies from prior evidence; if any are unavailable during implementation, record a blocker instead of editing dependency metadata unless the user approves.
- Do not import or vendor PDEBench code into the repo for this tranche. Use official download/data references in provenance, and implement local metrics only where the formula is documented in the summary.
- Do not alter the current ptychographic PyTorch workflows or config bridge. This SWE harness is a study-specific supervised smoke path.
- Do not modify model registry defaults, stable CDI modules, or core physics files.
- The ignored `.artifacts/` root is the machine-readable evidence surface; tracked docs should be concise and link to it.
- `/home/ollie/Documents/neurips/` remains untouched until Roadmap Phase 5.
- OpenFWI FlatVel-A remains a fallback recommendation only in this tranche. Do not download or run it here.

## Context Priming Before Edits

Re-read these before executing implementation tasks:

- `docs/index.md`
- `docs/findings.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-pdebench-swe-primary-smoke-gate/tranche-context.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/DATA_MANAGEMENT_GUIDE.md`
- `docs/development/INVOCATION_LOGGING_GUIDE.md`
- `docs/TESTING_GUIDE.md`

Required findings and policies to carry into implementation:

- `POLICY-001`: PyTorch is mandatory.
- `ANTIPATTERN-001`: no import-time data loading or hidden side effects.
- `PYTHON-ENV-001`: use PATH `python`.
- `FNO-DEPTH-001` and `FNO-DEPTH-002`: keep spectral smoke models tiny and memory bounded on the RTX 3090.
- `STABLE-CRASH-DEPTH-001`: do not overinterpret smoke convergence from one tiny seed; this tranche proves executability and data contracts, not final competitiveness.
- `FORWARD-SIG-001`: FNO/Hybrid-like models should use single-input `model(x)` semantics.
- `OUTPUT-COMPLEX-001`: relevant to CDI FNO/Hybrid outputs only; the SWE supervised adapter must document that it outputs real SWE state channels directly.

## Phases

### Phase A: Scope, Workspace, and Data Preflight

**Files:**
- Read: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Read: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-primary-smoke-gate/plan-phase/plan_path.txt`
- Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/`
- Create during execution: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/dataset_manifest.json`

- [ ] A1: Verify the current checkout and selected plan pointer.

Run:

```bash
pwd
git status --short
sed -n '1p' state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-primary-smoke-gate/plan-phase/plan_path.txt
```

Expected: working directory is `/home/ollie/Documents/PtychoPINN`; unrelated dirty files are noted and not reverted; pointer prints `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-primary-smoke-gate/execution_plan.md`.

- [ ] A2: Confirm Phase 0 and Phase 1 are complete and Phase 2 is the next unsatisfied roadmap gate.

Run:

```bash
python - <<'PY'
from pathlib import Path
import json

ledger = json.loads(Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json").read_text(encoding="utf-8"))
completed = set(ledger.get("completed_tranches", []))
assert "phase-0-evidence-inventory" in completed, "Phase 0 is not complete"
assert "phase-1-pde-benchmark-selection" in completed, "Phase 1 is not complete"
assert "phase-2-pdebench-swe-primary-smoke-gate" not in completed, "This tranche is already marked complete"
assert Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md").exists()
print("prior gates complete; SWE smoke gate is pending")
PY
```

Expected: prints `prior gates complete; SWE smoke gate is pending`.

- [ ] A3: Create and verify the ignored support artifact root.

Run:

```bash
RAW_ROOT=".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate"
mkdir -p "${RAW_ROOT}"
git check-ignore -v "${RAW_ROOT}/probe.json"
```

Expected: `git check-ignore` reports an ignore rule for `.artifacts/`.

- [ ] A4: Locate or stage the official PDEBench SWE HDF5 outside git.

Run:

```bash
python - <<'PY'
from pathlib import Path
import os

candidate = os.environ.get("PDEBENCH_SWE_H5")
if candidate:
    path = Path(candidate).expanduser()
else:
    data_root = Path(os.environ.get("PDE_DATA", "/home/ollie/Documents/pdebench-data")).expanduser()
    matches = list(data_root.rglob("2D_rdb_NA_NA.h5")) if data_root.exists() else []
    path = matches[0] if matches else None
if not path or not path.exists():
    raise SystemExit(
        "missing PDEBench SWE 2D_rdb_NA_NA.h5; set PDEBENCH_SWE_H5 or stage it with "
        "`python download_direct.py --root_folder $PDE_DATA --pde_name swe` from the PDEBench data downloader"
    )
print(path)
PY
```

Expected: prints the absolute HDF5 path. If missing, stage the file outside git using the official PDEBench downloader command or DaRUS datafile `133021`, then rerun this check.

- [ ] A5: Record a minimal dataset manifest before any training code runs.

Run after Unit 1 exists:

```bash
python scripts/studies/run_pdebench_swe_smoke.py \
  --data-file "${PDEBENCH_SWE_H5:?set absolute path to 2D_rdb_NA_NA.h5}" \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate \
  --dataset-source PDEBench \
  --dataset-source-url "https://github.com/pdebench/PDEBench" \
  --dataset-darus-id 133021 \
  --license-note "Record PDEBench repository/data license and DaRUS access terms before longer execution." \
  --inspect-only
```

Expected: writes `dataset_manifest.json` and `hdf5_metadata.json`, then exits `0` without model training.

### Phase B: HDF5 Manifest and Split/Data Contract

**Files:**
- Create: `scripts/studies/pdebench_swe/__init__.py`
- Create: `scripts/studies/pdebench_swe/manifest.py`
- Create: `scripts/studies/pdebench_swe/splits.py`
- Create: `scripts/studies/pdebench_swe/data.py`
- Test: `tests/studies/test_pdebench_swe_manifest.py`
- Test: `tests/studies/test_pdebench_swe_splits_data.py`

- [ ] B1: Write failing tests for HDF5 manifest extraction.

Run:

```bash
pytest tests/studies/test_pdebench_swe_manifest.py -v
```

Expected before implementation: fails because `scripts.studies.pdebench_swe.manifest` does not exist.

- [ ] B2: Implement `manifest.py` with recursive dataset inspection, file identity, checksum/size/mtime capture, source/license fields, state-dataset selection, and ambiguous-layout blockers.

Required functions:

```python
def file_identity(path: Path, *, sha256: bool = True) -> dict: ...
def inspect_hdf5(path: Path) -> dict: ...
def select_state_dataset(metadata: dict, requested: str | None = None) -> dict: ...
def write_dataset_manifests(..., output_root: Path) -> tuple[Path, Path]: ...
```

- [ ] B3: Run manifest tests to green.

Run:

```bash
pytest tests/studies/test_pdebench_swe_manifest.py -v
```

Expected: all manifest tests pass.

- [ ] B4: Write failing tests for deterministic split and one-step dataset behavior.

Run:

```bash
pytest tests/studies/test_pdebench_swe_splits_data.py -v
```

Expected before implementation: fails because split/data modules do not exist or lack required behavior.

- [ ] B5: Implement `splits.py` and `data.py`.

Required behavior:

- `build_trajectory_split(num_trajectories, seed=20260420, ratios=(0.8, 0.1, 0.1))` returns disjoint train/val/test trajectory IDs.
- `write_split_manifest(...)` records source file identity, HDF5 state dataset, axis order, seed, ratios, IDs, and one-step pair counts.
- `SweOneStepDataset` lazily opens HDF5 per process, maps one-step pairs, normalizes tensor axes to `(C,H,W)`, applies shared normalization, and records pad/crop metadata when padding is needed.
- Dataset loading must never read the full HDF5 into memory for smoke mode.

- [ ] B6: Run split/data tests to green.

Run:

```bash
pytest tests/studies/test_pdebench_swe_splits_data.py -v
```

Expected: all split/data tests pass with synthetic HDF5 fixtures.

### Phase C: Normalization and Metric Contract

**Files:**
- Create: `scripts/studies/pdebench_swe/metrics.py`
- Test: `tests/studies/test_pdebench_swe_metrics.py`

- [ ] C1: Write failing tests for train-only normalization and `err_RMSE`/`err_nRMSE`.

Run:

```bash
pytest tests/studies/test_pdebench_swe_metrics.py -v
```

Expected before implementation: fails because `metrics.py` does not exist or does not implement required formulas.

- [ ] C2: Implement local metric and normalization helpers.

Required functions:

```python
def compute_channel_stats(dataset, *, max_batches: int | None = None) -> dict: ...
def normalize_batch(batch: torch.Tensor, stats: dict) -> torch.Tensor: ...
def denormalize_batch(batch: torch.Tensor, stats: dict) -> torch.Tensor: ...
def err_rmse(prediction: torch.Tensor, target: torch.Tensor, *, dims=None) -> torch.Tensor: ...
def err_nrmse(prediction: torch.Tensor, target: torch.Tensor, *, dims=None, eps: float = 1e-12) -> torch.Tensor: ...
def metric_payload(prediction_batches, target_batches, *, normalized: bool, stats: dict | None) -> dict: ...
```

Metric notes:

- Compute primary `err_nRMSE` as relative L2 RMSE over the documented one-step evaluation axes.
- Emit per-channel and aggregate metrics.
- Record whether metrics are computed after denormalization or in normalized units. Prefer denormalized physical units when practical; if not, label the smoke metrics as normalized-unit smoke metrics.

- [ ] C3: Run metric tests to green.

Run:

```bash
pytest tests/studies/test_pdebench_swe_metrics.py -v
```

Expected: hand-calculated metric tests and JSON-serialization tests pass.

### Phase D: Supervised Model Adapters

**Files:**
- Create: `scripts/studies/pdebench_swe/models.py`
- Test: `tests/studies/test_pdebench_swe_models.py`

- [ ] D1: Write failing tests for model factory contracts.

Run:

```bash
pytest tests/studies/test_pdebench_swe_models.py -v
```

Expected before implementation: fails because `models.py` does not exist or lacks model factories.

- [ ] D2: Implement tiny supervised adapters for `hybrid_resnet`, `fno`, and `unet`.

Requirements:

- `build_model("hybrid_resnet", ...)` returns a real-channel PDE model with `model(x)` forward contract and output shape `(B,C,H,W)`.
- Reuse existing Hybrid ResNet building blocks where practical, but do not modify existing CDI generator modules.
- Use tiny defaults for smoke: `hidden_channels=8`, `fno_modes=4`, `fno_blocks=3`, `hybrid_downsample_steps=1`, `hybrid_resnet_blocks=1`, and pad/crop around the model if spatial dimensions need divisibility.
- `build_model("fno", ...)` uses `neuralop.models.FNO` if available. If unavailable, raise a typed blocker that the CLI writes as `runs/fno/blocker.json`.
- `build_model("unet", ...)` returns a compact local 2D U-Net baseline with direct real-channel output.
- All models must report parameter count and smoke config.

- [ ] D3: Run model tests to green.

Run:

```bash
pytest tests/studies/test_pdebench_swe_models.py -v
```

Expected: CPU forward/backward tests pass for all available models; unavailable FNO path produces a clear blocker object, not an uncaught import error.

### Phase E: Smoke Runner CLI and Provenance

**Files:**
- Create: `scripts/studies/pdebench_swe/smoke.py`
- Create: `scripts/studies/run_pdebench_swe_smoke.py`
- Test: `tests/studies/test_pdebench_swe_smoke_cli.py`

- [ ] E1: Write failing tests for CLI, invocation artifacts, duplicate-root guard, synthetic smoke execution, and per-model metrics/blockers.

Run:

```bash
pytest tests/studies/test_pdebench_swe_smoke_cli.py -v
```

Expected before implementation: fails because CLI and smoke runner do not exist.

- [ ] E2: Implement the smoke runner.

Required behavior:

- Parse the CLI listed in this plan.
- Write root `invocation.json` and `invocation.sh` using `scripts.studies.invocation_logging.write_invocation_artifacts`.
- Capture runtime provenance: git commit, dirty-state summary, Python executable/version, PyTorch/CUDA versions, package versions for `h5py`, `torch`, `neuralop`, GPU name, CUDA memory summary where available, command, PID, cwd, output root, and data manifest paths.
- Refuse non-empty `--output-root` unless `--allow-existing-output-root` is passed.
- Accept an optional `--run-id`; default to a unique UTC timestamp when omitted.
- Write `run_id` and the smoke process PID into `invocation.json`, root contract JSONs (`dataset_manifest.json`, `hdf5_metadata.json`, `split_manifest.json`, `normalization_stats.json`), and each per-model `metrics.json`, `blocker.json`, and `provenance.json` payload.
- When `--allow-existing-output-root` is used, overwrite the root contract JSONs for the current run; never treat pre-existing contract artifacts as evidence unless they are rewritten with the current `run_id`.
- Build or re-read source inputs as needed, then write current-run `dataset_manifest.json`, `hdf5_metadata.json`, `split_manifest.json`, and `normalization_stats.json`.
- Run each requested model independently so one blocker does not erase the other models' evidence.
- Write `runs/<model>/metrics.json` on success or `runs/<model>/blocker.json` on controlled failure.
- Exit nonzero only for pre-training contract failures that invalidate the whole gate, such as missing HDF5 file, ambiguous state dataset without override, invalid split, or no runnable model evidence.

- [ ] E3: Run CLI tests to green.

Run:

```bash
pytest tests/studies/test_pdebench_swe_smoke_cli.py -v
```

Expected: CLI tests pass with synthetic HDF5 fixtures and CPU smoke limits.

- [ ] E4: Run all focused tests for the new SWE harness.

Run:

```bash
pytest \
  tests/studies/test_pdebench_swe_manifest.py \
  tests/studies/test_pdebench_swe_splits_data.py \
  tests/studies/test_pdebench_swe_metrics.py \
  tests/studies/test_pdebench_swe_models.py \
  tests/studies/test_pdebench_swe_smoke_cli.py \
  -v
```

Expected: all focused tests pass.

### Phase F: Official SWE Smoke Execution

**Files/artifacts:**
- Write: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/dataset_manifest.json`
- Write: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/hdf5_metadata.json`
- Write: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/split_manifest.json`
- Write: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/normalization_stats.json`
- Write: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/runs/*/{metrics,blocker,provenance}.json`
- Write: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/logs/smoke.{run_id,started_at_ns,pid,exit_code}`

- [ ] F1: Re-run inspect-only mode against the official file and review `hdf5_metadata.json` before training.

Run:

```bash
python scripts/studies/run_pdebench_swe_smoke.py \
  --data-file "${PDEBENCH_SWE_H5:?set absolute path to 2D_rdb_NA_NA.h5}" \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate \
  --dataset-source PDEBench \
  --dataset-source-url "https://github.com/pdebench/PDEBench" \
  --dataset-darus-id 133021 \
  --license-note "Record PDEBench repository/data license and DaRUS access terms before longer execution." \
  --inspect-only \
  --allow-existing-output-root
```

Expected: manifest files exist and record `2D_rdb_NA_NA.h5`, source/access fields, tensor axes, dtype, selected state dataset, and dimensions. If state dataset or axis order is ambiguous, rerun with `--state-dataset` and/or `--axis-order` after recording the HDF5 metadata.

- [ ] F2: Launch the bounded official smoke in tmux with exact PID tracking.

Run:

```bash
tmux new-session -d -s swe_smoke_gate "bash -lc '
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
OUT=.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate
export OUT
mkdir -p \"\$OUT/logs\"
RUN_ID=\$(python - <<PY
from datetime import datetime, timezone
print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ"))
PY
)
export RUN_ID
python - <<PY
from pathlib import Path
import os
import time

out = Path(os.environ["OUT"])
(out / "logs/smoke.run_id").write_text(os.environ["RUN_ID"] + "\n", encoding="utf-8")
(out / "logs/smoke.started_at_ns").write_text(str(time.time_ns()) + "\n", encoding="utf-8")
PY
python scripts/studies/run_pdebench_swe_smoke.py \
  --data-file \"${PDEBENCH_SWE_H5:?set absolute path to 2D_rdb_NA_NA.h5}\" \
  --output-root \"\$OUT\" \
  --dataset-source PDEBench \
  --dataset-source-url \"https://github.com/pdebench/PDEBench\" \
  --dataset-darus-id 133021 \
  --license-note \"Record PDEBench repository/data license and DaRUS access terms before longer execution.\" \
  --split-seed 20260420 \
  --train-fraction 0.8 \
  --val-fraction 0.1 \
  --test-fraction 0.1 \
  --models hybrid_resnet,fno,unet \
  --epochs 1 \
  --batch-size 2 \
  --learning-rate 1e-3 \
  --max-train-trajectories 4 \
  --max-val-trajectories 2 \
  --max-test-trajectories 2 \
  --max-pairs-per-trajectory 1 \
  --max-train-batches 2 \
  --max-eval-batches 2 \
  --device cuda \
  --num-workers 0 \
  --run-id \"\$RUN_ID\" \
  --allow-existing-output-root \
  > \"\$OUT/logs/smoke_stdout.log\" 2> \"\$OUT/logs/smoke_stderr.log\" &
pid=\$!
echo \"\$pid\" > \"\$OUT/logs/smoke.pid\"
wait \"\$pid\"
code=\$?
echo \"\$code\" > \"\$OUT/logs/smoke.exit_code\"
exit \"\$code\"
'"
```

Expected: tmux session starts once. Do not launch another run to the same output root while this PID is active.

- [ ] F3: Wait on and verify the tracked smoke run.

Run:

```bash
tmux has-session -t swe_smoke_gate 2>/dev/null && tmux capture-pane -pt swe_smoke_gate -S -120 || true
python - <<'PY'
from pathlib import Path
import json

root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate")
run_id_path = root / "logs/smoke.run_id"
start_path = root / "logs/smoke.started_at_ns"
pid_path = root / "logs/smoke.pid"
exit_path = root / "logs/smoke.exit_code"
for path in [run_id_path, start_path, pid_path, exit_path]:
    if not path.exists():
        raise SystemExit(f"smoke run has not written {path.name} yet; inspect tmux session before proceeding")
if not exit_path.exists():
    raise SystemExit("smoke run has not written exit code yet; inspect tmux session before proceeding")
run_id = run_id_path.read_text(encoding="utf-8").strip()
start_ns = int(start_path.read_text(encoding="utf-8").strip())
tracked_pid = pid_path.read_text(encoding="utf-8").strip()
code = exit_path.read_text(encoding="utf-8").strip()
if code != "0":
    raise SystemExit(f"smoke command exited {code}; inspect logs under {root / 'logs'}")

def require_fresh(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"missing smoke contract artifact: {path}")
    if path.stat().st_mtime_ns < start_ns:
        raise SystemExit(f"stale smoke artifact predates tracked run start: {path}")

def load_fresh_json(path: Path) -> dict:
    require_fresh(path)
    return json.loads(path.read_text(encoding="utf-8"))

def extract_run_id(payload: dict) -> str | None:
    candidates = [
        payload.get("run_id"),
        payload.get("run", {}).get("run_id") if isinstance(payload.get("run"), dict) else None,
        payload.get("provenance", {}).get("run_id") if isinstance(payload.get("provenance"), dict) else None,
    ]
    return next((str(value) for value in candidates if value is not None), None)

root_artifacts = [
    root / "dataset_manifest.json",
    root / "hdf5_metadata.json",
    root / "split_manifest.json",
    root / "normalization_stats.json",
]
for path in root_artifacts:
    payload = load_fresh_json(path)
    if extract_run_id(payload) != run_id:
        raise SystemExit(f"{path} does not record current run_id {run_id}")
for model in ["hybrid_resnet", "fno", "unet"]:
    model_root = root / "runs" / model
    provenance = load_fresh_json(model_root / "provenance.json")
    if extract_run_id(provenance) != run_id:
        raise SystemExit(f"{model} provenance does not record current run_id {run_id}")
    pid_candidates = {
        provenance.get("pid"),
        provenance.get("process_pid"),
        provenance.get("smoke_pid"),
        provenance.get("provenance", {}).get("pid") if isinstance(provenance.get("provenance"), dict) else None,
    }
    if tracked_pid not in {str(value) for value in pid_candidates if value is not None}:
        raise SystemExit(f"{model} provenance does not match tracked PID {tracked_pid}")
    result_paths = [model_root / "metrics.json", model_root / "blocker.json"]
    written = [path for path in result_paths if path.exists()]
    if not written:
        raise SystemExit(f"{model} wrote neither metrics.json nor blocker.json")
    for path in written:
        payload = load_fresh_json(path)
        if extract_run_id(payload) != run_id:
            raise SystemExit(f"{path} does not record current run_id {run_id}")
print("official SWE smoke artifacts are fresh for the tracked PID/run_id")
PY
```

Expected: exit code `0`; required manifests are freshly written after `logs/smoke.started_at_ns`; each model has fresh provenance plus metrics or an explicit blocker; all checked JSON payloads record the current run ID, and per-model provenance matches the tracked PID.

### Phase G: Durable Smoke-Gate Summary and Documentation

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
- Modify: `docs/studies/index.md`
- Modify: `docs/index.md`

- [ ] G1: Write the durable smoke-gate summary.

Required decision logic:

- Use `proceed with longer SWE execution` only if data identity, HDF5 metadata, split manifest, metric contract, bounded data load, Hybrid ResNet-compatible smoke, and at least one baseline smoke succeeded, with the second baseline either succeeded or explicitly blocked for a narrow recoverable reason.
- Use `pivot to OpenFWI FlatVel-A` if official SWE data access, split, metric, or quick local baseline feasibility fails in a way that makes longer SWE execution unwise.
- Use `block for human decision` if both the primary path and fallback readiness require a storage/compute/scope decision. Do not use smoke metrics to decide that SWE is scientifically too weak or that Hybrid ResNet is noncompetitive; those decisions require a later longer/pilot benchmark-performance run.

- [ ] G2: Update `docs/studies/index.md` with a concise entry for `scripts/studies/run_pdebench_swe_smoke.py`.

The entry must name the official SWE file, one-step smoke scope, output artifact root, and the fact that this is a Phase 2 smoke gate rather than full PDE training.

- [ ] G3: Update `docs/index.md` with a concise entry for `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`.

The entry must make the smoke-gate summary discoverable for later Roadmap Phase 2 execution and fallback decisions.

### Phase H: Final Verification and Output Contract

**Files:**
- Read: `state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-primary-smoke-gate/plan-phase/plan_path.txt`
- Verify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-primary-smoke-gate/execution_plan.md`
- Verify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`

- [ ] H1: Run all focused pytest selectors.

Run:

```bash
pytest \
  tests/studies/test_pdebench_swe_manifest.py \
  tests/studies/test_pdebench_swe_splits_data.py \
  tests/studies/test_pdebench_swe_metrics.py \
  tests/studies/test_pdebench_swe_models.py \
  tests/studies/test_pdebench_swe_smoke_cli.py \
  -v
```

Expected: all focused tests pass. Archive logs under the ignored support root if the workflow requires retained test evidence.

- [ ] H2: Run the durable summary structural check from the tranche context.

Run:

```bash
python - <<'PY'
from pathlib import Path
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md")
text = path.read_text() if path.exists() else ""
required_terms = [
    "2D_rdb_NA_NA.h5",
    "20260420",
    "err_nRMSE",
    "Hybrid ResNet",
    "FNO",
    "U-Net",
    "proceed",
    "pivot",
    "block",
]
missing = [term for term in required_terms if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"smoke-gate summary missing expected terms: {missing}")
decision_terms = [
    "proceed with longer SWE execution",
    "pivot to OpenFWI FlatVel-A",
    "block for human decision",
]
hits = [term for term in decision_terms if term in text]
if len(hits) != 1:
    raise SystemExit(f"expected exactly one gate decision, found {hits}")
print("SWE smoke-gate summary contains required decision fields")
PY
```

Expected: prints `SWE smoke-gate summary contains required decision fields`.

- [ ] H3: Verify discoverability updates and artifact hygiene.

Run:

```bash
python - <<'PY'
from pathlib import Path

index = Path("docs/index.md").read_text(encoding="utf-8")
studies = Path("docs/studies/index.md").read_text(encoding="utf-8")
assert "pdebench_swe_smoke_gate.md" in index, "docs/index.md missing SWE smoke-gate summary"
assert "run_pdebench_swe_smoke.py" in studies, "docs/studies/index.md missing SWE smoke runbook"
print("discoverability and paper-facing boundary are valid")
PY
python - <<'PY'
import re
import subprocess

paths = [
    "scripts/studies/pdebench_swe",
    "scripts/studies/run_pdebench_swe_smoke.py",
    "tests/studies/test_pdebench_swe_manifest.py",
    "tests/studies/test_pdebench_swe_splits_data.py",
    "tests/studies/test_pdebench_swe_metrics.py",
    "tests/studies/test_pdebench_swe_models.py",
    "tests/studies/test_pdebench_swe_smoke_cli.py",
    "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md",
    "docs/studies/index.md",
    "docs/index.md",
]
status = subprocess.run(["git", "status", "--short", "--", *paths], check=True, text=True, capture_output=True).stdout
bad = [line for line in status.splitlines() if re.search(r"\.(h5|hdf5|npz|pt|pth|ckpt)(\s|$)", line)]
if bad:
    raise SystemExit("large/data artifact appears in planned tracked paths:\\n" + "\\n".join(bad))
print("planned tracked paths contain no data/checkpoint artifacts")
PY
```

Expected: discoverability check passes and no HDF5/checkpoint/data artifacts appear as tracked or untracked commit candidates.

- [ ] H4: Verify the output contract plan pointer is preserved.

Run:

```bash
python - <<'PY'
from pathlib import Path
pointer = Path("state/NEURIPS-HYBRID-RESNET-2026/tranche-drain/items/phase-2-pdebench-swe-primary-smoke-gate/plan-phase/plan_path.txt")
text = pointer.read_text(encoding="utf-8")
expected = "docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-primary-smoke-gate/execution_plan.md"
if text != expected + "\n" and text != expected:
    raise SystemExit(f"unexpected plan_path contents: {text!r}")
if not Path(expected).exists():
    raise SystemExit(f"plan target does not exist: {expected}")
print("plan_path contract preserved")
PY
```

Expected: prints `plan_path contract preserved`.

## Verification Commands

Use these as the targeted completion gate for this plan:

```bash
pytest \
  tests/studies/test_pdebench_swe_manifest.py \
  tests/studies/test_pdebench_swe_splits_data.py \
  tests/studies/test_pdebench_swe_metrics.py \
  tests/studies/test_pdebench_swe_models.py \
  tests/studies/test_pdebench_swe_smoke_cli.py \
  -v

python - <<'PY'
from pathlib import Path
path = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md")
text = path.read_text() if path.exists() else ""
required_terms = ["2D_rdb_NA_NA.h5", "20260420", "err_nRMSE", "Hybrid ResNet", "FNO", "U-Net", "proceed", "pivot", "block"]
missing = [term for term in required_terms if term.lower() not in text.lower()]
if missing:
    raise SystemExit(f"smoke-gate summary missing expected terms: {missing}")
decision_terms = ["proceed with longer SWE execution", "pivot to OpenFWI FlatVel-A", "block for human decision"]
hits = [term for term in decision_terms if term in text]
if len(hits) != 1:
    raise SystemExit(f"expected exactly one gate decision, found {hits}")
print("SWE smoke-gate summary contains required decision fields")
PY
```

The official HDF5 smoke command in Phase F is also required unless the implementation records a pre-training data-access blocker in `pdebench_swe_smoke_gate.md`.
When the official smoke command runs, use the Phase F3 freshness verifier as the evidence gate; an existence-only check is not sufficient for completion.

## Completion Criteria

- [ ] `2D_rdb_NA_NA.h5` is located or the summary records a precise data-access blocker.
- [ ] Dataset source, DaRUS ID or URL, license/access notes, local path, size/mtime/checksum, and disk feasibility are recorded.
- [ ] HDF5 field names, selected state dataset, channel order, tensor axes, trajectory/sample/time dimensions, dtype, and normalization metadata are recorded before training.
- [ ] Deterministic trajectory-level 80/10/10 split manifest exists with seed `20260420`.
- [ ] One-step `err_nRMSE` and `err_RMSE` are implemented locally or an upstream import is license-cleared and documented.
- [ ] Bounded data-load smoke succeeds without exceeding local disk/GPU constraints, or the precise blocker is recorded.
- [ ] Tiny Hybrid ResNet-compatible one-step train/eval smoke writes metrics/provenance, or the precise blocker is recorded.
- [ ] Tiny FNO and U-Net smoke baselines write metrics/provenance under the same contract, or blockers are recorded for any unavailable baseline.
- [ ] Official smoke artifacts are fresh relative to `logs/smoke.started_at_ns`, record the current `logs/smoke.run_id`, and per-model provenance matches the tracked PID in `logs/smoke.pid`; stale artifacts from earlier runs cannot satisfy the gate.
- [ ] `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md` ends with exactly one allowed gate decision.
- [ ] Focused pytest selectors pass for any new adapter, loader, split, metric, result-writer, or CLI code.
- [ ] `docs/studies/index.md` and `docs/index.md` are updated for durable discoverability.
- [ ] No CDI regeneration, `256x256` scaling, paper-facing `/home/ollie/Documents/neurips/` artifact, worktree, or stable core-module edit occurs.

## Artifacts Index

- Tranche plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-swe-primary-smoke-gate/execution_plan.md`
- Durable smoke-gate summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_swe_smoke_gate.md`
- Ignored support root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/`
- Dataset manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/dataset_manifest.json`
- HDF5 metadata: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/hdf5_metadata.json`
- Split manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/split_manifest.json`
- Normalization stats: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/normalization_stats.json`
- Model run artifacts: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/runs/`
- Logs: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-swe-primary-smoke-gate/logs/`
