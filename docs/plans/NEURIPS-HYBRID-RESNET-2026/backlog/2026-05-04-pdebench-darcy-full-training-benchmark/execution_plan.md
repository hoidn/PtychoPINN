# Darcy Full-Training Benchmark Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the authorized PDEBench Darcy beta `1.0` full-training benchmark on the fixed `8000 / 1000 / 1000` split for `hybrid_resnet_base`, `fno_base`, and `unet_strong`, then publish durable same-contract Phase 2 evidence.

**Architecture:** Reuse the existing Darcy adapter, split, normalization, metrics, and reporting path in `scripts/studies/pdebench_image128/`. First audit and align the benchmark contract surfaces, then run the three required profiles under benchmark mode with tmux plus `ptycho311` using a fresh run root and tracked PID ownership, and finally update the durable Darcy summary, selector ledger, and NeurIPS evidence indexes from the produced artifacts.

**Tech Stack:** PATH `python`, PyTorch, h5py, NumPy, tmux, existing `scripts/studies/run_pdebench_image128_suite.py` Darcy runner, `.artifacts/NEURIPS-HYBRID-RESNET-2026/` evidence roots.

---

## Selected Objective

- Close the active Darcy evidence gap by producing full-training benchmark rows for `hybrid_resnet_base`, `fno_base`, and `unet_strong` on the official PDEBench Darcy beta `1.0` file under the fixed full-split contract.

## Scope

- Reuse the existing Darcy static-operator implementation and benchmark runner rather than reopening adapter design.
- Launch the three required benchmark profiles on the fixed `8000 / 1000 / 1000` split with benchmark-mode provenance and same-contract reporting.
- Collate same-contract comparison outputs and literature-context caveats for the completed rows.
- Update the durable Darcy summary plus the required NeurIPS evidence and selector state surfaces.

## Explicit Non-Goals

- No CNS, CDI, candidate-lane, Phase 4, or Phase 5 work.
- No roadmap or steering rewrite unless an execution-time contract contradiction requires a narrowly scoped correction.
- No relabeling of readiness or capped pilot outputs as benchmark evidence.
- No new adapter or model-development work unless a narrow benchmark blocker is discovered and fixed in place.
- No manuscript prose or `/home/ollie/Documents/neurips/` artifact work.

## Binding Constraints And Source Of Truth

- Steering is binding: this item must strengthen required PDEBench evidence, keep equal-footing comparisons explicit, and avoid spending budget on optional follow-ups.
- The roadmap is binding: Darcy full-training is an allowed Phase 2 next step; smoke results never count as benchmark-performance evidence; meaningful benchmark rows must use the full available training split after holdout.
- The design is binding: use the supervised real-channel Hybrid ResNet adapter, preserve explicit protocol caveats for literature context, and do not broaden claims beyond the scoped PDEBench lane.
- Progress-ledger status matters: Darcy readiness is already implemented and benchmark-incomplete; this plan should finish the full-training benchmark without waiting on unrelated CNS work.
- Evidence-index maintenance is binding: result-producing execution must update `evidence_matrix.md`, `model_variant_index.json`, and any other applicable durable index surface before closeout.
- Long-running command ownership is binding: benchmark launches remain under implementation ownership until the tracked process exits successfully or an unrecoverable, documented blocker remains after a narrow fix attempt.
- Consistency ruling for this backlog item:
  - `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`
  are the authoritative Darcy benchmark contract.
- The checked-in `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json` is currently stale because it records `loss: "mae"` and `plateau_min_lr: 1e-4`. Implementation must reconcile that file to the authoritative contract before any benchmark launch. The benchmark contract for this item is relative L2 loss with `plateau_min_lr <= 1e-5` unless a new explicit override is written before training.

## Prerequisite Status

- Completed already:
  - Darcy static-operator loader, splits, normalization, metrics, model profiles, runner, and readiness path.
  - Durable readiness summary at `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`.
  - Existing benchmark budget path at `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json`.
- Still missing:
  - Full-training benchmark rows for `hybrid_resnet_base`, `fno_base`, and `unet_strong`.
  - Durable same-contract benchmark summary content and evidence-index entries for those rows.
  - Selector-facing ledger update that records either benchmark completion or an exact remaining blocker.

## Implementation Architecture

- Contract audit and launch preparation:
  - Validate the existing runner inputs, tests, data file, and benchmark budget, and correct any stale budget/config mismatch before training.
- Benchmark execution:
  - Run the three required profiles under one fixed Darcy contract, using a fresh timestamped run root inside the existing Darcy phase artifact root and one active writer at a time.
- Durable closeout:
  - Promote the completed run root into the Darcy summary, evidence indexes, and progress ledger without reopening unrelated roadmap phases.

## File And Artifact Targets

Mandatory contract inputs to read and honor:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json`

Mandatory repo surfaces expected to change:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-pdebench-darcy-full-training-benchmark/execution_report.md`
- `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-pdebench-darcy-full-training-benchmark-checks.json`
- `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-pdebench-darcy-full-training-benchmark-summary.json`

Mandatory generated run outputs:

- A fresh benchmark run root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/`
- `invocation.json` and `invocation.sh`
- `dataset_manifest.json`
- `hdf5_metadata.json`
- `split_manifest.json`
- `normalization_stats_input.json`
- `normalization_stats_target.json`
- `model_profile_hybrid_resnet_base.json`
- `model_profile_fno_base.json`
- `model_profile_unet_strong.json`
- `metrics_hybrid_resnet_base.json`
- `metrics_fno_base.json`
- `metrics_unet_strong.json`
- `comparison_summary.json`
- `comparison_summary.csv`
- `literature_context.json`

Preferred packaging that should be kept when available but must not decide success by itself:

- `comparison_<profile>_sample0.png`
- `comparison_<profile>_sample0.npz`

Conditional source-edit surfaces only if a narrow bug blocks honest execution:

- `scripts/studies/pdebench_image128/darcy.py`
- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/reporting.py`
- `tests/studies/test_pdebench_darcy_data.py`
- `tests/studies/test_pdebench_darcy_metrics.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

## Task 1: Contract Audit And Blocking Preflight

**Purpose:** confirm that the existing Darcy implementation is still launchable and that the benchmark contract is internally consistent before expensive training starts.

- [ ] Run the selected backlog item’s required deterministic input check. This is blocking.

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-pdebench-darcy-static-operator-benchmark/execution_plan.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing Darcy benchmark inputs: {missing}")
print("darcy benchmark inputs present")
PY
```

- [ ] Run the selected backlog item’s required test selector. This is blocking before long training.

```bash
pytest -q tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
```

- [ ] Audit the current benchmark contract surfaces and update `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json` so it matches the authoritative Darcy benchmark contract:
  - `task_id: "darcy"`
  - `mode: "benchmark"`
  - split counts `8000 / 1000 / 1000`
  - `primary_profiles: ["hybrid_resnet_base", "fno_base", "unet_strong"]`
  - `training_seed: 20260420`
  - `loss: "relative_l2"`
  - `loss_rationale` naming metric alignment
  - `optimizer: "adam"`
  - `learning_rate: 2e-4`
  - `scheduler: "ReduceLROnPlateau"`
  - `plateau_factor: 0.5`
  - `plateau_patience: 2`
  - `plateau_min_lr: 1e-5`
  - `plateau_threshold: 0.0`
  - current benchmark batch size / epochs / device / num_workers
- [ ] Run a blocking inspect-mode sanity check against the staged official file after the budget correction.

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task darcy \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/inspect_$(date -u +%Y%m%dT%H%M%SZ)
```

- [ ] Choose a fresh timestamped benchmark run root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/`. Do not reuse the readiness root and do not point the benchmark at a directory that already has an active writer.
- [ ] If any preflight or selector test fails because of a repo-local bug, path issue, or environment drift, diagnose, fix, and rerun within this item. Do not mark the item `BLOCKED` until a narrow fix attempt is documented and the failure still depends on missing data, unavailable GPU, external dependency outside current authority, or an authority conflict that cannot be resolved locally.

Supporting checks for this task:

```bash
python -m json.tool .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json >/dev/null
python - <<'PY'
from pathlib import Path
from scripts.studies.pdebench_image128.run_config import validate_darcy_run_budget
import json
path = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json")
validate_darcy_run_budget(json.loads(path.read_text(encoding="utf-8")))
print("darcy run budget valid")
PY
```

## Task 2: Full Benchmark Execution

**Purpose:** produce the three required full-training rows under one fixed Darcy contract without duplicating or downgrading the benchmark.

- [ ] Launch the benchmark in tmux using the `ptycho311` conda environment and a fresh output root. This is blocking work, not a fire-and-forget background task.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
python scripts/studies/run_pdebench_image128_suite.py \
  --task darcy \
  --mode benchmark \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/full_benchmark_<timestamp> \
  --run-budget .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark/run_budget.json \
  --device cuda
```

- [ ] Track the exact launched PID, wait for that PID, and only treat the run as complete when the process exits `0` and the required benchmark artifacts are freshly written. Do not use broad `pgrep -f` polling as the primary completion check.
- [ ] If the runner executes the three profiles sequentially in one invocation, preserve that contract. If a restart is required, rerun only in a way that preserves the full fixed contract and provenance.
- [ ] If a required profile fails:
  - inspect the profile-specific metrics or blocker payload
  - diagnose and apply the narrowest honest fix
  - rerun the affected benchmark attempt under the same contract
  - only leave the item benchmark-incomplete if the required profile still cannot complete after that attempt
- [ ] Treat missing optional comparison PNG/NPZ packaging as supporting-only reporting debt unless the core benchmark artifacts also failed. Do not let optional visuals alone convert a successful run into `CRASH`.
- [ ] Preserve equal footing:
  - same dataset file
  - same split manifest
  - same normalization stats policy
  - same loss and scheduler contract
  - same benchmark-mode runner
  - same evaluation metric family for all three required profiles

Blocking checks for this task:

```bash
python - <<'PY'
from pathlib import Path
root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark")
candidates = sorted([p for p in root.iterdir() if p.is_dir() and "full_benchmark_" in p.name])
if not candidates:
    raise SystemExit("missing full benchmark run root")
run_root = candidates[-1]
required = [
    run_root / "invocation.json",
    run_root / "invocation.sh",
    run_root / "dataset_manifest.json",
    run_root / "hdf5_metadata.json",
    run_root / "split_manifest.json",
    run_root / "normalization_stats_input.json",
    run_root / "normalization_stats_target.json",
    run_root / "model_profile_hybrid_resnet_base.json",
    run_root / "model_profile_fno_base.json",
    run_root / "model_profile_unet_strong.json",
    run_root / "metrics_hybrid_resnet_base.json",
    run_root / "metrics_fno_base.json",
    run_root / "metrics_unet_strong.json",
    run_root / "comparison_summary.json",
    run_root / "comparison_summary.csv",
    run_root / "literature_context.json",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing required benchmark artifacts: {missing}")
print(run_root)
PY
```

Supporting checks for this task:

```bash
python - <<'PY'
from pathlib import Path
import json
root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-darcy-static-operator-benchmark")
run_root = sorted([p for p in root.iterdir() if p.is_dir() and "full_benchmark_" in p.name])[-1]
summary = json.loads((run_root / "comparison_summary.json").read_text(encoding="utf-8"))
print(summary.get("profile_results", []))
PY
```

## Task 3: Durable Summary, Index, And Selector Closeout

**Purpose:** turn the benchmark outputs into discoverable Phase 2 evidence without leaking into later roadmap phases.

- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md` so it clearly distinguishes:
  - the earlier readiness evidence
  - the new full-training benchmark run root
  - exact profile completion status for `hybrid_resnet_base`, `fno_base`, and `unet_strong`
  - headline same-contract metrics from `comparison_summary.json`
  - literature calibration caveats
  - remaining blockers, if any
  - the resulting claim boundary for Darcy
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` with a Darcy row that points to the authoritative summary and artifact root.
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json` with the Darcy dataset contract and the completed benchmark rows. If the outcome is benchmark-incomplete, record only the rows that genuinely completed and mark the remaining required row state explicitly.
- [ ] Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` with the durable backlog outcome for this Darcy item, using the correct evidence tier for the produced result. Do not overclaim paper-grade authority if the benchmark remains incomplete.
- [ ] Update `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` with the selector-facing decision:
  - completed Darcy full benchmark, or
  - benchmark-incomplete with the exact blocker and next recommended selector action
- [ ] Write the implementation workflow outputs if they are owned by this phase:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-pdebench-darcy-full-training-benchmark/execution_report.md`
  - `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-pdebench-darcy-full-training-benchmark-checks.json`
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-pdebench-darcy-full-training-benchmark-summary.json`
- [ ] Do not touch `docs/index.md` or `docs/studies/index.md` unless the implementation adds a new durable summary path or discoverability would otherwise be stale. This item should normally refresh existing summary and evidence-index surfaces only.

Blocking checks for this task:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_darcy_static_operator_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing closeout surfaces: {missing}")
print("closeout surfaces present")
PY
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/dev/null
python -m json.tool state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json >/dev/null
```

Supporting checks for this task:

```bash
rg -n "darcy|2026-05-04-pdebench-darcy-full-training-benchmark" docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json
git diff --check -- docs/plans/NEURIPS-HYBRID-RESNET-2026 state/NEURIPS-HYBRID-RESNET-2026 artifacts/work/NEURIPS-HYBRID-RESNET-2026 artifacts/checks/NEURIPS-HYBRID-RESNET-2026
```

## Completion Rule

- This backlog item is complete only if the Darcy full benchmark produces same-contract benchmark evidence for all three required profiles and the durable summary plus selector/evidence indexes are updated to point at that evidence.
- If one or more required profiles remain incomplete after a documented narrow fix attempt, close the item as benchmark-incomplete with explicit blocker provenance rather than substituting capped or readiness outputs.
