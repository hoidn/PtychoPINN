# PDEBench GNOT Paper-Default CNS Compare Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a fresh `40`-epoch paper-default `gnot_cns_base` row on the existing capped PDEBench `2d_cfd_cns` contract in `ptycho311_2`, then compare that row against the pinned spectral anchor and update the durable CNS/GNOT interpretation without changing the fixed local contract.

**Architecture:** Treat prior GNOT integration and the paper-default profile patch as already landed. Start with deterministic preflight and the required backlog checks, use `ptycho311_2` as the only allowed runtime host, run one bounded fresh smoke/preflight before the `40`-epoch job, then write the result into the existing GNOT/CNS durable summaries with exact anchor-root provenance. Keep code edits contingency-only: only touch the existing `pdebench_image128` GNOT surfaces if the preflight proves the paper-default profile drifted or the validated runtime contract regressed.

**Tech Stack:** PATH `python`, conda env `ptycho311_2`, PyTorch `2.4.1+cu124`, DGL `2.4.0+cu124`, existing PDEBench CNS runner under `scripts/studies/pdebench_image128/`, pytest, compileall, tmux for long runs, Markdown/JSON/CSV artifacts under `.artifacts/`

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-22-pdebench-gnot-paper-default-cns-compare`
- Status: pending
- Date: 2026-04-22
- Scope owner: Roadmap Phase 2 capped external-baseline follow-up
- Selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-22-pdebench-gnot-paper-default-cns-compare/selected-item-context.md`
- Previous background plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_paper_default_cns_compare_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
- CNS summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/`

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/studies/index.md`
- `docs/workflows/pytorch.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-22-pdebench-gnot-paper-default-cns-compare/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_paper_default_cns_compare_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-21-pdebench-author-ffno-equal-footing-cns/execution_plan.md`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-22-pdebench-gnot-paper-default-cns-compare-plan-review.json`
- `scripts/studies/pdebench_image128/{run_config.py,cfd_cns.py,models.py,gnot_adapter.py,reporting.py}`
- `scripts/studies/run_pdebench_image128_suite.py`
- `tests/studies/{test_pdebench_image128_models.py,test_pdebench_image128_runner.py}`

## Objective

- Rerun `gnot_cns_base` with the already-patched paper-default recipe on the fixed local PDEBench CNS capped contract.
- Reuse the existing spectral anchor as the mandatory comparison target rather than rerunning that anchor.
- Update durable summaries and findings so they answer whether the paper-default rerun changes the first local GNOT read.

## Scope

- Use the validated `ptycho311_2` runtime path only.
- Keep the local CNS contract fixed:
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - task: `2d_cfd_cns`
  - resolution: `128x128`
  - `history_len=2`
  - split: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - batch size: `4`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Use the patched paper-default GNOT recipe already encoded in the repo:
  - `training_loss=relative_l2`
  - `optimizer=AdamW`
  - `learning_rate=1e-3`
  - `weight_decay=5e-5`
  - `scheduler=OneCycleLR`
  - `gnot_hidden=128`
- Keep all output roots fresh and timestamped.

## Explicit Non-Goals

- Do not add a new external-model integration path.
- Do not change the CNS data contract, split counts, `history_len`, resolution, or batch size.
- Do not search for a different runtime environment if `ptycho311_2` regresses; treat that as a blocker and record it.
- Do not rerun or alter the current spectral anchor just to make the compare easier.
- Do not widen this work into full-training benchmark-complete CNS claims, later roadmap phases, or `/home/ollie/Documents/neurips/` artifact assembly.
- Do not claim paper-faithful GNOT reproduction beyond the fixed local CNS contract.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Steering And Roadmap Constraints

- Steering required the author-FFNO lane to run before this GNOT rerun unless FFNO was blocked. The progress ledger shows the author-FFNO backlog item completed on 2026-04-22, so that ordering prerequisite is satisfied.
- Steering still requires explicit equal-footing boundaries. For this backlog item, “equal footing” means the data file, split, history contract, batch size, and reported metrics stay fixed. The GNOT training recipe is intentionally different from the local spectral recipe; that difference must be explicit in every summary and sidecar artifact.
- The roadmap and design still treat capped CNS runs as decision-support evidence only. This plan must not upgrade the result into benchmark-performance evidence because the run still uses a capped `512 / 64 / 64` slice rather than the full available training split.
- The selected-item context binds the work to the already integrated official GNOT source and forbids drifting back to the earlier local fairness-probe recipe.

## Prerequisite Status

- Satisfied from the progress ledger and durable summaries:
  - official `2d_cfd_cns` data file is staged and checksum-verified
  - CNS adapter, metrics, and reporting path are implemented and tested
  - the first local GNOT integration already succeeded in `ptycho311_2`
  - `gnot_cns_base` is already patched to the paper-default recipe in `run_config.py` and `cfd_cns.py`
  - the author-FFNO backlog item completed, satisfying the current steering order
- Still true after this item:
  - the result remains capped decision-support evidence only
  - full benchmark-complete CNS comparison work is still a later roadmap concern

## Fixed Reference Anchors

### Mandatory spectral anchor

- `spectral_resnet_bottleneck_base` `40`-epoch root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`

### Prior local GNOT fairness-probe row

- first GNOT row root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z`

### Optional contextual anchors

- `fno_base` `40`-epoch root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- `unet_strong` `40`-epoch root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- Mention optional anchors only if the durable summary pins these exact roots and labels them as contextual, not mandatory comparison targets.

## Implementation Architecture

- **Preflight / Repair Gate:** run the required deterministic checks and a targeted GNOT recipe assertion before any fresh execution. If they fail, repair only the existing GNOT profile/runner surfaces and matching tests; otherwise skip code edits entirely.
- **Runtime Execution Gate:** use `ptycho311_2` only, run one bounded fresh preflight to verify dependency and recipe health, then launch the `40`-epoch paper-default GNOT job in tmux with exact-PID waiting and a fresh output root.
- **Evidence / Interpretation Gate:** compare the fresh GNOT row against the pinned spectral anchor and the prior fairness-probe GNOT row using pinned artifact roots, write any one-off merged compare artifact inside the fresh run root, and update the durable GNOT/CNS summaries plus the active finding and ledger state.

## Concrete File And Artifact Targets

### Expected durable changes

- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify if interpretation changes or is tightened: `docs/findings.md`
- Modify for discoverability if the summary becomes the durable canonical record: `docs/index.md`
- Modify for study discoverability if absent: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`

### Fresh artifacts required

- Create: fresh smoke/preflight root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/`
- Create: fresh `40`-epoch run root under the same artifact root
- Create inside the `40`-epoch run root:
  - `invocation.json`
  - `invocation.sh`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_gnot_cns_base.json`
  - `model_profile_gnot_cns_base.json`
  - `comparison_gnot_cns_base_sample0.npz`
  - `comparison_gnot_cns_base_sample0.png`
- Create inside the `40`-epoch run root if the direct-compare collation is generated:
  - `compare_40ep_against_existing.json`
  - `compare_40ep_against_existing.csv`
  - optionally `compare_40ep_sample0.png`
  - optionally `compare_40ep_sample0_error.png`

### Contingency-only code/test repair surfaces

- `scripts/studies/pdebench_image128/run_config.py`
- `scripts/studies/pdebench_image128/cfd_cns.py`
- `scripts/studies/pdebench_image128/models.py`
- `scripts/studies/pdebench_image128/gnot_adapter.py`
- `tests/studies/test_pdebench_image128_models.py`
- `tests/studies/test_pdebench_image128_runner.py`

## Task 1: Freeze The Contract And Revalidate The Paper-Default Path

**Files:**
- Modify only if preflight fails: `scripts/studies/pdebench_image128/{run_config.py,cfd_cns.py,models.py,gnot_adapter.py}`
- Modify only if preflight fails: `tests/studies/{test_pdebench_image128_models.py,test_pdebench_image128_runner.py}`

- [ ] **Step 1: Record the fixed contract and anchor roots in the working notes for the run**

Use the exact dataset path, split counts, `history_len`, batch size, and anchor roots listed above. Do not invent a new spectral reference root during implementation.

- [ ] **Step 2: Run the backlog-mandated deterministic checks before any fresh execution**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q
python -m compileall -q scripts/studies/pdebench_image128
```

- [ ] **Step 3: Add the targeted GNOT paper-default assertion as a stronger preflight**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_runner.py::test_cfd_cns_gnot_profile_uses_paper_default_training_recipe -q
```

Expected: pass, proving the repo still encodes the intended paper-default recipe before the expensive run.

- [ ] **Step 4: If any check fails, repair only the existing GNOT profile/runner path**

Allowed repair scope:

- restore `gnot_cns_base` paper-default profile values in `run_config.py`
- restore GNOT-specific training-recipe handling in `cfd_cns.py`
- repair GNOT model-build wiring only if the validated path regressed
- add or update only the tests needed to lock the regression

If `ptycho311_2` compatibility itself is gone, stop and document a blocker instead of widening the environment search.

**Verification for Task 1**

- The two backlog-required checks pass.
- The targeted recipe test passes.
- If code changed, rerun the same three commands and capture the logs for the execution report.

## Task 2: Run A Fresh Paper-Default Runtime Preflight In `ptycho311_2`

**Files / artifacts:**
- Create: fresh smoke root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/`

- [ ] **Step 1: Activate the validated host and confirm package/runtime identity**

Inside the shell that will run the study:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311_2
export GNOT_ROOT=/home/ollie/Documents/PtychoPINN/.artifacts/external/gnot
python - <<'PY'
import torch, dgl
print({"python_ok": True, "torch": torch.__version__, "dgl": dgl.__version__})
PY
```

- [ ] **Step 2: Launch a bounded fresh smoke/preflight with a timestamped output root**

Recommended command shape:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot-paper-default-smoke-<timestamp> \
  --profiles gnot_cns_base \
  --history-len 2 \
  --epochs 1 \
  --batch-size 4 \
  --max-train-trajectories 8 \
  --max-val-trajectories 2 \
  --max-test-trajectories 2 \
  --max-windows-per-trajectory 2 \
  --device cuda \
  --num-workers 0
```

- [ ] **Step 3: Verify that the smoke artifacts prove the right recipe and dependency path**

Check the fresh smoke root for:

- exit status `0`
- `metrics_gnot_cns_base.json`
- `model_profile_gnot_cns_base.json`
- `comparison_summary.json`
- `comparison_gnot_cns_base_sample0.npz`
- `comparison_gnot_cns_base_sample0.png`

Then assert the metrics file records:

- `training_loss == "relative_l2"`
- `optimizer == "AdamW"`
- `scheduler == "OneCycleLR"`
- `learning_rate == 1e-3`
- `weight_decay == 5e-5`

**Verification for Task 2**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
run_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot-paper-default-smoke-<timestamp>")
required = [
    "comparison_summary.json",
    "metrics_gnot_cns_base.json",
    "model_profile_gnot_cns_base.json",
    "comparison_gnot_cns_base_sample0.npz",
    "comparison_gnot_cns_base_sample0.png",
]
missing = [name for name in required if not (run_root / name).exists()]
if missing:
    raise SystemExit(f"missing smoke artifacts: {missing}")
metrics = json.loads((run_root / "metrics_gnot_cns_base.json").read_text())
expected = {
    "training_loss": "relative_l2",
    "optimizer": "AdamW",
    "scheduler": "OneCycleLR",
    "learning_rate": 1e-3,
    "weight_decay": 5e-5,
}
for key, value in expected.items():
    if metrics.get(key) != value:
        raise SystemExit(f"unexpected {key}: {metrics.get(key)!r} != {value!r}")
print("paper-default smoke verified")
PY
```

If this fails, stop before the `40`-epoch run and either repair the validated GNOT path or record a blocker.

## Task 3: Launch The Fresh `40`-Epoch Paper-Default GNOT Run

**Files / artifacts:**
- Create: fresh `40`-epoch run root under `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/`

- [ ] **Step 1: Start the long run in tmux with the exact-PID guardrail**

Use tmux and activate `ptycho311_2` inside the pane. Follow the repo long-run rule: launch one command against one fresh output root, capture the exact PID, and `wait "$pid"` instead of polling broad process names.

Recommended command body:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot-paper-default-40ep-<timestamp> \
  --profiles gnot_cns_base \
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

- [ ] **Step 2: Refuse duplicate or stale output-root reuse**

Do not pass `--allow-existing-output-root`. If the chosen timestamped root already exists or another run is still writing there, stop and choose a new fresh root.

- [ ] **Step 3: Verify completion using the tracked exit code plus fresh required artifacts and metric consistency**

Required fresh artifacts:

- `comparison_summary.json`
- `comparison_summary.csv`
- `metrics_gnot_cns_base.json`
- `model_profile_gnot_cns_base.json`
- `dataset_manifest.json`
- `split_manifest.json`
- `invocation.json`
- `comparison_gnot_cns_base_sample0.npz`
- `comparison_gnot_cns_base_sample0.png`

**Verification for Task 3**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
run_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot-paper-default-40ep-<timestamp>")
required = [
    "comparison_summary.json",
    "comparison_summary.csv",
    "metrics_gnot_cns_base.json",
    "model_profile_gnot_cns_base.json",
    "dataset_manifest.json",
    "split_manifest.json",
    "invocation.json",
    "comparison_gnot_cns_base_sample0.npz",
    "comparison_gnot_cns_base_sample0.png",
]
missing = [name for name in required if not (run_root / name).exists()]
if missing:
    raise SystemExit(f"missing long-run artifacts: {missing}")

invocation = json.loads((run_root / "invocation.json").read_text())
argv = invocation.get("argv", [])

def arg_value(flag: str) -> str:
    try:
        idx = argv.index(flag)
    except ValueError as exc:
        raise SystemExit(f"invocation missing {flag}") from exc
    try:
        return argv[idx + 1]
    except IndexError as exc:
        raise SystemExit(f"invocation missing value for {flag}") from exc

expected_args = {
    "--task": "2d_cfd_cns",
    "--profiles": "gnot_cns_base",
    "--history-len": "2",
    "--epochs": "40",
    "--batch-size": "4",
    "--max-train-trajectories": "512",
    "--max-val-trajectories": "64",
    "--max-test-trajectories": "64",
    "--max-windows-per-trajectory": "8",
}
for flag, value in expected_args.items():
    observed = arg_value(flag)
    if observed != value:
        raise SystemExit(f"unexpected {flag}: {observed!r} != {value!r}")

metrics = json.loads((run_root / "metrics_gnot_cns_base.json").read_text())
expected_recipe = {
    "profile_id": "gnot_cns_base",
    "training_loss": "relative_l2",
    "optimizer": "AdamW",
    "scheduler": "OneCycleLR",
    "learning_rate": 1e-3,
    "weight_decay": 5e-5,
}
for key, value in expected_recipe.items():
    if metrics.get(key) != value:
        raise SystemExit(f"unexpected {key}: {metrics.get(key)!r} != {value!r}")
if len(metrics.get("train_epoch_losses", [])) != 40:
    raise SystemExit(
        f"expected 40 epoch losses, got {len(metrics.get('train_epoch_losses', []))}"
    )
if int(metrics.get("train_batches", 0)) <= 0:
    raise SystemExit(f"unexpected train_batches: {metrics.get('train_batches')!r}")
print({
    "relative_l2": metrics.get("relative_l2"),
    "err_nRMSE": metrics.get("err_nRMSE"),
    "fRMSE_high": metrics.get("fRMSE_high"),
})
PY
```

## Task 4: Write The Anchored Compare And Durable Interpretation

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify if needed: `docs/findings.md`
- Modify if absent: `docs/index.md`
- Modify if absent: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Create in the fresh run root if useful: `compare_40ep_against_existing.json`, `compare_40ep_against_existing.csv`, optional combined gallery PNGs

- [ ] **Step 1: Compare the fresh GNOT row against the pinned spectral anchor**

Load metrics from:

- fresh run: `metrics_gnot_cns_base.json`
- pinned spectral root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep/metrics_spectral_resnet_bottleneck_base.json`

Write a small sidecar compare artifact inside the fresh run root if it helps review. It must record:

- exact run roots
- task/data contract parity fields
- per-row metrics
- each row’s training recipe fields
- an explicit note that GNOT uses the paper-default `relative_l2` / `AdamW` / `OneCycleLR` recipe while the pinned spectral anchor is an existing local spectral row

Do not force this compare through `write_cross_run_compare()` unless the contract check is intentionally relaxed for mixed training recipes and covered by tests. A one-off inline Python collation is preferred here to avoid widening generic tooling.

- [ ] **Step 2: Compare the fresh paper-default row against the first local GNOT fairness probe**

Use the prior `10`-epoch fairness-probe root only for interpretation, not as an identical-contract merged compare artifact. The summary must state explicitly that the older run differs in both training recipe and epoch budget, so the delta is directional evidence rather than a clean one-variable ablation.

- [ ] **Step 3: Update the durable GNOT and CNS summaries**

`pdebench_gnot_cns_compare_summary.md` must state:

- the fresh run root
- that the run executed in `ptycho311_2`
- the exact paper-default recipe used
- the pinned spectral anchor root
- whether paper-default GNOT materially improved over the first local fairness-probe run
- whether GNOT still trails the spectral anchor on aggregate error
- whether the remaining failure mode is still mainly low-frequency/global structure error or something else
- that this remains capped decision-support evidence only

`pdebench_2d_cfd_cns_summary.md` should add or update the external-baseline section so the current GNOT status is discoverable alongside the author-FFNO lane and the canonical spectral/hybrid context.

- [ ] **Step 4: Update findings, discoverability, and ledger state**

- Update `docs/findings.md` if the fresh paper-default run materially changes, sharpens, or confirms `PDEBENCH-CNS-GNOT-001`.
- If `docs/index.md` and `docs/studies/index.md` do not already expose the durable GNOT summary as a discoverable source, add concise entries.
- Append a `post_completion_updates` progress-ledger record for this backlog item, following the completed author-FFNO backlog item as the structural precedent and using `tranche_id: "2026-04-22-pdebench-gnot-paper-default-cns-compare"`. Record:
  - update timestamp
  - decision and decision scope
  - artifact root
  - fresh run root
  - `summary_path`
  - `cns_summary_path`
  - `findings_changed`
  - `performance_assessment_complete: false`

**Verification for Task 4**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
docs = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json"),
]
missing = [str(path) for path in docs if not path.exists()]
if missing:
    raise SystemExit(f"missing durable outputs: {missing}")
summary_text = docs[0].read_text(encoding="utf-8")
required_terms = [
    "ptycho311_2",
    "gnot-paper-default-40ep",
    "spectral40ep",
    "decision-support",
]
missing_terms = [term for term in required_terms if term not in summary_text]
if missing_terms:
    raise SystemExit(f"summary missing terms: {missing_terms}")
ledger = json.loads(docs[2].read_text(encoding="utf-8"))
matches = [
    entry
    for entry in ledger.get("post_completion_updates", [])
    if entry.get("tranche_id") == "2026-04-22-pdebench-gnot-paper-default-cns-compare"
]
if len(matches) != 1:
    raise SystemExit(f"expected exactly one ledger entry for backlog item, got {len(matches)}")
entry = matches[0]
required_entry_fields = [
    "updated_at_utc",
    "decision",
    "decision_scope",
    "artifact_root",
    "fresh_run_root",
    "summary_path",
    "cns_summary_path",
    "findings_changed",
    "performance_assessment_complete",
]
missing_entry_fields = [field for field in required_entry_fields if field not in entry]
if missing_entry_fields:
    raise SystemExit(f"ledger entry missing fields: {missing_entry_fields}")
if entry["artifact_root"] != ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare":
    raise SystemExit(f"unexpected artifact_root: {entry['artifact_root']!r}")
if "gnot-paper-default-40ep" not in entry["fresh_run_root"]:
    raise SystemExit(f"unexpected fresh_run_root: {entry['fresh_run_root']!r}")
if entry["summary_path"] != "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md":
    raise SystemExit(f"unexpected summary_path: {entry['summary_path']!r}")
if entry["cns_summary_path"] != "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md":
    raise SystemExit(f"unexpected cns_summary_path: {entry['cns_summary_path']!r}")
print("durable outputs verified")
PY
```

## Required Deterministic Checks

These are mandatory for this backlog item and must be run at least once on the final code state, even if no code edits were needed:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py -q
python -m compileall -q scripts/studies/pdebench_image128
```

Recommended stronger supplement before the long run:

```bash
python -m pytest tests/studies/test_pdebench_image128_runner.py::test_cfd_cns_gnot_profile_uses_paper_default_training_recipe -q
```

## Completion Criteria

- A fresh `40`-epoch `gnot_cns_base` run exists under a timestamped root in `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/`.
- The run completed in `ptycho311_2` with exit code `0`.
- The fresh metrics and sample artifacts prove the paper-default recipe was actually used.
- The durable GNOT summary explicitly compares the fresh row against the pinned spectral anchor and the prior fairness-probe GNOT row with correct caveats.
- Any changed interpretation is reflected in `docs/findings.md` and the progress ledger.
- The work remains within the capped decision-support boundary and does not widen into later roadmap phases.
