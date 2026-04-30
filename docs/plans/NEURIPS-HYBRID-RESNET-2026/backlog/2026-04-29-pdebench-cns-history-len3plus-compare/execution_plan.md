# PDEBench CNS History Length 3+ Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether increasing the capped PDEBench `2d_cfd_cns` temporal-context contract from `history_len=2` to `history_len=3` improves the fixed four-row local CNS compare, and evaluate `history_len=4` only if an explicit post-`history_len=3` gate opens.

**Architecture:** Treat this as an audit-first, capped Roadmap Phase 2 context-ablation lane. Reuse the current CNS runner and history-delta reporting surfaces, freeze the audited `history_len=2` anchors into an item-local manifest, prove the `history_len=3` contract in `inspect` mode before any expensive run, execute the mandatory four-row `history_len=3` pilots at `10` and `40` epochs, emit cross-history sidecars against the frozen anchors, and only then decide whether the optional `history_len=4` branch is scientifically justified.

**Tech Stack:** PATH `python`, PyTorch (POLICY-001), `scripts/studies/pdebench_image128/`, pytest, compileall, tmux with `ptycho311` for long runs, Markdown/JSON/CSV/PNG artifacts under `.artifacts/`

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-pdebench-cns-history-len3plus-compare`
- Selection mode: `RECOVERED_IN_PROGRESS`
- Plan authority date: `2026-04-29`
- Scope owner: Roadmap Phase 2 capped CNS follow-up lane
- Selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/8/items/2026-04-29-pdebench-cns-history-len3plus-compare/selected-item-context.md`
- Recorded plan path source: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/8/items/2026-04-29-pdebench-cns-history-len3plus-compare/plan-phase/plan_path.txt`
- Previous plan path from selected-item context: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`
- CNS summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`

This document supersedes earlier plan content for this backlog item and is the execution authority for implementation.

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/steering.md`
- `docs/studies/index.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`
- `.artifacts/review/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare-plan-review.json`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/8/items/2026-04-29-pdebench-cns-history-len3plus-compare/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/8/items/2026-04-29-pdebench-cns-history-len3plus-compare/plan-phase/plan_path.txt`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md` (background only)

## Selected Backlog Objective

- Answer the bounded scientific question: does increasing temporal context from `history_len=2` to `history_len=3` help the capped local CNS compare?
- Preserve the same equal-footing four-row compare surface used by the completed `history_len=1` ablation:
  - `spectral_resnet_bottleneck_base`
  - `hybrid_resnet_cns`
  - `fno_base`
  - `unet_strong`
- Compare fresh longer-context rows against frozen audited `history_len=2` anchors at matching `10`- and `40`-epoch budgets.
- Evaluate `history_len=4` only after the `history_len=3` compare payloads exist and only if the gate in this plan explicitly opens.

## Scope

- Keep the official CNS dataset, capped split family, MSE loss, normalization, batch-size policy, epoch budgets, and metric family fixed.
- Treat temporal context as the only intended task-contract delta: `history_len=2 -> history_len=3`, with derived input-channel and window-eligibility counts recorded explicitly from emitted artifacts.
- Reuse the existing runner and history-delta reporting support if they pass audit; do not churn code just because the plan is fresh.
- Reuse the already audited `history_len=2` anchors from the completed `history_len=1` compare after item-local manifest validation.
- Run required fresh `history_len=3` four-row pilots at `10` and `40` epochs.
- Run `history_len=4` only as a contingent branch behind a written gate.

## Explicit Non-Goals

- Do not widen this work into rollout or autoregressive evaluation.
- Do not widen this work into full-training benchmark rows, suite-level PDE claims, manuscript artifacts, or `/home/ollie/Documents/neurips/` outputs.
- Do not change dataset path, split counts, `max_windows_per_trajectory`, optimizer family, scheduler family, training loss, batch size, or metric family to make longer-history rows look better.
- Do not rerun `history_len=1`; it is prerequisite context only.
- Do not silently add authored FFNO, GNOT, Darcy, SWE, hybrid-spectral architecture, spectral-modes work, physics regularization, or any unrelated backlog lane.
- Do not inspect, run, or summarize `history_len=4` as if it were mandatory. It exists only behind the explicit post-`history_len=3` gate.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, and Fairness Constraints

- Steering requires explicit equal-footing comparisons and forbids silently relaxing fairness constraints.
- The roadmap allows bounded Phase 2 capped CNS follow-ups, but they remain decision-support-only until full-training PDE benchmark gates are satisfied.
- The design, the PDEBench image-suite plan, and the CNS summary keep this lane on the official `2d_cfd_cns` file and the fixed denormalized metric family; this item must not drift into a new dataset, baseline family, or evaluation protocol.
- The selected backlog item makes `history_len=3` mandatory and allows `history_len=4` only behind an explicit gate.
- The current deterministic roadmap gate is a Phase 2 PDEBench plus Phase 3 CDI-preparation selection window, but this item remains Phase 2 work and must not be presented as satisfying Phase 3 CDI progress.
- The fixed equal-footing contract across fresh and reference rows is:
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - split: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - batch size: `4`
  - training loss: `mse`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
  - profiles: `spectral_resnet_bottleneck_base`, `hybrid_resnet_cns`, `fno_base`, `unet_strong`
- The only allowed scientific delta for the base compare is temporal-context length and its derived sample/input-channel contract:
  - reference: `history_len=2`, `concat u[t-2:t] -> u[t]`
  - fresh mandatory run: `history_len=3`, `concat u[t-3:t] -> u[t]`
  - optional gated run: `history_len=4`, `concat u[t-4:t] -> u[t]`
- Because the cap stays at `8` windows per trajectory, the emitted split counts can remain fixed even when raw eligible windows shrink. The durable summary must therefore report both:
  - raw uncapped eligibility change: `19 -> 18` windows per trajectory for `history_len=2 -> 3`, and `19 -> 17` if `history_len=4` opens
  - emitted capped split counts after `max_windows_per_trajectory=8`

## Prerequisite Status

- Satisfied from current durable state:
  - the official `2d_cfd_cns` file is staged and checksum-verified
  - the CNS runner already supports positive `history_len` values and `inspect` / `pilot` modes
  - `scripts/studies/pdebench_image128/reporting.py` and `tests/studies/test_pdebench_image128_runner.py` already contain history-delta compare support for `history_len=1` and `history_len=3` against `history_len=2`
  - the canonical CNS Hybrid shell is fixed to `hybrid_resnet_cns` with skip-add plus `pixelshuffle`
  - the completed `history_len=1` compare already established the fixed rule that only `history_len` and its derived sample/input-channel contract may differ in a cross-history compare
  - the progress ledger records the `history_len=1` compare as completed prerequisite context
  - the progress ledger records `2026-04-29-pdebench-cns-shared-blocks10-1024cap-longer-convergence` as complete; it is adjacent capped CNS evidence, not a prerequisite
  - the exact `history_len=2` anchor family already exists for all four rows at both `10` and `40` epochs, including the backfilled `40`-epoch `hybrid_resnet_cns` anchor
- Reusable frozen reference roots:
  - `10ep` spectral + hybrid: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
  - `10ep` FNO + U-Net: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
  - `40ep` spectral: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
  - `40ep` hybrid: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
  - `40ep` FNO + U-Net: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
- Not prerequisites for this item:
  - any new external baseline
  - full-training benchmark-complete CNS rows
  - manuscript assembly work under `/home/ollie/Documents/neurips/`
  - completion of any parallel capped CNS architecture or convergence lane whose contract does not alter the fixed history-2 anchors named above

## Implementation Architecture

- **Compare-contract unit:** `scripts/studies/pdebench_image128/reporting.py` and `tests/studies/test_pdebench_image128_runner.py` own the reference-manifest format, the history-delta compare payload, row-family labels, and the invariant that fixed-contract fields remain equal while history-derived fields may vary.
- **Execution unit:** `scripts/studies/pdebench_image128/cfd_cns.py` and `tests/studies/test_pdebench_cfd_cns_data.py` own emitted history metadata, split manifests, inspect mode, and the four-row pilot runs. Reuse the current parameterized path rather than adding a new runner.
- **Interpretation unit:** `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, `docs/index.md`, `docs/studies/index.md`, and `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` own the durable claim boundary. They must separate “more context changed metrics” from “more context also changed raw eligible-window counts.”

## Concrete File and Artifact Targets

### Code and Test Surfaces

- Audit and modify only if the pre-run checks expose a real contract gap:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `tests/studies/test_pdebench_image128_runner.py`
- Modify only if inspect or pilot artifacts show missing history metadata or contract emission:
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `tests/studies/test_pdebench_cfd_cns_data.py`
- Verification-only reuse:
  - `tests/studies/test_pdebench_cfd_cns_metrics.py`

### Durable Docs and State

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify `docs/findings.md` only if the result becomes a stable reusable rule rather than a summary-local conclusion

### Required Artifacts

- Create study root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/`
- Create verification log directory: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/verification/`
- Create item-local frozen manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history2_reference_runs.json`
- Create required inspect root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-inspect-<timestamp>/`
- Create required fresh run roots:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-10ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-<timestamp>/`
- Create required compare sidecars:
  - `compare_10ep_history3_against_history2.json`
  - `compare_10ep_history3_against_history2.csv`
  - `compare_40ep_history3_against_history2.json`
  - `compare_40ep_history3_against_history2.csv`
  - sample gallery PNGs only if targets align
- Create gate record: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4_gate_decision.json`
- Create optional gated artifacts only if the gate opens:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-inspect-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-pilot-10ep-<timestamp>/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-pilot-40ep-<timestamp>/`
  - `compare_10ep_history4_against_history2.json`
  - `compare_10ep_history4_against_history2.csv`
  - `compare_40ep_history4_against_history2.json`
  - `compare_40ep_history4_against_history2.csv`

## Execution Guardrails

- The selected backlog item’s required deterministic checks stay mandatory. Any focused selector in this plan is additive, not a replacement.
- Expensive training must wait for a green focused runner/reporting selector and then a green run of the unchanged backlog-item checks.
- If a test, import, path, or harness failure occurs, diagnose, fix, and rerun before considering the item blocked.
- Reserve `BLOCKED` for missing resources, unavailable hardware, roadmap conflict, user decision required, external dependency outside current authority, or an unrecoverable failure after a documented narrow fix attempt.
- For long runs, use tmux, activate `ptycho311`, track the exact launched PID, avoid duplicate launches into the same output root, and require both exit code `0` and fresh required artifacts before treating a run as complete.
- Archive each passing verification command under the item artifact root, preferably in `verification/`, so completion evidence remains local to this lane.
- This lane is capped decision-support evidence only. No result from this plan may be promoted to benchmark-complete PDE evidence.
- Preserve the pointer contract: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/8/items/2026-04-29-pdebench-cns-history-len3plus-compare/plan-phase/plan_path.txt` must continue to contain only `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md`.

## Required Deterministic Checks

The selected backlog item requires these unchanged checks:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Run this focused selector before any expensive pilot:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history or reference_run_manifest or cross_run_compare' -v
```

Archive the passing pytest and compileall logs used to justify completion under the active artifact root, per `docs/TESTING_GUIDE.md`.

## Task 1: Audit Current Cross-History Support and Patch Only if Needed

**Files:**
- Audit: `scripts/studies/pdebench_image128/reporting.py`
- Audit: `tests/studies/test_pdebench_image128_runner.py`
- Modify only if the audit shows a real gap

- [ ] Confirm the current helper and tests already support the required longer-context behavior:
  - fixed-equality fields still match exactly
  - allowed differences remain limited to `history_len`, derived `sample_contract`, and derived `input_channels`
  - fresh rows may have either smaller or larger `history_len` than the reference rows
  - row-family labels derive from actual history lengths rather than hard-coded `history1` labels
  - `hybrid_resnet_base` is still rejected as a proxy anchor for `hybrid_resnet_cns`
- [ ] If the focused selector already passes and the surfaces are adequate, record this task as audit-only and do not edit production files.
- [ ] If the focused selector fails or the audit finds a gap, tighten the tests first where needed, make the smallest justified patch, and rerun until green.

**Verification for Task 1**

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'history or reference_run_manifest or cross_run_compare' -v
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
```

## Task 2: Freeze the `history_len=2` Anchors and Inspect the `history_len=3` Contract

**Files:**
- Artifact-only work unless Task 1 or the inspect outputs expose a metadata gap

- [ ] Build `history2_reference_runs.json` under the item artifact root using the audited `10`- and `40`-epoch history-2 anchor rows for all four required profiles.
- [ ] Use the existing reporting helpers `build_reference_run_manifest(...)` and `write_reference_run_manifest(...)`; do not hand-author the manifest JSON.
- [ ] Confirm every reused reference row contains the required artifacts: `invocation.json`, `dataset_manifest.json`, `split_manifest.json`, `comparison_summary.json`, `metrics_<profile>.json`, `model_profile_<profile>.json`.
- [ ] Run `inspect` mode for `history_len=3` into a fresh output root so the plan records the derived contract before training.
- [ ] Verify the inspect artifacts show the expected `history_len=3` contract:
  - `input_channels=12`
  - `sample_contract=concat u[t-3:t] -> u[t]`
  - raw `windows_per_trajectory=18` and raw `available_windows=180000`
  - emitted split/window counts remain the fixed capped counts unless a real contract bug is found
- [ ] Read `hdf5_metadata.json` for `input_channels`, `windows_per_trajectory`, and `available_windows`; read `dataset_manifest.json` for `history_len` and `sample_contract`; read `split_manifest.json` for emitted `window_counts`.
- [ ] Do not inspect `history_len=4` in this task. That inspect belongs behind the later gate if it opens.

**Execution notes for Task 2**

Build the frozen reference manifest:

```bash
python - <<'PY'
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import build_reference_run_manifest, write_reference_run_manifest

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
payload = build_reference_run_manifest(
    task_id="2d_cfd_cns",
    dataset_file="/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
    split_counts={"train": 512, "val": 64, "test": 64},
    max_windows_per_trajectory=8,
    history_len=2,
    training_loss="mse",
    batch_size=4,
    metric_family=["err_RMSE", "err_nRMSE", "relative_l2", "fRMSE_low", "fRMSE_mid", "fRMSE_high"],
    required_rows={
        "10ep": [
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep",
                "profile_id": "spectral_resnet_bottleneck_base",
                "epochs": 10,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md",
            },
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep",
                "profile_id": "hybrid_resnet_cns",
                "epochs": 10,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md",
            },
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse",
                "profile_id": "fno_base",
                "epochs": 10,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md",
            },
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse",
                "profile_id": "unet_strong",
                "epochs": 10,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md",
            },
        ],
        "40ep": [
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep",
                "profile_id": "spectral_resnet_bottleneck_base",
                "epochs": 40,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md",
            },
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z",
                "profile_id": "hybrid_resnet_cns",
                "epochs": 40,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md",
            },
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse",
                "profile_id": "fno_base",
                "epochs": 40,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md",
            },
            {
                "run_root": ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse",
                "profile_id": "unet_strong",
                "epochs": 40,
                "source_document": "docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md",
            },
        ],
    },
)
write_reference_run_manifest(payload, artifact_root / "history2_reference_runs.json")
print("wrote history2 reference manifest")
PY
```

Inspect command:

```bash
python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-inspect-<timestamp> \
  --history-len 3 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

**Verification for Task 2**

```bash
python - <<'PY'
import json
from pathlib import Path

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
manifest_path = artifact_root / "history2_reference_runs.json"
inspect_root = max(artifact_root.glob("history3-inspect-*"))
metadata = json.loads((inspect_root / "hdf5_metadata.json").read_text())
dataset_manifest = json.loads((inspect_root / "dataset_manifest.json").read_text())
split_manifest = json.loads((inspect_root / "split_manifest.json").read_text())

assert manifest_path.exists(), manifest_path
assert int(metadata["history_len"]) == 3, metadata["history_len"]
assert int(metadata["input_channels"]) == 12, metadata["input_channels"]
assert int(metadata["windows_per_trajectory"]) == 18, metadata["windows_per_trajectory"]
assert int(metadata["available_windows"]) == 180000, metadata["available_windows"]
assert int(dataset_manifest["history_len"]) == 3, dataset_manifest["history_len"]
assert dataset_manifest["sample_contract"] == "concat u[t-3:t] -> u[t]", dataset_manifest["sample_contract"]
assert split_manifest["window_counts"] == {"train": 4096, "val": 512, "test": 512}, split_manifest["window_counts"]
print("history3 reference manifest and inspect contract look correct")
PY
```

## Task 3: Run the Mandatory `history_len=3` Pilots and Emit Cross-History Sidecars

**Files:**
- Reuse: `scripts/studies/pdebench_image128/cfd_cns.py`
- Reuse: `scripts/studies/pdebench_image128/reporting.py`
- Modify only if Task 1 or Task 2 exposed a real contract gap

- [ ] Keep this task blocked on green Task 1 verification and a green run of the unchanged backlog-item deterministic checks.
- [ ] Launch the required fresh `history_len=3` `10`-epoch pilot for the four fixed profiles into a fresh output root.
- [ ] Launch the required fresh `history_len=3` `40`-epoch pilot for the same four fixed profiles into a separate fresh output root.
- [ ] Use tmux for both long runs, activate `ptycho311`, track the exact launched PID, and do not reuse an output root that already has live writers.
- [ ] After each run exits cleanly, generate a history-delta compare payload with `write_history_delta_compare(...)` against the matching `history_len=2` frozen reference rows from Task 2.
- [ ] If cross-run gallery rendering is blocked by target mismatch, keep the compare JSON/CSV and record the blocker; that is not a scored-run failure.

**Execution notes for Task 3**

Pilot commands:

```bash
python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-10ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 3 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

```bash
python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 3 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

Generate compare sidecars after each fresh run:

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import write_history_delta_compare

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
manifest = json.loads((artifact_root / "history2_reference_runs.json").read_text())
fresh_profiles = ["spectral_resnet_bottleneck_base", "hybrid_resnet_cns", "fno_base", "unet_strong"]

write_history_delta_compare(
    output_root=artifact_root,
    epoch_label="10ep",
    fresh_run_root=max(artifact_root.glob("history3-pilot-10ep-*")),
    fresh_profile_ids=fresh_profiles,
    reference_rows=manifest["required_rows"]["10ep"],
)
write_history_delta_compare(
    output_root=artifact_root,
    epoch_label="40ep",
    fresh_run_root=max(artifact_root.glob("history3-pilot-40ep-*")),
    fresh_profile_ids=fresh_profiles,
    reference_rows=manifest["required_rows"]["40ep"],
)
print("wrote history3 compare sidecars")
PY
```

**Verification for Task 3**

```bash
python - <<'PY'
from pathlib import Path

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
required = [
    artifact_root / "compare_10ep_history3_against_history2.json",
    artifact_root / "compare_10ep_history3_against_history2.csv",
    artifact_root / "compare_40ep_history3_against_history2.json",
    artifact_root / "compare_40ep_history3_against_history2.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing history3 compare outputs: {missing}")
print("history3 compare outputs present")
PY
```

## Task 4: Evaluate the `history_len=4` Gate and Only Then Inspect or Run It

**Files:**
- Artifact-only unless a narrow contract gap is discovered

- [ ] Default the `history_len=4` branch to closed.
- [ ] Write `history4_gate_decision.json` after the `history_len=3` compare payloads exist, even if the gate stays closed.
- [ ] Open the gate only if the fresh `40`-epoch `history_len=3` spectral row improves on the matching `history_len=2` spectral row on both aggregate metrics and does not worsen the required high-frequency diagnostic:
  - improve `err_nRMSE`
  - improve `err_RMSE`
  - `fRMSE_high` is less than or equal to the reference value
- [ ] Record the `10`-epoch spectral direction in the same decision file as supporting context. If `10`-epoch and `40`-epoch signals disagree, keep the gate closed unless a written scientific reason is added to `history4_gate_decision.json` before any `history_len=4` inspect or pilot.
- [ ] If the gate stays closed, stop this branch after writing the decision record.
- [ ] If the gate opens, run a fresh `history_len=4` inspect and verify the derived contract before training:
  - `input_channels=16`
  - `sample_contract=concat u[t-4:t] -> u[t]`
  - raw `windows_per_trajectory=17` and raw `available_windows=170000`
  - emitted split/window counts remain `{"train": 4096, "val": 512, "test": 512}`
- [ ] If the gate opens, run fresh `history_len=4` `10`- and `40`-epoch pilots under the same frozen contract and emit the matching history-delta compare payloads against `history_len=2`.
- [ ] If the gate opens, the durable summary must explicitly state that `history_len=4` was a gated follow-up triggered by the stronger `40`-epoch spectral signal, not part of the mandatory base scope.

**Execution notes for Task 4**

Write the default-closed gate record from the emitted `history_len=3` compare payloads. Leave `scientific_reason` empty unless you are intentionally opening the gate despite a `10`-epoch / `40`-epoch disagreement and you can justify that choice in writing before any `history_len=4` run:

```bash
python - <<'PY'
import json
from pathlib import Path

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
compare_10 = json.loads((artifact_root / "compare_10ep_history3_against_history2.json").read_text())
compare_40 = json.loads((artifact_root / "compare_40ep_history3_against_history2.json").read_text())
scientific_reason = ""

def spectral_pair(payload):
    fresh_rows = {row["profile_id"]: row for row in payload["fresh_profile_results"]}
    reference_rows = {row["profile_id"]: row for row in payload["reference_profile_results"]}
    return fresh_rows["spectral_resnet_bottleneck_base"], reference_rows["spectral_resnet_bottleneck_base"]

def gate_signal(fresh, reference):
    return {
        "improves_err_nRMSE": float(fresh["err_nRMSE"]) < float(reference["err_nRMSE"]),
        "improves_err_RMSE": float(fresh["err_RMSE"]) < float(reference["err_RMSE"]),
        "preserves_fRMSE_high": float(fresh["fRMSE_high"]) <= float(reference["fRMSE_high"]),
    }

fresh_10, reference_10 = spectral_pair(compare_10)
fresh_40, reference_40 = spectral_pair(compare_40)
signal_10 = gate_signal(fresh_10, reference_10)
signal_40 = gate_signal(fresh_40, reference_40)
all_10 = all(signal_10.values())
all_40 = all(signal_40.values())
signals_agree = all_10 == all_40
gate_open = all_40 and (signals_agree or bool(scientific_reason.strip()))

payload = {
    "schema_version": "pdebench_cns_history4_gate_decision_v1",
    "gate_status": "open" if gate_open else "closed",
    "decision_rule": "Open only if the 40-epoch history3 spectral row improves err_nRMSE and err_RMSE while not worsening fRMSE_high against history2; disagreements with the 10-epoch signal require a written scientific reason before any history4 run.",
    "scientific_reason": scientific_reason,
    "signals_agree": signals_agree,
    "history3_context": {
        "10ep": {
            "signal": signal_10,
            "fresh_profile": fresh_10,
            "reference_profile": reference_10,
        },
        "40ep": {
            "signal": signal_40,
            "fresh_profile": fresh_40,
            "reference_profile": reference_40,
        },
    },
}

(artifact_root / "history4_gate_decision.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(payload["gate_status"])
PY
```

If the gate stays closed, stop after the decision file. If the gate opens, run the inspect and pilot sequence below.

Inspect command:

```bash
python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode inspect \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-inspect-<timestamp> \
  --history-len 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

Inspect verification:

```bash
python - <<'PY'
import json
from pathlib import Path

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
inspect_root = max(artifact_root.glob("history4-inspect-*"))
metadata = json.loads((inspect_root / "hdf5_metadata.json").read_text())
dataset_manifest = json.loads((inspect_root / "dataset_manifest.json").read_text())
split_manifest = json.loads((inspect_root / "split_manifest.json").read_text())

assert int(metadata["history_len"]) == 4, metadata["history_len"]
assert int(metadata["input_channels"]) == 16, metadata["input_channels"]
assert int(metadata["windows_per_trajectory"]) == 17, metadata["windows_per_trajectory"]
assert int(metadata["available_windows"]) == 170000, metadata["available_windows"]
assert int(dataset_manifest["history_len"]) == 4, dataset_manifest["history_len"]
assert dataset_manifest["sample_contract"] == "concat u[t-4:t] -> u[t]", dataset_manifest["sample_contract"]
assert split_manifest["window_counts"] == {"train": 4096, "val": 512, "test": 512}, split_manifest["window_counts"]
print("history4 inspect contract looks correct")
PY
```

Pilot commands:

```bash
python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-pilot-10ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 4 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

```bash
python scripts/studies/pdebench_image128/cfd_cns.py \
  --mode pilot \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4-pilot-40ep-<timestamp> \
  --profiles spectral_resnet_bottleneck_base,hybrid_resnet_cns,fno_base,unet_strong \
  --history-len 4 \
  --epochs 40 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8
```

Generate compare sidecars after the fresh `history_len=4` runs:

```bash
python - <<'PY'
import json
from pathlib import Path
from scripts.studies.pdebench_image128.reporting import write_history_delta_compare

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
manifest = json.loads((artifact_root / "history2_reference_runs.json").read_text())
fresh_profiles = ["spectral_resnet_bottleneck_base", "hybrid_resnet_cns", "fno_base", "unet_strong"]

write_history_delta_compare(
    output_root=artifact_root,
    epoch_label="10ep",
    fresh_run_root=max(artifact_root.glob("history4-pilot-10ep-*")),
    fresh_profile_ids=fresh_profiles,
    reference_rows=manifest["required_rows"]["10ep"],
)
write_history_delta_compare(
    output_root=artifact_root,
    epoch_label="40ep",
    fresh_run_root=max(artifact_root.glob("history4-pilot-40ep-*")),
    fresh_profile_ids=fresh_profiles,
    reference_rows=manifest["required_rows"]["40ep"],
)
print("wrote history4 compare sidecars")
PY
```

**Verification for Task 4**

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4_gate_decision.json")
payload = json.loads(path.read_text())
assert payload["gate_status"] in {"open", "closed"}, payload
print(f"history4 gate recorded: {payload['gate_status']}")
PY
```

If the gate opens, verify the required inspect and compare artifacts explicitly:

```bash
python - <<'PY'
import json
from pathlib import Path

artifact_root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare")
gate = json.loads((artifact_root / "history4_gate_decision.json").read_text())
if gate["gate_status"] != "open":
    print("history4 gate closed; no additional artifacts required")
    raise SystemExit(0)

required = [
    artifact_root / "compare_10ep_history4_against_history2.json",
    artifact_root / "compare_10ep_history4_against_history2.csv",
    artifact_root / "compare_40ep_history4_against_history2.json",
    artifact_root / "compare_40ep_history4_against_history2.csv",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing history4 compare outputs: {missing}")

compare_10 = json.loads((artifact_root / "compare_10ep_history4_against_history2.json").read_text())
compare_40 = json.loads((artifact_root / "compare_40ep_history4_against_history2.json").read_text())
for payload in [compare_10, compare_40]:
    assert payload["allowed_contract_delta"]["fresh_history_len"] == 4, payload["allowed_contract_delta"]
    assert payload["allowed_contract_delta"]["reference_history_len"] == 2, payload["allowed_contract_delta"]
print("history4 compare outputs present and correctly labeled")
PY
```

## Task 5: Publish the Durable Interpretation and State Updates

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify `docs/findings.md` only if the result becomes a stable reusable rule

- [ ] Write the new durable summary with the fixed compare contract, reference roots, fresh run roots, compare payload paths, exact raw eligible-window deltas, emitted capped split counts, and a clearly bounded interpretation.
- [ ] Update the CNS summary so the current discoverable CNS state includes the longer-history result and any gate decision about `history_len=4`.
- [ ] Update `docs/index.md` and `docs/studies/index.md` so the new summary is discoverable from the repo’s canonical entry points.
- [ ] Update the progress ledger with the plan path, artifact root, compare payloads, verification evidence locations, and the final decision phrased as capped decision-support evidence only.
- [ ] Only promote a new finding into `docs/findings.md` if the result is stable enough to act as a reusable project rule rather than a summary-local scientific observation.

**Durable interpretation requirements**

- The summary must state that this is capped context-ablation evidence only, not a benchmark-complete CNS ranking.
- The summary must state the exact cross-history contract.
- The summary must state both:
  - the raw eligibility deltas caused by longer history (`19 -> 18`, and `19 -> 17` if `history_len=4` opens)
  - the emitted capped split/window counts after `max_windows_per_trajectory=8`
- The summary must state whether `history_len=3` improved or degraded each of the four rows at `10` and `40` epochs.
- The summary must record whether the `history_len=4` gate stayed closed or opened, and why.
- The summary must not silently convert a context-ablation result into a general recommendation for the full PDEBench suite unless a later broader study proves that separately.

**Verification for Task 5**

```bash
python - <<'PY'
from pathlib import Path

required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md"),
    Path("docs/index.md"),
    Path("docs/studies/index.md"),
    Path("state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing durable outputs: {missing}")
print("durable outputs present")
PY
```

Verify the pointer file is still pointer-only:

```bash
python - <<'PY'
from pathlib import Path

pointer = Path("state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/8/items/2026-04-29-pdebench-cns-history-len3plus-compare/plan-phase/plan_path.txt")
expected = "docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md"
text = pointer.read_text(encoding="utf-8").strip()
if text != expected:
    raise SystemExit(f"unexpected plan_path contents: {text!r}")
print("plan_path pointer intact")
PY
```

## Completion Checklist

- [ ] Focused history/reporting selector is green.
- [ ] Required backlog-item deterministic checks are green and archived.
- [ ] `history2_reference_runs.json` exists and points at the audited `history_len=2` anchors.
- [ ] `history_len=3` inspect artifacts prove the expected derived contract before training.
- [ ] The inspect proof records both raw eligibility shrinkage and capped split/window counts.
- [ ] Fresh `history_len=3` `10`- and `40`-epoch pilot roots exited with code `0` and produced the required per-profile artifacts.
- [ ] `compare_10ep_history3_against_history2.*` and `compare_40ep_history3_against_history2.*` exist.
- [ ] `history4_gate_decision.json` exists; if the gate opened, the corresponding `history_len=4` inspect, pilot, and compare artifacts also exist.
- [ ] The durable summary, CNS summary, docs indexes, and progress ledger are updated and all keep the claim boundary at capped context-ablation evidence.
- [ ] `plan_path.txt` still contains only `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-pdebench-cns-history-len3plus-compare/execution_plan.md`.
