# BRDT Physics-Only Objective Ablation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Keep long-running commands under implementation ownership until the tracked PID exits cleanly and the required artifacts are freshly written; use `tmux` plus the `ptycho311` conda environment for launches that run longer than a short local check. Do not create worktrees.

**Goal:** Run an append-only BRDT neural-row ablation that keeps the completed four-row BRDT preflight contract fixed and changes only the neural training objective to `relative_physics_L2(A(q_pred_phys), observed_sinogram)`, then summarize whether U-Net and FNO collapse is primarily objective-induced or persists under physics-only training.

**Architecture:** Reuse the existing BRDT decision-support dataset, validated Born operator, `born_init_image` input contract, fixed sample set, and preflight bundle layout. Extend the BRDT preflight runner only enough to support one explicit physics-only objective variant and a neural-row-only append-only execution path, emit a machine-readable comparison against the completed supervised-plus-Born bundle, and update the NeurIPS evidence indexes without promoting BRDT into a required paper pillar.

**Tech Stack:** Python 3.11 via PATH `python`, PyTorch, existing `scripts/studies/born_rytov_dt/` orchestration, JSON/CSV/Markdown artifact summaries, `tmux`, `pytest`, `compileall`.

---

## Selected Objective

- Objective: train exactly `unet`, `fno_vanilla`, and `hybrid_resnet` with loss weights `image=0`, `physics=0`, `relative_physics=1`, `tv=0`, `positivity=0`.
- Scope: reuse the completed BRDT four-row preflight dataset/operator/input/split/normalization/metric/fixed-sample contract and write a new append-only ablation root plus durable summary and evidence-index updates.
- Explicit non-goals:
  - do not rerun or overwrite the completed supervised-plus-Born bundle;
  - do not launch the classical row, FFNO, direct-sinogram input, Rytov mode, limited-angle data, multi-seed robustness, or manuscript-facing BRDT packaging;
  - do not rewrite the NeurIPS roadmap, promote BRDT into CDI/CNS headline evidence, or touch `/home/ollie/Documents/neurips/index.md`;
  - do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Constraints And Prerequisites

- Steering and roadmap boundary:
  - BRDT remains a candidate third lane only; CDI `lines128` and PDEBench CNS remain the required manuscript pillars.
  - Equal-footing comparisons must stay explicit. The ablation must preserve the same dataset, operator, input mode, split, train-only normalization, metric schema, fixed-sample IDs, epoch budget, batch size, and learning rate unless a narrow fix requires a uniform documented change across all three neural rows.
  - The work must stay append-only and decision-support-only. Use explicit claim-boundary language such as `decision_support_append_only`; do not call any result paper-grade or manuscript-ready.
- Progress-ledger status that matters here:
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` confirms the initiative has completed Phase 0/1 and early Phase 2 tranches, but it does not elevate BRDT into a required paper phase.
  - The gating prerequisite for this item is the completed backlog item `2026-04-29-brdt-four-row-preflight`, not a new roadmap promotion.
- Read-only prerequisite bundle already present:
  - baseline root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
  - dataset id: `brdt128_decision_support_preflight`
  - split: `2048 / 256 / 256`
  - input mode: `born_init_image`
  - fixed sample seed/IDs: `17` and `[145, 83, 255, 126]`
  - current default neural objective in the baseline bundle: supervised image loss plus Born consistency with nonzero image/physics/relative-physics/TV/positivity terms
- Failure-handling rule:
  - do not mark the item `BLOCKED` for ordinary import/test/path/harness failures; diagnose, apply a narrow fix, and rerun.
  - reserve `BLOCKED` for missing prerequisite artifacts, unavailable hardware, duplicate-writer conflicts that cannot be safely resolved, roadmap/user-authority conflicts, or a launch failure that remains unrecoverable after a documented narrow fix attempt.

## Implementation Architecture

- Unit 1: Objective and row-selection contract
  - Owns the explicit physics-only objective surface, the neural-row-only selection path, and manifest lineage to the completed supervised-plus-Born preflight root.
- Unit 2: Append-only ablation bundle emission
  - Owns the long-run orchestration, per-row training/eval outputs, machine-readable metrics, dynamic-range diagnostics, fixed-sample arrays/visuals, and baseline-versus-ablation comparison payloads.
- Unit 3: Durable evidence discoverability
  - Owns the checked-in summary and the updates to `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `paper_evidence_index.md`.

## File And Artifact Targets

Mandatory code surfaces likely to change:

- `scripts/studies/born_rytov_dt/run_config.py`
- `scripts/studies/born_rytov_dt/run_preflight.py`
- `scripts/studies/born_rytov_dt/preflight_metrics.py`
- `tests/studies/test_born_rytov_dt_preflight.py`

Mandatory contract outputs for this item:

- new append-only artifact root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-physics-only-objective-ablation/`
- within that root:
  - `preflight_manifest.json`
  - `metrics.json`
  - `metrics.csv`
  - `metric_schema.json`
  - `visual_manifest.json`
  - `comparison_to_supervised_plus_born.json`
  - `comparison_to_supervised_plus_born.csv`
  - `rows/unet/`, `rows/fno_vanilla/`, `rows/hybrid_resnet/` with invocation provenance, row summary, and saved model state
  - `visuals/` plus `figures/source_arrays/`
- durable checked-in summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_physics_only_objective_ablation_summary.md`
- durable evidence/index updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

Preferred packaging only if it keeps the runner readable:

- a small helper under `scripts/studies/born_rytov_dt/` for comparison-summary emission instead of growing `run_preflight.py` further
- a dedicated objective label such as `relative_physics_only` rather than five unrelated free-form knobs

### Task 1: Add The Physics-Only Execution Contract

**Files:**
- Modify: `scripts/studies/born_rytov_dt/run_config.py`
- Modify: `scripts/studies/born_rytov_dt/run_preflight.py`
- Modify: `scripts/studies/born_rytov_dt/preflight_metrics.py`
- Test: `tests/studies/test_born_rytov_dt_preflight.py`

- [ ] Add one explicit execution surface for this item's objective contract. Prefer a named preset such as `relative_physics_only` that resolves exactly to `{"image": 0.0, "physics": 0.0, "relative_physics": 1.0, "tv": 0.0, "positivity": 0.0}` and is recorded verbatim in the emitted manifest.
- [ ] Keep the existing supervised-plus-Born preflight default unchanged when the new objective is not requested.
- [ ] Add a bounded row-selection surface so this ablation root trains only `unet`, `fno_vanilla`, and `hybrid_resnet`. The completed classical row is comparison context only and must not be rerun or copied into the new root as if it were newly executed.
- [ ] Record append-only lineage in `preflight_manifest.json`: source baseline root, source baseline `preflight_manifest.json`, source baseline `metrics.json`, claim boundary, selected rows, fixed sample IDs, training seed, and resolved objective weights.
- [ ] Emit machine-readable output dynamic-range diagnostics for each neural row. Use stable physical-`q` aggregate fields such as min/max/mean/std over the evaluation predictions so collapse detection is discoverable without parsing prose.
- [ ] If new metrics fields change the BRDT metric contract rather than merely adding optional `extra` metadata, bump the metric-schema version and describe the change in the summary instead of silently drifting the meaning of `brdt_preflight_metrics_v1`.

**Verification:**
- [ ] **Blocking:** add focused tests in `tests/studies/test_born_rytov_dt_preflight.py` that prove the new objective surface resolves to the exact five weights, the default preflight path is unchanged, the selected-row surface excludes the classical row, and the ablation manifest records baseline lineage plus dynamic-range fields.
- [ ] **Supporting:** `python -m scripts.studies.born_rytov_dt.run_preflight --dry-run ...` writes a fresh manifest and schema for the ablation root without touching the completed baseline root.

### Task 2: Pass Pre-Launch Gates Before Any Expensive Run

**Files:**
- Reuse: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json`
- Reuse: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json`
- Test: `tests/studies/test_born_rytov_dt_adapters.py`
- Test: `tests/studies/test_born_rytov_dt_preflight.py`

- [ ] Confirm the completed BRDT prerequisite inputs still exist and are readable before launch.
- [ ] Run the new focused preflight selector first so objective-specific regressions are caught before the long run.
- [ ] Run the backlog item's required deterministic checks before launching training, because the ablation run is expensive relative to these checks.
- [ ] Keep Python invocation on PATH `python`; do not introduce repo-specific interpreter wrappers.

**Verification:**
- [ ] **Blocking:** run the required input-presence command exactly:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_task_adapters.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing BRDT physics-only ablation inputs: {missing}")
print("brdt physics-only ablation inputs present")
PY
```

- [ ] **Blocking:** run the backlog-required test command exactly:

```bash
pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
```

- [ ] **Blocking:** run the backlog-required compile command exactly:

```bash
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```

- [ ] **Supporting:** if Task 1 adds a narrow new selector, run it before the broader required pytest command; do not replace the required command with the narrow selector.

### Task 3: Launch And Complete The Append-Only Neural Ablation

**Files:**
- Modify or reuse: `scripts/studies/born_rytov_dt/run_preflight.py`
- Output: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-physics-only-objective-ablation/`

- [ ] Launch the ablation in `tmux` under `ptycho311`, using the completed decision-support dataset manifest from the baseline preflight and the same fixed sample seed/IDs, epoch budget, batch size, learning rate, and training seed unless a narrow fix requires a uniform documented change across all three neural rows.
- [ ] Prefer an invocation equivalent to:

```bash
python -m scripts.studies.born_rytov_dt.run_preflight \
  --manifest .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/decision_support_dataset/dataset_manifest.json \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-physics-only-objective-ablation \
  --epochs 20 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --device cuda \
  --in-channels 1 \
  --fixed-sample-count 4 \
  --fixed-sample-seed 17 \
  --seed 42 \
  --hybrid-label hybrid_resnet \
  --objective-preset relative_physics_only \
  --rows unet,fno_vanilla,hybrid_resnet
```

- [ ] If implementation uses different flag names, keep the resolved contract identical and record it verbatim in `invocation.json` and `preflight_manifest.json`.
- [ ] Track the exact launched PID and wait on that PID. Do not use broad `pgrep -f` polling loops. Do not launch a duplicate run if another process is already writing to the same `--output-root`.
- [ ] Treat the run as complete only when the tracked PID exits `0` and the required top-level bundle artifacts plus all three per-row directories are freshly written.
- [ ] If one neural row fails for a normal code or environment reason, diagnose, apply the narrowest fix that preserves the approved contract, and rerun under the same output root or a documented fresh root; do not stop at the first failure unless the failure is truly unrecoverable.

**Verification:**
- [ ] **Blocking:** the new root contains fresh `preflight_manifest.json`, `metrics.json`, `metrics.csv`, `metric_schema.json`, `visual_manifest.json`, `comparison_to_supervised_plus_born.json`, and per-row directories for `unet`, `fno_vanilla`, and `hybrid_resnet`.
- [ ] **Blocking:** each neural row records final loss breakdown, parameter count, runtime, and output dynamic-range fields; the baseline root remains untouched.
- [ ] **Supporting:** confirm the new root reuses the same fixed sample IDs as the baseline bundle and writes fresh source arrays for those samples.

### Task 4: Emit The Comparison Read And Durable Summary

**Files:**
- Modify or add: comparison-emission logic under `scripts/studies/born_rytov_dt/`
- Add: `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_physics_only_objective_ablation_summary.md`

- [ ] Build a machine-readable comparison between the completed supervised-plus-Born bundle and the new physics-only bundle for `unet`, `fno_vanilla`, and `hybrid_resnet` only.
- [ ] For each row, report the baseline metrics, physics-only metrics, metric deltas, runtime, parameter count, final loss breakdown, and output dynamic-range diagnostics. Parameter counts should match the baseline unless implementation intentionally changed the model body, which this item does not authorize.
- [ ] Write a concise checked-in summary that answers the backlog question directly: whether U-Net and FNO collapse is primarily objective-induced or whether it persists even under pure relative-physics training, and whether Hybrid ResNet’s gap narrows, holds, or widens under the same objective change.
- [ ] State the claim boundary explicitly in the summary. The correct read is still candidate-lane decision support, not paper-grade BRDT evidence.
- [ ] Keep the completed supervised-plus-Born bundle and the existing BRDT paper-local assets read-only. Do not rewrite `tables/brdt_decision_support_metrics.*` or `figures/brdt_decision_support_recon.png` from this item.

**Verification:**
- [ ] **Blocking:** the summary cites the exact baseline root and ablation root, states that the baseline rows were not rerun or overwritten, and includes the explicit decision-support-only boundary.
- [ ] **Supporting:** if the diagnosis is ambiguous from scalar metrics alone, add one lightweight comparison field or figure inside the ablation root rather than widening scope into new experiments.

### Task 5: Update Durable Evidence Indexes

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

- [ ] Add a distinct BRDT objective-ablation contract entry in `model_variant_index.json`, or add equivalent objective-distinguishing fields, so the new neural rows cannot be mistaken for the earlier supervised-plus-Born rows.
- [ ] Add three new BRDT model-variant entries for `unet`, `fno_vanilla`, and `hybrid_resnet` under the new append-only physics-only root. Keep the earlier supervised-plus-Born entries unchanged.
- [ ] Add or extend one BRDT ablation family in `ablation_index.json` that ties the baseline supervised-plus-Born bundle and the new physics-only bundle together as a same-contract objective study.
- [ ] Update `evidence_matrix.md` and `paper_evidence_index.md` with the new backlog item, its summary authority, artifact root, and candidate-lane boundary. Use language that makes the append-only diagnostic value clear without relabeling BRDT as a required manuscript pillar.
- [ ] Leave `/home/ollie/Documents/neurips/index.md` untouched. This item is not the Phase 5 paper-facing evidence assembly pass.

**Verification:**
- [ ] **Blocking:** every new index pointer resolves to a real checked-in summary or artifact root, and no updated index entry calls the ablation `paper_grade`, `manuscript-ready`, `full_training`, or a CDI/CNS pillar.
- [ ] **Supporting:** if `docs/index.md` is not updated, state in the execution report that the authoritative discoverability surfaces for this result are `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `paper_evidence_index.md`.

## Final Closeout Checks

- [ ] Re-run the backlog-required `pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py` selector after the last code edit that affects BRDT execution.
- [ ] Re-run `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch` after the last code edit that affects BRDT execution.
- [ ] Verify the summary and evidence-index updates point at the final ablation root and the checked-in summary path.
- [ ] In the final execution report, explicitly state that normal verification failures were diagnosed and retried before any blocker decision, and identify any remaining blocker only if it meets the narrow blocker rule from this plan.
