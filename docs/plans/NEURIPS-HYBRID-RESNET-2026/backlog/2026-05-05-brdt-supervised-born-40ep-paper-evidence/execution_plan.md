# BRDT Supervised+Born 40-Epoch Paper Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees. Use `tmux` plus the `ptycho311` environment for multi-minute BRDT runs, track the exact launched PID, and treat a long run as complete only when that PID exits `0` and the expected artifacts are freshly written. Do not mark this item `BLOCKED` for ordinary test/import/path/environment failures; diagnose, apply a narrow fix, and rerun first. Reserve `BLOCKED` for missing hardware/resources, unavailable external dependencies that remain unrecoverable after a narrow fix attempt, roadmap conflict, or user decision required.

**Goal:** Produce a fresh same-contract BRDT `40`-epoch Hybrid ResNet versus FFNO artifact with per-epoch training history, `ReduceLROnPlateau` provenance, a convergence audit against the frozen `20`-epoch rows, and a sample-`255` visual/gate package that either promotes BRDT into additive paper evidence or explicitly preserves it as decision-support only.

**Architecture:** Reuse the locked BRDT candidate-lane contract and extend only the task-local `scripts/studies/born_rytov_dt/` training, runner, comparison, and reporting surfaces. Keep the work split into three units: training-contract instrumentation, an immutable `40`-epoch paper-evidence runner/bundle, and promotion/index surfaces. The fresh `40`-epoch artifact must stand on its own provenance and must not relabel or overwrite the earlier `20`-epoch decision-support roots.

**Tech Stack:** PATH `python`, PyTorch, HDF5, NumPy, JSON/CSV/Markdown, tmux in `ptycho311`, existing BRDT task-local data/model/eval/reporting surfaces, repo-local NeurIPS evidence indexes.

---

## Selected Backlog Objective

- Implement backlog item `2026-05-05-brdt-supervised-born-40ep-paper-evidence`.
- Rerun exactly two BRDT neural rows under the locked candidate-lane contract:
  - `hybrid_resnet` / paper label `SRU-Net` or `Hybrid ResNet`, whichever the row contract already fixes;
  - `ffno`.
- Keep the dataset, operator, input mode, split counts, train-only normalization, metric family, and fixed-sample lineage from the completed BRDT preflight plus FFNO extension.
- Change only the training-depth and provenance surface needed to evaluate possible paper promotion:
  - `40` epochs;
  - Adam with initial `lr=2e-4`;
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`;
  - batch size `16`;
  - seed `42`;
  - per-epoch history records and convergence audit.
- Emit a paper-facing sample-`255` visual bundle with shared scales and source arrays comparing:
  - ground truth;
  - the locked physics-based Born inverse / model-based classical prediction;
  - SRU-Net / Hybrid ResNet;
  - FFNO.

## Scope

- Reuse the locked BRDT candidate-lane contract:
  - dataset id `brdt128_decision_support_preflight`;
  - split `2048 / 256 / 256`;
  - operator `BornRytovForward2D` in Born mode with `N=D=128`, `A=64`, `wavelength_px=8.0`, `medium_ri=1.333`, `normalize=unitary_fft`;
  - input mode `born_init_image`;
  - physical `q` target and train-only normalization;
  - fixed-sample lineage seeded from the prior bundle, with sample `255` required in the paper-facing visual set.
- Reuse exactly the working supervised-plus-Born objective:
  - `image=1.0`;
  - `physics=0.1`;
  - `relative_physics=0.1`;
  - `tv=1e-5`;
  - `positivity=1e-4`.
- Generate row-local `history.json` and `history.csv` with at least:
  - epoch;
  - train total loss;
  - averaged component losses;
  - current learning rate;
  - scheduler-step metric;
  - LR-reduction events or equivalent scheduler state;
  - any validation-side loss only if the implementation adds it without changing the locked claim boundary.
- Generate a convergence audit comparing the new `40`-epoch Hybrid ResNet and FFNO rows against the frozen `20`-epoch rows under the same contract.
- Evaluate paper promotion only through a fresh `paper_evidence_gate.json` plus checked-in summary/index updates.

## Explicit Non-Goals

- Do not rerun `unet`, `fno_vanilla`, or any other BRDT row besides `hybrid_resnet` and `ffno`.
- Do not change the dataset contract, operator geometry, normalization rule, loss weights, split counts, or input mode.
- Do not add Rytov mode, direct-sinogram input, limited-angle data, natural-image phantoms, or multi-seed robustness.
- Do not run a new classical-training study; the physics-based Born inverse visual must reuse or deterministically regenerate the existing same-contract classical/model-based prediction only.
- Do not overwrite or relabel the existing `2026-04-29` preflight root or the `2026-05-04` FFNO extension root.
- Do not treat BRDT as a replacement for the required CDI `lines128` or PDEBench CNS pillars.
- Do not populate `/home/ollie/Documents/neurips/`; Phase 5 paper-facing artifact assembly remains paused.

## Steering, Roadmap, And Prerequisite Constraints

- Steering and roadmap keep BRDT in `candidate-brdt-preflight`: it is additive candidate work only, not a required pillar.
- The selected backlog item is authorized to attempt BRDT paper-evidence promotion only through a fresh `40`-epoch artifact plus an explicit evidence-package amendment or equivalent checked-in update. The old `20`-epoch rows may not be promoted by label change.
- Progress-ledger prerequisites that matter here are already complete:
  - `2026-04-29-brdt-four-row-preflight`;
  - `2026-05-04-brdt-ffno-row-extension`.
- The current durable BRDT authorities are still decision-support:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md` says `defer_after_preflight`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md` is `decision_support_append_only`.
- Fairness and apples-to-apples rules remain binding:
  - no silent protocol drift;
  - no mixing of mismatched rows or sample sets;
  - no promotion unless the fresh artifact proves the exact contract used for the comparison.
- Known BRDT lineage inconsistency must be handled explicitly, not silently:
  - `brdt_preflight_summary.md` narrates the classical/model-based Born row as blocked;
  - `brdt_ffno_row_extension_summary.md` says the baseline JSON records a completed model-based Born inverse row.
  - This item must choose and document one same-contract authority for the sample-`255` physics-based visual. If that provenance cannot be made explicit, the paper-evidence gate must fail.

## Implementation Architecture

### Unit A: Training Contract And Row Provenance

- Owns the scheduler/history fields, row-local training history, and per-row runtime provenance.
- Primary surfaces:
  - `scripts/studies/born_rytov_dt/run_config.py`
  - `scripts/studies/born_rytov_dt/run_preflight.py`
  - `scripts/studies/born_rytov_dt/train.py`
  - `scripts/studies/born_rytov_dt/reporting.py`

### Unit B: Immutable 40-Epoch Runner And Artifact Bundle

- Owns the new `40`-epoch entrypoint, immutable output root, baseline-lineage checks, fixed-sample handling, and sample-`255` visual/source-array bundle.
- Primary surfaces:
  - `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py` (new)
  - `scripts/studies/born_rytov_dt/extension_bundle.py`
  - `scripts/studies/born_rytov_dt/preflight_visuals.py`
  - `scripts/studies/born_rytov_dt/refresh_model_based_classical_row.py`

### Unit C: Convergence Audit, Promotion Gate, And Evidence Indexing

- Owns `20`→`40` metric/history deltas, promotion/fallback decisions, durable summary output, and evidence-surface updates keyed to the final gate result.
- Primary surfaces:
  - `scripts/studies/born_rytov_dt/convergence.py` (new) or `comparison.py` if extension is cleaner
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md` (new)
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/index.md` only if the new durable summary should be discoverable there

## File And Artifact Targets

### Likely Code And Test Changes

- Modify `scripts/studies/born_rytov_dt/run_config.py`
- Modify `scripts/studies/born_rytov_dt/run_preflight.py`
- Modify `scripts/studies/born_rytov_dt/train.py` only if shared training helpers are cleaner than duplicating loop logic
- Modify `scripts/studies/born_rytov_dt/reporting.py`
- Modify `scripts/studies/born_rytov_dt/extension_bundle.py`
- Modify `scripts/studies/born_rytov_dt/preflight_visuals.py` if needed for the sample-`255` compare/error layout
- Modify `scripts/studies/born_rytov_dt/refresh_model_based_classical_row.py` only if deterministic same-contract classical visual regeneration is required
- Create `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`
- Create `scripts/studies/born_rytov_dt/convergence.py` unless `comparison.py` cleanly owns the new audit
- Modify `tests/studies/test_born_rytov_dt_preflight.py`
- Modify `tests/studies/test_born_rytov_dt_adapters.py` only if adapter/runtime schema coverage changes

### Mandatory Contract Outputs

- Fresh artifact root:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
- Top-level run/provenance artifacts:
  - `invocation.json`
  - `invocation.sh`
  - `preflight_manifest.json` or equivalent bundle manifest naming this backlog item, the exact training contract, baseline lineage, pre-gate claim boundary, promotion status, and artifact pointers
  - `runtime_provenance.json` or an equivalent structured payload with:
    - repo git SHA and dirty-state note;
    - PATH `python` executable path and Python version;
    - PyTorch version;
    - CUDA runtime/version details;
    - GPU identity/count;
    - host provenance;
    - launch timestamps and tracked PID.
  - `dataset_identity_manifest.json` or equivalent structured payload with dataset id/source, reviewed dataset path, and checksum or reviewed checksum exception with size/mtime/source rationale
  - `split_manifest.json` recording the locked `2048 / 256 / 256` split and fixed-sample lineage
  - `run_exit_status.json` or equivalent exit-code proof tying the tracked PID to the final exit code and the retained run log path
  - captured run log under `logs/`
- Per-row artifacts:
  - `rows/hybrid_resnet/history.json`
  - `rows/hybrid_resnet/history.csv`
  - `rows/ffno/history.json`
  - `rows/ffno/history.csv`
  - `rows/*/row_summary.json`
  - `rows/*/model_profile.json` or equivalent row-level payload with architecture/profile id and parameter count
  - `rows/*/model_state.pt`
- Bundle metrics/audit artifacts:
  - `metrics.json`
  - `metrics.csv`
  - `combined_metrics.json`
  - `combined_metrics.csv`
  - `metric_schema.json`
  - `convergence_audit.json`
  - `convergence_audit.csv`
  - `paper_evidence_gate.json`
- Visual/source-array artifacts:
  - `visual_manifest.json`
  - sample-`255` compare and error panels under `visuals/`
  - source arrays under `figures/source_arrays/`

### Preferred Packaging And Discoverability Outputs

- Required new durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md`
- Regardless of promotion outcome, publish the fresh durable result through the package-level and non-manuscript evidence surfaces required by the selected item and roadmap:
  - update `paper_evidence_manifest.json` with the BRDT authority, lane status, claim boundary, artifact root, and any blocked/promotion-failure reasons;
  - update `paper_evidence_index.md` with a row whose tier/outcome matches the gate result;
  - update `docs/index.md` so the new summary is discoverable from the repo-wide documentation hub;
  - `evidence_matrix.md`
  - `model_variant_index.json`
  - `ablation_index.json`
- If the paper-evidence gate passes:
  - update `paper_evidence_package_design.md` or add a checked-in amendment section naming BRDT as additive evidence with exact rows, artifacts, claim boundary, and additive-only caveat;
  - optionally materialize repo-local table/figure assets under `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/` and `figures/` if those are the approved paper-local surfaces for this lane.

## Task 1: Lock The 40-Epoch Contract And Add Regression Tests

**Files:**

- Modify: `tests/studies/test_born_rytov_dt_preflight.py`
- Modify: `tests/studies/test_born_rytov_dt_adapters.py` only if row-schema or FFNO surface coverage changes
- Read/confirm: the authoritative inputs listed in the backlog item check command

- [ ] Run the backlog item’s required input-existence check before code changes so implementation starts from the expected BRDT authorities and artifact roots.
- [ ] Add failing tests that pin the new training contract fields:
  - `epochs=40`;
  - Adam `lr=2e-4`;
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`;
  - batch size `16`;
  - seed `42`.
- [ ] Add failing tests that require per-row `history.json` and `history.csv` with `40` epoch records and the expected keys.
- [ ] Add failing tests that require the new runner to:
  - accept only the locked baseline lineage inputs;
  - refuse baseline contract drift;
  - refuse output-root overlap or overwrite;
  - preserve or explicitly require sample `255`.
- [ ] Add failing tests that require a `paper_evidence_gate.json` payload distinguishing:
  - `claim_boundary` values `paper_evidence_brdt_additive` and `decision_support_convergence_followup`;
  - promotion-status outcomes `passed`, `failed`, and row-level or lane-level `blocked`;
  - the required provenance-artifact keys needed for package-level paper-evidence review.

**Verification**

- [ ] **Blocking:** run the required input-existence command from the backlog item verbatim.
- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k 'history or scheduler or 40ep or paper_evidence or convergence'`
- [ ] **Supporting:** `pytest -q tests/studies/test_born_rytov_dt_adapters.py -k 'ffno or hybrid or row_schema'`

## Task 2: Implement Scheduler-Provenance And Per-Epoch History

**Files:**

- Modify: `scripts/studies/born_rytov_dt/run_config.py`
- Modify: `scripts/studies/born_rytov_dt/run_preflight.py`
- Modify: `scripts/studies/born_rytov_dt/train.py` only if the shared training path should carry the same history schema
- Modify: `scripts/studies/born_rytov_dt/reporting.py`

- [ ] Extend the BRDT training contract schema to carry the plateau scheduler fields without breaking the existing `20`-epoch authorities.
- [ ] Update `_train_neural_row` in `run_preflight.py` to aggregate per-epoch averages for total loss and each component loss rather than keeping only the last batch breakdown.
- [ ] Step `ReduceLROnPlateau` once per epoch on a declared scheduler metric. Use training loss unless a validation-side metric is added in the same implementation without changing the claim boundary.
- [ ] Record current LR after scheduler stepping and capture LR-reduction events or equivalent state needed for the later convergence audit.
- [ ] Write row-local `history.json` and `history.csv` and surface their paths in row summaries/runtime metadata.
- [ ] Keep all operator and normalization semantics unchanged:
  - physical `q` remains the metric target;
  - unnormalize model outputs before the physics loss and operator calls.

**Verification**

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k 'history or scheduler or plateau'`
- [ ] **Supporting:** `python -m compileall -q scripts/studies/born_rytov_dt`

## Task 3: Add The Immutable 40-Epoch Paper-Evidence Runner

**Files:**

- Create: `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`
- Modify: `scripts/studies/born_rytov_dt/extension_bundle.py`
- Modify: `scripts/studies/born_rytov_dt/preflight_visuals.py` if a new sample-`255` compare/error layout is needed
- Modify: `scripts/studies/born_rytov_dt/refresh_model_based_classical_row.py` only if deterministic regeneration of the physics-based Born inverse visual is required

- [ ] Build a narrow runner that launches exactly `hybrid_resnet` and `ffno` under the locked BRDT contract and writes a fresh immutable artifact root for this backlog item only.
- [ ] Make the runner require lineage inputs from:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/`
- [ ] Make the runner refuse:
  - output-root reuse when another active writer is already targeting it;
  - overwrite of the baseline `20`-epoch roots;
  - fixed-sample drift that drops sample `255`.
- [ ] Write top-level provenance that includes:
  - backlog item id;
  - pre-gate claim boundary and a separate promotion-status field;
  - dataset/operator pointers;
  - exact training contract;
  - baseline lineage pointers to the `20`-epoch Hybrid and FFNO rows;
  - row roster and fixed-sample ids;
  - repo git SHA plus dirty-state note;
  - Python, PyTorch, CUDA, GPU, and host provenance;
  - dataset identity plus checksum or reviewed checksum exception;
  - split manifest pointer;
  - tracked PID and final exit-code proof path.
- [ ] Reuse or deterministically regenerate the physics-based Born inverse assets for sample `255` under the locked operator contract. If the classical/model-based provenance inconsistency cannot be resolved to one same-contract authority, record that in the gate payload and fail promotion.
- [ ] Emit a sample-`255` visual/source-array bundle with shared scales for ground truth, physics-based Born inverse, Hybrid ResNet, and FFNO.

**Intended long-run entrypoint**

```bash
python scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py \
  --baseline-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight \
  --ffno-extension-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence \
  --rows hybrid_resnet ffno \
  --epochs 40 \
  --batch-size 16 \
  --learning-rate 2e-4 \
  --scheduler reduce_on_plateau \
  --plateau-factor 0.5 \
  --plateau-patience 2 \
  --plateau-threshold 0.0 \
  --plateau-min-lr 1e-5 \
  --seed 42 \
  --fixed-sample-ids 145 83 255 126 \
  --required-paper-sample 255 \
  --device cuda
```

**Verification**

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k '40ep or baseline_lineage or fixed_sample or paper_visual'`
- [ ] **Supporting:** `python -m compileall -q scripts/studies/born_rytov_dt`

## Task 4: Implement The Convergence Audit And Promotion Gate

**Files:**

- Create: `scripts/studies/born_rytov_dt/convergence.py` unless `comparison.py` is the cleaner owner
- Modify: `scripts/studies/born_rytov_dt/comparison.py` only if reuse is simpler than a new module
- Modify: `scripts/studies/born_rytov_dt/reporting.py` if shared gate/summary serializers belong there

- [ ] Compare the fresh `40`-epoch Hybrid ResNet and FFNO rows against their frozen `20`-epoch lineage rows under the same metric schema.
- [ ] Compute and write `20`→`40` deltas for:
  - image-space physical-`q` metrics;
  - measurement-space metrics;
  - PSNR and SSIM;
  - parameter count and runtime.
- [ ] Compute convergence diagnostics from the new histories:
  - last-`5` and last-`10` epoch loss movement when available;
  - final LR;
  - number of LR reductions;
  - materially-improving-at-stop flag per row.
- [ ] Emit `convergence_audit.json` and `convergence_audit.csv`.
- [ ] Emit `paper_evidence_gate.json` with a conservative decision rule. Promotion must require at least:
  - both rows completed successfully;
  - both rows have `40` history records;
  - scheduler settings match the planned contract;
  - the artifact root has complete provenance and sample-`255` visual/source arrays, specifically repo git SHA/dirty state, Python/PyTorch/CUDA/GPU/host provenance, dataset identity/checksum or reviewed exception, split manifest, model profile/parameter count, run log, and exit-code proof;
  - the comparison remains same-contract with the frozen `20`-epoch lineage;
  - the checked-in evidence amendment is prepared and consistent with the gate result.
- [ ] If any non-blocking promotion condition fails, keep the output claim boundary at `decision_support_convergence_followup`, set promotion status to `failed`, record exact failed-gate reasons, and do not promote BRDT into the claim-bearing paper package.
- [ ] Use status `blocked` only for unrecoverable row-level or lane-level blockers that remain after the required narrow-fix attempt; record that status separately from the claim-boundary field.

**Verification**

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_preflight.py -k 'convergence or paper_evidence_gate'`
- [ ] **Supporting:** validate generated JSON payloads with `python -m json.tool` or an equivalent narrow fixture check

## Task 5: Run The 40-Epoch Rows And Validate The Artifact Root

**Files:**

- No new code targets beyond the Task 3 runner; this tranche executes the approved runner and validates the produced artifact root

- [ ] Before the expensive run, execute the backlog item’s required deterministic checks:
  - the authoritative-input existence command;
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`;
  - `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`.
- [ ] Launch the `40`-epoch runner in `tmux` from the `ptycho311` environment, track the exact PID, and keep ownership of the run until terminal success or a documented recoverable-failure retry is complete.
- [ ] Do not launch a duplicate run if another process is already writing to the same `--output-root`.
- [ ] On non-zero exit or missing/stale artifacts, diagnose the narrow cause, fix it, and rerun. Do not mark the item `BLOCKED` unless the remaining failure is an unrecoverable resource/dependency/authority issue.
- [ ] After a successful run, validate that the fresh artifact root contains the mandatory outputs listed in this plan and that the sample-`255` visual/source-array artifacts are present and fresh.

**Verification**

- [ ] **Blocking:** rerun the backlog item’s exact required deterministic checks before declaring execution complete:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_ffno_row_extension_summary.md"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/preflight_manifest.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json"),
    Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/combined_metrics.json"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing BRDT 40-epoch paper-evidence inputs: {missing}")
print("brdt 40-epoch paper-evidence inputs present")
PY
pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py
python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
```

- [ ] **Blocking:** run a narrow artifact-root existence check that confirms:
  - `history.json`/`history.csv` exist for both rows;
  - `convergence_audit.json` and `paper_evidence_gate.json` exist;
  - the sample-`255` visual/source-array artifacts exist;
  - the top-level manifest records this backlog item id, `40` epochs, a separate claim-boundary field, and a separate promotion-status field;
  - the artifact root carries repo/runtime provenance, dataset identity, split manifest, model-profile, run-log, and exit-code-proof payloads required by the paper-evidence package design.

## Task 6: Write The Durable Summary And Update Evidence Surfaces

**Files:**

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify conditionally: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- Modify: `docs/index.md`

- [ ] Write the durable summary as the new authority for this backlog item. It must include:
  - exact locked contract;
  - artifact root;
  - `20`→`40` metric deltas and convergence read;
  - sample-`255` visual provenance decision;
  - final claim boundary (`paper_evidence_brdt_additive` or `decision_support_convergence_followup`);
  - separate final lane status (`decision_support` or `blocked`) when promotion did not pass;
  - exact reasons for promotion success or failure.
- [ ] Update `evidence_matrix.md`, `model_variant_index.json`, and `ablation_index.json` so the fresh `40`-epoch rows and audit outputs remain discoverable even if promotion fails. These surfaces must preserve the final gate result and must not silently relabel the lane as paper-grade.
- [ ] Update `paper_evidence_manifest.json` with the new BRDT authority and status regardless of promotion outcome. The manifest entry must preserve the gate result, claim boundary, and any blocked/promotion-failure reasons.
- [ ] Update `paper_evidence_index.md` with a row whose tier/outcome matches the final gate result regardless of promotion outcome.
- [ ] Update `paper_evidence_package_design.md` only if `paper_evidence_gate.json` passes. The amendment must name:
  - exact rows;
  - exact artifact root;
  - the sample-`255` visual bundle;
  - the additive claim boundary;
  - any caveat that BRDT remains additive and does not replace CDI/CNS.
- [ ] Update `docs/index.md` to point to the new summary authority so the BRDT 40-epoch outcome remains discoverable from the repo-wide documentation hub.

**Verification**

- [ ] **Blocking:** run a narrow docs/index-surface integrity check that confirms every updated summary/index points to the same backlog item id, artifact root, and claim boundary.
- [ ] **Supporting:** re-run `python -m json.tool` on each edited JSON index or manifest file.

## Final Completion Criteria

- [ ] Fresh immutable artifact root exists for `2026-05-05-brdt-supervised-born-40ep-paper-evidence`.
- [ ] Only `hybrid_resnet` and `ffno` were rerun.
- [ ] Both rerun rows use the locked BRDT contract and the exact `40`-epoch scheduler/training recipe.
- [ ] Both rows emit `history.json` and `history.csv` with `40` per-epoch records.
- [ ] Sample `255` is present in the fixed-sample visual/source-array bundle.
- [ ] `convergence_audit.json` and `paper_evidence_gate.json` are present and consistent with the durable summary.
- [ ] The artifact root includes the paper-evidence provenance payloads required by this plan: repo/runtime provenance, dataset identity/checksum or reviewed exception, split manifest, model profile/parameter count, run log, and exit-code proof.
- [ ] `paper_evidence_manifest.json`, `paper_evidence_index.md`, and `docs/index.md` are updated and consistent with the final gate result.
- [ ] If paper promotion succeeds, the evidence-package amendment is checked in and consistent with the gate.
- [ ] If paper promotion fails, the durable summary and discoverability indexes preserve the non-promoted or blocked outcome explicitly and do not claim BRDT as paper-grade.
- [ ] The backlog item’s required deterministic checks have been run and passed after implementation.
