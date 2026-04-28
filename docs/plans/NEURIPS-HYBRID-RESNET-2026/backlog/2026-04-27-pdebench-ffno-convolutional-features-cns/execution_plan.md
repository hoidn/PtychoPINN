# PDEBench CNS FFNO Convolutional Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an authoritative capped `2d_cfd_cns` FFNO-family comparison that answers whether adding an explicit local convolutional branch inside the FFNO-close bottleneck materially improves the fixed local CNS contract against authored FFNO, local FFNO-close, and Hybrid-spectral anchors.

**Architecture:** Treat this backlog item as a bounded Phase 2 CNS extension lane, not a new benchmark tranche. First inspect and freeze the repo and artifact baseline because the bounded `ffno_bottleneck_localconv_base` implementation appears to exist already. Patch only if inspection disproves that baseline. Then produce or salvage authoritative `10`-epoch and `40`-epoch local-conv rows in `pilot` mode, backfill the missing same-contract `40`-epoch `ffno_bottleneck_base` anchor only if needed, collate cross-run compares, and publish durable repo-local interpretation without widening into full-training or paper-facing claims.

**Tech Stack:** PATH `python`; long runs in tmux with `ptycho311`; PyTorch/Lightning; `ptycho_torch/generators/`; `scripts/studies/pdebench_image128/`; pytest; compileall; Markdown/JSON/CSV/PNG artifacts under `.artifacts/`.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-27-pdebench-ffno-convolutional-features-cns`
- Selection mode: `RECOVERED_IN_PROGRESS`
- Status: fresh execution authority
- Date: `2026-04-28`
- Selected-item authority: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/19/items/2026-04-27-pdebench-ffno-convolutional-features-cns/selected-item-context.md`
- Plan-path authority: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/19/items/2026-04-27-pdebench-ffno-convolutional-features-cns/plan-phase/plan_path.txt`
- Previous-plan background: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-pdebench-ffno-convolutional-features-cns/execution_plan.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
- CNS summary sync target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/`

This document supersedes earlier revisions at this path and is the new execution authority for this backlog item. Implementation should rely on this plan plus the approved NeurIPS design, not on older queue notes or raw backlog prose.

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/steering.md`
- `docs/model_baselines.md`
- `docs/backlog/in_progress/2026-04-27-pdebench-ffno-convolutional-features-cns.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_plan.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/19/items/2026-04-27-pdebench-ffno-convolutional-features-cns/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/19/items/2026-04-27-pdebench-ffno-convolutional-features-cns/plan-phase/plan_path.txt`
- Previous-plan background at this same plan path

## Selected Objective

- Answer one narrow question on the fixed capped CNS lane: does adding explicit local convolutional features inside the FFNO-close bottleneck help enough to justify the extra machinery?
- Keep the compare on the existing local `2d_cfd_cns` contract and report both aggregate and high-frequency behavior, with `relative_l2` and `fRMSE_high` called out explicitly.
- Compare the new row against the required same-contract references:
  - `author_ffno_cns_base`
  - `ffno_bottleneck_base`
  - `spectral_resnet_bottleneck_base`
- Close the only known fairness gap before the `40`-epoch comparison: if no authoritative same-contract `40`-epoch `ffno_bottleneck_base` root exists, backfill it inside this item before interpreting the new `40`-epoch row.

## Scope

- Task remains **CNS only**:
  - task: `2d_cfd_cns`
  - dataset: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - resolution: `128x128`
- Fixed capped contract for all new or salvaged comparison rows in this item:
  - split caps: `512 / 64 / 64` trajectories
  - `max_windows_per_trajectory=8`
  - `history_len=2`
  - batch size `4`
  - training loss `mse`
  - metrics: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Execution sequence:
  - inspect the repo and this item’s artifact root
  - verify or minimally repair the bounded local-conv FFNO profile if inspection finds a real gap
  - freeze the `10`-epoch reference manifest and produce or salvage the authoritative `10`-epoch local-conv `pilot` run
  - freeze the `40`-epoch reference manifest, backfill `ffno_bottleneck_base` `40` epochs in `pilot` mode if still missing, then produce or salvage the authoritative `40`-epoch local-conv `pilot` run
  - collate cross-run compare sidecars for `10` and `40` epochs
  - publish durable summary, CNS summary sync, and progress-ledger completion state
- Architecture boundary for this item:
  - use exactly one new profile ID: `ffno_bottleneck_localconv_base`
  - keep the same shell as `ffno_bottleneck_base`
  - keep the same downsampling depth, hidden width, block count, skip style, and output head
  - change only the bottleneck internals by adding an explicit local `3x3` residual branch per FFNO block

## Explicit Non-Goals

- Do not widen this item into CDI, ptychography, SWE, Darcy, OpenFWI, or `/home/ollie/Documents/neurips/` artifact assembly.
- Do not treat this as a full-training benchmark tranche or a paper-facing competitiveness result.
- Do not expand into a broad FFNO sweep. This plan does **not** include:
  - convolutional stems
  - decoder-side refinement heads
  - multiple local-feature variants
  - kernel-size, gating, or norm sweeps
- Do not reopen separate CNS lanes:
  - `history_len=1`
  - spectral modes `32/32`
  - Hybrid-spectral sharing/depth ablations
  - GNOT
  - physics regularization
- Do not change split, normalization, epoch budgets, batch size, training loss, or metric family to make the new row look better.
- Do not silently relax equal-footing constraints or replace missing reference rows with mismatched epochs.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, and Fairness Constraints

- Preserve the approved roadmap phase order and evidence boundaries. This item strengthens Phase 2 CNS interpretation only and does not satisfy the roadmap’s full-training gate.
- Steering is binding: keep equal-footing comparisons explicit, keep the local CNS contract fixed, and avoid optional scope drift while required evidence is missing.
- The backlog item’s reviewer notes are binding:
  - this is an FFNO-family extension, not a Hybrid-spectral ablation
  - do not change split, loss, normalization, epoch budget, or metrics
  - report both aggregate `relative_l2` and high-frequency behavior
- The design, roadmap, and suite plan require capped outputs to remain decision-support only. No artifact from this item may be relabeled as benchmark-complete or paper-grade competitiveness evidence.
- Runs created inside this item must use PDEBench CNS `pilot` mode, not `readiness`, so newly generated `comparison_summary.json` files report `capped_decision_support_only` instead of smoke-only readiness evidence.
- Inherited reference roots may still carry earlier readiness provenance. Reuse them only as explicit historical anchors inside the compare manifests and summary text. If execution requires same-mode pilot reruns for those anchors too, record that broader rerun requirement as a blocker instead of silently widening scope.
- The CNS suite contract from the roadmap and CNS summary remains fixed:
  - dataset file, field order, `history_len=2`, normalization path, and denormalized metric family stay unchanged
  - the canonical Hybrid CNS shell remains `hybrid_resnet_cns`
- Long-running commands must run in tmux with `ptycho311` active, track the exact launched PID, and count as complete only when the tracked PID exits `0` and required artifacts are freshly written.
- Follow `REPORTING-ARTIFACT-BOUNDARY-001`: optional galleries may warn, but core success is decided by launcher exit status plus required manifests and metrics.
- Follow PYTHON-ENV-001: use PATH `python`.

## Prerequisite Status

### Satisfied

- The selected-item context and `docs/backlog/index.md` both mark this lane as having no active backlog prerequisite.
- The roadmap and backlog index both place this lane inside the active Phase 2 CNS ablation queue, not behind the already completed author-FFNO or GNOT external-baseline lanes.
- The official CNS file is already staged and verified in the current suite contract.
- The supervised CNS runner, normalization path, metric path, and compare helper already exist.
- Required reference rows are already available for `10` epochs:
  - `author_ffno_cns_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-20260422T232119Z`
  - `ffno_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
  - `spectral_resnet_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- Required reference rows are already available for `40` epochs for:
  - `author_ffno_cns_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
  - `spectral_resnet_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- Existing repo state and prior plan background both indicate the bounded local-conv profile already has code-level support:
  - generator support in `ptycho_torch/generators/ffno_bottleneck.py`
  - profile registration in `scripts/studies/pdebench_image128/run_config.py`
  - model builder wiring in `scripts/studies/pdebench_image128/models.py`
  - focused tests in `tests/torch/test_ffno_bottleneck.py` and `tests/studies/test_pdebench_image128_models.py`

### Open But Not Blocking

- No authoritative local-conv run roots are recorded yet in the progress ledger for `10` or `40` epochs. The artifact root may contain partial or salvageable outputs and must be inspected before rerunning.
- No authoritative same-contract `40`-epoch `ffno_bottleneck_base` run root is recorded in the selected-item context. This remains the only known fairness prerequisite for the `40`-epoch compare.
- No durable summary, CNS summary sync, or progress-ledger completion entry exists yet for this backlog item.

## Implementation Architecture

- **Inspection / Bounded Implementation Unit:** `ptycho_torch/generators/ffno_bottleneck.py`, `scripts/studies/pdebench_image128/models.py`, `scripts/studies/pdebench_image128/run_config.py`, `tests/torch/test_ffno_bottleneck.py`, and `tests/studies/test_pdebench_image128_models.py` own the local-conv profile contract. Implementation begins by verifying these surfaces and only patching the smallest missing or inconsistent piece if inspection disproves the current baseline.
- **Execution / Compare Unit:** `scripts/studies/run_pdebench_image128_suite.py`, `scripts/studies/pdebench_image128/reporting.py`, and `tests/studies/test_pdebench_image128_runner.py` own run orchestration, manifest collation, and cross-run compare output. This unit must preserve the fixed CNS contract and reuse the compare pattern already used by the authored-FFNO and modes-32 lanes.
- **Durable Docs / State Unit:** `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`, `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`, `docs/studies/index.md`, `docs/index.md`, `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`, and optionally `docs/findings.md` own the durable interpretation and completion record.

## Concrete File and Artifact Targets

### Repo Surfaces Likely To Change

- Inspect first, patch only if needed:
  - `ptycho_torch/generators/ffno_bottleneck.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `tests/torch/test_ffno_bottleneck.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py`
- Only if compare collation exposes a real blocker:
  - `scripts/studies/pdebench_image128/reporting.py`
- Expected durable updates at close:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `docs/studies/index.md`
  - `docs/index.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - `docs/findings.md` only if the completed lane produces a reusable engineering rule rather than a one-off experiment outcome

### Required Generated Artifacts

- Study root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/`
- Required inspection artifact:
  - `inspect-<timestamp>/` with a concise inspection note or JSON manifest that records whether code support was already present, which existing run roots were found, and which reruns remain necessary
- Required run roots:
  - `cns-ffno-localconv-10ep-<timestamp>/`
  - `cns-ffno-close-backfill-40ep-<timestamp>/` only if no valid same-contract `40`-epoch `ffno_bottleneck_base` root is discovered
  - `cns-ffno-localconv-40ep-<timestamp>/`
- Required manifests and compares:
  - `reference_runs_10ep.json`
  - `reference_runs_40ep.json`
  - `compare_10ep_against_existing.json`
  - `compare_10ep_against_existing.csv`
  - `compare_40ep_against_existing.json`
  - `compare_40ep_against_existing.csv`
- Optional but expected when target alignment succeeds:
  - `compare_10ep_sample0.png`
  - `compare_10ep_sample0_error.png`
  - `compare_40ep_sample0.png`
  - `compare_40ep_sample0_error.png`

## Fixed Reference Rows

### `10`-Epoch Required References

- `author_ffno_cns_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-20260422T232119Z`
- `ffno_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `spectral_resnet_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`

### `10`-Epoch Optional Continuity Rows

- `hybrid_resnet_cns`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- `fno_base`, `unet_strong`, and optionally `hybrid_resnet_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`

### `40`-Epoch Required References

- `author_ffno_cns_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- `spectral_resnet_bottleneck_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- `ffno_bottleneck_base`: a same-contract `40`-epoch row discovered during inspection or backfilled by this item

### `40`-Epoch Optional Continuity Rows

- `fno_base`, `unet_strong`, and optionally `hybrid_resnet_base`: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`

## Required Deterministic Checks

These checks come from the selected-item context and are mandatory execution evidence for this backlog item. Keep them even if stronger targeted checks are added.

- `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
- `python -m compileall -q scripts/studies/pdebench_image128`

Add this stronger targeted unit check whenever generator or profile surfaces are patched, or when inspection exposes ambiguity that needs explicit revalidation:

- `pytest -q tests/torch/test_ffno_bottleneck.py`

## Task 1: Inspect And Freeze The Local-Conv Baseline

**Purpose:** Determine whether this lane still needs code changes or whether the remaining work is execution and documentation only.

**Files / Artifacts:**

- Inspect:
  - `ptycho_torch/generators/ffno_bottleneck.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `tests/torch/test_ffno_bottleneck.py`
  - `tests/studies/test_pdebench_image128_models.py`
  - `tests/studies/test_pdebench_image128_runner.py`
- Generate:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/inspect-<timestamp>/`

- [ ] Inspect the current tree and confirm that `ffno_bottleneck_localconv_base` still differs from `ffno_bottleneck_base` only by the explicit local-conv provenance fields and the internal `3x3` branch.
- [ ] Inspect this item’s artifact root for partial local-conv or backfill runs. Reuse only roots whose `invocation.json`, metrics, profile config, and fixed CNS contract all match this plan.
- [ ] Reject any salvaged authoritative local-conv or backfill candidate whose `comparison_summary.json` still reports `mode: "readiness"` or `evidence_scope: "smoke_feasibility_only"`. Those roots may be cited as historical context, but they do not satisfy this plan’s newly generated evidence contract.
- [ ] If inspection finds a real code gap, patch only the smallest missing surface. Do not broaden the model family or runner contract.
- [ ] Run the required deterministic checks:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128`
- [ ] If generator or profile code changed, or inspection exposed ambiguity, also run `pytest -q tests/torch/test_ffno_bottleneck.py`.
- [ ] Write a concise inspection artifact that records:
  - whether local-conv code support was already present
  - whether authoritative `10`-epoch or `40`-epoch local-conv runs already exist
  - whether the `40`-epoch `ffno_bottleneck_base` fairness anchor still needs a backfill
  - whether any inherited reference roots remain readiness-provenance-only and, if so, whether that is accepted as an explicit historical-anchor caveat or elevated to a blocker

**Verification:**

- Required deterministic checks pass.
- The inspection artifact clearly states whether execution can proceed without further code edits.
- Any salvaged authoritative run root for this item is confirmed as `pilot` / `capped_decision_support_only`, or it is rejected and scheduled for rerun.
- If code was patched, the profile and build tests prove the local-conv row still preserves the canonical CNS shell and explicit provenance markers.

## Task 2: Produce The Authoritative `10`-Epoch Local-Conv Row

**Purpose:** Create or salvage the first authoritative same-contract local-conv result and compare it fairly against the fixed `10`-epoch anchors.

**Files / Artifacts:**

- Generate or reuse:
  - `reference_runs_10ep.json`
  - `cns-ffno-localconv-10ep-<timestamp>/`
  - `compare_10ep_against_existing.json`
  - `compare_10ep_against_existing.csv`
  - optional `compare_10ep_sample0.png`
  - optional `compare_10ep_sample0_error.png`

- [ ] Freeze `reference_runs_10ep.json` with the exact fixed reference roots listed in this plan. Do not silently substitute prettier or newer roots.
- [ ] If no authoritative `10`-epoch local-conv run already exists, launch it in tmux under `ptycho311` with the fixed contract in `pilot` mode:
  - `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-10ep-<timestamp> --profiles ffno_bottleneck_localconv_base --history-len 2 --epochs 10 --batch-size 4 --max-train-trajectories 512 --max-val-trajectories 64 --max-test-trajectories 64 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
- [ ] Inside the tmux shell, activate `ptycho311`, start the command once, capture the exact PID (`... & pid=$!; wait "$pid"`), and do not use `pgrep -f` polling loops or duplicate the same `--output-root`.
- [ ] Verify the run root contains at minimum:
  - `invocation.json`
  - `invocation.sh`
  - `comparison_summary.json`
  - `comparison_summary.csv`
  - `metrics_ffno_bottleneck_localconv_base.json`
  - `model_profile_ffno_bottleneck_localconv_base.json`
- [ ] Collate `compare_10ep_against_existing.json` and `.csv` against the required rows `author_ffno_cns_base`, `ffno_bottleneck_base`, and `spectral_resnet_bottleneck_base`. Keep optional continuity rows separate from the required fairness core.
- [ ] Validate the new local-conv `comparison_summary.json` before writing the cross-run compare: it must report `mode: "pilot"` and `evidence_scope: "capped_decision_support_only"`. If it does not, treat the run as non-authoritative and rerun or block.
- [ ] Render `compare_10ep_sample0.png` and `compare_10ep_sample0_error.png` only if target alignment succeeds. If galleries fail but core compare JSON and CSV are correct, record that as optional-reporting-only and continue.

**Verification:**

- The tracked tmux job exits `0`, or an authoritative matching run is explicitly reused.
- The authoritative local-conv `10`-epoch run reports `mode: "pilot"` and `evidence_scope: "capped_decision_support_only"`.
- `reference_runs_10ep.json`, `compare_10ep_against_existing.json`, and `compare_10ep_against_existing.csv` all parse successfully.
- The compare output explicitly reports `relative_l2` and `fRMSE_high` for the new row and the required references.

## Task 3: Close The `40`-Epoch Fairness Gap And Produce The Authoritative `40`-Epoch Local-Conv Row

**Purpose:** Ensure the `40`-epoch compare is genuinely equal-footing by supplying the missing local FFNO-close anchor if needed, then compare the local-conv row on the same contract.

**Files / Artifacts:**

- Generate or reuse:
  - `reference_runs_40ep.json`
  - `cns-ffno-close-backfill-40ep-<timestamp>/` only if needed
  - `cns-ffno-localconv-40ep-<timestamp>/`
  - `compare_40ep_against_existing.json`
  - `compare_40ep_against_existing.csv`
  - optional `compare_40ep_sample0.png`
  - optional `compare_40ep_sample0_error.png`

- [ ] Inspect for an existing authoritative same-contract `40`-epoch `ffno_bottleneck_base` root. Reuse it only if the invocation, profile, split caps, loss, metrics, artifact completeness, and `comparison_summary.json` mode and evidence fields all match this plan.
- [ ] If no valid root exists, backfill `ffno_bottleneck_base` `40` epochs under the fixed CNS contract in `pilot` mode:
  - `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-close-backfill-40ep-<timestamp> --profiles ffno_bottleneck_base --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 512 --max-val-trajectories 64 --max-test-trajectories 64 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
- [ ] Freeze `reference_runs_40ep.json` only after the `40`-epoch FFNO-close reference is resolved.
- [ ] If no authoritative `40`-epoch local-conv run already exists, launch it in tmux under the same contract in `pilot` mode:
  - `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-<timestamp> --profiles ffno_bottleneck_localconv_base --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 512 --max-val-trajectories 64 --max-test-trajectories 64 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
- [ ] For each long run, activate `ptycho311`, track the exact launched PID, wait on that PID, and verify completion from exit code plus required fresh artifacts.
- [ ] Verify the local-conv `40`-epoch run root contains the same minimum artifact set required in Task 2.
- [ ] Collate `compare_40ep_against_existing.json` and `.csv` with the required rows:
  - `author_ffno_cns_base`
  - `ffno_bottleneck_base`
  - `spectral_resnet_bottleneck_base`
- [ ] Validate the authoritative `40`-epoch local-conv run and any newly backfilled `ffno_bottleneck_base` root before writing the cross-run compare: each newly generated root must report `mode: "pilot"` and `evidence_scope: "capped_decision_support_only"`.
- [ ] Render optional `compare_40ep_sample0.png` and `compare_40ep_sample0_error.png` only if alignment succeeds.

**Verification:**

- A same-contract `40`-epoch `ffno_bottleneck_base` root is either reused with matching `pilot` evidence semantics or freshly backfilled in `pilot` mode.
- The authoritative local-conv `40`-epoch run reports `mode: "pilot"` and `evidence_scope: "capped_decision_support_only"`.
- `reference_runs_40ep.json`, `compare_40ep_against_existing.json`, and `compare_40ep_against_existing.csv` all parse successfully.
- The `40`-epoch compare does not mix epochs or silently drop the local FFNO-close anchor.

## Task 4: Publish Durable Interpretation And Queue-State Completion

**Purpose:** Convert the run outputs into durable repo-local knowledge and clear the lane from active execution state.

**Files / Artifacts:**

- Create or update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
  - `docs/studies/index.md`
  - `docs/index.md`
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
  - `docs/findings.md` only if a reusable implementation or workflow rule emerged

- [ ] Write the dedicated summary with:
  - the fixed capped CNS contract
  - the exact reference roots used
  - which roots were newly generated in `pilot` mode versus reused as inherited historical anchors
  - whether the `40`-epoch FFNO-close backfill was needed
  - the `10`-epoch and `40`-epoch local-conv results
  - explicit discussion of `relative_l2` and `fRMSE_high`
  - parameter-count or runtime overhead if it is material
  - a final interpretation about whether the local-conv addition is worth carrying forward
  - explicit evidence-boundary language that this remains capped decision-support evidence only
- [ ] Sync the CNS summary so the active CNS lane record includes this FFNO-family extension result.
- [ ] Update `docs/studies/index.md` and `docs/index.md` if the new summary becomes a durable discoverable entry.
- [ ] Add a completion entry to `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` capturing:
  - plan path
  - artifact root
  - authoritative run roots
  - compare artifact paths
  - evidence scope
  - metric interpretation
  - verification command results
  - any finding or blocker note that must persist
- [ ] Update `docs/findings.md` only if the lane produced a reusable engineering rule such as a compare-helper contract, provenance rule, or FFNO implementation invariant. Do not promote a model-ranking conclusion into `findings.md`.

**Verification:**

- The dedicated summary exists and cites both epoch slices plus the fixed evidence boundary.
- The CNS summary and progress ledger both point to the same authoritative artifacts.
- `progress_ledger.json` still parses after the update.
- Any index updates make the new summary discoverable from the standard docs entry points.

## Exit Criteria

- The local-conv profile is either verified as already correctly implemented or minimally repaired without widening scope.
- Required deterministic checks pass:
  - `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128`
  - plus `pytest -q tests/torch/test_ffno_bottleneck.py` when generator or profile surfaces were touched or revalidated after ambiguity
- An authoritative `10`-epoch local-conv run exists and is compared fairly against the fixed `10`-epoch reference set.
- The `40`-epoch fairness gap is closed by reusing or backfilling `ffno_bottleneck_base`, and an authoritative `40`-epoch local-conv compare exists against the fixed `40`-epoch reference set.
- Durable summary, CNS summary sync, and progress-ledger completion state are all updated.
- All newly generated authoritative outputs are explicitly labeled as `pilot` / `capped_decision_support_only`, and any inherited readiness-provenance reference roots are called out explicitly as historical anchors rather than silently treated as newly generated decision-support evidence.
