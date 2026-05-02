# PDEBench CNS Paper 2048-Cap Row Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the current bounded CNS paper bundle to a same-contract `2048 / 256 / 256`, `history_len=2`, `40`-epoch comparator lane only if every required headline row can be recovered or rerun under the exact locked local contract, while preserving the current `512 / 64 / 64` paper bundle as the active authority until that promotion is actually complete.

**Architecture:** Treat this item as a late Phase 2 evidence-strengthening pass, not a contract rewrite. Reuse the already completed `2048` spectral-family row(s), audit the remaining comparator coverage against the locked CNS contract, repair only the smallest study-local helper surfaces needed to support a same-cap `2048` audit/bundle, then recover or launch only the missing `2048` rows. Publish a separate extension summary in all cases; publish a replacement `2048` table/figure payload only if the full same-cap roster validates without mixing caps.

**Tech Stack:** PATH `python`; tmux with `ptycho311` activated for long runs; PyTorch/Lightning; `scripts/studies/pdebench_image128/`; pytest; compileall; repo-local `.artifacts/` plus Markdown/JSON/CSV/TeX summaries.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-paper-2048cap-row-extension`
- Selection mode: `ACTIVE_SELECTION`
- Authoritative backlog context:
  `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/4/items/2026-04-29-cns-paper-2048cap-row-extension/selected-item-context.md`
- Previous plan path used as background only:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Plan authority path:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/execution_plan.md`
- Preferred item artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/`

This document is the execution authority for this backlog item. It supersedes prior plan content at this path.

## Inputs Read

- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/workflows/pytorch.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/4/items/2026-04-29-cns-paper-2048cap-row-extension/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap/execution_plan.md`

## Selected Objective

- Promote the CNS comparator bundle to a same-contract `2048 / 256 / 256` capped row set as a later evidence-strengthening pass.
- Reuse the completed `2048 / 256 / 256` spectral-family scaling evidence where it satisfies the locked local CNS contract.
- Recover or run same-cap `2048 / 256 / 256`, `history_len=2`, `40`-epoch, `max_windows_per_trajectory=8` rows for:
  - `author_ffno_cns_base`
  - `fno_base`
  - `unet_strong`
- Keep the current paper-facing CNS authority on the completed fallback `512 / 64 / 64` bundle unless the new `2048` lane finishes with a complete same-cap headline roster and the same row-status/provenance checks.

## Scope

- Preserve the locked CNS local contract from the existing paper lane, except for the intended split-count increase:
  - official dataset:
    `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
  - `history_len=2`
  - split counts `2048 / 256 / 256`
  - `max_windows_per_trajectory=8`
  - `40` epochs
  - training loss `mse`
  - optimizer `Adam`, learning rate `2e-4`
  - `ReduceLROnPlateau` with factor `0.5`, patience `2`, threshold `0.0`, `min_lr=1e-5`
  - batch size `4`
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Required `2048` headline roster for promotion:
  - `spectral_resnet_bottleneck_base`
  - `author_ffno_cns_base`
  - `fno_base`
  - `unet_strong`
- Reuse, do not rerun, the completed `2048` spectral-family result if it matches the same contract and row-status rules.
- `hybrid_resnet_cns` remains optional continuity/support only:
  - do not make a missing `2048` Hybrid continuity row a headline blocker
  - do not rerun `hybrid_resnet_cns` unless a same-cap support row is trivially available and useful for the visual bundle
- Build a separate `2048` extension payload rather than rewriting the existing fallback `512` bundle in place.

## Explicit Non-Goals

- Do not delay or reopen the already usable fallback `512 / 64 / 64` CNS table/figure bundle or the existing paper evidence audit.
- Do not mix `2048` rows with `1024` or `512` rows in one headline CNS table.
- Do not relabel any capped CNS row as `paper_grade` or `full_training`.
- Do not widen this into full-training CNS benchmarking, history-length experimentation, FFNO proxy promotion, GNOT work, Darcy work, or CDI work.
- Do not treat repo-local FFNO proxy rows as substitutes for `author_ffno_cns_base`.
- Do not touch `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create `/home/ollie/Documents/neurips/` outputs.
- Do not create worktrees.

## Steering, Roadmap, And Prerequisite Constraints

- Steering is binding:
  - keep equal-footing comparisons explicit
  - preserve metric, split, and protocol boundaries from the approved CNS lane
  - do not silently relax fairness constraints to make `2048` promotion easier
  - this item is later, optional evidence strengthening; it must not displace required work already completed or still pending elsewhere
- Roadmap/design are binding:
  - this remains Phase 2 PDEBench CNS work
  - this remains capped decision-support evidence only
  - the full-available-training-split benchmark gate is still unmet and remains out of scope here
  - the current paper story stays asymmetric unless and until a later full-training CNS authority exists
- Current authority surfaces already completed:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- Prerequisite status:
  - the consumed progress ledger does not yet enumerate these later CNS paper-contract backlog items, so prerequisite truth comes from the consumed contract/row-lock/bundle/audit summaries rather than a ledger tranche entry
  - those consumed summaries show the current fallback `512` bundle is complete enough for bounded manuscript support now
  - this backlog item is therefore authorized as a non-blocking follow-up extension rather than a prerequisite-clearing task
- The `2048` spectral-family evidence already exists and must be reused if valid:
  - authority:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`
  - preserve its claim boundary:
    `capped_decision_support_only`
- `REPORTING-ARTIFACT-BOUNDARY-001` applies:
  - required metrics/manifests/validation decide success
  - optional gallery failures can stay warnings if the core run artifacts are complete
- `PYTHON-ENV-001` applies:
  - use plain PATH `python`
- Long-run guardrails are mandatory:
  - use tmux and activate `ptycho311`
  - do not launch duplicate writers to the same `--output-root`
  - track the exact launched PID
  - wait on that PID and record exit code
  - require fresh output artifacts plus exit code `0` before calling a rerun complete
- Failure handling:
  - do not mark the item `BLOCKED` for normal import, pytest, compileall, harness, or runner failures; diagnose, patch narrowly, and rerun first
  - reserve `BLOCKED` for missing data, unavailable hardware/host, roadmap conflict outside current authority, user decision required, or unrecoverable failure after a documented narrow fix attempt

## Implementation Architecture

- Same-cap audit unit:
  - audit which required `2048` rows already exist and satisfy the locked contract, and emit machine-readable rerun instructions for only the missing or incompatible rows
- Bundle orchestration unit:
  - generalize the current `1024`-specific audit/bundle helper only as far as needed to support a separate `2048` extension path without changing the meaning of the existing fallback `512` bundle
- Long-run execution unit:
  - recover or rerun only the missing `2048` comparator rows and keep ownership of those launches until verified completion or a narrow, documented unrecoverable blocker
- Authority-sync unit:
  - write the extension summary always; update evidence indexes and paper-audit surfaces only if the `2048` roster becomes a complete same-cap replacement authority

## Concrete File And Artifact Targets

- Study code that may change only if preflight or same-cap promotion is blocked by current helper behavior:
  - `scripts/studies/pdebench_image128/cns_paper_bundle.py`
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/run_pdebench_image128_suite.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
- Test surfaces that may need narrow additions or updates if the helper becomes parameterized:
  - `tests/studies/test_pdebench_image128_runner.py`
  - `tests/studies/test_pdebench_cfd_cns_metrics.py`
  - `tests/studies/test_paper_evidence_audit.py` only if a new `2048` bundle becomes the authoritative CNS audit input

- Mandatory contract outputs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_same_cap_audit.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_rerun_manifest.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
  - if promotion succeeds:
    - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/cns_paper_locked_rows_2048cap.json`
    - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/cns_paper_table_rows.json`
    - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/cns_paper_table_rows.csv`
    - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/cns_paper_table_rows.tex`
    - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/figure_manifest.json`
    - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/fixed_sample_manifest.json`
    - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/bundle_validation.json`
- Preferred packaging:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_same_cap_audit.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/*.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/rerun_candidates/<profile>-2048cap-40ep/`

- Durable index/state surfaces to update when durable knowledge changes:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json` only if the `2048` bundle becomes the new authoritative CNS payload
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md` only if the `2048` bundle becomes the new authoritative CNS audit input
  - `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` only if backlog-level durable state tracking is extended to reference this completed extension outcome
  - `docs/index.md` because the new extension summary is a durable planning surface
- `docs/findings.md` only if implementation uncovers a reusable runner/bundle rule beyond this item

## Required Deterministic Check Commands

These selected-item `check_commands` are mandatory execution authority for this
backlog item and must remain exact:

- `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
- `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`

Execution rule:

- Run both commands after Task 1 freezes the `2048` audit/rerun contract and
  before any fresh `2048` rerun launch, even if Task 2 decides no code edit is
  required.
- If Task 2 changes helper code, rerun both commands immediately after the edit
  pass before any expensive launch.
- Run both commands again at closeout so the extension summary can report that
  the required deterministic checks remained green for the final item state.
- Archive stdout/stderr logs for each invocation under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/verification/`.

### Task 1: Audit Same-Cap 2048 Coverage And Freeze The Promotion Contract

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_same_cap_audit.json`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_rerun_manifest.json`
- Prefer: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_same_cap_audit.md`
- Use: `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Use: `scripts/studies/pdebench_image128/reporting.py`

- [ ] Create the item-local artifact root and `verification/` directory.
- [ ] Encode the exact promotion contract for the `2048` lane:
  - required headline rows:
    `spectral_resnet_bottleneck_base`, `author_ffno_cns_base`, `fno_base`, `unet_strong`
  - split counts:
    `2048 / 256 / 256`
  - invariant contract fields:
    dataset file, history length, epoch budget, batch size, training loss, metric family, `max_windows_per_trajectory`
- [ ] Reuse the completed `2048` spectral-family run authority from
  `pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`; do not relaunch it if the row already passes the same contract and row-status checks.
- [ ] Search for same-contract `2048` roots for `author_ffno_cns_base`, `fno_base`, and `unet_strong`.
- [ ] Write a machine-readable audit that classifies each required row as one of:
  - reusable same-contract row
  - missing row requiring rerun
  - incompatible row with an explicit mismatch reason
- [ ] Emit rerun instructions only for rows that are still missing or incompatible after audit.
- [ ] Keep the current fallback `512` bundle immutable; the audit must not overwrite
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/`.

**Verification**

- Blocking:
  - the audit JSON exists and names the exact `2048 / 256 / 256`, `history_len=2`, `40`-epoch contract
  - the audit records every required headline row and does not mix in `512`/`1024` rows as headline candidates
  - the required backlog-item deterministic checks have been run once against the
    frozen item contract before any fresh rerun launch:
    - `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
    - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
- Supporting:
  - a markdown audit summary exists for human inspection

### Task 2: Generalize The Bundle Helper Only If The Current 1024-Specific Surface Blocks 2048 Promotion

**Files:**
- Modify only if required: `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Modify only if required: `scripts/studies/pdebench_image128/reporting.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Test only if audit authority routing changes: `tests/studies/test_paper_evidence_audit.py`

- [ ] Inspect whether the current helper is hardcoded to `1024` audit/manifest naming or bundle-kind semantics in a way that blocks a separate `2048` extension path.
- [ ] If the current helper is sufficient, leave code untouched and proceed.
- [ ] If it is not sufficient, patch only the smallest study-local surface needed to:
  - parameterize same-cap audit split counts
  - emit `2048`-named audit artifacts without changing historical `1024`/`512` outputs
  - produce a new `2048` bundle root instead of mutating the fallback `512` bundle root
  - preserve mixed-cap rejection and capped-decision-support labeling
- [ ] Add or update focused tests for:
  - same-cap contract discovery at `2048 / 256 / 256`
  - mixed-cap headline rejection
  - new bundle-validation payloads if a new `2048` bundle path is emitted
  - unchanged fallback behavior for the existing `512` bundle path

**Verification**

- Blocking before any expensive reruns:
  - `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
- Supporting:
  - if the `2048` bundle becomes the new audit input, run `pytest -q tests/studies/test_paper_evidence_audit.py`

### Task 3: Recover Or Rerun Only The Missing 2048 Comparator Rows

**Files:**
- Create as needed: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/rerun_candidates/<profile>-2048cap-40ep/`
- Use: `scripts/studies/run_pdebench_image128_suite.py`
- Use: `scripts/studies/pdebench_image128/cfd_cns.py`

- [ ] Launch no fresh run until Task 2 blocking checks are green.
- [ ] For each missing/incompatible required row, launch only that row under the fixed `2048 / 256 / 256` contract with:
  - `--task 2d_cfd_cns`
  - `--mode pilot`
  - `--profiles <missing_profile_id>`
  - `--history-len 2`
  - `--epochs 40`
  - `--batch-size 4`
  - `--max-train-trajectories 2048`
  - `--max-val-trajectories 256`
  - `--max-test-trajectories 256`
  - `--max-windows-per-trajectory 8`
  - `--device cuda`
  - `--num-workers 0`
- [ ] Use tmux plus `ptycho311`; keep implementation ownership of every launched rerun until terminal success or recoverable failure handling is complete.
- [ ] For each long run, record and verify:
  - tracked PID
  - exit code
  - `invocation.json`
  - `dataset_manifest.json`
  - `split_manifest.json`
  - `comparison_summary.json`
  - `metrics_<profile>.json`
  - `model_profile_<profile>.json`
  - fresh timestamps proving the artifacts were written by this rerun
- [ ] If a rerun fails because of a normal harness or environment issue, diagnose, patch narrowly, and rerun before considering the row blocked.
- [ ] If a row remains unavailable only because the dataset, GPU host, or another external prerequisite is unavailable after a narrow fix attempt, record a row-level blocker and stop launching further dependent promotion work.

**Verification**

- Blocking:
  - every rerun launched for this item exits `0`
  - every launched rerun has the required fresh artifacts
  - every recovered or rerun row matches the exact `2048` contract and keeps `evidence_scope=capped_decision_support_only`
- Supporting:
  - row-level comparison summaries exist for recovered/rerun rows

### Task 4: Lock The 2048 Bundle Conditionally And Publish Durable Authority Updates

**Files:**
- Create always: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
- Create on success: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/cns_paper_locked_rows_2048cap.json`
- Create on success: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/bundle_2048cap/`
- Update conditionally: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Update conditionally: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Update conditionally: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Update only if the authoritative CNS bundle changes: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- Update only if the authoritative CNS audit input changes: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- Update: `docs/index.md`

- [ ] If every required headline row is available under the same `2048` contract, write a new `2048` locked-row manifest and build a new bundle root with JSON/CSV/TeX rows, fixed-sample figure bundle, source arrays, and bundle validation.
- [ ] If any required row is still missing or incompatible, do not emit a mixed-cap headline bundle; instead:
  - keep the current fallback `512` bundle as the live paper authority
  - record the exact row-level blockers in the extension summary
  - keep any new `2048` rows as decision-support or partial promotion context only
- [ ] In all cases, write the extension summary with:
  - the audited `2048` contract
  - which rows were reused versus rerun
  - any row-level blockers
  - whether a replacement `2048` bundle was completed
  - the preserved claim boundary:
    capped decision-support only
- [ ] If a full `2048` replacement bundle exists, update discoverability surfaces so they point at the new authority while preserving the older fallback bundle as historical provenance.
- [ ] If only a partial `2048` extension exists, update discoverability surfaces just enough to expose the new summary/output without mislabeling it as the current CNS paper authority.
- [ ] Add `model_variant_index.json` entries for any new or newly promoted `2048` CNS rows; add a new dataset-contract entry such as `cns_history2_cap2048_40ep` if fresh or newly authoritative `2048` rows were produced.

**Verification**

- Blocking:
  - any emitted `2048` headline bundle has zero mixed-cap headline rows
  - any emitted `2048` headline bundle keeps every row at `capped_decision_support_only`
  - if the authoritative CNS bundle changes, the paper-evidence manifest/audit surfaces agree on the new bundle paths
  - the required backlog-item deterministic checks remain green after any code edits:
    - `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
    - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`
- Supporting:
  - if the authoritative CNS bundle changes, `pytest -q tests/studies/test_paper_evidence_audit.py`
  - manual inspection of the extension summary confirms that the older fallback `512` authority is either explicitly preserved or explicitly superseded by a same-cap `2048` authority, never silently replaced

## Completion Conditions

- Minimum acceptable completion:
  - the `2048` audit and extension summary exist
  - any reusable `2048` spectral-family evidence was correctly consumed
  - any missing comparator rows were either recovered/rerun under the fixed contract or given explicit row-level blockers
  - the current fallback `512` bundle remains the active authority if `2048` promotion is incomplete
- Full promotion completion:
  - all four required headline rows exist under the same `2048 / 256 / 256` contract
  - a new `2048` locked-row manifest and replacement bundle exist
  - all authority/index surfaces that should now point at `2048` are synchronized
- Not acceptable:
  - mixed-cap headline table
  - silent replacement of the current bundle without index/audit sync
  - promotion of any capped evidence to `paper_grade` or `full_training`
  - declaring `BLOCKED` without first attempting narrow diagnosis/fix/rerun for ordinary harness failures
