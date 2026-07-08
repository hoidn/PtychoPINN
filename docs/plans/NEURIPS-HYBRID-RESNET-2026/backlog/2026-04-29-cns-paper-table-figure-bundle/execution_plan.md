# CNS Paper Table And Figure Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the CNS paper-table and figure bundle under one honest capped contract, preferring a same-contract `1024 / 128 / 128` bundle if the full headline roster can be recovered or rerun, and otherwise falling back cleanly to the already locked `512 / 64 / 64` bundle.

**Architecture:** Treat the approved CNS contract decision, the existing row-lock summary, the roadmap's `1024-first / 512-fallback` rule, and the selected backlog item as binding together. Implementation proceeds in three units: first audit whether a same-contract `1024 / 128 / 128` headline roster can be made authoritative without contract drift; second refresh the contract and row-lock authority only if that upgrade is real; third assemble deterministic table, figure, and manifest outputs from whichever row-lock authority is current. No `/home/ollie/Documents/neurips/` outputs belong in this item.

**Tech Stack:** Python 3.11, PyTorch study helpers under `scripts/studies/pdebench_image128/`, NumPy/JSON/CSV/TeX serialization, matplotlib, pytest.

---

## Selected Backlog Objective

Build the CNS paper table and figure bundle for the current PDEBench `2d_cfd_cns` paper lane.

The bundle must:

- first try to upgrade the bounded headline roster to a same-contract `1024 / 128 / 128`, `history_len=2`, `40`-epoch, `max_windows_per_trajectory=8` capped lane
- reuse the existing `1024 / 128 / 128` spectral-family row already recorded in the roadmap/progress ledger
- inventory, recover, or rerun same-cap `author_ffno_cns_base`, `fno_base`, and `unet_strong` rows if needed
- refresh checked-in CNS contract/row-lock authority before using any upgraded `1024` roster
- fall back to the earlier complete `512 / 64 / 64` locked bundle if any required same-cap `1024` headline row cannot be produced honestly
- emit paper-ready JSON, CSV, and TeX metric tables plus fixed-sample visual bundles, source arrays, and a durable claim-boundary summary

## Scope

In scope:

- only the bounded capped CNS paper lane
- only the approved official file:
  `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- only the approved training recipe family:
  `history_len=2`, `40` epochs, `mse`, `Adam lr=2e-4`, `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`, batch size `4`, `max_windows_per_trajectory=8`
- only the approved metric family:
  `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- same-cap recovery or rerun work needed to determine whether the `1024 / 128 / 128` headline roster is real
- row-lock authority refresh if the `1024` roster is completed
- bundle assembly from the current authoritative locked-row manifest
- durable docs and index updates when new reusable or durable knowledge is created

Out of scope:

- any full-training CNS benchmark promotion
- widening the paper lane to `history_len=1`, `history_len=3`, GNOT, repo-local FFNO proxies, or `2048 / 256 / 256`
- changing dataset file, history length, normalization rules, optimizer/scheduler family, or authored-FFNO cutoff without a checked-in contract amendment
- manuscript prose
- Phase 5 `/home/ollie/Documents/neurips/` artifact assembly

## Explicit Non-Goals

- Do not silently relabel capped rows as `full_training` or `paper_grade`.
- Do not mix `1024 / 128 / 128` headline rows with `512 / 64 / 64` headline rows in one merged headline table.
- Do not let `hybrid_resnet_cns` silently replace a required headline row.
- Do not publish merged outputs with missing metrics or missing claim labels unless they are explicitly marked `benchmark_incomplete` or row-level missing-field statuses.
- Do not let figure generation choose different sample IDs, field order, or scales per row.

## Binding Constraints And Prerequisites

Strategic and roadmap constraints:

- `docs/steering.md` keeps equal-footing comparison rules and forbids silently relaxing fairness constraints.
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md` Phase `2.8e` and the later CNS queue notes require:
  - try a same-contract `1024 / 128 / 128` bounded bundle first
  - fall back to the earlier complete `512 / 64 / 64` lock if a required same-cap row is missing
  - never mix caps in one headline table
  - keep `2048 / 256 / 256` as later evidence-strengthening context only
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md` is the current contract authority and explicitly says split-count changes require a checked-in amendment or new decision rather than silent drift.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md` is the current row-lock authority and currently locks the `512 / 64 / 64` bundle as `capped_decision_support`.

Progress-ledger prerequisites already complete:

- `phase-2-pdebench-2d-cfd-cns-adapter`
- `phase-2-pdebench-2d-cfd-cns-readiness`
- `phase-2-pdebench-2d-cfd-cns-capped-comparison`
- `2026-04-21-pdebench-author-ffno-equal-footing-cns`
- `2026-04-29-cns-paper-contract-decision`
- `2026-04-29-cns-paper-benchmark-rows`
- `phase-2-pdebench-cns-hybrid-spectral-architecture-ablation`
- `2026-04-28-pdebench-cns-hybrid-spectral-scaling-2048cap` is complete but remains non-blocking context only

Current authority state that must be preserved:

- today the checked-in authoritative row lock is the `512 / 64 / 64` bundle
- the existing `1024 / 128 / 128` evidence is incomplete for the headline roster until same-cap `author_ffno_cns_base`, `fno_base`, and `unet_strong` rows are recovered or rerun and then locked
- reused CNS roots still lack standalone repo git SHA, dirty-state, run-log, and exit-code artifacts; any resulting bundle remains `capped_decision_support`

Blocking policy for this item:

- normal import failures, pytest failures, path mistakes, or harness bugs must be diagnosed, fixed, and rerun first
- reserve `BLOCKED` for missing external artifacts that cannot be recovered within current authority, unavailable hardware, roadmap conflict, or a required user decision

## Concrete File And Artifact Targets

Likely code targets:

- Create: `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Modify: `scripts/studies/pdebench_image128/reporting.py`
- Modify: `scripts/studies/pdebench_image128/visualization.py`
- Modify: `scripts/studies/run_pdebench_image128_suite.py` only if a stable CLI entrypoint for bundle assembly or `1024` audit is needed

Likely test targets:

- Modify: `tests/studies/test_pdebench_image128_runner.py`
- Modify: `tests/studies/test_pdebench_cfd_cns_metrics.py`
- Modify: `tests/studies/test_pdebench_image128_visualization.py`
- Modify: `tests/studies/test_studies_index_entries.py` if discoverability docs change

Likely documentation targets:

- Modify if the `1024` upgrade succeeds:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Modify if the `1024` upgrade succeeds:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- Modify if the locked-row authority changes:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`
- Create:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md`
- Modify:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify `docs/studies/index.md` if the bundle entrypoint is reusable/discoverable
- Modify `docs/index.md` if the new summary becomes durable top-level documentation

Bundle-local artifact root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/`

Expected bundle-local artifacts:

- `1024_same_cap_audit.json` and `.md`
- optional `1024` rerun manifests and launcher sidecars if new same-cap rows are run
- final bundle input manifest pointing at the authoritative locked-row JSON actually used
- canonical table payloads in JSON, CSV, and TeX
- fixed-sample manifest
- shared field-scale and shared error-scale manifests
- canonicalized source `.npz` files for every emitted figure panel
- rendered prediction/error panels for `density`, `Vx`, `Vy`, and `pressure`
- final bundle validation payload proving row roster, claim labels, and figure/table agreement

## Implementation Architecture

- `cns_paper_bundle.py` owns the orchestration boundary:
  `1024` audit, optional rerun manifest creation, locked-row manifest loading, table/figure assembly, and final validation.
- `reporting.py` owns reusable contract parsing, row normalization, missing-field labeling, and multi-format table serialization.
- `visualization.py` owns reusable CNS field-scale policy so per-field signed/scalar rules and shared error ranges are defined once.
- The checked-in contract and row-lock docs remain the only scientific authority. Code may not invent a new bundle contract that the docs do not record.

## Task 1: Audit The `1024 / 128 / 128` Upgrade Path Before Bundle Assembly

**Files:**

- Create or extend: `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Modify: `scripts/studies/pdebench_image128/reporting.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/`

- [ ] Load the current contract decision, current row-lock summary, and current locked-row JSON as explicit inputs.
- [ ] Audit whether the `1024 / 128 / 128` headline roster can be made same-contract for:
  `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`, `author_ffno_cns_base`.
- [ ] Reuse the existing spectral-family `1024` root from:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`.
- [ ] Inventory whether same-cap `1024` `fno_base`, `unet_strong`, and `author_ffno_cns_base` roots already exist with the exact approved file, split counts `1024 / 128 / 128`, `history_len=2`, `40` epochs, `mse`, batch size `4`, and `max_windows_per_trajectory=8`.
- [ ] If a required same-cap row is missing or contract-incompatible, prepare rerun commands using the existing runner rather than ad hoc scripts.
- [ ] Require the mandatory pytest and compileall checks to pass before launching any expensive rerun.
- [ ] If reruns are needed, launch them with tmux and the `ptycho311` environment, track exact PIDs, and record run completion only when exit code is `0` and required artifacts are freshly written.
- [ ] Write `1024_same_cap_audit.json` and `.md` that say one of:
  - `upgrade_ready`
  - `upgrade_ready_after_reruns`
  - `fallback_to_512_required`
- [ ] If the audit concludes `fallback_to_512_required`, record row-level blockers and stop the `1024` path without marking the whole backlog item blocked.

**Verification after Task 1:**

- [ ] Extend tests so audit helpers reject contract drift on dataset file, split counts, history length, epoch budget, training loss, batch size, or metric family.
- [ ] Extend tests so a mixed-cap headline roster is rejected.
- [ ] Extend tests so the `1024` audit emits deterministic `upgrade_ready` vs `fallback_to_512_required` outcomes from fixture inputs.
- [ ] Run the required selectors before any rerun:
  - `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
  - `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`

## Task 2: Refresh Contract And Row-Lock Authority Only If The `1024` Roster Is Real

**Files:**

- Modify when applicable:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Modify when applicable:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- Modify when applicable:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Test: `tests/studies/test_pdebench_cfd_cns_metrics.py`

- [ ] If Task 1 proves the `1024` headline roster is complete and same-contract, update the contract decision or write its explicit amendment before using that roster downstream.
- [ ] Record the exact widened capped lane:
  `1024 / 128 / 128`, `history_len=2`, `40` epochs, `max_windows_per_trajectory=8`, same official file, same normalization contract, same recipe family, same metric family, still `bounded_capped_decision_support`.
- [ ] Refresh the row-lock summary and locked-row JSON so the bundle consumes a checked-in `1024` authority surface rather than an in-memory audit result.
- [ ] Keep the claim boundary unchanged:
  the widened lane is still capped decision-support only.
- [ ] If the `1024` headline upgrade fails, do not modify the contract decision or row-lock authority; keep the existing `512 / 64 / 64` lock as the authoritative input.
- [ ] Handle `hybrid_resnet_cns` explicitly:
  - if a same-cap `1024` continuity row is recoverable or rerunnable under the same contract, lock it as continuity/support only
  - otherwise keep the existing `512` continuity row as narrative context only and exclude it from any upgraded same-cap merged headline table and shared-scale figure bundle

**Verification after Task 2:**

- [ ] Add tests for contract-amendment serialization and locked-row JSON consistency.
- [ ] Add a deterministic sync check that the authoritative row-lock document and JSON agree on row IDs, split counts, and row roles.
- [ ] If no `1024` upgrade occurs, add a validation step that the implementation still points at the unchanged `512` locked-row JSON.

## Task 3: Assemble Canonical Table Outputs From The Current Locked-Row Authority

**Files:**

- Create or extend: `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Modify: `scripts/studies/pdebench_image128/reporting.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`

- [ ] Load whichever locked-row JSON is authoritative after Task 2.
- [ ] Emit one canonical bundle-row payload and derive JSON, CSV, and TeX from that single source.
- [ ] Record for every row:
  `row_id`, `row_role`, `row_status`, `claim_scope`, split/cap/full-training labels, metrics, parameter count, runtime, hardware/runtime provenance note, source run root, and missing-field labels when applicable.
- [ ] Include the required metrics:
  `err_nRMSE`, `err_RMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`, parameter count, runtime, hardware, split/cap/full-training status, and row status.
- [ ] Never guess a precise hardware label from roadmap context alone. If the run artifact does not carry a precise accelerator string, emit an explicit artifact-derived note or missing-field marker.
- [ ] If a row is missing a required table field, mark it explicitly and propagate `benchmark_incomplete` at the bundle level rather than silently dropping it.
- [ ] Keep headline rows ordered deterministically and keep any continuity row separated clearly from the headline roster.

**Verification after Task 3:**

- [ ] Add tests for complete rows, missing-field rows, `benchmark_incomplete` propagation, continuity-row separation, and JSON/CSV/TeX row-order consistency.
- [ ] Add a bundle validation check that asserts:
  - no mixed-cap headline table exists
  - all headline rows share one authoritative contract
  - every emitted row remains `capped_decision_support`
  - no output advertises `paper_grade` or `full_training`

## Task 4: Freeze Shared Samples, Shared Scales, And Figure Source Arrays

**Files:**

- Create or extend: `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Modify: `scripts/studies/pdebench_image128/visualization.py`
- Test: `tests/studies/test_pdebench_image128_runner.py`
- Test: `tests/studies/test_pdebench_image128_visualization.py`

- [ ] Choose the figure bundle input rows from the authoritative locked-row manifest only.
- [ ] Freeze deterministic sample IDs by intersecting compatible source arrays across the rows actually included in the visual bundle.
- [ ] Use one shared visualization scale per field (`density`, `Vx`, `Vy`, `pressure`) across the common target and all accepted predictions for that field.
- [ ] Use one shared absolute-error scale per field across all accepted rows in the visual bundle.
- [ ] Copy or canonicalize the source `.npz` arrays into the bundle artifact root so the figures can be regenerated without scraping old run roots.
- [ ] Render prediction, ground-truth, and absolute-error panels for every included row and every selected sample.
- [ ] Emit manifests that map each figure back to sample ID, row ID, field name, scale entry, and copied source-array path.
- [ ] Fail closed on target mismatch, field-order mismatch, or incompatible sample availability.
- [ ] If the bundle upgraded only the headline roster to `1024`, do not mix a `512` continuity row into the shared-scale visual bundle.

**Verification after Task 4:**

- [ ] Add tests for deterministic sample selection, shared-scale determinism, signed velocity symmetry, zero-based error scales, source-array manifest completeness, and mismatch failures.
- [ ] Add a local bundle smoke test from fake locked rows that verifies the expected PNG, NPZ, and manifest set exists.

## Task 5: Write Durable Summary And Discoverability Updates

**Files:**

- Create:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md`
- Modify:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify `docs/studies/index.md` if the bundle entrypoint is intentionally reusable
- Modify `docs/index.md` if the new summary becomes part of the durable documentation map
- Test: `tests/studies/test_studies_index_entries.py` if any index/discoverability entry changes

- [ ] Write a durable summary that records:
  - which contract authority and which locked-row manifest the bundle used
  - whether the final bundle is `1024 / 128 / 128` or `512 / 64 / 64`
  - exact headline row roster
  - continuity-row handling
  - fixed sample IDs
  - shared scale policy
  - emitted artifact paths
  - explicit capped-decision-support claim boundary
  - preserved provenance gaps preventing `paper_grade`
- [ ] Update `pdebench_2d_cfd_cns_summary.md` so readers can find the paper-bundle authority quickly.
- [ ] Update `docs/studies/index.md` only if the new bundle builder is a reusable discoverable study surface.
- [ ] Update `docs/index.md` if the new durable summary should appear in the documentation hub.

**Verification after Task 5:**

- [ ] If index docs changed, extend or run `tests/studies/test_studies_index_entries.py`.
- [ ] Confirm the summary never claims full-training or paper-grade CNS evidence.

## Required Deterministic Checks

These backlog check commands are mandatory. Extend the existing tested modules instead of replacing them with a narrower selector:

- [ ] `pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_image128_runner.py`
- [ ] `python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py`

Add stronger bundle-specific checks after implementation:

- [ ] Run the bundle builder against the authoritative locked-row JSON and verify the emitted JSON, CSV, TeX, manifests, figure PNGs, and copied source arrays all exist and agree.
- [ ] If a `1024` upgrade occurred, run a doc/artifact sync check proving the contract decision, row-lock summary, and locked-row JSON all reflect the same widened capped lane.
- [ ] Archive pytest and compileall logs under the backlog artifact root or another linked verification path per `docs/TESTING_GUIDE.md`.

## Long-Run Gate

- [ ] Do not start any `1024` rerun until the mandatory pytest and compileall checks are green.
- [ ] If reruns are required, use tmux plus `ptycho311`, track the launched PID exactly, and confirm fresh required artifacts after exit code `0`.
- [ ] If a rerun fails for a normal code or environment reason, diagnose, fix, and rerun before considering fallback or `BLOCKED`.

## Execution Notes

- The safe default is the existing `512 / 64 / 64` locked bundle.
- The `1024` path is allowed only if it becomes a checked-in authoritative contract and row-lock, not merely a loose collection of runs.
- The final implementation must leave one clear answer in the durable summary:
  either `same_contract_1024_bundle_complete` or `fallback_512_bundle_used`.
