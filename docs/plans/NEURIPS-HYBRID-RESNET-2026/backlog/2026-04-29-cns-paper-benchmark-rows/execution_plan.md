# CNS Paper Benchmark Rows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lock the PDEBench `2d_cfd_cns` paper-row bundle under the already approved `bounded_capped_decision_support` contract by freezing the accepted same-contract headline rows, auditing their provenance and parity fields, and publishing one durable row-lock summary plus machine-readable manifest for downstream table/figure assembly.

**Architecture:** Treat this as a row-lock and provenance-packaging lane, not a new benchmark-execution lane. First verify that the selected same-contract run roots named by the approved CNS paper-contract decision are present and complete enough to serve as locked rows. Then patch only narrow runner/reporting metadata gaps if the audit proves a required field is missing. Finally emit one locked-row manifest and one durable summary that downstream CNS table/figure work can consume without rereading historical summaries or inferring hidden contract assumptions.

**Tech Stack:** PATH `python`, PyTorch/PDEBench image-suite surfaces under `scripts/studies/pdebench_image128/`, pytest, compileall, Markdown/JSON artifacts under `.artifacts/`, tmux with `ptycho311` only if a narrow same-contract rerun becomes unavoidable.

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Backlog item: `2026-04-29-cns-paper-benchmark-rows`
- Selection mode: `ACTIVE_SELECTION`
- Plan authority date: `2026-04-29`
- Scope owner: Roadmap Phase 2 CNS paper-row-lock lane
- Selected-item context: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-cns-paper-benchmark-rows/selected-item-context.md`
- Recorded plan path source: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-cns-paper-benchmark-rows/plan-phase/plan_path.txt`
- Previous plan path from selected-item context: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Durable summary target: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/`

This document supersedes earlier plan content for this backlog item and is the execution authority for implementation.

## Inputs Read

- `AGENTS.md`
- `docs/index.md`
- `docs/findings.md`
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`
- `docs/DEVELOPER_GUIDE.md`
- `docs/TESTING_GUIDE.md`
- `docs/COMMANDS_REFERENCE.md`
- `docs/workflows/pytorch.md`
- `docs/model_baselines.md`
- `docs/studies/index.md`
- `docs/backlog/index.md`
- `docs/steering.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- `docs/backlog/in_progress/2026-04-29-cns-paper-benchmark-rows.md`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-cns-paper-benchmark-rows/selected-item-context.md`
- `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-cns-paper-benchmark-rows/plan-phase/plan_path.txt`

## Selected Backlog Objective

- Produce or lock the CNS rows required by the selected paper evidence contract.
- Treat `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md` as the only authority for the CNS headline contract.
- Reuse the selected contract verbatim:
  - contract: `bounded_capped_decision_support`
  - history lane: `history_len=2`, `40` epochs, `512 / 64 / 64` trajectories, `max_windows_per_trajectory=8`, emitted windows `4096 / 512 / 512`
  - normalization: train-only per-field normalization fit on the `512` training trajectories, reused across all history slots and target channels, with evaluation reported in denormalized target space
  - training recipe: task-local `mse`, `Adam(lr=2e-4)`, `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`, batch size `4`
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`
- Lock the approved headline row roster exactly as:
  - `spectral_resnet_bottleneck_base`
  - `fno_base`
  - `unet_strong`
  - `author_ffno_cns_base`
- Keep `hybrid_resnet_cns` as an audited continuity/support row only in this pass.

## Scope

- Audit the accepted same-contract run roots named by the contract decision and confirm that each locked row has enough provenance, metrics, and asset pointers to serve as a downstream paper-row authority.
- Freeze one machine-readable locked-row manifest that carries row role, row status, contract fields, metrics, parameter count, runtime, provenance pointers, and source-array/sample-asset pointers.
- Write one durable CNS row-lock summary under `docs/plans/NEURIPS-HYBRID-RESNET-2026/` that states the locked roster, continuity row, excluded adjacent context, claim boundary, and any row-level blocker or incompatibility.
- Patch only the narrowest runner/reporting/test surfaces if the audit exposes a real missing emitted field that blocks row locking.
- Update discoverability and selector state only where durable project knowledge changes.

## Explicit Non-Goals

- Do not reopen the `history_len=3` lane for this item.
- Do not fund or imply a new authored-FFNO rerun under another temporal contract.
- Do not substitute repo-local `ffno_bottleneck_base` or `ffno_bottleneck_localconv_base` for `author_ffno_cns_base`.
- Do not make GNOT a required row in this headline bundle.
- Do not turn capped evidence into a full-training benchmark claim by wording alone.
- Do not build the final CNS paper table/figure bundle here. That belongs to `2026-04-29-cns-paper-table-figure-bundle`.
- Do not create `/home/ollie/Documents/neurips/` artifacts, manuscript prose, or Phase 5 evidence-index outputs.
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not create worktrees.

## Steering, Roadmap, and Fairness Constraints

- Steering requires explicit equal-footing comparisons and forbids silently relaxing fairness constraints to make the item easier.
- The roadmap keeps this work inside Phase 2 PDEBench CNS paper packaging. It must not be presented as Phase 3 CDI completion or Phase 5 evidence-bundle work.
- The approved design and image-suite plan require the official file, split family, history length, normalization contract, recipe contract, and metric family to stay fixed inside the headline bundle.
- The locked official CNS dataset remains:
  - `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- The current canonical local Hybrid CNS shell remains `hybrid_resnet_cns` with skip-add plus pixelshuffle, but that row is continuity/support only in this pass and must not displace the approved headline roster.
- Active findings and durable study decisions still apply:
  - `PDEBENCH-CNS-UPSAMPLER-001`: keep `pixelshuffle` as the canonical CNS Hybrid upsampler.
  - `PDEBENCH-CNS-BOTTLENECK-001`: the shared spectral bottleneck remains the stronger capped local spectral-family anchor than the original local Hybrid bottleneck.
  - `PDEBENCH-CNS-GNOT-001`: GNOT remains an optional protocol-divergent context row, not a same-contract headline row.
- If a required locked row proves incomplete or unavailable in the current workspace, record a truthful row-level `blocked` or `not_protocol_compatible` outcome after a documented narrow fix attempt. Do not drift to a different contract or proxy row.

## Prerequisite Status

- Satisfied prerequisite from the backlog dependency graph:
  - `2026-04-29-cns-paper-contract-decision` is complete and is now the authority for this item.
- The accepted same-contract headline rows already exist in durable summaries and should be reused by default:
  - `spectral_resnet_bottleneck_base`
    - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
  - `fno_base`
    - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
  - `unet_strong`
    - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse`
  - `author_ffno_cns_base`
    - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- The accepted continuity row also already exists:
  - `hybrid_resnet_cns`
    - run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
- Full-training CNS benchmark promotion remains intentionally out of scope:
  - the approved contract decision chose the capped lane because full-training same-contract promotion is not credible on one RTX 3090 before the 2026-05-04 and 2026-05-06 AOE deadlines
  - this item must preserve that claim boundary rather than weakening it

## Implementation Architecture

- **Audit unit:** inspect the accepted row roots and emit one item-local audit that proves or disproves same-contract parity, row completeness, and downstream usability.
- **Metadata-gap unit:** if the audit finds a missing emitted field that blocks row locking, patch only the narrowest shared reporting or runner surface, then rerun deterministic checks before re-auditing.
- **Row-lock unit:** emit one locked-row manifest and one durable summary that become the CNS row authority for downstream table/figure and package-audit work.
- **Sync unit:** update discoverability and the progress ledger so later selectors and implementation phases can consume the locked CNS contract without reconstructing it from historical summaries.

## Concrete File and Artifact Targets

### Code and Test Surfaces

- Audit and modify only if the row-lock audit proves a real metadata gap:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/run_pdebench_image128_suite.py`
  - `tests/studies/test_pdebench_image128_runner.py`
  - `tests/studies/test_pdebench_cfd_cns_data.py`
  - `tests/studies/test_pdebench_cfd_cns_metrics.py`

### Durable Docs and State

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify `docs/backlog/in_progress/2026-04-29-cns-paper-table-figure-bundle.md` only if the downstream item needs an explicit pointer to newly created row-lock artifacts or statuses
- Modify `docs/findings.md` only if implementation discovers a stable reusable reporting/provenance rule that extends beyond this paper-local lane

### Required Artifacts

- Create artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/`
- Create verification log root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/verification/`
- Create audit outputs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_row_lock_audit.md`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_row_lock_audit.json`
- Create locked-row manifest:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`

## Execution Guardrails

- The selected backlog item’s required deterministic checks remain mandatory and unchanged.
- No expensive training or broad rerun is part of this item by default. The default path is reuse plus audit, not fresh execution.
- If a required field is missing, diagnose whether the problem is:
  - already recoverable from existing run artifacts
  - recoverable via a narrow shared reporting patch and deterministic rerun of tests only
  - unrecoverable without a same-contract rerun
- Any same-contract rerun is exceptional. Before launching it:
  - get green deterministic checks
  - document why existing run roots are insufficient
  - keep the rerun limited to the exact missing required row and exact approved contract
  - use tmux
  - activate `ptycho311`
  - track the exact launched PID
  - avoid duplicate launches into the same output root
  - require both exit code `0` and fresh required artifacts before treating the rerun as complete
- Do not mark the item `BLOCKED` because of a normal import, path, test, or harness failure. Diagnose, fix, and rerun first.
- Reserve `BLOCKED` for missing accepted run roots, unavailable hardware, irrecoverable external-environment dependency, roadmap conflict, user decision required, or a failure that remains unrecoverable after a documented narrow fix attempt.
- Preserve the pointer contract: `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/iterations/1/items/2026-04-29-cns-paper-benchmark-rows/plan-phase/plan_path.txt` must continue to contain only `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/execution_plan.md`.

## Required Deterministic Checks

The selected backlog item requires these unchanged checks:

```bash
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Run this focused selector only if code under the CNS reporting or reference-manifest surfaces changes:

```bash
pytest tests/studies/test_pdebench_image128_runner.py -k 'reference_run_manifest or cross_run_compare or author_ffno' -v
```

Archive passing logs under the active artifact root per `docs/TESTING_GUIDE.md`.

## Task 1: Audit the Accepted CNS Rows and Their Contract Parity

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_row_lock_audit.md`
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_row_lock_audit.json`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_contract_decision.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`
- Audit: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`
- Audit the accepted run roots listed in the prerequisite section

- [ ] Run the required deterministic checks and archive the passing logs before trusting the existing runner/reporting surfaces.
- [ ] Build one audit record per accepted row with these fields:
  - `row_id`
  - `row_role` as `headline` or `continuity`
  - `row_status` target, initially `capped_decision_support`
  - `run_root`
  - dataset path and dataset identity pointers
  - split counts and emitted window counts
  - `history_len`
  - `max_windows_per_trajectory`
  - normalization contract summary
  - loss, optimizer, scheduler, batch size, epoch budget
  - metric family and actual metric values
  - parameter count and runtime
  - invocation/config/git/environment/log/exit-code pointers
  - prediction/source-array/sample-asset pointers needed by downstream table/figure work
- [ ] Confirm that every accepted row matches the approved same-contract lane. Treat any mismatch in official file, split family, `history_len`, normalization, loss, or metric family as a parity failure, not as an allowed variation.
- [ ] Record the explicitly excluded adjacent rows and why they stay out of the headline bundle:
  - `history_len=3` rows
  - `history_len=1` rows
  - GNOT rows
  - repo-local FFNO proxy rows
- [ ] Record whether each accepted row is fully reusable, missing only noncritical convenience artifacts, or missing a required row-lock field.

**Verification**

- [ ] The audit separates accepted headline rows, the continuity row, and excluded adjacent context.
- [ ] The audit explicitly states whether each accepted row remains `capped_decision_support`, becomes `blocked`, or becomes `not_protocol_compatible`.
- [ ] Archived logs exist for the required pytest and compileall checks.

## Task 2: Patch Narrow Metadata Gaps Only If the Audit Requires It

**Files:**
- Modify only if required by the audit:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/pdebench_image128/cfd_cns.py`
  - `scripts/studies/pdebench_image128/run_config.py`
  - `scripts/studies/run_pdebench_image128_suite.py`
  - `tests/studies/test_pdebench_image128_runner.py`
  - `tests/studies/test_pdebench_cfd_cns_data.py`
  - `tests/studies/test_pdebench_cfd_cns_metrics.py`

- [ ] Patch only the minimal surface needed to emit any missing required row-lock field identified in Task 1.
- [ ] Do not broaden the study contract, add new model families, or change the selected row roster while fixing metadata emission.
- [ ] Rerun the focused selector if code changed under reporting, manifest collation, or author-FFNO integration.
- [ ] Rerun the required deterministic checks after every code patch.
- [ ] If a required field still cannot be recovered without a new same-contract run, document the exact blocker in the audit and stop before expanding into broad reruns.

**Verification**

- [ ] Any code patch has a corresponding focused pytest result plus the full required deterministic checks.
- [ ] The post-patch audit shows the missing field resolved, or the unresolved blocker is explicit and narrow.

## Task 3: Emit the Locked CNS Row Manifest and Durable Summary

**Files:**
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_summary.md`

- [ ] Write `cns_paper_locked_rows.json` as the machine-readable authority for downstream CNS table/figure work.
- [ ] Include one manifest entry for each accepted row with at least:
  - `row_id`
  - `row_role`
  - `row_status`
  - `claim_scope`
  - `run_root`
  - contract fields copied from the approved decision
  - headline metrics and per-row parameter/runtime fields
  - provenance pointers
  - source-array or saved-sample asset pointers
  - notes describing any residual caveat that does not invalidate the row
- [ ] Write `pdebench_cns_paper_row_lock_summary.md` so future implementers do not need the raw backlog item or historical summaries. The summary must state:
  - the selected contract verbatim
  - the locked headline rows verbatim
  - `hybrid_resnet_cns` as continuity/support only
  - excluded adjacent rows and why they are excluded
  - any row-level `blocked` or `not_protocol_compatible` outcome
  - the claim boundary that this remains capped decision-support evidence, not full-training benchmark evidence
  - the handoff expectation that the next CNS item builds the table/figure bundle from this manifest and summary
- [ ] Update `paper_evidence_package_design.md` and `pdebench_2d_cfd_cns_summary.md` so they point to the new row-lock summary as the authoritative locked-row surface instead of forcing later work to infer the roster from older summaries.

**Verification**

- [ ] `cns_paper_locked_rows.json` parses successfully and contains exactly the approved headline rows plus the continuity row.
- [ ] The durable summary explicitly states `bounded_capped_decision_support`, the `history_len=2` lane, and the accepted authored-FFNO cutoff outcome.
- [ ] No excluded row is silently promoted into the locked headline roster.

## Task 4: Sync Discoverability and Selector State

**Files:**
- Modify: `docs/index.md`
- Modify: `docs/studies/index.md`
- Modify: `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- Modify `docs/backlog/in_progress/2026-04-29-cns-paper-table-figure-bundle.md` only if the downstream item needs explicit new manifest/summary paths

- [ ] Add discoverability entries for the new CNS row-lock summary where durable project knowledge changed.
- [ ] Update the progress ledger with a selector-relevant decision that the CNS paper rows are now locked under the approved capped contract and that the next CNS lane is the table/figure bundle rather than another row-selection pass.
- [ ] Keep the update narrow. Do not rewrite the roadmap, steering, or unrelated backlog items.
- [ ] Update the downstream CNS table/figure backlog item only if it needs a direct path to the locked-row manifest or durable summary.

**Verification**

- [ ] `docs/index.md` and `docs/studies/index.md` point to the new CNS row-lock summary.
- [ ] The progress ledger records the row-lock completion and the next downstream scope without widening the contract.

## Final Verification and Closeout

- [ ] Run the required deterministic checks one final time after all code/doc updates.
- [ ] Archive the final pytest and compileall logs under `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/verification/`.
- [ ] Run a final manifest sanity script such as:

```bash
python - <<'PY'
import json
from pathlib import Path

manifest = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json")
summary = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_row_lock_summary.md")

data = json.loads(manifest.read_text())
required_rows = {
    "spectral_resnet_bottleneck_base",
    "fno_base",
    "unet_strong",
    "author_ffno_cns_base",
    "hybrid_resnet_cns",
}
actual_rows = {entry["row_id"] for entry in data["rows"]}
if actual_rows != required_rows:
    raise SystemExit(f"unexpected locked rows: {sorted(actual_rows)}")
text = summary.read_text()
required_terms = [
    "bounded_capped_decision_support",
    "history_len=2",
    "spectral_resnet_bottleneck_base",
    "author_ffno_cns_base",
    "hybrid_resnet_cns",
    "capped decision-support",
]
missing = [term for term in required_terms if term not in text]
if missing:
    raise SystemExit(f"summary missing required terms: {missing}")
print("locked CNS row manifest and summary look consistent")
PY
```

- [ ] Confirm the plan pointer file still contains only:

```text
docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/execution_plan.md
```

## Completion Criteria

- The approved CNS contract is carried verbatim into one durable row-lock summary and one machine-readable locked-row manifest.
- The locked headline row roster is exactly `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`, and `author_ffno_cns_base`.
- `hybrid_resnet_cns` is preserved as continuity/support only.
- Any missing row is represented honestly as `blocked` or `not_protocol_compatible` after a documented narrow fix attempt, not by contract drift.
- Deterministic checks are green and logged.
- Downstream CNS table/figure work can proceed from the new summary and manifest without rereading the raw backlog item, steering, roadmap, or older CNS summaries.
