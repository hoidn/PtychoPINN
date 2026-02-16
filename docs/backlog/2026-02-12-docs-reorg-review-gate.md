# Backlog: Docs Reorg Review Gate

**Created:** 2026-02-12  
**Status:** Open  
**Priority:** Medium  
**Related:** `docs/governance/docs_reorg_rubric.md`, `docs/governance/docs_redirect_policy.md`, `docs/governance/docs_reorg_migration_ledger.md`, `docs/governance/docs_reorg_procedure.md`, `scripts/docs/build_docs_inventory.py`, `scripts/docs/score_docs_for_reorg.py`, `scripts/docs/generate_split_map.py`, `scripts/docs/run_docs_reorg_pipeline.py`, `tests/docs/`  
**Source Branch:** `docs-reorg-split-map`

## Problem

A docs reorganization workflow has been implemented on a separate branch and needs a structured review before integration into the working branch.

## Goal

Evaluate whether the new docs-governance workflow improves clarity and maintainability without adding confusion or process debt, then decide to accept (merge/cherry-pick) or reject (with or without follow-up changes).

## Review Scope

1. Governance docs quality and usability.
2. Pipeline tooling correctness and maintainability.
3. Test coverage signal quality (`tests/docs`).
4. Generated artifact usefulness (`docs/analysis/*`).
5. Discoverability impact in `docs/index.md`.

## Review Procedure

1. Inspect candidate commits on `docs-reorg-split-map`.
2. Run verification in the candidate branch:
   - `pytest tests/docs -v`
   - `pytest -q tests/test_grid_lines_compare_wrapper.py::test_wrapper_accepts_architecture_list`
3. Validate sample output freshness:
   - `python scripts/docs/run_docs_reorg_pipeline.py --root docs --out-dir docs/analysis`
4. Spot-check scoring decisions in `docs/analysis/docs_reorg_scores.csv` for obvious false positives/negatives.
5. Confirm `docs/index.md` additions are concise and non-duplicative.

## Decision Matrix

1. **Accept as-is:** Cherry-pick all docs-reorg commits into `fno-stable`.
2. **Accept with changes:** Cherry-pick subset, then patch scoring heuristics/docs text and re-run verification.
3. **Reject with rework request:** Do not merge now; open targeted follow-up backlog items for identified gaps.
4. **Reject fully:** Close this backlog item with rationale if workflow is net-negative.

## Acceptance Criteria

1. `tests/docs` pass on the target branch after integration.
2. New governance docs are understandable in one read and do not duplicate existing guidance heavily.
3. Pipeline outputs are reproducible and useful for triaging large/cluttered docs.
4. No regressions in existing checked baseline test(s).
5. Final decision (accept/reject) is recorded with a short rationale and commit references.

## Notes

Candidate commit range currently starts at `f758d97c` and ends at `d35dd6fb` on `docs-reorg-split-map`.

## Concrete Reorg Proposal (Content-Level)

This is the concrete split/move map to evaluate (not just process scaffolding).

### 1) `docs/index.md` -> hub + focused sub-indexes

1. Keep `docs/index.md` as a short navigation hub only:
   - `Critical Gotchas`
   - `Quick Start`
   - one-paragraph pointers to sub-index files
2. Move current long sections into new files:
   - `docs/index_architecture.md`:
     - `Architecture & Development`
     - `Core Module Documentation`
   - `docs/index_workflows.md`:
     - `Workflows & Scripts`
     - `Command Reference`
   - `docs/index_studies.md`:
     - `Studies & Analysis`
     - `Datasets & Experiments`
   - `docs/index_bugs.md`:
     - `Bug Reports & Fixes`
   - `docs/index_findings_navigation.md`:
     - `Finding Information` (`By Task`, `By User Type`)
3. Add top-nav links in each sub-index back to `docs/index.md`.
4. Leave a stable section anchor list in `docs/index.md` for compatibility.

### 2) `docs/DEVELOPER_GUIDE.md` -> core guide + references

1. Keep `docs/DEVELOPER_GUIDE.md` as the core guide:
   - `Document Purpose`
   - `The Core Concept: A "Two-System" Architecture`
   - `Critical Architectural Principles & Anti-Patterns`
   - `Data Pipeline: Contracts and Bookkeeping` (high-signal subsections only)
   - `Development Methodology & Architectural Principles` (TDD + contract boundaries)
2. Move dense reference sections into dedicated docs:
   - `docs/development/data_pipeline_reference.md`:
     - deep tensor/offset flow from sections `3.4`, `4.5`, `4.6`
   - `docs/development/evaluation_methods.md`:
     - current section `5.*`
   - `docs/development/logging_architecture.md`:
     - current section `6.*`
   - `docs/development/historical_learnings.md`:
     - current section `10.*` and the long historical case study
3. Add "Moved to ..." pointers in original section locations for one release cycle.

### 3) `docs/workflows/pytorch.md` -> landing + task-specific guides

1. Convert `docs/workflows/pytorch.md` into a landing page that links to:
   - `docs/workflows/pytorch_quickstart.md`:
     - current sections `1` to `7` and common workflow examples
   - `docs/workflows/pytorch_cli_flags.md`:
     - current section `12` (`Training Execution Flags`, `Inference Execution Flags`, CONFIG-001 notes)
   - `docs/workflows/pytorch_backend_integration.md`:
     - current section `13` (Ptychodus backend selection/routing/checkpoint compatibility)
   - `docs/workflows/pytorch_parity_and_tests.md`:
     - current sections `11` and `15`
     - troubleshooting entries tied to parity/runtime behavior
2. Keep compatibility redirects in `docs/workflows/pytorch.md` for moved anchors.

### 4) `docs/findings.md` -> normalized index + detail pages

1. Keep `docs/findings.md` as a compact ledger only:
   - one row per finding
   - strict synopsis length cap
   - status + link to detail page
2. Create per-finding detail pages:
   - `docs/findings/details/<FINDING_ID>.md` for long narratives currently embedded inline
   - move long blocks such as `DATA-002`, `PINN-CHUNKED-001`, `TF-NON-XLA-SHAPE-001`, `TF-XLA-BATCH-BROADCAST-001`
3. Create status navigation:
   - `docs/findings/active.md`
   - `docs/findings/resolved.md`
4. Keep old `docs/findings.md` path stable and link to new status pages.

### 5) Batch order for implementation

1. Batch A: split `docs/workflows/pytorch.md` (lowest risk, high payoff).
2. Batch B: split `docs/index.md` into sub-indexes.
3. Batch C: normalize `docs/findings.md` + detail pages.
4. Batch D: split `docs/DEVELOPER_GUIDE.md` last (highest risk due many inbound links).
