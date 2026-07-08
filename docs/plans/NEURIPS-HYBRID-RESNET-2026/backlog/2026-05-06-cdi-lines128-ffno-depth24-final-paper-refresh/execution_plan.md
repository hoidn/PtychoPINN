# CDI Lines128 FFNO Depth-24 Final Paper Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Do not create worktrees. If a refresh or verification command becomes long-running, keep it under the same implementation owner until exit status and fresh outputs are recorded.

**Goal:** Decide whether the depth-24 no-refiner FFNO pair replaces the current four-block no-refiner FFNO pair in the repo-local final CDI Lines128 paper assets, then regenerate the final FFNO-consuming paper assets and evidence indexes without rerunning or mutating any non-FFNO Lines128 row.

**Architecture:** Treat the current four-block no-refiner refresh as the active baseline until a documented promotion decision is made. First audit the completed depth-24 PINN and supervised companions against the corrected four-block pair under the locked `cdi_lines128_seed3` contract. Then refresh the paper-asset generators so they can emit provenance-safe final outputs from an explicitly chosen same-depth no-refiner FFNO pair while preserving the historical local-refiner proxy rows and the interim four-block no-refiner rows as discoverable lineage context. Reuse all non-FFNO Lines128 rows strictly by lineage.

**Tech Stack:** Python 3.11 via PATH `python`, Markdown/JSON/CSV/TeX/PNG paper assets, existing `scripts/studies/paper_*` refresh tooling, existing Lines128 FFNO summaries and artifact roots, `pytest`.

---

## Selected Backlog Objective

- Regenerate final CDI Lines128 paper-facing metrics, figures/images, model-configuration, efficiency, evidence-index, and manuscript-consumption assets after the depth-24 no-refiner FFNO rows landed.
- Make an explicit promotion decision: either the depth-24 no-refiner pair becomes the active final FFNO pair for paper-local CDI assets, or the depth-24 rows stay append-only depth evidence and the corrected four-block no-refiner pair remains canonical.

## Scope

- Consume only these FFNO-specific authorities:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_depth24_no_refiner_summary.md`
  - their referenced artifact roots under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/...`
- Reuse all non-FFNO Lines128 rows by lineage from the immutable six-row CDI table and the append-only U-NO extension.
- Refresh only repo-local paper assets and discovery/evidence surfaces under `docs/plans/NEURIPS-HYBRID-RESNET-2026/` and `docs/studies/index.md`.

## Explicit Non-Goals

- Do not rerun any non-FFNO Lines128 row.
- Do not edit `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.
- Do not change the locked `cdi_lines128_seed3` dataset/probe/training contract.
- Do not rewrite the immutable six-row CDI authority or the append-only U-NO source root.
- Do not write manuscript prose, update `/home/ollie/Documents/neurips/`, or edit `hybrid_resnet_neurips_first_draft.tex`.
- Do not reopen Phase 2 PDEBench work, Phase 4 scaling, or Phase 5 evidence-bundle work.

## Steering And Roadmap Constraints

- This is Phase 3 CDI packaging work inside the approved roadmap; preserve the roadmap phase boundaries and do not expand into later paper-bundle scope.
- Keep equal-footing reasoning explicit. The active manuscript-facing FFNO objective-control pair must stay a same-depth no-refiner pair; do not silently mix a depth-24 `pinn_ffno` with a depth-4 `supervised_ffno` in the canonical paper pair.
- Historical `fno_cnn_blocks=2` rows must remain labeled `FFNO-local proxy`.
- The corrected four-block no-refiner rows must remain discoverable as interim evidence and as the direct refiner-effect comparison even if depth-24 is promoted.
- Normal test/import/path failures are not grounds to mark the item `BLOCKED`; diagnose, fix, and rerun. Reserve `BLOCKED` for genuinely missing prerequisite artifacts, roadmap-authority conflicts, or external resource gaps that remain unrecoverable after a narrow documented fix attempt.

## Prerequisite Status

- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json` shows the campaign's earlier evidence-inventory and selection tranches are already completed and no project-level blocked tranche is currently recorded.
- For this item, the actual execution prerequisites are the completed Lines128 FFNO authorities above. Treat the following as required completed inputs before any asset regeneration:
  - corrected four-block `pinn_ffno` no-refiner rerun
  - corrected four-block `supervised_ffno` no-refiner rerun
  - completed pure-FFNO depth-24 PINN ablation
  - completed supervised FFNO depth-24 companion
  - existing no-refiner paper-refresh outputs and indexes
- If any required summary or artifact root is missing, first verify whether the path simply moved or the summary is stale; only escalate to `BLOCKED` if the authoritative prerequisite evidence itself cannot be recovered.

## Implementation Architecture

1. `Promotion decision and source selection`
   - Own the depth-4 versus depth-24 evaluation, the rule that the active manuscript-facing FFNO pair must stay same-depth, and the final chosen source roots plus claim boundary.
2. `Paper-asset generator refresh`
   - Own all code paths that currently hardcode active CDI FFNO roots or output names:
     - `scripts/studies/paper_results_refresh.py`
     - `scripts/studies/paper_model_config_table.py`
     - `scripts/studies/paper_efficiency_table.py`
   - Add the smallest possible configurability needed to swap between the corrected four-block pair and the depth-24 pair and to emit provenance-safe final-refresh outputs.
3. `Evidence and discoverability refresh`
   - Own the durable final summary plus updates to:
     - `evidence_matrix.md`
     - `paper_evidence_index.md`
     - `paper_evidence_manifest.json`
     - `model_variant_index.json`
     - `ablation_index.json`
     - `docs/studies/index.md`
   - Preserve lineage for the interim four-block no-refiner rows and the historical local-refiner proxy rows, including the machine-readable CDI visual-bundle references that point at the chosen final FFNO pair's source reconstructions.

## File And Artifact Targets

### Mandatory contract outputs

- Final durable summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`
- Updated durable indexes:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`
- Refreshed CDI FFNO manuscript-consumption asset families sourced from the chosen final no-refiner pair:
  - metrics family
  - objective-control family
  - phase-zoom figure family
  - model-config family
  - efficiency family
- Required canonical JSON surfaces that downstream consumers and deterministic checks already depend on:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`

### Preferred packaging

- Emit provenance-safe versioned copies for every regenerated final-refresh asset so the depth decision is discoverable from the filename, then mirror the chosen final state into the existing canonical manuscript-consumption filenames only after the versioned outputs pass audit.
- Use one consistent decision stem across the versioned outputs, for example:
  - `..._ffno_final_depth4pair...` if the corrected four-block pair stays canonical
  - `..._ffno_final_depth24pair...` if the depth-24 pair is promoted
- If implementation cannot keep the canonical pair same-depth without weakening the objective-control contract, prefer retaining the corrected four-block pair as canonical and keeping depth-24 strictly append-only.

### Item-local execution evidence

- Treat this backlog item's execution-evidence root as fixed:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/`
- Treat this backlog item's verification-log root as fixed:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/`
- Treat this backlog item's machine-readable checks report path as fixed:
  - `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json`
- Do not invent alternate archive locations for pytest, collect-only, compileall, or refresh-command logs. Every mandatory command in this plan must either archive its stdout/stderr under the fixed `verification/` root or be captured into the fixed checks JSON with the log path recorded there.

### Deterministic refresh invocation contract

- Before any final asset regeneration, add the narrowest missing CLI surface to `scripts/studies/paper_results_refresh.py` so implementation can select one approved FFNO pair and one approved versioned-output stem without ad hoc code edits:
  - `--cdi-final-ffno-pair {four_block_no_refiner,depth24_no_refiner}`
  - `--cdi-final-output-stem <ffno_final_depth4pair|ffno_final_depth24pair>`
- The selection flag must drive all active CDI FFNO paper-refresh outputs in one place:
  - `cdi_lines128_metrics_extended.{json,csv,tex}`
  - `cdi_lines128_objective_comparison.tex`
  - `cdi_lines128_pinn_metrics.tex`
  - `cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet*.png`
  - `model_config_by_benchmark.{json,csv,tex}`
  - `paper_efficiency_table.{json,csv,tex}`
- The output-stem flag must cause the refresh to emit provenance-safe versioned copies for every regenerated CDI FFNO asset before any canonical compatibility file is overwritten.
- After Task 1 freezes the promotion decision, implementation must run exactly one refresh command in this shape, replacing the pair/stem values with the chosen decision:

```bash
python scripts/studies/paper_results_refresh.py \
  --cdi-final-ffno-pair <four_block_no_refiner|depth24_no_refiner> \
  --cdi-final-output-stem <ffno_final_depth4pair|ffno_final_depth24pair> \
  --write-cdi-extended-assets \
  --write-cdi-phase-zoom-figure \
  --write-cdi-phase-zoom-per-panel-figure \
  --write-model-config-table \
  --write-efficiency-table \
  2>&1 | tee artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/paper_results_refresh_<ffno_final_depth4pair|ffno_final_depth24pair>.log
```

- The refresh is not complete until the archived log above exists, the chosen versioned outputs exist, and the canonical compatibility files have been audited against the same chosen pair.

### Pytest and archive-evidence contract

- Archive the first passing targeted pytest batch for this item to:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_preflight.log`
- Archive the additional first passing focused suite proofs for the model-config and efficiency writers to:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_model_config_preflight.log`
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_efficiency_preflight.log`
- If code changes after that first passing batch, rerun the same targeted pytest batch against the final repo state and archive it to:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_postfix.log`
- If code changes after the first passing model-config or efficiency proofs, rerun those selectors against the final repo state and archive them to:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_model_config_postfix.log`
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_efficiency_postfix.log`
- Archive the first passing compile smoke to:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/compileall_preflight.log`
- If code changes after that first compile smoke, rerun it against the final repo state and archive it to:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/compileall_postfix.log`
- If this item adds or renames any test module or selector, run `pytest --collect-only` on the affected modules and archive it to:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_collect.log`
- If tests are added or renamed, update both selector registries after the passing collect-only proof exists:
  - `docs/TESTING_GUIDE.md`
  - `docs/development/TEST_SUITE_INDEX.md`
- If implementation expands beyond the paper-refresh scripts/tests into broader reusable production workflow plumbing, rerun `pytest -v -m integration` and archive the passing output to:
  - `artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_integration_postfix.log`
- Record every archived command, exit code, and artifact/log path in `artifacts/checks/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh-checks.json` so implementation review does not have to infer what ran.

### Likely code and test files

- Code:
  - `scripts/studies/paper_results_refresh.py`
  - `scripts/studies/paper_model_config_table.py`
  - `scripts/studies/paper_efficiency_table.py`
  - only if required by the narrowest solution: `scripts/studies/metrics_tables.py`
- Tests:
  - `tests/studies/test_paper_results_refresh.py`
  - `tests/studies/test_paper_model_config_table.py`
  - `tests/studies/test_paper_efficiency_table.py`
  - add a new focused test module only if the existing paper-refresh tests become too awkward to extend

## Task Checklist

### Task 1: Audit prerequisites and freeze the promotion rule

**Files / artifacts**
- Read and compare:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_depth24_no_refiner_summary.md`
  - their artifact roots and comparison payloads
- Record the outcome in the new final summary draft

- [ ] Verify that the corrected four-block pair and both depth-24 companions exist and still report `fno_cnn_blocks=0`; confirm the depth-24 companions report `fno_blocks=24`.
- [ ] Extract the metric and compute deltas that matter for promotion:
  - PINN depth-24 improves all tracked scalar reconstruction metrics but is much more expensive.
  - supervised depth-24 is mixed, improves mostly phase-side metrics, and is explicitly not auto-promotable.
- [ ] Freeze the promotion rule before touching generator code:
  - the active manuscript-facing FFNO pair must remain same-depth;
  - depth-24 promotion must be justified against both accuracy and paper-facing fairness/clarity;
  - if the paired supervised row does not justify promotion, keep the corrected four-block pair canonical and retain depth-24 as labeled append-only evidence.
- [ ] State the chosen final pair explicitly in the new summary draft and carry that same decision through every refreshed asset and index.

**Verification**
- Blocking:
  - Presence audit for the prerequisite summaries, comparison payloads, and run roots.
  - Contract audit readback confirming `depth4 -> fno_blocks=4, fno_cnn_blocks=0` and `depth24 -> fno_blocks=24, fno_cnn_blocks=0`.
- Supporting:
  - If any prerequisite path fails, first repair the stale reference or summary path; do not stop at the first missing relative path without checking whether the artifact moved.

### Task 2: Parameterize the paper-refresh generators around an explicit final FFNO pair

**Files**
- Modify:
  - `scripts/studies/paper_results_refresh.py`
  - `scripts/studies/paper_model_config_table.py`
  - `scripts/studies/paper_efficiency_table.py`
- Test:
  - `tests/studies/test_paper_results_refresh.py`
  - `tests/studies/test_paper_model_config_table.py`
  - `tests/studies/test_paper_efficiency_table.py`

- [ ] Replace hardcoded four-block-only active CDI FFNO roots with a narrow source-selection layer that can choose between:
  - corrected four-block no-refiner pair
  - depth-24 no-refiner pair
- [ ] Add the deterministic selection/output CLI contract described above if it is not already present:
  - `--cdi-final-ffno-pair {four_block_no_refiner,depth24_no_refiner}`
  - `--cdi-final-output-stem <ffno_final_depth4pair|ffno_final_depth24pair>`
- [ ] Keep the selection pair-level for the active manuscript-facing FFNO rows; do not let one active canonical row point to depth 4 while its paired objective-control row points to depth 24.
- [ ] Add provenance fields to the regenerated JSON payloads so each active FFNO row records:
  - chosen root
  - whether it is `four_block_no_refiner` or `depth24_no_refiner`
  - preserved historical proxy lineage where applicable
  - the final-refresh claim boundary
- [ ] Add versioned output-path support keyed by the deterministic output stem so the refresh can emit authoritative provenance-safe assets without losing the earlier interim refresh by filename collision.
- [ ] Preserve the existing non-FFNO lineage behavior: all other Lines128 rows should still come from the immutable six-row base or the U-NO extension by lineage only.
- [ ] Ensure historical `fno_cnn_blocks=2` FFNO rows remain labeled `FFNO-local proxy` in every discovery or lineage surface that still mentions them.

**Verification**
- Blocking:
  - Run and archive the first passing targeted pytest batch:

```bash
pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom' \
  2>&1 | tee artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_preflight.log
pytest -q tests/studies/test_paper_model_config_table.py \
  2>&1 | tee artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_model_config_preflight.log
pytest -q tests/studies/test_paper_efficiency_table.py \
  2>&1 | tee artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_efficiency_preflight.log
```

  - If tests were added or renamed, archive collection proof before closing:

```bash
pytest --collect-only tests/studies/test_paper_results_refresh.py tests/studies/test_paper_model_config_table.py tests/studies/test_paper_efficiency_table.py -q \
  2>&1 | tee artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_collect.log
```
- Supporting:
  - Archive the compile smoke:

```bash
python -m compileall -q scripts/studies \
  > artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/compileall_preflight.log 2>&1
```

  - Add or extend assertions proving:
    - the active canonical FFNO roots are the chosen final pair;
    - proxy rows still render as proxy lineage;
    - versioned outputs use the chosen decision stem;
    - non-FFNO rows continue to resolve to the original lineage roots.

### Task 3: Regenerate final CDI FFNO paper-local assets

**Files / outputs**
- Regenerate versioned authoritative outputs under:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/`
- Refresh the machine-readable CDI visual-bundle manifest coverage in:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- Keep canonical compatibility outputs in sync for downstream consumers

- [ ] Run the paper-refresh generator(s) against the chosen final FFNO pair and produce a full versioned asset bundle for:
  - extended CDI metrics (`json/csv/tex`)
  - CDI objective-control table
  - CDI PINN table
  - CDI phase-zoom figures
  - model-config table (`json/csv/tex`)
  - paper-efficiency table (`json/csv/tex`)
- [ ] Use the deterministic refresh invocation contract from this plan rather than inventing a one-off command or editing constants by hand during execution.
- [ ] Refresh the CDI visual-bundle manifest entry so the chosen final FFNO pair is explicit in machine-readable form:
  - canonical and versioned phase-zoom figure paths
  - chosen `pinn_ffno` and `supervised_ffno` source recon paths
  - preserved non-FFNO source-array lineage roots
  - visible rows, crop metadata, and claim-boundary/depth-decision metadata needed to audit the figure bundle without re-inferring inputs
- [ ] After the versioned bundle passes audit, refresh the canonical manuscript-consumption filenames that current downstream consumers expect.
- [ ] Ensure the regenerated canonical `cdi_lines128_metrics_extended.json` remains truthful about the final-refresh claim boundary and the active FFNO source roots.
- [ ] Do not overwrite the underlying interim artifact roots; only regenerate derived tables/figures and their repo-local metadata.

**Verification**
- Blocking:
  - Rerun and archive the single deterministic refresh command that writes the CDI extended assets, phase-zoom figures, model-config table, and efficiency table.
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json >/tmp/cdi_lines128_metrics_extended.json.valid`
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json >/tmp/model_config_by_benchmark.json.valid`
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json >/tmp/paper_efficiency_table.json.valid`
  - Audit the regenerated JSON outputs to confirm the active `pinn_ffno` and `supervised_ffno` rows both point to the chosen same-depth no-refiner pair and both state the chosen depth explicitly.
  - Audit `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json` to confirm the CDI visual-bundle entry points at the refreshed phase-zoom outputs and the chosen final FFNO source recon paths.
  - Confirm the regenerated assets still reuse the unchanged non-FFNO rows by lineage rather than by rerun.
- Supporting:
  - CSV parse smoke check for regenerated CSV outputs.
  - A lightweight filename/provenance manifest mapping versioned authoritative outputs to canonical compatibility copies.

### Task 4: Refresh durable evidence and study discovery surfaces

**Files**
- Modify:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/studies/index.md`

- [ ] Update the evidence matrix so the current authority row accurately states whether the paper-local FFNO pair stayed four-block or was promoted to depth-24, and so the non-chosen depth remains discoverable as append-only evidence.
- [ ] Update the paper evidence index with a new completed backlog row for this final refresh, including:
  - tier
  - summary authority
  - artifact root or bundle location
  - downstream use
  - explicit supersession or retention relationship to the interim four-block refresh
- [ ] Update `paper_evidence_manifest.json` so the CDI pillar, row-registry, and output-target entries that surface the repo-local FFNO paper refresh point to:
  - the chosen final `pinn_ffno` / `supervised_ffno` roots
  - the refreshed canonical and versioned phase-zoom outputs
  - the preserved source-array lineage for the unchanged non-FFNO rows and historical FFNO proxy context
- [ ] Update `model_variant_index.json` so the active paper-local FFNO variants point at the chosen final roots while preserving separate entries for:
  - historical local-refiner proxies
  - corrected four-block no-refiner rows
  - depth-24 append-only rows
- [ ] Update `ablation_index.json` so the final refresh records the promotion decision and the preserved ablation status of any non-chosen depth family.
- [ ] Update `docs/studies/index.md` so the study catalog points users to the new final-refresh summary as the current paper-local CDI FFNO packaging authority while keeping the interim four-block refresh and both depth-24 studies discoverable as predecessor/append-only context.

**Verification**
- Blocking:
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json >/tmp/paper_evidence_manifest.json.valid`
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/tmp/model_variant_index.json.valid`
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json >/tmp/ablation_index.json.valid`
- Supporting:
  - Manual readback of the updated index entries to confirm terminology is consistent:
    - `FFNO-local proxy` only for historical `fno_cnn_blocks=2`
    - `four-block no-refiner` for the corrected interim pair
    - `depth24 no-refiner` only where the fresh depth-24 rows are explicitly intended

### Task 5: Write the durable final summary and close the refresh

**Files**
- Create/update:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`

- [ ] Write a durable summary that is self-sufficient for later manuscript or review work:
  - objective
  - consumed authorities
  - chosen final FFNO pair and why
  - exact active roots used by the regenerated assets
  - preserved lineage roots for four-block interim evidence and historical proxy context
  - regenerated asset list
  - verification commands and artifact locations
  - residual risks, especially if depth-24 remained append-only because pair-level promotion was not justified
- [ ] Make the summary the authoritative answer to whether final `FFNO + PINN` / `FFNO + supervised` paper rows are four-block or depth-24.
- [ ] State explicitly that non-FFNO rows were reused by lineage and not rerun.

**Verification**
- Blocking:
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json >/tmp/cdi_lines128_metrics_extended.json.valid`
  - Final readback of the summary and canonical asset JSON to ensure they agree on:
    - selected final pair
    - active roots
    - preserved interim four-block discoverability
    - preserved proxy labeling
- Supporting:
  - If the refresh produces an implementation-review note or stale-consumer audit, archive it under the item's `artifacts/work/.../verification/` directory and link it from the summary.

## Required Deterministic Checks

These are mandatory final checks for this backlog item unless a stronger superset is run and recorded:

```bash
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json >/tmp/paper_evidence_manifest.json.valid
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/tmp/model_variant_index.json.valid
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json >/tmp/ablation_index.json.valid
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json >/tmp/cdi_lines128_metrics_extended.json.valid
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json >/tmp/model_config_by_benchmark.json.valid
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json >/tmp/paper_efficiency_table.json.valid
```

Additional required blocking checks for this plan:

```bash
pytest -q tests/studies/test_paper_results_refresh.py -k 'cdi or objective or phase_zoom' 2>&1 | tee artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_preflight.log
pytest -q tests/studies/test_paper_model_config_table.py 2>&1 | tee artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_model_config_preflight.log
pytest -q tests/studies/test_paper_efficiency_table.py 2>&1 | tee artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/pytest_efficiency_preflight.log
```

Supporting checks:

```bash
python -m compileall -q scripts/studies > artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh/verification/compileall_preflight.log 2>&1
```

## Completion Criteria

- The final manuscript-facing CDI FFNO pair is explicitly decided and stays a same-depth no-refiner pair.
- Every final manuscript-facing `FFNO + PINN` or `FFNO + supervised` surface points to the chosen final no-refiner roots and states whether those rows are four-block or depth-24.
- The repo-local CDI visual comparison bundle remains machine-auditable: its manifest coverage names the refreshed figure outputs, chosen FFNO source recon paths, and preserved non-FFNO lineage/source-array roots.
- Historical `fno_cnn_blocks=2` rows remain labeled `FFNO-local proxy`.
- Four-block no-refiner rows remain discoverable as interim evidence and as the direct refiner-effect comparison even if depth-24 is not the active paper pair.
- Non-FFNO Lines128 rows are reused by lineage only.
- The regenerated paper-local CDI assets, durable summary, and all evidence/discovery indexes agree on the chosen final FFNO pair and claim boundary.

## Documentation Follow-Through

- Mandatory:
  - `docs/studies/index.md`
  - NeurIPS evidence indexes listed above
  - the new final summary
- Only if implementation introduces a new reusable repo-level workflow or stable discoverability surface:
  - update `docs/index.md`
- Not required:
  - `/home/ollie/Documents/neurips/index.md`
  - manuscript prose or TeX text edits
