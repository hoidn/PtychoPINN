# CNS Matched-Condition Table Refresh Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the PDEBench CNS manuscript authority so the headline table uses exactly one matched condition, choosing the `history_len=5` capped lane only if it is complete and internally consistent, otherwise preserving the locked `history_len=2`, `2048 / 256 / 256` authority.

**Architecture:** This is a deterministic packaging and documentation refresh, not a training item. Reuse the completed CNS authority bundles and summaries, extend the paper-results refresh path to audit and emit matched-condition CNS assets, then synchronize the manuscript and evidence indexes so every discoverability surface states the same selected headline condition and the same unchanged bounded capped claim boundary.

**Tech Stack:** Python 3.11, `scripts/studies/paper_results_refresh.py`, existing CNS bundle/reporting helpers under `scripts/studies/pdebench_image128/`, Markdown, JSON, CSV, TeX, pytest.

---

## Selected Backlog Objective

- Refresh the PDEBench CNS manuscript table so the headline model ranking uses one matched condition instead of best-observed rows drawn from mixed history lengths, caps, or variants.

## Scope

- Consume the completed `history_len=5`, `40`-epoch capped comparator gap-fill evidence for `author_ffno_cns_base`, `spectral_resnet_bottleneck_base` / SRU-Net*, `fno_base`, and `unet_strong`.
- Consume the current locked `history_len=2`, `2048 / 256 / 256`, `40`-epoch capped CNS authority bundle.
- Run a deterministic selection audit that:
  - prefers an all-row `history_len=5`, `512 / 64 / 64`, `40`-epoch headline table only if the four required rows are complete and internally consistent under one contract;
  - otherwise keeps the locked `history_len=2`, `2048 / 256 / 256`, `40`-epoch authority as the headline table.
- Re-emit matched-condition CNS JSON, CSV, and TeX payloads for the selected headline condition.
- Update manuscript-facing CNS table text and any figure references so they no longer present mixed-condition rows as a fair head-to-head ranking.
- Update the durable summary and required evidence-index surfaces.

## Explicit Non-Goals

- Do not rerun CNS training, pilot runs, or figure-generation experiments.
- Do not reopen the CNS paper-contract decision into full-training or paper-grade evidence.
- Do not mix `history_len=5` FFNO/SRU-Net* rows with `history_len=2` FNO/U-Net rows in the headline ranking.
- Do not change model architectures, metric definitions, normalization, or fixed visual samples after seeing metrics.
- Do not silently expand this item into other roadmap phases, `/home/ollie/Documents/neurips/` artifact assembly, or unrelated manuscript edits.

## Binding Steering And Roadmap Constraints

- Steering fairness is binding: the headline CNS table must remain apples-to-apples, and if a fair compare cannot be maintained, the plan must keep the incompatible outcome explicit rather than drifting protocol.
- The roadmap requirement is binding: the headline CNS model-ranking table uses one matched history length, cap, epoch budget, normalization, and metric schema; best-observed mixed-condition rows may survive only as labeled context.
- The capped CNS claim boundary remains `bounded_capped_decision_support_only` unless a later reviewed roadmap update changes it.
- The completed `history_len=5` gap-fill evidence currently remains `adjacent_capped_context_only`; if this item promotes it into the manuscript headline table, the summary must state that the manuscript selection changed while the overall capped claim boundary did not.
- The current `2048 / 256 / 256` bundle remains the larger-cap locked CNS authority unless this item explicitly records a different manuscript-table authority; do not erase its lineage or relabel it as superseded without justification.
- If the chosen headline table does not have same-condition fixed-sample visuals, the manuscript must not present a mismatched figure as if it were the headline same-condition comparison.
- Normal test, import, path, or harness failures are not automatic `BLOCKED` outcomes. Diagnose, apply a narrow fix, and rerun first. Reserve `BLOCKED` for missing authority artifacts, roadmap conflict, unavailable external resources, or unrecoverable failures after a documented narrow fix attempt.

## Prerequisite Status

- `2026-04-29-cns-paper-2048cap-row-extension`: complete in the progress ledger and supplies the current same-contract `2048 / 256 / 256`, `history_len=2`, `40`-epoch bundle plus `cns_paper_table_rows.{json,csv,tex}`.
- `2026-05-04-cns-history5-comparator-gap-fill`: complete in the progress ledger and supplies the missing `history_len=5`, `40`-epoch capped `fno_base` and `unet_strong` rows plus same-history comparison sidecars against authored FFNO and spectral SRU-Net*.
- No additional training prerequisite is open. This backlog item is implementation-local, deterministic refresh work only.

## Implementation Architecture

### 1. Selection Audit

- Extend the paper-results refresh path so it derives the candidate matched conditions from the completed CNS authorities rather than from hard-coded row constants.
- The audit must emit a machine-readable decision payload naming:
  - the selected headline condition;
  - the rejected candidate and exact rejection reason if fallback occurs;
  - the four selected rows, their source roots, and the fixed contract fields;
  - whether same-condition fixed-sample visuals are available.

### 2. CNS Asset Refresh

- Reuse existing CNS bundle/reporting helpers where possible to avoid forking metric formatting or row-label logic.
- Emit a fresh matched-condition table payload bundle for this backlog item, even if the selected condition numerically matches the existing `2048` authority, so the manuscript refresh has its own deterministic selection record.

### 3. Manuscript And Index Synchronization

- Update the manuscript, summary, and evidence indexes from the emitted selection payload so the repo has one consistent statement of the headline CNS table.
- Keep the larger-cap authority, the history-length context evidence, and any retained figure assets clearly labeled by role: headline table, adjacent context, or historical provenance.

## File And Artifact Targets

### Mandatory Contract Outputs

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Create: item-local refresh artifact root under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-matched-condition-table-refresh/` containing at minimum:
  - `matched_condition_decision.json`
  - `cns_paper_table_rows.json`
  - `cns_paper_table_rows.csv`
  - `cns_paper_table_rows.tex`
  - `source_lineage.json`
  - optional `figure_selection.json` when a same-condition figure is selected or explicitly declined
- Modify: `scripts/studies/paper_results_refresh.py`
- Modify or extend only if needed for shared logic:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/studies/index.md`

### Preferred Packaging

- Prefer generating paper-local CNS table assets under `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/`, for example:
  - `pdebench_cns_matched_condition_metrics.tex`
  - `pdebench_cns_matched_condition_metrics.csv`
  - `pdebench_cns_matched_condition_metrics.json`
- Prefer switching the manuscript CNS table to `\input{tables/...}` if that can be done without collateral formatting churn.
- Prefer copying or regenerating a paper-local CNS figure only when the selected matched condition has complete same-condition fixed-sample panels and manifest metadata.

### Conditional Sync Surfaces

- Update `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json` if the manuscript authority path, selected bundle root, or paper-local table asset location changes.
- Update `scripts/studies/paper_evidence_audit.py` only if its default CNS manuscript inputs no longer match the selected headline-table authority after this refresh.
- Update `docs/index.md` only if the new durable summary needs first-class discoverability beyond `paper_evidence_index.md` and `docs/studies/index.md`.

## Execution Tasks

### Task 1: Add A Deterministic Matched-Condition Selection Audit

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Modify: `tests/studies/test_paper_results_refresh.py`

- [ ] Replace the current CNS availability-only audit with a real matched-condition selector that can compare:
  - the locked `history_len=2`, `2048 / 256 / 256`, `40`-epoch authority bundle; and
  - the completed `history_len=5`, `512 / 64 / 64`, `40`-epoch comparator surface.
- [ ] Add or extend CLI flags so the refresh script can:
  - emit an audit/decision JSON without writing paper assets; and
  - write the selected matched-condition CNS assets into the item-local refresh root.
- [ ] Make the selector fail closed on contract mismatches. It must verify, at minimum, for the candidate headline lane:
  - row roster completeness for `author_ffno_cns_base`, `spectral_resnet_bottleneck_base`, `fno_base`, `unet_strong`;
  - one shared `history_len`, split/cap, epoch budget, normalization description, and metric family;
  - preserved capped claim boundary;
  - explicit source lineage to the completed h2 and h5 authorities.
- [ ] Add tests that cover:
  - happy-path selection of the h5 lane when all four rows are present and consistent;
  - fallback to h2 when an h5 row is missing or inconsistent;
  - preservation of best-observed or mixed-condition evidence as context only, not headline ranking metadata.

**Verification:**

- Blocking: `pytest -q tests/studies/test_paper_results_refresh.py -k "cns or matched or history5"`
- Supporting: run the refresh script in audit-only mode and inspect the emitted decision JSON to confirm the chosen lane and fallback reason are deterministic and human-readable.

### Task 2: Emit Refreshed CNS Table Assets From The Selected Lane

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Modify only if reuse is cleaner than local formatting logic:
  - `scripts/studies/pdebench_image128/reporting.py`
  - `scripts/studies/pdebench_image128/cns_paper_bundle.py`
- Modify or add tests under:
  - `tests/studies/test_paper_results_refresh.py`

- [ ] Emit a fresh item-local matched-condition payload bundle with `cns_paper_table_rows.{json,csv,tex}` and `source_lineage.json`.
- [ ] Preserve the manuscript label policy already used on CNS surfaces:
  - `author_ffno_cns_base -> FFNO`
  - `spectral_resnet_bottleneck_base -> SRU-Net*`
  - `fno_base -> FNO`
  - `unet_strong -> U-Net`
- [ ] Record the fixed contract directly in the emitted JSON payload:
  - `history_len`
  - `split_counts`
  - `max_windows_per_trajectory`
  - `epochs`
  - `batch_size`
  - `training_loss`
  - `metric_family`
  - `claim_boundary`
  - `source_summary_paths`
- [ ] If the selected lane has complete same-condition fixed-sample panels and manifest metadata, emit a same-condition figure selection record and preferred paper-local figure asset. If not, emit a negative decision record that keeps the headline table and figure roles separate.
- [ ] Do not mutate the original h2 or h5 source bundles in place unless reuse of an existing paper-local copy is explicitly safer and the summary records that choice.

**Verification:**

- Blocking: run the refresh script in write mode and verify the item-local refresh root contains the required JSON, CSV, and TeX outputs plus a machine-readable decision payload.
- Supporting: if same-condition figure packaging is emitted, verify the row roster and visible labels match the selected headline lane; if figure packaging is declined, verify the decision payload says why.

### Task 3: Refresh The Manuscript CNS Table Authority Without Mixed-Condition Language

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Modify or create preferred paper-local table assets under:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/`
- Modify only if same-condition visuals are selected:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/pdebench_cns_sample000_predictions.png`
  - or a new paper-local CNS figure filename that the summary and manuscript both name explicitly

- [ ] Replace the current “best observed result for each model family” narrative with text that states the selected headline CNS table is one matched condition.
- [ ] Remove the stale `h5_update_pending` comment and any wording that invites mixed-condition ranking.
- [ ] Keep the manuscript self-consistent with the selected lane:
  - if h5 is selected, describe it as the matched `history_len=5`, `512 / 64 / 64`, `40`-epoch capped table and keep h2 2048 as larger-cap bounded context only;
  - if h2 is retained, describe it as the matched `history_len=2`, `2048 / 256 / 256`, `40`-epoch capped table and keep h5 as adjacent same-history context only.
- [ ] Only let the manuscript figure claim same-condition support when the chosen figure actually comes from the selected headline lane. Otherwise label the figure as adjacent context or leave the existing figure untouched outside the headline comparison claim.
- [ ] Prefer switching the CNS table body to a generated paper-local TeX asset if that reduces future drift between code-emitted metrics and manuscript text.

**Verification:**

- Blocking: `rg -n "best observed result for each model family|h5_update_pending|not a matched head-to-head benchmark" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex` must show those stale phrases are removed or intentionally rewritten.
- Supporting: inspect the rendered table source or generated TeX to confirm the four headline rows and condition labels match the selected decision payload exactly.

### Task 4: Synchronize Durable Summary And Discoverability Surfaces

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- Modify: `docs/studies/index.md`
- Modify if required by the chosen authority path:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
  - `scripts/studies/paper_evidence_audit.py`

- [ ] Write the durable summary so it names:
  - the selected headline condition;
  - the rejected candidate lane and reason if fallback occurred;
  - the exact source authorities consumed;
  - the unchanged capped claim boundary;
  - whether the figure remained on the h2 bundle, moved to a same-condition lane, or was intentionally left as context;
  - the emitted item-local refresh root and verification logs.
- [ ] Update `paper_evidence_index.md` so the CNS manuscript-incorporation row points at the matched-condition refresh summary while preserving the 2048cap extension row and the h5 gap-fill row as distinct discoverability surfaces.
- [ ] Update `evidence_matrix.md` so the manuscript incorporation map names the matched-condition CNS table source and no longer implies a best-observed mixed-condition headline.
- [ ] Update `model_variant_index.json` minimally but explicitly so the four selected row entries carry current manuscript-headline lineage or equivalent discoverability metadata; do not rewrite unrelated schemas or rows.
- [ ] Update `docs/studies/index.md` so it points readers to the new matched-condition refresh summary as the discoverable source for the current manuscript CNS table decision.
- [ ] If the selected manuscript authority path changes any audit inputs, update `paper_evidence_manifest.json` and rerun the paper-evidence audit instead of leaving the manifest stale.

**Verification:**

- Blocking when `paper_evidence_manifest.json` or `scripts/studies/paper_evidence_audit.py` changes: `python scripts/studies/paper_evidence_audit.py --repo-root .`
- Supporting otherwise: `rg -n "matched-condition|history_len=5|history_len=2|best_observed" docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md docs/studies/index.md`

## Required Deterministic Checks

These checks are mandatory final gates for this backlog item. They may be supplemented by narrower selectors during implementation, but they are not dropped unless a stronger replacement is explicitly justified in the execution report.

### Final Blocking Checks

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/backlog/done/2026-04-29-cns-paper-2048cap-row-extension.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history5_comparator_gap_fill_summary.md"),
    Path("scripts/studies/paper_results_refresh.py"),
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise SystemExit(f"missing CNS matched-table inputs: {missing}")
print("CNS matched-table inputs present")
PY
pytest -q tests/studies/test_paper_results_refresh.py tests/studies/test_pdebench_cfd_cns_metrics.py
python -m compileall -q scripts/studies
```

### Conditional Blocking Check

- If this item changes `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json` or the assumptions in `scripts/studies/paper_evidence_audit.py`, also run:

```bash
python scripts/studies/paper_evidence_audit.py --repo-root .
```

## Completion Criteria

- A single matched-condition CNS headline table is selected and documented.
- Fresh matched-condition CNS JSON, CSV, and TeX payloads exist in the item-local refresh root.
- The manuscript no longer presents mixed-condition best-observed rows as the headline CNS ranking.
- The summary, paper evidence index, evidence matrix, model variant index, and study index all agree on the same selected CNS headline condition and claim boundary.
- No CNS training run is launched, and no later roadmap phase is implicitly pulled into scope.
