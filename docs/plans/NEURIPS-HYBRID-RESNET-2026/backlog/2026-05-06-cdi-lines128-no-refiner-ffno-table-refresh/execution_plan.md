# CDI Lines128 Four-Block No-Refiner FFNO Table/Figure Refresh Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` (recommended) or `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees.

**Goal:** Refresh the manuscript-facing `lines128` CDI metrics, objective-control table, figure/image inputs, model-config table, efficiency table, and discovery surfaces so any active `FFNO + PINN` or `FFNO + supervised` paper row points to the corrected four-block no-refiner artifacts rather than the historical `fno_cnn_blocks=2` proxy rows.

**Architecture:** Treat this as a pure paper-asset and lineage refresh. Reuse the corrected four-block no-refiner `pinn_ffno` and `supervised_ffno` rerun outputs by lineage, reuse all non-FFNO `lines128` rows by lineage from the completed base bundle plus U-NO extension, regenerate only the paper-local assets that currently consume the stale FFNO proxy rows, and preserve every historical `fno_cnn_blocks=2` row as caveated proxy context rather than deleting or overwriting it.

**Tech Stack:** PATH `python`, `scripts/studies/paper_results_refresh.py`, `scripts/studies/paper_model_config_table.py`, `scripts/studies/paper_efficiency_table.py`, Markdown/JSON/CSV/TeX evidence surfaces, pytest selectors for paper-refresh tooling.

---

## Selected Backlog Objective

- Replace manuscript-facing CDI `FFNO + PINN` and `FFNO + supervised` table/figure lineage with the corrected four-block no-refiner rows:
  - `fno_blocks=4`
  - `fno_modes=12`
  - `fno_width=32`
  - `fno_cnn_blocks=0`
- Regenerate the checked-in CDI paper assets that still encode the historical `FFNO-local proxy` rows as current paper rows.
- Preserve the historical `fno_cnn_blocks=2` rows in evidence indexes as labeled proxy lineage only.
- Keep this as the interim paper refresh only. The later depth-24 wave remains owned by `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`.

## Scope Boundaries

### In Scope

- Consume the completed corrected-row authorities from:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md`
- Rebuild the active CDI paper-local assets that currently depend on the old FFNO proxy lineage:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.{json,csv,tex}`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_pinn_metrics.tex` if the refresh script still emits it from the same source rows
  - current CDI figure/image source manifests or panels produced from `scripts/studies/paper_results_refresh.py` that still read the historical proxy FFNO roots
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.{json,csv,tex}`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.{json,csv,tex}`
- Refresh the durable discovery and paper-evidence surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
- Write an item-local refresh summary and archive verification logs/diffs under an item-local artifact root.

### Explicit Non-Goals

- Do not rerun `pinn_ffno`, `supervised_ffno`, or any non-FFNO model row in this item.
- Do not require, consume, or silently substitute depth-24 FFNO rows.
- Do not mutate the canonical completed non-FFNO `lines128` row metrics or overwrite historical artifact roots.
- Do not expand into multiseed CDI, natural-patch CDI, PDEBench, BRDT, WaveBench, `256x256`, or manuscript-prose work.
- Do not mark the item `BLOCKED` for ordinary test/import/path/serialization failures. Diagnose, narrow-fix, and rerun first. Reserve `BLOCKED` for missing prerequisite artifacts, unavailable hardware/resources, external dependency outside current authority, roadmap conflict, user decision required, or a failure that remains unrecoverable after a documented narrow fix attempt.

## Steering, Roadmap, And Fairness Constraints

- Steering keeps the active window on required paper evidence and forbids optional expansion while required packaging gaps remain. This refresh is allowed because current manuscript-facing CDI assets still depend on stale FFNO proxy rows.
- Roadmap scope remains Phase 3 CDI packaging only. Do not drift into Phase 2 PDE or later depth-24 final-refresh work.
- The `lines128` contract stays fixed. Non-FFNO rows remain reused by lineage from the completed base bundle and U-NO extension; the only row-family replacement in this item is the FFNO pair.
- Historical `fno_cnn_blocks=2` rows must remain visible only as labeled `FFNO-local proxy` context, never as the active paper-facing `FFNO + PINN` / `FFNO + supervised` rows.
- If the refreshed mixed-lineage CDI table payload can no longer honestly use the old table-level authority string, implementation must update the generated payload/schema/tests so the table-level claim boundary explicitly reflects the interim mixed-lineage no-refiner refresh rather than pretending the old six-row bundle itself changed.
- Ordinary refresh commands stay under implementation ownership until terminal success or recoverable failure handling completes. If a refresh command unexpectedly triggers model execution, stop and treat that as a scope defect.

## Prerequisite Status

- The selected backlog item declares two prerequisites:
  - `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`
  - `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`
- Those prerequisite summaries and fresh roots now exist and are the authority for this refresh:
  - corrected `pinn_ffno` root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun/runs/ffno_no_refiner_20260506T223454Z`
  - corrected `supervised_ffno` root:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun/runs/supervised_ffno_no_refiner_20260506T232535Z`
- The reused non-FFNO `lines128` authorities are already pinned and must be consumed by lineage without rediscovery:
  - immutable six-row base summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
    with authoritative root
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
    and claim boundary `complete_lines128_cdi_benchmark`
  - append-only U-NO extension summary:
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
    with authoritative extension root
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z`
    and claim boundary `complete_lines128_cdi_benchmark_plus_uno_extension`
- The consumed `progress_ledger.json` is still useful for high-level initiative status, but it only records early Phases 0-2 and does not enumerate these later Phase 3 CDI prerequisites. For this item, the authoritative prerequisite status is the selected backlog context plus the completed Phase 3 summary documents above.
- Current repo state shows a split surface:
  - `evidence_matrix.md`, `paper_evidence_index.md`, `model_variant_index.json`, and `ablation_index.json` already mention corrected no-refiner evidence.
  - Several generated tables and study-index text still advertise `FFNO-local proxy` as the active paper row.
- This item exists to resolve that split state coherently.

## Implementation Architecture

- **Unit 1: Authority and lineage audit**
  - Freeze the exact corrected FFNO source roots and the reused non-FFNO source roots.
  - Identify every paper-local asset that still resolves `FFNO-local proxy` as the active row.

- **Unit 2: Generator and payload refresh**
  - Update the existing paper-refresh generators so CDI tables, figure/image sources, model-config assets, and efficiency assets all point at the corrected no-refiner pair.
  - Preserve row-level historical proxy lineage as explicit caveated context where discovery surfaces need it.

- **Unit 3: Discovery and summary publication**
  - Regenerate the checked-in outputs, write a durable summary for the refresh, and synchronize the evidence indexes and study index so later manuscript tasks discover one consistent active FFNO story.

## File And Artifact Targets

### Mandatory contract outputs

- Refreshed plan authority:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/execution_plan.md`
- Item-local artifact root for logs, audits, and diffs:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/`
- Regenerated CDI paper assets:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.csv`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.tex`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_pinn_metrics.tex` if `write_cdi_extended_assets()` continues to emit it
- Regenerated CDI figure/image inputs if they currently consume the stale proxy roots:
  - outputs from `--write-cdi-phase-zoom-figure`
  - outputs from `--write-cdi-phase-zoom-per-panel-figure`
  - any checked-in CDI figure metadata or panel manifest emitted by `scripts/studies/paper_results_refresh.py`
- Regenerated packaging tables:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.csv`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.tex`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`
- Mandatory durable discovery updates:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
- Durable refresh summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`

### Likely code / test surfaces if the current generators still source stale FFNO roots

- `scripts/studies/paper_results_refresh.py`
- `scripts/studies/paper_model_config_table.py`
- `scripts/studies/paper_efficiency_table.py`
- `tests/studies/test_paper_results_refresh.py`
- `tests/studies/test_paper_model_config_table.py`
- `tests/studies/test_paper_efficiency_table.py`
- `scripts/studies/paper_evidence_audit.py` only if the refreshed table-level claim-boundary semantics require a narrow audit-tool update

### Documentation / index policy

- `docs/index.md` does not need an update unless implementation creates a new reusable generator, runbook, or authority document that should be discoverable from the hub.
- `docs/findings.md` does not need an update unless the work exposes a durable reusable rule about the paper-refresh tooling rather than just changing current CDI lineages.

## Execution Checklist

### Task 1: Freeze The Source Authorities And Enumerate Stale Consumers

- [ ] Read and record the exact corrected no-refiner FFNO authorities:
  - corrected `pinn_ffno` summary/root
  - corrected `supervised_ffno` summary/root
- [ ] Read and record the reused non-FFNO authorities:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
    -> immutable six-row base root
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
    (`claim_boundary=complete_lines128_cdi_benchmark`)
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`
    -> append-only extension root
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z`
    (`claim_boundary=complete_lines128_cdi_benchmark_plus_uno_extension`)
- [ ] Audit the current generated CDI paper-local assets and record which still present:
  - `FFNO-local proxy + PINN` as the active paper row
  - `FFNO-local proxy + supervised` as the active paper row
  - stale source roots, parameter counts, or runtime provenance from the proxy rows
- [ ] Audit whether the current figure/image outputs generated by `paper_results_refresh.py` still read the historical proxy roots.
- [ ] Decide whether the current refresh code can be reused as-is with only source-root/payload updates, or whether a narrow generator/test change is required.

Verification for Task 1:

- Blocking:
  - confirm both corrected prerequisite summaries and roots exist and are readable
- Supporting:
  - targeted `rg` inventory under `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables`, `model_variant_index.json`, `ablation_index.json`, `evidence_matrix.md`, `paper_evidence_index.md`, and `docs/studies/index.md`
  - item-local audit note listing every stale consumer found

### Task 2: Repair Only The Minimal Paper-Refresh Generators Needed

- [ ] If Task 1 shows the current generators still source historical proxy roots, patch only the minimal owners so the refreshed CDI assets consume:
  - corrected `pinn_ffno`
  - corrected `supervised_ffno`
  - reused non-FFNO rows by lineage
- [ ] Ensure the regenerated `cdi_lines128_metrics_extended.*` payload no longer claims the stale FFNO proxy rows are the active paper rows.
- [ ] Ensure the objective-control table is rebuilt only from the corrected no-refiner pair.
- [ ] Ensure the model-config table uses the corrected row roots, corrected parameter counts, and corrected FFNO caveat text for active CDI rows while still preserving historical proxy context where appropriate.
- [ ] Ensure the efficiency table uses corrected CDI FFNO provenance/runtime sources when those rows appear in the grouped output and keeps BRDT caveats untouched.
- [ ] If the refreshed extended CDI payload becomes a mixed-lineage table rather than the immutable original bundle, update the payload metadata/tests so the table-level claim boundary is explicit and honest.
- [ ] Add or tighten focused tests that prove:
  - corrected CDI FFNO rows are selected as active paper rows
  - historical proxy rows remain discoverable only as caveated context
  - no depth-24 row is required or silently substituted
  - non-FFNO row selection remains unchanged

Verification for Task 2:

- Blocking if code/tests changed:
  - `pytest -q tests/studies/test_paper_results_refresh.py -k "cdi or objective or phase_zoom"`
  - `pytest -q tests/studies/test_paper_model_config_table.py`
  - `pytest -q tests/studies/test_paper_efficiency_table.py`
  - `python -m compileall -q scripts/studies`
- Supporting:
  - any narrower new selector added for refreshed claim-boundary or lineage logic

### Task 3: Regenerate The Paper-Local CDI Tables And Figure Inputs

- [ ] Run the checked-in paper refresh entrypoints needed to regenerate the CDI assets, using PATH `python` from the repo root.
- [ ] Regenerate:
  - extended CDI metrics assets
  - objective-control TeX
  - any emitted `cdi_lines128_pinn_metrics.tex`
  - any checked-in CDI figure/image source panels or manifests that still depended on the historical proxy roots
- [ ] Keep the refresh deterministic and append-only with respect to summaries/logs:
  - archive command outputs under the item-local artifact root
  - do not overwrite historical run roots
  - do not trigger new model runs
- [ ] If a refresh command fails, diagnose and apply one narrow fix/rerun cycle before considering the item unrecoverable.

Verification for Task 3:

- Blocking:
  - regenerated table files exist and are freshly written
  - regenerated CDI figure/image inputs exist if the audit found they were stale consumers
  - active CDI FFNO table rows now resolve the corrected no-refiner roots
- Supporting:
  - `git diff` review for the regenerated TeX/CSV/JSON outputs
  - item-local captured stdout/stderr for the refresh commands

### Task 4: Regenerate Model-Config And Efficiency Packaging

- [ ] Rebuild `model_config_by_benchmark.{json,csv,tex}` from the corrected CDI FFNO sources.
- [ ] Rebuild `paper_efficiency_table.{json,csv,tex}` because the active CDI FFNO parameter/runtime/provenance sources change in this refresh.
- [ ] Ensure active Synthetic CDI FFNO rows in both packaging tables reference the corrected no-refiner roots and corrected parameter counts.
- [ ] Keep historical proxy rows labeled as proxy context where the packaging surfaces intentionally preserve them; do not let them remain the active CDI paper rows.
- [ ] Keep BRDT and other non-CDI benchmark rows unchanged unless the generator logic requires mechanical re-emission.

Verification for Task 4:

- Blocking:
  - regenerated model-config and efficiency JSON/CSV/TeX files exist
  - active Synthetic CDI FFNO rows no longer use the historical proxy label/root where a canonical paper row is intended
- Supporting:
  - targeted structured diff of CDI rows inside the regenerated packaging JSONs

### Task 5: Refresh Discovery Surfaces And Write The Durable Summary

- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`.
- [ ] The summary must record:
  - the objective of the interim refresh
  - the corrected no-refiner `pinn_ffno` and `supervised_ffno` source roots
  - the reused non-FFNO lineage authorities
  - which paper-local assets were regenerated
  - whether the table-level claim-boundary metadata changed
  - explicit confirmation that depth-24 rows were not consumed
  - explicit confirmation that historical `fno_cnn_blocks=2` rows remain proxy context only
- [ ] Update `model_variant_index.json`, `ablation_index.json`, `evidence_matrix.md`, `paper_evidence_index.md`, and `docs/studies/index.md` so they all tell the same active FFNO story after the refresh.
- [ ] If any discovery surface is genuinely non-applicable, record the reason in the summary and execution evidence instead of silently skipping it.

Verification for Task 5:

- Blocking:
  - the summary exists and names the exact corrected source roots
  - discovery surfaces distinguish active no-refiner CDI FFNO evidence from preserved historical proxy lineage
- Supporting:
  - targeted `rg` checks for `FFNO-local proxy`, `FFNO + PINN`, `FFNO + supervised`, `fno_cnn_blocks=2`, and `fno_blocks=24`

## Deterministic Verification Gate

Run these checks unless a stronger replacement is explicitly documented in the execution evidence.

### Required blocking checks after regeneration

- [ ] Backlog-item required JSON validation:

```bash
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json >/tmp/model_variant_index.json.valid
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json >/tmp/ablation_index.json.valid
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json >/tmp/cdi_lines128_metrics_extended.json.valid
```

- [ ] Stronger companion JSON validation for the other regenerated machine-readable outputs:

```bash
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json >/tmp/model_config_by_benchmark.json.valid
python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json >/tmp/paper_efficiency_table.json.valid
```

- [ ] CSV parse smoke check for all regenerated CSV assets:

```bash
python - <<'PY'
import csv
from pathlib import Path

paths = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.csv"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.csv"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv"),
]
for path in paths:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        raise SystemExit(f"{path} has no data rows")
print("csv assets parse")
PY
```

- [ ] Active-row lineage audit that fails if canonical CDI FFNO paper rows still point at the historical proxy lineage:

```bash
if rg -n "FFNO-local proxy \\+ PINN|FFNO-local proxy \\+ supervised|historical fno_cnn_blocks=2 local-refiner proxy; corrected no-refiner" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.* \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.* \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.* \
  docs/studies/index.md; then
  echo "historical proxy FFNO lineage is still exposed as an active CDI paper row" >&2
  exit 1
fi
```

- [ ] Depth guard that fails if this interim refresh consumed a depth-24 CDI FFNO row:

```bash
if rg -n "fno_blocks=24|depth24" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.* \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.* \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.* \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md; then
  echo "depth-24 FFNO content was consumed by the interim no-refiner refresh" >&2
  exit 1
fi
```

### Blocking checks only if generator code changed

- [ ] `pytest -q tests/studies/test_paper_results_refresh.py -k "cdi or objective or phase_zoom"`
- [ ] `pytest -q tests/studies/test_paper_model_config_table.py`
- [ ] `pytest -q tests/studies/test_paper_efficiency_table.py`
- [ ] `python -m compileall -q scripts/studies`

### Supporting checks

- [ ] `git diff -- scripts/studies/paper_results_refresh.py scripts/studies/paper_model_config_table.py scripts/studies/paper_efficiency_table.py tests/studies/test_paper_results_refresh.py tests/studies/test_paper_model_config_table.py tests/studies/test_paper_efficiency_table.py docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.csv docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.tex docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.csv docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.tex docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md docs/studies/index.md docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_no_refiner_ffno_table_refresh_summary.md`
- [ ] Archive the verification logs under `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh/verification/`

## Completion Standard

- Any active manuscript-facing CDI `FFNO + PINN` or `FFNO + supervised` row now points to the corrected four-block no-refiner artifacts.
- Historical `fno_cnn_blocks=2` rows remain preserved and discoverable only as explicitly caveated proxy lineage.
- No depth-24 row was required or silently substituted.
- No non-FFNO row was rerun or re-authored beyond mechanical asset regeneration and discovery-surface synchronization.
