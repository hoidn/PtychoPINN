# Paper Results Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refresh the NeurIPS result package so BRDT candidate outputs are paper-usable, CNS result tables consistently target `history_len=5`, U-NO appears in the relevant CDI tables, and CDI tables report direct error metrics beyond SSIM.

**Architecture:** Treat completed artifact roots as immutable inputs and build paper-local table/figure bundles from them. Do not rerun existing completed rows; when a requested row does not exist under the target contract, create a concrete backlog item that fills only that gap. Manuscript edits must cite generated table/figure files and preserve claim boundaries rather than embedding planning language.

**Tech Stack:** Python 3.11 in `ptycho311`, PyTorch artifact JSON/CSV/NPY roots, repo-local table helpers under `scripts/studies/`, LaTeX manuscript draft under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`, pytest, compileall, and paper-local PNG/TeX assets.

---

## Governing Inputs

- Manuscript draft:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Paper style guide:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md`
- Evidence audit:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- Evidence index:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Evidence matrix:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Model variant index:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- BRDT summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`
- BRDT completed artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
- CNS h5 spectral authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_spectral_history_len4plus_compare_summary.md`
- CNS h5 authored-FFNO authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_history_length_summary.md`
- Current CNS 2048 h2 authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_2048cap_extension_summary.md`
- CDI complete-table authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
- U-NO extension authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`

## Current Facts To Preserve

- The active NeurIPS manuscript draft already exists at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`.
- The active draft package zip already exists at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`.
- `/home/ollie/Documents/neurips` does not exist and must not be introduced as
  a second manuscript target in this plan.
- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex` is the older overlap-free PtychoPINN paper and must not receive these NeurIPS-specific results.
- BRDT is completed only as `decision_support_preflight_only`; it may be formatted for potential paper use, but not promoted as a manuscript pillar without a later evidence amendment.
- CNS `history_len=5` exists for:
  - `spectral_resnet_bottleneck_base` at 40 epochs;
  - `author_ffno_cns_base` at 40 epochs.
- CNS `history_len=5` is not yet available for the main-table comparator rows:
  - `fno_base`;
  - `unet_strong`.
- U-NO exists for the CDI Lines128 contract only:
  - `pinn_neuralop_uno`;
  - `supervised_neuralop_uno`.
- CDI row metrics already include `mse`; `rmse` should be derived as `sqrt(mse)` and written as a display-table metric without changing the underlying training artifacts.

## File Structure

- Create:
  `scripts/studies/paper_results_refresh.py`
  - Small orchestration/helper module for generating paper-local BRDT and CDI display tables from completed artifacts.
  - It must read existing metrics/manifest files and write derived table/figure assets only.
- Create:
  `tests/studies/test_paper_results_refresh.py`
  - Focused unit tests for BRDT table rendering, CDI RMSE derivation, U-NO row inclusion, and fail-closed CNS h5 row-gap detection.
- Create:
  `docs/backlog/active/2026-05-04-cns-history5-comparator-gap-fill.md`
  - Backlog item to run only missing `history_len=5`, 40-epoch CNS comparator rows needed before all CNS result tables can consistently use h5.
- Create or update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.{csv,tex,json}`
  - Paper-local BRDT candidate table based on the completed four-row preflight.
- Create or update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_decision_support_recon.png`
  - Paper-local copy or composed figure from the completed BRDT fixed-sample visual bundle.
- Create or update:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.{csv,tex,json}`
  - CDI Lines128 table including U-NO rows and columns for MAE, MSE, RMSE, and SSIM.
- Modify:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - Replace hand-maintained CDI table values with the generated extended table.
  - Add U-NO rows to the CDI table.
  - Add BRDT candidate table/figure in an appendix or candidate-results subsection.
  - Keep CNS h5 changes gated until the h5 comparator gap item completes.
- Modify:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - Add links to generated BRDT candidate table/figure assets and the CDI extended table.
  - Add a note that CNS h5 main-table promotion is pending `2026-05-04-cns-history5-comparator-gap-fill`.
- Modify:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - Add generated paper-asset rows without changing the scientific authority rows.
- Modify if generated asset paths are indexed there:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - Add paper-local table asset references only if the existing schema already has an appropriate artifact/reference field.

---

### Task 1: Preflight Artifact Audit And CNS h5 Gap Detection

**Files:**
- Create: `scripts/studies/paper_results_refresh.py`
- Create: `tests/studies/test_paper_results_refresh.py`

- [ ] **Step 1: Write the failing CNS h5 gap test**

Add this test to `tests/studies/test_paper_results_refresh.py`:

```python
from scripts.studies.paper_results_refresh import detect_cns_history5_gaps


def test_detect_cns_history5_gaps_reports_missing_main_comparators(tmp_path):
    available = {
        "author_ffno_cns_base": {"history_len": 5, "epochs": 40},
        "spectral_resnet_bottleneck_base": {"history_len": 5, "epochs": 40},
    }
    required = [
        "author_ffno_cns_base",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
    ]

    gaps = detect_cns_history5_gaps(available, required_rows=required)

    assert gaps == ["fno_base", "unet_strong"]
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_detect_cns_history5_gaps_reports_missing_main_comparators
```

Expected: fail with `ImportError` or missing function.

- [ ] **Step 3: Implement the minimal gap detector**

Add to `scripts/studies/paper_results_refresh.py`:

```python
from __future__ import annotations

from typing import Mapping, Sequence


def detect_cns_history5_gaps(
    available_rows: Mapping[str, Mapping[str, object]],
    *,
    required_rows: Sequence[str],
    history_len: int = 5,
    epochs: int = 40,
) -> list[str]:
    gaps: list[str] = []
    for row_id in required_rows:
        row = dict(available_rows.get(row_id, {}))
        if int(row.get("history_len", -1)) != history_len or int(row.get("epochs", -1)) != epochs:
            gaps.append(row_id)
    return gaps
```

- [ ] **Step 4: Run the test again**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_detect_cns_history5_gaps_reports_missing_main_comparators
```

Expected: pass.

- [ ] **Step 5: Add a real-artifact audit command**

Add a CLI mode to `scripts/studies/paper_results_refresh.py`:

```python
def audit_cns_history5_availability() -> dict[str, object]:
    available = {
        "author_ffno_cns_base": {"history_len": 5, "epochs": 40},
        "spectral_resnet_bottleneck_base": {"history_len": 5, "epochs": 40},
    }
    required = [
        "author_ffno_cns_base",
        "spectral_resnet_bottleneck_base",
        "fno_base",
        "unet_strong",
    ]
    gaps = detect_cns_history5_gaps(available, required_rows=required)
    return {"required_rows": required, "available_rows": available, "missing_rows": gaps}
```

The first version may hard-code the known completed h5 authorities. Do not invent h5 rows by scanning unrelated h2 or h3 artifacts.

- [ ] **Step 6: Commit**

```bash
git add scripts/studies/paper_results_refresh.py tests/studies/test_paper_results_refresh.py
git commit -m "plan support: add paper result refresh audit helper"
```

### Task 2: Add The CNS h5 Comparator Gap Backlog Item

**Files:**
- Create: `docs/backlog/active/2026-05-04-cns-history5-comparator-gap-fill.md`
- Modify: `docs/backlog/index.md`

- [ ] **Step 1: Create the backlog item**

Create `docs/backlog/active/2026-05-04-cns-history5-comparator-gap-fill.md`:

```markdown
---
priority: 30
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cns-history5-comparator-gap-fill/execution_plan.md
check_commands:
  - pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
  - python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
prerequisites:
  - 2026-05-01-cns-author-ffno-history-length-study
  - 2026-04-29-cns-spectral-history-len4plus-compare
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The manuscript owner wants CNS result tables to use history_len=5 where possible.
  - history_len=5 rows already exist for authored FFNO and spectral SRU-Net*, but not for FNO or U-Net comparators.
  - This item fills only the missing comparator gaps; it must not rerun completed h5 rows.
---

# Backlog Item: Fill CNS History-Len-5 Comparator Gaps

## Objective

- Run `history_len=5`, `40`-epoch PDEBench CNS comparator rows for `fno_base`
  and `unet_strong`, using the same capped local CNS contract family as the
  completed authored-FFNO and spectral SRU-Net* h5 studies.

## Scope

- Reuse the official `2d_cfd_cns` dataset and task-local CNS runner.
- Match the completed h5 studies on:
  - `history_len=5`;
  - `epochs=40`;
  - capped split `512 / 64 / 64`;
  - `max_windows_per_trajectory=8`;
  - emitted windows `4096 / 512 / 512`;
  - train-only normalization;
  - MSE loss;
  - metric family `err_RMSE`, `err_nRMSE`, `relative_l2`, `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`.
- Produce compare sidecars against each model's locked h2 row and against the h5 FFNO/SRU-Net* authorities.
- Update `pdebench_cns_history5_comparator_gap_fill_summary.md`, `paper_evidence_index.md`, `evidence_matrix.md`, and `model_variant_index.json`.

## Notes for Reviewer

- Do not rerun `author_ffno_cns_base` or `spectral_resnet_bottleneck_base`; consume the completed h5 authorities.
- Do not update the manuscript headline CNS table until both missing comparator rows are completed or explicitly blocked.
- If a row cannot complete, emit a row-level blocker and keep the current h2/2048 table as the active CNS authority.
```

- [ ] **Step 2: Update the backlog index**

Add one row to `docs/backlog/index.md` under the NeurIPS active backlog group, preserving the file's existing table format:

```markdown
| `2026-05-04-cns-history5-comparator-gap-fill` | active | Phase 2 CNS | Runs missing h5 FNO and U-Net comparator rows so h5 CNS result tables can be assembled without mixing history lengths. |
```

- [ ] **Step 3: Verify the backlog item is discoverable**

Run:

```bash
rg -n "2026-05-04-cns-history5-comparator-gap-fill|history_len=5|fno_base|unet_strong" docs/backlog/active docs/backlog/index.md
```

Expected: the active item and index row are found.

- [ ] **Step 4: Commit**

```bash
git add docs/backlog/active/2026-05-04-cns-history5-comparator-gap-fill.md docs/backlog/index.md
git commit -m "backlog: add CNS history5 comparator gap fill"
```

### Task 3: Generate BRDT Candidate Metrics Table And Reconstruction Figure

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Modify: `tests/studies/test_paper_results_refresh.py`
- Create:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.{csv,tex,json}`
- Create:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_decision_support_recon.png`

- [ ] **Step 1: Write a failing BRDT table-rendering test**

Add:

```python
from scripts.studies.paper_results_refresh import render_brdt_metrics_table


def test_render_brdt_metrics_table_keeps_blocked_classical_row():
    metrics = {
        "rows": [
            {"row_id": "classical_born_backprop", "row_status": "blocked"},
            {
                "row_id": "hybrid_resnet",
                "row_status": "completed",
                "image_relative_l2_phys": 0.319,
                "meas_relative_l2": 0.1992,
                "psnr_phys": 29.741,
                "ssim_phys": 0.9471,
            },
        ]
    }

    tex = render_brdt_metrics_table(metrics)

    assert "Classical Born backprop" in tex
    assert "blocked" in tex
    assert "Hybrid ResNet" in tex
    assert "0.319" in tex
    assert "0.199" in tex
```

- [ ] **Step 2: Run the failing test**

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_render_brdt_metrics_table_keeps_blocked_classical_row
```

Expected: fail with missing function.

- [ ] **Step 3: Implement BRDT table rendering**

Implement:

```python
BRDT_LABELS = {
    "classical_born_backprop": "Classical Born backprop",
    "unet": "U-Net",
    "fno_vanilla": "FNO",
    "hybrid_resnet": "Hybrid ResNet",
}


def render_brdt_metrics_table(metrics_payload: dict) -> str:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Row & Image rel. $L_2$ $\downarrow$ & Meas. rel. $L_2$ $\downarrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & Status \\",
        r"\midrule",
    ]
    for row in metrics_payload["rows"]:
        row_id = str(row["row_id"])
        status = str(row.get("row_status", row.get("status", "")))
        label = BRDT_LABELS.get(row_id, row_id.replace("_", r"\_"))
        if status == "blocked":
            lines.append(f"{label} & -- & -- & -- & -- & blocked \\\\")
            continue
        lines.append(
            f"{label} & {float(row['image_relative_l2_phys']):.3f} & "
            f"{float(row['meas_relative_l2']):.3f} & "
            f"{float(row['psnr_phys']):.2f} & "
            f"{float(row['ssim_phys']):.3f} & {status} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Add a CLI mode to write BRDT paper assets**

Implement `--write-brdt-assets` that:

1. reads `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/metrics.json`;
2. writes `tables/brdt_decision_support_metrics.tex`;
3. copies `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/visuals/brdt_compare_q.png` to `figures/brdt_decision_support_recon.png`;
4. writes a small JSON manifest recording source paths and claim boundary `decision_support_preflight_only`.

- [ ] **Step 5: Run the BRDT asset generation**

```bash
python scripts/studies/paper_results_refresh.py --write-brdt-assets
```

Expected files:

```text
docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.tex
docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.csv
docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.json
docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_decision_support_recon.png
```

- [ ] **Step 6: Verify generated content**

```bash
rg -n "Hybrid ResNet|Classical Born backprop|blocked|decision_support_preflight_only" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.*
test -s docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_decision_support_recon.png
```

Expected: table has all BRDT rows, classical row remains blocked, figure exists.

- [ ] **Step 7: Commit**

```bash
git add scripts/studies/paper_results_refresh.py tests/studies/test_paper_results_refresh.py \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.* \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_decision_support_recon.png
git commit -m "paper: generate BRDT candidate table and figure"
```

### Task 4: Generate Extended CDI Metrics Table With U-NO, MSE, And RMSE

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Modify: `tests/studies/test_paper_results_refresh.py`
- Create:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.{csv,tex,json}`

- [ ] **Step 1: Write a failing RMSE derivation test**

Add:

```python
from scripts.studies.paper_results_refresh import cdi_display_metrics


def test_cdi_display_metrics_derives_rmse_and_keeps_uno_rows():
    metrics = {
        "pinn_hybrid_resnet": {"metrics": {"mae": [0.0269, 0.0721], "mse": [0.0016, 0.0081], "ssim": [0.9881, 0.9947]}},
        "pinn_neuralop_uno": {"metrics": {"mae": [0.0932, 0.0683], "mse": [0.0121, 0.0049], "ssim": [0.8280, 0.9569]}},
    }

    rows = cdi_display_metrics(metrics)

    row_by_id = {row["row_id"]: row for row in rows}
    assert row_by_id["pinn_neuralop_uno"]["amp_rmse"] == 0.11
    assert row_by_id["pinn_hybrid_resnet"]["phase_rmse"] == 0.09
```

- [ ] **Step 2: Run the failing test**

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_cdi_display_metrics_derives_rmse_and_keeps_uno_rows
```

Expected: fail with missing function.

- [ ] **Step 3: Implement CDI metric extraction**

Implement:

```python
from math import sqrt


CDI_ROW_ORDER = [
    "baseline",
    "pinn",
    "pinn_fno_vanilla",
    "pinn_ffno",
    "pinn_hybrid_resnet",
    "pinn_spectral_resnet_bottleneck_net",
    "pinn_neuralop_uno",
    "supervised_neuralop_uno",
]

CDI_LABELS = {
    "baseline": ("CNN", "supervised"),
    "pinn": ("CNN", "PINN"),
    "pinn_fno_vanilla": ("FNO", "PINN"),
    "pinn_ffno": ("FFNO", "PINN"),
    "pinn_hybrid_resnet": ("SRU-Net", "PINN"),
    "pinn_spectral_resnet_bottleneck_net": ("Spectral SRU-Net", "PINN"),
    "pinn_neuralop_uno": ("U-NO", "PINN"),
    "supervised_neuralop_uno": ("U-NO", "supervised"),
}


def _pair(payload: dict, key: str) -> tuple[float, float]:
    raw = payload["metrics"][key]
    return float(raw[0]), float(raw[1])


def cdi_display_metrics(metrics_payload: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row_id in CDI_ROW_ORDER:
        if row_id not in metrics_payload:
            continue
        payload = dict(metrics_payload[row_id])
        amp_mae, phase_mae = _pair(payload, "mae")
        amp_mse, phase_mse = _pair(payload, "mse")
        amp_ssim, phase_ssim = _pair(payload, "ssim")
        model, training = CDI_LABELS[row_id]
        rows.append(
            {
                "row_id": row_id,
                "model": model,
                "training": training,
                "amp_mae": amp_mae,
                "phase_mae": phase_mae,
                "amp_mse": amp_mse,
                "phase_mse": phase_mse,
                "amp_rmse": sqrt(amp_mse),
                "phase_rmse": sqrt(phase_mse),
                "amp_ssim": amp_ssim,
                "phase_ssim": phase_ssim,
            }
        )
    return rows
```

- [ ] **Step 4: Implement CDI extended table rendering**

Render a compact table with one row per model:

```latex
\begin{tabular}{llrrrrrrrr}
\toprule
Model & Training & Amp MAE & Phase MAE & Amp MSE & Phase MSE & Amp RMSE & Phase RMSE & Amp SSIM & Phase SSIM \\
\midrule
...
\bottomrule
\end{tabular}
```

Use `\scriptsize` in the manuscript rather than dropping requested metrics.

- [ ] **Step 5: Add the CLI mode**

Implement `--write-cdi-extended-assets` that reads:

```text
.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z/metrics.json
```

and writes:

```text
docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json
docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.csv
docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.tex
```

- [ ] **Step 6: Generate and verify the table**

```bash
python scripts/studies/paper_results_refresh.py --write-cdi-extended-assets
rg -n "U-NO|RMSE|MSE|SRU-Net|Spectral SRU-Net" docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.tex
```

Expected: both U-NO rows and MSE/RMSE columns are present.

- [ ] **Step 7: Commit**

```bash
git add scripts/studies/paper_results_refresh.py tests/studies/test_paper_results_refresh.py \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.*
git commit -m "paper: add extended CDI metrics table with U-NO"
```

### Task 5: Update The Existing Manuscript Draft

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Replace the CDI table body with the generated extended table**

Use:

```latex
% table_metadata: task=CDI_lines128; authority=complete_lines128_cdi_benchmark_plus_uno_extension; source=docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.tex; metrics=mae,mse,rmse,ssim; includes_uno=true
\begin{table}[t]
\centering
\scriptsize
\caption{\textbf{Matched Lines128 CDI benchmark with direct error metrics.} All rows use the same $N=128$ line-pattern protocol. RMSE is derived as $\sqrt{\mathrm{MSE}}$ from the recorded MSE. Lower MAE, MSE, and RMSE are better; higher SSIM is better.}
\label{tab:cdi_lines128_complete}
\input{tables/cdi_lines128_metrics_extended.tex}
\end{table}
```

Expected: U-NO is table-visible, and CDI metrics include MSE/RMSE.

- [ ] **Step 2: Add BRDT as candidate/supplemental paper material**

Place in an appendix or a short "Candidate inverse-scattering extension" subsection:

```latex
% table_metadata: task=BRDT; authority=decision_support_preflight_only; source=docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_decision_support_metrics.tex; candidate_only=true
\begin{table}[t]
\centering
\small
\caption{\textbf{BRDT candidate preflight metrics.} Rows use the completed bounded BRDT decision-support preflight. The classical Born row is shown as blocked because the ODTbrain reference inversion is unavailable in the local environment. These values are candidate evidence, not a manuscript-pillar benchmark.}
\label{tab:brdt_candidate_metrics}
\input{tables/brdt_decision_support_metrics.tex}
\end{table}

% figure_metadata: task=BRDT; authority=decision_support_preflight_only; source=.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/visuals/brdt_compare_q.png; paper_copy=figures/brdt_decision_support_recon.png; candidate_only=true
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/brdt_decision_support_recon.png}
\caption{\textbf{BRDT candidate reconstruction comparison.} Fixed-sample physical-$q$ reconstructions from the completed bounded BRDT preflight.}
\label{fig:brdt_candidate_recon}
\end{figure}
```

Expected: BRDT is available for potential paper use while remaining clearly candidate-only.

- [ ] **Step 3: Do not change the CNS main table to h5 yet**

Leave the current CNS headline table unchanged until `2026-05-04-cns-history5-comparator-gap-fill` completes. Add a LaTeX comment next to the CNS table metadata:

```latex
% h5_update_pending: h5 rows exist for author_ffno_cns_base and spectral_resnet_bottleneck_base; fno_base and unet_strong are queued in docs/backlog/active/2026-05-04-cns-history5-comparator-gap-fill.md. Do not mix h5 and h2 rows in the headline CNS table.
```

Expected: the draft records the h5 intent without creating a mixed-history table.

- [ ] **Step 4: Remove internally-facing prose**

Run:

```bash
rg -n "now includes|paper-ready|paper-grade|decision-support|claim boundary|backlog|artifact root|authority|candidate evidence" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: only metadata comments may contain artifact/authority wording. Visible prose should say "candidate preflight" or "bounded preflight" only where needed for reader-facing clarity.

- [ ] **Step 5: Commit**

```bash
git add docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
git commit -m "paper: refresh result tables and candidate BRDT presentation"
```

### Task 6: Update Evidence Discoverability

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Modify if schema-compatible: `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`

- [ ] **Step 1: Add generated paper-asset pointers**

Add entries for:

```text
tables/cdi_lines128_metrics_extended.{tex,csv,json}
tables/brdt_decision_support_metrics.{tex,csv,json}
figures/brdt_decision_support_recon.png
```

Expected: a reader starting from `paper_evidence_index.md` can find the new paper-local assets.

- [ ] **Step 2: Record the CNS h5 gap item**

Add a short line:

```markdown
The all-h5 CNS headline table is pending `2026-05-04-cns-history5-comparator-gap-fill`, because h5 rows currently exist for authored FFNO and spectral SRU-Net* but not yet for FNO or U-Net.
```

- [ ] **Step 3: Verify discoverability**

Run:

```bash
rg -n "cdi_lines128_metrics_extended|brdt_decision_support|2026-05-04-cns-history5-comparator-gap-fill" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json
```

Expected: all generated assets and the CNS h5 gap item are discoverable.

- [ ] **Step 4: Commit**

```bash
git add docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json
git commit -m "docs: index refreshed paper result assets"
```

### Task 7: Verification And Existing Package Rebuild

**Files:**
- Read/verify generated assets and manuscript package.
- Modify after manuscript/table/figure verification:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`

- [ ] **Step 1: Run focused tests**

```bash
pytest -q tests/studies/test_paper_results_refresh.py
```

Expected: all tests pass.

- [ ] **Step 2: Run adjacent existing tests**

```bash
pytest -q tests/studies/test_metrics_tables.py tests/studies/test_lines128_paper_benchmark.py -k "uno or metrics or table"
```

Expected: pass. If selector collects too broadly or no tests match, rerun the nearest focused table-generation tests and record the exact outcome.

- [ ] **Step 3: Run compile checks**

```bash
python -m compileall -q scripts/studies scripts/studies/born_rytov_dt scripts/studies/pdebench_image128
```

Expected: exit 0.

- [ ] **Step 4: Compile the manuscript**

From `docs/plans/NEURIPS-HYBRID-RESNET-2026/`:

```bash
pdflatex -interaction=nonstopmode hybrid_resnet_neurips_first_draft.tex
```

Expected: PDF builds. Warnings are acceptable only if they are pre-existing citation/reference warnings and no figure/table path is missing.

- [ ] **Step 5: Inspect generated TeX for reader-facing wording**

```bash
rg -n "now includes|paper-ready|paper-grade|decision-support|claim boundary|artifact root|backlog item|authority" hybrid_resnet_neurips_first_draft.tex
```

Expected: no visible manuscript prose uses internal process wording. Metadata comments may contain source paths.

- [ ] **Step 6: Rebuild the existing draft package zip**

From `docs/plans/NEURIPS-HYBRID-RESNET-2026/`, rebuild the existing package
after the TeX and paper-local assets are verified:

```bash
zip -r scr_ptychography_neurips_draft_package.zip hybrid_resnet_neurips_first_draft.tex figures tables references.bib
```

Expected: the zip contains the same `.tex` and generated assets just verified.

- [ ] **Step 7: Final commit**

```bash
git status --short
git add docs/plans/NEURIPS-HYBRID-RESNET-2026 scripts/studies/paper_results_refresh.py tests/studies/test_paper_results_refresh.py docs/backlog/active/2026-05-04-cns-history5-comparator-gap-fill.md docs/backlog/index.md
git commit -m "paper: refresh NeurIPS result presentation plan outputs"
```

## Completion Criteria

- BRDT has a generated metrics table and reconstruction comparison figure sourced only from completed BRDT runs.
- CDI Lines128 table includes U-NO rows and direct MAE/MSE/RMSE/SSIM metrics.
- CNS h5 main-table promotion is not done by mixing histories. Missing h5 FNO/U-Net work is represented by a concrete active backlog item.
- Manuscript-visible prose avoids internal process language.
- Evidence indexes point to generated table/figure assets and record the CNS h5 gap.
- Focused tests, compileall, and manuscript compile have fresh passing output.

## Explicit Non-Goals

- Do not install ODTbrain or rerun the BRDT classical row in this plan.
- Do not rerun completed CDI or U-NO rows.
- Do not change CNS headline table rows to h5 until `fno_base` and `unet_strong` h5 rows exist or are explicitly blocked.
- Do not edit `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`.
- Do not claim BRDT is a manuscript pillar from the completed preflight alone.
