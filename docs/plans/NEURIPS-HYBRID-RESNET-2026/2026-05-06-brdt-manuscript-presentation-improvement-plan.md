# BRDT Manuscript Presentation Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking.

**Goal:** Bring the BRDT manuscript section up to a defensible CVPR-style presentation standard without overstating a provenance-limited BRDT bundle as a primary paper result.

**Architecture:** Keep the existing BRDT metrics and source arrays as historical context only until the corrected no-refiner BRDT FFNO rerun lands. The current 40-epoch `ffno` row was produced by the legacy FFNO-local-refiner proxy and must not be presented as a pure FFNO-paper-stack result. A manuscript-facing BRDT figure/table may either label that row as a proxy or wait for `2026-05-06-brdt-corrected-ffno-40ep-rerun`.

**Tech Stack:** Python 3.11, NumPy, Matplotlib, pytest, LaTeX, zip, existing `scripts/studies/paper_results_refresh.py`, existing NeurIPS draft under `docs/plans/NEURIPS-HYBRID-RESNET-2026/`.

---

## Evidence Boundary

Use these files as authority:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/metrics.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/figures/source_arrays/sample_0255_*`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/paper_evidence_gate.json`

The BRDT bundle remains provenance-limited for primary-paper claims because the evidence gate failed on `git_provenance` and `host_provenance`. The manuscript must not expose internal labels such as `decision_support_convergence_followup`, but the scientific wording must reflect that BRDT is not a primary benchmark claim.

Allowed manuscript wording:

- "controlled secondary inverse-scattering study"
- "among the reported 40-epoch BRDT rows"
- "same split, input representation, and supervised-plus-Born objective"
- "not used as a primary benchmark"

Avoid:

- "paper-grade BRDT evidence"
- "full BRDT benchmark"
- "strongest BRDT model" without the row/budget qualifier
- "SOTA" or any broad inverse-scattering competitiveness claim
- internal workflow terms such as "gate", "claim boundary", "decision support", "promotion failed", or "artifact root" in the manuscript body

## File Structure

- Modify: `scripts/studies/paper_results_refresh.py`
  - Add BRDT 40-epoch constants and a deterministic renderer for a manuscript figure that combines measurement context, target-domain reconstructions, and absolute-error maps.
  - Add a CLI flag such as `--write-brdt-context-figure`.
  - Keep the existing `--write-brdt-assets` behavior for the old four-row preflight asset unless intentionally superseding it.

- Modify: `tests/studies/test_paper_results_refresh.py`
  - Add focused tests for the BRDT panel assembly, row roster, shared-scale policy, and CLI output path metadata.

- Create generated artifact: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png`
  - Generated from source arrays, not manually edited.
  - Layout: measurement/input context on the left; square reconstruction and error grid on the right.

- Optional generated artifact: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_sinogram_residuals.png`
  - Only create if the main figure becomes too crowded.

- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - Reframe the BRDT result text.
  - Replace the one-row reconstruction figure with a `figure*` using the new context/reconstruction/error asset.
  - Tighten the BRDT table caption and surrounding interpretation.

- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md`
  - Add a short rule for inverse-problem figures whose measurement domain and reconstruction domain have different shapes.

- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`
  - Rebuild after LaTeX verification so the packaged `.tex`, `.pdf`, and figures match the source tree.

## Current-Scope Decisions

- The main BRDT qualitative figure should not present the current 40-epoch `ffno` column as pure FFNO. Until the corrected rerun lands, either remove BRDT FFNO from manuscript-facing claims or label it explicitly as a legacy FFNO-local-refiner proxy.
- Do not add FNO or U-Net to the main 40-epoch BRDT table or figure, because the available FNO/U-Net BRDT visuals are from the older 20-epoch preflight. Mixing them into the 40-epoch figure would make the figure look like a matched complete comparison when it is not.
- If FNO/U-Net are useful context, mention them only in a short sentence that points to the earlier bounded preflight or reserve them for a separate supplementary/candidate figure.
- The measurement panel should show the observed sinogram summarized as magnitude. The available 40-epoch source arrays include the model-based Born inverse comparator, but do not include a separately verified neural-input/Born-init tensor; therefore the main figure must not label any panel as the neural input unless that tensor is recovered from source artifacts.
- Reconstructions must share one color scale derived from the target physical `q` range. Error maps must share one error color scale.

## Task 1: Add BRDT Panel Assembly Tests

**Files:**
- Modify: `tests/studies/test_paper_results_refresh.py`

- [x] **Step 1: Add a fixture-style test for panel loading**

Add a test that imports the planned BRDT panel loader:

```python
from scripts.studies.paper_results_refresh import load_brdt_sample255_panels


def test_load_brdt_sample255_panels_uses_40ep_rows_and_measurement_context():
    panels = load_brdt_sample255_panels()

    assert panels.sample_id == 255
    assert panels.measurement_magnitude.shape == (64, 128)
    assert panels.target_q.shape == (128, 128)
    assert [row.row_id for row in panels.reconstruction_rows] == [
        "classical_born_backprop",
        "ffno",
        "hybrid_resnet",
    ]
    assert [row.label for row in panels.reconstruction_rows] == [
        "Model-based Born inverse",
        "FFNO",
        "SRU-Net",
    ]
    assert panels.reconstruction_vmin < panels.reconstruction_vmax
    assert panels.error_vmax > 0
```

- [x] **Step 2: Run the test and verify it fails**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_load_brdt_sample255_panels_uses_40ep_rows_and_measurement_context
```

Expected: FAIL because `load_brdt_sample255_panels` does not exist yet.

- [x] **Step 3: Add a renderer output smoke test**

Add:

```python
from scripts.studies.paper_results_refresh import write_brdt_context_figure


def test_write_brdt_context_figure_writes_expected_file(tmp_path):
    output = tmp_path / "brdt_sample_0255_context_recon_error.png"

    written = write_brdt_context_figure(output_path=output)

    assert written == output
    assert output.exists()
    assert output.stat().st_size > 20_000
```

- [x] **Step 4: Run the renderer test and verify it fails**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_write_brdt_context_figure_writes_expected_file
```

Expected: FAIL because `write_brdt_context_figure` does not exist yet.

## Task 2: Implement The BRDT Manuscript Figure Generator

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Test: `tests/studies/test_paper_results_refresh.py`

- [x] **Step 1: Add BRDT 40-epoch constants**

Add near the existing BRDT constants:

```python
BRDT_40EP_ROOT = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-05-05-brdt-supervised-born-40ep-paper-evidence"
)
BRDT_40EP_SOURCE_ARRAYS = BRDT_40EP_ROOT / "figures" / "source_arrays"
BRDT_CONTEXT_FIGURE = FIGURES_DIR / "brdt_sample_0255_context_recon_error.png"
```

- [x] **Step 2: Add small data containers**

Add dataclasses or lightweight named tuples:

```python
@dataclass(frozen=True)
class BrdtReconPanel:
    row_id: str
    label: str
    q_pred: np.ndarray
    abs_error: np.ndarray


@dataclass(frozen=True)
class BrdtSamplePanels:
    sample_id: int
    measurement_magnitude: np.ndarray
    target_q: np.ndarray
    reconstruction_rows: tuple[BrdtReconPanel, ...]
    reconstruction_vmin: float
    reconstruction_vmax: float
    error_vmax: float
```

- [x] **Step 3: Implement `load_brdt_sample255_panels`**

Behavior:

- Load `sample_0255_sino_obs.npy`.
- Compute `measurement_magnitude = np.linalg.norm(sino_obs, axis=-1)`.
- Load `sample_0255_q_target.npy`.
- Load `sample_0255_classical_born_backprop_q_pred.npy` as the model-based Born inverse row. Do not treat it as the neural input tensor unless a separately named input array is available.
- Load `sample_0255_ffno_q_pred.npy`.
- Load `sample_0255_hybrid_resnet_q_pred.npy`.
- Compute each row's absolute error against target.
- Use labels:
  - `classical_born_backprop` -> `Model-based Born inverse`
  - `ffno` -> `FFNO`
  - `hybrid_resnet` -> `SRU-Net`
- Use a shared reconstruction color scale from the target `q` min/max, not per-panel autoscaling.
- Use a shared error scale from the maximum absolute error across the three rows.

- [x] **Step 4: Run panel loading test**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_load_brdt_sample255_panels_uses_40ep_rows_and_measurement_context
```

Expected: PASS.

- [x] **Step 5: Implement `write_brdt_context_figure`**

Recommended layout:

```text
balanced 2x4 grid:
  Ground truth                | Model-based Born inverse | FFNO       | SRU-Net
  Observed sinogram magnitude | Born inverse error       | FFNO error | SRU-Net error
```

Implementation notes:

- Use `matplotlib.use("Agg")`.
- Use `GridSpec` or `subplot_mosaic`; avoid nested images inside one axes.
- Keep `figsize` around `(11.5, 4.2)` for `figure*`.
- Use small titles, no suptitle if it makes the figure crowded.
- Use one colorbar for the reconstruction row and one colorbar for the error row.
- Save at `dpi=220`.
- Return the output `Path`.

- [x] **Step 6: Run renderer test**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py::test_write_brdt_context_figure_writes_expected_file
```

Expected: PASS.

## Task 3: Add CLI Hook And Generate The Manuscript Figure

**Files:**
- Modify: `scripts/studies/paper_results_refresh.py`
- Generate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png`

- [x] **Step 1: Add CLI flag**

Add to `main()`:

```python
parser.add_argument(
    "--write-brdt-context-figure",
    action="store_true",
    help="Write the BRDT sample-255 measurement/context/reconstruction/error figure.",
)
```

Route it:

```python
if args.write_brdt_context_figure:
    outputs["brdt_context_figure"] = str(write_brdt_context_figure())
```

- [x] **Step 2: Run the targeted BRDT tests**

Run:

```bash
pytest -q tests/studies/test_paper_results_refresh.py -k "brdt"
```

Expected: PASS.

- [x] **Step 3: Generate the figure**

Run:

```bash
python scripts/studies/paper_results_refresh.py --write-brdt-context-figure
```

Expected:

- command exits `0`
- output JSON/stdout mentions `brdt_sample_0255_context_recon_error.png`
- generated file exists and is non-empty

- [x] **Step 4: Inspect figure dimensions**

Run:

```bash
python - <<'PY'
from PIL import Image
from pathlib import Path
p = Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png")
im = Image.open(p)
print(im.size)
assert im.size[0] >= 1800
assert im.size[1] >= 700
PY
```

Expected: prints a wide figure size suitable for `figure*`.

- [x] **Step 5: Record changed generator and figure files**

Do not commit during this task unless the operator explicitly asks. Report the changed files after verification.

## Task 4: Revise BRDT Manuscript Text And Figure Block

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [x] **Step 1: Reframe the experiment paragraph**

At the BRDT experiment protocol paragraph around the current "Secondary inverse-scattering benchmark: BRDT" subsection, replace broad language with:

```latex
The BRDT experiment is a controlled secondary inverse-scattering study. It uses
$128\times128$ scattering-potential fields and simulated complex sinograms from
a differentiable Born forward model with 64 illumination angles and 128 detector
samples. We use it to test whether the CDI reconstruction architecture transfers
to a related wave inverse problem; it is not used as a primary benchmark claim.
```

Then keep the concrete split/training details.

- [x] **Step 2: Tighten the results interpretation**

Replace the current claim paragraph with wording like:

```latex
Among the reported 40-epoch neural rows, SRU-Net gives the lower image-space
error on this BRDT split, with relative $L_2(q)$ of 0.2875 versus 0.3324 for
FFNO. The model-based Born inverse provides a non-neural reference. These
results support the qualitative transfer behavior of the architecture, while
the paper's main quantitative claims remain the CDI and CNS benchmarks.
```

- [x] **Step 3: Update table caption**

Keep the table if desired, but make the caption bounded:

```latex
\caption{\textbf{Controlled BRDT inverse-scattering study.} SRU-Net and FFNO
are trained for 40 epochs with the same supervised object loss plus
Born-consistency loss. The model-based Born inverse is a non-neural reference.
This secondary study is reported as transfer evidence rather than as a primary
benchmark. Runtime is measured over the 256-sample test split on the same GPU.}
```

- [x] **Step 4: Replace the one-column figure with a `figure*`**

Replace the active BRDT figure block with:

```latex
% figure_metadata: task=BRDT; source=.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/figures/source_arrays/sample_0255_*; paper_copy=figures/brdt_sample_0255_context_recon_error.png; sample_id=255; visible_rows=measurement_magnitude,ground_truth,classical_born_backprop,ffno,sru_net
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/brdt_sample_0255_context_recon_error.png}
\caption{\textbf{BRDT held-out qualitative comparison.} The figure separates
the BRDT input $|s_{\mathrm{obs}}(\theta,d)|$ from the target scattering
potential $q$. The top row shows the target and three estimates
$\hat{q}$; the bottom row shows the input sinogram magnitude followed by
absolute target-domain errors under a shared error scale.}
\label{fig:brdt_candidate_recon}
\end{figure*}
```

- [x] **Step 5: Scan for overclaiming and internal language**

Run:

```bash
rg -n "decision_support|claim boundary|gate|promotion|paper-grade|strongest BRDT|full BRDT|SOTA|additive paper" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: no matches, except intentional metadata comments if they do not use internal terms. Prefer keeping metadata comments technical and not visible to readers.

- [x] **Step 6: Record manuscript text changes**

Do not commit during this task unless the operator explicitly asks.

## Task 5: Update The Presentation Style Guide

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md`

- [x] **Step 1: Add an inverse-problem figure rule**

Add a short section:

```markdown
## Inverse-Problem Qualitative Figures

When the measurement/input domain has a different geometry from the target
domain, do not force it into the same method-comparison grid. Show the
measurement context separately, then compare only target-domain predictions and
errors in the rectangular method grid.

For BRDT-style data, the main figure should show:

- measurement context: sinogram magnitude, plus a separately verified model input only when that input tensor is present in the source arrays;
- reconstruction row: ground truth plus same-budget methods;
- error row: absolute errors under one shared scale;
- caption language that distinguishes controlled secondary studies from primary
  benchmark claims.
```

- [x] **Step 2: Verify no internal jargon appears in manuscript-facing guidance**

Run:

```bash
rg -n "gate|promotion failed|decision_support|claim_boundary" docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md
```

Expected: no matches.

- [x] **Step 3: Record the guide update**

Do not commit during this task unless the operator explicitly asks.

## Task 6: Compile And Rebuild The Paper Package

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`

- [x] **Step 1: Compile the draft**

Run:

```bash
cd docs/plans/NEURIPS-HYBRID-RESNET-2026
pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex
```

Expected: both commands exit `0`.

- [x] **Step 2: Inspect the compiled text for bounded BRDT language**

Run:

```bash
pdftotext hybrid_resnet_neurips_first_draft.pdf - | rg -n "controlled secondary|primary benchmark|BRDT|0\\.2875|0\\.3324"
```

Expected: finds bounded BRDT language and the reported metrics.

- [x] **Step 3: Rebuild the zip**

Run from `docs/plans/NEURIPS-HYBRID-RESNET-2026`:

```bash
tmp_dir="$(mktemp -d)"
tmp_zip="$tmp_dir/scr_ptychography_neurips_draft_package.zip"
zip -r "$tmp_zip" \
  hybrid_resnet_neurips_first_draft.tex \
  hybrid_resnet_neurips_first_draft.pdf \
  paper_presentation_style_guide.md \
  paper_evidence_package_audit_summary.md \
  evidence_matrix.md \
  figures/cdi_synthetic_line_pattern_srunet_fno_ffno.png \
  figures/pdebench_cns_sample000_predictions.png \
  figures/cdi_lines128_complete_compare_amp_phase.png \
  figures/brdt_decision_support_recon.png \
  figures/brdt_sample_0255_compare_q.png \
  figures/brdt_sample_0255_context_recon_error.png \
  figures/brdt_fno_objective_ablation.png \
  figures/cdi_lines128_complete_frc_curves.png \
  figures/cdi_synthetic_line_pattern_amp_phase.png \
  figures/cdi_lines128_phase_zoom_srunet_fno_ffno_uno.png \
  figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet.png \
  figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet_per_panel_scaled.png \
  tables/brdt_decision_support_metrics.json \
  tables/brdt_decision_support_metrics.csv \
  tables/brdt_decision_support_metrics.tex \
  tables/cdi_lines128_metrics_extended.tex \
  tables/cdi_lines128_metrics_extended.csv \
  tables/cdi_lines128_metrics_extended.json \
  tables/cdi_lines128_objective_comparison.tex \
  tables/cdi_lines128_pinn_metrics.tex \
  tables/model_config_by_benchmark.tex \
  tables/pdebench_cns_matched_condition_metrics.tex \
  tables/pdebench_cns_matched_condition_metrics.csv \
  tables/pdebench_cns_matched_condition_metrics.json
mv "$tmp_zip" scr_ptychography_neurips_draft_package.zip
rmdir "$tmp_dir"
```

This roster intentionally preserves the current package contents and includes every local figure/table file referenced by the manuscript. Do not add nonexistent files such as `refs.bib`, and do not drop unrelated packaged assets.

- [x] **Step 4: Verify zip freshness**

Run from repo root:

```bash
zip -T docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip
unzip -l docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip | rg "brdt_sample_0255_context_recon_error.png|brdt_fno_objective_ablation.png|model_config_by_benchmark.tex|hybrid_resnet_neurips_first_draft.pdf"
unzip -p docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip hybrid_resnet_neurips_first_draft.tex | cmp -s docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex - && echo tex_matches_zip
```

Expected:

- `zip -T` reports OK.
- New BRDT figures, model-configuration table, and PDF are present.
- `tex_matches_zip` is printed.

- [x] **Step 5: Remove loose LaTeX byproducts if untracked**

Run:

```bash
git status --short docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.{aux,out,log,pdf}
```

If `.aux`, `.out`, `.log`, or loose `.pdf` are untracked and not intentionally committed, remove them:

```bash
rm -f docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.aux \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.out \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.log \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.pdf
```

- [x] **Step 6: Record package rebuild**

Do not commit during this task unless the operator explicitly asks.

## Final Verification

- [x] Run targeted tests:

```bash
pytest -q tests/studies/test_paper_results_refresh.py -k "brdt"
```

- [x] Run compile check:

```bash
python -m compileall -q scripts/studies/paper_results_refresh.py
```

- [x] Run manuscript text scan:

```bash
rg -n "decision_support|claim boundary|gate|promotion|paper-grade|strongest BRDT|full BRDT|SOTA" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: no reader-visible overclaiming or internal workflow language.

- [x] Run zip freshness check:

```bash
zip -T docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip
unzip -p docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip hybrid_resnet_neurips_first_draft.tex | cmp -s docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex - && echo tex_matches_zip
```

Expected: OK and `tex_matches_zip`.

## Follow-Up Work

- A future BRDT retraining pass with full runtime provenance capture is required before BRDT can become additive paper evidence rather than a controlled secondary study.
- A same-budget BRDT FNO/U-Net rerun would be required before the main BRDT table can fairly include FNO and U-Net alongside the 40-epoch SRU-Net and FFNO rows.
- If space permits in a supplement, create a separate preflight context figure for the older 20-epoch FNO/U-Net rows, clearly labeled as preflight context rather than the 40-epoch comparison.
