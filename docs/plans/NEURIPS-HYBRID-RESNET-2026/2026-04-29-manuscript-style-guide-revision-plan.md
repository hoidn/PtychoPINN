# Manuscript Style Guide Revision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring `hybrid_resnet_neurips_first_draft.tex` up to the presentation standard defined in `paper_presentation_style_guide.md` without changing the underlying evidence.

**Architecture:** Treat this as a manuscript-structure and presentation pass, not a new experiment pass. The revision should convert the current long evidence dump into a CVPR-style paper shape: one primary CDI claim, one architecture/training-loop figure, one CNS secondary benchmark, and compact ablations with provenance preserved in comments.

**Tech Stack:** LaTeX article draft, TikZ/graphicx/booktabs, existing PNG/PDF figure assets, `pdflatex`, repo-local `.artifacts` evidence tables, and the existing self-contained zip package.

---

## Governing Inputs

- Style guide: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md`
- Manuscript: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Package zip: `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`
- Existing figures:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_synthetic_line_pattern_srunet_fno_ffno.png`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/pdebench_cns_sample000_predictions.png`
- CNS bundle source:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/`

## File Structure

- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - Main paper prose, section order, table formatting, captions, hidden metadata comments, and figure references.
- Modify or regenerate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_synthetic_line_pattern_srunet_fno_ffno.png`
  - Only if a better main CDI qualitative panel can be assembled from existing artifacts.
- Modify or regenerate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/pdebench_cns_sample000_predictions.png`
  - Only to match the final figure-grid style and `SRU-Net*` label policy.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`
  - Rebuild after TeX and figure changes.
- Do not modify experiment code or rerun training in this plan.
- Do not create a worktree; this repo's `AGENTS.md` explicitly says not to create worktrees.

## Current Gaps Against The Style Guide

- The current draft still reads like an evidence ledger in places: many CNS tables are inline instead of moved to appendix/supplement scope.
- Figure ordering is not CVPR-like: the architecture figure appears before the primary qualitative CDI result.
- Figure 1 is too implementation-internal and does not compare `CNN/U-Net`, `FNO/FFNO`, and `SRU-Net` as requested by the style guide.
- The CDI quantitative table mixes method and training procedure in one column instead of separating `Model` and `Training`.
- Captions are serviceable but not consistently claim-first.
- CNS evidence is currently spread across headline, temporal, spectral-depth, mode-sweep, convergence, and FFNO-local-feature tables; the style guide wants one CNS headline table plus one compact ablation table in the main paper.
- The `SRU-Net*` CNS naming decision is documented in comments, but the final manuscript should keep the visible explanation concise and move restoration detail into comments.
- Bibliography entries for FNO, local neural operators, and U-FNO are placeholders.

## Non-Goals

- Do not invent missing numerical results.
- Do not claim global state of the art for FFNO or SRU-Net.
- Do not erase provenance comments.
- Do not remove restore comments for omitted CNS `hybrid_resnet_cns` rows or columns.
- Do not relabel mixed-provenance CDI context rows as a matched benchmark.
- Do not turn CNS into a primary ptychography result.

---

### Task 1: Make The Main-Paper Skeleton Match The Style Guide

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Create a section-order checklist from the style guide**

Record the target order in a temporary note or editor checklist:

```text
Abstract
Introduction
Related Work
Methods
Experiments
Results
Discussion
Additional Experiments / Appendix Pointer
References
```

Expected: current sections can stay in this order, but figure and table placement must change.

- [ ] **Step 2: Move the primary qualitative CDI figure before dense architecture internals**

Move or duplicate the CDI qualitative figure so a reader sees reconstruction evidence before long methods details. Keep one canonical label:

```latex
\label{fig:cdi_main_qualitative}
```

Expected: the first main visual is the CDI qualitative result, not a dense TikZ internals diagram.

- [ ] **Step 3: Convert the current architecture figure into Figure 2**

Rename the label and caption to match the style guide:

```latex
\label{fig:srunet_architecture}
```

Use a claim-first caption:

```latex
\caption{\textbf{SRU-Net combines spectral global mixing with local image recovery.}
The generator uses spectral-local encoder blocks, a residual bottleneck, and a
convolutional decoder inside the physics-informed CDI training loop.}
```

Expected: architecture figure is still valid TeX, but the caption is paper-facing.

- [ ] **Step 4: Run a compile check**

Run:

```bash
tmp=$(mktemp -d)
cp docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex "$tmp/"
cp -R docs/plans/NEURIPS-HYBRID-RESNET-2026/figures "$tmp/"
(cd "$tmp" && pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex)
```

Expected: exit 0. Undefined references are acceptable on the first pass.

---

### Task 2: Rewrite Tables Into Paper-Facing Forms

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Rewrite the headline CDI table**

Replace the current one-column method table with the style-guide shape:

```latex
% table_metadata: task=CDI_lines128; model_rows=SRU-Net+PINN,FFNO+PINN;
% train_objects=2; test_objects=2; train_patches=8978; test_patches=1458;
% photons=1e9; epochs=40; source=.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet
\begin{table}[t]
\centering
\small
\caption{\textbf{Synthetic line-pattern CDI reconstruction.} Both models are
trained through the ptychographic forward model. Lower MAE and higher SSIM are
better.}
\label{tab:cdi_headline}
\begin{tabular}{llrrrr}
\toprule
Model & Training & Amp MAE $\downarrow$ & Phase MAE $\downarrow$ &
Amp SSIM $\uparrow$ & Phase SSIM $\uparrow$ \\
\midrule
FFNO & PINN & 0.0628 & 0.0828 & 0.9348 & 0.9816 \\
SRU-Net & PINN & \textbf{0.0269} & \textbf{0.0721} &
\textbf{0.9881} & \textbf{0.9947} \\
\bottomrule
\end{tabular}
\end{table}
```

Expected: PSNR leaves the headline table unless needed later.

- [ ] **Step 2: Convert the CDI SSIM table into a context table**

Use `Model`, `Training`, and `Setting` columns. Do not bold mixed-provenance rows.

Expected caption:

```latex
\caption{\textbf{CDI SSIM context across available line-pattern runs.}
Rows summarize available evidence and should not be read as a single matched
benchmark unless the setting column matches.}
```

- [ ] **Step 3: Rewrite the CNS headline table**

Use the style-guide CNS shape, keeping `SRU-Net$^*$` and the restore comment:

```latex
\begin{tabular}{lrrrr}
\toprule
Model & nRMSE $\downarrow$ & Rel. $L_2$ $\downarrow$ &
High-band RMSE $\downarrow$ & Params \\
\midrule
U-Net & 0.676 & 0.676 & 1.333 & 7.76M \\
FNO & 0.074 & 0.074 & 0.672 & 0.36M \\
SRU-Net$^*$ & 0.062 & 0.062 & 0.435 & 8.19M \\
FFNO & \textbf{0.028} & \textbf{0.028} & \textbf{0.121} & 1.07M \\
% restore_if_needed: SRU-Net continuity row maps to repo row hybrid_resnet_cns...
\bottomrule
\end{tabular}
```

Expected: runtime moves to prose or supplement unless it is central to a claim.

- [ ] **Step 4: Collapse CNS component evidence into one compact ablation table**

Replace scattered main-paper skip, upsampler, temporal, depth, and mode tables with one table in the main text. Keep the removed detailed tables as commented blocks or move them below an appendix marker.

Suggested main-paper ablation table:

```latex
% table_metadata: task=PDEBench_CNS; ablation=SRU-Net_components; source=multiple existing CNS capped studies; detailed source rows preserved in comments below
\begin{table}[t]
\centering
\small
\caption{\textbf{SRU-Net component ablations on PDEBench CNS.} Rows vary one
architectural or input-context choice at a time under the available bounded CNS
setting. Lower values are better.}
\label{tab:cns_ablation}
\begin{tabular}{lrr}
\toprule
Variant & nRMSE $\downarrow$ & High-band RMSE $\downarrow$ \\
\midrule
Legacy hybrid shell baseline & 0.224 & 1.243 \\
Legacy hybrid shell + skip-add & 0.105 & 0.697 \\
SRU-Net$^*$, two-frame & 0.062 & 0.435 \\
SRU-Net$^*$, three-frame & 0.046 & 0.347 \\
SRU-Net$^*$, 24 modes, longer run & 0.040 & 0.307 \\
\bottomrule
\end{tabular}
\end{table}
```

Expected: each row's exact source remains in hidden comments.

- [ ] **Step 5: Preserve detailed CNS tables as restorable comments or appendix-ready blocks**

For every removed table, add a comment like:

```latex
% supplement_candidate: original CNS Fourier-mode table preserved in git history;
% source=.artifacts/...; restore if supplement is added.
```

Expected: main paper becomes tighter without losing the restoration path.

- [ ] **Step 6: Run table-focused checks**

Run:

```bash
rg -n "blocked|contract|row lock|runner|history_len|seed|cap_label|capped_decision|paper_grade" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
rg -n "\\\\caption\\{|table_metadata|restore_if_needed|supplement_candidate" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: internal terms appear only in comments when scientifically necessary; every table has nearby metadata.

---

### Task 3: Bring Figures Up To The Visual Standard

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Possibly regenerate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_synthetic_line_pattern_srunet_fno_ffno.png`
- Possibly regenerate: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/pdebench_cns_sample000_predictions.png`

- [ ] **Step 1: Audit available CDI visual artifacts**

Run:

```bash
find .artifacts docs/plans/NEURIPS-HYBRID-RESNET-2026/figures -path '*lines128*' -o -path '*grid*line*' | sed -n '1,160p'
```

Expected: identify whether there are existing error maps or zoom panels for CDI.

- [ ] **Step 2: Decide whether to rebuild the CDI visual**

If existing CDI outputs already include ground truth, FNO, FFNO, SRU-Net, and error maps on shared samples, rebuild the visual with this order:

```text
Ground truth | CNN + PINN if available | FNO + PINN | FFNO-local proxy + PINN until no-refiner rerun lands | SRU-Net + PINN | error/zoom
```

If those assets are not available, keep the current CDI visual and add a hidden comment:

```latex
% figure_gap: CDI main qualitative lacks CNN + PINN and error-map columns; add when full Lines128 bundle exists.
```

- [ ] **Step 3: Simplify the CNS visual**

Main-paper CNS visual should prefer density and pressure if page budget is tight. Keep the current four-field figure only if it remains readable at `\linewidth`.

Expected options:

```latex
% figure_metadata: task=PDEBench_CNS; fields=density,pressure; visible_rows=target,FFNO,SRU-Net*,FNO,U-Net; omitted_row=hybrid_resnet_cns; source=...
```

or keep the current four-field figure with a caption that explicitly names the visual claim.

- [ ] **Step 4: Add figure metadata comments**

Before each figure, add:

```latex
% figure_metadata: task=...; sample_ids=...; visible_rows=...; omitted_rows=...; source=...
```

Expected: figures can be regenerated from comments without exposing artifact paths in prose.

- [ ] **Step 5: Compile after figure edits**

Run two passes:

```bash
tmp=$(mktemp -d)
cp docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex "$tmp/"
cp -R docs/plans/NEURIPS-HYBRID-RESNET-2026/figures "$tmp/"
(cd "$tmp" && pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex && pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex)
```

Expected: exit 0, no missing figure files.

---

### Task 4: Tighten Prose To CVPR Paper Tone

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Fix title and abstract polish**

Fix the title typo:

```latex
\title{Deep Spectral-Convolutional Nets for Physics-Informed Ptychographic Reconstruction}
```

Expected: no missing space in `Nets for`.

- [ ] **Step 2: Make the abstract match the four-sentence style-guide structure**

Use this shape:

```latex
Coherent diffractive imaging and ptychography recover structure from intensity-only diffraction measurements, coupling global wave physics with sharp local object features.
We introduce SRU-Net, a spectral-residual U-Net that combines Fourier-style global mixing with convolutional local recovery inside a physics-informed reconstruction loop.
On synthetic line-pattern CDI, SRU-Net + PINN improves amplitude and phase reconstruction over the historical FFNO-local proxy + PINN row on matched metrics.
On PDEBench CNS, a secondary supervised benchmark, FFNO is the lowest-error row while SRU-Net$^*$ remains ahead of FNO and U-Net baselines, supporting the broader global-local architecture thesis.
```

Expected: concise, no artifact/process language.

- [ ] **Step 3: Shorten methods where the figure carries detail**

Keep the mathematical definitions but remove repeated prose that duplicates the architecture table.

Expected: Methods still include:

```latex
\mathcal{L}_{\rm CDI}=\mathcal{D}(x,F_d(F_c(G_\theta(x))))+\lambda R(G_\theta(x))
```

and the CNS supervised objective.

- [ ] **Step 4: Replace weak captions**

Every main caption should use:

```latex
\caption{\textbf{Short claim sentence.} Task/protocol sentence. Visual or metric takeaway sentence.}
```

Expected: no caption only lists columns.

- [ ] **Step 5: Keep limitations scientific**

Discussion should mention:

- CDI evidence is currently strongest for SRU-Net + PINN versus the historical FFNO-local proxy + PINN row; pure FFNO comparison awaits the no-refiner rerun.
- CDI supervised equivalents are still needed to separate architecture from training objective.
- CNS is a secondary cross-domain architecture test.
- `SRU-Net*` marks a temporary CNS naming reconciliation issue.

Expected: no bureaucratic language such as `blocked`, `contract`, `row lock`, or `decision-support` in visible prose.

---

### Task 5: Replace Placeholder References

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Replace FNO placeholder**

Use the canonical FNO reference:

```latex
\bibitem{fno}
Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu,
Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar.
\newblock Fourier neural operator for parametric partial differential equations.
\newblock \emph{International Conference on Learning Representations}, 2021.
```

- [ ] **Step 2: Replace U-FNO or hybrid Fourier/U-Net placeholder**

Add a real U-FNO/hybrid neural-operator reference from `docs/litsurvey.md` or the style guide survey notes. Verify exact title before committing.

Run:

```bash
rg -n "U-FNO|FourierUnet|HUFNO|U-NO|localized" docs/litsurvey.md docs/plans/fno.md
```

Expected: no placeholder remains for `ufno`.

- [ ] **Step 3: Replace localized neural operator placeholder or remove citation**

If the manuscript no longer cites `localno`, remove the placeholder entry. If it does cite localized-kernel neural operators, add the exact reference.

Run:

```bash
rg -n "localno|localized-kernel|localized kernel" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex docs/litsurvey.md
```

Expected: no placeholder bibliography entries remain.

- [ ] **Step 4: Compile twice**

Run:

```bash
tmp=$(mktemp -d)
cp docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex "$tmp/"
cp -R docs/plans/NEURIPS-HYBRID-RESNET-2026/figures "$tmp/"
(cd "$tmp" && pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex && pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex)
```

Expected: exit 0. No undefined citations from removed placeholders.

---

### Task 6: Final Verification And Package Rebuild

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`

- [ ] **Step 1: Run manuscript hygiene scan**

Run:

```bash
rg -n "blocked|contract|cap_label|capped_decision|decision-support|paper_grade|row lock|runner|normalized runner|Evidence Targets|comparator" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: no visible-prose matches. Metadata-comment matches are acceptable only if needed for provenance.

- [ ] **Step 2: Run final compile from a clean temp package**

Run:

```bash
tmp=$(mktemp -d)
pkg="$tmp/sru_ptychography_neurips_draft_package"
mkdir -p "$pkg/figures"
cp docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex "$pkg/main.tex"
cp docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/*.png "$pkg/figures/"
cd "$pkg"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: exit 0 and `main.pdf` exists.

- [ ] **Step 3: Rebuild the self-contained zip**

Run from the temp parent directory:

```bash
cat > "$pkg/README.txt" <<'EOF'
Self-contained LaTeX package for the SRU-Net ptychography manuscript draft.

Build from this directory with:
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex

Contents:
  main.tex
  main.pdf
  figures/cdi_synthetic_line_pattern_srunet_fno_ffno.png
  figures/pdebench_cns_sample000_predictions.png
EOF

cd "$tmp"
zip -q /home/ollie/Documents/PtychoPINN/docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip \
  sru_ptychography_neurips_draft_package/README.txt \
  sru_ptychography_neurips_draft_package/main.tex \
  sru_ptychography_neurips_draft_package/main.pdf \
  sru_ptychography_neurips_draft_package/figures/cdi_synthetic_line_pattern_srunet_fno_ffno.png \
  sru_ptychography_neurips_draft_package/figures/pdebench_cns_sample000_predictions.png
```

Expected: zip contains only README, `main.tex`, `main.pdf`, and intended figures.

- [ ] **Step 4: Verify archive integrity**

Run:

```bash
unzip -t docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip
unzip -l docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip
git diff --check -- docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: archive test reports no errors; diff check exits 0.

- [ ] **Step 5: Review final diff**

Run:

```bash
git diff --stat -- docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex docs/plans/NEURIPS-HYBRID-RESNET-2026/figures docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip
```

Expected: changes are limited to manuscript, figure assets, and package zip.

---

## Acceptance Criteria

- The manuscript follows the style guide's main-paper budget:
  - one headline CDI table;
  - one CDI qualitative figure;
  - one architecture/training-loop figure;
  - one CNS headline table;
  - one compact CNS ablation table.
- Model and training procedure are separable in CDI tables.
- Mixed-provenance CDI rows are clearly marked as context.
- CNS is framed as a secondary cross-domain benchmark.
- `SRU-Net*` is used consistently for the CNS spectral-bottleneck row, with metadata comments mapping it to `spectral_resnet_bottleneck_base`.
- Removed CNS `hybrid_resnet_cns` rows/columns are preserved as restore comments.
- Visible prose does not use internal process language.
- Placeholder references are replaced or removed.
- `pdflatex` succeeds twice from a clean temp package.
- The self-contained zip is regenerated and passes `unzip -t`.
