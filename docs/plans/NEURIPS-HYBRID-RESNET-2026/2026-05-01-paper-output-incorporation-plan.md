# NeurIPS Paper Output Incorporation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Incorporate the CDI and CNS outputs that strengthen the NeurIPS Hybrid ResNet manuscript without importing results into the unrelated overlap-free PtychoPINN manuscript.

**Architecture:** Treat the complete Lines128 CDI bundle as the primary result, keep supervised FFNO and skip/residual evidence as secondary controls, and present PDEBench CNS with its reduced-data protocol stated plainly. The implementation should update the manuscript and paper-facing figure/table assets only; it must not rerun training or present readiness checks as experimental results.

**Tech Stack:** LaTeX article draft, repo-local `docs/plans/NEURIPS-HYBRID-RESNET-2026/` manuscript package, PNG figure assets, generated TeX/CSV tables under `.artifacts/`, `pdflatex`, `rg`, and `git diff`.

---

## Governing Inputs

- Manuscript draft:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Paper style guide:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_presentation_style_guide.md`
- Evidence audit:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_audit_summary.md`
- Evidence matrix:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- Complete CDI summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`
- Supervised FFNO extension:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_supervised_equivalent_rows_summary.md`
- CNS table/figure bundle:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_paper_table_figure_bundle_summary.md`
- Hybrid skip/residual ablation:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md`
- U-NO readiness summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_preflight_summary.md`
- WaveBench readiness summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight_summary.md`

## File Structure

- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - Promote the complete six-row CDI table to the main CDI result.
  - Move the older two-row FFNO-vs-SRU result into prose or remove it as a separate table.
  - Add a supervised FFNO control subsection or appendix paragraph.
  - Keep CNS as a secondary reduced-data benchmark.
  - Add a compact skip/residual ablation table in Additional Experiments or appendix scope.
  - Remove planning-style wording from manuscript prose.
- Modify or create figure assets under:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/`
  - Copy or regenerate paper-local versions of the complete CDI visual bundle:
    `compare_amp_phase.png` and `frc_curves.png`.
  - Keep or refresh the CNS fixed-sample visual bundle as a paper-local figure.
- Modify if needed:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - Add a short manuscript-incorporation note if the index does not already point readers from evidence outputs to the manuscript draft.
- Modify if needed:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`
  - Rebuild only after the TeX and paper-local figures compile cleanly.
- Do not modify:
  `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
  - The recent NeurIPS outputs do not belong in the overlap-free PtychoPINN manuscript.
- Do not include as result rows:
  - U-NO readiness outputs until benchmark metrics exist.
  - WaveBench readiness outputs until dataset/checkpoint and benchmark rows exist.
  - CNS full-training or SOTA claims until full-training evidence exists.

## Claim Boundaries

- CDI Lines128 complete table: primary result.
- Supervised FFNO extension: adjacent control, not the primary CDI table.
- PDEBench CNS table/figures: secondary reduced-data benchmark.
- Hybrid skip/residual ablation: architecture ablation for appendix or secondary discussion.
- U-NO and WaveBench: readiness only, no metric-table claims.

---

### Task 1: Audit Current Manuscript Against The Evidence Authorities

**Files:**
- Read: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Read: evidence summaries listed above

- [ ] **Step 1: Confirm the active manuscript target**

Run:

```bash
rg -n "\\\\title|\\\\section|CDI|CNS|FFNO|U-NO|WaveBench|table_metadata|figure_metadata" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: the target manuscript is the NeurIPS Hybrid ResNet draft, not `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`.

- [ ] **Step 2: Record current integration status**

Create a short scratch checklist:

```text
CDI complete six-row table: present / absent / outdated
CDI complete visual bundle: present / absent / outdated
Supervised FFNO extension: present / absent / mixed into primary table
CNS bounded table: present / absent / missing bounded wording
CNS fixed-sample figure: present / absent / outdated
Skip/residual ablation: present / absent
U-NO/WaveBench: result claims present? yes/no
```

Expected: the draft already contains some outputs but still needs claim-scope cleanup.

- [ ] **Step 3: Verify artifact roots exist before editing**

Run:

```bash
test -f .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/metrics_table.tex
test -f .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/visuals/compare_amp_phase.png
test -f .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/visuals/frc_curves.png
test -f .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/cns_paper_table_rows.tex
test -f docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md
```

Expected: exit 0. If any artifact is missing, stop and repair discoverability before editing the manuscript.

### Task 2: Promote The Complete CDI Bundle To The Primary CDI Result

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Create or overwrite:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_complete_compare_amp_phase.png`
- Create or overwrite:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_complete_frc_curves.png`

- [ ] **Step 1: Copy paper-local CDI figures from the authoritative root**

Run:

```bash
cp .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/visuals/compare_amp_phase.png \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_complete_compare_amp_phase.png
cp .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/visuals/frc_curves.png \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_complete_frc_curves.png
```

Expected: the manuscript can compile without reaching into `.artifacts/`.

- [ ] **Step 2: Replace the outdated CDI qualitative figure**

Replace the current `fig:cdi_main_qualitative` image include with:

```latex
% figure_metadata: task=CDI_lines128; source=.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/visuals/compare_amp_phase.png; paper_copy=figures/cdi_lines128_complete_compare_amp_phase.png; rows=baseline,pinn,pinn_hybrid_resnet,pinn_fno_vanilla,pinn_spectral_resnet_bottleneck_net,pinn_ffno
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/cdi_lines128_complete_compare_amp_phase.png}
\caption{\textbf{Matched Lines128 CDI reconstruction comparison.} The comparison covers CNN, FNO, FFNO, SRU-Net, and spectral SRU-Net rows under the same $N=128$ line-pattern protocol.}
\label{fig:cdi_main_qualitative}
\end{figure}
```

Expected: the first CDI figure reflects the complete bundle, not the older three-row visual.

- [ ] **Step 3: Add the FRC figure near the CDI table or appendix**

Add:

```latex
% figure_metadata: task=CDI_lines128; source=.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/visuals/frc_curves.png; paper_copy=figures/cdi_lines128_complete_frc_curves.png
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/cdi_lines128_complete_frc_curves.png}
\caption{\textbf{Fourier ring correlation for the matched Lines128 CDI rows.} FRC curves provide a spatial-frequency view of the same reconstruction bundle used for Table~\ref{tab:cdi_lines128_complete}.}
\label{fig:cdi_frc_curves}
\end{figure}
```

Expected: FRC curves become visible in the manuscript.

- [ ] **Step 4: Make the six-row table the primary CDI table**

Replace the current two-row `tab:cdi_headline` and mixed `tab:cdi_context` structure with one primary table:

```latex
% table_metadata: task=CDI_lines128; authority=complete_lines128_cdi_benchmark; source=.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/metrics_table.tex; contract=N128_gridsize1_seed3_train2_test2_epochs40_custom_Run1084_probe_pad_extrapolate
\begin{table}[t]
\centering
\small
\caption{\textbf{Matched Lines128 CDI benchmark.} All rows use the same $N=128$ synthetic line-pattern protocol. Lower MAE and higher SSIM are better.}
\label{tab:cdi_lines128_complete}
\begin{tabular}{llrrrr}
\toprule
Model & Training & Amp MAE $\downarrow$ & Phase MAE $\downarrow$ &
Amp SSIM $\uparrow$ & Phase SSIM $\uparrow$ \\
\midrule
CNN & supervised & 0.4027 & 0.7862 & 0.2772 & 0.4964 \\
CNN & PINN & 0.1232 & 0.7863 & 0.7326 & 0.2638 \\
FNO & PINN & 0.1248 & 0.1435 & 0.7409 & 0.9335 \\
FFNO & PINN & 0.0628 & 0.0828 & 0.9348 & 0.9816 \\
SRU-Net & PINN & 0.0269 & \textbf{0.0721} & 0.9881 & \textbf{0.9947} \\
Spectral SRU-Net & PINN & \textbf{0.0249} & 0.0929 & \textbf{0.9899} & 0.9722 \\
\bottomrule
\end{tabular}
\end{table}
```

Expected: the complete six-row comparison is the primary CDI result.

- [ ] **Step 5: Rewrite the surrounding prose**

Replace "SRU-Net + PINN versus historical FFNO-local proxy + PINN" framing with:

```latex
The Lines128 benchmark shows a clear separation between purely local,
purely spectral, and hybrid spectral-convolutional rows. SRU-Net gives the best
phase-side reconstruction among the PINN rows, while the spectral SRU-Net
variant gives the best amplitude-side MAE and SSIM. FFNO improves strongly over
the CNN and FNO PINN rows but does not match the hybrid rows on this CDI task.
```

Expected: the prose matches the six-row table instead of the older two-row subset.

### Task 3: Add The Supervised FFNO Extension As A Control, Not The Primary Table

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Add a compact supervised FFNO control table**

Add this after the primary CDI table or in Additional Experiments:

```latex
% table_metadata: task=CDI_lines128; authority=lines128_supervised_ffno_extension; source=.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-supervised-equivalent-rows/runs/supervised_ffno_extension_20260430T180217Z; comparison_outcome=non_identical_same_contract_comparison
\begin{table}[t]
\centering
\small
\caption{\textbf{FFNO training-procedure control on Lines128 CDI.} The supervised FFNO extension uses the same locked data contract as the FFNO PINN row. Lower MAE and higher SSIM are better.}
\label{tab:cdi_ffno_supervised_control}
\begin{tabular}{lrrrr}
\toprule
Training & Amp MAE $\downarrow$ & Phase MAE $\downarrow$ &
Amp SSIM $\uparrow$ & Phase SSIM $\uparrow$ \\
\midrule
supervised & 0.3864 & \textbf{0.0466} & 0.2484 & 0.9372 \\
PINN & \textbf{0.0628} & 0.0828 & \textbf{0.9348} & \textbf{0.9816} \\
\bottomrule
\end{tabular}
\end{table}
```

Expected: supervised FFNO is visible but does not alter the primary six-row table.

- [ ] **Step 2: Add interpretation text**

Add:

```latex
The supervised FFNO control improves phase MAE but collapses amplitude quality
relative to the historical FFNO-local proxy + PINN row, indicating that the physics-informed objective is not
just a weaker substitute for direct labels in this low-sample CDI setting.
```

Expected: the control clarifies training procedure without overgeneralizing to every architecture.

### Task 4: Keep CNS Table And Figure, But Tighten The Reduced-Data Claim

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Modify if outdated:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/pdebench_cns_sample000_predictions.png`

- [ ] **Step 1: Confirm CNS table values match the emitted bundle**

Run:

```bash
cat .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/cns_paper_table_rows.csv
```

Expected rows include `author_ffno_cns_base`, `spectral_resnet_bottleneck_base`, `fno_base`, and `unet_strong`.

- [ ] **Step 2: Ensure the CNS caption states the reduced-data protocol**

Update the CNS table caption to include:

```latex
All rows use 512 training trajectories, 64 validation trajectories, 64 test
trajectories, two-frame history, and 40 epochs; the table is a reduced-data
secondary benchmark rather than a full-training leaderboard.
```

Expected: no reader can mistake the CNS table for a full-training result.

- [ ] **Step 3: Preserve the CNS figure with the same boundary**

Update the CNS figure caption to include:

```latex
Panels use the same reduced-data protocol and fixed sample as Table~\ref{tab:cns_bundle}.
```

Expected: the figure inherits the CNS table protocol.

- [ ] **Step 4: Remove or soften SOTA language**

Run:

```bash
rg -n "state[- ]of[- ]the[- ]art|SOTA|leaderboard|best|lowest-error" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: "lowest-error" is allowed only within the displayed CNS comparison; no global SOTA claim remains.

### Task 5: Add The Hybrid Skip/Residual Ablation As Secondary Architecture Evidence

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Add an Additional Experiments subsection**

Add:

```latex
\subsection{Hybrid skip and residual-scale controls}
```

Expected: the ablation is not placed before primary results.

- [ ] **Step 2: Add a compact table**

Use:

```latex
% table_metadata: task=CDI_lines128; ablation=hybrid_skip_residual; authority=decision_support; source=.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation
\begin{table}[t]
\centering
\small
\caption{\textbf{Hybrid skip/residual controls on Lines128 CDI.} Rows isolate skip-fusion and residual-scale choices under the same Lines128 protocol. Lower MAE and higher FRC50 are better.}
\label{tab:cdi_skip_residual_ablation}
\begin{tabular}{lrrrr}
\toprule
Variant & Amp MAE $\downarrow$ & Phase MAE $\downarrow$ &
Amp FRC50 $\uparrow$ & Phase FRC50 $\uparrow$ \\
\midrule
SRU-Net anchor & 0.0269 & 0.0721 & 135.46 & 106.80 \\
skip-add & 0.0264 & \textbf{0.0610} & 135.39 & \textbf{135.96} \\
fixed residual scale & \textbf{0.0246} & 0.0773 & \textbf{135.92} & 106.68 \\
skip-add + fixed residual & 0.0289 & 0.0633 & 135.41 & 106.88 \\
\bottomrule
\end{tabular}
\end{table}
```

Expected: the ablation explains architecture choices without displacing the primary table.

- [ ] **Step 3: Add cautious interpretation**

Add:

```latex
Skip-add is the clearest phase-oriented variant, while fixed residual scaling is
the clearest amplitude-oriented variant. Combining the two did not produce a
constructive interaction in this small same-protocol ablation.
```

Expected: the claim matches the summary without promoting the ablation into a headline result.

### Task 6: Replace Planning-Style Future-Work Text

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Update Additional Experiments**

Replace the planning-style paragraph that lists desired synthetic line-pattern
rows as future work with:

```latex
The Lines128 CDI comparison spans local CNN, FNO, FFNO, SRU-Net, and spectral
SRU-Net generators under a shared synthetic line-pattern protocol. Additional
architecture families should be added only when they can use the same reporting
contract and answer a distinct comparison question.
```

Expected: the paper describes the comparison scope without progress-report phrasing.

- [ ] **Step 2: Describe U-NO and WaveBench as outside the reported result set**

Add:

```latex
U-NO and WaveBench are outside the reported result set. They require completed
benchmark rows under the same reporting standards before they can support
quantitative comparisons.
```

Expected: readiness work cannot be mistaken for paper results.

### Task 7: Update Paper Evidence Discoverability

**Files:**
- Modify if needed: `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- Modify if needed: `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`

- [ ] **Step 1: Add manuscript-incorporation pointers if missing**

Add a short section or row that maps:

```text
complete Lines128 CDI bundle -> tab:cdi_lines128_complete, fig:cdi_main_qualitative, fig:cdi_frc_curves
supervised FFNO extension -> tab:cdi_ffno_supervised_control
CNS paper bundle -> tab:cns_bundle, fig:cns_sample_predictions
skip/residual ablation -> tab:cdi_skip_residual_ablation
```

Expected: future readers can trace from completed backlog outputs to manuscript locations.

- [ ] **Step 2: Keep evidence tiers unchanged**

Run:

```bash
rg -n "paper_grade|paper_complete|decision_support|capped_decision_support|bounded" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md
```

Expected: no wording promotes CNS or skip/residual beyond their recorded evidence tier.

### Task 8: Compile And Package

**Files:**
- Modify if successful: `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`

- [ ] **Step 1: Compile the draft in place or in a temporary directory**

Run:

```bash
(cd docs/plans/NEURIPS-HYBRID-RESNET-2026 && pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex)
```

Expected: exit 0. If bibliography warnings remain from the draft state, record them but do not block on unrelated reference cleanup.

- [ ] **Step 2: Run manuscript consistency checks**

Run:

```bash
rg -n "TODO|placeholder|SOTA|leaderboard|full-training|paper-grade CNS|WaveBench.*result|U-NO.*result" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
rg -n "table_metadata|figure_metadata|authority=|source=.artifacts" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected: planning-style future-work language is gone or intentionally scoped; every inserted table/figure has source metadata.

- [ ] **Step 3: Rebuild the paper draft package if the zip is still used**

Run:

```bash
(cd docs/plans/NEURIPS-HYBRID-RESNET-2026 && zip -r scr_ptychography_neurips_draft_package.zip \
  hybrid_resnet_neurips_first_draft.tex \
  hybrid_resnet_neurips_first_draft.pdf \
  figures \
  paper_presentation_style_guide.md \
  paper_evidence_package_audit_summary.md \
  evidence_matrix.md)
```

Expected: zip rebuilds with the updated TeX, PDF, figures, and evidence pointers.

- [ ] **Step 4: Review the final diff**

Run:

```bash
git diff -- docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/figures \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip
```

Expected: only manuscript, paper-local figures, package, and discoverability docs changed.

- [ ] **Step 5: Commit**

Run:

```bash
git add docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/figures \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip
git commit -m "docs: incorporate NeurIPS evidence outputs into draft"
```

Expected: commit contains only the manuscript incorporation work.
