# SRU-Net Paper Presentation Style Guide

This guide captures the presentation style to use when revising
`hybrid_resnet_neurips_first_draft.tex`. It is aimed at a CVPR-style computer
vision paper: clear visual evidence, compact tables, direct captions, and a
methods section that makes the model and training procedure separable.

## Style Anchors

Use these papers as presentation references, not as citation substitutes.

| Paper | Presentation pattern to borrow | What it means for this paper |
| --- | --- | --- |
| ResNet, CVPR 2016: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf | Architecture figure plus compact error tables. Tables compare specific variants with minimal prose and clear best/worse direction. | Keep the architecture story simple: baseline local model, spectral operator model, and SRU-Net. Use ablations to justify residual, skip, and spectral-local choices. |
| FPN, CVPR 2017: https://openaccess.thecvf.com/content_cvpr_2017/papers/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.pdf | Side-by-side architectural alternatives labelled with small panel letters, followed by component ablations. | Figure 1 should compare architectural families, not only draw SRU-Net internals. Show U-Net/CNN, FNO/FFNO, and SRU-Net in one visual grammar. |
| pix2pix, CVPR 2017: https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf | Qualitative grids are evidence, not decoration. Each row is an example; each column is a method or target; captions tell the reader what visual differences matter. | CDI figures should use shared samples and columns ordered from input/ground truth through baselines to SRU-Net. Error maps should expose local edge and phase failures. |
| Deep Image Prior, CVPR 2018: https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf | Inverse-problem framing, concise method diagram, and task-specific visuals tied to the reconstruction objective. | Treat physics-informed CDI as an inverse imaging paper. Put the forward-model loop in the methods figure and distinguish it from the generator architecture. |
| PDEBench, NeurIPS 2022: https://arxiv.org/abs/2210.07182 | Benchmark tables must name task, metric, and protocol boundaries clearly. | CNS is a secondary cross-domain benchmark. Tables should say "PDEBench CNS" and should not imply CDI physics applies to fluids. |
| F-FNO/FFNO, ICLR 2023: https://arxiv.org/abs/2111.13802 | Operator baselines should be identified as architecture families and compared under matched data/loss conditions. | Write `FFNO + PINN`, `FNO + PINN`, and `SRU-Net + PINN` only when the FFNO row is no-refiner (`fno_cnn_blocks=0`). Historical CDI rows with `fno_cnn_blocks=2` must be labeled `FFNO-local proxy`. Use separate rows for supervised variants. |

## Global Rules

1. Lead with the scientific object: ptychographic reconstruction, not internal
   backlog or run names.
2. Use `SRU-Net` as the paper-facing architecture name.
3. Separate architecture from training procedure. Prefer `Model` and
   `Training` columns, or method names such as `SRU-Net + PINN` only when space
   is tight.
4. Avoid internal process language in the manuscript body: `blocked`,
   `contract`, `cap`, `controlled comparison`, `row lock`, `runner`, `history_len`,
   `seed`, and artifact-root names belong in comments, appendices, or provenance
   files unless they are scientifically relevant.
5. Keep provenance without putting it in the reader's face. Use LaTeX comments
   next to tables for sample counts, data limits, epochs, and source artifacts.
6. Captions should state the task and the visual/quantitative point. Do not use
   captions that only repeat axis labels or column names.
7. Use the first page for the main visual claim: reconstruction quality and
   local error differences should be visible before the reader reaches detailed
   ablations.

## Recommended Paper Flow

1. Abstract: one sentence on the inverse problem, one on SRU-Net, one on CDI
   result, one on CNS secondary benchmark.
2. Introduction: physics-informed ptychography, global-local architecture
   mismatch, proposed spectral-residual U-Net, evidence summary.
3. Related Work: PtychoPINN/physics-informed imaging, FNO/FFNO, U-Net and
   residual image models, hybrid spectral-convolutional models.
4. Methods: physics-informed CDI objective, SRU-Net architecture, training
   procedure notation, CNS adapter.
5. Experiments: CDI primary benchmark first; CNS secondary benchmark second.
6. Results: one headline CDI table, one CDI qualitative figure, one CNS headline
   table, one architecture ablation table.
7. Discussion: what the evidence supports, what remains open, and where the CNS
   benchmark strengthens or limits the architecture claim.

## Figure Style

### Figure Ordering

1. Teaser or Figure 1: qualitative CDI reconstruction comparison.
2. Figure 2: architecture and training loop.
3. Figure 3: CNS qualitative fields, if included in the main paper.
4. Later figures: ablations, convergence, or supplement-only diagnostics.

If page budget allows only two main figures, use:

1. CDI phase reconstruction comparison with a focused crop.
2. Combined architecture/training-loop figure.

### Visual Comparison Rules

Use a stable column order so readers can scan every figure:

1. Input or measurement summary, only if visually interpretable.
2. Ground truth.
3. CNN + PINN.
4. FNO + PINN.
5. FFNO-local proxy + PINN until the no-refiner FFNO rerun lands; then FFNO + PINN.
6. SRU-Net + PINN.
7. Error maps or zoomed error crops.

For CDI qualitative figures:

- Generate the main CDI qualitative figure from reconstruction arrays, not by
  cropping an existing PNG.
- Use amplitude and phase rows with a fixed center crop covering 50% of each
  image dimension.
- Use a shared target-anchored amplitude scale across all method columns.
- Use the global circular phase alignment rule before displaying phase panels.
- Keep the contrast-normalized phase-only version as a diagnostic alternate,
  not as the main qualitative figure.
- Use the same test samples across all models.
- Use shared color limits for reconstructions within a row.
- Use shared error color limits within an error-map row.
- Put SRU-Net at the far right so improvement is read as a left-to-right
  comparison.
- Include zoom boxes on local line intersections, edge crossings, or phase wraps
  when a full image hides the difference.
- Do not include probe amplitude/phase columns unless the point of the figure is
  probe recovery.
- Align each prediction to ground-truth phase by one global circular offset
  before wrapping/display.

For CNS fields:

- Use one or two representative time steps.
- Prefer density and pressure for main-paper visuals; include velocity in the
  supplement unless it is central to the claim.
- Show ground truth, FNO/FFNO, SRU-Net, and absolute error.
- Use identical color limits for each physical field.
- Use one caption sentence to connect errors to shocks, vortices, or coherent
  flow structures.

For inverse-problem figures whose measurement and target domains have different
geometries:

- Do not force measurements into the same method-comparison grid as
  reconstructions.
- Show measurement context separately, then compare only target-domain
  predictions and errors in the rectangular method grid.
- For BRDT-style data, show sinogram magnitude as the measurement context.
- Include a separately verified model-input panel only when that input tensor is
  present in the source arrays.
- Use one shared reconstruction color scale and one shared absolute-error scale
  across all target-domain method panels.
- Use caption language that distinguishes controlled secondary studies from
  primary benchmark claims.

### Architecture Figure Rules

The architecture figure should make the comparison family obvious before it
shows low-level block internals.

Use a two-level layout:

- Top row: `CNN/U-Net`, `FNO/FFNO`, and `SRU-Net`, drawn with comparable input
  and output boxes.
- Bottom row: the SRU-Net block definition and the physics-informed CDI loop.

Keep notation short:

- Encoder block:
  `$z_{\ell+1}=z_\ell+\sigma(S_\ell z_\ell+L_\ell z_\ell)$`
- Bottleneck block:
  `$u_{j+1}=u_j+\alpha_jR_j(u_j)$`
- CDI objective:
  `$\mathcal{L}_{\rm CDI}=\mathcal{D}(x,F_d(F_c(G_\theta(x))))$`

Avoid dense prose inside figure boxes. Put explanatory prose in the caption or
methods text.

## Figure Templates

### Teaser / Main Qualitative Figure

```latex
% figure_metadata: task=CDI_lines128; display_channels=amplitude,phase; crop_fraction=0.5; amplitude_display_scale=gt_crop_min_to_gt_crop_p99; phase_alignment=global_circular_offset_to_gt_before_wrapping; phase_colormap=twilight; phase_display_scale=gt_crop_min_to_gt_crop_p99_after_alignment
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/cdi_lines128_amp_phase_zoom_cnn_fno_ffno_uno_srunet.png}
\caption{\textbf{Amplitude and phase reconstruction comparison on
synthetically-generated diffraction data.} The cropped comparison shows ground
truth, CNN+PINN, FNO+PINN, FFNO+PINN, U-NO+PINN, and SRU-Net+PINN under the
same $N=128$ diffraction data protocol. Amplitude panels share a
target-anchored display scale, and each predicted phase is aligned to the
ground-truth phase by a single global circular offset before display.}
\label{fig:cdi_main_qualitative}
\end{figure}
```

Suggested grid:

| Row | Columns |
| --- | --- |
| Amplitude crop | Ground truth, CNN + PINN, FNO + PINN, FFNO + PINN, U-NO + PINN, SRU-Net + PINN |
| Phase crop | Ground truth, CNN + PINN, FNO + PINN, FFNO + PINN, U-NO + PINN, SRU-Net + PINN |

Alternate contrast-normalized version:

```latex
% figure_metadata: task=CDI_lines128; display_channel=phase; crop_fraction=0.5; phase_alignment=global_circular_offset_to_gt_before_wrapping; phase_colormap=twilight; phase_display_scale=per_panel_p01_to_p99_after_alignment
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet_per_panel_scaled.png}
\caption{\textbf{Contrast-normalized phase reconstruction comparison.}
Each panel uses an independent 1st--99th percentile phase scale after global
phase alignment, so colors reveal within-panel structure but are not directly
comparable across models.}
\label{fig:cdi_main_qualitative_per_panel_scaled}
\end{figure}
```

### Architecture Figure

```latex
\begin{figure*}[t]
\centering
% Prefer a clean vector PDF generated from TikZ or a plotting script.
\includegraphics[width=\linewidth]{figures/srunet_architecture_overview.pdf}
\caption{\textbf{SRU-Net combines global spectral mixing with local image
recovery.} CNN/U-Net baselines rely on local convolutional paths; FNO/FFNO
baselines apply global Fourier mixing; SRU-Net couples a spectral-local encoder,
residual bottleneck, convolutional decoder, and physics-informed CDI loss.}
\label{fig:srunet_architecture}
\end{figure*}
```

Suggested panel layout:

| Panel | Content |
| --- | --- |
| (a) | CNN/U-Net local encoder-decoder baseline |
| (b) | FNO/FFNO spectral operator baseline |
| (c) | SRU-Net spectral-local residual U-Net |
| (d) | CDI training loop: generator, consistency operator, diffraction operator, detector loss |

### CNS Qualitative Figure

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\linewidth]{figures/cns_qualitative_fields.pdf}
\caption{\textbf{PDEBench CNS secondary benchmark.} Shared test frames show
large-scale flow structure together with localized high-gradient regions.
SRU-Net is evaluated as a supervised field predictor; this benchmark tests the
same global-local architectural bias outside the CDI inverse problem.}
\label{fig:cns_qualitative}
\end{figure*}
```

Suggested grid:

| Field | Columns |
| --- | --- |
| Density | Ground truth, FFNO, SRU-Net, FFNO error, SRU-Net error |
| Pressure | Ground truth, FFNO, SRU-Net, FFNO error, SRU-Net error |

## Table Style

### Formatting Rules

- Use `booktabs`; avoid vertical rules.
- Use `\small` or `\scriptsize` only when necessary.
- Use metric arrows in headers: `MAE $\downarrow$`, `SSIM $\uparrow$`.
- Use three decimals for error metrics unless small differences matter.
- Use four decimals for SSIM.
- Bold the best value in each metric column when the comparison is matched.
- Do not bold across mixed-provenance rows.
- Put the primary model last or first consistently. For qualitative figures,
  put it last; for tables, put it first when it is the headline row.
- Use `Model` and `Training` columns when supervised and PINN rows appear in the
  same table.
- If rows have different data limits, training budgets, or provenance, either
  split the table or add a clear `Setting` column.
- Keep sample counts and artifact paths in a preceding LaTeX comment:
  `% table_metadata: task=...; train=...; test=...; epochs=...; source=...`

### Headline CDI Table Template

```latex
% table_metadata: task=CDI_lines128; train_objects=...; test_objects=...;
% train_patches=...; test_patches=...; photons=...; epochs=...; source=...
\begin{table}[t]
\centering
\small
\caption{\textbf{Synthetic line-pattern CDI reconstruction.} Models marked
PINN are trained through the same ptychographic forward model. Lower MAE and
higher SSIM are better.}
\label{tab:cdi_headline}
\begin{tabular}{llrrrr}
\toprule
Model & Training & Amp MAE $\downarrow$ & Phase MAE $\downarrow$ &
Amp SSIM $\uparrow$ & Phase SSIM $\uparrow$ \\
\midrule
CNN & PINN & ... & ... & ... & ... \\
FNO & PINN & ... & ... & ... & ... \\
FFNO & PINN & ... & ... & ... & ... \\
SRU-Net & PINN & \textbf{...} & \textbf{...} & \textbf{...} & \textbf{...} \\
\bottomrule
\end{tabular}
\end{table}
```

Use this table as the primary CDI quantitative result. Add supervised rows only
after the supervised equivalents exist:

```latex
CNN & Supervised & ... & ... & ... & ... \\
FNO & Supervised & ... & ... & ... & ... \\
FFNO & Supervised & ... & ... & ... & ... \\
SRU-Net & Supervised & ... & ... & ... & ... \\
```

### CDI Context Table Template

Use this only when rows are not fully matched but are still useful context. Do
not bold mixed-provenance rows.

```latex
% table_metadata: task=CDI_lines128; provenance=mixed; source=...
\begin{table}[t]
\centering
\small
\caption{\textbf{CDI SSIM context across available line-pattern runs.} Rows
summarize available evidence and should not be read as a single matched
benchmark unless the setting column matches.}
\label{tab:cdi_context}
\begin{tabular}{lllrr}
\toprule
Model & Training & Setting & Amp SSIM $\uparrow$ & Phase SSIM $\uparrow$ \\
\midrule
CNN & Supervised & legacy line-pattern run & ... & ... \\
CNN & PINN & legacy line-pattern run & ... & ... \\
FNO & PINN & legacy line-pattern run & ... & ... \\
FFNO & PINN & matched SRU-Net/FFNO run & ... & ... \\
SRU-Net & PINN & matched SRU-Net/FFNO run & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

### CNS Headline Table Template

```latex
% table_metadata: task=PDEBench_CNS; train_trajectories=...;
% val_trajectories=...; test_trajectories=...; windows=...; epochs=...; source=...
\begin{table}[t]
\centering
\small
\caption{\textbf{PDEBench CNS secondary benchmark.} All rows use the same
supervised next-state prediction task. Lower values are better.}
\label{tab:cns_headline}
\begin{tabular}{lrrrr}
\toprule
Model & nRMSE $\downarrow$ & Rel. $L_2$ $\downarrow$ &
High-band RMSE $\downarrow$ & Params \\
\midrule
U-Net & ... & ... & ... & ... \\
FNO & ... & ... & ... & ... \\
FFNO & \textbf{...} & \textbf{...} & \textbf{...} & ... \\
SRU-Net & ... & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

If FFNO is the strongest observed CNS row, say exactly that. Do not call it
global state of the art unless a same-protocol comparison set supports the
claim.

### Component Ablation Table Template

```latex
% table_metadata: task=PDEBench_CNS; ablation=SRU-Net_components; source=...
\begin{table}[t]
\centering
\small
\caption{\textbf{SRU-Net component ablation on PDEBench CNS.} The table varies
one architectural component at a time.}
\label{tab:srunet_ablation}
\begin{tabular}{lccccrr}
\toprule
Model & Spectral & Residual & Skip & Pixelshuffle &
nRMSE $\downarrow$ & High-band RMSE $\downarrow$ \\
\midrule
Local U-Net & -- & -- & \checkmark & -- & ... & ... \\
Spectral encoder & \checkmark & -- & -- & -- & ... & ... \\
SRU-Net base & \checkmark & \checkmark & -- & -- & ... & ... \\
SRU-Net & \checkmark & \checkmark & \checkmark & \checkmark & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

This table should replace scattered single-purpose tables when page budget is
tight.

## Caption Templates

Use this shape:

```latex
\caption{\textbf{Short claim sentence.} One sentence names the task and
training/evaluation condition. One sentence tells the reader what visual or
metric difference matters.}
```

Good examples:

- `\textbf{Synthetic line-pattern CDI reconstruction.} All models are trained
  through the same ptychographic forward model. SRU-Net preserves line edges and
  phase continuity more accurately than the spectral-operator baselines.`
- `\textbf{SRU-Net component ablation on PDEBench CNS.} The same supervised
  next-state prediction task is used for every row. Skip fusion and pixelshuffle
  improve local high-frequency recovery without changing the benchmark task.`

Avoid:

- Captions that only list columns.
- Captions that mention internal artifact state.
- Captions that make claims not visible in the table or figure.

## Model Naming

Use paper-facing names consistently.

| Internal / older name | Paper-facing name |
| --- | --- |
| `hybrid_resnet`, `pinn_hybrid_resnet`, `Hybrid ResNet`, `SCR` | `SRU-Net` |
| `pinn_hybrid_resnet` row | `SRU-Net + PINN` or `Model=SRU-Net, Training=PINN` |
| historical `pinn_ffno` row with `fno_cnn_blocks=2` | `FFNO-local proxy + PINN` |
| corrected `pinn_ffno` row with `fno_cnn_blocks=0` | `FFNO + PINN` |
| `pinn_fno` row | `FNO + PINN` |
| `baseline` CNN row trained without the forward-model loss | `CNN`, `Training=Supervised` or `Training=Non-PINN` depending on the actual run |
| `pinn` CNN row | `CNN + PINN` |

If the exact training procedure is uncertain, do not guess. Use a neutral note
in a provenance comment and omit the row from the main matched table until it is
verified.

## Decimal Precision

Recommended defaults:

| Metric type | Precision |
| --- | --- |
| MAE, RMSE, nRMSE, relative $L_2$ | 3 decimals in main paper; 4-6 in supplement if needed |
| SSIM | 4 decimals |
| PSNR | 2 decimals |
| Runtime | nearest second or one decimal, depending on spread |
| Parameters | compact format such as `8.39M` |

When a difference is smaller than the rounding unit, either increase precision
or do not describe the difference as meaningful.

## Hidden Metadata Comments

Every table should carry enough hidden metadata to reconstruct where it came
from. Keep it in comments directly above the table.

```latex
% table_metadata: task=CDI_lines128; model_rows=SRU-Net+PINN,FFNO+PINN;
% train_objects=2; test_objects=2; train_patches=8978; test_patches=1458;
% epochs=40; source=.artifacts/.../metrics_table.csv
```

For CNS:

```latex
% table_metadata: task=PDEBench_CNS; train_trajectories=512;
% val_trajectories=64; test_trajectories=64; train_windows=4096;
% val_windows=512; test_windows=512; epochs=40; source=.artifacts/...
```

These comments should not leak into prose unless the data limit is necessary to
interpret the claim.

## Main-Paper Table Budget

For a CVPR-length submission, target:

1. One CDI headline metrics table.
2. One CDI qualitative figure.
3. One architecture figure.
4. One CNS headline table.
5. One compact ablation table.

Move the following to the appendix unless they become central to the claim:

- Every intermediate CNS temporal-context table.
- Fourier-mode sweeps.
- Runtime-heavy convergence tables.
- Mixed-provenance context rows.
- Per-run provenance details.

## Revision Checklist

Before adding or revising a manuscript table or figure, check:

1. Does the caption state the scientific point?
2. Are model and training procedure separable?
3. Are all compared rows on the same task and setting?
4. If not, is the table explicitly marked as context rather than a matched
   comparison?
5. Are best values bolded only inside matched comparisons?
6. Are error maps, color scales, and sample order shared across visual methods?
7. Is provenance preserved in a comment without adding internal language to the
   paper body?
8. Would a reader understand the claim from the table/figure without reading
   implementation notes?
