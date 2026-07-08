# Manuscript Latest Outcome Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the NeurIPS manuscript, tables, and figures into agreement with the latest completed FFNO, BRDT, and CNS backlog outcomes without reintroducing obsolete FFNO qualifier language.

**Architecture:** Treat the manuscript-facing paper directory as the publication surface and update only claims that are contradicted by newer completed outcomes. CDI FFNO rows are already refreshed and should be verified, not rewritten. CNS should consume the existing plus-U-NO table bundle. BRDT remains a paper-facing result and must use the corrected FFNO metrics and a matching sample-255 visual from corrected source arrays.

**Tech Stack:** LaTeX manuscript, generated `.tex`/`.csv`/`.json` table assets, PNG figure assets, `pdflatex`, `pdftotext`, `rg`, `jq`.

---

## File Map

- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
  - Owns abstract, Experiments, Results, Discussion, table inclusion, figure inclusion, and captions.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.tex`
  - Manuscript-visible CNS table. Must add the completed U-NO row.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.csv`
  - Companion source for the active CNS table. Must match the visible table.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`
  - Companion source for the active CNS table. Either update directly from the plus-U-NO bundle or replace with the plus-U-NO JSON content while preserving the active filename.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.tex`
  - Add CNS U-NO row and remove stale BRDT FFNO parameter count if BRDT table is demoted or corrected.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.csv`
  - Companion source for model config appendix.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json`
  - Companion source for model config appendix. Validate with `jq`.
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png`
  - Regenerate from corrected BRDT sample-255 source arrays so the visual matches the corrected FFNO row.
- Optional create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/brdt_corrected_ffno_metrics.tex`
  - Only if BRDT stays as a visible numeric comparison. Prefer editing the existing inline table in the manuscript unless table reuse is needed.

## Authority Inputs

- CDI corrected FFNO rows:
  - `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`
  - `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`
  - `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh`
- Corrected BRDT FFNO:
  - `2026-05-06-brdt-corrected-ffno-40ep-rerun`
  - Metrics to use in the paper: image relative L2 `0.4509`, measurement relative L2 `0.2906`, PSNR `26.7354`, SSIM `0.9120`, params `27,394`.
  - Matching visual source arrays: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/figures/source_arrays/sample_0255_*`.
- CNS U-NO:
  - `2026-05-06-cns-uno-matched-condition-row-extension`
  - Existing table bundle: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.{tex,csv,json}`
  - U-NO metrics: nRMSE/rel-L2 `0.0384657681`, RMSE `0.9291841984`, low `2.1897497177`, mid `0.0786728188`, high `0.2606226206`, params `1,260,548`.

## Policy Decisions

- Do not reintroduce visible `no-refiner`, `local-refiner`, `proxy`, `decision-support`, or stale FFNO caveat language.
- Do not keep old BRDT FFNO metrics (`0.3324`, `29.38`, `0.9455`, `36.7k`) anywhere in visible manuscript text or active manuscript tables.
- Treat corrected BRDT as paper-facing evidence. Do not include "not paper evidence", "decision-support", "proxy", or similar qualifier language.
- Regenerate both the BRDT table values and the sample-255 figure from corrected BRDT artifacts before rebuilding.
- Do not change CDI metrics unless verification shows a stale value; CDI currently appears to already use corrected four-block FFNO rows.

---

### Task 1: Audit Active Manuscript Surfaces

**Files:**
- Read: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Read: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/*.tex`
- Read: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.{tex,csv,json}`

- [ ] **Step 1: Locate stale values and forbidden visible language**

Run:

```bash
rg -n "0\\.3324|29\\.38|0\\.9455|36\\.7k|0\\.5006|25\\.8262|0\\.8949|0\\.038466|0\\.260623|U-NO|no-refiner|local-refiner|proxy|decision-support|capped" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables
```

Expected:
- Old BRDT values are present before edits.
- CNS active table lacks U-NO before edits.
- Forbidden visible language should not be present in the manuscript; if it appears only in non-included legacy companion assets, do not touch unless the asset is regenerated into the paper.

- [ ] **Step 2: Verify CDI rows are already current**

Run:

```bash
nl -ba docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_pinn_metrics.tex
nl -ba docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_objective_comparison.tex
```

Expected:
- FFNO physics-consistency row has `0.0820`, `0.1380`, `0.8903`, `0.9596`.
- FFNO supervised row has `0.3515`, `0.0661`, `0.2650`, `0.9015`.

- [ ] **Step 3: Decide BRDT treatment from available corrected artifacts**

Run:

```bash
find .artifacts docs/plans/NEURIPS-HYBRID-RESNET-2026 -iname "*brdt*ffno*" -o -iname "*corrected*ffno*"
```

Expected:
- If corrected FFNO source arrays for sample 255 exist, BRDT can be regenerated if user wants it visible.
- If only metrics exist, remove or demote stale visible BRDT figure rather than showing a visual from the old FFNO row.

### Task 2: Update CNS Table to Include U-NO

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.tex`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.csv`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json`

- [ ] **Step 1: Replace active CNS table with plus-U-NO table values**

Use `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.tex` as the source of truth for the five-row visible table. If the plus table contains stale visible labels such as `capped`, keep those only in non-visible metadata if unavoidable; do not show them in the `.tex` table.

Required visible rows:

```tex
FFNO & 0.0198 & 1.130 & 0.042 & 0.102 \\
SRU-Net & 0.0331 & 1.854 & 0.171 & 0.262 \\
FNO & 0.0384 & 2.134 & 0.122 & 0.433 \\
U-NO & 0.0385 & 2.190 & 0.079 & 0.261 \\
U-Net & 0.5386 & 30.988 & 0.645 & 1.743 \\
```

- [ ] **Step 2: Update active CNS CSV/JSON companions**

Copy the U-NO row from `pdebench_cns_matched_condition_metrics_plus_uno.csv` and `.json` into the active `pdebench_cns_matched_condition_metrics.csv` and `.json`.

Expected JSON changes:
- `row_order` includes `neuralop_uno_cns_base` after `fno_base` or before `unet_strong`.
- `rows` includes manuscript label `U-NO`, params `1260548`, and the U-NO run root.
- Metadata may mention source provenance, but avoid visible/generated table wording like `capped`.

- [ ] **Step 3: Validate CNS table assets**

Run:

```bash
jq empty docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json
rg -n "U-NO|0\\.0385|0\\.260|1260548|capped|decision-support" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.*
```

Expected:
- JSON parses.
- U-NO appears in all active CNS table assets.
- `capped` and `decision-support` do not appear in the visible `.tex`; if they remain in CSV/JSON metadata, decide whether they should be normalized to `40-epoch 512/64/64 row`.

### Task 3: Update CNS Manuscript Prose

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`

- [ ] **Step 1: Update abstract CNS sentence**

Replace any sentence that says SRU-Net improves over CNN/U-Net/FNO without mentioning U-NO with:

```tex
On forward compressible Navier--Stokes prediction, FFNO remains strongest;
SRU-Net is second by aggregate error, while U-NO is close to FNO in aggregate
error and stronger at higher frequencies.
```

Use the manuscript’s current punctuation style. Do not use em dashes.

- [ ] **Step 2: Update Results CNS paragraph**

Replace the CNS result paragraph with:

```tex
Table~\ref{tab:cns_bundle} reports the CNS comparison. All models use
five input states, 512/64/64 train/validation/test trajectories,
40 training epochs, MSE loss, and the same optimizer and normalization
settings. FFNO gives the lowest aggregate and Fourier-band errors. SRU-Net is
second by aggregate error. U-NO is nearly tied with FNO on aggregate error, but
has lower mid- and high-frequency errors than FNO. U-Net is the weakest row in
this comparison.
```

- [ ] **Step 3: Update Discussion CNS sentence**

Replace:

```tex
PDEBench CNS shows that SRU-Net improves over FNO and U-Net in the 512/64/64
comparison, but FFNO remains the lowest-error CNS model.
```

with:

```tex
PDEBench CNS shows that FFNO remains the lowest-error model. SRU-Net is second
by aggregate error, while U-NO is close to FNO in aggregate error and stronger
on mid- and high-frequency errors.
```

- [ ] **Step 4: Scan CNS prose**

Run:

```bash
rg -n "CNN|U-NO|FFNO remains|second by aggregate|high-frequency" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected:
- Abstract, Results, and Discussion all mention the updated CNS ordering.
- Abstract no longer claims a CNS CNN comparison.

### Task 4: Correct BRDT Stale Evidence

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex`
- Optional modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png`
- Optional modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.*`

- [ ] **Step 1: Replace stale BRDT numeric claim in the abstract**

Replace the BRDT abstract sentence:

```tex
The same design transfers to Born--Rytov diffraction tomography, where SRU-Net reaches relative
$L_2$ of $0.288$ against $0.332$ for FFNO and $0.380$ for a model-based Born inverse.
```

with:

```tex
On Born--Rytov diffraction tomography, SRU-Net reaches relative $L_2=0.288$
against $0.451$ for FFNO and $0.380$ for a model-based Born inverse.
```

Rationale: the old FFNO value was stale and must be replaced by the corrected 40-epoch FFNO row.

- [ ] **Step 2: Update the BRDT Results prose and table**

Use the corrected 40-epoch values:

```tex
Model-based Born inverse & iterative inverse & 0.3796 & 28.23 & 0.9201 &
-- & 29.6 \\
SRU-Net & supervised + Born & \textbf{0.2876} & \textbf{30.64} & \textbf{0.9552} &
142k & \textbf{375.9} \\
FFNO & supervised + Born & 0.4509 & 26.74 & 0.9120 &
27.4k & 370.1 \\
```

- [ ] **Step 3: Regenerate the BRDT visual from corrected sample-255 arrays**

Regenerate `figures/brdt_sample_0255_context_recon_error.png` with:
- top row: target $q$, model-based Born inverse, FFNO, SRU-Net.
- bottom row: input sinogram magnitude, model-based Born error, FFNO error, SRU-Net error.
- shared target/estimate color scale for $q$ panels.
- separate sinogram color scale.
- shared error color scale for all target-domain error heatmaps.

- [ ] **Step 4: Remove stale old-BRDT values**

Run:

```bash
rg -n "0\\.3324|29\\.38|0\\.9455|36\\.7k|385\\.3|0\\.4509|26\\.74|0\\.9120|27\\.4k" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.*
```

Expected:
- Old values `0.3324`, `29.38`, `0.9455`, `36.7k`, and stale SRU-Net throughput `385.3` are absent from visible manuscript text and active tables.
- Corrected values `0.4509`, `26.74`, `0.9120`, and `27.4k` appear in the visible BRDT table.

### Task 5: Update Model Config Appendix for CNS U-NO and BRDT Decision

**Files:**
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.tex`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.csv`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json`

- [ ] **Step 1: Add CNS U-NO config row**

Add to visible `.tex` table after FNO or before U-Net:

```tex
PDEBench CNS & U-NO & supervised & 32 & 12 & 4 & -- & -- & 1,260,548 \\
```

If the U-NO model profile shows different width/modes/blocks, use the profile values instead of the placeholders above. The user-provided outcome only locks params and metrics, not all architectural hyperparameters.

- [ ] **Step 2: Update CSV/JSON model config sources**

Add row:

```csv
PDEBench CNS,U-NO,neuralop_uno_cns_base,neuralop_uno,supervised,history_len=5 -> next frame; 512 / 64 / 64; 40 epochs,32,12,4,not_applicable,not_recorded,U-NO multiscale operator,1260548,1260548,unique_effective_trainable_params,.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z/model_profile_neuralop_uno_cns_base.json,.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/runs/cns_uno_h5_512cap_40ep_20260507T024412Z/model_profile_neuralop_uno_cns_base.json,40-epoch CNS row
```

Add equivalent JSON object to `model_config_by_benchmark.json`.

- [ ] **Step 3: Handle BRDT FFNO config**

- Update BRDT FFNO params from `36,674` to `27,394`.
- Update source paths to the corrected BRDT FFNO rerun root.

- [ ] **Step 4: Validate model config assets**

Run:

```bash
jq empty docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.json
rg -n "U-NO|1,260,548|1260548|36,674|27,394|decision-support|capped" \
  docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.*
```

Expected:
- U-NO appears with `1,260,548` params.
- No visible `.tex` table contains `decision-support` or `capped`.
- BRDT FFNO parameter count is `27,394`.

### Task 6: Refresh Visual Decisions

**Files:**
- Inspect: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet.png`
- Modify: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/brdt_sample_0255_context_recon_error.png`
- Inspect: `docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/pdebench_cns_sample000_predictions.png`

- [ ] **Step 1: Leave CDI visual unchanged unless provenance is stale**

Run:

```bash
ls -lh docs/plans/NEURIPS-HYBRID-RESNET-2026/figures/cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet.png
rg -n "cdi_lines128_phase_zoom_cnn_fno_ffno_uno_srunet" docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected:
- CDI visual already includes CNN/FNO/FFNO/U-NO/SRU-Net and should remain.

- [ ] **Step 2: Regenerate the BRDT figure**

Regenerate the visible `figure*` asset for `brdt_sample_0255_context_recon_error.png`
from corrected sample-255 arrays. Verify no baked-in stale labels or values exist.

- [ ] **Step 3: Do not add CNS visual for U-NO unless a matched-condition figure exists**

The current CNS figure is adjacent context, not the active five-row table. Do not modify it for U-NO unless a same-contract U-NO visual exists and can be composed with the same sample and normalization.

### Task 7: Rebuild Manuscript and Run Text Scans

**Files:**
- Build output: `docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.pdf`

- [ ] **Step 1: Rebuild LaTeX**

Run:

```bash
cd docs/plans/NEURIPS-HYBRID-RESNET-2026
pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex
pdflatex -interaction=nonstopmode -halt-on-error hybrid_resnet_neurips_first_draft.tex
```

Expected:
- PDF builds successfully.
- Existing underfull/overfull layout warnings may remain.
- No missing figure/table errors.

- [ ] **Step 2: Scan rendered PDF for stale values and forbidden wording**

Run:

```bash
pdftotext docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.pdf - | \
  rg -n "0\\.3324|29\\.38|0\\.9455|36\\.7k|375\\.5|no-refiner|local-refiner|proxy|decision-support|capped|CNN, U-Net, and FNO|SRU-Net improves over CNN"
```

Expected:
- No matches.

- [ ] **Step 3: Scan rendered PDF for required updated CNS language**

Run:

```bash
pdftotext docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.pdf - | \
  rg -n "U-NO|0\\.0385|0\\.261|FFNO remains strongest|second by aggregate|mid- and high-frequency"
```

Expected:
- U-NO table row appears.
- Updated CNS prose appears.

- [ ] **Step 4: Clean LaTeX byproducts**

Run:

```bash
rm -f docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.aux \
      docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.out
```

Expected:
- Only intended `.tex`, table asset, optional figure, and `.pdf` changes remain.

### Task 8: Final Diff Review

**Files:**
- Review all modified files.

- [ ] **Step 1: Check git status**

Run:

```bash
git status --short docs/plans/NEURIPS-HYBRID-RESNET-2026
```

Expected:
- Modified manuscript `.tex` and `.pdf`.
- Modified CNS table `.tex/.csv/.json`.
- Modified model config `.tex/.csv/.json`.
- Optional BRDT figure only if regenerated.
- No `.aux` or `.out`.

- [ ] **Step 2: Review manuscript diff**

Run:

```bash
git diff -- docs/plans/NEURIPS-HYBRID-RESNET-2026/hybrid_resnet_neurips_first_draft.tex
```

Expected:
- Abstract no longer carries stale BRDT FFNO number.
- CNS text reflects U-NO row.
- BRDT visible table/figure are either removed/demoted or corrected consistently.
- No obsolete FFNO qualifier language.

- [ ] **Step 3: Review table diffs**

Run:

```bash
git diff -- docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.* \
            docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/model_config_by_benchmark.*
```

Expected:
- CNS U-NO added consistently.
- Model config U-NO added consistently.
- BRDT FFNO row matches chosen BRDT treatment.

---

## Completion Criteria

- CDI corrected FFNO values remain unchanged and verified.
- CNS visible table includes U-NO and prose describes the revised ordering.
- BRDT uses corrected FFNO metrics and a corrected sample-255 visual.
- Abstract has no stale BRDT number and no CNS claim about a CNN row.
- Rendered PDF contains no forbidden FFNO qualifier/proxy wording.
- LaTeX build succeeds and byproducts are removed.

## Notes for Implementer

- Do not use Python to hand-edit files; use `apply_patch` for manual edits.
- Do not regenerate figures from old BRDT roots.
- Do not stage or commit unless explicitly asked.
- A plan-document-review subagent was not dispatched when this plan was written because current session instructions only allow spawning subagents when the user explicitly asks for delegated or parallel agent work.
