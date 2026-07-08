---
priority: 4
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/execution_plan.md
check_commands:
  - python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run
  - python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke
  - python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch
prerequisites:
  - 2026-05-07-brdt-sinogram-input-adapter-contract
related_roadmap_phases:
  - candidate-brdt-sinogram-input
signals_for_selection:
  - The adapter-contract gate has to pass before this item runs.
  - This item produces the current BRDT manuscript evidence for the sinogram-input contract.
  - Existing 2026-05-06 BRDT rows remain Born-image-input lineage and cannot support sinogram-input claims.
---

# Backlog Item: BRDT Sinogram-Input 40-Epoch Paper Evidence

## Objective

Run the BRDT SRU-Net and FFNO learned rows for 40 epochs using measured complex
sinograms as model input, with the model-based Born inverse included only as a
non-learned reference.

## Scope

- Use the decision-support BRDT split `2048 / 256 / 256`.
- Train SRU-Net and FFNO for 40 epochs with supervised object loss plus
  Born-consistency loss, `ReduceLROnPlateau`, and per-epoch loss history.
- Use sample 255 for the manuscript BRDT visual source arrays.
- Emit metrics, combined metrics, source arrays, model profiles, histories,
  and throughput fields under:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/`.
- Refresh the BRDT manuscript table, Figure 3 source, model configuration
  table, efficiency table, evidence manifest, and manuscript zip after the run.

## Required Verification

- `preflight_manifest.json` records `input_mode=sinogram` and `in_channels=2`.
- Both neural rows complete 40 epochs.
- Metrics include image-space error, measurement error, PSNR, SSIM, parameter
  count, and evaluation throughput.
- Figure 3 source arrays include target `q`, measured `s_obs`, Born inverse,
  FFNO, and SRU-Net for sample 255.
- Paper-refresh scripts no longer point BRDT manuscript assets to the old
  `2026-05-06` Born-image-input root.
