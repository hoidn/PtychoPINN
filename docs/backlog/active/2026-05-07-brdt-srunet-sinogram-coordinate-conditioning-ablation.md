---
priority: 21
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/execution_plan.md
check_commands:
  - python -c "from pathlib import Path; required=[Path('docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_40ep_paper_evidence_summary.md'), Path('.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/combined_metrics.json')]; missing=[str(p) for p in required if not p.exists()]; assert not missing, missing; print('brdt sinogram-input authority present')"
  - python -m compileall -q scripts/studies/born_rytov_dt
prerequisites:
  - 2026-05-07-brdt-sinogram-input-40ep-paper-evidence
related_roadmap_phases:
  - candidate-brdt-sinogram-input
signals_for_selection:
  - The completed BRDT sinogram-input bundle showed weak image-space structure, especially for FFNO and still-improving SRU-Net.
  - The current SRU-Net sinogram-input adapter resizes angle/detector measurements to the object grid before a mostly translation-equivariant model body.
  - A coordinate-conditioned SRU-Net row can test whether explicit position channels help the sinogram-input lane without reopening the completed 40-epoch bundle.
---

# Backlog Item: BRDT SRU-Net Sinogram Coordinate-Conditioning Ablation

## Objective

Add one append-only BRDT sinogram-input ablation row that concatenates normalized
coordinate-grid channels to the SRU-Net input after the measured complex
sinogram has been resized to the object grid.

The purpose is to test whether explicit position conditioning helps SRU-Net
break the inappropriate translation-equivariant symmetry introduced by treating
angle/detector measurements as an image-like grid.

## Scope

- Reuse the completed `2026-05-07-brdt-sinogram-input-40ep-paper-evidence`
  dataset, operator, split, normalization, seed, optimizer, scheduler, loss
  weights, batch size, 40-epoch budget, and sample-255 visual policy.
- Run only one fresh learned row, with a row ID such as
  `sru_net_coordgrid`.
- Preserve the completed unconditioned `sru_net`, `ffno`, and
  `classical_born_backprop` rows by lineage; do not rerun them.
- Add exactly two normalized object-grid coordinate channels, `x` and `y`,
  after the sinogram-to-grid resize and before the SRU-Net body.
- Record adapter metadata showing `input_mode="sinogram"`,
  `sinogram_to_grid="bilinear_resize"`, and `coordinate_channels="object_xy"`.
- Emit metrics, row summary, model profile, history, source arrays, and
  sample-255 compare/error PNGs under a new artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/`.

## Required Comparison

Compare the fresh coordinate-conditioned row against the completed unconditioned
SRU-Net row from `2026-05-07-brdt-sinogram-input-40ep-paper-evidence`.

Report at minimum:

- image relative L2, image RMSE/MAE, PSNR proxy, and `ssim_phys`;
- measurement relative L2, measurement RMSE/MAE;
- parameter count and evaluation throughput;
- final and best observed loss values;
- whether the 40-epoch row is still materially improving at the stop point.

The summary must state that coordinate conditioning is a representational
diagnostic, not a physically principled inverse operator. It may improve the
minimal adapter, but it does not replace a Born adjoint, learned sinogram
encoder, or operator-aware reconstruction path.

## Required Verification

- The new row consumes measured complex sinograms, not `born_init_image`.
- The coordinate channels are deterministic, normalized, batch-broadcasted, and
  included in the saved adapter contract.
- Existing completed BRDT sinogram-input artifacts are read-only lineage inputs.
- The output bundle is append-only and does not overwrite the completed
  `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` root.
- The summary and evidence indexes distinguish the new coordinate-conditioned
  row from the unconditioned SRU-Net authority.
