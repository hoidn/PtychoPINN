# BRDT SRU-Net Sinogram Coordinate-Conditioning Ablation Plan

## Goal

Test whether adding explicit object-grid coordinate channels improves the BRDT
sinogram-input SRU-Net row, without rerunning the completed unconditioned
sinogram-input bundle.

## Contract

- Backlog item:
  `2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation`
- Roadmap phase: `candidate-brdt-sinogram-input`
- Prerequisite authority:
  `2026-05-07-brdt-sinogram-input-40ep-paper-evidence`
- Claim boundary: append-only diagnostic inside the BRDT sinogram-input lane.
- New artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation/`

## Implementation Scope

1. Add a coordinate-conditioned SRU-Net sinogram-input adapter path.
   - Start from `BRDTSinogramInputAdapter`.
   - Keep measured complex sinogram input shaped `(B, 2, 64, 128)`.
   - Resize to the object grid exactly as the unconditioned adapter does.
   - Concatenate deterministic normalized object-grid channels `x` and `y`.
   - Feed the resulting four-channel tensor to the SRU-Net body.
   - Record metadata:
     - `input_mode="sinogram"`
     - `sinogram_to_grid="bilinear_resize"`
     - `coordinate_channels="object_xy"`
     - `in_channels=4`

2. Add a narrow runner or runner mode for only the new row.
   - Reuse the completed 40-epoch BRDT sinogram-input settings:
     - split `2048 / 256 / 256`
     - seed `42`
     - batch size `16`
     - 40 epochs
     - Adam at `2e-4`
     - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
     - same supervised image loss plus Born-consistency loss
     - sample `255` visual/source-array policy
   - Do not rerun `classical_born_backprop`, `ffno`, or unconditioned
     `sru_net`.
   - Load completed comparator metrics/source arrays by lineage from
     `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/`.

3. Publish an append-only comparison bundle.
   - New row artifacts:
     - `rows/sru_net_coordgrid/history.{json,csv}`
     - `rows/sru_net_coordgrid/model_profile.json`
     - `rows/sru_net_coordgrid/row_summary.json`
   - Aggregate artifacts:
     - `combined_metrics.{json,csv}`
     - `comparison_summary.md`
     - `convergence_audit.{json,csv}`
     - `figures/source_arrays/sample_0255_*`
     - `visuals/sample_0255_compare_q.png`
     - `visuals/sample_0255_error_q.png`
     - `visuals/sample_0255_sinogram_residual.png`

4. Update discoverability.
   - Add a durable summary at:
     `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_srunet_sinogram_coordinate_conditioning_ablation_summary.md`
   - Add append-only references to:
     - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
     - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
   - Do not rewrite the completed BRDT sinogram-input authority or manuscript
     assets unless a later paper-refresh item explicitly promotes the result.

## Verification

Run the narrowest checks first:

```bash
pytest -q tests/studies/test_born_rytov_dt_adapters.py -k "sinogram or input_mode"
pytest -q tests/studies/test_born_rytov_dt_preflight.py -k "sinogram_input_40ep or input_mode"
python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run
python -m compileall -q scripts/studies/born_rytov_dt
```

After the row finishes, verify:

```bash
python - <<'PY'
from pathlib import Path
root = Path(".artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-srunet-sinogram-coordinate-conditioning-ablation")
required = [
    root / "combined_metrics.json",
    root / "rows/sru_net_coordgrid/row_summary.json",
    root / "rows/sru_net_coordgrid/model_profile.json",
    root / "visuals/sample_0255_compare_q.png",
    root / "visuals/sample_0255_error_q.png",
    root / "figures/source_arrays/sample_0255_sru_net_coordgrid_q_pred.npy",
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing coordinate-conditioning artifacts: {missing}")
print("brdt coordinate-conditioning artifacts present")
PY
```

## Non-Goals

- Do not add FFNO coordinate conditioning in this item.
- Do not compare multiple coordinate encodings.
- Do not rerun completed unconditioned rows.
- Do not relabel this as a physics-aware inverse operator.
- Do not promote BRDT above CDI or CNS evidence pillars.
