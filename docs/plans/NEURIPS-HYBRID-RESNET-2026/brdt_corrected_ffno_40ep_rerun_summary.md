# BRDT Corrected FFNO 40-Epoch Rerun Summary

> **Owning backlog item:** `2026-05-06-brdt-corrected-ffno-40ep-rerun`
> **Final claim boundary:** `paper_evidence_brdt_additive`
> **Promotion status:** `passed`
> **Lane status:** additive secondary evidence only. This bundle may support
> bounded BRDT manuscript context, but it does **not** replace the required
> CDI `lines128` or PDEBench CNS pillars and does **not** authorize
> `/home/ollie/Documents/neurips/` publication on its own.

## 1. Identity And Lineage

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Lane: Born-Rytov diffraction tomography (BRDT) additive candidate study
- Corrected `40`-epoch artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/`
- Read-only same-contract baseline:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
- Corrected pure-FFNO `20`-epoch prerequisite:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/`
- Historical proxy-only `40`-epoch bundle:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
- Execution plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/execution_plan.md`

This item reruns exactly two neural rows, `hybrid_resnet` and the corrected
no-refiner `ffno`, under the locked supervised+Born `2048 / 256 / 256` BRDT
contract with `40` epochs, Adam `2e-4`, batch `16`, seed `42`, and
`ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`.

## 2. Same-Contract And No-Refiner Audit

Comparison standard: exact equality for locked contract fields, with numerical
checks taken from the machine-written manifests.

The corrected `40`-epoch bundle preserves the baseline contract on:

- dataset id `brdt128_decision_support_preflight`
- split counts `2048 / 256 / 256`
- fixed sample ids `[145, 83, 255, 126]`
- input mode `born_init_image`
- operator geometry `N=D=128`, `A=64`, Born mode, `unitary_fft`,
  `wavelength_px=8.0`, `medium_ri=1.333`
- train-only physical-`q` normalization
- loss weights `image=1.0`, `physics=0.1`, `relative_physics=0.1`,
  `tv=1e-5`, `positivity=1e-4`

The corrected FFNO row remains the no-refiner BRDT adapter:

- architecture: `SpatialLifter -> SharedFactorizedFfnoBottleneck -> 1x1`
- `cnn_blocks`: absent by construction
- hidden channels `16`, modes `8`, blocks `4`, `mlp_ratio=2.0`
- parameter count `27394`

Machine-readable authorities:

- `preflight_manifest.json`
- `combined_manifest.json`
- `rows/ffno/model_profile.json`
- `paper_evidence_gate.json`

## 3. 20-Epoch To 40-Epoch Read

| Row | image_relative_l2_phys | meas_relative_l2 | PSNR_phys | SSIM_phys | Final LR | LR drops | Improving at stop |
|---|---:|---:|---:|---:|---:|---:|---|
| `hybrid_resnet` | `0.3190 -> 0.2876` (`-0.0314`) | `0.1992 -> 0.1743` (`-0.0250`) | `29.74 -> 30.64` (`+0.90`) | `0.9471 -> 0.9552` (`+0.0081`) | `1e-4` | `1` | `true` |
| `ffno` | `0.5006 -> 0.4509` (`-0.0498`) | `0.3395 -> 0.2906` (`-0.0489`) | `25.83 -> 26.74` (`+0.91`) | `0.8949 -> 0.9120` (`+0.0171`) | `2e-4` | `0` | `true` |

Final `40`-epoch metrics:

- `hybrid_resnet`: `image_relative_l2_phys=0.2876`,
  `meas_relative_l2=0.1743`, `psnr_phys=30.6404`, `ssim_phys=0.9552`,
  `parameter_count=142018`
- `ffno`: `image_relative_l2_phys=0.4509`, `meas_relative_l2=0.2906`,
  `psnr_phys=26.7354`, `ssim_phys=0.9120`, `parameter_count=27394`

Interpretation:

- Hybrid ResNet remains the strongest image-space BRDT row.
- The corrected pure-FFNO row remains materially weaker than Hybrid ResNet on
  this capped contract, but the paired `40`-epoch comparison is now clean:
  it no longer depends on the historical FFNO-local-refiner proxy.
- Both rows were still materially improving at stop, so this is additive
  secondary evidence, not a full-convergence or full-training BRDT claim.

## 4. Sample-255 Visual Provenance

The paper-facing sample authority is the corrected paired rerun bundle:

- `visuals/sample_0255_compare_q.png`
- `visuals/sample_0255_error_q.png`
- `visuals/sample_0255_sinogram_residual.png`
- `figures/source_arrays/sample_0255_*`

The model-based comparator remains the same-contract baseline row
`classical_born_backprop`, consumed by lineage from the frozen
`2026-04-29-brdt-four-row-preflight` bundle. The corrected paired rerun emits
fresh `hybrid_resnet` and `ffno` predictions plus source arrays in the same
root, so the visible `255` bundle is reproducible from checked source arrays.

## 5. Gate Result

- Final claim boundary: `paper_evidence_brdt_additive`
- Promotion status: `passed`
- Failed gate checks: none

The gate passed after the corrected paired rerun, runtime provenance, PID-based
exit-status proof, same-contract lineage surfaces, sample-`255` visual bundle,
and checked-in evidence surfaces were all made mutually consistent for this
backlog item and artifact root.

This promotion is intentionally narrow:

- BRDT remains additive secondary evidence only
- it does not replace CDI `lines128` or PDEBench CNS
- it does not authorize same-protocol full-training BRDT competitiveness claims
- it does not authorize `/home/ollie/Documents/neurips/` publication by itself

## 6. Residual Risks

- Both neural rows were still materially improving at stop; the correct read is
  bounded secondary evidence, not fully converged BRDT performance.
- The sample-`255` classical comparator is inherited by lineage rather than
  regenerated in this pass.
- The historical `2026-05-05` bundle remains preserved for provenance, but its
  FFNO row is proxy-only context and should not be cited as pure FFNO.

## 7. Artifact Inventory

- top-level provenance:
  `invocation.json`, `invocation.sh`, `runtime_provenance.json`,
  `run_exit_status.json`
- top-level metrics:
  `metrics.json`, `metrics.csv`, `combined_metrics.json`,
  `combined_metrics.csv`, `metric_schema.json`
- gate and audit:
  `paper_evidence_gate.json`, `convergence_audit.json`,
  `convergence_audit.csv`
- row-local outputs:
  `rows/hybrid_resnet/history.{json,csv}`, `rows/hybrid_resnet/model_profile.json`,
  `rows/ffno/history.{json,csv}`, `rows/ffno/model_profile.json`
- visuals:
  `visuals/sample_0255_compare_q.png`, `visuals/sample_0255_error_q.png`,
  `visuals/sample_0255_sinogram_residual.png`,
  `figures/source_arrays/sample_0255_*`
