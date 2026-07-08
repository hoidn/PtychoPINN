# BRDT Sinogram-Input 40-Epoch Paper Evidence Summary

> **Owning backlog item:** `2026-05-07-brdt-sinogram-input-40ep-paper-evidence`
> **Final claim boundary:** `paper_evidence_brdt_additive`
> **Promotion status:** `passed`
> **Lane status:** additive secondary evidence only. This bundle may support
> bounded BRDT manuscript context, but it does **not** replace the required
> CDI `lines128` or PDEBench CNS pillars and does **not** authorize
> `/home/ollie/Documents/neurips/` publication on its own.
> **Contract note:** this is the current BRDT learned-model authority for the
> `input_mode="sinogram"` lane. Historical `2026-05-05` and `2026-05-06`
> BRDT bundles remain valid only for the older `born_init_image` contract.

## 1. Identity And Locked Contract

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Lane: Born-Rytov diffraction tomography (BRDT) additive candidate study
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-brdt-sinogram-input-40ep-paper-evidence/`
- Prerequisite adapter authority:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_sinogram_input_adapter_contract.md`
- Historical Born-image-input lineage retained for provenance only:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-40ep-rerun/`

Locked same-contract fields preserved in this bundle:

- dataset id `brdt128_decision_support_preflight`
- split `2048 / 256 / 256`
- operator `BornRytovForward2D`, Born mode, `N=D=128`, `A=64`,
  `wavelength_px=8.0`, `medium_ri=1.333`, `normalize=unitary_fft`
- loss weights `image=1.0`, `physics=0.1`, `relative_physics=0.1`,
  `tv=1e-5`, `positivity=1e-4`
- optimizer Adam, `lr=2e-4`, batch size `16`, seed `42`, `40` epochs,
  `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-5)`
- fixed paper sample `255`

Changed fields versus the historical BRDT manuscript context:

- learned-model input contract: `input_mode="sinogram"`
- learned rows: `ffno`, `sru_net`
- model-based Born inverse retained only as a non-learned reference row

Comparison standard: exact equality for locked contract fields. Historical
`born_init_image` bundles are context only and are not same-contract numerical
promotion targets for this learned sinogram-input lane.

## 2. Outcome

The live rerun completed successfully under tracked PID `3253923`, wrote the
required provenance/gate/manifests, and passed the paper-evidence gate under
`paper_evidence_brdt_additive`.

Within the new sinogram-input contract, SRU-Net is the strongest learned row:
it beats FFNO and the non-learned Born inverse on image-space relative error,
measurement-space relative error, PSNR, and SSIM. The bundle is therefore the
current repo-local BRDT additive-secondary manuscript authority.

The numerical read is materially weaker than the preserved historical
`born_init_image` paired bundle. That regression is real and should be read as
part of the contract change, not hidden by reusing the old Born-image-input
authority.

## 3. Metric Read

| Row | image_relative_l2_phys | meas_relative_l2 | PSNR_phys | SSIM_phys | Params | Samples/s |
|---|---:|---:|---:|---:|---:|---:|
| `classical_born_backprop` | `3.4686` | `7.6931` | `9.0132` | `0.6246` | `0` | `356.4` |
| `ffno` | `1.0000` | `0.9999` | `19.8166` | `0.0508` | `27,538` | `387.3` |
| `sru_net` | `0.7165` | `0.4666` | `22.7117` | `0.7359` | `142,162` | `404.7` |

Convergence read from `convergence_audit.json`:

- both learned rows emitted the planned `40` history records
- `ffno` reduced LR five times and finished at `1e-5`
- `sru_net` kept LR at `2e-4` with no reductions
- both rows still register `materially_improving_at_stop=true`

Historical-context-only comparison against the prior `born_init_image`
authority:

- `sru_net` versus historical `hybrid_resnet`: `image_relative_l2_phys`
  `+0.4289`, `meas_relative_l2` `+0.2923`, `PSNR_phys -7.93`, `SSIM_phys -0.219`
- `ffno` versus historical corrected `ffno`: `image_relative_l2_phys`
  `+0.5491`, `meas_relative_l2` `+0.7093`, `PSNR_phys -6.92`, `SSIM_phys -0.861`

## 4. Sample-255 Visual Provenance

The paper-facing visual authority for this lane is the sample-`255` bundle
under:

- `visuals/sample_0255_compare_q.png`
- `visuals/sample_0255_error_q.png`
- `visuals/sample_0255_sinogram_residual.png`
- `figures/source_arrays/sample_0255_*`

The visible row roster is:

- measured sinogram magnitude
- target `q`
- `classical_born_backprop`
- `ffno`
- `sru_net`

The Born inverse remains a non-learned reference only. The learned rows are
fresh sinogram-input predictions produced in this bundle.

## 5. Gate Result

- Final claim boundary: `paper_evidence_brdt_additive`
- Promotion status: `passed`
- Failed gate checks: none

The passed gate is backed by:

- tracked-PID runtime provenance (`tracked_pid=3253923`)
- `run_exit_status.json` with `exit_code=0`
- dataset identity and split manifests
- `metrics.{json,csv}`, `combined_metrics.{json,csv}`, `combined_manifest.json`
- `convergence_audit.{json,csv}`
- sample-`255` visual/source-array bundle
- checked-in repo-local manuscript/discoverability surfaces updated to this
  backlog item and artifact root

This promotion remains deliberately narrow:

- BRDT is additive secondary context only
- it does not replace CDI `lines128` or PDEBench CNS
- it does not authorize same-protocol full-training BRDT competitiveness claims
- it does not authorize `/home/ollie/Documents/neurips/` publication by itself

## 6. Residual Risks

- Both learned rows were still materially improving at stop, so the correct
  read is bounded additive evidence rather than fully converged BRDT
  performance.
- The current sinogram-input lane regresses materially versus the preserved
  historical `born_init_image` authority; future comparisons must not blur the
  two contracts together.
- `preflight_manifest.json` preserves row-local `input_mode="sinogram"` truth,
  but its top-level legacy `input_mode` / `in_channels` fields remain unset.
  Downstream consumers should read the row-local contract fields or the
  adapter-contract summary for the authoritative learned-input definition.

## 7. Artifact Inventory

- provenance:
  `invocation.json`, `invocation.sh`, `runtime_provenance.json`,
  `run_exit_status.json`
- metrics:
  `metrics.json`, `metrics.csv`, `combined_metrics.json`,
  `combined_metrics.csv`, `metric_schema.json`, `combined_manifest.json`
- gate and audit:
  `paper_evidence_gate.json`, `convergence_audit.json`,
  `convergence_audit.csv`
- manifests:
  `dataset_identity_manifest.json`, `split_manifest.json`,
  `visual_manifest.json`, `preflight_manifest.json`
- row-local outputs:
  `rows/ffno/{history.json,history.csv,model_profile.json,row_summary.json}`,
  `rows/sru_net/{history.json,history.csv,model_profile.json,row_summary.json}`,
  `rows/classical_born_backprop/row_summary.json`
- visuals:
  `visuals/sample_0255_compare_q.png`, `visuals/sample_0255_error_q.png`,
  `visuals/sample_0255_sinogram_residual.png`,
  `figures/source_arrays/sample_0255_*`
