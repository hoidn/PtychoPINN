# BRDT Corrected FFNO Row Rerun Summary

> **Owning backlog item:** `2026-05-06-brdt-corrected-ffno-row-rerun`
> **Claim boundary:** `decision_support_append_only`. NOT manuscript or
> paper-grade evidence. BRDT remains a deferred candidate lane only; the
> required NeurIPS 2026 pillars are still CDI `lines128` and PDEBench CNS, and
> the `defer_after_preflight` decision in `brdt_preflight_summary.md` stands.
> **Pure-FFNO authority:** this summary and the corrected artifact root below
> are now the discoverable authority for BRDT pure-FFNO `20`-epoch reads.
> Historical `2026-05-04` and `2026-05-05` FFNO rows remain preserved
> local-refiner proxy lineage only.

## 1. Identity And Scope

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Lane: Born-Rytov diffraction tomography (BRDT) candidate study
- Corrected artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/`
- Read-only baseline four-row bundle:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
- Historical FFNO-local-refiner proxy root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/`
- Execution plan:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/execution_plan.md`

This item reruns exactly one BRDT row, `ffno`, under the original supervised +
Born `20`-epoch contract while keeping the baseline four-row bundle and the
historical FFNO extension root read-only.

## 2. Same-Contract Audit

Comparison standard: exact field equality for all locked contract surfaces named
in the plan.

The corrected rerun matches the baseline four-row preflight on every
fairness-relevant field:

- dataset id: `brdt128_decision_support_preflight`
- split counts: `2048 / 256 / 256`
- normalization stats from the baseline dataset manifest
- input mode: `born_init_image`
- operator pointer:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
- operator geometry: `N=D=128`, `A=64`, `wavelength_px=8.0`,
  `medium_ri=1.333`, Born mode, `unitary_fft`
- fixed sample ids: `[145, 83, 255, 126]`
- training contract: `20` epochs, batch `16`, Adam, `lr=2e-4`, seed `42`
- loss weights: `image=1.0`, `physics=0.1`, `relative_physics=0.1`,
  `tv=1e-5`, `positivity=1e-4`

Machine-readable audit:
`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/same_contract_audit.json`

## 3. Corrected FFNO Contract

The executed corrected row is the no-refiner BRDT FFNO adapter:

- architecture: `SpatialLifter -> SharedFactorizedFfnoBottleneck -> 1x1`
- `cnn_blocks`: rejected by construction
- hidden channels: `16`
- `fno_modes`: `8`
- `fno_blocks`: `4`
- `share_spectral_weights`: `false`
- `mlp_ratio`: `2.0`
- parameter count: `27394`

Row-local profile authority:
`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/rows/ffno/model_profile.json`

## 4. Result Versus Historical Proxy

The corrected pure-FFNO row is architecturally cleaner but materially weaker
than the historical FFNO-local-refiner proxy.

| Metric | Historical proxy | Corrected pure FFNO | Delta |
|---|---:|---:|---:|
| `image_relative_l2_phys` | 0.3421 | 0.5006 | +0.1585 |
| `meas_relative_l2` | 0.1859 | 0.3395 | +0.1537 |
| `psnr_phys` | 29.1326 | 25.8262 | -3.3064 |
| `ssim_phys` | 0.9420 | 0.8949 | -0.0470 |
| `parameter_count` | 36674 | 27394 | -9280 |

Corrected row runtime:

- `wall_time_train_s`: `108.16`
- `wall_time_eval_s`: `0.70`
- device: `NVIDIA GeForce RTX 3090`

## 5. Interpretation

- The historical `2026-05-04` FFNO row should now be read only as
  `FFNO-local proxy` lineage.
- The corrected `2026-05-06` row is the only pure-BRDT-FFNO `20`-epoch
  authority under the locked supervised+Born contract.
- The cleaner contract does not improve BRDT competitiveness on this capped
  lane; it weakens both image-space and measurement-space metrics relative to
  the historical proxy.
- BRDT remains deferred candidate context. This rerun repairs lineage and
  discoverability; it does not promote BRDT into the required NeurIPS evidence
  package.

## 6. Artifact Inventory

- corrected root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-brdt-corrected-ffno-row-rerun/`
- top-level provenance:
  `invocation.json`, `invocation.sh`
- row-local provenance:
  `rows/ffno/invocation.json`, `rows/ffno/invocation.sh`
- row-local outputs:
  `rows/ffno/row_summary.json`, `rows/ffno/model_profile.json`,
  `rows/ffno/history.json`, `rows/ffno/model_state.pt`
- bundle outputs:
  `preflight_manifest.json`, `metrics.json`, `metrics.csv`,
  `metric_schema.json`, `visual_manifest.json`, `combined_metrics.json`,
  `combined_metrics.csv`, `combined_manifest.json`
- audit outputs:
  `same_contract_audit.json`, `same_contract_audit.md`
