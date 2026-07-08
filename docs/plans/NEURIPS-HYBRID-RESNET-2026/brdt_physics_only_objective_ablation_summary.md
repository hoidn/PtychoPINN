# BRDT Physics-Only Objective Ablation Summary

> **Owning backlog item:** `2026-05-04-brdt-physics-only-objective-ablation`
> **Claim boundary:** `decision_support_append_only`. NOT manuscript or
> paper-grade. BRDT remains a deferred candidate lane only; the required
> NeurIPS 2026 pillars are still CDI `lines128` and PDEBench CNS.

## 1. Identity And Scope

- Initiative: `NEURIPS-HYBRID-RESNET-2026`.
- Lane: Born-Rytov diffraction tomography (BRDT) candidate study.
- Bundle role: append-only neural-row ablation that isolates the training
  objective. The completed four-row preflight is preserved unchanged and
  serves as the read-only baseline.
- Append-only ablation root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-physics-only-objective-ablation/`
- Read-only baseline (supervised image L1 + Born consistency):
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_physics_only_objective_ablation_plan.md`
- Governing design:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`

The baseline supervised+Born bundle, its `metrics.json`, its row
directories, and the manuscript-local assets
`tables/brdt_decision_support_metrics.*` and
`figures/brdt_decision_support_recon.png` are NOT rerun or rewritten by
this item.

## 2. Backlog Question

Does the U-Net / FNO collapse observed under the supervised image L1 +
Born consistency objective persist when the neural rows are trained with
a pure relative-physics objective, and does Hybrid ResNet's gap narrow,
hold, or widen under the same objective change?

## 3. Answer (Decision-Support Read)

- **FNO vanilla:** collapse is primarily *objective-induced*. With the
  supervised image term removed, FNO recovers a meaningful image-space
  and measurement-space agreement.
- **U-Net:** collapse *persists* under the physics-only objective. The
  image-space metric improves only marginally and the output dynamic
  range remains compressed below the target distribution, so U-Net at
  this capacity still cannot represent the full physical-`q` range under
  pure relative-physics training.
- **Hybrid ResNet:** the image-space gap to the model-based Born inverse
  *widens* under the physics-only objective. Hybrid ResNet retains the
  best measurement-space agreement of the three neural rows, but it
  loses its image-space edge once the supervised image L1 signal is
  zeroed out — its baseline lead is jointly attributable to architecture
  *and* to the supervised image term, not to architecture alone.

## 4. Same-Contract Guarantees

Same-contract evidence preserved across both bundles (see
`preflight_manifest.json` in each root):

- dataset id `brdt128_decision_support_preflight`, split 2048 / 256 / 256;
- locked Born forward operator
  (`grid_size=128`, `detector_size=128`, `angle_count=64`,
  `wavelength_px=8.0`, `medium_ri=1.333`, `mode=born`,
  `normalize=unitary_fft`);
- `born_init_image` input mode, `in_channels=1`;
- fixed-sample seed `17` and IDs `[145, 83, 255, 126]`;
- training contract: 20 epochs, batch size 16, learning rate `2e-4`, Adam,
  training seed `42`;
- per-architecture parameter counts unchanged (U-Net 18,465 / FNO 44,465 /
  Hybrid ResNet 142,018).

The only contract change is the neural-row loss weight vector, which the
ablation manifest records under `notes.objective_preset` and
`training_contract.loss_weights`:

| Loss weight       | Baseline | Physics-only ablation |
|-------------------|---------:|----------------------:|
| `image`           | 1.0      | 0.0                   |
| `physics`         | 0.1      | 0.0                   |
| `relative_physics`| 0.1      | 1.0                   |
| `tv`              | 1e-5     | 0.0                   |
| `positivity`      | 1e-4     | 0.0                   |

The classical model-based Born inverse row is *not* rerun; it is
comparison context only.

## 5. Per-Row Comparison

Numbers below are reproduced verbatim from
`comparison_to_supervised_plus_born.json` and
`comparison_to_supervised_plus_born.csv` under the ablation root. Lower
is better for `image_*`, `meas_*`, and `*_rmse`/`*_mae` metrics;
higher is better for `psnr_phys` and `ssim_phys`.

### 5.1 Image-space (physical `q`)

| Row           | Metric                  | Baseline | Physics-only | Δ (ablation − baseline) |
|---------------|-------------------------|---------:|-------------:|------------------------:|
| U-Net         | `image_relative_l2_phys`|   0.8638 |       0.8304 |               -0.0334 |
| U-Net         | `image_rmse_phys`       | 0.004899 |     0.004710 |             -0.000190 |
| U-Net         | `image_mae_phys`        | 0.001827 |     0.002191 |             +0.000364 |
| FNO vanilla   | `image_relative_l2_phys`|   0.9916 |       0.5658 |               -0.4258 |
| FNO vanilla   | `image_rmse_phys`       | 0.005624 |     0.003209 |             -0.002415 |
| FNO vanilla   | `image_mae_phys`        | 0.001947 |     0.002025 |             +0.000079 |
| Hybrid ResNet | `image_relative_l2_phys`|   0.3190 |       0.4990 |               +0.1800 |
| Hybrid ResNet | `image_rmse_phys`       | 0.001809 |     0.002830 |             +0.001021 |
| Hybrid ResNet | `image_mae_phys`        | 0.000650 |     0.001827 |             +0.001176 |

### 5.2 Measurement-space (sinogram)

| Row           | Metric              | Baseline | Physics-only | Δ |
|---------------|---------------------|---------:|-------------:|------------------------:|
| U-Net         | `meas_relative_l2`  |   0.6875 |       0.5616 |              -0.1260 |
| U-Net         | `meas_rmse`         | 0.005823 |     0.004756 |            -0.001067 |
| FNO vanilla   | `meas_relative_l2`  |   0.9808 |       0.2406 |              -0.7403 |
| FNO vanilla   | `meas_rmse`         | 0.008307 |     0.002037 |            -0.006270 |
| Hybrid ResNet | `meas_relative_l2`  |   0.1992 |       0.1574 |              -0.0419 |
| Hybrid ResNet | `meas_rmse`         | 0.001687 |     0.001333 |            -0.000354 |

### 5.3 Supporting diagnostics

| Row           | Metric     | Baseline | Physics-only | Δ |
|---------------|------------|---------:|-------------:|--------:|
| U-Net         | `psnr_phys`|   21.088 |       21.431 |  +0.343 |
| U-Net         | `ssim_phys`|   0.6129 |       0.6404 | +0.0275 |
| FNO vanilla   | `psnr_phys`|   19.889 |       24.763 |  +4.874 |
| FNO vanilla   | `ssim_phys`|   0.6779 |       0.8006 | +0.1227 |
| Hybrid ResNet | `psnr_phys`|   29.741 |       25.854 |  -3.886 |
| Hybrid ResNet | `ssim_phys`|   0.9471 |       0.8573 | -0.0898 |

### 5.4 Output dynamic-range diagnostics (physics-only run only)

For collapse detection. Target physical-`q` distribution has
`q_min=-0.0275`, `q_max=+0.0281`, `q_std≈0.0054` (from baseline
normalization stats):

| Row           | `physical_q_min` | `physical_q_max` | `physical_q_ptp` | `physical_q_mean` | `physical_q_std` |
|---------------|-----------------:|-----------------:|-----------------:|------------------:|-----------------:|
| U-Net         |          -0.0267 |           0.0213 |           0.0479 |            0.0012 |           0.0020 |
| FNO vanilla   |          -0.0257 |           0.0342 |           0.0599 |            0.0013 |           0.0046 |
| Hybrid ResNet |          -0.0386 |           0.0456 |           0.0842 |            0.0014 |           0.0052 |

Notes on dynamic range:

- U-Net's `physical_q_std≈0.0020` is well below the target `0.0054`.
  Together with a still-large `image_relative_l2_phys≈0.83`, this is a
  collapsed-output signature: the network outputs a low-variance,
  near-mean reconstruction that does not span the full sparse target
  distribution.
- FNO vanilla's `physical_q_std≈0.0046` is close to the target std and
  it now meets the measurement contract within `meas_relative_l2≈0.24`,
  so its baseline collapse was largely objective-induced rather than a
  capacity limit.
- Hybrid ResNet's `physical_q_std≈0.0052` is close to target and its
  measurement-space agreement is the best of the three rows. The
  image-space gap reflects loss of the supervised image L1 signal, not a
  loss of representational capacity.

The baseline bundle predates the eval-split `output_dynamic_range`
field, so the comparison file lists `output_dynamic_range.baseline:
null` for the eval-split block. To keep collapse detection populated on
both sides without rerunning or rewriting the baseline, the comparison
also carries a like-for-like
`output_dynamic_range.fixed_sample.{baseline,ablation,delta}` block
computed directly from each bundle's saved
`figures/source_arrays/sample_*_<row>_q_pred.npy` arrays — the same
four fixed samples in both bundles. The fixed-sample window confirms
the same direction as the eval-split read: U-Net's `physical_q_ptp`
shrinks against the baseline (collapse persists), FNO and Hybrid ResNet
move towards the target std on a like-for-like basis. Schema bumps
from `brdt_objective_ablation_comparison_v1` to
`brdt_objective_ablation_comparison_v2` to reflect the added
fixed-sample block and the per-component `final_loss_breakdown.delta`
field.

## 6. Lineage And Append-Only Contract

- `preflight_manifest.json` for the ablation root records:
  - `backlog_item: "2026-05-04-brdt-physics-only-objective-ablation"`,
  - `claim_boundary: "decision_support_append_only"`,
  - `notes.objective_preset: "relative_physics_only"`,
  - `notes.selected_row_ids: ["unet", "fno_vanilla", "hybrid_resnet"]`,
  - `notes.baseline_lineage` with the baseline root, its
    `preflight_manifest.json`, and its `metrics.json` paths.
- The classical Born inverse row in the baseline bundle is not rerun and
  not copied into the ablation root. It remains the same comparator it
  was in the baseline.
- The metric-schema version `brdt_preflight_metrics_v1` is unchanged.
  The new `output_dynamic_range` field lives under
  `runtime.extras.output_dynamic_range`, which the schema explicitly
  treats as optional `extras` metadata.

## 7. Roadmap And Discoverability

- This item is candidate-lane decision support only. It does NOT promote
  BRDT into a required NeurIPS 2026 pillar and does NOT supersede the
  prior `defer_after_preflight` decision recorded in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_preflight_summary.md`.
- The authoritative discoverability surfaces for this read are:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
- `/home/ollie/Documents/neurips/index.md` is intentionally NOT updated;
  this item is not the Phase 5 paper-facing evidence pass.
