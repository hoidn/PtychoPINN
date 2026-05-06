# BRDT FFNO Row Extension Summary

> **Owning backlog item:** `2026-05-04-brdt-ffno-row-extension`
> **Claim boundary:** `decision_support_append_only`. NOT manuscript or
> paper-grade. BRDT remains a deferred candidate lane only; the required
> NeurIPS 2026 pillars are still CDI `lines128` and PDEBench CNS, and the
> `defer_after_preflight` decision in `brdt_preflight_summary.md` stands.
> **Post-hoc architecture caveat (2026-05-06):** this historical row was
> generated before the BRDT FFNO adapter was corrected to remove
> post-bottleneck CNN refiners. Treat the recorded metrics as a legacy
> FFNO-local-refiner proxy result, not as a pure FFNO-paper-stack result.
> Current code uses `SpatialLifter -> SharedFactorizedFfnoBottleneck -> 1x1`
> output projection and rejects `cnn_blocks`; a pure-BRDT-FFNO comparison
> requires a fresh rerun.

## 1. Identity And Scope

- Initiative: `NEURIPS-HYBRID-RESNET-2026`.
- Lane: Born-Rytov diffraction tomography (BRDT) candidate study.
- Bundle role: append-only architecture row extension that adds exactly
  one new factorized Fourier operator (FFNO) row to the completed
  four-row BRDT preflight under the same dataset, operator, input
  contract, split, fixed-sample IDs, normalization, and supervised+Born
  training recipe. The completed four-row preflight is preserved
  unchanged and serves as the read-only baseline.
- Append-only extension root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/`
- Read-only baseline (supervised image L1 + Born consistency, four rows):
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/execution_plan.md`
- Governing design:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`

The baseline four-row bundle, its `metrics.json`, its row directories,
and the manuscript-local assets `tables/brdt_decision_support_metrics.*`
and `figures/brdt_decision_support_recon.png` are NOT rerun or
rewritten by this item.

## 2. Backlog Question

How does a factorized Fourier operator (FFNO) — a constant-resolution
spectral architecture distinct from the vanilla `neuralop.models.FNO`
row — perform under the locked BRDT decision-support contract relative
to the U-Net, FNO vanilla, and Hybrid ResNet baselines?

## 3. Answer (Decision-Support Read)

The numbers below are echoed verbatim from the baseline four-row
preflight bundle (`metrics.json` under
`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-four-row-preflight/`)
plus the FFNO row written by this extension. The plan binds the FFNO
extension to those baseline JSON files as they exist on disk and treats
them as authoritative and immutable; row contents are echoed without
re-interpretation against any external summary authority.

- **The legacy FFNO-local-refiner proxy is competitive with Hybrid ResNet at roughly one-quarter the
  parameter count.** At `parameter_count=36,674` (vs Hybrid ResNet's
  `142,018`), the historical row achieves nearly identical image-space and
  measurement-space metrics:

  | Metric | FFNO | Hybrid ResNet | FNO vanilla | U-Net | Model-based Born inverse |
  |---|---|---|---|---|---|
  | `image_mae_phys` | 0.000659 | 0.000650 | 0.001947 | 0.001827 | 0.001563 |
  | `image_rmse_phys` | 0.001940 | 0.001809 | 0.005624 | 0.004899 | 0.002153 |
  | `image_relative_l2_phys` | 0.3421 | 0.3190 | 0.9916 | 0.8638 | 0.3796 |
  | `meas_relative_l2` | 0.1859 | 0.1992 | 0.9808 | 0.6875 | 0.1077 |
  | `psnr_phys` | 29.13 | 29.74 | 19.89 | 21.09 | 28.23 |
  | `ssim_phys` | 0.9420 | 0.9471 | 0.6779 | 0.6129 | 0.9201 |
  | `parameter_count` | 36,674 | 142,018 | 44,465 | 18,465 | 0 |
  | `wall_time_train_s` | 109.97 | 79.25 | 85.11 | 69.00 | 0.0 |

- **The legacy proxy substantially outperforms FNO vanilla.** Despite using fewer
  parameters than `fno_vanilla` (`36,674` vs `44,465`), FFNO improves
  every blocking metric by a large margin: image-space relative L2
  drops from 0.99 to 0.34 and measurement-space relative L2 drops from
  0.98 to 0.19. The factorized Fourier formulation with shared spectral
  weights plus the then-present local-convolution refiner is materially
  stronger than the vanilla `neuralop` FNO at comparable capacity, but
  this does not establish the pure FFNO-paper-stack result.

- **FFNO does NOT displace Hybrid ResNet on this capped budget.**
  Hybrid ResNet retains a small but consistent edge on image-space
  metrics (image-space relative L2 0.319 vs 0.342). On measurement-
  space relative L2 FFNO is marginally better (0.186 vs 0.199). On
  image-space PSNR / SSIM Hybrid ResNet leads by ~0.6 dB / ~0.005.
  The bundle remains decision-support evidence and does not authorize
  architecture promotion. Hybrid ResNet stays the BRDT candidate-lane
  reference.

- **Model-based Born comparator status.** The baseline `metrics.json`
  records the classical row as `row_status="completed"` with
  `paper_label="Model-based Born inverse"` (an Adam direct-`q` solver
  under `model_based_born_inverse_v1`). The combined view echoes that
  row verbatim from the baseline. Note that this label / status
  diverges from the narrative recorded in
  `brdt_preflight_summary.md` Section 6 (which still describes the
  classical row as `blocked`). Reconciling the summary with the
  baseline JSON files is out of scope for this append-only extension
  item; the FFNO row makes no independent claim about the
  model-based comparator beyond echoing the baseline's recorded
  numbers.

## 4. Contract Inheritance From The Baseline

Every fairness-relevant field is inherited byte-for-byte from the
baseline bundle. The extension runner refuses to launch if any of the
following diverge:

| Field | Baseline value (also FFNO) |
|---|---|
| Dataset id | `brdt128_decision_support_preflight` |
| Split counts | `train=2048`, `val=256`, `test=256` |
| Operator geometry | `N=D=128`, `A=64`, `wavelength_px=8.0`, `medium_ri=1.333`, Born mode, `unitary_fft` |
| Operator pointer | the locked operator-validation artifact path |
| Input contract | `born_init_image`, `in_channels=1` |
| Fixed sample IDs | `[145, 83, 255, 126]` (seed 17) |
| Training contract | epochs `20`, batch `16`, Adam, lr `2e-4`, seed `42` |
| Loss weights | `image=1.0`, `physics=0.1`, `relative_physics=0.1`, `tv=1e-5`, `positivity=1e-4` |
| Claim boundary | extension only: `decision_support_append_only` |

The lineage block in the extension `preflight_manifest.json` records
both the absolute baseline-root path and the SHA-256 fingerprint of
those baseline contract fields, so a downstream reader can verify the
extension's contract inheritance without re-reading the four-row
bundle. The baseline four-row bundle's `preflight_manifest.json`,
`metrics.json`, and `metrics.csv` are byte-identical before and after
this run.

## 5. FFNO Architecture (Task-Local, Not CDI Registry)

The historical row used a task-local FFNO body wrapper that lived entirely
under `scripts/studies/born_rytov_dt/`:

- Input: `(B, 1, 128, 128)` `born_init_image` representation.
- Lifter: `ptycho_torch.generators.fno.SpatialLifter`
  (`in_channels=1 -> hidden_channels=16`).
- Body: `ptycho_torch.generators.ffno_bottleneck.SharedFactorizedFfnoBottleneck`
  with `n_blocks=4`, `modes=8`, `share_spectral_weights=False`,
  `mlp_ratio=2.0`, `norm="instance"`.
- Refinement: two residual local-conv blocks (`Conv2d 3x3 -> GELU ->
  Conv2d 3x3`) in the historical artifact-producing code only. Current
  BRDT FFNO code removes this refinement stage and rejects `cnn_blocks`.
- Output head: `Conv2d hidden_channels -> 1` 1x1 projection in
  normalized-`q` space; the lightning module unnormalizes before the
  Born consistency loss touches the operator (the same
  `dataset_contract.reject_normalized_q_to_operator` guard used by every
  other BRDT neural row).
- Output: `(B, 1, 128, 128)` real-channel `q_pred`.
- Historical artifact trainable parameters: `36,674`; current no-refiner
  BRDT FFNO adapter trainable parameters: `27,394`.

The wrapper is **not** registered in
`ptycho_torch.generators.registry`. BRDT's row contract is task-local
and intentionally never touches the CDI/PtychoPINN generator surface.
The architecture identity is `architecture_id="ffno"` and is distinct
from `architecture_id="fno_vanilla"`, which uses
`neuralop.models.FNO` and is in the baseline bundle. FFNO is **never**
silently aliased to `fno_vanilla`.

## 6. Append-Only Five-Row Combined View

Three combined-bundle artifacts are emitted under the extension root.
They reference the baseline by lineage rather than copying its rows
forward, so the baseline's row identity (`paper_label` / `architecture`
/ `row_status`) is preserved exactly as it was written by the four-row
preflight:

- `combined_metrics.json` — five-row metrics view with a leading
  `source` column (`baseline_lineage` for the original four rows,
  `extension` for FFNO).
- `combined_metrics.csv` — flat CSV mirror of
  `combined_metrics.json`.
- `combined_manifest.json` — top-level lineage manifest naming the
  baseline backlog item, the extension backlog item, the five-row
  metric pointers, and the baseline contract fingerprint that the
  FFNO row inherited.

Row order in the combined view:
`classical_born_backprop -> unet -> fno_vanilla -> hybrid_resnet -> ffno`.

## 7. Recommendation Delta

The earlier `defer_after_preflight` recommendation in
`brdt_preflight_summary.md` is unchanged. The FFNO row is decision
support that:

- **Strengthens** the candidate-lane evidence for spectral
  architectures by showing that a more carefully-formulated factorized
  Fourier operator produces dramatically better numbers than the
  vanilla `neuralop` FNO at comparable capacity, while remaining
  competitive with Hybrid ResNet at much lower parameter count.
- **Does not** authorize BRDT promotion into manuscript tables,
  figures, or `/home/ollie/Documents/neurips/`. The classical row is
  still blocked on the ODTbrain-validation contract, the bundle is
  still a capped 2048/256/256 decision-support tier rather than a full
  training benchmark, and the required NeurIPS pillars remain CDI
  `lines128` and PDEBench CNS.

If a future campaign elects to revisit BRDT promotion, the same
preconditions from `brdt_preflight_summary.md` apply: restore the
classical Born-backprop reference, lift the bundle from
`decision_support` to a full-training contract under a fresh written
plan, and author a separate roadmap or evidence-amendment plan that
names exact rows, budgets, fairness contract, artifacts, and claim
boundaries.

## 8. Artifact Inventory

Extension root
(`.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-brdt-ffno-row-extension/`):

- `preflight_manifest.json` — extension-only view with
  `backlog_item="2026-05-04-brdt-ffno-row-extension"`,
  `claim_boundary="decision_support_append_only"`,
  `baseline_lineage` block (baseline root + manifest/metrics paths +
  baseline contract fingerprint), and a single-row roster naming FFNO.
- `metrics.json`, `metrics.csv`, `metric_schema.json` — FFNO-only
  metrics under the locked schema (`brdt_preflight_metrics_v1`) with
  the `decision_support_append_only` claim boundary.
- `visual_manifest.json`, `visuals/`, `figures/source_arrays/` — fixed-
  sample `q` compare/error and sinogram-residual figures for FFNO and
  the saved `.npy` source arrays needed to regenerate them.
- `rows/ffno/` — invocation provenance (`invocation.json`,
  `invocation.sh`), `row_summary.json`, `model_state.pt`, and the
  FFNO row's `output_dynamic_range` collapse diagnostic.
- `combined_metrics.json`, `combined_metrics.csv`,
  `combined_manifest.json` — read-only-lineage five-row combined view
  bridging the baseline four rows and the FFNO extension.
- `logs/run_ffno_extension.log` — captured stdout/stderr from the live
  CUDA run (`wall_time_train_s≈110`, `wall_time_eval_s≈1.0` on RTX
  3090).
- `invocation.json`, `invocation.sh` — top-level run provenance.

Source-of-truth code:

- `scripts/studies/born_rytov_dt/run_ffno_extension.py` — authoritative
  entrypoint; validates the baseline bundle, asserts contract
  inheritance, runs the FFNO row only, writes the FFNO-only manifest
  and metrics, and emits the combined bundle.
- `scripts/studies/born_rytov_dt/extension_bundle.py` — baseline
  validator, contract-inheritance asserts, baseline contract
  fingerprint, and combined-bundle assembly. Intentionally separate
  from `comparison.py`, which remains the physics-only objective
  ablation comparator.
- `scripts/studies/born_rytov_dt/models.py` — task-local `_BRDTFFNO`
  wrapper plus FFNO branch in `BRDTModelAdapter`. The wrapper raises
  `AdapterBuildError` if the optional FFNO components are unavailable.
- `scripts/studies/born_rytov_dt/run_config.py` — adds `ffno` to
  `SUPPORTED_ARCHITECTURES` / `SUPPORTED_ROW_IDS` and `DEFAULT_ARCH_KWARGS`
  with the explicit FFNO defaults.

Tests:

- `tests/studies/test_born_rytov_dt_adapters.py` — FFNO adapter forward
  shape, distinct architecture identity vs `fno_vanilla`, and row
  schema acceptance for `ffno`.
- `tests/studies/test_born_rytov_dt_preflight.py` — baseline bundle
  validator (rejects missing files / wrong backlog item, accepts
  whatever row roster/status the baseline records — the FFNO extension
  is plan-bound to consume the baseline JSON files as they exist),
  contract inheritance asserts (rejects dataset/claim-boundary drift),
  combined-bundle five-row assembly that echoes baseline rows verbatim,
  the runner's dry-run manifest write, the runner's refusal to start
  when the baseline root is missing, and the live-path
  `combined_metrics.json` assembly.
