# PDEBench 2D Compressible Navier-Stokes Summary

## Status

- Initiative: `NEURIPS-HYBRID-RESNET-2026`
- Scope: Roadmap Phase 2 PDEBench `128x128` image-suite third member
- Task ID: `2d_cfd_cns`
- Status: official HDF5 verified; real-data schema, capped readiness smoke, and reusable CNS physics-regularization framework complete; full benchmark still pending
- Date: 2026-04-22
- Suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`
- CNS design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_design.md`
- CNS physics-regularization design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_physics_regularization_design.md`
- Current preflight summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md`

This summary records implementation/readiness state only. It does not create manuscript artifacts, benchmark-performance claims, model rankings, or `/home/ollie/Documents/neurips/` outputs.

## Implemented Contract

- Active suite tasks are now `swe`, `darcy`, and `2d_cfd_cns`.
- The selected CNS file is `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`, DaRUS datafile `164690`, expected bytes `55,050,245,208`, expected MD5 `21969082d0e9524bcc4708e216148e60`.
- The loader expects separate HDF5 field datasets in stable order: `density`, `Vx`, `Vy`, `pressure`.
- The primary supervised unit is one trajectory/time window with `history_len=2`: input `(8,128,128)` and target `(4,128,128)`.
- Splits are trajectory-level before history-window expansion, with seed `20260420`.
- Normalization is train-only per field and is reused across history slots and target fields.
- Metrics are computed on denormalized target-state predictions and include aggregate plus per-field RMSE, nRMSE, relative L2, and Fourier-space RMSE bands `fRMSE_low`, `fRMSE_mid`, and `fRMSE_high`.
- The local CNS fRMSE convention is fixed as `torch.fft.fft2(..., norm="ortho")` with fftshifted radial-frequency thirds: low `<= 1/3`, mid `(1/3, 2/3]`, and high `> 2/3` of the maximum radial frequency.
- The runner uses the supervised real-channel Hybrid ResNet/FNO/U-Net adapter path, not PtychoPINN physics layers or ptychographic `C` semantics.
- The canonical CNS Hybrid row is now `hybrid_resnet_cns`. It keeps the same `32`-channel / `12`-mode / `4`-block / `2`-downsample / `6`-ResNet shell as `hybrid_resnet_base`, promotes encoder-decoder skip fusion with `hybrid_skip_connections=on` and `hybrid_skip_style=add`, and now defaults to `hybrid_upsampler=pixelshuffle` based on the post-skip-add upsampler compare.
- The current CNS training-loss contract is `mse`, matching the official PDEBench FNO/U-Net forward baseline code for the compressible CFD benchmarks; earlier local MAE pilot runs remain useful only as superseded readiness evidence.
- A reusable PDEBench image-suite physics-loss framework now exists with a `2d_cfd_cns` backend. It is disabled by default, task-fails closed for unsupported tasks, and currently implements three denormalized terms: positivity on `density/pressure`, continuity residual, and global mass consistency.
- CNS HDF5 metadata now records `dx`, `dy`, `dt`, `eta`, `zeta`, and `boundary_condition="periodic"` so physics losses can be computed in physical units instead of normalization space.

## Verified Data Gate

The selected CNS file is now staged and verified:

- Final path: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`
- File size: `55,050,245,208` bytes
- MD5: `21969082d0e9524bcc4708e216148e60`
- Download log directory: `/home/ollie/Documents/pdebench-data/2d_cfd_cns/logs`

Real-data suite preflight is now fully ready:

- Ready tasks: `3 / 3`
- CNS schema: dynamic multi-field `density`, `Vx`, `Vy`, `pressure`
- Field shape: `[10000, 21, 128, 128]`
- Axis order: `NTHW`
- Available one-step windows with `history_len=2`: `190000`

Suite-level benchmark claims are still blocked, but now for benchmark-compute scope rather than missing-data scope.

## Readiness Smoke

Real-data capped readiness run:

- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T040020Z`
- Evidence scope: `smoke_feasibility_only`
- Metric interpretation: `sanity_only_not_benchmark_performance`
- Performance assessment complete: `false`

Epoch-1 training losses from the readiness run:

- `hybrid_resnet_base`: `0.9966974258`
- `fno_base`: `0.9634801149`
- `unet_tiny_smoke`: `0.9651259184`

Readiness evaluation outputs confirm the denormalized CNS metric contract is wired through the run summary:

- `hybrid_resnet_base`: `err_nRMSE=1.3146735429763794`, `fRMSE_high=1.886562466621399`
- `fno_base`: `err_nRMSE=1.2781659364700317`, `fRMSE_high=0.029776178300380707`
- `unet_tiny_smoke`: `err_nRMSE=1.2555538415908813`, `fRMSE_high=0.0635882243514061`

These values are readiness-only sanity artifacts. They are not benchmark-performance evidence and cannot support model ranking or paper competitiveness claims.

## Larger Capped Comparison

To move beyond the toy `2/1/1` trajectory smoke, a second capped real-data comparison was run with materially larger sample counts:

- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T041000Z`
- Mode: `readiness`
- Profiles: `hybrid_resnet_base`, `fno_base`, `unet_strong`
- Batch size: `4`
- Train/val/test trajectories: `64 / 8 / 8`
- `max_windows_per_trajectory`: `4`
- Train/val/test windows: `256 / 32 / 32`

Epoch-1 training losses from this larger capped run:

- `hybrid_resnet_base`: `0.7151282818522304`
- `fno_base`: `0.78266646200791`
- `unet_strong`: `0.7595781004056334`

Readiness comparison summary metrics:

- `hybrid_resnet_base`: `err_nRMSE=0.7567286491394043`, `fRMSE_high=1.6373682022094727`
- `fno_base`: `err_nRMSE=0.7614443898200989`, `fRMSE_high=0.11482542008161545`
- `unet_strong`: `err_nRMSE=0.7786170840263367`, `fRMSE_high=1.7014796733856201`

This run is still capped and remains readiness/decision-support evidence only. It is useful because it removes the trivial `2`-window training regime, but it is still not a benchmark-performance row.

## Corrected 10-Epoch MSE Comparison

After confirming that official PDEBench FNO and U-Net forward baselines train with `nn.MSELoss(reduction="mean")`, the local CNS runner was corrected from `mae` to `mse` and the larger capped comparison was rerun under the literature-aligned contract:

- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse`
- Mode: `readiness`
- Profiles: `hybrid_resnet_base`, `fno_base`, `unet_strong`
- Training loss: `mse`
- Batch size: `4`
- Epochs: `10`
- Train/val/test trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory`: `8`
- Train/val/test windows: `4096 / 512 / 512`

Per-profile MSE training losses:

- `hybrid_resnet_base`: `[0.5158408025, 0.2933245828, 0.1798223130, 0.1289558155, 0.1037858993, 0.0923368968, 0.0745479463, 0.0666449550, 0.0590354418, 0.0514632439]`
- `fno_base`: `[0.3939300724, 0.1417130202, 0.1096219451, 0.0918289570, 0.0804821433, 0.0725931004, 0.0666914391, 0.0618780411, 0.0577162401, 0.0540558439]`
- `unet_strong`: `[0.6829816124, 0.5247917346, 0.4211111818, 0.3430424545, 0.2905547233, 0.2513744799, 0.2221647620, 0.1975522787, 0.1784509105, 0.1647206898]`

Readiness comparison summary metrics from the corrected run:

- `hybrid_resnet_base`: `err_nRMSE=0.1945149451494217`, `fRMSE_high=1.0892750024795532`
- `fno_base`: `err_nRMSE=0.10634330660104752`, `fRMSE_high=0.9280493259429932`
- `unet_strong`: `err_nRMSE=0.6222500205039978`, `fRMSE_high=3.6293647289276123`

This run supersedes the earlier CNS MAE pilot for protocol alignment. It is still capped and therefore still readiness/decision-support evidence only, not a benchmark-performance row.

## Physics Regularization Framework

The approved design is recorded in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_2d_cfd_cns_physics_regularization_design.md`

Implemented framework contract:

- Shared entry point: the CNS runner accepts optional physics-regularization config and applies it in the shared training loop, so Hybrid, FNO, and U-Net can all use the same task-local loss path.
- Default-off behavior is preserved. Existing runs without explicit physics settings remain plain supervised `mse` runs.
- v1 CNS terms:
  - positivity: penalize negative denormalized `density` and `pressure`
  - continuity: penalize local mass-conservation residual
  - global mass: penalize denormalized total-mass drift from the latest history state to the predicted next state
- Unsupported tasks fail closed through the reusable builder instead of silently accepting no-op physics settings.

Bounded enabled smoke:

- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T210000Z-physics-smoke`
- Profile: `hybrid_resnet_base`
- Mode: `readiness`
- Train/val/test trajectories: `8 / 2 / 2`
- `max_windows_per_trajectory`: `2`
- Epochs: `1`
- Physics weights: `positivity=1.0`, `continuity=0.5`, `global_mass=0.25`

Observed enabled-smoke outcome:

- run completed with exit code `0`
- `physics_regularization_enabled=true`
- `physics_loss_terms=["positivity","continuity","global_mass"]`
- final epoch weighted physics total: `6641.8469`
- final epoch raw terms:
  - continuity: `13280.7893`
  - global mass: `4.0073`
  - positivity: `0.4503`
- readiness metrics:
  - `err_nRMSE=0.7428568602`
  - `fRMSE_high=4.6143236160`

Interpretation boundary:

- This smoke proves the framework is wired and artifacted, not that the current placeholder weights are good.
- The continuity term dominates at these first-pass weights, so any serious physics-regularized comparison needs weight tuning before benchmark interpretation.

## Canonical Hybrid Promotion

The capped skip-study promoted encoder-decoder skip fusion from a manual study variant to the canonical CNS Hybrid row:

- Run root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-skip-study/cns-skipadd-vs-base-10ep`
- Same slice for both rows: `512 / 64 / 64` trajectories, `8` windows per trajectory, `10` epochs, batch size `4`
- Compared rows:
  - `hybrid_resnet_base`
  - `hybrid_resnet_cns` effective architecture (`skip_add`)

Observed outcome on the capped readiness slice:

- `hybrid_resnet_base`: final train loss `0.0596839676`, `err_nRMSE=0.2244247347`, `fRMSE_high=1.2431740761`
- `hybrid_resnet_cns` / skip-add shell: final train loss `0.0316615416`, `err_nRMSE=0.1045547351`, `fRMSE_high=0.6970583200`

Interpretation:

- skip-add improved final training loss by about `47%`
- skip-add improved aggregate `err_nRMSE` by about `53%`
- skip-add improved `fRMSE_high` by about `44%`
- parameter count changed only slightly: `7,788,293 -> 7,793,509`

Decision:

- CNS default profile sets now use `hybrid_resnet_cns` rather than `hybrid_resnet_base`
- `hybrid_resnet_base` remains available for cross-task generic comparisons and backward-compatible study references
- the earlier manual study profile `hybrid_resnet_skip_add` remains as a discoverable study profile, but the canonical CNS row is `hybrid_resnet_cns`

## Post-Skip-Add Upsampler Compare

The upsampler artifact study was rerun after skip-add had already been promoted
into the canonical CNS shell:

- results summary:
  `docs/plans/2026-04-21-hybrid-upsampler-artifact-study-results.md`
- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep-20260421T221400Z`
- same slice for all rows: `512 / 64 / 64` trajectories, `8` windows per
  trajectory, `10` epochs, batch size `4`
- compared rows:
  - `hybrid_resnet_cns_transpose`
  - `hybrid_resnet_cns_interp_bilinear_conv`
  - `hybrid_resnet_cns_pixelshuffle`

Observed metrics on that capped slice:

- `hybrid_resnet_cns_transpose`: final train loss `0.0338358228`,
  `err_nRMSE=0.1304672956`, `fRMSE_high=0.7634432316`
- `hybrid_resnet_cns_interp_bilinear_conv`: final train loss `0.0350025710`,
  `err_nRMSE=0.0990807489`, `fRMSE_high=1.1363334656`
- `hybrid_resnet_cns_pixelshuffle`: final train loss `0.0325912039`,
  `err_nRMSE=0.0963200703`, `fRMSE_high=0.8639594913`

Interpretation:

- the old transpose decoder remained best on `fRMSE_high`, but retained more
  visible checkerboard-like texture
- bilinear improved aggregate error but had the worst high-frequency penalty and
  showed directional stripe-like artifacts instead of clean reconstructions
- pixelshuffle had the best aggregate error and was a better compromise than
  bilinear on high-frequency behavior

Decision:

- `hybrid_resnet_cns` now defaults to `pixelshuffle`
- the old transpose decoder remains available only as the explicit manual
  profile `hybrid_resnet_cns_transpose`
- bilinear remains manual-only

## Same-Shell Bottleneck Compare

The FFNO-close bottleneck tranche added a same-shell three-row comparison under
the canonical CNS skip-add shell:

- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T203200Z-ffno-cns-10ep`
- same task/slice: `2d_cfd_cns`, `512 / 64 / 64` trajectories, `8` windows per
  trajectory, `10` epochs, batch size `4`
- same shell for all rows:
  - `hybrid_downsample_steps=2`
  - `hybrid_skip_connections=True`
  - `hybrid_skip_style=add`
- compared bottlenecks:
  - `hybrid_resnet_cns`: local ResNet bottleneck
  - `spectral_resnet_bottleneck_base`: spectral-bypass ResNet bottleneck
  - `ffno_bottleneck_base`: FFNO-close spectral-plus-feedforward bottleneck

Observed metrics on that capped slice:

- `hybrid_resnet_cns`: `err_nRMSE=0.0944002941`, `err_RMSE=2.2813653946`,
  `fRMSE_high=0.8000375628`
- `spectral_resnet_bottleneck_base`: `err_nRMSE=0.0869938582`,
  `err_RMSE=2.1023747921`, `fRMSE_high=0.6955373287`
- `ffno_bottleneck_base`: `err_nRMSE=0.1110704020`,
  `err_RMSE=2.6842310429`, `fRMSE_high=0.7276504636`

Interpretation:

- the current spectral bottleneck row is best of the three on aggregate capped
  CNS error
- the canonical local `hybrid_resnet_cns` row is second
- the FFNO-close row did not improve over the local bottleneck on this first
  capped same-shell run

This evidence is still capped-readiness evidence only. It is enough to guide
local bottleneck work, but it is not benchmark-complete CNS ranking evidence.

## 40-Epoch Spectral Follow-Up

The shared-spectral bottleneck row was later extended to the same capped
40-epoch CNS budget previously used for the local Hybrid/FNO/U-Net comparison:

- run root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-bottleneck-cns-compare/readiness-cap-20260421T221008Z-spectral40ep`
- same slice:
  `2d_cfd_cns`, `512 / 64 / 64` trajectories, `8` windows per trajectory,
  batch size `4`, `40` epochs

Observed 40-epoch spectral metrics:

- `err_RMSE=1.4877649546`
- `err_nRMSE=0.0615620054`
- `relative_l2=0.0615620054`
- `fRMSE_low=3.4756414890`
- `fRMSE_mid=0.2800448835`
- `fRMSE_high=0.4349334538`

Relative to the earlier 40-epoch capped comparison:

- spectral beat `hybrid_resnet_base`
- spectral beat `fno_base`
- spectral remained far ahead of `unet_strong`

So the current local decision-support ranking for capped 40-epoch CNS is:

1. `spectral_resnet_bottleneck_base`
2. `fno_base`
3. `hybrid_resnet_base`
4. `unet_strong`

This is still capped-readiness evidence only, not benchmark-complete CNS
ranking evidence.

## Spectral Modes-32 Compare

The dedicated higher-mode follow-up is summarized in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes32_compare_summary.md`

That backlog item kept the capped local CNS contract fixed and changed only:

- `fno_modes: 12 -> 32`
- `spectral_bottleneck_modes: 12 -> 32`

Recovered and fresh run roots:

- reused fresh `10`-epoch row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-10ep-20260428T010825Z`
- authoritative fresh `40`-epoch row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/cns-spectral-modes32-40ep-20260428T014353Z`

Anchored compare sidecars:

- `10ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_10ep_against_existing.json`
- `40ep`:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-spectral-modes32-compare/compare_40ep_against_existing.json`

Observed outcome:

- at `10` epochs, modes-32 improved every tracked eval metric over the shared
  `12/12` spectral row and stayed ahead of the reused `fno_base` and
  `unet_strong` anchors
- at `40` epochs, modes-32 no longer beat the shared `12/12` spectral row on
  aggregate denormalized error:
  - `err_nRMSE 0.0645505` vs `0.0615620`
  - `err_RMSE 1.55999` vs `1.48776`
  - `fRMSE_low 3.6761` vs `3.4756`
- the same `40`-epoch run did improve the higher-frequency side of the
  contract:
  - `fRMSE_mid 0.2425` vs `0.2800`
  - `fRMSE_high 0.3472` vs `0.4349`

Interpretation boundary:

- the higher-mode row remains manual-only
- it is useful as capped high-frequency decision-support evidence
- it does not replace the shared `12/12` spectral row as the aggregate local
  capped `40`-epoch spectral reference

## Hybrid-Spectral Architecture Ablation

The dedicated same-shell sharing-plus-depth follow-up is summarized in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_arch_ablation_summary.md`

That backlog item kept dataset, split family, `history_len=2`, `mse`, batch
size, metrics, and the canonical CNS shell fixed, then reran only the approved
hybrid-spectral internal axes:

- weight sharing:
  - shared `spectral_resnet_bottleneck_base`
  - non-shared `spectral_resnet_bottleneck_noshare`
- shared spectral bottleneck depth:
  - `6` blocks: `spectral_resnet_bottleneck_base`
  - `8` blocks: `spectral_resnet_bottleneck_shared_blocks8`
  - `10` blocks: `spectral_resnet_bottleneck_shared_blocks10`

Fresh run roots:

- sharing `10` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-10ep-20260428T032825Z`
- sharing `40` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-sharing-40ep-20260428T035154Z`
- shared-depth `40` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z`
- finalist confirmation `1024 / 128 / 128`, `40` epochs:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`

Anchored compare sidecars:

- `10`-epoch sharing:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_sharing_10ep_against_existing.json`
- `40`-epoch sharing:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_sharing_40ep_against_existing.json`
- `40`-epoch shared depth:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_depth_40ep_against_existing.json`
- `1024 / 128 / 128` finalist confirmation:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/compare_finalists_1024cap_40ep_within_run.json`

During implementation-review repair, the sharing sidecars were regenerated from
the frozen reference manifests so both fresh spectral rows now appear as fresh
rows rather than substituting the shared base row into the required-reference
set. That repair did not change the interpretation below.

Observed outcome:

- at `10` epochs, disabling weight sharing improved aggregate denormalized error
  versus the shared base row, but it did not improve `fRMSE_high`
- at `40` epochs, the shared base row beat the non-shared row on every tracked
  eval metric, so the earlier non-shared gain did not hold
- on the separate shared-depth `40`-epoch tranche, the `10`-block shared row
  beat both `6` and `8` blocks on every tracked eval metric
- on the larger-cap `1024 / 128 / 128` confirmation slice, the aggregate lead
  returned to the shared base row:
  - `spectral_resnet_bottleneck_base`:
    `err_nRMSE=0.0435010`, `fRMSE_high=0.2983`
  - `spectral_resnet_bottleneck_shared_blocks10`:
    `err_nRMSE=0.0445733`, `fRMSE_high=0.2940`

Interpretation boundary:

- deeper shared spectral bottlenecks can help on the smaller capped lane
- the larger-cap confirmation does not support promoting the deeper shared row
  over the shared base row as the aggregate local capped reference
- this entire architecture lane remains capped decision-support evidence only,
  not a benchmark-complete CNS conclusion

## Markov History-1 Compare

The controlled lower-context follow-up is summarized in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_markov_history1_compare_summary.md`

That backlog item fixed dataset, trajectories, `max_windows_per_trajectory`,
batch size, `mse`, metric family, and epoch budgets, then changed only the
temporal-context contract from `history_len=2` to `history_len=1`:

- frozen history-2 manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2_reference_runs.json`
- backfilled `40`-epoch `hybrid_resnet_cns` anchor:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history2-hybrid-cns-pilot-40ep-20260423T223143Z`
- fresh history-1 `10`-epoch run:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-10ep-20260423T224907Z`
- fresh history-1 `40`-epoch run:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-markov-history1-compare/history1-pilot-40ep-20260423T230352Z`

Observed result on the capped four-row family:

- `spectral_resnet_bottleneck_base` did not improve under `history_len=1`
  at either budget:
  - `10ep`: `err_nRMSE 0.0869938582 -> 0.1139530987`
  - `40ep`: `err_nRMSE 0.0615620054 -> 0.0998256728`
- the four-row ranking stayed the same at both budgets:
  `spectral_resnet_bottleneck_base > hybrid_resnet_cns > fno_base > unet_strong`
- `hybrid_resnet_cns` and `fno_base` also worsened on aggregate denormalized
  error at both budgets
- `unet_strong` improved slightly at `10ep` and more clearly at `40ep`, but
  remained the worst row overall

Cross-history compare sidecars were written for both budgets:

- `compare_10ep_against_history2.json`
- `compare_40ep_against_history2.json`

Both sidecars blocked merged cross-run galleries because the saved sample-0
targets did not align exactly across the separate run roots. That blocker is
non-fatal and uses the explicit target-alignment check
`np.allclose(..., atol=1e-6, rtol=1e-6)`.

Interpretation boundary:

- this is capped pilot decision-support evidence only
- it does not replace the benchmark-complete gate
- it should be read against the frozen history-2 family above, not against the
  forbidden older `hybrid_resnet_base` proxy for the canonical CNS shell

## Longer History-3 Plus Compare

The controlled longer-context follow-up is summarized in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_history_len3plus_compare_summary.md`

That backlog item kept dataset, trajectories, `max_windows_per_trajectory`,
batch size, `mse`, metric family, and epoch budgets fixed, then changed only
the temporal-context contract from `history_len=2` to `history_len=3`:

- frozen history-2 manifest:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history2_reference_runs.json`
- fresh history-3 `10`-epoch run:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-10ep-20260429T071905Z`
- fresh history-3 `40`-epoch run:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history3-pilot-40ep-20260429T073705Z`

The raw eligible-window contract tightened exactly as expected:

- reference `history_len=2`: `19` windows per trajectory, `190000` total raw
  windows
- fresh `history_len=3`: `18` windows per trajectory, `180000` total raw
  windows
- optional `history_len=4` inspect-only proof:
  `17` windows per trajectory, `170000` total raw windows

The emitted capped split counts stayed fixed at `4096 / 512 / 512` because the
same `max_windows_per_trajectory=8` cap was preserved across all contexts.

Observed result on the capped four-row family:

- at `10` epochs, `history_len=3` did not help the stronger spectral or
  canonical Hybrid rows on aggregate error:
  - `spectral_resnet_bottleneck_base`:
    `err_nRMSE 0.0869938582 -> 0.1407901347`
  - `hybrid_resnet_cns`:
    `err_nRMSE 0.0944002941 -> 0.1119609326`
  - `fno_base` improved and moved into first place on the capped `10`-epoch
    slice:
    `err_nRMSE 0.1063433066 -> 0.0953909084`
- at `40` epochs, the longer-context contract materially improved the three
  stronger rows on aggregate denormalized metrics and `fRMSE_high`:
  - `spectral_resnet_bottleneck_base`:
    `err_nRMSE 0.0615620054 -> 0.0455205254`,
    `fRMSE_high 0.4349334538 -> 0.3467437923`
  - `hybrid_resnet_cns`:
    `err_nRMSE 0.0644183308 -> 0.0538428985`,
    `fRMSE_high 0.3683068156 -> 0.3200356364`
  - `fno_base`:
    `err_nRMSE 0.0740992129 -> 0.0567254014`,
    `fRMSE_high 0.6717720628 -> 0.6104770303`
  - `unet_strong` stayed last and was effectively flat-to-worse on aggregate

The optional `history_len=4` branch stayed closed:

- gate record:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-history-len3plus-compare/history4_gate_decision.json`
- reason:
  the `40`-epoch spectral row satisfied the gate rule, but the `10`-epoch and
  `40`-epoch spectral signals disagreed and no written scientific reason was
  added to override the default-closed rule
- consequence:
  no `history_len=4` pilot or `compare_*_history4_against_history2.*` payloads
  were authorized

Interpretation boundary:

- this is capped pilot decision-support evidence only
- it does not replace the benchmark-complete gate
- it suggests `history_len=3` can help the stronger rows once trained longer,
  but the mixed `10`/`40`-epoch signal is not stable enough to justify an
  automatic move to `history_len=4`

## Official Author FFNO Equal-Footing Follow-Up

The official authored FFNO baseline is tracked separately from the local
FFNO-close bottleneck proxy in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_author_ffno_equal_footing_summary.md`

That backlog item keeps the same local capped CNS contract fixed:

- task: `2d_cfd_cns`
- `history_len=2`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- batch size: `4`

Completed author compares:

- fresh `10`-epoch author row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-10ep-20260422T232119Z`
- merged compare:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_10ep_against_existing.json`

The merged `10`-epoch compare includes:

- `author_ffno_cns_base`
- `spectral_resnet_bottleneck_base`
- `fno_base`
- `unet_strong`
- optional continuity:
  `hybrid_resnet_cns`, `hybrid_resnet_base`

Observed `10`-epoch author result:

- `err_nRMSE=0.0878334790`
- `relative_l2=0.0878334790`
- `fRMSE_high=0.2596977651`

Relative to the existing local rows on this capped slice:

- the authored FFNO row is effectively tied with the shared-spectral row on
  aggregate denormalized error
- the authored FFNO row beats the earlier local `fno_base`,
  `hybrid_resnet_cns`, `hybrid_resnet_base`, and `unet_strong` rows on
  aggregate denormalized error
- the authored FFNO lane remains separate from `ffno_bottleneck_base`, which is
  still the repo-local FFNO-close proxy bottleneck experiment

- fresh `40`-epoch author row:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/author-ffno-40ep-20260422T234340Z`
- merged `40`-epoch compare:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-author-ffno-equal-footing-cns/compare_40ep_against_existing.json`

The merged `40`-epoch compare includes:

- `author_ffno_cns_base`
- `spectral_resnet_bottleneck_base`
- `fno_base`
- `unet_strong`
- optional continuity:
  `hybrid_resnet_base`

Observed `40`-epoch author result:

- `err_nRMSE=0.0281477310`
- `relative_l2=0.0281477310`
- `fRMSE_high=0.1210141182`

Relative to the existing local rows on this capped slice:

- the authored FFNO row clearly beats the shared-spectral row on aggregate
  denormalized error and on the saved-sample `fRMSE_high` view
- the authored FFNO row also beats the earlier local `fno_base`,
  `hybrid_resnet_base`, and `unet_strong` rows on aggregate denormalized error
- the merged `10`-epoch and `40`-epoch compares both rendered cross-run sample
  galleries, so no target-alignment blocker was needed for either approved
  epoch slice

This author-FFNO lane is still capped-readiness, decision-support-only evidence.
It does not change the benchmark-complete boundary for CNS.

## FFNO Local-Convolution Follow-Up

The repo-local FFNO-family extension that adds an explicit local residual branch
inside the FFNO-close bottleneck is tracked in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_convolutional_features_cns_summary.md`

That backlog item kept the same capped CNS contract fixed:

- task: `2d_cfd_cns`
- `history_len=2`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- batch size: `4`

Authoritative local-conv run roots:

- `10`-epoch:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-10ep-20260428T082501Z`
- `40`-epoch:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-localconv-40ep-20260428T090626Z`

The earlier partial root `cns-ffno-localconv-40ep-20260428T090543Z` is
documented as non-authoritative in the Task 1 inspection manifest and should
not be used as the `40`-epoch local-conv reference.

The item also had to backfill the missing same-contract `40`-epoch FFNO-close
anchor before the final compare:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/cns-ffno-close-backfill-40ep-20260428T084852Z`

Frozen cross-run compares:

- `10`-epoch:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/compare_10ep_against_existing.json`
- `40`-epoch:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-ffno-convolutional-features-cns/compare_40ep_against_existing.json`

Observed local-conv metrics:

- `10` epochs:
  `relative_l2=0.0846254751`, `fRMSE_high=0.6369161010`
- `40` epochs:
  `relative_l2=0.0557734184`, `fRMSE_high=0.3090891540`

Directional read:

- local-conv materially improved the repo-local `ffno_bottleneck_base` row on
  both approved budgets
- local-conv also beat the capped shared-spectral local anchor on aggregate
  error and `fRMSE_high` at both `10` and `40` epochs
- the official authored FFNO row remained stronger than the local-conv row on
  the `40`-epoch capped contract, so this result does not displace the authored
  lane or change the benchmark-complete boundary

Decision:

- carry `ffno_bottleneck_localconv_base` forward as the stronger repo-local
  FFNO-family follow-up profile for future bounded local studies
- do not promote it into a primary bundle from capped evidence alone

## Hybrid-Spectral To FFNO Parameter-Space Shell Probes

The bounded shell-bridge follow-up between the current shared-spectral lane and
the repo-local FFNO-family lane is tracked in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_ffno_parameter_space_summary.md`

That item kept the same capped CNS contract fixed and reused the frozen
Hybrid-spectral plus FFNO local-conv authorities. It only launched two fresh
`10`-epoch shell probes against `spectral_resnet_bottleneck_base`:

- `spectral_resnet_bottleneck_base_down1`
- `spectral_resnet_bottleneck_base_transpose`

Anchored compare artifacts:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/compare_10ep_against_existing.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-ffno-parameter-space/compare_10ep_against_existing.csv`

Observed fresh-row metrics:

- `spectral_resnet_bottleneck_base_down1`:
  `relative_l2=0.1049272269`, `fRMSE_high=0.9592211246`
- `spectral_resnet_bottleneck_base_transpose`:
  `relative_l2=0.1664283574`, `fRMSE_high=0.7184003592`
- anchor `spectral_resnet_bottleneck_base`:
  `relative_l2=0.0869938582`, `fRMSE_high=0.6955373287`

Directional read:

- the reduced-downsampling row cut parameter count and runtime sharply, but
  still lost on the declared promotion keys
- the transpose-decoder row was worse still on aggregate error and carried a
  large low-band penalty
- because neither probe was competitive or ambiguous relative to the spectral
  anchor on `relative_l2`, then `err_nRMSE`, then `fRMSE_high`, the `40`-epoch
  promotion set stayed empty

Decision:

- keep `spectral_resnet_bottleneck_base` as the aggregate local shell anchor
- keep `ffno_bottleneck_localconv_base` as the stronger repo-local FFNO-family
  alternative from the earlier local-conv follow-up
- treat this lane as a negative shell result on the capped contract, not as a
  benchmark or promotion signal

## Official GNOT Paper-Default Follow-Up

The official GNOT baseline is tracked separately from both the authored FFNO
lane and the local FFNO-close bottleneck proxy in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_gnot_cns_compare_summary.md`

That summary now covers three distinct GNOT checkpoints on the same capped CNS
contract:

- first equal-footing fairness probe:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-vs-spectral-cap512-10ep-20260422T200900Z`
- fresh paper-default smoke gate:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/gnot-paper-default-smoke-20260423T015239Z`
- same-day paper-default `40`-epoch follow-up:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-gnot-cns-compare/cns-gnot-paper-default-40ep-20260422T214016Z`

The fixed local contract stayed:

- task: `2d_cfd_cns`
- `history_len=2`
- trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- batch size: `4`

The paper-default follow-up switched only the GNOT training recipe:

- `training_loss=relative_l2`
- `optimizer=AdamW`
- `scheduler=OneCycleLR`
- `learning_rate=1e-3`
- `weight_decay=5e-5`

Observed paper-default `40`-epoch GNOT result:

- `err_nRMSE=0.1759552360`
- `relative_l2=0.1759552360`
- `fRMSE_low=10.1596441269`
- `fRMSE_high=0.0415550619`

Directional read:

- paper-default GNOT improved materially over the first fairness-probe GNOT row
  (`relative_l2 0.24565 -> 0.17596`)
- paper-default GNOT still trails the pinned spectral `40`-epoch anchor badly
  on aggregate error (`relative_l2 0.17596` vs `0.06156`)
- the remaining GNOT gap is still mainly low-frequency/global structure error,
  not high-frequency shock detail
- GNOT remains weaker than the authored `40`-epoch FFNO row on this capped
  contract

This GNOT lane is still capped-readiness, decision-support-only evidence. It
does not change the benchmark-complete boundary for CNS.

## Hybrid-Spectral 2048-Cap Scaling Follow-Up

The capped finalist scaling follow-up is tracked in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_hybrid_spectral_scaling_2048cap_summary.md`

That tranche kept the same fixed CNS contract and reused the frozen architecture-ablation finalists:

- task: `2d_cfd_cns`
- training loss: `mse`
- `history_len=2`
- `max_windows_per_trajectory=8`
- batch size: `4`
- epochs: `40`
- profiles:
  - `spectral_resnet_bottleneck_base`
  - `spectral_resnet_bottleneck_shared_blocks10`

Reference manifests:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_512cap_40ep.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/reference_runs_1024cap_40ep.json`

Fresh `2048cap` run root:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z`

Review-closeout deviation record:

- the generated inspect proof for this tranche lives at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/inspect-2048cap-20260428T232104Z/`
- the fresh `2048cap` run launch timestamp was `2026-04-28T20:20:10.547417+00:00`
  while the generated inspect-proof timestamp was
  `2026-04-28T23:20:12.056160+00:00`
- this tranche therefore closes as a documented inspect-gate sequencing
  deviation rather than as a clean inspect-before-launch execution; see the
  governing execution plan amendment and the `2048cap` summary for the exact
  rationale and boundaries

Observed `2048 / 256 / 256` metrics:

- `spectral_resnet_bottleneck_base`:
  `relative_l2=0.0421656668`, `err_nRMSE=0.0421656668`, `fRMSE_low=2.3713548183`, `fRMSE_mid=0.2230527103`, `fRMSE_high=0.3117601573`
- `spectral_resnet_bottleneck_shared_blocks10`:
  `relative_l2=0.0504393019`, `err_nRMSE=0.0504393019`, `fRMSE_low=2.8665962219`, `fRMSE_mid=0.2082921267`, `fRMSE_high=0.2969700098`

Directional read:

- the shared base row kept the aggregate lead at `2048cap`
- the deeper shared row retained only narrower `fRMSE_mid/high` advantages
- the `2048cap` extension strengthened the bounded conclusion from the earlier `1024cap` confirmation:
  - `spectral_resnet_bottleneck_base` still improves slightly on aggregate beyond `1024cap`
  - `spectral_resnet_bottleneck_shared_blocks10` regresses on aggregate beyond `1024cap`
- the post-run inspect repair does not change the model interpretation above; it
  only narrows what can be claimed about execution sequencing
- this item remains capped decision-support evidence only and does not justify promoting either spectral row into a benchmark-complete or default-profile claim

Reporting warning:

- the helper-generated scaling JSON/CSV succeeded
- the optional cross-run gallery was skipped because saved sample targets differ across the frozen reference roots, which is recorded in the scaling summary and payload as a non-fatal warning

This follow-up does not change the next gate for CNS:

- full-training same-protocol benchmark runs on the full available training split are still required for `hybrid_resnet_cns`, `fno_base`, and `unet_strong`

## Spectral Modes-24 Convergence Follow-Up

The capped convergence-oriented shared spectral follow-up is tracked in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_modes24_convergence_summary.md`

That tranche kept the same official `2d_cfd_cns` file and shell family but tightened the compare to exactly two rows on a larger capped slice:

- `spectral_resnet_bottleneck_base` with `fno_modes=12`, `spectral_bottleneck_modes=12`
- `spectral_resnet_bottleneck_modes24` with `fno_modes=24`, `spectral_bottleneck_modes=24`
- split: `1024 / 128 / 128`
- `history_len=2`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- epochs: `80`
- batch size: `16`

Observed outcome:

- the shared `12/12` base row finished ahead on every final reported eval metric at the capped `80`-epoch stop
- the generated convergence audit still marked both rows as materially improving because both late-window ratios stayed below the fixed `0.95` threshold
- the correct interpretation is therefore inconclusive capped evidence, not a clean promotion or rejection verdict for `24/24`

This follow-up does not change the current CNS gate:

- full-training same-protocol benchmark runs on the full available training split are still the next required step
- the `24/24` row remains manual-only and capped decision-support evidence

## Shared-Blocks10 1024-Cap Longer-Convergence Follow-Up

The capped longer-budget shared-blocks10 follow-up is tracked in:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_shared_blocks10_1024cap_longer_convergence_summary.md`

That tranche held the same official `2d_cfd_cns` file and fixed `1024 / 128 / 128` shell contract from the completed hybrid-spectral architecture-ablation item, but reran only one profile:

- `spectral_resnet_bottleneck_shared_blocks10`
- split: `1024 / 128 / 128`
- `history_len=2`
- `max_windows_per_trajectory=8`
- training loss: `mse`
- batch size: `4`
- reference budget: `40` epochs
- fresh budget: `80` epochs

Observed outcome:

- the fresh `80`-epoch shared-blocks10 row improved every tracked eval metric against its frozen `40`-epoch `1024cap` reference:
  - `err_nRMSE 0.0445733 -> 0.0375677`
  - `err_RMSE 1.06304 -> 0.895959`
  - `fRMSE_low 2.48274 -> 2.10180`
  - `fRMSE_mid 0.216790 -> 0.170457`
  - `fRMSE_high 0.293987 -> 0.215245`
- the fixed convergence audit still marks the row as materially improving at stop time because `late_window_ratio=0.790954 < 0.95`
- relative to the frozen `1024cap`, `40`-epoch shared-base row from the prerequisite architecture-ablation tranche, the fresh `80`-epoch shared-blocks10 row is now lower on every reported metric

Directional read:

- the old `40`-epoch shared-blocks10 `1024cap` row was materially under-converged
- that changes the earlier bounded `1024cap` interpretation that treated shared-blocks10 as only a narrow mid/high-frequency trade-off at that cap
- this remains mixed-budget capped evidence, not a same-budget architecture verdict:
  - the shared base row was not rerun at `80` epochs
  - the fresh shared-blocks10 row is still improving at the capped stop
  - the equal-budget `2048cap`, `40`-epoch follow-up still kept the shared base row ahead on aggregate while shared-blocks10 kept only narrower `fRMSE_mid/high` advantages

This follow-up does not change the current CNS gate:

- full-training same-protocol benchmark runs on the full available training split are still the next required step
- deeper shared spectral depth remains a manual follow-up rather than a default-profile promotion
- the longer-convergence result should be treated as capped decision-support evidence only

## Verification

Commands run from `/home/ollie/Documents/PtychoPINN`:

```bash
python -m pytest -q tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_image128_preflight.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_swe_splits_data.py tests/studies/test_pdebench_swe_metrics.py tests/studies/test_pdebench_swe_run_config.py tests/studies/test_pdebench_darcy_data.py tests/studies/test_pdebench_darcy_metrics.py
python scripts/studies/run_pdebench_image128_suite.py --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-128x128-image-suite-preflight --markdown-path docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md --no-sha256
python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode readiness --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T040020Z --profiles hybrid_resnet_base,fno_base,unet_tiny_smoke --history-len 2 --epochs 1 --batch-size 1 --max-train-trajectories 2 --max-val-trajectories 1 --max-test-trajectories 1 --max-windows-per-trajectory 1 --device cuda --num-workers 0
pytest -q tests/studies/test_pdebench_image128_runner.py tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py
python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode readiness --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse --profiles hybrid_resnet_base,fno_base,unet_strong --history-len 2 --epochs 10 --batch-size 4 --max-train-trajectories 512 --max-val-trajectories 64 --max-test-trajectories 64 --max-windows-per-trajectory 8 --device cuda --num-workers 0
pytest -q tests/studies/test_pdebench_cfd_cns_data.py tests/studies/test_pdebench_cfd_cns_metrics.py tests/studies/test_pdebench_physics_losses.py tests/studies/test_pdebench_image128_runner.py
python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode readiness --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T210000Z-physics-smoke --profiles hybrid_resnet_base --history-len 2 --epochs 1 --batch-size 4 --max-train-trajectories 8 --max-val-trajectories 2 --max-test-trajectories 2 --max-windows-per-trajectory 2 --device cuda --num-workers 0 --physics-regularization on --physics-loss-weights pos=1.0,cont=0.5,mass=0.25
python -m compileall -q scripts/studies/pdebench_image128 scripts/studies/run_pdebench_image128_suite.py
```

Observed results:

- `47 passed in 29.76s`
- suite preflight completed with exit code `0` and now reports `3 / 3` tasks ready
- capped CNS readiness run completed with exit code `0`
- `12 passed in 22.55s`
- corrected 10-epoch MSE capped run completed with exit code `0`
- `19 passed in 27.36s`
- bounded enabled physics smoke completed with exit code `0`
- `compileall` completed with exit code `0`

Key real-data artifacts:

- Raw preflight JSON: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-128x128-image-suite-preflight/pdebench_image128_suite_preflight.json`
- Markdown preflight: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_preflight.md`
- Readiness comparison summary: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T040020Z/comparison_summary.json`
- Readiness split manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T040020Z/split_manifest.json`
- Readiness normalization stats: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T040020Z/normalization_stats_state.json`
- Corrected 10-epoch MSE comparison summary: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse/comparison_summary.json`
- Corrected 10-epoch MSE split manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T183717Z-10ep-mse/split_manifest.json`
- Physics-regularized enabled smoke summary: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T210000Z-physics-smoke/comparison_summary.json`
- Physics-regularized enabled smoke metrics: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T210000Z-physics-smoke/metrics_hybrid_resnet_base.json`

## Next Gate

The next remaining work is benchmark scope, not readiness scope:

1. Launch same-protocol full-training benchmark runs on the full available CNS training split with `hybrid_resnet_cns`, `fno_base`, and `unet_strong`.
2. Keep the denormalized `err_RMSE`/`err_nRMSE`/`relative_l2` plus `fRMSE_low/mid/high` contract fixed across those benchmark runs.
3. Fold the resulting CNS benchmark row into the later suite summary once all required task-local full-training rows exist.
