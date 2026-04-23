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
