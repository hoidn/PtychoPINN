# PDEBench 2D Compressible Navier-Stokes 128x128 Design

## Design Metadata

- ID: `NEURIPS-HYBRID-RESNET-2026-pdebench-2d-cfd-cns`
- Title: PDEBench 2D Compressible Navier-Stokes 128x128 Adapter
- Status: draft design; active replacement for the third PDEBench image-suite member
- Date: 2026-04-20
- Source brief / issue: pivot the third PDEBench image-suite member from 2D diffusion-reaction to 2D Compressible Navier-Stokes (`2d_cfd`) if a `128x128` file is feasible.
- Experiment root: `/home/ollie/Documents/PtychoPINN/`
- Data root: `/home/ollie/Documents/pdebench-data/` or an approved external data root.
- Manuscript artifact root: `/home/ollie/Documents/neurips/` (future Phase 5 root; this design must not create it)

## Consumed Inputs and Authority

- Docs index: `docs/index.md`.
- Primary campaign design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`.
- Primary roadmap: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`.
- Suite plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`.
- Official PDEBench repository README, accessed 2026-04-20.
- Official PDEBench data-download README, accessed 2026-04-20.
- Official PDEBench `pdebench_data_urls.csv`, accessed 2026-04-20.
- Official PDEBench visualization script for expected 2D CFD field keys and filename convention, accessed 2026-04-20.
- Official PDEBench metric code and supplement tables for Fourier-space RMSE (`fRMSE low`, `fRMSE mid`, `fRMSE high`) on 2D CFD, accessed 2026-04-20.

## Feasibility Finding

PDEBench's high-level downloader lists `2d_cfd` as a `551 GB` family download. That full family is not a local-root target on the current filesystem because it is larger than the full `457 GB` root volume.

An individual official `128x128` CNS file is different. After deleting the abandoned diffusion-reaction partial download and later local cleanup, the root filesystem had `75,243,819,008` bytes available before CNS staging. The preferred single CNS file is `55,050,245,208` bytes, so the local root now clears the `60 GB` staging gate for one direct-file CNS attempt. The blocker was therefore storage hygiene/approval for a single file, not fundamental infeasibility.

The official URL manifest does expose individual `128x128` 2D CFD training files. The three checked direct files each report `Content-Length = 55,050,245,208` bytes, about `51.3 GiB`:

| Candidate | Official filename | DaRUS datafile | MD5 from manifest | Role |
| --- | --- | --- | --- | --- |
| CNS bridge | `2D_CFD_Rand_M0.1_Eta0.1_Zeta0.1_periodic_128_Train.hdf5` | `164688` | `b2733888745a1d64e36df813e979910b` | Smoothest official 128x128 CNS file; useful if the transonic file is too unstable or too expensive. |
| CNS primary | `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5` | `164690` | `21969082d0e9524bcc4708e216148e60` | Recommended first benchmark target: transonic and lower-viscosity while still an official 128x128 file. |
| CNS viscous transonic | `2D_CFD_Rand_M1.0_Eta0.1_Zeta0.1_periodic_128_Train.hdf5` | `164691` | `75a34f34a3dcd9bc8ae4b7adf8bf372c` | Secondary easier transonic row if a two-regime comparison fits. |

The official manifest's inviscid random and turbulent 2D CFD entries are `512x512`, not `128x128`. They are out of the local `128x128` image-suite scope unless a later plan uses external storage and explicitly adds downsampled or official-512 rows.

## Decision Summary

- Replace 2D diffusion-reaction as the third active PDEBench image-suite member with 2D Compressible Navier-Stokes (`2d_cfd_cns`).
- Do not download the full `2d_cfd` family. Target one official `128x128` training file first.
- Preferred first target: `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`.
- Storage gate: stage the file only after either an external data root is available or at least `60 GB` is free on the target filesystem. This gate was met locally on 2026-04-20 before starting the preferred CNS file download. Do not delete study artifacts, SWE, or Darcy data without explicit approval.
- Treat the `M0.1, eta=zeta=0.1` file as a bridge/fallback if the primary transonic file is too unstable for a first same-protocol comparison under the one-RTX-3090 budget.
- Keep diffusion-reaction as a deferred optional task, not an active suite dependency.

## Data Contract

The exact HDF5 schema must be verified after staging. The expected schema from PDEBench code is separate physical field datasets:

- `density`
- `Vx`
- `Vy`
- `pressure`
- coordinate arrays such as `x-coordinate`, `y-coordinate`, and possibly `t-coordinate`

Expected logical shape is one trajectory/sample axis, one time axis, two spatial axes, and four physical fields. The adapter should convert each example to channel-first tensors:

- Primary `history_len=2`: `input=(8,128,128)` and `target=(4,128,128)`.
- Required low-context ablation if budget permits: `history_len=1`, `input=(4,128,128)` and `target=(4,128,128)`.
- Optional later rows: longer history or multi-step rollout only after one-step rows are stable.

Field order must be explicit and stable: `[density, Vx, Vy, pressure]`.

## Split, Normalization, And Metrics

- Split at trajectory/sample level before expanding history windows.
- Meaningful benchmark rows must use the full available training split after validation/test holdout. Capped rows are readiness or pilot evidence only.
- Fit train-only per-field normalization and reuse the same field stats across history slots and targets.
- Primary metrics: denormalized aggregate nRMSE/RMSE over all four fields plus per-field nRMSE/RMSE.
- Required CNS spectral diagnostic: denormalized Fourier-space RMSE bands `fRMSE_low`, `fRMSE_mid`, and `fRMSE_high`, reported aggregate and per field where storage/report size permits. `fRMSE_high` is the primary shock/small-scale-structure diagnostic because pointwise RMSE can miss blurred or spectrally damped shocks.
- Implementation rule for fRMSE: first try to match PDEBench's 2D convention, which radially bins 2D FFT error and reports low/mid/high frequency ranges. If the local one-step tensor layout requires adaptation, define the exact radial bands in the CNS implementation plan before the first benchmark run and keep them fixed across Hybrid ResNet, FNO, and U-Net.
- Recommended additional diagnostics: shock/front error slices or gradient-weighted error, and conservation-drift checks if implementation cost is low.
- Rollout metrics are deferred until one-step evidence is stable.

## Model And Baseline Contract

All same-protocol rows use the supervised real-channel PDE adapter:

- `hybrid_resnet_cns`: canonical CNS Hybrid row. It keeps `fno_modes=12`, hidden width `32`, `fno_blocks=4`, and Hybrid ResNet downsample/local depth `2/6`, enables encoder-decoder skip fusion with `hybrid_skip_connections=on` and `hybrid_skip_style=add`, and now defaults to `hybrid_upsampler=pixelshuffle`.
- `hybrid_resnet_base`: remains the generic supervised Hybrid profile for other tasks; it is not the default CNS benchmark row anymore.
- `fno_base`: same history/target tensors and task split.
- Strong U-Net: a non-toy U-Net profile with recorded parameter count.

Training recipe guardrail:

- Adam `lr=2e-4`.
- `ReduceLROnPlateau(factor=0.5, patience=2, min_lr=1e-5, threshold=0.0)`.
- Scheduler floor must be no higher than `1e-5` for PDE benchmark rows unless a later plan records a justified pre-run override.

## Gates

- Data gate: official file staged under an ignored/external root with recorded URL, filename, MD5 or checksum policy, byte size, and license/access note.
- Schema gate: HDF5 keys, shapes, dtype, coordinate arrays, field order, and number of time steps recorded before adapter implementation.
- Storage gate: no file download to the root filesystem unless free space is at least `60 GB` before launch.
- Benchmark gate: no performance claim unless Hybrid ResNet, FNO, and strong U-Net complete on the same full training split, normalization, history length, and metrics.
- Suite gate: replacing diffusion-reaction with CNS is a deliberate scope change; the suite summary must not report diffusion-reaction as missing once this design is adopted.

## Planning Handoff

The next implementation plan should be a data/schema preflight and adapter-design tranche, not an expensive training tranche. It should decide:

- Whether the target data root is external or local after approved cleanup.
- Which single file is staged first, with `M1.0_Eta0.01_Zeta0.01` as the default.
- How to generalize the existing dynamic-task loader for separate-field HDF5 datasets.
- Which test modules own field stacking, history-window construction, split manifests, normalization, run-budget validation, denormalized nRMSE/RMSE, and frequency-band fRMSE metrics.
