# NeurIPS Hybrid ResNet PDE Benchmark Selection

## Scope and Gate

This is Roadmap Phase 1 for `NEURIPS-HYBRID-RESNET-2026`. It is a selection tranche, not an execution tranche: no PDE adapter, production dependency, full dataset download, PDE training, CDI anchor regeneration, `256x256` scaling, or `/home/ollie/Documents/neurips/` paper-facing artifact is created here.

Gate status: selected with an exact primary benchmark and exact fallback benchmark, then amended on 2026-04-20 to a compact native `128x128` PDEBench image suite.

- Primary benchmark: PDEBench 2D Shallow Water Equations (`swe`) forward prediction, using the official `2D_rdb_NA_NA.h5` file from the PDEBench `swe` download path (`python download_direct.py --root_folder $PDE_DATA --pde_name swe`) or the documented DaRUS datafile `133021`.
- Scope narrowing: Phase 0 named "PDEBench 2D incompressible Navier-Stokes or compressible fluid task." This Phase 1 selection first narrowed that family to `swe` because it was executable under the then-current disk constraint. On 2026-04-20, the user explicitly reopened 2D Compressible Navier-Stokes as the harder third suite member. Full `2d_cfd` remains a 551 GB family download, but official individual `128x128` random-viscous training files are available at about 55.05 GB each.
- Fallback benchmark: OpenFWI FlatVel-A 2D acoustic full waveform inversion. Phase 2 preflight uses `data1.npy`/`model1.npy` as the train smoke shard pair and `data49.npy`/`model49.npy` as the validation/test smoke shard pair if shard-level access is available; full fallback execution uses the Vel-family split `data(model)1-48.npy` for training and `data(model)49-60.npy` for test, which requires additional storage.
- Rejected for this phase: PDEArena Maxwell-3D, because full 3D adaptation and missing local `pdearena` setup exceed the planned Phase 2 risk budget.

All three neutral Phase 0 candidates were evaluated. Phase 0 did not select a primary or fallback; this document is the first durable primary/fallback decision.

## 2026-04-20 Scope Amendment

The approved Phase 2 PDE path is now a native `128x128` PDEBench image suite:

- SWE (`swe`), retaining the staged `2D_rdb_NA_NA.h5` file.
- Darcy Flow (`darcy`), expected from PDEBench as `2D_DarcyFlow_beta1.0_Train.hdf5`.
- 2D Compressible Navier-Stokes (`2d_cfd_cns`), expected first target `2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5`.

This amendment supersedes the old execution order of "SWE performance result, then OpenFWI fallback performance result" while preserving the original scorecard as selection provenance. OpenFWI FlatVel-A is reclassified as an optional fallback or adjacent inverse-wave extension because its velocity targets are `70x70`, not native `128x128`.

The current implementation handoff is `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`. The next CNS tranche should begin with data/storage/schema preflight plus a shared adapter plan, not an expensive training launch.

Smoke gates remain readiness checks only. No smoke metric from SWE, OpenFWI, Darcy, or 2D CNS can rank models, trigger a performance pivot, support a paper-facing claim, or satisfy the PDE competitiveness gate.

For 2D CNS, the benchmark metric contract must include denormalized nRMSE/RMSE plus PDEBench-style Fourier-space RMSE bands. `fRMSE_high` is the required shock/small-scale-structure diagnostic, with `fRMSE_low` and `fRMSE_mid` reported beside it to prevent high-frequency-only cherry-picking.

Meaningful benchmark-performance rows must train on the full available training split for the selected official file after validation/test holdout. If a selected task has 10,000 total samples and 2,000 are held out for validation/test, the benchmark row trains on the remaining 8,000 samples. Capped, subsampled, smoke, and pilot runs are operational evidence only.

## Documents and Sources Used

Repository documents and artifacts:

- `docs/index.md`
- `docs/findings.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`
- `state/NEURIPS-HYBRID-RESNET-2026/items/phase-1-pde-benchmark-selection/tranche-context.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_inventory.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/pde_candidate_inventory.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-0-evidence-inventory/environment_probe.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/environment_recheck.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json`

Primary benchmark sources, accessed 2026-04-20:

- PDEBench paper: https://arxiv.org/abs/2210.07182
- PDEBench repository: https://github.com/pdebench/PDEBench
- PDEBench license: https://raw.githubusercontent.com/pdebench/PDEBench/main/LICENSE.txt
- PDEBench data download README: https://raw.githubusercontent.com/pdebench/PDEBench/main/pdebench/data_download/README.md
- PDEArena paper: https://arxiv.org/abs/2209.15616
- PDEArena repository: https://github.com/pdearena/pdearena
- PDEArena data docs: https://pdearena.github.io/pdearena/datadownload/
- PDEArena generation docs: https://pdearena.github.io/pdearena/data/
- PDEArena architecture docs: https://pdearena.github.io/pdearena/architectures/
- OpenFWI paper: https://arxiv.org/abs/2111.02926
- OpenFWI project: https://openfwi-lanl.github.io/
- OpenFWI data page: https://openfwi-lanl.github.io/docs/data.html
- OpenFWI benchmark page: https://openfwi-lanl.github.io/docs/benchmark.html
- OpenFWI repository: https://github.com/lanl/OpenFWI

## Scorecard Schema

Rubric scale: `0=unacceptable`, `1=high risk`, `2=acceptable with caveats`, `3=strong`.

Decision rule: gate-first. A primary candidate must have acceptable metric clarity, runnable data access, RTX 3090 smoke feasibility, feasible local baselines, and a defensible spectral/local Hybrid ResNet paper story. Score totals explain the decision, but do not override failed gates.

Required fields used for each candidate: architectural fit, benchmark maturity, metric clarity, data access, data size, install burden, license/access, RTX 3090 feasibility, local baseline feasibility, published SOTA availability, paper-story fit, risks, required Phase 2 preflight, recommended status, and source-note path.

## Candidate Scorecard

| Candidate | Status | Total | Gate Summary |
| --- | --- | ---: | --- |
| PDEBench 2D Shallow Water Equations (`swe`) | primary | 26 | Strong architecture fit, mature benchmark, feasible local FNO/U-Net baselines, exact official data file `2D_rdb_NA_NA.h5`, and disk-feasible 6.2 GB size; this is the selected narrowing of the Phase 0 PDEBench fluids candidate. |
| OpenFWI FlatVel-A 2D acoustic full waveform inversion | fallback | 26 | Excellent inverse-wave story, clear MAE/RMSE/SSIM metrics, official split/shard naming, and simple-layer FlatVel-A task; full 43 GB dataset needs more disk, so Phase 2 starts with exact train/test shard pairs. |
| PDEArena Maxwell-3D | rejected | 18 | Strong wave/spectral theme, but missing local `pdearena`, unknown shard size, and likely full 3D Hybrid ResNet adaptation make it too risky for this tranche. |

The equal total for PDEBench and OpenFWI does not create a tie because the gate is not purely score-based. PDEBench has two immediately feasible local baselines and a disk-feasible full official fluid path. OpenFWI is the fallback because its smallest full dataset is larger than the currently free local disk, even though a small-shard path is plausible.

## Historical Primary Benchmark Decision

The original Phase 1 decision selected PDEBench 2D Shallow Water Equations (`swe`) as the primary Phase 2 benchmark.

Under that original decision, Phase 2 had to use the official `swe` data path: `2D_rdb_NA_NA.h5`, obtained through `python download_direct.py --root_folder $PDE_DATA --pde_name swe` or the documented DaRUS datafile `133021`. This kept the selected benchmark inside the PDEBench fluids family while making the Phase 2 handoff executable on the current machine. Generated NS/CFD subsets and full `ns_incom` or `2d_cfd` downloads were not part of the selected primary benchmark.

Rationale:

- The spectral/local Hybrid ResNet story is strong on 2D shallow-water prediction, where local wave structures and global field evolution are both relevant.
- Local baselines are practical: FNO and U-Net can run locally, with persistence or coarse rollout as a smoke baseline if protocol-compatible.
- The Phase 2 implementation can remain a runbook or narrow adapter instead of a broad architecture rewrite.
- The candidate preserves the required non-CDI PDE pillar before CDI polish.
- The narrowing from the Phase 0 NS/CFD wording is explicit and evidence-driven: `swe` is the only official PDEBench 2D fluid path identified in this pass that fits the local disk budget without replacing the benchmark with a generated subset.

## Historical Fallback Benchmark Decision

The original Phase 1 decision selected OpenFWI FlatVel-A 2D acoustic full waveform inversion as the fallback.

Under that original fallback decision, execution would start with FlatVel-A because it is the simple-layer Vel-family dataset used in the official repository examples, has published benchmark rows, and has the same 43 GB size as CurveVel-A while presenting lower setup risk. Phase 2 would first obtain `data1.npy`/`model1.npy` and `data49.npy`/`model49.npy` if shard-level access was available. The full FlatVel-A protocol uses 24k training samples from `data(model)1-48.npy` and 6k test samples from `data(model)49-60.npy`. Because the full 43 GB dataset exceeds current free disk, full fallback execution is blocked until additional storage is provided or an approved external data root is used.

Rationale:

- OpenFWI gives a clean inverse reconstruction story adjacent to CDI.
- Official MAE, RMSE, and SSIM metrics are clear.
- Official InversionNet is a plausible local baseline after environment compatibility proof.
- Published rows exist for InversionNet, VelocityGAN, UPFWI, and InversionNet3D, but must remain protocol-caveated unless locally reproduced.

## Rejected Candidate Rationale

PDEArena Maxwell-3D is not selected as primary or fallback.

Reasons:

- `pdearena` is unavailable in the current environment.
- Maxwell-3D is a true 3D task, while the current Hybrid ResNet code path is 2D CDI-oriented.
- A credible full 3D Hybrid ResNet adaptation would likely exceed the roadmap's minimal benchmark-adapter boundary.
- A 2D slice or channel-reduced policy may be useful for smoke checks, but would weaken same-benchmark comparisons.

## Metric Contract

Primary PDEBench SWE metric contract:

- Prediction target: one-step next-state prediction on the official PDEBench `swe` file `2D_rdb_NA_NA.h5`; Phase 2 must record the HDF5 field names, channel order, and tensor axes before training.
- Split expectation: deterministic trajectory-level train/validation/test manifest from the official `swe` file, using an 80/10/10 split and seed `20260420`; the manifest is a Phase 2 gate artifact and must be shared by Hybrid ResNet, FNO, and U-Net.
- Training-set expectation: any meaningful benchmark-performance row trains on the full train portion of that manifest after validation/test holdout. Capped trajectory or one-step-pair subsets are smoke/pilot evidence only unless a later plan changes the scientific task contract before training.
- Primary metric: PDEBench normalized RMSE (`err_nRMSE` from the upstream metrics contract, or a documented equivalent formula if code import is blocked by license terms), averaged over all predicted SWE state channels and one-step evaluation batches.
- Secondary metrics: PDEBench RMSE (`err_RMSE`), conserved-variable RMSE where applicable (`err_CSV`), maximum RMSE (`err_Max`), boundary RMSE (`err_BD`), Fourier-space RMSE (`err_F`), runtime, and peak GPU memory.
- Normalization and rollout caveats: the selected Phase 2 gate is one-step prediction only. Autoregressive rollout or pushforward evaluation may be added as secondary evidence after the one-step contract passes, but cannot replace it without an approved plan update.
- Published-number source: PDEBench paper/repository rows only after matching the `swe` task, split, preprocessing, one-step horizon, and metric; otherwise use them as protocol-caveated context.
- Phase 2 metrics/provenance output: write metrics JSON/CSV with dataset file, DaRUS/download URL, split manifest path, normalization, seed, command, git commit, package versions, GPU name, memory summary, and baseline definitions.

Fallback OpenFWI metric contract:

- Prediction target: seismic shot-gather to velocity-map inversion.
- Split expectation: FlatVel-A shard manifest for smoke using `data1.npy`/`model1.npy` and `data49.npy`/`model49.npy`; full FlatVel-A Vel-family split `data(model)1-48.npy` train and `data(model)49-60.npy` test only if disk permits.
- Primary metric: MAE.
- Secondary metrics: RMSE and SSIM.
- Published-number source: OpenFWI benchmark page rows, protocol-caveated unless full split and metric implementation are reproduced locally.

## Data Access and License Plan

Primary:

- Intended dataset family/shard: PDEBench SWE official full data file `2D_rdb_NA_NA.h5`.
- Expected path: DaRUS/PDEBench `swe` download via `python download_direct.py --root_folder $PDE_DATA --pde_name swe` or DaRUS datafile `133021`, stored outside git and recorded in Phase 2 provenance.
- License/access: PDEBench top-level code is MIT except where otherwise stated; some files carry a noncommercial research header. DaRUS data terms and exact imported code files must be recorded before use.
- Checksum/manifest expectation: Phase 2 must write selected URL, DOI where available, file list, checksum or size/mtime manifest, and the deterministic 80/10/10 trajectory split manifest.
- Disk/GPU implication: `swe` is listed as 6.2 GB and fits current disk; full `2d_cfd` and `ns_incom` do not. A bounded one-step SWE smoke run should fit the RTX 3090.

Fallback:

- Intended dataset family/shard: OpenFWI FlatVel-A, starting with `data1.npy`/`model1.npy` and `data49.npy`/`model49.npy`.
- Expected path: official FlatVel-A OpenFWI download link or verified shard mirror, stored outside git.
- License/access: code is BSD-3-Clause; datasets are Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
- Checksum/manifest expectation: Phase 2 must write exact shard list, URL, checksum or manifest, and split identity.
- Disk/GPU implication: full FlatVel-A is 43 GB and does not fit current free disk; the exact two-shard smoke pass should fit the RTX 3090 if access is available.

## Local Baseline Plan

Primary PDEBench baselines:

- FNO: expected to run locally through `neuralop` or a PDEBench-compatible FNO adapter, using the same SWE split manifest, one-step horizon, normalization, and `err_nRMSE` metric as Hybrid ResNet.
- U-Net: expected to run locally through PDEBench conventions or a narrow local 2D adapter, using the same SWE split manifest, one-step horizon, normalization, and `err_nRMSE` metric as Hybrid ResNet.
- Persistence: optional smoke sanity baseline only; it does not count as one of the two required local baselines unless the Phase 2 plan explicitly justifies that downgrade.

Fallback OpenFWI baselines:

- InversionNet: expected local baseline after compatibility proof with the current or isolated PyTorch environment, first on the FlatVel-A smoke shards.
- VelocityGAN: second local baseline if setup time allows; otherwise use a local U-Net/FNO-style adapter on the same FlatVel-A smoke shards and cite VelocityGAN as published context.
- UPFWI and InversionNet3D: published-SOTA context only for this roadmap unless separately approved because they are multi-GPU or 3D-heavy.

## Published SOTA Caveats

Published SOTA and benchmark rows are allowed only as labeled context. They are not same-protocol reproduction unless Phase 2 uses the same task, split, preprocessing, model code or accepted reimplementation, metric code, and evaluation horizon.

For PDEBench, published FNO/U-Net/PINN rows should be labeled as PDEBench-paper context until locally reproduced. For OpenFWI, published InversionNet/VelocityGAN/UPFWI/InversionNet3D rows should be labeled as OpenFWI benchmark context until the full official split and metrics are reproduced.

## Phase 2 Handoff

The historical Phase 2 handoff below records the original SWE-primary/OpenFWI-fallback path. After the 2026-04-20 amendment, execution should follow `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md` first. The old handoff remains useful for provenance and for a possible SWE-only or OpenFWI fallback decision.

Phase 2 smoke/data-load checks are required before any full PDE execution or CDI polish, but smoke metrics are readiness artifacts only.

Primary handoff:

1. Download or stage PDEBench `swe` file `2D_rdb_NA_NA.h5` outside git.
2. Record dataset URL or DaRUS ID `133021`, license/access terms, checksum or size/mtime manifest, expected disk use, and HDF5 field/channel metadata.
3. Write the deterministic 80/10/10 trajectory split manifest with seed `20260420`.
4. Load a bounded HDF5 shard with `h5py`.
5. Run one tiny Hybrid ResNet-compatible one-step train/eval pass on the RTX 3090.
6. Run FNO and U-Net under the same split, horizon, normalization, and `err_nRMSE` contract.
7. Write metrics and provenance before any longer run.

Fallback handoff:

1. Use OpenFWI FlatVel-A.
2. Prove `data1.npy`/`model1.npy` and `data49.npy`/`model49.npy` can be obtained without downloading the full 43 GB dataset, or secure additional storage.
3. Validate shapes `(500,5,1000,70)` and `(500,1,70,70)`.
4. Run one tiny InversionNet or local U-Net pass and one Hybrid ResNet-compatible pass.
5. Write MAE, RMSE, SSIM, normalization, seed, command, and environment provenance.

## Pivot and Blocked Conditions

Under the amended scope, pivot away from the PDEBench image suite only after the suite plan records which task failed and why. The older SWE-primary/OpenFWI fallback pivot remains available only as an explicit selector decision.

Historical pivot from PDEBench SWE primary to OpenFWI FlatVel-A fallback before spending CDI polish time applied if any of these gates failed:

- official `swe` data cannot be downloaded or loaded without full-dataset overrun
- deterministic SWE split manifest cannot be written from the official file
- one-step `err_nRMSE` metric contract cannot be implemented or license-cleared
- FNO/U-Net or equivalent local baselines cannot be run quickly
- Hybrid ResNet is clearly noncompetitive in a full-training local-baseline comparison that was explicitly launched for performance assessment; smoke, capped, and pilot gates alone cannot trigger a performance pivot

Mark the roadmap blocked instead of forcing a weak benchmark if both PDEBench SWE and OpenFWI FlatVel-A fail data access or metric gates. If a full canonical NS/CFD benchmark is required instead of SWE, the next human decision is whether to provide more storage/compute and approve a new primary, because NS/CFD is not selected by this Phase 1 gate.

## Non-Goals Confirmed

This Phase 1 tranche did not:

- add or install production dependencies
- implement a PDE adapter, metric parser, result writer, or CLI
- download full PDE datasets
- run PDE training or baseline training
- regenerate the `128x128` CDI anchor
- run CDI baselines or ablations
- run `256x256` scaling
- create `/home/ollie/Documents/neurips/` artifacts
- modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`
- create a git worktree

## Raw Artifact Links

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/environment_recheck.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/pde_scorecard.json`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/pdebench_fluids.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/pdearena_maxwell3d.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`

These files are ignored support artifacts and should not be committed.

## Verification

Executed checks from the Phase 1 implementation pass and review-fix pass are summarized in `artifacts/work/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/execution_report.md`.

Confirmed structural checks:

- `selection doc contains required decision fields`
- `scorecard structure is valid`
- `Phase 1 selection artifacts are structurally valid`
- `discoverability and output pointer are valid`
- `{'paper_facing_paths_to_inspect': []}`

No `pytest` selector was required because this tranche modifies Markdown and ignored JSON/source-note artifacts only.
