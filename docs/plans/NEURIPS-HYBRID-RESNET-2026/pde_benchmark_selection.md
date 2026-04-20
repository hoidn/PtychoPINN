# NeurIPS Hybrid ResNet PDE Benchmark Selection

## Scope and Gate

This is Roadmap Phase 1 for `NEURIPS-HYBRID-RESNET-2026`. It is a selection tranche, not an execution tranche: no PDE adapter, production dependency, full dataset download, PDE training, CDI anchor regeneration, `256x256` scaling, or `/home/ollie/Documents/neurips/` paper-facing artifact is created here.

Gate status: selected with a bounded primary path.

- Primary benchmark: PDEBench 2D fluids, narrowed for Phase 2 to a disk-feasible official fluid dataset path, with shallow-water (`swe`) as the preferred full-data path under the current disk limit. If the paper claim requires Navier-Stokes or compressible CFD specifically, Phase 2 must use an explicitly approved generated or downloaded smoke subset and label published rows as protocol-caveated context.
- Fallback benchmark: OpenFWI 2D acoustic full waveform inversion, narrowed to a FlatVel-A or CurveVel-A small-shard preflight unless additional disk is provided.
- Rejected for this phase: PDEArena Maxwell-3D, because full 3D adaptation and missing local `pdearena` setup exceed the planned Phase 2 risk budget.

All three neutral Phase 0 candidates were evaluated. Phase 0 did not select a primary or fallback; this document is the first durable primary/fallback decision.

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
| PDEBench 2D incompressible Navier-Stokes or compressible fluid task | primary | 26 | Strong architecture fit, mature benchmark, feasible local FNO/U-Net baselines, and a disk-feasible PDEBench fluid path via `swe`; canonical `ns_incom` and `2d_cfd` are disk-blocked here. |
| OpenFWI 2D acoustic full waveform inversion | fallback | 26 | Excellent inverse-wave story and clear MAE/RMSE/SSIM metrics; smallest full 2D dataset is 43 GB, so Phase 2 starts with a small-shard access proof or additional disk. |
| PDEArena Maxwell-3D | rejected | 17 | Strong wave/spectral theme, but missing local `pdearena`, unknown shard size, and likely full 3D Hybrid ResNet adaptation make it too risky for this tranche. |

The equal total for PDEBench and OpenFWI does not create a tie because the gate is not purely score-based. PDEBench has two immediately feasible local baselines and a disk-feasible full official fluid path. OpenFWI is the fallback because its smallest full dataset is larger than the currently free local disk, even though a small-shard path is plausible.

## Primary Benchmark Decision

Select PDEBench 2D fluids as the primary Phase 2 benchmark.

Phase 2 should use the disk-feasible PDEBench shallow-water (`swe`) full official dataset path unless the next plan explicitly approves a generated or smoke-sized NS/CFD variant. This keeps the selected benchmark in the PDEBench fluids family while respecting the local 30.51 GB free-disk constraint. Full `ns_incom` and full `2d_cfd` are not selected for immediate execution because their published download sizes exceed the current filesystem budget.

Rationale:

- The spectral/local Hybrid ResNet story is strongest on 2D fluid prediction, where local coherent structures and long-range fields are both relevant.
- Local baselines are practical: FNO and U-Net can run locally, with persistence or coarse rollout as a smoke baseline if protocol-compatible.
- The Phase 2 implementation can remain a runbook or narrow adapter instead of a broad architecture rewrite.
- The candidate preserves the required non-CDI PDE pillar before CDI polish.

## Fallback Benchmark Decision

Select OpenFWI 2D acoustic full waveform inversion as the fallback.

Fallback execution should start with FlatVel-A or CurveVel-A and prove small-shard access before training. The OpenFWI website lists FlatVel-A/B and CurveVel-A/B as 43 GB each with input shape `(5,1000,70)` and output shape `(1,70,70)`. The repository describes `.npy` batches of 500 samples and a Vel-family 24k/6k split. Because 43 GB exceeds current free disk, the fallback is ready only if small-shard access works or additional data storage is provided.

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

Primary PDEBench metric contract:

- Prediction target: 2D fluid field next-step or rollout prediction from the selected PDEBench fluid dataset.
- Split expectation: official selected HDF5 split or generated train/validation/test manifest recorded in Phase 2.
- Primary metric: normalized RMSE or the selected PDEBench field-error metric pinned from the upstream metric code.
- Secondary metrics: field-wise error, rollout-horizon error, runtime, and GPU memory.
- Normalization and rollout caveats: Phase 2 must record data normalization, time-history/time-future settings, rollout horizon, and whether the result is one-step, autoregressive, or pushforward.
- Published-number source: PDEBench paper/repository rows only after matching task and metric; otherwise use as protocol-caveated context.
- Phase 2 metrics/provenance output: write a metrics JSON/CSV with dataset identity, split, normalization, seed, command, git commit, package versions, GPU name, memory summary, and baseline definitions.

Fallback OpenFWI metric contract:

- Prediction target: seismic shot-gather to velocity-map inversion.
- Split expectation: FlatVel-A or CurveVel-A shard manifest for smoke; full official Vel-family split only if disk permits.
- Primary metric: MAE.
- Secondary metrics: RMSE and SSIM.
- Published-number source: OpenFWI benchmark page rows, protocol-caveated unless full split and metric implementation are reproduced locally.

## Data Access and License Plan

Primary:

- Intended dataset family/shard: PDEBench 2D fluids, with `swe` as the preferred full official disk-feasible path; NS/CFD only as an approved generated or smoke subset.
- Expected path: DaRUS/PDEBench download or generated HDF5 files, stored outside git and recorded in Phase 2 provenance.
- License/access: PDEBench top-level code is MIT except where otherwise stated; some files carry a noncommercial research header. DaRUS data terms and exact imported code files must be recorded before use.
- Checksum/manifest expectation: Phase 2 must write selected URL, DOI where available, file list, checksum or size/mtime manifest, and split identity.
- Disk/GPU implication: `swe` is listed as 6.2 GB and fits current disk; full `2d_cfd` and `ns_incom` do not. A bounded 2D smoke run should fit the RTX 3090.

Fallback:

- Intended dataset family/shard: OpenFWI FlatVel-A or CurveVel-A, starting with one or two `.npy` data/model shard pairs.
- Expected path: official OpenFWI download link or verified shard mirror, stored outside git.
- License/access: code is BSD-3-Clause; datasets are Creative Commons Attribution-NonCommercial-ShareAlike 4.0.
- Checksum/manifest expectation: Phase 2 must write exact shard list, URL, checksum or manifest, and split identity.
- Disk/GPU implication: full FlatVel-A is 43 GB and does not fit current free disk; a small-shard pass should fit the RTX 3090 if access is available.

## Local Baseline Plan

Primary PDEBench baselines:

- FNO: expected to run locally through `neuralop` or a PDEBench-compatible FNO adapter.
- U-Net: expected to run locally through PDEBench conventions or a narrow local 2D adapter.
- Persistence/coarse rollout: optional smoke sanity baseline if compatible with the selected task and rollout metric.

Fallback OpenFWI baselines:

- InversionNet: expected local baseline after compatibility proof with the current or isolated PyTorch environment.
- VelocityGAN: second local baseline if setup time allows; otherwise use a local U-Net/FNO-style adapter and cite VelocityGAN as published context.
- UPFWI and InversionNet3D: published-SOTA context only for this roadmap unless separately approved because they are multi-GPU or 3D-heavy.

## Published SOTA Caveats

Published SOTA and benchmark rows are allowed only as labeled context. They are not same-protocol reproduction unless Phase 2 uses the same task, split, preprocessing, model code or accepted reimplementation, metric code, and evaluation horizon.

For PDEBench, published FNO/U-Net/PINN rows should be labeled as PDEBench-paper context until locally reproduced. For OpenFWI, published InversionNet/VelocityGAN/UPFWI/InversionNet3D rows should be labeled as OpenFWI benchmark context until the full official split and metrics are reproduced.

## Phase 2 Handoff

Phase 2 must start with a smoke/data-load check for the selected primary before any full PDE execution or CDI polish.

Primary handoff:

1. Choose `swe` full official data or obtain explicit approval for generated/smoke NS/CFD.
2. Record dataset URL, license/access terms, checksum or manifest, split identity, and expected disk use.
3. Load a bounded HDF5 shard with `h5py`.
4. Run one tiny Hybrid ResNet-compatible train/eval pass on the RTX 3090.
5. Run FNO and U-Net, or one of those plus persistence/coarse baseline, under the same metric contract.
6. Write metrics and provenance before any longer run.

Fallback handoff:

1. Select FlatVel-A or CurveVel-A.
2. Prove one or two `.npy` data/model shard pairs can be obtained without downloading the full 43 GB dataset, or secure additional storage.
3. Validate shapes `(500,5,1000,70)` and `(500,1,70,70)`.
4. Run one tiny InversionNet or local U-Net pass and one Hybrid ResNet-compatible pass.
5. Write MAE, RMSE, SSIM, normalization, seed, command, and environment provenance.

## Pivot and Blocked Conditions

Pivot from PDEBench primary to OpenFWI fallback before spending CDI polish time if any of these gates fail:

- no disk-feasible PDEBench fluid path is accepted
- selected data cannot be downloaded, generated, or loaded without full-dataset overrun
- metric contract cannot be pinned
- FNO/U-Net or equivalent local baselines cannot be run quickly
- Hybrid ResNet is clearly noncompetitive in the smoke/early local baseline comparison

Mark the roadmap blocked instead of forcing a weak benchmark if both PDEBench and OpenFWI fail data access or metric gates. If a full canonical NS/CFD benchmark is required rather than a disk-feasible PDEBench fluid path, the next human decision is whether to provide more storage/compute or relax the exact PDEBench subtask.

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

Planned structural checks:

- selection document contains primary, fallback, RTX 3090, baseline, and SOTA decision fields
- scorecard JSON has exactly the three Phase 0 candidate IDs and a valid selected decision
- `docs/index.md` links this selection document
- `plan_path.txt` still points to `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-1-pde-benchmark-selection/execution_plan.md`
- no `/home/ollie/Documents/neurips/` paper-facing artifacts are created by this tranche
