# Phase 1 PDE Benchmark Selection Execution Report

## Completed In This Pass

- Verified the implementation review finding against the Phase 1 plan, current selection document, and scorecard: the previous output selected benchmark families but left primary/fallback execution contracts open.
- Amended `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md` to select one exact primary benchmark: PDEBench 2D Shallow Water Equations (`swe`) using `2D_rdb_NA_NA.h5`.
- Recorded the narrowing rationale from the Phase 0 PDEBench NS/CFD wording to `swe`: official PDEBench 2D fluid task, 6.2 GB listed size, executable under the current 30.51 GB free-disk constraint, while full `2d_cfd` and `ns_incom` are not selected.
- Amended the fallback to one exact benchmark: OpenFWI FlatVel-A 2D acoustic FWI, with `data1.npy`/`model1.npy` and `data49.npy`/`model49.npy` as the Phase 2 smoke shard policy.
- Updated the ignored Phase 1 scorecard and source notes to match the pinned primary/fallback contracts.
- Updated the `docs/index.md` entry for the selection document.
- Replaced "planned structural checks" language with executed verification evidence in the durable selection document.

## Completed Current-Scope Work

- Blocking review item fixed: the Phase 1 handoff now names an exact primary task/data path, split expectation, one-step policy, primary metric, local baseline protocol, and exact fallback dataset/shard policy.
- Low-severity review item fixed: the durable selection document now summarizes executed checks and links this execution report.
- Current-scope Phase 1 work remains limited to selection, source notes, scorecard, discoverability, and report updates. No PDE adapter, production dependency, dataset download, PDE training, CDI work, `256x256` scaling, paper-facing `/home/ollie/Documents/neurips/` artifact, worktree, or stable core-module edit was introduced.

## Follow-Up Work

- Phase 2 must start by staging PDEBench `swe` file `2D_rdb_NA_NA.h5`, recording URL/DaRUS ID, checksum or size/mtime manifest, license/access terms, HDF5 field/channel metadata, and deterministic 80/10/10 trajectory split manifest with seed `20260420`.
- Phase 2 must run the selected primary smoke/data-load gate before full PDE execution or CDI polish: bounded HDF5 load, tiny Hybrid ResNet-compatible one-step train/eval, FNO baseline, U-Net baseline, `err_nRMSE` metrics, runtime, memory, and provenance.
- If the primary fails, Phase 2 should pivot to OpenFWI FlatVel-A and first prove access to `data1.npy`/`model1.npy` and `data49.npy`/`model49.npy`, then run the MAE/RMSE/SSIM smoke protocol.
- CDI anchor regeneration, `256x256` scaling, and `/home/ollie/Documents/neurips/` paper-facing evidence assembly remain later roadmap phases.

## Residual Risks

- PDEBench `swe` is an explicit, documented narrowing from the Phase 0 NS/CFD wording; if the paper requires NS/CFD specifically, the roadmap needs a new storage/compute decision and a revised primary benchmark.
- PDEBench metric code carries noncommercial research license text in at least one source file, so Phase 2 may need a documented equivalent `err_nRMSE` implementation instead of importing upstream code directly.
- OpenFWI FlatVel-A full execution still exceeds current free disk at 43 GB; the fallback is quick only if exact shard-level access works or additional storage is provided.
- Published SOTA rows remain protocol-caveated unless Phase 2 reproduces the same task, split, preprocessing, horizon, metric code, and evaluation protocol.

## Verification

- `selection doc contains required decision fields`
- `scorecard decision is structurally valid`
- `source notes contain required evidence fields`
- `Phase 1 selection artifacts are structurally valid`
- `scorecard structure is valid`
- `discoverability and output pointer are valid`
- `{'paper_facing_paths_to_inspect': []}`

No `pytest` selector was required because this pass changed Markdown and ignored JSON/source-note/report artifacts only.
