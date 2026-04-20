# OpenFWI FlatVel-A Fallback Smoke Gate

## Scope

This is the Roadmap Phase 2 fallback smoke/data-access gate for OpenFWI FlatVel-A after the PDEBench SWE primary recorded `Decision: pivot to OpenFWI FlatVel-A`.

The delivered code path adds a narrow study harness for source/shard manifests, shape validation, deterministic smoke splits, normalization, MAE/RMSE/SSIM metrics, tiny Hybrid ResNet-compatible and U-Net local profiles, optional official InversionNet metadata/import/forward probing, invocation provenance, comparison collation, and controlled blocker output.

Explicit non-goals confirmed: no CDI anchor regeneration, no CDI baseline or ablation work, no 256x256 scaling, no `/home/ollie/Documents/neurips/` paper-facing artifact assembly, no full 43 GB OpenFWI FlatVel-A download, no benchmark-family switch, no PDEBench SWE rerun, no stable core physics/model edits, and no worktree creation.

## Documents And Artifacts Used

- Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Execution plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-openfwi-flatvel-a-fallback-smoke-gate/execution_plan.md`
- Tranche context: `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/tranche-context.md`
- Phase 1 selection: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- OpenFWI source note: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`
- SWE pivot summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- Raw support root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/`

## Source, Access, And License

- Source URL: `https://openfwi-lanl.github.io/docs/data.html`
- Dataset: OpenFWI FlatVel-A.
- Required smoke shards: `data1.npy`, `model1.npy`, `data49.npy`, `model49.npy`.
- License note: OpenFWI datasets are CC BY-NC-SA 4.0; OpenFWI code is BSD-3-Clause.
- Access status: blocked in this pass because `OPENFWI_FLATVEL_A_ROOT` was not set and the fallback candidate root `/home/ollie/Documents/openfwi-flatvel-a` did not contain the four required smoke shards.
- Data-root policy: OpenFWI data must remain outside git or under an ignored artifact/external root. This pass did not redistribute or download data.

## Shard Identity And Shape Validation

The smoke gate requires seismic shard shape `(500, 5, 1000, 70)` and velocity shard shape `(500, 1, 70, 70)` before any training. Because the required FlatVel-A shards were absent, shape validation did not run against real data.

The implemented real-run validator now treats the leading sample count as part of the FlatVel-A contract, not just the trailing tensor axes. Synthetic fixture runs must opt in explicitly with `--allow-synthetic-shard-samples`; those artifacts record `sample_count_contract: synthetic_fixture` and are not valid real FlatVel-A evidence.

Controlled blocker:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/data_access_blocker.json`

Missing files:

- `data1.npy`
- `model1.npy`
- `data49.npy`
- `model49.npy`

## Split, Normalization, And Preprocessing Contract

The implemented smoke contract uses seed `20260420`, `data1.npy`/`model1.npy` for train smoke, and `data49.npy`/`model49.npy` for validation/test smoke. Default caps are `32` train samples, `16` validation samples, and `16` test samples.

For local 2D adapters, raw seismic samples keep `input_raw` shape `(5, 1000, 70)` and are deterministically resized with bilinear interpolation to `(5, 70, 70)`. Velocity targets keep shape `(1, 70, 70)`. Normalization stats are computed from train split inputs and targets only. Because data access blocked, no real split manifest or normalization stats were generated for FlatVel-A shards.

## Metric Contract

The metric writer reports MAE as primary and RMSE/SSIM as secondary. Metrics are computed on denormalized velocity maps when target normalization is used. SSIM uses a per-sample data range from the combined prediction/target range with a minimum range guard for constant maps.

## Official InversionNet Compatibility

Official InversionNet compatibility was not attempted against a real external checkout because data access blocked before model execution. The implemented probe accepts `--official-openfwi-repo`, records repository path, git commit, and license path, imports `network.py`, `model.py`, or `models.py` from the supplied checkout, resolves `InversionNet` from either a class or `model_dict`, and runs a bounded CPU forward-shape probe from `(1, 5, 1000, 70)` to `(1, 1, 70, 70)`. Missing checkout, import failure, missing `InversionNet`, or forward/shape failure is recorded as a controlled official-code blocker. OpenFWI official rows remain published-context only unless a later tranche reproduces the full official split and metrics.

## Local Model Results Or Blockers

No real FlatVel-A model execution was launched because shard access blocked. The comparison summary records both local profiles as blocked by `missing_required_shards`:

- Hybrid ResNet-compatible profile: `hybrid_resnet_smoke`
- Baseline profile: `unet_smoke`

The implemented synthetic CPU tests verify that both profiles can train/evaluate on synthetic FlatVel-A-shaped shards and write metrics/provenance/comparison artifacts under the same smoke contract.

## Runtime, Memory, And Provenance

- Run ID: `openfwi-smoke-data-access-blocked-20260420`
- CUDA available: `true`
- GPU: `NVIDIA GeForce RTX 3090`
- Python: PATH `python`, Python 3.11.13 in the current environment
- Packages recorded: NumPy 1.26.4, torch 2.9.1, scikit-image 0.25.2
- Free disk at preflight: `25959723008` bytes
- Invocation artifacts:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/invocation.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/invocation.sh`
- Preflight artifacts:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/preflight/disk_gpu.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/preflight/package_provenance.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/preflight/openfwi_source_access.md`

## Published-Context Caveats

Published OpenFWI MAE/RMSE/SSIM rows for InversionNet, VelocityGAN, UPFWI, and InversionNet3D are not same-protocol evidence for this tranche. Any later paper use must label them as published context unless the full official split, preprocessing, model code or accepted reimplementation, and metric protocol are reproduced locally.

## Gate Checks

- Data access: blocked, with required shard names and candidate local root recorded.
- Shape validation: blocked by missing shards.
- Split manifest: implemented and tested on synthetic shards; not generated for real FlatVel-A data.
- Normalization: implemented and tested on synthetic train split only; not generated for real FlatVel-A data.
- Metrics: MAE, RMSE, and SSIM implemented and tested.
- Hybrid ResNet-compatible smoke: implemented and tested synthetically; blocked for real data.
- Local baseline smoke: U-Net implemented and tested synthetically; blocked for real data.
- Official InversionNet: metadata/import/forward probe implemented; not attempted against a real checkout because data access blocked.
- Long-run guard: the CLI accepts only the approved prelaunch layout at the selected run root (`logs/smoke.run_id`, `logs/smoke.started_at_ns`, and the current launcher-owned `logs/smoke.pid`) before writing. It still rejects other live or incomplete `logs/smoke.pid` markers at the selected output root and nested `runs/*/logs/smoke.pid` markers.
- Reused-root freshness: once data resolution succeeds, obsolete `data_access_blocker.json` files are cleared before current artifacts are written. The comparison collator also validates blocker run IDs and ignores a stale data-access blocker when current-run metrics exist, preventing stale blockers from overriding successful current metrics.
- Long-running tmux launch: not run because the freshness gate blocked before model execution.

## Raw Artifact Links

- Blocker: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/data_access_blocker.json`
- Comparison JSON: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/comparison_summary.json`
- Comparison CSV: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/comparison_summary.csv`

## Residual Risks

- The fallback PDE pillar remains blocked until the four pinned FlatVel-A smoke shards are staged or a human approves full/external storage.
- Synthetic smoke tests prove code paths and artifact schemas, not scientific viability on real OpenFWI data.
- Official InversionNet baseline execution still needs an external checkout plus real shard access; the compatibility probe only proves bounded import and forward-shape viability.

Decision: block for storage/data/human decision
