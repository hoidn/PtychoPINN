# OpenFWI FlatVel-A Fallback Smoke Gate

Status note, 2026-04-20: OpenFWI FlatVel-A remains a readiness-proven optional fallback or adjacent inverse-wave extension. It is not the immediate next benchmark-performance path while the amended native `128x128` PDEBench image-suite plan is viable: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_128x128_image_suite_plan.md`.

## Scope

This is the Roadmap Phase 2 fallback smoke/data-access gate for OpenFWI FlatVel-A after the PDEBench SWE primary recorded `Decision: pivot to OpenFWI FlatVel-A`.

The delivered code path adds a narrow study harness for source/shard manifests, shape validation, deterministic smoke splits, normalization, MAE/RMSE/SSIM metrics, tiny Hybrid ResNet-compatible and U-Net local profiles, optional official InversionNet metadata/import/forward probing, invocation provenance, comparison collation, and controlled blocker output.

Explicit non-goals confirmed: no CDI anchor regeneration, no CDI baseline or ablation work, no 256x256 scaling, no `/home/ollie/Documents/neurips/` paper-facing artifact assembly, no full 43 GB OpenFWI FlatVel-A download, no benchmark-family switch, no PDEBench SWE rerun, no stable core physics/model edits, and no worktree creation.

Post-review training-recipe and evidence-scope correction: smoke gates are valid only for data access, shape validation, metric plumbing, runtime/provenance, and adapter sanity. They are not benchmark-performance evidence, even when they use a grid-lines-style training recipe. The corrected historical smoke run below used `epochs=5`, `fno_modes=12`, hidden width `32`, `fno_blocks=4`, Hybrid ResNet downsample/local depth `2/6`, MAE training loss, Adam `lr=2e-4`, and `ReduceLROnPlateau` with factor `0.5`, patience `2`, and threshold `0.0`, but its scheduler floor predates the current PDE-study guardrail. Any rerun must use scheduler min LR no higher than `1e-5` or record a justified pre-run override. It is still a four-shard smoke/subset result, not the full OpenFWI benchmark, and cannot support model ranking, fallback rejection, or paper-facing OpenFWI claims.

## Documents And Artifacts Used

- Design: `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- Execution plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/tranches/phase-2-openfwi-flatvel-a-fallback-smoke-gate/execution_plan.md`
- Tranche context: `state/NEURIPS-HYBRID-RESNET-2026/items/phase-2-openfwi-flatvel-a-fallback-smoke-gate/tranche-context.md`
- Phase 1 selection: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_benchmark_selection.md`
- OpenFWI source note: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-1-pde-benchmark-selection/source_notes/openfwi_2d_acoustic_fwi.md`
- SWE pivot summary: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pde_execution_summary.md`
- Raw support root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/`
- Historical one-epoch real smoke run: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-real-20260420T193617Z/`
- Latest corrected real smoke run: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-gridrecipe-20260420T2032Z/`

## Source, Access, And License

- Source URL: `https://openfwi-lanl.github.io/docs/data.html`
- Dataset: OpenFWI FlatVel-A.
- Required smoke shards: `data1.npy`, `model1.npy`, `data49.npy`, `model49.npy`.
- License note: OpenFWI datasets are CC BY-NC-SA 4.0; OpenFWI code is BSD-3-Clause.
- Access status: unblocked after staging the four required smoke shards under `/home/ollie/Documents/openfwi-flatvel-a`.
- Data-root policy: OpenFWI data remain outside git. This pass staged only the four-shard smoke subset, not the full 43 GB FlatVel-A bundle.
- Staged source: Hugging Face mirror `ashynf/OpenFWI`, FlatVel_A files, observed repo commit `9ed98af51d841bc7b7a43911021f87df6a730db5`.

## Shard Identity And Shape Validation

The smoke gate requires seismic shard shape `(500, 5, 1000, 70)` and velocity shard shape `(500, 1, 70, 70)` before any training. The latest real smoke run validated both train and held-out shard pairs against that contract.

The implemented real-run validator now treats the leading sample count as part of the FlatVel-A contract, not just the trailing tensor axes. Synthetic fixture runs must opt in explicitly with `--allow-synthetic-shard-samples`; those artifacts record `sample_count_contract: synthetic_fixture` and are not valid real FlatVel-A evidence.

Original controlled blocker, preserved as historical evidence:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/data_access_blocker.json`

Staged file checksums:

- `data1.npy`: `30a51807c9d3dcbc137a948f922f44c4be53369fcda537d54e1eb4da26f0e38f`
- `model1.npy`: `6c721820e0b351315c9d0788bb1b085d6dc3820fe4e3dd73399b1496e35c1015`
- `data49.npy`: `1975f9e042a1bafe5df09d7bb6fe467d25bb8912e0b2129d865c0f431f70afbd`
- `model49.npy`: `93d49930a714d6cc5af802c478803864ed9129d63251ca03a4b1093cd06ba1f5`

## Split, Normalization, And Preprocessing Contract

The implemented smoke contract uses seed `20260420`, `data1.npy`/`model1.npy` for train smoke, and `data49.npy`/`model49.npy` for validation/test smoke. Default caps are `32` train samples, `16` validation samples, and `16` test samples.

For local 2D adapters, raw seismic samples keep `input_raw` shape `(5, 1000, 70)` and are deterministically resized with bilinear interpolation to `(5, 70, 70)`. Velocity targets keep shape `(1, 70, 70)`. Normalization stats are computed from train split inputs and targets only.

This is the current multichannel non-ptychography example. The Hybrid ResNet smoke model handles it through a supervised real-channel adapter, not through the CDI `PtychoPINN_Lightning` path: the first `SpatialLifter` receives `in_channels=5`, the full Hybrid ResNet encoder-bottleneck-decoder body operates on hidden channels, and the final `Conv2d` emits `out_channels=1`. No probe, scan-position tensor, ptychographic channel grouping, or real/imag-to-complex conversion is applied.

The latest real smoke run generated a split manifest and train-split normalization stats for the staged shards. The selected caps were `32` train samples, `16` validation samples, and `16` test samples.

## Metric Contract

The metric writer reports MAE as primary and RMSE/SSIM as secondary. Metrics are computed on denormalized velocity maps when target normalization is used. SSIM uses a per-sample data range from the combined prediction/target range with a minimum range guard for constant maps.

## Official InversionNet Compatibility

Official InversionNet compatibility was not attempted against a real external checkout because `--official-openfwi-repo` was not supplied. The implemented probe accepts `--official-openfwi-repo`, records repository path, git commit, and license path, imports `network.py`, `model.py`, or `models.py` from the supplied checkout, resolves `InversionNet` from either a class or `model_dict`, and runs a bounded CPU forward-shape probe from `(1, 5, 1000, 70)` to `(1, 1, 70, 70)`. Missing checkout, import failure, missing `InversionNet`, or forward/shape failure is recorded as a controlled official-code blocker. OpenFWI official rows remain published-context only unless a later tranche reproduces the full official split and metrics.

## Local Model Results Or Blockers

The latest corrected real FlatVel-A smoke run completed both local profiles under one shared data, split, normalization, metric, and grid-lines training recipe contract:

| Profile | Status | Parameters | Test MAE | Test RMSE | Test SSIM | Runtime sec |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `hybrid_resnet_smoke` | metrics | 7756610 | 407.7264404296875 | 509.9523010253906 | 0.43761187679964575 | 0.9315609931945801 |
| `unet_smoke` | metrics | 4977 | 657.5835571289062 | 783.5233154296875 | 0.5981672306860322 | 0.1500227451324463 |

Interpretation boundary: this table is a smoke sanity/provenance table only. The comparison summary records `recommended_decision_input: smoke_contract_complete`, `evidence_scope: smoke_feasibility_only`, `metric_interpretation: sanity_only_not_benchmark_performance`, and `performance_assessment_complete: false`; it intentionally does not publish `hybrid_MAE`, `best_baseline_MAE`, or `relative_gap_vs_best_baseline` summary fields because those would imply benchmark-performance assessment.

## Runtime, Memory, And Provenance

- Historical blocked run ID: `openfwi-smoke-data-access-blocked-20260420`
- Historical one-epoch real smoke run ID: `openfwi-smoke-real-20260420T193617Z`
- Latest corrected real smoke run ID: `openfwi-smoke-gridrecipe-20260420T2032Z`
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
- Latest run logs:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-gridrecipe-20260420T2032Z/logs/smoke.stdout.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-gridrecipe-20260420T2032Z/logs/smoke.stderr.log`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-gridrecipe-20260420T2032Z/logs/smoke.exit_code`

## Published-Context Caveats

Published OpenFWI MAE/RMSE/SSIM rows for InversionNet, VelocityGAN, UPFWI, and InversionNet3D are not same-protocol evidence for this tranche. Any later paper use must label them as published context unless the full official split, preprocessing, model code or accepted reimplementation, and metric protocol are reproduced locally.

## Gate Checks

- Data access: complete for the four-shard smoke subset; full 43 GB FlatVel-A remains unstaged.
- Shape validation: complete for `data1.npy`/`model1.npy` and `data49.npy`/`model49.npy`.
- Split manifest: generated for real staged FlatVel-A smoke shards with seed `20260420`.
- Normalization: generated from the real staged train split only.
- Metrics: MAE, RMSE, and SSIM implemented and tested.
- Hybrid ResNet-compatible smoke: completed on real staged smoke shards.
- Local baseline smoke: U-Net completed on real staged smoke shards.
- Official InversionNet: metadata/import/forward probe implemented; blocked because no external OpenFWI checkout was supplied.
- Long-run guard: the CLI accepts only the approved prelaunch layout at the selected run root (`logs/smoke.run_id`, `logs/smoke.started_at_ns`, and the current launcher-owned `logs/smoke.pid`) before writing. It still rejects other live or incomplete `logs/smoke.pid` markers at the selected output root and nested `runs/*/logs/smoke.pid` markers.
- Reused-root freshness: once data resolution succeeds, obsolete `data_access_blocker.json` files are cleared before current artifacts are written. The comparison collator also validates blocker run IDs and ignores a stale data-access blocker when current-run metrics exist, preventing stale blockers from overriding successful current metrics.
- Freshness validation: latest corrected real smoke artifacts passed `validate_fresh_artifacts(...)` for run ID `openfwi-smoke-gridrecipe-20260420T2032Z`.
- Training recipe guard: future OpenFWI competitiveness runs must use the corrected inherited Hybrid ResNet recipe or record an explicit plan-level override before launch. Smoke metrics remain adapter/data-access/runtime sanity evidence only, regardless of epoch count.

## Raw Artifact Links

- Blocker: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/data_access_blocker.json`
- Historical one-epoch real-run comparison JSON: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-real-20260420T193617Z/comparison_summary.json`
- Latest corrected real-run comparison JSON: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-gridrecipe-20260420T2032Z/comparison_summary.json`
- Latest corrected real-run comparison CSV: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-gridrecipe-20260420T2032Z/comparison_summary.csv`
- Latest corrected real-run shape validation: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-openfwi-flatvel-a-fallback-smoke-gate/runs/openfwi-smoke-gridrecipe-20260420T2032Z/shard_shapes.json`

## Residual Risks

- The smoke subset is not the full official FlatVel-A benchmark and cannot support paper-facing OpenFWI claims or any model-performance decision by itself, even under the corrected grid-lines recipe.
- Official InversionNet baseline execution still needs an external checkout and a later approved same-protocol execution.
- The next OpenFWI step should be a separately planned longer execution tranche with the staged data policy, source caveats, corrected Hybrid ResNet recipe, local baseline recipe, and compute budget made explicit.

Decision: proceed to OpenFWI longer execution
