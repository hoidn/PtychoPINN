# NeurIPS CDI Natural-Patch Expanded Benchmark Summary

- Date: `2026-05-05`
- Backlog item: `2026-05-04-cdi-natural-patch-expanded-benchmark`
- State: `benchmark_incomplete_with_recovered_metrics_only`
- Dataset id: `natural_patches128_fixedprobe_v1`
- Authoritative root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/runs/natural-patch-benchmark-20260505T213458Z`
- Fixed seed policy: `seed=3`
- Fixed sample ids: `0`, `500`, `999`

## Completed In This Pass

- Repaired five natural-patch harness defects discovered during the live benchmark pass:
  - probe lineage normalization now accepts `probe_manifest.json["pipeline_spec"]`
    when `canonical_pipeline` is absent
  - prepared-input reuse repairs stale `probe_lineage` metadata in place instead
    of reusing broken manifests
  - patchwise metric collation now skips curve-valued outputs such as `frc`
    instead of coercing them to scalars
  - TF PINN inference now accepts model predictions with more than three output
    tensors and still uses the first tensor as the reconstructed object
  - TF paper-row payloads now propagate `N` instead of emitting `None`
- Repaired the fixed-sample visual saver so singleton channel-first patch shapes
  like `(1, 128, 128)` no longer crash per-row post-processing.
- Recovered a metrics-only benchmark bundle from the completed
  `natural-patch-benchmark-20260505T213458Z` row artifacts after the tracked
  launcher exited on post-processing bugs in the original harness. The recovered
  run root contains `metrics.json`, `metric_schema.json`,
  `model_manifest.json`, `paper_benchmark_manifest.json`,
  `metrics_table.csv`, and `metrics_table.tex`. The `metrics.json` is now
  classified `benchmark_incomplete` because the launcher exit was non-zero,
  required provenance fields (`randomness`, full `dataset`/`splits` manifests,
  `outputs.exit_code_proof_json`, host/torch environment, dirty-state git note)
  are not yet emitted by this harness, and torch-row fixed-sample visuals were
  never backfilled into `visuals/`.
- Patched `run_natural_patch_benchmark` to call `write_paper_benchmark_bundle`
  with `require_row_provenance=True` and the executor's `row_statuses`, so any
  future natural-patch run that lacks the locked provenance scaffolding is
  classified `benchmark_incomplete` instead of being silently promoted to
  `paper_complete`.
- Added benchmark-mode regression tests asserting that incomplete row payloads
  and blocked/failed launcher outcomes downgrade the bundle to
  `benchmark_incomplete` with non-empty `missing_fields_by_row`.

## Accepted Six-Row Roster

- `baseline` -> `CDI CNN + supervised`
- `pinn` -> `CDI CNN + PINN`
- `pinn_hybrid_resnet` -> `Hybrid ResNet + PINN`
- `pinn_fno_vanilla` -> `FNO Vanilla + PINN`
- `pinn_ffno` -> `FFNO + PINN`
- `pinn_neuralop_uno` -> `U-NO + PINN`

## Benchmark Status And Boundary

- benchmark status: `benchmark_incomplete` (recovered metrics-only bundle; the
  authoritative tracked launcher exited `1`, torch-row fixed-sample visuals are
  missing, and the harness does not yet emit the locked provenance scaffolding
  required for `paper_complete` under `require_row_provenance=True`)
- claim boundary: `single_seed_natural_patch_expanded_object_cdi_only`
- evidence scope: one coherent six-row expanded-object CDI bundle on the frozen
  `natural_patches128_fixedprobe_v1` dataset under a fixed `seed=3` contract
- preserved authority distinction:
  - this is separate expanded-object CDI evidence on natural-image-derived
    object patches
  - it does **not** replace the existing `lines128` paper-table authority in
    `lines128_paper_benchmark_summary.md`
  - it should be read as single-seed expanded-object follow-up evidence, not as
    a same-contract substitute for the synthetic grid-lines table

## Single-Seed Outcome Table (advisory only)

> The metrics below come from the row-local artifacts of the recovered run. They
> are advisory only; the bundle is `benchmark_incomplete`, so these numbers are
> not authoritative for paper claims until a clean launcher-proof rerun is
> published with full provenance scaffolding.



| Row | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Amp FRC50 | Phase FRC50 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline` | 0.0716 | 0.3954 | 0.4864 | 0.6456 | 0.7902 | 0.7817 |
| `pinn` | 0.2923 | 1.4472 | 0.2440 | 0.2366 | 0.8565 | 0.8349 |
| `pinn_hybrid_resnet` | 0.2609 | 0.4374 | 0.0275 | 0.4425 | 0.8621 | 0.9395 |
| `pinn_fno_vanilla` | 0.1571 | 0.4237 | 0.0420 | 0.5307 | 0.8300 | 0.9338 |
| `pinn_ffno` | 0.1567 | 0.3961 | 0.0594 | 0.6041 | 0.8581 | 0.8339 |
| `pinn_neuralop_uno` | 0.1708 | 0.3997 | 0.0517 | 0.5981 | 0.8429 | 0.8793 |

## Current Read

- The natural-patch expanded-object ranking does **not** mirror the synthetic
  `lines128` headline table. On this single-seed natural-image-derived object
  contract, the supervised CNN baseline is the strongest row by amplitude MAE,
  phase MAE, amplitude SSIM, and phase SSIM.
- The learned operator rows contribute different strengths than the baseline:
  `pinn_hybrid_resnet` and `pinn_fno_vanilla` lead phase-side FRC50, while
  `pinn_hybrid_resnet` also posts the best amplitude FRC50.
- `pinn_ffno` and `pinn_neuralop_uno` are the closest learned rows to the
  baseline on phase MAE, but neither recovers the baseline's amplitude
  fidelity on this contract.
- The local CNN PINN row is the weakest outcome on this benchmark, especially
  on phase MAE/SSIM, so this run does not support promoting the plain PINN row
  for expanded-object CDI claims.

## Table And Artifact Outputs

- bundle outputs at the authoritative root:
  - `metrics.json`
  - `metric_schema.json`
  - `model_manifest.json`
  - `paper_benchmark_manifest.json`
  - `metrics_table.csv`
  - `metrics_table.tex`
  - `metrics_table_best.tex`
- row-local artifacts:
  - `runs/<row>/metrics.json`
  - `runs/<row>/history.json`
  - `runs/<row>/config.json`
  - `recons/<row>/recon.npz`
- durable visuals currently present:
  - baseline/PINN fixed-sample PNGs under `visuals/`
  - ground-truth-fixed shared-scale policy remains locked by
    `contract/fixed_sample_manifest.json` and `contract/shared_visual_scales.json`

## Verification

- required dataset/benchmark unit tests:
  - `pytest -q tests/studies/test_cdi_natural_patch_dataset.py tests/studies/test_cdi_natural_patch_benchmark.py`
- focused workflow regressions for live defects fixed during this pass:
  - `pytest -q tests/test_grid_lines_workflow.py -k 'first_prediction_output or build_tf_row_payload_uses_emitted_validation_loss'`
- compile gate:
  - `python -m compileall -q scripts/studies ptycho_torch ptycho/workflows`
- recovered authoritative bundle root:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/runs/natural-patch-benchmark-20260505T213458Z`

## Residual Risks

- The tracked tmux launcher for `natural-patch-benchmark-20260505T213458Z`
  exited `1`; the recovered bundle is now classified `benchmark_incomplete`,
  which is the honest reflection of that launcher proof gap.
- The recovered bundle does not backfill fixed-sample torch-row PNGs under
  `visuals/`, so torch rows will continue to fail the `visuals` provenance
  validator until the harness is rerun cleanly.
- The harness does not yet construct the full provenance scaffolding required
  by `require_row_provenance=True` (host/torch environment, git
  `dirty_state_note`, dataset and splits `manifest_json` payloads with sha256
  records, `randomness`, and `outputs.exit_code_proof_json`). This is a
  follow-up work item on top of the launcher rerun.
- This is single-seed evidence on `natural_patches128_fixedprobe_v1` only. It
  should not be generalized into broader expanded-object CDI claims without
  additional seeds or a later approved contract extension.

## Follow-Up Work

- Relaunch the natural-patch benchmark end-to-end until the tracked PID exits
  `0` and the harness emits torch-row fixed-sample PNGs directly.
- Extend `cdi_natural_patch_benchmark.py` to construct the full provenance
  payloads expected by `metrics_tables.py` (`environment.host`,
  `environment.torch_version`, `environment.cuda_version`, `environment.gpu`,
  `git.dirty_state_note`, dataset/splits `manifest_json` records with size and
  sha256, `randomness`, and `outputs.exit_code_proof_json`) so a clean rerun
  can legitimately reach `paper_complete` under the new validation.
