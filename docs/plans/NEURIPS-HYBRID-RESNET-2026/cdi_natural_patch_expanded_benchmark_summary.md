# NeurIPS CDI Natural-Patch Expanded Benchmark Summary

- Date: `2026-05-05`
- Backlog item: `2026-05-04-cdi-natural-patch-expanded-benchmark`
- State: `paper_complete_with_recovered_invocation_caveat`
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
- Recovered the final `paper_complete` benchmark bundle from the completed
  `natural-patch-benchmark-20260505T213458Z` row artifacts after the tracked
  launcher exited on post-processing bugs in the original harness. The recovered
  run root now contains `metrics.json`, `metric_schema.json`,
  `model_manifest.json`, `paper_benchmark_manifest.json`,
  `metrics_table.csv`, and `metrics_table.tex`.

## Accepted Six-Row Roster

- `baseline` -> `CDI CNN + supervised`
- `pinn` -> `CDI CNN + PINN`
- `pinn_hybrid_resnet` -> `Hybrid ResNet + PINN`
- `pinn_fno_vanilla` -> `FNO Vanilla + PINN`
- `pinn_ffno` -> `FFNO + PINN`
- `pinn_neuralop_uno` -> `U-NO + PINN`

## Benchmark Status And Boundary

- benchmark status: `paper_complete`
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

## Single-Seed Outcome Table

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
  exited `1`; the final `paper_complete` bundle was recovered afterward from
  the completed row artifacts in the same run root. Treat this as a recovered
  invocation caveat, not as a clean launcher proof.
- The recovered bundle does not yet backfill fixed-sample torch-row PNGs under
  `visuals/` even though the per-row metrics, configs, histories, and recons
  are durable.
- This is single-seed evidence on `natural_patches128_fixedprobe_v1` only. It
  should not be generalized into broader expanded-object CDI claims without
  additional seeds or a later approved contract extension.
