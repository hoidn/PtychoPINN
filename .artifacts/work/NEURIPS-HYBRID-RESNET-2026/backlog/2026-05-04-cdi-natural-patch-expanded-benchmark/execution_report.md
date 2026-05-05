# Execution Report

## Completed In This Pass

- Repaired the natural-patch benchmark harness so the locked fixed-probe
  dataset contract is consumable without mutating the dataset:
  `probe_manifest.json["pipeline_spec"]` now normalizes to the expected probe
  lineage string, and prepared-input reuse repairs stale
  `prepared_input_manifest.json` lineage metadata in place.
- Fixed live benchmark failures discovered during execution:
  patchwise metric collation now skips curve-valued outputs such as `frc`,
  TF PINN inference accepts prediction lists/tuples with extra tensors,
  TF paper-row payloads propagate `N=128`, and fixed-sample visual saving now
  accepts singleton channel-first patches.
- Completed the six-row expanded-object CDI launch under
  `runs/natural-patch-benchmark-20260505T213458Z` through row-local training,
  reconstruction, and metric emission for `baseline`, `pinn`,
  `pinn_hybrid_resnet`, `pinn_fno_vanilla`, `pinn_ffno`, and
  `pinn_neuralop_uno`.
- Recovered the final paper bundle from the completed row-local artifacts after
  the tracked tmux launcher exited during post-processing. The recovered run
  root now includes `metrics.json`, `metric_schema.json`,
  `model_manifest.json`, `paper_benchmark_manifest.json`,
  `metrics_table.csv`, `metrics_table.tex`, and `metrics_table_best.tex`.
- Published the durable summary and discoverability updates for this
  natural-patch expanded-object CDI lane without changing any `lines128`
  authority surface.

## Completed Plan Tasks

- Task 1: prerequisite presence gate, dataset-contract preflight, dry-run
  inspection, and prepared-input contract handling are complete.
- Task 2: narrow harness fixes and regression coverage for the execution gaps
  exposed by dry-run and live execution are complete.
- Task 3: the required six-row single-seed benchmark bundle is present at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/runs/natural-patch-benchmark-20260505T213458Z`
  with `benchmark_status="paper_complete"` after recovery.
- Task 4: summary and discoverability surfaces are updated:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`,
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`,
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`,
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`,
  `docs/studies/index.md`, and `docs/index.md`.

## Remaining Required Plan Tasks

- None for the approved single-pass publication scope.
- Optional follow-up only if stricter launcher-proof replacement is later
  required: rerun the benchmark end-to-end until the tracked tmux launcher exits
  `0` and all torch-row fixed-sample PNGs are emitted directly by the harness.

## Verification

- Required input gate passed:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/verification/required_input_gate_20260505T230936Z.log`
- Required benchmark tests passed:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/verification/pytest_selected_20260505T230936Z.log`
- Compile gate passed:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/verification/compileall_selected_20260505T230936Z.log`
- Integration verification passed:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/verification/pytest_integration_20260505T230936Z.log`
  (`5 passed, 4 skipped, 2198 deselected`).
- Recovery validation note written:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/verification/recovery_note_20260505T230936Z.md`
- Recovered bundle validation:
  `metrics.json` reports `benchmark_status="paper_complete"`,
  `paper_benchmark_manifest.json` records all six row statuses as completed,
  and `model_manifest.json` records `N=128` for every row.

## Residual Risks

- The tracked tmux launcher for
  `natural-patch-benchmark-20260505T213458Z` exited `1`; the final bundle is a
  recovered publication from completed row-local artifacts in that same run
  root, not a clean launcher-proof completion.
- Torch-row fixed-sample PNGs were not backfilled into `visuals/` during
  recovery, although row-local recons, configs, histories, and metrics are
  durable.
- This remains single-seed evidence on
  `natural_patches128_fixedprobe_v1`; it widens expanded-object CDI evidence but
  does not replace or supersede the `lines128` authority.
