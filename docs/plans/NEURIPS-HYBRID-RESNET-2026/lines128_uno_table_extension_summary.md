# NeurIPS Lines128 U-NO Table Extension Summary

- Date: `2026-05-04`
- Backlog item: `2026-04-30-cdi-lines128-uno-table-extension`
- State: `paper_complete`
- Claim boundary: `complete_lines128_cdi_benchmark_plus_uno_extension`
- Authoritative extension root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/runs/complete_table_plus_uno_20260504T100347Z`
- Immutable base authority preserved separately:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`
- Selected FNO comparator: `fno_vanilla`
- Fixed seed policy: `seed=3`
- Fixed sample ids: `0`, `1`

## Completed In This Pass

- Added a new `extend_with_uno` mode to `scripts/studies/lines128_paper_benchmark.py`
  that promotes the immutable six base rows by lineage and freshly launches only
  the two U-NO rows under the locked `lines128` CDI contract. The mode includes
  a strict execution-manifest validator (`_normalize_uno_extension_execution_manifest`)
  that enforces eight rows in canonical order, a single `base_complete_table_root`,
  the U-NO rows being the only `rerun_required` entries, and `architecture_id=neuralop_uno`
  for both fresh rows.
- Taught `scripts/studies/grid_lines_compare_wrapper.py` to route `pinn_neuralop_uno`
  and `supervised_neuralop_uno` through the existing Torch runner via new entries
  in `DEFAULT_TORCH_ROW_SPECS`, `TORCH_MODEL_IDS`, `PAPER_MODEL_LABELS`, and
  `PAPER_TRAINING_PROCEDURE_OVERRIDES`.
- Added a row-local provenance helper that emits `base_row_lineage.json` recording
  the immutable base root, both promoted base rows and freshly launched U-NO rows,
  and the canonical six-row claim boundary recovered from the base bundle.
- Narrow runtime fix: the U-NO upsampling stack uses `upsample_bicubic2d` which
  lacks a deterministic CUDA backward implementation, so the runner switches to
  `deterministic="warn"` mode only for `architecture=neuralop_uno`. All other
  architectures keep strict `deterministic=True`. Added focused regressions in
  `tests/torch/test_grid_lines_torch_runner.py` covering both modes.
- Added focused tests in `tests/studies/test_lines128_paper_benchmark.py` and
  `tests/test_grid_lines_compare_wrapper.py` covering the manifest validator,
  CLI mode plumbing, full eight-row promote-plus-fresh dispatch, lineage payload,
  base-root immutability, and `_torch_model_route` resolution for both U-NO rows.
- Launched only `pinn_neuralop_uno` and `supervised_neuralop_uno` fresh in tmux
  under `ptycho311`. Tracked the exact launched PID; the launcher exited cleanly
  (`exit_code=0`). Both rows produced row-local invocation/config/history/metrics/
  reconstruction artifacts plus `exit_code_proof.json`.
- Collated the eight-row extended bundle with merged metrics, manifests, and
  fixed-sample visuals; recorded base-row lineage and fresh U-NO row provenance.

## Final Eight-Row Roster

Promoted from the immutable base authority (no rerun, no overwrite):

- `baseline` -> `CDI CNN + supervised`
- `pinn` -> `CDI CNN + PINN`
- `pinn_hybrid_resnet` -> `Hybrid ResNet + PINN`
- `pinn_fno_vanilla` -> `FNO Vanilla + PINN`
- `pinn_spectral_resnet_bottleneck_net` -> `Spectral ResNet Bottleneck + PINN`
- `pinn_ffno` -> `FFNO + PINN`

Freshly launched in this item:

- `pinn_neuralop_uno` -> `U-NO + PINN` (architecture `neuralop_uno`)
- `supervised_neuralop_uno` -> `U-NO + supervised` (architecture `neuralop_uno`)

## Fresh U-NO Row Metrics

Reconstruction metrics under the locked `lines128` CDI contract,
seed `3`, fixed visual sample ids `0` and `1`:

| Row | Architecture | Training | Amp MAE | Phase MAE | Amp SSIM | Phase SSIM | Amp PSNR | Phase PSNR | Params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `pinn_neuralop_uno` | `neuralop_uno` | PINN | 0.093164 | 0.068291 | 0.827995 | 0.956859 | 66.918 | 69.470 | 2,515,718 |
| `supervised_neuralop_uno` | `neuralop_uno` | supervised | 0.320684 | 0.056251 | 0.268940 | 0.910490 | 55.985 | 71.351 | 2,515,716 |

These rows share the locked U-NO body (`uno_out_channels=[32, 64, 64, 32]`,
`uno_n_modes=[[12, 12], [12, 12], [12, 12], [12, 12]]`,
`uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2]]`, `hidden_channels=32`,
`lifting_channels=projection_channels=128`) and differ only by training procedure.

## Base Row Lineage

The promoted six rows in this bundle reproduce the immutable base authority bit-for-bit:

- `base_row_lineage.json` records the base `claim_boundary=complete_lines128_cdi_benchmark`
  alongside this extension's `complete_lines128_cdi_benchmark_plus_uno_extension`.
- A side-by-side stability audit at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/verification/base_row_metric_stability_audit.json`
  confirms exact `mae` parity for amp and phase across all six promoted rows.
- The base authority root's mtimes (`metrics.json`, `paper_benchmark_manifest.json`,
  per-row metrics) remain at `2026-04-30`; nothing in this item overwrote them.

## Fresh U-NO Row Provenance

Both `pinn_neuralop_uno` and `supervised_neuralop_uno` row roots contain:

- `invocation.json` / `invocation.sh` (runner-local invocation, including
  `extra.runtime_provenance` with `python_executable`, `python_version`, and a
  `torch` block (`version`, `cuda_version`, `cuda_available`, `device_name`),
  `extra.neuralop_provenance` with the `neuraloperator` package version,
  `neuralop.__version__`, and the live `neuralop.models.UNO` signature, plus
  `extra.git_commit` and `extra.git_dirty`)
- `config.json`, `history.json`, `metrics.json`
- `randomness_contract.json` capturing the seed contract together with the
  Lightning `deterministic_mode` (`"warn"` for U-NO) and the
  `deterministic_carve_out` rationale that scopes the relaxation to
  `architecture=neuralop_uno`
- `exit_code_proof.json`
- `launcher_completion.json` referencing the `Saved artifacts to ...` and
  `Torch runner complete. Artifacts in ...` row-completion markers from the
  in-process launcher transcript (the bundle root's `live_launch.log`)
- `model.pt` (Lightning checkpoint)
- per-row `stdout.log` and `stderr.log`
- recon at `recons/<row>/recon.npz`
- visuals at `visuals/amp_phase_<row>.png` and `visuals/amp_phase_error_<row>.png`

The U-NO row provenance fields above were retroactively populated by
`scripts/studies/lines128_uno_provenance_backfill.py` after the launch
completed, because the original runner did not capture the U-NO-specific
package/UNO-signature provenance or the deterministic-mode carve-out. The
backfill ran inside the same `ptycho311` environment that produced the rows;
each backfilled artifact records `provenance_backfilled_at_utc` (or
`backfilled_at_utc` for the randomness contract) plus a backfill rationale
field. Future fresh U-NO runs capture these fields automatically through the
extended `capture_runtime_provenance`/`capture_neuralop_provenance` helpers
in `scripts/studies/invocation_logging.py` and the extended
`_build_randomness_contract` in `scripts/studies/grid_lines_torch_runner.py`.

The fresh-row `launcher_completion.json` artifacts were retroactively emitted
during a follow-up review pass (using the existing `live_launch.log`
in-process launcher transcript) and the bundle's `metrics.json` /
`paper_benchmark_manifest.json` were rewritten in-place to mirror the new
`launcher_completion_json` pointers under each fresh row's `outputs` /
`artifacts`. Future fresh U-NO runs emit the artifact automatically through
`_finalize_uno_extension_launcher_completion` during collation in
`scripts/studies/lines128_paper_benchmark.py`, and the extension bundle gate
in `_collect_missing_fresh_row_launcher_completion_artifacts` downgrades the
bundle to `benchmark_incomplete` when the proof is missing.

The bundle's `paper_benchmark_manifest.json` records the runtime environment
(`python 3.11.13`, `torch 2.9.1+cu128`, `CUDA 12.8`, `NVIDIA RTX 3090`),
`neuraloperator==2.0.0` exposure verified by the live ptycho311 gate, and the
git commit captured at wrapper invocation time.

## Verification

- Required input presence:
  ```
  python - <<'PY'
  from pathlib import Path
  required = [
      Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_design.md"),
      Path(".artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/metrics.json"),
  ]
  missing = [str(p) for p in required if not p.exists()]
  if missing:
      raise SystemExit(f"missing U-NO table-extension inputs: {missing}")
  print("U-NO table-extension inputs present")
  PY
  ```
  log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/verification/required_inputs_check.log`

- Required deterministic gate:
  - `pytest -q tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py`
  - `206 passed, 51 warnings`
  - log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/verification/pytest_compare_runner.log`

- Lines128 paper-benchmark suite:
  - `pytest -q tests/studies/test_lines128_paper_benchmark.py`
  - `32 passed`
  - log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/verification/pytest_lines128_paper_benchmark.log`

- Compile gate:
  - `python -m compileall -q ptycho_torch scripts/studies` (exit `0`)
  - log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/verification/compileall.log`

- Live ptycho311 `neuraloperator==2.0.0` / `neuralop.models.UNO` gate:
  - log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/verification/ptycho311_neuralop_gate.log`

- Base-row metric stability audit:
  - all six promoted rows match the immutable base on `mae` (amp+phase)
  - log: `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-table-extension/verification/base_row_metric_stability_audit.json`

- Long-run launch:
  - tracked PID written to `runs/complete_table_plus_uno_20260504T100347Z/launcher_pid.txt`
  - exit code `0` written to `runs/complete_table_plus_uno_20260504T100347Z/launcher_exit_code.txt`
  - tmux session: `lines128-uno-20260504T100347Z`

## Outcome And Claim Boundary

- The extension's claim boundary is `complete_lines128_cdi_benchmark_plus_uno_extension`.
- The original six-row authority retains its `complete_lines128_cdi_benchmark`
  boundary; this item neither rewrites nor displaces that authority.
- Under the locked `lines128` CDI contract, the PINN U-NO row is competitive on
  phase metrics (phase MAE `0.0683`, phase SSIM `0.957`) but does not displace
  the existing `pinn_hybrid_resnet` or `pinn_spectral_resnet_bottleneck_net`
  anchors on amplitude. The supervised U-NO row trains stably to a meaningful
  phase result but its amplitude reconstruction is much weaker (amp MAE `0.32`,
  amp SSIM `0.27`), echoing the supervised-FFNO pattern recorded in
  `lines128_supervised_equivalent_rows_summary.md`.
- These rows are append-only paper-supporting comparators; ranking statements must
  cite the extended root and distinguish it from the original six-row authority.

## Remaining Caveats

- `pinn_neuralop_uno` and `supervised_neuralop_uno` both run with Lightning
  `deterministic="warn"` because `upsample_bicubic2d` has no deterministic CUDA
  backward implementation. This relaxation is recorded in row provenance
  (`randomness_contract.json`) and is scoped to the U-NO architecture only.
- Promoted base-row provenance is inherited from the immutable base bundle and
  retains the FFNO recovered-invocation caveat already documented in
  `lines128_paper_benchmark_summary.md`.
