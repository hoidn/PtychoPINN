## Completed In This Pass

- implemented the CDI bridge-study surfaces needed by the approved plan:
  row-matrix helpers, a dedicated runbook, row-spec-aware wrapper plumbing,
  override-aware Torch runner support, and the two new generator entries
  `spectral_resnet_bottleneck_linear_decoder` and
  `hybrid_resnet_ffno_bottleneck`
- added focused study and runner coverage for the frozen row roster, including
  a regression that fails unless row-local `model_id_override` controls the
  recon output path
- generated the checked-in preflight note plus machine-readable authorities
  `preflight/study_matrix.json` and `preflight/reference_runs.json`
- launched the fresh-row study under tmux, hit a recoverable recon-path bug on
  the first pass, repaired the runner, restored the reused spectral-anchor
  recon from a clean authoritative sibling root, resumed the remaining rows,
  and completed the final bundle with `analysis/bundle_validation.json`
  reporting `ok: true`
- wrote the durable CDI study summary and updated the repo discoverability and
  evidence indexes to register the new decision-support output

## Completed Plan Tasks

- reconstructed the fixed CDI contract and wrote the checked-in preflight note
- froze the exact reused-anchor and fresh-bridge matrix in checked-in Markdown
  plus machine-readable manifests
- extended the wrapper and runner so reused anchors and fresh same-base rows
  can coexist with distinct row IDs and row-local override payloads
- implemented the three frozen bridge rows exactly as authorized:
  `pinn_spectral_resnet_bottleneck_ds1`,
  `pinn_spectral_resnet_bottleneck_linear_decoder`, and
  `pinn_hybrid_resnet_ffno_bottleneck`
- added focused tests for matrix generation, row routing, registry plumbing,
  and the recon-path override regression
- ran the required deterministic checks before and after the expensive study
  launch
- completed the tmux-backed fresh-row execution and final anchored bundle
  collation
- produced the durable summary plus evidence/discoverability updates required
  by the roadmap gate

## Remaining Required Plan Tasks

- none

## Verification

- `pytest -q tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_writes_recon_artifact_under_model_id_override`
  - result: `1 passed, 2 warnings in 5.69s`
- `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - result: `191 passed, 49 warnings in 305.67s (0:05:05)`
- `python -m compileall -q ptycho_torch scripts/studies`
  - result: exit `0`
- `pytest -v -m integration`
  - result: `5 passed, 4 skipped, 1800 deselected, 2 warnings in 302.76s (0:05:02)`
- deterministic artifact validation:
  - `verification/artifact_validation.log` confirms that every frozen row has
    `invocation.json`, `config.json`, `history.json`, `metrics.json`,
    `exit_code_proof.json`, and `recons/<model_id>/recon.npz`
  - `analysis/bundle_validation.json` reports `"ok": true`
- archived launcher evidence:
  - `logs/launcher_first_attempt.log`
  - `logs/launcher_resume.log`
  - the resumed launcher ends with `__EXIT__:0`

## Residual Risks

- the study remains decision-support-only. No fresh bridge row is promoted into
  the current paper-grade CDI claim surface.
- the root-level `checkpoints/` and `lightning_logs/` directories are shared
  study-root transients rather than row-local durable provenance units; this is
  acceptable for the current harness but should remain out of any paper-facing
  provenance contract.
- the known non-fatal warning set remains present in verification:
  `tight_layout`, `skimage` SSIM/MS-SSIM warnings on degenerate cases, FRC
  divide warnings, and the pre-existing TensorFlow Addons compatibility
  warnings during `pytest -m integration`.
- `/home/ollie/Documents/neurips` was not present in this environment, so no
  manuscript-side evidence-map update was possible from this pass.
