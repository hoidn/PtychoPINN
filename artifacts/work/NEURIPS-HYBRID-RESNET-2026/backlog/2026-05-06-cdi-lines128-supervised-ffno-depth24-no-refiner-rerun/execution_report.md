# Execution Report

## Completed In This Pass

- Added default compare-wrapper support for the new `supervised_ffno_depth24`
  row id while preserving `supervised_ffno` as the corrected four-block
  no-refiner authority.
- Added focused regression coverage proving the new row is registered,
  preflight-resolvable, and depth-overridden to `fno_blocks=24` with
  `fno_cnn_blocks=0`.
- Ran the required deterministic gates, archived their logs under the item-local
  verification root, and recorded a supported-row preflight manifest for the
  new supervised depth-24 row.
- Launched exactly one tmux-owned supervised depth-24 rerun under the locked
  `lines128` contract, tracked the exact shell-owned PID, and waited through a
  clean `__EXIT_CODE__=0`.
- Verified the required row-local outputs, generated the machine-readable
  depth-only comparison payloads plus same-contract audit, wrote the durable
  summary, and refreshed the required discoverability surfaces.

## Completed Plan Tasks

- Task 1: Freeze Authorities And Run The Input Presence Gate
- Task 2: Add Minimal `supervised_ffno_depth24` Row Plumbing
- Task 3: Launch Exactly One Fresh Supervised Depth-24 No-Refiner Run
- Task 4: Audit Same-Contract Fairness And Depth-Only Delta
- Task 5: Write The Durable Summary And Refresh Discoverability

## Remaining Required Plan Tasks

- None.

## Verification

- Presence gate passed and was archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/input_presence_gate.log`.
- Required deterministic code gates passed:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
  - `python -m compileall -q ptycho_torch scripts/studies`
- JSON discovery surfaces validated:
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json > /dev/null`
  - `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json > /dev/null`
- Long-run completion proof:
  - tmux pane capture archived at
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/tmux_completion_capture.log`
  - shell exit marker: `__EXIT_CODE__=0`
  - wrapper invocation:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/runs/supervised_ffno_depth24_20260507T192840Z/invocation.json`
  - row-local proofs:
    `runs/supervised_ffno_depth24/launcher_completion.json` and
    `runs/supervised_ffno_depth24/exit_code_proof.json`
- Artifact presence gate passed for the fresh row-local outputs:
  - `runs/supervised_ffno_depth24/{invocation.json,config.json,history.json,metrics.json,model.pt,exit_code_proof.json,launcher_completion.json,stdout.log,stderr.log}`
  - `recons/supervised_ffno_depth24/recon.npz`
  - `visuals/amp_phase_supervised_ffno_depth24.png`
  - `visuals/amp_phase_error_supervised_ffno_depth24.png`
- Same-contract audit passed using exact equality on locked config/invocation
  fields and exact train/test NPZ equality after normalizing only
  `_metadata.creation_info.timestamp`:
  - no numerical tolerance was used (`atol=0`, `rtol=0` in practice)
  - audit artifact:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/contract_audit_supervised_depth24_vs_depth4.json`
- Machine-readable comparison payloads were generated at:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/comparison_supervised_depth24_vs_depth4.json`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/verification/comparison_supervised_depth24_vs_depth4.csv`

## Residual Risks

- This remains a single-seed, two-image-per-split CDI follow-up; it is useful
  for same-contract directionality, not broad statistical claims.
- The supervised depth-24 row improves several phase-side metrics and amplitude
  MAE versus the corrected four-block supervised row, but amplitude SSIM/FRC
  regress slightly, validation loss is worse, and runtime grows by more than
  `5x`; any paper-facing promotion still needs the later final-refresh
  judgment.
- The durable summary narrows interpretation to
  `cdi_supervised_ffno_depth_companion_only`, while the wrapper run itself keeps
  the generic row status `decision_support_append_only`.
