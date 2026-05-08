# Execution Report

## Completed In This Pass

- Added explicit probe-channel conditioning support to the compare-wrapper,
  torch runner, data payload, config bridge, and the Hybrid ResNet / FFNO
  generator construction paths so two new opt-in CDI `lines128` rows can
  append fixed-probe real/imag channels to the learned input while leaving
  default rows unchanged.
- Added focused regressions covering:
  - wrapper row routing for the new model IDs
  - runner-side conditioning assembly and contract recording
  - learned-input-channel plumbing through config and Lightning factory rebuilds
  - unsupervised loss use of preserved observed diffraction when learned input
    differs from the forward-model target
- Ran the approved long-run ablation and recovered from ordinary
  implementation failures until the tracked wrapper PID exited `0` and both
  row-local proof files were present.
- Published the durable append-only summary and refreshed the human- and
  machine-readable discovery surfaces for the new conditioning family.

## Completed Plan Tasks

- Tranche 1 complete: authoritative baseline lineage, fixed contract, and
  allowed deltas were recorded in
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/verification/contract_note.json`.
- Tranche 2 complete: compare-wrapper and torch-runner plumbing now support
  `input_conditioning_mode=probe_real_imag` and
  `learned_input_channels=3` for the two new row IDs only.
- Tranche 3 complete: wrapper preflight confirmed only
  `pinn_hybrid_resnet_probe_channels` and `pinn_ffno_probe_channels` were
  selected, preserved the fixed-probe lineage, and kept the authoritative
  unconditioned Hybrid and corrected pure-FFNO rows as lineage anchors.
- Tranche 4 complete: the long run finished successfully under tmux, the root
  invocation reports `status=completed` and `exit_code=0`, and both fresh rows
  emitted config/history/metrics/model/recon/visual artifacts plus row-local
  completion proof files.
- Tranche 5 complete: the append-only summary, evidence matrix,
  `model_variant_index.json`, `ablation_index.json`, and `docs/studies/index.md`
  now record the new probe-conditioning family.

## Remaining Required Plan Tasks

- None.

## Verification

- Blocking prerequisite gate passed:
  - `python` presence check archived at
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/verification/prerequisite_check.log`
- Deterministic code gates passed:
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or probe"`
    -> `18 passed, 144 deselected`
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or probe"`
    -> `24 passed, 66 deselected, 4 warnings`
  - `python -m compileall -q ptycho_torch scripts/studies`
- Comparison standard for the contract audit:
  - exact equality on the fixed-contract fields listed in
    `verification/contract_note.json`
  - allowed row-level deltas limited to
    `input_conditioning_mode: diffraction_only -> probe_real_imag`,
    `learned_input_channels: 1 -> 3`, and retention of
    `fno_cnn_blocks = 0` for the corrected pure-FFNO-conditioned row
- Long-run completion proof:
  - root invocation:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/invocation.json`
    reports `status=completed`, `exit_code=0`, and
    `finished_at_utc=2026-05-08T21:54:17.205094+00:00`
  - tmux capture:
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-probe-channel-conditioning-ablation/verification/final_tmux_capture.log`
  - row-local proof files:
    `runs/pinn_hybrid_resnet_probe_channels/{launcher_completion.json,exit_code_proof.json}`
    and
    `runs/pinn_ffno_probe_channels/{launcher_completion.json,exit_code_proof.json}`
- Fresh outcome summary:
  - `pinn_hybrid_resnet_probe_channels` vs `pinn_hybrid_resnet`:
    amp/phase MAE `0.028848 / 0.131669` vs `0.026939 / 0.072063`,
    amp/phase SSIM `0.987296 / 0.961703` vs `0.988114 / 0.994740`,
    amp/phase FRC50 `132.825383 / 0.929580` vs `135.464222 / 106.800609`
  - `pinn_ffno_probe_channels` vs corrected pure `pinn_ffno`:
    amp/phase MAE `0.073340 / 3.154161` vs `0.082043 / 0.137965`,
    amp/phase SSIM `0.907495 / 0.012315` vs `0.890305 / 0.959644`,
    amp/phase FRC50 `57.675396 / 48.611318` vs `61.465010 / 58.728607`

## Residual Risks

- The resulting evidence is append-only only. Neither conditioned row earned
  promotion over its authoritative unconditioned anchor.
- This is still a single-seed CDI ablation with `2 / 2` train/test images, so
  the result supports same-contract directionality only.
- Historical parameter-count fields do not align cleanly enough with the fresh
  conditioned rows to support an efficiency interpretation in this item; the
  summary and indexes intentionally restrict themselves to contract and
  reconstruction evidence.
- FFNO evaluation emitted negative-SSIM clamp warnings during MS-SSIM
  computation, but the tracked PID exited `0` and all required artifacts were
  written.
