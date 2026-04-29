# Completed In This Pass

- Re-audited the fixed-contract stable root at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`
  and confirmed the launched compare finished without needing a relaunch or any
  FFNO CDI-path code repair.
- Updated the durable preflight/summary/studies-index surfaces from
  in-progress to completed for this prerequisite `lines128` FFNO-versus-Hybrid
  CDI row pair.
- Wrote this execution report and the required implementation-state bundle for
  the completed pass.

# Completed Plan Tasks

- Task 1 completed: the stable root was reclassified from active-writer state
  to completed candidate after the tracked PIDs exited and the wrapper-level
  merged artifacts plus both row trees were present.
- Task 2 skipped by plan design: no FFNO CDI-contract failure remained once the
  original compare finished, so no code patch or regression-test addition was
  required in this pass.
- Task 3 completed with fresh evidence:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
    -> `161 passed, 43 warnings in 283.11s`
  - `python -m compileall -q ptycho_torch scripts/studies`
  - archived under
    `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/`
- Task 4 completed without relaunch: the fixed-contract compare root contains
  completed wrapper metrics/tables/visuals plus completed `pinn_hybrid_resnet`
  and `pinn_ffno` row outputs.
- Task 5 completed: the summary, preflight, and studies index now point to the
  same stable root and completed state.

# Remaining Required Plan Tasks

- None for the approved current-scope backlog item.
- The later four-row `lines128` paper benchmark remains separate follow-on work
  and is outside this completed prerequisite row-pair scope.

# Verification

- Fresh deterministic gates:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/20260429T032022Z_pytest.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/20260429T032022Z_compileall.log`
- Final wrapper metrics from
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet/metrics.json`
  show `pinn_hybrid_resnet` beating `pinn_ffno` on the reported amplitude and
  phase aggregates:
  - `pinn_hybrid_resnet`: amp/phase MAE `0.026939474 / 0.072063477`,
    amp/phase SSIM `0.988114297 / 0.994739987`, amp/phase PSNR
    `77.519370679 / 69.216286834`
  - `pinn_ffno`: amp/phase MAE `0.062772475 / 0.082838669`,
    amp/phase SSIM `0.934830340 / 0.981591519`, amp/phase PSNR
    `70.190080563 / 67.775916878`
- Final visual bundle present:
  - `visuals/compare_amp_phase.png`
  - `visuals/amp_phase_pinn_hybrid_resnet.png`
  - `visuals/amp_phase_pinn_ffno.png`

# Residual Risks

- The completed root has wrapper-level `invocation.json` / `invocation.sh`, but
  not separate per-row `runs/.../invocation.json` files; row provenance in this
  root is therefore carried by the wrapper invocation plus per-row
  `metrics.json`, `history.json`, and `randomness_contract.json`.
- The wrapper `invocation.json` fields `python_executable`, `git_commit`,
  `started_at`, and `finished_at` are null in this specific completed root, so
  later paper-facing packaging should cite the wrapper command and stable-root
  evidence directly rather than overclaiming missing runtime metadata.
