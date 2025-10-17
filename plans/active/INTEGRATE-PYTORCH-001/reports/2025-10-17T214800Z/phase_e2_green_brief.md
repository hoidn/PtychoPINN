# Phase E2.C/D Planning Brief — 2025-10-17T214800Z

## Purpose
Document supervisory planning for Phase E2 implementation (green phase) and parity verification after the red evidence loop (Attempt #79).

## Key Decisions
1. Authored detailed execution plan at `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` covering CLI wiring (E2.C) and parity evidence (E2.D).
2. Training CLI must mirror TensorFlow flags and call `run_cdi_example_torch` after CONFIG-001 bridge.
3. Inference CLI will load the Lightning checkpoint emitted by training and output amplitude/phase PNGs for parity validation.
4. Added explicit requirement to include `lightning` in the `[torch]` extras and expose `--disable_mlflow` flag to keep CI runs lightweight.
5. Established artifact expectations for green tests (`phase_e_backend_green.log`, `phase_e_integration_green.log`) and parity logs (`phase_e_tf_baseline.log`, `phase_e_torch_run.log`).

## Next Steps for Engineering Loop
- Execute Phase C checklist items C1–C5, storing results under a fresh ISO directory (recommended: `reports/<ISO8601>/phase_e2_green.md`).
- After green tests pass, progress to Phase D parity actions using commands listed in the plan and record metrics in `phase_e_parity_summary.md`.

## References
- Red-phase evidence: `reports/2025-10-17T213500Z/{phase_e_fixture_sync.md,red_phase.md,phase_e_red_integration.log}`
- Backend selector blueprint: `reports/2025-10-17T180500Z/phase_e_backend_design.md`
- Spec contract: `specs/ptychodus_api_spec.md` §4.5
- Policy alignment: `docs/findings.md#policy-001`

