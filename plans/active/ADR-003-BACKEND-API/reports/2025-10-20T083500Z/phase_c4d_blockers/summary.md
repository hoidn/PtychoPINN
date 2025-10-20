# ADR-003 Phase C4.D Blockers — Planning Summary (2025-10-20T083500Z)

## Snapshot
- **Focus:** Resolve remaining Phase C4.D blockers preventing full PyTorch workflow integration.
- **Evidence Reviewed:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/{pytest_integration.log,manual_cli_smoke.log,summary.md}`.
- **Key Issues:**
  1. `ptycho_torch/model_manager.py::load_torch_bundle` still raises `NotImplementedError`, halting inference CLI (`test_run_pytorch_train_save_load_infer`).
  2. Manual CLI smoke encounters channel-count RuntimeError (`conv1` expecting 1 channel, received 4) despite factory reporting `gridsize=(2, 2)`, signalling a model/config alignment gap.

## Plan Highlights
- **Phase A — Bundle Loader Enablement:** TDD cycle for `load_torch_bundle`, including a new pytest selector exercising bundle reconstruction and ensuring inference proceeds past bundle loading.
- **Phase B — Training Channel Parity:** Capture regression covering `gridsize` handling, patch factories/model so Lightning uses `gridsize**2` channels, and validate via manual CLI smoke.
- **Phase C — Close-Out:** Update C4 plan rows, record summaries, and log fix_plan Attempt #29 once Phases A/B succeed.

## Dependencies & References
- `specs/ptychodus_api_spec.md` §4.6–§4.8 (bundle lifecycle & CLI requirements)
- `docs/workflows/pytorch.md` §§5–7 (workflow orchestration & persistence)
- Findings: `CONFIG-001` (params.cfg bridge ordering), `BUG-TF-001` (gridsize synchronization)
- Prior artefacts: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T060955Z/phase_c4_cli_integration_debug/triage.md`, `c4_d3_dataloader_fix_summary.md`

## Reporting Discipline
- New logs/tests live under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/`.
- Use `{selector}_{red|green}.log` naming; manual CLI stdout captured as `manual_cli_smoke_green.log`.
- After implementation, ensure `docs/fix_plan.md` references this plan & final artefacts.
