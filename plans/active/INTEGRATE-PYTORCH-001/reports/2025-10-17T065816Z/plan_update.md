# Phase C Planning Update — 2025-10-17T065816Z

## Summary
- Authored `plans/active/INTEGRATE-PYTORCH-001/phase_c_data_pipeline.md` to break Phase C into phased tasks (C.A–C.E) covering data contract alignment, torch-optional test harness, implementation, validation, and documentation hand-off.
- Updated `plans/active/INTEGRATE-PYTORCH-001/implementation.md` Phase C guidance to reference the new plan and clarify deliverables for rows C1–C5.
- Established reporting conventions (ISO timestamped directories) and torch-optional testing requirements for upcoming data pipeline work.

## Key Decisions & References
- **Data contract first:** Require documentation of TensorFlow expectations before touching tests (`specs/data_contracts.md`, `specs/ptychodus_api_spec.md §4`).
- **Torch-optional tests:** New pytest module must guard imports to keep CI compatible when torch unavailable.
- **RawData reuse:** Implementation tasks insist on delegating to `ptycho.raw_data.RawData` to preserve existing grouping semantics and cache reuse.

## Next Steps
1. Execute Phase C.A tasks — capture `data_contract.md` and `torch_gap_matrix.md` artifacts under a fresh timestamped directory.
2. Blueprint torch-optional pytest module per Phase C.B and record red-phase logs.
3. Use new plan checkpoints to drive docs/fix_plan.md updates and `input.md` directives.
