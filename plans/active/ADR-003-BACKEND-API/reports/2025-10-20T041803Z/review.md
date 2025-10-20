# ADR-003 Phase C4.C Review — 2025-10-20T041803Z

## Context
- Initiative: ADR-003-BACKEND-API
- Focus: Phase C4.C implementation follow-up (training + inference CLI refactor)
- Source inputs: commit `ce376dee`, `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_cli_{train,inference}_green.log`, plan.md §C4.C.

## Findings
- **Training CLI (C4.C1–C4.C3) complete:** `ptycho_torch/train.py` now defines the four execution config flags, delegates config construction to `create_training_payload()`, and threads `PyTorchExecutionConfig` into `main()` and downstream workflows. Targeted pytest log confirms 6/6 tests passing.
- **Hardcode documentation pending (C4.C4):** Implementation removed the legacy `nphotons`, `K`, and `experiment_name` overrides, but the required `refactor_notes.md` artefact was not authored. Marked task `[P]` in plan.
- **Inference CLI incomplete (C4.C6–C4.C7):** Although new flags exist, the CLI still bypasses `create_inference_payload()` and proceeds to load checkpoints/NPZ files directly. Patched tests therefore hit real filesystem checks and fail with `FileNotFoundError`. CONFIG-001 ordering remains violated because params.cfg is never bridged before IO.
- **Regression risk:** `data/memmap/meta.json` was mutated by the same commit (shape 34→238). This appears to be an artefact of a local CLI smoke run and should be reverted or relocated back under reports.

## Evidence References
- Training CLI implementation: `ptycho_torch/train.py:381-589`
- Inference CLI gap: `ptycho_torch/inference.py:360-540`
- Failing pytest log: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_cli_inference_green.log`
- Data memmap drift: `git show ce376dee:data/memmap/meta.json`

## Recommended Next Steps
1. **Complete C4.C4:** Author `refactor_notes.md` summarising removed training hardcodes per plan guidance.
2. **Execute C4.C6–C4.C7:** Refactor inference CLI to call `create_inference_payload()`, enforce CONFIG-001 sequencing, and adapt tests accordingly. Ensure patched pytest suite passes without touching real checkpoints.
3. **Validation (C4.D):** Re-run targeted selectors (`tests/torch/test_cli_{train,inference}_torch.py`) and refresh the GREEN logs once inference path is factory-driven.
4. **Hygiene:** Restore `data/memmap/meta.json` to canonical values or move derived artefacts under the appropriate report directory.
