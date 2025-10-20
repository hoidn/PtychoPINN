Summary: Author inference CLI thin-wrapper blueprint (Phase D.C C1).
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase D.C (Inference CLI thin wrapper)
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/{inference_refactor.md,summary.md}

Do Now:
1. ADR-003-BACKEND-API C1 (inference blueprint) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:46 — Draft `inference_refactor.md` capturing helper layout, RawData ownership, warning/deprecation plan, CONFIG-001 enforcement, and call flow; tests: none
2. ADR-003-BACKEND-API C1 (artifact + plan sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:46 — Write `summary.md` for the blueprint, update plan row C1 to `[x]`, and note completion in docs/fix_plan attempts; tests: none

If Blocked: Document the blocker in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/blocker.md`, leave plan row C1 `[ ]`, and record the stall in docs/fix_plan.md before pausing.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:46 — Phase D.C kicks off with the inference blueprint; downstream tasks rely on this design.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/summary.md — Updated next steps point to C1–C3 work; blueprint must exist before RED tests.
- specs/ptychodus_api_spec.md:200 — CLI contracts require consistent backend routing and CONFIG-001 ordering; blueprint needs to map to spec.
- docs/workflows/pytorch.md:344 — Helper-based flow plus deprecated flag messaging must stay aligned when inference path is refactored.
- tests/torch/test_cli_inference_torch.py:1 — Existing GREEN tests show current behaviour; blueprint should call out future test adjustments.

How-To Map:
- `mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference`
- Model `inference_refactor.md` after `phase_d_cli_wrappers_training/training_refactor.md`: include sections for Context, Goals, Helper/Module layout, CLI argument handling, delegation flow, legacy flag behaviour, RawData ownership, CONFIG-001 enforcement, test strategy (RED→GREEN), and risks.
- Reference `ptycho_torch/inference.py` and training blueprint to map existing call graph vs desired helper usage; cite file:line anchors.
- Outline planned helper touchpoints (reuse `ptycho_torch/cli/shared.py` or extend it) and describe how bundle loading + reassembly should route through workflows.
- In `summary.md`, capture key decisions, open questions, and artifact pointers; update plan row C1 to `[x]` with completion note and append Attempt entry in docs/fix_plan.md referencing the new report directory.

Pitfalls To Avoid:
- Do not modify production code or tests this loop—blueprint only.
- Keep legacy MLflow path explicitly scoped (document whether deferred or wrapped) to avoid accidental regressions.
- Preserve CONFIG-001 ordering in the design; plan must call `update_legacy_dict` before data/model construction.
- Avoid assuming new helper modules beyond `ptycho_torch/cli/shared.py` without noting circular import risks.
- Use pytest-native language in the test strategy; no unittest scaffolds.
- Ensure artifact filenames follow the timestamped directory structure exactly.
- Cite spec/doc sections when proposing behaviour changes to maintain alignment.
- Document any open questions (e.g., MLflow support, reassembly ownership) rather than deferring silently.
- Run `git status` before staging to confirm only docs/plan files changed.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/plan.md:40
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T130900Z/phase_d_cli_wrappers/summary.md:23
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md:1
- ptycho_torch/inference.py:1
- tests/torch/test_cli_inference_torch.py:1
- specs/ptychodus_api_spec.md:180
- docs/workflows/pytorch.md:344

Next Up: Phase D.C C2 — RED inference CLI thin-wrapper tests once blueprint is approved.
