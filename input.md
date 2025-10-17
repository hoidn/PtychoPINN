Summary: Document override requirements for the config bridge and stage warning coverage.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Phase B.B5.D2 override matrix
Branch: feature/torchapi
Mapped tests: none — documentation-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T063400Z/{override_matrix.md,warning_notes.md}
Do Now:
1. INTEGRATE-PYTORCH-001 Attempt #25 — Draft the override matrix (D2 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md) using the spec field list and Attempt #24 values; save to override_matrix.md — tests: none
2. INTEGRATE-PYTORCH-001 Attempt #25 — Capture warning behaviour notes for missing overrides (reference plan_update.md @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T062820Z/plan_update.md) and log actionable repro steps in warning_notes.md — tests: none
If Blocked: Record observed adapter behaviour in warning_notes.md, include traceback or pytest selector if available, and flag the blocker in docs/fix_plan.md before exiting.
Priorities & Rationale:
- specs/ptychodus_api_spec.md:213 mandates legacy params parity; documenting overrides keeps adapter aligned as new fields appear.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md D2 requires explicit override guidance before warning tests can land.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T062820Z/plan_update.md notes train→infer layering quirks; matrix must capture which overrides persist.
- tests/torch/test_config_bridge.py:600 provides current ValueError messaging; reuse for warning expectations.
How-To Map:
- Re-read `specs/ptychodus_api_spec.md` §§5.1–5.3 and `field_matrix.md` (reports/2025-10-17T041908Z) to enumerate every field that needs an override, default, or warning.
- Inspect `ptycho_torch/config_bridge.py` for actual defaults and validation flows; be explicit about ValueError/Warning text so later tests can assert them.
- Reference Attempt #24 config values (`reports/2025-10-17T061500Z/summary.md`) to state which overrides survive into final params.cfg.
- Summarize findings in `override_matrix.md` using a table (Field | Source | Default | Override Needed | Warning/Error | Notes) and cite supporting doc paths.
- For warning_notes.md, outline how to trigger the existing divergence checks (e.g., drop nphotons override) and cite current pytest selector (`pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_nphotons_default_divergence_error -vv`) without rerunning unless necessary.
- Update docs/fix_plan.md Attempts after saving artifacts; mention both new files and any open questions.
Pitfalls To Avoid:
- Do not modify adapter or tests yet; this loop is documentation/evidence only.
- Keep artifacts under the provided timestamped directory; no outputs at repo root.
- Maintain torch-optional framing—avoid instructions that require torch tensors.
- Distinguish training vs inference overrides; note which keys inference overwrites.
- Avoid duplicating baseline_params.json; reference it instead.
- When quoting errors, capture exact message text to prevent drift in later tests.
Pointers:
- specs/ptychodus_api_spec.md:200
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T062820Z/plan_update.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/field_matrix.md
- tests/torch/test_config_bridge.py:600
Next Up: Extend warning coverage tests (parity plan D3) once matrix is published.
