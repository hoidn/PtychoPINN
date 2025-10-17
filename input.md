Summary: Make the config bridge parity tests runnable without torch and capture the first green check for probe_mask/nphotons.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -k N-direct -v
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/{parity_green_plan.md,implementation_notes.md,adapter_diff.md,pytest_phaseA.log}
Do Now: INTEGRATE-PYTORCH-001 Attempt #14 — Execute Phase A (tasks A1–A3) by enabling the parity harness without torch, then run `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -k N-direct -v 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/pytest_phaseA.log` to prove the selector no longer skips.
If Blocked: Record the skip reason, stack trace, or missing fallback design in `implementation_notes.md`, keep the partial pytest log, and notify the supervisor in docs/fix_plan.md Attempts History.
Priorities & Rationale:
- docs/fix_plan.md:63 — Attempt #14 now mandates the green-phase execution; we must unblock the harness before touching adapter fields.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md:12 — Phase A defines the enabling tasks and exit criteria we need this loop.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/summary.md:149 — P0 blockers highlight probe_mask/nphotons urgency once tests run.
- tests/torch/test_config_bridge.py:1 — Existing parity tests expect update_legacy_dict-driven assertions; we must keep their structure intact while removing skip conditions.
- ptycho_torch/config_params.py:1 — Top-level torch import currently triggers ImportError; the shim must intercept this without mutating behaviour when torch exists.
How-To Map:
1. Read Phase A in `parity_green_plan.md` and log initial observations (skip markers, import paths) into `implementation_notes.md` before code changes.
2. Inspect `tests/conftest.py` skip logic and `ptycho_torch/config_params.py` imports; design the optional torch shim (e.g., helper module or lazy import) and describe it in notes.
3. Implement the shim + pytest gating updates so `TestConfigBridgeParity` selectors are not auto-skipped when torch is absent; capture code decisions in `adapter_diff.md`.
4. Re-run the targeted selector with the command above; confirm the test executes (pass/fail acceptable, but must not SKIP) and store output as `pytest_phaseA.log` under the new report directory.
5. Document the resulting behaviour (including any remaining failures) in `implementation_notes.md`, call out follow-up actions for Phase B tasks, and stage artifacts in git.
Pitfalls To Avoid:
- Do not remove skip markers from genuinely torch-dependent tests; scope adjustments narrowly to config bridge selectors.
- Avoid introducing side-effects in `ptycho_torch/config_params.py`; keep the shim import-only and maintain current defaults.
- Preserve KEY_MAPPINGS behaviour — no manual params.cfg tweaks outside `update_legacy_dict` flows.
- Keep all new artifacts inside `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/`.
- Do not delete the red-phase logs or summary from the previous report directory; they remain authoritative history.
- Guard against accidentally catching ImportError for unrelated reasons; the shim should surface unexpected failures.
- Refrain from running the full pytest suite; stick to the targeted selector unless Phase A completes.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md:12
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/summary.md:149
- tests/torch/test_config_bridge.py:200
- ptycho_torch/config_params.py:1
- tests/conftest.py:12
Next Up: Phase B tasks from `parity_green_plan.md` (probe_mask conversion, nphotons override enforcement) once the harness runs without skips.
