### Turn Summary
Implemented lazy loading for ptycho/model.py to eliminate import-time side effects (Phase B of REFACTOR-MODEL-SINGLETON-001).
Key changes: added `_lazy_cache` + `_model_construction_done` guards, lazy getters for `log_scale`/`initial_probe_guess`/`probe_illumination`, wrapped model construction in `_build_module_level_models()`, added `__getattr__` for backward-compatible singleton access.
Both tests pass (2/2): `test_multi_n_model_creation` and `test_import_no_side_effects` verify Phase A and Phase B criteria respectively.
Next: Phase C (XLA re-enablement) or Phase D (migrate consumers to use `create_compiled_model()` API).
Artifacts: plans/active/REFACTOR-MODEL-SINGLETON-001/reports/2026-01-07T040000Z/ (pytest_phase_b.log)
