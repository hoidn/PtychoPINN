Summary: Remove the invalid kwargs from `to_model_config()` and normalize activation mapping so the config bridge MVP actually runs red→green.
Mode: none
Focus: INTEGRATE-PYTORCH-001 — Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T040158Z/{config_bridge_debug.md,pytest.log}
Do Now: INTEGRATE-PYTORCH-001 Attempt #10 — fix `ptycho_torch/config_bridge.py` so `to_model_config()` no longer passes unsupported kwargs, map PyTorch activations (e.g., `silu→swish`), harden required override validation, then rerun `pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg` and tee the output to the artifact log.
If Blocked: Capture a `python - <<'PY'` snippet showing the current failure (or success) instantiating `ModelConfig` via the bridge, drop the trace into `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T040158Z/pytest.log`, and expand `config_bridge_debug.md` with the blocker description.
Priorities & Rationale:
- docs/fix_plan.md:61 — Attempt #10 documents the regression; this loop must clear it before Phase B can advance.
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:48 — Phase B.B3 remains open until the adapter passes the targeted test without skip.
- ptycho_torch/config_bridge.py:125 — Current kwargs include `intensity_scale_trainable`; this is the root TypeError.
- tests/torch/test_config_bridge.py:94 — Target test exercises the full bridge workflow and must stay unskipped.
- specs/ptychodus_api_spec.md:222 — Spec enumerates the canonical ModelConfig fields; anything outside this list is unsupported.
How-To Map:
- Update `ptycho_torch/config_bridge.py` to build TF dataclass kwargs explicitly: drop `intensity_scale_trainable`, move that flag into `to_training_config`, and add an activation mapping dict (`{'silu': 'swish', 'SiLU': 'swish'}`) with a defensive error for unknown values.
- Tighten override checks: raise `ValueError` when `train_data_file`, `model_path`, or `test_data_file` are missing instead of silently passing `None`; keep overrides winning last.
- In `tests/torch/test_config_bridge.py`, replace the global torch skip with a lightweight stub so the test executes even when the real `torch` wheel is absent (e.g., inject `types.SimpleNamespace(Tensor=object)` into `sys.modules['torch']` before importing `ptycho_torch.config_params`). Document the shim at the top of the test file.
- Run `pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T040158Z/pytest.log` once to confirm the bridge now passes; include the shim log if the real torch import still fails.
- Append a "Resolution" section to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T040158Z/config_bridge_debug.md` summarizing the fix and linking to exact code lines.
Pitfalls To Avoid:
- Do not reintroduce `pytest.xfail` or global skips; keep the test authoritative.
- Avoid touching `ptycho/config/config.py`; the fix belongs in the PyTorch bridge/test only.
- Keep the bridge side-effect free—no `update_legacy_dict` calls inside the helpers.
- Preserve artifact paths; reuse the 2025-10-17T040158Z directory for new logs/notes.
- Ensure any torch stub is local to the test so production imports still fail loudly when torch is missing.
- Maintain ASCII-only edits and keep inline documentation concise.
- Do not downgrade the required override behaviour; failing fast with actionable errors is the goal.
- Limit pytest usage to the single selector above.
- Retain params.cfg snapshot/restore logic in the test fixture.
- Leave Phase B.B4 tasks untouched until this loop passes.
Pointers:
- docs/fix_plan.md:61
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:48
- ptycho_torch/config_bridge.py:125
- tests/torch/test_config_bridge.py:94
- specs/ptychodus_api_spec.md:222
Next Up: Expand Phase B.B4 parity tests once the MVP bridge passes without skip.
