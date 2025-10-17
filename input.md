Summary: Implement the MVP PyTorch→spec config bridge so the Phase B.B2 test goes green.
Mode: TDD
Focus: INTEGRATE-PYTORCH-001 — Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py -k mvp -v
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T034800Z/{bridge_notes.md,pytest.log}
Do Now: INTEGRATE-PYTORCH-001 Attempt #8 — Phase B.B3 implement `ptycho_torch.config_bridge` so `tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg` passes; run `pytest tests/torch/test_config_bridge.py -k mvp -v` and capture output to the artifact log.
If Blocked: Document partial results + blockers in bridge_notes.md, keep pytest output (even if skipped) in pytest.log, and update docs/fix_plan Attempt #8 before stopping.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:31-48 — Phase B.B3 is now the top priority and depends on the B.B2 test you just authored.
- specs/ptychodus_api_spec.md:61-149 — Spec mandates that `update_legacy_dict` is the only supported way to program params.cfg; the adapter must satisfy this contract.
- plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/open_questions.md:11-26 — Q1 favours eventual shared dataclasses, so keep the bridge modular for a future handoff.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/failing_test.md:11-118 — Failing test spells out the nine MVP fields and override flow you must support.
- docs/findings.md:9 — CONFIG-001 reminds us to populate params.cfg via the bridge before any legacy caller runs.
How-To Map:
- Add `ptycho_torch/config_bridge.py` that exposes `to_model_config`, `to_training_config`, and `to_inference_config` helpers. Each should accept the existing singleton configs plus `overrides: dict | None`, consolidate values into kwargs, and return the corresponding TensorFlow dataclass (`ModelConfig`, `TrainingConfig`, `InferenceConfig`).
- Implement tuple/enum translations: convert `DataConfig.grid_size` to `gridsize`, map `ModelConfig.mode` to `model_type` (`'Unsupervised'→'pinn'`, `'Supervised'→'supervised'`), lift `TrainingConfig.epochs` into `nepochs`, and pass `DataConfig.K` through to `neighbor_count`. Ensure overrides win last.
- Keep bridge functions side-effect free; defer all `update_legacy_dict` calls to the caller (the test will run them). Use helper functions or dataclasses.replace to avoid mutating the PyTorch singletons.
- Update `tests/torch/test_config_bridge.py` by removing the xfail marker so the test enforces the new implementation; retain params.cfg snapshot/restore logic. Add a `pytest.mark.torch` marker if you want the skip reasoning to stay explicit.
- Run `pytest tests/torch/test_config_bridge.py -k mvp -v 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T034800Z/pytest.log` once. If PyTorch is still unavailable, note the skip message and confirm adapter import by a smoke check in bridge_notes.md.
- Summarize design decisions, mapping tables, and any follow-up (e.g., prep for full dataclass refactor) in `bridge_notes.md`.
Pitfalls To Avoid:
- Do not mutate `ptycho_torch/config_params.py` singletons in place; build dataclass kwargs from copies.
- Don’t call `update_legacy_dict` inside the bridge helpers—callers manage ordering.
- Keep params.cfg clean; rely on the existing test fixture snapshot/restore behaviour.
- Avoid baking in non-square grid assumptions beyond what the spec already enforces (document any limitations).
- No broad pytest runs or CI suite—just the targeted selector above.
- Preserve ASCII-only content and adhere to repo formatting conventions.
- Leave the MVP scope intact; defer additional fields to Phase B.B4.
- Respect the existing artifact path and don’t rename files.
- Capture actual command output even if pytest skips so we can spot environment gaps.
- Keep the module import-safe (no heavy imports at module scope beyond what’s required for typing).
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:31-48
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T033500Z/failing_test.md:11-199
- specs/ptychodus_api_spec.md:61-149
- plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/open_questions.md:11-26
- tests/torch/test_config_bridge.py:20-153
Next Up: Once green, move to Phase B.B4 to extend parity coverage across the remaining config fields.
