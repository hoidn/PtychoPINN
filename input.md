Summary: Lock in the n_subsample override requirement in the config bridge parity suite.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Phase B.B5 parity follow-through (Phase C kickoff)
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py -k "n_subsample" -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T055335Z/{summary.md,pytest_n_subsample_red.log,pytest_n_subsample.log}
Do Now: INTEGRATE-PYTORCH-001 Attempt #22 — Add n_subsample parity tests to the config bridge suite, capture the failing selector, implement the adapter guard, then rerun `pytest tests/torch/test_config_bridge.py -k "n_subsample" -vv`
If Blocked: Store the failing selector output in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T055335Z/blocked.log`, document the blocker in summary.md, and record the attempt in docs/fix_plan.md.
Priorities & Rationale:
- parity_green_plan Phase C highlights n_subsample as the next override-required field; closing it unblocks remaining parity matrix rows (plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md).
- field_matrix.md flags `n_subsample` as override_required in both Training and Inference configs; tests must assert the spec contract (plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/field_matrix.md).
- specs/ptychodus_api_spec.md §5.2-§5.3 require explicit override handling to keep PyTorch and TensorFlow sampling semantics in sync.
- CONFIG-001 finding mandates params.cfg consistency; adding the guard prevents silent divergence when overrides are omitted (docs/findings.md).
How-To Map:
- Extend `tests/torch/test_config_bridge.py` with pytest cases covering missing vs explicit `n_subsample` overrides for both TrainingConfig and InferenceConfig; reuse `params_cfg_snapshot` to keep globals clean.
- Run the targeted selector red to confirm the new tests fail before implementation: `pytest tests/torch/test_config_bridge.py -k "n_subsample" -vv 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T055335Z/pytest_n_subsample_red.log`.
- Update `ptycho_torch/config_bridge.py` so both `to_training_config` and `to_inference_config` reject missing `n_subsample` overrides with actionable ValueErrors, then regenerate spec-side dataclasses with the override applied.
- Re-run the selector for the green pass and capture the log as `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T055335Z/pytest_n_subsample.log`; verify the message copy matches the field matrix guidance.
- Summarize decisions, error message text, and any deferred scope in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T055335Z/summary.md`, then update docs/fix_plan.md Attempts.
Pitfalls To Avoid:
- Do not introduce hard torch dependencies—parity suite must run in torch-free environments.
- Keep overrides dict construction explicit; avoid mutating shared dictionaries between tests.
- Preserve existing parity selectors and markers; no broad `pytest` runs beyond the scoped command.
- Ensure params.cfg snapshots restore state even when tests raise errors.
- Keep error message strings actionable and spec-aligned; include override syntax examples.
- Avoid touching unrelated adapter fields (probe_mask, nphotons) this loop.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/field_matrix.md
- specs/ptychodus_api_spec.md:1
- docs/findings.md:9
- tests/torch/test_config_bridge.py:1
- ptycho_torch/config_bridge.py:200
Next Up: 1) Extend parity coverage to subsample_seed overrides (parity plan Phase C), 2) Draft params.cfg baseline comparison test (Phase D1) once n_subsample guard is green.
