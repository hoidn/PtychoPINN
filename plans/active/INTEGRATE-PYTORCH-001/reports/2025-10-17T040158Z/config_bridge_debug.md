# Phase B.B3 Debug Notes — Config Bridge MVP

**Initiative:** INTEGRATE-PYTORCH-001  
**Phase:** B.B3 (Implementation validation)  
**Timestamp:** 2025-10-17T040158Z  
**Focus:** Diagnose why the new `ptycho_torch.config_bridge` module cannot satisfy the MVP contract when PyTorch is available.

---

## Findings Summary
1. **ModelConfig instantiation breaks immediately.** `to_model_config()` forwards `intensity_scale_trainable` from the PyTorch singleton, but `ptycho.config.config.ModelConfig` is a frozen dataclass without this field. Invoking `ModelConfig(intensity_scale_trainable=False)` raises `TypeError: ModelConfig.__init__() got an unexpected keyword argument 'intensity_scale_trainable'` (`ptycho/config/config.py:96`). The adapter will fail before reaching `update_legacy_dict`, so Attempt #9 is not actually green.
2. **Activation mapping is unsound.** PyTorch defaults `ModelConfig.amp_activation='silu'` (`ptycho_torch/config_params.py:45`), yet the TensorFlow side only recognizes `{'sigmoid','swish','softplus','relu'}` (`ptycho/model.py:406-416`). Passing 'silu' propagates an invalid activation and `get_amp_activation()` returns `ValueError`, breaking model construction. The adapter must translate PyTorch-only activations (e.g., map 'silu' to 'swish').
3. **Parities still depend on overrides for spec-critical paths.** MVP overrides currently inject `train_data_file`, `model_path`, and `n_groups`. Without baked-in defaults the adapter still fails the test contract if callers forget these overrides; we should consider stricter validation (`ValueError` with actionable messaging) before Phase B.B4 extends coverage.

---

## Evidence & References
- `ptycho_torch/config_bridge.py:70-122` — offending kwargs include `intensity_scale_trainable`.
- `ptycho/config/config.py:80-110` — ModelConfig dataclass fields (no intensity flags).
- `python - <<'PY'` reproduction in supervisor loop confirms TypeError (see loop log).
- `ptycho_torch/config_params.py:37-65` — PyTorch config defaults (`amp_activation='silu'`).
- `ptycho/model.py:406-419` — Allowed TensorFlow activation names; anything else returns `ValueError`.
- Test contract: `tests/torch/test_config_bridge.py:83-133` expects adapter to populate nine MVP fields without exception.

---

## Recommended Next Steps
1. **Strip unsupported kwargs for ModelConfig.** Build the TF dataclass kwargs explicitly, only including fields that exist (`N`, `gridsize`, `n_filters_scale`, `model_type`, `amp_activation`, `object_big`, `probe_big`, `probe_mask`, `pad_object`, `probe_scale`, `gaussian_smoothing_sigma`). Any PyTorch-specific flags (e.g., `intensity_scale_trainable`) should stay in TrainingConfig.
2. **Normalize activation values.** Add a mapping dict inside `to_model_config` that converts PyTorch names to TensorFlow equivalents (`{'silu': 'swish', 'SiLU': 'swish'}`) and raise a clear error when mapping is undefined.
3. **Validate override requirements.** For MVP make `overrides` mandatory for `train_data_file`, `model_path`, and `test_data_file` while providing descriptive error messages so the failing test reflects actionable guidance.
4. **Re-run the targeted pytest once PyTorch is usable.** Until then, consider injecting a lightweight stub (or guard) so we can execute the adapter path under unit test even without torch, preventing silent regressions.

---

## Hand-off Notes
- Update `plans/active/INTEGRATE-PYTORCH-001/implementation.md` Phase B.B3 checklist once fixes land. Leave Phase B in `in_progress` until the targeted test actually passes without skip.
- Coordinate with TEST-PYTORCH-001 to ensure activation/name normalization requirements get captured in their fixtures.
- Log fixes under docs/fix_plan.md Attempt #10 with this report directory and confirm state transition for this initiative.

