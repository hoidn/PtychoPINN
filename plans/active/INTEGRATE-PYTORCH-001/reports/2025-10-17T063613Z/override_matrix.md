# Override Matrix — Config Bridge (Phase B.B5.D2)

**Initiative:** INTEGRATE-PYTORCH-001  
**Timestamp:** 2025-10-17T063613Z  
**Author:** galph (supervisor evidence loop)  
**Scope:** Document current override behaviour for the config bridge adapter ahead of Phase B.B5.D3 warning tests. Findings sourced from `ptycho_torch/config_bridge.py`, `tests/torch/test_config_bridge.py`, baseline snapshots (`reports/2025-10-17T041908Z/baseline_params.json`), and fresh runs captured under this directory.

## 1. Train → Inference Layering Snapshot

Artifacts generated this loop:
- `train_params.json` — params.cfg state immediately after applying the **training** config.
- `final_params.json` — params.cfg state after applying the **inference** update on top.
- `train_vs_final_diff.json` — keys whose values change between the two stages.

Key observations from `train_vs_final_diff.json`:
- `model_path`, `debug`, `output_prefix`, `test_data_file_path`, `n_groups`, `n_subsample`, and `subsample_seed` are overwritten by the inference update. Training overrides for those keys do **not** survive the second `update_legacy_dict()` call.
- Training-only overrides that remain intact after inference: `train_data_file_path`, `mae_weight`, `nll_weight`, `realspace_*`, `probe.trainable`, `sequential_sampling`, `nphotons`, etc.
- Without explicit overrides, several spec-required fields stay at `None` or default values, signalling candidates for warning enforcement (see tables below).

## 2. ModelConfig Fields

| Field | Source Without Override | Behaviour When Override Missing | Validation / Tests | Notes |
| --- | --- | --- | --- | --- |
| `probe_mask` | PyTorch Optional[Tensor] → bool (`None` → `False`) | No warning; defaults to `False` | `test_model_config_probe_mask_translation`, `test_model_config_probe_mask_override` | Tensor→True path only exercised when torch available; consider warning when PyTorch mask Tensor is dropped in fallback mode. |
| `pad_object` | Hard-coded default `True` | No warning; stays `True` unless override provided | Covered implicitly via baseline compare | Potential warning candidate once PyTorch exposes this flag; currently spec default accepted. |
| `probe_scale` | Pulled from `DataConfig.probe_scale` (PyTorch default `1.0`) | No warning; silently passes through PyTorch default (diverges from TF default `4.0`) | `test_default_divergence_detection` (explicit override happy path) | **Gap:** No guard for default divergence. D3 should add warning/error when PyTorch default is detected without override. |
| `gaussian_smoothing_sigma` | Hard-coded default `0.0` | No warning; stays `0.0` | Baseline comparison | OK for now; spec default matches. |

## 3. TrainingConfig Fields

| Field | Source Without Override | Behaviour When Override Missing | Validation / Tests | Survives Inference Update? | Notes / Follow-up |
| --- | --- | --- | --- | --- | --- |
| `train_data_file` | `None` placeholder | Raises `ValueError` with actionable message | `test_train_data_file_required_error` (log: `pytest_missing_train_data.log`) | ✅ | Guard in place. |
| `test_data_file` | `None` placeholder | No warning; remains `None` until inference override applied | -- | ❌ | Training stage has no check; inference overrides final value. Consider adding warning to surface absent evaluation data. |
| `n_groups` | `None` placeholder | No warning; remains `None` → inference override supplies final value | MVP parity tests require explicit override but adapter does not enforce | ❌ | Candidate for D3 validation — missing override leaves params.cfg with `None` if inference also omits value. |
| `n_subsample` | `None`; PyTorch semantic collision ignored | Defaults to `None`; explicit override honoured | `test_training_config_n_subsample_*` | ⬜ (overwritten when inference provides value) | Current behaviour matches parity plan (no warning). |
| `subsample_seed` | `None` | No warning; inference override overwrites | -- | ❌ | Documented in diff; consider warning if neither stage supplies value. |
| `nphotons` | `DataConfig.nphotons` | Raises `ValueError` if PyTorch default `1e5` used without override | `test_nphotons_default_divergence_error` (log: `pytest_nphotons_error.log`) | ✅ | Guard effective; default divergence resolved via override. |
| `output_dir` | Defaults to `Path('training_outputs')` | No warning; overwritten by inference | -- | ❌ | Acceptable (training artefacts vs inference). |
| `positions_provided`, `probe_trainable`, `sequential_sampling` | Hard-coded defaults (True/False) | No warnings | Covered by parity tests | ✅ | Defaults aligned with spec. |
| `test_data_file` (duplicate entry for emphasis) | see above | -- | -- | -- | Ensure documentation emphasises inference override requirement. |

## 4. InferenceConfig Fields

| Field | Source Without Override | Behaviour When Override Missing | Validation / Tests | Notes |
| --- | --- | --- | --- | --- |
| `model_path` | `None` placeholder | Raises `ValueError` (“model_path is required…”) | `test_model_path_required_error` | Guard complete. |
| `test_data_file` | `None` placeholder | Raises `ValueError` (“test_data_file is required…”) | `test_inference_config_test_data_file_required_error` | Guard complete. |
| `n_groups`, `n_subsample`, `subsample_seed` | `None` placeholders | No warnings; remain `None` if overrides omitted | Coverage via positive-path tests only | Need D3 decision: either enforce overrides or document acceptable defaults. |
| `output_dir` | Defaults to `Path('inference_outputs')` | No warning | Baseline comparison ensures override works | Acceptable default for inference artefacts. |
| `debug` | Defaults to `False` | No warning | Baseline uses override to set `True` | Documented behaviour; no guard required. |

## 5. Warning Coverage Gaps (Targets for Phase B.B5.D3)

1. **probe_scale default divergence** — add warning/error when PyTorch default `1.0` is passed without override (spec default `4.0`).
2. **n_groups training stage** — enforce explicit override or warn when left `None`; otherwise legacy workflow receives `None` if inference also omits value.
3. **test_data_file (training stage)** — consider warning when omitted so callers know inference update is required for evaluation flows.
4. **n_subsample / subsample_seed layering** — confirm acceptable for inference to overwrite training value; if not, add guard preventing unexpected downgrades.
5. **Torch-less probe_mask** — document that tensor-backed mask collapses to `False`; decide whether to emit warning when TORCH_AVAILABLE but tensor is dropped (requires runtime detection).

## 6. Evidence Log

| Artifact | Purpose |
| --- | --- |
| `train_vs_final_diff.json` | Field-level diff between training-only params.cfg and final layered state. |
| `pytest_missing_train_data.log` | Confirms ValueError when `train_data_file` override omitted (guard working). |
| `pytest_nphotons_error.log` | Confirms ValueError + guidance when nphotons override missing (default divergence guard). |
| `train_params.json`, `final_params.json` | Reference snapshots for future plan phases. |

## 7. Recommendations

- Feed the gap list above into Phase B.B5.D3 test design — each row should map to a new warning/assertion test.
- Update `plans/active/INTEGRATE-PYTORCH-001/implementation.md` and `parity_green_plan.md` to mark D2 complete and reference this matrix.
- When implementing D3, prefer `pytest.raises` assertions similar to `test_nphotons_default_divergence_error` to preserve message coverage while keeping tests torch-optional.

