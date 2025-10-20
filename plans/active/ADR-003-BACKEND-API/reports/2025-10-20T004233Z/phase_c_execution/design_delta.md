# ADR-003 Phase C1.A1 — Execution Config Field Reconciliation

**Date:** 2025-10-20T004800Z
**Task:** C1.A1 — Reconcile field list + defaults
**Analyst:** Ralph (Engineer Loop)

---

## 1. Purpose

This document reconciles the execution configuration field list defined in two authoritative design sources:
- `factory_design.md` §2.2 (proposed dataclass structure)
- `override_matrix.md` §5 (comprehensive field inventory)

Goal: Identify any deltas (additions, removals, type changes, default divergences) and document rationale for inclusion/exclusion in the canonical `PyTorchExecutionConfig` dataclass.

---

## 2. Field-by-Field Comparison

| Field | Factory Design §2.2 | Override Matrix §5 | Status | Notes |
|-------|---------------------|-------------------|--------|-------|
| **accelerator** | str = 'auto' | str = 'auto' | ✅ MATCH | cpu/gpu/tpu/mps/auto |
| **strategy** | str = 'auto' | str = 'auto' | ✅ MATCH | auto/ddp/fsdp/deepspeed |
| **n_devices** | int = 1 | int = 1 | ✅ MATCH | Derived from CLI `--device` |
| **deterministic** | bool = True | bool = True | ✅ MATCH | Reproducibility flag |
| **num_workers** | int = 0 | int = 0 | ✅ MATCH | DataLoader workers |
| **pin_memory** | bool = False | bool = False | ✅ MATCH | DataLoader pin_memory |
| **persistent_workers** | bool = False | bool = False | ✅ MATCH | DataLoader persistent_workers |
| **prefetch_factor** | Optional[int] = None | Optional[int] = None | ✅ MATCH | DataLoader prefetch_factor |
| **learning_rate** | float = 1e-3 | float = 1e-3 | ✅ MATCH | Optimizer LR |
| **scheduler** | str = 'Default' | str = 'Default' | ✅ MATCH | LR scheduler type |
| **gradient_clip_val** | Optional[float] = None | Optional[float] = None | ✅ MATCH | Gradient clipping |
| **accum_steps** | int = 1 | int = 1 | ✅ MATCH | Gradient accumulation |
| **enable_checkpointing** | bool = True | bool = True | ✅ MATCH | Lightning checkpoint callback |
| **checkpoint_save_top_k** | int = 1 | int = 1 | ✅ MATCH | Top-K checkpoint retention |
| **checkpoint_monitor_metric** | str = 'val_loss' | str = 'val_loss' | ✅ MATCH | Metric for best checkpoint |
| **early_stop_patience** | int = 100 | int = 100 | ✅ MATCH | Early stopping patience |
| **enable_progress_bar** | bool = False | bool = False | ✅ MATCH | Progress bar visibility |
| **logger_backend** | Optional[str] = None | Optional[str] = None | ✅ MATCH | tensorboard/wandb/mlflow |
| **disable_mlflow** | bool = False | bool = False | ✅ MATCH | Legacy API MLflow toggle |
| **inference_batch_size** | Optional[int] = None | Optional[int] = None | ✅ MATCH | Inference-specific batch size |
| **middle_trim** | int = 0 | int = 0 | ✅ MATCH | Inference trimming |
| **pad_eval** | bool = False | bool = False | ✅ MATCH | Evaluation padding |

---

## 3. Delta Summary

**Total Fields:** 22
**Exact Matches:** 22
**Additions:** 0
**Removals:** 0
**Type Divergences:** 0
**Default Divergences:** 0

**Conclusion:** Factory design §2.2 and override matrix §5 are **fully aligned**. No reconciliation actions required.

---

## 4. Implementation Guidance

### 4.1 Dataclass Placement Decision

**Recommendation:** Implement in `ptycho/config/config.py` (Option A from factory_design.md §2.2).

**Rationale:**
1. **Single Source of Truth:** Aligns with CONFIG-001 principle (canonical dataclasses in `ptycho/config/`)
2. **Backend Symmetry:** PyTorch execution config lives alongside TensorFlow canonical configs
3. **Documentation Cohesion:** §6 KEY_MAPPINGS and update_legacy_dict() proximity
4. **Export Consistency:** Existing `__all__` export pattern in same module

**Alternative Rejected:** `ptycho_torch/config_params.py` (PyTorch-specific module) — would fracture config architecture and duplicate dataclass patterns.

### 4.2 Field Organization

Fields are logically grouped into 7 categories (preserve this structure in docstring):

1. **Hardware & Distributed Training:** accelerator, strategy, n_devices, deterministic
2. **Data Loading:** num_workers, pin_memory, persistent_workers, prefetch_factor
3. **Optimization:** learning_rate, scheduler, gradient_clip_val, accum_steps
4. **Checkpointing & Early Stopping:** enable_checkpointing, checkpoint_save_top_k, checkpoint_monitor_metric, early_stop_patience
5. **Logging & Experiment Tracking:** enable_progress_bar, logger_backend, disable_mlflow
6. **Inference-Specific:** inference_batch_size, middle_trim, pad_eval

### 4.3 Docstring Requirements

Per spec and workflow guidance:
- Reference POLICY-001 (PyTorch mandatory) — execution config requires torch>=2.2
- Reference CONFIG-001 (params.cfg sync) — execution config orthogonal to legacy params
- Cite `docs/workflows/pytorch.md` §§5-13 for usage examples
- Note that these fields control **runtime behavior** and do NOT affect model topology or data pipeline (unlike ModelConfig)

### 4.4 ASCII Ordering

Insert `PyTorchExecutionConfig` **after** `ModelConfig` and **before** `TrainingConfig` for alphabetical dataclass ordering within module.

---

## 5. Open Questions (Deferred to Phase C2)

None identified during field reconciliation. All questions captured in `factory_design.md` §7 remain under supervisor review.

---

## 6. Exit Criteria for C1.A1

- [x] Compared all 22 fields from factory_design.md vs override_matrix.md
- [x] Documented delta summary (ZERO deltas)
- [x] Provided implementation guidance (placement, organization, docstring, ordering)
- [x] Stored artifact at `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/design_delta.md`

**Next Task:** C1.A3 — Author RED tests in `tests/torch/test_execution_config.py`

---

## 7. References

- `factory_design.md:100-145` — PyTorchExecutionConfig proposed structure
- `override_matrix.md:89-114` — Execution config field inventory
- `ptycho/config/config.py:1-296` — Canonical config module structure
- `specs/ptychodus_api_spec.md` §4.8, §6 — CONFIG-001 + backend execution contract
- `docs/findings.md` POLICY-001, CONFIG-001 — Non-negotiable requirements
