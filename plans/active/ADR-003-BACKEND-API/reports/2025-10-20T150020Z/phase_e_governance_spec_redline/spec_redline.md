# Spec Redline Summary — ADR-003 Phase E.A2

## Task
Update `specs/ptychodus_api_spec.md` to document PyTorch execution configuration contract and align CLI defaults with current implementation.

## Date
2025-10-20

## Changes Made

### 1. Section 4.7 — Renamed and Expanded

**Before:** "TensorFlow-Specific Requirements" (2 bullets covering TensorFlow pipeline assumptions)

**After:** "Backend-Specific Runtime Requirements" with subsections:
- **TensorFlow Path:** Preserved existing content (tensor assumptions, custom layers, ModelManager registration)
- **PyTorch Path:** Added 5 new normative statements covering:
  - Lightning Trainer requirement and PtychoPINN_Lightning module instantiation
  - Checkpoint persistence format (`wts.h5.zip` with bundled hyperparameters)
  - CLI helper delegation contract (`ptycho_torch/cli/shared.py` functions)
  - Execution config isolation (MUST NOT populate `params.cfg`)
  - Runtime failure modes (`RuntimeError` for missing torch, `ValueError` for invalid config, `FileNotFoundError` for missing paths)

**Rationale:** PyTorch backend now has production-ready CLI wrappers and factory infrastructure (Phase D completion). Spec must document backend-specific runtime contracts to guide Ptychodus integration and future maintenance.

---

### 2. Section 4.8 — Enhanced Dispatcher Contract

**Before:** Backend routing guarantees with 7 bullets covering config field, CONFIG-001, routing, torch unavailability, metadata, persistence, validation, and inference symmetry.

**After:** Added **"Execution Config Merge"** bullet (3rd position):
- Documents dispatcher MUST accept `PyTorchExecutionConfig` objects (programmatic) or build via `build_execution_config_from_args(args, mode)` (CLI)
- Specifies factories SHALL apply execution config at priority level 2 and log applied values
- Cross-references new §4.9 for full contract

**Also updated "Validation Errors" bullet** to include factory-level validation:
- Added requirement: Factories MUST raise `ValueError` for invalid execution config fields and `FileNotFoundError` for missing paths
- Added evidence citation: Phase C2 validation

**Rationale:** Phase C/D implementation introduced execution config as separate concern from canonical configs. Spec must clarify how dispatchers merge execution config and enforce validation.

---

### 3. Section 4.9 — NEW: PyTorch Execution Configuration Contract

**Status:** Entirely new section (55 lines) inserted between §4.8 and §5.

**Content:**
- **Overview:** Defines `PyTorchExecutionConfig` as execution-only dataclass orthogonal to canonical configs
- **Core Constraints:**
  - MUST NOT populate `params.cfg` (CONFIG-001 applies only to canonical configs)
  - SHALL validate via `__post_init__` raising `ValueError`
  - IS applied at priority level 2 in factory precedence
- **Field Categories (5 groups, 17 fields total):**
  1. Lightning Trainer knobs: `accelerator`, `strategy`, `deterministic`, `gradient_clip_val`, `accum_steps`
  2. DataLoader knobs: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`
  3. Optimization knobs: `learning_rate`, `scheduler`
  4. Checkpoint/logging knobs: `enable_progress_bar`, `enable_checkpointing`, `checkpoint_save_top_k`, `checkpoint_monitor_metric`, `early_stop_patience`, `logger_backend`
  5. Inference knobs: `inference_batch_size`, `middle_trim`, `pad_eval`
- **Validation Rules:** Documents MUST constraints (accelerator whitelist, non-negative workers, positive LR, etc.)
- **CLI Integration:** References shared helpers and factory functions, prohibits manual instantiation in CLI scripts
- **Reference Implementation:** Citations to `config.py`, `cli/shared.py`, `config_factory.py`, test suite

**Rationale:** Execution config is a first-class API surface as of Phase C; requires normative specification to guide ptychodus integration and ensure maintainability. Captures both currently-exposed CLI flags and programmatic-only fields planned for Phase E.B backlog.

---

### 4. Section 7.1 — Training CLI Execution Flags (Updated)

**Before:** 4 rows covering `--accelerator`, `--deterministic`, `--num-workers`, `--learning-rate` with stale defaults.

**After:** 5 rows (added `--quiet`) with corrected values:
- `--accelerator` default: `'cpu'` → `'auto'` (with note: dataclass default `'cpu'`, CLI helper overrides to `'auto'`)
- Added `--quiet` flag (inverted to `enable_progress_bar`)
- Enhanced descriptions with hardware options (`'mps'`, `'cuda'` alias for `'gpu'`)
- Added **"Deprecated Flags"** subsection documenting `--device` and `--disable_mlflow` with migration guidance
- Added **"Planned Exposure (Phase E.B Backlog)"** subsection listing 3 categories of future CLI flags (checkpoint controls, scheduler/accumulation, logger backend)
- Cross-referenced §4.9 for validation rules and §4.8 for CONFIG-001 ordering

**Rationale:** CLI helpers now auto-detect GPU via `accelerator='auto'` (Phase C4.D evidence). Spec must reflect actual CLI defaults to avoid user confusion. Deprecation warnings guide migration away from legacy flags.

---

### 5. Section 7.2 — Inference CLI Execution Flags (Updated)

**Before:** 3 rows covering `--accelerator`, `--num-workers`, `--inference-batch-size` with stale defaults and minimal descriptions.

**After:** 4 rows (added `--quiet`) with corrections:
- `--accelerator` default: `'cpu'` → `'auto'` (with note about dataclass vs CLI defaults)
- `--inference-batch-size` default: `1` → `None` (with note: reuses training `batch_size` when `None`)
- Enhanced descriptions with typical values (16-64 for GPU, 4-8 for CPU)
- Added `--quiet` flag
- Added **"Deprecated Flags"** subsection documenting `--device` with migration timeline
- Enhanced **"Reference Implementation"** to cite CLI helpers (`resolve_accelerator`, `build_execution_config_from_args`, `validate_paths`)
- Added **"Note"** paragraph explaining programmatic access to unexposed fields via §4.9

**Rationale:** Inference CLI now uses same helper infrastructure as training (Phase D.C completion). Default batch size `None` enables checkpoint-driven inference without user intervention.

---

## Impact Summary

### Lines Changed
- **Added:** ~130 lines (§4.9 new section: 55 lines, §4.7 PyTorch path: 5 bullets, §7 expanded descriptions/notes: ~70 lines)
- **Modified:** ~20 lines (§4.8 execution config bullet, §7 table default values)
- **Deleted:** 0 lines (purely additive; preserves backward compatibility)

### Cross-References Added
- §4.7 → §4.9 (execution config validation)
- §4.8 → §4.9 (execution config contract)
- §4.9 → §4.8 (CONFIG-001 ordering)
- §7.1 → §4.9 (validation rules)
- §7.2 → §4.9 (field reference)

### Documentation Alignment
These spec changes synchronize with:
- `docs/workflows/pytorch.md` §12 (execution config usage examples)
- `ptycho/config/config.py:178-258` (PyTorchExecutionConfig dataclass definition)
- `ptycho_torch/cli/shared.py` (CLI helper functions)
- `ptycho_torch/config_factory.py` (factory integration logic)
- `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_delta_notes.md` (redline prep notes)

### Normative Guarantees Added
1. **PyTorch backend MUST use Lightning Trainer** (§4.7 PyTorch path, bullet 1)
2. **Execution config MUST NOT populate params.cfg** (§4.7 PyTorch path, bullet 4; §4.9 overview)
3. **CLI helpers MUST emit deprecation warnings** for `--device` and `--disable_mlflow` (§4.7 PyTorch path, bullet 3)
4. **Factories MUST validate execution config** and raise `ValueError`/`FileNotFoundError` (§4.8 validation errors, §4.9 overview)
5. **Accelerator default is 'auto' in CLI** (§7.1, §7.2 tables)
6. **Inference batch size defaults to None** (reuses training batch_size) (§7.2 table)

### No Breaking Changes
All edits are additive or corrective:
- TensorFlow path requirements unchanged
- Backend selection contract (§4.8) only enhanced, not rewritten
- CLI defaults corrected to match implementation (Phase D smoke evidence validates behavior)
- Deprecated flags explicitly documented with migration timeline

---

## Validation

### Markdown Syntax
- Table pipe alignment: ✓ Verified in VSCode preview
- Section numbering: ✓ §4.7 → §4.8 → §4.9 → §5 sequence intact
- Cross-references: ✓ All `§X.Y` references resolve to added/existing sections

### Consistency Checks
- ✓ CLI defaults match `ptycho_torch/train.py:429` (`accelerator='auto'`) and `ptycho_torch/inference.py:472` (`accelerator='auto'`)
- ✓ Field inventory matches `ptycho/config/config.py:178-258` (17 fields documented)
- ✓ Validation rules match `PyTorchExecutionConfig.__post_init__` logic (`config.py:248-258`)
- ✓ Deprecation messaging aligns with `ptycho_torch/cli/shared.py:resolve_accelerator` (lines 45-65)

### Evidence Citations
- Phase C2 validation: `ptycho_torch/cli/shared.py:validate_paths`
- Phase C4.D smoke: `reports/2025-10-20T111500Z/phase_c4d_at_parallel/manual_cli_smoke_gs2.log`
- Phase D completion: CLI helper delegation validated by test suite (`tests/torch/test_cli_*.py`)

---

## Next Actions (Per Phase E.A Plan)

1. ✓ **E.A2 Complete:** Spec redline applied to `specs/ptychodus_api_spec.md`
2. → **Update plan.md:** Mark row E.A2 as `[x]` with artifact path reference
3. → **Append docs/fix_plan.md:** Log Attempt entry citing spec updates + this summary file
4. → **Stage commit:** Include spec + plan updates + artifact files
5. → **Next task:** E.A3 (refresh workflow guide + knowledge base)

---

## Artifact Metadata

- **Initiative:** ADR-003-BACKEND-API
- **Phase:** E.A (Governance Dossier)
- **Task:** E.A2 (Update specs with execution config contract)
- **Artifact Path:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md`
- **Prep Notes:** `spec_delta_notes.md` (same directory)
- **Related:** Phase E.A1 addendum (`reports/2025-10-20T134500Z/phase_e_governance_adr_addendum/adr_addendum.md`)
