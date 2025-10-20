# ADR-003 Phase C4 CLI Integration — Loop Summary (C4.A+C4.B RED Phase Complete)

**Date:** 2025-10-20  
**Initiative:** ADR-003-BACKEND-API  
**Phase:** C4 — CLI Execution Config Exposure (Design + RED Scaffolds)  
**Loop Scope:** C4.A (4 design docs) + C4.B (10 RED test scaffolds)

---

## Problem Statement & Spec Requirements

**Objective:** Expose 5 high-priority PyTorch execution config knobs via CLI flags in training/inference scripts.

**Spec Citations:**
- `specs/ptychodus_api_spec.md` §4.8: "Backend selection routing... CONFIG-001 initialization required"
- `docs/workflows/pytorch.md` §6: "Deterministic mode enabled by default for reproducibility"
- `override_matrix.md` §5: "Priority Level 2 (Execution Config) — 5 high-priority knobs identified"

**Flags Selected (5 total):**
1. `--accelerator` (cpu/gpu/cuda/tpu/mps, default='auto') — hardware selection
2. `--deterministic` / `--no-deterministic` (bool, default=True) — reproducibility
3. `--num-workers` (int, default=0) — DataLoader parallelism  
4. `--learning-rate` (float, default=1e-3) — optimizer learning rate
5. `--inference-batch-size` (int, default=None) — inference DataLoader batch size override

---

## Work Completed (Step-by-Step)

### Phase C4.A — Design Artifacts (4 Documents, 1,760 Lines)

1. **cli_flag_inventory.md** (410 lines)
   - Consolidated 30 parameters (22 training, 8 inference)
   - Identified 15 TF canonical field gaps, 7 naming divergences
   - Documented hardcoded values: `learning_rate=1e-3`, `num_workers=0`

2. **flag_selection_rationale.md** (425 lines)
   - Selected 5 HIGH priority flags (user impact, low complexity, runtime safety)
   - Deferred 10 flags to Phase D (checkpoint/logger/scheduler knobs)
   - Documented override precedence: Explicit > Execution Config > CLI > PyTorch > TensorFlow defaults

3. **flag_naming_decisions.md** (380 lines)
   - CLI Convention: `--kebab-case` (new flags), `--snake_case` (legacy retained)
   - Dataclass Fields: `snake_case` (PEP 8)
   - Boolean Pattern: `--deterministic` (enable) + `--no-deterministic` (disable)
   - Special Case: `--accelerator` replaces `--device` (deprecation warning for backward compat)

4. **argparse_schema.md** (545 lines)
   - Complete specs for 5 flags: type, default, dest, action, choices, help text, validation
   - Two-layer validation: Argparse (type/choices) + Factory (range checks, cross-flag warnings)
   - Help text template: Purpose + Default + Guidance

**Exit Criteria:** ✅ C4.A complete (4 design docs, 5 flags documented, naming harmonized)

---

### Phase C4.B — RED Test Scaffolds (10 Tests, 440 Lines)

1. **tests/torch/test_cli_train_torch.py** (241 lines, 6 tests)
   - `test_accelerator_flag_roundtrip` — `--accelerator cpu` → `execution_config.accelerator=='cpu'`
   - `test_deterministic_flag_roundtrip` — `--deterministic` → `execution_config.deterministic==True`
   - `test_no_deterministic_flag_roundtrip` — `--no-deterministic` → `execution_config.deterministic==False`
   - `test_num_workers_flag_roundtrip` — `--num-workers 4` → `execution_config.num_workers==4`
   - `test_learning_rate_flag_roundtrip` — `--learning-rate 5e-4` → `execution_config.learning_rate==5e-4`
   - `test_multiple_execution_config_flags` — all 4 flags together (integration check)

2. **tests/torch/test_cli_inference_torch.py** (199 lines, 4 tests)
   - `test_accelerator_flag_roundtrip` — inference CLI accelerator test
   - `test_num_workers_flag_roundtrip` — inference CLI num_workers test
   - `test_inference_batch_size_flag_roundtrip` — `--inference-batch-size 32` test
   - `test_multiple_execution_config_flags` — all 3 inference flags together

**Test Strategy:**
- Mocking: `unittest.mock.patch` intercepts factory calls to capture `execution_config` argument
- CLI Invocation: `monkeypatch.setattr('sys.argv', ...)` simulates subprocess calls
- Assertions: Factory must receive `execution_config` with correct field values from CLI args

**RED Test Results:**
```bash
# Training CLI
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -vv
# Result: 6 FAILED in 5.03s (argparse.ArgumentError: unrecognized arguments)

# Inference CLI  
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py -vv
# Result: 4 FAILED in 3.77s (argparse.ArgumentError: unrecognized arguments)
```

**Failure Signature:** All 10 tests failed with `argparse.ArgumentError: unrecognized arguments: <flag>` — expected RED behavior (CLI doesn't define execution config flags yet).

3. **red_baseline.md** (310 lines)
   - Documented failure modes: argparse errors (CLI parsers missing flag definitions)
   - Established GREEN criteria: 10/10 tests PASS after argparse + factory integration
   - Defined acceptance checklist for Phase C4.C implementation

**Exit Criteria:** ✅ C4.B complete (10 RED tests, logs captured, baseline documented)

---

## Deliverables Summary

### Files Created (10 New Files, 2,590 Lines)
- Design docs: 4 files (cli_flag_inventory, flag_selection_rationale, flag_naming_decisions, argparse_schema)
- Test scaffolds: 2 files (test_cli_train_torch.py, test_cli_inference_torch.py)
- Baseline doc: 1 file (red_baseline.md)
- Logs: 2 files (pytest_cli_train_red.log, pytest_cli_inference_red.log)
- Summary: 1 file (this document)

### Files Modified
- `docs/fix_plan.md` — Appended Attempt entry for Phase C4.A+C4.B

### Artifact Storage
- All files under: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/`

---

## Test Results

### RED Phase (Current State)
- **Total Tests:** 10 (6 training, 4 inference)
- **Passed:** 0 (expected RED)
- **Failed:** 10 (argparse.ArgumentError)
- **Runtime:** Training 5.03s, Inference 3.77s

### GREEN Phase Acceptance Criteria (Phase C4.C)
- Training CLI: 6/6 tests PASS
- Inference CLI: 4/4 tests PASS  
- Factory smoke: GREEN (no regressions from Phase C3)
- Full suite: 271 passed, 0 new failures

---

## Architecture & Spec Alignment

**ADR-003 Alignment:**
- Design docs cite `override_matrix.md` precedence levels (5-tier hierarchy)
- Argparse schema matches PyTorchExecutionConfig dataclass field names
- RED tests validate factory receives execution_config from CLI args

**Spec Citations:**
- `specs/ptychodus_api_spec.md` §4.8 — backend selection, CONFIG-001 initialization
- `docs/workflows/pytorch.md` §6 — deterministic mode for reproducibility
- `override_matrix.md` §5 — execution config fields (22 total, 5 selected for C4)

**Search Results (File:Line Evidence):**
- Current CLI argparse: `ptycho_torch/train.py:366-404`, `ptycho_torch/inference.py:319-379` (no execution config flags)
- Execution config definition: `ptycho/config/config.py:72-90` (PyTorchExecutionConfig with 22 fields)
- Hardcoded values: `ptycho_torch/workflows/components.py:361` (num_workers=0), `components.py:538` (learning_rate=1e-3)
- Phase C3 wiring: `components.py:565-574` (Trainer kwargs accept execution_config)

---

## Next Loop (Phase C4.C — GREEN Implementation)

**Objective:** Implement argparse flags + execution config instantiation + factory integration to turn RED tests GREEN.

**Tasks:**
1. Add 5 argparse flags to `ptycho_torch/train.py` (~lines 370-395)
2. Add 3 argparse flags to `ptycho_torch/inference.py` (~lines 330-350)
3. Instantiate `PyTorchExecutionConfig` from parsed args (both scripts)
4. Pass `execution_config` to factory calls (both scripts)
5. Run RED → GREEN transition (expect 10/10 PASSED)
6. Run full suite regression (expect 271 passed, 0 new failures)

**Estimated Effort:** 2 hours

**Deliverables:**
- Modified `ptycho_torch/train.py`, `ptycho_torch/inference.py` (argparse + execution config wiring)
- GREEN logs: `pytest_cli_train_green.log`, `pytest_cli_inference_green.log`
- Full suite log: `pytest_full_suite_c4.log`

---

## Version Control

**Commit Message:**
```
ADR-003 Phase C4.A+C4.B: CLI execution config design + RED scaffolds

- Authored 4 design docs (cli_flag_inventory, flag_selection_rationale, flag_naming_decisions, argparse_schema)
- Created 10 RED pytest scaffolds (6 training, 4 inference CLI execution config tests)
- Captured RED logs (10 FAILED as expected with argparse.ArgumentError)
- Documented acceptance criteria in red_baseline.md

Design artifacts cover 5 high-priority execution config flags (accelerator, deterministic, num-workers, learning-rate, inference-batch-size).
RED tests establish TDD baseline for Phase C4.C GREEN implementation.

Test suite: 10 RED (expected), 0 new failures
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/
Module scope: CLI/config (argparse + execution config integration)
```

**Files Staged:** 10 new files + 1 modified (docs/fix_plan.md)

---

**Loop COMPLETE:** Phase C4.A (design) + C4.B (RED scaffolds) delivered. Ready for Phase C4.C (GREEN implementation) in next loop.

---

## 2025-10-20T120500Z — Documentation Update Snapshot (C4.E)

- **Scope:** Completed C4.E tasks (workflow guide, spec CLI tables, CLAUDE.md, implementation plan update).
- **Artifacts:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T120500Z/phase_c4_docs_update/{summary.md,docs_diff.txt}`.
- **Highlights:**
  - `docs/workflows/pytorch.md` §12 now documents execution config flags with gridsize=2 CLI example and CONFIG-001 guidance.
  - `specs/ptychodus_api_spec.md` §7 adds training/inference CLI flag tables mapped to `PyTorchExecutionConfig`.
  - `CLAUDE.md` Key Commands include PyTorch CLI example using new flags; CONFIG-001 bridging note retained.
  - Implementation plan Phase C4 row updated with completion timestamp and artifact pointer.
- **Next:** Proceed with C4.F checklist (comprehensive summary, fix_plan Attempt entry, Phase D prep notes, hygiene check).
