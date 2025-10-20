# ADR-003 Phase C4 CLI Integration Plan — Execution Config Exposure (2025-10-20T033100Z)

## Context

- **Initiative:** ADR-003-BACKEND-API (Standardize PyTorch backend API)
- **Phase Goal:** Expose execution config knobs via CLI flags in `ptycho_torch/train.py` and `ptycho_torch/inference.py`, collapse ad-hoc config construction onto factory calls, and align documentation.
- **Dependencies:**
  - Phase C3 COMPLETE: `PyTorchExecutionConfig` wired through workflows (`_train_with_lightning`, `_build_inference_dataloader`)
  - Factory design: `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md`
  - Override matrix: `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md`
  - CLI inventory: Explore agent outputs (training: 12 flags, inference: 10 flags)
- **Reporting Discipline:** All artifacts under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/`
- **Mode:** TDD (tests required)

## Targeted Test Selectors

```bash
# Training CLI execution config tests (to be authored)
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::test_execution_config_cli_flags -vv

# Inference CLI execution config tests (to be authored)
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::test_execution_config_cli_flags -vv

# Factory integration smoke (existing)
CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv

# Full regression
CUDA_VISIBLE_DEVICES="" pytest tests/ -v
```

---

## Phase C4 Tasks

### C4.A — CLI Flag Mapping & Design

**Goal:** Design argparse flag surface for execution config knobs, ensuring naming consistency with TensorFlow CLI where applicable.

| ID | Task | State | How/Why & Guidance |
|----|------|-------|-------------------|
| C4.A1 | Flag inventory consolidation | [x] | ✅ 2025-10-20 — `cli_flag_inventory.md` (410 lines) records 30 flags with defaults/destinations; cites `ptycho_torch/train.py`, `ptycho_torch/inference.py`, and `override_matrix.md` refs. |
| C4.A2 | Execution knob selection | [x] | ✅ 2025-10-20 — `flag_selection_rationale.md` (425 lines) justifies 5 execution config flags (`accelerator`, `deterministic`, `num-workers`, `learning-rate`, `inference-batch-size`) and defers others per plan scope. |
| C4.A3 | Naming harmonization | [x] | ✅ 2025-10-20 — `flag_naming_decisions.md` aligns CLI naming with TF precedents, documents dual-form boolean pattern (`--deterministic`/`--no-deterministic`). |
| C4.A4 | argparse schema design | [x] | ✅ 2025-10-20 — `argparse_schema.md` details option strings, types, defaults, help text TODOs, and validation for all 5 flags with spec citations. |

**Exit Criteria:** Four design docs authored, consensus on 5 high-priority flags, argparse schema ready for RED implementation.

---

### C4.B — TDD RED Phase (CLI Tests)

**Goal:** Author failing pytest CLI harness tests before implementing CLI changes.

| ID | Task | State | How/Why & Guidance |
|----|------|-------|-------------------|
| C4.B1 | Author training CLI test scaffold | [x] | ✅ 2025-10-20 — `tests/torch/test_cli_train_torch.py` (6 RED tests) patches `create_training_payload`, asserts execution config propagation; expected failure recorded in RED log. |
| C4.B2 | Author inference CLI test scaffold | [x] | ✅ 2025-10-20 — `tests/torch/test_cli_inference_torch.py` (4 RED tests) validates CLI → `InferencePayload.execution_config` mapping via patched factory. |
| C4.B3 | Capture RED logs | [x] | ✅ 2025-10-20 — Stored `pytest_cli_train_red.log` and `pytest_cli_inference_red.log` under report directory; failures are argparse `unrecognized arguments` + mock assertions (expected). |
| C4.B4 | Document RED baseline | [x] | ✅ 2025-10-20 — `red_baseline.md` summarises 10 failing tests, failure signatures, and GREEN exit criteria for C4.C/D. |

**Exit Criteria:** CLI test scaffolds authored with 6+ failing tests, RED logs captured, baseline documented.

---

### C4.C — Implementation (CLI Refactor + Factory Integration)

**Goal:** Refactor `ptycho_torch/train.py` and `inference.py` to accept execution config CLI flags and delegate to factory functions.

#### Training CLI Updates (`ptycho_torch/train.py`)

| ID | Task | State | How/Why & Guidance |
|----|------|-------|-------------------|
| C4.C1 | Add execution config argparse flags | [ ] | Insert argparse arguments after existing flags (around line 370-380): `--accelerator` (default 'auto'), `--deterministic` (action='store_true', default True), `--num-workers` (type=int, default=0), `--learning-rate` (type=float, default=1e-3). Include help text from `argparse_schema.md`. |
| C4.C2 | Replace ad-hoc config construction with factory | [ ] | At lines 464-545 (current config construction block): (a) Remove manual `DataConfig`, `ModelConfig`, `TrainingConfig` instantiation. (b) Call `create_training_payload()` factory with CLI args bundled as overrides dict. (c) Extract `tf_training_config` from payload for CONFIG-001 bridge. Reference `factory_design.md` §3 for call pattern. |
| C4.C3 | Thread execution config to workflows | [ ] | After factory call, extract `payload.execution_config`. Pass to workflow entry point: `run_cdi_example_torch(..., execution_config=payload.execution_config)`. Ensure `_train_with_lightning` receives it (already wired in C3). |
| C4.C4 | Remove hardcoded overrides | [ ] | Eliminate hardcoded `nphotons=1e9` (line 530), `K=7` (line 477), `experiment_name='ptychopinn_pytorch'` (line 494). Let factory defaults govern; only override if CLI flag present. Document removed hardcodes in `refactor_notes.md`. |

#### Inference CLI Updates (`ptycho_torch/inference.py`)

| ID | Task | State | How/Why & Guidance |
|----|------|-------|-------------------|
| C4.C5 | Add inference execution config flags | [ ] | Insert argparse flags (around line 330-340): `--inference-batch-size` (type=int, default=None, help='Override batch size for inference DataLoader'), `--num-workers` (type=int, default=0). |
| C4.C6 | Replace ad-hoc config with factory call | [ ] | At inference config construction (approx lines 400-450): Call `create_inference_payload()` with CLI overrides dict. Extract `payload.execution_config` and pass to inference workflow helpers. |
| C4.C7 | Maintain CONFIG-001 ordering | [ ] | Ensure factory call happens before any `RawData` or model loading. Verify `update_legacy_dict()` call still precedes workflow dispatch. Add assertion or log statement confirming params.cfg populated. |

**Exit Criteria:** Both CLI scripts refactored to use factories, execution config flags accepted and forwarded, hardcoded values eliminated, CONFIG-001 compliance maintained.

---

### C4.D — TDD GREEN Phase & Validation

**Goal:** Turn RED tests GREEN and validate full CLI-to-workflow roundtrip.

| ID | Task | State | How/Why & Guidance |
|----|------|-------|-------------------|
| C4.D1 | Run targeted CLI tests | [ ] | Execute: `pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv` and `pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv`. Capture GREEN logs to `pytest_cli_train_green.log` and `pytest_cli_inference_green.log`. All tests must PASS. |
| C4.D2 | Factory integration smoke | [ ] | Re-run factory tests: `pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv`. Ensure no regressions (expect GREEN from C2). Capture log to `pytest_factory_smoke.log`. |
| C4.D3 | Full regression suite | [ ] | Run: `CUDA_VISIBLE_DEVICES="" pytest tests/ -v 2>&1 \| tee pytest_full_suite_c4.log`. **Gate:** Zero new failures vs C3 baseline (271 passed, 17 skipped, 1 xfailed). Any failures → fix before proceeding. |
| C4.D4 | Manual CLI smoke test | [ ] | Execute training with new flags: `python -m ptycho_torch.train --train_data_file <path> --output_dir /tmp/cli_smoke --n_images 64 --max_epochs 1 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-4`. Verify: (a) flags accepted, (b) no argparse errors, (c) checkpoint created, (d) logs show correct execution config values. Capture stdout to `manual_cli_smoke.log`. |

**Exit Criteria:** All CLI tests GREEN, factory smoke GREEN, full suite passed, manual smoke successful with artifacts.

---

### C4.E — Documentation Updates

**Goal:** Synchronize specs, workflow guides, and ledger with new CLI surface.

| ID | Task | State | How/Why & Guidance |
|----|------|-------|-------------------|
| C4.E1 | Update workflow guide CLI sections | [ ] | Refresh `docs/workflows/pytorch.md` §13 (or create if missing): Document new execution config flags with examples, defaults, and use cases. Include CLI command templates for common workflows (CPU training, GPU training, custom learning rate). Cross-reference `argparse_schema.md`. |
| C4.E2 | Update spec CLI tables | [ ] | Add new flags to `specs/ptychodus_api_spec.md` (create §7 "CLI Reference" if missing): Table format: Flag \| Type \| Default \| Description \| Config Field. Include both training and inference flags. |
| C4.E3 | Refresh CLAUDE.md | [ ] | Update `CLAUDE.md` §5 "Key Commands" with new CLI examples using execution config flags. Keep examples minimal (1-2 lines). |
| C4.E4 | Update implementation plan | [ ] | Mark `plans/active/ADR-003-BACKEND-API/implementation.md` Phase C4 rows complete with artifact pointers (this plan, logs, refactor notes). Update checklist verification status. |

**Exit Criteria:** Four docs updated with CLI references, examples accurate, spec tables complete.

---

### C4.F — Ledger Close-Out & Phase D Prep

**Goal:** Log Attempt in fix_plan.md, author comprehensive summary, prepare Phase D handoff.

| ID | Task | State | How/Why & Guidance |
|----|------|-------|-------------------|
| C4.F1 | Author comprehensive summary | [ ] | Create `summary.md` documenting: (a) CLI flags added (5 training, 2 inference), (b) factory integration changes (file:line citations), (c) test results (RED→GREEN counts, regression status), (d) documentation updates (4 files), (e) exit criteria validation, (f) deferred work (Phase D knobs). Template similar to C3 summary. |
| C4.F2 | Update fix_plan.md | [ ] | Append new Attempt entry under ADR-003-BACKEND-API in `docs/fix_plan.md`. Include: timestamp, phase label (C4), task summary, artifact paths, test results, exit criteria checklist, next phase pointer. Follow `prompts/update_fix_plan.md` template. |
| C4.F3 | Phase D prep notes | [ ] | In `summary.md` §"Next Steps", enumerate Phase D prerequisites: (a) checkpoint callback knobs (`checkpoint_save_top_k`, `early_stop_patience`), (b) logger backend governance (MLflow vs TensorBoard), (c) LR scheduler selection (`scheduler` field), (d) remaining execution knobs (prefetch_factor, persistent_workers). Estimate effort and dependencies. |
| C4.F4 | Hygiene verification | [ ] | Ensure no loose files at repo root (`train_debug.log`, `*.json` cache files). All artifacts under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/`. Run `git status` and relocate any strays. |

**Exit Criteria:** Summary authored (comprehensive, with metrics), fix_plan Attempt logged, Phase D handoff notes captured, repo hygiene verified.

---

## Deferred to Phase D (Out of Scope for C4)

Per C3 summary and `override_matrix.md` analysis, the following knobs are **intentionally deferred**:

1. **Checkpoint Management Knobs:**
   - `--checkpoint-save-top-k` (requires Lightning ModelCheckpoint callback configuration)
   - `--checkpoint-monitor-metric` (requires metric name validation)
   - `--early-stop-patience` (currently hardcoded to 100 in legacy code)

2. **Logger Backend Selection:**
   - `--logger-backend` (MLflow vs TensorBoard governance decision pending)
   - MLflow autologging currently triggered via `--disable-mlflow` flag (inverse logic)

3. **Advanced Training Knobs:**
   - `--scheduler` (LR scheduler type: StepLR/ReduceLROnPlateau/CosineAnnealing)
   - `--prefetch-factor`, `--persistent-workers` (DataLoader performance knobs, not yet critical)

4. **Inference-Specific Knobs:**
   - `--middle-trim`, `--pad-eval` (reconstruction post-processing not yet implemented)

**Rationale:** These knobs require additional design work (callback wiring, governance alignment, scheduler factory) beyond simple argparse-to-dataclass mapping. Phase C4 focuses on **high-impact, low-complexity** execution config exposure.

---

## Verification Checklist (Phase C4 Complete)

- [x] **C4.A:** Four design docs authored (flag inventory, selection rationale, naming decisions, argparse schema)
- [x] **C4.B:** CLI test scaffolds authored with 6+ RED tests, logs captured
- [ ] **C4.C:** Training + inference CLI refactored to use factories, hardcoded values eliminated
- [ ] **C4.D:** All CLI tests GREEN, factory smoke GREEN, full suite passed (271 passed, 0 new failures)
- [ ] **C4.E:** Four docs updated (workflow guide §13, spec CLI tables, CLAUDE.md examples, implementation plan)
- [ ] **C4.F:** Summary authored, fix_plan Attempt logged, Phase D prep notes captured, hygiene verified

**Phase C Complete Gate:** All C4 subtasks `[x]`, exit criteria validated, artifacts consolidated under timestamped report directory.

---

## Artifact Manifest (Expected by Phase End)

```
plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/
├── plan.md                          # This file
├── summary.md                       # Comprehensive close-out narrative
├── cli_flag_inventory.md            # Consolidated flag mapping (C4.A1)
├── flag_selection_rationale.md      # High-priority flag justification (C4.A2)
├── flag_naming_decisions.md         # TF naming harmonization (C4.A3)
├── argparse_schema.md               # Complete flag schema with help text (C4.A4)
├── red_baseline.md                  # RED test baseline summary (C4.B4)
├── pytest_cli_train_red.log         # Training CLI RED log (C4.B3)
├── pytest_cli_inference_red.log     # Inference CLI RED log (C4.B3)
├── pytest_cli_train_green.log       # Training CLI GREEN log (C4.D1)
├── pytest_cli_inference_green.log   # Inference CLI GREEN log (C4.D1)
├── pytest_factory_smoke.log         # Factory integration smoke (C4.D2)
├── pytest_full_suite_c4.log         # Full regression suite (C4.D3)
├── manual_cli_smoke.log             # Manual CLI smoke test output (C4.D4)
├── refactor_notes.md                # Hardcode elimination documentation (C4.C4)
```

**Storage Discipline:** All logs, design docs, and notes stored under this directory. No artifacts at repo root.

---

## References

- **Factory Design:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md`
- **Override Matrix:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md` §5
- **Phase C3 Summary:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/summary.md`
- **Spec Reference:** `specs/ptychodus_api_spec.md` §4.8 (backend selection), §6 (execution config contract)
- **Workflow Guide:** `docs/workflows/pytorch.md` §5-13 (configuration + CLI usage)
- **Ledger:** `docs/fix_plan.md` ADR-003-BACKEND-API Attempts History
