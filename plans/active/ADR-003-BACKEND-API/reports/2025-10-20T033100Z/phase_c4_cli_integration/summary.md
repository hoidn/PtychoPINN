# Phase C4 CLI Integration Planning Summary

**Timestamp:** 2025-10-20T033100Z
**Initiative:** ADR-003-BACKEND-API
**Phase:** C4 (CLI Integration — Planning Only)
**Mode:** Docs
**Status:** PLANNING COMPLETE (awaiting execution)

---

## Executive Summary

Authored comprehensive Phase C4 execution plan for exposing PyTorch execution config knobs via CLI flags in `ptycho_torch/train.py` and `inference.py`. Plan defines 6 phased task groups (A–F) with 24 checklist items covering design, TDD RED/GREEN cycles, implementation, documentation, and ledger close-out.

**Key Deliverables Planned:**
- **5 training CLI flags:** `--accelerator`, `--deterministic`, `--num-workers`, `--learning-rate`, `--inference-batch-size`
- **2 inference CLI flags:** `--inference-batch-size`, `--num-workers`
- **Factory integration refactor:** Replace ad-hoc config construction (lines 464-545 in `train.py`) with `create_training_payload()` / `create_inference_payload()` calls
- **6+ CLI tests:** TDD RED→GREEN cycle validating argparse → factory → workflow roundtrip
- **4 documentation updates:** Workflow guide §13, spec CLI tables, CLAUDE.md examples, implementation plan

**Deferred to Phase D:** Checkpoint callbacks, logger backend governance, LR scheduler selection, advanced DataLoader knobs (9 fields per C3 summary).

---

## Planning Process

### Context Gathering (Parallel Exploration)

Launched 3 concurrent Explore subagents to inventory current state:

1. **Training CLI Inventory (`ptycho_torch/train.py:350-560`):**
   - **12 existing flags** cataloged: 9 new-interface flags (train_data_file, output_dir, max_epochs, n_images, gridsize, batch_size, device, test_data_file, disable_mlflow), 2 legacy flags (ptycho_dir, config), 1 execution knob (device)
   - **10 hardcoded values** identified: early_stop_patience=100, val_split=0.05, seed=42, nphotons=1e9, learning_rate=1e-3, experiment_name='ptychopinn_pytorch', K=7, mode='Unsupervised', amp_activation='silu', n_devices logic

2. **Inference CLI Inventory (`ptycho_torch/inference.py:260-380`):**
   - **10 existing flags** cataloged: 6 Phase E2.C2 flags (model_path, test_data, output_dir, n_images, device, quiet), 4 legacy MLflow flags
   - **Key gaps:** No execution config exposure, no device persistence, dtype enforcement hardcoded

3. **Execution Knob Analysis (C3 summary vs override_matrix.md §5):**
   - **10 knobs already wired** in Phase C3: accelerator, strategy, deterministic, gradient_clip_val, accum_steps, num_workers, pin_memory, enable_progress_bar, enable_checkpointing, inference_batch_size
   - **9 knobs defined but not consumed:** scheduler, persistent_workers, prefetch_factor, learning_rate (partially), logger_backend, checkpoint_save_top_k, checkpoint_monitor_metric, early_stop_patience, middle_trim, pad_eval
   - **6 knobs missing entirely:** seed_everything, log_every_n_steps, devices, val_check_interval, max_steps, default_root_dir

### Design Decisions

**High-Priority Flag Selection (C4.A2):**

Selected **5 execution config flags** for C4 based on:
- Already wired in workflows (C3 complete): accelerator, deterministic, num_workers, inference_batch_size
- High user demand: learning_rate (currently hardcoded 1e-3)
- Low implementation complexity: direct argparse → dataclass mapping

**Deferred Flags (9 total):**
- **Checkpoint knobs (3):** Require Lightning callback wiring beyond simple dataclass mapping
- **Logger backend (1):** Governance decision pending (MLflow vs TensorBoard)
- **Advanced training (2):** Scheduler selection, DataLoader performance knobs
- **Inference post-processing (2):** Trimming/padding not yet implemented
- **Learning rate (partial):** Exposed as flag but optimizer integration deferred

**Naming Convention:**
- Follow `--snake-case` argparse convention (consistent with existing flags)
- Match TensorFlow CLI naming where applicable (e.g., `--deterministic` aligns with reproducibility flags)
- Use explicit names for execution knobs: `--num-workers` (not `--workers`), `--learning-rate` (not `--lr`)

---

## Plan Structure Overview

### Phase C4.A — CLI Flag Mapping & Design (4 tasks)
**Goal:** Design argparse surface with comprehensive schema documentation.

- **C4.A1:** Consolidate Explore outputs into `cli_flag_inventory.md` (training: 12 flags, inference: 10 flags)
- **C4.A2:** Select high-priority knobs with rationale (`flag_selection_rationale.md`)
- **C4.A3:** Harmonize naming with TensorFlow CLI precedents (`flag_naming_decisions.md`)
- **C4.A4:** Author complete argparse schema with types, defaults, help text, validation logic

**Exit Criteria:** Four design docs authored, consensus on 5 training + 2 inference flags.

### Phase C4.B — TDD RED Phase (4 tasks)
**Goal:** Author failing CLI tests before implementation.

- **C4.B1:** Create `tests/torch/test_cli_train_torch.py` with 4 roundtrip tests (accelerator, deterministic, num_workers, learning_rate)
- **C4.B2:** Create `tests/torch/test_cli_inference_torch.py` with 2 roundtrip tests (inference_batch_size, num_workers)
- **C4.B3:** Capture RED logs for both test modules
- **C4.B4:** Document RED baseline with failure signatures and GREEN criteria

**Exit Criteria:** 6+ RED tests authored, logs captured, baseline documented.

### Phase C4.C — Implementation (7 tasks)
**Goal:** Refactor CLI scripts to use factory functions and expose execution config flags.

**Training CLI (`ptycho_torch/train.py`):**
- **C4.C1:** Add 4 argparse flags (accelerator, deterministic, num_workers, learning_rate)
- **C4.C2:** Replace ad-hoc config construction (lines 464-545) with `create_training_payload()` factory call
- **C4.C3:** Thread `payload.execution_config` to `run_cdi_example_torch()`
- **C4.C4:** Eliminate hardcoded overrides (nphotons=1e9, K=7, experiment_name)

**Inference CLI (`ptycho_torch/inference.py`):**
- **C4.C5:** Add 2 argparse flags (inference_batch_size, num_workers)
- **C4.C6:** Replace ad-hoc config with `create_inference_payload()` factory call
- **C4.C7:** Maintain CONFIG-001 ordering (factory → update_legacy_dict → workflow)

**Exit Criteria:** Both CLI scripts refactored, execution config flags functional, CONFIG-001 compliance maintained.

### Phase C4.D — TDD GREEN & Validation (4 tasks)
**Goal:** Turn RED tests GREEN and validate full CLI-to-workflow integration.

- **C4.D1:** Run CLI tests, capture GREEN logs (all 6 tests PASS expected)
- **C4.D2:** Run factory smoke tests (no regressions vs C2 baseline)
- **C4.D3:** Run full regression suite (GATE: 271 passed, 0 new failures vs C3)
- **C4.D4:** Execute manual CLI smoke test with new flags, verify artifacts

**Exit Criteria:** All tests GREEN, no regressions, manual smoke successful.

### Phase C4.E — Documentation Updates (4 tasks)
**Goal:** Synchronize specs and guides with new CLI surface.

- **C4.E1:** Update `docs/workflows/pytorch.md` §13 with CLI flag examples
- **C4.E2:** Add CLI reference table to `specs/ptychodus_api_spec.md` §7 (create if missing)
- **C4.E3:** Refresh `CLAUDE.md` §5 with execution config examples
- **C4.E4:** Mark Phase C4 complete in `implementation.md` with artifact pointers

**Exit Criteria:** Four docs updated with accurate CLI references and examples.

### Phase C4.F — Ledger Close-Out & Phase D Prep (4 tasks)
**Goal:** Log Attempt, author summary, prepare Phase D handoff.

- **C4.F1:** Author comprehensive `summary.md` with metrics, citations, validation
- **C4.F2:** Update `docs/fix_plan.md` with new Attempt entry (template: `prompts/update_fix_plan.md`)
- **C4.F3:** Document Phase D prerequisites in summary (checkpoint callbacks, logger backend, scheduler)
- **C4.F4:** Verify repo hygiene (no loose files at root, all artifacts under timestamped directory)

**Exit Criteria:** Summary authored, fix_plan updated, Phase D handoff notes captured, hygiene verified.

---

## Key Implementation Notes

### Factory Integration Pattern (C4.C2)

**Current Ad-Hoc Construction (lines 464-545):**
```python
# Manual config instantiation
data_config = DataConfig(N=inferred_N, grid_size=(args.gridsize, args.gridsize), K=7, nphotons=1e9)
model_config = ModelConfig(mode='Unsupervised', amp_activation='silu')
training_config = TrainingConfig(epochs=args.max_epochs, batch_size=args.batch_size, ...)

# Manual bridge to TensorFlow
tf_model_config = to_model_config(data_config, model_config)
tf_training_config = to_training_config(..., overrides={'n_groups': args.n_images, 'nphotons': 1e9})
update_legacy_dict(params.cfg, tf_training_config)
```

**Proposed Factory-Based Refactor:**
```python
# Single factory call replaces ~80 lines
from ptycho_torch.config_factory import create_training_payload

payload = create_training_payload(
    train_data_file=Path(args.train_data_file),
    overrides={
        'n_groups': args.n_images,
        'gridsize': args.gridsize,
        'batch_size': args.batch_size,
        'max_epochs': args.max_epochs,
        'output_dir': Path(args.output_dir),
        # Execution config knobs
        'accelerator': args.accelerator,
        'deterministic': args.deterministic,
        'num_workers': args.num_workers,
        'learning_rate': args.learning_rate,
    }
)

# CONFIG-001 compliance (factory handles this internally via populate_legacy_params)
# payload.tf_training_config already has params.cfg populated

# Thread execution config to workflows
run_cdi_example_torch(
    train_data=train_data,
    config=payload.tf_training_config,
    execution_config=payload.execution_config
)
```

**Benefits:**
- **73% code reduction estimate** (from factory_design.md Phase B1 analysis)
- **Single source of truth:** Factory governs override precedence, validation, bridge logic
- **Testability:** Factory unit tests already GREEN (C2), CLI tests validate argparse → factory mapping
- **Maintainability:** Future config changes isolated to factory, not scattered across CLI scripts

### CONFIG-001 Compliance Verification (C4.C7)

**Critical Ordering:**
1. **Infer probe size** (if applicable): `_infer_probe_size(train_data_file)` → includes NPZ loading
2. **Call factory:** `create_training_payload()` → internally calls `populate_legacy_params(params.cfg, tf_config)`
3. **Import ptycho modules:** Now safe because params.cfg is populated
4. **Load data:** `RawData.from_file()` → depends on params.cfg['gridsize'], params.cfg['N']
5. **Dispatch workflow:** `run_cdi_example_torch()` → assumes params.cfg synchronized

**Test Assertion (C4.D1):**
```python
def test_factory_populates_params_cfg_before_data_load():
    # Mock factory call
    payload = create_training_payload(train_data_file=..., overrides={...})

    # Assert params.cfg populated
    from ptycho import params
    assert params.cfg.get('N') == payload.tf_training_config.model.N
    assert params.cfg.get('gridsize') == payload.tf_training_config.model.gridsize

    # Now RawData.from_file() should succeed without shape errors
    raw_data = RawData.from_file(train_data_file)
    assert raw_data.diff3d.shape[0] == params.cfg.get('N')
```

---

## Deferred Work (Phase D Scope)

Per `override_matrix.md` §5 and C3 summary lines 96-105, the following execution config knobs are **intentionally excluded** from C4:

### Checkpoint Management Knobs (3 fields)
- **`--checkpoint-save-top-k`** (default: -1, save all) — Requires ModelCheckpoint callback configuration
- **`--checkpoint-monitor-metric`** (default: 'val_loss') — Requires metric name validation against Lightning module
- **`--early-stop-patience`** (currently hardcoded: 100) — Requires EarlyStopping callback integration

**Rationale:** These knobs modify Lightning callback behavior, not simple Trainer kwargs. Requires callback factory or builder pattern. Estimated effort: ~4 hours (design + TDD + integration).

### Logger Backend Selection (1 field)
- **`--logger-backend`** (choices: 'mlflow', 'tensorboard', 'none') — Governance decision pending

**Rationale:** Current implementation uses `--disable-mlflow` inverse flag (line 395 in `train.py`). Switching to `--logger-backend` requires MLflow vs TensorBoard equivalence validation and Phase D governance approval. Estimated effort: ~2 hours (governance alignment + refactor).

### Advanced Training Knobs (2 fields)
- **`--scheduler`** (choices: 'step', 'plateau', 'cosine', 'none') — Requires LR scheduler factory
- **`--prefetch-factor`, `--persistent-workers`** (DataLoader performance knobs) — Not yet critical for MVP

**Rationale:** Scheduler selection requires factory logic to instantiate different `torch.optim.lr_scheduler` classes based on string input. Prefetch/persistent workers are optimization knobs with marginal impact on CPU workflows. Estimated effort: ~3 hours (scheduler factory + tests).

### Inference Post-Processing Knobs (2 fields)
- **`--middle-trim`** (crop pixels from reassembly edges) — Reassembly helper not yet parameterized
- **`--pad-eval`** (pad reconstructions for evaluation) — Evaluation alignment logic pending

**Rationale:** These knobs require upstream changes to `_reassemble_cdi_image_torch()` and evaluation pipeline. Deferred until Phase E (Ptychodus integration) clarifies requirements. Estimated effort: ~2 hours (helper refactor).

**Total Deferred Effort:** ~11 hours across 8 knobs + governance time.

---

## Risk Assessment

### High Risk
- **CLI test maintenance:** Subprocess-based CLI tests may be brittle (env isolation, file path dependencies). Mitigation: Use `tmp_path` fixtures, mock external dependencies (MLflow, file I/O).
- **Argparse namespace conflicts:** Adding `--num-workers` may conflict with future DataLoader flags if not namespaced. Mitigation: Prefix with `--exec-` (e.g., `--exec-num-workers`) if conflicts arise.

### Medium Risk
- **Hardcode elimination regression:** Removing `nphotons=1e9` override may expose latent divergence if factory default differs. Mitigation: Explicit test asserting `nphotons=1e9` preserved via factory default (C4.D2).
- **CONFIG-001 ordering violation:** Factory call after data loading would break params.cfg contract. Mitigation: Static analysis + runtime assertion (C4.C7), pytest guards (C4.D1).

### Low Risk
- **Documentation drift:** CLI examples in workflow guide may become stale. Mitigation: Automated doc generation (future enhancement), manual review during Phase D.

---

## Success Metrics

**Quantitative:**
- **7 CLI flags added** (5 training, 2 inference)
- **6 tests GREEN** (4 training, 2 inference)
- **271 tests passed** (no regressions vs C3 baseline)
- **4 docs updated** (workflow guide, spec, CLAUDE.md, implementation plan)
- **~80 lines refactored** (ad-hoc config → factory calls)

**Qualitative:**
- CLI flags follow consistent naming convention (`--snake-case`)
- Factory integration reduces duplication (single source of truth)
- CONFIG-001 compliance validated via pytest assertions
- Documentation examples accurate and reproducible

---

## Artifact Manifest

All planning artifacts stored under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/`:

- ✅ **plan.md** — This comprehensive execution plan (24 checklist items, 6 phases)
- ✅ **summary.md** — This planning summary (context, decisions, deferred work, risks)

**To Be Generated (Execution Phase):**
- [ ] `cli_flag_inventory.md` (C4.A1)
- [ ] `flag_selection_rationale.md` (C4.A2)
- [ ] `flag_naming_decisions.md` (C4.A3)
- [ ] `argparse_schema.md` (C4.A4)
- [ ] `red_baseline.md` (C4.B4)
- [ ] `pytest_cli_train_red.log`, `pytest_cli_inference_red.log` (C4.B3)
- [ ] `pytest_cli_train_green.log`, `pytest_cli_inference_green.log` (C4.D1)
- [ ] `pytest_factory_smoke.log`, `pytest_full_suite_c4.log` (C4.D2, C4.D3)
- [ ] `manual_cli_smoke.log` (C4.D4)
- [ ] `refactor_notes.md` (C4.C4)

**Storage Discipline:** Zero artifacts at repo root. All logs/docs under timestamped directory per hygiene policy.

---

## Next Steps

**Immediate (Engineer Execution Loop):**
1. Execute Phase C4.A tasks (design docs) — estimated 1 hour
2. Execute Phase C4.B tasks (RED tests) — estimated 1.5 hours
3. Execute Phase C4.C tasks (CLI refactor) — estimated 2 hours
4. Execute Phase C4.D tasks (GREEN validation) — estimated 1 hour
5. Execute Phase C4.E tasks (documentation) — estimated 0.5 hours
6. Execute Phase C4.F tasks (ledger close-out) — estimated 0.5 hours

**Total Estimated Effort:** ~6.5 hours (single engineer, focused execution)

**Phase D Preparation:**
- Review deferred knobs (9 fields, ~11 hours effort)
- Schedule governance decision on logger backend (MLflow vs TensorBoard)
- Design checkpoint callback factory pattern
- Prioritize scheduler selection vs DataLoader performance knobs

**Blocking Issues:** None. All dependencies resolved (Phase C3 GREEN, factory baseline GREEN, specs aligned).

---

## References

- **Plan Document:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md`
- **Phase C Execution Plan:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md`
- **Factory Design:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md`
- **Override Matrix:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md` §5
- **Phase C3 Summary:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T025643Z/phase_c3_workflow_integration/summary.md`
- **Explore Agent Outputs:** Embedded in planning process (training CLI: 12 flags, inference CLI: 10 flags, execution knobs: 25 total)
