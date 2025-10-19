# Phase A Inventory — Loop Summary
**Initiative:** ADR-003-BACKEND-API
**Phase:** A — Architecture Carve-Out & Inventory
**Loop Timestamp:** 2025-10-19T225905Z (execution by ralph)
**Mode:** Docs-only (no code changes, no test execution)
**Outcome:** ✅ All Phase A tasks complete (A1.a–A3.c)

---

## 1. Deliverables Produced

| Artifact | Size | Key Findings |
|----------|------|--------------|
| **cli_inventory.md** | 21KB | 9 PyTorch training flags, 10 inference flags. **5 critical semantic mismatches** (epoch naming, activation defaults, neighbor count, probe size, n_groups vs n_images). **11 feature gaps** total (6 in PyTorch, 5 in TensorFlow). |
| **execution_knobs.md** | 265 lines | **54 unique parameters** across 11 categories. **35 knobs** recommended for `PyTorchExecutionConfig`. **9 hardcoded values** requiring CLI exposure. |
| **overlap_notes.md** | 17KB | **15 topics** audited. **7 complete** (CLI, config bridge, Lightning, MLflow, persistence, fixtures, parity). **2 critical gaps**: factory patterns + PyTorchExecutionConfig (FULL ADR-003 ownership). **1 governance gap**: ADR-003.md missing. **No blockers** for Phase B. |
| **logs/a1_cli_flags.txt** | 1.2KB | Raw grep output (19 add_argument calls) for traceability. |

---

## 2. Key Discoveries

### CLI Parity Gaps (Critical for Phase D)
**Semantic Mismatches:** Epoch naming (`--max_epochs` vs `nepochs`), activation defaults (`silu` vs `sigmoid`), neighbor count (K=6 vs K=4), probe size (auto-infer vs config), data grouping terminology (`n_images` vs `n_groups`).

**Missing in PyTorch:** `--n_subsample`, `--subsample_seed`, `--sequential_sampling`, `--phase_vmin/vmax`.

**Missing in TensorFlow:** `--device`, `--gridsize`, `--batch_size` (CLI exposure), `--disable_mlflow`, `--quiet`.

### Execution Knobs Catalog
**High-Priority:** `learning_rate`, `accelerator`, `num_workers`, `early_stop_patience`, `scheduler_type` (all hardcoded, easy to add).

**Backend-Specific:** Lightning (9 knobs), distributed (4), data loading (7), checkpointing (5), inference (4).

### Ownership Resolution
**INTEGRATE-PYTORCH-001 Complete:** CLI implementation, config bridge, Lightning wiring, MLflow integration, persistence.

**ADR-003 Exclusive:** Factory patterns (absent), PyTorchExecutionConfig (absent), ADR-003.md governance doc (absent).

**No Blockers:** All prerequisites available for Phase B.

---

## 3. Follow-Up Questions / Blockers

**None.** All clarifications captured in artifacts:
- MLflow positioning → recommend PyTorchExecutionConfig (overlap_notes.md §4)
- CLI naming harmonization → fix PyTorch only, defer TF (overlap_notes.md §4)
- ADR-003.md authoring → recommend Phase B alongside factory design (overlap_notes.md §5)

---

## 4. Suggested Priorities for Phase B

| Priority | Task | Dependencies |
|----------|------|--------------|
| **P0** | Author `PyTorchExecutionConfig` dataclass | execution_knobs.md catalog |
| **P0** | Design factory functions (`factory_design.md`) | cli_inventory.md, PyTorchExecutionConfig |
| **P1** | Implement `config_factory.py` | factory_design.md, config_bridge.py |
| **P1** | Extend `test_config_bridge.py` for factories | config_factory.py |
| **P2** | Author ADR-003.md governance doc | factory_design.md |

**Execution Order:** B0 (define PyTorchExecutionConfig) → B1 (factory_design.md) → B2 (config_factory.py) → B3 (tests) → B_gov (ADR-003.md).

---

## 5. Phase A Completion Checklist

- [x] A1.a: CLI flags captured to logs/a1_cli_flags.txt
- [x] A1.b-c: Mapped to config fields, compared with TensorFlow
- [x] A2.a-c: Cataloged 54 PyTorch execution knobs
- [x] A3.a-c: Cross-plan overlap audit (15 topics, 0 blockers)
- [x] All artifacts stored under phase_a_inventory/
- [x] summary.md populated with discoveries + Phase B priorities

**Exit Criteria Met:** ✅ Phase A complete.

---

## 6. Handoff to Phase B

**Ready to Proceed:** No prerequisites missing.

**Critical Inputs:**
- cli_inventory.md — Required/optional flags, validation rules
- execution_knobs.md — PyTorchExecutionConfig schema (54 knobs → 35 recommended)
- overlap_notes.md — Confirms factories are greenfield ADR-003 responsibility

**Recommended Start:** Read execution_knobs.md §§1–5 → draft PyTorchExecutionConfig schema → author factory_design.md → implement with TDD.

**Next Loop Mode:** TDD mode (test-first) for Phase B.
