# Phase C — Data Pipeline Parity Plan (INTEGRATE-PYTORCH-001)

## Context
- Initiative: INTEGRATE-PYTORCH-001
- Phase Goal: Deliver a torch-optional data pipeline that consumes canonical NPZ datasets, reuses the legacy grouping semantics, and presents model-ready tensors equivalent to the TensorFlow `PtychoDataContainer` contract.
- Dependencies: specs/data_contracts.md (NPZ schema); specs/ptychodus_api_spec.md (§4 data ingestion, §5.2 training config fields); docs/architecture.md (§3 data loading pipeline); docs/DEVELOPER_GUIDE.md (§3 data pipeline, §10 params lifecycle); plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/parity_map.md (Gap #2 details); plans/pytorch_integration_test_plan.md (fixture + runtime constraints).
- Coordination: TEST-PYTORCH-001 for shared fixtures; INTEGRATE-PYTORCH-000 stakeholder brief (Delta 2 data pipeline) for governance cues.

---

### Phase C.A — Canonical Data Contract Alignment
Goal: Establish an authoritative baseline for required keys, shapes, dtypes, and caching semantics before writing tests.
Prereqs: None (review dependencies above).
Exit Criteria: Artifact pair `data_contract.md` + `torch_gap_matrix.md` under a timestamped reports directory documenting TensorFlow expectations vs PyTorch behaviour.

| ID | Task Description | State | How/Why & Guidance (including API / document / artifact / source file references) |
| --- | --- | --- | --- |
| C.A1 | Summarize TensorFlow data pipeline contract | [x] | ✅ 2025-10-17 — See `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md` for canonical keys, shapes, dtypes, normalization, and cache naming derived from specs/data_contracts.md §1, specs/ptychodus_api_spec.md §4, and docs/architecture.md §3. |
| C.A2 | Inventory PyTorch dataset outputs and gaps | [x] | ✅ 2025-10-17 — Findings captured in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/torch_gap_matrix.md`, enumerating current `ptycho_torch/dset_loader_pt_mmap.py` tensors, cache behaviour, and divergences from the TensorFlow contract. |
| C.A3 | Decide minimum fixture & ROI for tests | [x] | ✅ 2025-10-17 — ROI defined in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md` §7 using the `datasets/fly/fly001_transposed.npz` subset, with preprocessing notes and TEST-PYTORCH-001 coordination items. |

---

### Phase C.B — Torch-Optional Test Harness (Red Phase)
Goal: Author failing pytest cases that encode the desired RawDataTorch and PtychoDataContainerTorch behaviour without requiring torch at import time.
Prereqs: Phase C.A artifacts.
Exit Criteria: New pytest module(s) under `tests/torch/` with targeted selectors returning RED state artifacts (logs) stored under the same timestamped reports directory.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C.B1 | Blueprint test module structure | [x] | ✅ 2025-10-17 — Torch-optional harness blueprint documented in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/test_blueprint.md`, covering fixtures, skip guards, and ROI guidance. |
| C.B2 | Author failing RawData parity test | [x] | Completed 2025-10-17 — see `reports/2025-10-17T071836Z/pytest_raw_data_red.log` capturing red-phase failure for `test_raw_data_torch_matches_tensorflow`. |
| C.B3 | Author failing data-container test | [x] | Completed 2025-10-17 — see `reports/2025-10-17T071836Z/pytest_data_container_red.log` documenting failing coverage for `test_data_container_shapes_and_dtypes` + Y dtype check. |

---

### Phase C.C — Implementation & Bridging (Green Phase)
Goal: Implement torch-agnostic wrappers that reuse legacy code paths while preparing for torch acceleration when available.
Prereqs: Phase C.B red tests + ROI definition.
Exit Criteria: All new tests green against the wrappers; adapters documented; probe/coords caching parity validated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C.C1 | Implement `RawDataTorch` adapter | [x] | ✅ 2025-10-17 — Implemented `ptycho_torch/raw_data_bridge.py` (324 lines) with delegation strategy. Wraps `ptycho.raw_data.RawData.from_coords_without_pc()` + `generate_grouped_data()`. Config bridge integration via constructor (auto `update_legacy_dict()` call). NumPy-first returns. Test green: `pytest -k raw_data -vv` → 1/1 PASSED. Artifacts: `reports/2025-10-17T073640Z/{pytest_raw_data_green.log,implementation_notes.md}`. Full regression: 184 passed, 0 new failures. |
| C.C2 | Implement `PtychoDataContainerTorch` | [x] | ✅ 2025-10-17 — Green artifacts `reports/2025-10-17T080500Z/{summary.md,pytest_data_container_green.log,pytest_y_dtype_green.log}` document torch-optional container parity (13 attributes, DATA-001 enforcement). |
| C.C3 | Bridge memory-mapped dataset usage | [x] | ✅ 2025-10-17 — Implemented `ptycho_torch/memmap_bridge.py`; see `reports/2025-10-17T084500Z/{implementation_strategy.md,pytest_memmap_green_final.log}` for adapter design + green selectors. |
| C.C4 | Update config bridge touchpoints | [x] | ✅ 2025-10-17 — Memmap bridge constructor requires dataclass configs and delegates through RawDataTorch (`config_bridge` auto-sync). Evidence: `reports/2025-10-17T084500Z/implementation_strategy.md` (Implementation Details). |

---

### Phase C.D — Validation & Regression Guardrails
Goal: Prove parity through automated tests and portability checks, then capture canonical logs.
Prereqs: Phase C.C implementation.
Exit Criteria: Targeted pytest selectors green, cached artifacts recorded, and documentation updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C.D1 | Run targeted pytest selectors | [x] | ✅ 2025-10-17 — Selector `pytest tests/torch/test_data_pipeline.py -k "memmap" -vv` captured in `reports/2025-10-17T084500Z/pytest_memmap_green_final.log` (delegation + deterministic generation). |
| C.D2 | Verify cache reuse semantics | [x] | ✅ 2025-10-17 — Documented cache-free deterministic behaviour in `reports/2025-10-17T084500Z/cache_semantics.md`; replaces `.groups_cache` check with deterministic-generation validation. |
| C.D3 | Update parity ledger & docs | [x] | ✅ 2025-10-17 — Updated `parity_map.md` §10A with comprehensive memmap bridge parity summary (delegation strategy, cache-free architecture, test results, spec compliance). Refreshed `implementation.md` C5 row to `[x]` with artifact links. Logged Attempt #39 in `docs/fix_plan.md`. Artifacts: `reports/2025-10-17T084246Z/parity_map_update.md`. |

---

### Phase C.E — Documentation & Hand-off
Goal: Ensure future phases (D/E) have clear entry points and that TEST-PYTORCH-001 can reuse fixtures.
Prereqs: Phase C.D results.
Exit Criteria: Updated workflow docs + recorded risks.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C.E1 | Document integration touchpoints | [ ] | Append section to `plans/pytorch_integration_test_plan.md` describing new RawDataTorch + DataContainer usage, including fixture paths and torch-optional guidance. |
| C.E2 | Log residual risks & TODOs | [ ] | Create `reports/<ts>/residual_risks.md` capturing any deferred torch-specific optimizations, memmap limitations, or dataset size assumptions. |
| C.E3 | Handoff to Workflow Phase D | [ ] | Update `input.md` and parity plans with clear next-step instructions for orchestrating training workflow refactor once data pipeline parity validated. |

---

**Reporting Conventions:** Replace `<ts>` with ISO8601 UTC timestamps per loop. Store red/green pytest logs, notebooks, or scripts under the same timestamped directory. Reference each artifact from docs/fix_plan.md attempts.

**Testing Philosophy:** Maintain torch-optional tests—design fixtures so they default to NumPy arrays and only import torch within guarded blocks. Provide explicit skip messages when torch unavailable, and capture both skip and green logs to demonstrate parity.
