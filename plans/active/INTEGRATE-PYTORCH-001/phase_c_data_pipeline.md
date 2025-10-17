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
| C.A1 | Summarize TensorFlow data pipeline contract | [ ] | Create `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/data_contract.md` capturing required structures from specs/data_contracts.md, specs/ptychodus_api_spec.md:§4, and docs/architecture.md:§3. Include keys, shapes, dtype, normalization, cache file naming. |
| C.A2 | Inventory PyTorch dataset outputs and gaps | [ ] | Analyze `ptycho_torch/dset_loader_pt_mmap.py` and related helpers; record actual tensors (names, shapes, devices) plus cache behaviour in `torch_gap_matrix.md`. Highlight mismatches (e.g., MemoryMappedTensor vs np.ndarray, missing RawData-level APIs). |
| C.A3 | Decide minimum fixture & ROI for tests | [ ] | Within `data_contract.md`, nominate a minimal NPZ fixture (ideally existing `datasets/fly/fly001_transposed.npz` subset) and specify ROI (sample count, gridsize). Note any preprocessing requirements and coordinate with TEST-PYTORCH-001. |

---

### Phase C.B — Torch-Optional Test Harness (Red Phase)
Goal: Author failing pytest cases that encode the desired RawDataTorch and PtychoDataContainerTorch behaviour without requiring torch at import time.
Prereqs: Phase C.A artifacts.
Exit Criteria: New pytest module(s) under `tests/torch/` with targeted selectors returning RED state artifacts (logs) stored under the same timestamped reports directory.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C.B1 | Blueprint test module structure | [ ] | Draft `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/test_blueprint.md` covering fixture strategy, torch optional guards (use `pytest.importorskip("torch")` inside tests only), and expected helpers (`make_raw_data_fixture()` delegating to TensorFlow RawData). |
| C.B2 | Author failing RawData parity test | [ ] | Create `tests/torch/test_data_pipeline.py::test_raw_data_torch_matches_tensorflow` asserting that grouped keys/values from new wrapper match TensorFlow RawData outputs for ROI. Capture red log via `pytest tests/torch/test_data_pipeline.py -k raw_data -vv`. |
| C.B3 | Author failing data-container test | [ ] | Extend same module with `test_data_container_shapes_and_dtypes` verifying `PtychoDataContainerTorch` exposes tensor attributes equivalent to `ptycho.loader.PtychoDataContainer`. Ensure tests run without torch installed by guarding creation logic. Store red log. |

---

### Phase C.C — Implementation & Bridging (Green Phase)
Goal: Implement torch-agnostic wrappers that reuse legacy code paths while preparing for torch acceleration when available.
Prereqs: Phase C.B red tests + ROI definition.
Exit Criteria: All new tests green against the wrappers; adapters documented; probe/coords caching parity validated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C.C1 | Implement `RawDataTorch` adapter | [ ] | Introduce module (location TBD, prefer `ptycho_torch/raw_data_bridge.py`) that wraps `ptycho.raw_data.RawData`, enforces params initialization, and exposes `generate_grouped_data()` delegating to legacy implementation. Ensure lazy torch imports (only inside tensor conversions) and document defaults referencing specs/ptychodus_api_spec.md:§4.3. |
| C.C2 | Implement `PtychoDataContainerTorch` | [ ] | Mirror TensorFlow container API (`X`, `Y`, `coords_nominal`, etc.) using torch tensors when available, NumPy fallback otherwise. Align dtype conversions with `ptycho/loader.py` logic; ensure outputs satisfy table defined in C.A1. |
| C.C3 | Bridge memory-mapped dataset usage | [ ] | Provide thin layer translating MemoryMappedTensor outputs into RawDataTorch inputs (or refactor dataset to delegate to RawData). Document decision in `reports/<ts>/implementation_notes.md` and ensure `.groups_cache.npz` reuse or new cache story with thresholds. |
| C.C4 | Update config bridge touchpoints | [ ] | Ensure new pipeline pulls configuration (N, gridsize, n_groups, neighbor_count, n_subsample) from dataclasses via existing adapter (`ptycho_torch/config_bridge.py`). Record callchain snippet in `implementation_notes.md`. |

---

### Phase C.D — Validation & Regression Guardrails
Goal: Prove parity through automated tests and portability checks, then capture canonical logs.
Prereqs: Phase C.C implementation.
Exit Criteria: Targeted pytest selectors green, cached artifacts recorded, and documentation updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C.D1 | Run targeted pytest selectors | [ ] | Execute `pytest tests/torch/test_data_pipeline.py -k "raw_data or data_container" -vv` with torch available and `PYTHONWARNINGS=default`. Save logs under `reports/<ts>/pytest_green.log`. |
| C.D2 | Verify cache reuse semantics | [ ] | Capture before/after timestamps or hash of `.groups_cache.npz` demonstrating cross-backend reuse; document in `reports/<ts>/cache_validation.md` with commands. |
| C.D3 | Update parity ledger & docs | [ ] | Refresh `plans/active/INTEGRATE-PYTORCH-001/implementation.md` Phase C rows to reflect completion, add summary bullet to `parity_map.md`, and note findings in docs/fix_plan.md Attempts History. |

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
