# Cross-Plan Overlap Audit — ADR-003 vs Upstream Initiatives
**Initiative:** ADR-003-BACKEND-API
**Phase:** A — Architecture Carve-Out
**Artifact:** Ownership & Responsibility Resolution
**Timestamp:** 2025-10-19T225905Z
**Reviewed Plans:** INTEGRATE-PYTORCH-001 (Phases C–E2), TEST-PYTORCH-001 (charter + implementation)

---

## 1. Ownership Resolution Table

| Topic | Existing Artifact | Current Owner | Status | Notes | ADR-003 Responsibility |
|-------|-------------------|---------------|--------|-------|------------------------|
| **CLI Implementation (Training)** | `ptycho_torch/train.py:366-520` (Phase E2.C1) | INTEGRATE-PYTORCH-001 | ✅ Complete | New interface (`cli_main()`) implemented with argparse, legacy interface preserved (`main()` L22-323). `--disable_mlflow` flag, path validation, probe-size inference wired. | **Phase B–C**: Refactor to use factory pattern; eliminate hardcoded overrides (L485-564). Current impl delegates directly to Lightning; ADR-003 must add execution config layer. |
| **CLI Implementation (Inference)** | `ptycho_torch/inference.py:293-572` (Phase E2.C2) | INTEGRATE-PYTORCH-001 | ✅ Complete | Dual-mode CLI (new Lightning path + legacy MLflow path) with checkpoint discovery logic (L413-429), NPZ validation, quiet mode. | **Phase C–D**: Refactor to delegate to inference workflow with execution config. Add missing flags per parity analysis (phase_vmin/vmax controls). |
| **Config Bridge Translation** | `ptycho_torch/config_bridge.py` (Phase B.B3) | INTEGRATE-PYTORCH-001 | ✅ Complete | Adapters `to_model_config()`, `to_training_config()`, `to_inference_config()` with critical transformations (grid_size→gridsize, mode→model_type, epochs→nepochs, K→neighbor_count, nll bool→float). | **Phase B**: Factory functions MUST USE config_bridge; no duplication. New factories will wrap bridge calls with override management. |
| **Lightning Trainer Wiring** | `ptycho_torch/workflows/components.py:_train_with_lightning()` | INTEGRATE-PYTORCH-001 | ✅ Complete | Lightning orchestration (Trainer init, deterministic mode, DDP strategy, checkpoint management) implemented in E2.C1–C3. | **Phase C**: Extract execution settings (device, strategy, deterministic, etc.) into `PyTorchExecutionConfig` dataclass. Current hardcoded values (deterministic=True L268, logger=False) should become configurable. |
| **MLflow Integration** | `ptycho_torch/train.py:516-519` (conditional autolog) | INTEGRATE-PYTORCH-001 | ✅ Complete | `--disable_mlflow` flag controls autolog suppression. Legacy `experiment_name`, `notes`, `model_name` in TrainingConfig L127-129. | **Phase B–C**: Move MLflow knobs to PyTorchExecutionConfig or document as "opt-in enhancement" vs canonical workflow (decision needed). TensorFlow has no equivalent; consider spec update. |
| **Persistence Contract (wts.h5.zip)** | `ptycho_torch/workflows/components.py:save_checkpoint_bundle()` | INTEGRATE-PYTORCH-001 | ✅ Complete | Checkpoint bundle created at `{output_dir}/wts.h5.zip` per Phase D4.C; compatible with TensorFlow artifact naming. Lightning `last.ckpt` stored in `checkpoints/` subdirectory. | **None (defer to future)**: Cross-backend loading not required for ADR-003 scope. Persistence format stable; no ADR-003 changes needed. |
| **Factory Patterns** | None (not implemented) | **→ ADR-003** | ❌ Not Started | **CRITICAL GAP**: No factory functions exist. PyTorch CLI constructs configs manually with scattered overrides (train.py L485-564). | **Phase B (FULL OWNERSHIP)**: ADR-003 MUST author `ptycho_torch/config_factory.py` with `create_training_config()`, `create_inference_config()` functions. Design documented in Phase B plan (`reports/<timestamp>/factory_design.md`). |
| **PyTorchExecutionConfig** | None | **→ ADR-003** | ❌ Not Started | **CRITICAL GAP**: No dataclass exists for backend-specific execution knobs (device, strategy, MLflow toggles, quiet mode, etc.). | **Phase C1 (FULL OWNERSHIP)**: ADR-003 MUST author dataclass in `ptycho_torch/config_params.py` or `ptycho/config/config.py` (location TBD). Catalog in `execution_knobs.md` provides schema requirements. |
| **Test Fixtures (Integration)** | `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` | TEST-PYTORCH-001 | ✅ Complete (Phase B3) | Minimal fixture (64 scan positions, stratified sampling) at `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`. SHA256: 6c2fbea0... Runtime: 14.53s (integration test). | **None**: ADR-003 reuses existing fixture. TEST-PYTORCH-001 owns fixture maintenance/updates. |
| **Parity Validation (Runtime)** | `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` | TEST-PYTORCH-001 | ✅ Complete (Phase D) | Runtime telemetry captured: 14.53s integration test, 3.82s fixture smoke test. Determinism verified via checkpoint embedding (Phase D1c). | **Phase D (CLI)**: ADR-003 smoke tests MUST validate that new factory-based CLI still passes integration test within 90s budget. |
| **CI Integration Hooks** | None (deferred in TEST-PYTORCH-001 Attempt #1) | **Shared (TEST-PYTORCH-001 + ADR-003)** | ⏳ Deferred | TEST-PYTORCH-001 identified CI integration as Phase E work (timeout=90s, retry=1, markers=integration+slow). ADR-003 CLI changes may affect CI selectors. | **Phase E (Advisory)**: ADR-003 MUST notify TEST-PYTORCH-001 if CLI refactor changes test invocation patterns (e.g., new required flags, different checkpoint paths). TEST-PYTORCH-001 owns CI pipeline updates. |
| **Documentation (docs/workflows/pytorch.md)** | `docs/workflows/pytorch.md` (Phase E3/F4 updates) | INTEGRATE-PYTORCH-001 | ✅ Complete | Sections §§2–6 (config setup, data loading, training workflow, checkpoints, inference), §12 (backend selection), §11 (regression test guidance) authored in Phases E3/F4. | **Phase D–E**: ADR-003 MUST update §§3–5 to document factory usage, PyTorchExecutionConfig fields, and new CLI examples. INTEGRATE-PYTORCH-001 handoff complete per `reports/2025-10-19T215800Z/phase_e3_docs_handoff/handoff_brief.md`. |
| **Spec Updates (ptychodus_api_spec.md)** | `specs/ptychodus_api_spec.md` §4.8 (backend selection) | INTEGRATE-PYTORCH-001 | ✅ Complete | Backend selection contract (CONFIG-001 compliance, routing guarantees, torch unavailability error messages) documented in Phase E1/F4. | **Phase E (Conditional)**: If PyTorchExecutionConfig introduces new fields consumed by Ptychodus reconstructor, ADR-003 MUST propose spec §4.9 addendum. If execution config is CLI/workflow-only (not exposed to Ptychodus), no spec change needed. |
| **ADR Documentation (docs/architecture/adr/ADR-003.md)** | ❌ **MISSING** | **→ ADR-003** | ❌ Not Started | `docs/architecture/adr/` directory does not exist. ADR-003 architectural decision rationale not formally documented. | **Phase B or Separate Governance Task**: MUST author ADR-003.md documenting two-layer architecture rationale (canonical vs execution config separation), factory pattern justification, and governance sign-off. Recommend authoring during Phase B alongside factory design doc to capture design decisions while fresh. |
| **Config Schema Maps** | `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md` | INTEGRATE-PYTORCH-001 | ✅ Reference Only | Field-by-field mapping of PyTorch → TensorFlow config translations (used to implement config_bridge). | **Phase B (Reference)**: ADR-003 factory design should cite this map for canonical field requirements. No ownership transfer; keep as reference artifact. |
| **Test Coverage (config_bridge.py)** | `tests/torch/test_config_bridge.py:1-162` | INTEGRATE-PYTORCH-001 | ✅ Complete | Enforces field translations (grid_size→gridsize, mode→model_type, etc.) and parity with TensorFlow dataclasses. | **Phase B3**: ADR-003 MUST extend these tests to cover factory functions. Add assertions for override enforcement, default value propagation, and invalid input handling. |
| **Dataloader Indexing (Follow-up)** | Issue `[INTEGRATE-PYTORCH-001-DATALOADER-INDEXING]` (Phase E2.D2) | INTEGRATE-PYTORCH-001 | ⚠️ Open Bug | IndexError at `ptycho_torch/dataloader.py:617` during integration test; neighbor indexing exceeds bounds. Tracked in `reports/2025-10-17T231500Z/parity_summary.md`. | **None (Out of Scope)**: ADR-003 does not fix dataloader bugs. If bug blocks Phase B factory testing, escalate to supervisor for INTEGRATE-PYTORCH-001 follow-up loop. |

---

## 2. ADR-003 Unique Responsibilities (Not Covered by Upstream Plans)

The following tasks are EXCLUSIVELY ADR-003 scope and have no overlap with completed INTEGRATE-PYTORCH-001 or TEST-PYTORCH-001 work:

| Responsibility | Rationale | Phase |
|----------------|-----------|-------|
| **Author `ptycho_torch/config_factory.py`** | No factory abstraction exists; PyTorch CLI uses manual config construction with scattered overrides. | Phase B2 |
| **Design + Implement `PyTorchExecutionConfig`** | Backend-specific knobs (device, strategy, MLflow, quiet mode) lack canonical home; currently mixed in TrainingConfig/CLI args. | Phase C1 |
| **Refactor CLI to delegate to factories** | Current CLI (train.py L366-520, inference.py L293-572) bypasses factory layer; must become thin wrappers. | Phase D |
| **Harmonize parameter naming** | Resolve `--max_epochs` vs `nepochs`, `--n_images` vs `n_groups` inconsistencies identified in parity analysis. | Phase D |
| **Add missing CLI flags** | Expose `--n_subsample`, `--subsample_seed`, `--sequential_sampling` per execution_knobs.md catalog. | Phase D |
| **Author ADR-003.md governance doc** | Capture architectural rationale for two-layer config pattern and factory introduction. | Phase B or E2 |
| **Deprecate legacy API (`ptycho_torch/api/`)** | Phase E1 task per implementation.md; mark modules deprecated or refactor to delegate to new workflows. | Phase E1 |

---

## 3. Prerequisites & Dependencies for ADR-003 Phase B

ADR-003 Phase B (factory design + implementation) depends on the following artifacts from upstream plans:

| Prerequisite | Source Plan | Artifact Path | Status | Notes |
|-------------|-------------|---------------|--------|-------|
| **Config Bridge Adapters** | INTEGRATE-PYTORCH-001 | `ptycho_torch/config_bridge.py` | ✅ Available | Factories MUST call `to_model_config()`, `to_training_config()`, `to_inference_config()` instead of reimplementing translation logic. |
| **CLI Flag Inventory** | ADR-003 Phase A (this loop) | `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/cli_inventory.md` | ✅ Available (current loop) | Defines required/optional flags, target config fields, and validation rules for factory inputs. |
| **Execution Knobs Catalog** | ADR-003 Phase A (this loop) | `execution_knobs.md` | ✅ Available (current loop) | Specifies PyTorchExecutionConfig schema (54 knobs cataloged). |
| **Integration Test Fixture** | TEST-PYTORCH-001 | `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` | ✅ Available | Smoke test for factory-based CLI changes; validates train→save→load→infer cycle. |
| **Parity Test Baseline** | TEST-PYTORCH-001 | `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` | ✅ Available | Factory refactor must maintain <90s integration test runtime (current: 14.53s). |
| **PyTorch Workflow Guidance** | INTEGRATE-PYTORCH-001 | `docs/workflows/pytorch.md` §§2–12 | ✅ Available | Factories must align with documented workflow patterns (config setup, data loading, checkpoint management). |
| **Backend Selection Contract** | INTEGRATE-PYTORCH-001 | `specs/ptychodus_api_spec.md` §4.8 | ✅ Available | Factories must honor CONFIG-001 compliance (call `update_legacy_dict()` before dispatch). |

**Blockers:** None identified. All prerequisites are available.

---

## 4. Unclear Ownership / Ambiguous Responsibilities

| Topic | Ambiguity | Recommendation |
|-------|-----------|----------------|
| **MLflow vs Canonical Workflow** | TrainingConfig includes MLflow fields (`experiment_name`, `notes`, `model_name`), but TensorFlow stack has no equivalent. Unclear if MLflow is "canonical" or PyTorch-specific enhancement. | **ADR-003 Phase B**: Treat MLflow as PyTorch-specific; move fields to PyTorchExecutionConfig. Document as "opt-in experiment tracking" in Phase E docs. If Ptychodus requires MLflow integration, escalate to spec update. |
| **CLI Naming Harmonization Timeline** | Parity analysis identifies `--max_epochs` vs `nepochs`, `--n_images` vs `n_groups` inconsistencies. Unclear if ADR-003 should fix immediately (Phase D) or defer to future TensorFlow CLI enhancement. | **ADR-003 Phase D**: Fix PyTorch CLI only (rename `--max_epochs` → `--nepochs`, `--n_images` → `--n_groups`). Document TensorFlow CLI gap in phase_a_inventory/cli_inventory.md §3.4. Propose TensorFlow backport in separate initiative. |
| **Cross-Backend Checkpoint Loading** | specs/ptychodus_api_spec.md §4.8 mentions "cross-backend artifact loading is OPTIONAL" but doesn't specify ownership for implementation. | **Defer to future initiative**: ADR-003 scope is PyTorch backend standardization only. If Ptychodus requires loading TensorFlow `.h5.zip` in PyTorch path (or vice versa), create separate [CROSS-BACKEND-COMPAT] initiative. |
| **CI Integration Coordination** | TEST-PYTORCH-001 plans CI integration (Phase E); ADR-003 CLI refactor may affect test selectors. Unclear which plan owns CI pipeline updates. | **Shared Responsibility**: ADR-003 Phase E MUST notify TEST-PYTORCH-001 of CLI changes (new flags, checkpoint path changes). TEST-PYTORCH-001 owns CI `.yml` updates. Document handoff in Phase E summary. |

---

## 5. Missing ADR-003 Governance Documentation

**Finding:** `docs/architecture/adr/ADR-003.md` does not exist. The `docs/architecture/adr/` directory is missing entirely.

**Impact:**
- Architectural rationale for two-layer config pattern (canonical vs execution) is undocumented
- Factory pattern introduction lacks formal justification
- No governance sign-off trail for backend API standardization decision

**Recommendation:**
1. **Create ADR directory:** `mkdir -p docs/architecture/adr/`
2. **Author ADR-003.md during Phase B** (alongside factory design doc) to capture:
   - Problem statement (scattered config construction, hardcoded overrides, lack of execution config abstraction)
   - Decision: Two-layer architecture (canonical dataclasses + PyTorchExecutionConfig)
   - Alternatives considered (single mega-config, env-var-only controls, CLI-only approach)
   - Consequences (code organization benefits, testing improvements, Ptychodus integration impact)
   - Governance status (Proposed → Accepted after Phase E validation)
3. **Reference ADR-003.md from:**
   - `docs/architecture.md` (add ADR index section)
   - `specs/ptychodus_api_spec.md` §4 (link to backend config architecture)
   - `docs/workflows/pytorch.md` §3 (configuration setup section)

**Authoring Timeline:**
- **Option A (Recommended):** Author in Phase B alongside factory design doc (`reports/<timestamp>/factory_design.md`) to capture design decisions while implementing
- **Option B:** Defer to Phase E2 as separate governance task after full implementation complete (risk: rationale details forgotten)

**Responsibility:** ADR-003 initiative (mandatory deliverable for Phase E governance sign-off)

---

## 6. Follow-Up Actions for Phase B

Based on this overlap audit, ADR-003 Phase B MUST:

1. **Reuse config_bridge.py adapters** — Do NOT reimplement translation logic; call existing `to_*_config()` functions from factory layer
2. **Cite Phase A inventories** — Factory design doc should reference `cli_inventory.md` (required/optional flags), `execution_knobs.md` (PyTorchExecutionConfig schema)
3. **Extend test_config_bridge.py** — Add factory coverage to existing test module (Phase B3 checklist item)
4. **Document MLflow positioning** — Decide whether MLflow fields belong in PyTorchExecutionConfig or remain in canonical TrainingConfig (recommendation: execution config)
5. **Plan ADR-003.md authoring** — Schedule governance doc creation (Phase B recommended, Phase E2 latest)
6. **Coordinate with TEST-PYTORCH-001** — Ensure integration test still passes after factory refactor; notify if checkpoint paths or CLI signatures change
7. **Prepare docs/workflows/pytorch.md updates** — Draft §3–§5 revisions showing factory usage examples (execute in Phase D–E)

**No Blockers Identified:** All prerequisites available; Phase B can proceed immediately after Phase A artifacts finalized.

---

## 7. References

**Upstream Plans Reviewed:**
- `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` — Phase C (CLI), Phase D (parity evidence)
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T225500Z/phase_e_closeout/closure_summary.md` — Phase E handoff
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/handoff_brief.md` — Docs transfer notes
- `plans/active/TEST-PYTORCH-001/implementation.md` — Test plan phases (charter → phased plan conversion)
- `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` — Parity validation

**Specification Sources:**
- `specs/ptychodus_api_spec.md` §4.8 — Backend selection contract
- `docs/workflows/pytorch.md` §§2–12 — Workflow guidance
- `docs/findings.md` POLICY-001 — PyTorch mandatory dependency policy

**Current Phase A Artifacts:**
- `cli_inventory.md` — CLI flag mappings + parity gaps
- `execution_knobs.md` — PyTorch execution knob catalog
- `plan.md` — Phase A task breakdown (A1–A3)

**Next:** Produce `summary.md` documenting Phase A outcomes and recommendations for Phase B.
