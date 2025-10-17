# Configuration Schema Mapping — Scope Notes

**Initiative:** INTEGRATE-PYTORCH-001 Phase B.B1
**Date:** 2025-10-17
**Artifact Companion:** `config_schema_map.md`

---

## Scope Definition

This document captures observations, unresolved questions, and scope decisions arising from the Phase B.B1 configuration schema audit. It complements the detailed field mapping table in `config_schema_map.md`.

---

## Key Findings

### 1. Architectural Divergence: Singleton vs Dataclass Pattern

**PyTorch Implementation:**
- Uses **four separate singleton-style dataclasses** (`DataConfig`, `ModelConfig`, `TrainingConfig`, `InferenceConfig`)
- No hierarchical nesting (flat structure)
- Many fields are PyTorch-specific (attention mechanisms, Lightning orchestration, MLflow tracking)
- Configuration mutation via `update_existing_config()` helper function

**TensorFlow Implementation:**
- Uses **three nested dataclasses** (`ModelConfig`, `TrainingConfig`, `InferenceConfig`)
- Hierarchical: `TrainingConfig.model` and `InferenceConfig.model` nest `ModelConfig`
- Minimal, spec-aligned fields only
- Immutability enforced (frozen dataclasses)
- Translation to legacy `params.cfg` via `update_legacy_dict()`

**Implication:** Direct structural compatibility is impossible. Translation layer required regardless of refactor vs dual-schema decision.

---

### 2. Critical Field Gaps Blocking Integration

**High-Priority Missing Fields (Phase B MVP Scope):**

1. **Data Pipeline Blockers:**
   - `train_data_file` / `test_data_file` → PyTorch uses `training_directories: List[str]` instead
   - `model_path` → Critical for `load_inference_bundle()` workflow
   - Impact: Cannot invoke reconstructor lifecycle methods without these

2. **Grouping & Sampling Blockers:**
   - `n_groups` → Core parameter for `RawData.generate_grouped_data()`
   - `neighbor_count` → K-NN search width (PyTorch has `K`, `K_quadrant` with unclear mapping)
   - Impact: Data pipeline shape mismatches if not synchronized

3. **Legacy Bridge Blockers:**
   - `output_dir` → Maps to `params.cfg['output_prefix']` via KEY_MAPPINGS
   - `positions_provided`, `probe_trainable` → Legacy flags consumed by model construction
   - Impact: Silent failures in legacy module initialization (per CONFIG-001 finding)

**Recommendation:** Phase B.B3 must implement these 9 fields minimum before proceeding to Phase C data pipeline work.

---

### 3. Semantic Collisions & Naming Conflicts

**Identified Conflicts Requiring Clarification:**

| Conflict | PyTorch | TensorFlow | Resolution Path |
|----------|---------|------------|-----------------|
| Neighbor count | `K: int = 6` (DataConfig)<br>`K_quadrant: int = 30` (DataConfig) | `neighbor_count: int = 4` (TrainingConfig/InferenceConfig) | **Q:** Are these the same semantic parameter? If yes, which default is correct? If no, what does each control? |
| Subsampling | `n_subsample: int = 7` (DataConfig) — "Subsampling factor for coordinates" | `n_subsample: Optional[int]` (TrainingConfig) — "Number of images to subsample before grouping" | **Q:** Different semantics (factor vs count). Which is authoritative? How do they interact? |
| Gridsize | `grid_size: Tuple[int, int] = (2, 2)` | `gridsize: int = 1` | **Q:** Does TensorFlow support non-square grids? If no, translation is `gridsize = grid_size[0]` with validation `grid_size[0] == grid_size[1]`. |
| Loss specification | `loss_function: Literal['MAE', 'Poisson']` (categorical choice) | `mae_weight: float = 0.0`<br>`nll_weight: float = 1.0` (weighted sum) | **Q:** TensorFlow supports mixed losses. How to translate PyTorch categorical choice? |

**Action Required:** Phase B.B2 must answer these questions before test implementation. Document decisions in this file.

---

### 4. Default Value Conflicts

**Critical Default Mismatches:**

1. **`nphotons`:**
   - PyTorch: `1e5` (100,000 photons)
   - TensorFlow: `1e9` (1 billion photons)
   - **Impact:** 4 orders of magnitude difference affects Poisson noise model and intensity scaling
   - **Decision:** Which is correct for ptychodus workflows? Must reconcile before MVP.

2. **`probe_scale`:**
   - PyTorch: `1.0`
   - TensorFlow: `4.0`
   - **Impact:** Affects probe normalization in `ptycho/probe.py:63`
   - **Decision:** Document rationale for each default; choose one for harmonized config.

3. **`amp_activation`:**
   - PyTorch: `'silu'` (freeform string)
   - TensorFlow: `'sigmoid'` (constrained Literal)
   - **Impact:** Different activation functions produce different reconstructions
   - **Decision:** Add validation layer or expand TensorFlow Literal to include 'silu'.

**Recommendation:** Capture default rationale from PyTorch development history (check commit messages, `docs/workflows/pytorch.md`) before finalizing harmonization strategy.

---

## Unresolved Questions (Cross-Reference with Open Questions Q1-Q8)

### Q1: Refactor vs Dual-Schema Strategy

**Context:** Stakeholder brief Question 1 (`plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/open_questions.md`).

**Evidence from This Audit:**

**Arguments for Refactor (Option A):**
- 12 missing spec-required fields in PyTorch → large translation burden
- KEY_MAPPINGS automatically available (no duplication)
- Single source of truth reduces maintenance drift
- Spec alignment enforced by dataclass validators

**Arguments for Dual-Schema (Option B):**
- 30+ PyTorch-specific fields (attention, multi-stage training, MLflow) don't belong in shared config
- Preserves PyTorch architectural independence
- Avoids breaking existing PyTorch `ptycho_torch/` consumers
- Allows gradual migration vs big-bang refactor

**Recommended Decision:** **Hybrid Approach**
1. PyTorch imports and **extends** TensorFlow dataclasses (inheritance)
2. Shared fields use TensorFlow names/types (spec-compliant)
3. PyTorch-specific fields added as optional extensions
4. Single `update_legacy_dict()` implementation with extended KEY_MAPPINGS

**Example:**
```python
from ptycho.config.config import ModelConfig as TFModelConfig

@dataclass
class ModelConfig(TFModelConfig):
    """PyTorch-extended model configuration."""
    # Inherit all TensorFlow fields (N, gridsize, model_type, etc.)

    # PyTorch-specific extensions
    cbam_encoder: bool = False
    eca_decoder: bool = False
    batch_norm: bool = False
    # ... (30+ additional fields)
```

**Next Action:** Document this decision in Phase B.B2 brief; update implementation plan.

---

### Q2: MVP vs Full Parity Field Scope

**Context:** Stakeholder brief Question 2.

**MVP Scope (Minimum 9 Fields for Basic Integration):**

Based on reconstructor contract (`specs/ptychodus_api_spec.md §4.2-4.6`):

**Lifecycle Essentials (3):**
- `model_path` — `open_model()` requirement
- `train_data_file` — `train()` requirement
- `test_data_file` — `reconstruct()` requirement

**Data Grouping Essentials (2):**
- `n_groups` — `generate_grouped_data()` requirement
- `neighbor_count` — K-NN search width

**Model Essentials (4):**
- `N` — tensor shape foundation
- `gridsize` — channel count determinant
- `model_type` — workflow selector
- `nphotons` — Poisson loss scaling

**Total:** 9 fields enable smoke test (train → save → load → infer) without silent failures.

**Full Parity Scope (75+ Fields):**
All fields documented in `specs/ptychodus_api_spec.md §5.1-5.3`. See `config_schema_map.md` for complete inventory.

**Recommendation:** Phase B.B3 implements MVP scope only. Phase B.B4 extends to full parity incrementally with parameterized tests.

---

### Q3: API Layer Integration Strategy

**Context:** Stakeholder brief Question 3 (Delta 2).

**Observation from Schema Audit:**
- PyTorch `api/base_api.py` provides `ConfigManager` class that wraps singleton configs
- `ConfigManager` exposes methods: `update_config()`, `get_config()`, `validate_config()`
- This layer is **above** raw dataclasses; not visible in reconstructor contract

**Implication for Config Bridge:**
- If using API layer: `ConfigManager` must translate to `update_legacy_dict()` internally
- If bypassing API: Direct dataclass instantiation → `update_legacy_dict()` (matches TensorFlow pattern)

**Recommendation:** Bypass API layer for config bridge (use direct dataclass pattern). Reserve API layer for training orchestration (Phase E). This keeps Phase B scope minimal and aligned with TensorFlow precedent.

**Next Action:** Document decision; update Phase C/D/E tasks accordingly.

---

## Scope Boundaries for Phase B.B3

### In-Scope: Configuration Schema Harmonization

**Tasks:**
1. Choose refactor strategy (recommended: hybrid inheritance approach above)
2. Implement missing 9 MVP fields in PyTorch config
3. Create translation layer for 6 type mismatches (see `config_schema_map.md` "🔄" rows)
4. Extend KEY_MAPPINGS with PyTorch-specific dotted keys
5. Write `update_legacy_dict()` tests for MVP fields

**Out-of-Scope for Phase B:**
- Full 75+ field implementation (defer to Phase B.B4)
- PyTorch-specific field semantics validation (30+ fields like attention toggles)
- DataConfig / DatagenConfig integration (defer to Phase C data pipeline work)
- MLflow / Lightning config integration (defer to Phase E training work)

---

## Recommendations for Phase B.B2 (Failing Test Implementation)

**Test Structure:**
```python
# tests/torch/test_config_bridge.py

def test_pytorch_config_populates_params_cfg_mvp_fields():
    """MVP config bridge test for 9 critical fields."""
    # 1. Create PyTorch config with MVP fields
    config = TrainingConfig(
        model=ModelConfig(N=128, gridsize=2, model_type='pinn'),
        train_data_file=Path('train.npz'),
        test_data_file=Path('test.npz'),
        n_groups=512,
        neighbor_count=7,
        nphotons=1e9,
    )

    # 2. Call bridge (expected to fail currently)
    from ptycho.config.config import update_legacy_dict
    import ptycho.params as params
    update_legacy_dict(params.cfg, config)

    # 3. Assert params.cfg populated correctly
    assert params.cfg['N'] == 128
    assert params.cfg['gridsize'] == 2
    assert params.cfg['model_type'] == 'pinn'
    assert params.cfg['train_data_file_path'] == 'train.npz'
    assert params.cfg['test_data_file_path'] == 'test.npz'
    assert params.cfg['n_groups'] == 512
    assert params.cfg['neighbor_count'] == 7
    assert params.cfg['nphotons'] == 1e9
    # ... (9 total assertions)
```

**Expected Failure Modes:**
1. `ImportError` if PyTorch config doesn't have these fields yet
2. `KeyError` if KEY_MAPPINGS missing PyTorch-specific translations
3. `AssertionError` on field value mismatches (e.g., `gridsize` tuple→int conversion)

**TDD Flow:**
- Write test (fails) → Implement MVP fields → Implement translation → Test passes
- Iterate for each MVP field batch (3-4 fields per commit)

---

## Open Action Items

**Before Phase B.B3 Implementation:**
1. Resolve Q1 (refactor strategy) — recommend hybrid inheritance approach documented above
2. Resolve Q2 (MVP scope) — 9 fields documented above
3. Resolve Q3 (API layer) — bypass for config bridge, use for orchestration
4. Answer semantic collision questions (neighbor_count, n_subsample, gridsize) — requires PyTorch dev consultation or spec clarification
5. Document default value rationale (nphotons, probe_scale, amp_activation) — check PyTorch commit history

**Phase B.B2 Deliverables:**
- `tests/torch/test_config_bridge.py` with failing MVP test (estimated: 50 lines)
- Decision brief documenting Q1-Q3 resolutions (estimated: 1 page)

**Phase B.B3 Deliverables:**
- Updated `ptycho_torch/config_params.py` with 9 MVP fields (or refactored to inherit from TensorFlow)
- Extended KEY_MAPPINGS in `ptycho/config/config.py` (or new `ptycho_torch/config_bridge.py`)
- Passing MVP config bridge test

---

## Risk Assessment

**High Risk:**
- **Semantic collision resolution delayed:** If neighbor_count/K mapping unclear, data grouping will fail silently (shape mismatches). **Mitigation:** Block Phase B.B3 until answered.
- **Default value conflicts unresolved:** nphotons 4-order-of-magnitude difference could invalidate physics model. **Mitigation:** Document rationale, add validation warnings.

**Medium Risk:**
- **PyTorch-specific field proliferation:** 30+ fields with no TensorFlow equivalent complicates maintenance. **Mitigation:** Isolate in separate config classes if using refactor approach.
- **API layer bypass regret:** If API layer provides critical abstractions, bypassing may require rework in Phase E. **Mitigation:** Document decision rationale; plan API integration for orchestration only.

**Low Risk:**
- **Type mismatches:** 6 known mismatches are straightforward to translate (tuple→int, bool→float, enum mapping). **Mitigation:** Unit test each conversion.

---

## References

- Configuration schema mapping (detailed): `config_schema_map.md` (same directory)
- Stakeholder brief: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/stakeholder_brief.md`
- Open questions tracker: `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/open_questions.md`
- Spec contract: `specs/ptychodus_api_spec.md §5`
- TensorFlow config: `ptycho/config/config.py`
- PyTorch config: `ptycho_torch/config_params.py`
- Legacy bridge gotchas: `docs/findings.md` (CONFIG-001)

---

**Next Loop:** Phase B.B2 — Implement failing test based on MVP scope and recommendations above.
