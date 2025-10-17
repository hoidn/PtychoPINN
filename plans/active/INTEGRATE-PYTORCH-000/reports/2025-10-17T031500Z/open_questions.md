# Open Questions for Governance Review

**Initiative:** INTEGRATE-PYTORCH-000 Phase C
**Date:** 2025-10-17
**Purpose:** Track architectural decisions blocking critical-path execution

---

## Critical Path Blockers (Resolve Immediately)

### Q1: Configuration Schema Strategy
**Question:** Should PyTorch configuration be refactored to use shared TensorFlow dataclasses, or should we maintain dual schemas with explicit translation layer?

**Impact:** Affects maintainability burden, test coverage scope, and Phase B implementation complexity

**Options:**
1. **Refactor PyTorch:** Migrate `ptycho_torch/config_params.py` singletons to import and use `ptycho.config.config.ModelConfig`, `TrainingConfig`, `InferenceConfig`
   - Pros: Single source of truth, eliminates schema drift risk, reuses existing KEY_MAPPINGS
   - Cons: Requires PyTorch codebase refactor, may break existing PyTorch-internal assumptions
2. **Dual Schema with Translation:** Keep PyTorch singletons, implement explicit translation layer
   - Pros: Minimal PyTorch disruption, preserves existing workflows
   - Cons: Ongoing maintenance of parallel schemas, translation bugs, test coverage duplication

**Recommended Decision:** Refactor PyTorch to shared dataclasses (Option 1) to eliminate long-term drift risk and align with CONFIG-001 finding

**Decision Forum:** INTEGRATE-PYTORCH-001 Phase B kickoff meeting
**Blocking:** INTEGRATE-PYTORCH-001 Phase B.B2 (schema harmonization implementation)

---

### Q2: Minimum Viable Configuration Surface
**Question:** Which of the 75+ documented config fields in `specs/ptychodus_api_spec.md §5` are truly required for minimal viable integration vs. full parity?

**Impact:** Affects Phase B scope definition and timeline — MVP approach enables faster iteration

**Context:**
- Spec documents ModelConfig (11 fields), TrainingConfig (19 fields), InferenceConfig (9 fields)
- Not all fields may be consumed by current ptychodus integration
- Some fields are TensorFlow-legacy and may not apply to PyTorch architecture

**Recommended Approach:**
1. Trace ptychodus reconstructor code (`ptychodus/src/ptychodus/model/ptychopinn/reconstructor.py`) to identify actually-consumed fields
2. Define MVP as "fields required for basic train→save→load→infer cycle"
3. Defer advanced fields (e.g., `gaussian_smoothing_sigma`, `probe_mask`) to Phase 7 (validation)

**Decision Forum:** INTEGRATE-PYTORCH-001 Phase A review with spec cross-reference
**Blocking:** INTEGRATE-PYTORCH-001 Phase B.B1 (field mapping audit scope)

---

### Q3: Integration Surface Selection
**Question:** Should ptychodus integration use the `ptycho_torch/api/` layer (ConfigManager, PtychoModel, Trainer, InferenceEngine) or bypass it for direct module access?

**Impact:** Affects Phases 3-6 implementation strategy, MLflow dependency policy, and long-term maintainability

**Options:**
1. **API-First Integration:** ptychodus calls `ptycho_torch/api/base_api.py` classes
   - Pros: Cleaner abstraction, built-in MLflow orchestration, Lightning integration
   - Cons: Adds MLflow dependency, less control over low-level operations, API stability risk
2. **Low-Level Integration:** ptychodus bypasses API for direct module calls (model, loader, workflows)
   - Pros: More control, fewer dependencies, consistent with TensorFlow integration pattern
   - Cons: Duplicates orchestration logic, API layer maintenance burden without users

**Recommended Decision:** Low-level integration (Option 2) to maintain consistency with TensorFlow pattern and minimize dependency surface

**Decision Forum:** INTEGRATE-PYTORCH-001 Phase A decision gate (after API layer capability assessment)
**Blocking:** INTEGRATE-PYTORCH-001 Phase C (data pipeline), Phase D (inference), Phase E (training)

---

## Deferred Decisions (Resolve Before Respective Phases)

### Q4: MLflow Persistence Contract Ownership
**Question:** If PyTorch backend uses MLflow for model persistence, who owns the contract — PyTorch backend internals or ptychodus integration glue?

**Impact:** Affects Phase 5 persistence adapter design and deployment requirements

**Context:**
- Current PyTorch uses `mlflow.pytorch.autolog()` and `save_mlflow()` / `load_from_mlflow()`
- TensorFlow uses `.h5.zip` archive format without MLflow
- Spec requires compatible persistence semantics (`specs/ptychodus_api_spec.md §4.6`)

**Recommendation:** Decouple MLflow from contract surface — implement Lightning checkpoint → `.h5.zip` adapter so ptychodus sees unified archive format regardless of internal persistence strategy

**Decision Forum:** INTEGRATE-PYTORCH-001 Phase E (training) + Phase F (persistence) sync
**Blocking:** INTEGRATE-PYTORCH-001 Phase F.F1 (persistence adapter design)

---

### Q5: Synthetic Data Generation Strategy
**Question:** Should `ptycho_torch/datagen/` replace or complement existing TensorFlow simulation workflows (`ptycho.diffsim`)?

**Impact:** Affects tooling standardization, maintenance burden, and dataset generation documentation

**Options:**
1. **Replace:** Deprecate TensorFlow simulation in favor of unified `datagen/` package
   - Pros: Single toolchain, modern implementation
   - Cons: Migration effort, potential TensorFlow workflow breakage
2. **Complement:** Keep both toolchains, ensure NPZ contract compatibility
   - Pros: No disruption to existing workflows
   - Cons: Ongoing maintenance of parallel implementations

**Recommendation:** Complement (Option 2) for now — validate `datagen/` NPZ outputs against spec, document differences, consider unification in future deprecation cycle

**Decision Forum:** INTEGRATE-PYTORCH-001 Phase C (data pipeline) review
**Blocking:** None (can proceed with validation-only approach)

---

### Q6: Reassembly Parity Tolerances
**Question:** What are acceptable numeric tolerances for PyTorch barycentric reassembly output differences vs. TensorFlow `tf_helper.reassemble_position()`?

**Impact:** Affects parity test pass/fail criteria and whether PyTorch reassembly can be considered compliant

**Context:**
- Different interpolation strategies and tensor layouts may produce slightly different outputs
- Need quantitative thresholds: RMSE, SSIM, phase correlation, etc.
- Tolerances may vary by use case (synthetic vs. experimental data)

**Recommended Approach:**
1. Run PyTorch and TensorFlow reassembly on identical synthetic fixtures
2. Measure reconstruction quality metrics (PSNR, SSIM, FRC) and raw tensor differences
3. Define tolerances based on scientific significance (e.g., SSIM > 0.95, PSNR > 30 dB)

**Decision Forum:** TEST-PYTORCH-001 fixture design session with INTEGRATE-PYTORCH-001 Phase D lead
**Blocking:** INTEGRATE-PYTORCH-001 Phase D.D4 (reassembly parity validation)

---

### Q7: Lightning/MLflow Dependency Policy
**Question:** Are PyTorch Lightning and MLflow mandatory dependencies for ptychodus integration, or should we support optional fallback paths?

**Impact:** Affects deployment requirements, CI configuration, and user environment setup burden

**Options:**
1. **Mandatory:** Require Lightning + MLflow in all PyTorch environments
   - Pros: Simpler implementation, full feature parity
   - Cons: Heavier dependency footprint, deployment friction
2. **Optional with Graceful Degradation:** Detect availability, provide fallback training loop
   - Pros: Lighter deployment, broader compatibility
   - Cons: More complex implementation, dual code paths

**Recommendation:** Start with mandatory (Option 1) for MVP, evaluate optional support based on user feedback in Phase 7 (validation)

**Decision Forum:** INTEGRATE-PYTORCH-001 Phase E (training) kickoff
**Blocking:** INTEGRATE-PYTORCH-001 Phase E.E1 (training workflow orchestration)

---

### Q8: Multi-Stage Training UI Exposure
**Question:** Should PyTorch multi-stage training logic (`stage_1/2/3_epochs` with physics weight scheduling) be exposed to ptychodus configuration UI, or remain PyTorch-internal?

**Impact:** Affects reconstructor settings surface and UI complexity

**Context:**
- PyTorch `train.py` embeds multi-stage training with configurable epoch counts and weight schedules
- TensorFlow training uses simpler single-stage configuration
- Exposing may provide user control but increases UI complexity

**Recommendation:** Keep PyTorch-internal for MVP (use sensible defaults), gather user feedback in Phase 7 before exposing to UI

**Decision Forum:** INTEGRATE-PYTORCH-001 Phase E (training) review after initial implementation
**Blocking:** None (can proceed with internal-only approach)

---

## Decision Log Template

When questions are resolved, log decisions here:

```markdown
### [Q-ID] Question Summary
**Decision:** [Selected option]
**Rationale:** [Why this was chosen]
**Date:** [YYYY-MM-DD]
**Decision Makers:** [Names/roles]
**Impact:** [What changes as a result]
**Follow-up Actions:** [Tasks spawned]
```

---

## References

**Stakeholder Brief:** `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T031500Z/stakeholder_brief.md`
**Delta Analysis:** `plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/delta_log.md`
**Canonical Plan:** `plans/ptychodus_pytorch_integration_plan.md`
**Spec Reference:** `specs/ptychodus_api_spec.md`
