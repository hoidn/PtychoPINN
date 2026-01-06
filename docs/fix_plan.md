# PtychoPINN Fix Plan Ledger (Condensed)

**Last Updated:** 2026-01-06 (housekeeping: Attempts History trimmed, done initiatives compressed)
**Active Focus:** FIX-TF-C1D-SCALED-RERUN-001 — Guard + scaled TF rerun evidence capture (Phase C1d)

---

**Housekeeping Notes:**
- Full Attempts History archived in `docs/fix_plan_archive.md` (snapshot 2026-01-06)
- Earlier snapshots: `docs/archive/2025-11-06_fix_plan_archive.md`, `docs/archive/2025-10-17_fix_plan_archive.md`, `docs/archive/2025-10-20_fix_plan_archive.md`
- Each initiative has a working plan at `plans/active/<ID>/implementation.md` and reports under `plans/active/<ID>/reports/`

---

## Active / Pending Initiatives

### [STUDY-SYNTH-FLY64-DOSE-OVERLAP-001] Synthetic fly64 dose/overlap study
- Depends on: Phase C/E/F artifacts under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/data/`; FEAT-LAZY-LOADING-001 to eliminate OOM wall (PINN-CHUNKED-001).
- Priority: High
- Status: blocked_escalation — Tier 3 dwell triggered 2025-11-16; full dense-test Phase G blocked on GPU/TF resource limits.
- Owner/Date: Ralph/2025-11-11
- Working Plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`
- Summary: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/summary.md`
- Reports Hub: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`
- Goals:
  - **G-full (blocked):** Phase G dense verification at full scale with non-zero Baseline rows, SSIM grid, metrics.
  - **G-scaled (next target):** Phase G on reduced configuration without OOM/timeout.
- Return Conditions:
  - **G-scaled:** Scaled rerun complete with populated Baseline rows, SSIM grid, verification_report.json, artifact_inventory.txt.
  - **G-full:** Follow-up initiative for GPU/TF limitations or decision recorded in docs/findings.md.
- Attempts History:
  - *First (2025-11-11):* Initial Phase G orchestrator execution failed on Phase C generation.
  - *Last (2025-11-16T110500Z):* Tier 3 enforcement logged under `analysis/dwell_escalation_report.md`; awaiting scaled rerun.
  - ... (see `docs/fix_plan_archive.md` and `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/` for full history).

---

### [FIX-PYTORCH-FORWARD-PARITY-001] Stabilize Torch Forward Patch Parity
- Depends on: INTEGRATE-PYTORCH-PARITY-001, FIX-COMPARE-MODELS-TRANSLATION-001
- Priority: High
- Status: blocked — Phase A/B complete (intensity_scale=9.882118); awaiting FIX-TF-C1D-SCALED-RERUN-001.
- Owner/Date: Ralph/2025-11-14
- Working Plan: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md`
- Reports Hub: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/`
- Notes: Phase A rerun confirmed TrainingPayload threading; B1 object_big defaults enforced. Commit 9a09ece2 threads intensity_scale through save/load bundle.
- Return Condition: Resume when FIX-TF-C1D-SCALED-RERUN-001 produces guard + scaled TF evidence or records a blocker.
- Attempts History:
  - *First (2025-11-14):* Phase A/B evidence captured.
  - *Last (2025-11-19T190500Z):* Tier-2 dwell enforcement; work tracked under FIX-TF-C1D-SCALED-RERUN-001.
  - ... (see `docs/fix_plan_archive.md` for full history).

---

### [FIX-IMPORT-SIDE-EFFECTS-001] Remove Global State Side-Effects in ptycho/model.py
- Depends on: `spec-ptycho-config-bridge.md` Compliance; `docs/debugging/QUICK_REFERENCE_PARAMS.md` guardrails.
- Priority: High
- Status: pending — planning artifacts created 2025-11-20; awaiting Phase A regression tests.
- Owner/Date: Ralph/2025-11-20
- Working Plan: `plans/active/FIX-IMPORT-SIDE-EFFECTS-001/implementation.md`
- Summary: `plans/active/FIX-IMPORT-SIDE-EFFECTS-001/summary.md`
- Reports Hub: `plans/active/FIX-IMPORT-SIDE-EFFECTS-001/reports/`
- Goals:
  - Remove module-scope `params.get()` calls from `ptycho/model.py`.
  - Ensure `autoencoder`/`diffraction_to_obj` instantiated via factories after `update_legacy_dict`.
  - Add regression tests proving `import ptycho.model` performs no `ptycho.params` access.
- Return Condition: Regression test passes with factory-only config access evidence in Reports Hub.

---

### [FEAT-LAZY-LOADING-001] Implement Lazy Tensor Allocation in loader.py
- Depends on: `spec-ptycho-workflow.md` Resource Constraints; finding `PINN-CHUNKED-001`.
- Priority: High
- Status: pending — planning artifacts created 2025-11-20; awaiting Phase A reproduction script.
- Owner/Date: Ralph/2025-11-20
- Working Plan: `plans/active/FEAT-LAZY-LOADING-001/implementation.md`
- Summary: `plans/active/FEAT-LAZY-LOADING-001/summary.md`
- Reports Hub: `plans/active/FEAT-LAZY-LOADING-001/reports/`
- Goals:
  - Refactor `PtychoDataContainer` to keep datasets in NumPy/mmap until batch request.
  - Provide streaming/batching APIs (`.as_dataset()` or equivalent).
  - Update dependent scripts (train_pinn, compare_models) to use lazy interfaces.
- Return Condition: Lazy container merged with OOM fix tests; STUDY-SYNTH-FLY64 unblocked.

---

### [FIX-TF-C1D-SCALED-RERUN-001] Phase C1d TensorFlow scaled rerun execution
- Depends on: FIX-PYTORCH-FORWARD-PARITY-001 (Phase C guard + regression tests merged)
- Priority: Critical
- Status: in_progress — guard/regression code landed; executing scaled TF rerun now.
- Owner/Date: Ralph/2025-11-19
- Working Plan: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md` (Phase C1d checklist)
- Summary: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/summary.md`
- Reports Hub: `plans/active/FIX-PYTORCH-FORWARD-PARITY-001/reports/2025-11-13T000000Z/forward_parity/`
- Do Now: Execute scaled TF rerun per input.md — (1) run guard pytest selector, (2) run scaled TF training CLI, (3) publish artifacts or log blocker.
- Return Conditions:
  - Guard selector log at `$HUB/green/pytest_tf_translation_guard.log` showing GREEN.
  - `$TF_BASE/analysis/forward_parity_debug_tf/` contains stats/offsets/PNGs (or blocker exists).
  - `$TF_BASE/cli/train_tf_phase_c1_scaled.log` non-empty.
  - Hub inventory/summary updated.
- Attempts History:
  - *First (2025-11-14T153800Z):* Guard PASSED GREEN, but TF CLI failed during eval with reshape error.
  - *Last (2025-11-20T002500Z):* Third-loop retrospective confirmed no new evidence since Nov 14.
  - ... (see `docs/fix_plan_archive.md` for full history).

---

### [PARALLEL-API-INFERENCE] Programmatic TF/PyTorch API parity
- Depends on: INTEGRATE-PYTORCH-PARITY-001 (backend selector wiring complete)
- Priority: Medium
- Status: planning — new initiative for backend selector demo script + helpers.
- Owner/Date: Ralph/2025-11-14T030000Z
- Working Plan: `plans/active/PARALLEL-API-INFERENCE/plan.md`
- Reports Hub: TBD
- Do Now:
  1. Extract TF inference helper from `scripts/inference/inference.py`.
  2. Build `scripts/pytorch_api_demo.py` for both backends.
  3. Add smoke test `tests/scripts/test_api_demo.py`.
  4. Document in `docs/workflows/pytorch.md`.

---

## Done / Archived Initiatives

*Full details in `docs/fix_plan_archive.md` (snapshot 2026-01-06) and respective `plans/active/<ID>/reports/` directories.*

### [INDEPENDENT-SAMPLING-CONTROL-PHASE6] Independent sampling control — Phase 6 guardrails
- Status: done — Phase 6A guardrails landed (explicit `enable_oversampling`/`neighbor_pool_size` plumbing, RawData gating per OVERSAMPLING-001, pytest coverage, docs refresh).
- Working Plan: `plans/active/independent-sampling-control/implementation.md`

### [FIX-COMPARE-MODELS-TRANSLATION-001] Dense Phase G translation guard
- Status: done — Batched reassembly (a80d4d2b) + XLA streaming (bf3f1b07) verified. Regression tests GREEN (2/2), compare_models exit 0, verification report 10/10.
- Working Plan: `plans/active/FIX-COMPARE-MODELS-TRANSLATION-001/implementation.md`

### [FIX-PHASE-C-GENERATION-001] Fix Phase C coordinate type bug
- Status: done — TypeError in `ptycho/raw_data.py:227` fixed by setting `TrainingConfig.n_images` in `studies/fly64_dose_overlap/generation.py`.
- Owner/Date: Ralph/2025-11-07

### [EXPORT-PTYCHODUS-PRODUCT-001] TF-side Ptychodus product exporter/importer + Run1084 conversion
- Status: done — Exporter/importer code, CLI, tests (3/3 PASSED), Run1084 HDF5, docs in DATA_MANAGEMENT_GUIDE.md. Commit a679e6fb.
- Working Plan: `plans/active/EXPORT-PTYCHODUS-PRODUCT-001/implementation_plan.md`

### [INTEGRATE-PYTORCH-PARITY-001] PyTorch backend API parity reactivation
- Status: done — Phase R CLI GPU-default handoff complete. Training/inference CLIs succeeded with POLICY-001 warnings + CUDA execution config.
- Working Plan: `plans/ptychodus_pytorch_integration_plan.md`
- Reports Hub: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/`

### [INTEGRATE-PYTORCH-PARITY-001B] CLI GPU-default evidence & execution-config regression capture
- Status: done — GPU-default CLI evidence + RED execution_config regression log archived.
- Reports Hub: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-11-13T150000Z/parity_reactivation/`

### [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2
- Status: archived 2025-10-20 — see `docs/archive/2025-10-20_fix_plan_archive.md`.

### [INTEGRATE-PYTORCH-001-DATALOADER] Restore PyTorch dataloader DATA-001 compliance
- Status: archived 2025-10-20 — see `docs/archive/2025-10-20_fix_plan_archive.md`.
