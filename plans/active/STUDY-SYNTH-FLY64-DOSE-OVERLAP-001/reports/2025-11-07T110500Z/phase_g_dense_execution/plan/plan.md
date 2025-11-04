# Phase G Dense Execution — Plan (2025-11-07T110500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)  
**Loop Type:** Planning → Implementation (TDD)

---

## Objectives

1. Reinstate deterministic **Phase C→G pipeline** execution for dose=1000, view=dense, splits=train/test using the orchestrator.
2. Extend the orchestrator to emit a **metrics summary** (`analysis/metrics_summary.{json,md}`) derived from comparison CSV outputs, failing fast when jobs are missing or the manifest reports failures.
3. Capture RED→GREEN proof for the new summary helper and archive the full pipeline run (CLI logs, manifests, comparison metrics).
4. Update documentation ledgers (fix_plan, findings untouched) once evidence is captured.

---

## Deliverables

- Updated `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py` with `summarize_phase_g_outputs()` helper invoked post-pipeline.
- New pytest coverage for the summary helper (temp hub fixture) with RED→GREEN logs.
- Artifact hub populated:
  - `cli/phase_[c-g]_*.log`
  - `analysis/comparison_manifest.json`
  - `analysis/metrics_summary.json`
  - `analysis/metrics_summary.md`
  - `data/phase_{c,d,e,f}/...` bundles
- `summary/summary.md` describing metrics (MS-SSIM phase/amplitude, MAE) for both splits.
- Doc sync updates if new pytest selectors are introduced.

---

## Task Breakdown

1. **TDD guardrail**
   - Author a failing test (`pytest tests/study/test_phase_g_dense_orchestrator.py::test_summarize_phase_g_outputs`) asserting summary files/contents.
   - Capture RED log under `red/pytest_red.log`.

2. **Implementation**
   - Add `summarize_phase_g_outputs(hub: Path)` to orchestrator.
   - Hook helper into `main()` after successful pipeline execution; raise `RuntimeError` when manifest reports failures or metrics CSV missing.
   - Ensure helper writes deterministic JSON/Markdown outputs referencing MS-SSIM + MAE per model.

3. **Validation**
   - Rerun targeted pytest selector (GREEN → `green/pytest_green.log`).
   - Execute orchestrator:
     ```bash
     export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
     python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
         --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T110500Z/phase_g_dense_execution \
         --dose 1000 \
         --view dense \
         --splits train test
     ```
   - Archive CLI logs beneath `cli/`.

4. **Doc Sync (triggered by new pytest)**
   - `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -vv` → `collect/pytest_collect.log`
   - Update `docs/TESTING_GUIDE.md` (Phase G entry) & `docs/development/TEST_SUITE_INDEX.md` to reference new selector.

5. **Reporting**
   - Summarize metrics in `summary/summary.md` (include SHA256 proof for comparison logs if produced).
   - Update `docs/fix_plan.md` Attempt entry with outcomes + pointers.

---

## Findings/Policies In Force

- POLICY-001 — PyTorch installed; pipeline trains via TensorFlow backend.
- CONFIG-001 — Legacy bridge intact; orchestrator must maintain params flow.
- DATA-001 — Phase C/D/E outputs must stay contract compliant; summary should confirm dataset counts.
- OVERSAMPLING-001 — Dense spacing thresholds unchanged (0.7 overlap).
- TYPE-PATH-001 — All new file system paths normalized via `Path`.

---

## Pitfalls / Guardrails

- Do **not** modify core physics (ptycho/model.py, diffsim.py, tf_helper.py).
- Ensure orchestrator helper tolerates pre-existing bundles (idempotent runs).
- Avoid writing artifacts outside hub; remove temporary scratch after final summary.
- If training CLI aborts, capture `analysis/blocker.log` and halt (no partial metrics).

