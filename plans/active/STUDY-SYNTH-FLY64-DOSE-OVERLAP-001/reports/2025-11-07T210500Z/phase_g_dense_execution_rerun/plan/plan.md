# Phase G Dense Execution Rerun — Clean Hub & Evidence Plan (2025-11-07T210500Z)

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001  
**Focus:** Phase G comparison & analysis (dense evidence)  
**Action Type:** Planning (supervisor loop)  
**Target State:** ready_for_implementation

---

## Context

- The metadata guard (`validate_phase_c_metadata`) and transformation enforcement are GREEN (attempt 2025-11-07T190500Z+exec); pytest + docs are up-to-date.  
- The metrics summarizer (`summarize_phase_g_outputs`) is also GREEN (attempt 2025-11-07T110500Z+exec).  
- However, we still lack real Phase C→G dense evidence because earlier runs generated stale Phase C artifacts without metadata. Those stale directories can cause the guard to fail silently if re-used.  
- We need a safe rerun process that guarantees fresh outputs (with metadata and canonical transformations) and captures full CLI/metrics evidence under a new hub.

## Objectives

1. Harden `run_phase_g_dense.py` with a reusable hub-preparation helper that detects stale data and supports an explicit `--clobber` option to wipe prior outputs before reruns.  
2. Cover the new guard path with pytest (RED→GREEN) to prevent regressions around hub cleaning semantics.  
3. Execute the dense Phase C→G pipeline (`dose=1000`, `view=dense`, `splits=train test`) using the clean hub to capture end-to-end evidence (logs, metrics summary, manifests).  
4. Summarize outcomes (metrics, guard behavior, artifacts) and sync the ledger/documentation.

## Deliverables

- Updated orchestrator script (`plans/.../bin/run_phase_g_dense.py`) exposing `prepare_hub()` + `--clobber` handling (read-only by default, wipe on demand).  
- Expanded pytest coverage in `tests/study/test_phase_g_dense_orchestrator.py` with RED→GREEN logs plus `--collect-only` proof stored in this hub.  
- CLI transcript(s) for the rerun plus generated artifacts under `analysis/` (metrics_summary.json/md, manifests, etc.).  
- Refreshed summary/summary.md, docs/fix_plan.md attempt log, galph_memory.md, and (if new selectors introduced) TESTING_GUIDE/Test Suite Index updates.

## Tasks

1. **TDD — RED:**  
   - Add helper import (similar to existing summary tests) for `prepare_hub()` inside `tests/study/test_phase_g_dense_orchestrator.py`.  
   - Write `test_prepare_hub_detects_stale_outputs` that seeds a fake hub with existing `data/phase_c/dose_1000_train/` content and asserts the helper raises `RuntimeError` when `clobber=False`.  
   - Capture the failure log at `reports/.../red/pytest_prepare_hub_red.log`.

2. **Implementation:**  
   - Implement `prepare_hub(hub: Path, *, clobber: bool) -> None` in the orchestrator script. Responsibilities:  
     - Normalize paths (TYPE-PATH-001); create hub directories if missing.  
     - Detect existing data/phase_* or analysis outputs; if any exist and `clobber` is False, raise RuntimeError with actionable guidance.  
     - When `clobber` is True, move any previous hub contents to `analysis/previous_run_<timestamp>/` or delete them (choose one, but document behavior) before recreating clean directories.  
   - Wire a new CLI flag `--clobber` (default False) that calls `prepare_hub` before queuing commands.

3. **TDD — GREEN & Collection:**  
   - Extend the test module with `test_prepare_hub_clobbers_previous_outputs` to confirm cleanup when `clobber=True`.  
   - Re-run targeted selectors (the two new tests plus existing metadata guard/summary tests) capturing GREEN logs under `.../green/` and update `collect/pytest_phase_g_orchestrator_collect.log` via `pytest --collect-only`.

4. **CLI Evidence:**  
   - Execute `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/.../bin/run_phase_g_dense.py --hub <this-hub> --dose 1000 --view dense --splits train test --clobber`.  
   - Tee stdout/stderr to `cli/phase_g_dense_pipeline.log`; on failure, write traceback to `analysis/blocker.log` and stop (do not retry blindly).

5. **Documentation & Ledger:**  
   - Summarize outcomes in `summary/summary.md` (include Turn Summary block).  
   - Update `docs/fix_plan.md` Attempts History, `galph_memory.md`, and (if selectors changed) `docs/TESTING_GUIDE.md` + `docs/development/TEST_SUITE_INDEX.md`.  
   - Record artifact locations and metrics in the summary.

## Guardrails & Findings Applied

- POLICY-001 — Remain backend-neutral; pipeline uses TensorFlow backend but guard must not assume TF-only semantics.  
- CONFIG-001 — Do not bypass `update_legacy_dict`; orchestrator CLI already handles this; helper must remain metadata-only.  
- DATA-001 — Never mutate NPZ payload contents; cleanups should only affect directory scaffolding.  
- TYPE-PATH-001 — Normalize all filesystem interactions (`Path.resolve()`), especially when deleting/moving directories.  
- OVERSAMPLING-001 — Ensure dense view parameters remain unchanged during rerun.

## Pitfalls

- Avoid deleting hub contents silently; require explicit `--clobber`.  
- Preserve previous evidence by moving to `analysis/previous_run_*` if possible; document the retention policy in RuntimeError text.  
- New pytest tests must clean up temp directories (use `tmp_path`).  
- Ensure CLI run sets `AUTHORITATIVE_CMDS_DOC`; script already warns but we need explicit export in How-To map.  
- Capture collect-only proof after GREEN so selectors remain traceable in registries.
