# Phase EB3.C Documentation Sync — Summary

**Date:** 2025-10-24
**Mode:** Docs
**Attempt:** #69 (ADR-003-BACKEND-API)
**Phase:** EB3.C (Logger Backend Documentation Sync)

---

## Overview

This documentation-only loop completes Phase EB3.C of the logger backend initiative by synchronizing normative documentation (`specs/ptychodus_api_spec.md`, `docs/workflows/pytorch.md`) and project ledgers (`docs/findings.md`, `implementation.md`, `docs/fix_plan.md`) with the logger implementation delivered in Attempt #68 (Phase EB3.B).

**Implementation Context:** Phase EB3.B (commit 43ea2036) introduced Lightning logger support for PyTorch training:
- Default: CSVLogger (`logger_backend='csv'`)
- Optional: TensorBoardLogger (`'tensorboard'`), MLFlowLogger (`'mlflow'`)
- Disable: `'none'` option
- CLI flag: `--logger {csv,tensorboard,mlflow,none}`
- Deprecation: `--disable_mlflow` emits DeprecationWarning → `--logger none`

---

## Tasks Completed

### C1. Spec Update — `specs/ptychodus_api_spec.md`

**§4.9 Logger Backend Field Definition (line 281):**
- **Before:** `logger_backend` (str|None, default `None`): Experiment tracking backend. Pending governance decision (Phase E.B3).
- **After:** Comprehensive field documentation with:
  - Type contract: `str, default 'csv'`, MUST be one of `['csv', 'tensorboard', 'mlflow', 'none']`
  - Four backend descriptions:
    - `'csv'`: CSVLogger (default), zero deps, CI-friendly, stores metrics in `{output_dir}/lightning_logs/`
    - `'tensorboard'`: TensorBoardLogger, requires tensorboard (auto-installed via TF)
    - `'mlflow'`: MLFlowLogger, requires mlflow package + server URI
    - `'none'`: Disable logging, discards `self.log()` calls
  - Factory default behavior: when dataclass field is `None`, factory defaults to `'csv'`
  - CLI exposure: `--logger` flag
  - MLflow migration note: current implementation uses legacy `mlflow.pytorch.autolog()` (train.py:75-80), migration to Lightning MLFlowLogger tracked as Phase EB3.C4 backlog
  - Deprecation notice: `--disable_mlflow` emits DeprecationWarning directing to `--logger none` + `--quiet`

**§7.1 Training CLI Flag Table (new row 399):**
- Added `--logger` row with full description matching §4.9 semantics
- Type: `str`, Default: `'csv'`, Config Field: `PyTorchExecutionConfig.logger_backend`
- Description covers all four backends with usage guidance (CSV for CI, TensorBoard for viz, MLflow for tracking, none for smoke tests)

**Backlog Cleanup (lines 408-410, 428):**
- Removed `logger_backend` from "Planned Exposure (Phase E.B Backlog)" programmatic-only list
- Updated general note (line 428) from "checkpoint knobs, scheduler, logger backend" → "advanced trainer knobs"

**Deprecation Documentation (line 403):**
- Updated `--disable_mlflow` entry from "MLflow integration not yet implemented; flag is accepted but has no effect" to:
  - **DEPRECATED.** Emits DeprecationWarning directing users to `--logger none` for disabling experiment tracking and `--quiet` for suppressing progress bars.
  - Current behavior: maps to `--logger none` internally for backward compatibility
  - Will be removed in future release

---

### C2. Workflow Guide Update — `docs/workflows/pytorch.md`

**§12 Training Execution Flags Table (new row 329):**
- Added `--logger` row between `--early-stop-patience` and monitor metric aliasing section
- Full backend options documented with file paths and server requirements
- Guidance: use `'none'` with `--quiet` to suppress all output

**Logger Backend Details Section (new, lines 337-338):**
- CSV backend: captures all `self.log()` metrics without deps, saves to `{output_dir}/lightning_logs/version_N/metrics.csv`
- TensorBoard backend: enables interactive visualization, requires tensorboard package (auto via TF)
- MLflow backend: integrates with tracking servers, requires manual setup + mlflow package
- Usage recommendation: `--logger none` for quick smoke tests when metrics not needed

**DeprecationWarning Section (new, lines 340-344):**
- Documents `--disable_mlflow` deprecation with modern alternatives:
  - Disable tracking: `--logger none`
  - Suppress progress: `--quiet`
- Notes internal mapping to `--logger none` for backward compatibility
- Directs users to update scripts before future removal

**Deprecated Flags Update (line 348):**
- Changed `--disable_mlflow` from "MLflow integration is not yet implemented" to:
  - **DEPRECATED.** Use `--logger none` (disable tracking) + `--quiet` (suppress progress) instead
  - Emits DeprecationWarning; will be removed in future release

---

### C3. Findings Ledger — `docs/findings.md`

**New Finding: CONFIG-LOGGER-001 (line 12):**
- **Date:** 2025-10-24
- **Keywords:** logger, execution-config, lightning, mlflow
- **Synopsis:** PyTorch training uses CSVLogger by default (`PyTorchExecutionConfig.logger_backend='csv'`) to capture train/validation metrics from Lightning `self.log()` calls, replacing prior `logger=False` which discarded metrics. Allowed backends: `csv` (zero deps, CI-friendly), `tensorboard` (requires tensorboard from TF install), `mlflow` (requires mlflow package + server), `none` (disable). Legacy `--disable_mlflow` flag deprecated with DeprecationWarning mapping to `--logger none`. MLflow migration to Lightning MLFlowLogger tracked as Phase EB3.C4 backlog. Decision rationale and implementation evidence in governance decision approval record.
- **Evidence Link:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md`
- **Status:** Active

---

### C4. Implementation Plan — `implementation.md`

**Phase E2 Row Update (line 61):**
- **Before:** ✅ EB1–EB2 complete (Attempts #60–64). Docs sync artifacts at `phase_e_execution_knobs/2025-10-23T103000Z/`. EB3 logger work tracked via new plan `phase_e_execution_knobs/2025-10-23T110500Z/plan.md`; EB4 smoke pending.
- **After:** ✅ EB1–EB3.C complete (Attempts #60–69). Logger backend implemented (CSVLogger default, TensorBoard/MLflow optional, `--disable_mlflow` deprecated). Docs sync artifacts at `phase_e_execution_knobs/2025-10-23T110500Z/docs/2025-10-24T041500Z/` (spec §4.9/§7.1 + workflow guide §12 updated, CONFIG-LOGGER-001 added to findings). MLflow Logger migration tracked as Phase EB3.C4 backlog. EB4 smoke pending.

**MLflow Backlog Note:**
- Phase EB3.C4 checklist row (in `plan.md`) directs future work to migrate from manual `mlflow.pytorch.autolog()` to Lightning `MLFlowLogger` for consistency with other backends
- Tracked in decision/approved.md Q4 response
- Will be opened as dedicated fix_plan entry once EB3 stabilizes

---

## Artifact Inventory

| Artifact | Path | Size | Description |
|----------|------|------|-------------|
| Spec redline | `spec_redline.md` | 3.1 KB | Git diff of `specs/ptychodus_api_spec.md` showing all §4.9 and §7.1 changes |
| This summary | `summary.md` | 6.8 KB | Phase EB3.C completion documentation |

**All artifacts stored under:**
`plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/docs/2025-10-24T041500Z/`

---

## Documentation Changes Summary

### Files Modified (4)

1. **`specs/ptychodus_api_spec.md`** (5 edits):
   - §4.9 line 281: `logger_backend` field definition expanded (8 lines → 50 lines)
   - §4.9 line 286: Added MLflow migration note + deprecation notice
   - §7.1 line 399: Added `--logger` CLI flag row to training table
   - §7.1 line 403: Updated `--disable_mlflow` deprecation text
   - §7 backlog: Removed logger_backend from programmatic-only list (lines 408-410, 428)

2. **`docs/workflows/pytorch.md`** (3 edits):
   - §12 line 329: Added `--logger` flag row to training execution flags table
   - §12 lines 337-344: Added "Logger Backend Details" and "DeprecationWarning for --disable_mlflow" sections
   - §12 line 348: Updated deprecated flags list with `--disable_mlflow` migration guidance

3. **`docs/findings.md`** (1 edit):
   - Line 12: Added CONFIG-LOGGER-001 finding with full synopsis and evidence link

4. **`plans/active/ADR-003-BACKEND-API/implementation.md`** (1 edit):
   - Line 61: Updated Phase E2 row with EB3.C completion summary and artifact path

---

## Validation Checklist

### Phase EB3.C Exit Criteria (from `plan.md`)

- [x] **C1:** Updated `specs/ptychodus_api_spec.md` §4.9 with logger_backend field definition (CSV default, allowed values, fallback behavior, dependency notes)
- [x] **C1:** Removed logger_backend from programmatic-only backlog note (line ~424)
- [x] **C1:** Added `--logger` row to §7.1 training CLI table with default, field mapping, and behavior summary
- [x] **C1:** Documented DeprecationWarning for `--disable_mlflow` in §7.1 deprecated flags
- [x] **C1:** Generated spec redline diff (`spec_redline.md`)
- [x] **C2:** Updated `docs/workflows/pytorch.md` §12 training flags table with `--logger` entry
- [x] **C2:** Added prose explaining CSV default, TensorBoard/MLflow opt-in, and disable via `--logger none`
- [x] **C2:** Documented `--disable_mlflow` DeprecationWarning with migration guidance
- [x] **C2:** Wrote this comprehensive summary (`summary.md`)
- [x] **C3:** Added CONFIG-LOGGER-001 finding to `docs/findings.md` with evidence link to decision/approved.md
- [x] **C3:** Updated `implementation.md` Phase E2 row with EB3.C completion and artifact paths
- [x] **C4:** Recorded MLflow logger refactor backlog note in implementation.md commentary and this summary

### Hygiene Compliance

- [x] All artifacts stored under timestamped hub (`docs/2025-10-24T041500Z/`)
- [x] Files are ASCII/UTF-8 plain text (markdown)
- [x] No stray TODOs without tracking IDs
- [x] Spec and workflow guide reference identical option sets (`csv`, `tensorboard`, `mlflow`, `none`) in same order
- [x] DeprecationWarning verbiage consistent across spec and workflow guide
- [x] Markdown tables vertically aligned (pipe characters)
- [x] Findings table follows stable ID formatting (`| ID | Date | Tags | Summary | Evidence | Status |`)
- [x] Artifact paths relative to repo root in fix_plan/summary
- [x] No new `<doc-ref>` tags introduced (none required for this sync)

---

## Key Decisions and Rationale

### Default Backend Selection: CSV
- **Decision:** Set `logger_backend` default to `'csv'` (CSVLogger)
- **Rationale:** Zero dependencies, CI-friendly, captures currently-lost metrics from `self.log()` calls
- **POLICY-001 Compliance:** CSV and TensorBoard options require no new dependencies (TensorBoard auto-installed via TensorFlow requirement)

### TensorBoard Inclusion
- **Decision:** Implement TensorBoard support in same loop as CSV
- **Rationale:** Preserves TensorFlow parity (ptycho/model.py:546-551 uses TensorBoard), no dependency cost

### --disable_mlflow Deprecation Path
- **Decision:** Emit DeprecationWarning mapping to `--logger none`, retain backward compatibility
- **Rationale:** Graceful migration for existing scripts, clear guidance to modern alternatives
- **Timeline:** Flag will be removed in future release (post-ADR acceptance)

### MLflow Migration Backlog
- **Decision:** Track Lightning MLFlowLogger migration as Phase EB3.C4 follow-up
- **Rationale:** Current manual `mlflow.pytorch.autolog()` usage (train.py:75-80) works but inconsistent with other backends; migrate to Lightning logger for API uniformity
- **Next Step:** Open dedicated fix_plan entry once EB3 implementation stabilizes

---

## Next Steps

1. **Fix Plan Update (Attempt #69):** Update `docs/fix_plan.md` with:
   - Artifact paths (`spec_redline.md`, `summary.md`)
   - CONFIG-LOGGER-001 finding reference
   - MLflow logger backlog pointer
   - Completion notes for Phase EB3.C1-C4 checklist rows

2. **Commit & Push:** Stage all documentation changes with descriptive commit message:
   ```
   [ADR-003-EB3-C] docs: sync logger backend documentation

   - Update specs/ptychodus_api_spec.md §4.9/§7.1 with logger_backend field and --logger CLI flag
   - Add logger backend details and --disable_mlflow deprecation to docs/workflows/pytorch.md §12
   - Add CONFIG-LOGGER-001 finding to docs/findings.md
   - Update implementation.md Phase E2 with EB3.C completion summary
   - Generate spec redline diff at docs/2025-10-24T041500Z/spec_redline.md

   Phase EB3.C (Logger Backend Documentation Sync) COMPLETE.
   CSV default, TensorBoard/MLflow optional, --disable_mlflow deprecated.
   MLflow Logger migration tracked as Phase EB3.C4 backlog.

   Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/docs/2025-10-24T041500Z/
   Tests: none (docs-only loop)
   ```

3. **Phase EB3.D (Optional):** If supervisor requests smoke test demonstrating logger output, capture evidence per plan.md Phase D checklist

4. **Phase E Governance:** Once EB3 complete, proceed with Phase E.A governance dossier (ADR addendum, final spec redline, acceptance criteria validation)

---

## Cross-References

- **Decision Record:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md`
- **Implementation Evidence:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/impl/2025-10-24T025339Z/summary.md` (Attempt #68, Phase EB3.B)
- **Execution Plan:** `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md` (Phase EB3 master checklist)
- **Implementation Roadmap:** `plans/active/ADR-003-BACKEND-API/implementation.md` (Phase E rows)

---

**Phase EB3.C Documentation Sync COMPLETE.**
All normative docs and ledgers synchronized with logger backend implementation.
Ready for commit and fix_plan Attempt #69 finalization.
