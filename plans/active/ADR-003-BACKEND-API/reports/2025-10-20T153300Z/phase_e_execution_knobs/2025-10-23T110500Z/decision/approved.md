# Phase EB3 Logger Backend Decision — Supervisor Approval

**Date:** 2025-10-23  
**Supervisor:** Galph (Codex)  
**Initiative:** ADR-003-BACKEND-API — Phase EB3 (Logger Governance)  
**Source Proposal:** `decision/proposal.md` (Attempt #66)

---

## Approved Decisions

- **Q1 — Default Logger Backend:** ✅ *Approve.* Set `PyTorchExecutionConfig.logger_backend` default to `'csv'` so Lightning CSVLogger captures loss metrics currently dropped by `logger=False`. Complies with POLICY-001 (no new dependencies) and aligns with CONFIG-001 sequencing requirements.
- **Q2 — TensorBoard Support in Phase EB3.B:** ✅ *Approve.* Implement `TensorBoardLogger` alongside CSV in the same loop to preserve TensorFlow parity (`ptycho/model.py:546-551`) without adding dependencies (tensorboard already shipped via TensorFlow requirement).
- **Q3 — `--disable_mlflow` Deprecation:** ✅ *Approve.* Emit `DeprecationWarning` mapping the flag to `--logger none` while pointing users to `--quiet` for progress suppression. Capture guidance in workflow docs during Phase C so existing scripts migrate gracefully.
- **Q4 — MLflow Refactor Follow-Up:** ✅ *Track as follow-up.* Record a backlog task to replace manual `mlflow.pytorch.autolog()` usage with the Lightning `MLFlowLogger` after CSV/TensorBoard work lands. Reference this in Phase C documentation updates and escalate into fix_plan once EB3 implementation stabilizes.

---

## Implementation Notes for Phase EB3.B

1. **TDD Scope:** Follow proposal’s seven-test matrix (3 CLI, 2 factory, 1 workflow, 1 integration). Capture RED logs before implementation.
2. **Configuration Handling:** Extend factory helpers to return the selected logger instance. Guard optional backends with actionable errors if dependencies absent.
3. **CLI Surface:** Introduce `--logger {csv,tensorboard,mlflow,none}` (+ alias `--logger-backend` if needed); emit DeprecationWarning for `--disable_mlflow`.
4. **Execution Wiring:** Replace `logger=False` in `_train_with_lightning` with the constructed logger. Ensure progress-bar handling stays tied to `enable_progress_bar`.
5. **Artifacts:** Store RED/GREEN pytest logs and integration evidence under `impl/<ISO8601>/` within this initiative hub.

---

## Follow-Up / Backlog

- Add a plan note (Phase C) capturing the MLflow refactor backlog so we open a dedicated fix_plan entry after EB3.B/C complete.
- When Phase C documentation sync runs, update `docs/findings.md` or initiative plan to mention the pending MLflow logger migration.

---

**Approval recorded by:** Galph  
**Next Step:** Proceed with Phase EB3.B per updated `input.md` instructions.
