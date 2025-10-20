# ADR-003 Phase C4.D Blockers — Close-Out Summary (2025-10-20T114500Z)

## Outcome
- **Status:** Phase C4.D blockers cleared. Bundle loader + Lightning parity fixes validated; plan close-out tasks (C1–C3) complete.
- **Artifacts:** Primary GREEN evidence captured under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T111500Z/phase_c4d_at_parallel/`.

## Evidence Recap
1. **Bundle Loader Regression:** `pytest_bundle_loader_green.log` (13.02 s) — `load_torch_bundle()` reconstructs both Lightning modules; no `NotImplementedError`.
2. **Lightning Gridsize Regression:** `pytest_gridsize_green.log` (4.99 s) — `test_lightning_training_respects_gridsize` asserts coords layout `(batch, gridsize**2, 1, 2)`; GREEN post-permute fix.
3. **End-to-End Integration:** `pytest_integration_green.log` (16.77 s) — `test_run_pytorch_train_save_load_infer` completes train→save→load→infer cycle without channel crashes.
4. **CLI Smoke (gridsize=2):** `manual_cli_smoke_gs2.log` — 1 epoch Lightning run succeeds on CPU with deterministic + execution config overrides; checkpoints emitted to `/tmp/cli_smoke`.

Additional supporting logs remain in this directory (`pytest_load_bundle_{red,green}.log`, `pytest_integration_phase_a.log`, etc.) for historical traceability.

## Decisions & Updates
- Updated `phase_c4_cli_integration/plan.md` rows C4.D3/C4.D4 to `[x]`, referencing the new artifact hub.
- Refreshed this summary to document GREEN evidence, runtime metrics, and residual follow-ups.
- Logged `docs/fix_plan.md` Attempt #35 summarizing parity validation and outstanding documentation work (C4.E/F).

## Follow-Up (Outside Phase C4.D Scope)
- **Documentation Refresh:** Coordinate Phase C4.E (workflow guide + spec updates) using the new evidence.
- **Ledger Hygiene:** Phase C4.F tasks remain open (comprehensive summary, Phase D prep notes).
- **Artifacts Discipline:** Continue storing subsequent outputs under timestamped subdirectories (e.g., future C4.E loops).

## References
- `specs/ptychodus_api_spec.md` §4.6–§4.8 — Bundle lifecycle & CLI routing.
- `docs/workflows/pytorch.md` §§5–7 — Persistence and CLI usage patterns (needs update with gridsize=2 smoke results).
- Findings: `CONFIG-001`, `BUG-TF-001`, `POLICY-001` — all satisfied by current validation run.
