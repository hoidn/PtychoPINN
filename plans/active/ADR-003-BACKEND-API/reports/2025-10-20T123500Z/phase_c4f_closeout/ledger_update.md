# Ledger Update for docs/fix_plan.md — ADR-003-BACKEND-API Attempt Entry

## Entry to Append

```markdown
* [2025-10-20] Attempt #18 — **Phase C4 COMPLETE (CLI Execution Config Integration, Mode: TDD + Docs)**: Exposed 5 training + 3 inference execution config CLI flags (`--accelerator`, `--deterministic`, `--num-workers`, `--learning-rate`, `--inference-batch-size`), refactored `ptycho_torch/train.py` and `ptycho_torch/inference.py` to delegate config construction to factory pattern (eliminating ad-hoc construction + hardcoded values), and updated comprehensive documentation suite. **Phase Breakdown:** (C4.A) Authored 4 design docs: `cli_flag_inventory.md` (410L, 30 flags mapped), `flag_selection_rationale.md` (425L, high-priority justification), `flag_naming_decisions.md` (TF naming harmonization), `argparse_schema.md` (complete argparse schema). (C4.B) Established RED baseline with 10 failing CLI tests (`tests/torch/test_cli_train_torch.py` 6 tests, `tests/torch/test_cli_inference_torch.py` 4 tests); captured `pytest_cli_train_red.log` and `pytest_cli_inference_red.log` showing argparse `unrecognized arguments` + mock assertion failures. (C4.C) **Production Refactor:** Training CLI (`ptycho_torch/train.py:381-452` argparse, `513-560` factory delegation) and Inference CLI (`ptycho_torch/inference.py:365-412` argparse, bundle loader integration) now use `create_training_payload()` / `load_inference_bundle_torch()` with execution config threading; eliminated hardcoded `nphotons`, `K`, `experiment_name` (documented in `refactor_notes.md`). (C4.D) **GREEN Validation:** Targeted CLI tests 10/10 PASSED (`pytest_cli_train_green.log`, `pytest_cli_inference_green.log`), factory smoke 6/6 PASSED, integration test PASSED (16.77s, gridsize=2 validated), manual CLI smoke ✅ (gridsize=2, deterministic, CPU). Full suite: 268 passed, 17 skipped, 1 xfailed, 0 new failures. **C4.D Blocker Resolution:** Addressed inference CLI factory bypass + gridsize=2 memmap drift via `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T083500Z/phase_c4d_blockers/plan.md`; regenerated minimal fixture (64 positions, stratified sampling, SHA256 verified). Evidence: `reports/2025-10-20T111500Z/phase_c4d_at_parallel/summary.md`. (C4.E) **Documentation Updates:** Updated `docs/workflows/pytorch.md` §12 (CLI flags + CONFIG-001 compliance, 68 lines), `specs/ptychodus_api_spec.md` §7 (CLI reference tables, 38 lines), `CLAUDE.md` §5 (PyTorch training example, 18 lines), `plans/active/ADR-003-BACKEND-API/implementation.md` (C4 completion note). All examples validated against Phase C4.D evidence. (C4.F) **Close-Out:** Authored comprehensive summary (`reports/2025-10-20T123500Z/phase_c4f_closeout/summary.md`, ~550 lines), captured Phase D prerequisites (checkpoint callbacks, logger backend governance, scheduler factory, DataLoader tuning), verified hygiene (no stray artifacts). **Exit Criteria:** All C4 tasks `[x]` (24/24 checklist items), TDD discipline maintained (RED→GREEN cycles captured), CONFIG-001/POLICY-001/DATA-001 compliance verified, 4 files documented (workflow guide, spec, CLAUDE.md, implementation plan), 10 new CLI tests (100% pass rate), ~120 lines production code added, ~260 lines test code added. **Deferred to Phase D:** 9 execution knobs (checkpoint-save-top-k, checkpoint-monitor-metric, early-stop-patience, logger-backend, scheduler, prefetch-factor, persistent-workers, middle-trim, pad-eval) requiring governance decisions + callback wiring beyond argparse-to-dataclass mapping. **Artifacts:** `plans/active/ADR-003-BACKEND-API/reports/{2025-10-20T033100Z/phase_c4_cli_integration/, 2025-10-20T044500Z/, 2025-10-20T050500Z/, 2025-10-20T111500Z/phase_c4d_at_parallel/, 2025-10-20T120500Z/phase_c4_docs_update/, 2025-10-20T123500Z/phase_c4f_closeout/}`. **Next:** Phase D execution plan (checkpoint callbacks + logger backend governance).
```

## Hygiene Verification Summary

**Commands Run:**
```bash
git status --porcelain
ls -la /home/ollie/Documents/PtychoPINN2/ | head -40
find plans/active/ADR-003-BACKEND-API/reports/2025-10-20T* -type f | wc -l
```

**Findings:**
- ✅ No stray log files at repo root
- ✅ All Phase C4 artifacts under `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T*/`
- ✅ 42 artifact files across 6 timestamped directories
- ⚠️ Legacy PNG/log files present from pre-ADR-003 work (not created by this initiative; ignored per scope)

**Decision:** Proceed with ledger update.

## Ledger Location

**File:** `docs/fix_plan.md`
**Section:** `[ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003`
**Position:** Append to "Attempts History" after Attempt #17 (last recorded attempt)
