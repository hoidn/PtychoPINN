# Phase G Manifest-Driven Comparison Plan

**Timestamp:** 2025-11-07T03:05:00Z  
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis  
**Loop Role:** Supervisor (planning)  
**Action State Target:** ready_for_implementation  

## Evidence Reviewed
- `docs/fix_plan.md` — latest attempts confirm comparison builder path fix landed (2025-11-07T010500Z+exec) but Phase C/D/E/F regeneration still pending.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T010500Z/phase_g_execution_real_runs/analysis/summary.md` — documents the new test enforcing dose-specific Phase E checkpoint layout; no execution evidence yet.
- `tests/study/test_dose_overlap_comparison.py` — fixture currently creates empty Phase F manifests; command construction omits `--tike_recon_path`.
- `docs/findings.md` (POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001, TYPE-PATH-001) — guardrails that remain applicable for Phase G execution.
- `docs/COMMANDS_REFERENCE.md` §compare_models and `docs/TESTING_GUIDE.md` §Phase G — confirm three-way comparisons must use iterative recon (`--tike_recon_path`) sourced from Phase F outputs.

## Gap Analysis
- `execute_comparison_jobs()` never reads the Phase F manifest to locate `ptychi_reconstruction.npz`; comparisons currently invoke `scripts.compare_models` without the iterative baseline, violating Phase G acceptance criteria.
- Phase G data roots from earlier attempts were cleaned; fresh Phase C/D/E/F artifacts must be regenerated inside this loop’s hub before comparisons can execute.
- Tests lack coverage for manifest-driven argument wiring; need a RED case that fails without the new flag and GREEN after implementation.

## Do Now (Engineer) Outline
1. **Code:** Extend `studies/fly64_dose_overlap/comparison.py::execute_comparison_jobs` to parse each job’s `phase_f_manifest`, derive `ptychi_reconstruction.npz` (or error if missing), and append `--tike_recon_path` to the compare command. Add minimal helper to memoize manifest JSON reads to avoid duplicate parsing.
2. **Tests:** Update `tests/study/test_dose_overlap_comparison.py` fixtures to emit manifest JSON entries with `jobs[0]["output_dir"]` and create dummy `ptychi_reconstruction.npz`. Add a focused test (RED→GREEN) asserting command includes the `--tike_recon_path` argument and fails fast when the NPZ is absent.
3. **Execution:** Using `plans/active/.../reports/2025-11-07T030500Z/phase_g_execution_real_runs/` as the hub, regenerate dose=1000 dense assets (Phase C→F), rerun deterministic training (gs1 baseline + gs2 dense), execute LSQML recon, then run `comparison` CLI for dense/train and dense/test with logs under `cli/`. Capture manifests, checksums, and comparison outputs under this hub.
4. **Validation:** Run targeted pytest selector after code changes (`pytest tests/study/test_dose_overlap_comparison.py -k manifest --maxfail=1 -vv`) and archive RED/GREEN logs; rerun full study suite if runtime permits, otherwise document deferral.

## Artifact Expectations
- `plan/plan.md` — how-to breakdown for Ralph (this loop).
- `analysis/summary.md` (this file) — planning rationale and references.
- RED/GREEN pytest logs under `red/` and `green/` with updated filenames (e.g., `pytest_phase_g_manifest_red.log`).
- CLI transcripts in `cli/` for generation, overlap, training, reconstruction, and comparisons.
- Updated `comparison_manifest.json` containing execution results with `tike_recon_path` reflected in `execution_results` entries.

## Risks & Mitigations
- **Missing pty-chi recon:** If Phase F rerun still fails, capture failure log in `analysis/blocker.log` and mark Attempt blocked; fallback is to scope execution to whichever conditions succeed.
- **Runtime drift:** Ensure deterministic seeds and CPU-only flags remain consistent with prior Phase E/F runs to keep evidence comparable (see `docs/TESTING_GUIDE.md` §Deterministic Runs).
- **Path regressions (TYPE-PATH-001):** Maintain `Path` coercion for all derived paths when parsing manifests; add explicit assertions in tests to guard.

## References
- docs/COMMANDS_REFERENCE.md:259 — canonical `scripts/compare_models.py` invocation including `--tike_recon_path`.
- docs/TESTING_GUIDE.md:183 — Phase G CLI execution requirements.
- specs/ptychodus_api_spec.md:220 — Phase E/F artifact expectations consumed by Phase G.
