# Phase G Manifest Wiring — Planning Loop (2025-11-07T05:05:00Z)

**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis  
**Loop Role:** Supervisor (planning)  
**Target State:** ready_for_implementation

## Evidence Reviewed
- `timeout 30 git pull --rebase` → already up to date (no upstream drift).
- `docs/fix_plan.md` — confirmed Phase G item still `in_progress`; noted prior planning loop (2025-11-07T03:05Z) flagged the manifest wiring gap.
- `plans/active/.../reports/2025-11-07T030500Z/phase_g_execution_real_runs/analysis/summary.md` — recapped scope for manifest-driven comparisons and outstanding execution evidence.
- `studies/fly64_dose_overlap/comparison.py` — verified `execute_comparison_jobs()` still shells out without `--tike_recon_path` despite each job having `phase_f_manifest` pointing at Phase F output.
- `tests/study/test_dose_overlap_comparison.py` — fixture currently stubs manifests as empty files; no coverage that the comparison CLI receives the Phase F reconstruction path.
- `docs/findings.md` (POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001, TYPE-PATH-001) — reaffirmed backend + path guardrails for Phase F/G.
- `docs/TESTING_GUIDE.md` §Phase G & `docs/COMMANDS_REFERENCE.md` §compare_models — confirm three-way runs require the iterative baseline via `--tike_recon_path`.

## Gap Analysis
- `execute_comparison_jobs()` never inspects `phase_f_manifest`; comparisons therefore omit the pty-chi reconstruction, breaking Phase G acceptance criteria and contradicting specs §4.8.
- Tests lack RED coverage for the missing flag; fixtures do not synthesize manifest JSON or the `ptychi_reconstruction.npz` file expected by `scripts.compare_models`.
- No fresh Phase C→F artifacts exist under a current hub; execution evidence from Phase G dense runs must be regenerated once code lands.

## Loop Objectives
1. Hand Ralph an implementation-ready Do Now that augments `execute_comparison_jobs()` to read each manifest, validate `ptychi_reconstruction.npz`, and append `--tike_recon_path` (Path-coerced, aligns with TYPE-PATH-001).
2. Require pytest coverage that fails without the new argument and passes once added, using fixture-generated manifests and reconstruction files.
3. Stage full Phase C→F regeneration (dose=1000, view=dense, splits train/test) followed by comparison CLI runs, with logs/artifacts saved in this hub.
4. Document guardrails (legacy config bridge, deterministic flags, manifest integrity) so execution evidence remains reproducible.

## Risks & Mitigations
- **Manifest schema drift:** Parse via `json.load` with explicit key assertions; emit clear errors if `ptychi_reconstruction.npz` missing.
- **Path normalization regressions:** Ensure all derived paths wrap with `Path()` before usage (TYPE-PATH-001) and extend tests to assert string→Path conversion.
- **Long-running CLI jobs:** Use deterministic seeds and CPU execution flags already codified in CLI wrappers; if CLI fails, capture logs under `analysis/blocker.log` and mark attempt blocked.

## Next Steps for Ralph
- Implement manifest parsing + CLI flag injection in `comparison.py` with supporting helpers/tests.
- Run targeted selector `pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv` (RED→GREEN) archiving logs in `red/` and `green/`.
- Regenerate Phase C/D/E/F assets and execute dense/train + dense/test comparisons, saving transcripts under `cli/` and manifests under `analysis/`.
- Update ledger + findings adherence post-execution.
