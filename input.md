Summary: Produce real Phase G dense/train,test evidence by scripting the Phase C→G pipeline and archiving metrics under the new hub.
Mode: Perf
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (dense execution evidence)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G2 dense comparisons real-run evidence
  - Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::main — orchestrate Phase C→G commands for a given hub, enforcing TYPE-PATH-001 via `Path`, propagating `AUTHORITATIVE_CMDS_DOC`, teeing stdout/stderr to per-phase log files, and halting on non-zero return codes with blocker log notes.
  - Validate: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs/green/pytest_phase_g_manifest_green.log
  - Execute: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs --dose 1000 --view dense --splits train test
  - Summarize: Inspect `{HUB}/analysis` and `{HUB}/summary` outputs, record MS-SSIM/MAE metrics and any anomalies in summary/summary.md, and update docs/fix_plan.md + galph_memory.md with artifact references.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs/green/pytest_phase_g_manifest_green.log
- python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs --dose 1000 --view dense --splits train test
- python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub ... --dose 1000 --view dense --splits train test --collect-only (if verification run needed without execution; optional guard)

Pitfalls To Avoid:
- Forgetting to normalize paths to `Path` objects inside the new script (violates TYPE-PATH-001).
- Running CLIs without `AUTHORITATIVE_CMDS_DOC`; keep the guard exported for each command.
- Allowing script to continue after a failed subprocess—must log blocker and exit immediately.
- Writing artifacts outside the hub; all logs/metrics belong under the timestamped directory.
- Skipping manifest validation; confirm Phase F outputs include `ptychi_reconstruction.npz` before launching comparisons.
- Modifying production modules (e.g., `ptycho/model.py`); scope is scripts + orchestration only.
- Ignoring CLI return codes when teeing output; ensure logs capture command exit status.
- Leaving temporary files in `tmp/`; clean up if the script creates scratch space.

If Blocked:
- Capture failing command + stderr in `{HUB}/analysis/blocker.log`, mark attempt `blocked` in docs/fix_plan.md, and update galph_memory.md with state=blocked and next steps.
- If training or reconstruction errors stem from missing datasets, record the manifest path and command, then stop—do not attempt ad-hoc dataset surgery.
- Should pytest selector fail unexpectedly, keep RED log, revert script changes if they caused regression, and request follow-up plan.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch backend remains required for LSQML; ensure environment retains torch>=2.2.
- CONFIG-001 — Training CLI already bridges legacy config; script must not bypass initialization order.
- DATA-001 — Regenerated NPZs must satisfy dataset contract; validator runs inside pipeline guard this.
- OVERSAMPLING-001 — Dense overlap spacing must remain per design; reuse CLI defaults without altering overlap ratios.
- TYPE-PATH-001 — All CLI inputs/outputs should leverage `Path` to avoid string/path mismatches.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:248 — Phase G execution expectations and evidence requirements.
- docs/COMMANDS_REFERENCE.md:259 — `scripts/compare_models.py` CLI arguments including `--tike_recon_path`.
- docs/TESTING_GUIDE.md:183 — Deterministic Phase F CLI invocation pattern with `AUTHORITATIVE_CMDS_DOC` guard.
- docs/findings.md:21 — TYPE-PATH-001 path normalization policy.
- studies/fly64_dose_overlap/comparison.py:161 — Manifest-driven comparison executor requiring real Phase F inputs.

Next Up (optional):
- Execute sparse view comparisons and archive metrics once dense evidence is green.

Doc Sync Plan:
- Run `pytest tests/study/test_dose_overlap_comparison.py --collect-only -k tike_recon_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs/collect/pytest_phase_g_manifest_collect.log` after GREEN pytest to reconfirm selector collection.

Mapped Tests Guardrail: Selector above collects ≥1 test; confirm via collect-only log.
