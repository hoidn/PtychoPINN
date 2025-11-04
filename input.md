Summary: Wire Phase F manifests into Phase G comparisons and prove it with dense/train,test reruns.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase G comparison & analysis (manifest-driven G2.1 dense execution)
Branch: feature/torchapi-newprompt
Mapped tests: pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/

Do Now:
- STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — G2.1 dense comparisons manifest integration
  - Implement: studies/fly64_dose_overlap/comparison.py::execute_comparison_jobs — read each Phase F manifest, require `ptychi_reconstruction.npz`, and append `--tike_recon_path` using `Path` objects (cache manifest parsing to avoid redundant IO).
  - Implement: tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_appends_tike_recon_path — extend fixtures to emit manifest JSON + recon files; assert RED failure when missing and GREEN when present.
  - Validate (RED): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/red/pytest_phase_g_manifest_red.log
  - Validate (GREEN): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/green/pytest_phase_g_manifest_green.log
  - Phase C regeneration (dose=1000): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.generation --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz --output-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/cli/phase_c_generation.log
  - Phase D dense overlap: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.overlap --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --output-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/analysis --doses 1000 --views dense | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/cli/phase_d_overlap.log
  - Phase E training (baseline gs1, TensorFlow): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-d-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/training --dose 1000 --view baseline --gridsize 1 --backend tensorflow | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/cli/training_baseline.log
  - Phase E training (dense gs2, TensorFlow): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.training --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-d-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/training --dose 1000 --view dense --gridsize 2 --backend tensorflow | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/cli/training_dense.log
  - Phase F LSQML recon (dense/train): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-d-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/reconstruction --dose 1000 --view dense --split train | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/cli/reconstruction_dense_train.log
  - Phase F LSQML recon (dense/test): AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-d-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/reconstruction --dose 1000 --view dense --split test | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/cli/reconstruction_dense_test.log
  - Phase G comparisons: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-e-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/training --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/reconstruction --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/analysis --dose 1000 --view dense --split train | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/cli/comparison_dense_train.log
  - Phase G comparisons: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.comparison --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-e-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/training --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/reconstruction --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/analysis --dose 1000 --view dense --split test | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/cli/comparison_dense_test.log
  - Archive outputs: summarize checksums/manifests in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/analysis/summary.md.

How-To Map:
- export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
- pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv
- python -m studies.fly64_dose_overlap.generation --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz --output-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c
- python -m studies.fly64_dose_overlap.overlap --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --output-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/analysis --doses 1000 --views dense
- python -m studies.fly64_dose_overlap.training --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-d-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/training --dose 1000 --view baseline --gridsize 1 --backend tensorflow
- python -m studies.fly64_dose_overlap.training --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-d-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/training --dose 1000 --view dense --gridsize 2 --backend tensorflow
- python -m studies.fly64_dose_overlap.reconstruction --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-d-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/reconstruction --dose 1000 --view dense --split train
- python -m studies.fly64_dose_overlap.reconstruction --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-d-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_d --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/reconstruction --dose 1000 --view dense --split test
- python -m studies.fly64_dose_overlap.comparison --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-e-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/training --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/reconstruction --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/analysis --dose 1000 --view dense --split train
- python -m studies.fly64_dose_overlap.comparison --phase-c-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/phase_c --phase-e-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/training --phase-f-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/data/reconstruction --artifact-root plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/analysis --dose 1000 --view dense --split test

Pitfalls To Avoid:
- Do not bypass CONFIG-001; confirm training CLI logs show legacy dict bridge.
- Do not hardcode string paths; wrap all filesystem inputs with `Path()` (TYPE-PATH-001).
- Avoid mutating manifests in place; treat JSON as read-only evidence and fail fast if keys missing.
- Keep CLI runs deterministic (CPU-only, fixed seeds per docs/TESTING_GUIDE.md).
- Capture RED/GREEN pytest logs and CLI transcripts under the hub; no artifacts at repo root.
- If Phase F recon fails, stop comparisons and document blocker rather than hacking around missing data.
- Ensure new pytest selector actually collects >0 tests before marking GREEN.
- No environment changes; rely on existing deps per Environment Freeze policy.
- Resist editing core physics modules (`ptycho/model.py`, etc.) unless plan explicitly says so (it does not).
- Clean up temporary files under `tmp/` if created during execution.

If Blocked:
- Record failure details + command in plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/analysis/blocker.log and note in docs/fix_plan.md Attempts History.
- Mark state `blocked` in galph_memory.md with reason, including CLI return code / stack trace.
- Pivot to regenerating prerequisite phases only if comparison wiring passes but a downstream CLI flakes; otherwise stop and await guidance.

Findings Applied (Mandatory):
- POLICY-001 — PyTorch backend requirements remain in force; recon CLI must run without disabling torch.
- CONFIG-001 — Ensure `update_legacy_dict` bridge remains exercised before TensorFlow workflows.
- DATA-001 — Regenerated NPZs must satisfy contract validated by existing CLI checks.
- OVERSAMPLING-001 — Dense overlap parameters already fixed; do not regress spacing constraints when rebuilding data.
- TYPE-PATH-001 — Normalize Phase F paths via `Path` to prevent string/Path mismatches.

Pointers:
- docs/TESTING_GUIDE.md:183 — Phase G execution checklists and deterministic flags.
- docs/COMMANDS_REFERENCE.md:259 — `scripts.compare_models` required arguments including `--tike_recon_path`.
- specs/ptychodus_api_spec.md:220 — Phase F output expectations consumed by Phase G.
- studies/fly64_dose_overlap/comparison.py:169 — Current CLI command assembly missing Phase F recon flag.
- tests/study/test_dose_overlap_comparison.py:140 — Existing execute_comparison_jobs coverage to extend for manifest-driven flag.
- docs/fix_plan.md:36 — Active attempt log for this initiative.

Next Up (optional):
- Regenerate sparse view comparisons once dense path is green to complete Phase G coverage.

Doc Sync Plan:
- After GREEN pytest, run `pytest tests/study/test_dose_overlap_comparison.py --collect-only -k tike_recon_path -vv | tee plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/collect/pytest_phase_g_manifest_collect.log` and update summary with collected node count.

Mapped Tests Guardrail: Selector above must collect ≥1 test; confirm via collect-only log before marking attempt done.
