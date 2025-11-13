# Phase D Overlap Metrics Implementation Hub (2025-11-12T010500Z)

<plan_update version="1.0">
  <trigger>Spec implementation + tests landed (d94f24f7 + green log), but the hub still lacks real CLI runs and metrics JSONs, so Phase G remains blocked.</trigger>
  <focus_id>STUDY-SYNTH-FLY64-DOSE-OVERLAP-001</focus_id>
  <documents_read>docs/index.md, docs/findings.md, specs/overlap_metrics.md, docs/GRIDSIZE_N_GROUPS_GUIDE.md, docs/TESTING_GUIDE.md, docs/development/TEST_SUITE_INDEX.md, docs/fix_plan.md, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md, galph_memory.md, input.md, studies/fly64_dose_overlap/overlap.py, tests/study/test_dose_overlap_overlap.py, plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics/{summary.md,analysis/artifact_inventory.txt}</documents_read>
  <current_plan_path>plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics/plan/plan.md</current_plan_path>
  <proposed_changes>Document the delivered commit/log, note the missing CLI artifacts, and restate the Do Now so Ralph runs the overlap CLI for gs1 + gs2 against data/phase_c/dose_1000 while archiving logs/metrics under this hub.</proposed_changes>
  <impacts>Phase G stays blocked until metrics JSON + bundle files exist; Phase E cannot resume until these outputs feed training manifests.</impacts>
  <ledger_updates>docs/fix_plan.md Latest Attempt, implementation.md Phase D update, input.md rewrite, summary.md Turn Summary.</ledger_updates>
  <status>approved</status>
</plan_update>

## Reality Check — 2025-11-12
- Commit `d94f24f7` rewrote `studies/fly64_dose_overlap/overlap.py` with disc-overlap math, Metric 1/2/3 helpers, deterministic `s_img` subsampling, and the new CLI arguments. The matching pytest module now has 18 targeted cases (log at `$HUB/green/pytest_phase_d_overlap.log`).
- The hub still lacks CLI executions: `cli/` and `metrics/` are empty, so there are no `train_metrics.json`, `test_metrics.json`, or `metrics_bundle.json` files for real Phase C data. Without those artifacts Phase G cannot rerun and Phase E cannot align manifests.
- Phase C dose_1000 data lives at `data/phase_c/dose_1000` with `patched_train.npz`/`patched_test.npz`, so nothing blocks immediate CLI runs once the environment guardrails are followed.

## Do Now — CLI metrics capture (blocking Phase E/G)
1. Guard the working directory and export the required env vars:
   ```bash
   test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"
   export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
   export HUB=$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics
   mkdir -p "$HUB"/{cli,metrics}
   ```
2. Refresh the Phase D pytest selector to tie evidence directly to these runs:
   ```bash
   pytest tests/study/test_dose_overlap_overlap.py::test_overlap_metrics_bundle -vv \
     | tee "$HUB"/green/pytest_phase_d_overlap_bundle_rerun.log
   ```
3. Execute the overlap CLI twice against `data/phase_c/dose_1000`, capturing stdout/stderr via `tee` and directing metrics to profile-specific folders:
   ```bash
   python -m studies.fly64_dose_overlap.overlap \
     --phase-c-root data/phase_c/dose_1000 \
     --output-root tmp/phase_d_overlap/gs1_s100_n512 \
     --artifact-root "$HUB"/metrics/gs1_s100_n512 \
     --gridsize 1 \
     --s-img 1.0 \
     --n-groups 512 \
     --neighbor-count 6 \
     --probe-diameter-px 38.4 \
     --rng-seed-subsample 456 \
     |& tee "$HUB"/cli/phase_d_overlap_gs1.log

   python -m studies.fly64_dose_overlap.overlap \
     --phase-c-root data/phase_c/dose_1000 \
     --output-root tmp/phase_d_overlap/gs2_s080_n512 \
     --artifact-root "$HUB"/metrics/gs2_s080_n512 \
     --gridsize 2 \
     --s-img 0.8 \
     --n-groups 512 \
     --neighbor-count 6 \
     --probe-diameter-px 38.4 \
     --rng-seed-subsample 456 \
     |& tee "$HUB"/cli/phase_d_overlap_gs2.log
   ```
4. Verify that each run produced `train_metrics.json`, `test_metrics.json`, and `metrics_bundle.json` inside `$HUB/metrics/<profile>/` with Metric 1/2/3 averages (Metric 1 omitted for gs1). Note the observed values in `summary.md`, refresh `analysis/artifact_inventory.txt`, and, if any command fails, store the log under `$HUB/red/` with command + exit code.

## Evidence expectations
- GREEN pytest log for the bundle selector (`green/pytest_phase_d_overlap_bundle_rerun.log`).
- CLI stdout/stderr per run under `cli/phase_d_overlap_gs1.log` and `cli/phase_d_overlap_gs2.log`.
- Metrics artifacts copied into `$HUB/metrics/gs1_s100_n512/` and `$HUB/metrics/gs2_s080_n512/` (each containing `train_metrics.json`, `test_metrics.json`, `metrics_bundle.json`).
- Updated `analysis/artifact_inventory.txt` listing the new files plus command references, and a short narrative snippet in `summary/` or `summary.md` describing Metric 1/2/3 outcomes.

## References
- `docs/index.md` — authoritative doc map.
- `docs/findings.md` — POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001, ACCEPTANCE-001 guardrails.
- `specs/overlap_metrics.md` — Metric 1/2/3 definitions + CLI contract.
- `docs/GRIDSIZE_N_GROUPS_GUIDE.md` — unified `n_groups` semantics.
- `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md` — pytest selectors + evidence policy.
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md` — Phased plan + Do Now context.
