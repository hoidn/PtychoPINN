Summary: Fix the Phase C metadata guard so the dense relaunch can clear [1/8] and finish Phase C→G with fresh metrics evidence.
Mode: TDD
Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 - Phase G comparison & analysis (dense real evidence + automated report)
Branch: feature/torchapi-newprompt
Mapped tests:
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_handles_patched_layout -vv
  - pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T081826Z/phase_c_metadata_guard_blocker/

Do Now:
- Focus: STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
- Implement: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py::validate_phase_c_metadata — walk the `dose_*` directories, require `patched_{train,test}.npz`+`_metadata`, and add pytest coverage in tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_handles_patched_layout so the new layout is exercised.
- Execute: python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber (rerun after the guard fix, capture a new run log, and confirm `[8/8]`).
- Document: Update "$HUB"/summary/summary.md with runtime/MS-SSIM/MAE deltas + guard evidence, refresh docs/fix_plan.md Attempts History, and note highlights/digest provenance.
- Validate: pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_handles_patched_layout -vv && pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_post_run_$(date -u +%Y-%m-%dT%H%M%SZ).log
- Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T081826Z/phase_c_metadata_guard_blocker/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. HUB=/home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run
3. rm -f "$HUB"/analysis/blocker.log
4. pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_handles_patched_layout -vv
5. RUN_LOG="$HUB"/cli/run_phase_g_dense_relaunch_$(date -u +%Y-%m-%dT%H%M%SZ).log
6. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub "$HUB" --dose 1000 --view dense --splits train test --clobber |& tee "$RUN_LOG"
7. rg "\[[1-8]/8\]" "$RUN_LOG" && tail -n 40 "$RUN_LOG"
8. for dose in 1000 10000 100000; do for split in train test; do test -f "$HUB"/data/phase_c/dose_${dose}/patched_${split}.npz || { echo "Missing Phase C $dose $split"; exit 1; }; done; done
9. for f in metrics_summary.json metrics_delta_summary.json metrics_delta_highlights.txt metrics_delta_highlights_preview.txt aggregate_highlights.txt aggregate_report.md metrics_digest.md; do test -f "$HUB"/analysis/"$f" || { echo "Missing $f"; exit 1; }; done
10. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/check_dense_highlights_match.py --hub "$PWD/$HUB" | tee "$HUB"/analysis/highlights_consistency_check_$(date -u +%Y-%m-%dT%H%M%SZ).log
11. python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/analyze_dense_metrics.py --metrics "$PWD/$HUB"/analysis/metrics_summary.json --highlights "$PWD/$HUB"/analysis/aggregate_highlights.txt --output "$PWD/$HUB"/analysis/metrics_digest.md | tee "$HUB"/analysis/metrics_digest_refresh_$(date -u +%Y-%m-%dT%H%M%SZ).log
12. find "$HUB" -maxdepth 3 -type f | sort > "$HUB"/analysis/artifact_inventory_$(date -u +%Y-%m-%dT%H%M%SZ).txt
13. pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest -vv | tee "$HUB"/green/pytest_orchestrator_dense_exec_post_run_$(date -u +%Y-%m-%dT%H%M%SZ).log
14. Edit "$HUB"/summary/summary.md and docs/fix_plan.md with runtime deltas, guard outcome, highlights/digest provenance, and artifact references.

Pitfalls To Avoid:
- Do not loosen metadata validation by skipping checks; assert `_metadata` on both patched_train/test NPZ files for each dose.
- Keep hub paths absolute (use `$PWD/$HUB` when invoking helpers) to satisfy TYPE-PATH-001.
- Avoid deleting Phase C data manually; rely on `--clobber` and keep archives under `$HUB`/archive/.
- Do not bypass AUTHORITATIVE_CMDS_DOC or highlights consistency; failures there remain hard blockers.
- Ensure pytest selectors run after code changes; stubs in tests/study/test_phase_g_dense_orchestrator.py must not regress other tests.
- Preserve GPU availability for the pipeline run; delay other GPU tasks until `[8/8]` completes.
- Keep blocker evidence if the guard still fails; capture logs before restarting.
- When editing summary/docs, reference the new artifact inventory and log filenames explicitly.
- Leave `run_phase_g_dense.py` command order unchanged; only adjust metadata validation logic.
- Respect STUDY-001 reporting requirements: MS-SSIM/MAE deltas vs Baseline and PtyChi with provenance links.

If Blocked:
- Capture the failing command, exit code, and append the last 40 lines of the relevant log to "$HUB"/cli/blocker_$(date -u +%Y%m%dT%H%M%SZ).log.
- Note the blocker (including exception text) in "$HUB"/summary/summary.md and docs/fix_plan.md, marking the attempt `blocked` with artifact references.
- If the validator still throws, archive the generated temp structure under plans/active/.../reports/2025-11-06T081826Z/phase_c_metadata_guard_blocker/ and stop for supervisor guidance.

Findings Applied (Mandatory):
- POLICY-001 — Keep PyTorch dependency assumptions untouched during the guard fix.
- CONFIG-001 — Ensure update_legacy_dict workflow remains intact inside Phase C helpers; do not alter init ordering.
- DATA-001 — Use MetadataManager to verify `_metadata` rather than hand-inspecting NPZ arrays.
- TYPE-PATH-001 — Maintain Path-normalized inputs/outputs inside the orchestrator and helper scripts.
- OVERSAMPLING-001 — Leave dense K/gridsize parameters as-is; guard work should be layout-only.
- STUDY-001 — Continue reporting MS-SSIM/MAE deltas vs Baseline and PtyChi with artifact links.
- PHASEC-METADATA-001 — Update validation to match the modern `dose_*` + patched_{train,test}.npz layout to avoid false blockers.

Pointers:
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:194
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:575
- tests/study/test_phase_g_dense_orchestrator.py:1
- /home/ollie/Documents/PtychoPINN2/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T115706Z/phase_g_dense_full_execution_real_run/cli/run_phase_g_dense_relaunch_2025-11-06T074519Z.log:1
- plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-06T081826Z/phase_c_metadata_guard_blocker/cli/blocker_phase_c_metadata.log:1
- docs/TESTING_GUIDE.md:333
- docs/findings.md:15

Next Up (optional):
- Stage sparse view relaunch prerequisites once dense view evidence is complete.

Doc Sync Plan (Conditional):
- Not applicable; selectors unchanged beyond the new unit test added this loop.

Mapped Tests Guardrail:
- Confirm `pytest tests/study/test_phase_g_dense_orchestrator.py::test_validate_phase_c_metadata_handles_patched_layout --collect-only -q` and `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_exec_runs_analyze_digest --collect-only -q` each collect 1 test before finishing.

Hard Gate:
- Do not mark the focus complete until the dense pipeline run shows `[8/8]`, Phase D–G artifacts + refreshed metrics digest exist, highlights consistency passes, both mapped pytest selectors pass with new logs under `$HUB`/green/, and docs/fix_plan.md + summary include MS-SSIM/MAE deltas with provenance.
