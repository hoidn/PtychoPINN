# Probe Mischaracterization Second-Pass Implementation Review

Decision: APPROVE

## Summary

Fresh implementation review found no unresolved high- or medium-severity in-scope findings. The required process-isolated numeric tranche is implemented, the full reviewer-facing grid completed with claim-safe gates, the paper-side metrics and figure trace back to the run root, and the manuscript/checklist/changelog use fixed-probe sensitivity language rather than a robustness or trainable-probe claim.

Reviewed consumed artifacts first:

- `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/approved_design.md`
- `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/implementation_plan.md`
- `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/execution_report.md`
- `state/revision-study-probe-second-20260413T011228Z/revision_context.md`
- `state/revision-study-probe-second-20260413T011228Z/implementation_open_findings.json`

Additional project and paper context checked: `docs/index.md`, `docs/findings.md`, `docs/DEVELOPER_GUIDE.md`, `docs/TESTING_GUIDE.md`, `docs/COMMANDS_REFERENCE.md`, and `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`.

## Prior Finding Reconciliation

The carried-forward open-findings ledger, `state/revision-study-probe-second-20260413T011228Z/implementation_open_findings.json`, contains no prior findings.

The stale review file being replaced had three historical entries. They are not carried forward as open findings:

- `IMPL-001`: RESOLVED. PDF/manuscript inspection evidence is present and the R3 probe-mischaracterization checklist item is closed with the fixed-probe sensitivity resolution note.
- `IMPL-002`: SUPERSEDED. The approved design explicitly makes parent/child source-fingerprint enforcement out of scope; I did not reintroduce a source-provenance gate.
- `IMPL-003`: RESOLVED. The current test suite covers smoke child-mode dispatch without parent `--output-root` preparation.

## Evidence

Implementation shape:

- `scripts/studies/probe_mischaracterization_stress_test.py` implements the study-local child CLI modes, canonical bundle serialization/validation, isolated smoke child, sequential condition child launches, PATH `python` subprocess commands, and `PYTHONPATH` pinning to the repo root.
- `tests/studies/test_probe_mischaracterization_stress_test.py` covers distinct normalization fields, bundle validation, child reconstruction/preflight, child-mode dispatch, source-provenance non-gates, subprocess PID recording, gate behavior, and paper export guards.
- `git diff -- ptycho/model.py ptycho/diffsim.py ptycho/tf_helper.py specs/data_contracts.md revision_designs/probe_mischaracterization_stress_test.md` is empty, so the protected stable-core files, shared data contract, and seed revision design are not edited.

Reviewer-facing run evidence:

- Run root: `.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_20260413T020739Z`
- `manifest.json` records `smoke_baseline_subprocess_isolation=true`, `per_condition_subprocess_isolation=true`, `child_launch_policy=smoke_then_conditions_sequential`, and `parent_trains_tensorflow_after_bundle_write=false`.
- The manifest records 11 child runs, including smoke, all with return code `0` and PIDs.
- `metrics.json` contains 10 requested reviewer-facing conditions, all with status `ok`.
- `canonical_condition_inputs_manifest.json` contains the required bundle fields, no bare `norm_Y_I`, the four distinct normalization field names, checksum/alias metadata, `YY_full` presence, and explicit absence reasons for absent grouping fields.
- `artifact_manifest.json` lists 119 artifacts; every listed path exists.
- The infrastructure gate passed, the baseline comparability gate is claim-safe, and the mild gate requires sensitivity language while still permitting export.

Paper and claim-boundary evidence:

- `/home/ollie/Documents/ptychopinnpaper2/data/probe_mischaracterization_metrics.json` records the run root, baseline policy, mild-perturbation gate, and `no_trainable_probe_variant_added=true`.
- `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png` has the same SHA-256 as the run-root figure: `955e5afb056a93c569034e3b25812dee105484443918f2adf35f5ab98b77cc16`.
- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex` states that the probe is supplied/pre-estimated and held fixed, reports the stress-test numbers with sensitivity language, and explicitly says not to interpret the result as robustness to probe error.
- `/home/ollie/Documents/ptychopinnpaper2/changelog.txt` and `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md` cite the same run root and paper-side assets.
- `pdftotext` inspection of `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.pdf` found the updated fixed-probe and sensitivity language. The PDF timestamp is newer than the TeX source. `latexmk` is unavailable in this environment, so I treated the existing fresh PDF text inspection as residual evidence rather than a new blocker.

Verification rerun in `ptycho311` during this review:

- `python -m pytest tests/studies/test_probe_mischaracterization_stress_test.py -v`: 32 passed in 11.39 s.
- `python -m pytest tests/test_grid_lines_invocation_logging.py -v`: 3 passed in 0.94 s.
- `python -m pytest tests/test_grid_lines_workflow.py -k "probe or pipeline or invocation" -v`: 22 passed, 29 deselected, 1 existing tight-layout warning in 3.87 s.
- `python -m py_compile scripts/studies/probe_mischaracterization_stress_test.py`: passed.

## Findings

No unresolved in-scope findings.

Non-blocking context:

- The current checkout and the paper repo contain unrelated dirty or untracked files from other revision work. They are outside this probe-mischaracterization tranche and do not touch the protected stable-core files or the seed probe-mischaracterization design.
- The parent run manifest records the Git `HEAD` from the working tree at launch time before the study script was later committed. The approved design explicitly excludes source-fingerprint enforcement and child Git commit matching gates, so I did not count that as a blocking implementation finding. The scientific artifact contract is instead carried by the run-local bundle manifest, child requests/invocations, checksums, gates, and paper metrics payload.
