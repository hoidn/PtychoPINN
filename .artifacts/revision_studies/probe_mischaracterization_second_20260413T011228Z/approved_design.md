# Approved Revision-Study Design: Probe Mischaracterization Second Pass

Status: revised design for implementation planning after design review
Created: 2026-04-13
Approved design path pointer: `state/revision-study-probe-second-20260413T011228Z/approved_design_path.txt`
Seed provenance: `revision_designs/probe_mischaracterization_stress_test.md`
Revision context: `state/revision-study-probe-second-20260413T011228Z/revision_context.md`
Update note: source-provenance enforcement is intentionally out of scope. The
study records ordinary parent/child invocation metadata, but it must not block
execution or export on parent/child source fingerprints, child Git commit
matching, or source-fingerprint drift.

The seed design is immutable provenance and must not be edited in place. This
document supersedes neither the seed nor the prior first-pass design. It
revises the second-pass design at the output-contract path to address
`DR-PROBE-SECOND-001`, the unresolved design-review finding about the child
process input bundle and normalization contract.

## Consumed Inputs and Authority

Authoritative consumed artifacts:

- `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/approved_design.md`
- `.artifacts/revision_studies/probe_mischaracterization_second_20260413T011228Z/design_review.json`
- `state/revision-study-probe-second-20260413T011228Z/revision_context.md`
- `revision_designs/probe_mischaracterization_stress_test.md`

Additional project context checked for this revision:

- `docs/index.md`
- `docs/findings.md`
- `docs/DEVELOPER_GUIDE.md`
- `ptycho/loader.py`
- `ptycho/workflows/grid_lines_workflow.py`
- `ptycho/data_preprocessing.py`

Authority order:

1. The consumed artifacts listed above and the current output contract.
2. Project docs, findings, and stable-core policies.
3. The current design-review finding `DR-PROBE-SECOND-001`.
4. Prior run artifacts and implementation-review findings.
5. The seed design as immutable provenance.

## Reviewer Issue and Manuscript Scope

Reviewer 3 comment 2 asks for a numerical stress test that injects controlled
error into the assumed probe. The scientific concern remains the same as in the
seed: the overlap-free `C_g=1` result depends on a supplied structured probe, so
probe mischaracterization may degrade reconstruction.

Related reviewer and manuscript scope:

- Reviewer 1 asks whether the probe is solved as an unknown or supplied.
- Reviewer 2 asks that the fixed-probe limitation be addressed more directly.
- The manuscript should not imply trainable-probe, joint probe, or position
  refinement variants.
- If a numeric result is produced, likely manuscript anchors are
  `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex` Methods,
  Results near Table 2, and Discussion.
- If a clean numeric result is not produced, the response should become a
  scoped text-only fixed-probe limitation rather than a partial stress-test
  claim.

Current first-pass evidence:

- The existing smoke gate selected `fixed_wrong_probe_training`: both container
  probe replacement and direct `ProbeIllumination` variable mutation produced
  zero object-output delta.
- The corrected full run
  `.artifacts/revision_studies/probe_mischaracterization/full_tranched_20260413T003740Z/`
  exited `2` via `infrastructure_failure_gate.status: stop_full_grid`.
- That run produced a diagnostic baseline only: amplitude SSIM
  `0.9140266098362918` and amplitude PSNR `69.11115271659475`. These values
  are not reviewer-facing evidence because the full grid did not reach the
  baseline-comparability, mild-perturbation, figure, or export gates.
- All nine non-baseline conditions failed after preflight, mostly with repeated
  `InternalError()` plus one `ResourceExhaustedError()`. No stress figure or
  paper export was produced.

## Decision Summary

Continue the numeric study only through a bounded process-isolation design. Use
the existing one-off study script as the owner, but extend it with a study-local
parent/child condition-runner mode so each TensorFlow training condition starts
in a fresh Python process.

Do not promote perturbation utilities to shared workflow APIs, do not change
the project NPZ data contract, and do not edit stable core physics/model files.
If process isolation still cannot produce a complete claim-safe grid, pivot to
a text-only reviewer response using the smoke-gate and OOM-stop evidence.

## Proposed Implementation Shape and Rationale

Chosen shape: modify the existing study-local script,
`scripts/studies/probe_mischaracterization_stress_test.py`, and its focused test
module, `tests/studies/test_probe_mischaracterization_stress_test.py`.

Rationale:

- A fresh one-off script is no longer the right shape because the repo already
  contains a study-local CLI, perturbation grid, runtime metadata capture, smoke gate,
  preflight checks, metrics aggregation, figure generation, export gates, and a
  matching test suite.
- Retrying in-process TensorFlow cleanup is not sufficient. The first-pass
  implementation already applied the focused cleanup from
  `TF-REPEATED-MODEL-OOM-001`, dropped the retained smoke baseline model, and
  still stopped through the infrastructure gate.
- A reusable workflow/API change is too broad. This is a reviewer-revision
  stress study, and the needed isolation can stay local to the study script.
- The design-review gap is at the parent/child artifact boundary, not in the
  scientific perturbation grid or the core model stack. Tighten that boundary
  by making the child input bundle a complete container-reconstruction contract
  with explicit normalization metadata.
- A direct text-only pivot is acceptable only after the bounded isolation
  attempt fails or proves scientifically unclean. Reviewer 3 specifically asked
  for a numerical stress test, so one more clean numeric attempt is justified if
  it does not touch stable core code.

Alternatives considered:

- Use the PyTorch backend: not selected for this pass. The current script and
  Table 2 parity path are TensorFlow grid-lines based, and the failure is
  process lifetime and TensorFlow allocator state, not an architectural need to
  port the study.
- Change `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`: not
  approved. Stable core files remain out of scope.
- Add reusable perturbation or process-runner utilities to
  `ptycho/workflows/grid_lines_workflow.py`: not approved unless a later plan
  demonstrates reuse outside this study.

## Study-Local Process-Isolation Architecture

Add an internal child mode to the existing CLI rather than a new top-level
workflow:

```text
parent process
  -> build canonical true probe
  -> simulate true-probe train/test data once
  -> persist canonical condition input bundle and checksums
  -> persist one assumed-probe file plus request JSON per condition
  -> launch one child Python process per condition, sequentially
  -> harvest condition metrics and return codes
  -> apply infrastructure, baseline, mild-perturbation, and export gates
```

Child process responsibilities:

```text
child process for condition_id
  -> load canonical condition input bundle
  -> load condition assumed probe
  -> validate bundle manifest, field presence/absence markers, and checksums
  -> reconstruct train/test PtychoDataContainer objects with the assumed probe
  -> assert canonical true-measurement checksums, normalization-field checksums,
     constructor-input checksums, and assumed-probe checksums
  -> configure legacy params with the assumed probe
  -> train/infer/stitch/evaluate only this condition
  -> write conditions/<condition_id>/metrics.json and optional train/recon files
  -> exit with a status recorded by the parent
```

The parent must launch children sequentially, not concurrently, to avoid GPU
contention. It should use PATH `python` in persisted commands and record the
child command, return code, start/end times, and output paths in the run
manifest. Internally, Python `subprocess.run([...], check=False)` is acceptable
because it tracks the exact child process. Shell runbooks used outside the
script must still follow the project long-run guardrail:
`cmd ... & pid=$!; wait "$pid"`.

To support child reconstruction without reusing live TensorFlow state, extend
the existing persisted canonical measurement artifact or add a new
study-local NPZ such as `canonical_condition_inputs.npz`, paired with a small
bundle manifest. This child bundle must be a complete study-local reconstruction
contract for the current `PtychoDataContainer` path, not a display/inspection
subset.

Required arrays and fields:

- `X_train`, `X_test`
- `Y_I_train`, `Y_I_test`
- `Y_phi_train`, `Y_phi_test`
- `coords_nominal_train`, `coords_nominal_test`
- `coords_true_train`, `coords_true_test`
- `YY_full_train`, `YY_full_test`
- `YY_ground_truth_test`, if available for metric/stitching parity
- `coords_offsets_train`, `coords_offsets_test`, if produced by the grid-lines
  path
- `nn_indices_train`, `nn_indices_test`, if present on the source containers
- `global_offsets_train`, `global_offsets_test`, if present on the source
  containers
- `local_offsets_train`, `local_offsets_test`, if present on the source
  containers
- `probe_true`, the canonical true probe used for measurement generation
- `probe_assumed_<condition_id>` or an equivalent per-condition assumed-probe
  NPZ written outside the canonical data bundle

Required normalization and physics fields must use distinct names:

- `norm_Y_I_train_container`: value passed to the train
  `PtychoDataContainer` constructor.
- `norm_Y_I_test_container`: value passed to the test
  `PtychoDataContainer` constructor.
- `norm_Y_I_test_stitch`: the test stitching/display normalization returned
  by the grid-lines path for reconstructing and evaluating the test image.
- `intensity_scale_model`: the model/physics intensity scale recorded in
  `params.cfg["intensity_scale"]` and used by the training/inference path.

Do not write or read a bare `norm_Y_I` field in the process-isolation bundle.
If two of these values are numerically identical for a particular run, the
manifest may record that aliasing explicitly, but the fields and checksums must
remain distinct so the child cannot mix the physics, container, and
stitching/display normalization systems.

The bundle manifest must also record, per field:

- dtype, shape, and checksum
- source split (`train`, `test`, or global)
- whether the field is required, present, or explicitly absent
- the exact `PtychoDataContainer(...)` constructor mapping used by the child
- checksum policy for fixed true-measurement arrays and per-condition assumed
  probes

If `YY_full` or any grouping offsets are absent in a source container, the
manifest must record explicit absence with a reason. Silent omission is a
preflight failure. The child must fail before training if it cannot rebuild
train and test containers with the same data, grouping metadata, and distinct
normalization fields as the parent.

This is a study artifact contract only. It must not become a shared dataset
schema and must not change `specs/data_contracts.md`.

## Source Data, Scripts, Figures, Tables, and Files Likely Touched

Source data and prior artifacts:

- `datasets/Run1084_recon3_postPC_shrunk_3.npz`
- `.artifacts/sim_lines_4x_metrics_2026-01-27/gs1_custom/metrics.json`
- `/home/ollie/Documents/ptychopinnpaper2/data/sim_lines_4x_metrics.json`
- `.artifacts/revision_studies/probe_mischaracterization/full_tranched_20260413T003740Z/`
  as diagnostic prior-run evidence only.

Repo files likely touched during implementation:

- `scripts/studies/probe_mischaracterization_stress_test.py`
- `tests/studies/test_probe_mischaracterization_stress_test.py`

Repo files to read/reuse but not edit unless a later plan explicitly approves:

- `ptycho/workflows/grid_lines_workflow.py`
- `ptycho/evaluation.py`
- `scripts/studies/invocation_logging.py`
- `ptycho/loader.py`

Stable core files that must not be edited under this design:

- `ptycho/model.py`
- `ptycho/diffsim.py`
- `ptycho/tf_helper.py`

Paper files likely touched only after publishability gates pass:

- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
- `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- `/home/ollie/Documents/ptychopinnpaper2/data/probe_mischaracterization_metrics.json`
- `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`
- Optional `/home/ollie/Documents/ptychopinnpaper2/tables/probe_mischaracterization_metrics.tex`

## Perturbation and Metric Contract

Keep the first-pass reviewer-facing grid unless the implementation plan
explicitly narrows it before a full run:

- `baseline`
- `phase_curvature_scale`: `0.75`, `0.50`, `0.25`
- `amplitude_blur_sigma_px`: `0.5`, `1.0`, `2.0`
- `phase_noise_sigma_rad`: `0.1*pi` seed `11`, `0.2*pi` seed `17`,
  `0.4*pi` seed `23`

Global amplitude scaling remains excluded from the reviewer-facing grid.

Primary metrics:

- Amplitude SSIM
- Amplitude PSNR

Secondary metrics:

- Amplitude MSE and MAE
- Phase SSIM, PSNR, MSE, and MAE as diagnostic outputs only

Required gates:

- Child-bundle preflight gate: fail before training if any required container
  array, `YY_full`/offset presence marker, distinct normalization field, or
  checksum is missing or ambiguous.
- Infrastructure gate: stop if more than two non-baseline conditions fail for
  the same infrastructure reason after process isolation.
- Baseline gate: compare rerun baseline against Table 2 SSIM/PSNR tolerances or
  explicitly adopt the rerun baseline and avoid old-number comparison.
- Mild-perturbation gate: require the three existing mild conditions and force
  sensitivity language if the metric drops exceed configured thresholds.
- Export gate: never write paper-side metrics unless the stress figure exists
  and the claim gates permit export.

## Dependency, Data-Contract, and Metadata Decisions

Dependencies:

- No new external dependency or solver is approved.
- Continue using TensorFlow/Keras, NumPy, scikit-image, matplotlib, and pytest
  as already used by the existing script.
- Use PATH `python` for commands and persisted invocations. For long runs, use
  tmux and activate `ptycho311` or otherwise ensure PATH resolves to that
  environment.

Data contract:

- Do not change shared NPZ or Ptychodus data contracts.
- Store process-isolation inputs in a run-local study artifact only.
- Preserve the invariant that true measurement arrays are fixed across all
  conditions and that only the reconstruction-side assumed probe changes.
- Preserve the three normalization systems from `docs/DEVELOPER_GUIDE.md`:
  physics/model intensity scaling, container/statistical normalization, and
  stitching/display normalization must remain separately named and audited.
- Never rescale `X_train`, `X_test`, `Y_I_train`, or `Y_I_test` just to satisfy
  a child-process interface. The child must receive the same normalized arrays
  as the parent and apply any model `intensity_scale_model` only at the same
  code boundary as the existing workflow.

Runtime and artifact metadata:

- Record parent and child invocations with command, argv, parsed args, cwd,
  timestamp, PID, return code, and output path.
- Record `python --version`, Python executable, git commit, TensorFlow, NumPy,
  and scikit-image versions in parent and child manifests where practical.
- Do not require child source fingerprints, parent/child Git commit matching, or
  source-fingerprint gates. If the implementation changes while a diagnostic run
  is active, treat that as an operational run-selection concern: rerun from the
  settled implementation before relying on reviewer-facing metrics.
- Record canonical true-probe checksum, condition assumed-probe checksum,
  perturbation type/value/seed, normalization-field checksums, constructor
  field mapping, energy-normalization policy, and data-array checksums for every
  condition.
- Record whether `norm_Y_I_train_container`, `norm_Y_I_test_container`,
  `norm_Y_I_test_stitch`, and `intensity_scale_model` are distinct or aliased
  by value. Aliasing by value is acceptable only when explicitly recorded; a
  shared field name is not acceptable.
- Preserve diagnostic prior-run evidence as diagnostic only; do not include its
  partial metrics in paper-side outputs.

## Pivot Criteria and Stop Conditions

Pivot to text-only response if any of the following occur:

- The process-isolated numeric grid still trips the infrastructure gate.
- A child process cannot rebuild condition containers from canonical true data
  without editing stable core files.
- The child bundle cannot preserve `YY_full`/grouping-offset presence and the
  distinct `norm_Y_I_train_container`, `norm_Y_I_test_container`,
  `norm_Y_I_test_stitch`, and `intensity_scale_model` fields.
- Parent/child preflight cannot prove fixed true-measurement checksums and
  equivalent train/test container constructor inputs before training.
- The full run cannot produce a successful unperturbed baseline.
- The rerun baseline cannot be made Table 2 comparable and the project does not
  explicitly adopt the rerun baseline for this stress result.
- A perturbation axis is a no-op for reported object metrics or changes data
  generation rather than only the assumed probe.
- The run would require a new external dependency, solver, or broad workflow API
  change.

Pivot response requirements:

- Use the probe-consumption smoke result to explain why pure test-time probe
  perturbation is a no-op for the reported object output in the current graph.
- State that a fixed-wrong-probe training/reconstruction numeric grid was
  attempted but stopped on repeated TensorFlow infrastructure failures.
- Strengthen Methods/Discussion fixed-probe limitation language without
  presenting diagnostic sparse metrics as reviewer-facing evidence.

If process isolation succeeds but mild perturbations degrade strongly, do not
pivot to text-only. Report sensitivity, narrow the robustness claim, and decide
whether the figure belongs in main text, appendix, or reviewer response.

## Required Final Assets

If the numeric study succeeds:

- Updated `scripts/studies/probe_mischaracterization_stress_test.py`
- Updated `tests/studies/test_probe_mischaracterization_stress_test.py`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/manifest.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/artifact_manifest.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/invocation.json`
- Per-child invocation/runtime records, either under each condition directory
  or embedded in the parent manifest
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/canonical_condition_inputs.npz`
  or equivalent run-local canonical input bundle
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/canonical_condition_inputs_manifest.json`
  or equivalent bundle manifest with array checksums, constructor mapping,
  `YY_full`/offset presence markers, and distinct normalization-field records
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/probe_consumption_smoke.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/metrics.csv`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/metrics.json`
- `.artifacts/revision_studies/probe_mischaracterization/<run_id>/figures/probe_mischaracterization_stress.png`
- Per-condition `conditions/<condition_id>/metrics.json`
- Per-condition `conditions/<condition_id>/recon/recon.npz` where successful
- Per-condition assumed-probe NPZ and representative probe amp/phase images
- `/home/ollie/Documents/ptychopinnpaper2/data/probe_mischaracterization_metrics.json`
- `/home/ollie/Documents/ptychopinnpaper2/figures/probe_mischaracterization_stress.png`
- Optional `/home/ollie/Documents/ptychopinnpaper2/tables/probe_mischaracterization_metrics.tex`
- Updates to `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
- Updates to `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`
- Updates to `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Reviewer-response text stating that no trainable-probe variant was added

If the study pivots to text-only:

- A run or decision artifact documenting the process-isolation stop condition
  and why numeric reviewer-facing evidence was abandoned.
- Manuscript and reviewer-response text that narrows the fixed-probe claim.
- Changelog and checklist updates reflecting a text-only resolution.
- No paper-side stress-test metrics or figure unless a complete claim-safe run
  produced them.

## Verification Commands and Inspection Checks

Targeted tests after implementation:

The focused study tests must include bundle serialization/reconstruction cases
with deliberately distinct `norm_Y_I_train_container`,
`norm_Y_I_test_container`, `norm_Y_I_test_stitch`, and
`intensity_scale_model` values so a regression cannot collapse them into one
field. They must also cover `YY_full` and grouping-offset preservation or
explicit absence recording.

```bash
pytest tests/studies/test_probe_mischaracterization_stress_test.py -v
pytest tests/test_grid_lines_invocation_logging.py -v
pytest tests/test_grid_lines_workflow.py -k "probe or pipeline or invocation" -v
python -m py_compile scripts/studies/probe_mischaracterization_stress_test.py
```

Dry-run checks:

```bash
python scripts/studies/probe_mischaracterization_stress_test.py \
  --output-root .artifacts/revision_studies/probe_mischaracterization/dry_run_process_isolated_YYYYMMDDTHHMMSSZ \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --dry-run
```

Smoke-only check:

```bash
python scripts/studies/probe_mischaracterization_stress_test.py \
  --output-root .artifacts/revision_studies/probe_mischaracterization/smoke_process_isolated_YYYYMMDDTHHMMSSZ \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --smoke-only
```

Full run pattern:

```bash
tmux new -s probe-mischar-process-isolated
conda activate ptycho311
output_root=.artifacts/revision_studies/probe_mischaracterization/full_process_isolated_YYYYMMDDTHHMMSSZ
test ! -e "$output_root"
python scripts/studies/probe_mischaracterization_stress_test.py \
  --output-root "$output_root" \
  --probe-npz datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --N 64 \
  --gridsize 1 \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nepochs 60 \
  --batch-size 16 \
  --probe-scale-mode pad_preserve \
  --probe-smoothing-sigma 0.5 \
  & pid=$!; wait "$pid"; status=$?; test "$status" -eq 0
```

Required inspection checks:

- `probe_consumption_smoke.json` still records the branch decision and does not
  permit unsupported pure test-time perturbation language.
- `manifest.json` records parent and child runtime metadata, child return codes,
  process-isolation policy, baseline gate, mild gate, infrastructure gate, and
  export policy.
- `canonical_condition_inputs.npz` or equivalent contains all arrays needed to
  rebuild condition containers.
- The bundle manifest contains no ambiguous bare `norm_Y_I` field. It records
  `norm_Y_I_train_container`, `norm_Y_I_test_container`,
  `norm_Y_I_test_stitch`, and `intensity_scale_model` under distinct names with
  checksums and alias metadata.
- The bundle manifest records `YY_full_train`, `YY_full_test`, train/test
  `nn_indices`, `global_offsets`, and `local_offsets` as present with checksums
  or explicitly absent with a source-container reason.
- A child preflight test reconstructs train/test `PtychoDataContainer` objects
  from the bundle and verifies constructor-input checksums before any
  TensorFlow model training starts.
- For every condition, preflight records fixed canonical data checksums and
  assumed-probe checksums.
- `metrics.csv` and `metrics.json` include all requested condition IDs, or the
  manifest records the approved narrowed condition set before launch.
- `figures/probe_mischaracterization_stress.png` exists before any paper export.
- `artifact_manifest.json` is fresh and every listed path exists.
- `git diff -- ptycho/model.py ptycho/diffsim.py ptycho/tf_helper.py` is empty.
- If paper files are changed, compile and inspect the paper PDF before marking
  the reviewer issue complete.
