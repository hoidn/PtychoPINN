# BRDT Dataset Preflight Execution Report

- Backlog item: `2026-04-29-brdt-dataset-preflight`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/execution_plan.md`
- State: `COMPLETED`
- Tier: `feasibility` (additive candidate work; not manuscript evidence)

## Completed In This Pass

- Fixed the remaining review-blocking dry-run contract defect in
  `scripts/studies/born_rytov_dt/generate_brdt_dataset.py` by threading
  the requested CLI `--noise-sigma` value through `write_dry_run()`.
  Dry-run mode now records the requested noise configuration in both
  `dry_run_manifest.json` (`noise.noise_sigma_physical_units`) and
  `dry_run_summary.json` (`noise_sigma_physical_units`) instead of
  hardcoding `0.0` or omitting the field.
- Strengthened the dry-run regression in
  `tests/studies/test_born_rytov_dt_dataset.py` to assert the requested
  non-default `--noise-sigma 0.002` value in both machine-readable
  outputs. The test failed before the generator patch and passes after
  it.
- Refreshed the checked-in dry-run artifacts under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/`
  so the durable evidence matches the fixed contract:
  - `dry_run_manifest.json`
  - `dry_run_summary.json`
- Updated the durable BRDT preflight summary to state that dry-run
  outputs preserve the requested noise configuration alongside the exact
  generation command.

## Completed Current-Scope Work

- The BRDT smoke-dataset contract remains locked around physical
  `q(x,z)=k_m^2((n/n_m)^2-1)`, train-only normalization,
  `forward_input_is_physical_q: true`, `model_output_space:
  normalized_q`, deterministic disjoint object seeds, and feasibility-
  only claim boundaries.
- The generator now satisfies the full current plan contract for the
  dry-run path: geometry validation, exact command provenance, requested
  noise-configuration capture, dry-run summary plus manifest skeleton,
  deterministic live generation, and live HDF5/manifest emission under
  the approved artifact root.
- The approved backlog verification contract still passes, and the new
  regression coverage directly covers the review findings that caused
  `REVISE`.

## Follow-Up Work

- `2026-04-29-brdt-task-adapters`: add Lightning/data-loading/loss
  surfaces that consume the locked smoke-dataset contract without
  renaming the stored HDF5 fields.
- `2026-04-29-brdt-four-row-preflight`: run the bounded classical/U-Net/
  FNO/SRU-or-Hybrid candidate-lane preflight on top of this dataset
  authority.

## Residual Risks

- ODTbrain inverse-side recovery is still not exercised locally
  (`brdt_operator_validation_report.md` remains
  `pass_with_documented_limits`); this stays a documented BRDT-lane
  limit but does not block dataset preflight.
- The smoke dataset is intentionally tiny (24 samples). It is suitable
  for feasibility and adapter bring-up only; anything benchmark-like
  must regenerate the later larger decision-support split under a
  separately authorized item.
- The HDF5 schema is now stable enough for downstream consumers to rely
  on, including the new dry-run skeleton, but renaming core stored
  fields (`q_true_physical`, `q_true_norm`, `sinogram_real`,
  `sinogram_imag`, `angle_mask`, `sample_seed`, `phantom_family`) would
  still be a contract break requiring explicit follow-up approval.

## Verification

- Blocking pytest:
  `pytest -q tests/studies/test_born_rytov_dt_dataset.py`
  → passes.
- Blocking compileall:
  `python -m compileall -q scripts/studies/born_rytov_dt`
  → success (no output, exit `0`).
- Focused red/green regression:
  `pytest -q tests/studies/test_born_rytov_dt_dataset.py -k dry_run_writes_manifest_skeleton_and_exact_command`
  → failed before the generator patch with `KeyError:
  'noise_sigma_physical_units'`, then passed after the patch.
- Dry-run refresh:
  `python -m scripts.studies.born_rytov_dt.generate_brdt_dataset --dry-run-manifest`
  → wrote `dry_run_summary.json` and `dry_run_manifest.json` with
  `verdict: ready_for_smoke_generation`, zero geometry mismatches, and
  `noise_sigma_physical_units: 0.001` in both outputs.
