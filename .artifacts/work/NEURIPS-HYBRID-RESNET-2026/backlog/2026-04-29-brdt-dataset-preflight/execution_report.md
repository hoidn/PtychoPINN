# BRDT Dataset Preflight Execution Report

- Backlog item: `2026-04-29-brdt-dataset-preflight`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/execution_plan.md`
- State: `COMPLETED`
- Tier: `feasibility` (additive candidate work; not manuscript evidence)

## Completed In This Pass

Addresses the two `High` items from the implementation review by
fixing the weak-scattering contract enforcement and wiring the
mandatory invocation provenance. The blocking checks contract is
unchanged.

- Enforced the `|delta_n| <= DELTA_N_MAX = 0.03` weak-scattering cap in
  `scripts/studies/born_rytov_dt/phantoms.py`. A new
  `_clip_to_weak_scattering(...)` helper clips the deviation from
  `n_m` after each family accumulates its perturbations, so
  overlapping `overlapping_ellipses`, `soft_blobs`, and
  `sparse_inclusions` patches can no longer drift past the candidate-
  lane envelope. Measured on the regenerated smoke dataset, the
  maximum `|delta_n|` per split is now:
  - train: `~0.0300`
  - val:   `~0.0294`
  - test:  `~0.0300`
  versus the previous unclamped run that reached `~0.053` on train and
  `~0.046` on test.
- Tightened `tests/studies/test_born_rytov_dt_dataset.py::
  test_phantom_generators_in_weak_scattering_envelope` to assert
  `|delta_n| <= DELTA_N_MAX + 1e-12` instead of the looser
  `<= 8 * DELTA_N_MAX` bound that allowed contract drift to pass.
  Added `test_phantom_envelope_holds_across_seeds_at_locked_grid`
  which sweeps 32 seeds at the production grid for each phantom
  family so the cap cannot be incidentally satisfied for a single
  seed.
- Wired `scripts/studies/invocation_logging.write_invocation_artifacts`
  into the BRDT generator CLI so that both `--dry-run-manifest` and
  live runs write canonical `invocation.json` / `invocation.sh` at the
  output root before launching expensive work, with `script`,
  `argv`, `command`, `parsed_args`, `cwd`, `timestamp_utc`, `pid`, plus
  `extra` carrying `git_sha`, `git_dirty`, `runtime_provenance`, the
  `mode` flag (`dry_run_manifest` vs `live`), and the
  `backlog_item` identifier.
- Added `test_dry_run_writes_invocation_provenance_artifacts` and
  `test_live_writes_invocation_provenance_artifacts` to lock the
  invocation contract against future drift.
- Regenerated the durable smoke artifacts under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/`
  so the checked-in dataset, manifest, dry-run files, and provenance
  artifacts all match the corrected contract.
- Updated `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
  to call out the explicit `|delta_n|` clip, the corresponding measured
  envelope on the regenerated dataset, and the mandatory
  `invocation.json` / `invocation.sh` provenance artifacts written by
  both dry-run and live modes.

## Completed Current-Scope Work

- Tranche 1 (Lock The Dataset Contract): unchanged. Manifest helpers,
  locked smoke geometry, deterministic split policy, locked phantom
  roster, normalize/unnormalize, geometry validation, and
  reject-normalized-q guard remain intact.
- Tranche 2 (Implement The Dry-Run And Smoke Dataset Generator): the
  generator continues to support geometry-only dry-run and live smoke
  modes, now also enforces the weak-scattering envelope at the
  phantom level and emits canonical invocation provenance artifacts at
  the output root in both modes. The dry-run not-ready/blocking-issues
  failure path remains in place.
- Tranche 3 (Durable Summary, Discoverability, And Final Gates): the
  durable preflight summary now documents the corrected weak-scattering
  cap (with measured `|delta_n|` per split) and the new invocation-
  provenance contract; `docs/index.md`, `docs/studies/index.md`, the
  evidence matrix, and the paper evidence index continue to discover
  this summary.

## Follow-Up Work

- Source the live generator's operator settings directly from
  `operator_validation.json` instead of re-declaring them through
  `dataset_contract`'s locked constants plus a mismatch check
  (review-tier follow-up; reduces drift risk against the operator
  authority, but the current implementation is correct for the
  delivered contract).

## Residual Risks

- ODTbrain inverse-side recovery is still not exercised locally;
  `brdt_operator_validation_report.md` remains
  `pass_with_documented_limits`. This stays a documented BRDT-lane
  limit but does not block dataset preflight.
- The smoke dataset is intentionally tiny (`16 train / 4 val / 4 test`).
  It is suitable for feasibility and adapter bring-up only; anything
  benchmark-like must regenerate the later larger decision-support
  split under a separately authorized item.
- The new `_clip_to_weak_scattering` step changes pixel values for
  generators that previously exceeded `DELTA_N_MAX`. Any consumer that
  cached HDF5 hashes from the prior delivery must regenerate the
  dataset; this is the intended consequence of bringing the artifacts
  back inside the candidate-lane contract.
- Renaming any of the core stored HDF5 fields (`q_true_physical`,
  `q_true_norm`, `sinogram_real`, `sinogram_imag`, `angle_mask`,
  `sample_seed`, `phantom_family`), the `blocking_issues` / `verdict`
  keys, or the `invocation.{json,sh}` provenance artifacts would still
  be a contract break and would require explicit follow-up approval.

## Verification

- Blocking pytest:
  `pytest -q tests/studies/test_born_rytov_dt_dataset.py`
  → `28 passed in 35.54s` (was 23 before this pass; +2 invocation
  tests, +1 phantom-envelope sweep, +2 from re-running the dry-run
  not-ready/regression coverage).
- Blocking compileall:
  `python -m compileall -q scripts/studies/born_rytov_dt`
  → success (no output, exit 0).
- Live smoke regeneration:
  `python -m scripts.studies.born_rytov_dt.generate_brdt_dataset`
  → wrote refreshed
  `dataset_manifest.json`, dataset HDF5 files, and provenance
  artifacts with measured per-split SNR `train=18.22 dB`,
  `val=16.37 dB`, `test=19.26 dB`.
- Dataset envelope spot-check:
  `max |delta_n|` per split now `~0.0300 / ~0.0294 / ~0.0300`
  (train / val / test), within the
  `DELTA_N_MAX = 0.03` candidate-lane cap.
- Provenance spot-check: the artifact root now contains
  `invocation.json` and `invocation.sh` next to the dataset and
  manifest files, matching the
  `docs/development/INVOCATION_LOGGING_GUIDE.md` contract.
