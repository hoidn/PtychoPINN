# BRDT Dataset Preflight Execution Report

- Backlog item: `2026-04-29-brdt-dataset-preflight`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/execution_plan.md`
- State: `COMPLETED`
- Tier: `feasibility` (additive candidate work; not manuscript evidence)

## Completed In This Pass

- Hardened the dry-run path in
  `scripts/studies/born_rytov_dt/generate_brdt_dataset.py` so that a
  missing or unreadable operator-validation artifact no longer crashes
  the CLI. `write_dry_run` now accepts an `Optional` operator-authority
  block plus an explicit `blocking_issues` list, and `main()` catches
  `FileNotFoundError`, `ValueError`, `json.JSONDecodeError`, and
  `OSError` while loading the operator authority in dry-run mode and
  records the failure as a `blocking_issue`. The verdict resolves to
  `not_ready` whenever there are geometry mismatches *or* blocking
  issues, and the CLI exits non-zero in that case.
- Surfaced the new field through both machine-readable artifacts:
  `dry_run_summary.json` carries top-level `blocking_issues`, and
  `dry_run_manifest.json` carries `extra.blocking_issues`. Both files
  are emitted on the missing-dependency path so the failure is concrete
  and machine-readable.
- Added a focused regression test
  `test_dry_run_missing_operator_validation_writes_not_ready_artifacts`
  in `tests/studies/test_born_rytov_dt_dataset.py` that runs the CLI
  with an intentionally missing operator-validation path and asserts:
  - exit code 2
  - both dry-run files exist
  - `verdict == "not_ready"` in summary and manifest
  - `operator_authority_block is None`
  - non-empty `blocking_issues` containing the recognizable
    "operator validation artifact not found" message
- Refreshed the durable dry-run artifacts under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-dataset-preflight/`
  so the checked-in `dry_run_manifest.json` and `dry_run_summary.json`
  match the new schema (the `blocking_issues: []` field is now present
  on the success path).
- Updated `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_dataset_preflight.md`
  to describe the missing-dependency dry-run behavior so the durable
  summary stays in sync with the schema.

## Completed Plan Tasks

- Tranche 1 (Lock The Dataset Contract): unchanged, remains satisfied
  by the existing `dataset_contract` surface, manifest helpers, locked
  smoke geometry, deterministic split policy, locked phantom roster,
  and the contract-level test suite.
- Tranche 2 (Implement The Dry-Run And Smoke Dataset Generator): the
  generator continues to support the geometry-only dry-run mode, the
  live smoke-generation mode, and the required HDF5 / manifest /
  provenance outputs. The dry-run mode now also satisfies the plan's
  "any missing dependency/storage/path issue" requirement on the
  failure path, not just the success path.
- Tranche 3 (Durable Summary, Discoverability, And Final Gates): the
  durable preflight summary, evidence-matrix entries, and docs-index
  pointers remain in place; the summary now also documents the
  not-ready / blocking-issues dry-run behavior.

## Remaining Required Plan Tasks

None within this item's scope. Follow-up items remain explicitly out
of scope:

- `2026-04-29-brdt-task-adapters`: add Lightning / dataloader / loss
  surfaces that consume the locked smoke-dataset contract.
- `2026-04-29-brdt-four-row-preflight`: run the bounded
  classical / U-Net / FNO / SRU-or-Hybrid candidate-lane preflight on
  top of this dataset authority.

## Verification

- Blocking pytest:
  `pytest -q tests/studies/test_born_rytov_dt_dataset.py`
  → 23 passed in 23.83s.
- Blocking compileall:
  `python -m compileall -q scripts/studies/born_rytov_dt`
  → success (no output, exit 0).
- Dry-run refresh (success path):
  `python -m scripts.studies.born_rytov_dt.generate_brdt_dataset --dry-run-manifest`
  → wrote `dry_run_summary.json` and `dry_run_manifest.json` with
  `verdict: ready_for_smoke_generation`, zero geometry mismatches, and
  `blocking_issues: []` in both files.
- Failure-path coverage:
  `pytest -q tests/studies/test_born_rytov_dt_dataset.py
  -k test_dry_run_missing_operator_validation_writes_not_ready_artifacts`
  → exercised end-to-end via subprocess; passes.

## Residual Risks

- ODTbrain inverse-side recovery is still not exercised locally
  (`brdt_operator_validation_report.md` remains
  `pass_with_documented_limits`); this stays a documented BRDT-lane
  limit but does not block dataset preflight.
- The smoke dataset is intentionally tiny (16 train / 4 val / 4 test).
  It is suitable for feasibility and adapter bring-up only; anything
  benchmark-like must regenerate the later larger decision-support
  split under a separately authorized item.
- Renaming any of the core stored HDF5 fields (`q_true_physical`,
  `q_true_norm`, `sinogram_real`, `sinogram_imag`, `angle_mask`,
  `sample_seed`, `phantom_family`) or the new `blocking_issues` /
  `verdict` keys would still be a contract break and would require
  explicit follow-up approval.
