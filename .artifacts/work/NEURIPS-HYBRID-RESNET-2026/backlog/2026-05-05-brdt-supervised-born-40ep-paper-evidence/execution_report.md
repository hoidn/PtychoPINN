# Execution Report

## Completed In This Pass

- Closed implementation-review High issue 1 (split tracked PID): the meta
  rebuild path no longer overwrites `runtime_provenance.tracked_pid` with the
  rebuild process PID. `_capture_extended_runtime_provenance` accepts a
  `tracked_pid_override` and `launch_timestamp_override`, and
  `_rebuild_meta_only_inner` reads the existing `run_exit_status.json` /
  `runtime_provenance.json` to recover the original training-run tracked PID
  and launch timestamp before rewriting. Rebuild-process metadata is now
  recorded under a separate `meta_rebuild` block in `runtime_provenance.json`
  (`rebuild_pid`, `rebuild_timestamp_utc`, `rebuild_git_sha`,
  `rebuild_git_dirty`, `rebuild_hostname`, `rebuild_platform`,
  `rebuild_gpu_count`, `rebuild_argv`).
- Closed implementation-review High issue 2 (no-overwrite/writer-lock guard
  bypass): `rebuild_meta_only` now acquires the same writer lock as the live
  training path and refuses to start when another writer is targeting the
  same output root. The completed-root refusal is intentionally NOT applied
  to the rebuild path because rebuild-meta exists to re-derive meta artifacts
  from a completed bundle; the writer-lock check still prevents rebuild from
  racing with an active training run.
- Closed implementation-review Medium issue (weak provenance checks):
  - `evidence_surfaces_prepared` now requires the durable summary at
    `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md`,
    `paper_evidence_index.md`, AND `paper_evidence_manifest.json` to all
    reference this backlog item, and the durable summary must additionally
    contain one of the known claim-boundary labels.
  - `same_contract_lineage` is no longer hard-coded `True`. The gate
    re-validates the baseline and FFNO-extension bundles on each gate
    computation and verifies that the current bundle's manifest
    `baseline_lineage` block points at the actual lineage roots, and that
    the dataset id matches across all three manifests.
  - `exit_code_proof` is now stricter: it requires the runtime-provenance
    `tracked_pid` to agree with the run-exit-status `tracked_pid`. A meta
    rebuild that does not preserve the original tracked PID will fail this
    check.
- Regenerated the on-disk meta artifacts at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
  via `--rebuild-meta-only` so the published bundle now carries the
  authoritative tracked PID (`2800210`) consistently in
  `runtime_provenance.json` and `run_exit_status.json`. The recomputed
  gate continues to pass honestly under the strengthened checks
  (`paper_evidence_brdt_additive`, `passed`, all eleven `provenance_checks`
  true).
- Updated the durable summary's Reproducibility & Meta Provenance section to
  describe the strengthened guarantees (writer-lock on rebuild, preserved
  tracked PID, content-level evidence surface check, active lineage
  re-validation).
- Added regression tests:
  - `test_rebuild_meta_only_refreshes_manifest_provenance_and_gate` was
    extended to assert `runtime_provenance.tracked_pid ==
    run_exit_status.tracked_pid`, that a `meta_rebuild` block exists, and
    that `exit_code_proof` is True under the strengthened equality check.
  - `test_rebuild_meta_only_refuses_active_writer` proves the rebuild path
    refuses to start when another live writer holds the lock.
  - `test_evidence_surfaces_consistency_check_requires_all_surfaces` proves
    the strengthened `_check_evidence_surfaces_consistent` helper requires
    all three discoverability surfaces to reference the backlog item and
    requires the durable summary to record a known claim-boundary label.

## Completed Current-Scope Work

- Implementation-review High issues 1 and 2 are addressed in code, on disk,
  and in regression tests.
- Implementation-review Medium issue (weak gate provenance checks) is
  addressed by replacing both checks with content-level validators.
- All approved plan tasks (1–6) remain satisfied. Required deterministic
  checks pass: `python` input-existence command (exit 0),
  `pytest -q tests/studies/test_born_rytov_dt_adapters.py
  tests/studies/test_born_rytov_dt_preflight.py` (111 passed),
  `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
  (clean).

## Follow-Up Work

- The implementation-review's only Follow-Up entry is preserved: regenerate
  the sample-`255` classical/model-based comparator under this backlog
  item's root rather than inheriting it by lineage from the frozen
  `2026-04-29` baseline bundle, if the team wants the paper-facing visual
  bundle to be single-authority rather than baseline-lineage-derived.

## Residual Risks

- Both rerun rows were still materially improving at stop and never reduced
  LR, so the result remains bounded additive evidence rather than a full
  convergence claim. (Carried over from prior pass.)
- The on-disk regeneration was performed on the currently-dirty `fno-stable`
  branch, so `runtime_provenance.json` records `git_dirty=true` (with the
  same flag in the new `meta_rebuild` block). After committing this pass's
  runner/test changes, an additional `--rebuild-meta-only` run on the clean
  tree would flip the rebuild's flag to `false`; the recorded values are
  honest at write time either way.
- The strengthened `evidence_surfaces_prepared` check is content-based
  (substring match) rather than schema-validated. A future change that
  renames the backlog-item identifier in only some of the discoverability
  surfaces would correctly fail the gate; a change that updates them all
  consistently to a different value would pass without flagging the rename.
  This is intentional: the gate validates internal consistency of the
  surfaces tracking this item, not the global correctness of those surfaces.
