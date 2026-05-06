# Execution Report

## Completed In This Pass

- Closed implementation-review High issue 1 (`exit_code_proof` is not a proof
  of completion). `_build_provenance_checks()` now requires
  `run_exit_status.json` and `runtime_provenance.json` to agree on
  `tracked_pid` AND requires `exit_code == 0` AND `status == "completed"`. The
  live training path in `_run_paper_evidence_inner` now defers writing
  `run_exit_status.json` until after training, evaluation, metrics, and visual
  materialization have all succeeded, and wraps the subsequent gate /
  provenance-check / manifest-reseed work in a `try/except` that overwrites
  the exit-status record with `exit_code=1`/`status="failed"` if any of those
  steps raise. A partial run can no longer leave a stale
  `"completed"`/`0` exit-status artifact behind.
- Closed implementation-review High issue 2 (`scheduler_matches_contract`
  reduced to a name check). Added `_scheduler_matches_contract` and routed
  both the live and `--rebuild-meta-only` gate-row constructions through it.
  The new helper checks scheduler name AND `plateau_factor`,
  `plateau_patience`, `plateau_threshold`, `plateau_min_lr` against the
  contract values, and additionally validates the surrounding optimizer
  recipe (`runtime.epochs`, `runtime.batch_size`, `runtime.learning_rate`).
  A bundle with drifted plateau settings or a drifted optimizer recipe now
  fails the gate even when the scheduler name matches.
- Closed implementation-review High issue 3 (gate provenance/consistency
  checks materially weaker than the approved contract):
  - `sample_255_visual_bundle` now also calls
    `_check_sample_visual_source_arrays`, which requires every per-sample,
    per-row source-array file to exist on disk for the configured
    `required_paper_sample` (`q_target`, `sino_obs`, classical comparator
    `q_pred`/`sino_pred`, Hybrid ResNet `q_pred`/`sino_pred`, FFNO
    `q_pred`/`sino_pred`). The check threads the sample id through
    `visual_status` and falls back to `preflight_manifest.required_paper_sample`.
  - `same_contract_lineage` now additionally compares the current bundle's
    `dataset.split_counts`, `dataset.normalization`, `operator.geometry`,
    `fixed_sample_ids`, per-row `input_mode`, and the training-contract
    fields `batch_size`, `learning_rate`, `optimizer`, `seed`, and
    `loss_weights` against the frozen baseline lineage. Drift on any of
    these locked invariants now fails the gate. The lineage check still
    re-validates the baseline and FFNO-extension bundles on each call and
    re-checks the current bundle's `baseline_lineage` pointers and shared
    dataset id.
  - `_check_evidence_surfaces_consistent` now also requires the repo-wide
    `docs/index.md` to reference this backlog item, in addition to the
    durable summary, paper-evidence index, and paper-evidence manifest.
- Regenerated the on-disk meta artifacts at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
  via `--rebuild-meta-only` so the published bundle now carries the
  strengthened provenance/consistency checks. Recomputed gate continues to
  pass honestly: `paper_evidence_brdt_additive` / `passed`, all eleven
  `provenance_checks` true, `failed_gate_checks=[]`. The rebuilt
  `runtime_provenance.json` and `run_exit_status.json` agree on
  `tracked_pid=2800210`, `exit_code=0`, `status="completed"`.
- Updated the durable summary's Reproducibility & Meta Provenance section to
  document the strengthened guarantees: stricter `exit_code_proof`, deferred
  live-path exit-status write, full scheduler-and-optimizer contract check,
  full source-array bundle check, full same-contract lineage invariants
  (split counts / fixed samples / operator geometry / normalization / input
  mode / training-contract fields), and the new `docs/index.md` requirement
  in `evidence_surfaces_prepared`.
- Added regression tests covering the strengthened checks:
  - `test_evidence_surfaces_consistency_check_requires_all_surfaces` was
    extended to require `docs/index.md` in addition to the existing three
    surfaces.
  - `test_scheduler_matches_contract_rejects_plateau_drift` proves that
    drifting `plateau_factor`, `plateau_patience`, `plateau_min_lr`, or
    `runtime.batch_size` fails the helper even when the scheduler name
    matches.
  - `test_sample_visual_source_arrays_check_requires_each_required_file`
    proves the helper fails when any one of the eight required per-sample,
    per-row source-array files is missing.
  - `test_exit_code_proof_requires_completed_status_and_zero_exit_code`
    proves that a `failed` status, a non-zero `exit_code`, and the
    happy-path triple all produce the correct `provenance_checks`
    `exit_code_proof` value.
  - `test_same_contract_lineage_check_detects_split_count_drift` proves
    drift on split counts, fixed-sample roster, operator geometry, or
    training-contract loss weights fails the lineage check.

## Completed Current-Scope Work

- All three implementation-review High issues (`exit_code_proof`,
  `scheduler_matches_contract`, gate provenance/consistency checks) are
  addressed in code, on disk, and in regression tests.
- All approved plan tasks (1–6) remain satisfied, with the strengthened gate
  checks now matching Tasks 4 and 6 verbatim. Required deterministic checks
  pass on the current branch:
  - `python` input-existence command (exit 0)
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py tests/studies/test_born_rytov_dt_preflight.py`
    (115 passed in 321.23s; up from 111 due to the four new regression
    tests added in this pass)
  - `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
    (clean)

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
  branch, so `runtime_provenance.json` records `git_dirty=true` for both
  the original training run and the new rebuild block. After committing
  this pass's runner/test/report changes, an additional
  `--rebuild-meta-only` run on the clean tree would flip the rebuild's flag
  to `false`; the recorded values are honest at write time either way.
- The strengthened `evidence_surfaces_prepared` check is content-based
  (substring match of the backlog item id and a known claim-boundary
  label). A future change that renames the backlog-item identifier in only
  some of the four discoverability surfaces (durable summary,
  paper-evidence index, paper-evidence manifest, `docs/index.md`) would
  correctly fail the gate; a change that updates them all consistently to a
  different value would pass without flagging the rename. This is
  intentional: the gate validates internal consistency of the surfaces
  tracking this item, not the global correctness of those surfaces.
- The strengthened `same_contract_lineage` check is invariant-based: it
  compares structurally serialized fields between the current and baseline
  manifests. New fields added in only one manifest revision will not be
  flagged automatically; only fields enumerated in `_LINEAGE_TRAINING_FIELDS`
  and the dataset/operator/fixed-sample/normalization/input-mode blocks are
  cross-checked.
