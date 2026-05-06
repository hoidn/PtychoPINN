# Execution Report

## Completed In This Pass

- Closed the implementation review's blocking gap by hardening the BRDT
  40-epoch paper-evidence runner so its checked-in artifacts can be honestly
  regenerated from the approved code path:
  - `run_paper_evidence` now re-seeds `preflight_manifest.json` with the gate's
    final `claim_boundary`, `promotion_status`, and `paper_evidence_gate_path`
    after training, so the manifest cannot advertise an additive promotion
    that the gate did not actually grant.
  - The runner refuses to start a duplicate live writer against the same
    `--output-root` (writer-lock file plus a `paper_evidence_gate.json` +
    `run_exit_status.json` completion check) and exposes
    `--force-overwrite` only as an explicit override.
  - `runtime_provenance.json` now carries the required additive provenance
    payload (`git_sha`, `git_dirty`, `hostname`, `platform`, `gpu_count`,
    `launch_timestamp_utc`, `tracked_pid`, optional `log_path`).
  - `run_exit_status.json` retains the tracked PID plus the run-log path.
  - The promotion gate's `provenance_checks` now validates `git_provenance`,
    `host_provenance`, `model_profiles`, `run_log_present`, and
    `evidence_surfaces_prepared` (the durable summary file existing and
    referencing this backlog item) in addition to the prior keys.
  - Added a `--rebuild-meta-only` mode that recomputes the manifest,
    provenance, audit, gate, and visual bundle from the existing per-row
    outputs without retraining and without releasing the no-overwrite guard.
- Regenerated the meta artifacts of
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
  via `--rebuild-meta-only` so the on-disk bundle now matches what the
  approved runner produces. The recomputed gate passes honestly with all
  provenance checks satisfied (`paper_evidence_brdt_additive`, `passed`).
- Extended the durable summary with a Reproducibility & Meta Provenance
  section that names the writer guard, manifest re-seed, and the
  `--rebuild-meta-only` regeneration path.
- Added regression tests covering the manifest re-seed, provenance keys,
  writer-lock refusal, completed-root refusal, and `rebuild_meta_only`
  refresh path (`tests/studies/test_born_rytov_dt_preflight.py`).

## Completed Current-Scope Work

- Implementation-review blocking issues 1, 2, and 3 are addressed: the
  runner now produces the artifacts that the review found missing, and the
  on-disk bundle has been refreshed to match.
- All approved plan tasks (1–6) remain satisfied; no plan deviation was
  required.

## Follow-Up Work

- Option to regenerate the sample-`255` classical comparator under this
  backlog item directly (rather than inheriting it by lineage) so the entire
  paper-facing visual bundle is freshly reproduced under one authority root.
  Recorded as the implementation review's only Follow-Up entry.

## Residual Risks

- Both rerun rows were still materially improving at stop and never reduced
  LR, so the result remains bounded additive evidence rather than a full
  convergence claim.
- The sample-`255` classical comparator is accepted from the frozen baseline
  lineage rather than regenerated in this pass.
- The `--rebuild-meta-only` regeneration was performed on the
  currently-dirty `fno-stable` branch, so `runtime_provenance.json` records
  `git_dirty=true`. After committing the runner/test changes, a follow-up
  `--rebuild-meta-only` pass on the clean tree would flip this to `false`;
  the recorded value is honest at write time either way.
