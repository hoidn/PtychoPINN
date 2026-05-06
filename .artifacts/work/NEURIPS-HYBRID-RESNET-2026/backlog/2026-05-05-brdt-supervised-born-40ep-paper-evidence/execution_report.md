# Execution Report

## Completed In This Pass

- Closed the implementation-review High issue: the on-disk
  `runtime_provenance.json` previously claimed top-level fields
  (`git_sha`, `git_dirty`, `hostname`, `platform`, `gpu_count`,
  `launch_timestamp_utc`) that came from the rebuild host's snapshot
  rather than the original 40-epoch training run. Even the prior pass's
  acknowledged-honest fields (`tracked_pid`, `launch_timestamp_utc`)
  disagreed with `invocation.json` (which records
  `timestamp_utc=2026-05-06T03:01:07.146943+00:00` for the same PID), so
  the bundle's runtime provenance was uniformly tainted and the
  `paper_evidence_brdt_additive` promotion was not trustworthy.
  Rerunning the training was out of scope for a review-driven correction
  pass, so this pass took the only honest reconstruction path: rebuild
  `runtime_provenance.json` from the preserved `invocation.json` (which
  was written by the original training process at startup) and demote
  the bundle when the unrecoverable fields cannot be honestly restored.
- Added `reconstruct_runtime_provenance_from_invocation(output_root)` in
  `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`. The
  helper reads the on-disk `invocation.json` and writes
  `runtime_provenance.json` with:
  - recovered fields (exactly as captured by the original training
    process at startup): `tracked_pid`, `pid`,
    `launch_timestamp_utc`, `python_executable`, `python_version`,
    `torch.version`/`cuda_version`/`cuda_available`/`device_name`,
    `cwd`, `pythonpath`, `ptycho_torch_file`;
  - unrecoverable fields (never preserved by `invocation.json`,
    therefore set to `null` rather than fabricated from the rebuild
    host): `git_sha`, `git_dirty`, `hostname`, `platform`,
    `gpu_count`;
  - a `provenance_reconstruction` block recording the source
    (`invocation.json`), source path, reconstruction timestamp,
    `recovered_fields` list, `unrecoverable_fields` list, and the
    rationale tying the loss back to the prior rebuild-host overwrite
    and the gate's expected `git_provenance` / `host_provenance`
    failures.
- Added a CLI flag `--reconstruct-runtime-provenance-from-invocation`
  that runs the reconstruction and then immediately performs the
  meta-only rebuild so the dependent artifacts (`preflight_manifest.json`,
  `paper_evidence_gate.json`, `metrics.json`/`combined_metrics.json`/
  `metric_schema.json`, convergence audit, visuals) propagate the demoted
  claim boundary.
- Relaxed `_amend_existing_runtime_provenance_for_rebuild` to recognize
  the new `provenance_reconstruction` block. When the block is present
  the amend path only requires the reconstruction's declared
  `recovered_fields` to be non-null, so the unrecoverable fields can
  pass through as `null` without the rebuild path either fabricating
  values or refusing to attach the `meta_rebuild` block. The existing
  strict-required-fields refusal still applies to bundles without a
  reconstruction block, preserving the prior pass's regression coverage.
- Applied the reconstruction + rebuild on the on-disk artifact bundle:
  - `runtime_provenance.json` now records the original training-run
    `pid=tracked_pid=2800210`, `launch_timestamp_utc=2026-05-06T03:01:07.146943+00:00`
    (matching `invocation.json`), `python_executable=/home/ollie/miniconda3/envs/ptycho311/bin/python3.11`,
    `python_version=3.11.13`, and the original
    `torch.version=2.9.1+cu128` / `cuda_version=12.8` /
    `cuda_available=true` / `device_name=NVIDIA GeForce RTX 3090`
    block; `git_sha`, `git_dirty`, `hostname`, `platform`, and
    `gpu_count` are explicitly `null` with the
    `provenance_reconstruction` block explaining why; the rebuild's
    pid/timestamp/git/host/argv now appear under a separate
    `meta_rebuild` block.
  - `paper_evidence_gate.json` records `claim_boundary=
    decision_support_convergence_followup`,
    `promotion_status=failed`,
    `failed_gate_checks=["git_provenance","host_provenance"]`. All
    other provenance checks (`runtime_provenance`,
    `python_provenance`, `torch_provenance`, `dataset_identity`,
    `split_manifest`, `model_profiles`, `run_log_present`,
    `sample_255_visual_bundle`, `exit_code_proof`,
    `same_contract_lineage`, `evidence_surfaces_prepared`) pass.
  - `metrics.json`, `combined_metrics.json`, `preflight_manifest.json`,
    and `metric_schema.json` now advertise the demoted claim boundary
    (`decision_support_convergence_followup`).
- Updated every checked-in evidence surface to reflect the demoted
  outcome and to satisfy the gate's `evidence_surfaces_prepared`
  cross-surface consistency check at the new boundary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md`:
    final claim boundary, promotion status, lane status, gate-result
    section, residual-risk section, and reproducibility section all
    record the demotion, the gate's failed checks, and the
    reconstruction path; a new front-matter warning links the
    reconstruction to the gate failure.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_manifest.json`:
    the `brdt:additive_40ep_bundle` registry entry now records
    `claim_boundary=decision_support_convergence_followup`,
    `row_status=decision_support`, `evidence_tier=decision_support`,
    `draftability=draftable_context_only`,
    `provenance_gaps=["git_provenance_unrecoverable_after_prior_rebuild_overwrote_runtime_provenance",
    "host_provenance_unrecoverable_after_prior_rebuild_overwrote_runtime_provenance"]`,
    `notes.promotion_status=failed`, and an explicit
    `notes.promotion_failure_reason` describing the lost original
    runtime provenance and the reconstruction path. The
    `claim_boundary_registry`, `manuscript_draftability`, and
    `blocked_claims` sections were updated in lockstep, including a
    new `brdt_paper_evidence_additive_promotion` blocked-claim entry
    explaining why an additive paper-evidence promotion of this bundle
    is currently blocked.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`:
    backlog row tier dropped from `bounded_capped` to
    `decision_support`; outcome column documents the failed gate, the
    failed-check list, and the path to a future paper-evidence
    promotion via retraining on a clean repo.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_package_design.md`:
    the prior `### BRDT Additive Amendment (2026-05-05)` section was
    removed (per the plan, the amendment is authorized only when the
    gate passes) and replaced with `### BRDT 40-Epoch Promotion
    Attempt — Failed (2026-05-05/06)`, which records the gate's
    `failed_gate_checks` value, the loss of original
    runtime provenance, the reconstruction path, and the constraint
    that an additive promotion requires retraining on a clean repo.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`: the
    BRDT row in the headline evidence list, the manuscript
    incorporation map, the `BRDT Decision-Support Bundle` section
    (renamed from `BRDT Additive Assets`), and the completed-backlog
    table were updated in lockstep with the demotion.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`:
    family `brdt_supervised_born_40ep_additive_promotion` records
    `claim_boundary=decision_support_convergence_followup`, removes
    the `promotion surface from decision-support lineage to bounded
    additive paper evidence` changed factor, and rewrites the
    interpretation to document the gate failure and the reconstruction
    path.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`:
    the `brdt_paper_evidence_2048cap_40ep` dataset contract and both
    `brdt__hybrid_resnet__...__paper_evidence_additive_40ep` /
    `brdt__ffno__...__paper_evidence_additive_40ep` row variants now
    advertise `claim_boundary=decision_support_convergence_followup`
    and `evidence_status=decision_support`.
  - `docs/index.md`: the BRDT 40-epoch summary entry's description and
    keywords now record the failed promotion and the reconstruction
    path so the demoted outcome is discoverable from the repo-wide
    documentation hub.
- Added regression coverage:
  - `test_reconstruct_runtime_provenance_from_invocation_restores_only_preserved_fields`
    seeds an `invocation.json` whose `extra.runtime_provenance` carries
    sentinel python/torch values plus a sentinel pid/timestamp_utc,
    runs the reconstruction, and asserts the recovered fields match
    invocation.json exactly while every unrecoverable field is `null`
    and the `provenance_reconstruction` block lists both groups with
    the rationale.
  - `test_reconstruct_runtime_provenance_then_rebuild_meta_only_demotes_gate`
    runs the full live training entrypoint, then reconstruction, then
    rebuild-meta-only, and asserts the resulting
    `paper_evidence_gate.json` records
    `claim_boundary=decision_support_convergence_followup`,
    `promotion_status=failed`, `failed_gate_checks` containing both
    `git_provenance` and `host_provenance`, while
    `python_provenance`/`torch_provenance` still pass.
  - `test_reconstruct_runtime_provenance_refuses_when_invocation_missing`
    proves the reconstruction path raises `FileNotFoundError` for a
    missing `invocation.json`, `RuntimeError` for an unparseable
    payload, and `RuntimeError` for a parseable payload that is missing
    `pid`/`timestamp_utc`.

## Completed Current-Scope Work

- Implementation-review High issue 1 is now closed honestly. The on-disk
  `runtime_provenance.json` no longer carries any value sourced from the
  rebuild host's snapshot at the top level: every recorded field either
  matches the original training process's `invocation.json` exactly or
  is `null` with an explicit `provenance_reconstruction` rationale.
- The `paper_evidence_gate.json` no longer passes on top of the
  compromised provenance surface. The bundle is now consistently
  represented as same-contract decision-support evidence
  (`claim_boundary=decision_support_convergence_followup`,
  `promotion_status=failed`) across the structured manifest, the
  durable summary, the package-design doc, the index/matrix surfaces,
  the model/ablation indexes, and `docs/index.md`.
- The plan's required deterministic checks pass on the current branch:
  - Input-existence command (exit 0).
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py
    tests/studies/test_born_rytov_dt_preflight.py` — 125 tests pass
    (122 previously passing + 3 new reconstruction tests).
  - `python -m compileall -q scripts/studies/born_rytov_dt
    ptycho_torch` (clean).
- All twelve top-level JSONs in the artifact bundle parse cleanly under
  `python -m json.tool`.

## Follow-Up Work

- **Promotion to `paper_evidence_brdt_additive` requires retraining.**
  Because the original training-run `git_sha`/`git_dirty`/
  `hostname`/`platform`/`gpu_count` were lost and cannot be honestly
  recovered, the only path to additive paper-evidence promotion of this
  lane is a fresh training run on a clean repo so the runner records
  full runtime provenance at training time. The runner's preserved
  preservation guard, the new reconstruction helper, and the relaxed
  amend path together make this safe: the live training path captures
  full provenance, the rebuild path preserves it exactly, and a future
  reconstruction-only path will only ever demote a bundle, never
  fabricate values.
- Carried over from prior pass: regenerate the sample-`255`
  classical/model-based comparator under this backlog item's own
  artifact root rather than inheriting it by lineage from the frozen
  `2026-04-29` baseline bundle, if the team wants the visual bundle to
  have single-root authority. This is independent of the runtime
  provenance loss and remains a follow-up rather than a current-scope
  blocker.

## Residual Risks

- **Original training-run git/host provenance is permanently lost.**
  The prior rebuild overwrote those fields before the runner's
  preservation guard was added. The reconstructed
  `runtime_provenance.json` faithfully marks them `null` with a
  `provenance_reconstruction` block, and the gate fails honestly on
  `git_provenance` and `host_provenance`. No fabrication has been
  introduced. The trade-off is that this bundle remains decision-support
  context only until a future retraining pass; that is the correct
  outcome of an honest provenance contract, not a remaining defect.
- The `--reconstruct-runtime-provenance-from-invocation` path implicitly
  trusts that `invocation.json` itself was not tampered with after
  training. The runner's prior contract is that `invocation.json` is
  written once by the original training process and is not rewritten
  by `--rebuild-meta-only`. If a future code change started rewriting
  `invocation.json` during rebuild, the reconstruction's "authoritative"
  source would silently drift. This is mitigated by the existing
  `_acquire_writer_lock` discipline (only one writer at a time) and by
  the fact that the rebuild path does not currently touch
  `invocation.json` — a regression here would require a deliberate code
  change rather than a process race.
- The relaxed `_amend_existing_runtime_provenance_for_rebuild` path
  trusts the reconstruction's `recovered_fields` list. A
  `provenance_reconstruction` block whose `recovered_fields` declared
  unrecoverable fields as recovered would let the rebuild bless null
  values silently. Mitigations: the helper that writes the
  reconstruction sets `recovered_fields` from a constant
  (`RECONSTRUCTABLE_RUNTIME_PROVENANCE_FIELDS`); the
  `test_reconstruct_runtime_provenance_then_rebuild_meta_only_demotes_gate`
  test asserts the gate's `git_provenance`/`host_provenance` checks
  still fail end-to-end on the reconstructed payload; and the helper
  raises rather than backfills if `invocation.json` does not record
  `pid` and `timestamp_utc`.
- The rows were still materially improving at stop and never reduced
  LR, so the underlying training result remains bounded same-contract
  decision-support evidence rather than a full-convergence claim. This
  predates this pass and is preserved in the durable summary.
- The strengthened `evidence_surfaces_prepared` check uses the
  structured `paper_evidence_manifest.json` entry as the authoritative
  source for the bundle's `claim_boundary` and verifies the durable
  summary, paper-evidence index, repo-wide `docs/index.md`, and (when
  the manifest reports the promoted boundary) the package-design doc
  all reference the same boundary, backlog item id, and canonical
  artifact root. The current state passes this check at the demoted
  boundary because every surface was updated in lockstep. A future
  partial update that drifts only some surfaces will continue to fail
  the check, by design.
