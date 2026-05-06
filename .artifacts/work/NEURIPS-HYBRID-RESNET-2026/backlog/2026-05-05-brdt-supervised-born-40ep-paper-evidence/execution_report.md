# Execution Report

## Completed In This Pass

- Closed implementation-review High issue 1 (`--rebuild-meta-only` rewrote
  the authoritative `runtime_provenance.json` from the rebuild process
  instead of preserving the original 40-epoch training-run provenance).
  The rebuild path no longer calls `_write_top_level_provenance` (which
  re-sampled `git_sha`/`git_dirty`/`hostname`/`gpu_count`/Python/PyTorch
  fields from the rebuild host). A new helper
  `_amend_existing_runtime_provenance_for_rebuild` instead reads the
  on-disk `runtime_provenance.json`, validates that every required
  original field is present, attaches the rebuild's pid/timestamp/git/host
  block under `meta_rebuild`, and writes the payload back. The dataset
  identity and split manifests, which are deterministically derivable
  from the dataset authority, are still regenerated via a new
  `_write_dataset_and_split_manifests` helper. If
  `runtime_provenance.json` is missing, unparseable, or lacks any of
  `git_sha`, `git_dirty`, `hostname`, `platform`, `gpu_count`,
  `python_executable`, `python_version`, `torch`, `launch_timestamp_utc`,
  or `tracked_pid`, the rebuild raises rather than fabricating values
  from the rebuild host. See
  `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`
  (`_amend_existing_runtime_provenance_for_rebuild`,
  `_write_dataset_and_split_manifests`, `_rebuild_meta_only_inner`).
- Closed implementation-review High issue 2 (the promotion gate
  underimplemented the plan's "repo git SHA/dirty, Python/PyTorch/CUDA/
  GPU/host provenance" prerequisite and the
  `paper_evidence_package_design.md` evidence-amendment requirement). The
  gate's `provenance_checks` payload now exposes two new keys:
  - `python_provenance` — true only when `runtime_provenance.json`
    records both `python_executable` and `python_version`;
  - `torch_provenance` — true only when the recorded `torch` block
    carries `version`, a non-null `cuda_available`, and `cuda_version`.
  The `evidence_surfaces_prepared` check now also requires
  `paper_evidence_package_design.md` to reference this backlog item,
  the canonical artifact root, and the manifest's authoritative
  claim-boundary string whenever the manifest's authoritative entry
  advertises the promoted boundary `paper_evidence_brdt_additive`. When
  the manifest still records the pre-gate boundary, the package-design
  amendment is not required (the plan only authorizes amending the
  design doc on a passed gate). See
  `scripts/studies/born_rytov_dt/run_brdt_40ep_paper_evidence.py`
  (`_build_provenance_checks`, `_check_evidence_surfaces_consistent`,
  `PAPER_EVIDENCE_PACKAGE_DESIGN_PATH`).
- Regenerated the on-disk bundle at
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-05-brdt-supervised-born-40ep-paper-evidence/`
  via `--rebuild-meta-only`. Recomputed gate continues to pass honestly:
  `claim_boundary=paper_evidence_brdt_additive` /
  `promotion_status=passed`, all thirteen `provenance_checks` true (now
  including `python_provenance` and `torch_provenance`),
  `failed_gate_checks=[]`. `runtime_provenance.json` retains the
  original training-run `pid=tracked_pid=2800210`,
  `launch_timestamp_utc=2026-05-06T03:54:32.039301+00:00`, the recorded
  `python_executable`/`python_version`/`torch.*`/`hostname`/`platform`/
  `gpu_count`/`git_sha`/`git_dirty` fields, and the rebuild's pid /
  timestamp / git SHA / host / argv now appear under a separate
  `meta_rebuild` block. `metrics.json`, `combined_metrics.json`,
  `metric_schema.json`, and `preflight_manifest.json` continue to
  advertise `claim_boundary=paper_evidence_brdt_additive`.
- Updated the durable summary's Reproducibility & Meta Provenance section
  (`docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_supervised_born_40ep_paper_evidence_summary.md`)
  to document the strengthened guarantees: the new `python_provenance`
  and `torch_provenance` provenance checks; the new requirement that
  `paper_evidence_package_design.md` reference this backlog item, root,
  and boundary whenever the manifest records the promoted boundary; and
  the rebuild path's preservation of every recorded original
  `runtime_provenance.json` field with refusal to fabricate when the
  original payload is missing or malformed.
- Added regression tests covering each strengthened check:
  - `test_rebuild_meta_only_preserves_original_runtime_provenance_fields`
    seeds `runtime_provenance.json` with sentinel values for
    `git_sha`/`git_dirty`/`hostname`/`platform`/`gpu_count`/
    `python_executable`/`python_version`/`torch`/`tracked_pid`/
    `launch_timestamp_utc` (all distinct from anything the rebuild
    process would record), runs `rebuild_meta_only`, and asserts every
    sentinel value is preserved exactly while a separate `meta_rebuild`
    block records the rebuild process's distinct identity.
  - `test_rebuild_meta_only_refuses_when_runtime_provenance_missing`
    proves the rebuild path raises `FileNotFoundError` for a missing
    payload, `RuntimeError` for an unparseable payload, and
    `RuntimeError` for a parseable payload missing required original
    fields.
  - `test_provenance_checks_validate_python_and_torch_fields` proves the
    gate's `python_provenance` check fails when `python_executable` is
    absent, the `torch_provenance` check fails when `torch.cuda_version`
    is missing, and the `torch_provenance` check fails when the entire
    `torch` block is dropped.
  - `test_evidence_surfaces_consistency_requires_package_design_when_promoted`
    proves the surfaces check fails when the manifest records the
    promoted boundary but `paper_evidence_package_design.md` is missing
    or fails to reference the canonical artifact root, passes when all
    five surfaces agree, and stays permissive when the manifest still
    records the pre-gate boundary so a non-promoted bundle is not
    blocked by the absence of an amendment.
  - The pre-existing
    `test_evidence_surfaces_consistency_check_requires_all_surfaces`
    test was extended to populate the package-design surface alongside
    the other four when asserting the all-surfaces-pass case, since
    the manifest in that fixture records the promoted boundary.

## Completed Current-Scope Work

- Both implementation-review High issues
  (`--rebuild-meta-only` overwriting authoritative runtime provenance
  from the rebuild process; gate underimplementing the
  Python/PyTorch/CUDA and `paper_evidence_package_design.md`
  prerequisites) are addressed in code, on disk, and in regression
  tests.
- All approved plan tasks (1–6) remain satisfied. The strengthened
  Task 4 promotion-gate prerequisites now match the plan's
  "Python/PyTorch/CUDA/GPU/host provenance" requirement explicitly via
  the new `python_provenance` and `torch_provenance` provenance checks,
  and the strengthened Task 6 discoverability check now matches the
  plan's "checked-in evidence amendment consistent with the gate
  result" requirement when the manifest reports the promoted boundary.
- Required deterministic checks pass on the current branch:
  - `python` input-existence command (exit 0)
  - `pytest -q tests/studies/test_born_rytov_dt_adapters.py
    tests/studies/test_born_rytov_dt_preflight.py` — all tests pass,
    including the 4 new regression tests added in this pass.
  - `python -m compileall -q scripts/studies/born_rytov_dt
    ptycho_torch` (clean).

## Follow-Up Work

- Implementation-review's only Follow-Up entry is preserved:
  regenerate the sample-`255` classical/model-based comparator under this
  backlog item's own artifact root rather than inheriting it by lineage from
  the frozen `2026-04-29` baseline bundle, if the team wants the
  paper-facing visual bundle to be single-authority rather than
  baseline-lineage-derived.

## Residual Risks

- Both rerun rows were still materially improving at stop and never reduced
  LR, so the result remains bounded additive evidence rather than a full
  convergence claim. (Carried over from prior pass.)
- The on-disk `runtime_provenance.json` was last fully overwritten by an
  earlier `--rebuild-meta-only` invocation that ran under the old (now
  fixed) rebuild path. The current top-level fields therefore reflect that
  prior rebuild's host snapshot rather than the original 2026-05-06
  03:54:32 training run; only `pid`/`tracked_pid` and
  `launch_timestamp_utc` are guaranteed to match the original training
  process. Going forward the new code preserves whatever is on disk
  exactly. A clean reset would require retraining the bundle, which is
  out of scope for this review-driven correction pass; this trade-off is
  accepted because the alternative (leaving the rebuild path able to
  fabricate or rotate provenance silently) is the worse failure mode.
- The on-disk regeneration was performed on the currently-dirty
  `fno-stable` branch, so the `meta_rebuild` block records
  `rebuild_git_dirty=true`. After committing this pass's runner / test /
  report changes, an additional `--rebuild-meta-only` run on the clean
  tree would flip the rebuild's flag to `false`; the recorded values are
  honest at write time either way.
- The strengthened `evidence_surfaces_prepared` check uses the structured
  `paper_evidence_manifest.json` entry as the authoritative claim-boundary
  source and only requires `paper_evidence_package_design.md` to reference
  the boundary when the manifest records the promoted boundary. A future
  change that updates the manifest's structured entry alongside all other
  surfaces consistently to a different value would pass without flagging
  the rename. This is intentional: the gate validates internal consistency
  of the surfaces tracking this item, not the global correctness of the
  boundary value. A change that updates only some surfaces will continue
  to fail the check.
- The `python_provenance` and `torch_provenance` checks read keys from
  `runtime_provenance.json` exactly as written by
  `_capture_extended_runtime_provenance`. A future refactor that renames
  any of `python_executable`, `python_version`, `torch.version`,
  `torch.cuda_version`, or `torch.cuda_available` would silently fail the
  new gate keys. The current shared writer/reader contract is the only
  guarantee against that drift; the regression tests assert the keys
  the gate reads, but a code-level rename in the writer plus a parallel
  rename in the gate would not be caught.
- The `--rebuild-meta-only` path is now strict against both missing
  `run_exit_status.json` and missing `runtime_provenance.json`. An
  operator who has retained sufficient original training evidence in
  alternate form but lost either JSON file will have to write it back
  manually with honest values before invoking the rebuild. This is the
  deliberate trade-off to prevent silent fabrication of completion or
  runtime evidence after the original artifacts have been lost.
