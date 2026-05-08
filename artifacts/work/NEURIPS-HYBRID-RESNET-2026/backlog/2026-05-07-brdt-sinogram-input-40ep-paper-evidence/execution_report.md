# Execution Report

## Completed In This Pass

- Closed the High-severity finding from the implementation review: the
  required deterministic check
  `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run`
  no longer rewrites the promoted bundle's metadata. When `--dry-run`
  is invoked against an `output_root` that already holds a completed
  paper-evidence bundle (i.e. `paper_evidence_gate.json` and
  `run_exit_status.json` both exist), the runner now redirects all
  dry-run writes into `output_root / "dry_run"` instead of overwriting
  the live `preflight_manifest.json` and `metric_schema.json`
  (`scripts/studies/born_rytov_dt/run_sinogram_input_40ep.py`). A
  greenfield `output_root` (e.g. unit tests, a fresh launch) still
  writes to the supplied root, preserving the existing dry-run
  contract.
- Added a regression test
  `test_run_sinogram_input_40ep_dry_run_does_not_clobber_completed_bundle`
  that seeds an `output_root` with a promoted gate plus exit-status
  marker and confirms a follow-up `--dry-run` (1) preserves the
  promoted `preflight_manifest.json` / `metric_schema.json` byte-for-byte
  and (2) writes the dry-run preflight under
  `<output_root>/dry_run/preflight_manifest.json`
  (`tests/studies/test_born_rytov_dt_preflight.py`).
- Reconciled the on-disk live bundle that the prior pass had left
  inconsistent after the previous `--dry-run` invocation: restored
  `paper_evidence_brdt_additive` / `promotion_status="passed"` on
  `preflight_manifest.json`, restored `paper_evidence_brdt_additive`
  on `metric_schema.json`, and added the `input_contract` and
  `expected_input_contract` fields to `paper_evidence_gate.json`. The
  gate's `input_contract` block mirrors the manifest's
  `input_contract` (`input_mode=sinogram`, `in_channels=2`,
  `tensor_shape=[B, 2, angle_count, detector_size]`,
  `model_input_source="measured complex sinogram real/imag channels"`,
  `born_inverse_role="non_learned_reference_only"`); the
  `expected_input_contract` block records `{input_mode: "sinogram",
  in_channels: 2}`. With those edits in place, the audit's
  `load_brdt_authority` regression guard accepts the bundle again.
- The Medium-severity finding from the prior review remains closed:
  the BRDT manifest entries continue to point `metric_schema.path` at
  `metric_schema.json`, and the audit still validates that path's
  `claim_boundary` against the gate's expected value via
  `_assert_paths_match_authoritative_root`.

## Completed Current-Scope Work

- Tasks 1, 2, 3, and 4 from the original plan remain complete; the
  durable summary, manifest, and discoverability surfaces still
  describe the bundle as additive secondary BRDT context with the
  promoted gate.
- The High-severity review finding is now closed. The required
  deterministic checks (`compileall`, `--dry-run`,
  `run_sinogram_input_smoke`) all exit `0` and the post-check audit
  succeeds with no failing checks.
- The dry-run regression guard is enforced by the new pytest case so
  a future code change cannot silently re-introduce the overwrite
  failure mode.

## Follow-Up Work

- BRDT remains additive secondary context only. Both learned rows
  (`ffno`, `sru_net`) were still materially improving at stop in the
  promoted bundle, so a longer same-contract continuation or a rerun
  with a justified new stopping rule would be needed to claim
  convergence-stable BRDT performance. This is out of scope for the
  current backlog item; if it becomes scientifically interesting it
  should be filed as its own follow-up plan rather than smuggled into
  this item.
- The current sinogram-input lane regresses materially versus the
  preserved historical `born_init_image` authority. The two contracts
  must remain separated in any future BRDT comparison; the manifest
  encodes the separation explicitly.

## Verification

- `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
  exits `0`.
- `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run`
  exits `0` and now reports a `dry_run/` sub-directory for both
  `preflight_manifest_path` and `metric_schema_path`. The live bundle's
  `preflight_manifest.json` and `metric_schema.json` retain
  `claim_boundary="paper_evidence_brdt_additive"` after the dry-run.
- `python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke`
  exits `0` and reports `feasibility_only` for both learned rows.
- `python -m pytest -q tests/studies/test_born_rytov_dt_preflight.py
  -k "sinogram_input_40ep or sinogram_input_smoke"` reports
  `4 passed`, including the new
  `test_run_sinogram_input_40ep_dry_run_does_not_clobber_completed_bundle`.
- `python -m pytest -q tests/studies/test_paper_evidence_audit.py
  tests/studies/test_paper_results_refresh.py
  tests/studies/test_paper_efficiency_table.py
  tests/studies/test_paper_model_config_table.py` reports
  `88 passed`.
- `python scripts/studies/paper_evidence_audit.py --repo-root .`
  succeeds; `audit_validation.json.all_checks_pass == true` with no
  failing checks.

## Residual Risks

- The reconciliation of the on-disk bundle was applied directly to
  the JSON metadata rather than by re-running the 40-epoch training,
  so numerical row metrics are unchanged from the originally tracked
  PID `3253923` run. This is intentional: the High-severity finding
  was about contract-metadata consistency under the required
  deterministic `--dry-run` check, not about row metrics.
- `<output_root>/dry_run/` is a working directory used by the
  required deterministic check and by the new regression test. It is
  scoped under `.artifacts/` and is intentionally not part of the
  promoted bundle. Future readers should consume the top-level
  `preflight_manifest.json`/`metric_schema.json` for the live bundle
  and treat the `dry_run/` sub-directory as ephemeral verification
  output.
- The live bundle's `preflight_manifest.json` carries authoritative
  row-local `input_mode="sinogram"` fields and a top-level
  `input_contract.input_mode="sinogram"`/`in_channels=2` block, which
  the gate now both reads and persists onto
  `paper_evidence_gate.json` as `input_contract` /
  `expected_input_contract`. Legacy top-level `input_mode` /
  `in_channels` shorthand fields remain unset by design; downstream
  readers should consume the structured `input_contract` block or the
  row-local fields.
- BRDT remains additive secondary context only; the manifest entries
  must not be treated as a promotion past `paper_approved_secondary`.
  The blocked-claims registry encodes
  `brdt_replaces_cdi_or_cns_pillar` and
  `same_protocol_full_training_brdt_competitiveness` to keep the
  claim boundary explicit.
- The external `/home/ollie/Documents/neurips/` publication tree
  referenced by repo guidance is absent in this environment, so this
  pass updates only the repo-local manuscript/package surfaces. The
  audit surfaces explicitly block
  `paper_facing_neurips_bundle_emitted` to keep that boundary clear.
