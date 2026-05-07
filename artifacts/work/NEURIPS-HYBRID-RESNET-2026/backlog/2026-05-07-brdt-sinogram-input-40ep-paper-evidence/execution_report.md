# Execution Report

## Completed In This Pass

- Made `build_paper_evidence_gate()` enforce the BRDT sinogram-input
  manifest contract. The gate now accepts `input_contract` and
  `expected_input_contract` arguments, fails promotion with explicit
  `input_contract.<key>` entries when the preflight manifest disagrees,
  and records both the actual and expected input contract on the gate
  payload (`scripts/studies/born_rytov_dt/convergence.py`).
- Wired `run_sinogram_input_40ep.py` to load the preflight manifest's
  `input_contract` block and pass it (alongside the new
  `EXPECTED_INPUT_CONTRACT = {input_mode: "sinogram", in_channels: 2}`)
  into `build_paper_evidence_gate()`. Future live runs cannot promote a
  bundle whose preflight manifest disagrees with the sinogram contract.
- Reconciled the existing on-disk bundle by re-applying the post-gate
  reseed to its `preflight_manifest.json`, `metric_schema.json`,
  `metrics.json`, and `combined_metrics.json`. All four now record
  `claim_boundary=paper_evidence_brdt_additive` and (where applicable)
  `promotion_status=passed`, eliminating the High-severity gate-vs-manifest
  inconsistency flagged by the implementation review.
- Fixed the BRDT manifest normalizer in `scripts/studies/paper_evidence_audit.py`
  so `metric_schema.path` resolves to `metric_schema.json` (not `metrics.json`).
  Added `metric_schema_path` and `preflight_manifest_path` to the BRDT
  default inputs and to the `_assert_paths_match_authoritative_root`
  pillar check.
- Extended `load_brdt_authority` so the same-pillar disagreement guard
  also reads `preflight_manifest.json` and the new `metric_schema.json`,
  and refuses to load if either disagrees with the gate on
  `claim_boundary` or `promotion_status`. This is the regression guard
  that closes the High-severity issue at the audit boundary.
- Reran `python scripts/studies/paper_evidence_audit.py --repo-root .`.
  The refreshed `paper_evidence_manifest.json` now records the corrected
  `metric_schema.path` for all three BRDT rows
  (`brdt:classical_born_backprop`, `brdt:ffno`, `brdt:sru_net`). The
  audit summary text was unchanged because the path key is not
  templated into it; the in-zip audit summary therefore did not need a
  rebuild.
- Added regression tests:
  - `test_paper_evidence_gate_enforces_sinogram_input_contract` covers
    the matching, mismatched, and missing input-contract paths through
    the gate.
  - `test_brdt_manifest_metric_schema_path_resolves_to_metric_schema_json`
    fails if any BRDT row in the rendered manifest carries a
    `metric_schema.path` that is not `metric_schema.json`.
  - `test_load_brdt_authority_rejects_preflight_manifest_disagreement`
    fails the audit when a tampered preflight manifest disagrees with
    the gate on `promotion_status` or `claim_boundary`.

## Completed Current-Scope Work

- The High-severity finding is resolved: the gate now enforces the
  input contract, the on-disk bundle is internally consistent, and the
  audit refuses to load a BRDT bundle whose preflight manifest disagrees
  with the gate.
- The Medium-severity finding is resolved: the BRDT manifest entries now
  point `metric_schema.path` at `metric_schema.json`, and the audit
  validates that path's `claim_boundary` against the gate's expected
  value.
- The follow-up regression tests requested by the review are in place
  and pass against the live bundle and live manifest.
- Tasks 1, 2, 3, 4 from the prior pass remain complete; nothing else in
  those units required rework in this pass.

## Follow-Up Work

- BRDT remains additive secondary context only. Both learned rows
  (`ffno`, `sru_net`) were still materially improving at stop, so a
  longer same-contract continuation or rerun with a justified new
  stopping rule would be needed to claim convergence-stable BRDT
  performance. This is not scoped to the current backlog item; it
  should be filed as its own follow-up plan if it becomes
  scientifically interesting.
- The current sinogram-input lane regresses materially versus the
  preserved historical `born_init_image` authority. The two contracts
  must remain separated in any future BRDT comparison; the new manifest
  encodes the separation explicitly.

## Verification

- `python -m compileall -q scripts/studies/born_rytov_dt ptycho_torch`
  exits `0`.
- `python -m scripts.studies.born_rytov_dt.run_sinogram_input_40ep --dry-run`
  exits `0` and emits `preflight_manifest.json` plus `metric_schema.json`.
- `python -m scripts.studies.born_rytov_dt.run_sinogram_input_smoke`
  exits `0` and reports `feasibility_only` for both learned rows.
- `python -m pytest -q tests/studies/test_born_rytov_dt_preflight.py
  tests/studies/test_paper_evidence_audit.py
  tests/studies/test_paper_results_refresh.py
  tests/studies/test_paper_efficiency_table.py
  tests/studies/test_paper_model_config_table.py`
  reports `187 passed in 377.28s`.
- `python scripts/studies/paper_evidence_audit.py --repo-root .`
  succeeds; `audit_validation.json.all_checks_pass == true` with no
  failing checks or summary-sync failures.
- On-disk reconciliation read-back:
  `preflight_manifest.json.claim_boundary=paper_evidence_brdt_additive`,
  `preflight_manifest.json.promotion_status=passed`,
  `metric_schema.json.claim_boundary=paper_evidence_brdt_additive`,
  `metrics.json.claim_boundary=paper_evidence_brdt_additive`,
  `combined_metrics.json.claim_boundary=paper_evidence_brdt_additive`.

## Residual Risks

- The reconciliation of the on-disk bundle was applied via the existing
  reseed helpers rather than by re-running the 40-epoch training, so
  numerical row metrics are unchanged from the originally tracked
  PID `3253923` run. This is intentional: the High-severity finding was
  about contract metadata consistency, not about row metrics.
- The live bundle's `preflight_manifest.json` carries authoritative
  row-local `input_mode="sinogram"` fields and a top-level
  `input_contract.input_mode="sinogram"`/`in_channels=2` block, which
  is what the gate now enforces. Legacy top-level `input_mode` /
  `in_channels` shorthand fields remain unset by design; downstream
  readers should consume the structured `input_contract` block or the
  row-local fields.
- BRDT remains additive secondary context only; the new manifest
  entries must not be treated as a promotion past
  `paper_approved_secondary`. The blocked-claims registry encodes
  `brdt_replaces_cdi_or_cns_pillar` and
  `same_protocol_full_training_brdt_competitiveness` to keep the claim
  boundary explicit.
- The external `/home/ollie/Documents/neurips/` publication tree
  referenced by repo guidance is absent in this environment, so this
  pass updates only the repo-local manuscript/package surfaces. The
  audit surfaces explicitly block `paper_facing_neurips_bundle_emitted`
  to keep that boundary clear.
