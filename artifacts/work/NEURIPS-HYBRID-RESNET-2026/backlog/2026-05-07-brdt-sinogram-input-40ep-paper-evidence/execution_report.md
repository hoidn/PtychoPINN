# Execution Report

## Completed In This Pass

- Added a BRDT additive-secondary authority section to
  `scripts/studies/paper_evidence_audit.py`: new default inputs, a
  `load_brdt_authority` loader that validates the gate / run exit / row
  roster against the live bundle, a row normalizer, manifest wiring under
  `additive_secondary_authorities.brdt`, claim/draftability/blocked-claim
  registry extensions, and `paper_approved_secondary` added to the frozen
  status vocabulary.
- Extended `validate_manifest` and `validate_summary_sync` to assert the
  BRDT additive-secondary authority is present and reflected in the rendered
  audit summary, and added three targeted tests in
  `tests/studies/test_paper_evidence_audit.py` for default inputs, manifest
  shape, and summary synchronization.
- Reran `python scripts/studies/paper_evidence_audit.py --repo-root .` so the
  authoritative `paper_evidence_manifest.json` and the
  `paper_evidence_package_audit_summary.md` now record the new
  `2026-05-07-brdt-sinogram-input-40ep-paper-evidence` root, claim boundary
  `paper_evidence_brdt_additive`, and `paper_approved_secondary` headline
  status alongside the historical `born_init_image` lineage notes.
- Fixed the stale BRDT efficiency-table wording in
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` and
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` so the durable
  narrative agrees with the active table contents (current additive-secondary
  BRDT under `paper_evidence_brdt_additive` rather than "historical secondary
  BRDT").
- Refreshed the in-zip copies of `paper_evidence_package_audit_summary.md`
  and `evidence_matrix.md` inside
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`
  so the packaged manuscript bundle no longer carries the stale audit view
  that omitted BRDT, and reverified the archive with `zip -T`.

## Completed Current-Scope Work

- Task 4 manifest/audit synchronization is now consistent with the durable
  discoverability surfaces: `paper_evidence_manifest.json`,
  `paper_evidence_package_audit_summary.md`, `paper_evidence_index.md`, and
  `evidence_matrix.md` all agree on the new BRDT backlog item, artifact root,
  claim boundary, and additive-secondary authority status.
- Tasks 1, 2, 3 from the prior pass remain complete; nothing in those units
  required rework.

## Follow-Up Work

- The BRDT bundle remains additive secondary context only. Both learned
  rows (`ffno`, `sru_net`) were still materially improving at stop, so a
  longer same-contract continuation or rerun with a justified new stopping
  rule would be needed to claim convergence-stable BRDT performance. This
  is not scoped to the current backlog item; it should be filed as its own
  follow-up plan if it becomes scientifically interesting.
- The current sinogram-input lane regresses materially versus the preserved
  historical `born_init_image` authority. The two contracts must remain
  separated in any future BRDT comparison; the new manifest now encodes the
  separation explicitly.

## Verification

- `python -m pytest -q tests/studies/test_paper_evidence_audit.py`
  `20 passed`
- `python -m pytest -q tests/studies/test_paper_evidence_audit.py
  tests/studies/test_paper_results_refresh.py
  tests/studies/test_paper_efficiency_table.py
  tests/studies/test_paper_model_config_table.py`
  `86 passed`
- `python scripts/studies/paper_evidence_audit.py --repo-root .`
  succeeded; `audit_validation.json.all_checks_pass == true` with new
  `brdt_additive_authority_present`,
  `brdt_headline_is_paper_approved_secondary`,
  `brdt_does_not_replace_pillars`,
  `brdt_root_present`, `brdt_status_present`, and
  `brdt_claim_boundary_present` all `true`.
- `zip -T docs/plans/NEURIPS-HYBRID-RESNET-2026/scr_ptychography_neurips_draft_package.zip`
  `OK`
- `unzip -p .../scr_ptychography_neurips_draft_package.zip
  paper_evidence_package_audit_summary.md | grep -c BRDT` returns `7`
  (the in-zip audit summary now records the BRDT additive authority).

## Residual Risks

- BRDT remains additive secondary context only; the new manifest entries
  must not be treated as a promotion of BRDT past
  `paper_approved_secondary`. The blocked-claims registry encodes
  `brdt_replaces_cdi_or_cns_pillar` and
  `same_protocol_full_training_brdt_competitiveness` to keep the claim
  boundary explicit.
- The live bundle's `preflight_manifest.json` keeps the authoritative
  row-local `input_mode="sinogram"` fields, but its legacy top-level
  `input_mode` / `in_channels` fields remain unset. Downstream readers
  should consume the row-local fields or the adapter-contract summary for
  the authoritative learned-input definition.
- The external `/home/ollie/Documents/neurips/` publication tree referenced
  by repo guidance is absent in this environment, so this pass updates only
  the repo-local manuscript/package surfaces. The audit surfaces explicitly
  block `paper_facing_neurips_bundle_emitted` to keep that boundary clear.
