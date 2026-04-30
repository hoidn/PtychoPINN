# Execution Report

## Completed In This Pass

- fixed the missing-package path in
  `scripts/studies/lines128_uno_preflight.py` so a missing `neuralop`
  import now triggers the one-shot `python -m pip install
  neuraloperator==2.0.0` retry required by the approved plan before emitting
  `blocked_neuraloperator_missing_or_incompatible`
- corrected `package_provenance.pip_show_present` so the helper treats the
  standard `WARNING: Package(s) not found: neuraloperator` transcript as
  `false` instead of misreporting any non-empty `pip show` text as package
  presence
- broadened constructor-failure handling so non-`TypeError` incompatibilities
  also emit a machine-readable blocker artifact instead of escaping the helper
- added focused regression coverage in
  `tests/studies/test_lines128_uno_preflight.py` for:
  missing-import install/retry success, honest missing-package provenance, and
  non-`TypeError` constructor incompatibilities

## Completed Current-Scope Work

- implementation review finding 1 is closed: the helper now performs the
  approved one-shot install-and-rerun path, refreshes `pip show` state after
  the install attempt, and records truthful package presence in the emitted
  provenance
- implementation review finding 2 is closed: incompatible `UNO`
  constructors now return `blocked_neuraloperator_missing_or_incompatible`
  with `package_status=constructor_incompatible` and a final
  `preflight_decision.json` artifact instead of crashing
- no design, layout, or ownership deviation was required; the frozen preflight
  contract, artifact root, and later generator-integration boundary remain
  unchanged

## Follow-Up Work

- later backlog item
  `2026-04-30-cdi-lines128-uno-generator-integration` remains the next
  implementation step for actual `neuralop_uno` registry/runtime wiring
- actual U-NO benchmark rows, append-only paper-table promotion, and launcher
  routing remain out of scope for this preflight fix pass

## Residual Risks

- the one-shot install branch is covered by regression tests but was not
  exercised live in this pass because the current `ptycho311` environment
  already imports `neuralop 2.0.0`
- the helper still only proves environment/API readiness; generator
  registration, Lightning wiring, launcher routing, and row execution are
  intentionally deferred to the later integration item
- the environment may still emit non-fatal TensorFlow/XLA duplicate
  registration warnings on startup; this pass did not change that behavior

## Verification

- focused pytest selector:
  `pytest -q tests/studies/test_lines128_uno_preflight.py`
  -> archived green log:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/pytest_review_followup.log`
- required deterministic input check:
  `python - <<'PY' ...`
  -> archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/required_inputs_check_review_followup.log`
- required import check in `ptycho311`:
  `python - <<'PY' import neuralop; from neuralop.models import UNO; ...`
  -> archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/neuralop_import_check_review_followup.log`
- compile gate:
  `python -m compileall -q scripts/studies`
  -> archived at
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-30-cdi-lines128-uno-design-preflight/verification/compileall_review_followup.log`
