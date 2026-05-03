# Execution Report: 2026-04-29-brdt-operator-validation

## Completed In This Pass

- Replaced the stubbed installed-`odtbrain` branch in
  `scripts/studies/born_rytov_dt/validate_operator.py` with real optional
  inverse-side wiring:
  - prefers `odtbrain.backpropagate_2d`, with `fourier_map_2d` as a fallback
    if needed;
  - generates `16` deterministic held-out weak-scattering Gaussian phantoms;
  - runs the Born operator in `odtbrain_compatible` mode to produce complex
    sinograms;
  - reconstructs with ODTbrain when available; and
  - evaluates low-frequency recovery of `q` after least-squares complex-scale
    alignment, recording `pass` or `fail` instead of the removed
    `wired_but_not_implemented` stub state.
- Added review-locking tests in
  `tests/studies/test_born_rytov_dt_validation.py` that now require:
  - the missing-dependency path to report only `dependency_unavailable`;
  - the installed-`odtbrain` path to produce a real `pass`/`fail` result with
    sample count and API provenance; and
  - the generated payload and committed JSON artifact to derive their
    direct-integral tolerance text from the actual check result.
- Regenerated
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
  so the durable artifact now records the correct direct-integral tolerance
  (`0.6`) and the current local optional-check skip reason
  (`dependency_unavailable`).
- Updated
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  so the human-facing authority no longer implies the inverse-side check is
  unimplemented; it now states that the check is implemented but not runnable
  locally without ODTbrain.

## Completed Current-Scope Work

- Fixed the blocking implementation-review issue: the optional ODTbrain
  inverse-side consistency check is now actually wired for installed
  environments and records `pass`, `fail`, or `dependency_unavailable` as the
  approved plan requires.
- Fixed the machine-readable artifact inconsistency: the `known_limits` text in
  `operator_validation.json` now stays aligned with the direct-integral
  tolerance emitted by the validation check itself.
- Re-ran the current-scope verification contract after the fixes:
  - `pytest -q tests/studies/test_born_rytov_dt_operator.py tests/studies/test_born_rytov_dt_validation.py`
    → `31 passed`
  - `python -m compileall -q ptycho_torch scripts/studies/born_rytov_dt`
    → success
  - `python -m scripts.studies.born_rytov_dt.validate_operator`
    → wrote fresh JSON/log artifacts, verdict `pass_with_documented_limits`

## Follow-Up Work

- None required for approval within the current backlog item.
- When ODTbrain becomes available locally, rerun
  `python -m scripts.studies.born_rytov_dt.validate_operator` so the optional
  inverse-side result can be exercised with the now-implemented path and
  archived under the same artifact root.

## Residual Risks

- The local environment still lacks ODTbrain, so the optional inverse-side
  check remains `dependency_unavailable` in the current artifact even though
  the installed-package branch is now implemented.
- The direct-integral oracle remains intentionally loose (`rel_l2 <= 0.6`) in
  the validated forward-cone regime because the FFT operator is periodic while
  the Hankel-integral oracle is free-space; that limitation is unchanged and
  remains documented in the report and JSON artifact.
- The new ODTbrain pass/fail criterion measures low-frequency recovery after
  least-squares complex-scale alignment. That is the intended current-scope
  feasibility signal, not a claim of exact absolute-scale or phase recovery.
