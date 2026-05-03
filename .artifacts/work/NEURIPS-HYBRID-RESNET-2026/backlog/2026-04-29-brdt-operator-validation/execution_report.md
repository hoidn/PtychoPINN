# Execution Report: 2026-04-29-brdt-operator-validation

## Completed In This Pass

- Added two review-locking regression tests in
  `tests/studies/test_born_rytov_dt_validation.py` and verified that they
  failed before the fix:
  - the installed-`backpropagate_2d` path was passing the in-medium
    `wavelength_px` where ODTbrain expects vacuum `res`;
  - the `fourier_map_2d` fallback path was receiving unsupported
    `weight_angles`/`onlyreal`/`padding` kwargs and failed immediately.
- Fixed `scripts/studies/born_rytov_dt/validate_operator.py` so the optional
  ODTbrain check now:
  - converts the BRDT in-medium wavelength to the vacuum wavelength expected
    by ODTbrain via `lambda_0 = wavelength_px * medium_ri`;
  - calls `backpropagate_2d` and `fourier_map_2d` with their own valid
    signatures instead of reusing one incompatible kwarg set; and
  - records the ODTbrain-side vacuum wavelength in the installed-package
    check details for artifact transparency.
- Regenerated the BRDT validation artifacts from the current checkout:
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
  - `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/logs/validate_operator.log`
- Updated
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  so the durable authority now documents the API-specific ODTbrain wiring and
  the vacuum-wavelength conversion at the optional inverse boundary.

## Completed Current-Scope Work

- Fixed both blocking review findings in already-implemented current-scope
  work:
  - the optional ODTbrain inverse-side check now has a real
    `fourier_map_2d` fallback instead of a broken shared call path;
  - the installed-package ODTbrain path now validates against the correct
    physical units by converting to vacuum wavelength at the ODTbrain
    boundary.
- Re-ran the current-scope verification contract after the fixes:
  - `pytest -q tests/studies/test_born_rytov_dt_operator.py tests/studies/test_born_rytov_dt_validation.py`
    → `33 passed`
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
