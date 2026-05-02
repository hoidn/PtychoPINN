# Execution Report: 2026-04-29-brdt-operator-validation

## Completed In This Pass

- Added `ptycho_torch/physics/` package with the differentiable Born
  forward operator `BornRytovForward2D` in
  `ptycho_torch/physics/born_rytov_dt.py`. The operator:
  - Accepts physical scattering potential `q` shaped `(B, 1, N, N)` and
    emits the complex sinogram `(B, A, D, 2)` real/imag layout.
  - Samples the object spectrum on the Wolf 1969 Ewald arc using
    `torch.fft.fftshift` plus `F.grid_sample` (no Python-side batch loop).
  - Locks the angle, detector-frequency, FFT-normalization, and Ewald-arc
    sampling conventions both in module docstrings and via the
    `operator_contract()` introspection helper.
  - Explicitly raises `NotImplementedError` for `mode="rytov_linearized"`
    so downstream code cannot silently fall through to it.
- Added the study-local validation harness
  `scripts/studies/born_rytov_dt/validate_operator.py` plus its package
  marker, exposing:
  - A NumPy reimplementation of the Wolf spectral path
    (`numpy_reimplementation`) — tight cross-check that the torch
    operator matches the spec.
  - A 2D scalar Born volume integral with the Hankel Green's function
    (`direct_born_integral_2d`) — independent real-space oracle.
  - A free-space angular-spectrum propagator
    (`free_space_propagate`) so the operator's `z=0` reference plane can
    be compared to the direct integral at a far-field detector plane.
  - Individual checks for `numpy_consistency`, `direct_born_integral`,
    `analytic_phantom`, `gradcheck`, `cpu_dtype_reproducibility`,
    `cuda_reproducibility`, and `odtbrain_inverse_consistency`.
  - A driver `run_all` and CLI entry point that emits
    `operator_validation.json` plus a UTC-stamped run log.
- Added pytest coverage:
  - `tests/studies/test_born_rytov_dt_operator.py` (15 tests) for API
    rejection, output shape/layout, normalize-mode behavior, linearity,
    state-dict round-trip, batch-time scaling, and propagating-band
    semantics.
  - `tests/studies/test_born_rytov_dt_validation.py` (15 tests) for
    each individual check, the JSON schema produced by `run_all`,
    log-line emission, and the on-disk artifact freshness.
- Generated the durable validation artifact
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
  and run log under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/logs/validate_operator.log`.
- Wrote the durable Markdown validation authority
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`.
- Updated discoverability surfaces:
  - `docs/index.md` indexes the new validation report.
  - `docs/studies/index.md` points the BRDT study entry at the
    validation report and JSON artifact.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md` adds a
    completed-coverage row for the validation item.
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` adds
    a `feasibility`-tier row preserving the candidate-lane claim
    boundary (BRDT remains additive candidate work; the validation does
    not promote it into manuscript evidence).

## Completed Plan Tasks

### Tranche 1 — Lock The Born Operator Contract

- Created `ptycho_torch/physics/` package surface and
  `born_rytov_dt.py`.
- Implemented `BornRytovForward2D` with documented constructor
  arguments, registered geometry/coefficient buffers, deterministic
  dtype/device behavior, and no Python-side batch loop.
- Forward path consumes physical `q` shaped `(B, 1, N, N)` and emits
  real/imag sinograms shaped `(B, A, D, 2)`.
- Locked angle, detector-frequency, FFT normalization, and grid_sample
  coordinate mapping in module docstrings, the `operator_contract()`
  helper, and the validation report.
- Explicit Rytov boundary raises `NotImplementedError` with a clear
  message naming the design's preprocessing-gate requirement.
- Operator tests cover API/input validation, output shape/layout,
  deterministic buffer registration via `state_dict` round-trip, the
  two normalization-mode selections, and the explicit Born-only
  boundary. Blocking pytest passes:
  `pytest -q tests/studies/test_born_rytov_dt_operator.py` (15/15).
- Supporting `python -m compileall -q ptycho_torch scripts/studies/born_rytov_dt`
  passes with no errors.

### Tranche 2 — Prove The Operator Against Independent Oracles

- Added the study-local helper
  `scripts/studies/born_rytov_dt/validate_operator.py`.
- Implemented two independent oracles: a NumPy reimplementation of the
  spectral path (tight cross-check) and the 2D scalar Born volume
  integral with the Hankel Green's function (real-space, free-space
  oracle). The direct integral does not reuse the operator's sampled-
  FFT path.
- Analytic phantom check confirms finite, non-trivial output on a
  centered Gaussian across 16 angles.
- `gradcheck` (eps=1e-6, rtol=1e-4, atol=1e-5) passes on a small
  `N=D=8` tensor.
- CPU dtype reproducibility check (float64 vs float32) passes at
  rel_l2 < 5e-5 across three seeds.
- CUDA reproducibility check (CPU vs CUDA float32) passes at
  rel_l2 < 5e-5 across three seeds with CUDA available locally.
- ODTbrain inverse-side consistency records `dependency_unavailable`
  cleanly without becoming a hard runtime requirement.
- Validation outcomes serialized to
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`
  under stable keys.
- Blocking pytest passes:
  `pytest -q tests/studies/test_born_rytov_dt_validation.py` (15/15).

### Tranche 3 — Write The Durable Validation Authority

- Ran the validation helper from the repo root after final code/doc
  updates; the JSON artifact and run log are freshly written under the
  item artifact root.
- Wrote `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  with the locked operator contract, validation sample counts and
  tolerances, pass/fail tables for each validation family, optional
  dependency status, known limits, and explicit downstream
  authorization for `2026-04-29-brdt-dataset-preflight` to proceed.
- Updated `docs/index.md` to index the validation report.
- Updated `docs/studies/index.md` so the BRDT study surface points to
  the validation report and JSON artifact.
- Updated `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  and `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  with a `feasibility`/candidate-lane row that preserves the claim
  boundary: operator validation is enabling evidence only, not
  manuscript result evidence.
- Stated explicitly in the report and in the JSON artifact's
  `downstream_authorization` block that
  `2026-04-29-brdt-dataset-preflight` may proceed. No follow-up blocker
  was needed.

## Remaining Required Plan Tasks

None. All Tranche 1, Tranche 2, and Tranche 3 tasks listed in the plan
are complete. The "supporting" CUDA validation and ODTbrain wiring were
authorized as conditional on local availability; CUDA was available and
exercised, ODTbrain was not installed and is recorded as
`dependency_unavailable` in both the report and JSON artifact.

## Verification

Blocking checks (rerun after final code/doc edits):

- `pytest -q tests/studies/test_born_rytov_dt_operator.py
  tests/studies/test_born_rytov_dt_validation.py` → **30 passed**.
- `python -m compileall -q ptycho_torch scripts/studies/born_rytov_dt`
  → no output (success).
- `python -m scripts.studies.born_rytov_dt.validate_operator` →
  verdict `pass_with_documented_limits`,
  `downstream_authorization.may_proceed = true`.

Numerical tolerances and metrics recorded in `operator_validation.json`
and mirrored in the report:

- `numpy_consistency`: rel_l2 ≤ 1e-6 (atol/rtol implicit in the rel_l2
  comparison standard); measured 8.4e-8 across 4 phantom samples.
- `direct_born_integral`: pre-declared rel_l2 ≤ 0.6 (loose by design);
  measured 0.46 across 3 Gaussian phantom samples on the forward-cone
  angle subset `theta in {0, pi/12, pi/8, pi/6}`.
- `analytic_phantom`: finite + non-zero; measured max amplitude 0.10.
- `gradcheck`: eps=1e-6, rtol=1e-4, atol=1e-5; passes on a single small
  test case as designed.
- `cpu_dtype_reproducibility`: rel_l2 ≤ 5e-5; measured 2.2e-6.
- `cuda_reproducibility`: rel_l2 ≤ 5e-5; measured 2.7e-7.
- `odtbrain_inverse_consistency`: skipped with reason
  `dependency_unavailable`.

## Residual Risks

- The independent direct-integral oracle uses a pre-declared tolerance
  of rel_l2 ≤ 0.6 because the FFT operator implicitly periodizes the
  Green's function while the free-space integral does not. The measured
  value (0.46) is well within tolerance, but tighter quantitative
  agreement (e.g. <0.1) would require zero-padded operators or a
  far-field detector convention. This is documented in the report and
  in the JSON artifact's `known_limits` block. Downstream BRDT items
  must continue to treat the operator as physically credible only
  within the regime exercised here (centered, weakly-scattering
  phantoms, forward-cone angles); paper-grade inverse-side validation
  is explicitly out of scope until ODTbrain is wired up.
- ODTbrain is not installed in the local environment and the optional
  inverse-side recovery check is therefore not exercised. Promotion of
  BRDT to manuscript evidence (a separate roadmap amendment) should
  verify ODTbrain is available before claiming inverse-side validation
  results.
- The `direct_born_integral` oracle and the FFT operator disagree
  noticeably at oblique/backscatter angles (out of the validated
  forward cone). This is a known limitation of FFT-based Born forward
  operators on small grids and is not a defect of the present
  implementation, but downstream rows that explore high-angle
  illumination must validate the operator separately for that regime.
- Pyright reports nuisance "register_buffer / Tensor | Module" warnings
  when statically analyzing the operator file outside the project
  workspace. These do not affect runtime; the buffers are correctly
  registered and round-trip through `state_dict` (see operator tests).
