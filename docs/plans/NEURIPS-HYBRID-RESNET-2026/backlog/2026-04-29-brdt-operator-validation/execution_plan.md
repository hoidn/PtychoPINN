# BRDT Operator Validation Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. This item authorizes only Born-mode forward-operator implementation, independent validation, and durable validation artifacts. Do not generate BRDT datasets, do not add BRDT training adapters, do not run the four-row BRDT preflight, do not register BRDT as a normal CDI generator, do not promote BRDT into manuscript evidence, and do not create worktrees. If any validation command becomes long-running, keep it under implementation ownership until terminal success or recoverable failure handling is complete; use tmux plus the `ptycho311` environment when appropriate.

**Goal:** Implement and independently validate a differentiable `BornRytovForward2D` Born forward operator so later BRDT dataset and training items can rely on a locked physical/operator contract.

**Architecture:** Split the work into three implementation units: the task-local PyTorch operator core, an independent validation harness with at least one non-self-generated oracle, and durable reporting/discoverability artifacts. The implementation must prove the operator against explicit conventions and tolerances before any BRDT dataset generation or neural training is allowed.

**Tech Stack:** PATH `python`, PyTorch (`torch.fft`, `grid_sample`, autograd), NumPy/SciPy for independent tiny-grid reference math as needed, optional `odtbrain` for inverse-side consistency, pytest, compileall, Markdown/JSON artifacts.

---

## Selected Objective

- Implement backlog item `2026-04-29-brdt-operator-validation`.
- Add or prototype `BornRytovForward2D` for the BRDT candidate lane under a
  task-appropriate physics surface.
- Lock the coordinate convention, FFT normalization, angle convention,
  detector-frequency convention, and real/imag output layout.
- Emit a pass/fail operator validation report before any downstream BRDT
  dataset, adapter, or training item proceeds.

## Scope

- Consume the binding design input at
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/born_rytov_dt_candidate_lane_design.md`.
- Add a task-local differentiable Born forward operator under
  `ptycho_torch/physics/`.
- Add the minimum study-local validation helper surface under
  `scripts/studies/born_rytov_dt/` needed to run and serialize operator
  validation.
- Add tests covering operator contract, shape/layout, independent numerical
  validation, gradient correctness, and dtype/device reproducibility behavior.
- Emit a durable Markdown validation report plus machine-readable JSON
  validation artifact.

## Explicit Non-Goals

- Do not generate BRDT datasets or dataset manifests in this item.
- Do not add BRDT loaders, Lightning modules, training scripts, or evaluation
  rows.
- Do not run the bounded four-row BRDT decision-support preflight.
- Do not implement Rytov mode beyond an explicit guarded boundary.
- Do not add limited-angle stress tests, FFNO rows, physics-only rows, or
  external FDTD mismatch context.
- Do not amend the NeurIPS roadmap, design brief, or manuscript claim set.
- Do not register BRDT as a normal CDI generator or alter CDI/CNS execution
  surfaces.

## Steering, Roadmap, and Candidate-Lane Constraints

- Steering and the roadmap keep CDI `lines128` and PDEBench CNS as the required
  manuscript pillars. BRDT remains additive candidate work on equal footing
  with WaveBench and cannot support paper-table claims without a later checked-in
  roadmap or evidence-package amendment.
- Because the required pillars are still active, this item must stay low-expense
  and validation-only. It may implement operator code and narrow validation
  helpers, but it must not drift into dataset generation or training.
- The reviewer note is binding: self-consistency-only checks are invalid. At
  least one independent oracle must be used so synthetic data and validation are
  not both driven solely by the same PyTorch operator path.
- Born mode is the only in-scope physical mode. If the public API exposes a
  Rytov mode selector for future compatibility, the implementation must keep it
  clearly unsupported or explicitly out of scope here.
- The operator contract must use physical scattering potential `q` as input and
  return complex detector data as real/imag channels. No normalization shortcut
  may change that contract inside the operator itself.
- Optional validation dependencies such as `odtbrain` must not become hard
  training/runtime requirements. Missing optional packages should yield a clear
  recorded skip state, not silent omission.

## Prerequisite Status

- Backlog frontmatter lists no formal prerequisites for this item.
- The progress ledger shows the initiative has already completed Phase 0
  evidence inventory, Phase 1 PDE benchmark selection, and several early Phase 2
  PDE/OpenFWI tranches, but no BRDT preflight tranche is complete yet.
- This item is the first executable BRDT gate. Downstream items
  `2026-04-29-brdt-dataset-preflight`,
  `2026-04-29-brdt-task-adapters`, and
  `2026-04-29-brdt-four-row-preflight`
  depend on the operator report being clear enough to trust the physical
  contract.
- Completion requires a checked-in report and machine-readable validation
  artifact. Downstream BRDT work must remain unauthorized if the report fails or
  leaves the contract ambiguous.

## Execution Rules

- Diagnose, fix, and rerun normal import, path, environment, numerical, or test
  failures before considering the item blocked.
- Reserve `BLOCKED` for missing external resources, unavailable hardware,
  unresolved dependency incompatibility after a documented narrow fix attempt,
  roadmap conflict, or an operator-validity gap that remains unrecoverable after
  a narrow reproduction/debug attempt.
- Keep scratch numerical outputs under `tmp/` or the item artifact root; remove
  repo-local scratch that is not part of the durable artifact set before
  completion.
- If `odtbrain` is unavailable locally, record `dependency_unavailable` in the
  report and JSON artifact and continue. Do not mark the item blocked solely for
  that reason.
- If CUDA is unavailable locally, record that the CUDA reproducibility check was
  not runnable. CPU validation remains the blocking baseline.

## Implementation Architecture

1. **Operator Core**
   - Owns `BornRytovForward2D`, geometry buffers, spectral sampling, FFT
     normalization, and the stable API contract from physical `q` to real/imag
     sinogram output.
2. **Independent Validation Harness**
   - Owns analytic or direct-integral oracle logic, gradient checks,
     dtype/device reproducibility checks, optional ODTbrain consistency, and the
     JSON serialization of validation outcomes.
3. **Durable Reporting And Discoverability**
   - Owns the Markdown validation report, artifact-root manifests/logs, and the
     discoverability/index updates that let later BRDT items find the operator
     authority without rereading raw backlog text.

## File and Artifact Targets

Mandatory contract outputs:

- `ptycho_torch/physics/__init__.py` if the new physics package surface needs an
  explicit export.
- `ptycho_torch/physics/born_rytov_dt.py`
- `scripts/studies/born_rytov_dt/__init__.py` if a study-local package is
  created.
- `scripts/studies/born_rytov_dt/validate_operator.py` and/or adjacent helper
  modules that compute independent validation evidence and serialize results.
- `tests/studies/test_born_rytov_dt_operator.py`
- `tests/studies/test_born_rytov_dt_validation.py`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`

Preferred packaging and discoverability updates:

- `docs/index.md` so the durable validation report is discoverable from the main
  documentation hub.
- `docs/studies/index.md` so the BRDT study entry points at the validation
  authority, not just the candidate design.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`

Supporting artifact locations, if useful:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/logs/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/figures/`
- `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/cases/`

Leave unchanged unless a concrete contract mismatch requires separate approval:

- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md`
- `docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md`

## Mandatory Durable Output Contract

Before this item is considered complete, the checked-in report and
`operator_validation.json` together must cover:

- operator identity:
  - module/class path
  - git SHA and dirty-state note
  - execution command(s)
  - environment summary, including Python, PyTorch, NumPy, and optional
    ODTbrain versions
- locked operator contract:
  - `grid_size`, `detector_size`, angle count/list summary
  - `wavelength_px`
  - `medium_ri`
  - mode status (`born` pass path; `rytov_linearized` unsupported or skipped)
  - FFT normalization mode
  - coordinate convention
  - detector-frequency convention
  - output layout `(B, A, D, 2)`
- independent validation evidence:
  - analytic phantom check status, sample count, tolerance, and summary metric
  - tiny-grid direct Born integral check status, sample count, tolerance, and
    relative-error summary
  - autograd or finite-difference gradient-check status, sample count, and
    tolerance
  - dtype reproducibility status for CPU float64 and CPU float32
  - CUDA reproducibility status when hardware is available, otherwise an
    explicit unavailable reason
  - ODTbrain inverse-side consistency status, version, sample count, and
    measured recovery summary if available, otherwise an explicit skipped reason
- interpretation fields:
  - overall verdict: `pass`, `fail`, or `pass_with_documented_limits`
  - any known scale or phase convention offset
  - any unresolved limitations that downstream dataset/training plans must honor
  - explicit statement that BRDT dataset/training work may proceed or must not
    proceed

The Markdown report should be readable on its own. The JSON artifact should use
stable keys so later BRDT plans can consume the operator authority without
re-reading raw implementation files.

## Execution Tranches

### Tranche 1: Lock The Born Operator Contract

**Purpose:** Create the task-local operator surface and freeze the Born-mode API
before numerical validation begins.

- [ ] Create the `ptycho_torch/physics/` package surface if it does not already
  exist and add `born_rytov_dt.py`.
- [ ] Implement `BornRytovForward2D` with documented constructor arguments,
  registered geometry buffers, deterministic dtype/device behavior, and no
  Python-side loops over batch dimension.
- [ ] Make the forward path consume physical `q` shaped `(B, 1, N, N)` and emit
  real/imag sinograms shaped `(B, A, D, 2)`.
- [ ] Lock the angle convention, detector-frequency ordering, FFT normalization,
  and `grid_sample` coordinate mapping in module docstrings or adjacent inline
  comments.
- [ ] Expose a clear boundary for unsupported Rytov behavior so downstream code
  cannot silently assume it works.
- [ ] Add or update operator tests for:
  - API/input validation
  - output shape/layout
  - deterministic buffer registration
  - normalization selection behavior
  - explicit Born-only boundary behavior

**Verification for Tranche 1**

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_operator.py`
- [ ] **Supporting:** `python -m compileall -q ptycho_torch scripts/studies/born_rytov_dt`
  may be run early for syntax/import feedback, but it does not replace the
  final blocking test gate.

### Tranche 2: Prove The Operator Against Independent Oracles

**Purpose:** Demonstrate that the operator is physically and numerically
credible without relying on self-consistency alone.

- [ ] Add the minimum study-local validation helper surface under
  `scripts/studies/born_rytov_dt/` that can execute and serialize operator
  validation runs.
- [ ] Implement at least one independent numerical oracle that does not reuse
  the same sampled-FFT path as the main operator. Preferred path: a tiny-grid
  direct Born integral comparison. Analytic point-phantom or Gaussian checks
  may supplement but must not be the only independence evidence.
- [ ] Add analytic phantom coverage that makes spectral support or phase/scale
  behavior easy to inspect.
- [ ] Add a small differentiable test case for `gradcheck` or an explicitly
  finite-difference comparison and record tolerances in code, not prose only.
- [ ] Add CPU dtype reproducibility checks for float64 and float32.
- [ ] Add CUDA reproducibility checks guarded by availability, with explicit
  skip behavior when CUDA is absent.
- [ ] Add optional ODTbrain inverse-side consistency wiring that records
  `pass`, `fail`, or `dependency_unavailable` without becoming a hard runtime
  requirement.
- [ ] Serialize the validation outcomes into
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-brdt-operator-validation/operator_validation.json`.

**Verification for Tranche 2**

- [ ] **Blocking:** `pytest -q tests/studies/test_born_rytov_dt_validation.py`
- [ ] **Supporting:** If CUDA is available, run the validation helper in a CUDA
  path and archive its log. If `odtbrain` is installed, run the optional
  inverse-side consistency path and archive its status. These checks become
  mandatory evidence only when the corresponding resource is locally available
  during execution; otherwise record the skip reason in the report and JSON.

### Tranche 3: Write The Durable Validation Authority

**Purpose:** Publish the operator verdict and make it discoverable for later
BRDT work.

- [ ] Run the validation helper from the repo root so the JSON artifact and any
  supporting logs or figures are freshly written under the item artifact root.
- [ ] Write
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/brdt_operator_validation_report.md`
  with:
  - the locked operator contract
  - validation sample counts and tolerances
  - pass/fail tables for each validation family
  - optional dependency status
  - any known scale or phase offset
  - explicit downstream authorization or stop decision
- [ ] Update `docs/index.md` to index the validation report.
- [ ] Update `docs/studies/index.md` so the BRDT study surface points to the
  validation authority.
- [ ] Update
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  and
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  with a `feasibility`/candidate-lane row that preserves the claim boundary:
  operator validation is enabling evidence only, not manuscript result evidence.
- [ ] State explicitly whether the next backlog item,
  `2026-04-29-brdt-dataset-preflight`, may proceed. If not, record the blocker
  and the narrow follow-up needed.

**Verification for Tranche 3**

- [ ] **Blocking:** Run the backlog item’s required deterministic checks exactly
  as written:
  - `pytest -q tests/studies/test_born_rytov_dt_operator.py tests/studies/test_born_rytov_dt_validation.py`
  - `python -m compileall -q ptycho_torch scripts/studies/born_rytov_dt`
- [ ] **Supporting:** Re-run the validation helper CLI or module entry point
  after the final code/doc updates and confirm the report and JSON paths exist
  and are freshly written.

## Completion Conditions

- The Born operator contract is implemented and documented.
- At least one independent oracle validates the operator; self-consistency-only
  evidence is absent.
- Gradient and CPU dtype checks pass; CUDA and ODTbrain outcomes are recorded as
  pass/fail/skipped with reasons.
- The durable report and machine-readable JSON artifact agree on tolerances,
  sample counts, and verdict.
- Discoverability surfaces are updated so later BRDT items can find the
  operator-validation authority directly.
- No BRDT dataset generation or neural training work is started inside this
  item.
