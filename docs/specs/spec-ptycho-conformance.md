# spec-ptycho-conformance.md — Acceptance Tests (Normative)

Overview (Normative)
- Purpose: Define executable acceptance tests (PTY‑AT‑XXX) to certify implementations conform to physics, math, data contracts, and runtime guardrails.

Conformance Profiles (Normative)
- Forward Physics Profile:
  - PTY‑AT‑001 Forward amplitude equivalence (FFT path, normalization).
  - PTY‑AT‑002 Poisson sampling semantics (amplitude→counts→sqrt).
- Grouping and Coordinates Profile:
  - PTY‑AT‑010 Group formation and shapes (C=gridsize², coords semantics).
  - PTY‑AT‑011 Translation extract/inverse reassemble parity (round‑trip).
  - PTY‑AT‑012 Channel index row‑major mapping preserved by format transforms.
- Model Runtime Profile:
  - PTY‑AT‑020 Intensity scaling symmetry (X and Y_I both scaled).
  - PTY‑AT‑021 Positive intensity monotonicity for Poisson NLL (no log of non‑positive) with activation/epsilon compliance.
- Workflow Integration Profile:
  - PTY‑AT‑030 Loader/Container shapes and dtypes (X, Y_I, Y_φ, coords_*).
  - PTY‑AT‑031 Inference path determinism with Poisson disabled.
  - PTY‑AT‑032 Ground‑truth absence handling (placeholder Y; MAE disabled).
  - PTY‑AT‑033 Coordinate key compatibility (canonical vs legacy keys accepted).
- Stitching/Evaluation Profile:
  - PTY‑AT‑040 Stitch border math (clip sizes) and shape.
  - PTY‑AT‑041 Stitching config contract (required keys present).

Acceptance Tests (Normative)
- PTY‑AT‑001 Forward amplitude equivalence
  - Setup: Build a random complex object patch and probe; compute Ψ, FFT2; compare `A = sqrt(shift(|F|²/(N²)))` to `PadAndDiffract` with `pad=False`.
  - Expectation: max relative error ≤ 1e‑6; energy conservation holds within tolerance.

- PTY‑AT‑002 Poisson sampling semantics
  - Setup: Fix seed; feed deterministic amplitude; sample observed amplitude via `observe_amplitude`; compare empirical mean of `A_obs²` to `A²`.
  - Expectation: `mean(A_obs²) ≈ A²` within statistical tolerance; amplitude non‑negative and finite.

- PTY‑AT‑010 Group formation and shapes
  - Setup: Generate grouped data with `gridsize ∈ {1,2,3}`; verify shapes for X_full, `coords_offsets (B,1,2,1)`, `coords_relative (B,1,2,C)`, `nn_indices (B,C)`.
  - Expectation: Shapes match; `coords_relative` obeys `local_offset_sign = −1`; for C=1 seeds directly used.

- PTY‑AT‑011 Translation round‑trip
  - Setup: From a full object, extract patches via `extract_patches_position` and reassemble via `reassemble_position` (normalized), no jitter, bilinear.
  - Expectation: Reassembled object equals original central region within 1e‑6.

- PTY‑AT‑020 Intensity scaling symmetry
  - Setup: Simulated dataset via `diffsim.mk_simdata`; verify both X and Y_I are divided by the same `intensity_scale` and `X * s = X_before_normalization`.
  - Expectation: Assertion holds; violations SHALL fail.

- PTY‑AT‑021 Positive intensity for Poisson NLL
  - Setup: Run a forward pass; ensure predicted intensity > 0 everywhere before log; record zeros/negatives.
  - Expectation: No non‑positive predictions; compliance via strictly‑positive activation or epsilon ≥ 1e−12 before log.

- PTY‑AT‑012 Channel mapping preservation
  - Setup: Create synthetic `channels (B,N,N,C)` with distinct per‑channel tags; round‑trip through `_channel_to_flat → _togrid → _grid_to_channel`.
  - Expectation: Per‑channel tags appear at row‑major positions `(row = c//gridsize, col = c%gridsize)`; exact equality required.

- PTY‑AT‑030 Loader/Container shapes and dtypes
  - Setup: Load grouped dict to `PtychoDataContainer`; validate X float32, Y complex64 (optional), coords float32 with exact shapes.
  - Expectation: All shape/dtype checks pass.

- PTY‑AT‑031 Inference determinism (Poisson disabled)
  - Setup: Disable Poisson draw in `diffract_obj`; run inference twice with identical inputs.
  - Expectation: Outputs bitwise equal; otherwise record and fail.

- PTY‑AT‑032 Ground‑truth absence handling
  - Setup: Provide grouped dict without `Y`; load via `loader.load(...)`.
  - Expectation: Loader emits a shape‑compatible complex64 placeholder `Y`; training configuration disables MAE terms (e.g., `mae_weight = 0`).

- PTY‑AT‑033 Coordinate key compatibility
  - Setup: Provide grouped dicts using (a) canonical `coords_offsets`/`coords_relative` and (b) legacy `coords_start_offsets`/`coords_start_relative`.
  - Expectation: Loader accepts either and produces identical `coords` tensors.

- PTY‑AT‑040 Stitch border math
  - Setup: Stitch known patch layouts with `outer_offset`; verify computed border and overall shape.
  - Expectation: Clipping per spec yields correct final dimensions.

- PTY‑AT‑041 Stitching config contract
  - Setup: Call stitching with and without required config keys.
  - Expectation: With `N, gridsize, offset, nimgs_test` present, shapes and border math are correct; missing keys SHALL raise.

Commands (Informative)
- Provide pytest selectors once tests are wired; current code exposes the necessary APIs to construct fixtures.

Notes (Informative)
- Determinism with Poisson enabled is not required; tests may use Poisson disabled or rely on statistical checks.
