# Canonical Simulation Config and Boundary-Matched Probe Extension Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an outer-only, boundary-matched quadratic probe-extension option and make probe construction, synthetic-object generation, scan geometry, and detector/noise settings discoverable through one canonical top-level `SimulationConfig`.

**Architecture:** Introduce `SimulationConfig` beside the existing model/training/inference configs, with typed probe, object, scan, and detector sections. Normalize probe preparation to the existing ordered pipeline contract; the new behavior is a pipeline operation, not a silent reinterpretation of legacy `pad_extrapolate`. Move reusable probe transforms out of the grid-lines workflow, adapt existing flat configs and `DatagenConfig` without breaking old artifacts, and record the fully resolved simulation recipe in generated-dataset metadata and identity.

**Tech Stack:** Python 3.11, dataclasses, NumPy, SciPy sparse linear algebra, scikit-image phase unwrapping, TOML/YAML study and CLI configuration, pytest.

## Implementation status (2026-07-15)

Implementation and claim-matched simulation verification are complete. The final
plan suite reports 626 passing tests and one pre-existing, simulation-independent
CNN decoder-channel assertion (`[4, 4]` versus `[8, 16]`); that same assertion
failed before the simulation-config correction and does not exercise any file or
contract in this plan. A later repository-wide smoke exposed one direct
flat/nested compatibility defect when `dataclasses.replace()` changed only the
flat smoke geometry; the caller now replaces the nested `SimulationConfig` in
lockstep, and its exact falsifier passes. The Run1084 inspection PNG and JSON are
available under `tmp/` as specified below.

The refactoring roadmap subsequently made a post-fix comprehensive repository
rerun a major-chunk completion gate. That rerun is pending, so implementation is
landed but roadmap closure is not yet claimed.

---

## Governing decisions

1. `pad_extrapolate` retains its current behavior: edge-padded amplitude, a fitted quadratic phase evaluated over the entire target probe, followed by smoothing when requested. Existing datasets and sealed study artifacts keep that meaning.
2. The new canonical option is the explicit pipeline operation:

   ```text
   smooth:0.5|pad_extrapolate_boundary_matched:128
   ```

   It preserves the prepared central complex probe and applies quadratic phase only outside it. Do not add another preferred `probe_scale_mode` enum value; legacy scale modes remain compatibility shorthands that normalize to pipelines.
3. “Boundary matched” initially means phase continuity (C0), not derivative continuity (C1). A biharmonic/C1 extension is outside this plan.
4. The outer phase is not obtained by simply pasting an independently fitted quadratic around the center. It is a boundary-conditioned extension:

   - Let the prepared source probe be `P_c = A_c exp(i phi_c)` with unwrapped phase `phi_c`.
   - Fit `q(x, y) = a r^2 + b` to `phi_c` using the existing radial least-squares convention.
   - On the outer annulus `Omega`, solve the discrete Dirichlet problem

     ```text
     Laplacian(u) = 0                      in Omega
     u = phi_c - q                        on the inner boundary
     u = 0                                on the outer target boundary
     phi_outer = q + u
     ```

   - Center-copy `P_c` exactly. Use `phi_outer` only outside the centered source footprint. Edge-pad the source amplitude into the outer region.

   This makes the extrapolated phase agree with the source phase at the 64×64 boundary while relaxing toward the fitted quadratic at the 128×128 boundary.
5. For the boundary-matched operation, any requested complex smoothing occurs before extension. No global post-extension smoothing may alter the copied center.
6. The simulation config owns the properties of generated data. Model/training configs may repeat shape fields for runtime compatibility, but composition must reject disagreement rather than choose one silently.
7. The new transform creates a new dataset identity. Do not relabel, overwrite, or reuse sealed datasets generated with global `pad_extrapolate`.
8. Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Canonical configuration shape

Add these public dataclasses in `ptycho/config/config.py` and export them from the config package:

```python
@dataclass(frozen=True)
class ProbeSimulationConfig:
    source: Literal["custom", "ideal"] = "custom"
    source_path: Path | None = None
    transform_pipeline: str = "pad_preserve:64"
    mask_diameter: float | None = None


@dataclass(frozen=True)
class SyntheticObjectConfig:
    kind: Literal["lines", "dead_leaves", "natural_patch"] = "lines"
    image_size: tuple[int, int] = (392, 392)
    objects_per_probe: int = 4
    diffractions_per_object: int = 7000
    set_phi: bool = False


@dataclass(frozen=True)
class ScanSimulationConfig:
    kind: Literal["grid", "nongrid"] = "grid"
    grid_size: tuple[int, int] = (1, 1)
    offset: int = 4
    outer_offset_train: int = 8
    outer_offset_test: int = 20
    train_groups: int = 2
    test_groups: int = 2
    buffer: int = 0


@dataclass(frozen=True)
class DetectorSimulationConfig:
    photons_per_pattern: float = 1e9
    beamstop_diameter: float | None = None


@dataclass(frozen=True)
class SimulationConfig:
    N: int = 64
    probe: ProbeSimulationConfig = field(default_factory=ProbeSimulationConfig)
    object: SyntheticObjectConfig = field(default_factory=SyntheticObjectConfig)
    scan: ScanSimulationConfig = field(default_factory=ScanSimulationConfig)
    detector: DetectorSimulationConfig = field(default_factory=DetectorSimulationConfig)
    seed: int | None = None
```

Names may be adjusted during implementation only to avoid an actual collision with an existing public API. Keep the ownership boundaries above. Training-only values such as epochs, optimizer, learning rate, batch size, loss weights, output directory, and architecture do not belong in `SimulationConfig`. Model-time probe masks also remain model config; `simulation.probe.mask_diameter` describes only a mask baked into generated data.

## Task 1: Lock the configuration ownership and parsing contract

**Files:**

- Create: `tests/test_simulation_config.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho/config/__init__.py`
- Modify: `docs/specs/spec-ptycho-config-bridge.md`

- [x] **Step 1: Write failing construction, validation, and round-trip tests**

  Cover nested default factories, `Path` conversion, YAML/TOML-shaped mapping construction, stable serialization, and validation errors for non-square/nonpositive dimensions, invalid photon counts, invalid object counts, incompatible custom-probe source/path values, and a probe pipeline whose final size differs from `SimulationConfig.N`.

- [x] **Step 2: Write a failing ownership/composition test**

  Assert that combining a generated-data recipe with a `ModelConfig` of a different `N` or grid size fails with a field-specific error. Assert that training-only fields are rejected if placed under `simulation` rather than being silently ignored.

- [x] **Step 3: Implement and export the typed dataclasses**

  Add `validate_simulation_config()`, a mapping loader that rejects unknown keys, and a canonical JSON-compatible serializer. Keep `load_yaml_config()` backward compatible; do not make existing model/training/inference callers adopt simulation config.

- [x] **Step 4: Extend the legacy bridge only for simulation-owned legacy keys**

  Define explicit mappings for the legacy simulation entry points (`N`, grid geometry, counts, photon level, seed, and probe construction fields). A simulation entry point must call `update_legacy_dict(params.cfg, simulation_config)` before invoking legacy data generation, independently of the later training-config bridge.

- [x] **Step 5: Update the internal bridge specification**

  Document the additional one-way flow:

  ```text
  SimulationConfig -> update_legacy_dict(params.cfg, simulation_config) -> legacy simulation
  ```

  State that it does not transfer ownership of training/model fields and that conflicts are errors.

- [x] **Step 6: Run the focused tests**

  Run: `python -m pytest tests/test_simulation_config.py tests/torch/test_config_bridge.py -q`

  Expected: all selected tests pass.

## Task 2: Extract probe transforms and prove the boundary-conditioned solver

**Files:**

- Create: `ptycho/simulation/__init__.py`
- Create: `ptycho/simulation/probe_transform.py`
- Create: `tests/test_simulation_probe_transform.py`
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `tests/test_grid_lines_workflow.py`

- [x] **Step 1: Add a small failing numerical feasibility test**

  On an 8×8 synthetic complex probe extended to 16×16, assert:

  - the copied center is exactly the prepared source complex array;
  - the inner Dirichlet values equal `phi_c - q` to tolerance;
  - the outer correction is zero to tolerance;
  - the discrete Laplacian residual in the free annulus is below a declared solver tolerance;
  - the result is finite and deterministic.

  This test is the feasibility gate. If the discrete domain cannot be defined without corner ambiguity or cannot meet the stated tolerance deterministically, stop and resolve the numerical contract; do not fall back to the old global quadratic fit.

- [x] **Step 2: Add failing behavioral tests for the new operation**

  Test 64→128 placement (`[32:96, 32:96]`), even and odd padding, a nonquadratic source phase, constant amplitude, nonconstant amplitude, and an input whose phase already is exactly quadratic. Compare phase through complex ratios at the seam so wrapping at ±π does not create false failures.

- [x] **Step 3: Freeze legacy behavior before moving code**

  Add a deterministic fixture/hash or exact-array test showing that `pad_extrapolate:128|smooth:0.5` produces the same output before and after extraction. This protects all existing dataset identities.

- [x] **Step 4: Move reusable pipeline code**

  Move parsing, normalization, serialization, interpolation, padding, smoothing, masking, and application into `ptycho.simulation.probe_transform`. Re-export the established public helpers from `ptycho.workflows.grid_lines_workflow` so existing callers continue to work.

- [x] **Step 5: Implement `pad_extrapolate_boundary_matched`**

  Use a deterministic SciPy sparse solve over the outer annulus. Make the boundary pixels and corner ownership explicit in code. Record the solver tolerance and measured residual in a transform result/metadata structure; do not expose an unconstrained iterative result as successful.

- [x] **Step 6: Normalize the new option through the pipeline**

  Support this explicit form:

  ```text
  smooth:0.5|pad_extrapolate_boundary_matched:128
  ```

  Reject post-extension `smooth` for this operation unless the implementation can prove it leaves the center unchanged. Keep all existing scale-mode normalizations unchanged.

- [x] **Step 7: Run focused transform tests**

  Run: `python -m pytest tests/test_simulation_probe_transform.py tests/test_grid_lines_workflow.py -q`

  Expected: new numerical tests pass and existing grid-lines probe tests remain unchanged.

## Task 3: Make `SimulationConfig` the generation API without breaking callers

**Files:**

- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `ptycho_torch/config_factory.py`
- Modify: `tests/test_grid_lines_workflow.py`
- Modify: `tests/torch/test_config_factory.py`
- Modify: `tests/studies/test_torch_ablation_configuration.py`

- [x] **Step 1: Add failing adapter tests for `GridLinesConfig`**

  Refactor its target shape to a workflow envelope containing `simulation: SimulationConfig` plus output/training controls. Preserve construction from the established flat fields through one compatibility adapter. If both nested and flat forms specify the same value differently, fail with both field paths in the message.

- [x] **Step 2: Route dataset construction through the resolved simulation config**

  `build_grid_lines_datasets()` and `simulate_grid_data()` must consume one validated recipe. Remove local defaults for probe/source/geometry/noise after resolution so there is no second source of truth.

- [x] **Step 3: Adapt Torch `DatagenConfig`**

  Keep its class and checkpoint tuple position readable. Add explicit conversion to/from `SimulationConfig` for the fields it currently owns (`objects_per_probe`, `diff_per_object`, `object_class`, `image_size`, `probe_paths`, `beamstop_diameter`). Do not remove or reorder serialized Torch config payloads in this change.

- [x] **Step 4: Verify old call shapes and new ownership**

  Assert existing flat `GridLinesConfig(...)` callers and default `DatagenConfig()` payloads resolve identically, while new callers can pass only `simulation=...` and obtain the same generated inputs.

- [x] **Step 5: Run focused workflow/config tests**

  Run: `python -m pytest tests/test_grid_lines_workflow.py tests/torch/test_config_factory.py tests/studies/test_torch_ablation_configuration.py -q`

  Expected: all selected tests pass; existing checkpoint/runtime config shapes are unchanged.

## Task 4: Expose simulation configuration at script and study boundaries

**Files:**

- Modify: `scripts/simulation/simulate_and_save.py`
- Modify: `scripts/simulation/run_with_synthetic_lines.py`
- Modify: `scripts/studies/grid_lines_workflow.py`
- Modify: `scripts/studies/ablation/configuration.py`
- Modify: `scripts/studies/ablation/dataset_reference.py`
- Modify: `scripts/studies/ablation/runtime_reference_spec.py`
- Modify: `tests/studies/test_torch_ablation_configuration.py`
- Modify: `tests/studies/test_grid_lines_reference_performance.py`
- Create: `tests/scripts/test_simulation_config_cli.py`

- [x] **Step 1: Add failing CLI parsing and precedence tests**

  Add `--simulation-config PATH` to generation entry points. The file supplies the full nested recipe. Retained legacy flags are compatibility overrides with one documented precedence rule: explicit CLI value over file value over dataclass default. Reject unknown keys and conflicting legacy aliases; never ignore them.

- [x] **Step 2: Add a top-level `simulation` study namespace**

  Accept immutable `simulation.*` paths separately from `dataset`, `data`, `model`, `training`, `inference`, and `execution`. Simulation paths describe generation and dataset identity; they are not per-training-arm overrides. Reject a matrix that varies a simulation field while pointing at one already-materialized dataset.

- [x] **Step 3: Preserve old study specs as readable historical inputs**

  Map the old grid-lines `[dataset]` recipe fields to a resolved `SimulationConfig` in the reader. New specs write `[simulation]` and a schema-version bump. Do not rewrite or reseal existing study artifacts.

- [x] **Step 4: Expose the boundary-matched operation**

  A new study or CLI config must be able to specify:

  ```toml
  [simulation]
  N = 128
  seed = 3

  [simulation.probe]
  source = "custom"
  source_path = "datasets/Run1084_recon3_postPC_shrunk_3.npz"
  transform_pipeline = "smooth:0.5|pad_extrapolate_boundary_matched:128"
  ```

  An omitted `mask_diameter` means no simulation-time probe mask; do not encode absence as `false` or a numeric sentinel.

- [x] **Step 5: Run focused boundary tests**

  Run: `python -m pytest tests/scripts/test_simulation_config_cli.py tests/studies/test_torch_ablation_configuration.py tests/studies/test_grid_lines_reference_performance.py -q`

  Expected: both legacy and new manifests resolve; only new manifests emit the top-level simulation schema.

## Task 5: Bind generated datasets to the exact simulation recipe

**Files:**

- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `scripts/studies/ablation/dataset_reference.py`
- Modify: `tests/test_grid_lines_workflow.py`
- Modify: `tests/studies/test_grid_lines_reference_performance.py`

- [x] **Step 1: Add failing metadata and identity tests**

  Require each newly generated dataset to record:

  - the canonical serialized `SimulationConfig`;
  - a stable digest of that resolved config;
  - source-probe path and content hash;
  - normalized transform pipeline;
  - transformed-probe content hash;
  - boundary method, solver tolerance, and measured residual when applicable;
  - object, scan, detector/noise, and seed values.

- [x] **Step 2: Include the simulation digest in cache/output identity**

  Ensure global quadratic and boundary-matched datasets cannot resolve to the same cache key or be mistaken for the same immutable dataset descriptor.

- [x] **Step 3: Reject mismatched reuse**

  If an output root already contains a dataset with a different simulation digest, fail before writing. Regeneration must use a distinct output identity or explicit safe replacement outside sealed artifact roots.

- [x] **Step 4: Verify without training**

  Build only the smallest synthetic fixture needed to inspect metadata and hashes; do not launch a model-training study as part of this task.

  Run: `python -m pytest tests/test_grid_lines_workflow.py tests/studies/test_grid_lines_reference_performance.py -q`

  Expected: metadata identity tests pass and old fixtures remain readable.

## Task 6: Update the discoverable configuration guidance

**Files:**

- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/DATA_GENERATION_GUIDE.md`
- Modify: `scripts/simulation/README.md`
- Modify: `docs/index.md` only if its existing routing sentence no longer describes `docs/CONFIGURATION.md`

- [x] **Step 1: Document the four top-level config families**

  Add `SimulationConfig` beside model, training, and inference. Include an ownership table showing where `N`, probe transforms, object family, scan geometry, photon count, beamstop, seed, model probe mask, epochs, and optimizer belong.

- [x] **Step 2: Document all supported probe pipelines**

  Clearly distinguish:

  - `pad_extrapolate`: legacy global quadratic phase, including the center;
  - `pad_extrapolate_boundary_matched`: preserved center, C0 boundary-conditioned outer phase;
  - `pad_preserve`, interpolation, smoothing, and composed pipelines.

  State the smoothing order and that changing a pipeline changes dataset identity.

- [x] **Step 3: Add complete grid-lines and dead-leaves examples**

  Show one config file per object family and the same CLI launch shape. Keep training values in their own config section/file.

- [x] **Step 4: Run focused text/parse checks**

  Run:

  ```bash
  python -m pytest tests/test_simulation_config.py tests/scripts/test_simulation_config_cli.py -q
  python - <<'PY'
  from pathlib import Path
  for name in (
      "docs/CONFIGURATION.md",
      "docs/DATA_GENERATION_GUIDE.md",
      "scripts/simulation/README.md",
  ):
      text = Path(name).read_text()
      assert "SimulationConfig" in text
      assert "pad_extrapolate_boundary_matched" in text
  PY
  ```

  Expected: tests and assertions pass.

## Task 7: Final focused verification and study handoff

**Files:**

- Create: `scripts/simulation/render_probe_extension_check.py`
- Create: `tests/scripts/test_render_probe_extension_check.py`

- [x] **Step 1: Run the claim-matched suite**

  ```bash
  python -m pytest \
    tests/test_simulation_config.py \
    tests/test_simulation_probe_transform.py \
    tests/scripts/test_simulation_config_cli.py \
    tests/scripts/test_render_probe_extension_check.py \
    tests/test_grid_lines_workflow.py \
    tests/torch/test_config_bridge.py \
    tests/torch/test_config_factory.py \
    tests/studies/test_torch_ablation_configuration.py \
    tests/studies/test_grid_lines_reference_performance.py -q
  ```

  Expected: all selected tests pass.

- [x] **Step 2: Add a deterministic visual-check renderer**

  Implement a small CLI that accepts the source probe, target size, smoothing value, and output path, then renders the legacy global extension and the new boundary-matched extension from the same prepared source. It must not generate or train on diffraction data.

  The PNG must be large enough for direct inspection and contain, at minimum:

  - source/prepared 64×64 amplitude and wrapped phase;
  - legacy global-quadratic 128×128 amplitude and wrapped phase;
  - boundary-matched 128×128 amplitude and wrapped phase;
  - a boundary-matched center-difference panel, `abs(P_new[32:96, 32:96] - P_prepared)`, with its numeric maximum in the title;
  - a wrapped phase-discontinuity map across the four inner seams, with the maximum seam error in the title;
  - an outer-region phase residual `wrap(phi_new - q)`, with the inner and outer square boundaries overlaid;
  - horizontal and vertical unwrapped-phase profiles through the probe center, with vertical markers at pixels 32 and 95, comparing source, global extension, boundary-matched extension, and fitted quadratic;
  - a short embedded annotation containing the canonical pipeline, source and output hashes, solver tolerance, and measured Laplacian residual.

  Use one shared amplitude scale and one shared cyclic phase scale across comparable panels. Do not use per-panel autoscaling that can hide a seam. Mark the preserved 64×64 footprint on every 128×128 image.

- [x] **Step 3: Test the renderer contract**

  Run the renderer on a deterministic fixture, assert that the PNG and a machine-readable sidecar JSON are written, and verify that the sidecar contains every displayed numeric check. Keep image-content assertions structural; the numerical transform tests remain the source of truth.

  Run: `python -m pytest tests/scripts/test_render_probe_extension_check.py -q`

  Expected: the renderer produces both artifacts with stable dimensions, panel labels, and matching sidecar values.

- [x] **Step 4: Perform one sealed-input computation, not a training sweep**

  Transform the known Run1084 64×64 source to 128×128 using the new pipeline and run the visual-check renderer. Write the primary outputs under an ignored artifact root and copy the final inspection pair to:

  ```text
  tmp/probe_extension_boundary_matched_check.png
  tmp/probe_extension_boundary_matched_check.json
  ```

  Confirm numerically that the central 64×64 complex values match the prepared source and all solver/seam tolerances pass before presenting the PNG to the user.

  The user-facing visual acceptance is:

  - the boundary-matched center is visually identical to the prepared source;
  - no phase jump is visible on any of the four inner borders or in the two center profiles;
  - the outer phase transitions smoothly toward the fitted quadratic;
  - the amplitude extension has no unintended zero ring, crop, shift, or asymmetric border;
  - the displayed numeric maxima agree with the JSON sidecar and the focused numerical tests.

  Visual inspection supplements these numerical gates; it cannot waive a failed equality, seam, or solver-residual assertion.

- [x] **Step 5: Prepare, but do not silently substitute, the study recipe**

  Create a new dataset recipe using the top-level simulation config and a new output identity. Any later model comparison must regenerate the dataset first and explicitly select that new identity. Existing `pad_extrapolate` results remain valid only for their recorded legacy/global transform.

## Completion criteria

- The canonical public configuration API has one validated top-level `SimulationConfig` covering probe construction, synthetic object generation, scan geometry, detector/noise, and seed.
- `smooth:0.5|pad_extrapolate_boundary_matched:128` preserves the prepared center exactly, is phase-continuous at the source boundary within declared tolerance, and relaxes to the fitted quadratic at the target boundary.
- The Run1084 visual-check PNG and JSON sidecar are generated under `tmp/`, expose all four inner seams and center profiles, and pass both the stated numerical gates and user visual inspection.
- Existing `pad_extrapolate` numerical output and historical study interpretation are unchanged.
- Grid-lines, generic simulation scripts, Torch `DatagenConfig`, and ablation study readers resolve through the same simulation ownership model while retaining read compatibility.
- New generated datasets carry a stable simulation-config digest and complete probe lineage, preventing accidental reuse across transform contracts.
- No model training run, threshold change, or historical artifact rewrite is required to complete this implementation.
