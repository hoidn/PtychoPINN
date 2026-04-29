# WaveBench Inverse Source Preflight Implementation Plan

> **For agentic workers:** This is a preflight plan. Do not implement the full
> WaveBench benchmark, do not run long training, and do not change the
> manuscript roadmap until this plan has established dataset, baseline, and
> forward-model feasibility. WaveBench is an additional candidate evidence
> lane alongside CDI and CNS.

**Goal:** Decide whether WaveBench inverse source reconstruction is ready to
become a secondary SRU-Net manuscript benchmark, and whether it can support
supervised-only, physics-informed, or hybrid supervised-plus-physics training.

**Architecture:** Treat this as a launch-readiness and evidence-gathering item.
The durable output is a checked-in preflight summary plus machine-readable
metadata describing selected dataset variant, tensor shapes, native baselines,
forward-model reproduction status, and recommended follow-up backlog items.

**Primary design input:**
`docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md`

---

## Selected Objective

- Implement backlog item
  `2026-04-29-wavebench-inverse-source-preflight`.
- Verify whether WaveBench inverse source reconstruction is a practical
  controlled 2D inverse-wave benchmark for SRU-Net, hybrid-spectral, FNO, FFNO,
  and U-Net comparisons.
- Separate the supervised benchmark readiness decision from the
  physics-informed readiness decision.

## Explicit Non-Goals

- Do not train the full model roster.
- Do not download or stage more WaveBench data than needed for metadata
  inspection, loader smoke tests, and forward-model checks.
- Do not alter the CDI or PDEBench CNS manuscript lanes.
- Do not create manuscript tables from preflight outputs.
- Do not claim physics-informed training is valid until the forward-model
  reproduction check passes.
- Do not describe WaveBench inverse source as geology or full waveform
  inversion.

## Binding Design Decisions

- The candidate benchmark is WaveBench inverse source reconstruction, not
  OpenFWI and not OpenSWI.
- The target is a 2D initial pressure/source field `q_0`.
- The observed input is boundary pressure over time.
- Native WaveBench FNO/U-Net baselines remain reference rows.
- Shared-encoder U-Net, SRU-Net, hybrid-spectral, FNO, and FFNO rows are the
  fair internal architecture comparison if the benchmark proceeds.
- Physics-informed variants require a differentiable forward map:

```math
F_c(q_0)=\mathcal{M}u(q_0;c)
```

  and a reproduction check:

```math
F_c(q_0) \approx y.
```

## Artifact Targets

Create:

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight.md`
- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/preflight_metadata.json`

Optional, if a small local smoke is run:

- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/loader_smoke/`
- `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/forward_model_smoke/`

The preflight summary must include:

- selected or rejected inverse-source dataset variant;
- exact observed-data tensor shape;
- exact target `q_0` tensor shape;
- wavespeed variant metadata;
- dataset source and local staging path, if staged;
- native FNO/U-Net checkpoint availability;
- baseline reproduction command feasibility;
- differentiable forward-model location or absence;
- reproduction-check result or blocker;
- recommended next backlog items.

## Task 1: Inspect WaveBench Repository And Dataset Surfaces

- [ ] Identify the current upstream WaveBench repository revision to use for
  preflight.
- [ ] Inspect README, notebooks, package modules, dataset names, and config
  surfaces for inverse source reconstruction.
- [ ] Identify all inverse-source dataset variants and record what distinguishes
  them, especially fixed wavespeed variants and OOD/variant split semantics.
- [ ] Determine whether datasets are available from Zenodo, SwitchDrive, or
  another stable source without manual credentials.
- [ ] Determine whether `.beton` loading via FFCV is required or whether a
  simpler inspection path exists.
- [ ] Record setup risks: dependency versions, CUDA requirements, FFCV
  constraints, PyTorch Lightning version expectations, and checkpoint-loading
  assumptions.

**Acceptance checks:**

- [ ] `preflight_metadata.json` records repository URL, revision/commit if
  cloned, dataset source URLs, and available inverse-source variant names.
- [ ] The checked-in preflight summary explains whether local setup is
  straightforward or likely to need a dedicated environment.

## Task 2: Inspect Tensor Shapes And Select First Variant

- [ ] Stage the minimum dataset subset needed to inspect one inverse-source
  variant, or document why staging is blocked.
- [ ] Load one train sample and one validation/test sample through the intended
  WaveBench loader.
- [ ] Record exact shapes, dtype, value ranges, and normalization conventions
  for:
  - observed boundary-time data `y`;
  - target initial pressure/source field `q_0`;
  - wavespeed field `c(x)`, if exposed;
  - boundary/receiver index metadata, if exposed.
- [ ] Decide the first runnable variant for later supervised benchmarking.
- [ ] Record whether `q_0` is already `128 x 128`, needs resizing, or requires
  a target-resolution decision.

**Acceptance checks:**

- [ ] Preflight summary names the first selected variant or records a clear
  blocker.
- [ ] `preflight_metadata.json` includes machine-readable tensor shape and
  normalization fields.

## Task 3: Verify Native Baseline Availability

- [ ] Locate native WaveBench FNO and U-Net baseline checkpoints, if available.
- [ ] Determine whether the provided checkpoints include the selected
  inverse-source variant.
- [ ] Identify the exact load/evaluation path for `LitModel.load_from_checkpoint`
  or its current equivalent.
- [ ] Run a minimal checkpoint-load smoke if a compatible checkpoint is locally
  available and dependencies are satisfied.
- [ ] If no checkpoint is available for the selected variant, identify the
  shortest native-baseline reproduction path from WaveBench configs/notebooks.

**Acceptance checks:**

- [ ] Preflight summary distinguishes `checkpoint_reusable`,
  `retrain_required`, and `not_available` for native FNO and native U-Net.
- [ ] Any checkpoint-load smoke output is archived under the preflight artifact
  root.

## Task 4: Evaluate Shared-Encoder Benchmark Feasibility

- [ ] Confirm whether boundary-time data can be represented as a stable 2D
  measurement image `y(t,b)`.
- [ ] Decide whether the first shared encoder should target
  `128 x 128 x C` directly or preserve native target resolution first.
- [ ] Identify the model families already available in this repo that can
  consume a `128 x 128 x C` tensor:
  - U-Net;
  - SRU-Net / hybrid-resnet;
  - hybrid-spectral / spectral-resnet;
  - FNO;
  - FFNO.
- [ ] Record missing adapter or output-head work needed before supervised rows
  can run.
- [ ] Decide whether the first follow-up should benchmark `C=32`, `C=64`, or
  both.

**Acceptance checks:**

- [ ] Preflight summary states whether supervised shared-encoder benchmarking is
  ready for a concrete implementation plan.
- [ ] Missing model/adapter work is listed as scoped follow-up work, not hidden
  inside the preflight.

## Task 5: Check Physics-Informed Forward-Model Feasibility

- [ ] Locate WaveBench forward-model generation code for inverse source
  variants, if present in the repository or linked materials.
- [ ] Determine whether the solver is differentiable in PyTorch, generated
  offline in MATLAB/NumPy, or not directly exposed.
- [ ] Record the exact wave equation convention, grid spacing, time step,
  boundary conditions, fixed wavespeed, measurement locations, and initial
  condition convention if available.
- [ ] If a differentiable or easily ported solver is available, run the smallest
  possible reproduction check:

```math
F_c(q_0) \approx y
```

  on one or a few ground-truth examples.
- [ ] If exact reproduction is not possible, decide whether approximate-model
  regularization is still worth a later design.

**Acceptance checks:**

- [ ] Preflight summary classifies physics readiness as one of:
  - `exact_physics_loop_ready`;
  - `approximate_physics_regularization_possible`;
  - `physics_loop_deferred`;
  - `physics_loop_not_recommended`.
- [ ] Any reproduction smoke reports residual magnitude, normalization, and
  known mismatch sources.

## Task 6: Write Follow-Up Recommendations

- [ ] Write the durable preflight summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_preflight.md`.
- [ ] Include a clear final recommendation:
  - `ready_for_supervised_plan`;
  - `ready_for_supervised_and_physics_plan`;
  - `needs_dataset_or_checkpoint_decision`;
  - `not_suitable_for_current_manuscript`.
- [ ] Draft proposed follow-up backlog items only as text inside the summary
  unless the user explicitly asks to create them:
  - supervised WaveBench shared-encoder benchmark;
  - WaveBench native-baseline reproduction;
  - WaveBench forward-model reproduction and physics-loss integration;
  - paper figure/table bundle.
- [ ] Record whether this should trigger a roadmap addition. Default answer:
  no roadmap change until preflight has proven readiness.

**Acceptance checks:**

- [ ] Summary and `preflight_metadata.json` agree on final status and selected
  variant.
- [ ] The backlog item can close without training any full benchmark rows.

## Required Verification

Run the backlog item's declared check command:

```bash
python - <<'PY'
from pathlib import Path
required = [
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/wavebench_inverse_source_benchmark_design.md"),
    Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-wavebench-inverse-source-preflight/execution_plan.md"),
]
missing = [str(p) for p in required if not p.exists()]
if missing:
    raise SystemExit(f"missing WaveBench preflight inputs: {missing}")
print("wavebench preflight inputs present")
PY
```

If code or loader smoke tests are added during implementation, also run the
narrowest relevant import/compile/test selectors and archive outputs under the
preflight artifact root.
