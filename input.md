Mode: Planning
Focus: PARALLEL-API-INFERENCE — Phase A: Exploration and TF helper extraction planning
Branch: feature/torchapi-newprompt-2
Mapped tests: none — exploration/planning only
Artifacts: plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T010000Z/

## Summary

Explore the TensorFlow inference code to understand refactoring scope for Task 1: Extract TF inference helper from `scripts/inference/inference.py`.

## Context

**Initiative Goal:** Provide a single Python entry point that exercises the Ptychodus backend selector with both TensorFlow and PyTorch backends (programmatic training + inference) without relying on the CLI wrappers.

**Task 1 Target:** Extract the reusable TF inference logic from `scripts/inference/inference.py` into a callable function that accepts `RawData`, model bundle path, and configuration — enabling programmatic callers to reuse it without CLI glue.

**Key Functions in inference.py:**
- `perform_inference()` (lines 321-428): Core inference logic
- `save_comparison_plot()` (lines 430-489): Output visualization
- `save_reconstruction_images()` (lines 490-531): Standalone image saving
- `setup_inference_configuration()` (lines 178-220): Config setup from args
- `interpret_sampling_parameters()` (lines 124-177): Sampling logic

## Do Now

### A1: Read and analyze `scripts/inference/inference.py`

**Goal:** Document the function signatures, dependencies, and refactoring approach for extraction.

Focus on:
1. `perform_inference()` dependencies — what imports, configs, and state does it need?
2. Output artifacts — what does it return vs. what does it save to disk?
3. CLI-specific vs. reusable code — identify pure inference logic
4. Integration points with `ptycho.workflows.backend_selector`

### A2: Check existing PyTorch inference helper

Read `ptycho_torch/inference.py` to understand the existing PyTorch inference API. Document similarities/differences with TF path.

### A3: Document extraction design

Write a brief design document to `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T010000Z/extraction_design.md`:
1. Proposed function signature for extracted TF helper
2. Dependencies to be passed as parameters (not global state)
3. Return value specification
4. CLI wrapper changes needed

## How-To Map

```bash
ARTIFACTS=plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T010000Z
mkdir -p "$ARTIFACTS"

# Read the core inference code
# Focus on perform_inference() and its callers

# Read PyTorch inference for comparison
# ptycho_torch/inference.py

# Write extraction design to $ARTIFACTS/extraction_design.md
```

## Pitfalls To Avoid

1. **DO NOT** modify code in this exploration phase — analysis only
2. **DO** document all global state dependencies (params.cfg, logging, etc.)
3. **DO NOT** assume CLI and programmatic paths are identical
4. **DO** reference POLICY-001 (PyTorch mandatory) for backend considerations
5. **Environment Freeze:** Do not install packages

## If Blocked

1. If inference.py has complex dependencies: document them and propose incremental extraction
2. Record blocker in `$ARTIFACTS/blocked_<timestamp>.md`

## Findings Applied

- **CONFIG-001:** Note where `update_legacy_dict()` is called in inference path
- **POLICY-001:** Both backends must be supported
- **DATA-001:** Note data contract compliance requirements

## Pointers

- TF inference script: `scripts/inference/inference.py` (737 lines)
- PyTorch inference: `ptycho_torch/inference.py`
- Backend selector: `ptycho/workflows/backend_selector.py`
- Initiative plan: `plans/active/PARALLEL-API-INFERENCE/plan.md`
- Ptychodus API spec: `specs/ptychodus_api_spec.md`

## Exit Criteria

1. Read and understand TF inference logic
2. Read and understand PyTorch inference for comparison
3. Design document written with proposed extraction approach
4. No code changes — exploration only
