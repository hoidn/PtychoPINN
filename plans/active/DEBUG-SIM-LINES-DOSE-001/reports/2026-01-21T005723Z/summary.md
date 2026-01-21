# D4e Investigation Summary — normalize_data() vs intensity_scale

## Date: 2026-01-21T01:07Z

## Result: D4e REJECTED — Wrong Scope

### Problem Statement
D4e was scoped to "align normalize_data() with dataset-derived scaling" by changing the normalization formula from `sqrt((N/2)² / batch_mean_sum)` to `sqrt(nphotons / batch_mean_sum)`.

### What Was Attempted
1. Modified `ptycho/raw_data.py::normalize_data()` to use the spec formula
2. Modified `ptycho/loader.py::normalize_data()` to match
3. Ran gs2_ideal scenario to verify

### Result: IMMEDIATE NaN COLLAPSE
- Training loss went to NaN by step 5 of epoch 1
- No useful reconstructions produced
- Reverted all changes

### Root Cause Analysis

**The two functions serve DIFFERENT purposes in a two-stage normalization pipeline:**

| Function | Formula | Purpose |
|----------|---------|---------|
| `normalize_data()` | `sqrt((N/2)² / batch_mean_sum)` | L2 normalization to fixed target `(N/2)²` |
| `intensity_scale` | `sqrt(nphotons / batch_mean_sum)` | Scales for Poisson NLL loss |

**Key insight:** The spec formula in `specs/spec-ptycho-core.md §Normalization Invariants` describes **intensity_scale**, NOT normalize_data().

The pipeline works as:
1. `normalize_data()` — ensures consistent model input scale regardless of photon count
2. `intensity_scale` — applies nphotons-dependent scaling for the loss function

Changing both to use the same nphotons-dependent formula **doubles** the scaling and causes numeric instability.

### Evidence
- Telemetry captured in `gs2_ideal/intensity_stats.json` before NaN
- NaN training log in `logs/gs2_ideal_runner.log`
- Post-revert tests: 7/7 passed (`logs/pytest_combined.log`)

### Next Actions
The ~6.6× amplitude bias is NOT caused by normalize_data() using a different formula. Investigate:
- Model architecture wiring
- Loss function scaling
- Interaction between two-stage normalization
- Training dynamics (gradient scaling, etc.)

### Files Modified (then reverted)
- `ptycho/raw_data.py` — normalize_data() docstring updated (kept)
- `ptycho/loader.py` — normalize_data() docstring updated (kept)
- `tests/test_loader_normalization.py` — tests rewritten for L2 normalization (kept)
