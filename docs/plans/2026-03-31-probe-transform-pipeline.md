# Probe Transform Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace one-off probe scale modes with a composable probe-transform pipeline that can express contracts such as `smooth:0.5|pad:128|interp:256`, while keeping existing `pad_preserve`, `interpolate`, and `pad_extrapolate` behavior backward compatible.

**Architecture:** Keep the public workflow surface backward compatible by introducing a new canonical probe-transform pipeline representation and treating legacy `probe_scale_mode` + `probe_smoothing_sigma` as a compatibility layer that normalizes into that pipeline. Apply the normalized pipeline in one place inside `ptycho.workflows.grid_lines_workflow`, persist the normalized provenance into dataset metadata, and update the `lines_256` dataset/runbook/docs to use an explicit two-stage contract for the new probe experiment family.

**Tech Stack:** Python 3.11, NumPy, SciPy `ndimage.zoom`, existing `grid_lines_workflow` dataset builder path, pytest.

---

## Design Summary

### Recommended approach

Introduce a new optional config field, `probe_transform_pipeline`, and make it the canonical representation of probe preprocessing.

- Canonical internal form: ordered list of probe-transform steps.
- Canonical persisted form: normalized string plus structured metadata.
- Legacy compatibility: if `probe_transform_pipeline` is unset, derive an equivalent pipeline from `probe_scale_mode` and `probe_smoothing_sigma`.

Example canonical string:

```text
smooth_complex:sigma=0.5|pad_complex:target_N=128|interpolate_complex:target_N=256,order=3,representation=real_imag
```

### Why this approach

- It supports future `N=512` or multi-stage probe preparation without inventing more enums.
- It keeps dataset provenance explicit and reviewable.
- It preserves the existing workflow API until callers are intentionally migrated.
- It separates "how the probe is transformed" from "which study is using it".

### Explicit non-goals

- Do not change the underlying simulation geometry, `set_phi`, scan coordinates, or train/test split semantics.
- Do not silently reinterpret old `probe_scale_mode` values.
- Do not remove legacy fields in this change.

## File Map

- Modify: `ptycho/workflows/grid_lines_workflow.py`
  - Add pipeline parsing/normalization and route probe scaling through it.
- Modify: `tests/test_grid_lines_workflow.py`
  - Add parser, normalization, and behavior tests for the new pipeline.
- Modify: `scripts/studies/runbooks/rebuild_lines_256_dataset.py`
  - Preserve canonical `pad_preserve` contract explicitly and allow future custom pipeline support if needed.
- Modify: `scripts/studies/grid_study_dataset_builder.py`
  - Ensure persisted dataset metadata includes the normalized probe-transform provenance.
- Modify: `docs/studies/lines_256_dataset.md`
  - Document canonical `pad_preserve` vs new two-stage pipeline contract.
- Modify: `docs/findings.md`
  - Record the new provenance rule and compatibility boundary.
- Optional follow-up: `scripts/studies/runbooks/` helper for manual dataset variants
  - Only if the first implementation shows repeated manual rebuild friction.

## Task 1: Define the pipeline contract

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Test: `tests/test_grid_lines_workflow.py`

- [ ] **Step 1: Add failing parser/normalization tests**

Cover:
- legacy `pad_preserve` normalization
- legacy `interpolate` normalization
- explicit pipeline string parsing
- invalid operation rejection
- final-size mismatch rejection

Run:

```bash
pytest --collect-only tests/test_grid_lines_workflow.py -q
pytest tests/test_grid_lines_workflow.py -k "probe and pipeline" -v
```

- [ ] **Step 2: Add a structured pipeline representation**

In `grid_lines_workflow.py`, add focused helpers such as:

```python
def parse_probe_transform_pipeline(spec: str) -> list[dict[str, object]]:
    ...

def normalize_probe_transform_pipeline(
    *,
    target_N: int,
    probe_shape: tuple[int, int],
    probe_scale_mode: str | None,
    probe_smoothing_sigma: float | None,
    probe_transform_pipeline: str | None,
) -> tuple[str, list[dict[str, object]]]:
    ...
```

- [ ] **Step 3: Define the canonical compatibility mapping**

Map old behavior into normalized pipelines:

- `interpolate` -> `interpolate_complex:target_N=<N>,order=3,representation=real_imag` plus optional leading `smooth_complex`
- `pad_preserve` -> optional `smooth_complex` then `pad_complex:target_N=<N>`
- `pad_extrapolate` -> `pad_extrapolate_complex:target_N=<N>` plus optional trailing `smooth_complex`

- [ ] **Step 4: Run the focused parser tests**

Run:

```bash
pytest tests/test_grid_lines_workflow.py -k "pipeline or pad_preserve or interpolate" -v
```

- [ ] **Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat: add probe transform pipeline contract"
```

## Task 2: Route probe scaling through the pipeline

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Test: `tests/test_grid_lines_workflow.py`

- [ ] **Step 1: Add failing behavior tests for actual execution**

Cover:
- `smooth -> pad:128 -> interp:256` returns a `256x256` probe
- direct legacy `pad_preserve` path still matches current behavior
- direct legacy `interpolate` path still matches current behavior
- two-stage pipeline does not numerically equal direct `64 -> 256` interpolate

- [ ] **Step 2: Implement `apply_probe_transform_pipeline()`**

Use the existing low-level operations:
- `smooth_complex_array(...)`
- `_pad_complex_probe(...)`
- `interpolate_array(...)`
- existing pad-extrapolate logic

Keep the actual transform implementation in one place.

- [ ] **Step 3: Make `scale_probe()`/`scale_probe_with_mode()` compatibility shims**

Ensure existing callers still work while the normalized pipeline becomes the real execution path.

- [ ] **Step 4: Run focused unit tests**

Run:

```bash
pytest tests/test_grid_lines_workflow.py -k "scale_probe" -v
```

- [ ] **Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py tests/test_grid_lines_workflow.py
git commit -m "feat: execute probe transforms through pipeline"
```

## Task 3: Persist normalized provenance in datasets

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `scripts/studies/grid_study_dataset_builder.py`
- Test: `tests/test_grid_lines_workflow.py`

- [ ] **Step 1: Add failing metadata tests**

Cover persisted metadata for:
- legacy canonical `lines_256` dataset
- explicit two-stage pipeline dataset

Assertions should check behavior/provenance, not prompt text.

- [ ] **Step 2: Persist normalized pipeline metadata**

Ensure saved metadata includes:
- `probe_transform_pipeline`
- `probe_transform_steps`
- `probe_scale_mode_legacy` when applicable
- `probe_smoothing_sigma_legacy` when applicable

If the metadata layer only supports flat fields, serialize the structured steps as JSON.

- [ ] **Step 3: Keep canonical `lines_256` metadata stable**

The current canonical dataset should still clearly say it is `pad_preserve`-equivalent, even if the internal implementation now routes through a normalized pipeline.

- [ ] **Step 4: Run metadata tests**

Run:

```bash
pytest tests/test_grid_lines_workflow.py -k "metadata and probe" -v
```

- [ ] **Step 5: Commit**

```bash
git add ptycho/workflows/grid_lines_workflow.py scripts/studies/grid_study_dataset_builder.py tests/test_grid_lines_workflow.py
git commit -m "feat: persist normalized probe transform provenance"
```

## Task 4: Add the two-stage `lines_256` dataset path

**Files:**
- Modify: `scripts/studies/runbooks/rebuild_lines_256_dataset.py`
- Modify: `docs/studies/lines_256_dataset.md`
- Optional modify: new helper under `scripts/studies/runbooks/`
- Test: `tests/studies/test_rebuild_lines_256_dataset.py`

- [ ] **Step 1: Add failing runbook/config tests**

Cover:
- canonical rebuild still uses `pad_preserve`
- a new alternate rebuild path can express `smooth -> pad:128 -> interp:256`
- summary output records the normalized pipeline

- [ ] **Step 2: Add explicit alternate dataset contract support**

Recommended shape:
- keep `build_lines_256_dataset()` canonical and unchanged for the main study
- add a separate helper or optional profile argument for probe-contract variants

Do not silently redefine the canonical dataset.

- [ ] **Step 3: Update the dataset note**

Document three distinct contracts:
- canonical `pad_preserve`
- direct `interpolate`
- new two-stage `pad128_then_interpolate256`-style contract

Explicitly say the two-stage path is a different dataset family requiring a fresh baseline.

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest --collect-only tests/studies/test_rebuild_lines_256_dataset.py -q
pytest tests/studies/test_rebuild_lines_256_dataset.py -v
```

- [ ] **Step 5: Commit**

```bash
git add scripts/studies/runbooks/rebuild_lines_256_dataset.py docs/studies/lines_256_dataset.md tests/studies/test_rebuild_lines_256_dataset.py
git commit -m "feat: add two-stage probe dataset contract for lines256"
```

## Task 5: Record the policy and validate the study path

**Files:**
- Modify: `docs/findings.md`
- Modify: `docs/studies/lines_256_controller_loop.md`
- Test/verify: orchestrator and dataset build smoke commands

- [ ] **Step 1: Add a findings entry**

Record:
- probe preprocessing must be treated as dataset-contract provenance
- any normalized pipeline change invalidates prior baselines/accepted states

- [ ] **Step 2: Update the controller-loop doc**

Clarify that probe preprocessing changes are study-input changes, not architecture changes.

- [ ] **Step 3: Run the required validation**

Run:

```bash
pytest tests/test_grid_lines_workflow.py tests/studies/test_rebuild_lines_256_dataset.py -v
python scripts/studies/runbooks/rebuild_lines_256_dataset.py --help
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
```

- [ ] **Step 4: Manual dataset-contract smoke**

Build one alternate dataset variant and verify:
- train/test NPZs exist
- metadata records the normalized pipeline
- `probeGuess` visualization matches the expected two-stage support

- [ ] **Step 5: Commit**

```bash
git add docs/findings.md docs/studies/lines_256_controller_loop.md
git commit -m "docs: record probe transform pipeline policy"
```

## Verification Commands

```bash
pytest --collect-only tests/test_grid_lines_workflow.py tests/studies/test_rebuild_lines_256_dataset.py -q
pytest tests/test_grid_lines_workflow.py tests/studies/test_rebuild_lines_256_dataset.py -v
python scripts/studies/runbooks/rebuild_lines_256_dataset.py --help
PYTHONPATH=/home/ollie/Documents/agent-orchestration python -m orchestrator run workflows/agent_orchestration/lines_256_session_controller.yaml --dry-run --stream-output
```

## Completion Criteria

- [ ] Probe preprocessing can be expressed as a normalized ordered pipeline rather than only enum modes.
- [ ] Legacy `pad_preserve`, `interpolate`, and `pad_extrapolate` callers still behave compatibly.
- [ ] Dataset metadata persists normalized probe-transform provenance.
- [ ] The `lines_256` docs clearly separate canonical `pad_preserve` from alternate two-stage probe contracts.
- [ ] A manual two-stage `64 -> pad 128 -> interpolate 256` dataset build is reproducible and visibly distinct from both canonical `pad_preserve` and direct `64 -> 256` interpolation.

## Notes for Execution

- Keep the first implementation small: support only the operations already present in the repo (`smooth_complex`, `pad_complex`, `interpolate_complex`, `pad_extrapolate_complex`).
- Prefer flat, explicit metadata over clever abstractions.
- Do not refactor unrelated dataset-builder or controller code in the same change.
- Treat this as a dataset-contract extension, not as a `lines_256` architecture-loop change.
