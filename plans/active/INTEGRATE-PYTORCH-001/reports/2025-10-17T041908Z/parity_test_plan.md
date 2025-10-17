# Phase B.B4 Parity Test Expansion Plan

**Initiative:** INTEGRATE-PYTORCH-001  
**Phase:** B.B4 — Extend configuration parity tests beyond MVP 9-field contract  
**Timestamp:** 2025-10-17T041908Z  
**Owner (next loop):** Ralph (engineer loop executing prompts/main.md)

## Context
- Stakeholder delta: Configuration schema divergence (Delta 1) identified by INTEGRATE-PYTORCH-000 remains the top blocker to PyTorch backend adoption. MVP bridge now green after Attempt #11, but only nine fields are verified.
- Objective: Author a reusable, parameterized pytest matrix that exercises **every spec-required field** (ModelConfig, TrainingConfig, InferenceConfig) plus KEY_MAPPINGS and default-behaviour nuances so future config additions fail fast.
- Guardrails:
  - Respect CONFIG-001 finding — every test must initialize `params.cfg` via `update_legacy_dict()`.
  - Preserve PyTorch test auto-skip behaviour (`tests/conftest.py` → `pytest.importorskip('torch')`). When skip triggers, capture the red/green intent in artifacts.
  - Avoid mutating core bridge logic yet; focus on failing/xfailing tests that illuminate deltas. Any implementation work remains in subsequent loops.
- Authoritative references: `specs/ptychodus_api_spec.md §5.1-5.3`, `ptycho/config/config.py:70-180`, `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md`, `tests/torch/test_config_bridge.py` (current MVP test), and KEY_MAPPINGS in `ptycho/config/config.py:231-252`.

---

### Phase A — Field Matrix Consolidation
Goal: Translate the narrative mapping in `config_schema_map.md` into a machine-consumable schema that the test suite can iterate over.
Prereqs: Review stakeholder brief Delta 1 + Attempt #11 resolution notes to understand current adapter capabilities and known gaps.
Exit Criteria: Canonical table saved to `reports/2025-10-17T041908Z/field_matrix.md` (or `.csv`) enumerating each field with source, target, default handling, and test expectation flags (direct/transform/override/skip).

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Derive canonical TensorFlow baseline | [ ] | Instantiate `ModelConfig`, `TrainingConfig`, `InferenceConfig` with explicit non-default values (covering every field) in a scratch script under `reports/2025-10-17T041908Z/fixtures.py`. Record chosen values + rationale in `field_matrix.md` referencing spec rows. |
| A2 | Annotate PyTorch → TF transformations | [ ] | For each field, classify as `direct`, `transform`, `override_required`, `unsupported`. Source values from `config_schema_map.md` and double-check KEY_MAPPINGS. |
| A3 | Flag default divergence + skip policy | [ ] | Document where PyTorch defaults differ from spec (e.g., `probe_scale`, `nphotons`). Specify expected behaviour (override vs accept PyTorch default) and whether to assert equality or allow tolerance. |

---

### Phase B — Dataclass Translation Tests
Goal: Build parameterized pytest cases that validate the adapter returns TensorFlow dataclasses with correct values before touching `params.cfg`.
Prereqs: Completed field matrix with explicit expected values.
Exit Criteria: New test module (or extension of `tests/torch/test_config_bridge.py`) containing parameterized test(s) covering all `direct` and `transform` fields, failing (xfail allowed) where implementation is missing.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Design pytest parameter sets | [ ] | Translate `field_matrix` rows into `pytest.param` structures grouped by config (model/training/inference). Store generator helper in `reports/2025-10-17T041908Z/testcase_design.md` with mapping back to spec lines. |
| B2 | Author failing dataclass assertions | [ ] | Implement `test_all_model_fields_translate`, `test_all_training_fields_translate`, `test_all_inference_fields_translate`. Each should: instantiate PyTorch configs, call adapter, and assert dataclass attributes equal expected values. Use `pytest.mark.parametrize` with skipif torch missing. |
| B3 | Encode known gaps as xfail | [ ] | For fields not yet implemented (e.g., `probe_mask`, `gaussian_smoothing_sigma` overrides), wrap individual `pytest.param(..., marks=pytest.mark.xfail(reason="…", strict=True))` so TDD signal is explicit. |

Expected test selector (red phase):
```
pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_all_model_fields_translate -v
```
Record red output in `reports/2025-10-17T041908Z/pytest_red.log`.

---

### Phase C — params.cfg Parity & KEY_MAPPINGS Validation
Goal: Ensure `update_legacy_dict()` produces the exact legacy dictionary expected with TensorFlow dataclasses, covering KEY_MAPPINGS and path conversions.
Prereqs: Dataclass translation tests in place (even if red/xfail).
Exit Criteria: Parameterized assertions that compare `params.cfg` snapshots from adapter-driven configs against a canonical TF-only baseline.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Capture TF baseline snapshot | [ ] | Write helper `capture_params(cfg_source)` that clears `params.cfg`, runs `update_legacy_dict`, and returns a sorted dict. Baseline uses pure TensorFlow configs with same values as field matrix. Store baseline dump in `reports/2025-10-17T041908Z/baseline_params.json`. |
| C2 | Compare adapter vs baseline | [ ] | Implement `test_params_cfg_matches_baseline` asserting equality for every key flagged `direct`/`transform`. Use `dict.items()` diff to produce helpful assertion messages. |
| C3 | Assert override-required warnings | [ ] | For each `override_required` field (paths, `n_groups`, etc.), assert tests raise/xfail with actionable message when overrides missing, and pass when provided. Document outcomes in `reports/2025-10-17T041908Z/override_matrix.md`. |

---

### Phase D — Reporting & Exit Criteria Validation
Goal: Summarize red/green status, maintain ledger alignment, and define hand-off for implementation loop that will make the tests pass.
Prereqs: Phases A-C executed (tests may still fail/xfail).
Exit Criteria: Summary note saved to `reports/2025-10-17T041908Z/summary.md` detailing failing fields, priority order for fixes, and any spec misalignments uncovered. docs/fix_plan.md Attempt updated with artifact links.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Summarize test outcomes | [ ] | Capture per-field pass/xfail/fail status (table) and reference pytest log. |
| D2 | Flag spec or plan updates | [ ] | If new divergences discovered, note required spec amendments and tag INTEGRATE-PYTORCH-000 governance plan. |
| D3 | Update ledger & steering | [ ] | Ensure Attempt entry mentions planned selectors, artifact directory, and instructions for implementation loop to flip tests green. |

---

## Verification Checklist
- [ ] `field_matrix.md` catalogues every spec-required field with expected handling
- [ ] New/updated pytest cases committed (red/xfail) with selectors recorded
- [ ] `baseline_params.json` captured for TF-only reference
- [ ] pytest red run logged to `pytest_red.log`
- [ ] Summary report enumerates failures + next implementation steps

## References
- specs/ptychodus_api_spec.md:213-291 — Field definitions & contract expectations
- ptycho/config/config.py:70-252 — Authoritative dataclasses + KEY_MAPPINGS
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T032218Z/config_schema_map.md — Prior mapping analysis
- tests/torch/test_config_bridge.py — Existing MVP test harness (extend, don’t replace)
- docs/TESTING_GUIDE.md — TDD methodology + pytest usage expectations
- docs/debugging/QUICK_REFERENCE_PARAMS.md — CONFIG-001 guardrail for params.cfg initialization

## Notes for Execution Loop
- Maintain skip/xfail hygiene so CI without PyTorch remains green while documenting missing functionality.
- Prefer pure-Python fixtures over writing new helper modules inside `ptycho_torch/`; keep tests self-contained.
- When choosing explicit values, avoid defaults to ensure bridge copies actual values instead of falling back silently.
- Capture any newly discovered transformation requirements back into `config_bridge_debug.md` if they impact Phase B.B3 logic.
