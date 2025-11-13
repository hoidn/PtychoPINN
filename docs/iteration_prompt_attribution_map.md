# Prompt Rule → Code Implementation Attribution Map

**Analysis Date:** 2025-11-12
**Prompt Version:** vNext (supervisor.md, main.md)
**Iteration Range:** 289–312
**Methodology:** Read-only analysis; cross-reference prompt rules with code changes

---

## Purpose

This document maps **prompt engineering rules** (from `prompts/supervisor.md` and `prompts/main.md`) to **concrete code changes** in the analyzed iteration window. It answers: *"When a new prompt rule was introduced, what was the first observable implementation or compliance artifact?"*

---

## Prompt Rule Inventory (vNext)

### Category A: Process Discipline

#### Rule A1: Dwell Enforcement (Three-Tier Policy)
**Source:** `prompts/supervisor.md:63-66`
**Text:**
```
- Tier 1 (dwell=2): must hand off runnable production task or switch focus
- Tier 2 (dwell=4): if Ralph did not execute, document blocker, create blocker focus, switch
- Tier 3 (dwell=6): absolute limit — force-block, escalate, mandatory switch
```

**First Implementation Evidence:**
- **Iteration:** 293
- **Artifact:** `docs/fix_plan.md` (escalation notes for STUDY-SYNTH-FLY64)
- **Anchor:** `docs/fix_plan.md:43-48` (dwell escalation report references)
- **Observation:** Supervisor logs show dwell counter incrementing from iter 291→303; Tier 3 triggered around iter 303 (blocker logged)

---

#### Rule A2: Environment Freeze (Hard Constraint)
**Source:** `prompts/supervisor.md:68`
**Text:**
```
Do not propose/execute environment changes unless focus is environment maintenance.
```

**Implementation Evidence:**
- **Iteration:** All (289–312)
- **Compliance:** 100% — no `pip install`, `conda install`, or `requirements.txt` changes detected
- **Anchor:** Implicit (absence of package management commits)
- **Note:** One `setup.py` edit in iter 304, but only metadata (not dependencies)

---

#### Rule A3: Evidence-Only Git Exceptions
**Source:** `prompts/supervisor.md:82-86`
**Text:**
```
If every dirty path is under current Reports Hub, skip pull/rebase and log evidence_only_dirty=true.
```

**First Implementation Evidence:**
- **Iteration:** 293
- **Code Change:** `scripts/orchestration/git_bus.py` (conditional rebase logic)
- **Anchor:** `scripts/orchestration/git_bus.py:*` (iter 293)
- **Impact:** Reduced spurious merge conflicts when only evidence files changed

---

### Category B: Implementation Discipline

#### Rule B1: Stall-Autonomy (Implementation Nucleus)
**Source:** `prompts/main.md:66-71`
**Text:**
```
Extract the smallest viable code change from brief; execute nucleus first.
Allowed nucleus surfaces: initiative bin/ scripts, tests/**, scripts/tools/**.
```

**First Implementation Evidence:**
- **Iteration:** 300
- **Code Change:** `scripts/orchestration/git_bus.py` (minimal hygiene function before broader orchestration work)
- **Anchor:** `scripts/orchestration/git_bus.py:*` (iter 300)
- **Pattern:** Subsequent iterations (301, 302) show similar "guard function + minimal test → expand scope" pattern

---

#### Rule B2: Spec Precedence Over Arch
**Source:** `prompts/main.md:7-9`
**Text:**
```
Treat SPEC as normative; use ARCH for implementation detail.
If SPEC and ARCH conflict, prefer SPEC.
```

**Implementation Evidence:**
- **Iteration:** 302 (config bridge), 305 (Ptychodus I/O)
- **Anchors:**
  - `ptycho/config/config.py:*` (CONFIG-001 bridge per spec)
  - `ptycho/io/ptychodus_product_io.py:*` (DATA-001 HDF5 format per spec)
- **Observation:** Both changes cite spec sections in commit messages; ARCH docs updated post-facto

---

#### Rule B3: Refactoring Discipline (Atomic)
**Source:** `prompts/main.md:39-40`
**Text:**
```
If moving/renaming: a) create new, b) move code, c) search repo for old imports, d) update all, e) delete obsolete, f) validate.
```

**Implementation Evidence:**
- **Iteration:** 306
- **Code Change:** Backend selector refactor (`ptycho/workflows/components.py` + CLI updates)
- **Anchor:** `ptycho/workflows/components.py:*`, `scripts/training/train.py:*`, `scripts/inference/inference.py:*`
- **Observation:** No dangling imports detected; atomic commit pattern (old + new coexist, then old removed in next commit)

---

### Category C: Testing & Validation

#### Rule C1: Test Style (Native Pytest)
**Source:** `prompts/main.md:42`
**Text:**
```
Use native pytest; do not mix unittest.TestCase.
```

**Implementation Evidence:**
- **Iteration:** All (289–312)
- **Compliance:** 100% of new tests use `pytest` conventions (`test_*.py` with plain functions or `pytest.mark`)
- **Anchor:** `tests/study/test_dose_overlap_overlap.py:*`, `tests/scripts/test_inference_backend_selector.py:*`
- **Note:** No `unittest.TestCase` subclasses added in analyzed window

---

#### Rule C2: Comprehensive Testing Gate
**Source:** `prompts/main.md:44`
**Text:**
```
Run configured linters/formatters/type-checkers for touched code; resolve new errors before full test run.
```

**Implementation Evidence:**
- **Iteration:** 302, 305 (high-impact changes)
- **Artifact:** `ruff check` logs not directly committed, but GREEN pytest evidence suggests linting passed
- **Anchor:** Implicit (no linter errors in CI logs; all iterations have passing pytest suites)

---

### Category D: Scientific & Technical Hygiene

#### Rule D1: PyTorch Device Discipline
**Source:** `prompts/main.md:46`
**Text:**
```
Keep dtype/device agnostic code; avoid .cpu()/.cuda() in production paths.
```

**Implementation Evidence:**
- **Iteration:** 302 (config bridge), 306 (backend selector)
- **Anchor:** `ptycho/workflows/components.py:*` (backend abstraction)
- **Observation:** No hardcoded `.cuda()` calls in new code; device passed via config

---

#### Rule D2: Deterministic Seeds
**Source:** `prompts/main.md:45`
**Text:**
```
Respect units/dimensions; deterministic seeds; numeric tolerances (atol/rtol).
```

**Implementation Evidence:**
- **Iteration:** 299–301 (overlap metrics)
- **Anchor:** `studies/fly64_dose_overlap/overlap.py:*` (RNG seed parameters added to metrics bundle)
- **Code Example:** `--rng-seed-subsample 456` CLI flag introduced

---

### Category E: Policy Compliance (Findings.md)

#### Policy E1: POLICY-001 (PyTorch Mandatory)
**Source:** `docs/findings.md`, `prompts/main.md`
**Text:**
```
PyTorch (torch ≥ 2.2) is mandatory. PyTorch workflows must run update_legacy_dict before touching legacy modules.
```

**Implementation Evidence:**
- **Iteration:** 302
- **Anchor:** `ptycho/config/config.py:*` (config bridge implementation)
- **Code Pattern:** All PyTorch code paths call `update_legacy_dict(params.cfg, config)` before TF modules

---

#### Policy E2: CONFIG-001 (Config Bridge)
**Source:** `docs/findings.md`
**Text:**
```
update_legacy_dict(params.cfg, config) before touching legacy modules.
```

**Implementation Evidence:**
- **Iteration:** 302
- **Anchor:** `ptycho/workflows/components.py:*` (backend selector)
- **Observation:** Bridge function called in all PyTorch training/inference entry points

---

#### Policy E3: DATA-001 (Ptychodus HDF5 Format)
**Source:** `specs/data_contracts.md`, `docs/findings.md`
**Text:**
```
Ptychodus product files must follow HDF5 schema: /entry/data/diffraction, /entry/instrument/probe, /entry/sample/object.
```

**Implementation Evidence:**
- **Iteration:** 305
- **Anchor:** `ptycho/io/ptychodus_product_io.py:*` (HDF5 exporter)
- **Validation:** Run1084 conversion in iter 305 evidence logs; h5py structure verified

---

#### Policy E4: ACCEPTANCE-001 (Geometry-Aware Bounds)
**Source:** `docs/findings.md` (added during analyzed window)
**Text:**
```
Dense overlap acceptance must respect geometric bounds: (area / (pi * (threshold/2)^2)) / n_positions, capped at ≤0.10.
```

**Implementation Evidence:**
- **Iteration:** 301–302
- **Anchor:** `studies/fly64_dose_overlap/overlap.py:334-555` (acceptance floor logic)
- **Test:** `tests/study/test_dose_overlap_overlap.py::test_generate_overlap_views_dense_acceptance_floor`

---

## Summary Table: Rule → Code Attribution

| Rule ID | Category | Prompt File | First Iter | Code Anchor | Test Anchor |
|---------|----------|-------------|-----------|-------------|-------------|
| A1 | Dwell Enforcement | supervisor.md:63 | 293 | `docs/fix_plan.md:43` | N/A (process-level) |
| A2 | Environment Freeze | supervisor.md:68 | All | Implicit | N/A |
| A3 | Evidence-Only Git | supervisor.md:82 | 293 | `scripts/orchestration/git_bus.py:*` | N/A |
| B1 | Stall-Autonomy | main.md:66 | 300 | `scripts/orchestration/git_bus.py:*` | N/A |
| B2 | Spec Precedence | main.md:7 | 302 | `ptycho/config/config.py:*` | `pytest tests/config/` |
| B3 | Atomic Refactor | main.md:39 | 306 | `ptycho/workflows/components.py:*` | `pytest tests/scripts/test_*_backend_selector.py` |
| C1 | Native Pytest | main.md:42 | All | `tests/**/*.py` | All test files |
| C2 | Testing Gate | main.md:44 | 302+ | Implicit | GREEN pytest logs |
| D1 | Device Agnostic | main.md:46 | 302 | `ptycho/workflows/components.py:*` | N/A |
| D2 | Deterministic Seeds | main.md:45 | 299 | `studies/fly64_dose_overlap/overlap.py:*` | `pytest tests/study/test_dose_overlap_overlap.py` |
| E1 | POLICY-001 (PyTorch) | findings.md | 302 | `ptycho/config/config.py:*` | `pytest tests/config/` |
| E2 | CONFIG-001 (Bridge) | findings.md | 302 | `ptycho/workflows/components.py:*` | `pytest tests/scripts/test_training_backend_selector.py` |
| E3 | DATA-001 (HDF5) | data_contracts.md | 305 | `ptycho/io/ptychodus_product_io.py:*` | `pytest tests/io/test_ptychodus_product_io.py` |
| E4 | ACCEPTANCE-001 | findings.md | 301 | `studies/fly64_dose_overlap/overlap.py:334` | `pytest tests/study/...::test_generate_overlap_views_dense_acceptance_floor` |

---

## Observations

### High Compliance
- **Environment Freeze (A2):** 100% adherence; no unauthorized package changes
- **Native Pytest (C1):** 100% of new tests follow pytest conventions
- **Policy E1-E4:** All implemented with validating tests

### Delayed Implementation
- **Dwell Enforcement (A1):** Rule existed before iter 289, but visible enforcement (escalation notes) appeared in iter 293
- **Evidence-Only Git (A3):** Likely implemented earlier (git hygiene improvements in iter 293)

### Rule Coupling
- Rules B2 (Spec Precedence) and E1–E4 (Policies) are tightly coupled: policy rules are spec-derived, enforced via B2
- Rules D1–D2 (Scientific Hygiene) often implemented together (e.g., iter 302: device discipline + config parity)

---

## Limitations

1. **Pre-Window Rules:** Many rules predated iteration 289; this map only captures first *observable* implementation in analyzed window
2. **Implicit Compliance:** Some rules (e.g., Environment Freeze) show compliance via absence of violations, not explicit code
3. **Process vs. Code Rules:** Dwell enforcement (A1) is supervisor-level; cannot map to production code
4. **Partial Visibility:** Full implementation may span multiple iterations; map shows earliest anchor

---

## Usage Recommendations

### For Prompt Engineers
- **Add traceability markers:** Embed rule IDs (e.g., `# RULE-B1: stall-autonomy nucleus`) in code comments for easier attribution
- **Pre-commit hooks:** Validate policy compliance (E1–E4) via automated checks

### For Process Auditors
- **Automate attribution:** Extend this map via `git blame` on policy-relevant files
- **Track coverage:** Maintain live dashboard of rule → code mappings

### For Developers
- **Consult before major changes:** Check this map to understand which rules govern your module
- **Test new rules:** For each prompt update, identify expected code changes and add to this map

---

**Report Author:** Claude Code Iteration Analysis Agent
**Linked Reports:**
- Full analysis: `docs/iteration_analysis_report.md`
- Executive summary: `docs/iteration_insights_executive.md`
- Data artifact: `docs/iteration_analysis_data.json`
