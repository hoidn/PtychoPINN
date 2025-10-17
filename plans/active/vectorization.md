# Plan — VECTOR-TRICUBIC: Tricubic & Absorption Vectorization

## Context
- Initiative: VECTOR-TRICUBIC (perf initiative covering tricubic interpolation + detector absorption in `nanoBragg` PyTorch backend)
- Phase Goal: Deliver production-ready vectorized kernels with validated speedup and acceptance-test parity.
- Scope: `../nanoBragg/src/nanobrag_torch/models/crystal.py`, `../nanoBragg/src/nanobrag_torch/utils/physics.py`, `../nanoBragg/tests/test_at_str_002.py`, detector absorption loops in `models/detector.py`.
- Dependencies & Key References:
  - `docs/index.md` — documentation map (PtychoPINN side) to respect cross-repo workflow conventions.
  - `docs/DEVELOPER_GUIDE.md` — dual-system architecture + performance guardrails for TensorFlow stack (ensures we do not regress shared patterns when cross-pollinating ideas).
  - `docs/findings.md` — ensure no prior vectorization findings conflict with new plan (none logged).
  - `../nanoBragg/specs/spec-a.md` + `../nanoBragg/arch.md` — normative spec + architecture for tricubic interpolation and detector pipeline.
  - `../nanoBragg/tests/test_at_str_002.py` — acceptance coverage for tricubic interpolation + fallback behaviour.
  - `../nanoBragg/src/nanobrag_torch/models/crystal.py` — current implementation (partial vectorization, still per-ray loops in helpers).
  - `../nanoBragg/src/nanobrag_torch/utils/physics.py` — `polin1/polin2/polin3` kernels that remain loop-bound.
  - `../nanoBragg/tests` runner guidance (pytest-based) — authoritative tests to protect acceptance behaviour.
- Artifact Convention: Allocate reports under `plans/active/VECTOR-TRICUBIC/reports/<ISO8601>/` with `summary.md`, `pytest.log`, `perf.json`, and any plots.

## Phase A — Baseline Characterization
Goal: Establish correctness + performance baselines for tricubic interpolation & detector absorption prior to changes.
Prerqs: Clean repo states for both `PtychoPINN` and `nanoBragg`; confirm `pytest tests/test_at_str_002.py -q` passes; record environment (Python, torch, device).
Exit Criteria: Baseline pytest output + microbenchmark timings captured with documentation of ROI selection.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Capture acceptance baseline | [ ] | Run `pytest tests/test_at_str_002.py -vv` in `../nanoBragg`; store log + collected warnings under `reports/<timestamp>/pytest_baseline.log`. |
| A2 | Record detector absorption coverage | [ ] | Identify canonical test (likely `tests/test_at_abs_001.py`); if absent, note gap for Phase B and mark as risk. |
| A3 | Microbenchmark existing kernels | [ ] | Use `python scripts/benchmarks/tricubic_baseline.py --sizes 256 512 --repeats 5` (if script missing, build quick harness) capturing throughput (ns/sample) + device info. |
| A4 | Summarize findings | [ ] | Draft `summary.md` describing baseline timings, acceptance status, and profiling hotspots (point to loops in `polin2/polin3`). |

## Phase B — Vectorization Design & TDD Red Scaffold
Goal: Define target tensor shapes, batching semantics, and failing tests that encode expected vectorized behaviour (batched HKL queries, multi-pixel evaluation).
Prerqs: Phase A baseline artifacts, spec clarification on interpolation semantics for batched queries, agreement on dtype/device neutrality.
Exit Criteria: Approved design notes + new failing tests (RED) that describe desired vectorized behaviour for tricubic + absorption code paths.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Derive batched interpolation contract | [ ] | Document expected tensor shapes (e.g., `(N_rays, 4)` offsets, `(N_rays, 4,4,4)` sub-cubes); cite spec clause 9.4 (tricubic) and ADR-08 (differentiability). |
| B2 | Author RED tests | [ ] | Add `tests/test_tricubic_vectorized.py` covering batched inputs + default_F fallback; ensure tests fail under current implementation. |
| B3 | Extend absorption tests | [ ] | If lacking, craft targeted test verifying batched absorption path uses broadcasting, with CPU + CUDA parametrization (skip CUDA if unavailable). |
| B4 | Update plan & docs | [ ] | Embed test selectors + expectations in `summary.md`; annotate docs/fix_plan Attempts History with RED status. |

## Phase C — Implementation & GREEN Pass
Goal: Implement minimal vectorized kernels to satisfy RED tests while preserving spec behaviour.
Prerqs: RED tests committed (failing), design approved, benchmarks ready for rerun.
Exit Criteria: Tests pass (`pytest tests/test_tricubic_vectorized.py tests/test_at_str_002.py -vv`), interpolation functions operate on batches without python loops, absorption kernel uses pure tensor math, and differentiability confirmed via gradient check.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Vectorize `polin1/2/3` | [ ] | Replace inner loops with tensor contractions (e.g., `torch.einsum`/`matmul`), ensuring shape `(B, 4)` support and dtype/device parity. |
| C2 | Refactor `_tricubic_interpolation` | [ ] | Accept `(B,)` h/k/l, gather 4×4×4 neighborhoods in a single `torch.gather` call; maintain fallback semantics, update warning gating logic. |
| C3 | Vectorize detector absorption | [ ] | Broadcast incident directions × layer depths, remove `for` loops; respect default fallback for zero thickness. |
| C4 | GREEN validation | [ ] | Run targeted pytest selection + gradient check (e.g., `pytest tests/test_gradients.py -k tricubic` if available; else script `gradcheck_tricubic.py`). Store logs under reports. |

## Phase D — Performance Verification & Refactor Polish
Goal: Quantify speedups, document results, and ensure code quality (naming, comments, docs) meets standards.
Prerqs: Phase C passing tests, instrumentation harness ready.
Exit Criteria: Documented perf gains (CPU + CUDA), optional refactors (cleanup), final artifacts archived, docs updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Re-run microbenchmarks | [ ] | Repeat Phase A harness; compare before/after (ideally ≥3× speedup). Capture JSON + Markdown summary. |
| D2 | Full acceptance sweep | [ ] | Execute `pytest tests/test_at_str_002.py tests/test_at_abs_001.py -vv` and any new vectorized tests; collect logs + coverage notes. |
| D3 | Refactor & doc polish | [ ] | Remove debug prints, add concise comments for tricky tensor reshapes, ensure autodoc strings reflect new broadcasting. |
| D4 | Update knowledge base | [ ] | Log perf results + new conventions in `docs/findings.md` if impactful; update `docs/fix_plan.md` status + attempts. |

## Risks & Mitigations
- Script gaps: Some benchmark/test harnesses may live in `nanoBragg4`; if missing, adapt from historical artifacts or create minimal replacements (document commands).
- Device variance: Ensure CUDA-specific skips guard missing GPU; plan requires CPU baseline at minimum.
- Autograd regression: Validate `requires_grad` pathways using `torch.autograd.gradcheck` on double-precision clones.
- Cross-repo coordination: Track commits across both repositories; update `galph_memory.md` with cross-project links each loop.

## Reporting Expectations
- Each supervisor/engineer loop MUST record artifacts in `plans/active/VECTOR-TRICUBIC/reports/<timestamp>/`.
- `summary.md` should include: baseline metrics, vectorization diffs, decision log, follow-up steps.
- Attach microbenchmark JSON/CSV, pytest logs, and any flamegraphs (if used).

## Exit Criteria (Initiative)
- Vectorized tricubic & absorption kernels merged with passing acceptance + gradient tests.
- Documented ≥2× speedup on CPU and GPU for representative workloads.
- Updated knowledge base entry summarizing approach and reproduction steps.
- docs/fix_plan.md and input.md signal implementation complete and direct follow-on tasks (Phase D3/D4) if any remain.
