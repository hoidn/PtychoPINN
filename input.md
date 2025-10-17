Summary: Capture Phase A baselines for VECTOR-TRICUBIC by recording acceptance tests and microbenchmarks before any vectorization work.
Mode: Perf
Focus: [VECTOR-TRICUBIC-001] Vectorize tricubic interpolation & detector absorption (nanoBragg PyTorch)
Branch: feature/torchapi
Mapped tests: pytest tests/test_at_str_002.py -vv; pytest tests/test_at_abs_001.py -vv
Artifacts: plans/active/VECTOR-TRICUBIC/reports/2025-10-17T003424Z/{summary.md,pytest_baseline.log,perf.json}
Do Now: Run Phase A tasks A1–A3 from plans/active/vectorization.md (baseline pytest selectors + tricubic microbenchmark) from the nanoBragg repo; capture outputs in the artifact directory.
If Blocked: Log the blocker (e.g., missing benchmark script) in summary.md, attach any partial logs, and halt further execution pending supervisor guidance.
Priorities & Rationale:
- plans/active/vectorization.md:18 — Phase A goal demands acceptance + perf baselines before design work.
- plans/active/vectorization.md:25 — Task A1 specifies `pytest tests/test_at_str_002.py -vv` as mandatory evidence.
- plans/active/vectorization.md:27 — Task A3 calls for the tricubic benchmark to quantify current performance.
- docs/fix_plan.md:67 — Fix-plan item requires baseline artifacts under VECTOR-TRICUBIC reports prior to RED tests.
- ../nanoBragg/arch.md:213 — Architecture ADR-06 enforces tricubic fallback semantics we must observe during baseline capture.
How-To Map:
- `cd ../nanoBragg`
- `pytest tests/test_at_str_002.py -vv 2>&1 | tee ../PtychoPINN/plans/active/VECTOR-TRICUBIC/reports/2025-10-17T003424Z/pytest_baseline.log`
- `pytest tests/test_at_abs_001.py -vv 2>&1 | tee -a ../PtychoPINN/plans/active/VECTOR-TRICUBIC/reports/2025-10-17T003424Z/pytest_baseline.log`
- If `scripts/benchmarks/tricubic_baseline.py` exists: `python scripts/benchmarks/tricubic_baseline.py --sizes 256 512 --repeats 5 --outdir ../PtychoPINN/plans/active/VECTOR-TRICUBIC/reports/2025-10-17T003424Z`
- Else run `python -m nanobrag_torch.debug.tricubic_profile --out ../PtychoPINN/plans/active/VECTOR-TRICUBIC/reports/2025-10-17T003424Z/perf.json` (document command + rationale in summary.md).
- Record environment snapshot: `python -m pip freeze | sort > ../PtychoPINN/plans/active/VECTOR-TRICUBIC/reports/2025-10-17T003424Z/requirements.txt`.
- Summarize timings, hardware, pass/skip counts in `summary.md` (same directory).
Pitfalls To Avoid:
- Do not edit production code or tests during this evidence loop.
- Keep artifact paths exactly as specified; no ad-hoc directories.
- Avoid partial pytest runs beyond listed selectors unless directed; capture full outputs even if they pass.
- Ensure GPU usage is documented; if unavailable, note CPU-only baseline explicitly.
- Preserve warning output (especially interpolation fallback) in logs.
- Verify conda/env activation before running benchmarks to prevent inconsistent torch builds.
Pointers:
- plans/active/vectorization.md:23
- plans/active/vectorization.md:37
- docs/fix_plan.md:67
- ../nanoBragg/tests/test_at_str_002.py:1
- ../nanoBragg/arch.md:206
Next Up: Phase B design + RED tests once baselines and summary are committed.
