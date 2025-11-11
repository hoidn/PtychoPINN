# Iteration Analysis Audit — Executive Summary

**Date:** 2025-11-11
**Branch:** feature/torchapi-newprompt
**Analysis Window:** Iterations 262–293 (30 iterations)
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 (Synthetic fly64 dose/overlap study)

---

## Key Findings

### 1. Planning Saturation (87% of iterations)
- **26 out of 30 iterations** were planning/documentation-only loops with no production code changes
- Dwell enforcement (max 2 planning loops per focus) **not observed** until external intervention at i=290
- Implementation floor rule (max 1 docs-only loop) violated for 23 consecutive iterations (i=262–284)

### 2. Critical Blockers Consumed 13 Iterations
- **Workspace mismatch (i=277–281):** Parallel clone confusion (PtychoPINN vs PtychoPINN2) — 5 iterations
- **Git rebase thrashing (i=282–288):** 200+ consecutive rebase failures due to unstaged changes — 7 iterations
- **Resolution:** Workspace guards (`pwd -P` checks) + autostash workflow in git_bus.py

### 3. Implementation Breakthroughs (3 high-impact commits)
**Iteration 285 (score=34):** First productive execution after 23 planning loops
- Launched Phase C→G dense pipeline with `--clobber`
- Phase C dose_1000 dataset fully generated (4.7GB: canonical + simulated + patched splits)

**Iteration 289 (score=44, highest):** Pytest GREEN + full pipeline launch
- Test guard `test_run_phase_g_dense_post_verify_only_executes_chain` PASSED (1/1)
- PYTHONPATH configuration resolved ModuleNotFoundError
- Phase C→G orchestration running in background

**Iteration 292 (score=43):** Dual policy + feature commit
1. **PYTHON-ENV-001 policy** (commit 811f4264): Interpreter discipline established in `docs/DEVELOPER_GUIDE.md:127`; all orchestrators/wrappers updated to use active interpreter
2. **Dose filtering feature** (commit c8eb30bb): Added `--dose`/`--doses` CLI args to `studies/fly64_dose_overlap/generation.py:42`; reduced Phase C runtime from 30min to 5min

### 4. Prompt Rule Adoption (Delayed)
- Major prompt revision at **i=44→45** (2025-10-17) added dwell enforcement + implementation floor
- Evidence of enforcement **not observed until i=290+** (200+ iterations later)
- External intervention (commit 317c360a "unblock execution and stop planning loops") appears to trigger enforcement

---

## Score Summary (0–100 scale)

**Distribution:**
- **0–20 (low):** 10 iterations (33%) — pure planning or git sync
- **21–30 (moderate-low):** 13 iterations (43%) — planning with artifact commits
- **31–40 (moderate):** 5 iterations (17%) — execution prep + partial implementation
- **41–50 (high):** 2 iterations (7%) — full implementation with passing tests

**Statistics:**
- Mean: 24.5, Median: 27.0, Std Dev: 9.8
- Min: 0 (i=263, i=291, i=293 — pure git sync)
- Max: 44 (i=289 — pytest + pipeline launch)

**ASCII Plot:**
```
Score
 50 |
 45 |                           *  *
 40 |
 35 |                       *
 30 |  ** ****** **              *
 25 |           *  **   *     **
 20 |                ***   * *
 15 |*   *               **
 10 | *                           *
  5 |
  0 |                               *
    +--------------------------------
     262          272          282          292
```

---

## Rule → Code Attribution (Selected)

| Prompt Rule | Implementation | Iteration | Impact |
|------------|---------------|-----------|--------|
| PYTHON-ENV-001 | `docs/DEVELOPER_GUIDE.md:127` | i=292 | Eliminated ModuleNotFoundError |
| DATA-001 (allow_pickle) | `studies/fly64_dose_overlap/overlap.py:232` | i=287 | Unblocked Phase D |
| TEST-CLI-001 (pytest before CLI) | `plans/.../green/pytest_post_verify_only.log` | i=285, i=289 | Validated test infra before expensive runs |
| TYPE-PATH-001 (hub-relative paths) | `plans/.../bin/run_phase_g_dense.py:234` | i=270 | Fixed success banner path refs |
| Dwell enforcement | `galph_memory.md` tracking | i=290+ | Ended 23-loop planning saturation |

---

## Recommendations

### Immediate (High Priority)
1. **Strengthen dwell enforcement:** Supervisor should auto-block planning loops after dwell=2 (no manual override exceptions)
2. **Pre-flight workspace checks:** Add `assert_workspace()` to all CLI entry points (prevent PtychoPINN vs PtychoPINN2 confusion)
3. **Default autostash:** Git orchestration should always stash unstaged changes before pull/rebase (prevent 200+ rebase failure cycles)

### Short-term (Medium Priority)
4. **Automate iteration scoring:** Build `scripts/tools/score_iterations.py` with diff/semantic/summary heuristics → CSV output
5. **Dwell compliance dashboard:** Add `scripts/tools/check_dwell_violations.py` to flag violations across all foci
6. **Prompt rule index:** Create `docs/prompt_rule_index.md` with bidirectional map (Rule ↔ Code path:line)

### Long-term (Low Priority)
7. **Pre/post prompt revision study:** Collect pre-i=44 iteration data for statistical comparison (need n≥30 samples per group)
8. **Iteration score real-time dashboard:** Web UI or CLI tool to visualize score trends, blocker clusters, inflection points

---

## Artifacts Delivered

1. **Full audit report:** `docs/iteration_analysis_audit_full.md` (8 sections, 60+ pages)
2. **Extended score CSV:** `docs/iteration_scores_262-293_extended.csv` (30 rows, 9 columns)
3. **Executive summary:** `docs/iteration_audit_executive_summary.md` (this document)

---

## Limitations

- **Small n:** Only 30 iterations analyzed (262–293); pre-i=262 data not examined
- **Autocorrelation:** All iterations clustered on single initiative (STUDY-SYNTH-FLY64); not independent samples
- **No causal inference:** Cannot attribute score changes to prompt revisions without controlled experiment
- **Selection bias:** Analyzed window post-dates prompt revision by 200+ iterations; learning curve effects present
- **Environment Freeze respected:** No packages installed/upgraded; missing tool blockers recorded but not resolved

---

**End of Executive Summary**

For detailed analysis, see: `docs/iteration_analysis_audit_full.md`
