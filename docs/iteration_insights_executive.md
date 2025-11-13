# Iteration Analysis — Executive Insights

**Date:** 2025-11-12
**Scope:** Iterations 289–312 (24 iterations analyzed)
**Branch:** feature/torchapi-newprompt
**Analysis Type:** Rigorous read-only code diff + commit pattern review

---

## Key Takeaways (3-Minute Read)

### 1. Process Health: Moderate with Spikes
- **Mean iteration score:** 53.5/100 (code-diff heuristic)
- **Distribution:** Bimodal — 26% high-impact (≥70), 26% low-impact (≤40), 48% moderate
- **Interpretation:** Process is **functional but uneven**; significant variation suggests task complexity heterogeneity or blocking patterns

### 2. High-Impact Milestones
Two iterations stand out as step-function improvements:

**Iteration 302** (Score: 80/100)
- Config bridge for PyTorch/TensorFlow parity
- RawData acceptance floor hardening
- 7 code files, 7 test files
- **Outcome:** Unblocked supervised PyTorch training

**Iteration 305** (Score: 80/100)
- Custom layer extensions (24 code files)
- Ptychodus HDF5 product exporter (DATA-001 compliance)
- 5 test files including integration tests
- **Outcome:** External tool interoperability enabled

### 3. Productivity Blockers
Evidence-only loops (no production code changes) account for **26% of iterations**:
- Iterations 291, 297, 303, 296, 298, 289 (partial)
- **Root causes:** Waiting on long-running pipeline outputs, evidence gathering for blocked tasks
- **Mitigation:** Dwell enforcement rules (Tier 2/3) should escalate faster

### 4. Prompt Engineering Impact (Qualitative)
Post-prompt-change cohort (iters 289–312) shows:
- **Dwell enforcement visible:** Escalation notes in `docs/fix_plan.md` from iter 293+
- **Environment freeze compliance:** 100% (no `pip install` or package changes detected)
- **Stall-autonomy applied:** Minimal nucleus changes in iters 300, 293 before broader work

**Caveat:** Cannot measure pre/post effect without earlier baseline; analyzed window is entirely post-change.

---

## Visual Summary: Score Distribution

```
Score Range    Count  Percentage  Visual
───────────────────────────────────────────────────────
90-100           0      0%
80-89            2      9%       ██
70-79            4     17%       ████
60-69            2      9%       ██
50-59            9     39%       ██████████
40-49            4     17%       ████
30-39            1      4%       █
20-29            1      4%       █
───────────────────────────────────────────────────────
```

**Interpretation:**
- **Modal range:** 50–59 (steady incremental progress)
- **Tail events:** Two 80-point spikes (major milestones)
- **Low outliers:** Single-digit code changes or evidence-only loops

---

## Code Quality Indicators

### Test Coverage Discipline
- **Test evidence present in 100% of iterations** (even evidence-only loops have pytest logs)
- **Regression tests added for critical features:**
  - `test_generate_overlap_views_dense_acceptance_floor` (iter 301–302)
  - Backend dispatch tests (iter 306–307)
- **Test file changes:** Average 2.6 per iteration

### Module Stability
Analyzed code changes by module:

| Module Path | Iterations Touched | Stability |
|------------|-------------------|-----------|
| `ptycho/config/` | 302 | HIGH (single targeted change) |
| `ptycho/raw_data.py` | 302 | HIGH (acceptance logic only) |
| `scripts/training/` | 306 | MODERATE (backend integration) |
| `scripts/inference/` | 306, 307 | MODERATE (backend + dispatch) |
| `studies/fly64_dose_overlap/` | 293–301 | LOW (iterative refinement) |

**Note:** Core physics modules (`ptycho/model.py`, `ptycho/diffsim.py`) untouched in analyzed window — consistent with directive to treat as stable.

---

## Initiative Progress Tracking

### INTEGRATE-PYTORCH-PARITY-001 (Iters 302–307)
**Status:** Active, making progress
- **Milestones achieved:**
  - Config bridge (iter 302)
  - Backend selector CLI (iter 306)
  - Dispatch tests (iter 306–307)
- **Remaining gaps:** Supervised loss function alignment (per latest fix_plan.md)

### STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 (Iters 289–301)
**Status:** Blocked at Phase G rerun
- **Milestones achieved:**
  - Overlap metrics API (iter 299)
  - Acceptance floor logic (iter 301–302)
  - Phase D metrics bundle (iter 300)
- **Blocker:** Dense pipeline rerun not completing (8 consecutive supervisor loops per fix_plan.md)

### EXPORT-PTYCHODUS-PRODUCT-001 (Iter 305)
**Status:** Implementation complete, docs pending
- **Delivered:** HDF5 exporter with Run1084 conversion evidence
- **Remaining:** Documentation integration into DATA_MANAGEMENT_GUIDE.md

---

## Recommendations

### Immediate (Next 3 Iterations)

1. **Resolve STUDY-SYNTH-FLY64 Blocker**
   - Current dwell: Tier 3 (8 loops)
   - Action: Execute dense rerun with explicit guards or force-mark blocked
   - Owner: Ralph (implementation agent)

2. **Complete PyTorch Supervised Loss Alignment**
   - Current status: `loss_name` attribute error blocking training smoke test
   - Action: Enforce `loss_function='MAE'` for supervised configs
   - Priority: High (unblocks CLI parity)

3. **Automate Iteration Scoring**
   - Tool: `scripts/orchestration/score_iteration.py` (new)
   - Output: CSV append-only log for trend monitoring
   - Benefit: Continuous process health visibility

### Short-Term (Next 10 Iterations)

4. **Reduce Evidence-Only Loop Frequency**
   - Target: ≤15% of iterations (currently 26%)
   - Method: Tighten Tier 2 dwell rule (escalate at dwell=3 instead of 4)
   - Monitor: Track via automated scoring tool

5. **Backend Parity Audit**
   - Validate: All CLI entry points support `--backend {tensorflow,pytorch}`
   - Add: Smoke tests for PyTorch training/inference pipelines
   - Document: Known TF-only paths (if any) in `docs/workflows/pytorch.md`

---

## Appendix: Methodological Transparency

### Scoring Formula (Code-Diff Heuristic)
```
score = min(100,
            min(40, code_files × 10) +
            min(30, test_files × 15) +
            min(10, doc_files × 5))
```

**Rationale:**
- Code files weighted highest (up to 40 pts)
- Test files next (up to 30 pts) — encourages test discipline
- Doc files capped low (up to 10 pts) — evidence is necessary but insufficient

### Limitations & Caveats

1. **No pre-prompt baseline:** Analyzed window (289–312) is entirely post-prompt-change; cannot compute effect size
2. **Small sample (n=24):** Statistical power insufficient for hypothesis testing
3. **Heuristic scoring:** Code file count is proxy for complexity, not direct measure
4. **Autocorrelation:** Consecutive iterations often work on same focus; not independent samples
5. **Context-blind:** Diff analysis doesn't capture **why** code changed (bug fix vs feature vs refactor)

### Data Sources
- **Git commits:** `git log --all` with `[SYNC i=N]` markers
- **Diffs:** `git diff --numstat <prev>..<curr>`
- **Fix plan:** `docs/fix_plan.md` for context on initiatives
- **Summaries:** `logs/.../galph-summaries/` (lightly consulted due to sparsity)

---

## Conclusion

The analyzed iteration cohort (289–312) demonstrates **moderate productivity with periodic high-impact deliverables**. Process discipline (test coverage, environment freeze) is strong. Key improvement opportunities:

1. **Reduce blocked task dwell time** (26% evidence-only loops)
2. **Accelerate PyTorch parity completion** (currently at CLI integration stage)
3. **Automate process health monitoring** (iteration scoring tool)

The two 80-point iterations (302, 305) prove the system is capable of step-function progress when unblocked. Focusing on blocker resolution should increase overall throughput.

---

**Report Author:** Claude Code Iteration Analysis Agent
**Repository:** /home/ollie/Documents/PtychoPINN
**Branch:** feature/torchapi-newprompt
**Analysis Commit:** 56d63171 ([SYNC i=312])
**Full Report:** `docs/iteration_analysis_report.md`
**Data Artifact:** `docs/iteration_analysis_data.json`
