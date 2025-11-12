# Dwell Escalation Strengthening Proposal

## Current Problem

From `galph_memory.md` audit (iterations 296-300):
- **dwell=9** reached with repeated `ready_for_implementation` status
- No focus switch despite 9 consecutive planning loops
- Supervisor issues same Do Now repeatedly; Ralph never executes or hits different blocker each time

From `prompts/supervisor.md:59-60`:
> "Remain in `gathering_evidence`/`planning` ≤ 2 consecutive turns per focus. On the third, either transition to `ready_for_implementation` with a <em>runnable</em> production task for Ralph or switch focus and record the block."

**Gap:** No enforcement after `ready_for_implementation` is issued. Dwell continues incrementing without consequences.

---

## Proposed Changes

### **1. Add Hard Thresholds to `prompts/supervisor.md`**

**Location:** `prompts/supervisor.md:59` (inside `<loop_discipline>`)

**Replace current text:**
```xml
<strong>Dwell enforcement (hard):</strong> Remain in `gathering_evidence` or `planning` at most two consecutive turns per focus. On the third, either set `ready_for_implementation` with a code task or switch focus and record the block.
```

**With strengthened version:**
```xml
<strong>Dwell enforcement (three-tier hard gate):</strong>

  <strong>Tier 1 (dwell=2):</strong> On the second consecutive planning/doc loop, you MUST either:
    (a) Transition to `ready_for_implementation` with a <em>runnable</em> production code task for Ralph, OR
    (b) Switch focus and record the blocker in `docs/fix_plan.md` + `galph_memory.md`

  <strong>Tier 2 (dwell=4):</strong> If Ralph did not execute the ready_for_implementation Do Now (check `galph_memory.md` + git log for ralph commits), you MUST:
    (a) Document the PRECISE blocker in `docs/fix_plan.md` under the focus item:
        - What command was supposed to run
        - Why it didn't execute (module error? git lock? Phase C timing?)
        - What prerequisite is missing
    (b) Create a NEW focus item for the blocker (e.g., "FIX-GIT-LOCK-001", "FIX-PHASEC-TIMING-001")
    (c) Switch to the blocker focus OR mark current focus `blocked` with return condition
    (d) Reset dwell=0 for the NEW focus

  <strong>Tier 3 (dwell=6, ABSOLUTE LIMIT):</strong> If you reach dwell=6 for any reason:
    (a) STOP all planning for this focus
    (b) Force-mark the focus `blocked` in `docs/fix_plan.md` with status `blocked_escalation`
    (c) Create an `analysis/dwell_escalation_report.md` in the current hub documenting:
        - All attempts (iterations + commits)
        - Recurring blockers
        - Recommended intervention (manual fix? scope reduction? pivot?)
    (d) MANDATORY focus switch to highest-priority non-blocked item
    (e) Log the escalation in `galph_memory.md` with `action_type=escalation`

  <strong>Enforcement check (startup step 0):</strong> Before any planning, read the last `galph_memory.md` entry for the active focus and check dwell. If dwell >= 4, verify a ralph commit landed since the last `ready_for_implementation`. If not, apply Tier 2. If dwell >= 6, apply Tier 3 immediately.
```

---

### **2. Add Ralph Execution Tracking to `galph_memory.md` Format**

**Location:** `prompts/supervisor.md:266-267` (inside `<fsm>`)

**Add after the current format line:**
```xml
End-of-turn logging (required): append in `galph_memory.md`
`focus=<id/slug>` `state=<gathering_evidence|planning|ready_for_implementation>` `dwell=<n>`
`artifacts=<plans/active/<initiative>/reports/<timestamp>/>` `next_action=<one-liner or 'switch_focus'>`
```

**Extended format:**
```xml
End-of-turn logging (required): append in `galph_memory.md`
`focus=<id/slug>` `state=<gathering_evidence|planning|ready_for_implementation>` `dwell=<n>`
`ralph_last_commit=<sha8_of_last_ralph_commit_for_this_focus_or_'none'>`
`artifacts=<plans/active/<initiative>/reports/<timestamp>/>` `next_action=<one-liner or 'switch_focus'>`

To populate `ralph_last_commit`:
  - After every turn, run: `git log --all --oneline --grep='RALPH AUTO' --grep='RALPH MANUAL' -n 1 --format='%h' -- '<affected_paths>'`
  - Compare to previous entry's `ralph_last_commit`
  - If changed → ralph executed successfully, reset dwell=0
  - If unchanged for 2+ turns while state=ready_for_implementation → apply Tier 2 escalation
```

---

### **3. Mandate Blocker Documentation Template**

**Create new file:** `docs/templates/blocker_report_template.md`

```markdown
# Blocker Report: [FOCUS-ID]

**Date:** [ISO8601Z timestamp]
**Dwell at escalation:** [N]
**Trigger:** [Tier 2 / Tier 3 / manual]

## Summary
[2-3 sentences: what was being attempted, why it's blocked]

## Attempts History
[List all iterations + commits where this blocker appeared]

- **Iteration NNN (commit XXXXXXXX):**
  Attempt: [what was tried]
  Result: [error signature or partial success]
  Evidence: `path/to/log:line`

[Repeat for each attempt]

## Root Cause Analysis
[Best current understanding of why this keeps failing]

## Prerequisite(s)
[List missing dependencies or infrastructure fixes needed]

- [ ] [Prerequisite 1 - e.g., "Fix .git/index.lock cleanup in supervisor preflight"]
- [ ] [Prerequisite 2 - e.g., "Implement Phase C --dose filtering"]

## Recommended Action
[What should happen next: manual intervention? scope reduction? dependency creation?]

## Return Condition
[Specific condition that would unblock this focus]

Example: "Once FIX-GIT-LOCK-001 is done and `git status` shows no lock file for 3 consecutive iterations, retry this focus's Do Now."
```

**Reference from:** `prompts/supervisor.md:Tier 2/Tier 3` enforcement

---

### **4. Add Startup Dwell Check**

**Location:** `prompts/supervisor.md:66` (startup_steps, step 0)

**Current text:**
```xml
0. <strong>Dwell tracking (persistent):</strong> If `galph_memory.md` is missing, create it... [existing logic]
```

**Add after existing dwell logic:**
```xml
   <strong>Dwell escalation gate (pre-planning check):</strong>
   - If dwell >= 6: Apply Tier 3 immediately (force-blocked, create escalation report, switch focus)
   - If dwell >= 4 AND state=ready_for_implementation: Check if ralph commit landed since last turn
     - Run: `git log --oneline --all -n 1 --grep='RALPH' -- '<focus_paths>'`
     - If no new ralph commit: Apply Tier 2 (document blocker, create blocker focus, switch)
   - If dwell == 2 AND state=planning: Remind yourself to either set ready_for_implementation or switch focus THIS turn
```

---

## Implementation Checklist

### Phase A: Prompt Updates (1 iteration)
- [ ] Edit `prompts/supervisor.md:59-60` with three-tier enforcement
- [ ] Edit `prompts/supervisor.md:266-267` to add `ralph_last_commit` tracking
- [ ] Edit `prompts/supervisor.md:66` startup step 0 with pre-planning dwell gate
- [ ] Create `docs/templates/blocker_report_template.md`
- [ ] Update `CLAUDE.md:27-28` to reference the three-tier system
- [ ] Commit changes with message: "Strengthen dwell escalation: three-tier hard gates + blocker documentation"

### Phase B: Validation (2 iterations)
- [ ] Test Tier 2 enforcement:
  - Pick a focus with dwell=4 and no recent ralph commits
  - Verify supervisor documents blocker and switches focus
  - Check `docs/fix_plan.md` for blocker focus creation

- [ ] Test Tier 3 enforcement:
  - Simulate dwell=6 scenario (or use current STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 if still stalled)
  - Verify supervisor creates `analysis/dwell_escalation_report.md`
  - Verify force-blocked status in `docs/fix_plan.md`
  - Verify focus switch occurs

### Phase C: Retroactive Cleanup (1 iteration)
- [ ] For current focus with dwell=9:
  - Apply Tier 3 manually
  - Create `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/.../analysis/dwell_escalation_report.md`
  - Document all 9 attempts with evidence links
  - Identify the actual recurring blocker (likely: Phase C timing + git lock combo)
  - Create blocker focus items: `FIX-GIT-LOCK-001`, `FIX-PHASEC-TIMING-001`
  - Mark current focus `blocked_escalation` with return condition

---

## Expected Outcomes

### Immediate (iterations 301-305)
- **Dwell ceiling enforced:** No focus will exceed dwell=6
- **Blocker visibility:** Recurring infrastructure issues documented with evidence
- **Focus mobility:** Supervisor switches to productive work instead of replanning same task

### Medium-term (iterations 306-320)
- **Blocker fixes prioritized:** Infrastructure issues (git lock, Phase C timing) get dedicated focus items
- **Process velocity:** Average dwell per focus drops from current ~4.5 to target ~2.0
- **Implementation ratio:** Ratio of implementation:planning iterations improves from ~27% to target ~50%

### Long-term (iterations 321+)
- **Reduced stagnation:** Planning plateaus (like iter 271-280, 297-298) prevented
- **Better escalation data:** `analysis/dwell_escalation_report.md` files provide audit trail for process improvements
- **Clearer blocker tracking:** Infrastructure debt becomes visible and measurable

---

## Rollout Plan

1. **Now:** Review this proposal
2. **Next iteration:** Implement Phase A (prompt updates)
3. **Following 2 iterations:** Validate enforcement (Phase B)
4. **Cleanup iteration:** Apply Tier 3 retroactively to current dwell=9 focus

**Estimated total effort:** 4 iterations to full enforcement
