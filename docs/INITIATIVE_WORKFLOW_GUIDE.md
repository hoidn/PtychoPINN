# Initiative Planning & Execution System

*A structured approach to planning and executing development work through phased implementation*

---

## 📌 **WHAT IS THIS SYSTEM?**

This is an AI-assisted development workflow that uses structured planning documents and phase-based execution. It's designed to work with AI coding assistants (Claude, ChatGPT, Cursor, etc.) to break complex work into manageable phases.

**Key Principles:**
- Documentation-driven development
- Phase-based execution with clear checkpoints
- AI-assisted but human-controlled workflow
- Test-driven validation at each phase

**Requirements:**
- An AI assistant that can read/write files
- A project with version control (git)
- 15 minutes to set up the structure

---

## 📁 **INITIAL SETUP**

### First Time Setup
```bash
# Create the required directories
mkdir -p plans/active plans/archive plans/templates

# Create the master status tracker
touch PROJECT_STATUS.md
```

### File Structure You'll Create
```
your-project/
├── PROJECT_STATUS.md          # Master tracker (create this first)
└── plans/
    ├── active/               # Current work
    └── archive/              # Completed work
```

### Artifact Storage Standard
Keep evidence lean. For each initiative, maintain a single `plans/active/<initiative-id>/summary.md` and prepend a Turn Summary per loop. Store bulky artifacts outside the repo (or under a git‑ignored `.artifacts/` folder) and link to them from the plan/ledger. Do not create timestamped report directories.

### Dwell Enforcement Policy (Process)
This workflow participates in the supervisor/engineer loop defined in `prompts/supervisor.md` and `prompts/main.md`. Dwell (consecutive planning/doc loops for the same focus) is enforced via a three‑tier policy:

- Tier 1 (dwell=2): supervisor must hand off a runnable production task (with pytest selector) or switch focus.
- Tier 2 (dwell=4): if the engineer did not execute, supervisor documents the blocker (quoted Do Now, command/selector, minimal error) and switches to a blocker focus or marks the current one blocked with a return condition.
- Tier 3 (dwell=6): absolute limit — force‑block, record a dwell escalation note in the initiative’s summary, and switch focus.

Supervisor records `ralph_last_commit=<sha8|none>` in `galph_memory.md` and checks git log each loop to detect execution. See `CLAUDE.md` §2.10 and `docs/templates/blocker_report_template.md`.

### Plan Maintenance
Each initiative (or focus) owns a single evolving plan file such as `plans/active/<initiative-id>/implementation.md` or a dedicated focus document. Update that file in place—add new sections, checklists, or notes as work progresses—instead of generating a brand-new `plan/plan.md` every loop. Create a new plan file only when the scope fundamentally changes (for example, splitting off a new focus or initiative). Cross-reference the active plan path from `docs/fix_plan.md` so every loop knows which document to edit.

### Using `prompts/plan_generation.md` for new initiatives
For substantial new focuses that will span more than a single Ralph loop (for example, cross-module features, complex blockers, or multi-backend parity work), prefer using the plan‑generation prompt to seed the Working Plan instead of hand-authoring the entire document.

Workflow:

- Write a concise `$TASK_DESCRIPTION` capturing:
  - The user-visible problem or goal.
  - Any known selectors/commands and key existing artifacts (logs/metrics/reports).
  - Hard constraints (for example, modules that must not be edited without explicit scope).
- Run `prompts/plan_generation.md` with `$TASK_DESCRIPTION` as input.
- The prompt will:
  - Create `plans/active/<initiative-id>/implementation.md` from the template, customized for this focus.
  - Add a corresponding `## [<initiative-id>]` entry to `docs/fix_plan.md` pointing at the new Working Plan and, optionally, a `summary.md`/reports hub.
- After this step, treat the generated plan and ledger entry as authoritative for the initiative. Subsequent supervisor loops should:
  - Update the Working Plan in place.
  - Update `docs/fix_plan.md` Attempts History and the initiative’s `summary.md` as usual.

For very small, one-loop blockers (single selector, single file, single change), you may continue to create/adjust the plan inline without invoking `prompts/plan_generation.md`.

### PROJECT_STATUS.md Template
```markdown
# Project Status Tracker

## Active Initiatives
| Initiative | Branch | Phase | Progress | Started |
|------------|--------|-------|----------|---------|
| [Name] | feature/name | 1/3 | ████░░░░ 25% | 2024-03-15 |

## Completed Initiatives
| Initiative | Duration | Outcome | Archived |
|------------|----------|---------|----------|
| [Previous] | 5 days | Success | plans/archive/2024-03/previous/ |

## Blocked/Paused
| Initiative | Reason | Next Steps |
|------------|--------|------------|
| [Name] | Waiting for API docs | Resume when available |

## Upcoming
- [ ] Performance optimization
- [ ] User authentication
- [ ] Dashboard redesign
```

---

## 🎯 **STARTING YOUR FIRST INITIATIVE**

Tell your AI assistant (adapt to your tool):

**For Claude/Cursor:**
"/customplan for adding user authentication"

**For ChatGPT:**
"Create an R&D plan following the initiative system for adding user authentication"

**For GitHub Copilot Chat:**
"Generate a phased implementation plan for user authentication using the plans/active structure"

**Universal Prompt:**
"I want to plan a new development initiative. Create a structured plan document at plans/active/[feature-name]/plan.md with problem statement, success criteria, and deliverables. Then break it into implementation phases."

---

## 🚀 **COMPLETE WORKFLOW EXAMPLE**

Let's walk through adding a new feature: "FRC Metric Implementation"

### Step 1: Create R&D Plan
```bash
# Start with the planning command
/customplan
# Or: "Create an R&D plan for FRC metric implementation"
```

AI will ask clarifying questions:
- "What problem are you trying to solve?"
- "What's the expected outcome?"
- "What are the core capabilities needed?"

AI creates:
- `plans/active/frc-metric/plan.md`
- Updates `PROJECT_STATUS.md`

### Step 2: Generate Implementation Plan
```bash
# Break down into phases
/implementation
# Or: "Create implementation plan from the current R&D plan"
```

AI reads the R&D plan and creates:
- `plans/active/frc-metric/implementation.md`
- Typically 2-4 phases + final phase

### Step 3: Start Phase 1
```bash
# Generate detailed checklist
/phase-checklist 1
# Or: "Generate phase 1 checklist from implementation plan"
```

AI creates:
- `plans/active/frc-metric/phase_1_checklist.md`
- 10-20 specific tasks with file paths

### Step 4: Work Through Phase 1
```bash
# You manually:
1. Open phase_1_checklist.md
2. Complete tasks in order
3. Mark each task [x] as done
4. Test as you go
```

### Step 5: Complete Phase 1
```bash
# When all tasks done
/complete-phase
# Or: "Verify phase completion and generate next phase"
```

AI will:
1. Verify the success test passes
2. Update progress tracking
3. Generate phase_2_checklist.md
4. Update PROJECT_STATUS.md

### Step 6: Continue Through Phases
Repeat steps 4-5 for each phase until the initiative is complete.

### Step 7: Initiative Complete!
When the final phase is done, AI will:
- Archive to `plans/archive/2024-03-frc-metric/`
- Update PROJECT_STATUS.md
- Ask for your next objective

---

## 📋 **COMMAND REFERENCE**

These commands are prompts for your AI assistant. Copy them exactly or adapt to your AI's style:

| Command | Purpose | Creates | AI Instructions |
|---------|---------|---------|-----------------|
| `/customplan` or "Create an R&D plan for [topic]" | Start new initiative | `plans/active/<n>/plan.md` | AI analyzes project and creates structured plan |
| `/implementation` or "Create implementation plan" | Create phase breakdown | `plans/active/<n>/implementation.md` | AI reads plan.md and creates phases |
| `/phase-checklist N` or "Generate phase N checklist" | Generate task list | `plans/active/<n>/phase_N_checklist.md` | AI creates 10-20 specific tasks |
| `/complete-phase` or "Complete current phase" | Finish current phase | Next checklist or archives | AI verifies completion and prepares next |

---

## 📄 **DOCUMENT TEMPLATES**

### plan.md Structure
```markdown
# Initiative: [Name]

## Problem Statement
[What needs solving]

## Success Criteria
- [ ] Criterion 1 (measurable)
- [ ] Criterion 2 (testable)

## Deliverables
1. [Concrete output 1]
2. [Concrete output 2]

## Constraints
- Time: [X days]
- Dependencies: [List any]
- Risks: [Potential issues]
```

### implementation.md Structure
```markdown
# Implementation Plan: [Name]

## Phase 1: [Core Functionality]
**Duration:** 1 day
**Success Test:** `pytest tests/test_core.py`
**Deliverables:**
- Core algorithm implementation
- Unit tests

## Phase 2: [Integration]
**Duration:** 1 day
**Success Test:** `pytest tests/test_integration.py`
**Deliverables:**
- API endpoints
- Integration with existing code

## Final Phase: [Testing & Documentation]
**Duration:** 0.5 days
**Success Test:** All tests pass, docs complete
**Deliverables:**
- Complete test coverage
- User documentation
- Code cleanup
```

### phase_checklist.md Structure
```markdown
# Phase [N]: [Name]

## Tasks
- [ ] Create file: src/metrics/frc.py
- [ ] Implement FRC.calculate() method
- [ ] Add unit test: tests/test_frc.py
- [ ] Update imports in src/metrics/__init__.py
- [ ] Run tests: pytest tests/test_frc.py
- [ ] Add docstrings to all functions
- [ ] Update README with usage example

## Success Test
```bash
pytest tests/test_frc.py -v
```

## Notes
[Add any decisions or problems encountered]
```

---

## ✅ **SUCCESS TESTS EXPLAINED**

Each phase must have a verifiable success test. Examples:

**Phase 1 Success Test:**
```bash
# Unit tests pass
python -m pytest tests/test_frc_metric.py

# Or integration works
curl http://localhost:8000/api/metrics/frc

# Or feature functions
python -c "from metrics import FRC; print(FRC.calculate(data))"
```

**Success tests should be:**
- Automated (runnable via command)
- Specific (not "code works")
- Fast (under 30 seconds)
- Deterministic (same result each time)

---

## 🔍 **TYPICAL INITIATIVE PATTERNS**

### Small Feature (1-2 days)
```
Phase 1: Implementation → Final: Testing & Docs
```

### Medium Feature (3-5 days)
```
Phase 1: Core Logic → Phase 2: Integration → Final: Testing & Docs
```

### Large Feature (1-2 weeks)
```
Phase 1: Data Model → Phase 2: Core Logic → Phase 3: UI/API → Final: Testing & Docs
```

### Refactoring Initiative
```
Phase 1: Analysis & Prep → Phase 2: Migration → Phase 3: Cleanup → Final: Verification
```

---

## 🔧 **CUSTOMIZING FOR YOUR PROJECT**

### For Small Projects/Prototypes
- Use 2 phases maximum (Implementation + Testing)
- Skip formal documentation phase
- Combine planning and implementation commands

### For Large Teams
- Add review checkpoints between phases
- Create shared PROJECT_STATUS.md in wiki/docs
- Use PR descriptions to link to phase checklists

### For Research/Experimental Work
- Add "Exploration Phase" before implementation
- Include "Failed Approaches" section in plans
- Success tests can be "learned X about Y"

### For Legacy Codebases
- Add "Analysis Phase" to understand existing code
- Include "Regression Testing" in each phase
- Document deprecated code removal in checklists

---

## 👥 **TEAM COLLABORATION**

### Solo Developer
- Follow guide as written
- Self-review at phase boundaries

### Pair Programming
- One drives checklist, other implements
- Switch roles each phase
- Both review before phase completion

### Team Projects
- Assign initiatives to individuals
- Review implementation plans in team meeting
- PR reviews at phase boundaries
- Update shared PROJECT_STATUS.md

---

## 💡 **BEST PRACTICES**

### Planning Phase
- Be specific about success criteria
- Include concrete deliverables
- Limit scope to 1-2 weeks max
- Define clear test cases upfront

### Implementation Phase
- Each phase should produce something testable
- Include both unit and integration tests
- Document as you go, not at the end
- Commit at phase boundaries

### Working Through Checklists
- Complete tasks in order when possible
- Update checklist frequently
- Add notes about decisions/problems
- Don't skip the verification steps

### Phase Completion
- Always run the success test
- Let AI verify before proceeding
- Create PRs at phase boundaries
- Keep the main branch stable

---

## 🚨 **HANDLING INTERRUPTIONS**

### Urgent Hotfix Needed
```bash
# 1. Pause current initiative
echo "PAUSED: $(date)" >> current_phase_checklist.md
git stash

# 2. Create hotfix
git checkout -b hotfix/issue-name
# Fix issue
git commit -m "Hotfix: [description]"

# 3. Resume initiative
git checkout feature/current
git stash pop
```

### Initiative Needs Major Pivot
1. Document lessons learned in current plan.md
2. Archive current work to plans/paused/
3. Create new plan with updated approach
4. Reference old plan in new documentation

### Initiative Cancelled
1. Document why in plan.md
2. Archive to plans/cancelled/
3. Update PROJECT_STATUS.md with outcome
4. Extract any reusable work

---

## 🚦 **COMMON ISSUES & SOLUTIONS**

### "Success test failed"
- AI will stop and diagnose
- Check incomplete tasks
- Review error messages
- May need to fix and retry

### "Can't find implementation plan"
- Check PROJECT_STATUS.md for correct path
- Verify file exists in plans/active/
- May need to regenerate

### "Too many phases"
- Refocus on MVP for this cycle
- Move features to "Future Work"
- Create follow-up initiative

### "Phase too large"
- Break into smaller phases
- Each phase: 1 day of work ideal
- Can manually edit implementation.md

---

## 🔗 **INTEGRATION WITH GIT**

### Recommended Git Workflow
```bash
# Start initiative
git checkout -b feature/frc-metric

# After each phase
git add -A
git commit -m "[FRC Metric] Phase 1: Core implementation"
git push origin feature/frc-metric

# After final phase
git checkout main
git merge feature/frc-metric
git push origin main
```

### Branch Naming
- Feature branches: `feature/<initiative-name>`
- Hotfixes: `hotfix/<issue-name>`
- Experiments: `experiment/<idea-name>`

---

## 🎮 **QUICK TERMINAL WORKFLOW**

```bash
# Monday: Start new feature
/customplan
/implementation
/phase-checklist 1

# Tuesday: Work through phase 1
vim plans/active/feature/phase_1_checklist.md
# ... do work, mark tasks complete ...
/complete-phase

# Wednesday: Phase 2
# ... work through phase_2_checklist.md ...
/complete-phase

# Thursday: Final phase
# ... complete validation & docs ...
/complete-phase

# Friday: Start next feature!
/customplan
```

---

## 📊 **TRACKING PROGRESS**

### Check Current Status
```bash
# See everything at a glance
cat PROJECT_STATUS.md

# See current phase details
cat plans/active/*/implementation.md | grep "Current Phase"

# Count completed tasks
grep -c "\[x\]" plans/active/*/phase_*_checklist.md
```

### Visual Progress
The system uses progress bars:
- ░░░░░░░░░░░░░░░░ 0%
- ████░░░░░░░░░░░░ 25%  
- ████████░░░░░░░░ 50%
- ████████████░░░░ 75%
- ████████████████ 100%

---

## 🚦 **READY TO START?**

1. Create your PROJECT_STATUS.md file
2. Think about your next objective
3. Run `/customplan` or ask your AI to create an R&D plan
4. Let the system guide you!

The key is to start small, complete one full cycle, and then adapt the system to your needs.
