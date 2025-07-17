# Quick Start Guide: Initiative Planning & Execution System

*A complete walkthrough of the planning and implementation workflow*

---

## 🚀 **OVERVIEW**

This system provides a structured approach to planning and executing development initiatives through three main phases:

1. **Planning** - Define what to build (`/customplan`)
2. **Implementation Planning** - Define how to build it (`/implementation`)  
3. **Execution** - Build it step by step (`/complete-phase`)

---

## 📁 **INITIAL SETUP**

### First Time Setup
```bash
# Create the required directories
mkdir -p plans/active plans/archive plans/templates

# Create the master status tracker
touch PROJECT_STATUS.md

# Optional: Copy command files to a commands/ directory
mkdir commands
cp customplan.md implementation.md complete-phase.md phase-checklist.md commands/
```

### File Structure You'll Create
```
your-project/
├── PROJECT_STATUS.md          # Master tracker (create this first)
├── plans/
│   ├── active/               # Current work
│   └── archive/              # Completed work
└── commands/                 # AI command references (optional)
```

---

## 🎯 **COMPLETE WORKFLOW EXAMPLE**

Let's walk through adding a new feature: "FRC Metric Implementation"

### Step 1: Create R&D Plan
```bash
# Start with the planning command
/customplan
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
```

AI reads the R&D plan and creates:
- `plans/active/frc-metric/implementation.md`
- Typically 2-4 phases + final phase

### Step 3: Start Phase 1
```bash
# Generate detailed checklist
/phase-checklist 1
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
```

AI will:
1. Verify the success test passes
2. Update progress tracking
3. Generate phase_2_checklist.md
4. Update PROJECT_STATUS.md

### Step 6: Continue Through Phases
```bash
# Repeat for each phase
/complete-phase  # After phase 2
/complete-phase  # After final phase
```

### Step 7: Initiative Complete!
When the final phase is done, AI will:
- Archive to `plans/archive/2024-03-frc-metric/`
- Update PROJECT_STATUS.md
- Ask for your next objective

---

## 📋 **COMMAND REFERENCE**

| Command | Purpose | Creates |
|---------|---------|---------|
| `/customplan` | Start new initiative | `plans/active/<n>/plan.md` |
| `/implementation` | Create phase breakdown | `plans/active/<n>/implementation.md` |
| `/phase-checklist N` | Generate task list | `plans/active/<n>/phase_N_checklist.md` |
| `/complete-phase` | Finish current phase | Next checklist or archives |

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

## 🚨 **COMMON ISSUES & SOLUTIONS**

### "Success test failed"
- AI will stop and diagnose
- Check incomplete tasks
- Review error messages
- May need to fix and retry

### "Can't find implementation plan"
- Check PROJECT_STATUS.md for correct path
- Verify file has `<!-- ACTIVE IMPLEMENTATION PLAN -->`
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

## 🎯 **SUCCESS METRICS**

Track your productivity:
- **Initiatives/month:** How many completed?
- **Average duration:** Getting faster?
- **Phase accuracy:** Estimated vs actual time
- **Test coverage:** More tests = fewer bugs

---

## 🚦 **READY TO START?**

1. Create your PROJECT_STATUS.md file
2. Think about your next objective
3. Run `/customplan`
4. Let the system guide you!

---

*For detailed documentation, see:*
- `customplan.md` - R&D planning details
- `implementation.md` - Phase planning details
- `complete-phase.md` - Execution details
- `path-conventions.md` - File organization rules