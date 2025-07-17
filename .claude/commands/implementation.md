# Command: /implementation

**Goal:** Generate and save a phased implementation plan document based on the most recent R&D plan.

---

## ⚠️ **IMPORTANT: YOUR TASK AS THE AI AGENT**

Your **only task** is to **GENERATE A MARKDOWN DOCUMENT AND SAVE IT TO A FILE**. Do **NOT** execute any of the steps described in the plan you create.

Your process is:
1. **Read Project Status:** Read `PROJECT_STATUS.md` to identify the current active initiative and its path.
2. **Read the R&D Plan:** Navigate to the initiative path and read `plan.md`.
3. **Decompose into Logical Phases:** Break down the work into 1-5 phases based on complexity:
   - Simple tasks: 1 implementation phase + 1 finalization phase
   - Complex tasks: 2-4 implementation phases + 1 finalization phase
4. **Generate Implementation Plan:** Use the template below to create the implementation plan.
5. **Save the File:** Save to `<initiative-path>/implementation.md`
6. **Update Project Status:** Update the current phase in `PROJECT_STATUS.md` to "Phase 1: <name>"
7. **Confirm and Present:** Announce the saved file location and present the content for review.

---

## 📋 **IMPLEMENTATION PLAN TEMPLATE**

```markdown
<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan

**Project:** <Copy from R&D Plan>
**Initiative Path:** `plans/active/<initiative-name>/`
**Created:** <Today's date YYYY-MM-DD>

**Core Technologies:** <List main languages/frameworks that will be used>

---

## 📄 **DOCUMENT HIERARCHY**

This document orchestrates the implementation of the objective defined in the main R&D plan. The full set of documents for this initiative is:

- **`plan.md`** - The high-level R&D Plan
  - **`implementation.md`** - This file - The Phased Implementation Plan
    - `phase_1_checklist.md` - Detailed checklist for Phase 1
    - `phase_2_checklist.md` - Detailed checklist for Phase 2 (if applicable)
    - ... (additional phase checklists as needed)
    - `phase_final_checklist.md` - Checklist for the Final Phase

---

## 🎯 **PHASE-BASED IMPLEMENTATION**

**Overall Goal:** <A one-sentence summary of the final state after all phases are complete>

**Total Estimated Duration:** <Sum of all phase estimates>

---

## 📋 **IMPLEMENTATION PHASES**

### **Phase 1: <Descriptive Name for First Logical Component>**

**Goal:** <A concise description of this phase's objective>

**Deliverable:** <The specific, tangible output of this phase>

**Estimated Duration:** <X hours/days>

**Key Tasks:**
- <High-level task 1>
- <High-level task 2>
- <High-level task 3>

**Dependencies:** None (first phase)

**Implementation Checklist:** `phase_1_checklist.md`

**Success Test:** <Specific, verifiable test command or check>
*Example: "Running `python src/module.py --test` shows the new function output"*

---

### **Phase 2: <Descriptive Name for Second Component>** *(if needed)*

**Goal:** <A concise description of this phase's objective>

**Deliverable:** <The specific, tangible output of this phase>

**Estimated Duration:** <X hours/days>

**Key Tasks:**
- <High-level task 1>
- <High-level task 2>

**Dependencies:** Requires Phase 1 completion

**Implementation Checklist:** `phase_2_checklist.md`

**Success Test:** <Specific, verifiable test command or check>

---

### **Final Phase: Validation & Documentation**

**Goal:** Validate the complete implementation and update all relevant documentation.

**Deliverable:** Fully tested feature with updated documentation and closed initiative.

**Estimated Duration:** <Usually 2-4 hours>

**Key Tasks:**
- Run comprehensive tests for <specific features from previous phases>
- Verify <specific success criteria from R&D plan>
- Update documentation for <specific modules/features that changed>
- Create example usage for <new capability>
- Archive initiative and update project status

**Dependencies:** All previous phases complete

**Implementation Checklist:** `phase_final_checklist.md`

**Success Test:** All R&D plan success criteria are met, all tests pass, documentation is updated.

---

## 📊 **PROGRESS TRACKING**

### Phase Status:
- [ ] **Phase 1:** <Name> - 0% complete
- [ ] **Phase 2:** <Name> - 0% complete *(if applicable)*
- [ ] **Final Phase:** Validation & Documentation - 0% complete

**Current Phase:** Phase 1: <Name>
**Overall Progress:** ░░░░░░░░░░░░░░░░ 0%

### Milestone Timeline:
- **Week 1:** Complete Phase 1, begin Phase 2
- **Week 2:** Complete Phase 2 and Final Phase

---

## 🚀 **GETTING STARTED**

1. **Generate Phase 1 Checklist:** Run `/phase-checklist 1` to create the detailed checklist
2. **Begin Implementation:** Follow the checklist tasks in order
3. **Track Progress:** Update task states in the checklist as you work
4. **Complete Phase:** Run `/complete-phase` when all Phase 1 tasks are done

---

## ⚠️ **RISK MITIGATION**

**Potential Blockers:**
- <Identified risk 1>: Mitigation strategy
- <Identified risk 2>: Mitigation strategy

**Rollback Plan:**
- Git: Each phase should be a separate commit/PR
- Data: Back up any modified data before each phase
- Config: Keep copies of configuration files
```

---

## 📁 **PROJECT STATUS UPDATE**

Update the following fields in PROJECT_STATUS.md:

```markdown
**Current Phase:** Phase 1: <First Phase Name>
**Progress:** ░░░░░░░░░░░░░░░░ 0%
**Next Milestone:** <Deliverable from Phase 1>
**Implementation Plan:** `plans/active/<initiative-name>/implementation.md`
```

---

## 💡 **PHASE DECOMPOSITION GUIDELINES**

When breaking work into phases:

1. **Each phase should produce a verifiable deliverable**
   - Bad: "Refactor code" 
   - Good: "Refactored module with passing unit tests"

2. **Phases should be logically independent when possible**
   - Can the phase be tested independently?
   - Does it produce something useful on its own?

3. **Consider natural boundaries:**
   - Backend changes → Frontend changes → Integration
   - Core logic → Error handling → Optimization
   - Data model → Business logic → UI

4. **Size phases appropriately:**
   - Minimum: 2-4 hours of work
   - Maximum: 2-3 days of work
   - Ideal: 1 day of focused work

5. **Final phase is always validation/documentation**
   - Not generic tasks, but specific to what was built
   - Include testing the full integration
   - Update all affected documentation
