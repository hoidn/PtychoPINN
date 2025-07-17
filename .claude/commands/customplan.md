# Command: /customplan

**Goal:** Generate and save a focused R&D plan document for the next development cycle.

---

## ⚠️ **IMPORTANT: YOUR TASK AS THE AI AGENT**

Your **only task** is to **GENERATE A MARKDOWN DOCUMENT AND SAVE IT TO A FILE**. Do **NOT** execute any of the steps described in the plan you create.

Your process is:
1. **Understand the Objective:** Engage with the user to clearly understand the technical problem, proposed solution, and expected outcomes.
2. **Create Initiative Name:** Generate a kebab-case initiative name (e.g., `coordinate-based-alignment`, `multi-trial-statistics`).
3. **Generate Plan Content:** Use the "R&D PLAN TEMPLATE" below to create the full markdown content.
4. **Save the File:** 
   - Create directory: `plans/active/<initiative-name>/`
   - Save to: `plans/active/<initiative-name>/plan.md`
5. **Update Project Status:**
   - Update or create `PROJECT_STATUS.md` at the project root
   - Add the new initiative as the current active initiative
6. **Confirm and Present:** Announce that you have saved the files and present the full content for review.

---

## 📋 **R&D PLAN TEMPLATE**

```markdown
# R&D Plan: <Initiative Name in Title Case>

*Created: <Today's date in YYYY-MM-DD format>*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** <Title Case Name matching the folder name>
*Example: "Coordinate-Based Alignment" for `coordinate-based-alignment/`*

**Problem Statement:** <A single sentence describing the specific technical or scientific limitation being addressed>

**Proposed Solution / Hypothesis:** <How this work will solve the problem and the expected outcome>

**Scope & Deliverables:** <A list of the concrete outputs of this work>
- <Deliverable 1>
- <Deliverable 2>
- <Deliverable 3>

---

## 🔬 **EXPERIMENTAL DESIGN & CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**
1. **Capability 1:** <A specific, actionable engineering or scientific task>
2. **Capability 2:** <Another specific, actionable task>
3. **Capability 3:** <Another task>

**Future Work (Out of scope for now):**
- <A potential follow-up idea that will not be addressed in this cycle>
- <Another out-of-scope idea>

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Modify:**
- `<path/to/module1.py>`: <Brief reason for modification>
- `<path/to/module2.py>`: <Brief reason for modification>

**New Modules to Create:**
- `<path/to/new_module.py>`: <Purpose of the new module>

**Key Dependencies / APIs:**
- **Internal:** <List of internal functions, classes, or modules this work will depend on>
  - `module.function()` - <Brief description>
- **External:** <List of key third-party libraries this work will use>
  - `numpy` - <Usage purpose>
  - `scipy.ndimage` - <Usage purpose>

**Data Requirements:**
- **Input Data:** <Description of the necessary input data and its format>
- **Expected Output Format:** <Description of the final data artifacts that will be produced>

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit Tests:**
- [ ] **Test Case 1:** <A specific unit test to verify a piece of the new logic>
- [ ] **Test Case 2:** <Another specific unit test>

**Integration / Regression Tests:**
- [ ] <A test to ensure the new code works within the larger system>
- [ ] <Another integration test>

**Success Criteria (How we know we're done):**
- <A measurable, verifiable outcome that proves the objective was met>
- <A process-based completion criterion>
- <A documentation-based completion criterion>

---

## 📁 **File Organization**

**Initiative Path:** `plans/active/<initiative-name>/`

**Planning Documents:**
- `plan.md` - This R&D specification (this file)
- `implementation.md` - Phased implementation plan (to be created next)
- `phase_*_checklist.md` - Detailed checklists for each phase

**Next Step:** Run `/implementation` to generate the phased implementation plan.
```

---

## 📁 **PROJECT STATUS UPDATE TEMPLATE**

When updating PROJECT_STATUS.md, add or update the following section:

```markdown
## 📍 Current Active Initiative

**Name:** <Initiative Name in Title Case>
**Path:** `plans/active/<initiative-name>/`
**Started:** <Today's date YYYY-MM-DD>
**Current Phase:** Planning
**Progress:** ░░░░░░░░░░░░░░░░ 0%
**Next Milestone:** Generate implementation plan
**R&D Plan:** `plans/active/<initiative-name>/plan.md`
**Implementation Plan:** *To be created*
```

---

## 💡 **GUIDANCE FOR ENGAGING WITH THE USER**

Before creating the plan, ask clarifying questions if needed:

1. **For vague objectives:** "Could you describe the specific technical limitation or problem you're trying to solve?"
2. **For missing context:** "What is the current behavior, and how should it change?"
3. **For scope clarity:** "What would you consider the minimum viable solution for this cycle?"
4. **For validation:** "How will we verify that the solution works correctly?"

