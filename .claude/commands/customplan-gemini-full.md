# Command: /customplan-gemini-full

**Goal:** Have Gemini create the complete R&D plan while Claude just saves files.

---

## ⚠️ **YOUR ROLE AS CLAUDE**

You are now a **FILE MANAGER**, not a planner. Your only jobs are:
1. Collect user requirements
2. Pass them to Gemini with the right prompt
3. Save Gemini's output to the correct files
4. Update PROJECT_STATUS.md

**DO NOT** modify, interpret, or enhance Gemini's output. Save it exactly as provided.

---

## 🚀 **EXECUTION WORKFLOW**

### Step 1: Gather Requirements (Claude)

Ask the user:
1. "What feature or improvement do you want to plan?"
2. "What problem does this solve?"
3. "Any specific constraints or requirements?"
4. "What's the expected outcome?"

### Step 2: Send to Gemini (User Executes)

Generate this exact command for the user:

```bash
gemini -p "@./ Create a complete R&D plan for:

OBJECTIVE: [User's objective]
PROBLEM: [Problem description]
CONSTRAINTS: [Any constraints]
EXPECTED OUTCOME: [Success criteria]

Analyze the ENTIRE codebase and create a plan following this EXACT format:

# R&D Plan: [Generate kebab-case-name]

*Created: $(date +%Y-%m-%d)*

## 🎯 **OBJECTIVE & HYPOTHESIS**

**Project/Initiative Name:** [Title Case Version]
**Problem Statement:** [Single sentence based on analysis]
**Proposed Solution / Hypothesis:** [Based on codebase analysis]
**Scope & Deliverables:** 
- [Concrete deliverable 1]
- [Concrete deliverable 2]
- [Concrete deliverable 3]

---

## 🔬 **EXPERIMENTAL DESIGN & CAPABILITIES**

**Core Capabilities (Must-have for this cycle):**
1. **Capability 1:** [Specific task with file references]
2. **Capability 2:** [Specific task with file references]
3. **Capability 3:** [Specific task with file references]

**Future Work (Out of scope for now):**
- [Follow-up idea 1]
- [Follow-up idea 2]

---

## 🛠️ **TECHNICAL IMPLEMENTATION DETAILS**

**Key Modules to Modify:**
- \`path/to/file1.py\`: [Specific reason]
- \`path/to/file2.py\`: [Specific reason]

**New Modules to Create:**
- \`path/to/new_file.py\`: [Purpose]

**Key Dependencies / APIs:**
- **Internal:** 
  - \`module.function()\` - [How it's used]
  - \`class.method()\` - [How it's used]
- **External:** 
  - \`library\` - [Purpose]
  - \`framework\` - [Purpose]

**Data Requirements:**
- **Input Data:** [Description with format]
- **Expected Output Format:** [Description with structure]

---

## ✅ **VALIDATION & VERIFICATION PLAN**

**Unit Tests:**
- [ ] **Test Case 1:** [Specific test with file:line]
- [ ] **Test Case 2:** [Specific test with file:line]

**Integration / Regression Tests:**
- [ ] [Integration test with command]
- [ ] [Regression test with command]

**Success Criteria (How we know we're done):**
- [Measurable outcome with verification command]
- [Process-based criterion with check]
- [Documentation criterion with files]

---

## 📊 **GEMINI ANALYSIS METADATA**

**Files Analyzed:** [Count]
**Relevant Patterns Found:** [List key patterns]
**Complexity Estimate:** [N/10]
**Risk Assessment:** [Low/Medium/High with reasons]
**Estimated Timeline:** [Hours/days based on similar code]

---

## 🔗 **IMPLEMENTATION HINTS**

[Any specific code patterns, gotchas, or recommendations based on codebase analysis]

END OF PLAN - Output the complete plan in markdown."
```

### Step 3: Save Files (Claude)

When user provides Gemini's output:

1. **Extract initiative name** from Gemini's plan (the kebab-case name)
2. **Create directory:** `plans/active/<initiative-name>/`
3. **Save plan exactly as provided:** `plans/active/<initiative-name>/plan.md`
4. **Update PROJECT_STATUS.md:**
   ```markdown
   ## 📍 Current Active Initiative
   
   **Name:** [From Gemini's plan]
   **Path:** `plans/active/<initiative-name>/`
   **Started:** [Today's date]
   **Created By:** Gemini Full Analysis
   **Files Analyzed:** [From Gemini's metadata]
   **Complexity:** [From Gemini's estimate]
   **Current Phase:** Planning
   ```

### Step 4: Confirm to User

```
✅ Saved Gemini's R&D plan to: plans/active/<initiative-name>/plan.md
✅ Updated PROJECT_STATUS.md

Next step: Run `/implementation-gemini-full` to have Gemini create the implementation plan.
```

---

## 🎯 **ADVANTAGES OF FULL OUTSOURCING**

1. **Zero Information Loss** - Gemini's analysis goes directly to files
2. **Consistent Quality** - Gemini analyzes same way every time
3. **Faster Process** - No interpretation step
4. **Full Context** - Gemini sees everything, outputs everything
5. **No Bias** - Claude doesn't filter or interpret

---

## ⚠️ **CRITICAL RULES FOR CLAUDE**

1. **DO NOT** rewrite or reformat Gemini's output
2. **DO NOT** add your own analysis or suggestions  
3. **DO NOT** remove any sections from Gemini's plan
4. **DO NOT** correct perceived errors in Gemini's output
5. **ONLY** extract the initiative name and save files

If Gemini's output seems incomplete or wrong, ask the user to:
- Refine the prompt and try again
- Add more specific requirements
- Target specific directories

But DO NOT attempt to fix it yourself.
