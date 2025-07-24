# Command: /implementation-gemini-full <initiative-path>

**Goal:** Autonomously generate a complete, code-aware, phased implementation plan by delegating the analysis and authoring to Gemini, then saving the resulting artifacts to the project structure.

**Usage:**
- `/implementation-gemini-full plans/active/real-time-notifications`

**Prerequisites:**
- An R&D plan (`plan.md`) must exist at the specified `<initiative-path>`.

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**YOUR ROLE IS AN AUTONOMOUS ORCHESTRATOR AND FILE MANAGER.**
1.  You MUST parse the R&D plan from the specified `<initiative-path>/plan.md`.
2.  You MUST run `repomix` to create a complete, fresh snapshot of the codebase context.
3.  You MUST build a structured prompt file (`impl-prompt.md`) using the XML format.
4.  You MUST execute `gemini -p "@impl-prompt.md"` to delegate the implementation plan generation.
5.  You MUST save Gemini's response **exactly as provided** to the correct output file.
6.  You MUST update `PROJECT_STATUS.md` with the new phase information.

**DO NOT:**
-   ❌ Modify, interpret, or enhance Gemini's output in any way.
-   ❌ Create the implementation plan yourself. Your job is to run the process.
-   ❌ Skip any step. The workflow is non-negotiable.

---

## 🤖 **YOUR EXECUTION WORKFLOW**

### Step 1: Prepare Context from the R&D Plan

Parse arguments and load the high-level R&D plan that will guide this implementation.

```bash
# Parse arguments
INITIATIVE_PATH="$1"
RD_PLAN_PATH="$INITIATIVE_PATH/plan.md"

# Verify the R&D plan exists
if [ ! -f "$RD_PLAN_PATH" ]; then
    echo "❌ ERROR: R&D plan not found at '$RD_PLAN_PATH'."
    echo "Please run /customplan-gemini-full first."
    exit 1
fi

# Read the entire content of the R&D plan.
RD_PLAN_CONTENT=$(cat "$RD_PLAN_PATH")

echo "✅ Successfully loaded R&D plan from '$RD_PLAN_PATH'."
```

### Step 2: Aggregate Codebase Context with Repomix

Create a comprehensive and reliable context snapshot of the entire project for Gemini.

```bash
# Use repomix for a complete, single-file context snapshot.
npx repomix@latest . \
  --include "**/*.{js,py,md,sh,json,c,h,log,yml,toml}" \
  --ignore "build/**,node_modules/**,dist/**,*.lock"

# Verify that the context was created successfully.
if [ ! -s ./repomix-output.xml ]; then
    echo "❌ ERROR: Repomix failed to generate the codebase context. Aborting."
    exit 1
fi

echo "✅ Codebase context aggregated into repomix-output.xml."
```

### Step 3: MANDATORY - Build the Prompt File

You will now build the prompt for Gemini in a file using the structured XML pattern.

#### Step 3.1: Create Base Prompt File
```bash
# Clean start for the prompt file
rm -f ./impl-prompt.md 2>/dev/null

# Create the structured prompt with placeholders using the v3.0 XML pattern
cat > ./impl-prompt.md << 'PROMPT'
<task>
You are an expert Lead Software Engineer. Your task is to create a complete, phased implementation plan based on a high-level R&D plan.

Your implementation plan must be deeply informed by an analysis of the provided codebase. You will break the project down into logical, testable phases, and for each phase, you will define the goals, tasks, and success criteria.

<steps>
<1>
Analyze the `<rd_plan_context>` to understand the project's overall objectives, scope, and technical specifications.
</1>
<2>
Thoroughly analyze the entire `<codebase_context>` to identify natural boundaries for phasing, dependencies, existing code patterns, and potential risks.
</2>
<3>
Generate the complete, phased implementation plan. The plan must strictly adhere to the format specified in `<output_format>`. All sections must be filled out based on your analysis.
</3>
</steps>

<context>
<rd_plan_context>
[Placeholder for the content of plan.md]
</rd_plan_context>

<codebase_context>
<!-- Placeholder for content from repomix-output.xml -->
</codebase_context>
</context>

<output_format>
Your entire response must be a single Markdown block containing the implementation plan. Do not include any conversational text before or after the plan. The format is non-negotiable.

<!-- ACTIVE IMPLEMENTATION PLAN -->
<!-- DO NOT MISTAKE THIS FOR A TEMPLATE. THIS IS THE OFFICIAL SOURCE OF TRUTH FOR THE PROJECT'S PHASED PLAN. -->

# Phased Implementation Plan
...
[The entire detailed Markdown template from the original prompt goes here, verbatim. It's an excellent template.]
...
## 📊 **GEMINI ANALYSIS METADATA**
...
END OF PLAN
</output_format>
</task>
PROMPT
```

#### Step 3.2: Append Dynamic Context to the Prompt File
```bash
# Inject the R&D plan and the repomix context into the prompt file.
# Using a temp file for the plan handles multi-line variables and special characters safely.
echo "$RD_PLAN_CONTENT" > ./tmp/rd_plan.txt
sed -i.bak -e '/\[Placeholder for the content of plan.md\]/r ./tmp/rd_plan.txt' -e '//d' ./impl-prompt.md

# Append the codebase context
echo -e "\n<codebase_context>" >> ./impl-prompt.md
cat ./repomix-output.xml >> ./impl-prompt.md
echo -e "\n</codebase_context>" >> ./impl-prompt.md

echo "✅ Successfully built structured prompt file: ./impl-prompt.md"
```

### Step 4: MANDATORY - Execute Gemini Analysis

You MUST now execute Gemini using the single, clean, and verifiable prompt file.

```bash
# Execute Gemini with the fully-formed prompt file
gemini -p "@./impl-prompt.md"
```

### Step 5: Save Implementation Plan and Update Project Status

Your final role: receive the output from your command and save it without modification.

```bash
# [You will receive Gemini's implementation plan as a response from the command above]
# For this example, we'll assume the response is captured into $GEMINI_RESPONSE.

# Define the output path
OUTPUT_PATH="$INITIATIVE_PATH/implementation.md"

# Save the plan exactly as received.
echo "$GEMINI_RESPONSE" > "$OUTPUT_PATH"

# Verify the file was saved
if [ ! -s "$OUTPUT_PATH" ]; then
    echo "❌ ERROR: Failed to save Gemini's output to '$OUTPUT_PATH'."
    exit 1
fi
echo "✅ Saved Gemini's implementation plan to: $OUTPUT_PATH"

# Update PROJECT_STATUS.md with details from the new plan
# (A real implementation would use a script to parse and replace sections)
echo "Updating PROJECT_STATUS.md with new phase information..."
# (Logic to parse $GEMINI_RESPONSE for phase count, duration, etc., and update PROJECT_STATUS.md would go here)
echo "✅ Updated PROJECT_STATUS.md"

# Announce completion to the user
echo ""
echo "Next step: Run \`/phase-checklist-gemini-full 1 $INITIATIVE_PATH\` to have Gemini create the detailed Phase 1 checklist."
```

---

## ✅ **VERIFICATION CHECKLIST**

Before reporting completion, verify you have performed these steps:
-   [ ] Parsed the initiative path from arguments and loaded `plan.md`.
-   [ ] Successfully ran `repomix` to generate `repomix-output.xml`.
-   [ ] Created `./impl-prompt.md` with the correct XML structure.
-   [ ] Injected all dynamic context (`rd_plan_context`, `codebase_context`) into the prompt file.
-   [ ] **I EXECUTED the `gemini -p "@./impl-prompt.md"` command.** ← MANDATORY
-   [ ] I received Gemini's implementation plan response.
-   [ ] I saved the plan to the correct `implementation.md` file.
-   [ ] I updated `PROJECT_STATUS.md`.
