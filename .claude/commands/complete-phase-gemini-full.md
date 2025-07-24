# Command: /complete-phase-gemini-full <initiative-path>

**Goal:** Autonomously verify the completion of the current project phase, report the verdict, and prepare for the next phase by delegating all analysis and decision-making to Gemini.

**Usage:**
- `/complete-phase-gemini-full plans/active/real-time-notifications`

**Prerequisites:**
- An `implementation.md` and `PROJECT_STATUS.md` must exist.
- The command should be run after a phase's checklist is believed to be complete.

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**YOUR ROLE IS AN AUTONOMOUS ORCHESTRATOR AND FILE MANAGER. YOU MAKE NO DECISIONS.**
1.  You MUST identify the current phase and its success criteria from the project files.
2.  You MUST run `repomix` to create a complete, fresh snapshot of the codebase, including the recent changes.
3.  You MUST build a structured prompt file (`verify-prompt.md`) using the XML format.
4.  You MUST execute `gemini -p "@verify-prompt.md"` to delegate the entire verification process.
5.  You MUST parse Gemini's verdict from the response.
6.  You MUST execute the correct file management actions (advance phase or create fix-list) based **only** on Gemini's verdict.

**DO NOT:**
-   ❌ Make any judgment calls on whether a phase is complete.
-   ❌ Modify, interpret, or enhance Gemini's analysis.
-   ❌ Skip any step. The workflow is non-negotiable.
-   ❌ Proceed to the next phase if Gemini's verdict is anything other than `COMPLETE`.

---

## 🤖 **YOUR EXECUTION WORKFLOW**

### Step 1: Prepare Context from Project State

Parse arguments and load all necessary context from the project's planning and status documents.

```bash
# Parse arguments
INITIATIVE_PATH="$1"
IMPLEMENTATION_PLAN_PATH="$INITIATIVE_PATH/implementation.md"
PROJECT_STATUS_PATH="PROJECT_STATUS.md" # Assuming it's in the root

# Verify required files exist
if [ ! -f "$IMPLEMENTATION_PLAN_PATH" ] || [ ! -f "$PROJECT_STATUS_PATH" ]; then
    echo "❌ ERROR: Required project files (implementation.md or PROJECT_STATUS.md) not found."
    exit 1
fi

# Extract current phase number and info from project files
# This requires a robust parsing method (e.g., awk, sed, or a script)
CURRENT_PHASE_NUMBER=$(grep 'Current Phase:' "$PROJECT_STATUS_PATH" | sed 's/Current Phase: Phase \([0-9]*\).*/\1/')
CURRENT_PHASE_INFO=$(awk "/## Phase $CURRENT_PHASE_NUMBER/{f=1;p=1} /## Phase/{if(!p){f=0}; p=0} f" "$IMPLEMENTATION_PLAN_PATH")
# Also extract the checklist for the current phase
CURRENT_PHASE_CHECKLIST=$(cat "$INITIATIVE_PATH/phase_${CURRENT_PHASE_NUMBER}_checklist.md")

if [ -z "$CURRENT_PHASE_NUMBER" ] || [ -z "$CURRENT_PHASE_INFO" ]; then
    echo "❌ ERROR: Could not determine current phase from project files."
    exit 1
fi

echo "✅ Loaded context for current Phase $CURRENT_PHASE_NUMBER."
```

### Step 2: Aggregate Codebase Context with Repomix

Create a comprehensive snapshot of the project's current state, including the newly implemented changes.

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
rm -f ./verify-prompt.md 2>/dev/null

# Create the structured prompt with placeholders using the v3.0 XML pattern
cat > ./verify-prompt.md << 'PROMPT'
<task>
You are an automated, rigorous Quality Assurance and Verification system. Your task is to perform a complete verification of a software development phase and determine if it is complete. You must be strict and objective.

<steps>
<1>
Analyze the provided context: `<phase_info>`, `<phase_checklist>`, and the full `<codebase_context>`.
</1>
<2>
Perform all verification checks as detailed in the `<output_format>` section. This includes implementation review, test analysis, integration checks, and quality checks.
</2>
<3>
Execute the success test and compare the actual output to the expected output.
</3>
<4>
Provide a definitive, overall verdict: `COMPLETE` or `INCOMPLETE`. This is the most critical part of your output.
</4>
<5>
If the verdict is `INCOMPLETE`, provide a list of all `BLOCKER` issues.
</5>
<6>
If the verdict is `COMPLETE`, provide a detailed preparation plan for the next phase.
</6>
</steps>

<context>
<phase_info>
[Placeholder for the current phase info from implementation.md]
</phase_info>

<phase_checklist>
[Placeholder for the current phase's checklist.md]
</phase_checklist>

<codebase_context>
<!-- Placeholder for content from repomix-output.xml -->
</codebase_context>
</context>

<output_format>
Your entire response must be a single Markdown block.
The most important line of your output MUST be `OVERALL_VERDICT: [COMPLETE|INCOMPLETE]`.
Do not include any conversational text before or after your analysis.

OVERALL_VERDICT: [COMPLETE|INCOMPLETE]

## 1. IMPLEMENTATION VERIFICATION
...
[The entire detailed Markdown template from the original prompt goes here, verbatim.]
...
## 8. GEMINI VERIFICATION SUMMARY
...
END OF VERIFICATION
</output_format>
</task>
PROMPT
```

#### Step 3.2: Append Dynamic Context to the Prompt File
```bash
# Inject all context into the prompt file.
# Using temp files handles multi-line variables and special characters safely.
echo "$CURRENT_PHASE_INFO" > ./tmp/phase_info.txt
echo "$CURRENT_PHASE_CHECKLIST" > ./tmp/phase_checklist.txt

sed -i.bak -e '/\[Placeholder for the current phase info from implementation.md\]/r ./tmp/phase_info.txt' -e '//d' ./verify-prompt.md
sed -i.bak -e '/\[Placeholder for the current phase.s checklist.md\]/r ./tmp/phase_checklist.txt' -e '//d' ./verify-prompt.md

# Append the codebase context
echo -e "\n<codebase_context>" >> ./verify-prompt.md
cat ./repomix-output.xml >> ./verify-prompt.md
echo -e "\n</codebase_context>" >> ./verify-prompt.md

echo "✅ Successfully built structured prompt file: ./verify-prompt.md"
```

### Step 4: MANDATORY - Execute Gemini Verification

You MUST now execute Gemini using the single, clean, and verifiable prompt file.

```bash
# Execute Gemini with the fully-formed prompt file
gemini -p "@./verify-prompt.md"
```

### Step 5: Process Gemini's Verdict and Manage Files

Your final role: parse Gemini's verdict and execute the corresponding file operations.

```bash
# [You will receive Gemini's verification report as a response from the command above]
# For this example, we'll assume the response is captured into $GEMINI_RESPONSE.

# Parse the verdict from the first line of the response. This is reliable.
VERDICT=$(echo "$GEMINI_RESPONSE" | grep '^OVERALL_VERDICT: ' | sed 's/^OVERALL_VERDICT: //')
REPORT_CONTENT=$(echo "$GEMINI_RESPONSE" | sed '1d') # Get the rest of the content

if [ "$VERDICT" == "COMPLETE" ]; then
    echo "✅ Phase $CURRENT_PHASE_NUMBER VERIFIED COMPLETE by Gemini."
    echo "$REPORT_CONTENT" # Display the full successful report

    # Update implementation.md and PROJECT_STATUS.md
    # (Logic to mark phase complete and update status would go here)
    echo "Updating project tracking files..."

    # Save next phase prep if it exists
    # (Logic to parse "NEXT PHASE PREPARATION" section and save to phase_<n+1>_prep.md)

    # Check if this was the final phase
    # (Logic to check implementation.md for more phases)
    # If final, archive the project.
    # If not final, announce next step.
    echo "Next step: Run \`/phase-checklist-gemini-full $((CURRENT_PHASE_NUMBER + 1)) $INITIATIVE_PATH\` to generate the detailed checklist for the next phase."

elif [ "$VERDICT" == "INCOMPLETE" ]; then
    echo "❌ Phase $CURRENT_PHASE_NUMBER verification FAILED."
    
    # Extract and save the list of blockers
    BLOCKERS=$(echo "$REPORT_CONTENT" | awk '/## 1. IMPLEMENTATION VERIFICATION/,/## 2. TEST VERIFICATION/' | grep 'BLOCKER')
    FIX_LIST_PATH="$INITIATIVE_PATH/phase_${CURRENT_PHASE_NUMBER}_fixes.md"
    echo "# Phase $CURRENT_PHASE_NUMBER Fix-List (Blockers Only)" > "$FIX_LIST_PATH"
    echo "Generated on $(date)" >> "$FIX_LIST_PATH"
    echo "$BLOCKERS" >> "$FIX_LIST_PATH"

    echo "Gemini found BLOCKERS that must be fixed:"
    echo "$BLOCKERS"
    echo ""
    echo "A detailed fix-list has been saved to: $FIX_LIST_PATH"
    echo "After fixing all blockers, run this command again to re-verify."

else
    echo "❌ ERROR: Could not determine phase verdict from Gemini's output."
    echo "--- Gemini's Raw Output ---"
    echo "$GEMINI_RESPONSE"
    exit 1
fi
```
