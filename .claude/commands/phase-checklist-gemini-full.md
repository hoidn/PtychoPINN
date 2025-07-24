# Command: /phase-checklist-gemini-full <phase-number> <initiative-path>

**Goal:** Autonomously generate a complete, highly-detailed, code-aware implementation checklist for a specific project phase using Gemini for analysis and generation.

**Usage:**
- `/phase-checklist-gemini-full 2 initiatives/my-project`

**Prerequisites:**
- An `implementation.md` file must exist at the specified `<initiative-path>`.
- The file must contain a clearly defined section for the given `<phase-number>`.

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**YOUR ROLE IS TO ORCHESTRATE, NOT TO AUTHOR.**
1.  You MUST parse the phase information from the specified `implementation.md` file.
2.  You MUST run `repomix` to create a complete, fresh snapshot of the codebase context.
3.  You MUST build a structured prompt file (`checklist-prompt.md`) using the XML format.
4.  You MUST execute `gemini -p "@checklist-prompt.md"` to delegate the checklist generation.
5.  You MUST save Gemini's response **exactly as provided** to the correct output file.

**DO NOT:**
-   ❌ Modify, interpret, or add comments to Gemini's output. You are a file manager.
-   ❌ Create the checklist yourself. Your job is to run the process.
-   ❌ Skip any step, especially the `repomix` context aggregation or the `gemini` execution.

---

## 🤖 **YOUR EXECUTION WORKFLOW**

### Step 1: Prepare Context from Local Files

Parse arguments and extract the necessary information from the project's planning documents.

```bash
# Parse arguments
PHASE_NUMBER="$1"
INITIATIVE_PATH="$2"
IMPLEMENTATION_PLAN_PATH="$INITIATIVE_PATH/implementation.md"

# Verify the implementation plan exists
if [ ! -f "$IMPLEMENTATION_PLAN_PATH" ]; then
    echo "❌ ERROR: Implementation plan not found at '$IMPLEMENTATION_PLAN_PATH'."
    exit 1
fi

# Extract the entire section for the specified phase from the markdown file.
# This requires a robust parsing method (e.g., awk, sed, or a script)
# to grab all content between "## Phase [N]" and the next "## Phase".
# For this example, we'll assume the content is extracted into a variable.
PHASE_INFO_CONTENT=$(awk "/## Phase $PHASE_NUMBER/{f=1;p=1} /## Phase/{if(!p){f=0}; p=0} f" "$IMPLEMENTATION_PLAN_PATH")

if [ -z "$PHASE_INFO_CONTENT" ]; then
    echo "❌ ERROR: Could not find or extract content for Phase $PHASE_NUMBER in '$IMPLEMENTATION_PLAN_PATH'."
    exit 1
fi

echo "✅ Successfully extracted info for Phase $PHASE_NUMBER."
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

You will now build the prompt for Gemini in a file using the structured XML pattern. This replaces the large, inline prompt string.

#### Step 3.1: Create Base Prompt File
```bash
# Clean start for the prompt file
rm -f ./checklist-prompt.md 2>/dev/null

# Create the structured prompt with placeholders using the v3.0 XML pattern
cat > ./checklist-prompt.md << 'PROMPT'
<task>
You are an expert software engineer and project manager. Your task is to create a complete, ultra-detailed, step-by-step implementation checklist for a given project phase.

The checklist must be so detailed that a developer can execute it by copying and pasting code and commands directly. You must analyze the provided codebase context to inform your guidance, referencing existing patterns, APIs, and potential gotchas.

<steps>
<1>
Analyze the `<phase_info>` to understand the goals, deliverables, and success criteria for this phase.
</1>
<2>
Thoroughly analyze the entire `<codebase_context>` to understand existing code patterns, APIs, and testing frameworks.
</2>
<3>
Generate the complete checklist, populating all dynamic sections ([Gemini: ...]) based on your analysis. The final output must strictly adhere to the Markdown table format specified in `<output_format>`.
</3>
</steps>

<context>
<phase_info>
[Placeholder for the Phase N section from implementation.md]
</phase_info>

<codebase_context>
<!-- Placeholder for content from repomix-output.xml -->
</codebase_context>
</context>

<output_format>
Your entire response must be a single Markdown block containing the checklist. Do not include any conversational text before or after the checklist. The checklist format is non-negotiable.

[The entire detailed Markdown table structure from the original prompt goes here, verbatim. It's an excellent template.]

# Phase [N]: [Phase Name] Checklist
...
| ID  | Task Description                                   | State | How/Why & API Guidance                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          -
...
## 📊 Gemini Analysis Metadata
...
END OF CHECKLIST

</output_format>
</task>
PROMPT
```

#### Step 3.2: Append Dynamic Context to the Prompt File
```bash
# Inject the phase information and the repomix context into the prompt file.
# This is a critical step that replaces the placeholders.

# Using a temp file for the phase info handles multi-line variables and special characters safely.
echo "$PHASE_INFO_CONTENT" > ./tmp/phase_info.txt
sed -i.bak -e '/\[Placeholder for the Phase N section from implementation.md\]/r ./tmp/phase_info.txt' -e '//d' ./checklist-prompt.md

# Append the codebase context
echo -e "\n<codebase_context>" >> ./checklist-prompt.md
cat ./repomix-output.xml >> ./checklist-prompt.md
echo -e "\n</codebase_context>" >> ./checklist-prompt.md

# Verify the prompt file was created correctly
if [ ! -s ./checklist-prompt.md ]; then
    echo "❌ ERROR: Failed to build the prompt file ./checklist-prompt.md. Aborting."
    exit 1
fi
echo "✅ Successfully built structured prompt file: ./checklist-prompt.md"
```

### Step 4: MANDATORY - Execute Gemini Analysis

You MUST now execute Gemini using the single, clean, and verifiable prompt file.

```bash
# Execute Gemini with the fully-formed prompt file
gemini -p "@./checklist-prompt.md"
```

### Step 5: Save Gemini's Checklist

Your final role: receive the output from your command and save it without modification.

```bash
# [You will receive Gemini's markdown checklist as a response from the command above]
# For this example, we'll assume the response is captured into $GEMINI_RESPONSE.

# Define the output path
OUTPUT_PATH="$INITIATIVE_PATH/phase_${PHASE_NUMBER}_checklist.md"

# Save the checklist exactly as received.
echo "$GEMINI_RESPONSE" > "$OUTPUT_PATH"

# Verify the file was saved
if [ ! -s "$OUTPUT_PATH" ]; then
    echo "❌ ERROR: Failed to save Gemini's output to '$OUTPUT_PATH'."
    exit 1
fi

# Announce completion to the user
echo "✅ Saved Gemini's complete Phase $PHASE_NUMBER checklist to: $OUTPUT_PATH"
echo ""
echo "The checklist is ready for execution and contains highly detailed, code-aware tasks."
```

---

## ✅ **VERIFICATION CHECKLIST**

Before reporting completion, verify you have performed these steps:
-   [ ] Parsed phase info from `implementation.md`.
-   [ ] Successfully ran `repomix` to generate `repomix-output.xml`.
-   [ ] Created `./checklist-prompt.md` with the correct XML structure.
-   [ ] Injected all dynamic context (`phase_info`, `codebase_context`) into the prompt file.
-   [ ] **I EXECUTED the `gemini -p "@./checklist-prompt.md"` command.** ← MANDATORY
-   [ ] I received Gemini's checklist response.
-   [ ] I saved the response **unmodified** to the correct output file.
