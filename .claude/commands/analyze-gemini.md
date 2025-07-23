# Command: /analyze-gemini-v3 [analysis_query] [optional_baseline_ref]

**Goal:** Get a comprehensive, code-aware analysis from Gemini on any given topic, feature, or problem, leveraging a fresh perspective and deep codebase context.

**Usage Examples:**
-   **Feature Planning:** `/analyze-gemini-v3 How should I implement a new caching layer for the API?`
-   **Refactoring:** `/analyze-gemini-v3 What is the best way to refactor the User model to be more modular? main`
-   **Security Audit:** `/analyze-gemini-v3 Analyze the authentication flow for potential security vulnerabilities.`
-   **Onboarding:** `/analyze-gemini-v3 Explain the data processing pipeline from end to end.`
-   **Code Review:** `/analyze-gemini-v3 Review the changes for the new feature branch and suggest improvements. v1.2.3`

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**THIS IS THE CORE PURPOSE OF THIS COMMAND:**
1.  You MUST run context analysis (including `git` analysis if a baseline is provided).
2.  You MUST build the complete, structured prompt in `analysis-prompt.md`.
3.  You MUST execute `gemini -p "@analysis-prompt.md"`.
4.  You MUST wait for and process Gemini's response.
5.  You MUST report Gemini's findings to the user.

**DO NOT:**
-   ❌ Stop after context analysis.
-   ❌ Provide your own analysis instead of running Gemini.
-   ❌ Skip the Gemini execution for ANY reason.

---

## 🤖 **CONTEXT: YOU ARE CLAUDE CODE**

You are Claude Code - the autonomous command-line tool that executes shell commands directly. You are an **orchestrator**, not the primary analyst. Your job is to prepare the context and delegate the deep thinking to Gemini.

---

## 📋 **YOUR EXECUTION WORKFLOW**

### Step 1: Assess Current Context & Parse Arguments

This step intelligently parses the user's full input to separate the multi-word query from the optional baseline reference.

```bash
# Ensure we have a tmp directory for our analysis files
mkdir -p ./tmp

# The special $ARGUMENTS variable contains the entire string typed by the user after the command.
# This is the ONLY reliable way to get the user's full input.
RAW_ARGS="$ARGUMENTS"

# Default values
ANALYSIS_QUERY=""
BASELINE_REF=""

if [ -z "$RAW_ARGS" ]; then
    echo "❌ ERROR: An analysis query is required."
    echo "Usage: /analyze-gemini-v3 <Your question here> [optional-git-ref]"
    exit 1
fi

# --- Intelligent Parsing Logic ---
# We assume the baseline ref, if present, is the LAST word and is a valid git ref.
# Everything else is part of the query.

# Get the last word from the arguments
LAST_WORD=$(echo "$RAW_ARGS" | awk '{print $NF}')

# Check if the last word is a valid git ref. This is the key verification step.
if git rev-parse --verify "$LAST_WORD" >/dev/null 2>&1; then
    # It's a valid ref, so we treat it as the baseline.
    BASELINE_REF="$LAST_WORD"
    # The query is everything *except* the last word.
    ANALYSIS_QUERY=$(echo "$RAW_ARGS" | sed 's/ [^ ]*$//')
    echo "✅ Query identified: \"$ANALYSIS_QUERY\""
    echo "✅ Baseline identified: \"$BASELINE_REF\""
else
    # The last word is not a valid ref, so the entire string is the query.
    ANALYSIS_QUERY="$RAW_ARGS"
    BASELINE_REF=""
    echo "✅ Query identified: \"$ANALYSIS_QUERY\""
    echo "ℹ️ No valid baseline ref found in query."
fi
```

### Step 2: Run Context Analysis (You Execute This)

This step gathers information about the codebase's current state and recent changes.

```bash
# --- GIT CONTEXT GATHERING ---
# This part is now conditional on a baseline being provided.

echo "## ANALYSIS CONTEXT" > ./tmp/analysis_context.txt
echo "Generated on: $(date)" >> ./tmp/analysis_context.txt

# Always get recent commit history
echo -e "\n## RECENT COMMITS" >> ./tmp/analysis_context.txt
git log -n 10 --pretty=format:"%h %ad - %s [%an]" --date=short >> ./tmp/analysis_context.txt

# Always get current status
echo -e "\n## CURRENT GIT STATUS" >> ./tmp/analysis_context.txt
git status --porcelain >> ./tmp/analysis_context.txt

# Only perform diff if a baseline was provided
if [ -n "$BASELINE_REF" ]; then
    echo "✅ Using provided baseline for comparison: $BASELINE_REF"
    echo -e "\n## BASELINE USED: $BASELINE_REF" >> ./tmp/analysis_context.txt
    echo -e "\n## DIFF STATISTICS (from $BASELINE_REF to HEAD)" >> ./tmp/analysis_context.txt
    git diff "$BASELINE_REF"..HEAD --stat >> ./tmp/analysis_context.txt
    echo -e "\n## DETAILED CODE CHANGES" >> ./tmp/analysis_context.txt
    git diff "$BASELINE_REF"..HEAD -- ptycho/ src/ configs/ | head -2000 >> ./tmp/analysis_context.txt
else
    echo "ℹ️ No baseline provided. Skipping diff analysis."
fi

# --- CODEBASE CONTEXT GATHERING --- you MUST run this whether or not repomix-output.xml already exists
npx repomix@latest . --include "**/*.sh,**/*.md,**/*.py,**/*.c,**/*.h,**/*.json,**/*.log" --ignore ".aider.chat.history.md,build/**,tests/**"

echo "✅ Context gathering complete."
```

### Step 3: MANDATORY - Execute Gemini Analysis

**🔴 STOP - THIS STEP IS MANDATORY - DO NOT SKIP**

#### Step 3.1: Build Prompt File
You MUST now populate this generalized command template and save it to `analysis-prompt.md`.

```bash
# Clean start for the prompt file
rm -f analysis-prompt.md 2>/dev/null

# Create the structured prompt with placeholders
cat > analysis-prompt.md << 'PROMPT'
<task>
Perform a comprehensive, code-aware analysis based on the user's request, providing a fresh and expert perspective.

Carry out the following steps:
<steps>
<1>
**Review Project Documentation:** Thoroughly read `CLAUDE.md` and `DEVELOPER_GUIDE.md` (if present in the codebase context) to understand project architecture, conventions, and goals.
</1>
<2>
**Analyze the Request:** Carefully review the user's request in the `<analysis_request>` section.
</2>
<3>
**Synthesize All Context:** Correlate the user's request with the provided `<git_context>` (recent changes and diffs) and the full `<codebase_context>` (the entire codebase).
</3>
<4>
**Formulate a Response:** Based on your complete understanding, generate a detailed analysis that directly addresses the user's request, following the specified `<output_format>`.
</4>
</steps>

<analysis_request>
## PRIMARY GOAL / QUESTION
[Placeholder for the user's main analysis query]
</analysis_request>

<git_context>
<!-- Placeholder for content from analysis_context.txt -->
</git_context>

<codebase_context>
<!-- Placeholder for content from repomix-output.xml -->
</codebase_context>

<guidelines>
## GUIDELINES FOR ANALYSIS
1.  **Think from First Principles:** Do not just accept the user's premises. Analyze the code to form your own expert opinion.
2.  **Consider Alternatives:** What are other ways to achieve the goal? What are the trade-offs (performance, security, maintainability)?
3.  **Identify Risks & Dependencies:** What could go wrong? What other parts of the system will be affected?
4.  **Cite Your Evidence:** Refer to specific files, functions, and code patterns from the provided context to support your conclusions.
</guidelines>

<output format>
Please provide your analysis in the following structure:

1.  **Executive Summary:** A brief, high-level answer to the user's primary question.
2.  **Detailed Analysis:** A thorough breakdown of your findings. This is the main body of your response. Use code snippets and file references.
3.  **Actionable Recommendations / Plan:** A concrete set of next steps. This could be an implementation plan, a list of refactoring tasks, or specific commands to run.
4.  **Potential Risks & Considerations:** A list of potential pitfalls, trade-offs, or dependencies to be aware of.
</output format>
</task>
PROMPT
```

#### Step 3.2: Inject Dynamic Content and Execute
You MUST now EXECUTE the following shell commands:

```bash
# Inject the dynamic content into the prompt file
# Using a temp file for the query handles special characters and multi-line input safely.
echo "$ANALYSIS_QUERY" > ./tmp/query.txt
sed -i.bak -e '/\[Placeholder for the user.s main analysis query\]/r ./tmp/query.txt' -e '//d' analysis-prompt.md

# Append the git context
echo -e "\n<git_context>" >> analysis-prompt.md
cat ./tmp/analysis_context.txt >> analysis-prompt.md
echo -e "\n</git_context>" >> analysis-prompt.md

# Append the codebase context
echo -e "\n<codebase_context>" >> analysis-prompt.md
cat repomix-output.xml >> analysis-prompt.md
echo -e "\n</codebase_context>" >> analysis-prompt.md

# Execute the command
gemini -p "@analysis-prompt.md"
```

### Step 4: Process and Report GEMINI'S Findings

After Gemini responds, you will synthesize its findings into a clear report for the user.

```markdown
## 🎯 Expert Analysis from Gemini

Based on your query, Gemini performed a deep analysis of the codebase. Here are its findings:

### Executive Summary
[Gemini's summary of the issue/plan]

### Detailed Analysis
[Gemini's detailed breakdown, including code snippets and file references]

### Action Plan (Based on Gemini's Recommendations)
1.  [First concrete step recommended by Gemini]
2.  [Second concrete step recommended by Gemini]
3.  [Third concrete step recommended by Gemini]

### Risks & Considerations
-   [A risk identified by Gemini]
-   [A trade-off highlighted by Gemini]
```
