# Command: /geminictx [query]

**Goal:** Leverage a two-pass AI workflow to provide a comprehensive, context-aware answer to a user's query about the codebase. Pass 1 uses Gemini to identify relevant files, and Pass 2 uses your own (Claude's) synthesis capabilities on the full content of those files.

**Usage:**
- `/geminictx "how does authentication work?"`
- `/geminictx "explain the data loading pipeline"`

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**This command follows a deliberate, non-negotiable two-pass workflow:**
1.  **Context Aggregation:** You MUST first run `repomix` to create a complete snapshot of the codebase.
2.  **Pass 1 (Gemini as Context Locator):** You MUST build a structured prompt file and execute `gemini -p` to identify a list of relevant files based on the user's query and the `repomix` context.
3.  **Pass 2 (Claude as Synthesizer):** You MUST then read the full content of EVERY file Gemini identified to build your own deep context before providing a synthesized answer.

**DO NOT:**
-   ❌ Skip the `repomix` step. The entire workflow depends on this complete context.
-   ❌ Guess which files are relevant. You must delegate this to Gemini.
-   ❌ Only read Gemini's one-sentence justifications. You must read the **full file contents**.
-   ❌ Answer the user's query before you have completed Pass 1 and read all identified files in Pass 2.

---

## 🤖 **YOUR EXECUTION WORKFLOW**

### Step 1: Gather Codebase Context with Repomix

First, create a comprehensive and reliable context snapshot of the entire project.

```bash
# The user's query is passed as $ARGUMENTS
USER_QUERY="$ARGUMENTS"

# Use repomix for a complete, single-file context snapshot.
# This is more robust than a long list of @-references.
npx repomix@latest . \
  --include "**/*.{js,py,md,sh,json,c,h,log}" \
  --ignore "build/**,node_modules/**,dist/**,*.lock"

# Verify that the context was created successfully.
if [ ! -s ./repomix-output.xml ]; then
    echo "❌ ERROR: Repomix failed to generate the codebase context. Aborting."
    exit 1
fi

echo "✅ Codebase context aggregated into repomix-output.xml."
```

### Step 2: Build and Execute Pass 1 (Gemini as Context Locator)

Now, build a structured prompt in a file to ask Gemini to find the relevant files.

#### Step 2.1: Build the Prompt File
```bash
# Clean start for the prompt file
rm -f ./gemini-pass1-prompt.md 2>/dev/null

# Create the structured prompt using the v3.0 XML pattern
cat > ./gemini-pass1-prompt.md << 'PROMPT'
<task>
You are a **Context Locator**. Your sole purpose is to analyze the provided codebase context and identify the most relevant files for answering the user's query. Do not answer the query yourself.

<steps>
<1>
Analyze the user's `<query>`.
</1>
<2>
Scan the entire `<codebase_context>` to find all files (source code, documentation, configs) that are relevant to the query.
</2>
<3>
For each relevant file you identify, provide your output in the strict format specified in `<output_format>`.
</3>
</steps>

<context>
<query>
[Placeholder for the user's query]
</query>

<codebase_context>
<!-- Placeholder for content from repomix-output.xml -->
</codebase_context>
</context>

<output_format>
Your output must be a list of entries. Each entry MUST follow this exact format, ending with three dashes on a new line.

FILE: [exact/path/to/file.ext]
RELEVANCE: [A concise, one-sentence explanation of why this file is relevant.]
---

Do not include any other text, conversation, or summaries in your response.
</output_format>
</task>
PROMPT
```

#### Step 2.2: Append Dynamic Context
```bash
# Inject the user's query and the repomix context into the prompt file.
# Using a temp file for the query handles special characters safely.
echo "$USER_QUERY" > ./tmp/user_query.txt
sed -i.bak -e '/\[Placeholder for the user.s query\]/r ./tmp/user_query.txt' -e '//d' ./gemini-pass1-prompt.md

# Append the codebase context
echo -e "\n<codebase_context>" >> ./gemini-pass1-prompt.md
cat ./repomix-output.xml >> ./gemini-pass1-prompt.md
echo -e "\n</codebase_context>" >> ./gemini-pass1-prompt.md

echo "✅ Built structured prompt for Pass 1: ./gemini-pass1-prompt.md"
```

#### Step 2.3: Execute Gemini
```bash
# Execute Gemini with the single, clean prompt file.
gemini -p "@./gemini-pass1-prompt.md"
```

### Step 3: Process Gemini's Response & Prepare for Pass 2

After receiving the list of files from Gemini, parse the output and prepare to read the files.

```bash
# [You will receive Gemini's response, e.g., captured in $GEMINI_RESPONSE]
# For this example, we'll simulate parsing the response to get a file list.

# Parse the output to get a clean list of file paths.
# This is a robust way to extract just the file paths for the next step.
FILE_LIST=$(echo "$GEMINI_RESPONSE" | grep '^FILE: ' | sed 's/^FILE: //')

# Verify that Gemini returned relevant files.
if [ -z "$FILE_LIST" ]; then
    echo "⚠️ Gemini did not identify any specific files for your query. I will attempt to answer based on general project knowledge, but the answer may be incomplete."
    # You might choose to exit here or proceed with caution.
    exit 0
fi

echo "Gemini identified the following relevant files:"
echo "$FILE_LIST"
```

### Step 4: Execute Pass 2 (Claude as Synthesizer)

This is your primary role. Read the full content of the identified files to build deep context.

```bash
# Announce what you are doing for transparency.
echo "Now reading the full content of each identified file to build a deep understanding..."

# You will now iterate through the FILE_LIST and read each one.
# For each file in FILE_LIST:
#   - Verify the file exists (e.g., if [ -f "$file" ]; then ...).
#   - Read its full content into your working memory.
#   - Announce: "Reading: `path/to/file.ext`..."

# After reading all files, you are ready to synthesize the answer.
```

### Step 5: Present Your Synthesized Analysis

Your final output to the user should follow the well-structured format from your original prompt.

```markdown
Based on your query, Gemini identified the following key files, which I have now read and analyzed in their entirety:

-   `path/to/relevant/file1.ext`
-   `path/to/relevant/file2.ext`
-   `docs/relevant_guide.md`

Here is a synthesized analysis of how they work together to address your question.

### Summary
[Provide a 2-3 sentence, high-level answer to the user's query based on your comprehensive analysis of the files.]

### Detailed Breakdown

#### **Core Logic in `path/to/relevant/file1.ext`**
[Explain the role of this file. Reference specific functions or classes you have read.]

**Key Code Snippet:**
\`\`\`[language]
[Quote a critical code block from the file that you have read.]
\`\`\`

#### **Workflow Orchestration in `path/to/relevant/file2.ext`**
[Explain how this file uses or connects to the core logic from the first file.]

**Key Code Snippet:**
\`\`\`[language]
[Quote a relevant snippet showing the interaction.]
\`\`\`

### How It All Connects
[Provide a brief narrative explaining the data flow or call chain between the identified components.]

### Conclusion
[End with a concluding thought or a question to guide the user's next step.]
```
