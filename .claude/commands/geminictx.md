### Refined Command: `/geminictx`

<file path=".claude/commands/geminictx.md">
# Command: /geminictx [query]

**Goal:** Use Gemini to identify the most relevant files for a given query, then have Claude read and synthesize that context to provide a comprehensive answer.

**Usage:**
- `/geminictx how does authentication work?`
- `/geminictx explain the data loading pipeline`

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**This command follows a deliberate two-pass workflow:**
1.  **Pass 1 (Gemini as Identifier):** You MUST execute a `gemini -p` command to have it analyze the codebase and return a list of relevant files.
2.  **Pass 2 (Claude as Synthesizer):** You MUST then read the full content of EVERY file Gemini identified to build your own deep context before providing a synthesized answer.

**DO NOT:**
-   ❌ Skip the Gemini step and guess which files are relevant.
-   ❌ Only read the justifications from Gemini; you must read the full file contents.
-   ❌ Present an answer before you have read all the identified files.

---

## 🤖 **YOUR EXECUTION WORKFLOW**

### Step 1: Execute Gemini to Identify Relevant Context

You will execute the following command, which includes default context paths and the user's query.

```bash
# The user's query is passed as $ARGUMENTS
USER_QUERY="$ARGUMENTS"

gemini -p "@src/ @ptycho/ @scripts/ @docs/ <query>
$USER_QUERY
</query>

<task>
Based on the user's query, your task is to act as a context locator. Analyze the provided directories and identify the most relevant files (source code, documentation, configs, etc.) to answer the query.

For each relevant file you identify, provide your output in the following strict format:

FILE: [exact/path/to/file.ext]
RELEVANCE: [A concise, one-sentence explanation of why this file is relevant to the user's query.]
---
</task>"
```

### Step 2: Read and Synthesize the Identified Files

After receiving the list of files from Gemini, you will perform the following actions:

1.  **Parse the Output:** Extract the list of file paths from Gemini's response.
2.  **Read Full Content:** Read the entire content of each identified file into your working context. Announce which files you are reading for transparency.
3.  **Synthesize and Answer:** After reading all files, construct your final response to the user, following the structure below.

### Step 3: Present Your Synthesized Analysis

Your final output to the user should be structured as follows:

```markdown
Based on your query, Gemini identified the following key files, which I have now read and analyzed:

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

#### **Documentation in `docs/relevant_guide.md`**
[Reference the official documentation, explaining how it clarifies the implementation.]

### How It All Connects
[Provide a brief narrative explaining the data flow or call chain between the identified components. For example: "The process starts in `file2.py`, which calls the `core_function()` from `file1.py` using parameters defined in the documentation..."]

### Conclusion
[End with a concluding thought or a question to guide the user's next step, e.g., "This covers the authentication flow. Would you like to dive deeper into the JWT validation logic specifically?"]
```

---

### Why This Two-Pass Approach is Effective

This workflow is deliberately designed to play to the strengths of both models and provide the most robust context for you, the agent:

1.  **Comprehensive Discovery:** Gemini's large context window scans the entire relevant codebase to act as an intelligent "grep," ensuring no key files are missed.
2.  **Focused Context for Claude:** You receive a curated list of the most important files. By reading their full content, you load a dense, relevant, and manageable context into your working memory.
3.  **Deep Synthesis:** With the full text of the key files available, you can provide a much deeper and more accurate synthesis than if you were relying only on Gemini's brief summaries. This is critical for complex follow-up tasks.
4.  **Transparency:** The user sees exactly which files were identified as relevant, making the process clear and auditable.
