# Git Diff Analysis Prompt

## Agent Instructions

Consider this list of file paths:
files="`$ARGUMENTS`"

Collect all git diffs and analyze them with Gemini in a single batch.

### Step 1: Collect All Diffs

```bash
# Create a combined diff file
for file in $files; do
    echo "=== FILE: $file ==="
    git diff "$file" || git diff --cached "$file"
    echo -e "\n=== END FILE: $file ===\n"
done > /tmp/all_diffs.txt
```

### Step 2: Send to Gemini for Analysis

```bash
gemini -p "@/tmp/all_diffs.txt Analyze these git diffs as a code reviewer.

For EACH file, determine if the changes are an improvement or worsening based on:
- Code quality and readability
- Performance impact
- Maintainability
- Security implications
- Bug risk
- Best practices

Format your response EXACTLY as:

FILE: [filename]
VERDICT: [IMPROVEMENT|WORSENING|NEUTRAL]
REASON: [1-2 sentence explanation]
---

After analyzing all files, provide:

OVERALL ASSESSMENT:
- Total files: [count]
- Improvements: [count]
- Worsenings: [count]
- Net impact: [POSITIVE|NEGATIVE|MIXED]
- Recommendation: [COMMIT|REVIEW_FIRST|DO_NOT_COMMIT]
- Key concerns: [list any critical issues]"
```

### Step 3: Parse and Display Results

Extract verdicts for each file and color-code the output:
- 🟢 IMPROVEMENT (green)
- 🔴 WORSENING (red)
- 🟡 NEUTRAL (yellow)

### Complete One-Liner Version

```bash
{ for f in $files; do echo "=== $f ==="; git diff "$f" || git diff --cached "$f"; done; } | \
gemini -p "$(cat -) 

Rate each file's diff: IMPROVEMENT/WORSENING/NEUTRAL with reason.
Give overall: COMMIT/REVIEW_FIRST/DO_NOT_COMMIT"
```

### Expected Gemini Output Format

```
FILE: src/main.py
VERDICT: IMPROVEMENT
REASON: Refactored complex function into smaller, testable units.
---

FILE: src/utils.py
VERDICT: WORSENING
REASON: Removed error handling that could cause runtime failures.
---

FILE: tests/test_main.py
VERDICT: IMPROVEMENT
REASON: Added comprehensive test coverage for edge cases.
---

OVERALL ASSESSMENT:
- Total files: 3
- Improvements: 2
- Worsenings: 1
- Net impact: POSITIVE
- Recommendation: REVIEW_FIRST
- Key concerns: Missing error handling in utils.py needs attention before commit.
```
