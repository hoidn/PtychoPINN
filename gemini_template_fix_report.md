# Gemini Command Template Fix Report

## Overview
This document describes the issues found in the Gemini command template within `.claude/commands/generate-agent-checklist-v2.md` and how they were resolved.

## Issues Identified

### 1. Complex Template Substitution Pattern
**Problem**: The original approach created a template with placeholders that required complex `sed` substitution:
```bash
# Original problematic approach
cat > ./doc-plan-prompt.md << 'PROMPT'
<user_objective>
[Placeholder for the user's objective]
</user_objective>

<codebase_context>
<!-- Placeholder for content from repomix-output.xml -->
</codebase_context>
PROMPT

# Complex sed substitution attempts
sed -i.bak -e '/\[Placeholder for the user.s objective\]/r ./tmp/user_objective.txt' -e '//d' ./doc-plan-prompt.md
```

**Issues**:
- Typo in placeholder text ("user.s" instead of "user's")
- Error-prone sed pattern matching
- Required creating temporary files in non-existent directories
- Risk of substitution failures if placeholders weren't on their own lines

### 2. XML Structure Corruption
**Problem**: The approach would append content after closing XML tags:
```bash
# This would append after </context>, breaking XML structure
echo -e "\n<codebase_context>" >> ./doc-plan-prompt.md
cat ./repomix-output.xml >> ./doc-plan-prompt.md
echo -e "\n</codebase_context>" >> ./doc-plan-prompt.md
```

### 3. Missing Variable Capture
**Problem**: The Gemini command output wasn't properly captured:
```bash
# Original - output goes to stdout, not captured
gemini -p "@./doc-plan-prompt.md"

# Later assumes $GEMINI_RESPONSE exists, but it was never populated
awk '/---PRIORITIZED_MODULES_START---/,/---PRIORITIZED_MODULES_END---/' <<< "$GEMINI_RESPONSE"
```

### 4. Directory Structure Assumptions
**Problem**: Created files in `./tmp/` without ensuring the directory exists:
```bash
echo "$ARGUMENTS" > ./tmp/user_objective.txt  # ./tmp/ might not exist
```

## Solution: Append-Only Approach

### Key Improvements

1. **Sequential File Building**: Build the prompt file incrementally without placeholders:
```bash
# Start the file
cat > ./doc-plan-prompt.md << 'PROMPT'
<task>
...
<user_objective>
PROMPT

# Append user objective directly
echo "$ARGUMENTS" >> ./doc-plan-prompt.md

# Continue appending structure
cat >> ./doc-plan-prompt.md << 'PROMPT'
</user_objective>

<codebase_context>
PROMPT

# Append repomix output
cat ./repomix-output.xml >> ./doc-plan-prompt.md

# Close structure
cat >> ./doc-plan-prompt.md << 'PROMPT'
</codebase_context>
...
PROMPT
```

2. **Proper Output Capture**:
```bash
# Capture Gemini response into variable
GEMINI_RESPONSE=$(gemini -p "@./doc-plan-prompt.md")

# Save for debugging
echo "$GEMINI_RESPONSE" > ./gemini_response_raw.txt
```

3. **No Temporary Directories**: All files are created in the current directory.

4. **Clear XML Structure**: The append-only approach maintains proper XML nesting naturally.

## Benefits of the Fix

1. **Reliability**: No complex pattern matching or substitution that could fail
2. **Debuggability**: Each step is clear and the intermediate states can be inspected
3. **Simplicity**: The flow is linear and easy to understand
4. **Robustness**: No assumptions about directory structure or placeholder formats
5. **Maintainability**: Adding new sections is straightforward - just append more content

## Verification

To verify the fix works correctly:
1. The generated `doc-plan-prompt.md` should have proper XML structure
2. The `$ARGUMENTS` value should appear in the correct location
3. The `repomix-output.xml` content should be properly embedded
4. The Gemini response should be captured in both the variable and `gemini_response_raw.txt`

## Conclusion

The append-only approach is superior for building structured prompt files because it:
- Eliminates complex text manipulation
- Maintains file structure integrity
- Is more predictable and debuggable
- Follows the principle of simplicity