# Command: /impl-checklist [additional-requirements]

**Goal:** Generate a detailed implementation checklist based on the debugging findings from `/debug-gemini-v3`, breaking down the fix into concrete, actionable steps.

**Usage:** 
- `/impl-checklist` - Uses debug context and findings from previous session
- `/impl-checklist "maintain backwards compatibility"` - Adds extra requirements

**Prerequisites:** 
- Must be run after `/debug-gemini-v3` has completed
- Reuses the `./tmp/debug_context.txt` file and all @ context from debug session
- Assumes root cause and fix approach were identified

---

## 🔴 **CRITICAL: MANDATORY EXECUTION FLOW**

**THIS COMMAND MUST:**
1. You MUST identify the recommended fix from previous debug session
2. You MUST execute `gemini -p` to generate the checklist
3. You MUST process Gemini's response
4. You MUST output a complete implementation checklist

**DO NOT:**
- ❌ Create your own checklist without running Gemini
- ❌ Skip the gemini execution
- ❌ Provide generic steps instead of specific ones

---

## 🤖 **CONTEXT: YOU ARE CLAUDE CODE**

You are Claude Code - the autonomous command-line tool. You:
- **Execute** commands directly
- **Generate** implementation checklists via Gemini
- **Complete** the entire workflow autonomously

---

## 📋 **YOUR EXECUTION WORKFLOW**

### Step 1: Extract Fix Recommendations from Context

Identify from the previous conversation (not from files):
- The root cause that Gemini found
- The recommended fix approach from Gemini's analysis
- Any specific files/lines mentioned
- Technical details of the solution

```bash
# Setup
mkdir -p ./tmp

# Parse any additional requirements from arguments
EXTRA_CONTEXT="$ARGUMENTS"
if [ -n "$EXTRA_CONTEXT" ]; then
    echo "Additional requirements: $EXTRA_CONTEXT"
fi

# The fix details come from the conversation context where Gemini's 
# analysis was reported, not from regenerating the analysis
```

### Step 2: Verify Debug Context Exists

Since this command follows `/debug-gemini-v3`, verify the debug context is available:

```bash
# Check if debug context from previous run exists
if [ ! -f ./tmp/debug_context.txt ]; then
    echo "❌ ERROR: No debug context found from /debug-gemini-v3"
    echo "Please run /debug-gemini-v3 first to identify the issue"
    exit 1
fi

echo "✅ Found debug context from previous session"

# Create a summary of the fix to implement
echo "## FIX IMPLEMENTATION CONTEXT" > ./tmp/fix_summary.txt
echo "Debug session completed at: $(date)" >> ./tmp/fix_summary.txt

if [ -n "$EXTRA_CONTEXT" ]; then
    echo -e "\n## ADDITIONAL IMPLEMENTATION REQUIREMENTS" >> ./tmp/fix_summary.txt
    echo "$EXTRA_CONTEXT" >> ./tmp/fix_summary.txt
fi

echo -e "\n## IMPLEMENTATION FOCUS" >> ./tmp/fix_summary.txt
echo "[This will be populated from the conversation context about the recommended fix]" >> ./tmp/fix_summary.txt
```

### Step 3: MANDATORY - Execute Gemini for Checklist Generation

**🔴 THIS STEP IS MANDATORY - DO NOT SKIP**

Execute this command to generate the implementation checklist:

```bash
# YOU MUST EXECUTE THIS COMMAND - Uses same context as /debug-gemini-v3
gemini -p "@CLAUDE.md @DEVELOPER_GUIDE.md @PROJECT_STATUS.md @src/ @ptycho/ @tests/ @docs/ @configs/ @scripts/ @examples/ @./tmp/debug_context.txt @./tmp/fix_summary.txt Generate a detailed implementation checklist:

## 📚 FIRST: REVIEW PROJECT STANDARDS

**Before creating the checklist, review:**
1. **CLAUDE.md** - Project conventions and patterns
2. **DEVELOPER_GUIDE.md** - Development workflow and standards
3. **The debug_context.txt** - Contains the git history, diffs, and changes analyzed in the previous session
4. **Existing code patterns** in the affected areas

## 🔧 FIX TO IMPLEMENT

Based on the previous debugging session with /debug-gemini-v3:
- Review the debug_context.txt which contains:
  - Recent commits and changes
  - Baseline comparison (what was working vs what's broken)
  - Detailed code diffs
  - Current git status
- The root cause and recommended fix approach discussed in the conversation
- Any additional implementation requirements from fix_summary.txt

## 📋 GENERATE IMPLEMENTATION CHECKLIST

Create a **detailed, step-by-step implementation checklist** that includes:

### 1. **Pre-Implementation Steps**
- [ ] Review current implementation of affected code
- [ ] Backup/branch setup commands
- [ ] Environment preparation
- [ ] Dependency checks

### 2. **Core Implementation Steps**
Break down the fix into atomic, testable steps:
- [ ] Step with specific file:line and exact change
- [ ] Include code snippets where helpful
- [ ] Order steps to maintain working state
- [ ] Flag any risky changes

### 3. **Testing Checklist**
- [ ] Unit test updates needed
- [ ] Integration test scenarios
- [ ] Manual testing steps
- [ ] Edge cases to verify

### 4. **Validation Steps**
- [ ] How to verify the fix works
- [ ] Regression checks
- [ ] Performance impact checks
- [ ] Security considerations

### 5. **Documentation Updates**
- [ ] Code comments needed
- [ ] README updates
- [ ] API documentation
- [ ] Changelog entry

### 6. **Deployment Considerations**
- [ ] Migration steps (if any)
- [ ] Feature flags needed
- [ ] Rollback plan
- [ ] Monitoring additions

## 🎯 CHECKLIST REQUIREMENTS

The checklist should be:
- **Specific**: Include exact file paths, line numbers, and code changes
- **Atomic**: Each checkbox is one concrete action
- **Ordered**: Steps in logical sequence
- **Testable**: Each step has a clear completion criteria
- **Safe**: Include verification after risky changes

## 📝 OUTPUT FORMAT

Provide the checklist in markdown with:
- Checkbox format for easy tracking
- Code snippets in fenced blocks
- File paths in backticks
- Warnings in bold
- Time estimates for major sections

Example format:
```markdown
## 🔧 Implementation Checklist: [Issue Name]

**Estimated Time:** 2-3 hours
**Risk Level:** Medium
**Dependencies:** None

### Phase 1: Preparation (15 min)
- [ ] Create feature branch: `git checkout -b fix/issue-name`
- [ ] Run existing tests: `npm test src/affected-module`
- [ ] Document current behavior (screenshot/log)

### Phase 2: Implementation (1.5 hours)
- [ ] Update `src/module/file.js:45-52`:
  ```javascript
  // OLD CODE
  if (condition) {
    problematicLogic();
  }
  
  // NEW CODE  
  if (condition && additionalCheck) {
    fixedLogic();
  }
  ```
- [ ] Add error handling in `src/utils/helper.js:23`
- [ ] **CRITICAL**: Update config in `config/app.json`

[Continue with all phases...]
```

Remember to make each step concrete and actionable!"
```

**VERIFICATION:** Confirm you have:
- [ ] Executed the gemini command
- [ ] Received the complete checklist
- [ ] Processed the response

### Step 4: Format and Present the Checklist

After Gemini responds:

```bash
# Save the checklist for reference
echo "Gemini's Implementation Checklist:" > ./tmp/implementation_checklist.md
# [Gemini's response will be appended here]

# Present to user
echo "✅ Implementation Checklist Generated"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cat ./tmp/implementation_checklist.md
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Checklist saved to: ./tmp/implementation_checklist.md"
echo "💡 Tip: Copy this checklist to your issue tracker or PR description"
```

---

## ✅ **EXECUTION CHECKLIST**

Before reporting completion:
- [ ] I verified debug context exists from `/debug-gemini-v3`
- [ ] I identified the fix approach from conversation context
- [ ] I created fix_summary.txt with any additional requirements
- [ ] **I EXECUTED `gemini -p` command with all @ context** ← MANDATORY
- [ ] I received Gemini's checklist
- [ ] I presented the formatted checklist

---

## 📊 **SAMPLE INTERACTIONS**

### ✅ Correct Execution:

```
User: /impl-checklist

You: "✅ Found debug context from previous session
      
      I see from the debug session that the root cause was nginx stripping 
      Authorization headers for requests >8KB. Let me generate a detailed 
      implementation checklist for this fix..."
      
     [You verify debug context exists]
     [YOU EXECUTE gemini -p command with all @ context files]
     [You process response]
     
     "✅ Implementation Checklist Generated
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     
     ## 🔧 Implementation Checklist: Fix Authorization Header Stripping
     
     **Estimated Time:** 1-2 hours
     **Risk Level:** High (affects authentication)
     
     ### Phase 1: Preparation (15 min)
     - [ ] Create branch: `git checkout -b fix/auth-header-nginx`
     - [ ] Backup nginx config: `cp /etc/nginx/conf.d/api.conf /tmp/api.conf.backup`
     - [ ] Test current behavior with: `curl -X POST https://api/endpoint -H "Authorization: Bearer $TOKEN" -d @large_payload.json`
     
     ### Phase 2: Implementation (45 min)
     - [ ] Edit `/etc/nginx/conf.d/api.conf:23`:
       ```nginx
       # OLD
       client_body_buffer_size 8k;
       
       # NEW  
       client_body_buffer_size 16k;
       proxy_pass_request_headers on;
       proxy_set_header Authorization $http_authorization;
       ```
     [... rest of checklist ...]"
```

### ❌ What NOT to do:

```
User: /impl-checklist

You: "Based on the previous findings, here's my checklist:
     1. Fix nginx config
     2. Test the changes
     3. Deploy"
     
     ❌ FAILED - You didn't execute Gemini!
```

---

## ⚠️ **PATTERNS TO AVOID**

### Don't Create Generic Checklists
**WRONG:**
```markdown
- [ ] Fix the bug
- [ ] Test the fix
- [ ] Deploy
```

**RIGHT:**
```markdown
- [ ] Update `/etc/nginx/conf.d/api.conf:23` - increase client_body_buffer_size to 16k
- [ ] Test with: `curl -X POST https://api/endpoint -H "Authorization: Bearer $TOKEN" -d @8kb_payload.json`
- [ ] Deploy using: `ansible-playbook deploy.yml --tags nginx`
```

---

## 💡 **ENHANCED PATTERNS**

### Pattern: Add Implementation Constraints
```bash
# Add constraints to fix_summary.txt
echo -e "\n## IMPLEMENTATION CONSTRAINTS" >> ./tmp/fix_summary.txt
echo "Must maintain backwards compatibility" >> ./tmp/fix_summary.txt
echo "Cannot require service restart during business hours" >> ./tmp/fix_summary.txt
echo "Need feature flag for gradual rollout" >> ./tmp/fix_summary.txt
```

### Pattern: Reference Specific Debug Findings
```bash
# Extract key findings from debug context
if grep -q "nginx" ./tmp/debug_context.txt; then
    echo -e "\n## NGINX-SPECIFIC REQUIREMENTS" >> ./tmp/fix_summary.txt
    echo "Focus on nginx configuration changes identified in debug" >> ./tmp/fix_summary.txt
    echo "Include nginx reload procedures" >> ./tmp/fix_summary.txt
fi
```

### Pattern: Time-boxed Implementation
```bash
# Request time estimates based on debug complexity
echo -e "\n## TIME CONSTRAINTS" >> ./tmp/fix_summary.txt
echo "Based on debug findings, need fix deployed within 4 hours" >> ./tmp/fix_summary.txt
echo "Prefer incremental approach if possible" >> ./tmp/fix_summary.txt
```

---

## 🔧 **TROUBLESHOOTING**

### No Previous Debug Context
If `/debug-gemini-v3` wasn't run:
```bash
if [ ! -f ./tmp/debug_context.txt ]; then
    echo "❌ ERROR: No debug context found"
    echo "Please run /debug-gemini-v3 first to identify the issue"
    echo ""
    echo "Usage:"
    echo "1. /debug-gemini-v3 [baseline]  # Identify the issue"
    echo "2. /impl-checklist             # Generate fix checklist"
    exit 1
fi
```

### Stale Debug Context
If the debug context might be outdated:
```bash
# Check age of debug context
if [ -f ./tmp/debug_context.txt ]; then
    age=$(find ./tmp/debug_context.txt -mmin +60 | wc -l)
    if [ "$age" -gt 0 ]; then
        echo "⚠️ Warning: Debug context is over 60 minutes old"
        echo "Consider running /debug-gemini-v3 again for fresh analysis"
    fi
fi
```

---

## 🚀 **Quick Reference**

When user runs `/impl-checklist [context]`:
1. Verify debug context exists from `/debug-gemini-v3`
2. Create fix summary with any additional requirements
3. **EXECUTE gemini -p command** with same @ context files (MANDATORY)
4. Present detailed implementation checklist
5. Save checklist for reference

**Key Points:**
- Reuses `./tmp/debug_context.txt` from previous `/debug-gemini-v3` run
- Includes all the same @ project files for full context
- No need to regenerate git logs/diffs
- Assumes fix approach was identified in previous session

**The command succeeds when you've executed Gemini and delivered a specific, actionable checklist.**
