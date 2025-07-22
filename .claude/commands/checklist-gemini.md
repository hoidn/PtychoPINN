# Command: /impl-checklist [context]

**Goal:** Generate a detailed implementation checklist based on the debugging findings from `/debug-gemini-v3`, breaking down the fix into concrete, actionable steps.

**Usage:** 
- `/impl-checklist` - Uses latest debug findings from context
- `/impl-checklist "additional context"` - Adds extra context to consider

**Prerequisites:** Should be run after `/debug-gemini-v3` has identified the root cause and fix approach

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

### Step 1: Extract Fix Recommendations

Identify from the previous conversation:
- The root cause that was found
- The recommended fix approach
- Any specific files/lines mentioned
- Technical details of the solution

```bash
# Setup
mkdir -p ./tmp

# Parse any additional context from arguments
EXTRA_CONTEXT="$ARGUMENTS"
if [ -n "$EXTRA_CONTEXT" ]; then
    echo "Additional context: $EXTRA_CONTEXT"
fi
```

### Step 2: Prepare Context for Gemini

Build a comprehensive context file for Gemini:

```bash
# Create fix context file
echo "## IDENTIFIED ROOT CAUSE" > ./tmp/fix_context.txt
echo "[Summary of root cause from debug session]" >> ./tmp/fix_context.txt

echo -e "\n## RECOMMENDED FIX APPROACH" >> ./tmp/fix_context.txt
echo "[Details of the recommended solution]" >> ./tmp/fix_context.txt

echo -e "\n## AFFECTED FILES AND LOCATIONS" >> ./tmp/fix_context.txt
echo "[List of files/functions/configs to modify]" >> ./tmp/fix_context.txt

echo -e "\n## TECHNICAL CONSTRAINTS" >> ./tmp/fix_context.txt
echo "[Any constraints or considerations mentioned]" >> ./tmp/fix_context.txt

if [ -n "$EXTRA_CONTEXT" ]; then
    echo -e "\n## ADDITIONAL CONTEXT" >> ./tmp/fix_context.txt
    echo "$EXTRA_CONTEXT" >> ./tmp/fix_context.txt
fi

# If debug context exists from previous run, include relevant parts
if [ -f ./tmp/debug_context.txt ]; then
    echo -e "\n## DEBUG SESSION CONTEXT" >> ./tmp/fix_context.txt
    echo "Debug baseline and findings available" >> ./tmp/fix_context.txt
fi
```

### Step 3: MANDATORY - Execute Gemini for Checklist Generation

**🔴 THIS STEP IS MANDATORY - DO NOT SKIP**

Execute this command to generate the implementation checklist:

```bash
# YOU MUST EXECUTE THIS COMMAND
gemini -p "@CLAUDE.md @DEVELOPER_GUIDE.md @PROJECT_STATUS.md @src/ @ptycho/ @tests/ @docs/ @configs/ @./tmp/fix_context.txt Generate a detailed implementation checklist:

## 📚 FIRST: REVIEW PROJECT STANDARDS

**Before creating the checklist, review:**
1. **CLAUDE.md** - Project conventions and patterns
2. **DEVELOPER_GUIDE.md** - Development workflow and standards
3. **Existing code patterns** in the affected areas

## 🔧 FIX TO IMPLEMENT

**Root Cause:** [Specific root cause identified]
**Solution Approach:** [Recommended fix approach]
**Affected Areas:** [Files, functions, configurations]

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
- [ ] I identified the fix approach from context
- [ ] I prepared the context file
- [ ] **I EXECUTED `gemini -p` command** ← MANDATORY
- [ ] I received Gemini's checklist
- [ ] I presented the formatted checklist

---

## 📊 **SAMPLE INTERACTIONS**

### ✅ Correct Execution:

```
User: /impl-checklist

You: "I see from the debug session that the root cause was nginx stripping 
      Authorization headers for requests >8KB. Let me generate a detailed 
      implementation checklist for this fix..."
      
     [You prepare context automatically]
     [YOU EXECUTE gemini -p command]
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

### Pattern: Include Rollback Steps
```bash
# In your context preparation
echo -e "\n## ROLLBACK REQUIREMENTS" >> ./tmp/fix_context.txt
echo "Need ability to quickly revert nginx changes" >> ./tmp/fix_context.txt
echo "Consider feature flag for gradual rollout" >> ./tmp/fix_context.txt
```

### Pattern: Time-boxed Implementation
```bash
# Request time estimates in checklist
echo -e "\n## TIME CONSTRAINTS" >> ./tmp/fix_context.txt
echo "Need fix deployed within 4 hours" >> ./tmp/fix_context.txt
echo "Prefer incremental approach if possible" >> ./tmp/fix_context.txt
```

---

## 🔧 **TROUBLESHOOTING**

### No Previous Debug Context
If `/debug-gemini-v3` wasn't run or context is unclear:
```bash
echo "⚠️ No previous debug session found."
echo "Please provide details about the fix to implement:"
echo "- What is the root cause?"
echo "- What is the recommended solution?"
# Then proceed with user-provided information
```

### Checklist Too Generic
If Gemini provides high-level steps, re-run with more specific context:
```bash
# Add more specific requirements
echo -e "\n## SPECIFIC REQUIREMENTS" >> ./tmp/fix_context.txt
echo "Need exact nginx directives to add" >> ./tmp/fix_context.txt
echo "Include specific test payloads" >> ./tmp/fix_context.txt
echo "Provide monitoring queries to add" >> ./tmp/fix_context.txt
```

---

## 🚀 **Quick Reference**

When user runs `/impl-checklist [context]`:
1. Extract fix approach from previous debug session
2. Prepare comprehensive context file
3. **EXECUTE gemini -p command** (MANDATORY)
4. Present detailed implementation checklist
5. Save checklist for reference

**The command succeeds when you've executed Gemini and delivered a specific, actionable checklist.**
