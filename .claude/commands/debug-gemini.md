# Command: /debug-gemini

**Goal:** Get comprehensive debugging help from Gemini with fresh perspective, especially when Claude might have tunnel vision about the root cause.

---

## ⚠️ **YOUR ROLE AS CLAUDE**

You will:
1. Share your current understanding of the bug
2. Explicitly state your assumptions and suspected causes
3. Ask Gemini to challenge your assumptions
4. Request alternative explanations you might have missed
5. Include EXTENSIVE context (not just code where you think the bug is)

---

## 🚀 **EXECUTION WORKFLOW**

### Step 1: Gather Your Current Understanding (Claude)

Before asking Gemini, document:
1. **Symptoms:** What exactly is happening?
2. **Your Theory:** What you think is causing it
3. **What You've Tried:** Debugging steps taken
4. **Your Assumptions:** What you're taking for granted
5. **Tunnel Vision Risk:** What you might be overlooking

### Step 2: Generate Comprehensive Debug Command

Create this command for the user, being GENEROUS with paths:

```bash
gemini -p "@src/ @ptycho/ @tests/ @docs/ @configs/ @logs/ @.github/ @scripts/ @benchmarks/ @examples/ Debug this issue with FRESH EYES:

## ISSUE SUMMARY
**Symptoms:** [Detailed symptoms with specific errors, stack traces, or behaviors]
**When It Happens:** [Specific conditions, inputs, or sequences that trigger it]
**When It Doesn't Happen:** [Cases where it works fine]
**Environment:** [Dev/staging/prod, OS, versions]

## CLAUDE'S CURRENT UNDERSTANDING
**My Leading Theory:** [What Claude thinks is wrong]
**Evidence For This Theory:** 
- [Specific observation 1]
- [Specific observation 2]

**Code I'm Focused On:**
- `[file:line]` - [Why Claude suspects this]
- `[file:line]` - [Why Claude suspects this]

**What I've Already Tried:**
1. [Debugging step 1 and result]
2. [Debugging step 2 and result]
3. [Debugging step 3 and result]

## MY ASSUMPTIONS (PLEASE CHALLENGE THESE)
1. [Assumption about the system]
2. [Assumption about the data flow]
3. [Assumption about dependencies]
4. [Assumption about configuration]

## GEMINI: PLEASE PROVIDE FRESH PERSPECTIVE

1. **Challenge My Assumptions:** What am I taking for granted that might be wrong?

2. **Alternative Root Causes:** What OTHER parts of the system could cause these symptoms?
   - Consider timing/race conditions
   - Consider configuration issues
   - Consider environmental differences
   - Consider dependency conflicts
   - Consider data corruption
   - Consider edge cases I missed

3. **Check Project Documentation:** 
   - Are there known issues in README.md, CHANGELOG.md, or docs/?
   - Are there migration guides I missed?
   - Are there deprecation warnings?
   - Are there similar fixed issues in closed PRs/issues?

4. **Analyze Wider Context:**
   - What related systems could be involved?
   - What recent changes in OTHER modules could affect this?
   - What implicit dependencies exist?
   - What assumptions does the code make about its environment?

5. **Suggest Non-Obvious Debug Steps:**
   - What diagnostic commands would reveal more?
   - What logging should be added?
   - What state should be inspected?
   - What tools could help (profilers, tracers, etc.)?

6. **Pattern Recognition:**
   - Have you seen similar symptoms in this codebase before?
   - What patterns in the code could lead to this behavior?
   - Are there anti-patterns that match these symptoms?

7. **Systemic Issues:**
   - Could this be a design flaw rather than a bug?
   - Are there architectural issues at play?
   - Is this a symptom of technical debt?

## SPECIFIC AREAS TO INVESTIGATE

Beyond my focus areas, please specifically check:
- Race conditions in async/concurrent code
- State management issues
- Cache invalidation problems
- Off-by-one errors
- Timezone/locale issues
- Memory leaks or resource exhaustion
- Network timeouts or retries
- Permission or security constraints
- Build/compilation issues
- Version mismatches

## OUTPUT FORMAT

Please provide:
1. **Most Likely Alternative Causes** (ranked by probability)
2. **Specific Things to Check** (with exact commands/locations)
3. **Debug Strategy** (systematic approach)
4. **Quick Experiments** (to prove/disprove theories)
5. **Long-term Fixes** (if this reveals systemic issues)

Remember: I might be completely wrong about where the bug is. Look everywhere, not just where I'm pointing."
```

### Step 3: Process Gemini's Fresh Perspective

When Gemini responds:

1. **Highlight Surprising Findings:**
   ```markdown
   ## 🎯 Fresh Insights from Gemini
   
   ### Things Claude Missed:
   - [Unexpected cause 1]
   - [Overlooked connection 2]
   
   ### Challenged Assumptions:
   - Claude assumed [X], but actually [Y]
   - Claude focused on [A], but [B] is more likely
   ```

2. **Create New Debug Plan:**
   Based on Gemini's analysis, create a systematic debug approach

3. **Update Understanding:**
   Document what tunnel vision caused you to miss

---

## 💡 **ENHANCED DEBUG PATTERNS**

### Pattern 1: Performance Degradation
```bash
gemini -p "@src/ @benchmarks/ @profiling/ @logs/ @monitoring/ @configs/ @docs/performance/ 
Performance degraded after [change]. I think it's [cause], but need fresh eyes.
[Include specific metrics, timeline, what changed]
Check for: memory leaks, N+1 queries, cache misses, lock contention, GC pressure"
```

### Pattern 2: Intermittent Failures
```bash
gemini -p "@src/ @tests/ @.github/workflows/ @logs/ @configs/ @infrastructure/
Intermittent test failures. I think it's [race condition in X], but could be wrong.
[Include failure rate, patterns, logs]
Check for: test pollution, timezone issues, external dependencies, resource limits"
```

### Pattern 3: Integration Issues
```bash
gemini -p "@src/ @docs/api/ @examples/ @integration_tests/ @configs/ @docker/
API integration failing. I think it's [auth issue], but customer says it worked before.
[Include request/response, versions, environment]
Check for: API changes, version mismatches, network policies, SSL/TLS issues"
```

### Pattern 4: Data Corruption
```bash
gemini -p "@src/ @migrations/ @docs/data/ @scripts/ @tests/fixtures/ @configs/
Data corruption in [table/field]. I think it's [bad migration], but could be deeper.
[Include samples, timeline, affected records]
Check for: race conditions, transaction issues, encoding problems, precision loss"
```

---

## 🎯 **ANTI-TUNNEL VISION CHECKLIST**

Before sending to Gemini, ask yourself:
- [ ] Am I including areas OUTSIDE where I think the bug is?
- [ ] Have I included all documentation, not just code?
- [ ] Am I sharing what WORKS, not just what's broken?
- [ ] Have I listed my assumptions explicitly?
- [ ] Am I open to being completely wrong?

---

## 📊 **SAMPLE INTERACTION**

```
Claude: "I think the auth bug is in the JWT validation at auth.py:45..."
[Generates Gemini command with full context]

Gemini: "The JWT validation is fine. The issue is actually in the nginx 
config at /etc/nginx/conf.d/api.conf:23 - it's stripping the Authorization 
header for requests over 8KB. This explains why it only fails for users 
with large permission sets."

Claude: "I was completely focused on the Python code and missed the 
infrastructure layer! Here's a new debug plan based on your insight..."
```

---

## 🚨 **COMMON TUNNEL VISION TRAPS**

Share these with Gemini to check:

1. **Looking Where the Error Appears** (not where it originates)
2. **Assuming Recent Changes** (when old code hit new conditions)
3. **Focusing on Code** (when it's config/environment/data)
4. **Debugging Symptoms** (not root causes)
5. **Trusting Error Messages** (when they're misleading)
6. **Assuming Local = Production** (environment differences)
7. **Following Stack Traces** (missing async/timing issues)
8. **Checking Application Layer** (missing infrastructure/OS issues)

---

## 🔄 **ITERATIVE DEBUGGING**

After Gemini's first analysis:

```bash
# If new theory emerges
gemini -p "@[new_relevant_paths]/ Gemini suggested [theory]. 
Let's deep dive into [specific area] to verify...
[Include Gemini's evidence]
Please analyze [specific aspect] in detail."

# If multiple theories exist
gemini -p "@src/ Here are the top 3 theories:
1. [Gemini's top theory]
2. [Alternative theory]
3. [Claude's original theory]
Design experiments to distinguish between these."
```

---

## 💡 **MAXIMIZING GEMINI'S HELP**

### DO Include:
- Error messages AND success cases
- Logs from MULTIPLE sources
- Configuration files (all of them)
- Documentation and comments
- Test files (even passing ones)
- CI/CD configurations
- Monitoring/metrics data
- Example user inputs
- Environment details

### DO Ask For:
- Alternative explanations
- Non-obvious connections
- Historical patterns
- Systemic issues
- Missing safeguards
- Better error handling
- Preventive measures

### DON'T:
- Limit paths to suspected areas
- Hide your wrong assumptions
- Skip "irrelevant" details
- Focus only on recent changes

---

## 🎯 **SUCCESS METRICS**

Track your tunnel vision improvement:
- How often was your initial theory wrong?
- How many fresh insights did Gemini provide?
- How much debugging time was saved?
- What patterns of tunnel vision do you have?

Remember: The goal is to break out of tunnel vision and see the bug from completely new angles!
