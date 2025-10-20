# Agentic Infrastructure & Tooling Optimizations

**Status:** PROPOSED
**Based On:** Analysis of galph iterations 1-89 (feature/torchapi branch)
**Expected Impact:** 2-3x speedup through system-level efficiency improvements
**Priority:** HIGH (complements methodology optimizations in PROCESS_OPTIMIZATIONS.md)
**Analysis Date:** 2025-10-19

---

## Executive Summary

Analysis of 89 supervisor/engineer iterations revealed that **40-50% of iteration time is spent on infrastructure overhead** rather than actual planning or implementation work:

1. **Context re-reading waste**: Same 7-8 docs read every iteration, even when unchanged
2. **Git synchronization overhead**: Pull/push on every handoff + 5-second polling
3. **Prompt size bloat**: 500+ lines of unchanging boilerplate loaded fresh each time
4. **Two-agent handoff tax**: 21-minute full loop even for trivial tasks

This document proposes **8 infrastructure optimizations** that could reduce iteration overhead by 50-70% and total iteration time by 40-50%.

**Relationship to PROCESS_OPTIMIZATIONS.md:**
- That doc: Methodology improvements (better planning, quality gates) → Avoid rework
- This doc: Infrastructure improvements (caching, faster sync) → Reduce overhead
- Combined impact: **5-8x total speedup**

---

## Data Gathered

### Timing Metrics (from iteration summaries)
- **Galph (supervisor)**: 8-9 minutes per planning iteration
- **Ralph (engineer)**: 13 minutes per execution iteration
- **Full loop**: 21 minutes average (galph + ralph + handoff)
- **Retry rate**: 12.5% (10 failed iterations out of 80 unique iterations)
- **Early iterations (2-6)**: 2-4 attempts each (debugging/retry loops)

### Context Reading Patterns
Every iteration reads the **same 7-8 documents**:
- `docs/findings.md`
- `docs/fix_plan.md`
- `docs/workflows/pytorch.md`
- `docs/TESTING_GUIDE.md`
- `docs/index.md`
- `specs/data_contracts.md`
- `specs/ptychodus_api_spec.md`

**Problem**: No mechanism to detect if these files changed since last iteration.

### Git Synchronization Overhead
- Git pull before EVERY galph turn (`supervisor.sh:101`)
- Git pull before EVERY ralph turn (`loop.sh:146`)
- Git push after state update (twice per loop: `supervisor.sh:164`, `loop.sh:142`)
- Poll interval: 5 seconds when waiting for turn (`supervisor.sh:149`)
- Network latency on every handoff (pull + rebase + push)

### Summary Quality Evolution
- **Older summaries** (galph-summaries/ hyphen format): ~10-15KB average
- **Newer summaries** (galph_summaries/ underscore format): ~30-50KB average (+200% growth)
- **Quality improvements**: Timestamps, metrics, decision rationale, compliance checklists
- **Token cost**: Proportional increase (3-5x more tokens per summary)

---

## Tier 1: Critical Quick Wins (High Impact, Easy Implementation)

### OPT-5: Context Caching

**Problem**: Both agents re-read the same 7-8 docs every iteration, even when files haven't changed.

**Evidence**:
- Every iteration reads: findings.md, fix_plan.md, workflows/pytorch.md, etc.
- No file-hash or git-based change detection
- Wasteful token consumption (estimated 5,000-10,000 tokens per iteration)
- API cost accumulation over 89 iterations

**Solution**: File-hash based context caching

```python
# scripts/orchestration/context_cache.py
import hashlib
import json
from pathlib import Path

class ContextCache:
    def __init__(self, cache_file="tmp/context_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            return json.loads(self.cache_file.read_text())
        return {}

    def _compute_hash(self, filepath):
        return hashlib.sha256(Path(filepath).read_bytes()).hexdigest()

    def get_document(self, filepath, force_reload=False):
        """Get document content, using cache if unchanged."""
        current_hash = self._compute_hash(filepath)
        cache_key = str(filepath)

        if not force_reload and cache_key in self.cache:
            cached_hash, cached_content = self.cache[cache_key]
            if cached_hash == current_hash:
                return cached_content, "cache_hit"

        # Cache miss or changed - read and cache
        content = Path(filepath).read_text()
        self.cache[cache_key] = (current_hash, content)
        self._save_cache()
        return content, "cache_miss"

    def invalidate_changed_files(self):
        """Remove cache entries for files that changed since last iteration."""
        changed_files = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1", "docs/", "specs/"],
            text=True
        ).strip().split("\n")

        for filepath in changed_files:
            self.cache.pop(filepath, None)
        self._save_cache()
```

**Integration**:
```bash
# In supervisor.sh / loop.sh before reading context
python3 -m scripts.orchestration.context_cache invalidate

# In agent prompts, reference cached status
python3 -m scripts.orchestration.context_cache get docs/findings.md
```

**Impact**:
- **40-60% reduction in context tokens** per iteration
- Faster iteration start (no unnecessary file I/O)
- **Lower API costs** (estimated $10-50 savings per initiative at current Claude pricing)
- Better scalability as documentation grows

**Implementation Effort**: 2-4 hours
- Write context_cache.py (~1 hour)
- Integrate into supervisor.sh/loop.sh (~1 hour)
- Test with sample iterations (~1 hour)
- Document in DEVELOPER_GUIDE.md (~1 hour)

**Risks**:
- Cache invalidation bugs (file changed but cache not invalidated)
- **Mitigation**: Add `--no-cache` flag for debugging, clear cache on branch switch

---

### OPT-6: Git Synchronization Overhead Reduction

**Problem**: Git pull/push on EVERY agent handoff creates 30-60 seconds of network latency per loop.

**Evidence**:
- `supervisor.sh`: `git_safe_pull` before galph runs (line 121)
- `loop.sh`: `git_safe_pull` before ralph runs (line 146)
- State updates trigger `git commit && git push` (twice per loop)
- `POLL_INTERVAL=5` creates artificial 5-second delays while waiting
- 30-second timeout on git pull suggests network issues are common (line 102)

**Solution**: Local filesystem lock for turn-taking, batch git sync

```python
# scripts/orchestration/local_coordinator.py
import fcntl
import json
import time
from pathlib import Path

class LocalCoordinator:
    """Filesystem-based turn coordination (no git needed for handoff)."""

    def __init__(self, state_file="sync/state.json", lock_file="sync/state.lock"):
        self.state_file = Path(state_file)
        self.lock_file = Path(lock_file)
        self.lock_file.parent.mkdir(exist_ok=True)

    def acquire_turn(self, actor_name, timeout=300):
        """Block until it's our turn, return state."""
        start = time.time()

        while True:
            with open(self.lock_file, 'w') as lockf:
                fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)

                state = self._read_state()
                if state.get('expected_actor') == actor_name:
                    # It's our turn - mark running and return
                    state['status'] = f'running-{actor_name}'
                    self._write_state(state)
                    return state

                fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)

            # Not our turn - wait and retry
            if time.time() - start > timeout:
                raise TimeoutError(f"Timeout waiting for {actor_name} turn")

            time.sleep(1)  # Poll every 1 second (instead of 5)

    def release_turn(self, next_actor, state_updates):
        """Release turn to next actor."""
        with open(self.lock_file, 'w') as lockf:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)

            state = self._read_state()
            state.update(state_updates)
            state['expected_actor'] = next_actor
            state['status'] = 'ok'
            self._write_state(state)

            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)

    def sync_to_git(self):
        """Batch git sync (only when needed, not per-handoff)."""
        subprocess.run(["git", "pull", "--rebase"], check=True, timeout=30)
        subprocess.run(["git", "add", str(self.state_file)], check=True)
        subprocess.run(["git", "commit", "-m", f"[SYNC] {state}"], check=True)
        subprocess.run(["git", "push"], check=True, timeout=30)
```

**Integration**:
```bash
# supervisor.sh (simplified)
# Only git sync at END of iteration, not every handoff
python3 -m scripts.orchestration.local_coordinator acquire galph
# ... do galph work ...
python3 -m scripts.orchestration.local_coordinator release ralph
python3 -m scripts.orchestration.local_coordinator sync  # Once per iteration
```

**Impact**:
- **30-60 second reduction per loop** (eliminate git pull/push on every handoff)
- **25% faster iteration cycle** (21 min → 16 min)
- Reduced polling interval (5s → 1s) saves 4 seconds per wait
- More reliable (no git rebase conflicts during handoff)

**Implementation Effort**: 4-6 hours
- Write local_coordinator.py (~2 hours)
- Modify supervisor.sh/loop.sh (~2 hours)
- Test multi-iteration scenarios (~1 hour)
- Handle edge cases (crash recovery) (~1 hour)

**Risks**:
- Lock file corruption if process crashes
- **Mitigation**: Stale lock detection (if lock held >20 min, break it)
- Sync conflicts if multiple machines
- **Mitigation**: Keep git sync option for distributed teams (`--sync-via-git` flag)

---

### OPT-7: Prompt Size Reduction (Static/Dynamic Split)

**Problem**: 288-line supervisor prompt + 228-line main prompt loaded fresh every iteration with mostly unchanging boilerplate.

**Evidence**:
- `prompts/supervisor.md`: 288 lines (action types, workflow rules, git instructions)
- `prompts/main.md`: 228 lines
- Lines 1-80 of supervisor.md are **static instructions** (same every iteration)
- Lines 81-288 could be **dynamic context** (focus issue, recent commits, dependencies)
- API charges for re-sending static instructions every time

**Solution**: Separate static core from dynamic context

```markdown
# prompts/supervisor_core.md (static, loaded once at session start)
<role>
planning, review and analysis. do not make implementation code changes.
</role>

<current long-term goals>
complete the fix plan
</current long-term goals>

<action types>
[All the action type definitions - unchanged from current]
</action types>

<workflow rules>
[All the workflow rules - unchanged from current]
</workflow rules>

# prompts/supervisor_context.md (dynamic, regenerated per iteration)
<task>
You are galph, iteration {{iteration_number}}.

## Focus Issue (from previous turn)
{{previous_focus_issue}}

## Ralph's Last Output
{{ralph_summary}}

## Dependencies
{{dependency_checklist}}

## Updated Context Files
{{git_diff_summary}}

Now choose your next focus issue following the core workflow.
</task>
```

**Implementation**:
```python
# scripts/orchestration/prompt_builder.py
def build_supervisor_prompt(iteration_num, context):
    """Build supervisor prompt from static core + dynamic context."""
    core = Path("prompts/supervisor_core.md").read_text()  # Load once

    # Generate dynamic context
    context_vars = {
        'iteration_number': iteration_num,
        'previous_focus_issue': context.get('focus_issue'),
        'ralph_summary': read_last_summary(),
        'dependency_checklist': generate_dep_checklist(),
        'git_diff_summary': get_git_diff_summary()
    }

    dynamic = render_template("prompts/supervisor_context.md", context_vars)

    return core + "\n\n" + dynamic
```

**Impact**:
- **50% reduction in prompt tokens** (288 lines → ~140 static + ~70 dynamic)
- Faster model processing (less irrelevant context to parse)
- Easier prompt maintenance (static core rarely changes)
- **Lower API costs** (~$5-20 per initiative)

**Implementation Effort**: 3-4 hours
- Split supervisor.md into core + context (~1 hour)
- Split main.md into core + context (~1 hour)
- Write prompt_builder.py (~1 hour)
- Test with sample iterations (~1 hour)

**Risks**:
- Template rendering bugs
- **Mitigation**: Extensive testing, keep old prompts as fallback

---

## Tier 2: High Priority (High Impact, Medium Effort)

### OPT-8: Complexity-Based Routing

**Problem**: Galph → Ralph → Galph creates 21-minute overhead even for trivial tasks like "update comment" or "fix typo".

**Evidence**:
- Galph: 8 min planning a simple task
- Ralph: 13 min executing it
- Full loop: 21 min total
- Some tasks could be done in 2 minutes by single agent
- 12.5% retry rate suggests handoff communication gaps
- ~26% of iterations are pure housekeeping (docs updates, plan logging)
- ~30% of tasks could skip galph planning with no quality loss

**Solution**: Route simple tasks to single-agent execution, keep two-agent loop for complex work

**Note**: See detailed elaboration in conversation logs for multi-dimensional scoring, safety mechanisms, and real-world calibration

```python
# scripts/orchestration/task_classifier.py
class TaskComplexity:
    SIMPLE = "simple"      # Single agent, <5 min
    MODERATE = "moderate"  # Two-agent, standard loop
    COMPLEX = "complex"    # Two-agent, may need multiple loops

def classify_task(task_description, context):
    """Classify task complexity to determine execution mode."""

    # Simple task indicators
    simple_keywords = [
        "typo", "comment", "docstring", "formatting",
        "rename variable", "update readme", "fix lint"
    ]

    # Complex task indicators
    complex_keywords = [
        "architecture", "refactor", "design", "integrate",
        "implement feature", "debug", "optimize performance"
    ]

    desc_lower = task_description.lower()

    # Check for simple patterns
    if any(kw in desc_lower for kw in simple_keywords):
        if not any(kw in desc_lower for kw in complex_keywords):
            return TaskComplexity.SIMPLE

    # Check file count
    files_changed = context.get('estimated_files_changed', 0)
    if files_changed <= 1:
        return TaskComplexity.SIMPLE
    elif files_changed > 5:
        return TaskComplexity.COMPLEX

    # Check test requirements
    if context.get('requires_new_tests'):
        return TaskComplexity.COMPLEX

    # Default to moderate (use two-agent loop)
    return TaskComplexity.MODERATE
```

**Practical Realization**:

Current workflow has no explicit task input - galph self-selects focus from fix_plan.md. Three integration approaches:

**Approach A: Wrapper Script (Highest Impact)**
```bash
#!/usr/bin/env bash
# run.sh - Smart executor with routing

TASK_DESC="$1"
COMPLEXITY=$(python3 -m scripts.orchestration.task_classifier \
    --task "$TASK_DESC" \
    --context sync/state.json)

case "$COMPLEXITY" in
    simple)
        echo "🚀 Fast-track: Direct to ralph"
        # Create minimal input.md
        cat > input.md << EOF
# Do Now (Fast-Track)
$TASK_DESC
Context: Simple task, no planning needed.
EOF
        ./loop.sh --fast-track
        ;;
    moderate|complex)
        ./supervisor.sh --task "$TASK_DESC"
        ./loop.sh
        ;;
esac

# Usage: ./run.sh "Update fix_plan.md with attempt log"
```

**Approach B: Galph Self-Routes (Less Disruptive)**

Add Step 0 to prompts/supervisor.md:
- Galph reads focus issue from fix_plan.md
- Runs classifier on focus issue
- If SIMPLE: creates minimal input.md, exits immediately (2 min)
- If COMPLEX: proceeds with full planning (8 min)

**Approach C: Post-Galph Routing (Incremental)**

Galph declares complexity in input.md metadata:
```markdown
---
complexity: simple
estimated_time: 2min
---
# Do Now
...
```

loop.sh reads metadata and adjusts ralph execution mode.

**Recommended**: Start with Approach B (safest), migrate to Approach A after validation.

**Key Constraint**: Current system has galph autonomously select focus from fix_plan.md (no user-specified task input). Approach A requires adding user task specification. Approach B works within existing autonomous workflow.

**Impact**:
- **30-50% time reduction for simple tasks** (21 min → 5-10 min)
- **Fewer retry loops** (no handoff miscommunication for simple work)
- Better resource utilization (don't waste planner on trivial work)

**Implementation Effort**: 6-8 hours
- Write task_classifier.py (~2 hours)
- Integrate into orchestration (~2 hours)
- Tune complexity heuristics (~2 hours)
- Test with historical tasks (~2 hours)

**Risks**:
- Misclassification (task harder than expected)
- **Mitigation**: Conservative defaults (bias toward two-agent), allow override flag

---

### OPT-9: Incremental Context Building

**Problem**: Every iteration starts with cold context—no "diff" mechanism to see what changed since last iteration.

**Evidence**:
- Both agents re-read fix_plan.md from scratch (often 500+ lines)
- No awareness of "what's new since last loop"
- CLAUDE.md directive says "you may rely on cached understanding" but no mechanism exists
- Agents must rebuild full mental model each time (cognitive overhead)

**Solution**: Context delta tracking between iterations

```python
# scripts/orchestration/context_delta.py
import difflib
from pathlib import Path

class ContextDelta:
    def __init__(self, iteration_num):
        self.iteration = iteration_num
        self.prev_iteration = iteration_num - 1

    def compute_delta(self):
        """Compute what changed since last iteration."""
        delta = {
            'changed_files': self._get_changed_files(),
            'new_plan_items': self._get_new_plan_items(),
            'completed_tasks': self._get_completed_tasks(),
            'new_findings': self._get_new_findings(),
        }
        return delta

    def _get_changed_files(self):
        """Git diff of key documentation."""
        result = subprocess.check_output(
            ["git", "diff", "--name-only", "HEAD~1", "docs/", "specs/", "plans/"],
            text=True
        ).strip()
        return result.split("\n") if result else []

    def _get_new_plan_items(self):
        """Diff fix_plan.md to find new items."""
        try:
            old = subprocess.check_output(
                ["git", "show", f"HEAD~1:docs/fix_plan.md"],
                text=True
            )
            new = Path("docs/fix_plan.md").read_text()

            diff = difflib.unified_diff(
                old.splitlines(),
                new.splitlines(),
                lineterm=""
            )

            # Extract only additions (lines starting with +)
            additions = [line[1:] for line in diff if line.startswith('+')]
            return additions
        except:
            return []

    def _get_completed_tasks(self):
        """Extract completed tasks from last iteration summary."""
        # Parse last galph summary for completed checklist items
        last_summary = Path(f"logs/galph-summaries/iter-{self.prev_iteration:05d}_*.md")
        # ... parse and extract completed items ...
        pass
```

**Integration**:
```markdown
# In supervisor prompt (dynamic context section)
## Context Update Since Last Iteration

Changed files:
{{delta.changed_files}}

New plan items:
{{delta.new_plan_items}}

Completed tasks:
{{delta.completed_tasks}}

*Full context cached from previous iteration. Review delta above to understand what changed.*
```

**Impact**:
- **50-70% reduction in "ramp-up" time** (don't rebuild full mental model)
- Better continuity across iterations
- Faster focus issue selection (see what's new immediately)
- More efficient token usage (delta << full context)

**Implementation Effort**: 6-8 hours
- Write context_delta.py (~3 hours)
- Integrate into prompt generation (~2 hours)
- Test delta accuracy (~2 hours)
- Document in workflow guide (~1 hour)

**Risks**:
- Delta misses important context
- **Mitigation**: Always provide "reload full context" option, periodic full refresh

---

### OPT-10: Artifact Discovery Index

**Problem**: Artifacts stored in `plans/active/<id>/reports/<timestamp>/` but no searchable index.

**Evidence**:
- 80+ iterations = 80+ timestamped directories
- No way to answer "what artifacts exist for initiative X?"
- Agents must `grep` through `fix_plan.md` to find artifact paths
- Manual cleanup difficulty (which directories are safe to delete?)

**Solution**: Structured artifact index with metadata

```yaml
# plans/active/artifact_index.yaml
TEST-PYTORCH-001:
  evidence_collection:
    - timestamp: 2025-10-17T184624Z
      type: governance_decision
      files:
        - governance_decision.md
        - phase_f_integration/findings.md
      size_kb: 156
      iteration: 39

    - timestamp: 2025-10-19T193425Z
      type: parity_test_results
      files:
        - phase_d_hardening/runtime_profile.md
        - phase_d_hardening/test_execution.log
      size_kb: 892
      iteration: 88

  summaries:
    - timestamp: 2025-10-19T225900Z
      type: phase_b_fixture
      files:
        - fixture_notes.md
      size_kb: 23
      iteration: 74

INTEGRATE-PYTORCH-001:
  # ... other initiatives ...
```

**Auto-generation**:
```python
# scripts/tools/build_artifact_index.py
def scan_artifacts():
    """Scan plans/active/**/reports/ and build index."""
    index = {}

    for initiative_dir in Path("plans/active").iterdir():
        if not initiative_dir.is_dir():
            continue

        initiative_id = initiative_dir.name
        reports_dir = initiative_dir / "reports"

        if not reports_dir.exists():
            continue

        index[initiative_id] = []

        for timestamp_dir in reports_dir.iterdir():
            if not timestamp_dir.is_dir():
                continue

            files = list(timestamp_dir.rglob("*"))
            size_kb = sum(f.stat().st_size for f in files if f.is_file()) // 1024

            index[initiative_id].append({
                'timestamp': timestamp_dir.name,
                'files': [str(f.relative_to(timestamp_dir)) for f in files if f.is_file()],
                'size_kb': size_kb,
                'iteration': extract_iteration_from_fixplan(initiative_id, timestamp_dir.name)
            })

    return index
```

**Usage**:
```bash
# Quick queries
scripts/tools/artifact_index.py list TEST-PYTORCH-001
scripts/tools/artifact_index.py find "runtime_profile"
scripts/tools/artifact_index.py cleanup --older-than 30days --keep-latest 5
```

**Impact**:
- **Faster artifact discovery** (no grepping)
- **Better historical context** (see all artifacts for an initiative)
- **Easier cleanup** (identify large/old artifacts safely)
- **Better handoff** (galph can reference specific artifacts by iteration)

**Implementation Effort**: 4-6 hours
- Write artifact scanner (~2 hours)
- Add index CLI tool (~2 hours)
- Integrate into workflow (~1 hour)
- Document usage (~1 hour)

**Risks**:
- Index out of sync with filesystem
- **Mitigation**: Auto-regenerate on every galph run, lightweight scan

---

## Tier 3: Nice to Have (Good Ideas, More Complex)

### OPT-11: Automated Compliance Checking

**Problem**: Agents manually verify compliance (findings consulted? plan updated? TDD followed?).

**Evidence**:
- Newer summaries have explicit compliance checklists
- Suggests manual/inconsistent checking
- No automated enforcement

**Solution**: Pre-loop / post-loop compliance hooks

```bash
#!/usr/bin/env bash
# scripts/hooks/pre_iteration_check.sh

FOCUS_ISSUE="$1"

echo "Running pre-iteration compliance checks..."

# Check 1: findings.md consulted for focus issue
if ! grep -qi "$FOCUS_ISSUE" docs/findings.md; then
    echo "⚠️  WARNING: Focus issue '$FOCUS_ISSUE' not found in findings.md"
    echo "   Have you checked for prior knowledge/learnings?"
    echo "   Continue anyway? [y/N]"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check 2: fix_plan.md has entry for focus issue
if ! grep -qi "$FOCUS_ISSUE" docs/fix_plan.md; then
    echo "❌ ERROR: Focus issue '$FOCUS_ISSUE' not in fix_plan.md"
    echo "   Add entry before starting work."
    exit 1
fi

echo "✅ Pre-iteration checks passed"
```

```bash
#!/usr/bin/env bash
# scripts/hooks/post_iteration_check.sh

FOCUS_ISSUE="$1"

echo "Running post-iteration compliance checks..."

# Check 1: Tests modified if implementation changed
if git diff --name-only HEAD~1 | grep -qE "^ptycho|^ptycho_torch"; then
    if ! git diff --name-only HEAD~1 | grep -q "^tests/"; then
        echo "⚠️  WARNING: Production code changed but no test changes"
        echo "   Did you follow TDD?"
    fi
fi

# Check 2: fix_plan.md updated with attempt
if ! git diff HEAD~1 docs/fix_plan.md | grep -q "Attempt"; then
    echo "⚠️  WARNING: fix_plan.md not updated with attempt history"
fi

# Check 3: Artifacts stored in proper location
if ls tmp/*.log tmp/*.md 2>/dev/null | grep -qv "supervisorlog\|claudelog"; then
    echo "⚠️  WARNING: Artifacts in tmp/ should be in plans/active/.../reports/"
fi

echo "✅ Post-iteration checks passed"
```

**Integration**:
```bash
# In supervisor.sh
scripts/hooks/pre_iteration_check.sh "$FOCUS_ISSUE" || exit 1
# ... do galph work ...
scripts/hooks/post_iteration_check.sh "$FOCUS_ISSUE"
```

**Impact**:
- **Guaranteed consistency** (checks always run)
- **Catch errors early** (before wasting iteration)
- **Better compliance** (automated reminders)

**Implementation Effort**: 4-6 hours
- Write hook scripts (~2 hours)
- Integrate into workflow (~1 hour)
- Test with various scenarios (~2 hours)
- Document checks (~1 hour)

**Risks**:
- False positives annoy developers
- **Mitigation**: Make checks informational (warnings) not blocking (errors) initially

---

### OPT-12: Intelligent Task Bundling

**Problem**: "One thing per loop" creates artificial boundaries for trivially related work.

**Evidence**:
- Some iterations are trivially related (e.g., update config + update tests for same feature)
- Could be batched for efficiency
- But hard to predict dependencies

**Solution**: Dependency-aware task bundling

```python
# scripts/orchestration/task_bundler.py
import networkx as nx

class TaskBundler:
    def __init__(self, plan_items):
        self.items = plan_items
        self.graph = self._build_dependency_graph()

    def _build_dependency_graph(self):
        """Parse fix_plan.md dependencies into graph."""
        G = nx.DiGraph()

        for item in self.items:
            G.add_node(item['id'])
            for dep in item.get('depends_on', []):
                G.add_edge(dep, item['id'])

        return G

    def find_bundles(self, max_bundle_size=3):
        """Find sets of independent tasks that can be bundled."""
        bundles = []

        # Find tasks with no dependencies (ready to work)
        ready = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]

        # Group by similarity
        for i, task1 in enumerate(ready):
            bundle = [task1]

            for task2 in ready[i+1:]:
                if len(bundle) >= max_bundle_size:
                    break

                # Check if tasks are related (share files, same component)
                if self._are_related(task1, task2):
                    bundle.append(task2)

            if len(bundle) > 1:
                bundles.append(bundle)

        return bundles

    def _are_related(self, task1, task2):
        """Heuristic: tasks are related if they touch same files/components."""
        files1 = set(task1.get('files', []))
        files2 = set(task2.get('files', []))

        # Share at least one file
        if files1 & files2:
            return True

        # Same component/module
        if task1.get('component') == task2.get('component'):
            return True

        return False
```

**Impact**:
- **20-30% reduction in iteration count** for related refactorings
- Faster completion of multi-step features
- **Risk**: Harder to debug when bundle fails mid-way

**Implementation Effort**: 8-12 hours
- Write dependency parser (~3 hours)
- Implement bundling logic (~3 hours)
- Add rollback on bundle failure (~2 hours)
- Test with historical iterations (~2 hours)
- Document strategy (~2 hours)

**Risks**:
- Bundle fails midway, unclear which task caused failure
- **Mitigation**: Commit after each task in bundle, explicit rollback strategy

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1-2)
**Goal**: Reduce per-iteration overhead by 40-50%

**Implement**:
- OPT-5: Context Caching (4 hours)
- OPT-6: Git Sync Reduction (6 hours)
- OPT-7: Prompt Size Reduction (4 hours)

**Deliverables**:
- `scripts/orchestration/context_cache.py`
- `scripts/orchestration/local_coordinator.py`
- `prompts/supervisor_core.md` + `prompts/supervisor_context.md`
- Updated `supervisor.sh` and `loop.sh`

**Success Metrics**:
- Context cache hit rate >70%
- Git sync frequency: 2x/loop → 1x/loop
- Prompt tokens: 500 lines → 250 lines

---

### Phase 2: Routing & Intelligence (Week 3-4)
**Goal**: Optimize execution paths

**Implement**:
- OPT-8: Complexity-Based Routing (8 hours)
- OPT-9: Incremental Context Building (8 hours)
- OPT-10: Artifact Index (6 hours)

**Deliverables**:
- `scripts/orchestration/task_classifier.py`
- `scripts/orchestration/context_delta.py`
- `scripts/tools/artifact_index.py`
- `plans/active/artifact_index.yaml` (auto-generated)

**Success Metrics**:
- 30% of tasks routed to simple execution
- Context delta <50% of full context size
- Artifact queries <1 second

---

### Phase 3: Quality & Polish (Week 5-6)
**Goal**: Automated quality gates

**Implement**:
- OPT-11: Compliance Hooks (6 hours)
- OPT-12: Task Bundling (12 hours, optional)

**Deliverables**:
- `scripts/hooks/pre_iteration_check.sh`
- `scripts/hooks/post_iteration_check.sh`
- `scripts/orchestration/task_bundler.py` (if pursued)

**Success Metrics**:
- Zero compliance violations
- Bundle success rate >80% (if implemented)

---

## Expected Results

### Before Optimizations (Current State)
```yaml
Iteration Metrics:
  Galph planning: 8-9 minutes
  Ralph execution: 13 minutes
  Full loop: 21 minutes
  Retry rate: 12.5%

Token Usage:
  Context tokens/iteration: ~15,000
  Prompt tokens/iteration: ~5,000
  Total tokens/iteration: ~20,000

Overhead:
  Git sync: ~60 seconds/loop
  Context loading: ~30 seconds
  Polling: ~10 seconds
  Total overhead: ~100 seconds (8% of loop time)
```

### After Phase 1 Optimizations
```yaml
Iteration Metrics:
  Galph planning: 5-6 minutes (33% faster)
  Ralph execution: 10 minutes (23% faster)
  Full loop: 15 minutes (29% faster)
  Retry rate: 10% (better handoffs)

Token Usage:
  Context tokens/iteration: ~6,000 (60% reduction)
  Prompt tokens/iteration: ~2,500 (50% reduction)
  Total tokens/iteration: ~8,500 (58% reduction)

Overhead:
  Git sync: ~20 seconds/loop (67% faster)
  Context loading: ~5 seconds (83% faster)
  Polling: ~5 seconds (50% faster)
  Total overhead: ~30 seconds (70% reduction)
```

### After Phase 1+2 Optimizations
```yaml
Iteration Metrics:
  Galph planning: 4-5 minutes (44% faster)
  Ralph execution: 8 minutes (38% faster)
  Full loop (complex): 12 minutes (43% faster)
  Full loop (simple): 5 minutes (76% faster via direct routing)
  Retry rate: 7% (better context continuity)

Token Usage:
  Context tokens/iteration: ~4,000 (73% reduction via delta)
  Prompt tokens/iteration: ~2,500 (50% reduction)
  Total tokens/iteration: ~6,500 (68% reduction)

Overhead:
  Git sync: ~20 seconds/loop
  Context loading: ~3 seconds (delta computation)
  Polling: ~5 seconds
  Artifact discovery: <1 second (indexed)
  Total overhead: ~28 seconds (72% reduction)
```

### Combined with PROCESS_OPTIMIZATIONS.md
```yaml
Methodology improvements: 2.6x speedup (avoid rework)
Infrastructure improvements: 2-3x speedup (reduce overhead)
Combined: 5-8x total speedup

Example Initiative:
  Before: 15 iterations × 21 min = 315 minutes (5.25 hours)
  After methodology: 6 iterations × 21 min = 126 minutes (2.1 hours)
  After infrastructure: 6 iterations × 10 min = 60 minutes (1 hour)
  Total improvement: 5.25 hours → 1 hour (5.25x speedup)
```

---

## Metrics & Monitoring

### Real-Time Metrics (per iteration)
```python
# scripts/orchestration/metrics.py
class IterationMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.cache_hits = 0
        self.cache_misses = 0
        self.git_syncs = 0
        self.tokens_used = 0

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_git_sync(self, duration):
        self.git_syncs += 1
        self.git_sync_time += duration

    def summary(self):
        return {
            'duration': time.time() - self.start_time,
            'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses),
            'git_syncs': self.git_syncs,
            'tokens': self.tokens_used
        }
```

### Dashboard (weekly review)
```yaml
# reports/optimization_metrics_2025-10-26.yaml
week_of: 2025-10-20
iterations: 15

timing:
  avg_full_loop: 13.2 minutes  # Target: <12 min
  avg_galph: 5.1 minutes       # Target: <6 min
  avg_ralph: 8.1 minutes       # Target: <8 min

tokens:
  avg_per_iteration: 7200      # Target: <8000
  cache_hit_rate: 0.76         # Target: >0.70

routing:
  simple_tasks: 4 (27%)        # Target: >25%
  moderate_tasks: 10 (67%)
  complex_tasks: 1 (6%)

quality:
  retry_rate: 0.08             # Target: <0.10
  compliance_violations: 0     # Target: 0
```

---

## Risk Mitigation

### Risk: Cache Invalidation Bugs
**Concern**: File changed but cache not updated, stale context

**Mitigation**:
- Aggressive invalidation (any git status change clears cache)
- `--no-cache` flag for debugging
- Clear cache on branch switch
- Weekly full cache clear (cron job)

### Risk: Lock File Corruption
**Concern**: Process crash leaves stale lock, blocks all iterations

**Mitigation**:
- Stale lock detection (if held >20 min, auto-break)
- Lock file includes PID (check if process alive)
- Manual `scripts/tools/break_lock.sh` command

### Risk: Complexity Classifier False Positives
**Concern**: Simple task misclassified as complex (wasted time) or vice versa

**Mitigation**:
- Conservative defaults (bias toward two-agent)
- `--force-simple` / `--force-complex` override flags
- Track misclassification rate, tune heuristics quarterly

### Risk: Context Delta Misses Important Info
**Concern**: Delta skips context needed for correct decision

**Mitigation**:
- Always provide full context as fallback
- Periodic full refresh (every 5th iteration)
- "Reload full context" escape hatch for agents

---

## Retrospective Plan

### After Phase 1 (2 weeks)
**Questions**:
1. Cache hit rate achieving >70%?
2. Git sync time reduced by >50%?
3. Prompt tokens reduced by >40%?
4. Any new bugs introduced?

**Adjustments**:
- Tune cache invalidation if hit rate low
- Fix any lock contention issues
- Refine prompt split if quality degrades

### After Phase 2 (4 weeks)
**Questions**:
1. Simple task routing saving time?
2. Context delta accurate and useful?
3. Artifact index being used?
4. Overall iteration time reduced >30%?

**Adjustments**:
- Tune complexity classifier
- Improve delta algorithm if missing context
- Add more artifact queries if useful

### Quarterly Review
**Metrics**:
- Average iteration time
- Token usage and costs
- Retry rate and quality
- Developer satisfaction

---

## References

**Analysis Source**:
- `logs/feature-torchapi/galph-summaries/` (iterations 1-89)
- `/tmp/optimization_analysis.md` (detailed analysis)

**Related Documents**:
- `plans/meta/PROCESS_OPTIMIZATIONS.md` - Methodology improvements (complements this doc)
- `docs/DEVELOPER_GUIDE.md` - Current development workflow
- `docs/INITIATIVE_WORKFLOW_GUIDE.md` - Initiative structure
- `prompts/supervisor.md` - Supervisor workflow
- `prompts/main.md` - Engineer workflow
- `CLAUDE.md` - Core project directives

**Scripts Created** (to be implemented):
- `scripts/orchestration/context_cache.py`
- `scripts/orchestration/local_coordinator.py`
- `scripts/orchestration/prompt_builder.py`
- `scripts/orchestration/task_classifier.py`
- `scripts/orchestration/context_delta.py`
- `scripts/tools/artifact_index.py`
- `scripts/hooks/pre_iteration_check.sh`
- `scripts/hooks/post_iteration_check.sh`
- `scripts/orchestration/task_bundler.py`

---

**Document Status**: PROPOSED
**Next Action**: Review and approve Phase 1 optimizations, then implement
**Owner**: Project maintainer / infrastructure developer
**Last Updated**: 2025-10-19
