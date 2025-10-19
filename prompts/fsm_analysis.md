# Galph/Ralph as Finite State Machines: Analysis and Diagrams

## Overview

This document analyzes whether the Galph/Ralph two-agent development system can be understood as a finite state machine (FSM), provides detailed state diagrams, and characterizes what formal model best describes the system.

## Executive Summary

**The Galph/Ralph system exhibits strong FSM-like control structure at the protocol level, but operates on unbounded state space at the computational level.**

The system is best characterized as a **Hierarchical Extended Finite State Machine (HEFSM)** or **Protocol State Machine** with:
- Finite, discrete control states
- Bounded transition rules with guards
- Unbounded memory (code, docs, git history)
- Complex semantic computations (LLM inference, code analysis)

---

## FSM-Like Characteristics

### 1. Explicit State Tracking
From `prompts/supervisor.md`, Galph must declare an **Action State** for each focus issue:
- `[gathering_evidence]`
- `[planning]`
- `[ready_for_implementation]`

**Bounded dwell constraint**: Maximum 2 consecutive turns in same state before forced transition or focus change.

### 2. Discrete Action Types
Galph selects from a **finite set**:
- Evidence Collection (with Callchain Tracing subtype)
- Debug (Hypothesis + Triage)
- Planning
- Review/Housekeeping

### 3. Deterministic Control Flow
Both agents follow **structured step sequences**:
- **Galph**: Steps 0→1→2→3→3.1→3.2→3.5→3.7→4
- **Ralph (debug mode)**: Steps 0→1→2→3→4→4.5→5→6→7

### 4. Pass/Fail Gates
Clear **transition guards**:
- Galph: "If in same state > 2 turns → MUST transition"
- Ralph: "If tests fail → rollback code, keep docs"
- "Start Gate": Must have `in_progress` item before proceeding
- "End Gate": Must have Attempts History before commit

---

## Non-FSM Characteristics

### 1. Unbounded State Space
- `docs/fix_plan.md` can grow indefinitely (unbounded task list)
- `galph_memory.md` accumulates historical context (though pruned, conceptually unbounded)
- The "focus issue" can be any entry from an unbounded set
- **The entire codebase is part of the state** (evolves unboundedly)
- Artifact directories accumulate historical data

### 2. Unbounded Memory
A true FSM has **no memory** beyond its current state. But:
- Galph remembers prior attempts via `galph_memory.md`
- The ledger (`docs/fix_plan.md`) is persistent memory
- Git history is unbounded memory
- Plans under `plans/active/` accumulate context

### 3. Non-Deterministic Transitions
- "Flip a coin" (step 2 in supervisor.md) - probabilistic choice
- LLM-based decisions are fundamentally **non-deterministic**
- Focus issue selection from potentially many candidates
- "Think deeply" steps involve semantic reasoning, not simple state lookup

### 4. Complex Semantic Guards
Transitions depend on:
- Code analysis results (unbounded complexity)
- Test execution outcomes (arbitrary programs)
- Documentation semantic analysis (LLM inference)
- Not simple boolean conditions on finite state

### 5. No Terminal States
The system is designed to **run indefinitely**:
- Individual fix_plan items can reach `done`, but the system continues
- There's no global "accepting state" - it's a continuous improvement loop

---

## Better Formal Models

### 1. Hierarchical Extended Finite State Machine (HEFSM)
- **States**: Action states + control flow states
- **Variables**: focus_issue, mode, action_type, turn_count (bounded), fix_plan_content (unbounded!)
- **Guards**: Turn count checks, test results, git sync status
- **Actions**: Complex computations (LLM inference, code analysis, test execution)
- **Hierarchy**: Phases contain checklists contain sub-tasks

### 2. Pushdown Automaton (PDA)
- Plans have nested structure (phases → checklists → sub-items)
- Can "push" into a sub-task and "pop" back to parent context
- The call stack of plans is more powerful than FSM but still formal

### 3. Two-Level Control Architecture
- **Outer loop** (meta-level): Galph's action state machine - relatively simple FSM
- **Inner loop** (object-level): Ralph's execution - unbounded computation
- Similar to CPU (FSM) + Memory (unbounded) architecture

### 4. Protocol State Machine
Like network protocols (TCP, TLS handshake):
- Defined message sequences (input.md → Ralph → git → Galph)
- States represent conversation position
- Guards ensure protocol compliance
- But richer semantics than typical protocols

### 5. OODA Loop (Observe-Orient-Decide-Act)
Military decision-making cycle:
- **Observe**: Read fix_plan, git state, test results
- **Orient**: Analyze context, review docs
- **Decide**: Select action type and focus issue
- **Act**: Execute, commit, update ledgers
- Iterative refinement with learning

---

## Verdict

**The system can be partially modeled as an FSM, but only at the control-flow level, not at the full semantic level.**

### What IS an FSM:
The **control discipline**:
- Bounded action states with forced transitions
- Discrete action type enumeration
- Step sequence structure
- Pass/fail gates

### What is NOT an FSM:
The **computational content**:
- Unbounded state (code, docs, history)
- Unbounded memory (ledgers, artifacts)
- Semantic reasoning (LLM decisions)
- Complex computation (test execution, code analysis)

### Analogy: CPU Architecture
Like a **CPU executing arbitrary programs**:
- CPU microarchitecture: FSM (fetch-decode-execute states)
- Program behavior: Not FSM (unbounded loops, recursion, memory)

The Galph/Ralph system:
- Control protocol: FSM-like (states, guards, transitions)
- Work execution: General computation (code changes, analysis, LLM inference)

### Best Technical Term:
**"Formally structured agentic protocol with FSM control discipline"**

It has the **rigor** of an FSM (explicit states, bounded transitions, deterministic steps) but the **power** of a general agent system (semantic reasoning, unbounded problem-solving, learning from feedback).

---

# State Diagrams

## Diagram 1: Galph Action State Machine (Simple View)

```
┌─────────────────────────────────────────────────────────────────┐
│                      GALPH SUPERVISOR FSM                        │
└─────────────────────────────────────────────────────────────────┘

                    ┌──────────────────┐
                    │   LOOP START     │
                    │  (git pull)      │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Read Context    │
                    │ (fix_plan.md,    │
                    │  galph_memory)   │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │  SELECT ACTION STATE        │
              │  for current focus_issue    │
              └──┬───────────┬──────────┬───┘
                 │           │          │
        ┌────────▼──┐   ┌────▼────┐   ┌▼──────────┐
        │gathering_ │   │planning │   │ready_for_ │
        │evidence   │   │         │   │implement  │
        │           │   │         │   │           │
        │[turn ≤ 2] │   │[turn≤2] │   │[execute]  │
        └────┬──────┘   └────┬────┘   └─────┬─────┘
             │               │               │
             │ turn++        │ turn++        │ turn=0
             │ if turn>2     │ if turn>2     │
             │ ──────┐       │ ──────┐       │
             │       │       │       │       │
             ▼       ▼       ▼       ▼       ▼
        ┌────────────────────────────────────────┐
        │    SELECT ACTION TYPE                  │
        │  ┌──────────┬──────────┬──────────┐   │
        │  │Evidence  │  Debug   │ Planning │   │
        │  │Collection│ (H+T)    │          │   │
        │  └────┬─────┴─────┬────┴────┬─────┘   │
        │       │           │         │         │
        │       │  ┌────────┴─────┐   │         │
        │       │  │Review/House- │   │         │
        │       │  │  keeping     │   │         │
        │       │  └──────────────┘   │         │
        └───────┼─────────┼───────────┼─────────┘
                │         │           │
                ▼         ▼           ▼
        ┌────────────────────────────────────┐
        │   EXECUTE ACTION                   │
        │   (read docs, analyze, write)      │
        └────────────────┬───────────────────┘
                         │
                ┌────────▼─────────┐
                │  Write input.md  │
                │  for Ralph       │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │ Update Memory    │
                │ (galph_memory,   │
                │  fix_plan)       │
                └────────┬─────────┘
                         │
                ┌────────▼─────────┐
                │  Git Commit      │
                │  & Push          │
                └────────┬─────────┘
                         │
                         ▼
                    ┌─────────┐
                    │   END   │◄─────── (next invocation
                    │  LOOP   │          starts here)
                    └─────────┘
```

**Key Features:**
- **Bounded state**: 3 action states with max 2-turn dwell time
- **Forced transitions**: System cannot remain stuck in evidence/planning indefinitely
- **Deterministic steps**: Structured sequence from context → selection → execution → handoff
- **Loop invariant**: Each iteration produces input.md for Ralph and updates shared state

---

## Diagram 2: Ralph Engineer FSM (Debug Mode)

```
┌─────────────────────────────────────────────────────────────────┐
│                    RALPH ENGINEER FSM (DEBUG)                    │
└─────────────────────────────────────────────────────────────────┘

    ┌──────────────┐
    │ START LOOP   │
    │ (git pull)   │
    └──────┬───────┘
           │
    ┌──────▼───────┐
    │ Read input.md│
    │ Read fix_plan│
    └──────┬───────┘
           │
    ┌──────▼────────────────┐
    │ Step 0: SETUP         │
    │ - Read specs/arch     │
    │ - Verify fix_plan     │
    │ - Check parity matrix │
    └──────┬────────────────┘
           │
    ┌──────▼────────────────┐
    │ Step 1: REPRODUCE     │
    │ - Map AT→test         │
    │ - Run parity command  │
    │ - Capture metrics     │
    └──────┬────────────────┘
           │
    ┌──────▼────────────────┐
    │ Step 2: TRIAGE        │
    │ - Geometry checklist  │
    │ - Units/conventions   │
    │ - Pivots/invariants   │
    └──────┬────────────────┘
           │
    ┌──────▼────────────────┐
    │ Step 3: TRACE         │
    │ - Generate C trace    │
    │ - Generate Py trace   │
    │ - Find 1st divergence │
    └──────┬────────────────┘
           │
    ┌──────▼────────────────┐
    │ Step 4: FIX           │
    │ - Minimal code change │
    │ - Re-run failing case │
    └──────┬────────────────┘
           │
    ┌──────▼────────────────┐
    │ Step 5: GATES         │
    │ - Re-run parity tests │
    │ - Check thresholds    │
    └──────┬────────────────┘
           │
           │   ┌────PASS?─────┐
           │   │              │
      ┌────▼───▼──┐      ┌────▼────┐
      │   PASS    │      │  FAIL   │
      │           │      │         │
      │ Keep code │      │ Rollback│
      │ Keep docs │      │  code   │
      │           │      │ Keep    │
      │           │      │ docs    │
      └────┬──────┘      └────┬────┘
           │                  │
           └──────┬───────────┘
                  │
         ┌────────▼────────────┐
         │ Step 6: FINALIZE    │
         │ - Full test suite   │
         │ - Update fix_plan   │
         │ - Code review       │
         └────────┬────────────┘
                  │
         ┌────────▼────────────┐
         │ Git Commit & Push   │
         └────────┬────────────┘
                  │
         ┌────────▼────────────┐
         │    END LOOP         │
         └─────────────────────┘
```

**Key Features:**
- **Linear state progression**: Steps 0→1→2→3→4→5→6 (debugging workflow)
- **Binary decision gate**: PASS/FAIL at Step 5 determines code retention
- **Rollback mechanism**: Failed attempts preserve docs but revert code
- **Mandatory checkpoints**: Cannot skip steps (enforced by prompts/debug.md)

---

## Diagram 3: Galph-Ralph Interaction (Two-Agent System)

```
┌──────────────────────────────────────────────────────────────────────┐
│                    GALPH-RALPH TWO-AGENT LOOP                         │
└──────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────┐                    ┌──────────────────────┐
  │   GALPH DOMAIN      │                    │    RALPH DOMAIN      │
  │   (Supervisor)      │                    │    (Engineer)        │
  └─────────────────────┘                    └──────────────────────┘
           │                                            │
           │                                            │
    ┌──────▼──────┐                            ┌───────▼────────┐
    │ Read State  │                            │  Wait for      │
    │ - fix_plan  │                            │  input.md      │
    │ - memory    │                            │                │
    │ - git log   │                            │  (idle)        │
    └──────┬──────┘                            └────────────────┘
           │                                            ▲
    ┌──────▼──────┐                                    │
    │  Analyze    │                                    │
    │  & Plan     │                                    │
    │             │                                    │
    │ Select:     │                                    │
    │ - Focus     │                                    │
    │ - Action    │                                    │
    │ - Mode      │                                    │
    └──────┬──────┘                                    │
           │                                            │
    ┌──────▼──────┐                                    │
    │   Write     │                                    │
    │  input.md   │──────────────┐                     │
    │             │              │                     │
    │ Do Now:     │              │                     │
    │ - Task      │              │                     │
    │ - Tests     │              │                     │
    │ - Artifacts │              │                     │
    └──────┬──────┘              │                     │
           │                     │                     │
    ┌──────▼──────┐              │                     │
    │ Git Commit  │              │                     │
    │   & Push    │              │                     │
    └──────┬──────┘              │                     │
           │                     │                     │
    ┌──────▼──────┐         ┌────▼─────────────┐       │
    │   Signal    │         │  input.md        │       │
    │   Ralph     │────────►│  (handoff)       │───────┤
    │             │         │                  │       │
    └─────────────┘         └──────────────────┘       │
           │                                            │
           │                                     ┌──────▼──────┐
           │                                     │ Ralph Reads │
           │                                     │  input.md   │
           │                                     └──────┬──────┘
           │                                            │
           │                                     ┌──────▼──────┐
           │                                     │  Execute    │
           │                                     │  Do Now     │
           │                                     │             │
           │                                     │ - Run tests │
           │                                     │ - Edit code │
           │                                     │ - Generate  │
           │                                     │   artifacts │
           │                                     └──────┬──────┘
           │                                            │
           │                                     ┌──────▼──────┐
           │                                     │   Update    │
           │                                     │  fix_plan   │
           │                                     │  (Attempts  │
           │                                     │   History)  │
           │                                     └──────┬──────┘
           │                                            │
           │         ┌──────────────────┐        ┌──────▼──────┐
           │         │  Git State       │        │ Git Commit  │
           │         │  (updated code,  │◄───────│   & Push    │
           │         │   docs, tests)   │        │             │
           │         └──────────────────┘        └─────────────┘
           │                     │                      │
           │                     │                      │
    ┌──────▼──────┐              │                      │
    │   Review    │              │                      │
    │  Ralph's    │◄─────────────┴──────────────────────┘
    │   Work      │
    │             │
    │ - Read diff │
    │ - Check     │
    │   attempts  │
    │ - Evaluate  │
    └──────┬──────┘
           │
           │
           └──────► (Next Galph iteration)


  ┌─────────────────────────────────────────────────────┐
  │  SHARED STATE (Git Repository)                      │
  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
  │  - docs/fix_plan.md         (ledger)                │
  │  - galph_memory.md          (supervisor memory)     │
  │  - input.md                 (task directive)        │
  │  - plans/active/*/          (phased plans)          │
  │  - Code, tests, artifacts   (work products)         │
  └─────────────────────────────────────────────────────┘
```

**Key Features:**
- **Message-passing protocol**: input.md serves as typed message from Galph to Ralph
- **Shared state synchronization**: Git commits create synchronization points
- **Asynchronous execution**: Agents run in separate invocations (not concurrent)
- **State persistence**: Git repository acts as durable shared memory

---

## Diagram 4: Hierarchical State Machine (Full View)

```
┌────────────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL FSM VIEW                                │
└────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── META-LEVEL (Galph) ────────────────────────────┐
│                                                                         │
│   Action States (max 2 turns each):                                    │
│                                                                         │
│   ╔═══════════════════╗  turn++  ╔════════════╗  turn++  ╔═══════════╗│
│   ║ GATHERING_        ║─────────►║  PLANNING  ║─────────►║  READY_   ║│
│   ║ EVIDENCE          ║          ║            ║          ║  FOR_IMPL ║│
│   ║                   ║◄─────────║            ║◄─────────║           ║│
│   ║ [turn ≤ 2]        ║  change  ║ [turn ≤ 2] ║  blocked ║ [execute] ║│
│   ╚═══════════════════╝  focus   ╚════════════╝          ╚═══════════╝│
│            │                           │                        │       │
│            │ if turn > 2               │ if turn > 2            │       │
│            └───────────┬───────────────┘                        │       │
│                        │                                         │       │
│                        ▼                                         │       │
│                  ┌──────────┐                                   │       │
│                  │ BLOCKED  │                                   │       │
│                  │ Change   │                                   │       │
│                  │ focus or │                                   │       │
│                  │ delegate │                                   │       │
│                  └──────────┘                                   │       │
│                                                                  │       │
│   Action Types (selected per turn):                             │       │
│   ┌─────────────┬──────────┬──────────┬──────────────┐         │       │
│   │ Evidence    │  Debug   │ Planning │ Review/      │         │       │
│   │ Collection  │  (H+T)   │          │ Housekeeping │         │       │
│   └─────────────┴──────────┴──────────┴──────────────┘         │       │
│                                                                  ▼       │
└─────────────────────────────────────────────────────────────────┼───────┘
                                                                   │
                                     input.md                      │
                                     (handoff)                     │
                                                                   │
┌──────────────────────── OBJECT-LEVEL (Ralph) ──────────────────┼───────┐
│                                                                  │       │
│   Mode Flags (modulate execution):                              │       │
│   ┌────────┬─────────┬────────┬───────┬────────┐              │       │
│   │  TDD   │ Parity  │  Perf  │  Docs │  none  │              │       │
│   └────────┴─────────┴────────┴───────┴────────┘              │       │
│                                                                  │       │
│   Control Flow States:                                          │       │
│                                                                  ▼       │
│   ┌─────────┐     ┌──────────┐     ┌─────────┐     ┌──────────────┐   │
│   │  SETUP  │────►│REPRODUCE │────►│ TRIAGE  │────►│    TRACE     │   │
│   │ (Step 0)│     │ (Step 1) │     │(Step 2) │     │   (Step 3)   │   │
│   └─────────┘     └──────────┘     └─────────┘     └──────┬───────┘   │
│                                                             │           │
│   ┌─────────────────────────────────────────────────────────┘           │
│   │                                                                     │
│   ▼                                                                     │
│   ┌─────────┐     ┌──────────┐     ┌──────────────┐                   │
│   │   FIX   │────►│  GATES   │────►│   Decision   │                   │
│   │(Step 4) │     │ (Step 5) │     │   (Pass/Fail)│                   │
│   └─────────┘     └──────────┘     └───┬──────┬───┘                   │
│                                         │      │                        │
│                                     PASS│      │FAIL                    │
│                                         │      │                        │
│                        ┌────────────────┘      └────────────┐          │
│                        ▼                                      ▼          │
│                  ┌──────────┐                          ┌──────────┐    │
│                  │ FINALIZE │                          │ ROLLBACK │    │
│                  │ (Step 6) │                          │  + LOG   │    │
│                  │          │                          │          │    │
│                  │ - Tests  │                          │ - Revert │    │
│                  │ - Docs   │                          │   code   │    │
│                  │ - Review │                          │ - Keep   │    │
│                  └────┬─────┘                          │   docs   │    │
│                       │                                └────┬─────┘    │
│                       │                                     │          │
│                       └───────────┬─────────────────────────┘          │
│                                   ▼                                     │
│                            ┌─────────────┐                             │
│                            │ GIT COMMIT  │                             │
│                            │   & PUSH    │                             │
│                            └──────┬──────┘                             │
│                                   │                                     │
└───────────────────────────────────┼─────────────────────────────────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │   Signal Galph     │
                         │   (next iteration) │
                         └────────────────────┘
```

**Key Features:**
- **Two-level hierarchy**: Meta-level (Galph planning) and object-level (Ralph execution)
- **Mode modulation**: TDD/Parity/Perf/Docs flags parameterize Ralph's behavior
- **Cross-layer handoff**: input.md carries mode + focus + action from Galph to Ralph
- **Bounded meta-states**: Galph's 3 action states with 2-turn limit
- **Unbounded object-states**: Ralph can execute arbitrary code changes

---

## Diagram 5: Fix Plan Item Lifecycle

```
┌────────────────────────────────────────────────────────────────┐
│           FIX PLAN ITEM LIFECYCLE (FSM per item)               │
└────────────────────────────────────────────────────────────────┘

                      ┌──────────┐
                      │   NEW    │
                      │ (created │
                      │ by Galph)│
                      └────┬─────┘
                           │
                  Galph selects
                           │
                      ┌────▼─────┐
             ┌────────│ PENDING  │◄──────┐
             │        │          │       │
             │        └──────────┘       │
             │             │             │
             │   Galph/Ralph             │
             │    selects                │
             │             │             │
             │        ┌────▼─────┐       │
             │        │IN_PROGRESS│      │
             │        │          │       │
             │        │ Attempt N│       │
             │        └────┬─────┘       │
             │             │             │
             │    Ralph executes         │
             │             │             │
             │    ┌────────▼─────┐       │
             │    │ ATTEMPTED    │       │
             │    │              │       │
             │    │ (Attempts    │       │
             │    │  History     │       │
             │    │  updated)    │       │
             │    └───┬──────────┘       │
             │        │                  │
             │        │                  │
             │    Outcome?               │
             │        │                  │
      ┌──────┴────┬───┴───┬─────────┐   │
      │           │       │         │   │
  ┌───▼──┐   ┌───▼──┐ ┌──▼───┐ ┌──▼───┴───┐
  │FAILED│   │PARTIAL│ │SUCCESS│ │ BLOCKED  │
  │      │   │       │ │       │ │          │
  │Keep  │   │Keep   │ │Mark   │ │Document  │
  │active│   │active │ │DONE   │ │blocker   │
  └───┬──┘   └───┬───┘ └──┬────┘ └──┬───────┘
      │          │        │         │
      │          │        │         │
      │     Retry│        │         │Wait for
      │     (incr│        │         │dependency
      │     attempt)      │         │
      │          │        │         │
      └──────────┼────────┼─────────┘
                 │        │
                 │        ▼
                 │   ┌─────────┐
                 │   │  DONE   │
                 │   │         │
                 │   │(Status: │
                 │   │complete)│
                 │   └─────────┘
                 │        │
                 │        │
                 │   ┌────▼─────┐
                 │   │ARCHIVED  │
                 │   │(moved to │
                 │   │archive/) │
                 │   └──────────┘
                 │
                 └──► (stays in fix_plan
                      for next attempt)
```

**Key Features:**
- **Per-item FSM**: Each fix_plan.md entry follows this lifecycle independently
- **Persistent retry**: FAILED/PARTIAL items remain IN_PROGRESS for next attempt
- **Success terminal**: Only SUCCESS outcome marks item DONE
- **Dependency blocking**: BLOCKED state for items waiting on other work
- **Archive path**: DONE items eventually moved to archive/ when fix_plan grows large

---

## Summary Table: FSM vs Non-FSM Features

| Feature | FSM-Like? | Details |
|---------|-----------|---------|
| **Control States** | ✅ Yes | 3 action states (gathering_evidence, planning, ready_for_impl) |
| **Action Types** | ✅ Yes | 4 discrete types (Evidence, Debug, Planning, Review) |
| **Transitions** | ✅ Yes | Bounded turn count (≤2), forced transitions |
| **Gates** | ✅ Yes | Pass/fail, start gate, end gate |
| **Step Sequence** | ✅ Yes | Deterministic order (0→1→2→...→6) |
| **State Space** | ❌ No | Unbounded (code, docs, git history) |
| **Memory** | ❌ No | Persistent ledgers, artifacts, git history |
| **Decisions** | ❌ No | LLM-based semantic reasoning |
| **Guards** | ❌ No | Complex (test results, code analysis) |
| **Terminal States** | ❌ No | Continuous loop, no global accept state |

---

## Implications for System Design

### Strengths of FSM Structure
1. **Predictable control flow**: Debugging is easier with explicit states
2. **Bounded loops**: 2-turn limit prevents infinite stalling
3. **Clear handoffs**: input.md protocol is well-defined
4. **Auditability**: State transitions are logged in git history

### Limitations of FSM Model
1. **Cannot capture semantic complexity**: LLM decisions, code analysis
2. **Unbounded memory not modeled**: Git history, ledgers grow indefinitely
3. **Task complexity varies**: Some items take 1 attempt, others 20+
4. **Non-determinism**: LLM responses, test outcomes, environment

### Design Recommendations
1. **Maintain FSM discipline**: Keep control states simple and bounded
2. **Separate control from computation**: FSM manages flow, LLM handles semantics
3. **Explicit state tracking**: Always log current action state in galph_memory.md
4. **Bounded retry limits**: Consider max attempts per fix_plan item
5. **Escape hatches**: Allow manual intervention when FSM constraints conflict with progress

---

## References

- `prompts/supervisor.md`: Galph action state definitions and transitions
- `prompts/debug.md`: Ralph debugging workflow steps
- `prompts/main.md`: General Ralph loop mechanics
- `docs/INITIATIVE_WORKFLOW_GUIDE.md`: Phase-based planning structure
- `docs/fix_plan.md`: Living ledger of all work items

---

**Document Version:** 1.0
**Created:** 2025-10-19
**Author:** Claude Code (Sonnet 4.5)
