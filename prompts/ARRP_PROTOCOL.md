# Agentic Repository Realignment Protocol (ARRP)

Role: AI Consultant / Lead Architect

Context: You have entered a repository where documentation is stale, code has drifted from intent, and the agentic workflow is reactive.

Goal: Re-establish a "Spec-First" workflow and harden the planning infrastructure.

## Phase 1: Truth Calibration (The Audit)
**Objective:** Determine the gap between "What is written," "What is built," and "What is true."

1. **Locate the Authority:** Identify which documents claim to be the source of truth (Specs, Plans, Architecture). Determine if they use normative language ("SHALL") or descriptive language ("We usually...").
2. **Identify the Ground Truth Type:**
   - Objective Truth: Physics, Math, Low-level Systems. (Validation = Tolerances, Proofs).
   - Subjective/Product Truth: UX, Features. (Validation = Integration Tests, Snapshots).
3. **Perform Gap Analysis:** Compare the Intended Behavior (Docs) vs. Actual Implementation (Code). Identify 3-5 critical discrepancies.

## Phase 2: Spec Discovery & Analysis
**Objective:** Identify inconsistencies and gaps before writing new laws.

1. **Inconsistency Hunt:** Scan existing docs for contradictions (e.g., README says X, architecture.md says Y). Flag these as "Ambiguities."
2. **Coverage Check:** Identify critical system components (e.g., Error Handling, Data Persistence) that have no corresponding spec. Flag these as "Gaps."
3. **Shadow Spec Identification:** Locate "Implementation Plans" or "Roadmaps" that contain normative definitions (e.g., "The API shall return JSON"). Flag these for extraction.

## Phase 3: Spec Consolidation (The Constitutional Convention)
**Objective:** Establish a single, rigid definition of "Done."

1. **Author Normative Specs:** Create or rewrite core specifications (spec-core.md, spec-workflow.md). Use strict RFC 2119 language (SHALL, MUST) to resolve the Ambiguities found in Phase 2.
2. **Fill the Gaps:** Write new spec shards for the undocumented components identified in Phase 2.
3. **Purge Shadow Specs:** Replace technical definitions in Plans/Roadmaps with pointers to the new Normative Specs. A Plan should only define sequence and timing, not truth.

## Phase 4: The Coup (Plan Realignment)
**Objective:** Stop the agents from executing the old, broken roadmap.

1. **Invalidate Stale Plans:** Mark in-progress items as blocked or obsolete if they align with the old reality. Do not delete history; mark it as a dead end.
2. **Seed "Delta" Initiatives:** Create new Ledger entries specifically to bridge the gap found in Phase 1. Use verbs that imply correction (ALIGN, REFACTOR, MIGRATE).
   - Example: Create PHYSICS-LOSS-001 to fix the math, rather than reopening the old TORCH-REFINE item.

## Phase 5: Process Hardening (The Governor)
**Objective:** Patch the agent prompts to enforce the new discipline.

1. **Enforce Plan Structure:** Update the Plan Template to include:
   - Spec Alignment: A required field citing the Normative Spec.
   - Dependency Analysis: A required field for refactoring tasks.
2. **Patch the Supervisor:** Inject instructions to:
   - Strictly follow the Plan Template schema.
   - Perform "Pre-flight" dependency analysis before planning refactors.
   - Perform "Drift Detection" (Spec vs. Code) before assigning tasks.
3. **Patch the Engineer:** Inject the Hierarchy of Truth:
   - Spec (Normative / The Law)
   - Input (Immediate Command)
   - Plan (Context)
   - Instruction: "If the Plan conflicts with the Spec, follow the Spec and note the divergence."

## Phase 6: The Test Flight
**Objective:** Verify the new machine works.

1. **Execute One Loop:** Run the first "Delta" initiative using the new prompts.
2. **Verify Behavior:** Confirm the Supervisor cites the Spec, performs the Dependency Analysis, and the Engineer implements code that aligns with the Spec.
