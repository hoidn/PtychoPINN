# Agentic Prompt Migration Guide (Project‑Agnostic Template)

This template guides migrating and maintaining agent prompts under `prompts/` so they stay aligned with a project’s living documentation, specs, testing guidance, and artifact policies. It is designed to bootstrap from the project’s `docs/index.md` and light repository scans so it works across heterogeneous codebases.

## Purpose
- Keep supervisor/engineer prompts synchronized with current sources of truth.
- Prevent stale paths, broken test mappings, and missing artifact destinations.
- Provide a repeatable, discovery‑first process that adapts to any repository with a minimal set of assumptions.

## Scope and Assumptions
- Applies to any repository that contains:
  - `docs/` with an index file (`docs/index.md`) or equivalent index.
  - `prompts/` containing one or more prompt files (e.g., `prompts/supervisor.md`, `prompts/main.md`, `prompts/debug.md`).
  - Optionally `tests/` and `plans/` (structures may vary).
- Legacy projects often include a ledger at `docs/fix_plan.md`; new projects may need to initialize it (instructions below).
- If present, `CLAUDE.md` defines agent workflow rules (with `AGENTS.md` symlinked to it) and must be respected.

## Outcomes
- Prompts reference the correct, living documents discovered in this project.
- A lightweight mapping of “authoritative sources” is captured so it can be refreshed later.
- Artifact routing is consistent: `plans/active/<initiative-id>/reports/<YYYY-MM-DDTHHMMSSZ>/`.
- A fix‑plan ledger exists at `docs/fix_plan.md` and records loop attempts and artifact paths.
- Agent workflow doc (`CLAUDE.md`, exposed via the `AGENTS.md` symlink) is present and linked from the doc index.

---

## Pre‑Bootstrap: Profile the “Model” Project Structure

When migrating prompts into a new codebase, first profile a known‑good “model” project whose agent workflow you want to emulate. The goal is to extract patterns and a dependency graph before bootstrapping the new repository.

Terminology
- Model project: the reference repository with working prompts and agent docs.
- Target project: the new or different repository receiving the migrated prompts.

Environment variables (suggested)
- `MODEL_ROOT=<path-to-model-repo>`
- `TARGET_ROOT=<path-to-target-repo>`

Model inventory (from `MODEL_ROOT`)
```
cd "$MODEL_ROOT"
rg --files prompts | rg "\.md$" || true
rg --files docs | rg -n "index\.md$|fix_plan\.md$|findings\.md$|architecture|developer|TESTING_GUIDE|TEST_SUITE_INDEX|debug|workflow" || true
rg --files | rg -n "^CLAUDE\.md$|AGENTS\.md$" || true
rg --files specs | rg "\.md$" || true
rg --files docs/debugging | rg -n "\.md$|\.mdx$|\.ipynb$" || true
rg --files scripts/orchestration | rg -n "\.py$|\.sh$" || true
```

Capture a concise structure summary with categories and paths (free‑form JSON or Markdown). This becomes the basis for the migration plan and dependency graph.

Target inventory (from `TARGET_ROOT`)
```
cd "$TARGET_ROOT"
rg --files prompts | rg "\.md$" || true
rg --files docs | rg -n "index\.md$|fix_plan\.md$|findings\.md$|architecture|developer|TESTING_GUIDE|TEST_SUITE_INDEX|debug|workflow" || true
rg --files | rg -n "^CLAUDE\.md$|AGENTS\.md$" || true
rg --files specs | rg "\.md$" || true
rg --files docs/debugging | rg -n "\.md$|\.mdx$|\.ipynb$" || true
rg --files scripts/orchestration | rg -n "\.py$|\.sh$" || true
```

Diff the two inventories to identify:
- Documents to copy verbatim and adapt (paths, naming, policies).
- Documents to derive from the target repo’s existing code/tests (e.g., testing index, workflows).
- Documents that must be authored ab initio (e.g., findings ledger for a new codebase).

Store both inventories under the migration artifacts directory for traceability.

## Bootstrap: Discover Authoritative Sources

Always begin with the project’s documentation index, then locate category docs by scanning. Record results as placeholders for later substitution in prompts.

1) Read `docs/index.md` (or top‑level README if missing) to learn canonical locations.

2) Identify authoritative docs by category and record placeholders:
- Specs (normative contracts)
  - Likely under `specs/*.md` or discoverable via keywords “specification”, “contract”, “API”.
  - Placeholder: `<SPEC_DOCS>` (list of paths).
- Architecture and developer guidance
  - Common: `docs/architecture.md`, ADRs under `docs/architecture/` or `docs/adr*/`, developer guides like `docs/DEVELOPER_GUIDE.md`.
  - Placeholders: `<ARCH_DOC>`, `<DEV_GUIDE>`.
- Testing
  - Guides or suite maps: e.g., `docs/TESTING_GUIDE.md`, `docs/development/TEST_SUITE_INDEX.md`.
  - Placeholders: `<TEST_GUIDE>`, `<TEST_INDEX>`.
- Debugging and workflows
  - E.g., `docs/debugging/debugging.md`, `docs/workflows/*.md`.
  - Placeholders: `<DEBUG_GUIDE>`, `<WORKFLOW_DOCS>` (list).
- Findings, ledger, and agent workflow
  - Findings ledger: `docs/findings.md` (if present).
  - Fix plan ledger: `docs/fix_plan.md` (create if missing).
- Agent workflow: `CLAUDE.md` (root policy surfaced via `AGENTS.md` symlink).
  - Placeholders: `<FINDINGS_DOC>`, `<LEDGER_DOC>`, `<AGENT_WORKFLOW_DOCS>`.

3) Discovery helpers (run from repo root; use whichever apply):
```
rg -n "specification|contract|API" docs specs || true
rg --files specs | rg "\.md$" || true
rg -n "architecture|ADR|developer guide" docs || true
rg --files docs | rg -i "architecture|adr|developer|guide.*\.md$" || true
rg -n "pytest|testing|test suite|selector" docs || true
rg --files docs/development | rg -i "test.*index.*\.md$" || true
rg --files docs | rg -i "fix_plan\.md|findings\.md|workflow|debug" || true
rg --files prompts | rg "\.md$" || true
```

4) Optional mapping artifact for this run (kept up to date manually):
`docs/prompt_sources_map.json`
```
{
  "specs": ["<SPEC_DOCS>"],
  "architecture": "<ARCH_DOC>",
  "developer_guide": "<DEV_GUIDE>",
  "testing_guide": "<TEST_GUIDE>",
  "test_index": "<TEST_INDEX>",
  "debugging": "<DEBUG_GUIDE>",
  "workflows": ["<WORKFLOW_DOCS>"],
  "findings": "<FINDINGS_DOC>",
  "ledger": "<LEDGER_DOC>",
  "agent_workflow": ["<AGENT_WORKFLOW_DOCS>"]
}
```

---

## Inventory Existing Prompt References

Goal: surface stale paths, repository‑specific examples, and outdated artifact destinations.

Commands:
```
rg -n "spec|contract|api|architecture|testing|TESTING_GUIDE|TEST_SUITE_INDEX|findings|fix_plan|reports" prompts || true
rg -n "arch\.md|scripts/validation|golden_suite|nanoBragg|legacy|old_path" prompts || true
rg -n "plans/active|reports|artifacts|\boutputs\b" prompts || true
```

Record per file:
- Which documents are referenced and whether they exist now.
- Hard‑coded example commands or modules that look project‑specific.
- Current artifact routing and consistency.

---

## Map and Replace: Align Prompts to Current Sources

- Replace required‑reading sections with the discovered sources:
  - Always include `docs/index.md`.
  - Include `<SPEC_DOCS>`, `<ARCH_DOC>`, `<DEV_GUIDE>`, `<TEST_GUIDE>`, `<TEST_INDEX>`, `<DEBUG_GUIDE>`, `<WORKFLOW_DOCS>`, `<FINDINGS_DOC>`, `<LEDGER_DOC>`, and agent workflow docs if present.
- Unify emphasis by role:
  - Supervisor (planning/review): index, specs, architecture, ledger, findings, agent workflow.
  - Engineer (execution/TDD): specs, testing guide/index, workflows, ledger update steps.
- Derive commands from authoritative documents (testing guide/index). If a mapping does not exist, instruct the engineer to author a minimal targeted test first.

---

## Document Category Strategy: Copy/Adapt vs Derive vs Ab Initio

Use the model and target inventories to decide how to materialize each document category.

- Prompts (`prompts/*.md`): Copy from model → adapt
  - Preserve structure and role emphasis; update references to discovered sources and artifact policy.
  - Replace any model‑specific commands with target project equivalents from `<TEST_GUIDE>/<TEST_INDEX>` and workflow docs.
  - Ensure standard ops prompts are included (copy if present in model; create/adapt if missing):
    - `prompts/postmortem_hardening.md`
    - `prompts/doc_sync_sop.md`
    - `prompts/pyrefly.md`
- Orchestration scripts (`scripts/orchestration/*`): Copy if missing
  - If the target repo does not already include orchestration helpers (e.g., `check_input.py`, `focus_check.py`, `plan_lint.py`), copy them from the model repository.
  - If equivalents exist, diff and adapt minimally (paths, policy references) rather than overwriting.
  - Keep them repo‑agnostic: avoid hard‑coding absolute paths or environment‑specific assumptions.
- CLAUDE.md (root): Ab initio or copy skeleton → adapt
  - If the model’s CLAUDE.md matches your desired policy, copy and adapt references and rules. Otherwise, author fresh using the outline below.
- AGENTS.md (symlink + optional scoped overrides): Configure
  - Create a root `AGENTS.md` symlink pointing to `CLAUDE.md`; author sub‑scoped files only if a subtree truly diverges (language/toolchain) and clearly document how they override the root policy.
- Docs index (`docs/index.md`): Derive
  - Build or update from the target repo’s actual documents so the index is authoritative for this codebase.
- Specs (`specs/*.md`): Project‑specific
  - Copy only if domain‑appropriate; otherwise author or import from the target’s spec source.
- Architecture (`docs/architecture.md`): Derive or copy skeleton → adapt
  - Summarize actual architecture; seed from model only when structures align.
- Testing Guide (`docs/TESTING_GUIDE.md`): Derive or copy skeleton → adapt
  - Reflect target test framework, markers, skip policies, and commands.
- Test Suite Index (`docs/development/TEST_SUITE_INDEX.md`): Derive
  - Generate from target test discovery; document authoritative selectors.
- Debugging/Workflows (`docs/debugging/*`, `docs/workflows/*`): Copy if missing, otherwise derive or copy skeleton → adapt
  - If the target lacks debugging/workflow docs, copy the model’s folder(s) as a starting point and then adapt paths, commands, and selectors to the target.
  - If equivalents exist, prefer deriving from the target repo and selectively merging useful patterns from the model.
  - Keep examples portable and reference authoritative test selectors from `<TEST_GUIDE>/<TEST_INDEX>`.
- Findings (`docs/findings.md`): Ab initio
  - New ledger of decisions, gaps, and policies for the target project.
- Fix Plan (`docs/fix_plan.md`): Ab initio (seeded)
  - Create and seed from `plans/` materials, open issues, and TODO scans; treat as master ledger.

Document each decision (copy/derive/ab initio) in your migration artifact summary.

---

## Dependency Graph of Documentation and Prompts

Understand the order and prerequisites by mapping dependencies. A simple form is bullets; optionally render a diagram for clarity.

Bulleted dependencies (typical)
- Prompts depend on: docs index, specs, architecture/dev guide, testing guide/index, debugging/workflows, findings, fix plan, agent workflow docs, artifact policy.
- CLAUDE.md depends on: docs index, artifact and ledger policy, prompts overview.
- Scoped `AGENTS.md` overrides depend on: coding conventions, test policy, file organization, artifact policy; inherit from the root policy and only document deltas.
- Testing Guide depends on: actual test layout, fixtures, markers, CI constraints.
- Test Suite Index depends on: test files and authoritative selectors.
- Architecture docs depend on: code structure, ADRs (if any), and spec constraints.
- Fix Plan depends on: project goals, plans/, issues/TODOs, artifact policy.
- Findings depends on: migration discoveries, policy decisions, and gaps.

Optional Mermaid outline
```
flowchart TD
  Specs --> Prompts
  Arch --> Prompts
  TestGuide --> Prompts
  TestIndex --> Prompts
  Workflows --> Prompts
  Debug --> Prompts
  Findings --> Prompts
  FixPlan --> Prompts
  AgentDocs(CLAUDE + AGENTS overrides) --> Prompts
  Index --> Specs & Arch & TestGuide & TestIndex & Workflows & Debug & Findings & FixPlan & AgentDocs
```

Practical rule of thumb (order): Index → AgentDocs → Artifact/Ledger policy → Testing docs → Prompts → Validation.

## Artifact Routing Policy

- Standard destination for loop artifacts:
  - `plans/active/<initiative-id>/reports/<YYYY-MM-DDTHHMMSSZ>/`
- Prompt requirements:
  - Every loop that produces logs, traces, test outputs, or summaries writes into the above directory.
  - The directory path is recorded in the ledger entry for that loop (`docs/fix_plan.md`).
- Notes:
  - Use UTC ISO8601 timestamps with `Z` suffix.
  - Derive `<initiative-id>` from the active focus (short, stable slug).
  - Avoid writing artifacts at repo root; keep scratch data in `tmp/` and ignore in commits.

---

## Initialize or Refresh the Fix Plan (docs/fix_plan.md)

If `docs/fix_plan.md` exists (legacy):
- Skim for stale sections; normalize headings and Attempts History.
- Ensure it is treated as the single ledger for all loops.

If missing (new project):
- Create `docs/fix_plan.md` as the master task ledger.
- Seed from material under `plans/`, open issues, and TODOs in docs/prompts:
```
rg -n "^## TODO|\bTODO\b|Fix plan|initiative" docs prompts plans || true
```
- Summarize actionable items into a numbered or slugged checklist.
- Include per‑item: Dependencies, Exit Criteria, Attempts History, and Artifact Path fields.

---

## Initialize Agent Workflow Docs (CLAUDE.md / AGENTS.md)

Goal
- Ensure clear, versioned guidance for agent behavior is present and discoverable.
- Provide a repo‑root policy (`CLAUDE.md`) and scoped overrides (subdirectory `AGENTS.md` files, with the root `AGENTS.md` symlinked to `CLAUDE.md`) that prompts can rely on.

Locations
- Place `CLAUDE.md` at the repository root.
- Symlink the root `AGENTS.md` to `CLAUDE.md` (`ln -sf CLAUDE.md AGENTS.md`) so historical references resolve; optionally add purposeful `AGENTS.md` files in subdirectories that need different local conventions. Deeper files override shallower ones within their directory tree. Direct user/developer instructions always take precedence.

Minimal CLAUDE.md (root) outline
- Purpose: Define agent roles, loop workflow (Supervisor/Engineer), and core rules.
- Sources of truth and reading order: `docs/index.md`, `<SPEC_DOCS>`, `<ARCH_DOC>`, `<DEV_GUIDE>`, `<TEST_GUIDE>`/`<TEST_INDEX>`, `<DEBUG_GUIDE>`, `<FINDINGS_DOC>`, `<LEDGER_DOC>`.
- Core directives:
  - One focus per loop; supervisor writes `input.md`; engineer executes it.
  - Consult findings first for known issues/patterns.
  - Treat specs as normative; propose architecture doc updates when conflicts arise.
  - TDD for features/bug fixes; targeted tests first; run full suite at most once after code changes.
  - Artifact policy: `plans/active/<initiative-id>/reports/<YYYY-MM-DDTHHMMSSZ>/`.
  - Ledger policy: Update `docs/fix_plan.md` each loop with attempt notes and artifact paths.
- Environment/safety:
  - Prefer editable installs for local dev; avoid `sys.path` hacks.
  - Observe repo lint/format/tooling standards if documented.
  - Mention sandbox/approval conventions if the project uses them.
- Pointers: Where to find `prompts/`, smoke tests, and common commands.

Minimal scoped `AGENTS.md` outline (subdirectory overrides)
- Scope and precedence: Applies to this directory tree; deeper files override shallower ones. Direct developer/user instructions supersede AGENTS.md.
- Delta-focused guidance: capture only the deviations from the root `CLAUDE.md` policy.
- Coding conventions: style (formatter/linter), naming, dependency constraints, language/runtime versions.
- Test policy: framework (pytest/unittest), markers (slow/gpu), skip rules, how to run targeted vs full tests.
- File organization: expected locations for binaries, configs, fixtures, and test data.
- Prompt expectations: do not modify identified stable modules without plan; how to reference config/secrets; artifact policy.
- Change hygiene: when to update docs, commit message format, and how to record plan deltas.

Bootstrap procedure
- If missing, create `CLAUDE.md` using the outline above, then symlink `AGENTS.md` to it. Add sub‑scoped `AGENTS.md` only if a subdirectory truly diverges (e.g., different language/toolchain or strict local rules).
- Link both in the doc map:
  - Add an “Agent Workflow” entry to `docs/index.md` pointing to `CLAUDE.md`, noting that `AGENTS.md` is a symlink, and describing how scoped overrides behave.
- Update prompts so supervisor/engineer required‑reading includes these under “Agent Workflow Docs”.
- Ensure the ledger exists and is referenced: if missing, create `docs/fix_plan.md` and seed from `plans/` and TODO scans.

Verification checklist
- Files exist and the symlink resolves:
```
ls CLAUDE.md
readlink AGENTS.md
```
- `docs/index.md` includes an Agent Workflow section referencing CLAUDE.md and explaining the AGENTS.md symlink + scoped override rules.
- Prompts reference CLAUDE.md, mention the AGENTS.md symlink, and document how scoped overrides behave.
- First supervised loop writes artifacts under the standard reports path and records it in `docs/fix_plan.md`.

---

## Modernize Examples and Testing References

- Testing
  - Prefer pytest unless the project mandates otherwise; prompts must not instruct ad‑hoc `sys.path` manipulation.
  - Source commands/selectors from `<TEST_GUIDE>` and `<TEST_INDEX>`. If no authoritative mapping exists, instruct engineers to author a minimal targeted test first.
  - Avoid mixing `unittest.TestCase` with pytest parametrization unless the project requires it (document exceptions).
- CLI/workflow examples
  - Replace repo‑specific binaries with documented CLIs or `python -m <module>` entry points sourced from the index/workflow docs.
- Debugging mode
  - Use `<DEBUG_GUIDE>` for hypothesis‑driven evidence steps; save outputs under the standard reports path.

---

## Quality Gates for Prompt Updates

- Consistency
  - Required‑reading sections across prompts reference the same authoritative docs (with role‑specific emphasis).
  - Artifact routing matches the standard destination and appears in supervisor and engineer flows.
- No stale tokens
```
rg -n "arch\.md|golden_suite|nanoBragg|scripts/validation|reports/debug" prompts || true
rg -n "TODO: replace|<SPEC_DOCS>|<ARCH_DOC>|<DEV_GUIDE>|<TEST_GUIDE>|<TEST_INDEX>|<DEBUG_GUIDE>|<WORKFLOW_DOCS>|<FINDINGS_DOC>|<LEDGER_DOC>|<AGENT_WORKFLOW_DOCS>" prompts || true
```
- Verification of paths
  - For every referenced path in prompts, confirm existence where applicable.
- Minimal functional smoke
  - If the repo defines a standard quick test or smoke in `<TEST_GUIDE>`, include it in the engineer prompt as a reproducible validation step.

---

## Automation Aids (Optional)

Search/replace mapping for prompt path updates
- Create a simple mapping file `migration/path_map.txt` with `from|to` pairs (one per line), e.g.:
```
docs/architecture.md|docs/architecture.md
docs/TESTING_GUIDE.md|docs/TESTING_GUIDE.md
docs/development/TEST_SUITE_INDEX.md|docs/development/TEST_SUITE_INDEX.md
docs/findings.md|docs/findings.md
docs/fix_plan.md|docs/fix_plan.md
```
- Apply across prompts (GNU sed; macOS use `-i ''`):
```
while IFS='|' read -r FROM TO; do
  [ -z "$FROM" ] && continue
  sed -i "s~${FROM}~${TO}~g" prompts/*.md
done < migration/path_map.txt
```

Prompt reference audit
```
rg -n "docs/|specs/|tests/|plans/active|reports|CLAUDE\.md|AGENTS\.md" prompts || true
```

Index link insertion (idempotent example)
```
awk 'BEGIN{added=0} /Agent Workflow/ {added=1} {print} END{if(!added){print "\n- Agent Workflow: see CLAUDE.md (repo root); AGENTS.md symlinks to CLAUDE.md and scoped overrides live in subdirectories."}}' \
  docs/index.md > docs/index.tmp && mv docs/index.tmp docs/index.md
```

Inventory export (very lightweight)
```
{
  echo "# Model inventory"; cd "$MODEL_ROOT" && rg --files | sort | head -n 99999
  echo "# Target inventory"; cd "$TARGET_ROOT" && rg --files | sort | head -n 99999
} > plans/active/<initiative-id>/reports/<timestamp>/inventories.txt
```

## Migration Workflow (Step‑by‑Step)

0) Profile the model and target
   - Produce inventories for both repos; store under `plans/active/<initiative>/reports/<timestamp>/`.
   - Decide category strategies (copy/adapt vs derive vs ab initio) and sketch the dependency graph.

0.5) Prepare copy/adapt workspace (optional)
   - Set `MODEL_ROOT` and `TARGET_ROOT`. For copy/adapt categories, copy skeletons:
```
rsync -av --include='*/' --include='*.md' --exclude='*' "$MODEL_ROOT/prompts/" "$TARGET_ROOT/prompts/"

# Copy orchestration scripts if the target lacks them
if [ ! -d "$TARGET_ROOT/scripts/orchestration" ] || [ -z "$(ls -A "$TARGET_ROOT/scripts/orchestration" 2>/dev/null)" ]; then
  mkdir -p "$TARGET_ROOT/scripts/orchestration"
  rsync -av --include='*.py' --include='*.sh' --exclude='*' \
    "$MODEL_ROOT/scripts/orchestration/" "$TARGET_ROOT/scripts/orchestration/"
fi

# Copy debugging docs if the target lacks them
if [ ! -d "$TARGET_ROOT/docs/debugging" ] || [ -z "$(ls -A "$TARGET_ROOT/docs/debugging" 2>/dev/null)" ]; then
  mkdir -p "$TARGET_ROOT/docs/debugging"
  rsync -av --include='*/' --include='*.md' --include='*.mdx' --exclude='*' \
    "$MODEL_ROOT/docs/debugging/" "$TARGET_ROOT/docs/debugging/"
fi

# Ensure standard ops prompts exist (copy if present in model)
for f in postmortem_hardening.md doc_sync_sop.md pyrefly.md; do
  if [ ! -f "$TARGET_ROOT/prompts/$f" ] && [ -f "$MODEL_ROOT/prompts/$f" ]; then
    rsync -av "$MODEL_ROOT/prompts/$f" "$TARGET_ROOT/prompts/$f"
  fi
done
```

1) Build the source map (target)
   - Read `docs/index.md`; discover category docs; populate `docs/prompt_sources_map.json` (optional).

2) Inventory current prompts (target)
   - List all `prompts/*.md`; extract referenced paths/commands; identify stale tokens and hard‑coded examples.

3) Initialize agent workflow and ledger (target)
   - Create/adapt `CLAUDE.md`, symlink the root `AGENTS.md` to it, and add “Agent Workflow” to `docs/index.md`.
   - Create/refresh `docs/fix_plan.md`; seed from `plans/` and TODO scans.

4) Update prompts (target)
   - Replace required‑reading with discovered sources; normalize artifact routing and ledger references; update example commands per `<TEST_GUIDE>/<TEST_INDEX>` and workflow docs.

5) Validate
   - Scan for stale tokens and placeholder strings; verify referenced files exist; optionally run test discovery (e.g., `pytest --collect-only`).

6) Commit hygiene
   - Commit prompt updates, source map, ledger initialization/refresh, and doc index updates.
   - Use clear messages (e.g., `chore(prompts): align to discovered specs/arch/tests; artifacts -> plans/active/<initiative>/reports`).

7) Document lessons learned
   - Add non‑obvious findings or doc gaps to `docs/findings.md`; propose index updates where structure changed.

---

## Fallbacks and Variations

- If `docs/index.md` is missing, treat the repo `README.md` as the index and scan `docs/` recursively.
- If specs are absent, treat architecture + developer guide as provisional sources and document the gap in findings.
- If `tests/` don’t exist, set expectations in prompts for how to add smoke/unit tests, referencing `<TEST_GUIDE>` once created.
- If `docs/fix_plan.md` is absent, create it, state its role as the master ledger, and seed from `plans/` and TODO scans.

---

## Appendix A: Placeholders

- `<SPEC_DOCS>`: list of spec/contract files.
- `<ARCH_DOC>`: path to architecture overview.
- `<DEV_GUIDE>`: path to developer guide.
- `<TEST_GUIDE>`: path to testing guide.
- `<TEST_INDEX>`: path to test suite map/index.
- `<DEBUG_GUIDE>`: path to debugging SOP.
- `<WORKFLOW_DOCS>`: list of workflow guides (if any).
- `<FINDINGS_DOC>`: path to findings ledger.
- `<LEDGER_DOC>`: path to fix plan ledger (usually `docs/fix_plan.md`).
- `<AGENT_WORKFLOW_DOCS>`: agent rules (`CLAUDE.md`, surfaced via the `AGENTS.md` symlink and any scoped overrides).

---

## Appendix B: Quick Checklists

Required‑reading alignment
- [ ] Index included (`docs/index.md` or equivalent)
- [ ] Specs included (`<SPEC_DOCS>`)
- [ ] Architecture + Dev Guide included (`<ARCH_DOC>`, `<DEV_GUIDE>`)
- [ ] Testing Guide + Test Index included (`<TEST_GUIDE>`, `<TEST_INDEX>`) 
- [ ] Debugging + Workflows included (`<DEBUG_GUIDE>`, `<WORKFLOW_DOCS>`) 
- [ ] Findings + Ledger included (`<FINDINGS_DOC>`, `<LEDGER_DOC>`) 
- [ ] Agent workflow docs included (`<AGENT_WORKFLOW_DOCS>`) 

Artifact routing
- [ ] `plans/active/<initiative-id>/reports/<YYYY-MM-DDTHHMMSSZ>/` used consistently
- [ ] Artifact directory recorded in `docs/fix_plan.md`

No stale tokens
- [ ] Legacy paths/tools removed (per `rg` scans)
- [ ] Placeholder tokens resolved

Verification
- [ ] Referenced files exist (where applicable)
- [ ] Optional: test discovery or smoke steps are documented
- [ ] Orchestration scripts present under `scripts/orchestration/` (copied if missing), or absence documented with rationale
- [ ] Debugging docs present under `docs/debugging/` (copied if missing), or absence documented with rationale
 - [ ] Standard ops prompts present under `prompts/`: `postmortem_hardening.md`, `doc_sync_sop.md`, `pyrefly.md` (copied or authored if missing)
