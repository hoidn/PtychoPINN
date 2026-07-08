# Documentation Consistency Pass — Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Briefs/reports under `.superpowers/sdd/ext/`, ledger `.superpowers/sdd/progress.md`. One implementer per task, docs-only unless a task says otherwise, review gate per task or one batch review at the end (controller's choice; all tasks are low-risk).

**Date:** 2026-07-07 · **Branch:** `fno-stable` · **Status:** COMPLETE (2026-07-07)

**Closing status (2026-07-07):** All seven tasks landed. Batch review (2026-07-07): Approved, zero findings, all six task verdicts ✅; whole-tree `pytest` collection verified at 2532 tests / 0 errors. Durable authority-story summary recorded in `docs/findings.md` (DOCS-CONSISTENCY-001).
**Input:** six-partition read-only audit (routing hubs, spec surfaces, testing docs, core guides, findings.md, status/workflow surfaces), 2026-07-07. Audit evidence lives in the session transcript; findings are restated here with file:line anchors so tasks are self-contained.

**Goal:** Resolve the contradictions and routing failures found by the global documentation audit without weakening any contract: one spec-tree authority story, Ralph/Galph deprecation made explicit, `docs/index.md` restored as a trustworthy hub, testing docs made executable-as-written, findings.md re-anchored, and stale inventories labeled rather than rewritten.

**User adjudications (binding):**
1. **Ralph/Galph and `docs/fix_plan.md` are DEPRECATED.** Current work is tracked via `docs/plans/` + `docs/findings.md` (+ session ledgers). Surfaces that present the loop as live get explicit deprecation banners and rerouting, not staleness hedges.
2. **Discoverability via `docs/index.md` is a priority.** The hub must route to every live authority surface; anything archived must be labeled as such where it is routed.

## Global constraints

- Docs-only except where a task explicitly says otherwise (Task 5 touches `scripts/tools/generate_test_index.py` and optionally `tests/study/*.py`; Task 6 optionally touches one script bootstrap). Protected files untouchable as always (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`, `ptycho/config/config.py`, `ptycho_torch/config_params.py`, `ptycho_torch/model.py`).
- **Never weaken a contract.** Supersede with banners + pointers; do not delete normative text. Deletions allowed only for: `docs/index_improved.md` (orphaned twin, zero inbound refs) and the dangling symlink `docs/PROJECT_STATUS.md`.
- **Index-routing invariant:** any task that adds, renames, deprecates, or archives a doc must patch `docs/index.md` routing in the same commit.
- Shared tree with a concurrent session: `git status` before editing; explicit `git add` paths only; `git diff --cached` staged-purity check before every commit (foreign hunk ⇒ unstage, BLOCKED). No "claude" in commit messages, no trailers, never `--no-verify`, no pushes.
- One commit per task, conventional `docs:`/`test:`/`chore:` messages as given.
- Verification for every task: the grep/link checks listed in the task, run after editing, output in the task report.
- Out of scope (recorded, deliberately not done here): NEURIPS campaign-entry bloat in `docs/index.md` (~250 lines; defer restructure until the NEURIPS work is quiescent — inbound anchors at risk); `docs/workflow_queue/` mechanism liveness (separate controller, not Ralph); rewriting the four CLI-inventory docs line-by-line (they get banners, not rewrites); any spec content changes beyond those listed.

---

### Task 1 — Ralph/Galph deprecation sweep (ONE commit)

**Status: ✅ COMPLETE — commit `1d025c51`**

**Files:** `docs/fix_plan.md`, `docs/fix_plan_archive.md`, `docs/steering.md`, `prompts/main.md`, `prompts/main2.md`, `prompts/supervisor.md`, `prompts/supervisor2.md`, `prompts/index.md`, `CLAUDE.md`, `docs/index.md`.

1. Top-of-file banner on `docs/fix_plan.md` and `docs/fix_plan_archive.md`:
   `> **DEPRECATED (2026-07-07).** The Ralph/Galph supervisor–engineer loop and this focus ledger are retired. Current work is tracked in docs/plans/ (one plan per initiative) and docs/findings.md. Retained for history; do not select work from this file.`
2. Same-style banner on all four prompt files (`prompts/main.md`, `main2.md`, `supervisor.md`, `supervisor2.md`): deprecated 2026-07-07, retained for history, not process authorities.
3. `prompts/index.md`: mark the `main/supervisor` and `main2/supervisor2` rows **Deprecated** (the "2" variants additionally "experimental, never promoted").
4. ~~`docs/steering.md` banner~~ — **CORRECTED during execution:** steering.md is NOT Ralph-owned (it feeds the separate NeurIPS steered-backlog-drain workflow, per docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-*.md). Moved to Task 4 step 9b with neutral as-of wording; T1 implementer correctly skipped it.
5. `CLAUDE.md` rerouting (three edits):
   - §2.8 "Testing proof is mandatory" — cite `docs/TESTING_GUIDE.md` only (drop `prompts/main.md`).
   - §3 closing line "cited by `prompts/main.md`, `prompts/supervisor.md`, or the active plan" → "cited by the active plan".
   - §3 Reference Map: no entry may present the Ralph prompts as live process docs (git prompts `git_setup_agent.md`/`git_hygiene.md` are NOT Ralph — keep them).
6. `docs/index.md`: find every routing entry for `fix_plan.md`, Ralph loop, `steering.md` (grep `fix_plan\|Ralph\|steering`) and add "(deprecated)" / "(historical)" to the link text; do not delete rows.
7. Verify: `grep -rn "fix_plan.md" CLAUDE.md docs/index.md prompts/index.md` shows only deprecated-labeled references; each banner present (`head -3` each file).
8. Commit: `docs: mark Ralph/Galph loop and fix_plan ledger deprecated, reroute authorities`

### Task 2 — Spec-tree authority reconciliation (ONE commit)

**Status: ✅ COMPLETE — commit `4647e806`**

**Files:** `CLAUDE.md`, `specs/overlap_metrics.md`, `specs/ptychovit_interop_contract.md`, `docs/specs/spec-ptycho-config-bridge.md`, `docs/specs/spec-ptycho-interfaces.md`, `docs/findings.md`, `docs/index.md`.

Adjudicated authority story: **both spec trees are live, with disjoint scopes** — `specs/` owns external interop contracts (data_contracts, ptychodus_api, compare_models, ptychovit_interop); `docs/specs/` owns the internal `spec-ptycho-*` shard system (with its PTY-AT conformance tests). The one filename collision (`overlap_metrics.md`) is resolved in favor of `docs/specs/overlap_metrics.md` (it matches ACCEPTANCE-001, 2025-11-11).

1. `CLAUDE.md` §1 authority stack: `prefer SPECs (specs/ for external interop contracts; docs/specs/ for internal spec-ptycho-* shards), then project documentation, then prompt files.`
2. `specs/overlap_metrics.md`: top banner `> **SUPERSEDED (2025-11-11, ACCEPTANCE-001).** The current normative overlap-metrics contract is docs/specs/overlap_metrics.md (adds required geometry_acceptance_bound / effective_min_acceptance outputs). Retained for history.` Do not edit its body.
3. `specs/ptychovit_interop_contract.md` §5: add one known-gap note after the §5.2 non-compliance clause: `> Known gap: bridge inference currently uses scan-wise mean aggregation (finding PTYCHOVIT-ASSEMBLY-001, Active); this requirement stands and the implementation is the acknowledged deviation.`
4. `docs/specs/spec-ptycho-config-bridge.md:40`: replace the 6-value architecture enum with the current 14-value `Literal` from `ptycho/config/config.py:100-102` / `ptycho_torch/config_params.py:52-54`, and note the source of truth is those Literals.
5. `docs/specs/spec-ptycho-interfaces.md:23-32`: replace the duplicated NPZ key/dtype table with a pointer: `Canonical NPZ contract: spec-ptycho-core.md §Raw NPZ (RawData.from_file); this section defers to it.` (keep any interfaces-only content).
6. `docs/findings.md` DATA-001 (line ~41): reroute — H5/Ptychodus product contract → `specs/data_contracts.md`; standalone-NPZ contract (`diff3d` key) → `docs/specs/spec-ptycho-core.md`.
7. `docs/index.md` Specifications section: state the two-tree scope split in one sentence; ensure all 5 `specs/` files are routed (the superseded one labeled as such) alongside the `docs/specs/` shards.
8. Verify: `diff <(grep -c . specs/overlap_metrics.md) ...` not needed — instead `grep -n "SUPERSEDED" specs/overlap_metrics.md`, `grep -n "docs/specs" CLAUDE.md`, `grep -rn "spec-ptycho-core" docs/findings.md`.
9. Commit: `docs: reconcile specs/ and docs/specs/ authority, supersede stale overlap_metrics copy`

### Task 3 — findings.md accuracy sweep + guide echoes (ONE commit)

**Status: ✅ COMPLETE — commit `58285252`**

**Files:** `docs/findings.md`, `docs/workflows/pytorch.md`, `docs/CONFIGURATION.md`.

1. GRIDLINES-OBJECT-BIG-001 (row ~line 8): scope the rule to gridsize=1 — the torch runner now sets `object_big = cfg.gridsize > 1` (`scripts/studies/grid_lines_torch_runner.py:1255`); cross-ref TORCH-GS2-CENTRAL-MASK-MERGE-001 for the gs2 branch; fix the stale `:278-292` citation. TF grid-lines workflow (`ptycho/workflows/grid_lines_workflow.py:504`) still hardcodes `object_big=False` — say so explicitly.
2. Guide echoes of the same stale claim: `docs/workflows/pytorch.md:81` and `docs/CONFIGURATION.md:59` "sets object_big=False" → gridsize-conditional wording matching (1).
3. FORWARD-SIG-001 (row ~line 63): status Active → **Superseded** by the generator-registry unification (all architectures now go through `forward_predict(x, positions, probe, input_scale_factor)`, `scripts/studies/grid_lines_torch_runner.py:1497-1560`); note its own cited test now asserts the opposite of the old claim; fix the stale citation.
4. Re-anchor drifted line citations (content unchanged, verified still true): TORCH-PADDED-SIZE-001 → `ptycho_torch/helper.py:493`; TORCH-REASSEMBLY-NORM-001 and TORCH-GS2-CENTRAL-MASK-MERGE-001 → `ptycho_torch/helper.py:38` (+ central-mask logic ~140-207); DEVICE-HANDOFF-001 → `ptycho_torch/workflows/components.py:1364`; OUTPUT-COMPLEX-001 → `scripts/studies/grid_lines_torch_runner.py:148`; GRIDLINES-PROBE-BIG-001 → `:1256`. Re-verify each anchor before writing it (the tree moves).
5. Add the 6 missing index-table rows (entries exist, rows don't): DATA-002, PINN-CHUNKED-001, TF-NON-XLA-SHAPE-001, TF-XLA-BATCH-BROADCAST-001, REPORTING-ARTIFACT-BOUNDARY-001, LINES256-PROPOSAL-CANONICALIZATION-001 — source status from each entry's own Status line.
6. `docs/workflows/pytorch.md` loss-mode/CLI section (~line 109): document `--count-scale-mode {auto,off}` — default `off`; `auto` is opt-in, attaches physics scale for units-correct NLL, and is NOT outcome-preserving (POISSON-SCALE-001); lifted and unlifted runs are not comparable.
7. Verify: every edited finding ID greps to exactly one table row + one entry; cited anchors spot-checked (`sed -n` the cited lines, confirm symbol present); `grep -n "count-scale-mode" docs/workflows/pytorch.md`.
8. Commit: `docs: re-anchor findings ledger, supersede FORWARD-SIG-001, scope object_big finding to gridsize=1`

### Task 4 — index.md discoverability overhaul (ONE commit) — USER-PRIORITY

**Status: ✅ COMPLETE — commit `bfcd72ba`**

**Files:** `docs/index.md`, `docs/index_improved.md` (delete), `docs/PROJECT_STATUS.md` (delete dangling symlink), `docs/mindmap.md`, `prompts/index.md`.

1. Delete `docs/index_improved.md` (orphaned full twin of the hub: zero inbound references, last commit 2026-02-16; `git rm`).
2. Delete the dangling symlink `docs/PROJECT_STATUS.md` (target `../PROJECT_STATUS.md` was gitignored/deleted in e5387d46; `git rm`). Remove/retarget the two hub entries (`docs/index.md:594,1001`) that route to it — point status seekers at `docs/plans/` + `docs/findings.md` (consistent with Task 1).
3. Fix broken/moved links in `docs/index.md`: `backlog/2026-02-11-tf-loader-keras3-fallback.md` → `backlog/paused/2026-02-11-tf-loader-keras3-fallback.md`; sweep for other dead relative links with a link-check loop (extract `](...)` targets, test -e each, fix or label).
4. Route the unrouted `docs/bugs/` files — at minimum the two **Status: Open** bugs `POISSON_LOSS_TF_TORCH_MISMATCH.md` and `FNO_INPUT_TRANSFORM_IGNORED.md`, plus `2026-02-05-object-big-coords-relative-regression.md` and `POISSON_MAE_UNIT_MISMATCH_PYTORCH.md`, with status labels.
5. Add an **Archived / historical inventories** subsection routing the four CLI docs (`CLI_FLAGS_MAPPING.md`, `cli_flags_quick_reference.md`, `pytorch_cli_inventory.md`, `cli_config_dataflow.md`) with their snapshot caveat (banners added in Task 6), plus `memory.md` retitled truthfully ("Sampling Control Phase 5 context — historical", not "Memory Optimization Guide").
6. Critical Gotchas table, MODULE-SINGLETON-001 row (~line 11): rewrite to reflect FULLY RESOLVED (2026-01-07) status per findings.md — keep only the residual guidance that is still live, link the finding.
7. Disambiguate the two "Workflow Guide" labels: retitle the `docs/WORKFLOW_GUIDE.md` entry to "CLI Workflow Guide (train/infer/compare)"; `docs/INITIATIVE_WORKFLOW_GUIDE.md` keeps the initiative-process label.
8. Cross-link the hubs: one line in `docs/index.md` → `prompts/index.md` (prompt catalog), one line back. Fix `prompts/index.md:20-21,27` repo-root-relative yaml paths (`workflows/agent_orchestration/...` → `../workflows/agent_orchestration/...`).
9. `docs/mindmap.md`: banner `> **Unpopulated template — not a knowledge index for this project.**` (its self-description claims primary-authority status with zero content).
9b. `docs/steering.md` (moved from Task 1 with corrected attribution): banner `> **As-of 2026-04-28.** Steering input for the NeurIPS steered-backlog-drain workflow (docs/plans/2026-04-22-neurips-steered-backlog-drain-workflow-design.md); verify priorities are current before acting.` Label its `docs/index.md` entry "(as-of 2026-04-28)". Add `docs/steering.md` to this task's file list and commit pathspec.
10. Verify: link-check loop over `docs/index.md` reports zero dead relative links; `grep -rn "index_improved" docs/ prompts/ CLAUDE.md` → nothing; `test -e docs/PROJECT_STATUS.md` → absent.
11. Commit: `docs: index.md discoverability pass — fix links, route open bugs and archives, remove orphaned twin`

### Task 5 — Testing surfaces made executable-as-written (ONE commit; touches generator script + optionally tests/study)

**Status: ✅ COMPLETE — commit `43e88da1`**

**Files:** `scripts/tools/generate_test_index.py`, `docs/development/TEST_SUITE_INDEX.md` (regenerated), `docs/TESTING_GUIDE.md`, optionally `tests/study/test_dose_overlap_*.py` (6 files).

1. Fix `_module_command` (`scripts/tools/generate_test_index.py:84-87`): emit `python -m pytest tests/...` for modules with no `unittest.TestCase` subclass (the AST scan already exists in the file); keep unittest form only for genuine TestCase modules. Update the hardcoded "How to Run Tests" header block (lines ~157-159) to present pytest as the project standard (mirroring `docs/TESTING_GUIDE.md:7-11`), with unittest as the legacy exception.
2. Regenerate: `python scripts/tools/generate_test_index.py` → new `docs/development/TEST_SUITE_INDEX.md` (closes the 75-file gap including the 07-02..07-07 VarPro/reassembly test files; MANUAL_OVERRIDES preserved by the generator). Diff-review the regenerated file: no rows lost, commands now pytest-form for pytest-style modules.
3. `docs/TESTING_GUIDE.md` fixes:
   - `:45,:456` `test_model_manager` → `test_model_manager_persistence` (the file that exists).
   - `:250,253` drop the hardcoded "(10 tests)"/"(2 tests)" counts (they drift); same for the `:620` template's leftover "**Total Tests:** 172" → bracketed placeholder.
   - `:610` `$PYTHON_BIN` → `python` (PYTHON-ENV-001).
   - Replace the boilerplate CI section (~:734-736) with the real gate: `main` is gated by the `pytest-cpu` job (`.github/workflows/tests.yml` on main) running `bash ci/run_ci_tests.sh` = `pytest tests/torch -m "not slow"` CPU-only with `ci/known_failures.txt` deselects and `ci/collect_ignores.txt` ignores (policy: `docs/ci.md` on main); give the local reproduction command `CUDA_VISIBLE_DEVICES="" bash ci/run_ci_tests.sh` and note `ci/` + the workflow exist on main but not on `fno-stable`.
   - Whole-tree caveat next to `pytest tests/ -q` (`:10`): `tests/study/test_dose_overlap_*.py` import the top-level `studies.` package which is gitignored (`.gitignore:167`) — collection breaks on any fresh clone.
4. **Decision point (default: do it, flag loudly in the report):** add `pytest.importorskip("studies.fly64_dose_overlap", reason="branch-local studies/ package not present (gitignored; see TESTING_GUIDE CI section)")` guards at the top of the 6 `tests/study/test_dose_overlap_*.py` files so whole-tree collection completes everywhere; where `studies/` exists locally the tests still run. This is dependency gating, not test disabling. If the implementer finds the tests already guarded or the import structure unsuitable, report BLOCKED on this step only and land the rest.
5. Verify: `CUDA_VISIBLE_DEVICES="" python -m pytest tests/ --collect-only -q 2>&1 | tail -3` → collection completes with 0 errors (if step 4 done); regenerated index contains the 07-07 test files; `grep -c "unittest" docs/development/TEST_SUITE_INDEX.md` plausibly small; `grep -n "PYTHON_BIN" docs/TESTING_GUIDE.md` → nothing.
6. Commit: `test: fix test-index generator pytest commands, regenerate index, repair testing guide claims`

### Task 6 — Core-guide command fixes + inventory banners (ONE commit)

**Status: ✅ COMPLETE — commit `f9f618fb`**

**Files:** `docs/COMMANDS_REFERENCE.md`, `docs/workflows/pytorch.md`, `docs/CONFIGURATION.md`, `docs/CLI_FLAGS_MAPPING.md`, `docs/cli_flags_quick_reference.md`, `docs/pytorch_cli_inventory.md`, `docs/cli_config_dataflow.md`, optionally `scripts/studies/render_hybrid_resnet_schematics.py`.

1. `docs/COMMANDS_REFERENCE.md:35`: `shuffle_dataset_tool.py` command → `--input-file converted_data.npz --output-file shuffled_data.npz --seed 42` (argparse has no positionals — verified error).
2. `docs/COMMANDS_REFERENCE.md:365`: `--output plots/` → `--output plots/generalization_plot.png` (param is a filename; trailing slash silently writes a file named `plots`).
3. `docs/workflows/pytorch.md:569`: split the `--scheduler` row per-CLI — `ptycho_torch/train.py` accepts `{Default,Exponential,MultiStage,Adaptive}` (`train.py:743`); `grid_lines_torch_runner.py` separately accepts `WarmupCosine`/`ReduceLROnPlateau`.
4. `docs/workflows/pytorch.md:565`: `--accelerator` default is `'auto'` (resolves cuda-if-available), not `'cuda'` (`ptycho/config/config.py:252`).
5. `docs/CONFIGURATION.md:66-78`: label the PyTorch execution-params table "illustrative subset — full field list: `PyTorchExecutionConfig` in `ptycho/config/config.py`".
6. Historical-snapshot banner on all four CLI-inventory docs: `> **Historical snapshot (2025-10-19, pre-ADR-003) — superseded by docs/workflows/pytorch.md §12 and each CLI's --help. Not maintained.`` (pytorch_cli_inventory.md even cites a different checkout path — quote that in its banner as provenance.)
7. **Optional code fix (do unless it fails trivially):** `scripts/studies/render_hybrid_resnet_schematics.py` — add the same `REPO_ROOT` sys.path bootstrap used by `grid_lines_torch_runner.py:50-51` so the documented command works from a clean shell; smoke-test `python scripts/studies/render_hybrid_resnet_schematics.py --help`.
8. Verify: run each corrected command's `--help`/dry form where cheap; `grep -n "Historical snapshot" docs/CLI_FLAGS_MAPPING.md docs/cli_flags_quick_reference.md docs/pytorch_cli_inventory.md docs/cli_config_dataflow.md` → 4 hits.
9. Commit: `docs: fix broken documented commands, correct pytorch workflow table, banner stale CLI inventories`

### Task 7 — Close-out (ONE commit)

**Status: ✅ COMPLETE — this commit**

**Files:** `docs/findings.md` (optional single new entry), this plan.

1. Optional but recommended: one findings.md entry `DOCS-CONSISTENCY-001` (type: process/finding) recording the pass: date, the six audit partitions, the two user adjudications (Ralph deprecation; index-first discoverability), the spec-tree scope split decision, and the commit list — so future sessions inherit the authority story without re-auditing.
2. Mark this plan's tasks complete with commit hashes; Status → COMPLETE.
3. Commit: `docs: close out 2026-07-07 documentation consistency pass`

---

## Deferred / follow-ups (not in this plan)
- `docs/index.md` NEURIPS campaign-entry consolidation (~250 lines → `paper_evidence_index.md`): after NEURIPS work quiesces; inbound anchors at risk.
- `docs/workflow_queue/` liveness adjudication (separate lines256 session-controller mechanism; needs its own owner decision).
- Porting `.github/workflows/tests.yml` + `ci/` to `fno-stable` (or documenting that the gate applies only on main): flag to user at merge time.
- Full refresh of `docs/specs/spec-ptycho-config-bridge.md` beyond the enum row, and any deeper spec-shard modernization.
