# Documentation Sync SOP (Template)

Purpose
- Establish a repeatable process to find and fix documentation gaps so that debugging workflows, acceptance tests, scripts, and golden data stay in lock‑step with the codebase.

When To Use
- Before/after major refactors or parity work.
- When a failing AT suggests doc/code drift.
- When links, commands, or examples in docs feel stale.

Scope & Assumptions
- Repo has structured docs (e.g., `docs/architecture.md`, `docs/debugging/`, `docs/development/`, `docs/workflows/`).
- Acceptance tests and a mapping (“Parallel Validation Matrix”) exist (or can be authored) in `docs/TESTING_GUIDE.md` or `docs/development/TEST_SUITE_INDEX.md`.
- Debugging flows produce C↔Py traces and an end‑to‑end parity harness under `scripts/`.

Inputs
- `docs/**` shards, `scripts/**`, `tests/**`, `plans/**`, `specs/**`, `datasets/**`, `examples/**`.
- Acceptance tests and thresholds from specs/testing strategy.

Process (single pass; repeat if needed)
1) Topology Scan (inventory what exists)
- List doc shards and anchors:
  - `rg -n "^#|^## |Last Updated|Verified Against" docs`
- Inventory cross‑references to code/assets:
  - `rg -n "scripts/|tests/|plans/|specs/|datasets/|examples/" docs`
- Locate the AT→tests mapping (“Parallel Validation Matrix”):
  - Open `docs/TESTING_GUIDE.md` (and, if needed, `docs/development/TEST_SUITE_INDEX.md`) and verify the section is present.

2) Cross‑Reference Audit (detect broken/missing links)
- Verify referenced files exist:
  - For each path found in the previous step, `ls <path>` and note missing ones.
- Flag non‑existent or outdated references (e.g., `processes.xml`, missing `compare_traces.py`).
- Detect placeholders:
  - `rg -n "\[DATE\]|\[VERSION\]" docs`

3) Consistency Audit (high‑value themes)
- Acceptance test anchoring:
  - Ensure primary debugging SOP explicitly references the Parallel Validation Matrix and acceptance thresholds.
- Parallel trace discipline:
  - Ensure a Trace Schema exists (names, units, precision, prefixes) in the primary debugging doc.
- Convention/pivot/unit rules:
  - Confirm they’re documented once in specs/architecture and referenced from debugging checklists.
- Dataset & artifact references:
  - Confirm NPZ dataset names, coordinate conventions, and output paths match across docs, `datasets/`, and test fixtures.

4) Prioritize Fixes (in this order)
- Broken/missing references and wrong paths.
- Stale examples contradicting golden data (pixels, units, filenames).
- Missing Trace Schema blocks.
- SOPs not bound to acceptance tests/spec thresholds.
- Placeholder metadata that undermines credibility.

5) Patch Patterns (surgical, minimal)
- Replace dead links with authoritative sources (e.g., point to `docs/TESTING_GUIDE.md` or `docs/COMMANDS_REFERENCE.md`).
- Add/normalize a “Trace Schema” block in `docs/debugging/debugging.md`.
- Align golden examples (canonical pixel, units) across docs and golden logs.
- Update checklists to reference existing helpers or provide a generic `diff` fallback.
- Add a “CI Gates” section to the testing guide (e.g., integration workflow + parity checks).
- Require explicit convention selection in tests/harnesses to avoid hidden switching.

6) Verification (fast, scriptable)
- Re‑scan for broken links/placeholders.
  - `rg -n "\[DATE\]|\[VERSION\]" docs`
  - `rg -n "scripts/|tests/|plans/|specs/|datasets/|examples/" docs | awk '{print $2}' | while read p; do [ -e "$p" ] || echo MISSING "$p"; done`
- Grep for dataset and artifact consistency (e.g., fly64 datasets, nphotons studies):
  - `rg -n "fly64|nphotons|generalization" docs tests`
- Ensure SOPs cite acceptance thresholds and give canonical repro commands.

7) CI Integration (prevent drift)
- Document and implement two light‑weight gates:
  - Visual regression gate: run the integration workflow test (e.g., `python -m unittest tests.test_integration_workflow`) or an equivalent scripted workflow, fail if generated reconstructions diverge beyond documented thresholds, and save key artifacts under `plans/active/<initiative>/reports/`.
  - Metric/trace gate: execute the authoritative parity or evaluation selector documented in `docs/TESTING_GUIDE.md` (or the relevant report script) and fail on the first metrics/trace deviation; capture logs under the same reports directory.

Deliverables
- List of patched files and brief change notes.
- Commands used for audit and verification.
- Remaining TODOs (if any) with owners and due dates.

Success Criteria
- No broken cross‑references from docs → repo files.
- Debugging SOP cites acceptance tests and Parallel Validation Matrix with canonical repro.
- A Trace Schema is present and used.
- Golden references (pixel/units/files) are consistent across docs and artifacts.
- CI gates documented and (optionally) wired.

Reusable Checklists
- Cross‑Ref Health
  - [ ] Every `docs/**` path to `scripts/**`, `tests/**`, `plans/**`, `specs/**`, `datasets/**` exists.
  - [ ] No `[DATE]` or `[VERSION]` placeholders remain.
- Debug SOP Quality
  - [ ] SOP cites ATs and Parallel Validation Matrix.
  - [ ] Trace Schema defined (names, units, precision, prefixes).
  - [ ] Canonical repro commands present.
- Consistency
  - [ ] Dataset names, pixel conventions, and units align across docs, `datasets/`, and test fixtures.
  - [ ] Convention/pivot rules documented once; checklists link back.
- CI Gates
  - [ ] Visual parity gate defined.
  - [ ] Trace parity gate defined.

Maintenance Cadence
- Run this SOP after each major parity change, release, or when parity tests start failing.

Notes
- Keep changes minimal and focused. When in doubt, point to authoritative sources rather than duplicating logic across multiple docs.
