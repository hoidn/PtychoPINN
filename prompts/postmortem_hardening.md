# Postmortem + Prompt Hardening Meta‑Prompt

Use this meta‑prompt to analyze an agent incident, identify the first process divergence, and apply tight, minimal edits to prompts so the failure class cannot recur. Keep edits project‑agnostic and as short as possible.

## Inputs
- Incident context: branch, timestamp, commands run, visible symptoms
- Logs: `tmp/supervisorlog-latest.txt`, `tmp/claudelog-latest.txt`, `galph_memory.md:1`
- Process docs: `docs/index.md:1`, `docs/fix_plan.md:1` (+ any referenced `plans/active/...`), `prompts/supervisor.md:1`, `prompts/main.md:1`, `prompts/debug.md:1`
- Authoritative commands doc (project SOT): `docs/TESTING_GUIDE.md:1` (or repo‑specific path)
- Orchestration state (if any): `sync/state.json:1`

## Goal
Produce a tiny, surgical change set to prompts that:
- Prevents the observed failure mode
- Is project‑agnostic (no hard‑coded tools/paths)
- Uses minimal words (1–2 sentences per section)
- Respects Protected Assets and existing conventions

## Method
1) Reconstruct Behavior
- Summarize exact agent actions (commands + outcomes)
- Map each action to process expectations (quote prompt/doc lines)
- Identify the First Process Divergence (FPD) with `file:line` pointer

2) Classify Root Cause
- Choose one: Prompt defect (rule missing/ambiguous), Doc drift (stale/incorrect doc), Orchestration gap (turn/state/branch), Operator lapse (ignored instruction)
- State the minimum guardrail that would have prevented FPD

3) Draft Minimal, Project‑Agnostic Edits
- Supervisor (Do‑Now composer)
  - Phase Gate (Evidence): “If the active plan phase is Evidence, Do Now SHALL be evidence‑only: run the project’s authoritative reference reproduction commands and capture artifacts; at most run discovery/listing (no execution). Do not include runnable actions.”
  - Validation Preflight: “Before listing any file/identifier/command in Do Now, validate it resolves via the project’s documented discovery/listing; if validation fails, omit such actions, note ‘(none — evidence‑only this loop)’, and add a TODO in the plan ledger.”
  - Authoritative Doc Pointer: “Resolve reference commands and discovery via `docs/TESTING_GUIDE.md` (or project‑specific path).”
- Main (engineer loop)
  - Fallback: “If a Do‑Now action fails validation or resolution, execute the current plan phase’s evidence‑only workflow and append a TODO in `docs/fix_plan.md` describing the missing mapping.”
- Debug (equivalence/triage)
  - First‑divergence rule: “Use first‑divergence against the authoritative trace/ground truth before modifying tests or thresholds.”

4) Verify Against Repo Reality
- Confirm the authoritative doc exists and declares:
  - Ground‑truth reproduction commands (evidence)
  - Discovery/listing command (no execution)
- Ensure edits are jargon‑free (ground truth, discovery/listing, plan ledger)
- Ensure no protected artifacts are renamed/deleted (check `docs/index.md:1`)

5) Apply Patch (or Stage Proposal)
- Keep edits to 1–2 sentences per affected prompt section
- Place lines where Do‑Now is composed (supervisor), where loop actions branch (main), and at debug entry (debug)
- Include `file:line` anchors in your postmortem output for human review
- Add a `docs/fix_plan.md` Attempts entry summarizing: FPD, minimal guardrail, files changed, artifacts, and next monitoring step

6) Sanity Gates
- Branch discipline: assert expected branch before committing; push via explicit refspec `HEAD:<branch>`
- Evidence‑only means no runnable changes/tests; discovery/listing only
- On any validation failure: default to evidence‑only and record the gap in the plan ledger

## Output Checklist (for the postmortem)
- Incident summary (what/where/when; exact commands)
- FPD with `file:line` and why prompts allowed it
- Minimal guardrail (quote the one‑liner) + rationale
- Proposed patch list (files + approximate insertion points)
- Plan ledger delta (Attempts entry + artifact paths)
- Next monitoring step (what to watch in the next loop)

## Notes
- Variables you may reference in edits: `AUTHORITATIVE_CMDS_DOC` (default `docs/TESTING_GUIDE.md`), `PLAN_LEDGER` (default `docs/fix_plan.md`), and the project’s discovery/listing command as documented.
- Keep everything project‑agnostic: prefer “ground truth” and “discovery/listing” over tool names.
