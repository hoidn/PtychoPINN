# ADR-0007: Remove Timestamped “Report Hubs” from the Agentic Workflow

Status: accepted

Date: 2025-11-12

## Context

The workflow used timestamped “Reports Hubs” under `plans/active/<initiative>/reports/<ISO8601Z>/` to store per‑loop artifacts. This introduced friction (bulky commits, special git rules), duplicated sources of truth (hub summaries vs. fix_plan ledger), and allowed dwell resets on non‑code evidence.

## Decision

- Deprecate report hubs. Do not create new `reports/<timestamp>/` directories.
- For each initiative, keep a single `plans/active/<initiative>/summary.md` and prepend a short Turn Summary per loop.
- Store bulky artifacts outside the repo or under a git‑ignored `.artifacts/` folder; link from the plan/ledger.
- Dwell resets only after implementation evidence (production/test code commits), never on “analysis uploads”.
- Remove “evidence‑only” git exceptions; always perform normal pull/rebase hygiene.

## Consequences

- Prompts and CLAUDE updated to reflect the lean artifacts policy and dwell gates.
- Initiative Workflow Guide updated; hub references now historical.
- Historical hubs remain as‑is for reference; no migration of old evidence is required.

## Alternatives Considered

- Keep hubs but tighten size rules: rejected (still encourages committing bulky evidence, split sources of truth).
- Move hubs out of git: functionally similar to this decision; we prefer a neutral `.artifacts/` convention + external storage links.

## Follow‑ups

- Scrub remaining hub mentions in docs as they surface.
- Optional: add a CI check that rejects new `plans/**/reports/**` additions and large binaries in docs/plans.

