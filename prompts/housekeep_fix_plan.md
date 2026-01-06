# Prompt: Fix Plan Housekeeping (Size‑Aware, Archive‑First)

Use this prompt when the fix plan ledger (`docs/fix_plan.md`) has become unwieldy or stale. The goal is to:
- keep **all active and pending initiatives** actionable and well‑documented,
- aggressively move historical detail into `docs/fix_plan_archive.md`, and
- ensure the live ledger stays within a practical size budget.

Target: `docs/fix_plan.md` should be **< 70,000 characters** (per `wc -c`).

---

## Scope

- Preserve **all initiatives** (IDs and basic metadata) that are still relevant:
  - `Status: pending`, `in_progress`, or `blocked` *must* remain in `docs/fix_plan.md`.
  - `Status: done` or `archived` may be compacted in the live ledger, but never silently dropped.
- Preserve **full historical detail** in `docs/fix_plan_archive.md` and the `plans/active/<id>/reports/` trees; the live ledger may point to those instead of inlining everything.
- Keep the **Execution Roadmap** tiers up to date (Tier 1/2/3 entries).
- Ensure **every initiative** in the live ledger has:
  - a corresponding `plans/active/<ID>/implementation.md` file, and
  - at least one referenced reports directory (even if only via a generic pointer).

---

## Procedure

### 1. Snapshot the current ledger into the archive

1. Ensure the archive file exists:
   - If `docs/fix_plan_archive.md` is missing, create it as an empty file with a header.
2. Append a dated snapshot of the *current* ledger:
   ```bash
   printf '\n\n---\n\n# Snapshot $(date -u +%Y-%m-%d) Fix Plan Ledger\n\n' >> docs/fix_plan_archive.md
   cat docs/fix_plan.md >> docs/fix_plan_archive.md
   ```
3. Do **not** delete or overwrite earlier snapshots in the archive; it is an append‑only history.

### 2. Build a trimmed ledger in a temporary file

Work against a temp file first so it’s easy to measure and review before overwriting the real ledger.

1. **Create `docs/fix_plan_temp.md` from `docs/fix_plan.md` programmatically.**
   - Do *not* edit `docs/fix_plan.md` in place.
   - Use a small script or structured editing to:
     - copy over the header, “Last Updated”, Working Agreements, and Execution Roadmap sections, and
     - reconstruct the `## Active / Pending Initiatives` section with compressed Attempts History as described below.
2. Edit `docs/fix_plan_temp.md`:
   - Keep:
     - Header and **Last Updated** line (update the date and the note about the archive).
     - **Working Agreements**.
     - **Execution Roadmap** tiers and their initiative lists.
     - The `## Active / Pending Initiatives` section and all initiative IDs.
   - For each initiative block (`### [ID] ...`):
     - **Always keep**:
       - `Depends on:`
       - `Status:`
       - `Priority:`
       - `Owner/Date:`
       - `Exit Criteria:` (full list; do not truncate these).
       - `Working Plan: plans/active/<ID>/implementation.md`
     - **Trim Attempts History**:
       - If `Status` is `pending`, `in_progress`, or `blocked`:
         - Keep only a *small* subset of Attempts History inline:
           - First bullet (earliest attempt).
           - Last bullet (most recent attempt).
           - Optionally one “pivotal” attempt if clearly marked (e.g. a major decision).
         - Insert a summary line, e.g.:
           > `... (see docs/fix_plan_archive.md and plans/active/<ID>/reports/ for full history and metrics).`
       - If `Status` is `done` or `archived`:
         - It’s enough to keep a **single** bullet summarizing completion and pointing to the archive:
           > `* See docs/fix_plan_archive.md (snapshot YYYY‑MM‑DD) and plans/active/<ID>/reports/ for full Attempts History.`
         - Do *not* inline multi‑page Attempts History for completed work.

3. Use this rule of thumb:
   - **Active/pending/blocked** initiatives retain full structure (metadata + exit criteria), with **compressed** Attempts History.
   - **Done/archived** initiatives become compact summaries (1–2 bullets) that point at the archive + reports.

### 3. Enforce the size budget

1. Measure the temp ledger size:
   ```bash
   wc -c docs/fix_plan_temp.md
   ```
2. If the size is **≥ 70,000**:
   - Further compress Attempts History for done initiatives (prefer a single summary bullet).
   - Consider trimming intermediate Attempts bullets for active initiatives down to just first/last + one pivotal entry.
   - Re‑run `wc -c` until the file is **< 70,000**.

### 4. Verify references and plans

For every initiative still present in `docs/fix_plan_temp.md`:

- Confirm its plan file exists:
  ```bash
  ls plans/active/<ID>/implementation.md
  ```
  - If missing, create a minimal stub:
    ```bash
    mkdir -p plans/active/<ID>
    printf '# Implementation Plan: <ID>\n\nStatus: pending (stub created to satisfy docs/fix_plan.md reference).\n' > plans/active/<ID>/implementation.md
    ```
- Ensure references to reports are either:
  - explicit paths (e.g., `plans/active/<ID>/reports/2025-11-22T123456Z/`), or
  - generic pointers (“see reports under `plans/active/<ID>/reports/`”).

### 5. Promote the temp ledger and commit

1. Once `docs/fix_plan_temp.md` looks correct and is under the size budget:
   ```bash
   cp docs/fix_plan_temp.md docs/fix_plan.md
   ```
2. Stage and commit:
   ```bash
   git add docs/fix_plan.md docs/fix_plan_archive.md
   git commit -m "docs: housekeep fix plan ledger"
   ```
3. Optionally remove the temp file after commit:
   ```bash
   rm -f docs/fix_plan_temp.md
   ```

---

## Tips & Guardrails

- **Never drop active work:** if an initiative is still `pending`, `in_progress`, or `blocked`, its ID, dependencies, exit criteria, and working plan must remain in the live ledger.
- **Archive is the source of truth for history:** it’s always safe to replace long inline Attempts History with a pointer *after* you’ve appended a snapshot to `docs/fix_plan_archive.md`.
- **Keep tiers coherent:** every initiative listed in a Tier block should have a matching `### [ID]` section under `## Active / Pending Initiatives`.
- **Be explicit about where history lives:** for any trimmed Attempts History, add a line pointing to both `docs/fix_plan_archive.md` and the initiative’s `plans/active/<ID>/reports/` directories. This prevents future ambiguity about where to find full context.
