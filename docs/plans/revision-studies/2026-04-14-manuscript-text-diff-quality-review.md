# Manuscript Text Diff Quality Review Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create a worktree for this project. Do not use subagents unless the user explicitly authorizes them.

**Goal:** Systematically audit every text difference between `/home/ollie/Documents/ptychopinnpaper2/oldversion.tex` and `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`, then keep, tighten, or revert each changed prose increment so the revised manuscript answers reviewer comments succinctly without losing good original text or importing repo-internal terminology.

**Architecture:** Treat the manuscript review as a tracked adjudication pass, not an open-ended rewrite. First create a diff ledger that enumerates every changed paragraph, caption, table note, and discussion sentence. Then decide each text increment with a reviewer-response mapping and a strict quality gate. Finally apply minimal manuscript edits, compile, inspect the PDF text, and update the reviewer checklist.

**Tech Stack:** LaTeX, `git diff --no-index`, `rg`, `python` from PATH for optional text extraction, `pdflatex`, `pdftotext`.

---

## Compliance Matrix

- [ ] **Project instruction:** `/home/ollie/Documents/PtychoPINN/AGENTS.md` says no worktrees and paper revisions must consult/update `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`.
- [ ] **Project instruction:** Treat `revision_plan.md` as stale; use `reviewer_revision_checklist.md`, reviewer comments, and current manuscript files as the source of truth.
- [ ] **Documentation read:** Re-read `/home/ollie/Documents/PtychoPINN/docs/index.md` and `/home/ollie/Documents/PtychoPINN/docs/findings.md` before executing edits.
- [ ] **Artifact hygiene:** Keep bulky generated diff artifacts out of the tracked repo or under git-ignored artifact directories. Keep the human audit ledger small and readable.

## Source Files

**Paper repo:** `/home/ollie/Documents/ptychopinnpaper2`

**Primary inputs:**
- `/home/ollie/Documents/ptychopinnpaper2/oldversion.tex`
- `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_comments/1.txt`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_comments/2.txt`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_comments/3.txt`
- `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`

**Likely manuscript-only edits:**
- Modify: `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`
- Modify: `/home/ollie/Documents/ptychopinnpaper2/changelog.txt`

**Audit artifact to create:**
- Create: `/home/ollie/Documents/ptychopinnpaper2/docs/backlog/2026-04-14-manuscript-text-diff-quality-audit.md`

Do not modify result JSON, generated tables, or figures during this pass unless the text review exposes a direct contradiction that cannot be fixed in manuscript prose alone.

## Reviewer-Response Map

Use these reviewer IDs in the audit ledger:

- `R1-self-contained`: manuscript should stand alone relative to Hoidn et al. 2023.
- `R1-flow`: post-Introduction text should not read like bullet-point notes.
- `R1-fig1-cdi`: Fig. 1 should clarify whether it shows single-shot PtychoPINN.
- `R1-extended-probe`: explain why extended-probe handling belongs in Neural Network Architecture.
- `R1-epie-reference`: define the iterative reference method in the OOD figure.
- `R1-fixed-probe-question`: clarify whether probes are supplied or solved.
- `R2-fig-order`: Fig. 1 should appear before Fig. 2 discussion.
- `R2-fig1-ground-truth`: Fig. 1 should include ground truth.
- `R2-direct-metric`: address SSIM concern with direct-error/MSE-derived metric.
- `R3-non-ml-baseline`: include or scope a standard non-ML single-shot CDI baseline.
- `R3-probe-sensitivity`: add fixed-probe mischaracterization stress-test or limitation wording.
- `R3-ood-metrics`: add quantitative Fig. 5 OOD metrics.
- `R3-ood-phase-discussion`: discuss low-frequency phase artifacts without overclaiming.
- `R3-epsilon`: address Eq. 1 epsilon sensitivity.
- `R3-dose-source`: clarify dose-ablation data source.
- `R3-k4`: explain `K=4`.

## Quality Gate

For each changed text increment, apply all criteria below.

Keep or tighten only if the increment:
- directly answers one of the reviewer IDs above, or is a genuine formatting/flow improvement;
- preserves original scientific meaning unless a reviewer response required changing it;
- is maximally succinct for the manuscript body;
- uses paper-facing terminology, not repo/process terminology;
- avoids internal artifact language such as `accepted`, `generated-split`, `same-bundle`, `comparator bundle`, `run artifact`, `artifact-only`, `params.cfg`, `checksum`, `direct-stitch`, `oracle shift`, `Table-2-comparable`, or run-root paths;
- keeps methodological caveats only when needed for a reader to interpret a figure/table correctly;
- does not move detailed provenance that belongs in `data/README.md`, manifests, or response letters into the manuscript;
- keeps LaTeX formatting improvements that improve readability, such as bullet-to-paragraph rewrites, unless they introduce awkward prose.

If a text increment fails the gate, prefer this decision order:
1. `TIGHTEN`: retain the reviewer-relevant fact in one concise sentence.
2. `REVERT_TEXT`: restore old text if the new text is unnecessary or inferior.
3. `ESCALATE`: stop and ask the user if a scientific claim would change.

## Initial Diff Inventory

The first inspection found this high-level diff shape:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git diff --no-index --stat -- oldversion.tex ptychopinn_2025.tex || true
# oldversion.tex => ptychopinn_2025.tex | 96 ++++++++++++++++-------------------
# 1 file changed, 43 insertions(+), 53 deletions(-)
```

Known changed increments to audit:

- `D01`: Introduction contribution list changed from `enumerate` to one paragraph.
- `D02`: Fig. 1 changed from four subfigures to one ground-truth composite and new caption.
- `D03`: notation list changed from bullet list to one paragraph.
- `D04`: Data Preprocessing adds fixed supplied-probe/no-joint-optimization sentence.
- `D05`: Overlap-Free Reconstruction paragraph adds Fig. 1 ground-truth wording and PyNX HIO/ER result sentence.
- `D06`: Table 2 adds same-split PyNX HIO/ER benchmark row.
- `D07`: Table 2 explanatory paragraph adds PyNX/version/support/evaluation caveats.
- `D08`: probe-mischaracterization paragraph and figure added.
- `D09`: OOD Generalization paragraph adds Table 3 metrics and evaluation details.
- `D10`: Table 3 added for Fig. 5 OOD metrics.
- `D11`: Fig. 5 commented-out scale-bar lines removed.
- `D12`: Discussion overlap-free paragraph adds PyNX HIO/ER scope sentence.
- `D13`: Discussion adds OOD phase-artifact paragraph.
- `D14`: Open problems paragraph adds probe-mischaracterization limitation sentence.

The executor must confirm this list against the fresh diff before editing; add any missed increments to the ledger.

## Phase A - Build The Diff Ledger

- [ ] **A1: Reconfirm working tree context**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git status --short
test -f oldversion.tex
test -f ptychopinn_2025.tex
```

Expected: both `.tex` files exist. The paper repo may have unrelated dirty files from earlier revision work; do not revert them.

- [ ] **A2: Capture a mechanical manuscript diff**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
mkdir -p artifacts/work/text-quality-audit
git diff --no-index --unified=12 -- oldversion.tex ptychopinn_2025.tex \
  > artifacts/work/text-quality-audit/oldversion-vs-ptychopinn_2025.diff || true
git diff --no-index --word-diff=porcelain -- oldversion.tex ptychopinn_2025.tex \
  > artifacts/work/text-quality-audit/oldversion-vs-ptychopinn_2025.worddiff || true
```

Expected: both files are written. `git diff --no-index` may exit `1` when differences exist; that is expected.

- [ ] **A3: Create the audit ledger skeleton**

Create `/home/ollie/Documents/ptychopinnpaper2/docs/backlog/2026-04-14-manuscript-text-diff-quality-audit.md` with:

```markdown
# Manuscript Text Diff Quality Audit

Reference: `oldversion.tex`
Candidate: `ptychopinn_2025.tex`
Plan: `/home/ollie/Documents/PtychoPINN/docs/plans/revision-studies/2026-04-14-manuscript-text-diff-quality-review.md`

## Decision Codes
- KEEP
- TIGHTEN
- REVERT_TEXT
- KEEP_FORMATTING
- ESCALATE

## Diff Ledger

| ID | Location | Reviewer link | Old text role | New text role | Decision | Action | Rationale |
| --- | --- | --- | --- | --- | --- | --- | --- |
```

- [ ] **A4: Populate one ledger row per diff increment**

For each `Dxx`, record:
- exact location in `ptychopinn_2025.tex` using line numbers from `nl -ba ptychopinn_2025.tex`;
- reviewer ID(s) from the reviewer-response map;
- short old-text role, not a full quotation;
- short new-text role;
- preliminary decision and rationale.

Do not edit the manuscript in Phase A.

## Phase B - Adjudicate Each Text Increment

- [ ] **B1: Review formatting-only diffs**

Evaluate `D01`, `D03`, and `D11`.

Expected likely decisions:
- `D01`: `KEEP_FORMATTING` if the paragraph reads cleanly and avoids bullet-point presentation.
- `D03`: `KEEP_FORMATTING` if the notation paragraph remains readable; otherwise split into two short sentences.
- `D11`: `KEEP` if only stale commented scale-bar placeholders were removed.

- [ ] **B2: Review Fig. 1 text**

Evaluate `D02` and the Fig. 1 reference in `D05`.

Quality target:
- Keep the ground-truth/composite change.
- Tighten any caption clause that over-explains internal layout.
- Use `overlap-free single-frame PtychoPINN CDI ($C_g=1$)` and `overlapped ptychography ($C_g=4$)`.
- Avoid explaining implementation history, source recovery, or figure-generation process in the manuscript.

- [ ] **B3: Review fixed-probe statements**

Evaluate `D04`, `D08`, and `D14`.

Quality target:
- Keep a concise Methods sentence that the probe is supplied/pre-estimated and held fixed.
- Keep a concise Results statement that the stress test shows sensitivity, not robustness.
- Keep a concise Discussion limitation sentence.
- Remove detailed condition lists from the main text if they make the manuscript read like a lab notebook; let the figure/table carry numbers unless the comparison is essential.

- [ ] **B4: Review non-ML baseline wording**

Evaluate `D05`, `D06`, `D07`, and `D12`.

Quality target:
- Keep the benchmark if it is reviewer-required and the row remains in Table 2.
- Tighten prose to avoid internal process terms.
- Replace internal phrases:
  - `accepted same-generated-split` -> `same test split` or remove if not necessary;
  - `direct-stitch` -> remove or describe only if scientifically necessary;
  - `oracle shift` -> avoid unless the paper defines it;
  - `comparator bundle` -> remove;
  - `PyNX 2024.1` -> keep only in a table note or data availability, not central prose, unless citation/provenance requires it.

- [ ] **B5: Review OOD metrics wording**

Evaluate `D09`, `D10`, and `D13`.

Quality target:
- Keep the metrics table if it answers `R3-ood-metrics`.
- Tighten the paragraph before Table 3 so it says what was evaluated and the outcome, not the full implementation recipe.
- Table caption can state essential scoring choices: held-out test half, unscaled amplitude MSE/PSNR, MSE-derived PSNR, phase-plane alignment.
- Remove repo-internal terms such as `params.cfg["offset"]`, `archived with run artifacts`, and detailed sensitivity diagnostics unless needed to interpret values.
- Keep phase-artifact discussion cautious and physical; avoid unsupported mechanism claims.

- [ ] **B6: Check unresolved reviewer items**

Before editing, scan `reviewer_revision_checklist.md` for unchecked text tasks:
- `R1-self-contained`
- `R1-flow`
- `R1-extended-probe`
- `R1-epie-reference`
- `R1/R2/R3 fixed-probe assumption`
- `R2-direct-metric`
- `R3-epsilon`
- `R3-dose-source`
- `R3-k4`
- changelog expansion

If a diff increment is currently the only response to an unchecked item, do not revert it without replacing it with a tighter response.

## Phase C - Apply Minimal Manuscript Edits

- [ ] **C1: Patch only the reviewed manuscript increments**

Use `apply_patch` for manual edits. Edit `/home/ollie/Documents/ptychopinnpaper2/ptychopinn_2025.tex` in small batches by section:
- Introduction/Fig. 1/Methods
- Overlap-Free Reconstruction/Table 2/probe stress test
- OOD metrics/Table 3
- Discussion

After each batch, update the ledger row `Decision`, `Action`, and `Rationale`.

- [ ] **C2: Preserve valid original text**

For each `REVERT_TEXT` decision, restore the old paragraph from `oldversion.tex` and add only the minimal reviewer-response sentence if required.

Check with:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git diff --no-index --unified=8 -- oldversion.tex ptychopinn_2025.tex | sed -n '1,260p' || true
```

Expected: every remaining prose diff is explainable by the audit ledger.

- [ ] **C3: Remove internal terminology from manuscript prose**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
rg -n "accepted|generated-split|same-bundle|comparator bundle|run artifact|artifact-only|params\\.cfg|checksum|direct-stitch|oracle shift|Table-2-comparable|run_|output-root" ptychopinn_2025.tex || true
```

Expected: no hits in manuscript prose. If a hit is in a code comment or unavoidable table label, justify it in the audit ledger; otherwise rewrite it.

## Phase D - Update Tracking Documents

- [ ] **D1: Update reviewer checklist**

Modify `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`:
- add a progress note linking the audit ledger and this plan;
- mark no substantive reviewer item complete solely because of this cleanup unless the manuscript text now clearly satisfies it;
- if the cleanup changes a previous resolution, update that resolution note.

- [ ] **D2: Update changelog**

Modify `/home/ollie/Documents/ptychopinnpaper2/changelog.txt` only after manuscript edits are final. The changelog entry should describe reviewer-facing changes, not internal cleanup mechanics.

## Phase E - Verification

- [ ] **E1: Compile LaTeX**

Run two passes unless the first pass clearly says no rerun is required:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
pdflatex -interaction=nonstopmode -halt-on-error ptychopinn_2025.tex
pdflatex -interaction=nonstopmode -halt-on-error ptychopinn_2025.tex
```

Expected: exit code `0`. Record overfull boxes separately; do not call the paper clean if new severe layout problems appear near edited text.

- [ ] **E2: Inspect generated text**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
pdftotext -layout ptychopinn_2025.pdf artifacts/work/text-quality-audit/ptychopinn_2025.txt
rg -n "accepted|generated-split|same-bundle|comparator bundle|run artifact|artifact-only|params\\.cfg|checksum|direct-stitch|oracle shift|Table-2-comparable|TODO|X nm|Tike" artifacts/work/text-quality-audit/ptychopinn_2025.txt || true
```

Expected: no internal process terms in the compiled paper. Any remaining `Tike`/`X nm`/`TODO` hit must be fixed or explicitly justified.

- [ ] **E3: Reconcile diff ledger with final diff**

Run:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git diff --no-index --stat -- oldversion.tex ptychopinn_2025.tex || true
git diff --no-index --unified=8 -- oldversion.tex ptychopinn_2025.tex \
  > artifacts/work/text-quality-audit/final-oldversion-vs-ptychopinn_2025.diff || true
```

Expected:
- every final text diff appears in the audit ledger;
- every ledger row has a decision;
- reverted/tightened items no longer contain repo-internal or unnecessarily detailed prose.

- [ ] **E4: Whitespace check**

Run targeted check to avoid generated-log noise:

```bash
cd /home/ollie/Documents/ptychopinnpaper2
git diff --check -- ptychopinn_2025.tex reviewer_revision_checklist.md changelog.txt docs/backlog/2026-04-14-manuscript-text-diff-quality-audit.md
```

Expected: no whitespace errors in edited source/tracking files.

## Completion Criteria

- [ ] The audit ledger lists every meaningful text diff increment between `oldversion.tex` and `ptychopinn_2025.tex`.
- [ ] Each ledger row has a reviewer link, decision, action, and concise rationale.
- [ ] Manuscript prose contains no repo-internal jargon or implementation/provenance detail that belongs in artifacts or response notes.
- [ ] Valid reviewer responses remain in the manuscript, but in maximally succinct paper-facing language.
- [ ] Valid formatting improvements, especially bullet-to-paragraph flow improvements, are kept where they improve readability.
- [ ] `reviewer_revision_checklist.md` and `changelog.txt` reflect the final text state.
- [ ] `pdflatex` exits `0`, compiled text inspection passes, and targeted `git diff --check` passes.

## Execution Notes

- Do not stage or commit unless the user asks.
- Do not modify `oldversion.tex`.
- Do not treat `revision_plan.md` as authoritative.
- If a reviewer-response sentence requires a scientific claim not supported by the current figures/tables, stop and ask the user instead of inventing support.
