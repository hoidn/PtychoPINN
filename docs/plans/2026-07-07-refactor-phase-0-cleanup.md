# Refactor Phase 0 — Subtractive Cleanup (execution plan)

> **Frame:** `docs/plans/2026-07-07-refactoring-roadmap.md` (Phase 0). Targets root
> generator **RG3** (fork accumulation). Pure removal — no behavior change.
> Execute top-down; commit in logical groups. **No AI/Claude attribution** in commit
> messages. Do not check out `main`; do not create worktrees.

**Goal:** Remove ~1.5k+ LOC of grep-verified zero-importer dead modules, delete
dead-target tests that break collection, and archive dated one-off scripts — so the
search space for every later phase shrinks and no edit can land in a dead copy.

**Risk:** LOW. Every deletion is preceded by a zero-importer re-grep. The only gate
is the torch test suite staying green (a hidden import surfaces there).

**Anchors verified against HEAD `22d77509`.**

---

## Task 1 — Delete certain-dead modules

**Files (delete):**

| Path | LOC | Re-verify command |
|---|---:|---|
| `ptycho/logging.py` | 315 | `grep -rn -E "from ptycho\.logging import\|from ptycho import logging\|import ptycho\.logging\|from \.logging import" ptycho ptycho_torch scripts tests` (only a *commented* hit at `tf_helper.py:177` allowed) |
| `ptycho/autotest/testing.py` | 163 | `grep -rn "autotest.testing\|autotest import testing" ptycho ptycho_torch scripts tests` (docstring-only hit at `configuration.py:5` allowed) — **keep the rest of `autotest/`** |
| `ptycho/trash/model2.py` | 148 | `grep -rn "trash.model2\|from ptycho.trash" ptycho scripts tests` |
| `ptycho/plotting.py` | 113 | precise-import grep |
| `ptycho/losses.py` | 67 | precise-import grep |
| `ptycho/classes.py` | 61 | precise-import grep |
| `ptycho/workflows/visualize_results.py` | 63 | `grep -rn visualize_results ptycho scripts tests` |
| `ptycho/visualization.py` | 36 | precise-import grep (exclude `scripts/studies/pdebench_image128/visualization.py`) |
| `ptycho/function_logger.py` | 16 | precise-import grep |
| `ptycho_torch/train_dummy.py` | 75 | `grep -rn train_dummy ptycho_torch scripts tests` |

**File (edit):** `ptycho/misc.py` — remove the dead `make_invocation_counter` copy
(`:270`) and the commented `#def g(h)` block (`:283-322`, self-labelled "TODO
deprecated, moved to logging.py"). The live copy is `autotest/debug.py:44`.

- [ ] **1.1** Run every re-verify grep above. On ANY non-comment/non-docstring
      importer → STOP and reclassify that file.
- [ ] **1.2** Delete the 10 modules.
- [ ] **1.3** Edit `misc.py`: remove the dead function + commented block. Grep
      `make_invocation_counter` repo-wide → only `autotest/debug.py` remains.
- [ ] **1.4** `pytest --collect-only -q` → no new collection errors.
- [ ] **1.5** Torch gate: `pytest tests/torch -m "not slow"` (+ project
      deselect/ignore lists) == recorded green baseline.
- [ ] **1.6** Commit: `refactor: delete zero-importer dead modules in ptycho/`

---

## Task 2 — Resolve confidence-gated deletions individually

Each has a caveat; do NOT bulk-delete.

- [ ] **2.1 `ptycho/train.py` + `ptycho/train_supervised.py`** (~249 LOC). Header
      says "DO NOT USE"; 0 real importers. **Caveat:** shipped via
      `setup.py:17 scripts=['ptycho/train.py']`. Delete both files AND remove the
      `setup.py` entry in the same commit. Confirm real entrypoints
      (`pyproject.toml` `ptycho_train`/`ptycho_inference`) still resolve.
- [ ] **2.2 `ptycho_torch/train_full.py`** (199 LOC). Orphan `__main__`, 0 importers
      (the `config_params.py:179` "only used in train_full" comment is the sole
      reference). Grep runbooks/docs for `train_full` invocation; if none, delete and
      drop the now-orphaned `config_params` fields flagged "only used in train_full";
      else move to `_archive/`.
- [ ] **2.3 `ptycho_torch/api/`** (~3,100 LOC, deprecated ADR-003). **Defer to
      Phase 2** (it's built on the `config_params` family being collapsed there).
      Do not delete here.
- [ ] **2.4 `ptycho/single_image_frc.py`** (510 LOC, untracked). **Defer** to
      `docs/plans/2026-04-13-single-image-frc-removal.md` (already targets it).
- [ ] **2.5** Gate + commit per resolved item.

---

## Task 3 — Delete dead-target tests + fix collection

- [ ] **3.1** `tests/study/test_dose_overlap_*.py` (6–7 files, ~60 tests) — targets
      removed; `studies/fly64_dose_overlap/` holds only `__pycache__`. These are the
      6 hard collection errors breaking `tests/study/`. **Decide:** delete the tests,
      OR restore `studies/fly64_dose_overlap/*.py` if the study is still wanted
      (ask owner). Default: delete.
- [ ] **3.2** Delete `tests/old_test_tf_helper.py` (orphan, superseded).
- [ ] **3.3** Delete `tests/test_misc.py::test_memoize_simulated_data` (the
      `unittest.skip`-for-a-changed-API dead test).
- [ ] **3.4** `pytest --collect-only -q tests/study` → 0 collection errors.
- [ ] **3.5** Commit: `test: remove dead-target tests, fix study collection`

---

## Task 4 — Archive dated one-shot scripts

> **RESOLUTION (2026-07-07): DEFERRED — target list did not survive verification.**
> A reference sweep (docs + tests + sibling scripts) showed the plan's "dated one-shot"
> premise is false for the bulk of the list:
> - `diagnose_placement.py` is imported/invoked by 3 **live** sibling scripts
>   (`flux_sweep_eval.py`, `recon_quality_gate.py`, `compose_varpro_comparison_grid.py`).
> - `lines128_uno_preflight`, `run_fresh_ptychovit_initial_metrics`,
>   `verify_fresh_ptychovit_initial_metrics`, `run_corrected_ffno_{40ep_,}rerun`,
>   `analyze_dense_metrics` are each referenced by **active tests** (`tests/**`) and
>   authoritative docs (`TEST_SUITE_INDEX.md`, `docs/index.md`, `workflows/ptychovit.md`).
>   Moving them would break those tests / invalidate the pointers.
> - `diagnose_reconstruction.py` is the middle step of an active 3-step 2026-07-01
>   varpro-ablation diagnostic recipe whose siblings are doc-referenced — archiving it
>   alone is incoherent.
> - `export_cdi_natural_patch_patchwise_predictions.py` is UNTRACKED, recent (Jul 6), and
>   unreferenced — i.e. another initiative's uncommitted scratch; deleting files not
>   created by this initiative on the shared working tree is out of scope.
> - **4.2 plan error:** `scripts/training/` is NOT empty — it holds the live `ptycho_train`
>   entrypoint `scripts/training/train.py` (`pyproject.toml [project.scripts]`). Deleting it
>   would break the console entrypoint. Do NOT execute.
> - `debug_fno_{gradients,activations}.py` are referenced by `tests/torch/test_debug_fno_*`
>   → KEEP (as the plan already noted).
>
> Net: no target is *provably* safe to archive here. Study-script consolidation with proper
> reference-updating is already scoped to **Phase 2 Task 6** (shared study-runner scaffold);
> that is the correct home for this work. Task 4 yields no action in Phase 0.

Create `scripts/studies/_archive/` (git-tracked) and move; delete only if trivially
git-reproducible.

- [ ] **4.1** Move: `diagnose_{placement,reconstruction,stitching}.py`, the
      `born_rytov_dt/run_corrected_ffno_*rerun.py`, `lines128_uno_provenance_backfill.py`,
      `regenerate_lines128_*_manifest.py`, `lines128_uno_preflight.py`, the
      `*_fresh_ptychovit_initial_metrics.py` pair, `scripts/study/` shims.
- [ ] **4.2** Delete the untracked
      `scripts/studies/export_cdi_natural_patch_patchwise_predictions.py` and the empty
      `scripts/training/` dir; delete `scripts/debug_fno_{gradients,activations}.py`
      one-offs (confirm no test references first —
      `test_debug_fno_*` exist in the gate, so **keep** those two if the tests import them).
- [ ] **4.3** Grep `docs/COMMANDS_REFERENCE.md`, `README.md`, `docs/studies/index.md`
      for any archived script name; if referenced, update the pointer or don't archive
      (they may be documented entrypoints).
- [ ] **4.4** Commit: `chore: archive dated one-shot study scripts`

---

## Verification & Exit

- `pytest tests/torch -m "not slow"` green; `pytest --collect-only` clean.
- ~1.5k LOC (Task 1) + up to ~450 (Task 2) removed; ~60 dead tests gone; collection fixed.
- **Exit:** no dangling imports; gate green; `setup.py`/`pyproject` entrypoints resolve.
