# PtychoPINN Refactoring Roadmap

> **Status:** roadmap / strategic frame. Phases 0–1 are execution-ready; Phases
> 2–3 are move-level with gates (design detail deferred to per-phase plans).
> **How to use:** execute top-down. Phase 0 and Phase 1 are the recommended
> near-term work. Later phases depend on the safety net (P1) and coverage (P2)
> existing first. Commit per step; commit messages carry **NO** AI/Claude
> attribution (repo rule). Do **not** check out `main`; do **not** create worktrees.

**Goal:** Pay down accumulated debt across the dual-backend (TF `ptycho/` +
PyTorch `ptycho_torch/`) codebase — remove dead code, contain global state, make
load-bearing conventions explicit, consolidate duplication, and (eventually)
introduce a backend-neutral core — without disturbing the frozen physics core or
regressing parity fixtures.

**Evidence base:** a 6-analysis subagent sweep (2026-07-07): repo-wide dead code,
structural metrics, architecture/layering, test-suite health, script sprawl,
cross-cutting design smells. Anchors originally verified against `22d77509`;
re-verified against HEAD `74524eeb` on 2026-07-10 (see revision below).

**Target branch:** `fno-stable`.

---

## REVISION 2026-07-10 — status + drift since draft

**History rebuilt (2026-07-08):** `fno-stable` was re-rooted at `f66e8d43` (overlay
of pre-rebase `979bd517`; resnet family restored in `06cd27e6`). Pre-rebase history
— including this doc's anchor `22d77509`, the P1 Wave-A commits
(`d6ac2d65`/`940ea1d6`/`f8d44eaf`), and the pipeline plan's `a1d52011` — now lives
only on `fno-stable-archive-20260707`. **The content of those commits survived the
overlay** (verified at HEAD: seal live in `params.py:81–94,143`; dead modules gone).
Commit-sha citations across this plan set are provenance, not branch pointers.

**Phase status:**
- **P0 — effectively DONE** (executed pre-rebase, `c3ac2415` et al. on the archive;
  carried through the overlay). All 0a modules deleted; `train.py`/
  `train_supervised.py` + the `setup.py` scripts entry gone; `train_full.py` gone;
  `old_test_tf_helper.py` + all of `tests/test_misc.py` gone. The dose-overlap
  collection errors were fixed differently than planned — module-level **skips**
  (branch-local `studies/` is gitignored), so 0b's "delete to fix collection"
  motivation is obsolete; collection is clean (53 collected, 0 errors). Residuals
  listed in the phase plan. **P2's P0-dependency is satisfied.**
- **P1 — Wave A done, Wave B open.** Seal (warn-mode) + entrypoint wiring + dead
  `and False` branch landed. Wave B's gate needs restating post-rebase: the gs2
  item numbers are unverifiable against the new history; the live contention is now
  the **absolute-scaling migration** (`ffda33a7`→`73cb928b`, Jul 8–9), which touched
  `ptycho_torch/workflows/components.py` as recently as Jul 9 and **removed 1b.2's
  target** (`Y_patches = None` no longer exists in the file). `grid_lines_workflow.py`
  and `model_manager.py` are quiet since the overlay. See the P1 plan's updated status.
- **P2/P3 — not started.** P2 is now unblocked (P0 done) but must coordinate with
  the gs2 plan's initiative-scoped READ-ONLY set (`ptycho/config/config.py`,
  `ptycho_torch/config_params.py`, `ptycho_torch/model.py`) while gs2 is active —
  P2 moves 1–3 edit exactly those files.

**Root-generator evidence drift (re-measured at `74524eeb`):**
- **RG1 shrank:** torch modules importing `ptycho.*` are now **5** (was 14):
  `config_bridge`, `config_factory`, `helper`, `workflows/components`, plus
  `dset_loader_pt_mmap` (already on the pipeline dead list → effectively 4). The
  cross-backend runtime call persists (`components.py:1979` imports `tf_helper`,
  `:1988` calls `reassemble_position`) and gained an `except Exception →
  mean-reassembly` fallback (`:1989`) — a **new** silent-degrade site in P1 1b's genus.
- **RG3 grew:** the `model_type` registry gained variants (`config.py:101` — 14
  architectures incl. new hybrid/spectral forks).
- **RG5 re-based:** `ptycho/model.py` is **914 LOC** (the 2484 figure is stale);
  the oversized central modules are torch-side and growing:
  `ptycho_torch/model.py` 2759, `dataloader.py` 1710 (was 961), `reassembly.py`
  1613 (was 1410). P2's characterization tests must pin **post-absolute-scaling**
  behavior, not the pre-migration outputs this plan set assumed.
- **RG2 unchanged:** `update_legacy_dict` grep now ~212 hits repo-wide.

**New program-level facts:** a CI test gate now exists (`ci/` + workflow backported
in `a73303af`; see `docs/ci.md`); `main` is a rebuilt 9-commit split chain (no
hybrid/stable_hybrid archs); fno-stable→main promotion stays gated + owner-approved.
Known flake: 5 bit-exact parity tests fail on some GH-runner CPUs — treat CI parity
failures as suspect-flake before suspect-regression; **local** parity fixtures
remain the hard merge gate.

**Next actionable:** (1) close P0 with the residual sweep (phase plan); (2) re-check
hot-file contention, then execute P1 Wave B with 1b.2 re-pointed at the *current*
silent-degrade sites (incl. `components.py:1989`); (3) P2 once gs2's READ-ONLY hold
lifts or is scoped around.

---

## The debt reduces to 5 root generators

Attack generators, not the ~30 individual symptoms. Each phase targets specific ones.

| RG | Root generator | What it produces | Evidence |
|---|---|---|---|
| **RG1** | **No backend-neutral core.** Torch hard-depends on the TF package; `params.py:56 import tensorflow` means `import ptycho_torch` pulls in TF; torch calls TF at runtime for parity. | "Physics fix must land 2–3×"; hand-maintained parity apparatus; sign/units/axis findings. | 5 torch modules import `ptycho.*` (was 14 at draft); `ptycho_torch/workflows/components.py:1979→1988 tf_helper.reassemble_position` |
| **RG2** | **`params.cfg` global blackboard.** Config flows as side effects, not through the call graph; the dict holds computed *arrays* (probe, intensity_scale), not just knobs. | Order-dependence (CONFIG-001); stale-state-across-runs (shipped SINGLETON-SAVE-001 bug); test fragility; no in-process parallelism; the 4-store config mess. | `probe.py:66 params.set('probe', …)`; ~24 files / ~98 `update_legacy_dict` sites; 406 `params.cfg` refs in tests; no per-test restore |
| **RG3** | **Research forks with no demolition date.** alpha/beta/main variants, `_old` fns, dead stubs, RED-phase test scaffolds, per-experiment script forks. | ~7.6k LOC dead code + ~10–13k LOC relocatable script duplication. | A1 dead list; A5 5 duplicate clusters; A4 17 RED-phase files |
| **RG4** | **Implicit load-bearing conventions.** sign/axis/units/stitch invariants exist only as mutual consistency of distant sites — unrepresented in types or data. | Every silent-plausible-output bug in `docs/findings.md`; naive fixes cause *compensating* errors. | TORCH-REASSEMBLY-SIGN-001, TORCH-GS2-STITCH-001, RECT-MAE-UNITS-001, FORMAT-001, REASSEMBLY-BRIDGE-001 |
| **RG5** | **Eroded layering / scripts-as-second-library.** A study runner lives in the package and imports `scripts.*`; studies reach into package privates; oversized central modules. | scripts↔package cycle; "where does this belong" ambiguity; wide-blast-radius god modules. | `grid_lines_workflow.py:28,1095`; 254 intra-scripts imports; `ptycho_torch/model.py` 2759 LOC (ptycho/model.py is 914 — the 2484 figure was stale) |

**Dependency structure:** RG3 is independent and cheapest. RG2-seal + RG4 are
additive safety work. RG1 and the deep half of RG5 are structural and require RG2
contained first. This ordering is *forced* by the codebase's dominant failure
mode — **silent plausible output** — which means the loud-failure net (P1) must
precede structural change (P3), or P3 reintroduces exactly the ledger's bugs.

---

## Global Constraints (bind every phase)

- **STABLE-CORE frozen (CLAUDE.md §6):** `ptycho/model.py`, `ptycho/diffsim.py`,
  `ptycho/tf_helper.py` must NOT be modified. *All* leverage is in the non-frozen
  shell (bridges, factories, workflows, dataloaders, params.py, config). Every
  migration below is expressible as "the shell passes/tags/validates explicitly
  and mirrors into the legacy dict" — keeping the frozen core's reads unchanged.
- **Real data must keep loading.** Live datasets violate the *nominal* contract
  (amplitude-vs-intensity units, `(H,W,N)` vs `(N,H,W)` layout). Never add
  value-based asserts to loaders — assert on *structure* at *seams*, and record
  heuristic inferences as provenance instead of rejecting.
- **Parity fixtures stay green** (`tests/fixtures/varpro_parity/*`,
  `test_cross_branch_rectangular_parity.py`, forward-parity fixtures). They pin
  intentional quirks (e.g. `RectangularMAELoss` double-square) — do not "clean up"
  behavior they encode.
- **Convert silent → loud before wrong → right.** A guard that *raises* is worth
  more per line here than any refactor, and is additive/safe under the freeze.
- **Every experiment path gets a demolition date.** New variants land with either
  a parity test tying them to the reference, or an expiry note.
- **POLICY-001:** PyTorch ≥2.2 mandatory; torch workflows run
  `update_legacy_dict` before touching legacy modules.

---

## Program overview

| Phase | Emphasis | Targets | Risk | Impact | Depends on |
|---|---|---|---|---|---|
| **0** | Subtractive cleanup | RG3 | **Low** | −~7.6k dead LOC; smaller search space | — |
| **1** | Safety net (**highest leverage**) | RG2-seal, RG4, error-handling | **Low** (additive) | silent→loud; contains global state | — |
| **2** | Consolidate shell + backfill coverage | RG3, RG5-scripts, config | **Med** | −~10k LOC; config 4-store→2; test the thin-covered central modules | P0 |
| **3** | Extract backend-neutral core | RG1, RG2, RG5-deep | **High** | "physics fix lands once"; maintainable dual backend | P1, P2 |

Phases 0 and 1 are independent of each other and can proceed in parallel.

---

## Phase 0 — Subtractive cleanup (effort S, risk LOW)

> **STATUS 2026-07-10: effectively DONE** (see revision above and the phase plan's
> execution-status block for residuals). Section kept as drafted for provenance.

Pure removal. Re-run the zero-importer grep at execution time before each delete.

### 0a. Delete confirmed-dead modules (re-verified against `22d77509`)

| Target | LOC | Evidence | Caveat |
|---|---:|---|---|
| `ptycho/logging.py` | 315 | 0 importers (only a *commented* import at `tf_helper.py:177`); embeds toy demo fns | verify no string-based lookup of `load_logged_data` first |
| `ptycho/autotest/testing.py` | 163 | 0 importers (lone hit is a docstring in `configuration.py`); rest of `autotest/` is LIVE | delete only `testing.py`, keep the package |
| `ptycho/trash/model2.py` | 148 | in `trash/`, 0 importers, last touched 2023 | — |
| `ptycho/plotting.py` | 113 | 0 importers | — |
| `ptycho/losses.py` | 67 | 0 importers; "prototype implementations in commented form" | — |
| `ptycho/classes.py` | 61 | 0 importers; never-populated stub | — |
| `ptycho/workflows/visualize_results.py` | 63 | 0 importers, orphan `__main__` | — |
| `ptycho/visualization.py` | 36 | 0 importers | — |
| `ptycho/function_logger.py` | 16 | 0 importers | — |
| `ptycho_torch/train_dummy.py` | 75 | 0 importers, no `__main__` | — |
| dead `make_invocation_counter` copy + commented block | ~50 | `ptycho/misc.py:270,283-322`; self-labelled "TODO deprecated, moved to logging.py" | live copy is `autotest/debug.py:44` |

Plus the **already-known pipeline dead code** (deferred to the pipeline plan's Phase 0
so they aren't deleted twice): `reassembly_alpha.py`, `reassembly_beta.py`,
`beta_modules/reassembly.py`, top-level `torch/`, `dset_loader_pt_mmap.py`,
single-file `datagen.py`, `reassembly.py:722/1323`. See
`docs/plans/2026-07-06-pipeline-consolidation-tiers-0-2.md` Phase 0.

**Confidence-gated (do NOT bulk-delete — resolve each first):**
- `ptycho/train.py` (+ `train_supervised.py`, ~249 LOC) — header says "DO NOT USE",
  0 real importers, **but shipped via `setup.py:17 scripts=['ptycho/train.py']`.**
  Deletion must also remove that `setup.py` entry.
- `ptycho_torch/train_full.py` (199 LOC) — orphan `__main__`, 0 importers, but
  `config_params.py:179` comments reference "only used in train_full". Confirm no
  external runbook calls `python -m` before deleting; else archive.
- `ptycho_torch/api/` (~3,100 LOC, deprecated ADR-003 island) — 0 production
  consumers; only `tests/torch/test_api_deprecation.py` imports it. Delete the
  package **and** that test together, or keep both. (Recommended: delete in Phase 2
  as part of the config unification, since `api/` is built on the `config_params`
  family being collapsed there.)
- `ptycho/single_image_frc.py` (510 LOC) — **untracked** orphan already targeted by
  `docs/plans/2026-04-13-single-image-frc-removal.md`; superseded by `evaluation.py`.

### 0b. Delete dead-target tests + fix collection (A4)

- `tests/study/test_dose_overlap_*.py` (6–7 files, ~60 tests) — targets removed;
  `studies/fly64_dose_overlap/` is empty but for `__pycache__`. These 6 hard
  **collection errors** break the whole `tests/study/` package. Delete the tests
  (or restore `studies/fly64_dose_overlap/*.py` if the study is still wanted).
- `tests/old_test_tf_helper.py` — orphan, non-collected, superseded.
- `tests/test_misc.py::test_memoize_simulated_data` — `unittest.skip` for a
  long-changed API; dead.

### 0c. Archive dated one-shot scripts (A5)

Move to `scripts/studies/_archive/` (or delete if git-reproducible), don't leave in
the live tree: `diagnose_{placement,reconstruction,stitching}.py`, the `*_rerun` /
`*_backfill` / `*_preflight` one-offs, the `fresh_ptychovit` pair, `scripts/study/`
shims. Delete the untracked `export_cdi_natural_patch_patchwise_predictions.py` and
the empty `scripts/training/` dir.

- [ ] 0.1 Re-grep zero-importer for every 0a target; STOP on any live importer.
- [ ] 0.2 Delete 0a certain list (11 modules). Grep-confirm no dangling refs.
- [ ] 0.3 Resolve the 4 confidence-gated items individually (setup.py edit; runbook check).
- [ ] 0.4 Delete 0b dead tests; confirm `pytest --collect-only` has 0 collection errors.
- [ ] 0.5 Archive 0c scripts.
- [ ] 0.6 Full gate: `pytest tests/torch -m "not slow"` (+ project deselect/ignore) == green baseline; a hidden import surfaces here.
- [ ] 0.7 Commit in logical groups (modules / tests / scripts).

**Exit:** ~1.5–7.6k LOC removed; collection clean; gate green.

---

## Phase 1 — Safety net (effort M, risk LOW; **highest leverage**)

> **STATUS 2026-07-10: Wave A done** (1a.1 seal + 1b.3 dead branch); **Wave B open**
> (1a.2–1a.4, 1b.1, 1b.2 re-pointed, 1c). See the phase plan's execution-status block.

Additive only. No frozen-core edits. Converts the codebase's silent failures into
loud ones and contains RG2. This is the precondition that makes Phase 3 safe.

### 1a. Seal `params.cfg` + evict the blackboard (the single highest-leverage move)

- [ ] **1a.1** In `ptycho/params.py` (not frozen), wrap `cfg` writes with a
      **seal**: after the entrypoint bridge runs, further `set()`/`update_legacy_dict`
      writes outside a whitelist (`intensity_scale`, `probe`, `timestamp` — the known
      blackboard keys) **log a warning** with the caller. Start in warn-mode (no
      raise) to surface every out-of-order write without breaking runs. Also gate the
      unconditional `DEBUG: Setting` print (`params.py:124`) behind the logger.
- [ ] **1a.2** Add a `reset()` lifecycle API for the model singleton
      (`ptycho/model.py` is frozen — put the reset in a non-core module or as an
      additive ~6-line function behind a plan exception; it's the missing lifecycle
      hook behind SINGLETON-SAVE-001, not a physics change).
- [ ] **1a.3** In `tests/conftest.py`, add an **autouse fixture** that
      snapshots/restores `params.cfg` and clears the model-construction singleton
      per test. Removes the order-dependence across the 406 test refs.
- [ ] **1a.4** Evict the blackboard *arrays* at the workflow layer (non-frozen):
      thread `intensity_scale` and the probe as explicit arguments through
      `workflows/components.py` / `grid_lines_workflow.py`, while still mirroring
      them into `params.cfg` so the frozen core's reads (`model.py:224,258`) keep
      working. The dict becomes write-once-then-mirror, not a live channel.

### 1b. Kill silent-degrade error handling (A6)

Policy: *an except block may not change what data/model the pipeline computes with;
it may only add context and re-raise, or guard an explicitly optional side artifact.*

- [ ] **1b.1** `grid_lines_workflow.py:1224` — replace the catch-all +
      substring-match-`"fft"/"xla"` → return `None` with a specific exception class
      caught at the specific call; re-raise everything else. A missing study row
      fails the run unless `--allow-partial`. (In an FFT codebase, matching "fft" in
      messages is a landmine.)
- [ ] **1b.2** *(re-pointed 2026-07-10 — the original `Y_patches = None` target no
      longer exists post-absolute-scaling)* Triage the **current** `except Exception`
      sites in `ptycho_torch/workflows/components.py` (`:1080,1747,1753,1796,1989`)
      against the policy above; the `:1989` TF-reassembly → mean-reassembly fallback
      is a confirmed violation (changes what the pipeline computes with).
- [ ] **1b.3** `model_manager.py:205-242` — the 4-way format fallback cascade with a
      dead `and False` branch and print-and-continue makes "loaded the wrong thing"
      indistinguishable from success (SINGLETON-SAVE-001 genus). Collapse to one
      declared format per artifact version, loud failure otherwise; delete the dead branch.

### 1c. Make conventions explicit as provenance + structural asserts (RG4)

- [ ] **1c.1** Extend NPZ `_metadata` (`ptycho/metadata.py`) with declared
      `units {amplitude|intensity}`, `axis_order {NHW|HWN}`,
      `coords_convention {xy|rowcol}`, `offset_sign {-1|+1}`. Loaders *tag* on read;
      absent tags fall back to today's heuristics (FORMAT-001 transpose detector,
      intensity/amplitude percentile) but the inference is **recorded and logged** —
      auditable guessing, not silent. Old files keep loading forever.
- [ ] **1c.2** Add **structural** seam asserts (not value asserts): the
      `grid_lines_workflow.py:846` `shape[-1]==gridsize**2` guard must distinguish
      "false because gridsize==1" from "false because channels-first" and raise on
      the latter (TORCH-GS2-STITCH-001 checkerboard); `translate_xla`'s modular gather
      and `shift_and_sum`'s `reshape(-1,2)` must assert `batch % n_offsets == 0`
      instead of silently broadcasting.
- [ ] **1c.3** Promote 2–3 highest-value Active findings from `docs/findings.md`
      into code (a metadata tag, a boundary assert, or a parity test), then mark them
      Resolved. Prioritize the sign/units ones that have already regressed.

- [ ] 1.x After each sub-item: run the torch gate (green) + confirm parity fixtures byte-identical. Commit per sub-item.

**Exit:** out-of-order `cfg` writes are logged; tests are order-independent; the
three silent-degrade handlers fail loud; data conventions are tagged + asserted at
seams; no frozen-core edits; parity green.

---

## Phase 2 — Consolidate shell + backfill coverage (effort L, risk MED)

Depends on P0. **Rule: add isolating unit tests to a module BEFORE refactoring it.**
A4 flagged the exact central modules being changed as thin-covered:
`reassembly.py` (1410 LOC, **0 direct unit tests**), `model.py`, `dataloader.py`.

**Moves:**
1. **Delete `ptycho_torch/api/`** (deprecated, zero prod consumers) + its
   deprecation test. Removes the second orchestration stack + the second config family.
2. **Collapse `config_params` into the canonical `ptycho/config/config.py` schema**
   behind a **drift-detection test** that walks both dataclasses and fails on any
   semantically-equal-but-differently-named field missing from the bridge table
   (`gridsize`/`grid_size`, `nepochs`/`epochs`, `K`/`neighbor_count`, …). Turns future
   drift into a red test. Shrinks the 4-store config toward 2.
3. **Extract a framework-free `ptycho/geometry.py`** (NumPy-only `get_relative_coords`,
   grouping, offset conventions), imported by both backends — deletes the sign-fork
   substrate permanently (frozen `tf_helper.py` untouched).
4. **Promote the 5×-duplicated reconstruction metrics** into
   `ptycho_torch/eval/eval_metrics.py` / `ptycho/evaluation.py`; studies import them
   (A5 cluster A).
5. **Extract one shared study-runner scaffold** (`run_config` dataclass + argparse +
   train/infer + manifest write) into the package; sub-studies keep only deltas (A5
   cluster A/C/D/E — the single biggest LOC win). Add a `scripts/lib/repo_env.py` to
   kill the ~15 hardcoded `/home/ollie` paths.

**Exit:** config 4-store → 2 with a drift guard; geometry unified; ~10k LOC of
script duplication removed/relocated; the central modules have isolating tests.

---

## Phase 3 — Extract backend-neutral core (effort XL, risk HIGH)

Depends on P1 (safety net) + P2 (coverage + config unification). Move-level; each
move gets its own detailed plan before execution. Sequence (per the A3 analysis):

1. **Evict `grid_lines_workflow.py` from `ptycho/workflows/` into `scripts/studies/`**
   (or promote its 3 reusable helpers into the package) — kills the only
   package→scripts imports and the scripts↔package cycle. Small blast radius (~8
   importers); do first.
2. **Extract a backend-neutral core** (config dataclasses + `RawData` + metadata +
   a **TF-free** `params.py`). The `import tensorflow` at `params.py:56` serves
   almost nothing; removing it lets `ptycho_torch` import `core` instead of
   `ptycho.*` (5 wrong edges as of 2026-07-10, effectively 4 — the smaller count
   makes this move cheaper than at draft). **Precondition for moves 3–4.**
3. **Define a `Reassembler` protocol** and give torch inference its own
   implementation — deletes the cross-backend runtime call
   (`components.py:1979/1988`), consolidates the reassembly variants behind one contract,
   parity-tested once. This is the "physics fix lands once" seam. (The pipeline plan's
   stitch-core extraction is the torch half of this.)
4. **Replace `backend_selector`'s string dispatch with a `TrainingBackend` interface**
   + split `workflows/components.py` into composable public steps
   (`load → train → predict → stitch → evaluate`). Inverts the `ptycho → ptycho_torch`
   dependency and collapses the 5 hand-rolled study eval loops onto public steps.
5. **Replace the `object_big` magic gating with an explicit reassembly strategy**
   (`forward_assembly: Literal['single_patch','overlap_merge']`) chosen once at
   config validation, rejecting contradictory flag combos instead of silently
   ignoring them (GRIDLINES-OBJECT-BIG-001 class).

**Gates:** each move lands behind the P1 seam-asserts + P2 coverage; parity fixtures
are the merge gate for anything touching coords/reassembly; the `model.py:142` lazy
singleton (`_lazy_cache`/`_model_construction_done`) is the template for the params
migration.

---

## Relationship to existing plans

- `docs/plans/2026-07-06-pipeline-consolidation.md` (+ `-tiers-0-2.md`) is a
  **vertical slice** through this roadmap (reassembly/inference/solver specifically).
  Its dead-code deletion = part of P0; its solver + stitch-core extraction = the
  torch half of P3 move 3. No rework — that plan is one column of this frame.
- This roadmap is the **horizontal frame**; individual phases spawn their own
  execution-ready plans (P1 especially — the `params.cfg` seal deserves its own
  detailed plan before execution).

## Verification protocol (every phase)

1. `pytest tests/torch -m "not slow"` (+ project deselect/ignore) matches the
   recorded green baseline — no new failures.
2. Parity fixtures byte-identical (P1–P3, anything touching physics/coords/config).
   **Local** runs are authoritative; on CI (post-`a73303af` gate, `docs/ci.md`),
   the 5 bit-exact parity tests are a known CPU-dependent flake — rule out the
   flake before treating a CI parity failure as a regression.
3. P2/P3 behavior-changing moves: re-run one study/inference smoke; record NCC/MAE
   deltas before marking complete.
4. Commit per step; **no AI/Claude attribution** in messages.

## Sequencing

```
P0 (delete) ─┐
             ├─► P2 (consolidate + coverage) ─► P3 (core extraction)
P1 (net) ────┘                                    ▲
   └──────────────────────────────────────────────┘  (P1 is a precondition for P3)
```

P0 and P1 are independent and are the recommended near-term work. Start P3 only once
the net (P1) and coverage (P2) exist.
