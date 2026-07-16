# Refactor Phase 1 — Safety Net (execution plan)

> **Current disposition (2026-07-15) — historical execution record.**
> [`2026-07-07-refactoring-roadmap.md`](2026-07-07-refactoring-roadmap.md) is the
> sole live status, dependency, and work-selection authority for this initiative.
> The warn-mode `params.cfg` seal and the dead model-loading branch landed; test
> isolation, explicit array threading, fail-fast handlers, metadata conventions,
> and seam assertions remain candidate work requiring a fresh plan against current
> code. Post-roadmap CI/probe/scaling work superseded the line anchors and the old
> global units interpretation; current CI behavior is governed by
> [`spec-ptycho-core.md`](../specs/spec-ptycho-core.md). The task bodies below are
> retained for hazards and provenance only: they are not independently executable
> and do not create completion gates.

> **Frame:** `docs/plans/2026-07-07-refactoring-roadmap.md` (Phase 1). Targets root
> generators **RG2** (global-state seal) and **RG4** (implicit conventions) + the
> error-handling anti-patterns. **Highest-leverage phase** and the precondition that
> makes Phase 3 safe. **Additive only — no frozen-core edits.** TDD: every task is
> test-first. Commit per task. **No AI/Claude attribution.**

**Goal:** Convert this codebase's dominant failure mode (silent plausible output)
into loud failure: seal `params.cfg`, make tests order-independent, evict the
computed-array blackboard to the workflow layer, replace three silent-degrade error
handlers with fail-fast, and make load-bearing data conventions explicit as tagged
provenance + structural seam-asserts.

**Architecture:** All changes live in the non-frozen shell (`params.py`, `conftest`,
`workflows/*`, `metadata.py`, `model_manager.py`). Frozen files (`ptycho/model.py`,
`diffsim.py`, `tf_helper.py`) are only *read* — the shell writes/tags/validates and
mirrors into the legacy dict so frozen reads are unchanged.

**Anchors verified against HEAD `22d77509`:** `params.py` — `DEFAULT_CFG:59`,
`cfg=DEFAULT_CFG.copy():76`, `def set:123`, `def get:131`; `model.py` frozen
singleton — `_lazy_cache:142`, `_model_construction_done:143`;
`probe.py:66 params.set('probe', probe/norm)`; `conftest.py` has no per-test cfg
fixture; `grid_lines_workflow.py:1224` degrade-to-None; `components.py:395` inner
bare `except: Y_patches=None`; `model_manager.py:205` dead `and False` branch.

## Global constraints

- Real data must keep loading. Structural seam checks remain important, while
  value checks explicitly required by the normative data contract (for example,
  finite nonnegative count intensity) still apply.
- Historical parity intent was to preserve fixture inputs and numerical behavior.
  Current work uses registered tolerances plus exact shape/index requirements;
  floating output is not presumed byte-identical.
- Frozen-core files unmodified. The model-singleton reset is done by *external*
  manipulation in conftest (tests already do this), NOT by editing `model.py`.

---

> **EXECUTION STATUS (2026-07-07, updated 2026-07-10) — split into two waves:**
> **WAVE A = DONE** (SDD; ledger `.superpowers/sdd/progress-refactor-p1.md`):
> Task 1 (seal, warn-mode) `d6ac2d65`; Task 3c **PARTIAL** — dead `and False` branch only `940ea1d6`
> (its fail-loud fallback-restructure = "part b" is DEFERRED to Wave B); review Minors `f8d44eaf`.
> Whole-branch review PASS/keep-as-is; full torch gate exonerated (0 Wave-A failures, 904 passed).
> **2026-07-10 rebase note:** those commits now live on `fno-stable-archive-20260707` only —
> `fno-stable` was re-rooted at `f66e8d43` (2026-07-08). Wave-A **content verified present at
> HEAD `74524eeb`** (seal at `params.py:81-94,143`; entrypoint `seal()` calls at
> `config_factory.py:332,531`, `components.py:653`; `and False` branch gone — sibling dead guard
> `if False and` remains at `model_manager.py:278` for 3c-part-b).
> **WAVE B = still DEFERRED, gate RESTATED (2026-07-10; historical snapshot):** the original
> "gs2 #34/#42/#43/#44" markers are unverifiable post-rebase. They do not impose a
> standing READ-ONLY hold: coordinate only with an actually active concurrent executor under
> the current repository authority rules. At that snapshot, `components.py` was last touched
> Jul 9 by the **absolute-scaling migration** (`ffda33a7`→`73cb928b`);
> `grid_lines_workflow.py`/`model_manager.py` quiet since the overlay. Before starting Wave B,
> re-check in-flight initiatives on those three files (shared-tree rule), then proceed.
> **Anchor drift for Wave B:** Task 3b's target is GONE — `Y_patches = None` no longer exists in
> `components.py`; re-point 3b at the current `except Exception` sites
> (`components.py:1080,1747,1753,1796,1989`) under the Task-3 policy. The `:1989` site
> (TF-reassembly failure → silent mean-reassembly fallback) is a confirmed policy violation and
> the priority target. Task 4b.1's guard is now at `grid_lines_workflow.py:844` (was :846);
> 3a's substring-match degrade confirmed still live at `:1226`.

## Task 1 — Seal `params.cfg` (warn-mode)

**Files:** `ptycho/params.py` (edit); `tests/test_params_seal.py` (create).

**Design:** add module-level `_sealed = False` and
`_SEAL_WHITELIST = {'intensity_scale', 'probe', 'timestamp'}` (the known blackboard
keys). Add `def seal()` / `def unseal()`. In `set()` (`:123`): if `_sealed` and
`key not in _SEAL_WHITELIST`, `logger.warning("post-seal params.cfg write: %s (from %s)", key, caller)`.
Also route the existing unconditional `print("DEBUG: Setting ...")` through the logger
at debug level. Warn-mode first (no raise) so every out-of-order write surfaces
without breaking runs. Entry points (`config_factory` payload build /
`setup_configuration`) call `params.seal()` after the bridge completes.

- [ ] **1.1** Write `tests/test_params_seal.py::test_post_seal_nonwhitelist_write_warns`:
      `params.unseal(); params.set('gridsize', 1); params.seal();` then assert a
      `params.set('gridsize', 2)` emits a warning (via `caplog`) and that
      `params.set('intensity_scale', 5.0)` (whitelisted) does NOT. Add
      `test_unsealed_writes_silent`.
- [ ] **1.2** Run → FAIL (no `seal`/`_sealed`).
- [ ] **1.3** Implement `_sealed`, `_SEAL_WHITELIST`, `seal()`, `unseal()`, the
      `set()` guard, and the logger routing.
- [ ] **1.4** Run → PASS.
- [ ] **1.5** Call `params.seal()` at the end of the payload-build path in
      `ptycho_torch/config_factory.py` (the CONFIG-001 checkpoint) and
      `ptycho/workflows/components.py:setup_configuration`. Guard with `unseal()` at
      the top of each entrypoint so re-runs in one process reset cleanly.
- [ ] **1.6** Torch gate green; **inspect the warning log** — every post-seal write it
      flags is a latent order-dependence to fix in later tasks (record them in the plan).
- [ ] **1.7** Commit: `feat: seal params.cfg with whitelist + warn on post-seal writes`

---

## Task 2 — Order-independent tests (autouse cfg + singleton restore)

**Files:** `tests/conftest.py` (edit).

**Design:** a **function-scoped autouse** fixture that snapshots `params.cfg`
before each test and restores it after, and clears the frozen model singleton by
external attribute manipulation (`ptycho.model._lazy_cache.clear()`;
`ptycho.model._model_construction_done = False`) — the same reset tests already
hand-roll (A6), now centralized. No `model.py` edit.

- [ ] **2.1** Write `tests/test_conftest_cfg_isolation.py`: test A does
      `params.cfg['gridsize'] = 99`; test B asserts `params.get('gridsize')` is the
      default (proves restore). Order the two so B follows A.
- [ ] **2.2** Run → FAIL (state leaks A→B).
- [ ] **2.3** Add the autouse fixture to `tests/conftest.py` (snapshot deep-copy of
      `params.cfg`; on teardown replace contents; reset the two `model` module
      globals). Also add `params.unseal()` in setup so sealing from one test doesn't
      leak.
- [ ] **2.4 Historical gate (superseded):** run the isolation selector and the
      affected contract selectors. A full torch gate is not automatically
      required here under the central roadmap's evidence policy.
- [ ] **2.5** Commit: `test: autouse params.cfg + model-singleton isolation fixture`

---

## Task 3 — Fail-fast error handlers

Policy: an `except` may add context and re-raise, or guard an explicitly optional
side artifact — it may NOT change what data/model the pipeline computes with.

### 3a — `run_pinn_inference` degrade-to-None (`grid_lines_workflow.py:1224`)

- [ ] **3a.1** Test `tests/test_grid_lines_inference_failfast.py`: a non-XLA
      exception raised inside inference **propagates** (not swallowed to None); assert
      an unrelated error whose message contains `"fft"` is NOT misclassified as XLA.
- [ ] **3a.2** Run → FAIL (currently substring-matches `"fft"/"xla"/"dynamic"` → None).
- [ ] **3a.3** Replace with catching the specific XLA/shape exception class at the
      specific call; re-raise everything else. Thread an `allow_partial=False` flag so
      a study can opt into None-on-failure explicitly; default fails the row loudly.
- [ ] **3a.4** Run → PASS; gate green. Commit.

### 3b — silent label null (`components.py:395`)

- [ ] **3b.1** The OUTER shape check (`:384-393`) already warns + nulls on genuine
      shape mismatch — keep it. The INNER `except Exception: Y_patches = None`
      (`:394-396`) hides real bugs. Test: an unexpected error in Y inspection
      propagates.
- [ ] **3b.2** Remove the inner bare except (let it raise) OR narrow to the specific
      attribute/shape error the outer branch already handles. Run gate. Commit.

### 3c — model-loading cascade (`model_manager.py:205-243`)

- [ ] **3c.1** Delete the dead `elif os.path.exists(h5_model_path) and False:` branch
      (`:205`) entirely. Replace the print-and-`pass` fallbacks with a single declared
      format per artifact version + a loud `raise` (with the attempted paths in the
      message) when none match. Test: loading a metadata-only/absent artifact raises a
      clear error instead of returning a wrong/blank model (SINGLETON-SAVE-001 genus).
- [ ] **3c.2** Run gate + any `model_manager` tests. Commit.

---

## Task 4 — Explicit conventions: provenance tags + structural seam-asserts

### 4a — Declared provenance in metadata (`ptycho/metadata.py`)

> **Superseded design:** do not introduce generic `units` plus heuristic
> inference as a parallel contract. Current work must implement the inseparable
> `scale_contract_version` / `measurement_domain` pair and field-specific layout,
> frame, and probe-gauge semantics from `docs/specs/spec-ptycho-core.md`.
> Compatibility heuristics, if retained, must be explicit and non-claim-bearing.

- [ ] **4a.1** Test: writing a dataset stamps `units {amplitude|intensity}`,
      `axis_order {NHW|HWN}`, `coords_convention {xy|rowcol}`, `offset_sign {-1|+1}`
      into `_metadata`; reading a dataset **without** those keys falls back to today's
      heuristic (FORMAT-001 transpose detector; intensity/amplitude percentile) AND
      **logs the inferred value**. Old files must still load.
- [ ] **4a.2** Implement the schema extension + tag-on-read with logged inference.
      Do NOT reject untagged/contract-violating data (real datasets violate the
      nominal contract).
- [ ] **4a.3** Gate + a round-trip test on an existing fixture dataset. Commit.

### 4b — Structural seam-asserts (non-frozen seams only)

- [ ] **4b.1** `grid_lines_workflow.py:846` — the `shape[-1] == gridsize**2` guard
      must distinguish "false because gridsize==1" (skip, fine) from "false because
      channels-first tensor" (RAISE — TORCH-GS2-STITCH-001 checkerboard). Test both
      branches.
- [ ] **4b.2** `ptycho/projective_warp_xla.py` `translate_xla` (NOT frozen) — assert
      `batch % n_offsets == 0` before the modular gather instead of silently wrapping.
- [ ] **4b.3** **Frozen-core caveat:** the `shift_and_sum` `reshape(-1,2)` in
      `tf_helper.py` cannot be guarded in place (frozen). Add the `batch % n_offsets`
      assert at its **caller** in the non-frozen workflow instead.
- [ ] **4b.4** Gate + the new seam tests. Commit.

### 4c — Promote findings into code

- [ ] **4c.1** Pick the 2–3 highest-value Active findings in `docs/findings.md` that
      describe a *convention* (prioritize the sign/units ones that already regressed —
      TORCH-REASSEMBLY-SIGN-001, a units one). For each, land either a metadata tag
      (4a), a boundary assert (4b), or a parity test that pins the net convention.
- [ ] **4c.2** Mark each promoted finding Resolved in `docs/findings.md` with a
      pointer to the code/test that now enforces it. Commit.

---

## Verification & Exit

1. **Historical gate (superseded):** current work uses claim-matched selectors,
   registered numerical tolerances, exact shape/index checks, and the checked-in
   CI harness only at the boundary named by the central roadmap.
2. The seal warn-log is empty on a clean CLI run (or every entry is triaged).
3. Test isolation is checked with a deterministic two-order focused selector.
   The historical `pytest-randomly` command is not available in the current
   environment and is not an exit gate.
4. **Exit:** out-of-order `cfg` writes logged; tests isolated; three silent-degrade
   handlers fail loud; conventions tagged + asserted at seams; 2–3 findings Resolved;
   zero frozen-core edits.
