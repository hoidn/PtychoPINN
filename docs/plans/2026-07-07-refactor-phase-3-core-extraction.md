# Refactor Phase 3 — Backend-Neutral Core Extraction (staged plan)

> **Current disposition (2026-07-15) — historical staged design.**
> [`2026-07-07-refactoring-roadmap.md`](2026-07-07-refactoring-roadmap.md) is the
> sole live status, dependency, and work-selection authority for this initiative.
> No move below is complete. Later CI/probe/scaling changes grew the target modules,
> moved call sites, and invalidated the recorded importer/LOC baselines; Move 1 also
> overlaps current uncommitted grid-workflow changes. The interface sketches and
> hazards remain useful provenance, but every move still requires the fresh design
> spike/sub-plan already required by this document. The historical bodies below do
> not create completion gates. They are not independently executable.

> **Frame:** `docs/plans/2026-07-07-refactoring-roadmap.md` (Phase 3). Targets root
> generators **RG1** (no shared core) and the deep half of **RG5** (layering).
> **Depends on Phase 1** (silent→loud net) and **Phase 2** (config guard + central-
> module coverage). Highest payoff, highest risk, done last.

**Historical goal (superseded):** give the two backends a shared neutral core so
a physics/convention fix lands once. The live roadmap instead defines scientific
semantics and independent oracles once while retaining backend-native numerical
kernels where autograd, tracing, or runtime behavior requires them.

> **DRIFT NOTE (2026-07-10, re-verified at HEAD `74524eeb`):** Move 2 got cheaper —
> torch modules importing `ptycho.*` are now **5** (`config_bridge`, `config_factory`,
> `helper`, `workflows/components`, `dset_loader_pt_mmap`; the last dies with the
> pipeline plan → effectively 4), not 14. `params.py:56 import tensorflow` still
> present. Move 3's cross-backend runtime call moved to `components.py:1979` (import)
> / `:1988` (call) and gained a silent mean-reassembly fallback at `:1989` — P1
> Wave-B's re-pointed 3b should fail-loud it *before* this move replaces it. Move 1
> anchors intact (`grid_lines_workflow.py` still 2143 LOC; `scripts.*` imports at
> `:28,1095`). `ptycho/model.py` is 914 LOC (stale 2484 figure); the lazy singleton
> template is at `model.py:142-143`.

**Why staged, not bite-sized:** Moves 2–5 each carry real design questions (where the
core boundary sits, what the protocol signatures are, how `object_big` maps to an
explicit strategy). Per project process, **each move gets its own execution-ready
sub-plan authored after a short design spike** — this document sequences the moves,
fixes their interfaces at the sketch level, and states the gates. Only Move 1 is
mechanical enough to execute directly. Writing fake TDD steps for the others now
would be placeholder debt.

## Global constraints (inherited, load-bearing here)

- **Frozen core** (`ptycho/model.py`, `diffsim.py`, `tf_helper.py`) unmodified — the
  core-extraction produces a NEW neutral layer that both backends import; it does not
  move code out of the frozen TF files.
- **Current evidence rule:** fixture identity is preserved, exact shape/index
  mappings remain exact, and numerical results use registered tolerances. The
  affected governing contract determines the gate.
- **Current coordinate rule:** preserve the canonical semantic `(x, y)` and
  `local_offset_sign = -1`; represent storage transposes/order differences with
  typed layout adapters, not a free-form sign parameter.
- **Never let a compensating error survive a fix** — every move certifies correctness
  end-to-end per path with an independent oracle, not by comparing a path against itself.

---

## Move 1 — Evict `grid_lines_workflow.py` from the package (mechanical; do first)

> **Superseded architecture:** do not move this widely consumed workflow
> wholesale into `scripts/`. Decompose reusable application behavior by cohesion,
> keep compatibility exports during migration, and remove the remaining upward
> `scripts.studies.invocation_logging` dependency explicitly.

**Problem:** a 2,143-line study runner lives in `ptycho/workflows/` and imports
`scripts.*` (`grid_lines_workflow.py:28,1095`), creating a scripts↔package cycle
(RG5). Blast radius small: ~8 importers, all in `scripts/studies/`.

- [ ] **1.1** Grep every importer of `ptycho.workflows.grid_lines_workflow`.
- [ ] **1.2** Identify its 2–3 genuinely reusable helpers (`stitch_predictions`,
      `save_recon_artifact`) vs study-specific orchestration.
- [ ] **1.3** Promote the reusable helpers into a proper package module (e.g.
      `ptycho/workflows/stitching.py`), and move the study-orchestration bulk to
      `scripts/studies/grid_lines_workflow.py`. Repoint importers.
- [ ] **1.4** Confirm the package no longer imports `scripts.*` (grep
      `ptycho/ -e "import scripts" -e "from scripts"` → empty). Gate + study smoke. Commit.

**Exit:** the scripts↔package cycle is gone; the package has no upward deps.

---

## Move 2 — Backend-neutral core + TF-free `params` (DESIGN SPIKE required)

> **Superseded architecture:** do not move `RawData` or `params.cfg` wholesale
> into a neutral core. The live roadmap limits this direction to framework-free
> value objects/data records, makes `params.py` TF-free as a separate lifecycle
> slice, and requires an import-smoke feasibility proof before extraction.

**Problem (RG1):** `ptycho_torch` hard-imports `ptycho.*` (14 modules) and
`params.py:56 import tensorflow` drags TF into every torch import. There is no neutral
layer both backends can share.

**Target:** extract a `ptycho/core/` (or top-level `ptycho_core`) holding: the config
dataclasses (`config.py`), `RawData`, `metadata`, `geometry` (from Phase 2), and a
**TF-free** `params.py` (the `import tensorflow` serves almost nothing — spike
confirms). Both backends import `core` instead of `ptycho.*`.

**Design spike must resolve:** (a) exact module set that moves vs stays; (b) whether
`params.py`'s TF import has any real consumer (grep `tf\.` usage in `params.py`);
(c) import-rewrite mechanics (both packages + scripts + tests) — likely a codemod;
(d) how the frozen TF core keeps reading `params.cfg` after the move (it must — the
sealed dict from Phase 1 stays as the legacy sink).

**Gate:** this is the **precondition for Moves 3–4** (you cannot define a backend
interface while one backend is a hard import of the other). Land behind the full gate
+ parity fixtures; the import rewrite is mechanical but wide — do it as one atomic,
reviewable codemod commit, not piecemeal.

---

## Move 3 — `Reassembler` protocol (design spike; torch half already scoped)

> **Superseded architecture:** the simple protocol below conflates
> training-forward assembly, patch placement, and VarPro scale calibration. The
> live roadmap requires separate assembly, calibration, and reconstruction-policy
> contracts.

**Problem (RG1):** torch calls TF at runtime for reassembly
(`ptycho_torch/workflows/components.py:1535 → tf_helper.reassemble_position`), and
three torch variants coexist. No shared contract.

**Target interface (sketch):**
```
class Reassembler(Protocol):
    def reassemble(self, patches, offsets, patch_size, *, weighting) -> canvas
```
with a TF impl (wraps the frozen `tf_helper` functions — no core edit, just an
adapter) and a torch impl (the barycentric stitch core). The torch stitch-core
extraction is **already fully designed** in
`docs/plans/2026-07-06-pipeline-consolidation-tiers-0-2.md` Phase 2b — reuse it as
this move's torch implementation.

**Design spike must resolve:** the exact `offsets` convention the protocol declares
(carry the Phase-2 `geometry` typed coords, so the y,x↔x,y swap is a checked
conversion at the adapter, not a remembered `[..., ::-1]`); and the window knob
(`middle_trim` vs `stitch_crop_size` — the pipeline plan already chose `middle_trim`).

**Gate:** deletes the cross-backend runtime import; parity-tested once via the
existing fixtures. Depends on Move 2 for where the protocol lives. This is the
highest physics-parity payoff per line changed.

---

## Move 4 — `TrainingBackend` interface + composable workflow steps (design spike)

> **Superseded architecture:** do not introduce the broad ABC/registration design
> below. The live roadmap prefers explicit two-backend dispatch and only narrow
> trainer or inference-session protocols where an observed coupling justifies
> them.

**Problem (RG5):** `backend_selector.py` dispatches on a string to name-mangled twins
(`run_cdi_example` vs `run_cdi_example_torch`); studies reach into
`_train_with_lightning` and hand-roll 5 copies of "load→predict→stitch→evaluate"
because the package exposes only monolithic `run_cdi_example*`.

**Target:** an ABC `TrainingBackend` with `train(container, config) -> Results`,
`load_bundle(path) -> Model`, `infer(model, container) -> patches`; backends
**register** themselves (inverting `ptycho → ptycho_torch`). Split
`workflows/components.py` (both backends) into composable public steps
`load / train / predict / stitch / evaluate`, and collapse the 5 study eval loops onto
them.

**Design spike must resolve:** the public step signatures; how `parse_arguments`
(currently in the workflow layer) moves up to `scripts/`; the registration mechanism.

**Gate:** largest blast radius (CLIs, ptychodus interop, studies). Requires Moves 2
and 3. Land step-by-step behind Phase-2 coverage; each extracted public step ships
with its own unit test before a study is repointed at it.

---

## Move 5 — Explicit reassembly strategy replacing `object_big` gating (design spike)

> **Superseded architecture:** `object_big` also controls output channels/layout,
> grouping, merge behavior, and checkpoint interpretation. Its replacement is a
> versioned compatibility migration with separate output-layout, merge-policy,
> and support/canvas-geometry axes after the config/model and reconstruction
> slices; the single Boolean-like strategy below is insufficient.

**Problem (RG4/RG5):** `object_big` is derived from gridsize on some paths, free on
others, hardcoded on a third; it silently gates whole code paths and makes knobs
inert (GRIDLINES-OBJECT-BIG-001 and 3 sibling findings).

**Target:** a single `forward_assembly: Literal['single_patch','overlap_merge']`
(+ weighting for merge) chosen once at config-validation.
`validate_model_config` rejects contradictions (e.g. `training_patch_weighting` set
with `single_patch`) instead of ignoring them. In torch (not frozen), materialize as a
constructor-selected `merge_fn` so inert combinations become unconstructable; in the
frozen TF graph, keep the boolean but make the **bridge** the enforcement point
(cross-check `object.big` vs `gridsize` vs declared strategy when writing the legacy
dict).

**Design spike must resolve:** the derivation rule (single place); the migration
(introduce the field alongside `object_big`, assert consistency for one release, then
flip readers backend-by-backend). Behavior changes ONLY where flags currently
*contradict* the data — exactly today's bugs — so parity fixtures stay green.

**Gate:** independent of Moves 3–4 but benefits from Move 2's config unification.

---

## Sequencing

```
Move 1 (evict, mechanical) ──► Move 2 (core + TF-free params, SPIKE, precondition)
                                      ├──► Move 3 (Reassembler)  ──┐
                                      └──► Move 4 (TrainingBackend) ┤──► done
Move 5 (strategy) ── independent, benefits from Move 2 ───────────┘
```

**Historical sequencing (superseded):** the instruction to execute Move 1
directly no longer applies. Select work from the live roadmap's dependency model.

## Verification & Exit (per move)

1. **Historical gate (superseded):** current work uses claim-matched selectors,
   exact shape/index checks, registered numerical tolerances, and the checked-in
   CI harness at the boundary named by the live roadmap.
2. Each move ships with an independent-oracle test (not path-vs-itself).
3. Behavior-changing moves record NCC/MAE deltas before merge.
4. **Historical exit (superseded):** current completion outcomes are owned by the
   live roadmap; this document's neutral-core, universal-reassembler, broad
   backend-interface, and single-strategy targets are not current gates.
