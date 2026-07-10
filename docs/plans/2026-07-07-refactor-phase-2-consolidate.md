# Refactor Phase 2 — Consolidate Shell + Backfill Coverage (execution plan)

> **Frame:** `docs/plans/2026-07-07-refactoring-roadmap.md` (Phase 2). Targets root
> generators **RG3** (script duplication), **RG5** (scripts-as-library), and the
> config fork. **Depends on Phase 0** (dead code gone). TDD. Commit per task.
> **No AI/Claude attribution.**

**Goal:** Collapse the four-store config toward two behind a drift guard, delete the
deprecated second orchestration stack (`api/`), extract the framework-free geometry
that seeds the sign-fork, and de-duplicate the ~10k LOC of copy-pasted study
infrastructure — while first backfilling isolating tests on the central modules the
later work touches.

**Risk:** MED. Contained: the giant study scripts have **zero fan-in** (A2), so their
blast radius is bounded; the config and geometry moves are the higher-risk part and
are gated on the coverage added in Task 1.

**Load-bearing rule (from A4):** *add isolating unit tests to a module before
refactoring it.* `reassembly.py` (1410 LOC) has **0 direct unit tests**; `model.py`
(torch) and `dataloader.py` (961) are thin.

> **DRIFT NOTE (2026-07-10, re-verified at HEAD `74524eeb`):**
> - **P0 dependency SATISFIED** (P0 effectively done — see its status block).
> - **Coordination gate:** the active gs2 plan declares `ptycho/config/config.py`,
>   `ptycho_torch/config_params.py`, `ptycho_torch/model.py` READ-ONLY for its
>   scope — Tasks 2/3 (and Task 1's model tests if they need fixture changes)
>   edit exactly those files. Confirm gs2 is closed or scoped around before Task 3.
> - **Target modules GREW via the absolute-scaling migration** (`ffda33a7`→`73cb928b`,
>   Jul 8–9): `dataloader.py` 961→**1710**, `reassembly.py` 1410→**1613**, torch
>   `model.py` **2759**. Task-1 characterization tests must pin **post-migration**
>   behavior (count-unit VarPro, physical batches) — do not reuse any pre-Jul-8
>   expected outputs.
> - Task 3.2 anchor moved: `hybrid_resnet_*`/`ffno_*` exec-config leak now at
>   `config.py:296-326`; the `model_type` registry (`config.py:101`) gained variants.
> - Task 6.1 partially eroded: `0d07c479` (Jul 9) already de-hardcoded 4 study
>   scripts; **14 files** still carry `/home/ollie` literals — the task stands.
> - `api/` still present (Task 2 unchanged); pipeline-plan Phase 0 dead code
>   (`reassembly_alpha/beta`, `beta_modules`, top-level `torch/`,
>   `dset_loader_pt_mmap.py`, `datagen.py`) still present — that plan has NOT
>   executed; its anchors (`a1d52011`) are also archive-only now.

## Global constraints

- Frozen core untouched; parity fixtures green; preserve the coord **sign flip** and
  **axis transpose** (per-pipeline invariants — Phase 1's tags/asserts guard them).
- Do not break reproducibility of documented paper scripts (grep
  `docs/COMMANDS_REFERENCE.md` before moving any referenced script).

---

## Task 1 — Backfill characterization tests (do FIRST)

- [ ] **1.1** `tests/torch/test_reassembly_unit.py` — isolating tests for the wired
      barycentric path in `ptycho_torch/reassembly.py`: canvas shape, probe-weighting
      on/off, VarPro s1/s2 application, on a small fixed input. Pin current outputs as
      a characterization fixture (this is the net that lets Phase 3 refactor it).
- [ ] **1.2** Thin-but-targeted unit tests for `ptycho_torch/dataloader.py` coord/
      grouping/normalize paths and for the torch `model.py` forward-predict contract.
- [ ] **1.3** Run green; commit. These are the merge gate for Tasks 3–4 and Phase 3.

---

## Task 2 — Delete the deprecated `api/` stack

`ptycho_torch/api/` (~3,100 LOC) is deprecated (ADR-003), built on the
`config_params` family, with zero production consumers (only
`tests/torch/test_api_deprecation.py`).

- [ ] **2.1** Grep-confirm no non-test importer of `ptycho_torch.api.*`.
- [ ] **2.2** Delete the `api/` package AND `test_api_deprecation.py` together.
- [ ] **2.3** Gate green; commit: `refactor: remove deprecated ptycho_torch/api stack`.

---

## Task 3 — Config drift guard + collapse toward canonical

- [ ] **3.1** Write `tests/test_config_drift.py`: walk `ptycho/config/config.py`
      dataclasses and `ptycho_torch/config_params.py` dataclasses; fail on any
      semantically-equal-but-differently-named field NOT covered by the bridge table
      (`config_bridge.py` transforms + `config.py:720 KEY_MAPPINGS`). Seed it with the
      known pairs (`gridsize`/`grid_size`, `nepochs`/`epochs`, `K`/`neighbor_count`,
      `nll`/`nll_weight`, `mode`/`model_type`). Run → it documents current drift.
- [ ] **3.2** Move the architecture-defining fields that leaked into
      `PyTorchExecutionConfig` (`config.py:287-326` `hybrid_resnet_*`, `ffno_*`,
      `spectral_bottleneck_*` — CONFIG-002 loophole) back to the model config; update
      the bridge; make the drift test pass.
- [ ] **3.3** Confirm no remaining consumer of the dead `.get()/.set()` singleton
      config API (already partly covered — `dset_loader_pt_mmap.py` deleted in the
      pipeline Phase 0). Gate; commit.

**Exit:** the 4-store config is reduced toward 2 (canonical dataclasses + write-only
legacy dict) and future drift is a red test, not a silent dropped field.

---

## Task 4 — Extract framework-free `ptycho/geometry.py`

`get_relative_coords` is duplicated (`raw_data.py:929` TF vs
`patch_generator.py:386` torch) and is the substrate of the sign-fork
(TORCH-REASSEMBLY-SIGN-001). It is NumPy-only — no framework dependence.

- [ ] **4.1** Test: a new `ptycho/geometry.py::get_relative_coords` reproduces BOTH
      backends' current outputs given their respective `local_offset_sign` and axis
      convention passed **explicitly** as arguments (not via a module global). Pin
      with the Phase-1 sign-parity fixture.
- [ ] **4.2** Implement `geometry.py` with sign + axis as explicit params. Migrate the
      torch caller (`patch_generator.py`) to import it; keep `raw_data.py` calling the
      identical logic (it's not frozen, but preserve its `-1` sign + `(M,1,2,C)` axis
      exactly).
- [ ] **4.3** Run the sign/axis parity fixtures — MUST be byte-identical. Commit.

**Note:** this move is atomic — do not partially migrate (partial reversals are the
documented failure mode). Both backends flip to `geometry.py` in one commit, gated by
the parity fixtures.

---

## Task 5 — Promote duplicated reconstruction metrics

The 5× study `metrics.py` (MAE/RMSE/SSIM/FRC) duplicate
`ptycho_torch/eval/eval_metrics.py` + `ptycho/evaluation.py` (A5 cluster A).

- [ ] **5.1** Consolidate the metric implementations into
      `ptycho_torch/eval/eval_metrics.py` (single MAE/RMSE/RelL2/SSIM with one
      signature); add a unit test pinning values against one existing study's numbers.
- [ ] **5.2** Repoint `{wavebench_shared_encoder,openfwi_flatvel_a,pdebench_swe,pdebench_image128}/metrics.py`
      to import from the package; delete the duplicated bodies.
- [ ] **5.3** Run the affected study smoke(s); confirm identical metrics. Commit.

---

## Task 6 — Shared study-runner scaffold + repo-env helper

Biggest LOC win (A5 clusters A/C/D/E). Do LAST in this phase (highest churn).

- [ ] **6.1** Create `scripts/lib/repo_env.py` (`repo_root()` via `Path(__file__)`/git)
      and replace the ~15 hardcoded `/home/ollie/...` literals (Cluster B). Test:
      `repo_root()` resolves from any CWD.
- [ ] **6.2** Extract the shared study-runner scaffold (`run_config` dataclass +
      argparse + train/infer invocation + manifest write) into
      `ptycho_torch/studies/base_runner.py` (or `scripts/lib/`); refactor the 5 SciML
      sub-studies to keep only their deltas. Keep each study's smoke test green.
- [ ] **6.3** (Optional, high-churn) merge the `grid_lines_*` orchestrators and the
      `paper_*` table generators onto the scaffold — only if their smokes cover the
      merge. Otherwise defer.
- [ ] **6.4** Commit per sub-item; run every touched study's smoke.

---

## Verification & Exit

- Task-1 characterization tests + all study smokes green after each task.
- Parity/sign fixtures byte-identical (Tasks 4).
- Config drift test green; `api/` gone; geometry unified; ~10k LOC of duplication
  removed/relocated; central modules now have isolating tests.
- **Exit:** the shell is de-duplicated and the config fork is guarded — Phase 3 can
  now build interfaces on a smaller, tested surface.
