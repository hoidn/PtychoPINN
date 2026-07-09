# Split main's overlay squash into granular logically-grouped commits

**Goal:** Replace the single ~365-file squash `8bd24e58` on public/internal `main` with a chain of ~9 logically-grouped commits whose **final tree is byte-identical** to the current one, then re-anchor the 4 TF-parity cherry-picks on top. History becomes blame/bisect-navigable; content does not change at all.

**Why:** `8bd24e58` (parent `32082e91`) is a 258-add/83-modify/24-delete monolith. Its own commit message already enumerates the logical groups (generators, model/dataloader rework, configuration, tooling, tests, cleanup, fixes) — the split materializes those groups as commits.

## Method: path partition (declared limitation)

Each file moves from its base state to its **final** state in exactly one commit — no intra-file splitting. Consequences, stated honestly:

- Every intermediate commit is **syntax-clean** (verified: `compileall` per commit) but only the **final** tree carries the full cross-file import closure (verified: the import gate + full CI gate already passed on this exact tree — `8bd24e58` and the ported tip are both GH-green).
- Commit ordering minimizes intermediate import breakage: config → generators → torch core → tooling; **all deletions last** (so nothing that still references a legacy module outlives its updated importers).

## Commit chain (parent `32082e91`, tip tree == `4593800a` == `8bd24e58^{tree}`)

| # | Commit | Contents |
|---|--------|----------|
| 1 | `chore: repo config, packaging, and CI environment` | .gitignore, .gitmodules, CLAUDE.md, pyproject, setup.py, requirements-ci.txt, tests/conftest.py, bootstrap_git.sh + its test |
| 2 | `feat(tf): TF-side data pipeline and workflow updates` | ptycho/ (raw_data incl. normalize_data float32 fix, loader, params, evaluation, image/, datagen, model_manager, tf_helper, workflows/) + their tests |
| 3 | `feat(config): extend canonical config dataclasses and bridges` | ptycho/config/config.py, ptycho_torch/config_{params,bridge,factory}.py + config tests + baseline fixture |
| 4 | `feat(torch): add FNO generator family with registry-based architecture selection` | ptycho_torch/generators/** + generator READMEs + generator/registry tests (headline commit) |
| 5 | `feat(torch): rework Lightning model, dataloader, and training around the registry` | ptycho_torch/ remainder (model, dataloader, helper, physics/, probe_mask, api/, eval/, workflows/) + torch core tests + varpro parity fixtures |
| 6 | `feat: ptychovit interop package and bridge` | ptycho/interop/**, ptychovit spec, ptychovit scripts + tests |
| 7 | `feat(tooling): study runners, dataset builders, overlay transform, analysis tooling` | scripts/** remainder (studies, grid_study, tools, internal, main_overlay transform), examples/, tests/studies + tests/study + tool tests |
| 8 | `docs: README and external interop specs` | README.md, specs/* remainder |
| 9 | `cleanup: remove dead legacy modules and stale artifacts` | all 24 deletions (ptycho legacy modules, loaders/, history/, autotest, trash/, bundled npz, stale logs, PlotNeuralNet + diagram leftovers) |

Then re-anchor (same trees, same messages/authorship, new parents): the 4 TF-parity cherry-picks currently at `2a4cd758..1857a85d`.

## Invariants (all machine-checked before push)

1. **Partition exactness:** every path in `git diff-tree -r 32082e91 8bd24e58` assigned to exactly one group; union == the full 365, intersection == empty.
2. **Tip identity:** tree of commit 9 == `4593800a` (byte-identical; `git diff-tree` empty). Re-anchored pick #4 tree == `1857a85d^{tree}`.
3. **Per-commit syntax:** `compileall` green on every intermediate tree (extracted tree objects).
4. **Message hygiene:** no "claude", no trailers, imperative mood.
5. No content re-verification needed at tip: the exact final tree is already CI-green on GitHub (runs on `8bd24e58` and the ported tip); the new push triggers a fresh run anyway (known caveat: the 5-test bit-exact runner flake — rerun once if red).

## Mechanics

Scratch-index surgery in the main checkout (no worktrees, no branch switches): start from `32082e91`'s tree in a temp `GIT_INDEX_FILE`; per group, `update-index --cacheinfo` each path to its final mode+blob (from `8bd24e58`) or remove it (deletions); `write-tree` + `commit-tree -p <prev>`. Classifier script + per-group file lists kept under `.artifacts/rebase-fno-stable-2026-07/split/`.

## Push & rollback

- Force-push the new tip to internal and origin main (ruleset currently disabled; user re-enables after green).
- Rollback anchor: pre-split tip `1857a85d` (= squash `8bd24e58` + 4 picks); superseded by the split chain.

## EXECUTED 2026-07-08

Chain `8221e69e → 898b42ee → 7a61a0aa → 92368728 → adaa879e → 7325c925 → 7bceb8ca → 7ca611af → 0ed9a42d` (9 split commits) + re-anchored picks `1e5218f1 → 8c54b2a0 → 0175a647 → 4bb52def`. All invariants verified: partition exact (365/365, disjoint), tip tree byte-identical (`4593800a`), content identical to pre-split tip, compileall green on all 9 intermediate trees, hygiene clean. Force-pushed to internal + origin main = `4bb52def`.

## Risks

- One more public history rewrite (last one — after this, main only advances by ordinary commits).
- Intermediate commits are not individually CI-green (path-partition limit above); bisect granularity improves from 1 unit to ~13 regardless.
