# Task 28 Report - Bridge Convergence Validation Closeout

**Date:** 2026-07-12
**Branch:** `feature/ci-compatibility-ablation`
**Status:** `complete_final`
**Implementation commits:** `65c802e17`, `9c6b08ca2`, `292b18be8`
**Canonical root:**
`.artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun`

## Outcome And Root Cause

Task 28 closed the corrected-physics dictionary-to-mmap bridge regression.
The original rung1a quality failure was caused by normalization ownership, not
different measurement bytes or sampler policy. Grid-lines simulator output is
already normalized by `ptycho.diffsim.illuminate_and_diffract`; applying the
generic loader's Batch-derived RMS and physics constants conditioned it a
second time. The executable bridge now uses unit `dictionary_parity` for these
simulator-owned normalized amplitudes. The public `DataConfig` default,
explicit legacy `Batch` selections, and CI/count behavior remain unchanged.

The canonical rerun returned PASS against rung0. Its verdict reason and
protocol-failure reason are both null. Task 28 adjudicates only rung0 versus
canonical rung1a; it does not claim that Tasks 22-24 ran or that the corrected
multi-seed compatibility matrix passed.

## Implementation And Reviews

- `65c802e17` aligned ladder normalization ownership, archived obsolete
  diagnostic rungs from the executable TOML, and retained historical support.
- `9c6b08ca2` tightened fail-closed normalization evidence requirements.
- `292b18be8` pinned gain-calibration normalization behavior.
- Implementation spec-compliance review: APPROVED.
- Implementation code-quality review: APPROVED.
- Finalization quality re-review: APPROVED after corrective commit `6c4863c6c54600e4086662052c1ccab68080371e`.

## RED And GREEN Evidence

The focused test was updated before the status/docs patch. It made the fresh
PASS canonical, retained the old FAIL as historical evidence, and required all
live status surfaces to carry the canonical path, hashes, metrics, dependency
state, and retention decision.

```bash
python -m pytest -q \
  tests/studies/test_grid_lines_bridge_ladder.py::test_task28_rung1a_pass_evidence_is_canonical_v2 \
  tests/studies/test_grid_lines_bridge_ladder.py::test_task28_ladder_report_pins_canonical_pass \
  tests/studies/test_grid_lines_bridge_ladder.py::test_task28_historical_rung1a_fail_remains_pinned \
  tests/studies/test_grid_lines_bridge_ladder.py::test_task28_historical_fail_report_remains_pinned \
  tests/studies/test_grid_lines_bridge_ladder.py::test_task28_surfaces_route_canonical_pass_and_unblock_task22
```

RED result: `4 passed, 1 failed`. The expected failure was the stale plan,
which lacked the canonical PASS delta/pin and still routed Tasks 22-24 through
Task 28's open blocker.

GREEN result after the authority-surface patch: `5 passed in 1.37s`.

The first full adjacent run then exposed that the two fresh canonical files
were still git-ignored rather than promoted: `267 passed, 2 failed`. Both files
were force-added without changing their bytes. The repeated full run passed:

```bash
python -m pytest -q \
  tests/studies/test_grid_lines_bridge_ladder.py \
  tests/studies/test_step_parity.py \
  tests/studies/test_cross_eval_matrix.py \
  tests/studies/test_gain_calibration.py
# 269 passed, 50 warnings in 49.38s
```

The warnings are the existing `config_factory.py` test-data and populated-cfg
warnings; there were no test failures.

## GPU Command And Inputs

The canonical single-rung GPU execution used the checked ladder spec, the
existing exact generic twin, seed 3, gain 16 inherited from the resolved rung
configuration, and a fresh output root:

```bash
python -m scripts.studies.ablation.runtime_ladder \
  --spec scripts/studies/specs/grid_lines_bridge_ladder.toml \
  --datasets-root .artifacts/bridge_ladder/task28_gain16_seed3/datasets \
  --output-root .artifacts/bridge_ladder/task28_gain16_seed3_unit_scaling_rerun \
  --rung rung1a_mmap_full_scanset \
  --seed 3 \
  --base-dir .
```

The resolved rung uses Hybrid ResNet, N=128, C=1, 5 epochs, batch size 16,
legacy amplitude MAE, `amplitude_physics_gain=16`, mmap loader,
`mmap_bounds_filter=off`, `mmap_scale_convention=dictionary_parity`, sequential
training sampler, modes probe layout, and the historical gated evaluator.

Exact twin SHA-256 values remained unchanged:

- train: `628cac77ef85c3927e3d5407f509556f054267e71e567aed67500b8de5f6ae4e`
- test: `17b2aea9a9deeb3ead2ab78771f19b33a2612b2666196e20dd45fa1a51f2275b`

## Canonical PASS Evidence

| Field | Value |
|---|---|
| Amp MAE | `0.07664361596107483` |
| Amp SSIM | `0.8913340876617375` |
| Phase MAE | `0.11914721730481034` |
| Phase SSIM | `0.9632217816205675` |
| Absolute amp delta from rung0 | `0.0054688232603687 <= 0.02` |
| Absolute phase delta from rung0 | `0.0013551856818027 <= 0.01` |
| Verdict / reason / protocol failure | `pass` / null / null |
| Rung evidence SHA-256 | `a8741f93511cc90eb26777b1ff2cab0235ce812cff3661bad168632fce9ce711` |
| Ladder report SHA-256 | `2f95c0e4651a2a48449797f4a34aed557ce70962c8e82dced0d40b4aa95a0cf6` |

Inference reuses training normalization. The training normalization SHA-256 is
`757559a3e42ecab03b244ffc8202092948cc4b65547799e491c00b68d10f99a1`;
the inference normalization SHA-256 is
`279dbef649a7ddb2ca329c3977b652a541b8d0cd5cf6248212db7310953cfc36`.

## Historical Preservation

The prior FAIL and diagnostic artifacts were not rewritten. Direct SHA-256
verification reproduced every recorded seal:

- old rung1a FAIL:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1a_mmap_full_scanset/rung_evidence.json`,
  `f7c3eba38f0a93529cc37315d1d3e42f43f1dd5b1e5b239300d68cdd71f9c132`;
- old FAIL report:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/ladder_report.json`,
  `130096cf45fb9e193308f272c84b0179f1948d2c0abacf700634dfec762303c7`;
- rung1c normalization:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1c_normalization_regime/rung_evidence.json`,
  `b9886b498880c35d4ef5e1a7c18b8c229e41704fd407879d431e2226e65940da`;
- rung1d sampler:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1d_sampler_shuffle/rung_evidence.json`,
  `6df72e84ece6203f8c76326b635dd4835abb59cb9d00757cd3de6d75cd47fcad`;
- rung1e sampler plus unit normalization:
  `.artifacts/bridge_ladder/task28_gain16_seed3/convergence/rung1e_sampler_plus_unit_norm/rung_evidence.json`,
  `92f61a63870ea59993938e206aec52053f901907be7b23d6f9de4b76018cb897`;
- rung1f historical probe layout:
  `.artifacts/bridge_ladder/seed3_split/rung1f_probe_layout/rung_evidence.json`,
  `230b35b9511483e6e409ab5a3e611e925e7ba09fd22bf10c8b3efbbdb2aae324`.

Rungs 1c-1f are already archived from the current TOML. The remaining
rung1b-through-rung8 scaffold and historical injection/parser support are
retained until Task 29 producer retirement. This is a conservative retention
decision, not remaining Task 28 work, and no tombstone is authorized here.

## Next Dependency And Concerns

Task 22 is pending-unblocked and is the next executable task; it has not run.
Tasks 23 and 24 remain dependency-pending on Tasks 22 and 23 respectively, not
blocked by Task 28. Task 29 still executes after Task 24 and owns producer
retirement plus any associated historical-support tombstone decision.

No scientific concern remains open for Task 28. The remaining uncertainty is
downstream: Task 22 must establish the convergence budget and re-diagnose CNN
saturation before Task 23 can execute the corrected multi-seed matrix.
