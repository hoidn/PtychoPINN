# Phase 2 PDEBench Spectral Weight-Sharing CNS Compare Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a fourth PDEBench CNS bottleneck variant that is identical to the current spectral bottleneck row except that the spectral weights are not shared across bottleneck depth, then run a capped 10-epoch CNS comparison against the shared-weight spectral row.

**Architecture:** Keep the current canonical `hybrid_resnet_cns` skip-add shell fixed and reuse the existing `spectral_resnet_bottleneck_net` family. The only changed variable is `spectral_bottleneck_share_weights`: `True` for the existing shared spectral row and `False` for the new non-shared spectral row. Compare those two rows on the same capped `2d_cfd_cns` slice and render prediction/error galleries from the saved NPZ artifacts. No worktree is used because repo policy explicitly forbids worktrees here.

**Tech Stack:** Python via PATH `python`, PyTorch, existing PDEBench image-suite runner, pytest, compileall, matplotlib, tmux with `ptycho311` for the long run

---

## Initiative

- ID: `NEURIPS-HYBRID-RESNET-2026`
- Tranche ID: `phase-2-pdebench-spectral-weight-sharing-cns-compare`
- Status: pending
- Date: 2026-04-21
- Scope owner: Roadmap Phase 2 optional architecture ablation
- Related design:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_resnet_bottleneck_design.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_design.md`
- Related implementation summary:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_ffno_bottleneck_summary.md`
- Artifact root: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/`

## Compliance Matrix

- [ ] **Repo guidance:** Re-read `AGENTS.md`, `docs/index.md`, and `docs/findings.md` before edits; use PATH `python`; use tmux for the long run; do not create worktrees.
- [ ] **Fairness contract:** The new non-shared row must keep the same shell as `spectral_resnet_bottleneck_base`:
  - same `hybrid_resnet_cns` skip-add shell
  - same `hidden_channels=32`
  - same `fno_modes=12`
  - same `fno_blocks=4`
  - same `hybrid_downsample_steps=2`
  - same `hybrid_resnet_blocks=6`
  - same `spectral_bottleneck_blocks=6`
  - same `spectral_bottleneck_modes=12`
  - same `spectral_bottleneck_gate_init=0.1`
  - same `spectral_bottleneck_gate_mode="shared"`
  - same decoder / output head / loss / scheduler / split
  Only `spectral_bottleneck_share_weights` may change.
- [ ] **Naming contract:** The new row must remain in the `spectral_resnet_bottleneck_*` namespace and not be promoted into primary or readiness-required bundle lists.
- [ ] **Evidence boundary:** The 10-epoch capped CNS run is decision-support evidence only, not a benchmark claim.

## File Structure

### Modified files

- Modify: `scripts/studies/pdebench_image128/run_config.py`
  - Add a manual opt-in non-shared spectral profile
- Modify: `tests/studies/test_pdebench_image128_models.py`
  - Add profile-config and shape tests for the non-shared row
- Optional modify: `tests/studies/test_pdebench_image128_runner.py`
  - Only if the runner/reporting path needs explicit coverage for the new profile id

### New files

- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md`
  - durable summary of the ablation and capped results

### Reused files

- Reuse: `scripts/studies/pdebench_image128/models.py`
  - no code-path change expected if `share_spectral_weights=False` already works
- Reuse: `scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py`
  - reuse the existing gallery script because it already renders arbitrary profile combinations from saved comparison NPZs

## Task 1: Add Red Tests For The New Profile

**Files:**
- Modify: `tests/studies/test_pdebench_image128_models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`

- [ ] **Step 1: Add a failing profile-config test**

```python
def test_spectral_noshare_profile_only_flips_weight_sharing():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    shared = get_model_profile("spectral_resnet_bottleneck_base").to_model_config()
    noshare = get_model_profile("spectral_resnet_bottleneck_noshare").to_model_config()

    assert noshare["base_model"] == "spectral_resnet_bottleneck_net"
    assert shared["spectral_bottleneck_share_weights"] is True
    assert noshare["spectral_bottleneck_share_weights"] is False
    assert {
        key: value
        for key, value in noshare.items()
        if key not in {"profile_id", "spectral_bottleneck_share_weights", "evidence_scope"}
    } == {
        key: value
        for key, value in shared.items()
        if key not in {"profile_id", "spectral_bottleneck_share_weights", "evidence_scope"}
    }
```

- [ ] **Step 2: Add a failing shape/build test**

```python
def test_spectral_noshare_profile_builds_under_canonical_cns_shell():
    from scripts.studies.pdebench_image128.models import build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile
    import torch

    model = build_model_from_profile(
        get_model_profile("spectral_resnet_bottleneck_noshare"),
        in_channels=8,
        out_channels=4,
        spatial_shape=(128, 128),
    )
    y = model(torch.zeros(1, 8, 128, 128))
    assert tuple(y.shape) == (1, 4, 128, 128)
    assert model.module.skip_connections_enabled is True
    assert model.module.hybrid_skip_style == "add"
```

- [ ] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -k 'spectral_noshare' -v
```

Expected: FAIL because the new profile id does not exist yet.

## Task 2: Add The Manual Non-Shared Spectral Profile

**Files:**
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Test: `tests/studies/test_pdebench_image128_models.py`

- [ ] **Step 1: Add the new manual profile**

Add:

```python
"spectral_resnet_bottleneck_noshare": ModelProfile(
    profile_id="spectral_resnet_bottleneck_noshare",
    base_model="spectral_resnet_bottleneck_net",
    hidden_channels=32,
    fno_modes=12,
    fno_blocks=4,
    hybrid_downsample_steps=2,
    hybrid_resnet_blocks=6,
    hybrid_skip_connections=True,
    hybrid_skip_style="add",
    spectral_bottleneck_blocks=6,
    spectral_bottleneck_modes=12,
    spectral_bottleneck_share_weights=False,
    spectral_bottleneck_gate_init=0.1,
    spectral_bottleneck_gate_mode="shared",
    evidence_scope="readiness-only",
),
```

- [ ] **Step 2: Keep bundle lists unchanged**

Confirm the new profile does not enter:

- `PRIMARY_CFD_CNS_PROFILE_IDS`
- `READINESS_CFD_CNS_PROFILE_IDS`
- `PRIMARY_DARCY_PROFILE_IDS`

- [ ] **Step 3: Re-run the focused tests**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -k 'spectral_noshare' -v
```

Expected: PASS

- [ ] **Step 4: Re-run the broader targeted slice**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -q
python -m pytest tests/studies/test_pdebench_image128_runner.py -q
```

Expected: PASS

## Task 3: Verify The Existing Model Path Works Without Shared Spectral Weights

**Files:**
- Reuse: `scripts/studies/pdebench_image128/models.py`
- Reuse: `ptycho_torch/generators/spectral_resnet_bottleneck.py`

- [ ] **Step 1: Confirm no implementation change is required**

Inspect the current spectral bottleneck constructor and verify that:

- `share_spectral_weights=False` already instantiates one `FactorizedSpectralConv2d` per block
- the same shell and skip-add path are still used
- parameter counting still happens before first forward

- [ ] **Step 2: If the path already works, leave the implementation unchanged**

Do not edit generator/model code unless a real bug appears in the build or run.

- [ ] **Step 3: If a bug appears, add the smallest targeted fix plus regression test**

Any fix must preserve the shared profile behavior and must not change shell wiring.

## Task 4: Run Compile Verification

**Files:**
- Reuse: `scripts/studies/pdebench_image128/*`
- Reuse: `ptycho_torch/generators/spectral_resnet_bottleneck.py`

- [ ] **Step 1: Run compile verification**

Run:

```bash
python -m compileall -q scripts/studies/pdebench_image128 ptycho_torch/generators/spectral_resnet_bottleneck.py
```

Expected: PASS with no output.

## Task 5: Run The Capped 10-Epoch CNS Compare

**Files:**
- Output only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/`

- [ ] **Step 1: Define the run contract**

Use the same protocol as the recent bottleneck compare:

- task: `2d_cfd_cns`
- mode: `readiness`
- train / val / test trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- history len: `2`
- loss: `mse`
- batch size: `4`
- epochs: `10`
- device: `cuda`

Profiles:

- `spectral_resnet_bottleneck_base`
- `spectral_resnet_bottleneck_noshare`

- [ ] **Step 2: Launch the run in tmux**

Run:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep \
  --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_noshare \
  --history-len 2 \
  --epochs 10 \
  --batch-size 4 \
  --max-train-trajectories 512 \
  --max-val-trajectories 64 \
  --max-test-trajectories 64 \
  --max-windows-per-trajectory 8 \
  --device cuda \
  --num-workers 0
```

Expected: both profiles complete with `comparison_*.npz`, `metrics_*.json`, and `comparison_summary.json`.

- [ ] **Step 3: Render the comparison galleries**

Run:

```bash
python scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py \
  --run-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep \
  --sample-index 0 \
  --baseline-profile spectral_resnet_bottleneck_base \
  --variant-profiles spectral_resnet_bottleneck_noshare \
  --output-png .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/gallery_sample0.png \
  --output-error-png .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-spectral-weight-sharing-cns-compare/cns-spectral-share-vs-noshare-10ep/gallery_sample0_error.png
```

Expected: one prediction gallery and one error gallery.

- [ ] **Step 4: Copy the galleries into `tmp/` for quick inspection**

Copy:

- `gallery_sample0.png`
- `gallery_sample0_error.png`

into `tmp/` with descriptive names.

## Task 6: Write The Durable Summary

**Files:**
- Create: `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_spectral_weight_sharing_summary.md`

- [ ] **Step 1: Record implementation scope**

Document:

- new profile id
- exact fairness boundary
- whether generator/model code changed or not
- verification commands and results

- [ ] **Step 2: Record capped-run results**

Include:

- run root
- split sizes
- parameter counts
- train-loss traces
- `err_RMSE`
- `err_nRMSE`
- `relative_l2`
- `fRMSE_low/mid/high`
- gallery paths

- [ ] **Step 3: Make the claim boundary explicit**

State clearly that this is:

- a capped 10-epoch readiness compare
- decision-support evidence only
- not a benchmark-complete architecture ranking

## Task 7: Final Verification

**Files:**
- Reuse prior outputs only

- [ ] **Step 1: Re-run the final verification slice**

Run:

```bash
python -m pytest tests/studies/test_pdebench_image128_models.py -q
python -m pytest tests/studies/test_pdebench_image128_runner.py -q
python -m compileall -q scripts/studies/pdebench_image128 ptycho_torch/generators/spectral_resnet_bottleneck.py
```

Expected: PASS

- [ ] **Step 2: Confirm artifact completeness**

Check that the run root contains:

- `comparison_spectral_resnet_bottleneck_base_sample0.npz`
- `comparison_spectral_resnet_bottleneck_noshare_sample0.npz`
- `metrics_spectral_resnet_bottleneck_base.json`
- `metrics_spectral_resnet_bottleneck_noshare.json`
- `comparison_summary.json`
- `gallery_sample0.png`
- `gallery_sample0_error.png`

- [ ] **Step 3: Decision rule**

If the non-shared row clearly improves over the shared row on aggregate error without creating obvious qualitative regressions, keep it as a serious follow-up candidate. Otherwise retain the shared-weight row as the default spectral bottleneck reference and stop here.
