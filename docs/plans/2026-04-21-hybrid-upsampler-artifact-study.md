# Hybrid Upsampler Artifact Study Implementation Plan

## Revision Note

This plan originally targeted the pre-skip-add `hybrid_resnet_base` shell and that run was completed under:

- `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-10ep`

That historical run is still useful as pre-skip-add evidence, but it no longer answers the current question. The canonical CNS Hybrid row is now `hybrid_resnet_cns`, which enables skip-add. All steps below are therefore updated to rerun the upsampler study under the post-skip-add canonical CNS shell so the only changed variable is the upsampler.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the bounded CNS upsampler artifact study under the canonical post-skip-add shell, compare two decoder variants against `hybrid_resnet_cns`, and generate visual comparison PNGs that show whether any remaining checkerboard/grid texture is reduced without giving up the skip-add gains.

**Architecture:** Keep the current `CycleGanUpsampler` baseline intact and add study-local opt-in decoder variants inside the PDEBench image-suite model adapter rather than changing the global generator contract first. Compare three Hybrid profiles on the same capped CNS slice and fixed seed: canonical `hybrid_resnet_cns` with transpose upsampling, `hybrid_resnet_cns_interp_bilinear_conv`, and `hybrid_resnet_cns_pixelshuffle`. The fairness boundary is strict: same `hybrid_resnet_cns` shell, same skip-add policy, same encoder/downsample path, same bottleneck, same output head; only `hybrid_upsampler` changes. Then render side-by-side prediction and absolute-error galleries from the saved NPZ artifacts. No worktree is used because repo policy explicitly forbids worktrees here.

**Tech Stack:** Python, PyTorch, pytest, matplotlib, tmux, existing PDEBench CNS study harness

---

## File Structure

- Modify: `scripts/studies/pdebench_image128/models.py`
  - Add study-local upsampler modules and thread an explicit Hybrid upsampler choice through `HybridResnetImageModel`.
- Modify: `scripts/studies/pdebench_image128/run_config.py`
  - Register manual opt-in Hybrid profiles for the post-skip-add upsampler study.
- Create: `scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py`
  - Build compact PNG galleries from saved comparison NPZs.
- Modify: `tests/studies/test_pdebench_image128_models.py`
  - Cover construction/shape preservation for the new Hybrid variants.
- Modify: `tests/studies/test_pdebench_image128_runner.py`
  - Cover profile resolution and artifact emission for the study profiles if the runner surface changes.
- Create: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/`
  - Store run outputs and rendered galleries outside git.

### Task 1: Add Failing Model-Variant Tests

**Files:**
- Modify: `tests/studies/test_pdebench_image128_models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Modify: `scripts/studies/pdebench_image128/models.py`

- [ ] **Step 1: Write the failing tests for the study profiles**

```python
def test_hybrid_upsampler_study_profiles_build_and_preserve_shape():
    from scripts.studies.pdebench_image128.models import build_model_from_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    for profile_id in [
        "hybrid_resnet_cns",
        "hybrid_resnet_cns_interp_bilinear_conv",
        "hybrid_resnet_cns_pixelshuffle",
    ]:
        model = build_model_from_profile(
            get_model_profile(profile_id),
            in_channels=8,
            out_channels=4,
            spatial_shape=(128, 128),
        )
        y = model(torch.zeros(1, 8, 128, 128))
        assert tuple(y.shape) == (1, 4, 128, 128)
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `pytest tests/studies/test_pdebench_image128_models.py -k upsampler_study_profiles -v`

Expected: FAIL because `run_config.py` does not yet define the post-skip-add profile IDs and the Hybrid builder cannot yet express "same `hybrid_resnet_cns` shell, different upsampler only".

- [ ] **Step 3: Add a second failing test for profile config differences**

```python
def test_hybrid_upsampler_study_profiles_only_change_decoder_choice():
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    base = get_model_profile("hybrid_resnet_cns").to_model_config()
    interp = get_model_profile("hybrid_resnet_cns_interp_bilinear_conv").to_model_config()
    pixel = get_model_profile("hybrid_resnet_cns_pixelshuffle").to_model_config()

    assert interp["base_model"] == "hybrid_resnet"
    assert pixel["base_model"] == "hybrid_resnet"
    assert interp["hybrid_skip_connections"] is True
    assert pixel["hybrid_skip_connections"] is True
    assert interp["hybrid_skip_style"] == "add"
    assert pixel["hybrid_skip_style"] == "add"
    assert interp["hybrid_upsampler"] == "interp_bilinear_conv"
    assert pixel["hybrid_upsampler"] == "pixelshuffle"
```

- [ ] **Step 4: Run the focused test to verify it fails for the expected reason**

Run: `pytest tests/studies/test_pdebench_image128_models.py -k 'decoder_choice or upsampler_study_profiles' -v`

Expected: FAIL because the new post-skip-add profile IDs do not exist yet.

- [ ] **Step 5: Commit the red phase**

```bash
git add tests/studies/test_pdebench_image128_models.py
git commit -m "test: define hybrid upsampler study profile expectations"
```

### Task 2: Implement Study-Local Upsampler Variants

**Files:**
- Modify: `scripts/studies/pdebench_image128/models.py`
- Modify: `scripts/studies/pdebench_image128/run_config.py`
- Test: `tests/studies/test_pdebench_image128_models.py`

- [ ] **Step 1: Add a bilinear-or-nearest interpolate-plus-conv block**

Implement a small local block in `models.py`:

```python
class InterpConvUpsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mode: str = "bilinear"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in {"bilinear", "bicubic"}:
            x = F.interpolate(x, scale_factor=2, mode=self.mode, align_corners=False)
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        return self.proj(x)
```

- [ ] **Step 2: Add a pixel-shuffle upsampler block**

```python
class PixelShuffleUpsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
```

- [ ] **Step 3: Thread an explicit `hybrid_upsampler` selection through the Hybrid adapter**

Update the Hybrid constructor and builder path so it can select:

```python
def _make_hybrid_upsampler(kind: str, in_channels: int, out_channels: int) -> nn.Module:
    if kind == "cyclegan_transpose":
        return CycleGanUpsampler(in_channels, out_channels)
    if kind == "interp_bilinear_conv":
        return InterpConvUpsampler(in_channels, out_channels, mode="bilinear")
    if kind == "interp_nearest_conv":
        return InterpConvUpsampler(in_channels, out_channels, mode="nearest")
    if kind == "pixelshuffle":
        return PixelShuffleUpsampler(in_channels, out_channels)
    raise ValueError(f"unknown hybrid upsampler: {kind}")
```

- [ ] **Step 4: Register the study profiles in `run_config.py`**

Add manual opt-in profiles:

```python
"hybrid_resnet_cns_interp_bilinear_conv": ModelProfile(
    profile_id="hybrid_resnet_cns_interp_bilinear_conv",
    base_model="hybrid_resnet",
    hidden_channels=32,
    fno_modes=12,
    fno_blocks=4,
    hybrid_downsample_steps=2,
    hybrid_resnet_blocks=6,
    hybrid_skip_connections=True,
    hybrid_skip_style="add",
    hybrid_upsampler="interp_bilinear_conv",
    evidence_scope="readiness-only",
),
"hybrid_resnet_cns_pixelshuffle": ModelProfile(
    profile_id="hybrid_resnet_cns_pixelshuffle",
    base_model="hybrid_resnet",
    hidden_channels=32,
    fno_modes=12,
    fno_blocks=4,
    hybrid_downsample_steps=2,
    hybrid_resnet_blocks=6,
    hybrid_skip_connections=True,
    hybrid_skip_style="add",
    hybrid_upsampler="pixelshuffle",
    evidence_scope="readiness-only",
),
```

Keep `hybrid_resnet_cns` unchanged and make the new profiles manual `--profiles` choices only. They must not enter `PRIMARY_CFD_CNS_PROFILE_IDS` or `READINESS_CFD_CNS_PROFILE_IDS`.

- [ ] **Step 5: Run the model tests to verify they pass**

Run: `pytest tests/studies/test_pdebench_image128_models.py -k 'upsampler_study_profiles or decoder_choice' -v`

Expected: PASS

- [ ] **Step 6: Run the broader image128 unit slice**

Run: `pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py`

Expected: PASS with no regressions to existing baseline profiles.

- [ ] **Step 7: Commit the implementation**

```bash
git add scripts/studies/pdebench_image128/models.py scripts/studies/pdebench_image128/run_config.py tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
git commit -m "feat: add hybrid upsampler study variants"
```

### Task 3: Add a Gallery Renderer for Visual Artifact Inspection

**Files:**
- Create: `scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py`
- Test: manual CLI validation against saved NPZs

- [ ] **Step 1: Write the rendering script**

The script should accept:

```text
--run-root
--sample-index
--baseline-profile
--variant-profiles
--output-png
--output-error-png
```

It should:
- load `comparison_<profile>_sample<idx>.npz`
- assert all targets match
- render one PNG for `ground truth + predictions`
- render one PNG for `absolute error`
- use per-field shared scales across columns

- [ ] **Step 2: Dry-run the renderer against the existing Hybrid baseline artifact**

Run:

```bash
python scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py \
  --run-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-2d-cfd-cns/readiness-cap-20260421T185926Z-40ep-mse \
  --sample-index 0 \
  --baseline-profile hybrid_resnet_base \
  --variant-profiles hybrid_resnet_base \
  --output-png tmp/hybrid_gallery_smoke.png \
  --output-error-png tmp/hybrid_error_gallery_smoke.png
```

Expected: PASS and both PNGs appear, even in the one-profile smoke case.

- [ ] **Step 3: Commit the renderer**

```bash
git add scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py
git commit -m "feat: add hybrid upsampler artifact gallery renderer"
```

### Task 4: Run the Bounded Artifact Study

**Files:**
- Output only: `.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/`
- Reuse: `scripts/studies/run_pdebench_image128_suite.py`
- Reuse: `scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py`

- [ ] **Step 1: Define the bounded study contract**

Use the same CNS slice and seed as the current capped comparison for comparability:
- task: `2d_cfd_cns`
- mode: `readiness`
- train/val/test trajectories: `512 / 64 / 64`
- `max_windows_per_trajectory=8`
- loss: current CNS `mse`
- batch size: `4`
- device: `cuda`

To keep this a simple study, run:
- baseline `hybrid_resnet_cns`
- variant `hybrid_resnet_cns_interp_bilinear_conv`
- variant `hybrid_resnet_cns_pixelshuffle`

Prefer `10` epochs first. Only extend to `20` or `40` if the visual result is ambiguous.

- [ ] **Step 2: Launch the study run in tmux**

Run:

```bash
python scripts/studies/run_pdebench_image128_suite.py \
  --task 2d_cfd_cns \
  --mode readiness \
  --data-root /home/ollie/Documents/pdebench-data \
  --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep \
  --profiles hybrid_resnet_cns,hybrid_resnet_cns_interp_bilinear_conv,hybrid_resnet_cns_pixelshuffle \
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

Expected: all three profiles complete with `comparison_*.npz` outputs.

- [ ] **Step 3: Render the comparison galleries**

Run:

```bash
python scripts/studies/pdebench_image128/render_hybrid_upsampler_gallery.py \
  --run-root .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep \
  --sample-index 0 \
  --baseline-profile hybrid_resnet_cns \
  --variant-profiles hybrid_resnet_cns_interp_bilinear_conv,hybrid_resnet_cns_pixelshuffle \
  --output-png .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep/gallery_sample0.png \
  --output-error-png .artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-hybrid-upsampler-artifact-study/cns-upsampler-cns-shell-10ep/gallery_sample0_error.png
```

Expected: one prediction gallery and one error gallery suitable for quick visual inspection.

- [ ] **Step 4: Record the visual decision**

Inspect:
- whether the checkerboard/grid texture is visibly reduced in the bilinear-conv variant
- whether pixel shuffle removes the texture without oversmoothing
- whether either variant regresses coarse structures or shock edges

Write a short markdown note:

`docs/plans/2026-04-21-hybrid-upsampler-artifact-study-results.md`

Include:
- run root
- profile list
- explicit note that this is the post-skip-add `hybrid_resnet_cns` shell rerun
- sample PNG paths
- one-paragraph visual conclusion
- whether to promote any variant beyond study-only status

- [ ] **Step 5: Commit the study note**

```bash
git add docs/plans/2026-04-21-hybrid-upsampler-artifact-study-results.md
git commit -m "docs: record hybrid upsampler artifact study result"
```

### Task 5: Verification Before Any Promotion

**Files:**
- Reuse prior outputs only

- [ ] **Step 1: Run the final verification slice**

Run:

```bash
pytest -q tests/studies/test_pdebench_image128_models.py tests/studies/test_pdebench_image128_runner.py
python -m compileall -q scripts/studies/pdebench_image128
```

Expected: PASS

- [ ] **Step 2: Confirm artifact completeness**

Check that the study run root contains:
- `comparison_hybrid_resnet_cns_sample0.npz`
- `comparison_hybrid_resnet_cns_interp_bilinear_conv_sample0.npz`
- `comparison_hybrid_resnet_cns_pixelshuffle_sample0.npz`
- `gallery_sample0.png`
- `gallery_sample0_error.png`

- [ ] **Step 3: Decide next action**

If one variant clearly reduces the checkerboard/grid texture without giving back the skip-add gains, promote it into the next bounded CNS run. Otherwise keep the change study-local and stop here.
