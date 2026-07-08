# FNO Vanilla NeuralOperator Alignment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace `fno_vanilla` with a NeuralOperator-style FNO baseline (pointwise lift, spectral+1×1 blocks, optional coordinate grid + padding) while preserving the existing output contract.

**Architecture:** Rebuild `FnoVanillaGeneratorModule` as a standard FNO stack: 1×1 lift → repeated (SpectralConv + 1×1 conv + GELU) → 1×1 projection. Add coordinate-grid concatenation and spatial padding (cropped back to original size). Keep outputs `(B, H, W, C, 2)` or `(amp, phase)` unchanged.

**Tech Stack:** PyTorch, neuralop (required), pytest.

---

### Task 1: Add FNO Vanilla “standard” contract tests (RED)

**Files:**
- Modify: `tests/torch/test_fno_generators.py`

**Step 1: Write the failing tests**

Add to `TestFnoVanillaGenerator`:

```python
    def test_vanilla_lift_is_pointwise(self):
        model = FnoVanillaGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=2,
            modes=8,
            C=4,
            use_grid=False,
            padding=0,
        )
        assert model.lift.kernel_size == (1, 1)
        for block in model.blocks:
            assert block.pointwise.kernel_size == (1, 1)

    def test_vanilla_grid_adds_two_channels(self):
        model = FnoVanillaGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=2,
            modes=8,
            C=4,
            use_grid=True,
            padding=0,
        )
        assert model.lift.in_channels == 6  # C=4 + 2 grid channels

    def test_vanilla_padding_preserves_shape(self):
        model = FnoVanillaGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=16,
            n_blocks=2,
            modes=8,
            C=4,
            use_grid=False,
            padding=8,
        )
        x = torch.randn(2, 4, 32, 32)
        out = model(x)
        assert out.shape == (2, 32, 32, 4, 2)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/torch/test_fno_generators.py::TestFnoVanillaGenerator -v`

Expected: FAIL (missing `use_grid`/`padding`, no `lift`/`block.pointwise` attributes).

**Step 3: Commit the failing tests**

```bash
git add tests/torch/test_fno_generators.py
git commit -m "test: define standard FNO vanilla contract"
```

---

### Task 2: Implement standard FNO vanilla module (GREEN)

**Files:**
- Modify: `ptycho_torch/generators/fno_vanilla.py`

**Step 1: Replace the module with a standard FNO stack**

Implement the following structure (replace `SpatialLifter`/`PtychoBlock` usage):

```python
import torch.nn.functional as F
from neuralop.layers.spectral_convolution import SpectralConv


class VanillaFnoBlock(nn.Module):
    def __init__(self, channels: int, modes: int = 12):
        super().__init__()
        self.spectral = SpectralConv(channels, channels, n_modes=(modes, modes))
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral(x) + self.pointwise(x)
        return self.act(x)
```

Then update `FnoVanillaGeneratorModule`:

```python
class FnoVanillaGeneratorModule(nn.Module):
    def __init__(..., use_grid: bool = True, padding: int = 8, ...):
        self.use_grid = use_grid
        self.padding = padding
        self.lift = nn.Conv2d(in_channels * C + (2 if use_grid else 0), hidden_channels, kernel_size=1)
        self.blocks = nn.ModuleList([VanillaFnoBlock(hidden_channels, modes=modes) for _ in range(n_blocks)])
        self.proj1 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(hidden_channels, out_channels * C, kernel_size=1)
        self.act = nn.GELU()
        # amp_phase branch same as before

    def _grid(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        xs = torch.linspace(0, 1, W, device=x.device, dtype=x.dtype)
        ys = torch.linspace(0, 1, H, device=x.device, dtype=x.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        return grid.repeat(B, 1, 1, 1)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        if self.use_grid:
            x = torch.cat([x, self._grid(x)], dim=1)
        if self.padding > 0:
            x = F.pad(x, (0, self.padding, 0, self.padding))
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        if self.output_mode == "amp_phase":
            ...
        x = self.act(self.proj1(x))
        x = self.proj2(x)
        if self.padding > 0:
            x = x[..., :H, :W]
        x = x.view(B, 2, self.C, H, W).permute(0, 3, 4, 2, 1)
        return x
```

**Step 2: Run tests to verify they pass**

Run: `pytest tests/torch/test_fno_generators.py::TestFnoVanillaGenerator -v`

Expected: PASS.

**Step 3: Commit the implementation**

```bash
git add ptycho_torch/generators/fno_vanilla.py
git commit -m "feat: align fno_vanilla with standard FNO blocks"
```

---

### Task 3: Update docs to reflect the new vanilla definition

**Files:**
- Modify: `docs/architecture_torch.md`
- Modify: `ptycho_torch/generators/README.md`
- Modify: `docs/CONFIGURATION.md`

**Step 1: Update architecture docs**

In `docs/architecture_torch.md` generator table and description:
- Describe `fno_vanilla` as NeuralOperator-style: pointwise 1×1 lift, spectral+1×1 blocks, coordinate grid, padding/crop.

**Step 2: Update generator README**

Replace the existing “FNO Vanilla” description with the new standard block/lift summary, note that it appends a normalized (x, y) grid, and that `neuralop` is required.

**Step 3: Update configuration doc description**

In `docs/CONFIGURATION.md` architecture entry, add a clause: “NeuralOperator-style constant‑resolution FNO (pointwise lift, spectral+1×1 blocks, grid coords, padding/crop).”

**Step 4: Commit docs**

```bash
git add docs/architecture_torch.md ptycho_torch/generators/README.md docs/CONFIGURATION.md
git commit -m "docs: redefine fno_vanilla as standard FNO baseline"
```

---

### Task 4: Full targeted verification

**Step 1: Run the full FNO/Hybrid generator tests**

Run: `pytest tests/torch/test_fno_generators.py -v`

Expected: PASS.

**Step 2: Optional smoke via registry**

Run: `pytest tests/torch/test_generator_registry.py::test_resolve_generator_fno_vanilla -v`

Expected: PASS.

---

## Notes / Decisions
- `fno_vanilla` now uses a standard FNO block (SpectralConv + 1×1 conv + GELU), not `PtychoBlock`.
- `neuralop` is a hard dependency for `fno_vanilla`; there is no fallback path.
- Coordinate grid is appended by default (`use_grid=True`), padding defaults to 8 and is cropped back to `(H, W)`.
- Output contracts remain unchanged.

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-02-05-fno-vanilla-neuralop-alignment.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
