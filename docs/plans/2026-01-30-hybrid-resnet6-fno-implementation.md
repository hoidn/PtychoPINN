# Hybrid ResNet‑6 + FNO Vanilla Generators Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `fno_vanilla` and `hybrid_resnet` generator backbones, wire them into the registry + grid‑lines workflows, and validate with targeted tests and a one‑epoch grid‑lines run.

**Architecture:** Constant‑resolution FNO baseline; FNO encoder → ResNet‑6 bottleneck → CycleGAN upsamplers decoder; optional 1×1 adapter between encoder output and ResNet bottleneck when channel widths differ.

**Tech Stack:** PyTorch, Lightning, existing generator registry, pytest.

---

### Task 0: Create an isolated worktree for implementation

**Files:** none

**Step 1: Create worktree**

```bash
git worktree add -b feat/hybrid-resnet6-fno-v2 ../PtychoPINN-hybrid-resnet6
```

**Step 2: Enter worktree**

```bash
cd ../PtychoPINN-hybrid-resnet6
```

**Step 3: Confirm clean state**

```bash
git status -sb
```
Expected: clean or only intentional local changes.

**Step 4: Commit**

```bash
git commit --allow-empty -m "chore: start hybrid_resnet6 + fno_vanilla work"
```

---

### Task 1: Add failing tests for `fno_vanilla` forward contract

**Files:**
- Modify: `tests/torch/test_fno_generators.py`

**Step 1: Write the failing test**

```python
# add near the existing generator tests

def test_fno_vanilla_generator_output_shape_real_imag():
    module = FnoVanillaGeneratorModule(
        in_channels=1,
        out_channels=2,
        hidden_channels=16,
        n_blocks=2,
        modes=4,
        C=4,
        input_transform="none",
        output_mode="real_imag",
    )
    x = torch.randn(2, 4, 32, 32)
    y = module(x)
    assert y.shape == (2, 32, 32, 4, 2)
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_fno_generators.py::test_fno_vanilla_generator_output_shape_real_imag -v`

Expected: FAIL (ImportError or NameError for `FnoVanillaGeneratorModule`).

**Step 3: Write minimal implementation**

Create `ptycho_torch/generators/fno_vanilla.py`:

```python
import math
import torch
import torch.nn as nn
from typing import Optional
from ptycho_torch.generators.fno import SpatialLifter, PtychoBlock

class FnoVanillaGeneratorModule(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 hidden_channels: int = 32,
                 n_blocks: int = 4,
                 modes: int = 12,
                 C: int = 4,
                 input_transform: str = "none",
                 output_mode: str = "real_imag"):
        super().__init__()
        self.C = C
        self.output_mode = output_mode
        self.lifter = SpatialLifter(in_channels * C, hidden_channels,
                                    input_transform=input_transform)
        self.blocks = nn.ModuleList([
            PtychoBlock(hidden_channels, modes=modes)
            for _ in range(n_blocks)
        ])
        if output_mode == "amp_phase":
            self.output_amp = nn.Conv2d(hidden_channels, C, kernel_size=1)
            self.output_phase = nn.Conv2d(hidden_channels, C, kernel_size=1)
        else:
            self.output_proj = nn.Conv2d(hidden_channels, out_channels * C, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.lifter(x)
        for block in self.blocks:
            x = block(x)
        if self.output_mode == "amp_phase":
            amp = torch.sigmoid(self.output_amp(x))
            phase = math.pi * torch.tanh(self.output_phase(x))
            return amp, phase
        x = self.output_proj(x)
        x = x.view(B, 2, self.C, H, W).permute(0, 3, 4, 2, 1)
        return x
```

Also add generator wrapper:

```python
class FnoVanillaGenerator:
    name = "fno_vanilla"
    def __init__(self, config):
        self.config = config
    def build_model(self, pt_configs):
        from ptycho_torch.model import PtychoPINN_Lightning
        data_config = pt_configs["data_config"]
        model_config = pt_configs["model_config"]
        training_config = pt_configs["training_config"]
        inference_config = pt_configs["inference_config"]
        C = getattr(data_config, "C", 4)
        fno_width = getattr(model_config, "fno_width", 32)
        fno_blocks = getattr(model_config, "fno_blocks", 4)
        fno_modes = getattr(model_config, "fno_modes", 12)
        input_transform = getattr(model_config, "fno_input_transform", "none")
        output_mode = getattr(model_config, "generator_output_mode", "real_imag")
        generator_mode = "amp_phase" if output_mode == "amp_phase" else "real_imag"
        core = FnoVanillaGeneratorModule(
            in_channels=1,
            out_channels=2,
            hidden_channels=fno_width,
            n_blocks=fno_blocks,
            modes=fno_modes,
            C=C,
            input_transform=input_transform,
            output_mode=generator_mode,
        )
        return PtychoPINN_Lightning(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
            generator_module=core,
            generator_output=output_mode,
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_fno_generators.py::test_fno_vanilla_generator_output_shape_real_imag -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/generators/fno_vanilla.py tests/torch/test_fno_generators.py
git commit -m "test+feat: add fno_vanilla generator module"
```

---

### Task 2: Add failing tests for `hybrid_resnet` forward + amp/phase bounds

**Files:**
- Modify: `tests/torch/test_fno_generators.py`

**Step 1: Write failing tests**

```python

def test_hybrid_resnet_output_shape_real_imag():
    module = HybridResnetGeneratorModule(
        in_channels=1,
        out_channels=2,
        hidden_channels=16,
        n_blocks=3,
        modes=4,
        C=4,
        input_transform="none",
        output_mode="real_imag",
    )
    x = torch.randn(2, 4, 32, 32)
    y = module(x)
    assert y.shape == (2, 32, 32, 4, 2)


def test_hybrid_resnet_amp_phase_bounds():
    module = HybridResnetGeneratorModule(
        in_channels=1,
        out_channels=2,
        hidden_channels=16,
        n_blocks=3,
        modes=4,
        C=4,
        input_transform="none",
        output_mode="amp_phase",
    )
    x = torch.randn(1, 4, 32, 32)
    amp, phase = module(x)
    assert amp.min() >= 0.0
    assert amp.max() <= 1.0
    assert phase.min() >= -math.pi - 1e-3
    assert phase.max() <= math.pi + 1e-3
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/torch/test_fno_generators.py::test_hybrid_resnet_output_shape_real_imag -v`

Expected: FAIL (ImportError or NameError for `HybridResnetGeneratorModule`).

**Step 3: Write minimal implementation**

Create `ptycho_torch/generators/resnet_components.py`:

```python
import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True, eps=1e-5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetBottleneck(nn.Module):
    def __init__(self, channels: int, n_blocks: int = 6):
        super().__init__()
        self.blocks = nn.Sequential(*[ResnetBlock(channels) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CycleGanUpsampler(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels,
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(out_channels, affine=True, eps=1e-5),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
```

Create `ptycho_torch/generators/hybrid_resnet.py`:

```python
import math
import torch
import torch.nn as nn
from typing import Optional
from ptycho_torch.generators.fno import SpatialLifter, PtychoBlock
from ptycho_torch.generators.resnet_components import ResnetBottleneck, CycleGanUpsampler

class HybridResnetGeneratorModule(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 2,
                 hidden_channels: int = 32,
                 n_blocks: int = 4,
                 modes: int = 12,
                 C: int = 4,
                 input_transform: str = "none",
                 output_mode: str = "real_imag",
                 max_hidden_channels: Optional[int] = None,
                 resnet_blocks: int = 6,
                 resnet_width: Optional[int] = None):
        super().__init__()
        self.C = C
        self.output_mode = output_mode
        self.lifter = SpatialLifter(in_channels * C, hidden_channels,
                                    input_transform=input_transform)
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        ch = hidden_channels
        for i in range(n_blocks):
            self.encoder_blocks.append(PtychoBlock(ch, modes=modes))
            if i < n_blocks - 1:
                next_ch = ch * 2
                if max_hidden_channels is not None:
                    next_ch = min(next_ch, max_hidden_channels)
                self.downsample.append(nn.Conv2d(ch, next_ch, kernel_size=2, stride=2))
                ch = next_ch
        target_width = ch if resnet_width is None else resnet_width
        # 1x1 adapter only if needed
        self.adapter = nn.Identity()
        if ch != target_width:
            self.adapter = nn.Conv2d(ch, target_width, kernel_size=1)
        self.resnet = ResnetBottleneck(target_width, n_blocks=resnet_blocks)
        # CycleGAN upsamplers (2x) to return to input resolution
        self.up1 = CycleGanUpsampler(target_width, target_width // 2)
        self.up2 = CycleGanUpsampler(target_width // 2, target_width // 4)
        out_ch = target_width // 4
        if self.output_mode == "amp_phase":
            self.output_amp = nn.Conv2d(out_ch, C, kernel_size=1)
            self.output_phase = nn.Conv2d(out_ch, C, kernel_size=1)
        else:
            self.output_proj = nn.Conv2d(out_ch, out_channels * C, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x = self.lifter(x)
        for i, block in enumerate(self.encoder_blocks):
            x = block(x)
            if i < len(self.downsample):
                x = self.downsample[i](x)
        x = self.adapter(x)
        x = self.resnet(x)
        x = self.up1(x)
        x = self.up2(x)
        if self.output_mode == "amp_phase":
            amp = torch.sigmoid(self.output_amp(x))
            phase = math.pi * torch.tanh(self.output_phase(x))
            return amp, phase
        x = self.output_proj(x)
        x = x.view(B, 2, self.C, H, W).permute(0, 3, 4, 2, 1)
        return x
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/torch/test_fno_generators.py::test_hybrid_resnet_output_shape_real_imag -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/generators/resnet_components.py ptycho_torch/generators/hybrid_resnet.py tests/torch/test_fno_generators.py
git commit -m "test+feat: add hybrid_resnet generator core"
```

---

### Task 3: Add registry + config support for new architectures

**Files:**
- Modify: `ptycho_torch/generators/registry.py`
- Modify: `ptycho/config/config.py`
- Modify: `ptycho_torch/config_params.py`
- Modify: `tests/torch/test_generator_registry.py`

**Step 1: Write failing tests**

```python

def test_resolve_generator_fno_vanilla():
    cfg = TrainingConfig(model=ModelConfig(architecture="fno_vanilla", N=64, gridsize=1))
    gen = resolve_generator(cfg)
    assert gen.name == "fno_vanilla"


def test_resolve_generator_hybrid_resnet():
    cfg = TrainingConfig(model=ModelConfig(architecture="hybrid_resnet", N=64, gridsize=1))
    gen = resolve_generator(cfg)
    assert gen.name == "hybrid_resnet"
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/torch/test_generator_registry.py::test_resolve_generator_fno_vanilla -v`

Expected: FAIL (unknown architecture).

**Step 3: Write minimal implementation**

Update `ptycho_torch/generators/registry.py`:

```python
from ptycho_torch.generators.fno_vanilla import FnoVanillaGenerator
from ptycho_torch.generators.hybrid_resnet import HybridResnetGenerator

_REGISTRY = {
    "cnn": CnnGenerator,
    "fno": FnoGenerator,
    "hybrid": HybridGenerator,
    "stable_hybrid": StableHybridGenerator,
    "fno_vanilla": FnoVanillaGenerator,
    "hybrid_resnet": HybridResnetGenerator,
}
```

Update `ptycho/config/config.py`:

```python
architecture: Literal[
    "cnn", "fno", "hybrid", "stable_hybrid", "fno_vanilla", "hybrid_resnet"
] = "cnn"

valid_arches = {"cnn", "fno", "hybrid", "stable_hybrid", "fno_vanilla", "hybrid_resnet"}
```

Update `ptycho_torch/config_params.py`:

```python
architecture: Literal[
    "cnn", "fno", "hybrid", "stable_hybrid", "fno_vanilla", "hybrid_resnet"
] = "cnn"
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/torch/test_generator_registry.py -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add ptycho_torch/generators/registry.py ptycho/config/config.py ptycho_torch/config_params.py tests/torch/test_generator_registry.py
git commit -m "feat: register fno_vanilla and hybrid_resnet architectures"
```

---

### Task 4: Update grid‑lines Torch runner to accept new architectures

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Write failing test**

```python

def test_runner_accepts_hybrid_resnet(tmp_path):
    cfg = TorchRunnerConfig(
        train_npz=tmp_path / "train.npz",
        test_npz=tmp_path / "test.npz",
        output_dir=tmp_path,
        architecture="hybrid_resnet",
    )
    _, training_config, _ = setup_torch_configs(cfg)
    assert training_config.model.architecture == "hybrid_resnet"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_accepts_hybrid_resnet -v`

Expected: FAIL (architecture not in Literal/choices).

**Step 3: Write minimal implementation**

Update `scripts/studies/grid_lines_torch_runner.py`:

```python
arch_literal = cast(Literal[
    "cnn", "fno", "hybrid", "stable_hybrid", "fno_vanilla", "hybrid_resnet"
], cfg.architecture)
```

Update CLI choices:

```python
parser.add_argument("--architecture", type=str, required=True,
                    choices=["fno", "hybrid", "stable_hybrid", "fno_vanilla", "hybrid_resnet"],
                    help="Generator architecture to use")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::test_runner_accepts_hybrid_resnet -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py
git commit -m "feat: allow hybrid_resnet/fno_vanilla in torch grid-lines runner"
```

---

### Task 5: Update grid‑lines compare wrapper to route new architectures

**Files:**
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`

**Step 1: Write failing tests**

```python

def test_wrapper_handles_hybrid_resnet(monkeypatch, tmp_path):
    args = parse_args(["--architectures", "hybrid_resnet"])
    assert "hybrid_resnet" in args.architectures
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_hybrid_resnet -v`

Expected: FAIL (unknown architecture or ordering mismatch).

**Step 3: Write minimal implementation**

Update default architectures and order mapping in `scripts/studies/grid_lines_compare_wrapper.py`:

```python
parser.add_argument("--architectures", type=str,
                    default="cnn,baseline,fno,hybrid,stable_hybrid,fno_vanilla,hybrid_resnet")

if "fno_vanilla" in architectures:
    order.append("pinn_fno_vanilla")
if "hybrid_resnet" in architectures:
    order.append("pinn_hybrid_resnet")
```

Ensure the torch runner dispatch path allows both new architectures:

```python
if arch in ("fno", "hybrid", "stable_hybrid", "fno_vanilla", "hybrid_resnet"):
    run_torch_runner(...)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_hybrid_resnet -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/studies/grid_lines_compare_wrapper.py tests/test_grid_lines_compare_wrapper.py
git commit -m "feat: wire hybrid_resnet/fno_vanilla into compare wrapper"
```

---

### Task 6: Update docs and spec bridge

**Files:**
- Modify: `docs/CONFIGURATION.md`
- Modify: `docs/architecture_torch.md`
- Modify: `docs/workflows/pytorch.md`
- Modify: `ptycho_torch/generators/README.md`
- Modify: `docs/specs/spec-ptycho-config-bridge.md`
- Modify: `docs/development/TEST_SUITE_INDEX.md`

**Step 1: Update documentation**

Add `fno_vanilla` and `hybrid_resnet` to all architecture lists/tables and mention the CycleGAN upsampler parameters and output‑mode defaults.

**Step 2: Update test index**

Run (from repo root):

```bash
pytest tests/torch/test_fno_generators.py --collect-only -q > tmp/pytest_collect_fno_generators.txt
```

Update `docs/development/TEST_SUITE_INDEX.md` entries for any new tests and archive the collect-only log under the active plan artifact location.

**Step 3: Commit**

```bash
git add docs/CONFIGURATION.md docs/architecture_torch.md docs/workflows/pytorch.md ptycho_torch/generators/README.md docs/specs/spec-ptycho-config-bridge.md docs/development/TEST_SUITE_INDEX.md

git commit -m "docs: document hybrid_resnet and fno_vanilla architectures"
```

---

### Task 7: Verification and proof‑of‑life

**Files:** none

**Step 1: Run targeted tests**

```bash
pytest tests/torch/test_fno_generators.py -v
pytest tests/torch/test_generator_registry.py -v
pytest tests/torch/test_grid_lines_torch_runner.py -v
pytest tests/test_grid_lines_compare_wrapper.py -v
```

Expected: PASS.

**Step 2: Run grid‑lines proof (single epoch)**

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --train-npz outputs/grid_lines_multi_N128/datasets/N128/gs1/train.npz \
  --test-npz outputs/grid_lines_multi_N128/datasets/N128/gs1/test.npz \
  --architectures fno_vanilla,hybrid_resnet \
  --epochs 1 \
  --output-dir outputs/grid_lines_multi_N128
```

Expected: both architectures complete 1 epoch, produce `runs/pinn_fno_vanilla/` and `runs/pinn_hybrid_resnet/`, and emit recon/metrics artifacts.

**Step 3: Commit verification logs**

```bash
# Add logs or paths as required by TESTING_GUIDE
```

---

## Execution Handoff

Plan complete and saved to `docs/plans/2026-01-30-hybrid-resnet6-fno-implementation.md`.

Two execution options:
1. **Subagent‑Driven (this session)** — Use superpowers:subagent-driven-development
2. **Parallel Session (separate)** — Open a new session and use superpowers:executing-plans

Which approach?
