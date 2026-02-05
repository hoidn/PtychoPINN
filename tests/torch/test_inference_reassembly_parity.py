import numpy as np
import pytest
import torch

from types import SimpleNamespace


@pytest.mark.torch
def test_inference_helper_uses_reassembly(tmp_path):
    """
    Verifies that PyTorch inference helper stitches patches via position-aware
    reassembly rather than averaging across the batch. Uses a stub model and
    synthetic RawData with non-zero offsets so reassembly produces a different
    canvas than a naive average.
    """
    # Import here to avoid import-time torch failures breaking collection
    from ptycho.raw_data import RawData
    from ptycho_torch import helper as hh
    from ptycho_torch.inference import _run_inference_and_reconstruct
    from ptycho_torch.config_params import DataConfig, ModelConfig

    device = 'cpu'

    # Synthetic dataset: 16 patches of size 32x32 with non-zero offsets
    B, N = 16, 32
    rng = np.random.default_rng(42)
    # Create a simple coordinate grid with a spread to ensure non-zero offsets
    xcoords = rng.uniform(-50, 50, size=B).astype(np.float32)
    ycoords = rng.uniform(-40, 40, size=B).astype(np.float32)
    scan_index = np.arange(B, dtype=np.int64)
    diff3d = np.zeros((B, N, N), dtype=np.float32)  # not used by the stub
    probe = np.ones((N, N), dtype=np.complex64)

    raw = RawData(xcoords, ycoords, xcoords, ycoords, diff3d, probe, scan_index)

    # Stub model: forward_predict returns a single-pixel impulse at patch center
    class StubModel:
        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

        @torch.no_grad()
        def forward_predict(self, x, positions, probe, input_scale_factor):
            # x: (B, 1, N, N) or (B, N, N) after helper transforms
            B = x.shape[0]
            patch = torch.zeros((B, 1, N, N), dtype=torch.complex64, device=x.device)
            patch[:, 0, N // 2, N // 2] = torch.complex(torch.tensor(1.0, device=x.device), torch.tensor(0.0, device=x.device))
            return patch

    model = StubModel()

    # Minimal configs for the helper
    tf_infer_cfg = SimpleNamespace(n_groups=B, stitch_crop_size=20)  # map to number of patches taken
    exec_cfg = SimpleNamespace(accelerator='cpu')

    # Current inference helper output (was averaging pre-fix)
    amp_cli, ph_cli = _run_inference_and_reconstruct(
        model=model,
        raw_data=raw,
        config=tf_infer_cfg,
        execution_config=exec_cfg,
        device=device,
        quiet=True,
    )

    # Ground-truth reassembly using helper (position-aware)
    with torch.no_grad():
        # Prepare predicted patches and offsets
        x = torch.from_numpy(diff3d).to(device=device, dtype=torch.float32)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B,1,N,N)
        probe_t = torch.from_numpy(probe).to(device=device, dtype=torch.complex64)
        if probe_t.ndim == 2:
            probe_t = probe_t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        inputs = model.forward_predict(x, torch.zeros((B, 1, 1, 2)), probe_t, torch.ones((B, 1, 1, 1)))

        # Offsets: center to COM in pixel units, shape (B,1,1,2)
        dx = torch.from_numpy(xcoords - xcoords.mean()).to(device=device, dtype=torch.float32)
        dy = torch.from_numpy(ycoords - ycoords.mean()).to(device=device, dtype=torch.float32)
        offsets = torch.stack([dx, dy], dim=-1).view(B, 1, 1, 2)

        # Minimal configs for reassembly padding
        data_cfg = DataConfig(N=N, grid_size=(1, 1))
        model_cfg = ModelConfig()
        model_cfg.C_forward = 1
        imgs_merged, _, _ = hh.reassemble_patches_position_real(
            inputs, offsets, data_cfg, model_cfg, crop_size=tf_infer_cfg.stitch_crop_size
        )
        amp_gt = torch.abs(imgs_merged[0]).cpu().numpy()

    # Expect: helper output equals position-aware reassembly (shape + values)
    assert amp_cli.shape == amp_gt.shape, (
        f"Expected stitched canvas shape {amp_gt.shape}, got {amp_cli.shape}"
    )
    # Numeric closeness (allow tiny tolerance from dtype conversions)
    mae = np.mean(np.abs(amp_cli - amp_gt))
    assert mae < 1e-6, f"Reassembled amplitude mismatch: MAE={mae}"


@pytest.mark.torch
def test_reassembly_canvas_padding_invariants(monkeypatch):
    """
    Ensures inference reassembly enforces dynamic canvas padding and non-zero offsets.
    """
    from ptycho.raw_data import RawData
    from ptycho_torch.inference import _run_inference_and_reconstruct
    from ptycho_torch import helper as hh

    B, N = 8, 16
    xcoords = np.array([0.0, 12.0, -6.0, 4.0, -3.0, 7.0, -8.0, 1.0], dtype=np.float32)
    ycoords = np.array([5.0, -4.0, 2.0, -6.0, 9.0, -3.0, 1.0, -8.0], dtype=np.float32)
    scan_index = np.arange(B, dtype=np.int64)
    diff3d = np.zeros((B, N, N), dtype=np.float32)
    probe = np.ones((N, N), dtype=np.complex64)
    raw = RawData(xcoords, ycoords, xcoords, ycoords, diff3d, probe, scan_index)

    class StubModel:
        def to(self, *_args, **_kwargs):
            return self

        def eval(self):
            return self

        @torch.no_grad()
        def forward_predict(self, x, positions, probe, input_scale_factor):
            return torch.ones((x.shape[0], 1, x.shape[-1], x.shape[-1]), dtype=torch.complex64, device=x.device)

    model = StubModel()
    tf_infer_cfg = SimpleNamespace(n_groups=B, stitch_crop_size=20)
    exec_cfg = SimpleNamespace(accelerator='cpu')

    captured = {}

    def fake_reassemble(patches, offsets, data_cfg, model_cfg, **kwargs):
        captured['crop_size'] = kwargs.get('crop_size')
        captured['offsets'] = offsets.detach().cpu().numpy()
        padded_size = patches.shape[-1]
        canvas = torch.zeros((patches.shape[0], padded_size, padded_size), dtype=torch.complex64, device=patches.device)
        return canvas, None, None

    monkeypatch.setattr(hh, 'reassemble_patches_position_real', fake_reassemble)
    _run_inference_and_reconstruct(
        model=model,
        raw_data=raw,
        config=tf_infer_cfg,
        execution_config=exec_cfg,
        device='cpu',
        quiet=True,
    )

    assert 'crop_size' in captured, "reassemble helper was not invoked"
    assert captured['crop_size'] == min(N, tf_infer_cfg.stitch_crop_size)

    offsets = captured['offsets'].reshape(B, 2)
    assert not np.allclose(offsets, 0.0), "Offsets should not collapse to zero"
    np.testing.assert_allclose(offsets[:, 0], xcoords - xcoords.mean(), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(offsets[:, 1], ycoords - ycoords.mean(), rtol=1e-6, atol=1e-6)
