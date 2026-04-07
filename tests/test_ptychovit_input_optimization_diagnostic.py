import json
from pathlib import Path

import h5py
import numpy as np
import torch


def test_project_input_enforces_nonnegative_and_max_bound():
    from scripts.studies.ptychovit_input_optimization_diagnostic import project_input

    x = torch.tensor([[-3.0, 0.5, 2.0, 200.0]], dtype=torch.float32)
    projected = project_input(x, min_value=0.0, max_value=100.0)

    assert torch.all(projected >= 0.0)
    assert torch.all(projected <= 100.0)
    assert torch.isclose(projected[0, 1], torch.tensor(0.5))


def test_stationary_criterion_triggers_on_small_gradient_norm():
    from scripts.studies.ptychovit_input_optimization_diagnostic import is_stationary

    assert is_stationary(1.0e-7, threshold=1.0e-6)
    assert not is_stationary(1.0e-4, threshold=1.0e-6)


def test_objective_components_reported_with_expected_keys():
    from scripts.studies.ptychovit_input_optimization_diagnostic import compute_objective

    pred_amp = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    pred_phase = torch.tensor([[0.0, 0.1], [0.2, 0.3]], dtype=torch.float32)
    pred_diff_amp = torch.tensor([[1.0, 1.5], [2.0, 2.5]], dtype=torch.float32)
    target_diff_amp = torch.tensor([[1.0, 1.0], [2.0, 2.0]], dtype=torch.float32)

    objective, components = compute_objective(
        pred_amp=pred_amp,
        pred_phase=pred_phase,
        weights={
            "amp_var": 1.0,
            "phase_var": 1.0,
            "tv": 0.1,
            "forward_consistency": 1.0,
        },
        pred_diff_amp=pred_diff_amp,
        target_diff_amp=target_diff_amp,
        input_tensor=target_diff_amp,
    )

    expected_keys = {"amp_var", "phase_var", "tv", "forward_consistency", "total"}
    assert expected_keys.issubset(set(components.keys()))
    assert torch.isfinite(objective)


def test_cli_writes_json_report_with_required_fields(tmp_path: Path, monkeypatch):
    from scripts.studies import ptychovit_input_optimization_diagnostic as diag

    repo = tmp_path / "ptycho-vit"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "config.yaml").write_text("data: {}\nmodel: {}\n")

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"checkpoint")
    test_dp = tmp_path / "test_dp.hdf5"
    test_para = tmp_path / "test_para.hdf5"
    test_dp.write_bytes(b"dp")
    test_para.write_bytes(b"para")
    out_dir = tmp_path / "diag"

    def _fake_run_diagnostic(args):
        _ = args
        return {
            "objective_history": [0.1, 0.2],
            "grad_norm_history": [1.0, 0.5],
            "stationary_step": 1,
            "input_stats": [{"min": 0.0, "max": 1.0, "mean": 0.5, "std": 0.1}],
            "normalization_context": {"normalization": 123.0, "scale": 10000.0},
            "config": {"steps": 2, "lr": 1.0e-2},
        }

    monkeypatch.setattr(diag, "run_diagnostic", _fake_run_diagnostic)
    rc = diag.main(
        [
            "--output-dir",
            str(out_dir),
            "--checkpoint",
            str(checkpoint),
            "--ptychovit-repo",
            str(repo),
            "--test-dp",
            str(test_dp),
            "--test-para",
            str(test_para),
        ]
    )

    assert rc == 0
    report_path = out_dir / "diagnostic_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    required = {
        "objective_history",
        "grad_norm_history",
        "stationary_step",
        "input_stats",
        "normalization_context",
        "config",
    }
    assert required.issubset(report.keys())


def test_cli_rejects_invalid_bounds_and_weights(tmp_path: Path):
    from scripts.studies import ptychovit_input_optimization_diagnostic as diag

    repo = tmp_path / "ptycho-vit"
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "config.yaml").write_text("data: {}\nmodel: {}\n")

    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_bytes(b"checkpoint")
    test_dp = tmp_path / "test_dp.hdf5"
    test_para = tmp_path / "test_para.hdf5"
    test_dp.write_bytes(b"dp")
    test_para.write_bytes(b"para")
    out_dir = tmp_path / "diag_bad"

    try:
        diag.main(
            [
                "--output-dir",
                str(out_dir),
                "--checkpoint",
                str(checkpoint),
                "--ptychovit-repo",
                str(repo),
                "--test-dp",
                str(test_dp),
                "--test-para",
                str(test_para),
                "--input-max",
                "-1",
            ]
        )
    except ValueError as exc:
        assert "input-max" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid --input-max")

    try:
        diag.main(
            [
                "--output-dir",
                str(out_dir),
                "--checkpoint",
                str(checkpoint),
                "--ptychovit-repo",
                str(repo),
                "--test-dp",
                str(test_dp),
                "--test-para",
                str(test_para),
                "--w-amp-var",
                "-0.1",
            ]
        )
    except ValueError as exc:
        assert "w-amp-var" in str(exc)
    else:
        raise AssertionError("Expected ValueError for negative objective weight")


def _write_bridge_like_pair(dp_path: Path, para_path: Path) -> None:
    dp_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(dp_path, "w") as dp_file:
        dp = np.stack(
            [
                np.full((8, 8), 4.0, dtype=np.float32),
                np.full((8, 8), 9.0, dtype=np.float32),
            ],
            axis=0,
        )
        dp_file.create_dataset("dp", data=dp)

    with h5py.File(para_path, "w") as para_file:
        obj = para_file.create_dataset(
            "object",
            data=(np.ones((1, 32, 32), dtype=np.float32) + 1j * np.zeros((1, 32, 32), dtype=np.float32)).astype(
                np.complex64
            ),
        )
        obj.attrs["pixel_height_m"] = 1.0e-9
        obj.attrs["pixel_width_m"] = 1.0e-9
        probe = para_file.create_dataset("probe", data=np.ones((1, 1, 8, 8), dtype=np.complex64))
        probe.attrs["pixel_height_m"] = 1.0e-9
        probe.attrs["pixel_width_m"] = 1.0e-9
        para_file.create_dataset("probe_position_x_m", data=np.array([0.0, 1.0e-9], dtype=np.float64))
        para_file.create_dataset("probe_position_y_m", data=np.array([0.0, 2.0e-9], dtype=np.float64))


def _write_stub_ptychovit_repo(repo_dir: Path) -> None:
    repo_dir.mkdir(parents=True, exist_ok=True)
    (repo_dir / "model").mkdir(parents=True, exist_ok=True)
    (repo_dir / "model" / "__init__.py").write_text("")
    (repo_dir / "config.yaml").write_text(
        "data:\n"
        "  scale: 10000.0\n"
        "  cache_object: false\n"
        "  max_probe_modes: 8\n"
        "model: {}\n"
    )
    (repo_dir / "data.py").write_text(
        "from pathlib import Path\n"
        "import pickle\n"
        "import h5py\n"
        "import numpy as np\n"
        "import torch\n"
        "from torch.utils.data import Dataset\n"
        "\n"
        "class PtychographyDataset(Dataset):\n"
        "    def __init__(self, file_path, scale=100000.0, normalization_dict_path=None, apply_noise=False, cache_object=False, max_probe_modes=8):\n"
        "        _ = (apply_noise, cache_object, max_probe_modes)\n"
        "        self.dp_file = Path(file_path)\n"
        "        self.para_file = self.dp_file.with_name(self.dp_file.stem[:-3] + '_para' + self.dp_file.suffix)\n"
        "        self.object_name = self.dp_file.stem[:-3]\n"
        "        self.scale = float(scale)\n"
        "        self.normalization = 100000.0\n"
        "        if normalization_dict_path is not None:\n"
        "            with open(normalization_dict_path, 'rb') as f:\n"
        "                d = pickle.load(f)\n"
        "            self.normalization = float(d.get(self.object_name, self.normalization))\n"
        "        with h5py.File(self.dp_file, 'r') as f:\n"
        "            self.dp = np.asarray(f['dp'], dtype=np.float32)\n"
        "        with h5py.File(self.para_file, 'r') as f:\n"
        "            self.probe = np.asarray(f['probe'], dtype=np.complex64)\n"
        "            px = np.asarray(f['probe_position_x_m'], dtype=np.float32)\n"
        "            py = np.asarray(f['probe_position_y_m'], dtype=np.float32)\n"
        "            self.positions = np.stack([py, px], axis=1)\n"
        "    def __len__(self):\n"
        "        return int(self.dp.shape[0])\n"
        "    def __getitem__(self, idx):\n"
        "        intensity = (self.dp[idx] / self.normalization) * self.scale\n"
        "        diff_amp = np.sqrt(np.clip(intensity, a_min=0.0, a_max=None)).astype(np.float32)\n"
        "        amp = np.ones_like(diff_amp, dtype=np.float32)\n"
        "        phase = np.zeros_like(diff_amp, dtype=np.float32)\n"
        "        return (\n"
        "            torch.from_numpy(diff_amp).unsqueeze(0),\n"
        "            torch.from_numpy(amp).unsqueeze(0),\n"
        "            torch.from_numpy(phase).unsqueeze(0),\n"
        "            torch.from_numpy(self.probe),\n"
        "            torch.from_numpy(self.positions[idx]),\n"
        "            self.normalization,\n"
        "            self.scale,\n"
        "        )\n"
    )
    (repo_dir / "model" / "model.py").write_text(
        "import torch\n"
        "\n"
        "class PtychoViT(torch.nn.Module):\n"
        "    def __init__(self, config=None):\n"
        "        super().__init__()\n"
        "        _ = config\n"
        "        self.gain = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32))\n"
        "    def forward(self, x, probe, normalization, scale):\n"
        "        _ = (probe, normalization, scale)\n"
        "        pred_diff = x * self.gain\n"
        "        amp = (0.1 * x.squeeze(1) * self.gain) + 1.0\n"
        "        phase = 0.05 * x.squeeze(1)\n"
        "        return pred_diff, amp.unsqueeze(1), phase.unsqueeze(1)\n"
    )


def test_diagnostic_runs_with_bridge_artifacts_and_stops_at_stationary(tmp_path: Path):
    from scripts.studies import ptychovit_input_optimization_diagnostic as diag

    repo = tmp_path / "ptycho-vit"
    _write_stub_ptychovit_repo(repo)
    dp_path = tmp_path / "interop" / "test_dp.hdf5"
    para_path = tmp_path / "interop" / "test_para.hdf5"
    _write_bridge_like_pair(dp_path, para_path)

    with h5py.File(dp_path, "r") as dp_file:
        max_dp = float(np.max(np.asarray(dp_file["dp"])))
    norm_path = dp_path.parent / "normalization.pkl"
    import pickle

    with norm_path.open("wb") as handle:
        pickle.dump({"test": max_dp}, handle)

    checkpoint = tmp_path / "best_model.pth"
    torch.save({"gain": torch.tensor(1.0, dtype=torch.float32)}, checkpoint)

    out_dir = tmp_path / "diag_out"
    rc = diag.main(
        [
            "--output-dir",
            str(out_dir),
            "--checkpoint",
            str(checkpoint),
            "--ptychovit-repo",
            str(repo),
            "--test-dp",
            str(dp_path),
            "--test-para",
            str(para_path),
            "--steps",
            "5",
            "--lr",
            "1e-2",
            "--stationary-threshold",
            "1e9",
            "--input-max",
            "100.0",
        ]
    )

    assert rc == 0
    report = json.loads((out_dir / "diagnostic_report.json").read_text())
    assert report["objective_history"]
    assert report["grad_norm_history"]
    assert int(report["stationary_step"]) <= 5
    assert all(np.isfinite(float(v)) for v in report["objective_history"])
    assert all(np.isfinite(float(v)) for v in report["grad_norm_history"])
    png_path = out_dir / "stationary_point.png"
    assert png_path.exists()
    assert png_path.stat().st_size > 0
