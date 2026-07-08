"""Tests for the Lines128 NeuralOperator U-NO preflight helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class _UnoReturnsBchw(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        *,
        lifting_channels=256,
        projection_channels=256,
        positional_embedding="grid",
        n_layers=4,
        uno_n_modes=None,
        **_kwargs,
    ):
        self.init_kwargs = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "hidden_channels": hidden_channels,
            "lifting_channels": lifting_channels,
            "projection_channels": projection_channels,
            "positional_embedding": positional_embedding,
            "n_layers": n_layers,
            "uno_n_modes": uno_n_modes,
        }
        super().__init__()

    def forward(self, x):
        batch, _channels, height, width = x.shape
        return torch.zeros((batch, 2, height, width), dtype=x.dtype, device=x.device)


class _UnoReturnsComplexLike(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        batch, _channels, height, width = x.shape
        return torch.zeros((batch, 1, height, width, 2), dtype=x.dtype, device=x.device)


class _UnoRejectsListModes(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels, *, uno_n_modes=None, **_kwargs):
        if isinstance(uno_n_modes, (list, tuple)):
            raise TypeError("list modes unsupported")
        self.uno_n_modes = uno_n_modes
        super().__init__()

    def forward(self, x):
        batch, _channels, height, width = x.shape
        return torch.zeros((batch, 2, height, width), dtype=x.dtype, device=x.device)


class _UnoReturnsWrongChannels(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        batch, _channels, height, width = x.shape
        return torch.zeros((batch, 3, height, width), dtype=x.dtype, device=x.device)


class _UnoRaisesValueError(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        raise ValueError("bad constructor shape")


def test_run_lines128_uno_preflight_blocks_when_neuralop_import_is_missing(tmp_path, monkeypatch):
    from scripts.studies import lines128_uno_preflight as preflight

    def _missing_import():
        raise ModuleNotFoundError("no module named neuralop")

    monkeypatch.setattr(preflight, "_import_uno_dependencies", _missing_import)
    monkeypatch.setattr(preflight, "_capture_pip_show_text", lambda: "")
    monkeypatch.setattr(
        preflight,
        "current_runtime_provenance",
        lambda **_kwargs: {"python_executable": "python", "torch_version": "2.6.0", "gpu": "unknown"},
    )

    decision = preflight.run_lines128_uno_preflight(output_root=tmp_path)

    assert decision["status"] == "blocked_neuraloperator_missing_or_incompatible"
    assert decision["package_status"] == "missing_or_incompatible"
    assert "neuralop" in decision["blocker_reason"]
    assert (tmp_path / "preflight_decision.json").exists()
    assert _read_json(tmp_path / "preflight_decision.json")["status"] == decision["status"]


def test_run_lines128_uno_preflight_retries_missing_import_after_install_attempt(tmp_path, monkeypatch):
    from scripts.studies import lines128_uno_preflight as preflight

    fake_neuralop = type("__FakeNeuralOp__", (), {"__version__": "2.0.0", "__file__": "/tmp/neuralop/__init__.py"})
    import_attempts = {"count": 0}
    install_attempts: list[str] = []
    pip_show_texts = iter(
        [
            "WARNING: Package(s) not found: neuraloperator\n",
            "Name: neuraloperator\nVersion: 2.0.0\n",
        ]
    )

    def _import_once_missing_then_succeed():
        import_attempts["count"] += 1
        if import_attempts["count"] == 1:
            raise ModuleNotFoundError("no module named neuralop")
        return fake_neuralop, _UnoReturnsBchw

    monkeypatch.setattr(preflight, "_import_uno_dependencies", _import_once_missing_then_succeed)
    monkeypatch.setattr(preflight, "_capture_pip_show_text", lambda: next(pip_show_texts))
    monkeypatch.setattr(
        preflight,
        "_attempt_neuraloperator_install",
        lambda: install_attempts.append("neuraloperator==2.0.0"),
        raising=False,
    )
    monkeypatch.setattr(
        preflight,
        "current_runtime_provenance",
        lambda **_kwargs: {"python_executable": "python", "torch_version": "2.6.0", "gpu": "unknown"},
    )

    decision = preflight.run_lines128_uno_preflight(output_root=tmp_path)

    assert install_attempts == ["neuraloperator==2.0.0"]
    assert import_attempts["count"] == 2
    assert decision["status"] == "ready_for_uno_generator_integration"
    assert decision["package_provenance"]["pip_show_present"] is True
    assert "Name: neuraloperator" in (tmp_path / "pip_show_neuraloperator.txt").read_text(encoding="utf-8")


def test_run_lines128_uno_preflight_marks_pip_show_missing_warning_as_not_present(tmp_path, monkeypatch):
    from scripts.studies import lines128_uno_preflight as preflight

    install_attempts: list[str] = []

    def _missing_import():
        raise ModuleNotFoundError("no module named neuralop")

    monkeypatch.setattr(preflight, "_import_uno_dependencies", _missing_import)
    monkeypatch.setattr(
        preflight,
        "_capture_pip_show_text",
        lambda: "WARNING: Package(s) not found: neuraloperator\n",
    )
    monkeypatch.setattr(
        preflight,
        "_attempt_neuraloperator_install",
        lambda: install_attempts.append("neuraloperator==2.0.0"),
        raising=False,
    )
    monkeypatch.setattr(
        preflight,
        "current_runtime_provenance",
        lambda **_kwargs: {"python_executable": "python", "torch_version": "2.6.0", "gpu": "unknown"},
    )

    decision = preflight.run_lines128_uno_preflight(output_root=tmp_path)

    assert install_attempts == ["neuraloperator==2.0.0"]
    assert decision["status"] == "blocked_neuraloperator_missing_or_incompatible"
    assert decision["package_provenance"]["pip_show_present"] is False


def test_run_lines128_uno_preflight_records_signature_provenance_and_frozen_defaults(
    tmp_path,
    monkeypatch,
):
    from scripts.studies import lines128_uno_preflight as preflight

    fake_neuralop = type("__FakeNeuralOp__", (), {"__version__": "2.0.0", "__file__": "/tmp/neuralop/__init__.py"})
    monkeypatch.setattr(
        preflight,
        "_import_uno_dependencies",
        lambda: (fake_neuralop, _UnoReturnsBchw),
    )
    monkeypatch.setattr(preflight, "_capture_pip_show_text", lambda: "Name: neuraloperator\nVersion: 2.0.0\n")
    monkeypatch.setattr(
        preflight,
        "current_runtime_provenance",
        lambda **_kwargs: {"python_executable": "python", "torch_version": "2.6.0", "gpu": "unknown"},
    )

    decision = preflight.run_lines128_uno_preflight(output_root=tmp_path)

    assert decision["status"] == "ready_for_uno_generator_integration"
    assert decision["package_provenance"]["module_version"] == "2.0.0"
    assert decision["frozen_uno_settings"]["hidden_channels"] == 32
    assert decision["frozen_uno_settings"]["lifting_channels"] == 128
    assert decision["frozen_uno_settings"]["projection_channels"] == 128
    assert decision["frozen_uno_settings"]["n_layers"] == 4
    assert decision["frozen_uno_settings"]["positional_embedding"] == "grid"
    assert decision["frozen_uno_settings"]["generator_output_mode"] == "real_imag"
    assert decision["frozen_uno_settings"]["uno_out_channels"] == [32, 64, 64, 32]
    assert decision["frozen_uno_settings"]["uno_scalings"] == [[1.0, 1.0], [0.5, 0.5], [1, 1], [2, 2]]
    assert decision["frozen_uno_settings"]["channel_mlp_skip"] == "linear"
    assert decision["frozen_uno_settings"]["uno_n_modes"] == [[12, 12], [12, 12], [12, 12], [12, 12]]
    assert decision["shape_probe"]["accepted"] is True
    assert decision["shape_probe"]["raw_output_shape"] == [2, 2, 128, 128]
    assert "uno_n_modes" in decision["uno_signature"]["text"]
    assert (tmp_path / "environment_probe.json").exists()
    assert (tmp_path / "uno_signature.json").exists()
    assert (tmp_path / "uno_shape_probe.json").exists()


def test_run_lines128_uno_preflight_accepts_lossless_real_imag_layout(tmp_path, monkeypatch):
    from scripts.studies import lines128_uno_preflight as preflight

    fake_neuralop = type("__FakeNeuralOp__", (), {"__version__": "2.0.0", "__file__": "/tmp/neuralop/__init__.py"})
    monkeypatch.setattr(
        preflight,
        "_import_uno_dependencies",
        lambda: (fake_neuralop, _UnoReturnsComplexLike),
    )
    monkeypatch.setattr(preflight, "_capture_pip_show_text", lambda: "Name: neuraloperator\nVersion: 2.0.0\n")
    monkeypatch.setattr(
        preflight,
        "current_runtime_provenance",
        lambda **_kwargs: {"python_executable": "python", "torch_version": "2.6.0", "gpu": "unknown"},
    )

    decision = preflight.run_lines128_uno_preflight(output_root=tmp_path)

    assert decision["status"] == "ready_for_uno_generator_integration"
    assert decision["shape_probe"]["accepted"] is True
    assert decision["shape_probe"]["adapter_required"] is True
    assert decision["shape_probe"]["mapping"] == "bchw2_to_bhwc1r2"


def test_run_lines128_uno_preflight_falls_back_to_scalar_uno_n_modes_when_sequence_is_rejected(
    tmp_path,
    monkeypatch,
):
    from scripts.studies import lines128_uno_preflight as preflight

    fake_neuralop = type("__FakeNeuralOp__", (), {"__version__": "2.0.0", "__file__": "/tmp/neuralop/__init__.py"})
    monkeypatch.setattr(
        preflight,
        "_import_uno_dependencies",
        lambda: (fake_neuralop, _UnoRejectsListModes),
    )
    monkeypatch.setattr(preflight, "_capture_pip_show_text", lambda: "Name: neuraloperator\nVersion: 2.0.0\n")
    monkeypatch.setattr(
        preflight,
        "current_runtime_provenance",
        lambda **_kwargs: {"python_executable": "python", "torch_version": "2.6.0", "gpu": "unknown"},
    )

    decision = preflight.run_lines128_uno_preflight(output_root=tmp_path)

    assert decision["status"] == "ready_for_uno_generator_integration"
    assert decision["frozen_uno_settings"]["uno_n_modes"] == 12
    assert decision["frozen_uno_settings"]["uno_n_modes_form"] == "scalar_fallback"
    assert "rejected" in decision["frozen_uno_settings"]["uno_n_modes_note"]


def test_run_lines128_uno_preflight_blocks_on_unmappable_output_layout(tmp_path, monkeypatch):
    from scripts.studies import lines128_uno_preflight as preflight

    fake_neuralop = type("__FakeNeuralOp__", (), {"__version__": "2.0.0", "__file__": "/tmp/neuralop/__init__.py"})
    monkeypatch.setattr(
        preflight,
        "_import_uno_dependencies",
        lambda: (fake_neuralop, _UnoReturnsWrongChannels),
    )
    monkeypatch.setattr(preflight, "_capture_pip_show_text", lambda: "Name: neuraloperator\nVersion: 2.0.0\n")
    monkeypatch.setattr(
        preflight,
        "current_runtime_provenance",
        lambda **_kwargs: {"python_executable": "python", "torch_version": "2.6.0", "gpu": "unknown"},
    )

    decision = preflight.run_lines128_uno_preflight(output_root=tmp_path)

    assert decision["status"] == "blocked_uno_shape_contract_mismatch"
    assert decision["shape_probe"]["accepted"] is False
    assert decision["shape_probe"]["raw_output_shape"] == [2, 3, 128, 128]
    assert "real_imag" in decision["blocker_reason"]


def test_run_lines128_uno_preflight_blocks_on_non_typeerror_constructor_incompatibility(
    tmp_path,
    monkeypatch,
):
    from scripts.studies import lines128_uno_preflight as preflight

    fake_neuralop = type("__FakeNeuralOp__", (), {"__version__": "2.0.0", "__file__": "/tmp/neuralop/__init__.py"})
    monkeypatch.setattr(
        preflight,
        "_import_uno_dependencies",
        lambda: (fake_neuralop, _UnoRaisesValueError),
    )
    monkeypatch.setattr(preflight, "_capture_pip_show_text", lambda: "Name: neuraloperator\nVersion: 2.0.0\n")
    monkeypatch.setattr(
        preflight,
        "current_runtime_provenance",
        lambda **_kwargs: {"python_executable": "python", "torch_version": "2.6.0", "gpu": "unknown"},
    )

    try:
        decision = preflight.run_lines128_uno_preflight(output_root=tmp_path)
    except Exception as exc:  # pragma: no cover - converted into an assertion failure
        pytest.fail(f"preflight should emit a blocker artifact instead of raising: {exc}")

    assert decision["status"] == "blocked_neuraloperator_missing_or_incompatible"
    assert decision["package_status"] == "constructor_incompatible"
    assert "bad constructor shape" in decision["blocker_reason"]
    assert (tmp_path / "preflight_decision.json").exists()
