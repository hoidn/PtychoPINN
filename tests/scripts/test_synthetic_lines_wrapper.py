"""Migration contracts for the retired synthetic-lines wrapper."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_wrapper_warns_once_and_delegates_to_simulation_stage(monkeypatch, tmp_path):
    from scripts.simulation import run_with_synthetic_lines as wrapper

    captured = {}

    def fake_main(argv):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(
        wrapper,
        "_run_synthetic_main",
        fake_main,
    )

    with pytest.warns(DeprecationWarning) as warnings_seen:
        result = wrapper.main(
            [
                "--output-dir",
                str(tmp_path / "out"),
                "--probe-size",
                "64",
                "--seed",
                "9",
            ]
        )

    assert len(warnings_seen) == 1
    assert result == 0
    assert captured["argv"] == [
        "--stages",
        "simulate",
        "--output-root",
        str(tmp_path / "out"),
        "--N",
        "64",
        "--seed",
        "9",
    ]


def test_wrapper_translates_legacy_photon_and_buffer_names():
    from scripts.simulation import run_with_synthetic_lines as wrapper

    assert wrapper.translate_legacy_arguments(
        [
            "--output-dir",
            "out",
            "--n-photons",
            "1e8",
            "--buffer=32",
        ]
    ) == [
        "--stages",
        "simulate",
        "--output-root",
        "out",
        "--N",
        "64",
        "--photons-per-pattern",
        "1e8",
        "--scan-buffer=32",
    ]


def test_wrapper_preserves_the_legacy_default_probe_size():
    from scripts.simulation import run_with_synthetic_lines as wrapper

    assert wrapper.translate_legacy_arguments(["--output-dir", "out"]) == [
        "--stages",
        "simulate",
        "--output-root",
        "out",
        "--N",
        "64",
    ]


@pytest.mark.parametrize(
    "argv, message",
    [
        (
            ["--output-dir", "out", "--simulation-config", "legacy.toml"],
            "legacy --simulation-config.*--config",
        ),
        (["--output-dir", "out", "--n-images", "8"], "--n-images.*ambiguous"),
        (["--output-dir", "out", "--stages", "train"], "--stages.*owned"),
        (["--output-dir", "out", "--output-root", "other"], "--output-root.*owned"),
        (["--output-dir", "out", "--visualize"], "--visualize.*no equivalent"),
    ],
)
def test_wrapper_rejects_ambiguous_legacy_combinations(argv, message):
    from scripts.simulation import run_with_synthetic_lines as wrapper

    with pytest.raises(ValueError, match=message):
        wrapper.translate_legacy_arguments(argv)


def test_wrapper_contains_no_simulation_or_training_implementation():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts/simulation/run_with_synthetic_lines.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    function_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function_names.add(node.name)

    forbidden_prefixes = (
        "numpy",
        "tensorflow",
        "torch",
        "ptycho.diffsim",
        "ptycho.simulation",
        "ptycho.workflows.training",
    )
    assert not any(name.startswith(forbidden_prefixes) for name in imported), imported
    assert (
        not {
            "generate_and_save_synthetic_input",
            "run_simulation_workflow",
            "resolve_synthetic_simulation",
        }
        & function_names
    )
