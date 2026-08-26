"""Executable contract for the ordinary direct Torch example."""

from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_programmatic_torch_example_is_direct_and_executable(monkeypatch) -> None:
    example = Path("examples/programmatic_torch.py")
    source = example.read_text()
    assert len(source.splitlines()) <= 50

    imports = {
        (node.module, alias.name, alias.asname)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    plain_imports = {
        (alias.name, alias.asname)
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imports == {
        ("ptycho_torch.inference", "reconstruct", None),
        ("ptycho_torch.train", "train", None),
    }
    assert plain_imports == {("matplotlib.pyplot", "plt")}

    calls = []
    model = Path("outputs/run1084_cnn/wts.h5.zip")
    amplitude, phase = object(), object()

    def train(dataset, output_dir, settings):
        calls.append(("train", dataset, output_dir, settings))
        return model

    def reconstruct(received_model, dataset):
        calls.append(("reconstruct", received_model, dataset))
        return SimpleNamespace(amplitude=amplitude, phase=phase)

    class Axis:
        def imshow(self, value, *, cmap):
            calls.append(("imshow", value, cmap))

    pyplot = ModuleType("matplotlib.pyplot")
    pyplot.subplots = lambda *_args: (object(), [Axis(), Axis()])
    pyplot.show = lambda: calls.append(("show",))
    matplotlib = ModuleType("matplotlib")
    matplotlib.pyplot = pyplot
    inference = ModuleType("ptycho_torch.inference")
    inference.reconstruct = reconstruct
    training = ModuleType("ptycho_torch.train")
    training.train = train
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    monkeypatch.setitem(sys.modules, "ptycho_torch.inference", inference)
    monkeypatch.setitem(sys.modules, "ptycho_torch.train", training)

    runpy.run_path(example)

    data = "datasets/Run1084_recon3_postPC_shrunk_3.npz"
    assert calls == [
        ("train", data, "outputs/run1084_cnn", {
            "architecture": "cnn",
            "training_groups": 256,
            "nphotons": 1e9,
            "epochs": 1,
        }),
        ("reconstruct", model, data),
        ("imshow", amplitude, "gray"),
        ("imshow", phase, "twilight"),
        ("show",),
    ]
