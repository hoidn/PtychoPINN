import numpy as np

from ptycho import misc, params


def test_explicit_path_prefix_matches_legacy_format_and_ignores_poisoned_global(
    monkeypatch,
):
    poisoned_cfg = {
        "label": "wrong-label",
        "output_prefix": "wrong-output",
        "timestamp": "01/01/1999, 00:00:00",
    }
    monkeypatch.setattr(params, "cfg", poisoned_cfg)

    result = misc.get_path_prefix_explicit(
        label="experiment",
        output_prefix="outputs",
        timestamp="07/28/2026, 17:15:04",
    )

    assert result == "outputs/07-28-2026-17.15.04_experiment/"
    assert params.cfg == poisoned_cfg


def test_legacy_memoizer_does_not_retain_params_cfg_alias(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PTYCHO_DISABLE_MEMOIZE", raising=False)
    monkeypatch.setenv("PTYCHO_MEMOIZE_KEY_MODE", "dataset")
    monkeypatch.setattr(params, "cfg", {"N": 64})
    observed_N = []

    @misc.memoize_disk_and_memory
    def operation(_value):
        observed_N.append(params.cfg["N"])
        return np.asarray([len(observed_N)])

    assert operation("same").item() == 1
    monkeypatch.setattr(params, "cfg", {"N": 128})
    assert operation("same").item() == 2
    assert observed_N == [64, 128]
