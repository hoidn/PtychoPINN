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
