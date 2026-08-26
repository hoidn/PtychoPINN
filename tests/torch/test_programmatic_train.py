"""Focused contracts for direct Torch training settings resolution."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from ptycho_torch.config_factory import resolve_training_payload
from ptycho_torch.config_resolution import TRAINING_INPUT_RULES


@pytest.fixture
def training_npz(tmp_path: Path) -> Path:
    path = tmp_path / "train.npz"
    np.savez(path, probeGuess=np.ones((64, 64), dtype=np.complex64))
    return path


def _resolve(training_npz: Path, tmp_path: Path, **settings):
    return resolve_training_payload(
        train_data_file=training_npz,
        output_dir=tmp_path / "out",
        overrides={"training_groups": 8, **settings},
        profile="ci",
    )


def test_candidate_pool_omission_uses_complete_acquisition(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = _resolve(training_npz, tmp_path, gridsize=1)

    assert payload.pt_data_config.n_raw_frames_selected is None


def test_neighbor_default_covers_gridsize_squared_minus_center(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = _resolve(training_npz, tmp_path, gridsize=3)

    assert payload.pt_data_config.neighbor_count >= 8
    assert payload.pt_data_config.n_raw_frames_selected is None


def test_candidate_pool_explicit_raw_cap_is_preserved(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = _resolve(
        training_npz,
        tmp_path,
        gridsize=1,
        n_raw_frames_selected=5,
    )

    assert payload.pt_data_config.n_raw_frames_selected == 5
    assert payload.tf_training_config.train_raw_selection == 5


@pytest.mark.parametrize("raw_cap", [0, -1, 1.5, True])
def test_candidate_pool_rejects_nonpositive_or_nonintegral_raw_cap(
    training_npz: Path,
    tmp_path: Path,
    raw_cap: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="n_raw_frames_selected must be an exact positive integer",
    ):
        _resolve(
            training_npz,
            tmp_path,
            gridsize=1,
            n_raw_frames_selected=raw_cap,
        )


@pytest.mark.parametrize("neighbor_count", [0, 1.5, True])
def test_neighbor_count_requires_an_exact_positive_integer(
    training_npz: Path,
    tmp_path: Path,
    neighbor_count: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="neighbor_count must be an exact positive integer",
    ):
        _resolve(
            training_npz,
            tmp_path,
            gridsize=1,
            neighbor_count=neighbor_count,
        )


def test_neighbor_count_below_group_requirement_fails_instead_of_overwriting(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"neighbor_count=7.*at least 8.*gridsize=3",
    ):
        _resolve(training_npz, tmp_path, gridsize=3, neighbor_count=7)


def test_neighbor_count_stays_positive_for_gridsize_one_bridge(
    training_npz: Path,
    tmp_path: Path,
) -> None:
    payload = _resolve(training_npz, tmp_path, gridsize=1)

    assert payload.pt_data_config.neighbor_count > 0
    assert payload.tf_training_config.neighbor_count > 0


def test_configuration_table_matches_training_input_rules() -> None:
    guide = Path("docs/CONFIGURATION.md").read_text()
    start = "<!-- programmatic-torch-settings:start -->"
    end = "<!-- programmatic-torch-settings:end -->"
    assert guide.count(start) == guide.count(end) == 1
    section = guide.split(start, 1)[1].split(end, 1)[0]
    documented = dict(
        re.findall(r"^\| `([^`]+)` \| `([^`]+)` \|$", section, re.MULTILINE)
    )
    expected = {rule.canonical: rule.owner for rule in TRAINING_INPUT_RULES}

    assert documented == expected
    assert "Compatibility aliases (non-preferred)" in section
    assert "`model_type` → `mode`" in section
    assert "`max_epochs` → `epochs`" in section
