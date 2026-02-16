from pathlib import Path


def test_study_index_registers_fly001_n128_external_runbook():
    index = Path("docs/studies/index.md").read_text()
    assert "grid-lines-external-fly001-n128-top-train-bottom-test-e40" in index

    script = Path("scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_bottom_test_e40.sh")
    assert script.exists()
