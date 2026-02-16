from pathlib import Path


def test_study_index_registers_fly001_n128_external_runbook():
    index = Path("docs/studies/index.md").read_text()
    assert "grid-lines-external-fly001-n128-top-train-full-test-e40" in index

    script = Path("scripts/studies/runbooks/grid_lines_external_fly001_n128_top_train_full_test_e40.sh")
    assert script.exists()


def test_docs_index_links_fly001_128_dataset_guide():
    index = Path("docs/index.md").read_text()
    assert "FLY001 N=128 Dataset Guide" in index
    assert Path("docs/FLY001_128_DATASET_GUIDE.md").exists()


def test_commands_reference_contains_fly001_128_recipe():
    commands = Path("docs/COMMANDS_REFERENCE.md").read_text()
    assert "fly001_128_top_half_converted.npz" in commands
    assert "fly001_128_full_test_converted.npz" in commands
