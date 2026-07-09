from pathlib import Path


def test_build_lines_256_dataset_uses_canonical_set_phi_config(monkeypatch, tmp_path):
    from scripts.studies.runbooks import rebuild_lines_256_dataset

    captured = {}

    def fake_build_datasets(*, dataset_source, cfg, required_ns, **kwargs):
        captured["dataset_source"] = dataset_source
        captured["cfg"] = cfg
        captured["required_ns"] = list(required_ns)
        captured["kwargs"] = kwargs
        return {
            256: {
                "train_npz": str(tmp_path / "datasets" / "N256" / "gs1" / "train.npz"),
                "test_npz": str(tmp_path / "datasets" / "N256" / "gs1" / "test.npz"),
                "gt_recon": str(tmp_path / "recons" / "gt" / "recon.npz"),
                "tag": "N256",
            }
        }

    monkeypatch.setattr(rebuild_lines_256_dataset, "build_datasets", fake_build_datasets)

    summary = rebuild_lines_256_dataset.build_lines_256_dataset(
        output_root=tmp_path,
        probe_npz=tmp_path / "probe.npz",
    )

    cfg = captured["cfg"]
    assert captured["dataset_source"] == "synthetic_lines"
    assert captured["required_ns"] == [256]
    assert cfg.N == 256
    assert cfg.gridsize == 1
    assert cfg.output_dir == tmp_path
    assert cfg.probe_npz == tmp_path / "probe.npz"
    assert cfg.nimgs_train == 2
    assert cfg.nimgs_test == 1
    assert cfg.nphotons == 1e9
    assert cfg.size == 392
    assert cfg.offset == 4
    assert cfg.outer_offset_train == 8
    assert cfg.outer_offset_test == 20
    assert cfg.probe_source == "custom"
    assert cfg.probe_scale_mode == "pad_preserve"
    assert cfg.probe_smoothing_sigma == 0.5
    assert cfg.set_phi is True

    assert summary["train_npz"] == tmp_path / "datasets" / "N256" / "gs1" / "train.npz"
    assert summary["test_npz"] == tmp_path / "datasets" / "N256" / "gs1" / "test.npz"
    assert summary["set_phi"] is True


def test_build_lines_256_dataset_accepts_explicit_probe_transform_pipeline(monkeypatch, tmp_path):
    from scripts.studies.runbooks import rebuild_lines_256_dataset

    captured = {}

    def fake_build_datasets(*, dataset_source, cfg, required_ns, **kwargs):
        captured["dataset_source"] = dataset_source
        captured["cfg"] = cfg
        captured["required_ns"] = list(required_ns)
        captured["kwargs"] = kwargs
        return {
            256: {
                "train_npz": str(tmp_path / "datasets" / "N256" / "gs1" / "train.npz"),
                "test_npz": str(tmp_path / "datasets" / "N256" / "gs1" / "test.npz"),
                "gt_recon": str(tmp_path / "recons" / "gt" / "recon.npz"),
                "tag": "N256",
            }
        }

    monkeypatch.setattr(rebuild_lines_256_dataset, "build_datasets", fake_build_datasets)

    summary = rebuild_lines_256_dataset.build_lines_256_dataset(
        output_root=tmp_path,
        probe_npz=tmp_path / "probe.npz",
        probe_transform_pipeline="smooth:0.5|pad:128|interp:256",
    )

    cfg = captured["cfg"]
    assert cfg.N == 256
    assert cfg.probe_transform_pipeline == "smooth:0.5|pad:128|interp:256"
    assert cfg.probe_scale_mode == "pipeline"
    assert cfg.probe_smoothing_sigma == 0.0
    assert summary["config"]["probe_transform_pipeline"] == "smooth:0.5|pad:128|interp:256"
