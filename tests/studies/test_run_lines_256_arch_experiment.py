from pathlib import Path

import numpy as np
import pytest


def test_build_runner_cmd_pins_lines_256_invariants(tmp_path):
    from scripts.studies import run_lines_256_arch_experiment as wrapper

    args = wrapper.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--fno-modes",
            "24",
            "--fno-width",
            "48",
            "--hybrid-downsample-op",
            "avgpool_conv",
        ]
    )

    cmd = wrapper.build_runner_cmd(args)

    assert wrapper.LINES_256_TRAIN_NPZ == Path("outputs/lines_256_arch_improvement/datasets/N256/gs1/train.npz")
    assert wrapper.LINES_256_TEST_NPZ == Path("outputs/lines_256_arch_improvement/datasets/N256/gs1/test.npz")
    assert cmd[:2] == ["python", "scripts/studies/grid_lines_torch_runner.py"]
    assert cmd[cmd.index("--train-npz") + 1] == str(wrapper.LINES_256_TRAIN_NPZ)
    assert cmd[cmd.index("--test-npz") + 1] == str(wrapper.LINES_256_TEST_NPZ)
    assert cmd[cmd.index("--output-dir") + 1] == str(tmp_path / "run")
    assert cmd[cmd.index("--epochs") + 1] == "20"
    assert cmd[cmd.index("--seed") + 1] == "3"
    assert cmd[cmd.index("--N") + 1] == "256"
    assert cmd[cmd.index("--gridsize") + 1] == "1"
    assert cmd[cmd.index("--architecture") + 1] == "hybrid_resnet"
    assert cmd[cmd.index("--scheduler") + 1] == "ReduceLROnPlateau"
    assert cmd[cmd.index("--plateau-min-lr") + 1] == "0.0002"
    assert cmd[cmd.index("--fno-modes") + 1] == "24"
    assert cmd[cmd.index("--fno-width") + 1] == "48"
    assert cmd[cmd.index("--hybrid-downsample-op") + 1] == "avgpool_conv"
    assert "--no-probe-mask" in cmd
    assert "--torch-mae-pred-l2-match-target" in cmd
    assert "--no-torch-mae-pred-l2-match-target" not in cmd


def test_build_runner_cmd_applies_stagea_best_n256_preset(tmp_path):
    from scripts.studies import run_lines_256_arch_experiment as wrapper

    args = wrapper.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--preset",
            "stagea_best_n256",
        ]
    )

    cmd = wrapper.build_runner_cmd(args)

    assert cmd[cmd.index("--fno-modes") + 1] == "24"
    assert cmd[cmd.index("--fno-width") + 1] == "64"


def test_build_runner_cmd_applies_stagec_avgpool_n256_preset(tmp_path):
    from scripts.studies import run_lines_256_arch_experiment as wrapper

    args = wrapper.parse_args(
        [
            "--output-dir",
            str(tmp_path / "run"),
            "--preset",
            "stagec_avgpool_n256",
        ]
    )

    cmd = wrapper.build_runner_cmd(args)

    assert cmd[cmd.index("--hybrid-downsample-op") + 1] == "avgpool_conv"
    assert cmd[cmd.index("--fno-modes") + 1] == "12"
    assert cmd[cmd.index("--fno-width") + 1] == "32"
    assert cmd[cmd.index("--hybrid-resnet-blocks") + 1] == "6"


def test_main_runs_runner_and_writes_invocation(monkeypatch, tmp_path):
    from scripts.studies import run_lines_256_arch_experiment as wrapper

    calls = []

    def fake_run(cmd, check, capture_output, text):
        from subprocess import CompletedProcess

        assert check is False
        assert capture_output is True
        assert text is True
        calls.append(cmd)
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run_dir = output_dir / "runs" / "pinn_hybrid_resnet"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text('{"amp_ssim": 0.75}')
        return CompletedProcess(cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)
    output_dir = tmp_path / "wrapper_out"
    rc = wrapper.main(
        [
            "--output-dir",
            str(output_dir),
            "--fno-modes",
            "16",
        ]
    )

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][0:2] == ["python", "scripts/studies/grid_lines_torch_runner.py"]
    assert calls[0][calls[0].index("--fno-modes") + 1] == "16"
    assert (output_dir / "invocation.json").exists()
    assert (output_dir / "invocation.sh").exists()
    assert (output_dir / "driver_stdout.log").exists()
    assert (output_dir / "driver_stderr.log").exists()


def test_main_does_not_require_probe_rerender_after_success(monkeypatch, tmp_path):
    from scripts.studies import run_lines_256_arch_experiment as wrapper

    def fake_run(cmd, check, capture_output, text):
        from subprocess import CompletedProcess

        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        gt_dir = output_dir / "recons" / "gt"
        pred_dir = output_dir / "recons" / "pinn_hybrid_resnet"
        gt_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.savez(
            gt_dir / "recon.npz",
            amp=np.ones((8, 8), dtype=np.float32),
            phase=np.zeros((8, 8), dtype=np.float32),
        )
        np.savez(
            pred_dir / "recon.npz",
            amp=np.ones((8, 8), dtype=np.float32),
            phase=np.zeros((8, 8), dtype=np.float32),
        )
        visuals_dir = output_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)
        (visuals_dir / "compare_amp_phase.png").write_text("plain-compare", encoding="utf-8")
        return CompletedProcess(cmd, returncode=0, stdout="ok\n", stderr="")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    output_dir = tmp_path / "wrapper_out"
    rc = wrapper.main(["--output-dir", str(output_dir)])

    assert rc == 0
    assert (output_dir / "visuals" / "compare_amp_phase.png").read_text(encoding="utf-8") == "plain-compare"
    assert not (output_dir / "visuals" / "compare_amp_phase_probe.png").exists()


def test_main_raises_when_runner_fails(monkeypatch, tmp_path):
    from scripts.studies import run_lines_256_arch_experiment as wrapper

    def fake_run(cmd, check, capture_output, text):
        from subprocess import CompletedProcess

        return CompletedProcess(cmd, returncode=2, stdout="", stderr="boom")

    monkeypatch.setattr(wrapper.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="lines_256 arch experiment failed"):
        wrapper.main(["--output-dir", str(tmp_path / "wrapper_out")])
