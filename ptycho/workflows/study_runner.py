"""Generic Hydra-based study runner.

Composes an arm config from a study's ``conf/`` tree, writes it as a
self-contained ``arm.yaml``, and executes the synthetic pipeline runner on it
as a subprocess. Design contract:
docs/superpowers/specs/2026-08-07-hydra-study-runner-design.md

Invocation (single arm):
    ptycho_study --config-dir studies/<study>/conf --config-name config \\
        family=lines profile=ci_nll model.rect_s1s2_init=ones

Invocation (matrix sweep):
    ptycho_study --config-dir studies/<study>/conf --config-name config -m \\
        family=lines,deadleaves profile=legacy_mae,legacy_nll,ci_nll
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIRED_STUDY_KEYS = ("name", "runner_script", "sentinel")


def _git_state(repo: Path) -> dict[str, Any]:
    def run(*args: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True
        )

    head = run("rev-parse", "HEAD")
    if head.returncode != 0:
        return {"git_commit": None, "git_dirty": None}
    status = run("status", "--porcelain")
    return {
        "git_commit": head.stdout.strip(),
        "git_dirty": bool(status.stdout.strip()),
    }


def _prepare_shared_datasets(arm: dict[str, Any], out: Path, study: dict[str, Any]) -> None:
    """Point <out>/datasets at a per-(family, domain) shared dir.

    First arm per key simulates through the symlink; later arms find
    manifest.json and drop the simulate stage (the runner re-verifies split
    artifacts against the manifest before training). Serial sweeps only.
    """
    family = study.get("family_name")
    domain = (arm.get("simulation") or {}).get("measurement_domain")
    missing = [
        key
        for key, value in (
            ("study.family_name", family),
            ("simulation.measurement_domain", domain),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"shared_datasets requires {', '.join(missing)}: dataset sharing keys "
            "off (family_name, measurement_domain); a fallback would alias "
            "distinct datasets into one shared directory"
        )
    shared = (out.parent / "datasets_shared" / f"{family}__{domain}").resolve()
    shared.mkdir(parents=True, exist_ok=True)
    link = out / "datasets"
    if link.is_symlink() or link.exists():
        if not (link.is_symlink() and link.resolve() == shared):
            raise RuntimeError(
                f"{link} exists and is not a symlink to {shared}; refusing to touch it"
            )
    else:
        link.symlink_to(shared, target_is_directory=True)
    if (shared / "manifest.json").exists():
        workflow = arm.setdefault("workflow", {})
        workflow["stages"] = [s for s in workflow.get("stages", []) if s != "simulate"]


@hydra.main(version_base="1.2", config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    hc = HydraConfig.get()
    out = Path(hc.runtime.output_dir).resolve()

    arm = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(arm, dict):
        raise ValueError("study config must compose to a mapping")
    study = arm.pop("study", None)
    if not isinstance(study, dict):
        raise ValueError("study config must define a top-level 'study' mapping")
    missing = [key for key in _REQUIRED_STUDY_KEYS if not study.get(key)]
    if missing:
        raise ValueError(f"study config missing keys: {', '.join(missing)}")

    sentinel = out / str(study["sentinel"])
    if sentinel.exists():
        print(f"[study] skip {out.name}: {study['sentinel']} already present")
        return

    runner_root = Path(study.get("runner_root") or REPO_ROOT).resolve()
    runner = (runner_root / str(study["runner_script"])).resolve()
    if not runner.is_file():
        raise FileNotFoundError(f"runner script not found: {runner}")

    out.mkdir(parents=True, exist_ok=True)
    if study.get("shared_datasets"):
        _prepare_shared_datasets(arm, out, study)
    arm.setdefault("workflow", {})["output_root"] = str(out)
    arm_path = out / "arm.yaml"
    OmegaConf.save(OmegaConf.create(arm), arm_path)

    provenance = {
        "study": study["name"],
        "invoked_at": datetime.now(timezone.utc).isoformat(),
        "overrides": list(hc.overrides.task),
        "runner": str(runner),
        "runner_root": str(runner_root),
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "python": sys.executable,
        **_git_state(runner_root),
    }
    (out / "study_provenance.json").write_text(json.dumps(provenance, indent=2))

    env = os.environ.copy()
    cuda_devices = study.get("cuda_visible_devices")
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)
    env.setdefault("PYTHONUNBUFFERED", "1")

    cmd = [sys.executable, str(runner), "--config", str(arm_path)]
    log_path = out / "runner.log"
    print(f"[study] run {out.name}: {' '.join(cmd)}")
    print(f"[study] log: {log_path}")
    with log_path.open("wb") as log:
        result = subprocess.run(
            cmd, cwd=runner_root, env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"runner failed for {out.name} (exit {result.returncode}); see {log_path}"
        )
    if not sentinel.exists():
        raise RuntimeError(
            f"runner exited 0 for {out.name} but sentinel "
            f"{study['sentinel']} was not produced"
        )
    print(f"[study] done {out.name}")


if __name__ == "__main__":
    main()
