from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests
import torch


def _download_checkpoint(file_id: str, destination: Path) -> str:
    session = requests.Session()
    warning = session.get(
        f"https://drive.google.com/uc?export=download&id={file_id}",
        timeout=120,
    )
    warning.raise_for_status()
    text = warning.text
    confirm = re.search(r'name="confirm" value="([^"]+)"', text)
    uuid = re.search(r'name="uuid" value="([^"]+)"', text)
    size_label = re.search(r"\(([^)]+)\)</span> is too large", text)
    if not confirm or not uuid:
        raise ValueError(f"could not recover Google Drive confirm token for {file_id}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with session.get(
        "https://drive.usercontent.google.com/download",
        params={
            "id": file_id,
            "export": "download",
            "confirm": confirm.group(1),
            "uuid": uuid.group(1),
        },
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return size_label.group(1) if size_label else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and load representative WaveBench checkpoints.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--wavebench-root", default="tmp/wavebench_repo")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--download-dir", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    wavebench_root = (repo_root / args.wavebench_root).resolve()
    sys.path.insert(0, str(wavebench_root))
    from wavebench.nn.pl_model_wrapper import LitModel

    download_dir = Path(args.download_dir).resolve()
    items = {
        "fno": {
            "variant": "fno-depth-4",
            "file_id": "1Id2-BrE9md3ypqHysTITYVGMfQQ3qsbJ",
        },
        "unet": {
            "variant": "unet-ch-32",
            "file_id": "17FXa31RSMA-7vwRB_492Ex8AY-2YgSdl",
        },
    }

    results = {"baseline_load_smokes": {}}
    for baseline_name, item in items.items():
        destination = download_dir / f"{baseline_name}_{item['variant']}.ckpt"
        size_label = _download_checkpoint(item["file_id"], destination)
        checkpoint = torch.load(str(destination), map_location="cpu")
        result = {
            "variant": item["variant"],
            "file_id": item["file_id"],
            "google_drive_size_label": size_label,
            "local_checkpoint_path": str(destination),
            "local_size_bytes": destination.stat().st_size,
            "checkpoint_hyper_parameters": checkpoint.get("hyper_parameters"),
        }
        try:
            model = LitModel.load_from_checkpoint(str(destination), map_location="cpu")
        except Exception as exc:  # pragma: no cover - exercised via live probe
            result["load_ok"] = False
            result["load_error"] = f"{type(exc).__name__}: {exc}"
        else:
            result["load_ok"] = True
            result["model_class"] = type(model.model).__name__
            result["model_config"] = dict(model.hparams["model_config"])
            result["training_hparams"] = {
                key: model.hparams[key]
                for key in [
                    "max_num_steps",
                    "eta_min",
                    "weight_decay",
                    "learning_rate",
                    "loss_fun_type",
                ]
            }
        results["baseline_load_smokes"][baseline_name] = result

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
