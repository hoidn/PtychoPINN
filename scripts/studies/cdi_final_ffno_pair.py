"""Shared authority for selecting the final CDI FFNO same-depth paper pair."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = (
    REPO_ROOT
    / ".artifacts"
    / "work"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
)

CDI_HISTORICAL_SUPERVISED_PROXY_METRICS_JSON = (
    ARTIFACTS_ROOT
    / "2026-04-29-cdi-lines128-supervised-equivalent-rows"
    / "runs"
    / "supervised_ffno_extension_20260430T180217Z"
    / "runs"
    / "supervised_ffno"
    / "metrics.json"
)


@dataclass(frozen=True)
class CdiFinalFfnoPair:
    pair_key: str
    output_stem: str
    depth_label: str
    claim_boundary: str
    pinn_root: Path
    pinn_source_row_id: str
    supervised_root: Path
    supervised_source_row_id: str

    @property
    def pinn_metrics_json(self) -> Path:
        return self.pinn_root / "runs" / self.pinn_source_row_id / "metrics.json"

    @property
    def supervised_metrics_json(self) -> Path:
        return self.supervised_root / "runs" / self.supervised_source_row_id / "metrics.json"

    @property
    def pinn_config_json(self) -> Path:
        return self.pinn_root / "runs" / self.pinn_source_row_id / "config.json"

    @property
    def supervised_config_json(self) -> Path:
        return self.supervised_root / "runs" / self.supervised_source_row_id / "config.json"

    @property
    def pinn_recon_npz(self) -> Path:
        return self.pinn_root / "recons" / self.pinn_source_row_id / "recon.npz"

    @property
    def supervised_recon_npz(self) -> Path:
        return self.supervised_root / "recons" / self.supervised_source_row_id / "recon.npz"

    @property
    def pinn_model_manifest_json(self) -> Path:
        return self.pinn_root / "model_manifest.json"

    @property
    def supervised_model_manifest_json(self) -> Path:
        return self.supervised_root / "model_manifest.json"

    def source_row_ids(self) -> dict[str, str]:
        return {
            "pinn_ffno": self.pinn_source_row_id,
            "supervised_ffno": self.supervised_source_row_id,
        }

    def provenance_payload(self) -> dict[str, object]:
        return {
            "pair_key": self.pair_key,
            "depth_label": self.depth_label,
            "claim_boundary": self.claim_boundary,
            "source_row_ids": self.source_row_ids(),
            "source_roots": {
                "pinn_ffno": str(self.pinn_root.relative_to(REPO_ROOT)),
                "supervised_ffno": str(self.supervised_root.relative_to(REPO_ROOT)),
            },
            "source_metrics_json": {
                "pinn_ffno": str(self.pinn_metrics_json.relative_to(REPO_ROOT)),
                "supervised_ffno": str(self.supervised_metrics_json.relative_to(REPO_ROOT)),
            },
            "source_config_json": {
                "pinn_ffno": str(self.pinn_config_json.relative_to(REPO_ROOT)),
                "supervised_ffno": str(self.supervised_config_json.relative_to(REPO_ROOT)),
            },
            "source_model_manifest_json": {
                "pinn_ffno": str(self.pinn_model_manifest_json.relative_to(REPO_ROOT)),
                "supervised_ffno": str(self.supervised_model_manifest_json.relative_to(REPO_ROOT)),
            },
        }


FOUR_BLOCK_NO_REFINER_PAIR = CdiFinalFfnoPair(
    pair_key="four_block_no_refiner",
    output_stem="ffno_final_depth4pair",
    depth_label="depth4_no_refiner",
    claim_boundary=(
        "complete_lines128_cdi_benchmark_plus_uno_extension_"
        "with_final_four_block_no_refiner_ffno_pair"
    ),
    pinn_root=(
        ARTIFACTS_ROOT
        / "2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun"
        / "runs"
        / "ffno_no_refiner_20260506T223454Z"
    ),
    pinn_source_row_id="pinn_ffno",
    supervised_root=(
        ARTIFACTS_ROOT
        / "2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun"
        / "runs"
        / "supervised_ffno_no_refiner_20260506T232535Z"
    ),
    supervised_source_row_id="supervised_ffno",
)

DEPTH24_NO_REFINER_PAIR = CdiFinalFfnoPair(
    pair_key="depth24_no_refiner",
    output_stem="ffno_final_depth24pair",
    depth_label="depth24_no_refiner",
    claim_boundary=(
        "complete_lines128_cdi_benchmark_plus_uno_extension_"
        "with_final_depth24_no_refiner_ffno_pair"
    ),
    pinn_root=(
        ARTIFACTS_ROOT
        / "2026-05-06-cdi-lines128-ffno-depth24-ablation"
        / "runs"
        / "ffno_depth24_20260507T052301Z"
    ),
    pinn_source_row_id="pinn_ffno_depth24",
    supervised_root=(
        ARTIFACTS_ROOT
        / "2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun"
        / "runs"
        / "supervised_ffno_depth24_20260507T192840Z"
    ),
    supervised_source_row_id="supervised_ffno_depth24",
)


PAIR_BY_KEY = {
    FOUR_BLOCK_NO_REFINER_PAIR.pair_key: FOUR_BLOCK_NO_REFINER_PAIR,
    DEPTH24_NO_REFINER_PAIR.pair_key: DEPTH24_NO_REFINER_PAIR,
}


def resolve_cdi_final_ffno_pair(pair_key: str) -> CdiFinalFfnoPair:
    try:
        return PAIR_BY_KEY[pair_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CDI final FFNO pair {pair_key!r}; expected one of {sorted(PAIR_BY_KEY)}"
        ) from exc
