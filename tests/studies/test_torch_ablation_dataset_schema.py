from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from tests.studies.ablation_dataset_fixtures import (
    _api,
    _array_sha256,
    _bundle,
    _canonical_probe,
    _file_sha256,
)


@pytest.mark.parametrize(
    "case", ["embedded_with_path", "external_without_path", "none"]
)
def test_rejects_invalid_reference_role_path_combinations(
    tmp_path: Path, case: str
) -> None:
    api = _api()
    descriptor = _bundle(
        tmp_path,
        kind="experimental",
        truth="reference_reconstruction",
        truth_location="embedded_test" if case == "embedded_with_path" else None,
    )
    if case == "embedded_with_path":
        descriptor["reference"] = "test.npz"
        descriptor["reference_sha256"] = descriptor["test_sha256"]
    elif case == "external_without_path":
        descriptor.pop("reference")
        descriptor.pop("reference_sha256")
    else:
        descriptor["truth_location"] = "none"
        descriptor.pop("reference")
        descriptor.pop("reference_sha256")

    with pytest.raises(api.DatasetError, match="reference|truth_location|external_npz"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_loads_standalone_v1_and_resolves_relative_paths_from_descriptor(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, dataset_id="standalone")
    dose = descriptor["dose"]
    path = tmp_path / "dataset.toml"
    path.write_text(
        f'''[schema]
version = 1

[dataset]
id = "standalone"
kind = "synthetic"
format = "npz_mmap"
scale_contract_version = "ci_intensity_v2"
measurement_domain = "count_intensity"
truth = "object_truth"
measurement_key = "diff3d"
probe_key = "probeGuess"
x_key = "xcoords"
y_key = "ycoords"
truth_key = "objectGuess"
truth_location = "embedded_test"
coords_convention = "xy_pixels"
detector_shape = [2, 2]
grouping_max_C = 4
probe_modes = 1
train = "{(tmp_path / "train.npz").as_posix()}"
test = "test.npz"
provenance = "provenance.json"
train_sha256 = "{descriptor["train_sha256"]}"
test_sha256 = "{descriptor["test_sha256"]}"
provenance_sha256 = "{descriptor["provenance_sha256"]}"

[dataset.probe]
source = "tiny_fixture"
calibration = "count_amplitude"
gauge = "physical_count_amplitude"
mask_policy = "model_config"
sha256 = "{descriptor["probe"]["sha256"]}"

[dataset.dose.train]
counts_mean = {dose["train"]["counts_mean"]}
photons_per_image_min = {dose["train"]["photons_per_image_min"]}
photons_per_image_mean = {dose["train"]["photons_per_image_mean"]}
max_observed_count = {dose["train"]["max_observed_count"]}
dtype_max = {dose["train"]["dtype_max"]}
saturation_fraction = {dose["train"]["saturation_fraction"]}

[dataset.dose.test]
counts_mean = {dose["test"]["counts_mean"]}
photons_per_image_min = {dose["test"]["photons_per_image_min"]}
photons_per_image_mean = {dose["test"]["photons_per_image_mean"]}
max_observed_count = {dose["test"]["max_observed_count"]}
dtype_max = {dose["test"]["dtype_max"]}
saturation_fraction = {dose["test"]["saturation_fraction"]}
''',
        encoding="utf-8",
    )

    validated = api.load_standalone_dataset(path)

    assert validated.descriptor.id == "standalone"
    assert validated.descriptor.path_origin == "descriptor_directory"
    assert validated.descriptor.descriptor_path == path.resolve()
    assert validated.descriptor.train == (tmp_path / "train.npz").resolve()
    assert (
        validated.descriptor.path_declarations.train
        == (tmp_path / "train.npz").as_posix()
    )
    assert validated.descriptor.path_declarations.test == "test.npz"

    outside = tmp_path.parent / f"{tmp_path.name}_standalone_outside.npz"
    outside.write_bytes((tmp_path / "test.npz").read_bytes())
    for field in ("train", "test"):
        forged = replace(validated.descriptor, **{field: outside.resolve()})
        with pytest.raises(
            api.DatasetError, match=f"{field}.*origin|declared|identity"
        ):
            api.preflight_dataset(forged)

    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'test = "test.npz"', 'test = "../outside.npz"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(api.DatasetError, match="test.*descriptor directory|escape"):
        api.load_standalone_dataset(path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda d: d.update(extra="value"), "unknown field"),
        (lambda d: d.update(extra={"nested": True}), "unknown field"),
        (lambda d: d.update(kind="simulation"), "kind"),
        (lambda d: d.update(format="npz"), "format"),
        (lambda d: d.update(measurement_domain="counts"), "measurement_domain"),
        (lambda d: d.update(scale_contract_version="legacy_v1"), "supported pair"),
        (lambda d: d.update(train_sha256="A" * 64), "train_sha256"),
        (lambda d: d.update(detector_shape=[2, 3]), "detector_shape"),
        (lambda d: d.update(grouping_max_C=True), "grouping_max_C"),
        (lambda d: d.update(probe_modes=0), "probe_modes"),
        (lambda d: d.update(train=""), "train"),
        (lambda d: d["probe"].update(mask_policy="unknown"), "mask_policy"),
        (lambda d: d["probe"].update(extra="claim"), "unknown field"),
        (lambda d: d["dose"]["train"].update(counts_mean=True), "counts_mean"),
        (
            lambda d: d["dose"]["train"].update(saturation_fraction=1.1),
            "saturation_fraction",
        ),
        (lambda d: d["dose"]["train"].update(extra=1), "unknown field"),
    ],
)
def test_rejects_closed_schema_violations(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None], match: str
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    mutation(descriptor)

    with pytest.raises(api.DatasetError, match=match):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda d: d.pop("truth_key"), "truth_key"),
        (lambda d: d.update(reference="reference.npz"), "reference"),
        (lambda d: d.pop("dose"), "dose"),
        (lambda d: d["probe"].update(calibration="legacy_normalized"), "CI/count"),
        (lambda d: d["probe"].update(gauge="legacy_normalized"), "CI/count"),
    ],
)
def test_rejects_missing_or_forbidden_role_fields(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None], match: str
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    mutation(descriptor)

    with pytest.raises(api.DatasetError, match=match):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


@pytest.mark.parametrize("case", ["mixed", "partial", "identical_split"])
def test_probe_descriptor_hash_identity_is_exactly_one_form(
    tmp_path: Path, case: str
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    probe_hash = descriptor["probe"]["sha256"]
    if case == "mixed":
        descriptor["probe"].update(train_sha256=probe_hash, test_sha256=probe_hash)
    else:
        del descriptor["probe"]["sha256"]
        descriptor["probe"]["train_sha256"] = probe_hash
        if case == "identical_split":
            descriptor["probe"]["test_sha256"] = probe_hash

    with pytest.raises(api.DatasetError, match="probe.*hash|sha256|shorthand"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


def test_probe_identity_distinguishes_equal_values_with_different_dtypes(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    test_path = tmp_path / descriptor["test"]
    with np.load(test_path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    payload["probeGuess"] = payload["probeGuess"].astype(np.complex128)
    np.savez(test_path, **payload)
    descriptor["test_sha256"] = _file_sha256(test_path)
    train_probe_hash = descriptor["probe"].pop("sha256")
    test_probe_hash = _array_sha256(_canonical_probe(payload["probeGuess"]))
    descriptor["probe"].update(
        train_sha256=train_probe_hash,
        test_sha256=test_probe_hash,
    )

    provenance_path = tmp_path / descriptor["provenance"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    record = provenance["datasets"][descriptor["_id"]]
    record["files"]["test"] = descriptor["test_sha256"]
    record["probe"]["test_sha256"] = test_probe_hash
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    descriptor["provenance_sha256"] = _file_sha256(provenance_path)

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.descriptor.probe.sha256 is None
    assert validated.descriptor.probe.train_sha256 == train_probe_hash
    assert validated.descriptor.probe.test_sha256 == test_probe_hash


def test_probe_identity_uses_shorthand_for_truly_identical_arrays(
    tmp_path: Path,
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    assert validated.descriptor.probe.sha256 is not None
    assert validated.descriptor.probe.train_sha256 is None
    assert validated.descriptor.probe.test_sha256 is None


def test_normalized_amplitude_forbids_dose(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path, domain="normalized_amplitude")
    descriptor["dose"] = {"train": {}, "test": {}}

    with pytest.raises(api.DatasetError, match="forbidden"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


@pytest.mark.parametrize("bad_path", ["/tmp/absolute.npz", "../escape.npz"])
def test_checked_paths_must_stay_repository_relative(
    tmp_path: Path, bad_path: str
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    descriptor["train"] = bad_path

    with pytest.raises(api.DatasetError, match="repository-root-relative"):
        api.load_checked_dataset(descriptor.pop("_id"), descriptor, repo_root=tmp_path)


@pytest.mark.parametrize("field", ["train", "test", "provenance"])
def test_checked_preflight_rejects_replaced_path_identity(
    tmp_path: Path, field: str
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    outside_dir = tmp_path.parent / f"{tmp_path.name}_outside"
    outside_dir.mkdir()
    declaration = getattr(validated.descriptor.path_declarations, field)
    assert isinstance(declaration, str)
    outside = outside_dir / declaration
    outside.write_bytes(getattr(validated.descriptor, field).read_bytes())

    forged = replace(validated.descriptor, **{field: outside.resolve()})
    with pytest.raises(api.DatasetError, match=f"{field}.*repository|origin|identity"):
        api.preflight_dataset(forged)


def test_checked_preflight_rejects_retargeted_symlink_escape(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    link = tmp_path / "train_link.npz"
    link.symlink_to(tmp_path / "train.npz")
    descriptor["train"] = link.name
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )

    outside = tmp_path.parent / f"{tmp_path.name}_outside_train.npz"
    outside.write_bytes((tmp_path / "train.npz").read_bytes())
    link.unlink()
    link.symlink_to(outside)

    with pytest.raises(api.DatasetError, match="train.*repository|symlink|origin"):
        api.preflight_dataset(validated.descriptor)


def _relocated_checked_bundle(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    repo_root = tmp_path / "clean-checkout"
    external_root = tmp_path / "source-artifacts" / "datasets_v2"
    descriptor = _bundle(external_root, dataset_id="relocated")
    lexical_root = Path(".artifacts/ci_compatibility/datasets_v2")
    for field in ("train", "test", "provenance"):
        descriptor[field] = (lexical_root / descriptor[field]).as_posix()
    bundle_link = repo_root / lexical_root
    bundle_link.parent.mkdir(parents=True)
    bundle_link.symlink_to(external_root, target_is_directory=True)
    return repo_root, external_root, descriptor


def test_checked_preflight_accepts_single_external_bundle_root_symlink(
    tmp_path: Path,
) -> None:
    api = _api()
    repo_root, external_root, descriptor = _relocated_checked_bundle(tmp_path)

    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=repo_root
    )

    assert validated.descriptor.train == (external_root / "train.npz").resolve()
    assert validated.descriptor.test == (external_root / "test.npz").resolve()
    assert validated.descriptor.provenance == (
        external_root / "provenance.json"
    ).resolve()


def test_relocated_bundle_rejects_nested_file_symlink(tmp_path: Path) -> None:
    api = _api()
    repo_root, external_root, descriptor = _relocated_checked_bundle(tmp_path)
    original_train = external_root / "train.npz"
    nested_target = external_root / "nested-train-target.npz"
    original_train.replace(nested_target)
    original_train.symlink_to(nested_target.name)

    with pytest.raises(api.DatasetError, match="train.*nested symlink"):
        api.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=repo_root
        )


def test_relocated_bundle_rejects_path_under_unrelated_external_symlink(
    tmp_path: Path,
) -> None:
    api = _api()
    repo_root, external_root, descriptor = _relocated_checked_bundle(tmp_path)
    unrelated_link = repo_root / ".artifacts/ci_compatibility/unrelated"
    unrelated_link.symlink_to(external_root, target_is_directory=True)
    descriptor["train"] = ".artifacts/ci_compatibility/unrelated/train.npz"

    with pytest.raises(api.DatasetError, match="train.*configured bundle root"):
        api.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=repo_root
        )


def test_real_bundle_root_still_rejects_symlink_escape(tmp_path: Path) -> None:
    api = _api()
    repo_root = tmp_path / "clean-checkout"
    bundle_root = repo_root / ".artifacts/ci_compatibility/datasets_v2"
    descriptor = _bundle(bundle_root, dataset_id="real-root")
    lexical_root = Path(".artifacts/ci_compatibility/datasets_v2")
    for field in ("train", "test", "provenance"):
        descriptor[field] = (lexical_root / descriptor[field]).as_posix()
    outside = tmp_path / "outside-train.npz"
    (bundle_root / "train.npz").replace(outside)
    (bundle_root / "train.npz").symlink_to(outside)

    with pytest.raises(api.DatasetError, match="train.*repository root"):
        api.load_checked_dataset(
            descriptor.pop("_id"), descriptor, repo_root=repo_root
        )


def test_standalone_rejects_unknown_top_level_and_requires_id(tmp_path: Path) -> None:
    api = _api()
    path = tmp_path / "bad.toml"
    path.write_text("[schema]\nversion=1\n[dataset]\nkind='synthetic'\n[extra]\nx=1\n")

    with pytest.raises(api.DatasetError, match="unknown table"):
        api.load_standalone_dataset(path)

    path.write_text("[schema]\nversion=1\n[dataset]\nkind='synthetic'\n")
    with pytest.raises(api.DatasetError, match="id"):
        api.load_standalone_dataset(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", True),
        ("id", []),
        ("probe_modes", True),
        ("grouping_max_C", 1.0),
        ("detector_shape", [2, 2]),
        ("path_origin", "forged"),
        ("path_base", "not-a-path"),
        ("descriptor_path", "not-a-path"),
    ],
)
def test_preflight_revalidates_forged_descriptor_scalar_types(
    tmp_path: Path, field: str, value: Any
) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    forged = replace(validated.descriptor, **{field: value})

    with pytest.raises(api.DatasetError, match=field):
        api.preflight_dataset(forged)


def test_preflight_revalidates_forged_probe_identity_form(tmp_path: Path) -> None:
    api = _api()
    descriptor = _bundle(tmp_path)
    validated = api.load_checked_dataset(
        descriptor.pop("_id"), descriptor, repo_root=tmp_path
    )
    forged_probe = replace(
        validated.descriptor.probe,
        train_sha256=validated.descriptor.probe.sha256,
    )

    with pytest.raises(api.DatasetError, match="probe.*hash|mixed|shorthand"):
        api.preflight_dataset(replace(validated.descriptor, probe=forged_probe))
