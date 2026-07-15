from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from discovery_os.sidecars.errors import ModelExecutionError
from discovery_os.sidecars.experts import ChempropExpert
from discovery_os.sidecars.generators import MatterGenGenerator, ReinventGenerator
from discovery_os.sidecars.weight_binding import (
    WeightBindingError,
    attest_file_revision,
    directory_inventory_sha256,
    require_snapshot_member,
    verify_huggingface_snapshot,
)


def _snapshot(tmp_path: Path) -> tuple[Path, str, str]:
    repository = "example/reviewed-model"
    revision = "a" * 40
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "weights.bin").write_bytes(b"reviewed model bytes")
    cache = root / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "download.json").write_text("mutable cache", encoding="utf-8")
    (root / ".snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "repository": repository,
                "revision": revision,
                "inventory_sha256": directory_inventory_sha256(root),
            }
        ),
        encoding="utf-8",
    )
    return root, repository, revision


def test_verified_snapshot_binds_repository_revision_and_every_model_byte(tmp_path: Path) -> None:
    root, repository, revision = _snapshot(tmp_path)
    assert verify_huggingface_snapshot(
        root, repository=repository, revision=revision
    ) == root.resolve()

    # Download-manager metadata is deliberately outside the model inventory.
    (root / ".cache" / "huggingface" / "download.json").write_text(
        "updated cache", encoding="utf-8"
    )
    verify_huggingface_snapshot(root, repository=repository, revision=revision)

    (root / "weights.bin").write_bytes(b"tampered model bytes")
    with pytest.raises(WeightBindingError, match="inventory"):
        verify_huggingface_snapshot(root, repository=repository, revision=revision)


def test_snapshot_marker_rejects_duplicate_identity_keys(tmp_path: Path) -> None:
    root, repository, revision = _snapshot(tmp_path)
    inventory = directory_inventory_sha256(root)
    (root / ".snapshot.json").write_text(
        '{"schema_version":"1.0","repository":"example/reviewed-model",'
        '"repository":"attacker/model","revision":"'
        + revision
        + '","inventory_sha256":"'
        + inventory
        + '"}',
        encoding="utf-8",
    )
    with pytest.raises(WeightBindingError, match="duplicate key"):
        verify_huggingface_snapshot(root, repository=repository, revision=revision)


def test_manual_checkpoint_revision_is_measured_not_trusted_from_environment(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "task.ckpt"
    checkpoint.write_bytes(b"task checkpoint")
    measured = attest_file_revision(checkpoint, declared_revision=None, label="chemprop")
    assert measured.startswith("sha256:") and len(measured) == 71
    with pytest.raises(WeightBindingError, match="declared"):
        attest_file_revision(
            checkpoint,
            declared_revision="sha256:" + "0" * 64,
            label="chemprop",
        )


def test_snapshot_member_rejects_symlink_indirection(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = root / "weights.bin"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this host")
    with pytest.raises(WeightBindingError, match="symlink"):
        require_snapshot_member(root, "weights.bin", kind="file")


def test_mattergen_rechecks_snapshot_bytes_immediately_before_lazy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "mattergen"
    checkpoint.mkdir()
    weight = checkpoint / "last.ckpt"
    weight.write_bytes(b"attested")
    runtime = MatterGenGenerator(checkpoint_path=str(checkpoint), device="cpu")
    weight.write_bytes(b"changed-after-attestation")
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.require_module",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(ModelExecutionError, match="changed after runtime attestation"):
        runtime._load_model("cpu")


def test_reinvent_rechecks_prior_bytes_immediately_before_lazy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prior = tmp_path / "prior.model"
    prior.write_bytes(b"attested")
    runtime = ReinventGenerator(model_file=str(prior), device="cpu")
    prior.write_bytes(b"changed-after-attestation")
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.shutil.which",
        lambda _name: "reinvent",
    )

    with pytest.raises(ModelExecutionError, match="changed after runtime attestation"):
        runtime._load_model("cpu")


@pytest.mark.parametrize(
    ("changed_artifact", "message"),
    (
        ("prior", "prior bytes changed"),
        ("executable", "executable bytes changed"),
    ),
)
def test_reinvent_rechecks_external_artifacts_before_every_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed_artifact: str,
    message: str,
) -> None:
    prior = tmp_path / "prior.model"
    prior.write_bytes(b"attested-prior")
    executable = tmp_path / ("reinvent.exe" if os.name == "nt" else "reinvent")
    executable.write_bytes(b"attested-executable")
    executable.chmod(0o700)
    runtime = ReinventGenerator(
        model_file=str(prior), executable=str(executable), device="cpu"
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.shutil.which",
        lambda _name: str(executable),
    )
    runtime._ensure_loaded()

    selected = prior if changed_artifact == "prior" else executable
    selected.write_bytes(b"changed-after-first-load")

    # The attestation runs before request fields are consumed or a subprocess
    # starts, so a sentinel is sufficient to exercise the invocation boundary.
    with pytest.raises(ModelExecutionError, match=message):
        runtime.generate(object())  # type: ignore[arg-type]


def test_chemprop_rechecks_checkpoint_immediately_before_lazy_load(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "task.ckpt"
    checkpoint.write_bytes(b"attested")
    runtime = ChempropExpert(
        checkpoint_path=str(checkpoint),
        property_names=("target",),
        property_units=("dimensionless",),
        device="cpu",
    )
    checkpoint.write_bytes(b"changed-after-attestation")

    with pytest.raises(ModelExecutionError, match="changed after runtime attestation"):
        runtime._load_model("cpu")
