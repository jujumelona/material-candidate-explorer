"""Strict local bundle validation for the pinned AIRS/QHNet research code.

QHNet is not distributed as a versioned inference package.  The sidecar uses
the exact source archive recorded in the integration manifest and an
operator-selected, dataset-specific checkpoint/config pair.  This module
contains only validation and attestation; importing it never imports Torch or
executes checkpoint code.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from .weight_binding import WeightBindingError, directory_inventory_sha256, sha256_file


QHNET_COMPONENT_ID = "qhnet-source"
QHNET_SOURCE_REVISION = "4a16c68a7da707c521019067dec51c227c10de45"
QHNET_ARCHIVE_SHA256 = "221b693b15decd084bb2595ea924d9b59f75ec72f5f2048d7cc1f27c5c56e528"
QHNET_POSITION_SCALE_TO_BOHR = 1.8897261258369282

# Files imported by ``models.get_model``.  Their digests are from the exact
# manifest-pinned commit, not from a mutable branch.
_REQUIRED_SOURCE_FILES = {
    "models/__init__.py": "37259ad24f5301a4da96ebb441a2978ceb222f2bd9200e40b56b2d7480d5a971",
    "models/ori_QHNet_with_bias.py": "bd8ca7927d785a5b70a0a7a27d1108e65e5ad7c2dc828010cd729c386ebdc8a8",
    "models/ori_QHNet_wo_bias.py": "33fb48d7c62841b6c5bdaa654d310a160504bf8e92cf0ca2a524f4c0c75ea865",
    "models/functional.py": "26d0c2bfa3d28ec7540c2de3b57c960ea63c82c13f7f46443cc9fd259b52d021",
    "models/modules/__init__.py": "b13869c16703c2e489c2aa6858995a5db639d2b43b706d760c2e4556e05c1f16",
    "models/modules/exponential_bernstein_radial_basis_functions.py": (
        "2aa52eeb3ae74f02caf78d11f81d046dae6ae69b5b302fdafc94ee6447a5bc70"
    ),
}

_CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "source_revision",
        "model_version",
        "dataset_id",
        "dtype",
        "basis",
        "hamiltonian_unit",
        "position_unit",
        "position_scale_to_bohr",
        "molecular_charge",
        "spin_multiplicity",
        "allowed_atomic_number_sequences",
    }
)


@dataclass(frozen=True, slots=True)
class QHNetSourceAttestation:
    archive_root: Path
    qhnet_root: Path
    archive_sha256: str
    source_inventory_sha256: str


@dataclass(frozen=True, slots=True)
class QHNetRuntimeConfig:
    path: Path
    sha256: str
    model_version: str
    dataset_id: str
    dtype: str
    basis: str
    hamiltonian_unit: str
    molecular_charge: int
    spin_multiplicity: int
    allowed_atomic_number_sequences: tuple[tuple[int, ...], ...]

    @property
    def max_orbital_dimension(self) -> int:
        return max(
            sum(5 if atomic_number <= 2 else 14 for atomic_number in sequence)
            for sequence in self.allowed_atomic_number_sequences
        )


@dataclass(frozen=True, slots=True)
class QHNetBundleAttestation:
    checkpoint_path: Path
    config_path: Path
    checkpoint_sha256: str
    config_sha256: str
    revision: str


def verify_qhnet_source_bundle(path: str | Path) -> QHNetSourceAttestation:
    """Verify bootstrap marker and every executable QHNet source file."""

    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise WeightBindingError("QHNET_SOURCE_PATH must not be a symbolic link")
    archive_root = selected.resolve(strict=True)
    if not archive_root.is_dir():
        raise WeightBindingError("QHNET_SOURCE_PATH must point to a source directory")
    marker_path = archive_root / ".discovery-source.json"
    if not marker_path.is_file() or marker_path.is_symlink():
        raise WeightBindingError(
            f"QHNet bootstrap source marker is missing: {marker_path}"
        )
    marker = _strict_json_file(marker_path, max_bytes=16_384, label="QHNet source marker")
    expected_marker = {
        "schema_version": "1.0",
        "component_id": QHNET_COMPONENT_ID,
        "revision": QHNET_SOURCE_REVISION,
        "sha256": QHNET_ARCHIVE_SHA256,
    }
    if set(marker) - {*expected_marker, "inventory_sha256"} or any(
        marker.get(key) != value for key, value in expected_marker.items()
    ):
        raise WeightBindingError(
            "QHNet source marker does not match the manifest-pinned archive and commit"
        )
    declared_inventory = marker.get("inventory_sha256")
    if not isinstance(declared_inventory, str) or re.fullmatch(
        r"[0-9a-f]{64}", declared_inventory
    ) is None:
        raise WeightBindingError("QHNet source marker inventory_sha256 is required and invalid")
    if directory_inventory_sha256(
        archive_root,
        exclude_names=frozenset({".discovery-source.json"}),
        exclude_directory_names=frozenset(),
    ) != declared_inventory:
        raise WeightBindingError("QHNet archive inventory differs from its bootstrap marker")

    qhnet_root = (archive_root / "OpenDFT" / "QHNet").resolve(strict=True)
    try:
        qhnet_root.relative_to(archive_root)
    except ValueError as exc:
        raise WeightBindingError("QHNet source subdirectory escapes its archive root") from exc
    if not qhnet_root.is_dir() or qhnet_root.is_symlink():
        raise WeightBindingError("manifest-pinned OpenDFT/QHNet source directory is missing")
    for relative, expected_sha256 in _REQUIRED_SOURCE_FILES.items():
        member = _confined_regular_file(qhnet_root, relative)
        actual = sha256_file(member)
        if actual != expected_sha256:
            raise WeightBindingError(
                f"QHNet source file {relative!r} differs from commit {QHNET_SOURCE_REVISION}"
            )
    return QHNetSourceAttestation(
        archive_root=archive_root,
        qhnet_root=qhnet_root,
        archive_sha256=QHNET_ARCHIVE_SHA256,
        source_inventory_sha256=directory_inventory_sha256(qhnet_root),
    )


def load_qhnet_runtime_config(path: str | Path) -> QHNetRuntimeConfig:
    """Load the sidecar's strict dataset/checkpoint interpretation config."""

    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise WeightBindingError("QHNET_CONFIG_PATH must not be a symbolic link")
    resolved = selected.resolve(strict=True)
    if not resolved.is_file():
        raise WeightBindingError("QHNET_CONFIG_PATH must point to a regular JSON file")
    payload = _strict_json_file(resolved, max_bytes=65_536, label="QHNet config")
    unknown = set(payload) - _CONFIG_KEYS
    missing = _CONFIG_KEYS - set(payload)
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise WeightBindingError("QHNet config fields are not exact: " + "; ".join(details))
    if payload["schema_version"] != "1.0":
        raise WeightBindingError("QHNet config schema_version must be '1.0'")
    if payload["source_revision"] != QHNET_SOURCE_REVISION:
        raise WeightBindingError("QHNet config source_revision does not match the manifest")
    model_version = payload["model_version"]
    if model_version not in {"QHNet_w_bias", "QHNet_wo_bias"}:
        raise WeightBindingError(
            "QHNet config model_version must be QHNet_w_bias or QHNet_wo_bias"
        )
    dataset_id = _nonblank(payload["dataset_id"], "dataset_id", max_length=128)
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", dataset_id) is None:
        raise WeightBindingError("QHNet config dataset_id is not a safe identifier")
    dtype = payload["dtype"]
    if dtype not in {"float32", "float64"}:
        raise WeightBindingError("QHNet config dtype must be float32 or float64")
    basis = _nonblank(payload["basis"], "basis", max_length=1_024)
    hamiltonian_unit = _nonblank(
        payload["hamiltonian_unit"], "hamiltonian_unit", max_length=128
    )
    if payload["position_unit"] != "angstrom":
        raise WeightBindingError(
            "QHNet config position_unit must be angstrom; the pinned preprocessing converts it to bohr"
        )
    scale = payload["position_scale_to_bohr"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale):
        raise WeightBindingError("QHNet config position_scale_to_bohr must be finite")
    if float(scale) != QHNET_POSITION_SCALE_TO_BOHR:
        raise WeightBindingError(
            "QHNet config position_scale_to_bohr differs from pinned upstream preprocessing"
        )
    if payload["molecular_charge"] != 0 or isinstance(payload["molecular_charge"], bool):
        raise WeightBindingError("pinned MD17 QHNet checkpoints require molecular_charge=0")
    if payload["spin_multiplicity"] != 1 or isinstance(payload["spin_multiplicity"], bool):
        raise WeightBindingError("pinned MD17 QHNet checkpoints require spin_multiplicity=1")
    sequences_raw = payload["allowed_atomic_number_sequences"]
    if not isinstance(sequences_raw, list) or not 1 <= len(sequences_raw) <= 64:
        raise WeightBindingError(
            "QHNet config allowed_atomic_number_sequences must contain 1 to 64 sequences"
        )
    sequences: list[tuple[int, ...]] = []
    for sequence_raw in sequences_raw:
        if not isinstance(sequence_raw, list) or not 1 <= len(sequence_raw) <= 64:
            raise WeightBindingError("each QHNet atomic-number sequence must contain 1 to 64 atoms")
        sequence: list[int] = []
        for atomic_number in sequence_raw:
            if (
                isinstance(atomic_number, bool)
                or not isinstance(atomic_number, int)
                or not 1 <= atomic_number <= 9
            ):
                raise WeightBindingError(
                    "pinned QHNet supports atomic numbers 1 through 9 only"
                )
            sequence.append(atomic_number)
        sequences.append(tuple(sequence))
    if len(sequences) != len(set(sequences)):
        raise WeightBindingError("QHNet config contains duplicate atomic-number sequences")
    max_orbitals = max(
        sum(5 if atomic_number <= 2 else 14 for atomic_number in sequence)
        for sequence in sequences
    )
    if max_orbitals * max_orbitals > 65_536:
        raise WeightBindingError(
            "QHNet configured Hamiltonian exceeds the sidecar's 65,536-value tensor limit"
        )
    return QHNetRuntimeConfig(
        path=resolved,
        sha256=sha256_file(resolved),
        model_version=model_version,
        dataset_id=dataset_id,
        dtype=dtype,
        basis=basis,
        hamiltonian_unit=hamiltonian_unit,
        molecular_charge=0,
        spin_multiplicity=1,
        allowed_atomic_number_sequences=tuple(sequences),
    )


def attest_qhnet_bundle(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    declared_revision: str | None = None,
) -> QHNetBundleAttestation:
    """Bind both files into one immutable manual-weight revision."""

    checkpoint = Path(checkpoint_path).expanduser()
    config = Path(config_path).expanduser()
    checkpoint_sha256 = sha256_file(checkpoint)
    config_sha256 = sha256_file(config)
    encoded = json.dumps(
        {
            "checkpoint_sha256": checkpoint_sha256,
            "config_sha256": config_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    revision = f"bundle-sha256:{hashlib.sha256(encoded).hexdigest()}"
    declared = (declared_revision or "").strip().lower()
    if declared and declared != revision:
        raise WeightBindingError(
            f"QHNet declared weight revision {declared_revision!r} does not match {revision}"
        )
    return QHNetBundleAttestation(
        checkpoint_path=checkpoint.resolve(strict=True),
        config_path=config.resolve(strict=True),
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        revision=revision,
    )


def _confined_regular_file(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise WeightBindingError("QHNet source member path is not confined")
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise WeightBindingError(f"QHNet source member is a symlink: {cursor}")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise WeightBindingError("QHNet source member escapes the source root") from exc
    if not resolved.is_file():
        raise WeightBindingError(f"QHNet source member is not a regular file: {resolved}")
    return resolved


def _strict_json_file(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    if path.stat().st_size <= 0 or path.stat().st_size > max_bytes:
        raise WeightBindingError(f"{label} is empty or exceeds {max_bytes} bytes")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise WeightBindingError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise WeightBindingError(f"{label} contains non-finite number {value}")

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except WeightBindingError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WeightBindingError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise WeightBindingError(f"{label} must be a JSON object")
    return payload


def _nonblank(value: Any, name: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise WeightBindingError(
            f"QHNet config {name} must be a non-blank string up to {max_length} characters"
        )
    if any(ord(character) < 32 for character in value):
        raise WeightBindingError(f"QHNet config {name} contains control characters")
    return value.strip()


__all__ = [
    "QHNET_ARCHIVE_SHA256",
    "QHNET_COMPONENT_ID",
    "QHNET_POSITION_SCALE_TO_BOHR",
    "QHNET_SOURCE_REVISION",
    "QHNetBundleAttestation",
    "QHNetRuntimeConfig",
    "QHNetSourceAttestation",
    "attest_qhnet_bundle",
    "load_qhnet_runtime_config",
    "verify_qhnet_source_bundle",
]
