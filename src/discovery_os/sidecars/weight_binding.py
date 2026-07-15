"""Immutable local-artifact attestation for isolated model sidecars.

The integration manifest records where a checkpoint came from.  This module
binds that declaration to the local object that a sidecar will actually open.
It intentionally performs no download and never falls back to a network cache.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class WeightBindingError(ValueError):
    """Raised when a local checkpoint cannot prove its declared identity."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise WeightBindingError(f"snapshot marker contains duplicate key {key!r}")
        payload[key] = value
    return payload


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return the SHA-256 of one regular, non-symlink file."""

    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise WeightBindingError(f"checkpoint must not be a symlink: {selected}")
    resolved = selected.resolve(strict=True)
    if not resolved.is_file():
        raise WeightBindingError(f"checkpoint must be a regular non-symlink file: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_inventory_sha256(
    path: str | os.PathLike[str],
    *,
    exclude_names: frozenset[str] = frozenset({".snapshot.json"}),
    exclude_directory_names: frozenset[str] = frozenset({".cache"}),
) -> str:
    """Hash paths, sizes, and bytes for a deterministic directory inventory."""

    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise WeightBindingError(f"checkpoint directory must not be a symlink: {selected}")
    root = selected.resolve(strict=True)
    if not root.is_dir():
        raise WeightBindingError(f"checkpoint must be a regular non-symlink directory: {root}")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if any(part in exclude_directory_names for part in relative.parts[:-1]):
            continue
        if candidate.is_symlink():
            raise WeightBindingError(f"checkpoint inventory contains a symlink: {candidate}")
        if candidate.is_file() and candidate.name not in exclude_names:
            files.append(candidate)
    if not files:
        raise WeightBindingError(f"checkpoint directory contains no files: {root}")
    digest = hashlib.sha256()
    for candidate in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        size = candidate.stat().st_size
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def verify_huggingface_snapshot(
    path: str | os.PathLike[str],
    *,
    repository: str,
    revision: str,
) -> Path:
    """Verify bootstrap's completion marker and return the confined snapshot root."""

    selected = Path(path).expanduser()
    if selected.is_symlink():
        raise WeightBindingError("Hugging Face snapshot path must not be a symlink")
    root = selected.resolve(strict=True)
    if not root.is_dir():
        raise WeightBindingError("Hugging Face snapshot path must be a non-symlink directory")
    marker = root / ".snapshot.json"
    if not marker.is_file() or marker.is_symlink():
        raise WeightBindingError(f"verified snapshot marker is missing: {marker}")
    try:
        payload = json.loads(
            marker.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WeightBindingError(f"snapshot marker is unreadable: {marker}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise WeightBindingError("snapshot marker has an unsupported schema")
    if payload.get("repository") != repository or payload.get("revision") != revision:
        raise WeightBindingError(
            "snapshot marker repository/revision does not match the integration manifest"
        )
    declared_inventory = payload.get("inventory_sha256")
    if not isinstance(declared_inventory, str) or (
        len(declared_inventory) != 64
        or any(char not in "0123456789abcdef" for char in declared_inventory)
    ):
        raise WeightBindingError("snapshot marker is missing a valid inventory_sha256")
    actual_inventory = directory_inventory_sha256(root)
    if actual_inventory != declared_inventory:
        raise WeightBindingError(
            "snapshot file inventory does not match its completion marker"
        )
    return root


def require_snapshot_member(
    snapshot_root: str | os.PathLike[str],
    relative_path: str,
    *,
    kind: str,
) -> Path:
    """Resolve a required member without allowing path escape or symlinks."""

    selected_root = Path(snapshot_root).expanduser()
    if selected_root.is_symlink():
        raise WeightBindingError("verified snapshot root must not be a symlink")
    root = selected_root.resolve(strict=True)
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise WeightBindingError("checkpoint member must be a confined relative path")
    lexical_candidate = root.joinpath(*relative.parts)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise WeightBindingError(f"checkpoint member must not contain a symlink: {current}")
    candidate = lexical_candidate.resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise WeightBindingError("checkpoint member escapes its verified snapshot") from exc
    if kind == "file" and not candidate.is_file():
        raise WeightBindingError(f"required checkpoint file is missing: {candidate}")
    if kind == "directory" and not candidate.is_dir():
        raise WeightBindingError(f"required checkpoint directory is missing: {candidate}")
    return candidate


def attest_file_revision(
    path: str | os.PathLike[str],
    *,
    declared_revision: str | None,
    label: str,
) -> str:
    """Return ``sha256:<digest>`` and reject conflicting SHA-style declarations."""

    actual = f"sha256:{sha256_file(path)}"
    declared = (declared_revision or "").strip().lower()
    if declared.startswith("sha256:") and declared != actual:
        raise WeightBindingError(
            f"{label} declared {declared_revision!r} but the selected file is {actual}"
        )
    return actual


def bound_generator_revision(*, artifact_attestation: str, parameters: dict[str, Any]) -> str:
    """Bind generator-static settings into the effective weight identity."""

    encoded = json.dumps(
        {
            "artifact_attestation": artifact_attestation,
            "runtime_parameters": parameters,
        },
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"bound-sha256:{hashlib.sha256(encoded).hexdigest()}"


__all__ = [
    "WeightBindingError",
    "attest_file_revision",
    "bound_generator_revision",
    "directory_inventory_sha256",
    "require_snapshot_member",
    "sha256_file",
    "verify_huggingface_snapshot",
]
