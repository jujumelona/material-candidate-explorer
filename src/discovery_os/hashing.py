"""Deterministic hashing helpers used by caching and provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def jsonable(value: Any) -> Any:
    """Convert supported values to a canonical JSON-compatible structure."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=jsonable,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def bytes_hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def candidate_content_hash(candidate: Any) -> str:
    """Hash candidate scientific content without recursively hashing its reference."""

    if not isinstance(candidate, BaseModel):
        raise TypeError("candidate must be a Pydantic model")
    payload = candidate.model_dump(mode="json", exclude={"candidate_ref"})
    # Added in schema 1.0 as a backward-compatible lineage strengthening.  An
    # empty default must not invalidate hashes created before the field existed.
    if not payload.get("parent_candidate_refs"):
        payload.pop("parent_candidate_refs", None)
    return stable_hash(payload)


def crystal_fingerprint(structure: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a crystallographic fingerprint without changing candidate hashing.

    pymatgen remains an optional dependency and is imported only when this helper
    is called.  See :mod:`discovery_os.crystal_identity` for tolerance controls.
    """

    from .crystal_identity import crystal_structure_fingerprint

    return crystal_structure_fingerprint(structure, **kwargs)


def canonical_structure_hash(structure: Any, **kwargs: Any) -> str:
    """Return the scientific crystal-identity hash for a periodic structure."""

    from .crystal_identity import canonical_structure_hash as _canonical_structure_hash

    return _canonical_structure_hash(structure, **kwargs)
