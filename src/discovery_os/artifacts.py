"""Confined artifact storage for tool results and reports."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .hashing import bytes_hash


_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


class ArtifactStore:
    """Writes artifacts only below a configured root.

    Model output never becomes an absolute path or an executable command.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def safe_component(value: str) -> str:
        component = _SAFE_COMPONENT.sub("_", value).strip("._")
        if not component:
            raise ValueError("artifact path component is empty")
        return component[:160]

    def resolve(self, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("artifact path must be relative and cannot traverse parents")
        result = (self.root / relative).resolve()
        if result != self.root and self.root not in result.parents:
            raise ValueError("artifact path escapes the configured root")
        return result

    def write_bytes(self, relative_path: str | Path, payload: bytes) -> tuple[str, str]:
        path = self.resolve(relative_path)
        digest = bytes_hash(payload)
        if path.exists():
            existing_digest = bytes_hash(path.read_bytes())
            if existing_digest != digest:
                raise FileExistsError(
                    f"immutable artifact path already contains different content: {relative_path}"
                )
            return path.relative_to(self.root).as_posix(), digest
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        return path.relative_to(self.root).as_posix(), digest

    def write_json(self, relative_path: str | Path, value: Any) -> tuple[str, str]:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=lambda item: item.model_dump(mode="json"),
        ).encode("utf-8")
        return self.write_bytes(relative_path, payload)

    def read_bytes(
        self,
        relative_path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> bytes:
        """Read a confined artifact and optionally verify its content hash."""

        path = self.resolve(relative_path)
        payload = path.read_bytes()
        digest = bytes_hash(payload)
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"artifact hash mismatch for {relative_path}")
        return payload

    def read_json(
        self,
        relative_path: str | Path,
        *,
        expected_sha256: str | None = None,
    ) -> Any:
        payload = self.read_bytes(relative_path, expected_sha256=expected_sha256)
        return json.loads(payload.decode("utf-8"))
