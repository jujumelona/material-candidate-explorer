"""Small runtime compatibility helpers shared by isolated model environments."""

from __future__ import annotations

try:  # Python 3.11+
    from enum import StrEnum as StrEnum
except ImportError:  # pragma: no cover - exercised in Python 3.10 sidecars
    from enum import Enum

    class StrEnum(str, Enum):
        """Python 3.10-compatible subset of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return str(self.value)


__all__ = ["StrEnum"]
