"""Small, model-neutral records returned by in-process model runtimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from discovery_os.fusion_schemas import FeatureStatus, TensorRole
from discovery_os.schemas import CandidateRepresentation


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Immutable identity bound by an administrator when a sidecar starts."""

    model_id: str
    model_version: str
    adapter_version: str
    code_revision: str
    weight_revision: str
    capabilities: frozenset[Literal["features", "generate"]]
    projection_version: str | None = None
    runtime_parameters_hash: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "model_id",
            "model_version",
            "adapter_version",
            "code_revision",
            "weight_revision",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")
        if not self.capabilities or not self.capabilities.issubset({"features", "generate"}):
            raise ValueError("capabilities must contain features and/or generate")
        if self.runtime_parameters_hash is not None and (
            len(self.runtime_parameters_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.runtime_parameters_hash)
        ):
            raise ValueError("runtime_parameters_hash must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SidecarLimits:
    """Resource limits enforced independently of the model package."""

    max_request_bytes: int = 8 * 1024 * 1024
    max_batch_size: int = 32
    max_concurrency: int = 1
    max_queue_size: int = 2
    request_timeout_seconds: float = 900.0

    def __post_init__(self) -> None:
        if self.max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive")
        if not 1 <= self.max_batch_size <= 1_024:
            raise ValueError("max_batch_size must be between 1 and 1024")
        if not 1 <= self.max_concurrency <= 64:
            raise ValueError("max_concurrency must be between 1 and 64")
        if not 0 <= self.max_queue_size <= 1_024:
            raise ValueError("max_queue_size must be between 0 and 1024")
        if not 0.1 <= self.request_timeout_seconds <= 86_400:
            raise ValueError("request_timeout_seconds must be between 0.1 and 86400")


@dataclass(frozen=True, slots=True)
class PropertyResult:
    property_name: str
    value: float
    unit: str | None = None
    uncertainty: float | None = None
    out_of_domain: bool = False
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ExpertResult:
    """Actual model output before it is projected into the wire schema."""

    values: Any | None
    tensor_role: TensorRole = TensorRole.CUSTOM
    projection_id: str = "raw-model-output-v1"
    entity_type: str | None = None
    entity_ids: tuple[str, ...] = ()
    mask: tuple[bool, ...] = ()
    pooling: Literal["none", "mean", "sum", "cls", "attention", "custom"] = "none"
    normalization: str = "none"
    coordinate_frame: str | None = None
    basis: str | None = None
    unit_semantics: dict[str, str] = field(default_factory=dict)
    properties: tuple[PropertyResult, ...] = ()
    quality_flags: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    status: FeatureStatus = FeatureStatus.SUCCESS


@dataclass(frozen=True, slots=True)
class GeneratedCandidateData:
    """A generated scientific object before lineage/content addressing."""

    representations: tuple[CandidateRepresentation, ...]
    name: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    novelty_rationale: str | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.representations:
            raise ValueError("generated candidate data requires a representation")


@dataclass(frozen=True, slots=True)
class GeneratedBatch:
    candidates: tuple[GeneratedCandidateData, ...]
    warnings: tuple[str, ...] = ()


__all__ = [
    "ExpertResult",
    "GeneratedBatch",
    "GeneratedCandidateData",
    "ModelIdentity",
    "PropertyResult",
    "SidecarLimits",
]
