"""Static allow-list for specialist feature encoders."""

from __future__ import annotations

from .fusion_protocols import ExpertEncoder
from .fusion_schemas import ExpertDescriptor, ScientificModality
from .hashing import stable_hash
from .schemas import Candidate


class ExpertRegistry:
    def __init__(self) -> None:
        self._encoders: dict[str, ExpertEncoder] = {}
        self._descriptors: dict[str, ExpertDescriptor] = {}

    def register(self, encoder: ExpertEncoder, *, replace: bool = False) -> None:
        descriptor = ExpertDescriptor.model_validate_json(
            encoder.descriptor.model_dump_json(),
            strict=True,
        )
        if descriptor.expert_id in self._encoders and not replace:
            raise ValueError(f"expert {descriptor.expert_id!r} is already registered")
        self._encoders[descriptor.expert_id] = encoder
        self._descriptors[descriptor.expert_id] = descriptor

    def get(self, expert_id: str) -> ExpertEncoder:
        try:
            return self._encoders[expert_id]
        except KeyError as exc:
            raise KeyError(f"expert {expert_id!r} is not in the allow-list") from exc

    def describe(self, *, available_only: bool = False) -> list[ExpertDescriptor]:
        rows = [
            ExpertDescriptor.model_validate_json(item.model_dump_json(), strict=True)
            for item in self._descriptors.values()
        ]
        if available_only:
            rows = [item for item in rows if item.available]
        return sorted(rows, key=lambda item: item.expert_id)

    def compatible(
        self,
        candidate: Candidate,
        *,
        modality: ScientificModality | None = None,
        available_only: bool = True,
    ) -> list[ExpertEncoder]:
        representation_kinds = {item.kind for item in candidate.representations}
        rows: list[ExpertEncoder] = []
        for expert_id, encoder in self._encoders.items():
            descriptor = self._descriptors[expert_id]
            if available_only and not descriptor.available:
                continue
            if candidate.candidate_type not in descriptor.supported_candidate_types:
                continue
            if not representation_kinds.intersection(descriptor.supported_representations):
                continue
            if modality is not None and modality not in descriptor.modalities:
                continue
            rows.append(encoder)
        return sorted(
            rows,
            key=lambda item: next(
                expert_id
                for expert_id, registered in self._encoders.items()
                if registered is item
            ),
        )

    def bound_descriptor(self, encoder: ExpertEncoder) -> ExpertDescriptor:
        """Return the immutable registration snapshot and detect descriptor drift."""

        expert_id = encoder.descriptor.expert_id
        try:
            expected = self._descriptors[expert_id]
        except KeyError as exc:
            raise ValueError("encoder descriptor no longer identifies a registered expert") from exc
        if stable_hash(encoder.descriptor) != stable_hash(expected):
            raise ValueError(f"expert {expert_id!r} mutated its registered descriptor")
        return ExpertDescriptor.model_validate_json(expected.model_dump_json(), strict=True)

    def __contains__(self, expert_id: str) -> bool:
        return expert_id in self._encoders


__all__ = ["ExpertRegistry"]
