"""Protocols implemented by specialist encoders and local or remote fusion controllers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .fusion_schemas import (
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    FusionGenerationRequest,
    FusionGenerationResponse,
    FusionOutput,
    FusionRequest,
    FusionRevisionProposal,
    FusionRevisionRequest,
)


@runtime_checkable
class ExpertEncoder(Protocol):
    @property
    def descriptor(self) -> ExpertDescriptor:
        ...

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        ...


@runtime_checkable
class FusionBackend(Protocol):
    def fuse(self, request: FusionRequest) -> FusionOutput:
        ...

    def propose_revision(self, request: FusionRevisionRequest) -> FusionRevisionProposal:
        ...


@runtime_checkable
class FusionCandidateGenerator(Protocol):
    def generate(self, request: FusionGenerationRequest) -> FusionGenerationResponse:
        ...


__all__ = ["ExpertEncoder", "FusionBackend", "FusionCandidateGenerator"]
