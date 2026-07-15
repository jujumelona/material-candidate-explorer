from __future__ import annotations

from collections.abc import Callable

import pytest

from discovery_os.hashing import candidate_content_hash
from discovery_os.schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    ObjectiveDirection,
    PropertyObjective,
    RepresentationKind,
    ToolCall,
)


@pytest.fixture
def candidate_factory() -> Callable[..., Candidate]:
    def make_candidate(
        *,
        candidate_id: str = "MOL-001",
        candidate_type: CandidateType = CandidateType.SMALL_MOLECULE,
        domain: DiscoveryDomain | None = None,
        value: str = "CCO",
        representation_kind: RepresentationKind | None = None,
        version: int = 1,
    ) -> Candidate:
        if domain is None:
            domain = (
                DiscoveryDomain.MEDICINAL_CHEMISTRY
                if candidate_type == CandidateType.SMALL_MOLECULE
                else DiscoveryDomain.GENERAL_MATERIALS
            )
        if representation_kind is None:
            representation_kind = (
                RepresentationKind.SMILES
                if candidate_type == CandidateType.SMALL_MOLECULE
                else RepresentationKind.CHEMICAL_FORMULA
            )
        candidate = Candidate(
            candidate_id=candidate_id,
            candidate_type=candidate_type,
            domain=domain,
            name=f"fixture {candidate_id}",
            representations=[
                CandidateRepresentation(kind=representation_kind, value=value)
            ],
            attributes={"fixture": True},
            provenance={"source": "pytest"},
        )
        reference = CandidateRef(
            candidate_id=candidate_id,
            version=version,
            content_hash=candidate_content_hash(candidate),
        )
        return candidate.model_copy(update={"candidate_ref": reference})

    return make_candidate


@pytest.fixture
def goal_factory() -> Callable[..., DiscoveryGoal]:
    def make_goal(
        *,
        domain: DiscoveryDomain = DiscoveryDomain.MEDICINAL_CHEMISTRY,
        candidate_types: list[CandidateType] | None = None,
    ) -> DiscoveryGoal:
        if candidate_types is None:
            candidate_types = [
                CandidateType.SMALL_MOLECULE
                if domain == DiscoveryDomain.MEDICINAL_CHEMISTRY
                else CandidateType.COMPOSITION
            ]
        return DiscoveryGoal(
            goal_id=f"GOAL-{domain.value}",
            domain=domain,
            title="Test discovery goal",
            scientific_question="Can a candidate satisfy the target property?",
            objectives=[
                PropertyObjective(
                    property_name="target_property",
                    direction=ObjectiveDirection.MAXIMIZE,
                )
            ],
            validation_profile_id=f"{domain.value}-v1",
            candidate_types=candidate_types,
            max_cycles=2,
        )

    return make_goal


@pytest.fixture
def tool_call_factory() -> Callable[..., ToolCall]:
    def make_call(**overrides: object) -> ToolCall:
        values: dict[str, object] = {
            "call_id": "CALL-001",
            "tool_name": "common_rules",
            "operation": "validate_candidate",
            "candidate_ids": ["MOL-001"],
            "requested_properties": ["representation_valid"],
            "conditions": {},
            "evidence_kind": "computational",
            "method_class": "rule_based",
            "fidelity": "cheap",
            "priority": 1.0,
            "reason": "Run an allow-listed deterministic validator.",
            "max_runtime_seconds": 30,
            "resource_budget": {},
        }
        values.update(overrides)
        return ToolCall.model_validate(values)

    return make_call
