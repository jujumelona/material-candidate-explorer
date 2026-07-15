from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from discovery_os.artifacts import ArtifactStore
from discovery_os.evidence_fusion import (
    MATTERGEN_SUPPORTED_CONDITIONS,
    EvidenceDrivenFusionBackend,
)
from discovery_os.fusion_registry import ExpertRegistry
from discovery_os.fusion_runtime import FusionRuntime
from discovery_os.fusion_schemas import (
    ContentArtifactRef,
    DiagnosticProperty,
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    ExpertProvenance,
    FeatureSemantics,
    FeatureStatus,
    FusionDecisionContext,
    FusionFeatureInput,
    FusionRequest,
    FusionRevisionRequest,
    NumericTensor,
    ScientificModality,
    ScientificWorkspace,
    TensorDType,
    TensorRole,
    UnifiedLatentStateRef,
    WorkspaceEntity,
    WorkspaceEntityRole,
    WorkspaceMode,
)
from discovery_os.hashing import candidate_content_hash, stable_hash
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
)


def _candidate(
    candidate_id: str = "li2o",
    *,
    formula: str | None = "Li2O",
) -> Candidate:
    if formula is None:
        candidate_type = CandidateType.SMALL_MOLECULE
        domain = DiscoveryDomain.MEDICINAL_CHEMISTRY
        representation = CandidateRepresentation(
            kind=RepresentationKind.SMILES,
            value="CCO",
            canonical=True,
        )
    else:
        candidate_type = CandidateType.COMPOSITION
        domain = DiscoveryDomain.GENERAL_MATERIALS
        representation = CandidateRepresentation(
            kind=RepresentationKind.CHEMICAL_FORMULA,
            value=formula,
            canonical=True,
        )
    candidate = Candidate(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        domain=domain,
        representations=[representation],
    )
    return candidate.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate_id,
                version=1,
                content_hash=candidate_content_hash(candidate),
            )
        }
    )


def _goal(
    *objectives: PropertyObjective,
    molecular: bool = False,
) -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="evidence-fusion-goal",
        domain=(
            DiscoveryDomain.MEDICINAL_CHEMISTRY
            if molecular
            else DiscoveryDomain.GENERAL_MATERIALS
        ),
        title="Evidence-driven fusion",
        scientific_question="Which deterministic MatterGen condition should be explored?",
        objectives=list(objectives),
        validation_profile_id=(
            "medicinal-chemistry-v1" if molecular else "general-materials-v1"
        ),
        candidate_types=[
            CandidateType.SMALL_MOLECULE if molecular else CandidateType.COMPOSITION
        ],
    )


def _workspace(candidate: Candidate, context: Candidate | None = None) -> ScientificWorkspace:
    entities = [
        WorkspaceEntity(
            entity_id="primary",
            role=WorkspaceEntityRole.PRIMARY_CANDIDATE,
            candidate_ref=candidate.candidate_ref,
        )
    ]
    if context is not None:
        entities.append(
            WorkspaceEntity(
                entity_id="context",
                role=WorkspaceEntityRole.CONTEXT,
                candidate_ref=context.candidate_ref,
            )
        )
    return ScientificWorkspace(
        workspace_id="evidence-workspace",
        primary_entity_id="primary",
        entities=entities,
    )


def _feature(
    candidate: Candidate,
    *,
    feature_id: str,
    expert_id: str,
    value: float,
    unit: str | None = "eV/atom",
    property_name: str = "energy_above_hull",
    entity_id: str = "primary",
    shape: int = 2,
    tensor_offset: float = 0.0,
    status: FeatureStatus = FeatureStatus.SUCCESS,
) -> FusionFeatureInput:
    return FusionFeatureInput(
        feature_id=feature_id,
        workspace_entity_id=entity_id,
        payload=ExpertFeaturePayload(
            workspace_entity_id=entity_id,
            candidate_ref=candidate.candidate_ref,
            expert_id=expert_id,
            modality=ScientificModality.CRYSTAL_MATERIAL,
            feature_space=f"{expert_id}-private-space",
            status=status,
            tensor=NumericTensor(
                shape=[shape],
                values=[tensor_offset + float(index) for index in range(shape)],
            ),
            semantics=FeatureSemantics(
                tensor_role=TensorRole.GLOBAL_EMBEDDING,
                projection_id=f"{expert_id}-private-space",
                pooling="mean",
                normalization="expert-private",
            ),
            properties=[
                DiagnosticProperty(
                    property_name=property_name,
                    value=value,
                    unit=unit,
                    source=expert_id,
                )
            ],
            provenance=ExpertProvenance(
                expert_id=expert_id,
                adapter_version="1.0.0",
                model_version="fixture-1",
                code_revision="fixture-code",
                weight_revision="fixture-weight",
                parameters_hash=stable_hash({"expert": expert_id}),
                seed=7,
            ),
        ),
    )


def _request(
    *,
    candidate: Candidate,
    goal: DiscoveryGoal,
    features: list[FusionFeatureInput],
    context_candidate: Candidate | None = None,
    decision_context: FusionDecisionContext | None = None,
    failed: list[str] | None = None,
    missing: list[str] | None = None,
    cycle: int = 2,
) -> FusionRequest:
    return FusionRequest(
        goal=goal,
        candidate_ref=candidate.candidate_ref,
        workspace=_workspace(candidate, context_candidate),
        workspace_mode=WorkspaceMode.ON,
        cycle=cycle,
        seed=7,
        features=features,
        decision_context=decision_context or FusionDecisionContext(),
        failed_expert_ids=failed or [],
        missing_expert_ids=missing or [],
    )


def _revision_request(
    request: FusionRequest,
    output,
    *,
    candidate: Candidate | None = None,
    state_backend_id: str = "evidence-rule-fusion",
    latent: NumericTensor | None = None,
) -> FusionRevisionRequest:
    latent = latent or output.latent
    state = UnifiedLatentStateRef(
        state_id="state-evidence",
        state_version=1,
        candidate_ref=request.candidate_ref,
        workspace_id=request.workspace.workspace_id,
        workspace_entities=request.workspace.entities,
        workspace_relations=request.workspace.relations,
        cycle=request.cycle,
        latent_artifact=ContentArtifactRef(
            artifact_id="latent-evidence",
            relative_path="fusion/latents/evidence.json",
            sha256="a" * 64,
            media_type="application/json",
            byte_size=1,
        ),
        latent_content_hash=stable_hash(latent),
        dtype=latent.dtype,
        shape=latent.shape,
        source_feature_ids=output.used_feature_ids,
        goal_hash=stable_hash(request.goal),
        seed=request.seed,
        backend_id=state_backend_id,
        backend_version="1.0.0",
        code_revision="deterministic-evidence-controller-v1",
        weight_revision="no-learned-weights",
    )
    # FusionRequest stores only the primary ref. Tests use the matching fixture
    # unless a non-default candidate is supplied explicitly.
    materialized = candidate or _candidate(request.candidate_ref.candidate_id)
    if materialized.candidate_ref != request.candidate_ref:
        raise AssertionError("test candidate fixture does not match request")
    return FusionRevisionRequest(
        goal=request.goal,
        candidate=materialized,
        state=state,
        latent=latent,
        features=request.features,
        decision_context=request.decision_context,
    )


def test_fuse_records_primary_evidence_state_without_averaging_embeddings() -> None:
    candidate = _candidate()
    context_candidate = _candidate("context", formula="NaCl")
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        )
    )
    features = [
        _feature(
            candidate,
            feature_id="matter",
            expert_id="mattersim",
            value=0.12,
            shape=2,
            tensor_offset=-1000.0,
        ),
        _feature(
            candidate,
            feature_id="chg",
            expert_id="chgnet",
            value=0.10,
            shape=5,
            tensor_offset=1_000_000.0,
        ),
        _feature(
            candidate,
            feature_id="partial",
            expert_id="partial-expert",
            value=-999.0,
            shape=7,
            status=FeatureStatus.PARTIAL,
        ),
        _feature(
            context_candidate,
            feature_id="context-feature",
            expert_id="context-expert",
            value=999.0,
            entity_id="context",
            shape=9,
        ),
    ]
    request = _request(
        candidate=candidate,
        goal=goal,
        features=features,
        context_candidate=context_candidate,
        failed=["failed-expert"],
        missing=["missing-expert"],
        decision_context=FusionDecisionContext(
            guidance_alpha=0.7,
            previous_objective_improvement=0.2,
            structural_collapse_rate=0.25,
            exploration_branch="stability",
        ),
    )

    output = EvidenceDrivenFusionBackend().fuse(request)

    assert output.latent.dtype == TensorDType.FLOAT64
    assert output.latent.shape == [8]
    assert output.latent.values == pytest.approx(
        [2.0, 2.0, 3.0, -0.12, 1.0 / 6.0, 1.0, 0.25, 0.7]
    )
    assert output.used_feature_ids == ["matter", "chg", "partial"]
    assert output.ignored_feature_ids == ["context-feature"]
    assert output.backend_id == "evidence-rule-fusion"
    assert output.weight_revision == "no-learned-weights"
    assert any("not combined" in warning for warning in output.warnings)
    assert any("Partial" in warning for warning in output.warnings)

    changed_tensors = request.model_copy(
        update={
            "features": [
                feature.model_copy(
                    update={
                        "payload": feature.payload.model_copy(
                            update={
                                "tensor": NumericTensor(shape=[1], values=[42.0])
                            }
                        )
                    }
                )
                for feature in request.features
            ]
        }
    )
    assert EvidenceDrivenFusionBackend().fuse(changed_tensors).latent == output.latent


@pytest.mark.parametrize(
    ("branch", "expected_hull"),
    [
        ("stability", 0.0),
        ("target_property", 0.03),
        ("novelty", 0.08),
        ("expert_disagreement", 0.03),
        ("pareto", 0.05),
    ],
)
def test_revision_uses_branch_specific_hull_and_preserves_explicit_targets(
    branch: str,
    expected_hull: float,
) -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        ),
        PropertyObjective(
            property_name="space_group",
            direction=ObjectiveDirection.TARGET,
            target_value=225,
        ),
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="m", expert_id="mattersim", value=0.01),
            _feature(candidate, feature_id="c", expert_id="chgnet", value=0.01),
        ],
        decision_context=FusionDecisionContext(exploration_branch=branch),
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)

    proposal = backend.propose_revision(_revision_request(request, output, candidate=candidate))
    changes = {item.property_name: item for item in proposal.desired_changes}

    assert set(changes).issubset(MATTERGEN_SUPPORTED_CONDITIONS)
    assert changes["energy_above_hull"].target_value == expected_hull
    assert changes["space_group"].target_value == 225
    if branch == "expert_disagreement":
        assert proposal.confidence <= 0.2
        assert any("additional expert" in note for note in proposal.safety_notes)


def test_unstable_hull_focuses_zero_and_mev_evidence_is_converted() -> None:
    candidate = _candidate()
    objective = PropertyObjective(
        property_name="energy_above_hull",
        direction=ObjectiveDirection.MINIMIZE,
        unit="eV/atom",
    )
    goal = _goal(objective)
    backend = EvidenceDrivenFusionBackend()
    unstable = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="m", expert_id="mattersim", value=0.20),
            _feature(candidate, feature_id="c", expert_id="chgnet", value=0.15),
        ],
        decision_context=FusionDecisionContext(exploration_branch="novelty"),
    )
    unstable_output = backend.fuse(unstable)
    unstable_proposal = backend.propose_revision(
        _revision_request(unstable, unstable_output)
    )
    assert {
        item.property_name: item.target_value
        for item in unstable_proposal.desired_changes
    }["energy_above_hull"] == 0.0

    mixed_units = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="m", expert_id="mattersim", value=0.01),
            _feature(
                candidate,
                feature_id="c",
                expert_id="chgnet",
                value=10.0,
                unit="meV/atom",
            ),
        ],
    )
    mixed_output = backend.fuse(mixed_units)
    mixed_proposal = backend.propose_revision(
        _revision_request(mixed_units, mixed_output)
    )
    mixed_changes = {
        item.property_name: item for item in mixed_proposal.desired_changes
    }
    assert mixed_changes["energy_above_hull"].target_value == 0.05


def test_hull_unitless_mixed_panel_fails_closed_and_mev_goal_is_canonicalized() -> None:
    candidate = _candidate()
    backend = EvidenceDrivenFusionBackend()
    unitless_goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit=None,
        )
    )
    mixed = _request(
        candidate=candidate,
        goal=unitless_goal,
        features=[
            _feature(candidate, feature_id="ev", expert_id="ev", value=0.01),
            _feature(
                candidate,
                feature_id="mev",
                expert_id="mev",
                value=10.0,
                unit="meV/atom",
            ),
        ],
    )
    mixed_output = backend.fuse(mixed)
    mixed_proposal = backend.propose_revision(_revision_request(mixed, mixed_output))

    assert mixed_output.latent.values[3:5] == [0.0, 0.0]
    assert all(
        item.property_name != "energy_above_hull"
        for item in mixed_proposal.desired_changes
    )

    mev_goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.TARGET,
            target_value=50.0,
            unit="meV/atom",
        )
    )
    mev_request = _request(
        candidate=candidate,
        goal=mev_goal,
        features=[
            _feature(
                candidate,
                feature_id="mev-only",
                expert_id="mattersim",
                value=100.0,
                unit="meV/atom",
            )
        ],
    )
    mev_output = backend.fuse(mev_request)
    mev_proposal = backend.propose_revision(
        _revision_request(mev_request, mev_output)
    )

    assert mev_output.latent.values[3] == pytest.approx(-0.05)
    assert mev_proposal.desired_changes[0].target_value == pytest.approx(0.05)


def test_invalid_hull_goal_unit_cannot_emit_numeric_hull_target() -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.TARGET,
            target_value=1.0,
            unit="kJ/mol",
        )
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="hull", expert_id="mattersim", value=0.01)
        ],
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    proposal = backend.propose_revision(_revision_request(request, output))

    assert output.latent.values[3] == 0.0
    assert all(
        item.property_name != "energy_above_hull"
        for item in proposal.desired_changes
    )


def test_condition_targets_are_canonical_and_type_checked() -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="chemical_system",
            direction=ObjectiveDirection.TARGET,
            target_value="O-Li",
        ),
        PropertyObjective(
            property_name="space_group",
            direction=ObjectiveDirection.TARGET,
            target_value=231,
        ),
        PropertyObjective(
            property_name="dft_band_gap",
            direction=ObjectiveDirection.TARGET,
            target_value="wide",
        ),
        PropertyObjective(
            property_name="ml_bulk_modulus",
            direction=ObjectiveDirection.TARGET,
            target_value=120,
        ),
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(
                candidate,
                feature_id="diagnostic",
                expert_id="mattersim",
                value=1.0,
                unit=None,
                property_name="unrelated_score",
            )
        ],
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    proposal = backend.propose_revision(_revision_request(request, output))
    targets = {
        item.property_name: item.target_value for item in proposal.desired_changes
    }

    assert targets == {"chemical_system": "Li-O", "ml_bulk_modulus": 120.0}


@pytest.mark.parametrize(("target", "accepted"), [(225, True), (0, False), (231, False), (225.0, False), ("225", False)])
def test_space_group_requires_integer_between_one_and_230(
    target: object,
    accepted: bool,
) -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="space_group",
            direction=ObjectiveDirection.TARGET,
            target_value=target,
        )
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(
                candidate,
                feature_id="diagnostic",
                expert_id="mattersim",
                value=1.0,
                property_name="unrelated_score",
            )
        ],
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    proposal = backend.propose_revision(_revision_request(request, output))
    space_groups = [
        item.target_value
        for item in proposal.desired_changes
        if item.property_name == "space_group"
    ]

    assert space_groups == ([225] if accepted else [])


def test_same_expert_conflicting_routes_do_not_create_cross_expert_disagreement() -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        )
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="a1", expert_id="same", value=0.0),
            _feature(candidate, feature_id="a2", expert_id="same", value=100.0),
            _feature(candidate, feature_id="b", expert_id="other", value=0.02),
        ],
    )

    output = EvidenceDrivenFusionBackend().fuse(request)

    assert output.latent.values[1] == 2.0
    assert output.latent.values[3] == pytest.approx(-0.02)
    assert output.latent.values[4] == 0.0


def test_revision_uses_only_primary_features_cited_by_latent_state() -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        )
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="source", expert_id="source", value=0.01)
        ],
        decision_context=FusionDecisionContext(exploration_branch="novelty"),
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    revision = _revision_request(request, output)
    injected = revision.model_copy(
        update={
            "features": [
                *revision.features,
                _feature(
                    candidate,
                    feature_id="injected-one",
                    expert_id="injected-one",
                    value=0.8,
                ),
                _feature(
                    candidate,
                    feature_id="injected-two",
                    expert_id="injected-two",
                    value=0.9,
                ),
            ]
        }
    )

    proposal = backend.propose_revision(injected)

    assert {
        item.property_name: item.target_value for item in proposal.desired_changes
    }["energy_above_hull"] == 0.08


def test_revision_rejects_context_feature_cited_as_latent_source() -> None:
    candidate = _candidate()
    context_candidate = _candidate("context", formula="NaCl")
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        )
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        context_candidate=context_candidate,
        features=[
            _feature(candidate, feature_id="primary", expert_id="primary", value=0.1),
            _feature(
                context_candidate,
                feature_id="context",
                expert_id="context",
                value=0.0,
                entity_id="context",
            ),
        ],
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    revision = _revision_request(request, output)
    foreign_state = revision.state.model_copy(
        update={"source_feature_ids": ["context"]}
    )
    foreign_source = revision.model_copy(update={"state": foreign_state})

    with pytest.raises(ValueError, match="source features.*primary"):
        backend.propose_revision(foreign_source)


@pytest.mark.parametrize(
    "context",
    [
        FusionDecisionContext(
            guidance_alpha=0.6,
            previous_objective_improvement=0.2,
            structural_collapse_rate=0.1,
        ),
        FusionDecisionContext(
            guidance_alpha=0.4,
            previous_objective_improvement=-0.2,
            structural_collapse_rate=0.1,
        ),
        FusionDecisionContext(
            guidance_alpha=0.4,
            previous_objective_improvement=0.2,
            structural_collapse_rate=0.2,
        ),
    ],
)
def test_revision_rejects_decision_context_that_does_not_match_latent(
    context: FusionDecisionContext,
) -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        )
    )
    original_context = FusionDecisionContext(
        guidance_alpha=0.4,
        previous_objective_improvement=0.2,
        structural_collapse_rate=0.1,
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="source", expert_id="source", value=0.1)
        ],
        decision_context=original_context,
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    revision = _revision_request(request, output).model_copy(
        update={"decision_context": context}
    )

    with pytest.raises(ValueError, match="latent controls.*decision context"):
        backend.propose_revision(revision)


def test_missing_experts_count_as_non_success_and_reduce_confidence() -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        )
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="source", expert_id="source", value=0.1)
        ],
        missing=["missing-a", "missing-b", "missing-c"],
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    proposal = backend.propose_revision(_revision_request(request, output))

    assert output.latent.values[1:3] == [1.0, 3.0]
    assert proposal.confidence == pytest.approx(0.1875)
    assert any("non-successful evaluator" in warning for warning in output.warnings)


def test_non_hull_expert_extrema_are_not_invented_as_generation_targets() -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="dft_band_gap",
            direction=ObjectiveDirection.MAXIMIZE,
            unit="eV",
        )
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(
                candidate,
                feature_id="band-gap",
                expert_id="mattersim",
                value=2.0,
                unit="eV",
                property_name="dft_band_gap",
            )
        ],
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    proposal = backend.propose_revision(_revision_request(request, output))

    assert all(
        item.property_name != "dft_band_gap"
        for item in proposal.desired_changes
    )


def test_request_rejects_contradictory_runtime_owned_expert_metadata() -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
        )
    )
    feature = _feature(candidate, feature_id="m", expert_id="mattersim", value=0.1)

    with pytest.raises(ValidationError, match="successful primary experts"):
        _request(
            candidate=candidate,
            goal=goal,
            features=[feature],
            failed=["mattersim"],
        )


def test_revision_rejects_foreign_or_malformed_controller_latent() -> None:
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        )
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[
            _feature(candidate, feature_id="m", expert_id="mattersim", value=0.1)
        ],
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)

    foreign = _revision_request(request, output, state_backend_id="other-backend")
    with pytest.raises(ValueError, match="not produced"):
        backend.propose_revision(foreign)

    malformed_latent = NumericTensor(
        dtype=TensorDType.FLOAT32,
        shape=[2],
        values=[2.0, 1.0],
    )
    malformed = _revision_request(request, output, latent=malformed_latent)
    with pytest.raises(ValueError, match="float64.*shape"):
        backend.propose_revision(malformed)


class _RuntimeEncoder:
    def __init__(self, expert_id: str, *, fail: bool = False) -> None:
        self.fail = fail
        self._descriptor = ExpertDescriptor(
            expert_id=expert_id,
            display_name=expert_id,
            adapter_version="1.0.0",
            modalities=[ScientificModality.CRYSTAL_MATERIAL],
            supported_candidate_types=[CandidateType.COMPOSITION],
            supported_representations=[RepresentationKind.CHEMICAL_FORMULA],
            feature_spaces=[f"{expert_id}-private-space"],
        )

    @property
    def descriptor(self) -> ExpertDescriptor:
        return self._descriptor

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        if self.fail:
            raise RuntimeError("fixture evaluator failed")
        return _feature(
            request.candidate,
            feature_id="unused",
            expert_id=self.descriptor.expert_id,
            value=0.1,
        ).payload.model_copy(
            update={
                "workspace_entity_id": request.workspace_entity_id,
                "feature_space": request.feature_space,
                "provenance": _feature(
                    request.candidate,
                    feature_id="unused-again",
                    expert_id=self.descriptor.expert_id,
                    value=0.1,
                ).payload.provenance.model_copy(update={"seed": request.seed}),
            }
        )


def test_runtime_supplies_primary_failure_count_and_decision_context(tmp_path: Path) -> None:
    registry = ExpertRegistry()
    registry.register(_RuntimeEncoder("success"))
    registry.register(_RuntimeEncoder("failed", fail=True))
    runtime = FusionRuntime(
        registry,
        EvidenceDrivenFusionBackend(),
        ArtifactStore(tmp_path),
    )
    candidate = _candidate()
    goal = _goal(
        PropertyObjective(
            property_name="energy_above_hull",
            direction=ObjectiveDirection.MINIMIZE,
            unit="eV/atom",
        )
    )

    report = runtime.update(
        goal=goal,
        candidate=candidate,
        cycle=4,
        seed=9,
        workspace_mode=WorkspaceMode.ON,
        expert_ids=["success", "failed"],
        decision_context=FusionDecisionContext(
            guidance_alpha=0.35,
            previous_objective_improvement=-0.1,
            structural_collapse_rate=0.4,
            exploration_branch="pareto",
        ),
    )

    assert report.latent_state is not None
    latent = runtime.materialize_latent(report.latent_state)
    assert latent.values[0:3] == [4.0, 1.0, 1.0]
    assert latent.values[5:] == [0.0, 0.4, 0.35]
    assert report.failed_expert_ids == ["failed"]
    assert report.revision_proposal is not None
    assert report.revision_proposal.confidence < 0.5


def test_no_supported_evidence_target_is_invented_for_nonmaterial_candidate() -> None:
    candidate = _candidate("molecule", formula=None)
    goal = _goal(
        PropertyObjective(property_name="score", direction=ObjectiveDirection.MAXIMIZE),
        molecular=True,
    )
    feature = _feature(
        candidate,
        feature_id="score",
        expert_id="unimol",
        value=1.0,
        unit=None,
        property_name="score",
    )
    request = _request(candidate=candidate, goal=goal, features=[feature])
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    state = UnifiedLatentStateRef(
        state_id="state-molecule",
        state_version=1,
        candidate_ref=candidate.candidate_ref,
        workspace_id=request.workspace.workspace_id,
        workspace_entities=request.workspace.entities,
        cycle=request.cycle,
        latent_artifact=ContentArtifactRef(
            artifact_id="latent-molecule",
            relative_path="fusion/latents/molecule.json",
            sha256="b" * 64,
            media_type="application/json",
            byte_size=1,
        ),
        latent_content_hash=stable_hash(output.latent),
        dtype=output.latent.dtype,
        shape=output.latent.shape,
        source_feature_ids=output.used_feature_ids,
        goal_hash=stable_hash(goal),
        seed=request.seed,
        backend_id=output.backend_id,
        backend_version=output.backend_version,
        code_revision=output.code_revision,
        weight_revision=output.weight_revision,
    )
    proposal = backend.propose_revision(
        FusionRevisionRequest(
            goal=goal,
            candidate=candidate,
            state=state,
            latent=output.latent,
            features=request.features,
            decision_context=request.decision_context,
        )
    )

    assert len(proposal.desired_changes) == 1
    change = proposal.desired_changes[0]
    assert change.property_name == "chemical_system"
    assert change.direction == "preserve"
    assert change.target_value is None
    assert proposal.confidence == 0.0
    assert any("no target" in note.lower() for note in proposal.safety_notes)


def test_live_literature_branch_becomes_molecule_generation_prior_not_score() -> None:
    candidate = _candidate("mol", formula=None)
    goal = _goal(
        PropertyObjective(
            property_name="binding_affinity",
            direction=ObjectiveDirection.MAXIMIZE,
        ),
        molecular=True,
    )
    feature = _feature(
        candidate,
        feature_id="feature-mol",
        expert_id="molecule-expert",
        property_name="binding_affinity",
        value=0.7,
        unit=None,
    )
    context = FusionDecisionContext(
        evidence_branch_id="branch-kras",
        evidence_branch_kind="derivative_or_analog",
        evidence_claim_ids=["claim-kras"],
        evidence_generator_hints={
            "seed_entities": ["Compound X"],
            "scaffold_smiles": ["CCO"],
            "target_contexts": ["KRAS G12D"],
            "mechanisms": ["inhibition"],
            "search_mode": "analog_or_derivative",
        },
        evidence_rationale="Recent source-grounded KRAS evidence",
    )
    request = _request(
        candidate=candidate,
        goal=goal,
        features=[feature],
        decision_context=context,
    )
    backend = EvidenceDrivenFusionBackend()
    output = backend.fuse(request)
    proposal = backend.propose_revision(_revision_request(request, output, candidate=candidate))
    assert proposal.preferred_generator_ids == ["reinvent4", "chemformer"]
    assert any(change.property_name == "scaffold_smiles" for change in proposal.desired_changes)
    assert "search branch" in " ".join(proposal.safety_notes)
    assert proposal.confidence <= 0.95
