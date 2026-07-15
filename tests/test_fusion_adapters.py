from __future__ import annotations

import json

import pytest

from discovery_os.fusion_adapters import FusionAdapterError, HttpExpertEncoder, RemoteFusionBackend
from discovery_os.fusion_schemas import (
    ChangeAxis,
    ContentArtifactRef,
    DesiredChange,
    DiagnosticProperty,
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    ExpertProvenance,
    FeatureSemantics,
    FusionDecisionContext,
    FusionFeatureInput,
    FusionOutput,
    FusionRequest,
    FusionRevisionProposal,
    FusionRevisionRequest,
    NumericTensor,
    ScientificModality,
    ScientificWorkspace,
    TensorRole,
    UnifiedLatentStateRef,
    WorkspaceMode,
    WorkspaceEntity,
    WorkspaceEntityRole,
)
from discovery_os.hashing import candidate_content_hash, stable_hash
from discovery_os.fusion_runtime import FusionRuntime
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


class _Response:
    def __init__(self, payload: dict, *, status_code: int = 200) -> None:
        self.content = json.dumps(payload).encode()
        self.status_code = status_code
        self.headers = {"Content-Length": str(len(self.content))}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def post(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


def _candidate() -> Candidate:
    candidate = Candidate(
        candidate_id="molecule-1",
        candidate_type=CandidateType.SMALL_MOLECULE,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[CandidateRepresentation(kind=RepresentationKind.SMILES, value="CCO")],
    )
    return candidate.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate.candidate_id,
                version=1,
                content_hash=candidate_content_hash(candidate),
            )
        }
    )


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="g1",
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        title="Molecule",
        scientific_question="Test API contracts",
        objectives=[
            PropertyObjective(property_name="score", direction=ObjectiveDirection.MAXIMIZE)
        ],
        validation_profile_id="medicinal-chemistry-v1",
        candidate_types=[CandidateType.SMALL_MOLECULE],
    )


def _descriptor() -> ExpertDescriptor:
    return ExpertDescriptor(
        expert_id="unimol",
        display_name="Uni-Mol",
        adapter_version="1.0.0",
        modalities=[ScientificModality.MOLECULE_2D],
        supported_candidate_types=[CandidateType.SMALL_MOLECULE],
        supported_representations=[RepresentationKind.SMILES],
        feature_spaces=["unimol-cls-v1"],
    )


def _feature_payload() -> ExpertFeaturePayload:
    return ExpertFeaturePayload(
        workspace_entity_id="primary",
        candidate_ref=_candidate().candidate_ref,
        expert_id="unimol",
        modality=ScientificModality.MOLECULE_2D,
        feature_space="unimol-cls-v1",
        tensor=NumericTensor(shape=[2], values=[1.0, 2.0]),
        semantics=FeatureSemantics(
            tensor_role=TensorRole.GLOBAL_EMBEDDING,
            projection_id="unimol-cls-v1",
            pooling="mean",
            normalization="fixture-standardized",
        ),
        properties=[DiagnosticProperty(property_name="score", value=1.0)],
        provenance=ExpertProvenance(
            expert_id="unimol",
            adapter_version="1.0.0",
            model_version="0.1.6",
            code_revision="a" * 40,
            weight_revision="b" * 40,
            parameters_hash=stable_hash({}),
            projection_version="unimol-cls-v1",
            seed=1,
        ),
    )


def _feature_request() -> ExpertFeatureRequest:
    return ExpertFeatureRequest(
        workspace_entity_id="primary",
        candidate=_candidate(),
        goal=_goal(),
        modality=ScientificModality.MOLECULE_2D,
        feature_space="unimol-cls-v1",
        cycle=0,
        seed=1,
    )


def _workspace() -> ScientificWorkspace:
    candidate_ref = _candidate().candidate_ref
    return ScientificWorkspace(
        workspace_id="workspace-1",
        primary_entity_id="primary",
        entities=[
            WorkspaceEntity(
                entity_id="primary",
                role=WorkspaceEntityRole.PRIMARY_CANDIDATE,
                candidate_ref=candidate_ref,
            )
        ],
    )


def test_http_expert_encoder_uses_fixed_endpoint_and_idempotency_key() -> None:
    payload = _feature_payload()
    session = _Session([_Response(payload.model_dump(mode="json"))])
    encoder = HttpExpertEncoder(_descriptor(), "http://127.0.0.1:8102", session=session)

    result = encoder.encode(_feature_request())

    assert result.expert_id == "unimol"
    url, kwargs = session.calls[0]
    assert url == "http://127.0.0.1:8102/v1/features"
    assert len(kwargs["headers"]["Idempotency-Key"]) == 64
    assert kwargs["allow_redirects"] is False


def test_http_expert_encoder_rejects_nonlocal_cleartext_and_mismatched_expert() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        HttpExpertEncoder(_descriptor(), "http://example.com")

    payload = _feature_payload().model_copy(update={"expert_id": "other"})
    session = _Session([_Response(payload.model_dump(mode="json"))])
    encoder = HttpExpertEncoder(_descriptor(), "http://localhost:8102", session=session)
    with pytest.raises(FusionAdapterError, match="expert_id"):
        encoder.encode(_feature_request())


def test_expert_runtime_parameters_hash_is_bound_by_client_and_core_runtime() -> None:
    expected_hash = "f" * 64
    descriptor = _descriptor().model_copy(
        update={"metadata": {"parameters_hash": expected_hash}}
    )
    payload = _feature_payload()
    session = _Session([_Response(payload.model_dump(mode="json"))])
    encoder = HttpExpertEncoder(descriptor, "http://localhost:8102", session=session)
    with pytest.raises(FusionAdapterError, match="parameters_hash"):
        encoder.encode(_feature_request())

    with pytest.raises(ValueError, match="parameters_hash"):
        FusionRuntime._validate_feature_payload(payload, _feature_request(), descriptor)


def test_http_expert_encoder_rejects_oversized_or_unknown_fields() -> None:
    payload = _feature_payload().model_dump(mode="json")
    payload["unexpected"] = True
    session = _Session([_Response(payload)])
    encoder = HttpExpertEncoder(_descriptor(), "http://localhost:8102", session=session)
    with pytest.raises(FusionAdapterError, match="strict"):
        encoder.encode(_feature_request())

    session = _Session([_Response(_feature_payload().model_dump(mode="json"))])
    encoder = HttpExpertEncoder(
        _descriptor(),
        "http://localhost:8102",
        session=session,
        max_response_bytes=10,
    )
    with pytest.raises(FusionAdapterError, match="size"):
        encoder.encode(_feature_request())


def test_remote_fusion_backend_accounts_for_all_features() -> None:
    feature = FusionFeatureInput(
        feature_id="f1",
        workspace_entity_id="primary",
        payload=_feature_payload(),
    )
    request = FusionRequest(
        goal=_goal(),
        candidate_ref=_candidate().candidate_ref,
        workspace=_workspace(),
        workspace_mode=WorkspaceMode.ON,
        cycle=0,
        seed=1,
        features=[feature],
        decision_context=FusionDecisionContext(
            guidance_alpha=0.7,
            previous_objective_improvement=0.1,
            structural_collapse_rate=0.2,
            exploration_branch="stability",
        ),
        failed_expert_ids=["failed-expert"],
        missing_expert_ids=["missing-expert"],
    )
    output = FusionOutput(
        latent=NumericTensor(shape=[2], values=[1.0, 2.0]),
        used_feature_ids=["f1"],
        backend_id="fusion",
        backend_version="1",
        code_revision="c" * 40,
        weight_revision="d" * 40,
    )
    session = _Session([_Response(output.model_dump(mode="json"))])
    backend = RemoteFusionBackend("http://localhost:9000", session=session)

    result = backend.fuse(request)

    assert result.used_feature_ids == ["f1"]
    assert session.calls[0][0].endswith("/v1/fuse")
    assert set(session.calls[0][1]["json"]) == {
        "schema_version",
        "goal",
        "candidate_ref",
        "workspace",
        "workspace_mode",
        "cycle",
        "seed",
        "features",
        "previous_latent",
        "previous_state_id",
    }


def test_remote_fusion_backend_rejects_unknown_feature_and_stale_revision() -> None:
    feature = FusionFeatureInput(
        feature_id="f1",
        workspace_entity_id="primary",
        payload=_feature_payload(),
    )
    request = FusionRequest(
        goal=_goal(),
        candidate_ref=_candidate().candidate_ref,
        workspace=_workspace(),
        workspace_mode=WorkspaceMode.ON,
        cycle=0,
        seed=1,
        features=[feature],
    )
    bad = FusionOutput(
        latent=NumericTensor(shape=[1], values=[1.0]),
        used_feature_ids=["unknown"],
        backend_id="fusion",
        backend_version="1",
        code_revision="c" * 40,
        weight_revision="d" * 40,
    )
    backend = RemoteFusionBackend(
        "http://localhost:9000",
        session=_Session([_Response(bad.model_dump(mode="json"))]),
    )
    with pytest.raises(FusionAdapterError, match="unknown"):
        backend.fuse(request)


def test_remote_fusion_backend_uses_fixed_revision_endpoint() -> None:
    candidate = _candidate()
    workspace = _workspace()
    feature = FusionFeatureInput(
        feature_id="f1",
        workspace_entity_id="primary",
        payload=_feature_payload(),
    )
    latent = NumericTensor(shape=[2], values=[1.0, 2.0])
    state = UnifiedLatentStateRef(
        state_id="state-1",
        state_version=1,
        candidate_ref=candidate.candidate_ref,
        workspace_id=workspace.workspace_id,
        workspace_entities=workspace.entities,
        cycle=0,
        latent_artifact=ContentArtifactRef(
            artifact_id="latent-artifact",
            relative_path="fusion/latents/state-1.json",
            sha256="a" * 64,
            media_type="application/json",
            byte_size=2,
        ),
        latent_content_hash=stable_hash(latent),
        dtype=latent.dtype,
        shape=latent.shape,
        source_feature_ids=["f1"],
        goal_hash=stable_hash(_goal()),
        seed=1,
        backend_id="fusion",
        backend_version="1",
        code_revision="c" * 40,
        weight_revision="d" * 40,
    )
    request = FusionRevisionRequest(
        goal=_goal(),
        candidate=candidate,
        state=state,
        latent=latent,
        features=[feature],
        decision_context=FusionDecisionContext(
            guidance_alpha=0.8,
            exploration_branch="target_property",
        ),
    )
    proposal = FusionRevisionProposal(
        parent_candidate_ref=candidate.candidate_ref,
        state_id=state.state_id,
        desired_changes=[
            DesiredChange(
                axis=ChangeAxis.TARGET_PROPERTY,
                direction="increase",
                property_name="score",
                rationale="fixture",
            )
        ],
        confidence=0.5,
        rationale="fixture revision",
    )
    session = _Session([_Response(proposal.model_dump(mode="json"))])
    backend = RemoteFusionBackend("http://localhost:9000", session=session)

    result = backend.propose_revision(request)

    assert result.state_id == "state-1"
    assert session.calls[0][0].endswith("/v1/revise")
    assert set(session.calls[0][1]["json"]) == {
        "schema_version",
        "goal",
        "candidate",
        "state",
        "latent",
        "features",
    }


def test_remote_fusion_extended_context_requires_explicit_opt_in() -> None:
    feature = FusionFeatureInput(
        feature_id="f1",
        workspace_entity_id="primary",
        payload=_feature_payload(),
    )
    request = FusionRequest(
        goal=_goal(),
        candidate_ref=_candidate().candidate_ref,
        workspace=_workspace(),
        workspace_mode=WorkspaceMode.ON,
        cycle=0,
        seed=1,
        features=[feature],
        decision_context=FusionDecisionContext(guidance_alpha=0.75),
        failed_expert_ids=["failed-expert"],
        missing_expert_ids=["missing-expert"],
    )
    output = FusionOutput(
        latent=NumericTensor(shape=[2], values=[1.0, 2.0]),
        used_feature_ids=["f1"],
        backend_id="fusion",
        backend_version="1",
        code_revision="c" * 40,
        weight_revision="d" * 40,
    )
    session = _Session([_Response(output.model_dump(mode="json"))])
    backend = RemoteFusionBackend(
        "http://localhost:9000",
        session=session,
        send_extended_request_context=True,
    )

    backend.fuse(request)

    payload = session.calls[0][1]["json"]
    assert payload["decision_context"]["guidance_alpha"] == 0.75
    assert payload["failed_expert_ids"] == ["failed-expert"]
    assert payload["missing_expert_ids"] == ["missing-expert"]


def test_http_clients_reject_reserved_headers_and_nonfinite_timeouts() -> None:
    with pytest.raises(ValueError, match="reserved"):
        HttpExpertEncoder(
            _descriptor(),
            "https://example.com",
            headers={"Content-Type": "text/plain"},
        )
    with pytest.raises(ValueError, match="positive"):
        RemoteFusionBackend("https://example.com", timeout=float("nan"))
    with pytest.raises(ValueError, match="require HTTPS"):
        RemoteFusionBackend(
            "http://example.com",
            allow_insecure_http=True,
            headers={"Authorization": "Bearer secret"},
        )
