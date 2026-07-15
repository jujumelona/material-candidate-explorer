from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from discovery_os.fusion_schemas import (
    FusionGenerationRequest,
    FusionGenerationResponse,
    GenerationControls,
    ScientificWorkspace,
    WorkspaceEntity,
    WorkspaceEntityRole,
    WorkspaceMode,
    WorkspaceRunConfig,
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
from discovery_os.sidecars import (
    GeneratedBatch,
    GeneratedCandidateData,
    ModelIdentity,
    create_sidecar_app,
)


def _parent() -> Candidate:
    candidate = Candidate(
        candidate_id="parent-crystal",
        candidate_type=CandidateType.CRYSTAL,
        domain=DiscoveryDomain.INORGANIC_MATERIALS,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.CIF,
                value="data_parent\n_cell_length_a 3\n",
                canonical=True,
            )
        ],
    )
    reference = CandidateRef(
        candidate_id=candidate.candidate_id,
        version=1,
        content_hash=candidate_content_hash(candidate),
    )
    return candidate.model_copy(update={"candidate_ref": reference})


def _request(
    *,
    generator_seed: int | None = 101,
    shared_seed: int = 7,
    candidate_types: list[CandidateType] | None = None,
    candidate_count: int = 1,
) -> FusionGenerationRequest:
    parent = _parent()
    goal = DiscoveryGoal(
        goal_id="crystal-goal",
        domain=DiscoveryDomain.INORGANIC_MATERIALS,
        title="Generate a crystal",
        scientific_question="Can a generated crystal improve stability?",
        objectives=[
            PropertyObjective(
                property_name="stability",
                direction=ObjectiveDirection.MAXIMIZE,
            )
        ],
        validation_profile_id="materials-v1",
        candidate_types=candidate_types or [CandidateType.CRYSTAL],
    )
    workspace = ScientificWorkspace(
        workspace_id="workspace-1",
        primary_entity_id="primary",
        entities=[
            WorkspaceEntity(
                entity_id="primary",
                role=WorkspaceEntityRole.PRIMARY_CANDIDATE,
                candidate_ref=parent.candidate_ref,
            )
        ],
    )
    config = WorkspaceRunConfig(
        workspace_mode=WorkspaceMode.OFF,
        seed=shared_seed,
        generator_seed=generator_seed,
        goal_hash=stable_hash(goal),
        parent_candidate_ref=parent.candidate_ref,
        pair_key="mattergen-pair",
        cohort_index=0,
        generator_id="mattergen",
        generator_version="1.0.0",
        generator_code_revision="code-revision",
        generator_weight_revision="weight-revision",
        generator_parameters_hash=stable_hash({"temperature": 1.0}),
        decoder_config_hash=stable_hash({"decoder": "fixture"}),
        postprocessing_hash=stable_hash({"post": "fixture"}),
        resource_budget_hash=stable_hash({"gpu": 0}),
        evaluator_panel_hash=stable_hash({"panel": "fixture"}),
        candidate_count=candidate_count,
        generation_controls=GenerationControls(decision_reason="contract fixture"),
    )
    return FusionGenerationRequest(
        goal=goal,
        parent_candidate=parent,
        workspace=workspace,
        workspace_mode=WorkspaceMode.OFF,
        run_config=config,
    )


@dataclass
class _Runtime:
    representation_kind: RepresentationKind = RepresentationKind.CIF
    loaded: bool = True
    load_failed: bool = False
    supported: bool = True
    device: str = "cpu"

    def provenance_parameters(self) -> dict[str, object]:
        return {"runtime": "fixture", "device": self.device}

    def generate(self, request: FusionGenerationRequest) -> GeneratedBatch:
        value = (
            "data_generated\n_cell_length_a 4\n"
            if self.representation_kind == RepresentationKind.CIF
            else "CCO"
        )
        return GeneratedBatch(
            candidates=(
                GeneratedCandidateData(
                    representations=(
                        CandidateRepresentation(
                            kind=self.representation_kind,
                            value=value,
                            canonical=True,
                        ),
                    ),
                ),
            )
        )


def _identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="mattergen",
        model_version="1.0.0",
        adapter_version="1.0.0",
        code_revision="code-revision",
        weight_revision="weight-revision",
        capabilities=frozenset({"generate"}),
    )


def _generate(request: FusionGenerationRequest, runtime: _Runtime | None = None):
    app = create_sidecar_app(identity=_identity(), runtime=runtime or _Runtime())
    with TestClient(app) as client:
        return client.post(
            "/v1/generate",
            json=request.model_dump(mode="json", exclude_none=False),
        )


def test_mattergen_output_contract_sets_crystal_type_and_cif() -> None:
    response = _generate(_request())

    assert response.status_code == 200, response.text
    payload = FusionGenerationResponse.model_validate(response.json())
    assert payload.candidates[0].candidate_type == CandidateType.CRYSTAL
    assert payload.candidates[0].representations[0].kind == RepresentationKind.CIF
    assert payload.pair_slots[0].pair_slot == 0
    assert payload.pair_slots[0].stream_position == 0
    assert payload.pair_slots[0].candidate_ref == payload.candidates[0].candidate_ref


def test_generator_contract_fails_closed_for_wrong_goal_type_or_representation() -> None:
    wrong_goal = _generate(_request(candidate_types=[CandidateType.SMALL_MOLECULE]))
    wrong_representation = _generate(
        _request(),
        _Runtime(representation_kind=RepresentationKind.SMILES),
    )

    assert wrong_goal.status_code == 502
    assert wrong_goal.json()["error"]["code"] == "invalid_model_output"
    assert wrong_representation.status_code == 502
    assert wrong_representation.json()["error"]["code"] == "invalid_model_output"


def test_effective_generator_seed_controls_identity_and_provenance() -> None:
    first = FusionGenerationResponse.model_validate(
        _generate(_request(generator_seed=101, shared_seed=7)).json()
    )
    same_effective_seed = FusionGenerationResponse.model_validate(
        _generate(_request(generator_seed=101, shared_seed=999)).json()
    )
    different_generator_seed = FusionGenerationResponse.model_validate(
        _generate(_request(generator_seed=102, shared_seed=7)).json()
    )

    assert first.provenance.seed == same_effective_seed.provenance.seed == 101
    assert first.pair_slots[0].batch_seed == 101
    assert first.candidates[0].candidate_ref == same_effective_seed.candidates[0].candidate_ref
    assert (
        first.candidates[0].candidate_ref
        != different_generator_seed.candidates[0].candidate_ref
    )


def test_generator_rejects_scientific_duplicates_despite_output_filename_metadata() -> None:
    class DuplicateRuntime(_Runtime):
        def generate(self, request: FusionGenerationRequest) -> GeneratedBatch:
            return GeneratedBatch(
                candidates=tuple(
                    GeneratedCandidateData(
                        representations=(
                            CandidateRepresentation(
                                kind=RepresentationKind.CIF,
                                value="data_duplicate\n_cell_length_a 4\n",
                                canonical=True,
                                metadata={"source_entry": f"batch-{index}.cif"},
                            ),
                        )
                    )
                    for index in range(2)
                )
            )

    response = _generate(_request(candidate_count=2), DuplicateRuntime())

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"
    assert "duplicate scientific outputs" in response.json()["error"]["message"]
