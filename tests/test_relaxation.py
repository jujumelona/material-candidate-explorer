from __future__ import annotations

from dataclasses import dataclass

from fastapi.testclient import TestClient

from discovery_os.hashing import candidate_content_hash
from discovery_os.relaxation import (
    PeriodicRelaxationPayload,
    PeriodicRelaxationRequest,
    PeriodicRelaxationResult,
    PeriodicRelaxationSettings,
)
from discovery_os.schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    RepresentationKind,
)
from discovery_os.sidecars import ExpertResult, ModelIdentity, create_sidecar_app


def _candidate() -> Candidate:
    draft = Candidate(
        candidate_id="relax-crystal",
        candidate_type=CandidateType.CRYSTAL,
        domain=DiscoveryDomain.INORGANIC_MATERIALS,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.CIF,
                value="data_fixture\n_cell_length_a 4\n",
                canonical=False,
            )
        ],
    )
    return draft.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=draft.candidate_id,
                version=1,
                content_hash=candidate_content_hash(draft),
            )
        }
    )


@dataclass
class _Runtime:
    device: str = "cpu"
    loaded: bool = True
    load_failed: bool = False
    supported: bool = True

    def provenance_parameters(self) -> dict[str, object]:
        return {"runtime": "relaxation-fixture"}

    def encode(self, _request) -> ExpertResult:
        return ExpertResult(values=[[0.0, 0.0, 0.0]])

    def relax(self, _request: PeriodicRelaxationRequest) -> PeriodicRelaxationResult:
        return PeriodicRelaxationResult(
            completed_steps=10,
            converged=False,
            initial_max_force_eV_A=0.3,
            final_max_force_eV_A=0.08,
            initial_energy_eV=-10.0,
            final_energy_eV=-10.2,
            volume_change_fraction=-0.02,
            minimum_distance_before_A=1.5,
            minimum_distance_after_A=1.6,
            relaxed_cif="data_relaxed\n_cell_length_a 3.9\n",
            warnings=("optimizer exhausted its step budget",),
        )


def _identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="mattersim",
        model_version="1.2.5",
        adapter_version="1.0.0",
        code_revision="fixture-code",
        weight_revision="fixture-weight",
        capabilities=frozenset({"features"}),
    )


def test_relax_endpoint_separates_execution_from_convergence() -> None:
    request = PeriodicRelaxationRequest(
        candidate=_candidate(),
        settings=PeriodicRelaxationSettings(
            requested_steps=10,
            target_fmax_eV_A=0.05,
        ),
        seed=7,
    )
    app = create_sidecar_app(identity=_identity(), runtime=_Runtime())

    with TestClient(app) as client:
        response = client.post(
            "/v1/relax",
            json=request.model_dump(mode="json", exclude_none=False),
        )

    assert response.status_code == 200, response.text
    payload = PeriodicRelaxationPayload.model_validate(response.json())
    assert payload.execution_succeeded is True
    assert payload.converged is False
    assert payload.strict_gate_passed is False
    assert payload.gate_failures == [
        "optimizer_not_converged",
        "final_force_above_target",
    ]
    assert payload.completed_steps == payload.requested_steps == 10
    assert payload.provenance["seed"] == 7


def test_relax_endpoint_rejects_non_periodic_candidate() -> None:
    draft = Candidate(
        candidate_id="molecule",
        candidate_type=CandidateType.SMALL_MOLECULE,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.SMILES,
                value="CCO",
                canonical=True,
            )
        ],
    )
    molecule = draft.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=draft.candidate_id,
                version=1,
                content_hash=candidate_content_hash(draft),
            )
        }
    )
    payload = {
        "schema_version": "1.0",
        "candidate": molecule.model_dump(mode="json"),
        "settings": PeriodicRelaxationSettings().model_dump(mode="json"),
        "seed": 0,
    }
    app = create_sidecar_app(identity=_identity(), runtime=_Runtime())

    with TestClient(app) as client:
        response = client.post("/v1/relax", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
