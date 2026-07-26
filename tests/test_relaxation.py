from __future__ import annotations

from dataclasses import dataclass, replace

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


_RELAXED_CIF = """data_relaxed
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 3.9
_cell_length_b 3.9
_cell_length_c 3.9
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0 0 0
"""

_COLLAPSED_CIF = """data_collapsed
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 4
_cell_length_b 4
_cell_length_c 4
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0 0 0
O1 O 0.01 0 0
"""


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
            atom_count=1,
            volume_change_fraction=-0.02,
            minimum_distance_before_A=1.5,
            minimum_distance_after_A=1.6,
            relaxed_cif=_RELAXED_CIF,
            initial_stress_eV_A3=(0.01, 0.01, 0.01, 0.0, 0.0, 0.0),
            final_stress_eV_A3=(0.005, 0.005, 0.005, 0.0, 0.0, 0.0),
            warnings=("optimizer exhausted its step budget",),
        )


class _MissingStressRuntime(_Runtime):
    def relax(self, request: PeriodicRelaxationRequest) -> PeriodicRelaxationResult:
        return replace(
            super().relax(request),
            converged=True,
            final_max_force_eV_A=0.03,
            final_stress_eV_A3=None,
        )


class _CollapsedGeometryRuntime(_Runtime):
    def relax(self, request: PeriodicRelaxationRequest) -> PeriodicRelaxationResult:
        return replace(
            super().relax(request),
            converged=True,
            final_max_force_eV_A=0.03,
            atom_count=2,
            minimum_distance_after_A=0.04,
            relaxed_cif=_COLLAPSED_CIF,
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
    assert payload.geometry_gate.is_valid is True
    assert payload.final_stress is not None
    assert payload.final_stress.frobenius_norm_GPa > 0.0
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


def test_relax_endpoint_missing_stress_fails_gate_without_hiding_execution() -> None:
    request = PeriodicRelaxationRequest(candidate=_candidate(), seed=3)
    app = create_sidecar_app(identity=_identity(), runtime=_MissingStressRuntime())

    with TestClient(app) as client:
        response = client.post(
            "/v1/relax",
            json=request.model_dump(mode="json", exclude_none=False),
        )

    payload = PeriodicRelaxationPayload.model_validate(response.json())
    assert payload.execution_succeeded is True
    assert payload.converged is True
    assert payload.geometry_gate.is_valid is True
    assert payload.final_stress is None
    assert payload.strict_gate_passed is False
    assert payload.gate_failures == ["final_stress_unavailable"]


def test_relax_endpoint_executes_named_geometry_validator_and_fails_closed() -> None:
    request = PeriodicRelaxationRequest(candidate=_candidate(), seed=5)
    app = create_sidecar_app(identity=_identity(), runtime=_CollapsedGeometryRuntime())

    with TestClient(app) as client:
        response = client.post(
            "/v1/relax",
            json=request.model_dump(mode="json", exclude_none=False),
        )

    payload = PeriodicRelaxationPayload.model_validate(response.json())
    assert payload.execution_succeeded is True
    assert payload.geometry_gate.validator == "validate_crystal_geometry"
    assert payload.geometry_gate.is_valid is False
    assert payload.strict_gate_passed is False
    assert "minimum_distance_below_safety_threshold" in payload.gate_failures
    assert "invalid_relaxed_geometry" in payload.gate_failures
