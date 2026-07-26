from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

from discovery_os.artifacts import ArtifactStore
from discovery_os.fusion_exploration import ExpertEvidenceStore
from discovery_os.evidence_fusion import EvidenceDrivenFusionBackend
from discovery_os.fusion_adapters import (
    HttpExpertEncoder,
    HttpPeriodicRelaxationClient,
)
from discovery_os.fusion_loop import FusionLoopRunner
from discovery_os.fusion_registry import ExpertRegistry
from discovery_os.fusion_runtime import FusionRuntime
from discovery_os.fusion_schemas import (
    ContentArtifactRef,
    DiagnosticProperty,
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    ExpertFeatureRef,
    ExpertProvenance,
    FeatureSemantics,
    FusionGenerationResponse,
    GeneratorProvenance,
    NumericTensor,
    ScientificModality,
    TensorRole,
    WorkspaceMode,
    WorkspaceRunConfig,
)
from discovery_os.fusion_search import FusionSearchRunner, FusionSearchStatus
from discovery_os.hashing import candidate_content_hash, stable_hash
from discovery_os.materials_screening import select_dft_handoff_refs
from discovery_os.relaxation import (
    PeriodicGeometryGateReport,
    PeriodicRelaxationPayload,
    PeriodicRelaxationRequest,
    PeriodicStressTensor,
)
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
import discovery_os.screening_validation as screening_validation
from discovery_os.screening_validation import MaterialScreeningValidationRunner


def _cif(*, lattice: float, oxygen: float) -> str:
    return f"""data_fixture
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a {lattice}
_cell_length_b {lattice}
_cell_length_c {lattice}
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
Li2 Li 0.5 0.5 0.5
O1 O {oxygen} {oxygen} {oxygen}
"""


def _candidate(candidate_id: str, *, lattice: float, oxygen: float) -> Candidate:
    draft = Candidate(
        candidate_id=candidate_id,
        candidate_type=CandidateType.CRYSTAL,
        domain=DiscoveryDomain.INORGANIC_MATERIALS,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.CIF,
                value=_cif(lattice=lattice, oxygen=oxygen),
                media_type="chemical/x-cif",
            ),
            CandidateRepresentation(
                kind=RepresentationKind.CHEMICAL_FORMULA,
                value="Li2O",
                canonical=True,
            ),
        ],
        attributes={
            "composition_key": "Li2O",
            "chemical_system": "Li-O",
            "elements": ["Li", "O"],
        },
    )
    return draft.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate_id,
                version=1,
                content_hash=candidate_content_hash(draft),
            )
        }
    )


def _candidate_with_energies(
    candidate_id: str,
    *,
    lattice: float,
    oxygen: float,
    mattersim_energy: float,
    chgnet_energy: float,
    parent: CandidateRef | None = None,
) -> Candidate:
    candidate = _candidate(candidate_id, lattice=lattice, oxygen=oxygen)
    draft = candidate.model_copy(
        update={
            "candidate_ref": None,
            "parent_candidate_ids": (
                [parent.candidate_id] if parent is not None else []
            ),
            "parent_candidate_refs": [parent] if parent is not None else [],
            "attributes": {
                **candidate.attributes,
                "model_energies": {
                    "mattersim": mattersim_energy,
                    "chgnet": chgnet_energy,
                },
            },
        }
    )
    return draft.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate_id,
                version=1,
                content_hash=candidate_content_hash(draft),
            )
        }
    )


def _feature(
    candidate: Candidate,
    expert_id: str,
    *,
    energy: float,
    entity_ids: list[str] | None = None,
) -> ExpertFeaturePayload:
    ids = entity_ids or ["Li-1", "Li-2", "O-1"]
    if expert_id == "mattersim":
        shape = [3, 3]
        values = [
            0.01,
            0.00,
            0.00,
            0.00,
            0.01,
            0.00,
            0.00,
            0.00,
            0.01,
        ]
        projection = "mattersim-force-v1"
        units = {
            "tensor": "eV/angstrom",
            "energy_per_atom": "eV/atom",
            "stress_norm": "eV/angstrom^3",
        }
        stress_value = 0.001
        stress_unit = "eV/angstrom^3"
    else:
        shape = [3, 4]
        values = [
            0.012,
            0.000,
            0.000,
            0.0,
            0.000,
            0.012,
            0.000,
            0.0,
            0.000,
            0.000,
            0.012,
            0.0,
        ]
        projection = "chgnet-force-magmom-v1"
        units = {
            "columns_0_2": "eV/angstrom",
            "energy_per_atom": "eV/atom",
            "stress_norm": "GPa",
        }
        stress_value = 0.2
        stress_unit = "GPa"
    return ExpertFeaturePayload(
        workspace_entity_id=f"primary-{candidate.candidate_id}",
        candidate_ref=candidate.candidate_ref,
        expert_id=expert_id,
        modality=ScientificModality.CRYSTAL_MATERIAL,
        feature_space=f"{expert_id}-atomic-v1",
        tensor=NumericTensor(shape=shape, values=values),
        semantics=FeatureSemantics(
            tensor_role=TensorRole.CUSTOM,
            projection_id=projection,
            entity_type="atom",
            entity_ids=ids,
            normalization="none",
            coordinate_frame="Cartesian xyz",
            unit_semantics=units,
        ),
        properties=[
            DiagnosticProperty(
                property_name="energy_per_atom",
                value=energy,
                unit="eV/atom",
                source=expert_id,
            ),
            DiagnosticProperty(
                property_name="max_force",
                value=0.012,
                unit="eV/angstrom",
                source=expert_id,
            ),
            DiagnosticProperty(
                property_name="stress_norm",
                value=stress_value,
                unit=stress_unit,
                source=expert_id,
            ),
        ],
        provenance=ExpertProvenance(
            expert_id=expert_id,
            adapter_version="1.0.0",
            model_version=f"{expert_id}-model-v1",
            code_revision=f"{expert_id}-code-v1",
            weight_revision=f"{expert_id}-weight-v1",
            projection_version=projection,
            parameters_hash=stable_hash({"expert": expert_id}),
            seed=7,
        ),
    )


def _store_feature(
    artifacts: ArtifactStore,
    evidence: ExpertEvidenceStore,
    payload: ExpertFeaturePayload,
) -> str:
    feature_id = f"feature-{payload.candidate_ref.candidate_id}-{payload.expert_id}"
    path, digest = artifacts.write_json(
        f"fusion/source-features/{feature_id}.json",
        payload,
    )
    ref = ExpertFeatureRef(
        feature_id=feature_id,
        workspace_entity_id=payload.workspace_entity_id,
        candidate_ref=payload.candidate_ref,
        goal_hash=stable_hash("screening-goal"),
        expert_id=payload.expert_id,
        modality=payload.modality,
        feature_space=payload.feature_space,
        status=payload.status,
        artifact=ContentArtifactRef(
            artifact_id=f"artifact-{feature_id}",
            relative_path=path,
            sha256=digest,
            media_type="application/json",
            byte_size=len(artifacts.read_bytes(path)),
        ),
        tensor_dtype=payload.tensor.dtype,
        tensor_shape=payload.tensor.shape,
        semantics=payload.semantics,
        properties=payload.properties,
        quality_flags=payload.quality_flags,
        warnings=payload.warnings,
        provenance=payload.provenance,
    )
    return evidence.put(payload, ref).evidence_id


def _stress(value: float) -> PeriodicStressTensor:
    norm = math.sqrt(3.0) * value
    return PeriodicStressTensor(
        components_eV_A3=[value, value, value, 0.0, 0.0, 0.0],
        frobenius_norm_eV_A3=norm,
        frobenius_norm_GPa=norm * 160.2176634,
    )


class _RelaxClient:
    def __init__(
        self,
        expert_id: str,
        energies: dict[str, float],
        *,
        missing_stress: bool = False,
        invalid_geometry: bool = False,
    ) -> None:
        self.expert_id = expert_id
        self.energies = energies
        self.missing_stress = missing_stress
        self.invalid_geometry = invalid_geometry
        self.calls: list[PeriodicRelaxationRequest] = []

    def relax(self, request: PeriodicRelaxationRequest) -> PeriodicRelaxationPayload:
        self.calls.append(request)
        candidate = request.candidate
        return PeriodicRelaxationPayload(
            candidate_ref=candidate.candidate_ref,
            expert_id=self.expert_id,
            execution_succeeded=True,
            optimizer=request.settings.optimizer,
            requested_steps=request.settings.requested_steps,
            completed_steps=8,
            converged=True,
            target_fmax_eV_A=request.settings.target_fmax_eV_A,
            atom_count=3,
            initial_max_force_eV_A=0.2,
            final_max_force_eV_A=0.02,
            initial_energy_eV=self.energies[candidate.candidate_id] + 0.2,
            final_energy_eV=self.energies[candidate.candidate_id],
            initial_stress=_stress(0.002),
            final_stress=None if self.missing_stress else _stress(0.001),
            volume_change_fraction=0.01,
            minimum_distance_before_A=1.2,
            minimum_distance_after_A=1.25,
            relaxed_structure=CandidateRepresentation(
                kind=RepresentationKind.CIF,
                value=candidate.representations[0].value,
                media_type="chemical/x-cif",
            ),
            geometry_gate=PeriodicGeometryGateReport(
                atom_count=3,
                volume_A3=64.0,
                volume_per_atom_A3=64.0 / 3.0,
                minimum_distance_A=1.25,
                minimum_distance_threshold_A=0.7,
                is_valid=not self.invalid_geometry,
                errors=(
                    ["minimum periodic distance below threshold"]
                    if self.invalid_geometry
                    else []
                ),
            ),
            strict_gate_passed=not (
                self.missing_stress or self.invalid_geometry
            ),
            gate_failures=[
                *(
                    ["final_stress_unavailable"]
                    if self.missing_stress
                    else []
                ),
                *(
                    ["invalid_relaxed_geometry"]
                    if self.invalid_geometry
                    else []
                ),
            ],
            provenance={
                "model_version": f"{self.expert_id}-model-v1",
                "weight_revision": f"{self.expert_id}-weight-v1",
                "seed": request.seed,
            },
        )


def _panel(tmp_path: Path):
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    first = _candidate("candidate-a", lattice=4.0, oxygen=0.25)
    second = _candidate("candidate-b", lattice=4.2, oxygen=0.30)
    evidence_by_ref: dict[str, list[str]] = {}
    for candidate, matter_energy, chgnet_energy in (
        (first, -2.0, -5.0),
        (second, -1.0, -4.0),
    ):
        ids = [
            _store_feature(
                artifacts,
                evidence,
                _feature(candidate, "mattersim", energy=matter_energy),
            ),
            _store_feature(
                artifacts,
                evidence,
                _feature(candidate, "chgnet", energy=chgnet_energy),
            ),
        ]
        evidence_by_ref[stable_hash(candidate.candidate_ref)] = ids
    mattersim = _RelaxClient(
        "mattersim",
        {"candidate-a": -6.0, "candidate-b": -3.0},
    )
    chgnet = _RelaxClient(
        "chgnet",
        {"candidate-a": -15.0, "candidate-b": -12.0},
    )
    runner = MaterialScreeningValidationRunner(
        evidence,
        mattersim_client=mattersim,
        chgnet_client=chgnet,
    )
    return runner, mattersim, chgnet, [first, second], evidence_by_ref


def test_executed_screening_calls_reliability_disagreement_pareto_and_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, mattersim, chgnet, candidates, evidence_by_ref = _panel(tmp_path)
    calls = {"relative": 0, "proxy": 0, "pareto": 0}
    original_relative = screening_validation.composition_relative_energy_disagreement
    original_proxy = screening_validation.assess_candidate_proxy
    original_pareto = screening_validation.rank_composition_scoped_pareto

    def relative(rows):
        calls["relative"] += 1
        return original_relative(rows)

    def proxy(request, calibration):
        calls["proxy"] += 1
        return original_proxy(request, calibration)

    def pareto(vectors):
        calls["pareto"] += 1
        return original_pareto(vectors)

    monkeypatch.setattr(
        screening_validation,
        "composition_relative_energy_disagreement",
        relative,
    )
    monkeypatch.setattr(screening_validation, "assess_candidate_proxy", proxy)
    monkeypatch.setattr(screening_validation, "rank_composition_scoped_pareto", pareto)

    first = runner.evaluate(candidates, evidence_by_ref, seed=11)

    assert first.relaxations_executed == 4
    assert first.relaxation_cache_hits == 0
    assert len(mattersim.calls) == len(chgnet.calls) == 2
    assert calls == {"relative": 1, "proxy": 2, "pareto": 1}
    assert len(first.pareto_ranking) == 2
    assert all(item.status == "complete" for item in first.receipts)
    assert all(
        item.composition_relative_energy.status == "available"
        for item in first.receipts
    )
    assert all(
        item.screening_vector.disagreement.risk == "low"
        for item in first.receipts
    )
    assert all(
        item.screening_vector.disagreement.raw_energy_per_atom_abs_diff_eV == 3.0
        and item.screening_vector.disagreement.energy_comparison_basis
        == "composition_relative_aligned"
        for item in first.receipts
    )
    assert all(
        item.proxy_reliability.status == "uncalibrated_or_ood"
        and item.dft_escalation
        for item in first.receipts
    )

    second = runner.evaluate(candidates, evidence_by_ref, seed=11)

    assert second.relaxations_executed == 0
    assert second.relaxation_cache_hits == 4
    assert len(mattersim.calls) == len(chgnet.calls) == 2
    assert calls == {"relative": 2, "proxy": 4, "pareto": 2}


def test_misaligned_atom_ids_fail_closed_before_relaxation(tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    candidate = _candidate("candidate-misaligned", lattice=4.0, oxygen=0.25)
    ids = [
        _store_feature(
            artifacts,
            evidence,
            _feature(candidate, "mattersim", energy=-2.0),
        ),
        _store_feature(
            artifacts,
            evidence,
            _feature(
                candidate,
                "chgnet",
                energy=-5.0,
                entity_ids=["Li-2", "Li-1", "O-1"],
            ),
        ),
    ]
    mattersim = _RelaxClient("mattersim", {candidate.candidate_id: -6.0})
    chgnet = _RelaxClient("chgnet", {candidate.candidate_id: -15.0})
    runner = MaterialScreeningValidationRunner(
        evidence,
        mattersim_client=mattersim,
        chgnet_client=chgnet,
    )

    result = runner.evaluate(
        [candidate],
        {stable_hash(candidate.candidate_ref): ids},
    )

    assert result.receipts[0].status == "unknown"
    assert result.receipts[0].dft_escalation is True
    assert result.pareto_ranking == []
    assert mattersim.calls == chgnet.calls == []


def test_missing_final_stress_is_not_a_passing_relaxation_gate(
    tmp_path: Path,
) -> None:
    runner, _mattersim, _chgnet, candidates, evidence_by_ref = _panel(tmp_path)
    runner.chgnet_client = _RelaxClient(
        "chgnet",
        {"candidate-a": -15.0, "candidate-b": -12.0},
        missing_stress=True,
    )

    result = runner.evaluate(candidates, evidence_by_ref)

    assert all(item.status == "complete" for item in result.receipts)
    assert all(
        item.chgnet_relaxation.payload.strict_gate_passed is False
        and "final_stress_unavailable"
        in item.chgnet_relaxation.payload.gate_failures
        for item in result.receipts
    )
    assert all(
        item.screening_vector.relaxation_gate_passed is False
        and item.dft_escalation is True
        for item in result.receipts
    )


def test_invalid_geometry_is_ranked_for_audit_but_excluded_from_dft_handoff(
    tmp_path: Path,
) -> None:
    runner, _mattersim, _chgnet, candidates, evidence_by_ref = _panel(tmp_path)
    runner.mattersim_client = _RelaxClient(
        "mattersim",
        {"candidate-a": -6.0, "candidate-b": -3.0},
        invalid_geometry=True,
    )

    result = runner.evaluate(candidates, evidence_by_ref)

    assert all(item.status == "invalid_geometry" for item in result.receipts)
    assert all(
        item.screening_vector.geometry_valid is False
        and item.pareto_rank is not None
        for item in result.receipts
    )
    assert (
        select_dft_handoff_refs(
            result.pareto_ranking,
            [item.screening_vector for item in result.receipts],
            top_k=2,
        )
        == []
    )


class _MaterialEncoder:
    def __init__(self, expert_id: str) -> None:
        self.expert_id = expert_id
        self._descriptor = ExpertDescriptor(
            expert_id=expert_id,
            display_name=expert_id,
            adapter_version="1.0.0",
            modalities=[ScientificModality.CRYSTAL_MATERIAL],
            supported_candidate_types=[CandidateType.CRYSTAL],
            supported_representations=[RepresentationKind.CIF],
            feature_spaces=[f"{expert_id}-atomic-v1"],
        )

    @property
    def descriptor(self):
        return self._descriptor

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        energy = float(request.candidate.attributes["model_energies"][self.expert_id])
        payload = _feature(
            request.candidate,
            self.expert_id,
            energy=energy,
        )
        return ExpertFeaturePayload.model_validate_json(
            payload.model_copy(
                update={
                    "workspace_entity_id": request.workspace_entity_id,
                    "modality": request.modality,
                    "feature_space": request.feature_space,
                    "provenance": payload.provenance.model_copy(
                        update={"seed": request.seed}
                    ),
                }
            ).model_dump_json(),
            strict=True,
        )


class _MaterialGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request) -> FusionGenerationResponse:
        self.calls += 1
        parent = request.parent_candidate.candidate_ref
        candidates = [
            _candidate_with_energies(
                f"generated-material-{index}",
                lattice=4.0 + index * 0.2,
                oxygen=0.25 + index * 0.05,
                mattersim_energy=-2.0 + index,
                chgnet_energy=-5.0 + index,
                parent=parent,
            )
            for index in range(request.run_config.candidate_count)
        ]
        config = request.run_config
        return FusionGenerationResponse(
            candidates=candidates,
            provenance=GeneratorProvenance(
                generator_id=config.generator_id,
                generator_version=config.generator_version,
                code_revision=config.generator_code_revision,
                weight_revision=config.generator_weight_revision,
                parameters_hash=config.generator_parameters_hash,
                seed=config.effective_generator_seed,
            ),
        )


def _material_goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="executed-material-screening-goal",
        domain=DiscoveryDomain.INORGANIC_MATERIALS,
        title="Execute two-MLIP screening",
        scientific_question="Which relaxed Li2O candidates should enter DFT?",
        objectives=[
            PropertyObjective(
                property_name="energy_per_atom",
                direction=ObjectiveDirection.MINIMIZE,
                unit="eV/atom",
            ),
            PropertyObjective(
                property_name="max_force",
                direction=ObjectiveDirection.MINIMIZE,
                unit="eV/angstrom",
            ),
        ],
        validation_profile_id="inorganic-materials-v1",
        candidate_types=[CandidateType.CRYSTAL],
    )


def _material_config(root: Candidate) -> WorkspaceRunConfig:
    return WorkspaceRunConfig(
        workspace_mode=WorkspaceMode.ON,
        seed=19,
        goal_hash=stable_hash(_material_goal()),
        parent_candidate_ref=root.candidate_ref,
        pair_key="executed-material-screening",
        cohort_index=0,
        generator_id="mattergen",
        generator_version="1.0.0",
        generator_code_revision="fixture-mattergen-code",
        generator_weight_revision="fixture-mattergen-weight",
        generator_parameters_hash="1" * 64,
        decoder_config_hash="2" * 64,
        postprocessing_hash="3" * 64,
        resource_budget_hash="4" * 64,
        evaluator_panel_hash="5" * 64,
        candidate_count=2,
    )


def test_fusion_search_executes_screening_before_dft_shortlist(
    tmp_path: Path,
) -> None:
    root = _candidate_with_energies(
        "material-root",
        lattice=3.8,
        oxygen=0.20,
        mattersim_energy=-1.5,
        chgnet_energy=-4.5,
    )
    registry = ExpertRegistry()
    registry.register(_MaterialEncoder("mattersim"))
    registry.register(_MaterialEncoder("chgnet"))
    runtime = FusionRuntime(
        registry,
        EvidenceDrivenFusionBackend(),
        ArtifactStore(tmp_path),
    )
    evidence = ExpertEvidenceStore(runtime.artifact_store)
    mattersim = _RelaxClient(
        "mattersim",
        {
            "material-root": -4.5,
            "generated-material-0": -6.0,
            "generated-material-1": -3.0,
        },
    )
    chgnet = _RelaxClient(
        "chgnet",
        {
            "material-root": -13.5,
            "generated-material-0": -15.0,
            "generated-material-1": -12.0,
        },
    )
    validator = MaterialScreeningValidationRunner(
        evidence,
        mattersim_client=mattersim,
        chgnet_client=chgnet,
    )
    search = FusionSearchRunner(
        FusionLoopRunner(runtime, _MaterialGenerator()),
        evidence,
        material_screening_validator=validator,
    )

    persisted = search.run(
        search_id="executed-material-screening",
        goal=_material_goal(),
        initial_candidate=root,
        base_run_config=_material_config(root),
        rounds=1,
        frontier_width=2,
        expert_ids=["mattersim", "chgnet"],
        required_primary_evaluator_ids=["mattersim", "chgnet"],
        modality=ScientificModality.CRYSTAL_MATERIAL,
    )

    report = persisted.report
    assert report.status == FusionSearchStatus.COMPLETED
    screened = [
        item for item in report.candidate_records if item.material_screening is not None
    ]
    assert len(screened) == 3
    assert len(mattersim.calls) == len(chgnet.calls) == 3
    assert all(
        item.material_screening_artifact is not None
        and runtime.artifact_store.resolve(
            item.material_screening_artifact.relative_path
        ).is_file()
        for item in screened
    )
    assert all(
        item.material_screening.composition_relative_energy.status == "available"
        for item in screened
    )
    assert report.validation_handoff_candidate_refs
    assert {
        stable_hash(item) for item in report.validation_handoff_candidate_refs
    }.issubset(
        {stable_hash(item.candidate.candidate_ref) for item in screened}
    )
    assert any(
        "Executed MatterSim/CHGNet relaxation screening" in reason
        for item in report.ranked_candidates
        for reason in item.rationale
    )


def test_fusion_search_auto_binds_http_expert_relax_routes(
    tmp_path: Path,
) -> None:
    registry = ExpertRegistry()
    for expert_id, port in (("mattersim", 8110), ("chgnet", 8113)):
        descriptor = ExpertDescriptor(
            expert_id=expert_id,
            display_name=expert_id,
            adapter_version="1.0.0",
            modalities=[ScientificModality.CRYSTAL_MATERIAL],
            supported_candidate_types=[CandidateType.CRYSTAL],
            supported_representations=[RepresentationKind.CIF],
            feature_spaces=[f"{expert_id}-atomic-v1"],
            metadata={
                "model_version": f"{expert_id}-model-v1",
                "code_revision": f"{expert_id}-code-v1",
                "weight_revision": f"{expert_id}-weight-v1",
                "parameters_hash": stable_hash({"expert": expert_id}),
            },
        )
        registry.register(
            HttpExpertEncoder(
                descriptor,
                f"http://127.0.0.1:{port}",
                session=object(),
            )
        )
    runtime = FusionRuntime(
        registry,
        EvidenceDrivenFusionBackend(),
        ArtifactStore(tmp_path),
    )
    search = FusionSearchRunner(
        FusionLoopRunner(runtime, _MaterialGenerator()),
        ExpertEvidenceStore(runtime.artifact_store),
    )

    assert search.material_screening_validator is not None
    assert isinstance(
        search.material_screening_validator.mattersim_client,
        HttpPeriodicRelaxationClient,
    )
    assert isinstance(
        search.material_screening_validator.chgnet_client,
        HttpPeriodicRelaxationClient,
    )
    assert (
        search.material_screening_validator.mattersim_client.base_url
        == "http://127.0.0.1:8110"
    )
    assert (
        search.material_screening_validator.chgnet_client.base_url
        == "http://127.0.0.1:8113"
    )


class _HttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.content = json.dumps(payload).encode("utf-8")
        self.status_code = 200
        self.headers = {"Content-Length": str(len(self.content))}

    def raise_for_status(self) -> None:
        return None


class _HttpSession:
    def __init__(self, response: _HttpResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, dict[str, object]]] = []

    def post(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_http_relaxation_client_calls_fixed_route_and_validates_identity() -> None:
    candidate = _candidate_with_energies(
        "http-relax-candidate",
        lattice=4.0,
        oxygen=0.25,
        mattersim_energy=-2.0,
        chgnet_energy=-5.0,
    )
    request = PeriodicRelaxationRequest(candidate=candidate, seed=41)
    local = _RelaxClient("mattersim", {candidate.candidate_id: -6.0})
    raw = local.relax(request)
    payload = raw.model_copy(
        update={
            "provenance": {
                **raw.provenance,
                "adapter_version": "1.0.0",
            }
        }
    )
    session = _HttpSession(_HttpResponse(payload.model_dump(mode="json")))
    descriptor = ExpertDescriptor(
        expert_id="mattersim",
        display_name="MatterSim",
        adapter_version="1.0.0",
        modalities=[ScientificModality.CRYSTAL_MATERIAL],
        supported_candidate_types=[CandidateType.CRYSTAL],
        supported_representations=[RepresentationKind.CIF],
        feature_spaces=["mattersim-atomic-v1"],
    )
    encoder = HttpExpertEncoder(
        descriptor,
        "http://127.0.0.1:8110",
        session=session,
    )

    result = encoder.periodic_relaxation_client().relax(request)

    assert result == payload
    url, kwargs = session.calls[0]
    assert url == "http://127.0.0.1:8110/v1/relax"
    assert kwargs["allow_redirects"] is False
    assert len(kwargs["headers"]["Idempotency-Key"]) == 64
