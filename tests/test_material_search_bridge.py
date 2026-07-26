from __future__ import annotations

from collections import Counter
from pathlib import Path

import discovery_os
import pytest

from discovery_os.artifacts import ArtifactStore
from discovery_os.cli import main, make_parser
from discovery_os.evidence_fusion import EvidenceDrivenFusionBackend
from discovery_os.fusion_exploration import ExpertEvidenceStore
from discovery_os.fusion_loop import FusionLoopRunner
from discovery_os.fusion_registry import ExpertRegistry
from discovery_os.fusion_runtime import FusionRuntime
from discovery_os.fusion_schemas import (
    DiagnosticProperty,
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
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
from discovery_os.fusion_search import (
    FusionSearchRunner,
    SearchBudget,
    SearchControlPoint,
    SearchControlSweep,
)
from discovery_os.hashing import candidate_content_hash, stable_hash
from discovery_os.material_applications import build_material_application_brief
from discovery_os.material_search_bridge import (
    MaterialApplicationFusionSearchBridge,
    MaterialApplicationFusionSearchRequest,
    build_material_application_fusion_search_request,
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


def _brief():
    return build_material_application_brief(
        "Find a stable bulk crystal at 300 K and ambient pressure",
        material_field="general_inorganic",
        problem_context={"temperature": 300, "pressure": 101_325},
        explicit_role_ids=["stable_bulk_phase"],
    )


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="application-bulk-search-goal",
        domain=DiscoveryDomain.INORGANIC_MATERIALS,
        title="Stable bulk crystal screening",
        scientific_question="Which generated bulk crystals merit validation?",
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
        validation_profile_id="inorganic_materials-v1",
        candidate_types=[CandidateType.CRYSTAL],
    )


def _candidate(
    candidate_id: str,
    values: dict[str, dict[str, float]],
    *,
    parent: CandidateRef | None = None,
    cif_index: int = 0,
) -> Candidate:
    draft = Candidate(
        candidate_id=candidate_id,
        candidate_type=CandidateType.CRYSTAL,
        domain=DiscoveryDomain.INORGANIC_MATERIALS,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.CHEMICAL_FORMULA,
                value="MgB2",
                canonical=True,
            ),
            CandidateRepresentation(
                kind=RepresentationKind.CIF,
                value=(
                    f"data_{candidate_id}\n"
                    f"_cell_length_a {3.0 + cif_index * 0.01:.4f}\n"
                    "_cell_length_b 3.0000\n"
                    "_cell_length_c 3.0000\n"
                    "_cell_angle_alpha 90\n"
                    "_cell_angle_beta 90\n"
                    "_cell_angle_gamma 90\n"
                ),
                media_type="chemical/x-cif",
                canonical=True,
            )
        ],
        parent_candidate_ids=[parent.candidate_id] if parent else [],
        parent_candidate_refs=[parent] if parent else [],
        attributes={"mlip_diagnostics_by_expert": values},
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


class _EnergyExpert:
    def __init__(self, expert_id: str) -> None:
        self.expert_id = expert_id
        self.calls: Counter[str] = Counter()
        self._descriptor = ExpertDescriptor(
            expert_id=expert_id,
            display_name=expert_id,
            adapter_version="1.0.0",
            modalities=[ScientificModality.CRYSTAL_MATERIAL],
            supported_candidate_types=[CandidateType.CRYSTAL],
            supported_representations=[RepresentationKind.CIF],
            feature_spaces=["energy-screen-v1"],
        )

    @property
    def descriptor(self) -> ExpertDescriptor:
        return self._descriptor

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        candidate = request.candidate
        self.calls[candidate.candidate_id] += 1
        values = candidate.attributes["mlip_diagnostics_by_expert"][self.expert_id]
        energy = float(values["energy_per_atom"])
        max_force = float(values["max_force"])
        return ExpertFeaturePayload(
            workspace_entity_id=request.workspace_entity_id,
            candidate_ref=candidate.candidate_ref,
            expert_id=self.expert_id,
            modality=request.modality,
            feature_space=request.feature_space,
            tensor=NumericTensor(shape=[2], values=[energy, max_force]),
            semantics=FeatureSemantics(
                tensor_role=TensorRole.GLOBAL_EMBEDDING,
                projection_id="energy-screen-v1",
                pooling="none",
                normalization="fixture-identity",
            ),
            properties=[
                DiagnosticProperty(
                    property_name="energy_per_atom",
                    value=energy,
                    unit="eV/atom",
                    uncertainty=0.01,
                    source=self.expert_id,
                ),
                DiagnosticProperty(
                    property_name="max_force",
                    value=max_force,
                    unit="eV/angstrom",
                    uncertainty=0.005,
                    source=self.expert_id,
                ),
            ],
            provenance=ExpertProvenance(
                expert_id=self.expert_id,
                adapter_version="1.0.0",
                model_version=f"{self.expert_id}-model-v1",
                code_revision=f"{self.expert_id}-code-v1",
                weight_revision=f"{self.expert_id}-weight-v1",
                projection_version="energy-screen-v1",
                parameters_hash=stable_hash({"expert": self.expert_id}),
                seed=request.seed,
            ),
        )


class _MatterGenFixture:
    expected_generator_id = "mattergen"
    expected_generator_version = "1.0.0"
    expected_code_revision = "mattergen-fixture-code-v1"
    expected_weight_revision = "mattergen-fixture-weight-v1"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request) -> FusionGenerationResponse:
        call = self.calls
        self.calls += 1
        parent = request.parent_candidate.candidate_ref
        assert parent is not None
        candidates = [
            _candidate(
                f"generated-{call}-{index}",
                {
                    "mattersim": {
                        "energy_per_atom": -3.0 - index * 0.01,
                        "max_force": 0.08 + index * 0.002,
                    },
                    "chgnet": {
                        "energy_per_atom": -2.7 - index * 0.01,
                        "max_force": 0.09 + index * 0.002,
                    },
                },
                parent=parent,
                cif_index=call * request.run_config.candidate_count + index + 1,
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


def _root_candidate() -> Candidate:
    return _candidate(
        "application-search-root",
        {
            "mattersim": {"energy_per_atom": -2.5, "max_force": 0.12},
            "chgnet": {"energy_per_atom": -2.2, "max_force": 0.13},
        },
    )


def _config(parent: Candidate, *, count: int = 1) -> WorkspaceRunConfig:
    return WorkspaceRunConfig(
        workspace_mode=WorkspaceMode.ON,
        seed=29,
        goal_hash=stable_hash(_goal()),
        parent_candidate_ref=parent.candidate_ref,
        pair_key="application-search",
        cohort_index=0,
        generator_id="mattergen",
        generator_version="1.0.0",
        generator_code_revision="mattergen-fixture-code-v1",
        generator_weight_revision="mattergen-fixture-weight-v1",
        generator_parameters_hash="1" * 64,
        decoder_config_hash="2" * 64,
        postprocessing_hash="3" * 64,
        resource_budget_hash="4" * 64,
        evaluator_panel_hash="5" * 64,
        candidate_count=count,
    )


def _search_runner(tmp_path: Path):
    registry = ExpertRegistry()
    experts = [_EnergyExpert("mattersim"), _EnergyExpert("chgnet")]
    for expert in experts:
        registry.register(expert)
    runtime = FusionRuntime(
        registry,
        EvidenceDrivenFusionBackend(),
        ArtifactStore(tmp_path),
    )
    generator = _MatterGenFixture()
    runner = FusionSearchRunner(
        FusionLoopRunner(runtime, generator),
        ExpertEvidenceStore(runtime.artifact_store),
    )
    return runner, generator, experts


def test_bridge_executes_real_multi_round_fusion_search_and_persists_receipt(
    tmp_path: Path,
) -> None:
    parent = _root_candidate()
    request = build_material_application_fusion_search_request(
        _brief(),
        role_id="stable_bulk_phase",
        search_id="application-bridge-search",
        goal=_goal(),
        initial_candidate=parent,
        base_run_config=_config(parent),
        expert_ids=["mattersim", "chgnet"],
        rounds=3,
        frontier_width=1,
        ranking_limit=16,
    )
    runner, generator, experts = _search_runner(tmp_path)

    persisted = MaterialApplicationFusionSearchBridge(runner).execute(
        brief=_brief(),
        request=request,
    )

    report = persisted.report
    assert report.search.report.rounds_requested == 3
    assert report.search.report.rounds_completed == 3
    assert generator.calls > 1
    assert all(expert.calls for expert in experts)
    assert report.generation_execution_succeeded is True
    assert report.specialist_feature_evaluation_succeeded is True
    assert report.application_property_scoring_performed is False
    assert report.application_claim_created is False
    assert report.application_field_profile_id == "general_inorganic-workflow-v1"
    assert report.runtime_validation_profile_id == "inorganic_materials-v1"
    assert report.diagnostic_objective_names == [
        "energy_per_atom",
        "max_force",
    ]
    assert report.search_result_semantics == (
        "diagnostic-bulk-crystal-priority-not-application-fitness"
    )
    assert report.validation_handoff_candidate_refs
    assert report.unexecuted_application_validator_ids
    null_stability_rows = [
        item
        for item in report.search.report.ranked_candidates
        if "stability" in item.branch_scores
        and item.branch_scores["stability"] is None
    ]
    assert null_stability_rows
    assert all(
        any(
            "contributed no reciprocal-rank priority" in rationale
            for rationale in item.rationale
        )
        for item in null_stability_rows
    )
    path = runner.loop_runner.runtime.artifact_store.resolve(
        persisted.report_artifact.relative_path
    )
    assert path.is_file()
    assert path.read_bytes()


def test_bridge_rejects_retrieval_seed_without_structural_parent() -> None:
    parent = _root_candidate()
    nonstructural = parent.model_copy(
        update={
            "candidate_ref": None,
            "representations": [
                CandidateRepresentation(
                    kind=RepresentationKind.CHEMICAL_FORMULA,
                    value="Li2O",
                    canonical=True,
                )
            ],
        }
    )
    nonstructural = nonstructural.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=nonstructural.candidate_id,
                version=1,
                content_hash=candidate_content_hash(nonstructural),
            )
        }
    )
    config = _config(parent).model_copy(
        update={"parent_candidate_ref": nonstructural.candidate_ref}
    )

    with pytest.raises(ValueError, match="CIF, mmCIF, or POSCAR"):
        MaterialApplicationFusionSearchRequest(
            brief_id=_brief().brief_id,
            role_id="stable_bulk_phase",
            search_id="no-structure-search",
            goal=_goal(),
            initial_candidate=nonstructural,
            base_run_config=config,
            rounds=3,
            expert_ids=["mattersim", "chgnet"],
            required_primary_evaluator_ids=["mattersim", "chgnet"],
            search_budget=SearchBudget(
                max_generation_calls=8,
                max_generated_candidates=8,
            ),
        )


def test_bridge_requires_three_rounds_global_8_to_32_budget_and_real_panel() -> None:
    parent = _root_candidate()
    with pytest.raises(ValueError, match="greater than or equal to 3"):
        MaterialApplicationFusionSearchRequest(
            brief_id=_brief().brief_id,
            role_id="stable_bulk_phase",
            search_id="single-round-search",
            goal=_goal(),
            initial_candidate=parent,
            base_run_config=_config(parent),
            rounds=2,
            expert_ids=["mattersim", "chgnet"],
            required_primary_evaluator_ids=["mattersim", "chgnet"],
            search_budget=SearchBudget(
                max_generation_calls=8,
                max_generated_candidates=8,
            ),
        )
    with pytest.raises(ValueError, match="global 8 to 32"):
        MaterialApplicationFusionSearchRequest(
            brief_id=_brief().brief_id,
            role_id="stable_bulk_phase",
            search_id="small-global-budget-search",
            goal=_goal(),
            initial_candidate=parent,
            base_run_config=_config(parent),
            rounds=3,
            expert_ids=["mattersim", "chgnet"],
            required_primary_evaluator_ids=["mattersim", "chgnet"],
            search_budget=SearchBudget(
                max_generation_calls=7,
                max_generated_candidates=7,
            ),
        )
    with pytest.raises(ValueError, match="dummy, mock, or placeholder"):
        MaterialApplicationFusionSearchRequest(
            brief_id=_brief().brief_id,
            role_id="stable_bulk_phase",
            search_id="fake-panel-search",
            goal=_goal(),
            initial_candidate=parent,
            base_run_config=_config(parent),
            rounds=3,
            expert_ids=["dummy-expert", "chgnet"],
            required_primary_evaluator_ids=["dummy-expert", "chgnet"],
            search_budget=SearchBudget(
                max_generation_calls=8,
                max_generated_candidates=8,
            ),
        )

    retry_budget = MaterialApplicationFusionSearchRequest(
        brief_id=_brief().brief_id,
        role_id="stable_bulk_phase",
        search_id="independent-retry-budget",
        goal=_goal(),
        initial_candidate=parent,
        base_run_config=_config(parent),
        rounds=3,
        expert_ids=["mattersim", "chgnet"],
        required_primary_evaluator_ids=["mattersim", "chgnet"],
        search_budget=SearchBudget(
            max_generation_calls=32,
            max_generated_candidates=16,
        ),
    )
    assert retry_budget.search_budget.max_generation_calls == 32
    assert retry_budget.search_budget.max_generated_candidates == 16


def test_control_sweep_requires_global_budget_for_every_pre_round_three_attempt() -> None:
    parent = _root_candidate()
    sweep = SearchControlSweep(
        points=[
            SearchControlPoint(alpha=0.25, temperature=1.4, label="explore"),
            SearchControlPoint(alpha=0.50, temperature=1.0, label="center"),
            SearchControlPoint(alpha=0.75, temperature=0.7, label="refine"),
        ],
        include_adaptive_center=True,
        max_variants_per_parent=3,
    )
    with pytest.raises(ValueError, match="third adaptive round"):
        MaterialApplicationFusionSearchRequest(
            brief_id=_brief().brief_id,
            role_id="stable_bulk_phase",
            search_id="underfunded-control-sweep",
            goal=_goal(),
            initial_candidate=parent,
            base_run_config=_config(parent),
            rounds=3,
            expert_ids=["mattersim", "chgnet"],
            required_primary_evaluator_ids=["mattersim", "chgnet"],
            control_sweep=sweep,
            search_budget=SearchBudget(
                max_generation_calls=8,
                max_generated_candidates=8,
            ),
        )

    request = MaterialApplicationFusionSearchRequest(
        brief_id=_brief().brief_id,
        role_id="stable_bulk_phase",
        search_id="funded-control-sweep",
        goal=_goal(),
        initial_candidate=parent,
        base_run_config=_config(parent),
        rounds=3,
        expert_ids=["mattersim", "chgnet"],
        required_primary_evaluator_ids=["mattersim", "chgnet"],
        control_sweep=sweep,
        search_budget=SearchBudget(
            max_generation_calls=20,
            max_generated_candidates=8,
        ),
    )
    assert request.search_budget.max_generation_calls == 20
    assert request.search_budget.max_generated_candidates == 8


def test_bridge_rejects_goal_outside_selected_role() -> None:
    parent = _root_candidate()
    wrong_goal = _goal().model_copy(
        update={
            "objectives": [
                PropertyObjective(
                    property_name="turnover_frequency",
                    direction=ObjectiveDirection.MAXIMIZE,
                    unit="1/s",
                )
            ]
        }
    )
    wrong_config = _config(parent).model_copy(
        update={"goal_hash": stable_hash(wrong_goal)}
    )

    with pytest.raises(ValueError, match="bulk diagnostic allowlist"):
        build_material_application_fusion_search_request(
            _brief(),
            role_id="stable_bulk_phase",
            search_id="wrong-role-objective",
            goal=wrong_goal,
            initial_candidate=parent,
            base_run_config=wrong_config,
            expert_ids=["mattersim", "chgnet"],
            rounds=3,
        )


def test_bridge_rejects_non_bulk_application_role() -> None:
    brief = build_material_application_brief(
        "Find a transparent electrode for an optoelectronic stack",
        material_field="semiconductor",
        explicit_role_ids=["transparent_electrode"],
    )
    parent = _root_candidate()
    goal = _goal().model_copy(
        update={
            "validation_profile_id": brief.field_plan.profile.profile_id,
            "objectives": [
                PropertyObjective(
                    property_name="sheet_resistance",
                    direction=ObjectiveDirection.MINIMIZE,
                    unit="ohm/square",
                )
            ],
        }
    )
    config = _config(parent).model_copy(update={"goal_hash": stable_hash(goal)})

    with pytest.raises(ValueError, match="bulk crystal"):
        build_material_application_fusion_search_request(
            brief,
            role_id="transparent_electrode",
            search_id="interface-only-role",
            goal=goal,
            initial_candidate=parent,
            base_run_config=config,
            expert_ids=["mattersim", "chgnet"],
            rounds=3,
        )


def test_bridge_is_exported_and_cli_contract_is_discoverable(capsys) -> None:
    assert discovery_os.MaterialApplicationFusionSearchBridge is (
        MaterialApplicationFusionSearchBridge
    )
    parsed = make_parser().parse_args(
        [
            "material-fusion-search",
            "--brief",
            "brief.json",
            "--role",
            "stable_bulk_phase",
            "--search-id",
            "bridge-cli",
            "--goal",
            "goal.json",
            "--parent",
            "parent.json",
            "--run-config",
            "config.json",
            "--generator",
            "mattergen",
            "--rounds",
            "3",
            "--expert",
            "mattersim",
            "--expert",
            "chgnet",
        ]
    )
    assert parsed.handler.__name__ == "_material_fusion_search"
    assert parsed.expert == ["mattersim", "chgnet"]

    assert main(["schema", "MaterialApplicationFusionSearchRequest"]) == 0
    schema = capsys.readouterr().out
    assert '"bulk-crystal-search-triage-only"' in schema
    assert '"application_property_scoring_requested"' in schema
