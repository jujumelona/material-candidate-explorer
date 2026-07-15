from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import discovery_os
import discovery_os.cli as cli_module
import pytest
from pydantic import ValidationError

from discovery_os.artifacts import ArtifactStore
from discovery_os.cli import main
from discovery_os.fusion_exploration import ExpertEvidenceStore, SchedulerObservation
from discovery_os.fusion_loop import FusionLoopRunner
from discovery_os.fusion_reference import MeanFusionBackend
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
    FusionSearchStatus,
    SearchBranchFailurePayload,
    SearchCandidateRecord,
    SearchControlPoint,
    SearchControlSweep,
    SearchRepresentationArtifactEncoding,
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


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="search-goal",
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        title="Branch-preserving search fixture",
        scientific_question="Can raw expert vectors drive independent search branches?",
        objectives=[
            PropertyObjective(
                property_name="stability",
                direction=ObjectiveDirection.MAXIMIZE,
                unit="arb",
            ),
            PropertyObjective(
                property_name="target_score",
                direction=ObjectiveDirection.MAXIMIZE,
                unit="arb",
            ),
        ],
        validation_profile_id="general-materials-v1",
        candidate_types=[CandidateType.COMPOSITION],
    )


def _candidate(
    candidate_id: str,
    formula: str,
    expert_properties: dict[str, dict[str, float]],
    *,
    parent: CandidateRef | None = None,
    representation_metadata: dict[str, str] | None = None,
    cif_value: str | None = None,
) -> Candidate:
    draft = Candidate(
        candidate_id=candidate_id,
        candidate_type=CandidateType.COMPOSITION,
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.CHEMICAL_FORMULA,
                value=formula,
                canonical=True,
                metadata=representation_metadata or {},
            ),
            CandidateRepresentation(
                kind=RepresentationKind.CIF,
                value=cif_value or f"data_{candidate_id}\n_cell_length_a 3.0",
                media_type="chemical/x-cif",
            ),
            CandidateRepresentation(
                kind=RepresentationKind.SMILES,
                value="C=C",
                canonical=True,
            ),
            CandidateRepresentation(
                kind=RepresentationKind.FASTA,
                value=">fixture-rna\nACGU",
                metadata={"sequence_type": "rna"},
            ),
            CandidateRepresentation(
                kind=RepresentationKind.PROTEIN_SEQUENCE,
                value="ACDE",
            ),
            CandidateRepresentation(
                kind=RepresentationKind.RNA_SEQUENCE,
                value="ACGU",
            ),
            CandidateRepresentation(
                kind=RepresentationKind.CUSTOM,
                value='{"path":"../../escape","values":[1,2]}',
                media_type="application/json",
                metadata={"unsafe_label": "../../must-not-enter-path"},
            ),
        ],
        parent_candidate_ids=[parent.candidate_id] if parent is not None else [],
        parent_candidate_refs=[parent] if parent is not None else [],
        attributes={"expert_properties": expert_properties},
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


class _VectorEncoder:
    def __init__(self, expert_id: str, *, all_ood: bool = False) -> None:
        self.expert_id = expert_id
        self.all_ood = all_ood
        self.calls: Counter[str] = Counter()
        self._descriptor = ExpertDescriptor(
            expert_id=expert_id,
            display_name=expert_id,
            adapter_version="1.0.0",
            modalities=[ScientificModality.CRYSTAL_MATERIAL],
            supported_candidate_types=[CandidateType.COMPOSITION],
            supported_representations=[
                RepresentationKind.CHEMICAL_FORMULA,
                RepresentationKind.CIF,
            ],
            feature_spaces=["search-projection-v1"],
        )

    @property
    def descriptor(self) -> ExpertDescriptor:
        return self._descriptor

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        candidate_id = request.candidate.candidate_id
        self.calls[candidate_id] += 1
        values = request.candidate.attributes["expert_properties"][self.expert_id]
        stability = float(values["stability"])
        target = float(values["target_score"])
        return ExpertFeaturePayload(
            workspace_entity_id=request.workspace_entity_id,
            candidate_ref=request.candidate.candidate_ref,
            expert_id=self.expert_id,
            modality=request.modality,
            feature_space=request.feature_space,
            tensor=NumericTensor(shape=[2], values=[stability, target]),
            semantics=FeatureSemantics(
                tensor_role=TensorRole.GLOBAL_EMBEDDING,
                projection_id="search-projection-v1",
                pooling="none",
                normalization="fixture-identity",
            ),
            properties=[
                DiagnosticProperty(
                    property_name="stability",
                    value=stability,
                    unit="arb",
                    uncertainty=0.05,
                    out_of_domain=self.all_ood,
                    source=self.expert_id,
                ),
                DiagnosticProperty(
                    property_name="target_score",
                    value=target,
                    unit="arb",
                    uncertainty=0.05,
                    out_of_domain=self.all_ood,
                    source=self.expert_id,
                ),
            ],
            provenance=ExpertProvenance(
                expert_id=self.expert_id,
                adapter_version="1.0.0",
                model_version=f"{self.expert_id}-model-v1",
                code_revision=f"{self.expert_id}-code-v1",
                weight_revision=f"{self.expert_id}-weight-v1",
                projection_version="search-projection-v1",
                parameters_hash=stable_hash({"expert": self.expert_id}),
                seed=request.seed,
            ),
        )


class _BatchGenerator:
    def __init__(self) -> None:
        self.calls = 0
        self.controls = []
        self.requests = []

    def generate(self, request) -> FusionGenerationResponse:
        call = self.calls
        self.calls += 1
        self.controls.append(request.run_config.generation_controls)
        self.requests.append(
            {
                "round": request.run_config.cohort_index,
                "pair_key": request.run_config.pair_key,
                "parent": stable_hash(request.parent_candidate.candidate_ref),
                "base_seed": request.run_config.seed,
                "generator_seed": request.run_config.effective_generator_seed,
            }
        )
        parent_ref = request.parent_candidate.candidate_ref
        candidates = []
        for index in range(request.run_config.candidate_count):
            if index % 2 == 0:
                panel = {
                    "expert-a": {
                        "stability": 0.90 - call * 0.001,
                        "target_score": 0.40 + call * 0.001,
                    },
                    "expert-b": {"stability": 0.80, "target_score": 0.30},
                }
            else:
                panel = {
                    "expert-a": {"stability": 0.40, "target_score": 0.90},
                    "expert-b": {"stability": 0.10, "target_score": 0.10},
                }
            candidates.append(
                _candidate(
                    f"generated-{call}-{index}",
                    f"MgB{call + index + 2}",
                    panel,
                    parent=parent_ref,
                )
            )
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
            warnings=[f"fixture generator call {call}"],
        )


class _DominatingGenerator:
    """Generate children that improve every raw expert/property dimension."""

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, request) -> FusionGenerationResponse:
        call = self.calls
        self.calls += 1
        parent_ref = request.parent_candidate.candidate_ref
        parent_panel = request.parent_candidate.attributes["expert_properties"]
        candidates = []
        for index in range(request.run_config.candidate_count):
            delta = 0.1 + index * 0.01
            panel = {
                expert_id: {
                    property_name: float(value) + delta
                    for property_name, value in properties.items()
                }
                for expert_id, properties in parent_panel.items()
            }
            candidates.append(
                _candidate(
                    f"dominating-{call}-{index}",
                    f"Al{call + 2}O{index + 3}",
                    panel,
                    parent=parent_ref,
                )
            )
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


class _DuplicateStructureGenerator:
    """Emit different lineage records for the same exact scientific content."""

    def generate(self, request) -> FusionGenerationResponse:
        parent_ref = request.parent_candidate.candidate_ref
        candidates = [
            _candidate(
                f"duplicate-structure-{index}",
                "Li2O",
                {
                    "expert-a": {
                        "stability": 0.8 + index * 0.1,
                        "target_score": 0.8 + index * 0.1,
                    },
                    "expert-b": {
                        "stability": 0.8 + index * 0.1,
                        "target_score": 0.8 + index * 0.1,
                    },
                },
                parent=parent_ref,
                representation_metadata={"source_entry": f"batch-{index}.cif"},
                cif_value="data_Li2O\n_cell_length_a 3.0",
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


class _OneBranchFailingGenerator(_BatchGenerator):
    def generate(self, request) -> FusionGenerationResponse:
        if (
            request.run_config.cohort_index == 1
            and request.run_config.pair_key.endswith("-stability")
        ):
            raise RuntimeError("stability worker unavailable\nGPU 0 reset")
        return super().generate(request)


class _AlwaysFailingGenerator:
    def generate(self, request) -> FusionGenerationResponse:
        raise RuntimeError("generator sidecar unavailable")


class _StructurallyCollapsingGenerator(_BatchGenerator):
    def generate(self, request) -> FusionGenerationResponse:
        result = super().generate(request)
        return result.model_copy(
            update={"warnings": ["structural_collapse: invalid generated lattice"]}
        )


def _runtime(
    tmp_path: Path, *, all_ood: bool = False
) -> tuple[FusionRuntime, list[_VectorEncoder]]:
    registry = ExpertRegistry()
    encoders = [
        _VectorEncoder("expert-a", all_ood=all_ood),
        _VectorEncoder("expert-b", all_ood=all_ood),
    ]
    for encoder in encoders:
        registry.register(encoder)
    runtime = FusionRuntime(
        registry,
        MeanFusionBackend(dimension=2),
        ArtifactStore(tmp_path),
    )
    return runtime, encoders


def _config(parent: Candidate) -> WorkspaceRunConfig:
    return WorkspaceRunConfig(
        workspace_mode=WorkspaceMode.ON,
        seed=23,
        goal_hash=stable_hash(_goal()),
        parent_candidate_ref=parent.candidate_ref,
        pair_key="search-fixture",
        cohort_index=0,
        generator_id="batch-generator",
        generator_version="1.0.0",
        generator_code_revision="batch-generator-code-v1",
        generator_weight_revision="batch-generator-weight-v1",
        generator_parameters_hash="1" * 64,
        decoder_config_hash="2" * 64,
        postprocessing_hash="3" * 64,
        resource_budget_hash="4" * 64,
        evaluator_panel_hash="5" * 64,
        candidate_count=2,
    )


def _initial_candidate() -> Candidate:
    return _candidate(
        "search-root",
        "MgB2",
        {
            "expert-a": {"stability": 0.5, "target_score": 0.5},
            "expert-b": {"stability": 0.5, "target_score": 0.5},
        },
    )


def test_search_runs_independent_branches_with_cache_lineage_and_raw_history(
    tmp_path: Path,
) -> None:
    runtime, encoders = _runtime(tmp_path)
    generator = _BatchGenerator()
    root = _initial_candidate()
    evidence = ExpertEvidenceStore(runtime.artifact_store)
    search = FusionSearchRunner(FusionLoopRunner(runtime, generator), evidence)
    observation_contexts = []

    def observations(context):
        observation_contexts.append(context)
        return SchedulerObservation(
            objective_improvement=0.1,
            structural_collapse_rate=0.1,
            high_disagreement_candidates=[root.candidate_ref],
        )

    persisted = search.run(
        search_id="branch-search",
        goal=_goal(),
        initial_candidate=root,
        base_run_config=_config(root),
        rounds=3,
        frontier_width=1,
        expert_ids=["expert-a", "expert-b"],
        observation_provider=observations,
    )
    report = persisted.report

    assert report.status == FusionSearchStatus.COMPLETED
    assert report.rounds_completed == 3
    assert len(report.branches) == 5
    assert all(branch.pool_record_ids for branch in report.branches)
    assert all(branch.frontier_record_ids for branch in report.branches)
    assert len(observation_contexts) == 15
    assert any(item.automatic_observation is None for item in observation_contexts)
    assert any(item.automatic_observation is not None for item in observation_contexts)
    assert all(
        root.candidate_ref in decision.observation.high_disagreement_candidates
        for branch in report.branches
        for decision in branch.scheduler_history
    )
    assert all(
        decision.observation.objective_improvement == 0.1
        and decision.observation.structural_collapse_rate == 0.1
        for branch in report.branches
        for decision in branch.scheduler_history
    )
    # Retained elites are reconsidered beside new children. Re-evaluating an
    # unchanged elite advances its own branch-local latent state one cycle at a
    # time instead of silently switching to another branch's state.
    assert {item.requested_cycle for item in report.cycle_records} == {0, 2, 3}
    assert [len(item.cycle_record_ids) for item in report.round_history] == [1, 5, 5]
    assert len(report.history_artifacts) == len(report.cycle_records) == 11
    records_by_id = {item.record_id: item for item in report.candidate_records}
    for branch_name, frontier_ids in report.round_history[-1].branch_frontiers.items():
        assert len(frontier_ids) == 1
        assert str(records_by_id[frontier_ids[0]].source_branch) == branch_name
    for cycle in [item for item in report.cycle_records if item.round_index == 2]:
        branch_name = str(cycle.branch)
        prior_frontier_id = report.round_history[1].branch_frontiers[branch_name][0]
        assert (
            records_by_id[cycle.parent_record_id].latent_state.previous_state_id
            == records_by_id[prior_frontier_id].latent_state.state_id
        )
    assert all(item.generation_warnings for item in report.cycle_records)
    assert all(item.generation_provenance.generator_id == "batch-generator" for item in report.cycle_records)
    assert all(
        item.run_config.generation_controls == item.controls
        for item in report.cycle_records
    )
    assert report.base_generator_parameters_hash == "1" * 64
    assert all(
        item.run_config.generator_parameters_hash
        == stable_hash(
            {
                "base_parameters_hash": report.base_generator_parameters_hash,
                "generation_controls": item.controls,
            }
        )
        for item in report.cycle_records
    )

    # Two positive observations reduce alpha independently in every branch;
    # round 2 must receive those updated controls through cloned run configs.
    round_two = [item for item in report.cycle_records if item.round_index == 2]
    assert round_two
    assert all(item.controls.alpha == 0.4 for item in round_two)
    assert all(item.controls.schedule_step == 2 for item in round_two)
    assert any(item.alpha == 0.4 for item in generator.controls)

    # Parent re-evaluation across later cycles and overlapping branches uses the
    # FusionRuntime content cache: each expert executes at most once per content.
    for encoder in encoders:
        assert encoder.calls
        assert set(encoder.calls.values()) == {1}

    generated_records = [
        item for item in report.candidate_records if item.generation_provenance is not None
    ]
    assert generated_records
    smiles_paths = set()
    for record in generated_records:
        assert len(record.evidence_ids) == 2
        assert {row.kind for row in record.candidate.representations} >= {
            RepresentationKind.CHEMICAL_FORMULA,
            RepresentationKind.CIF,
        }
        raw = runtime.artifact_store.read_bytes(
            record.candidate_artifact.relative_path,
            expected_sha256=record.candidate_artifact.sha256,
        )
        restored = Candidate.model_validate_json(raw, strict=True)
        assert restored == record.candidate
        assert len(evidence.by_candidate(record.candidate)) == 2

        assert len(record.representation_artifact_refs) == len(
            record.candidate.representations
        )
        by_index = {
            item.representation_index: item
            for item in record.representation_artifact_refs
        }
        assert set(by_index) == set(range(len(record.candidate.representations)))
        expected_extensions = {
            RepresentationKind.CIF: ".cif",
            RepresentationKind.SMILES: ".smi",
            RepresentationKind.FASTA: ".fasta",
            RepresentationKind.PROTEIN_SEQUENCE: ".txt",
            RepresentationKind.RNA_SEQUENCE: ".txt",
            RepresentationKind.CHEMICAL_FORMULA: ".json",
            RepresentationKind.CUSTOM: ".json",
        }
        for index, representation in enumerate(record.candidate.representations):
            reference = by_index[index]
            artifact = reference.artifact
            assert artifact.relative_path.startswith("fusion/search/representations/")
            assert ".." not in artifact.relative_path
            assert artifact.relative_path.endswith(
                expected_extensions[RepresentationKind(str(representation.kind))]
            )
            payload = runtime.artifact_store.read_bytes(
                artifact.relative_path,
                expected_sha256=artifact.sha256,
            )
            assert len(payload) == artifact.byte_size
            if str(reference.encoding) == SearchRepresentationArtifactEncoding.RAW_UTF8:
                assert payload.decode("utf-8") == representation.value
            else:
                assert str(reference.encoding) == (
                    SearchRepresentationArtifactEncoding.CANONICAL_JSON
                )
                restored_representation = CandidateRepresentation.model_validate_json(
                    payload,
                    strict=True,
                )
                assert restored_representation == representation
            if str(representation.kind) == RepresentationKind.SMILES:
                smiles_paths.add(artifact.relative_path)

    # Identical SMILES are content-addressed once even when the same scientific
    # representation appears in many candidates and independent branches.
    assert len(smiles_paths) == 1

    tampered = generated_records[0].model_dump(mode="json")
    tampered["representation_artifact_refs"][0]["artifact"]["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="content validation"):
        SearchCandidateRecord.model_validate(tampered)

    report_bytes = runtime.artifact_store.read_bytes(
        persisted.report_artifact.relative_path,
        expected_sha256=persisted.report_artifact.sha256,
    )
    assert len(report_bytes) == persisted.report_artifact.byte_size
    assert report_bytes
    assert report.scientific_claim == "diagnostic_only"


def test_validation_handoff_is_pareto_stability_only_and_exact_content_unique(
    tmp_path: Path,
) -> None:
    runtime, _encoders = _runtime(tmp_path)
    root = _initial_candidate()
    result = FusionSearchRunner(
        FusionLoopRunner(runtime, _DuplicateStructureGenerator()),
        ExpertEvidenceStore(runtime.artifact_store),
    ).run(
        search_id="validation-handoff-search",
        goal=_goal(),
        initial_candidate=root,
        base_run_config=_config(root),
        rounds=1,
        frontier_width=4,
        expert_ids=["expert-a", "expert-b"],
        required_primary_evaluator_ids=["expert-a", "expert-b"],
    )

    report = result.report
    allowed = {
        stable_hash(selected.candidate_ref)
        for branch in report.final_selection.branches
        if str(branch.branch) in {"pareto", "stability"}
        for selected in branch.candidates
    }
    handoff = report.validation_handoff_candidate_refs
    assert handoff
    assert len(handoff) == len({stable_hash(item) for item in handoff})
    assert {stable_hash(item) for item in handoff}.issubset(allowed)

    candidate_by_ref = {
        stable_hash(record.candidate.candidate_ref): record.candidate
        for record in report.candidate_records
    }
    formulas = [
        candidate_by_ref[stable_hash(ref)].representations[0].value
        for ref in handoff
    ]
    # The two generated candidates differ only in lineage/output filename and
    # are one exact scientific structure at this high-cost validation boundary.
    assert formulas.count("Li2O") == 1

    tampered = json.loads(report.model_dump_json())
    tampered["validation_handoff_candidate_refs"].append(
        tampered["validation_handoff_candidate_refs"][0]
    )
    with pytest.raises(ValidationError, match="validation handoff"):
        type(report).model_validate_json(json.dumps(tampered), strict=True)

    legacy = json.loads(report.model_dump_json())
    legacy.pop("validation_handoff_candidate_refs")
    migrated = type(report).model_validate_json(json.dumps(legacy), strict=True)
    assert migrated.validation_handoff_candidate_refs == handoff


def test_search_exhausts_fail_closed_when_primary_experts_are_ood(
    tmp_path: Path,
) -> None:
    runtime, _encoders = _runtime(tmp_path, all_ood=True)
    generator = _BatchGenerator()
    root = _initial_candidate()
    evidence = ExpertEvidenceStore(runtime.artifact_store)

    result = FusionSearchRunner(
        FusionLoopRunner(runtime, generator), evidence
    ).run(
        search_id="ood-search",
        goal=_goal(),
        initial_candidate=root,
        base_run_config=_config(root),
        rounds=3,
        frontier_width=1,
        expert_ids=["expert-a", "expert-b"],
    )

    assert result.report.status == FusionSearchStatus.EXHAUSTED
    assert result.report.rounds_completed == 1
    assert all(not item.frontier_record_ids for item in result.report.branches)
    assert all(not item.pool_record_ids for item in result.report.branches)
    assert result.report.final_selection.excluded_candidates
    assert all(
        not item.selection_eligible
        for item in result.report.candidate_records
    )
    assert any(
        "out-of-domain" in reason
        for item in result.report.final_selection.excluded_candidates
        for reason in item.reasons
    )
    # Raw payloads are preserved even though no candidate is allowed to advance.
    assert len(evidence.by_evaluator("expert-a")) >= 3
    assert len(evidence.by_evaluator("expert-b")) >= 3
    pareto = next(
        item for item in result.report.branches if str(item.branch) == "pareto"
    )
    assert len(pareto.scheduler_history) == 1
    assert pareto.scheduler_history[0].observation.objective_improvement == 0.0
    # An expert/OOD exclusion is not evidence that a structure collapsed.
    assert pareto.scheduler_history[0].observation.structural_collapse_rate == 0.0


def test_search_records_one_branch_failure_and_continues_other_frontiers(
    tmp_path: Path,
) -> None:
    runtime, _encoders = _runtime(tmp_path)
    root = _initial_candidate()
    result = FusionSearchRunner(
        FusionLoopRunner(runtime, _OneBranchFailingGenerator()),
        ExpertEvidenceStore(runtime.artifact_store),
    ).run(
        search_id="partial-search",
        goal=_goal(),
        initial_candidate=root,
        base_run_config=_config(root),
        rounds=3,
        frontier_width=1,
        expert_ids=["expert-a", "expert-b"],
    )

    report = result.report
    assert report.status == FusionSearchStatus.PARTIAL
    assert report.rounds_completed == 3
    assert report.cycle_records
    assert len(report.failure_records) == 1
    failure = report.failure_records[0]
    assert str(failure.branch) == "stability"
    assert failure.round_index == 1
    assert failure.requested_cycle == 2
    assert failure.parent_candidate_ref is not None
    # Stability had no active cycle in round 0, so its branch-local scheduler
    # has not observed a result yet when its first frontier fails in round 1.
    assert failure.controls.schedule_step == 0
    assert failure.cause_type == "RuntimeError"
    assert failure.cause == "stability worker unavailable GPU 0 reset"
    assert report.round_history[1].failure_record_ids == [failure.failure_id]
    assert report.round_history[1].cycle_record_ids
    round_zero_stability = report.round_history[0].branch_frontiers["stability"]
    round_one_stability = report.round_history[1].branch_frontiers["stability"]
    assert round_zero_stability == round_one_stability
    round_two_stability_cycles = [
        item
        for item in report.cycle_records
        if item.round_index == 2 and str(item.branch) == "stability"
    ]
    assert len(round_two_stability_cycles) == 1
    records_by_id = {item.record_id: item for item in report.candidate_records}
    assert (
        records_by_id[round_two_stability_cycles[0].parent_record_id].candidate.candidate_ref
        == failure.parent_candidate_ref
    )

    raw = runtime.artifact_store.read_bytes(
        failure.failure_artifact.relative_path,
        expected_sha256=failure.failure_artifact.sha256,
    )
    assert len(raw) == failure.failure_artifact.byte_size
    payload = SearchBranchFailurePayload.model_validate_json(raw, strict=True)
    assert payload.round_index == failure.round_index
    assert payload.branch == failure.branch
    assert payload.parent_candidate_ref == failure.parent_candidate_ref
    assert payload.controls == failure.controls


def test_search_emits_valid_failed_report_when_every_frontier_fails(
    tmp_path: Path,
) -> None:
    runtime, _encoders = _runtime(tmp_path)
    root = _initial_candidate()
    result = FusionSearchRunner(
        FusionLoopRunner(runtime, _AlwaysFailingGenerator()),
        ExpertEvidenceStore(runtime.artifact_store),
    ).run(
        search_id="failed-search",
        goal=_goal(),
        initial_candidate=root,
        base_run_config=_config(root),
        rounds=3,
        frontier_width=1,
        expert_ids=["expert-a", "expert-b"],
    )

    report = result.report
    assert report.status == FusionSearchStatus.FAILED
    assert report.rounds_completed == 3
    assert report.cycle_records == []
    assert report.history_artifacts == []
    assert len(report.failure_records) == 3
    assert all(
        item.parent_candidate_ref == root.candidate_ref
        for item in report.failure_records
    )
    assert {item.requested_cycle for item in report.failure_records} == {0}
    for round_record, failure in zip(
        report.round_history, report.failure_records, strict=True
    ):
        assert round_record.cycle_record_ids == []
        assert round_record.candidate_record_ids == []
        assert round_record.failure_record_ids == [failure.failure_id]
        assert set(round_record.branch_frontiers) == {
            "stability",
            "target_property",
            "novelty",
            "expert_disagreement",
            "pareto",
        }
        # The initial frontier has no candidate record yet, but is retained as
        # the identical immutable parent and retried in the next round.
        assert all(not rows for rows in round_record.branch_frontiers.values())
    stored = runtime.artifact_store.read_bytes(
        result.report_artifact.relative_path,
        expected_sha256=result.report_artifact.sha256,
    )
    assert len(stored) == result.report_artifact.byte_size


def test_only_explicit_structural_collapse_signals_affect_scheduler(
    tmp_path: Path,
) -> None:
    runtime, _encoders = _runtime(tmp_path)
    root = _initial_candidate()
    result = FusionSearchRunner(
        FusionLoopRunner(runtime, _StructurallyCollapsingGenerator()),
        ExpertEvidenceStore(runtime.artifact_store),
    ).run(
        search_id="collapse-search",
        goal=_goal(),
        initial_candidate=root,
        base_run_config=_config(root),
        rounds=2,
        frontier_width=1,
        expert_ids=["expert-a", "expert-b"],
    )

    report = result.report
    # Collapsed children fail closed, while the last safe parent remains as the
    # bounded elite frontier for subsequent exploration.
    assert report.status == FusionSearchStatus.COMPLETED
    generated = [
        item for item in report.candidate_records if item.generation_provenance
    ]
    assert generated
    assert all(not item.selection_eligible for item in generated)
    assert all(
        item.structural_collapse_reasons
        == ["structural_collapse: invalid generated lattice"]
        for item in generated
    )
    records_by_id = {item.record_id: item for item in report.candidate_records}
    assert all(
        records_by_id[record_id].candidate.candidate_ref == root.candidate_ref
        for branch in report.branches
        for record_id in branch.frontier_record_ids
    )
    pareto = next(item for item in report.branches if str(item.branch) == "pareto")
    assert pareto.scheduler_history[0].observation.structural_collapse_rate == 1.0


def test_generator_seeds_are_branch_specific_and_reproducible(
    tmp_path: Path,
) -> None:
    def execute(path: Path):
        runtime, _encoders = _runtime(path)
        generator = _BatchGenerator()
        root = _initial_candidate()
        result = FusionSearchRunner(
            FusionLoopRunner(runtime, generator),
            ExpertEvidenceStore(runtime.artifact_store),
        ).run(
            search_id="seed-search",
            goal=_goal(),
            initial_candidate=root,
            base_run_config=_config(root),
            rounds=2,
            frontier_width=1,
            expert_ids=["expert-a", "expert-b"],
        )
        return generator, result

    first_generator, first = execute(tmp_path / "first")
    second_generator, second = execute(tmp_path / "second")

    assert first_generator.requests == second_generator.requests
    second_round = [
        item for item in first_generator.requests if item["round"] == 1
    ]
    assert len(second_round) == 5
    assert len({item["generator_seed"] for item in second_round}) == 5
    assert {item["base_seed"] for item in first_generator.requests} == {23}
    assert all(
        item.run_config.generator_seed is not None
        and item.run_config.effective_generator_seed
        == item.generation_provenance.seed
        for item in first.report.cycle_records
    )
    assert {item.run_config.seed for item in first.report.cycle_records} == {23}
    assert {item.latent_state.seed for item in first.report.candidate_records} == {23}
    assert [
        item.run_config.generator_seed for item in first.report.cycle_records
    ] == [item.run_config.generator_seed for item in second.report.cycle_records]


def test_search_automatically_schedules_from_raw_cross_round_dominance(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime, _encoders = _runtime(tmp_path)
    decision_contexts = []
    original_update = runtime.update

    def recording_update(**kwargs):
        if kwargs["workspace_mode"] == WorkspaceMode.ON:
            decision_contexts.append(kwargs.get("decision_context"))
        return original_update(**kwargs)

    monkeypatch.setattr(runtime, "update", recording_update)
    root = _initial_candidate()
    result = FusionSearchRunner(
        FusionLoopRunner(runtime, _DominatingGenerator()),
        ExpertEvidenceStore(runtime.artifact_store),
    ).run(
        search_id="automatic-search",
        goal=_goal(),
        initial_candidate=root,
        base_run_config=_config(root),
        rounds=3,
        frontier_width=1,
        expert_ids=["expert-a", "expert-b"],
    )

    histories = {
        str(item.branch): item.scheduler_history for item in result.report.branches
    }
    persistent_pools = {
        str(item.branch): item for item in result.report.branches
    }
    assert all(
        len(persistent_pools[name].pool_record_ids)
        > len(persistent_pools[name].frontier_record_ids)
        for name in ("stability", "target_property", "novelty", "pareto")
    )
    assert len(histories["pareto"]) == 3
    assert len(histories["expert_disagreement"]) == 0
    assert all(
        len(rows) == 2
        for name, rows in histories.items()
        if name not in {"pareto", "expert_disagreement"}
    )
    assert all(
        decision.observation.objective_improvement == 1.0
        and decision.observation.structural_collapse_rate == 0.0
        for rows in histories.values()
        for decision in rows
    )
    # This remains positive even though selector display scores are rescaled in
    # every round: scheduling compares raw cross-round utility vectors instead.
    final_controls = {
        str(item.branch): item.controls for item in result.report.branches
    }
    assert final_controls["pareto"].alpha == 0.3
    assert final_controls["expert_disagreement"].alpha == 0.5
    assert all(
        controls.alpha == 0.4
        for name, controls in final_controls.items()
        if name not in {"pareto", "expert_disagreement"}
    )
    assert decision_contexts
    assert all(item is not None and item.exploration_branch for item in decision_contexts)
    assert decision_contexts[0].exploration_branch == "pareto"
    assert decision_contexts[0].previous_objective_improvement is None
    assert decision_contexts[0].structural_collapse_rate == 0.0
    assert any(
        item.exploration_branch == "pareto"
        and item.guidance_alpha == 0.4
        and item.previous_objective_improvement == 1.0
        and item.structural_collapse_rate == 0.0
        for item in decision_contexts
    )


def test_fusion_search_cli_reads_inputs_and_emits_persisted_report(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    runtime, _encoders = _runtime(tmp_path / "artifacts")
    root = _initial_candidate()
    goal_path = tmp_path / "goal.json"
    parent_path = tmp_path / "parent.json"
    config_path = tmp_path / "config.json"
    context_path = tmp_path / "context.json"
    relations_path = tmp_path / "relations.json"
    goal_path.write_text(_goal().model_dump_json(), encoding="utf-8")
    parent_path.write_text(root.model_dump_json(), encoding="utf-8")
    config_path.write_text(_config(root).model_dump_json(), encoding="utf-8")
    context_path.write_text("[]", encoding="utf-8")
    relations_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "_fusion_runtime_from_environment",
        lambda _artifacts: runtime,
    )
    monkeypatch.setattr(
        cli_module,
        "build_generator_from_environment",
        lambda _generator, *, required: _DominatingGenerator(),
    )

    exit_code = main(
        [
            "fusion-search",
            "--search-id",
            "cli-search",
            "--goal",
            str(goal_path),
            "--parent",
            str(parent_path),
            "--run-config",
            str(config_path),
            "--generator",
            "mattergen",
            "--rounds",
            "1",
            "--frontier-width",
            "1",
            "--context",
            str(context_path),
            "--relations",
            str(relations_path),
            "--modality",
            ScientificModality.CRYSTAL_MATERIAL.value,
            "--expert",
            "expert-a",
            "--expert",
            "expert-b",
            "--required-evaluator",
            "expert-a",
            "--required-evaluator",
            "expert-b",
            "--artifacts",
            str(tmp_path / "ignored-by-monkeypatch"),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["report"]["search_id"] == "cli-search"
    assert payload["report"]["rounds_completed"] == 1
    artifact = payload["report_artifact"]
    stored = runtime.artifact_store.read_bytes(
        artifact["relative_path"], expected_sha256=artifact["sha256"]
    )
    assert len(stored) == artifact["byte_size"]


def test_exploration_and_search_types_are_public_package_exports() -> None:
    assert discovery_os.ExpertEvidenceStore is ExpertEvidenceStore
    assert discovery_os.FusionSearchRunner is FusionSearchRunner
    assert discovery_os.SearchCandidateRecord is SearchCandidateRecord
    assert "ExplorationBranch" in discovery_os.__all__
    assert "SearchRepresentationArtifactRef" in discovery_os.__all__


def test_control_sweep_repeats_alpha_temperature_and_emits_multi_candidate_ranking(
    tmp_path: Path,
) -> None:
    runtime, _encoders = _runtime(tmp_path)
    generator = _BatchGenerator()
    root = _initial_candidate()
    result = FusionSearchRunner(
        FusionLoopRunner(runtime, generator),
        ExpertEvidenceStore(runtime.artifact_store),
    ).run(
        search_id="control-sweep-search",
        goal=_goal(),
        initial_candidate=root,
        base_run_config=_config(root),
        rounds=1,
        frontier_width=4,
        expert_ids=["expert-a", "expert-b"],
        control_sweep=SearchControlSweep(
            points=[
                SearchControlPoint(alpha=0.25, temperature=1.4, label="explore"),
                SearchControlPoint(alpha=0.75, temperature=0.7, label="exploit"),
            ],
            include_adaptive_center=True,
            max_variants_per_parent=3,
        ),
        ranking_limit=10,
    )
    report = result.report

    assert len(report.cycle_records) == 3
    assert len(report.control_attempts) == 3
    assert all(item.success for item in report.control_attempts)
    assert {
        (item.controls.alpha, item.controls.temperature)
        for item in report.control_attempts
    } == {(0.25, 1.4), (0.5, 1.0), (0.75, 0.7)}
    assert len({item.run_config.effective_generator_seed for item in report.cycle_records}) == 3
    assert len(report.ranked_candidates) >= 2
    assert [item.rank for item in report.ranked_candidates] == list(
        range(1, len(report.ranked_candidates) + 1)
    )
    assert all(item.expert_property_vectors for item in report.ranked_candidates)
    assert all(item.branch_ranks for item in report.ranked_candidates)
    assert all(
        "raw scientific values are not averaged" in item.rationale[0]
        for item in report.ranked_candidates
    )
    assert report.ranked_candidates[0].candidate.candidate_ref == (
        report.ranked_candidates[0].candidate_ref
    )


def test_cli_control_sweep_is_enabled_by_default_and_can_be_disabled() -> None:
    parser = cli_module.make_parser()
    enabled = parser.parse_args(
        [
            "fusion-search",
            "--search-id",
            "sweep-enabled",
            "--goal",
            "goal.json",
            "--parent",
            "parent.json",
            "--run-config",
            "config.json",
            "--generator",
            "mattergen",
            "--rounds",
            "2",
        ]
    )
    assert enabled.no_control_sweep is False
    assert enabled.max_control_variants == 3
    assert enabled.ranking_limit == 50

    disabled = parser.parse_args(
        [
            "fusion-search",
            "--search-id",
            "sweep-disabled",
            "--goal",
            "goal.json",
            "--parent",
            "parent.json",
            "--run-config",
            "config.json",
            "--generator",
            "mattergen",
            "--rounds",
            "2",
            "--no-control-sweep",
        ]
    )
    assert disabled.no_control_sweep is True
