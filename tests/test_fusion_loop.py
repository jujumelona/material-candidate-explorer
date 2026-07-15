from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from discovery_os.artifacts import ArtifactStore
from discovery_os.fusion_loop import FusionLoopRunner, WorkspaceBenchmarkRunner
from discovery_os.fusion_reference import MeanFusionBackend
from discovery_os.fusion_registry import ExpertRegistry
from discovery_os.fusion_runtime import FusionRuntime, FusionRuntimeError
from discovery_os.fusion_schemas import (
    DiagnosticProperty,
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    ExpertProvenance,
    FeatureSemantics,
    FusionDecisionContext,
    FusionGenerationResponse,
    FusionBatchIterationReport,
    GenerationPairSlot,
    GeneratorProvenance,
    NumericTensor,
    ScientificModality,
    TensorRole,
    WorkspaceEntityInput,
    WorkspaceEntityRole,
    WorkspaceMode,
    WorkspacePairedRunReport,
    WorkspaceRelation,
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


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="closed-loop-goal",
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        title="Closed-loop fixture",
        scientific_question="Can the structured loop be executed deterministically?",
        objectives=[
            PropertyObjective(
                property_name="score",
                direction=ObjectiveDirection.MAXIMIZE,
                unit="arb",
            )
        ],
        validation_profile_id="general-materials-v1",
        candidate_types=[CandidateType.COMPOSITION],
    )


def _candidate(candidate_id: str, formula: str, score: float) -> Candidate:
    draft = Candidate(
        candidate_id=candidate_id,
        candidate_type=CandidateType.COMPOSITION,
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.CHEMICAL_FORMULA,
                value=formula,
                canonical=True,
            )
        ],
        attributes={"diagnostic_score": score},
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


def _protein() -> Candidate:
    draft = Candidate(
        candidate_id="protein-target",
        candidate_type=CandidateType.PROTEIN,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.PROTEIN_SEQUENCE,
                value="MKT",
                canonical=True,
            )
        ],
        attributes={"diagnostic_score": 0.5},
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


class _Encoder:
    def __init__(self, *, protein: bool = False, mutate: bool = False) -> None:
        self.mutate = mutate
        if protein:
            expert_id = "protein-fixture"
            modality = ScientificModality.PROTEIN_SEQUENCE
            candidate_type = CandidateType.PROTEIN
            representation = RepresentationKind.PROTEIN_SEQUENCE
        else:
            expert_id = "material-fixture"
            modality = ScientificModality.CRYSTAL_MATERIAL
            candidate_type = CandidateType.COMPOSITION
            representation = RepresentationKind.CHEMICAL_FORMULA
        self._descriptor = ExpertDescriptor(
            expert_id=expert_id,
            display_name=expert_id,
            adapter_version="1.0.0",
            modalities=[modality],
            supported_candidate_types=[candidate_type],
            supported_representations=[representation],
            feature_spaces=["common-projection-v1"],
        )

    @property
    def descriptor(self) -> ExpertDescriptor:
        return self._descriptor

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        if self.mutate:
            request.candidate.attributes["tampered"] = True
        score = float(request.candidate.attributes["diagnostic_score"])
        return ExpertFeaturePayload(
            workspace_entity_id=request.workspace_entity_id,
            candidate_ref=request.candidate.candidate_ref,
            expert_id=self.descriptor.expert_id,
            modality=request.modality,
            feature_space=request.feature_space,
            tensor=NumericTensor(shape=[2], values=[score, score + 1.0]),
            semantics=FeatureSemantics(
                tensor_role=TensorRole.GLOBAL_EMBEDDING,
                projection_id="common-projection-v1",
                pooling="mean",
                normalization="fixture-standardized",
            ),
            properties=[
                DiagnosticProperty(
                    property_name="score",
                    value=score,
                    unit="arb",
                    uncertainty=0.1,
                    source=self.descriptor.expert_id,
                )
            ],
            provenance=ExpertProvenance(
                expert_id=self.descriptor.expert_id,
                adapter_version="1.0.0",
                model_version="fixture-v1",
                code_revision="fixture-code",
                weight_revision="fixture-weight",
                projection_version="common-projection-v1",
                parameters_hash=stable_hash({"fixture": self.descriptor.expert_id}),
                seed=request.seed,
            ),
        )


class _Generator:
    def generate(self, request):
        suffix = "on" if request.workspace_mode == WorkspaceMode.ON else "off"
        score = 3.0 if suffix == "on" else 1.5
        formula = "MgB3" if suffix == "on" else "MgB2"
        parent_ref = request.parent_candidate.candidate_ref
        draft = Candidate(
            candidate_id=f"generated-{suffix}",
            candidate_type=CandidateType.COMPOSITION,
            domain=DiscoveryDomain.GENERAL_MATERIALS,
            representations=[
                CandidateRepresentation(
                    kind=RepresentationKind.CHEMICAL_FORMULA,
                    value=formula,
                    canonical=True,
                )
            ],
            parent_candidate_ids=[parent_ref.candidate_id],
            parent_candidate_refs=[parent_ref],
            attributes={"diagnostic_score": score},
        )
        generated = draft.model_copy(
            update={
                "candidate_ref": CandidateRef(
                    candidate_id=draft.candidate_id,
                    version=1,
                    content_hash=candidate_content_hash(draft),
                )
            }
        )
        config = request.run_config
        return FusionGenerationResponse(
            candidate=generated,
            provenance=GeneratorProvenance(
                generator_id=config.generator_id,
                generator_version=config.generator_version,
                code_revision=config.generator_code_revision,
                weight_revision=config.generator_weight_revision,
                parameters_hash=config.generator_parameters_hash,
                runtime_parameters_hash="6" * 64,
                seed=config.seed,
            ),
            pair_slots=[
                GenerationPairSlot(
                    pair_slot=0,
                    candidate_ref=generated.candidate_ref,
                    batch_seed=config.effective_generator_seed,
                    stream_position=0,
                )
            ],
        )


class _MutatingGenerator(_Generator):
    def generate(self, request):
        request.run_config.generator_id = "mutated-generator"
        return super().generate(request)


class _BatchGenerator(_Generator):
    def generate(self, request):
        parent_ref = request.parent_candidate.candidate_ref
        candidates = []
        for index in range(request.run_config.candidate_count):
            draft = Candidate(
                candidate_id=f"generated-branch-{index}",
                candidate_type=CandidateType.COMPOSITION,
                domain=DiscoveryDomain.GENERAL_MATERIALS,
                representations=[
                    CandidateRepresentation(
                        kind=RepresentationKind.CHEMICAL_FORMULA,
                        value=f"MgB{index + 2}",
                        canonical=True,
                    )
                ],
                parent_candidate_ids=[parent_ref.candidate_id],
                parent_candidate_refs=[parent_ref],
                attributes={"diagnostic_score": float(index + 2)},
            )
            candidates.append(
                draft.model_copy(
                    update={
                        "candidate_ref": CandidateRef(
                            candidate_id=draft.candidate_id,
                            version=1,
                            content_hash=candidate_content_hash(draft),
                        )
                    }
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
                runtime_parameters_hash="6" * 64,
                seed=config.seed,
            ),
            pair_slots=[
                GenerationPairSlot(
                    pair_slot=index,
                    candidate_ref=candidate.candidate_ref,
                    batch_seed=config.effective_generator_seed,
                    stream_position=index,
                )
                for index, candidate in enumerate(candidates)
            ],
        )


class _ReorderedOnGenerator(_BatchGenerator):
    def generate(self, request):
        response = super().generate(request)
        if request.workspace_mode == WorkspaceMode.ON:
            return response.model_copy(
                update={
                    "candidates": list(reversed(response.candidates)),
                }
            )
        return response


class _MissingPairSlotsGenerator(_Generator):
    def generate(self, request):
        return super().generate(request).model_copy(update={"pair_slots": []})


class _DuplicatePairSlotsGenerator(_BatchGenerator):
    def generate(self, request):
        response = super().generate(request)
        duplicated = list(response.pair_slots)
        if len(duplicated) > 1:
            duplicated[1] = duplicated[1].model_copy(update={"pair_slot": 0})
        return response.model_copy(update={"pair_slots": duplicated})


class _RuntimeDriftGenerator(_Generator):
    def generate(self, request):
        response = super().generate(request)
        runtime_hash = "6" * 64 if request.workspace_mode == WorkspaceMode.OFF else "7" * 64
        return response.model_copy(
            update={
                "provenance": response.provenance.model_copy(
                    update={"runtime_parameters_hash": runtime_hash}
                )
            }
        )


class _MissingRuntimeHashGenerator(_Generator):
    def generate(self, request):
        response = super().generate(request)
        return response.model_copy(
            update={
                "provenance": response.provenance.model_copy(
                    update={"runtime_parameters_hash": None}
                )
            }
        )


class _ExpectedRuntimeHashGenerator(_Generator):
    expected_runtime_parameters_hash = "7" * 64


def _runtime(tmp_path: Path, *, protein: bool = False, mutate: bool = False) -> FusionRuntime:
    registry = ExpertRegistry()
    registry.register(_Encoder(mutate=mutate))
    if protein:
        registry.register(_Encoder(protein=True))
    return FusionRuntime(registry, MeanFusionBackend(dimension=2), ArtifactStore(tmp_path))


def _config(mode: WorkspaceMode, parent: Candidate) -> WorkspaceRunConfig:
    return WorkspaceRunConfig(
        workspace_mode=mode,
        seed=17,
        goal_hash=stable_hash(_goal()),
        parent_candidate_ref=parent.candidate_ref,
        pair_key="closed-loop-pair",
        cohort_index=0,
        generator_id="fixture-generator",
        generator_version="1.0.0",
        generator_code_revision="fixture-generator-code",
        generator_weight_revision="fixture-generator-weight",
        generator_parameters_hash="1" * 64,
        decoder_config_hash="2" * 64,
        postprocessing_hash="3" * 64,
        resource_budget_hash="4" * 64,
        evaluator_panel_hash="5" * 64,
        candidate_count=1,
    )


def test_closed_loop_reextracts_child_and_updates_latent(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent", "MgB2", 1.0)
    report = FusionLoopRunner(runtime, _Generator()).iterate(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        run_config=_config(WorkspaceMode.ON, parent),
    )

    assert report.after_revision.cycle == 1
    assert report.after_revision.latent_state.state_version == 2
    assert report.after_revision.latent_state.previous_state_id == (
        report.before_revision.latent_state.state_id
    )
    assert all(
        item.candidate_ref == report.generation.candidate.candidate_ref
        for item in report.after_revision.feature_refs
    )
    assert parent.candidate_ref in report.generation.candidate.parent_candidate_refs


def test_closed_loop_binds_generation_alpha_and_previous_observation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-decision-context", "MgB2", 1.0)
    controls = _config(WorkspaceMode.ON, parent).generation_controls.model_copy(
        update={"alpha": 0.73}
    )
    config = _config(WorkspaceMode.ON, parent).model_copy(
        update={"generation_controls": controls}
    )
    context = FusionDecisionContext(
        guidance_alpha=0.73,
        previous_objective_improvement=1.0,
        structural_collapse_rate=0.25,
    )
    seen = []
    original_update = runtime.update

    def recording_update(**kwargs):
        seen.append(kwargs.get("decision_context"))
        return original_update(**kwargs)

    monkeypatch.setattr(runtime, "update", recording_update)

    FusionLoopRunner(runtime, _Generator()).iterate(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        run_config=config,
        decision_context=context,
    )

    assert len(seen) == 2
    assert all(item == context for item in seen)


def test_closed_loop_rejects_decision_alpha_that_differs_from_generator_controls(
    tmp_path: Path,
) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-mismatched-context", "MgB2", 1.0)

    with pytest.raises(FusionRuntimeError, match="alpha differs"):
        FusionLoopRunner(runtime, _Generator()).iterate(
            goal=_goal(),
            parent_candidate=parent,
            cycle=0,
            run_config=_config(WorkspaceMode.ON, parent),
            decision_context=FusionDecisionContext(guidance_alpha=0.9),
        )


def test_closed_loop_generates_and_re_evaluates_a_candidate_batch(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-batch", "MgB2", 1.0)
    config = _config(WorkspaceMode.ON, parent).model_copy(
        update={"candidate_count": 3}
    )

    report = FusionLoopRunner(runtime, _BatchGenerator()).iterate(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        run_config=config,
    )

    assert len(report.generation.generated_candidates) == 3
    assert len(report.after_revisions) == 3
    assert {
        item.latent_state.previous_state_id for item in report.after_revisions
    } == {report.before_revision.latent_state.state_id}
    assert {
        item.candidate_ref.candidate_id for item in report.after_revisions
    } == {"generated-branch-0", "generated-branch-1", "generated-branch-2"}


def test_closed_loop_materializes_one_shot_expert_iterable(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-iterable", "MgB2", 1.0)

    report = FusionLoopRunner(runtime, _Generator()).iterate(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        run_config=_config(WorkspaceMode.ON, parent),
        expert_ids=iter(["material-fixture"]),
    )

    assert report.after_revision.failed_expert_ids == []
    assert {item.expert_id for item in report.after_revision.feature_refs} == {
        "material-fixture"
    }


def test_configured_generator_runtime_hash_is_enforced(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-runtime-attestation", "MgB2", 1.0)

    with pytest.raises(FusionRuntimeError, match="runtime parameters"):
        FusionLoopRunner(runtime, _ExpectedRuntimeHashGenerator()).iterate(
            goal=_goal(),
            parent_candidate=parent,
            cycle=0,
            run_config=_config(WorkspaceMode.ON, parent),
        )


def test_expert_feature_cache_reuses_exact_prior_evaluation(tmp_path: Path) -> None:
    encoder = _Encoder()
    calls = 0
    original = encoder.encode

    def counted(request):
        nonlocal calls
        calls += 1
        return original(request)

    encoder.encode = counted
    registry = ExpertRegistry()
    registry.register(encoder)
    runtime = FusionRuntime(
        registry,
        MeanFusionBackend(dimension=2),
        ArtifactStore(tmp_path),
    )
    candidate = _candidate("cached-candidate", "SiC", 0.8)

    first = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=0,
        seed=17,
        workspace_mode=WorkspaceMode.OFF,
    )
    second = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=5,
        seed=17,
        workspace_mode=WorkspaceMode.OFF,
    )

    assert calls == 1
    assert first.feature_refs == second.feature_refs


def test_paired_runner_generates_real_off_on_arms(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent", "MgB2", 1.0)
    report = WorkspaceBenchmarkRunner(runtime, _Generator()).run_pair(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        off_config=_config(WorkspaceMode.OFF, parent),
        on_config=_config(WorkspaceMode.ON, parent),
    )

    assert report.comparison.paired_configuration is True
    assert report.comparison.objective_deltas[0].signed_improvement == pytest.approx(1.5)
    assert report.comparison.scientific_claim == "diagnostic_only"
    assert report.off_snapshot.candidate.candidate_id == "generated-off"
    assert report.on_snapshot.candidate.candidate_id == "generated-on"


def test_paired_on_arm_binds_configured_guidance_alpha(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-paired-context", "MgB2", 1.0)
    controls = _config(WorkspaceMode.ON, parent).generation_controls.model_copy(
        update={"alpha": 0.73}
    )
    off_config = _config(WorkspaceMode.OFF, parent).model_copy(
        update={"generation_controls": controls}
    )
    on_config = _config(WorkspaceMode.ON, parent).model_copy(
        update={"generation_controls": controls}
    )
    seen = []
    original_update = runtime.update

    def recording_update(**kwargs):
        seen.append((kwargs["workspace_mode"], kwargs.get("decision_context")))
        return original_update(**kwargs)

    monkeypatch.setattr(runtime, "update", recording_update)

    WorkspaceBenchmarkRunner(runtime, _Generator()).run_pair(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        off_config=off_config,
        on_config=on_config,
    )

    off_contexts = [context for mode, context in seen if mode == WorkspaceMode.OFF]
    on_contexts = [context for mode, context in seen if mode == WorkspaceMode.ON]
    assert off_contexts == [None, None]
    assert len(on_contexts) == 2
    assert all(context.guidance_alpha == 0.73 for context in on_contexts)
    assert all(context.previous_objective_improvement is None for context in on_contexts)
    assert all(context.structural_collapse_rate == 0.0 for context in on_contexts)


def test_paired_runner_compares_every_candidate_in_a_batch(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-paired-batch", "MgB2", 1.0)
    off_config = _config(WorkspaceMode.OFF, parent).model_copy(
        update={"candidate_count": 3}
    )
    on_config = _config(WorkspaceMode.ON, parent).model_copy(
        update={"candidate_count": 3}
    )

    report = WorkspaceBenchmarkRunner(runtime, _BatchGenerator()).run_pair(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        off_config=off_config,
        on_config=on_config,
    )

    assert len(report.off_generation.generated_candidates) == 3
    assert len(report.on_generation.generated_candidates) == 3
    assert len(report.off_snapshots) == 3
    assert len(report.on_snapshots) == 3
    assert len(report.comparisons) == 3
    assert [row.off_candidate_ref for row in report.comparisons] == [
        row.candidate.candidate_ref for row in report.off_snapshots
    ]
    assert [row.on_candidate_ref for row in report.comparisons] == [
        row.candidate.candidate_ref for row in report.on_snapshots
    ]
    with pytest.raises(AttributeError, match="more than one comparison"):
        _ = report.comparison


def test_paired_runner_rejects_reordered_candidate_metadata(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-reordered", "MgB2", 1.0)
    off_config = _config(WorkspaceMode.OFF, parent).model_copy(
        update={"candidate_count": 3}
    )
    on_config = _config(WorkspaceMode.ON, parent).model_copy(
        update={"candidate_count": 3}
    )

    with pytest.raises(FusionRuntimeError, match="invalid response"):
        WorkspaceBenchmarkRunner(runtime, _ReorderedOnGenerator()).run_pair(
            goal=_goal(),
            parent_candidate=parent,
            cycle=0,
            off_config=off_config,
            on_config=on_config,
        )


@pytest.mark.parametrize(
    "generator,candidate_count,error",
    [
        (_MissingPairSlotsGenerator(), 1, "pair-slot metadata"),
        (_DuplicatePairSlotsGenerator(), 2, "invalid response"),
    ],
)
def test_paired_runner_rejects_missing_or_duplicate_pair_slots(
    tmp_path: Path,
    generator,
    candidate_count: int,
    error: str,
) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate(f"parent-slots-{candidate_count}", "MgB2", 1.0)
    off_config = _config(WorkspaceMode.OFF, parent).model_copy(
        update={"candidate_count": candidate_count}
    )
    on_config = _config(WorkspaceMode.ON, parent).model_copy(
        update={"candidate_count": candidate_count}
    )

    with pytest.raises(FusionRuntimeError, match=error):
        WorkspaceBenchmarkRunner(runtime, generator).run_pair(
            goal=_goal(),
            parent_candidate=parent,
            cycle=0,
            off_config=off_config,
            on_config=on_config,
        )


@pytest.mark.parametrize("generator", [_RuntimeDriftGenerator(), _MissingRuntimeHashGenerator()])
def test_paired_runner_requires_equal_non_null_runtime_attestation(
    tmp_path: Path,
    generator,
) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-runtime-drift", "MgB2", 1.0)

    with pytest.raises(FusionRuntimeError, match="equal attested runtime parameters"):
        WorkspaceBenchmarkRunner(runtime, generator).run_pair(
            goal=_goal(),
            parent_candidate=parent,
            cycle=0,
            off_config=_config(WorkspaceMode.OFF, parent),
            on_config=_config(WorkspaceMode.ON, parent),
        )


def test_batch_report_rejects_cross_goal_or_workspace_lineage(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent-report-lineage", "MgB2", 1.0)
    config = _config(WorkspaceMode.ON, parent).model_copy(update={"candidate_count": 3})
    report = FusionLoopRunner(runtime, _BatchGenerator()).iterate(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        run_config=config,
    )
    payload = json.loads(report.model_dump_json())
    child = payload["after_revisions"][0]
    child["goal_hash"] = "f" * 64
    child["latent_state"]["goal_hash"] = "f" * 64
    child["workspace"]["workspace_id"] = "different-workspace"
    child["latent_state"]["workspace_id"] = "different-workspace"

    with pytest.raises(ValidationError, match="outside its workspace or goal|breaks lineage"):
        FusionBatchIterationReport.model_validate_json(json.dumps(payload), strict=True)


@pytest.mark.parametrize("mutation", ["candidate_count", "pair_key"])
def test_paired_report_rejects_cross_arm_configuration_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate(f"parent-report-{mutation}", "MgB2", 1.0)
    report = WorkspaceBenchmarkRunner(runtime, _Generator()).run_pair(
        goal=_goal(),
        parent_candidate=parent,
        cycle=0,
        off_config=_config(WorkspaceMode.OFF, parent),
        on_config=_config(WorkspaceMode.ON, parent),
    )
    payload = json.loads(report.model_dump_json())
    if mutation == "candidate_count":
        payload["off_snapshots"][0]["run_config"]["candidate_count"] = 2
        payload["on_snapshots"][0]["run_config"]["candidate_count"] = 2
    else:
        payload["on_snapshots"][0]["run_config"]["pair_key"] = "different-pair"

    with pytest.raises(ValidationError, match="candidate count|configurations do not match"):
        WorkspacePairedRunReport.model_validate_json(json.dumps(payload), strict=True)


def test_multi_entity_workspace_fuses_candidate_and_target(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, protein=True)
    candidate = _candidate("candidate", "MgB2", 1.0)
    target = WorkspaceEntityInput(
        entity_id="target-protein",
        role=WorkspaceEntityRole.TARGET,
        candidate=_protein(),
    )
    report = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=0,
        seed=17,
        workspace_mode=WorkspaceMode.ON,
        context_entities=[target],
        relations=[
            WorkspaceRelation(
                relation_id="candidate-target",
                subject_entity_id="primary",
                predicate="interacts_with",
                object_entity_id="target-protein",
            )
        ],
    )

    assert len(report.workspace.entities) == 2
    assert {item.expert_id for item in report.feature_refs} == {
        "material-fixture",
        "protein-fixture",
    }
    assert report.latent_state.workspace_id == report.workspace.workspace_id


def test_workspace_distinguishes_two_roles_for_the_same_candidate_ref(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    candidate = _candidate("shared-material", "MgB2", 1.0)
    report = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=0,
        seed=17,
        workspace_mode=WorkspaceMode.ON,
        context_entities=[
            WorkspaceEntityInput(
                entity_id="comparison-instance",
                role=WorkspaceEntityRole.CONTEXT,
                candidate=candidate,
            )
        ],
    )

    assert {item.workspace_entity_id for item in report.feature_refs} == {
        "primary",
        "comparison-instance",
    }
    assert len({item.feature_id for item in report.feature_refs}) == 2


def test_unrelated_previous_state_and_encoder_mutation_fail_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    first = runtime.update(
        goal=_goal(),
        candidate=_candidate("candidate-a", "MgB2", 1.0),
        cycle=0,
        seed=17,
        workspace_mode=WorkspaceMode.ON,
    )
    with pytest.raises(FusionRuntimeError, match="unrelated"):
        runtime.update(
            goal=_goal(),
            candidate=_candidate("candidate-b", "MgB3", 2.0),
            cycle=1,
            seed=17,
            workspace_mode=WorkspaceMode.ON,
            previous_state=first.latent_state,
        )

    mutating_runtime = _runtime(tmp_path / "mutating", mutate=True)
    with pytest.raises(FusionRuntimeError, match="mutated"):
        mutating_runtime.update(
            goal=_goal(),
            candidate=_candidate("candidate-c", "MgB2", 1.0),
            cycle=0,
            seed=17,
            workspace_mode=WorkspaceMode.ON,
        )


def test_generator_request_mutation_fails_closed(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    parent = _candidate("parent", "MgB2", 1.0)
    with pytest.raises(FusionRuntimeError, match="generator mutated"):
        FusionLoopRunner(runtime, _MutatingGenerator()).iterate(
            goal=_goal(),
            parent_candidate=parent,
            cycle=0,
            run_config=_config(WorkspaceMode.ON, parent),
        )


def test_workspace_relations_cannot_change_within_latent_lineage(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path, protein=True)
    candidate = _candidate("candidate", "MgB2", 1.0)
    target = WorkspaceEntityInput(
        entity_id="target-protein",
        role=WorkspaceEntityRole.TARGET,
        candidate=_protein(),
    )
    first = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=0,
        seed=17,
        workspace_mode=WorkspaceMode.ON,
        context_entities=[target],
        relations=[
            WorkspaceRelation(
                relation_id="candidate-target",
                subject_entity_id="primary",
                predicate="interacts_with",
                object_entity_id="target-protein",
            )
        ],
    )

    with pytest.raises(FusionRuntimeError, match="relations changed"):
        runtime.update(
            goal=_goal(),
            candidate=candidate,
            cycle=1,
            seed=17,
            workspace_mode=WorkspaceMode.ON,
            previous_state=first.latent_state,
            context_entities=[target],
            relations=[
                WorkspaceRelation(
                    relation_id="candidate-target",
                    subject_entity_id="primary",
                    predicate="evaluated_in",
                    object_entity_id="target-protein",
                )
            ],
        )
