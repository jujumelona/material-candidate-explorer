from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from discovery_os.artifacts import ArtifactStore
from discovery_os.fusion_metrics import compare_workspace_snapshots
from discovery_os.fusion_reference import MeanFusionBackend
from discovery_os.fusion_registry import ExpertRegistry
from discovery_os.fusion_runtime import FusionRuntime, FusionRuntimeError
from discovery_os.fusion_schemas import (
    ContentArtifactRef,
    DiagnosticProperty,
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    ExpertProvenance,
    FeatureSemantics,
    FeatureStatus,
    NumericTensor,
    ScientificModality,
    TensorRole,
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


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="goal-fusion",
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        title="Fusion comparison",
        scientific_question="Does a paired workspace arm move the diagnostic score?",
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


def _candidate(
    candidate_id: str,
    formula: str,
    *,
    score: float,
    coordinates: list[list[float]] | None = None,
    lattice: list[list[float]] | None = None,
    version: int = 1,
) -> Candidate:
    attributes: dict = {"diagnostic_score": score}
    if coordinates is not None:
        attributes["coordinates"] = coordinates
        attributes["coordinate_labels"] = ["Mg", "B"]
    if lattice is not None:
        attributes["lattice"] = lattice
    candidate = Candidate(
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
        attributes=attributes,
    )
    return candidate.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate_id,
                version=version,
                content_hash=candidate_content_hash(candidate),
            )
        }
    )


class _Encoder:
    def __init__(self, expert_id: str = "fixture-expert", *, fail: bool = False) -> None:
        self.fail = fail
        self._descriptor = ExpertDescriptor(
            expert_id=expert_id,
            display_name="Fixture expert",
            adapter_version="1.0.0",
            modalities=[ScientificModality.CRYSTAL_MATERIAL],
            supported_candidate_types=[CandidateType.COMPOSITION],
            supported_representations=[RepresentationKind.CHEMICAL_FORMULA],
            feature_spaces=["fixture-aligned-v1"],
        )

    @property
    def descriptor(self) -> ExpertDescriptor:
        return self._descriptor

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        if self.fail:
            raise RuntimeError("fixture failed")
        score = float(request.candidate.attributes["diagnostic_score"])
        return ExpertFeaturePayload(
            workspace_entity_id=request.workspace_entity_id,
            candidate_ref=request.candidate.candidate_ref,
            expert_id=self.descriptor.expert_id,
            modality=request.modality,
            feature_space=request.feature_space,
            status=FeatureStatus.SUCCESS,
            tensor=NumericTensor(shape=[2], values=[score, score + 1.0]),
            semantics=FeatureSemantics(
                tensor_role=TensorRole.GLOBAL_EMBEDDING,
                projection_id="fixture-aligned-v1",
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
                model_version="fixture-1",
                code_revision="fixture-code",
                weight_revision="fixture-weight",
                parameters_hash=stable_hash({"fixture": True}),
                projection_version="fixture-aligned-v1",
                seed=request.seed,
            ),
        )


def _runtime(tmp_path: Path, *, with_failure: bool = False) -> FusionRuntime:
    registry = ExpertRegistry()
    registry.register(_Encoder())
    if with_failure:
        registry.register(_Encoder("failed-expert", fail=True))
    return FusionRuntime(registry, MeanFusionBackend(dimension=2), ArtifactStore(tmp_path))


def _run_config(
    mode: WorkspaceMode,
    parent_ref: CandidateRef,
    *,
    seed: int = 7,
) -> WorkspaceRunConfig:
    return WorkspaceRunConfig(
        workspace_mode=mode,
        seed=seed,
        goal_hash=stable_hash(_goal()),
        parent_candidate_ref=parent_ref,
        pair_key="fixture-pair",
        cohort_index=0,
        generator_id="fixture-generator",
        generator_version="1.0.0",
        generator_code_revision="fixture-generator-code",
        generator_weight_revision="fixture-generator-weight",
        generator_parameters_hash="d" * 64,
        decoder_config_hash="a" * 64,
        postprocessing_hash="b" * 64,
        resource_budget_hash="c" * 64,
        evaluator_panel_hash="e" * 64,
        candidate_count=1,
    )


def test_tensor_and_artifact_contracts_reject_invalid_payloads() -> None:
    with pytest.raises(ValidationError, match="shape"):
        NumericTensor(shape=[3], values=[1.0, 2.0])
    with pytest.raises(ValidationError, match="parent|relative"):
        ContentArtifactRef(
            artifact_id="artifact",
            relative_path="../outside.json",
            sha256="a" * 64,
            media_type="application/json",
            byte_size=1,
        )


def test_fusion_runtime_keeps_workspace_off_feature_only(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    candidate = _candidate("candidate-off", "MgB2", score=1.0)

    report = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=0,
        seed=7,
        workspace_mode=WorkspaceMode.OFF,
    )

    assert report.latent_state is None
    assert report.revision_proposal is None
    assert len(report.feature_refs) == 1
    feature = report.feature_refs[0]
    assert (tmp_path / feature.artifact.relative_path).is_file()
    payload = json.loads((tmp_path / feature.artifact.relative_path).read_text(encoding="utf-8"))
    assert payload["tensor"]["values"] == [1.0, 2.0]


def test_fusion_runtime_updates_latent_and_reloads_previous_state(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    candidate = _candidate("candidate-on", "MgB2", score=2.0)

    first = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=0,
        seed=11,
        workspace_mode=WorkspaceMode.ON,
    )
    second = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=1,
        seed=11,
        workspace_mode=WorkspaceMode.ON,
        previous_state=first.latent_state,
    )

    assert first.latent_state is not None
    assert first.revision_proposal is not None
    assert second.latent_state is not None
    assert second.latent_state.state_version == 2
    assert second.latent_state.previous_state_id == first.latent_state.state_id
    assert first.latent_state.latent_artifact.relative_path.startswith("fusion/latents/")


def test_fusion_runtime_preserves_optional_expert_failure(tmp_path) -> None:
    runtime = _runtime(tmp_path, with_failure=True)
    candidate = _candidate("candidate-partial", "MgB2", score=2.0)

    report = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=0,
        seed=5,
        workspace_mode=WorkspaceMode.ON,
    )

    assert report.failed_expert_ids == ["failed-expert"]
    assert any("fixture failed" in warning for warning in report.warnings)
    assert report.latent_state is not None


def test_fusion_runtime_rejects_stale_candidate_hash(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    candidate = _candidate("candidate-stale", "MgB2", score=2.0)
    candidate.attributes["diagnostic_score"] = 99.0

    with pytest.raises(FusionRuntimeError, match="stale"):
        runtime.update(
            goal=_goal(),
            candidate=candidate,
            cycle=0,
            seed=5,
            workspace_mode=WorkspaceMode.ON,
        )


def test_workspace_comparison_reports_paired_diagnostics_only(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    off_candidate = _candidate(
        "candidate-off",
        "MgB2",
        score=1.0,
        coordinates=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        lattice=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    on_candidate = _candidate(
        "candidate-on",
        "MgB3",
        score=2.5,
        coordinates=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        lattice=[[1.1, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    )
    off_report = runtime.update(
        goal=_goal(),
        candidate=off_candidate,
        cycle=0,
        seed=7,
        workspace_mode=WorkspaceMode.OFF,
    )
    on_report = runtime.update(
        goal=_goal(),
        candidate=on_candidate,
        cycle=0,
        seed=7,
        workspace_mode=WorkspaceMode.ON,
    )
    parent_ref = _candidate("candidate-parent", "MgB2", score=0.0).candidate_ref
    off_snapshot = runtime.snapshot(
        off_candidate,
        off_report,
        _run_config(WorkspaceMode.OFF, parent_ref),
    )
    on_snapshot = runtime.snapshot(
        on_candidate,
        on_report,
        _run_config(WorkspaceMode.ON, parent_ref),
    )

    comparison = compare_workspace_snapshots(
        off_snapshot,
        on_snapshot,
        _goal(),
        artifact_store=runtime.artifact_store,
    )

    assert comparison.paired_configuration is True
    assert comparison.element_total_variation is not None
    assert comparison.element_total_variation > 0
    assert comparison.element_jensen_shannon_divergence is not None
    assert comparison.ordered_coordinate_rms_displacement == pytest.approx(0.2 / 2**0.5)
    assert comparison.lattice_frobenius_distance == pytest.approx(0.1)
    assert comparison.objective_deltas[0].signed_improvement == pytest.approx(1.5)
    assert comparison.scientific_claim == "diagnostic_only"


def test_workspace_comparison_detects_unpaired_seed(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    off_candidate = _candidate("candidate-off", "MgB2", score=1.0)
    on_candidate = _candidate("candidate-on", "MgB2", score=1.5)
    off_report = runtime.update(
        goal=_goal(),
        candidate=off_candidate,
        cycle=0,
        seed=7,
        workspace_mode=WorkspaceMode.OFF,
    )
    on_report = runtime.update(
        goal=_goal(),
        candidate=on_candidate,
        cycle=0,
        seed=8,
        workspace_mode=WorkspaceMode.ON,
    )
    parent_ref = _candidate("candidate-parent", "MgB2", score=0.0).candidate_ref
    off_snapshot = runtime.snapshot(
        off_candidate,
        off_report,
        _run_config(WorkspaceMode.OFF, parent_ref, seed=7),
    )
    on_snapshot = runtime.snapshot(
        on_candidate,
        on_report,
        _run_config(WorkspaceMode.ON, parent_ref, seed=8),
    )

    comparison = compare_workspace_snapshots(
        off_snapshot,
        on_snapshot,
        _goal(),
        artifact_store=runtime.artifact_store,
    )

    assert comparison.paired_configuration is False
    assert any("differs" in caveat for caveat in comparison.caveats)


def test_workspace_comparison_marks_ood_objectives_incomparable(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    off_candidate = _candidate("candidate-off", "MgB2", score=1.0)
    on_candidate = _candidate("candidate-on", "MgB3", score=5.0)
    off_report = runtime.update(
        goal=_goal(),
        candidate=off_candidate,
        cycle=0,
        seed=7,
        workspace_mode=WorkspaceMode.OFF,
    )
    on_report = runtime.update(
        goal=_goal(),
        candidate=on_candidate,
        cycle=0,
        seed=7,
        workspace_mode=WorkspaceMode.ON,
    )
    parent_ref = _candidate("candidate-parent", "MgB2", score=0.0).candidate_ref
    off_snapshot = runtime.snapshot(
        off_candidate,
        off_report,
        _run_config(WorkspaceMode.OFF, parent_ref),
    )
    on_snapshot = runtime.snapshot(
        on_candidate,
        on_report,
        _run_config(WorkspaceMode.ON, parent_ref),
    ).model_copy(
        update={
            "aggregate_properties": [
                DiagnosticProperty(
                    property_name="score",
                    value=5.0,
                    unit="arb",
                    uncertainty=1.0,
                    out_of_domain=True,
                    source="fixture-expert",
                )
            ]
        }
    )

    with pytest.raises(ValueError, match="aggregate"):
        compare_workspace_snapshots(
            off_snapshot,
            on_snapshot,
            _goal(),
            artifact_store=runtime.artifact_store,
        )
    delta = compare_workspace_snapshots(off_snapshot, on_snapshot, _goal()).objective_deltas[0]

    assert delta.comparable is False
    assert delta.out_of_domain is True
    assert delta.signed_improvement is None


def test_snapshot_rejects_forged_feature_provenance(tmp_path) -> None:
    runtime = _runtime(tmp_path)
    candidate = _candidate("candidate-forged", "MgB2", score=1.0)
    report = runtime.update(
        goal=_goal(),
        candidate=candidate,
        cycle=0,
        seed=7,
        workspace_mode=WorkspaceMode.OFF,
    )
    original = report.feature_refs[0]
    forged = original.model_copy(
        update={
            "provenance": original.provenance.model_copy(
                update={"weight_revision": "forged-weight"}
            )
        }
    )
    forged_report = report.model_copy(update={"feature_refs": [forged]})

    with pytest.raises(FusionRuntimeError, match="artifact"):
        runtime.snapshot(
            candidate,
            forged_report,
            _run_config(WorkspaceMode.OFF, candidate.candidate_ref),
        )
