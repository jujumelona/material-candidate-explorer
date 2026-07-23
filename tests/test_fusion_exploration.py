from __future__ import annotations

from pathlib import Path

import pytest

from discovery_os.artifacts import ArtifactStore
from discovery_os.fusion_exploration import (
    AdaptiveGenerationScheduler,
    CandidatePool,
    CandidatePoolEntry,
    DeterministicExplorationSelector,
    ExpertEvidenceConflict,
    ExpertEvidenceStore,
    ExplorationBranch,
)
from discovery_os.fusion_schemas import (
    ContentArtifactRef,
    DiagnosticProperty,
    ExpertFeaturePayload,
    ExpertFeatureRef,
    ExpertProvenance,
    FeatureSemantics,
    FeatureStatus,
    GenerationControls,
    NumericTensor,
    ScientificModality,
    TensorRole,
)
from discovery_os.hashing import candidate_content_hash, stable_hash
from discovery_os.schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    GoalConstraint,
    ObjectiveDirection,
    PropertyObjective,
    RepresentationKind,
)


def _candidate(candidate_id: str, formula: str) -> Candidate:
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


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="exploration-goal",
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        title="Explore without averaging experts",
        scientific_question="Which candidates preserve independent expert tradeoffs?",
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


def _payload_and_ref(
    artifacts: ArtifactStore,
    candidate: Candidate,
    expert_id: str,
    properties: dict[str, float],
    *,
    out_of_domain: bool = False,
    failed: bool = False,
    tensor_values: tuple[float, float] = (1.0, 2.0),
    goal_hash: str | None = None,
) -> tuple[ExpertFeaturePayload, ExpertFeatureRef]:
    feature_id = f"feat-{candidate.candidate_id}-{expert_id}"
    provenance = ExpertProvenance(
        expert_id=expert_id,
        adapter_version="1.0.0",
        model_version=f"{expert_id}-model-v1",
        code_revision=f"{expert_id}-code-v1",
        weight_revision=f"{expert_id}-weight-v1",
        parameters_hash=stable_hash(
            {"candidate": candidate.candidate_ref, "expert": expert_id}
        ),
        projection_version=f"{expert_id}-projection-v1",
        seed=7,
    )
    diagnostic = [
        DiagnosticProperty(
            property_name=name,
            value=value,
            unit="arb",
            uncertainty=0.1,
            out_of_domain=out_of_domain,
            source=expert_id,
        )
        for name, value in sorted(properties.items())
    ]
    payload = ExpertFeaturePayload(
        workspace_entity_id="primary",
        candidate_ref=candidate.candidate_ref,
        expert_id=expert_id,
        modality=ScientificModality.CRYSTAL_MATERIAL,
        feature_space=f"{expert_id}-space-v1",
        status=FeatureStatus.FAILED if failed else FeatureStatus.SUCCESS,
        tensor=None if failed else NumericTensor(shape=[2], values=list(tensor_values)),
        semantics=(
            None
            if failed
            else FeatureSemantics(
                tensor_role=TensorRole.GLOBAL_EMBEDDING,
                projection_id=f"{expert_id}-projection-v1",
                pooling="mean",
                normalization="fixture-standardized",
            )
        ),
        properties=[] if failed else diagnostic,
        provenance=provenance,
    )
    source_path, source_sha = artifacts.write_json(
        f"fusion/source-features/{feature_id}.json", payload
    )
    source_size = len(artifacts.read_bytes(source_path))
    ref = ExpertFeatureRef(
        feature_id=feature_id,
        workspace_entity_id="primary",
        candidate_ref=candidate.candidate_ref,
        goal_hash=goal_hash or stable_hash(_goal()),
        expert_id=expert_id,
        modality=payload.modality,
        feature_space=payload.feature_space,
        status=payload.status,
        artifact=ContentArtifactRef(
            artifact_id=f"artifact-{feature_id}",
            relative_path=source_path,
            sha256=source_sha,
            media_type="application/json",
            byte_size=source_size,
        ),
        tensor_dtype=payload.tensor.dtype if payload.tensor is not None else None,
        tensor_shape=payload.tensor.shape if payload.tensor is not None else [],
        semantics=payload.semantics,
        properties=payload.properties,
        quality_flags=payload.quality_flags,
        warnings=payload.warnings,
        provenance=payload.provenance,
    )
    return payload, ref


def _store_candidate_panel(
    evidence: ExpertEvidenceStore,
    artifacts: ArtifactStore,
    candidate: Candidate,
    rows: dict[str, dict[str, float]],
    *,
    out_of_domain_expert: str | None = None,
    failed_expert: str | None = None,
) -> CandidatePoolEntry:
    evidence_ids = []
    for expert_id, properties in sorted(rows.items()):
        payload, ref = _payload_and_ref(
            artifacts,
            candidate,
            expert_id,
            properties,
            out_of_domain=expert_id == out_of_domain_expert,
            failed=expert_id == failed_expert,
            tensor_values=(10_000.0, -10_000.0) if expert_id == "expert-b" else (1.0, 2.0),
        )
        evidence_ids.append(evidence.put(payload, ref).evidence_id)
    return CandidatePoolEntry(candidate=candidate, evidence_ids=evidence_ids)


def test_expert_evidence_store_preserves_originals_and_rebuilds_indexes(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    candidate = _candidate("candidate-a", "MgB2")
    payload, ref = _payload_and_ref(
        artifacts,
        candidate,
        "expert-a",
        {"stability": 0.8, "target_score": 0.6},
        tensor_values=(123.0, -456.0),
    )

    stored = evidence.put(payload, ref)
    envelope = evidence.load(stored)

    assert envelope.payload == payload
    assert envelope.feature_ref == ref
    assert envelope.payload.tensor.values == [123.0, -456.0]
    assert stored.artifact.relative_path.startswith(
        "fusion/expert-evidence/objects/"
    )
    assert Path(stored.artifact.relative_path).stem == stored.artifact.sha256
    assert artifacts.resolve(stored.artifact.relative_path).is_relative_to(artifacts.root)

    restarted = ExpertEvidenceStore(ArtifactStore(tmp_path))
    assert restarted.by_candidate(candidate) == [stored]
    assert restarted.by_evaluator("expert-a") == [stored]
    assert restarted.by_cache_key(stored.cache_key) == [stored]
    assert restarted.query(
        candidate=candidate.candidate_ref,
        evaluator_id="expert-a",
        cache_key=stored.cache_key,
    ) == [stored]


def test_expert_evidence_store_scans_objects_once_per_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    candidate = _candidate("candidate-indexed", "SiC")
    stored_rows = []
    for expert_id in ("expert-a", "expert-b"):
        payload, ref = _payload_and_ref(
            artifacts,
            candidate,
            expert_id,
            {"stability": 0.7, "target_score": 0.6},
        )
        stored_rows.append(evidence.put(payload, ref))

    real_rglob = Path.rglob
    scan_count = 0
    object_root = artifacts.resolve(ExpertEvidenceStore.PREFIX)

    def counting_rglob(path: Path, pattern: str):
        nonlocal scan_count
        if path == object_root and pattern == "*.json":
            scan_count += 1
        return real_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", counting_rglob)
    restarted = ExpertEvidenceStore(ArtifactStore(tmp_path))
    for _ in range(5):
        assert restarted.by_candidate(candidate) == sorted(
            stored_rows, key=lambda item: item.evidence_id
        )
        assert restarted.by_evaluator("expert-a") == [stored_rows[0]]
        assert restarted.get(stored_rows[1].evidence_id) == stored_rows[1]
    assert scan_count == 1


def test_expert_evidence_store_rejects_same_cache_key_with_changed_output(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    candidate = _candidate("candidate-conflict", "LiFePO4")
    first, first_ref = _payload_and_ref(
        artifacts,
        candidate,
        "expert-a",
        {"stability": 0.5, "target_score": 0.5},
    )
    stored = evidence.put(first, first_ref)

    changed = first.model_copy(
        update={
            "properties": [
                DiagnosticProperty(
                    property_name="stability",
                    value=0.9,
                    unit="arb",
                    source="expert-a",
                ),
                DiagnosticProperty(
                    property_name="target_score",
                    value=0.5,
                    unit="arb",
                    source="expert-a",
                ),
            ]
        }
    )
    changed_path, changed_sha = artifacts.write_json(
        "fusion/source-features/changed-output.json", changed
    )
    changed_ref = first_ref.model_copy(
        update={
            "feature_id": "feat-candidate-conflict-expert-a-changed",
            "artifact": ContentArtifactRef(
                artifact_id="artifact-changed-output",
                relative_path=changed_path,
                sha256=changed_sha,
                media_type="application/json",
                byte_size=len(artifacts.read_bytes(changed_path)),
            ),
            "properties": changed.properties,
        }
    )

    assert evidence.cache_key_for(
        changed,
        goal_hash=changed_ref.goal_hash,
    ) == stored.cache_key
    with pytest.raises(ExpertEvidenceConflict, match="different original output"):
        evidence.put(changed, changed_ref)


def test_expert_evidence_cache_identity_separates_goals_for_the_same_output(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    candidate = _candidate("candidate-multi-goal", "BN")
    first_goal_hash = stable_hash(_goal())
    second_goal_hash = stable_hash(
        _goal().model_copy(
            update={
                "goal_id": "exploration-goal-2",
                "scientific_question": "Does a different goal preserve cache isolation?",
            }
        )
    )
    payload, first_ref = _payload_and_ref(
        artifacts,
        candidate,
        "expert-a",
        {"stability": 0.5, "target_score": 0.5},
        goal_hash=first_goal_hash,
    )
    second_ref = first_ref.model_copy(update={"goal_hash": second_goal_hash})

    first = evidence.put(payload, first_ref)
    second = evidence.put(payload, second_ref)

    assert first.cache_key != second.cache_key
    assert first.evidence_id != second.evidence_id
    assert first.goal_hash == first_goal_hash
    assert second.goal_hash == second_goal_hash
    assert evidence.by_cache_key(first.cache_key) == [first]
    assert evidence.by_cache_key(second.cache_key) == [second]
    assert {item.evidence_id for item in evidence.by_candidate(candidate)} == {
        first.evidence_id,
        second.evidence_id,
    }


def test_selector_builds_deterministic_pareto_novelty_and_disagreement_branches(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    rows = [
        _store_candidate_panel(
            evidence,
            artifacts,
            _candidate("candidate-a", "MgB2"),
            {
                "expert-a": {"stability": 0.90, "target_score": 0.70},
                "expert-b": {"stability": 0.80, "target_score": 0.60},
            },
        ),
        _store_candidate_panel(
            evidence,
            artifacts,
            _candidate("candidate-b", "MgB3"),
            {
                "expert-a": {"stability": 0.70, "target_score": 0.95},
                "expert-b": {"stability": 0.70, "target_score": 0.90},
            },
        ),
        _store_candidate_panel(
            evidence,
            artifacts,
            _candidate("candidate-c", "MgB"),
            {
                "expert-a": {"stability": 0.60, "target_score": 0.50},
                "expert-b": {"stability": 0.60, "target_score": 0.50},
            },
        ),
        _store_candidate_panel(
            evidence,
            artifacts,
            _candidate("candidate-disagreement", "Mg2B3"),
            {
                "expert-a": {"stability": 0.90, "target_score": 0.90},
                "expert-b": {"stability": 0.10, "target_score": 0.10},
            },
        ),
    ]
    pool = CandidatePool(pool_id="fixture-pool", entries=rows)
    selector = DeterministicExplorationSelector(
        evidence, disagreement_threshold=0.50
    )

    first = selector.select(pool, _goal(), limit_per_branch=4)
    second = selector.select(pool, _goal(), limit_per_branch=4)

    assert first == second
    assert [item.branch for item in first.branches] == [
        branch.value for branch in ExplorationBranch
    ]
    by_branch = {item.branch: item.candidates for item in first.branches}
    pareto_ids = {
        item.candidate_ref.candidate_id
        for item in by_branch[ExplorationBranch.PARETO]
    }
    assert pareto_ids == {
        "candidate-a",
        "candidate-b",
        "candidate-disagreement",
    }
    disagreement = by_branch[ExplorationBranch.EXPERT_DISAGREEMENT]
    assert [item.candidate_ref.candidate_id for item in disagreement] == [
        "candidate-disagreement"
    ]
    assert disagreement[0].expert_property_vectors["expert-a"][0].value in {
        0.9
    }
    assert set(disagreement[0].expert_property_vectors) == {
        "expert-a",
        "expert-b",
    }
    assert first.scientific_claim == "diagnostic_only"


def test_selector_never_uses_raw_energy_to_dominate_another_composition(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    goal = DiscoveryGoal(
        goal_id="composition-scoped-energy",
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        title="Do not compare raw energy gauges across compositions",
        scientific_question="Which candidates are non-dominated within composition?",
        objectives=[
            PropertyObjective(
                property_name="energy_per_atom",
                direction=ObjectiveDirection.MINIMIZE,
                unit="arb",
                required=True,
            )
        ],
        validation_profile_id="raw-energy-scope-v1",
        candidate_types=[CandidateType.COMPOSITION],
    )

    def entry(candidate_id: str, formula: str, energy: float) -> CandidatePoolEntry:
        return _store_candidate_panel(
            evidence,
            artifacts,
            _candidate(candidate_id, formula),
            {
                "expert-a": {"energy_per_atom": energy},
                "expert-b": {"energy_per_atom": energy + 0.01},
            },
        )

    pool = CandidatePool(
        pool_id="composition-scoped-pool",
        entries=[
            entry("different-low", "Li", -100.0),
            entry("different-high", "LiO", -1.0),
            entry("same-better", "SiC", -2.0),
            entry("same-worse", "SiC", -1.0),
        ],
    )

    selection = DeterministicExplorationSelector(evidence).select(
        pool,
        goal,
        limit_per_branch=10,
    )
    pareto = next(
        item for item in selection.branches if item.branch == ExplorationBranch.PARETO
    )

    assert {item.candidate_ref.candidate_id for item in pareto.candidates} == {
        "different-low",
        "different-high",
        "same-better",
    }
    assert "same-worse" not in {
        item.candidate_ref.candidate_id for item in pareto.candidates
    }
    assert "raw energies are compared only within one reduced composition" in (
        pareto.candidates[0].rationale
    )


def test_selector_excludes_ood_and_failed_candidates_without_imputation(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    valid = _store_candidate_panel(
        evidence,
        artifacts,
        _candidate("candidate-valid", "SiC"),
        {
            "expert-a": {"stability": 0.7, "target_score": 0.8},
            "expert-b": {"stability": 0.6, "target_score": 0.7},
        },
    )
    ood = _store_candidate_panel(
        evidence,
        artifacts,
        _candidate("candidate-ood", "Si2C"),
        {
            "expert-a": {"stability": 0.9, "target_score": 0.9},
            "expert-b": {"stability": 0.9, "target_score": 0.9},
        },
        out_of_domain_expert="expert-b",
    )
    failed = _store_candidate_panel(
        evidence,
        artifacts,
        _candidate("candidate-failed", "SiC2"),
        {
            "expert-a": {"stability": 0.9, "target_score": 0.9},
            "expert-b": {"stability": 0.9, "target_score": 0.9},
        },
        failed_expert="expert-b",
    )

    selection = DeterministicExplorationSelector(evidence).select(
        CandidatePool(entries=[valid, ood, failed]), _goal()
    )

    assert {
        item.candidate_ref.candidate_id for item in selection.excluded_candidates
    } == {"candidate-ood", "candidate-failed"}
    assert all(
        {item.candidate_ref.candidate_id for item in branch.candidates}
        <= {"candidate-valid"}
        for branch in selection.branches
    )
    reasons = " ".join(
        reason
        for item in selection.excluded_candidates
        for reason in item.reasons
    )
    assert "out-of-domain" in reasons
    assert "failed" in reasons


def test_selector_uses_goal_weights_and_rejects_missing_required_objectives(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    stability_candidate = _store_candidate_panel(
        evidence,
        artifacts,
        _candidate("candidate-stability", "Al2O3"),
        {
            "expert-a": {"stability": 1.0, "target_score": 0.0},
            "expert-b": {"stability": 1.0, "target_score": 0.0},
        },
    )
    target_candidate = _store_candidate_panel(
        evidence,
        artifacts,
        _candidate("candidate-target", "TiO2"),
        {
            "expert-a": {"stability": 0.0, "target_score": 1.0},
            "expert-b": {"stability": 0.0, "target_score": 1.0},
        },
    )
    missing_required = _store_candidate_panel(
        evidence,
        artifacts,
        _candidate("candidate-missing", "ZnO"),
        {
            "expert-a": {"stability": 0.9},
            "expert-b": {"stability": 0.9},
        },
    )
    pool = CandidatePool(
        pool_id="weighted-required-pool",
        entries=[stability_candidate, target_candidate, missing_required],
    )

    def weighted_goal(stability_weight: float, target_weight: float) -> DiscoveryGoal:
        return _goal().model_copy(
            update={
                "objectives": [
                    PropertyObjective(
                        property_name="stability",
                        direction=ObjectiveDirection.MAXIMIZE,
                        unit="arb",
                        weight=stability_weight,
                        required=True,
                    ),
                    PropertyObjective(
                        property_name="target_score",
                        direction=ObjectiveDirection.MAXIMIZE,
                        unit="arb",
                        weight=target_weight,
                        required=True,
                    ),
                ]
            }
        )

    selector = DeterministicExplorationSelector(evidence)
    stability_weighted = selector.select(
        pool, weighted_goal(10.0, 1.0), limit_per_branch=1
    )
    target_weighted = selector.select(
        pool, weighted_goal(1.0, 10.0), limit_per_branch=1
    )
    stability_target_branch = next(
        item
        for item in stability_weighted.branches
        if item.branch == ExplorationBranch.TARGET_PROPERTY
    )
    target_target_branch = next(
        item
        for item in target_weighted.branches
        if item.branch == ExplorationBranch.TARGET_PROPERTY
    )

    assert stability_target_branch.candidates[0].candidate_ref.candidate_id == (
        "candidate-stability"
    )
    assert target_target_branch.candidates[0].candidate_ref.candidate_id == (
        "candidate-target"
    )
    missing = next(
        item
        for item in stability_weighted.excluded_candidates
        if item.candidate_ref.candidate_id == "candidate-missing"
    )
    assert "missing required goal objectives: target_score" in missing.reasons


def test_selector_fails_closed_for_satisfy_and_hard_numeric_constraints(
    tmp_path: Path,
) -> None:
    artifacts = ArtifactStore(tmp_path)
    evidence = ExpertEvidenceStore(artifacts)
    entries = [
        _store_candidate_panel(
            evidence,
            artifacts,
            _candidate("candidate-pass", "LiCoO2"),
            {
                "expert-a": {"stability": 0.8, "target_score": 0.9},
                "expert-b": {"stability": 0.75, "target_score": 0.85},
            },
        ),
        _store_candidate_panel(
            evidence,
            artifacts,
            _candidate("candidate-satisfy-fail", "LiNiO2"),
            {
                "expert-a": {"stability": 0.6, "target_score": 0.9},
                "expert-b": {"stability": 0.8, "target_score": 0.9},
            },
        ),
        _store_candidate_panel(
            evidence,
            artifacts,
            _candidate("candidate-constraint-fail", "LiMnO2"),
            {
                "expert-a": {"stability": 0.8, "target_score": 0.7},
                "expert-b": {"stability": 0.8, "target_score": 0.9},
            },
        ),
        _store_candidate_panel(
            evidence,
            artifacts,
            _candidate("candidate-constraint-missing", "LiFeO2"),
            {
                "expert-a": {"stability": 0.8},
                "expert-b": {"stability": 0.8},
            },
        ),
    ]
    goal = _goal().model_copy(
        update={
            "objectives": [
                PropertyObjective(
                    property_name="stability",
                    direction=ObjectiveDirection.SATISFY,
                    unit="arb",
                    lower_bound=0.7,
                    required=True,
                )
            ],
            "constraints": [
                GoalConstraint(
                    constraint_id="minimum-target-score",
                    description="Every reported target score must meet the floor.",
                    property_name="target_score",
                    operator="gte",
                    value=0.8,
                    hard=True,
                )
            ],
        }
    )

    selection = DeterministicExplorationSelector(evidence).select(
        CandidatePool(pool_id="constraint-pool", entries=entries), goal
    )
    selected_ids = {
        item.candidate_ref.candidate_id
        for branch in selection.branches
        for item in branch.candidates
    }
    assert selected_ids == {"candidate-pass"}
    excluded = {
        item.candidate_ref.candidate_id: " ".join(item.reasons)
        for item in selection.excluded_candidates
    }
    assert "violates lower_bound" in excluded["candidate-satisfy-fail"]
    assert "minimum-target-score" in excluded["candidate-constraint-fail"]
    assert "missing property 'target_score'" in excluded[
        "candidate-constraint-missing"
    ]

    criterionless = goal.model_copy(
        update={
            "objectives": [
                PropertyObjective(
                    property_name="stability",
                    direction=ObjectiveDirection.SATISFY,
                    unit="arb",
                    required=True,
                )
            ],
            "constraints": [],
        }
    )
    criterionless_selection = DeterministicExplorationSelector(evidence).select(
        CandidatePool(pool_id="criterionless-pool", entries=[entries[0]]),
        criterionless,
    )
    assert not any(
        branch.candidates for branch in criterionless_selection.branches
    )
    assert "no explicit numeric criterion" in " ".join(
        reason
        for item in criterionless_selection.excluded_candidates
        for reason in item.reasons
    )


def test_adaptive_scheduler_tracks_reasons_and_keeps_controls_bounded() -> None:
    precision = AdaptiveGenerationScheduler(
        GenerationControls(
            alpha=0.5,
            temperature=1.0,
            mutation_strength=0.2,
            diversity_strength=0.3,
        )
    )
    precision.update(improvement=0.1, structural_collapse_rate=0.1)
    precise = precision.update(improvement=0.2, structural_collapse_rate=0.1)
    assert precise.alpha == pytest.approx(0.4)
    assert "precision" in precise.decision_reason

    exploration = AdaptiveGenerationScheduler()
    exploration.update(improvement=0.0, structural_collapse_rate=0.1)
    exploratory = exploration.update(
        improvement=0.0, structural_collapse_rate=0.1
    )
    assert exploratory.temperature > 1.0
    assert exploratory.mutation_strength > 0.2
    assert "stagnated" in exploratory.decision_reason

    candidate = _candidate("candidate-branch", "BN")
    safer = exploration.update(
        improvement=0.0,
        structural_collapse_rate=0.3,
        high_disagreement_candidates=[candidate.candidate_ref],
    )
    assert safer.temperature < exploratory.temperature
    assert safer.mutation_strength < exploratory.mutation_strength
    assert "collapse increased" in safer.decision_reason
    assert "independent exploration branch" in safer.decision_reason
    assert exploration.history[-1].observation.high_disagreement_candidates == [
        candidate.candidate_ref
    ]

    bounded = AdaptiveGenerationScheduler(
        GenerationControls(
            alpha=0.0,
            temperature=4.99,
            mutation_strength=0.99,
            diversity_strength=0.99,
        )
    )
    for _ in range(100):
        controls = bounded.update(
            improvement=0.0, structural_collapse_rate=0.0
        )
    assert 0.0 <= controls.alpha <= 1.0
    assert 0.01 <= controls.temperature <= 5.0
    assert 0.0 <= controls.mutation_strength <= 1.0
    assert 0.0 <= controls.diversity_strength <= 1.0
    assert controls.schedule_step == 100


def test_classifier_free_guidance_scheduler_uses_correct_explore_exploit_direction() -> None:
    scheduler = AdaptiveGenerationScheduler(
        GenerationControls(
            alpha=0.5,
            temperature=1.0,
            mutation_strength=0.2,
            diversity_strength=0.3,
        ),
        alpha_semantics="classifier_free_guidance",
    )

    scheduler.update(improvement=0.1, structural_collapse_rate=0.1)
    exploit = scheduler.update(improvement=0.2, structural_collapse_rate=0.1)
    assert exploit.alpha == pytest.approx(0.6)
    assert "condition-focused exploitation" in exploit.decision_reason

    scheduler.update(improvement=0.0, structural_collapse_rate=0.1)
    explore = scheduler.update(improvement=0.0, structural_collapse_rate=0.1)
    assert explore.alpha == pytest.approx(0.5)
    assert "broaden sampling" in explore.decision_reason

    collapse = scheduler.update(improvement=0.0, structural_collapse_rate=0.5)
    assert collapse.alpha == pytest.approx(0.4)
    assert "classifier-free guidance" in collapse.decision_reason
