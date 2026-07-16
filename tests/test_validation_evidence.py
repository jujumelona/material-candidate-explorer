from __future__ import annotations

from datetime import datetime, timezone

import pytest

from discovery_os.literature_rag import (
    EvidenceBranch,
    EvidenceBranchKind,
    EvidenceClaim,
    EvidenceGraph,
    EvidencePolarity,
    EvidenceStage,
    LiteratureQuery,
    LiteratureRecord,
    LiteratureSource,
    RagEvidenceBundle,
    RagSearchPlan,
    SourceRetrievalStatus,
    SourceRunStatus,
)
from discovery_os.validation_evidence import (
    ValidationEvidenceRequest,
    ValidationEvidenceRouter,
    ValidationEvidenceStage,
    ValidationEvidenceStatus,
    build_validation_evidence_prompt,
    fusion_decision_context_from_stage_evidence,
    fusion_decision_contexts_from_stage_evidence,
)


def _bundle() -> RagEvidenceBundle:
    now = datetime.now(timezone.utc)
    record = LiteratureRecord(
        record_id="LIT-li-o",
        title="Reported Li-O phase stability and synthesis conditions",
        abstract="Li-O phases were characterized under bounded synthesis conditions.",
        source_ids={"crossref": "10.1000/li-o"},
        source_queries=["query-li-o"],
        retrieved_at=now,
    )
    claim = EvidenceClaim(
        claim_id="CLAIM-li-o",
        source_record_id=record.record_id,
        subject="Li-O",
        predicate="has reported phases under",
        object="bounded synthesis conditions",
        polarity=EvidencePolarity.SUPPORTS,
        stage=EvidenceStage.MATERIAL_CHARACTERIZATION,
        support_text=record.abstract,
        confidence=0.8,
    )
    plan = RagSearchPlan(
        plan_id="RPLAN-li-o",
        user_prompt="Li-O phase evidence",
        generated_at=now,
        planner_id="fixture",
        planner_version="1",
        queries=[
            LiteratureQuery(
                query_id="query-li-o",
                source=LiteratureSource.CROSSREF,
                query="Li-O phase stability",
                rationale="fixture",
            )
        ],
    )
    return RagEvidenceBundle(
        bundle_id="RBUNDLE-li-o",
        created_at=now,
        search_plan=plan,
        source_statuses=[
            SourceRetrievalStatus(
                source=LiteratureSource.CROSSREF,
                status=SourceRunStatus.SUCCESS,
                query_ids=["query-li-o"],
                result_count=1,
            )
        ],
        records=[record],
        claims=[claim],
        graph=EvidenceGraph(graph_id="EGRAPH-li-o", nodes=[], edges=[]),
        branches=[
            EvidenceBranch(
                branch_id="EBRANCH-li-o",
                kind=EvidenceBranchKind.MATERIAL_COMPOSITION,
                title="Preserve the evidence-linked chemical system",
                rationale="One source-grounded material composition branch.",
                source_claim_ids=[claim.claim_id],
                generator_hints={"chemical_system": "Li-O"},
                priority=0.8,
            )
        ],
    )


class _Pipeline:
    def __init__(
        self,
        *,
        error: Exception | None = None,
        bundle: RagEvidenceBundle | None = None,
    ) -> None:
        self.error = error
        self.bundle = bundle or _bundle()
        self.calls: list[dict] = []

    def run(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.error is not None:
            raise self.error
        return self.bundle


def test_generation_stage_routes_official_sources_and_binds_fusion_context(tmp_path) -> None:
    pipeline = _Pipeline()
    router = ValidationEvidenceRouter(pipeline, artifact_root=tmp_path)
    request = ValidationEvidenceRequest(
        stage=ValidationEvidenceStage.GENERATION_PRIOR,
        chemical_system="Li-O",
        observations={"target_hull_ranges_eV_atom": [0.0, 0.03, 0.06]},
    )

    run = router.run(request)

    assert run.report.status == ValidationEvidenceStatus.COMPLETED
    assert run.report.property_score_created is False
    assert run.report.bundle_relative_path is not None
    assert (tmp_path / run.report.bundle_relative_path).is_file()
    assert run.report_path.is_file()
    assert set(pipeline.calls[0]["sources"]) == {
        LiteratureSource.CROSSREF,
        LiteratureSource.ARXIV,
        LiteratureSource.OPENALEX,
        LiteratureSource.MCP,
    }
    context = fusion_decision_context_from_stage_evidence(
        run,
        guidance_alpha=0.45,
        exploration_branch="pareto",
    )
    assert context.evidence_branch_id == "EBRANCH-li-o"
    assert context.evidence_claim_ids == ["CLAIM-li-o"]
    assert context.evidence_generator_hints == {"chemical_system": "Li-O"}


def test_stage_retrieval_failure_remains_unknown(tmp_path) -> None:
    router = ValidationEvidenceRouter(
        _Pipeline(error=RuntimeError("provider unavailable")),
        artifact_root=tmp_path,
    )

    run = router.run(
        ValidationEvidenceRequest(
            stage=ValidationEvidenceStage.MLIP_DISAGREEMENT,
            chemical_system="Li-O",
            observations={"high_disagreement_candidates": 2},
        )
    )

    assert run.report.status == ValidationEvidenceStatus.UNKNOWN
    assert run.report.bundle_id is None
    assert run.report.record_count == 0
    assert run.report.reason == "stage_evidence_retrieval_failed:RuntimeError"
    assert run.report.property_score_created is False


def test_stage_prompt_is_bounded_and_observations_reject_secrets() -> None:
    request = ValidationEvidenceRequest(
        stage=ValidationEvidenceStage.DFT_HANDOFF,
        chemical_system="Li-O",
        composition_keys=["Li2O"],
        observations={"shortlist_size": 3, "dft_executed": False},
    )
    prompt = build_validation_evidence_prompt(request)
    assert "reference phases" in prompt
    assert "dft_executed" in prompt
    assert "Do not invent material properties" in prompt

    with pytest.raises(ValueError, match="cannot contain secrets"):
        ValidationEvidenceRequest(
            stage=ValidationEvidenceStage.IDENTITY_NOVELTY,
            chemical_system="Li-O",
            observations={"api_key": "must-not-be-serialized"},
        )


def test_non_generation_evidence_cannot_steer_generator(tmp_path) -> None:
    run = ValidationEvidenceRouter(_Pipeline(), artifact_root=tmp_path).run(
        ValidationEvidenceRequest(
            stage=ValidationEvidenceStage.IDENTITY_NOVELTY,
            chemical_system="Li-O",
        )
    )
    with pytest.raises(ValueError, match="only generation-prior"):
        fusion_decision_context_from_stage_evidence(
            run,
            guidance_alpha=0.5,
            exploration_branch="novelty",
        )


def test_profile_context_allocation_shares_one_evidence_policy(tmp_path) -> None:
    bundle = _bundle()
    second_branch = EvidenceBranch(
        branch_id="EBRANCH-li-o-space-group",
        kind=EvidenceBranchKind.MATERIAL_CONDITION,
        title="Explore an evidence-linked symmetry branch",
        rationale="A second source-grounded branch for independent workers.",
        source_claim_ids=["CLAIM-li-o"],
        generator_hints={"space_group": 225},
        priority=0.7,
    )
    bundle = bundle.model_copy(
        update={"branches": [*bundle.branches, second_branch]}
    )
    run = ValidationEvidenceRouter(
        _Pipeline(bundle=bundle), artifact_root=tmp_path
    ).run(
        ValidationEvidenceRequest(
            stage=ValidationEvidenceStage.GENERATION_PRIOR,
            chemical_system="Li-O",
        )
    )
    contexts = fusion_decision_contexts_from_stage_evidence(
        run,
        controls=[
            (0.25, "stability"),
            (0.45, "pareto"),
            (0.65, "target_property"),
            (0.85, "novelty"),
        ],
    )
    assert [item.guidance_alpha for item in contexts] == [0.25, 0.45, 0.65, 0.85]
    assert [item.exploration_branch for item in contexts] == [
        "stability",
        "pareto",
        "target_property",
        "novelty",
    ]
    assert [item.evidence_branch_id for item in contexts] == [
        "EBRANCH-li-o",
        "EBRANCH-li-o-space-group",
        "EBRANCH-li-o",
        "EBRANCH-li-o-space-group",
    ]
