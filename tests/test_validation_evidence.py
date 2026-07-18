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
from discovery_os.mcp_client import McpClientError
from discovery_os.validation_evidence import (
    McpContractVerificationStatus,
    ValidationHandoffKind,
    ValidationEvidenceRequest,
    ValidationEvidenceRouter,
    ValidationEvidenceStage,
    ValidationEvidenceStatus,
    build_validation_evidence_prompt,
    build_validation_evidence_router_from_environment,
    fusion_decision_context_from_stage_evidence,
    fusion_decision_contexts_from_stage_evidence,
    validation_evidence_route,
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
        self.retriever = _Retriever()

    def run(self, prompt: str, **kwargs):
        self.calls.append({"prompt": prompt, **kwargs})
        if self.error is not None:
            raise self.error
        return self.bundle


class _ContractClient:
    def require_tool_contract(self, name, *, accepted_arguments, result_collection):
        assert name == "search_materials"
        assert accepted_arguments == ("query", "max_results", "from_date", "to_date")
        assert result_collection == "records"
        return {"tool_name": name}


class _Retriever:
    mcp_client = _ContractClient()
    mcp_tool = "search_materials"


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
    assert run.report.mcp_contract_status == McpContractVerificationStatus.VERIFIED
    assert run.report.mcp_tool_name == "search_materials"
    assert run.report.handoff.kind == ValidationHandoffKind.GENERATION_CONSTRAINT_CONTEXT
    assert run.report.handoff.evidence_available is True
    assert run.report.handoff.can_steer_generation is True
    assert run.report.handoff.evidence_claim_ids == ["CLAIM-li-o"]
    assert run.report.handoff.evidence_branch_ids == ["EBRANCH-li-o"]
    assert run.report.handoff.validator_execution_state == "not_executed"
    assert run.report.handoff.unknown_not_pass is True
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
    assert run.report.handoff.evidence_available is False
    assert run.report.handoff.can_steer_generation is False
    assert run.report.handoff.unknown_not_pass is True
    assert run.report.handoff.validator_execution_state == "not_executed"


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


def test_all_stage_routes_are_closed_typed_and_fail_closed() -> None:
    expected = {
        ValidationEvidenceStage.GENERATION_PRIOR: (
            ValidationHandoffKind.GENERATION_CONSTRAINT_CONTEXT,
            "FusionDecisionContext",
            True,
        ),
        ValidationEvidenceStage.IDENTITY_NOVELTY: (
            ValidationHandoffKind.IDENTITY_NOVELTY_CONTEXT,
            "ScientificNoveltyAssessment",
            False,
        ),
        ValidationEvidenceStage.MLIP_DISAGREEMENT: (
            ValidationHandoffKind.MLIP_DISAGREEMENT_CONTEXT,
            "ModelDisagreement",
            False,
        ),
        ValidationEvidenceStage.RELAXATION_VALIDATION: (
            ValidationHandoffKind.RELAXATION_GATE_CONTEXT,
            "PeriodicRelaxationPayload",
            False,
        ),
        ValidationEvidenceStage.DFT_HANDOFF: (
            ValidationHandoffKind.DFT_PREPARATION_CONTEXT,
            "DFTInputHandoffReport",
            False,
        ),
    }
    expected_sources = {
        ValidationEvidenceStage.GENERATION_PRIOR: [
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.OPENALEX,
            LiteratureSource.MCP,
        ],
        ValidationEvidenceStage.IDENTITY_NOVELTY: [
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.OPENALEX,
            LiteratureSource.MCP,
        ],
        ValidationEvidenceStage.MLIP_DISAGREEMENT: [
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.MCP,
        ],
        ValidationEvidenceStage.RELAXATION_VALIDATION: [
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.MCP,
        ],
        ValidationEvidenceStage.DFT_HANDOFF: [
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.MCP,
        ],
    }
    for stage, (kind, payload_schema, can_steer) in expected.items():
        route = validation_evidence_route(stage)
        assert route.stage == stage
        assert route.literature_sources == expected_sources[stage]
        assert route.official_validators == [
            item.authority_id for item in route.validator_authorities
        ]
        assert all(
            item.availability
            in {"implemented", "sidecar_required", "credential_required", "external_required"}
            for item in route.validator_authorities
        )
        assert all(item.failure_policy == "unknown-not-pass" for item in route.validator_authorities)
        assert route.failure_policy.record_absence == "not-proof-of-novelty-or-validity"
        assert route.mcp_contract.tool_environment_variable == (
            f"MATERIAL_RAG_MCP_TOOL_{stage.value.upper()}"
        )
        assert route.mcp_contract.selection_policy == "administrator-configured-allowlist-only"
        assert route.mcp_contract.required_record_fields == ["source_id", "title"]
        assert route.handoff_contract.kind == kind
        assert route.handoff_contract.payload_schema == payload_schema
        assert route.handoff_contract.can_steer_generation is can_steer
        assert route.handoff_contract.evidence_can_replace_validator is False


def test_non_allowlisted_bundle_source_fails_closed(tmp_path) -> None:
    bundle = _bundle()
    bad_query = bundle.search_plan.queries[0].model_copy(
        update={"source": LiteratureSource.PUBMED}
    )
    bad_plan = bundle.search_plan.model_copy(update={"queries": [bad_query]})
    bundle = bundle.model_copy(update={"search_plan": bad_plan})

    run = ValidationEvidenceRouter(
        _Pipeline(bundle=bundle), artifact_root=tmp_path
    ).run(
        ValidationEvidenceRequest(
            stage=ValidationEvidenceStage.MLIP_DISAGREEMENT,
            chemical_system="Li-O",
        )
    )

    assert run.report.status == ValidationEvidenceStatus.UNKNOWN
    assert run.report.reason == (
        "stage_route_contract_violation:search_plan_used_nonselected_source"
    )
    assert run.bundle is None
    assert run.report.handoff.evidence_available is False


def test_failed_mcp_tool_contract_omits_mcp_and_keeps_other_sources_partial(tmp_path) -> None:
    class BadClient:
        def require_tool_contract(self, *args, **kwargs):
            raise McpClientError("schema mismatch")

    pipeline = _Pipeline()
    pipeline.retriever.mcp_client = BadClient()
    run = ValidationEvidenceRouter(pipeline, artifact_root=tmp_path).run(
        ValidationEvidenceRequest(
            stage=ValidationEvidenceStage.GENERATION_PRIOR,
            chemical_system="Li-O",
        )
    )

    assert LiteratureSource.MCP not in pipeline.calls[0]["sources"]
    assert run.report.status == ValidationEvidenceStatus.PARTIAL
    assert run.report.mcp_contract_status == McpContractVerificationStatus.FAILED
    assert run.report.mcp_tool_name is None
    assert run.report.handoff.evidence_available is True
    assert any("tool contract failed" in item for item in run.report.warnings)


def test_stage_specific_mcp_tool_environment_builds_isolated_pipeline(tmp_path) -> None:
    router = build_validation_evidence_router_from_environment(
        artifact_root=tmp_path,
        environ={
            "VALIDATION_EVIDENCE_ENABLED": "1",
            "MATERIAL_RAG_MCP_URL": "https://mcp.example/evidence",
            "MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT": "search_mlip_limits",
        },
    )

    mlip_pipeline = router.pipelines_by_stage[ValidationEvidenceStage.MLIP_DISAGREEMENT]
    assert mlip_pipeline is not None
    assert mlip_pipeline.retriever.mcp_tool == "search_mlip_limits"
    identity_pipeline = router.pipelines_by_stage[ValidationEvidenceStage.IDENTITY_NOVELTY]
    assert identity_pipeline is not None
    assert identity_pipeline.retriever.mcp_client is None
    assert identity_pipeline.retriever.mcp_tool is None
