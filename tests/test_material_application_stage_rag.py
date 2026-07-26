from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from discovery_os import material_decision_runner as runner_module
from discovery_os.literature_rag import (
    EvidenceClaim,
    EvidenceGraphBuilder,
    EvidencePolarity,
    EvidenceStage,
    LiteratureQuery,
    LiteratureRecord,
    LiteratureSource,
    QueryIntentCoverage,
    RagEvidenceBundle,
    RagSearchPlan,
    SourceRetrievalStatus,
    SourceRunStatus,
)
from discovery_os.material_decision_runner import MaterialDecisionRunner
from discovery_os.material_domains import MATERIAL_EVIDENCE_STAGES
from discovery_os.material_stage_research import stage_research_policy
from discovery_os.schemas import MaterialField


class _FakeStagePipeline:
    def __init__(
        self,
        *,
        environment: dict[str, str],
        calls: list[dict[str, object]],
        fail_stage: str | None,
        missing_intent_stage: str | None,
        mcp_record_contract: object,
    ) -> None:
        self.environment = dict(environment)
        self.calls = calls
        self.fail_stage = fail_stage
        self.missing_intent_stage = missing_intent_stage
        self.mcp_record_contract = mcp_record_contract
        self.retriever = SimpleNamespace(
            mcp_client=(
                object()
                if self.environment.get("MATERIAL_RAG_MCP_TOOL")
                else None
            )
        )

    def run(self, prompt: str, **kwargs: object) -> RagEvidenceBundle:
        stage_line = next(
            line for line in prompt.splitlines() if line.startswith("Validation stage:")
        )
        stage = stage_line.split(":", 1)[1].strip().rstrip(".")
        sources = list(kwargs["sources"])
        blueprints = list(kwargs["query_blueprints"])
        policy_id = str(kwargs["query_policy_id"])
        policy_version = str(kwargs["query_policy_version"])
        self.calls.append(
            {
                "stage": stage,
                "prompt": prompt,
                "sources": sources,
                "environment": self.environment,
                "max_branches": kwargs["max_branches"],
                "query_blueprints": blueprints,
                "query_policy_id": policy_id,
                "query_policy_version": policy_version,
                "mcp_record_contract": self.mcp_record_contract,
            }
        )
        if stage == self.fail_stage:
            raise RuntimeError("fixture stage failure")
        now = datetime.now(timezone.utc)
        queries = [
            LiteratureQuery(
                query_id=(
                    f"QUERY-{stage}-{source}-{blueprint.intent_id}"
                ),
                source=source,
                query=blueprint.query,
                rationale=blueprint.rationale,
                intent_id=blueprint.intent_id,
                expected_record_types=list(
                    blueprint.expected_record_types
                ),
                mcp_arguments=dict(blueprint.mcp_arguments),
            )
            for source in sources
            for blueprint in blueprints
        ]
        query_by_source_intent = {
            (str(item.source), item.intent_id): item
            for item in queries
        }
        omitted_intent = (
            blueprints[-1].intent_id
            if stage == self.missing_intent_stage
            else None
        )
        records: list[LiteratureRecord] = []
        claims: list[EvidenceClaim] = []
        coverage: list[QueryIntentCoverage] = []
        for blueprint in blueprints:
            intent_queries = [
                item for item in queries if item.intent_id == blueprint.intent_id
            ]
            intent_records: list[LiteratureRecord] = []
            if blueprint.intent_id != omitted_intent:
                first_query = query_by_source_intent[
                    (str(sources[0]), blueprint.intent_id)
                ]
                record = LiteratureRecord(
                    record_id=f"RECORD-{stage}-{blueprint.intent_id}",
                    title=(
                        f"ITO evidence for {stage} "
                        f"{blueprint.intent_id}"
                    ),
                    abstract=(
                        "ITO was evaluated as a transparent electrode in the "
                        f"{stage} {blueprint.intent_id} evidence scope."
                    ),
                    doi=(
                        "10.1000/"
                        f"{stage.replace('_', '-')}-"
                        f"{blueprint.intent_id.replace('_', '-')}"
                    ),
                    source_ids={
                        str(sources[0]): (
                            f"SOURCE-{stage}-{blueprint.intent_id}"
                        )
                    },
                    source_queries=[first_query.query_id],
                    urls=[
                        "https://example.test/"
                        f"{stage}/{blueprint.intent_id}"
                    ],
                    retrieved_at=now,
                )
                records.append(record)
                intent_records.append(record)
                claims.append(
                    EvidenceClaim(
                        claim_id=f"CLAIM-{stage}-{blueprint.intent_id}",
                        source_record_id=record.record_id,
                        subject="ITO",
                        predicate="was evaluated as",
                        object="a transparent electrode",
                        polarity=EvidencePolarity.SUPPORTS,
                        stage=EvidenceStage.MATERIAL_CHARACTERIZATION,
                        support_text=record.abstract,
                        confidence=0.9,
                    )
                )
            coverage.append(
                QueryIntentCoverage(
                    intent_id=blueprint.intent_id,
                    query_ids=[item.query_id for item in intent_queries],
                    record_ids=[
                        item.record_id for item in intent_records
                    ],
                    sources_with_records=(
                        [sources[0]] if intent_records else []
                    ),
                    status=(
                        "covered" if intent_records else "no_records"
                    ),
                )
            )
        return RagEvidenceBundle(
            bundle_id=f"BUNDLE-{stage}",
            created_at=now,
            search_plan=RagSearchPlan(
                plan_id=f"PLAN-{stage}",
                user_prompt=prompt,
                generated_at=now,
                planner_id="fixture-stage-planner",
                planner_version="1",
                query_policy_id=policy_id,
                query_policy_version=policy_version,
                required_intent_ids=[
                    item.intent_id for item in blueprints
                ],
                queries=queries,
            ),
            source_statuses=[
                SourceRetrievalStatus(
                    source=source,
                    status=SourceRunStatus.SUCCESS,
                    query_ids=[
                        item.query_id
                        for item in queries
                        if item.source == source
                    ],
                    result_count=(len(records) if source == sources[0] else 0),
                )
                for source in sources
            ],
            records=records,
            claims=claims,
            graph=EvidenceGraphBuilder().build(claims),
            branches=[],
            intent_coverage=coverage,
        )


def _install_fake_pipeline(
    monkeypatch,
    *,
    fail_stage: str | None = None,
    missing_intent_stage: str | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def factory(*, environ, require_model, mcp_record_contract):
        assert require_model is False
        return _FakeStagePipeline(
            environment=dict(environ),
            calls=calls,
            fail_stage=fail_stage,
            missing_intent_stage=missing_intent_stage,
            mcp_record_contract=mcp_record_contract,
        )

    monkeypatch.setattr(
        runner_module,
        "build_literature_rag_from_environment",
        factory,
    )
    return calls


def _run_stage_rag(
    tmp_path: Path,
    monkeypatch,
    *,
    fail_stage: str | None = None,
    missing_intent_stage: str | None = None,
):
    calls = _install_fake_pipeline(
        monkeypatch,
        fail_stage=fail_stage,
        missing_intent_stage=missing_intent_stage,
    )
    runner = MaterialDecisionRunner(
        artifact_root=tmp_path,
        environ={
            "MATERIAL_RAG_MCP_URL": "http://127.0.0.1:9999/mcp",
            "MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP": "1",
            "MATERIAL_RAG_MCP_TOOL": "generic_material_evidence",
            "MATERIAL_APPLICATION_RAG_MCP_TOOL": "application_material_evidence",
            "MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY": "crystal_identity_evidence",
        },
    )
    run = runner.run(
        "Compare ITO for a transparent electrode.",
        material_field=MaterialField.SEMICONDUCTOR,
        main_model_routing="off",
        explicit_role_ids=["transparent_electrode"],
        run_rag=True,
    )
    return run, calls


def test_application_rag_runs_five_isolated_stages_with_admin_mcp_precedence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run, calls = _run_stage_rag(tmp_path, monkeypatch)

    assert [item["stage"] for item in calls] == list(MATERIAL_EVIDENCE_STAGES)
    assert len(run.rag_stage_receipts) == 5
    assert all(
        item.one_validation_stage_per_request
        and item.property_scoring_performed is False
        for item in run.rag_stage_receipts
    )
    assert all(
        item.status == "completed" and not item.missing_intent_ids
        for item in run.rag_stage_receipts
    )
    for call in calls:
        stage = str(call["stage"])
        policy = stage_research_policy(stage)
        prompt = str(call["prompt"])
        assert prompt.count("Validation stage:") == 1
        assert f"Validation stage: {stage}." in prompt
        assert (
            f"Code-owned research policy: {policy.policy_id}@"
            f"{policy.policy_version}."
        ) in prompt
        assert call["max_branches"] == 1
        assert call["query_policy_id"] == policy.policy_id
        assert call["query_policy_version"] == policy.policy_version
        assert [
            item.intent_id for item in call["query_blueprints"]
        ] == [item.intent_id for item in policy.query_intents]
        assert call["mcp_record_contract"] == policy.mcp.runtime_contract(
            stage
        )
        tool = call["environment"]["MATERIAL_RAG_MCP_TOOL"]
        if stage == "identity_novelty":
            assert tool == "crystal_identity_evidence"
        else:
            assert tool == "application_material_evidence"
        source_values = {str(item) for item in call["sources"]}
        assert "mcp" in source_values
        if stage in {"generation_prior", "identity_novelty"}:
            assert source_values == {"crossref", "arxiv", "openalex", "mcp"}
        else:
            assert source_values == {"crossref", "arxiv", "mcp"}
        receipt = next(
            item
            for item in run.rag_stage_receipts
            if item.evidence_stage == stage
        )
        assert receipt.research_policy_id == policy.policy_id
        assert receipt.research_policy_version == policy.policy_version
        assert receipt.required_intent_ids == [
            item.intent_id for item in policy.query_intents
        ]
        assert receipt.mcp_scope_argument == policy.mcp.scope_argument
        assert (
            receipt.mcp_required_stage_metadata_fields
            == policy.mcp.required_stage_metadata_fields
        )

    assert run.rag_bundle_id == run.report.rag_bundle_id
    assert run.rag_bundle_id is not None
    artifact_by_kind = {item.kind: item for item in run.artifacts}
    assert "application_rag_bundle" in artifact_by_kind
    for stage in MATERIAL_EVIDENCE_STAGES:
        kind = f"application_rag_{stage}"
        assert kind in artifact_by_kind
        assert (tmp_path / artifact_by_kind[kind].relative_path).is_file()
    composite = json.loads(
        (
            tmp_path
            / artifact_by_kind["application_rag_bundle"].relative_path
        ).read_text(encoding="utf-8")
    )
    assert composite["branches"] == []
    expected_records = sum(
        len(stage_research_policy(stage).query_intents)
        for stage in MATERIAL_EVIDENCE_STAGES
    )
    assert len(composite["records"]) == expected_records
    assert len(composite["claims"]) == expected_records
    assert (
        len({item["record_id"] for item in composite["records"]})
        == expected_records
    )
    assert all(
        item["intent_id"] is None
        and item["expected_record_types"] == []
        and item["mcp_arguments"] == {}
        for item in composite["search_plan"]["queries"]
    )
    assert all(
        item["raw_metadata"]["application_research_policy_id"]
        and item["raw_metadata"]["application_query_intent_ids"]
        for item in composite["records"]
    )
    assert "separate single-stage requests" in composite["warnings"][0]


def test_failed_application_rag_stage_is_explicit_and_not_used(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run, calls = _run_stage_rag(
        tmp_path,
        monkeypatch,
        fail_stage="mlip_disagreement",
    )

    assert len(calls) == 5
    receipt = next(
        item
        for item in run.rag_stage_receipts
        if item.evidence_stage == "mlip_disagreement"
    )
    assert receipt.status == "failed"
    assert receipt.source_bundle_id is None
    assert receipt.missing_intent_ids == [
        item.intent_id
        for item in stage_research_policy("mlip_disagreement").query_intents
    ]
    assert "RuntimeError" in receipt.error
    assert "fixture stage failure" not in receipt.error
    artifact_kinds = {item.kind for item in run.artifacts}
    assert "application_rag_mlip_disagreement" not in artifact_kinds
    assert "application_rag_bundle" in artifact_kinds


def test_application_rag_receipt_exposes_missing_query_intent_coverage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run, _calls = _run_stage_rag(
        tmp_path,
        monkeypatch,
        missing_intent_stage="relaxation_validation",
    )

    policy = stage_research_policy("relaxation_validation")
    receipt = next(
        item
        for item in run.rag_stage_receipts
        if item.evidence_stage == "relaxation_validation"
    )
    assert receipt.status == "completed_with_missing_intents"
    assert receipt.missing_intent_ids == [
        policy.query_intents[-1].intent_id
    ]
    assert receipt.source_bundle_id == "BUNDLE-relaxation_validation"


def test_explicit_application_sources_are_intersected_per_stage_and_do_not_enable_mcp(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch)
    runner = MaterialDecisionRunner(
        artifact_root=tmp_path,
        environ={
            "MATERIAL_RAG_MCP_URL": "http://127.0.0.1:9999/mcp",
            "MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP": "1",
            "MATERIAL_APPLICATION_RAG_MCP_TOOL": "application_material_evidence",
        },
    )

    run = runner.run(
        "Compare ITO for a transparent electrode.",
        material_field=MaterialField.SEMICONDUCTOR,
        main_model_routing="off",
        explicit_role_ids=["transparent_electrode"],
        run_rag=True,
        rag_sources=[
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.OPENALEX,
        ],
    )

    assert len(calls) == 5
    for call in calls:
        stage = str(call["stage"])
        source_values = {str(item) for item in call["sources"]}
        assert "mcp" not in source_values
        assert "MATERIAL_RAG_MCP_TOOL" not in call["environment"]
        if stage in {"generation_prior", "identity_novelty"}:
            assert source_values == {"crossref", "arxiv", "openalex"}
        else:
            assert source_values == {"crossref", "arxiv"}
    assert all(item.selected_mcp_tool is None for item in run.rag_stage_receipts)


def test_source_allowed_only_in_early_stages_fails_later_stages_without_aborting(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = _install_fake_pipeline(monkeypatch)
    run = MaterialDecisionRunner(artifact_root=tmp_path).run(
        "Compare ITO for a transparent electrode.",
        material_field=MaterialField.SEMICONDUCTOR,
        main_model_routing="off",
        explicit_role_ids=["transparent_electrode"],
        run_rag=True,
        rag_sources=[LiteratureSource.OPENALEX],
    )

    assert [item["stage"] for item in calls] == [
        "generation_prior",
        "identity_novelty",
    ]
    failed = [
        item for item in run.rag_stage_receipts if item.status == "failed"
    ]
    assert [item.evidence_stage for item in failed] == [
        "mlip_disagreement",
        "relaxation_validation",
        "dft_handoff",
    ]
    assert all(
        "No caller-requested evidence provider" in (item.error or "")
        for item in failed
    )
