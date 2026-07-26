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
    RagEvidenceBundle,
    RagSearchPlan,
    SourceRetrievalStatus,
    SourceRunStatus,
)
from discovery_os.material_decision_runner import MaterialDecisionRunner
from discovery_os.material_domains import MATERIAL_EVIDENCE_STAGES
from discovery_os.schemas import MaterialField


class _FakeStagePipeline:
    def __init__(
        self,
        *,
        environment: dict[str, str],
        calls: list[dict[str, object]],
        fail_stage: str | None,
    ) -> None:
        self.environment = dict(environment)
        self.calls = calls
        self.fail_stage = fail_stage
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
        self.calls.append(
            {
                "stage": stage,
                "prompt": prompt,
                "sources": sources,
                "environment": self.environment,
                "max_branches": kwargs["max_branches"],
            }
        )
        if stage == self.fail_stage:
            raise RuntimeError("fixture stage failure")
        now = datetime.now(timezone.utc)
        query_id = f"QUERY-{stage}"
        query = LiteratureQuery(
            query_id=query_id,
            source=sources[0],
            query=f"ITO transparent electrode {stage}",
            rationale="stage isolation fixture",
        )
        record = LiteratureRecord(
            record_id=f"RECORD-{stage}",
            title=f"ITO evidence for {stage}",
            abstract=(
                f"ITO was evaluated as a transparent electrode in the {stage} "
                "evidence scope."
            ),
            doi=f"10.1000/{stage.replace('_', '-')}",
            source_ids={str(sources[0]): f"SOURCE-{stage}"},
            source_queries=[query_id],
            urls=[f"https://example.test/{stage}"],
            retrieved_at=now,
        )
        claim = EvidenceClaim(
            claim_id=f"CLAIM-{stage}",
            source_record_id=record.record_id,
            subject="ITO",
            predicate="was evaluated as",
            object="a transparent electrode",
            polarity=EvidencePolarity.SUPPORTS,
            stage=EvidenceStage.MATERIAL_CHARACTERIZATION,
            support_text=record.abstract,
            confidence=0.9,
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
                queries=[query],
            ),
            source_statuses=[
                SourceRetrievalStatus(
                    source=source,
                    status=SourceRunStatus.SUCCESS,
                    query_ids=[query_id],
                    result_count=1,
                )
                for source in sources
            ],
            records=[record],
            claims=[claim],
            graph=EvidenceGraphBuilder().build([claim]),
            branches=[],
        )


def _install_fake_pipeline(
    monkeypatch,
    *,
    fail_stage: str | None = None,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def factory(*, environ, require_model):
        assert require_model is False
        return _FakeStagePipeline(
            environment=dict(environ),
            calls=calls,
            fail_stage=fail_stage,
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
):
    calls = _install_fake_pipeline(monkeypatch, fail_stage=fail_stage)
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
    for call in calls:
        stage = str(call["stage"])
        prompt = str(call["prompt"])
        assert prompt.count("Validation stage:") == 1
        assert f"Validation stage: {stage}." in prompt
        assert call["max_branches"] == 1
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
    assert len(composite["records"]) == 5
    assert len(composite["claims"]) == 5
    assert len({item["record_id"] for item in composite["records"]}) == 5
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
    assert "RuntimeError" in receipt.error
    assert "fixture stage failure" not in receipt.error
    artifact_kinds = {item.kind for item in run.artifacts}
    assert "application_rag_mlip_disagreement" not in artifact_kinds
    assert "application_rag_bundle" in artifact_kinds


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
