"""End-to-end coordinator for natural-language material application questions.

The runner performs bounded intent routing, optional source-grounded literature
retrieval, candidate-seed assembly, and evidence-closed role-scoped ranking.
It does not silently invoke generators, specialist property models, DFT, or
experiments.  Those remain explicit downstream validators; supplied
observations can enter ranking only through the strict recommendation schema.
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from .hashing import stable_hash
from .literature_rag import (
    EvidenceClaim,
    EvidenceGraphBuilder,
    JsonEvidenceIndex,
    LiteratureQuery,
    LiteratureRecord,
    LiteratureSource,
    RagEvidenceBundle,
    RagSearchPlan,
    SourceRetrievalStatus,
    SourceRunStatus,
    build_literature_rag_from_environment,
    save_evidence_bundle,
)
from .material_applications import (
    ApplicationEvidenceTask,
    MainModelMaterialApplicationClassifier,
    MaterialApplicationBrief,
    MaterialApplicationModelRun,
    build_main_model_material_application_classifier_from_environment,
    build_material_application_brief,
)
from .material_domains import (
    MATERIAL_EVIDENCE_STAGES,
    MaterialEvidenceStage,
    MaterialFieldModelRun,
    build_main_model_material_field_classifier_from_environment,
    build_material_domain_plan,
)
from .material_recommendation import (
    MaterialApplicationCandidate,
    MaterialApplicationObservation,
    MaterialDecisionPreference,
    MaterialRecommendationReport,
    candidates_from_application_seeds,
    rank_material_application_candidates,
)
from .schemas import Identifier, JsonValue, MaterialField, StrictSchema


MainModelRoutingMode = str

_STAGE_MCP_TOOL_ENV: dict[MaterialEvidenceStage, str] = {
    "generation_prior": "MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR",
    "identity_novelty": "MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY",
    "mlip_disagreement": "MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT",
    "relaxation_validation": "MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION",
    "dft_handoff": "MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF",
}


class MaterialDecisionArtifact(StrictSchema):
    artifact_id: Identifier
    kind: Identifier
    relative_path: str = Field(min_length=1, max_length=2_000)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class ApplicationRagStageReceipt(StrictSchema):
    evidence_stage: MaterialEvidenceStage
    task_ids: list[Identifier] = Field(min_length=1)
    role_ids: list[Identifier] = Field(min_length=1)
    allowed_literature_sources: list[
        Literal["crossref", "arxiv", "openalex"]
    ] = Field(default_factory=list)
    required_mcp_capabilities: list[Identifier] = Field(default_factory=list)
    selected_mcp_tool: Identifier | None = None
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal[
        "completed",
        "completed_with_source_failures",
        "failed",
    ]
    source_bundle_id: Identifier | None = None
    error: str | None = Field(default=None, max_length=4_000)
    one_validation_stage_per_request: Literal[True] = True
    property_scoring_performed: Literal[False] = False

    @model_validator(mode="after")
    def _receipt_is_consistent(self) -> "ApplicationRagStageReceipt":
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("application RAG task identifiers must be unique")
        if len(self.role_ids) != len(set(self.role_ids)):
            raise ValueError("application RAG role identifiers must be unique")
        if self.status == "failed":
            if self.source_bundle_id is not None or not self.error:
                raise ValueError("failed application RAG stage needs only an error")
        elif self.source_bundle_id is None or self.error is not None:
            raise ValueError("completed application RAG stage needs a source bundle")
        return self


class MaterialDecisionRun(StrictSchema):
    run_id: Identifier
    brief: MaterialApplicationBrief
    report: MaterialRecommendationReport
    rag_bundle_id: Identifier | None = None
    rag_stage_receipts: list[ApplicationRagStageReceipt] = Field(
        default_factory=list
    )
    artifacts: list[MaterialDecisionArtifact] = Field(default_factory=list)
    generation_or_specialist_execution_performed: Literal[False] = False
    scientific_status: Literal[
        "application-decision-support-with-explicit-downstream-validation"
    ] = (
        "application-decision-support-with-explicit-downstream-validation"
    )

    @model_validator(mode="after")
    def _run_is_closed(self) -> "MaterialDecisionRun":
        if self.report.brief.brief_id != self.brief.brief_id:
            raise ValueError("material decision report cites another brief")
        if self.report.brief.model_dump(mode="json") != self.brief.model_dump(
            mode="json"
        ):
            raise ValueError("material decision run and report briefs differ")
        if self.report.rag_bundle_id != self.rag_bundle_id:
            raise ValueError("material decision RAG bundle identifiers differ")
        stages = [item.evidence_stage for item in self.rag_stage_receipts]
        if len(stages) != len(set(stages)):
            raise ValueError("application RAG stage receipts must be unique")
        return self


@dataclass(frozen=True)
class _ModelRuns:
    field: MaterialFieldModelRun | None
    application: MaterialApplicationModelRun | None


@dataclass(frozen=True)
class _ApplicationRagRuns:
    composite: RagEvidenceBundle | None
    receipts: tuple[ApplicationRagStageReceipt, ...]
    stage_bundles: tuple[tuple[MaterialEvidenceStage, RagEvidenceBundle], ...]


class MaterialDecisionRunner:
    """Run the safe application-selection front end and persist its outputs."""

    def __init__(
        self,
        *,
        artifact_root: Path | str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.artifact_root = (
            Path(artifact_root).resolve() if artifact_root is not None else None
        )
        self.environ = dict(environ) if environ is not None else None

    def run(
        self,
        question: str,
        *,
        material_field: MaterialField | str = "AUTO",
        chemical_system: str | None = None,
        problem_context: Mapping[str, JsonValue] | None = None,
        main_model_routing: MainModelRoutingMode = "auto",
        explicit_role_ids: Sequence[str] | None = None,
        require_condition_complete: bool = False,
        include_retrieval_seeds: bool = True,
        candidates: Iterable[MaterialApplicationCandidate] = (),
        observations: Iterable[MaterialApplicationObservation] = (),
        preferences: Iterable[MaterialDecisionPreference] = (),
        rag_bundle: RagEvidenceBundle | None = None,
        run_rag: bool = False,
        rag_sources: Sequence[LiteratureSource] | None = None,
        rag_from_date: date | None = None,
        rag_to_date: date | None = None,
        rag_max_results_per_query: int = 12,
    ) -> MaterialDecisionRun:
        routing_mode = str(main_model_routing).strip().casefold()
        if routing_mode not in {"auto", "required", "off"}:
            raise ValueError("main_model_routing must be auto, required, or off")
        if rag_bundle is not None and run_rag:
            raise ValueError("provide a RAG bundle or run RAG, not both")
        context = dict(problem_context or {})
        model_runs = self._classify(
            question,
            material_field=material_field,
            chemical_system=chemical_system,
            problem_context=context,
            routing_mode=routing_mode,
        )
        brief = build_material_application_brief(
            question,
            material_field=material_field,
            chemical_system=chemical_system,
            problem_context=context,
            field_model_run=model_runs.field,
            application_model_run=model_runs.application,
            explicit_role_ids=explicit_role_ids,
            require_condition_complete=require_condition_complete,
        )
        bundle = rag_bundle
        rag_runs = _ApplicationRagRuns(
            composite=bundle,
            receipts=(),
            stage_bundles=(),
        )
        if run_rag:
            rag_runs = self._run_application_rag(
                brief,
                sources=rag_sources,
                from_date=rag_from_date,
                to_date=rag_to_date,
                max_results_per_query=rag_max_results_per_query,
            )
            bundle = rag_runs.composite
        candidate_rows = list(candidates)
        if include_retrieval_seeds:
            candidate_rows = [
                *candidates_from_application_seeds(brief),
                *candidate_rows,
            ]
        if not candidate_rows:
            raise ValueError(
                "no candidates are available; enable retrieval seeds or supply candidates"
            )
        if bundle is not None:
            candidate_rows = _link_candidates_to_exact_rag_claims(
                candidate_rows,
                bundle,
            )
        report = rank_material_application_candidates(
            brief,
            candidates=candidate_rows,
            observations=observations,
            preferences=preferences,
            rag_bundle=bundle,
        )
        run_payload = {
            "brief_id": brief.brief_id,
            "report_id": report.report_id,
            "rag_bundle_id": bundle.bundle_id if bundle else None,
        }
        run = MaterialDecisionRun(
            run_id=f"MDRUN-{stable_hash(run_payload)[:24]}",
            brief=brief,
            report=report,
            rag_bundle_id=bundle.bundle_id if bundle else None,
            rag_stage_receipts=list(rag_runs.receipts),
        )
        if self.artifact_root is not None:
            artifacts = self._persist(
                run,
                bundle,
                stage_bundles=rag_runs.stage_bundles,
            )
            run = run.model_copy(update={"artifacts": artifacts})
            # Persist the final run receipt after the artifact inventory is closed.
            run_path = self.artifact_root / run.run_id / "material-decision-run.json"
            _write_text(run_path, run.model_dump_json(indent=2) + "\n")
        return run

    def _classify(
        self,
        question: str,
        *,
        material_field: MaterialField | str,
        chemical_system: str | None,
        problem_context: Mapping[str, JsonValue],
        routing_mode: str,
    ) -> _ModelRuns:
        requested_auto = str(material_field).strip().casefold() in {
            "",
            "auto",
            "자동",
        }
        if routing_mode == "off":
            return _ModelRuns(field=None, application=None)
        field_classifier = build_main_model_material_field_classifier_from_environment(
            environ=self.environ,
            required=routing_mode == "required" and requested_auto,
        )
        field_run = None
        if requested_auto and field_classifier is not None:
            field_run = field_classifier.classify(
                question,
                chemical_system=chemical_system,
                problem_context=problem_context,
            )
        provisional = build_material_domain_plan(
            material_field,
            prompt=question,
            chemical_system=chemical_system,
            problem_context=problem_context,
            model_run=field_run,
        )
        application_classifier: MainModelMaterialApplicationClassifier | None = (
            build_main_model_material_application_classifier_from_environment(
                environ=self.environ,
                required=routing_mode == "required",
            )
        )
        application_run = None
        if application_classifier is not None:
            application_run = application_classifier.classify(
                question,
                material_field=provisional.resolution.selected_field,
                problem_context=problem_context,
            )
        return _ModelRuns(field=field_run, application=application_run)

    def _run_application_rag(
        self,
        brief: MaterialApplicationBrief,
        *,
        sources: Sequence[LiteratureSource] | None,
        from_date: date | None,
        to_date: date | None,
        max_results_per_query: int,
    ) -> _ApplicationRagRuns:
        if not 1 <= max_results_per_query <= 50:
            raise ValueError("RAG max results per query must be between 1 and 50")
        base_environment = dict(self.environ or {})
        if self.environ is None:
            import os

            base_environment = dict(os.environ)

        requested_sources = (
            [LiteratureSource(str(item)) for item in sources]
            if sources is not None
            else None
        )
        if requested_sources is not None and not requested_sources:
            raise ValueError("application RAG source selection cannot be empty")
        receipts: list[ApplicationRagStageReceipt] = []
        stage_bundles: list[
            tuple[MaterialEvidenceStage, RagEvidenceBundle]
        ] = []
        for stage in MATERIAL_EVIDENCE_STAGES:
            tasks = [
                task
                for role in brief.roles
                for task in role.evidence_tasks
                if task.evidence_stage == stage
            ]
            if not tasks:
                receipts.append(
                    ApplicationRagStageReceipt(
                        evidence_stage=stage,
                        task_ids=[f"{stage}-missing-task"],
                        role_ids=[brief.roles[0].role_id],
                        allowed_literature_sources=["crossref"],
                        request_hash=stable_hash(
                            {
                                "brief_id": brief.brief_id,
                                "stage": stage,
                                "error": "missing-stage-task",
                            }
                        ),
                        status="failed",
                        error=(
                            "No code-owned application evidence task was registered "
                            f"for stage {stage}."
                        ),
                    )
                )
                continue

            allowed = set(tasks[0].allowed_literature_sources)
            for task in tasks[1:]:
                allowed.intersection_update(task.allowed_literature_sources)
            if not allowed:
                raise ValueError(
                    f"application RAG stage {stage} has no common allowed source"
                )
            if requested_sources is None:
                selected_literature = [
                    LiteratureSource(item)
                    for item in ("crossref", "arxiv", "openalex")
                    if item in allowed
                ]
                request_mcp = True
            else:
                # A caller selects providers for the complete application
                # workflow.  Each isolated stage then intersects that request
                # with its code-owned allowlist.  For example, OpenAlex is used
                # by generation/identity but is deliberately omitted from the
                # later validation stages instead of aborting the whole run.
                selected_literature = [
                    item
                    for item in requested_sources
                    if item != LiteratureSource.MCP and item.value in allowed
                ]
                request_mcp = LiteratureSource.MCP in requested_sources

            stage_environment, selected_tool = _application_stage_rag_environment(
                base_environment,
                stage,
                mcp_requested=request_mcp,
            )
            prompt = _application_stage_rag_prompt(brief, stage, tasks)
            request_hash = stable_hash(
                {
                    "brief_id": brief.brief_id,
                    "stage": stage,
                    "task_ids": [item.task_id for item in tasks],
                    "prompt": prompt,
                    "sources": [item.value for item in selected_literature],
                    "mcp_tool": selected_tool,
                    "from_date": from_date,
                    "to_date": to_date,
                    "max_results_per_query": max_results_per_query,
                }
            )
            if not selected_literature and not selected_tool:
                receipts.append(
                    ApplicationRagStageReceipt(
                        evidence_stage=stage,
                        task_ids=[item.task_id for item in tasks],
                        role_ids=list(
                            dict.fromkeys(item.role_id for item in tasks)
                        ),
                        allowed_literature_sources=[],
                        required_mcp_capabilities=list(
                            dict.fromkeys(
                                capability
                                for task in tasks
                                for capability in task.mcp_capabilities
                            )
                        ),
                        selected_mcp_tool=None,
                        request_hash=request_hash,
                        status="failed",
                        error=(
                            "No caller-requested evidence provider is allowed and "
                            f"configured for application RAG stage {stage}; no "
                            "evidence from this stage was used."
                        ),
                    )
                )
                continue
            try:
                pipeline = build_literature_rag_from_environment(
                    environ=stage_environment,
                    require_model=False,
                )
                selected_sources = list(selected_literature)
                if (
                    request_mcp
                    and getattr(pipeline.retriever, "mcp_client", None)
                    is not None
                ):
                    selected_sources.append(LiteratureSource.MCP)
                if not selected_sources:
                    raise ValueError(
                        f"application RAG stage {stage} has no configured source"
                    )
                index = (
                    JsonEvidenceIndex(
                        self.artifact_root
                        / ".application-rag-index"
                        / str(stage)
                    )
                    if self.artifact_root is not None
                    else None
                )
                bundle = pipeline.run(
                    prompt,
                    sources=selected_sources,
                    from_date=from_date,
                    to_date=to_date,
                    max_results_per_query=max_results_per_query,
                    # Application evidence never becomes a generator branch.
                    # The composite below discards any generic planner branches.
                    max_branches=1,
                    index=index,
                )
                failed_sources = any(
                    item.status in {SourceRunStatus.FAILED, SourceRunStatus.SKIPPED}
                    for item in bundle.source_statuses
                )
                receipts.append(
                    ApplicationRagStageReceipt(
                        evidence_stage=stage,
                        task_ids=[item.task_id for item in tasks],
                        role_ids=list(
                            dict.fromkeys(item.role_id for item in tasks)
                        ),
                        allowed_literature_sources=[
                            item.value for item in selected_literature
                        ],
                        required_mcp_capabilities=list(
                            dict.fromkeys(
                                capability
                                for task in tasks
                                for capability in task.mcp_capabilities
                            )
                        ),
                        selected_mcp_tool=selected_tool,
                        request_hash=request_hash,
                        status=(
                            "completed_with_source_failures"
                            if failed_sources
                            else "completed"
                        ),
                        source_bundle_id=bundle.bundle_id,
                    )
                )
                stage_bundles.append((stage, bundle))
            except Exception as exc:
                receipts.append(
                    ApplicationRagStageReceipt(
                        evidence_stage=stage,
                        task_ids=[item.task_id for item in tasks],
                        role_ids=list(
                            dict.fromkeys(item.role_id for item in tasks)
                        ),
                        allowed_literature_sources=[
                            item.value for item in selected_literature
                        ],
                        required_mcp_capabilities=list(
                            dict.fromkeys(
                                capability
                                for task in tasks
                                for capability in task.mcp_capabilities
                            )
                        ),
                        selected_mcp_tool=selected_tool,
                        request_hash=request_hash,
                        status="failed",
                        error=_safe_error(exc),
                    )
                )
        composite = _compose_application_rag_bundles(brief, stage_bundles)
        return _ApplicationRagRuns(
            composite=composite,
            receipts=tuple(receipts),
            stage_bundles=tuple(stage_bundles),
        )

    def _persist(
        self,
        run: MaterialDecisionRun,
        bundle: RagEvidenceBundle | None,
        *,
        stage_bundles: Sequence[
            tuple[MaterialEvidenceStage, RagEvidenceBundle]
        ] = (),
    ) -> list[MaterialDecisionArtifact]:
        assert self.artifact_root is not None
        root = self.artifact_root / run.run_id
        paths: list[tuple[str, Path]] = []
        brief_path = root / "application-brief.json"
        report_path = root / "material-recommendation.json"
        markdown_path = root / "material-recommendation.md"
        csv_path = root / "material-recommendation.csv"
        _write_text(brief_path, run.brief.model_dump_json(indent=2) + "\n")
        _write_text(report_path, run.report.model_dump_json(indent=2) + "\n")
        _write_text(markdown_path, _report_markdown(run.report))
        _write_report_csv(csv_path, run.report)
        paths.extend(
            [
                ("application_brief", brief_path),
                ("recommendation_json", report_path),
                ("recommendation_markdown", markdown_path),
                ("recommendation_csv", csv_path),
            ]
        )
        if bundle is not None:
            bundle_path = root / "application-rag-bundle.json"
            save_evidence_bundle(bundle, bundle_path)
            paths.append(("application_rag_bundle", bundle_path))
        for stage, stage_bundle in stage_bundles:
            stage_path = root / "application-rag" / f"{stage}.json"
            save_evidence_bundle(stage_bundle, stage_path)
            paths.append((f"application_rag_{stage}", stage_path))
        return [
            MaterialDecisionArtifact(
                artifact_id=f"MDA-{stable_hash([kind, _sha256(path)])[:24]}",
                kind=kind,
                relative_path=path.relative_to(self.artifact_root).as_posix(),
                sha256=_sha256(path),
            )
            for kind, path in paths
        ]


def _application_stage_rag_environment(
    base_environment: Mapping[str, str],
    stage: MaterialEvidenceStage,
    *,
    mcp_requested: bool = True,
) -> tuple[dict[str, str], str | None]:
    environment = dict(base_environment)
    if not mcp_requested:
        for key in (
            "MATERIAL_RAG_MCP_URL",
            "MATERIAL_RAG_MCP_TOOL",
            "MATERIAL_RAG_MCP_TOKEN",
            "MATERIAL_RAG_MCP_TIMEOUT_SECONDS",
            "MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP",
        ):
            environment.pop(key, None)
        return environment, None
    dedicated = str(
        environment.get(_STAGE_MCP_TOOL_ENV[stage]) or ""
    ).strip()
    application = str(
        environment.get("MATERIAL_APPLICATION_RAG_MCP_TOOL") or ""
    ).strip()
    generic = str(environment.get("MATERIAL_RAG_MCP_TOOL") or "").strip()
    selected = dedicated or application or generic
    if selected:
        # The endpoint and all tool names are administrator configuration.
        # The prompt and reasoning model never contribute to this precedence.
        environment["MATERIAL_RAG_MCP_TOOL"] = selected
    else:
        for key in (
            "MATERIAL_RAG_MCP_URL",
            "MATERIAL_RAG_MCP_TOOL",
            "MATERIAL_RAG_MCP_TOKEN",
            "MATERIAL_RAG_MCP_TIMEOUT_SECONDS",
            "MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP",
        ):
            environment.pop(key, None)
    return environment, selected or None


def _application_stage_rag_prompt(
    brief: MaterialApplicationBrief,
    stage: MaterialEvidenceStage,
    tasks: Sequence[ApplicationEvidenceTask],
) -> str:
    if not tasks or any(item.evidence_stage != stage for item in tasks):
        raise ValueError("application RAG request must contain exactly one stage")
    rows = [
        "Stage-bounded material application evidence request.",
        f"Validation stage: {stage}.",
        "Do not answer or perform work for another validation stage.",
        f"User question: {brief.user_question}",
        f"Code-owned material field: {brief.material_field}.",
        f"Question kind: {brief.question_kind}.",
        (
            "Retrieve source-grounded supporting, conflicting, null, and negative "
            "evidence. Literature and MCP records are context only and must never "
            "be converted into a generated candidate's property or performance score."
        ),
        (
            "Every useful numeric record must retain material/phase/stack, component "
            "role, value, unit, complete conditions, geometry or thickness, measured "
            "or calculated method, sample/process, uncertainty, negative/null status, "
            "stable source identifier, and exact support span."
        ),
    ]
    if brief.target_context:
        rows.append(
            "Declared target context: "
            + json.dumps(
                brief.target_context,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    role_by_id = {item.role_id: item for item in brief.roles}
    for task in tasks:
        role = role_by_id[task.role_id]
        rows.append(f"Role {role.role_id}: {role.description}")
        rows.append("Claim boundary: " + role.claim_boundary)
        rows.append("Failure modes: " + " | ".join(role.failure_modes))
        rows.append(
            f"[{task.category}; stage={task.evidence_stage}] "
            + " | ".join(task.questions)
        )
        rows.append(
            "Required record fields: "
            + ", ".join(task.required_record_fields)
        )
        if task.mcp_capabilities:
            rows.append(
                "Read-only structured evidence capabilities: "
                + ", ".join(task.mcp_capabilities)
                + ". The administrator configuration, never this prompt, selects "
                "an MCP endpoint and tool."
            )
    if stage != "generation_prior":
        rows.append(
            "This later-stage evidence may request follow-up validation but cannot "
            "steer a generator or revise a candidate."
        )
    return "\n".join(rows)


def _compose_application_rag_bundles(
    brief: MaterialApplicationBrief,
    stage_bundles: Sequence[
        tuple[MaterialEvidenceStage, RagEvidenceBundle]
    ],
) -> RagEvidenceBundle | None:
    """Namespace and combine citation evidence without combining stage requests."""

    if not stage_bundles:
        return None
    queries: list[LiteratureQuery] = []
    statuses: list[SourceRetrievalStatus] = []
    records: list[LiteratureRecord] = []
    claims: list[EvidenceClaim] = []
    warnings = [
        (
            "Composite application evidence was retrieved through separate "
            "single-stage requests; it is citation context only and has no "
            "generator branches or property-scoring authority."
        )
    ]
    source_bundle_ids: list[dict[str, str]] = []
    for stage, bundle in stage_bundles:
        source_bundle_ids.append(
            {"stage": str(stage), "bundle_id": bundle.bundle_id}
        )
        query_map = {
            item.query_id: (
                f"ARQ-{stable_hash([stage, bundle.bundle_id, item.query_id])[:24]}"
            )
            for item in bundle.search_plan.queries
        }
        record_map = {
            item.record_id: (
                f"ARR-{stable_hash([stage, bundle.bundle_id, item.record_id])[:24]}"
            )
            for item in bundle.records
        }
        claim_map = {
            item.claim_id: (
                f"ARC-{stable_hash([stage, bundle.bundle_id, item.claim_id])[:24]}"
            )
            for item in bundle.claims
        }
        for item in bundle.search_plan.queries:
            queries.append(
                item.model_copy(
                    update={"query_id": query_map[item.query_id]},
                    deep=True,
                )
            )
        for item in bundle.source_statuses:
            mapped_query_ids = [
                query_map.get(
                    query_id,
                    f"ARQ-{stable_hash([stage, bundle.bundle_id, query_id])[:24]}",
                )
                for query_id in item.query_ids
            ]
            statuses.append(
                item.model_copy(
                    update={"query_ids": mapped_query_ids},
                    deep=True,
                )
            )
        for item in bundle.records:
            mapped_source_queries = [
                query_map.get(
                    query_id,
                    f"ARQ-{stable_hash([stage, bundle.bundle_id, query_id])[:24]}",
                )
                for query_id in item.source_queries
            ]
            raw_metadata = dict(item.raw_metadata)
            raw_metadata["application_evidence_stage"] = str(stage)
            raw_metadata["source_bundle_id"] = bundle.bundle_id
            records.append(
                item.model_copy(
                    update={
                        "record_id": record_map[item.record_id],
                        "source_queries": mapped_source_queries,
                        "raw_metadata": raw_metadata,
                    },
                    deep=True,
                )
            )
        for item in bundle.claims:
            qualifiers = dict(item.qualifiers)
            qualifiers["application_evidence_stage"] = str(stage)
            qualifiers["source_bundle_id"] = bundle.bundle_id
            claims.append(
                item.model_copy(
                    update={
                        "claim_id": claim_map[item.claim_id],
                        "source_record_id": record_map[item.source_record_id],
                        "qualifiers": qualifiers,
                    },
                    deep=True,
                )
            )
        warnings.extend(
            f"[{stage}; {bundle.bundle_id}] {item}"
            for item in bundle.warnings
        )

    graph = EvidenceGraphBuilder().build(claims)
    created_at = datetime.now(timezone.utc)
    plan_payload = {
        "brief_id": brief.brief_id,
        "source_bundles": source_bundle_ids,
        "queries": [item.model_dump(mode="json") for item in queries],
    }
    plan = RagSearchPlan(
        plan_id=f"ARPLAN-{stable_hash(plan_payload)[:24]}",
        user_prompt=brief.user_question,
        generated_at=created_at,
        planner_id="application-stage-composite",
        planner_version="1.0",
        concepts=[str(brief.material_field)],
        target_entities=[item.role_id for item in brief.roles],
        queries=queries,
    )
    bundle_payload = {
        "brief_id": brief.brief_id,
        "source_bundles": source_bundle_ids,
        "plan_id": plan.plan_id,
        "record_ids": [item.record_id for item in records],
        "claim_ids": [item.claim_id for item in claims],
    }
    return RagEvidenceBundle(
        bundle_id=f"ARB-{stable_hash(bundle_payload)[:24]}",
        created_at=created_at,
        search_plan=plan,
        source_statuses=statuses,
        records=records,
        claims=claims,
        graph=graph,
        branches=[],
        warnings=warnings,
    )


def _safe_error(exc: Exception) -> str:
    return (
        f"{type(exc).__name__}: stage retrieval failed; no evidence from this "
        "request was used."
    )


def _link_candidates_to_exact_rag_claims(
    candidates: Sequence[MaterialApplicationCandidate],
    bundle: RagEvidenceBundle,
) -> list[MaterialApplicationCandidate]:
    """Link only literal candidate mentions; never infer a property or score."""

    linked: list[MaterialApplicationCandidate] = []
    for candidate in candidates:
        claim_ids = list(candidate.evidence_claim_ids)
        needle = candidate.material_or_stack.strip()
        if len(needle) >= 2:
            for claim in bundle.claims:
                corpus = " ".join(
                    [
                        claim.subject,
                        claim.object,
                        claim.support_text,
                    ]
                )
                if _literal_material_mention(needle, corpus):
                    claim_ids.append(claim.claim_id)
        linked.append(
            candidate.model_copy(
                update={"evidence_claim_ids": list(dict.fromkeys(claim_ids))}
            )
        )
    return linked


def _literal_material_mention(needle: str, corpus: str) -> bool:
    normalized_needle = re.sub(r"\s+", " ", needle.casefold()).strip()
    normalized_corpus = re.sub(r"\s+", " ", corpus.casefold()).strip()
    if re.search(r"[가-힣]", normalized_needle):
        return normalized_needle in normalized_corpus
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_needle)}(?![a-z0-9])",
            normalized_corpus,
        )
        is not None
    )


def _report_markdown(report: MaterialRecommendationReport) -> str:
    rows = [
        f"# Material application decision: {report.brief.user_question}",
        "",
        (
            "Role-scoped decision support only. Unlike components and operating "
            "conditions are not cross-ranked."
        ),
        "",
    ]
    for portfolio in report.role_recommendations:
        rows.extend(
            [
                f"## {portfolio.role_profile.display_name}",
                "",
                portfolio.role_claim_boundary,
                "",
                "| Candidate | Origin | Hard gates | Pareto | Decision score | Evidence | Why | Unknowns / next validation |",
                "|---|---|---:|---:|---:|---:|---|---|",
            ]
        )
        for item in portfolio.candidates:
            next_step = (
                item.next_validations[0]
                if item.next_validations
                else "No additional step recorded."
            )
            rows.append(
                "| "
                + " | ".join(
                    [
                        _md(item.candidate.material_or_stack),
                        item.candidate.origin,
                        item.hard_gate_status,
                        str(item.pareto_front or "UNKNOWN"),
                        (
                            f"{item.pool_relative_decision_score:.2f}"
                            if item.pool_relative_decision_score is not None
                            else "UNKNOWN"
                        ),
                        f"{item.evidence_completeness_score:.1f}%",
                        _md(", ".join(item.why_selected)),
                        _md(next_step),
                    ]
                )
                + " |"
            )
        rows.append("")
    if report.unresolved_questions:
        rows.extend(
            [
                "## Unresolved context",
                "",
                *[f"- {_md(item)}" for item in report.unresolved_questions],
                "",
            ]
        )
    return "\n".join(rows).rstrip() + "\n"


def _write_report_csv(
    path: Path,
    report: MaterialRecommendationReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "role_id",
                "candidate_id",
                "material_or_stack",
                "origin",
                "comparison_group_id",
                "rank",
                "pareto_front",
                "hard_gate_status",
                "pool_relative_decision_score",
                "evidence_completeness_score",
                "uncertainty_status",
                "why_selected",
                "tradeoffs",
                "citations",
                "next_validations",
            ],
        )
        writer.writeheader()
        for portfolio in report.role_recommendations:
            for item in portfolio.candidates:
                writer.writerow(
                    {
                        "role_id": portfolio.role_id,
                        "candidate_id": item.candidate.candidate_id,
                        "material_or_stack": item.candidate.material_or_stack,
                        "origin": item.candidate.origin,
                        "comparison_group_id": item.comparison_group_id or "",
                        "rank": item.rank_within_role_and_condition or "",
                        "pareto_front": item.pareto_front or "",
                        "hard_gate_status": item.hard_gate_status,
                        "pool_relative_decision_score": (
                            item.pool_relative_decision_score
                            if item.pool_relative_decision_score is not None
                            else ""
                        ),
                        "evidence_completeness_score": item.evidence_completeness_score,
                        "uncertainty_status": item.evidence_uncertainty_status,
                        "why_selected": ";".join(item.why_selected),
                        "tradeoffs": ";".join(item.main_tradeoffs),
                        "citations": ";".join(
                            citation.doi or citation.record_id
                            for citation in item.citations
                        ),
                        "next_validations": ";".join(item.next_validations),
                    }
                )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


__all__ = [
    "ApplicationRagStageReceipt",
    "MaterialDecisionArtifact",
    "MaterialDecisionRun",
    "MaterialDecisionRunner",
]
