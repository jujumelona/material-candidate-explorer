"""Stage-specific, source-grounded evidence routing for material validation.

The router complements numerical validators; it never replaces them.  Each
stage selects a bounded set of scholarly providers and the one administrator-
configured MCP tool.  Retrieval failures remain ``unknown`` and record counts
never become material-property scores.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from ._compat import StrEnum
from .fusion_schemas import FusionDecisionContext
from .hashing import stable_hash
from .literature_rag import (
    EvidenceBranchAssignment,
    LiteratureEvidencePolicy,
    LiteratureRagPipeline,
    LiteratureSource,
    RagEvidenceBundle,
    SourceRetrievalStatus,
    SourceRunStatus,
    build_literature_rag_from_environment,
    save_evidence_bundle,
)
from .schemas import CandidateRef, DiscoveryGoal, Identifier, NonEmptyText, StrictSchema


class ValidationEvidenceStage(StrEnum):
    GENERATION_PRIOR = "generation_prior"
    IDENTITY_NOVELTY = "identity_novelty"
    MLIP_DISAGREEMENT = "mlip_disagreement"
    RELAXATION_VALIDATION = "relaxation_validation"
    DFT_HANDOFF = "dft_handoff"


class ValidationEvidenceStatus(StrEnum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    SKIPPED = "skipped"


class ValidationEvidenceRoute(StrictSchema):
    stage: ValidationEvidenceStage
    literature_sources: list[LiteratureSource] = Field(min_length=1)
    official_validators: list[Identifier] = Field(min_length=1)
    validator_role: Literal["runtime-authority-not-invoked-by-evidence-router"] = (
        "runtime-authority-not-invoked-by-evidence-router"
    )
    mcp_policy: Literal["configured-tool-only"] = "configured-tool-only"


class ValidationEvidenceRequest(StrictSchema):
    stage: ValidationEvidenceStage
    chemical_system: NonEmptyText
    candidate_refs: list[CandidateRef] = Field(default_factory=list, max_length=128)
    composition_keys: list[str] = Field(default_factory=list, max_length=128)
    observations: dict[str, JsonValue] = Field(default_factory=dict)
    focus: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def _request_is_bounded_and_secret_free(self) -> "ValidationEvidenceRequest":
        ref_keys = [
            (item.candidate_id, item.version, item.content_hash)
            for item in self.candidate_refs
        ]
        if len(ref_keys) != len(set(ref_keys)):
            raise ValueError("validation evidence candidate_refs must be unique")
        normalized = [item.strip() for item in self.composition_keys]
        if any(not item or len(item) > 256 for item in normalized):
            raise ValueError("validation evidence composition keys are invalid")
        if len(normalized) != len(set(normalized)):
            raise ValueError("validation evidence composition keys must be unique")
        if _contains_sensitive_key(self.observations):
            raise ValueError("validation evidence observations cannot contain secrets")
        return self


class ValidationEvidenceReport(StrictSchema):
    report_id: Identifier
    stage: ValidationEvidenceStage
    status: ValidationEvidenceStatus
    route: ValidationEvidenceRoute
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundle_id: Identifier | None = None
    bundle_relative_path: str | None = Field(default=None, max_length=2_000)
    source_statuses: list[SourceRetrievalStatus] = Field(default_factory=list)
    record_count: int = Field(default=0, ge=0)
    claim_count: int = Field(default=0, ge=0)
    branch_count: int = Field(default=0, ge=0)
    reason: str | None = Field(default=None, max_length=4_000)
    warnings: list[str] = Field(default_factory=list)
    scientific_role: Literal["search_and_validation_context_only"] = (
        "search_and_validation_context_only"
    )
    property_score_created: Literal[False] = False

    @model_validator(mode="after")
    def _unknown_and_bundle_boundaries(self) -> "ValidationEvidenceReport":
        if self.status in {
            ValidationEvidenceStatus.UNKNOWN,
            ValidationEvidenceStatus.SKIPPED,
        } and not self.reason:
            raise ValueError("unknown/skipped validation evidence requires a reason")
        if (self.bundle_id is None) != (self.bundle_relative_path is None):
            raise ValueError("bundle identity and path must be present together")
        if self.bundle_id is None and any(
            (self.record_count, self.claim_count, self.branch_count)
        ):
            raise ValueError("evidence counts require a persisted bundle")
        return self


@dataclass(frozen=True, slots=True)
class ValidationEvidenceRun:
    report: ValidationEvidenceReport
    bundle: RagEvidenceBundle | None
    report_path: Path


_ROUTES: dict[ValidationEvidenceStage, ValidationEvidenceRoute] = {
    ValidationEvidenceStage.GENERATION_PRIOR: ValidationEvidenceRoute(
        stage=ValidationEvidenceStage.GENERATION_PRIOR,
        literature_sources=[
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.OPENALEX,
            LiteratureSource.MCP,
        ],
        official_validators=[
            "mattergen-supported-condition-allowlist",
            "evidence-driven-fusion-controller",
        ],
    ),
    ValidationEvidenceStage.IDENTITY_NOVELTY: ValidationEvidenceRoute(
        stage=ValidationEvidenceStage.IDENTITY_NOVELTY,
        literature_sources=[
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.OPENALEX,
            LiteratureSource.MCP,
        ],
        official_validators=[
            "pymatgen-structure-matcher",
            "materials-project-find-structure",
        ],
    ),
    ValidationEvidenceStage.MLIP_DISAGREEMENT: ValidationEvidenceRoute(
        stage=ValidationEvidenceStage.MLIP_DISAGREEMENT,
        literature_sources=[
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.MCP,
        ],
        official_validators=[
            "mattersim-sidecar",
            "chgnet-sidecar",
            "cross-model-unit-normalized-disagreement",
        ],
    ),
    ValidationEvidenceStage.RELAXATION_VALIDATION: ValidationEvidenceRoute(
        stage=ValidationEvidenceStage.RELAXATION_VALIDATION,
        literature_sources=[
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.MCP,
        ],
        official_validators=[
            "ase-periodic-optimizer",
            "mattersim-relaxation",
            "chgnet-relaxation",
        ],
    ),
    ValidationEvidenceStage.DFT_HANDOFF: ValidationEvidenceRoute(
        stage=ValidationEvidenceStage.DFT_HANDOFF,
        literature_sources=[
            LiteratureSource.CROSSREF,
            LiteratureSource.ARXIV,
            LiteratureSource.MCP,
        ],
        official_validators=[
            "periodic-dft-backend-contract",
            "external-pseudopotential-review",
            "reference-phase-convergence-review",
        ],
    ),
}


_STAGE_INSTRUCTIONS: dict[ValidationEvidenceStage, str] = {
    ValidationEvidenceStage.GENERATION_PRIOR: (
        "Find experimentally characterized phases, negative or failed synthesis "
        "conditions, composition ranges, and reported stability constraints."
    ),
    ValidationEvidenceStage.IDENTITY_NOVELTY: (
        "Find reported phases and crystallographic records for these compositions. "
        "Text matches are context only; structure identity is decided by tolerance-aware "
        "structure matching and configured structure databases."
    ),
    ValidationEvidenceStage.MLIP_DISAGREEMENT: (
        "Find model validation limits, out-of-domain chemistry, magnetic or charge-state "
        "effects, and published calculations that may explain cross-potential disagreement."
    ),
    ValidationEvidenceStage.RELAXATION_VALIDATION: (
        "Find reported phase transformations, mechanical or dynamical instabilities, "
        "pressure/temperature conditions, and relaxation or phonon evidence."
    ),
    ValidationEvidenceStage.DFT_HANDOFF: (
        "Find reference phases, magnetic ordering, functional or Hubbard-U choices, "
        "pseudopotential/convergence considerations, and phonon validation workflows."
    ),
}


class ValidationEvidenceRouter:
    """Run one bounded, independently persisted evidence retrieval per stage."""

    def __init__(
        self,
        pipeline: LiteratureRagPipeline | None,
        *,
        artifact_root: Path | str,
        enabled: bool = True,
        from_date: date | None = None,
        to_date: date | None = None,
        max_results_per_query: int = 8,
        max_branches: int = 12,
    ) -> None:
        if not 1 <= max_results_per_query <= 50:
            raise ValueError("max_results_per_query must be between 1 and 50")
        if not 1 <= max_branches <= 50:
            raise ValueError("max_branches must be between 1 and 50")
        if from_date and to_date and from_date > to_date:
            raise ValueError("from_date must not be after to_date")
        self.pipeline = pipeline
        self.artifact_root = Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.from_date = from_date
        self.to_date = to_date
        self.max_results_per_query = max_results_per_query
        self.max_branches = max_branches

    def run(
        self,
        request: ValidationEvidenceRequest,
        *,
        goal: DiscoveryGoal | None = None,
    ) -> ValidationEvidenceRun:
        route = _ROUTES[request.stage]
        prompt = build_validation_evidence_prompt(request)
        request_hash = stable_hash(request)
        prompt_hash = stable_hash(prompt)
        if not self.enabled:
            report = self._report(
                request=request,
                route=route,
                status=ValidationEvidenceStatus.SKIPPED,
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                reason="stage_evidence_disabled",
            )
            return self._persist(report, None)
        if self.pipeline is None:
            report = self._report(
                request=request,
                route=route,
                status=ValidationEvidenceStatus.UNKNOWN,
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                reason="literature_rag_pipeline_not_configured",
            )
            return self._persist(report, None)
        try:
            bundle = self.pipeline.run(
                prompt,
                goal=goal,
                sources=[
                    LiteratureSource(str(source))
                    for source in route.literature_sources
                ],
                from_date=self.from_date,
                to_date=self.to_date,
                max_results_per_query=self.max_results_per_query,
                max_branches=self.max_branches,
            )
        except Exception as exc:
            report = self._report(
                request=request,
                route=route,
                status=ValidationEvidenceStatus.UNKNOWN,
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                reason=f"stage_evidence_retrieval_failed:{type(exc).__name__}",
            )
            return self._persist(report, None)

        degraded = any(
            item.status in {
                SourceRunStatus.PARTIAL,
                SourceRunStatus.SKIPPED,
                SourceRunStatus.FAILED,
            }
            for item in bundle.source_statuses
        )
        if not bundle.records:
            status = ValidationEvidenceStatus.UNKNOWN
            reason = "no_source_grounded_records_retrieved"
        elif degraded:
            status = ValidationEvidenceStatus.PARTIAL
            reason = "one_or_more_evidence_sources_were_unavailable"
        else:
            status = ValidationEvidenceStatus.COMPLETED
            reason = None
        bundle_relative = (
            Path("validation-evidence")
            / str(request.stage)
            / f"{bundle.bundle_id}.json"
        )
        save_evidence_bundle(bundle, self.artifact_root / bundle_relative)
        report = self._report(
            request=request,
            route=route,
            status=status,
            request_hash=request_hash,
            prompt_hash=prompt_hash,
            bundle=bundle,
            bundle_relative_path=bundle_relative.as_posix(),
            reason=reason,
        )
        return self._persist(report, bundle)

    def _report(
        self,
        *,
        request: ValidationEvidenceRequest,
        route: ValidationEvidenceRoute,
        status: ValidationEvidenceStatus,
        request_hash: str,
        prompt_hash: str,
        bundle: RagEvidenceBundle | None = None,
        bundle_relative_path: str | None = None,
        reason: str | None = None,
    ) -> ValidationEvidenceReport:
        payload = {
            "stage": request.stage,
            "status": status,
            "request_hash": request_hash,
            "prompt_hash": prompt_hash,
            "bundle_id": bundle.bundle_id if bundle else None,
        }
        return ValidationEvidenceReport(
            report_id=f"VREPORT-{stable_hash(payload)[:24]}",
            stage=request.stage,
            status=status,
            route=route,
            request_hash=request_hash,
            prompt_hash=prompt_hash,
            bundle_id=bundle.bundle_id if bundle else None,
            bundle_relative_path=bundle_relative_path,
            source_statuses=list(bundle.source_statuses) if bundle else [],
            record_count=len(bundle.records) if bundle else 0,
            claim_count=len(bundle.claims) if bundle else 0,
            branch_count=len(bundle.branches) if bundle else 0,
            reason=reason,
            warnings=list(bundle.warnings) if bundle else [],
        )

    def _persist(
        self,
        report: ValidationEvidenceReport,
        bundle: RagEvidenceBundle | None,
    ) -> ValidationEvidenceRun:
        relative = (
            Path("validation-evidence")
            / str(report.stage)
            / f"{report.report_id}.report.json"
        )
        target = self.artifact_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return ValidationEvidenceRun(report=report, bundle=bundle, report_path=target)


def build_validation_evidence_prompt(request: ValidationEvidenceRequest) -> str:
    candidate_ids = [item.candidate_id for item in request.candidate_refs]
    observations = json.dumps(
        request.observations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(observations) > 12_000:
        raise ValueError("validation evidence observations exceed 12000 characters")
    rows = [
        f"Validation stage: {request.stage}.",
        f"Chemical system: {request.chemical_system}.",
        _STAGE_INSTRUCTIONS[request.stage],
        "Retrieve source-grounded supporting, conflicting, null, and negative evidence. "
        "Do not invent material properties and do not treat absence of a record as novelty.",
    ]
    if request.composition_keys:
        rows.append("Reduced compositions: " + ", ".join(request.composition_keys) + ".")
    if candidate_ids:
        rows.append("Candidate identifiers: " + ", ".join(candidate_ids) + ".")
    if request.focus:
        rows.append("Stage focus: " + request.focus.strip())
    if request.observations:
        rows.append("Machine observations (search context only): " + observations)
    return "\n".join(rows)


def fusion_decision_context_from_stage_evidence(
    run: ValidationEvidenceRun,
    *,
    guidance_alpha: float,
    exploration_branch: Literal[
        "stability",
        "target_property",
        "novelty",
        "expert_disagreement",
        "pareto",
    ],
    round_index: int = 0,
) -> FusionDecisionContext:
    """Bind a source-closed generation-prior branch to a fusion iteration."""

    if run.report.stage != ValidationEvidenceStage.GENERATION_PRIOR:
        raise ValueError("only generation-prior evidence can guide generation")
    base = {
        "guidance_alpha": guidance_alpha,
        "exploration_branch": exploration_branch,
    }
    if run.bundle is None or not run.bundle.branches:
        return FusionDecisionContext(**base)
    assignment = LiteratureEvidencePolicy(run.bundle).select(
        round_index=round_index,
        exploration_branch=exploration_branch,
    )
    return _decision_context_from_assignment(
        guidance_alpha=guidance_alpha,
        exploration_branch=exploration_branch,
        assignment=assignment,
    )


def fusion_decision_contexts_from_stage_evidence(
    run: ValidationEvidenceRun,
    *,
    controls: Sequence[
        tuple[
            float,
            Literal[
                "stability",
                "target_property",
                "novelty",
                "expert_disagreement",
                "pareto",
            ],
        ]
    ],
) -> list[FusionDecisionContext]:
    """Allocate several search workers without resetting evidence-branch state.

    A single :class:`LiteratureEvidencePolicy` instance is shared across the
    allocation so independent profile workers explore different supported
    branches when the evidence bundle contains them.
    """

    if run.report.stage != ValidationEvidenceStage.GENERATION_PRIOR:
        raise ValueError("only generation-prior evidence can guide generation")
    if run.bundle is None or not run.bundle.branches:
        return [
            FusionDecisionContext(
                guidance_alpha=guidance_alpha,
                exploration_branch=exploration_branch,
            )
            for guidance_alpha, exploration_branch in controls
        ]
    policy = LiteratureEvidencePolicy(run.bundle)
    contexts: list[FusionDecisionContext] = []
    for round_index, (guidance_alpha, exploration_branch) in enumerate(controls):
        assignment = policy.select(
            round_index=round_index,
            exploration_branch=exploration_branch,
        )
        contexts.append(
            _decision_context_from_assignment(
                guidance_alpha=guidance_alpha,
                exploration_branch=exploration_branch,
                assignment=assignment,
            )
        )
    return contexts


def _decision_context_from_assignment(
    *,
    guidance_alpha: float,
    exploration_branch: Literal[
        "stability",
        "target_property",
        "novelty",
        "expert_disagreement",
        "pareto",
    ],
    assignment: EvidenceBranchAssignment | None,
) -> FusionDecisionContext:
    if assignment is None:
        return FusionDecisionContext(
            guidance_alpha=guidance_alpha,
            exploration_branch=exploration_branch,
        )
    return FusionDecisionContext(
        guidance_alpha=guidance_alpha,
        exploration_branch=exploration_branch,
        evidence_branch_id=assignment.branch_id,
        evidence_branch_kind=str(assignment.branch_kind),
        evidence_claim_ids=list(assignment.source_claim_ids),
        evidence_generator_hints=dict(assignment.generator_hints),
        evidence_rationale=assignment.rationale,
    )


def build_validation_evidence_router_from_environment(
    *,
    artifact_root: Path | str,
    environ: Mapping[str, str] | None = None,
    enabled: bool | None = None,
) -> ValidationEvidenceRouter:
    values = os.environ if environ is None else environ
    effective_enabled = (
        str(values.get("VALIDATION_EVIDENCE_ENABLED", "")).strip() == "1"
        if enabled is None
        else enabled
    )
    pipeline = (
        build_literature_rag_from_environment(
            environ=values,
            require_model=str(
                values.get("VALIDATION_EVIDENCE_REQUIRE_RAG_MODEL", "")
            ).strip()
            == "1",
        )
        if effective_enabled
        else None
    )
    return ValidationEvidenceRouter(
        pipeline,
        artifact_root=artifact_root,
        enabled=effective_enabled,
        from_date=_optional_iso_date(values.get("VALIDATION_EVIDENCE_FROM_DATE")),
        to_date=_optional_iso_date(values.get("VALIDATION_EVIDENCE_TO_DATE")),
        max_results_per_query=int(
            str(values.get("VALIDATION_EVIDENCE_MAX_RESULTS", "8"))
        ),
        max_branches=int(
            str(values.get("VALIDATION_EVIDENCE_MAX_BRANCHES", "12"))
        ),
    )


def _optional_iso_date(value: object) -> date | None:
    raw = str(value or "").strip()
    return date.fromisoformat(raw) if raw else None


def _contains_sensitive_key(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if any(
                marker in normalized
                for marker in ("api_key", "token", "secret", "password", "credential")
            ):
                return True
            if _contains_sensitive_key(child):
                return True
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


__all__ = [
    "ValidationEvidenceReport",
    "ValidationEvidenceRequest",
    "ValidationEvidenceRoute",
    "ValidationEvidenceRouter",
    "ValidationEvidenceRun",
    "ValidationEvidenceStage",
    "ValidationEvidenceStatus",
    "build_validation_evidence_prompt",
    "build_validation_evidence_router_from_environment",
    "fusion_decision_context_from_stage_evidence",
    "fusion_decision_contexts_from_stage_evidence",
]
