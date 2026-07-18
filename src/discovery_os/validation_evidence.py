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
from .mcp_client import McpClientError
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


class ValidationCapability(StrEnum):
    """Allow-listed in-process or sidecar capability for a validation stage."""

    GENERATION_CONDITION_GUARD = "generation_condition_guard"
    EVIDENCE_SEARCH_CONTROL = "evidence_search_control"
    CRYSTAL_IDENTITY = "crystal_identity"
    EXTERNAL_STRUCTURE_NOVELTY = "external_structure_novelty"
    PERIODIC_MLIP_INFERENCE = "periodic_mlip_inference"
    CROSS_MODEL_DISAGREEMENT = "cross_model_disagreement"
    PERIODIC_RELAXATION = "periodic_relaxation"
    RELAXED_GEOMETRY_GATE = "relaxed_geometry_gate"
    PERIODIC_DFT_PREPARATION = "periodic_dft_preparation"
    DFT_METHOD_REVIEW = "dft_method_review"


class ValidatorAvailability(StrEnum):
    IMPLEMENTED = "implemented"
    SIDECAR_REQUIRED = "sidecar_required"
    CREDENTIAL_REQUIRED = "credential_required"
    EXTERNAL_REQUIRED = "external_required"


class McpEvidenceCapability(StrEnum):
    GENERATION_PRIOR_SEARCH = "materials_generation_prior_search"
    CRYSTAL_REFERENCE_SEARCH = "crystal_identity_reference_search"
    MLIP_LIMIT_SEARCH = "mlip_domain_limit_search"
    RELAXATION_INSTABILITY_SEARCH = "relaxation_instability_search"
    DFT_METHOD_SEARCH = "periodic_dft_method_search"


class ValidationHandoffKind(StrEnum):
    GENERATION_CONSTRAINT_CONTEXT = "generation_constraint_context"
    IDENTITY_NOVELTY_CONTEXT = "identity_novelty_context"
    MLIP_DISAGREEMENT_CONTEXT = "mlip_disagreement_context"
    RELAXATION_GATE_CONTEXT = "relaxation_gate_context"
    DFT_PREPARATION_CONTEXT = "dft_preparation_context"


class McpContractVerificationStatus(StrEnum):
    VERIFIED = "verified"
    NOT_CONFIGURED = "not_configured"
    NOT_CHECKED = "not_checked"
    NOT_VERIFIABLE = "not_verifiable"
    FAILED = "failed"


class ValidatorAuthority(StrictSchema):
    authority_id: Identifier
    capability: ValidationCapability
    implementation: NonEmptyText
    input_schema: Identifier
    output_schema: Identifier
    availability: ValidatorAvailability = ValidatorAvailability.IMPLEMENTED
    execution_role: Literal["runtime-authority-not-invoked-by-evidence-router"] = (
        "runtime-authority-not-invoked-by-evidence-router"
    )
    failure_policy: Literal["unknown-not-pass"] = "unknown-not-pass"


class McpEvidenceContract(StrictSchema):
    capability: McpEvidenceCapability
    tool_environment_variable: Identifier
    fallback_tool_environment_variable: Literal["MATERIAL_RAG_MCP_TOOL"] = (
        "MATERIAL_RAG_MCP_TOOL"
    )
    accepted_arguments: list[Literal["query", "max_results", "from_date", "to_date"]] = (
        Field(min_length=4, max_length=4)
    )
    result_collection: Literal["records"] = "records"
    required_record_fields: list[Literal["source_id", "title"]] = Field(
        min_length=2,
        max_length=2,
    )
    selection_policy: Literal["administrator-configured-allowlist-only"] = (
        "administrator-configured-allowlist-only"
    )
    failure_policy: Literal["source-unknown-do-not-fallback-to-model-memory"] = (
        "source-unknown-do-not-fallback-to-model-memory"
    )

    @model_validator(mode="after")
    def _fixed_adapter_contract(self) -> "McpEvidenceContract":
        if self.accepted_arguments != ["query", "max_results", "from_date", "to_date"]:
            raise ValueError("MCP evidence arguments must match the bounded adapter contract")
        if self.required_record_fields != ["source_id", "title"]:
            raise ValueError("MCP evidence records require source_id and title")
        return self


class ValidationFailurePolicy(StrictSchema):
    retrieval_exception: Literal["unknown"] = "unknown"
    no_records: Literal["unknown"] = "unknown"
    degraded_source: Literal["partial"] = "partial"
    validator_failure: Literal["unknown-not-pass"] = "unknown-not-pass"
    record_absence: Literal["not-proof-of-novelty-or-validity"] = (
        "not-proof-of-novelty-or-validity"
    )


class ValidationHandoffContract(StrictSchema):
    kind: ValidationHandoffKind
    consumer: Identifier
    payload_schema: Identifier
    validator_result_required: Literal[True] = True
    evidence_can_replace_validator: Literal[False] = False
    can_steer_generation: bool = False


class ValidationEvidenceHandoff(StrictSchema):
    handoff_id: Identifier
    report_id: Identifier
    stage: ValidationEvidenceStage
    kind: ValidationHandoffKind
    consumer: Identifier
    payload_schema: Identifier
    evidence_status: ValidationEvidenceStatus
    candidate_refs: list[CandidateRef] = Field(default_factory=list, max_length=128)
    composition_keys: list[str] = Field(default_factory=list, max_length=128)
    bundle_id: Identifier | None = None
    evidence_claim_ids: list[Identifier] = Field(default_factory=list, max_length=1_000)
    evidence_branch_ids: list[Identifier] = Field(default_factory=list, max_length=50)
    validator_authority_ids: list[Identifier] = Field(min_length=1)
    validator_execution_state: Literal["not_executed"] = "not_executed"
    decision_authority: Literal["validator-required"] = "validator-required"
    evidence_available: bool = False
    can_steer_generation: bool = False
    unknown_not_pass: Literal[True] = True
    property_score_created: Literal[False] = False

    @model_validator(mode="after")
    def _handoff_is_fail_closed(self) -> "ValidationEvidenceHandoff":
        has_evidence = self.bundle_id is not None and self.evidence_status in {
            ValidationEvidenceStatus.COMPLETED,
            ValidationEvidenceStatus.PARTIAL,
        }
        if self.evidence_available != has_evidence:
            raise ValueError("handoff evidence_available must match persisted usable evidence")
        if len(self.evidence_claim_ids) != len(set(self.evidence_claim_ids)):
            raise ValueError("handoff evidence claim identifiers must be unique")
        if len(self.evidence_branch_ids) != len(set(self.evidence_branch_ids)):
            raise ValueError("handoff evidence branch identifiers must be unique")
        if self.stage != ValidationEvidenceStage.GENERATION_PRIOR and self.evidence_branch_ids:
            raise ValueError("non-generation handoffs cannot expose generator branches")
        may_steer = (
            self.stage == ValidationEvidenceStage.GENERATION_PRIOR
            and self.evidence_available
            and bool(self.evidence_branch_ids)
        )
        if self.can_steer_generation != may_steer:
            raise ValueError(
                "only generation-prior handoffs with usable evidence may steer generation"
            )
        return self


class ValidationEvidenceRoute(StrictSchema):
    stage: ValidationEvidenceStage
    literature_sources: list[LiteratureSource] = Field(min_length=1)
    official_validators: list[Identifier] = Field(min_length=1)
    validator_authorities: list[ValidatorAuthority] = Field(min_length=1)
    mcp_contract: McpEvidenceContract
    failure_policy: ValidationFailurePolicy = Field(default_factory=ValidationFailurePolicy)
    handoff_contract: ValidationHandoffContract
    validator_role: Literal["runtime-authority-not-invoked-by-evidence-router"] = (
        "runtime-authority-not-invoked-by-evidence-router"
    )
    mcp_policy: Literal["configured-tool-only"] = "configured-tool-only"

    @model_validator(mode="after")
    def _route_is_closed(self) -> "ValidationEvidenceRoute":
        if len(self.literature_sources) != len(set(self.literature_sources)):
            raise ValueError("validation evidence sources must be unique")
        authority_ids = [item.authority_id for item in self.validator_authorities]
        if len(authority_ids) != len(set(authority_ids)):
            raise ValueError("validator authority identifiers must be unique")
        if self.official_validators != authority_ids:
            raise ValueError("official_validators must exactly match validator authorities")
        expected_tool_env = f"MATERIAL_RAG_MCP_TOOL_{str(self.stage).upper()}"
        if self.mcp_contract.tool_environment_variable != expected_tool_env:
            raise ValueError("MCP tool environment variable must be stage-specific")
        if self.handoff_contract.can_steer_generation != (
            self.stage == ValidationEvidenceStage.GENERATION_PRIOR
        ):
            raise ValueError("only generation-prior route may steer generation")
        return self


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
    mcp_contract_status: McpContractVerificationStatus
    mcp_tool_name: Identifier | None = None
    handoff: ValidationEvidenceHandoff
    reason: str | None = Field(default=None, max_length=4_000)
    warnings: list[str] = Field(default_factory=list)
    scientific_role: Literal["search_and_validation_context_only"] = (
        "search_and_validation_context_only"
    )
    property_score_created: Literal[False] = False

    @model_validator(mode="after")
    def _unknown_and_bundle_boundaries(self) -> "ValidationEvidenceReport":
        if self.route.stage != self.stage:
            raise ValueError("validation evidence route stage does not match report")
        if self.handoff.report_id != self.report_id or self.handoff.stage != self.stage:
            raise ValueError("validation evidence handoff does not match report")
        if self.handoff.kind != self.route.handoff_contract.kind:
            raise ValueError("validation evidence handoff kind does not match route")
        if self.handoff.validator_authority_ids != self.route.official_validators:
            raise ValueError("handoff validator authorities do not match route")
        if len(self.handoff.evidence_claim_ids) != self.claim_count:
            raise ValueError("handoff claim identifiers do not match report count")
        expected_handoff_branches = (
            self.branch_count
            if self.stage == ValidationEvidenceStage.GENERATION_PRIOR
            else 0
        )
        if len(self.handoff.evidence_branch_ids) != expected_handoff_branches:
            raise ValueError("handoff branch identifiers do not match report contract")
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
        if self.status in {
            ValidationEvidenceStatus.COMPLETED,
            ValidationEvidenceStatus.PARTIAL,
        } and (self.bundle_id is None or self.record_count == 0):
            raise ValueError("completed/partial evidence requires source-grounded records")
        if self.status == ValidationEvidenceStatus.PARTIAL and not self.reason:
            raise ValueError("partial validation evidence requires a reason")
        source_values = [LiteratureSource(str(item.source)) for item in self.source_statuses]
        if len(source_values) != len(set(source_values)):
            raise ValueError("validation evidence source statuses must be unique")
        if not set(source_values).issubset(set(self.route.literature_sources)):
            raise ValueError("validation evidence report contains a non-allowlisted source")
        if self.mcp_contract_status == McpContractVerificationStatus.VERIFIED:
            if not self.mcp_tool_name:
                raise ValueError("verified MCP contract requires the resolved tool name")
        elif self.mcp_tool_name is not None:
            raise ValueError("unverified MCP contract cannot cite a resolved tool name")
        return self


@dataclass(frozen=True, slots=True)
class ValidationEvidenceRun:
    report: ValidationEvidenceReport
    bundle: RagEvidenceBundle | None
    report_path: Path


def _validator(
    authority_id: str,
    capability: ValidationCapability,
    implementation: str,
    input_schema: str,
    output_schema: str,
    availability: ValidatorAvailability = ValidatorAvailability.IMPLEMENTED,
) -> ValidatorAuthority:
    return ValidatorAuthority(
        authority_id=authority_id,
        capability=capability,
        implementation=implementation,
        input_schema=input_schema,
        output_schema=output_schema,
        availability=availability,
    )


def _mcp_contract(
    stage: ValidationEvidenceStage,
    capability: McpEvidenceCapability,
) -> McpEvidenceContract:
    return McpEvidenceContract(
        capability=capability,
        tool_environment_variable=f"MATERIAL_RAG_MCP_TOOL_{stage.value.upper()}",
        accepted_arguments=["query", "max_results", "from_date", "to_date"],
        required_record_fields=["source_id", "title"],
    )


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
        validator_authorities=[
            _validator(
                "mattergen-supported-condition-allowlist",
                ValidationCapability.GENERATION_CONDITION_GUARD,
                "discovery_os.sidecars.generators.MatterGenRuntime",
                "FusionGenerationRequest",
                "FusionGenerationResponse",
                ValidatorAvailability.SIDECAR_REQUIRED,
            ),
            _validator(
                "evidence-driven-fusion-controller",
                ValidationCapability.EVIDENCE_SEARCH_CONTROL,
                "discovery_os.evidence_fusion.EvidenceDrivenFusionBackend",
                "FusionRevisionRequest",
                "FusionRevisionProposal",
            ),
        ],
        mcp_contract=_mcp_contract(
            ValidationEvidenceStage.GENERATION_PRIOR,
            McpEvidenceCapability.GENERATION_PRIOR_SEARCH,
        ),
        handoff_contract=ValidationHandoffContract(
            kind=ValidationHandoffKind.GENERATION_CONSTRAINT_CONTEXT,
            consumer="discovery_os.fusion_schemas.FusionDecisionContext",
            payload_schema="FusionDecisionContext",
            can_steer_generation=True,
        ),
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
        validator_authorities=[
            _validator(
                "pymatgen-structure-matcher",
                ValidationCapability.CRYSTAL_IDENTITY,
                "discovery_os.crystal_identity.group_crystal_structures",
                "Candidate",
                "CrystalGroupingResult",
            ),
            _validator(
                "materials-project-find-structure",
                ValidationCapability.EXTERNAL_STRUCTURE_NOVELTY,
                "discovery_os.novelty.MaterialsProjectStructureLookup",
                "Candidate",
                "ExternalNoveltyOutcome",
                ValidatorAvailability.CREDENTIAL_REQUIRED,
            ),
        ],
        mcp_contract=_mcp_contract(
            ValidationEvidenceStage.IDENTITY_NOVELTY,
            McpEvidenceCapability.CRYSTAL_REFERENCE_SEARCH,
        ),
        handoff_contract=ValidationHandoffContract(
            kind=ValidationHandoffKind.IDENTITY_NOVELTY_CONTEXT,
            consumer="discovery_os.novelty.StagedNoveltyAssessor",
            payload_schema="ScientificNoveltyAssessment",
        ),
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
        validator_authorities=[
            _validator(
                "mattersim-sidecar",
                ValidationCapability.PERIODIC_MLIP_INFERENCE,
                "MatterSim /v1/features sidecar",
                "ExpertFeatureRequest",
                "ExpertFeaturePayload",
                ValidatorAvailability.SIDECAR_REQUIRED,
            ),
            _validator(
                "chgnet-sidecar",
                ValidationCapability.PERIODIC_MLIP_INFERENCE,
                "CHGNet /v1/features sidecar",
                "ExpertFeatureRequest",
                "ExpertFeaturePayload",
                ValidatorAvailability.SIDECAR_REQUIRED,
            ),
            _validator(
                "cross-model-unit-normalized-disagreement",
                ValidationCapability.CROSS_MODEL_DISAGREEMENT,
                "discovery_os.materials_screening.classify_model_disagreement",
                "MLIPScreeningPrediction",
                "ModelDisagreement",
            ),
        ],
        mcp_contract=_mcp_contract(
            ValidationEvidenceStage.MLIP_DISAGREEMENT,
            McpEvidenceCapability.MLIP_LIMIT_SEARCH,
        ),
        handoff_contract=ValidationHandoffContract(
            kind=ValidationHandoffKind.MLIP_DISAGREEMENT_CONTEXT,
            consumer="discovery_os.materials_screening.classify_model_disagreement",
            payload_schema="ModelDisagreement",
        ),
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
        validator_authorities=[
            _validator(
                "ase-periodic-optimizer",
                ValidationCapability.RELAXED_GEOMETRY_GATE,
                "discovery_os.crystal_identity.validate_crystal_geometry",
                "PeriodicRelaxationPayload",
                "CrystalGeometryReport",
            ),
            _validator(
                "mattersim-relaxation",
                ValidationCapability.PERIODIC_RELAXATION,
                "MatterSim /v1/relax sidecar",
                "PeriodicRelaxationRequest",
                "PeriodicRelaxationPayload",
                ValidatorAvailability.SIDECAR_REQUIRED,
            ),
            _validator(
                "chgnet-relaxation",
                ValidationCapability.PERIODIC_RELAXATION,
                "CHGNet /v1/relax sidecar",
                "PeriodicRelaxationRequest",
                "PeriodicRelaxationPayload",
                ValidatorAvailability.SIDECAR_REQUIRED,
            ),
        ],
        mcp_contract=_mcp_contract(
            ValidationEvidenceStage.RELAXATION_VALIDATION,
            McpEvidenceCapability.RELAXATION_INSTABILITY_SEARCH,
        ),
        handoff_contract=ValidationHandoffContract(
            kind=ValidationHandoffKind.RELAXATION_GATE_CONTEXT,
            consumer="discovery_os.relaxation.PeriodicRelaxationResult",
            payload_schema="PeriodicRelaxationPayload",
        ),
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
        validator_authorities=[
            _validator(
                "periodic-dft-backend-contract",
                ValidationCapability.PERIODIC_DFT_PREPARATION,
                "discovery_os.dft_handoff.PortablePeriodicDFTInputBackend",
                "Candidate",
                "DFTInputHandoffReport",
            ),
            _validator(
                "external-pseudopotential-review",
                ValidationCapability.DFT_METHOD_REVIEW,
                "user-selected periodic DFT backend",
                "DFTInputManifest",
                "PeriodicDFTCalculationResult",
                ValidatorAvailability.EXTERNAL_REQUIRED,
            ),
            _validator(
                "reference-phase-convergence-review",
                ValidationCapability.DFT_METHOD_REVIEW,
                "user-selected periodic DFT backend",
                "DFTInputManifest",
                "PeriodicDFTCalculationResult",
                ValidatorAvailability.EXTERNAL_REQUIRED,
            ),
        ],
        mcp_contract=_mcp_contract(
            ValidationEvidenceStage.DFT_HANDOFF,
            McpEvidenceCapability.DFT_METHOD_SEARCH,
        ),
        handoff_contract=ValidationHandoffContract(
            kind=ValidationHandoffKind.DFT_PREPARATION_CONTEXT,
            consumer="discovery_os.dft_handoff.PeriodicDFTBackend",
            payload_schema="DFTInputHandoffReport",
        ),
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
        pipelines_by_stage: Mapping[
            ValidationEvidenceStage, LiteratureRagPipeline | None
        ]
        | None = None,
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
        self.pipelines_by_stage = dict(pipelines_by_stage or {})
        unknown_stages = set(self.pipelines_by_stage) - set(ValidationEvidenceStage)
        if unknown_stages:
            raise ValueError(f"unknown validation evidence stages: {sorted(unknown_stages)}")
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
        pipeline = self.pipelines_by_stage.get(request.stage, self.pipeline)
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
                mcp_contract_status=McpContractVerificationStatus.NOT_CHECKED,
                mcp_tool_name=None,
                reason="stage_evidence_disabled",
            )
            return self._persist(report, None)
        if pipeline is None:
            report = self._report(
                request=request,
                route=route,
                status=ValidationEvidenceStatus.UNKNOWN,
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                mcp_contract_status=McpContractVerificationStatus.NOT_CONFIGURED,
                mcp_tool_name=None,
                reason="literature_rag_pipeline_not_configured",
            )
            return self._persist(report, None)
        mcp_status, mcp_tool_name, mcp_warning = _verify_mcp_contract(
            pipeline,
            route,
        )
        selected_sources = list(route.literature_sources)
        if mcp_status in {
            McpContractVerificationStatus.FAILED,
            McpContractVerificationStatus.NOT_VERIFIABLE,
        }:
            selected_sources = [
                source for source in selected_sources if source != LiteratureSource.MCP
            ]
        try:
            bundle = pipeline.run(
                prompt,
                goal=goal,
                sources=[
                    LiteratureSource(str(source))
                    for source in selected_sources
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
                mcp_contract_status=mcp_status,
                mcp_tool_name=mcp_tool_name,
                reason=f"stage_evidence_retrieval_failed:{type(exc).__name__}",
            )
            return self._persist(report, None)

        contract_violation = _bundle_contract_violation(
            bundle,
            route=route,
            selected_sources=selected_sources,
        )
        if contract_violation is not None:
            report = self._report(
                request=request,
                route=route,
                status=ValidationEvidenceStatus.UNKNOWN,
                request_hash=request_hash,
                prompt_hash=prompt_hash,
                mcp_contract_status=mcp_status,
                mcp_tool_name=mcp_tool_name,
                reason=f"stage_route_contract_violation:{contract_violation}",
                extra_warnings=[mcp_warning] if mcp_warning else [],
            )
            return self._persist(report, None)

        degraded = any(
            item.status in {
                SourceRunStatus.PARTIAL,
                SourceRunStatus.SKIPPED,
                SourceRunStatus.FAILED,
            }
            for item in bundle.source_statuses
        ) or mcp_status in {
            McpContractVerificationStatus.FAILED,
            McpContractVerificationStatus.NOT_CONFIGURED,
            McpContractVerificationStatus.NOT_VERIFIABLE,
        }
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
            mcp_contract_status=mcp_status,
            mcp_tool_name=mcp_tool_name,
            bundle=bundle,
            bundle_relative_path=bundle_relative.as_posix(),
            reason=reason,
            extra_warnings=[mcp_warning] if mcp_warning else [],
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
        mcp_contract_status: McpContractVerificationStatus,
        mcp_tool_name: str | None,
        bundle: RagEvidenceBundle | None = None,
        bundle_relative_path: str | None = None,
        reason: str | None = None,
        extra_warnings: Sequence[str] = (),
    ) -> ValidationEvidenceReport:
        payload = {
            "stage": request.stage,
            "status": status,
            "request_hash": request_hash,
            "prompt_hash": prompt_hash,
            "bundle_id": bundle.bundle_id if bundle else None,
        }
        report_id = f"VREPORT-{stable_hash(payload)[:24]}"
        evidence_available = bool(
            bundle
            and bundle.records
            and status
            in {ValidationEvidenceStatus.COMPLETED, ValidationEvidenceStatus.PARTIAL}
        )
        handoff_payload = {
            "report_id": report_id,
            "stage": request.stage,
            "kind": route.handoff_contract.kind,
            "bundle_id": bundle.bundle_id if bundle else None,
            "candidate_refs": request.candidate_refs,
            "composition_keys": request.composition_keys,
        }
        handoff = ValidationEvidenceHandoff(
            handoff_id=f"VHANDOFF-{stable_hash(handoff_payload)[:24]}",
            report_id=report_id,
            stage=request.stage,
            kind=route.handoff_contract.kind,
            consumer=route.handoff_contract.consumer,
            payload_schema=route.handoff_contract.payload_schema,
            evidence_status=status,
            candidate_refs=list(request.candidate_refs),
            composition_keys=list(request.composition_keys),
            bundle_id=bundle.bundle_id if bundle else None,
            evidence_claim_ids=(
                [item.claim_id for item in bundle.claims] if bundle else []
            ),
            evidence_branch_ids=(
                [item.branch_id for item in bundle.branches]
                if bundle and request.stage == ValidationEvidenceStage.GENERATION_PRIOR
                else []
            ),
            validator_authority_ids=list(route.official_validators),
            evidence_available=evidence_available,
            can_steer_generation=(
                route.handoff_contract.can_steer_generation
                and evidence_available
                and bool(bundle and bundle.branches)
            ),
        )
        return ValidationEvidenceReport(
            report_id=report_id,
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
            mcp_contract_status=mcp_contract_status,
            mcp_tool_name=mcp_tool_name,
            handoff=handoff,
            reason=reason,
            warnings=[
                *(list(bundle.warnings) if bundle else []),
                *[item for item in extra_warnings if item],
            ],
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


def validation_evidence_route(
    stage: ValidationEvidenceStage,
) -> ValidationEvidenceRoute:
    """Return an immutable-by-copy view of the closed route registry."""

    route = _ROUTES[ValidationEvidenceStage(str(stage))]
    return ValidationEvidenceRoute.model_validate_json(
        route.model_dump_json(),
        strict=True,
    )


def _verify_mcp_contract(
    pipeline: LiteratureRagPipeline,
    route: ValidationEvidenceRoute,
) -> tuple[McpContractVerificationStatus, str | None, str | None]:
    retriever = getattr(pipeline, "retriever", None)
    if retriever is None:
        return (
            McpContractVerificationStatus.NOT_VERIFIABLE,
            None,
            "MCP source was omitted because the custom RAG pipeline exposes no retriever contract",
        )
    client = getattr(retriever, "mcp_client", None)
    tool_name = getattr(retriever, "mcp_tool", None)
    if client is None and not tool_name:
        return (
            McpContractVerificationStatus.NOT_CONFIGURED,
            None,
            "MCP source is optional and was not configured for this stage",
        )
    if client is None or not isinstance(tool_name, str) or not tool_name.strip():
        return (
            McpContractVerificationStatus.FAILED,
            None,
            "MCP source was omitted because its client/tool configuration is incomplete",
        )
    try:
        client.require_tool_contract(
            tool_name,
            accepted_arguments=tuple(route.mcp_contract.accepted_arguments),
            result_collection=route.mcp_contract.result_collection,
        )
    except (McpClientError, TypeError, ValueError) as exc:
        return (
            McpContractVerificationStatus.FAILED,
            None,
            f"MCP source was omitted because its tool contract failed: {type(exc).__name__}",
        )
    return McpContractVerificationStatus.VERIFIED, tool_name, None


def _bundle_contract_violation(
    bundle: RagEvidenceBundle,
    *,
    route: ValidationEvidenceRoute,
    selected_sources: Sequence[LiteratureSource],
) -> str | None:
    """Reject provider/tool expansion outside the selected stage route."""

    allowlisted = {LiteratureSource(str(item)) for item in route.literature_sources}
    selected = {LiteratureSource(str(item)) for item in selected_sources}
    if not selected.issubset(allowlisted):
        return "selected_source_not_allowlisted"
    query_sources = {
        LiteratureSource(str(item.source)) for item in bundle.search_plan.queries
    }
    if not query_sources.issubset(selected):
        return "search_plan_used_nonselected_source"
    status_sources = {
        LiteratureSource(str(item.source)) for item in bundle.source_statuses
    }
    if len(status_sources) != len(bundle.source_statuses):
        return "duplicate_source_status"
    if not status_sources.issubset(selected):
        return "source_status_used_nonselected_source"
    if query_sources != status_sources:
        return "queried_source_missing_retrieval_status"
    allowed_source_keys = {item.value for item in selected}
    queried_source_keys = {item.value for item in query_sources}
    for record in bundle.records:
        if not set(record.source_ids).issubset(allowed_source_keys):
            return "record_used_nonselected_source"
        if not set(record.source_ids).issubset(queried_source_keys):
            return "record_source_was_not_queried"
    return None


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
    if (
        not run.report.handoff.can_steer_generation
        or run.bundle is None
        or not run.bundle.branches
    ):
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
    if (
        not run.report.handoff.can_steer_generation
        or run.bundle is None
        or not run.bundle.branches
    ):
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
    require_model = (
        str(values.get("VALIDATION_EVIDENCE_REQUIRE_RAG_MODEL", "")).strip()
        == "1"
    )
    pipeline: LiteratureRagPipeline | None = None
    pipelines_by_stage: dict[
        ValidationEvidenceStage, LiteratureRagPipeline | None
    ] = {}
    if effective_enabled:
        stage_tool_values = {
            stage: str(
                values.get(_ROUTES[stage].mcp_contract.tool_environment_variable, "")
            ).strip()
            for stage in ValidationEvidenceStage
        }
        if any(stage_tool_values.values()):
            mcp_url = str(values.get("MATERIAL_RAG_MCP_URL", "")).strip()
            if not mcp_url:
                raise ValueError(
                    "stage-specific MATERIAL_RAG_MCP_TOOL_* configuration requires "
                    "MATERIAL_RAG_MCP_URL"
                )
            fallback_tool = str(values.get("MATERIAL_RAG_MCP_TOOL", "")).strip()
            for stage in ValidationEvidenceStage:
                stage_values = dict(values)
                selected_tool = stage_tool_values[stage] or fallback_tool
                if selected_tool:
                    stage_values["MATERIAL_RAG_MCP_TOOL"] = selected_tool
                else:
                    for key in (
                        "MATERIAL_RAG_MCP_URL",
                        "MATERIAL_RAG_MCP_TOOL",
                        "MATERIAL_RAG_MCP_TOKEN",
                        "MATERIAL_RAG_MCP_TIMEOUT_SECONDS",
                        "MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP",
                    ):
                        stage_values.pop(key, None)
                pipelines_by_stage[stage] = build_literature_rag_from_environment(
                    environ=stage_values,
                    require_model=require_model,
                )
        else:
            pipeline = build_literature_rag_from_environment(
                environ=values,
                require_model=require_model,
            )
    return ValidationEvidenceRouter(
        pipeline,
        artifact_root=artifact_root,
        pipelines_by_stage=pipelines_by_stage,
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
    "ValidationEvidenceHandoff",
    "ValidationEvidenceRequest",
    "ValidationEvidenceRoute",
    "ValidationEvidenceRouter",
    "ValidationEvidenceRun",
    "ValidationEvidenceStage",
    "ValidationEvidenceStatus",
    "ValidationCapability",
    "ValidationFailurePolicy",
    "ValidationHandoffContract",
    "ValidationHandoffKind",
    "ValidatorAuthority",
    "ValidatorAvailability",
    "McpContractVerificationStatus",
    "McpEvidenceCapability",
    "McpEvidenceContract",
    "build_validation_evidence_prompt",
    "build_validation_evidence_router_from_environment",
    "fusion_decision_context_from_stage_evidence",
    "fusion_decision_contexts_from_stage_evidence",
    "validation_evidence_route",
]
