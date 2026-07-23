"""Code-owned routing for field-specific crystalline-material discovery.

Literature retrieval, MCP database access, numerical calculation, and
experimental confirmation have different scientific authority.  The profiles
in this module keep those roles separate and make missing field-specific
calculations explicit instead of converting them into optimistic scores.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Protocol

from pydantic import Field, model_validator

from .schemas import (
    CandidateType,
    DiscoveryDomain,
    Identifier,
    JsonValue,
    MaterialField,
    NonEmptyText,
    Probability,
    StrictSchema,
)
from .hashing import stable_hash


MaterialEvidenceStage = Literal[
    "generation_prior",
    "identity_novelty",
    "mlip_disagreement",
    "relaxation_validation",
    "dft_handoff",
]
ValidatorAvailability = Literal[
    "implemented",
    "sidecar_required",
    "credential_required",
    "external_required",
]

MATERIAL_EVIDENCE_STAGES: tuple[MaterialEvidenceStage, ...] = (
    "generation_prior",
    "identity_novelty",
    "mlip_disagreement",
    "relaxation_validation",
    "dft_handoff",
)


class MaterialPropertyRequirement(StrictSchema):
    property_name: Identifier
    unit: NonEmptyText
    scientific_role: NonEmptyText
    required_context: list[Identifier] = Field(default_factory=list)
    preferred_calculations: list[NonEmptyText] = Field(min_length=1)
    experimental_confirmation: list[NonEmptyText] = Field(min_length=1)
    required_for_field_claim: bool = True
    missing_result_policy: Literal["unknown-not-pass"] = "unknown-not-pass"


class DomainValidatorSpec(StrictSchema):
    validator_id: Identifier
    stage: MaterialEvidenceStage
    role: NonEmptyText
    authority: NonEmptyText
    properties: list[Identifier] = Field(default_factory=list)
    availability: ValidatorAvailability
    evidence_kind: Literal[
        "structural",
        "database",
        "machine_learning",
        "physics_simulation",
        "experimental",
        "control",
    ]
    can_create_property_scores: bool = False
    result_if_not_executed: Literal["unknown"] = "unknown"

    @model_validator(mode="after")
    def _score_authority_is_explicit(self) -> "DomainValidatorSpec":
        if self.can_create_property_scores:
            if not self.properties:
                raise ValueError(
                    "a score-producing domain validator must declare its properties"
                )
            if self.evidence_kind not in {
                "machine_learning",
                "physics_simulation",
                "experimental",
            }:
                raise ValueError(
                    "structural, database, and control validators cannot create "
                    "field-property scores"
                )
            if self.stage == "generation_prior":
                raise ValueError(
                    "generation-prior validators cannot create field-property scores"
                )
        return self


class MaterialStageRoute(StrictSchema):
    material_field: MaterialField
    profile_id: Identifier
    stage: MaterialEvidenceStage
    rag_questions: list[NonEmptyText] = Field(min_length=1)
    mcp_capabilities: list[Identifier] = Field(default_factory=list)
    validators: list[DomainValidatorSpec] = Field(min_length=1)
    can_steer_generation: bool = False
    literature_role: Literal["context-not-property-authority"] = (
        "context-not-property-authority"
    )
    mcp_selection_policy: Literal["code-owned-capability-admin-configured-tool"] = (
        "code-owned-capability-admin-configured-tool"
    )
    unknown_is_pass: Literal[False] = False
    property_score_created_by_route: Literal[False] = False

    @model_validator(mode="after")
    def _route_is_fail_closed(self) -> "MaterialStageRoute":
        if self.can_steer_generation != (self.stage == "generation_prior"):
            raise ValueError("only generation_prior may steer candidate generation")
        if any(item.stage != self.stage for item in self.validators):
            raise ValueError("domain validator stage does not match its stage route")
        ids = [item.validator_id for item in self.validators]
        if len(ids) != len(set(ids)):
            raise ValueError("domain validator identifiers must be unique within a stage")
        if len(self.rag_questions) != len(set(self.rag_questions)):
            raise ValueError("RAG questions must be unique within a stage")
        if len(self.mcp_capabilities) != len(set(self.mcp_capabilities)):
            raise ValueError("MCP capabilities must be unique within a stage")
        return self


class MaterialFieldProfile(StrictSchema):
    profile_id: Identifier
    profile_version: Literal["1.0"] = "1.0"
    material_field: MaterialField
    name: NonEmptyText
    discovery_domain: DiscoveryDomain
    candidate_types: list[CandidateType] = Field(min_length=1)
    application_subtypes: list[Identifier] = Field(min_length=1)
    scope: NonEmptyText
    required_problem_context: list[Identifier] = Field(default_factory=list)
    properties: list[MaterialPropertyRequirement] = Field(min_length=1)
    stage_routes: list[MaterialStageRoute] = Field(
        min_length=len(MATERIAL_EVIDENCE_STAGES),
        max_length=len(MATERIAL_EVIDENCE_STAGES),
    )
    research_reference_ids: list[NonEmptyText] = Field(min_length=1)
    field_claim_boundary: NonEmptyText
    t4_scope: Literal[
        "generic-crystal-screening-only",
        "generic-plus-field-triage",
    ] = "generic-plus-field-triage"

    @model_validator(mode="after")
    def _profile_is_complete(self) -> "MaterialFieldProfile":
        stages = [item.stage for item in self.stage_routes]
        if tuple(stages) != MATERIAL_EVIDENCE_STAGES:
            raise ValueError("field profile must contain the five ordered evidence stages")
        if any(item.material_field != self.material_field for item in self.stage_routes):
            raise ValueError("stage route material field does not match profile")
        if any(item.profile_id != self.profile_id for item in self.stage_routes):
            raise ValueError("stage route profile id does not match profile")
        names = [item.property_name for item in self.properties]
        if len(names) != len(set(names)):
            raise ValueError("field property names must be unique")
        if len(self.application_subtypes) != len(set(self.application_subtypes)):
            raise ValueError("field application subtypes must be unique")
        if len(self.required_problem_context) != len(
            set(self.required_problem_context)
        ):
            raise ValueError("required problem-context fields must be unique")
        if len(self.research_reference_ids) != len(set(self.research_reference_ids)):
            raise ValueError("field research-reference identifiers must be unique")
        score_authorities = {
            property_name
            for route in self.stage_routes
            for validator in route.validators
            if validator.can_create_property_scores
            for property_name in validator.properties
        }
        missing_score_authorities = [
            item.property_name
            for item in self.properties
            if item.required_for_field_claim
            and item.property_name not in score_authorities
        ]
        if missing_score_authorities:
            raise ValueError(
                "required field properties need a named score-producing validator: "
                + ", ".join(missing_score_authorities)
            )
        return self


class MaterialFieldResolution(StrictSchema):
    requested: NonEmptyText
    selected_field: MaterialField
    profile_id: Identifier
    selection_mode: Literal[
        "explicit",
        "auto-keyword",
        "auto-default",
        "auto-ambiguous",
        "auto-model",
        "auto-consensus",
        "auto-model-conflict",
    ]
    confidence: Literal["high", "medium", "low"]
    matched_terms: dict[str, list[str]] = Field(default_factory=dict)
    ambiguous_fields: list[MaterialField] = Field(default_factory=list)
    secondary_fields: list[MaterialField] = Field(default_factory=list)
    application_subtype: Identifier | None = None
    model_decision_id: Identifier | None = None
    model_confidence: Probability | None = None
    requires_operator_choice: bool = False
    reason: NonEmptyText


class MaterialFieldModelDecision(StrictSchema):
    """Untrusted structured proposal returned by the main reasoning model."""

    primary_field: MaterialField | None = None
    secondary_fields: list[MaterialField] = Field(default_factory=list, max_length=4)
    application_subtype: Identifier | None = None
    confidence: Probability
    evidence_spans: list[NonEmptyText] = Field(default_factory=list, max_length=12)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=1_000)
    decision_summary: NonEmptyText

    @model_validator(mode="after")
    def _decision_is_bounded(self) -> "MaterialFieldModelDecision":
        if len(self.secondary_fields) != len(set(self.secondary_fields)):
            raise ValueError("model secondary material fields must be unique")
        if self.primary_field in self.secondary_fields:
            raise ValueError("primary material field cannot also be secondary")
        if self.needs_clarification != bool(self.clarification_question):
            raise ValueError(
                "clarification_question must be present exactly when clarification is needed"
            )
        if self.primary_field is None and not self.needs_clarification:
            raise ValueError("a model decision requires a primary field or clarification")
        if self.primary_field is not None and not self.evidence_spans:
            raise ValueError("a model-selected field requires quoted input evidence spans")
        normalized_spans = [_normalize_evidence_text(span) for span in self.evidence_spans]
        if any(len(span) < 3 or len(span) > 500 for span in normalized_spans):
            raise ValueError(
                "model evidence spans must contain 3 to 500 normalized characters"
            )
        if len(normalized_spans) != len(set(normalized_spans)):
            raise ValueError("model evidence spans must be unique")
        if self.application_subtype and self.primary_field is None:
            raise ValueError("application subtype requires a primary field")
        return self


class MaterialFieldModelRun(StrictSchema):
    decision_id: Identifier
    model_id: Identifier
    model_version: Identifier
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    user_prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: MaterialFieldModelDecision
    evidence_spans_verified: Literal[True] = True
    endpoint_or_tool_selection_performed: Literal[False] = False


class JsonFieldClassificationModel(Protocol):
    model_id: str
    model_version: str

    def complete_json(self, *, operation: str, system: str, user: str) -> Any: ...


class MainModelMaterialFieldClassifier:
    """Ask the main model for a typed field hypothesis, then verify its evidence."""

    def __init__(self, model: JsonFieldClassificationModel) -> None:
        self.model = model

    def classify(
        self,
        prompt: str,
        *,
        chemical_system: str | None = None,
        problem_context: Mapping[str, JsonValue] | None = None,
    ) -> MaterialFieldModelRun:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("material field classification prompt cannot be empty")
        context = dict(problem_context or {})
        if _contains_sensitive_context_key(context):
            raise ValueError("material field classifier context cannot contain secrets")
        context_chemical_system = context.get("chemical_system")
        if (
            chemical_system
            and not _context_value_is_missing(context_chemical_system)
            and _normalize(str(context_chemical_system)) != _normalize(chemical_system)
        ):
            raise ValueError(
                "chemical_system conflicts with problem_context chemical_system"
            )
        profiles = {
            str(field): {
                "scope": profile.scope,
                "application_subtypes": profile.application_subtypes,
                "required_problem_context": profile.required_problem_context,
                "properties": [item.property_name for item in profile.properties],
                "claim_boundary": profile.field_claim_boundary,
            }
            for field, profile in MATERIAL_FIELD_PROFILES.items()
        }
        payload = self.model.complete_json(
            operation="classify-material-field",
            system=(
                "You classify crystalline-material discovery requests. Return JSON only. "
                "Choose the scientific application, not merely an element or property word. "
                "Distinguish electrode from solid electrolyte, semiconductor from photovoltaic "
                "absorber, bulk catalyst from surface/reaction claims, and conventional from "
                "unconventional superconductivity. Preserve genuinely secondary fields. Quote "
                "short exact spans from the supplied prompt/context as evidence. If the intended "
                "application or operating conditions are insufficient or several primary fields "
                "remain equally plausible, request clarification. Never select an API, MCP tool, "
                "calculation engine, endpoint, or pass/fail result."
            ),
            user=json.dumps(
                {
                    "prompt": prompt,
                    "chemical_system": chemical_system,
                    "problem_context": context,
                    "allowed_profiles": profiles,
                    "required_output": {
                        "primary_field": "allowed field or null",
                        "secondary_fields": ["allowed field"],
                        "application_subtype": "subtype of primary field or null",
                        "confidence": "number from 0 to 1",
                        "evidence_spans": [
                            "short exact quote from prompt, chemical system, or context"
                        ],
                        "needs_clarification": "boolean",
                        "clarification_question": "string or null",
                        "decision_summary": "short scientific routing summary",
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        try:
            decision = MaterialFieldModelDecision.model_validate_json(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                strict=True,
            )
        except Exception as exc:
            raise ValueError(
                "main model returned an invalid material-field decision"
            ) from exc
        if decision.primary_field is not None and decision.application_subtype:
            allowed_subtypes = MATERIAL_FIELD_PROFILES[
                MaterialField(str(decision.primary_field))
            ].application_subtypes
            if decision.application_subtype not in allowed_subtypes:
                raise ValueError(
                    "main model selected an application subtype outside the field profile"
                )
        corpus = _normalize_evidence_text(
            " ".join(
                [
                    prompt,
                    chemical_system or "",
                    json.dumps(
                        context,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        )
        invalid_spans = [
            span
            for span in decision.evidence_spans
            if _normalize_evidence_text(span) not in corpus
        ]
        if invalid_spans:
            raise ValueError(
                "main model field decision cited evidence not present in the input"
            )
        application_corpus = _normalize_evidence_text(
            " ".join(
                [
                    prompt,
                    json.dumps(
                        context,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ]
            )
        )
        if decision.primary_field is not None and not any(
            _normalize_evidence_text(span) in application_corpus
            for span in decision.evidence_spans
        ):
            raise ValueError(
                "main model field decision requires application evidence from the "
                "prompt or problem context; a chemical system alone is insufficient"
            )
        decision_payload = {
            "model_id": self.model.model_id,
            "model_version": self.model.model_version,
            "prompt": prompt,
            "chemical_system": chemical_system,
            "problem_context": context,
            "decision": decision,
        }
        return MaterialFieldModelRun(
            decision_id=f"MFDEC-{stable_hash(decision_payload)[:24]}",
            model_id=self.model.model_id,
            model_version=self.model.model_version,
            prompt_hash=stable_hash(
                {
                    "prompt": prompt,
                    "chemical_system": chemical_system,
                    "problem_context": context,
                }
            ),
            user_prompt_hash=stable_hash(prompt),
            decision=decision,
        )


class MaterialDomainPlan(StrictSchema):
    resolution: MaterialFieldResolution
    main_model_run: MaterialFieldModelRun | None = None
    profile: MaterialFieldProfile
    stages: list[MaterialStageRoute] = Field(
        min_length=len(MATERIAL_EVIDENCE_STAGES),
        max_length=len(MATERIAL_EVIDENCE_STAGES),
    )
    problem_context: dict[str, JsonValue] = Field(default_factory=dict)
    missing_required_context: list[Identifier] = Field(default_factory=list)
    field_route_ready: bool = False
    externally_reported_property_names: list[Identifier] = Field(default_factory=list)
    unexecuted_required_properties: list[Identifier] = Field(default_factory=list)
    scientific_status: Literal[
        "routing-plan-only-no-field-property-calculation"
    ] = "routing-plan-only-no-field-property-calculation"

    @model_validator(mode="after")
    def _plan_matches_profile(self) -> "MaterialDomainPlan":
        if _contains_sensitive_context_key(self.problem_context):
            raise ValueError("material domain problem context cannot contain secrets")
        if self.main_model_run is None:
            if self.resolution.model_decision_id is not None:
                raise ValueError("resolution cites a missing main-model decision")
        elif self.resolution.model_decision_id != self.main_model_run.decision_id:
            raise ValueError("resolution main-model decision id does not match plan")
        if self.resolution.selected_field != self.profile.material_field:
            raise ValueError("field resolution does not match selected profile")
        if self.stages != self.profile.stage_routes:
            raise ValueError("plan stages must be a defensive copy of profile stages")
        expected_context = [
            name
            for name in self.profile.required_problem_context
            if _context_value_is_missing(self.problem_context.get(name))
        ]
        if self.missing_required_context != expected_context:
            raise ValueError("missing required problem context must be explicit and ordered")
        if self.field_route_ready != (
            not self.resolution.requires_operator_choice and not expected_context
        ):
            raise ValueError("field_route_ready does not match routing prerequisites")
        expected = [
            item.property_name
            for item in self.profile.properties
            if item.required_for_field_claim
            and item.property_name not in self.externally_reported_property_names
        ]
        if self.unexecuted_required_properties != expected:
            raise ValueError("unexecuted required properties must be explicit and ordered")
        return self


class MaterialPropertyObservation(StrictSchema):
    """One property result from a named numerical or experimental authority."""

    observation_id: Identifier
    candidate_id: Identifier
    material_field: MaterialField
    property_name: Identifier
    validator_id: Identifier
    status: Literal["success", "failed", "unknown", "incomparable"]
    value: JsonValue = None
    unit: NonEmptyText
    conditions: dict[str, JsonValue] = Field(default_factory=dict)
    provenance_id: Identifier
    raw_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authority_kind: Literal["numerical_validator", "experimental_validator"]
    literature_or_mcp_derived: Literal[False] = False

    @model_validator(mode="after")
    def _successful_observation_has_a_value(self) -> "MaterialPropertyObservation":
        if _contains_sensitive_context_key(self.conditions) or (
            self.value is not None
            and _contains_sensitive_context_key(self.value)
        ):
            raise ValueError("material property observations cannot contain secrets")
        if self.status == "success" and self.value is None:
            raise ValueError("successful material property observation requires a value")
        if self.status != "success" and self.value is not None:
            raise ValueError("non-success property observation cannot expose a value")
        return self


class MaterialPropertyDecision(StrictSchema):
    property_name: Identifier
    status: Literal["available", "unknown", "incomparable", "conflicting"]
    accepted_observation_ids: list[Identifier] = Field(default_factory=list)
    rejected_observation_ids: list[Identifier] = Field(default_factory=list)
    accepted_conditions: dict[str, JsonValue] = Field(default_factory=dict)
    reason: NonEmptyText
    value_aggregation_performed: Literal[False] = False
    unknown_is_pass: Literal[False] = False

    @model_validator(mode="after")
    def _observation_sets_are_disjoint(self) -> "MaterialPropertyDecision":
        if len(self.accepted_observation_ids) != len(
            set(self.accepted_observation_ids)
        ):
            raise ValueError("accepted property observations must be unique")
        if len(self.rejected_observation_ids) != len(
            set(self.rejected_observation_ids)
        ):
            raise ValueError("rejected property observations must be unique")
        if set(self.accepted_observation_ids) & set(self.rejected_observation_ids):
            raise ValueError("a property observation cannot be accepted and rejected")
        if self.status in {"available", "conflicting"} and not (
            self.accepted_observation_ids
        ):
            raise ValueError("available/conflicting property decisions need observations")
        if self.status in {"unknown", "incomparable"} and self.accepted_observation_ids:
            raise ValueError("unknown/incomparable decisions cannot accept observations")
        return self


class MaterialFieldResultAssessment(StrictSchema):
    candidate_id: Identifier
    material_field: MaterialField
    profile_id: Identifier
    decisions: list[MaterialPropertyDecision] = Field(min_length=1)
    target_conditions: dict[str, JsonValue] = Field(default_factory=dict)
    missing_target_conditions: list[Identifier] = Field(default_factory=list)
    ready_for_field_computational_ranking: bool = False
    claim_level: Literal["computational-triage-only"] = "computational-triage-only"
    literature_or_mcp_property_substitution_performed: Literal[False] = False

    @model_validator(mode="after")
    def _readiness_matches_required_properties(self) -> "MaterialFieldResultAssessment":
        if _contains_sensitive_context_key(self.target_conditions):
            raise ValueError("material field target conditions cannot contain secrets")
        registry = globals().get("MATERIAL_FIELD_PROFILES")
        if not registry:
            raise ValueError("material field profile registry is unavailable")
        profile = registry[MaterialField(str(self.material_field))]
        if self.profile_id != profile.profile_id:
            raise ValueError("field assessment profile id does not match material field")
        expected_property_names = [
            item.property_name for item in profile.properties
        ]
        if [item.property_name for item in self.decisions] != expected_property_names:
            raise ValueError(
                "field assessment decisions must cover every profile property in order"
            )
        required_target_conditions = list(
            dict.fromkeys(
                context_name
                for requirement in profile.properties
                if requirement.required_for_field_claim
                for context_name in requirement.required_context
            )
        )
        expected_missing = [
            name
            for name in required_target_conditions
            if _context_value_is_missing(self.target_conditions.get(name))
        ]
        if self.missing_target_conditions != expected_missing:
            raise ValueError("missing target conditions must be explicit and ordered")
        for decision, requirement in zip(self.decisions, profile.properties):
            if decision.status in {"available", "conflicting"}:
                if set(decision.accepted_conditions) != set(
                    requirement.required_context
                ):
                    raise ValueError(
                        "accepted property conditions do not match the requirement"
                    )
                if any(
                    name in self.target_conditions
                    and _stable_json_value(decision.accepted_conditions[name])
                    != _stable_json_value(self.target_conditions[name])
                    for name in requirement.required_context
                ):
                    raise ValueError(
                        "accepted property conditions do not match target conditions"
                    )
            elif decision.accepted_conditions:
                raise ValueError(
                    "unavailable property decisions cannot expose accepted conditions"
                )
        expected = (
            not self.missing_target_conditions
            and all(item.status == "available" for item in self.decisions)
        )
        if self.ready_for_field_computational_ranking != expected:
            raise ValueError("field ranking readiness does not match property decisions")
        return self


def _property(
    name: str,
    unit: str,
    role: str,
    calculations: tuple[str, ...],
    experiments: tuple[str, ...],
    *,
    context: tuple[str, ...] = (),
) -> MaterialPropertyRequirement:
    return MaterialPropertyRequirement(
        property_name=name,
        unit=unit,
        scientific_role=role,
        required_context=list(context),
        preferred_calculations=list(calculations),
        experimental_confirmation=list(experiments),
    )


def _validator(
    validator_id: str,
    stage: MaterialEvidenceStage,
    role: str,
    authority: str,
    *,
    properties: tuple[str, ...] = (),
    availability: ValidatorAvailability,
    evidence_kind: Literal[
        "structural",
        "database",
        "machine_learning",
        "physics_simulation",
        "experimental",
        "control",
    ],
    scores: bool = False,
) -> DomainValidatorSpec:
    return DomainValidatorSpec(
        validator_id=validator_id,
        stage=stage,
        role=role,
        authority=authority,
        properties=list(properties),
        availability=availability,
        evidence_kind=evidence_kind,
        can_create_property_scores=scores,
    )


_COMMON_RAG: dict[MaterialEvidenceStage, tuple[str, ...]] = {
    "generation_prior": (
        "Which compositions, prototypes, synthesis windows, precursor choices, and failed syntheses are reported?",
        "Which operating conditions and confounders define the intended material function?",
    ),
    "identity_novelty": (
        "Which experimentally reported or computed structures match the composition and prototype?",
        "Which database identifiers, polymorphs, pressure phases, disorder, or partial occupancies can confound identity?",
    ),
    "mlip_disagreement": (
        "Where are the selected interatomic potentials out of domain for this chemistry, bonding, charge state, or pressure?",
        "Which published benchmarks expose energy, force, stress, or relaxation failure modes?",
    ),
    "relaxation_validation": (
        "Which reconstructions, decomposition pathways, soft modes, finite-temperature phases, or kinetic traps are reported?",
        "Which relaxation settings and convergence checks are required for this material class?",
    ),
    "dft_handoff": (
        "Which exchange-correlation treatment, pseudopotential, spin, charge, dispersion, relativistic, and convergence settings are validated?",
        "Which field-specific high-fidelity calculations and experimental controls are required before making the target claim?",
    ),
}

_COMMON_MCP: dict[MaterialEvidenceStage, tuple[str, ...]] = {
    "generation_prior": (
        "scholarly-materials-search",
        "synthesis-procedure-and-negative-result-search",
    ),
    "identity_novelty": (
        "optimade-federated-structure-search",
        "materials-project-structure-search",
        "cod-crystallography-search",
    ),
    "mlip_disagreement": (
        "mlip-model-card-and-benchmark-search",
        "openkim-test-search",
    ),
    "relaxation_validation": (
        "nomad-or-aiida-provenance-search",
        "phase-and-phonon-reference-search",
    ),
    "dft_handoff": (
        "validated-method-and-convergence-search",
        "nomad-or-aiida-provenance-search",
    ),
}


def _common_validators(
    stage: MaterialEvidenceStage,
    *,
    dft_validators: tuple[DomainValidatorSpec, ...],
) -> tuple[DomainValidatorSpec, ...]:
    if stage == "generation_prior":
        return (
            _validator(
                "mattergen-supported-condition-guard",
                stage,
                "Permit only generator conditions actually supported by the bound MatterGen snapshot.",
                "discovery_os.sidecars.generators.MatterGenRuntime",
                availability="sidecar_required",
                evidence_kind="control",
            ),
        )
    if stage == "identity_novelty":
        return (
            _validator(
                "pymatgen-structure-matcher",
                stage,
                "Tolerance-aware periodic structure identity and duplicate grouping.",
                "discovery_os.crystal_identity.group_crystal_structures",
                availability="implemented",
                evidence_kind="structural",
            ),
            _validator(
                "materials-project-and-optimade-lookup",
                stage,
                "External structure-reference lookup; absence remains unknown rather than proof of novelty.",
                "Materials Project API plus administrator-configured OPTIMADE MCP",
                availability="credential_required",
                evidence_kind="database",
            ),
        )
    if stage == "mlip_disagreement":
        return (
            _validator(
                "mattersim-chgnet-common-geometry-panel",
                stage,
                "Independent energy, force, and stress screening on the same geometry.",
                "MatterSim 5M and CHGNet 0.3.0 sidecars",
                properties=("energy", "forces", "stress"),
                availability="sidecar_required",
                evidence_kind="machine_learning",
                scores=True,
            ),
        )
    if stage == "relaxation_validation":
        return (
            _validator(
                "independent-periodic-relaxation-panel",
                stage,
                "Relax with both MLIPs and check geometry, convergence, and model disagreement.",
                "MatterSim and CHGNet /v1/relax sidecars",
                properties=("relaxed_energy", "max_force", "stress"),
                availability="sidecar_required",
                evidence_kind="machine_learning",
                scores=True,
            ),
        )
    return dft_validators


def _profile(
    *,
    field: MaterialField,
    name: str,
    domain: DiscoveryDomain,
    candidate_types: tuple[CandidateType, ...],
    application_subtypes: tuple[str, ...],
    scope: str,
    context: tuple[str, ...],
    properties: tuple[MaterialPropertyRequirement, ...],
    domain_focus: dict[MaterialEvidenceStage, tuple[str, ...]],
    domain_mcp: dict[MaterialEvidenceStage, tuple[str, ...]],
    dft_validators: tuple[DomainValidatorSpec, ...],
    references: tuple[str, ...],
    boundary: str,
) -> MaterialFieldProfile:
    profile_id = f"{field.value}-workflow-v1"
    routes: list[MaterialStageRoute] = []
    for stage in MATERIAL_EVIDENCE_STAGES:
        validators = _common_validators(stage, dft_validators=dft_validators)
        routes.append(
            MaterialStageRoute(
                material_field=field,
                profile_id=profile_id,
                stage=stage,
                rag_questions=[
                    *_COMMON_RAG[stage],
                    *domain_focus.get(stage, ()),
                ],
                mcp_capabilities=[
                    *_COMMON_MCP[stage],
                    *domain_mcp.get(stage, ()),
                ],
                validators=list(validators),
                can_steer_generation=(stage == "generation_prior"),
            )
        )
    return MaterialFieldProfile(
        profile_id=profile_id,
        material_field=field,
        name=name,
        discovery_domain=domain,
        candidate_types=list(candidate_types),
        application_subtypes=list(application_subtypes),
        scope=scope,
        required_problem_context=list(context),
        properties=list(properties),
        stage_routes=routes,
        research_reference_ids=list(references),
        field_claim_boundary=boundary,
    )


def _dft(
    validator_id: str,
    role: str,
    authority: str,
    properties: tuple[str, ...],
) -> DomainValidatorSpec:
    return _validator(
        validator_id,
        "dft_handoff",
        role,
        authority,
        properties=properties,
        availability="external_required",
        evidence_kind="physics_simulation",
        scores=True,
    )


GENERAL_INORGANIC_PROFILE = _profile(
    field=MaterialField.GENERAL_INORGANIC,
    name="General inorganic crystal",
    domain=DiscoveryDomain.INORGANIC_MATERIALS,
    candidate_types=(CandidateType.CRYSTAL,),
    application_subtypes=("bulk_crystal",),
    scope="Periodic inorganic crystals screened for identity, stability, dynamics, and a declared target property.",
    context=("chemical_system", "target_property", "temperature", "pressure"),
    properties=(
        _property(
            "energy_above_hull",
            "eV/atom",
            "Thermodynamic competition against reference phases.",
            ("Converged DFT formation energies and a compatible phase diagram.",),
            ("Phase-pure synthesis with quantitative phase analysis.",),
            context=("pressure", "temperature"),
        ),
        _property(
            "minimum_phonon_frequency",
            "THz",
            "Minimum signed harmonic phonon frequency of the proposed phase.",
            ("DFPT or finite-displacement phonons with convergence and non-analytical corrections when relevant.",),
            ("Temperature-dependent diffraction or vibrational spectroscopy.",),
            context=("pressure", "temperature"),
        ),
    ),
    domain_focus={
        "dft_handoff": (
            "Establish a compatible reference-phase set before interpreting energy above hull.",
            "Require phonons or an explicitly justified finite-temperature stability calculation.",
        ),
    },
    domain_mcp={
        "dft_handoff": ("materials-project-phase-diagram-search",),
    },
    dft_validators=(
        _dft(
            "reference-phase-dft-and-phase-diagram",
            "Recompute candidate and competing phases with one compatible method.",
            "AiiDA/Common Workflows with Quantum ESPRESSO, VASP, ABINIT, or equivalent",
            ("energy_above_hull",),
        ),
        _dft(
            "phonon-stability-workflow",
            "Check harmonic dynamics and document treatment of soft modes.",
            "Phonopy, phono3py, or DFPT",
            ("minimum_phonon_frequency",),
        ),
    ),
    references=(
        "doi:10.1038/s41586-023-06735-9",
        "doi:10.1038/s41586-023-06734-w",
        "materialsproject:materials-methodology",
        "aiida:common-workflows-and-provenance",
    ),
    boundary="MatterGen plus two universal MLIPs is candidate triage, not proof of thermodynamic, dynamic, synthetic, or functional validity.",
)


BATTERY_ELECTRODE_PROFILE = _profile(
    field=MaterialField.BATTERY_ELECTRODE,
    name="Battery electrode",
    domain=DiscoveryDomain.BATTERIES,
    candidate_types=(CandidateType.CRYSTAL, CandidateType.BATTERY_MATERIAL),
    application_subtypes=("insertion_electrode", "conversion_electrode"),
    scope="Intercalation or conversion electrodes evaluated across state of charge, voltage, capacity, kinetics, cycling, and safety.",
    context=("working_ion", "charged_state", "discharged_state", "voltage_window", "temperature"),
    properties=(
        _property(
            "average_voltage",
            "V",
            "Cell voltage from compatible endpoint free energies.",
            ("DFT insertion/conversion reaction energies across enumerated states of charge.",),
            ("Galvanostatic or potentiostatic voltage profile in a documented cell.",),
            context=("working_ion", "reference_electrode", "state_of_charge"),
        ),
        _property(
            "specific_capacity",
            "mAh/g",
            "Reversible charge per active-material mass.",
            ("Stoichiometric redox-electron count with accessible endpoint structures.",),
            ("Measured reversible capacity, first-cycle efficiency, and retention.",),
            context=("working_ion", "cycling_protocol"),
        ),
        _property(
            "ion_migration_barrier",
            "eV",
            "Kinetic barrier for the mobile ion.",
            ("Converged NEB pathways and, where needed, finite-temperature AIMD.",),
            ("Rate capability, impedance, and diffusion measurement.",),
            context=("working_ion", "state_of_charge", "temperature"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Separate cathode, anode, insertion, conversion, and solid-electrolyte claims.",
            "Retrieve charged and discharged structures, working-ion stoichiometries, and failed cycling conditions.",
        ),
        "identity_novelty": (
            "Search every relevant state-of-charge structure rather than only the pristine host.",
        ),
        "dft_handoff": (
            "Enumerate endpoint and intermediate states before voltage or capacity calculation.",
            "Use NEB/AIMD for transport and retain metastability, volume change, and oxygen-loss risks.",
        ),
    },
    domain_mcp={
        "generation_prior": ("battery-literature-and-protocol-search",),
        "identity_novelty": ("materials-project-insertion-electrode-search",),
        "dft_handoff": ("battery-electrode-data-and-voltage-search",),
    },
    dft_validators=(
        _dft(
            "battery-reaction-phase-diagram",
            "Compute compatible endpoint/intermediate energies, voltage, capacity, and volume change.",
            "pymatgen battery analysis plus converged periodic DFT",
            ("average_voltage", "specific_capacity", "volume_change", "electrochemical_stability"),
        ),
        _dft(
            "working-ion-neb-or-aimd",
            "Validate mobile-ion kinetics at declared composition and temperature.",
            "CI-NEB and/or ab initio molecular dynamics",
            ("ion_migration_barrier", "ionic_diffusivity"),
        ),
    ),
    references=(
        "materialsproject:insertion-electrode-api",
        "pymatgen:analysis-battery",
        "doi:10.1038/s41586-023-06735-9",
    ),
    boundary="A stable host structure is not a battery electrode claim; voltage, accessible capacity, ion kinetics, charged-state stability, cycling, and safety remain separate authorities.",
)


SOLID_ELECTROLYTE_PROFILE = _profile(
    field=MaterialField.SOLID_ELECTROLYTE,
    name="Solid electrolyte and ionic conductor",
    domain=DiscoveryDomain.BATTERIES,
    candidate_types=(CandidateType.CRYSTAL, CandidateType.BATTERY_MATERIAL),
    application_subtypes=(
        "crystalline_solid_electrolyte",
        "non_battery_ionic_conductor",
        "solid_solid_interface",
    ),
    scope="Crystalline solid electrolytes evaluated for bulk and interfacial ion transport, stability windows, mechanics, and manufacturability.",
    context=("mobile_ion", "temperature", "electrode_pair", "defect_concentration", "microstructure"),
    properties=(
        _property(
            "ionic_conductivity",
            "S/cm",
            "Mobile-ion conductivity at a stated temperature and microstructure.",
            ("Long-timescale AIMD or validated MLIP MD with uncertainty and finite-size checks.",),
            ("Impedance spectroscopy separating bulk and grain-boundary response.",),
            context=("mobile_ion", "temperature", "microstructure"),
        ),
        _property(
            "migration_barrier",
            "eV",
            "Rate-limiting migration barrier and pathway connectivity.",
            ("Site/path enumeration followed by converged NEB.",),
            ("Temperature-dependent conductivity activation energy.",),
            context=("mobile_ion", "defect_concentration"),
        ),
        _property(
            "electrochemical_stability_window",
            "V",
            "Thermodynamic and kinetic compatibility with both electrodes.",
            ("Grand-potential phase diagrams plus explicit interface/reaction calculations.",),
            ("Cyclic voltammetry and interfacial aging in specified cells.",),
            context=("electrode_pair", "temperature"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve mobile-ion sublattice, disorder, dopants, processing atmosphere, densification, and grain-boundary evidence.",
        ),
        "mlip_disagreement": (
            "Check whether the MLIP training domain covers mobile-ion defects, high-temperature disorder, and interfaces.",
        ),
        "dft_handoff": (
            "Do not infer conductivity from a static structure; require connected paths plus NEB or finite-temperature dynamics.",
            "Evaluate electrode interfaces and decomposition products, not only intrinsic bulk stability.",
        ),
    },
    domain_mcp={
        "generation_prior": ("solid-electrolyte-protocol-search",),
        "dft_handoff": ("ionic-conductor-and-interface-data-search",),
    },
    dft_validators=(
        _dft(
            "mobile-ion-path-neb",
            "Enumerate connected sites and calculate migration barriers.",
            "pymatgen diffusion analysis plus CI-NEB",
            ("migration_barrier",),
        ),
        _dft(
            "finite-temperature-ion-transport",
            "Estimate diffusivity and conductivity with uncertainty and sampling diagnostics.",
            "AIMD or uncertainty-audited MLIP MD",
            ("ionic_conductivity", "ionic_diffusivity"),
        ),
        _dft(
            "electrode-interface-grand-potential",
            "Assess electrochemical window and interfacial reaction products.",
            "Grand-potential phase diagrams and explicit interface DFT",
            ("electrochemical_stability_window",),
        ),
    ),
    references=(
        "doi:10.1038/s41586-023-06735-9",
        "materialsproject:phase-diagram-and-battery-methodology",
    ),
    boundary="Static MLIP energy and force agreement cannot establish ionic conductivity or electrode compatibility.",
)


SUPERCONDUCTOR_PROFILE = _profile(
    field=MaterialField.SUPERCONDUCTOR,
    name="Superconductor",
    domain=DiscoveryDomain.SUPERCONDUCTORS,
    candidate_types=(CandidateType.CRYSTAL,),
    application_subtypes=("electron_phonon", "unconventional"),
    scope="Conventional or candidate unconventional superconductors under explicit pressure, field, isotope, and temperature conditions.",
    context=("pressure", "temperature", "magnetic_field", "pairing_assumption", "isotope"),
    properties=(
        _property(
            "critical_temperature",
            "K",
            "Superconducting transition temperature under declared conditions.",
            ("Electron-phonon/anisotropic Eliashberg calculation for a justified conventional mechanism; otherwise mechanism-specific many-body evidence.",),
            ("Coincident zero resistance and bulk diamagnetic/Meissner response.",),
            context=("pressure", "magnetic_field", "isotope"),
        ),
        _property(
            "electron_phonon_coupling",
            "dimensionless",
            "Conventional pairing strength with converged phonon and k/q meshes.",
            ("DFPT plus Wannier-interpolated electron-phonon calculation.",),
            ("Isotope, tunnelling, heat-capacity, or spectroscopic evidence.",),
            context=("pressure",),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Separate ambient-pressure, high-pressure, conventional, and unconventional claims.",
            "Retrieve pressure paths, competing phases, decomposition, isotope effects, and null magnetic or transport results.",
        ),
        "dft_handoff": (
            "Require pressure-dependent phase and phonon stability before electron-phonon Tc.",
            "Do not apply Allen-Dynes outside a justified phonon-mediated regime.",
        ),
    },
    domain_mcp={
        "generation_prior": ("superconducting-materials-and-pressure-search",),
        "identity_novelty": ("superconductor-database-search",),
        "dft_handoff": ("electron-phonon-method-and-data-search",),
    },
    dft_validators=(
        _dft(
            "pressure-phase-and-phonon-workflow",
            "Establish stable structures and phonons at every claimed pressure.",
            "Quantum ESPRESSO/ABINIT/VASP plus phonon workflow",
            ("energy_above_hull", "minimum_phonon_frequency"),
        ),
        _dft(
            "epw-or-eliashberg-workflow",
            "Calculate converged electron-phonon coupling and Tc only for a justified conventional mechanism.",
            "EPW/Quantum ESPRESSO or equivalent Eliashberg implementation",
            ("electron_phonon_coupling", "critical_temperature"),
        ),
    ),
    references=(
        "epw:official-documentation-and-method-paper",
        "quantum-espresso:phonon-and-electron-phonon",
        "allen-dynes:doi:10.1103/PhysRevB.12.905",
    ),
    boundary="Predicted Tc alone is not superconductivity; phase identity plus coincident zero-resistance and magnetic screening signatures are required experimentally.",
)


HETEROGENEOUS_CATALYST_PROFILE = _profile(
    field=MaterialField.HETEROGENEOUS_CATALYST,
    name="Heterogeneous catalyst or electrocatalyst",
    domain=DiscoveryDomain.CATALYSTS,
    candidate_types=(CandidateType.CRYSTAL, CandidateType.CATALYST),
    application_subtypes=(
        "thermal_catalysis",
        "electrocatalysis_her",
        "electrocatalysis_oer",
        "electrocatalysis_orr",
        "electrocatalysis_co2rr",
        "electrocatalysis_nrr",
        "photocatalysis",
    ),
    scope="Bulk, surface, interface, or supported catalysts evaluated for reaction-, facet-, coverage-, potential-, and environment-specific activity, selectivity, and durability.",
    context=("reaction", "facet", "coverage", "temperature", "pressure", "electrode_potential", "ph"),
    properties=(
        _property(
            "reaction_free_energy",
            "eV",
            "Free-energy landscape for a declared mechanism and operating condition.",
            ("Surface/slab DFT with adsorption, solvation, field, entropy, and coverage corrections.",),
            ("Product-resolved activity under documented reaction conditions.",),
            context=("reaction", "facet", "coverage", "temperature", "pressure", "electrode_potential", "ph"),
        ),
        _property(
            "activation_barrier",
            "eV",
            "Kinetic barrier for rate- or selectivity-determining elementary steps.",
            ("Transition-state search or NEB plus microkinetic analysis.",),
            ("Turnover frequency and kinetic/isotope dependence.",),
            context=("reaction", "facet", "coverage", "temperature"),
        ),
        _property(
            "durability",
            "h",
            "Resistance to dissolution, reconstruction, poisoning, and support/interface change.",
            ("Surface Pourbaix/ab initio thermodynamics and reconstruction or dissolution calculations.",),
            ("Long-duration operation with post-mortem and operando characterization.",),
            context=("reaction", "temperature", "pressure", "electrode_potential", "ph"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve reaction, active site, facet, support, coverage, electrolyte, pH, potential, temperature, pressure, and deactivation evidence.",
        ),
        "identity_novelty": (
            "Bulk-crystal novelty cannot establish surface, interface, defect, or active-site novelty.",
        ),
        "dft_handoff": (
            "Construct surfaces, adsorbates, coverages, interfaces, and competing mechanisms before ranking activity.",
            "Use free-energy and microkinetic quantities; a single adsorption energy is not universal activity.",
        ),
    },
    domain_mcp={
        "generation_prior": ("catalysis-reaction-and-condition-search",),
        "identity_novelty": ("open-catalyst-and-surface-data-search",),
        "dft_handoff": ("ocp-oc20-oc22-and-catalysis-data-search",),
    },
    dft_validators=(
        _dft(
            "surface-adsorbate-free-energy-workflow",
            "Enumerate relevant facets, sites, adsorbates, coverages, and free-energy corrections.",
            "ASE/CatKit plus converged slab DFT and computational hydrogen electrode where applicable",
            ("reaction_free_energy", "adsorption_energy"),
        ),
        _dft(
            "transition-state-and-microkinetic-workflow",
            "Calculate key barriers and integrate a condition-specific microkinetic model.",
            "NEB/dimer transition-state search plus CatMAP or equivalent",
            ("activation_barrier", "turnover_frequency", "selectivity"),
        ),
        _validator(
            "operando-durability-validation",
            "dft_handoff",
            "Measure retained activity over a declared duration and preserve operando/post-mortem evidence.",
            "Application-specific experimental durability protocol",
            properties=("durability",),
            availability="external_required",
            evidence_kind="experimental",
            scores=True,
        ),
    ),
    references=(
        "ocp:oc20-doi:10.1021/acscatal.0c04525",
        "ocp:oc22-doi:10.1021/acscatal.2c05426",
        "catmap:doi:10.1002/cctc.201300825",
    ),
    boundary="Bulk stability or an OCP surrogate score is not catalytic activity; surface state, reaction conditions, kinetics, selectivity, and durability require separate validation.",
)


SEMICONDUCTOR_PROFILE = _profile(
    field=MaterialField.SEMICONDUCTOR,
    name="Electronic semiconductor",
    domain=DiscoveryDomain.INORGANIC_MATERIALS,
    candidate_types=(CandidateType.CRYSTAL,),
    application_subtypes=(
        "electronic_semiconductor",
        "transparent_conductor",
        "power_semiconductor",
    ),
    scope="Electronic and optoelectronic semiconductors evaluated for quasiparticle gaps, band edges, transport, defects, dielectric response, and stability.",
    context=("carrier_type", "temperature", "doping", "strain", "dimensionality"),
    properties=(
        _property(
            "band_gap",
            "eV",
            "Quasiparticle or experimentally calibrated optical/electronic gap.",
            ("Converged hybrid-functional or GW calculation with spin-orbit coupling when relevant.",),
            ("Optical absorption plus transport or photoemission characterization.",),
            context=("temperature", "strain"),
        ),
        _property(
            "carrier_mobility",
            "cm^2/(V s)",
            "Carrier mobility for a stated temperature and doping regime.",
            ("Electron-phonon scattering and Boltzmann transport beyond effective mass alone.",),
            ("Hall or field-effect mobility with carrier density and temperature.",),
            context=("carrier_type", "temperature", "doping"),
        ),
        _property(
            "minimum_native_defect_formation_energy",
            "eV",
            "Lowest relevant native-defect formation energy under declared chemical-potential and Fermi-level conditions.",
            ("Finite-size-corrected charged-defect supercells and competing-phase chemical potentials.",),
            ("Defect spectroscopy and controlled doping.",),
            context=("fermi_level", "chemical_potentials", "charge_state"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve dopability, native defects, polymorphs, oxidation, strain, and temperature-dependent electronic evidence.",
        ),
        "dft_handoff": (
            "Use hybrid/GW and spin-orbit coupling when semilocal DFT is inadequate.",
            "Enumerate charged defects and chemical-potential limits before a defect-tolerance or dopability claim.",
        ),
    },
    domain_mcp={
        "identity_novelty": ("electronic-materials-database-search",),
        "dft_handoff": ("gw-hybrid-defect-and-transport-method-search",),
    },
    dft_validators=(
        _dft(
            "hybrid-or-gw-electronic-structure",
            "Calculate calibrated band structure, band edges, dielectric response, and effective masses.",
            "Quantum ESPRESSO/Yambo, ABINIT, VASP, or equivalent hybrid/GW workflow",
            ("band_gap", "band_edges", "dielectric_response", "effective_mass"),
        ),
        _dft(
            "charged-defect-and-transport-workflow",
            "Evaluate finite-size-corrected defects, dopability, and electron-phonon-limited transport.",
            "doped/pydefect plus AMSET, EPW, or equivalent",
            ("minimum_native_defect_formation_energy", "carrier_mobility"),
        ),
    ),
    references=(
        "materialsproject:electronic-structure-methodology",
        "amset:doi:10.1063/5.0040355",
        "aiida:common-workflows-and-provenance",
    ),
    boundary="A semilocal-DFT or ML band-gap estimate is triage only and cannot establish transport, defects, dopability, or device performance.",
)


PHOTOVOLTAIC_ABSORBER_PROFILE = _profile(
    field=MaterialField.PHOTOVOLTAIC_ABSORBER,
    name="Photovoltaic absorber",
    domain=DiscoveryDomain.INORGANIC_MATERIALS,
    candidate_types=(CandidateType.CRYSTAL,),
    application_subtypes=(
        "single_junction_absorber",
        "tandem_top_absorber",
        "ordered_hybrid_perovskite",
    ),
    scope="Solar absorbers evaluated for quasiparticle/optical gaps, absorption, radiative efficiency, defects, interfaces, and operational stability.",
    context=("absorber_thickness", "temperature", "illumination", "contacts", "chemical_potentials"),
    properties=(
        _property(
            "optical_absorption_coefficient",
            "cm^-1",
            "Absorption coefficient at a declared photon energy, polarization, and structure.",
            ("GW or calibrated hybrid electronic structure plus BSE/independent-particle optics as justified.",),
            ("Thickness-resolved absorption or spectroscopic ellipsometry.",),
            context=("photon_energy", "polarization", "temperature"),
        ),
        _property(
            "slme",
            "fraction",
            "Spectroscopic limited maximum efficiency at a stated thickness.",
            ("SLME from a converged absorption spectrum and direct/indirect gap information.",),
            ("Certified device efficiency and loss analysis; SLME itself is computational.",),
            context=("absorber_thickness", "temperature"),
        ),
        _property(
            "nonradiative_recombination_rate",
            "s^-1",
            "Non-radiative recombination rate for declared defects, interfaces, carrier density, and temperature.",
            ("Charged-defect, interface band-alignment, and recombination calculations.",),
            ("Lifetime, photoluminescence, defect spectroscopy, and device stability.",),
            context=("chemical_potentials", "contacts", "temperature", "carrier_concentration"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve optical gap, absorption thickness, defect chemistry, contact compatibility, degradation, and failed device evidence.",
        ),
        "dft_handoff": (
            "Calculate an absorption spectrum before SLME; do not use band gap alone as photovoltaic efficiency.",
            "Add defects, interfaces, and operational stability before a device claim.",
        ),
    },
    domain_mcp={
        "generation_prior": ("photovoltaic-material-and-device-search",),
        "dft_handoff": ("optical-defect-interface-data-search",),
    },
    dft_validators=(
        _dft(
            "quasiparticle-optics-and-slme",
            "Calculate quasiparticle/optical response and thickness-dependent SLME.",
            "GW/hybrid plus optical workflow and SLME implementation",
            ("optical_absorption_coefficient", "slme"),
        ),
        _dft(
            "photovoltaic-defect-interface-workflow",
            "Evaluate native defects, band alignment, interfaces, and decomposition pathways.",
            "Charged-defect and interface DFT workflow",
            ("nonradiative_recombination_rate", "band_alignment", "operational_stability"),
        ),
    ),
    references=(
        "slme:doi:10.1103/PhysRevLett.108.068701",
        "materialsproject:optical-absorption-methodology",
    ),
    boundary="Band gap near an empirical optimum does not establish absorption, SLME, defect tolerance, interface compatibility, or device efficiency.",
)


THERMOELECTRIC_PROFILE = _profile(
    field=MaterialField.THERMOELECTRIC,
    name="Thermoelectric",
    domain=DiscoveryDomain.INORGANIC_MATERIALS,
    candidate_types=(CandidateType.CRYSTAL,),
    application_subtypes=("bulk_n_type", "bulk_p_type", "two_dimensional"),
    scope="Thermoelectrics evaluated at explicit temperature, carrier concentration, and microstructure using coupled electronic and phonon transport.",
    context=("temperature", "carrier_concentration", "carrier_type", "microstructure"),
    properties=(
        _property(
            "power_factor",
            "W/(m K^2)",
            "Seebeck-squared conductivity at stated temperature and doping.",
            ("Boltzmann electronic transport with a justified scattering model.",),
            ("Simultaneous Seebeck coefficient and electrical conductivity.",),
            context=("temperature", "carrier_concentration", "carrier_type"),
        ),
        _property(
            "lattice_thermal_conductivity",
            "W/(m K)",
            "Phonon thermal conductivity at a declared temperature and structure.",
            ("Converged second/third-order force constants and phonon Boltzmann transport.",),
            ("Thermal diffusivity/conductivity with density and microstructure.",),
            context=("temperature", "microstructure"),
        ),
        _property(
            "zt",
            "dimensionless",
            "Coupled thermoelectric figure of merit using consistent conditions.",
            ("Integrated electronic and lattice transport with electronic thermal conductivity.",),
            ("Co-measured S, sigma, kappa, and temperature on representative samples.",),
            context=("temperature", "carrier_concentration", "microstructure"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve temperature-, doping-, direction-, and microstructure-resolved transport rather than room-temperature scalar summaries.",
        ),
        "dft_handoff": (
            "Keep power factor and lattice thermal conductivity as independent tasks before combining ZT.",
            "Record scattering-time assumptions and convergence of third-order force constants.",
        ),
    },
    domain_mcp={
        "generation_prior": ("thermoelectric-transport-data-search",),
        "dft_handoff": ("phonon-and-boltzmann-transport-search",),
    },
    dft_validators=(
        _dft(
            "electronic-boltzmann-transport",
            "Calculate Seebeck, conductivity, and electronic thermal transport versus temperature and doping.",
            "BoltzTraP2, AMSET, or EPW",
            ("power_factor",),
        ),
        _dft(
            "anharmonic-phonon-transport",
            "Calculate lattice thermal conductivity with converged anharmonic force constants.",
            "phono3py or ShengBTE",
            ("lattice_thermal_conductivity",),
        ),
        _dft(
            "thermoelectric-zt-integration",
            "Combine co-conditioned electronic and lattice quantities without mixing incompatible assumptions.",
            "Audited thermoelectric transport post-processing",
            ("zt",),
        ),
    ),
    references=(
        "boltztrap2:doi:10.1016/j.cpc.2017.03.015",
        "phono3py:doi:10.1103/PhysRevB.91.094306",
        "shengbte:doi:10.1016/j.cpc.2014.02.015",
    ),
    boundary="A favorable band structure or low harmonic phonon frequency alone cannot establish power factor, thermal conductivity, or ZT.",
)


MAGNETIC_MATERIAL_PROFILE = _profile(
    field=MaterialField.MAGNETIC_MATERIAL,
    name="Magnetic material",
    domain=DiscoveryDomain.INORGANIC_MATERIALS,
    candidate_types=(CandidateType.CRYSTAL,),
    application_subtypes=(
        "hard_magnet",
        "soft_magnet",
        "two_dimensional_magnet",
        "spintronic_magnet",
        "magnetocaloric",
    ),
    scope="Permanent, soft, spintronic, or magnetocaloric materials evaluated across magnetic order, correlations, spin-orbit coupling, temperature, and field.",
    context=("magnetic_application", "temperature", "magnetic_field", "oxidation_states"),
    properties=(
        _property(
            "magnetic_ordering_energy",
            "eV/atom",
            "Energy difference from the lowest justified magnetic configuration among enumerated orders.",
            ("Magnetic-order enumeration with converged spin-polarized DFT and justified +U/hybrid treatment.",),
            ("Neutron or magnetic diffraction and magnetometry.",),
            context=("temperature", "magnetic_field"),
        ),
        _property(
            "magnetocrystalline_anisotropy",
            "MJ/m^3",
            "Spin-orbit-driven anisotropy for the intended application.",
            ("Dense-k-mesh spin-orbit calculations with orientation convergence.",),
            ("Torque or angle-dependent magnetometry.",),
            context=("temperature",),
        ),
        _property(
            "ordering_temperature",
            "K",
            "Curie or Néel temperature from an explicit spin model.",
            ("Exchange extraction followed by statistical-mechanical simulation.",),
            ("Temperature-dependent susceptibility, heat capacity, and diffraction.",),
            context=("magnetic_field",),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve oxidation states, magnetic structures, +U choices, spin-orbit effects, anisotropy, coercivity, and temperature-dependent evidence.",
        ),
        "dft_handoff": (
            "Enumerate magnetic orderings before comparing energies or band structures.",
            "Treat anisotropy and ordering temperature as separate SOC and statistical-mechanics tasks.",
        ),
    },
    domain_mcp={
        "identity_novelty": ("magnetic-structure-database-search",),
        "dft_handoff": ("magnetic-order-soc-and-exchange-search",),
    },
    dft_validators=(
        _dft(
            "magnetic-order-and-correlation-workflow",
            "Enumerate orderings and justify oxidation, spin, and correlation settings.",
            "pymatgen magnetic ordering plus spin-polarized DFT(+U/hybrid)",
            ("magnetic_ordering_energy",),
        ),
        _dft(
            "soc-anisotropy-exchange-temperature-workflow",
            "Calculate anisotropy/exchange and estimate ordering temperature with an explicit spin model.",
            "SOC DFT plus atomistic spin dynamics or Monte Carlo",
            ("magnetocrystalline_anisotropy", "ordering_temperature"),
        ),
    ),
    references=(
        "materialsproject:magnetic-properties-methodology",
        "pymatgen:magnetic-structure-enumeration",
    ),
    boundary="A single ferromagnetic initialization or predicted moment cannot establish the magnetic ground state, anisotropy, coercivity, or ordering temperature.",
)


FERROELECTRIC_PIEZOELECTRIC_PROFILE = _profile(
    field=MaterialField.FERROELECTRIC_PIEZOELECTRIC,
    name="Ferroelectric or piezoelectric",
    domain=DiscoveryDomain.INORGANIC_MATERIALS,
    candidate_types=(CandidateType.CRYSTAL,),
    application_subtypes=("ferroelectric", "piezoelectric"),
    scope="Polar, ferroelectric, and piezoelectric crystals evaluated for symmetry, dynamic stability, polarization, switching, dielectric response, and electromechanical coupling.",
    context=("temperature", "electric_field", "stress", "orientation", "domain_state"),
    properties=(
        _property(
            "spontaneous_polarization",
            "C/m^2",
            "Berry-phase polarization difference along an insulating switching path.",
            ("Berry-phase polarization relative to a justified non-polar reference.",),
            ("Polarization-electric-field loop with leakage controls.",),
            context=("orientation", "temperature"),
        ),
        _property(
            "switching_barrier",
            "eV per formula unit",
            "Barrier along a physically meaningful polarization-switching path.",
            ("NEB or constrained structural path with insulating-state checks.",),
            ("Coercive field and switching kinetics with domain analysis.",),
            context=("electric_field", "stress"),
        ),
        _property(
            "piezoelectric_strain_coefficient",
            "pm/V",
            "Selected symmetry-resolved strain coefficient for a declared tensor component.",
            ("DFPT dielectric, Born effective charge, elastic, and piezoelectric tensors.",),
            ("Orientation-resolved piezoelectric and dielectric measurement.",),
            context=("orientation", "tensor_component", "temperature", "stress"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve polar space groups, non-polar references, phase transitions, leakage, domains, and fatigue evidence.",
        ),
        "dft_handoff": (
            "Confirm insulating character along the polarization path and calculate full symmetry-resolved tensors.",
            "Separate structural polarity from switchable ferroelectricity.",
        ),
    },
    domain_mcp={
        "identity_novelty": ("polar-and-ferroelectric-structure-search",),
        "dft_handoff": ("dfpt-polarization-and-piezoelectric-search",),
    },
    dft_validators=(
        _dft(
            "dfpt-polar-response-workflow",
            "Calculate phonons, Born charges, dielectric, elastic, and piezoelectric tensors.",
            "Quantum ESPRESSO/ABINIT/VASP DFPT or equivalent",
            ("piezoelectric_strain_coefficient",),
        ),
        _dft(
            "berry-phase-switching-workflow",
            "Calculate polarization difference and an insulating switching path/barrier.",
            "Berry-phase polarization plus NEB/constrained path",
            ("spontaneous_polarization", "switching_barrier"),
        ),
    ),
    references=(
        "materialsproject:piezoelectric-and-dielectric-methodology",
        "modern-polarization:doi:10.1103/RevModPhys.66.899",
    ),
    boundary="A polar space group is not proof of switchable ferroelectricity; polarization path, leakage, switching, domains, and fatigue remain required.",
)


STRUCTURAL_ALLOY_PROFILE = _profile(
    field=MaterialField.STRUCTURAL_ALLOY,
    name="Structural or high-temperature alloy",
    domain=DiscoveryDomain.INORGANIC_MATERIALS,
    candidate_types=(CandidateType.CRYSTAL, CandidateType.ALLOY),
    application_subtypes=(
        "structural_alloy",
        "high_temperature_alloy",
        "creep_resistant_alloy",
        "corrosion_resistant_alloy",
        "high_entropy_alloy",
    ),
    scope="Ordered or disordered structural alloys evaluated across phase equilibria, elasticity, defects, fracture, creep, oxidation, processing, and service conditions.",
    context=("composition_range", "temperature", "pressure", "processing_history", "service_environment"),
    properties=(
        _property(
            "mixing_gibbs_free_energy",
            "eV/atom",
            "Phase equilibria including disorder and finite-temperature effects.",
            ("DFT cluster expansion/phonons integrated with CALPHAD or another validated thermodynamic model.",),
            ("Phase diagram and quantitative microstructure characterization.",),
            context=("composition_range", "temperature", "processing_history"),
        ),
        _property(
            "youngs_modulus",
            "GPa",
            "Orientation- and temperature-resolved elastic stiffness for the declared load axis.",
            ("Converged elastic tensor followed by defect, slip, surface, or fracture calculations as needed.",),
            ("Tensile, creep, fatigue, hardness, and fracture testing under service conditions.",),
            context=("temperature", "orientation", "microstructure"),
        ),
        _property(
            "service_degradation_rate",
            "s^-1",
            "Fractional degradation rate for one declared service mechanism and environment.",
            ("Surface/reaction thermodynamics and kinetic or mesoscale modeling.",),
            ("Long-duration exposure with mass change and microstructure analysis.",),
            context=("service_environment", "degradation_mechanism", "temperature", "time"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve composition ranges, processing history, disorder, precipitates, grain size, service environment, creep, fatigue, and failed alloys.",
        ),
        "dft_handoff": (
            "Do not infer finite-temperature alloy stability from a single ordered 0 K cell.",
            "Separate elastic constants from strength, toughness, creep, fatigue, and corrosion claims.",
        ),
    },
    domain_mcp={
        "generation_prior": ("alloy-processing-and-failure-search",),
        "dft_handoff": ("calphad-elastic-defect-and-service-data-search",),
    },
    dft_validators=(
        _dft(
            "finite-temperature-alloy-thermodynamics",
            "Model disorder and phase equilibria across composition and temperature.",
            "DFT/cluster expansion plus pycalphad or validated CALPHAD database",
            ("mixing_gibbs_free_energy",),
        ),
        _dft(
            "elastic-defect-and-service-workflow",
            "Calculate elastic tensors and task-specific defects/surfaces while preserving experimental service gates.",
            "Periodic DFT plus atomistic/mesoscale workflow",
            ("youngs_modulus", "service_degradation_rate"),
        ),
    ),
    references=(
        "pycalphad:doi:10.1186/2193-9772-3-4",
        "materialsproject:elasticity-methodology",
    ),
    boundary="0 K stability and elastic moduli cannot establish alloy strength, toughness, creep, fatigue, oxidation, or processability.",
)


POROUS_FRAMEWORK_PROFILE = _profile(
    field=MaterialField.POROUS_FRAMEWORK,
    name="Porous framework, MOF, or zeolite",
    domain=DiscoveryDomain.INORGANIC_MATERIALS,
    candidate_types=(CandidateType.CRYSTAL,),
    application_subtypes=(
        "adsorption",
        "separation",
        "storage",
    ),
    scope="Porous frameworks evaluated for accessible geometry, adsorption/selectivity, diffusion, framework flexibility, activation, and chemical stability.",
    context=("guest_species", "temperature", "pressure", "humidity", "activation_state"),
    properties=(
        _property(
            "accessible_volume_fraction",
            "dimensionless",
            "Probe- and activation-state-dependent accessible volume fraction.",
            ("Periodic geometry analysis with an explicit probe and disorder/solvent treatment.",),
            ("Gas sorption and structural characterization after activation.",),
            context=("guest_species", "activation_state"),
        ),
        _property(
            "adsorption_selectivity",
            "dimensionless",
            "Mixture adsorption selectivity under declared thermodynamic conditions.",
            ("Validated-force-field or ab initio GCMC with charges, flexibility, and mixture conditions.",),
            ("Pure and mixture adsorption isotherms with cycling.",),
            context=("guest_species", "temperature", "pressure", "humidity"),
        ),
        _property(
            "framework_decomposition_free_energy",
            "eV/atom",
            "Decomposition free energy against declared products at the operating conditions.",
            ("Phonon/elastic/flexible-framework simulation plus reaction stability as appropriate.",),
            ("PXRD and adsorption retained after thermal/humidity/chemical cycling.",),
            context=("temperature", "humidity", "activation_state"),
        ),
    ),
    domain_focus={
        "generation_prior": (
            "Retrieve activation, residual solvent, disorder, humidity, framework flexibility, mixture conditions, and failed syntheses.",
        ),
        "identity_novelty": (
            "Normalize solvent/disorder carefully and compare topology as well as crystallographic structure.",
        ),
        "dft_handoff": (
            "State probe, force field, charges, framework flexibility, and mixture conditions for porosity or adsorption.",
            "Require activation and water/chemical stability before a deployable adsorption claim.",
        ),
    },
    domain_mcp={
        "identity_novelty": ("core-mof-csd-cod-and-zeolite-search",),
        "dft_handoff": ("adsorption-isotherm-and-force-field-search",),
    },
    dft_validators=(
        _dft(
            "probe-resolved-porosity-workflow",
            "Calculate accessible geometry for explicit probe and activation assumptions.",
            "Zeo++ or PoreBlazer",
            ("accessible_volume_fraction",),
        ),
        _dft(
            "gcmc-mixture-adsorption-workflow",
            "Calculate uptake/selectivity with validated charges, force fields, flexibility, and conditions.",
            "RASPA or equivalent GCMC workflow",
            ("adsorption_selectivity",),
        ),
        _dft(
            "framework-stability-workflow",
            "Assess dynamic, mechanical, activation, and water/chemical stability.",
            "Periodic DFT/phonons plus flexible-framework or reactive simulation",
            ("framework_decomposition_free_energy",),
        ),
    ),
    references=(
        "core-mof:doi:10.1021/acs.jced.9b00835",
        "raspa:doi:10.1080/08927022.2015.1010082",
        "zeopp:doi:10.1021/acs.chemmater.7b01475",
    ),
    boundary="A geometric pore or rigid-framework GCMC result cannot establish accessible capacity, mixture selectivity, activation success, flexibility, or humidity stability.",
)


MATERIAL_FIELD_PROFILES: dict[MaterialField, MaterialFieldProfile] = {
    profile.material_field: profile
    for profile in (
        GENERAL_INORGANIC_PROFILE,
        BATTERY_ELECTRODE_PROFILE,
        SOLID_ELECTROLYTE_PROFILE,
        SUPERCONDUCTOR_PROFILE,
        HETEROGENEOUS_CATALYST_PROFILE,
        SEMICONDUCTOR_PROFILE,
        PHOTOVOLTAIC_ABSORBER_PROFILE,
        THERMOELECTRIC_PROFILE,
        MAGNETIC_MATERIAL_PROFILE,
        FERROELECTRIC_PIEZOELECTRIC_PROFILE,
        STRUCTURAL_ALLOY_PROFILE,
        POROUS_FRAMEWORK_PROFILE,
    )
}


_EXPLICIT_ALIASES: dict[str, MaterialField] = {
    "general": MaterialField.GENERAL_INORGANIC,
    "inorganic": MaterialField.GENERAL_INORGANIC,
    "battery": MaterialField.BATTERY_ELECTRODE,
    "electrode": MaterialField.BATTERY_ELECTRODE,
    "solid electrolyte": MaterialField.SOLID_ELECTROLYTE,
    "ionic conductor": MaterialField.SOLID_ELECTROLYTE,
    "superconductivity": MaterialField.SUPERCONDUCTOR,
    "catalyst": MaterialField.HETEROGENEOUS_CATALYST,
    "electrocatalyst": MaterialField.HETEROGENEOUS_CATALYST,
    "semiconductor": MaterialField.SEMICONDUCTOR,
    "solar absorber": MaterialField.PHOTOVOLTAIC_ABSORBER,
    "photovoltaic": MaterialField.PHOTOVOLTAIC_ABSORBER,
    "thermoelectric": MaterialField.THERMOELECTRIC,
    "magnetic": MaterialField.MAGNETIC_MATERIAL,
    "ferroelectric": MaterialField.FERROELECTRIC_PIEZOELECTRIC,
    "piezoelectric": MaterialField.FERROELECTRIC_PIEZOELECTRIC,
    "alloy": MaterialField.STRUCTURAL_ALLOY,
    "mof": MaterialField.POROUS_FRAMEWORK,
    "zeolite": MaterialField.POROUS_FRAMEWORK,
    "다공성": MaterialField.POROUS_FRAMEWORK,
    "배터리": MaterialField.BATTERY_ELECTRODE,
    "전극": MaterialField.BATTERY_ELECTRODE,
    "고체전해질": MaterialField.SOLID_ELECTROLYTE,
    "이온전도체": MaterialField.SOLID_ELECTROLYTE,
    "초전도": MaterialField.SUPERCONDUCTOR,
    "촉매": MaterialField.HETEROGENEOUS_CATALYST,
    "반도체": MaterialField.SEMICONDUCTOR,
    "태양전지": MaterialField.PHOTOVOLTAIC_ABSORBER,
    "열전": MaterialField.THERMOELECTRIC,
    "자성": MaterialField.MAGNETIC_MATERIAL,
    "강유전": MaterialField.FERROELECTRIC_PIEZOELECTRIC,
    "압전": MaterialField.FERROELECTRIC_PIEZOELECTRIC,
    "합금": MaterialField.STRUCTURAL_ALLOY,
    "제올라이트": MaterialField.POROUS_FRAMEWORK,
}
for _field in MaterialField:
    _EXPLICIT_ALIASES[_field.value] = _field


_DETECTION_TERMS: dict[MaterialField, dict[str, int]] = {
    MaterialField.BATTERY_ELECTRODE: {
        "battery": 3,
        "cathode": 4,
        "anode": 4,
        "electrode": 3,
        "intercalation": 4,
        "배터리": 3,
        "양극": 4,
        "음극": 4,
        "전극": 3,
    },
    MaterialField.SOLID_ELECTROLYTE: {
        "solid electrolyte": 7,
        "ionic conductor": 6,
        "ionic conductivity": 5,
        "ion conductor": 5,
        "고체전해질": 7,
        "이온전도체": 6,
        "이온전도도": 5,
    },
    MaterialField.SUPERCONDUCTOR: {
        "superconductor": 5,
        "superconductivity": 5,
        "critical temperature": 3,
        "meissner": 5,
        "초전도": 5,
        "임계온도": 3,
        "마이스너": 5,
    },
    MaterialField.HETEROGENEOUS_CATALYST: {
        "heterogeneous catalyst": 6,
        "electrocatalyst": 6,
        "photocatalyst": 6,
        "catalyst": 4,
        "adsorption energy": 3,
        "촉매": 4,
        "전기촉매": 6,
        "광촉매": 6,
    },
    MaterialField.SEMICONDUCTOR: {
        "semiconductor": 5,
        "carrier mobility": 4,
        "dopability": 4,
        "band gap": 2,
        "반도체": 5,
        "이동도": 3,
        "도핑": 3,
    },
    MaterialField.PHOTOVOLTAIC_ABSORBER: {
        "photovoltaic": 7,
        "solar cell": 7,
        "solar absorber": 7,
        "slme": 6,
        "태양전지": 7,
        "광흡수체": 6,
    },
    MaterialField.THERMOELECTRIC: {
        "thermoelectric": 7,
        "seebeck": 6,
        "power factor": 5,
        "lattice thermal conductivity": 4,
        "열전": 7,
        "제벡": 6,
    },
    MaterialField.MAGNETIC_MATERIAL: {
        "magnetic material": 6,
        "ferromagnet": 5,
        "antiferromagnet": 5,
        "magnetocaloric": 6,
        "anisotropy": 3,
        "자성": 5,
        "강자성": 5,
        "반강자성": 5,
    },
    MaterialField.FERROELECTRIC_PIEZOELECTRIC: {
        "ferroelectric": 7,
        "piezoelectric": 7,
        "spontaneous polarization": 5,
        "강유전": 7,
        "압전": 7,
        "자발분극": 5,
    },
    MaterialField.STRUCTURAL_ALLOY: {
        "structural alloy": 6,
        "high entropy alloy": 7,
        "superalloy": 6,
        "creep": 4,
        "fatigue": 4,
        "구조용 합금": 6,
        "고엔트로피 합금": 7,
        "초합금": 6,
        "크리프": 4,
        "피로": 4,
    },
    MaterialField.POROUS_FRAMEWORK: {
        "metal-organic framework": 8,
        "porous framework": 7,
        "zeolite": 6,
        "mof": 5,
        "gas adsorption": 4,
        "금속유기골격체": 8,
        "다공성": 5,
        "제올라이트": 6,
    },
}


def get_material_field_profile(
    material_field: MaterialField | str,
) -> MaterialFieldProfile:
    field = MaterialField(str(material_field))
    return MaterialFieldProfile.model_validate_json(
        MATERIAL_FIELD_PROFILES[field].model_dump_json(),
        strict=True,
    )


def _resolve_material_field_deterministic(
    requested: MaterialField | str | None,
    *,
    prompt: str = "",
) -> MaterialFieldResolution:
    raw = str(requested or "AUTO").strip()
    normalized = _normalize(raw)
    if normalized not in {"", "auto", "자동"}:
        field = _EXPLICIT_ALIASES.get(normalized)
        if field is None:
            try:
                field = MaterialField(normalized.replace(" ", "_"))
            except ValueError as exc:
                choices = ", ".join(item.value for item in MaterialField)
                raise ValueError(
                    f"unknown material field {raw!r}; use AUTO or one of: {choices}"
                ) from exc
        profile = MATERIAL_FIELD_PROFILES[field]
        return MaterialFieldResolution(
            requested=raw,
            selected_field=field,
            profile_id=profile.profile_id,
            selection_mode="explicit",
            confidence="high",
            reason="The operator selected a code-owned material-field profile.",
        )

    haystack = _normalize(prompt)
    scores: dict[MaterialField, int] = {}
    matches: dict[str, list[str]] = {}
    for field, terms in _DETECTION_TERMS.items():
        field_matches = [
            term for term in terms if _contains_term(haystack, _normalize(term))
        ]
        if field_matches:
            scores[field] = sum(terms[term] for term in field_matches)
            matches[field.value] = sorted(field_matches)
    if not scores:
        profile = MATERIAL_FIELD_PROFILES[MaterialField.GENERAL_INORGANIC]
        return MaterialFieldResolution(
            requested=raw or "AUTO",
            selected_field=MaterialField.GENERAL_INORGANIC,
            profile_id=profile.profile_id,
            selection_mode="auto-default",
            confidence="low",
            reason=(
                "No unambiguous field term was present; the conservative general "
                "inorganic workflow was selected and no specialized property claim is implied."
            ),
        )
    best_score = max(scores.values())
    winners = sorted(
        (field for field, score in scores.items() if score == best_score),
        key=lambda item: item.value,
    )
    if len(winners) > 1:
        profile = MATERIAL_FIELD_PROFILES[MaterialField.GENERAL_INORGANIC]
        return MaterialFieldResolution(
            requested=raw or "AUTO",
            selected_field=MaterialField.GENERAL_INORGANIC,
            profile_id=profile.profile_id,
            selection_mode="auto-ambiguous",
            confidence="low",
            matched_terms=matches,
            ambiguous_fields=winners,
            requires_operator_choice=True,
            reason=(
                "Several material fields received the same deterministic score; "
                "the run is restricted to general screening until the operator chooses one."
            ),
        )
    field = winners[0]
    profile = MATERIAL_FIELD_PROFILES[field]
    second = max((score for key, score in scores.items() if key != field), default=0)
    confidence: Literal["high", "medium", "low"] = (
        "high" if best_score >= 6 and best_score >= second + 3 else "medium"
    )
    return MaterialFieldResolution(
        requested=raw or "AUTO",
        selected_field=field,
        profile_id=profile.profile_id,
        selection_mode="auto-keyword",
        confidence=confidence,
        matched_terms=matches,
        reason=(
            "A deterministic, code-owned keyword scorer selected the unique "
            "highest-scoring material field; no model output chose the route."
        ),
    )


def resolve_material_field(
    requested: MaterialField | str | None,
    *,
    prompt: str = "",
    chemical_system: str | None = None,
    problem_context: Mapping[str, JsonValue] | None = None,
    model_run: MaterialFieldModelRun | None = None,
) -> MaterialFieldResolution:
    """Reconcile explicit input, deterministic evidence, and the main AI.

    Explicit operator selection always wins.  In ``AUTO`` mode, a verified
    high-confidence main-model decision can resolve a default or keyword tie.
    Agreement is recorded as consensus; disagreement becomes an operator
    decision instead of silently choosing one scientific workflow.
    """

    deterministic = _resolve_material_field_deterministic(
        requested,
        prompt=prompt,
    )
    if deterministic.selection_mode == "explicit" or model_run is None:
        return deterministic
    context = dict(problem_context or {})
    if model_run.user_prompt_hash != stable_hash(prompt.strip()) or (
        model_run.prompt_hash
        != stable_hash(
            {
                "prompt": prompt.strip(),
                "chemical_system": chemical_system,
                "problem_context": context,
            }
        )
    ):
        raise ValueError(
            "main-model material-field decision was produced for different "
            "classification inputs"
        )

    decision = model_run.decision
    model_primary = (
        MaterialField(str(decision.primary_field))
        if decision.primary_field is not None
        else None
    )
    secondary = [MaterialField(str(item)) for item in decision.secondary_fields]
    model_confidence = float(decision.confidence)
    confidence: Literal["high", "medium", "low"] = (
        "high"
        if model_confidence >= 0.85
        else "medium"
        if model_confidence >= 0.70
        else "low"
    )
    if (
        decision.needs_clarification
        or model_primary is None
        or model_confidence < 0.70
    ):
        ambiguous = list(
            dict.fromkeys(
                [
                    *deterministic.ambiguous_fields,
                    *(
                        [deterministic.selected_field]
                        if deterministic.selected_field
                        != MaterialField.GENERAL_INORGANIC
                        else []
                    ),
                    *([model_primary] if model_primary is not None else []),
                    *secondary,
                ]
            )
        )
        profile = MATERIAL_FIELD_PROFILES[MaterialField.GENERAL_INORGANIC]
        return MaterialFieldResolution(
            requested=deterministic.requested,
            selected_field=MaterialField.GENERAL_INORGANIC,
            profile_id=profile.profile_id,
            selection_mode="auto-model-conflict",
            confidence="low",
            matched_terms=deterministic.matched_terms,
            ambiguous_fields=ambiguous,
            secondary_fields=secondary,
            model_decision_id=model_run.decision_id,
            model_confidence=model_confidence,
            requires_operator_choice=True,
            reason=(
                "The main model requested clarification or did not meet the "
                "code-owned confidence threshold; specialized routing is blocked."
            ),
        )

    if deterministic.selection_mode in {"auto-default", "auto-ambiguous"}:
        profile = MATERIAL_FIELD_PROFILES[model_primary]
        return MaterialFieldResolution(
            requested=deterministic.requested,
            selected_field=model_primary,
            profile_id=profile.profile_id,
            selection_mode="auto-model",
            confidence=confidence,
            matched_terms=deterministic.matched_terms,
            secondary_fields=secondary,
            application_subtype=decision.application_subtype,
            model_decision_id=model_run.decision_id,
            model_confidence=model_confidence,
            reason=(
                "The verified main-model decision resolved a deterministic "
                "default or tie using exact evidence spans from the user input."
            ),
        )

    if deterministic.selected_field == model_primary:
        profile = MATERIAL_FIELD_PROFILES[model_primary]
        return MaterialFieldResolution(
            requested=deterministic.requested,
            selected_field=model_primary,
            profile_id=profile.profile_id,
            selection_mode="auto-consensus",
            confidence=confidence,
            matched_terms=deterministic.matched_terms,
            secondary_fields=secondary,
            application_subtype=decision.application_subtype,
            model_decision_id=model_run.decision_id,
            model_confidence=model_confidence,
            reason=(
                "The deterministic scorer and verified main-model decision agree "
                "on the primary material field."
            ),
        )

    ambiguous = list(
        dict.fromkeys(
            [
                deterministic.selected_field,
                model_primary,
                *secondary,
            ]
        )
    )
    profile = MATERIAL_FIELD_PROFILES[MaterialField.GENERAL_INORGANIC]
    return MaterialFieldResolution(
        requested=deterministic.requested,
        selected_field=MaterialField.GENERAL_INORGANIC,
        profile_id=profile.profile_id,
        selection_mode="auto-model-conflict",
        confidence="low",
        matched_terms=deterministic.matched_terms,
        ambiguous_fields=ambiguous,
        secondary_fields=secondary,
        model_decision_id=model_run.decision_id,
        model_confidence=model_confidence,
        requires_operator_choice=True,
        reason=(
            "The deterministic scorer and main model selected different primary "
            "fields; the system will not silently choose a scientific workflow."
        ),
    )


def build_material_domain_plan(
    requested: MaterialField | str | None,
    *,
    prompt: str = "",
    chemical_system: str | None = None,
    calculated_property_names: Iterable[str] = (),
    problem_context: Mapping[str, JsonValue] | None = None,
    model_run: MaterialFieldModelRun | None = None,
) -> MaterialDomainPlan:
    provided_context = dict(problem_context or {})
    context_chemical_system = provided_context.get("chemical_system")
    if (
        chemical_system
        and not _context_value_is_missing(context_chemical_system)
        and _normalize(str(context_chemical_system)) != _normalize(chemical_system)
    ):
        raise ValueError(
            "chemical_system conflicts with problem_context chemical_system"
        )
    resolution = resolve_material_field(
        requested,
        prompt=prompt,
        chemical_system=chemical_system,
        problem_context=provided_context,
        model_run=model_run,
    )
    profile = get_material_field_profile(resolution.selected_field)
    reported = list(dict.fromkeys(str(name) for name in calculated_property_names))
    if chemical_system and "chemical_system" not in provided_context:
        provided_context["chemical_system"] = chemical_system
    missing_context = [
        name
        for name in profile.required_problem_context
        if _context_value_is_missing(provided_context.get(name))
    ]
    missing = [
        item.property_name
        for item in profile.properties
        if item.required_for_field_claim and item.property_name not in reported
    ]
    return MaterialDomainPlan(
        resolution=resolution,
        main_model_run=(
            model_run if resolution.model_decision_id is not None else None
        ),
        profile=profile,
        stages=list(profile.stage_routes),
        problem_context=provided_context,
        missing_required_context=missing_context,
        field_route_ready=(
            not resolution.requires_operator_choice and not missing_context
        ),
        externally_reported_property_names=reported,
        unexecuted_required_properties=missing,
    )


def material_stage_route(
    material_field: MaterialField | str,
    stage: MaterialEvidenceStage | str,
) -> MaterialStageRoute:
    normalized_stage = str(stage)
    if normalized_stage not in MATERIAL_EVIDENCE_STAGES:
        raise ValueError(f"unknown material evidence stage: {stage!r}")
    profile = MATERIAL_FIELD_PROFILES[MaterialField(str(material_field))]
    route = next(item for item in profile.stage_routes if item.stage == normalized_stage)
    return MaterialStageRoute.model_validate_json(route.model_dump_json(), strict=True)


def assess_material_field_results(
    material_field: MaterialField | str,
    *,
    candidate_id: str,
    observations: Iterable[MaterialPropertyObservation],
    target_conditions: Mapping[str, JsonValue] | None = None,
) -> MaterialFieldResultAssessment:
    """Gate field-specific ranking without averaging incompatible evidence.

    Only results from a profile's named score-producing validators are
    accepted.  Unit or required-condition mismatches are incomparable, failed
    executions remain unknown, and disagreeing successful values are preserved
    as a conflict.
    """

    profile = get_material_field_profile(material_field)
    rows = list(observations)
    targets = dict(target_conditions or {})
    if _contains_sensitive_context_key(targets):
        raise ValueError("material field target conditions cannot contain secrets")
    required_target_conditions = list(
        dict.fromkeys(
            context_name
            for requirement in profile.properties
            if requirement.required_for_field_claim
            for context_name in requirement.required_context
        )
    )
    missing_target_conditions = [
        name
        for name in required_target_conditions
        if _context_value_is_missing(targets.get(name))
    ]
    if any(item.candidate_id != candidate_id for item in rows):
        raise ValueError("all material property observations must match candidate_id")
    if any(item.material_field != profile.material_field for item in rows):
        raise ValueError("all observations must match the selected material field")
    allowed: dict[str, set[str]] = {}
    for route in profile.stage_routes:
        for validator in route.validators:
            if not validator.can_create_property_scores:
                continue
            for property_name in validator.properties:
                allowed.setdefault(property_name, set()).add(validator.validator_id)

    decisions: list[MaterialPropertyDecision] = []
    for requirement in profile.properties:
        relevant = [
            item for item in rows if item.property_name == requirement.property_name
        ]
        accepted: list[MaterialPropertyObservation] = []
        rejected_ids: list[str] = []
        incomparable = False
        incompatible_condition_sets = False
        for item in relevant:
            validator_allowed = item.validator_id in allowed.get(
                requirement.property_name,
                set(),
            )
            conditions_complete = all(
                not _context_value_is_missing(item.conditions.get(name))
                for name in requirement.required_context
            )
            target_matches = all(
                name not in targets
                or _stable_json_value(item.conditions.get(name))
                == _stable_json_value(targets[name])
                for name in requirement.required_context
            )
            if (
                not validator_allowed
                or item.unit != requirement.unit
                or not conditions_complete
                or not target_matches
            ):
                rejected_ids.append(item.observation_id)
                incomparable = True
                continue
            if item.status == "success":
                accepted.append(item)
            elif item.status == "incomparable":
                incomparable = True
            else:
                rejected_ids.append(item.observation_id)

        condition_groups: dict[str, list[MaterialPropertyObservation]] = {}
        for item in accepted:
            normalized_conditions = {
                name: item.conditions[name]
                for name in requirement.required_context
            }
            condition_groups.setdefault(
                _stable_json_value(normalized_conditions),
                [],
            ).append(item)
        if len(condition_groups) > 1:
            rejected_ids.extend(item.observation_id for item in accepted)
            accepted = []
            incomparable = True
            incompatible_condition_sets = True

        serialized_values = {
            _stable_json_value(item.value) for item in accepted
        }
        if len(serialized_values) > 1:
            status: Literal["available", "unknown", "incomparable", "conflicting"] = (
                "conflicting"
            )
            reason = (
                "Named validators returned different normalized values; no averaging "
                "or automatic winner selection was performed."
            )
        elif accepted:
            status = "available"
            reason = (
                "At least one named validator returned a unit- and condition-complete "
                "result. This permits computational ranking only."
            )
        elif incomparable:
            status = "incomparable"
            reason = (
                "Successful rows describe different operating-condition sets; "
                "they cannot be merged or treated as conflicting measurements."
                if incompatible_condition_sets
                else (
                    "Available rows used an unapproved validator, incompatible "
                    "unit, missing required condition, or a condition different "
                    "from the requested target."
                )
            )
        else:
            status = "unknown"
            reason = "No successful named validator result is available."
        accepted_conditions = (
            {
                name: accepted[0].conditions[name]
                for name in requirement.required_context
            }
            if accepted
            else {}
        )
        decisions.append(
            MaterialPropertyDecision(
                property_name=requirement.property_name,
                status=status,
                accepted_observation_ids=[
                    item.observation_id for item in accepted
                ],
                rejected_observation_ids=list(dict.fromkeys(rejected_ids)),
                accepted_conditions=accepted_conditions,
                reason=reason,
            )
        )
    return MaterialFieldResultAssessment(
        candidate_id=candidate_id,
        material_field=profile.material_field,
        profile_id=profile.profile_id,
        decisions=decisions,
        target_conditions=targets,
        missing_target_conditions=missing_target_conditions,
        ready_for_field_computational_ranking=(
            not missing_target_conditions
            and all(item.status == "available" for item in decisions)
        ),
    )


def build_main_model_material_field_classifier_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    required: bool = False,
) -> MainModelMaterialFieldClassifier | None:
    """Build the optional main-AI classifier from an OpenAI-compatible endpoint.

    Dedicated ``MATERIAL_FIELD_MODEL_*`` settings take precedence.  Existing
    ``RAG_MODEL_*`` settings are reused when dedicated settings are absent so a
    notebook needs only one trusted reasoning endpoint.  No literature or MCP
    result is consulted during this pre-retrieval classification.
    """

    values = os.environ if environ is None else environ
    dedicated_base_url = str(
        values.get("MATERIAL_FIELD_MODEL_API_URL") or ""
    ).strip()
    dedicated_model_name = str(
        values.get("MATERIAL_FIELD_MODEL_NAME") or ""
    ).strip()
    use_dedicated = bool(dedicated_base_url or dedicated_model_name)
    if use_dedicated and bool(dedicated_base_url) != bool(dedicated_model_name):
        raise ValueError(
            "MATERIAL_FIELD_MODEL_API_URL and MATERIAL_FIELD_MODEL_NAME must "
            "be configured together"
        )
    base_url = (
        dedicated_base_url
        if use_dedicated
        else str(values.get("RAG_MODEL_API_URL") or "").strip()
    )
    model_name = (
        dedicated_model_name
        if use_dedicated
        else str(values.get("RAG_MODEL_NAME") or "").strip()
    )
    if bool(base_url) != bool(model_name):
        raise ValueError(
            "material field model API URL and model name must be configured together"
        )
    if not base_url:
        if required:
            raise ValueError(
                "main-AI material field routing requires MATERIAL_FIELD_MODEL_API_URL/"
                "MATERIAL_FIELD_MODEL_NAME or RAG_MODEL_API_URL/RAG_MODEL_NAME"
            )
        return None
    api_key = str(
        (
            values.get("MATERIAL_FIELD_MODEL_API_KEY")
            if use_dedicated
            else values.get("RAG_MODEL_API_KEY")
        )
        or ""
    ).strip()
    timeout = float(
        str(
            (
                values.get("MATERIAL_FIELD_MODEL_TIMEOUT_SECONDS")
                if use_dedicated
                else values.get("RAG_MODEL_TIMEOUT_SECONDS")
            )
            or "180"
        )
    )
    if not 1.0 <= timeout <= 600.0:
        raise ValueError(
            "material field model timeout must be between 1 and 600 seconds"
        )
    from .literature_rag import OpenAICompatibleRagModel

    return MainModelMaterialFieldClassifier(
        OpenAICompatibleRagModel(
            base_url,
            model_name,
            api_key=api_key or None,
            timeout=timeout,
        )
    )


def _normalize(value: str) -> str:
    return re.sub(r"[\s_-]+", " ", value.casefold()).strip()


def _normalize_evidence_text(value: str) -> str:
    """Normalize case and whitespace without changing quoted punctuation."""

    return re.sub(r"\s+", " ", value.casefold()).strip()


def _contains_term(haystack: str, term: str) -> bool:
    if not term:
        return False
    if re.search(r"[가-힣]", term):
        return term in haystack
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", haystack) is not None


def _context_value_is_missing(value: JsonValue | None) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, dict)):
        return not value
    return False


def _contains_sensitive_context_key(value: object) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(key).casefold(),
            ).strip("_")
            if any(
                marker in normalized
                for marker in (
                    "api_key",
                    "access_key",
                    "private_key",
                    "client_secret",
                    "token",
                    "secret",
                    "password",
                    "credential",
                    "authorization",
                    "bearer",
                )
            ):
                return True
            if _contains_sensitive_context_key(child):
                return True
    elif isinstance(value, list):
        return any(_contains_sensitive_context_key(item) for item in value)
    return False


def _stable_json_value(value: JsonValue) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


__all__ = [
    "BATTERY_ELECTRODE_PROFILE",
    "DomainValidatorSpec",
    "FERROELECTRIC_PIEZOELECTRIC_PROFILE",
    "GENERAL_INORGANIC_PROFILE",
    "HETEROGENEOUS_CATALYST_PROFILE",
    "MAGNETIC_MATERIAL_PROFILE",
    "MATERIAL_EVIDENCE_STAGES",
    "MATERIAL_FIELD_PROFILES",
    "MaterialDomainPlan",
    "MaterialFieldProfile",
    "MaterialFieldResolution",
    "MaterialFieldResultAssessment",
    "MaterialFieldModelDecision",
    "MaterialFieldModelRun",
    "MainModelMaterialFieldClassifier",
    "MaterialPropertyRequirement",
    "MaterialPropertyObservation",
    "MaterialPropertyDecision",
    "MaterialStageRoute",
    "PHOTOVOLTAIC_ABSORBER_PROFILE",
    "POROUS_FRAMEWORK_PROFILE",
    "SEMICONDUCTOR_PROFILE",
    "SOLID_ELECTROLYTE_PROFILE",
    "STRUCTURAL_ALLOY_PROFILE",
    "SUPERCONDUCTOR_PROFILE",
    "THERMOELECTRIC_PROFILE",
    "build_material_domain_plan",
    "build_main_model_material_field_classifier_from_environment",
    "assess_material_field_results",
    "get_material_field_profile",
    "material_stage_route",
    "resolve_material_field",
]
