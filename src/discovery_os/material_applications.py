"""Code-owned application and component routing for material selection.

This layer sits above :mod:`discovery_os.material_domains`.  A broad field is
not a device component: a semiconductor channel, gate dielectric, contact,
interconnect, and heat spreader require different properties, conditions, and
authoritative validators.  The main reasoning model may propose a bounded
intent and component set, but code owns every role, criterion, evidence route,
and validator allowlist.

The profiles intentionally contain *retrieval seeds*, not recommendations.
They help RAG find incumbents, alternatives, negative results, and validation
methods.  A seed never receives a material-performance score.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, TypeAlias

from pydantic import Field, model_validator

from .hashing import stable_hash
from .material_domains import (
    MATERIAL_FIELD_PROFILES,
    JsonFieldClassificationModel,
    MaterialDomainPlan,
    MaterialEvidenceStage,
    MaterialFieldModelRun,
    _contains_sensitive_context_key,
    _context_value_is_missing,
    _normalize_evidence_text,
    build_main_model_material_field_classifier_from_environment,
    build_material_domain_plan,
)
from .schemas import (
    Identifier,
    JsonValue,
    MaterialField,
    NonEmptyText,
    Probability,
    StrictSchema,
)


ApplicationQuestionKind = Literal[
    "component_map",
    "component_selection",
    "novel_material_discovery",
    "compare_given_candidates",
]
ApplicationDecompositionMode = Literal[
    "single-role",
    "role-portfolio",
    "needs-clarification",
]
CriterionDirection = Literal[
    "maximize",
    "minimize",
    "target",
    "range",
    "user_defined",
]
CriterionCategory = Literal[
    "performance",
    "reliability",
    "integration",
    "resource_safety",
]
RepresentationScope = Literal[
    "bulk_crystal",
    "thin_film",
    "interface_stack",
    "patterned_device",
    "composite",
]
ApplicationEvidenceCategory = Literal[
    "requirements_and_metrics",
    "incumbents_and_tradeoffs",
    "candidate_evidence",
    "negative_and_failure_evidence",
    "validation_and_reproducibility",
]


class ApplicationCriterion(StrictSchema):
    criterion_id: Identifier
    property_name: Identifier
    unit: NonEmptyText
    category: CriterionCategory
    direction: CriterionDirection
    required_for_ranking: bool = True
    required_context: list[Identifier] = Field(default_factory=list)
    validator_ids: list[Identifier] = Field(min_length=1)
    preferred_calculations: list[NonEmptyText] = Field(default_factory=list)
    experimental_confirmation: list[NonEmptyText] = Field(min_length=1)
    scientific_caution: NonEmptyText
    literature_or_mcp_can_score: Literal[False] = False

    @model_validator(mode="after")
    def _criterion_is_unambiguous(self) -> "ApplicationCriterion":
        if len(self.required_context) != len(set(self.required_context)):
            raise ValueError("criterion required context must be unique")
        if len(self.validator_ids) != len(set(self.validator_ids)):
            raise ValueError("criterion validator identifiers must be unique")
        return self


class ApplicationCandidateSeed(StrictSchema):
    seed_id: Identifier
    material_family: NonEmptyText
    examples: list[NonEmptyText] = Field(min_length=1)
    rationale: NonEmptyText
    research_reference_ids: list[NonEmptyText] = Field(min_length=1)
    scientific_role: Literal["retrieval-seed-not-ranked-result"] = (
        "retrieval-seed-not-ranked-result"
    )
    performance_score: None = None


class ApplicationEvidenceTask(StrictSchema):
    task_id: Identifier
    role_id: Identifier
    category: ApplicationEvidenceCategory
    evidence_stage: MaterialEvidenceStage
    questions: list[NonEmptyText] = Field(min_length=1)
    allowed_literature_sources: list[
        Literal["crossref", "arxiv", "openalex"]
    ] = Field(default_factory=lambda: ["crossref", "arxiv", "openalex"])
    mcp_capabilities: list[Identifier] = Field(default_factory=list)
    required_record_fields: list[Identifier] = Field(min_length=1)
    scientific_role: Literal["search-and-validation-context-only"] = (
        "search-and-validation-context-only"
    )
    can_create_property_scores: Literal[False] = False
    prompt_or_model_can_choose_mcp_tool: Literal[False] = False

    @model_validator(mode="after")
    def _task_fields_are_unique(self) -> "ApplicationEvidenceTask":
        if len(self.questions) != len(set(self.questions)):
            raise ValueError("application evidence questions must be unique")
        if len(self.mcp_capabilities) != len(set(self.mcp_capabilities)):
            raise ValueError("application MCP capabilities must be unique")
        if len(self.required_record_fields) != len(
            set(self.required_record_fields)
        ):
            raise ValueError("required application evidence fields must be unique")
        return self


class ApplicationRoleProfile(StrictSchema):
    role_id: Identifier
    profile_version: Literal["1.0"] = "1.0"
    material_field: MaterialField
    display_name: NonEmptyText
    description: NonEmptyText
    aliases: list[NonEmptyText] = Field(min_length=1)
    representation_scopes: list[RepresentationScope] = Field(min_length=1)
    bulk_cif_scope: Literal[
        "can-screen-bulk-only",
        "insufficient-interface-or-device-required",
    ]
    required_problem_context: list[Identifier] = Field(default_factory=list)
    criteria: list[ApplicationCriterion] = Field(min_length=1)
    candidate_seeds: list[ApplicationCandidateSeed] = Field(default_factory=list)
    evidence_tasks: list[ApplicationEvidenceTask] = Field(min_length=5, max_length=5)
    failure_modes: list[NonEmptyText] = Field(min_length=1)
    research_reference_ids: list[NonEmptyText] = Field(min_length=1)
    claim_boundary: NonEmptyText

    @model_validator(mode="after")
    def _role_profile_is_closed(self) -> "ApplicationRoleProfile":
        if len(self.aliases) != len(set(self.aliases)):
            raise ValueError("application role aliases must be unique")
        if len(self.representation_scopes) != len(
            set(self.representation_scopes)
        ):
            raise ValueError("application representation scopes must be unique")
        if len(self.required_problem_context) != len(
            set(self.required_problem_context)
        ):
            raise ValueError("application required context must be unique")
        criterion_ids = [item.criterion_id for item in self.criteria]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("application criterion identifiers must be unique")
        seed_ids = [item.seed_id for item in self.candidate_seeds]
        if len(seed_ids) != len(set(seed_ids)):
            raise ValueError("application seed identifiers must be unique")
        categories = [item.category for item in self.evidence_tasks]
        expected: list[ApplicationEvidenceCategory] = [
            "requirements_and_metrics",
            "incumbents_and_tradeoffs",
            "candidate_evidence",
            "negative_and_failure_evidence",
            "validation_and_reproducibility",
        ]
        if categories != expected:
            raise ValueError(
                "application role needs the five ordered evidence task categories"
            )
        expected_stages: list[MaterialEvidenceStage] = [
            "generation_prior",
            "identity_novelty",
            "mlip_disagreement",
            "relaxation_validation",
            "dft_handoff",
        ]
        if [item.evidence_stage for item in self.evidence_tasks] != expected_stages:
            raise ValueError(
                "application evidence tasks must follow the five validation stages"
            )
        route_by_stage = {
            route.stage: route
            for route in MATERIAL_FIELD_PROFILES[
                MaterialField(str(self.material_field))
            ].stage_routes
        }
        for task in self.evidence_tasks:
            expected_sources = (
                ["crossref", "arxiv", "openalex"]
                if task.evidence_stage
                in {"generation_prior", "identity_novelty"}
                else ["crossref", "arxiv"]
            )
            if task.allowed_literature_sources != expected_sources:
                raise ValueError(
                    "application evidence sources must follow the stage policy"
                )
            if task.mcp_capabilities != list(
                route_by_stage[task.evidence_stage].mcp_capabilities
            ):
                raise ValueError(
                    "application MCP capabilities must match the selected stage route"
                )
        if any(item.role_id != self.role_id for item in self.evidence_tasks):
            raise ValueError("application evidence task role does not match profile")
        if (
            self.bulk_cif_scope == "insufficient-interface-or-device-required"
            and self.representation_scopes == ["bulk_crystal"]
        ):
            raise ValueError("an interface/device role cannot be bulk-crystal only")
        return self


class MaterialApplicationModelDecision(StrictSchema):
    """Untrusted main-AI proposal; code validates every selected role."""

    question_kind: ApplicationQuestionKind
    selected_role_ids: list[Identifier] = Field(default_factory=list, max_length=16)
    application_subtype: Identifier | None = None
    extracted_context: dict[str, JsonValue] = Field(default_factory=dict)
    objective_priorities: list[Identifier] = Field(default_factory=list, max_length=32)
    confidence: Probability
    evidence_spans: list[NonEmptyText] = Field(min_length=1, max_length=16)
    needs_clarification: bool = False
    clarification_question: str | None = Field(default=None, max_length=1_000)
    decision_summary: NonEmptyText
    endpoint_or_tool_selection_performed: Literal[False] = False

    @model_validator(mode="after")
    def _model_decision_is_bounded(self) -> "MaterialApplicationModelDecision":
        if _contains_sensitive_context_key(self.extracted_context):
            raise ValueError("application model context cannot contain secrets")
        if len(self.selected_role_ids) != len(set(self.selected_role_ids)):
            raise ValueError("application model role identifiers must be unique")
        if len(self.objective_priorities) != len(set(self.objective_priorities)):
            raise ValueError("application model objectives must be unique")
        if self.needs_clarification != bool(self.clarification_question):
            raise ValueError(
                "clarification question must be present exactly when required"
            )
        if not self.selected_role_ids and not self.needs_clarification:
            raise ValueError("application model must select a role or clarify")
        normalized = [_normalize_evidence_text(item) for item in self.evidence_spans]
        if any(len(item) < 2 or len(item) > 500 for item in normalized):
            raise ValueError("application evidence spans must contain 2 to 500 chars")
        if len(normalized) != len(set(normalized)):
            raise ValueError("application evidence spans must be unique")
        return self


class MaterialApplicationModelRun(StrictSchema):
    decision_id: Identifier
    model_id: Identifier
    model_version: Identifier
    material_field: MaterialField
    prompt_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: MaterialApplicationModelDecision
    evidence_spans_verified: Literal[True] = True
    role_allowlist_verified: Literal[True] = True
    endpoint_or_tool_selection_performed: Literal[False] = False


class MaterialApplicationBrief(StrictSchema):
    brief_id: Identifier
    user_question: NonEmptyText
    material_field: MaterialField
    question_kind: ApplicationQuestionKind
    decomposition_mode: ApplicationDecompositionMode
    field_plan: MaterialDomainPlan
    main_application_model_run: MaterialApplicationModelRun | None = None
    roles: list[ApplicationRoleProfile] = Field(min_length=1)
    target_context: dict[str, JsonValue] = Field(default_factory=dict)
    missing_context_by_role: dict[str, list[Identifier]] = Field(default_factory=dict)
    candidate_seeds_by_role: dict[str, list[ApplicationCandidateSeed]] = Field(
        default_factory=dict
    )
    evidence_tasks: list[ApplicationEvidenceTask] = Field(min_length=5)
    clarification_question: str | None = Field(default=None, max_length=1_000)
    ready_for_condition_complete_scoring: bool = False
    cross_role_ranking_allowed: Literal[False] = False
    prompt_or_model_selected_validator: Literal[False] = False
    scientific_status: Literal[
        "application-routing-and-evidence-plan-only"
    ] = "application-routing-and-evidence-plan-only"

    @model_validator(mode="after")
    def _brief_matches_roles(self) -> "MaterialApplicationBrief":
        if _contains_sensitive_context_key(self.target_context):
            raise ValueError("application target context cannot contain secrets")
        if self.field_plan.resolution.selected_field != self.material_field:
            raise ValueError("application brief field does not match domain plan")
        role_ids = [item.role_id for item in self.roles]
        if len(role_ids) != len(set(role_ids)):
            raise ValueError("application brief roles must be unique")
        if any(item.material_field != self.material_field for item in self.roles):
            raise ValueError("application brief cannot mix material fields")
        if set(self.missing_context_by_role) != set(role_ids):
            raise ValueError("missing-context map must cover every selected role")
        if set(self.candidate_seeds_by_role) != set(role_ids):
            raise ValueError("candidate-seed map must cover every selected role")
        expected_tasks = [
            task for role in self.roles for task in role.evidence_tasks
        ]
        if self.evidence_tasks != expected_tasks:
            raise ValueError("application evidence tasks must follow selected roles")
        expected_missing = {
            role.role_id: [
                name
                for name in _role_required_context(role)
                if _context_value_is_missing(self.target_context.get(name))
            ]
            for role in self.roles
        }
        if self.missing_context_by_role != expected_missing:
            raise ValueError("missing application context must be explicit and ordered")
        expected_ready = (
            self.decomposition_mode != "needs-clarification"
            and not self.field_plan.resolution.requires_operator_choice
            and not bool(
                self.main_application_model_run
                and self.main_application_model_run.decision.needs_clarification
            )
            and all(not names for names in expected_missing.values())
        )
        if self.ready_for_condition_complete_scoring != expected_ready:
            raise ValueError("application scoring readiness is inconsistent")
        if self.decomposition_mode == "needs-clarification":
            if not self.clarification_question:
                raise ValueError("needs-clarification brief requires a question")
        elif self.clarification_question:
            raise ValueError("ready brief cannot retain a clarification question")
        return self


class MainModelMaterialApplicationClassifier:
    """Use a reasoning model only for typed intent and role hypotheses."""

    def __init__(self, model: JsonFieldClassificationModel) -> None:
        self.model = model

    def classify(
        self,
        question: str,
        *,
        material_field: MaterialField | str,
        problem_context: Mapping[str, JsonValue] | None = None,
    ) -> MaterialApplicationModelRun:
        question = question.strip()
        if not question:
            raise ValueError("material application question cannot be empty")
        field = MaterialField(str(material_field))
        context = dict(problem_context or {})
        if _contains_sensitive_context_key(context):
            raise ValueError("application classifier context cannot contain secrets")
        roles = application_roles_for_field(field)
        role_payload = {
            role.role_id: {
                "description": role.description,
                "aliases": role.aliases,
                "required_problem_context": role.required_problem_context,
                "criteria": [
                    {
                        "criterion_id": criterion.criterion_id,
                        "property_name": criterion.property_name,
                        "unit": criterion.unit,
                    }
                    for criterion in role.criteria
                ],
                "claim_boundary": role.claim_boundary,
            }
            for role in roles
        }
        payload = self.model.complete_json(
            operation="classify-material-application",
            system=(
                "Classify a material-selection question into the supplied code-owned "
                "component roles. Return JSON only. A broad question asking which "
                "materials belong in which parts is component_map and should retain "
                "several roles. A specific component is component_selection. Separate "
                "known-material comparison from novel-material discovery. Quote exact "
                "input spans. Extract only conditions literally present in the input. "
                "Request clarification when a single-component decision lacks essential "
                "application conditions. Never choose an API, RAG provider, MCP endpoint "
                "or tool, calculation engine, validator, score, or pass/fail result."
            ),
            user=json.dumps(
                {
                    "question": question,
                    "material_field": str(field),
                    "problem_context": context,
                    "allowed_roles": role_payload,
                    "required_output": {
                        "question_kind": (
                            "component_map | component_selection | "
                            "novel_material_discovery | compare_given_candidates"
                        ),
                        "selected_role_ids": ["allowed role id"],
                        "application_subtype": "short label or null",
                        "extracted_context": {
                            "only_code_or_user_supplied_condition_name": "JSON value"
                        },
                        "objective_priorities": ["criterion id from selected roles"],
                        "confidence": "number from 0 to 1",
                        "evidence_spans": ["exact quote from question or context"],
                        "needs_clarification": "boolean",
                        "clarification_question": "string or null",
                        "decision_summary": "short routing summary",
                        "endpoint_or_tool_selection_performed": False,
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        )
        try:
            decision = MaterialApplicationModelDecision.model_validate_json(
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
                "main model returned an invalid material-application decision"
            ) from exc
        allowed_roles = {item.role_id: item for item in roles}
        _validate_application_model_decision(
            decision,
            material_field=field,
            roles_by_id=allowed_roles,
            question=question,
            input_context=context,
        )
        decision_payload = {
            "model_id": self.model.model_id,
            "model_version": self.model.model_version,
            "question": question,
            "material_field": field,
            "problem_context": context,
            "decision": decision,
        }
        return MaterialApplicationModelRun(
            decision_id=f"MADEC-{stable_hash(decision_payload)[:24]}",
            model_id=self.model.model_id,
            model_version=self.model.model_version,
            material_field=field,
            prompt_hash=stable_hash(
                {
                    "question": question,
                    "material_field": field,
                    "problem_context": context,
                }
            ),
            decision=decision,
        )


def _criterion(
    criterion_id: str,
    property_name: str,
    unit: str,
    category: CriterionCategory,
    direction: CriterionDirection,
    context: Sequence[str],
    validators: Sequence[str],
    calculations: Sequence[str],
    experiments: Sequence[str],
    caution: str,
    *,
    required: bool = True,
) -> ApplicationCriterion:
    return ApplicationCriterion(
        criterion_id=criterion_id,
        property_name=property_name,
        unit=unit,
        category=category,
        direction=direction,
        required_for_ranking=required,
        required_context=list(context),
        validator_ids=list(validators),
        preferred_calculations=list(calculations),
        experimental_confirmation=list(experiments),
        scientific_caution=caution,
    )


def _seed(
    seed_id: str,
    family: str,
    examples: Sequence[str],
    rationale: str,
    references: Sequence[str],
) -> ApplicationCandidateSeed:
    return ApplicationCandidateSeed(
        seed_id=seed_id,
        material_family=family,
        examples=list(examples),
        rationale=rationale,
        research_reference_ids=list(references),
    )


_REQUIRED_EVIDENCE_FIELDS = [
    "material_or_stack",
    "component_role",
    "property_name",
    "value",
    "unit",
    "complete_conditions",
    "geometry_or_thickness",
    "measured_or_calculated",
    "method",
    "sample_or_process",
    "uncertainty",
    "negative_or_null_result",
    "stable_source_id",
    "exact_support_span",
]


def _evidence_tasks(
    role_id: str,
    display_name: str,
    criteria: Sequence[ApplicationCriterion],
    *,
    material_field: MaterialField,
) -> list[ApplicationEvidenceTask]:
    properties = ", ".join(item.property_name for item in criteria)
    context = ", ".join(
        dict.fromkeys(
            name for item in criteria for name in item.required_context
        )
    )
    questions: list[
        tuple[ApplicationEvidenceCategory, MaterialEvidenceStage, str]
    ] = [
        (
            "requirements_and_metrics",
            "generation_prior",
            (
                f"For {display_name}, which application-specific metrics, units, "
                f"operating conditions, geometry, and comparison rules are required "
                f"for {properties}? Include {context or 'declared use conditions'}."
            ),
        ),
        (
            "incumbents_and_tradeoffs",
            "identity_novelty",
            (
                f"For {display_name}, identify incumbent and alternative material or "
                "stack identities, aliases, polymorphs, processing states, and external "
                "database matches. Preserve scoped no-match as unknown, not novelty."
            ),
        ),
        (
            "candidate_evidence",
            "mlip_disagreement",
            (
                f"For candidate structures considered for {display_name}, retrieve "
                "model cards, benchmark domains, and bonding, charge, spin, surface, "
                "or interface limitations relevant to interpreting separate MLIP "
                "outputs. Do not average models or infer a property value."
            ),
        ),
        (
            "negative_and_failure_evidence",
            "relaxation_validation",
            (
                f"Find negative, null, reliability, degradation, interface, processing, "
                f"toxicity, scarcity, and failed-integration evidence for {display_name}."
            ),
        ),
        (
            "validation_and_reproducibility",
            "dft_handoff",
            (
                f"Find authoritative calculations, measurements, standards, controls, "
                f"and reproducibility requirements needed to validate {display_name}; "
                "distinguish bulk, film, interface, and patterned-device authority."
            ),
        ),
    ]
    route_by_stage = {
        route.stage: route
        for route in MATERIAL_FIELD_PROFILES[material_field].stage_routes
    }
    if set(route_by_stage) != {
        "generation_prior",
        "identity_novelty",
        "mlip_disagreement",
        "relaxation_validation",
        "dft_handoff",
    }:
        raise ValueError("material field must define exactly the five evidence stages")
    return [
        ApplicationEvidenceTask(
            task_id=f"{role_id}-{category}",
            role_id=role_id,
            category=category,
            evidence_stage=stage,
            questions=[question],
            allowed_literature_sources=(
                ["crossref", "arxiv", "openalex"]
                if stage in {"generation_prior", "identity_novelty"}
                else ["crossref", "arxiv"]
            ),
            mcp_capabilities=list(route_by_stage[stage].mcp_capabilities),
            required_record_fields=list(_REQUIRED_EVIDENCE_FIELDS),
        )
        for category, stage, question in questions
    ]


def _role(
    *,
    role_id: str,
    field: MaterialField,
    display_name: str,
    description: str,
    aliases: Sequence[str],
    scopes: Sequence[RepresentationScope],
    bulk_cif_scope: Literal[
        "can-screen-bulk-only",
        "insufficient-interface-or-device-required",
    ],
    context: Sequence[str],
    criteria: Sequence[ApplicationCriterion],
    seeds: Sequence[ApplicationCandidateSeed],
    capabilities: Sequence[str],
    failure_modes: Sequence[str],
    references: Sequence[str],
    boundary: str,
) -> ApplicationRoleProfile:
    return ApplicationRoleProfile(
        role_id=role_id,
        material_field=field,
        display_name=display_name,
        description=description,
        aliases=list(aliases),
        representation_scopes=list(scopes),
        bulk_cif_scope=bulk_cif_scope,
        required_problem_context=list(context),
        criteria=list(criteria),
        candidate_seeds=list(seeds),
        evidence_tasks=_evidence_tasks(
            role_id,
            display_name,
            criteria,
            material_field=field,
        ),
        failure_modes=list(failure_modes),
        research_reference_ids=list(references),
        claim_boundary=boundary,
    )


IRDS_MM = "https://irds.ieee.org/images/files/pdf/2024/2024IRDS_MM.pdf"
IRDS_BC = "https://irds.ieee.org/images/files/pdf/2024/2024IRDS_BC.pdf"
NIST_FET = (
    "https://www.nist.gov/publications/"
    "how-report-and-benchmark-emerging-field-effect-transistors"
)
POWER_SURVEY = "https://www.osti.gov/biblio/1568046"
TRANSPARENT_REVIEW = "https://www.nature.com/articles/nphoton.2012.282"
CONTACT_REVIEW = "https://www.nature.com/articles/nmat4452"
HIGH_K_NIST = (
    "https://www.nist.gov/publications/"
    "challenges-high-kappa-gate-dielectrics-future-mos-devices"
)
BA_THERMAL = "https://pubmed.ncbi.nlm.nih.gov/29976796/"
NIST_THERMAL = (
    "https://www.nist.gov/programs-projects/"
    "thermoreflectance-thermal-property-measurements-heterogeneously-integrated"
)
NIST_RESPONSIVITY = (
    "https://www.nist.gov/programs-projects/spectral-responsivity-measurement"
)


def _semiconductor_roles() -> tuple[ApplicationRoleProfile, ...]:
    field = MaterialField.SEMICONDUCTOR
    roles: list[ApplicationRoleProfile] = []

    logic = [
        _criterion(
            "logic-channel-band-gap",
            "band_gap",
            "eV",
            "performance",
            "target",
            ("temperature", "strain", "channel_thickness"),
            (
                "hybrid-or-gw-electronic-structure",
                "optical-and-photoemission-characterization",
            ),
            ("HSE/GW plus SOC when relevant",),
            ("Optical/photoemission band-edge characterization",),
            "Band gap alone cannot predict leakage, electrostatics, or drive current.",
        ),
        _criterion(
            "logic-channel-mobility",
            "carrier_mobility",
            "cm^2/(V s)",
            "performance",
            "maximize",
            ("carrier_type", "temperature", "carrier_density", "channel_thickness"),
            (
                "electron-phonon-transport-workflow",
                "hall-or-field-effect-mobility-measurement",
            ),
            ("Electron-phonon BTE at matched density and geometry",),
            ("Hall/FET mobility with density, capacitance, and geometry",),
            "Mobility from unmatched geometry or density is incomparable.",
        ),
        _criterion(
            "logic-channel-contact-resistance",
            "contact_resistance",
            "ohm micrometer",
            "integration",
            "minimize",
            (
                "carrier_type",
                "carrier_density",
                "contact_length",
                "temperature",
            ),
            ("device-negf-contact-screen", "multi-length-tlm-measurement"),
            ("Interface DFT/NEGF as screening only",),
            ("At least four-length TLM or equivalent Kelvin structure",),
            "A work function or ideal Schottky barrier is not contact resistance.",
        ),
        _criterion(
            "logic-channel-ion-ioff",
            "ion_ioff_ratio",
            "dimensionless",
            "performance",
            "maximize",
            ("vdd", "temperature", "gate_length", "ioff_definition"),
            ("device-tcad-or-negf", "transistor-transfer-measurement"),
            ("Calibrated TCAD or NEGF with the full stack",),
            ("Bidirectional ID-VG/ID-VD with leakage and device statistics",),
            "A bulk crystal calculation cannot establish transistor Ion/Ioff.",
        ),
    ]
    roles.append(
        _role(
            role_id="logic_channel",
            field=field,
            display_name="logic transistor channel",
            description=(
                "Channel material for scaled nFET, pFET, GAA, CFET, or related "
                "logic architectures."
            ),
            aliases=(
                "logic channel",
                "transistor channel",
                "nFET",
                "pFET",
                "GAA",
                "CFET",
                "채널",
                "로직",
                "트랜지스터",
            ),
            scopes=("bulk_crystal", "thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=(
                "carrier_type",
                "device_architecture",
                "gate_length",
                "vdd",
                "temperature",
                "carrier_density",
                "channel_thickness",
                "contact_length",
                "ioff_definition",
            ),
            criteria=logic,
            seeds=(
                _seed(
                    "logic-si",
                    "silicon and strained silicon",
                    ("Si", "strained Si"),
                    "Manufacturing-mature baseline that anchors device comparisons.",
                    (IRDS_MM, NIST_FET),
                ),
                _seed(
                    "logic-ge-iii-v",
                    "Ge/SiGe and III-V channels",
                    ("Ge", "SiGe", "InGaAs"),
                    "High-transport alternatives with polarity and integration trade-offs.",
                    (IRDS_MM, NIST_FET),
                ),
                _seed(
                    "logic-low-dimensional",
                    "low-dimensional channels",
                    ("MoS2", "WS2", "WSe2", "semiconducting CNT"),
                    "Electrostatically thin research candidates whose contacts and "
                    "wafer-scale integration require separate validation.",
                    (IRDS_BC, NIST_FET),
                ),
            ),
            capabilities=(
                "electronic-materials-database-search",
                "device-benchmark-evidence-search",
                "interface-and-contact-evidence-search",
            ),
            failure_modes=(
                "contact resistance hides intrinsic channel transport",
                "mobility is compared at unmatched density or geometry",
                "dielectric/interface traps degrade device behavior",
                "wafer-scale integration or variability is unproven",
            ),
            references=(IRDS_MM, IRDS_BC, NIST_FET),
            boundary=(
                "A bulk band gap, effective mass, or mobility estimate is not a "
                "logic-device recommendation; the full channel/contact/gate stack "
                "and matched device measurements are required."
            ),
        )
    )

    power = [
        _criterion(
            "power-critical-field",
            "critical_breakdown_field",
            "MV/cm",
            "performance",
            "maximize",
            ("temperature", "doping", "leakage_criterion", "device_geometry"),
            ("high-field-impact-ionization-workflow", "breakdown-measurement"),
            ("Impact-ionization/high-field transport model",),
            ("Breakdown test with geometry and leakage criterion",),
            "Empirical gap-to-field estimates are screening proxies, not breakdown.",
        ),
        _criterion(
            "power-mobility",
            "carrier_mobility",
            "cm^2/(V s)",
            "performance",
            "maximize",
            ("carrier_type", "temperature", "doping"),
            ("electron-phonon-transport-workflow", "hall-mobility-measurement"),
            ("Electron-phonon BTE at target doping and temperature",),
            ("Hall mobility versus doping and temperature",),
            "Mobility must be matched to the intended carrier, doping, and temperature.",
        ),
        _criterion(
            "power-thermal-conductivity",
            "lattice_thermal_conductivity",
            "W/(m K)",
            "reliability",
            "maximize",
            ("temperature", "crystal_orientation"),
            ("anharmonic-phonon-bte", "thermal-conductivity-measurement"),
            ("Anharmonic phonon BTE",),
            ("TDTR/3-omega/laser-flash with orientation and temperature",),
            "Bulk thermal conductivity does not determine packaged junction temperature.",
        ),
        _criterion(
            "power-dynamic-ron",
            "dynamic_static_ron_ratio",
            "dimensionless",
            "reliability",
            "minimize",
            (
                "temperature",
                "blocking_voltage",
                "switching_frequency",
                "pulse_protocol",
            ),
            ("trapping-device-model", "pulsed-dynamic-ron-measurement"),
            ("Calibrated trap-aware device model",),
            ("Pulsed I-V/double-pulse dynamic Ron",),
            "An intrinsic figure of merit omits dynamic trapping and packaging.",
        ),
    ]
    roles.append(
        _role(
            role_id="power_switch",
            field=field,
            display_name="power semiconductor switch",
            description="Active material for unipolar or bipolar power switching.",
            aliases=(
                "power semiconductor",
                "power switch",
                "MOSFET",
                "HEMT",
                "전력 반도체",
                "전력소자",
                "스위치",
            ),
            scopes=("bulk_crystal", "thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="can-screen-bulk-only",
            context=(
                "blocking_voltage",
                "current_density",
                "switching_frequency",
                "device_geometry",
                "temperature",
                "doping",
                "cooling_stack",
            ),
            criteria=power,
            seeds=(
                _seed(
                    "power-incumbents",
                    "incumbent power semiconductors",
                    ("Si", "4H-SiC", "GaN"),
                    "Mature comparison anchors spanning voltage, frequency, and thermal regimes.",
                    (POWER_SURVEY,),
                ),
                _seed(
                    "power-ultrawide-gap",
                    "ultrawide-band-gap research candidates",
                    ("beta-Ga2O3", "diamond", "c-BN", "AlN"),
                    "High intrinsic potential with doping, contacts, defects, or heat-removal gaps.",
                    (POWER_SURVEY,),
                ),
            ),
            capabilities=(
                "power-electronics-property-search",
                "high-field-reliability-evidence-search",
                "thermal-property-search",
            ),
            failure_modes=(
                "estimated critical field is reported as measured breakdown",
                "poor p-type doping or contact formation",
                "dynamic trapping increases on-resistance",
                "package thermal path erases bulk-material advantage",
            ),
            references=(POWER_SURVEY, IRDS_MM),
            boundary=(
                "Baliga or Johnson figures of merit are intrinsic screening metrics, "
                "not proof of device breakdown, dynamic losses, reliability, or packaging."
            ),
        )
    )

    transparent = [
        _criterion(
            "transparent-sheet-resistance",
            "sheet_resistance",
            "ohm/square",
            "performance",
            "minimize",
            ("film_thickness", "temperature", "deposition_process"),
            ("film-transport-model", "four-point-sheet-resistance"),
            ("Film transport with morphology and thickness",),
            ("Four-point probe and Hall measurement",),
            "Bulk conductivity cannot substitute for measured film sheet resistance.",
        ),
        _criterion(
            "transparent-transmittance",
            "spectral_transmittance",
            "fraction",
            "performance",
            "maximize",
            ("wavelength_range", "film_thickness", "substrate", "spectral_weighting"),
            ("optical-stack-model", "uv-vis-nir-spectrophotometry"),
            ("Optical dielectric function plus full stack model",),
            ("UV-Vis-NIR spectrum with substrate baseline",),
            "A band gap alone cannot establish transparency.",
        ),
        _criterion(
            "transparent-haze",
            "haze",
            "percent",
            "integration",
            "minimize",
            ("wavelength_range", "film_thickness", "substrate"),
            ("light-scattering-film-model", "haze-measurement"),
            ("Morphology-aware optical scattering",),
            ("Calibrated haze measurement",),
            "Haze requirements differ between display, photovoltaic, and heater use.",
        ),
        _criterion(
            "transparent-work-function",
            "work_function",
            "eV",
            "integration",
            "target",
            ("surface_condition", "measurement_environment"),
            ("surface-electronic-structure", "ups-or-kelvin-probe"),
            ("Surface/slab electronic structure",),
            ("UPS or calibrated Kelvin probe",),
            "Work function is surface- and process-dependent.",
            required=False,
        ),
    ]
    roles.append(
        _role(
            role_id="transparent_electrode",
            field=field,
            display_name="transparent conducting electrode",
            description="Conducting film that must meet an application-specific optical budget.",
            aliases=(
                "transparent conductor",
                "transparent electrode",
                "TCO",
                "투명 전극",
                "투명전극",
                "투명 도전",
            ),
            scopes=("thin_film", "interface_stack", "patterned_device", "composite"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=(
                "application",
                "wavelength_range",
                "spectral_weighting",
                "film_thickness",
                "substrate",
                "target_sheet_resistance",
                "deposition_process",
            ),
            criteria=transparent,
            seeds=(
                _seed(
                    "transparent-oxides",
                    "transparent conducting oxides",
                    ("ITO", "FTO", "AZO", "La:BaSnO3"),
                    "Established and emerging oxide-film baselines.",
                    (TRANSPARENT_REVIEW,),
                ),
                _seed(
                    "transparent-networks",
                    "conductive networks and ultrathin films",
                    ("Ag nanowire", "metal mesh", "ultrathin Ag", "graphene", "CNT"),
                    "Flexible or low-temperature alternatives with haze, roughness, "
                    "stability, and contact trade-offs.",
                    (TRANSPARENT_REVIEW,),
                ),
            ),
            capabilities=(
                "thin-film-optical-transport-search",
                "transparent-electrode-reliability-search",
            ),
            failure_modes=(
                "transmittance and resistance use different thicknesses",
                "spectral weighting is omitted",
                "film morphology or substrate baseline is missing",
                "flexibility or environmental stability is untested",
            ),
            references=(TRANSPARENT_REVIEW,),
            boundary=(
                "Bulk conductivity plus band gap is not a transparent-electrode "
                "score; matched film thickness, spectrum, sheet resistance, haze, "
                "substrate, processing, and reliability are required."
            ),
        )
    )

    contact = [
        _criterion(
            "contact-resistance",
            "contact_resistance",
            "ohm micrometer",
            "performance",
            "minimize",
            (
                "semiconductor",
                "carrier_type",
                "carrier_density",
                "contact_length",
                "temperature",
                "anneal_process",
            ),
            ("interface-negf-screen", "multi-length-tlm-measurement"),
            ("Interface DFT/NEGF as a screening model",),
            ("Scaled multi-length TLM/CBKR/Kelvin",),
            "Work function or ideal barrier height cannot replace contact resistance.",
        ),
        _criterion(
            "contact-specific-resistivity",
            "specific_contact_resistivity",
            "ohm cm^2",
            "performance",
            "minimize",
            (
                "semiconductor",
                "carrier_type",
                "carrier_density",
                "temperature",
                "anneal_process",
            ),
            ("contact-transport-screen", "specific-contact-resistivity-measurement"),
            ("Interface transport calculation",),
            ("TLM/CBKR with current-crowding correction",),
            "The extracted value is geometry- and model-dependent.",
        ),
        _criterion(
            "contact-thermal-stability",
            "contact_resistance_drift",
            "fraction",
            "reliability",
            "minimize",
            ("anneal_temperature", "anneal_time", "ambient"),
            ("interface-reaction-workflow", "contact-anneal-reliability-test"),
            ("Interface reaction/phase-stability screening",),
            ("Resistance plus TEM/XPS after anneal",),
            "An initially low resistance is insufficient without process stability.",
        ),
    ]
    roles.append(
        _role(
            role_id="source_drain_contact",
            field=field,
            display_name="source/drain electrical contact",
            description="Material and interface stack joining a semiconductor to circuitry.",
            aliases=(
                "source drain contact",
                "ohmic contact",
                "contact metal",
                "소스 드레인",
                "오믹 접촉",
                "접촉 금속",
                "콘택트",
            ),
            scopes=("interface_stack", "patterned_device", "thin_film"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=(
                "semiconductor",
                "surface_phase",
                "carrier_type",
                "carrier_density",
                "contact_length",
                "contact_geometry",
                "temperature",
                "anneal_process",
            ),
            criteria=contact,
            seeds=(
                _seed(
                    "contact-silicide-refractory",
                    "silicides and refractory contact stacks",
                    ("NiSi", "CoSi-family", "TiN", "W", "Mo"),
                    "Process-dependent incumbent families for Si and compound semiconductors.",
                    (IRDS_MM, CONTACT_REVIEW),
                ),
                _seed(
                    "contact-low-dimensional",
                    "low-dimensional and semimetal contacts",
                    ("Bi", "Sb", "metallic TMD", "graphene"),
                    "Research paths for reducing pinning in selected 2D interfaces.",
                    (CONTACT_REVIEW,),
                ),
            ),
            capabilities=(
                "semiconductor-contact-evidence-search",
                "interface-chemistry-search",
                "contact-metrology-search",
            ),
            failure_modes=(
                "Fermi-level pinning invalidates a work-function ranking",
                "interface reaction changes the intended phase",
                "current crowding corrupts contact extraction",
                "anneal or reliability drift is omitted",
            ),
            references=(IRDS_MM, CONTACT_REVIEW),
            boundary=(
                "A metal work function, isolated-interface DFT barrier, or bulk "
                "resistivity cannot establish a manufacturable low-resistance contact."
            ),
        )
    )

    interconnect = [
        _criterion(
            "interconnect-line-resistance",
            "line_resistance",
            "ohm/micrometer",
            "performance",
            "minimize",
            (
                "line_level",
                "line_width",
                "line_height",
                "line_length",
                "liner_barrier_thickness",
                "temperature",
            ),
            ("size-effect-interconnect-transport", "patterned-line-resistance"),
            ("Surface/grain-boundary size-effect transport",),
            ("Patterned line/via resistance at target dimensions",),
            "Bulk resistivity cannot rank nanometre interconnects.",
        ),
        _criterion(
            "interconnect-electromigration",
            "electromigration_lifetime",
            "hour",
            "reliability",
            "maximize",
            ("current_density", "temperature", "line_geometry", "test_protocol"),
            ("atomistic-diffusion-screen", "accelerated-electromigration-test"),
            ("Diffusion/cohesive screening",),
            ("Accelerated electromigration with failure criterion",),
            "A migration barrier alone omits microstructure and current crowding.",
        ),
        _criterion(
            "interconnect-fill-yield",
            "electrical_yield",
            "fraction",
            "integration",
            "maximize",
            ("patterning_process", "line_geometry", "sample_count"),
            ("process-fill-simulation", "via-chain-yield-measurement"),
            ("Process fill and stress simulation",),
            ("Via-chain/line yield statistics",),
            "An ideal crystal cannot establish pattern fill or manufacturing yield.",
        ),
    ]
    roles.append(
        _role(
            role_id="interconnect_conductor",
            field=field,
            display_name="on-chip interconnect conductor",
            description="Local, intermediate, global, or via conductor at scaled dimensions.",
            aliases=(
                "interconnect",
                "wiring",
                "BEOL",
                "via",
                "배선",
                "인터커넥트",
                "비아",
            ),
            scopes=("thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=(
                "line_level",
                "line_width",
                "line_height",
                "line_length",
                "liner_barrier_thickness",
                "grain_structure",
                "temperature",
                "current_density",
                "patterning_process",
            ),
            criteria=interconnect,
            seeds=(
                _seed(
                    "interconnect-metals",
                    "scaled interconnect metals",
                    ("Cu", "Co", "Ru", "Mo", "W"),
                    "Incumbent and post-Cu candidates whose order changes with dimensions "
                    "and liner/barrier volume.",
                    (IRDS_MM,),
                ),
                _seed(
                    "interconnect-low-dimensional",
                    "low-dimensional research conductors",
                    ("CNT", "graphene", "topological semimetal"),
                    "Research candidates requiring contact, integration, and yield validation.",
                    (IRDS_BC,),
                ),
            ),
            capabilities=(
                "scaled-interconnect-property-search",
                "electromigration-reliability-search",
                "beol-integration-search",
            ),
            failure_modes=(
                "bulk resistivity is extrapolated to a nanoscale line",
                "barrier or liner volume is omitted",
                "grain and surface scattering are ignored",
                "electromigration and fill yield are untested",
            ),
            references=(IRDS_MM, IRDS_BC),
            boundary=(
                "A low bulk resistivity is not a scaled-interconnect result; actual "
                "line geometry, liner/barrier, microstructure, resistance, "
                "electromigration, and integration yield are required."
            ),
        )
    )

    dielectric = [
        _criterion(
            "gate-eot",
            "equivalent_oxide_thickness",
            "nm",
            "performance",
            "minimize",
            ("channel", "gate_stack", "physical_thickness"),
            ("gate-stack-electrostatics", "moscap-capacitance-measurement"),
            ("Full gate-stack electrostatics",),
            ("MOSCAP/FET C-V and ellipsometry/TEM",),
            "Bulk dielectric constant alone cannot establish EOT.",
        ),
        _criterion(
            "gate-leakage",
            "leakage_current_density",
            "A/cm^2",
            "reliability",
            "minimize",
            ("electric_field", "temperature", "physical_thickness", "electrode_stack"),
            ("defect-aware-tunneling-model", "gate-leakage-measurement"),
            ("Band-offset/defect-aware tunnelling",),
            ("MOSCAP/FET I-V at field and temperature",),
            "Leakage is thickness, interface, defect, and electrode dependent.",
        ),
        _criterion(
            "gate-interface-traps",
            "interface_trap_density",
            "1/(cm^2 eV)",
            "integration",
            "minimize",
            ("channel", "temperature", "measurement_frequency", "process"),
            ("interface-defect-workflow", "cv-conductance-method"),
            ("Interface defect calculations",),
            ("C-V/conductance and device hysteresis",),
            "A perfect bulk dielectric omits channel-interface traps.",
        ),
        _criterion(
            "gate-tddb",
            "tddb_lifetime",
            "hour",
            "reliability",
            "maximize",
            ("electric_field", "temperature", "failure_criterion", "area"),
            ("defect-degradation-screen", "tddb-weibull-test"),
            ("Defect/degradation screening",),
            ("TDDB Weibull plus BTI",),
            "A calculated breakdown field is not a lifetime distribution.",
        ),
    ]
    roles.append(
        _role(
            role_id="gate_dielectric",
            field=field,
            display_name="gate dielectric stack",
            description="Insulating gate stack controlling a specified channel material.",
            aliases=(
                "gate dielectric",
                "gate oxide",
                "high-k",
                "게이트 절연막",
                "게이트 산화막",
                "고유전",
            ),
            scopes=("thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=(
                "channel",
                "target_eot",
                "gate_architecture",
                "electric_field",
                "temperature",
                "max_leakage",
                "process_temperature",
                "lifetime_requirement",
            ),
            criteria=dielectric,
            seeds=(
                _seed(
                    "gate-oxide-baselines",
                    "oxide gate dielectrics",
                    ("SiO2", "HfO2", "HfSiON", "Al2O3", "ZrO2"),
                    "Baseline and high-k families with interface and reliability trade-offs.",
                    (HIGH_K_NIST, IRDS_MM),
                ),
                _seed(
                    "gate-emerging-interfaces",
                    "emerging gate interfaces",
                    ("hBN", "CaF2", "single-crystal metal oxide"),
                    "Research interfaces for low-dimensional channels.",
                    (IRDS_BC,),
                ),
            ),
            capabilities=(
                "gate-dielectric-property-search",
                "interface-trap-and-band-offset-search",
                "tddb-bti-reliability-search",
            ),
            failure_modes=(
                "high dielectric constant hides small band offsets",
                "interface traps degrade mobility and threshold stability",
                "thin-film leakage differs from bulk behavior",
                "TDDB or BTI lifetime is untested",
            ),
            references=(HIGH_K_NIST, IRDS_MM),
            boundary=(
                "High bulk permittivity is not sufficient; EOT, leakage, offsets, "
                "interface traps, reliability, and process compatibility must be "
                "validated on the intended channel stack."
            ),
        )
    )

    barrier = [
        _criterion(
            "barrier-diffusivity",
            "diffusion_coefficient",
            "m^2/s",
            "reliability",
            "minimize",
            ("diffusant", "temperature", "film_thickness", "microstructure"),
            ("defect-aware-diffusion-workflow", "sims-or-tem-diffusion-test"),
            ("Defect and grain-boundary diffusion barriers",),
            ("ToF-SIMS/TEM/EDS after anneal or bias-temperature stress",),
            "Perfect-crystal migration omits pinholes and grain boundaries.",
        ),
        _criterion(
            "barrier-resistance-penalty",
            "line_resistance_penalty",
            "fraction",
            "integration",
            "minimize",
            ("line_geometry", "barrier_thickness", "liner_stack", "temperature"),
            ("scaled-line-transport", "patterned-line-resistance"),
            ("Full scaled line transport",),
            ("Patterned-line resistance with and without barrier",),
            "Barrier effectiveness cannot be separated from conductor volume loss.",
        ),
        _criterion(
            "barrier-adhesion",
            "adhesion_energy",
            "J/m^2",
            "integration",
            "maximize",
            ("adjacent_materials", "surface_termination", "process"),
            ("interface-adhesion-workflow", "adhesion-delamination-test"),
            ("Interface adhesion and reaction screening",),
            ("Four-point bend or validated delamination method",),
            "Ideal-interface adhesion may not represent deposited films.",
        ),
    ]
    roles.append(
        _role(
            role_id="metal_diffusion_barrier",
            field=field,
            display_name="metal diffusion barrier or liner",
            description="Ultrathin barrier/liner in a conductor and dielectric stack.",
            aliases=(
                "diffusion barrier",
                "barrier liner",
                "TaN",
                "확산 방지막",
                "배리어",
                "라이너",
            ),
            scopes=("thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=(
                "diffusant",
                "adjacent_materials",
                "maximum_thickness",
                "line_geometry",
                "deposition_process",
                "anneal_temperature",
                "anneal_time",
                "electric_field",
            ),
            criteria=barrier,
            seeds=(
                _seed(
                    "barrier-incumbents",
                    "refractory metal and nitride barriers",
                    ("Ta/TaN", "TiN", "WN"),
                    "Conventional barrier families that anchor thickness and reliability.",
                    (IRDS_MM,),
                ),
                _seed(
                    "barrier-self-forming",
                    "self-forming and low-dimensional barriers",
                    ("Mn/MnN", "Co/Ru liner", "graphene", "hBN", "MoS2"),
                    "Thin alternatives whose coverage, defects, and integration dominate.",
                    (IRDS_MM, IRDS_BC),
                ),
            ),
            capabilities=(
                "diffusion-barrier-evidence-search",
                "beol-interface-reliability-search",
            ),
            failure_modes=(
                "pinhole or grain-boundary diffusion is omitted",
                "line-resistance penalty is hidden",
                "poor conformality or adhesion",
                "accelerated reliability conditions are absent",
            ),
            references=(IRDS_MM, IRDS_BC),
            boundary=(
                "A perfect-crystal migration barrier cannot establish an ultrathin "
                "film's coverage, diffusion lifetime, adhesion, or line-resistance cost."
            ),
        )
    )

    thermal = [
        _criterion(
            "thermal-conductivity",
            "thermal_conductivity",
            "W/(m K)",
            "performance",
            "maximize",
            ("temperature", "orientation", "thickness"),
            ("anharmonic-phonon-bte", "thermal-conductivity-measurement"),
            ("Anharmonic phonon BTE tensor",),
            ("TDTR/FDTR/3-omega/laser-flash",),
            "Bulk conductivity does not determine a bonded stack's thermal resistance.",
        ),
        _criterion(
            "thermal-boundary-resistance",
            "thermal_boundary_resistance",
            "m^2 K/W",
            "integration",
            "minimize",
            ("interface_stack", "temperature", "bonding_process"),
            ("interface-thermal-transport-model", "tdtr-interface-measurement"),
            ("Interface MD/NEGF or diffuse-mismatch screening",),
            ("TDTR/FDTR on the exact bonded interface",),
            "Interface resistance can dominate even with a high-k bulk material.",
        ),
        _criterion(
            "thermal-cte-mismatch",
            "cte_mismatch",
            "1/K",
            "reliability",
            "minimize",
            ("temperature_range", "adjacent_materials", "orientation"),
            ("thermoelastic-stack-model", "thermal-cycle-measurement"),
            ("Anisotropic thermoelastic stack model",),
            ("Dilatometry plus thermal-cycle reliability",),
            "Thermal conductivity alone omits stress and delamination.",
        ),
        _criterion(
            "thermal-stack-resistance",
            "stack_thermal_resistance",
            "K/W",
            "performance",
            "minimize",
            ("heat_flux", "hotspot_geometry", "stack_geometry", "cooling_boundary"),
            ("finite-element-thermal-model", "device-junction-thermometry"),
            ("Calibrated finite-element heat-flow model",),
            ("Raman/electrical junction thermometry",),
            "A material-only value cannot predict junction temperature.",
        ),
    ]
    roles.append(
        _role(
            role_id="thermal_spreader_or_substrate",
            field=field,
            display_name="thermal spreader, substrate, or interface",
            description="Heat-removal material or stack near a semiconductor junction.",
            aliases=(
                "heat spreader",
                "thermal spreader",
                "thermal substrate",
                "thermal interface",
                "TIM",
                "방열",
                "열 확산",
                "열전도 기판",
            ),
            scopes=("bulk_crystal", "thin_film", "interface_stack", "composite"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=(
                "location",
                "heat_flux",
                "hotspot_geometry",
                "thickness",
                "orientation",
                "temperature_range",
                "electrical_insulation",
                "adjacent_materials",
                "bonding_process",
            ),
            criteria=thermal,
            seeds=(
                _seed(
                    "thermal-high-k",
                    "high-thermal-conductivity solids",
                    ("CVD diamond", "cubic BAs", "BP", "AlN", "SiC"),
                    "Bulk heat-spreading candidates whose interfaces need separate evidence.",
                    (BA_THERMAL, NIST_THERMAL),
                ),
                _seed(
                    "thermal-anisotropic-composite",
                    "anisotropic and composite spreaders",
                    ("hBN", "graphite", "graphene", "Cu/diamond composite"),
                    "Architecture-dependent options requiring tensor and interface data.",
                    (NIST_THERMAL,),
                ),
            ),
            capabilities=(
                "thermal-property-search",
                "thermal-boundary-conductance-search",
                "package-reliability-search",
            ),
            failure_modes=(
                "bulk conductivity is substituted for stack resistance",
                "orientation, thickness, or interface is omitted",
                "CTE mismatch causes delamination",
                "electrical insulation or process compatibility is untested",
            ),
            references=(BA_THERMAL, NIST_THERMAL),
            boundary=(
                "Bulk thermal conductivity cannot establish junction temperature or "
                "package reliability; geometry, interfaces, anisotropy, CTE, and "
                "bonding must be validated."
            ),
        )
    )

    detector = [
        _criterion(
            "detector-responsivity",
            "spectral_responsivity",
            "A/W",
            "performance",
            "maximize",
            ("wavelength", "bias", "temperature", "optical_power", "active_area"),
            ("device-drift-diffusion-optics", "calibrated-responsivity-measurement"),
            ("Optical plus carrier-transport device model",),
            ("Traceably calibrated spectral responsivity",),
            "Band gap or absorption alone cannot establish responsivity.",
        ),
        _criterion(
            "detector-dark-current",
            "dark_current_density",
            "A/cm^2",
            "performance",
            "minimize",
            ("bias", "temperature", "active_area"),
            ("defect-recombination-device-model", "dark-current-measurement"),
            ("Defect-aware device transport",),
            ("Dark I-V with area and temperature",),
            "Dark current is device- and defect-dependent.",
        ),
        _criterion(
            "detector-nep",
            "noise_equivalent_power",
            "W/sqrt(Hz)",
            "performance",
            "minimize",
            ("wavelength", "bias", "temperature", "frequency", "bandwidth"),
            ("noise-device-model", "measured-noise-spectrum"),
            ("Device noise model",),
            ("Measured noise spectrum plus calibrated responsivity",),
            "Shot-noise-only assumptions cannot be reported as measured detectivity.",
        ),
        _criterion(
            "detector-bandwidth",
            "three_db_bandwidth",
            "Hz",
            "performance",
            "maximize",
            ("bias", "temperature", "load", "active_area"),
            ("transient-device-model", "frequency-response-measurement"),
            ("Transient carrier and RC model",),
            ("Calibrated frequency response or impulse measurement",),
            "A material lifetime alone does not establish packaged bandwidth.",
        ),
    ]
    roles.append(
        _role(
            role_id="photodetector_absorber",
            field=field,
            display_name="photodetector absorber",
            description="Spectral absorber and device stack for a defined detector band.",
            aliases=(
                "photodetector",
                "detector absorber",
                "image sensor",
                "광검출기",
                "포토디텍터",
                "이미지 센서",
            ),
            scopes=("bulk_crystal", "thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=(
                "wavelength",
                "spectral_band",
                "detector_type",
                "bias",
                "temperature",
                "optical_power",
                "active_area",
                "bandwidth",
                "cooling",
            ),
            criteria=detector,
            seeds=(
                _seed(
                    "detector-uv-visible",
                    "UV and visible detector families",
                    ("AlGaN/GaN", "SiC", "beta-Ga2O3", "Si", "GaAs"),
                    "Spectral-band-specific baselines requiring device measurements.",
                    (NIST_RESPONSIVITY,),
                ),
                _seed(
                    "detector-ir",
                    "near- and mid-infrared detector families",
                    ("Ge", "InGaAs/InP", "InSb", "HgCdTe", "type-II superlattice"),
                    "Infrared candidates whose cooling, noise, and integration differ.",
                    (NIST_RESPONSIVITY,),
                ),
            ),
            capabilities=(
                "optoelectronic-property-search",
                "photodetector-calibration-search",
                "detector-noise-reliability-search",
            ),
            failure_modes=(
                "band gap is converted directly into responsivity",
                "noise is assumed rather than measured",
                "wavelength, bias, area, or temperature is missing",
                "array and CMOS integration are untested",
            ),
            references=(NIST_RESPONSIVITY,),
            boundary=(
                "Band gap or absorption is only absorber screening; calibrated "
                "responsivity, dark current, measured noise, bandwidth, conditions, "
                "and device integration are required."
            ),
        )
    )

    # These three profiles intentionally remain device-specific and must not be
    # collapsed into the detector profile.
    emitter_criteria = [
        _criterion(
            "emitter-eqe",
            "external_quantum_efficiency",
            "fraction",
            "performance",
            "maximize",
            ("wavelength", "current_density", "temperature", "device_geometry"),
            ("recombination-device-model", "integrating-sphere-eqe-measurement"),
            ("Carrier/recombination and optical extraction model",),
            ("Calibrated EQE and optical power",),
            "Photoluminescence is not electroluminescent device efficiency.",
        ),
        _criterion(
            "emitter-lifetime",
            "operational_lifetime",
            "hour",
            "reliability",
            "maximize",
            ("current_density", "temperature", "failure_criterion", "ambient"),
            ("degradation-device-model", "accelerated-emitter-lifetime-test"),
            ("Defect and degradation screening",),
            ("Accelerated lifetime with declared extrapolation",),
            "Initial efficiency cannot establish operational lifetime.",
        ),
    ]
    roles.append(
        _role(
            role_id="light_emitter",
            field=field,
            display_name="semiconductor light emitter",
            description="LED, laser, or related emissive active-region material and stack.",
            aliases=("light emitter", "LED", "laser diode", "발광", "LED 소재", "레이저 다이오드"),
            scopes=("thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=("wavelength", "current_density", "temperature", "epitaxy", "device_geometry"),
            criteria=emitter_criteria,
            seeds=(
                _seed(
                    "emitter-families",
                    "semiconductor emitter families",
                    ("InGaN/GaN", "AlGaInP/GaAs", "InP-family", "perovskite", "organic"),
                    "Wavelength- and architecture-specific retrieval seeds.",
                    (IRDS_BC,),
                ),
            ),
            capabilities=("light-emitter-evidence-search", "emitter-reliability-search"),
            failure_modes=(
                "photoluminescence is substituted for EQE",
                "efficiency droop is omitted",
                "epitaxial defects or extraction are ignored",
            ),
            references=(IRDS_BC,),
            boundary=(
                "Photoluminescence or a bulk band structure cannot establish LED/laser "
                "efficiency, droop, linewidth, or lifetime."
            ),
        )
    )

    modulator_criteria = [
        _criterion(
            "modulator-vpi-l",
            "vpi_length_product",
            "V cm",
            "performance",
            "minimize",
            ("wavelength", "temperature", "electrode_geometry", "frequency"),
            ("electro-optic-device-model", "vpi-measurement"),
            ("Full optical/electrical mode-overlap model",),
            ("Vpi versus frequency and wavelength",),
            "A bulk electro-optic coefficient cannot establish VpiL.",
        ),
        _criterion(
            "modulator-insertion-loss",
            "insertion_loss",
            "dB",
            "performance",
            "minimize",
            ("wavelength", "device_length", "coupling_definition"),
            ("waveguide-loss-model", "calibrated-insertion-loss-measurement"),
            ("Waveguide and electrode loss model",),
            ("Calibrated on-chip insertion loss",),
            "Loss must be measured on the complete device and coupling convention.",
        ),
        _criterion(
            "modulator-bandwidth",
            "electro_optic_bandwidth",
            "GHz",
            "performance",
            "maximize",
            ("wavelength", "drive_voltage", "load", "device_length"),
            ("traveling-wave-device-model", "electro-optic-s21-measurement"),
            ("Traveling-wave RF/optical model",),
            ("Electro-optic S21 measurement",),
            "A material response time alone cannot establish system bandwidth.",
        ),
    ]
    roles.append(
        _role(
            role_id="electro_optic_modulator",
            field=field,
            display_name="electro-optic modulator",
            description="Material and integrated stack for electrically controlled optical modulation.",
            aliases=("electro optic modulator", "optical modulator", "변조기", "광변조", "전기광학"),
            scopes=("thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=("wavelength", "data_rate", "drive_voltage", "temperature", "footprint"),
            criteria=modulator_criteria,
            seeds=(
                _seed(
                    "modulator-families",
                    "integrated electro-optic platforms",
                    ("Si", "SiGe", "III-V", "thin-film LiNbO3", "BaTiO3", "EO polymer"),
                    "Platform-specific retrieval seeds with voltage/loss/bandwidth trade-offs.",
                    (IRDS_BC,),
                ),
            ),
            capabilities=("electro-optic-device-search", "photonics-integration-search"),
            failure_modes=(
                "bulk Pockels coefficient is substituted for VpiL",
                "insertion loss or RF bandwidth is omitted",
                "thermal drift and fabrication integration are untested",
            ),
            references=(IRDS_BC,),
            boundary=(
                "A bulk electro-optic coefficient is not a modulator score; VpiL, "
                "loss, bandwidth, energy, geometry, and integration are required."
            ),
        )
    )

    substrate_criteria = [
        _criterion(
            "substrate-lattice-mismatch",
            "lattice_mismatch",
            "fraction",
            "integration",
            "minimize",
            ("epilayer", "orientation", "temperature"),
            ("epitaxial-strain-workflow", "xrd-epitaxy-characterization"),
            ("Epitaxial strain and critical-thickness model",),
            ("Reciprocal-space XRD and TEM",),
            "Bulk lattice constants alone omit relaxation and defect formation.",
        ),
        _criterion(
            "substrate-dislocation-density",
            "threading_dislocation_density",
            "1/cm^2",
            "reliability",
            "minimize",
            ("epilayer", "thickness", "growth_process"),
            ("dislocation-formation-screen", "xrd-tem-etch-pit-density"),
            ("Dislocation formation/propagation screening",),
            ("XRD/TEM/etch-pit density",),
            "A lattice-matched pair may still form high defect density.",
        ),
        _criterion(
            "substrate-thermal-conductivity",
            "thermal_conductivity",
            "W/(m K)",
            "performance",
            "maximize",
            ("temperature", "orientation", "thickness"),
            ("anharmonic-phonon-bte", "thermal-conductivity-measurement"),
            ("Anharmonic phonon BTE",),
            ("TDTR/3-omega/laser-flash",),
            "The bonded interface may dominate the substrate path.",
        ),
    ]
    roles.append(
        _role(
            role_id="substrate_or_buffer",
            field=field,
            display_name="epitaxial substrate or buffer",
            description="Substrate/buffer supporting a semiconductor active layer.",
            aliases=("substrate", "buffer layer", "epitaxy", "기판", "버퍼층", "에피택시"),
            scopes=("bulk_crystal", "thin_film", "interface_stack"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=("epilayer", "orientation", "growth_process", "temperature", "thickness"),
            criteria=substrate_criteria,
            seeds=(
                _seed(
                    "substrate-families",
                    "semiconductor substrates and buffers",
                    ("Si", "SOI", "sapphire", "SiC", "GaN", "AlN"),
                    "Application-dependent substrate and buffer baselines.",
                    (IRDS_MM,),
                ),
            ),
            capabilities=("epitaxy-substrate-search", "dislocation-and-interface-search"),
            failure_modes=(
                "lattice mismatch is used without growth conditions",
                "thermal expansion or wafer availability is omitted",
                "interface and threading defects are unmeasured",
            ),
            references=(IRDS_MM,),
            boundary=(
                "Lattice constants and bulk stability do not establish epitaxial quality, "
                "wafer manufacturability, or device-ready defect density."
            ),
        )
    )

    passivation_criteria = [
        _criterion(
            "passivation-interface-traps",
            "interface_trap_density",
            "1/(cm^2 eV)",
            "performance",
            "minimize",
            ("semiconductor_surface", "process", "temperature", "measurement_frequency"),
            ("surface-defect-workflow", "cv-or-conductance-measurement"),
            ("Surface/interface defect calculation",),
            ("C-V/conductance or gated device extraction",),
            "Bulk chemistry cannot establish surface passivation quality.",
        ),
        _criterion(
            "passivation-bias-drift",
            "threshold_voltage_drift",
            "V",
            "reliability",
            "minimize",
            ("electric_field", "temperature", "stress_time", "ambient"),
            ("charge-trapping-workflow", "bias-temperature-stress"),
            ("Charge-trapping and diffusion screening",),
            ("Bias-temperature stress with recovery protocol",),
            "Initial interface quality is not long-term stability.",
        ),
    ]
    roles.append(
        _role(
            role_id="surface_passivation",
            field=field,
            display_name="semiconductor surface passivation",
            description="Surface/interface treatment that suppresses traps and degradation.",
            aliases=("passivation", "surface treatment", "패시베이션", "표면 보호", "표면 처리"),
            scopes=("thin_film", "interface_stack", "patterned_device"),
            bulk_cif_scope="insufficient-interface-or-device-required",
            context=("semiconductor_surface", "process", "temperature", "electric_field", "ambient"),
            criteria=passivation_criteria,
            seeds=(
                _seed(
                    "passivation-families",
                    "dielectric and chemical passivation",
                    ("SiO2", "Al2O3", "SiNx", "sulfur treatment", "organic monolayer"),
                    "Surface- and process-specific retrieval seeds.",
                    (IRDS_MM,),
                ),
            ),
            capabilities=("surface-passivation-search", "interface-reliability-search"),
            failure_modes=(
                "bulk dielectric properties are used as passivation evidence",
                "surface termination and process history are missing",
                "bias/thermal/environmental drift is untested",
            ),
            references=(IRDS_MM,),
            boundary=(
                "A bulk material property cannot establish passivation; the exact "
                "surface, process, trap density, and stress stability must be measured."
            ),
        )
    )

    return tuple(roles)


SEMICONDUCTOR_APPLICATION_ROLES = _semiconductor_roles()


def _generic_direction(property_name: str) -> CriterionDirection:
    lowered = property_name.casefold()
    maximize_markers = (
        "conductivity",
        "capacity",
        "voltage",
        "critical_temperature",
        "turnover",
        "selectivity",
        "durability",
        "absorption",
        "slme",
        "zt",
        "coercivity",
        "remanence",
        "curie",
        "polarization",
        "piezoelectric",
        "yield",
        "toughness",
        "uptake",
        "mobility",
    )
    minimize_markers = (
        "energy_above_hull",
        "resistance",
        "overpotential",
        "degradation",
        "force",
        "leakage",
        "loss",
    )
    if any(marker in lowered for marker in maximize_markers):
        return "maximize"
    if any(marker in lowered for marker in minimize_markers):
        return "minimize"
    return "user_defined"


def _generic_roles_for_field(
    field: MaterialField,
) -> tuple[ApplicationRoleProfile, ...]:
    profile = MATERIAL_FIELD_PROFILES[field]
    validators_by_property: dict[str, list[str]] = {}
    for route in profile.stage_routes:
        for validator in route.validators:
            if not validator.can_create_property_scores:
                continue
            for property_name in validator.properties:
                validators_by_property.setdefault(property_name, []).append(
                    validator.validator_id
                )
    roles: list[ApplicationRoleProfile] = []
    for subtype in profile.application_subtypes:
        field_criteria = [
            _criterion(
                f"{subtype}-{requirement.property_name}",
                requirement.property_name,
                requirement.unit,
                "performance",
                _generic_direction(requirement.property_name),
                requirement.required_context,
                validators_by_property.get(
                    requirement.property_name,
                    ["external-field-validator-required"],
                ),
                requirement.preferred_calculations,
                requirement.experimental_confirmation,
                (
                    "This criterion is valid only with the declared field profile, "
                    "unit, complete conditions, and named validator provenance."
                ),
                required=requirement.required_for_field_claim,
            )
            for requirement in profile.properties
        ]
        criteria = [
            *field_criteria,
            *_role_specific_criteria(role_id),
        ]
        roles.append(
            _role(
                role_id=subtype,
                field=field,
                display_name=subtype.replace("_", " "),
                description=(
                    f"{profile.name} application role `{subtype}` using the existing "
                    "field-specific property and validator contract."
                ),
                aliases=tuple(dict.fromkeys((subtype, subtype.replace("_", " ")))),
                scopes=("bulk_crystal",),
                bulk_cif_scope="can-screen-bulk-only",
                context=profile.required_problem_context,
                criteria=criteria,
                seeds=(),
                capabilities=tuple(
                    dict.fromkeys(
                        capability
                        for route in profile.stage_routes
                        for capability in route.mcp_capabilities
                    )
                ),
                failure_modes=(
                    "required operating conditions are missing",
                    "a generic crystal proxy is substituted for a field property",
                    "literature context is converted into a material-performance score",
                ),
                references=profile.research_reference_ids,
                boundary=profile.field_claim_boundary,
            )
        )
    return tuple(roles)


_FIELD_ROLE_SPECS: dict[
    MaterialField,
    tuple[
        tuple[
            str,
            str,
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            tuple[str, ...],
            Literal[
                "can-screen-bulk-only",
                "insufficient-interface-or-device-required",
            ],
        ],
        ...,
    ],
] = {
    MaterialField.GENERAL_INORGANIC: (
        (
            "stable_bulk_phase",
            "stable inorganic bulk phase",
            ("stable crystal", "bulk phase", "안정한 결정", "무기 결정", "합성 가능한 결정"),
            ("chemical_system", "temperature", "pressure"),
            ("oxide", "nitride", "carbide", "boride", "chalcogenide", "halide"),
            (
                "reference-energy policy mismatch",
                "temperature or pressure selects another phase",
                "soft modes or decomposition remain untested",
            ),
            "can-screen-bulk-only",
        ),
        (
            "general_thermal_management",
            "general thermal-management material",
            ("heat spreader", "thermal insulator", "방열재", "단열재", "열관리 소재"),
            ("temperature", "geometry", "orientation", "interface_stack"),
            ("diamond", "c-BN", "AlN", "SiC", "zirconate ceramic"),
            (
                "bulk and thin-film thermal properties differ",
                "anisotropy and interface resistance are omitted",
                "porosity or defects dominate service behavior",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "optical_window",
            "optical or infrared window",
            (
                "optical window",
                "IR window",
                "투명창",
                "적외선 창",
                "적외선 광학 창",
                "레이저 광학재",
            ),
            ("wavelength_range", "thickness", "temperature", "polarization", "incident_angle"),
            ("sapphire", "MgF2", "CaF2", "ZnSe", "transparent spinel"),
            (
                "band gap is substituted for spectral transmittance",
                "multiphonon or surface-scattering loss is omitted",
                "laser damage and birefringence are untested",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "refractory_component",
            "refractory or ultra-high-temperature component",
            ("refractory", "ultra high temperature ceramic", "내화재", "초고온 세라믹", "열차폐"),
            ("temperature", "stress", "service_environment", "service_time"),
            ("ZrB2", "HfB2", "SiC", "refractory carbide", "rare-earth zirconate"),
            (
                "oxidation or volatilization",
                "thermal-shock cracking",
                "coating/substrate mismatch",
            ),
            "can-screen-bulk-only",
        ),
        (
            "electrical_insulator",
            "electrical insulator or dielectric",
            ("electrical insulator", "dielectric", "절연재", "유전체"),
            ("electric_field", "temperature", "frequency", "geometry", "interface_stack"),
            ("Al2O3", "SiO2", "AlN", "h-BN", "high-k oxide"),
            (
                "defect-assisted leakage",
                "interface breakdown",
                "frequency or temperature mismatch",
            ),
            "insufficient-interface-or-device-required",
        ),
    ),
    MaterialField.BATTERY_ELECTRODE: (
        (
            "battery_positive_electrode_active",
            "battery positive-electrode active material",
            ("cathode", "positive electrode", "양극", "양극재", "양극 활물질", "캐소드"),
            (
                "working_ion",
                "cell_use_case",
                "reference_electrode",
                "charged_state",
                "discharged_state",
                "voltage_window",
                "temperature",
                "cycling_protocol",
                "rate",
                "cycle_number",
                "mass_basis",
            ),
            (
                "layered oxide",
                "spinel cathode",
                "polyanion cathode",
                "Prussian blue analogue",
                "conversion fluoride or sulfide",
            ),
            (
                "charged-state instability or gas evolution",
                "electrolyte oxidation and CEI growth",
                "transition-metal dissolution",
                "volume-change cracking and voltage fade",
            ),
            "can-screen-bulk-only",
        ),
        (
            "battery_negative_electrode_active",
            "battery negative-electrode active material",
            ("anode", "negative electrode", "음극", "음극재", "음극 활물질", "애노드"),
            (
                "working_ion",
                "cell_use_case",
                "reference_electrode",
                "charged_state",
                "discharged_state",
                "voltage_window",
                "temperature",
                "cycling_protocol",
                "rate",
                "cycle_number",
                "mass_basis",
            ),
            (
                "graphite",
                "hard carbon",
                "silicon or tin alloying anode",
                "titanate or niobate insertion anode",
                "conversion anode",
            ),
            (
                "metal plating or dendrite formation",
                "SEI growth and first-cycle loss",
                "large volume change and particle fracture",
                "slow transport or unsafe low-potential operation",
            ),
            "can-screen-bulk-only",
        ),
    ),
    MaterialField.SOLID_ELECTROLYTE: (
        (
            "solid_electrolyte_bulk_separator",
            "bulk solid-electrolyte separator",
            (
                "solid electrolyte",
                "solid ionic conductor",
                "고체전해질",
                "고체 전해질",
                "고체 이온전도체",
                "전고체 전해질",
                "전고체 배터리",
            ),
            (
                "mobile_ion",
                "electrode_pair",
                "temperature",
                "microstructure",
                "stack_pressure",
                "thickness",
                "target_current_density",
                "processing_atmosphere",
            ),
            (
                "garnet oxide electrolyte",
                "NASICON or LISICON",
                "argyrodite or sulfide electrolyte",
                "halide electrolyte",
                "borohydride electrolyte",
            ),
            (
                "grain-boundary transport bottleneck",
                "mixed electronic conduction or filament penetration",
                "electrode-interface decomposition",
                "moisture sensitivity or contact-loss fracture",
            ),
            "can-screen-bulk-only",
        ),
        (
            "solid_electrolyte_interface_buffer",
            "solid-electrolyte interface buffer or coating",
            (
                "interface buffer",
                "solid electrolyte coating",
                "계면 완충층",
                "전극 전해질 코팅",
                "고체 고체 계면층",
            ),
            (
                "mobile_ion",
                "electrode_pair",
                "temperature",
                "stack_pressure",
                "thickness",
                "processing_atmosphere",
            ),
            (
                "oxide coating",
                "phosphate coating",
                "halide buffer",
                "polymer-ceramic interlayer",
            ),
            (
                "space-charge or interfacial resistance",
                "interdiffusion and decomposition",
                "poor conformality or contact loss",
                "electron leakage through the buffer",
            ),
            "insufficient-interface-or-device-required",
        ),
    ),
    MaterialField.SUPERCONDUCTOR: (
        (
            "high_field_magnet_conductor",
            "high-field superconducting magnet conductor",
            (
                "high field magnet",
                "MRI magnet",
                "fusion magnet",
                "고자기장 자석",
                "MRI 자석",
                "핵융합 자석",
            ),
            (
                "temperature",
                "applied_field",
                "field_orientation",
                "pressure",
                "strain",
                "sample_form",
                "measurement_criterion",
            ),
            ("Nb-Ti", "Nb3Sn", "REBCO coated conductor", "Bi-2212", "iron-based wire"),
            (
                "critical-current collapse at high field",
                "irreversible strain damage and anisotropy",
                "quench or delamination",
                "short-sample performance does not scale to long wire",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "superconducting_power_conductor",
            "superconducting power cable or winding",
            ("superconducting cable", "power conductor", "초전도 케이블", "송전선", "모터 권선"),
            (
                "temperature",
                "field_amplitude",
                "frequency",
                "waveform",
                "cable_geometry",
                "bend_radius",
            ),
            ("REBCO tape", "MgB2 wire", "Bi-2223 tape", "Nb-Ti multifilament"),
            (
                "AC loss and cryogenic heat leak",
                "joint resistance",
                "bending or handling damage",
                "stabilizer and filament geometry dominate",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "superconducting_rf_resonator",
            "superconducting RF resonator or cavity",
            ("RF cavity", "superconducting resonator", "초전도 RF", "공진기", "가속기 캐비티"),
            ("temperature", "frequency", "peak_rf_field", "surface_state"),
            ("high-purity Nb", "Nb3Sn coating", "MgB2 coating", "multilayer SIS"),
            (
                "surface defects, hydrogen, or contamination",
                "field emission and trapped flux",
                "coating nonuniformity and quench",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "josephson_device",
            "Josephson junction or cryogenic logic material stack",
            ("Josephson junction", "qubit", "조셉슨 접합", "큐비트", "극저온 로직"),
            ("temperature", "junction_area", "barrier_thickness", "process", "frequency"),
            ("Al/AlOx/Al", "Nb/AlOx/Nb", "NbN", "TiN", "granular Al"),
            (
                "TLS and interface loss",
                "junction drift or oxide nonuniformity",
                "process incompatibility and yield variation",
            ),
            "insufficient-interface-or-device-required",
        ),
    ),
    MaterialField.HETEROGENEOUS_CATALYST: (
        (
            "heterogeneous_catalyst_active_phase",
            "heterogeneous catalyst active phase or site",
            (
                "heterogeneous catalyst",
                "electrocatalyst",
                "photocatalyst",
                "active site",
                "불균일 촉매",
                "고체 촉매",
                "전기촉매",
                "광촉매",
                "활성점",
            ),
            (
                "reaction",
                "target_product",
                "catalysis_mode",
                "facet",
                "active_site",
                "support",
                "coverage",
                "temperature",
                "pressure",
                "feed_composition",
                "solvent_or_electrolyte",
                "electrode_potential",
                "ph",
                "target_duration",
            ),
            (
                "transition-metal surface or alloy",
                "oxide perovskite or spinel",
                "carbide nitride or phosphide",
                "supported nanoparticle",
                "single-atom catalyst",
            ),
            (
                "wrong reconstructed active surface",
                "unmodeled coverage, solvent, or competing pathway",
                "activity-selectivity trade-off",
                "poisoning, coking, sintering, dissolution, or support change",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "catalyst_support_or_interface",
            "catalyst support or active-phase interface",
            ("catalyst support", "support interaction", "촉매 지지체", "담지체", "촉매 계면"),
            (
                "reaction",
                "active_phase",
                "loading",
                "particle_size",
                "temperature",
                "pressure",
                "feed_composition",
                "solvent_or_electrolyte",
                "target_duration",
            ),
            ("oxide support", "carbon support", "zeolite support", "nitride or carbide support"),
            (
                "support changes the active phase",
                "sintering or detachment",
                "mass-transfer limitations",
                "support corrosion or pore blockage",
            ),
            "insufficient-interface-or-device-required",
        ),
    ),
    MaterialField.PHOTOVOLTAIC_ABSORBER: (
        (
            "photovoltaic_single_junction_absorber",
            "single-junction photovoltaic absorber",
            ("solar absorber", "active layer", "태양전지 흡수층", "광흡수층", "광활성층"),
            (
                "device_architecture",
                "illumination_spectrum",
                "target_band_gap_interval",
                "absorber_thickness",
                "temperature",
                "contacts",
                "chemical_potentials",
                "carrier_concentration",
                "operating_environment",
            ),
            (
                "Si, CdTe, or CIGS baseline",
                "chalcopyrite or kesterite",
                "halide perovskite",
                "phosphide or nitride",
                "Zintl or defect-tolerant chalcogenide",
            ),
            (
                "gap-only false positive or weak absorption",
                "deep-defect nonradiative recombination",
                "poor dopability or interface alignment",
                "light, heat, moisture, or ion-migration degradation",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "photovoltaic_tandem_top_absorber",
            "tandem photovoltaic top-cell absorber",
            ("tandem top absorber", "wide gap top cell", "탠덤 상부셀", "탠덤 탑셀", "광대역갭 상부셀"),
            (
                "bottom_cell",
                "illumination_spectrum",
                "target_band_gap_interval",
                "absorber_thickness",
                "temperature",
                "interconnect_stack",
                "operating_environment",
            ),
            ("wide-gap perovskite", "III-V top cell", "chalcogenide top absorber", "oxide or nitride absorber"),
            (
                "current mismatch and parasitic absorption",
                "phase segregation or interface recombination",
                "transparent-contact and interconnect loss",
                "stability protocol mismatch",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "photovoltaic_transport_or_contact_layer",
            "photovoltaic transport or contact layer",
            ("electron transport layer", "hole transport layer", "ETL", "HTL", "수송층", "접촉층"),
            (
                "absorber",
                "carrier_type",
                "band_alignment",
                "thickness",
                "deposition_process",
                "temperature",
                "operating_environment",
            ),
            ("metal oxide ETL", "organic HTL", "self-assembled monolayer", "transparent conductor"),
            (
                "band-offset or contact loss",
                "chemical reaction with the absorber",
                "ion migration and interfacial recombination",
                "thermal or moisture instability",
            ),
            "insufficient-interface-or-device-required",
        ),
    ),
    MaterialField.THERMOELECTRIC: (
        (
            "thermoelectric_n_type_leg",
            "n-type thermoelectric leg",
            (
                "n type thermoelectric",
                "n leg",
                "n형 열전",
                "n형 열전소재",
                "n형 열전 레그",
            ),
            (
                "generator_or_cooler",
                "cold_side_temperature",
                "hot_side_temperature",
                "carrier_concentration",
                "orientation",
                "microstructure",
                "contact_material",
                "service_environment",
                "service_duration",
            ),
            ("Bi2Te3", "Mg3Sb2", "half-Heusler", "skutterudite", "Zintl", "SiGe"),
            (
                "constant-relaxation-time false ranking",
                "bipolar conduction or carrier-density mismatch",
                "sample and temperature data are mixed",
                "contact resistance or high-temperature degradation",
            ),
            "can-screen-bulk-only",
        ),
        (
            "thermoelectric_p_type_leg",
            "p-type thermoelectric leg",
            (
                "p type thermoelectric",
                "p leg",
                "p형 열전",
                "p형 열전소재",
                "p형 열전 레그",
            ),
            (
                "generator_or_cooler",
                "cold_side_temperature",
                "hot_side_temperature",
                "carrier_concentration",
                "orientation",
                "microstructure",
                "contact_material",
                "service_environment",
                "service_duration",
            ),
            ("Bi2Te3", "PbTe", "SnSe", "half-Heusler", "skutterudite", "Zintl"),
            (
                "constant-relaxation-time false ranking",
                "bipolar conduction or carrier-density mismatch",
                "anisotropy or microstructure mismatch",
                "contact interdiffusion and thermal-expansion mismatch",
            ),
            "can-screen-bulk-only",
        ),
        (
            "thermoelectric_contact_or_interconnect",
            "thermoelectric contact or interconnect",
            ("thermoelectric contact", "열전 접촉", "열전 전극", "열전 인터커넥트"),
            (
                "leg_material",
                "temperature_range",
                "contact_geometry",
                "bonding_process",
                "service_duration",
            ),
            ("Ni contact", "Cu contact", "diffusion barrier", "active braze"),
            (
                "contact resistance and interdiffusion",
                "thermal-expansion mismatch and cracking",
                "reaction layer growth during service",
            ),
            "insufficient-interface-or-device-required",
        ),
    ),
    MaterialField.MAGNETIC_MATERIAL: (
        (
            "permanent_magnet",
            "permanent magnet",
            ("permanent magnet", "hard magnet", "영구자석", "모터 자석"),
            ("temperature", "field", "field_orientation", "microstructure", "sample_geometry"),
            ("Nd-Fe-B", "Sm-Co", "ferrite", "MnAl", "MnBi", "L1_0 FeNi"),
            (
                "grain-boundary coercivity loss",
                "thermal demagnetization or corrosion",
                "brittle fracture and critical-element dependence",
            ),
            "can-screen-bulk-only",
        ),
        (
            "soft_magnetic_core",
            "soft-magnetic core",
            ("soft magnet", "transformer core", "인덕터 코어", "변압기 코어", "모터 코어"),
            ("temperature", "frequency", "waveform", "peak_flux_density", "thickness", "stress"),
            ("electrical steel", "permalloy", "FeCo", "ferrite", "amorphous alloy", "nanocrystalline alloy"),
            (
                "eddy-current or hysteresis loss",
                "saturation at operating flux",
                "stress-induced anisotropy and processing loss",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "spintronic_layer",
            "spintronic or magnetic-tunnel-junction layer",
            ("spintronic", "MRAM", "magnetic tunnel junction", "스핀트로닉스", "자성 터널 접합"),
            ("temperature", "bias", "frequency", "layer_stack", "thickness", "anneal_process"),
            ("CoFeB/MgO", "Heusler alloy", "L1_0 FePt", "antiferromagnet", "2D magnet"),
            (
                "interface intermixing or dead layer",
                "damping and write-current trade-off",
                "thermal stability and retention failure",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "magnetocaloric_refrigerant",
            "magnetocaloric refrigerant",
            ("magnetocaloric", "magnetic cooling", "자기냉각", "자기열량"),
            ("temperature_window", "field_swing", "cycle_rate", "cycle_count"),
            ("Gd", "La(Fe,Si)13-H", "MnFe(P,Si)", "Ni-Mn Heusler", "FeRh"),
            (
                "thermal or magnetic hysteresis",
                "narrow working span and cracking",
                "cycle degradation or critical elements",
            ),
            "can-screen-bulk-only",
        ),
    ),
    MaterialField.FERROELECTRIC_PIEZOELECTRIC: (
        (
            "nonvolatile_ferroelectric_memory",
            "nonvolatile ferroelectric memory",
            ("FeRAM", "FeFET", "ferroelectric memory", "강유전 메모리"),
            ("temperature", "electric_field", "frequency", "film_thickness", "electrode_stack", "cycle_count"),
            ("doped HfO2", "HfZrO2", "PZT", "BiFeO3", "layered perovskite"),
            (
                "wake-up, imprint, fatigue, or depolarization",
                "leakage and electrode-interface reaction",
                "retention and endurance trade-off",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "piezoelectric_actuator",
            "piezoelectric actuator",
            ("piezoelectric actuator", "precision actuator", "압전 액추에이터", "정밀 구동기"),
            ("temperature", "electric_field", "frequency", "orientation", "stress", "domain_state"),
            ("PZT", "PMN-PT", "BaTiO3", "KNN", "BNT-based", "ScAlN"),
            (
                "hysteresis, depoling, or cracking",
                "field-induced phase drift",
                "lead or toxicity constraint",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "piezoelectric_sensor_transducer",
            "piezoelectric sensor or transducer",
            ("piezoelectric sensor", "ultrasound transducer", "압전 센서", "초음파 트랜스듀서"),
            ("temperature", "frequency", "orientation", "tensor_component", "acoustic_load"),
            ("PZT", "relaxor-PT", "PVDF", "AlN", "ScAlN", "ZnO", "1-3 composite"),
            (
                "acoustic mismatch",
                "dielectric heating or aging",
                "tensor and orientation mismatch",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "dielectric_energy_storage",
            "dielectric energy-storage capacitor",
            ("pulse capacitor", "dielectric energy storage", "고에너지 커패시터", "펄스 커패시터"),
            ("temperature", "electric_field", "frequency", "film_thickness", "cycle_count"),
            ("relaxor ferroelectric", "antiferroelectric", "BaTiO3-based", "HfO2-based", "dielectric composite"),
            (
                "leakage and local heating",
                "dielectric breakdown",
                "low discharge efficiency and fatigue",
            ),
            "insufficient-interface-or-device-required",
        ),
    ),
    MaterialField.STRUCTURAL_ALLOY: (
        (
            "lightweight_load_bearing",
            "lightweight load-bearing alloy",
            ("lightweight structural", "aerospace alloy", "경량 구조재", "항공 구조재"),
            ("processing_route", "heat_treatment", "microstructure", "temperature", "strain_rate", "service_environment"),
            ("Al alloy", "Mg alloy", "Ti alloy", "high-strength steel", "Al-Li alloy"),
            (
                "anisotropy and notch sensitivity",
                "corrosion fatigue",
                "weld or heat-affected-zone degradation",
            ),
            "can-screen-bulk-only",
        ),
        (
            "high_temperature_load_bearing",
            "high-temperature load-bearing alloy",
            (
                "turbine blade",
                "creep resistant",
                "high-temperature load-bearing",
                "hot corrosion",
                "고온 구조 합금",
                "고온 하중 지지",
                "고온",
                "고온 부품",
                "고온 부재",
                "내열합금",
                "크리프 합금",
                "터빈 블레이드",
            ),
            ("processing_route", "heat_treatment", "microstructure", "temperature", "stress", "service_time", "environment"),
            ("Ni superalloy", "Co superalloy", "ferritic-martensitic steel", "ODS alloy", "refractory HEA"),
            (
                "precipitate coarsening and TCP phase formation",
                "creep-fatigue interaction",
                "oxidation or hot corrosion",
            ),
            "can-screen-bulk-only",
        ),
        (
            "corrosion_resistant_component",
            "corrosion-resistant structural component",
            (
                "corrosion resistant",
                "hot corrosion",
                "marine alloy",
                "내식재",
                "내식 구조재",
                "고온 부식",
                "해양합금",
                "화학플랜트 합금",
            ),
            ("processing_route", "microstructure", "temperature", "service_environment", "stress", "service_time"),
            ("stainless steel", "Ni-Cr-Mo alloy", "duplex steel", "Ti alloy", "CoCr"),
            (
                "pitting or galvanic attack",
                "sensitization",
                "hydrogen embrittlement or stress-corrosion cracking",
            ),
            "can-screen-bulk-only",
        ),
        (
            "wear_resistant_component",
            "wear-resistant component",
            ("wear resistant", "bearing", "tooling", "내마모재", "베어링", "금형"),
            ("processing_route", "microstructure", "temperature", "load", "sliding_speed", "counterface", "environment"),
            ("tool steel", "cemented carbide", "CoCr", "hardfacing alloy", "metal-matrix composite"),
            (
                "brittle chipping",
                "adhesive or oxidation wear",
                "counterface and lubrication incompatibility",
            ),
            "can-screen-bulk-only",
        ),
    ),
    MaterialField.POROUS_FRAMEWORK: (
        (
            "gas_storage",
            "porous material for gas storage",
            ("gas storage", "hydrogen storage", "methane storage", "가스 저장", "수소 저장", "메탄 저장"),
            (
                "guest_species",
                "temperature",
                "adsorption_pressure",
                "desorption_pressure",
                "humidity",
                "activation_state",
                "pellet_density",
                "cycle_count",
            ),
            ("MOF", "COF", "zeolite", "porous carbon"),
            (
                "high uptake but low deliverable capacity",
                "low packed density or pore collapse",
                "heat management and cycle retention",
            ),
            "can-screen-bulk-only",
        ),
        (
            "gas_separation",
            "porous material for gas separation",
            ("gas separation", "PSA", "membrane separation", "가스 분리", "막 분리"),
            (
                "guest_species",
                "mixture_composition",
                "temperature",
                "adsorption_pressure",
                "desorption_pressure",
                "humidity",
                "pellet_density",
            ),
            ("SIFSIX", "MOF-74", "ZIF", "UiO", "zeolite", "COF"),
            (
                "pure-component selectivity fails in mixtures",
                "diffusion limitation or water competition",
                "pelletization and process productivity loss",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "carbon_capture",
            "porous material for carbon capture",
            ("carbon capture", "direct air capture", "CO2 capture", "탄소 포집", "이산화탄소 포집"),
            (
                "mixture_composition",
                "temperature",
                "adsorption_pressure",
                "desorption_pressure",
                "humidity",
                "cycle_count",
            ),
            ("amine-appended MOF", "SIFSIX", "MOF-74", "UiO", "zeolite 13X", "porous polymer"),
            (
                "water displacement or oxidative degradation",
                "excess adsorption enthalpy and regeneration energy",
                "low process-level productivity",
            ),
            "insufficient-interface-or-device-required",
        ),
        (
            "atmospheric_water_harvesting",
            "porous material for atmospheric water harvesting",
            ("water harvesting", "atmospheric water", "공기 중 물 포집", "대기 수분 포집"),
            ("temperature", "relative_humidity", "desorption_condition", "cycle_count", "device_geometry"),
            ("MOF-801", "MOF-303", "Al-fumarate", "zeolite", "hygroscopic composite"),
            (
                "wrong humidity-step placement",
                "slow kinetics or excessive regeneration heat",
                "hydrolysis, salt leakage, or cycle degradation",
            ),
            "insufficient-interface-or-device-required",
        ),
    ),
}


_RoleCriterionSpec: TypeAlias = tuple[
    str,
    str,
    CriterionCategory,
    CriterionDirection,
    tuple[str, ...],
    tuple[str, ...],
    str,
    str,
    str,
]


# These criteria supplement the field-level scientific contract with quantities
# that distinguish the actual component or service role.  The strings name
# validator *authorities*, not MCP tools.  Retrieval may find methods and
# caveats for these authorities, but it cannot create their property values.
_ROLE_CRITERION_SPECS: dict[str, tuple[_RoleCriterionSpec, ...]] = {
    "stable_bulk_phase": (
        (
            "phase_purity",
            "fraction",
            "reliability",
            "maximize",
            ("synthesis_route", "temperature", "pressure", "sample_preparation"),
            ("quantitative-phase-analysis", "powder-xrd-rietveld-measurement"),
            "Reference-aware simulated diffraction followed by multiphase refinement",
            "Quantitative XRD or neutron diffraction with impurity detection limits",
            "A low hull energy does not establish single-phase synthesis or phase purity.",
        ),
        (
            "synthesis_yield",
            "fraction",
            "integration",
            "maximize",
            (
                "synthesis_route",
                "precursors",
                "temperature",
                "pressure",
                "reaction_time",
            ),
            ("replicated-synthesis-yield-validation",),
            "Thermochemical reaction-path screening with competing products",
            "Replicated, mass-balanced synthesis with phase-resolved product yield",
            "A reported structure or one successful crystallite is not a bulk synthesis yield.",
        ),
    ),
    "general_thermal_management": (
        (
            "thermal_conductivity",
            "W/(m K)",
            "performance",
            "maximize",
            ("temperature", "orientation", "microstructure", "thickness"),
            ("anharmonic-phonon-transport", "tdtr-or-3omega-measurement"),
            "Converged anharmonic phonon BTE including isotope and defect assumptions",
            "TDTR, 3-omega, or steady-state measurement with density and orientation",
            "Bulk single-crystal conductivity cannot be assigned to a porous film or composite.",
        ),
        (
            "thermal_boundary_conductance",
            "MW/(m^2 K)",
            "integration",
            "maximize",
            (
                "temperature",
                "interface_stack",
                "surface_preparation",
                "bonding_process",
            ),
            ("interface-phonon-transport-workflow", "tdtr-interface-measurement"),
            "Atomistic interface transport on the declared termination and bonding state",
            "Stack-specific TDTR with interfacial-layer and roughness characterization",
            "Bulk thermal conductivity is not thermal boundary conductance.",
        ),
    ),
    "optical_window": (
        (
            "spectral_transmittance",
            "fraction",
            "performance",
            "maximize",
            (
                "wavelength_range",
                "thickness",
                "temperature",
                "polarization",
                "incident_angle",
                "surface_finish",
            ),
            ("electromagnetic-optics-stack-workflow", "spectrophotometry-measurement"),
            "Complex dielectric-function and multilayer optical calculation",
            "Calibrated spectral transmittance with reflection and scatter separated",
            "Band gap is not a transmission spectrum and cannot include surfaces or thickness.",
        ),
        (
            "laser_induced_damage_threshold",
            "J/cm^2",
            "reliability",
            "maximize",
            (
                "wavelength",
                "pulse_duration",
                "spot_size",
                "repetition_rate",
                "surface_finish",
            ),
            ("iso-21254-laser-damage-test",),
            "Defect and absorption screening only; no first-principles damage claim",
            "ISO 21254 style damage-probability test on the finished optic",
            "Damage thresholds from different pulse regimes or surface finishes are incomparable.",
        ),
    ),
    "refractory_component": (
        (
            "minimum_creep_rate",
            "s^-1",
            "reliability",
            "minimize",
            (
                "temperature",
                "stress",
                "service_environment",
                "microstructure",
                "service_time",
            ),
            ("high-temperature-creep-workflow", "constant-load-creep-measurement"),
            "Mechanism-aware creep modeling with grain size and phase fractions",
            "Constant-load creep curve with primary, secondary, and tertiary regimes",
            "A 0 K elastic modulus does not establish high-temperature creep resistance.",
        ),
        (
            "oxidation_mass_change_rate",
            "kg/(m^2 s)",
            "reliability",
            "minimize",
            (
                "temperature",
                "service_environment",
                "gas_flow",
                "surface_finish",
                "service_time",
            ),
            ("high-temperature-oxidation-kinetics", "thermogravimetric-oxidation-test"),
            "Competing-oxide thermodynamics and transport-kinetics screening",
            "Thermogravimetry plus scale adhesion and cross-section analysis",
            "Oxidation mass gain can hide volatilization or spallation unless products are resolved.",
        ),
    ),
    "electrical_insulator": (
        (
            "dielectric_breakdown_strength",
            "MV/m",
            "reliability",
            "maximize",
            (
                "electric_field_waveform",
                "temperature",
                "frequency",
                "geometry",
                "thickness",
                "electrode_stack",
            ),
            ("field-dependent-defect-transport", "weibull-breakdown-measurement"),
            "Defect-assisted transport and local-field screening on the declared stack",
            "Area-scaled Weibull breakdown test with ramp protocol and failure statistics",
            "A band gap or ideal dielectric constant is not a breakdown-strength measurement.",
        ),
        (
            "dielectric_loss_tangent",
            "dimensionless",
            "performance",
            "minimize",
            ("temperature", "frequency", "electric_field", "geometry", "humidity"),
            ("frequency-dependent-dielectric-response", "impedance-spectroscopy-measurement"),
            "Frequency-dependent ionic and electronic polarization calculation",
            "Calibrated impedance or resonator measurement over the operating band",
            "One low-frequency permittivity value cannot establish operating-frequency loss.",
        ),
    ),
    "battery_positive_electrode_active": (
        (
            "capacity_retention",
            "fraction",
            "reliability",
            "maximize",
            (
                "working_ion",
                "voltage_window",
                "temperature",
                "cycling_protocol",
                "rate",
                "cycle_number",
                "mass_basis",
            ),
            ("full-cell-cycling-protocol-validation",),
            "Charged-state phase, stress, and side-reaction screening",
            "Replicated full-cell cycling with formation protocol and electrode loading",
            "Half-cell retention or a short protocol is not long-term full-cell retention.",
        ),
        (
            "thermal_runaway_onset_temperature",
            "K",
            "resource_safety",
            "maximize",
            (
                "working_ion",
                "charged_state",
                "state_of_charge",
                "electrolyte",
                "heating_rate",
                "sample_mass",
            ),
            ("charged-electrode-calorimetry", "accelerating-rate-calorimetry"),
            "Charged-phase oxygen-release and electrolyte-reaction screening",
            "DSC or ARC on a declared charged electrode/electrolyte assembly",
            "A stable discharged bulk host cannot establish charged-cell thermal safety.",
        ),
    ),
    "battery_negative_electrode_active": (
        (
            "first_cycle_coulombic_efficiency",
            "fraction",
            "performance",
            "maximize",
            (
                "working_ion",
                "electrolyte",
                "formation_protocol",
                "rate",
                "temperature",
                "mass_basis",
            ),
            ("electrode-first-cycle-validation",),
            "Surface-reaction and irreversible-trapping screening",
            "Replicated formation cycles with complete charge accounting",
            "Theoretical capacity omits SEI loss and irreversible ion trapping.",
        ),
        (
            "cycling_volume_change",
            "fraction",
            "reliability",
            "minimize",
            (
                "working_ion",
                "state_of_charge",
                "cycle_number",
                "temperature",
                "microstructure",
            ),
            ("state-resolved-chemo-mechanical-workflow", "operando-dilatometry"),
            "State-resolved relaxed volumes plus particle-scale stress modeling",
            "Operando diffraction or dilatometry over repeated cycles",
            "End-member lattice volume alone does not establish electrode swelling or fracture.",
        ),
    ),
    "solid_electrolyte_bulk_separator": (
        (
            "electronic_conductivity",
            "S/cm",
            "resource_safety",
            "minimize",
            (
                "temperature",
                "electrode_pair",
                "electronic_blocking_condition",
                "microstructure",
            ),
            ("defect-electronic-transport-workflow", "dc-polarization-measurement"),
            "Charged-defect electronic transport under declared chemical potentials",
            "Hebb-Wagner or DC polarization with ionic/electronic separation",
            "Total conductivity cannot be treated as ionic when electronic leakage is unresolved.",
        ),
        (
            "critical_current_density",
            "mA/cm^2",
            "reliability",
            "maximize",
            (
                "mobile_ion",
                "electrode_pair",
                "temperature",
                "stack_pressure",
                "thickness",
                "areal_capacity",
                "protocol",
            ),
            ("symmetric-cell-critical-current-test",),
            "Defect, fracture, and filament-path screening only",
            "Stepwise symmetric-cell current test with short-circuit diagnostics",
            "Bulk ionic conductivity does not establish resistance to filamentary shorting.",
        ),
    ),
    "solid_electrolyte_interface_buffer": (
        (
            "area_specific_resistance",
            "ohm cm^2",
            "integration",
            "minimize",
            (
                "mobile_ion",
                "electrode_pair",
                "temperature",
                "stack_pressure",
                "thickness",
                "state_of_charge",
                "processing_atmosphere",
            ),
            ("interface-transport-workflow", "interface-impedance-measurement"),
            "Space-charge and atomistic migration screening for the explicit interface",
            "Equivalent-circuit-qualified interface impedance on a matched stack",
            "Bulk electrolyte conductivity cannot be converted into interface resistance.",
        ),
        (
            "interface_reaction_energy",
            "eV/atom",
            "integration",
            "maximize",
            (
                "electrode_pair",
                "state_of_charge",
                "temperature",
                "chemical_potentials",
                "reference_phase_set",
            ),
            ("interface-grand-potential-workflow", "operando-interface-analysis"),
            "Grand-potential reaction enumeration using a common reference phase set",
            "Operando or post-mortem phase analysis on the assembled interface",
            "The sign convention must be fixed: more negative values mean a stronger reaction drive.",
        ),
    ),
    "high_field_magnet_conductor": (
        (
            "critical_current_density",
            "A/mm^2",
            "performance",
            "maximize",
            (
                "temperature",
                "applied_field",
                "field_orientation",
                "strain",
                "electric_field_criterion",
                "sample_form",
            ),
            ("field-strain-superconducting-current-model", "four-probe-critical-current-test"),
            "Pinning-aware current screening for the declared field and strain",
            "Four-probe Ic/Jc measurement with field orientation and electric criterion",
            "Self-field film Jc is incomparable to engineering conductor Jc at high field.",
        ),
        (
            "irreversible_strain_limit",
            "fraction",
            "reliability",
            "maximize",
            (
                "temperature",
                "applied_field",
                "field_orientation",
                "sample_form",
                "loading_mode",
            ),
            ("superconducting-strain-limit-test",),
            "Multiscale conductor strain and crack-initiation screening",
            "Load-unload critical-current test defining irreversible degradation",
            "Crystal elastic strain is not the irreversible limit of a composite conductor.",
        ),
    ),
    "superconducting_power_conductor": (
        (
            "ac_loss_per_length",
            "W/m",
            "performance",
            "minimize",
            (
                "temperature",
                "field_amplitude",
                "frequency",
                "waveform",
                "transport_current",
                "cable_geometry",
            ),
            ("coupled-electromagnetic-ac-loss-workflow", "calorimetric-ac-loss-test"),
            "Hysteresis, coupling, and eddy-current model of the full cable geometry",
            "Calorimetric or electrical AC-loss measurement under the matched waveform",
            "A DC critical current does not determine AC loss or cryogenic heat load.",
        ),
        (
            "joint_resistance",
            "ohm",
            "integration",
            "minimize",
            (
                "temperature",
                "transport_current",
                "joint_geometry",
                "joining_process",
                "cycle_count",
            ),
            ("cryogenic-joint-resistance-measurement",),
            "Current-transfer and contact-resistivity screening of the declared joint",
            "Four-terminal joint resistance before and after thermal/current cycling",
            "Bulk superconductor properties cannot establish cable-joint resistance.",
        ),
    ),
    "superconducting_rf_resonator": (
        (
            "microwave_surface_resistance",
            "ohm",
            "performance",
            "minimize",
            ("temperature", "frequency", "peak_rf_field", "surface_state"),
            ("mattis-bardeen-surface-impedance", "rf-cavity-surface-resistance-test"),
            "Mechanism-appropriate surface-impedance calculation including residual loss",
            "Calibrated cavity or resonator surface-resistance measurement",
            "Bulk critical temperature cannot predict residual surface and trapped-flux loss.",
        ),
        (
            "quench_field",
            "T",
            "reliability",
            "maximize",
            (
                "temperature",
                "frequency",
                "surface_state",
                "cavity_geometry",
                "cooldown_protocol",
            ),
            ("rf-thermal-quench-workflow", "vertical-cavity-quench-test"),
            "Coupled RF-current and thermal-runaway screening with measured defects",
            "Vertical cavity test with thermometry and second-sound localization",
            "Thermodynamic critical field alone does not establish cavity quench field.",
        ),
    ),
    "josephson_device": (
        (
            "junction_critical_current_density",
            "A/cm^2",
            "performance",
            "target",
            (
                "temperature",
                "junction_area",
                "barrier_thickness",
                "process",
                "measurement_voltage_criterion",
            ),
            ("josephson-transport-workflow", "junction-iv-measurement"),
            "Tunneling transport on the explicit electrode/barrier stack",
            "Four-terminal junction I-V with area statistics and switching distribution",
            "A bulk superconducting gap does not set a fabricated junction current density.",
        ),
        (
            "junction_parameter_variation",
            "fraction",
            "reliability",
            "minimize",
            (
                "temperature",
                "junction_area",
                "wafer_process",
                "sample_count",
                "aging_time",
            ),
            ("wafer-scale-junction-statistics",),
            "Process-sensitivity screening only",
            "Wafer-scale critical-current or resistance-area distribution with aging",
            "One junction cannot establish process uniformity, yield, or drift.",
        ),
    ),
    "heterogeneous_catalyst_active_phase": (
        (
            "turnover_frequency",
            "s^-1",
            "performance",
            "maximize",
            (
                "reaction",
                "target_product",
                "active_site",
                "facet",
                "coverage",
                "temperature",
                "pressure",
                "feed_composition",
                "solvent_or_electrolyte",
                "electrode_potential",
                "ph",
            ),
            ("transition-state-and-microkinetic-workflow", "site-normalized-rate-measurement"),
            "Free-energy landscape and microkinetics on the declared active state",
            "Differential-rate measurement normalized by defensible active-site count",
            "Adsorption energy, current density, or mass activity alone is not turnover frequency.",
        ),
        (
            "product_selectivity",
            "fraction",
            "performance",
            "maximize",
            (
                "reaction",
                "target_product",
                "temperature",
                "pressure",
                "feed_composition",
                "conversion",
                "solvent_or_electrolyte",
                "electrode_potential",
                "ph",
            ),
            ("competing-pathway-microkinetic-workflow", "product-resolved-catalysis-test"),
            "Competing transition-state network with transport sensitivity",
            "Carbon/mass-balanced product analysis at declared conversion",
            "Activity for one pathway cannot establish selectivity against omitted products.",
        ),
    ),
    "catalyst_support_or_interface": (
        (
            "active_phase_support_adhesion_energy",
            "J/m^2",
            "integration",
            "maximize",
            (
                "active_phase",
                "support",
                "facet",
                "interface_termination",
                "coverage",
                "temperature",
            ),
            ("interface-adhesion-dft-workflow", "nanoparticle-adhesion-characterization"),
            "Termination- and stoichiometry-resolved interface separation energy",
            "Microscopy/spectroscopy plus adhesion or detachment test on the real support",
            "One ideal interface energy cannot establish nanoparticle stability under reaction.",
        ),
        (
            "active_phase_sintering_rate",
            "s^-1",
            "reliability",
            "minimize",
            (
                "active_phase",
                "loading",
                "particle_size",
                "temperature",
                "pressure",
                "feed_composition",
                "target_duration",
            ),
            ("supported-particle-sintering-kinetics", "operando-particle-size-measurement"),
            "Support-dependent diffusion and Ostwald-ripening kinetics",
            "Operando or time-resolved microscopy/chemisorption under reaction conditions",
            "Initial dispersion or adhesion is not a long-duration sintering rate.",
        ),
    ),
    "photovoltaic_single_junction_absorber": (
        (
            "minority_carrier_lifetime",
            "s",
            "performance",
            "maximize",
            (
                "temperature",
                "carrier_concentration",
                "injection_level",
                "sample_thickness",
                "surface_passivation",
                "chemical_potentials",
            ),
            ("nonradiative-capture-workflow", "time-resolved-photoluminescence"),
            "Defect capture coefficients and recombination kinetics at matched populations",
            "Injection-dependent TRPL or photoconductance with surface/bulk separation",
            "A favorable band gap or defect formation energy is not a carrier lifetime.",
        ),
        (
            "stabilized_power_conversion_efficiency",
            "fraction",
            "performance",
            "maximize",
            (
                "device_architecture",
                "illumination_spectrum",
                "absorber_thickness",
                "temperature",
                "contacts",
                "active_area",
                "stabilization_time",
            ),
            ("certified-photovoltaic-device-test",),
            "Drift-diffusion device modeling using independently validated inputs",
            "Stabilized maximum-power-point measurement with area and spectrum declared",
            "SLME and simulated efficiency are not certified device efficiency.",
        ),
    ),
    "photovoltaic_tandem_top_absorber": (
        (
            "current_matching_error",
            "fraction",
            "integration",
            "minimize",
            (
                "bottom_cell",
                "illumination_spectrum",
                "absorber_thickness",
                "interconnect_stack",
                "optical_stack",
                "temperature",
            ),
            ("tandem-optical-electrical-workflow", "subcell-eqe-measurement"),
            "Transfer-matrix optics plus subcell transport for the complete stack",
            "Bias-light EQE and current-voltage measurement resolving both subcells",
            "A target band gap alone cannot establish current matching in a real stack.",
        ),
        (
            "phase_segregation_fraction",
            "fraction",
            "reliability",
            "minimize",
            (
                "illumination_spectrum",
                "illumination_intensity",
                "temperature",
                "bias",
                "operating_time",
                "composition",
            ),
            ("photoinduced-phase-stability-workflow", "operando-phase-segregation-test"),
            "Photo-carrier-coupled free-energy and ion-migration screening",
            "Operando diffraction or spectroscopic phase quantification under bias/light",
            "Dark equilibrium stability does not establish light-induced phase stability.",
        ),
    ),
    "photovoltaic_transport_or_contact_layer": (
        (
            "interface_recombination_velocity",
            "cm/s",
            "integration",
            "minimize",
            (
                "absorber",
                "carrier_type",
                "band_alignment",
                "interface_stack",
                "surface_preparation",
                "temperature",
                "carrier_concentration",
            ),
            ("interface-defect-recombination-workflow", "surface-recombination-measurement"),
            "Interface-defect capture and band-bending calculation on the explicit stack",
            "Thickness- or lifetime-series extraction with passivation controls",
            "Bulk mobility or band alignment alone cannot establish interface recombination.",
        ),
        (
            "contact_sheet_resistance",
            "ohm/square",
            "integration",
            "minimize",
            (
                "thickness",
                "temperature",
                "carrier_concentration",
                "deposition_process",
                "anneal_process",
            ),
            ("thin-film-electronic-transport", "four-point-probe-sheet-resistance"),
            "Film transport including carrier density and grain-boundary assumptions",
            "Four-point probe or van der Pauw measurement on the processed layer",
            "Bulk resistivity cannot be substituted without the actual film thickness and process.",
        ),
    ),
    "thermoelectric_n_type_leg": (
        (
            "bipolar_onset_temperature",
            "K",
            "reliability",
            "maximize",
            (
                "carrier_type",
                "carrier_concentration",
                "temperature_range",
                "band_gap_method",
            ),
            ("bipolar-transport-workflow", "temperature-dependent-hall-seebeck-test"),
            "Two-carrier Boltzmann transport using a validated temperature-dependent gap",
            "Simultaneous Hall, Seebeck, and conductivity measurement across temperature",
            "A 0 K band gap does not determine the onset of bipolar transport.",
        ),
        (
            "compressive_strength",
            "MPa",
            "reliability",
            "maximize",
            ("temperature", "orientation", "microstructure", "sample_geometry"),
            ("thermoelectric-mechanical-workflow", "high-temperature-compression-test"),
            "Anisotropic thermoelastic and defect-sensitive strength screening",
            "Compression testing on processed legs across the operating range",
            "Single-crystal elastic constants do not establish processed-leg strength.",
        ),
    ),
    "thermoelectric_p_type_leg": (
        (
            "bipolar_onset_temperature",
            "K",
            "reliability",
            "maximize",
            (
                "carrier_type",
                "carrier_concentration",
                "temperature_range",
                "band_gap_method",
            ),
            ("bipolar-transport-workflow", "temperature-dependent-hall-seebeck-test"),
            "Two-carrier Boltzmann transport using a validated temperature-dependent gap",
            "Simultaneous Hall, Seebeck, and conductivity measurement across temperature",
            "A 0 K band gap does not determine the onset of bipolar transport.",
        ),
        (
            "compressive_strength",
            "MPa",
            "reliability",
            "maximize",
            ("temperature", "orientation", "microstructure", "sample_geometry"),
            ("thermoelectric-mechanical-workflow", "high-temperature-compression-test"),
            "Anisotropic thermoelastic and defect-sensitive strength screening",
            "Compression testing on processed legs across the operating range",
            "Single-crystal elastic constants do not establish processed-leg strength.",
        ),
    ),
    "thermoelectric_contact_or_interconnect": (
        (
            "specific_contact_resistivity",
            "ohm cm^2",
            "integration",
            "minimize",
            (
                "leg_material",
                "temperature_range",
                "contact_geometry",
                "bonding_process",
                "current_density",
            ),
            ("thermoelectric-interface-transport", "contact-resistivity-measurement"),
            "Band/metal alignment and current-transfer screening of the bonded interface",
            "Transfer-length or four-terminal contact measurement over temperature",
            "Leg electrical conductivity does not establish bonded-contact resistivity.",
        ),
        (
            "reaction_layer_growth_rate",
            "m/s",
            "reliability",
            "minimize",
            (
                "leg_material",
                "contact_material",
                "temperature_range",
                "bonding_process",
                "service_duration",
            ),
            ("interface-diffusion-kinetics", "aged-interface-cross-section-analysis"),
            "Multicomponent diffusion and reaction-phase kinetic modeling",
            "Time-resolved aged cross sections with reaction-layer quantification",
            "An initially low contact resistance does not establish long-term interface stability.",
        ),
    ),
    "permanent_magnet": (
        (
            "maximum_energy_product",
            "kJ/m^3",
            "performance",
            "maximize",
            ("temperature", "field_orientation", "microstructure", "sample_geometry"),
            ("micromagnetic-demagnetization-workflow", "closed-loop-hysteresis-measurement"),
            "Microstructure-resolved demagnetization and recoil-curve modeling",
            "Closed-loop B-H measurement with demagnetization correction",
            "Saturation magnetization and anisotropy alone do not establish usable BHmax.",
        ),
        (
            "coercive_field",
            "kA/m",
            "reliability",
            "maximize",
            (
                "temperature",
                "field_orientation",
                "microstructure",
                "sample_geometry",
                "field_sweep_rate",
            ),
            ("micromagnetic-coercivity-workflow", "hysteresis-coercivity-measurement"),
            "Defect- and grain-boundary-resolved reversal modeling",
            "Hysteresis measurement with texture and demagnetization correction",
            "Ideal magnetocrystalline anisotropy is not coercivity.",
        ),
    ),
    "soft_magnetic_core": (
        (
            "specific_core_loss",
            "W/kg",
            "performance",
            "minimize",
            (
                "temperature",
                "frequency",
                "waveform",
                "peak_flux_density",
                "thickness",
                "stress",
            ),
            ("dynamic-hysteresis-and-eddy-current-workflow", "iec-core-loss-measurement"),
            "Coupled hysteresis, classical, and excess-loss model for processed geometry",
            "IEC-style loss measurement under the declared waveform and flux",
            "DC coercivity cannot establish high-frequency core loss.",
        ),
        (
            "saturation_flux_density",
            "T",
            "performance",
            "maximize",
            ("temperature", "stress", "microstructure", "field_waveform"),
            ("finite-temperature-magnetic-workflow", "saturation-induction-measurement"),
            "Finite-temperature magnetic-state calculation with phase fractions",
            "High-field polarization measurement on the processed core material",
            "Atomic magnetic moment is not the saturation flux density of a real core.",
        ),
    ),
    "spintronic_layer": (
        (
            "tunnel_magnetoresistance_ratio",
            "fraction",
            "performance",
            "maximize",
            (
                "temperature",
                "bias",
                "layer_stack",
                "thickness",
                "anneal_process",
                "junction_area",
            ),
            ("spin-dependent-interface-transport", "mtj-resistance-state-measurement"),
            "Spin-dependent tunneling on the atomically explicit annealed stack",
            "Four-terminal parallel/antiparallel resistance over bias and temperature",
            "Bulk spin polarization cannot establish tunnel magnetoresistance.",
        ),
        (
            "gilbert_damping",
            "dimensionless",
            "performance",
            "minimize",
            ("temperature", "frequency", "layer_stack", "thickness", "anneal_process"),
            ("soc-magnetization-dynamics-workflow", "ferromagnetic-resonance-measurement"),
            "SOC-resolved damping including disorder and interface assumptions",
            "Broadband FMR with inhomogeneous broadening separated",
            "A calculated clean-bulk damping value is not a processed multilayer value.",
        ),
    ),
    "magnetocaloric_refrigerant": (
        (
            "isothermal_magnetic_entropy_change",
            "J/(kg K)",
            "performance",
            "maximize",
            (
                "temperature_window",
                "field_swing",
                "field_sweep_rate",
                "thermal_history",
            ),
            ("magneto-thermodynamic-free-energy-workflow", "calorimetric-magnetization-test"),
            "Field- and temperature-dependent free energy with phase hysteresis",
            "Direct calorimetry or validated Maxwell analysis using dense isotherms",
            "Sparse hysteretic magnetization curves can create spurious entropy-change peaks.",
        ),
        (
            "magnetic_hysteresis_loss",
            "J/kg",
            "reliability",
            "minimize",
            ("temperature_window", "field_swing", "cycle_rate", "cycle_count"),
            ("magnetostructural-hysteresis-workflow", "cyclic-magnetocaloric-test"),
            "Coupled magnetic/structural transition pathway and hysteresis modeling",
            "Cycle-resolved magnetic and thermal hysteresis measurement",
            "Peak entropy change without hysteresis loss cannot establish refrigeration utility.",
        ),
    ),
    "nonvolatile_ferroelectric_memory": (
        (
            "endurance_cycles",
            "cycle",
            "reliability",
            "maximize",
            (
                "temperature",
                "electric_field",
                "frequency",
                "film_thickness",
                "electrode_stack",
                "pulse_protocol",
            ),
            ("ferroelectric-fatigue-workflow", "switched-polarization-endurance-test"),
            "Defect/charge-migration and switching-fatigue screening of the stack",
            "Pulse endurance with switched-charge, leakage, wake-up, and breakdown separated",
            "A reversible Berry-phase path does not establish device endurance.",
        ),
        (
            "retention_time",
            "s",
            "reliability",
            "maximize",
            (
                "temperature",
                "film_thickness",
                "electrode_stack",
                "written_state",
                "read_protocol",
            ),
            ("ferroelectric-retention-kinetics", "accelerated-retention-test"),
            "Depolarization-field and thermally activated back-switching model",
            "Accelerated retention with state, read disturbance, and extrapolation model",
            "Remanent polarization measured immediately after writing is not retention.",
        ),
    ),
    "piezoelectric_actuator": (
        (
            "free_actuation_strain",
            "fraction",
            "performance",
            "maximize",
            (
                "temperature",
                "electric_field",
                "frequency",
                "orientation",
                "stress",
                "domain_state",
                "drive_waveform",
            ),
            ("nonlinear-piezoelectric-actuation-workflow", "laser-displacement-actuation-test"),
            "Field-dependent domain and phase response under the declared preload",
            "Bipolar/unipolar displacement loop with field and preload declared",
            "A small-signal piezoelectric coefficient does not establish large-signal stroke.",
        ),
        (
            "blocking_stress",
            "MPa",
            "performance",
            "maximize",
            (
                "temperature",
                "electric_field",
                "frequency",
                "orientation",
                "device_geometry",
            ),
            ("coupled-electromechanical-actuator-workflow", "blocked-force-measurement"),
            "Nonlinear electromechanical simulation of the actual actuator geometry",
            "Blocked-force measurement with active area and preload calibration",
            "Free strain and stiffness measured separately cannot be multiplied without a validated device model.",
        ),
    ),
    "piezoelectric_sensor_transducer": (
        (
            "piezoelectric_voltage_coefficient",
            "V m/N",
            "performance",
            "maximize",
            (
                "temperature",
                "frequency",
                "orientation",
                "tensor_component",
                "stress",
                "domain_state",
            ),
            ("dfpt-polar-response-workflow", "resonance-or-direct-piezoelectric-test"),
            "Tensor-resolved dielectric and piezoelectric response at matched boundary conditions",
            "Direct or resonance measurement with poling, orientation, and uncertainty",
            "A strain coefficient without dielectric response is not a voltage coefficient.",
        ),
        (
            "dielectric_loss_tangent",
            "dimensionless",
            "performance",
            "minimize",
            ("temperature", "frequency", "electric_field", "orientation"),
            ("frequency-dependent-dielectric-response", "impedance-spectroscopy-measurement"),
            "Frequency-dependent dielectric response including domain-wall assumptions",
            "Impedance or resonator measurement on the poled transducer material",
            "Room-temperature low-field loss is incomparable to operating-frequency loss.",
        ),
    ),
    "dielectric_energy_storage": (
        (
            "recoverable_energy_density",
            "J/cm^3",
            "performance",
            "maximize",
            (
                "temperature",
                "electric_field",
                "frequency",
                "film_thickness",
                "electrode_stack",
                "waveform",
            ),
            ("field-dependent-polarization-energy-workflow", "polarization-loop-energy-test"),
            "Field-dependent polarization and loss integration for the explicit stack",
            "Calibrated P-E loop integration with leakage and parasitics removed",
            "Permittivity times field squared is not recoverable energy for a hysteretic dielectric.",
        ),
        (
            "charge_discharge_efficiency",
            "fraction",
            "performance",
            "maximize",
            (
                "temperature",
                "electric_field",
                "frequency",
                "film_thickness",
                "cycle_count",
            ),
            ("dielectric-loss-and-switching-workflow", "pulse-discharge-efficiency-test"),
            "Hysteretic and conductive loss calculation over the drive cycle",
            "Pulse charge/discharge energy measurement over life and temperature",
            "Recoverable energy density without loss does not establish capacitor efficiency.",
        ),
    ),
    "lightweight_load_bearing": (
        (
            "specific_yield_strength",
            "kN m/kg",
            "performance",
            "maximize",
            (
                "processing_route",
                "heat_treatment",
                "microstructure",
                "temperature",
                "orientation",
                "strain_rate",
            ),
            ("microstructure-sensitive-strength-workflow", "tensile-density-measurement"),
            "Crystal-plasticity or strengthening model using measured phase and grain statistics",
            "Density and tensile yield measured on the same processed condition",
            "Ideal shear strength divided by theoretical density is not engineering specific strength.",
        ),
        (
            "fatigue_strength",
            "MPa",
            "reliability",
            "maximize",
            (
                "processing_route",
                "microstructure",
                "temperature",
                "stress_ratio",
                "frequency",
                "cycle_count",
                "service_environment",
            ),
            ("fatigue-crack-initiation-workflow", "sn-fatigue-test"),
            "Defect-population and microstructure-sensitive fatigue screening",
            "Statistical S-N testing with runouts and surface condition retained",
            "Monotonic yield strength cannot establish fatigue performance.",
        ),
    ),
    "high_temperature_load_bearing": (
        (
            "creep_rupture_life",
            "h",
            "reliability",
            "maximize",
            (
                "processing_route",
                "heat_treatment",
                "microstructure",
                "temperature",
                "stress",
                "service_environment",
            ),
            ("finite-temperature-creep-workflow", "creep-rupture-test"),
            "Precipitate/dislocation creep and phase-evolution modeling",
            "Constant-load creep-rupture test with elongation and failure mode",
            "0 K phase stability and modulus do not establish creep-rupture life.",
        ),
        (
            "oxidation_mass_gain",
            "kg/m^2",
            "reliability",
            "minimize",
            (
                "temperature",
                "service_environment",
                "service_time",
                "gas_flow",
                "surface_condition",
            ),
            ("alloy-oxidation-kinetics", "cyclic-oxidation-test"),
            "Selective oxidation and diffusion modeling with actual alloy activities",
            "Isothermal/cyclic oxidation with scale spallation and volatilization accounted",
            "A protective equilibrium oxide does not establish scale growth or adhesion.",
        ),
    ),
    "corrosion_resistant_component": (
        (
            "corrosion_penetration_rate",
            "mm/year",
            "reliability",
            "minimize",
            (
                "processing_route",
                "microstructure",
                "temperature",
                "service_environment",
                "flow_condition",
                "exposure_time",
            ),
            ("aqueous-corrosion-thermodynamics-kinetics", "standard-immersion-corrosion-test"),
            "Pourbaix/passive-film screening under declared chemistry and flow",
            "Mass-loss or penetration test with localized attack reported separately",
            "A noble equilibrium potential cannot establish service corrosion rate.",
        ),
        (
            "stress_corrosion_crack_growth_rate",
            "m/s",
            "reliability",
            "minimize",
            (
                "microstructure",
                "temperature",
                "service_environment",
                "stress_intensity",
                "loading_mode",
                "exposure_time",
            ),
            ("environment-assisted-cracking-workflow", "stress-corrosion-crack-growth-test"),
            "Hydrogen/environment-assisted crack-tip kinetics screening",
            "Fracture-mechanics SCC growth test in the declared environment",
            "General corrosion resistance does not imply immunity to stress-corrosion cracking.",
        ),
    ),
    "wear_resistant_component": (
        (
            "specific_wear_rate",
            "mm^3/(N m)",
            "reliability",
            "minimize",
            (
                "temperature",
                "load",
                "sliding_speed",
                "counterface",
                "environment",
                "lubrication",
                "surface_finish",
            ),
            ("tribological-contact-workflow", "pin-on-disk-wear-test"),
            "Contact-mechanics and wear-mechanism screening for the declared tribosystem",
            "Standardized wear-volume measurement with counterface and debris analysis",
            "Hardness alone cannot establish wear rate across different tribosystems.",
        ),
        (
            "friction_coefficient",
            "dimensionless",
            "performance",
            "minimize",
            (
                "temperature",
                "load",
                "sliding_speed",
                "counterface",
                "environment",
                "lubrication",
            ),
            ("tribofilm-and-contact-workflow", "tribometer-friction-test"),
            "Interface/tribofilm energetics and contact-mechanics screening",
            "Time-resolved tribometer measurement including running-in and steady state",
            "Friction is a system property and cannot be assigned to one bulk phase alone.",
        ),
    ),
    "gas_storage": (
        (
            "deliverable_gravimetric_capacity",
            "mol/kg",
            "performance",
            "maximize",
            (
                "guest_species",
                "temperature",
                "adsorption_pressure",
                "desorption_pressure",
                "humidity",
                "activation_state",
            ),
            ("flexible-framework-gcmc-workflow", "high-pressure-cyclic-isotherm-test"),
            "Mixture-aware GCMC with validated force field, charges, and flexibility",
            "Excess/absolute cyclic isotherms with skeletal mass and activation declared",
            "Maximum uptake is not deliverable capacity between operating pressures.",
        ),
        (
            "deliverable_volumetric_capacity",
            "mol/L",
            "performance",
            "maximize",
            (
                "guest_species",
                "temperature",
                "adsorption_pressure",
                "desorption_pressure",
                "pellet_density",
                "activation_state",
            ),
            ("packed-bed-adsorption-workflow", "pellet-volumetric-isotherm-test"),
            "Crystal-to-pellet packing and adsorption model using measured density",
            "Volumetric cyclic uptake on shaped material with envelope density",
            "Crystal-density volumetric uptake can overstate a porous powder or pellet.",
        ),
    ),
    "gas_separation": (
        (
            "mixture_adsorption_selectivity",
            "dimensionless",
            "performance",
            "maximize",
            (
                "guest_species",
                "mixture_composition",
                "temperature",
                "pressure",
                "humidity",
                "activation_state",
            ),
            ("mixture-gcmc-workflow", "multicomponent-breakthrough-test"),
            "Flexible-framework multicomponent GCMC with validated interactions",
            "Competitive mixture isotherm or breakthrough with mass balance",
            "Ideal selectivity from pure-component Henry constants may fail in mixtures.",
        ),
        (
            "mixture_working_capacity",
            "mol/kg",
            "performance",
            "maximize",
            (
                "guest_species",
                "mixture_composition",
                "temperature",
                "adsorption_pressure",
                "desorption_pressure",
                "humidity",
            ),
            ("cyclic-mixture-adsorption-workflow", "cyclic-breakthrough-test"),
            "Cycle-level equilibrium and mass-transfer process simulation",
            "Repeated adsorption/desorption breakthrough with retained composition",
            "Equilibrium selectivity alone does not establish cyclic separation productivity.",
        ),
    ),
    "carbon_capture": (
        (
            "co2_working_capacity",
            "mol/kg",
            "performance",
            "maximize",
            (
                "mixture_composition",
                "temperature",
                "adsorption_pressure",
                "desorption_pressure",
                "humidity",
                "cycle_count",
            ),
            ("humid-mixture-adsorption-workflow", "humid-flue-or-air-breakthrough-test"),
            "Competitive CO2/water adsorption and cycle simulation",
            "Humid multicomponent breakthrough over repeated regeneration cycles",
            "Dry pure-CO2 uptake does not establish flue-gas or direct-air-capture capacity.",
        ),
        (
            "regeneration_energy",
            "kJ/mol",
            "performance",
            "minimize",
            (
                "mixture_composition",
                "temperature",
                "adsorption_pressure",
                "desorption_pressure",
                "humidity",
                "regeneration_method",
            ),
            ("process-level-carbon-capture-workflow", "calorimetry-plus-cycle-energy-test"),
            "Process simulation including sensible, latent, vacuum, and compression work",
            "Calorimetry and measured cycle energy on shaped sorbent",
            "Isosteric heat alone is not total process regeneration energy.",
        ),
    ),
    "atmospheric_water_harvesting": (
        (
            "water_uptake_swing",
            "g/g",
            "performance",
            "maximize",
            (
                "temperature",
                "relative_humidity",
                "desorption_condition",
                "activation_state",
                "cycle_count",
            ),
            ("water-framework-adsorption-workflow", "dynamic-vapor-sorption-test"),
            "Water-cluster adsorption with framework flexibility and hysteresis",
            "Dynamic vapor sorption over the complete adsorption/desorption window",
            "Saturation water uptake is not the usable swing over ambient humidity.",
        ),
        (
            "water_productivity",
            "L/(kg day)",
            "performance",
            "maximize",
            (
                "temperature",
                "relative_humidity",
                "desorption_condition",
                "cycle_count",
                "device_geometry",
                "energy_input",
            ),
            ("coupled-water-harvester-workflow", "outdoor-or-climate-chamber-device-test"),
            "Coupled adsorption, heat, and mass-transfer simulation of the device",
            "Climate-chamber or outdoor daily water collection with energy balance",
            "Material uptake cannot be converted to daily device productivity without kinetics and heat transfer.",
        ),
    ),
}


_FIELD_BASELINE_DIRECTIONS: dict[str, CriterionDirection] = {
    "energy_above_hull": "minimize",
    "minimum_phonon_frequency": "maximize",
    "average_voltage": "user_defined",
    "specific_capacity": "maximize",
    "ion_migration_barrier": "minimize",
    "ionic_conductivity": "maximize",
    "migration_barrier": "minimize",
    "electrochemical_stability_window": "maximize",
    "critical_temperature": "maximize",
    "electron_phonon_coupling": "maximize",
    "reaction_free_energy": "target",
    "activation_barrier": "minimize",
    "durability": "maximize",
    "optical_absorption_coefficient": "maximize",
    "slme": "maximize",
    "nonradiative_recombination_rate": "minimize",
    "power_factor": "maximize",
    "lattice_thermal_conductivity": "minimize",
    "zt": "maximize",
    "magnetic_ordering_energy": "user_defined",
    "magnetocrystalline_anisotropy": "user_defined",
    "ordering_temperature": "maximize",
    "spontaneous_polarization": "user_defined",
    "switching_barrier": "user_defined",
    "piezoelectric_strain_coefficient": "maximize",
    "mixing_gibbs_free_energy": "minimize",
    "youngs_modulus": "user_defined",
    "service_degradation_rate": "minimize",
    "accessible_volume_fraction": "maximize",
    "adsorption_selectivity": "maximize",
    "framework_decomposition_free_energy": "maximize",
}

_ROLE_REQUIRED_BASE_PROPERTIES: dict[str, frozenset[str]] = {
    "stable_bulk_phase": frozenset(
        {"energy_above_hull", "minimum_phonon_frequency"}
    ),
    "general_thermal_management": frozenset(),
    "optical_window": frozenset(),
    "refractory_component": frozenset(),
    "electrical_insulator": frozenset(),
    "battery_positive_electrode_active": frozenset(
        {"average_voltage", "specific_capacity", "ion_migration_barrier"}
    ),
    "battery_negative_electrode_active": frozenset(
        {"average_voltage", "specific_capacity", "ion_migration_barrier"}
    ),
    "solid_electrolyte_bulk_separator": frozenset(
        {
            "ionic_conductivity",
            "migration_barrier",
            "electrochemical_stability_window",
        }
    ),
    "solid_electrolyte_interface_buffer": frozenset(
        {"ionic_conductivity", "electrochemical_stability_window"}
    ),
    "high_field_magnet_conductor": frozenset({"critical_temperature"}),
    "superconducting_power_conductor": frozenset({"critical_temperature"}),
    "superconducting_rf_resonator": frozenset({"critical_temperature"}),
    "josephson_device": frozenset({"critical_temperature"}),
    "heterogeneous_catalyst_active_phase": frozenset(
        {"reaction_free_energy", "activation_barrier", "durability"}
    ),
    "catalyst_support_or_interface": frozenset({"durability"}),
    "photovoltaic_single_junction_absorber": frozenset(
        {
            "optical_absorption_coefficient",
            "slme",
            "nonradiative_recombination_rate",
        }
    ),
    "photovoltaic_tandem_top_absorber": frozenset(
        {
            "optical_absorption_coefficient",
            "slme",
            "nonradiative_recombination_rate",
        }
    ),
    "photovoltaic_transport_or_contact_layer": frozenset(
        {"nonradiative_recombination_rate"}
    ),
    "thermoelectric_n_type_leg": frozenset(
        {"power_factor", "lattice_thermal_conductivity", "zt"}
    ),
    "thermoelectric_p_type_leg": frozenset(
        {"power_factor", "lattice_thermal_conductivity", "zt"}
    ),
    "thermoelectric_contact_or_interconnect": frozenset(),
    "permanent_magnet": frozenset({"ordering_temperature"}),
    "soft_magnetic_core": frozenset({"ordering_temperature"}),
    "spintronic_layer": frozenset({"ordering_temperature"}),
    "magnetocaloric_refrigerant": frozenset({"ordering_temperature"}),
    "nonvolatile_ferroelectric_memory": frozenset(
        {"spontaneous_polarization", "switching_barrier"}
    ),
    "piezoelectric_actuator": frozenset(
        {
            "spontaneous_polarization",
            "switching_barrier",
            "piezoelectric_strain_coefficient",
        }
    ),
    "piezoelectric_sensor_transducer": frozenset(
        {"piezoelectric_strain_coefficient"}
    ),
    "dielectric_energy_storage": frozenset(),
    "lightweight_load_bearing": frozenset(
        {"mixing_gibbs_free_energy", "service_degradation_rate"}
    ),
    "high_temperature_load_bearing": frozenset(
        {"mixing_gibbs_free_energy", "service_degradation_rate"}
    ),
    "corrosion_resistant_component": frozenset(
        {"mixing_gibbs_free_energy", "service_degradation_rate"}
    ),
    "wear_resistant_component": frozenset(
        {"mixing_gibbs_free_energy", "service_degradation_rate"}
    ),
    "gas_storage": frozenset(
        {"accessible_volume_fraction", "framework_decomposition_free_energy"}
    ),
    "gas_separation": frozenset(
        {
            "accessible_volume_fraction",
            "adsorption_selectivity",
            "framework_decomposition_free_energy",
        }
    ),
    "carbon_capture": frozenset(
        {
            "accessible_volume_fraction",
            "adsorption_selectivity",
            "framework_decomposition_free_energy",
        }
    ),
    "atmospheric_water_harvesting": frozenset(
        {"accessible_volume_fraction", "framework_decomposition_free_energy"}
    ),
}


def _field_baseline_criteria(
    field: MaterialField,
    role_id: str,
) -> list[ApplicationCriterion]:
    """Retain field physics in addition to role-specific service metrics."""

    profile = MATERIAL_FIELD_PROFILES[field]
    required_properties = _ROLE_REQUIRED_BASE_PROPERTIES.get(
        role_id,
        frozenset(),
    )
    validators_by_property: dict[str, list[str]] = {}
    for route in profile.stage_routes:
        for validator in route.validators:
            if not validator.can_create_property_scores:
                continue
            for property_name in validator.properties:
                validators_by_property.setdefault(property_name, []).append(
                    validator.validator_id
                )
    criteria: list[ApplicationCriterion] = []
    for requirement in profile.properties:
        direction = _FIELD_BASELINE_DIRECTIONS.get(
            requirement.property_name,
            "user_defined",
        )
        criteria.append(
            _criterion(
                f"{role_id}-{requirement.property_name}",
                requirement.property_name,
                requirement.unit,
                "performance",
                direction,
                requirement.required_context,
                validators_by_property.get(
                    requirement.property_name,
                    ["external-field-validator-required"],
                ),
                requirement.preferred_calculations,
                requirement.experimental_confirmation,
                (
                    "This is a field-level physical prerequisite, not a complete "
                    "application score. It remains unknown unless a named validator "
                    "returns the exact unit and conditions; role-specific film, "
                    "interface, device, process, or service evidence is still required."
                ),
                required=(
                    requirement.required_for_field_claim
                    and requirement.property_name in required_properties
                ),
            )
        )
    return criteria


def _role_specific_criteria(role_id: str) -> list[ApplicationCriterion]:
    specs = _ROLE_CRITERION_SPECS.get(role_id)
    if not specs:
        raise ValueError(
            f"missing code-owned application criteria for role {role_id!r}"
        )
    return [
        _criterion(
            f"{role_id}-{property_name}",
            property_name,
            unit,
            category,
            direction,
            required_context,
            validator_ids,
            (calculation,),
            (experiment,),
            caution,
        )
        for (
            property_name,
            unit,
            category,
            direction,
            required_context,
            validator_ids,
            calculation,
            experiment,
            caution,
        ) in specs
    ]


def _specialized_roles_for_field(
    field: MaterialField,
) -> tuple[ApplicationRoleProfile, ...]:
    specs = _FIELD_ROLE_SPECS.get(field)
    if not specs:
        return _generic_roles_for_field(field)
    profile = MATERIAL_FIELD_PROFILES[field]
    capabilities = tuple(
        dict.fromkeys(
            capability
            for route in profile.stage_routes
            for capability in route.mcp_capabilities
        )
    )
    roles: list[ApplicationRoleProfile] = []
    for (
        role_id,
        display_name,
        aliases,
        extra_context,
        seed_examples,
        failure_modes,
        bulk_scope,
    ) in specs:
        if role_id not in _ROLE_REQUIRED_BASE_PROPERTIES:
            raise ValueError(
                f"missing field-baseline applicability policy for role {role_id!r}"
            )
        criteria = [
            *_field_baseline_criteria(field, role_id),
            *_role_specific_criteria(role_id),
        ]
        seed = _seed(
            f"{role_id}-families",
            f"{display_name} retrieval families",
            seed_examples,
            (
                "Code-owned examples used to retrieve incumbent, emerging, and "
                "negative-evidence branches; they are not ranked recommendations."
            ),
            profile.research_reference_ids,
        )
        roles.append(
            _role(
                role_id=role_id,
                field=field,
                display_name=display_name,
                description=(
                    f"Role-scoped {profile.name} selection for {display_name}; "
                    "comparison is permitted only under identical conditions."
                ),
                aliases=aliases,
                scopes=(
                    ("bulk_crystal",)
                    if bulk_scope == "can-screen-bulk-only"
                    else ("bulk_crystal", "thin_film", "interface_stack", "patterned_device")
                ),
                bulk_cif_scope=bulk_scope,
                context=tuple(
                    dict.fromkeys(
                        (
                            *extra_context,
                            *(
                                name
                                for criterion in criteria
                                for name in criterion.required_context
                            ),
                        )
                    )
                ),
                criteria=criteria,
                seeds=(seed,),
                capabilities=capabilities,
                failure_modes=failure_modes,
                references=profile.research_reference_ids,
                boundary=(
                    profile.field_claim_boundary
                    + " This application role additionally requires its declared "
                    "component, geometry, processing, operating, reliability, and "
                    "service conditions; unlike roles are never cross-ranked."
                ),
            )
        )
    return tuple(roles)


APPLICATION_ROLE_PROFILES: dict[
    MaterialField, tuple[ApplicationRoleProfile, ...]
] = {
    field: (
        SEMICONDUCTOR_APPLICATION_ROLES
        if field == MaterialField.SEMICONDUCTOR
        else _specialized_roles_for_field(field)
    )
    for field in MaterialField
}


_BROAD_COMPONENT_TERMS = (
    "which part",
    "which component",
    "what material where",
    "component map",
    "device stack",
    "어떤 부분",
    "어느 부분",
    "어디에",
    "부품별",
    "소재 맵",
)
_COMPARE_TERMS = ("compare", "versus", " vs ", "비교", "중에서")
_DISCOVERY_TERMS = (
    "discover",
    "new material",
    "novel material",
    "generate",
    "신물질",
    "신소재",
    "새로운 소재",
    "발견",
    "생성",
)


def application_roles_for_field(
    material_field: MaterialField | str,
) -> tuple[ApplicationRoleProfile, ...]:
    field = MaterialField(str(material_field))
    return tuple(
        ApplicationRoleProfile.model_validate_json(
            item.model_dump_json(),
            strict=True,
        )
        for item in APPLICATION_ROLE_PROFILES[field]
    )


def get_application_role_profile(
    material_field: MaterialField | str,
    role_id: str,
) -> ApplicationRoleProfile:
    for role in application_roles_for_field(material_field):
        if role.role_id == role_id:
            return role
    raise KeyError(
        f"unknown application role {role_id!r} for field {material_field!r}"
    )


def infer_application_question_kind(question: str) -> ApplicationQuestionKind:
    normalized = _normalize_for_match(question)
    if any(term in normalized for term in _COMPARE_TERMS):
        return "compare_given_candidates"
    if any(term in normalized for term in _DISCOVERY_TERMS):
        return "novel_material_discovery"
    if any(term in normalized for term in _BROAD_COMPONENT_TERMS):
        return "component_map"
    return "component_selection"


def _match_roles(
    question: str,
    roles: Sequence[ApplicationRoleProfile],
) -> list[ApplicationRoleProfile]:
    normalized = _normalize_for_match(question)
    matched: list[ApplicationRoleProfile] = []
    for role in roles:
        if any(_term_in_text(normalized, alias) for alias in role.aliases):
            matched.append(role)
    return matched


def build_material_application_brief(
    question: str,
    *,
    material_field: MaterialField | str = "AUTO",
    chemical_system: str | None = None,
    problem_context: Mapping[str, JsonValue] | None = None,
    field_model_run: MaterialFieldModelRun | None = None,
    application_model_run: MaterialApplicationModelRun | None = None,
    explicit_role_ids: Sequence[str] | None = None,
    require_condition_complete: bool = False,
) -> MaterialApplicationBrief:
    """Compile a natural-language material question into bounded role portfolios.

    Broad component-map questions deliberately return multiple role-specific
    portfolios.  They never produce one cross-role ranking.
    """

    question = question.strip()
    if not question:
        raise ValueError("material application question cannot be empty")
    context = dict(problem_context or {})
    if _contains_sensitive_context_key(context):
        raise ValueError("application problem context cannot contain secrets")
    field_plan = build_material_domain_plan(
        material_field,
        prompt=question,
        chemical_system=chemical_system,
        problem_context=context,
        model_run=field_model_run,
    )
    field = MaterialField(str(field_plan.resolution.selected_field))
    allowed_roles = application_roles_for_field(field)
    by_id = {item.role_id: item for item in allowed_roles}

    if application_model_run is not None:
        if application_model_run.material_field != field:
            raise ValueError("application model run belongs to another material field")
        expected_prompt_hash = stable_hash(
            {
                "question": question,
                "material_field": field,
                "problem_context": context,
            }
        )
        if application_model_run.prompt_hash != expected_prompt_hash:
            raise ValueError(
                "application model run is stale for the current question or context"
            )
        _validate_application_model_decision(
            application_model_run.decision,
            material_field=field,
            roles_by_id=by_id,
            question=question,
            input_context=context,
        )
        model_ids = application_model_run.decision.selected_role_ids
        if any(role_id not in by_id for role_id in model_ids):
            raise ValueError("application model run contains a non-allowlisted role")
        model_question_kind = application_model_run.decision.question_kind
    else:
        model_ids = []
        model_question_kind = None

    if explicit_role_ids is not None:
        selected_ids = list(explicit_role_ids)
        if not selected_ids:
            raise ValueError("explicit application roles cannot be empty")
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError("explicit application roles must be unique")
        if any(role_id not in by_id for role_id in selected_ids):
            raise ValueError("explicit application role is outside the field allowlist")
    elif model_ids:
        selected_ids = list(model_ids)
    else:
        matched = _match_roles(question, allowed_roles)
        inferred_kind = infer_application_question_kind(question)
        if matched:
            selected_ids = [item.role_id for item in matched]
        elif (
            field_plan.resolution.application_subtype
            and field_plan.resolution.application_subtype in by_id
        ):
            selected_ids = [field_plan.resolution.application_subtype]
        elif inferred_kind == "component_map" or field == MaterialField.SEMICONDUCTOR:
            selected_ids = [item.role_id for item in allowed_roles]
        else:
            selected_ids = [item.role_id for item in allowed_roles]

    selected_roles = [by_id[role_id] for role_id in selected_ids]
    if application_model_run is not None:
        context.update(
            _validated_extracted_context(
                application_model_run.decision,
                roles_by_id=by_id,
                selected_role_ids=selected_ids,
                question=question,
                input_context=context,
            )
        )
    question_kind = model_question_kind or infer_application_question_kind(question)
    clarification: str | None = None
    model_clarifies = bool(
        application_model_run
        and application_model_run.decision.needs_clarification
    )
    if model_clarifies:
        clarification = application_model_run.decision.clarification_question
    elif (
        require_condition_complete
        and any(
            _context_value_is_missing(context.get(name))
            for role in selected_roles
            for name in _role_required_context(role)
        )
    ):
        missing = list(
            dict.fromkeys(
                name
                for role in selected_roles
                for name in _role_required_context(role)
                if _context_value_is_missing(context.get(name))
            )
        )
        clarification = (
            "Provide the missing operating and integration conditions: "
            + ", ".join(missing)
        )

    if clarification:
        mode: ApplicationDecompositionMode = "needs-clarification"
    elif len(selected_roles) == 1:
        mode = "single-role"
    else:
        mode = "role-portfolio"

    missing_by_role = {
        role.role_id: [
            name
            for name in _role_required_context(role)
            if _context_value_is_missing(context.get(name))
        ]
        for role in selected_roles
    }
    seeds_by_role = {
        role.role_id: list(role.candidate_seeds)
        for role in selected_roles
    }
    evidence_tasks = [
        task for role in selected_roles for task in role.evidence_tasks
    ]
    payload = {
        "question": question,
        "field_plan_id": field_plan.resolution.profile_id,
        "material_field": field,
        "question_kind": question_kind,
        "role_ids": selected_ids,
        "target_context": context,
        "application_model_decision_id": (
            application_model_run.decision_id
            if application_model_run is not None
            else None
        ),
    }
    return MaterialApplicationBrief(
        brief_id=f"MAB-{stable_hash(payload)[:24]}",
        user_question=question,
        material_field=field,
        question_kind=question_kind,
        decomposition_mode=mode,
        field_plan=field_plan,
        main_application_model_run=application_model_run,
        roles=selected_roles,
        target_context=context,
        missing_context_by_role=missing_by_role,
        candidate_seeds_by_role=seeds_by_role,
        evidence_tasks=evidence_tasks,
        clarification_question=clarification,
        ready_for_condition_complete_scoring=(
            mode != "needs-clarification"
            and not field_plan.resolution.requires_operator_choice
            and not model_clarifies
            and all(not names for names in missing_by_role.values())
        ),
    )


def build_main_model_material_application_classifier_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    required: bool = False,
) -> MainModelMaterialApplicationClassifier | None:
    """Reuse the trusted field/RAG reasoning endpoint for intent classification."""

    field_classifier = build_main_model_material_field_classifier_from_environment(
        environ=environ,
        required=required,
    )
    if field_classifier is None:
        return None
    return MainModelMaterialApplicationClassifier(field_classifier.model)


def _normalize_for_match(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


def _term_in_text(haystack: str, term: str) -> bool:
    normalized_term = _normalize_for_match(term)
    if not normalized_term:
        return False
    if re.search(r"[가-힣]", normalized_term):
        return normalized_term in haystack
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_term)}(?![a-z0-9])",
            haystack,
        )
        is not None
    )


def _stable_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _role_required_context(role: ApplicationRoleProfile) -> list[str]:
    """Return every condition needed by the role and its ranking criteria."""

    return list(
        dict.fromkeys(
            (
                *role.required_problem_context,
                *(
                    name
                    for criterion in role.criteria
                    if criterion.required_for_ranking
                    for name in criterion.required_context
                ),
            )
        )
    )


def _semantic_text(value: str) -> str:
    return re.sub(r"[^\w]+", " ", value.casefold(), flags=re.UNICODE).strip()


def _semantic_phrase_present(haystack: str, needle: str) -> bool:
    normalized_haystack = _semantic_text(haystack)
    normalized_needle = _semantic_text(needle)
    if not normalized_haystack or not normalized_needle:
        return False
    return f" {normalized_needle} " in f" {normalized_haystack} "


def _literal_scalar_supported(value: JsonValue, question: str) -> bool:
    normalized_question = _normalize_evidence_text(question)
    if isinstance(value, bool):
        return _semantic_phrase_present(
            normalized_question,
            "true" if value else "false",
        )
    if isinstance(value, (int, float)):
        candidates = {str(value), format(value, "g")}
        if float(value).is_integer():
            candidates.add(str(int(value)))
        return any(
            re.search(
                rf"(?<![0-9.]){re.escape(candidate)}(?![0-9.])",
                normalized_question,
            )
            is not None
            for candidate in candidates
        )
    if isinstance(value, str):
        normalized = _normalize_evidence_text(value)
        return bool(normalized) and normalized in normalized_question
    return False


def _validated_extracted_context(
    decision: MaterialApplicationModelDecision,
    *,
    roles_by_id: Mapping[str, ApplicationRoleProfile],
    selected_role_ids: Sequence[str],
    question: str,
    input_context: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
    allowed_new_keys = {
        name
        for role_id in selected_role_ids
        if role_id in roles_by_id
        for name in _role_required_context(roles_by_id[role_id])
    }
    verified: dict[str, JsonValue] = {}
    for key, value in decision.extracted_context.items():
        if key in input_context:
            if _stable_json(value) != _stable_json(input_context[key]):
                raise ValueError(
                    "main model extracted context must exactly match input context"
                )
            verified[key] = value
            continue
        if key not in allowed_new_keys:
            raise ValueError(
                "main model extracted a context key outside selected role requirements"
            )
        if isinstance(value, list):
            if not value or any(
                isinstance(item, (list, dict))
                or item is None
                or not _literal_scalar_supported(item, question)
                for item in value
            ):
                raise ValueError(
                    "new model-extracted context lists need literal scalar support"
                )
        elif (
            isinstance(value, dict)
            or value is None
            or not _literal_scalar_supported(value, question)
        ):
            raise ValueError(
                "new model-extracted context values need literal question support"
            )
        verified[key] = value
    return verified


def _decision_has_meaningful_application_evidence(
    decision: MaterialApplicationModelDecision,
    *,
    material_field: MaterialField,
    roles_by_id: Mapping[str, ApplicationRoleProfile],
) -> bool:
    terms: list[str] = [
        material_field.value.replace("_", " "),
        MATERIAL_FIELD_PROFILES[material_field].name,
    ]
    for role_id in decision.selected_role_ids:
        role = roles_by_id.get(role_id)
        if role is None:
            continue
        terms.extend((role.role_id.replace("_", " "), role.display_name))
        terms.extend(role.aliases)
    if decision.application_subtype:
        terms.append(decision.application_subtype.replace("_", " "))
    normalized_terms = {
        _semantic_text(term)
        for term in terms
        if _semantic_text(term)
    }
    for span in decision.evidence_spans:
        normalized_span = _semantic_text(span)
        if not normalized_span:
            continue
        for term in normalized_terms:
            if normalized_span == term:
                return True
            if len(normalized_span) >= 4 and (
                _semantic_phrase_present(normalized_span, term)
                or _semantic_phrase_present(term, normalized_span)
            ):
                return True
    return False


def _validate_application_model_decision(
    decision: MaterialApplicationModelDecision,
    *,
    material_field: MaterialField,
    roles_by_id: Mapping[str, ApplicationRoleProfile],
    question: str,
    input_context: Mapping[str, JsonValue],
) -> None:
    unknown_roles = [
        role_id
        for role_id in decision.selected_role_ids
        if role_id not in roles_by_id
    ]
    if unknown_roles:
        raise ValueError("main model selected a role outside the code allowlist")
    allowed_criteria = {
        item.criterion_id
        for role_id in decision.selected_role_ids
        for item in roles_by_id[role_id].criteria
    }
    if any(
        criterion_id not in allowed_criteria
        for criterion_id in decision.objective_priorities
    ):
        raise ValueError("main model selected an objective outside the role criteria")
    if decision.confidence < 0.70:
        raise ValueError("main model application confidence is below 0.70")
    allowed_subtypes = {
        *decision.selected_role_ids,
        *MATERIAL_FIELD_PROFILES[material_field].application_subtypes,
    }
    if (
        decision.application_subtype is not None
        and decision.application_subtype not in allowed_subtypes
    ):
        raise ValueError(
            "main model selected an application subtype outside the code allowlist"
        )
    corpus = _normalize_evidence_text(
        " ".join(
            [
                question,
                json.dumps(
                    input_context,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            ]
        )
    )
    if any(
        _normalize_evidence_text(span) not in corpus
        for span in decision.evidence_spans
    ):
        raise ValueError(
            "main model application decision cited evidence outside the input"
        )
    if not _decision_has_meaningful_application_evidence(
        decision,
        material_field=material_field,
        roles_by_id=roles_by_id,
    ):
        raise ValueError(
            "main model evidence does not meaningfully support the application role"
        )
    _validated_extracted_context(
        decision,
        roles_by_id=roles_by_id,
        selected_role_ids=decision.selected_role_ids,
        question=question,
        input_context=input_context,
    )


__all__ = [
    "APPLICATION_ROLE_PROFILES",
    "ApplicationCandidateSeed",
    "ApplicationCriterion",
    "ApplicationEvidenceTask",
    "ApplicationRoleProfile",
    "MainModelMaterialApplicationClassifier",
    "MaterialApplicationBrief",
    "MaterialApplicationModelDecision",
    "MaterialApplicationModelRun",
    "SEMICONDUCTOR_APPLICATION_ROLES",
    "application_roles_for_field",
    "build_main_model_material_application_classifier_from_environment",
    "build_material_application_brief",
    "get_application_role_profile",
    "infer_application_question_kind",
]
