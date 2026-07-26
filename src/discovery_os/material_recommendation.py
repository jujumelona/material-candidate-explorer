"""Evidence-closed, role-scoped material recommendation reports.

The ranking contract deliberately separates:

* literature/RAG support for why a material family should be investigated;
* search-triage priority from a generator or exploration loop;
* condition-complete properties from named numerical or experimental
  validators; and
* pool-relative decision support computed only inside one component role and
  one exact operating-condition group.

No citation count, RAG record count, search priority, model confidence, or
missing value becomes material-performance credit.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from pydantic import Field, model_validator

from .hashing import stable_hash
from .literature_rag import (
    EvidenceClaim,
    LiteratureRecord,
    RagEvidenceBundle,
)
from .material_applications import (
    ApplicationCriterion,
    ApplicationRoleProfile,
    CriterionCategory,
    CriterionDirection,
    MaterialApplicationBrief,
)
from .material_domains import (
    _contains_sensitive_context_key,
    _context_value_is_missing,
)
from .schemas import (
    Identifier,
    JsonValue,
    NonEmptyText,
    Probability,
    StrictSchema,
)


CandidateOrigin = Literal[
    "retrieval_seed",
    "reported_reference",
    "structured_database",
    "generated",
    "user_supplied",
]
ObservationStatus = Literal["success", "failed", "unknown", "incomparable"]
CriterionStatus = Literal["available", "unknown", "incomparable", "conflicting"]
HardGateStatus = Literal["pass", "fail", "unknown", "not_configured"]
EvidenceUncertaintyStatus = Literal[
    "bounded",
    "point_only",
    "unknown",
    "conflicting",
]
IdentityStatus = Literal[
    "match",
    "database_scoped_no_match",
    "unknown",
    "not_checked",
]
ModelDisagreementStatus = Literal["low", "medium", "high", "unknown", "not_applicable"]


class MaterialDecisionPreference(StrictSchema):
    criterion_id: Identifier
    weight: float = Field(default=0.0, ge=0.0, le=100.0)
    direction_override: Literal["maximize", "minimize", "target", "range"] | None = None
    hard_minimum: float | None = None
    hard_maximum: float | None = None
    target_value: float | None = None
    target_tolerance: float | None = Field(default=None, ge=0.0)
    source: Literal["operator", "source_closed_spec"] = "operator"
    provenance_id: Identifier | None = None

    @model_validator(mode="after")
    def _preference_is_complete(self) -> "MaterialDecisionPreference":
        values = (
            self.hard_minimum,
            self.hard_maximum,
            self.target_value,
            self.target_tolerance,
        )
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("decision preference values must be finite")
        if (
            self.hard_minimum is not None
            and self.hard_maximum is not None
            and self.hard_minimum > self.hard_maximum
        ):
            raise ValueError("hard minimum cannot exceed hard maximum")
        direction = self.direction_override
        if direction == "target" and self.target_value is None:
            raise ValueError("target direction requires target_value")
        if direction == "range" and (
            self.hard_minimum is None or self.hard_maximum is None
        ):
            raise ValueError("range direction requires hard minimum and maximum")
        if self.target_tolerance is not None and self.target_value is None:
            raise ValueError("target tolerance requires target_value")
        if self.source == "source_closed_spec" and self.provenance_id is None:
            raise ValueError("source-closed preference requires provenance")
        return self


class MaterialApplicationCandidate(StrictSchema):
    candidate_id: Identifier
    role_id: Identifier
    material_or_stack: NonEmptyText
    phase_or_stack: str | None = Field(default=None, max_length=4_000)
    origin: CandidateOrigin
    candidate_ref: str | None = Field(default=None, max_length=1_000)
    structure_or_record_id: str | None = Field(default=None, max_length=1_000)
    evidence_claim_ids: list[Identifier] = Field(default_factory=list)
    research_reference_ids: list[NonEmptyText] = Field(default_factory=list)
    provenance_id: Identifier
    triage_priority_score: Probability | None = None
    triage_score_semantics: Literal[
        "search-priority-only-not-application-fitness"
    ] = "search-priority-only-not-application-fitness"
    model_disagreement: ModelDisagreementStatus = "unknown"
    external_identity_status: IdentityStatus = "not_checked"

    @model_validator(mode="after")
    def _candidate_references_are_unique(self) -> "MaterialApplicationCandidate":
        if len(self.evidence_claim_ids) != len(set(self.evidence_claim_ids)):
            raise ValueError("candidate evidence claim identifiers must be unique")
        if len(self.research_reference_ids) != len(
            set(self.research_reference_ids)
        ):
            raise ValueError("candidate research references must be unique")
        if self.origin == "retrieval_seed" and self.triage_priority_score is not None:
            raise ValueError("retrieval seeds cannot have search priority")
        return self


class MaterialApplicationObservation(StrictSchema):
    observation_id: Identifier
    candidate_id: Identifier
    role_id: Identifier
    property_name: Identifier
    validator_id: Identifier
    status: ObservationStatus
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: NonEmptyText
    conditions: dict[str, JsonValue] = Field(default_factory=dict)
    method: NonEmptyText
    sample_or_model_scope: NonEmptyText
    authority_kind: Literal[
        "numerical_validator",
        "experimental_validator",
        "trusted_structured_database_validator",
    ]
    uncertainty_kind: Literal[
        "calibrated_interval",
        "measurement_interval",
        "numerical_interval",
        "not_quantified",
    ] = "not_quantified"
    provenance_id: Identifier
    raw_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    literature_or_mcp_derived: Literal[False] = False

    @model_validator(mode="after")
    def _observation_value_contract(self) -> "MaterialApplicationObservation":
        if _contains_sensitive_context_key(self.conditions):
            raise ValueError("application observations cannot contain secrets")
        numeric = (self.value, self.lower_bound, self.upper_bound)
        if self.status == "success":
            if self.value is None or not math.isfinite(self.value):
                raise ValueError("successful observation requires a finite value")
            if (self.lower_bound is None) != (self.upper_bound is None):
                raise ValueError("uncertainty bounds must be present together")
            if self.lower_bound is not None:
                assert self.upper_bound is not None
                if not (
                    math.isfinite(self.lower_bound)
                    and math.isfinite(self.upper_bound)
                ):
                    raise ValueError("uncertainty bounds must be finite")
                if not self.lower_bound <= self.value <= self.upper_bound:
                    raise ValueError("uncertainty bounds must contain the value")
                if self.uncertainty_kind == "not_quantified":
                    raise ValueError("bounded observation needs an uncertainty kind")
            elif self.uncertainty_kind != "not_quantified":
                raise ValueError(
                    "quantified uncertainty kind requires lower and upper bounds"
                )
        elif any(value is not None for value in numeric):
            raise ValueError("non-success observations cannot expose numeric values")
        return self


class RecommendationCitation(StrictSchema):
    claim_id: Identifier
    record_id: Identifier
    title: NonEmptyText
    doi: str | None = Field(default=None, max_length=512)
    urls: list[str] = Field(default_factory=list)
    exact_support_span: NonEmptyText
    polarity: Literal["supports", "contradicts", "null", "uncertain"]
    retrieved_record_only_not_property_validator: Literal[True] = True


class CandidateCriterionResult(StrictSchema):
    criterion_id: Identifier
    property_name: Identifier
    category: CriterionCategory
    direction: CriterionDirection
    status: CriterionStatus
    value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: NonEmptyText
    conditions: dict[str, JsonValue] = Field(default_factory=dict)
    accepted_observation_ids: list[Identifier] = Field(default_factory=list)
    rejected_observation_ids: list[Identifier] = Field(default_factory=list)
    hard_gate_status: HardGateStatus = "not_configured"
    reason_code: Identifier
    reason: NonEmptyText
    value_aggregation_performed: Literal[False] = False
    literature_or_mcp_property_substitution_performed: Literal[False] = False
    unknown_is_pass: Literal[False] = False

    @model_validator(mode="after")
    def _criterion_result_is_consistent(self) -> "CandidateCriterionResult":
        if len(self.accepted_observation_ids) != len(
            set(self.accepted_observation_ids)
        ):
            raise ValueError("accepted observation ids must be unique")
        if len(self.rejected_observation_ids) != len(
            set(self.rejected_observation_ids)
        ):
            raise ValueError("rejected observation ids must be unique")
        if set(self.accepted_observation_ids) & set(self.rejected_observation_ids):
            raise ValueError("accepted and rejected observations must be disjoint")
        values = (self.value, self.lower_bound, self.upper_bound)
        if self.status == "available":
            if self.value is None or not self.accepted_observation_ids:
                raise ValueError("available criterion requires value and evidence")
            if (self.lower_bound is None) != (self.upper_bound is None):
                raise ValueError("criterion bounds must be present together")
        elif any(value is not None for value in values):
            raise ValueError("unavailable criterion cannot expose a value")
        if self.status in {"unknown", "incomparable"} and self.accepted_observation_ids:
            raise ValueError("unknown/incomparable criterion cannot accept evidence")
        if self.status == "conflicting" and not self.accepted_observation_ids:
            raise ValueError("conflicting criterion must preserve successful evidence")
        return self


class MaterialRecommendationCandidate(StrictSchema):
    candidate: MaterialApplicationCandidate
    comparison_group_id: Identifier | None = None
    rank_within_role_and_condition: int | None = Field(default=None, gt=0)
    pareto_front: int | None = Field(default=None, gt=0)
    hard_gate_status: HardGateStatus
    criterion_results: list[CandidateCriterionResult] = Field(min_length=1)
    performance_vector: list[Identifier] = Field(default_factory=list)
    reliability_vector: list[Identifier] = Field(default_factory=list)
    integration_vector: list[Identifier] = Field(default_factory=list)
    resource_safety_vector: list[Identifier] = Field(default_factory=list)
    evidence_completeness_score: float = Field(ge=0.0, le=100.0)
    evidence_uncertainty_status: EvidenceUncertaintyStatus
    pool_relative_decision_score: float | None = Field(
        default=None, ge=0.0, le=100.0
    )
    score_semantics: Literal[
        "operator-weighted-pool-relative-decision-support-or-null"
    ] = "operator-weighted-pool-relative-decision-support-or-null"
    why_selected: list[Identifier] = Field(min_length=1)
    why_not_top: list[Identifier] = Field(default_factory=list)
    main_tradeoffs: list[NonEmptyText] = Field(default_factory=list)
    uncertainty_reasons: list[NonEmptyText] = Field(default_factory=list)
    citations: list[RecommendationCitation] = Field(default_factory=list)
    next_validations: list[NonEmptyText] = Field(default_factory=list)
    claim_boundary: Literal[
        "decision-support-candidate-not-scientific-discovery"
    ] = "decision-support-candidate-not-scientific-discovery"

    @model_validator(mode="after")
    def _candidate_result_is_closed(self) -> "MaterialRecommendationCandidate":
        criterion_ids = [item.criterion_id for item in self.criterion_results]
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("candidate criterion results must be unique")
        categorized = {
            "performance": self.performance_vector,
            "reliability": self.reliability_vector,
            "integration": self.integration_vector,
            "resource_safety": self.resource_safety_vector,
        }
        for category, identifiers in categorized.items():
            expected = [
                item.criterion_id
                for item in self.criterion_results
                if item.category == category
            ]
            if identifiers != expected:
                raise ValueError("candidate criterion vectors are inconsistent")
        if self.pool_relative_decision_score is not None:
            if self.comparison_group_id is None or self.hard_gate_status == "fail":
                raise ValueError("scored candidate needs a valid comparison group")
        if self.rank_within_role_and_condition is not None and (
            self.comparison_group_id is None
        ):
            raise ValueError("ranked candidate needs a comparison group")
        return self


class MaterialRoleRecommendation(StrictSchema):
    role_id: Identifier
    role_profile: ApplicationRoleProfile
    candidates: list[MaterialRecommendationCandidate] = Field(min_length=1)
    comparison_group_count: int = Field(ge=0)
    scalar_score_created_without_operator_weights: Literal[False] = False
    cross_condition_ranking_performed: Literal[False] = False
    literature_or_mcp_performance_scoring_performed: Literal[False] = False
    role_claim_boundary: NonEmptyText

    @model_validator(mode="after")
    def _portfolio_matches_profile(self) -> "MaterialRoleRecommendation":
        if self.role_id != self.role_profile.role_id:
            raise ValueError("role recommendation profile mismatch")
        if any(item.candidate.role_id != self.role_id for item in self.candidates):
            raise ValueError("candidate belongs to another role")
        group_ids = {
            item.comparison_group_id
            for item in self.candidates
            if item.comparison_group_id is not None
        }
        if self.comparison_group_count != len(group_ids):
            raise ValueError("comparison group count is inconsistent")
        if self.role_claim_boundary != self.role_profile.claim_boundary:
            raise ValueError("role claim boundary must come from the profile")
        return self


class MaterialRecommendationReport(StrictSchema):
    report_id: Identifier
    brief: MaterialApplicationBrief
    role_recommendations: list[MaterialRoleRecommendation] = Field(min_length=1)
    rag_bundle_id: Identifier | None = None
    unresolved_questions: list[NonEmptyText] = Field(default_factory=list)
    warnings: list[NonEmptyText] = Field(default_factory=list)
    cross_role_ranking_performed: Literal[False] = False
    search_priority_used_as_application_fitness: Literal[False] = False
    citation_or_record_count_used_as_performance_score: Literal[False] = False
    missing_value_imputed_as_zero: Literal[False] = False
    overall_claim_boundary: Literal[
        "role-scoped-decision-support-not-proof-of-suitability-or-novelty"
    ] = "role-scoped-decision-support-not-proof-of-suitability-or-novelty"

    @model_validator(mode="after")
    def _report_covers_brief_roles(self) -> "MaterialRecommendationReport":
        expected = [item.role_id for item in self.brief.roles]
        actual = [item.role_id for item in self.role_recommendations]
        if actual != expected:
            raise ValueError("recommendation report must cover brief roles in order")
        return self


def candidates_from_application_seeds(
    brief: MaterialApplicationBrief,
) -> list[MaterialApplicationCandidate]:
    """Materialize unscored retrieval seeds for a scenario-map response."""

    candidates: list[MaterialApplicationCandidate] = []
    seen: set[tuple[str, str]] = set()
    for role in brief.roles:
        for seed in brief.candidate_seeds_by_role[role.role_id]:
            for index, example in enumerate(seed.examples, start=1):
                key = (role.role_id, example.casefold())
                if key in seen:
                    continue
                seen.add(key)
                payload = {
                    "brief_id": brief.brief_id,
                    "role_id": role.role_id,
                    "seed_id": seed.seed_id,
                    "example": example,
                }
                candidates.append(
                    MaterialApplicationCandidate(
                        candidate_id=f"SEED-{stable_hash(payload)[:20]}",
                        role_id=role.role_id,
                        material_or_stack=example,
                        phase_or_stack=seed.material_family,
                        origin="retrieval_seed",
                        evidence_claim_ids=[],
                        research_reference_ids=seed.research_reference_ids,
                        provenance_id=seed.seed_id,
                    )
                )
    return candidates


def rank_material_application_candidates(
    brief: MaterialApplicationBrief,
    *,
    candidates: Iterable[MaterialApplicationCandidate],
    observations: Iterable[MaterialApplicationObservation] = (),
    preferences: Iterable[MaterialDecisionPreference] = (),
    rag_bundle: RagEvidenceBundle | None = None,
) -> MaterialRecommendationReport:
    """Build role- and condition-scoped candidate portfolios.

    A scalar score is produced only when the operator supplies non-zero
    criterion weights and at least two hard-gate-eligible candidates have all
    required, comparable observations in the same exact condition group.
    """

    candidate_rows = list(candidates)
    observation_rows = list(observations)
    preference_rows = list(preferences)
    if not candidate_rows:
        raise ValueError("material recommendation needs at least one candidate")
    candidate_ids = [item.candidate_id for item in candidate_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("material recommendation candidate ids must be unique")
    selected_roles = {item.role_id: item for item in brief.roles}
    if any(item.role_id not in selected_roles for item in candidate_rows):
        raise ValueError("candidate role was not selected in the application brief")
    known_candidates = set(candidate_ids)
    if any(item.candidate_id not in known_candidates for item in observation_rows):
        raise ValueError("observation cites an unknown application candidate")
    candidate_role = {
        item.candidate_id: item.role_id for item in candidate_rows
    }
    if any(
        candidate_role[item.candidate_id] != item.role_id
        for item in observation_rows
    ):
        raise ValueError("observation role does not match its candidate")

    preference_by_id: dict[str, MaterialDecisionPreference] = {}
    for preference in preference_rows:
        if preference.criterion_id in preference_by_id:
            raise ValueError("decision preferences must be unique by criterion")
        preference_by_id[preference.criterion_id] = preference
    allowed_criteria = {
        criterion.criterion_id
        for role in brief.roles
        for criterion in role.criteria
    }
    if any(key not in allowed_criteria for key in preference_by_id):
        raise ValueError("decision preference references an unselected criterion")
    _validate_source_closed_preferences(preference_rows, rag_bundle)

    citations_by_candidate = _resolve_candidate_citations(
        candidate_rows,
        rag_bundle,
    )
    observations_by_candidate: dict[str, list[MaterialApplicationObservation]] = (
        defaultdict(list)
    )
    for observation in observation_rows:
        observations_by_candidate[observation.candidate_id].append(observation)

    portfolios: list[MaterialRoleRecommendation] = []
    for role in brief.roles:
        role_candidates = [
            item for item in candidate_rows if item.role_id == role.role_id
        ]
        if not role_candidates:
            raise ValueError(
                f"application brief role {role.role_id!r} has no candidate"
            )
        assessed: dict[str, list[CandidateCriterionResult]] = {}
        for candidate in role_candidates:
            assessed[candidate.candidate_id] = [
                _assess_criterion(
                    role,
                    candidate,
                    criterion,
                    observations_by_candidate[candidate.candidate_id],
                    brief.target_context,
                    preference_by_id.get(criterion.criterion_id),
                )
                for criterion in role.criteria
            ]
        comparison_blockers = _role_comparison_blockers(brief, role)
        group_members = (
            {}
            if comparison_blockers
            else _condition_groups(
                role,
                role_candidates,
                assessed,
                preference_by_id,
            )
        )
        group_rankings: dict[str, tuple[dict[str, int], dict[str, float | None]]] = {}
        for group_id, members in group_members.items():
            fronts = _robust_pareto_fronts(
                role,
                members,
                assessed,
                preference_by_id,
            )
            scores = _pool_relative_scores(
                role,
                members,
                assessed,
                preference_by_id,
            )
            ordered = sorted(
                members,
                key=lambda item: (
                    fronts.get(item.candidate_id, 10**9),
                    -(
                        scores[item.candidate_id]
                        if scores.get(item.candidate_id) is not None
                        else -1.0
                    ),
                    item.candidate_id,
                ),
            )
            ranks: dict[str, int] = {}
            previous_rank_key: tuple[int, float | None] | None = None
            dense_rank = 0
            for item in ordered:
                rank_key = (
                    fronts.get(item.candidate_id, 10**9),
                    scores.get(item.candidate_id),
                )
                if rank_key != previous_rank_key:
                    dense_rank += 1
                    previous_rank_key = rank_key
                ranks[item.candidate_id] = dense_rank
            group_rankings[group_id] = (ranks, scores)

        results: list[MaterialRecommendationCandidate] = []
        member_group = {
            candidate.candidate_id: group_id
            for group_id, members in group_members.items()
            for candidate in members
        }
        fronts_by_group = {
            group_id: _robust_pareto_fronts(
                role,
                members,
                assessed,
                preference_by_id,
            )
            for group_id, members in group_members.items()
        }
        for candidate in role_candidates:
            criteria = assessed[candidate.candidate_id]
            group_id = member_group.get(candidate.candidate_id)
            ranks: dict[str, int] = {}
            scores: dict[str, float | None] = {}
            if group_id is not None:
                ranks, scores = group_rankings[group_id]
            hard_gate = _candidate_hard_gate(criteria)
            required = [
                item
                for item, profile_criterion in zip(
                    criteria,
                    role.criteria,
                    strict=True,
                )
                if profile_criterion.required_for_ranking
            ]
            available_count = sum(
                item.status == "available" for item in required
            )
            completeness = (
                100.0 * available_count / len(required) if required else 100.0
            )
            uncertainty_status = _uncertainty_status(required)
            pareto_front = (
                fronts_by_group[group_id].get(candidate.candidate_id)
                if group_id is not None
                else None
            )
            score = scores.get(candidate.candidate_id) if group_id else None
            why_selected, why_not_top, tradeoffs = _candidate_explanations(
                candidate,
                criteria,
                pareto_front,
                score,
                hard_gate,
            )
            uncertainty_reasons = _uncertainty_reasons(candidate, criteria)
            next_validations = _next_validations(role, criteria, candidate)
            by_category = {
                category: [
                    item.criterion_id
                    for item in criteria
                    if item.category == category
                ]
                for category in (
                    "performance",
                    "reliability",
                    "integration",
                    "resource_safety",
                )
            }
            results.append(
                MaterialRecommendationCandidate(
                    candidate=candidate,
                    comparison_group_id=group_id,
                    rank_within_role_and_condition=(
                        ranks.get(candidate.candidate_id) if group_id else None
                    ),
                    pareto_front=pareto_front,
                    hard_gate_status=hard_gate,
                    criterion_results=criteria,
                    performance_vector=by_category["performance"],
                    reliability_vector=by_category["reliability"],
                    integration_vector=by_category["integration"],
                    resource_safety_vector=by_category["resource_safety"],
                    evidence_completeness_score=round(completeness, 6),
                    evidence_uncertainty_status=uncertainty_status,
                    pool_relative_decision_score=score,
                    why_selected=why_selected,
                    why_not_top=why_not_top,
                    main_tradeoffs=tradeoffs,
                    uncertainty_reasons=uncertainty_reasons,
                    citations=citations_by_candidate[candidate.candidate_id],
                    next_validations=next_validations,
                )
            )
        results.sort(
            key=lambda item: (
                item.comparison_group_id or "~unscored",
                item.rank_within_role_and_condition or 10**9,
                -item.evidence_completeness_score,
                item.candidate.candidate_id,
            )
        )
        portfolios.append(
            MaterialRoleRecommendation(
                role_id=role.role_id,
                role_profile=role,
                candidates=results,
                comparison_group_count=len(group_members),
                role_claim_boundary=role.claim_boundary,
            )
        )

    unresolved = list(
        dict.fromkeys(
            (
                [brief.clarification_question]
                if brief.clarification_question
                else []
            )
            + [
                f"{role_id}: {', '.join(names)}"
                for role_id, names in brief.missing_context_by_role.items()
                if names
            ]
            + [
                f"{role.role_id}: {reason}"
                for role in brief.roles
                for reason in _role_comparison_blockers(brief, role)
            ]
        )
    )
    warnings = [
        "Candidates are compared only inside one component role and exact condition group.",
        "Pool-relative decision scores require explicit operator weights and are not probabilities.",
        "RAG citations support investigation context; they do not validate generated-candidate properties.",
        "A database-scoped no-match is not proof of scientific novelty.",
    ]
    if any(_role_comparison_blockers(brief, role) for role in brief.roles):
        warnings.append(
            "At least one role remains an unscored portfolio because field routing, "
            "clarification, role context, or required target conditions are unresolved."
        )
    payload = {
        "brief_id": brief.brief_id,
        "candidate_ids": candidate_ids,
        "observation_ids": [item.observation_id for item in observation_rows],
        "preference_ids": [
            {
                "criterion_id": item.criterion_id,
                "preference": item,
            }
            for item in preference_rows
        ],
        "rag_bundle_id": rag_bundle.bundle_id if rag_bundle else None,
    }
    return MaterialRecommendationReport(
        report_id=f"MAR-{stable_hash(payload)[:24]}",
        brief=brief,
        role_recommendations=portfolios,
        rag_bundle_id=rag_bundle.bundle_id if rag_bundle else None,
        unresolved_questions=unresolved,
        warnings=warnings,
    )


def _validate_source_closed_preferences(
    preferences: Sequence[MaterialDecisionPreference],
    bundle: RagEvidenceBundle | None,
) -> None:
    source_closed = [
        item for item in preferences if item.source == "source_closed_spec"
    ]
    if not source_closed:
        return
    if bundle is None:
        raise ValueError(
            "source-closed decision preferences require a supplied RAG evidence bundle"
        )
    supporting_claim_ids = {
        item.claim_id
        for item in bundle.claims
        if str(item.polarity) == "supports"
    }
    for preference in source_closed:
        assert preference.provenance_id is not None
        if preference.provenance_id not in supporting_claim_ids:
            raise ValueError(
                "source-closed decision preference provenance must match a "
                "supporting claim in the supplied RAG bundle"
            )


def _role_comparison_blockers(
    brief: MaterialApplicationBrief,
    role: ApplicationRoleProfile,
) -> list[str]:
    """Return reasons this role must remain an unscored portfolio."""

    blockers: list[str] = []
    if brief.field_plan.resolution.requires_operator_choice:
        blockers.append("material-field routing requires an operator choice")
    if brief.decomposition_mode == "needs-clarification":
        blockers.append(
            brief.clarification_question
            or "the application brief requires clarification"
        )
    missing_role_context = brief.missing_context_by_role.get(role.role_id, [])
    if missing_role_context:
        blockers.append(
            "missing role context: " + ", ".join(missing_role_context)
        )
    required_target_conditions = list(
        dict.fromkeys(
            name
            for criterion in role.criteria
            if criterion.required_for_ranking
            for name in criterion.required_context
        )
    )
    missing_target_conditions = [
        name
        for name in required_target_conditions
        if _context_value_is_missing(brief.target_context.get(name))
    ]
    if missing_target_conditions:
        blockers.append(
            "missing required target conditions: "
            + ", ".join(missing_target_conditions)
        )
    return list(dict.fromkeys(blockers))


def _assess_criterion(
    role: ApplicationRoleProfile,
    candidate: MaterialApplicationCandidate,
    criterion: ApplicationCriterion,
    observations: Sequence[MaterialApplicationObservation],
    target_context: Mapping[str, JsonValue],
    preference: MaterialDecisionPreference | None,
) -> CandidateCriterionResult:
    relevant = [
        item for item in observations if item.property_name == criterion.property_name
    ]
    accepted: list[MaterialApplicationObservation] = []
    rejected: list[str] = []
    incomparable = False
    for item in relevant:
        if item.role_id != role.role_id or item.candidate_id != candidate.candidate_id:
            raise ValueError("criterion observation candidate/role mismatch")
        conditions_complete = all(
            not _context_value_is_missing(item.conditions.get(name))
            for name in criterion.required_context
        )
        target_matches = all(
            name not in target_context
            or _context_value_is_missing(target_context.get(name))
            or _stable_json(item.conditions.get(name))
            == _stable_json(target_context[name])
            for name in criterion.required_context
        )
        if (
            item.validator_id not in criterion.validator_ids
            or item.unit != criterion.unit
            or not conditions_complete
            or not target_matches
        ):
            rejected.append(item.observation_id)
            incomparable = True
        elif item.status == "success":
            accepted.append(item)
        elif item.status == "incomparable":
            incomparable = True
            rejected.append(item.observation_id)
        else:
            rejected.append(item.observation_id)

    condition_groups: dict[str, list[MaterialApplicationObservation]] = defaultdict(list)
    for item in accepted:
        scoped = {
            name: item.conditions[name] for name in criterion.required_context
        }
        condition_groups[_stable_json(scoped)].append(item)
    if len(condition_groups) > 1:
        rejected.extend(item.observation_id for item in accepted)
        accepted = []
        incomparable = True

    signatures = {
        (
            item.value,
            item.lower_bound,
            item.upper_bound,
            item.unit,
            _stable_json(
                {
                    name: item.conditions[name]
                    for name in criterion.required_context
                }
            ),
        )
        for item in accepted
    }
    direction = (
        preference.direction_override
        if preference and preference.direction_override
        else criterion.direction
    )
    if len(signatures) > 1:
        status: CriterionStatus = "conflicting"
        value = lower = upper = None
        conditions: dict[str, JsonValue] = {}
        reason_code = "CONFLICTING_NAMED_VALIDATORS"
        reason = (
            "Named validators returned different values or uncertainty intervals "
            "under the same required conditions; no averaging was performed."
        )
        hard_gate: HardGateStatus = "unknown"
    elif accepted:
        status = "available"
        value = accepted[0].value
        lower = accepted[0].lower_bound
        upper = accepted[0].upper_bound
        conditions = {
            name: accepted[0].conditions[name]
            for name in criterion.required_context
        }
        hard_gate = _criterion_hard_gate(
            value,
            lower,
            upper,
            preference,
        )
        reason_code = "CONDITION_COMPLETE_NAMED_VALIDATOR"
        reason = (
            "A named validator returned the exact unit and complete required "
            "conditions. The original observation remains the authority."
        )
    elif incomparable:
        status = "incomparable"
        value = lower = upper = None
        conditions = {}
        hard_gate = "unknown" if _preference_has_gate(preference) else "not_configured"
        reason_code = "UNIT_VALIDATOR_OR_CONDITION_MISMATCH"
        reason = (
            "Available rows used an unapproved validator, wrong unit, incomplete "
            "conditions, or conditions different from the requested target."
        )
    else:
        status = "unknown"
        value = lower = upper = None
        conditions = {}
        hard_gate = "unknown" if _preference_has_gate(preference) else "not_configured"
        reason_code = "NO_SUCCESSFUL_NAMED_VALIDATOR"
        reason = "No successful named validator result is available."
    return CandidateCriterionResult(
        criterion_id=criterion.criterion_id,
        property_name=criterion.property_name,
        category=criterion.category,
        direction=direction,
        status=status,
        value=value,
        lower_bound=lower,
        upper_bound=upper,
        unit=criterion.unit,
        conditions=conditions,
        accepted_observation_ids=[item.observation_id for item in accepted],
        rejected_observation_ids=list(dict.fromkeys(rejected)),
        hard_gate_status=hard_gate,
        reason_code=reason_code,
        reason=reason,
    )


def _criterion_hard_gate(
    value: float | None,
    lower_bound: float | None,
    upper_bound: float | None,
    preference: MaterialDecisionPreference | None,
) -> HardGateStatus:
    if not _preference_has_gate(preference):
        return "not_configured"
    assert preference is not None
    assert value is not None
    if lower_bound is None or upper_bound is None:
        # A point estimate with unquantified uncertainty is not a robust
        # constraint result. An exact deterministic authority can still expose
        # an explicit zero-width interval after applying its own policy.
        return "unknown"
    low = lower_bound if lower_bound is not None else value
    high = upper_bound if upper_bound is not None else value
    unknown = False
    if preference.hard_minimum is not None:
        if high < preference.hard_minimum:
            return "fail"
        if low < preference.hard_minimum:
            unknown = True
    if preference.hard_maximum is not None:
        if low > preference.hard_maximum:
            return "fail"
        if high > preference.hard_maximum:
            unknown = True
    if preference.target_value is not None and preference.target_tolerance is not None:
        target_low = preference.target_value - preference.target_tolerance
        target_high = preference.target_value + preference.target_tolerance
        if high < target_low or low > target_high:
            return "fail"
        if low < target_low or high > target_high:
            unknown = True
    return "unknown" if unknown else "pass"


def _preference_has_gate(
    preference: MaterialDecisionPreference | None,
) -> bool:
    return bool(
        preference
        and (
            preference.hard_minimum is not None
            or preference.hard_maximum is not None
            or (
                preference.target_value is not None
                and preference.target_tolerance is not None
            )
        )
    )


def _candidate_hard_gate(
    criteria: Sequence[CandidateCriterionResult],
) -> HardGateStatus:
    statuses = [item.hard_gate_status for item in criteria]
    if "fail" in statuses:
        return "fail"
    configured = [item for item in statuses if item != "not_configured"]
    if not configured:
        return "not_configured"
    if "unknown" in configured:
        return "unknown"
    return "pass"


def _condition_groups(
    role: ApplicationRoleProfile,
    candidates: Sequence[MaterialApplicationCandidate],
    assessed: Mapping[str, Sequence[CandidateCriterionResult]],
    preferences: Mapping[str, MaterialDecisionPreference],
) -> dict[str, list[MaterialApplicationCandidate]]:
    groups: dict[str, list[MaterialApplicationCandidate]] = defaultdict(list)
    for candidate in candidates:
        rows = assessed[candidate.candidate_id]
        required_pairs = [
            (criterion, result)
            for criterion, result in zip(role.criteria, rows, strict=True)
            if criterion.required_for_ranking
        ]
        if not required_pairs or any(
            result.status != "available" for _, result in required_pairs
        ):
            continue
        if any(
            result.lower_bound is None or result.upper_bound is None
            for _, result in required_pairs
        ):
            # Robust ranking needs explicit intervals. Point-only evidence is
            # retained in the report but is not treated as certain.
            continue
        if _candidate_hard_gate(rows) in {"fail", "unknown"}:
            continue
        if any(
            _direction_parameters(
                result.direction,
                preferences.get(criterion.criterion_id),
            )
            is None
            for criterion, result in required_pairs
        ):
            continue
        signature = {
            result.criterion_id: {
                "unit": result.unit,
                "conditions": result.conditions,
                "direction": result.direction,
                "target_parameters": _direction_parameters(
                    result.direction,
                    preferences.get(criterion.criterion_id),
                ),
            }
            for criterion, result in required_pairs
        }
        group_id = f"MCG-{stable_hash({'role': role.role_id, 'scope': signature})[:20]}"
        groups[group_id].append(candidate)
    return dict(groups)


def _direction_parameters(
    direction: CriterionDirection,
    preference: MaterialDecisionPreference | None,
) -> dict[str, float] | None:
    if direction in {"maximize", "minimize"}:
        return {}
    if direction == "target":
        if preference is None or preference.target_value is None:
            return None
        return {"target": preference.target_value}
    if direction == "range":
        if (
            preference is None
            or preference.hard_minimum is None
            or preference.hard_maximum is None
        ):
            return None
        return {
            "minimum": preference.hard_minimum,
            "maximum": preference.hard_maximum,
        }
    return None


def _utility_interval(
    result: CandidateCriterionResult,
    preference: MaterialDecisionPreference | None,
) -> tuple[float, float]:
    assert result.status == "available" and result.value is not None
    low = result.lower_bound if result.lower_bound is not None else result.value
    high = result.upper_bound if result.upper_bound is not None else result.value
    if result.direction == "maximize":
        return low, high
    if result.direction == "minimize":
        return -high, -low
    if result.direction == "target":
        assert preference is not None and preference.target_value is not None
        target = preference.target_value
        worst = -max(abs(low - target), abs(high - target))
        best = (
            0.0
            if low <= target <= high
            else -min(abs(low - target), abs(high - target))
        )
        return worst, best
    if result.direction == "range":
        assert preference is not None
        assert preference.hard_minimum is not None
        assert preference.hard_maximum is not None
        midpoint = (preference.hard_minimum + preference.hard_maximum) / 2.0
        worst = -max(abs(low - midpoint), abs(high - midpoint))
        best = (
            0.0
            if low <= midpoint <= high
            else -min(abs(low - midpoint), abs(high - midpoint))
        )
        return worst, best
    raise ValueError("user-defined direction needs an operator override")


def _robust_pareto_fronts(
    role: ApplicationRoleProfile,
    candidates: Sequence[MaterialApplicationCandidate],
    assessed: Mapping[str, Sequence[CandidateCriterionResult]],
    preferences: Mapping[str, MaterialDecisionPreference],
) -> dict[str, int]:
    if not candidates:
        return {}
    vectors: dict[str, list[tuple[float, float]]] = {}
    for candidate in candidates:
        vector: list[tuple[float, float]] = []
        for criterion, result in zip(
            role.criteria,
            assessed[candidate.candidate_id],
            strict=True,
        ):
            if not criterion.required_for_ranking:
                continue
            vector.append(
                _utility_interval(
                    result,
                    preferences.get(criterion.criterion_id),
                )
            )
        vectors[candidate.candidate_id] = vector

    remaining = {item.candidate_id for item in candidates}
    fronts: dict[str, int] = {}
    front_index = 1
    while remaining:
        current = [
            candidate_id
            for candidate_id in sorted(remaining)
            if not any(
                _robustly_dominates(vectors[other], vectors[candidate_id])
                for other in remaining
                if other != candidate_id
            )
        ]
        if not current:
            raise RuntimeError("robust Pareto sorting failed to make progress")
        for candidate_id in current:
            fronts[candidate_id] = front_index
            remaining.remove(candidate_id)
        front_index += 1
    return fronts


def _robustly_dominates(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> bool:
    if len(left) != len(right) or not left:
        return False
    no_worse = all(
        left_worst >= right_best
        for (left_worst, _), (_, right_best) in zip(left, right, strict=True)
    )
    strictly_better = any(
        left_worst > right_best
        for (left_worst, _), (_, right_best) in zip(left, right, strict=True)
    )
    return no_worse and strictly_better


def _pool_relative_scores(
    role: ApplicationRoleProfile,
    candidates: Sequence[MaterialApplicationCandidate],
    assessed: Mapping[str, Sequence[CandidateCriterionResult]],
    preferences: Mapping[str, MaterialDecisionPreference],
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {
        item.candidate_id: None for item in candidates
    }
    weighted = [
        criterion
        for criterion in role.criteria
        if preferences.get(criterion.criterion_id)
        and preferences[criterion.criterion_id].weight > 0
    ]
    if not weighted or len(candidates) < 2:
        return scores
    total_weight = sum(preferences[item.criterion_id].weight for item in weighted)
    utilities: dict[str, dict[str, float]] = {
        candidate.candidate_id: {} for candidate in candidates
    }
    for candidate in candidates:
        results = {
            item.criterion_id: item
            for item in assessed[candidate.candidate_id]
        }
        if any(results[item.criterion_id].status != "available" for item in weighted):
            return scores
        for criterion in weighted:
            result = results[criterion.criterion_id]
            worst, _ = _utility_interval(
                result,
                preferences[criterion.criterion_id],
            )
            utilities[candidate.candidate_id][criterion.criterion_id] = worst
    normalized: dict[str, dict[str, float]] = {
        candidate.candidate_id: {} for candidate in candidates
    }
    for criterion in weighted:
        values = [
            utilities[candidate.candidate_id][criterion.criterion_id]
            for candidate in candidates
        ]
        low = min(values)
        high = max(values)
        for candidate in candidates:
            value = utilities[candidate.candidate_id][criterion.criterion_id]
            normalized[candidate.candidate_id][criterion.criterion_id] = (
                0.5 if math.isclose(low, high) else (value - low) / (high - low)
            )
    for candidate in candidates:
        weighted_sum = sum(
            normalized[candidate.candidate_id][criterion.criterion_id]
            * preferences[criterion.criterion_id].weight
            for criterion in weighted
        )
        scores[candidate.candidate_id] = round(
            100.0 * weighted_sum / total_weight,
            6,
        )
    return scores


def _uncertainty_status(
    required: Sequence[CandidateCriterionResult],
) -> EvidenceUncertaintyStatus:
    if any(item.status == "conflicting" for item in required):
        return "conflicting"
    if any(item.status != "available" for item in required):
        return "unknown"
    if required and all(item.lower_bound is not None for item in required):
        return "bounded"
    return "point_only"


def _candidate_explanations(
    candidate: MaterialApplicationCandidate,
    criteria: Sequence[CandidateCriterionResult],
    pareto_front: int | None,
    score: float | None,
    hard_gate: HardGateStatus,
) -> tuple[list[str], list[str], list[str]]:
    why: list[str] = []
    why_not: list[str] = []
    tradeoffs: list[str] = []
    if candidate.origin == "retrieval_seed":
        why.append("SOURCE_BACKED_RETRIEVAL_SEED")
    if pareto_front == 1:
        why.append("ROBUST_NON_DOMINATED_WITHIN_ROLE_AND_CONDITION")
    elif pareto_front is not None:
        why.append(f"PARETO_FRONT_{pareto_front}")
        why_not.append("DOMINATED_WITHIN_MATCHED_ROLE_AND_CONDITION")
    if score is not None:
        why.append("OPERATOR_WEIGHTED_POOL_RELATIVE_SCORE_AVAILABLE")
    if hard_gate == "pass":
        why.append("ALL_CONFIGURED_HARD_GATES_PASS")
    elif hard_gate == "fail":
        why_not.append("AT_LEAST_ONE_CONFIGURED_HARD_GATE_FAILS")
    elif hard_gate == "unknown":
        why_not.append("HARD_GATE_UNCERTAINTY_OR_MISSING_EVIDENCE")
    available = [item for item in criteria if item.status == "available"]
    if available:
        why.append("NAMED_VALIDATOR_EVIDENCE_AVAILABLE")
    else:
        why_not.append("NO_CONDITION_COMPLETE_PERFORMANCE_EVIDENCE")
    for item in criteria:
        if item.status == "unknown":
            tradeoffs.append(
                f"{item.criterion_id}: required result is unknown."
            )
        elif item.status == "incomparable":
            tradeoffs.append(
                f"{item.criterion_id}: available evidence is condition/unit incomparable."
            )
        elif item.status == "conflicting":
            tradeoffs.append(
                f"{item.criterion_id}: named validators conflict; no average was used."
            )
    if candidate.model_disagreement == "high":
        why_not.append("HIGH_MODEL_DISAGREEMENT")
    if candidate.external_identity_status == "database_scoped_no_match":
        why.append("EXTERNAL_DATABASE_SCOPED_NO_MATCH_RETAINED_AS_CONTEXT_ONLY")
    return (
        list(dict.fromkeys(why or ["UNSCORED_CANDIDATE_RETAINED"])),
        list(dict.fromkeys(why_not)),
        list(dict.fromkeys(tradeoffs)),
    )


def _uncertainty_reasons(
    candidate: MaterialApplicationCandidate,
    criteria: Sequence[CandidateCriterionResult],
) -> list[str]:
    reasons: list[str] = []
    if any(item.status == "unknown" for item in criteria):
        reasons.append("One or more role criteria have no successful named validator.")
    if any(item.status == "incomparable" for item in criteria):
        reasons.append("Some evidence uses an incompatible validator, unit, or condition.")
    if any(item.status == "conflicting" for item in criteria):
        reasons.append("Named validators disagree and their outputs were not averaged.")
    if any(
        item.status == "available" and item.lower_bound is None
        for item in criteria
    ):
        reasons.append("At least one available property has no quantified uncertainty.")
    if candidate.model_disagreement == "high":
        reasons.append("Expert-model disagreement is high; retain an escalation branch.")
    if candidate.external_identity_status == "unknown":
        reasons.append("External identity/novelty lookup is unresolved.")
    if candidate.external_identity_status == "database_scoped_no_match":
        reasons.append(
            "Scoped database no-match is not proof of scientific novelty."
        )
    return list(dict.fromkeys(reasons))


def _next_validations(
    role: ApplicationRoleProfile,
    criteria: Sequence[CandidateCriterionResult],
    candidate: MaterialApplicationCandidate,
) -> list[str]:
    by_id = {item.criterion_id: item for item in role.criteria}
    steps: list[str] = []
    for result in criteria:
        if result.status != "available":
            criterion = by_id[result.criterion_id]
            steps.append(
                f"{result.criterion_id}: run one named validator "
                f"({', '.join(criterion.validator_ids)}) with unit {criterion.unit} "
                f"and conditions {', '.join(criterion.required_context) or 'declared scope'}."
            )
    if candidate.model_disagreement == "high":
        steps.append(
            "Resolve high model disagreement with a higher-fidelity calculation or "
            "matched experiment before promotion."
        )
    if candidate.origin == "retrieval_seed":
        steps.append(
            "Close this retrieval seed to exact literature/database records, then "
            "create a typed candidate and run role-specific validators."
        )
    return list(dict.fromkeys(steps))


def _resolve_candidate_citations(
    candidates: Sequence[MaterialApplicationCandidate],
    bundle: RagEvidenceBundle | None,
) -> dict[str, list[RecommendationCitation]]:
    claimed = {
        claim_id
        for candidate in candidates
        for claim_id in candidate.evidence_claim_ids
    }
    if claimed and bundle is None:
        raise ValueError("candidate evidence claim ids require a RAG evidence bundle")
    if bundle is None:
        return {item.candidate_id: [] for item in candidates}
    claims: dict[str, EvidenceClaim] = {
        item.claim_id: item for item in bundle.claims
    }
    records: dict[str, LiteratureRecord] = {
        item.record_id: item for item in bundle.records
    }
    if any(claim_id not in claims for claim_id in claimed):
        raise ValueError("candidate cites a claim outside the RAG bundle")
    output: dict[str, list[RecommendationCitation]] = {}
    for candidate in candidates:
        citations: list[RecommendationCitation] = []
        for claim_id in candidate.evidence_claim_ids:
            claim = claims[claim_id]
            record = records.get(claim.source_record_id)
            if record is None:
                raise ValueError("candidate claim cites an unknown literature record")
            citations.append(
                RecommendationCitation(
                    claim_id=claim.claim_id,
                    record_id=record.record_id,
                    title=record.title,
                    doi=record.doi,
                    urls=record.urls,
                    exact_support_span=claim.support_text,
                    polarity=str(claim.polarity),
                )
            )
        output[candidate.candidate_id] = citations
    return output


def _stable_json(value: object) -> str:
    import json

    return json.dumps(
        _canonicalize_json_numbers(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonicalize_json_numbers(value: object) -> object:
    """Make JSON-equivalent integer/float conditions compare identically.

    This is deliberately not a unit converter. It only removes the lexical
    distinction between an integral JSON float such as ``300.0`` and the same
    JSON number written as ``300``.
    """

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else value
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_json_numbers(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_json_numbers(item) for item in value]
    return value


__all__ = [
    "CandidateCriterionResult",
    "MaterialApplicationCandidate",
    "MaterialApplicationObservation",
    "MaterialDecisionPreference",
    "MaterialRecommendationCandidate",
    "MaterialRecommendationReport",
    "MaterialRoleRecommendation",
    "RecommendationCitation",
    "candidates_from_application_seeds",
    "rank_material_application_candidates",
]
