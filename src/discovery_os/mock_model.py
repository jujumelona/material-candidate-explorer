"""Deterministic discovery-model implementation for integration tests.

This model deliberately has no scientific inference capability.  It emits
small plans for the allow-listed dummy generator and basic registered
validators, labels every prediction as an uncalibrated ``model_prior``, and
uses conservative wording that cannot be mistaken for experimental evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from .schemas import (
    ApplicabilityAssessment,
    Candidate,
    CandidatePlan,
    CandidatePrediction,
    CandidateProposalRequest,
    CandidateRevision,
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    Fidelity,
    GenerationTask,
    GoalCompileRequest,
    Hypothesis,
    HypothesisBatch,
    HypothesisRequest,
    ObjectiveDirection,
    PredictionBatch,
    PredictionError,
    PredictionRequest,
    PropertyObjective,
    PropertyPrediction,
    ResultAnalysis,
    ResultAnalysisRequest,
    RevisionOperation,
    RevisionPlan,
    RevisionRequest,
    StopDecision,
    StopDecisionRequest,
    StopReason,
    SuccessCriterion,
    ToolCall,
    ToolDescriptor,
    ToolOperationDescriptor,
    UncertaintyKind,
    ValidationPlan,
    ValidationPlanningRequest,
)


@dataclass(frozen=True)
class _DomainDefaults:
    candidate_type: CandidateType
    objective: str
    title: str


_DOMAIN_DEFAULTS: dict[DiscoveryDomain, _DomainDefaults] = {
    DiscoveryDomain.MEDICINAL_CHEMISTRY: _DomainDefaults(
        CandidateType.SMALL_MOLECULE,
        "target_activity",
        "Mock medicinal-chemistry discovery goal",
    ),
    DiscoveryDomain.INORGANIC_MATERIALS: _DomainDefaults(
        CandidateType.COMPOSITION,
        "target_property",
        "Mock inorganic-material discovery goal",
    ),
    DiscoveryDomain.SUPERCONDUCTORS: _DomainDefaults(
        CandidateType.COMPOSITION,
        "critical_temperature",
        "Mock superconductor discovery goal",
    ),
    DiscoveryDomain.POLYMERS: _DomainDefaults(
        CandidateType.COMPOSITION,
        "target_property",
        "Mock polymer discovery goal",
    ),
    DiscoveryDomain.BATTERIES: _DomainDefaults(
        CandidateType.COMPOSITION,
        "capacity",
        "Mock battery-material discovery goal",
    ),
    DiscoveryDomain.CATALYSTS: _DomainDefaults(
        CandidateType.COMPOSITION,
        "activity",
        "Mock catalyst discovery goal",
    ),
    DiscoveryDomain.GENERAL_MATERIALS: _DomainDefaults(
        CandidateType.COMPOSITION,
        "target_property",
        "Mock general-material discovery goal",
    ),
}


_DOMAIN_KEYWORDS: tuple[tuple[DiscoveryDomain, tuple[str, ...]], ...] = (
    (DiscoveryDomain.SUPERCONDUCTORS, ("superconduct", "초전도")),
    (DiscoveryDomain.BATTERIES, ("battery", "batteries", "electrolyte", "배터리", "전지", "전해질")),
    (DiscoveryDomain.CATALYSTS, ("catalyst", "catalysis", "촉매")),
    (DiscoveryDomain.POLYMERS, ("polymer", "plastic", "고분자", "중합체", "플라스틱")),
    (DiscoveryDomain.MEDICINAL_CHEMISTRY, ("drug", "medicine", "medicinal", "therapeut", "의약", "약물", "신약", "치료제")),
    (DiscoveryDomain.INORGANIC_MATERIALS, ("inorganic", "crystal", "ceramic", "무기", "결정", "세라믹")),
)


class MockDiscoveryModel:
    """Safe, deterministic implementation of all eight model decisions."""

    def __init__(
        self,
        max_cycles: int = 2,
        *,
        stop_after_cycles: int | None = None,
        candidates_per_task: int = 2,
    ) -> None:
        configured_cycles = max_cycles if stop_after_cycles is None else stop_after_cycles
        self.max_cycles = _positive_int(configured_cycles, "max_cycles")
        self.stop_after_cycles = self.max_cycles
        self.candidates_per_task = _positive_int(
            candidates_per_task, "candidates_per_task"
        )

    def compile_goal(self, request: GoalCompileRequest) -> DiscoveryGoal:
        domain = _domain_from_request(request)
        defaults = _DOMAIN_DEFAULTS[domain]
        normalized_text = " ".join(request.user_text.split())
        goal_id = f"MOCK-GOAL-{_digest(domain.value, normalized_text, length=12)}"
        return DiscoveryGoal(
            goal_id=goal_id,
            domain=domain,
            title=defaults.title,
            scientific_question=normalized_text,
            objectives=[
                PropertyObjective(
                    property_name=defaults.objective,
                    direction=ObjectiveDirection.MAXIMIZE,
                    weight=1.0,
                    required=True,
                    rationale=(
                        "Placeholder objective extracted by MockDiscoveryModel; "
                        "a real model or a human must refine its scientific definition."
                    ),
                )
            ],
            constraints=[],
            success_criteria=[
                SuccessCriterion(
                    criterion_id="computational-screen",
                    description=(
                        "Registered computational sanity checks must complete; "
                        "this is not experimental validation."
                    ),
                    property_name=defaults.objective,
                    operator="exists",
                    evidence_kind=EvidenceKind.COMPUTATIONAL,
                    required=True,
                ),
                SuccessCriterion(
                    criterion_id="experimental-replication",
                    description=(
                        "The target property requires controlled experimental evidence "
                        "and independent replication before a discovery claim."
                    ),
                    property_name=defaults.objective,
                    operator="exists",
                    evidence_kind=EvidenceKind.EXPERIMENTAL,
                    required=True,
                ),
            ],
            validation_profile_id=(
                request.requested_validation_profile_id or f"{domain.value}-v1"
            ),
            candidate_types=[defaults.candidate_type],
            assumptions=[
                "Mock outputs are deterministic integration fixtures, not fitted scientific predictions.",
                "Computational checks cannot establish experimental observation or independent replication.",
            ],
            exclusions=[
                "Automatic laboratory execution",
                "Claims of medical efficacy, safety, superconductivity, or material performance",
            ],
            max_cycles=self.max_cycles,
        )

    def propose_hypotheses(self, request: HypothesisRequest) -> HypothesisBatch:
        objective_names = [item.property_name for item in request.goal.objectives]
        objective = objective_names[0] if objective_names else "target_property"
        hypothesis_id = (
            f"MOCK-H-{request.state.cycle:03d}-"
            f"{_digest(request.goal.goal_id, objective, length=8)}"
        )
        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            statement=(
                f"A generated candidate may warrant screening for {objective}; "
                "this is a placeholder hypothesis, not evidence of the property."
            ),
            mechanism=(
                "The mock model supplies only a deterministic integration hypothesis. "
                "A domain model and external validation must establish any mechanism."
            ),
            predicted_observations=[
                "A registered representation validator accepts at least one candidate.",
                f"A qualified external method measures or computes {objective} with uncertainty.",
            ],
            falsification_criteria=[
                "Registered sanity checks reject the candidate representation or composition.",
                "Qualified external evidence fails the declared acceptance criterion.",
            ],
            related_objectives=objective_names,
            confidence=0.1,
            assumptions=[
                "No learned scientific relationship is used by this mock hypothesis."
            ],
        )
        return HypothesisBatch(
            hypotheses=[hypothesis][: request.max_hypotheses],
            batch_reason=(
                "Deterministic pipeline fixture only; the hypothesis must not be "
                "reported as a scientific finding."
            ),
        )

    def propose_candidates(self, request: CandidateProposalRequest) -> CandidatePlan:
        candidate_type = _mock_candidate_type(request.goal)
        target_properties = {
            item.property_name: (
                item.target_value
                if item.target_value is not None
                else str(item.direction)
            )
            for item in request.goal.objectives
        }
        task = GenerationTask(
            task_id=(
                f"MOCK-GEN-{request.state.cycle:03d}-"
                f"{_digest(request.goal.goal_id, candidate_type.value, length=8)}"
            ),
            generator_name="dummy_generator",
            candidate_type=candidate_type,
            requested_count=self.candidates_per_task,
            hypothesis_ids=[item.hypothesis_id for item in request.hypotheses.hypotheses],
            target_properties=target_properties,
            diversity_strength=0.5,
            novelty_strength=0.0,
            conditions={
                "sample_family": _sample_family(candidate_type),
                "domain": str(request.goal.domain),
            },
            max_runtime_seconds=60,
            reason=(
                "Generate deterministic small-molecule or composition samples "
                "for orchestration tests; no novelty or performance is implied."
            ),
        )
        return CandidatePlan(
            tasks=[task],
            plan_reason=(
                "Use only the registered dummy_generator integration fixture. "
                "Generated samples are not discoveries."
            ),
        )

    def predict_candidates(self, request: PredictionRequest) -> PredictionBatch:
        properties = _prediction_properties(request)
        predictions: list[CandidatePrediction] = []
        for candidate in request.candidates:
            candidate_properties = [
                PropertyPrediction(
                    property_name=property_name,
                    value=_deterministic_prior(candidate.candidate_id, property_name),
                    uncertainty=0.5,
                    uncertainty_kind=UncertaintyKind.EPISTEMIC,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    confidence=0.1,
                    method="model_prior",
                    calibrated=False,
                    assumptions=[
                        "Uniform mock-scale prior; it is not trained on domain data."
                    ],
                    applicability_warnings=[
                        "Mock prior is outside any established applicability domain."
                    ],
                    applicability=ApplicabilityAssessment(
                        in_domain=False,
                        score=0.0,
                        domain_description="No applicability domain exists for MockDiscoveryModel.",
                        reasons=["No fitted predictor or calibration dataset is attached."],
                    ),
                )
                for property_name in properties
            ]
            predictions.append(
                CandidatePrediction(
                    candidate_id=candidate.candidate_id,
                    properties=candidate_properties,
                    overall_confidence=0.1,
                    out_of_distribution=True,
                    risks=[
                        "Values are deterministic model_prior fixtures, not measurements or validated predictions."
                    ],
                    recommended_validation_properties=properties,
                )
            )
        return PredictionBatch(
            predictions=predictions,
            batch_warnings=[
                "MockDiscoveryModel emitted uncalibrated model_prior values only; do not use them for scientific decisions."
            ],
        )

    def plan_validation(self, request: ValidationPlanningRequest) -> ValidationPlan:
        calls: list[ToolCall] = []
        information_gain: dict[str, float] = {}
        remaining_runtime = request.max_total_runtime_seconds
        specifications = (
            (
                "common_rules",
                "validate_candidate",
                ("representation_valid", "lineage_valid"),
                "Run the registered code-side representation and lineage sanity gate.",
                0.9,
            ),
            (
                "rdkit",
                "validate_molecule",
                ("validity", "canonical_smiles", "molecular_weight", "logp", "tpsa"),
                "Run registered low-cost molecular parsing and descriptor checks.",
                0.85,
            ),
            (
                "composition_rules",
                "validate_composition",
                ("formula_validity", "element_count", "molar_mass"),
                "Run registered low-cost composition syntax and element checks.",
                0.85,
            ),
        )

        for tool_name, operation_name, desired_properties, reason, gain in specifications:
            selected = _registered_operation(
                request.available_tools,
                tool_name,
                operation_name,
            )
            if selected is None:
                continue
            descriptor, operation = selected
            candidates = [
                candidate
                for candidate in request.candidates
                if _operation_supports_candidate(operation, candidate)
            ]
            if not candidates:
                continue
            conditions = _safe_default_conditions(operation)
            if conditions is None:
                continue
            if remaining_runtime is not None and remaining_runtime < 1:
                break
            max_runtime = min(operation.default_max_runtime_seconds, 60)
            if remaining_runtime is not None:
                max_runtime = min(max_runtime, remaining_runtime)
            produced = set(operation.produced_properties)
            requested_properties = [
                property_name
                for property_name in desired_properties
                if property_name in produced
            ]
            call_id = (
                f"MOCK-{tool_name}-{operation_name}-"
                f"{_digest(*(item.candidate_id for item in candidates), length=8)}"
            )
            calls.append(
                ToolCall(
                    call_id=call_id,
                    tool_name=tool_name,
                    operation=operation_name,
                    candidate_ids=[item.candidate_id for item in candidates],
                    requested_properties=requested_properties,
                    conditions=conditions,
                    evidence_kind=EvidenceKind.COMPUTATIONAL,
                    method_class=operation.method_class,
                    fidelity=Fidelity.CHEAP,
                    priority=gain,
                    reason=reason,
                    max_runtime_seconds=max_runtime,
                    resource_budget=descriptor.default_resource_budget,
                    retry_limit=0,
                    cache_allowed=True,
                )
            )
            information_gain[call_id] = gain
            if remaining_runtime is not None:
                remaining_runtime -= max_runtime

        if calls:
            plan_reason = (
                "Only registered common, RDKit, and composition sanity operations "
                "were selected. These checks provide computational evidence only."
            )
        else:
            plan_reason = (
                "No compatible registered basic validator was available; return an "
                "empty safe plan instead of inventing a tool or executable command."
            )
        return ValidationPlan(
            intents=[],
            calls=calls,
            expected_information_gain=information_gain,
            plan_reason=plan_reason,
        )

    def analyze_results(self, request: ResultAnalysisRequest) -> ResultAnalysis:
        records_by_candidate: dict[str, list[EvidenceRecord]] = {}
        for record in request.observed_evidence.records:
            records_by_candidate.setdefault(record.candidate_id, []).append(record)

        keep: list[str] = []
        remove: list[str] = []
        revise: list[str] = []
        confirmed: list[str] = []
        rejected: list[str] = []
        for candidate in request.candidates:
            records = records_by_candidate.get(candidate.candidate_id, [])
            if _explicitly_invalid(records):
                remove.append(candidate.candidate_id)
                rejected.append(
                    f"{candidate.candidate_id} failed a registered sanity criterion; "
                    "the mock model does not infer a scientific cause."
                )
            elif any(_record_succeeded(record) for record in records):
                keep.append(candidate.candidate_id)
                confirmed.append(
                    f"{candidate.candidate_id} completed at least one registered "
                    "computational check; this is not experimental validation."
                )
            elif records:
                revise.append(candidate.candidate_id)
            else:
                keep.append(candidate.candidate_id)

        if revise:
            next_action = "revise"
        elif request.candidates and len(remove) == len(request.candidates):
            next_action = "generate_new"
        else:
            next_action = "validate_more"

        return ResultAnalysis(
            confirmed_findings=confirmed,
            rejected_assumptions=rejected,
            prediction_errors=_prediction_errors(request),
            hypothesis_updates=[],
            newly_detected_patterns=[],
            candidates_to_keep=keep,
            candidates_to_remove=remove,
            candidates_to_revise=revise,
            next_recommended_action=next_action,
            rationale=(
                "This deterministic mock analysis only summarizes normalized tool "
                "status. It does not establish efficacy, safety, superconductivity, "
                "material performance, experimental observation, or replication."
            ),
        )

    def revise_candidates(self, request: RevisionRequest) -> RevisionPlan:
        evidence_by_candidate: dict[str, list[str]] = {}
        properties_by_candidate: dict[str, list[str]] = {}
        for record in request.evidence.records:
            evidence_by_candidate.setdefault(record.candidate_id, []).append(record.evidence_id)
            properties_by_candidate.setdefault(record.candidate_id, []).extend(
                result.property_name for result in record.properties
            )

        known_ids = {candidate.candidate_id for candidate in request.candidates}
        revisions: list[CandidateRevision] = []
        retain = [
            candidate_id
            for candidate_id in request.analysis.candidates_to_keep
            if candidate_id in known_ids
        ]
        retire = [
            candidate_id
            for candidate_id in request.analysis.candidates_to_remove
            if candidate_id in known_ids
        ]
        for candidate_id in request.analysis.candidates_to_revise:
            if candidate_id not in known_ids:
                continue
            evidence_ids = _ordered_unique(evidence_by_candidate.get(candidate_id, []))
            if not evidence_ids:
                retain.append(candidate_id)
                continue
            revisions.append(
                CandidateRevision(
                    revision_id=(
                        f"MOCK-REV-{_digest(request.state.run_id, candidate_id, length=10)}"
                    ),
                    candidate_id=candidate_id,
                    operation=RevisionOperation.REGENERATE,
                    feature="candidate representation",
                    proposed_change={
                        "strategy": "request_a_new_dummy_sample",
                        "executable_code": False,
                    },
                    preserve_features=[],
                    expected_effects={
                        "scope": "integration test only",
                        "scientific_improvement_claimed": False,
                    },
                    required_followup_properties=_ordered_unique(
                        properties_by_candidate.get(candidate_id, [])
                    ),
                    based_on_evidence_ids=evidence_ids,
                    reason=(
                        "A non-success tool status permits a deterministic regeneration "
                        "proposal only; improvement is not predicted."
                    ),
                )
            )

        retain = [item for item in _ordered_unique(retain) if item not in set(retire)]
        return RevisionPlan(
            revisions=revisions,
            generation_tasks=[],
            candidates_to_retain=retain,
            candidates_to_retire=_ordered_unique(retire),
            plan_reason=(
                "Revision entries are structured mock proposals. They contain no code "
                "and require the normal generator and validation runtimes."
            ),
        )

    def decide_stop(self, request: StopDecisionRequest) -> StopDecision:
        configured_limit = min(self.stop_after_cycles, request.goal.max_cycles)
        # ``DiscoveryState.cycle`` and the engine's request value are completed
        # cycle counts (the first completed cycle is 1), not zero-based indexes.
        completed_cycles = max(request.cycle, request.state.cycle)
        if request.history_summary is not None:
            completed_cycles = max(
                completed_cycles,
                request.history_summary.completed_cycles,
            )
        reached_limit = completed_cycles >= configured_limit
        best_candidate_ids = _best_candidate_ids(request)
        unmet = [item.description for item in request.goal.success_criteria if item.required]

        if reached_limit:
            return StopDecision(
                stop=True,
                reason_code=StopReason.MAX_CYCLES_REACHED,
                reason=(
                    "Configured mock cycle limit reached. Stopping the integration "
                    "loop does not imply that any scientific success criterion was met."
                ),
                unmet_criteria=unmet,
                best_candidate_ids=best_candidate_ids,
                recommended_next_action="human_review",
            )

        next_action = "validate_more"
        if request.latest_analysis is not None:
            proposed = request.latest_analysis.next_recommended_action
            if proposed in {"validate_more", "revise", "generate_new"}:
                next_action = proposed
        return StopDecision(
            stop=False,
            reason_code=StopReason.CONTINUE,
            reason=(
                f"Mock cycle {completed_cycles} of {configured_limit} completed; "
                "continue only through registered runtimes and evidence gates."
            ),
            unmet_criteria=unmet,
            best_candidate_ids=best_candidate_ids,
            recommended_next_action=next_action,
        )


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _domain_from_request(request: GoalCompileRequest) -> DiscoveryDomain:
    if request.domain_hint is not None:
        return DiscoveryDomain(str(request.domain_hint))
    text = request.user_text.casefold()
    for domain, keywords in _DOMAIN_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return domain
    return DiscoveryDomain.GENERAL_MATERIALS


def _mock_candidate_type(goal: DiscoveryGoal) -> CandidateType:
    available = {str(item) for item in goal.candidate_types}
    if CandidateType.SMALL_MOLECULE.value in available:
        return CandidateType.SMALL_MOLECULE
    return CandidateType.COMPOSITION


def _sample_family(candidate_type: CandidateType) -> str:
    if candidate_type == CandidateType.SMALL_MOLECULE:
        return "small_molecule_samples"
    return "composition_samples"


def _prediction_properties(request: PredictionRequest) -> list[str]:
    requested = [item.strip() for item in request.requested_properties if item.strip()]
    if not requested:
        requested = [item.property_name for item in request.goal.objectives]
    return _ordered_unique(requested)


def _deterministic_prior(candidate_id: str, property_name: str) -> float:
    raw = int(_digest(candidate_id, property_name, length=8), 16)
    return round(0.2 + (raw / 0xFFFFFFFF) * 0.6, 6)


def _registered_operation(
    descriptors: list[ToolDescriptor],
    tool_name: str,
    operation_name: str,
) -> tuple[ToolDescriptor, ToolOperationDescriptor] | None:
    for descriptor in descriptors:
        if descriptor.tool_name != tool_name or not descriptor.available:
            continue
        for operation in descriptor.operations:
            if operation.operation != operation_name or operation.requires_human_approval:
                continue
            if EvidenceKind.COMPUTATIONAL not in operation.evidence_kinds:
                continue
            if Fidelity.CHEAP not in operation.supported_fidelities:
                continue
            return descriptor, operation
    return None


def _operation_supports_candidate(
    operation: ToolOperationDescriptor,
    candidate: Candidate,
) -> bool:
    return (
        candidate.domain in operation.supported_domains
        and candidate.candidate_type in operation.supported_candidate_types
    )


def _safe_default_conditions(
    operation: ToolOperationDescriptor,
) -> dict[str, object] | None:
    conditions: dict[str, object] = {}
    for parameter in operation.condition_parameters:
        if parameter.required and parameter.default is None:
            return None
        if parameter.required:
            conditions[parameter.name] = parameter.default
    return conditions


def _record_succeeded(record: EvidenceRecord) -> bool:
    return record.status in {EvidenceStatus.SUCCESS, EvidenceStatus.PARTIAL}


def _explicitly_invalid(records: list[EvidenceRecord]) -> bool:
    validity_names = {
        "validity",
        "formula_validity",
        "representation_valid",
        "composition_validity",
        "structure_validity",
    }
    for record in records:
        if record.status == EvidenceStatus.FAILED and record.operation in {
            "validate_candidate",
            "validate_molecule",
            "validate_composition",
        }:
            return True
        for result in record.properties:
            if result.property_name.casefold() not in validity_names:
                continue
            if result.meets_criterion is False or result.value is False:
                return True
    return False


def _prediction_errors(request: ResultAnalysisRequest) -> list[PredictionError]:
    predictions = {
        (prediction.candidate_id, prop.property_name): prop
        for prediction in request.predictions_before_validation
        for prop in prediction.properties
    }
    errors: list[PredictionError] = []
    for record in request.observed_evidence.records:
        for observed in record.properties:
            predicted = predictions.get((record.candidate_id, observed.property_name))
            if predicted is None:
                continue
            predicted_value = _number(predicted.value)
            observed_value = _number(observed.value)
            if predicted_value is None or observed_value is None:
                continue
            denominator = max(abs(observed_value), 1.0e-12)
            errors.append(
                PredictionError(
                    candidate_id=record.candidate_id,
                    property_name=observed.property_name,
                    predicted_value=predicted_value,
                    observed_value=observed_value,
                    normalized_error=abs(predicted_value - observed_value) / denominator,
                    likely_causes=[
                        "The mock model_prior is untrained and uncalibrated; disagreement has no learned diagnostic interpretation."
                    ],
                )
            )
    return errors


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _best_candidate_ids(request: StopDecisionRequest) -> list[str]:
    result: list[str] = []
    for assessment in request.validation_assessments:
        if str(assessment.status) != "rejected":
            result.append(assessment.candidate_id)
    if request.history_summary is not None:
        result.extend(request.history_summary.best_candidate_ids)
    return _ordered_unique(result)


def _ordered_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _digest(*parts: str, length: int) -> str:
    material = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()[:length].upper()


__all__ = ["MockDiscoveryModel"]
