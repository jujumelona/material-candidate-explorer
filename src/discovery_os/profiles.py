"""Deterministic validation profiles and conservative evidence gate evaluation.

The discovery model may propose validation work, but it cannot declare a
candidate validated.  This module makes that decision from normalized evidence
using code-owned, versioned profiles.  In particular, computational evidence
can never satisfy an experimental requirement and missing provenance can never
be treated as independent replication.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from collections.abc import Callable
from ._compat import StrEnum

from pydantic import Field, model_validator

from .schemas import (
    CandidateValidationAssessment,
    CandidateValidationStatus,
    ClaimLevel,
    DiscoveryDomain,
    EvidenceBatch,
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    Fidelity,
    GateDecision,
    Identifier,
    MethodClass,
    NonEmptyText,
    PositiveInt,
    RequirementDecision,
    StrictSchema,
    VerificationStatus,
)


class PropertyMatch(StrEnum):
    ANY = "any"
    ALL = "all"


class EvidenceRequirement(StrictSchema):
    requirement_id: Identifier
    description: NonEmptyText
    evidence_kind: EvidenceKind
    property_names: list[Identifier] = Field(default_factory=list)
    property_match: PropertyMatch = PropertyMatch.ANY
    minimum_fidelity: Fidelity
    allowed_tools: list[Identifier] = Field(default_factory=list)
    allowed_operations: list[Identifier] = Field(default_factory=list)
    allowed_method_classes: list[MethodClass] = Field(default_factory=list)
    minimum_records: PositiveInt = 1
    minimum_independent_sources: PositiveInt = 1
    require_meets_criterion: bool = True
    acceptable_statuses: list[EvidenceStatus] = Field(default_factory=lambda: [EvidenceStatus.SUCCESS])

    @model_validator(mode="after")
    def _validate_requirement(self) -> EvidenceRequirement:
        if self.evidence_kind == EvidenceKind.EXPERIMENTAL and self.minimum_fidelity != Fidelity.EXPERIMENTAL:
            raise ValueError("experimental requirements must use experimental fidelity")
        if self.evidence_kind == EvidenceKind.COMPUTATIONAL and self.minimum_fidelity == Fidelity.EXPERIMENTAL:
            raise ValueError("computational requirements cannot use experimental fidelity")
        return self


class ValidationGate(StrictSchema):
    gate_id: Identifier
    name: NonEmptyText
    description: NonEmptyText
    claim_level: ClaimLevel
    requirements: list[EvidenceRequirement] = Field(min_length=1)
    mandatory: bool = True
    reject_on_failure: bool = False

    @model_validator(mode="after")
    def _unique_requirements(self) -> ValidationGate:
        ids = [item.requirement_id for item in self.requirements]
        if len(ids) != len(set(ids)):
            raise ValueError("requirement_id values must be unique within a gate")
        if self.claim_level == ClaimLevel.GENERATED:
            raise ValueError("generated is an initial state, not an evidence gate")
        return self


class ValidationProfile(StrictSchema):
    profile_id: Identifier
    profile_version: Identifier
    domain: DiscoveryDomain
    name: NonEmptyText
    description: NonEmptyText
    gates: list[ValidationGate] = Field(min_length=1)
    final_claim_level: ClaimLevel = ClaimLevel.INDEPENDENTLY_REPLICATED

    @model_validator(mode="after")
    def _validate_profile(self) -> ValidationProfile:
        ids = [item.gate_id for item in self.gates]
        if len(ids) != len(set(ids)):
            raise ValueError("gate_id values must be unique within a profile")
        available_levels = {item.claim_level for item in self.gates if item.mandatory}
        if self.final_claim_level not in available_levels:
            raise ValueError("final_claim_level must be represented by a mandatory gate")
        return self


class ValidationGateEvaluator:
    """Deterministic evaluator for a fixed profile, evidence set, and trust verifier."""

    def __init__(
        self,
        *,
        experimental_record_verifier: Callable[[EvidenceRecord], bool] | None = None,
    ) -> None:
        self.experimental_record_verifier = experimental_record_verifier

    def evaluate(
        self,
        profile: ValidationProfile,
        candidate_id: str,
        evidence: EvidenceBatch | Sequence[EvidenceRecord],
    ) -> CandidateValidationAssessment:
        records = list(evidence.records if isinstance(evidence, EvidenceBatch) else evidence)
        candidate_records = [record for record in records if record.candidate_id == candidate_id]

        gate_decisions = [self.evaluate_gate(gate, candidate_records) for gate in profile.gates]
        decision_by_id = {decision.gate_id: decision for decision in gate_decisions}

        attained_level = ClaimLevel.GENERATED
        for level in (
            ClaimLevel.COMPUTATIONALLY_PLAUSIBLE,
            ClaimLevel.EXPERIMENTALLY_OBSERVED,
            ClaimLevel.INDEPENDENTLY_REPLICATED,
        ):
            prerequisite_gates = [
                gate
                for gate in profile.gates
                if gate.mandatory and _LEVEL_RANK[gate.claim_level] <= _LEVEL_RANK[level]
            ]
            if prerequisite_gates and all(decision_by_id[gate.gate_id].passed for gate in prerequisite_gates):
                attained_level = level
            else:
                break

        all_mandatory_passed = all(
            decision_by_id[gate.gate_id].passed for gate in profile.gates if gate.mandatory
        )
        rejecting_failure = any(
            gate.reject_on_failure and decision_by_id[gate.gate_id].status == "failed"
            for gate in profile.gates
            if gate.mandatory
        )

        matched_ids = _ordered_unique(
            evidence_id
            for decision in gate_decisions
            for evidence_id in decision.matched_evidence_ids
        )
        matched_records = [record for record in candidate_records if record.evidence_id in set(matched_ids)]
        matched_kinds = _ordered_unique(record.evidence_kind for record in matched_records)

        caveats: list[str] = []
        if rejecting_failure:
            status = CandidateValidationStatus.REJECTED
            caveats.append("A rejection-critical validation gate has explicit contrary evidence.")
        elif all_mandatory_passed and attained_level == profile.final_claim_level:
            if attained_level == ClaimLevel.COMPUTATIONALLY_PLAUSIBLE:
                status = CandidateValidationStatus.COMPUTATIONALLY_SUPPORTED
            else:
                # The schema performs an additional independent check that
                # experimental evidence is cited for experimental final levels.
                status = CandidateValidationStatus.EXPERIMENTALLY_VALIDATED
        elif attained_level == ClaimLevel.COMPUTATIONALLY_PLAUSIBLE:
            status = CandidateValidationStatus.COMPUTATIONALLY_SUPPORTED
            caveats.append("Computational support is not experimental validation.")
        elif candidate_records:
            status = CandidateValidationStatus.INCONCLUSIVE
            caveats.append("Required validation gates remain failed or unsupported.")
        else:
            status = CandidateValidationStatus.UNVALIDATED
            caveats.append("No candidate-specific evidence was supplied.")

        for gate, decision in zip(profile.gates, gate_decisions, strict=True):
            if gate.mandatory and not decision.passed:
                caveats.append(f"Unmet gate {gate.gate_id}: {decision.reason}")

        return CandidateValidationAssessment(
            candidate_id=candidate_id,
            profile_id=profile.profile_id,
            status=status,
            claim_level=attained_level,
            gate_decisions=gate_decisions,
            matched_evidence_ids=matched_ids,
            matched_evidence_kinds=matched_kinds,
            caveats=caveats,
        )

    def evaluate_gate(
        self,
        gate: ValidationGate,
        evidence: EvidenceBatch | Sequence[EvidenceRecord],
    ) -> GateDecision:
        records = list(evidence.records if isinstance(evidence, EvidenceBatch) else evidence)
        requirement_decisions = [self.evaluate_requirement(requirement, records) for requirement in gate.requirements]
        passed = all(decision.satisfied for decision in requirement_decisions)

        if passed:
            status = "passed"
            reason = "Every required evidence condition is satisfied."
        elif any("conflict" in decision.reason.casefold() for decision in requirement_decisions):
            status = "insufficient_evidence"
            reason = "Contradictory qualifying evidence remains unresolved."
        elif any(decision.status == "failed" for decision in requirement_decisions):
            status = "failed"
            reason = "At least one requirement has explicit contrary evidence."
        else:
            status = "insufficient_evidence"
            reason = "One or more required evidence conditions are not documented."

        matched_ids = _ordered_unique(
            evidence_id
            for decision in requirement_decisions
            for evidence_id in decision.matched_evidence_ids
        )
        return GateDecision(
            gate_id=gate.gate_id,
            passed=passed,
            status=status,
            requirement_decisions=requirement_decisions,
            matched_evidence_ids=matched_ids,
            reason=reason,
        )

    def evaluate_requirement(
        self,
        requirement: EvidenceRequirement,
        records: Sequence[EvidenceRecord],
    ) -> RequirementDecision:
        compatible = [
            record
            for record in records
            if _record_is_compatible(
                record,
                requirement,
                self.experimental_record_verifier,
            )
        ]
        property_names = {name.casefold() for name in requirement.property_names}

        observed: dict[str, list[tuple[EvidenceRecord, object]]] = {name: [] for name in property_names}
        for record in compatible:
            for result in record.properties:
                normalized_name = result.property_name.casefold()
                if normalized_name in observed:
                    observed[normalized_name].append((record, result))

        if property_names:
            if requirement.property_match == PropertyMatch.ALL:
                present = all(observed[name] for name in property_names)
            else:
                present = any(observed[name] for name in property_names)
        else:
            present = bool(compatible)

        explicit_negative = any(
            result.meets_criterion is False
            for results in observed.values()
            for _, result in results
        )
        if requirement.require_meets_criterion and property_names:
            property_passes = {
                name: any(result.meets_criterion is True for _, result in observed[name])
                for name in property_names
            }
            if requirement.property_match == PropertyMatch.ALL:
                criterion_met = all(property_passes.values())
            else:
                criterion_met = any(property_passes.values())
        else:
            criterion_met = present

        reported_records = _records_contributing_to_properties(compatible, property_names)
        if not property_names:
            reported_records = compatible

        if requirement.require_meets_criterion and property_names:
            positive_records = [
                record
                for record in reported_records
                if any(
                    result.property_name.casefold() in property_names and result.meets_criterion is True
                    for result in record.properties
                )
            ]
        else:
            positive_records = reported_records
        record_count_ok = len({record.evidence_id for record in positive_records}) >= requirement.minimum_records

        source_properties: dict[str, set[str]] = {}
        source_records: dict[str, list[EvidenceRecord]] = {}
        for record in positive_records:
            if requirement.minimum_independent_sources > 1:
                # No fallback is allowed when a profile asks for independence.
                source_key = record.source_id
            else:
                source_key = record.source_id or f"{record.tool_name}:{record.tool_version}"
            if source_key is None:
                continue
            source_records.setdefault(source_key, []).append(record)
            if property_names:
                source_properties.setdefault(source_key, set()).update(
                    result.property_name.casefold()
                    for result in record.properties
                    if result.property_name.casefold() in property_names
                    and (not requirement.require_meets_criterion or result.meets_criterion is True)
                )

        # Never invent independence from tool name or record count when a
        # profile asks for multiple sources: source_key above is then present
        # only for records carrying an explicit source_id.
        if property_names:
            if requirement.property_match == PropertyMatch.ALL:
                independent_sources = {
                    source for source, names in source_properties.items() if property_names.issubset(names)
                }
            else:
                independent_sources = {source for source, names in source_properties.items() if names}
        else:
            independent_sources = set(source_records)
        source_count_ok = len(independent_sources) >= requirement.minimum_independent_sources

        conflict = criterion_met and explicit_negative
        satisfied = (
            present
            and criterion_met
            and record_count_ok
            and source_count_ok
            and not conflict
        )
        cited_records = positive_records if satisfied else reported_records
        matched_ids = _ordered_unique(record.evidence_id for record in cited_records)
        missing: list[str] = []
        if not present:
            missing.append("required property evidence")
        elif not criterion_met:
            missing.append("an explicit meets_criterion=true result")
        if not record_count_ok:
            missing.append(f"at least {requirement.minimum_records} qualifying record(s)")
        if not source_count_ok:
            missing.append(
                f"at least {requirement.minimum_independent_sources} explicitly identified independent source(s)"
            )
        if conflict:
            missing.append("resolution of contradictory qualifying evidence")

        if satisfied:
            status = "passed"
            reason = "The normalized evidence satisfies this requirement."
        elif conflict:
            status = "insufficient_evidence"
            reason = (
                "Qualifying positive and negative results conflict; the conflict must be "
                "resolved before this requirement can pass."
            )
        elif explicit_negative:
            status = "failed"
            reason = "A qualifying result explicitly reports that the criterion is not met."
        else:
            status = "insufficient_evidence"
            reason = "Missing: " + "; ".join(missing or ["qualifying evidence"])

        return RequirementDecision(
            requirement_id=requirement.requirement_id,
            satisfied=satisfied,
            status=status,
            matched_evidence_ids=matched_ids,
            missing=missing,
            reason=reason,
        )


# Short public spelling used by orchestration code.
GateEvaluator = ValidationGateEvaluator


def get_validation_profile(domain_or_profile: DiscoveryDomain | str) -> ValidationProfile:
    """Return a defensive copy of a built-in profile by domain or profile id."""

    key = str(domain_or_profile)
    profile = _PROFILES_BY_ID.get(key)
    if profile is None:
        try:
            domain = DiscoveryDomain(key)
        except ValueError as exc:
            known = ", ".join(sorted(_PROFILES_BY_ID))
            raise KeyError(f"unknown validation domain/profile {key!r}; known profiles: {known}") from exc
        profile = VALIDATION_PROFILES[domain]
    return profile.model_copy(deep=True)


def evaluate_candidate(
    profile: ValidationProfile,
    candidate_id: str,
    evidence: EvidenceBatch | Sequence[EvidenceRecord],
    *,
    experimental_record_verifier: Callable[[EvidenceRecord], bool] | None = None,
) -> CandidateValidationAssessment:
    """Functional wrapper around :class:`ValidationGateEvaluator`."""

    return ValidationGateEvaluator(
        experimental_record_verifier=experimental_record_verifier
    ).evaluate(profile, candidate_id, evidence)


def _record_is_compatible(
    record: EvidenceRecord,
    requirement: EvidenceRequirement,
    experimental_record_verifier: Callable[[EvidenceRecord], bool] | None = None,
) -> bool:
    if record.candidate_ref is None:
        return False
    if record.evidence_kind != requirement.evidence_kind:
        return False
    if (
        requirement.evidence_kind == EvidenceKind.EXPERIMENTAL
        and record.verification.status != VerificationStatus.VERIFIED
    ):
        return False
    if requirement.evidence_kind == EvidenceKind.EXPERIMENTAL:
        if experimental_record_verifier is None:
            return False
        try:
            if not experimental_record_verifier(record):
                return False
        except Exception:
            return False
    if record.status not in requirement.acceptable_statuses:
        return False
    if requirement.allowed_tools and record.tool_name not in requirement.allowed_tools:
        return False
    if requirement.allowed_operations and record.operation not in requirement.allowed_operations:
        return False
    if requirement.allowed_method_classes and record.method_class not in requirement.allowed_method_classes:
        return False
    if requirement.evidence_kind == EvidenceKind.EXPERIMENTAL:
        return record.fidelity == Fidelity.EXPERIMENTAL
    details = record.computational_details
    if details is None:
        return False
    if record.fidelity in {Fidelity.MEDIUM, Fidelity.HIGH}:
        if not record.parameters_hash:
            return False
        if not record.convergence_checks or not all(record.convergence_checks.values()):
            return False
        if not (record.container_digest or details.code_revision):
            return False
    if record.method_class == MethodClass.MACHINE_LEARNING:
        if (
            not details.model_name
            or not details.model_version
            or not details.dataset_versions
            or details.applicability is None
            or not details.applicability.in_domain
            or not details.calibration_method
        ):
            return False
    disqualifying_flags = {
        "failed_convergence",
        "out_of_domain",
        "invalid_output",
        "unresolved_conflict",
    }
    if any(flag.casefold() in disqualifying_flags for flag in details.quality_flags):
        return False
    requested_properties = {name.casefold() for name in requirement.property_names}
    relevant_results = [
        result
        for result in record.properties
        if not requested_properties
        or result.property_name.casefold() in requested_properties
    ]
    for result in relevant_results:
        if result.applicability is not None and not result.applicability.in_domain:
            return False
        if any(flag.casefold() in disqualifying_flags for flag in result.quality_flags):
            return False
        if (
            record.fidelity in {Fidelity.MEDIUM, Fidelity.HIGH}
            and isinstance(result.value, (int, float))
            and not isinstance(result.value, bool)
            and result.uncertainty is None
            and not (result.lower_bound is not None and result.upper_bound is not None)
        ):
            return False
        if record.method_class == MethodClass.MACHINE_LEARNING:
            if (
                result.applicability is None
                or not result.applicability.in_domain
                or not result.calibration_method
            ):
                return False
    return _FIDELITY_RANK[record.fidelity] >= _FIDELITY_RANK[requirement.minimum_fidelity]


def _records_contributing_to_properties(
    records: Sequence[EvidenceRecord],
    property_names: set[str],
) -> list[EvidenceRecord]:
    if not property_names:
        return list(records)
    return [
        record
        for record in records
        if any(result.property_name.casefold() in property_names for result in record.properties)
    ]


def _ordered_unique(values: Iterable[object]) -> list:
    result: list = []
    seen: set[object] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _requirement(
    requirement_id: str,
    description: str,
    evidence_kind: EvidenceKind,
    properties: tuple[str, ...],
    *,
    fidelity: Fidelity,
    match: PropertyMatch = PropertyMatch.ANY,
    records: int = 1,
    sources: int = 1,
    methods: tuple[MethodClass, ...] = (),
) -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=requirement_id,
        description=description,
        evidence_kind=evidence_kind,
        property_names=list(properties),
        property_match=match,
        minimum_fidelity=fidelity,
        minimum_records=records,
        minimum_independent_sources=sources,
        require_meets_criterion=True,
        allowed_method_classes=list(methods),
    )


def _gate(
    gate_id: str,
    name: str,
    level: ClaimLevel,
    requirements: tuple[EvidenceRequirement, ...],
    *,
    reject: bool = False,
) -> ValidationGate:
    return ValidationGate(
        gate_id=gate_id,
        name=name,
        description=f"Code-owned {name.lower()} gate.",
        claim_level=level,
        requirements=list(requirements),
        reject_on_failure=reject,
    )


def _profile(
    domain: DiscoveryDomain,
    name: str,
    gates: tuple[ValidationGate, ...],
) -> ValidationProfile:
    return ValidationProfile(
        profile_id=f"{domain.value}-v1",
        profile_version="1.0",
        domain=domain,
        name=name,
        description=(
            "A conservative staged profile. Computational plausibility is reported separately "
            "from experimental observation, and final validation requires independent replication."
        ),
        gates=list(gates),
    )


_COMPUTATIONAL = EvidenceKind.COMPUTATIONAL
_EXPERIMENTAL = EvidenceKind.EXPERIMENTAL
_CHEAP = Fidelity.CHEAP
_MEDIUM = Fidelity.MEDIUM
_HIGH = Fidelity.HIGH
_LAB = Fidelity.EXPERIMENTAL
_CP = ClaimLevel.COMPUTATIONALLY_PLAUSIBLE
_EO = ClaimLevel.EXPERIMENTALLY_OBSERVED
_IR = ClaimLevel.INDEPENDENTLY_REPLICATED


MEDICINAL_CHEMISTRY_PROFILE = _profile(
    DiscoveryDomain.MEDICINAL_CHEMISTRY,
    "Medicinal chemistry validation",
    (
        _gate("med-structure", "Chemical structure validity", _CP, (
            _requirement("med-validity", "A chemically valid, sanitizable structure.", _COMPUTATIONAL, ("validity", "chemical_validity"), fidelity=_CHEAP),
            _requirement("med-synthesis", "A plausible synthesis route or accessibility assessment.", _COMPUTATIONAL, ("synthetic_accessibility", "synthesis_feasibility"), fidelity=_CHEAP),
        ), reject=True),
        _gate("med-in-silico", "Activity, selectivity, and safety prediction", _CP, (
            _requirement("med-activity-pred", "Predicted activity for the declared target.", _COMPUTATIONAL, ("target_activity", "binding_affinity", "potency"), fidelity=_MEDIUM),
            _requirement("med-selectivity-pred", "Predicted selectivity or off-target assessment.", _COMPUTATIONAL, ("selectivity", "off_target_risk"), fidelity=_MEDIUM),
            _requirement("med-admet-pred", "ADMET and toxicity risk assessment.", _COMPUTATIONAL, ("admet", "toxicity", "safety_margin"), fidelity=_MEDIUM),
        )),
        _gate("med-experimental", "Experimental identity and biological activity", _EO, (
            _requirement("med-identity", "Measured identity and purity of a physical sample.", _EXPERIMENTAL, ("identity", "purity"), fidelity=_LAB, match=PropertyMatch.ALL, methods=(MethodClass.ANALYTICAL_MEASUREMENT,)),
            _requirement("med-activity-exp", "Controlled experimental target activity.", _EXPERIMENTAL, ("target_activity", "potency", "binding_affinity"), fidelity=_LAB, methods=(MethodClass.BIOASSAY,)),
            _requirement("med-safety-exp", "Relevant experimental toxicity or safety result.", _EXPERIMENTAL, ("toxicity", "cell_viability", "safety_margin"), fidelity=_LAB, methods=(MethodClass.BIOASSAY,)),
        )),
        _gate("med-replication", "Independent biological replication", _IR, (
            _requirement("med-activity-replicated", "Target activity reproduced by independent sources.", _EXPERIMENTAL, ("target_activity", "potency", "binding_affinity"), fidelity=_LAB, records=2, sources=2, methods=(MethodClass.BIOASSAY,)),
        )),
    ),
)


INORGANIC_MATERIALS_PROFILE = _profile(
    DiscoveryDomain.INORGANIC_MATERIALS,
    "Inorganic materials validation",
    (
        _gate("inorganic-validity", "Composition and structure validity", _CP, (
            _requirement("inorganic-composition", "Charge-balanced, chemically valid composition.", _COMPUTATIONAL, ("composition_validity", "charge_balance"), fidelity=_CHEAP, match=PropertyMatch.ALL),
            _requirement("inorganic-stability", "Thermodynamic stability against competing phases.", _COMPUTATIONAL, ("energy_above_hull", "thermodynamic_stability"), fidelity=_HIGH),
            _requirement("inorganic-dynamics", "No disqualifying dynamic instability.", _COMPUTATIONAL, ("phonon_stability", "dynamic_stability"), fidelity=_HIGH),
        ), reject=True),
        _gate("inorganic-property", "High-fidelity target-property prediction", _CP, (
            _requirement("inorganic-target-pred", "High-fidelity predicted target property.", _COMPUTATIONAL, ("target_property",), fidelity=_HIGH),
        )),
        _gate("inorganic-experimental", "Synthesized phase and measured property", _EO, (
            _requirement("inorganic-phase", "Experimental phase identity and composition.", _EXPERIMENTAL, ("phase_identity", "composition"), fidelity=_LAB, match=PropertyMatch.ALL, methods=(MethodClass.MATERIALS_CHARACTERIZATION, MethodClass.ANALYTICAL_MEASUREMENT)),
            _requirement("inorganic-target-exp", "Target property measured under documented conditions.", _EXPERIMENTAL, ("target_property",), fidelity=_LAB),
        )),
        _gate("inorganic-replication", "Independent property replication", _IR, (
            _requirement("inorganic-target-replicated", "Target property independently reproduced.", _EXPERIMENTAL, ("target_property",), fidelity=_LAB, records=2, sources=2),
        )),
    ),
)


SUPERCONDUCTOR_PROFILE = _profile(
    DiscoveryDomain.SUPERCONDUCTORS,
    "Superconductor validation",
    (
        _gate("sc-stability", "Structure and stability plausibility", _CP, (
            _requirement("sc-structure-valid", "Valid composition and crystal structure.", _COMPUTATIONAL, ("structure_validity", "composition_validity"), fidelity=_CHEAP),
            _requirement("sc-thermo-stable", "Thermodynamic stability against competing phases.", _COMPUTATIONAL, ("energy_above_hull", "thermodynamic_stability"), fidelity=_HIGH),
            _requirement("sc-phonon-stable", "Dynamical or phonon stability assessment.", _COMPUTATIONAL, ("phonon_stability", "dynamic_stability"), fidelity=_HIGH),
        ), reject=True),
        _gate("sc-prediction", "Superconductivity prediction", _CP, (
            _requirement("sc-electronic", "Relevant electronic structure support.", _COMPUTATIONAL, ("electronic_structure", "density_of_states", "fermi_surface"), fidelity=_HIGH),
            _requirement("sc-tc-pred", "A documented superconducting transition prediction.", _COMPUTATIONAL, ("critical_temperature", "tc", "electron_phonon_coupling"), fidelity=_HIGH),
        )),
        _gate("sc-phase", "Synthesized phase characterization", _EO, (
            _requirement("sc-phase-identity", "Physical sample phase and composition confirmed.", _EXPERIMENTAL, ("phase_identity", "composition"), fidelity=_LAB, match=PropertyMatch.ALL, methods=(MethodClass.MATERIALS_CHARACTERIZATION,)),
        )),
        _gate("sc-signatures", "Transport and magnetic superconducting signatures", _EO, (
            _requirement("sc-zero-resistance", "Zero-resistance transition under documented conditions.", _EXPERIMENTAL, ("zero_resistance", "resistive_transition"), fidelity=_LAB, methods=(MethodClass.ANALYTICAL_MEASUREMENT, MethodClass.MATERIALS_CHARACTERIZATION)),
            _requirement("sc-meissner", "Meissner effect or matching diamagnetic response.", _EXPERIMENTAL, ("meissner_effect", "diamagnetic_response", "magnetic_susceptibility"), fidelity=_LAB, methods=(MethodClass.ANALYTICAL_MEASUREMENT, MethodClass.MATERIALS_CHARACTERIZATION)),
        ), reject=True),
        _gate("sc-replication", "Independent superconductivity replication", _IR, (
            _requirement("sc-signatures-replicated", "Both transport and magnetic signatures independently reproduced.", _EXPERIMENTAL, ("zero_resistance", "meissner_effect"), fidelity=_LAB, match=PropertyMatch.ALL, records=2, sources=2),
        )),
    ),
)


POLYMER_PROFILE = _profile(
    DiscoveryDomain.POLYMERS,
    "Polymer validation",
    (
        _gate("polymer-plausibility", "Polymer structure and synthesis plausibility", _CP, (
            _requirement("polymer-validity", "Valid repeat unit, valence, and connectivity.", _COMPUTATIONAL, ("polymer_validity", "structure_validity"), fidelity=_CHEAP),
            _requirement("polymer-synthesis", "Polymerization and synthesis feasibility.", _COMPUTATIONAL, ("polymerization_feasibility", "synthesis_feasibility"), fidelity=_MEDIUM),
            _requirement("polymer-property-pred", "Predicted target bulk property.", _COMPUTATIONAL, ("target_property",), fidelity=_MEDIUM),
        ), reject=True),
        _gate("polymer-experimental", "Polymer identity and bulk performance", _EO, (
            _requirement("polymer-characterization", "Repeat-unit identity and molecular-weight distribution.", _EXPERIMENTAL, ("identity", "molecular_weight_distribution"), fidelity=_LAB, match=PropertyMatch.ALL),
            _requirement("polymer-target-exp", "Target thermal, mechanical, optical, or transport property measured.", _EXPERIMENTAL, ("target_property",), fidelity=_LAB),
            _requirement("polymer-durability", "Relevant stability or durability measurement.", _EXPERIMENTAL, ("durability", "thermal_stability", "chemical_stability"), fidelity=_LAB),
        )),
        _gate("polymer-replication", "Independent polymer performance replication", _IR, (
            _requirement("polymer-target-replicated", "Target property independently reproduced.", _EXPERIMENTAL, ("target_property",), fidelity=_LAB, records=2, sources=2),
        )),
    ),
)


BATTERY_PROFILE = _profile(
    DiscoveryDomain.BATTERIES,
    "Battery materials validation",
    (
        _gate("battery-plausibility", "Electrochemical material plausibility", _CP, (
            _requirement("battery-validity", "Composition and structure validity.", _COMPUTATIONAL, ("composition_validity", "structure_validity"), fidelity=_CHEAP),
            _requirement("battery-stability", "Thermodynamic/electrochemical stability window.", _COMPUTATIONAL, ("electrochemical_stability", "stability_window"), fidelity=_HIGH),
            _requirement("battery-transport", "Ion transport and voltage/capacity prediction.", _COMPUTATIONAL, ("ionic_conductivity", "diffusion_barrier", "voltage", "capacity"), fidelity=_HIGH),
        ), reject=True),
        _gate("battery-experimental", "Cell performance and material identity", _EO, (
            _requirement("battery-characterization", "Experimental phase/composition characterization.", _EXPERIMENTAL, ("phase_identity", "composition"), fidelity=_LAB, match=PropertyMatch.ALL),
            _requirement("battery-performance", "Capacity, efficiency, rate, or conductivity measured in a controlled cell.", _EXPERIMENTAL, ("capacity", "coulombic_efficiency", "rate_capability", "ionic_conductivity"), fidelity=_LAB, methods=(MethodClass.ELECTROCHEMICAL_TEST,)),
            _requirement("battery-cycling", "Multi-cycle retention and degradation evidence.", _EXPERIMENTAL, ("cycle_life", "capacity_retention", "degradation_rate"), fidelity=_LAB, methods=(MethodClass.ELECTROCHEMICAL_TEST,)),
            _requirement("battery-safety", "Relevant thermal or abuse safety evidence.", _EXPERIMENTAL, ("thermal_stability", "safety", "abuse_tolerance"), fidelity=_LAB),
        )),
        _gate("battery-replication", "Independent cell replication", _IR, (
            _requirement("battery-performance-replicated", "Performance reproduced in independently identified cells or sources.", _EXPERIMENTAL, ("capacity", "coulombic_efficiency", "ionic_conductivity"), fidelity=_LAB, records=2, sources=2, methods=(MethodClass.ELECTROCHEMICAL_TEST,)),
        )),
    ),
)


CATALYST_PROFILE = _profile(
    DiscoveryDomain.CATALYSTS,
    "Catalyst validation",
    (
        _gate("catalyst-plausibility", "Catalyst structure and mechanism plausibility", _CP, (
            _requirement("catalyst-validity", "Chemically valid catalyst composition/structure.", _COMPUTATIONAL, ("composition_validity", "structure_validity"), fidelity=_CHEAP),
            _requirement("catalyst-activity-pred", "Predicted reaction energetics or catalytic activity.", _COMPUTATIONAL, ("activation_energy", "adsorption_energy", "turnover_frequency", "activity"), fidelity=_HIGH),
            _requirement("catalyst-selectivity-pred", "Predicted selectivity for the target pathway.", _COMPUTATIONAL, ("selectivity", "reaction_selectivity"), fidelity=_HIGH),
        ), reject=True),
        _gate("catalyst-experimental", "Controlled catalytic performance", _EO, (
            _requirement("catalyst-identity", "Experimental catalyst identity and active phase.", _EXPERIMENTAL, ("phase_identity", "active_site_identity"), fidelity=_LAB),
            _requirement("catalyst-activity-exp", "Activity or turnover measured with controls.", _EXPERIMENTAL, ("activity", "turnover_frequency", "conversion"), fidelity=_LAB),
            _requirement("catalyst-selectivity-exp", "Selectivity or yield measured with material balance.", _EXPERIMENTAL, ("selectivity", "yield"), fidelity=_LAB),
            _requirement("catalyst-stability-exp", "Stability, deactivation, or reuse measured.", _EXPERIMENTAL, ("stability", "deactivation_rate", "reuse_cycles"), fidelity=_LAB),
        )),
        _gate("catalyst-replication", "Independent catalysis replication", _IR, (
            _requirement("catalyst-performance-replicated", "Activity and selectivity independently reproduced.", _EXPERIMENTAL, ("activity", "selectivity"), fidelity=_LAB, match=PropertyMatch.ALL, records=2, sources=2),
        )),
    ),
)


GENERAL_MATERIALS_PROFILE = _profile(
    DiscoveryDomain.GENERAL_MATERIALS,
    "General materials validation",
    (
        _gate("material-plausibility", "Material validity and property plausibility", _CP, (
            _requirement("material-validity", "Valid composition or structure.", _COMPUTATIONAL, ("composition_validity", "structure_validity", "validity"), fidelity=_CHEAP),
            _requirement("material-stability", "Relevant stability assessment.", _COMPUTATIONAL, ("stability", "thermodynamic_stability", "formation_energy"), fidelity=_MEDIUM),
            _requirement("material-target-pred", "Predicted target property.", _COMPUTATIONAL, ("target_property",), fidelity=_MEDIUM),
        ), reject=True),
        _gate("material-experimental", "Material identity and target-property observation", _EO, (
            _requirement("material-identity", "Experimental identity, phase, or composition.", _EXPERIMENTAL, ("identity", "phase_identity", "composition"), fidelity=_LAB),
            _requirement("material-target-exp", "Target property measured under documented conditions.", _EXPERIMENTAL, ("target_property",), fidelity=_LAB),
        )),
        _gate("material-replication", "Independent materials replication", _IR, (
            _requirement("material-target-replicated", "Target property independently reproduced.", _EXPERIMENTAL, ("target_property",), fidelity=_LAB, records=2, sources=2),
        )),
    ),
)


VALIDATION_PROFILES: dict[DiscoveryDomain, ValidationProfile] = {
    DiscoveryDomain.MEDICINAL_CHEMISTRY: MEDICINAL_CHEMISTRY_PROFILE,
    DiscoveryDomain.INORGANIC_MATERIALS: INORGANIC_MATERIALS_PROFILE,
    DiscoveryDomain.SUPERCONDUCTORS: SUPERCONDUCTOR_PROFILE,
    DiscoveryDomain.POLYMERS: POLYMER_PROFILE,
    DiscoveryDomain.BATTERIES: BATTERY_PROFILE,
    DiscoveryDomain.CATALYSTS: CATALYST_PROFILE,
    DiscoveryDomain.GENERAL_MATERIALS: GENERAL_MATERIALS_PROFILE,
}

_PROFILES_BY_ID = {profile.profile_id: profile for profile in VALIDATION_PROFILES.values()}
_FIDELITY_RANK = {Fidelity.CHEAP: 0, Fidelity.MEDIUM: 1, Fidelity.HIGH: 2}
_LEVEL_RANK = {
    ClaimLevel.GENERATED: 0,
    ClaimLevel.COMPUTATIONALLY_PLAUSIBLE: 1,
    ClaimLevel.EXPERIMENTALLY_OBSERVED: 2,
    ClaimLevel.INDEPENDENTLY_REPLICATED: 3,
}


__all__ = [
    "BATTERY_PROFILE",
    "CATALYST_PROFILE",
    "EvidenceRequirement",
    "GENERAL_MATERIALS_PROFILE",
    "GateEvaluator",
    "INORGANIC_MATERIALS_PROFILE",
    "MEDICINAL_CHEMISTRY_PROFILE",
    "POLYMER_PROFILE",
    "PropertyMatch",
    "SUPERCONDUCTOR_PROFILE",
    "VALIDATION_PROFILES",
    "ValidationGate",
    "ValidationGateEvaluator",
    "ValidationProfile",
    "evaluate_candidate",
    "get_validation_profile",
]
