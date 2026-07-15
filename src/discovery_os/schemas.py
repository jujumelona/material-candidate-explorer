"""Versioned, JSON-safe contracts for the discovery orchestration layer.

The contracts in this module deliberately describe *plans and evidence*, not
arbitrary executable code.  Every model rejects unknown fields so that a model
response cannot silently smuggle unrecognised instructions into a runtime.
"""

from __future__ import annotations

from ._compat import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated, Literal, TypeAlias

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator


SCHEMA_VERSION = "1.0"

Identifier: TypeAlias = Annotated[str, Field(min_length=1, max_length=256)]
NonEmptyText: TypeAlias = Annotated[str, Field(min_length=1, max_length=20_000)]
Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
NonNegativeFloat: TypeAlias = Annotated[float, Field(ge=0.0)]
PositiveInt: TypeAlias = Annotated[int, Field(gt=0)]
JsonObject: TypeAlias = dict[str, JsonValue]
JsonScalar: TypeAlias = str | int | float | bool | None


class StrictSchema(BaseModel):
    """Base for wire contracts shared by local and remote model adapters."""

    schema_version: Literal["1.0"] = SCHEMA_VERSION

    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        str_strip_whitespace=True,
        validate_assignment=True,
        validate_default=True,
        use_enum_values=True,
    )


class DiscoveryDomain(StrEnum):
    MEDICINAL_CHEMISTRY = "medicinal_chemistry"
    INORGANIC_MATERIALS = "inorganic_materials"
    SUPERCONDUCTORS = "superconductors"
    POLYMERS = "polymers"
    BATTERIES = "batteries"
    CATALYSTS = "catalysts"
    GENERAL_MATERIALS = "general_materials"


class CandidateType(StrEnum):
    SMALL_MOLECULE = "small_molecule"
    BIOLOGIC = "biologic"
    CRYSTAL = "crystal"
    COMPOSITION = "composition"
    ALLOY = "alloy"
    POLYMER = "polymer"
    BATTERY_MATERIAL = "battery_material"
    CATALYST = "catalyst"
    REACTION = "reaction"
    PROTEIN = "protein"
    RNA = "rna"
    CELL_STATE = "cell_state"
    CUSTOM = "custom"


class RepresentationKind(StrEnum):
    SMILES = "smiles"
    SELFIES = "selfies"
    INCHI = "inchi"
    SDF = "sdf"
    PROTEIN_SEQUENCE = "protein_sequence"
    RNA_SEQUENCE = "rna_sequence"
    FASTA = "fasta"
    PDB = "pdb"
    MMCIF = "mmcif"
    CHEMICAL_FORMULA = "chemical_formula"
    CIF = "cif"
    POSCAR = "poscar"
    POLYMER_REPEAT_UNIT = "polymer_repeat_unit"
    REACTION_SMILES = "reaction_smiles"
    CELL_EXPRESSION = "cell_expression"
    XYZ = "xyz"
    EXTXYZ = "extxyz"
    ELECTRONIC_STRUCTURE = "electronic_structure"
    CUSTOM = "custom"


class Fidelity(StrEnum):
    CHEAP = "cheap"
    MEDIUM = "medium"
    HIGH = "high"
    EXPERIMENTAL = "experimental"


class EvidenceKind(StrEnum):
    """The boundary must stay explicit; computation is not an experiment."""

    COMPUTATIONAL = "computational"
    EXPERIMENTAL = "experimental"


class VerificationStatus(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    REJECTED = "rejected"


class EvidenceStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    TIMEOUT = "timeout"


class MethodClass(StrEnum):
    RULE_BASED = "rule_based"
    MACHINE_LEARNING = "machine_learning"
    PHYSICS_SIMULATION = "physics_simulation"
    QUANTUM_CHEMISTRY = "quantum_chemistry"
    MOLECULAR_SIMULATION = "molecular_simulation"
    ANALYTICAL_MEASUREMENT = "analytical_measurement"
    BIOASSAY = "bioassay"
    MATERIALS_CHARACTERIZATION = "materials_characterization"
    ELECTROCHEMICAL_TEST = "electrochemical_test"
    OTHER = "other"


class UncertaintyKind(StrEnum):
    NONE = "none"
    EPISTEMIC = "epistemic"
    ALEATORIC = "aleatoric"
    STATISTICAL = "statistical"
    NUMERICAL = "numerical"
    MEASUREMENT = "measurement"
    EXPERIMENTAL = "experimental"
    COMBINED = "combined"
    UNKNOWN = "unknown"


class ClaimLevel(StrEnum):
    """Conservative evidence ladder; later levels may only be assigned by policy."""

    GENERATED = "generated"
    COMPUTATIONALLY_PLAUSIBLE = "computationally_plausible"
    EXPERIMENTALLY_OBSERVED = "experimentally_observed"
    INDEPENDENTLY_REPLICATED = "independently_replicated"


class ObjectiveDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"
    RANGE = "range"
    SATISFY = "satisfy"


class PropertyObjective(StrictSchema):
    property_name: Identifier
    direction: ObjectiveDirection
    unit: str | None = Field(default=None, max_length=128)
    target_value: JsonScalar = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    weight: NonNegativeFloat = 1.0
    required: bool = True
    rationale: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def _check_objective_shape(self) -> PropertyObjective:
        if self.direction == ObjectiveDirection.TARGET and self.target_value is None:
            raise ValueError("target objectives require target_value")
        if self.direction == ObjectiveDirection.RANGE:
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("range objectives require lower_bound and upper_bound")
            if self.lower_bound > self.upper_bound:
                raise ValueError("lower_bound must not exceed upper_bound")
        return self


class GoalConstraint(StrictSchema):
    constraint_id: Identifier
    description: NonEmptyText
    property_name: str | None = Field(default=None, max_length=256)
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "between", "contains", "excludes"] | None = None
    value: JsonValue = None
    hard: bool = True


class SuccessCriterion(StrictSchema):
    criterion_id: Identifier
    description: NonEmptyText
    property_name: str | None = Field(default=None, max_length=256)
    operator: Literal["eq", "ne", "gt", "gte", "lt", "lte", "between", "exists", "true", "false"] = "exists"
    threshold: JsonValue = None
    evidence_kind: EvidenceKind | None = None
    required: bool = True


class GoalCompileRequest(StrictSchema):
    user_text: NonEmptyText
    domain_hint: DiscoveryDomain | None = None
    context: JsonObject = Field(default_factory=dict)
    requested_validation_profile_id: str | None = Field(default=None, max_length=256)


class DiscoveryGoal(StrictSchema):
    goal_id: Identifier
    domain: DiscoveryDomain
    title: NonEmptyText
    scientific_question: NonEmptyText
    objectives: list[PropertyObjective] = Field(min_length=1)
    constraints: list[GoalConstraint] = Field(default_factory=list)
    success_criteria: list[SuccessCriterion] = Field(default_factory=list)
    validation_profile_id: Identifier
    candidate_types: list[CandidateType] = Field(min_length=1)
    assumptions: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    max_cycles: PositiveInt = 10

    @model_validator(mode="after")
    def _unique_goal_ids(self) -> DiscoveryGoal:
        _ensure_unique((item.constraint_id for item in self.constraints), "constraint_id")
        _ensure_unique((item.criterion_id for item in self.success_criteria), "criterion_id")
        return self


class DiscoveryState(StrictSchema):
    run_id: Identifier
    goal_id: Identifier
    cycle: int = Field(default=0, ge=0)
    hypothesis_ids: list[Identifier] = Field(default_factory=list)
    candidate_ids: list[Identifier] = Field(default_factory=list)
    evidence_ids: list[Identifier] = Field(default_factory=list)
    rejected_candidate_ids: list[Identifier] = Field(default_factory=list)
    metrics: JsonObject = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class DiscoveryHistorySummary(StrictSchema):
    run_id: Identifier
    completed_cycles: int = Field(ge=0)
    generated_candidate_count: int = Field(default=0, ge=0)
    evaluated_candidate_count: int = Field(default=0, ge=0)
    experimental_candidate_count: int = Field(default=0, ge=0)
    failed_call_count: int = Field(default=0, ge=0)
    key_findings: list[str] = Field(default_factory=list)
    rejected_assumptions: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    best_candidate_ids: list[Identifier] = Field(default_factory=list)
    aggregate_metrics: JsonObject = Field(default_factory=dict)


class HypothesisStatus(StrEnum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    WEAKENED = "weakened"
    REJECTED = "rejected"


class Hypothesis(StrictSchema):
    hypothesis_id: Identifier
    statement: NonEmptyText
    mechanism: NonEmptyText
    predicted_observations: list[str] = Field(min_length=1)
    falsification_criteria: list[str] = Field(min_length=1)
    related_objectives: list[str] = Field(default_factory=list)
    confidence: Probability
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)
    contradicting_evidence_ids: list[Identifier] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class HypothesisRequest(StrictSchema):
    goal: DiscoveryGoal
    state: DiscoveryState
    history_summary: DiscoveryHistorySummary | None = None
    max_hypotheses: PositiveInt = 8


class HypothesisBatch(StrictSchema):
    hypotheses: list[Hypothesis]
    batch_reason: str | None = Field(default=None, max_length=8_000)

    @model_validator(mode="after")
    def _unique_hypotheses(self) -> HypothesisBatch:
        _ensure_unique((item.hypothesis_id for item in self.hypotheses), "hypothesis_id")
        return self


class CandidateRepresentation(StrictSchema):
    kind: RepresentationKind
    value: NonEmptyText
    media_type: str | None = Field(default=None, max_length=256)
    format_version: str | None = Field(default=None, max_length=128)
    canonical: bool = False
    metadata: JsonObject = Field(default_factory=dict)


class CandidateRef(StrictSchema):
    """Immutable reference used by cache keys, plans, and evidence provenance."""

    candidate_id: Identifier
    version: PositiveInt
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")


class Candidate(StrictSchema):
    candidate_id: Identifier
    candidate_type: CandidateType
    domain: DiscoveryDomain
    candidate_ref: CandidateRef | None = None
    name: str | None = Field(default=None, max_length=512)
    representations: list[CandidateRepresentation] = Field(min_length=1)
    parent_candidate_ids: list[Identifier] = Field(default_factory=list)
    parent_candidate_refs: list[CandidateRef] = Field(default_factory=list)
    hypothesis_ids: list[Identifier] = Field(default_factory=list)
    generation_task_id: str | None = Field(default=None, max_length=256)
    attributes: JsonObject = Field(default_factory=dict)
    novelty_rationale: str | None = Field(default=None, max_length=8_000)
    provenance: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_candidate_ref(self) -> Candidate:
        if self.candidate_ref is not None and self.candidate_ref.candidate_id != self.candidate_id:
            raise ValueError("candidate_ref.candidate_id must match candidate_id")
        if self.candidate_ref is not None:
            from .hashing import candidate_content_hash

            if candidate_content_hash(self) != self.candidate_ref.content_hash:
                raise ValueError("candidate_ref.content_hash must match candidate content")
        parent_ref_ids = [item.candidate_id for item in self.parent_candidate_refs]
        if len(parent_ref_ids) != len(set(parent_ref_ids)):
            raise ValueError("duplicate parent_candidate_refs are not allowed")
        if not set(parent_ref_ids).issubset(self.parent_candidate_ids):
            raise ValueError("parent_candidate_refs must also appear in parent_candidate_ids")
        return self


class ParameterType(StrEnum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


class ParameterDescriptor(StrictSchema):
    name: Identifier
    value_type: ParameterType
    required: bool = False
    description: str | None = Field(default=None, max_length=2_000)
    allowed_values: list[JsonScalar] = Field(default_factory=list)
    default: JsonValue = None
    unit: str | None = Field(default=None, max_length=128)


class ResourceBudget(StrictSchema):
    cpu_cores: NonNegativeFloat = 0.0
    gpu_count: NonNegativeFloat = 0.0
    memory_gb: NonNegativeFloat = 0.0
    storage_gb: NonNegativeFloat = 0.0
    estimated_cost: NonNegativeFloat = 0.0
    extras: dict[str, NonNegativeFloat] = Field(default_factory=dict)


class ToolOperationDescriptor(StrictSchema):
    operation: Identifier
    description: NonEmptyText
    supported_domains: list[DiscoveryDomain] = Field(min_length=1)
    supported_candidate_types: list[CandidateType] = Field(min_length=1)
    method_class: MethodClass = MethodClass.OTHER
    produced_properties: list[Identifier] = Field(default_factory=list)
    evidence_kinds: list[EvidenceKind] = Field(min_length=1)
    supported_fidelities: list[Fidelity] = Field(min_length=1)
    condition_parameters: list[ParameterDescriptor] = Field(default_factory=list)
    default_max_runtime_seconds: PositiveInt = 300
    requires_human_approval: bool = False


class ToolDescriptor(StrictSchema):
    tool_name: Identifier
    tool_version: Identifier
    adapter_version: Identifier
    description: NonEmptyText
    operations: list[ToolOperationDescriptor] = Field(min_length=1)
    available: bool = True
    deterministic: bool = False
    default_resource_budget: ResourceBudget = Field(default_factory=ResourceBudget)
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_operations(self) -> ToolDescriptor:
        _ensure_unique((item.operation for item in self.operations), "operation")
        return self


class GeneratorDescriptor(StrictSchema):
    generator_name: Identifier
    generator_version: Identifier
    adapter_version: Identifier
    description: NonEmptyText
    supported_domains: list[DiscoveryDomain] = Field(min_length=1)
    supported_candidate_types: list[CandidateType] = Field(min_length=1)
    accepted_parameters: list[ParameterDescriptor] = Field(default_factory=list)
    supports_parent_conditioning: bool = False
    deterministic: bool = False
    available: bool = True
    default_resource_budget: ResourceBudget = Field(default_factory=ResourceBudget)
    metadata: JsonObject = Field(default_factory=dict)


class GenerationTask(StrictSchema):
    task_id: Identifier | None = None
    generator_name: Identifier
    candidate_type: CandidateType
    requested_count: PositiveInt
    parent_candidate_ids: list[Identifier] = Field(default_factory=list)
    hypothesis_ids: list[Identifier] = Field(default_factory=list)
    target_properties: JsonObject = Field(default_factory=dict)
    preserve_features: list[str] = Field(default_factory=list)
    modify_features: list[str] = Field(default_factory=list)
    forbidden_features: list[str] = Field(default_factory=list)
    diversity_strength: Probability = 0.5
    novelty_strength: Probability = 0.5
    conditions: JsonObject = Field(default_factory=dict)
    max_runtime_seconds: PositiveInt = 3_600
    resource_budget: ResourceBudget = Field(default_factory=ResourceBudget)
    reason: NonEmptyText


class ValidationIntent(StrictSchema):
    """Model-selected scientific intent that deterministic code may compile.

    An intent cannot name a shell command.  A PlanCompiler must match it to an
    allow-listed ``ToolOperationDescriptor`` and produce concrete ``ToolCall``
    objects after compatibility and budget checks.
    """

    intent_id: Identifier
    candidate_refs: list[CandidateRef] = Field(min_length=1)
    requested_properties: list[Identifier] = Field(min_length=1)
    required_evidence_kind: EvidenceKind
    minimum_fidelity: Fidelity
    preferred_method_classes: list[MethodClass] = Field(default_factory=list)
    conditions: JsonObject = Field(default_factory=dict)
    priority: Probability
    reason: NonEmptyText
    max_runtime_seconds: PositiveInt
    resource_budget: ResourceBudget = Field(default_factory=ResourceBudget)

    @model_validator(mode="after")
    def _check_intent_fidelity(self) -> ValidationIntent:
        if self.required_evidence_kind == EvidenceKind.EXPERIMENTAL and self.minimum_fidelity != Fidelity.EXPERIMENTAL:
            raise ValueError("experimental intents must request experimental fidelity")
        if self.required_evidence_kind == EvidenceKind.COMPUTATIONAL and self.minimum_fidelity == Fidelity.EXPERIMENTAL:
            raise ValueError("computational intents cannot request experimental fidelity")
        return self


class CandidateProposalRequest(StrictSchema):
    goal: DiscoveryGoal
    state: DiscoveryState
    hypotheses: HypothesisBatch
    available_generators: list[GeneratorDescriptor] = Field(default_factory=list)


class CandidatePlan(StrictSchema):
    tasks: list[GenerationTask]
    plan_reason: str | None = Field(default=None, max_length=8_000)

    @model_validator(mode="after")
    def _unique_task_ids(self) -> CandidatePlan:
        ids = [task.task_id for task in self.tasks if task.task_id is not None]
        _ensure_unique(ids, "task_id")
        return self


class CandidateBatch(StrictSchema):
    candidates: list[Candidate]
    generation_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_candidates(self) -> CandidateBatch:
        _ensure_unique((item.candidate_id for item in self.candidates), "candidate_id")
        return self


class PropertyPrediction(StrictSchema):
    property_name: Identifier
    value: JsonScalar
    unit: str | None = Field(default=None, max_length=128)
    uncertainty: NonNegativeFloat | None = None
    uncertainty_kind: UncertaintyKind = UncertaintyKind.UNKNOWN
    lower_bound: float | None = None
    upper_bound: float | None = None
    confidence: Probability | None = None
    method: str | None = Field(default=None, max_length=512)
    calibrated: bool = False
    calibration_method: str | None = Field(default=None, max_length=1_000)
    conditions: JsonObject = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    applicability_warnings: list[str] = Field(default_factory=list)
    applicability: ApplicabilityAssessment | None = None

    @model_validator(mode="after")
    def _check_interval(self) -> PropertyPrediction:
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        return self


class CandidatePrediction(StrictSchema):
    candidate_id: Identifier
    properties: list[PropertyPrediction]
    overall_confidence: Probability | None = None
    out_of_distribution: bool = False
    risks: list[str] = Field(default_factory=list)
    recommended_validation_properties: list[str] = Field(default_factory=list)


class ApplicabilityAssessment(StrictSchema):
    in_domain: bool
    score: Probability | None = None
    domain_description: str | None = Field(default=None, max_length=4_000)
    reasons: list[str] = Field(default_factory=list)


class PredictionRequest(StrictSchema):
    goal: DiscoveryGoal
    state: DiscoveryState
    candidates: list[Candidate]
    requested_properties: list[str] = Field(default_factory=list)


class PredictionBatch(StrictSchema):
    predictions: list[CandidatePrediction]
    batch_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_predictions(self) -> PredictionBatch:
        _ensure_unique((item.candidate_id for item in self.predictions), "prediction candidate_id")
        return self


class ToolCall(StrictSchema):
    call_id: Identifier
    tool_name: Identifier
    operation: Identifier
    candidate_ids: list[Identifier] = Field(min_length=1)
    requested_properties: list[Identifier] = Field(default_factory=list)
    conditions: JsonObject = Field(default_factory=dict)
    evidence_kind: EvidenceKind = EvidenceKind.COMPUTATIONAL
    method_class: MethodClass = MethodClass.OTHER
    fidelity: Fidelity
    priority: Probability
    reason: NonEmptyText
    max_runtime_seconds: PositiveInt
    resource_budget: ResourceBudget | dict[str, NonNegativeFloat] = Field(default_factory=ResourceBudget)
    depends_on_call_ids: list[Identifier] = Field(default_factory=list)
    retry_limit: int = Field(default=0, ge=0, le=10)
    cache_allowed: bool = True

    @model_validator(mode="after")
    def _check_evidence_fidelity(self) -> ToolCall:
        if self.evidence_kind == EvidenceKind.EXPERIMENTAL and self.fidelity != Fidelity.EXPERIMENTAL:
            raise ValueError("experimental calls must use experimental fidelity")
        if self.evidence_kind == EvidenceKind.COMPUTATIONAL and self.fidelity == Fidelity.EXPERIMENTAL:
            raise ValueError("computational calls cannot use experimental fidelity")
        return self


class ValidationPlanningRequest(StrictSchema):
    goal: DiscoveryGoal
    state: DiscoveryState
    candidates: list[Candidate]
    predictions: PredictionBatch
    available_tools: list[ToolDescriptor]
    validation_profile_id: str | None = Field(default=None, max_length=256)
    prior_evidence: EvidenceBatch | None = None
    max_total_runtime_seconds: PositiveInt | None = None
    total_resource_budget: ResourceBudget | None = None


class ValidationPlan(StrictSchema):
    intents: list[ValidationIntent] = Field(default_factory=list)
    calls: list[ToolCall]
    expected_information_gain: dict[str, Probability] = Field(default_factory=dict)
    plan_reason: NonEmptyText

    @model_validator(mode="after")
    def _validate_call_graph(self) -> ValidationPlan:
        _ensure_unique((intent.intent_id for intent in self.intents), "intent_id")
        call_ids = [call.call_id for call in self.calls]
        _ensure_unique(call_ids, "call_id")
        known = set(call_ids)
        for call in self.calls:
            unknown = set(call.depends_on_call_ids) - known
            if unknown:
                raise ValueError(f"call {call.call_id!r} has unknown dependencies: {sorted(unknown)}")
            if call.call_id in call.depends_on_call_ids:
                raise ValueError(f"call {call.call_id!r} cannot depend on itself")
        intent_ids = {intent.intent_id for intent in self.intents}
        extra_gain_ids = set(self.expected_information_gain) - known - intent_ids
        if extra_gain_ids:
            raise ValueError(f"information gain references unknown calls or intents: {sorted(extra_gain_ids)}")
        return self


class PropertyResult(StrictSchema):
    property_name: Identifier
    value: JsonScalar
    unit: str | None = Field(default=None, max_length=128)
    uncertainty: NonNegativeFloat | None = None
    uncertainty_kind: UncertaintyKind = UncertaintyKind.UNKNOWN
    confidence: Probability | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    meets_criterion: bool | None = None
    criterion: str | None = Field(default=None, max_length=2_000)
    conditions: JsonObject = Field(default_factory=dict)
    calibration_method: str | None = Field(default=None, max_length=1_000)
    applicability: ApplicabilityAssessment | None = None
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_result_interval(self) -> PropertyResult:
        if self.lower_bound is not None and self.upper_bound is not None and self.lower_bound > self.upper_bound:
            raise ValueError("lower_bound must not exceed upper_bound")
        return self


class ComputationalEvidenceDetails(StrictSchema):
    method_name: Identifier
    method_version: Identifier
    model_name: str | None = Field(default=None, max_length=256)
    model_version: str | None = Field(default=None, max_length=256)
    dataset_versions: dict[str, str] = Field(default_factory=dict)
    parameters: JsonObject = Field(default_factory=dict)
    random_seed: int | None = None
    hardware: str | None = Field(default=None, max_length=1_000)
    code_revision: str | None = Field(default=None, max_length=256)
    applicability: ApplicabilityAssessment | None = None
    calibration_method: str | None = Field(default=None, max_length=512)
    quality_flags: list[str] = Field(default_factory=list)


class ExperimentalEvidenceDetails(StrictSchema):
    protocol_id: Identifier
    sample_id: Identifier
    laboratory: Identifier
    instrument: str = Field(min_length=1, max_length=512)
    operator: str = Field(min_length=1, max_length=256)
    replicate_id: str = Field(min_length=1, max_length=256)
    controls: list[str] = Field(min_length=1)
    blinded: bool | None = None
    conditions: JsonObject = Field(min_length=1)


class EvidenceVerification(StrictSchema):
    status: VerificationStatus = VerificationStatus.NOT_APPLICABLE
    verifier_id: str | None = Field(default=None, max_length=256)
    attestation_id: str | None = Field(default=None, max_length=256)
    method: str | None = Field(default=None, max_length=256)
    reason: NonEmptyText

    @model_validator(mode="after")
    def _verified_requires_attestation(self) -> EvidenceVerification:
        if self.status == VerificationStatus.VERIFIED:
            if not self.verifier_id or not self.attestation_id or not self.method:
                raise ValueError(
                    "verified evidence requires verifier_id, attestation_id, and method"
                )
        return self


class EvidenceRecord(StrictSchema):
    evidence_id: Identifier
    call_id: Identifier
    candidate_id: Identifier
    tool_name: Identifier
    tool_version: Identifier
    operation: Identifier
    method_class: MethodClass = MethodClass.OTHER
    status: EvidenceStatus
    evidence_kind: EvidenceKind
    fidelity: Fidelity
    properties: list[PropertyResult] = Field(default_factory=list)
    failure_modes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    runtime_seconds: NonNegativeFloat
    input_hash: Identifier
    output_hash: Identifier
    parameters_hash: str | None = Field(default=None, max_length=256)
    container_digest: str | None = Field(default=None, max_length=512)
    convergence_checks: dict[str, bool] = Field(default_factory=dict)
    source_id: str | None = Field(default=None, max_length=256)
    candidate_ref: CandidateRef | None = None
    computational_details: ComputationalEvidenceDetails | None = None
    experimental_details: ExperimentalEvidenceDetails | None = None
    verification: EvidenceVerification = Field(
        default_factory=lambda: EvidenceVerification(
            status=VerificationStatus.NOT_APPLICABLE,
            reason="Verification is not applicable to computational evidence.",
        )
    )
    observed_at: AwareDatetime | None = None

    @model_validator(mode="after")
    def _enforce_evidence_boundary(self) -> EvidenceRecord:
        for artifact_path in self.artifact_paths:
            posix = PurePosixPath(artifact_path.replace("\\", "/"))
            windows = PureWindowsPath(artifact_path)
            if (
                not artifact_path
                or posix.is_absolute()
                or windows.is_absolute()
                or ".." in posix.parts
                or windows.drive
            ):
                raise ValueError(
                    "artifact_paths must be confined relative paths without parent traversal"
                )
        if self.candidate_ref is not None and self.candidate_ref.candidate_id != self.candidate_id:
            raise ValueError("candidate_ref.candidate_id must match candidate_id")
        if self.evidence_kind == EvidenceKind.COMPUTATIONAL:
            if self.fidelity == Fidelity.EXPERIMENTAL:
                raise ValueError("computational evidence cannot use experimental fidelity")
            if self.experimental_details is not None:
                raise ValueError("computational evidence cannot contain experimental_details")
            if self.computational_details is None:
                raise ValueError("computational evidence requires computational_details")
            if self.verification.status != VerificationStatus.NOT_APPLICABLE:
                raise ValueError("computational evidence must use not_applicable verification")
        else:
            if self.fidelity != Fidelity.EXPERIMENTAL:
                raise ValueError("experimental evidence must use experimental fidelity")
            if self.computational_details is not None:
                raise ValueError("experimental evidence cannot contain computational_details")
            if self.experimental_details is None:
                raise ValueError("experimental evidence requires experimental_details")
            if not self.source_id:
                raise ValueError("experimental evidence requires an explicit source_id")
            if self.verification.status == VerificationStatus.NOT_APPLICABLE:
                raise ValueError("experimental evidence requires a verification result")
            if self.status == EvidenceStatus.SUCCESS and self.verification.status != VerificationStatus.VERIFIED:
                raise ValueError("successful experimental evidence must be verified")
            if self.status == EvidenceStatus.SUCCESS and not self.artifact_paths:
                raise ValueError(
                    "successful experimental evidence requires immutable raw-data artifacts"
                )
            if self.status == EvidenceStatus.SUCCESS and self.observed_at is None:
                raise ValueError("successful experimental evidence requires observed_at")
        return self


class EvidenceBatch(StrictSchema):
    records: list[EvidenceRecord]
    batch_id: str | None = Field(default=None, max_length=256)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_evidence(self) -> EvidenceBatch:
        _ensure_unique((item.evidence_id for item in self.records), "evidence_id")
        return self


class PredictionError(StrictSchema):
    candidate_id: Identifier
    property_name: Identifier
    predicted_value: float | None = None
    observed_value: float | None = None
    normalized_error: NonNegativeFloat | None = None
    likely_causes: list[str] = Field(default_factory=list)


class HypothesisUpdate(StrictSchema):
    hypothesis_id: Identifier
    previous_confidence: Probability
    updated_confidence: Probability
    updated_status: HypothesisStatus
    supporting_evidence_ids: list[Identifier] = Field(default_factory=list)
    contradicting_evidence_ids: list[Identifier] = Field(default_factory=list)
    rationale: NonEmptyText


class ResultAnalysisRequest(StrictSchema):
    goal: DiscoveryGoal
    current_hypotheses: list[Hypothesis]
    candidates: list[Candidate]
    predictions_before_validation: list[CandidatePrediction]
    observed_evidence: EvidenceBatch
    history_summary: DiscoveryHistorySummary


class ResultAnalysis(StrictSchema):
    confirmed_findings: list[str] = Field(default_factory=list)
    rejected_assumptions: list[str] = Field(default_factory=list)
    prediction_errors: list[PredictionError] = Field(default_factory=list)
    hypothesis_updates: list[HypothesisUpdate] = Field(default_factory=list)
    newly_detected_patterns: list[str] = Field(default_factory=list)
    candidates_to_keep: list[Identifier] = Field(default_factory=list)
    candidates_to_remove: list[Identifier] = Field(default_factory=list)
    candidates_to_revise: list[Identifier] = Field(default_factory=list)
    next_recommended_action: Literal["validate_more", "revise", "generate_new", "finish"]
    rationale: str | None = Field(default=None, max_length=8_000)

    @model_validator(mode="after")
    def _disjoint_candidate_actions(self) -> ResultAnalysis:
        keep = set(self.candidates_to_keep)
        remove = set(self.candidates_to_remove)
        revise = set(self.candidates_to_revise)
        if keep & remove or keep & revise or remove & revise:
            raise ValueError("candidate keep/remove/revise sets must be disjoint")
        return self


class RevisionOperation(StrEnum):
    SUBSTITUTE = "substitute"
    ADD = "add"
    REMOVE = "remove"
    REPLACE = "replace"
    OPTIMIZE = "optimize"
    REGENERATE = "regenerate"


class CandidateRevision(StrictSchema):
    revision_id: Identifier
    candidate_id: Identifier
    operation: RevisionOperation
    feature: NonEmptyText
    proposed_change: JsonValue
    preserve_features: list[str] = Field(default_factory=list)
    expected_effects: JsonObject = Field(default_factory=dict)
    required_followup_properties: list[str] = Field(default_factory=list)
    based_on_evidence_ids: list[Identifier] = Field(min_length=1)
    reason: NonEmptyText


class RevisionRequest(StrictSchema):
    goal: DiscoveryGoal
    state: DiscoveryState
    candidates: list[Candidate]
    evidence: EvidenceBatch
    analysis: ResultAnalysis
    available_generators: list[GeneratorDescriptor] = Field(default_factory=list)


class RevisionPlan(StrictSchema):
    revisions: list[CandidateRevision] = Field(default_factory=list)
    generation_tasks: list[GenerationTask] = Field(default_factory=list)
    candidates_to_retain: list[Identifier] = Field(default_factory=list)
    candidates_to_retire: list[Identifier] = Field(default_factory=list)
    plan_reason: NonEmptyText

    @model_validator(mode="after")
    def _validate_revision_plan(self) -> RevisionPlan:
        _ensure_unique((item.revision_id for item in self.revisions), "revision_id")
        overlap = set(self.candidates_to_retain) & set(self.candidates_to_retire)
        if overlap:
            raise ValueError(f"candidates cannot be both retained and retired: {sorted(overlap)}")
        return self


class StopReason(StrEnum):
    SUCCESS_CRITERIA_MET = "success_criteria_met"
    BUDGET_EXHAUSTED = "budget_exhausted"
    MAX_CYCLES_REACHED = "max_cycles_reached"
    NO_PROGRESS = "no_progress"
    SAFETY_BLOCK = "safety_block"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONTINUE = "continue"


class StopDecisionRequest(StrictSchema):
    goal: DiscoveryGoal
    state: DiscoveryState
    cycle: int = Field(ge=0)
    history_summary: DiscoveryHistorySummary | None = None
    latest_analysis: ResultAnalysis | None = None
    validation_assessments: list[CandidateValidationAssessment] = Field(default_factory=list)


class StopDecision(StrictSchema):
    stop: bool
    reason_code: StopReason
    reason: NonEmptyText
    unmet_criteria: list[str] = Field(default_factory=list)
    best_candidate_ids: list[Identifier] = Field(default_factory=list)
    recommended_next_action: Literal["validate_more", "revise", "generate_new", "finish", "human_review"]

    @model_validator(mode="after")
    def _consistent_stop(self) -> StopDecision:
        if self.stop and self.reason_code == StopReason.CONTINUE:
            raise ValueError("a stop decision cannot use reason_code='continue'")
        if not self.stop and self.recommended_next_action == "finish":
            raise ValueError("a continue decision cannot recommend finish")
        return self


class RequirementDecision(StrictSchema):
    requirement_id: Identifier
    satisfied: bool
    status: Literal["passed", "failed", "insufficient_evidence"]
    matched_evidence_ids: list[Identifier] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    reason: NonEmptyText

    @model_validator(mode="after")
    def _check_requirement_decision(self) -> RequirementDecision:
        if self.satisfied != (self.status == "passed"):
            raise ValueError("satisfied must be true exactly when status is passed")
        if self.satisfied and not self.matched_evidence_ids:
            raise ValueError("a satisfied requirement must cite evidence")
        return self


class GateDecision(StrictSchema):
    gate_id: Identifier
    passed: bool
    status: Literal["passed", "failed", "insufficient_evidence"]
    requirement_decisions: list[RequirementDecision]
    matched_evidence_ids: list[Identifier] = Field(default_factory=list)
    reason: NonEmptyText

    @model_validator(mode="after")
    def _check_gate_decision(self) -> GateDecision:
        if self.passed != (self.status == "passed"):
            raise ValueError("passed must be true exactly when status is passed")
        if self.passed and (not self.requirement_decisions or not all(r.satisfied for r in self.requirement_decisions)):
            raise ValueError("a passed gate requires every requirement to pass")
        return self


class CandidateValidationStatus(StrEnum):
    UNVALIDATED = "unvalidated"
    COMPUTATIONALLY_SUPPORTED = "computationally_supported"
    EXPERIMENTALLY_VALIDATED = "experimentally_validated"
    REJECTED = "rejected"
    INCONCLUSIVE = "inconclusive"


class CandidateValidationAssessment(StrictSchema):
    candidate_id: Identifier
    profile_id: Identifier
    status: CandidateValidationStatus
    claim_level: ClaimLevel = ClaimLevel.GENERATED
    gate_decisions: list[GateDecision]
    matched_evidence_ids: list[Identifier] = Field(default_factory=list)
    matched_evidence_kinds: list[EvidenceKind] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _never_infer_validation_without_evidence(self) -> CandidateValidationAssessment:
        passed = bool(self.gate_decisions) and all(gate.passed for gate in self.gate_decisions)
        if self.status == CandidateValidationStatus.EXPERIMENTALLY_VALIDATED:
            if not passed:
                raise ValueError("experimentally_validated requires every profile gate to pass")
            if not self.matched_evidence_ids:
                raise ValueError("experimentally_validated requires cited evidence")
            if EvidenceKind.EXPERIMENTAL not in self.matched_evidence_kinds:
                raise ValueError("experimentally_validated requires experimental evidence")
            if self.claim_level not in {ClaimLevel.EXPERIMENTALLY_OBSERVED, ClaimLevel.INDEPENDENTLY_REPLICATED}:
                raise ValueError("experimentally_validated requires an experimental claim level")
        if self.status == CandidateValidationStatus.COMPUTATIONALLY_SUPPORTED:
            if EvidenceKind.COMPUTATIONAL not in self.matched_evidence_kinds:
                raise ValueError("computationally_supported requires computational evidence")
            if not self.matched_evidence_ids:
                raise ValueError("computationally_supported requires cited evidence")
            if self.claim_level != ClaimLevel.COMPUTATIONALLY_PLAUSIBLE:
                raise ValueError("computationally_supported requires computationally_plausible claim level")
        if self.claim_level in {ClaimLevel.EXPERIMENTALLY_OBSERVED, ClaimLevel.INDEPENDENTLY_REPLICATED}:
            if EvidenceKind.EXPERIMENTAL not in self.matched_evidence_kinds:
                raise ValueError("experimental claim levels require experimental evidence")
        return self


class FinalCandidateReport(StrictSchema):
    candidate: Candidate
    predictions: CandidatePrediction | None = None
    validation: CandidateValidationAssessment
    evidence_ids: list[Identifier] = Field(default_factory=list)
    disposition_reason: NonEmptyText
    recommended_next_steps: list[str] = Field(default_factory=list)


class DiscoveryFinalReport(StrictSchema):
    run_id: Identifier
    goal: DiscoveryGoal
    stop_decision: StopDecision
    candidate_reports: list[FinalCandidateReport]
    history_summary: DiscoveryHistorySummary
    conclusions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    safety_and_ethics_notes: list[str] = Field(default_factory=list)
    generated_at: str | None = Field(default=None, max_length=128)


# Convenient public spelling for stores and API handlers.
FinalReport = DiscoveryFinalReport


def _ensure_unique(values: object, label: str) -> None:
    """Raise a stable validation error for duplicate identifiers."""

    materialized = list(values)  # type: ignore[arg-type]
    if len(materialized) != len(set(materialized)):
        raise ValueError(f"duplicate {label} values are not allowed")


__all__ = [
    "SCHEMA_VERSION",
    "Candidate",
    "CandidateBatch",
    "CandidatePlan",
    "CandidatePrediction",
    "CandidateProposalRequest",
    "CandidateRepresentation",
    "CandidateRef",
    "CandidateRevision",
    "CandidateType",
    "CandidateValidationAssessment",
    "CandidateValidationStatus",
    "ClaimLevel",
    "ComputationalEvidenceDetails",
    "DiscoveryDomain",
    "DiscoveryFinalReport",
    "DiscoveryGoal",
    "DiscoveryHistorySummary",
    "DiscoveryState",
    "EvidenceBatch",
    "EvidenceKind",
    "EvidenceRecord",
    "EvidenceStatus",
    "ExperimentalEvidenceDetails",
    "EvidenceVerification",
    "Fidelity",
    "FinalCandidateReport",
    "FinalReport",
    "GateDecision",
    "GenerationTask",
    "GeneratorDescriptor",
    "GoalCompileRequest",
    "GoalConstraint",
    "Hypothesis",
    "HypothesisBatch",
    "HypothesisRequest",
    "HypothesisStatus",
    "HypothesisUpdate",
    "ApplicabilityAssessment",
    "MethodClass",
    "ObjectiveDirection",
    "ParameterDescriptor",
    "ParameterType",
    "PredictionBatch",
    "PredictionError",
    "PredictionRequest",
    "PropertyObjective",
    "PropertyPrediction",
    "PropertyResult",
    "ReportCandidate",
    "RequirementDecision",
    "ResourceBudget",
    "ResultAnalysis",
    "ResultAnalysisRequest",
    "RevisionOperation",
    "RevisionPlan",
    "RevisionRequest",
    "StopDecision",
    "StopDecisionRequest",
    "StopReason",
    "StrictSchema",
    "SuccessCriterion",
    "ToolCall",
    "ToolDescriptor",
    "ToolOperationDescriptor",
    "UncertaintyKind",
    "VerificationStatus",
    "ValidationIntent",
    "ValidationPlan",
    "ValidationPlanningRequest",
]


# Backward-friendly semantic alias; it remains the same strict model.
ReportCandidate = FinalCandidateReport
