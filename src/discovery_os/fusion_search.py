"""Multi-round, branch-preserving search over the validated Fusion loop.

The search runner does not invent objective scores and never averages latent or
expert outputs.  It delegates scientific-vector selection to
``DeterministicExplorationSelector`` and records every raw primary expert
payload through ``ExpertEvidenceStore`` before a candidate can enter a branch.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import fsum
from typing import Literal, Protocol

from pydantic import Field, JsonValue, model_validator

from ._compat import StrEnum
from .fusion_exploration import (
    AdaptiveGenerationScheduler,
    CandidatePool,
    CandidatePoolEntry,
    DeterministicExplorationSelector,
    ExcludedCandidate,
    ExpertEvidenceStore,
    ExplorationBranch,
    ExplorationBranchResult,
    ExplorationSelection,
    SchedulerDecision,
    SchedulerObservation,
)
from .fusion_loop import FusionLoopRunner
from .fusion_runtime import FusionRuntimeError
from .fusion_schemas import (
    ContentArtifactRef,
    DiagnosticProperty,
    ExpertFeaturePayload,
    FeatureStatus,
    FusionBatchIterationReport,
    FusionDecisionContext,
    GenerationControls,
    GeneratorProvenance,
    ScientificModality,
    UnifiedLatentStateRef,
    WorkspaceEntityInput,
    WorkspaceMode,
    WorkspaceRelation,
    WorkspaceRunConfig,
)
from .hashing import bytes_hash, candidate_content_hash, canonical_json, stable_hash
from .literature_rag import EvidenceBranchAssignment, LiteratureEvidencePolicy
from .schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    DiscoveryGoal,
    Identifier,
    NonEmptyText,
    ObjectiveDirection,
    PropertyObjective,
    RepresentationKind,
    StrictSchema,
)


class FusionSearchError(RuntimeError):
    """Raised when a search cannot preserve lineage, evidence, or provenance."""


class FusionSearchStatus(StrEnum):
    COMPLETED = "completed"
    EXHAUSTED = "exhausted"
    PARTIAL = "partial"
    FAILED = "failed"


class SearchRepresentationArtifactEncoding(StrEnum):
    RAW_UTF8 = "raw_utf8"
    CANONICAL_JSON = "canonical_json"


@dataclass(frozen=True)
class _RepresentationArtifactSpec:
    profile: str
    extension: str
    media_type: str
    encoding: SearchRepresentationArtifactEncoding


_RAW_REPRESENTATION_SPECS = {
    RepresentationKind.CIF: _RepresentationArtifactSpec(
        "cif", ".cif", "chemical/x-cif", SearchRepresentationArtifactEncoding.RAW_UTF8
    ),
    RepresentationKind.MMCIF: _RepresentationArtifactSpec(
        "cif", ".cif", "chemical/x-cif", SearchRepresentationArtifactEncoding.RAW_UTF8
    ),
    RepresentationKind.SMILES: _RepresentationArtifactSpec(
        "smiles",
        ".smi",
        "chemical/x-daylight-smiles",
        SearchRepresentationArtifactEncoding.RAW_UTF8,
    ),
    RepresentationKind.REACTION_SMILES: _RepresentationArtifactSpec(
        "smiles",
        ".smi",
        "chemical/x-daylight-smiles",
        SearchRepresentationArtifactEncoding.RAW_UTF8,
    ),
    RepresentationKind.FASTA: _RepresentationArtifactSpec(
        "fasta", ".fasta", "text/x-fasta", SearchRepresentationArtifactEncoding.RAW_UTF8
    ),
    RepresentationKind.PROTEIN_SEQUENCE: _RepresentationArtifactSpec(
        "sequence",
        ".txt",
        "text/plain; charset=utf-8",
        SearchRepresentationArtifactEncoding.RAW_UTF8,
    ),
    RepresentationKind.RNA_SEQUENCE: _RepresentationArtifactSpec(
        "sequence",
        ".txt",
        "text/plain; charset=utf-8",
        SearchRepresentationArtifactEncoding.RAW_UTF8,
    ),
    RepresentationKind.PDB: _RepresentationArtifactSpec(
        "pdb", ".pdb", "chemical/x-pdb", SearchRepresentationArtifactEncoding.RAW_UTF8
    ),
    RepresentationKind.SDF: _RepresentationArtifactSpec(
        "sdf",
        ".sdf",
        "chemical/x-mdl-sdfile",
        SearchRepresentationArtifactEncoding.RAW_UTF8,
    ),
    RepresentationKind.POSCAR: _RepresentationArtifactSpec(
        "poscar",
        ".vasp",
        "text/plain; charset=utf-8",
        SearchRepresentationArtifactEncoding.RAW_UTF8,
    ),
    RepresentationKind.XYZ: _RepresentationArtifactSpec(
        "xyz", ".xyz", "chemical/x-xyz", SearchRepresentationArtifactEncoding.RAW_UTF8
    ),
    RepresentationKind.EXTXYZ: _RepresentationArtifactSpec(
        "extxyz",
        ".extxyz",
        "chemical/x-extxyz",
        SearchRepresentationArtifactEncoding.RAW_UTF8,
    ),
}
_JSON_REPRESENTATION_SPEC = _RepresentationArtifactSpec(
    "json",
    ".json",
    "application/vnd.discovery-os.candidate-representation+json",
    SearchRepresentationArtifactEncoding.CANONICAL_JSON,
)


class SearchRepresentationArtifactRef(StrictSchema):
    representation_index: int = Field(ge=0)
    kind: RepresentationKind
    encoding: SearchRepresentationArtifactEncoding
    artifact: ContentArtifactRef

    @model_validator(mode="after")
    def _profile_is_consistent(self) -> SearchRepresentationArtifactRef:
        spec = _representation_artifact_spec(self.kind)
        if str(self.encoding) != spec.encoding.value:
            raise ValueError("representation artifact encoding does not match its kind")
        if not self.artifact.relative_path.endswith(spec.extension):
            raise ValueError("representation artifact extension does not match its kind")
        if self.artifact.media_type != spec.media_type:
            raise ValueError("representation artifact media type does not match its kind")
        return self


class SearchCandidateRecord(StrictSchema):
    record_id: Identifier
    candidate: Candidate
    candidate_artifact: ContentArtifactRef
    representation_artifact_refs: list[SearchRepresentationArtifactRef] = Field(
        min_length=1
    )
    source_branch: ExplorationBranch
    round_index: int = Field(ge=0)
    evaluated_cycle: int = Field(ge=0)
    latent_state: UnifiedLatentStateRef
    evidence_ids: list[Identifier] = Field(min_length=1)
    generation_provenance: GeneratorProvenance | None = None
    generation_warnings: list[str] = Field(default_factory=list)
    generation_controls: GenerationControls
    report_warnings: list[str] = Field(default_factory=list)
    selection_eligible: bool = True
    exclusion_reasons: list[str] = Field(default_factory=list)
    structural_collapse_reasons: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def _record_is_consistent(self) -> SearchCandidateRecord:
        candidate_ref = self.candidate.candidate_ref
        if candidate_ref is None:
            raise ValueError("search candidate record requires candidate_ref")
        if candidate_content_hash(self.candidate) != candidate_ref.content_hash:
            raise ValueError("search candidate record has a stale candidate_ref")
        if len(self.representation_artifact_refs) != len(self.candidate.representations):
            raise ValueError("every candidate representation requires one raw artifact")
        indexes = [item.representation_index for item in self.representation_artifact_refs]
        if sorted(indexes) != list(range(len(self.candidate.representations))):
            raise ValueError("representation artifact indexes must cover the candidate exactly")
        for item in self.representation_artifact_refs:
            representation = self.candidate.representations[item.representation_index]
            if str(item.kind) != str(representation.kind):
                raise ValueError("representation artifact kind differs from the candidate")
            expected = _representation_artifact_ref(
                item.representation_index,
                representation,
            )
            if item != expected:
                raise ValueError("representation artifact reference failed content validation")
        if self.latent_state.candidate_ref != candidate_ref:
            raise ValueError("search candidate latent state belongs to another candidate")
        if self.latent_state.cycle != self.evaluated_cycle:
            raise ValueError("search candidate cycle does not match latent state")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate expert evidence ids are not allowed")
        if self.selection_eligible and self.exclusion_reasons:
            raise ValueError("eligible candidates cannot carry exclusion reasons")
        if self.structural_collapse_reasons:
            if self.selection_eligible:
                raise ValueError("structurally collapsed candidates cannot be selection eligible")
            if not set(self.structural_collapse_reasons).issubset(self.exclusion_reasons):
                raise ValueError(
                    "structural collapse reasons must be preserved as exclusion reasons"
                )
        return self


class SearchCycleRecord(StrictSchema):
    cycle_record_id: Identifier
    round_index: int = Field(ge=0)
    branch: ExplorationBranch
    evidence_branch_id: Identifier | None = None
    evidence_branch_kind: str | None = Field(default=None, max_length=128)
    evidence_claim_ids: list[Identifier] = Field(default_factory=list)
    evidence_generator_hints: dict[str, JsonValue] = Field(default_factory=dict)
    requested_cycle: int = Field(ge=0)
    parent_record_id: Identifier
    child_record_ids: list[Identifier] = Field(min_length=1)
    run_config: WorkspaceRunConfig
    controls: GenerationControls
    generation_provenance: GeneratorProvenance
    generation_warnings: list[str] = Field(default_factory=list)
    parent_evidence_ids: list[Identifier] = Field(min_length=1)
    child_evidence_ids: dict[str, list[Identifier]]
    iteration_artifact: ContentArtifactRef
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _cycle_is_consistent(self) -> SearchCycleRecord:
        if len(self.evidence_claim_ids) != len(set(self.evidence_claim_ids)):
            raise ValueError("duplicate evidence claim ids in search cycle")
        if self.evidence_branch_id is None and (
            self.evidence_branch_kind
            or self.evidence_claim_ids
            or self.evidence_generator_hints
        ):
            raise ValueError("evidence cycle payload requires evidence_branch_id")
        if self.evidence_branch_id is not None and not self.evidence_claim_ids:
            raise ValueError("evidence-guided cycle requires source claim ids")
        if self.run_config.workspace_mode != WorkspaceMode.ON:
            raise ValueError("fusion search cycles require workspace ON")
        if self.run_config.generation_controls != self.controls:
            raise ValueError("cycle controls differ from the generator run configuration")
        if len(self.child_record_ids) != len(set(self.child_record_ids)):
            raise ValueError("duplicate child record ids are not allowed")
        if len(self.parent_evidence_ids) != len(set(self.parent_evidence_ids)):
            raise ValueError("duplicate parent evidence ids are not allowed")
        if set(self.child_evidence_ids) != set(self.child_record_ids):
            raise ValueError("every child record requires its original expert evidence ids")
        return self


class SearchBranchFailurePayload(StrictSchema):
    """Immutable failure details persisted before search execution continues."""

    search_id: Identifier
    round_index: int = Field(ge=0)
    branch: ExplorationBranch
    requested_cycle: int = Field(ge=0)
    parent_candidate_ref: CandidateRef
    controls: GenerationControls
    cause_type: Identifier
    cause: NonEmptyText


class SearchBranchFailureRecord(SearchBranchFailurePayload):
    failure_id: Identifier
    failure_artifact: ContentArtifactRef

    @model_validator(mode="after")
    def _artifact_is_content_addressed(self) -> SearchBranchFailureRecord:
        payload = SearchBranchFailurePayload(
            search_id=self.search_id,
            round_index=self.round_index,
            branch=self.branch,
            requested_cycle=self.requested_cycle,
            parent_candidate_ref=self.parent_candidate_ref,
            controls=self.controls,
            cause_type=self.cause_type,
            cause=self.cause,
        )
        encoded = canonical_json(payload).encode("utf-8")
        digest = bytes_hash(encoded)
        expected_path = (
            f"fusion/search/{self.search_id}/failures/{digest[:2]}/{digest}.json"
        )
        expected_id = f"SFAIL-{digest[:32]}"
        artifact = self.failure_artifact
        if self.failure_id != expected_id or artifact.artifact_id != expected_id:
            raise ValueError("search branch failure id is not content-addressed")
        if (
            artifact.relative_path != expected_path
            or artifact.sha256 != digest
            or artifact.byte_size != len(encoded)
            or artifact.media_type
            != "application/vnd.discovery-os.fusion-search-failure+json"
        ):
            raise ValueError("search branch failure artifact failed content validation")
        return self


class SearchObservationContext(StrictSchema):
    search_id: Identifier
    round_index: int = Field(ge=0)
    branch: ExplorationBranch
    controls_used: GenerationControls
    selected_candidate_refs: list[CandidateRef]
    high_disagreement_candidate_refs: list[CandidateRef]
    completed_cycle_record_ids: list[Identifier]
    automatic_observation: SchedulerObservation | None = None


class SearchObservationProvider(Protocol):
    def __call__(
        self, context: SearchObservationContext
    ) -> SchedulerObservation | None: ...


class SearchControlPoint(StrictSchema):
    """One alpha/temperature operating point for a generator attempt."""

    alpha: float = Field(ge=0.0, le=1.0)
    temperature: float = Field(ge=0.01, le=5.0)
    label: Identifier


class SearchControlSweep(StrictSchema):
    """Bounded adaptive alpha/temperature sweep around scheduler controls.

    Each point is interpreted as an offset from the base run configuration and
    then translated around the branch scheduler's current controls.  This keeps
    the sweep responsive to prior-round outcomes instead of replaying one fixed
    grid forever.
    """

    points: list[SearchControlPoint] = Field(min_length=1, max_length=64)
    include_adaptive_center: bool = True
    max_variants_per_parent: int = Field(default=3, ge=1, le=64)

    @model_validator(mode="after")
    def _points_are_unique(self) -> SearchControlSweep:
        keys = [(item.alpha, item.temperature) for item in self.points]
        if len(keys) != len(set(keys)):
            raise ValueError("control sweep points must be unique")
        labels = [item.label for item in self.points]
        if len(labels) != len(set(labels)):
            raise ValueError("control sweep labels must be unique")
        return self


class SearchControlAttemptRecord(StrictSchema):
    attempt_id: Identifier
    round_index: int = Field(ge=0)
    branch: ExplorationBranch
    parent_candidate_ref: CandidateRef
    variant_index: int = Field(ge=0)
    controls: GenerationControls
    success: bool
    cycle_record_id: Identifier | None = None
    error_type: str | None = Field(default=None, max_length=256)
    error: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def _attempt_is_consistent(self) -> SearchControlAttemptRecord:
        if self.success:
            if self.cycle_record_id is None or self.error_type is not None or self.error is not None:
                raise ValueError("successful control attempts require a cycle and no error")
        elif self.cycle_record_id is not None or self.error_type is None or self.error is None:
            raise ValueError("failed control attempts require an error and no cycle")
        expected = "SATtempt-" + stable_hash(
            {
                "round_index": self.round_index,
                "branch": self.branch,
                "parent_candidate_ref": self.parent_candidate_ref,
                "variant_index": self.variant_index,
                "controls": self.controls,
                "success": self.success,
                "cycle_record_id": self.cycle_record_id,
                "error_type": self.error_type,
                "error": self.error,
            }
        )[:32]
        if self.attempt_id != expected:
            raise ValueError("control attempt id is not content-addressed")
        return self


class RankedSearchCandidate(StrictSchema):
    """Unified multi-candidate output without averaging incompatible physics."""

    rank: int = Field(gt=0)
    candidate_record_id: Identifier
    candidate_ref: CandidateRef
    candidate: Candidate
    priority_score: float = Field(ge=0.0, le=1.0)
    pareto_member: bool
    branch_ranks: dict[str, int]
    branch_scores: dict[str, float]
    expert_property_vectors: dict[str, list[DiagnosticProperty]]
    mean_reported_uncertainty: float | None = Field(default=None, ge=0.0)
    max_reported_uncertainty: float | None = Field(default=None, ge=0.0)
    expert_disagreement_score: float = Field(default=0.0, ge=0.0)
    source_branch: ExplorationBranch
    source_round: int = Field(ge=0)
    generation_controls: GenerationControls
    generation_warnings: list[str] = Field(default_factory=list)
    rationale: list[NonEmptyText] = Field(min_length=1)

    @model_validator(mode="after")
    def _candidate_identity_is_consistent(self) -> RankedSearchCandidate:
        if self.candidate.candidate_ref != self.candidate_ref:
            raise ValueError("ranked candidate payload and reference differ")
        if not self.branch_ranks:
            raise ValueError("ranked candidate requires at least one branch membership")
        if set(self.branch_ranks) != set(self.branch_scores):
            raise ValueError("ranked candidate branch rank/score keys differ")
        if any(rank <= 0 for rank in self.branch_ranks.values()):
            raise ValueError("ranked candidate branch ranks must be positive")
        return self


class SearchRoundRecord(StrictSchema):
    round_index: int = Field(ge=0)
    cycle_record_ids: list[Identifier] = Field(default_factory=list)
    candidate_record_ids: list[Identifier] = Field(default_factory=list)
    failure_record_ids: list[Identifier] = Field(default_factory=list)
    selection: ExplorationSelection
    branch_frontiers: dict[str, list[Identifier]]

    @model_validator(mode="after")
    def _round_ids_are_unique(self) -> SearchRoundRecord:
        if len(self.cycle_record_ids) != len(set(self.cycle_record_ids)):
            raise ValueError("duplicate cycle ids are not allowed in a round")
        if len(self.candidate_record_ids) != len(set(self.candidate_record_ids)):
            raise ValueError("duplicate candidate record ids are not allowed in a round")
        if len(self.failure_record_ids) != len(set(self.failure_record_ids)):
            raise ValueError("duplicate failure record ids are not allowed in a round")
        if not self.cycle_record_ids and not self.failure_record_ids:
            raise ValueError("a search round requires a completed cycle or a recorded failure")
        if set(self.branch_frontiers) != {item.value for item in ExplorationBranch}:
            raise ValueError("a search round must preserve every branch frontier")
        return self


class SearchBranchReport(StrictSchema):
    branch: ExplorationBranch
    pool_record_ids: list[Identifier]
    frontier_record_ids: list[Identifier]
    controls: GenerationControls
    scheduler_history: list[SchedulerDecision] = Field(default_factory=list)

    @model_validator(mode="after")
    def _branch_ids_are_unique(self) -> SearchBranchReport:
        if len(self.pool_record_ids) != len(set(self.pool_record_ids)):
            raise ValueError("duplicate branch pool record ids")
        if len(self.frontier_record_ids) != len(set(self.frontier_record_ids)):
            raise ValueError("duplicate branch frontier record ids")
        if not set(self.frontier_record_ids).issubset(self.pool_record_ids):
            raise ValueError("branch frontier must be contained in its candidate pool")
        return self


class FusionSearchReport(StrictSchema):
    search_id: Identifier
    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_run_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_generator_parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    rounds_requested: int = Field(gt=0)
    rounds_completed: int = Field(ge=0)
    status: FusionSearchStatus
    candidate_records: list[SearchCandidateRecord]
    cycle_records: list[SearchCycleRecord]
    failure_records: list[SearchBranchFailureRecord] = Field(default_factory=list)
    round_history: list[SearchRoundRecord]
    history_artifacts: list[ContentArtifactRef]
    branches: list[SearchBranchReport]
    final_selection: ExplorationSelection
    control_sweep: SearchControlSweep | None = None
    control_attempts: list[SearchControlAttemptRecord] = Field(default_factory=list)
    ranking_limit: int = Field(default=50, gt=0, le=1_024)
    ranked_candidates: list[RankedSearchCandidate] = Field(default_factory=list)
    ranking_method: Literal["weighted_reciprocal_rank_fusion_v1"] = (
        "weighted_reciprocal_rank_fusion_v1"
    )
    # Added after the first persisted search-report contract.  Omitted legacy
    # reports deterministically reconstruct the shortlist from their closed
    # final selection; newly written reports always persist it explicitly.
    validation_handoff_candidate_refs: list[CandidateRef] = Field(default_factory=list)
    scientific_claim: Literal["diagnostic_only"] = "diagnostic_only"

    @model_validator(mode="after")
    def _report_is_closed(self) -> FusionSearchReport:
        if self.rounds_completed != len(self.round_history):
            raise ValueError("round count does not match search history")
        record_ids = [item.record_id for item in self.candidate_records]
        cycle_ids = [item.cycle_record_id for item in self.cycle_records]
        failure_ids = [item.failure_id for item in self.failure_records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("duplicate candidate record ids in final search report")
        if len(cycle_ids) != len(set(cycle_ids)):
            raise ValueError("duplicate cycle record ids in final search report")
        if len(failure_ids) != len(set(failure_ids)):
            raise ValueError("duplicate failure record ids in final search report")
        if len(self.history_artifacts) != len(self.cycle_records):
            raise ValueError("every search cycle requires one persisted history artifact")
        branch_names = [str(item.branch) for item in self.branches]
        if set(branch_names) != {item.value for item in ExplorationBranch}:
            raise ValueError("final report must preserve all exploration branches")
        known = set(record_ids)
        if any(
            record_id not in known
            for branch in self.branches
            for record_id in branch.pool_record_ids + branch.frontier_record_ids
        ):
            raise ValueError("branch report cites an unknown candidate record")
        expected_handoff = _validation_handoff_refs(
            self.final_selection,
            self.candidate_records,
        )
        if "validation_handoff_candidate_refs" not in self.model_fields_set:
            object.__setattr__(
                self,
                "validation_handoff_candidate_refs",
                expected_handoff,
            )
        elif self.validation_handoff_candidate_refs != expected_handoff:
            raise ValueError(
                "validation handoff must be the exact de-duplicated Pareto/stability shortlist"
            )
        known_cycles = set(cycle_ids)
        known_failures = set(failure_ids)
        attempt_ids = [item.attempt_id for item in self.control_attempts]
        if len(attempt_ids) != len(set(attempt_ids)):
            raise ValueError("duplicate control attempt ids in final search report")
        for attempt in self.control_attempts:
            if attempt.success and attempt.cycle_record_id not in known_cycles:
                raise ValueError("successful control attempt cites an unknown cycle")
        expected_ranking = _ranked_candidate_results(
            self.final_selection,
            self.candidate_records,
            limit=self.ranking_limit,
        )
        if "ranked_candidates" not in self.model_fields_set:
            object.__setattr__(self, "ranked_candidates", expected_ranking)
        elif self.ranked_candidates != expected_ranking:
            raise ValueError("ranked candidate list must match the final selection")
        if [item.rank for item in self.ranked_candidates] != list(
            range(1, len(self.ranked_candidates) + 1)
        ):
            raise ValueError("ranked candidate positions must be contiguous")
        for round_record in self.round_history:
            if not set(round_record.cycle_record_ids).issubset(known_cycles):
                raise ValueError("round history cites an unknown search cycle")
            if not set(round_record.failure_record_ids).issubset(known_failures):
                raise ValueError("round history cites an unknown branch failure")
        if self.status == FusionSearchStatus.COMPLETED:
            if self.failure_records or self.rounds_completed != self.rounds_requested:
                raise ValueError("completed search cannot contain failures or missing rounds")
        elif self.status == FusionSearchStatus.EXHAUSTED:
            if self.failure_records or self.rounds_completed >= self.rounds_requested:
                raise ValueError("exhausted search must stop early without branch failures")
        elif self.status == FusionSearchStatus.PARTIAL:
            if not self.failure_records or not self.cycle_records:
                raise ValueError("partial search requires both successes and failures")
        elif self.status == FusionSearchStatus.FAILED:
            if not self.failure_records or self.cycle_records:
                raise ValueError("failed search requires failures and no completed cycles")
        return self


class PersistedFusionSearchReport(StrictSchema):
    report: FusionSearchReport
    report_artifact: ContentArtifactRef


@dataclass(frozen=True)
class _FrontierNode:
    candidate: Candidate
    previous_state: UnifiedLatentStateRef | None
    requested_cycle: int
    workspace_id: str | None
    origin_record: SearchCandidateRecord | None


class FusionSearchRunner:
    """Run batch Fusion iterations while keeping five independent branches."""

    def __init__(
        self,
        loop_runner: FusionLoopRunner,
        evidence_store: ExpertEvidenceStore,
        *,
        selector: DeterministicExplorationSelector | None = None,
        scheduler_factory: Callable[
            [GenerationControls], AdaptiveGenerationScheduler
        ] | None = None,
    ) -> None:
        if (
            loop_runner.runtime.artifact_store.root
            != evidence_store.artifact_store.root
        ):
            raise ValueError(
                "FusionLoopRunner and ExpertEvidenceStore must share one artifact root"
            )
        self.loop_runner = loop_runner
        self.evidence_store = evidence_store
        self.selector = selector or DeterministicExplorationSelector(evidence_store)
        self.scheduler_factory = scheduler_factory or (
            lambda controls: AdaptiveGenerationScheduler(controls)
        )

    def run(
        self,
        *,
        search_id: str,
        goal: DiscoveryGoal,
        initial_candidate: Candidate,
        base_run_config: WorkspaceRunConfig,
        rounds: int,
        initial_cycle: int = 0,
        initial_state: UnifiedLatentStateRef | None = None,
        expert_ids: Iterable[str] | None = None,
        required_primary_evaluator_ids: Iterable[str] | None = None,
        modality: ScientificModality | None = None,
        context_entities: Iterable[WorkspaceEntityInput] | None = None,
        relations: Iterable[WorkspaceRelation] | None = None,
        workspace_id: str | None = None,
        frontier_width: int = 4,
        observation_provider: SearchObservationProvider | None = None,
        evidence_policy: LiteratureEvidencePolicy | None = None,
        control_sweep: SearchControlSweep | None = None,
        ranking_limit: int = 50,
    ) -> PersistedFusionSearchReport:
        safe_search_id = self.loop_runner.runtime.artifact_store.safe_component(search_id)
        if safe_search_id != search_id:
            raise ValueError("search_id must already be a safe artifact path component")
        if isinstance(rounds, bool) or rounds <= 0:
            raise ValueError("rounds must be a positive integer")
        if isinstance(frontier_width, bool) or frontier_width <= 0:
            raise ValueError("frontier_width must be a positive integer")
        if isinstance(ranking_limit, bool) or not 1 <= ranking_limit <= 1_024:
            raise ValueError("ranking_limit must be between one and 1024")
        if control_sweep is not None:
            control_sweep = SearchControlSweep.model_validate_json(
                control_sweep.model_dump_json(), strict=True
            )
        goal = DiscoveryGoal.model_validate_json(goal.model_dump_json(), strict=True)
        initial_candidate = Candidate.model_validate_json(
            initial_candidate.model_dump_json(), strict=True
        )
        base_run_config = WorkspaceRunConfig.model_validate_json(
            base_run_config.model_dump_json(), strict=True
        )
        self._validate_initial(
            goal,
            initial_candidate,
            base_run_config,
            initial_cycle=initial_cycle,
            initial_state=initial_state,
        )
        if initial_state is not None:
            initial_cycle = initial_state.cycle + 1
            workspace_id = initial_state.workspace_id

        requested_experts = None if expert_ids is None else list(dict.fromkeys(expert_ids))
        required_evaluators = list(
            dict.fromkeys(
                required_primary_evaluator_ids
                if required_primary_evaluator_ids is not None
                else (requested_experts or [])
            )
        )
        contexts = list(context_entities or [])
        relation_rows = list(relations or [])
        branch_order = list(ExplorationBranch)
        schedulers = {
            branch: self.scheduler_factory(base_run_config.generation_controls)
            for branch in branch_order
        }
        frontiers: dict[ExplorationBranch, list[_FrontierNode]] = {
            branch: [] for branch in branch_order
        }
        frontiers[ExplorationBranch.PARETO] = [
            _FrontierNode(
                candidate=initial_candidate,
                previous_state=initial_state,
                requested_cycle=initial_cycle,
                workspace_id=workspace_id,
                origin_record=None,
            )
        ]
        branch_pool_records: dict[
            ExplorationBranch, dict[str, SearchCandidateRecord]
        ] = {
            branch: {} for branch in branch_order
        }
        candidate_records: dict[str, SearchCandidateRecord] = {}
        cycle_records: list[SearchCycleRecord] = []
        failure_records: list[SearchBranchFailureRecord] = []
        control_attempts: list[SearchControlAttemptRecord] = []
        history_artifacts: list[ContentArtifactRef] = []
        round_history: list[SearchRoundRecord] = []
        final_selection = self._empty_selection(f"{search_id}-unstarted")

        for round_index in range(rounds):
            round_cycles: list[SearchCycleRecord] = []
            round_failures: list[SearchBranchFailureRecord] = []
            generated_records: list[SearchCandidateRecord] = []
            current_by_ref: dict[str, SearchCandidateRecord] = {}
            branch_records_by_ref: dict[
                ExplorationBranch, dict[str, SearchCandidateRecord]
            ] = {branch: {} for branch in branch_order}
            failed_frontiers: dict[ExplorationBranch, list[_FrontierNode]] = {
                branch: [] for branch in branch_order
            }
            # Elitism is explicit: every branch selector sees the retained
            # historical frontiers as well as this round's new children.  A
            # candidate shared by branches is represented once, deterministically.
            for branch in branch_order:
                retained_records = [
                    *branch_pool_records[branch].values(),
                    *[
                        node.origin_record
                        for node in frontiers[branch]
                        if node.origin_record is not None
                    ],
                ]
                for record in sorted(
                    retained_records,
                    key=lambda item: (
                        _candidate_ref_key(item.candidate.candidate_ref),
                        item.record_id,
                    ),
                ):
                    if not record.selection_eligible:
                        continue
                    ref_key = _candidate_ref_key(record.candidate.candidate_ref)
                    branch_prior = branch_records_by_ref[branch].get(ref_key)
                    if branch_prior is None:
                        branch_records_by_ref[branch][ref_key] = record
                    elif branch_prior.candidate != record.candidate:
                        raise FusionSearchError(
                            "one retained branch reference resolved to different candidates"
                        )
                    elif (
                        record.evaluated_cycle,
                        record.round_index,
                        record.record_id,
                    ) > (
                        branch_prior.evaluated_cycle,
                        branch_prior.round_index,
                        branch_prior.record_id,
                    ):
                        branch_records_by_ref[branch][ref_key] = record
                    prior = current_by_ref.get(ref_key)
                    if prior is None:
                        current_by_ref[ref_key] = record
                    elif prior.candidate != record.candidate:
                        raise FusionSearchError(
                            "one retained candidate reference resolved to different candidates"
                        )
                    elif record.record_id < prior.record_id:
                        current_by_ref[ref_key] = record
            for branch in branch_order:
                nodes = sorted(
                    frontiers[branch],
                    key=lambda item: _candidate_ref_key(item.candidate.candidate_ref),
                )
                evidence_assignment = (
                    evidence_policy.select(
                        round_index=round_index,
                        exploration_branch=branch.value,
                    )
                    if evidence_policy is not None
                    else None
                )
                for node in nodes:
                    adaptive_controls = schedulers[branch].controls
                    control_variants = _control_variants(
                        base=base_run_config.generation_controls,
                        adaptive=adaptive_controls,
                        sweep=control_sweep,
                    )
                    successful_variant = False
                    pending_failures: list[tuple[int, GenerationControls, Exception]] = []
                    for variant_index, controls in enumerate(control_variants):
                        run_config = self._clone_config(
                            base_run_config,
                            parent=node.candidate.candidate_ref,
                            branch=branch,
                            round_index=round_index,
                            controls=controls,
                            control_variant_index=variant_index,
                            control_variant_count=len(control_variants),
                        )
                        decision_context = self._decision_context(
                            branch,
                            controls,
                            schedulers[branch].history,
                            evidence_assignment=evidence_assignment,
                        )
                        try:
                            iteration = self.loop_runner.iterate(
                                goal=goal,
                                parent_candidate=node.candidate,
                                cycle=node.requested_cycle,
                                run_config=run_config,
                                previous_state=node.previous_state,
                                expert_ids=requested_experts,
                                modality=modality,
                                context_entities=contexts,
                                relations=relation_rows,
                                workspace_id=node.workspace_id,
                                decision_context=decision_context,
                            )
                        except Exception as exc:
                            pending_failures.append((variant_index, controls, exc))
                            control_attempts.append(
                                _control_attempt_record(
                                    round_index=round_index,
                                    branch=branch,
                                    parent=node,
                                    variant_index=variant_index,
                                    controls=controls,
                                    success=False,
                                    cause=exc,
                                )
                            )
                            continue
                        cycle, parent_record, children = self._capture_iteration(
                            search_id=search_id,
                            round_index=round_index,
                            branch=branch,
                            parent=node,
                            run_config=run_config,
                            iteration=iteration,
                            decision_context=decision_context,
                        )
                        successful_variant = True
                        control_attempts.append(
                            _control_attempt_record(
                                round_index=round_index,
                                branch=branch,
                                parent=node,
                                variant_index=variant_index,
                                controls=controls,
                                success=True,
                                cycle_record_id=cycle.cycle_record_id,
                            )
                        )
                        candidate_records[parent_record.record_id] = parent_record
                        evaluated_records = [parent_record, *children]
                        for record in evaluated_records:
                            ref_key = _candidate_ref_key(record.candidate.candidate_ref)
                            branch_prior = branch_records_by_ref[branch].get(ref_key)
                            if branch_prior is None:
                                branch_records_by_ref[branch][ref_key] = record
                            elif branch_prior.candidate != record.candidate:
                                raise FusionSearchError(
                                    "one branch candidate reference resolved to different candidates"
                                )
                            elif record.selection_eligible and (
                                not branch_prior.selection_eligible
                                or (
                                    record.evaluated_cycle,
                                    record.round_index,
                                    record.record_id,
                                )
                                > (
                                    branch_prior.evaluated_cycle,
                                    branch_prior.round_index,
                                    branch_prior.record_id,
                                )
                            ):
                                branch_records_by_ref[branch][ref_key] = record

                            prior = current_by_ref.get(ref_key)
                            if prior is None:
                                current_by_ref[ref_key] = record
                            elif prior.candidate != record.candidate:
                                raise FusionSearchError(
                                    "one candidate reference resolved to different raw candidates"
                                )
                            elif (
                                record.selection_eligible
                                and not prior.selection_eligible
                            ) or (
                                record.selection_eligible == prior.selection_eligible
                                and (
                                    record.evaluated_cycle,
                                    record.round_index,
                                    record.record_id,
                                )
                                > (
                                    prior.evaluated_cycle,
                                    prior.round_index,
                                    prior.record_id,
                                )
                            ):
                                current_by_ref[ref_key] = record

                        for child in children:
                            candidate_records[child.record_id] = child
                            generated_records.append(child)
                        round_cycles.append(cycle)
                        cycle_records.append(cycle)
                        history_artifacts.append(
                            self._persist_object(
                                search_id,
                                "history",
                                cycle,
                                artifact_prefix="SCHIST",
                                media_type="application/vnd.discovery-os.fusion-search-cycle+json",
                            )
                        )
                    if not successful_variant:
                        failed_frontiers[branch].append(node)
                        for _variant_index, controls, exc in pending_failures:
                            failure = self._capture_failure(
                                search_id=search_id,
                                round_index=round_index,
                                branch=branch,
                                parent=node,
                                controls=controls,
                                cause=exc,
                            )
                            failure_records.append(failure)
                            round_failures.append(failure)
                        if evidence_policy is not None:
                            evidence_policy.observe(
                                round_index=round_index,
                                exploration_branch=branch.value,
                                objective_improvement=None,
                                structural_collapse_rate=1.0,
                                failed=True,
                            )

            unique_current = sorted(
                current_by_ref.values(),
                key=lambda item: _candidate_ref_key(item.candidate.candidate_ref),
            )
            preeligible = [item for item in unique_current if item.selection_eligible]
            if preeligible:
                pool = CandidatePool(
                    pool_id=f"{search_id}-round-{round_index}",
                    entries=[
                        CandidatePoolEntry(
                            candidate=item.candidate,
                            evidence_ids=item.evidence_ids,
                        )
                        for item in preeligible
                    ],
                    required_evaluator_ids=required_evaluators,
                )
                selection = self.selector.select(
                    pool,
                    goal,
                    limit_per_branch=frontier_width,
                )
            else:
                selection = self._empty_selection(
                    f"{search_id}-round-{round_index}",
                    excluded=[
                        ExcludedCandidate(
                            candidate_ref=item.candidate.candidate_ref,
                            reasons=item.exclusion_reasons
                            or ["candidate was not eligible for expert-vector selection"],
                        )
                        for item in unique_current
                    ],
                )
            final_selection = selection
            excluded_by_ref = {
                _candidate_ref_key(item.candidate_ref): item.reasons
                for item in selection.excluded_candidates
            }
            for key, reasons in excluded_by_ref.items():
                record = current_by_ref.get(key)
                if record is None:
                    continue
                updated = record.model_copy(
                    update={
                        "selection_eligible": False,
                        "exclusion_reasons": list(reasons),
                    }
                )
                updated = SearchCandidateRecord.model_validate_json(
                    updated.model_dump_json(), strict=True
                )
                current_by_ref[key] = updated
                candidate_records[updated.record_id] = updated
                for branch in branch_order:
                    # A fail-closed scientific identity must not re-enter from
                    # an older retained pool record in the next round.
                    branch_pool_records[branch].pop(key, None)

            next_frontiers: dict[ExplorationBranch, list[_FrontierNode]] = {
                branch: [] for branch in branch_order
            }
            for result in selection.branches:
                branch = ExplorationBranch(str(result.branch))
                for selected in result.candidates:
                    ref_key = _candidate_ref_key(selected.candidate_ref)
                    record = branch_records_by_ref[branch].get(ref_key)
                    if record is None:
                        record = current_by_ref.get(ref_key)
                    if record is None or not record.selection_eligible:
                        raise FusionSearchError(
                            "selector cited an unavailable or fail-closed candidate"
                        )
                    next_frontiers[branch].append(
                        _FrontierNode(
                            candidate=record.candidate,
                            previous_state=record.latent_state,
                            requested_cycle=record.latent_state.cycle + 1,
                            workspace_id=record.latent_state.workspace_id,
                            origin_record=record,
                        )
                    )

            branch_frontier_ids: dict[str, list[str]] = {}
            for branch in branch_order:
                # A failed worker must not erase its last safe parent.  Failed
                # parents take deterministic priority, then selector-ranked
                # elites fill the remaining bounded frontier slots.
                retained_after_failure: list[_FrontierNode] = []
                for node in sorted(
                    failed_frontiers[branch],
                    key=lambda item: _candidate_ref_key(item.candidate.candidate_ref),
                ):
                    if node.origin_record is None:
                        retained_after_failure.append(node)
                        continue
                    ref_key = _candidate_ref_key(node.candidate.candidate_ref)
                    record = branch_records_by_ref[branch].get(
                        ref_key, node.origin_record
                    )
                    if (
                        record.selection_eligible
                        and ref_key not in excluded_by_ref
                    ):
                        retained_after_failure.append(
                            _FrontierNode(
                                candidate=record.candidate,
                                previous_state=record.latent_state,
                                requested_cycle=record.latent_state.cycle + 1,
                                workspace_id=record.latent_state.workspace_id,
                                origin_record=record,
                            )
                        )
                bounded: list[_FrontierNode] = []
                seen_refs: set[str] = set()
                for node in [*retained_after_failure, *next_frontiers[branch]]:
                    ref_key = _candidate_ref_key(node.candidate.candidate_ref)
                    if ref_key in seen_refs:
                        continue
                    seen_refs.add(ref_key)
                    bounded.append(node)
                    if len(bounded) == frontier_width:
                        break
                next_frontiers[branch] = bounded
                frontier_ids = [
                    item.origin_record.record_id
                    for item in bounded
                    if item.origin_record is not None
                ]
                for node in bounded:
                    record = node.origin_record
                    if record is None:
                        continue
                    ref_key = _candidate_ref_key(record.candidate.candidate_ref)
                    branch_pool_records[branch][ref_key] = record
                branch_frontier_ids[branch.value] = list(frontier_ids)

            disagreement_refs = [
                item.candidate_ref
                for result in selection.branches
                if str(result.branch) == ExplorationBranch.EXPERT_DISAGREEMENT.value
                for item in result.candidates
            ]
            for branch in branch_order:
                automatic = self._automatic_observation(
                    branch=branch,
                    round_cycles=round_cycles,
                    candidate_records=candidate_records,
                    selector_exclusions=excluded_by_ref,
                    goal=goal,
                    required_evaluator_ids=required_evaluators,
                    high_disagreement_candidates=disagreement_refs,
                )
                context = SearchObservationContext(
                    search_id=search_id,
                    round_index=round_index,
                    branch=branch,
                    controls_used=schedulers[branch].controls,
                    selected_candidate_refs=[
                        item.candidate.candidate_ref for item in next_frontiers[branch]
                    ],
                    high_disagreement_candidate_refs=disagreement_refs,
                    completed_cycle_record_ids=[
                        item.cycle_record_id
                        for item in round_cycles
                        if str(item.branch) == branch.value
                    ],
                    automatic_observation=automatic,
                )
                supplied = (
                    observation_provider(context)
                    if observation_provider is not None
                    else None
                )
                if supplied is None and automatic is None:
                    continue
                selected = automatic
                if supplied is not None:
                    selected = SchedulerObservation.model_validate_json(
                        supplied.model_dump_json(), strict=True
                    )
                assert selected is not None
                merged_disagreement = _unique_candidate_refs(
                    [
                        *disagreement_refs,
                        *(
                            automatic.high_disagreement_candidates
                            if automatic is not None
                            else []
                        ),
                        *selected.high_disagreement_candidates,
                    ]
                )
                observed = SchedulerObservation(
                    objective_improvement=selected.objective_improvement,
                    structural_collapse_rate=selected.structural_collapse_rate,
                    high_disagreement_candidates=merged_disagreement,
                )
                schedulers[branch].observe(observed)
                if evidence_policy is not None:
                    evidence_policy.observe(
                        round_index=round_index,
                        exploration_branch=branch.value,
                        objective_improvement=observed.objective_improvement,
                        structural_collapse_rate=observed.structural_collapse_rate,
                        failed=False,
                    )

            frontiers = next_frontiers
            round_record = SearchRoundRecord(
                round_index=round_index,
                cycle_record_ids=[item.cycle_record_id for item in round_cycles],
                candidate_record_ids=sorted(
                    {item.record_id for item in generated_records}
                ),
                failure_record_ids=[item.failure_id for item in round_failures],
                selection=selection,
                branch_frontiers=branch_frontier_ids,
            )
            round_history.append(round_record)
            if not any(frontiers.values()):
                break

        if failure_records:
            status = (
                FusionSearchStatus.PARTIAL
                if cycle_records
                else FusionSearchStatus.FAILED
            )
        elif len(round_history) < rounds:
            status = FusionSearchStatus.EXHAUSTED
        else:
            status = FusionSearchStatus.COMPLETED

        branch_reports = [
            SearchBranchReport(
                branch=branch,
                pool_record_ids=[
                    record.record_id
                    for _ref_key, record in sorted(
                        branch_pool_records[branch].items()
                    )
                ],
                frontier_record_ids=[
                    item.origin_record.record_id
                    for item in frontiers[branch]
                    if item.origin_record is not None
                ],
                controls=schedulers[branch].controls,
                scheduler_history=schedulers[branch].history,
            )
            for branch in branch_order
        ]
        report = FusionSearchReport(
            search_id=search_id,
            goal_hash=stable_hash(goal),
            base_run_config_hash=stable_hash(base_run_config),
            base_generator_parameters_hash=base_run_config.generator_parameters_hash,
            rounds_requested=rounds,
            rounds_completed=len(round_history),
            status=status,
            candidate_records=sorted(
                candidate_records.values(), key=lambda item: item.record_id
            ),
            cycle_records=cycle_records,
            failure_records=failure_records,
            round_history=round_history,
            history_artifacts=history_artifacts,
            branches=branch_reports,
            final_selection=final_selection,
            control_sweep=control_sweep,
            control_attempts=control_attempts,
            ranking_limit=ranking_limit,
            ranked_candidates=_ranked_candidate_results(
                final_selection,
                candidate_records.values(),
                limit=ranking_limit,
            ),
            validation_handoff_candidate_refs=_validation_handoff_refs(
                final_selection,
                candidate_records.values(),
            ),
        )
        report_artifact = self._persist_object(
            search_id,
            "reports",
            report,
            artifact_prefix="SREPORT",
            media_type="application/vnd.discovery-os.fusion-search-report+json",
        )
        return PersistedFusionSearchReport(
            report=report,
            report_artifact=report_artifact,
        )

    @staticmethod
    def _decision_context(
        branch: ExplorationBranch,
        controls: GenerationControls,
        history: list[SchedulerDecision],
        *,
        evidence_assignment: EvidenceBranchAssignment | None = None,
    ) -> FusionDecisionContext:
        """Bind the controls in use to the last completed branch observation.

        A search round cannot claim observations from candidates that have not
        been generated and evaluated yet.  The first branch step therefore
        carries ``None`` for improvement and the schema's neutral zero collapse
        rate; later steps carry only that branch's most recent observation.
        """

        previous = history[-1].observation if history else None
        return FusionDecisionContext(
            guidance_alpha=controls.alpha,
            exploration_branch=branch.value,
            previous_objective_improvement=(
                previous.objective_improvement if previous is not None else None
            ),
            structural_collapse_rate=(
                previous.structural_collapse_rate if previous is not None else 0.0
            ),
            evidence_branch_id=(
                evidence_assignment.branch_id if evidence_assignment is not None else None
            ),
            evidence_branch_kind=(
                str(evidence_assignment.branch_kind)
                if evidence_assignment is not None
                else None
            ),
            evidence_claim_ids=(
                list(evidence_assignment.source_claim_ids)
                if evidence_assignment is not None
                else []
            ),
            evidence_generator_hints=(
                dict(evidence_assignment.generator_hints)
                if evidence_assignment is not None
                else {}
            ),
            evidence_rationale=(
                evidence_assignment.rationale if evidence_assignment is not None else None
            ),
        )

    def _automatic_observation(
        self,
        *,
        branch: ExplorationBranch,
        round_cycles: list[SearchCycleRecord],
        candidate_records: dict[str, SearchCandidateRecord],
        selector_exclusions: dict[str, list[str]],
        goal: DiscoveryGoal,
        required_evaluator_ids: list[str],
        high_disagreement_candidates: list[CandidateRef],
    ) -> SchedulerObservation | None:
        """Derive one branch observation without normalized or averaged scores.

        Objective movement is deliberately ordinal.  ``+1`` means that the
        current raw expert-utility set strictly Pareto-covers the parent set;
        ``-1`` means the reverse.  Mixed trade-offs, missing evidence, changed
        dimensions, and incompatible units all fail closed to ``0``.  The
        scheduler only needs the sign/trend, so this avoids adding quantities
        with unrelated scientific units.
        """

        cycles = [item for item in round_cycles if item.branch == branch]
        if not cycles:
            return None
        child_records = [
            candidate_records[record_id]
            for cycle in cycles
            for record_id in cycle.child_record_ids
        ]
        if not child_records:
            return None
        parent_records = [
            candidate_records[cycle.parent_record_id] for cycle in cycles
        ]
        collapsed = [item for item in child_records if item.structural_collapse_reasons]
        current_records = [
            item
            for item in child_records
            if item.selection_eligible
            and _candidate_ref_key(item.candidate.candidate_ref)
            not in selector_exclusions
        ]
        previous_records = [item for item in parent_records if item.selection_eligible]
        improvement = self._raw_objective_improvement(
            previous_records,
            current_records,
            goal=goal,
            required_evaluator_ids=required_evaluator_ids,
        )
        return SchedulerObservation(
            objective_improvement=improvement,
            structural_collapse_rate=len(collapsed) / len(child_records),
            high_disagreement_candidates=_unique_candidate_refs(
                high_disagreement_candidates
            ),
        )

    def _raw_objective_improvement(
        self,
        previous_records: list[SearchCandidateRecord],
        current_records: list[SearchCandidateRecord],
        *,
        goal: DiscoveryGoal,
        required_evaluator_ids: list[str],
    ) -> float:
        if not previous_records or not current_records:
            return 0.0
        previous = [
            self._raw_utility_vector(
                item,
                goal=goal,
                required_evaluator_ids=required_evaluator_ids,
            )
            for item in previous_records
        ]
        current = [
            self._raw_utility_vector(
                item,
                goal=goal,
                required_evaluator_ids=required_evaluator_ids,
            )
            for item in current_records
        ]
        if any(item is None for item in [*previous, *current]):
            return 0.0
        prior_vectors = [item for item in previous if item is not None]
        current_vectors = [item for item in current if item is not None]
        dimensions = set(prior_vectors[0])
        if not dimensions or any(
            set(item) != dimensions for item in [*prior_vectors, *current_vectors]
        ):
            return 0.0
        current_covers_previous = _set_pareto_covers(
            current_vectors, prior_vectors, dimensions
        )
        previous_covers_current = _set_pareto_covers(
            prior_vectors, current_vectors, dimensions
        )
        if current_covers_previous and not previous_covers_current:
            return 1.0
        if previous_covers_current and not current_covers_previous:
            return -1.0
        return 0.0

    def _raw_utility_vector(
        self,
        record: SearchCandidateRecord,
        *,
        goal: DiscoveryGoal,
        required_evaluator_ids: list[str],
    ) -> dict[tuple[str, str, str | None], float] | None:
        objectives = {item.property_name: item for item in goal.objectives}
        if len(objectives) != len(goal.objectives):
            return None
        values: dict[tuple[str, str, str | None], float] = {}
        evaluator_ids: set[str] = set()
        returned_properties: set[str] = set()
        candidate_ref = record.candidate.candidate_ref
        for evidence_id in sorted(record.evidence_ids):
            try:
                envelope = self.evidence_store.load(
                    self.evidence_store.get(evidence_id)
                )
            except Exception:
                return None
            payload = envelope.payload
            if (
                payload.candidate_ref != candidate_ref
                or payload.status != FeatureStatus.SUCCESS
                or any(item.out_of_domain for item in payload.properties)
            ):
                return None
            evaluator_ids.add(payload.expert_id)
            for prop in payload.properties:
                objective = objectives.get(prop.property_name)
                if objective is None:
                    continue
                if objective.unit is not None and prop.unit != objective.unit:
                    return None
                dimension = (payload.expert_id, prop.property_name, prop.unit)
                if dimension in values:
                    return None
                utility = _raw_goal_utility(prop.value, objective)
                if utility is None:
                    return None
                values[dimension] = utility
                returned_properties.add(prop.property_name)
        if not set(required_evaluator_ids).issubset(evaluator_ids):
            return None
        required_properties = {
            item.property_name for item in goal.objectives if item.required
        }
        if not required_properties.issubset(returned_properties):
            return None
        return values or None

    @staticmethod
    def _validate_initial(
        goal: DiscoveryGoal,
        candidate: Candidate,
        config: WorkspaceRunConfig,
        *,
        initial_cycle: int,
        initial_state: UnifiedLatentStateRef | None,
    ) -> None:
        if isinstance(initial_cycle, bool) or initial_cycle < 0:
            raise ValueError("initial_cycle must be non-negative")
        candidate_ref = candidate.candidate_ref
        if candidate_ref is None or candidate_content_hash(candidate) != candidate_ref.content_hash:
            raise FusionSearchError("initial candidate requires a current immutable reference")
        if config.workspace_mode != WorkspaceMode.ON:
            raise FusionSearchError("fusion search requires a workspace ON run configuration")
        if config.goal_hash != stable_hash(goal):
            raise FusionSearchError("base run configuration belongs to another goal")
        if config.parent_candidate_ref != candidate_ref:
            raise FusionSearchError("base run configuration belongs to another parent")
        if initial_state is not None:
            if (
                initial_state.candidate_ref != candidate_ref
                or initial_state.goal_hash != stable_hash(goal)
                or initial_state.seed != config.seed
            ):
                raise FusionSearchError("initial latent state is unrelated to the search")

    def _capture_iteration(
        self,
        *,
        search_id: str,
        round_index: int,
        branch: ExplorationBranch,
        parent: _FrontierNode,
        run_config: WorkspaceRunConfig,
        iteration: FusionBatchIterationReport,
        decision_context: FusionDecisionContext,
    ) -> tuple[SearchCycleRecord, SearchCandidateRecord, list[SearchCandidateRecord]]:
        before = iteration.before_revision
        if before.latent_state is None:
            raise FusionSearchError("search iteration has no before latent state")
        parent_evidence, parent_reasons = self._ingest_primary(before)
        parent_record = self._candidate_record(
            search_id=search_id,
            candidate=parent.candidate,
            branch=branch,
            round_index=round_index,
            state=before.latent_state,
            evidence_ids=parent_evidence,
            controls=run_config.generation_controls,
            generation_provenance=(
                parent.origin_record.generation_provenance
                if parent.origin_record is not None
                else None
            ),
            generation_warnings=(
                parent.origin_record.generation_warnings
                if parent.origin_record is not None
                else []
            ),
            report_warnings=before.warnings,
            exclusion_reasons=parent_reasons,
        )
        children: list[SearchCandidateRecord] = []
        child_evidence: dict[str, list[str]] = {}
        for candidate, report in zip(
            iteration.generation.generated_candidates,
            iteration.after_revisions,
            strict=True,
        ):
            if report.latent_state is None:
                raise FusionSearchError("generated candidate has no latent state")
            evidence_ids, reasons = self._ingest_primary(report)
            if parent_reasons:
                reasons = [
                    "parent evaluation failed closed: " + "; ".join(parent_reasons),
                    *reasons,
                ]
            child = self._candidate_record(
                search_id=search_id,
                candidate=candidate,
                branch=branch,
                round_index=round_index,
                state=report.latent_state,
                evidence_ids=evidence_ids,
                controls=run_config.generation_controls,
                generation_provenance=iteration.generation.provenance,
                generation_warnings=iteration.generation.warnings,
                report_warnings=report.warnings,
                exclusion_reasons=reasons,
            )
            children.append(child)
            child_evidence[child.record_id] = child.evidence_ids

        iteration_artifact = self._persist_object(
            search_id,
            "iterations",
            iteration,
            artifact_prefix="SITER",
            media_type="application/vnd.discovery-os.fusion-search-iteration+json",
        )
        cycle_id = "SCYCLE-" + stable_hash(
            {
                "search_id": search_id,
                "round": round_index,
                "branch": branch,
                "parent": parent_record.record_id,
                "children": [item.record_id for item in children],
                "run_config": run_config,
                "iteration": iteration_artifact.sha256,
                "evidence_branch_id": decision_context.evidence_branch_id,
                "evidence_claim_ids": decision_context.evidence_claim_ids,
                "evidence_generator_hints": decision_context.evidence_generator_hints,
            }
        )[:32]
        cycle = SearchCycleRecord(
            cycle_record_id=cycle_id,
            round_index=round_index,
            branch=branch,
            evidence_branch_id=decision_context.evidence_branch_id,
            evidence_branch_kind=decision_context.evidence_branch_kind,
            evidence_claim_ids=decision_context.evidence_claim_ids,
            evidence_generator_hints=decision_context.evidence_generator_hints,
            requested_cycle=parent.requested_cycle,
            parent_record_id=parent_record.record_id,
            child_record_ids=[item.record_id for item in children],
            run_config=run_config,
            controls=run_config.generation_controls,
            generation_provenance=iteration.generation.provenance,
            generation_warnings=iteration.generation.warnings,
            parent_evidence_ids=parent_evidence,
            child_evidence_ids=child_evidence,
            iteration_artifact=iteration_artifact,
            warnings=before.warnings + [row for item in iteration.after_revisions for row in item.warnings],
        )
        return cycle, parent_record, children

    def _ingest_primary(
        self,
        report,
    ) -> tuple[list[str], list[str]]:
        primary_id = report.workspace.primary_entity_id
        primary_refs = [
            item for item in report.feature_refs if item.workspace_entity_id == primary_id
        ]
        reasons = [
            *[f"missing requested expert {item!r}" for item in report.missing_expert_ids],
            *[f"failed expert {item!r}" for item in report.failed_expert_ids],
        ]
        if not primary_refs:
            reasons.append("primary candidate has no persisted expert feature payload")
        evidence_ids: list[str] = []
        for ref in primary_refs:
            try:
                raw = self.loop_runner.runtime.artifact_store.read_bytes(
                    ref.artifact.relative_path,
                    expected_sha256=ref.artifact.sha256,
                )
                if len(raw) != ref.artifact.byte_size:
                    raise ValueError("feature artifact byte size changed")
                payload = ExpertFeaturePayload.model_validate_json(raw, strict=True)
                if (
                    payload.workspace_entity_id != primary_id
                    or payload.candidate_ref != report.candidate_ref
                ):
                    raise ValueError("primary feature payload cites another entity")
                stored = self.evidence_store.put(payload, ref)
            except Exception as exc:
                raise FusionSearchError(
                    f"primary expert artifact failed verification: {type(exc).__name__}: {exc}"
                ) from exc
            evidence_ids.append(stored.evidence_id)
            if payload.status != FeatureStatus.SUCCESS:
                reasons.append(
                    f"primary evaluator {payload.expert_id!r} status is {payload.status!s}"
                )
            if any(item.out_of_domain for item in payload.properties):
                reasons.append(
                    f"primary evaluator {payload.expert_id!r} returned out-of-domain data"
                )
        return sorted(set(evidence_ids)), sorted(set(reasons))

    def _candidate_record(
        self,
        *,
        search_id: str,
        candidate: Candidate,
        branch: ExplorationBranch,
        round_index: int,
        state: UnifiedLatentStateRef,
        evidence_ids: list[str],
        controls: GenerationControls,
        generation_provenance: GeneratorProvenance | None,
        generation_warnings: list[str],
        report_warnings: list[str],
        exclusion_reasons: list[str],
    ) -> SearchCandidateRecord:
        if not evidence_ids:
            raise FusionSearchError("candidate has no original primary expert evidence")
        candidate_artifact = self._persist_object(
            search_id,
            "candidates",
            candidate,
            artifact_prefix="SCAND",
            media_type="application/vnd.discovery-os.raw-candidate+json",
        )
        representation_artifact_refs = self._persist_representations(candidate)
        record_id = "SCREC-" + stable_hash(
            {
                "candidate": candidate.candidate_ref,
                "state_id": state.state_id,
                "branch": branch,
                "round": round_index,
                "evidence_ids": evidence_ids,
                "generation_controls": controls,
            }
        )[:32]
        structural_collapse_reasons = _explicit_structural_collapse_reasons(
            [*generation_warnings, *report_warnings, *exclusion_reasons]
        )
        all_exclusion_reasons = sorted(
            set([*exclusion_reasons, *structural_collapse_reasons])
        )
        return SearchCandidateRecord(
            record_id=record_id,
            candidate=candidate,
            candidate_artifact=candidate_artifact,
            representation_artifact_refs=representation_artifact_refs,
            source_branch=branch,
            round_index=round_index,
            evaluated_cycle=state.cycle,
            latent_state=state,
            evidence_ids=evidence_ids,
            generation_provenance=generation_provenance,
            generation_warnings=generation_warnings,
            generation_controls=controls,
            report_warnings=report_warnings,
            selection_eligible=not all_exclusion_reasons,
            exclusion_reasons=all_exclusion_reasons,
            structural_collapse_reasons=structural_collapse_reasons,
        )

    def _capture_failure(
        self,
        *,
        search_id: str,
        round_index: int,
        branch: ExplorationBranch,
        parent: _FrontierNode,
        controls: GenerationControls,
        cause: Exception,
    ) -> SearchBranchFailureRecord:
        payload = SearchBranchFailurePayload(
            search_id=search_id,
            round_index=round_index,
            branch=branch,
            requested_cycle=parent.requested_cycle,
            parent_candidate_ref=self._validate_parent(
                parent.candidate.candidate_ref
            ),
            controls=controls,
            cause_type=type(cause).__name__,
            cause=_safe_failure_cause(cause),
        )
        artifact = self._persist_object(
            search_id,
            "failures",
            payload,
            artifact_prefix="SFAIL",
            media_type="application/vnd.discovery-os.fusion-search-failure+json",
        )
        return SearchBranchFailureRecord(
            **payload.model_dump(mode="python"),
            failure_id=artifact.artifact_id,
            failure_artifact=artifact,
        )

    def _persist_representations(
        self,
        candidate: Candidate,
    ) -> list[SearchRepresentationArtifactRef]:
        stored: list[SearchRepresentationArtifactRef] = []
        for index, representation in enumerate(candidate.representations):
            encoded, _spec = _representation_artifact_payload(representation)
            reference = _representation_artifact_ref(index, representation)
            written, digest = self.loop_runner.runtime.artifact_store.write_bytes(
                reference.artifact.relative_path,
                encoded,
            )
            if (
                written != reference.artifact.relative_path
                or digest != reference.artifact.sha256
            ):
                raise FusionSearchError(
                    "artifact store changed a candidate representation artifact"
                )
            stored.append(reference)
        return stored

    @staticmethod
    def _validate_parent(parent: CandidateRef | None) -> CandidateRef:
        if parent is None:
            raise FusionSearchError("search frontier candidate has no immutable reference")
        return parent

    def _clone_config(
        self,
        base: WorkspaceRunConfig,
        *,
        parent: CandidateRef | None,
        branch: ExplorationBranch,
        round_index: int,
        controls: GenerationControls,
        control_variant_index: int = 0,
        control_variant_count: int = 1,
    ) -> WorkspaceRunConfig:
        parent = self._validate_parent(parent)
        pair_key = f"{base.pair_key}-{branch.value}"
        if control_variant_count > 1:
            pair_key = f"{pair_key}-control-{control_variant_index}"
        if len(pair_key) > 256:
            pair_key = f"SEARCH-{stable_hash(pair_key)[:48]}"
        payload = base.model_dump(mode="json")
        payload.update(
            {
                "parent_candidate_ref": parent.model_dump(mode="json"),
                "pair_key": pair_key,
                "cohort_index": round_index,
                "generator_seed": _generator_sampling_seed(
                    base_seed=base.seed,
                    round_index=round_index,
                    branch=branch,
                    parent=parent,
                    control_variant_index=control_variant_index,
                    control_variant_count=control_variant_count,
                ),
                "generation_controls": controls.model_dump(mode="json"),
                "generator_parameters_hash": stable_hash(
                    {
                        "base_parameters_hash": base.generator_parameters_hash,
                        "generation_controls": controls,
                    }
                ),
            }
        )
        return WorkspaceRunConfig.model_validate_json(
            canonical_json(payload), strict=True
        )

    def _persist_object(
        self,
        search_id: str,
        category: str,
        value,
        *,
        artifact_prefix: str,
        media_type: str,
    ) -> ContentArtifactRef:
        encoded = canonical_json(value).encode("utf-8")
        digest = bytes_hash(encoded)
        relative_path = (
            f"fusion/search/{search_id}/{category}/{digest[:2]}/{digest}.json"
        )
        written, written_digest = self.loop_runner.runtime.artifact_store.write_bytes(
            relative_path,
            encoded,
        )
        if written_digest != digest:
            raise FusionSearchError("artifact store changed a search object digest")
        return ContentArtifactRef(
            artifact_id=f"{artifact_prefix}-{digest[:32]}",
            relative_path=written,
            sha256=digest,
            media_type=media_type,
            byte_size=len(encoded),
        )

    @staticmethod
    def _empty_selection(
        pool_id: str,
        *,
        excluded: list[ExcludedCandidate] | None = None,
    ) -> ExplorationSelection:
        return ExplorationSelection(
            pool_id=pool_id,
            property_dimensions=[],
            branches=[
                ExplorationBranchResult(branch=branch, candidates=[])
                for branch in ExplorationBranch
            ],
            excluded_candidates=excluded or [],
        )


def _control_variants(
    *,
    base: GenerationControls,
    adaptive: GenerationControls,
    sweep: SearchControlSweep | None,
) -> list[GenerationControls]:
    if sweep is None:
        return [
            GenerationControls.model_validate_json(
                adaptive.model_dump_json(), strict=True
            )
        ]
    variants: list[GenerationControls] = []
    seen: set[tuple[float, float, float, float]] = set()

    def add(alpha: float, temperature: float, label: str) -> None:
        key = (
            round(max(0.0, min(1.0, alpha)), 12),
            round(max(0.01, min(5.0, temperature)), 12),
            round(adaptive.mutation_strength, 12),
            round(adaptive.diversity_strength, 12),
        )
        if key in seen or len(variants) >= sweep.max_variants_per_parent:
            return
        seen.add(key)
        variants.append(
            GenerationControls(
                alpha=key[0],
                temperature=key[1],
                mutation_strength=adaptive.mutation_strength,
                diversity_strength=adaptive.diversity_strength,
                schedule_step=adaptive.schedule_step,
                decision_reason=(
                    f"{adaptive.decision_reason}; alpha/temperature sweep {label} "
                    f"(alpha={key[0]:.6g}, temperature={key[1]:.6g})"
                ),
            )
        )

    if sweep.include_adaptive_center:
        add(adaptive.alpha, adaptive.temperature, "adaptive-center")
    for point in sweep.points:
        add(
            adaptive.alpha + (point.alpha - base.alpha),
            adaptive.temperature + (point.temperature - base.temperature),
            point.label,
        )
    if not variants:
        raise FusionSearchError("control sweep produced no valid alpha/temperature variants")
    return variants


def _control_attempt_record(
    *,
    round_index: int,
    branch: ExplorationBranch,
    parent: _FrontierNode,
    variant_index: int,
    controls: GenerationControls,
    success: bool,
    cycle_record_id: str | None = None,
    cause: Exception | None = None,
) -> SearchControlAttemptRecord:
    parent_ref = FusionSearchRunner._validate_parent(parent.candidate.candidate_ref)
    error_type = None if cause is None else type(cause).__name__
    error = None if cause is None else _safe_failure_cause(cause)
    payload = {
        "round_index": round_index,
        "branch": branch,
        "parent_candidate_ref": parent_ref,
        "variant_index": variant_index,
        "controls": controls,
        "success": success,
        "cycle_record_id": cycle_record_id,
        "error_type": error_type,
        "error": error,
    }
    return SearchControlAttemptRecord(
        attempt_id="SATtempt-" + stable_hash(payload)[:32],
        **payload,
    )


_RANK_BRANCH_WEIGHTS = {
    ExplorationBranch.PARETO.value: 1.25,
    ExplorationBranch.STABILITY.value: 1.0,
    ExplorationBranch.TARGET_PROPERTY.value: 1.0,
    ExplorationBranch.NOVELTY.value: 0.45,
    ExplorationBranch.EXPERT_DISAGREEMENT.value: 0.25,
}


def _ranked_candidate_results(
    selection: ExplorationSelection,
    records: Iterable[SearchCandidateRecord],
    *,
    limit: int,
) -> list[RankedSearchCandidate]:
    latest_by_ref: dict[str, SearchCandidateRecord] = {}
    for record in records:
        if not record.selection_eligible:
            continue
        key = _candidate_ref_key(record.candidate.candidate_ref)
        prior = latest_by_ref.get(key)
        if prior is None:
            latest_by_ref[key] = record
        elif prior.candidate != record.candidate:
            raise FusionSearchError("one ranked candidate reference resolved to different payloads")
        elif (
            record.evaluated_cycle,
            record.round_index,
            record.record_id,
        ) > (
            prior.evaluated_cycle,
            prior.round_index,
            prior.record_id,
        ):
            latest_by_ref[key] = record

    branch_ranks: dict[str, dict[str, int]] = {}
    branch_scores: dict[str, dict[str, float]] = {}
    expert_vectors: dict[str, dict[str, list[DiagnosticProperty]]] = {}
    for branch_result in selection.branches:
        branch_name = str(branch_result.branch)
        for rank, selected in enumerate(branch_result.candidates, start=1):
            key = _candidate_ref_key(selected.candidate_ref)
            if key not in latest_by_ref:
                continue
            branch_ranks.setdefault(key, {})[branch_name] = rank
            branch_scores.setdefault(key, {})[branch_name] = selected.score
            prior_vectors = expert_vectors.get(key)
            if prior_vectors is None:
                expert_vectors[key] = selected.expert_property_vectors
            elif canonical_json(prior_vectors) != canonical_json(
                selected.expert_property_vectors
            ):
                raise FusionSearchError(
                    "branch rankings disagree on one candidate's original expert vector"
                )

    reciprocal_k = 20.0
    maximum = fsum(
        weight / (reciprocal_k + 1.0)
        for weight in _RANK_BRANCH_WEIGHTS.values()
    )
    rows = []
    for key, ranks in branch_ranks.items():
        record = latest_by_ref[key]
        raw = fsum(
            _RANK_BRANCH_WEIGHTS[name] / (reciprocal_k + rank)
            for name, rank in ranks.items()
        )
        priority = 0.0 if maximum <= 0.0 else min(1.0, raw / maximum)
        vectors = expert_vectors[key]
        uncertainties = [
            prop.uncertainty
            for properties in vectors.values()
            for prop in properties
            if prop.uncertainty is not None
        ]
        branch_score_map = branch_scores[key]
        rows.append(
            {
                "key": key,
                "record": record,
                "priority": priority,
                "pareto": ExplorationBranch.PARETO.value in ranks,
                "ranks": ranks,
                "scores": branch_score_map,
                "vectors": vectors,
                "mean_uncertainty": (
                    fsum(uncertainties) / len(uncertainties) if uncertainties else None
                ),
                "max_uncertainty": max(uncertainties) if uncertainties else None,
                "disagreement": max(
                    0.0,
                    branch_score_map.get(
                        ExplorationBranch.EXPERT_DISAGREEMENT.value, 0.0
                    ),
                ),
            }
        )
    rows.sort(
        key=lambda item: (
            -item["priority"],
            not item["pareto"],
            item["ranks"].get(ExplorationBranch.PARETO.value, 10**9),
            item["ranks"].get(ExplorationBranch.STABILITY.value, 10**9),
            item["ranks"].get(ExplorationBranch.TARGET_PROPERTY.value, 10**9),
            item["record"].candidate.candidate_id,
            item["record"].candidate.candidate_ref.version,
            item["record"].candidate.candidate_ref.content_hash,
        )
    )
    output: list[RankedSearchCandidate] = []
    for rank, item in enumerate(rows[:limit], start=1):
        record = item["record"]
        branch_names = sorted(
            item["ranks"],
            key=lambda name: (
                item["ranks"][name],
                name,
            ),
        )
        rationale = [
            "Unified priority uses weighted reciprocal branch ranks; raw scientific values are not averaged.",
            "Candidate entered branches: " + ", ".join(branch_names),
        ]
        if item["pareto"]:
            rationale.append("Candidate is non-dominated in the final expert objective panel.")
        if item["disagreement"] > 0.0:
            rationale.append(
                "Cross-expert disagreement is preserved as a follow-up signal, not a quality bonus."
            )
        output.append(
            RankedSearchCandidate(
                rank=rank,
                candidate_record_id=record.record_id,
                candidate_ref=record.candidate.candidate_ref,
                candidate=record.candidate,
                priority_score=item["priority"],
                pareto_member=item["pareto"],
                branch_ranks=item["ranks"],
                branch_scores=item["scores"],
                expert_property_vectors=item["vectors"],
                mean_reported_uncertainty=item["mean_uncertainty"],
                max_reported_uncertainty=item["max_uncertainty"],
                expert_disagreement_score=item["disagreement"],
                source_branch=record.source_branch,
                source_round=record.round_index,
                generation_controls=record.generation_controls,
                generation_warnings=record.generation_warnings,
                rationale=rationale,
            )
        )
    return output


def _candidate_ref_key(ref: CandidateRef | None) -> str:
    if ref is None:
        raise FusionSearchError("candidate is missing its immutable reference")
    return stable_hash(ref)


def _validation_handoff_refs(
    selection: ExplorationSelection,
    records: Iterable[SearchCandidateRecord],
) -> list[CandidateRef]:
    """Return a bounded, exact-content-de-duplicated validation shortlist.

    Pareto candidates take priority, followed by stability candidates.  This
    is only a handoff boundary for a separately configured high-cost validator;
    it does not claim that DFT, relaxation, phonons, or experiments ran.  The
    de-duplication is exact over scientific representation content and ignores
    batch output filenames.  Symmetry/tolerance-aware crystal matching remains
    the responsibility of a structure-standardization connector.
    """

    by_ref: dict[str, Candidate] = {}
    for record in records:
        key = _candidate_ref_key(record.candidate.candidate_ref)
        prior = by_ref.get(key)
        if prior is not None and prior != record.candidate:
            raise ValueError(
                "one handoff candidate reference resolved to different candidates"
            )
        by_ref[key] = record.candidate

    selected_by_branch = {
        ExplorationBranch(str(result.branch)): result for result in selection.branches
    }
    refs: list[CandidateRef] = []
    seen_scientific_content: set[str] = set()
    for branch in (ExplorationBranch.PARETO, ExplorationBranch.STABILITY):
        result = selected_by_branch.get(branch)
        if result is None:
            continue
        for selected in result.candidates:
            candidate = by_ref.get(_candidate_ref_key(selected.candidate_ref))
            if candidate is None:
                raise ValueError(
                    "validation handoff selection cites an unknown candidate"
                )
            scientific_content = stable_hash(
                {
                    "candidate_type": candidate.candidate_type,
                    "domain": candidate.domain,
                    "representations": [
                        _scientific_representation(item)
                        for item in candidate.representations
                    ],
                }
            )
            if scientific_content in seen_scientific_content:
                continue
            seen_scientific_content.add(scientific_content)
            refs.append(selected.candidate_ref)
    return refs


def _scientific_representation(
    representation: CandidateRepresentation,
) -> dict[str, object]:
    payload = representation.model_dump(mode="json")
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("source_entry", None)
    payload["metadata"] = metadata
    return payload


def _generator_sampling_seed(
    *,
    base_seed: int,
    round_index: int,
    branch: ExplorationBranch,
    parent: CandidateRef,
    control_variant_index: int = 0,
    control_variant_count: int = 1,
) -> int:
    """Derive a reproducible 32-bit generator seed without changing expert seeds."""

    payload = {
        "contract": (
            "fusion-search-generator-seed-v1"
            if control_variant_count == 1
            else "fusion-search-generator-seed-v2-control-sweep"
        ),
        "base_seed": base_seed,
        "round_index": round_index,
        "branch": branch,
        "parent_candidate_ref": parent,
    }
    if control_variant_count > 1:
        payload["control_variant_index"] = control_variant_index
        payload["control_variant_count"] = control_variant_count
    digest = stable_hash(payload)
    return int(digest[:8], 16)


def _safe_failure_cause(cause: Exception) -> str:
    message = " ".join(str(cause).split())
    if not message:
        message = "exception did not provide a message"
    return message[:4_000]


def _explicit_structural_collapse_reasons(reasons: Iterable[str]) -> list[str]:
    """Return only machine-readable structural-collapse signals.

    Expert failures, out-of-domain results, and selector exclusions are not
    structural collapse. Generators/evaluators must emit the explicit
    ``structural_collapse:`` prefix for the scheduler to count the event.
    """

    return sorted(
        {
            reason
            for reason in reasons
            if reason.strip().casefold().startswith("structural_collapse:")
        }
    )


def _unique_candidate_refs(refs: Iterable[CandidateRef]) -> list[CandidateRef]:
    by_key = {_candidate_ref_key(item): item for item in refs}
    return [by_key[key] for key in sorted(by_key)]


def _raw_goal_utility(value: float, objective: PropertyObjective) -> float | None:
    """Map one raw value to its goal ordering without population normalization."""

    if objective.direction == ObjectiveDirection.MAXIMIZE:
        return value
    if objective.direction == ObjectiveDirection.MINIMIZE:
        return -value
    if objective.direction == ObjectiveDirection.TARGET:
        target = objective.target_value
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            return None
        return -abs(value - float(target))
    if objective.direction == ObjectiveDirection.RANGE:
        if objective.lower_bound is None or objective.upper_bound is None:
            return None
        if value < objective.lower_bound:
            return -(objective.lower_bound - value)
        if value > objective.upper_bound:
            return -(value - objective.upper_bound)
        return 0.0
    return None


def _weakly_dominates(
    left: dict[tuple[str, str, str | None], float],
    right: dict[tuple[str, str, str | None], float],
    dimensions: set[tuple[str, str, str | None]],
) -> bool:
    return all(left[item] >= right[item] for item in dimensions)


def _set_pareto_covers(
    left: list[dict[tuple[str, str, str | None], float]],
    right: list[dict[tuple[str, str, str | None], float]],
    dimensions: set[tuple[str, str, str | None]],
) -> bool:
    return all(
        any(_weakly_dominates(candidate, baseline, dimensions) for candidate in left)
        for baseline in right
    )


def _representation_artifact_spec(
    kind: RepresentationKind | str,
) -> _RepresentationArtifactSpec:
    normalized = RepresentationKind(str(kind))
    return _RAW_REPRESENTATION_SPECS.get(normalized, _JSON_REPRESENTATION_SPEC)


def _representation_artifact_payload(
    representation: CandidateRepresentation,
) -> tuple[bytes, _RepresentationArtifactSpec]:
    spec = _representation_artifact_spec(representation.kind)
    if spec.encoding == SearchRepresentationArtifactEncoding.RAW_UTF8:
        encoded = representation.value.encode("utf-8")
    else:
        encoded = canonical_json(representation).encode("utf-8")
    return encoded, spec


def _representation_artifact_ref(
    index: int,
    representation: CandidateRepresentation,
) -> SearchRepresentationArtifactRef:
    encoded, spec = _representation_artifact_payload(representation)
    digest = bytes_hash(encoded)
    relative_path = (
        "fusion/search/representations/"
        f"{spec.profile}/{digest[:2]}/{digest}{spec.extension}"
    )
    identity = stable_hash(
        {
            "sha256": digest,
            "profile": spec.profile,
            "extension": spec.extension,
            "media_type": spec.media_type,
        }
    )
    return SearchRepresentationArtifactRef(
        representation_index=index,
        kind=representation.kind,
        encoding=spec.encoding,
        artifact=ContentArtifactRef(
            artifact_id=f"SREP-{identity[:32]}",
            relative_path=relative_path,
            sha256=digest,
            media_type=spec.media_type,
            byte_size=len(encoded),
        ),
    )


__all__ = [
    "FusionSearchError",
    "FusionSearchReport",
    "FusionSearchRunner",
    "FusionSearchStatus",
    "PersistedFusionSearchReport",
    "SearchBranchReport",
    "SearchBranchFailurePayload",
    "SearchBranchFailureRecord",
    "SearchCandidateRecord",
    "SearchCycleRecord",
    "SearchObservationContext",
    "SearchObservationProvider",
    "SearchRepresentationArtifactEncoding",
    "SearchRepresentationArtifactRef",
    "SearchRoundRecord",
]
