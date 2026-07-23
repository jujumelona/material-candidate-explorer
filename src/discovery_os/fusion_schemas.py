"""Strict wire contracts for expert features and the scientific fusion layer.

Fusion records are diagnostic state.  They are deliberately separate from
``EvidenceRecord`` so that an embedding, latent score, or workspace ablation
can never be mistaken for scientific validation.
"""

from __future__ import annotations

import math
from ._compat import StrEnum
from pathlib import PurePosixPath, PureWindowsPath
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .hashing import candidate_content_hash, stable_hash
from .schemas import (
    Candidate,
    CandidateRef,
    CandidateType,
    DiscoveryGoal,
    Identifier,
    NonEmptyText,
    Probability,
    RepresentationKind,
    StrictSchema,
)


Sha256 = str


class ScientificModality(StrEnum):
    CRYSTAL_MATERIAL = "crystal_material"
    MOLECULE_2D = "molecule_2d"
    MOLECULE_3D = "molecule_3d"
    PROTEIN_SEQUENCE = "protein_sequence"
    PROTEIN_STRUCTURE = "protein_structure"
    RNA_SEQUENCE = "rna_sequence"
    RNA_STRUCTURE = "rna_structure"
    CELL_STATE = "cell_state"
    ELECTRONIC_STRUCTURE = "electronic_structure"


class WorkspaceMode(StrEnum):
    OFF = "off"
    ON = "on"


class FeatureStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class TensorDType(StrEnum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"


class TensorRole(StrEnum):
    GLOBAL_EMBEDDING = "global_embedding"
    TOKEN_EMBEDDING = "token_embedding"
    ATOM_EMBEDDING = "atom_embedding"
    CELL_EMBEDDING = "cell_embedding"
    HAMILTONIAN = "hamiltonian"
    CUSTOM = "custom"


class NumericTensor(StrictSchema):
    """Bounded JSON tensor used only on the local/HTTP wire.

    The runtime immediately stores it as a content-addressed artifact.  It is
    not written into discovery checkpoints or evidence records.
    """

    dtype: TensorDType = TensorDType.FLOAT32
    shape: list[int] = Field(min_length=1, max_length=8)
    values: list[float] = Field(min_length=1, max_length=65_536)

    @model_validator(mode="after")
    def _shape_matches_values(self) -> NumericTensor:
        if any(item <= 0 for item in self.shape):
            raise ValueError("tensor dimensions must be positive")
        if math.prod(self.shape) != len(self.values):
            raise ValueError("tensor shape does not match the number of values")
        return self


class FeatureSemantics(StrictSchema):
    """Scientific meaning required to interpret a specialist tensor."""

    tensor_role: TensorRole
    projection_id: Identifier
    entity_type: str | None = Field(default=None, max_length=128)
    entity_ids: list[str] = Field(default_factory=list, max_length=65_536)
    mask: list[bool] = Field(default_factory=list, max_length=65_536)
    pooling: Literal["none", "mean", "sum", "cls", "attention", "custom"] = "none"
    normalization: str = Field(min_length=1, max_length=512)
    coordinate_frame: str | None = Field(default=None, max_length=512)
    basis: str | None = Field(default=None, max_length=1_024)
    unit_semantics: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _entity_metadata_is_consistent(self) -> FeatureSemantics:
        if self.mask and not self.entity_ids:
            raise ValueError("feature mask requires entity_ids")
        if self.mask and len(self.mask) != len(self.entity_ids):
            raise ValueError("feature mask must match entity_ids")
        if len(self.entity_ids) != len(set(self.entity_ids)):
            raise ValueError("duplicate feature entity_ids are not allowed")
        return self


class ExpertRoute(StrictSchema):
    modality: ScientificModality
    feature_space: Identifier
    representation_kinds: list[RepresentationKind] = Field(min_length=1)
    candidate_types: list[CandidateType] = Field(default_factory=list)

    @model_validator(mode="after")
    def _route_lists_are_unique(self) -> ExpertRoute:
        if len(self.representation_kinds) != len(set(self.representation_kinds)):
            raise ValueError("duplicate route representation kinds are not allowed")
        if len(self.candidate_types) != len(set(self.candidate_types)):
            raise ValueError("duplicate route candidate types are not allowed")
        return self


class ContentArtifactRef(StrictSchema):
    artifact_id: Identifier
    relative_path: NonEmptyText
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    media_type: str = Field(min_length=1, max_length=256)
    byte_size: int = Field(ge=0)

    @model_validator(mode="after")
    def _path_is_confined(self) -> ContentArtifactRef:
        posix = PurePosixPath(self.relative_path)
        windows = PureWindowsPath(self.relative_path)
        if posix.is_absolute() or windows.is_absolute():
            raise ValueError("artifact path must be relative")
        if ".." in posix.parts or ".." in windows.parts:
            raise ValueError("artifact path cannot traverse parent directories")
        if "\\" in self.relative_path:
            raise ValueError("artifact path must use portable forward slashes")
        return self


class ExpertProvenance(StrictSchema):
    expert_id: Identifier
    adapter_version: Identifier
    model_version: Identifier
    code_revision: Identifier
    weight_revision: Identifier
    dataset_revision: str | None = Field(default=None, max_length=512)
    projection_version: str | None = Field(default=None, max_length=256)
    parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    device: str | None = Field(default=None, max_length=256)
    seed: int | None = Field(default=None, ge=0)


class DiagnosticProperty(StrictSchema):
    property_name: Identifier
    value: float
    unit: str | None = Field(default=None, max_length=128)
    uncertainty: float | None = Field(default=None, ge=0.0)
    out_of_domain: bool = False
    source: str | None = Field(default=None, max_length=512)


class ExpertDescriptor(StrictSchema):
    expert_id: Identifier
    display_name: NonEmptyText
    adapter_version: Identifier
    modalities: list[ScientificModality] = Field(min_length=1)
    supported_candidate_types: list[CandidateType] = Field(min_length=1)
    supported_representations: list[RepresentationKind] = Field(min_length=1)
    feature_spaces: list[Identifier] = Field(min_length=1)
    routes: list[ExpertRoute] = Field(default_factory=list)
    api_protocol: Literal["expert-feature-v1"] = "expert-feature-v1"
    available: bool = True
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _descriptor_lists_are_unique(self) -> ExpertDescriptor:
        for label, values in (
            ("modalities", self.modalities),
            ("supported_candidate_types", self.supported_candidate_types),
            ("supported_representations", self.supported_representations),
            ("feature_spaces", self.feature_spaces),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} are not allowed")
        route_keys: set[tuple[str, str, tuple[str, ...], tuple[str, ...]]] = set()
        for route in self.routes:
            key = (
                str(route.modality),
                route.feature_space,
                tuple(sorted(str(item) for item in route.representation_kinds)),
                tuple(sorted(str(item) for item in route.candidate_types)),
            )
            if key in route_keys:
                raise ValueError("duplicate expert routes are not allowed")
            route_keys.add(key)
            if route.modality not in self.modalities:
                raise ValueError("expert route modality must be declared")
            if route.feature_space not in self.feature_spaces:
                raise ValueError("expert route feature space must be declared")
            if not set(route.representation_kinds).issubset(self.supported_representations):
                raise ValueError("expert route representations must be declared")
            if route.candidate_types and not set(route.candidate_types).issubset(
                self.supported_candidate_types
            ):
                raise ValueError("expert route candidate types must be declared")
        return self


class ExpertFeatureRequest(StrictSchema):
    workspace_entity_id: Identifier
    candidate: Candidate
    goal: DiscoveryGoal
    modality: ScientificModality
    feature_space: Identifier
    cycle: int = Field(ge=0)
    seed: int = Field(ge=0)

    @model_validator(mode="after")
    def _candidate_is_immutable(self) -> ExpertFeatureRequest:
        if self.candidate.candidate_ref is None:
            raise ValueError("expert feature extraction requires candidate_ref")
        return self


class ExpertFeaturePayload(StrictSchema):
    workspace_entity_id: Identifier
    candidate_ref: CandidateRef
    expert_id: Identifier
    modality: ScientificModality
    feature_space: Identifier
    status: FeatureStatus = FeatureStatus.SUCCESS
    tensor: NumericTensor | None = None
    semantics: FeatureSemantics | None = None
    properties: list[DiagnosticProperty] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: ExpertProvenance

    @model_validator(mode="after")
    def _successful_feature_has_tensor(self) -> ExpertFeaturePayload:
        if self.status == FeatureStatus.SUCCESS and self.tensor is None:
            raise ValueError("successful feature payload requires a tensor")
        if self.status == FeatureStatus.SUCCESS and self.semantics is None:
            raise ValueError("successful feature payload requires tensor semantics")
        if self.status == FeatureStatus.FAILED and self.tensor is not None:
            raise ValueError("failed feature payload cannot contain a tensor")
        if self.status == FeatureStatus.FAILED and self.properties:
            raise ValueError("failed feature payload cannot contain diagnostic properties")
        if self.tensor is not None and self.semantics is None:
            raise ValueError("tensor payloads require tensor semantics")
        if self.tensor is not None and self.semantics is not None:
            if self.semantics.entity_ids:
                if self.tensor.shape[0] != len(self.semantics.entity_ids):
                    raise ValueError("tensor entity axis does not match entity_ids")
                if self.semantics.mask and len(self.semantics.mask) != len(
                    self.semantics.entity_ids
                ):
                    raise ValueError("feature mask must match entity_ids")
        names = [item.property_name for item in self.properties]
        if len(names) != len(set(names)):
            raise ValueError("duplicate diagnostic properties are not allowed")
        return self


class ExpertFeatureRef(StrictSchema):
    feature_id: Identifier
    workspace_entity_id: Identifier
    candidate_ref: CandidateRef
    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    expert_id: Identifier
    modality: ScientificModality
    feature_space: Identifier
    status: FeatureStatus
    artifact: ContentArtifactRef
    tensor_dtype: TensorDType | None = None
    tensor_shape: list[int] = Field(default_factory=list, max_length=8)
    semantics: FeatureSemantics | None = None
    properties: list[DiagnosticProperty] = Field(default_factory=list)
    quality_flags: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: ExpertProvenance


class FusionFeatureInput(StrictSchema):
    feature_id: Identifier
    workspace_entity_id: Identifier
    payload: ExpertFeaturePayload

    @model_validator(mode="after")
    def _entity_binding_matches(self) -> FusionFeatureInput:
        if self.workspace_entity_id != self.payload.workspace_entity_id:
            raise ValueError("feature input entity id does not match payload")
        return self


class WorkspaceEntityRole(StrEnum):
    PRIMARY_CANDIDATE = "primary_candidate"
    TARGET = "target"
    CONTEXT = "context"
    ENVIRONMENT = "environment"
    ASSAY = "assay"


class WorkspaceEntity(StrictSchema):
    entity_id: Identifier
    role: WorkspaceEntityRole
    candidate_ref: CandidateRef


class WorkspaceEntityInput(StrictSchema):
    entity_id: Identifier
    role: WorkspaceEntityRole
    candidate: Candidate

    @model_validator(mode="after")
    def _entity_has_ref(self) -> WorkspaceEntityInput:
        if self.candidate.candidate_ref is None:
            raise ValueError("workspace entities require candidate_ref")
        return self


class WorkspaceRelation(StrictSchema):
    relation_id: Identifier
    subject_entity_id: Identifier
    predicate: Literal[
        "binds_to",
        "interacts_with",
        "evaluated_in",
        "derived_from",
        "contains",
        "custom",
    ]
    object_entity_id: Identifier
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class ScientificWorkspace(StrictSchema):
    workspace_id: Identifier
    primary_entity_id: Identifier
    entities: list[WorkspaceEntity] = Field(min_length=1)
    relations: list[WorkspaceRelation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _workspace_graph_is_valid(self) -> ScientificWorkspace:
        entity_ids = [item.entity_id for item in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("duplicate workspace entity ids are not allowed")
        by_id = {item.entity_id: item for item in self.entities}
        if self.primary_entity_id not in by_id:
            raise ValueError("primary_entity_id is not present in workspace")
        if by_id[self.primary_entity_id].role != WorkspaceEntityRole.PRIMARY_CANDIDATE:
            raise ValueError("primary entity must have primary_candidate role")
        if sum(item.role == WorkspaceEntityRole.PRIMARY_CANDIDATE for item in self.entities) != 1:
            raise ValueError("workspace must contain exactly one primary candidate")
        relation_ids = [item.relation_id for item in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("duplicate workspace relation ids are not allowed")
        for relation in self.relations:
            if relation.subject_entity_id not in by_id or relation.object_entity_id not in by_id:
                raise ValueError("workspace relation cites an unknown entity")
        return self


class FusionDecisionContext(StrictSchema):
    """Runtime observations supplied to a deterministic fusion controller.

    These values are search-control metadata, not learned features or
    scientific predictions.  Failed/missing expert identifiers remain
    runtime-owned fields on :class:`FusionRequest` so callers cannot forge the
    evaluator outcome.
    """

    guidance_alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    previous_objective_improvement: float | None = None
    structural_collapse_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    exploration_branch: Literal[
        "stability",
        "target_property",
        "novelty",
        "expert_disagreement",
        "pareto",
    ] | None = None
    evidence_branch_id: Identifier | None = None
    evidence_branch_kind: str | None = Field(default=None, max_length=128)
    evidence_claim_ids: list[Identifier] = Field(default_factory=list)
    evidence_generator_hints: dict[str, JsonValue] = Field(default_factory=dict)
    evidence_rationale: str | None = Field(default=None, max_length=20_000)

    @model_validator(mode="after")
    def _evidence_context_is_closed(self) -> "FusionDecisionContext":
        if len(self.evidence_claim_ids) != len(set(self.evidence_claim_ids)):
            raise ValueError("duplicate evidence claim ids are not allowed")
        has_branch_payload = bool(
            self.evidence_branch_kind
            or self.evidence_claim_ids
            or self.evidence_generator_hints
            or self.evidence_rationale
        )
        if has_branch_payload and self.evidence_branch_id is None:
            raise ValueError("evidence branch payload requires evidence_branch_id")
        if self.evidence_branch_id is not None and not self.evidence_claim_ids:
            raise ValueError("evidence branch requires at least one source claim id")
        return self


class FusionRequest(StrictSchema):
    goal: DiscoveryGoal
    candidate_ref: CandidateRef
    workspace: ScientificWorkspace
    workspace_mode: Literal["on"] = "on"
    cycle: int = Field(ge=0)
    seed: int = Field(ge=0)
    features: list[FusionFeatureInput] = Field(min_length=1)
    decision_context: FusionDecisionContext = Field(
        default_factory=FusionDecisionContext
    )
    failed_expert_ids: list[Identifier] = Field(default_factory=list)
    missing_expert_ids: list[Identifier] = Field(default_factory=list)
    previous_latent: NumericTensor | None = None
    previous_state_id: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def _features_match_candidate(self) -> FusionRequest:
        ids = [item.feature_id for item in self.features]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate feature_id values are not allowed")
        if len(self.failed_expert_ids) != len(set(self.failed_expert_ids)):
            raise ValueError("duplicate failed expert ids are not allowed")
        if len(self.missing_expert_ids) != len(set(self.missing_expert_ids)):
            raise ValueError("duplicate missing expert ids are not allowed")
        if set(self.failed_expert_ids) & set(self.missing_expert_ids):
            raise ValueError("failed and missing expert ids must be disjoint")
        primary = next(
            item for item in self.workspace.entities if item.entity_id == self.workspace.primary_entity_id
        )
        if primary.candidate_ref != self.candidate_ref:
            raise ValueError("workspace primary candidate does not match request candidate")
        successful_primary_experts = {
            item.payload.expert_id
            for item in self.features
            if item.workspace_entity_id == self.workspace.primary_entity_id
            and item.payload.status == FeatureStatus.SUCCESS
        }
        if successful_primary_experts & (
            set(self.failed_expert_ids) | set(self.missing_expert_ids)
        ):
            raise ValueError(
                "successful primary experts cannot also be failed or missing"
            )
        for feature in self.features:
            entity = next(
                (
                    item
                    for item in self.workspace.entities
                    if item.entity_id == feature.workspace_entity_id
                ),
                None,
            )
            if entity is None or feature.payload.candidate_ref != entity.candidate_ref:
                raise ValueError("every feature must reference its exact workspace entity")
            if feature.payload.status == FeatureStatus.FAILED:
                raise ValueError("failed expert features cannot enter fusion")
            if feature.payload.tensor is None:
                raise ValueError("fusion features require a tensor")
        if (self.previous_latent is None) != (self.previous_state_id is None):
            raise ValueError("previous_latent and previous_state_id must be supplied together")
        return self


class FusionOutput(StrictSchema):
    latent: NumericTensor
    used_feature_ids: list[Identifier] = Field(min_length=1)
    ignored_feature_ids: list[Identifier] = Field(default_factory=list)
    alignment_scores: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    backend_id: Identifier
    backend_version: Identifier
    code_revision: Identifier
    weight_revision: Identifier

    @model_validator(mode="after")
    def _feature_sets_are_disjoint(self) -> FusionOutput:
        if set(self.used_feature_ids) & set(self.ignored_feature_ids):
            raise ValueError("used and ignored feature ids must be disjoint")
        if len(self.used_feature_ids) != len(set(self.used_feature_ids)):
            raise ValueError("duplicate used_feature_ids are not allowed")
        if len(self.ignored_feature_ids) != len(set(self.ignored_feature_ids)):
            raise ValueError("duplicate ignored_feature_ids are not allowed")
        return self


class UnifiedLatentStateRecord(StrictSchema):
    """Content-addressed durable metadata for one latent state."""

    state_version: int = Field(gt=0)
    candidate_ref: CandidateRef
    workspace_id: Identifier
    workspace_entities: list[WorkspaceEntity] = Field(min_length=1)
    workspace_relations: list[WorkspaceRelation] = Field(default_factory=list)
    workspace_mode: Literal["on"] = "on"
    cycle: int = Field(ge=0)
    latent_artifact: ContentArtifactRef
    latent_content_hash: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    dtype: TensorDType
    shape: list[int] = Field(min_length=1, max_length=8)
    source_feature_ids: list[Identifier] = Field(min_length=1)
    previous_state_id: str | None = Field(default=None, max_length=256)
    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)
    backend_id: Identifier
    backend_version: Identifier
    code_revision: Identifier
    weight_revision: Identifier
    warnings: list[str] = Field(default_factory=list)


class UnifiedLatentStateRef(UnifiedLatentStateRecord):
    state_id: Identifier
    # Optional only for decoding legacy/test fixtures.  Runtime-produced or
    # runtime-consumed states are required to carry and verify this artifact.
    state_artifact: ContentArtifactRef | None = None

    @model_validator(mode="after")
    def _state_workspace_is_consistent(self) -> UnifiedLatentStateRef:
        if len(self.source_feature_ids) != len(set(self.source_feature_ids)):
            raise ValueError("duplicate source_feature_ids are not allowed")
        entity_ids = [item.entity_id for item in self.workspace_entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("duplicate latent workspace entity ids are not allowed")
        relation_ids = [item.relation_id for item in self.workspace_relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("duplicate latent workspace relation ids are not allowed")
        for relation in self.workspace_relations:
            if (
                relation.subject_entity_id not in entity_ids
                or relation.object_entity_id not in entity_ids
            ):
                raise ValueError("latent workspace relation cites an unknown entity")
        primary_matches = [
            item
            for item in self.workspace_entities
            if item.role == WorkspaceEntityRole.PRIMARY_CANDIDATE
        ]
        if len(primary_matches) != 1 or primary_matches[0].candidate_ref != self.candidate_ref:
            raise ValueError("latent state requires one matching primary workspace entity")
        return self


class ChangeAxis(StrEnum):
    ELEMENT_DISTRIBUTION = "element_distribution"
    COORDINATES_3D = "coordinates_3d"
    LATTICE = "lattice"
    MOLECULAR_STRUCTURE = "molecular_structure"
    PROTEIN_SEQUENCE = "protein_sequence"
    RNA_SEQUENCE = "rna_sequence"
    CELL_STATE = "cell_state"
    ELECTRONIC_STRUCTURE = "electronic_structure"
    TARGET_PROPERTY = "target_property"


class DesiredChange(StrictSchema):
    axis: ChangeAxis
    direction: Literal["increase", "decrease", "target", "preserve", "explore"]
    property_name: str | None = Field(default=None, max_length=256)
    target_value: JsonValue = None
    unit: str | None = Field(default=None, max_length=128)
    source: Literal[
        "controller_rule",
        "declared_search_prior",
        "explicit_goal",
        "expert_evidence",
        "literature_generator_hint",
        "parent_candidate",
    ] = "controller_rule"
    rationale: NonEmptyText

    @model_validator(mode="after")
    def _property_target_is_named(self) -> DesiredChange:
        if self.axis == ChangeAxis.TARGET_PROPERTY and self.property_name is None:
            raise ValueError("target_property changes require property_name")
        if self.direction == "target" and self.target_value is None:
            raise ValueError("target direction requires target_value")
        return self


class FusionRevisionRequest(StrictSchema):
    goal: DiscoveryGoal
    candidate: Candidate
    state: UnifiedLatentStateRef
    latent: NumericTensor
    features: list[FusionFeatureInput] = Field(min_length=1)
    decision_context: FusionDecisionContext = Field(
        default_factory=FusionDecisionContext
    )

    @model_validator(mode="after")
    def _references_match(self) -> FusionRevisionRequest:
        if self.candidate.candidate_ref is None:
            raise ValueError("revision requires candidate_ref")
        if self.state.candidate_ref != self.candidate.candidate_ref:
            raise ValueError("fusion state is stale for this candidate")
        if self.latent.dtype != self.state.dtype or self.latent.shape != self.state.shape:
            raise ValueError("revision latent metadata does not match state")
        if (
            self.state.latent_content_hash is None
            or stable_hash(self.latent) != self.state.latent_content_hash
        ):
            raise ValueError("revision latent hash does not match state")
        feature_ids = [item.feature_id for item in self.features]
        if len(feature_ids) != len(set(feature_ids)):
            raise ValueError("duplicate revision feature ids are not allowed")
        if not set(self.state.source_feature_ids).issubset(feature_ids):
            raise ValueError("revision is missing latent source features")
        workspace_entities = {item.entity_id: item for item in self.state.workspace_entities}
        if any(
            item.workspace_entity_id not in workspace_entities
            or item.payload.candidate_ref
            != workspace_entities[item.workspace_entity_id].candidate_ref
            for item in self.features
        ):
            raise ValueError("revision feature is outside the latent workspace")
        return self


class FusionRevisionProposal(StrictSchema):
    parent_candidate_ref: CandidateRef
    state_id: Identifier
    desired_changes: list[DesiredChange] = Field(min_length=1)
    preferred_generator_ids: list[Identifier] = Field(default_factory=list)
    confidence: Probability
    rationale: NonEmptyText
    safety_notes: list[str] = Field(default_factory=list)


class FusionCycleReport(StrictSchema):
    candidate_ref: CandidateRef
    workspace: ScientificWorkspace
    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_mode: WorkspaceMode
    cycle: int = Field(ge=0)
    feature_refs: list[ExpertFeatureRef]
    latent_state: UnifiedLatentStateRef | None = None
    revision_proposal: FusionRevisionProposal | None = None
    missing_expert_ids: list[Identifier] = Field(default_factory=list)
    failed_expert_ids: list[Identifier] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _workspace_boundary(self) -> FusionCycleReport:
        if len(self.missing_expert_ids) != len(set(self.missing_expert_ids)):
            raise ValueError("duplicate missing expert ids are not allowed")
        if len(self.failed_expert_ids) != len(set(self.failed_expert_ids)):
            raise ValueError("duplicate failed expert ids are not allowed")
        primary = next(
            item for item in self.workspace.entities if item.entity_id == self.workspace.primary_entity_id
        )
        if primary.candidate_ref != self.candidate_ref:
            raise ValueError("cycle report candidate must be the workspace primary entity")
        workspace_entities = {item.entity_id: item for item in self.workspace.entities}
        if any(
            ref.workspace_entity_id not in workspace_entities
            or ref.candidate_ref != workspace_entities[ref.workspace_entity_id].candidate_ref
            or ref.goal_hash != self.goal_hash
            for ref in self.feature_refs
        ):
            raise ValueError("cycle report contains a feature outside its workspace or goal")
        if self.workspace_mode == WorkspaceMode.OFF:
            if self.latent_state is not None or self.revision_proposal is not None:
                raise ValueError("workspace OFF cannot emit latent state or revision proposal")
        elif self.latent_state is None:
            raise ValueError("workspace ON requires latent state")
        elif (
            self.latent_state.candidate_ref != self.candidate_ref
            or self.latent_state.workspace_id != self.workspace.workspace_id
            or self.latent_state.workspace_entities != self.workspace.entities
            or self.latent_state.workspace_relations != self.workspace.relations
            or self.latent_state.cycle != self.cycle
            or self.latent_state.goal_hash != self.goal_hash
        ):
            raise ValueError("cycle report latent state is inconsistent")
        if self.latent_state is not None:
            usable_feature_ids = {
                item.feature_id
                for item in self.feature_refs
                if item.status != FeatureStatus.FAILED and item.tensor_dtype is not None
            }
            if not set(self.latent_state.source_feature_ids).issubset(usable_feature_ids):
                raise ValueError("latent state cites a feature outside the cycle report")
        if self.revision_proposal is not None and (
            self.revision_proposal.parent_candidate_ref != self.candidate_ref
            or self.latent_state is None
            or self.revision_proposal.state_id != self.latent_state.state_id
        ):
            raise ValueError("cycle report revision is inconsistent")
        return self


class GenerationControls(StrictSchema):
    """Bounded controls passed to a generator for one exploration step."""

    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    temperature: float = Field(default=1.0, ge=0.01, le=5.0)
    mutation_strength: float = Field(default=0.2, ge=0.0, le=1.0)
    diversity_strength: float = Field(default=0.3, ge=0.0, le=1.0)
    schedule_step: int = Field(default=0, ge=0)
    decision_reason: NonEmptyText = "initial controls"


class WorkspaceRunConfig(StrictSchema):
    workspace_mode: WorkspaceMode
    seed: int = Field(ge=0)
    # Set by FusionSearchRunner so a stateful generator sidecar can reject and
    # replace duplicates across calls before expensive expert evaluation.
    search_session_id: Identifier | None = None
    generator_seed: int | None = Field(default=None, ge=0)
    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_candidate_ref: CandidateRef
    pair_key: Identifier
    cohort_index: int = Field(ge=0)
    generator_id: Identifier
    generator_version: Identifier
    generator_code_revision: Identifier
    generator_weight_revision: Identifier
    generator_parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    decoder_config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    postprocessing_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    resource_budget_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluator_panel_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metric_implementation_version: Identifier = "workspace-metrics-v2"
    candidate_count: int = Field(gt=0, le=1_024)
    generation_controls: GenerationControls = Field(default_factory=GenerationControls)

    @property
    def effective_generator_seed(self) -> int:
        """Sampling seed, falling back to the shared run seed for old configs."""

        return self.seed if self.generator_seed is None else self.generator_seed


class FusionWorkspaceSnapshot(StrictSchema):
    candidate: Candidate
    workspace: ScientificWorkspace
    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: WorkspaceMode
    feature_refs: list[ExpertFeatureRef]
    latent_state: UnifiedLatentStateRef | None = None
    aggregate_properties: list[DiagnosticProperty] = Field(default_factory=list)
    run_config: WorkspaceRunConfig
    missing_expert_ids: list[Identifier] = Field(default_factory=list)
    failed_expert_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def _snapshot_matches_mode(self) -> FusionWorkspaceSnapshot:
        if self.candidate.candidate_ref is None:
            raise ValueError("workspace snapshot requires candidate_ref")
        if self.run_config.workspace_mode != self.mode:
            raise ValueError("run_config workspace mode must match snapshot mode")
        if self.run_config.goal_hash != self.goal_hash:
            raise ValueError("snapshot goal hash must match run configuration")
        if self.mode == WorkspaceMode.OFF and self.latent_state is not None:
            raise ValueError("workspace OFF snapshot cannot contain latent state")
        if self.mode == WorkspaceMode.ON and self.latent_state is None:
            raise ValueError("workspace ON snapshot requires latent state")
        primary = next(
            item for item in self.workspace.entities if item.entity_id == self.workspace.primary_entity_id
        )
        if primary.candidate_ref != self.candidate.candidate_ref:
            raise ValueError("snapshot candidate must be the workspace primary entity")
        if self.latent_state is not None and (
            self.latent_state.workspace_id != self.workspace.workspace_id
            or self.latent_state.candidate_ref != self.candidate.candidate_ref
            or self.latent_state.goal_hash != self.goal_hash
            or self.latent_state.workspace_entities != self.workspace.entities
            or self.latent_state.workspace_relations != self.workspace.relations
        ):
            raise ValueError("snapshot latent state is inconsistent")
        if self.latent_state is not None:
            usable_feature_ids = {
                item.feature_id
                for item in self.feature_refs
                if item.status != FeatureStatus.FAILED and item.tensor_dtype is not None
            }
            if not set(self.latent_state.source_feature_ids).issubset(usable_feature_ids):
                raise ValueError("snapshot latent state cites an unknown feature")
        return self


class GeneratorProvenance(StrictSchema):
    generator_id: Identifier
    generator_version: Identifier
    code_revision: Identifier
    weight_revision: Identifier
    parameters_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_parameters_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    seed: int = Field(ge=0)


class GenerationPairSlot(StrictSchema):
    """Bind one generated candidate to its honest batch stream position.

    ``batch_seed`` is the seed used for the whole generator invocation.  It is
    deliberately not described as an independent per-candidate seed.
    """

    pair_slot: int = Field(ge=0, le=1_023)
    candidate_ref: CandidateRef
    batch_seed: int = Field(ge=0)
    stream_position: int = Field(ge=0, le=1_023)


class FusionGenerationRequest(StrictSchema):
    goal: DiscoveryGoal
    parent_candidate: Candidate
    workspace: ScientificWorkspace
    workspace_mode: WorkspaceMode
    run_config: WorkspaceRunConfig
    revision_proposal: FusionRevisionProposal | None = None
    latent_state: UnifiedLatentStateRef | None = None
    # Sidecars cannot dereference coordinator-local artifact paths.  The
    # coordinator therefore carries the bounded, verified tensor on ON wires.
    latent_payload: NumericTensor | None = None

    @model_validator(mode="after")
    def _generation_mode_is_consistent(self) -> FusionGenerationRequest:
        if self.parent_candidate.candidate_ref is None:
            raise ValueError("generation parent requires candidate_ref")
        if self.run_config.workspace_mode != self.workspace_mode:
            raise ValueError("generation config mode must match request mode")
        if self.run_config.goal_hash != stable_hash(self.goal):
            raise ValueError("generation config goal hash does not match goal")
        if self.run_config.parent_candidate_ref != self.parent_candidate.candidate_ref:
            raise ValueError("generation config parent does not match request parent")
        primary = next(
            item for item in self.workspace.entities if item.entity_id == self.workspace.primary_entity_id
        )
        if primary.candidate_ref != self.parent_candidate.candidate_ref:
            raise ValueError("generation workspace primary does not match parent")
        if self.workspace_mode == WorkspaceMode.OFF:
            if (
                self.revision_proposal is not None
                or self.latent_state is not None
                or self.latent_payload is not None
            ):
                raise ValueError("workspace OFF generation cannot receive fusion state")
        else:
            if (
                self.revision_proposal is None
                or self.latent_state is None
                or self.latent_payload is None
            ):
                raise ValueError(
                    "workspace ON generation requires state, latent payload, and revision proposal"
                )
            if (
                self.latent_state.candidate_ref != self.parent_candidate.candidate_ref
                or self.latent_state.workspace_id != self.workspace.workspace_id
                or self.latent_state.workspace_entities != self.workspace.entities
                or self.latent_state.workspace_relations != self.workspace.relations
                or self.revision_proposal.parent_candidate_ref
                != self.parent_candidate.candidate_ref
                or self.revision_proposal.state_id != self.latent_state.state_id
            ):
                raise ValueError("workspace ON generation state is inconsistent")
            if (
                self.latent_payload.dtype != self.latent_state.dtype
                or self.latent_payload.shape != self.latent_state.shape
            ):
                raise ValueError("workspace ON latent payload does not match state metadata")
            if (
                self.latent_state.latent_content_hash is None
                or stable_hash(self.latent_payload)
                != self.latent_state.latent_content_hash
            ):
                raise ValueError("workspace ON latent payload hash does not match state")
        return self


class FusionGenerationResponse(StrictSchema):
    # ``candidate`` keeps generator-v1 single-output sidecars wire-compatible;
    # new sidecars should return ``candidates`` for batch exploration.
    candidate: Candidate | None = None
    candidates: list[Candidate] = Field(default_factory=list, max_length=1_024)
    provenance: GeneratorProvenance
    pair_slots: list[GenerationPairSlot] = Field(default_factory=list, max_length=1_024)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _candidate_is_content_addressed(self) -> FusionGenerationResponse:
        if (self.candidate is None) == (not self.candidates):
            raise ValueError("generation response requires exactly one of candidate or candidates")
        rows = self.candidates if self.candidates else [self.candidate]
        refs: list[tuple[str, int]] = []
        for candidate in rows:
            if candidate is None or candidate.candidate_ref is None:
                raise ValueError("every generated candidate requires candidate_ref")
            if candidate_content_hash(candidate) != candidate.candidate_ref.content_hash:
                raise ValueError("generated candidate_ref content hash is stale")
            refs.append(
                (candidate.candidate_ref.candidate_id, candidate.candidate_ref.version)
            )
        if len(refs) != len(set(refs)):
            raise ValueError("generation response contains duplicate candidate refs")
        if self.pair_slots:
            if len(self.pair_slots) != len(rows):
                raise ValueError("pair-slot metadata must cover every generated candidate")
            expected_slots = list(range(len(rows)))
            actual_slots = [item.pair_slot for item in self.pair_slots]
            if actual_slots != expected_slots:
                raise ValueError("pair slots must be unique and in canonical batch order")
            if [item.candidate_ref for item in self.pair_slots] != [
                item.candidate_ref for item in rows if item is not None
            ]:
                raise ValueError("pair-slot metadata is reordered or cites another candidate")
            if any(item.batch_seed != self.provenance.seed for item in self.pair_slots):
                raise ValueError("pair-slot batch seed must match generator provenance")
            stream_positions = [item.stream_position for item in self.pair_slots]
            if len(stream_positions) != len(set(stream_positions)):
                raise ValueError("pair-slot stream positions must be unique")
        return self

    @property
    def generated_candidates(self) -> list[Candidate]:
        if self.candidates:
            return list(self.candidates)
        return [self.candidate] if self.candidate is not None else []

    @property
    def candidates_by_pair_slot(self) -> dict[int, Candidate]:
        """Return the explicit pairing map, or an empty map for legacy responses."""

        if not self.pair_slots:
            return {}
        candidates = self.generated_candidates
        return {
            metadata.pair_slot: candidate
            for metadata, candidate in zip(self.pair_slots, candidates, strict=True)
        }


class FusionIterationReport(StrictSchema):
    before_revision: FusionCycleReport
    generation: FusionGenerationResponse
    after_revision: FusionCycleReport

    @model_validator(mode="after")
    def _iteration_is_contiguous(self) -> FusionIterationReport:
        generated = self.generation.generated_candidates
        if len(generated) != 1:
            raise ValueError("single fusion iteration requires exactly one generated candidate")
        generated_ref = generated[0].candidate_ref
        if (
            self.before_revision.workspace_mode != WorkspaceMode.ON
            or self.after_revision.workspace_mode != WorkspaceMode.ON
            or self.after_revision.cycle != self.before_revision.cycle + 1
            or self.after_revision.candidate_ref != generated_ref
            or self.before_revision.latent_state is None
            or self.after_revision.latent_state is None
            or self.after_revision.latent_state.previous_state_id
            != self.before_revision.latent_state.state_id
        ):
            raise ValueError("fusion iteration reports must form one contiguous lineage")
        return self


def _workspace_context_signature(workspace: ScientificWorkspace) -> str:
    """Hash context and relations while allowing the primary candidate to change."""

    primary_id = workspace.primary_entity_id
    entities = sorted(
        (
            item.entity_id,
            str(item.role),
            item.candidate_ref.model_dump(mode="json"),
        )
        for item in workspace.entities
        if item.entity_id != primary_id
    )
    relations: list[dict[str, JsonValue]] = []
    for relation in workspace.relations:
        row = relation.model_dump(mode="json")
        if row["subject_entity_id"] == primary_id:
            row["subject_entity_id"] = "__primary__"
        if row["object_entity_id"] == primary_id:
            row["object_entity_id"] = "__primary__"
        relations.append(row)
    return stable_hash(
        {
            "entities": entities,
            "relations": sorted(relations, key=stable_hash),
        }
    )


class FusionBatchIterationReport(StrictSchema):
    before_revision: FusionCycleReport
    generation: FusionGenerationResponse
    after_revisions: list[FusionCycleReport] = Field(min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def _batch_is_contiguous(self) -> FusionBatchIterationReport:
        generated = self.generation.generated_candidates
        if len(generated) != len(self.after_revisions):
            raise ValueError("every generated candidate requires one after-revision report")
        if self.before_revision.workspace_mode != WorkspaceMode.ON:
            raise ValueError("batch fusion iteration requires workspace ON")
        before_state = self.before_revision.latent_state
        if before_state is None:
            raise ValueError("batch fusion iteration requires a before latent state")
        before_ref = self.before_revision.candidate_ref
        before_context = _workspace_context_signature(self.before_revision.workspace)
        for candidate, report in zip(generated, self.after_revisions, strict=True):
            if (
                report.workspace_mode != WorkspaceMode.ON
                or report.cycle != self.before_revision.cycle + 1
                or report.candidate_ref != candidate.candidate_ref
                or before_ref not in candidate.parent_candidate_refs
                or before_ref.candidate_id not in candidate.parent_candidate_ids
                or report.goal_hash != self.before_revision.goal_hash
                or report.workspace.workspace_id
                != self.before_revision.workspace.workspace_id
                or _workspace_context_signature(report.workspace) != before_context
                or report.latent_state is None
                or report.latent_state.previous_state_id != before_state.state_id
                or report.latent_state.state_version != before_state.state_version + 1
                or report.latent_state.seed != before_state.seed
            ):
                raise ValueError("batch after-revision report breaks lineage")
            candidate_ref = candidate.candidate_ref
            if (
                candidate_ref is not None
                and candidate_ref.candidate_id == before_ref.candidate_id
                and candidate_ref.version <= before_ref.version
            ):
                raise ValueError("batch candidate lineage cannot roll back its parent version")
        return self

    @property
    def after_revision(self) -> FusionCycleReport:
        """Compatibility accessor for callers requesting exactly one child."""

        if len(self.after_revisions) != 1:
            raise AttributeError("batch report has more than one after revision")
        return self.after_revisions[0]


class ObjectiveDelta(StrictSchema):
    property_name: Identifier
    direction: str = Field(min_length=1, max_length=64)
    unit: str | None = Field(default=None, max_length=128)
    off_value: float | None = None
    on_value: float | None = None
    raw_delta: float | None = None
    signed_improvement: float | None = None
    off_uncertainty: float | None = Field(default=None, ge=0.0)
    on_uncertainty: float | None = Field(default=None, ge=0.0)
    out_of_domain: bool = False
    comparable: bool
    caveat: str | None = Field(default=None, max_length=2_000)


class WorkspaceComparisonReport(StrictSchema):
    comparison_pair_id: Identifier
    off_candidate_ref: CandidateRef
    on_candidate_ref: CandidateRef
    off_scientific_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    on_scientific_identity_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    paired_configuration: bool
    element_total_variation: float | None = Field(default=None, ge=0.0, le=1.0)
    element_jensen_shannon_divergence: float | None = Field(default=None, ge=0.0, le=1.0)
    ordered_coordinate_rms_displacement: float | None = Field(default=None, ge=0.0)
    lattice_frobenius_distance: float | None = Field(default=None, ge=0.0)
    molecular_structure_changed: bool | None = None
    protein_sequence_normalized_edit_distance: float | None = Field(default=None, ge=0.0, le=1.0)
    rna_sequence_normalized_edit_distance: float | None = Field(default=None, ge=0.0, le=1.0)
    representation_kinds_changed: list[RepresentationKind] = Field(default_factory=list)
    objective_deltas: list[ObjectiveDelta] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    scientific_claim: Literal["diagnostic_only"] = "diagnostic_only"


class WorkspacePairedRunReport(StrictSchema):
    off_generation: FusionGenerationResponse
    on_generation: FusionGenerationResponse
    off_snapshots: list[FusionWorkspaceSnapshot] = Field(min_length=1, max_length=1_024)
    on_snapshots: list[FusionWorkspaceSnapshot] = Field(min_length=1, max_length=1_024)
    comparisons: list[WorkspaceComparisonReport] = Field(min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def _pair_references_match(self) -> WorkspacePairedRunReport:
        off_generated = self.off_generation.generated_candidates
        on_generated = self.on_generation.generated_candidates
        if (
            len(off_generated) != len(on_generated)
            or len(off_generated) != len(self.off_snapshots)
            or len(on_generated) != len(self.on_snapshots)
            or len(off_generated) != len(self.comparisons)
        ):
            raise ValueError("paired report batch lengths are inconsistent")
        off_config = self.off_snapshots[0].run_config
        on_config = self.on_snapshots[0].run_config
        if len(off_generated) != off_config.candidate_count or len(on_generated) != on_config.candidate_count:
            raise ValueError("paired report candidate count does not match its run configuration")
        if (
            off_config.workspace_mode != WorkspaceMode.OFF
            or on_config.workspace_mode != WorkspaceMode.ON
            or off_config.model_dump(mode="json", exclude={"workspace_mode"})
            != on_config.model_dump(mode="json", exclude={"workspace_mode"})
        ):
            raise ValueError("paired report OFF/ON run configurations do not match")

        off_runtime_hash = self.off_generation.provenance.runtime_parameters_hash
        on_runtime_hash = self.on_generation.provenance.runtime_parameters_hash
        if (
            off_runtime_hash is None
            or on_runtime_hash is None
            or off_runtime_hash != on_runtime_hash
        ):
            raise ValueError("paired report requires equal attested generator runtime parameters")

        if len(self.off_generation.pair_slots) != len(off_generated) or len(
            self.on_generation.pair_slots
        ) != len(on_generated):
            raise ValueError("paired report requires explicit pair-slot metadata")
        off_by_slot = self.off_generation.candidates_by_pair_slot
        on_by_slot = self.on_generation.candidates_by_pair_slot
        if set(off_by_slot) != set(on_by_slot) or set(off_by_slot) != set(
            range(len(off_generated))
        ):
            raise ValueError("paired report pair-slot sets are inconsistent")
        off_slot_metadata = {item.pair_slot: item for item in self.off_generation.pair_slots}
        on_slot_metadata = {item.pair_slot: item for item in self.on_generation.pair_slots}
        if any(
            off_slot_metadata[slot].batch_seed != off_config.effective_generator_seed
            or on_slot_metadata[slot].batch_seed != on_config.effective_generator_seed
            or off_slot_metadata[slot].stream_position
            != on_slot_metadata[slot].stream_position
            for slot in sorted(off_by_slot)
        ):
            raise ValueError("paired report batch seed or stream positions are inconsistent")

        parent_ref = off_config.parent_candidate_ref
        base_workspace_id = self.off_snapshots[0].workspace.workspace_id
        base_context = _workspace_context_signature(self.off_snapshots[0].workspace)
        base_goal_hash = self.off_snapshots[0].goal_hash
        for slot, off_snapshot, on_snapshot, comparison in zip(
            sorted(off_by_slot),
            self.off_snapshots,
            self.on_snapshots,
            self.comparisons,
            strict=True,
        ):
            off_candidate = off_by_slot[slot]
            on_candidate = on_by_slot[slot]
            if (
                off_snapshot.mode != WorkspaceMode.OFF
                or on_snapshot.mode != WorkspaceMode.ON
                or off_candidate.candidate_ref != off_snapshot.candidate.candidate_ref
                or on_candidate.candidate_ref != on_snapshot.candidate.candidate_ref
                or comparison.off_candidate_ref != off_snapshot.candidate.candidate_ref
                or comparison.on_candidate_ref != on_snapshot.candidate.candidate_ref
                or parent_ref not in off_candidate.parent_candidate_refs
                or parent_ref not in on_candidate.parent_candidate_refs
                or parent_ref.candidate_id not in off_candidate.parent_candidate_ids
                or parent_ref.candidate_id not in on_candidate.parent_candidate_ids
                or off_snapshot.goal_hash != base_goal_hash
                or on_snapshot.goal_hash != base_goal_hash
                or off_snapshot.workspace.workspace_id != base_workspace_id
                or on_snapshot.workspace.workspace_id != base_workspace_id
                or _workspace_context_signature(off_snapshot.workspace) != base_context
                or _workspace_context_signature(on_snapshot.workspace) != base_context
            ):
                raise ValueError("paired report snapshots and comparison are inconsistent")
            for candidate in (off_candidate, on_candidate):
                candidate_ref = candidate.candidate_ref
                if (
                    candidate_ref is not None
                    and candidate_ref.candidate_id == parent_ref.candidate_id
                    and candidate_ref.version <= parent_ref.version
                ):
                    raise ValueError("paired candidate lineage cannot roll back its parent version")
        for generation, snapshots in (
            (self.off_generation, self.off_snapshots),
            (self.on_generation, self.on_snapshots),
        ):
            provenance = generation.provenance
            config = snapshots[0].run_config
            if (
                provenance.generator_id != config.generator_id
                or provenance.generator_version != config.generator_version
                or provenance.code_revision != config.generator_code_revision
                or provenance.weight_revision != config.generator_weight_revision
                or provenance.parameters_hash != config.generator_parameters_hash
                or provenance.seed != config.effective_generator_seed
            ):
                raise ValueError("paired report generation provenance is inconsistent")
            if any(snapshot.run_config != config for snapshot in snapshots[1:]):
                raise ValueError("paired report arm contains mixed run configurations")
        return self

    @property
    def off_snapshot(self) -> FusionWorkspaceSnapshot:
        """Compatibility accessor for callers requesting exactly one OFF child."""

        if len(self.off_snapshots) != 1:
            raise AttributeError("paired report has more than one OFF snapshot")
        return self.off_snapshots[0]

    @property
    def on_snapshot(self) -> FusionWorkspaceSnapshot:
        """Compatibility accessor for callers requesting exactly one ON child."""

        if len(self.on_snapshots) != 1:
            raise AttributeError("paired report has more than one ON snapshot")
        return self.on_snapshots[0]

    @property
    def comparison(self) -> WorkspaceComparisonReport:
        """Compatibility accessor for callers requesting exactly one comparison."""

        if len(self.comparisons) != 1:
            raise AttributeError("paired report has more than one comparison")
        return self.comparisons[0]


__all__ = [
    "ChangeAxis",
    "ContentArtifactRef",
    "DesiredChange",
    "DiagnosticProperty",
    "ExpertDescriptor",
    "ExpertRoute",
    "ExpertFeaturePayload",
    "ExpertFeatureRef",
    "ExpertFeatureRequest",
    "ExpertProvenance",
    "FeatureSemantics",
    "FeatureStatus",
    "FusionCycleReport",
    "FusionDecisionContext",
    "FusionFeatureInput",
    "FusionGenerationRequest",
    "FusionGenerationResponse",
    "FusionBatchIterationReport",
    "FusionIterationReport",
    "GenerationPairSlot",
    "GenerationControls",
    "FusionOutput",
    "FusionRequest",
    "FusionRevisionProposal",
    "FusionRevisionRequest",
    "FusionWorkspaceSnapshot",
    "GeneratorProvenance",
    "NumericTensor",
    "ObjectiveDelta",
    "ScientificModality",
    "ScientificWorkspace",
    "TensorDType",
    "TensorRole",
    "UnifiedLatentStateRef",
    "UnifiedLatentStateRecord",
    "WorkspaceComparisonReport",
    "WorkspaceMode",
    "WorkspaceEntity",
    "WorkspaceEntityInput",
    "WorkspaceEntityRole",
    "WorkspaceRelation",
    "WorkspaceRunConfig",
    "WorkspacePairedRunReport",
]
