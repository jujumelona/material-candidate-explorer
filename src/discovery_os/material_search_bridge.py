"""Typed bridge from an application role to the real Fusion search runtime.

The application layer identifies a material function and the evidence needed
to judge it.  It does not, by itself, contain a crystal structure that can be
used as a generator parent.  This module therefore executes only when the
caller supplies an immutable structural parent, a matching ``DiscoveryGoal``,
an attested workspace run configuration, and an explicit expert panel.

The bridge invokes :class:`FusionSearchRunner` directly.  It never converts a
retrieval seed or a literature/MCP record into a structure, and it never turns
the diagnostic search ranking into an application-property score.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from .fusion_search import (
    FusionSearchError,
    FusionSearchRunner,
    FusionSearchStatus,
    PersistedFusionSearchReport,
    SearchBudget,
    SearchControlSweep,
)
from .fusion_schemas import (
    ContentArtifactRef,
    ScientificModality,
    UnifiedLatentStateRef,
    WorkspaceEntityInput,
    WorkspaceMode,
    WorkspaceRelation,
    WorkspaceRunConfig,
)
from .hashing import bytes_hash, candidate_content_hash, canonical_json, stable_hash
from .material_applications import (
    ApplicationRoleProfile,
    MaterialApplicationBrief,
)
from .profiles import get_validation_profile
from .schemas import (
    Candidate,
    CandidateRef,
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    Identifier,
    RepresentationKind,
    StrictSchema,
)


_MATERIAL_DOMAINS = {
    DiscoveryDomain.INORGANIC_MATERIALS.value,
    DiscoveryDomain.SUPERCONDUCTORS.value,
    DiscoveryDomain.BATTERIES.value,
    DiscoveryDomain.CATALYSTS.value,
    DiscoveryDomain.GENERAL_MATERIALS.value,
}
_STRUCTURAL_REPRESENTATIONS = {
    RepresentationKind.CIF.value,
    RepresentationKind.MMCIF.value,
    RepresentationKind.POSCAR.value,
}
_STRUCTURAL_CANDIDATE_TYPES = {
    CandidateType.CRYSTAL.value,
    CandidateType.COMPOSITION.value,
    CandidateType.ALLOY.value,
    CandidateType.BATTERY_MATERIAL.value,
    CandidateType.CATALYST.value,
}
_FORBIDDEN_RUNTIME_MARKERS = ("dummy", "mock", "placeholder")
_BULK_DIAGNOSTIC_OBJECTIVES = {
    "energy_per_atom": ("minimize", "eV/atom"),
    "max_force": ("minimize", "eV/angstrom"),
}


class MaterialApplicationFusionSearchRequest(StrictSchema):
    """One explicit, executable bulk-crystal search for one application role."""

    schema_version: Literal["1.0"] = "1.0"
    brief_id: Identifier
    role_id: Identifier
    search_id: Identifier
    goal: DiscoveryGoal
    initial_candidate: Candidate
    base_run_config: WorkspaceRunConfig
    rounds: int = Field(default=4, ge=3, le=100)
    initial_cycle: int = Field(default=0, ge=0)
    initial_state: UnifiedLatentStateRef | None = None
    expert_ids: list[Identifier] = Field(min_length=2)
    required_primary_evaluator_ids: list[Identifier] = Field(min_length=2)
    modality: Literal["crystal_material"] = "crystal_material"
    context_entities: list[WorkspaceEntityInput] = Field(default_factory=list)
    relations: list[WorkspaceRelation] = Field(default_factory=list)
    workspace_id: Identifier | None = None
    frontier_width: int = Field(default=1, ge=1, le=64)
    control_sweep: SearchControlSweep | None = None
    search_budget: SearchBudget
    ranking_limit: int = Field(default=50, ge=1, le=1_024)
    execution_scope: Literal["bulk-crystal-search-triage-only"] = (
        "bulk-crystal-search-triage-only"
    )
    retrieval_seed_promoted_to_structure: Literal[False] = False
    application_property_scoring_requested: Literal[False] = False
    application_rag_used_as_runtime_validator: Literal[False] = False

    @model_validator(mode="after")
    def _request_is_executable(self) -> "MaterialApplicationFusionSearchRequest":
        if len(self.expert_ids) != len(set(self.expert_ids)):
            raise ValueError("material search expert identifiers must be unique")
        if len(self.required_primary_evaluator_ids) != len(
            set(self.required_primary_evaluator_ids)
        ):
            raise ValueError(
                "required material-search evaluator identifiers must be unique"
            )
        if not set(self.required_primary_evaluator_ids).issubset(self.expert_ids):
            raise ValueError(
                "required material-search evaluators must be in the expert panel"
            )
        runtime_ids = [
            self.base_run_config.generator_id,
            *self.expert_ids,
            *self.required_primary_evaluator_ids,
        ]
        if any(
            marker in str(runtime_id).casefold()
            for runtime_id in runtime_ids
            for marker in _FORBIDDEN_RUNTIME_MARKERS
        ):
            raise ValueError(
                "application execution cannot bind a dummy, mock, or placeholder runtime"
            )
        if str(self.goal.domain) not in _MATERIAL_DOMAINS:
            raise ValueError("application Fusion search requires a material goal")
        if str(self.initial_candidate.domain) != str(self.goal.domain):
            raise ValueError("application search parent and goal domains differ")
        if str(self.initial_candidate.candidate_type) not in {
            str(item) for item in self.goal.candidate_types
        }:
            raise ValueError("application search parent type is outside the goal")
        if str(self.initial_candidate.candidate_type) not in _STRUCTURAL_CANDIDATE_TYPES:
            raise ValueError("application Fusion search requires a structural candidate")
        representation_kinds = {
            str(item.kind) for item in self.initial_candidate.representations
        }
        if not representation_kinds.intersection(_STRUCTURAL_REPRESENTATIONS):
            raise ValueError(
                "application Fusion search requires an explicit CIF, mmCIF, or POSCAR"
            )
        parent_ref = self.initial_candidate.candidate_ref
        if (
            parent_ref is None
            or candidate_content_hash(self.initial_candidate)
            != parent_ref.content_hash
        ):
            raise ValueError(
                "application Fusion search requires a current immutable parent reference"
            )
        config = self.base_run_config
        if config.workspace_mode != WorkspaceMode.ON:
            raise ValueError("application Fusion search requires workspace ON")
        if config.goal_hash != stable_hash(self.goal):
            raise ValueError("application search configuration belongs to another goal")
        if config.parent_candidate_ref != parent_ref:
            raise ValueError(
                "application search configuration belongs to another parent"
            )
        if config.search_session_id not in {None, self.search_id}:
            raise ValueError(
                "application search configuration belongs to another search session"
            )
        budget = self.search_budget
        if not 8 <= budget.max_generated_candidates <= 32:
            raise ValueError(
                "application material search requires a global 8 to 32 candidate budget"
            )
        if not 3 <= budget.max_generation_calls <= 128:
            raise ValueError(
                "application material search requires 3 to 128 generation calls"
            )
        if config.candidate_count > budget.max_generated_candidates:
            raise ValueError(
                "per-call candidate count exceeds the global candidate budget"
            )
        variants_per_parent = (
            self.control_sweep.max_variants_per_parent
            if self.control_sweep is not None
            else 1
        )
        branch_frontier_width = min(
            self.frontier_width,
            config.candidate_count,
        )
        # Round one seeds the Pareto branch.  Round two may then exercise all
        # five code-owned branches.  Reserve one additional call so round three
        # can consume the scheduler observation from round two.
        parent_attempts_before_third_round = 1 + 5 * branch_frontier_width
        calls_to_enter_third_round = (
            parent_attempts_before_third_round * variants_per_parent + 1
        )
        candidates_to_enter_third_round = (
            (parent_attempts_before_third_round + 1)
            * config.candidate_count
        )
        if budget.max_generation_calls < calls_to_enter_third_round:
            raise ValueError(
                "generation-call budget cannot reach the third adaptive round "
                f"(needs at least {calls_to_enter_third_round})"
            )
        if budget.max_generated_candidates < candidates_to_enter_third_round:
            raise ValueError(
                "generated-candidate budget cannot reach the third adaptive round "
                f"(needs at least {candidates_to_enter_third_round})"
            )
        if self.initial_state is not None:
            if (
                self.initial_state.candidate_ref != parent_ref
                or self.initial_state.goal_hash != stable_hash(self.goal)
                or self.initial_state.seed != config.seed
            ):
                raise ValueError(
                    "application search initial state is unrelated to the request"
                )
        return self


class MaterialApplicationFusionSearchReport(StrictSchema):
    """Closed execution receipt whose scientific meaning remains diagnostic."""

    execution_id: Identifier
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    application_brief: MaterialApplicationBrief
    role_profile: ApplicationRoleProfile
    request: MaterialApplicationFusionSearchRequest
    application_field_profile_id: Identifier
    runtime_validation_profile_id: Identifier
    diagnostic_objective_names: list[Identifier] = Field(min_length=1)
    search: PersistedFusionSearchReport
    search_status: FusionSearchStatus
    generation_execution_succeeded: bool
    specialist_feature_evaluation_succeeded: bool
    validation_handoff_candidate_refs: list[CandidateRef] = Field(
        default_factory=list
    )
    unexecuted_application_validator_ids: list[Identifier] = Field(
        default_factory=list
    )
    condition_complete_for_application_ranking: bool
    application_property_scoring_performed: Literal[False] = False
    application_claim_created: Literal[False] = False
    search_result_semantics: Literal[
        "diagnostic-bulk-crystal-priority-not-application-fitness"
    ] = "diagnostic-bulk-crystal-priority-not-application-fitness"
    later_stage_validation_required: Literal[True] = True

    @model_validator(mode="after")
    def _report_is_closed(self) -> "MaterialApplicationFusionSearchReport":
        request = self.request
        brief = self.application_brief
        role = self.role_profile
        search_report = self.search.report
        if request.brief_id != brief.brief_id:
            raise ValueError("application search request cites another brief")
        matching_roles = [
            item for item in brief.roles if item.role_id == request.role_id
        ]
        if len(matching_roles) != 1 or matching_roles[0] != role:
            raise ValueError("application search role is not selected by the brief")
        if "bulk_crystal" not in role.representation_scopes:
            raise ValueError(
                "selected application role has no bulk-crystal screening scope"
            )
        if brief.field_plan.resolution.requires_operator_choice:
            raise ValueError(
                "ambiguous material-field routing cannot execute a specialized search"
            )
        _validate_goal_for_role(brief, role, request.goal)
        expected_runtime_profile = get_validation_profile(request.goal.domain)
        if request.goal.validation_profile_id != expected_runtime_profile.profile_id:
            raise ValueError(
                "application search goal uses another runtime validation profile"
            )
        if (
            self.application_field_profile_id
            != brief.field_plan.profile.profile_id
        ):
            raise ValueError(
                "application field profile id differs from the selected brief"
            )
        if self.runtime_validation_profile_id != expected_runtime_profile.profile_id:
            raise ValueError(
                "runtime validation profile id differs from the executed goal"
            )
        expected_diagnostics = [
            item.property_name for item in request.goal.objectives
        ]
        if self.diagnostic_objective_names != expected_diagnostics:
            raise ValueError(
                "diagnostic objective names differ from the executed goal"
            )
        _validate_diagnostic_objectives(request.goal)
        if search_report.search_id != request.search_id:
            raise ValueError("application bridge received another search report")
        if search_report.goal_hash != stable_hash(request.goal):
            raise ValueError("application bridge search report belongs to another goal")
        executed_config = request.base_run_config.model_copy(
            update={"search_session_id": request.search_id}
        )
        if search_report.base_run_config_hash != stable_hash(executed_config):
            raise ValueError(
                "application bridge search report belongs to another run configuration"
            )
        if search_report.rounds_requested != request.rounds:
            raise ValueError("application bridge search round count changed")
        expected_success = bool(search_report.cycle_records)
        if self.generation_execution_succeeded != expected_success:
            raise ValueError("generation success flag does not match executed cycles")
        if self.specialist_feature_evaluation_succeeded != expected_success:
            raise ValueError(
                "specialist evaluation success flag does not match executed cycles"
            )
        if (
            self.validation_handoff_candidate_refs
            != search_report.validation_handoff_candidate_refs
        ):
            raise ValueError(
                "application bridge validation handoff differs from Fusion search"
            )
        expected_validators = list(
            dict.fromkeys(
                validator_id
                for criterion in role.criteria
                for validator_id in criterion.validator_ids
            )
        )
        if self.unexecuted_application_validator_ids != expected_validators:
            raise ValueError(
                "application bridge must preserve every unexecuted role validator"
            )
        if (
            self.condition_complete_for_application_ranking
            != brief.ready_for_condition_complete_scoring
        ):
            raise ValueError(
                "application bridge condition-completeness flag differs from the brief"
            )
        expected_request_hash = stable_hash(request)
        if self.request_hash != expected_request_hash:
            raise ValueError("application search request hash is stale")
        expected_id = (
            "MAFEXEC-"
            + stable_hash(
                {
                    "brief_id": brief.brief_id,
                    "role_id": role.role_id,
                    "request_hash": expected_request_hash,
                    "search_artifact_sha256": self.search.report_artifact.sha256,
                }
            )[:24]
        )
        if self.execution_id != expected_id:
            raise ValueError("application search execution id is not content-addressed")
        return self


class PersistedMaterialApplicationFusionSearchReport(StrictSchema):
    report: MaterialApplicationFusionSearchReport
    report_artifact: ContentArtifactRef

    @model_validator(mode="after")
    def _artifact_is_content_addressed(
        self,
    ) -> "PersistedMaterialApplicationFusionSearchReport":
        encoded = canonical_json(self.report).encode("utf-8")
        digest = bytes_hash(encoded)
        expected_path = (
            f"fusion/search/{self.report.request.search_id}/application-bridge/"
            f"{digest[:2]}/{digest}.json"
        )
        artifact = self.report_artifact
        expected_id = f"MAFREPORT-{digest[:32]}"
        if artifact.artifact_id != expected_id:
            raise ValueError("application bridge report id is not content-addressed")
        if (
            artifact.relative_path != expected_path
            or artifact.sha256 != digest
            or artifact.byte_size != len(encoded)
            or artifact.media_type
            != "application/vnd.discovery-os.material-application-fusion-search+json"
        ):
            raise ValueError(
                "application bridge report artifact failed content validation"
            )
        return self


class MaterialApplicationFusionSearchBridge:
    """Validate one application role, then invoke the real search runner."""

    def __init__(self, search_runner: FusionSearchRunner) -> None:
        self.search_runner = search_runner

    def execute(
        self,
        *,
        brief: MaterialApplicationBrief,
        request: MaterialApplicationFusionSearchRequest,
    ) -> PersistedMaterialApplicationFusionSearchReport:
        brief = MaterialApplicationBrief.model_validate_json(
            brief.model_dump_json(),
            strict=True,
        )
        request = MaterialApplicationFusionSearchRequest.model_validate_json(
            request.model_dump_json(),
            strict=True,
        )
        role = _selected_bulk_role(brief, request)
        _validate_goal_for_role(brief, role, request.goal)

        search = self.search_runner.run(
            search_id=request.search_id,
            goal=request.goal,
            initial_candidate=request.initial_candidate,
            base_run_config=request.base_run_config,
            rounds=request.rounds,
            initial_cycle=request.initial_cycle,
            initial_state=request.initial_state,
            expert_ids=request.expert_ids,
            required_primary_evaluator_ids=request.required_primary_evaluator_ids,
            modality=ScientificModality.CRYSTAL_MATERIAL,
            context_entities=request.context_entities,
            relations=request.relations,
            workspace_id=request.workspace_id,
            frontier_width=request.frontier_width,
            # Application RAG is citation/validation context and is not silently
            # promoted into generator authority by this bridge.
            evidence_policy=None,
            control_sweep=request.control_sweep,
            search_budget=request.search_budget,
            ranking_limit=request.ranking_limit,
        )
        if not isinstance(search, PersistedFusionSearchReport):
            raise FusionSearchError(
                "FusionSearchRunner returned the wrong persisted report type"
            )
        search = PersistedFusionSearchReport.model_validate_json(
            search.model_dump_json(),
            strict=True,
        )
        request_hash = stable_hash(request)
        report = MaterialApplicationFusionSearchReport(
            execution_id=(
                "MAFEXEC-"
                + stable_hash(
                    {
                        "brief_id": brief.brief_id,
                        "role_id": role.role_id,
                        "request_hash": request_hash,
                        "search_artifact_sha256": search.report_artifact.sha256,
                    }
                )[:24]
            ),
            request_hash=request_hash,
            application_brief=brief,
            role_profile=role,
            request=request,
            application_field_profile_id=brief.field_plan.profile.profile_id,
            runtime_validation_profile_id=request.goal.validation_profile_id,
            diagnostic_objective_names=[
                item.property_name for item in request.goal.objectives
            ],
            search=search,
            search_status=search.report.status,
            generation_execution_succeeded=bool(search.report.cycle_records),
            specialist_feature_evaluation_succeeded=bool(
                search.report.cycle_records
            ),
            validation_handoff_candidate_refs=list(
                search.report.validation_handoff_candidate_refs
            ),
            unexecuted_application_validator_ids=list(
                dict.fromkeys(
                    validator_id
                    for criterion in role.criteria
                    for validator_id in criterion.validator_ids
                )
            ),
            condition_complete_for_application_ranking=(
                brief.ready_for_condition_complete_scoring
            ),
        )
        return self._persist(report)

    def _persist(
        self,
        report: MaterialApplicationFusionSearchReport,
    ) -> PersistedMaterialApplicationFusionSearchReport:
        encoded = canonical_json(report).encode("utf-8")
        digest = bytes_hash(encoded)
        relative_path = (
            f"fusion/search/{report.request.search_id}/application-bridge/"
            f"{digest[:2]}/{digest}.json"
        )
        written, written_digest = (
            self.search_runner.loop_runner.runtime.artifact_store.write_bytes(
                relative_path,
                encoded,
            )
        )
        if written_digest != digest:
            raise FusionSearchError(
                "artifact store changed an application bridge report digest"
            )
        return PersistedMaterialApplicationFusionSearchReport(
            report=report,
            report_artifact=ContentArtifactRef(
                artifact_id=f"MAFREPORT-{digest[:32]}",
                relative_path=written,
                sha256=digest,
                media_type=(
                    "application/vnd.discovery-os."
                    "material-application-fusion-search+json"
                ),
                byte_size=len(encoded),
            ),
        )


def build_material_application_fusion_search_request(
    brief: MaterialApplicationBrief,
    *,
    role_id: str,
    search_id: str,
    goal: DiscoveryGoal,
    initial_candidate: Candidate,
    base_run_config: WorkspaceRunConfig,
    expert_ids: Sequence[str],
    required_primary_evaluator_ids: Sequence[str] | None = None,
    rounds: int = 4,
    initial_cycle: int = 0,
    initial_state: UnifiedLatentStateRef | None = None,
    context_entities: Sequence[WorkspaceEntityInput] = (),
    relations: Sequence[WorkspaceRelation] = (),
    workspace_id: str | None = None,
    frontier_width: int = 1,
    control_sweep: SearchControlSweep | None = None,
    search_budget: SearchBudget | None = None,
    ranking_limit: int = 50,
) -> MaterialApplicationFusionSearchRequest:
    """Construct and cross-check a request before any runtime call."""

    panel = list(dict.fromkeys(str(item) for item in expert_ids))
    required = list(
        dict.fromkeys(
            str(item)
            for item in (
                required_primary_evaluator_ids
                if required_primary_evaluator_ids is not None
                else panel
            )
        )
    )
    bounded_budget = search_budget or SearchBudget(
        max_generation_calls=8,
        max_generated_candidates=8,
    )
    request = MaterialApplicationFusionSearchRequest(
        brief_id=brief.brief_id,
        role_id=role_id,
        search_id=search_id,
        goal=goal,
        initial_candidate=initial_candidate,
        base_run_config=base_run_config,
        rounds=rounds,
        initial_cycle=initial_cycle,
        initial_state=initial_state,
        expert_ids=panel,
        required_primary_evaluator_ids=required,
        context_entities=list(context_entities),
        relations=list(relations),
        workspace_id=workspace_id,
        frontier_width=frontier_width,
        control_sweep=control_sweep,
        search_budget=bounded_budget,
        ranking_limit=ranking_limit,
    )
    role = _selected_bulk_role(brief, request)
    _validate_goal_for_role(brief, role, request.goal)
    return request


def _selected_bulk_role(
    brief: MaterialApplicationBrief,
    request: MaterialApplicationFusionSearchRequest,
) -> ApplicationRoleProfile:
    if request.brief_id != brief.brief_id:
        raise ValueError("application search request cites another brief")
    matches = [item for item in brief.roles if item.role_id == request.role_id]
    if len(matches) != 1:
        raise ValueError(
            "application search role must be selected exactly once by the brief"
        )
    role = matches[0]
    if "bulk_crystal" not in role.representation_scopes:
        raise ValueError(
            "selected application role cannot be screened by a bulk crystal search"
        )
    if brief.field_plan.resolution.requires_operator_choice:
        raise ValueError(
            "ambiguous material-field routing requires an operator choice before search"
        )
    return role


def _validate_goal_for_role(
    brief: MaterialApplicationBrief,
    role: ApplicationRoleProfile,
    goal: DiscoveryGoal,
) -> None:
    if role.material_field != brief.material_field:
        raise ValueError("application role and brief material fields differ")
    if str(goal.domain) != str(brief.field_plan.profile.discovery_domain):
        raise ValueError(
            "application search goal domain differs from the selected material field"
        )
    expected_runtime_profile = get_validation_profile(goal.domain)
    if goal.validation_profile_id != expected_runtime_profile.profile_id:
        raise ValueError(
            "application search goal must use the code-owned runtime validation profile"
        )
    _validate_diagnostic_objectives(goal)


def _validate_diagnostic_objectives(goal: DiscoveryGoal) -> None:
    """Accept only properties actually shared by the trusted MLIP search panel."""

    for objective in goal.objectives:
        expected = _BULK_DIAGNOSTIC_OBJECTIVES.get(objective.property_name)
        if expected is None:
            raise ValueError(
                "application Fusion goal objective is outside the code-owned "
                "bulk diagnostic allowlist: "
                + objective.property_name
            )
        expected_direction, expected_unit = expected
        if str(objective.direction) != expected_direction:
            raise ValueError(
                f"{objective.property_name} must use {expected_direction} direction"
            )
        if objective.unit != expected_unit:
            raise ValueError(
                f"{objective.property_name} must use canonical unit {expected_unit}"
            )


__all__ = [
    "MaterialApplicationFusionSearchBridge",
    "MaterialApplicationFusionSearchReport",
    "MaterialApplicationFusionSearchRequest",
    "PersistedMaterialApplicationFusionSearchReport",
    "build_material_application_fusion_search_request",
]
