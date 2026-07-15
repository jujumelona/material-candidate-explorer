"""Closed-loop generation and paired workspace OFF/ON execution."""

from __future__ import annotations

from collections.abc import Iterable

from .fusion_metrics import compare_workspace_snapshots
from .fusion_protocols import FusionCandidateGenerator
from .fusion_runtime import FusionRuntime, FusionRuntimeError
from .fusion_schemas import (
    FusionBatchIterationReport,
    FusionDecisionContext,
    FusionGenerationRequest,
    FusionGenerationResponse,
    ScientificModality,
    UnifiedLatentStateRef,
    WorkspaceEntityInput,
    WorkspaceMode,
    WorkspacePairedRunReport,
    WorkspaceRelation,
    WorkspaceRunConfig,
)
from .hashing import candidate_content_hash, stable_hash
from .schemas import Candidate, DiscoveryGoal


class FusionLoopRunner:
    """Execute revision -> generation -> re-extraction -> latent update."""

    def __init__(self, runtime: FusionRuntime, generator: FusionCandidateGenerator) -> None:
        self.runtime = runtime
        self.generator = generator

    def iterate(
        self,
        *,
        goal: DiscoveryGoal,
        parent_candidate: Candidate,
        cycle: int,
        run_config: WorkspaceRunConfig,
        previous_state: UnifiedLatentStateRef | None = None,
        expert_ids: Iterable[str] | None = None,
        modality: ScientificModality | None = None,
        context_entities: Iterable[WorkspaceEntityInput] | None = None,
        relations: Iterable[WorkspaceRelation] | None = None,
        workspace_id: str | None = None,
        decision_context: FusionDecisionContext | None = None,
    ) -> FusionBatchIterationReport:
        try:
            run_config = WorkspaceRunConfig.model_validate_json(
                run_config.model_dump_json(),
                strict=True,
            )
        except Exception as exc:
            raise FusionRuntimeError(f"run configuration is invalid: {exc}") from exc
        if run_config.workspace_mode != WorkspaceMode.ON:
            raise FusionRuntimeError("closed-loop iteration requires workspace ON")
        try:
            decision_context = FusionDecisionContext.model_validate_json(
                (
                    decision_context
                    or FusionDecisionContext(
                        guidance_alpha=run_config.generation_controls.alpha,
                    )
                ).model_dump_json(),
                strict=True,
            )
        except Exception as exc:
            raise FusionRuntimeError(f"fusion decision context is invalid: {exc}") from exc
        if decision_context.guidance_alpha != run_config.generation_controls.alpha:
            raise FusionRuntimeError(
                "fusion decision context alpha differs from generation controls"
            )
        contexts = list(context_entities or [])
        relation_rows = list(relations or [])
        requested_experts = None if expert_ids is None else list(expert_ids)
        self._validate_config(goal, parent_candidate, run_config)
        self._validate_generator_binding(run_config)
        before = self.runtime.update(
            goal=goal,
            candidate=parent_candidate,
            cycle=cycle,
            seed=run_config.seed,
            workspace_mode=WorkspaceMode.ON,
            previous_state=previous_state,
            expert_ids=requested_experts,
            modality=modality,
            context_entities=contexts,
            relations=relation_rows,
            workspace_id=workspace_id,
            propose_revision=True,
            decision_context=decision_context,
        )
        if before.latent_state is None or before.revision_proposal is None:
            raise FusionRuntimeError("fusion backend did not produce state and revision")
        generation_request = FusionGenerationRequest(
            goal=goal,
            parent_candidate=parent_candidate,
            workspace=before.workspace,
            workspace_mode=WorkspaceMode.ON,
            run_config=run_config,
            revision_proposal=before.revision_proposal,
            latent_state=before.latent_state,
            latent_payload=self.runtime.materialize_latent(before.latent_state),
        )
        generation = self._call_generator(generation_request)
        generation = self._validated_generation(
            generation,
            parent_candidate,
            run_config,
            goal,
            expected_runtime_parameters_hash=getattr(
                self.generator,
                "expected_runtime_parameters_hash",
                None,
            ),
        )
        after_revisions = [
            self.runtime.update(
                goal=goal,
                candidate=generated,
                cycle=cycle + 1,
                seed=run_config.seed,
                workspace_mode=WorkspaceMode.ON,
                previous_state=before.latent_state,
                expert_ids=requested_experts,
                modality=modality,
                context_entities=contexts,
                relations=relation_rows,
                workspace_id=before.workspace.workspace_id,
                propose_revision=True,
                decision_context=decision_context,
            )
            for generated in generation.generated_candidates
        ]
        report = FusionBatchIterationReport(
            before_revision=before,
            generation=generation,
            after_revisions=after_revisions,
        )
        digest = stable_hash(report)
        self.runtime.artifact_store.write_json(
            f"fusion/iterations/ITER-{digest[:24]}.json",
            report,
        )
        return report

    @staticmethod
    def _request_copy(request: FusionGenerationRequest) -> FusionGenerationRequest:
        return FusionGenerationRequest.model_validate_json(
            request.model_dump_json(),
            strict=True,
        )

    def _call_generator(
        self,
        request: FusionGenerationRequest,
    ) -> FusionGenerationResponse:
        wire_request = self._request_copy(request)
        request_hash = stable_hash(wire_request)
        response = self.generator.generate(wire_request)
        if stable_hash(wire_request) != request_hash:
            raise FusionRuntimeError("generator mutated its request")
        return response

    def _validate_generator_binding(self, config: WorkspaceRunConfig) -> None:
        for attribute, actual in (
            ("expected_generator_id", config.generator_id),
            ("expected_generator_version", config.generator_version),
            ("expected_code_revision", config.generator_code_revision),
            ("expected_weight_revision", config.generator_weight_revision),
        ):
            expected = getattr(self.generator, attribute, None)
            if expected is not None and expected != actual:
                raise FusionRuntimeError(
                    f"selected generator binding does not match {attribute}"
                )

    @staticmethod
    def _validate_config(
        goal: DiscoveryGoal,
        parent: Candidate,
        config: WorkspaceRunConfig,
    ) -> None:
        if parent.candidate_ref is None:
            raise FusionRuntimeError("generation parent requires candidate_ref")
        if candidate_content_hash(parent) != parent.candidate_ref.content_hash:
            raise FusionRuntimeError("generation parent candidate_ref is stale")
        if config.goal_hash != stable_hash(goal):
            raise FusionRuntimeError("run configuration belongs to another goal")
        if config.parent_candidate_ref != parent.candidate_ref:
            raise FusionRuntimeError("run configuration belongs to another parent candidate")
    @staticmethod
    def _validated_generation(
        response: FusionGenerationResponse,
        parent: Candidate,
        config: WorkspaceRunConfig,
        goal: DiscoveryGoal,
        *,
        expected_runtime_parameters_hash: str | None = None,
    ) -> FusionGenerationResponse:
        if not isinstance(response, FusionGenerationResponse):
            raise FusionRuntimeError("generator returned the wrong response type")
        try:
            response = FusionGenerationResponse.model_validate_json(
                response.model_dump_json(),
                strict=True,
            )
        except Exception as exc:
            raise FusionRuntimeError(f"generator returned an invalid response: {exc}") from exc
        parent_ref = parent.candidate_ref
        if parent_ref is None:
            raise FusionRuntimeError("generation parent requires candidate_ref")
        generated_candidates = response.generated_candidates
        if len(generated_candidates) != config.candidate_count:
            raise FusionRuntimeError("generator returned a different candidate count than requested")
        for row in generated_candidates:
            generated = Candidate.model_validate_json(row.model_dump_json(), strict=True)
            generated_ref = generated.candidate_ref
            if generated_ref is None:
                raise FusionRuntimeError("generated candidate requires candidate_ref")
            if candidate_content_hash(generated) != generated_ref.content_hash:
                raise FusionRuntimeError("generated candidate_ref content hash is stale")
            if parent_ref not in generated.parent_candidate_refs:
                raise FusionRuntimeError(
                    "generated candidate does not cite the exact parent reference"
                )
            if parent_ref.candidate_id not in generated.parent_candidate_ids:
                raise FusionRuntimeError("generated candidate does not cite its parent id")
            if (
                generated_ref.candidate_id == parent_ref.candidate_id
                and generated_ref.version <= parent_ref.version
            ):
                raise FusionRuntimeError("same-id generated candidates must advance the version")
            if generated.candidate_type not in goal.candidate_types:
                raise FusionRuntimeError("generated candidate type is outside the goal")
            if generated.domain != goal.domain:
                raise FusionRuntimeError("generated candidate domain is outside the goal")
        provenance = response.provenance
        expected = (
            config.generator_id,
            config.generator_version,
            config.generator_code_revision,
            config.generator_weight_revision,
            config.generator_parameters_hash,
            config.effective_generator_seed,
        )
        actual = (
            provenance.generator_id,
            provenance.generator_version,
            provenance.code_revision,
            provenance.weight_revision,
            provenance.parameters_hash,
            provenance.seed,
        )
        if actual != expected:
            raise FusionRuntimeError("generator provenance does not match run configuration")
        if (
            expected_runtime_parameters_hash is not None
            and provenance.runtime_parameters_hash != expected_runtime_parameters_hash
        ):
            raise FusionRuntimeError(
                "generator runtime parameters do not match the configured attestation"
            )
        return response


class WorkspaceBenchmarkRunner:
    """Generate and evaluate a fail-closed paired OFF/ON candidate batch."""

    def __init__(self, runtime: FusionRuntime, generator: FusionCandidateGenerator) -> None:
        self.runtime = runtime
        self.generator = generator

    def run_pair(
        self,
        *,
        goal: DiscoveryGoal,
        parent_candidate: Candidate,
        cycle: int,
        off_config: WorkspaceRunConfig,
        on_config: WorkspaceRunConfig,
        expert_ids: Iterable[str] | None = None,
        modality: ScientificModality | None = None,
        context_entities: Iterable[WorkspaceEntityInput] | None = None,
        relations: Iterable[WorkspaceRelation] | None = None,
        workspace_id: str | None = None,
    ) -> WorkspacePairedRunReport:
        try:
            off_config = WorkspaceRunConfig.model_validate_json(
                off_config.model_dump_json(),
                strict=True,
            )
            on_config = WorkspaceRunConfig.model_validate_json(
                on_config.model_dump_json(),
                strict=True,
            )
        except Exception as exc:
            raise FusionRuntimeError(f"paired run configuration is invalid: {exc}") from exc
        FusionLoopRunner._validate_config(goal, parent_candidate, off_config)
        FusionLoopRunner._validate_config(goal, parent_candidate, on_config)
        if off_config.workspace_mode != WorkspaceMode.OFF:
            raise FusionRuntimeError("OFF run configuration must use workspace OFF")
        if on_config.workspace_mode != WorkspaceMode.ON:
            raise FusionRuntimeError("ON run configuration must use workspace ON")
        bound_runner = FusionLoopRunner(self.runtime, self.generator)
        bound_runner._validate_generator_binding(off_config)
        bound_runner._validate_generator_binding(on_config)
        left = off_config.model_dump(mode="json", exclude={"workspace_mode"})
        right = on_config.model_dump(mode="json", exclude={"workspace_mode"})
        if left != right:
            raise FusionRuntimeError("paired OFF/ON run configurations must otherwise match exactly")

        contexts = list(context_entities or [])
        relation_rows = list(relations or [])
        requested_experts = None if expert_ids is None else list(expert_ids)
        off_before = self.runtime.update(
            goal=goal,
            candidate=parent_candidate,
            cycle=cycle,
            seed=off_config.seed,
            workspace_mode=WorkspaceMode.OFF,
            expert_ids=requested_experts,
            modality=modality,
            context_entities=contexts,
            relations=relation_rows,
            workspace_id=workspace_id,
            propose_revision=False,
        )
        on_before = self.runtime.update(
            goal=goal,
            candidate=parent_candidate,
            cycle=cycle,
            seed=on_config.seed,
            workspace_mode=WorkspaceMode.ON,
            expert_ids=requested_experts,
            modality=modality,
            context_entities=contexts,
            relations=relation_rows,
            workspace_id=off_before.workspace.workspace_id,
            propose_revision=True,
            decision_context=FusionDecisionContext(
                guidance_alpha=on_config.generation_controls.alpha,
            ),
        )
        if on_before.latent_state is None or on_before.revision_proposal is None:
            raise FusionRuntimeError("workspace ON did not produce state and revision")

        off_generation = bound_runner._call_generator(
            FusionGenerationRequest(
                goal=goal,
                parent_candidate=parent_candidate,
                workspace=off_before.workspace,
                workspace_mode=WorkspaceMode.OFF,
                run_config=off_config,
            )
        )
        on_generation = bound_runner._call_generator(
            FusionGenerationRequest(
                goal=goal,
                parent_candidate=parent_candidate,
                workspace=on_before.workspace,
                workspace_mode=WorkspaceMode.ON,
                run_config=on_config,
                revision_proposal=on_before.revision_proposal,
                latent_state=on_before.latent_state,
                latent_payload=self.runtime.materialize_latent(on_before.latent_state),
            )
        )
        off_generation = FusionLoopRunner._validated_generation(
            off_generation,
            parent_candidate,
            off_config,
            goal,
            expected_runtime_parameters_hash=getattr(
                self.generator,
                "expected_runtime_parameters_hash",
                None,
            ),
        )
        on_generation = FusionLoopRunner._validated_generation(
            on_generation,
            parent_candidate,
            on_config,
            goal,
            expected_runtime_parameters_hash=getattr(
                self.generator,
                "expected_runtime_parameters_hash",
                None,
            ),
        )

        off_runtime_hash = off_generation.provenance.runtime_parameters_hash
        on_runtime_hash = on_generation.provenance.runtime_parameters_hash
        if (
            off_runtime_hash is None
            or on_runtime_hash is None
            or off_runtime_hash != on_runtime_hash
        ):
            raise FusionRuntimeError(
                "paired OFF/ON generation requires equal attested runtime parameters"
            )
        off_by_slot = self._pairing_map(off_generation, off_config, label="OFF")
        on_by_slot = self._pairing_map(on_generation, on_config, label="ON")
        if set(off_by_slot) != set(on_by_slot):
            raise FusionRuntimeError("paired OFF/ON generation returned different pair-slot sets")
        off_metadata = {item.pair_slot: item for item in off_generation.pair_slots}
        on_metadata = {item.pair_slot: item for item in on_generation.pair_slots}
        ordered_slots = sorted(off_by_slot)
        if any(
            off_metadata[slot].stream_position != on_metadata[slot].stream_position
            for slot in ordered_slots
        ):
            raise FusionRuntimeError(
                "paired OFF/ON generation returned different batch stream positions"
            )
        off_candidates = [off_by_slot[slot] for slot in ordered_slots]
        on_candidates = [on_by_slot[slot] for slot in ordered_slots]

        off_after = [
            self.runtime.update(
                goal=goal,
                candidate=candidate,
                cycle=cycle + 1,
                seed=off_config.seed,
                workspace_mode=WorkspaceMode.OFF,
                expert_ids=requested_experts,
                modality=modality,
                context_entities=contexts,
                relations=relation_rows,
                workspace_id=off_before.workspace.workspace_id,
                propose_revision=False,
            )
            for candidate in off_candidates
        ]
        on_after = [
            self.runtime.update(
                goal=goal,
                candidate=candidate,
                cycle=cycle + 1,
                seed=on_config.seed,
                workspace_mode=WorkspaceMode.ON,
                previous_state=on_before.latent_state,
                expert_ids=requested_experts,
                modality=modality,
                context_entities=contexts,
                relations=relation_rows,
                workspace_id=on_before.workspace.workspace_id,
                propose_revision=False,
                decision_context=FusionDecisionContext(
                    guidance_alpha=on_config.generation_controls.alpha,
                ),
            )
            for candidate in on_candidates
        ]
        off_snapshots = [
            self.runtime.snapshot(candidate, report, off_config)
            for candidate, report in zip(
                off_candidates,
                off_after,
                strict=True,
            )
        ]
        on_snapshots = [
            self.runtime.snapshot(candidate, report, on_config)
            for candidate, report in zip(
                on_candidates,
                on_after,
                strict=True,
            )
        ]
        comparisons = [
            compare_workspace_snapshots(
                off_snapshot,
                on_snapshot,
                goal,
                artifact_store=self.runtime.artifact_store,
            )
            for off_snapshot, on_snapshot in zip(
                off_snapshots,
                on_snapshots,
                strict=True,
            )
        ]
        report = WorkspacePairedRunReport(
            off_generation=off_generation,
            on_generation=on_generation,
            off_snapshots=off_snapshots,
            on_snapshots=on_snapshots,
            comparisons=comparisons,
        )
        digest = stable_hash(report)
        self.runtime.artifact_store.write_json(
            f"fusion/comparisons/PAIR-{digest[:24]}.json",
            report,
        )
        return report

    @staticmethod
    def _pairing_map(
        generation: FusionGenerationResponse,
        config: WorkspaceRunConfig,
        *,
        label: str,
    ) -> dict[int, Candidate]:
        candidates = generation.generated_candidates
        if len(generation.pair_slots) != len(candidates):
            raise FusionRuntimeError(
                f"paired {label} generation requires pair-slot metadata for every candidate"
            )
        result = generation.candidates_by_pair_slot
        expected = set(range(config.candidate_count))
        if set(result) != expected:
            raise FusionRuntimeError(
                f"paired {label} generation returned missing or duplicate pair slots"
            )
        if any(
            item.batch_seed != config.effective_generator_seed
            for item in generation.pair_slots
        ):
            raise FusionRuntimeError(
                f"paired {label} generation pair slots cite the wrong batch seed"
            )
        return result


__all__ = ["FusionLoopRunner", "WorkspaceBenchmarkRunner"]
