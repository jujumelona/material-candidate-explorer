"""Execution runtime for specialist features and the configured fusion backend.

The runtime owns lineage, routing, artifact integrity, and OFF/ON boundaries.
Specialist services and the fusion model only receive validated copies of wire
objects and cannot directly mutate discovery records.
"""

from __future__ import annotations

from collections.abc import Iterable

from .artifacts import ArtifactStore
from .fusion_metrics import aggregate_diagnostic_properties
from .fusion_protocols import ExpertEncoder, FusionBackend
from .fusion_registry import ExpertRegistry
from .fusion_schemas import (
    ContentArtifactRef,
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRef,
    ExpertFeatureRequest,
    ExpertRoute,
    FeatureStatus,
    FusionCycleReport,
    FusionDecisionContext,
    FusionFeatureInput,
    FusionOutput,
    FusionRequest,
    FusionRevisionProposal,
    FusionRevisionRequest,
    FusionWorkspaceSnapshot,
    NumericTensor,
    ScientificModality,
    ScientificWorkspace,
    UnifiedLatentStateRef,
    UnifiedLatentStateRecord,
    WorkspaceEntity,
    WorkspaceEntityInput,
    WorkspaceEntityRole,
    WorkspaceMode,
    WorkspaceRelation,
    WorkspaceRunConfig,
)
from .hashing import candidate_content_hash, canonical_json, stable_hash
from .schemas import Candidate, DiscoveryGoal


class FusionRuntimeError(RuntimeError):
    pass


class FusionRuntime:
    def __init__(
        self,
        registry: ExpertRegistry,
        backend: FusionBackend,
        artifact_store: ArtifactStore,
        *,
        allow_backend_change: bool = False,
        allow_latent_shape_change: bool = False,
    ) -> None:
        self.registry = registry
        self.backend = backend
        self.artifact_store = artifact_store
        self.allow_backend_change = allow_backend_change
        self.allow_latent_shape_change = allow_latent_shape_change

    def update(
        self,
        *,
        goal: DiscoveryGoal,
        candidate: Candidate,
        cycle: int,
        seed: int,
        workspace_mode: WorkspaceMode,
        decision_context: FusionDecisionContext | None = None,
        previous_state: UnifiedLatentStateRef | None = None,
        expert_ids: Iterable[str] | None = None,
        modality: ScientificModality | None = None,
        propose_revision: bool = True,
        context_entities: Iterable[WorkspaceEntityInput] | None = None,
        relations: Iterable[WorkspaceRelation] | None = None,
        workspace_id: str | None = None,
    ) -> FusionCycleReport:
        if cycle < 0 or seed < 0:
            raise ValueError("cycle and seed must be non-negative")
        if workspace_mode == WorkspaceMode.OFF and previous_state is not None:
            raise FusionRuntimeError("workspace OFF cannot consume a latent state")

        try:
            goal_copy = DiscoveryGoal.model_validate_json(goal.model_dump_json(), strict=True)
            decision_context = FusionDecisionContext.model_validate_json(
                (decision_context or FusionDecisionContext()).model_dump_json(),
                strict=True,
            )
            if previous_state is not None:
                previous_state = UnifiedLatentStateRef.model_validate_json(
                    previous_state.model_dump_json(),
                    strict=True,
                )
        except Exception as exc:
            raise FusionRuntimeError(f"fusion input contract is invalid: {exc}") from exc
        goal_hash = stable_hash(goal_copy)
        primary = self._validated_candidate_copy(candidate, label="primary candidate")
        contexts = [
            WorkspaceEntityInput.model_validate_json(item.model_dump_json(), strict=True)
            for item in (context_entities or [])
        ]
        for item in contexts:
            if item.role == WorkspaceEntityRole.PRIMARY_CANDIDATE:
                raise FusionRuntimeError("context entities cannot use primary_candidate role")
            self._validated_candidate_copy(
                item.candidate,
                label=f"workspace entity {item.entity_id!r}",
            )
        relation_rows = [
            WorkspaceRelation.model_validate_json(item.model_dump_json(), strict=True)
            for item in (relations or [])
        ]
        workspace = self._build_workspace(
            goal_hash=goal_hash,
            primary=primary,
            contexts=contexts,
            relations=relation_rows,
            workspace_id=workspace_id,
            previous_state=previous_state,
        )
        if previous_state is not None:
            self._validate_previous_state(
                previous_state,
                goal_hash=goal_hash,
                primary=primary,
                workspace=workspace,
                cycle=cycle,
                seed=seed,
            )

        requested = None if expert_ids is None else list(dict.fromkeys(expert_ids))
        feature_inputs: list[FusionFeatureInput] = []
        feature_refs: list[ExpertFeatureRef] = []
        failed: set[str] = set()
        matched: set[str] = set()
        primary_failed: set[str] = set()
        primary_matched: set[str] = set()
        warnings: list[str] = []

        entities = [
            WorkspaceEntityInput(
                entity_id="primary",
                role=WorkspaceEntityRole.PRIMARY_CANDIDATE,
                candidate=primary,
            ),
            *contexts,
        ]
        for entity in entities:
            compatible = self.registry.compatible(entity.candidate, modality=modality)
            if requested is not None:
                compatible = [
                    item for item in compatible if item.descriptor.expert_id in requested
                ]
            for encoder in compatible:
                descriptor = self.registry.bound_descriptor(encoder)
                try:
                    routes = self._matching_routes(
                        descriptor,
                        entity.candidate,
                        modality=modality,
                    )
                except Exception as exc:
                    failed.add(descriptor.expert_id)
                    if entity.entity_id == workspace.primary_entity_id:
                        primary_failed.add(descriptor.expert_id)
                    warnings.append(
                        f"{entity.entity_id}/{descriptor.expert_id}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if not routes:
                    continue
                matched.add(descriptor.expert_id)
                if entity.entity_id == workspace.primary_entity_id:
                    primary_matched.add(descriptor.expert_id)
                for selected_modality, feature_space in routes:
                    request = ExpertFeatureRequest(
                        workspace_entity_id=entity.entity_id,
                        candidate=Candidate.model_validate(
                            entity.candidate.model_dump(mode="json")
                        ),
                        goal=DiscoveryGoal.model_validate(goal_copy.model_dump(mode="json")),
                        modality=selected_modality,
                        feature_space=feature_space,
                        cycle=cycle,
                        seed=seed,
                    )
                    request_hash = stable_hash(request)
                    try:
                        cached = self._load_cached_feature(request, descriptor)
                        if cached is None:
                            payload = encoder.encode(request)
                            self.registry.bound_descriptor(encoder)
                            if stable_hash(request) != request_hash:
                                raise FusionRuntimeError("encoder mutated its request")
                            payload = self._validate_feature_payload(
                                payload,
                                request,
                                descriptor,
                            )
                            feature_ref = self._persist_feature(
                                payload,
                                goal_hash=stable_hash(request.goal),
                            )
                            self._persist_feature_cache(request, descriptor, feature_ref)
                        else:
                            payload, feature_ref = cached
                    except FusionRuntimeError:
                        raise
                    except Exception as exc:
                        failed.add(descriptor.expert_id)
                        if entity.entity_id == workspace.primary_entity_id:
                            primary_failed.add(descriptor.expert_id)
                        warnings.append(
                            f"{entity.entity_id}/{descriptor.expert_id}/"
                            f"{selected_modality}: {type(exc).__name__}: {exc}"
                        )
                        continue
                    feature_refs.append(feature_ref)
                    if payload.status != FeatureStatus.FAILED and payload.tensor is not None:
                        feature_inputs.append(
                            FusionFeatureInput(
                                feature_id=feature_ref.feature_id,
                                workspace_entity_id=entity.entity_id,
                                payload=payload,
                            )
                        )
                        if payload.status == FeatureStatus.PARTIAL:
                            warnings.append(
                                f"{entity.entity_id}/{descriptor.expert_id}: partial feature used."
                            )
                    else:
                        failed.add(descriptor.expert_id)
                        if entity.entity_id == workspace.primary_entity_id:
                            primary_failed.add(descriptor.expert_id)

        missing = sorted(set(requested or []) - matched)
        primary_usable_experts = {
            item.payload.expert_id
            for item in feature_inputs
            if item.workspace_entity_id == workspace.primary_entity_id
        }
        primary_failed_ids = sorted(primary_failed - primary_usable_experts)
        primary_missing_ids = sorted(
            set(requested or []) - primary_matched - primary_failed
        )
        if stable_hash(goal) != goal_hash:
            raise FusionRuntimeError("an expert mutated the source goal")
        if candidate_content_hash(candidate) != candidate.candidate_ref.content_hash:
            raise FusionRuntimeError("an expert mutated the source candidate")

        if workspace_mode == WorkspaceMode.OFF:
            report = FusionCycleReport(
                candidate_ref=primary.candidate_ref,
                workspace=workspace,
                goal_hash=goal_hash,
                workspace_mode=workspace_mode,
                cycle=cycle,
                feature_refs=feature_refs,
                missing_expert_ids=missing,
                failed_expert_ids=sorted(failed),
                warnings=warnings,
            )
            self._persist_report(report)
            return report

        primary_ref = primary.candidate_ref
        if primary_ref is None:  # guarded by _validated_candidate_copy
            raise FusionRuntimeError("primary candidate is missing candidate_ref")
        if not any(
            item.workspace_entity_id == workspace.primary_entity_id
            and item.payload.candidate_ref == primary_ref
            for item in feature_inputs
        ):
            raise FusionRuntimeError(
                "workspace ON requires at least one successful primary-candidate feature"
            )
        previous_latent = (
            self._load_latent(previous_state) if previous_state is not None else None
        )
        fusion_request = FusionRequest(
            goal=goal_copy,
            candidate_ref=primary_ref,
            workspace=workspace,
            workspace_mode=workspace_mode,
            cycle=cycle,
            seed=seed,
            features=feature_inputs,
            decision_context=decision_context,
            failed_expert_ids=primary_failed_ids,
            missing_expert_ids=primary_missing_ids,
            previous_latent=previous_latent,
            previous_state_id=previous_state.state_id if previous_state else None,
        )
        backend_request = FusionRequest.model_validate(
            fusion_request.model_dump(mode="json")
        )
        request_hash = stable_hash(backend_request)
        output = self.backend.fuse(backend_request)
        if stable_hash(backend_request) != request_hash:
            raise FusionRuntimeError("fusion backend mutated its request")
        if not isinstance(output, FusionOutput):
            raise FusionRuntimeError("fusion backend returned the wrong response type")
        try:
            output = FusionOutput.model_validate_json(
                output.model_dump_json(),
                strict=True,
            )
        except Exception as exc:
            raise FusionRuntimeError(f"fusion backend returned invalid output: {exc}") from exc
        input_ids = {item.feature_id for item in feature_inputs}
        accounted = set(output.used_feature_ids) | set(output.ignored_feature_ids)
        if accounted != input_ids:
            raise FusionRuntimeError("fusion backend did not account for every feature")
        self._validate_backend_continuity(output, previous_state)

        state = self._persist_latent(
            output,
            request=fusion_request,
            previous_state=previous_state,
        )
        revision: FusionRevisionProposal | None = None
        if propose_revision:
            revision_request = FusionRevisionRequest(
                goal=goal_copy,
                candidate=primary,
                state=state,
                latent=output.latent,
                features=feature_inputs,
                decision_context=decision_context,
            )
            backend_revision_request = FusionRevisionRequest.model_validate(
                revision_request.model_dump(mode="json")
            )
            revision_hash = stable_hash(backend_revision_request)
            revision = self.backend.propose_revision(backend_revision_request)
            if stable_hash(backend_revision_request) != revision_hash:
                raise FusionRuntimeError("fusion backend mutated its revision request")
            if not isinstance(revision, FusionRevisionProposal):
                raise FusionRuntimeError("fusion backend returned the wrong revision type")
            try:
                revision = FusionRevisionProposal.model_validate_json(
                    revision.model_dump_json(),
                    strict=True,
                )
            except Exception as exc:
                raise FusionRuntimeError(
                    f"fusion backend returned an invalid revision: {exc}"
                ) from exc
            if revision.parent_candidate_ref != primary_ref or revision.state_id != state.state_id:
                raise FusionRuntimeError(
                    "fusion revision references stale state or candidate content"
                )

        report = FusionCycleReport(
            candidate_ref=primary_ref,
            workspace=workspace,
            goal_hash=goal_hash,
            workspace_mode=workspace_mode,
            cycle=cycle,
            feature_refs=feature_refs,
            latent_state=state,
            revision_proposal=revision,
            missing_expert_ids=missing,
            failed_expert_ids=sorted(failed),
            warnings=warnings + output.warnings,
        )
        self._persist_report(report)
        return report

    def snapshot(
        self,
        candidate: Candidate,
        report: FusionCycleReport,
        run_config: WorkspaceRunConfig,
    ) -> FusionWorkspaceSnapshot:
        try:
            report = FusionCycleReport.model_validate_json(
                report.model_dump_json(),
                strict=True,
            )
            run_config = WorkspaceRunConfig.model_validate_json(
                run_config.model_dump_json(),
                strict=True,
            )
        except Exception as exc:
            raise FusionRuntimeError(f"snapshot input contract is invalid: {exc}") from exc
        verified_candidate = self._validated_candidate_copy(candidate, label="snapshot candidate")
        if verified_candidate.candidate_ref != report.candidate_ref:
            raise FusionRuntimeError("report does not belong to the candidate")
        if run_config.workspace_mode != report.workspace_mode:
            raise FusionRuntimeError("run configuration mode does not match report")
        if run_config.goal_hash != report.goal_hash:
            raise FusionRuntimeError("run configuration goal does not match report")
        if report.latent_state is not None:
            self.materialize_latent(report.latent_state)

        verified_refs: list[ExpertFeatureRef] = []
        property_groups: list[list] = []
        for feature_ref in report.feature_refs:
            encoded = self.artifact_store.read_bytes(
                feature_ref.artifact.relative_path,
                expected_sha256=feature_ref.artifact.sha256,
            )
            payload = ExpertFeaturePayload.model_validate_json(encoded, strict=True)
            expected_ref = self._feature_ref_from_payload(
                payload,
                artifact=feature_ref.artifact,
                goal_hash=report.goal_hash,
            )
            if (
                len(encoded) != feature_ref.artifact.byte_size
                or feature_ref.artifact.artifact_id
                != f"ART-{feature_ref.artifact.sha256[:24]}"
                or feature_ref.artifact.media_type
                != "application/vnd.discovery-os.expert-feature+json"
                or expected_ref != feature_ref
            ):
                raise FusionRuntimeError("feature artifact does not match its reference")
            verified_refs.append(feature_ref)
            # Goal objectives are scoped to the primary candidate by default.
            # Context/target properties remain preserved in raw feature
            # artifacts and must not be silently averaged into primary scores.
            if (
                payload.status != FeatureStatus.FAILED
                and payload.workspace_entity_id == report.workspace.primary_entity_id
            ):
                property_groups.append(payload.properties)
        properties, aggregation_warnings = aggregate_diagnostic_properties(property_groups)
        snapshot = FusionWorkspaceSnapshot(
            candidate=verified_candidate,
            workspace=report.workspace,
            goal_hash=report.goal_hash,
            mode=report.workspace_mode,
            feature_refs=verified_refs,
            latent_state=report.latent_state,
            aggregate_properties=properties,
            run_config=run_config,
            missing_expert_ids=report.missing_expert_ids,
            failed_expert_ids=report.failed_expert_ids,
        )
        digest = stable_hash(snapshot)
        self.artifact_store.write_json(
            f"fusion/snapshots/SNAP-{digest[:24]}.json",
            snapshot,
        )
        if aggregation_warnings:
            # Warnings remain diagnostic; snapshot schema intentionally contains
            # only values that are safe to compare.
            self.artifact_store.write_json(
                f"fusion/snapshots/SNAP-{digest[:24]}.warnings.json",
                {"schema_version": "1.0", "warnings": aggregation_warnings},
            )
        return snapshot

    @staticmethod
    def _validated_candidate_copy(candidate: Candidate, *, label: str) -> Candidate:
        try:
            candidate_copy = Candidate.model_validate_json(
                candidate.model_dump_json(),
                strict=True,
            )
        except Exception as exc:
            raise FusionRuntimeError(
                f"{label} is invalid or has a stale candidate_ref: {exc}"
            ) from exc
        candidate_ref = candidate_copy.candidate_ref
        if candidate_ref is None:
            raise FusionRuntimeError(f"{label} requires an immutable candidate_ref")
        if candidate_content_hash(candidate_copy) != candidate_ref.content_hash:
            raise FusionRuntimeError(f"{label} candidate_ref content hash is stale")
        return candidate_copy

    def _build_workspace(
        self,
        *,
        goal_hash: str,
        primary: Candidate,
        contexts: list[WorkspaceEntityInput],
        relations: list[WorkspaceRelation],
        workspace_id: str | None,
        previous_state: UnifiedLatentStateRef | None,
    ) -> ScientificWorkspace:
        if any(item.entity_id == "primary" for item in contexts):
            raise FusionRuntimeError("workspace entity id 'primary' is reserved")
        context_ids = [item.entity_id for item in contexts]
        if len(context_ids) != len(set(context_ids)):
            raise FusionRuntimeError("duplicate workspace context entity ids")
        resolved_id = workspace_id
        if previous_state is not None:
            if resolved_id is not None and resolved_id != previous_state.workspace_id:
                raise FusionRuntimeError("workspace_id does not match previous latent state")
            resolved_id = previous_state.workspace_id
        if resolved_id is None:
            resolved_id = "WS-" + stable_hash(
                {
                    "goal_hash": goal_hash,
                    "context": [
                        {
                            "entity_id": item.entity_id,
                            "role": item.role,
                            "candidate_ref": item.candidate.candidate_ref,
                        }
                        for item in contexts
                    ],
                    "relations": relations,
                }
            )[:24]
        primary_ref = primary.candidate_ref
        if primary_ref is None:
            raise FusionRuntimeError("workspace primary candidate is missing its reference")
        return ScientificWorkspace(
            workspace_id=resolved_id,
            primary_entity_id="primary",
            entities=[
                WorkspaceEntity(
                    entity_id="primary",
                    role=WorkspaceEntityRole.PRIMARY_CANDIDATE,
                    candidate_ref=primary_ref,
                ),
                *[
                    WorkspaceEntity(
                        entity_id=item.entity_id,
                        role=item.role,
                        candidate_ref=item.candidate.candidate_ref,
                    )
                    for item in contexts
                ],
            ],
            relations=relations,
        )

    def _validate_previous_state(
        self,
        state: UnifiedLatentStateRef,
        *,
        goal_hash: str,
        primary: Candidate,
        workspace: ScientificWorkspace,
        cycle: int,
        seed: int,
    ) -> None:
        self._load_state_record(state)
        if state.goal_hash != goal_hash:
            raise FusionRuntimeError("previous latent state belongs to a different goal")
        if state.seed != seed:
            raise FusionRuntimeError("previous latent state uses a different seed")
        if cycle != state.cycle + 1:
            raise FusionRuntimeError("fusion cycles must advance by exactly one")
        primary_ref = primary.candidate_ref
        if primary_ref != state.candidate_ref and state.candidate_ref not in primary.parent_candidate_refs:
            raise FusionRuntimeError(
                "previous latent state is unrelated to the current candidate lineage"
            )
        if (
            primary_ref != state.candidate_ref
            and primary_ref is not None
            and primary_ref.candidate_id == state.candidate_ref.candidate_id
            and primary_ref.version <= state.candidate_ref.version
        ):
            raise FusionRuntimeError("same-id candidate lineage cannot roll back its version")
        if state.workspace_id != workspace.workspace_id:
            raise FusionRuntimeError("previous latent state belongs to a different workspace")
        prior_context = sorted(
            (
                item.entity_id,
                str(item.role),
                item.candidate_ref.model_dump(mode="json"),
            )
            for item in state.workspace_entities
            if item.role != WorkspaceEntityRole.PRIMARY_CANDIDATE
        )
        current_context = sorted(
            (
                item.entity_id,
                str(item.role),
                item.candidate_ref.model_dump(mode="json"),
            )
            for item in workspace.entities
            if item.role != WorkspaceEntityRole.PRIMARY_CANDIDATE
        )
        if prior_context != current_context:
            raise FusionRuntimeError("workspace context changed across a latent-state lineage")
        if state.workspace_relations != workspace.relations:
            raise FusionRuntimeError("workspace relations changed across a latent-state lineage")

    @staticmethod
    def _matching_routes(
        descriptor: ExpertDescriptor,
        candidate: Candidate,
        *,
        modality: ScientificModality | None,
    ) -> list[tuple[ScientificModality, str]]:
        kinds = {item.kind for item in candidate.representations}
        if descriptor.routes:
            rows = [
                route
                for route in descriptor.routes
                if (modality is None or route.modality == modality)
                and (not route.candidate_types or candidate.candidate_type in route.candidate_types)
                and bool(kinds.intersection(route.representation_kinds))
            ]
            return list(
                dict.fromkeys((row.modality, row.feature_space) for row in rows)
            )
        modalities = (
            [modality]
            if modality is not None and modality in descriptor.modalities
            else list(descriptor.modalities)
        )
        if len(modalities) != 1 or len(descriptor.feature_spaces) != 1:
            raise FusionRuntimeError(
                "multi-route expert requires explicit ExpertRoute declarations"
            )
        return [(modalities[0], descriptor.feature_spaces[0])]

    @staticmethod
    def _validate_feature_payload(
        payload: ExpertFeaturePayload,
        request: ExpertFeatureRequest,
        descriptor: ExpertDescriptor,
    ) -> ExpertFeaturePayload:
        if not isinstance(payload, ExpertFeaturePayload):
            raise TypeError("encoder returned the wrong response type")
        payload = ExpertFeaturePayload.model_validate_json(
            payload.model_dump_json(),
            strict=True,
        )
        if candidate_content_hash(request.candidate) != request.candidate.candidate_ref.content_hash:
            raise ValueError("encoder mutated candidate content")
        if payload.candidate_ref != request.candidate.candidate_ref:
            raise ValueError("encoder returned a stale candidate_ref")
        if payload.workspace_entity_id != request.workspace_entity_id:
            raise ValueError("encoder returned the wrong workspace_entity_id")
        if payload.expert_id != descriptor.expert_id:
            raise ValueError("encoder returned the wrong expert_id")
        if payload.modality != request.modality:
            raise ValueError("encoder returned the wrong modality")
        if payload.feature_space != request.feature_space:
            raise ValueError("encoder returned the wrong feature_space")
        provenance = payload.provenance
        if provenance.expert_id != descriptor.expert_id:
            raise ValueError("feature provenance returned the wrong expert_id")
        if provenance.adapter_version != descriptor.adapter_version:
            raise ValueError("feature provenance adapter version is inconsistent")
        if provenance.seed != request.seed:
            raise ValueError("feature provenance seed is inconsistent")
        if (
            payload.semantics is not None
            and provenance.projection_version is not None
            and provenance.projection_version != payload.semantics.projection_id
        ):
            raise ValueError("feature projection provenance is inconsistent")
        for metadata_key, provenance_name in (
            ("model_version", "model_version"),
            ("code_revision", "code_revision"),
            ("weight_revision", "weight_revision"),
            ("parameters_hash", "parameters_hash"),
        ):
            expected = descriptor.metadata.get(metadata_key)
            if expected is not None and expected != getattr(provenance, provenance_name):
                raise ValueError(f"feature provenance {provenance_name} is inconsistent")
        return payload

    def _persist_feature(
        self,
        payload: ExpertFeaturePayload,
        *,
        goal_hash: str,
    ) -> ExpertFeatureRef:
        digest = stable_hash(payload)
        feature_id = f"FEAT-{digest[:24]}"
        relative_path, sha256 = self.artifact_store.write_json(
            f"fusion/features/{feature_id}.json",
            payload,
        )
        byte_size = len(
            self.artifact_store.read_bytes(relative_path, expected_sha256=sha256)
        )
        artifact = ContentArtifactRef(
            artifact_id=f"ART-{sha256[:24]}",
            relative_path=relative_path,
            sha256=sha256,
            media_type="application/vnd.discovery-os.expert-feature+json",
            byte_size=byte_size,
        )
        return self._feature_ref_from_payload(
            payload,
            artifact=artifact,
            goal_hash=goal_hash,
        )

    @staticmethod
    def _feature_cache_key(
        request: ExpertFeatureRequest,
        descriptor: ExpertDescriptor,
    ) -> str:
        # cycle is deliberately absent: a specialist evaluation is a pure
        # function of immutable content, goal, route, model binding, and seed.
        return stable_hash(
            {
                "contract": "expert-feature-cache-v1",
                "workspace_entity_id": request.workspace_entity_id,
                "candidate_ref": request.candidate.candidate_ref,
                "goal_hash": stable_hash(request.goal),
                "modality": request.modality,
                "feature_space": request.feature_space,
                "seed": request.seed,
                "descriptor": descriptor,
            }
        )

    def _persist_feature_cache(
        self,
        request: ExpertFeatureRequest,
        descriptor: ExpertDescriptor,
        feature_ref: ExpertFeatureRef,
    ) -> None:
        cache_key = self._feature_cache_key(request, descriptor)
        self.artifact_store.write_json(
            f"fusion/cache/features/{cache_key}.json",
            {
                "schema_version": "1.0",
                "cache_key": cache_key,
                "feature_ref": feature_ref,
            },
        )

    def _load_cached_feature(
        self,
        request: ExpertFeatureRequest,
        descriptor: ExpertDescriptor,
    ) -> tuple[ExpertFeaturePayload, ExpertFeatureRef] | None:
        cache_key = self._feature_cache_key(request, descriptor)
        relative_path = f"fusion/cache/features/{cache_key}.json"
        if not self.artifact_store.resolve(relative_path).exists():
            return None
        try:
            record = self.artifact_store.read_json(relative_path)
            if (
                not isinstance(record, dict)
                or record.get("schema_version") != "1.0"
                or record.get("cache_key") != cache_key
            ):
                raise FusionRuntimeError("feature cache record is invalid")
            feature_ref = ExpertFeatureRef.model_validate_json(
                canonical_json(record.get("feature_ref")),
                strict=True,
            )
            encoded = self.artifact_store.read_bytes(
                feature_ref.artifact.relative_path,
                expected_sha256=feature_ref.artifact.sha256,
            )
            payload = ExpertFeaturePayload.model_validate_json(encoded, strict=True)
            payload = self._validate_feature_payload(payload, request, descriptor)
            expected_ref = self._feature_ref_from_payload(
                payload,
                artifact=feature_ref.artifact,
                goal_hash=stable_hash(request.goal),
            )
            if (
                len(encoded) != feature_ref.artifact.byte_size
                or expected_ref != feature_ref
            ):
                raise FusionRuntimeError("feature cache artifact does not match its reference")
            return payload, feature_ref
        except FusionRuntimeError:
            raise
        except Exception as exc:
            raise FusionRuntimeError(f"feature cache validation failed: {exc}") from exc

    @staticmethod
    def _feature_ref_from_payload(
        payload: ExpertFeaturePayload,
        *,
        artifact: ContentArtifactRef,
        goal_hash: str,
    ) -> ExpertFeatureRef:
        digest = stable_hash(payload)
        feature_id = f"FEAT-{digest[:24]}"
        return ExpertFeatureRef(
            feature_id=feature_id,
            workspace_entity_id=payload.workspace_entity_id,
            candidate_ref=payload.candidate_ref,
            goal_hash=goal_hash,
            expert_id=payload.expert_id,
            modality=payload.modality,
            feature_space=payload.feature_space,
            status=payload.status,
            artifact=artifact,
            tensor_dtype=payload.tensor.dtype if payload.tensor is not None else None,
            tensor_shape=list(payload.tensor.shape) if payload.tensor is not None else [],
            semantics=payload.semantics,
            properties=payload.properties,
            quality_flags=payload.quality_flags,
            warnings=payload.warnings,
            provenance=payload.provenance,
        )

    def _validate_backend_continuity(
        self,
        output: FusionOutput,
        previous_state: UnifiedLatentStateRef | None,
    ) -> None:
        if previous_state is None:
            return
        old_backend = (
            previous_state.backend_id,
            previous_state.backend_version,
            previous_state.code_revision,
            previous_state.weight_revision,
        )
        new_backend = (
            output.backend_id,
            output.backend_version,
            output.code_revision,
            output.weight_revision,
        )
        if not self.allow_backend_change and old_backend != new_backend:
            raise FusionRuntimeError("fusion backend provenance changed within a state lineage")
        if not self.allow_latent_shape_change and (
            previous_state.dtype != output.latent.dtype
            or previous_state.shape != output.latent.shape
        ):
            raise FusionRuntimeError("latent dtype or shape changed within a state lineage")

    def _persist_latent(
        self,
        output: FusionOutput,
        *,
        request: FusionRequest,
        previous_state: UnifiedLatentStateRef | None,
    ) -> UnifiedLatentStateRef:
        # The tensor and all state metadata are independently content-addressed.
        latent_digest = stable_hash(output.latent)
        provisional_id = f"LATENT-TENSOR-{latent_digest[:24]}"
        relative_path, sha256 = self.artifact_store.write_json(
            f"fusion/latents/{provisional_id}.json",
            output.latent,
        )
        byte_size = len(
            self.artifact_store.read_bytes(relative_path, expected_sha256=sha256)
        )
        artifact = ContentArtifactRef(
            artifact_id=f"ART-{sha256[:24]}",
            relative_path=relative_path,
            sha256=sha256,
            media_type="application/vnd.discovery-os.latent+json",
            byte_size=byte_size,
        )
        record = UnifiedLatentStateRecord(
            state_version=(previous_state.state_version + 1 if previous_state else 1),
            candidate_ref=request.candidate_ref,
            workspace_id=request.workspace.workspace_id,
            workspace_entities=request.workspace.entities,
            workspace_relations=request.workspace.relations,
            cycle=request.cycle,
            latent_artifact=artifact,
            latent_content_hash=stable_hash(output.latent),
            dtype=output.latent.dtype,
            shape=output.latent.shape,
            source_feature_ids=output.used_feature_ids,
            previous_state_id=previous_state.state_id if previous_state else None,
            goal_hash=stable_hash(request.goal),
            seed=request.seed,
            backend_id=output.backend_id,
            backend_version=output.backend_version,
            code_revision=output.code_revision,
            weight_revision=output.weight_revision,
            warnings=output.warnings,
        )
        record_digest = stable_hash(record)
        state_id = f"LATENT-{record_digest[:24]}"
        state_path, state_sha256 = self.artifact_store.write_json(
            f"fusion/states/{state_id}.json",
            record,
        )
        state_bytes = self.artifact_store.read_bytes(
            state_path,
            expected_sha256=state_sha256,
        )
        state_artifact = ContentArtifactRef(
            artifact_id=f"ART-{state_sha256[:24]}",
            relative_path=state_path,
            sha256=state_sha256,
            media_type="application/vnd.discovery-os.latent-state+json",
            byte_size=len(state_bytes),
        )
        return UnifiedLatentStateRef(
            **record.model_dump(mode="python"),
            state_id=state_id,
            state_artifact=state_artifact,
        )

    def _load_state_record(self, state: UnifiedLatentStateRef) -> UnifiedLatentStateRecord:
        artifact = state.state_artifact
        if artifact is None:
            raise FusionRuntimeError("latent state is missing its durable state artifact")
        encoded = self.artifact_store.read_bytes(
            artifact.relative_path,
            expected_sha256=artifact.sha256,
        )
        if (
            len(encoded) != artifact.byte_size
            or artifact.artifact_id != f"ART-{artifact.sha256[:24]}"
            or artifact.media_type != "application/vnd.discovery-os.latent-state+json"
        ):
            raise FusionRuntimeError("latent state artifact metadata does not match")
        record = UnifiedLatentStateRecord.model_validate_json(encoded, strict=True)
        expected_id = f"LATENT-{stable_hash(record)[:24]}"
        if state.state_id != expected_id:
            raise FusionRuntimeError("latent state id does not match its state artifact")
        state_values = state.model_dump(
            mode="json",
            exclude={"state_id", "state_artifact"},
        )
        if state_values != record.model_dump(mode="json"):
            raise FusionRuntimeError("latent state metadata does not match its state artifact")
        return record

    def _load_latent(self, state: UnifiedLatentStateRef) -> NumericTensor:
        encoded = self.artifact_store.read_bytes(
            state.latent_artifact.relative_path,
            expected_sha256=state.latent_artifact.sha256,
        )
        if (
            len(encoded) != state.latent_artifact.byte_size
            or state.latent_artifact.artifact_id
            != f"ART-{state.latent_artifact.sha256[:24]}"
            or state.latent_artifact.media_type
            != "application/vnd.discovery-os.latent+json"
        ):
            raise FusionRuntimeError("latent artifact metadata does not match state")
        tensor = NumericTensor.model_validate_json(encoded, strict=True)
        if tensor.dtype != state.dtype or tensor.shape != state.shape:
            raise FusionRuntimeError("latent artifact metadata does not match state")
        if (
            state.latent_content_hash is None
            or stable_hash(tensor) != state.latent_content_hash
        ):
            raise FusionRuntimeError("latent artifact content hash does not match state")
        return tensor

    def materialize_latent(self, state: UnifiedLatentStateRef) -> NumericTensor:
        """Hash-verify a latent before including it in a sidecar request."""

        self._load_state_record(state)
        return self._load_latent(state)

    def _persist_report(self, report: FusionCycleReport) -> None:
        digest = stable_hash(report)
        self.artifact_store.write_json(
            f"fusion/reports/REPORT-{digest[:24]}.json",
            report,
        )


__all__ = ["FusionRuntime", "FusionRuntimeError"]
