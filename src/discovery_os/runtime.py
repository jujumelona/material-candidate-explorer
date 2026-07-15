"""Sequential, checkpoint-friendly execution of compiled tool plans."""

from __future__ import annotations

import time

from .artifacts import ArtifactStore
from .cache import JsonCache
from .hashing import stable_hash
from .registry import ToolRegistry
from .schemas import (
    Candidate,
    ComputationalEvidenceDetails,
    EvidenceBatch,
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    ToolCall,
    ValidationPlan,
)


class ToolRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        artifact_store: ArtifactStore,
        cache: JsonCache | None = None,
        max_retry_limit: int = 0,
    ) -> None:
        self.registry = registry
        self.artifact_store = artifact_store
        self.cache = cache
        self.max_retry_limit = max_retry_limit

    def execute_plan(
        self,
        plan: ValidationPlan,
        *,
        candidates: list[Candidate],
    ) -> EvidenceBatch:
        candidate_map = {item.candidate_id: item for item in candidates}
        records: list[EvidenceRecord] = []
        plan_warnings: list[str] = []
        call_succeeded: dict[str, bool] = {}
        for call in plan.calls:
            selected = self._selected_candidates(call, candidate_map)
            if any(not call_succeeded.get(dependency, False) for dependency in call.depends_on_call_ids):
                batch = self._failure_batch(
                    call,
                    selected,
                    status=EvidenceStatus.FAILED,
                    failure="dependency_not_satisfied",
                )
            else:
                batch = self._execute_call(call, selected)
            plan_warnings.extend(batch.warnings)
            artifact_batch = batch.model_copy(
                update={
                    "warnings": [
                        warning for warning in batch.warnings if warning != "cache_hit"
                    ]
                }
            )
            artifact_digest = stable_hash(artifact_batch.model_dump(mode="json"))
            artifact_relative, _ = self.artifact_store.write_json(
                (
                    f"evidence/{self.artifact_store.safe_component(call.call_id)}/"
                    f"{artifact_digest}.json"
                ),
                artifact_batch.model_dump(mode="json"),
            )
            materialized: list[EvidenceRecord] = []
            for record in batch.records:
                paths = list(dict.fromkeys([*record.artifact_paths, artifact_relative]))
                materialized.append(record.model_copy(update={"artifact_paths": paths}))
            records.extend(materialized)
            call_succeeded[call.call_id] = bool(materialized) and all(
                record.status in {EvidenceStatus.SUCCESS, EvidenceStatus.PARTIAL}
                for record in materialized
            )
        return EvidenceBatch(
            batch_id=f"EVIDENCE-{stable_hash([record.evidence_id for record in records])[:16]}",
            records=records,
            warnings=list(dict.fromkeys(plan_warnings)),
        )

    def _execute_call(self, call: ToolCall, candidates: list[Candidate]) -> EvidenceBatch:
        if call.retry_limit > self.max_retry_limit:
            return self._failure_batch(
                call,
                candidates,
                status=EvidenceStatus.FAILED,
                failure="retry_limit_exceeds_runtime_policy",
            )
        adapter = self.registry.get(call.tool_name)
        descriptor = adapter.descriptor
        if not descriptor.available:
            return self._failure_batch(
                call,
                candidates,
                status=EvidenceStatus.FAILED,
                failure="tool_unavailable",
            )
        if not any(operation.operation == call.operation for operation in descriptor.operations):
            return self._failure_batch(
                call,
                candidates,
                status=EvidenceStatus.FAILED,
                failure="operation_not_allowlisted",
            )
        cache_key = self._cache_key(call, candidates, descriptor.tool_version)
        if self.cache is not None and call.cache_allowed and descriptor.deterministic:
            cached = self.cache.get(cache_key)
            if cached is not None:
                batch = EvidenceBatch.model_validate(cached)
                refreshed: list[EvidenceRecord] = []
                for record in batch.records:
                    evidence_id = f"EVD-{stable_hash([call.call_id, record.candidate_id, record.output_hash])[:20]}"
                    refreshed.append(
                        record.model_copy(
                            update={
                                "call_id": call.call_id,
                                "evidence_id": evidence_id,
                                "artifact_paths": [],
                            }
                        )
                    )
                return EvidenceBatch(
                    records=refreshed,
                    batch_id=None,
                    warnings=["cache_hit"],
                )

        last_failure = "tool_execution_failed"
        for _attempt in range(call.retry_limit + 1):
            started = time.perf_counter()
            try:
                timed_runner = getattr(adapter, "run_with_timeout", None)
                if callable(timed_runner):
                    raw_result = timed_runner(
                        call,
                        candidates,
                        timeout_seconds=call.max_runtime_seconds,
                    )
                else:
                    # In-process rule validators are allowed to complete
                    # synchronously. External/long-running adapters cannot be
                    # registered as available without run_with_timeout().
                    raw_result = adapter.run(call, candidates)
                runtime_seconds = time.perf_counter() - started
                if runtime_seconds > call.max_runtime_seconds:
                    last_failure = "timeout_after_completed_in_process_call"
                    break
                batch = adapter.normalize(call, candidates, raw_result, runtime_seconds)
                self._validate_batch(call, candidates, batch)
                if self.cache is not None and call.cache_allowed and descriptor.deterministic:
                    self.cache.put(cache_key, batch.model_dump(mode="json"))
                return batch
            except TimeoutError:
                # run_with_timeout() contracts must not return until their
                # process/container is confirmed terminated. Never retry a
                # timeout because termination is part of the trust boundary.
                last_failure = "timeout"
                break
            except Exception as exc:
                last_failure = f"{type(exc).__name__}: {str(exc)[:1000]}"
        status = (
            EvidenceStatus.TIMEOUT
            if last_failure.startswith("timeout")
            else EvidenceStatus.FAILED
        )
        return self._failure_batch(call, candidates, status=status, failure=last_failure)

    @staticmethod
    def _selected_candidates(
        call: ToolCall, candidate_map: dict[str, Candidate]
    ) -> list[Candidate]:
        missing = set(call.candidate_ids) - set(candidate_map)
        if missing:
            raise ValueError(f"tool call references unknown candidates: {sorted(missing)}")
        return [candidate_map[candidate_id] for candidate_id in call.candidate_ids]

    @staticmethod
    def _validate_batch(
        call: ToolCall, candidates: list[Candidate], batch: EvidenceBatch
    ) -> None:
        expected = {item.candidate_id for item in candidates}
        observed = {item.candidate_id for item in batch.records}
        if expected != observed:
            raise ValueError(
                f"adapter result candidate set mismatch; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
            )
        if any(record.call_id != call.call_id for record in batch.records):
            raise ValueError("adapter normalized evidence under the wrong call_id")
        if any(record.tool_name != call.tool_name for record in batch.records):
            raise ValueError("adapter normalized evidence under the wrong tool_name")
        if any(record.evidence_kind != call.evidence_kind for record in batch.records):
            raise ValueError("adapter changed the requested evidence kind")

    def _failure_batch(
        self,
        call: ToolCall,
        candidates: list[Candidate],
        *,
        status: EvidenceStatus,
        failure: str,
    ) -> EvidenceBatch:
        descriptor = self.registry.get(call.tool_name).descriptor
        records = []
        for candidate in candidates:
            output_hash = stable_hash({"status": status, "failure": failure})
            records.append(
                EvidenceRecord(
                    evidence_id=f"EVD-{stable_hash([call.call_id, candidate.candidate_id, output_hash])[:20]}",
                    call_id=call.call_id,
                    candidate_id=candidate.candidate_id,
                    candidate_ref=candidate.candidate_ref,
                    tool_name=call.tool_name,
                    tool_version=descriptor.tool_version,
                    operation=call.operation,
                    method_class=call.method_class,
                    status=status,
                    evidence_kind=call.evidence_kind,
                    fidelity=call.fidelity,
                    properties=[],
                    failure_modes=[failure],
                    runtime_seconds=0.0,
                    input_hash=stable_hash([call, candidate.candidate_ref or candidate]),
                    output_hash=output_hash,
                    parameters_hash=stable_hash(call.conditions),
                    convergence_checks={"operation_completed": False},
                    computational_details=(
                        ComputationalEvidenceDetails(
                            method_name=call.tool_name,
                            method_version=descriptor.tool_version,
                            parameters=call.conditions,
                            code_revision="discovery-os-0.1.0",
                        )
                        if call.evidence_kind == EvidenceKind.COMPUTATIONAL
                        else None
                    ),
                )
            )
        return EvidenceBatch(records=records)

    @staticmethod
    def _cache_key(call: ToolCall, candidates: list[Candidate], tool_version: str) -> str:
        return stable_hash(
            {
                "tool": call.tool_name,
                "tool_version": tool_version,
                "operation": call.operation,
                "candidate_refs": [item.candidate_ref for item in candidates],
                "fallback_candidates": [
                    item if item.candidate_ref is None else None for item in candidates
                ],
                "properties": call.requested_properties,
                "conditions": call.conditions,
                "evidence_kind": call.evidence_kind,
                "fidelity": call.fidelity,
            }
        )


__all__ = ["ToolRuntime"]
