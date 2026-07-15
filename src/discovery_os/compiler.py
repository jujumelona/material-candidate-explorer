"""Deterministic compilation of untrusted model plans into executable plans."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hashing import stable_hash
from .registry import ToolRegistry
from .schemas import (
    Candidate,
    CandidateType,
    DiscoveryGoal,
    EvidenceKind,
    Fidelity,
    MethodClass,
    ParameterDescriptor,
    ResourceBudget,
    ToolCall,
    ToolDescriptor,
    ToolOperationDescriptor,
    ValidationIntent,
    ValidationPlan,
)


class PlanCompilationError(ValueError):
    pass


@dataclass(frozen=True)
class RuntimePolicy:
    max_calls_per_plan: int = 64
    max_runtime_per_call_seconds: int = 86_400
    max_total_runtime_seconds: int = 172_800
    max_cpu_cores_per_call: float = 32
    max_gpu_count_per_call: float = 8
    max_memory_gb_per_call: float = 256
    max_estimated_cost_per_plan: float = 0
    max_retry_limit_per_call: int = 0
    allow_experimental_execution: bool = False
    allow_mock_tools: bool = False
    inject_mandatory_sanity_checks: bool = True
    approved_operations: frozenset[tuple[str, str]] = field(default_factory=frozenset)


class PlanCompiler:
    """Treat model output as a proposal, never as executable authority."""

    def __init__(self, registry: ToolRegistry, policy: RuntimePolicy | None = None) -> None:
        self.registry = registry
        self.policy = policy or RuntimePolicy()

    def compile(
        self,
        proposed: ValidationPlan,
        *,
        goal: DiscoveryGoal,
        candidates: list[Candidate],
    ) -> ValidationPlan:
        candidate_map = {item.candidate_id: item for item in candidates}
        wrong_domain = [
            item.candidate_id for item in candidates if item.domain != goal.domain
        ]
        if wrong_domain:
            raise PlanCompilationError(
                f"candidates do not belong to goal domain {goal.domain!r}: {wrong_domain}"
            )
        calls = list(proposed.calls)
        for intent in proposed.intents:
            calls.append(self._compile_intent(intent, candidate_map))
        if self.policy.inject_mandatory_sanity_checks:
            calls = self._inject_mandatory(calls, candidates)
        self._validate_plan(calls, goal=goal, candidate_map=candidate_map)
        ordered = self._topological_order(calls)
        information_gain: dict[str, float] = {}
        compiled_ids = {call.call_id for call in ordered}
        for identifier, value in proposed.expected_information_gain.items():
            if identifier in compiled_ids:
                information_gain[identifier] = value
            elif f"INTENT-{identifier}" in compiled_ids:
                information_gain[f"INTENT-{identifier}"] = value
        return ValidationPlan(
            intents=proposed.intents,
            calls=ordered,
            expected_information_gain=information_gain,
            plan_reason=(
                proposed.plan_reason
                + " Deterministic policy compilation and mandatory sanity checks applied."
            ),
        )

    def _compile_intent(
        self,
        intent: ValidationIntent,
        candidate_map: dict[str, Candidate],
    ) -> ToolCall:
        intent_candidates: list[Candidate] = []
        for reference in intent.candidate_refs:
            candidate = candidate_map.get(reference.candidate_id)
            if candidate is None or candidate.candidate_ref != reference:
                raise PlanCompilationError(
                    f"intent {intent.intent_id!r} references an unknown or stale candidate"
                )
            intent_candidates.append(candidate)

        matches: list[tuple[ToolDescriptor, ToolOperationDescriptor]] = []
        for descriptor in self.registry.describe_tools(available_only=True):
            if descriptor.metadata.get("mock", False) and not self.policy.allow_mock_tools:
                continue
            for operation in descriptor.operations:
                if intent.required_evidence_kind not in operation.evidence_kinds:
                    continue
                if intent.minimum_fidelity not in operation.supported_fidelities:
                    continue
                if intent.preferred_method_classes and operation.method_class not in intent.preferred_method_classes:
                    continue
                if not set(intent.requested_properties).issubset(operation.produced_properties):
                    continue
                if any(candidate.domain not in operation.supported_domains for candidate in intent_candidates):
                    continue
                if any(candidate.candidate_type not in operation.supported_candidate_types for candidate in intent_candidates):
                    continue
                matches.append((descriptor, operation))
        if not matches:
            raise PlanCompilationError(f"no allow-listed operation can satisfy intent {intent.intent_id!r}")
        matches.sort(key=lambda item: (item[1].requires_human_approval, item[0].tool_name, item[1].operation))
        descriptor, operation = matches[0]
        return ToolCall(
            call_id=f"INTENT-{intent.intent_id}",
            tool_name=descriptor.tool_name,
            operation=operation.operation,
            candidate_ids=[item.candidate_id for item in intent_candidates],
            requested_properties=intent.requested_properties,
            conditions=intent.conditions,
            evidence_kind=intent.required_evidence_kind,
            method_class=operation.method_class,
            fidelity=intent.minimum_fidelity,
            priority=intent.priority,
            reason=intent.reason,
            max_runtime_seconds=intent.max_runtime_seconds,
            resource_budget=intent.resource_budget,
        )

    def _inject_mandatory(
        self, calls: list[ToolCall], candidates: list[Candidate]
    ) -> list[ToolCall]:
        result = list(calls)
        rules: list[tuple[str, str, set[str], list[str], MethodClass]] = [
            (
                "common_rules",
                "validate_candidate",
                {str(item.candidate_type) for item in candidates},
                ["representation_valid", "lineage_valid"],
                MethodClass.RULE_BASED,
            ),
            (
                "rdkit",
                "validate_molecule",
                {str(CandidateType.SMALL_MOLECULE)},
                ["validity", "canonical_smiles", "molecular_weight", "logp", "tpsa"],
                MethodClass.RULE_BASED,
            ),
            (
                "composition_rules",
                "validate_composition",
                {
                    str(CandidateType.CRYSTAL),
                    str(CandidateType.COMPOSITION),
                    str(CandidateType.ALLOY),
                    str(CandidateType.BATTERY_MATERIAL),
                    str(CandidateType.CATALYST),
                },
                ["formula_validity", "element_count", "molar_mass"],
                MethodClass.RULE_BASED,
            ),
        ]
        for tool_name, operation, supported_types, properties, method_class in rules:
            if tool_name not in self.registry:
                continue
            descriptor = self.registry.get(tool_name).descriptor
            if not descriptor.available:
                continue
            ids = [
                item.candidate_id
                for item in candidates
                if str(item.candidate_type) in supported_types
            ]
            if not ids:
                continue
            already_covered = {
                candidate_id
                for call in result
                if call.tool_name == tool_name and call.operation == operation
                for candidate_id in call.candidate_ids
            }
            missing = [candidate_id for candidate_id in ids if candidate_id not in already_covered]
            if not missing:
                continue
            suffix = stable_hash([tool_name, operation, missing])[:10]
            result.append(
                ToolCall(
                    call_id=f"POLICY-{tool_name}-{suffix}",
                    tool_name=tool_name,
                    operation=operation,
                    candidate_ids=missing,
                    requested_properties=properties,
                    conditions={},
                    evidence_kind=EvidenceKind.COMPUTATIONAL,
                    method_class=method_class,
                    fidelity=Fidelity.CHEAP,
                    priority=1.0,
                    reason="Mandatory code-side representation sanity gate.",
                    max_runtime_seconds=min(300, self.policy.max_runtime_per_call_seconds),
                    resource_budget=descriptor.default_resource_budget,
                )
            )
        return result

    def _validate_plan(
        self,
        calls: list[ToolCall],
        *,
        goal: DiscoveryGoal,
        candidate_map: dict[str, Candidate],
    ) -> None:
        if len(calls) > self.policy.max_calls_per_plan:
            raise PlanCompilationError("plan exceeds the maximum number of calls")
        ids = [call.call_id for call in calls]
        if len(ids) != len(set(ids)):
            raise PlanCompilationError("compiled plan contains duplicate call IDs")
        total_runtime = 0
        total_cost = 0.0
        for call in calls:
            if call.tool_name not in self.registry:
                raise PlanCompilationError(f"tool {call.tool_name!r} is not registered")
            adapter = self.registry.get(call.tool_name)
            descriptor = adapter.descriptor
            if not descriptor.available:
                raise PlanCompilationError(f"tool {call.tool_name!r} is unavailable")
            if descriptor.metadata.get("mock", False) and not self.policy.allow_mock_tools:
                raise PlanCompilationError(f"mock tool {call.tool_name!r} is disabled by policy")
            operation = self._operation(descriptor, call.operation)
            if operation.requires_human_approval and (
                call.tool_name,
                call.operation,
            ) not in self.policy.approved_operations:
                raise PlanCompilationError(
                    f"operation {call.tool_name}.{call.operation} requires explicit approval"
                )
            if call.evidence_kind == EvidenceKind.EXPERIMENTAL and not self.policy.allow_experimental_execution:
                raise PlanCompilationError("models cannot directly execute experimental operations")
            if call.evidence_kind not in operation.evidence_kinds:
                raise PlanCompilationError("requested evidence kind is unsupported by the operation")
            if call.fidelity not in operation.supported_fidelities:
                raise PlanCompilationError("requested fidelity is unsupported by the operation")
            if call.method_class != operation.method_class:
                raise PlanCompilationError("method_class does not match the registered operation")
            if operation.produced_properties and not set(call.requested_properties).issubset(
                operation.produced_properties
            ):
                raise PlanCompilationError(
                    f"call {call.call_id!r} requests properties not produced by the operation"
                )
            selected: list[Candidate] = []
            for candidate_id in call.candidate_ids:
                candidate = candidate_map.get(candidate_id)
                if candidate is None:
                    raise PlanCompilationError(f"call references unknown candidate {candidate_id!r}")
                selected.append(candidate)
            if any(item.domain not in operation.supported_domains for item in selected):
                raise PlanCompilationError("candidate domain is incompatible with the operation")
            if any(item.candidate_type not in operation.supported_candidate_types for item in selected):
                raise PlanCompilationError("candidate type is incompatible with the operation")
            self._validate_conditions(call.conditions, operation.condition_parameters)
            budget = self._budget(call.resource_budget)
            if call.max_runtime_seconds > self.policy.max_runtime_per_call_seconds:
                raise PlanCompilationError("call runtime exceeds policy")
            if call.retry_limit > self.policy.max_retry_limit_per_call:
                raise PlanCompilationError("call retry_limit exceeds policy")
            if budget.cpu_cores > self.policy.max_cpu_cores_per_call:
                raise PlanCompilationError("call CPU request exceeds policy")
            if budget.gpu_count > self.policy.max_gpu_count_per_call:
                raise PlanCompilationError("call GPU request exceeds policy")
            if budget.memory_gb > self.policy.max_memory_gb_per_call:
                raise PlanCompilationError("call memory request exceeds policy")
            attempts = call.retry_limit + 1
            total_runtime += call.max_runtime_seconds * attempts
            total_cost += budget.estimated_cost * attempts
        if total_runtime > self.policy.max_total_runtime_seconds:
            raise PlanCompilationError("plan runtime exceeds policy")
        if self.policy.max_estimated_cost_per_plan <= 0 and total_cost > 0:
            raise PlanCompilationError("paid calls are disabled by policy")
        if (
            self.policy.max_estimated_cost_per_plan > 0
            and total_cost > self.policy.max_estimated_cost_per_plan
        ):
            raise PlanCompilationError("plan estimated cost exceeds policy")

    @staticmethod
    def _operation(descriptor: ToolDescriptor, name: str) -> ToolOperationDescriptor:
        for operation in descriptor.operations:
            if operation.operation == name:
                return operation
        raise PlanCompilationError(
            f"operation {name!r} is not allow-listed for tool {descriptor.tool_name!r}"
        )

    @classmethod
    def _validate_conditions(
        cls, conditions: dict[str, Any], descriptors: list[ParameterDescriptor]
    ) -> None:
        by_name = {item.name: item for item in descriptors}
        unknown = set(conditions) - set(by_name)
        if unknown:
            raise PlanCompilationError(f"unknown operation conditions: {sorted(unknown)}")
        missing = {item.name for item in descriptors if item.required} - set(conditions)
        if missing:
            raise PlanCompilationError(f"missing operation conditions: {sorted(missing)}")
        for name, value in conditions.items():
            descriptor = by_name[name]
            if not cls._matches_type(value, descriptor.value_type):
                raise PlanCompilationError(f"condition {name!r} has the wrong JSON type")
            if descriptor.allowed_values and value not in descriptor.allowed_values:
                raise PlanCompilationError(f"condition {name!r} is outside its allow-list")

    @staticmethod
    def _matches_type(value: Any, expected: str) -> bool:
        if expected == "string":
            return isinstance(value, str)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        return False

    @staticmethod
    def _budget(value: ResourceBudget | dict[str, float]) -> ResourceBudget:
        if isinstance(value, ResourceBudget):
            return value
        aliases = {
            "cpu": "cpu_cores",
            "gpu": "gpu_count",
            "memory": "memory_gb",
            "storage": "storage_gb",
            "cost": "estimated_cost",
        }
        normalized: dict[str, Any] = {}
        extras: dict[str, float] = {}
        fields = set(ResourceBudget.model_fields) - {"schema_version", "extras"}
        for key, amount in value.items():
            target = aliases.get(key, key)
            if target in fields:
                normalized[target] = amount
            else:
                extras[key] = amount
        normalized["extras"] = extras
        return ResourceBudget.model_validate(normalized)

    @staticmethod
    def _topological_order(calls: list[ToolCall]) -> list[ToolCall]:
        by_id = {call.call_id: call for call in calls}
        dependencies = {call.call_id: set(call.depends_on_call_ids) for call in calls}
        unknown = set().union(*dependencies.values()) - set(by_id) if dependencies else set()
        if unknown:
            raise PlanCompilationError(f"unknown call dependencies: {sorted(unknown)}")
        result: list[ToolCall] = []
        ready = [call for call in calls if not dependencies[call.call_id]]
        while ready:
            ready.sort(key=lambda item: (-item.priority, item.call_id))
            call = ready.pop(0)
            result.append(call)
            for other in calls:
                if call.call_id in dependencies[other.call_id]:
                    dependencies[other.call_id].remove(call.call_id)
                    if not dependencies[other.call_id] and other not in result and other not in ready:
                        ready.append(other)
        if len(result) != len(calls):
            raise PlanCompilationError("call dependency graph contains a cycle")
        return result


__all__ = ["PlanCompilationError", "PlanCompiler", "RuntimePolicy"]
