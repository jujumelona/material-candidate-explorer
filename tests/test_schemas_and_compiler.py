from __future__ import annotations

import pytest
from pydantic import ValidationError

from discovery_os.compiler import PlanCompilationError, PlanCompiler, RuntimePolicy
from discovery_os.schemas import (
    CandidateType,
    DiscoveryDomain,
    GoalCompileRequest,
    ValidationPlan,
)
from discovery_os.tool_adapters import build_default_tool_registry


def test_strict_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        GoalCompileRequest.model_validate(
            {
                "user_text": "Find a stable material",
                "shell_command": "python arbitrary_payload.py",
            }
        )


def test_tool_call_rejects_an_unsafe_executable_field(tool_call_factory) -> None:
    payload = tool_call_factory().model_dump(mode="json")
    payload["command"] = "rm -rf /"

    with pytest.raises(ValidationError, match="extra_forbidden"):
        tool_call_factory(**payload)


@pytest.mark.parametrize(
    "dependencies, message",
    [
        (["CALL-001"], "cannot depend on itself"),
        (["DOES-NOT-EXIST"], "unknown dependencies"),
    ],
)
def test_validation_plan_rejects_unsafe_dependencies(
    tool_call_factory, dependencies: list[str], message: str
) -> None:
    call = tool_call_factory(depends_on_call_ids=dependencies)

    with pytest.raises(ValidationError, match=message):
        ValidationPlan(
            calls=[call],
            expected_information_gain={},
            plan_reason="Unsafe dependency graph fixture.",
        )


def test_plan_compiler_rejects_a_multi_call_cycle(
    candidate_factory, goal_factory, tool_call_factory
) -> None:
    candidate = candidate_factory()
    first = tool_call_factory(
        call_id="CALL-A",
        candidate_ids=[candidate.candidate_id],
        depends_on_call_ids=["CALL-B"],
    )
    second = tool_call_factory(
        call_id="CALL-B",
        candidate_ids=[candidate.candidate_id],
        depends_on_call_ids=["CALL-A"],
    )
    proposed = ValidationPlan(
        calls=[first, second],
        expected_information_gain={},
        plan_reason="Cyclic graph fixture.",
    )
    compiler = PlanCompiler(
        build_default_tool_registry(include_placeholders=False),
        RuntimePolicy(inject_mandatory_sanity_checks=False),
    )

    with pytest.raises(PlanCompilationError, match="contains a cycle"):
        compiler.compile(proposed, goal=goal_factory(), candidates=[candidate])


def test_plan_compiler_injects_domain_appropriate_mandatory_validators(
    candidate_factory, goal_factory
) -> None:
    molecule = candidate_factory(
        candidate_id="MOL-MANDATORY",
        domain=DiscoveryDomain.GENERAL_MATERIALS,
    )
    material = candidate_factory(
        candidate_id="MAT-MANDATORY",
        candidate_type=CandidateType.COMPOSITION,
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        value="BaTiO3",
    )
    registry = build_default_tool_registry(include_placeholders=False)
    compiled = PlanCompiler(registry).compile(
        ValidationPlan(
            calls=[],
            expected_information_gain={},
            plan_reason="The model proposed no validators.",
        ),
        goal=goal_factory(
            domain=DiscoveryDomain.GENERAL_MATERIALS,
            candidate_types=[CandidateType.SMALL_MOLECULE, CandidateType.COMPOSITION],
        ),
        candidates=[molecule, material],
    )

    by_operation = {(call.tool_name, call.operation): call for call in compiled.calls}
    assert by_operation[("common_rules", "validate_candidate")].candidate_ids == [
        molecule.candidate_id,
        material.candidate_id,
    ]
    assert by_operation[("composition_rules", "validate_composition")].candidate_ids == [
        material.candidate_id
    ]
    if registry.get("rdkit").descriptor.available:
        assert by_operation[("rdkit", "validate_molecule")].candidate_ids == [
            molecule.candidate_id
        ]


def _compiler_without_injection() -> PlanCompiler:
    return PlanCompiler(
        build_default_tool_registry(include_placeholders=False),
        RuntimePolicy(inject_mandatory_sanity_checks=False),
    )


def _plan_with(call) -> ValidationPlan:
    return ValidationPlan(
        calls=[call],
        expected_information_gain={},
        plan_reason="Untrusted model plan fixture.",
    )


def test_plan_compiler_rejects_unknown_tool(
    candidate_factory, goal_factory, tool_call_factory
) -> None:
    candidate = candidate_factory()
    call = tool_call_factory(
        tool_name="shell",
        operation="execute",
        candidate_ids=[candidate.candidate_id],
    )

    with pytest.raises(PlanCompilationError, match="not registered"):
        _compiler_without_injection().compile(
            _plan_with(call), goal=goal_factory(), candidates=[candidate]
        )


def test_plan_compiler_rejects_unknown_operation(
    candidate_factory, goal_factory, tool_call_factory
) -> None:
    candidate = candidate_factory()
    call = tool_call_factory(
        operation="arbitrary_python",
        candidate_ids=[candidate.candidate_id],
    )

    with pytest.raises(PlanCompilationError, match="not allow-listed"):
        _compiler_without_injection().compile(
            _plan_with(call), goal=goal_factory(), candidates=[candidate]
        )


def test_plan_compiler_rejects_unknown_condition(
    candidate_factory, goal_factory, tool_call_factory
) -> None:
    candidate = candidate_factory()
    call = tool_call_factory(
        candidate_ids=[candidate.candidate_id],
        conditions={"command": "arbitrary executable text"},
    )

    with pytest.raises(PlanCompilationError, match="unknown operation conditions"):
        _compiler_without_injection().compile(
            _plan_with(call), goal=goal_factory(), candidates=[candidate]
        )


@pytest.mark.parametrize(
    "budget, message",
    [
        ({"cpu": 33}, "CPU request exceeds policy"),
        ({"cost": 0.01}, "paid calls are disabled"),
    ],
)
def test_plan_compiler_rejects_budget_violations(
    candidate_factory,
    goal_factory,
    tool_call_factory,
    budget: dict[str, float],
    message: str,
) -> None:
    candidate = candidate_factory()
    call = tool_call_factory(
        candidate_ids=[candidate.candidate_id],
        resource_budget=budget,
    )

    with pytest.raises(PlanCompilationError, match=message):
        _compiler_without_injection().compile(
            _plan_with(call), goal=goal_factory(), candidates=[candidate]
        )


def test_plan_compiler_rejects_mock_tools_by_default(
    candidate_factory, goal_factory, tool_call_factory
) -> None:
    candidate = candidate_factory()
    call = tool_call_factory(
        tool_name="dummy_simulation",
        operation="simulate",
        candidate_ids=[candidate.candidate_id],
        requested_properties=["target_property"],
        method_class="physics_simulation",
    )

    with pytest.raises(PlanCompilationError, match="mock tool .* disabled"):
        _compiler_without_injection().compile(
            _plan_with(call), goal=goal_factory(), candidates=[candidate]
        )


def test_plan_compiler_rejects_model_controlled_retries_by_default(
    candidate_factory, goal_factory, tool_call_factory
) -> None:
    candidate = candidate_factory()
    call = tool_call_factory(
        candidate_ids=[candidate.candidate_id],
        retry_limit=1,
    )

    with pytest.raises(PlanCompilationError, match="retry_limit exceeds policy"):
        _compiler_without_injection().compile(
            _plan_with(call), goal=goal_factory(), candidates=[candidate]
        )
