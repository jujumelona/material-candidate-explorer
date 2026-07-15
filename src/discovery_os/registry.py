"""Allow-listed tool and generator registries."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .schemas import (
    Candidate,
    CandidateBatch,
    EvidenceBatch,
    GenerationTask,
    GeneratorDescriptor,
    ToolCall,
    ToolDescriptor,
)


@runtime_checkable
class ToolAdapter(Protocol):
    @property
    def descriptor(self) -> ToolDescriptor:
        ...

    def run(self, call: ToolCall, candidates: list[Candidate]) -> Any:
        ...

    def normalize(
        self,
        call: ToolCall,
        candidates: list[Candidate],
        raw_result: Any,
        runtime_seconds: float,
    ) -> EvidenceBatch:
        ...


@runtime_checkable
class GeneratorAdapter(Protocol):
    @property
    def descriptor(self) -> GeneratorDescriptor:
        ...

    def generate(self, task: GenerationTask, parents: list[Candidate]) -> CandidateBatch:
        ...


class ToolRegistry:
    """Only explicitly registered adapters can ever be invoked."""

    def __init__(self) -> None:
        self._adapters: dict[str, ToolAdapter] = {}

    def register(self, adapter: ToolAdapter, *, replace: bool = False) -> None:
        descriptor = adapter.descriptor
        has_nontrivial_operation = any(
            str(operation.method_class) != "rule_based"
            for operation in descriptor.operations
        )
        if (
            descriptor.available
            and has_nontrivial_operation
            and not descriptor.metadata.get("mock", False)
            and not callable(getattr(adapter, "run_with_timeout", None))
        ):
            raise ValueError(
                f"tool {descriptor.tool_name!r} must implement killable/cooperative "
                "run_with_timeout() before it can be registered as available"
            )
        if descriptor.tool_name in self._adapters and not replace:
            raise ValueError(f"tool {descriptor.tool_name!r} is already registered")
        self._adapters[descriptor.tool_name] = adapter

    def get(self, tool_name: str) -> ToolAdapter:
        try:
            return self._adapters[tool_name]
        except KeyError as exc:
            raise KeyError(f"tool {tool_name!r} is not in the allow-list") from exc

    def describe_tools(self, *, available_only: bool = False) -> list[ToolDescriptor]:
        descriptors = [adapter.descriptor for adapter in self._adapters.values()]
        if available_only:
            descriptors = [descriptor for descriptor in descriptors if descriptor.available]
        return sorted(descriptors, key=lambda item: item.tool_name)

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._adapters


class GeneratorRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, GeneratorAdapter] = {}

    def register(self, adapter: GeneratorAdapter, *, replace: bool = False) -> None:
        descriptor = adapter.descriptor
        if (
            descriptor.available
            and not descriptor.metadata.get("mock", False)
            and not callable(getattr(adapter, "generate_with_timeout", None))
        ):
            raise ValueError(
                f"generator {descriptor.generator_name!r} must implement killable/cooperative "
                "generate_with_timeout() before it can be registered as available"
            )
        if descriptor.generator_name in self._adapters and not replace:
            raise ValueError(f"generator {descriptor.generator_name!r} is already registered")
        self._adapters[descriptor.generator_name] = adapter

    def get(self, generator_name: str) -> GeneratorAdapter:
        try:
            return self._adapters[generator_name]
        except KeyError as exc:
            raise KeyError(f"generator {generator_name!r} is not in the allow-list") from exc

    def describe_generators(self, *, available_only: bool = False) -> list[GeneratorDescriptor]:
        descriptors = [adapter.descriptor for adapter in self._adapters.values()]
        if available_only:
            descriptors = [descriptor for descriptor in descriptors if descriptor.available]
        return sorted(descriptors, key=lambda item: item.generator_name)

    def __contains__(self, generator_name: str) -> bool:
        return generator_name in self._adapters


__all__ = [
    "GeneratorAdapter",
    "GeneratorRegistry",
    "ToolAdapter",
    "ToolRegistry",
]
