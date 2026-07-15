"""Allow-listed candidate generators and their runtime."""

from __future__ import annotations

from typing import Any
import time

from .hashing import candidate_content_hash, stable_hash
from .registry import GeneratorRegistry
from .schemas import (
    Candidate,
    CandidateBatch,
    CandidatePlan,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    GenerationTask,
    GeneratorDescriptor,
    ParameterDescriptor,
    ParameterType,
    RepresentationKind,
    ResourceBudget,
)


class GeneratorExecutionError(RuntimeError):
    pass


class DummyGenerator:
    """Known fixtures for integration tests; never claims novelty."""

    _MOLECULES = [
        ("caffeine fixture", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"),
        ("acetaminophen fixture", "CC(=O)NC1=CC=C(O)C=C1"),
        ("aspirin fixture", "CC(=O)OC1=CC=CC=C1C(=O)O"),
    ]
    _FORMULAS = {
        DiscoveryDomain.SUPERCONDUCTORS: [
            ("magnesium diboride fixture", "MgB2"),
            ("YBCO fixture", "YBa2Cu3O7"),
            ("LSCO fixture", "La1.85Sr0.15CuO4"),
        ],
        DiscoveryDomain.BATTERIES: [
            ("lithium iron phosphate fixture", "LiFePO4"),
            ("lithium cobalt oxide fixture", "LiCoO2"),
            ("LLZO fixture", "Li7La3Zr2O12"),
        ],
        DiscoveryDomain.CATALYSTS: [
            ("titania fixture", "TiO2"),
            ("platinum nickel fixture", "Pt3Ni"),
        ],
        DiscoveryDomain.INORGANIC_MATERIALS: [
            ("barium titanate fixture", "BaTiO3"),
            ("silica fixture", "SiO2"),
        ],
        DiscoveryDomain.GENERAL_MATERIALS: [
            ("silicon carbide fixture", "SiC"),
            ("alumina fixture", "Al2O3"),
        ],
        DiscoveryDomain.POLYMERS: [
            ("polyethylene repeat fixture", "C2H4"),
        ],
        DiscoveryDomain.MEDICINAL_CHEMISTRY: [
            ("carbon fixture", "C"),
        ],
    }

    def __init__(self) -> None:
        self._descriptor = GeneratorDescriptor(
            generator_name="dummy_generator",
            generator_version="1.0",
            adapter_version="1.0",
            description="Known molecule/composition fixtures for end-to-end tests only.",
            supported_domains=list(DiscoveryDomain),
            supported_candidate_types=[
                CandidateType.SMALL_MOLECULE,
                CandidateType.CRYSTAL,
                CandidateType.COMPOSITION,
                CandidateType.ALLOY,
                CandidateType.BATTERY_MATERIAL,
                CandidateType.CATALYST,
                CandidateType.POLYMER,
                CandidateType.CUSTOM,
            ],
            accepted_parameters=[
                ParameterDescriptor(
                    name="domain",
                    value_type=ParameterType.STRING,
                    required=False,
                    allowed_values=[item.value for item in DiscoveryDomain],
                ),
                ParameterDescriptor(
                    name="sample_family",
                    value_type=ParameterType.STRING,
                    required=False,
                    allowed_values=["small_molecule_samples", "composition_samples"],
                ),
            ],
            deterministic=True,
            available=True,
            default_resource_budget=ResourceBudget(cpu_cores=0.1),
            metadata={"mock": True},
        )

    @property
    def descriptor(self) -> GeneratorDescriptor:
        return self._descriptor

    def generate(self, task: GenerationTask, parents: list[Candidate]) -> CandidateBatch:
        domain = self._domain(task, parents)
        candidate_type = CandidateType(str(task.candidate_type))
        task_id = task.task_id or f"GEN-{stable_hash(task)[:12]}"
        candidates: list[Candidate] = []
        warnings = [
            "dummy_generator emits known fixtures; outputs are integration-test candidates, not novel discoveries"
        ]
        if candidate_type == CandidateType.SMALL_MOLECULE:
            source: list[tuple[str, str, RepresentationKind]] = [
                (name, value, RepresentationKind.SMILES) for name, value in self._MOLECULES
            ]
        else:
            fixtures = self._FORMULAS.get(domain, self._FORMULAS[DiscoveryDomain.GENERAL_MATERIALS])
            source = [
                (name, value, RepresentationKind.CHEMICAL_FORMULA) for name, value in fixtures
            ]
        count = min(task.requested_count, len(source))
        if count < task.requested_count:
            warnings.append(
                f"dummy fixture catalog contains only {count} compatible unique candidates"
            )
        for index, (name, value, representation_kind) in enumerate(source[:count]):
            prefix = "MOL" if candidate_type == CandidateType.SMALL_MOLECULE else "MAT"
            candidate_id = f"{prefix}-{stable_hash([task_id, domain, candidate_type, value, index])[:16]}"
            candidate = Candidate(
                candidate_id=candidate_id,
                candidate_type=candidate_type,
                domain=domain,
                name=name,
                representations=[
                    CandidateRepresentation(
                        kind=representation_kind,
                        value=value,
                        canonical=False,
                    )
                ],
                parent_candidate_ids=[item.candidate_id for item in parents],
                hypothesis_ids=task.hypothesis_ids,
                generation_task_id=task_id,
                attributes={"demo_fixture": True},
                novelty_rationale="No novelty claim: this is a known integration-test fixture.",
                provenance={
                    "generator": self.descriptor.generator_name,
                    "generator_version": self.descriptor.generator_version,
                    "task_reason": task.reason,
                },
            )
            reference = CandidateRef(
                candidate_id=candidate_id,
                version=(
                    1
                    + max(
                        (
                            parent.candidate_ref.version
                            for parent in parents
                            if parent.candidate_ref is not None
                        ),
                        default=0,
                    )
                ),
                content_hash=candidate_content_hash(candidate),
            )
            candidates.append(candidate.model_copy(update={"candidate_ref": reference}))
        return CandidateBatch(candidates=candidates, generation_warnings=warnings)

    @staticmethod
    def _domain(task: GenerationTask, parents: list[Candidate]) -> DiscoveryDomain:
        if parents:
            return DiscoveryDomain(str(parents[0].domain))
        requested = task.conditions.get("domain")
        if isinstance(requested, str):
            try:
                return DiscoveryDomain(requested)
            except ValueError:
                pass
        candidate_type = str(task.candidate_type)
        defaults = {
            str(CandidateType.SMALL_MOLECULE): DiscoveryDomain.MEDICINAL_CHEMISTRY,
            str(CandidateType.BATTERY_MATERIAL): DiscoveryDomain.BATTERIES,
            str(CandidateType.CATALYST): DiscoveryDomain.CATALYSTS,
            str(CandidateType.POLYMER): DiscoveryDomain.POLYMERS,
        }
        return defaults.get(candidate_type, DiscoveryDomain.GENERAL_MATERIALS)


class UnavailableGenerator:
    def __init__(
        self,
        name: str,
        supported_types: list[CandidateType],
        description: str,
    ) -> None:
        self._descriptor = GeneratorDescriptor(
            generator_name=name,
            generator_version="not-configured",
            adapter_version="1.0-contract",
            description=description,
            supported_domains=list(DiscoveryDomain),
            supported_candidate_types=supported_types,
            available=False,
            metadata={"connector_status": "backend_required"},
        )

    @property
    def descriptor(self) -> GeneratorDescriptor:
        return self._descriptor

    def generate(self, task: GenerationTask, parents: list[Candidate]) -> CandidateBatch:
        raise RuntimeError(
            f"generator {self.descriptor.generator_name!r} requires a configured backend"
        )


class GeneratorRuntime:
    def __init__(
        self,
        registry: GeneratorRegistry,
        *,
        max_candidates_per_task: int = 1_000,
        allow_mock: bool = True,
        max_runtime_per_task_seconds: int = 86_400,
        max_cpu_cores_per_task: float = 32,
        max_gpu_count_per_task: float = 8,
        max_memory_gb_per_task: float = 256,
    ) -> None:
        self.registry = registry
        self.max_candidates_per_task = max_candidates_per_task
        self.allow_mock = allow_mock
        self.max_runtime_per_task_seconds = max_runtime_per_task_seconds
        self.max_cpu_cores_per_task = max_cpu_cores_per_task
        self.max_gpu_count_per_task = max_gpu_count_per_task
        self.max_memory_gb_per_task = max_memory_gb_per_task

    def execute(
        self,
        plan: CandidatePlan,
        *,
        existing_candidates: list[Candidate] | None = None,
    ) -> CandidateBatch:
        existing = {item.candidate_id: item for item in (existing_candidates or [])}
        generated: list[Candidate] = []
        warnings: list[str] = []
        for task_index, original_task in enumerate(plan.tasks):
            task = original_task
            if task.task_id is None:
                task = task.model_copy(
                    update={"task_id": f"GEN-{task_index}-{stable_hash(task)[:12]}"}
                )
            if task.requested_count > self.max_candidates_per_task:
                raise GeneratorExecutionError("generation request exceeds candidate policy")
            if task.generator_name not in self.registry:
                raise GeneratorExecutionError(
                    f"generator {task.generator_name!r} is not registered"
                )
            adapter = self.registry.get(task.generator_name)
            descriptor = adapter.descriptor
            if not descriptor.available:
                raise GeneratorExecutionError(
                    f"generator {task.generator_name!r} is unavailable"
                )
            if descriptor.metadata.get("mock", False) and not self.allow_mock:
                raise GeneratorExecutionError("mock generators are disabled")
            if task.max_runtime_seconds > self.max_runtime_per_task_seconds:
                raise GeneratorExecutionError("generation runtime request exceeds policy")
            if task.resource_budget.cpu_cores > self.max_cpu_cores_per_task:
                raise GeneratorExecutionError("generation CPU request exceeds policy")
            if task.resource_budget.gpu_count > self.max_gpu_count_per_task:
                raise GeneratorExecutionError("generation GPU request exceeds policy")
            if task.resource_budget.memory_gb > self.max_memory_gb_per_task:
                raise GeneratorExecutionError("generation memory request exceeds policy")
            self._validate_conditions(task.conditions, descriptor.accepted_parameters)
            candidate_type = CandidateType(str(task.candidate_type))
            if candidate_type not in descriptor.supported_candidate_types:
                raise GeneratorExecutionError("generator does not support the candidate type")
            missing_parents = set(task.parent_candidate_ids) - set(existing)
            if missing_parents:
                raise GeneratorExecutionError(
                    f"generation task has unknown parents: {sorted(missing_parents)}"
                )
            parents = [existing[item] for item in task.parent_candidate_ids]
            started = time.perf_counter()
            try:
                timed_generator = getattr(adapter, "generate_with_timeout", None)
                if callable(timed_generator):
                    batch = timed_generator(
                        task,
                        parents,
                        timeout_seconds=task.max_runtime_seconds,
                    )
                else:
                    batch = adapter.generate(task, parents)
                if time.perf_counter() - started > task.max_runtime_seconds:
                    raise TimeoutError("in-process generator exceeded its deadline")
            except TimeoutError as exc:
                raise GeneratorExecutionError(
                    f"generator {task.generator_name!r} timed out"
                ) from exc
            if len(batch.candidates) > task.requested_count:
                raise GeneratorExecutionError("generator returned more candidates than requested")
            for candidate in batch.candidates:
                if candidate.candidate_type != candidate_type:
                    raise GeneratorExecutionError("generator returned the wrong candidate type")
                if candidate.generation_task_id != task.task_id:
                    raise GeneratorExecutionError("candidate generation_task_id does not match")
                if candidate.candidate_ref is None:
                    raise GeneratorExecutionError("generated candidates require immutable CandidateRef")
                if candidate.candidate_ref.content_hash != candidate_content_hash(candidate):
                    raise GeneratorExecutionError("generated candidate content hash is invalid")
                parent_versions = [
                    parent.candidate_ref.version
                    for parent in parents
                    if parent.candidate_ref is not None
                ]
                if (
                    parent_versions
                    and candidate.candidate_ref.version <= max(parent_versions)
                ):
                    raise GeneratorExecutionError(
                        "revised candidate version must be greater than every parent version"
                    )
                if candidate.candidate_id in existing:
                    raise GeneratorExecutionError("generator returned a duplicate candidate ID")
                existing[candidate.candidate_id] = candidate
                generated.append(candidate)
            warnings.extend(batch.generation_warnings)
        return CandidateBatch(candidates=generated, generation_warnings=warnings)

    @staticmethod
    def _validate_conditions(
        conditions: dict[str, Any], descriptors: list[ParameterDescriptor]
    ) -> None:
        by_name = {item.name: item for item in descriptors}
        unknown = set(conditions) - set(by_name)
        if unknown:
            raise GeneratorExecutionError(
                f"unknown generator conditions: {sorted(unknown)}"
            )
        missing = {item.name for item in descriptors if item.required} - set(conditions)
        if missing:
            raise GeneratorExecutionError(
                f"missing generator conditions: {sorted(missing)}"
            )
        for name, value in conditions.items():
            descriptor = by_name[name]
            expected = str(descriptor.value_type)
            valid = {
                "string": isinstance(value, str),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
                "array": isinstance(value, list),
                "object": isinstance(value, dict),
            }.get(expected, False)
            if not valid:
                raise GeneratorExecutionError(
                    f"generator condition {name!r} has the wrong JSON type"
                )
            if descriptor.allowed_values and value not in descriptor.allowed_values:
                raise GeneratorExecutionError(
                    f"generator condition {name!r} is outside its allow-list"
                )


def build_default_generator_registry(*, include_placeholders: bool = True) -> GeneratorRegistry:
    registry = GeneratorRegistry()
    registry.register(DummyGenerator())
    if include_placeholders:
        registry.register(
            UnavailableGenerator(
                "genmol",
                [CandidateType.SMALL_MOLECULE],
                "Molecular generation connector; model backend required.",
            )
        )
        registry.register(
            UnavailableGenerator(
                "mattergen",
                [CandidateType.CRYSTAL, CandidateType.COMPOSITION],
                "Crystal/material generation connector; model backend required.",
            )
        )
        registry.register(
            UnavailableGenerator(
                "protein_generator",
                [CandidateType.BIOLOGIC],
                "Protein generation connector; model backend required.",
            )
        )
        registry.register(
            UnavailableGenerator(
                "composition_generator",
                [
                    CandidateType.COMPOSITION,
                    CandidateType.ALLOY,
                    CandidateType.BATTERY_MATERIAL,
                    CandidateType.CATALYST,
                ],
                "Composition generator connector; model backend required.",
            )
        )
        registry.register(
            UnavailableGenerator(
                "reaction_generator",
                [CandidateType.REACTION],
                "Reaction/synthesis generator connector; backend required.",
            )
        )
        registry.register(
            UnavailableGenerator(
                "internal_core",
                list(CandidateType),
                "Future internally trained discovery generator interface.",
            )
        )
    return registry


__all__ = [
    "DummyGenerator",
    "GeneratorExecutionError",
    "GeneratorRuntime",
    "UnavailableGenerator",
    "build_default_generator_registry",
]
