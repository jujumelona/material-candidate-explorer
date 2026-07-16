from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from discovery_os.fusion_schemas import (
    DiagnosticProperty,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    FusionGenerationRequest,
    FusionGenerationResponse,
    GenerationControls,
    ScientificModality,
    ScientificWorkspace,
    TensorRole,
    WorkspaceEntity,
    WorkspaceEntityRole,
    WorkspaceMode,
    WorkspaceRunConfig,
)
from discovery_os.hashing import candidate_content_hash, stable_hash
from discovery_os.schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    ObjectiveDirection,
    PropertyObjective,
    RepresentationKind,
)
from discovery_os.sidecars import (
    BoltzExpert,
    ESMExpert,
    ExpertResult,
    GeneratedBatch,
    GeneratedCandidateData,
    ModelIdentity,
    ModelExecutionError,
    OptionalDependencyError,
    PropertyResult,
    QHNetExpert,
    SidecarLimits,
    UniMolExpert,
    create_sidecar_app,
)
from discovery_os.sidecars.base import LazyModelAdapter, numeric_tensor_data
from discovery_os.sidecars.conversions import periodic_atom_entity_ids
from discovery_os.sidecars.generators import MatterGenGenerator
from discovery_os.sidecars.app import _BoundedExecutor
from discovery_os.sidecars.errors import SidecarBusyError, UnsupportedModelError
from discovery_os.sidecars.experts import _minimum_periodic_distance


def test_periodic_atom_ids_survive_parser_reordering() -> None:
    class AseLike:
        def get_chemical_symbols(self):
            return ["Li", "O"]

        def get_scaled_positions(self, *, wrap: bool):
            assert wrap is True
            return [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]]

    pymatgen_like = [
        SimpleNamespace(
            specie=SimpleNamespace(symbol="O"),
            frac_coords=[0.25, 0.25, 0.25],
        ),
        SimpleNamespace(
            specie=SimpleNamespace(symbol="Li"),
            frac_coords=[1.0, 0.0, 0.0],
        ),
    ]

    ase_ids = periodic_atom_entity_ids(AseLike())
    pymatgen_ids = periodic_atom_entity_ids(pymatgen_like)

    assert ase_ids != pymatgen_ids
    assert set(ase_ids) == set(pymatgen_ids)


def test_single_atom_distance_uses_reduced_cell() -> None:
    class Cell:
        def niggli_reduce(self):
            return ([[0.1, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], None)

    class Atoms:
        def __len__(self):
            return 1

        def get_cell(self):
            return Cell()

    assert _minimum_periodic_distance(Atoms()) == 0.1


def test_multi_atom_distance_includes_nearest_periodic_self_image() -> None:
    class Cell:
        def niggli_reduce(self):
            return ([[0.4, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]], None)

    class Atoms:
        def __len__(self):
            return 2

        def get_cell(self):
            return Cell()

        def get_all_distances(self, *, mic: bool):
            assert mic is True
            return [[0.0, 2.0], [2.0, 0.0]]

    assert _minimum_periodic_distance(Atoms()) == 0.4


def _candidate() -> Candidate:
    candidate = Candidate(
        candidate_id="molecule-1",
        candidate_type=CandidateType.SMALL_MOLECULE,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.SMILES,
                value="CCO",
                canonical=True,
            )
        ],
    )
    return candidate.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate.candidate_id,
                version=1,
                content_hash=candidate_content_hash(candidate),
            )
        }
    )


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="goal-1",
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        title="Molecule generation",
        scientific_question="Can a generated molecule improve the target?",
        objectives=[
            PropertyObjective(
                property_name="activity",
                direction=ObjectiveDirection.MAXIMIZE,
            )
        ],
        validation_profile_id="medicinal-v1",
        candidate_types=[CandidateType.SMALL_MOLECULE],
    )


def _workspace() -> ScientificWorkspace:
    return ScientificWorkspace(
        workspace_id="workspace-1",
        primary_entity_id="primary",
        entities=[
            WorkspaceEntity(
                entity_id="primary",
                role=WorkspaceEntityRole.PRIMARY_CANDIDATE,
                candidate_ref=_candidate().candidate_ref,
            )
        ],
    )


def _feature_request() -> ExpertFeatureRequest:
    return ExpertFeatureRequest(
        workspace_entity_id="primary",
        candidate=_candidate(),
        goal=_goal(),
        modality=ScientificModality.MOLECULE_2D,
        feature_space="unimol-cls-v1",
        cycle=0,
        seed=7,
    )


def _generation_request(*, candidate_count: int = 2) -> FusionGenerationRequest:
    goal = _goal()
    parent = _candidate()
    run_config = WorkspaceRunConfig(
        workspace_mode=WorkspaceMode.OFF,
        seed=7,
        goal_hash=stable_hash(goal),
        parent_candidate_ref=parent.candidate_ref,
        pair_key="pair-1",
        cohort_index=0,
        generator_id="reinvent4",
        generator_version="4.8",
        generator_code_revision="code-revision",
        generator_weight_revision="weight-revision",
        generator_parameters_hash=stable_hash({"temperature": 1.2}),
        decoder_config_hash=stable_hash({"decoder": 1}),
        postprocessing_hash=stable_hash({"post": 1}),
        resource_budget_hash=stable_hash({"gpu": 0}),
        evaluator_panel_hash=stable_hash({"panel": 1}),
        candidate_count=candidate_count,
        generation_controls=GenerationControls(
            alpha=0.4,
            temperature=1.2,
            mutation_strength=0.3,
            diversity_strength=0.8,
            decision_reason="pytest batch",
        ),
    )
    return FusionGenerationRequest(
        goal=goal,
        parent_candidate=parent,
        workspace=_workspace(),
        workspace_mode=WorkspaceMode.OFF,
        run_config=run_config,
    )


class _ExpertRuntime:
    loaded = True
    load_failed = False
    supported = True
    device = "cpu"

    def __init__(self) -> None:
        self.requests: list[ExpertFeatureRequest] = []

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        self.requests.append(request)
        return ExpertResult(
            values=[[0.25, -0.5, 1.5]],
            tensor_role=TensorRole.GLOBAL_EMBEDDING,
            projection_id="unimol-cls-v1",
            entity_type="molecule",
            entity_ids=("molecule",),
            pooling="cls",
            normalization="fixture-none",
            properties=(PropertyResult("activity", 0.7, source="fixture-model"),),
        )


class _GeneratorRuntime:
    loaded = True
    load_failed = False
    supported = True
    device = "cpu"

    def __init__(self) -> None:
        self.controls: list[GenerationControls] = []

    def generate(self, request: FusionGenerationRequest) -> GeneratedBatch:
        self.controls.append(request.run_config.generation_controls)
        rows = tuple(
            GeneratedCandidateData(
                representations=(
                    CandidateRepresentation(
                        kind=RepresentationKind.SMILES,
                        value=value,
                        canonical=True,
                    ),
                ),
                attributes={"raw_index": index},
            )
            for index, value in enumerate(("CCN", "CCC", "CCCl")[: request.run_config.candidate_count])
        )
        return GeneratedBatch(candidates=rows, warnings=("fixture generator",))


def _expert_identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="unimol",
        model_version="0.1.6",
        adapter_version="1.0.0",
        code_revision="code-revision",
        weight_revision="weight-revision",
        capabilities=frozenset({"features"}),
    )


def _generator_identity() -> ModelIdentity:
    return ModelIdentity(
        model_id="reinvent4",
        model_version="4.8",
        adapter_version="1.0.0",
        code_revision="code-revision",
        weight_revision="weight-revision",
        capabilities=frozenset({"generate"}),
    )


def test_feature_sidecar_emits_exact_central_schema_and_raw_properties() -> None:
    runtime = _ExpertRuntime()
    app = create_sidecar_app(identity=_expert_identity(), runtime=runtime)

    with TestClient(app) as client:
        health = client.get("/health")
        response = client.post(
            "/v1/features",
            json=_feature_request().model_dump(mode="json", exclude_none=False),
        )

    assert health.status_code == 200
    assert health.json()["status"] == "ready"
    assert response.status_code == 200, response.text
    payload = ExpertFeaturePayload.model_validate(response.json())
    assert payload.candidate_ref == _candidate().candidate_ref
    assert payload.tensor is not None
    assert payload.tensor.shape == [1, 3]
    assert payload.tensor.values == [0.25, -0.5, 1.5]
    assert payload.semantics is not None
    assert payload.semantics.entity_ids == ["molecule"]
    assert payload.properties == [
        DiagnosticProperty(property_name="activity", value=0.7, source="fixture-model")
    ]
    assert payload.provenance.expert_id == "unimol"
    assert payload.provenance.seed == 7
    assert runtime.requests[0].workspace_entity_id == "primary"


def test_generator_sidecar_returns_content_addressed_batch_and_controls() -> None:
    runtime = _GeneratorRuntime()
    app = create_sidecar_app(identity=_generator_identity(), runtime=runtime)
    request = _generation_request(candidate_count=2)

    with TestClient(app) as client:
        response = client.post(
            "/v1/generate",
            json=request.model_dump(mode="json", exclude_none=False),
        )

    assert response.status_code == 200, response.text
    result = FusionGenerationResponse.model_validate(response.json())
    assert result.candidate is None
    assert len(result.candidates) == 2
    assert [item.pair_slot for item in result.pair_slots] == [0, 1]
    assert [item.stream_position for item in result.pair_slots] == [0, 1]
    assert [item.candidate_ref for item in result.pair_slots] == [
        item.candidate_ref for item in result.candidates
    ]
    assert len(
        {
            (
                item.candidate_ref.candidate_id,
                item.candidate_ref.version,
                item.candidate_ref.content_hash,
            )
            for item in result.candidates
        }
    ) == 2
    for candidate in result.candidates:
        assert candidate.parent_candidate_refs == [_candidate().candidate_ref]
        assert candidate.parent_candidate_ids == [_candidate().candidate_id]
        assert candidate_content_hash(candidate) == candidate.candidate_ref.content_hash
        assert candidate.provenance["generator_id"] == "reinvent4"
    assert result.provenance.parameters_hash == request.run_config.generator_parameters_hash
    assert result.provenance.seed == 7
    assert runtime.controls[0].temperature == 1.2
    assert runtime.controls[0].diversity_strength == 0.8


def test_generator_batch_limit_and_identity_are_enforced() -> None:
    app = create_sidecar_app(
        identity=_generator_identity(),
        runtime=_GeneratorRuntime(),
        limits=SidecarLimits(max_batch_size=1),
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/generate",
            json=_generation_request(candidate_count=2).model_dump(mode="json", exclude_none=False),
        )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_limit_exceeded"

    request = _generation_request(candidate_count=1)
    request.run_config.generator_version = "forged-version"
    with TestClient(
        create_sidecar_app(identity=_generator_identity(), runtime=_GeneratorRuntime())
    ) as client:
        response = client.post(
            "/v1/generate",
            json=request.model_dump(mode="json", exclude_none=False),
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"


def test_strict_request_and_body_limits_reject_before_model_execution() -> None:
    runtime = _ExpertRuntime()
    app = create_sidecar_app(
        identity=_expert_identity(),
        runtime=runtime,
        limits=SidecarLimits(max_request_bytes=1_024),
    )
    raw = _feature_request().model_dump(mode="json", exclude_none=False)
    raw.pop("schema_version")
    with TestClient(app) as client:
        invalid = client.post("/v1/features", json=raw)
        oversized = client.post(
            "/v1/features",
            content=b"{" + b"x" * 2_000 + b"}",
            headers={"content-type": "application/json"},
        )
    assert invalid.status_code == 413 or invalid.status_code == 422
    assert invalid.json()["error"]["code"] in {"invalid_request", "request_limit_exceeded"}
    assert oversized.status_code == 413
    assert runtime.requests == []


def test_unsupported_adapter_fails_closed_without_tensor() -> None:
    class UnsupportedRuntime:
        supported = False
        loaded = False
        load_failed = False
        reason = "fixture is intentionally unavailable"
        install_action = "configure a real model"

        def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
            raise UnsupportedModelError("No synthetic tensor is returned by this fixture")

    identity = ModelIdentity(
        model_id="qhnet-source",
        model_version="pinned-source",
        adapter_version="1.0.0",
        code_revision="code-revision",
        weight_revision="weight-revision",
        capabilities=frozenset({"features"}),
    )
    app = create_sidecar_app(identity=identity, runtime=UnsupportedRuntime())
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "unsupported"
        response = client.post(
            "/v1/features",
            json=_feature_request().model_dump(mode="json", exclude_none=False),
        )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["code"] == "unsupported_model"
    assert "No synthetic tensor" in body["error"]["message"]
    assert "tensor" not in body


def test_optional_model_import_is_lazy_and_missing_package_is_actionable(monkeypatch) -> None:
    adapter = UniMolExpert(device="cpu")
    assert adapter.loaded is False
    original = importlib.import_module

    def missing(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "unimol_tools":
            raise ModuleNotFoundError(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr("discovery_os.sidecars.base.importlib.import_module", missing)
    with pytest.raises(OptionalDependencyError, match="unimol_tools"):
        adapter.encode(_feature_request())
    assert adapter.load_failed is True


class _CountingLazyAdapter(LazyModelAdapter[object]):
    def __init__(self) -> None:
        super().__init__(device="cpu")
        self.load_count = 0

    def _load_model(self, device: str) -> object:
        self.load_count += 1
        return object()

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        self._ensure_loaded()
        return ExpertResult(
            values=[[1.0]],
            tensor_role=TensorRole.GLOBAL_EMBEDDING,
            projection_id="counting-v1",
            entity_ids=("molecule",),
        )


def test_lazy_checkpoint_loader_runs_once_across_requests() -> None:
    runtime = _CountingLazyAdapter()
    app = create_sidecar_app(identity=_expert_identity(), runtime=runtime)
    payload = _feature_request().model_dump(mode="json", exclude_none=False)
    with TestClient(app) as client:
        first = client.post("/v1/features", json=payload)
        second = client.post("/v1/features", json=payload)
    assert first.status_code == second.status_code == 200
    assert runtime.load_count == 1


def test_runtime_never_accepts_prebuilt_forged_payload() -> None:
    class BadRuntime(_ExpertRuntime):
        def encode(self, request: ExpertFeatureRequest) -> Any:
            return {"tensor": [0.0]}

    app = create_sidecar_app(identity=_expert_identity(), runtime=BadRuntime())
    with TestClient(app) as client:
        response = client.post(
            "/v1/features",
            json=_feature_request().model_dump(mode="json", exclude_none=False),
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "invalid_model_output"


def test_duplicate_json_keys_are_rejected() -> None:
    valid = _feature_request().model_dump(mode="json", exclude_none=False)
    encoded = json.dumps(valid, separators=(",", ":"))
    duplicate = encoded[:-1] + ',"schema_version":"1.0"}'
    app = create_sidecar_app(identity=_expert_identity(), runtime=_ExpertRuntime())
    with TestClient(app) as client:
        response = client.post(
            "/v1/features",
            content=duplicate.encode(),
            headers={"content-type": "application/json"},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_bounded_executor_rejects_when_worker_and_queue_are_full() -> None:
    started = threading.Event()
    release = threading.Event()
    executor = _BoundedExecutor(
        SidecarLimits(max_concurrency=1, max_queue_size=0, request_timeout_seconds=2)
    )

    def blocking() -> str:
        started.set()
        release.wait(timeout=2)
        return "done"

    async def scenario() -> None:
        first = asyncio.create_task(executor.run(blocking))
        await asyncio.to_thread(started.wait, 1)
        with pytest.raises(SidecarBusyError):
            await executor.run(lambda: "must not run")
        release.set()
        assert await first == "done"

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        executor.shutdown()


def test_unimol_3d_route_uses_official_atoms_coordinates_form(monkeypatch) -> None:
    class FakeAtoms:
        def get_chemical_symbols(self) -> list[str]:
            return ["C", "O"]

        def get_positions(self) -> Any:
            return SimpleNamespace(tolist=lambda: [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])

    class FakeUniMol:
        def __init__(self) -> None:
            self.model_input: Any = None

        def get_repr(self, model_input: Any, *, return_atomic_reprs: bool) -> dict[str, Any]:
            self.model_input = model_input
            assert return_atomic_reprs is True
            return {"cls_repr": [[1.0, 2.0]], "atomic_reprs": [[[3.0], [4.0]]]}

    candidate = Candidate(
        candidate_id="molecule-3d",
        candidate_type=CandidateType.SMALL_MOLECULE,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[
            CandidateRepresentation(kind=RepresentationKind.XYZ, value="2\nfixture\nC 0 0 0\nO 1.2 0 0")
        ],
    )
    candidate = candidate.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate.candidate_id,
                version=1,
                content_hash=candidate_content_hash(candidate),
            )
        }
    )
    request = ExpertFeatureRequest(
        workspace_entity_id="primary",
        candidate=candidate,
        goal=_goal(),
        modality=ScientificModality.MOLECULE_3D,
        feature_space="unimol-cls-v1",
        cycle=0,
        seed=1,
    )
    model = FakeUniMol()
    adapter = UniMolExpert(device="cpu")
    adapter._model = model
    adapter._resolved_device = "cpu"
    monkeypatch.setattr(
        "discovery_os.sidecars.experts.candidate_to_ase",
        lambda *_args, **_kwargs: FakeAtoms(),
    )

    result = adapter.encode(request)

    assert result.values == [[1.0, 2.0]]
    assert model.model_input == {
        "atoms": [["C", "O"]],
        "coordinates": [[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]],
    }


def test_mattergen_revision_targets_are_used_without_inventing_direction_values() -> None:
    generator = MatterGenGenerator(pretrained_name="dft_band_gap", device="cpu")
    request = SimpleNamespace(
        goal=SimpleNamespace(
            objectives=[SimpleNamespace(property_name="dft_band_gap", target_value=1.5)]
        ),
        revision_proposal=SimpleNamespace(
            desired_changes=[
                SimpleNamespace(
                    property_name="dft_band_gap",
                    target_value=1.5,
                    direction="target",
                ),
                SimpleNamespace(
                    property_name="ml_bulk_modulus",
                    target_value=None,
                    direction="increase",
                ),
            ]
        ),
        parent_candidate=_candidate(),
    )

    conditions, warnings = generator._conditions(request)

    assert conditions == {"dft_band_gap": 1.5}
    assert any("raw unified latent" in warning for warning in warnings)
    assert any("was not invented" in warning for warning in warnings)

    request.revision_proposal.desired_changes[0].target_value = 2.0
    overridden, override_warnings = generator._conditions(request)
    assert overridden == {"dft_band_gap": 2.0}
    assert any("overrides the original goal target" in warning for warning in override_warnings)


def test_mattergen_python_runtime_loads_checkpoint_once(monkeypatch, tmp_path: Path) -> None:
    counts = {"checkpoint": 0, "prepare": 0, "generate": 0}

    class FakeCheckpoint:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["model_path"] == checkpoint_path.resolve()
            assert kwargs["load_epoch"] == "last"
            assert kwargs["config_overrides"] == []
            assert kwargs["strict_checkpoint_loading"] is True
            counts["checkpoint"] += 1

    class FakeCrystalGenerator:
        def __init__(self, **kwargs: Any) -> None:
            self.properties_to_condition_on = kwargs["properties_to_condition_on"]
            self.diffusion_guidance_factor = kwargs["diffusion_guidance_factor"]
            self.model = self

        def prepare(self) -> None:
            counts["prepare"] += 1

        def to(self, device: str) -> Any:
            assert device == "cpu"
            return self

        def generate(self, *, batch_size: int, num_batches: int, output_dir: str) -> list[object]:
            counts["generate"] += 1
            assert num_batches == 1
            assert output_dir
            return [object() for _ in range(batch_size)]

    def fake_module(name: str, *, install_hint: str) -> Any:
        if name == "mattergen.generator":
            return SimpleNamespace(CrystalGenerator=FakeCrystalGenerator)
        if name == "mattergen.common.utils.data_classes":
            return SimpleNamespace(MatterGenCheckpointInfo=FakeCheckpoint)
        if name == "numpy":
            return SimpleNamespace(random=SimpleNamespace(seed=lambda _seed: None))
        if name == "torch":
            return SimpleNamespace(
                manual_seed=lambda _seed: None,
                cuda=SimpleNamespace(manual_seed_all=lambda _seed: None),
            )
        raise AssertionError(name)

    monkeypatch.setattr("discovery_os.sidecars.generators.require_module", fake_module)
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.pymatgen_to_cif",
        lambda structure, *, max_bytes: "data_fixture\n_cell_length_a 1\n",
    )
    canonical_counter = iter(range(10_000))

    def fake_canonicalize(structure: object, **_kwargs: Any) -> Any:
        index = next(canonical_counter)
        return SimpleNamespace(
            canonical_cif=f"data_fixture_{index}\n_cell_length_a 1\n",
            structure_hash=f"hash-{index}",
            source_atom_count=1,
            primitive_atom_count=1,
            conventional_atom_count=1,
            space_group_symbol="P1",
            space_group_number=1,
        )

    def fake_group(structures: tuple[Any, ...], **_kwargs: Any) -> Any:
        return SimpleNamespace(
            groups=tuple(
                SimpleNamespace(representative_index=index)
                for index, _structure in enumerate(structures)
            )
        )

    monkeypatch.setattr(
        "discovery_os.sidecars.generators.canonicalize_crystal_structure",
        fake_canonicalize,
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.group_crystal_structures",
        fake_group,
    )
    checkpoint_path = tmp_path / "mattergen-checkpoint"
    checkpoint_path.mkdir()
    (checkpoint_path / "last.ckpt").write_bytes(b"fixture")
    runtime = MatterGenGenerator(
        pretrained_name="mattergen_base",
        checkpoint_path=str(checkpoint_path),
        device="cpu",
    )
    request = _generation_request(candidate_count=2)

    first = runtime.generate(request)
    second = runtime.generate(request)

    assert len(first.candidates) == len(second.candidates) == 2
    assert counts == {"checkpoint": 1, "prepare": 1, "generate": 2}


def test_mattergen_replaces_structurematcher_duplicates_before_returning_batch(
    monkeypatch,
) -> None:
    class RawStructure:
        def __init__(self, identity: str) -> None:
            self.identity = identity

    class FakeGenerator:
        properties_to_condition_on: dict[str, object] = {}
        diffusion_guidance_factor: float = 0.0

        def __init__(self) -> None:
            self.batch_sizes: list[int] = []

        def generate(self, *, batch_size: int, num_batches: int, output_dir: str) -> list[Any]:
            assert num_batches == 1
            assert output_dir
            self.batch_sizes.append(batch_size)
            if len(self.batch_sizes) == 1:
                return [RawStructure("duplicate"), RawStructure("duplicate")]
            return [RawStructure("replacement")]

    def fake_canonicalize(structure: RawStructure, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            canonical_cif=f"data_{structure.identity}\n_cell_length_a 4\n",
            structure_hash=f"hash-{structure.identity}",
            source_atom_count=2,
            primitive_atom_count=2,
            conventional_atom_count=2,
            space_group_symbol="P1",
            space_group_number=1,
        )

    def fake_group(structures: tuple[Any, ...], **_kwargs: Any) -> Any:
        indexes: dict[str, list[int]] = {}
        for index, structure in enumerate(structures):
            indexes.setdefault(structure.structure_hash, []).append(index)
        return SimpleNamespace(
            groups=tuple(
                SimpleNamespace(representative_index=members[0], member_indices=tuple(members))
                for members in indexes.values()
            )
        )

    monkeypatch.setattr(
        "discovery_os.sidecars.generators.pymatgen_to_cif",
        lambda structure, *, max_bytes: f"data_raw_{structure.identity}\n",
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.canonicalize_crystal_structure",
        fake_canonicalize,
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.group_crystal_structures",
        fake_group,
    )
    monkeypatch.setattr("discovery_os.sidecars.generators._seed_mattergen", lambda _seed: None)
    model = FakeGenerator()
    runtime = MatterGenGenerator(device="cpu")
    runtime._model = model
    runtime._resolved_device = "cpu"

    batch = runtime.generate(_generation_request(candidate_count=2))

    assert model.batch_sizes == [2, 1]
    assert len(batch.candidates) == 2
    assert {
        item.attributes["crystal_identity"]["canonical_structure_sha256"]
        for item in batch.candidates
    } == {"hash-duplicate", "hash-replacement"}
    assert all(not item.representations[0].canonical for item in batch.candidates)
    assert {
        item.representations[0].value for item in batch.candidates
    } == {"data_raw_duplicate", "data_raw_replacement"}
    assert [item.provenance["raw_generation_seed"] for item in batch.candidates] == [7, 8]
    funnel = batch.candidates[0].attributes["generation_funnel"]
    assert funnel == {
        "requested_samples": 2,
        "raw_model_structures": 3,
        "parsed_structures": 3,
        "exact_file_unique": 2,
        "crystallographically_unique": 2,
        "geometry_valid": 2,
        "raw_geometry_valid": 3,
        "requested_unique_candidates": 2,
        "parse_rejected": 0,
        "geometry_rejected": 0,
        "canonicalization_rejected": 0,
        "duplicates_removed": 1,
        "generation_rounds": 2,
    }
    assert any("duplicates_removed=1" in warning for warning in batch.warnings)
    assert any("StructureMatcher removed 1" in warning for warning in batch.warnings)
    assert batch.candidates[0].attributes["generation_funnel_hashes"] == {
        "exact_file_sha256s": sorted(
            {
                item.provenance["source_exact_sha256"]
                for item in batch.candidates
            }
        )
    }


def test_esm_adapter_uses_pinned_biohub_esm3_api(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, Any] = {}

    class FakeTensor:
        def __init__(self, data: Any) -> None:
            self.data = data

        def dim(self) -> int:
            value = self.data
            dimensions = 0
            while isinstance(value, list):
                dimensions += 1
                value = value[0]
            return dimensions

        @property
        def shape(self) -> tuple[int, ...]:
            value = self.data
            shape: list[int] = []
            while isinstance(value, list):
                shape.append(len(value))
                value = value[0]
            return tuple(shape)

        def __getitem__(self, key: Any) -> "FakeTensor":
            return FakeTensor(self.data[key])

        def mean(self, *, dim: int, keepdim: bool) -> "FakeTensor":
            assert dim == 0 and keepdim is True
            width = len(self.data[0])
            row = [sum(item[index] for item in self.data) / len(self.data) for index in range(width)]
            return FakeTensor([row])

        def tolist(self) -> Any:
            return self.data

    class FakeProtein:
        def __init__(self, *, sequence: str) -> None:
            self.sequence = sequence

    class FakeLogitsConfig:
        def __init__(self, *, sequence: bool, return_embeddings: bool) -> None:
            calls["config"] = (sequence, return_embeddings)

    class FakeModel:
        def to(self, device: str) -> "FakeModel":
            calls["device"] = device
            return self

        def eval(self) -> None:
            calls["eval"] = True

        def encode(self, protein: FakeProtein) -> object:
            calls["sequence"] = protein.sequence
            return object()

        def logits(self, protein_tensor: object, config: FakeLogitsConfig) -> Any:
            return SimpleNamespace(
                embeddings=FakeTensor([[[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
            )

    class FakeESM3:
        @classmethod
        def from_pretrained(cls, name: str) -> FakeModel:
            calls["model_name"] = name
            return FakeModel()

    def fake_module(name: str, *, install_hint: str) -> Any:
        if name == "esm.models.esm3":
            return SimpleNamespace(ESM3=FakeESM3)
        if name == "esm.sdk.api":
            return SimpleNamespace(ESMProtein=FakeProtein, LogitsConfig=FakeLogitsConfig)
        if name == "esm.pretrained":
            return SimpleNamespace(data_root=lambda _model: None)
        raise AssertionError(name)

    monkeypatch.setattr("discovery_os.sidecars.experts.require_module", fake_module)
    candidate = Candidate(
        candidate_id="protein-1",
        candidate_type=CandidateType.PROTEIN,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[
            CandidateRepresentation(kind=RepresentationKind.PROTEIN_SEQUENCE, value="AC")
        ],
    )
    candidate = candidate.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate.candidate_id,
                version=1,
                content_hash=candidate_content_hash(candidate),
            )
        }
    )
    request = ExpertFeatureRequest(
        workspace_entity_id="primary",
        candidate=candidate,
        goal=_goal(),
        modality=ScientificModality.PROTEIN_SEQUENCE,
        feature_space="esm-sequence-v1",
        cycle=0,
        seed=2,
    )

    snapshot = tmp_path / "esm-snapshot"
    weight = snapshot / "data" / "weights" / "esm3_sm_open_v1.pth"
    weight.parent.mkdir(parents=True)
    weight.write_bytes(b"fixture")
    result = ESMExpert(
        model_name="esm3_sm_open_v1",
        snapshot_path=str(snapshot),
        device="cpu",
    ).encode(request)

    assert numeric_tensor_data(result.values) == ([1, 2], [2.0, 3.0])
    assert calls == {
        "model_name": "esm3_sm_open_v1",
        "device": "cpu",
        "eval": True,
        "sequence": "AC",
        "config": (True, True),
    }
