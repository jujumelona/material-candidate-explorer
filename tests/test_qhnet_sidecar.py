from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from discovery_os.fusion_schemas import ExpertFeatureRequest, ScientificModality, TensorRole
from discovery_os.hashing import candidate_content_hash
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
from discovery_os.sidecars import QHNetExpert
from discovery_os.sidecars.errors import CandidateConversionError, ModelExecutionError
from discovery_os.sidecars.qhnet import (
    QHNET_ARCHIVE_SHA256,
    QHNET_COMPONENT_ID,
    QHNET_SOURCE_REVISION,
    QHNetSourceAttestation,
    attest_qhnet_bundle,
    load_qhnet_runtime_config,
    verify_qhnet_source_bundle,
)
from discovery_os.sidecars.weight_binding import WeightBindingError
from discovery_os.sidecars.weight_binding import directory_inventory_sha256


def _config_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "source_revision": QHNET_SOURCE_REVISION,
        "model_version": "QHNet_w_bias",
        "dataset_id": "water-dft",
        "dtype": "float32",
        "basis": "PBE/def2-SVP with QHNet transformed orbital order",
        "hamiltonian_unit": "hartree",
        "position_unit": "angstrom",
        "position_scale_to_bohr": 1.8897261258369282,
        "molecular_charge": 0,
        "spin_multiplicity": 1,
        "allowed_atomic_number_sequences": [[8, 1, 1]],
    }


def _write_config(path: Path) -> Path:
    path.write_text(json.dumps(_config_payload()), encoding="utf-8")
    return path


def _candidate(*, atomic_order: str = "OHH") -> Candidate:
    symbols = {
        "OHH": ("O", "H", "H"),
        "HOH": ("H", "O", "H"),
    }[atomic_order]
    xyz = "3\nwater\n" + "\n".join(
        f"{symbol} {index}.0 0.0 0.0" for index, symbol in enumerate(symbols)
    )
    candidate = Candidate(
        candidate_id=f"water-{atomic_order.lower()}",
        candidate_type=CandidateType.SMALL_MOLECULE,
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        representations=[
            CandidateRepresentation(kind=RepresentationKind.XYZ, value=xyz, canonical=True)
        ],
        attributes={"charge": 0, "spin_multiplicity": 1},
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


def _request(candidate: Candidate) -> ExpertFeatureRequest:
    return ExpertFeatureRequest(
        workspace_entity_id="primary",
        candidate=candidate,
        goal=DiscoveryGoal(
            goal_id="qhnet-goal",
            domain=DiscoveryDomain.GENERAL_MATERIALS,
            title="Hamiltonian prediction",
            scientific_question="What Hamiltonian does the selected QHNet checkpoint predict?",
            objectives=[
                PropertyObjective(
                    property_name="hamiltonian_matrix",
                    direction=ObjectiveDirection.TARGET,
                    target_value=0.0,
                )
            ],
            validation_profile_id="qhnet-fixture",
            candidate_types=[CandidateType.SMALL_MOLECULE],
        ),
        modality=ScientificModality.ELECTRONIC_STRUCTURE,
        feature_space="qhnet-hamiltonian-v1",
        cycle=0,
        seed=0,
    )


def _fake_source(tmp_path: Path) -> QHNetSourceAttestation:
    root = tmp_path / "source"
    qhnet_root = root / "OpenDFT" / "QHNet"
    qhnet_root.mkdir(parents=True, exist_ok=True)
    return QHNetSourceAttestation(
        archive_root=root,
        qhnet_root=qhnet_root,
        archive_sha256=QHNET_ARCHIVE_SHA256,
        source_inventory_sha256="a" * 64,
    )


def _adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    checkpoint_payload: object | None = None,
) -> QHNetExpert:
    checkpoint = tmp_path / "model.pt"
    torch.save(
        checkpoint_payload
        if checkpoint_payload is not None
        else {"state_dict": {"weight": torch.tensor([1.0])}},
        checkpoint,
    )
    config = _write_config(tmp_path / "config.json")
    source = _fake_source(tmp_path)
    monkeypatch.setattr(
        "discovery_os.sidecars.experts.verify_qhnet_source_bundle",
        lambda _: source,
    )
    return QHNetExpert(
        source_path=str(source.archive_root),
        checkpoint_path=str(checkpoint),
        config_path=str(config),
        device="cpu",
    )


def test_config_and_checkpoint_are_one_attested_bundle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    config = _write_config(tmp_path / "config.json")
    parsed = load_qhnet_runtime_config(config)
    assert parsed.model_version == "QHNet_w_bias"
    assert parsed.max_orbital_dimension == 24

    attested = attest_qhnet_bundle(checkpoint, config)
    assert attested.revision.startswith("bundle-sha256:")
    config.write_text(json.dumps({**_config_payload(), "dtype": "float64"}), encoding="utf-8")
    with pytest.raises(WeightBindingError, match="does not match"):
        attest_qhnet_bundle(checkpoint, config, declared_revision=attested.revision)


def test_config_rejects_unknown_fields_and_unbounded_hamiltonian(tmp_path: Path) -> None:
    unknown = {**_config_payload(), "invented_architecture": True}
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps(unknown), encoding="utf-8")
    with pytest.raises(WeightBindingError, match="unknown"):
        load_qhnet_runtime_config(path)

    oversized = _config_payload()
    oversized["allowed_atomic_number_sequences"] = [[8] * 19]
    path.write_text(json.dumps(oversized), encoding="utf-8")
    with pytest.raises(WeightBindingError, match="65,536"):
        load_qhnet_runtime_config(path)


def test_source_marker_and_executable_inventory_are_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import discovery_os.sidecars.qhnet as qhnet

    root = tmp_path / "source"
    qhnet_root = root / "OpenDFT" / "QHNet"
    member = qhnet_root / "models" / "__init__.py"
    member.parent.mkdir(parents=True)
    member.write_text("VALUE = 1\n", encoding="utf-8")
    digest = hashlib.sha256(member.read_bytes()).hexdigest()
    monkeypatch.setattr(qhnet, "_REQUIRED_SOURCE_FILES", {"models/__init__.py": digest})
    source_inventory = directory_inventory_sha256(
        root,
        exclude_names=frozenset({".discovery-source.json"}),
        exclude_directory_names=frozenset(),
    )
    (root / ".discovery-source.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "component_id": QHNET_COMPONENT_ID,
                "revision": QHNET_SOURCE_REVISION,
                "sha256": QHNET_ARCHIVE_SHA256,
                "inventory_sha256": source_inventory,
            }
        ),
        encoding="utf-8",
    )
    verified = verify_qhnet_source_bundle(root)
    assert verified.qhnet_root == qhnet_root.resolve()
    member.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(WeightBindingError, match="inventory differs"):
        verify_qhnet_source_bundle(root)


def test_checkpoint_loading_uses_strict_state_dict_and_official_get_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import discovery_os.sidecars.experts as experts

    adapter = _adapter(tmp_path, monkeypatch)

    class FakeModel:
        def __init__(self) -> None:
            self.strict: bool | None = None
            self.device: str | None = None
            self.evaluating = False

        def load_state_dict(self, state: Any, *, strict: bool) -> None:
            assert set(state) == {"weight"}
            self.strict = strict

        def set(self, device: str) -> None:
            self.device = device

        def to(self, **kwargs: Any) -> FakeModel:
            assert kwargs["dtype"] is torch.float32
            return self

        def eval(self) -> None:
            self.evaluating = True

    model = FakeModel()
    versions: list[str] = []
    monkeypatch.setattr(
        experts,
        "_load_qhnet_models_package",
        lambda _: SimpleNamespace(
            get_model=lambda args: versions.append(args.version) or model,
        ),
    )
    monkeypatch.setattr(
        experts,
        "require_module",
        lambda name, **_: torch if name == "torch" else SimpleNamespace(Data=lambda **kw: kw),
    )
    runtime = adapter._load_model("cpu")
    assert runtime["model"] is model
    assert versions == ["QHNet_w_bias"]
    assert model.strict is True
    assert model.device == "cpu"
    assert model.evaluating is True


def test_graph_preprocessing_and_full_hamiltonian_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import discovery_os.sidecars.experts as experts

    adapter = _adapter(tmp_path, monkeypatch)

    class FakePositions(list):
        def tolist(self) -> list[list[float]]:
            return list(self)

    class FakeAtoms:
        def __init__(self, atomic_numbers: list[int]) -> None:
            self.atomic_numbers = atomic_numbers

        def get_pbc(self) -> list[bool]:
            return [False, False, False]

        def get_atomic_numbers(self) -> list[int]:
            return self.atomic_numbers

        def get_positions(self) -> FakePositions:
            return FakePositions([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    class FakeData:
        def __init__(self, **values: Any) -> None:
            self.__dict__.update(values)

    class FakeModel:
        def __call__(self, data: FakeData) -> dict[str, torch.Tensor]:
            assert data.atoms.tolist() == [[8], [1], [1]]
            assert data.batch.tolist() == [0, 0, 0]
            assert data.ptr.tolist() == [0, 3]
            assert data.pos[1, 0].item() == pytest.approx(1.8897261258369282)
            assert torch.is_grad_enabled() is False
            return {"hamiltonian": torch.eye(24, dtype=torch.float32).unsqueeze(0)}

    def fake_candidate_to_ase(candidate: Candidate, **_: Any) -> FakeAtoms:
        symbols = [line.split()[0] for line in candidate.representations[0].value.splitlines()[2:]]
        return FakeAtoms([{"H": 1, "O": 8}[symbol] for symbol in symbols])

    monkeypatch.setattr(experts, "candidate_to_ase", fake_candidate_to_ase)
    adapter._resolved_device = "cpu"
    adapter._model = {
        "model": FakeModel(),
        "torch": torch,
        "data_class": FakeData,
        "dtype": torch.float32,
    }
    result = adapter.encode(_request(_candidate()))
    assert result.tensor_role == TensorRole.HAMILTONIAN
    assert len(result.values) == 24
    assert len(result.entity_ids) == 24
    assert result.properties == ()
    assert result.unit_semantics == {"hamiltonian_matrix_element": "hartree"}
    assert any("not fabricated" in warning for warning in result.warnings)

    with pytest.raises(CandidateConversionError, match="outside the configured checkpoint scope"):
        adapter.encode(_request(_candidate(atomic_order="HOH")))


def test_checkpoint_with_extra_payload_key_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import discovery_os.sidecars.experts as experts

    adapter = _adapter(
        tmp_path,
        monkeypatch,
        checkpoint_payload={
            "state_dict": {"weight": torch.tensor([1.0])},
            "unreviewed": "payload",
        },
    )
    monkeypatch.setattr(
        experts,
        "_load_qhnet_models_package",
        lambda _: SimpleNamespace(get_model=lambda _: SimpleNamespace()),
    )
    monkeypatch.setattr(
        experts,
        "require_module",
        lambda name, **_: torch if name == "torch" else SimpleNamespace(Data=dict),
    )
    with pytest.raises(ModelExecutionError, match="official state_dict"):
        adapter._load_model("cpu")


def test_checkpoint_mutation_after_attestation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = _adapter(tmp_path, monkeypatch)
    torch.save({"state_dict": {"weight": torch.tensor([2.0])}}, adapter.checkpoint_path)

    with pytest.raises(ModelExecutionError, match="changed after runtime attestation"):
        adapter._load_model("cpu")


def test_qhnet_preflight_requires_exact_compatibility_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from discovery_os.sidecars import cli

    monkeypatch.setattr(cli, "_module_available", lambda _: True)
    versions = dict(cli._QHNET_RUNTIME_VERSIONS)
    monkeypatch.setattr(
        cli.importlib.metadata,
        "version",
        lambda name: "2.2.0+cu118" if name == "torch" else versions[name],
    )
    cli._validate_runtime_dependency("qhnet")

    monkeypatch.setattr(
        cli.importlib.metadata,
        "version",
        lambda name: "2.3.0" if name == "torch" else versions[name],
    )
    with pytest.raises(ValueError, match="torch==2.2.0"):
        cli._validate_runtime_dependency("qhnet")
