from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from discovery_os.configured_experts import build_expert_registry_from_environment
from discovery_os.fusion_schemas import ExpertFeatureRequest, ScientificModality
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
from discovery_os.sidecars.errors import CandidateConversionError
from discovery_os.sidecars.experts import (
    CHGNetExpert,
    ChempropExpert,
    ESMExpert,
    MatterSimExpert,
    PySCFExpert,
    RNAFMExpert,
    UMAExpert,
    UniMolExpert,
    _ase_force_result,
)


def _candidate(
    candidate_type: CandidateType,
    kind: RepresentationKind,
    *,
    attributes: dict[str, Any] | None = None,
) -> Candidate:
    candidate = Candidate(
        candidate_id=f"candidate-{candidate_type}-{kind}",
        candidate_type=candidate_type,
        domain=(
            DiscoveryDomain.MEDICINAL_CHEMISTRY
            if candidate_type == CandidateType.SMALL_MOLECULE
            else DiscoveryDomain.GENERAL_MATERIALS
        ),
        representations=[
            CandidateRepresentation(kind=kind, value="fixture", canonical=True)
        ],
        attributes=attributes or {},
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


def _request(
    candidate: Candidate,
    *,
    modality: ScientificModality,
    feature_space: str,
) -> ExpertFeatureRequest:
    goal = DiscoveryGoal(
        goal_id="route-correctness",
        domain=candidate.domain,
        title="Validate one specialist route",
        scientific_question="Is this request scientifically compatible with the specialist?",
        objectives=[
            PropertyObjective(
                property_name="target",
                direction=ObjectiveDirection.MAXIMIZE,
            )
        ],
        validation_profile_id="route-correctness-v1",
        candidate_types=[candidate.candidate_type],
    )
    return ExpertFeatureRequest(
        workspace_entity_id="primary",
        candidate=candidate,
        goal=goal,
        modality=modality,
        feature_space=feature_space,
        cycle=0,
        seed=1,
    )


def test_configured_routes_do_not_mislabel_force_fields_as_electronic_structure() -> None:
    default = {
        item.expert_id: item
        for item in build_expert_registry_from_environment(
            environ={}, include_unconfigured=True
        ).describe()
    }

    pyscf = default["pyscf"]
    assert set(pyscf.supported_candidate_types) == {"small_molecule", "custom"}
    assert set(pyscf.supported_representations) == {"xyz", "sdf"}

    for expert_id in ("mattersim", "chgnet"):
        descriptor = default[expert_id]
        assert descriptor.modalities == ["crystal_material"]
        assert set(descriptor.supported_representations) == {"cif", "poscar"}

    chemprop = default["chemprop"]
    assert set(chemprop.supported_candidate_types) == {"small_molecule", "catalyst"}
    assert chemprop.supported_representations == ["smiles"]

    uma = default["uma"]
    assert uma.modalities == ["crystal_material"]
    assert set(uma.supported_representations) == {"cif", "poscar"}
    assert uma.metadata["task_name"] == "omat"

    esm = default["esm"]
    assert esm.modalities == ["protein_sequence"]
    assert {
        (route.modality, tuple(route.representation_kinds))
        for route in esm.routes
    } == {
        ("protein_sequence", ("protein_sequence",)),
        ("protein_sequence", ("fasta",)),
        ("protein_sequence", ("pdb",)),
    }

    rnafm = default["rnafm"]
    assert rnafm.modalities == ["rna_sequence"]
    assert {tuple(route.representation_kinds) for route in rnafm.routes} == {
        ("rna_sequence",),
        ("fasta",),
    }


def test_uma_descriptor_is_task_aware_and_rejects_unreviewed_tasks() -> None:
    omol = {
        item.expert_id: item
        for item in build_expert_registry_from_environment(
            environ={"UMA_TASK_NAME": "omol"}, include_unconfigured=True
        ).describe()
    }["uma"]
    assert omol.modalities == ["molecule_3d"]
    assert omol.supported_candidate_types == ["small_molecule"]
    assert set(omol.supported_representations) == {"xyz", "extxyz", "sdf"}
    assert omol.metadata["task_name"] == "omol"

    with pytest.raises(ValueError, match="explicit reviewed route"):
        build_expert_registry_from_environment(
            environ={"UMA_TASK_NAME": "oc20"}, include_unconfigured=True
        )


def test_direct_adapters_reject_requests_outside_their_declared_routes(
    tmp_path: Path,
) -> None:
    pyscf = PySCFExpert()
    with pytest.raises(CandidateConversionError, match="candidate type"):
        pyscf.encode(
            _request(
                _candidate(CandidateType.CRYSTAL, RepresentationKind.CIF),
                modality=ScientificModality.ELECTRONIC_STRUCTURE,
                feature_space="pyscf-orbital-v1",
            )
        )
    assert not pyscf.loaded

    mattersim = MatterSimExpert()
    with pytest.raises(CandidateConversionError, match="required representations"):
        mattersim.encode(
            _request(
                _candidate(CandidateType.CRYSTAL, RepresentationKind.XYZ),
                modality=ScientificModality.CRYSTAL_MATERIAL,
                feature_space="mattersim-atomic-v1",
            )
        )
    assert not mattersim.loaded

    chgnet = CHGNetExpert()
    with pytest.raises(CandidateConversionError, match="required representations"):
        chgnet.encode(
            _request(
                _candidate(CandidateType.CRYSTAL, RepresentationKind.EXTXYZ),
                modality=ScientificModality.CRYSTAL_MATERIAL,
                feature_space="chgnet-atomic-v1",
            )
        )
    assert not chgnet.loaded

    checkpoint = tmp_path / "chemprop.ckpt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    chemprop = ChempropExpert(
        checkpoint_path=str(checkpoint),
        property_names=("fixture_property",),
        property_units=("dimensionless",),
    )
    with pytest.raises(CandidateConversionError, match="candidate type"):
        chemprop.encode(
            _request(
                _candidate(CandidateType.POLYMER, RepresentationKind.POLYMER_REPEAT_UNIT),
                modality=ScientificModality.MOLECULE_2D,
                feature_space="chemprop-mpn-v1",
            )
        )
    assert not chemprop.loaded

    omat = UMAExpert(task_name="omat")
    with pytest.raises(CandidateConversionError, match="candidate type"):
        omat.encode(
            _request(
                _candidate(CandidateType.SMALL_MOLECULE, RepresentationKind.SDF),
                modality=ScientificModality.CRYSTAL_MATERIAL,
                feature_space="uma-atomic-v1",
            )
        )
    assert not omat.loaded

    omol = UMAExpert(task_name="omol")
    with pytest.raises(CandidateConversionError, match="candidate type"):
        omol.encode(
            _request(
                _candidate(CandidateType.CRYSTAL, RepresentationKind.CIF),
                modality=ScientificModality.MOLECULE_3D,
                feature_space="uma-atomic-v1",
            )
        )
    assert not omol.loaded

    unimol = UniMolExpert(device="cpu")
    with pytest.raises(CandidateConversionError, match="feature space"):
        unimol.encode(
            _request(
                _candidate(CandidateType.SMALL_MOLECULE, RepresentationKind.SMILES),
                modality=ScientificModality.MOLECULE_2D,
                feature_space="forged-unimol-space",
            )
        )
    assert not unimol.loaded

    esm = ESMExpert(device="cpu")
    with pytest.raises(CandidateConversionError, match="modality"):
        esm.encode(
            _request(
                _candidate(CandidateType.PROTEIN, RepresentationKind.PROTEIN_SEQUENCE),
                modality=ScientificModality.PROTEIN_STRUCTURE,
                feature_space="esm-sequence-v1",
            )
        )
    assert not esm.loaded

    rnafm = RNAFMExpert(device="cpu")
    with pytest.raises(CandidateConversionError, match="modality"):
        rnafm.encode(
            _request(
                _candidate(CandidateType.RNA, RepresentationKind.RNA_SEQUENCE),
                modality=ScientificModality.RNA_STRUCTURE,
                feature_space="rnafm-t12-v1",
            )
        )
    assert not rnafm.loaded


def test_chemprop_requires_named_unit_bearing_task_outputs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "chemprop.ckpt"
    checkpoint.write_bytes(b"fixture-checkpoint")

    with pytest.raises(ValueError, match="property name"):
        ChempropExpert(
            checkpoint_path=str(checkpoint),
            property_names=(),
            property_units=(),
        )
    with pytest.raises(ValueError, match="one non-blank property unit"):
        ChempropExpert(
            checkpoint_path=str(checkpoint),
            property_names=("solubility",),
            property_units=(),
        )


def test_uma_tasks_validate_periodicity_before_model_loading(monkeypatch: Any) -> None:
    class FakeAtoms:
        def __init__(self, pbc: list[bool]) -> None:
            self._pbc = pbc

        def get_pbc(self) -> list[bool]:
            return self._pbc

    periodic_request = _request(
        _candidate(CandidateType.CRYSTAL, RepresentationKind.CIF),
        modality=ScientificModality.CRYSTAL_MATERIAL,
        feature_space="uma-atomic-v1",
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.experts.candidate_to_ase",
        lambda *_args, **_kwargs: FakeAtoms([False, False, False]),
    )
    omat = UMAExpert(task_name="omat")
    with pytest.raises(CandidateConversionError, match="fully periodic"):
        omat.encode(periodic_request)
    assert not omat.loaded

    molecular_request = _request(
        _candidate(CandidateType.SMALL_MOLECULE, RepresentationKind.XYZ),
        modality=ScientificModality.MOLECULE_3D,
        feature_space="uma-atomic-v1",
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.experts.candidate_to_ase",
        lambda *_args, **_kwargs: FakeAtoms([True, True, True]),
    )
    omol = UMAExpert(task_name="omol")
    with pytest.raises(CandidateConversionError, match="non-periodic"):
        omol.encode(molecular_request)
    assert not omol.loaded


@pytest.mark.parametrize("source", ["MatterSim", "UMA:uma-s-1p2"])
def test_ase_force_experts_report_total_and_per_atom_energy_without_claiming_hull_energy(
    source: str,
) -> None:
    class FakeAtoms:
        def __len__(self) -> int:
            return 3

        def get_potential_energy(self) -> float:
            return -12.0

        def get_forces(self) -> list[list[float]]:
            return [[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3]]

        def get_stress(self, *, voigt: bool) -> Any:
            assert voigt is False
            raise RuntimeError("fixture has no stress")

    result = _ase_force_result(FakeAtoms(), source=source)
    properties = {item.property_name: item for item in result.properties}

    assert properties["energy"].value == pytest.approx(-12.0)
    assert properties["energy"].unit == "eV"
    assert properties["energy_per_atom"].value == pytest.approx(-4.0)
    assert properties["energy_per_atom"].unit == "eV/atom"
    assert properties["energy_per_atom"].source == source
    assert "energy_above_hull" not in properties
    assert result.unit_semantics["energy_per_atom"] == "eV/atom"
    assert result.warnings == (
        "upstream calculator did not expose stress for this structure",
    )


def test_pyscf_uhf_frontiers_are_selected_across_spin_channels(monkeypatch: Any) -> None:
    class FakeAtoms:
        def get_pbc(self) -> list[bool]:
            return [False, False, False]

        def get_chemical_symbols(self) -> list[str]:
            return ["N", "O"]

        def get_positions(self) -> Any:
            return SimpleNamespace(tolist=lambda: [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]])

    molecule = SimpleNamespace(nelec=(3, 2))

    class FakeCalculation:
        converged = True
        mo_energy = (
            [-2.0, -1.0, -0.2, 0.4],
            [-2.1, -0.8, 0.1, 0.3],
        )

        def kernel(self) -> float:
            return -108.5

    fake_gto = SimpleNamespace(M=lambda **_kwargs: molecule)
    fake_scf = SimpleNamespace(
        RHF=lambda _molecule: pytest.fail("open-shell request must not use RHF"),
        UHF=lambda _molecule: FakeCalculation(),
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.experts.candidate_to_ase",
        lambda *_args, **_kwargs: FakeAtoms(),
    )
    adapter = PySCFExpert()
    adapter._model = (fake_gto, fake_scf)
    adapter._resolved_device = "cpu"
    request = _request(
        _candidate(
            CandidateType.SMALL_MOLECULE,
            RepresentationKind.XYZ,
            attributes={"spin": 1},
        ),
        modality=ScientificModality.ELECTRONIC_STRUCTURE,
        feature_space="pyscf-orbital-v1",
    )

    result = adapter.encode(request)

    assert result.values == [[-2.0, -1.0, -0.2, 0.4, -2.1, -0.8, 0.1, 0.3]]
    properties = {item.property_name: item.value for item in result.properties}
    assert properties["homo_energy"] == pytest.approx(-0.2)
    assert properties["lumo_energy"] == pytest.approx(0.1)
    assert properties["homo_lumo_gap"] == pytest.approx(0.3)
