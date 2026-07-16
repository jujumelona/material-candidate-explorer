from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from discovery_os import novelty as novelty_module
from discovery_os.artifacts import ArtifactStore
from discovery_os.crystal_identity import PymatgenRequiredError
from discovery_os.dft_handoff import (
    DFTInputManifest,
    PortablePeriodicDFTInputBackend,
)
from discovery_os.fusion_schemas import ContentArtifactRef
from discovery_os.hashing import candidate_content_hash
from discovery_os.novelty import (
    ExternalNoveltyOutcome,
    MaterialsProjectStructureLookup,
    NoveltyStatus,
    ProjectNoveltyIndex,
    StagedNoveltyAssessor,
    scientific_fingerprint,
)
from discovery_os.schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    RepresentationKind,
)


CIF = """data_LiO
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Li1 Li 0 0 0
O1 O 0.5 0.5 0.5
"""

POSCAR = """Li O seed
1.0
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
Li O
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""


def _candidate(
    candidate_id: str,
    *,
    cif: str = CIF,
    formula: str = "LiO",
    include_poscar: bool = True,
) -> Candidate:
    representations = [
        CandidateRepresentation(
            kind=RepresentationKind.CIF,
            value=cif,
            media_type="chemical/x-cif",
            canonical=True,
        ),
        CandidateRepresentation(
            kind=RepresentationKind.CHEMICAL_FORMULA,
            value=formula,
            media_type="text/plain",
        ),
    ]
    if include_poscar:
        representations.append(
            CandidateRepresentation(
                kind=RepresentationKind.POSCAR,
                value=POSCAR,
                media_type="text/plain",
            )
        )
    draft = Candidate(
        candidate_id=candidate_id,
        candidate_type=CandidateType.CRYSTAL,
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        representations=representations,
        provenance={"fixture_id": candidate_id},
    )
    reference = CandidateRef(
        candidate_id=candidate_id,
        version=1,
        content_hash=candidate_content_hash(draft),
    )
    return draft.model_copy(update={"candidate_ref": reference})


def _molecule(candidate_id: str, smiles: str = "CCO") -> Candidate:
    draft = Candidate(
        candidate_id=candidate_id,
        candidate_type=CandidateType.SMALL_MOLECULE,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.SMILES,
                value=smiles,
                media_type="chemical/x-daylight-smiles",
                canonical=True,
            )
        ],
        provenance={"fixture_id": candidate_id},
    )
    reference = CandidateRef(
        candidate_id=candidate_id,
        version=1,
        content_hash=candidate_content_hash(draft),
    )
    return draft.model_copy(update={"candidate_ref": reference})


def _rocksalt_cifs() -> tuple[str, str, str]:
    core = pytest.importorskip("pymatgen.core")
    cif_module = pytest.importorskip("pymatgen.io.cif")
    lattice = core.Lattice.cubic(5.64)
    base = core.Structure(
        lattice,
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    reordered = core.Structure(
        lattice,
        ["Cl", "Na"],
        [[0.5, 0.5, 0.5], [0.0, 0.0, 0.0]],
    )
    supercell = base.copy()
    supercell.make_supercell([2, 1, 1])
    return tuple(
        str(cif_module.CifWriter(item))
        for item in (base, reordered, supercell)
    )


class _NoMatchLookup:
    provider_id = "fixture-database"

    def lookup(self, _candidate: Candidate) -> ExternalNoveltyOutcome:
        return ExternalNoveltyOutcome(
            provider_id=self.provider_id,
            status=NoveltyStatus.NO_MATCH,
            method="fixture-structure-match-v1",
            query_count=1,
        )


def test_crystal_novelty_groups_reordered_and_supercell_batch_and_history() -> None:
    base_cif, reordered_cif, supercell_cif = _rocksalt_cifs()
    first = _candidate("candidate-a", cif=base_cif, formula="NaCl", include_poscar=False)
    duplicate = _candidate("candidate-b", cif=reordered_cif, formula="NaCl", include_poscar=False)
    supercell = _candidate("candidate-c", cif=supercell_cif, formula="NaCl", include_poscar=False)
    history_match = _candidate("history-a", cif=base_cif, formula="NaCl", include_poscar=False)

    results = StagedNoveltyAssessor(_NoMatchLookup()).assess(
        [first, duplicate, supercell],
        project_history=ProjectNoveltyIndex([history_match]),
    )

    assert len({scientific_fingerprint(item) for item in (first, duplicate, supercell)}) == 1
    assert all(item.within_batch.status == NoveltyStatus.MATCH for item in results)
    assert all(item.within_batch.method == "pymatgen-structure-matcher-v1" for item in results)
    assert all(item.project_history.status == NoveltyStatus.MATCH for item in results)
    assert all(item.external_database.status == NoveltyStatus.NO_MATCH for item in results)
    assert all(item.overall_status == NoveltyStatus.MATCH for item in results)
    assert "not proof of universal novelty" in results[0].scope_note


def test_missing_history_and_external_provider_are_strictly_unknown() -> None:
    result = StagedNoveltyAssessor().assess([_molecule("candidate-a")])[0]

    assert result.within_batch.status == NoveltyStatus.NO_MATCH
    assert result.project_history.status == NoveltyStatus.UNKNOWN
    assert result.project_history.reason == "project_history_not_provided"
    assert result.external_database.status == NoveltyStatus.UNKNOWN
    assert result.overall_status == NoveltyStatus.UNKNOWN


def test_all_configured_no_match_stages_remain_scope_limited_no_match() -> None:
    result = StagedNoveltyAssessor(_NoMatchLookup()).assess(
        [_molecule("candidate-a")],
        project_history=ProjectNoveltyIndex(),
    )[0]

    assert result.overall_status == NoveltyStatus.NO_MATCH
    assert result.scope_note.startswith("no_match means")


def test_missing_crystal_tooling_is_unknown_not_an_exact_text_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_candidate: Candidate):
        raise PymatgenRequiredError("fixture missing optional dependency")

    monkeypatch.setattr(novelty_module, "_canonical_candidate_crystal", unavailable)
    result = StagedNoveltyAssessor(_NoMatchLookup()).assess(
        [_candidate("candidate-a")],
        project_history=ProjectNoveltyIndex(),
    )[0]

    assert result.scientific_fingerprint is None
    assert result.within_batch.status == NoveltyStatus.UNKNOWN
    assert result.within_batch.reason == "crystal_identity_dependency_not_installed"
    assert result.overall_status == NoveltyStatus.UNKNOWN


def test_materials_project_missing_key_and_api_failure_never_become_no_match() -> None:
    calls = []

    def should_not_run(_key: str):
        calls.append("called")
        raise AssertionError("factory must not run without a credential")

    missing = MaterialsProjectStructureLookup(None, rester_factory=should_not_run)
    missing_result = missing.lookup(_candidate("candidate-a"))
    assert missing_result.status == NoveltyStatus.UNKNOWN
    assert missing_result.reason == "materials_project_api_key_not_configured"
    assert calls == []

    class BrokenClient:
        def find_structure(self, *_args, **_kwargs):
            raise TimeoutError("secret-bearing provider detail")

        def close(self):
            return None

    failed = MaterialsProjectStructureLookup(
        "runtime-secret",
        rester_factory=lambda _key: BrokenClient(),
    ).lookup(_candidate("candidate-b"))
    assert failed.status == NoveltyStatus.UNKNOWN
    assert failed.reason == "materials_project_lookup_failed:TimeoutError"
    assert "runtime-secret" not in failed.model_dump_json()
    assert "secret-bearing" not in failed.model_dump_json()


def test_materials_project_structure_ids_are_preserved_as_external_matches() -> None:
    class Client:
        def get_material_ids(self, formula):
            assert formula == "LiO"
            return ["mp-3", "mp-1", "mp-2"]

        def find_structure(self, path, **kwargs):
            assert path.endswith("candidate.cif")
            assert kwargs["allow_multiple_results"] is True
            return ["mp-2", "mp-1", "mp-2"]

        def close(self):
            return None

    outcome = MaterialsProjectStructureLookup(
        "runtime-secret",
        rester_factory=lambda _key: Client(),
    ).lookup(_candidate("candidate-a"))

    assert outcome.status == NoveltyStatus.MATCH
    assert [item.record_id for item in outcome.matches] == ["mp-1", "mp-2"]
    assert outcome.composition_match_count == 3
    assert outcome.structure_match_count == 2
    assert outcome.closest_match_id == "mp-1"
    assert outcome.closest_distance is None
    assert all(item.match_kind == "tolerance-aware-structure-match" for item in outcome.matches)


def test_periodic_dft_handoff_writes_only_top_candidates_and_null_results(tmp_path) -> None:
    pytest.importorskip("pymatgen.core")
    store = ArtifactStore(tmp_path / "artifacts")
    backend = PortablePeriodicDFTInputBackend(
        calculation_type="vc-relax",
        ecutwfc_ry=70,
        ecutrho_ry=560,
        kpoint_grid=(6, 6, 6),
    )
    report = backend.prepare_inputs(
        [_candidate("candidate-a"), _candidate("candidate-b")],
        artifact_store=store,
        top_k=1,
    )

    assert report.candidates_received == 2
    assert len(report.packages) == 1
    package = report.packages[0]
    names = {item.relative_path.rsplit("/", 1)[-1] for item in package.manifest.input_artifacts}
    assert names == {"structure.cif", "POSCAR", "pw.in"}
    assert package.manifest.pseudopotentials_included is False
    assert package.manifest.calculation_executed is False
    assert package.manifest.uncalculated_properties.model_dump() == {
        "schema_version": "1.0",
        "total_energy_eV": None,
        "energy_per_atom_eV": None,
        "formation_energy_eV_per_atom": None,
        "energy_above_hull_eV_per_atom": None,
    }

    manifest = json.loads(
        store.read_bytes(
            package.manifest_artifact.relative_path,
            expected_sha256=package.manifest_artifact.sha256,
        )
    )
    assert manifest["uncalculated_properties"]["formation_energy_eV_per_atom"] is None
    assert manifest["uncalculated_properties"]["energy_above_hull_eV_per_atom"] is None
    all_files = [path.name.casefold() for path in store.root.rglob("*") if path.is_file()]
    assert "potcar" not in all_files
    assert not any(name.endswith((".upf", ".psp8", ".psml")) for name in all_files)

    pw_ref = next(item for item in package.manifest.input_artifacts if item.relative_path.endswith("/pw.in"))
    pw_text = store.read_bytes(pw_ref.relative_path).decode("utf-8")
    assert "NOT EXECUTABLE AS-IS" in pw_text
    assert "INSERT_EXTERNAL_PSEUDO_DIR" in pw_text
    assert "ecutwfc = 70" in pw_text


def test_cif_only_handoff_still_writes_quantum_espresso_skeleton(tmp_path) -> None:
    pytest.importorskip("pymatgen.core")
    store = ArtifactStore(tmp_path / "artifacts")
    report = PortablePeriodicDFTInputBackend().prepare_inputs(
        [_candidate("candidate-a", include_poscar=False)],
        artifact_store=store,
        top_k=1,
    )
    names = {
        item.relative_path.rsplit("/", 1)[-1]
        for item in report.packages[0].manifest.input_artifacts
    }
    assert names == {"structure.cif", "POSCAR", "pw.in"}
    pw_ref = next(
        item
        for item in report.packages[0].manifest.input_artifacts
        if item.relative_path.endswith("/pw.in")
    )
    pw_text = store.read_bytes(pw_ref.relative_path).decode("utf-8")
    assert "nat = 2" in pw_text
    assert "ntyp = 2" in pw_text
    assert "CELL_PARAMETERS angstrom\n4.000000000000" in pw_text
    assert "ATOMIC_POSITIONS crystal\nLi 0.000000000000" in pw_text
    assert "EXTERNAL_PSEUDO_Li.UPF" in pw_text
    assert "EXTERNAL_PSEUDO_O.UPF" in pw_text
    assert "INSERT_NAT" not in pw_text
    assert "INSERT_NTYP" not in pw_text


def test_input_packager_exposes_execution_contract_but_never_fakes_results() -> None:
    backend = PortablePeriodicDFTInputBackend()

    with pytest.raises(NotImplementedError, match="only packages inputs"):
        backend.relax(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="only packages inputs"):
        backend.static_energy(None)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="only packages inputs"):
        backend.phonon(None)  # type: ignore[arg-type]


def test_dft_handoff_limits_shortlist_to_five_candidates(tmp_path) -> None:
    backend = PortablePeriodicDFTInputBackend()

    with pytest.raises(ValueError, match="between 1 and 5"):
        backend.prepare_inputs(
            [_candidate("candidate-a")],
            artifact_store=ArtifactStore(tmp_path / "artifacts"),
            top_k=6,
        )


def test_dft_manifest_rejects_bundled_pseudopotential() -> None:
    candidate = _candidate("candidate-a")

    def artifact(path: str) -> ContentArtifactRef:
        return ContentArtifactRef(
            artifact_id=f"artifact-{path.replace('.', '-')}",
            relative_path=path,
            sha256="a" * 64,
            media_type="text/plain",
            byte_size=1,
        )

    with pytest.raises(ValidationError, match="must not bundle POTCAR or pseudopotentials"):
        DFTInputManifest(
            backend_id="fixture-backend",
            backend_version="1.0.0",
            candidate_ref=candidate.candidate_ref,
            shortlist_rank=1,
            input_artifacts=[
                artifact("dft/structure.cif"),
                artifact("dft/pw.in"),
                artifact("dft/Li.UPF"),
            ],
            required_external_configuration=["fixture external configuration"],
        )
