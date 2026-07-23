from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from discovery_os import novelty as novelty_module
from discovery_os.artifacts import ArtifactStore
from discovery_os.crystal_identity import (
    CrystalMatchAssessment,
    CrystalMatchRelation,
    CrystalMatcherSettings,
    PymatgenRequiredError,
)
from discovery_os.dft_handoff import (
    DFTConvergencePlan,
    DFTInputManifest,
    KPointSamplingPlan,
    PeriodicDFTCalculationResult,
    PortablePeriodicDFTInputBackend,
)
from discovery_os.fusion_schemas import ContentArtifactRef
from discovery_os.hashing import candidate_content_hash
from discovery_os.novelty import (
    ExternalNoveltyOutcome,
    LIVE_MOVING_SNAPSHOT_UNPINNED,
    MaterialsProjectStructureLookup,
    NoveltyMatch,
    NoveltyStatus,
    ProjectNoveltyIndex,
    StagedNoveltyAssessor,
    reserve_external_no_match_portfolio_slot,
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


def _fixture_crystal_assessment(
    relation: CrystalMatchRelation,
) -> CrystalMatchAssessment:
    strict = CrystalMatcherSettings(
        ltol=0.02,
        stol=0.05,
        angle_tol=1.0,
        primitive_cell=True,
        scale=False,
        attempt_supercell=True,
        allow_subset=False,
        max_relative_volume_difference=0.03,
    )
    scaled = CrystalMatcherSettings(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
        allow_subset=False,
    )
    return CrystalMatchAssessment(
        relation=relation,
        strict_match=(relation == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE),
        scaled_match=(
            relation
            in {
                CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE,
                CrystalMatchRelation.SCALED_SAME_PROTOTYPE,
            }
        ),
        relative_volume_difference=(
            0.0 if relation == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE else 0.1
        ),
        strict_settings=strict,
        scaled_settings=scaled,
    )
class _NoMatchLookup:
    provider_id = "fixture-database"
    client_version = "fixture-client-1.0"
    database_version_or_release = "fixture-release-2026-07"
    matcher_policy = "fixture-strict-matcher-v1"
    matcher_settings = {"scale": False, "ltol": 0.02}

    def lookup(self, _candidate: Candidate) -> ExternalNoveltyOutcome:
        return ExternalNoveltyOutcome(
            provider_id=self.provider_id,
            client_version=self.client_version,
            database_version_or_release=self.database_version_or_release,
            retrieved_at=datetime.now(timezone.utc),
            query_sha256="a" * 64,
            matcher_policy=self.matcher_policy,
            matcher_settings=self.matcher_settings,
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


def test_dft_portfolio_reserves_one_strict_external_no_match_slot() -> None:
    first = _molecule("candidate-a", "CC")
    second = _molecule("candidate-b", "CCC")
    external = _molecule("candidate-c", "CCCC")
    unknown_assessments = StagedNoveltyAssessor().assess([first, second])
    external_assessment = StagedNoveltyAssessor(_NoMatchLookup()).assess(
        [external],
        project_history=ProjectNoveltyIndex(),
    )

    selection = reserve_external_no_match_portfolio_slot(
        base_candidate_refs=[first.candidate_ref, second.candidate_ref],
        eligible_candidate_refs=[
            first.candidate_ref,
            second.candidate_ref,
            external.candidate_ref,
        ],
        assessments=[*unknown_assessments, *external_assessment],
        top_k=2,
    )

    assert selection.selected_candidate_refs == [
        first.candidate_ref,
        external.candidate_ref,
    ]
    assert selection.reserved_external_no_match_ref == external.candidate_ref


def test_unknown_external_novelty_gets_no_portfolio_credit() -> None:
    first = _molecule("candidate-a", "CC")
    second = _molecule("candidate-b", "CCC")
    assessments = StagedNoveltyAssessor().assess([first, second])

    selection = reserve_external_no_match_portfolio_slot(
        base_candidate_refs=[first.candidate_ref],
        eligible_candidate_refs=[first.candidate_ref, second.candidate_ref],
        assessments=assessments,
        top_k=2,
    )

    # The ordinary science-gated ranking still fills the available DFT slot;
    # unknown external novelty changes neither priority nor eligibility.
    assert selection.selected_candidate_refs == [
        first.candidate_ref,
        second.candidate_ref,
    ]
    assert selection.reserved_external_no_match_ref is None


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


def test_materials_project_structure_ids_are_preserved_as_external_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        novelty_module,
        "classify_crystal_structure_relation",
        lambda *_args, **_kwargs: _fixture_crystal_assessment(
            CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE
        ),
    )

    class Client:
        def get_database_version(self):
            return "fixture-mp-release-2026-07"

        def get_material_ids(self, formula):
            assert formula == "LiO"
            return ["mp-3", "mp-1", "mp-2"]

        def find_structure(self, path, **kwargs):
            assert path.endswith("candidate.cif")
            assert kwargs["allow_multiple_results"] is True
            return ["mp-2", "mp-1", "mp-2"]

        def get_structure_by_material_id(self, material_id):
            assert material_id in {"mp-1", "mp-2"}
            return CIF

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
    assert all(item.match_kind == "strict_material_duplicate" for item in outcome.matches)
    assert outcome.database_version_or_release == "fixture-mp-release-2026-07"
    assert outcome.query_sha256 != "0" * 64
    assert outcome.matcher_settings["remote_scaled_prefilter"]["scale"] is True
    assert outcome.matcher_settings["local_strict_recheck"]["scale"] is False
    assert outcome.similarity_findings == []


def _external_outcome(
    provider_id: str,
    status: NoveltyStatus,
    *,
    moving_snapshot: bool = False,
) -> ExternalNoveltyOutcome:
    matches = (
        [
            NoveltyMatch(
                source_id=provider_id,
                record_id=f"{provider_id}-record",
                match_kind="strict_material_duplicate",
            )
        ]
        if status == NoveltyStatus.MATCH
        else []
    )
    database_release = (
        LIVE_MOVING_SNAPSHOT_UNPINNED
        if moving_snapshot
        else f"{provider_id}-release-1"
    )
    reason = None
    if status == NoveltyStatus.UNKNOWN:
        reason = "fixture_provider_unavailable"
    elif status == NoveltyStatus.NO_MATCH and moving_snapshot:
        reason = (
            f"no_match_in:{LIVE_MOVING_SNAPSHOT_UNPINNED}:"
            "not_reproducible_against_a_pinned_database_release"
        )
    return ExternalNoveltyOutcome(
        provider_id=provider_id,
        client_version="fixture-client-1",
        database_version_or_release=database_release,
        retrieved_at=datetime.now(timezone.utc),
        query_sha256=("b" if provider_id.endswith("b") else "a") * 64,
        matcher_policy="fixture-local-strict-v1",
        matcher_settings={"scale": False, "ltol": 0.02},
        status=status,
        method="fixture-structure-match-v1",
        query_count=1,
        matches=matches,
        structure_match_count=len(matches),
        closest_match_id=matches[0].record_id if matches else None,
        reason=reason,
    )


class _StaticLookup:
    client_version = "fixture-client-1"
    matcher_policy = "fixture-local-strict-v1"
    matcher_settings = {"scale": False, "ltol": 0.02}

    def __init__(self, outcome: ExternalNoveltyOutcome) -> None:
        self.outcome = outcome
        self.provider_id = outcome.provider_id
        self.database_version_or_release = outcome.database_version_or_release

    def lookup(self, _candidate: Candidate) -> ExternalNoveltyOutcome:
        return self.outcome


def test_external_provider_aggregation_is_fail_closed_and_keeps_each_result() -> None:
    no_match = _StaticLookup(_external_outcome("provider-a", NoveltyStatus.NO_MATCH))
    unknown = _StaticLookup(_external_outcome("provider-b", NoveltyStatus.UNKNOWN))
    candidate = _molecule("candidate-multi-provider")

    unresolved = StagedNoveltyAssessor([no_match, unknown]).assess(
        [candidate], project_history=ProjectNoveltyIndex()
    )[0].external_database
    assert unresolved.status == NoveltyStatus.UNKNOWN
    assert [item.provider_id for item in unresolved.provider_results] == [
        "provider-a",
        "provider-b",
    ]
    assert unresolved.query_count == 2
    assert unresolved.provider_id == "multi-provider-aggregate"

    all_clear = StagedNoveltyAssessor(
        [
            no_match,
            _StaticLookup(_external_outcome("provider-b", NoveltyStatus.NO_MATCH)),
        ]
    ).assess([candidate], project_history=ProjectNoveltyIndex())[0].external_database
    assert all_clear.status == NoveltyStatus.NO_MATCH

    matched = StagedNoveltyAssessor(
        [
            unknown,
            _StaticLookup(_external_outcome("provider-a", NoveltyStatus.MATCH)),
        ]
    ).assess([candidate], project_history=ProjectNoveltyIndex())[0].external_database
    assert matched.status == NoveltyStatus.MATCH
    assert matched.structure_match_count == 1


def test_external_provider_exception_preserves_provider_id_and_never_becomes_no_match() -> None:
    class BrokenLookup:
        provider_id = "broken-structure-provider"

        def lookup(self, _candidate):
            raise TimeoutError("credential-bearing detail must not escape")

    stage = StagedNoveltyAssessor(BrokenLookup()).assess(
        [_molecule("candidate-provider-failure")],
        project_history=ProjectNoveltyIndex(),
    )[0].external_database

    assert stage.status == NoveltyStatus.UNKNOWN
    assert stage.provider_id == "broken-structure-provider"
    assert stage.provider_results[0].provider_id == "broken-structure-provider"
    assert stage.provider_results[0].reason == "external_lookup_failed:TimeoutError"
    assert "credential-bearing" not in stage.model_dump_json()


def test_unpinned_live_snapshot_no_match_requires_an_explicit_scope_warning() -> None:
    with pytest.raises(ValidationError, match="unpinned live-snapshot no-match"):
        ExternalNoveltyOutcome(
            provider_id="fixture-live-provider",
            client_version="fixture-client-1",
            database_version_or_release=LIVE_MOVING_SNAPSHOT_UNPINNED,
            retrieved_at=datetime.now(timezone.utc),
            query_sha256="c" * 64,
            matcher_policy="fixture-strict-v1",
            matcher_settings={"scale": False},
            status=NoveltyStatus.NO_MATCH,
            method="fixture-match-v1",
            query_count=1,
            structure_match_count=0,
        )


def test_materials_project_scaled_hit_is_similarity_not_hard_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        novelty_module,
        "classify_crystal_structure_relation",
        lambda *_args, **_kwargs: _fixture_crystal_assessment(
            CrystalMatchRelation.SCALED_SAME_PROTOTYPE
        ),
    )
    scaled_cif = CIF.replace("4.0", "4.4")

    class Client:
        def get_database_version(self):
            return "fixture-mp-release-2026-07"

        def find_structure(self, *_args, **_kwargs):
            return ["mp-scaled"]

        def get_material_ids(self, _formula):
            return ["mp-scaled"]

        def get_structure_by_material_id(self, material_id):
            assert material_id == "mp-scaled"
            return scaled_cif

        def close(self):
            return None

    outcome = MaterialsProjectStructureLookup(
        "runtime-secret",
        client_version="fixture-mp-api-1",
        rester_factory=lambda _key: Client(),
    ).lookup(_candidate("candidate-scaled-similarity"))

    assert outcome.status == NoveltyStatus.NO_MATCH
    assert outcome.matches == []
    assert outcome.structure_match_count == 0
    assert [item.record_id for item in outcome.similarity_findings] == ["mp-scaled"]
    assert outcome.similarity_findings[0].match_kind == "scaled_same_prototype"
    assert "rejected_by_local_strict_policy" in (outcome.reason or "")


def test_materials_project_unpinned_no_match_is_explicitly_qualified() -> None:
    class Client:
        def find_structure(self, *_args, **_kwargs):
            return []

        def get_material_ids(self, _formula):
            return []

        def close(self):
            return None

    outcome = MaterialsProjectStructureLookup(
        "runtime-secret",
        client_version="fixture-mp-api-1",
        rester_factory=lambda _key: Client(),
    ).lookup(_candidate("candidate-live-snapshot"))

    assert outcome.status == NoveltyStatus.NO_MATCH
    assert outcome.database_version_or_release == LIVE_MOVING_SNAPSHOT_UNPINNED
    assert LIVE_MOVING_SNAPSHOT_UNPINNED in (outcome.reason or "")
    assert "not_reproducible" in (outcome.reason or "")


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
    assert package.manifest.kpoint_sampling.mode == "explicit_grid"
    assert package.manifest.kpoint_sampling.realized_grid == (6, 6, 6)
    assert package.manifest.convergence_plan.status == "required_not_executed"
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
    assert "INSERT_CONVERGED_ECUTWFC_RY" in pw_text
    sampling = report.packages[0].manifest.kpoint_sampling
    assert sampling.mode == "reciprocal_spacing"
    assert sampling.target_spacing_A_inv == pytest.approx(0.30)
    assert sampling.realized_grid == (6, 6, 6)


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
            kpoint_sampling=KPointSamplingPlan(
                mode="explicit_grid",
                realized_grid=(4, 4, 4),
                reciprocal_vector_lengths_A_inv=(1.0, 1.0, 1.0),
            ),
            convergence_plan=DFTConvergencePlan(),
            input_artifacts=[
                artifact("dft/structure.cif"),
                artifact("dft/pw.in"),
                artifact("dft/Li.UPF"),
            ],
            required_external_configuration=["fixture external configuration"],
        )


def test_executed_dft_result_schema_rejects_unproven_completion() -> None:
    candidate = _candidate("candidate-result")
    output_artifact = ContentArtifactRef(
        artifact_id="artifact-output",
        relative_path="dft/output.json",
        sha256="a" * 64,
        media_type="application/json",
        byte_size=1,
    )
    convergence_artifact = ContentArtifactRef(
        artifact_id="artifact-convergence",
        relative_path="dft/convergence.json",
        sha256="b" * 64,
        media_type="application/json",
        byte_size=1,
    )

    with pytest.raises(ValidationError, match="input-manifest hash"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="static_energy",
            status="completed",
            output_artifacts=[output_artifact],
            convergence_evidence_artifacts=[convergence_artifact],
            method_policy_hash="c" * 64,
            converged=True,
            energy_per_atom_eV=-3.0,
        )

    with pytest.raises(ValidationError, match="require immutable output artifacts"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="static_energy",
            status="completed",
            input_manifest_sha256="d" * 64,
            convergence_evidence_artifacts=[convergence_artifact],
            method_policy_hash="c" * 64,
            converged=True,
            energy_per_atom_eV=-3.0,
        )

    with pytest.raises(ValidationError, match="convergence-evidence artifacts"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="static_energy",
            status="completed",
            input_manifest_sha256="d" * 64,
            output_artifacts=[output_artifact],
            method_policy_hash="c" * 64,
            converged=True,
            energy_per_atom_eV=-3.0,
        )

    with pytest.raises(ValidationError, match="failed DFT results must not expose"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="static_energy",
            status="failed",
            energy_per_atom_eV=-3.0,
        )

    for completed_evidence in (
        {"output_artifacts": [output_artifact]},
        {"convergence_evidence_artifacts": [convergence_artifact]},
        {"converged": False},
        {"reference_set_hash": "e" * 64},
        {"phonon_q_mesh": (2, 2, 2)},
    ):
        with pytest.raises(ValidationError, match="must not expose completed evidence"):
            PeriodicDFTCalculationResult(
                candidate_ref=candidate.candidate_ref,
                calculation_stage="static_energy",
                status="failed",
                input_manifest_sha256="d" * 64,
                method_policy_hash="c" * 64,
                **completed_evidence,
            )

    failed = PeriodicDFTCalculationResult(
        candidate_ref=candidate.candidate_ref,
        calculation_stage="static_energy",
        status="failed",
        input_manifest_sha256="d" * 64,
        method_policy_hash="c" * 64,
        notes=["backend exited before a converged result was available"],
    )
    assert failed.input_manifest_sha256 == "d" * 64
    assert failed.method_policy_hash == "c" * 64

    with pytest.raises(ValidationError, match="reference-set hash"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="static_energy",
            status="completed",
            input_manifest_sha256="d" * 64,
            output_artifacts=[output_artifact],
            convergence_evidence_artifacts=[convergence_artifact],
            method_policy_hash="c" * 64,
            converged=True,
            energy_per_atom_eV=-3.0,
            formation_energy_eV_per_atom=-0.5,
        )

    with pytest.raises(ValidationError, match="reference-set hash"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="static_energy",
            status="completed",
            input_manifest_sha256="d" * 64,
            output_artifacts=[output_artifact],
            convergence_evidence_artifacts=[convergence_artifact],
            method_policy_hash="c" * 64,
            converged=True,
            energy_per_atom_eV=-3.0,
            energy_above_hull_eV_per_atom=0.01,
        )

    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="static_energy",
            status="completed",
            input_manifest_sha256="d" * 64,
            output_artifacts=[output_artifact],
            convergence_evidence_artifacts=[convergence_artifact],
            method_policy_hash="c" * 64,
            reference_set_hash="e" * 64,
            converged=True,
            energy_per_atom_eV=-3.0,
            formation_energy_eV_per_atom=-0.5,
            energy_above_hull_eV_per_atom=-0.01,
        )

    with pytest.raises(ValidationError, match="completed phonon results require"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="phonon",
            status="completed",
            input_manifest_sha256="d" * 64,
            output_artifacts=[output_artifact],
            convergence_evidence_artifacts=[convergence_artifact],
            method_policy_hash="c" * 64,
            converged=True,
            has_imaginary_modes=False,
        )


def test_executed_dft_result_schema_accepts_only_closed_static_and_phonon_receipts() -> None:
    candidate = _candidate("candidate-valid-result")
    output_artifact = ContentArtifactRef(
        artifact_id="artifact-valid-output",
        relative_path="dft/valid-output.json",
        sha256="c" * 64,
        media_type="application/json",
        byte_size=2,
    )
    convergence_artifact = ContentArtifactRef(
        artifact_id="artifact-valid-convergence",
        relative_path="dft/valid-convergence.json",
        sha256="f" * 64,
        media_type="application/json",
        byte_size=2,
    )

    static = PeriodicDFTCalculationResult(
        candidate_ref=candidate.candidate_ref,
        calculation_stage="static_energy",
        status="completed",
        input_manifest_sha256="a" * 64,
        output_artifacts=[output_artifact],
        convergence_evidence_artifacts=[convergence_artifact],
        method_policy_hash="d" * 64,
        reference_set_hash="e" * 64,
        converged=True,
        total_energy_eV=-12.0,
        energy_per_atom_eV=-3.0,
        formation_energy_eV_per_atom=-0.5,
        energy_above_hull_eV_per_atom=0.01,
    )
    phonon = PeriodicDFTCalculationResult(
        candidate_ref=candidate.candidate_ref,
        calculation_stage="phonon",
        status="completed",
        input_manifest_sha256="a" * 64,
        output_artifacts=[output_artifact],
        convergence_evidence_artifacts=[convergence_artifact],
        method_policy_hash="d" * 64,
        converged=True,
        phonon_q_mesh=(4, 4, 4),
        minimum_frequency_THz=0.2,
        imaginary_mode_tolerance_THz=0.1,
        has_imaginary_modes=False,
    )

    assert static.input_manifest_sha256 == "a" * 64
    assert static.convergence_evidence_artifacts == [convergence_artifact]
    assert static.energy_above_hull_eV_per_atom == pytest.approx(0.01)
    assert static.reference_set_hash == "e" * 64
    assert phonon.phonon_q_mesh == (4, 4, 4)
    assert phonon.has_imaginary_modes is False


@pytest.mark.parametrize(
    ("minimum_frequency", "tolerance", "classification"),
    [
        (-0.2, 0.1, False),
        (0.2, 0.1, True),
    ],
)
def test_phonon_result_rejects_mode_classification_inconsistent_with_tolerance(
    minimum_frequency: float,
    tolerance: float,
    classification: bool,
) -> None:
    candidate = _candidate("candidate-phonon-classification")
    output_artifact = ContentArtifactRef(
        artifact_id="artifact-phonon-output",
        relative_path="dft/phonon-output.json",
        sha256="a" * 64,
        media_type="application/json",
        byte_size=2,
    )
    convergence_artifact = ContentArtifactRef(
        artifact_id="artifact-phonon-convergence",
        relative_path="dft/phonon-convergence.json",
        sha256="b" * 64,
        media_type="application/json",
        byte_size=2,
    )

    with pytest.raises(ValidationError, match="phonon mode classification must equal"):
        PeriodicDFTCalculationResult(
            candidate_ref=candidate.candidate_ref,
            calculation_stage="phonon",
            status="completed",
            input_manifest_sha256="c" * 64,
            output_artifacts=[output_artifact],
            convergence_evidence_artifacts=[convergence_artifact],
            method_policy_hash="d" * 64,
            converged=True,
            phonon_q_mesh=(4, 4, 4),
            minimum_frequency_THz=minimum_frequency,
            imaginary_mode_tolerance_THz=tolerance,
            has_imaginary_modes=classification,
        )


def test_phonon_result_treats_negative_tolerance_boundary_as_non_imaginary() -> None:
    candidate = _candidate("candidate-phonon-boundary")
    output_artifact = ContentArtifactRef(
        artifact_id="artifact-boundary-output",
        relative_path="dft/boundary-output.json",
        sha256="a" * 64,
        media_type="application/json",
        byte_size=2,
    )
    convergence_artifact = ContentArtifactRef(
        artifact_id="artifact-boundary-convergence",
        relative_path="dft/boundary-convergence.json",
        sha256="b" * 64,
        media_type="application/json",
        byte_size=2,
    )

    result = PeriodicDFTCalculationResult(
        candidate_ref=candidate.candidate_ref,
        calculation_stage="phonon",
        status="completed",
        input_manifest_sha256="c" * 64,
        output_artifacts=[output_artifact],
        convergence_evidence_artifacts=[convergence_artifact],
        method_policy_hash="d" * 64,
        converged=True,
        phonon_q_mesh=(4, 4, 4),
        minimum_frequency_THz=-0.1,
        imaginary_mode_tolerance_THz=0.1,
        has_imaginary_modes=False,
    )

    assert result.has_imaginary_modes is False
