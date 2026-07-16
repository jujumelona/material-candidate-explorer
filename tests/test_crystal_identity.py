from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from discovery_os import crystal_identity
from discovery_os.crystal_identity import (
    InvalidCrystalGeometryError,
    PymatgenRequiredError,
    canonical_structure_hash,
    canonicalize_crystal_structure,
    crystal_structure_fingerprint,
    exact_file_hash,
    group_crystal_structures,
    validate_crystal_geometry,
)
from discovery_os.hashing import canonical_structure_hash as hashing_structure_hash
from discovery_os.hashing import crystal_fingerprint


def test_exact_file_hash_uses_unmodified_bytes(tmp_path: Path) -> None:
    raw = b"data_example\r\n_cell_length_a 4.0\r\n"
    path = tmp_path / "example.cif"
    path.write_bytes(raw)

    assert exact_file_hash(path) == hashlib.sha256(raw).hexdigest()
    assert exact_file_hash(raw) == hashlib.sha256(raw).hexdigest()
    assert exact_file_hash(raw.decode("utf-8")) == hashlib.sha256(raw).hexdigest()
    assert exact_file_hash(raw) != exact_file_hash(raw.replace(b"\r\n", b"\n"))


def test_missing_pymatgen_fails_with_an_actionable_optional_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str):
        raise ModuleNotFoundError("fixture")

    monkeypatch.setattr(crystal_identity.importlib, "import_module", missing)

    with pytest.raises(PymatgenRequiredError, match="optional dependency 'pymatgen'"):
        canonical_structure_hash("data_fixture", fmt="cif")


def _rocksalt_structures():
    core = pytest.importorskip("pymatgen.core")
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
    translated = base.copy()
    translated.translate_sites(range(len(translated)), [0.25, 0.125, 0.375], frac_coords=True)
    supercell = base.copy()
    supercell.make_supercell([2, 1, 1])
    return base, reordered, translated, supercell


def test_canonical_identity_ignores_atom_order_origin_and_supercell() -> None:
    base, reordered, translated, supercell = _rocksalt_structures()

    hashes = {
        canonical_structure_hash(item)
        for item in (base, reordered, translated, supercell)
    }
    grouped = group_crystal_structures((base, reordered, translated, supercell))

    assert len(hashes) == 1
    assert grouped.unique_indices == (0,)
    assert grouped.groups[0].member_indices == (0, 1, 2, 3)


def test_different_cif_text_for_same_structure_has_one_scientific_identity() -> None:
    base, reordered, _translated, _supercell = _rocksalt_structures()
    cif_module = pytest.importorskip("pymatgen.io.cif")
    first = str(cif_module.CifWriter(base))
    second = str(cif_module.CifWriter(reordered)).replace("data_NaCl", "data_reordered")

    assert exact_file_hash(first) != exact_file_hash(second)
    assert canonical_structure_hash(first, fmt="cif") == canonical_structure_hash(
        second,
        fmt="cif",
    )
    assert crystal_structure_fingerprint(first, fmt="cif") == crystal_fingerprint(
        second,
        fmt="cif",
    )
    assert hashing_structure_hash(first, fmt="cif") == canonical_structure_hash(
        first,
        fmt="cif",
    )


def test_geometry_validation_rejects_collapsed_periodic_contacts() -> None:
    core = pytest.importorskip("pymatgen.core")
    collapsed = core.Structure(
        core.Lattice.cubic(4.0),
        ["Li", "Li"],
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]],
    )

    report = validate_crystal_geometry(collapsed, raise_on_error=False)

    assert report.is_valid is False
    assert report.minimum_distance_angstrom == pytest.approx(0.04)
    with pytest.raises(InvalidCrystalGeometryError, match="minimum periodic atom distance"):
        canonicalize_crystal_structure(collapsed)
