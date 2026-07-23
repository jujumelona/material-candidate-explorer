from __future__ import annotations

import pytest

from discovery_os.crystal_identity import (
    CrystalMatchRelation,
    classify_crystal_structure_relation,
    group_crystal_structures,
)


def test_empty_grouping_still_exposes_strict_matcher_provenance() -> None:
    grouped = group_crystal_structures(())

    assert grouped.groups == ()
    assert grouped.matcher_settings.scale is False
    assert grouped.matcher_settings.primitive_cell is True
    assert grouped.matcher_settings.attempt_supercell is True
    assert grouped.matcher_settings.allow_subset is False
    assert grouped.matcher_settings.max_relative_volume_difference == pytest.approx(0.03)


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
    translated.translate_sites(
        range(len(translated)),
        [0.25, 0.125, 0.375],
        frac_coords=True,
    )
    supercell = base.copy()
    supercell.make_supercell([2, 1, 1])
    return base, reordered, translated, supercell


def test_strict_relation_preserves_reordered_origin_and_supercell_equivalence() -> None:
    base, reordered, translated, supercell = _rocksalt_structures()

    for equivalent in (reordered, translated, supercell):
        assessment = classify_crystal_structure_relation(base, equivalent)
        assert assessment.relation == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE
        assert assessment.hard_deduplication_allowed is True
        assert assessment.strict_settings.scale is False
        assert assessment.strict_settings.attempt_supercell is True
        assert assessment.relative_volume_difference == pytest.approx(0.0)

    grouped = group_crystal_structures((base, reordered, translated, supercell))
    assert grouped.unique_indices == (0,)
    assert grouped.groups[0].member_indices == (0, 1, 2, 3)
    assert grouped.matcher_settings.scale is False
    assert grouped.matcher_settings.attempt_supercell is True
    assert grouped.ambiguous_comparisons == ()


def test_isotropic_expansion_is_scaled_prototype_not_hard_duplicate() -> None:
    core = pytest.importorskip("pymatgen.core")
    base, _reordered, _translated, _supercell = _rocksalt_structures()
    expanded = core.Structure(
        core.Lattice.cubic(5.64 * 1.10),
        ["Na", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    assessment = classify_crystal_structure_relation(base, expanded)

    assert assessment.relation == CrystalMatchRelation.SCALED_SAME_PROTOTYPE
    assert assessment.hard_deduplication_allowed is False
    assert assessment.strict_match is False
    assert assessment.scaled_match is True
    assert assessment.strict_settings.scale is False
    assert assessment.scaled_settings.scale is True
    assert assessment.relative_volume_difference > 0.25

    # Even caller-supplied legacy matcher tolerances cannot bypass the strict
    # relative-volume guard used for destructive grouping.
    grouped = group_crystal_structures(
        (base, expanded),
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
    )
    assert grouped.unique_indices == (0, 1)
    assert grouped.duplicate_count == 0
    assert grouped.matcher_settings.scale is False


def test_different_materials_are_distinct_and_preserved() -> None:
    core = pytest.importorskip("pymatgen.core")
    base, _reordered, _translated, _supercell = _rocksalt_structures()
    different = core.Structure(
        core.Lattice.cubic(5.64),
        ["K", "Cl"],
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )

    assessment = classify_crystal_structure_relation(base, different)
    grouped = group_crystal_structures((base, different))

    assert assessment.relation == CrystalMatchRelation.DISTINCT
    assert assessment.hard_deduplication_allowed is False
    assert grouped.unique_indices == (0, 1)
