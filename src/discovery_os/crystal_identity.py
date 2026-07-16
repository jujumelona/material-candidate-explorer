"""Crystallographic identity, canonicalization, and geometry validation.

``candidate_content_hash`` deliberately hashes the complete candidate record.  The
helpers in this module answer a different question: whether two periodic crystal
representations describe the same structure within explicit crystallographic
tolerances.  pymatgen is loaded lazily so the model-neutral coordinator keeps its
small core dependency set.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_FINGERPRINT_SCHEMA = "crystal-structure-fingerprint-v1"


class CrystalIdentityError(ValueError):
    """A crystal could not be parsed, standardized, or compared safely."""


class PymatgenRequiredError(CrystalIdentityError):
    """The optional crystallographic dependency is unavailable."""


class InvalidCrystalGeometryError(CrystalIdentityError):
    """A parsed crystal fails a hard geometry-safety constraint."""


@dataclass(frozen=True, slots=True)
class CrystalGeometryReport:
    """Bounded geometry checks performed before identity or model evaluation."""

    atom_count: int
    volume_angstrom3: float
    volume_per_atom_angstrom3: float
    minimum_distance_angstrom: float | None
    minimum_distance_threshold_angstrom: float
    is_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CanonicalCrystalStructure:
    """A deterministic primitive representation plus preserved standard-cell data."""

    canonical_structure: Any
    primitive_structure: Any
    conventional_structure: Any
    canonical_cif: str
    fingerprint: dict[str, Any]
    structure_hash: str
    geometry: CrystalGeometryReport
    source_atom_count: int
    primitive_atom_count: int
    conventional_atom_count: int
    space_group_symbol: str | None
    space_group_number: int | None


@dataclass(frozen=True, slots=True)
class CrystalStructureGroup:
    """Input indexes judged equivalent by pymatgen ``StructureMatcher``."""

    representative_index: int
    member_indices: tuple[int, ...]
    representative_hash: str


@dataclass(frozen=True, slots=True)
class CrystalGroupingResult:
    """Canonicalized structures and their stable, input-order duplicate groups."""

    canonical_structures: tuple[CanonicalCrystalStructure, ...]
    groups: tuple[CrystalStructureGroup, ...]

    @property
    def unique_indices(self) -> tuple[int, ...]:
        return tuple(group.representative_index for group in self.groups)

    @property
    def duplicate_count(self) -> int:
        return len(self.canonical_structures) - len(self.groups)


def exact_file_hash(value: str | bytes | bytearray | memoryview | Path) -> str:
    """Return SHA-256 of exact bytes, without crystallographic normalization.

    ``Path`` values are streamed from disk.  ``str`` values are treated as file
    contents and encoded as UTF-8, which avoids ambiguous path-vs-content behavior.
    """

    digest = hashlib.sha256()
    if isinstance(value, Path):
        with value.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if isinstance(value, str):
        digest.update(value.encode("utf-8"))
        return digest.hexdigest()
    if isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(bytes(value))
        return digest.hexdigest()
    raise TypeError("exact_file_hash expects file contents or a pathlib.Path")


def parse_crystal_structure(
    value: Any,
    *,
    fmt: str | None = None,
    max_atoms: int = 20_000,
) -> Any:
    """Parse a pymatgen Structure, CIF/POSCAR text, bytes, or a file path."""

    modules = _pymatgen_modules()
    structure_type = modules["core"].Structure
    if isinstance(value, structure_type):
        structure = value.copy()
    else:
        inferred_format = fmt
        if isinstance(value, Path):
            if not value.is_file():
                raise CrystalIdentityError(f"crystal file does not exist: {value}")
            raw = value.read_text(encoding="utf-8")
            if inferred_format is None:
                inferred_format = _format_from_suffix(value.suffix)
        elif isinstance(value, bytes):
            try:
                raw = value.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise CrystalIdentityError("crystal bytes must be UTF-8 text") from exc
        elif isinstance(value, str):
            raw = value
        else:
            raise TypeError(
                "crystal input must be a pymatgen Structure, CIF/POSCAR text, bytes, or Path"
            )
        if not raw.strip():
            raise CrystalIdentityError("crystal representation is empty")
        normalized_format = _normalize_format(inferred_format or _infer_text_format(raw))
        try:
            structure = structure_type.from_str(raw, fmt=normalized_format)
        except Exception as exc:
            raise CrystalIdentityError(
                f"pymatgen could not parse {normalized_format}: {type(exc).__name__}: {exc}"
            ) from exc
    if not 1 <= len(structure) <= max_atoms:
        raise CrystalIdentityError(f"crystal atom count must be between 1 and {max_atoms}")
    return structure


def validate_crystal_geometry(
    value: Any,
    *,
    fmt: str | None = None,
    max_atoms: int = 20_000,
    minimum_distance_angstrom: float = 0.5,
    minimum_volume_per_atom_angstrom3: float = 0.1,
    maximum_volume_per_atom_angstrom3: float = 1_000_000.0,
    raise_on_error: bool = True,
) -> CrystalGeometryReport:
    """Reject non-finite, collapsed, over-occupied, or implausible cells.

    The minimum periodic distance is exact for structures up to 2,048 sites.  For
    larger inputs a bounded neighbor search still detects every distance below the
    configured safety threshold; ``minimum_distance_angstrom`` is then ``None``
    when no unsafe contact is present.
    """

    if not math.isfinite(minimum_distance_angstrom) or minimum_distance_angstrom <= 0:
        raise ValueError("minimum_distance_angstrom must be finite and positive")
    if (
        not math.isfinite(minimum_volume_per_atom_angstrom3)
        or minimum_volume_per_atom_angstrom3 <= 0
    ):
        raise ValueError("minimum_volume_per_atom_angstrom3 must be finite and positive")
    if (
        not math.isfinite(maximum_volume_per_atom_angstrom3)
        or maximum_volume_per_atom_angstrom3 <= minimum_volume_per_atom_angstrom3
    ):
        raise ValueError(
            "maximum_volume_per_atom_angstrom3 must exceed the positive minimum"
        )
    structure = parse_crystal_structure(value, fmt=fmt, max_atoms=max_atoms)
    errors: list[str] = []
    warnings: list[str] = []
    atom_count = len(structure)
    matrix = [[float(item) for item in row] for row in structure.lattice.matrix]
    fractional = [[float(item) for item in row] for row in structure.frac_coords]
    if any(not math.isfinite(item) for row in matrix for item in row):
        errors.append("lattice contains a non-finite value")
    if any(not math.isfinite(item) for row in fractional for item in row):
        errors.append("fractional coordinates contain a non-finite value")
    volume = float(structure.volume)
    if not math.isfinite(volume) or volume <= 0:
        errors.append("cell volume must be finite and positive")
        volume_per_atom = math.nan
    else:
        volume_per_atom = volume / atom_count
        if volume_per_atom < minimum_volume_per_atom_angstrom3:
            errors.append(
                "cell volume per atom is below the configured collapse threshold "
                f"({volume_per_atom:.8g} < {minimum_volume_per_atom_angstrom3:.8g} A^3)"
            )
        if volume_per_atom > maximum_volume_per_atom_angstrom3:
            errors.append(
                "cell volume per atom exceeds the configured sparse-cell threshold "
                f"({volume_per_atom:.8g} > {maximum_volume_per_atom_angstrom3:.8g} A^3)"
            )
    for index, site in enumerate(structure):
        occupancies = [float(amount) for amount in site.species.values()]
        if not occupancies or any(not math.isfinite(item) or item <= 0 for item in occupancies):
            errors.append(f"site {index} has a non-positive or non-finite occupancy")
            continue
        total = sum(occupancies)
        if total > 1.0 + 1e-8:
            errors.append(f"site {index} occupancy exceeds one ({total:.8g})")
        elif total < 1.0 - 1e-8:
            warnings.append(f"site {index} is partially occupied ({total:.8g})")

    minimum_distance = _minimum_periodic_distance(
        structure,
        safety_threshold=minimum_distance_angstrom,
    )
    if minimum_distance is not None and minimum_distance < minimum_distance_angstrom:
        errors.append(
            "minimum periodic atom distance is below the configured safety threshold "
            f"({minimum_distance:.8g} < {minimum_distance_angstrom:.8g} A)"
        )
    report = CrystalGeometryReport(
        atom_count=atom_count,
        volume_angstrom3=volume,
        volume_per_atom_angstrom3=volume_per_atom,
        minimum_distance_angstrom=minimum_distance,
        minimum_distance_threshold_angstrom=minimum_distance_angstrom,
        is_valid=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
    if errors and raise_on_error:
        raise InvalidCrystalGeometryError("; ".join(errors))
    return report


def canonicalize_crystal_structure(
    value: Any,
    *,
    fmt: str | None = None,
    max_atoms: int = 20_000,
    minimum_distance_angstrom: float = 0.5,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0,
    coordinate_decimals: int = 10,
    max_cif_bytes: int = 4 * 1024 * 1024,
) -> CanonicalCrystalStructure:
    """Return a deterministic Niggli-reduced primitive cell and standard-cell context."""

    if not math.isfinite(symprec) or symprec <= 0:
        raise ValueError("symprec must be finite and positive")
    if not math.isfinite(angle_tolerance) or angle_tolerance <= 0:
        raise ValueError("angle_tolerance must be finite and positive")
    if not 4 <= coordinate_decimals <= 14:
        raise ValueError("coordinate_decimals must be between 4 and 14")
    source = parse_crystal_structure(value, fmt=fmt, max_atoms=max_atoms)
    geometry = validate_crystal_geometry(
        source,
        max_atoms=max_atoms,
        minimum_distance_angstrom=minimum_distance_angstrom,
    )
    modules = _pymatgen_modules()
    analyzer_type = modules["analyzer"].SpacegroupAnalyzer
    space_group_symbol: str | None = None
    space_group_number: int | None = None
    try:
        analyzer = analyzer_type(
            source,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )
        primitive = analyzer.get_primitive_standard_structure(
            international_monoclinic=True,
            keep_site_properties=False,
        )
        conventional = analyzer.get_conventional_standard_structure(
            international_monoclinic=True,
            keep_site_properties=False,
        )
        space_group_symbol = str(analyzer.get_space_group_symbol())
        space_group_number = int(analyzer.get_space_group_number())
    except Exception:
        try:
            primitive = source.get_primitive_structure(
                tolerance=symprec,
                use_site_props=False,
            )
        except Exception:
            primitive = source.copy()
        conventional = source.copy()
    if not 1 <= len(primitive) <= max_atoms:
        raise CrystalIdentityError("primitive-cell standardization produced an invalid atom count")
    try:
        reduced = primitive.get_reduced_structure(reduction_algo="niggli")
    except Exception as exc:
        raise CrystalIdentityError(
            f"pymatgen Niggli reduction failed: {type(exc).__name__}: {exc}"
        ) from exc
    canonical = _canonical_site_order(reduced, decimals=coordinate_decimals)
    fingerprint = _fingerprint_payload(
        canonical,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        coordinate_decimals=coordinate_decimals,
    )
    structure_hash = _payload_hash(fingerprint)
    try:
        writer = modules["cif"].CifWriter(
            canonical,
            symprec=None,
            significant_figures=coordinate_decimals,
        )
        canonical_cif = str(writer)
    except Exception as exc:
        raise CrystalIdentityError(
            f"pymatgen could not serialize canonical CIF: {type(exc).__name__}: {exc}"
        ) from exc
    if not canonical_cif or len(canonical_cif.encode("utf-8")) > max_cif_bytes:
        raise CrystalIdentityError("canonical CIF is empty or exceeds the representation limit")
    return CanonicalCrystalStructure(
        canonical_structure=canonical,
        primitive_structure=primitive,
        conventional_structure=conventional,
        canonical_cif=canonical_cif,
        fingerprint=fingerprint,
        structure_hash=structure_hash,
        geometry=geometry,
        source_atom_count=len(source),
        primitive_atom_count=len(primitive),
        conventional_atom_count=len(conventional),
        space_group_symbol=space_group_symbol,
        space_group_number=space_group_number,
    )


def crystal_structure_fingerprint(value: Any, **kwargs: Any) -> dict[str, Any]:
    """Return the JSON-compatible canonical primitive-cell fingerprint."""

    return canonicalize_crystal_structure(value, **kwargs).fingerprint


def canonical_structure_hash(value: Any, **kwargs: Any) -> str:
    """Return SHA-256 of the canonical primitive-cell fingerprint."""

    return canonicalize_crystal_structure(value, **kwargs).structure_hash


def group_crystal_structures(
    structures: list[Any] | tuple[Any, ...],
    *,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    canonicalization_kwargs: dict[str, Any] | None = None,
) -> CrystalGroupingResult:
    """Group equivalent structures with ``StructureMatcher`` in stable input order.

    Canonical hashes are useful cache keys, but grouping intentionally does not
    rely on hash equality: ``StructureMatcher`` is the final scientific duplicate
    decision and accepts primitive/supercell-equivalent representations.
    """

    if not structures:
        return CrystalGroupingResult(canonical_structures=(), groups=())
    kwargs = dict(canonicalization_kwargs or {})
    canonical = tuple(
        item if isinstance(item, CanonicalCrystalStructure) else canonicalize_crystal_structure(item, **kwargs)
        for item in structures
    )
    matcher_type = _pymatgen_modules()["matcher"].StructureMatcher
    matcher = matcher_type(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=True,
        scale=True,
        attempt_supercell=True,
        allow_subset=False,
    )
    members: list[list[int]] = []
    for index, current in enumerate(canonical):
        matched = False
        for group in members:
            representative = canonical[group[0]]
            try:
                equivalent = bool(
                    matcher.fit(
                        representative.canonical_structure,
                        current.canonical_structure,
                        symmetric=True,
                    )
                )
            except TypeError:
                # Compatibility with older pymatgen releases lacking ``symmetric``.
                equivalent = bool(
                    matcher.fit(
                        representative.canonical_structure,
                        current.canonical_structure,
                    )
                )
            except Exception as exc:
                raise CrystalIdentityError(
                    f"StructureMatcher comparison failed: {type(exc).__name__}: {exc}"
                ) from exc
            if equivalent:
                group.append(index)
                matched = True
                break
        if not matched:
            members.append([index])
    groups = tuple(
        CrystalStructureGroup(
            representative_index=group[0],
            member_indices=tuple(group),
            representative_hash=canonical[group[0]].structure_hash,
        )
        for group in members
    )
    return CrystalGroupingResult(canonical_structures=canonical, groups=groups)


def deduplicate_crystal_structures(
    structures: list[Any] | tuple[Any, ...],
    **kwargs: Any,
) -> tuple[CanonicalCrystalStructure, ...]:
    """Return first-occurrence representatives from StructureMatcher groups."""

    result = group_crystal_structures(structures, **kwargs)
    return tuple(result.canonical_structures[index] for index in result.unique_indices)


def _pymatgen_modules() -> dict[str, Any]:
    try:
        modules = {
            "core": importlib.import_module("pymatgen.core"),
            "analyzer": importlib.import_module("pymatgen.symmetry.analyzer"),
            "cif": importlib.import_module("pymatgen.io.cif"),
        }
        try:
            # MatterGen 1.0.x and long-lived pymatgen releases use this path.
            matcher = importlib.import_module("pymatgen.analysis.structure_matcher")
        except (ImportError, ModuleNotFoundError):
            # pymatgen-core 2026+ moved StructureMatcher into the core namespace.
            matcher = importlib.import_module("pymatgen.core.structure_matcher")
        modules["matcher"] = matcher
    except (ImportError, ModuleNotFoundError) as exc:
        raise PymatgenRequiredError(
            "crystal identity requires optional dependency 'pymatgen'; install it in the "
            "crystal/MatterGen environment before parsing or comparing structures"
        ) from exc
    return modules


def _format_from_suffix(suffix: str) -> str:
    normalized = suffix.lower()
    if normalized == ".cif":
        return "cif"
    if normalized in {".vasp", ".poscar", ".contcar"}:
        return "poscar"
    raise CrystalIdentityError(
        f"cannot infer crystal format from suffix {suffix!r}; pass fmt='cif' or fmt='poscar'"
    )


def _infer_text_format(raw: str) -> str:
    lowered = raw.lower()
    if "_cell_length_a" in lowered or lowered.lstrip().startswith("data_"):
        return "cif"
    raise CrystalIdentityError("cannot infer crystal text format; pass fmt='cif' or fmt='poscar'")


def _normalize_format(fmt: str) -> str:
    normalized = fmt.strip().lower()
    if normalized == "cif":
        return "cif"
    if normalized in {"poscar", "vasp", "contcar"}:
        return "poscar"
    raise CrystalIdentityError("supported crystal formats are CIF and POSCAR")


def _minimum_periodic_distance(structure: Any, *, safety_threshold: float) -> float | None:
    atom_count = len(structure)
    if atom_count <= 2_048:
        values: list[float] = []
        distance_matrix = structure.distance_matrix
        for row_index in range(atom_count):
            for column_index in range(row_index + 1, atom_count):
                value = float(distance_matrix[row_index, column_index])
                if value > 1e-12 and math.isfinite(value):
                    values.append(value)
        try:
            lattice = structure.lattice.get_lll_reduced_lattice()
            for row in lattice.matrix:
                value = math.sqrt(sum(float(item) ** 2 for item in row))
                if value > 1e-12 and math.isfinite(value):
                    values.append(value)
        except Exception:
            pass
        return min(values) if values else None
    try:
        _center, _neighbor, _images, distances = structure.get_neighbor_list(
            r=safety_threshold,
            numerical_tol=1e-12,
            exclude_self=True,
        )
    except TypeError:
        _center, _neighbor, _images, distances = structure.get_neighbor_list(
            r=safety_threshold,
            numerical_tol=1e-12,
        )
    values = [
        float(item)
        for item in distances
        if math.isfinite(float(item)) and float(item) > 1e-12
    ]
    return min(values) if values else None


def _canonical_site_order(structure: Any, *, decimals: int) -> Any:
    structure_type = _pymatgen_modules()["core"].Structure
    sites: list[tuple[str, Any, tuple[float, float, float]]] = []
    for site in structure:
        species_key = json.dumps(
            _species_payload(site),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        coordinate = tuple(_wrap_fractional(float(item), decimals) for item in site.frac_coords)
        sites.append((species_key, site.species, coordinate))
    alternatives: list[tuple[tuple[tuple[str, tuple[float, float, float]], ...], list[Any]]] = []
    for _origin_key, _origin_species, origin in sites:
        shifted: list[tuple[str, Any, tuple[float, float, float]]] = []
        for species_key, species, coordinate in sites:
            translated = tuple(
                _wrap_fractional(coordinate[axis] - origin[axis], decimals)
                for axis in range(3)
            )
            shifted.append((species_key, species, translated))
        shifted.sort(key=lambda item: (item[0], item[2]))
        key = tuple((item[0], item[2]) for item in shifted)
        alternatives.append((key, shifted))
    _, selected = min(alternatives, key=lambda item: item[0])
    try:
        return structure_type(
            structure.lattice,
            [item[1] for item in selected],
            [item[2] for item in selected],
            coords_are_cartesian=False,
            to_unit_cell=True,
            validate_proximity=False,
        )
    except Exception as exc:
        raise CrystalIdentityError(
            f"canonical site ordering failed: {type(exc).__name__}: {exc}"
        ) from exc


def _species_payload(site: Any) -> list[dict[str, float | str]]:
    rows = [
        {"species": str(species), "occupancy": round(float(amount), 12)}
        for species, amount in site.species.items()
    ]
    rows.sort(key=lambda item: str(item["species"]))
    return rows


def _wrap_fractional(value: float, decimals: int) -> float:
    wrapped = value % 1.0
    rounded = round(wrapped, decimals)
    tolerance = 10.0 ** (-decimals)
    if abs(rounded) <= tolerance or abs(rounded - 1.0) <= tolerance:
        return 0.0
    return rounded


def _fingerprint_payload(
    structure: Any,
    *,
    symprec: float,
    angle_tolerance: float,
    coordinate_decimals: int,
) -> dict[str, Any]:
    lattice = structure.lattice
    return {
        "schema": _FINGERPRINT_SCHEMA,
        "standardization": {
            "cell": "primitive_niggli",
            "symprec_angstrom": symprec,
            "angle_tolerance_degrees": angle_tolerance,
            "coordinate_decimals": coordinate_decimals,
        },
        "composition": {
            str(key): round(float(value), 12)
            for key, value in sorted(structure.composition.as_dict().items())
        },
        "lattice": {
            "lengths_angstrom": [round(float(item), coordinate_decimals) for item in lattice.abc],
            "angles_degrees": [
                round(float(item), coordinate_decimals) for item in lattice.angles
            ],
            "volume_angstrom3": round(float(lattice.volume), coordinate_decimals),
        },
        "sites": [
            {
                "species": _species_payload(site),
                "fractional_coordinates": [
                    _wrap_fractional(float(item), coordinate_decimals)
                    for item in site.frac_coords
                ],
            }
            for site in structure
        ],
    }


def _payload_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CanonicalCrystalStructure",
    "CrystalGeometryReport",
    "CrystalGroupingResult",
    "CrystalIdentityError",
    "CrystalStructureGroup",
    "InvalidCrystalGeometryError",
    "PymatgenRequiredError",
    "canonical_structure_hash",
    "canonicalize_crystal_structure",
    "crystal_structure_fingerprint",
    "deduplicate_crystal_structures",
    "exact_file_hash",
    "group_crystal_structures",
    "parse_crystal_structure",
    "validate_crystal_geometry",
]
