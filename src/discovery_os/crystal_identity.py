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
from enum import Enum
from pathlib import Path
from typing import Any


_FINGERPRINT_SCHEMA = "crystal-structure-fingerprint-v1"
_IDENTITY_FINGERPRINT_SCHEMA = "crystal-raw-identity-fingerprint-v2"
CRYSTAL_IDENTITY_CANONICALIZATION = "source-niggli-no-symmetry-v2"

_STRICT_MATCH_LATTICE_TOLERANCE = 0.02
_STRICT_MATCH_SITE_TOLERANCE = 0.05
_STRICT_MATCH_ANGLE_TOLERANCE = 1.0
_STRICT_MATCH_MAX_RELATIVE_VOLUME_DIFFERENCE = 0.03
_SCALED_MATCH_LATTICE_TOLERANCE = 0.2
_SCALED_MATCH_SITE_TOLERANCE = 0.3
_SCALED_MATCH_ANGLE_TOLERANCE = 5.0


class CrystalIdentityError(ValueError):
    """A crystal could not be parsed, standardized, or compared safely."""


class PymatgenRequiredError(CrystalIdentityError):
    """The optional crystallographic dependency is unavailable."""


class InvalidCrystalGeometryError(CrystalIdentityError):
    """A parsed crystal fails a hard geometry-safety constraint."""


class CrystalMatchRelation(str, Enum):
    """Scientific relation between two species-preserving periodic structures.

    Only ``STRICT_MATERIAL_DUPLICATE`` is eligible for hard deduplication.
    ``SCALED_SAME_PROTOTYPE`` deliberately preserves both candidates because
    matching required normalization to an equivalent volume.  ``AMBIGUOUS`` is
    fail-closed: a comparison problem or direction-dependent result must never
    silently remove a candidate.
    """

    STRICT_MATERIAL_DUPLICATE = "strict_material_duplicate"
    SCALED_SAME_PROTOTYPE = "scaled_same_prototype"
    AMBIGUOUS = "ambiguous"
    DISTINCT = "distinct"


@dataclass(frozen=True, slots=True)
class CrystalMatcherSettings:
    """Complete, reviewable ``StructureMatcher`` settings for one comparison."""

    ltol: float
    stol: float
    angle_tol: float
    primitive_cell: bool
    scale: bool
    attempt_supercell: bool
    allow_subset: bool
    max_relative_volume_difference: float | None = None


@dataclass(frozen=True, slots=True)
class CrystalMatchAssessment:
    """Typed result separating hard identity from scaled prototype similarity."""

    relation: CrystalMatchRelation
    strict_match: bool | None
    scaled_match: bool | None
    relative_volume_difference: float
    strict_settings: CrystalMatcherSettings
    scaled_settings: CrystalMatcherSettings
    reason: str | None = None

    @property
    def hard_deduplication_allowed(self) -> bool:
        """Whether one of the two candidate records may be removed scientifically."""

        return self.relation == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE


@dataclass(frozen=True, slots=True)
class CrystalAmbiguousComparison:
    """A fail-closed pair that could not be assigned a strict identity result."""

    left_index: int
    right_index: int
    reason: str


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
    """Separate symmetry-standardized context from deletion-safe identity.

    ``canonical_structure`` and ``structure_hash`` preserve the historical
    primitive standardized scientific/prototype fingerprint.  Hard deletion
    uses only ``identity_structure`` and ``identity_structure_hash``, which are
    Niggli-reduced directly from the parsed source without symmetry refinement.
    """

    canonical_structure: Any
    identity_structure: Any
    primitive_structure: Any
    conventional_structure: Any
    canonical_cif: str
    fingerprint: dict[str, Any]
    structure_hash: str
    identity_fingerprint: dict[str, Any]
    identity_structure_hash: str
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
    matcher_settings: CrystalMatcherSettings
    ambiguous_comparisons: tuple[CrystalAmbiguousComparison, ...] = ()

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
    """Return separate prototype context and non-symmetrized hard identity."""

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
        # Hard identity must retain the generated geometry. Applying
        # SpacegroupAnalyzer's symprec before this step could idealize a
        # genuine symmetry-breaking displacement and create a false duplicate.
        identity_reduced = source.get_reduced_structure(reduction_algo="niggli")
    except Exception as exc:
        raise CrystalIdentityError(
            f"pymatgen Niggli reduction failed: {type(exc).__name__}: {exc}"
        ) from exc
    canonical = _canonical_site_order(reduced, decimals=coordinate_decimals)
    identity = _canonical_site_order(
        identity_reduced,
        decimals=coordinate_decimals,
    )
    fingerprint = _fingerprint_payload(
        canonical,
        symprec=symprec,
        angle_tolerance=angle_tolerance,
        coordinate_decimals=coordinate_decimals,
    )
    structure_hash = _payload_hash(fingerprint)
    identity_fingerprint = _identity_fingerprint_payload(
        identity,
        coordinate_decimals=coordinate_decimals,
    )
    identity_structure_hash = _payload_hash(identity_fingerprint)
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
        identity_structure=identity,
        primitive_structure=primitive,
        conventional_structure=conventional,
        canonical_cif=canonical_cif,
        fingerprint=fingerprint,
        structure_hash=structure_hash,
        identity_fingerprint=identity_fingerprint,
        identity_structure_hash=identity_structure_hash,
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


def classify_crystal_structure_relation(
    first: Any,
    second: Any,
    *,
    strict_ltol: float = _STRICT_MATCH_LATTICE_TOLERANCE,
    strict_stol: float = _STRICT_MATCH_SITE_TOLERANCE,
    strict_angle_tol: float = _STRICT_MATCH_ANGLE_TOLERANCE,
    strict_max_relative_volume_difference: float = (
        _STRICT_MATCH_MAX_RELATIVE_VOLUME_DIFFERENCE
    ),
    scaled_ltol: float = _SCALED_MATCH_LATTICE_TOLERANCE,
    scaled_stol: float = _SCALED_MATCH_SITE_TOLERANCE,
    scaled_angle_tol: float = _SCALED_MATCH_ANGLE_TOLERANCE,
    canonicalization_kwargs: dict[str, Any] | None = None,
) -> CrystalMatchAssessment:
    """Classify a pair without confusing scaled similarity with a duplicate.

    Both matchers preserve species identity, reduce primitive cells, and permit
    equivalent supercell descriptions.  The strict pass does *not* normalize
    volumes and also applies a symmetric relative-volume guard.  Only when that
    pass fails is the scale-normalized matcher used to identify a related
    prototype.  Matcher failures and direction-dependent legacy results are
    returned as ``AMBIGUOUS`` so callers preserve both candidates.
    """

    strict_settings = _matcher_settings(
        ltol=strict_ltol,
        stol=strict_stol,
        angle_tol=strict_angle_tol,
        scale=False,
        max_relative_volume_difference=strict_max_relative_volume_difference,
    )
    scaled_settings = _matcher_settings(
        ltol=scaled_ltol,
        stol=scaled_stol,
        angle_tol=scaled_angle_tol,
        scale=True,
        max_relative_volume_difference=None,
    )
    kwargs = dict(canonicalization_kwargs or {})
    canonical = tuple(
        item
        if isinstance(item, CanonicalCrystalStructure)
        else canonicalize_crystal_structure(item, **kwargs)
        for item in (first, second)
    )
    relative_volume_difference = _relative_volume_difference(
        canonical[0].identity_structure,
        canonical[1].identity_structure,
    )
    strict_match, strict_reason = _fit_with_settings(
        canonical[0].identity_structure,
        canonical[1].identity_structure,
        strict_settings,
    )
    if strict_match is None:
        return CrystalMatchAssessment(
            relation=CrystalMatchRelation.AMBIGUOUS,
            strict_match=None,
            scaled_match=None,
            relative_volume_difference=relative_volume_difference,
            strict_settings=strict_settings,
            scaled_settings=scaled_settings,
            reason=strict_reason,
        )
    volume_within_strict_limit = (
        relative_volume_difference
        <= strict_max_relative_volume_difference
    )
    if strict_match and volume_within_strict_limit:
        return CrystalMatchAssessment(
            relation=CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE,
            strict_match=True,
            scaled_match=True,
            relative_volume_difference=relative_volume_difference,
            strict_settings=strict_settings,
            scaled_settings=scaled_settings,
        )

    scaled_match, scaled_reason = _fit_with_settings(
        canonical[0].identity_structure,
        canonical[1].identity_structure,
        scaled_settings,
    )
    if scaled_match is None:
        return CrystalMatchAssessment(
            relation=CrystalMatchRelation.AMBIGUOUS,
            strict_match=strict_match,
            scaled_match=None,
            relative_volume_difference=relative_volume_difference,
            strict_settings=strict_settings,
            scaled_settings=scaled_settings,
            reason=scaled_reason,
        )
    if strict_match and not volume_within_strict_limit and not scaled_match:
        return CrystalMatchAssessment(
            relation=CrystalMatchRelation.AMBIGUOUS,
            strict_match=True,
            scaled_match=False,
            relative_volume_difference=relative_volume_difference,
            strict_settings=strict_settings,
            scaled_settings=scaled_settings,
            reason="strict_match_exceeded_volume_guard_but_scaled_match_failed",
        )
    return CrystalMatchAssessment(
        relation=(
            CrystalMatchRelation.SCALED_SAME_PROTOTYPE
            if scaled_match
            else CrystalMatchRelation.DISTINCT
        ),
        strict_match=(strict_match and volume_within_strict_limit),
        scaled_match=scaled_match,
        relative_volume_difference=relative_volume_difference,
        strict_settings=strict_settings,
        scaled_settings=scaled_settings,
        reason=(
            "strict_match_exceeded_relative_volume_guard"
            if strict_match and not volume_within_strict_limit
            else None
        ),
    )


def group_crystal_structures(
    structures: list[Any] | tuple[Any, ...],
    *,
    ltol: float = _STRICT_MATCH_LATTICE_TOLERANCE,
    stol: float = _STRICT_MATCH_SITE_TOLERANCE,
    angle_tol: float = _STRICT_MATCH_ANGLE_TOLERANCE,
    max_relative_volume_difference: float = (
        _STRICT_MATCH_MAX_RELATIVE_VOLUME_DIFFERENCE
    ),
    canonicalization_kwargs: dict[str, Any] | None = None,
) -> CrystalGroupingResult:
    """Hard-group strict material duplicates in stable input order.

    Canonical hashes are useful cache keys, but grouping intentionally does not
    rely on hash equality: ``StructureMatcher`` is the final scientific duplicate
    decision and accepts primitive/supercell-equivalent representations.  Hard
    grouping always uses ``scale=False`` and a relative-volume guard.  Use
    :func:`classify_crystal_structure_relation` to detect scale-normalized
    same-prototype candidates without deleting them.
    """

    settings = _matcher_settings(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        scale=False,
        max_relative_volume_difference=max_relative_volume_difference,
    )
    if not structures:
        return CrystalGroupingResult(
            canonical_structures=(),
            groups=(),
            matcher_settings=settings,
        )
    kwargs = dict(canonicalization_kwargs or {})
    canonical = tuple(
        item if isinstance(item, CanonicalCrystalStructure) else canonicalize_crystal_structure(item, **kwargs)
        for item in structures
    )
    members: list[list[int]] = []
    ambiguous: list[CrystalAmbiguousComparison] = []
    for index, current in enumerate(canonical):
        matched = False
        for group in members:
            representative = canonical[group[0]]
            volume_difference = _relative_volume_difference(
                representative.identity_structure,
                current.identity_structure,
            )
            equivalent, reason = _fit_with_settings(
                representative.identity_structure,
                current.identity_structure,
                settings,
            )
            if equivalent is None:
                ambiguous.append(
                    CrystalAmbiguousComparison(
                        left_index=group[0],
                        right_index=index,
                        reason=reason or "strict_structure_match_ambiguous",
                    )
                )
                continue
            equivalent = bool(
                equivalent
                and volume_difference <= max_relative_volume_difference
            )
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
            representative_hash=canonical[group[0]].identity_structure_hash,
        )
        for group in members
    )
    return CrystalGroupingResult(
        canonical_structures=canonical,
        groups=groups,
        matcher_settings=settings,
        ambiguous_comparisons=tuple(ambiguous),
    )


def deduplicate_crystal_structures(
    structures: list[Any] | tuple[Any, ...],
    **kwargs: Any,
) -> tuple[CanonicalCrystalStructure, ...]:
    """Return first-occurrence representatives from StructureMatcher groups."""

    result = group_crystal_structures(structures, **kwargs)
    return tuple(result.canonical_structures[index] for index in result.unique_indices)


def _matcher_settings(
    *,
    ltol: float,
    stol: float,
    angle_tol: float,
    scale: bool,
    max_relative_volume_difference: float | None,
) -> CrystalMatcherSettings:
    values = (ltol, stol, angle_tol)
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("StructureMatcher tolerances must be finite and positive")
    if (
        max_relative_volume_difference is not None
        and (
            not math.isfinite(max_relative_volume_difference)
            or max_relative_volume_difference < 0
        )
    ):
        raise ValueError(
            "max_relative_volume_difference must be finite and non-negative"
        )
    return CrystalMatcherSettings(
        ltol=float(ltol),
        stol=float(stol),
        angle_tol=float(angle_tol),
        primitive_cell=True,
        scale=scale,
        attempt_supercell=True,
        allow_subset=False,
        max_relative_volume_difference=(
            float(max_relative_volume_difference)
            if max_relative_volume_difference is not None
            else None
        ),
    )


def _fit_with_settings(
    first: Any,
    second: Any,
    settings: CrystalMatcherSettings,
) -> tuple[bool | None, str | None]:
    matcher_type = _pymatgen_modules()["matcher"].StructureMatcher
    matcher = matcher_type(
        ltol=settings.ltol,
        stol=settings.stol,
        angle_tol=settings.angle_tol,
        primitive_cell=settings.primitive_cell,
        scale=settings.scale,
        attempt_supercell=settings.attempt_supercell,
        allow_subset=settings.allow_subset,
    )
    try:
        return bool(matcher.fit(first, second, symmetric=True)), None
    except TypeError:
        # Older pymatgen releases do not expose the symmetric keyword.  Requiring
        # both directions prevents ordering-dependent deletion in those versions.
        try:
            forward = bool(matcher.fit(first, second))
            reverse = bool(matcher.fit(second, first))
        except Exception as exc:
            return None, f"structure_match_failed:{type(exc).__name__}"
        if forward != reverse:
            return None, "structure_match_direction_dependent"
        return forward, None
    except Exception as exc:
        return None, f"structure_match_failed:{type(exc).__name__}"


def _relative_volume_difference(first: Any, second: Any) -> float:
    """Return symmetric per-atom volume difference.

    A total-cell comparison would incorrectly reject an otherwise equivalent
    primitive/supercell pair before ``StructureMatcher`` can establish the
    mapping.
    """

    if len(first) < 1 or len(second) < 1:
        raise CrystalIdentityError("volume comparison requires non-empty structures")
    first_volume = float(first.volume) / len(first)
    second_volume = float(second.volume) / len(second)
    if (
        not math.isfinite(first_volume)
        or not math.isfinite(second_volume)
        or first_volume <= 0
        or second_volume <= 0
    ):
        raise CrystalIdentityError(
            "relative-volume comparison requires finite positive cell volumes"
        )
    mean_volume = (first_volume + second_volume) / 2.0
    return abs(first_volume - second_volume) / mean_volume


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
    payload = _structure_fingerprint_fields(
        structure,
        coordinate_decimals=coordinate_decimals,
    )
    return {
        "schema": _FINGERPRINT_SCHEMA,
        "standardization": {
            "cell": "primitive_niggli",
            "symprec_angstrom": symprec,
            "angle_tolerance_degrees": angle_tolerance,
            "coordinate_decimals": coordinate_decimals,
        },
        **payload,
    }


def _identity_fingerprint_payload(
    structure: Any,
    *,
    coordinate_decimals: int,
) -> dict[str, Any]:
    payload = _structure_fingerprint_fields(
        structure,
        coordinate_decimals=coordinate_decimals,
    )
    return {
        "schema": _IDENTITY_FINGERPRINT_SCHEMA,
        "standardization": {
            "cell": "source_niggli",
            "symmetry_refinement_used_for_identity": False,
            "coordinate_decimals": coordinate_decimals,
        },
        **payload,
    }


def _structure_fingerprint_fields(
    structure: Any,
    *,
    coordinate_decimals: int,
) -> dict[str, Any]:
    lattice = structure.lattice
    return {
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
    "CRYSTAL_IDENTITY_CANONICALIZATION",
    "CrystalAmbiguousComparison",
    "CanonicalCrystalStructure",
    "CrystalGeometryReport",
    "CrystalGroupingResult",
    "CrystalIdentityError",
    "CrystalMatchAssessment",
    "CrystalMatcherSettings",
    "CrystalMatchRelation",
    "CrystalStructureGroup",
    "InvalidCrystalGeometryError",
    "PymatgenRequiredError",
    "canonical_structure_hash",
    "canonicalize_crystal_structure",
    "classify_crystal_structure_relation",
    "crystal_structure_fingerprint",
    "deduplicate_crystal_structures",
    "exact_file_hash",
    "group_crystal_structures",
    "parse_crystal_structure",
    "validate_crystal_geometry",
]
