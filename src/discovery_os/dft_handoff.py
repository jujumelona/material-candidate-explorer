"""Portable, non-executing input handoff for periodic DFT validation.

This module prepares reviewable input packages; it does not run a DFT engine.
POTCAR files and pseudopotentials are never copied into an artifact package,
and calculated energies remain explicitly null until an external backend has
actually completed and validated a calculation.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from pathlib import PurePosixPath
from typing import Literal, Protocol

from pydantic import Field, model_validator

from .artifacts import ArtifactStore
from .crystal_identity import parse_crystal_structure, validate_crystal_geometry
from .fusion_schemas import ContentArtifactRef
from .hashing import canonical_json, stable_hash
from .schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    Identifier,
    NonEmptyText,
    RepresentationKind,
    StrictSchema,
)


_PSEUDOPOTENTIAL_SUFFIXES = frozenset({".upf", ".psp8", ".psml", ".pot"})


class UncalculatedPeriodicProperties(StrictSchema):
    total_energy_eV: None = None
    energy_per_atom_eV: None = None
    formation_energy_eV_per_atom: None = None
    energy_above_hull_eV_per_atom: None = None


class DFTInputManifest(StrictSchema):
    backend_id: Identifier
    backend_version: Identifier
    candidate_ref: CandidateRef
    shortlist_rank: int = Field(gt=0)
    calculation_engine: Literal["quantum_espresso_pw"] = "quantum_espresso_pw"
    calculation_type: Literal["vc-relax", "relax", "scf"] = "vc-relax"
    input_artifacts: list[ContentArtifactRef] = Field(min_length=2)
    pseudopotentials_included: Literal[False] = False
    calculation_executed: Literal[False] = False
    required_external_configuration: list[NonEmptyText] = Field(min_length=1)
    uncalculated_properties: UncalculatedPeriodicProperties = Field(
        default_factory=UncalculatedPeriodicProperties
    )
    warnings: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def _package_has_no_external_potential_files(self) -> "DFTInputManifest":
        paths = [PurePosixPath(item.relative_path) for item in self.input_artifacts]
        names = [item.name.casefold() for item in paths]
        if "potcar" in names or any(path.suffix.casefold() in _PSEUDOPOTENTIAL_SUFFIXES for path in paths):
            raise ValueError("DFT handoff packages must not bundle POTCAR or pseudopotentials")
        if len({item.relative_path for item in self.input_artifacts}) != len(
            self.input_artifacts
        ):
            raise ValueError("duplicate DFT input artifact paths are not allowed")
        if not any(path.name == "structure.cif" for path in paths):
            raise ValueError("DFT handoff package requires structure.cif")
        if not any(path.name in {"POSCAR", "pw.in"} for path in paths):
            raise ValueError("DFT handoff package requires POSCAR or a Quantum ESPRESSO input skeleton")
        return self


class DFTInputPackage(StrictSchema):
    manifest: DFTInputManifest
    manifest_artifact: ContentArtifactRef

    @model_validator(mode="after")
    def _manifest_artifact_is_json(self) -> "DFTInputPackage":
        if self.manifest_artifact.media_type != "application/json":
            raise ValueError("DFT handoff manifest artifact must be JSON")
        if PurePosixPath(self.manifest_artifact.relative_path).name != "manifest.json":
            raise ValueError("DFT handoff manifest must be named manifest.json")
        return self


class DFTInputHandoffReport(StrictSchema):
    backend_id: Identifier
    backend_version: Identifier
    candidates_received: int = Field(ge=0)
    top_k: int = Field(gt=0, le=5)
    packages: list[DFTInputPackage] = Field(default_factory=list)
    calculation_executed: Literal[False] = False
    scientific_boundary: NonEmptyText = (
        "input preparation is not a DFT result; relaxation, convergence, references, and hull construction remain external"
    )

    @model_validator(mode="after")
    def _package_count_is_bounded(self) -> "DFTInputHandoffReport":
        if len(self.packages) > min(self.candidates_received, self.top_k):
            raise ValueError("DFT handoff contains more packages than the requested shortlist")
        if any(
            item.manifest.backend_id != self.backend_id
            or item.manifest.backend_version != self.backend_version
            for item in self.packages
        ):
            raise ValueError("DFT package backend identity differs from the report")
        return self


class PeriodicDFTCalculationResult(StrictSchema):
    candidate_ref: CandidateRef
    calculation_stage: Literal["relax", "static_energy", "phonon"]
    status: Literal["completed", "failed"]
    output_artifacts: list[ContentArtifactRef] = Field(default_factory=list)
    total_energy_eV: float | None = None
    energy_per_atom_eV: float | None = None
    formation_energy_eV_per_atom: float | None = None
    energy_above_hull_eV_per_atom: float | None = None
    has_imaginary_modes: bool | None = None
    notes: list[NonEmptyText] = Field(default_factory=list)


class PeriodicDFTBackend(Protocol):
    backend_id: str
    backend_version: str

    def prepare_inputs(
        self,
        candidates: Sequence[Candidate],
        *,
        artifact_store: ArtifactStore,
        top_k: int,
    ) -> DFTInputHandoffReport: ...

    def relax(self, package: DFTInputPackage) -> PeriodicDFTCalculationResult: ...

    def static_energy(self, package: DFTInputPackage) -> PeriodicDFTCalculationResult: ...

    def phonon(self, package: DFTInputPackage) -> PeriodicDFTCalculationResult: ...


class PortablePeriodicDFTInputBackend:
    """Convert ranked CIF candidates into reviewable QE/POSCAR input packages.

    This class is an input packager, not a calculation engine.  It deliberately
    raises ``NotImplementedError`` for the execution methods in
    :class:`PeriodicDFTBackend` so callers cannot mistake generated files for a
    completed relaxation, static-energy calculation, or phonon calculation.
    """

    backend_id = "periodic-dft-input-packager"
    backend_version = "1.1.0"

    def __init__(
        self,
        *,
        calculation_type: Literal["vc-relax", "relax", "scf"] = "vc-relax",
        ecutwfc_ry: float = 60.0,
        ecutrho_ry: float = 480.0,
        kpoint_grid: tuple[int, int, int] = (4, 4, 4),
    ) -> None:
        if ecutwfc_ry <= 0 or ecutrho_ry <= 0 or ecutrho_ry < ecutwfc_ry:
            raise ValueError("DFT cutoffs must be positive and ecutrho must not be below ecutwfc")
        if len(kpoint_grid) != 3 or any(
            isinstance(item, bool) or not isinstance(item, int) or item <= 0
            for item in kpoint_grid
        ):
            raise ValueError("kpoint_grid must contain three positive integers")
        self.calculation_type = calculation_type
        self.ecutwfc_ry = float(ecutwfc_ry)
        self.ecutrho_ry = float(ecutrho_ry)
        self.kpoint_grid = tuple(kpoint_grid)

    def prepare_inputs(
        self,
        candidates: Sequence[Candidate],
        *,
        artifact_store: ArtifactStore,
        top_k: int,
    ) -> DFTInputHandoffReport:
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
            raise ValueError("top_k must be an integer between 1 and 5")
        packages = [
            self._prepare_one(candidate, rank=rank, artifact_store=artifact_store)
            for rank, candidate in enumerate(candidates[:top_k], 1)
        ]
        return DFTInputHandoffReport(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            candidates_received=len(candidates),
            top_k=top_k,
            packages=packages,
        )

    def relax(self, package: DFTInputPackage) -> PeriodicDFTCalculationResult:
        del package
        raise NotImplementedError(
            "PortablePeriodicDFTInputBackend only packages inputs; configure an "
            "executing PeriodicDFTBackend to run a relaxation"
        )

    def static_energy(
        self,
        package: DFTInputPackage,
    ) -> PeriodicDFTCalculationResult:
        del package
        raise NotImplementedError(
            "PortablePeriodicDFTInputBackend only packages inputs; configure an "
            "executing PeriodicDFTBackend to run a static-energy calculation"
        )

    def phonon(self, package: DFTInputPackage) -> PeriodicDFTCalculationResult:
        del package
        raise NotImplementedError(
            "PortablePeriodicDFTInputBackend only packages inputs; configure an "
            "executing PeriodicDFTBackend to run a phonon calculation"
        )

    def _prepare_one(
        self,
        candidate: Candidate,
        *,
        rank: int,
        artifact_store: ArtifactStore,
    ) -> DFTInputPackage:
        reference = _required_periodic_candidate(candidate)
        cif = _required_representation(candidate, RepresentationKind.CIF)
        structure = parse_crystal_structure(cif.value, fmt="cif")
        validate_crystal_geometry(structure)
        if not bool(getattr(structure, "is_ordered", False)):
            raise ValueError(
                "DFT input generation requires a fully ordered crystal structure"
            )
        poscar_text = _poscar_from_structure(structure)
        qe_text = self._qe_input(candidate, structure)
        package_hash = stable_hash(
            {
                "candidate_ref": reference,
                "backend_id": self.backend_id,
                "backend_version": self.backend_version,
                "calculation_type": self.calculation_type,
                "ecutwfc_ry": self.ecutwfc_ry,
                "ecutrho_ry": self.ecutrho_ry,
                "kpoint_grid": self.kpoint_grid,
            }
        )
        candidate_component = artifact_store.safe_component(candidate.candidate_id)
        root = (
            f"dft/handoff/{rank:03d}-{candidate_component}-{package_hash[:12]}"
        )
        artifacts = [
            _write_artifact(
                artifact_store,
                f"{root}/structure.cif",
                _normalized_text(cif.value).encode("utf-8"),
                media_type="chemical/x-cif",
                role="cif",
            ),
            _write_artifact(
                artifact_store,
                f"{root}/POSCAR",
                _normalized_text(poscar_text).encode("utf-8"),
                media_type="text/plain",
                role="poscar-generated-from-cif",
            ),
        ]
        artifacts.append(
            _write_artifact(
                artifact_store,
                f"{root}/pw.in",
                qe_text.encode("utf-8"),
                media_type="text/plain",
                role="quantum-espresso-input",
            )
        )
        warnings = [
            "No calculation was executed; all energy, formation-energy, and hull fields remain null.",
            "Pseudopotentials, POTCAR files, scheduler scripts, and credentials are intentionally excluded.",
            "POSCAR and pw.in were generated from structure.cif with pymatgen; review chemistry and convergence settings before execution.",
        ]
        manifest = DFTInputManifest(
            backend_id=self.backend_id,
            backend_version=self.backend_version,
            candidate_ref=reference,
            shortlist_rank=rank,
            calculation_type=self.calculation_type,
            input_artifacts=artifacts,
            required_external_configuration=[
                "Supply reviewed external pseudopotential files matching the EXTERNAL_PSEUDO_<ELEMENT>.UPF references in pw.in; no files are bundled.",
                "Set external pseudo_dir, scratch outdir, and a run-specific prefix in pw.in.",
                "Set convergence-tested cutoffs, k-point density, spin, charge, and functional policies for the target chemistry.",
                "Run relaxation and static calculations before computing reference-consistent formation energy or a convex hull.",
            ],
            warnings=warnings,
        )
        encoded_manifest = (canonical_json(manifest) + "\n").encode("utf-8")
        manifest_artifact = _write_artifact(
            artifact_store,
            f"{root}/manifest.json",
            encoded_manifest,
            media_type="application/json",
            role="manifest",
        )
        return DFTInputPackage(
            manifest=manifest,
            manifest_artifact=manifest_artifact,
        )

    def _qe_input(self, candidate: Candidate, structure: object) -> str:
        kx, ky, kz = self.kpoint_grid
        species: dict[str, float] = {}
        atomic_positions: list[str] = []
        for site in structure:  # type: ignore[union-attr]
            try:
                symbol = str(site.specie.symbol)
                mass = float(site.specie.atomic_mass)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "DFT input generation requires one ordered elemental species per site"
                ) from exc
            species.setdefault(symbol, mass)
            coordinates = [float(value) for value in site.frac_coords]
            atomic_positions.append(
                f"{symbol} " + " ".join(f"{value:.12f}" for value in coordinates)
            )
        if not species or not atomic_positions:
            raise ValueError("DFT input generation requires at least one atomic site")

        atomic_species = [
            f"{symbol} {species[symbol]:.8f} EXTERNAL_PSEUDO_{symbol}.UPF"
            for symbol in sorted(species)
        ]
        lattice_rows = [
            " ".join(f"{float(value):.12f}" for value in row)
            for row in structure.lattice.matrix  # type: ignore[union-attr]
        ]
        dynamics = ""
        if self.calculation_type in {"relax", "vc-relax"}:
            dynamics += "&IONS\n  ion_dynamics = 'bfgs'\n/\n"
        if self.calculation_type == "vc-relax":
            dynamics += "&CELL\n  cell_dynamics = 'bfgs'\n/\n"

        return f"""! Quantum ESPRESSO pw.x input generated by Discovery OS.
! NOT EXECUTABLE AS-IS. No pseudopotential files are bundled.
! Candidate: {candidate.candidate_id}
! Lattice and fractional coordinates were converted from structure.cif by pymatgen.
&CONTROL
  calculation = '{self.calculation_type}'
  prefix = 'INSERT_PREFIX'
  pseudo_dir = 'INSERT_EXTERNAL_PSEUDO_DIR'
  outdir = 'INSERT_SCRATCH_DIR'
/
&SYSTEM
  ibrav = 0
  nat = {len(atomic_positions)}
  ntyp = {len(species)}
  ecutwfc = {self.ecutwfc_ry:.8g}
  ecutrho = {self.ecutrho_ry:.8g}
/
&ELECTRONS
  conv_thr = 1.0d-8
/
{dynamics}ATOMIC_SPECIES
{chr(10).join(atomic_species)}
CELL_PARAMETERS angstrom
{chr(10).join(lattice_rows)}
ATOMIC_POSITIONS crystal
{chr(10).join(atomic_positions)}
K_POINTS automatic
{kx} {ky} {kz} 0 0 0
"""


def _poscar_from_structure(structure: object) -> str:
    try:
        module = importlib.import_module("pymatgen.io.vasp.inputs")
        poscar_type = getattr(module, "Poscar")
        return str(poscar_type(structure, sort_structure=True))
    except (ImportError, ModuleNotFoundError):
        raise
    except Exception as exc:
        raise ValueError(
            f"pymatgen could not serialize POSCAR: {type(exc).__name__}: {exc}"
        ) from exc


def _write_artifact(
    store: ArtifactStore,
    relative_path: str,
    payload: bytes,
    *,
    media_type: str,
    role: str,
) -> ContentArtifactRef:
    normalized_path, digest = store.write_bytes(relative_path, payload)
    return ContentArtifactRef(
        artifact_id=f"DFT-{stable_hash([role, normalized_path, digest])[:32]}",
        relative_path=normalized_path,
        sha256=digest,
        media_type=media_type,
        byte_size=len(payload),
    )


def _required_periodic_candidate(candidate: Candidate) -> CandidateRef:
    if candidate.candidate_ref is None:
        raise ValueError("DFT handoff requires immutable candidate_ref values")
    if not any(
        item.kind in {RepresentationKind.CIF, RepresentationKind.POSCAR}
        for item in candidate.representations
    ):
        raise ValueError("DFT handoff requires a periodic CIF or POSCAR representation")
    return candidate.candidate_ref


def _required_representation(
    candidate: Candidate,
    kind: RepresentationKind,
) -> CandidateRepresentation:
    value = _optional_representation(candidate, kind)
    if value is None:
        raise ValueError(f"DFT handoff requires a {kind} representation")
    return value


def _optional_representation(
    candidate: Candidate,
    kind: RepresentationKind,
) -> CandidateRepresentation | None:
    rows = [item for item in candidate.representations if item.kind == kind]
    if not rows:
        return None
    canonical = [item for item in rows if item.canonical]
    return canonical[0] if len(canonical) == 1 else rows[0]


def _normalized_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"


__all__ = [
    "DFTInputHandoffReport",
    "DFTInputManifest",
    "DFTInputPackage",
    "PeriodicDFTCalculationResult",
    "PeriodicDFTBackend",
    "PortablePeriodicDFTInputBackend",
    "UncalculatedPeriodicProperties",
]
