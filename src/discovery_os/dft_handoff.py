"""Portable, non-executing input handoff for periodic DFT validation.

This module prepares reviewable input packages; it does not run a DFT engine.
POTCAR files and pseudopotentials are never copied into an artifact package,
and calculated energies remain explicitly null until an external backend has
actually completed and validated a calculation.
"""

from __future__ import annotations

import importlib
import math
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


class KPointSamplingPlan(StrictSchema):
    """Structure-bound sampling proposal; it is not convergence evidence."""

    mode: Literal["explicit_grid", "reciprocal_spacing"]
    realized_grid: tuple[int, int, int]
    target_spacing_A_inv: float | None = Field(default=None, gt=0.0)
    reciprocal_vector_lengths_A_inv: tuple[float, float, float]
    convergence_required: Literal[True] = True

    @model_validator(mode="after")
    def _grid_is_positive(self) -> "KPointSamplingPlan":
        if any(item <= 0 for item in self.realized_grid):
            raise ValueError("realized k-point grid values must be positive")
        if any(item <= 0.0 or not math.isfinite(item) for item in self.reciprocal_vector_lengths_A_inv):
            raise ValueError("reciprocal-vector lengths must be finite and positive")
        if self.mode == "reciprocal_spacing" and self.target_spacing_A_inv is None:
            raise ValueError("reciprocal-spacing mode requires a target spacing")
        if self.mode == "explicit_grid" and self.target_spacing_A_inv is not None:
            raise ValueError("explicit-grid mode must not claim a target spacing")
        return self


class DFTConvergencePlan(StrictSchema):
    """Required external sweeps; every item remains unexecuted in this package."""

    status: Literal["required_not_executed"] = "required_not_executed"
    cutoff_scale_factors: tuple[float, ...] = ()
    kpoint_spacing_scale_factors: tuple[float, ...] = (1.25, 1.0, 0.8)
    target_energy_change_meV_atom: float = Field(default=1.0, gt=0.0)
    target_force_change_eV_A: float = Field(default=0.01, gt=0.0)
    target_stress_change_GPa: float = Field(default=0.1, gt=0.0)
    requires_pseudopotential_specific_cutoff_review: Literal[True] = True
    requires_magnetic_and_electronic_state_review: Literal[True] = True


class DFTInputManifest(StrictSchema):
    backend_id: Identifier
    backend_version: Identifier
    candidate_ref: CandidateRef
    shortlist_rank: int = Field(gt=0)
    calculation_engine: Literal["quantum_espresso_pw"] = "quantum_espresso_pw"
    calculation_type: Literal["vc-relax", "relax", "scf"] = "vc-relax"
    kpoint_sampling: KPointSamplingPlan
    convergence_plan: DFTConvergencePlan
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
    input_manifest_sha256: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    output_artifacts: list[ContentArtifactRef] = Field(default_factory=list)
    convergence_evidence_artifacts: list[ContentArtifactRef] = Field(
        default_factory=list
    )
    method_policy_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reference_set_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    converged: bool | None = None
    total_energy_eV: float | None = None
    energy_per_atom_eV: float | None = None
    formation_energy_eV_per_atom: float | None = None
    energy_above_hull_eV_per_atom: float | None = Field(default=None, ge=0.0)
    has_imaginary_modes: bool | None = None
    phonon_q_mesh: tuple[int, int, int] | None = None
    minimum_frequency_THz: float | None = None
    imaginary_mode_tolerance_THz: float | None = Field(default=None, ge=0.0)
    notes: list[NonEmptyText] = Field(default_factory=list)

    @model_validator(mode="after")
    def _executed_result_is_fail_closed(self) -> "PeriodicDFTCalculationResult":
        scientific_values = (
            self.total_energy_eV,
            self.energy_per_atom_eV,
            self.formation_energy_eV_per_atom,
            self.energy_above_hull_eV_per_atom,
            self.has_imaginary_modes,
            self.minimum_frequency_THz,
            self.imaginary_mode_tolerance_THz,
        )
        if self.status == "failed":
            if any(item is not None for item in scientific_values):
                raise ValueError("failed DFT results must not expose scientific values")
            if (
                self.output_artifacts
                or self.convergence_evidence_artifacts
                or self.converged is not None
                or self.reference_set_hash is not None
                or self.phonon_q_mesh is not None
            ):
                raise ValueError("failed DFT results must not expose completed evidence")
            return self
        if self.input_manifest_sha256 is None:
            raise ValueError("completed DFT results require an input-manifest hash")
        if not self.output_artifacts:
            raise ValueError("completed DFT results require immutable output artifacts")
        if not self.convergence_evidence_artifacts:
            raise ValueError(
                "completed DFT results require immutable convergence-evidence artifacts"
            )
        if self.method_policy_hash is None:
            raise ValueError("completed DFT results require a method policy hash")
        if self.converged is not True:
            raise ValueError("completed DFT results require explicit convergence")
        if self.calculation_stage in {"relax", "static_energy"} and self.energy_per_atom_eV is None:
            raise ValueError("completed relax/static results require energy_per_atom_eV")
        if self.calculation_stage == "phonon":
            if (
                self.phonon_q_mesh is None
                or any(item <= 0 for item in self.phonon_q_mesh)
                or self.minimum_frequency_THz is None
                or self.imaginary_mode_tolerance_THz is None
                or self.has_imaginary_modes is None
            ):
                raise ValueError(
                    "completed phonon results require q-mesh, minimum frequency, imaginary-mode tolerance, and mode classification"
                )
            expected_imaginary_modes = (
                self.minimum_frequency_THz < -self.imaginary_mode_tolerance_THz
            )
            if self.has_imaginary_modes is not expected_imaginary_modes:
                raise ValueError(
                    "phonon mode classification must equal minimum_frequency_THz < -imaginary_mode_tolerance_THz"
                )
        if (
            self.formation_energy_eV_per_atom is not None
            or self.energy_above_hull_eV_per_atom is not None
        ) and self.reference_set_hash is None:
            raise ValueError(
                "formation energy or energy above hull requires a reference-set hash"
            )
        if (
            self.energy_above_hull_eV_per_atom is not None
            and self.formation_energy_eV_per_atom is None
        ):
            raise ValueError(
                "energy above hull requires formation energy"
            )
        return self


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
    backend_version = "1.2.0"

    def __init__(
        self,
        *,
        calculation_type: Literal["vc-relax", "relax", "scf"] = "vc-relax",
        ecutwfc_ry: float | None = None,
        ecutrho_ry: float | None = None,
        kpoint_grid: tuple[int, int, int] | None = None,
        target_kpoint_spacing_A_inv: float = 0.30,
    ) -> None:
        if (ecutwfc_ry is None) != (ecutrho_ry is None):
            raise ValueError("ecutwfc_ry and ecutrho_ry must be supplied together")
        if ecutwfc_ry is not None and (
            ecutwfc_ry <= 0
            or ecutrho_ry is None
            or ecutrho_ry <= 0
            or ecutrho_ry < ecutwfc_ry
        ):
            raise ValueError("DFT cutoffs must be positive and ecutrho must not be below ecutwfc")
        if kpoint_grid is not None and (
            len(kpoint_grid) != 3
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in kpoint_grid
            )
        ):
            raise ValueError("kpoint_grid must contain three positive integers")
        if (
            not math.isfinite(target_kpoint_spacing_A_inv)
            or target_kpoint_spacing_A_inv <= 0.0
        ):
            raise ValueError("target_kpoint_spacing_A_inv must be finite and positive")
        self.calculation_type = calculation_type
        self.ecutwfc_ry = float(ecutwfc_ry) if ecutwfc_ry is not None else None
        self.ecutrho_ry = float(ecutrho_ry) if ecutrho_ry is not None else None
        self.kpoint_grid = tuple(kpoint_grid) if kpoint_grid is not None else None
        self.target_kpoint_spacing_A_inv = float(target_kpoint_spacing_A_inv)

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
        kpoint_sampling = self._kpoint_sampling(structure)
        convergence_plan = DFTConvergencePlan(
            cutoff_scale_factors=(0.8, 1.0, 1.2)
            if self.ecutwfc_ry is not None
            else (),
        )
        qe_text = self._qe_input(
            candidate,
            structure,
            kpoint_grid=kpoint_sampling.realized_grid,
        )
        package_hash = stable_hash(
            {
                "candidate_ref": reference,
                "backend_id": self.backend_id,
                "backend_version": self.backend_version,
                "calculation_type": self.calculation_type,
                "ecutwfc_ry": self.ecutwfc_ry,
                "ecutrho_ry": self.ecutrho_ry,
                "kpoint_sampling": kpoint_sampling,
                "convergence_plan": convergence_plan,
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
            kpoint_sampling=kpoint_sampling,
            convergence_plan=convergence_plan,
            input_artifacts=artifacts,
            required_external_configuration=[
                "Supply reviewed external pseudopotential files matching the EXTERNAL_PSEUDO_<ELEMENT>.UPF references in pw.in; no files are bundled.",
                "Set external pseudo_dir, scratch outdir, and a run-specific prefix in pw.in.",
                "Resolve pseudopotential-specific cutoffs, then execute and retain the manifest convergence sweeps for energy, force, stress, and reciprocal k-point spacing.",
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

    def _kpoint_sampling(self, structure: object) -> KPointSamplingPlan:
        try:
            lengths = tuple(
                float(item)
                for item in structure.lattice.reciprocal_lattice.abc  # type: ignore[union-attr]
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("DFT input generation requires reciprocal lattice vectors") from exc
        if len(lengths) != 3 or any(item <= 0.0 or not math.isfinite(item) for item in lengths):
            raise ValueError("reciprocal lattice vectors must have three finite positive lengths")
        if self.kpoint_grid is not None:
            return KPointSamplingPlan(
                mode="explicit_grid",
                realized_grid=self.kpoint_grid,
                reciprocal_vector_lengths_A_inv=lengths,
            )
        grid = tuple(
            max(1, int(math.ceil(item / self.target_kpoint_spacing_A_inv)))
            for item in lengths
        )
        return KPointSamplingPlan(
            mode="reciprocal_spacing",
            realized_grid=grid,
            target_spacing_A_inv=self.target_kpoint_spacing_A_inv,
            reciprocal_vector_lengths_A_inv=lengths,
        )

    def _qe_input(
        self,
        candidate: Candidate,
        structure: object,
        *,
        kpoint_grid: tuple[int, int, int],
    ) -> str:
        kx, ky, kz = kpoint_grid
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

        cutoff_lines = (
            f"  ecutwfc = {self.ecutwfc_ry:.8g}\n"
            f"  ecutrho = {self.ecutrho_ry:.8g}"
            if self.ecutwfc_ry is not None and self.ecutrho_ry is not None
            else (
                "  ! Resolve these values from the selected external pseudopotentials "
                "and a retained convergence sweep.\n"
                "  ecutwfc = INSERT_CONVERGED_ECUTWFC_RY\n"
                "  ecutrho = INSERT_CONVERGED_ECUTRHO_RY"
            )
        )

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
{cutoff_lines}
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
    "DFTConvergencePlan",
    "DFTInputHandoffReport",
    "DFTInputManifest",
    "DFTInputPackage",
    "KPointSamplingPlan",
    "PeriodicDFTCalculationResult",
    "PeriodicDFTBackend",
    "PortablePeriodicDFTInputBackend",
    "UncalculatedPeriodicProperties",
]
