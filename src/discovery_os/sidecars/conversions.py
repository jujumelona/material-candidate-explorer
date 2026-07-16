"""Strict conversion helpers between central Candidates and model inputs."""

from __future__ import annotations

import io
import json
import math
import re
from collections.abc import Iterable
from typing import Any

from discovery_os.schemas import Candidate, CandidateRepresentation, RepresentationKind

from .base import require_module
from .errors import CandidateConversionError, ModelOutputError


_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO")
_RNA_ALPHABET = frozenset("ACGUNRYKMSWBDHVX")
_PDB_RESIDUES = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "SEC": "U",
    "PYL": "O",
    "ASX": "B",
    "GLX": "Z",
    "UNK": "X",
}


def representation(
    candidate: Candidate,
    kinds: Iterable[RepresentationKind],
) -> CandidateRepresentation:
    """Choose a canonical supported representation, rejecting ambiguity."""

    allowed = set(kinds)
    matches = [item for item in candidate.representations if item.kind in allowed]
    if not matches:
        names = ", ".join(sorted(str(item) for item in allowed))
        raise CandidateConversionError(
            f"candidate {candidate.candidate_id!r} has none of the required representations: {names}"
        )
    canonical = [item for item in matches if item.canonical]
    if len(canonical) > 1:
        raise CandidateConversionError("candidate has multiple canonical representations for one route")
    return canonical[0] if canonical else matches[0]


def candidate_smiles(
    candidate: Candidate,
    *,
    kinds: Iterable[RepresentationKind] = (
        RepresentationKind.SMILES,
        RepresentationKind.POLYMER_REPEAT_UNIT,
        RepresentationKind.REACTION_SMILES,
    ),
) -> str:
    """Return one validated line from an explicitly allowed SMILES-like route."""

    item = representation(candidate, kinds)
    value = item.value.strip()
    if not value or any(char in value for char in "\r\n\x00"):
        raise CandidateConversionError("SMILES representation must be one non-empty line")
    return value


def candidate_sequence(candidate: Candidate, *, molecule: str) -> str:
    if molecule == "protein":
        kinds = (RepresentationKind.PROTEIN_SEQUENCE, RepresentationKind.FASTA)
        alphabet = _PROTEIN_ALPHABET
    elif molecule == "rna":
        kinds = (RepresentationKind.RNA_SEQUENCE, RepresentationKind.FASTA)
        alphabet = _RNA_ALPHABET
    else:
        raise ValueError("molecule must be protein or rna")
    try:
        raw = representation(candidate, kinds).value
    except CandidateConversionError:
        if molecule != "protein":
            raise
        return protein_sequence_from_pdb(candidate)
    lines = [line.strip() for line in raw.splitlines() if line.strip() and not line.startswith(">")]
    sequence = "".join(lines).replace(" ", "").upper()
    if not sequence:
        raise CandidateConversionError(f"{molecule} sequence is empty")
    invalid = sorted(set(sequence) - alphabet)
    if invalid:
        raise CandidateConversionError(
            f"{molecule} sequence contains unsupported symbols: {''.join(invalid)}"
        )
    return sequence


def protein_sequence_from_pdb(candidate: Candidate) -> str:
    """Deterministically recover one protein-chain sequence from PDB ATOM rows."""

    raw = representation(candidate, (RepresentationKind.PDB,)).value
    residues: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    chains: set[str] = set()
    in_first_model = True
    for line in raw.splitlines():
        record = line[:6].strip().upper()
        if record == "MODEL":
            if not in_first_model:
                break
            in_first_model = False
            continue
        if record == "ENDMDL":
            break
        if record != "ATOM" or len(line) < 27:
            continue
        altloc = line[16:17]
        if altloc not in {" ", "A", "1"}:
            continue
        chain = line[21:22].strip() or "_"
        key = (chain, line[22:26].strip(), line[26:27].strip())
        if key in seen:
            continue
        residue = line[17:20].strip().upper()
        symbol = _PDB_RESIDUES.get(residue)
        if symbol is None:
            raise CandidateConversionError(f"PDB contains unsupported residue {residue!r}")
        seen.add(key)
        chains.add(chain)
        residues.append(symbol)
    if not residues:
        raise CandidateConversionError("PDB contains no protein ATOM residues")
    if len(chains) != 1:
        raise CandidateConversionError(
            "ESM sequence extraction requires a single PDB chain; use Boltz for multimers"
        )
    return "".join(residues)


def candidate_to_ase(
    candidate: Candidate,
    *,
    max_atoms: int = 20_000,
    kinds: Iterable[RepresentationKind] = (
        RepresentationKind.CIF,
        RepresentationKind.POSCAR,
        RepresentationKind.XYZ,
        RepresentationKind.EXTXYZ,
        RepresentationKind.SDF,
    ),
) -> Any:
    """Convert CIF/POSCAR/XYZ/EXTXYZ/SDF to a validated ASE Atoms object."""

    ase_io = require_module("ase.io", install_hint="install ASE in this sidecar environment")
    item = representation(candidate, kinds)
    formats = {
        RepresentationKind.CIF: "cif",
        RepresentationKind.POSCAR: "vasp",
        RepresentationKind.XYZ: "xyz",
        RepresentationKind.EXTXYZ: "extxyz",
        RepresentationKind.SDF: "sdf",
    }
    try:
        atoms = ase_io.read(io.StringIO(item.value), format=formats[item.kind], index=0)
    except Exception as exc:
        raise CandidateConversionError(
            f"ASE could not parse the {item.kind} representation: {type(exc).__name__}: {exc}"
        ) from exc
    _validate_atoms(atoms, max_atoms=max_atoms)
    return atoms


def candidate_to_pymatgen(candidate: Candidate, *, max_atoms: int = 20_000) -> Any:
    """Convert a structure representation to pymatgen Structure."""

    core = require_module(
        "pymatgen.core",
        install_hint="install pymatgen in the CHGNet/PySCF sidecar environment",
    )
    try:
        item = representation(candidate, (RepresentationKind.CIF, RepresentationKind.POSCAR))
    except CandidateConversionError:
        atoms = candidate_to_ase(candidate, max_atoms=max_atoms)
        adaptor_module = require_module(
            "pymatgen.io.ase",
            install_hint="install pymatgen with ASE conversion support",
        )
        try:
            structure = adaptor_module.AseAtomsAdaptor.get_structure(atoms)
        except Exception as exc:
            raise CandidateConversionError(
                f"pymatgen could not convert ASE Atoms: {type(exc).__name__}: {exc}"
            ) from exc
    else:
        fmt = "cif" if item.kind == RepresentationKind.CIF else "poscar"
        try:
            structure = core.Structure.from_str(item.value, fmt=fmt)
        except Exception as exc:
            raise CandidateConversionError(
                f"pymatgen could not parse {item.kind}: {type(exc).__name__}: {exc}"
            ) from exc
    if len(structure) <= 0 or len(structure) > max_atoms:
        raise CandidateConversionError(f"structure atom count must be between 1 and {max_atoms}")
    return structure


def ase_to_cif(atoms: Any, *, max_bytes: int = 4 * 1024 * 1024) -> str:
    ase_io = require_module("ase.io", install_hint="install ASE in the MatterGen sidecar")
    _validate_atoms(atoms, max_atoms=20_000)
    buffer = io.StringIO()
    try:
        ase_io.write(buffer, atoms, format="cif")
    except Exception as exc:
        raise ModelOutputError(f"ASE could not serialize generated CIF: {type(exc).__name__}: {exc}") from exc
    value = buffer.getvalue()
    if not value or len(value.encode("utf-8")) > max_bytes:
        raise ModelOutputError("generated CIF is empty or exceeds the representation limit")
    return value


def pymatgen_to_cif(structure: Any, *, max_bytes: int = 4 * 1024 * 1024) -> str:
    cif_module = require_module(
        "pymatgen.io.cif",
        install_hint="install pymatgen in this sidecar environment",
    )
    try:
        value = str(cif_module.CifWriter(structure))
    except Exception as exc:
        raise ModelOutputError(
            f"pymatgen could not serialize generated CIF: {type(exc).__name__}: {exc}"
        ) from exc
    if not value or len(value.encode("utf-8")) > max_bytes:
        raise ModelOutputError("generated CIF is empty or exceeds the representation limit")
    return value


def atom_entity_ids(atoms_or_structure: Any) -> tuple[str, ...]:
    try:
        count = len(atoms_or_structure)
    except Exception as exc:
        raise ModelOutputError("model structure has no atom count") from exc
    return tuple(f"atom:{index}" for index in range(count))


def periodic_atom_entity_ids(atoms_or_structure: Any) -> tuple[str, ...]:
    """Bind periodic atom rows by species and wrapped fractional position.

    Index-only labels cannot detect a parser that reorders sites.  MatterSim,
    CHGNet, and UMA use these coordinate-bearing IDs so downstream force
    comparisons must explicitly realign the same physical sites.
    """

    try:
        if callable(getattr(atoms_or_structure, "get_scaled_positions", None)):
            symbols = [str(item) for item in atoms_or_structure.get_chemical_symbols()]
            coordinates = atoms_or_structure.get_scaled_positions(wrap=True)
        else:
            sites = list(atoms_or_structure)
            symbols = []
            coordinates = []
            for site in sites:
                specie = getattr(site, "specie", None)
                symbol = getattr(specie, "symbol", None)
                if symbol is None:
                    raise ValueError("periodic site is not an ordered elemental species")
                symbols.append(str(symbol))
                coordinates.append(site.frac_coords)
    except Exception as exc:
        raise ModelOutputError(
            "periodic structure does not expose species and fractional coordinates"
        ) from exc
    if not symbols or len(symbols) != len(coordinates):
        raise ModelOutputError("periodic atom identity has an invalid site count")

    identifiers: list[str] = []
    occurrences: dict[str, int] = {}
    for symbol, coordinate in zip(symbols, coordinates, strict=True):
        values = [_wrapped_fractional_coordinate(item) for item in coordinate]
        if len(values) != 3:
            raise ModelOutputError("periodic atom identity requires three coordinates")
        base = f"site:{symbol}:" + ":".join(f"{item:.8f}" for item in values)
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        identifiers.append(base if occurrence == 0 else f"{base}#{occurrence}")
    if len(identifiers) != len(set(identifiers)):
        raise ModelOutputError("periodic atom identities are not unique")
    return tuple(identifiers)


def _wrapped_fractional_coordinate(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("fractional coordinate is not finite")
    wrapped = round(number % 1.0, 8)
    if math.isclose(wrapped, 0.0, abs_tol=1e-8) or math.isclose(
        wrapped, 1.0, abs_tol=1e-8
    ):
        return 0.0
    return wrapped


def ase_chemical_system(atoms: Any) -> str:
    try:
        symbols = sorted(set(str(item) for item in atoms.get_chemical_symbols()))
    except Exception as exc:
        raise CandidateConversionError("could not derive a chemical system from parent atoms") from exc
    if not symbols or any(not re.fullmatch(r"[A-Z][a-z]?", item) for item in symbols):
        raise CandidateConversionError("parent contains invalid chemical symbols")
    return "-".join(symbols)


def cell_expression(
    candidate: Candidate,
    *,
    max_genes: int = 65_536,
) -> tuple[list[str], list[float], str]:
    """Parse the canonical one-cell ``genes``/``values`` JSON representation.

    The shape is deliberately explicit rather than accepting a gene-to-value
    object whose duplicate keys would already have been discarded by a JSON
    parser::

        {
          "genes": ["TP53", "GAPDH"],
          "values": [2.0, 8.5],
          "value_semantics": "raw_counts"
        }
    """

    item = representation(candidate, (RepresentationKind.CELL_EXPRESSION, RepresentationKind.CUSTOM))
    try:
        raw = json.loads(item.value, object_pairs_hook=_unique_json_object)
    except (TypeError, ValueError) as exc:
        raise CandidateConversionError("cell expression must be valid duplicate-free JSON") from exc
    if not isinstance(raw, dict) or set(raw) != {"genes", "values", "value_semantics"}:
        raise CandidateConversionError(
            "cell expression must contain exactly genes, values, and value_semantics"
        )
    raw_genes = raw["genes"]
    raw_values = raw["values"]
    if not isinstance(raw_genes, list) or not isinstance(raw_values, list):
        raise CandidateConversionError("cell expression genes and values must be arrays")
    if len(raw_genes) != len(raw_values):
        raise CandidateConversionError("cell expression genes and values lengths must match")
    if not raw_genes or len(raw_genes) > max_genes:
        raise CandidateConversionError(f"cell expression must contain 1..{max_genes} genes")
    value_semantics = raw["value_semantics"]
    if value_semantics not in {"raw_counts", "normalized_log1p"}:
        raise CandidateConversionError(
            "cell expression value_semantics must be raw_counts or normalized_log1p"
        )
    genes: list[str] = []
    values: list[float] = []
    for gene, value in zip(raw_genes, raw_values, strict=True):
        if not isinstance(gene, str) or not gene.strip() or len(gene) > 256:
            raise CandidateConversionError("cell expression contains an invalid gene name")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CandidateConversionError("cell expression contains a non-numeric value")
        number = float(value)
        if not math.isfinite(number):
            raise CandidateConversionError("cell expression contains NaN or infinity")
        genes.append(gene.strip())
        values.append(number)
    if len(genes) != len(set(genes)):
        raise CandidateConversionError("cell expression contains duplicate genes")
    return genes, values, value_semantics


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _validate_atoms(atoms: Any, *, max_atoms: int) -> None:
    try:
        count = len(atoms)
        numbers = list(atoms.get_atomic_numbers())
        positions = atoms.get_positions().tolist()
    except Exception as exc:
        raise CandidateConversionError("parsed object is not a valid ASE Atoms instance") from exc
    if count <= 0 or count > max_atoms or len(numbers) != count or len(positions) != count:
        raise CandidateConversionError(f"structure atom count must be between 1 and {max_atoms}")
    for row in positions:
        if len(row) != 3 or any(not math.isfinite(float(value)) for value in row):
            raise CandidateConversionError("structure contains invalid Cartesian coordinates")
    if any(not isinstance(value, int) or value <= 0 for value in numbers):
        raise CandidateConversionError("structure contains invalid atomic numbers")


__all__ = [
    "ase_chemical_system",
    "ase_to_cif",
    "atom_entity_ids",
    "candidate_sequence",
    "candidate_smiles",
    "candidate_to_ase",
    "candidate_to_pymatgen",
    "cell_expression",
    "pymatgen_to_cif",
    "periodic_atom_entity_ids",
    "protein_sequence_from_pdb",
    "representation",
]
