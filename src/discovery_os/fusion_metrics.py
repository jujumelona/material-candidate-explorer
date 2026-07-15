"""Deterministic workspace OFF/ON diagnostics.

These metrics compare generated candidates.  They do not participate in the
evidence gate and cannot establish efficacy, stability, superconductivity, or
any other scientific claim.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from .artifacts import ArtifactStore
from .chemistry import FormulaError, parse_formula
from .fusion_schemas import (
    DiagnosticProperty,
    ExpertFeaturePayload,
    FeatureStatus,
    FusionWorkspaceSnapshot,
    NumericTensor,
    ObjectiveDelta,
    WorkspaceComparisonReport,
    WorkspaceMode,
)
from .hashing import candidate_content_hash, stable_hash
from .schemas import (
    Candidate,
    CandidateType,
    DiscoveryGoal,
    ObjectiveDirection,
    RepresentationKind,
)


_BOHR_TO_ANGSTROM = 0.529177210903
_LENGTH_UNIT_TO_ANGSTROM = {
    "a": 1.0,
    "angstrom": 1.0,
    "angstroms": 1.0,
    "å": 1.0,
    "Å": 1.0,
    "nm": 10.0,
    "nanometer": 10.0,
    "nanometers": 10.0,
    "nanometre": 10.0,
    "nanometres": 10.0,
    "pm": 0.01,
    "picometer": 0.01,
    "picometers": 0.01,
    "picometre": 0.01,
    "picometres": 0.01,
    "m": 1.0e10,
    "meter": 1.0e10,
    "meters": 1.0e10,
    "metre": 1.0e10,
    "metres": 1.0e10,
    "bohr": _BOHR_TO_ANGSTROM,
    "bohrs": _BOHR_TO_ANGSTROM,
    "a0": _BOHR_TO_ANGSTROM,
    "atomicunit": _BOHR_TO_ANGSTROM,
    "atomicunitoflength": _BOHR_TO_ANGSTROM,
}
_DYNAMIC_FIRST_AXIS_ROLES = {
    "atom_embedding",
    "cell_embedding",
    "token_embedding",
}
_PROTEIN_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYBXZJUO*")
_RNA_ALPHABET = frozenset("ACGUNRYSWKMBDHV")


def scientific_identity_hash(candidate: Candidate) -> str:
    """Hash scientific representations without run/generator provenance."""

    payload = {
        "candidate_type": str(candidate.candidate_type),
        "domain": str(candidate.domain),
        "representations": sorted(
            (
                {
                    "kind": str(item.kind),
                    "value": item.value,
                    "media_type": item.media_type,
                    "format_version": item.format_version,
                }
                for item in candidate.representations
            ),
            key=lambda item: (item["kind"], item["value"]),
        ),
        "scientific_attributes": {
            key: candidate.attributes[key]
            for key in (
                "coordinates",
                "coordinate_labels",
                "coordinate_unit",
                "lattice",
                "lattice_unit",
                "element_distribution",
                "cell_state",
                "electronic_structure",
            )
            if key in candidate.attributes
        },
    }
    return stable_hash(payload)


def compare_workspace_snapshots(
    off: FusionWorkspaceSnapshot,
    on: FusionWorkspaceSnapshot,
    goal: DiscoveryGoal,
    *,
    artifact_store: ArtifactStore | None = None,
) -> WorkspaceComparisonReport:
    try:
        off = FusionWorkspaceSnapshot.model_validate_json(
            off.model_dump_json(),
            strict=True,
        )
        on = FusionWorkspaceSnapshot.model_validate_json(
            on.model_dump_json(),
            strict=True,
        )
        goal = DiscoveryGoal.model_validate_json(goal.model_dump_json(), strict=True)
    except Exception as exc:
        raise ValueError(f"workspace comparison input contract is invalid: {exc}") from exc
    if off.mode != WorkspaceMode.OFF or on.mode != WorkspaceMode.ON:
        raise ValueError("comparison requires an OFF snapshot followed by an ON snapshot")
    goal_hash = stable_hash(goal)
    if off.goal_hash != goal_hash or on.goal_hash != goal_hash:
        raise ValueError("workspace snapshots do not belong to the supplied goal")
    for label, snapshot in (("OFF", off), ("ON", on)):
        candidate_ref = snapshot.candidate.candidate_ref
        if candidate_ref is None or candidate_content_hash(snapshot.candidate) != candidate_ref.content_hash:
            raise ValueError(f"{label} snapshot contains a stale candidate reference")

    caveats = [
        "Workspace deltas are diagnostics, not scientific evidence or causal proof.",
        "OFF and ON candidates still require the same independent validation gates.",
    ]
    artifacts_verified = artifact_store is not None
    if artifact_store is not None:
        _verify_snapshot_artifacts(off, artifact_store)
        _verify_snapshot_artifacts(on, artifact_store)
    else:
        caveats.append(
            "Feature and latent artifacts were not re-read; paired_configuration is fail-closed."
        )
    paired = artifacts_verified and _paired_run_config(off, on)
    if not paired:
        caveats.append(
            "Goal, parent, seed, generator/evaluator provenance, context, or paired budget differs."
        )

    off_distribution = _element_distribution(off.candidate)
    on_distribution = _element_distribution(on.candidate)
    total_variation: float | None = None
    js_divergence: float | None = None
    if off_distribution is not None and on_distribution is not None:
        total_variation = _total_variation(off_distribution, on_distribution)
        js_divergence = _jensen_shannon(off_distribution, on_distribution)
    else:
        caveats.append("Element-distribution metrics require parseable formula data in both arms.")

    coordinate_rmsd = _ordered_coordinate_rms(off.candidate, on.candidate)
    if coordinate_rmsd is None:
        caveats.append(
            "3D displacement was not computed unless coordinate labels and ordered arrays matched exactly."
        )
    else:
        caveats.append(
            "3D displacement uses an ordered coordinate RMS without rotational or symmetry alignment."
        )

    lattice_distance = _lattice_distance(off.candidate, on.candidate)
    if lattice_distance is None:
        caveats.append("Lattice distance requires two numeric 3x3 lattice matrices.")

    off_objective_properties, off_scoped = _primary_diagnostic_properties(off)
    on_objective_properties, on_scoped = _primary_diagnostic_properties(on)
    if off_scoped or on_scoped:
        caveats.append(
            "Objective deltas use primary-entity properties only; context properties are not pooled."
        )
    objective_deltas = _objective_deltas(
        goal,
        off_objective_properties,
        on_objective_properties,
    )
    changed_kinds = _changed_representation_kinds(off.candidate, on.candidate)

    pair_hash = stable_hash(
        {
            "off": off.candidate.candidate_ref,
            "on": on.candidate.candidate_ref,
            "off_config": off.run_config,
            "on_config": on.run_config,
            "goal_hash": goal_hash,
            "metric_implementation_version": off.run_config.metric_implementation_version,
        }
    )
    return WorkspaceComparisonReport(
        comparison_pair_id=f"WPAIR-{pair_hash[:24]}",
        off_candidate_ref=off.candidate.candidate_ref,
        on_candidate_ref=on.candidate.candidate_ref,
        off_scientific_identity_hash=scientific_identity_hash(off.candidate),
        on_scientific_identity_hash=scientific_identity_hash(on.candidate),
        paired_configuration=paired,
        element_total_variation=total_variation,
        element_jensen_shannon_divergence=js_divergence,
        ordered_coordinate_rms_displacement=coordinate_rmsd,
        lattice_frobenius_distance=lattice_distance,
        molecular_structure_changed=_structure_changed(off.candidate, on.candidate),
        protein_sequence_normalized_edit_distance=_sequence_distance(
            off.candidate,
            on.candidate,
            {RepresentationKind.PROTEIN_SEQUENCE},
        ),
        rna_sequence_normalized_edit_distance=_sequence_distance(
            off.candidate,
            on.candidate,
            {RepresentationKind.RNA_SEQUENCE},
        ),
        representation_kinds_changed=changed_kinds,
        objective_deltas=objective_deltas,
        caveats=caveats,
    )


def _verify_snapshot_artifacts(
    snapshot: FusionWorkspaceSnapshot,
    artifact_store: ArtifactStore,
) -> None:
    property_groups: list[list[DiagnosticProperty]] = []
    for feature_ref in snapshot.feature_refs:
        encoded = artifact_store.read_bytes(
            feature_ref.artifact.relative_path,
            expected_sha256=feature_ref.artifact.sha256,
        )
        if len(encoded) != feature_ref.artifact.byte_size:
            raise ValueError("feature artifact byte_size does not match its reference")
        if (
            feature_ref.artifact.artifact_id
            != f"ART-{feature_ref.artifact.sha256[:24]}"
            or feature_ref.artifact.media_type
            != "application/vnd.discovery-os.expert-feature+json"
        ):
            raise ValueError("feature artifact metadata does not match its reference")
        payload = ExpertFeaturePayload.model_validate_json(encoded, strict=True)
        digest_prefix = feature_ref.feature_id.removeprefix("FEAT-")
        if (
            not stable_hash(payload).startswith(digest_prefix)
            or payload.workspace_entity_id != feature_ref.workspace_entity_id
            or payload.candidate_ref != feature_ref.candidate_ref
            or payload.expert_id != feature_ref.expert_id
            or payload.modality != feature_ref.modality
            or payload.feature_space != feature_ref.feature_space
            or payload.status != feature_ref.status
            or (payload.tensor.dtype if payload.tensor is not None else None)
            != feature_ref.tensor_dtype
            or (list(payload.tensor.shape) if payload.tensor is not None else [])
            != feature_ref.tensor_shape
            or payload.semantics != feature_ref.semantics
            or payload.properties != feature_ref.properties
            or payload.quality_flags != feature_ref.quality_flags
            or payload.warnings != feature_ref.warnings
            or payload.provenance != feature_ref.provenance
        ):
            raise ValueError("feature artifact does not match snapshot feature reference")
        if payload.status != FeatureStatus.FAILED:
            property_groups.append(payload.properties)
    recomputed, _warnings = aggregate_diagnostic_properties(property_groups)
    expected = sorted(
        (item.model_dump(mode="json") for item in snapshot.aggregate_properties),
        key=lambda item: item["property_name"],
    )
    actual = sorted(
        (item.model_dump(mode="json") for item in recomputed),
        key=lambda item: item["property_name"],
    )
    if actual != expected:
        raise ValueError("snapshot aggregate properties do not match feature artifacts")

    if snapshot.latent_state is not None:
        state = snapshot.latent_state
        encoded = artifact_store.read_bytes(
            state.latent_artifact.relative_path,
            expected_sha256=state.latent_artifact.sha256,
        )
        if len(encoded) != state.latent_artifact.byte_size:
            raise ValueError("latent artifact byte_size does not match its reference")
        if (
            state.latent_artifact.artifact_id
            != f"ART-{state.latent_artifact.sha256[:24]}"
            or state.latent_artifact.media_type
            != "application/vnd.discovery-os.latent+json"
        ):
            raise ValueError("latent artifact metadata does not match its reference")
        latent = NumericTensor.model_validate_json(encoded, strict=True)
        if latent.dtype != state.dtype or latent.shape != state.shape:
            raise ValueError("latent artifact does not match snapshot latent state")


def aggregate_diagnostic_properties(
    property_groups: Iterable[list[DiagnosticProperty]],
) -> tuple[list[DiagnosticProperty], list[str]]:
    grouped: dict[str, list[DiagnosticProperty]] = {}
    for properties in property_groups:
        for item in properties:
            grouped.setdefault(item.property_name, []).append(item)

    result: list[DiagnosticProperty] = []
    warnings: list[str] = []
    for name, rows in sorted(grouped.items()):
        units = {item.unit for item in rows}
        if len(units) != 1:
            warnings.append(f"Skipped {name!r}: experts returned incompatible units.")
            continue
        result.append(
            DiagnosticProperty(
                property_name=name,
                value=sum(item.value for item in rows) / len(rows),
                unit=rows[0].unit,
                uncertainty=max(
                    (item.uncertainty for item in rows if item.uncertainty is not None),
                    default=None,
                ),
                out_of_domain=any(item.out_of_domain for item in rows),
                source="diagnostic mean across " + ", ".join(
                    sorted({item.source or "unspecified" for item in rows})
                ),
            )
        )
    return result, warnings


def _primary_diagnostic_properties(
    snapshot: FusionWorkspaceSnapshot,
) -> tuple[list[DiagnosticProperty], bool]:
    """Return objective properties scoped to the primary workspace entity.

    Existing single-entity snapshots retain their stored aggregate so that an
    explicitly supplied diagnostic override remains visible when artifacts are
    not available.  As soon as a context feature contributes a property, the
    unscoped aggregate is unsafe and is recomputed from primary feature refs.
    """

    primary_id = snapshot.workspace.primary_entity_id
    has_context_properties = any(
        item.workspace_entity_id != primary_id
        and item.status != FeatureStatus.FAILED
        and bool(item.properties)
        for item in snapshot.feature_refs
    )
    if not has_context_properties:
        return list(snapshot.aggregate_properties), False
    primary_groups = [
        item.properties
        for item in snapshot.feature_refs
        if item.workspace_entity_id == primary_id and item.status != FeatureStatus.FAILED
    ]
    properties, _warnings = aggregate_diagnostic_properties(primary_groups)
    return properties, True


def _paired_run_config(
    off: FusionWorkspaceSnapshot,
    on: FusionWorkspaceSnapshot,
) -> bool:
    off_payload = off.run_config.model_dump(mode="json", exclude={"workspace_mode"})
    on_payload = on.run_config.model_dump(mode="json", exclude={"workspace_mode"})
    if off_payload != on_payload:
        return False
    if off.goal_hash != on.goal_hash or off.workspace.workspace_id != on.workspace.workspace_id:
        return False
    if _canonical_workspace_context(off) != _canonical_workspace_context(on):
        return False
    if sorted(off.missing_expert_ids) != sorted(on.missing_expert_ids):
        return False
    if sorted(off.failed_expert_ids) != sorted(on.failed_expert_ids):
        return False
    return _feature_panel_provenance(off) == _feature_panel_provenance(on)


def _canonical_workspace_context(snapshot: FusionWorkspaceSnapshot) -> dict[str, Any]:
    primary_id = snapshot.workspace.primary_entity_id
    entities = sorted(
        (
            item.entity_id,
            str(item.role),
            item.candidate_ref.model_dump(mode="json"),
        )
        for item in snapshot.workspace.entities
        if item.entity_id != primary_id
    )
    relations = []
    for relation in snapshot.workspace.relations:
        row = relation.model_dump(mode="json")
        if row["subject_entity_id"] == primary_id:
            row["subject_entity_id"] = "__primary__"
        if row["object_entity_id"] == primary_id:
            row["object_entity_id"] = "__primary__"
        relations.append(row)
    return {"entities": entities, "relations": sorted(relations, key=stable_hash)}


def _feature_panel_provenance(snapshot: FusionWorkspaceSnapshot) -> list[tuple[str, ...]]:
    return sorted(
        (
            item.expert_id,
            item.workspace_entity_id,
            item.provenance.adapter_version,
            item.provenance.model_version,
            item.provenance.code_revision,
            item.provenance.weight_revision,
            item.provenance.parameters_hash,
            item.provenance.projection_version or "",
            item.provenance.dataset_revision or "",
            item.provenance.device or "",
            str(item.modality),
            item.feature_space,
            str(item.tensor_dtype or ""),
            _tensor_shape_contract(item.tensor_shape, item.semantics),
            _feature_semantics_contract(item.semantics),
            str(item.status),
        )
        for item in snapshot.feature_refs
    )


def _tensor_shape_contract(shape: list[int], semantics: Any) -> str:
    """Hash fixed tensor dimensions without candidate-specific entity axes."""

    role = str(semantics.tensor_role) if semantics is not None else ""
    dynamic_axes = 0
    if shape and (
        role in _DYNAMIC_FIRST_AXIS_ROLES
        or (semantics is not None and bool(semantics.entity_ids))
    ):
        dynamic_axes = 1
    if role == "hamiltonian" and len(shape) >= 2:
        # Hamiltonians commonly have two basis/entity axes.  Their rank and
        # trailing channel dimensions remain part of the evaluator contract.
        dynamic_axes = 2
    return stable_hash(
        {
            "rank": len(shape),
            "dynamic_axes": dynamic_axes,
            "fixed_dimensions": list(shape[dynamic_axes:]),
        }
    )


def _feature_semantics_contract(semantics: Any) -> str:
    if semantics is None:
        return ""
    return stable_hash(
        {
            "tensor_role": str(semantics.tensor_role),
            "projection_id": semantics.projection_id,
            "entity_type": semantics.entity_type,
            "pooling": semantics.pooling,
            "normalization": semantics.normalization,
            "coordinate_frame": semantics.coordinate_frame,
            "basis": semantics.basis,
            "unit_semantics": semantics.unit_semantics,
        }
    )


def _representations(candidate: Candidate) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for item in candidate.representations:
        values.setdefault(str(item.kind), []).append(item.value.strip())
    return {key: sorted(rows) for key, rows in values.items()}


def _changed_representation_kinds(
    off: Candidate,
    on: Candidate,
) -> list[RepresentationKind]:
    left = _representations(off)
    right = _representations(on)
    changed = [kind for kind in set(left) | set(right) if left.get(kind) != right.get(kind)]
    return [RepresentationKind(kind) for kind in sorted(changed)]


def _element_distribution(candidate: Candidate) -> dict[str, float] | None:
    formulas = [
        item.value
        for item in candidate.representations
        if item.kind == RepresentationKind.CHEMICAL_FORMULA
    ]
    if formulas:
        try:
            composition = parse_formula(formulas[0])
        except FormulaError:
            return None
        return _normalize_distribution(composition)

    raw = candidate.attributes.get("element_distribution")
    if not isinstance(raw, dict):
        return None
    composition: dict[str, float] = {}
    for element, count in raw.items():
        if not isinstance(element, str) or isinstance(count, bool) or not isinstance(count, (int, float)):
            return None
        if not math.isfinite(float(count)) or count < 0:
            return None
        composition[element] = float(count)
    return _normalize_distribution(composition)


def _normalize_distribution(composition: dict[str, float]) -> dict[str, float] | None:
    total = sum(composition.values())
    if total <= 0:
        return None
    return {element: count / total for element, count in composition.items()}


def _total_variation(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in keys)


def _jensen_shannon(left: dict[str, float], right: dict[str, float]) -> float:
    keys = set(left) | set(right)
    midpoint = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in keys}

    def divergence(source: dict[str, float]) -> float:
        total = 0.0
        for key in keys:
            value = source.get(key, 0.0)
            if value > 0:
                total += value * math.log2(value / midpoint[key])
        return total

    return 0.5 * (divergence(left) + divergence(right))


def _numeric_matrix(value: Any, *, rows: int | None = None, columns: int = 3) -> list[list[float]] | None:
    if not isinstance(value, list) or (rows is not None and len(value) != rows):
        return None
    output: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != columns:
            return None
        converted: list[float] = []
        for item in row:
            if isinstance(item, bool) or not isinstance(item, (int, float)):
                return None
            number = float(item)
            if not math.isfinite(number):
                return None
            converted.append(number)
        output.append(converted)
    return output


def _ordered_coordinate_rms(left: Candidate, right: Candidate) -> float | None:
    left_coordinates = _numeric_matrix(left.attributes.get("coordinates"))
    right_coordinates = _numeric_matrix(right.attributes.get("coordinates"))
    left_labels = left.attributes.get("coordinate_labels")
    right_labels = right.attributes.get("coordinate_labels")
    scales = _length_scales(left, right, "coordinate_unit")
    if (
        left_coordinates is None
        or right_coordinates is None
        or scales is None
        or not left_coordinates
        or len(left_coordinates) != len(right_coordinates)
        or not isinstance(left_labels, list)
        or left_labels != right_labels
        or len(left_labels) != len(left_coordinates)
    ):
        return None
    squared = sum(
        (a * scales[0] - b * scales[1]) ** 2
        for left_row, right_row in zip(left_coordinates, right_coordinates, strict=True)
        for a, b in zip(left_row, right_row, strict=True)
    )
    return math.sqrt(squared / len(left_coordinates))


def _lattice_distance(left: Candidate, right: Candidate) -> float | None:
    left_lattice = _numeric_matrix(left.attributes.get("lattice"), rows=3)
    right_lattice = _numeric_matrix(right.attributes.get("lattice"), rows=3)
    scales = _length_scales(left, right, "lattice_unit")
    if left_lattice is None or right_lattice is None or scales is None:
        return None
    return math.sqrt(
        sum(
            (a * scales[0] - b * scales[1]) ** 2
            for left_row, right_row in zip(left_lattice, right_lattice, strict=True)
            for a, b in zip(left_row, right_row, strict=True)
        )
    )


def _length_scales(
    left: Candidate,
    right: Candidate,
    attribute_name: str,
) -> tuple[float, float] | None:
    left_unit = left.attributes.get(attribute_name)
    right_unit = right.attributes.get(attribute_name)
    # Preserve legacy fixtures only when both arms omit units.  A one-sided or
    # unknown declaration is never silently treated as a common scale.
    if left_unit is None and right_unit is None:
        return 1.0, 1.0
    if not isinstance(left_unit, str) or not isinstance(right_unit, str):
        return None
    left_scale = _length_unit_to_angstrom(left_unit)
    right_scale = _length_unit_to_angstrom(right_unit)
    if left_scale is None or right_scale is None:
        return None
    return left_scale, right_scale


def _length_unit_to_angstrom(unit: str) -> float | None:
    normalized = (
        unit.strip()
        .casefold()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )
    return _LENGTH_UNIT_TO_ANGSTROM.get(normalized)


def _structure_changed(left: Candidate, right: Candidate) -> bool | None:
    left_key = _canonical_molecule_key(left)
    right_key = _canonical_molecule_key(right)
    if left_key is None or right_key is None:
        return None
    return left_key != right_key


def _canonical_molecule_key(candidate: Candidate) -> str | None:
    representations = [
        item
        for item in candidate.representations
        if item.kind
        in {RepresentationKind.SMILES, RepresentationKind.SELFIES, RepresentationKind.INCHI}
    ]
    if not representations:
        return None
    try:
        from rdkit import Chem, rdBase
    except ImportError:
        return None

    canonical: set[str] = set()
    with rdBase.BlockLogs():
        for representation in representations:
            value = representation.value.strip()
            try:
                if representation.kind == RepresentationKind.SMILES:
                    molecule = Chem.MolFromSmiles(value)
                elif representation.kind == RepresentationKind.INCHI:
                    molecule = Chem.MolFromInchi(value)
                else:
                    try:
                        import selfies
                    except ImportError:
                        return None
                    molecule = Chem.MolFromSmiles(selfies.decoder(value))
            except Exception:
                return None
            if molecule is None:
                return None
            canonical.add(
                Chem.MolToSmiles(
                    molecule,
                    canonical=True,
                    isomericSmiles=True,
                )
            )
    if len(canonical) != 1:
        # Conflicting representations on one candidate are not comparable.
        return None
    return next(iter(canonical))


def _sequence_distance(
    left: Candidate,
    right: Candidate,
    kinds: set[RepresentationKind],
) -> float | None:
    sequence_kind = next(iter(kinds), None) if len(kinds) == 1 else None
    if sequence_kind == RepresentationKind.PROTEIN_SEQUENCE:
        expected_type = CandidateType.PROTEIN
        sequence_label = "protein"
        alphabet = _PROTEIN_ALPHABET
    elif sequence_kind == RepresentationKind.RNA_SEQUENCE:
        expected_type = CandidateType.RNA
        sequence_label = "rna"
        alphabet = _RNA_ALPHABET
    else:
        return None
    left_sequence = _candidate_sequence(
        left,
        sequence_kind=sequence_kind,
        expected_type=expected_type,
        sequence_label=sequence_label,
        alphabet=alphabet,
    )
    right_sequence = _candidate_sequence(
        right,
        sequence_kind=sequence_kind,
        expected_type=expected_type,
        sequence_label=sequence_label,
        alphabet=alphabet,
    )
    if left_sequence is None or right_sequence is None:
        return None
    length = max(len(left_sequence), len(right_sequence))
    if length == 0:
        return 0.0
    return _levenshtein(left_sequence, right_sequence) / length


def _candidate_sequence(
    candidate: Candidate,
    *,
    sequence_kind: RepresentationKind,
    expected_type: CandidateType,
    sequence_label: str,
    alphabet: frozenset[str],
) -> str | None:
    explicit = [
        _canonical_sequence(item.value, alphabet)
        for item in candidate.representations
        if item.kind == sequence_kind
    ]
    if explicit:
        if any(item is None for item in explicit):
            return None
        unique = {item for item in explicit if item is not None}
        return next(iter(unique)) if len(unique) == 1 else None

    fasta_values: list[str] = []
    for representation in candidate.representations:
        if representation.kind != RepresentationKind.FASTA:
            continue
        declared = _declared_sequence_type(representation.metadata)
        if candidate.candidate_type == expected_type:
            if declared is not None and declared != sequence_label:
                return None
        elif candidate.candidate_type == CandidateType.BIOLOGIC:
            if declared != sequence_label:
                continue
        else:
            continue
        parsed = _parse_single_fasta(representation.value, alphabet)
        if parsed is None:
            return None
        fasta_values.append(parsed)
    unique_fasta = set(fasta_values)
    return next(iter(unique_fasta)) if len(unique_fasta) == 1 else None


def _declared_sequence_type(metadata: dict[str, Any]) -> str | None:
    for key in ("sequence_type", "polymer_type", "biopolymer_type"):
        value = metadata.get(key)
        if isinstance(value, str):
            normalized = value.strip().casefold()
            if normalized in {"protein", "peptide", "amino_acid"}:
                return "protein"
            if normalized in {"rna", "ribonucleic_acid"}:
                return "rna"
    return None


def _parse_single_fasta(value: str, alphabet: frozenset[str]) -> str | None:
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if len(lines) < 2 or not lines[0].startswith(">"):
        return None
    if any(line.startswith(">") for line in lines[1:]):
        return None
    return _canonical_sequence("".join(lines[1:]), alphabet)


def _canonical_sequence(value: str, alphabet: frozenset[str]) -> str | None:
    sequence = "".join(value.split()).upper()
    if not sequence or any(symbol not in alphabet for symbol in sequence):
        return None
    return sequence


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for index, left_value in enumerate(left, start=1):
        current = [index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _objective_deltas(
    goal: DiscoveryGoal,
    off_properties: list[DiagnosticProperty],
    on_properties: list[DiagnosticProperty],
) -> list[ObjectiveDelta]:
    off = {item.property_name: item for item in off_properties}
    on = {item.property_name: item for item in on_properties}
    output: list[ObjectiveDelta] = []
    for objective in goal.objectives:
        left = off.get(objective.property_name)
        right = on.get(objective.property_name)
        if left is None or right is None:
            output.append(
                ObjectiveDelta(
                    property_name=objective.property_name,
                    direction=str(objective.direction),
                    unit=objective.unit,
                    comparable=False,
                    caveat="The property was not returned in both workspace arms.",
                )
            )
            continue
        if left.unit != right.unit or (objective.unit is not None and left.unit != objective.unit):
            output.append(
                ObjectiveDelta(
                    property_name=objective.property_name,
                    direction=str(objective.direction),
                    unit=objective.unit,
                    off_value=left.value,
                    on_value=right.value,
                    off_uncertainty=left.uncertainty,
                    on_uncertainty=right.uncertainty,
                    comparable=False,
                    caveat="Property units differ between the goal or workspace arms.",
                )
            )
            continue
        if left.out_of_domain or right.out_of_domain:
            output.append(
                ObjectiveDelta(
                    property_name=objective.property_name,
                    direction=str(objective.direction),
                    unit=objective.unit or left.unit,
                    off_value=left.value,
                    on_value=right.value,
                    off_uncertainty=left.uncertainty,
                    on_uncertainty=right.uncertainty,
                    out_of_domain=True,
                    comparable=False,
                    caveat="At least one diagnostic value is outside its model domain.",
                )
            )
            continue
        raw_delta = right.value - left.value
        improvement = _signed_improvement(objective, left.value, right.value)
        output.append(
            ObjectiveDelta(
                property_name=objective.property_name,
                direction=str(objective.direction),
                unit=objective.unit or left.unit,
                off_value=left.value,
                on_value=right.value,
                raw_delta=raw_delta,
                signed_improvement=improvement,
                off_uncertainty=left.uncertainty,
                on_uncertainty=right.uncertainty,
                out_of_domain=False,
                comparable=improvement is not None,
                caveat=(
                    None
                    if improvement is not None
                    else "This objective has no deterministic numeric direction metric."
                ),
            )
        )
    return output


def _signed_improvement(objective: Any, off: float, on: float) -> float | None:
    if objective.direction == ObjectiveDirection.MAXIMIZE:
        return on - off
    if objective.direction == ObjectiveDirection.MINIMIZE:
        return off - on
    if objective.direction == ObjectiveDirection.TARGET:
        if isinstance(objective.target_value, bool) or not isinstance(
            objective.target_value,
            (int, float),
        ):
            return None
        target = float(objective.target_value)
        return abs(off - target) - abs(on - target)
    if objective.direction == ObjectiveDirection.RANGE:
        lower = objective.lower_bound
        upper = objective.upper_bound
        if lower is None or upper is None:
            return None

        def distance(value: float) -> float:
            if value < lower:
                return lower - value
            if value > upper:
                return value - upper
            return 0.0

        return distance(off) - distance(on)
    return None


__all__ = [
    "aggregate_diagnostic_properties",
    "compare_workspace_snapshots",
    "scientific_identity_hash",
]
