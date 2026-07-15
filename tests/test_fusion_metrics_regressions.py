from __future__ import annotations

from types import SimpleNamespace

import pytest

from discovery_os.fusion_metrics import (
    _feature_panel_provenance,
    _lattice_distance,
    _objective_deltas,
    _ordered_coordinate_rms,
    _primary_diagnostic_properties,
    _sequence_distance,
    _structure_changed,
)
from discovery_os.fusion_schemas import (
    DiagnosticProperty,
    ExpertProvenance,
    FeatureSemantics,
    FeatureStatus,
    TensorRole,
)
from discovery_os.hashing import stable_hash
from discovery_os.schemas import (
    Candidate,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    ObjectiveDirection,
    PropertyObjective,
    RepresentationKind,
)


def _candidate(
    candidate_id: str,
    *,
    candidate_type: CandidateType = CandidateType.CUSTOM,
    representations: list[CandidateRepresentation] | None = None,
    attributes: dict | None = None,
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        candidate_type=candidate_type,
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        representations=representations
        or [CandidateRepresentation(kind=RepresentationKind.CUSTOM, value=candidate_id)],
        attributes=attributes or {},
    )


def _fasta(value: str, *, sequence_type: str | None = None) -> CandidateRepresentation:
    metadata = {"sequence_type": sequence_type} if sequence_type is not None else {}
    return CandidateRepresentation(
        kind=RepresentationKind.FASTA,
        value=value,
        metadata=metadata,
    )


def test_fasta_distances_are_type_scoped_and_headers_are_removed() -> None:
    protein_left = _candidate(
        "protein-left",
        candidate_type=CandidateType.PROTEIN,
        representations=[_fasta(">left\nACD E")],
    )
    protein_right = _candidate(
        "protein-right",
        candidate_type=CandidateType.PROTEIN,
        representations=[_fasta(">right\nACDF")],
    )
    assert _sequence_distance(
        protein_left,
        protein_right,
        {RepresentationKind.PROTEIN_SEQUENCE},
    ) == pytest.approx(0.25)
    assert (
        _sequence_distance(
            protein_left,
            protein_right,
            {RepresentationKind.RNA_SEQUENCE},
        )
        is None
    )

    rna_left = _candidate(
        "rna-left",
        candidate_type=CandidateType.BIOLOGIC,
        representations=[_fasta(">left\nACGU", sequence_type="rna")],
    )
    rna_right = _candidate(
        "rna-right",
        candidate_type=CandidateType.BIOLOGIC,
        representations=[_fasta(">right\nACGA", sequence_type="rna")],
    )
    assert _sequence_distance(
        rna_left,
        rna_right,
        {RepresentationKind.RNA_SEQUENCE},
    ) == pytest.approx(0.25)
    assert (
        _sequence_distance(
            rna_left,
            rna_right,
            {RepresentationKind.PROTEIN_SEQUENCE},
        )
        is None
    )


def test_multi_record_or_invalid_fasta_fails_closed() -> None:
    multi = _candidate(
        "multi",
        candidate_type=CandidateType.PROTEIN,
        representations=[_fasta(">one\nACD\n>two\nEFG")],
    )
    valid = _candidate(
        "valid",
        candidate_type=CandidateType.PROTEIN,
        representations=[_fasta(">valid\nACD")],
    )
    assert (
        _sequence_distance(multi, valid, {RepresentationKind.PROTEIN_SEQUENCE})
        is None
    )


def test_coordinate_and_lattice_metrics_convert_known_length_units() -> None:
    left = _candidate(
        "left",
        attributes={
            "coordinates": [[1.0, 0.0, 0.0]],
            "coordinate_labels": ["C1"],
            "coordinate_unit": "angstrom",
            "lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "lattice_unit": "angstrom",
        },
    )
    right = _candidate(
        "right",
        attributes={
            "coordinates": [[0.1, 0.0, 0.0]],
            "coordinate_labels": ["C1"],
            "coordinate_unit": "nm",
            "lattice": [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]],
            "lattice_unit": "nm",
        },
    )
    assert _ordered_coordinate_rms(left, right) == pytest.approx(0.0)
    assert _lattice_distance(left, right) == pytest.approx(0.0)


def test_coordinate_and_lattice_metrics_reject_unknown_or_one_sided_units() -> None:
    left = _candidate(
        "left",
        attributes={
            "coordinates": [[0.0, 0.0, 0.0]],
            "coordinate_labels": ["C1"],
            "coordinate_unit": "furlong",
            "lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "lattice_unit": "angstrom",
        },
    )
    right = _candidate(
        "right",
        attributes={
            "coordinates": [[0.0, 0.0, 0.0]],
            "coordinate_labels": ["C1"],
            "coordinate_unit": "furlong",
            "lattice": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
    )
    assert _ordered_coordinate_rms(left, right) is None
    assert _lattice_distance(left, right) is None


def test_molecular_structure_uses_canonical_graph_identity() -> None:
    pytest.importorskip("rdkit")
    ethanol_a = _candidate(
        "ethanol-a",
        candidate_type=CandidateType.SMALL_MOLECULE,
        representations=[CandidateRepresentation(kind=RepresentationKind.SMILES, value="CCO")],
    )
    ethanol_b = _candidate(
        "ethanol-b",
        candidate_type=CandidateType.SMALL_MOLECULE,
        representations=[CandidateRepresentation(kind=RepresentationKind.SMILES, value="OCC")],
    )
    ethylamine = _candidate(
        "ethylamine",
        candidate_type=CandidateType.SMALL_MOLECULE,
        representations=[CandidateRepresentation(kind=RepresentationKind.SMILES, value="CCN")],
    )
    invalid = _candidate(
        "invalid",
        candidate_type=CandidateType.SMALL_MOLECULE,
        representations=[CandidateRepresentation(kind=RepresentationKind.SMILES, value="C(")],
    )

    assert _structure_changed(ethanol_a, ethanol_b) is False
    assert _structure_changed(ethanol_a, ethylamine) is True
    assert _structure_changed(ethanol_a, invalid) is None


def _property(value: float, *, source: str) -> DiagnosticProperty:
    return DiagnosticProperty(
        property_name="score",
        value=value,
        unit="arb",
        uncertainty=0.1,
        source=source,
    )


def _property_snapshot(primary_value: float, context_value: float) -> SimpleNamespace:
    primary = _property(primary_value, source="primary-expert")
    context = _property(context_value, source="context-expert")
    mixed = _property((primary_value + context_value) / 2.0, source="unsafe-mixed")
    return SimpleNamespace(
        workspace=SimpleNamespace(primary_entity_id="primary"),
        aggregate_properties=[mixed],
        feature_refs=[
            SimpleNamespace(
                workspace_entity_id="primary",
                status=FeatureStatus.SUCCESS,
                properties=[primary],
            ),
            SimpleNamespace(
                workspace_entity_id="target",
                status=FeatureStatus.SUCCESS,
                properties=[context],
            ),
        ],
    )


def test_objective_properties_are_scoped_to_primary_entity() -> None:
    off_properties, off_scoped = _primary_diagnostic_properties(
        _property_snapshot(1.0, 100.0)
    )
    on_properties, on_scoped = _primary_diagnostic_properties(
        _property_snapshot(3.0, -100.0)
    )
    goal = DiscoveryGoal(
        goal_id="goal",
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        title="Primary property",
        scientific_question="Do not mix target properties into the primary objective.",
        objectives=[
            PropertyObjective(
                property_name="score",
                direction=ObjectiveDirection.MAXIMIZE,
                unit="arb",
            )
        ],
        validation_profile_id="general-materials-v1",
        candidate_types=[CandidateType.CUSTOM],
    )
    delta = _objective_deltas(goal, off_properties, on_properties)[0]

    assert off_scoped is True
    assert on_scoped is True
    assert off_properties[0].value == pytest.approx(1.0)
    assert on_properties[0].value == pytest.approx(3.0)
    assert delta.signed_improvement == pytest.approx(2.0)


def _panel_ref(
    *,
    entity_ids: list[str],
    shape: list[int],
    normalization: str = "layer-norm-v1",
) -> SimpleNamespace:
    return SimpleNamespace(
        expert_id="atom-expert",
        workspace_entity_id="primary",
        provenance=ExpertProvenance(
            expert_id="atom-expert",
            adapter_version="1.0.0",
            model_version="1.0.0",
            code_revision="code-v1",
            weight_revision="weight-v1",
            parameters_hash=stable_hash({"cutoff": 5.0}),
            projection_version="projection-v1",
            seed=7,
        ),
        modality="molecule_3d",
        feature_space="atom-space-v1",
        tensor_dtype="float32",
        tensor_shape=shape,
        semantics=FeatureSemantics(
            tensor_role=TensorRole.ATOM_EMBEDDING,
            projection_id="projection-v1",
            entity_type="atom",
            entity_ids=entity_ids,
            mask=[True] * len(entity_ids),
            pooling="none",
            normalization=normalization,
            coordinate_frame="cartesian",
            unit_semantics={"coordinates": "angstrom"},
        ),
        status=FeatureStatus.SUCCESS,
    )


def test_pairing_ignores_dynamic_entity_axis_but_keeps_fixed_contract() -> None:
    two_atoms = SimpleNamespace(
        feature_refs=[_panel_ref(entity_ids=["a", "b"], shape=[2, 4])]
    )
    three_atoms = SimpleNamespace(
        feature_refs=[_panel_ref(entity_ids=["x", "y", "z"], shape=[3, 4])]
    )
    wrong_width = SimpleNamespace(
        feature_refs=[_panel_ref(entity_ids=["x", "y", "z"], shape=[3, 5])]
    )
    wrong_normalization = SimpleNamespace(
        feature_refs=[
            _panel_ref(
                entity_ids=["x", "y", "z"],
                shape=[3, 4],
                normalization="different-normalization",
            )
        ]
    )

    baseline = _feature_panel_provenance(two_atoms)
    assert baseline == _feature_panel_provenance(three_atoms)
    assert baseline != _feature_panel_provenance(wrong_width)
    assert baseline != _feature_panel_provenance(wrong_normalization)
