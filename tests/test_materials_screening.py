from __future__ import annotations

import pytest

from discovery_os.materials_screening import (
    CandidateScreeningVector,
    MLIPScreeningPrediction,
    classify_model_disagreement,
    force_rmse,
    rank_composition_scoped_pareto,
    select_dft_handoff_refs,
    stress_norm_gpa,
)
from discovery_os.mlip_reliability import CompositionRelativeEnergyDisagreement
from discovery_os.schemas import CandidateRef


def _vector(
    name: str,
    composition: str,
    *,
    matter_energy: float,
    chgnet_energy: float,
    force: float,
    force_rmse_value: float = 0.02,
    relaxed: bool = True,
) -> CandidateScreeningVector:
    mattersim = MLIPScreeningPrediction(
        expert_id="mattersim",
        energy_per_atom_eV=matter_energy,
        max_force_eV_A=force,
    )
    chgnet = MLIPScreeningPrediction(
        expert_id="chgnet",
        energy_per_atom_eV=chgnet_energy,
        max_force_eV_A=force,
    )
    return CandidateScreeningVector(
        candidate_ref=CandidateRef(
            candidate_id=name,
            version=1,
            content_hash=(name.encode().hex() + "0" * 64)[:64],
        ),
        composition_key=composition,
        mattersim=mattersim,
        chgnet=chgnet,
        common_geometry_mattersim=mattersim,
        common_geometry_chgnet=chgnet,
        common_geometry_alignment_id=f"{name}-common-input-geometry",
        disagreement=classify_model_disagreement(
            mattersim,
            chgnet,
            force_rmse_eV_A=force_rmse_value,
            relative_energy=CompositionRelativeEnergyDisagreement(
                candidate_id=name,
                reduced_composition=composition,
                first_model_id="mattersim",
                second_model_id="chgnet",
                pool_size=2,
                status="available",
                alignment_artifact_id=f"{composition}-relaxed-panel",
                first_relative_energy_eV_atom=0.0,
                second_relative_energy_eV_atom=0.0,
                relative_energy_abs_diff_eV_atom=0.0,
                first_rank=1,
                second_rank=1,
                rank_abs_diff=0,
                dft_escalation=False,
            ),
        ),
        geometry_valid=True,
        relaxation_gate_passed=relaxed,
    )


def test_force_rmse_uses_xyz_columns_only() -> None:
    assert force_rmse([[1.0, 2.0, 3.0]], [[1.0, 2.0, 0.0, 99.0]]) == 3**0.5


def test_pareto_energy_is_never_compared_across_compositions() -> None:
    low_absolute = _vector(
        "a",
        "Li",
        matter_energy=-100.0,
        chgnet_energy=-100.0,
        force=0.1,
    )
    high_absolute = _vector(
        "b",
        "Li-O",
        matter_energy=-1.0,
        chgnet_energy=-1.0,
        force=0.1,
    )

    ranked = rank_composition_scoped_pareto([low_absolute, high_absolute])

    assert [row.composition_pareto_front for row in ranked] == [1, 1]


def test_high_disagreement_is_kept_for_dft_handoff() -> None:
    stable = _vector(
        "stable",
        "Li-O",
        matter_energy=-3.0,
        chgnet_energy=-3.0,
        force=0.01,
    )
    disagreement = _vector(
        "disagreement",
        "Li2-O",
        matter_energy=-2.0,
        chgnet_energy=-1.7,
        force=0.2,
        force_rmse_value=0.3,
        relaxed=False,
    )
    ranked = rank_composition_scoped_pareto([stable, disagreement])

    selected = select_dft_handoff_refs(ranked, [stable, disagreement], top_k=2)

    assert stable.candidate_ref in selected
    assert disagreement.candidate_ref in selected


def test_stress_is_normalized_and_contributes_to_disagreement() -> None:
    mattersim = MLIPScreeningPrediction(
        expert_id="mattersim",
        energy_per_atom_eV=-3.0,
        max_force_eV_A=0.01,
        stress_norm=0.2,
        stress_unit="eV/angstrom^3",
    )
    chgnet = MLIPScreeningPrediction(
        expert_id="chgnet",
        energy_per_atom_eV=-3.0,
        max_force_eV_A=0.01,
        stress_norm=1.0,
        stress_unit="GPa",
    )

    disagreement = classify_model_disagreement(
        mattersim,
        chgnet,
        force_rmse_eV_A=0.01,
    )

    assert stress_norm_gpa(mattersim) == 0.2 * 160.2176634
    assert disagreement.stress_norm_abs_diff_GPa is not None
    assert disagreement.risk == "high"
    assert disagreement.dft_escalation is True


def test_required_missing_comparisons_escalate_as_uncertainty() -> None:
    mattersim = MLIPScreeningPrediction(
        expert_id="mattersim",
        energy_per_atom_eV=-3.0,
        max_force_eV_A=0.01,
    )
    chgnet = MLIPScreeningPrediction(
        expert_id="chgnet",
        energy_per_atom_eV=-3.0,
        max_force_eV_A=0.01,
    )

    disagreement = classify_model_disagreement(
        mattersim,
        chgnet,
        force_rmse_eV_A=0.01,
        require_stress_comparison=True,
        require_relaxed_structure_comparison=True,
    )

    assert disagreement.risk == "high"
    assert disagreement.dft_escalation is True
    assert disagreement.uncertainty_reasons == [
        "composition_relative_energy_comparison_unavailable",
        "cross_model_stress_comparison_unavailable",
        "relaxed_structure_comparison_unavailable",
    ]


def test_raw_cross_model_energy_offset_is_audit_only() -> None:
    mattersim = MLIPScreeningPrediction(
        expert_id="mattersim",
        energy_per_atom_eV=-100.0,
        max_force_eV_A=0.01,
    )
    chgnet = MLIPScreeningPrediction(
        expert_id="chgnet",
        energy_per_atom_eV=25.0,
        max_force_eV_A=0.01,
    )
    relative = CompositionRelativeEnergyDisagreement(
        candidate_id="aligned",
        reduced_composition="Li-O",
        first_model_id="mattersim",
        second_model_id="chgnet",
        pool_size=2,
        status="available",
        alignment_artifact_id="aligned-relaxed-panel",
        first_relative_energy_eV_atom=0.0,
        second_relative_energy_eV_atom=0.0,
        relative_energy_abs_diff_eV_atom=0.0,
        first_rank=1,
        second_rank=1,
        rank_abs_diff=0,
        dft_escalation=False,
    )

    disagreement = classify_model_disagreement(
        mattersim,
        chgnet,
        force_rmse_eV_A=0.01,
        relative_energy=relative,
    )

    assert disagreement.raw_energy_per_atom_abs_diff_eV == 125.0
    assert disagreement.energy_comparison_basis == "composition_relative_aligned"
    assert disagreement.risk == "low"
    assert disagreement.dft_escalation is False


def test_relative_energy_from_another_same_composition_candidate_is_rejected() -> None:
    vector = _vector(
        "candidate-a",
        "Li-O",
        matter_energy=-3.0,
        chgnet_energy=-2.9,
        force=0.01,
    )
    swapped = vector.disagreement.model_copy(
        update={"composition_relative_candidate_id": "candidate-b"}
    )

    with pytest.raises(ValueError, match="another candidate or composition"):
        CandidateScreeningVector(
            candidate_ref=vector.candidate_ref,
            composition_key=vector.composition_key,
            mattersim=vector.mattersim,
            chgnet=vector.chgnet,
            common_geometry_mattersim=vector.common_geometry_mattersim,
            common_geometry_chgnet=vector.common_geometry_chgnet,
            common_geometry_alignment_id=vector.common_geometry_alignment_id,
            disagreement=swapped,
            geometry_valid=True,
            relaxation_gate_passed=True,
        )


def test_missing_composition_relative_energy_fails_closed() -> None:
    mattersim = MLIPScreeningPrediction(
        expert_id="mattersim",
        energy_per_atom_eV=-3.0,
        max_force_eV_A=0.01,
    )
    chgnet = MLIPScreeningPrediction(
        expert_id="chgnet",
        energy_per_atom_eV=-3.0,
        max_force_eV_A=0.01,
    )

    disagreement = classify_model_disagreement(
        mattersim,
        chgnet,
        force_rmse_eV_A=0.01,
    )

    assert disagreement.energy_comparison_basis == "unknown"
    assert disagreement.risk == "high"
    assert disagreement.dft_escalation is True


def test_disagreement_threshold_contract_rejects_inverted_or_negative_values() -> None:
    mattersim = MLIPScreeningPrediction(
        expert_id="mattersim",
        energy_per_atom_eV=-3.0,
        max_force_eV_A=0.01,
    )
    chgnet = MLIPScreeningPrediction(
        expert_id="chgnet",
        energy_per_atom_eV=-3.0,
        max_force_eV_A=0.01,
    )

    with pytest.raises(ValueError, match="high >= medium"):
        classify_model_disagreement(
            mattersim,
            chgnet,
            force_rmse_eV_A=0.01,
            medium_energy_threshold_eV=0.2,
            high_energy_threshold_eV=0.1,
        )
    with pytest.raises(ValueError, match="rank thresholds"):
        classify_model_disagreement(
            mattersim,
            chgnet,
            force_rmse_eV_A=0.01,
            medium_rank_difference=-1,
        )


def test_pareto_crowding_retains_objective_boundaries_before_center() -> None:
    left = _vector(
        "left",
        "Li-O",
        matter_energy=-3.0,
        chgnet_energy=-1.0,
        force=0.05,
    )
    center = _vector(
        "center",
        "Li-O",
        matter_energy=-2.0,
        chgnet_energy=-2.0,
        force=0.05,
    )
    right = _vector(
        "right",
        "Li-O",
        matter_energy=-1.0,
        chgnet_energy=-3.0,
        force=0.05,
    )

    ranked = rank_composition_scoped_pareto([center, right, left])
    by_id = {item.candidate_ref.candidate_id: item for item in ranked}

    assert by_id["left"].composition_pareto_front == 1
    assert by_id["center"].composition_pareto_front == 1
    assert by_id["right"].composition_pareto_front == 1
    assert by_id["left"].pareto_boundary is True
    assert by_id["right"].pareto_boundary is True
    assert by_id["center"].pareto_boundary is False
    assert by_id["left"].global_priority_rank < by_id["center"].global_priority_rank
    assert by_id["right"].global_priority_rank < by_id["center"].global_priority_rank


def test_dft_handoff_shortlist_covers_compositions_before_filling_duplicates() -> None:
    first = _vector(
        "li-o-best",
        "Li-O",
        matter_energy=-3.0,
        chgnet_energy=-3.0,
        force=0.01,
    )
    second = _vector(
        "li-o-next",
        "Li-O",
        matter_energy=-2.9,
        chgnet_energy=-2.9,
        force=0.01,
    )
    different = _vector(
        "na-cl",
        "Na-Cl",
        matter_energy=-1.0,
        chgnet_energy=-1.0,
        force=0.02,
    )
    vectors = [first, second, different]
    ranked = rank_composition_scoped_pareto(vectors)

    selected = select_dft_handoff_refs(ranked, vectors, top_k=2)

    selected_ids = {item.candidate_id for item in selected}
    assert "na-cl" in selected_ids
    assert len(selected_ids & {"li-o-best", "li-o-next"}) == 1
