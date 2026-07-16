from __future__ import annotations

from discovery_os.materials_screening import (
    CandidateScreeningVector,
    MLIPScreeningPrediction,
    classify_model_disagreement,
    force_rmse,
    rank_composition_scoped_pareto,
    select_dft_handoff_refs,
    stress_norm_gpa,
)
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
        disagreement=classify_model_disagreement(
            mattersim,
            chgnet,
            force_rmse_eV_A=force_rmse_value,
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
        "cross_model_stress_comparison_unavailable",
        "relaxed_structure_comparison_unavailable",
    ]
