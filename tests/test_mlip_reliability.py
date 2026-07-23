from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from discovery_os.mlip_reliability import (
    CandidateProxyRequest,
    ChemistryExchangeabilityScope,
    CompositionEnergyPair,
    ExpertWeightRevision,
    HeldOutCoverageMetadata,
    SplitConformalCalibrationArtifact,
    SplitConformalMetricCalibration,
    assess_candidate_proxy,
    composition_relative_energy_disagreement,
    finite_sample_split_conformal_quantile,
    finite_sample_split_conformal_rank,
    force_disagreement_metrics,
)


def _expert_revisions(*, chgnet_revision: str = "sha256:chgnet-v1") -> list[ExpertWeightRevision]:
    return [
        ExpertWeightRevision(
            expert_id="mattersim",
            weight_revision="sha256:mattersim-v1",
        ),
        ExpertWeightRevision(
            expert_id="chgnet",
            weight_revision=chgnet_revision,
        ),
    ]


def _calibration() -> SplitConformalCalibrationArtifact:
    return SplitConformalCalibrationArtifact(
        artifact_id="li-o-energy-calibration-v1",
        expert_weight_revisions=_expert_revisions(),
        dft_method="QE 7.4; PBE; SSSP precision v1.3; converged settings v2",
        dft_reference_hash="d" * 64,
        chemistry_scope=ChemistryExchangeabilityScope(
            scope_id="periodic-bulk-li-o-near-equilibrium-v1",
            allowed_elements=["Li", "O"],
            allowed_reduced_compositions=["Li2O", "LiO2"],
            exchangeability_statement=(
                "Periodic bulk Li-O crystals sampled by the same generator and "
                "relaxation policy as the calibration and held-out sets."
            ),
        ),
        alpha=0.2,
        calibration_sample_count=4,
        metrics=[
            SplitConformalMetricCalibration(
                proxy_name="relative_energy",
                unit="eV/atom",
                calibration_scores=[0.3, 0.1, 0.4, 0.2],
                quantile_rank=4,
                quantile=0.4,
            )
        ],
        held_out_coverage=[
            HeldOutCoverageMetadata(
                proxy_name="relative_energy",
                dataset_hash="e" * 64,
                sample_count=20,
                empirical_coverage=0.8,
                mean_interval_width=0.8,
            )
        ],
    )


def _proxy_request(
    *,
    elements: list[str] | None = None,
    chgnet_revision: str = "sha256:chgnet-v1",
) -> CandidateProxyRequest:
    return CandidateProxyRequest(
        candidate_id="candidate-001",
        proxy_name="relative_energy",
        unit="eV/atom",
        proxy_value=0.25,
        scope_id="periodic-bulk-li-o-near-equilibrium-v1",
        elements=elements or ["Li", "O"],
        reduced_composition="Li2O",
        expert_weight_revisions=_expert_revisions(
            chgnet_revision=chgnet_revision
        ),
        dft_method="QE 7.4; PBE; SSSP precision v1.3; converged settings v2",
        dft_reference_hash="d" * 64,
    )


def test_finite_sample_quantile_uses_n_plus_one_ceiling_rank() -> None:
    scores = [0.11, 0.52, 0.19, 0.34, 0.27]

    assert finite_sample_split_conformal_rank(5, 0.2) == 5
    assert finite_sample_split_conformal_quantile(scores, 0.2) == 0.52


def test_finite_quantile_rejects_unattainable_nominal_coverage() -> None:
    with pytest.raises(ValueError, match="too small"):
        finite_sample_split_conformal_quantile([0.1, 0.2, 0.3], 0.01)


def test_calibration_artifact_recomputes_and_validates_quantile() -> None:
    payload = _calibration().model_dump()
    payload["metrics"][0]["quantile"] = 0.3

    with pytest.raises(ValidationError, match="incorrect quantile"):
        SplitConformalCalibrationArtifact.model_validate(payload)


def test_matching_scope_exposes_interval_and_declared_coverage_metadata() -> None:
    result = assess_candidate_proxy(_proxy_request(), _calibration())

    assert result.status == "calibrated_in_scope"
    assert result.exchangeability_scope_match is True
    assert result.interval_lower == pytest.approx(-0.15)
    assert result.interval_upper == pytest.approx(0.65)
    assert result.nominal_coverage == pytest.approx(0.8)
    assert result.held_out_empirical_coverage == pytest.approx(0.8)
    assert result.dft_escalation is False


@pytest.mark.parametrize(
    ("candidate_request", "reason"),
    [
        (_proxy_request(elements=["Li", "O", "Na"]), "chemistry_elements_out_of_scope"),
        (
            _proxy_request(chgnet_revision="sha256:chgnet-different"),
            "expert_weight_revision_mismatch",
        ),
    ],
)
def test_scope_or_weight_mismatch_has_no_probabilistic_claim(
    candidate_request: CandidateProxyRequest,
    reason: str,
) -> None:
    result = assess_candidate_proxy(candidate_request, _calibration())

    assert result.status == "uncalibrated_or_ood"
    assert result.exchangeability_scope_match is False
    assert result.interval_lower is None
    assert result.interval_upper is None
    assert result.alpha is None
    assert result.nominal_coverage is None
    assert result.held_out_empirical_coverage is None
    assert result.dft_escalation is True
    assert reason in result.reasons


def _energy_rows(
    *,
    first_offset: float = 0.0,
    second_offset: float = 0.0,
    alignment: str | None = "paired-structures-v1",
) -> list[CompositionEnergyPair]:
    values = [
        ("a", -10.0, -100.0),
        ("b", -9.8, -99.7),
        ("c", -9.5, -99.6),
    ]
    return [
        CompositionEnergyPair(
            candidate_id=candidate_id,
            reduced_composition="Li2O",
            first_model_id="mattersim",
            second_model_id="chgnet",
            first_energy_per_atom_eV=first_energy + first_offset,
            second_energy_per_atom_eV=second_energy + second_offset,
            alignment_artifact_id=alignment,
        )
        for candidate_id, first_energy, second_energy in values
    ]


def test_composition_relative_energy_disagreement_is_gauge_offset_invariant() -> None:
    baseline = composition_relative_energy_disagreement(_energy_rows())
    shifted = composition_relative_energy_disagreement(
        _energy_rows(first_offset=123.4, second_offset=-71.2)
    )

    assert [row.status for row in baseline] == ["available"] * 3
    for left, right in zip(baseline, shifted, strict=True):
        assert left.first_relative_energy_eV_atom == pytest.approx(
            right.first_relative_energy_eV_atom
        )
        assert left.second_relative_energy_eV_atom == pytest.approx(
            right.second_relative_energy_eV_atom
        )
        assert left.relative_energy_abs_diff_eV_atom == pytest.approx(
            right.relative_energy_abs_diff_eV_atom
        )
        assert left.first_rank == right.first_rank
        assert left.second_rank == right.second_rank
        assert left.rank_abs_diff == right.rank_abs_diff


def test_singleton_composition_is_unknown_and_escalated() -> None:
    result = composition_relative_energy_disagreement(_energy_rows()[:1])[0]

    assert result.status == "unknown"
    assert result.relative_energy_abs_diff_eV_atom is None
    assert result.dft_escalation is True
    assert result.uncertainty_reasons == ["singleton_composition_pool"]


def test_missing_energy_alignment_is_unknown_and_never_low() -> None:
    result = composition_relative_energy_disagreement(
        _energy_rows(alignment=None)
    )

    assert all(row.status == "unknown" for row in result)
    assert all(row.dft_escalation for row in result)
    assert all(
        "energy_pair_alignment_missing" in row.uncertainty_reasons
        for row in result
    )


def test_force_metrics_preserve_a_single_atom_outlier() -> None:
    first = [[0.0, 0.0, 0.0] for _ in range(100)]
    second = [[0.0, 0.0, 0.0] for _ in range(100)]
    first[-1] = [1.0, 0.0, 0.0]

    result = force_disagreement_metrics(first, second)

    assert result.component_rmse_eV_A == pytest.approx(math.sqrt(1.0 / 300.0))
    assert result.per_atom_vector_rms_eV_A == pytest.approx(0.1)
    assert result.max_atom_vector_norm_eV_A == pytest.approx(1.0)
    assert result.max_atom_vector_norm_eV_A > 10 * result.component_rmse_eV_A
