"""Calibrated reliability primitives for periodic MLIP screening.

The contracts in this module deliberately separate three different claims:

* split-conformal intervals are available only for the exact expert weights,
  DFT reference, and declared exchangeability scope used for calibration;
* cross-model energies are compared only after removing each model's
  composition-local energy offset; and
* force disagreement retains atom-local outliers instead of reducing every
  component to one potentially diluted mean.

None of these helpers turns MLIP agreement into a thermodynamic-stability
claim.  Out-of-scope or incomplete evidence is explicitly unknown and is
escalated to a reference-consistent DFT workflow.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from .schemas import Identifier, NonEmptyText, Probability, StrictSchema


class ExpertWeightRevision(StrictSchema):
    """Exact expert identity used to create or consume a calibration."""

    expert_id: Identifier
    weight_revision: Identifier


class ChemistryExchangeabilityScope(StrictSchema):
    """Declared chemistry population for which conformal coverage is assessed.

    An empty ``allowed_reduced_compositions`` list means any reduced
    composition made solely from ``allowed_elements``.  It does not mean that
    arbitrary material classes or operating conditions are exchangeable; those
    restrictions must be stated in ``exchangeability_statement`` and encoded
    into the versioned ``scope_id`` by the producer.
    """

    scope_id: Identifier
    allowed_elements: list[Identifier] = Field(min_length=1, max_length=118)
    allowed_reduced_compositions: list[Identifier] = Field(
        default_factory=list,
        max_length=4096,
    )
    exchangeability_statement: NonEmptyText

    @model_validator(mode="after")
    def _unique_scope_members(self) -> ChemistryExchangeabilityScope:
        if len(set(self.allowed_elements)) != len(self.allowed_elements):
            raise ValueError("allowed_elements must not contain duplicates")
        if len(set(self.allowed_reduced_compositions)) != len(
            self.allowed_reduced_compositions
        ):
            raise ValueError(
                "allowed_reduced_compositions must not contain duplicates"
            )
        return self


class SplitConformalMetricCalibration(StrictSchema):
    """Absolute-residual calibration scores for one proxy and unit."""

    proxy_name: Identifier
    unit: Identifier
    nonconformity_score: Literal["absolute_residual"] = "absolute_residual"
    calibration_scores: list[float] = Field(min_length=1)
    quantile_rank: int = Field(gt=0)
    quantile: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _non_negative_scores(self) -> SplitConformalMetricCalibration:
        if any(score < 0.0 for score in self.calibration_scores):
            raise ValueError("absolute-residual calibration scores must be non-negative")
        return self


class HeldOutCoverageMetadata(StrictSchema):
    """Independent held-out diagnostics; not a universal coverage guarantee."""

    proxy_name: Identifier
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_count: int = Field(gt=0)
    empirical_coverage: Probability
    mean_interval_width: float = Field(ge=0.0)
    independent_from_calibration: Literal[True] = True
    exchangeability_scope_match_verified: Literal[True] = True


class SplitConformalCalibrationArtifact(StrictSchema):
    """Versioned calibration bound to exact models and DFT reference labels."""

    artifact_version: Literal["split-conformal-v1"] = "split-conformal-v1"
    artifact_id: Identifier
    expert_weight_revisions: list[ExpertWeightRevision] = Field(
        min_length=1,
        max_length=32,
    )
    dft_method: NonEmptyText
    dft_reference_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    chemistry_scope: ChemistryExchangeabilityScope
    alpha: float = Field(gt=0.0, lt=1.0)
    calibration_sample_count: int = Field(gt=0)
    metrics: list[SplitConformalMetricCalibration] = Field(
        min_length=1,
        max_length=128,
    )
    held_out_coverage: list[HeldOutCoverageMetadata] = Field(
        min_length=1,
        max_length=128,
    )

    @model_validator(mode="after")
    def _validate_calibration_contract(self) -> SplitConformalCalibrationArtifact:
        expert_ids = [item.expert_id for item in self.expert_weight_revisions]
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("expert_weight_revisions must have unique expert_id values")

        metric_names = [metric.proxy_name for metric in self.metrics]
        if len(set(metric_names)) != len(metric_names):
            raise ValueError("metrics must have unique proxy_name values")

        coverage_names = [item.proxy_name for item in self.held_out_coverage]
        if len(set(coverage_names)) != len(coverage_names):
            raise ValueError("held_out_coverage must have unique proxy_name values")
        if set(metric_names) != set(coverage_names):
            raise ValueError(
                "held_out_coverage must contain exactly one row for every metric"
            )

        for metric in self.metrics:
            if len(metric.calibration_scores) != self.calibration_sample_count:
                raise ValueError(
                    "every metric must retain calibration_sample_count scores"
                )
            expected_rank = finite_sample_split_conformal_rank(
                self.calibration_sample_count,
                self.alpha,
            )
            expected_quantile = finite_sample_split_conformal_quantile(
                metric.calibration_scores,
                self.alpha,
            )
            if metric.quantile_rank != expected_rank:
                raise ValueError(
                    f"metric {metric.proxy_name!r} has an incorrect quantile_rank"
                )
            if not math.isclose(
                metric.quantile,
                expected_quantile,
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"metric {metric.proxy_name!r} has an incorrect quantile"
                )
        return self


class CandidateProxyRequest(StrictSchema):
    """Candidate proxy and provenance presented for calibrated assessment."""

    candidate_id: Identifier
    proxy_name: Identifier
    unit: Identifier
    proxy_value: float
    scope_id: Identifier
    elements: list[Identifier] = Field(min_length=1, max_length=118)
    reduced_composition: Identifier
    expert_weight_revisions: list[ExpertWeightRevision] = Field(
        min_length=1,
        max_length=32,
    )
    dft_method: NonEmptyText
    dft_reference_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _unique_request_members(self) -> CandidateProxyRequest:
        if len(set(self.elements)) != len(self.elements):
            raise ValueError("elements must not contain duplicates")
        expert_ids = [item.expert_id for item in self.expert_weight_revisions]
        if len(set(expert_ids)) != len(expert_ids):
            raise ValueError("expert_weight_revisions must have unique expert_id values")
        return self


class CandidateProxyReliabilityAssessment(StrictSchema):
    """A calibrated interval, or a fail-closed out-of-scope decision."""

    candidate_id: Identifier
    proxy_name: Identifier
    status: Literal["calibrated_in_scope", "uncalibrated_or_ood"]
    exchangeability_scope_match: bool
    calibration_artifact_id: Identifier | None = None
    interval_lower: float | None = None
    interval_upper: float | None = None
    alpha: float | None = Field(default=None, gt=0.0, lt=1.0)
    nominal_coverage: Probability | None = None
    calibration_quantile: float | None = Field(default=None, ge=0.0)
    held_out_empirical_coverage: Probability | None = None
    dft_escalation: bool
    reasons: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def _coverage_claim_boundary(self) -> CandidateProxyReliabilityAssessment:
        calibrated_fields = (
            self.calibration_artifact_id,
            self.interval_lower,
            self.interval_upper,
            self.alpha,
            self.nominal_coverage,
            self.calibration_quantile,
            self.held_out_empirical_coverage,
        )
        if self.status == "uncalibrated_or_ood":
            if self.exchangeability_scope_match:
                raise ValueError("uncalibrated_or_ood cannot claim a scope match")
            if any(value is not None for value in calibrated_fields):
                raise ValueError(
                    "out-of-scope assessments cannot expose intervals or coverage claims"
                )
            if not self.dft_escalation:
                raise ValueError("uncalibrated_or_ood assessments must escalate to DFT")
            return self

        if not self.exchangeability_scope_match:
            raise ValueError("calibrated_in_scope requires an exchangeability scope match")
        if any(value is None for value in calibrated_fields):
            raise ValueError(
                "calibrated_in_scope requires provenance, interval, and coverage metadata"
            )
        if self.interval_lower > self.interval_upper:  # type: ignore[operator]
            raise ValueError("interval_lower cannot exceed interval_upper")
        return self


def finite_sample_split_conformal_rank(sample_count: int, alpha: float) -> int:
    """Return ``ceil((n + 1) * (1 - alpha))`` for split conformal.

    If the requested miscoverage is smaller than ``1 / (n + 1)``, a finite
    calibration score cannot represent the required order statistic.  The
    caller must add calibration samples or choose a supported ``alpha`` rather
    than silently clipping the rank and overstating coverage.
    """

    if isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and strictly between zero and one")
    rank = math.ceil((sample_count + 1) * (1.0 - alpha))
    if rank < 1 or rank > sample_count:
        raise ValueError(
            "alpha is too small for a finite split-conformal quantile at this sample count"
        )
    return rank


def finite_sample_split_conformal_quantile(
    calibration_scores: Sequence[float],
    alpha: float,
) -> float:
    """Return the finite-sample split-conformal order statistic."""

    if not calibration_scores:
        raise ValueError("calibration_scores must not be empty")
    scores = [float(score) for score in calibration_scores]
    if any(not math.isfinite(score) or score < 0.0 for score in scores):
        raise ValueError("calibration_scores must be finite and non-negative")
    rank = finite_sample_split_conformal_rank(len(scores), alpha)
    return sorted(scores)[rank - 1]


def assess_candidate_proxy(
    request: CandidateProxyRequest,
    calibration: SplitConformalCalibrationArtifact | None,
) -> CandidateProxyReliabilityAssessment:
    """Apply a conformal interval only when every applicability key matches."""

    reasons: list[str] = []
    if calibration is None:
        reasons.append("calibration_artifact_missing")
        return _uncalibrated_assessment(request, reasons)

    if request.scope_id != calibration.chemistry_scope.scope_id:
        reasons.append("exchangeability_scope_id_mismatch")
    if not set(request.elements).issubset(calibration.chemistry_scope.allowed_elements):
        reasons.append("chemistry_elements_out_of_scope")
    allowed_compositions = calibration.chemistry_scope.allowed_reduced_compositions
    if (
        allowed_compositions
        and request.reduced_composition not in allowed_compositions
    ):
        reasons.append("reduced_composition_out_of_scope")
    if _expert_revision_map(request.expert_weight_revisions) != _expert_revision_map(
        calibration.expert_weight_revisions
    ):
        reasons.append("expert_weight_revision_mismatch")
    if request.dft_method != calibration.dft_method:
        reasons.append("dft_method_mismatch")
    if request.dft_reference_hash != calibration.dft_reference_hash:
        reasons.append("dft_reference_hash_mismatch")

    metric = next(
        (
            item
            for item in calibration.metrics
            if item.proxy_name == request.proxy_name
        ),
        None,
    )
    if metric is None:
        reasons.append("proxy_not_calibrated")
    elif metric.unit != request.unit:
        reasons.append("proxy_unit_mismatch")

    if reasons:
        return _uncalibrated_assessment(request, reasons)

    assert metric is not None
    held_out = next(
        item
        for item in calibration.held_out_coverage
        if item.proxy_name == request.proxy_name
    )
    return CandidateProxyReliabilityAssessment(
        candidate_id=request.candidate_id,
        proxy_name=request.proxy_name,
        status="calibrated_in_scope",
        exchangeability_scope_match=True,
        calibration_artifact_id=calibration.artifact_id,
        interval_lower=request.proxy_value - metric.quantile,
        interval_upper=request.proxy_value + metric.quantile,
        alpha=calibration.alpha,
        nominal_coverage=1.0 - calibration.alpha,
        calibration_quantile=metric.quantile,
        held_out_empirical_coverage=held_out.empirical_coverage,
        dft_escalation=False,
        reasons=["exact_calibration_scope_match"],
    )


def _uncalibrated_assessment(
    request: CandidateProxyRequest,
    reasons: list[str],
) -> CandidateProxyReliabilityAssessment:
    return CandidateProxyReliabilityAssessment(
        candidate_id=request.candidate_id,
        proxy_name=request.proxy_name,
        status="uncalibrated_or_ood",
        exchangeability_scope_match=False,
        dft_escalation=True,
        reasons=reasons,
    )


def _expert_revision_map(
    rows: Sequence[ExpertWeightRevision],
) -> dict[str, str]:
    return {item.expert_id: item.weight_revision for item in rows}


class CompositionEnergyPair(StrictSchema):
    """Aligned per-candidate energy pair for one reduced composition."""

    candidate_id: Identifier
    reduced_composition: Identifier
    first_model_id: Identifier
    second_model_id: Identifier
    first_energy_per_atom_eV: float
    second_energy_per_atom_eV: float
    alignment_artifact_id: Identifier | None = None

    @model_validator(mode="after")
    def _different_models(self) -> CompositionEnergyPair:
        if self.first_model_id == self.second_model_id:
            raise ValueError("energy disagreement requires two different models")
        return self


class CompositionRelativeEnergyDisagreement(StrictSchema):
    """Offset-invariant energy/rank disagreement within one composition pool."""

    candidate_id: Identifier
    reduced_composition: Identifier
    first_model_id: Identifier
    second_model_id: Identifier
    pool_size: int = Field(gt=0)
    status: Literal["available", "unknown"]
    alignment_artifact_id: Identifier | None = None
    first_relative_energy_eV_atom: float | None = Field(default=None, ge=0.0)
    second_relative_energy_eV_atom: float | None = Field(default=None, ge=0.0)
    relative_energy_abs_diff_eV_atom: float | None = Field(default=None, ge=0.0)
    first_rank: int | None = Field(default=None, gt=0)
    second_rank: int | None = Field(default=None, gt=0)
    rank_abs_diff: int | None = Field(default=None, ge=0)
    dft_escalation: bool
    uncertainty_reasons: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unknown_is_not_low_risk(self) -> CompositionRelativeEnergyDisagreement:
        metrics = (
            self.first_relative_energy_eV_atom,
            self.second_relative_energy_eV_atom,
            self.relative_energy_abs_diff_eV_atom,
            self.first_rank,
            self.second_rank,
            self.rank_abs_diff,
        )
        if self.status == "unknown":
            if any(value is not None for value in metrics):
                raise ValueError("unknown relative-energy evidence cannot expose metrics")
            if not self.dft_escalation:
                raise ValueError("unknown relative-energy evidence must escalate to DFT")
            if not self.uncertainty_reasons:
                raise ValueError("unknown relative-energy evidence requires a reason")
            return self
        if any(value is None for value in metrics):
            raise ValueError("available relative-energy evidence requires every metric")
        if self.alignment_artifact_id is None:
            raise ValueError(
                "available relative-energy evidence requires an alignment artifact"
            )
        return self


def composition_relative_energy_disagreement(
    rows: Sequence[CompositionEnergyPair],
) -> list[CompositionRelativeEnergyDisagreement]:
    """Compare offset-free energies and ranks only within reduced compositions.

    ``alignment_artifact_id`` attests that both energy columns refer to the same
    candidate structures and energy-per-atom convention.  It is not a learned
    energy offset.  A singleton pool, absent alignment, or inconsistent
    alignment is unknown and therefore escalated, never labelled low risk.
    """

    if not rows:
        return []
    grouped: dict[str, list[tuple[int, CompositionEnergyPair]]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row.reduced_composition].append((index, row))

    outputs: dict[int, CompositionRelativeEnergyDisagreement] = {}
    for composition_rows in grouped.values():
        model_pairs = {
            (row.first_model_id, row.second_model_id)
            for _, row in composition_rows
        }
        if len(model_pairs) != 1:
            raise ValueError(
                "each reduced-composition pool must retain one ordered model pair"
            )
        candidate_ids = [row.candidate_id for _, row in composition_rows]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError(
                "candidate_id values must be unique within a reduced composition"
            )

        reasons: list[str] = []
        alignment_ids = {
            row.alignment_artifact_id
            for _, row in composition_rows
            if row.alignment_artifact_id is not None
        }
        if len(composition_rows) < 2:
            reasons.append("singleton_composition_pool")
        if any(row.alignment_artifact_id is None for _, row in composition_rows):
            reasons.append("energy_pair_alignment_missing")
        elif len(alignment_ids) != 1:
            reasons.append("energy_pair_alignment_inconsistent")

        if reasons:
            for index, row in composition_rows:
                outputs[index] = _unknown_relative_energy(row, len(composition_rows), reasons)
            continue

        first_minimum = min(
            row.first_energy_per_atom_eV for _, row in composition_rows
        )
        second_minimum = min(
            row.second_energy_per_atom_eV for _, row in composition_rows
        )
        first_relative = {
            row.candidate_id: max(0.0, row.first_energy_per_atom_eV - first_minimum)
            for _, row in composition_rows
        }
        second_relative = {
            row.candidate_id: max(0.0, row.second_energy_per_atom_eV - second_minimum)
            for _, row in composition_rows
        }

        for index, row in composition_rows:
            first_value = first_relative[row.candidate_id]
            second_value = second_relative[row.candidate_id]
            first_rank = _competition_rank(first_value, first_relative.values())
            second_rank = _competition_rank(second_value, second_relative.values())
            outputs[index] = CompositionRelativeEnergyDisagreement(
                candidate_id=row.candidate_id,
                reduced_composition=row.reduced_composition,
                first_model_id=row.first_model_id,
                second_model_id=row.second_model_id,
                pool_size=len(composition_rows),
                status="available",
                alignment_artifact_id=row.alignment_artifact_id,
                first_relative_energy_eV_atom=first_value,
                second_relative_energy_eV_atom=second_value,
                relative_energy_abs_diff_eV_atom=abs(first_value - second_value),
                first_rank=first_rank,
                second_rank=second_rank,
                rank_abs_diff=abs(first_rank - second_rank),
                dft_escalation=False,
            )

    return [outputs[index] for index in range(len(rows))]


def _competition_rank(value: float, population: Sequence[float]) -> int:
    return 1 + sum(other < value for other in population)


def _unknown_relative_energy(
    row: CompositionEnergyPair,
    pool_size: int,
    reasons: Sequence[str],
) -> CompositionRelativeEnergyDisagreement:
    return CompositionRelativeEnergyDisagreement(
        candidate_id=row.candidate_id,
        reduced_composition=row.reduced_composition,
        first_model_id=row.first_model_id,
        second_model_id=row.second_model_id,
        pool_size=pool_size,
        status="unknown",
        alignment_artifact_id=row.alignment_artifact_id,
        dft_escalation=True,
        uncertainty_reasons=list(reasons),
    )


class ForceDisagreementMetrics(StrictSchema):
    """Cross-model force differences with both global and local summaries."""

    atom_count: int = Field(gt=0)
    component_rmse_eV_A: float = Field(ge=0.0)
    per_atom_vector_rms_eV_A: float = Field(ge=0.0)
    max_atom_vector_norm_eV_A: float = Field(ge=0.0)


def force_disagreement_metrics(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> ForceDisagreementMetrics:
    """Retain the maximum per-atom vector error alongside RMS summaries."""

    if len(first) != len(second) or not first:
        raise ValueError("force tensors require the same non-zero atom count")

    vector_squared: list[float] = []
    for first_row, second_row in zip(first, second, strict=True):
        if len(first_row) != 3 or len(second_row) != 3:
            raise ValueError("force tensors must contain exactly xyz components")
        squared = 0.0
        for first_value, second_value in zip(first_row, second_row, strict=True):
            left = float(first_value)
            right = float(second_value)
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError("force tensors must be finite")
            squared += (left - right) ** 2
        vector_squared.append(squared)

    atom_count = len(vector_squared)
    sum_squared = sum(vector_squared)
    return ForceDisagreementMetrics(
        atom_count=atom_count,
        component_rmse_eV_A=math.sqrt(sum_squared / (3 * atom_count)),
        per_atom_vector_rms_eV_A=math.sqrt(sum_squared / atom_count),
        max_atom_vector_norm_eV_A=math.sqrt(max(vector_squared)),
    )


__all__ = [
    "CandidateProxyReliabilityAssessment",
    "CandidateProxyRequest",
    "ChemistryExchangeabilityScope",
    "CompositionEnergyPair",
    "CompositionRelativeEnergyDisagreement",
    "ExpertWeightRevision",
    "ForceDisagreementMetrics",
    "HeldOutCoverageMetadata",
    "SplitConformalCalibrationArtifact",
    "SplitConformalMetricCalibration",
    "assess_candidate_proxy",
    "composition_relative_energy_disagreement",
    "finite_sample_split_conformal_quantile",
    "finite_sample_split_conformal_rank",
    "force_disagreement_metrics",
]
