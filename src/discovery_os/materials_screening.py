"""Evidence-preserving cross-MLIP screening and composition-scoped Pareto ranks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

from .mlip_reliability import CompositionRelativeEnergyDisagreement
from .schemas import CandidateRef, Identifier, StrictSchema


class GenerationConditions(StrictSchema):
    """Targets sent to a generator; these are not measured validation values."""

    profile_id: Identifier
    guidance_alpha: float = Field(ge=0.0, le=1.0)
    target_energy_above_hull_eV_atom: float | None = None


class ThermodynamicValidation(StrictSchema):
    """Explicitly uncalculated until a reference-consistent DFT workflow runs."""

    formation_energy_eV_atom: None = None
    computed_energy_above_hull_eV_atom: None = None
    reference_phase_set: None = None
    method: None = None


class MLIPScreeningPrediction(StrictSchema):
    expert_id: Identifier
    energy_per_atom_eV: float
    max_force_eV_A: float = Field(ge=0.0)
    stress_norm: float | None = Field(default=None, ge=0.0)
    stress_unit: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _stress_unit_boundary(self) -> MLIPScreeningPrediction:
        if (self.stress_norm is None) != (self.stress_unit is None):
            raise ValueError("stress_norm and stress_unit must be present together")
        return self


class ModelDisagreement(StrictSchema):
    # Retained for audit only. Different MLIPs can have incompatible absolute
    # energy gauges, so this value never drives risk classification.
    raw_energy_per_atom_abs_diff_eV: float = Field(ge=0.0)
    energy_comparison_basis: Literal[
        "composition_relative_aligned", "unknown"
    ]
    composition_relative_energy_abs_diff_eV_atom: float | None = Field(
        default=None, ge=0.0
    )
    composition_relative_rank_abs_diff: int | None = Field(default=None, ge=0)
    composition_relative_candidate_id: Identifier | None = None
    composition_relative_composition_key: Identifier | None = None
    composition_relative_alignment_artifact_id: Identifier | None = None
    force_rmse_eV_A: float = Field(ge=0.0)
    stress_norm_abs_diff_GPa: float | None = Field(default=None, ge=0.0)
    relaxed_structure_match: bool | None = None
    risk: Literal["low", "medium", "high"]
    dft_escalation: bool
    uncertainty_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _high_risk_is_escalated(self) -> ModelDisagreement:
        if self.risk == "high" and not self.dft_escalation:
            raise ValueError("high model disagreement must be escalated to DFT")
        relative_fields = (
            self.composition_relative_energy_abs_diff_eV_atom,
            self.composition_relative_rank_abs_diff,
            self.composition_relative_candidate_id,
            self.composition_relative_composition_key,
            self.composition_relative_alignment_artifact_id,
        )
        if self.energy_comparison_basis == "unknown":
            if any(item is not None for item in relative_fields):
                raise ValueError("unknown energy comparison cannot expose relative metrics")
            if not self.dft_escalation or not self.uncertainty_reasons:
                raise ValueError("unknown energy comparison must fail closed to DFT")
        elif any(item is None for item in relative_fields):
            raise ValueError(
                "composition-relative energy comparison requires value, rank, and composition"
            )
        return self


class CandidateScreeningVector(StrictSchema):
    candidate_ref: CandidateRef
    composition_key: Identifier
    # Pareto axes from each expert's independently relaxed final geometry.
    mattersim: MLIPScreeningPrediction
    chgnet: MLIPScreeningPrediction
    # Same-input geometry receipts used only for force/stress disagreement and
    # the audit-only raw energy offset.  They must never replace relaxed Pareto
    # values or composition-relative energy evidence.
    common_geometry_mattersim: MLIPScreeningPrediction
    common_geometry_chgnet: MLIPScreeningPrediction
    common_geometry_alignment_id: Identifier
    disagreement: ModelDisagreement
    geometry_valid: bool
    relaxation_gate_passed: bool

    @model_validator(mode="after")
    def _expert_identity(self) -> CandidateScreeningVector:
        if self.mattersim.expert_id != "mattersim":
            raise ValueError("mattersim prediction must retain expert_id='mattersim'")
        if self.chgnet.expert_id != "chgnet":
            raise ValueError("chgnet prediction must retain expert_id='chgnet'")
        if self.common_geometry_mattersim.expert_id != "mattersim":
            raise ValueError(
                "common_geometry_mattersim must retain expert_id='mattersim'"
            )
        if self.common_geometry_chgnet.expert_id != "chgnet":
            raise ValueError(
                "common_geometry_chgnet must retain expert_id='chgnet'"
            )
        expected_raw = abs(
            self.common_geometry_mattersim.energy_per_atom_eV
            - self.common_geometry_chgnet.energy_per_atom_eV
        )
        if not math.isclose(
            self.disagreement.raw_energy_per_atom_abs_diff_eV,
            expected_raw,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "raw energy audit does not match the aligned common-geometry predictions"
            )
        if (
            self.disagreement.energy_comparison_basis
            == "composition_relative_aligned"
            and (
                self.disagreement.composition_relative_candidate_id
                != self.candidate_ref.candidate_id
                or self.disagreement.composition_relative_composition_key
                != self.composition_key
            )
        ):
            raise ValueError(
                "composition-relative energy evidence belongs to another candidate or composition"
            )
        return self


class ParetoRankedScreening(StrictSchema):
    candidate_ref: CandidateRef
    composition_key: Identifier
    composition_pareto_front: int = Field(gt=0)
    global_priority_rank: int = Field(gt=0)
    pareto_crowding_distance: float = Field(default=0.0, ge=0.0)
    pareto_boundary: bool = False
    dft_escalation: bool
    rationale: list[str] = Field(min_length=1)


def force_rmse(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> float:
    """Compare force xyz columns after the caller has preserved atom ordering."""

    if len(first) != len(second) or not first:
        raise ValueError("force tensors require the same non-zero atom count")
    squared: list[float] = []
    for left, right in zip(first, second, strict=True):
        if len(left) < 3 or len(right) < 3:
            raise ValueError("force tensors require at least xyz columns")
        for left_value, right_value in zip(left[:3], right[:3], strict=True):
            a = float(left_value)
            b = float(right_value)
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("force tensors must be finite")
            squared.append((a - b) ** 2)
    return math.sqrt(sum(squared) / len(squared))


def classify_model_disagreement(
    mattersim: MLIPScreeningPrediction,
    chgnet: MLIPScreeningPrediction,
    *,
    force_rmse_eV_A: float,
    relative_energy: CompositionRelativeEnergyDisagreement | None = None,
    relaxed_structure_match: bool | None = None,
    medium_energy_threshold_eV: float = 0.08,
    high_energy_threshold_eV: float = 0.15,
    medium_force_threshold_eV_A: float = 0.12,
    high_force_threshold_eV_A: float = 0.25,
    medium_stress_threshold_GPa: float = 5.0,
    high_stress_threshold_GPa: float = 15.0,
    medium_rank_difference: int = 1,
    high_rank_difference: int = 2,
    require_stress_comparison: bool = False,
    require_relaxed_structure_comparison: bool = False,
) -> ModelDisagreement:
    """Classify disagreement without treating raw MLIP offsets as uncertainty.

    Energy can influence the diagnostic risk only through an aligned,
    composition-relative panel. Missing/singleton/misaligned evidence is
    unknown and therefore escalated rather than interpreted as agreement.
    """

    raw_energy = abs(mattersim.energy_per_atom_eV - chgnet.energy_per_atom_eV)
    if force_rmse_eV_A < 0 or not math.isfinite(force_rmse_eV_A):
        raise ValueError("force_rmse_eV_A must be finite and non-negative")
    for label, medium_value, high_value in (
        (
            "composition-relative energy",
            medium_energy_threshold_eV,
            high_energy_threshold_eV,
        ),
        ("force", medium_force_threshold_eV_A, high_force_threshold_eV_A),
        ("stress", medium_stress_threshold_GPa, high_stress_threshold_GPa),
    ):
        if (
            not math.isfinite(medium_value)
            or not math.isfinite(high_value)
            or medium_value < 0.0
            or high_value < medium_value
        ):
            raise ValueError(
                f"{label} thresholds must be finite, non-negative, and high >= medium"
            )
    if (
        isinstance(medium_rank_difference, bool)
        or isinstance(high_rank_difference, bool)
        or not isinstance(medium_rank_difference, int)
        or not isinstance(high_rank_difference, int)
        or medium_rank_difference < 0
        or high_rank_difference < medium_rank_difference
    ):
        raise ValueError(
            "rank thresholds must be non-negative integers with high >= medium"
        )
    stress_values = (
        stress_norm_gpa(mattersim),
        stress_norm_gpa(chgnet),
    )
    stress = (
        abs(stress_values[0] - stress_values[1])
        if stress_values[0] is not None and stress_values[1] is not None
        else None
    )
    uncertainty_reasons: list[str] = []
    relative_energy_value: float | None = None
    relative_rank_difference: int | None = None
    relative_candidate_id: str | None = None
    relative_composition: str | None = None
    relative_alignment_artifact_id: str | None = None
    energy_basis: Literal["composition_relative_aligned", "unknown"] = "unknown"
    if relative_energy is None:
        uncertainty_reasons.append("composition_relative_energy_comparison_unavailable")
    elif (
        relative_energy.first_model_id != mattersim.expert_id
        or relative_energy.second_model_id != chgnet.expert_id
    ):
        uncertainty_reasons.append("composition_relative_energy_model_alignment_mismatch")
    elif relative_energy.status == "unknown":
        uncertainty_reasons.extend(relative_energy.uncertainty_reasons)
    else:
        assert relative_energy.relative_energy_abs_diff_eV_atom is not None
        assert relative_energy.rank_abs_diff is not None
        energy_basis = "composition_relative_aligned"
        relative_energy_value = relative_energy.relative_energy_abs_diff_eV_atom
        relative_rank_difference = relative_energy.rank_abs_diff
        relative_candidate_id = relative_energy.candidate_id
        relative_composition = relative_energy.reduced_composition
        relative_alignment_artifact_id = relative_energy.alignment_artifact_id
    if require_stress_comparison and stress is None:
        uncertainty_reasons.append("cross_model_stress_comparison_unavailable")
    if require_relaxed_structure_comparison and relaxed_structure_match is None:
        uncertainty_reasons.append("relaxed_structure_comparison_unavailable")
    high = (
        (
            relative_energy_value is not None
            and relative_energy_value >= high_energy_threshold_eV
        )
        or (
            relative_rank_difference is not None
            and relative_rank_difference >= high_rank_difference
        )
        or force_rmse_eV_A >= high_force_threshold_eV_A
        or (stress is not None and stress >= high_stress_threshold_GPa)
        or relaxed_structure_match is False
        or bool(uncertainty_reasons)
    )
    medium = (
        (
            relative_energy_value is not None
            and relative_energy_value >= medium_energy_threshold_eV
        )
        or (
            relative_rank_difference is not None
            and relative_rank_difference >= medium_rank_difference
        )
        or force_rmse_eV_A >= medium_force_threshold_eV_A
        or (stress is not None and stress >= medium_stress_threshold_GPa)
    )
    risk: Literal["low", "medium", "high"] = (
        "high" if high else "medium" if medium else "low"
    )
    return ModelDisagreement(
        raw_energy_per_atom_abs_diff_eV=raw_energy,
        energy_comparison_basis=energy_basis,
        composition_relative_energy_abs_diff_eV_atom=relative_energy_value,
        composition_relative_rank_abs_diff=relative_rank_difference,
        composition_relative_candidate_id=relative_candidate_id,
        composition_relative_composition_key=relative_composition,
        composition_relative_alignment_artifact_id=(
            relative_alignment_artifact_id
        ),
        force_rmse_eV_A=force_rmse_eV_A,
        stress_norm_abs_diff_GPa=stress,
        relaxed_structure_match=relaxed_structure_match,
        risk=risk,
        dft_escalation=high,
        uncertainty_reasons=uncertainty_reasons,
    )


def stress_norm_gpa(prediction: MLIPScreeningPrediction) -> float | None:
    """Normalize a reported stress norm to GPa for cross-model comparison."""

    if prediction.stress_norm is None:
        return None
    unit = (
        (prediction.stress_unit or "")
        .strip()
        .casefold()
        .replace(" ", "")
        .replace("Å", "angstrom")
        .replace("ångström", "angstrom")
        .replace("ångstrom", "angstrom")
        .replace("å", "angstrom")
    )
    if unit == "gpa":
        return prediction.stress_norm
    if unit in {
        "ev/angstrom^3",
        "ev/ang^3",
        "ev/a^3",
    }:
        return prediction.stress_norm * 160.2176634
    raise ValueError(
        f"unsupported stress unit {prediction.stress_unit!r}; expected GPa or eV/angstrom^3"
    )


def rank_composition_scoped_pareto(
    vectors: Sequence[CandidateScreeningVector],
) -> list[ParetoRankedScreening]:
    """Pareto-rank MLIP energies only against candidates of one composition.

    Absolute MLIP energies from different stoichiometries never enter the same
    dominance comparison.  The global ordering uses Pareto-front number, safety
    gates, force envelope, disagreement risk, and immutable candidate identity.
    """

    groups: dict[str, list[CandidateScreeningVector]] = defaultdict(list)
    for vector in vectors:
        groups[vector.composition_key].append(vector)
    fronts: dict[tuple[str, int, str], int] = {}
    front_members: dict[tuple[str, int], list[CandidateScreeningVector]] = {}
    for composition, rows in groups.items():
        remaining = list(rows)
        front = 1
        while remaining:
            nondominated = [
                row
                for row in remaining
                if not any(
                    _dominates(other, row)
                    for other in remaining
                    if other.candidate_ref != row.candidate_ref
                )
            ]
            if not nondominated:
                raise RuntimeError("Pareto ranking failed to identify a non-dominated row")
            front_members[(composition, front)] = list(nondominated)
            for row in nondominated:
                key = (
                    row.candidate_ref.candidate_id,
                    row.candidate_ref.version,
                    row.candidate_ref.content_hash,
                )
                fronts[key] = front
                selected = {_ref_key(item.candidate_ref) for item in nondominated}
                remaining = [
                    item
                    for item in remaining
                    if _ref_key(item.candidate_ref) not in selected
                ]
            front += 1

    crowding: dict[tuple[str, int, str], tuple[bool, float]] = {}
    for members in front_members.values():
        crowding.update(_pareto_crowding(members))

    risk_order = {"low": 0, "medium": 1, "high": 2}
    ordered = sorted(
        vectors,
        key=lambda row: (
            not row.geometry_valid,
            not row.relaxation_gate_passed,
            fronts[
                (
                    row.candidate_ref.candidate_id,
                    row.candidate_ref.version,
                    row.candidate_ref.content_hash,
                )
            ],
            not crowding[_ref_key(row.candidate_ref)][0],
            -crowding[_ref_key(row.candidate_ref)][1],
            max(row.mattersim.max_force_eV_A, row.chgnet.max_force_eV_A),
            risk_order[row.disagreement.risk],
            row.composition_key,
            row.candidate_ref.candidate_id,
            row.candidate_ref.content_hash,
        ),
    )
    output: list[ParetoRankedScreening] = []
    for priority, row in enumerate(ordered, 1):
        key = (
            row.candidate_ref.candidate_id,
            row.candidate_ref.version,
            row.candidate_ref.content_hash,
        )
        rationale = [
            "MLIP energy dominance was evaluated only within the reduced composition",
            f"composition Pareto front {fronts[key]}",
        ]
        boundary, distance = crowding[key]
        rationale.append(
            (
                "NSGA-II crowding boundary retained to preserve the Pareto envelope"
                if boundary
                else f"NSGA-II normalized crowding distance {distance:.12g}"
            )
        )
        if row.disagreement.dft_escalation:
            rationale.append("high cross-model disagreement retained for DFT escalation")
        if not row.relaxation_gate_passed:
            rationale.append("strict MLIP relaxation gate did not pass")
        output.append(
            ParetoRankedScreening(
                candidate_ref=row.candidate_ref,
                composition_key=row.composition_key,
                composition_pareto_front=fronts[key],
                global_priority_rank=priority,
                pareto_crowding_distance=distance,
                pareto_boundary=boundary,
                dft_escalation=row.disagreement.dft_escalation,
                rationale=rationale,
            )
        )
    return output


def select_dft_handoff_refs(
    ranked: Sequence[ParetoRankedScreening],
    vectors: Sequence[CandidateScreeningVector],
    *,
    top_k: int,
) -> list[CandidateRef]:
    """Select a composition-diverse portfolio with a disagreement-risk slot."""

    if isinstance(top_k, bool) or not 1 <= top_k <= 5:
        raise ValueError("top_k must be between 1 and 5")
    by_ref = {_ref_key(item.candidate_ref): item for item in vectors}
    eligible = [
        row
        for row in ranked
        if by_ref[_ref_key(row.candidate_ref)].geometry_valid
        and (
            by_ref[_ref_key(row.candidate_ref)].relaxation_gate_passed
            or row.dft_escalation
        )
    ]
    selected: list[ParetoRankedScreening] = []
    selected_refs: set[tuple[str, int, str]] = set()
    selected_compositions: set[str] = set()
    # Cover distinct reduced compositions first so a prolific stoichiometry
    # cannot consume every expensive DFT handoff slot.
    for row in eligible:
        if row.composition_key in selected_compositions:
            continue
        selected.append(row)
        selected_refs.add(_ref_key(row.candidate_ref))
        selected_compositions.add(row.composition_key)
        if len(selected) == top_k:
            break
    if len(selected) < top_k:
        for row in eligible:
            if _ref_key(row.candidate_ref) in selected_refs:
                continue
            selected.append(row)
            selected_refs.add(_ref_key(row.candidate_ref))
            if len(selected) == top_k:
                break
    escalations = [row for row in eligible if row.dft_escalation]
    if escalations and not any(row.dft_escalation for row in selected):
        if len(selected) == top_k:
            selected[-1] = escalations[0]
        else:
            selected.append(escalations[0])
    return [row.candidate_ref for row in selected]


def _pareto_crowding(
    members: Sequence[CandidateScreeningVector],
) -> dict[tuple[str, int, str], tuple[bool, float]]:
    """Compute deterministic NSGA-II crowding inside one composition/front."""

    if not members:
        return {}
    distances = {_ref_key(item.candidate_ref): 0.0 for item in members}
    boundaries: set[tuple[str, int, str]] = set()
    dimensions = (
        lambda row: row.mattersim.energy_per_atom_eV,
        lambda row: row.chgnet.energy_per_atom_eV,
        lambda row: row.mattersim.max_force_eV_A,
        lambda row: row.chgnet.max_force_eV_A,
    )
    if len(members) <= 2:
        boundaries.update(_ref_key(item.candidate_ref) for item in members)
    else:
        for value in dimensions:
            ordered = sorted(
                members,
                key=lambda item: (value(item), _ref_key(item.candidate_ref)),
            )
            low = float(value(ordered[0]))
            high = float(value(ordered[-1]))
            if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
                continue
            boundaries.add(_ref_key(ordered[0].candidate_ref))
            boundaries.add(_ref_key(ordered[-1].candidate_ref))
            scale = high - low
            for index in range(1, len(ordered) - 1):
                key = _ref_key(ordered[index].candidate_ref)
                distances[key] += (
                    float(value(ordered[index + 1]))
                    - float(value(ordered[index - 1]))
                ) / scale
    return {
        key: (key in boundaries, round(distance, 12))
        for key, distance in distances.items()
    }


def _dominates(
    left: CandidateScreeningVector,
    right: CandidateScreeningVector,
) -> bool:
    if left.composition_key != right.composition_key:
        return False
    left_values = (
        left.mattersim.energy_per_atom_eV,
        left.chgnet.energy_per_atom_eV,
        left.mattersim.max_force_eV_A,
        left.chgnet.max_force_eV_A,
    )
    right_values = (
        right.mattersim.energy_per_atom_eV,
        right.chgnet.energy_per_atom_eV,
        right.mattersim.max_force_eV_A,
        right.chgnet.max_force_eV_A,
    )
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


def _ref_key(reference: CandidateRef) -> tuple[str, int, str]:
    return (reference.candidate_id, reference.version, reference.content_hash)


__all__ = [
    "CandidateScreeningVector",
    "GenerationConditions",
    "MLIPScreeningPrediction",
    "ModelDisagreement",
    "ParetoRankedScreening",
    "ThermodynamicValidation",
    "classify_model_disagreement",
    "force_rmse",
    "rank_composition_scoped_pareto",
    "select_dft_handoff_refs",
    "stress_norm_gpa",
]
