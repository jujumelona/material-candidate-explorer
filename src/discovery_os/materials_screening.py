"""Evidence-preserving cross-MLIP screening and composition-scoped Pareto ranks."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from typing import Literal

from pydantic import Field, model_validator

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
    energy_per_atom_abs_diff_eV: float = Field(ge=0.0)
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
        return self


class CandidateScreeningVector(StrictSchema):
    candidate_ref: CandidateRef
    composition_key: Identifier
    mattersim: MLIPScreeningPrediction
    chgnet: MLIPScreeningPrediction
    disagreement: ModelDisagreement
    geometry_valid: bool
    relaxation_gate_passed: bool

    @model_validator(mode="after")
    def _expert_identity(self) -> CandidateScreeningVector:
        if self.mattersim.expert_id != "mattersim":
            raise ValueError("mattersim prediction must retain expert_id='mattersim'")
        if self.chgnet.expert_id != "chgnet":
            raise ValueError("chgnet prediction must retain expert_id='chgnet'")
        expected = abs(
            self.mattersim.energy_per_atom_eV - self.chgnet.energy_per_atom_eV
        )
        if not math.isclose(
            self.disagreement.energy_per_atom_abs_diff_eV,
            expected,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("energy disagreement does not match the expert predictions")
        return self


class ParetoRankedScreening(StrictSchema):
    candidate_ref: CandidateRef
    composition_key: Identifier
    composition_pareto_front: int = Field(gt=0)
    global_priority_rank: int = Field(gt=0)
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
    relaxed_structure_match: bool | None = None,
    medium_energy_threshold_eV: float = 0.08,
    high_energy_threshold_eV: float = 0.15,
    medium_force_threshold_eV_A: float = 0.12,
    high_force_threshold_eV_A: float = 0.25,
    medium_stress_threshold_GPa: float = 5.0,
    high_stress_threshold_GPa: float = 15.0,
    require_stress_comparison: bool = False,
    require_relaxed_structure_comparison: bool = False,
) -> ModelDisagreement:
    """Classify disagreement without turning agreement into a stability claim."""

    energy = abs(mattersim.energy_per_atom_eV - chgnet.energy_per_atom_eV)
    if force_rmse_eV_A < 0 or not math.isfinite(force_rmse_eV_A):
        raise ValueError("force_rmse_eV_A must be finite and non-negative")
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
    if require_stress_comparison and stress is None:
        uncertainty_reasons.append("cross_model_stress_comparison_unavailable")
    if require_relaxed_structure_comparison and relaxed_structure_match is None:
        uncertainty_reasons.append("relaxed_structure_comparison_unavailable")
    high = (
        energy >= high_energy_threshold_eV
        or force_rmse_eV_A >= high_force_threshold_eV_A
        or (stress is not None and stress >= high_stress_threshold_GPa)
        or relaxed_structure_match is False
        or bool(uncertainty_reasons)
    )
    medium = (
        energy >= medium_energy_threshold_eV
        or force_rmse_eV_A >= medium_force_threshold_eV_A
        or (stress is not None and stress >= medium_stress_threshold_GPa)
    )
    risk: Literal["low", "medium", "high"] = (
        "high" if high else "medium" if medium else "low"
    )
    return ModelDisagreement(
        energy_per_atom_abs_diff_eV=energy,
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
    """Select safe Pareto leaders while reserving a slot for disagreement risk."""

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
    selected = list(eligible[:top_k])
    escalations = [row for row in eligible if row.dft_escalation]
    if escalations and not any(row.dft_escalation for row in selected):
        if len(selected) == top_k:
            selected[-1] = escalations[0]
        else:
            selected.append(escalations[0])
    return [row.candidate_ref for row in selected]


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
