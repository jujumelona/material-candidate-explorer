"""Strict wire records for periodic MLIP structure relaxation.

Feature inference and relaxation are intentionally separate operations.  A
successful HTTP/model invocation does not imply that an optimizer converged or
that the resulting geometry passed the safety gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field, model_validator

from .schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    RepresentationKind,
    StrictSchema,
)


class PeriodicRelaxationSettings(StrictSchema):
    """Bounded optimizer controls and post-relaxation safety thresholds."""

    optimizer: Literal["FIRE", "BFGS"] = "FIRE"
    requested_steps: int = Field(default=100, ge=1, le=2_000)
    target_fmax_eV_A: float = Field(default=0.05, gt=0.0, le=10.0)
    relax_cell: bool = True
    minimum_distance_safety_A: float = Field(default=0.7, gt=0.0, le=10.0)
    max_abs_volume_change_fraction: float = Field(default=0.35, ge=0.0, le=10.0)


class PeriodicRelaxationRequest(StrictSchema):
    candidate: Candidate
    settings: PeriodicRelaxationSettings = Field(
        default_factory=PeriodicRelaxationSettings
    )
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _periodic_crystal_only(self) -> PeriodicRelaxationRequest:
        if self.candidate.candidate_ref is None:
            raise ValueError("periodic relaxation requires an immutable candidate_ref")
        if self.candidate.candidate_type not in {
            CandidateType.CRYSTAL,
            CandidateType.ALLOY,
            CandidateType.BATTERY_MATERIAL,
            CandidateType.CATALYST,
        }:
            raise ValueError("periodic relaxation requires a periodic material candidate")
        if not any(
            item.kind in {RepresentationKind.CIF, RepresentationKind.POSCAR}
            for item in self.candidate.representations
        ):
            raise ValueError("periodic relaxation requires CIF or POSCAR")
        return self


class PeriodicRelaxationPayload(StrictSchema):
    """One executed optimizer run, including an independent strict gate."""

    candidate_ref: CandidateRef
    expert_id: str = Field(min_length=1, max_length=128)
    execution_succeeded: bool
    optimizer: Literal["FIRE", "BFGS"]
    requested_steps: int = Field(ge=1, le=2_000)
    completed_steps: int = Field(ge=0, le=2_000)
    converged: bool
    target_fmax_eV_A: float = Field(gt=0.0, le=10.0)
    initial_max_force_eV_A: float = Field(ge=0.0)
    final_max_force_eV_A: float = Field(ge=0.0)
    initial_energy_eV: float
    final_energy_eV: float
    volume_change_fraction: float
    minimum_distance_before_A: float = Field(gt=0.0)
    minimum_distance_after_A: float = Field(gt=0.0)
    relaxed_structure: CandidateRepresentation
    strict_gate_passed: bool
    gate_failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _result_is_consistent(self) -> PeriodicRelaxationPayload:
        if not self.execution_succeeded:
            raise ValueError("a relaxation payload is emitted only after execution succeeds")
        if self.completed_steps > self.requested_steps:
            raise ValueError("completed_steps cannot exceed requested_steps")
        if self.relaxed_structure.kind != RepresentationKind.CIF:
            raise ValueError("relaxed_structure must be a CIF representation")
        if self.strict_gate_passed:
            if not self.converged or self.gate_failures:
                raise ValueError("a passing strict gate requires convergence and no failures")
        elif not self.gate_failures:
            raise ValueError("a failed strict gate requires at least one reason")
        return self


@dataclass(frozen=True, slots=True)
class PeriodicRelaxationResult:
    """In-process runtime result before identity/provenance are bound."""

    completed_steps: int
    converged: bool
    initial_max_force_eV_A: float
    final_max_force_eV_A: float
    initial_energy_eV: float
    final_energy_eV: float
    volume_change_fraction: float
    minimum_distance_before_A: float
    minimum_distance_after_A: float
    relaxed_cif: str
    warnings: tuple[str, ...] = ()
    runtime_metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "PeriodicRelaxationPayload",
    "PeriodicRelaxationRequest",
    "PeriodicRelaxationResult",
    "PeriodicRelaxationSettings",
]
