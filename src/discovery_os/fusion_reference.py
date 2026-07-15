"""Small deterministic fusion backend for contract tests and smoke checks.

This is not a scientific model.  The user's trained fusion AI should replace
it in production.
"""

from __future__ import annotations

from .fusion_schemas import (
    ChangeAxis,
    DesiredChange,
    FusionOutput,
    FusionRequest,
    FusionRevisionProposal,
    FusionRevisionRequest,
    NumericTensor,
)


class MeanFusionBackend:
    """Element-wise mean of already aligned expert tensors."""

    def __init__(self, *, dimension: int | None = None) -> None:
        self.dimension = dimension

    def fuse(self, request: FusionRequest) -> FusionOutput:
        tensors = [item.payload.tensor for item in request.features]
        if any(item is None for item in tensors):
            raise ValueError("reference fusion requires a tensor from every expert")
        materialized = [item for item in tensors if item is not None]
        shapes = {tuple(item.shape) for item in materialized}
        dtypes = {str(item.dtype) for item in materialized}
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError("expert features need an explicit projection into one feature space")
        values_length = len(materialized[0].values)
        if self.dimension is not None and values_length != self.dimension:
            raise ValueError("feature dimension does not match configured fusion dimension")
        values = [
            sum(tensor.values[index] for tensor in materialized) / len(materialized)
            for index in range(values_length)
        ]
        return FusionOutput(
            latent=NumericTensor(
                dtype=materialized[0].dtype,
                shape=list(materialized[0].shape),
                values=values,
            ),
            used_feature_ids=[item.feature_id for item in request.features],
            ignored_feature_ids=[],
            backend_id="mean-reference",
            backend_version="1.0.0",
            code_revision="builtin-mean-reference-v1",
            weight_revision="no-weights",
            warnings=["Reference mean fusion is diagnostic test code, not a trained model."],
        )

    def propose_revision(self, request: FusionRevisionRequest) -> FusionRevisionProposal:
        first_objective = request.goal.objectives[0]
        objective_direction = str(first_objective.direction)
        if objective_direction == "maximize":
            change_direction = "increase"
            target_value = None
        elif objective_direction == "minimize":
            change_direction = "decrease"
            target_value = None
        elif objective_direction == "target":
            change_direction = "target"
            target_value = first_objective.target_value
        elif objective_direction == "range":
            change_direction = "target"
            target_value = [first_objective.lower_bound, first_objective.upper_bound]
        else:
            change_direction = "preserve"
            target_value = None
        return FusionRevisionProposal(
            parent_candidate_ref=request.state.candidate_ref,
            state_id=request.state.state_id,
            desired_changes=[
                DesiredChange(
                    axis=ChangeAxis.TARGET_PROPERTY,
                    direction=change_direction,
                    property_name=first_objective.property_name,
                    target_value=target_value,
                    rationale="Exercise the structured revision port for the first objective.",
                )
            ],
            confidence=0.0,
            rationale="Deterministic smoke-test proposal; replace with a trained fusion backend.",
            safety_notes=["Do not treat this proposal as scientific evidence."],
        )


__all__ = ["MeanFusionBackend"]
