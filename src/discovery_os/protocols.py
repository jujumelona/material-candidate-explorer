"""Ports implemented by local, remote, and test discovery models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .schemas import (
    CandidatePlan,
    CandidateProposalRequest,
    DiscoveryGoal,
    GoalCompileRequest,
    HypothesisBatch,
    HypothesisRequest,
    PredictionBatch,
    PredictionRequest,
    ResultAnalysis,
    ResultAnalysisRequest,
    RevisionPlan,
    RevisionRequest,
    StopDecision,
    StopDecisionRequest,
    ValidationPlan,
    ValidationPlanningRequest,
)


@runtime_checkable
class DiscoveryModel(Protocol):
    """Structured decision interface for any discovery model backend.

    Implementations may run in-process or over HTTP.  They return only strict
    Pydantic contracts; execution, allow-listing, budgets, persistence, and
    final validation decisions remain responsibilities of deterministic code.
    """

    def compile_goal(self, request: GoalCompileRequest) -> DiscoveryGoal:
        ...

    def propose_hypotheses(self, request: HypothesisRequest) -> HypothesisBatch:
        ...

    def propose_candidates(self, request: CandidateProposalRequest) -> CandidatePlan:
        ...

    def predict_candidates(self, request: PredictionRequest) -> PredictionBatch:
        ...

    def plan_validation(self, request: ValidationPlanningRequest) -> ValidationPlan:
        ...

    def analyze_results(self, request: ResultAnalysisRequest) -> ResultAnalysis:
        ...

    def revise_candidates(self, request: RevisionRequest) -> RevisionPlan:
        ...

    def decide_stop(self, request: StopDecisionRequest) -> StopDecision:
        ...


__all__ = ["DiscoveryModel"]
