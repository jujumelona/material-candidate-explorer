"""Closed-loop discovery orchestration with code-owned evidence decisions."""

from __future__ import annotations

from datetime import UTC, datetime

from .compiler import PlanCompiler
from .hashing import stable_hash
from .generators import GeneratorRuntime
from .profiles import GateEvaluator, get_validation_profile
from .protocols import DiscoveryModel
from .runtime import ToolRuntime
from .schemas import (
    Candidate,
    CandidatePlan,
    CandidateProposalRequest,
    CandidateValidationAssessment,
    DiscoveryDomain,
    DiscoveryFinalReport,
    FinalCandidateReport,
    GoalCompileRequest,
    HypothesisBatch,
    HypothesisRequest,
    PredictionRequest,
    ResultAnalysisRequest,
    RevisionRequest,
    StopDecision,
    StopDecisionRequest,
    StopReason,
    ValidationPlanningRequest,
)
from .store import JsonDiscoveryStore


class DiscoveryEngine:
    def __init__(
        self,
        model: DiscoveryModel,
        generator_runtime: GeneratorRuntime,
        tool_runtime: ToolRuntime,
        store: JsonDiscoveryStore,
        plan_compiler: PlanCompiler,
        gate_evaluator: GateEvaluator | None = None,
    ) -> None:
        self.model = model
        self.generator_runtime = generator_runtime
        self.tool_runtime = tool_runtime
        self.store = store
        self.plan_compiler = plan_compiler
        self.gate_evaluator = gate_evaluator or GateEvaluator()

    def run(
        self,
        user_goal: str,
        *,
        max_cycles: int = 10,
        domain_hint: DiscoveryDomain | str | None = None,
    ) -> DiscoveryFinalReport:
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if self.store.checkpoint is None:
            goal = self.model.compile_goal(
                GoalCompileRequest(
                    user_text=user_goal,
                    domain_hint=(DiscoveryDomain(str(domain_hint)) if domain_hint else None),
                )
            )
            state = self.store.create_state(goal)
        else:
            goal = self.store.checkpoint.goal
            state = self.store.checkpoint.state
        cycle_limit = min(max_cycles, goal.max_cycles)
        profile = get_validation_profile(goal.validation_profile_id)
        if profile.domain != goal.domain:
            raise ValueError(
                f"validation profile {profile.profile_id!r} does not match goal domain {goal.domain!r}"
            )
        active_candidates = self.store.latest_candidates()
        latest_hypotheses = self._latest_hypotheses()
        next_action = "validate_more" if active_candidates else "generate_new"
        latest_analysis = self.store.checkpoint.analyses[-1] if self.store.checkpoint.analyses else None
        final_decision: StopDecision | None = None

        while state.cycle < cycle_limit:
            if next_action == "generate_new" or not active_candidates:
                latest_hypotheses = self.model.propose_hypotheses(
                    HypothesisRequest(
                        goal=goal,
                        state=state,
                        history_summary=self.store.history_summary(),
                    )
                )
                self.store.save_hypotheses(latest_hypotheses)
                candidate_plan = self.model.propose_candidates(
                    CandidateProposalRequest(
                        goal=goal,
                        state=state,
                        hypotheses=latest_hypotheses,
                        available_generators=self.generator_runtime.registry.describe_generators(),
                    )
                )
                self.store.save_candidate_plan(candidate_plan)
                generated = self.generator_runtime.execute(
                    candidate_plan,
                    existing_candidates=self.store.candidates(),
                )
                self._validate_generated_domain(generated.candidates, goal.domain)
                self.store.save_candidates(generated)
                active_candidates = generated.candidates
            if not active_candidates:
                final_decision = StopDecision(
                    stop=True,
                    reason_code=StopReason.NO_PROGRESS,
                    reason="No candidates are available for validation.",
                    unmet_criteria=["candidate generation"],
                    recommended_next_action="human_review",
                )
                self.store.save_stop_decision(final_decision)
                break

            predictions = self.model.predict_candidates(
                PredictionRequest(
                    goal=goal,
                    state=state,
                    candidates=active_candidates,
                    requested_properties=[item.property_name for item in goal.objectives],
                )
            )
            self.store.save_predictions(predictions)
            proposed_plan = self.model.plan_validation(
                ValidationPlanningRequest(
                    goal=goal,
                    state=state,
                    candidates=active_candidates,
                    predictions=predictions,
                    available_tools=self.tool_runtime.registry.describe_tools(),
                    validation_profile_id=profile.profile_id,
                )
            )
            compiled_plan = self.plan_compiler.compile(
                proposed_plan,
                goal=goal,
                candidates=active_candidates,
            )
            self.store.save_validation_plan(compiled_plan)
            evidence = self.tool_runtime.execute_plan(
                compiled_plan,
                candidates=active_candidates,
            )
            self.store.save_evidence(evidence)
            latest_analysis = self.model.analyze_results(
                ResultAnalysisRequest(
                    goal=goal,
                    current_hypotheses=latest_hypotheses.hypotheses,
                    candidates=active_candidates,
                    predictions_before_validation=predictions.predictions,
                    observed_evidence=evidence,
                    history_summary=self.store.history_summary(),
                )
            )
            self.store.save_analysis(latest_analysis)

            next_action = latest_analysis.next_recommended_action
            if next_action == "revise":
                revision_plan = self.model.revise_candidates(
                    RevisionRequest(
                        goal=goal,
                        state=state,
                        candidates=active_candidates,
                        evidence=evidence,
                        analysis=latest_analysis,
                        available_generators=self.generator_runtime.registry.describe_generators(),
                    )
                )
                self.store.save_revision_plan(revision_plan)
                if revision_plan.generation_tasks:
                    executable_revision = CandidatePlan(
                        tasks=revision_plan.generation_tasks,
                        plan_reason=revision_plan.plan_reason,
                    )
                    self.store.save_candidate_plan(executable_revision)
                    revised = self.generator_runtime.execute(
                        executable_revision,
                        existing_candidates=self.store.candidates(),
                    )
                    self._validate_generated_domain(revised.candidates, goal.domain)
                    self.store.save_candidates(revised)
                    active_candidates = revised.candidates
                    next_action = "validate_more"
                else:
                    # CandidateRevision is a structured request, not permission
                    # for the engine to mutate a scientific representation.
                    next_action = "generate_new"

            state = self.store.build_next_state()
            assessments = self._assess_all(profile.profile_id)
            self.store.save_assessments(assessments)

            if state.cycle >= cycle_limit:
                final_decision = StopDecision(
                    stop=True,
                    reason_code=StopReason.MAX_CYCLES_REACHED,
                    reason=f"The configured cycle limit ({cycle_limit}) was reached.",
                    unmet_criteria=self._unmet_gate_ids(assessments),
                    best_candidate_ids=[
                        item.candidate_id
                        for item in assessments
                        if item.status in {"computationally_supported", "experimentally_validated"}
                    ],
                    recommended_next_action=(
                        "finish"
                        if any(item.status == "experimentally_validated" for item in assessments)
                        else "human_review"
                    ),
                )
            else:
                model_decision = self.model.decide_stop(
                    StopDecisionRequest(
                        goal=goal,
                        state=state,
                        cycle=state.cycle,
                        history_summary=self.store.history_summary(),
                        latest_analysis=latest_analysis,
                        validation_assessments=assessments,
                    )
                )
                final_decision = self._enforce_stop_policy(model_decision, assessments)
            self.store.save_stop_decision(final_decision)
            if final_decision.stop:
                break
            if next_action == "finish":
                next_action = "validate_more"

        if final_decision is None:
            assessments = self._assess_all(profile.profile_id)
            final_decision = StopDecision(
                stop=True,
                reason_code=StopReason.NO_PROGRESS,
                reason="The run ended without completing an execution cycle.",
                unmet_criteria=self._unmet_gate_ids(assessments),
                recommended_next_action="human_review",
            )
            self.store.save_stop_decision(final_decision)
        report = self._build_final_report(final_decision, profile.profile_id)
        report_digest = stable_hash(report.model_dump(mode="json"))
        self.tool_runtime.artifact_store.write_json(
            (
                f"reports/{self.tool_runtime.artifact_store.safe_component(self.store.run_id)}-"
                f"{report_digest}.json"
            ),
            report.model_dump(mode="json"),
        )
        return report

    def _latest_hypotheses(self) -> HypothesisBatch:
        checkpoint = self.store.checkpoint
        assert checkpoint is not None
        return HypothesisBatch(hypotheses=list(checkpoint.hypotheses))

    @staticmethod
    def _validate_generated_domain(
        candidates: list[Candidate], domain: DiscoveryDomain | str
    ) -> None:
        wrong = [item.candidate_id for item in candidates if item.domain != domain]
        if wrong:
            raise ValueError(
                f"generator returned candidates outside goal domain {domain!r}: {wrong}"
            )

    def _assess_all(self, profile_id: str) -> list[CandidateValidationAssessment]:
        checkpoint = self.store.checkpoint
        assert checkpoint is not None
        profile = get_validation_profile(profile_id)
        return [
            self.gate_evaluator.evaluate(profile, candidate.candidate_id, checkpoint.evidence)
            for candidate in checkpoint.candidates
        ]

    @staticmethod
    def _unmet_gate_ids(assessments: list[CandidateValidationAssessment]) -> list[str]:
        return sorted(
            {
                gate.gate_id
                for assessment in assessments
                for gate in assessment.gate_decisions
                if not gate.passed
            }
        )

    @staticmethod
    def _enforce_stop_policy(
        decision: StopDecision,
        assessments: list[CandidateValidationAssessment],
    ) -> StopDecision:
        final_validated = any(item.status == "experimentally_validated" for item in assessments)
        if (
            decision.stop
            and decision.reason_code == StopReason.SUCCESS_CRITERIA_MET
            and not final_validated
        ):
            return StopDecision(
                stop=False,
                reason_code=StopReason.INSUFFICIENT_EVIDENCE,
                reason=(
                    "The model suggested success, but code-owned experimental and independent "
                    "replication gates are incomplete."
                ),
                unmet_criteria=DiscoveryEngine._unmet_gate_ids(assessments),
                best_candidate_ids=decision.best_candidate_ids,
                recommended_next_action="validate_more",
            )
        return decision

    def _build_final_report(
        self,
        stop_decision: StopDecision,
        profile_id: str,
    ) -> DiscoveryFinalReport:
        checkpoint = self.store.checkpoint
        assert checkpoint is not None
        assessments = self._assess_all(profile_id)
        self.store.save_assessments(assessments)
        assessment_by_id = {item.candidate_id: item for item in assessments}
        prediction_by_id = {item.candidate_id: item for item in checkpoint.predictions}
        reports: list[FinalCandidateReport] = []
        for candidate in checkpoint.candidates:
            assessment = assessment_by_id[candidate.candidate_id]
            if assessment.status == "experimentally_validated":
                disposition = "All versioned profile gates, including independent experimental replication, are documented."
            elif assessment.status == "computationally_supported":
                disposition = "Computational lead only; experimental validation remains required."
            elif assessment.status == "rejected":
                disposition = "A rejection-critical gate contains explicit contrary evidence."
            else:
                disposition = "Evidence is incomplete or inconclusive; no discovery claim is made."
            next_steps = [
                f"Complete gate {gate.gate_id}: {gate.reason}"
                for gate in assessment.gate_decisions
                if not gate.passed
            ]
            reports.append(
                FinalCandidateReport(
                    candidate=candidate,
                    predictions=prediction_by_id.get(candidate.candidate_id),
                    validation=assessment,
                    evidence_ids=[
                        item.evidence_id
                        for item in checkpoint.evidence
                        if item.candidate_id == candidate.candidate_id
                    ],
                    disposition_reason=disposition,
                    recommended_next_steps=next_steps,
                )
            )
        validated = [
            item.candidate.candidate_id
            for item in reports
            if item.validation.status == "experimentally_validated"
        ]
        conclusions = (
            [f"Candidates completing the configured profile: {', '.join(validated)}"]
            if validated
            else [
                "No candidate completed the configured experimental and independent-replication profile."
            ]
        )
        return DiscoveryFinalReport(
            run_id=self.store.run_id,
            goal=checkpoint.goal,
            stop_decision=stop_decision,
            candidate_reports=reports,
            history_summary=self.store.history_summary(),
            conclusions=conclusions,
            limitations=[
                "A finite validator catalog cannot cover every material, mechanism, or experimental condition.",
                "Mock generator candidates are known fixtures and do not demonstrate novelty.",
                "Unavailable high-fidelity connectors must be installed, configured, calibrated, and convergence-tested.",
            ],
            safety_and_ethics_notes=[
                "Computational screening is not medical efficacy, clinical safety, synthesis proof, or a superconductivity claim.",
                "Laboratory work, regulated studies, and hazardous-material handling require qualified human oversight.",
            ],
            generated_at=datetime.now(UTC).isoformat(),
        )


__all__ = ["DiscoveryEngine"]
