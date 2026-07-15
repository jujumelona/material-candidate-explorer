"""Checkpointed discovery state and append-only audit events."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from collections.abc import Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .hashing import stable_hash
from .schemas import (
    Candidate,
    CandidateBatch,
    CandidatePlan,
    CandidatePrediction,
    CandidateValidationAssessment,
    DiscoveryGoal,
    DiscoveryHistorySummary,
    DiscoveryState,
    EvidenceBatch,
    EvidenceRecord,
    Hypothesis,
    HypothesisBatch,
    PredictionBatch,
    ResultAnalysis,
    RevisionPlan,
    StopDecision,
    ValidationPlan,
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RunCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: str = "1"
    run_id: str
    goal: DiscoveryGoal
    state: DiscoveryState
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    candidate_plans: list[CandidatePlan] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    predictions: list[CandidatePrediction] = Field(default_factory=list)
    validation_plans: list[ValidationPlan] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    analyses: list[ResultAnalysis] = Field(default_factory=list)
    revision_plans: list[RevisionPlan] = Field(default_factory=list)
    stop_decisions: list[StopDecision] = Field(default_factory=list)
    assessments: list[CandidateValidationAssessment] = Field(default_factory=list)
    created_at: str
    updated_at: str


class JsonDiscoveryStore:
    """Persists the complete loop after every state transition.

    A candidate ID is immutable. Evidence carrying a ``CandidateRef`` is
    rejected if its version/content hash does not match the stored candidate.
    This prevents evidence from a parent candidate being silently reused after
    a revision.
    """

    def __init__(
        self,
        root: str | Path,
        run_id: str | None = None,
        *,
        experimental_record_verifier: Callable[[EvidenceRecord], bool] | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or f"RUN-{uuid4().hex[:16]}"
        safe_run_id = self._safe_run_id(self.run_id)
        self.run_dir = (self.root / safe_run_id).resolve()
        if self.run_dir == self.root or self.root not in self.run_dir.parents:
            raise ValueError("run_id escapes the configured store root")
        self.checkpoint_path = self.run_dir / "checkpoint.json"
        self.audit_path = self.run_dir / "audit.jsonl"
        self._checkpoint: RunCheckpoint | None = None
        self.experimental_record_verifier = experimental_record_verifier

    @property
    def checkpoint(self) -> RunCheckpoint | None:
        """Return an isolated snapshot; callers cannot mutate persisted state by alias."""

        return self._checkpoint.model_copy(deep=True) if self._checkpoint is not None else None

    @classmethod
    def resume(
        cls,
        root: str | Path,
        run_id: str,
        *,
        experimental_record_verifier: Callable[[EvidenceRecord], bool] | None = None,
    ) -> JsonDiscoveryStore:
        store = cls(
            root=root,
            run_id=run_id,
            experimental_record_verifier=experimental_record_verifier,
        )
        if not store.checkpoint_path.exists():
            raise FileNotFoundError(f"no checkpoint found for run {run_id!r}")
        store._checkpoint = RunCheckpoint.model_validate_json(
            store.checkpoint_path.read_text(encoding="utf-8")
        )
        return store

    @staticmethod
    def _safe_run_id(run_id: str) -> str:
        if (
            not run_id
            or run_id in {".", ".."}
            or run_id.endswith((".", " "))
            or any(
                ch
                not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-"
                for ch in run_id
            )
        ):
            raise ValueError("run_id contains unsafe path characters")
        reserved = {"CON", "PRN", "AUX", "NUL", "CLOCK$"}
        reserved.update({f"COM{index}" for index in range(1, 10)})
        reserved.update({f"LPT{index}" for index in range(1, 10)})
        if run_id.split(".", 1)[0].upper() in reserved:
            raise ValueError("run_id uses a reserved Windows device name")
        return run_id

    def create_state(self, goal: DiscoveryGoal) -> DiscoveryState:
        if self._checkpoint is not None:
            if self._checkpoint.goal.goal_id != goal.goal_id:
                raise ValueError("resumed store goal does not match requested goal")
            return self._checkpoint.state.model_copy(deep=True)
        if self.checkpoint_path.exists():
            raise FileExistsError(
                f"checkpoint already exists for {self.run_id!r}; use JsonDiscoveryStore.resume()"
            )
        now = utc_now()
        state = DiscoveryState(run_id=self.run_id, goal_id=goal.goal_id, cycle=0)
        self._checkpoint = RunCheckpoint(
            run_id=self.run_id,
            goal=goal.model_copy(deep=True),
            state=state,
            created_at=now,
            updated_at=now,
        )
        self._commit("run_created", {"goal_id": goal.goal_id})
        return state.model_copy(deep=True)

    def save_hypotheses(self, batch: HypothesisBatch) -> None:
        checkpoint = self._required()
        known = {item.hypothesis_id for item in checkpoint.hypotheses}
        additions = [
            item.model_copy(deep=True)
            for item in batch.hypotheses
            if item.hypothesis_id not in known
        ]
        checkpoint.hypotheses.extend(additions)
        checkpoint.state.hypothesis_ids = list(
            dict.fromkeys([*checkpoint.state.hypothesis_ids, *(item.hypothesis_id for item in additions)])
        )
        self._commit("hypotheses_saved", {"ids": [item.hypothesis_id for item in additions]})

    def save_candidate_plan(self, plan: CandidatePlan) -> None:
        self._required().candidate_plans.append(plan.model_copy(deep=True))
        self._commit("candidate_plan_saved", {"task_count": len(plan.tasks)})

    def save_candidates(self, batch: CandidateBatch) -> None:
        checkpoint = self._required()
        existing = {item.candidate_id: item for item in checkpoint.candidates}
        for candidate in batch.candidates:
            previous = existing.get(candidate.candidate_id)
            if previous is not None:
                if stable_hash(previous) != stable_hash(candidate):
                    raise ValueError(f"candidate_id {candidate.candidate_id!r} is immutable")
                continue
            missing_parents = set(candidate.parent_candidate_ids) - set(existing)
            if missing_parents:
                raise ValueError(
                    f"candidate {candidate.candidate_id!r} has unknown parents: {sorted(missing_parents)}"
                )
            if candidate.parent_candidate_ids:
                if candidate.candidate_ref is None:
                    raise ValueError("revised candidates require an immutable CandidateRef")
                parent_versions = [
                    existing[parent_id].candidate_ref.version
                    for parent_id in candidate.parent_candidate_ids
                    if existing[parent_id].candidate_ref is not None
                ]
                if parent_versions and candidate.candidate_ref.version <= max(parent_versions):
                    raise ValueError(
                        "revised candidate version must be greater than every parent version"
                    )
            stored_candidate = candidate.model_copy(deep=True)
            checkpoint.candidates.append(stored_candidate)
            existing[candidate.candidate_id] = stored_candidate
        checkpoint.state.candidate_ids = [item.candidate_id for item in checkpoint.candidates]
        self._commit("candidates_saved", {"ids": [item.candidate_id for item in batch.candidates]})

    def save_predictions(self, batch: PredictionBatch) -> None:
        checkpoint = self._required()
        by_id = {item.candidate_id: item for item in checkpoint.predictions}
        for prediction in batch.predictions:
            by_id[prediction.candidate_id] = prediction.model_copy(deep=True)
        checkpoint.predictions = list(by_id.values())
        self._commit("predictions_saved", {"candidate_ids": list(by_id)})

    def save_validation_plan(self, plan: ValidationPlan) -> None:
        self._required().validation_plans.append(plan.model_copy(deep=True))
        self._commit("validation_plan_saved", {"call_ids": [call.call_id for call in plan.calls]})

    def save_evidence(self, batch: EvidenceBatch) -> None:
        checkpoint = self._required()
        candidates = {item.candidate_id: item for item in checkpoint.candidates}
        known = {item.evidence_id: item for item in checkpoint.evidence}
        for record in batch.records:
            if record.evidence_kind == "experimental" and record.status == "success":
                if (
                    self.experimental_record_verifier is None
                    or not self.experimental_record_verifier(record)
                ):
                    raise ValueError(
                        "successful experimental evidence is not trusted by the configured store verifier"
                    )
            candidate = candidates.get(record.candidate_id)
            if candidate is None:
                raise ValueError(f"evidence references unknown candidate {record.candidate_id!r}")
            if record.candidate_ref is not None:
                if candidate.candidate_ref is None or record.candidate_ref != candidate.candidate_ref:
                    raise ValueError(
                        f"evidence {record.evidence_id!r} references a stale candidate version"
                    )
            previous = known.get(record.evidence_id)
            if previous is not None:
                if stable_hash(previous) != stable_hash(record):
                    raise ValueError(
                        f"evidence_id {record.evidence_id!r} is immutable and has conflicting content"
                    )
            else:
                stored_record = record.model_copy(deep=True)
                checkpoint.evidence.append(stored_record)
                known[record.evidence_id] = stored_record
        checkpoint.state.evidence_ids = [item.evidence_id for item in checkpoint.evidence]
        self._commit("evidence_saved", {"ids": [item.evidence_id for item in batch.records]})

    def save_analysis(self, analysis: ResultAnalysis) -> None:
        checkpoint = self._required()
        checkpoint.analyses.append(analysis.model_copy(deep=True))
        checkpoint.state.rejected_candidate_ids = list(
            dict.fromkeys([*checkpoint.state.rejected_candidate_ids, *analysis.candidates_to_remove])
        )
        self._commit(
            "analysis_saved",
            {"next_action": analysis.next_recommended_action},
        )

    def save_revision_plan(self, plan: RevisionPlan) -> None:
        self._required().revision_plans.append(plan.model_copy(deep=True))
        self._commit("revision_plan_saved", {"revision_count": len(plan.revisions)})

    def save_stop_decision(self, decision: StopDecision) -> None:
        self._required().stop_decisions.append(decision.model_copy(deep=True))
        self._commit("stop_decision_saved", {"stop": decision.stop, "reason": decision.reason_code})

    def save_assessments(self, assessments: list[CandidateValidationAssessment]) -> None:
        self._required().assessments = [item.model_copy(deep=True) for item in assessments]
        self._commit("assessments_saved", {"candidate_ids": [item.candidate_id for item in assessments]})

    def build_next_state(self) -> DiscoveryState:
        checkpoint = self._required()
        checkpoint.state.cycle += 1
        self._commit("cycle_advanced", {"cycle": checkpoint.state.cycle})
        return checkpoint.state.model_copy(deep=True)

    def history_summary(self) -> DiscoveryHistorySummary:
        checkpoint = self._required()
        failed = sum(record.status in {"failed", "timeout"} for record in checkpoint.evidence)
        experimental_candidates = {
            record.candidate_id for record in checkpoint.evidence if record.evidence_kind == "experimental"
        }
        latest_analysis = checkpoint.analyses[-1] if checkpoint.analyses else None
        return DiscoveryHistorySummary(
            run_id=self.run_id,
            completed_cycles=checkpoint.state.cycle,
            generated_candidate_count=len(checkpoint.candidates),
            evaluated_candidate_count=len({record.candidate_id for record in checkpoint.evidence}),
            experimental_candidate_count=len(experimental_candidates),
            failed_call_count=failed,
            key_findings=(latest_analysis.confirmed_findings if latest_analysis else []),
            rejected_assumptions=(latest_analysis.rejected_assumptions if latest_analysis else []),
            unresolved_questions=[],
            best_candidate_ids=(latest_analysis.candidates_to_keep if latest_analysis else []),
            aggregate_metrics={"evidence_records": len(checkpoint.evidence)},
        )

    def candidates(self) -> list[Candidate]:
        return [item.model_copy(deep=True) for item in self._required().candidates]

    def latest_candidates(self) -> list[Candidate]:
        """Return candidates created by the most recently saved generation plan."""

        checkpoint = self._required()
        if not checkpoint.candidate_plans:
            return []
        task_ids = {task.task_id for task in checkpoint.candidate_plans[-1].tasks if task.task_id}
        selected = [item for item in checkpoint.candidates if item.generation_task_id in task_ids]
        result = selected or list(checkpoint.candidates)
        return [item.model_copy(deep=True) for item in result]

    def evidence_for(self, candidate_id: str) -> list[EvidenceRecord]:
        return [
            item.model_copy(deep=True)
            for item in self._required().evidence
            if item.candidate_id == candidate_id
        ]

    def latest_predictions(self) -> list[CandidatePrediction]:
        return [item.model_copy(deep=True) for item in self._required().predictions]

    def _required(self) -> RunCheckpoint:
        if self._checkpoint is None:
            raise RuntimeError("create_state must be called before using the store")
        return self._checkpoint

    def _commit(self, event_type: str, details: dict[str, Any]) -> None:
        checkpoint = self._required()
        checkpoint.updated_at = utc_now()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = checkpoint.model_dump_json(indent=2).encode("utf-8")
        fd, temp_name = tempfile.mkstemp(prefix=".checkpoint.", suffix=".tmp", dir=self.run_dir)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self.checkpoint_path)
        finally:
            if os.path.exists(temp_name):
                os.unlink(temp_name)
        event = {
            "at": checkpoint.updated_at,
            "event": event_type,
            "run_id": self.run_id,
            "cycle": checkpoint.state.cycle,
            "details": details,
            "checkpoint_hash": stable_hash(checkpoint),
        }
        with self.audit_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


__all__ = ["JsonDiscoveryStore", "RunCheckpoint"]
