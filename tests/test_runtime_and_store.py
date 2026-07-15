from __future__ import annotations

import pytest

from discovery_os.artifacts import ArtifactStore
from discovery_os.cache import JsonCache
from discovery_os.runtime import ToolRuntime
from discovery_os.registry import ToolRegistry
from discovery_os.schemas import (
    CandidateBatch,
    CandidateRef,
    ComputationalEvidenceDetails,
    EvidenceBatch,
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    Fidelity,
    MethodClass,
    ToolDescriptor,
    ToolOperationDescriptor,
    ValidationPlan,
)
from discovery_os.store import JsonDiscoveryStore
from discovery_os.tool_adapters import build_default_tool_registry


def test_deterministic_tool_runtime_uses_content_cache(
    tmp_path, candidate_factory, tool_call_factory
) -> None:
    candidate = candidate_factory()
    registry = build_default_tool_registry(include_placeholders=False)
    runtime = ToolRuntime(
        registry,
        ArtifactStore(tmp_path / "artifacts"),
        JsonCache(tmp_path / "cache"),
    )
    first_call = tool_call_factory(
        call_id="CALL-CACHE-FIRST", candidate_ids=[candidate.candidate_id]
    )
    second_call = first_call.model_copy(update={"call_id": "CALL-CACHE-SECOND"})

    first = runtime.execute_plan(
        ValidationPlan(
            calls=[first_call],
            expected_information_gain={},
            plan_reason="Populate the deterministic cache.",
        ),
        candidates=[candidate],
    )
    second = runtime.execute_plan(
        ValidationPlan(
            calls=[second_call],
            expected_information_gain={},
            plan_reason="Reuse the deterministic cache.",
        ),
        candidates=[candidate],
    )

    assert "cache_hit" not in first.warnings
    assert "cache_hit" in second.warnings
    assert second.records[0].runtime_seconds == first.records[0].runtime_seconds
    assert second.records[0].call_id == second_call.call_id
    assert second.records[0].candidate_ref == candidate.candidate_ref


def test_checkpoint_can_resume_complete_state(tmp_path, candidate_factory, goal_factory) -> None:
    store = JsonDiscoveryStore(tmp_path, run_id="RUN-RESUME")
    goal = goal_factory()
    candidate = candidate_factory()
    store.create_state(goal)
    store.save_candidates(CandidateBatch(candidates=[candidate]))
    store.build_next_state()

    resumed = JsonDiscoveryStore.resume(tmp_path, "RUN-RESUME")

    assert resumed.checkpoint is not None
    assert resumed.checkpoint.goal == goal
    assert resumed.checkpoint.state.cycle == 1
    assert resumed.candidates() == [candidate]
    assert resumed.checkpoint_path.exists()
    assert resumed.audit_path.exists()


def test_store_rejects_evidence_for_a_stale_candidate_ref(
    tmp_path, candidate_factory, goal_factory
) -> None:
    store = JsonDiscoveryStore(tmp_path, run_id="RUN-STALE")
    candidate = candidate_factory()
    store.create_state(goal_factory())
    store.save_candidates(CandidateBatch(candidates=[candidate]))
    stale_ref = CandidateRef(
        candidate_id=candidate.candidate_id,
        version=candidate.candidate_ref.version + 1,
        content_hash="f" * 64,
    )
    stale_record = EvidenceRecord(
        evidence_id="EVD-STALE",
        call_id="CALL-STALE",
        candidate_id=candidate.candidate_id,
        candidate_ref=stale_ref,
        tool_name="common_rules",
        tool_version="1.0",
        operation="validate_candidate",
        method_class=MethodClass.RULE_BASED,
        status=EvidenceStatus.SUCCESS,
        evidence_kind=EvidenceKind.COMPUTATIONAL,
        fidelity=Fidelity.CHEAP,
        properties=[],
        runtime_seconds=0.0,
        input_hash="input-hash",
        output_hash="output-hash",
        computational_details=ComputationalEvidenceDetails(
            method_name="common_rules",
            method_version="1.0",
        ),
    )

    with pytest.raises(ValueError, match="stale candidate version"):
        store.save_evidence(EvidenceBatch(records=[stale_record]))

    assert store.checkpoint is not None
    assert store.checkpoint.evidence == []


def test_store_rejects_conflicting_content_for_the_same_evidence_id(
    tmp_path, candidate_factory, goal_factory
) -> None:
    store = JsonDiscoveryStore(tmp_path, run_id="RUN-EVIDENCE-IMMUTABLE")
    candidate = candidate_factory()
    store.create_state(goal_factory())
    store.save_candidates(CandidateBatch(candidates=[candidate]))
    record = EvidenceRecord(
        evidence_id="EVD-IMMUTABLE",
        call_id="CALL-1",
        candidate_id=candidate.candidate_id,
        candidate_ref=candidate.candidate_ref,
        tool_name="common_rules",
        tool_version="1.0",
        operation="validate_candidate",
        method_class=MethodClass.RULE_BASED,
        status=EvidenceStatus.SUCCESS,
        evidence_kind=EvidenceKind.COMPUTATIONAL,
        fidelity=Fidelity.CHEAP,
        properties=[],
        runtime_seconds=0.0,
        input_hash="input-hash",
        output_hash="output-one",
        computational_details=ComputationalEvidenceDetails(
            method_name="common_rules",
            method_version="1.0",
        ),
    )
    store.save_evidence(EvidenceBatch(records=[record]))
    record.warnings.append("caller-side mutation")
    assert store.checkpoint.evidence[0].warnings == []
    snapshot = store.checkpoint
    snapshot.evidence[0].warnings.append("snapshot mutation")
    assert store.checkpoint.evidence[0].warnings == []

    with pytest.raises(ValueError, match="conflicting content"):
        store.save_evidence(
            EvidenceBatch(
                records=[record.model_copy(update={"output_hash": "output-two"})]
            )
        )


def test_artifact_store_never_overwrites_different_content(tmp_path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    first_path, first_hash = artifacts.write_bytes("raw/result.bin", b"first")
    same_path, same_hash = artifacts.write_bytes("raw/result.bin", b"first")

    assert (first_path, first_hash) == (same_path, same_hash)
    with pytest.raises(FileExistsError, match="immutable artifact"):
        artifacts.write_bytes("raw/result.bin", b"different")


def test_nontrivial_available_tool_requires_a_terminating_timeout_contract() -> None:
    class UnsafeLongRunningAdapter:
        descriptor = ToolDescriptor(
            tool_name="unsafe_long_tool",
            tool_version="1",
            adapter_version="1",
            description="A test adapter without a killable timeout contract.",
            operations=[
                ToolOperationDescriptor(
                    operation="simulate",
                    description="Potentially long simulation.",
                    supported_domains=["general_materials"],
                    supported_candidate_types=["composition"],
                    method_class="physics_simulation",
                    evidence_kinds=["computational"],
                    supported_fidelities=["high"],
                )
            ],
            available=True,
        )

        def run(self, call, candidates):  # pragma: no cover - registration rejects it
            raise AssertionError("must not run")

        def normalize(self, call, candidates, raw_result, runtime_seconds):
            raise AssertionError("must not normalize")

    with pytest.raises(ValueError, match="run_with_timeout"):
        ToolRegistry().register(UnsafeLongRunningAdapter())


@pytest.mark.parametrize("run_id", [".", "..", "CON", "nul.txt", "RUN."])
def test_store_rejects_unsafe_or_reserved_run_ids(tmp_path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        JsonDiscoveryStore(tmp_path, run_id=run_id)
