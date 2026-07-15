from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from discovery_os.artifacts import ArtifactStore
from discovery_os.store import JsonDiscoveryStore
from discovery_os.experimental import (
    ExperimentalEvidenceImporter,
    ExperimentalEvidenceSubmission,
)
from discovery_os.profiles import (
    EvidenceRequirement,
    ValidationGate,
    ValidationGateEvaluator,
    ValidationProfile,
    get_validation_profile,
)
from discovery_os.schemas import (
    CandidateValidationStatus,
    CandidateBatch,
    CandidateRef,
    ApplicabilityAssessment,
    ClaimLevel,
    ComputationalEvidenceDetails,
    DiscoveryDomain,
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceVerification,
    ExperimentalEvidenceDetails,
    Fidelity,
    MethodClass,
    PropertyResult,
    VerificationStatus,
)


def _evidence(
    evidence_id: str,
    *,
    kind: EvidenceKind,
    fidelity: Fidelity,
    source_id: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        call_id=f"CALL-{evidence_id}",
        candidate_id="CANDIDATE-1",
        candidate_ref=CandidateRef(
            candidate_id="CANDIDATE-1",
            version=1,
            content_hash="f" * 64,
        ),
        tool_name="fixture_tool",
        tool_version="1.0",
        operation="measure_target",
        method_class=(
            MethodClass.MATERIALS_CHARACTERIZATION
            if kind == EvidenceKind.EXPERIMENTAL
            else MethodClass.RULE_BASED
        ),
        status=EvidenceStatus.SUCCESS,
        evidence_kind=kind,
        fidelity=fidelity,
        properties=[
            PropertyResult(
                property_name="target_property",
                value=True,
                meets_criterion=True,
            )
        ],
        runtime_seconds=0.0,
        input_hash=f"input-{evidence_id}",
        output_hash=f"output-{evidence_id}",
        source_id=source_id,
        computational_details=(
            ComputationalEvidenceDetails(
                method_name="fixture_tool",
                method_version="1.0",
            )
            if kind == EvidenceKind.COMPUTATIONAL
            else None
        ),
        experimental_details=(
            ExperimentalEvidenceDetails(
                protocol_id="TEST-PROTOCOL",
                sample_id="TEST-SAMPLE",
                laboratory=source_id or "TEST-LAB",
                instrument="TEST-INSTRUMENT",
                operator="TEST-OPERATOR",
                replicate_id=evidence_id,
                controls=["positive-control"],
                conditions={"temperature_k": 298.15},
            )
            if kind == EvidenceKind.EXPERIMENTAL
            else None
        ),
        verification=(
            EvidenceVerification(
                status=VerificationStatus.VERIFIED,
                verifier_id="test-verifier",
                attestation_id=f"attestation-{evidence_id}",
                method="test-signature",
                reason="Verified fixture evidence.",
            )
            if kind == EvidenceKind.EXPERIMENTAL
            else EvidenceVerification(
                status=VerificationStatus.NOT_APPLICABLE,
                reason="Computational fixture.",
            )
        ),
        artifact_paths=(
            [f"experimental/{evidence_id}/raw.csv"]
            if kind == EvidenceKind.EXPERIMENTAL
            else []
        ),
        observed_at=(
            datetime.now(UTC) if kind == EvidenceKind.EXPERIMENTAL else None
        ),
    )


def _three_level_profile() -> ValidationProfile:
    return ValidationProfile(
        profile_id="test-three-level-v1",
        profile_version="1.0",
        domain=DiscoveryDomain.GENERAL_MATERIALS,
        name="Three-level evidence boundary",
        description="Computational plausibility, observation, then replication.",
        gates=[
            ValidationGate(
                gate_id="computational",
                name="Computational plausibility",
                description="A cheap computational screen passes.",
                claim_level=ClaimLevel.COMPUTATIONALLY_PLAUSIBLE,
                requirements=[
                    EvidenceRequirement(
                        requirement_id="computed-target",
                        description="Target property passes a computation.",
                        evidence_kind=EvidenceKind.COMPUTATIONAL,
                        property_names=["target_property"],
                        minimum_fidelity=Fidelity.CHEAP,
                    )
                ],
            ),
            ValidationGate(
                gate_id="experimental",
                name="Experimental observation",
                description="The target is measured experimentally.",
                claim_level=ClaimLevel.EXPERIMENTALLY_OBSERVED,
                requirements=[
                    EvidenceRequirement(
                        requirement_id="observed-target",
                        description="Target property passes an experiment.",
                        evidence_kind=EvidenceKind.EXPERIMENTAL,
                        property_names=["target_property"],
                        minimum_fidelity=Fidelity.EXPERIMENTAL,
                    )
                ],
            ),
            ValidationGate(
                gate_id="replication",
                name="Independent replication",
                description="Two independent sources reproduce the result.",
                claim_level=ClaimLevel.INDEPENDENTLY_REPLICATED,
                requirements=[
                    EvidenceRequirement(
                        requirement_id="replicated-target",
                        description="Two explicit sources report the target.",
                        evidence_kind=EvidenceKind.EXPERIMENTAL,
                        property_names=["target_property"],
                        minimum_fidelity=Fidelity.EXPERIMENTAL,
                        minimum_records=2,
                        minimum_independent_sources=2,
                    )
                ],
            ),
        ],
    )


def test_cheap_computational_records_cannot_cross_experimental_claim_boundary() -> None:
    records = [
        _evidence(
            "EVD-COMP-1",
            kind=EvidenceKind.COMPUTATIONAL,
            fidelity=Fidelity.CHEAP,
            source_id="SOURCE-A",
        ),
        _evidence(
            "EVD-COMP-2",
            kind=EvidenceKind.COMPUTATIONAL,
            fidelity=Fidelity.CHEAP,
            source_id="SOURCE-B",
        ),
    ]

    assessment = ValidationGateEvaluator().evaluate(
        _three_level_profile(), "CANDIDATE-1", records
    )

    assert assessment.status == CandidateValidationStatus.COMPUTATIONALLY_SUPPORTED
    assert assessment.claim_level == ClaimLevel.COMPUTATIONALLY_PLAUSIBLE
    assert assessment.matched_evidence_kinds == [EvidenceKind.COMPUTATIONAL]
    assert assessment.gate_decisions[0].passed
    assert not assessment.gate_decisions[1].passed
    assert not assessment.gate_decisions[2].passed


def test_conflicting_positive_and_negative_evidence_stays_inconclusive() -> None:
    positive = _evidence(
        "EVD-POSITIVE",
        kind=EvidenceKind.COMPUTATIONAL,
        fidelity=Fidelity.CHEAP,
    )
    negative = _evidence(
        "EVD-NEGATIVE",
        kind=EvidenceKind.COMPUTATIONAL,
        fidelity=Fidelity.CHEAP,
    )
    negative.properties[0].meets_criterion = False

    decision = ValidationGateEvaluator().evaluate_gate(
        _three_level_profile().gates[0], [positive, negative]
    )

    assert not decision.passed
    assert decision.status == "insufficient_evidence"
    assert "Contradictory" in decision.reason
    assert set(decision.matched_evidence_ids) == {
        "EVD-POSITIVE",
        "EVD-NEGATIVE",
    }


def test_schema_rejects_computational_evidence_labeled_experimental() -> None:
    with pytest.raises(ValidationError, match="computational evidence cannot use experimental fidelity"):
        _evidence(
            "EVD-MISLABELED",
            kind=EvidenceKind.COMPUTATIONAL,
            fidelity=Fidelity.EXPERIMENTAL,
        )


def test_computational_evidence_requires_method_provenance() -> None:
    payload = _evidence(
        "EVD-COMP-PROVENANCE",
        kind=EvidenceKind.COMPUTATIONAL,
        fidelity=Fidelity.CHEAP,
    ).model_dump(mode="json")
    payload["computational_details"] = None

    with pytest.raises(ValidationError, match="requires computational_details"):
        EvidenceRecord.model_validate(payload)


def test_successful_experimental_evidence_requires_verified_attestation() -> None:
    payload = _evidence(
        "EVD-EXP-VERIFY",
        kind=EvidenceKind.EXPERIMENTAL,
        fidelity=Fidelity.EXPERIMENTAL,
        source_id="LAB-A",
    ).model_dump(mode="json")
    payload["verification"] = {
        "schema_version": "1.0",
        "status": "unverified",
        "reason": "No verifier.",
    }

    with pytest.raises(ValidationError, match="must be verified"):
        EvidenceRecord.model_validate(payload)


def test_evidence_rejects_artifact_path_traversal() -> None:
    payload = _evidence(
        "EVD-PATH",
        kind=EvidenceKind.COMPUTATIONAL,
        fidelity=Fidelity.CHEAP,
    ).model_dump(mode="json")
    payload["artifact_paths"] = ["../../outside.txt"]

    with pytest.raises(ValidationError, match="confined relative paths"):
        EvidenceRecord.model_validate(payload)


def test_high_fidelity_computation_needs_reproducibility_and_convergence() -> None:
    requirement = EvidenceRequirement(
        requirement_id="high-quality",
        description="High-fidelity physics result with convergence provenance.",
        evidence_kind=EvidenceKind.COMPUTATIONAL,
        property_names=["target_property"],
        minimum_fidelity=Fidelity.HIGH,
    )
    record = _evidence(
        "EVD-HIGH",
        kind=EvidenceKind.COMPUTATIONAL,
        fidelity=Fidelity.HIGH,
    )
    evaluator = ValidationGateEvaluator()

    assert not evaluator.evaluate_requirement(requirement, [record]).satisfied

    record.parameters_hash = "parameters-hash"
    record.convergence_checks = {"energy": True, "forces": True}
    record.computational_details.code_revision = "code-revision"
    assert evaluator.evaluate_requirement(requirement, [record]).satisfied


def test_property_level_out_of_domain_result_cannot_pass_a_gate() -> None:
    requirement = EvidenceRequirement(
        requirement_id="applicability",
        description="Only in-domain predictions qualify.",
        evidence_kind=EvidenceKind.COMPUTATIONAL,
        property_names=["target_property"],
        minimum_fidelity=Fidelity.CHEAP,
    )
    record = _evidence(
        "EVD-OOD",
        kind=EvidenceKind.COMPUTATIONAL,
        fidelity=Fidelity.CHEAP,
    )
    record.properties[0].applicability = ApplicabilityAssessment(
        in_domain=False,
        reasons=["Outside the declared composition range."],
    )

    decision = ValidationGateEvaluator().evaluate_requirement(requirement, [record])
    assert not decision.satisfied
    assert decision.status == "insufficient_evidence"


def _submission(submission_id: str, source_id: str) -> ExperimentalEvidenceSubmission:
    return ExperimentalEvidenceSubmission(
        submission_id=submission_id,
        source_id=source_id,
        protocol_id="BIOASSAY-PROTOCOL-1",
        sample_id=f"SAMPLE-{submission_id}",
        laboratory=source_id,
        operation="target_activity_assay",
        method_class=MethodClass.BIOASSAY,
        properties=[
            PropertyResult(
                property_name="target_activity",
                value=0.8,
                meets_criterion=True,
                criterion="Predeclared assay threshold",
            )
        ],
        replicate_id=f"REPLICATE-{submission_id}",
        instrument="ASSAY-READER-1",
        operator="TEST-OPERATOR",
        controls=["positive-control", "negative-control"],
        conditions={"temperature_k": 298.15},
        observed_at=datetime.now(UTC),
    )


def test_lab_submission_requires_an_explicit_source_id() -> None:
    payload = _submission("ONE", "LAB-A").model_dump(mode="json")
    payload.pop("source_id")

    with pytest.raises(ValidationError, match="source_id"):
        ExperimentalEvidenceSubmission.model_validate(payload)


def test_independent_replication_needs_two_distinct_explicit_lab_sources(
    tmp_path, candidate_factory
) -> None:
    candidate = candidate_factory()
    def verifier(submission, _hashes):
        return EvidenceVerification(
            status=VerificationStatus.VERIFIED,
            verifier_id="test-verifier",
            attestation_id=f"attestation-{submission.submission_id}",
            method="test-signature",
            reason="Trusted test verifier accepted immutable raw data.",
        )

    importer = ExperimentalEvidenceImporter(
        ArtifactStore(tmp_path / "lab-artifacts"), verifier=verifier
    )
    first = importer.import_submission(
        _submission("ONE", "LAB-A"), candidate, attachments={"raw.csv": b"one"}
    )
    same_source = importer.import_submission(
        _submission("TWO", "LAB-A"), candidate, attachments={"raw.csv": b"two"}
    )
    independent_source = importer.import_submission(
        _submission("THREE", "LAB-B"),
        candidate,
        attachments={"raw.csv": b"three"},
    )
    profile = get_validation_profile(DiscoveryDomain.MEDICINAL_CHEMISTRY)
    replication_gate = next(
        gate for gate in profile.gates if gate.gate_id == "med-replication"
    )
    evaluator = ValidationGateEvaluator(
        experimental_record_verifier=importer.verify_record
    )

    duplicated = evaluator.evaluate_gate(
        replication_gate, [*first.records, *same_source.records]
    )
    independent = evaluator.evaluate_gate(
        replication_gate, [*first.records, *independent_source.records]
    )

    assert not duplicated.passed
    assert "at least 2 explicitly identified independent source(s)" in (
        duplicated.requirement_decisions[0].missing
    )
    assert independent.passed
    assert {first.records[0].source_id, independent_source.records[0].source_id} == {
        "LAB-A",
        "LAB-B",
    }


def test_unverified_lab_submission_cannot_satisfy_an_experimental_gate(
    tmp_path, candidate_factory
) -> None:
    candidate = candidate_factory()
    importer = ExperimentalEvidenceImporter(ArtifactStore(tmp_path / "unverified"))
    batch = importer.import_submission(
        _submission("UNVERIFIED", "LAB-A"),
        candidate,
        attachments={"raw.csv": b"unverified"},
    )

    assert batch.records[0].status == EvidenceStatus.PARTIAL
    assert batch.records[0].verification.status == VerificationStatus.UNVERIFIED
    gate = next(
        gate
        for gate in get_validation_profile(DiscoveryDomain.MEDICINAL_CHEMISTRY).gates
        if gate.gate_id == "med-replication"
    )
    assert not ValidationGateEvaluator().evaluate_gate(gate, batch.records).passed


def test_store_and_gate_require_the_configured_importer_attestation(
    tmp_path, candidate_factory, goal_factory
) -> None:
    candidate = candidate_factory()

    def verifier(submission, _hashes):
        return EvidenceVerification(
            status=VerificationStatus.VERIFIED,
            verifier_id="authorized-lab-registry",
            attestation_id=f"signed-{submission.submission_id}",
            method="test-signature",
            reason="Authorized source and raw data hashes verified.",
        )

    importer = ExperimentalEvidenceImporter(
        ArtifactStore(tmp_path / "trusted-artifacts"), verifier=verifier
    )
    batch = importer.import_submission(
        _submission("TRUSTED", "LAB-A"),
        candidate,
        attachments={"raw.csv": b"trusted raw data"},
    )

    untrusted_store = JsonDiscoveryStore(tmp_path / "untrusted-store")
    untrusted_store.create_state(goal_factory())
    untrusted_store.save_candidates(CandidateBatch(candidates=[candidate]))
    with pytest.raises(ValueError, match="not trusted"):
        untrusted_store.save_evidence(batch)

    trusted_store = JsonDiscoveryStore(
        tmp_path / "trusted-store",
        experimental_record_verifier=importer.verify_record,
    )
    trusted_store.create_state(goal_factory())
    trusted_store.save_candidates(CandidateBatch(candidates=[candidate]))
    trusted_store.save_evidence(batch)
    assert trusted_store.checkpoint.evidence[0].evidence_id == batch.records[0].evidence_id

    restarted_importer = ExperimentalEvidenceImporter(
        ArtifactStore(tmp_path / "trusted-artifacts")
    )
    assert restarted_importer.verify_record(batch.records[0])
    resumed = JsonDiscoveryStore.resume(
        tmp_path / "trusted-store",
        trusted_store.run_id,
        experimental_record_verifier=restarted_importer.verify_record,
    )
    gate = next(
        gate
        for gate in get_validation_profile(DiscoveryDomain.MEDICINAL_CHEMISTRY).gates
        if gate.gate_id == "med-replication"
    )
    resumed_decision = ValidationGateEvaluator(
        experimental_record_verifier=restarted_importer.verify_record
    ).evaluate_gate(gate, resumed.checkpoint.evidence)
    assert not resumed_decision.passed
    assert resumed_decision.requirement_decisions[0].matched_evidence_ids == [
        batch.records[0].evidence_id
    ]
    assert not restarted_importer.verify_record(
        batch.records[0].model_copy(update={"output_hash": "tampered-output"})
    )
    raw_path = restarted_importer.artifact_store.resolve(
        batch.records[0].artifact_paths[0]
    )
    raw_path.write_bytes(b"tampered raw data")
    assert not restarted_importer.verify_record(batch.records[0])


def test_importer_rejects_colliding_sanitized_attachment_names(
    tmp_path, candidate_factory
) -> None:
    candidate = candidate_factory()
    importer = ExperimentalEvidenceImporter(ArtifactStore(tmp_path / "collisions"))

    with pytest.raises(ValueError, match="normalize to the same safe name"):
        importer.import_submission(
            _submission("COLLISION", "LAB-A"),
            candidate,
            attachments={"raw data.csv": b"one", "raw@data.csv": b"two"},
        )
