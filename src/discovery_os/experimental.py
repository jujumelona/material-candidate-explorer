"""Human-controlled import path for laboratory evidence.

The discovery model cannot invoke this module through ToolRuntime. A caller
must explicitly construct a signed/traceable submission and provide attachment
bytes; adapters never execute instrument commands.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable
import json

from pydantic import AwareDatetime, Field, model_validator

from .artifacts import ArtifactStore
from .hashing import bytes_hash, stable_hash
from .schemas import (
    Candidate,
    EvidenceBatch,
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    EvidenceVerification,
    ExperimentalEvidenceDetails,
    Fidelity,
    MethodClass,
    PropertyResult,
    StrictSchema,
    JsonObject,
    VerificationStatus,
)


class ExperimentalEvidenceSubmission(StrictSchema):
    submission_id: str = Field(min_length=1, max_length=256)
    source_id: str = Field(min_length=1, max_length=256)
    protocol_id: str = Field(min_length=1, max_length=256)
    sample_id: str = Field(min_length=1, max_length=256)
    laboratory: str = Field(min_length=1, max_length=256)
    operation: str = Field(min_length=1, max_length=256)
    method_class: MethodClass
    properties: list[PropertyResult] = Field(min_length=1)
    instrument: str = Field(min_length=1, max_length=512)
    operator: str = Field(min_length=1, max_length=256)
    replicate_id: str = Field(min_length=1, max_length=256)
    controls: list[str] = Field(min_length=1)
    blinded: bool | None = None
    conditions: JsonObject = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)
    observed_at: AwareDatetime
    attestation_payload: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def _properties_have_predeclared_decisions(self) -> ExperimentalEvidenceSubmission:
        for result in self.properties:
            if result.meets_criterion is None or not result.criterion:
                raise ValueError(
                    "experimental properties require meets_criterion and a predeclared criterion"
                )
        return self


@runtime_checkable
class ExperimentalEvidenceVerifier(Protocol):
    """Application-owned verifier for signatures, lab identity, and raw data."""

    def verify(
        self,
        submission: ExperimentalEvidenceSubmission,
        attachment_hashes: dict[str, str],
    ) -> EvidenceVerification:
        ...


class ExperimentalEvidenceImporter:
    _ALLOWED_METHODS = {
        MethodClass.ANALYTICAL_MEASUREMENT,
        MethodClass.BIOASSAY,
        MethodClass.MATERIALS_CHARACTERIZATION,
        MethodClass.ELECTROCHEMICAL_TEST,
        MethodClass.OTHER,
    }

    def __init__(
        self,
        artifact_store: ArtifactStore,
        *,
        verifier: ExperimentalEvidenceVerifier
        | Callable[[ExperimentalEvidenceSubmission, dict[str, str]], EvidenceVerification]
        | None = None,
    ) -> None:
        self.artifact_store = artifact_store
        self.verifier = verifier
        self._verified_record_hashes: dict[str, str] = {}

    def import_submission(
        self,
        submission: ExperimentalEvidenceSubmission,
        candidate: Candidate,
        *,
        attachments: dict[str, bytes] | None = None,
    ) -> EvidenceBatch:
        if candidate.candidate_ref is None:
            raise ValueError("laboratory evidence requires an immutable CandidateRef")
        if submission.method_class not in self._ALLOWED_METHODS:
            raise ValueError("computational method classes cannot be imported as experiments")
        artifact_paths: list[str] = []
        attachment_hashes: dict[str, str] = {}
        normalized_names: dict[str, str] = {}
        prepared_attachments: list[tuple[str, str, bytes]] = []
        for name, payload in sorted((attachments or {}).items()):
            safe_name = self.artifact_store.safe_component(name)
            previous_name = normalized_names.get(safe_name)
            if previous_name is not None and previous_name != name:
                raise ValueError(
                    f"attachment names {previous_name!r} and {name!r} normalize to the same safe name"
                )
            normalized_names[safe_name] = name
            prepared_attachments.append((name, safe_name, payload))
        for _name, safe_name, payload in prepared_attachments:
            content_digest = bytes_hash(payload)
            relative, digest = self.artifact_store.write_bytes(
                (
                    f"experimental/{self.artifact_store.safe_component(submission.submission_id)}/"
                    f"{content_digest}-{safe_name}"
                ),
                payload,
            )
            artifact_paths.append(relative)
            attachment_hashes[relative] = digest
        verification = self._verify(submission, attachment_hashes)
        verified = verification.status == VerificationStatus.VERIFIED
        evidence_status = (
            EvidenceStatus.SUCCESS
            if verified
            else (
                EvidenceStatus.FAILED
                if verification.status == VerificationStatus.REJECTED
                else EvidenceStatus.PARTIAL
            )
        )
        input_hash = stable_hash(
            {
                "submission": submission,
                "candidate_ref": candidate.candidate_ref,
                "attachments": attachment_hashes,
            }
        )
        output_hash = stable_hash(
            {
            "properties": submission.properties,
            "attachments": attachment_hashes,
            "verification": verification,
            }
        )
        record = EvidenceRecord(
            evidence_id=f"EVD-LAB-{stable_hash([submission.submission_id, candidate.candidate_id, output_hash])[:16]}",
            call_id=f"LAB-{submission.submission_id}",
            candidate_id=candidate.candidate_id,
            candidate_ref=candidate.candidate_ref,
            tool_name="lab_import",
            tool_version="1.0",
            operation=submission.operation,
            method_class=submission.method_class,
            status=evidence_status,
            evidence_kind=EvidenceKind.EXPERIMENTAL,
            fidelity=Fidelity.EXPERIMENTAL,
            properties=submission.properties,
            warnings=list(
                dict.fromkeys(
                    [
                        *submission.warnings,
                        *(
                            []
                            if verified
                            else [
                                "experimental submission is unverified and cannot satisfy validation gates"
                            ]
                        ),
                    ]
                )
            ),
            artifact_paths=artifact_paths,
            runtime_seconds=0.0,
            input_hash=input_hash,
            output_hash=output_hash,
            source_id=submission.source_id,
            experimental_details=ExperimentalEvidenceDetails(
                protocol_id=submission.protocol_id,
                sample_id=submission.sample_id,
                laboratory=submission.laboratory,
                instrument=submission.instrument,
                operator=submission.operator,
                replicate_id=submission.replicate_id,
                controls=submission.controls,
                blinded=submission.blinded,
                conditions=submission.conditions,
            ),
            verification=verification,
            observed_at=submission.observed_at,
        )
        if verified:
            record_hash = stable_hash(record)
            self._verified_record_hashes[record.evidence_id] = record_hash
            self._persist_attestation(
                record,
                record_hash,
                attachment_hashes,
                submission.attestation_payload,
            )
        return EvidenceBatch(records=[record], batch_id=f"LAB-BATCH-{submission.submission_id}")

    def verify_record(self, record: EvidenceRecord) -> bool:
        """Revalidate an emitted record and every content-addressed raw artifact."""

        record_hash = stable_hash(record)
        expected = self._verified_record_hashes.get(record.evidence_id)
        attestation = self._load_attestation(record, record_hash)
        if expected is None and attestation is not None:
            expected = str(attestation.get("record_hash", ""))
            self._verified_record_hashes[record.evidence_id] = expected
        if expected is None or record_hash != expected or attestation is None:
            return False
        if record.evidence_kind != EvidenceKind.EXPERIMENTAL or record.status != EvidenceStatus.SUCCESS:
            return False
        attested_artifacts = attestation.get("attachment_hashes")
        if not isinstance(attested_artifacts, dict):
            return False
        if set(attested_artifacts) != set(record.artifact_paths):
            return False
        for relative in record.artifact_paths:
            try:
                path = self.artifact_store.resolve(relative)
                claimed_digest = path.name.split("-", 1)[0].lower()
                if (
                    len(claimed_digest) != 64
                    or any(ch not in "0123456789abcdef" for ch in claimed_digest)
                    or not path.is_file()
                    or bytes_hash(path.read_bytes()) != claimed_digest
                    or attested_artifacts.get(relative) != claimed_digest
                ):
                    return False
            except (OSError, ValueError):
                return False
        return bool(record.artifact_paths)

    def _attestation_path(self, evidence_id: str, record_hash: str) -> str:
        safe_id = self.artifact_store.safe_component(evidence_id)
        return f"experimental/attestations/{safe_id}-{record_hash}.json"

    def _persist_attestation(
        self,
        record: EvidenceRecord,
        record_hash: str,
        attachment_hashes: dict[str, str],
        attestation_payload: JsonObject,
    ) -> None:
        self.artifact_store.write_json(
            self._attestation_path(record.evidence_id, record_hash),
            {
                "format_version": "1",
                "evidence_id": record.evidence_id,
                "record_hash": record_hash,
                "verification": record.verification.model_dump(mode="json"),
                "attestation_payload": attestation_payload,
                "attachment_hashes": attachment_hashes,
            },
        )

    def _load_attestation(
        self,
        record: EvidenceRecord,
        record_hash: str,
    ) -> dict[str, object] | None:
        try:
            path = self.artifact_store.resolve(
                self._attestation_path(record.evidence_id, record_hash)
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                return None
            if (
                payload.get("format_version") != "1"
                or payload.get("evidence_id") != record.evidence_id
                or payload.get("record_hash") != record_hash
                or payload.get("verification")
                != record.verification.model_dump(mode="json")
            ):
                return None
            return payload
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def _verify(
        self,
        submission: ExperimentalEvidenceSubmission,
        attachment_hashes: dict[str, str],
    ) -> EvidenceVerification:
        if not attachment_hashes:
            return EvidenceVerification(
                status=VerificationStatus.UNVERIFIED,
                reason="At least one immutable raw-data attachment is required.",
            )
        if self.verifier is None:
            return EvidenceVerification(
                status=VerificationStatus.UNVERIFIED,
                reason="No application-owned laboratory attestation verifier is configured.",
            )
        try:
            if hasattr(self.verifier, "verify"):
                result = self.verifier.verify(submission, attachment_hashes)  # type: ignore[union-attr]
            else:
                result = self.verifier(submission, attachment_hashes)  # type: ignore[operator]
            if not isinstance(result, EvidenceVerification):
                raise TypeError("verifier must return EvidenceVerification")
            if result.status == VerificationStatus.NOT_APPLICABLE:
                raise ValueError("experimental verifier returned not_applicable")
            return result
        except Exception as exc:
            return EvidenceVerification(
                status=VerificationStatus.REJECTED,
                reason=f"Verifier rejected the submission: {type(exc).__name__}: {str(exc)[:500]}",
            )


__all__ = [
    "ExperimentalEvidenceImporter",
    "ExperimentalEvidenceSubmission",
    "ExperimentalEvidenceVerifier",
]
