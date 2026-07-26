"""Fail-closed receipts for external scientific validators.

The routing profiles name calculations that may be executed outside this
repository.  A validator name or a provenance label is not evidence that the
calculation ran.  These contracts bind a reported property to one candidate,
one condition set, one method policy, immutable inputs and outputs, and an
explicit convergence or experimental quality-control decision.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .fusion_schemas import ContentArtifactRef
from .hashing import stable_hash
from .schemas import Identifier, MaterialField, NonEmptyText, StrictSchema
from .specialist_workflows import (
    missing_specialist_condition_fields,
    specialist_workflow_policy,
    specialist_workflow_policy_sha256,
)


Sha256 = str


class SpecialistOutputEvidence(StrictSchema):
    """A semantic output role bound to one immutable workflow artifact."""

    evidence_label: Identifier
    artifact_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    parser_locator: NonEmptyText


class ScientificGateReceipt(StrictSchema):
    """Result of one policy-required scientific validity gate."""

    gate_id: Identifier
    status: Literal["passed", "failed"]
    evidence_labels: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def _evidence_labels_are_unique(self) -> "ScientificGateReceipt":
        if len(self.evidence_labels) != len(set(self.evidence_labels)):
            raise ValueError("scientific-gate evidence labels must be unique")
        return self


class SpecialistExecutionReceipt(StrictSchema):
    """Structured evidence emitted by an external numerical or lab workflow.

    ``completed`` means that the workflow process and its parser completed.  It
    does not by itself make a numerical result scientifically usable:
    convergence (or experimental quality control) is checked separately by
    :meth:`permits_property_value`.
    """

    receipt_id: Identifier
    material_field: MaterialField
    validator_id: Identifier
    validator_contract_version: Identifier
    workflow_policy_id: Identifier
    workflow_policy_version: Identifier
    execution_kind: Literal["numerical_simulation", "experimental_measurement"]
    candidate_id: Identifier
    candidate_input_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    property_name: Identifier
    unit: str = Field(min_length=1, max_length=256)
    conditions: dict[str, JsonValue] = Field(default_factory=dict)
    conditions_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    method_family: Identifier
    method_id: Identifier
    method_version: Identifier
    method_policy_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    input_manifest_sha256: Sha256 = Field(pattern=r"^[0-9a-f]{64}$")
    workflow_engine: Identifier
    workflow_version: Identifier
    workflow_code_revision: Identifier
    execution_status: Literal["completed", "failed", "cancelled", "partial"]
    workflow_exit_code: int | None = None
    parser_status: Literal["success", "failed", "not_run"]
    output_artifacts: list[ContentArtifactRef] = Field(default_factory=list)
    primary_output_sha256: Sha256 | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    convergence_status: Literal["passed", "failed", "not_applicable"]
    convergence_evidence_artifacts: list[ContentArtifactRef] = Field(
        default_factory=list
    )
    output_evidence: list[SpecialistOutputEvidence] = Field(default_factory=list)
    scientific_gate_receipts: list[ScientificGateReceipt] = Field(
        default_factory=list
    )
    quality_control_passed: bool
    provenance_id: Identifier

    @model_validator(mode="after")
    def _receipt_is_internally_bound(self) -> "SpecialistExecutionReceipt":
        policy = specialist_workflow_policy(
            self.material_field,
            self.property_name,
            self.validator_id,
        )
        if self.workflow_policy_id != policy.policy_id:
            raise ValueError(
                "receipt workflow_policy_id does not match the code-owned policy"
            )
        if self.workflow_policy_version != policy.policy_version:
            raise ValueError(
                "receipt workflow_policy_version does not match the code-owned policy"
            )
        if self.validator_contract_version != policy.validator_contract_version:
            raise ValueError(
                "receipt validator_contract_version does not match the code-owned policy"
            )
        if self.unit != policy.unit:
            raise ValueError(
                "receipt unit does not match the code-owned workflow policy"
            )
        if self.execution_kind != policy.execution_kind:
            raise ValueError(
                "receipt execution_kind does not match the code-owned workflow policy"
            )
        if not policy.permits_method(
            method_family=self.method_family,
            method_id=self.method_id,
        ):
            raise ValueError(
                "receipt method family/id is not allowed by the code-owned policy"
            )
        if self.method_policy_sha256 != specialist_workflow_policy_sha256(policy):
            raise ValueError(
                "receipt method_policy_sha256 does not bind the code-owned policy"
            )
        missing_conditions = missing_specialist_condition_fields(
            policy,
            self.conditions,
        )
        if missing_conditions:
            raise ValueError(
                "receipt is missing policy-required scientific conditions: "
                + ", ".join(missing_conditions)
            )
        if self.conditions_sha256 != stable_hash(self.conditions):
            raise ValueError(
                "receipt conditions_sha256 must bind the exact condition set"
            )

        output_paths = [item.relative_path for item in self.output_artifacts]
        output_hashes = [item.sha256 for item in self.output_artifacts]
        all_artifact_hashes = {
            *output_hashes,
            *(item.sha256 for item in self.convergence_evidence_artifacts),
        }
        convergence_paths = [
            item.relative_path for item in self.convergence_evidence_artifacts
        ]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("receipt output artifact paths must be unique")
        if len(convergence_paths) != len(set(convergence_paths)):
            raise ValueError(
                "receipt convergence-evidence artifact paths must be unique"
            )
        evidence_labels = [
            item.evidence_label for item in self.output_evidence
        ]
        if len(evidence_labels) != len(set(evidence_labels)):
            raise ValueError("receipt output-evidence labels must be unique")
        if any(
            item.artifact_sha256 not in all_artifact_hashes
            for item in self.output_evidence
        ):
            raise ValueError(
                "receipt output evidence must bind a declared immutable artifact"
            )
        gate_ids = [item.gate_id for item in self.scientific_gate_receipts]
        if len(gate_ids) != len(set(gate_ids)):
            raise ValueError("receipt scientific-gate identifiers must be unique")
        unknown_gate_ids = set(gate_ids) - set(policy.required_scientific_gate_ids)
        if unknown_gate_ids:
            raise ValueError(
                "receipt contains scientific gates outside the code-owned policy: "
                + ", ".join(sorted(unknown_gate_ids))
            )
        if any(
            evidence_label not in set(evidence_labels)
            for gate in self.scientific_gate_receipts
            for evidence_label in gate.evidence_labels
        ):
            raise ValueError(
                "scientific-gate evidence must reference declared output-evidence labels"
            )

        if self.execution_status == "completed":
            if self.workflow_exit_code != 0:
                raise ValueError(
                    "completed specialist execution requires workflow_exit_code=0"
                )
            if self.parser_status != "success":
                raise ValueError(
                    "completed specialist execution requires a successful parser"
                )
            if not self.output_artifacts or self.primary_output_sha256 is None:
                raise ValueError(
                    "completed specialist execution requires immutable output artifacts"
                )
            if self.primary_output_sha256 not in output_hashes:
                raise ValueError(
                    "primary_output_sha256 must identify a declared output artifact"
                )
            missing_outputs = set(
                policy.required_output_evidence_labels
            ) - set(evidence_labels)
            if missing_outputs:
                raise ValueError(
                    "completed receipt is missing policy-required output evidence: "
                    + ", ".join(sorted(missing_outputs))
                )
            missing_gates = set(policy.required_scientific_gate_ids) - set(gate_ids)
            if missing_gates:
                raise ValueError(
                    "completed receipt is missing policy-required scientific gates: "
                    + ", ".join(sorted(missing_gates))
                )
        else:
            if self.primary_output_sha256 is not None:
                raise ValueError(
                    "non-completed specialist execution cannot declare a primary output"
                )
            if self.quality_control_passed:
                raise ValueError(
                    "non-completed specialist execution cannot pass quality control"
                )
            if self.convergence_status == "passed":
                raise ValueError(
                    "non-completed specialist execution cannot pass convergence"
                )

        if self.execution_kind == "numerical_simulation":
            if self.convergence_status != "passed" and self.quality_control_passed:
                raise ValueError(
                    "numerical quality control cannot pass without convergence"
                )
        elif self.convergence_status != "not_applicable":
            raise ValueError(
                "experimental receipts must mark numerical convergence not_applicable"
            )

        if (
            self.quality_control_passed
            and not self.convergence_evidence_artifacts
        ):
            raise ValueError(
                "passed validation requires immutable convergence or quality-control evidence"
            )
        gate_statuses = {
            item.gate_id: item.status for item in self.scientific_gate_receipts
        }
        all_required_gates_passed = all(
            gate_statuses.get(gate_id) == "passed"
            for gate_id in policy.required_scientific_gate_ids
        )
        if self.quality_control_passed and not all_required_gates_passed:
            raise ValueError(
                "quality_control_passed requires every policy scientific gate to pass"
            )
        return self

    def permits_property_value(self) -> bool:
        """Return whether this receipt can authorize a property observation."""

        if (
            self.execution_status != "completed"
            or self.workflow_exit_code != 0
            or self.parser_status != "success"
            or not self.quality_control_passed
            or self.primary_output_sha256 is None
            or not self.output_artifacts
            or not self.convergence_evidence_artifacts
        ):
            return False
        try:
            policy = specialist_workflow_policy(
                self.material_field,
                self.property_name,
                self.validator_id,
            )
        except ValueError:
            return False
        if (
            self.workflow_policy_id != policy.policy_id
            or self.workflow_policy_version != policy.policy_version
            or self.validator_contract_version
            != policy.validator_contract_version
            or self.unit != policy.unit
            or self.execution_kind != policy.execution_kind
            or not policy.permits_method(
                method_family=self.method_family,
                method_id=self.method_id,
            )
            or self.method_policy_sha256
            != specialist_workflow_policy_sha256(policy)
            or missing_specialist_condition_fields(policy, self.conditions)
        ):
            return False
        output_labels = {
            item.evidence_label for item in self.output_evidence
        }
        gate_statuses = {
            item.gate_id: item.status for item in self.scientific_gate_receipts
        }
        if not set(policy.required_output_evidence_labels).issubset(output_labels):
            return False
        if not all(
            gate_statuses.get(gate_id) == "passed"
            for gate_id in policy.required_scientific_gate_ids
        ):
            return False
        if self.execution_kind == "numerical_simulation":
            return self.convergence_status == "passed"
        return self.convergence_status == "not_applicable"


def specialist_execution_receipt_sha256(
    receipt: SpecialistExecutionReceipt,
) -> str:
    """Return the canonical hash used to bind an observation to its receipt."""

    return stable_hash(receipt.model_dump(mode="json"))


__all__ = [
    "ScientificGateReceipt",
    "SpecialistExecutionReceipt",
    "SpecialistOutputEvidence",
    "specialist_execution_receipt_sha256",
]
