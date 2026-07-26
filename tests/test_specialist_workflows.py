from __future__ import annotations

import pytest

from discovery_os.fusion_schemas import ContentArtifactRef
from discovery_os.hashing import stable_hash
from discovery_os.material_domains import MATERIAL_FIELD_PROFILES
from discovery_os.schemas import MaterialField
from discovery_os.specialist_validation import (
    ScientificGateReceipt,
    SpecialistExecutionReceipt,
    SpecialistOutputEvidence,
)
from discovery_os.specialist_workflows import (
    SPECIALIST_WORKFLOW_POLICIES,
    SpecialistWorkflowPolicy,
    specialist_workflow_policy,
    specialist_workflow_policy_sha256,
)


def _conditions(policy: SpecialistWorkflowPolicy) -> dict[str, object]:
    return {name: f"test-{name}" for name in policy.required_condition_fields}


def _receipt_payload(policy: SpecialistWorkflowPolicy) -> dict[str, object]:
    conditions = _conditions(policy)
    result_sha = stable_hash(["result", policy.policy_id])
    qc_sha = stable_hash(["qc", policy.policy_id])
    method = policy.allowed_methods[0]
    return {
        "receipt_id": f"receipt-{policy.policy_id}",
        "material_field": MaterialField(str(policy.material_field)),
        "validator_id": policy.validator_id,
        "validator_contract_version": policy.validator_contract_version,
        "workflow_policy_id": policy.policy_id,
        "workflow_policy_version": policy.policy_version,
        "execution_kind": policy.execution_kind,
        "candidate_id": "candidate-policy-test",
        "candidate_input_sha256": stable_hash(["candidate-policy-test"]),
        "property_name": policy.property_name,
        "unit": policy.unit,
        "conditions": conditions,
        "conditions_sha256": stable_hash(conditions),
        "method_family": method.method_family,
        "method_id": method.method_id,
        "method_version": "test-method-version",
        "method_policy_sha256": specialist_workflow_policy_sha256(policy),
        "input_manifest_sha256": stable_hash(["input-manifest", policy.policy_id]),
        "workflow_engine": "test-specialist-engine",
        "workflow_version": "test-engine-version",
        "workflow_code_revision": "test-code-revision",
        "execution_status": "completed",
        "workflow_exit_code": 0,
        "parser_status": "success",
        "output_artifacts": [
            ContentArtifactRef(
                artifact_id=f"output-{policy.policy_id}",
                relative_path=f"external/{policy.policy_id}/result.json",
                sha256=result_sha,
                media_type="application/json",
                byte_size=1,
            )
        ],
        "primary_output_sha256": result_sha,
        "convergence_status": (
            "not_applicable"
            if policy.execution_kind == "experimental_measurement"
            else "passed"
        ),
        "convergence_evidence_artifacts": [
            ContentArtifactRef(
                artifact_id=f"qc-{policy.policy_id}",
                relative_path=f"external/{policy.policy_id}/qc.json",
                sha256=qc_sha,
                media_type="application/json",
                byte_size=1,
            )
        ],
        "output_evidence": [
            SpecialistOutputEvidence(
                evidence_label=label,
                artifact_sha256=result_sha,
                parser_locator=f"$.{label}",
            )
            for label in policy.required_output_evidence_labels
        ],
        "scientific_gate_receipts": [
            ScientificGateReceipt(
                gate_id=gate_id,
                status="passed",
                evidence_labels=[policy.required_output_evidence_labels[0]],
            )
            for gate_id in policy.required_scientific_gate_ids
        ],
        "quality_control_passed": True,
        "provenance_id": "specialist-policy-test",
    }


def test_closed_registry_covers_every_required_property_in_all_twelve_profiles() -> None:
    assert len(MATERIAL_FIELD_PROFILES) == 12
    covered: set[tuple[MaterialField, str, str]] = set()
    for field, profile in MATERIAL_FIELD_PROFILES.items():
        for requirement in profile.properties:
            validators = [
                validator
                for route in profile.stage_routes
                for validator in route.validators
                if validator.can_create_property_scores
                and requirement.property_name in validator.properties
            ]
            assert validators
            for validator in validators:
                policy = specialist_workflow_policy(
                    field,
                    requirement.property_name,
                    validator.validator_id,
                )
                assert policy.unit == requirement.unit
                assert policy.required_condition_fields == requirement.required_context
                covered.add(
                    (field, requirement.property_name, validator.validator_id)
                )
    assert covered == set(SPECIALIST_WORKFLOW_POLICIES)


def test_every_code_owned_policy_can_issue_a_fully_bound_receipt() -> None:
    for policy in SPECIALIST_WORKFLOW_POLICIES.values():
        receipt = SpecialistExecutionReceipt.model_validate(
            _receipt_payload(policy),
            strict=True,
        )
        assert receipt.permits_property_value() is True


def test_old_generic_completed_receipt_cannot_authorize_a_property() -> None:
    policy = specialist_workflow_policy(
        MaterialField.GENERAL_INORGANIC,
        "energy_above_hull",
        "reference-phase-dft-and-phase-diagram",
    )
    payload = _receipt_payload(policy)
    for field in (
        "material_field",
        "workflow_policy_id",
        "workflow_policy_version",
        "method_family",
        "output_evidence",
        "scientific_gate_receipts",
    ):
        payload.pop(field)
    with pytest.raises(ValueError):
        SpecialistExecutionReceipt.model_validate(payload, strict=True)


@pytest.mark.parametrize(
    ("field", "property_name", "validator_id", "shortcut_method"),
    [
        (
            MaterialField.PHOTOVOLTAIC_ABSORBER,
            "slme",
            "quasiparticle-optics-and-slme",
            ("scalar_band_gap_proxy", "band-gap-only-slme"),
        ),
        (
            MaterialField.THERMOELECTRIC,
            "power_factor",
            "electronic-boltzmann-transport",
            ("constant_relaxation_time", "uncalibrated-constant-tau"),
        ),
        (
            MaterialField.MAGNETIC_MATERIAL,
            "ordering_temperature",
            "soc-anisotropy-exchange-temperature-workflow",
            ("zero_kelvin_dft", "single-order-energy-difference"),
        ),
        (
            MaterialField.FERROELECTRIC_PIEZOELECTRIC,
            "spontaneous_polarization",
            "berry-phase-switching-workflow",
            ("symmetry_only", "polar-space-group"),
        ),
        (
            MaterialField.STRUCTURAL_ALLOY,
            "service_degradation_rate",
            "elastic-defect-and-service-workflow",
            ("zero_kelvin_elasticity", "elastic-tensor-only"),
        ),
        (
            MaterialField.POROUS_FRAMEWORK,
            "adsorption_selectivity",
            "gcmc-mixture-adsorption-workflow",
            ("periodic_probe_geometry", "zeopp-accessible-volume"),
        ),
    ],
)
def test_scientific_shortcuts_are_rejected_by_exact_method_contract(
    field: MaterialField,
    property_name: str,
    validator_id: str,
    shortcut_method: tuple[str, str],
) -> None:
    policy = specialist_workflow_policy(field, property_name, validator_id)
    payload = _receipt_payload(policy)
    payload["method_family"], payload["method_id"] = shortcut_method
    with pytest.raises(ValueError, match="method family/id is not allowed"):
        SpecialistExecutionReceipt.model_validate(payload, strict=True)


def test_completed_receipt_requires_semantic_outputs_and_every_scientific_gate() -> None:
    policy = specialist_workflow_policy(
        MaterialField.SUPERCONDUCTOR,
        "critical_temperature",
        "epw-or-eliashberg-workflow",
    )
    payload = _receipt_payload(policy)
    payload["output_evidence"] = list(payload["output_evidence"])[:-1]
    with pytest.raises(
        ValueError,
        match="missing policy-required output evidence",
    ):
        SpecialistExecutionReceipt.model_validate(payload, strict=True)

    payload = _receipt_payload(policy)
    payload["scientific_gate_receipts"] = list(
        payload["scientific_gate_receipts"]
    )[1:]
    with pytest.raises(
        ValueError,
        match="missing policy-required scientific gates",
    ):
        SpecialistExecutionReceipt.model_validate(payload, strict=True)


def test_failed_required_scientific_gate_blocks_quality_control() -> None:
    policy = specialist_workflow_policy(
        MaterialField.POROUS_FRAMEWORK,
        "adsorption_selectivity",
        "gcmc-mixture-adsorption-workflow",
    )
    payload = _receipt_payload(policy)
    gates = list(payload["scientific_gate_receipts"])
    gates[0] = gates[0].model_copy(update={"status": "failed"})
    payload["scientific_gate_receipts"] = gates
    with pytest.raises(
        ValueError,
        match="every policy scientific gate to pass",
    ):
        SpecialistExecutionReceipt.model_validate(payload, strict=True)
