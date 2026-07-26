from __future__ import annotations

from typing import Any

import pytest

from discovery_os.material_applications import (
    MainModelMaterialApplicationClassifier,
    application_roles_for_field,
    build_material_application_brief,
    get_application_role_profile,
)
from discovery_os.material_domains import MATERIAL_FIELD_PROFILES
from discovery_os.schemas import MaterialField


class _ApplicationModel:
    model_id = "role-criteria-fixture"
    model_version = "1"

    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision

    def complete_json(self, *, operation: str, system: str, user: str) -> Any:
        assert operation == "classify-material-application"
        assert "allowed_roles" in user
        return self.decision


def _decision(
    *,
    evidence_span: str = "n-type thermoelectric",
    extracted_context: dict[str, object] | None = None,
    confidence: float = 0.9,
    application_subtype: str | None = None,
) -> dict[str, object]:
    return {
        "question_kind": "component_selection",
        "selected_role_ids": ["thermoelectric_n_type_leg"],
        "application_subtype": application_subtype,
        "extracted_context": extracted_context or {},
        "objective_priorities": [],
        "confidence": confidence,
        "evidence_spans": [evidence_span],
        "needs_clarification": False,
        "clarification_question": None,
        "decision_summary": "Select the allowlisted n-type thermoelectric role.",
        "endpoint_or_tool_selection_performed": False,
    }


def _classifier(decision: dict[str, object]) -> MainModelMaterialApplicationClassifier:
    return MainModelMaterialApplicationClassifier(_ApplicationModel(decision))


def test_every_non_semiconductor_role_adds_closed_role_specific_criteria() -> None:
    for field in MaterialField:
        if field == MaterialField.SEMICONDUCTOR:
            continue
        field_properties = {
            requirement.property_name
            for requirement in MATERIAL_FIELD_PROFILES[field].properties
        }
        for role in application_roles_for_field(field):
            supplements = [
                criterion
                for criterion in role.criteria
                if criterion.property_name not in field_properties
            ]
            assert len(supplements) >= 2, role.role_id
            for criterion in supplements:
                assert criterion.criterion_id.startswith(f"{role.role_id}-")
                assert criterion.unit.strip()
                assert criterion.required_context
                assert criterion.validator_ids
                assert criterion.preferred_calculations
                assert criterion.experimental_confirmation
                assert criterion.scientific_caution.strip()
                assert criterion.literature_or_mcp_can_score is False


@pytest.mark.parametrize(
    (
        "field",
        "role_id",
        "property_name",
        "unit",
        "direction",
        "required_context",
        "validator_id",
    ),
    [
        (
            MaterialField.GENERAL_INORGANIC,
            "general_thermal_management",
            "thermal_boundary_conductance",
            "MW/(m^2 K)",
            "maximize",
            "interface_stack",
            "tdtr-interface-measurement",
        ),
        (
            MaterialField.BATTERY_ELECTRODE,
            "battery_positive_electrode_active",
            "thermal_runaway_onset_temperature",
            "K",
            "maximize",
            "state_of_charge",
            "accelerating-rate-calorimetry",
        ),
        (
            MaterialField.SOLID_ELECTROLYTE,
            "solid_electrolyte_interface_buffer",
            "area_specific_resistance",
            "ohm cm^2",
            "minimize",
            "stack_pressure",
            "interface-impedance-measurement",
        ),
        (
            MaterialField.SUPERCONDUCTOR,
            "high_field_magnet_conductor",
            "critical_current_density",
            "A/mm^2",
            "maximize",
            "electric_field_criterion",
            "four-probe-critical-current-test",
        ),
        (
            MaterialField.HETEROGENEOUS_CATALYST,
            "heterogeneous_catalyst_active_phase",
            "turnover_frequency",
            "s^-1",
            "maximize",
            "coverage",
            "site-normalized-rate-measurement",
        ),
        (
            MaterialField.PHOTOVOLTAIC_ABSORBER,
            "photovoltaic_tandem_top_absorber",
            "current_matching_error",
            "fraction",
            "minimize",
            "bottom_cell",
            "subcell-eqe-measurement",
        ),
        (
            MaterialField.THERMOELECTRIC,
            "thermoelectric_contact_or_interconnect",
            "specific_contact_resistivity",
            "ohm cm^2",
            "minimize",
            "bonding_process",
            "contact-resistivity-measurement",
        ),
        (
            MaterialField.MAGNETIC_MATERIAL,
            "permanent_magnet",
            "maximum_energy_product",
            "kJ/m^3",
            "maximize",
            "microstructure",
            "closed-loop-hysteresis-measurement",
        ),
        (
            MaterialField.FERROELECTRIC_PIEZOELECTRIC,
            "nonvolatile_ferroelectric_memory",
            "endurance_cycles",
            "cycle",
            "maximize",
            "pulse_protocol",
            "switched-polarization-endurance-test",
        ),
        (
            MaterialField.STRUCTURAL_ALLOY,
            "high_temperature_load_bearing",
            "creep_rupture_life",
            "h",
            "maximize",
            "stress",
            "creep-rupture-test",
        ),
        (
            MaterialField.POROUS_FRAMEWORK,
            "carbon_capture",
            "regeneration_energy",
            "kJ/mol",
            "minimize",
            "regeneration_method",
            "process-level-carbon-capture-workflow",
        ),
    ],
)
def test_representative_role_criterion_contracts_are_exact(
    field: MaterialField,
    role_id: str,
    property_name: str,
    unit: str,
    direction: str,
    required_context: str,
    validator_id: str,
) -> None:
    role = get_application_role_profile(field, role_id)
    criterion = next(
        item for item in role.criteria if item.property_name == property_name
    )

    assert criterion.unit == unit
    assert criterion.direction == direction
    assert required_context in criterion.required_context
    assert validator_id in criterion.validator_ids
    assert criterion.literature_or_mcp_can_score is False


def test_each_role_uses_exact_stage_order_source_policy_and_route_capabilities() -> None:
    expected_stages = [
        "generation_prior",
        "identity_novelty",
        "mlip_disagreement",
        "relaxation_validation",
        "dft_handoff",
    ]
    for field in MaterialField:
        route_by_stage = {
            route.stage: route
            for route in MATERIAL_FIELD_PROFILES[field].stage_routes
        }
        for role in application_roles_for_field(field):
            assert [
                task.evidence_stage for task in role.evidence_tasks
            ] == expected_stages
            for task in role.evidence_tasks:
                assert task.allowed_literature_sources == (
                    ["crossref", "arxiv", "openalex"]
                    if task.evidence_stage
                    in {"generation_prior", "identity_novelty"}
                    else ["crossref", "arxiv"]
                )
                assert task.mcp_capabilities == list(
                    route_by_stage[task.evidence_stage].mcp_capabilities
                )
                assert task.can_create_property_scores is False
                assert task.prompt_or_model_can_choose_mcp_tool is False


def test_thermal_spreader_alias_routes_to_semiconductor_thermal_role() -> None:
    brief = build_material_application_brief(
        "Compare a gate dielectric and thermal spreader.",
        material_field=MaterialField.SEMICONDUCTOR,
    )

    assert [role.role_id for role in brief.roles] == [
        "gate_dielectric",
        "thermal_spreader_or_substrate",
    ]


def test_model_can_extract_allowlisted_literal_temperature_and_brief_merges_it() -> None:
    question = "Find an n-type thermoelectric leg at 700 K."
    run = _classifier(
        _decision(extracted_context={"hot_side_temperature": 700})
    ).classify(
        question,
        material_field=MaterialField.THERMOELECTRIC,
    )

    brief = build_material_application_brief(
        question,
        material_field=MaterialField.THERMOELECTRIC,
        application_model_run=run,
    )

    assert brief.target_context["hot_side_temperature"] == 700
    assert "hot_side_temperature" not in brief.missing_context_by_role[
        "thermoelectric_n_type_leg"
    ]
    assert brief.ready_for_condition_complete_scoring is False


@pytest.mark.parametrize(
    ("extracted_context", "message"),
    [
        ({"hallucinated_condition": 700}, "outside selected role requirements"),
        ({"hot_side_temperature": 900}, "literal question support"),
        ({"hot_side_temperature": {"value": 700}}, "literal question support"),
    ],
)
def test_model_rejects_hallucinated_context_keys_and_values(
    extracted_context: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _classifier(
            _decision(extracted_context=extracted_context)
        ).classify(
            "Find an n-type thermoelectric leg at 700 K.",
            material_field=MaterialField.THERMOELECTRIC,
        )


def test_model_existing_context_must_match_exactly() -> None:
    with pytest.raises(ValueError, match="exactly match input context"):
        _classifier(
            _decision(extracted_context={"hot_side_temperature": 701})
        ).classify(
            "Find an n-type thermoelectric leg at 700 K.",
            material_field=MaterialField.THERMOELECTRIC,
            problem_context={"hot_side_temperature": 700},
        )


def test_model_rejects_low_confidence_fake_subtype_and_trivial_span() -> None:
    question = "Find an n-type thermoelectric leg at 700 K."
    with pytest.raises(ValueError, match="below 0.70"):
        _classifier(_decision(confidence=0.01)).classify(
            question,
            material_field=MaterialField.THERMOELECTRIC,
        )
    with pytest.raises(ValueError, match="subtype outside the code allowlist"):
        _classifier(_decision(application_subtype="invented_subtype")).classify(
            question,
            material_field=MaterialField.THERMOELECTRIC,
        )
    with pytest.raises(ValueError, match="meaningfully support"):
        _classifier(_decision(evidence_span="at")).classify(
            question,
            material_field=MaterialField.THERMOELECTRIC,
        )


def test_brief_rejects_model_run_from_another_question() -> None:
    original = "Find an n-type thermoelectric leg at 700 K."
    run = _classifier(_decision()).classify(
        original,
        material_field=MaterialField.THERMOELECTRIC,
    )

    with pytest.raises(ValueError, match="stale"):
        build_material_application_brief(
            "Find an n-type thermoelectric leg at 800 K.",
            material_field=MaterialField.THERMOELECTRIC,
            application_model_run=run,
        )


def test_readiness_covers_all_required_ranking_context_and_preserves_portfolio() -> None:
    role = get_application_role_profile(
        MaterialField.THERMOELECTRIC,
        "thermoelectric_n_type_leg",
    )
    required = list(
        dict.fromkeys(
            [
                *role.required_problem_context,
                *[
                    name
                    for criterion in role.criteria
                    if criterion.required_for_ranking
                    for name in criterion.required_context
                ],
            ]
        )
    )
    context = {name: f"declared-{name}" for name in required}
    context.pop("band_gap_method")

    brief = build_material_application_brief(
        "Select an n-type thermoelectric leg.",
        material_field=MaterialField.THERMOELECTRIC,
        problem_context=context,
        explicit_role_ids=["thermoelectric_n_type_leg"],
    )

    assert brief.decomposition_mode == "single-role"
    assert brief.missing_context_by_role["thermoelectric_n_type_leg"] == [
        "band_gap_method"
    ]
    assert brief.ready_for_condition_complete_scoring is False

    broad = build_material_application_brief(
        "Which materials should be used for each battery electrode component?",
        material_field=MaterialField.BATTERY_ELECTRODE,
    )
    assert broad.decomposition_mode == "role-portfolio"
    assert broad.ready_for_condition_complete_scoring is False
    assert broad.clarification_question is None
