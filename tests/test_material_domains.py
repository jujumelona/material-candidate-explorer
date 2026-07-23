from __future__ import annotations

import json

import pytest

from discovery_os.cli import main
from discovery_os.material_domains import (
    MATERIAL_EVIDENCE_STAGES,
    MATERIAL_FIELD_PROFILES,
    MainModelMaterialFieldClassifier,
    MaterialPropertyObservation,
    assess_material_field_results,
    build_main_model_material_field_classifier_from_environment,
    build_material_domain_plan,
    get_material_field_profile,
    material_stage_route,
    resolve_material_field,
)
from discovery_os.schemas import MaterialField
from discovery_os.validation_evidence import (
    ValidationEvidenceRequest,
    ValidationEvidenceRouter,
    ValidationEvidenceStage,
    build_validation_evidence_prompt,
)


class _FieldModel:
    model_id = "main-science-router"
    model_version = "test-v1"

    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def complete_json(self, **_kwargs):
        return self.payload


def test_every_material_field_has_a_complete_fail_closed_route() -> None:
    canonical_units = {
        "eV/atom",
        "THz",
        "V",
        "mAh/g",
        "eV",
        "S/cm",
        "dimensionless",
        "h",
        "cm^2/(V s)",
        "cm^-1",
        "fraction",
        "s^-1",
        "W/(m K^2)",
        "W/(m K)",
        "MJ/m^3",
        "K",
        "C/m^2",
        "eV per formula unit",
        "pm/V",
        "GPa",
    }
    assert set(MATERIAL_FIELD_PROFILES) == set(MaterialField)
    for field, profile in MATERIAL_FIELD_PROFILES.items():
        assert profile.material_field == field
        assert tuple(item.stage for item in profile.stage_routes) == MATERIAL_EVIDENCE_STAGES
        assert profile.properties
        assert profile.research_reference_ids
        assert all(
            item.missing_result_policy == "unknown-not-pass"
            for item in profile.properties
        )
        assert all(item.unit in canonical_units for item in profile.properties)
        score_authorities = {
            property_name
            for route in profile.stage_routes
            for validator in route.validators
            if validator.can_create_property_scores
            for property_name in validator.properties
        }
        assert all(
            item.property_name in score_authorities
            for item in profile.properties
            if item.required_for_field_claim
        )
        for route in profile.stage_routes:
            assert route.can_steer_generation == (
                route.stage == "generation_prior"
            )
            assert route.property_score_created_by_route is False
            assert route.unknown_is_pass is False
            assert route.rag_questions
            assert route.validators
            assert all(
                item.result_if_not_executed == "unknown"
                for item in route.validators
            )


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("high ionic conductivity solid electrolyte", MaterialField.SOLID_ELECTROLYTE),
        ("ambient pressure Meissner superconductor", MaterialField.SUPERCONDUCTOR),
        ("CO2RR electrocatalyst surface", MaterialField.HETEROGENEOUS_CATALYST),
        ("high-ZT thermoelectric", MaterialField.THERMOELECTRIC),
        ("태양전지 광흡수체", MaterialField.PHOTOVOLTAIC_ABSORBER),
        ("고엔트로피 합금 크리프", MaterialField.STRUCTURAL_ALLOY),
        ("MOF gas adsorption", MaterialField.POROUS_FRAMEWORK),
    ],
)
def test_auto_field_resolution_is_deterministic(
    prompt: str,
    expected: MaterialField,
) -> None:
    first = resolve_material_field("AUTO", prompt=prompt)
    second = resolve_material_field("AUTO", prompt=prompt)
    assert first == second
    assert first.selected_field == expected
    assert first.selection_mode == "auto-keyword"
    assert first.requires_operator_choice is False


def test_explicit_field_overrides_prompt_and_unknown_field_fails() -> None:
    resolution = resolve_material_field(
        "battery_electrode",
        prompt="This prompt says superconductor and Meissner.",
    )
    assert resolution.selected_field == MaterialField.BATTERY_ELECTRODE
    assert resolution.selection_mode == "explicit"
    with pytest.raises(ValueError, match="unknown material field"):
        resolve_material_field("imaginary_material_field")


def test_main_ai_field_decision_is_evidence_verified_and_reconciled() -> None:
    classifier = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "solid_electrolyte",
                "secondary_fields": ["battery_electrode"],
                "application_subtype": "crystalline_solid_electrolyte",
                "confidence": 0.94,
                "evidence_spans": ["solid electrolyte", "ionic conductivity"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "The requested function is bulk ion transport.",
            }
        )
    )
    model_run = classifier.classify(
        "Find a solid electrolyte with high ionic conductivity.",
        chemical_system="Li-P-S",
    )
    resolution = resolve_material_field(
        "AUTO",
        prompt="Find a solid electrolyte with high ionic conductivity.",
        chemical_system="Li-P-S",
        model_run=model_run,
    )
    assert resolution.selection_mode == "auto-consensus"
    assert resolution.selected_field == MaterialField.SOLID_ELECTROLYTE
    assert resolution.secondary_fields == [MaterialField.BATTERY_ELECTRODE]
    assert resolution.application_subtype == "crystalline_solid_electrolyte"
    assert resolution.model_decision_id == model_run.decision_id
    assert model_run.endpoint_or_tool_selection_performed is False


def test_main_ai_can_resolve_keyword_default_but_conflict_needs_operator() -> None:
    model_run = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "photovoltaic_absorber",
                "secondary_fields": [],
                "application_subtype": "single_junction_absorber",
                "confidence": 0.91,
                "evidence_spans": ["light absorbing layer"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "The application is a solar absorber.",
            }
        )
    ).classify("Design a light absorbing layer.")
    resolved = resolve_material_field(
        "AUTO",
        prompt="Design a light absorbing layer.",
        model_run=model_run,
    )
    assert resolved.selection_mode == "auto-model"
    assert resolved.selected_field == MaterialField.PHOTOVOLTAIC_ABSORBER

    conflict_run = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "photovoltaic_absorber",
                "secondary_fields": [],
                "application_subtype": "single_junction_absorber",
                "confidence": 0.91,
                "evidence_spans": ["battery cathode"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "The model proposed a different application.",
            }
        )
    ).classify("Find a battery cathode.")
    conflict = resolve_material_field(
        "AUTO",
        prompt="Find a battery cathode.",
        model_run=conflict_run,
    )
    assert conflict.selection_mode == "auto-model-conflict"
    assert conflict.selected_field == MaterialField.GENERAL_INORGANIC
    assert conflict.requires_operator_choice is True


def test_main_ai_field_classifier_rejects_hallucinated_input_evidence() -> None:
    classifier = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "superconductor",
                "secondary_fields": [],
                "application_subtype": "electron_phonon",
                "confidence": 0.99,
                "evidence_spans": ["Meissner effect at 200 K"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "A superconductivity request.",
            }
        )
    )
    with pytest.raises(ValueError, match="evidence not present"):
        classifier.classify("Find an interesting hydrogen-rich crystal.")


def test_main_ai_field_classifier_rejects_coerced_confidence_and_chemical_only_evidence() -> None:
    boolean_confidence = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "superconductor",
                "secondary_fields": [],
                "application_subtype": "electron_phonon",
                "confidence": True,
                "evidence_spans": ["superconductor"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "Invalid boolean confidence.",
            }
        )
    )
    with pytest.raises(ValueError, match="invalid material-field decision"):
        boolean_confidence.classify("Find a superconductor.")

    chemical_only = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "superconductor",
                "secondary_fields": [],
                "application_subtype": "electron_phonon",
                "confidence": 0.99,
                "evidence_spans": ["Li-P-S"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "A composition does not establish an application.",
            }
        )
    )
    with pytest.raises(ValueError, match="chemical system alone"):
        chemical_only.classify(
            "Find a useful crystalline material.",
            chemical_system="Li-P-S",
        )


def test_main_ai_field_classifier_rejects_tool_selection_fields() -> None:
    classifier = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "superconductor",
                "secondary_fields": [],
                "application_subtype": "electron_phonon",
                "confidence": 0.95,
                "evidence_spans": ["superconductor"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "A superconducting-material request.",
                "mcp_tool": "run_dft",
            }
        )
    )
    with pytest.raises(ValueError, match="invalid material-field decision"):
        classifier.classify("Find a superconductor.")


def test_main_ai_field_run_is_bound_to_chemical_system_and_context() -> None:
    prompt = "Find a solid electrolyte with high ionic conductivity."
    model_run = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "solid_electrolyte",
                "secondary_fields": [],
                "application_subtype": "crystalline_solid_electrolyte",
                "confidence": 0.95,
                "evidence_spans": ["solid electrolyte"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "Bulk ion transport is the application.",
            }
        )
    ).classify(
        prompt,
        chemical_system="Li-P-S",
        problem_context={"temperature": 300},
    )
    with pytest.raises(ValueError, match="different classification inputs"):
        resolve_material_field(
            "AUTO",
            prompt=prompt,
            chemical_system="Li-La-Zr-O",
            problem_context={"temperature": 300},
            model_run=model_run,
        )
    with pytest.raises(ValueError, match="different classification inputs"):
        resolve_material_field(
            "AUTO",
            prompt=prompt,
            chemical_system="Li-P-S",
            problem_context={"temperature": 500},
            model_run=model_run,
        )


def test_main_ai_evidence_check_does_not_rewrite_quoted_punctuation() -> None:
    classifier = MainModelMaterialFieldClassifier(
        _FieldModel(
            {
                "primary_field": "solid_electrolyte",
                "secondary_fields": [],
                "application_subtype": "crystalline_solid_electrolyte",
                "confidence": 0.95,
                "evidence_spans": ["solid electrolyte"],
                "needs_clarification": False,
                "clarification_question": None,
                "decision_summary": "A solid-electrolyte request.",
            }
        )
    )
    with pytest.raises(ValueError, match="evidence not present"):
        classifier.classify("Find a solid-electrolyte.")


def test_main_ai_environment_builder_requires_a_complete_endpoint_pair() -> None:
    with pytest.raises(ValueError, match="must be configured together"):
        build_main_model_material_field_classifier_from_environment(
            environ={"MATERIAL_FIELD_MODEL_API_URL": "https://model.example/v1"},
        )
    classifier = build_main_model_material_field_classifier_from_environment(
        environ={
            "RAG_MODEL_API_URL": "https://fallback.example/v1",
            "RAG_MODEL_NAME": "fallback-model",
            "MATERIAL_FIELD_MODEL_API_URL": "https://field.example/v1",
            "MATERIAL_FIELD_MODEL_NAME": "field-router",
        },
        required=True,
    )
    assert classifier is not None
    assert classifier.model.model_id == "field-router"
    assert classifier.model.endpoint == "https://field.example/v1/chat/completions"
    with pytest.raises(ValueError, match="must be configured together"):
        build_main_model_material_field_classifier_from_environment(
            environ={
                "MATERIAL_FIELD_MODEL_API_URL": "https://field.example/v1",
                "RAG_MODEL_NAME": "must-not-cross-pair",
            },
        )


def test_auto_ambiguity_falls_back_without_specialized_claim() -> None:
    resolution = resolve_material_field(
        "AUTO",
        prompt="catalyst carrier mobility",
    )
    assert resolution.selection_mode == "auto-ambiguous"
    assert resolution.selected_field == MaterialField.GENERAL_INORGANIC
    assert resolution.requires_operator_choice is True
    assert set(resolution.ambiguous_fields) == {
        MaterialField.HETEROGENEOUS_CATALYST,
        MaterialField.SEMICONDUCTOR,
    }


def test_domain_plan_does_not_fabricate_unexecuted_properties() -> None:
    plan = build_material_domain_plan(
        MaterialField.THERMOELECTRIC,
        prompt="thermoelectric",
    )
    expected = [
        item.property_name
        for item in plan.profile.properties
        if item.required_for_field_claim
    ]
    assert plan.externally_reported_property_names == []
    assert plan.unexecuted_required_properties == expected
    assert plan.field_route_ready is False
    assert plan.missing_required_context == [
        "temperature",
        "carrier_concentration",
        "carrier_type",
        "microstructure",
    ]
    assert plan.scientific_status == "routing-plan-only-no-field-property-calculation"

    ready = build_material_domain_plan(
        MaterialField.THERMOELECTRIC,
        problem_context={
            "temperature": 800,
            "carrier_concentration": 1e19,
            "carrier_type": "n",
            "microstructure": "dense polycrystal",
        },
    )
    assert ready.field_route_ready is True
    assert ready.missing_required_context == []


def test_domain_plan_rejects_secret_or_conflicting_context() -> None:
    with pytest.raises(ValueError, match="cannot contain secrets"):
        build_material_domain_plan(
            MaterialField.GENERAL_INORGANIC,
            problem_context={"nested": {"client-secret": "must-not-persist"}},
        )
    with pytest.raises(ValueError, match="chemical_system conflicts"):
        build_material_domain_plan(
            MaterialField.GENERAL_INORGANIC,
            chemical_system="Li-O",
            problem_context={"chemical_system": "Fe-O"},
        )
    with pytest.raises(ValueError, match="cannot contain secrets"):
        MaterialPropertyObservation(
            observation_id="obs-secret",
            candidate_id="candidate-secret",
            material_field=MaterialField.GENERAL_INORGANIC,
            property_name="energy_above_hull",
            validator_id="reference-phase-dft-and-phase-diagram",
            status="success",
            value=0.01,
            unit="eV/atom",
            conditions={"Authorization": "must-not-persist"},
            provenance_id="prov-secret",
            raw_artifact_sha256="c" * 64,
            authority_kind="numerical_validator",
        )


def test_field_property_ranking_requires_named_validators_units_and_conditions() -> None:
    common = {
        "candidate_id": "candidate-te-1",
        "material_field": MaterialField.THERMOELECTRIC,
        "status": "success",
        "provenance_id": "prov-1",
        "raw_artifact_sha256": "a" * 64,
        "authority_kind": "numerical_validator",
    }
    assessment = assess_material_field_results(
        MaterialField.THERMOELECTRIC,
        candidate_id="candidate-te-1",
        target_conditions={
            "temperature": 800,
            "carrier_concentration": 1e19,
            "carrier_type": "n",
            "microstructure": "dense polycrystal",
        },
        observations=[
            MaterialPropertyObservation(
                **common,
                observation_id="obs-power",
                property_name="power_factor",
                validator_id="electronic-boltzmann-transport",
                value=0.004,
                unit="W/(m K^2)",
                conditions={
                    "temperature": 800,
                    "carrier_concentration": 1e19,
                    "carrier_type": "n",
                },
            ),
            MaterialPropertyObservation(
                **common,
                observation_id="obs-kappa",
                property_name="lattice_thermal_conductivity",
                validator_id="anharmonic-phonon-transport",
                value=1.1,
                unit="W/(m K)",
                conditions={
                    "temperature": 800,
                    "microstructure": "dense polycrystal",
                },
            ),
            MaterialPropertyObservation(
                **common,
                observation_id="obs-zt",
                property_name="zt",
                validator_id="thermoelectric-zt-integration",
                value=1.4,
                unit="dimensionless",
                conditions={
                    "temperature": 800,
                    "carrier_concentration": 1e19,
                    "microstructure": "dense polycrystal",
                },
            ),
        ],
    )
    assert assessment.ready_for_field_computational_ranking is True
    assert {item.status for item in assessment.decisions} == {"available"}
    assert assessment.claim_level == "computational-triage-only"
    assert assessment.missing_target_conditions == []

    wrong_unit = assess_material_field_results(
        MaterialField.THERMOELECTRIC,
        candidate_id="candidate-te-1",
        observations=[
            MaterialPropertyObservation(
                **common,
                observation_id="obs-wrong-unit",
                property_name="power_factor",
                validator_id="electronic-boltzmann-transport",
                value=4.0,
                unit="mW/(m K^2)",
                conditions={
                    "temperature": 800,
                    "carrier_concentration": 1e19,
                    "carrier_type": "n",
                },
            )
        ],
    )
    assert wrong_unit.ready_for_field_computational_ranking is False
    assert wrong_unit.decisions[0].status == "incomparable"
    assert wrong_unit.literature_or_mcp_property_substitution_performed is False


def test_field_property_assessment_does_not_merge_operating_conditions() -> None:
    common = {
        "candidate_id": "candidate-te-conditions",
        "material_field": MaterialField.THERMOELECTRIC,
        "property_name": "power_factor",
        "validator_id": "electronic-boltzmann-transport",
        "status": "success",
        "unit": "W/(m K^2)",
        "provenance_id": "prov-conditions",
        "raw_artifact_sha256": "b" * 64,
        "authority_kind": "numerical_validator",
    }
    assessment = assess_material_field_results(
        MaterialField.THERMOELECTRIC,
        candidate_id="candidate-te-conditions",
        target_conditions={
            "temperature": 800,
            "carrier_concentration": 1e19,
            "carrier_type": "n",
            "microstructure": "dense polycrystal",
        },
        observations=[
            MaterialPropertyObservation(
                **common,
                observation_id="obs-300k",
                value=0.002,
                conditions={
                    "temperature": 300,
                    "carrier_concentration": 1e19,
                    "carrier_type": "n",
                },
            ),
            MaterialPropertyObservation(
                **common,
                observation_id="obs-800k",
                value=0.004,
                conditions={
                    "temperature": 800,
                    "carrier_concentration": 1e19,
                    "carrier_type": "n",
                },
            ),
        ],
    )
    power_factor = assessment.decisions[0]
    assert power_factor.status == "available"
    assert power_factor.accepted_observation_ids == ["obs-800k"]
    assert power_factor.rejected_observation_ids == ["obs-300k"]
    assert power_factor.accepted_conditions["temperature"] == 800
    assert assessment.ready_for_field_computational_ranking is False

    without_target = assess_material_field_results(
        MaterialField.THERMOELECTRIC,
        candidate_id="candidate-te-conditions",
        observations=[
            MaterialPropertyObservation(
                **common,
                observation_id="obs-300k",
                value=0.002,
                conditions={
                    "temperature": 300,
                    "carrier_concentration": 1e19,
                    "carrier_type": "n",
                },
            ),
            MaterialPropertyObservation(
                **common,
                observation_id="obs-800k",
                value=0.004,
                conditions={
                    "temperature": 800,
                    "carrier_concentration": 1e19,
                    "carrier_type": "n",
                },
            ),
        ],
    )
    assert without_target.decisions[0].status == "incomparable"
    assert without_target.decisions[0].accepted_observation_ids == []
    assert "operating-condition sets" in without_target.decisions[0].reason


def test_stage_evidence_prompt_is_field_specific_but_tool_selection_stays_admin_owned() -> None:
    request = ValidationEvidenceRequest(
        stage=ValidationEvidenceStage.DFT_HANDOFF,
        chemical_system="Li-O",
        material_field=MaterialField.SOLID_ELECTROLYTE,
        application_subtype="crystalline_solid_electrolyte",
        problem_context={
            "mobile_ion": "Li",
            "temperature": 300,
            "electrode_pair": "Li|NMC",
        },
    )
    prompt = build_validation_evidence_prompt(request)
    route = material_stage_route(
        MaterialField.SOLID_ELECTROLYTE,
        "dft_handoff",
    )
    assert "solid_electrolyte-workflow-v1" in prompt
    assert "mobile-ion-path-neb" in prompt
    assert "crystalline_solid_electrolyte" in prompt
    assert '"temperature":300' in prompt
    assert "configured stage tool remains the only callable MCP tool" in prompt
    assert route.property_score_created_by_route is False
    assert route.can_steer_generation is False


def test_evidence_report_persists_domain_route_without_executing_validators(
    tmp_path,
) -> None:
    run = ValidationEvidenceRouter(
        None,
        artifact_root=tmp_path,
        enabled=False,
    ).run(
        ValidationEvidenceRequest(
            stage=ValidationEvidenceStage.DFT_HANDOFF,
            chemical_system="Fe-Co",
            material_field=MaterialField.MAGNETIC_MATERIAL,
            application_subtype="hard_magnet",
            problem_context={"temperature": 300, "magnetic_field": 0},
        )
    )
    assert run.report.material_field == MaterialField.MAGNETIC_MATERIAL
    assert run.report.domain_route is not None
    assert run.report.domain_route.profile_id == "magnetic_material-workflow-v1"
    assert run.report.application_subtype == "hard_magnet"
    assert run.report.handoff.application_subtype == "hard_magnet"
    assert run.report.handoff.domain_validator_execution_state == "not_executed"
    assert run.report.handoff.domain_validator_ids == [
        "magnetic-order-and-correlation-workflow",
        "soc-anisotropy-exchange-temperature-workflow",
    ]
    assert run.report.property_score_created is False

    with pytest.raises(ValueError, match="outside the material-field profile"):
        ValidationEvidenceRequest(
            stage=ValidationEvidenceStage.DFT_HANDOFF,
            chemical_system="Fe-Co",
            material_field=MaterialField.MAGNETIC_MATERIAL,
            application_subtype="single_junction_absorber",
        )


def test_profile_and_routes_are_returned_as_defensive_copies() -> None:
    first = get_material_field_profile(MaterialField.BATTERY_ELECTRODE)
    second = get_material_field_profile(MaterialField.BATTERY_ELECTRODE)
    assert first is not second
    assert first.stage_routes[0] is not second.stage_routes[0]


def test_material_route_cli_emits_auditable_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "material-route",
                "--field",
                "AUTO",
                "--prompt",
                "Find a thermoelectric with low lattice thermal conductivity",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["resolution"]["selected_field"] == "thermoelectric"
    assert len(payload["stages"]) == 5
    assert payload["externally_reported_property_names"] == []

    assert (
        main(
            [
                "material-route",
                "--field",
                "solid_electrolyte",
                "--prompt",
                "Find a solid electrolyte",
                "--use-main-model",
            ]
        )
        == 0
    )
    explicit = json.loads(capsys.readouterr().out)
    assert explicit["resolution"]["selection_mode"] == "explicit"
    assert explicit["main_model_run"] is None

    with pytest.raises(SystemExit, match="must contain valid JSON"):
        main(
            [
                "material-route",
                "--field",
                "AUTO",
                "--prompt",
                "Find a thermoelectric",
                "--context-json",
                "{not-json}",
            ]
        )
