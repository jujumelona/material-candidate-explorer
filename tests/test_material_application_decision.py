from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from discovery_os.cli import main
from discovery_os.literature_rag import (
    EvidenceClaim,
    EvidenceGraph,
    EvidencePolarity,
    EvidenceStage,
    LiteratureQuery,
    LiteratureRecord,
    LiteratureSource,
    RagEvidenceBundle,
    RagSearchPlan,
    SourceRetrievalStatus,
    SourceRunStatus,
)
from discovery_os.material_applications import (
    MainModelMaterialApplicationClassifier,
    application_roles_for_field,
    build_material_application_brief,
)
from discovery_os.material_decision_runner import MaterialDecisionRunner
from discovery_os.material_recommendation import (
    MaterialApplicationCandidate,
    MaterialApplicationObservation,
    MaterialDecisionPreference,
    candidates_from_application_seeds,
    rank_material_application_candidates,
)
from discovery_os.schemas import MaterialField


_SHA256 = "a" * 64
_TRANSPARENT_CONTEXT: dict[str, object] = {
    "application": "display",
    "film_thickness": 100.0,
    "temperature": 300.0,
    "deposition_process": "sputter",
    "wavelength_range": "400-700 nm",
    "substrate": "glass",
    "spectral_weighting": "photopic",
    "target_sheet_resistance": 15.0,
}


def test_application_context_rejects_credentials_before_routing() -> None:
    with pytest.raises(ValueError, match="cannot contain secrets"):
        build_material_application_brief(
            "Choose a cathode material for a sodium-ion battery.",
            material_field=MaterialField.BATTERY_ELECTRODE,
            problem_context={"OPENALEX_API_KEY": "must-not-enter-an-artifact"},
        )


def _transparent_brief():
    return build_material_application_brief(
        "Compare materials for a transparent electrode.",
        material_field=MaterialField.SEMICONDUCTOR,
        problem_context=_TRANSPARENT_CONTEXT,
        explicit_role_ids=["transparent_electrode"],
    )


def _candidate(
    candidate_id: str,
    *,
    role_id: str = "transparent_electrode",
    material: str | None = None,
    evidence_claim_ids: list[str] | None = None,
) -> MaterialApplicationCandidate:
    return MaterialApplicationCandidate(
        candidate_id=candidate_id,
        role_id=role_id,
        material_or_stack=material or candidate_id,
        origin="user_supplied",
        evidence_claim_ids=evidence_claim_ids or [],
        provenance_id=f"provenance-{candidate_id}",
    )


def _conditions(property_name: str, *, temperature: float = 300.0) -> dict[str, object]:
    if property_name == "sheet_resistance":
        return {
            "film_thickness": 100.0,
            "temperature": temperature,
            "deposition_process": "sputter",
        }
    if property_name == "spectral_transmittance":
        return {
            "wavelength_range": "400-700 nm",
            "film_thickness": 100.0,
            "substrate": "glass",
            "spectral_weighting": "photopic",
        }
    if property_name == "haze":
        return {
            "wavelength_range": "400-700 nm",
            "film_thickness": 100.0,
            "substrate": "glass",
        }
    raise AssertionError(f"unsupported fixture property: {property_name}")


def _observation(
    *,
    observation_id: str,
    candidate_id: str,
    property_name: str,
    validator_id: str,
    value: float,
    unit: str,
    conditions: dict[str, object],
    lower_bound: float | None = None,
    upper_bound: float | None = None,
) -> MaterialApplicationObservation:
    return MaterialApplicationObservation(
        observation_id=observation_id,
        candidate_id=candidate_id,
        role_id="transparent_electrode",
        property_name=property_name,
        validator_id=validator_id,
        status="success",
        value=value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit=unit,
        conditions=conditions,
        method="fixture method",
        sample_or_model_scope="same film fixture",
        authority_kind="experimental_validator",
        uncertainty_kind=(
            "measurement_interval"
            if lower_bound is not None
            else "not_quantified"
        ),
        provenance_id=f"provenance-{observation_id}",
        raw_artifact_sha256=_SHA256,
    )


def _result_for_property(result: Any, property_name: str):
    return next(
        item
        for item in result.criterion_results
        if item.property_name == property_name
    )


def _rag_bundle() -> RagEvidenceBundle:
    now = datetime.now(timezone.utc)
    record = LiteratureRecord(
        record_id="LIT-ITO-REFERENCE",
        title="ITO transparent-electrode reference",
        abstract=(
            "ITO was evaluated as a transparent electrode under a declared "
            "thin-film process."
        ),
        doi="10.1000/ito-reference",
        source_ids={"crossref": "10.1000/ito-reference"},
        source_queries=["QUERY-ITO"],
        urls=["https://doi.org/10.1000/ito-reference"],
        retrieved_at=now,
    )
    claim = EvidenceClaim(
        claim_id="CLAIM-ITO",
        source_record_id=record.record_id,
        subject="ITO",
        predicate="was evaluated as",
        object="a transparent electrode",
        polarity=EvidencePolarity.SUPPORTS,
        stage=EvidenceStage.MATERIAL_CHARACTERIZATION,
        support_text=record.abstract,
        confidence=0.9,
    )
    plan = RagSearchPlan(
        plan_id="PLAN-ITO",
        user_prompt="ITO transparent electrode",
        generated_at=now,
        planner_id="fixture-planner",
        planner_version="1",
        queries=[
            LiteratureQuery(
                query_id="QUERY-ITO",
                source=LiteratureSource.CROSSREF,
                query="ITO transparent electrode",
                rationale="fixture source closure",
            )
        ],
    )
    return RagEvidenceBundle(
        bundle_id="BUNDLE-ITO",
        created_at=now,
        search_plan=plan,
        source_statuses=[
            SourceRetrievalStatus(
                source=LiteratureSource.CROSSREF,
                status=SourceRunStatus.SUCCESS,
                query_ids=["QUERY-ITO"],
                result_count=1,
            )
        ],
        records=[record],
        claims=[claim],
        graph=EvidenceGraph(graph_id="GRAPH-ITO", nodes=[], edges=[]),
        branches=[],
    )


class _ApplicationModel:
    model_id = "fixture-application-model"
    model_version = "1"

    def __init__(self, decision: dict[str, object]) -> None:
        self.decision = decision

    def complete_json(self, *, operation: str, system: str, user: str) -> Any:
        assert operation == "classify-material-application"
        assert "allowed_roles" in user
        assert "Never choose an API" in system
        return self.decision


def _model_decision(
    *,
    role_id: str,
    evidence_span: str,
) -> dict[str, object]:
    return {
        "question_kind": "component_selection",
        "selected_role_ids": [role_id],
        "application_subtype": None,
        "extracted_context": {},
        "objective_priorities": [],
        "confidence": 0.9,
        "evidence_spans": [evidence_span],
        "needs_clarification": False,
        "clarification_question": None,
        "decision_summary": "Bounded fixture routing decision.",
        "endpoint_or_tool_selection_performed": False,
    }


def test_broad_korean_semiconductor_question_returns_role_portfolios_not_one_score() -> None:
    brief = build_material_application_brief(
        "반도체에서 어느 부분에 어떤 소재를 쓰는 게 맞을까?",
        material_field=MaterialField.SEMICONDUCTOR,
    )

    assert brief.question_kind == "component_map"
    assert brief.decomposition_mode == "role-portfolio"
    assert len(brief.roles) > 5
    assert brief.cross_role_ranking_allowed is False

    seeds = candidates_from_application_seeds(brief)
    assert len(seeds) > len(brief.roles)
    assert {item.role_id for item in seeds} == {
        item.role_id for item in brief.roles
    }

    report = rank_material_application_candidates(brief, candidates=seeds)
    assert report.cross_role_ranking_performed is False
    assert len(report.role_recommendations) == len(brief.roles)
    assert all(
        item.pool_relative_decision_score is None
        for portfolio in report.role_recommendations
        for item in portfolio.candidates
    )


def test_broad_korean_battery_question_decomposes_positive_and_negative_electrodes() -> None:
    brief = build_material_application_brief(
        "배터리에는 부품별로 어떤 소재가 맞을까?",
        material_field=MaterialField.BATTERY_ELECTRODE,
    )

    assert brief.question_kind == "component_map"
    assert brief.decomposition_mode == "role-portfolio"
    assert [item.role_id for item in brief.roles] == [
        "battery_positive_electrode_active",
        "battery_negative_electrode_active",
    ]
    assert all(brief.candidate_seeds_by_role[item.role_id] for item in brief.roles)


def test_every_material_field_has_roles_and_five_closed_evidence_tasks() -> None:
    expected_categories = [
        "requirements_and_metrics",
        "incumbents_and_tradeoffs",
        "candidate_evidence",
        "negative_and_failure_evidence",
        "validation_and_reproducibility",
    ]
    for field in MaterialField:
        roles = application_roles_for_field(field)
        assert roles, field
        for role in roles:
            assert len(role.evidence_tasks) == 5
            assert [item.category for item in role.evidence_tasks] == expected_categories
            assert all(item.can_create_property_scores is False for item in role.evidence_tasks)
            assert all(
                item.prompt_or_model_can_choose_mcp_tool is False
                for item in role.evidence_tasks
            )


@pytest.mark.parametrize(
    ("role_id", "evidence_span", "message"),
    [
        (
            "invented_super_material_role",
            "transparent electrode",
            "outside the code allowlist",
        ),
        (
            "transparent_electrode",
            "a sentence not present in the input",
            "outside the input",
        ),
    ],
)
def test_main_application_model_rejects_untrusted_roles_and_nonliteral_evidence(
    role_id: str,
    evidence_span: str,
    message: str,
) -> None:
    classifier = MainModelMaterialApplicationClassifier(
        _ApplicationModel(
            _model_decision(
                role_id=role_id,
                evidence_span=evidence_span,
            )
        )
    )

    with pytest.raises(ValueError, match=message):
        classifier.classify(
            "Choose a transparent electrode.",
            material_field=MaterialField.SEMICONDUCTOR,
        )


def test_wrong_unit_and_wrong_target_condition_remain_incomparable() -> None:
    brief = _transparent_brief()
    candidate = _candidate("CANDIDATE-INVALID")
    role = brief.roles[0]
    sheet = next(
        item for item in role.criteria if item.property_name == "sheet_resistance"
    )
    observations = [
        _observation(
            observation_id="OBS-WRONG-UNIT",
            candidate_id=candidate.candidate_id,
            property_name=sheet.property_name,
            validator_id=sheet.validator_ids[0],
            value=10.0,
            unit="ohm",
            conditions=_conditions(sheet.property_name),
        ),
        _observation(
            observation_id="OBS-WRONG-CONDITION",
            candidate_id=candidate.candidate_id,
            property_name=sheet.property_name,
            validator_id=sheet.validator_ids[0],
            value=11.0,
            unit=sheet.unit,
            conditions=_conditions(sheet.property_name, temperature=350.0),
        ),
    ]

    report = rank_material_application_candidates(
        brief,
        candidates=[candidate],
        observations=observations,
    )
    result = report.role_recommendations[0].candidates[0]
    sheet_result = _result_for_property(result, "sheet_resistance")

    assert sheet_result.status == "incomparable"
    assert sheet_result.value is None
    assert set(sheet_result.rejected_observation_ids) == {
        "OBS-WRONG-UNIT",
        "OBS-WRONG-CONDITION",
    }
    assert result.comparison_group_id is None
    assert result.pool_relative_decision_score is None


def test_conflicting_named_validator_values_are_preserved_without_averaging() -> None:
    brief = _transparent_brief()
    candidate = _candidate("CANDIDATE-CONFLICT")
    sheet = next(
        item
        for item in brief.roles[0].criteria
        if item.property_name == "sheet_resistance"
    )
    assert len(sheet.validator_ids) >= 2
    observations = [
        _observation(
            observation_id="OBS-CONFLICT-A",
            candidate_id=candidate.candidate_id,
            property_name=sheet.property_name,
            validator_id=sheet.validator_ids[0],
            value=10.0,
            unit=sheet.unit,
            conditions=_conditions(sheet.property_name),
        ),
        _observation(
            observation_id="OBS-CONFLICT-B",
            candidate_id=candidate.candidate_id,
            property_name=sheet.property_name,
            validator_id=sheet.validator_ids[1],
            value=14.0,
            unit=sheet.unit,
            conditions=_conditions(sheet.property_name),
        ),
    ]

    report = rank_material_application_candidates(
        brief,
        candidates=[candidate],
        observations=observations,
    )
    sheet_result = _result_for_property(
        report.role_recommendations[0].candidates[0],
        "sheet_resistance",
    )

    assert sheet_result.status == "conflicting"
    assert sheet_result.value is None
    assert sheet_result.accepted_observation_ids == [
        "OBS-CONFLICT-A",
        "OBS-CONFLICT-B",
    ]
    assert sheet_result.value_aggregation_performed is False
    assert sheet_result.reason_code == "CONFLICTING_NAMED_VALIDATORS"


def test_robust_pareto_and_operator_weighted_score_are_role_condition_scoped() -> None:
    brief = _transparent_brief()
    role = brief.roles[0]
    candidates = [_candidate("CANDIDATE-A"), _candidate("CANDIDATE-B")]
    values = {
        "CANDIDATE-A": {
            "sheet_resistance": (10.0, 9.0, 11.0),
            "spectral_transmittance": (0.92, 0.91, 0.93),
            "haze": (1.0, 0.9, 1.1),
        },
        "CANDIDATE-B": {
            "sheet_resistance": (20.0, 19.0, 21.0),
            "spectral_transmittance": (0.80, 0.79, 0.81),
            "haze": (2.0, 1.9, 2.1),
        },
    }
    observations: list[MaterialApplicationObservation] = []
    for candidate in candidates:
        for criterion in role.criteria:
            if not criterion.required_for_ranking:
                continue
            value, low, high = values[candidate.candidate_id][
                criterion.property_name
            ]
            observations.append(
                _observation(
                    observation_id=(
                        f"OBS-{candidate.candidate_id}-{criterion.property_name}"
                    ),
                    candidate_id=candidate.candidate_id,
                    property_name=criterion.property_name,
                    validator_id=criterion.validator_ids[0],
                    value=value,
                    lower_bound=low,
                    upper_bound=high,
                    unit=criterion.unit,
                    conditions=_conditions(criterion.property_name),
                )
            )
    preferences = [
        MaterialDecisionPreference(
            criterion_id=criterion.criterion_id,
            weight=float(index),
            source="operator",
        )
        for index, criterion in enumerate(
            [item for item in role.criteria if item.required_for_ranking],
            start=1,
        )
    ]

    report = rank_material_application_candidates(
        brief,
        candidates=candidates,
        observations=observations,
        preferences=preferences,
    )
    portfolio = report.role_recommendations[0]
    by_id = {item.candidate.candidate_id: item for item in portfolio.candidates}

    assert portfolio.comparison_group_count == 1
    assert by_id["CANDIDATE-A"].comparison_group_id == by_id[
        "CANDIDATE-B"
    ].comparison_group_id
    assert by_id["CANDIDATE-A"].pareto_front == 1
    assert by_id["CANDIDATE-B"].pareto_front == 2
    assert by_id["CANDIDATE-A"].rank_within_role_and_condition == 1
    assert by_id["CANDIDATE-B"].rank_within_role_and_condition == 2
    assert by_id["CANDIDATE-A"].pool_relative_decision_score == pytest.approx(100.0)
    assert by_id["CANDIDATE-B"].pool_relative_decision_score == pytest.approx(0.0)
    assert report.cross_role_ranking_performed is False
    assert portfolio.scalar_score_created_without_operator_weights is False


def test_rag_claims_must_close_to_bundle_and_emit_exact_citations() -> None:
    brief = _transparent_brief()
    bundle = _rag_bundle()
    candidate = _candidate(
        "CANDIDATE-ITO",
        material="ITO",
        evidence_claim_ids=["CLAIM-ITO"],
    )

    report = rank_material_application_candidates(
        brief,
        candidates=[candidate],
        rag_bundle=bundle,
    )
    result = report.role_recommendations[0].candidates[0]

    assert report.rag_bundle_id == bundle.bundle_id
    assert len(result.citations) == 1
    assert result.citations[0].record_id == "LIT-ITO-REFERENCE"
    assert result.citations[0].exact_support_span == bundle.claims[0].support_text
    assert result.citations[0].retrieved_record_only_not_property_validator is True
    assert result.pool_relative_decision_score is None

    with pytest.raises(ValueError, match="outside the RAG bundle"):
        rank_material_application_candidates(
            brief,
            candidates=[
                _candidate(
                    "CANDIDATE-UNBOUND",
                    evidence_claim_ids=["CLAIM-NOT-IN-BUNDLE"],
                )
            ],
            rag_bundle=bundle,
        )


def test_retrieval_seed_report_keeps_missing_properties_unknown_not_zero() -> None:
    brief = build_material_application_brief(
        "배터리에는 부품별로 어떤 소재가 맞을까?",
        material_field=MaterialField.BATTERY_ELECTRODE,
    )
    report = rank_material_application_candidates(
        brief,
        candidates=candidates_from_application_seeds(brief),
    )

    assert report.missing_value_imputed_as_zero is False
    for portfolio in report.role_recommendations:
        for candidate in portfolio.candidates:
            assert candidate.evidence_completeness_score == 0.0
            assert candidate.pool_relative_decision_score is None
            assert candidate.comparison_group_id is None
            assert all(item.status == "unknown" for item in candidate.criterion_results)
            assert all(item.value is None for item in candidate.criterion_results)


def test_runner_persists_json_csv_and_markdown_reports(tmp_path: Path) -> None:
    runner = MaterialDecisionRunner(
        artifact_root=tmp_path,
        environ={},
    )
    run = runner.run(
        "배터리에는 부품별로 어떤 소재가 맞을까?",
        material_field=MaterialField.BATTERY_ELECTRODE,
        main_model_routing="off",
    )

    kinds = {item.kind for item in run.artifacts}
    assert {
        "application_brief",
        "recommendation_json",
        "recommendation_markdown",
        "recommendation_csv",
    } <= kinds
    by_kind = {item.kind: tmp_path / item.relative_path for item in run.artifacts}
    assert all(path.is_file() for path in by_kind.values())

    report_json = json.loads(
        by_kind["recommendation_json"].read_text(encoding="utf-8")
    )
    assert report_json["cross_role_ranking_performed"] is False
    assert "role_recommendations" in report_json
    assert "Role-scoped decision support only" in by_kind[
        "recommendation_markdown"
    ].read_text(encoding="utf-8")
    assert "candidate_id" in by_kind["recommendation_csv"].read_text(
        encoding="utf-8"
    ).splitlines()[0]

    receipt = (
        tmp_path / run.run_id / "material-decision-run.json"
    )
    assert receipt.is_file()
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt_payload["generation_or_specialist_execution_performed"] is False
    assert receipt_payload["artifacts"]


def test_material_recommend_cli_emits_machine_readable_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "material-recommend",
            "--prompt",
            "Choose positive and negative electrode materials for a sodium-ion battery.",
            "--field",
            MaterialField.BATTERY_ELECTRODE.value,
            "--main-model-routing",
            "off",
            "--role",
            "battery_positive_electrode_active",
            "--role",
            "battery_negative_electrode_active",
            "--artifacts",
            str(tmp_path),
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["brief"]["field_plan"]["resolution"]["profile_id"] == (
        "battery_electrode-workflow-v1"
    )
    assert payload["generation_or_specialist_execution_performed"] is False
    assert len(payload["report"]["role_recommendations"]) == 2
    assert (tmp_path / payload["run_id"] / "material-decision-run.json").is_file()
