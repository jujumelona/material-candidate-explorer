from __future__ import annotations

from datetime import datetime, timezone

import pytest

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
from discovery_os.material_applications import build_material_application_brief
from discovery_os.material_recommendation import (
    MaterialApplicationCandidate,
    MaterialApplicationObservation,
    MaterialDecisionPreference,
    rank_material_application_candidates,
)


_SHA = "a" * 64
_TRANSPARENT_CONTEXT: dict[str, object] = {
    "application": "display",
    "wavelength_range": "400-700 nm",
    "spectral_weighting": "photopic",
    "film_thickness": 100,
    "substrate": "glass",
    "target_sheet_resistance": 15,
    "deposition_process": "sputter",
    "temperature": 300,
}


def _transparent_brief(*, temperature: object = 300):
    context = dict(_TRANSPARENT_CONTEXT)
    context["temperature"] = temperature
    return build_material_application_brief(
        "Compare transparent electrode candidates.",
        material_field="semiconductor",
        problem_context=context,
        explicit_role_ids=["transparent_electrode"],
    )


def _candidate(candidate_id: str, *, role_id: str = "transparent_electrode"):
    return MaterialApplicationCandidate(
        candidate_id=candidate_id,
        role_id=role_id,
        material_or_stack=candidate_id,
        origin="user_supplied",
        provenance_id=f"PROV-{candidate_id}",
    )


def _transparent_conditions(property_name: str) -> dict[str, object]:
    if property_name == "sheet_resistance":
        return {
            "film_thickness": 100.0,
            "temperature": 300.0,
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
    raise AssertionError(property_name)


def _transparent_observations(
    brief,
    candidate_id: str,
    *,
    bounded: bool,
    identical: bool = False,
) -> list[MaterialApplicationObservation]:
    role = brief.roles[0]
    index = 0 if identical or candidate_id == "A" else 1
    values = {
        "sheet_resistance": (10.0, 20.0),
        "spectral_transmittance": (0.90, 0.80),
        "haze": (1.0, 2.0),
    }
    rows: list[MaterialApplicationObservation] = []
    for criterion in role.criteria:
        if not criterion.required_for_ranking:
            continue
        value = values[criterion.property_name][index]
        width = 0.1 if abs(value) <= 2.0 else 1.0
        rows.append(
            MaterialApplicationObservation(
                observation_id=f"OBS-{candidate_id}-{criterion.property_name}",
                candidate_id=candidate_id,
                role_id=role.role_id,
                property_name=criterion.property_name,
                validator_id=criterion.validator_ids[0],
                status="success",
                value=value,
                lower_bound=value - width if bounded else None,
                upper_bound=value + width if bounded else None,
                unit=criterion.unit,
                conditions=_transparent_conditions(criterion.property_name),
                method="fixture",
                sample_or_model_scope="matched film",
                authority_kind="experimental_validator",
                uncertainty_kind=(
                    "measurement_interval" if bounded else "not_quantified"
                ),
                provenance_id=f"PROV-OBS-{candidate_id}-{criterion.property_name}",
                raw_artifact_sha256=_SHA,
            )
        )
    return rows


def _rag_bundle() -> RagEvidenceBundle:
    now = datetime.now(timezone.utc)
    record = LiteratureRecord(
        record_id="RECORD-SPEC",
        title="Source-closed transparent-electrode specification",
        abstract="The specification defines a transparent-electrode decision policy.",
        doi="10.1000/source-closed-spec",
        source_ids={"crossref": "10.1000/source-closed-spec"},
        source_queries=["QUERY-SPEC"],
        urls=["https://doi.org/10.1000/source-closed-spec"],
        retrieved_at=now,
    )
    claim = EvidenceClaim(
        claim_id="CLAIM-SPEC",
        source_record_id=record.record_id,
        subject="transparent electrode",
        predicate="uses",
        object="a source-closed decision policy",
        polarity=EvidencePolarity.SUPPORTS,
        stage=EvidenceStage.MATERIAL_CHARACTERIZATION,
        support_text=record.abstract,
        confidence=0.9,
    )
    return RagEvidenceBundle(
        bundle_id="BUNDLE-SPEC",
        created_at=now,
        search_plan=RagSearchPlan(
            plan_id="PLAN-SPEC",
            user_prompt="transparent electrode specification",
            generated_at=now,
            planner_id="fixture",
            planner_version="1",
            queries=[
                LiteratureQuery(
                    query_id="QUERY-SPEC",
                    source=LiteratureSource.CROSSREF,
                    query="transparent electrode specification",
                    rationale="fixture",
                )
            ],
        ),
        source_statuses=[
            SourceRetrievalStatus(
                source=LiteratureSource.CROSSREF,
                status=SourceRunStatus.SUCCESS,
                query_ids=["QUERY-SPEC"],
                result_count=1,
            )
        ],
        records=[record],
        claims=[claim],
        graph=EvidenceGraph(graph_id="GRAPH-SPEC", nodes=[], edges=[]),
        branches=[],
    )


def test_unresolved_auto_field_routing_preserves_an_unranked_portfolio() -> None:
    brief = build_material_application_brief(
        "Compare a cathode catalyst material.",
        material_field="AUTO",
        problem_context={
            "chemical_system": "Li-Ni-O",
            "target_property": "stability",
            "temperature": 300,
            "pressure": 0,
        },
        explicit_role_ids=["stable_bulk_phase"],
    )
    assert brief.field_plan.resolution.requires_operator_choice is True
    role = brief.roles[0]
    candidates = [
        _candidate("A", role_id=role.role_id),
        _candidate("B", role_id=role.role_id),
    ]
    report = rank_material_application_candidates(
        brief,
        candidates=candidates,
    )

    assert all(
        item.comparison_group_id is None
        and item.rank_within_role_and_condition is None
        and item.pareto_front is None
        for item in report.role_recommendations[0].candidates
    )
    assert any("operator choice" in item for item in report.unresolved_questions)


def test_null_target_condition_is_not_a_false_mismatch_but_blocks_ranking() -> None:
    brief = _transparent_brief(temperature=None)
    candidate = _candidate("A")
    report = rank_material_application_candidates(
        brief,
        candidates=[candidate],
        observations=_transparent_observations(
            brief,
            candidate.candidate_id,
            bounded=True,
        ),
    )
    result = report.role_recommendations[0].candidates[0]
    sheet = next(
        item
        for item in result.criterion_results
        if item.property_name == "sheet_resistance"
    )

    assert sheet.status == "available"
    assert result.comparison_group_id is None
    assert result.rank_within_role_and_condition is None
    assert any(
        "missing required target conditions: temperature" in item
        for item in report.unresolved_questions
    )


def test_missing_role_context_blocks_comparison_even_with_complete_observations() -> None:
    context = dict(_TRANSPARENT_CONTEXT)
    context.pop("application")
    brief = build_material_application_brief(
        "Compare transparent electrode candidates.",
        material_field="semiconductor",
        problem_context=context,
        explicit_role_ids=["transparent_electrode"],
    )
    assert brief.ready_for_condition_complete_scoring is False
    candidates = [_candidate("A"), _candidate("B")]
    observations = [
        row
        for candidate in candidates
        for row in _transparent_observations(
            brief,
            candidate.candidate_id,
            bounded=True,
        )
    ]
    report = rank_material_application_candidates(
        brief,
        candidates=candidates,
        observations=observations,
    )

    assert all(
        item.comparison_group_id is None
        and item.rank_within_role_and_condition is None
        for item in report.role_recommendations[0].candidates
    )
    assert any(
        "missing role context: application" in item
        for item in report.unresolved_questions
    )


def test_integral_float_conditions_compare_canonically_and_pareto_ties_share_rank() -> None:
    brief = _transparent_brief(temperature=300)
    candidates = [_candidate("A"), _candidate("B")]
    observations = [
        row
        for candidate in candidates
        for row in _transparent_observations(
            brief,
            candidate.candidate_id,
            bounded=True,
            identical=True,
        )
    ]
    report = rank_material_application_candidates(
        brief,
        candidates=candidates,
        observations=observations,
    )
    results = report.role_recommendations[0].candidates

    assert len({item.comparison_group_id for item in results}) == 1
    assert results[0].comparison_group_id is not None
    assert {item.pareto_front for item in results} == {1}
    assert {item.rank_within_role_and_condition for item in results} == {1}
    assert all(item.pool_relative_decision_score is None for item in results)


def test_point_only_observations_cannot_hard_pass_or_enter_robust_ranking() -> None:
    brief = _transparent_brief()
    candidates = [_candidate("A"), _candidate("B")]
    observations = [
        row
        for candidate in candidates
        for row in _transparent_observations(
            brief,
            candidate.candidate_id,
            bounded=False,
        )
    ]
    sheet = next(
        item
        for item in brief.roles[0].criteria
        if item.property_name == "sheet_resistance"
    )
    report = rank_material_application_candidates(
        brief,
        candidates=candidates,
        observations=observations,
        preferences=[
            MaterialDecisionPreference(
                criterion_id=sheet.criterion_id,
                hard_maximum=15.0,
            )
        ],
    )

    for result in report.role_recommendations[0].candidates:
        sheet_result = next(
            item
            for item in result.criterion_results
            if item.property_name == "sheet_resistance"
        )
        assert sheet_result.hard_gate_status == "unknown"
        assert result.hard_gate_status == "unknown"
        assert result.evidence_uncertainty_status == "point_only"
        assert result.comparison_group_id is None
        assert result.rank_within_role_and_condition is None


def test_source_closed_preferences_must_resolve_inside_the_supplied_bundle() -> None:
    brief = _transparent_brief()
    candidate = _candidate("A")
    criterion_id = brief.roles[0].criteria[0].criterion_id
    unbound = MaterialDecisionPreference(
        criterion_id=criterion_id,
        weight=1.0,
        source="source_closed_spec",
        provenance_id="CLAIM-NOT-IN-BUNDLE",
    )

    with pytest.raises(ValueError, match="require a supplied RAG evidence bundle"):
        rank_material_application_candidates(
            brief,
            candidates=[candidate],
            preferences=[unbound],
        )
    with pytest.raises(ValueError, match="must match a supporting claim"):
        rank_material_application_candidates(
            brief,
            candidates=[candidate],
            preferences=[unbound],
            rag_bundle=_rag_bundle(),
        )

    report = rank_material_application_candidates(
        brief,
        candidates=[candidate],
        preferences=[
            unbound.model_copy(update={"provenance_id": "CLAIM-SPEC"})
        ],
        rag_bundle=_rag_bundle(),
    )
    assert report.rag_bundle_id == "BUNDLE-SPEC"
