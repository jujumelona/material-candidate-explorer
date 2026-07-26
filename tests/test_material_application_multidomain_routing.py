from __future__ import annotations

import pytest

from discovery_os.material_applications import build_material_application_brief
from discovery_os.material_domains import build_material_domain_plan
from discovery_os.schemas import MaterialField


@pytest.mark.parametrize(
    ("question", "expected_field", "expected_role"),
    [
        (
            "전고체 배터리용 전해질 소재를 찾아줘",
            MaterialField.SOLID_ELECTROLYTE,
            "solid_electrolyte_bulk_separator",
        ),
        (
            "고온 구조 합금 후보를 비교해줘",
            MaterialField.STRUCTURAL_ALLOY,
            "high_temperature_load_bearing",
        ),
        (
            "고온 염화물 환경의 하중 지지 부품에는 어떤 소재가 맞을까?",
            MaterialField.STRUCTURAL_ALLOY,
            "high_temperature_load_bearing",
        ),
        (
            "이산화탄소 포집 소재로 무엇이 적합할까",
            MaterialField.POROUS_FRAMEWORK,
            "carbon_capture",
        ),
        (
            "700 K에서 쓸 n형 열전 소재를 비교해줘",
            MaterialField.THERMOELECTRIC,
            "thermoelectric_n_type_leg",
        ),
        (
            "적외선 광학 창 소재 후보를 찾아줘",
            MaterialField.GENERAL_INORGANIC,
            "optical_window",
        ),
    ],
)
def test_korean_application_questions_route_to_the_intended_field_and_role(
    question: str,
    expected_field: MaterialField,
    expected_role: str,
) -> None:
    field_plan = build_material_domain_plan("AUTO", prompt=question)
    assert field_plan.resolution.selected_field == expected_field
    assert field_plan.resolution.requires_operator_choice is False

    brief = build_material_application_brief(
        question,
        material_field="AUTO",
    )
    assert brief.material_field == expected_field
    assert [item.role_id for item in brief.roles] == [expected_role]
    assert brief.cross_role_ranking_allowed is False


def test_broad_battery_question_keeps_positive_and_negative_roles_separate() -> None:
    brief = build_material_application_brief(
        "고속 충전 배터리에서 양극과 음극에는 각각 어떤 소재가 맞을까?",
        material_field="AUTO",
    )

    assert brief.material_field == MaterialField.BATTERY_ELECTRODE
    assert {item.role_id for item in brief.roles} == {
        "battery_positive_electrode_active",
        "battery_negative_electrode_active",
    }
    assert brief.decomposition_mode == "role-portfolio"
    assert brief.cross_role_ranking_allowed is False
