from __future__ import annotations

import json

import pytest

from discovery_os.model_adapters import (
    LocalDiscoveryModel,
    ModelOutputError,
    RemoteDiscoveryModel,
)
from discovery_os.schemas import DiscoveryGoal, GoalCompileRequest, PropertyObjective


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="GOAL-ADAPTER",
        domain="general_materials",
        title="Adapter contract",
        scientific_question="Can the strict adapter parse this response?",
        objectives=[
            PropertyObjective(property_name="target_property", direction="maximize")
        ],
        validation_profile_id="general_materials-v1",
        candidate_types=["composition"],
    )


def test_local_model_uses_fixed_operation_and_strictly_validates_response() -> None:
    calls: list[tuple[str, dict, type]] = []

    def backend(*, operation: str, request_json: str, response_schema: type) -> str:
        calls.append((operation, json.loads(request_json), response_schema))
        return _goal().model_dump_json()

    result = LocalDiscoveryModel(backend).compile_goal(
        GoalCompileRequest(user_text="general material target")
    )

    assert result.goal_id == "GOAL-ADAPTER"
    assert calls[0][0] == "compile_goal"
    assert calls[0][1]["user_text"] == "general material target"
    assert calls[0][2] is DiscoveryGoal


def test_local_model_rejects_extra_model_output_fields() -> None:
    payload = _goal().model_dump(mode="json")
    payload["python_code"] = "print('must never run')"

    with pytest.raises(ModelOutputError, match="extra_forbidden"):
        LocalDiscoveryModel(lambda **_: payload).compile_goal(
            GoalCompileRequest(user_text="reject extra fields")
        )


def test_local_model_rejects_missing_explicit_schema_version() -> None:
    payload = _goal().model_dump(mode="json")
    payload.pop("schema_version")

    with pytest.raises(ModelOutputError, match="explicitly include supported schema_version"):
        LocalDiscoveryModel(lambda **_: payload).compile_goal(
            GoalCompileRequest(user_text="version is mandatory on the wire")
        )


def test_remote_model_posts_to_fixed_endpoint_and_revalidates_json() -> None:
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return _goal().model_dump(mode="json")

    class Session:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def post(self, url: str, **kwargs):
            self.calls.append({"url": url, **kwargs})
            return Response()

    session = Session()
    model = RemoteDiscoveryModel(
        "https://model.invalid/v1/",
        timeout=(2, 30),
        auth_headers={"Authorization": "Bearer test"},
        session=session,
    )
    result = model.compile_goal(GoalCompileRequest(user_text="remote contract"))

    assert result.goal_id == "GOAL-ADAPTER"
    assert session.calls[0]["url"] == "https://model.invalid/v1/compile-goal"
    assert session.calls[0]["timeout"] == (2, 30)
    assert session.calls[0]["headers"]["Authorization"] == "Bearer test"


@pytest.mark.parametrize("url", ["", "file:///tmp/model", "model.example/path"])
def test_remote_model_rejects_non_http_base_urls(url: str) -> None:
    with pytest.raises(ValueError, match="base_url"):
        RemoteDiscoveryModel(url)
