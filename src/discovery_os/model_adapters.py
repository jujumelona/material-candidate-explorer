"""Strict adapters for local and HTTP discovery-model backends.

Both adapters are trust boundaries: model output is always serialized as JSON
and validated again as the exact response contract.  Response text is never
evaluated, imported, or otherwise interpreted as executable code.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, TypeVar, cast
from urllib.parse import urlsplit

import requests
from pydantic import BaseModel, ValidationError

from .schemas import (
    CandidatePlan,
    CandidateProposalRequest,
    DiscoveryGoal,
    GoalCompileRequest,
    HypothesisBatch,
    HypothesisRequest,
    PredictionBatch,
    PredictionRequest,
    ResultAnalysis,
    ResultAnalysisRequest,
    RevisionPlan,
    RevisionRequest,
    StopDecision,
    StopDecisionRequest,
    StrictSchema,
    ValidationPlan,
    ValidationPlanningRequest,
)


ResponseT = TypeVar("ResponseT", bound=StrictSchema)
ModelPayload: TypeAlias = str | bytes | bytearray | Mapping[str, Any] | BaseModel


class StructuredModelBackend(Protocol):
    """Recommended local structured-generation backend contract."""

    def generate_structured(
        self,
        *,
        operation: str,
        request_json: str,
        response_schema: type[StrictSchema],
    ) -> ModelPayload:
        ...


MODEL_ENDPOINTS = MappingProxyType(
    {
        "compile_goal": "/compile-goal",
        "propose_hypotheses": "/propose-hypotheses",
        "propose_candidates": "/propose-candidates",
        "predict_candidates": "/predict-candidates",
        "plan_validation": "/plan-validation",
        "analyze_results": "/analyze-results",
        "revise_candidates": "/revise-candidates",
        "decide_stop": "/decide-stop",
    }
)


class ModelOutputError(RuntimeError):
    """A model call failed to produce the required structured response."""

    def __init__(
        self,
        operation: str,
        expected_type: type[BaseModel],
        detail: str,
    ) -> None:
        self.operation = operation
        self.expected_type = expected_type
        self.detail = detail
        super().__init__(
            f"model operation {operation!r} did not produce a valid "
            f"{expected_type.__name__}: {detail}"
        )


class LocalDiscoveryModel:
    """Adapt a trusted in-process structured generator to ``DiscoveryModel``.

    Supported backend entry points, in priority order, are
    ``generate_structured``, ``structured_generate``, ``generate``, and the
    backend object itself when callable.  Common signatures such as
    ``(request_json, response_schema=...)`` and ``(operation, request, schema)``
    are supported.  Backend output remains untrusted and is revalidated.
    """

    def __init__(
        self,
        backend: StructuredModelBackend | Callable[..., ModelPayload] | None = None,
        *,
        structured_backend: StructuredModelBackend | Callable[..., ModelPayload] | None = None,
    ) -> None:
        if backend is not None and structured_backend is not None:
            raise TypeError("provide either backend or structured_backend, not both")
        selected = backend if backend is not None else structured_backend
        if selected is None:
            raise TypeError("a structured backend or callable is required")
        self.backend = selected
        self._target = _select_backend_target(selected)

    def compile_goal(self, request: GoalCompileRequest) -> DiscoveryGoal:
        return self._call("compile_goal", request, GoalCompileRequest, DiscoveryGoal)

    def propose_hypotheses(self, request: HypothesisRequest) -> HypothesisBatch:
        return self._call("propose_hypotheses", request, HypothesisRequest, HypothesisBatch)

    def propose_candidates(self, request: CandidateProposalRequest) -> CandidatePlan:
        return self._call("propose_candidates", request, CandidateProposalRequest, CandidatePlan)

    def predict_candidates(self, request: PredictionRequest) -> PredictionBatch:
        return self._call("predict_candidates", request, PredictionRequest, PredictionBatch)

    def plan_validation(self, request: ValidationPlanningRequest) -> ValidationPlan:
        return self._call("plan_validation", request, ValidationPlanningRequest, ValidationPlan)

    def analyze_results(self, request: ResultAnalysisRequest) -> ResultAnalysis:
        return self._call("analyze_results", request, ResultAnalysisRequest, ResultAnalysis)

    def revise_candidates(self, request: RevisionRequest) -> RevisionPlan:
        return self._call("revise_candidates", request, RevisionRequest, RevisionPlan)

    def decide_stop(self, request: StopDecisionRequest) -> StopDecision:
        return self._call("decide_stop", request, StopDecisionRequest, StopDecision)

    def _call(
        self,
        operation: str,
        request: StrictSchema,
        request_type: type[StrictSchema],
        response_type: type[ResponseT],
    ) -> ResponseT:
        _require_request_type(operation, request, request_type)
        request_json = request.model_dump_json(by_alias=True, exclude_none=False)
        request_payload = request.model_dump(mode="json", by_alias=True, exclude_none=False)
        try:
            raw = _invoke_structured_backend(
                self._target,
                operation=operation,
                request=request,
                request_json=request_json,
                request_payload=request_payload,
                response_type=response_type,
            )
        except ModelOutputError:
            raise
        except Exception as exc:
            raise ModelOutputError(
                operation,
                response_type,
                f"local structured backend failed: {type(exc).__name__}: {exc}",
            ) from exc
        return _validate_model_response(operation, response_type, raw)


class RemoteDiscoveryModel:
    """HTTP JSON implementation of the eight-method discovery-model port."""

    ENDPOINTS = MODEL_ENDPOINTS

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float | tuple[float, float] = 300.0,
        auth_headers: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        session: Any | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.timeout = _validate_timeout(timeout)
        merged_headers = dict(headers or {})
        merged_headers.update(auth_headers or {})
        self.headers = _validate_headers(merged_headers)
        # The module default keeps requests.post monkey-patchable.  An injected
        # Session/compatible client can provide pooling or deterministic tests.
        self._http = session if session is not None else requests

    def compile_goal(self, request: GoalCompileRequest) -> DiscoveryGoal:
        return self._post("compile_goal", request, GoalCompileRequest, DiscoveryGoal)

    def propose_hypotheses(self, request: HypothesisRequest) -> HypothesisBatch:
        return self._post("propose_hypotheses", request, HypothesisRequest, HypothesisBatch)

    def propose_candidates(self, request: CandidateProposalRequest) -> CandidatePlan:
        return self._post("propose_candidates", request, CandidateProposalRequest, CandidatePlan)

    def predict_candidates(self, request: PredictionRequest) -> PredictionBatch:
        return self._post("predict_candidates", request, PredictionRequest, PredictionBatch)

    def plan_validation(self, request: ValidationPlanningRequest) -> ValidationPlan:
        return self._post("plan_validation", request, ValidationPlanningRequest, ValidationPlan)

    def analyze_results(self, request: ResultAnalysisRequest) -> ResultAnalysis:
        return self._post("analyze_results", request, ResultAnalysisRequest, ResultAnalysis)

    def revise_candidates(self, request: RevisionRequest) -> RevisionPlan:
        return self._post("revise_candidates", request, RevisionRequest, RevisionPlan)

    def decide_stop(self, request: StopDecisionRequest) -> StopDecision:
        return self._post("decide_stop", request, StopDecisionRequest, StopDecision)

    def _post(
        self,
        operation: str,
        request: StrictSchema,
        request_type: type[StrictSchema],
        response_type: type[ResponseT],
    ) -> ResponseT:
        _require_request_type(operation, request, request_type)
        url = f"{self.base_url}{self.ENDPOINTS[operation]}"
        payload = request.model_dump(mode="json", by_alias=True, exclude_none=False)
        try:
            response = self._http.post(
                url,
                json=payload,
                headers=dict(self.headers),
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            raise ModelOutputError(
                operation,
                response_type,
                f"HTTP request failed: {type(exc).__name__}: {exc}",
            ) from exc

        try:
            raw = response.json()
        except Exception as exc:
            raise ModelOutputError(
                operation,
                response_type,
                f"HTTP response was not valid JSON: {type(exc).__name__}: {exc}",
            ) from exc
        return _validate_model_response(operation, response_type, raw)


def _select_backend_target(backend: object) -> Callable[..., ModelPayload]:
    for attribute in ("generate_structured", "structured_generate", "generate"):
        target = getattr(backend, attribute, None)
        if callable(target):
            return cast(Callable[..., ModelPayload], target)
    if callable(backend):
        return cast(Callable[..., ModelPayload], backend)
    raise TypeError(
        "structured backend must be callable or expose "
        "generate_structured()/structured_generate()/generate()"
    )


def _invoke_structured_backend(
    target: Callable[..., ModelPayload],
    *,
    operation: str,
    request: StrictSchema,
    request_json: str,
    request_payload: dict[str, Any],
    response_type: type[StrictSchema],
) -> ModelPayload:
    """Invoke common structured-backend signatures without retrying calls."""

    values: dict[str, Any] = {
        "operation": operation,
        "method": operation,
        "method_name": operation,
        "request": request_json,
        "request_json": request_json,
        "request_model": request,
        "payload": request_payload,
        "request_payload": request_payload,
        "response_schema": response_type,
        "response_model": response_type,
        "response_type": response_type,
        "output_schema": response_type,
        "output_model": response_type,
        "schema": response_type,
        "json_schema": response_type.model_json_schema(),
    }
    try:
        signature = inspect.signature(target)
    except (TypeError, ValueError):
        return target(
            operation=operation,
            request_json=request_json,
            response_schema=response_type,
        )

    args: list[Any] = []
    kwargs: dict[str, Any] = {}
    has_var_positional = False
    has_var_keyword = False
    explicit_parameters = 0
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            has_var_positional = True
            continue
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
            continue
        if parameter.name not in values:
            if parameter.default is inspect.Parameter.empty:
                raise TypeError(
                    f"unsupported required backend parameter {parameter.name!r}; "
                    "use operation, request_json/request/payload, and "
                    "response_schema/response_model/schema"
                )
            continue
        explicit_parameters += 1
        value = values[parameter.name]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args.append(value)
        else:
            kwargs[parameter.name] = value

    if explicit_parameters == 0 and has_var_positional and not has_var_keyword:
        args.extend((operation, request_json, response_type))
    elif explicit_parameters == 0 and has_var_keyword:
        kwargs.update(
            operation=operation,
            request_json=request_json,
            response_schema=response_type,
        )
    return target(*args, **kwargs)


def _require_request_type(
    operation: str,
    request: object,
    request_type: type[StrictSchema],
) -> None:
    if not isinstance(request, request_type):
        raise TypeError(
            f"{operation} requires {request_type.__name__}, got {type(request).__name__}"
        )


def _validate_model_response(
    operation: str,
    response_type: type[ResponseT],
    raw: object,
) -> ResponseT:
    try:
        if isinstance(raw, BaseModel):
            encoded: str | bytes | bytearray = raw.model_dump_json(
                by_alias=True,
                exclude_none=False,
            )
        elif isinstance(raw, (str, bytes, bytearray)):
            encoded = raw
        elif isinstance(raw, Mapping):
            encoded = json.dumps(
                dict(raw),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
        else:
            raise TypeError(
                "response must be a JSON object, JSON text/bytes, or Pydantic model; "
                f"got {type(raw).__name__}"
            )
        parsed = json.loads(encoded)
        if not isinstance(parsed, dict):
            raise TypeError("model response must be a top-level JSON object")
        if parsed.get("schema_version") != "1.0":
            raise ValueError(
                "model response must explicitly include supported schema_version='1.0'"
            )
        # JSON-mode strict validation accepts JSON enum strings while refusing
        # Python-style type coercion such as the string "1" into an integer.
        normalized_json = json.dumps(
            parsed,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return response_type.model_validate_json(normalized_json, strict=True)
    except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ModelOutputError(
            operation,
            response_type,
            f"strict schema validation failed: {exc}",
        ) from exc


def _normalize_base_url(base_url: str) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty HTTP(S) URL")
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must use http or https and include a host")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query string or fragment")
    return normalized


def _validate_timeout(
    timeout: float | tuple[float, float],
) -> float | tuple[float, float]:
    values = timeout if isinstance(timeout, tuple) else (timeout,)
    if len(values) not in {1, 2}:
        raise ValueError("timeout must be a positive number or (connect, read) pair")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value <= 0
        for value in values
    ):
        raise ValueError("timeout values must be positive numbers")
    return timeout


def _validate_headers(headers: Mapping[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("HTTP header names and values must be strings")
        if not name or "\r" in name or "\n" in name or "\r" in value or "\n" in value:
            raise ValueError("HTTP headers must not be empty or contain line breaks")
        validated[name] = value
    return validated


__all__ = [
    "LocalDiscoveryModel",
    "MODEL_ENDPOINTS",
    "ModelOutputError",
    "RemoteDiscoveryModel",
    "StructuredModelBackend",
]
