"""Fixed HTTP clients for specialist encoders and a user-owned fusion model."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from typing import Any, TypeVar
from urllib.parse import urlsplit

import requests

from .fusion_schemas import (
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    FusionGenerationRequest,
    FusionGenerationResponse,
    FusionOutput,
    FusionRequest,
    FusionRevisionProposal,
    FusionRevisionRequest,
)
from .hashing import stable_hash
from .schemas import StrictSchema


ResponseT = TypeVar("ResponseT", bound=StrictSchema)


class FusionAdapterError(RuntimeError):
    def __init__(self, operation: str, message: str) -> None:
        self.operation = operation
        super().__init__(f"{operation}: {message}")


class HttpExpertEncoder:
    """Feature encoder client with a code-owned descriptor and endpoint."""

    ENDPOINT = "/v1/features"

    def __init__(
        self,
        descriptor: ExpertDescriptor,
        base_url: str,
        *,
        timeout: float | tuple[float, float] = (10.0, 300.0),
        headers: Mapping[str, str] | None = None,
        session: Any | None = None,
        max_response_bytes: int = 16 * 1024 * 1024,
        allow_insecure_http: bool = False,
    ) -> None:
        self._descriptor = descriptor
        self.headers = _validate_headers(headers or {})
        self.base_url = _normalize_base_url(
            base_url,
            allow_insecure_http=allow_insecure_http,
            authenticated=_has_authorization(self.headers),
        )
        self.timeout = _validate_timeout(timeout)
        self.max_response_bytes = _positive_size(max_response_bytes)
        self._http = session if session is not None else requests

    @property
    def descriptor(self) -> ExpertDescriptor:
        return self._descriptor

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        descriptor = self.descriptor
        if not descriptor.available:
            raise FusionAdapterError("features", f"expert {descriptor.expert_id!r} unavailable")
        if request.candidate.candidate_type not in descriptor.supported_candidate_types:
            raise FusionAdapterError("features", "candidate type is unsupported")
        if request.modality not in descriptor.modalities:
            raise FusionAdapterError("features", "modality is unsupported")
        if request.feature_space not in descriptor.feature_spaces:
            raise FusionAdapterError("features", "feature space is unsupported")
        kinds = {item.kind for item in request.candidate.representations}
        if not kinds.intersection(descriptor.supported_representations):
            raise FusionAdapterError("features", "candidate has no supported representation")
        if descriptor.routes and not any(
            route.modality == request.modality
            and route.feature_space == request.feature_space
            and (not route.candidate_types or request.candidate.candidate_type in route.candidate_types)
            and bool(kinds.intersection(route.representation_kinds))
            for route in descriptor.routes
        ):
            raise FusionAdapterError("features", "requested expert route is unsupported")

        result = _post_json(
            self._http,
            f"{self.base_url}{self.ENDPOINT}",
            operation="features",
            request=request,
            response_type=ExpertFeaturePayload,
            timeout=self.timeout,
            headers=self.headers,
            max_response_bytes=self.max_response_bytes,
        )
        expected_ref = request.candidate.candidate_ref
        if result.workspace_entity_id != request.workspace_entity_id:
            raise FusionAdapterError("features", "response workspace_entity_id is inconsistent")
        if result.candidate_ref != expected_ref:
            raise FusionAdapterError("features", "response candidate_ref does not match request")
        if result.expert_id != descriptor.expert_id:
            raise FusionAdapterError("features", "response expert_id does not match descriptor")
        if result.modality != request.modality:
            raise FusionAdapterError("features", "response modality does not match request")
        if result.feature_space != request.feature_space:
            raise FusionAdapterError("features", "response feature_space does not match request")
        if result.provenance.expert_id != descriptor.expert_id:
            raise FusionAdapterError("features", "response provenance expert_id is inconsistent")
        if result.provenance.adapter_version != descriptor.adapter_version:
            raise FusionAdapterError("features", "response adapter_version is inconsistent")
        if result.provenance.seed != request.seed:
            raise FusionAdapterError("features", "response seed is inconsistent")
        for metadata_key, provenance_name in (
            ("model_version", "model_version"),
            ("code_revision", "code_revision"),
            ("weight_revision", "weight_revision"),
            ("parameters_hash", "parameters_hash"),
        ):
            expected = descriptor.metadata.get(metadata_key)
            if expected is not None and expected != getattr(
                result.provenance, provenance_name
            ):
                raise FusionAdapterError(
                    "features", f"response provenance {provenance_name} is inconsistent"
                )
        return result


class RemoteFusionBackend:
    """HTTP port for the fusion AI supplied by the user.

    Only the fixed ``/v1/fuse`` and ``/v1/revise`` endpoints are reachable.
    URLs or commands returned by a model are never executed.  Search-control
    fields introduced for the local deterministic backend are excluded from
    the legacy ``fusion-v1`` wire payload unless the caller explicitly opts in.
    """

    FUSE_ENDPOINT = "/v1/fuse"
    REVISE_ENDPOINT = "/v1/revise"
    _LEGACY_FUSE_FIELDS = {
        "schema_version",
        "goal",
        "candidate_ref",
        "workspace",
        "workspace_mode",
        "cycle",
        "seed",
        "features",
        "previous_latent",
        "previous_state_id",
    }
    _LEGACY_REVISE_FIELDS = {
        "schema_version",
        "goal",
        "candidate",
        "state",
        "latent",
        "features",
    }

    def __init__(
        self,
        base_url: str,
        *,
        expected_backend_id: str | None = None,
        expected_backend_version: str | None = None,
        expected_code_revision: str | None = None,
        expected_weight_revision: str | None = None,
        timeout: float | tuple[float, float] = (10.0, 300.0),
        headers: Mapping[str, str] | None = None,
        session: Any | None = None,
        max_response_bytes: int = 16 * 1024 * 1024,
        allow_insecure_http: bool = False,
        send_extended_request_context: bool = False,
    ) -> None:
        if not isinstance(send_extended_request_context, bool):
            raise TypeError("send_extended_request_context must be a bool")
        self.headers = _validate_headers(headers or {})
        self.base_url = _normalize_base_url(
            base_url,
            allow_insecure_http=allow_insecure_http,
            authenticated=_has_authorization(self.headers),
        )
        self.timeout = _validate_timeout(timeout)
        self.max_response_bytes = _positive_size(max_response_bytes)
        self.expected_backend_id = expected_backend_id
        self.expected_backend_version = expected_backend_version
        self.expected_code_revision = expected_code_revision
        self.expected_weight_revision = expected_weight_revision
        self.send_extended_request_context = send_extended_request_context
        self._http = session if session is not None else requests

    def fuse(self, request: FusionRequest) -> FusionOutput:
        result = _post_json(
            self._http,
            f"{self.base_url}{self.FUSE_ENDPOINT}",
            operation="fuse",
            request=request,
            response_type=FusionOutput,
            timeout=self.timeout,
            headers=self.headers,
            max_response_bytes=self.max_response_bytes,
            include_request_fields=(
                None
                if self.send_extended_request_context
                else self._LEGACY_FUSE_FIELDS
            ),
        )
        requested = {item.feature_id for item in request.features}
        returned = set(result.used_feature_ids) | set(result.ignored_feature_ids)
        if not set(result.used_feature_ids).issubset(requested):
            raise FusionAdapterError("fuse", "backend cited an unknown feature_id")
        if returned != requested:
            raise FusionAdapterError("fuse", "backend must account for every input feature")
        expected_provenance = (
            self.expected_backend_id,
            self.expected_backend_version,
            self.expected_code_revision,
            self.expected_weight_revision,
        )
        actual_provenance = (
            result.backend_id,
            result.backend_version,
            result.code_revision,
            result.weight_revision,
        )
        for expected, actual in zip(expected_provenance, actual_provenance, strict=True):
            if expected is not None and expected != actual:
                raise FusionAdapterError("fuse", "backend provenance does not match configuration")
        return result

    def propose_revision(self, request: FusionRevisionRequest) -> FusionRevisionProposal:
        result = _post_json(
            self._http,
            f"{self.base_url}{self.REVISE_ENDPOINT}",
            operation="revise",
            request=request,
            response_type=FusionRevisionProposal,
            timeout=self.timeout,
            headers=self.headers,
            max_response_bytes=self.max_response_bytes,
            include_request_fields=(
                None
                if self.send_extended_request_context
                else self._LEGACY_REVISE_FIELDS
            ),
        )
        if result.parent_candidate_ref != request.state.candidate_ref:
            raise FusionAdapterError("revise", "revision references the wrong candidate")
        if result.state_id != request.state.state_id:
            raise FusionAdapterError("revise", "revision references the wrong latent state")
        return result


class HttpFusionCandidateGenerator:
    """Fixed ``generator-v1`` client used by MatterGen/REINVENT sidecars."""

    ENDPOINT = "/v1/generate"

    def __init__(
        self,
        base_url: str,
        *,
        expected_generator_id: str | None = None,
        expected_generator_version: str | None = None,
        expected_code_revision: str | None = None,
        expected_weight_revision: str | None = None,
        expected_runtime_parameters_hash: str | None = None,
        timeout: float | tuple[float, float] = (10.0, 900.0),
        headers: Mapping[str, str] | None = None,
        session: Any | None = None,
        max_response_bytes: int = 32 * 1024 * 1024,
        allow_insecure_http: bool = False,
    ) -> None:
        self.headers = _validate_headers(headers or {})
        self.base_url = _normalize_base_url(
            base_url,
            allow_insecure_http=allow_insecure_http,
            authenticated=_has_authorization(self.headers),
        )
        self.timeout = _validate_timeout(timeout)
        self.max_response_bytes = _positive_size(max_response_bytes)
        self.expected_generator_id = expected_generator_id
        self.expected_generator_version = expected_generator_version
        self.expected_code_revision = expected_code_revision
        self.expected_weight_revision = expected_weight_revision
        self.expected_runtime_parameters_hash = expected_runtime_parameters_hash
        self._http = session if session is not None else requests

    def generate(self, request: FusionGenerationRequest) -> FusionGenerationResponse:
        if (
            self.expected_generator_id is not None
            and request.run_config.generator_id != self.expected_generator_id
        ):
            raise FusionAdapterError("generate", "run config selects the wrong generator id")
        if (
            self.expected_generator_version is not None
            and request.run_config.generator_version != self.expected_generator_version
        ):
            raise FusionAdapterError("generate", "run config selects the wrong generator version")
        if (
            self.expected_code_revision is not None
            and request.run_config.generator_code_revision != self.expected_code_revision
        ):
            raise FusionAdapterError("generate", "run config selects the wrong code revision")
        if (
            self.expected_weight_revision is not None
            and request.run_config.generator_weight_revision
            != self.expected_weight_revision
        ):
            raise FusionAdapterError("generate", "run config selects the wrong weight revision")
        result = _post_json(
            self._http,
            f"{self.base_url}{self.ENDPOINT}",
            operation="generate",
            request=request,
            response_type=FusionGenerationResponse,
            timeout=self.timeout,
            headers=self.headers,
            max_response_bytes=self.max_response_bytes,
        )
        candidates = result.generated_candidates
        if len(candidates) != request.run_config.candidate_count:
            raise FusionAdapterError("generate", "generator returned the wrong candidate count")
        parent_ref = request.parent_candidate.candidate_ref
        for candidate in candidates:
            if (
                candidate.candidate_ref is None
                or parent_ref is None
                or parent_ref.candidate_id not in candidate.parent_candidate_ids
                or parent_ref not in candidate.parent_candidate_refs
            ):
                raise FusionAdapterError("generate", "generated candidate must cite its parent")
            if candidate.candidate_type not in request.goal.candidate_types:
                raise FusionAdapterError("generate", "generated candidate type is outside the goal")
            if candidate.domain != request.goal.domain:
                raise FusionAdapterError("generate", "generated candidate domain is outside the goal")
        if result.provenance.generator_id != request.run_config.generator_id:
            raise FusionAdapterError("generate", "generator provenance id is inconsistent")
        if result.provenance.generator_version != request.run_config.generator_version:
            raise FusionAdapterError("generate", "generator provenance version is inconsistent")
        if result.provenance.code_revision != request.run_config.generator_code_revision:
            raise FusionAdapterError("generate", "generator code revision is inconsistent")
        if result.provenance.weight_revision != request.run_config.generator_weight_revision:
            raise FusionAdapterError("generate", "generator weight revision is inconsistent")
        if result.provenance.parameters_hash != request.run_config.generator_parameters_hash:
            raise FusionAdapterError("generate", "generator parameters hash is inconsistent")
        if (
            self.expected_runtime_parameters_hash is not None
            and result.provenance.runtime_parameters_hash
            != self.expected_runtime_parameters_hash
        ):
            raise FusionAdapterError(
                "generate", "generator runtime parameters do not match configuration"
            )
        if result.provenance.seed != request.run_config.effective_generator_seed:
            raise FusionAdapterError("generate", "generator seed is inconsistent")
        return result


def _post_json(
    http: Any,
    url: str,
    *,
    operation: str,
    request: StrictSchema,
    response_type: type[ResponseT],
    timeout: float | tuple[float, float],
    headers: Mapping[str, str],
    max_response_bytes: int,
    include_request_fields: set[str] | None = None,
) -> ResponseT:
    payload = request.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=False,
        include=include_request_fields,
    )
    request_headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        **headers,
        "Idempotency-Key": stable_hash(payload),
    }
    try:
        response = http.post(
            url,
            json=payload,
            headers=request_headers,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        )
    except Exception as exc:
        raise FusionAdapterError(
            operation,
            f"HTTP request failed: {type(exc).__name__}: {exc}",
        ) from exc

    try:
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int) and 300 <= status_code < 400:
            raise FusionAdapterError(operation, "HTTP redirects are not allowed")
        try:
            response.raise_for_status()
        except Exception as exc:
            raise FusionAdapterError(
                operation,
                f"HTTP request failed: {type(exc).__name__}: {exc}",
            ) from exc

        response_headers = getattr(response, "headers", {}) or {}
        content_length = response_headers.get("Content-Length") or response_headers.get(
            "content-length"
        )
        if content_length is not None:
            try:
                declared_length = int(content_length)
                if declared_length < 0:
                    raise ValueError
                if declared_length > max_response_bytes:
                    raise FusionAdapterError(operation, "HTTP response exceeds size limit")
            except ValueError as exc:
                raise FusionAdapterError(operation, "invalid Content-Length header") from exc

        if callable(getattr(response, "iter_content", None)):
            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                encoded_chunk = bytes(chunk)
                total += len(encoded_chunk)
                if total > max_response_bytes:
                    raise FusionAdapterError(operation, "HTTP response exceeds size limit")
                chunks.append(encoded_chunk)
            encoded = b"".join(chunks)
        else:
            content = getattr(response, "content", None)
            if content is not None:
                encoded = bytes(content)
            else:
                raw = response.json()
                encoded = json.dumps(
                    raw,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
        if not encoded:
            raise ValueError("response body is empty")
        if len(encoded) > max_response_bytes:
            raise FusionAdapterError(operation, "HTTP response exceeds size limit")
        raw = json.loads(encoded.decode("utf-8"), object_pairs_hook=_unique_object)
        if not isinstance(raw, dict):
            raise TypeError("response must be a top-level JSON object")
        if raw.get("schema_version") != "1.0":
            raise ValueError("response must explicitly include schema_version='1.0'")
        normalized = json.dumps(
            raw,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        return response_type.model_validate_json(normalized, strict=True)
    except FusionAdapterError:
        raise
    except Exception as exc:
        raise FusionAdapterError(
            operation,
            f"response read or strict validation failed: {type(exc).__name__}: {exc}",
        ) from exc
    finally:
        _close_response(response)


def _normalize_base_url(
    base_url: str,
    *,
    allow_insecure_http: bool,
    authenticated: bool = False,
) -> str:
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty HTTP(S) URL")
    normalized = base_url.strip().rstrip("/")
    parsed = urlsplit(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("base_url must use http or https and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("credentials must be supplied as headers, not in base_url")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain a query string or fragment")
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme == "http" and parsed.hostname not in local_hosts and authenticated:
        raise ValueError("authenticated non-local endpoints require HTTPS")
    if parsed.scheme == "http" and parsed.hostname not in local_hosts and not allow_insecure_http:
        raise ValueError("non-local expert endpoints require HTTPS")
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
        or not math.isfinite(float(value))
        or value <= 0
        for value in values
    ):
        raise ValueError("timeout values must be positive numbers")
    return timeout


def _validate_headers(headers: Mapping[str, str]) -> dict[str, str]:
    reserved = {"host", "content-length", "content-type", "idempotency-key"}
    token = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
    result: dict[str, str] = {}
    for name, value in headers.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("HTTP header names and values must be strings")
        if not name or not token.fullmatch(name) or any(
            mark in name + value for mark in ("\r", "\n")
        ):
            raise ValueError("HTTP headers cannot be empty or contain line breaks")
        if name.lower() in reserved:
            raise ValueError(f"HTTP header {name!r} is reserved")
        result[name] = value
    return result


def _has_authorization(headers: Mapping[str, str]) -> bool:
    return any(name.lower() == "authorization" for name in headers)


def _close_response(response: Any) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _positive_size(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    return value


__all__ = [
    "FusionAdapterError",
    "HttpExpertEncoder",
    "HttpFusionCandidateGenerator",
    "RemoteFusionBackend",
]
