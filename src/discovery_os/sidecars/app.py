"""FastAPI application factory for isolated expert and generator sidecars."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from pydantic import ValidationError

from discovery_os.fusion_schemas import (
    DiagnosticProperty,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    ExpertProvenance,
    FeatureSemantics,
    GenerationPairSlot,
    FusionGenerationRequest,
    FusionGenerationResponse,
    GeneratorProvenance,
    NumericTensor,
)
from discovery_os.hashing import candidate_content_hash, stable_hash
from discovery_os.schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    RepresentationKind,
    StrictSchema,
)

from .base import numeric_tensor_data, runtime_provenance_parameters
from .errors import (
    ModelOutputError,
    ModelTimeoutError,
    RequestLimitError,
    SidecarBusyError,
    SidecarError,
)
from .types import ExpertResult, GeneratedBatch, GeneratedCandidateData, ModelIdentity, SidecarLimits


LOGGER = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=StrictSchema)


def create_sidecar_app(
    *,
    identity: ModelIdentity,
    runtime: Any,
    limits: SidecarLimits | None = None,
    title: str | None = None,
) -> Any:
    """Create a strict, bounded FastAPI app around one isolated model runtime.

    FastAPI is imported here, not at module import time, so installing the core
    JSON contracts never pulls a web server into every specialist environment.
    """

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "FastAPI is required only in sidecar environments; install fastapi and uvicorn"
        ) from exc

    configured_limits = limits or SidecarLimits()
    _validate_runtime(identity, runtime)
    executor = _BoundedExecutor(configured_limits)

    @asynccontextmanager
    async def lifespan(_: Any) -> Any:
        try:
            yield
        finally:
            await asyncio.to_thread(executor.shutdown)
            close = getattr(runtime, "close", None)
            if callable(close):
                await asyncio.to_thread(close)

    app = FastAPI(
        title=title or f"Discovery OS {identity.model_id} sidecar",
        version=identity.adapter_version,
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.add_middleware(_BodyLimitMiddleware, max_bytes=configured_limits.max_request_bytes)

    async def health() -> Any:
        loaded = bool(getattr(runtime, "loaded", False))
        failed = bool(getattr(runtime, "load_failed", False))
        supported = bool(getattr(runtime, "supported", True))
        status = "unsupported" if not supported else "error" if failed else "ready" if loaded else "lazy"
        return JSONResponse(
            content={
                "schema_version": "1.0",
                "status": status,
                "ready": supported and not failed,
                "loaded": loaded,
                "model_id": identity.model_id,
                "model_version": identity.model_version,
                "adapter_version": identity.adapter_version,
                "code_revision": identity.code_revision,
                "weight_revision": identity.weight_revision,
                "capabilities": sorted(identity.capabilities),
                "device": str(getattr(runtime, "device", "unknown")),
                "limits": {
                    "max_request_bytes": configured_limits.max_request_bytes,
                    "max_batch_size": configured_limits.max_batch_size,
                    "max_concurrency": configured_limits.max_concurrency,
                    "max_queue_size": configured_limits.max_queue_size,
                    "request_timeout_seconds": configured_limits.request_timeout_seconds,
                },
            }
        )

    app.get("/health", include_in_schema=True)(health)

    if "features" in identity.capabilities:

        async def features(request: Any) -> Any:
            try:
                feature_request = await _strict_request(
                    request,
                    ExpertFeatureRequest,
                    max_bytes=configured_limits.max_request_bytes,
                )
                raw = await executor.run(runtime.encode, feature_request)
                if not isinstance(raw, ExpertResult):
                    raise ModelOutputError("expert runtime must return ExpertResult")
                payload = _feature_payload(identity, runtime, feature_request, raw)
                return JSONResponse(content=payload.model_dump(mode="json", exclude_none=False))
            except SidecarError as exc:
                return _error_response(JSONResponse, exc)
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                LOGGER.info("rejected invalid feature request: %s", type(exc).__name__)
                return _error_response(
                    JSONResponse,
                    RequestLimitError("request failed strict expert-feature-v1 validation"),
                    status_code=422,
                    code="invalid_request",
                )
            except Exception:
                LOGGER.exception("unhandled expert sidecar failure")
                return _error_response(
                    JSONResponse,
                    SidecarError("expert runtime failed; inspect sidecar logs"),
                )

        features.__annotations__["request"] = Request
        app.post("/v1/features", include_in_schema=True)(features)

    if "generate" in identity.capabilities:

        async def generate(request: Any) -> Any:
            try:
                generation_request = await _strict_request(
                    request,
                    FusionGenerationRequest,
                    max_bytes=configured_limits.max_request_bytes,
                )
                count = generation_request.run_config.candidate_count
                if count > configured_limits.max_batch_size:
                    raise RequestLimitError(
                        f"candidate_count {count} exceeds sidecar max_batch_size "
                        f"{configured_limits.max_batch_size}"
                    )
                _bind_generator_request(identity, generation_request)
                raw = await executor.run(runtime.generate, generation_request)
                if not isinstance(raw, GeneratedBatch):
                    raise ModelOutputError("generator runtime must return GeneratedBatch")
                response = _generation_response(identity, runtime, generation_request, raw)
                return JSONResponse(content=response.model_dump(mode="json", exclude_none=False))
            except SidecarError as exc:
                return _error_response(JSONResponse, exc)
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError) as exc:
                LOGGER.info("rejected invalid generation request: %s", type(exc).__name__)
                return _error_response(
                    JSONResponse,
                    RequestLimitError("request failed strict generator-v1 validation"),
                    status_code=422,
                    code="invalid_request",
                )
            except Exception:
                LOGGER.exception("unhandled generator sidecar failure")
                return _error_response(
                    JSONResponse,
                    SidecarError("generator runtime failed; inspect sidecar logs"),
                )

        generate.__annotations__["request"] = Request
        app.post("/v1/generate", include_in_schema=True)(generate)

    return app


def _feature_payload(
    identity: ModelIdentity,
    runtime: Any,
    request: ExpertFeatureRequest,
    result: ExpertResult,
) -> ExpertFeaturePayload:
    tensor = None
    semantics = None
    if result.values is not None:
        shape, values = numeric_tensor_data(result.values)
        if result.entity_ids and shape[0] != len(result.entity_ids):
            raise ModelOutputError("expert entity_ids do not match the tensor's first axis")
        tensor = NumericTensor(shape=shape, values=values)
        semantics = FeatureSemantics(
            tensor_role=result.tensor_role,
            projection_id=result.projection_id,
            entity_type=result.entity_type,
            entity_ids=list(result.entity_ids),
            mask=list(result.mask),
            pooling=result.pooling,
            normalization=result.normalization,
            coordinate_frame=result.coordinate_frame,
            basis=result.basis,
            unit_semantics=result.unit_semantics,
        )
    properties = [
        DiagnosticProperty(
            property_name=item.property_name,
            value=item.value,
            unit=item.unit,
            uncertainty=item.uncertainty,
            out_of_domain=item.out_of_domain,
            source=item.source,
        )
        for item in result.properties
    ]
    parameters = runtime_provenance_parameters(runtime)
    payload = ExpertFeaturePayload(
        workspace_entity_id=request.workspace_entity_id,
        candidate_ref=request.candidate.candidate_ref,
        expert_id=identity.model_id,
        modality=request.modality,
        feature_space=request.feature_space,
        status=result.status,
        tensor=tensor,
        semantics=semantics,
        properties=properties,
        quality_flags=list(result.quality_flags),
        warnings=list(result.warnings),
        provenance=ExpertProvenance(
            expert_id=identity.model_id,
            adapter_version=identity.adapter_version,
            model_version=identity.model_version,
            code_revision=identity.code_revision,
            weight_revision=identity.weight_revision,
            projection_version=identity.projection_version or result.projection_id,
            parameters_hash=stable_hash(parameters),
            device=str(getattr(runtime, "device", "unknown")),
            seed=request.seed,
        ),
    )
    return _strict_roundtrip(payload)


def _generation_response(
    identity: ModelIdentity,
    runtime: Any,
    request: FusionGenerationRequest,
    result: GeneratedBatch,
) -> FusionGenerationResponse:
    count = request.run_config.candidate_count
    if len(result.candidates) != count:
        raise ModelOutputError(
            f"generator runtime returned {len(result.candidates)} candidates, expected {count}"
        )
    output_type, required_representation, allowed_domains = _generator_output_contract(
        identity.model_id
    )
    if output_type not in request.goal.candidate_types:
        raise ModelOutputError(
            f"goal does not allow the {output_type!s} output produced by {identity.model_id}"
        )
    if request.goal.domain not in allowed_domains:
        raise ModelOutputError(
            f"{identity.model_id} is not configured for goal domain {request.goal.domain!s}"
        )
    for generated in result.candidates:
        if required_representation not in {item.kind for item in generated.representations}:
            raise ModelOutputError(
                f"{identity.model_id} output requires representation "
                f"{required_representation!s}"
            )
    runtime_parameters = runtime_provenance_parameters(runtime)
    runtime_parameters_hash = stable_hash(runtime_parameters)
    scientific_identities = [
        stable_hash(
            {
                "candidate_type": output_type,
                "domain": request.goal.domain,
                "representations": [
                    _candidate_identity_representation(item)
                    for item in generated.representations
                ],
            }
        )
        for generated in result.candidates
    ]
    if len(scientific_identities) != len(set(scientific_identities)):
        raise ModelOutputError(
            "generator returned duplicate scientific outputs in one candidate batch"
        )
    candidates = [
        _content_address_candidate(
            identity,
            request,
            item,
            output_type=output_type,
            runtime_parameters_hash=runtime_parameters_hash,
        )
        for item in result.candidates
    ]
    response = FusionGenerationResponse(
        candidates=candidates,
        provenance=GeneratorProvenance(
            generator_id=identity.model_id,
            generator_version=identity.model_version,
            code_revision=identity.code_revision,
            weight_revision=identity.weight_revision,
            parameters_hash=request.run_config.generator_parameters_hash,
            runtime_parameters_hash=runtime_parameters_hash,
            seed=request.run_config.effective_generator_seed,
        ),
        pair_slots=[
            GenerationPairSlot(
                pair_slot=index,
                candidate_ref=candidate.candidate_ref,
                batch_seed=request.run_config.effective_generator_seed,
                stream_position=index,
            )
            for index, candidate in enumerate(candidates)
            if candidate.candidate_ref is not None
        ],
        warnings=list(result.warnings),
    )
    return _strict_roundtrip(response)


def _content_address_candidate(
    identity: ModelIdentity,
    request: FusionGenerationRequest,
    data: GeneratedCandidateData,
    *,
    output_type: CandidateType,
    runtime_parameters_hash: str,
) -> Candidate:
    parent_ref = request.parent_candidate.candidate_ref
    if parent_ref is None:
        raise ModelOutputError("generation request parent has no immutable reference")
    digest = stable_hash(
        {
            "generator": {
                "id": identity.model_id,
                "version": identity.model_version,
                "code_revision": identity.code_revision,
                "weight_revision": identity.weight_revision,
                "runtime_parameters_hash": runtime_parameters_hash,
            },
            "parent": parent_ref,
            "goal_hash": request.run_config.goal_hash,
            "candidate_type": output_type,
            "domain": request.goal.domain,
            "pair_key": request.run_config.pair_key,
            "seed": request.run_config.effective_generator_seed,
            "cohort": request.run_config.cohort_index,
            "parameters_hash": request.run_config.generator_parameters_hash,
            "decoder_config_hash": request.run_config.decoder_config_hash,
            "postprocessing_hash": request.run_config.postprocessing_hash,
            "representations": [
                _candidate_identity_representation(item) for item in data.representations
            ],
            "attributes": data.attributes,
            "generator_provenance": data.provenance,
        }
    )
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity.model_id).strip("-") or "candidate"
    candidate_id = f"{prefix}-{digest[:24]}"
    provenance = {
        **data.provenance,
        "generator_id": identity.model_id,
        "generator_version": identity.model_version,
        "code_revision": identity.code_revision,
        "weight_revision": identity.weight_revision,
        "parameters_hash": request.run_config.generator_parameters_hash,
        "runtime_parameters_hash": runtime_parameters_hash,
        "seed": request.run_config.effective_generator_seed,
    }
    candidate = Candidate(
        candidate_id=candidate_id,
        candidate_type=output_type,
        domain=request.goal.domain,
        name=data.name,
        representations=list(data.representations),
        parent_candidate_ids=[parent_ref.candidate_id],
        parent_candidate_refs=[parent_ref],
        generation_task_id=request.run_config.pair_key,
        attributes=data.attributes,
        novelty_rationale=data.novelty_rationale,
        provenance=provenance,
    )
    reference = CandidateRef(
        candidate_id=candidate_id,
        version=1,
        content_hash=candidate_content_hash(candidate),
    )
    return _strict_roundtrip(candidate.model_copy(update={"candidate_ref": reference}))


def _generator_output_contract(
    model_id: str,
) -> tuple[CandidateType, RepresentationKind, frozenset[DiscoveryDomain]]:
    """Return the reviewed scientific output type, representation, and domains."""

    if model_id == "mattergen":
        return (
            CandidateType.CRYSTAL,
            RepresentationKind.CIF,
            frozenset(
                {
                    DiscoveryDomain.INORGANIC_MATERIALS,
                    DiscoveryDomain.SUPERCONDUCTORS,
                    DiscoveryDomain.BATTERIES,
                    DiscoveryDomain.CATALYSTS,
                    DiscoveryDomain.GENERAL_MATERIALS,
                }
            ),
        )
    if model_id == "reinvent4":
        return (
            CandidateType.SMALL_MOLECULE,
            RepresentationKind.SMILES,
            frozenset(
                {
                    DiscoveryDomain.MEDICINAL_CHEMISTRY,
                    DiscoveryDomain.POLYMERS,
                    DiscoveryDomain.BATTERIES,
                    DiscoveryDomain.CATALYSTS,
                    DiscoveryDomain.GENERAL_MATERIALS,
                }
            ),
        )
    raise ModelOutputError(f"no reviewed output contract exists for generator {model_id!r}")


def _candidate_identity_representation(item: CandidateRepresentation) -> dict[str, Any]:
    """Return scientific representation content without batch-output filenames."""

    payload = item.model_dump(mode="json")
    metadata = dict(payload.get("metadata") or {})
    metadata.pop("source_entry", None)
    payload["metadata"] = metadata
    return payload


def _bind_generator_request(identity: ModelIdentity, request: FusionGenerationRequest) -> None:
    config = request.run_config
    expected = (
        identity.model_id,
        identity.model_version,
        identity.code_revision,
        identity.weight_revision,
    )
    actual = (
        config.generator_id,
        config.generator_version,
        config.generator_code_revision,
        config.generator_weight_revision,
    )
    if expected != actual:
        raise ModelOutputError("generation run_config does not match this sidecar's bound identity")


def _validate_runtime(identity: ModelIdentity, runtime: Any) -> None:
    if "features" in identity.capabilities and not callable(getattr(runtime, "encode", None)):
        raise TypeError("features capability requires runtime.encode(request)")
    if "generate" in identity.capabilities and not callable(getattr(runtime, "generate", None)):
        raise TypeError("generate capability requires runtime.generate(request)")
    if identity.runtime_parameters_hash is not None:
        if stable_hash(runtime_provenance_parameters(runtime)) != identity.runtime_parameters_hash:
            raise TypeError("runtime parameters changed after sidecar identity was bound")


async def _strict_request(request: Any, schema: type[SchemaT], *, max_bytes: int) -> SchemaT:
    body = await request.body()
    if not body or len(body) > max_bytes:
        raise RequestLimitError("request body is empty or exceeds the configured byte limit")
    raw = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0":
        raise ValueError("request must explicitly include schema_version='1.0'")
    normalized = json.dumps(raw, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return schema.model_validate_json(normalized, strict=True)


def _strict_roundtrip(value: SchemaT) -> SchemaT:
    encoded = value.model_dump_json(exclude_none=False)
    return type(value).model_validate_json(encoded, strict=True)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _error_response(
    response_class: Any,
    error: SidecarError,
    *,
    status_code: int | None = None,
    code: str | None = None,
) -> Any:
    return response_class(
        status_code=status_code or error.status_code,
        content={
            "schema_version": "1.0",
            "error": {"code": code or error.error_code, "message": error.safe_message},
        },
    )


class _BoundedExecutor:
    def __init__(self, limits: SidecarLimits) -> None:
        self._limits = limits
        self._executor = ThreadPoolExecutor(
            max_workers=limits.max_concurrency,
            thread_name_prefix="discovery-sidecar",
        )
        self._slots = threading.BoundedSemaphore(
            limits.max_concurrency + limits.max_queue_size
        )

    async def run(self, function: Any, *args: Any) -> Any:
        if not self._slots.acquire(blocking=False):
            raise SidecarBusyError("sidecar concurrency and queue capacity are exhausted")
        try:
            future = self._executor.submit(function, *args)
        except BaseException:
            self._slots.release()
            raise
        future.add_done_callback(lambda _: self._slots.release())
        wrapped = asyncio.wrap_future(future)
        try:
            return await asyncio.wait_for(
                asyncio.shield(wrapped), timeout=self._limits.request_timeout_seconds
            )
        except TimeoutError as exc:
            # The worker continues to completion and retains its capacity slot.
            # Python model calls cannot be killed safely; CLI adapters also have
            # process-level timeouts that terminate their subprocesses.
            raise ModelTimeoutError(
                f"model call exceeded {self._limits.request_timeout_seconds:g} seconds; "
                "the bounded worker remains occupied until the upstream call exits"
            ) from exc

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)


class _BodyTooLarge(Exception):
    pass


class _BodyLimitMiddleware:
    """ASGI receive wrapper that bounds both Content-Length and chunked bodies."""

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        declared = headers.get(b"content-length")
        if declared is not None:
            try:
                length = int(declared)
            except ValueError:
                await _send_asgi_error(send, 400, "invalid_content_length", "invalid Content-Length")
                return
            if length < 0 or length > self.max_bytes:
                await _send_asgi_error(
                    send,
                    413,
                    "request_limit_exceeded",
                    "request body exceeds the configured byte limit",
                )
                return
        received = 0

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _send_asgi_error(
                send,
                413,
                "request_limit_exceeded",
                "request body exceeds the configured byte limit",
            )


async def _send_asgi_error(send: Any, status: int, code: str, message: str) -> None:
    body = json.dumps(
        {"schema_version": "1.0", "error": {"code": code, "message": message}},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


__all__ = ["create_sidecar_app"]
