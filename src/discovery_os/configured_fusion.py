"""Environment bindings for the fusion controller and generator sidecars.

When no remote fusion endpoint is configured, orchestration uses the local,
weight-free evidence controller.  Remote endpoints are accepted only from
code-owned environment variable names.  Plain HTTP is limited to loopback
development endpoints and requires an explicit opt-in; deployed services must
use HTTPS.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from urllib.parse import urlsplit

from .fusion_adapters import HttpFusionCandidateGenerator, RemoteFusionBackend
from .fusion_protocols import FusionBackend
from .integration_manifest import load_integration_manifest


INSECURE_LOCAL_HTTP_ENV = "DISCOVERY_ALLOW_INSECURE_LOCAL_HTTP"
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class FusionConfigurationError(ValueError):
    """Raised when an environment binding violates the fixed API policy."""


def build_fusion_backend_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    required: bool = True,
) -> FusionBackend:
    """Build the configured remote backend or the local evidence controller.

    An unset or whitespace-only ``FUSION_API_URL`` selects
    :class:`EvidenceDrivenFusionBackend`; no learned weights or endpoint are
    required for that deterministic controller.  A configured URL preserves
    the existing ``RemoteFusionBackend`` behavior: ``FUSION_API_TOKEN`` is
    optional and, when present, is sent as a bearer token.  Local HTTP testing
    additionally requires either ``FUSION_ALLOW_INSECURE_LOCAL_HTTP`` or the
    process-wide ``DISCOVERY_ALLOW_INSECURE_LOCAL_HTTP`` flag.

    ``required`` remains accepted for source compatibility.  It no longer
    makes a missing Fusion URL an error because a usable local backend is
    always available; it still documents call-site intent.  Remote requests
    retain the legacy ``fusion-v1`` payload by default.  The local-only search
    context fields are sent only with the explicit
    ``FUSION_SEND_EXTENDED_REQUEST_CONTEXT=1`` opt-in.
    """

    values = environ if environ is not None else os.environ
    configured_url = values.get("FUSION_API_URL")
    if configured_url is None or not configured_url.strip():
        # Keep this import lazy so the remote-only configuration path remains
        # independent from the local controller implementation at import time.
        from .evidence_fusion import EvidenceDrivenFusionBackend

        return EvidenceDrivenFusionBackend()

    base_url = _configured_url(
        values,
        url_env="FUSION_API_URL",
        required=True,
    )
    assert base_url is not None  # FUSION_API_URL was checked immediately above.
    provenance_values = {
        "backend_id": values.get("FUSION_BACKEND_ID"),
        "backend_version": values.get("FUSION_BACKEND_VERSION"),
        "code_revision": values.get("FUSION_BACKEND_CODE_REVISION"),
        "weight_revision": values.get("FUSION_BACKEND_WEIGHT_REVISION"),
    }
    supplied = {key: value.strip() for key, value in provenance_values.items() if value and value.strip()}
    if supplied and len(supplied) != len(provenance_values):
        raise FusionConfigurationError(
            "fusion backend provenance must set ID, VERSION, CODE_REVISION, and WEIGHT_REVISION together"
        )
    return RemoteFusionBackend(
        base_url,
        expected_backend_id=supplied.get("backend_id"),
        expected_backend_version=supplied.get("backend_version"),
        expected_code_revision=supplied.get("code_revision"),
        expected_weight_revision=supplied.get("weight_revision"),
        headers=_authorization_headers(values.get("FUSION_API_TOKEN")),
        allow_insecure_http=False,
        send_extended_request_context=_truthy(
            values.get("FUSION_SEND_EXTENDED_REQUEST_CONTEXT")
        ),
    )


def build_generator_from_environment(
    component_id: str,
    *,
    environ: Mapping[str, str] | None = None,
    required: bool = True,
) -> HttpFusionCandidateGenerator | None:
    """Build one manifest allow-listed ``generator-v1`` sidecar client.

    The URL variable is read from the component's integration-manifest entry.
    Its token variable is the matching ``*_API_TOKEN`` name (for example,
    ``MATTERGEN_API_TOKEN`` or ``REINVENT_API_TOKEN``).
    """

    values = environ if environ is not None else os.environ
    component = next(
        (
            item
            for item in load_integration_manifest().components
            if item.component_id == component_id
        ),
        None,
    )
    if component is None:
        raise FusionConfigurationError(
            f"generator component {component_id!r} is not in the integration manifest"
        )
    if component.api is None or component.api.protocol != "generator-v1":
        raise FusionConfigurationError(
            f"component {component_id!r} is not an allow-listed generator-v1 service"
        )

    url_env = component.api.base_url_env
    base_url = _configured_url(values, url_env=url_env, required=required)
    if base_url is None:
        return None
    token_env = _token_env_for_url(url_env)
    expected_version = component.install.version or (
        component.source.release if component.source is not None else None
    )
    expected_weight_revision = _component_weight_revision(
        component,
        values,
        url_env=url_env,
    )
    return HttpFusionCandidateGenerator(
        base_url,
        expected_generator_id=component.component_id,
        expected_generator_version=expected_version,
        expected_code_revision=(
            component.source.revision if component.source is not None else None
        ),
        expected_weight_revision=expected_weight_revision,
        expected_runtime_parameters_hash=_optional_sha256(
            values,
            f"{url_env.removesuffix('_API_URL')}_RUNTIME_PARAMETERS_HASH",
        ),
        headers=_authorization_headers(values.get(token_env)),
        allow_insecure_http=False,
    )


def build_generators_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, HttpFusionCandidateGenerator]:
    """Build every configured manifest generator, omitting unset endpoints."""

    values = environ if environ is not None else os.environ
    result: dict[str, HttpFusionCandidateGenerator] = {}
    for component in load_integration_manifest().components:
        if component.api is None or component.api.protocol != "generator-v1":
            continue
        client = build_generator_from_environment(
            component.component_id,
            environ=values,
            required=False,
        )
        if client is not None:
            result[component.component_id] = client
    return result


def _configured_url(
    environ: Mapping[str, str],
    *,
    url_env: str,
    required: bool,
) -> str | None:
    raw = environ.get(url_env)
    if raw is None or not raw.strip():
        if required:
            raise FusionConfigurationError(f"{url_env} is required")
        return None

    base_url = raw.strip()
    parsed = urlsplit(base_url)
    if parsed.scheme == "http":
        if parsed.hostname not in _LOCAL_HOSTS:
            raise FusionConfigurationError(
                f"{url_env} must use HTTPS; plain HTTP is restricted to loopback hosts"
            )
        component_flag = f"{url_env.removesuffix('_API_URL')}_ALLOW_INSECURE_LOCAL_HTTP"
        if not (
            _truthy(environ.get(component_flag))
            or _truthy(environ.get(INSECURE_LOCAL_HTTP_ENV))
        ):
            raise FusionConfigurationError(
                f"{url_env} uses local HTTP; set {component_flag}=1 "
                f"or {INSECURE_LOCAL_HTTP_ENV}=1 to opt in"
            )
    return base_url


def _token_env_for_url(url_env: str) -> str:
    if not url_env.endswith("_URL"):
        raise FusionConfigurationError(
            f"manifest API environment name {url_env!r} cannot derive a token name"
        )
    return f"{url_env[:-4]}_TOKEN"


def _component_weight_revision(component, environ: Mapping[str, str], *, url_env: str) -> str:
    env_name = f"{url_env.removesuffix('_API_URL')}_WEIGHT_REVISION"
    configured = environ.get(env_name)
    if configured and configured.strip():
        return configured.strip()
    exact = {item.revision for item in component.weights if item.revision is not None}
    unresolved = [item for item in component.weights if item.revision is None]
    if len(exact) == 1 and not unresolved:
        return next(iter(exact))
    if not component.weights:
        return "no-external-weight"
    raise FusionConfigurationError(
        f"{env_name} is required because {component.component_id!r} uses managed/manual or ambiguous weights"
    )


def _authorization_headers(token: str | None) -> dict[str, str] | None:
    if token is None or not token.strip():
        return None
    return {"Authorization": f"Bearer {token.strip()}"}


def _optional_sha256(environ: Mapping[str, str], name: str) -> str | None:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise FusionConfigurationError(f"{name} must be a lowercase SHA-256")
    return value


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes"})


__all__ = [
    "FusionConfigurationError",
    "INSECURE_LOCAL_HTTP_ENV",
    "build_fusion_backend_from_environment",
    "build_generator_from_environment",
    "build_generators_from_environment",
]
