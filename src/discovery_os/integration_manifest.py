"""Typed, non-executable manifest for optional scientific integrations."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from ._compat import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator

from .fusion_schemas import ScientificModality
from .hashing import stable_hash
from .schemas import Identifier, NonEmptyText, StrictSchema


_PIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?==[^\s=<>!~]+$")
_PYTHON = re.compile(r"^3\.\d{1,2}$")
_SLUG = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,126}[A-Za-z0-9])?$")
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}


class InstallKind(StrEnum):
    LOCAL_PROJECT = "local_project"
    PYPI = "pypi"
    GITHUB_ARCHIVE = "github_archive"
    REMOTE_API = "remote_api"


class InstallSpec(StrictSchema):
    kind: InstallKind
    python: str | None = Field(default=None, max_length=16)
    package: str | None = Field(default=None, max_length=256)
    version: str | None = Field(default=None, max_length=128)
    import_name: str | None = Field(default=None, pattern=r"^[A-Za-z_][A-Za-z0-9_.]*$")
    local_path: str | None = Field(default=None, max_length=1_024)
    constraints: list[str] = Field(default_factory=list)
    cuda_extra: str | None = Field(default=None, max_length=128)
    extra_index_urls: list[str] = Field(default_factory=list)
    find_links: list[str] = Field(default_factory=list)
    archive_url: str | None = Field(default=None, max_length=2_048)
    archive_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    archive_size_bytes: int | None = Field(default=None, gt=0)
    archive_subdirectory: str | None = Field(default=None, max_length=1_024)
    install_local: bool = True

    @model_validator(mode="after")
    def _validate_install_shape(self) -> InstallSpec:
        if self.python is not None and not _PYTHON.fullmatch(self.python):
            raise ValueError("python must pin a major.minor version such as 3.11")
        for constraint in self.constraints:
            if not _PIN.fullmatch(constraint):
                raise ValueError("constraints must use exact package==version pins")
        for url in [*self.extra_index_urls, *self.find_links]:
            _https_url(url)
            if urlsplit(url).hostname not in {
                "download.pytorch.org",
                "data.pyg.org",
                "pypi.org",
            }:
                raise ValueError("package indexes and find-links hosts are not allow-listed")
        if self.kind == InstallKind.LOCAL_PROJECT:
            if self.python is None or self.local_path is None:
                raise ValueError("local_project requires python and local_path")
            _safe_relative(self.local_path)
        elif self.kind == InstallKind.PYPI:
            if self.python is None or self.package is None or self.version is None:
                raise ValueError("pypi install requires python, package, and version")
            if any(mark in self.version for mark in " <>=!~*"):
                raise ValueError("pypi version must be an exact version, not a range")
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", self.package):
                raise ValueError("pypi package name is invalid")
        elif self.kind == InstallKind.GITHUB_ARCHIVE:
            if (
                self.python is None
                or self.archive_url is None
                or self.archive_sha256 is None
                or self.archive_size_bytes is None
            ):
                raise ValueError("github_archive requires python, URL, SHA-256, and size")
            _https_url(self.archive_url)
            if urlsplit(self.archive_url).hostname != "codeload.github.com":
                raise ValueError("source archives must use codeload.github.com")
            if self.archive_subdirectory:
                _safe_relative(self.archive_subdirectory)
        elif self.kind == InstallKind.REMOTE_API:
            forbidden = [
                self.python,
                self.package,
                self.version,
                self.import_name,
                self.local_path,
                self.archive_url,
                self.archive_sha256,
            ]
            if any(item is not None for item in forbidden):
                raise ValueError("remote_api cannot contain local install instructions")
        return self


class SourceSpec(StrictSchema):
    repository: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    release: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def _source_is_https(self) -> SourceSpec:
        _https_url(self.repository)
        if urlsplit(self.repository).hostname != "github.com":
            raise ValueError("source repositories must use github.com")
        return self


class LicenseSpec(StrictSchema):
    code_license: NonEmptyText
    code_license_url: str | None = Field(default=None, max_length=2_048)
    model_license: str | None = Field(default=None, max_length=512)
    model_license_url: str | None = Field(default=None, max_length=2_048)
    requires_acceptance: bool = False
    acceptance_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]+$")

    @model_validator(mode="after")
    def _license_urls_and_acceptance(self) -> LicenseSpec:
        if self.code_license_url is not None:
            _https_url(self.code_license_url)
        if self.model_license_url is not None:
            _https_url(self.model_license_url)
        if self.requires_acceptance and self.acceptance_env is None:
            raise ValueError("restricted licenses require an explicit acceptance_env")
        return self


class WeightSpec(StrictSchema):
    weight_id: Identifier
    kind: Literal["huggingface", "https", "managed", "manual"]
    repository: str | None = Field(default=None, max_length=512)
    revision: str | None = Field(default=None, pattern=r"^[0-9a-f]{40}$")
    download_url: str | None = Field(default=None, max_length=2_048)
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    expected_size_bytes: int | None = Field(default=None, ge=0)
    gated: bool = False
    token_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]+$")
    acceptance_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]+$")
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_weight_source(self) -> WeightSpec:
        _safe_slug(self.weight_id)
        if self.kind == "huggingface":
            if self.repository is None or self.revision is None:
                raise ValueError("Hugging Face weights require repository and revision")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", self.repository):
                raise ValueError("Hugging Face repository must use owner/name")
        elif self.kind == "https":
            if (
                self.revision is None
                or self.download_url is None
                or self.sha256 is None
                or self.expected_size_bytes is None
                or self.expected_size_bytes <= 0
            ):
                raise ValueError(
                    "HTTPS weights require an immutable revision, URL, SHA-256, and positive size"
                )
        elif self.kind == "manual" and self.download_url is None:
            raise ValueError("manual weights require download_url")
        if self.download_url is not None:
            _https_url(self.download_url)
        if self.gated and (self.token_env is None or self.acceptance_env is None):
            raise ValueError("gated weights require token_env and acceptance_env")
        return self


class ApiBinding(StrictSchema):
    protocol: Literal["expert-feature-v1", "generator-v1", "tool-v1", "fusion-v1"]
    base_url_env: str = Field(pattern=r"^[A-Z][A-Z0-9_]+$")
    default_port: int = Field(gt=0, le=65_535)


class ResourceHint(StrictSchema):
    storage_gb: float = Field(ge=0.0)
    memory_gb: float = Field(ge=0.0)
    vram_gb: float = Field(ge=0.0)


class IntegrationComponent(StrictSchema):
    component_id: Identifier
    display_name: NonEmptyText
    role: Literal["core", "generator", "expert", "predictor", "quantum", "fusion"]
    modalities: list[ScientificModality] = Field(default_factory=list)
    platforms: list[Literal["windows", "linux", "darwin"]] = Field(min_length=1)
    accelerators: list[Literal["cpu", "cuda", "mps", "remote"]] = Field(min_length=1)
    dependencies: list[Identifier] = Field(default_factory=list)
    install: InstallSpec
    source: SourceSpec | None = None
    license: LicenseSpec
    weights: list[WeightSpec] = Field(default_factory=list)
    api: ApiBinding | None = None
    resources: ResourceHint
    status: Literal["stable", "research", "legacy", "user_supplied"]
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _component_lists_are_unique(self) -> IntegrationComponent:
        _safe_slug(self.component_id)
        for dependency in self.dependencies:
            _safe_slug(dependency)
        for label, values in (
            ("platforms", self.platforms),
            ("accelerators", self.accelerators),
            ("dependencies", self.dependencies),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {label} values are not allowed")
        weight_ids = [item.weight_id for item in self.weights]
        if len(weight_ids) != len(set(weight_ids)):
            raise ValueError("duplicate weight ids are not allowed")
        return self


class IntegrationProfile(StrictSchema):
    description: NonEmptyText
    components: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def _profile_components_are_safe(self) -> IntegrationProfile:
        for component in self.components:
            _safe_slug(component)
        return self


class IntegrationManifest(StrictSchema):
    manifest_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: str = Field(min_length=1, max_length=128)
    resolution_cutoff: str = Field(min_length=1, max_length=128)
    uv_version: str = Field(min_length=1, max_length=128)
    huggingface_hub_version: str = Field(min_length=1, max_length=128)
    profiles: dict[str, IntegrationProfile]
    components: list[IntegrationComponent] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_graph(self) -> IntegrationManifest:
        try:
            generated = datetime.fromisoformat(self.generated_at.replace("Z", "+00:00"))
            cutoff = datetime.fromisoformat(self.resolution_cutoff.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("manifest timestamps must be RFC 3339 datetimes") from exc
        if generated.tzinfo is None or cutoff.tzinfo is None:
            raise ValueError("manifest timestamps must include a timezone")
        if cutoff > generated:
            raise ValueError("resolution_cutoff cannot be later than generated_at")
        if generated.astimezone(timezone.utc) > datetime.now(timezone.utc):
            raise ValueError("manifest generated_at cannot be in the future")
        component_by_id = {item.component_id: item for item in self.components}
        if len(component_by_id) != len(self.components):
            raise ValueError("duplicate component_id values are not allowed")
        for component in self.components:
            unknown = set(component.dependencies) - set(component_by_id)
            if unknown:
                raise ValueError(
                    f"component {component.component_id!r} has unknown dependencies: {sorted(unknown)}"
                )
        for name, profile in self.profiles.items():
            _safe_slug(name)
            unknown = set(profile.components) - set(component_by_id)
            if unknown:
                raise ValueError(f"profile {name!r} cites unknown components: {sorted(unknown)}")
            if len(profile.components) != len(set(profile.components)):
                raise ValueError(f"profile {name!r} contains duplicate components")
        _topological(component_by_id, component_by_id)
        return self

    def resolve_profile(self, profile_name: str) -> list[IntegrationComponent]:
        try:
            requested = self.profiles[profile_name].components
        except KeyError as exc:
            raise KeyError(f"unknown integration profile {profile_name!r}") from exc
        component_by_id = {item.component_id: item for item in self.components}
        selected: dict[str, IntegrationComponent] = {}

        def include(component_id: str) -> None:
            if component_id in selected:
                return
            component = component_by_id[component_id]
            for dependency in component.dependencies:
                include(dependency)
            selected[component_id] = component

        for component_id in requested:
            include(component_id)
        return _topological(selected, component_by_id)


def load_integration_manifest(path: str | Path | None = None) -> IntegrationManifest:
    if path is not None:
        manifest_path = Path(path)
    else:
        configured = os.getenv("DISCOVERY_INTEGRATION_MANIFEST")
        candidates = [
            Path(configured) if configured else None,
            Path(__file__).resolve().parents[2] / "integrations" / "manifest.v1.json",
            Path(sys.prefix) / "share" / "discovery-os" / "integrations" / "manifest.v1.json",
        ]
        manifest_path = next(
            (item for item in candidates if item is not None and item.is_file()),
            Path("integrations/manifest.v1.json"),
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("integration manifest must be a JSON object")
    material = dict(payload)
    revision = material.pop("manifest_revision", None)
    if not isinstance(revision, str) or stable_hash(material) != revision:
        raise ValueError("integration manifest revision does not match its content")
    return IntegrationManifest.model_validate(payload)


def _topological(
    selected: dict[str, IntegrationComponent],
    all_components: dict[str, IntegrationComponent],
) -> list[IntegrationComponent]:
    result: list[IntegrationComponent] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(component_id: str) -> None:
        if component_id in visited or component_id not in selected:
            return
        if component_id in visiting:
            raise ValueError("integration dependency graph contains a cycle")
        visiting.add(component_id)
        for dependency in all_components[component_id].dependencies:
            visit(dependency)
        visiting.remove(component_id)
        visited.add(component_id)
        result.append(all_components[component_id])

    for component_id in selected:
        visit(component_id)
    return result


def _https_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("integration URLs must use HTTPS and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("integration URLs cannot embed credentials")
    if parsed.fragment:
        raise ValueError("integration URLs cannot contain fragments")


def _safe_relative(value: str) -> None:
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or ".." in posix.parts or ".." in windows.parts:
        raise ValueError("manifest paths must be relative and confined")


def _safe_slug(value: str) -> None:
    if not _SLUG.fullmatch(value) or value.lower() in _WINDOWS_RESERVED:
        raise ValueError(f"unsafe integration identifier {value!r}")


__all__ = [
    "ApiBinding",
    "InstallKind",
    "IntegrationComponent",
    "IntegrationManifest",
    "IntegrationProfile",
    "LicenseSpec",
    "ResourceHint",
    "SourceSpec",
    "WeightSpec",
    "load_integration_manifest",
]
