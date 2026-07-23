#!/usr/bin/env python3
"""Dependency orchestrator that runs before Discovery OS is installed.

The manifest is data only: arbitrary commands are not accepted.  Each model
receives its own environment because their Python, Torch, CUDA, and NumPy
requirements are mutually incompatible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = WORKSPACE / "integrations" / "manifest.v1.json"
DEFAULT_ROOT = WORKSPACE / ".discovery"
TRUTHY = {"1", "true", "yes", "accept", "accepted"}
TRUSTED_DEFAULT_MANIFEST_REVISION = (
    "f1964864652b1020b4905a0787961dda240a6890105e47865e70f6a6799d6141"
)
SAFE_SLUG = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?$")
PACKAGE_PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*==[^\s=<>!~]+$")
PYTHON_MINOR = re.compile(r"^3\.\d{1,2}$")
WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{number}" for number in range(1, 10)),
    *(f"lpt{number}" for number in range(1, 10)),
}
MAX_ARCHIVE_COMPRESSED_BYTES = 2 * 1024**3
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 32 * 1024**3
MAX_ARCHIVE_MEMBERS = 250_000
MAX_ARCHIVE_EXPANSION_RATIO = 200
SIDECAR_RUNTIME_PINS = (
    "pydantic==2.13.4",
    "requests==2.34.2",
    "fastapi==0.139.0",
    "uvicorn==0.51.0",
)


class BootstrapError(RuntimeError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise BootstrapError(f"duplicate JSON key {key!r}")
        payload[key] = value
    return payload


def load_manifest(
    path: Path,
    *,
    allow_custom_manifest: bool = False,
) -> dict[str, Any]:
    resolved_path = path.resolve()
    is_default = resolved_path == DEFAULT_MANIFEST.resolve()
    if not is_default and not allow_custom_manifest:
        raise BootstrapError(
            "custom manifests are disabled; pass --allow-custom-manifest to trust this file"
        )
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"cannot read integration manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise BootstrapError("manifest must be a JSON object")
    _validate_manifest_revision(payload)
    _validate_manifest_shape(payload)
    if is_default and payload["manifest_revision"] != TRUSTED_DEFAULT_MANIFEST_REVISION:
        raise BootstrapError(
            "default integration manifest does not match the trusted revision embedded in bootstrap.py"
        )
    return payload


def resolve_profile(manifest: dict[str, Any], profile_name: str) -> list[dict[str, Any]]:
    profiles = manifest["profiles"]
    if profile_name not in profiles:
        raise BootstrapError(
            f"unknown profile {profile_name!r}; choose one of: {', '.join(sorted(profiles))}"
        )
    components = {row["component_id"]: row for row in manifest["components"]}
    selected: dict[str, dict[str, Any]] = {}
    visiting: set[str] = set()

    def include(component_id: str) -> None:
        if component_id in selected:
            return
        if component_id in visiting:
            raise BootstrapError("component dependency graph contains a cycle")
        try:
            component = components[component_id]
        except KeyError as exc:
            raise BootstrapError(f"unknown component dependency {component_id!r}") from exc
        visiting.add(component_id)
        for dependency in component["dependencies"]:
            include(dependency)
        visiting.remove(component_id)
        selected[component_id] = component

    for component_id in profiles[profile_name]["components"]:
        include(component_id)
    return list(selected.values())


def _component_requires_environment(component: dict[str, Any]) -> bool:
    """Return whether installing a component must produce an isolated runtime.

    A source-only GitHub archive normally needs only its verified source tree.
    When that archive exposes an API, however, the workspace sidecar and the
    archive's pinned dependencies still have to be installed in an environment.
    """

    install = component["install"]
    if install["kind"] in {"local_project", "pypi"}:
        return True
    return install["kind"] == "github_archive" and (
        install.get("install_local", True) or component.get("api") is not None
    )


def build_plan(
    manifest: dict[str, Any],
    profile_name: str,
    *,
    accelerator: str,
    accepted_licenses: set[str],
    include_weights: bool = False,
) -> dict[str, Any]:
    host_platform = _host_platform()
    selected_accelerator = _detect_accelerator() if accelerator == "auto" else accelerator
    components = resolve_profile(manifest, profile_name)
    rows: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    for component in components:
        install = component["install"]
        action = "install"
        reason: str | None = None
        component_accelerator: str | None = selected_accelerator
        fallback_from: str | None = None
        if host_platform not in component["platforms"]:
            action = "unsupported_platform"
            reason = f"requires one of {component['platforms']}"
        elif (
            install["kind"] != "github_archive"
            or install.get("install_local", True)
            or component.get("api") is not None
        ):
            if selected_accelerator not in component["accelerators"]:
                if selected_accelerator in {"cuda", "mps"} and "cpu" in component["accelerators"]:
                    fallback_from = selected_accelerator
                    component_accelerator = "cpu"
                else:
                    action = "unsupported_accelerator"
                    reason = f"requires one of {component['accelerators']}"
        if action == "install" and not _license_accepted(component, accepted_licenses):
            action = "license_required"
            reason = (
                f"explicitly accept {component['component_id']} or set "
                f"{component['license'].get('acceptance_env')}=1"
            )
        elif action == "install" and install["kind"] == "remote_api":
            action = "remote_configuration"
            reason = f"set {component['api']['base_url_env']}"
        elif (
            action == "install"
            and install["kind"] == "github_archive"
            and not install.get("install_local", True)
            and component.get("api") is None
        ):
            action = "download_source"
            component_accelerator = None

        weight_actions = _plan_weights(
            component,
            accepted_licenses=accepted_licenses,
            include_weights=include_weights,
        )
        if action not in {"install", "download_source"}:
            unresolved.append(
                {
                    "component_id": component["component_id"],
                    "item": "component",
                    "status": action,
                }
            )
        if include_weights and action in {"install", "download_source"}:
            for weight_row in weight_actions:
                if weight_row["action"] != "download":
                    unresolved.append(
                        {
                            "component_id": component["component_id"],
                            "item": weight_row["weight_id"],
                            "status": weight_row["action"],
                        }
                    )
        rows.append(
            {
                "component_id": component["component_id"],
                "display_name": component["display_name"],
                "action": action,
                "reason": reason,
                "accelerator": component_accelerator,
                "accelerator_fallback_from": fallback_from,
                "environment": (
                    f"envs/{component['component_id']}"
                    if _component_requires_environment(component)
                    else None
                ),
                "python": install.get("python"),
                "storage_gb": component["resources"]["storage_gb"],
                "weights": [item["weight_id"] for item in component["weights"]],
                "weight_actions": weight_actions,
                "status": component["status"],
            }
        )
    return {
        "schema_version": "1.0",
        "manifest_revision": manifest["manifest_revision"],
        "profile": profile_name,
        "platform": host_platform,
        "accelerator": selected_accelerator,
        "estimated_storage_gb": sum(row["storage_gb"] for row in rows),
        "include_weights": include_weights,
        "status": "ready" if not unresolved else "partial",
        "unresolved": unresolved,
        "components": rows,
    }


def _plan_weights(
    component: dict[str, Any],
    *,
    accepted_licenses: set[str],
    include_weights: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for weight in component["weights"]:
        action = "not_requested"
        if include_weights:
            if weight["kind"] == "managed":
                action = "managed_by_upstream"
            elif weight["kind"] == "manual":
                action = "manual_download_required"
            else:
                acceptance_env = weight.get("acceptance_env")
                accepted = component["component_id"] in accepted_licenses or (
                    acceptance_env is not None
                    and os.getenv(acceptance_env, "").strip().lower() in TRUTHY
                )
                if acceptance_env and not accepted:
                    action = "license_required"
                else:
                    token_env = weight.get("token_env")
                    if weight.get("gated") and (token_env is None or not os.getenv(token_env)):
                        action = "credential_required"
                    else:
                        action = "download"
        rows.append({"weight_id": weight["weight_id"], "action": action})
    return rows


class Installer:
    def __init__(
        self,
        manifest: dict[str, Any],
        root: Path,
        *,
        accelerator: str,
        accepted_licenses: set[str],
        include_weights: bool,
        dry_run: bool,
        allow_external_root: bool = False,
    ) -> None:
        self.manifest = manifest
        self.root = _confined_root(root, allow_external=allow_external_root)
        self.accelerator = accelerator
        self.accepted_licenses = accepted_licenses
        self.include_weights = include_weights
        self.dry_run = dry_run
        self._uv: list[str] | None = None
        self._bootstrap_python: Path | None = None

    def install(self, profile_name: str) -> dict[str, Any]:
        plan = build_plan(
            self.manifest,
            profile_name,
            accelerator=self.accelerator,
            accepted_licenses=self.accepted_licenses,
            include_weights=self.include_weights,
        )
        disk_preflight = self._disk_preflight(plan)
        plan["disk_preflight"] = disk_preflight
        if not disk_preflight["ok"]:
            plan["status"] = "partial"
            plan["unresolved"].append(
                {
                    "component_id": "bootstrap",
                    "item": "disk_space",
                    "status": "insufficient_disk_space",
                }
            )
        if self.dry_run:
            plan["dry_run"] = True
            return plan

        if not disk_preflight["ok"]:
            raise BootstrapError(
                "insufficient disk space: "
                f"need approximately {disk_preflight['required_gb']:.2f} GiB, "
                f"have {disk_preflight['free_gb']:.2f} GiB"
            )

        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.resolve() != self.root:
            raise BootstrapError("bootstrap root changed during initialization")
        for name in (
            "envs",
            "sources",
            "downloads",
            "models",
            "inventories",
            "state",
            "cache",
            "cache/uv",
            "cache/pip",
            "cache/huggingface",
            "cache/boltz",
            "cache/unimol",
            "tmp",
        ):
            directory = self.root / name
            directory.mkdir(parents=True, exist_ok=True)
            _assert_below(directory.resolve(), self.root)

        components = {item["component_id"]: item for item in self.manifest["components"]}
        states: dict[str, Any] = {}
        for row in plan["components"]:
            component = components[row["component_id"]]
            action = row["action"]
            if action not in {"install", "download_source"}:
                states[row["component_id"]] = {
                    "status": action,
                    "reason": row["reason"],
                }
                continue
            try:
                print(f"[bootstrap] {action}: {row['component_id']}", flush=True)
                state = self._install_component(
                    component,
                    effective_accelerator=row.get("accelerator"),
                )
                if self.include_weights:
                    state["weights"] = self._install_weights(component)
                states[row["component_id"]] = state
            except Exception as exc:
                states[row["component_id"]] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }

        unresolved = _installation_unresolved(states, include_weights=self.include_weights)
        overall = "complete" if not unresolved else "partial"
        state = {
            "schema_version": "1.0",
            "manifest_revision": self.manifest["manifest_revision"],
            "profile": profile_name,
            "platform": plan["platform"],
            "accelerator": plan["accelerator"],
            "status": overall,
            "requested_weights": self.include_weights,
            "replayable_lock": False,
            "unresolved": unresolved,
            "components": states,
        }
        state_path = (self.root / "state" / "install-state.json").resolve()
        _assert_below(state_path, self.root)
        _atomic_json(state_path, state)
        return state

    def _disk_preflight(self, plan: dict[str, Any]) -> dict[str, Any]:
        required_gb = 0.0
        components = {item["component_id"]: item for item in self.manifest["components"]}
        for row in plan["components"]:
            if row["action"] not in {"install", "download_source"}:
                continue
            component_id = row["component_id"]
            component = components[component_id]
            environment = (self.root / "envs" / component_id).resolve()
            source_marker = (
                self.root / "sources" / component_id / ".discovery-source.json"
            ).resolve()
            needs_environment = _component_requires_environment(component)
            needs_source = component["install"]["kind"] == "github_archive"
            already_present = (
                (not needs_environment or _environment_python(environment).is_file())
                and (not needs_source or source_marker.is_file())
            )
            if self.include_weights:
                weight_specs = {
                    item["weight_id"]: item for item in components[component_id]["weights"]
                }
                for weight_row in row["weight_actions"]:
                    if weight_row["action"] != "download":
                        continue
                    weight = weight_specs[weight_row["weight_id"]]
                    completion_marker = (
                        self.root
                        / "models"
                        / component_id
                        / weight_row["weight_id"]
                        / weight["revision"]
                        / (
                            ".artifact.json"
                            if weight["kind"] == "https"
                            else ".snapshot.json"
                        )
                    ).resolve()
                    if not completion_marker.is_file():
                        already_present = False
                        break
            if not already_present:
                required_gb += float(row["storage_gb"])
        ancestor = _nearest_existing_ancestor(self.root)
        usage = shutil.disk_usage(ancestor)
        required_bytes = int(required_gb * 1024**3)
        return {
            "path": str(ancestor),
            "required_gb": required_gb,
            "free_gb": usage.free / 1024**3,
            "ok": usage.free >= required_bytes,
            "conservative_estimate": True,
        }

    def _install_component(
        self,
        component: dict[str, Any],
        *,
        effective_accelerator: str | None,
    ) -> dict[str, Any]:
        install = component["install"]
        kind = install["kind"]
        if kind == "github_archive":
            source_path = self._install_archive(component)
            if not install.get("install_local", True):
                # Research repositories such as QHNet are not installable
                # Python packages, but their API sidecar still needs an
                # isolated, pinned runtime.  Install declared dependencies and
                # the workspace sidecar without attempting ``pip install`` on
                # the source tree.
                if component.get("api") is None:
                    return {
                        "status": "source_ready",
                        "source_path": str(source_path),
                        "source_revision": component["source"]["revision"],
                    }
                environment = self._prepare_environment(component)
                self._uv_install(
                    component,
                    environment,
                    effective_accelerator=effective_accelerator,
                )
                return self._environment_state(
                    component,
                    environment,
                    source_path=source_path,
                    effective_accelerator=effective_accelerator,
                )
            environment = self._prepare_environment(component)
            self._uv_install(
                component,
                environment,
                local_path=source_path,
                effective_accelerator=effective_accelerator,
            )
            return self._environment_state(
                component,
                environment,
                source_path=source_path,
                effective_accelerator=effective_accelerator,
            )
        if kind in {"local_project", "pypi"}:
            environment = self._prepare_environment(component)
            self._uv_install(
                component,
                environment,
                effective_accelerator=effective_accelerator,
            )
            return self._environment_state(
                component,
                environment,
                effective_accelerator=effective_accelerator,
            )
        raise BootstrapError(f"unsupported install kind {kind!r}")

    def _prepare_environment(self, component: dict[str, Any]) -> Path:
        uv = self._ensure_uv()
        install = component["install"]
        python_version = install["python"]
        environment = (self.root / "envs" / component["component_id"]).resolve()
        _assert_below(environment, self.root)
        python_path = _environment_python(environment)
        if python_path.exists():
            result = subprocess.run(
                [str(python_path), "--version"],
                check=True,
                capture_output=True,
                text=True,
            )
            reported = (result.stdout or result.stderr).strip()
            if not reported.startswith(f"Python {python_version}."):
                raise BootstrapError(
                    f"existing {component['component_id']} environment uses {reported}; "
                    f"expected Python {python_version}.x"
                )
            return environment
        existing_python = _find_existing_python(python_version)
        python_request = str(existing_python) if existing_python is not None else python_version
        if existing_python is None:
            _run([*uv, "python", "install", python_version], env=self._uv_env())
        _run(
            [*uv, "venv", str(environment), "--python", python_request],
            env=self._uv_env(),
        )
        if not python_path.exists():
            raise BootstrapError(f"uv did not create {python_path}")
        return environment

    def _uv_install(
        self,
        component: dict[str, Any],
        environment: Path,
        *,
        local_path: Path | None = None,
        effective_accelerator: str | None,
    ) -> None:
        uv = self._ensure_uv()
        install = component["install"]
        python_path = _environment_python(environment)
        base = [
            *uv,
            "pip",
            "install",
            "--python",
            str(python_path),
            "--exclude-newer",
            self.manifest["resolution_cutoff"],
        ]
        for url in install.get("extra_index_urls", []):
            if _index_url_applies(url, effective_accelerator):
                base.extend(["--index", url])
        for url in install.get("find_links", []):
            if _index_url_applies(url, effective_accelerator):
                base.extend(["--find-links", url])

        if install["kind"] == "local_project":
            if install.get("constraints"):
                _run([*base, *install["constraints"]], env=self._uv_env())
            project_path = (WORKSPACE / install["local_path"]).resolve()
            _assert_below(project_path, WORKSPACE)
            _run(
                [*base, "--no-deps", "--editable", str(project_path)],
                env=self._uv_env(),
            )
        elif install["kind"] == "pypi":
            package = install["package"]
            if effective_accelerator == "cuda" and install.get("cuda_extra"):
                package = f"{package}[{install['cuda_extra']}]"
            requirement = f"{package}=={install['version']}"
            _run([*base, requirement, *install.get("constraints", [])], env=self._uv_env())
        else:
            if not install.get("install_local", True):
                constraints = install.get("constraints", [])
                if not constraints:
                    raise BootstrapError(
                        "source-only API archive requires pinned runtime constraints"
                    )
                _run([*base, *constraints], env=self._uv_env())
            else:
                if local_path is None:
                    raise BootstrapError("archive install is missing extracted source path")
                _run(
                    [*base, str(local_path), *install.get("constraints", [])],
                    env=self._uv_env(),
                )

        api = component.get("api")
        if api is not None and api.get("protocol") in {
            "expert-feature-v1",
            "generator-v1",
            "tool-v1",
        }:
            sidecar_base = [
                *uv,
                "pip",
                "install",
                "--python",
                str(python_path),
                "--exclude-newer",
                self.manifest["resolution_cutoff"],
            ]
            _run([*sidecar_base, *SIDECAR_RUNTIME_PINS], env=self._uv_env())
            _run(
                [
                    *sidecar_base,
                    "--no-deps",
                    "--editable",
                    str(WORKSPACE),
                ],
                env=self._uv_env(),
            )

        result = subprocess.run(
            [*uv, "pip", "freeze", "--python", str(python_path)],
            check=True,
            capture_output=True,
            text=True,
            env=self._uv_env(),
        )
        inventory_path = (
            self.root / "inventories" / f"{component['component_id']}.freeze.txt"
        ).resolve()
        _assert_below(inventory_path, self.root)
        _atomic_text(inventory_path, result.stdout)

    def _install_archive(self, component: dict[str, Any]) -> Path:
        install = component["install"]
        digest = install["archive_sha256"]
        downloads_root = (self.root / "downloads").resolve()
        _assert_below(downloads_root, self.root)
        archive = (
            downloads_root / f"{component['component_id']}-{digest[:16]}.tar.gz"
        ).resolve()
        _assert_below(archive, downloads_root)
        _download_verified(
            install["archive_url"],
            archive,
            expected_sha256=digest,
            expected_size=install["archive_size_bytes"],
        )
        sources_root = (self.root / "sources").resolve()
        _assert_below(sources_root, self.root)
        destination = (sources_root / component["component_id"]).resolve()
        _assert_below(destination, sources_root)
        marker = destination / ".discovery-source.json"
        if marker.is_file():
            try:
                existing = json.loads(
                    marker.read_text(encoding="utf-8"),
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except (OSError, UnicodeError, json.JSONDecodeError, BootstrapError) as exc:
                raise BootstrapError(f"source marker is unreadable: {marker}") from exc
            if (
                existing.get("schema_version") != "1.0"
                or existing.get("component_id") != component["component_id"]
                or existing.get("revision") != component["source"]["revision"]
                or existing.get("sha256") != digest
            ):
                raise BootstrapError(
                    f"source directory {destination} belongs to a different revision"
                )
            declared_inventory = existing.get("inventory_sha256")
            if not isinstance(declared_inventory, str) or not re.fullmatch(
                r"[0-9a-f]{64}", declared_inventory
            ):
                raise BootstrapError(
                    f"source marker has no byte inventory: {marker}; move the source aside and rerun"
                )
            actual_inventory = _directory_inventory_sha256(
                destination,
                excluded_names=frozenset({".discovery-source.json"}),
                excluded_directories=frozenset(),
            )
            if actual_inventory != declared_inventory:
                raise BootstrapError(
                    f"source directory {destination} does not match its recorded byte inventory"
                )
        elif destination.exists():
            raise BootstrapError(f"unmanaged source directory already exists: {destination}")
        else:
            _extract_archive(
                archive,
                destination,
                sources_root,
                marker_payload={
                    "schema_version": "1.0",
                    "component_id": component["component_id"],
                    "revision": component["source"]["revision"],
                    "sha256": digest,
                },
            )
        subdirectory = install.get("archive_subdirectory")
        source_path = (destination / subdirectory).resolve() if subdirectory else destination.resolve()
        _assert_below(source_path, destination.resolve())
        if not source_path.is_dir():
            raise BootstrapError(f"archive subdirectory does not exist: {source_path}")
        return source_path

    def _install_weights(self, component: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for weight in component["weights"]:
            weight_id = weight["weight_id"]
            if weight["kind"] == "https":
                parsed = urlsplit(weight["download_url"])
                filename = PurePosixPath(parsed.path).name
                if re.fullmatch(
                    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,254}[A-Za-z0-9])?",
                    filename,
                ) is None:
                    raise BootstrapError(
                        f"HTTPS weight {component['component_id']}/{weight_id} "
                        "has an unsafe artifact filename"
                    )
                models_root = (self.root / "models").resolve()
                _assert_below(models_root, self.root)
                destination_root = (
                    models_root
                    / component["component_id"]
                    / weight_id
                    / weight["revision"]
                ).resolve()
                _assert_below(destination_root, models_root)
                destination = (destination_root / filename).resolve()
                _assert_below(destination, destination_root)
                marker = destination_root / ".artifact.json"
                if destination.is_symlink() or marker.is_symlink():
                    raise BootstrapError(
                        f"HTTPS weight path contains a symlink for "
                        f"{component['component_id']}/{weight_id}"
                    )
                if marker.is_file():
                    try:
                        existing = json.loads(
                            marker.read_text(encoding="utf-8"),
                            object_pairs_hook=_reject_duplicate_json_keys,
                        )
                    except (
                        OSError,
                        UnicodeError,
                        json.JSONDecodeError,
                        BootstrapError,
                    ) as exc:
                        raise BootstrapError(
                            f"existing artifact marker is unreadable for "
                            f"{component['component_id']}/{weight_id}"
                        ) from exc
                    expected_marker = {
                        "schema_version": "1.0",
                        "download_url": weight["download_url"],
                        "revision": weight["revision"],
                        "filename": filename,
                        "sha256": weight["sha256"],
                        "size_bytes": weight["expected_size_bytes"],
                    }
                    if existing != expected_marker:
                        raise BootstrapError(
                            f"existing artifact marker does not match the trusted "
                            f"manifest for {component['component_id']}/{weight_id}"
                        )
                _download_verified(
                    weight["download_url"],
                    destination,
                    expected_sha256=weight["sha256"],
                    expected_size=weight["expected_size_bytes"],
                )
                _atomic_json(
                    marker,
                    {
                        "schema_version": "1.0",
                        "download_url": weight["download_url"],
                        "revision": weight["revision"],
                        "filename": filename,
                        "sha256": weight["sha256"],
                        "size_bytes": weight["expected_size_bytes"],
                    },
                )
                results[weight_id] = {
                    "status": "downloaded",
                    "path": str(destination),
                    "revision": weight["revision"],
                    "sha256": weight["sha256"],
                }
                continue
            if weight["kind"] == "managed":
                results[weight_id] = {
                    "status": "managed_by_upstream",
                    "notes": weight.get("notes", []),
                }
                continue
            if weight["kind"] == "manual":
                results[weight_id] = {
                    "status": "manual_download_required",
                    "download_url": weight["download_url"],
                    "notes": weight.get("notes", []),
                }
                continue
            acceptance_env = weight.get("acceptance_env")
            accepted = component["component_id"] in self.accepted_licenses or (
                acceptance_env is not None
                and os.getenv(acceptance_env, "").strip().lower() in TRUTHY
            )
            if acceptance_env and not accepted:
                results[weight_id] = {
                    "status": "license_required",
                    "acceptance_env": acceptance_env,
                }
                continue
            token_env = weight.get("token_env")
            if weight.get("gated") and (token_env is None or not os.getenv(token_env)):
                results[weight_id] = {
                    "status": "credential_required",
                    "token_env": token_env,
                }
                continue
            models_root = (self.root / "models").resolve()
            _assert_below(models_root, self.root)
            destination = (
                models_root
                / component["component_id"]
                / weight_id
                / weight["revision"]
            ).resolve()
            _assert_below(destination, models_root)
            marker = destination / ".snapshot.json"
            if marker.is_file():
                try:
                    existing = json.loads(
                        marker.read_text(encoding="utf-8"),
                        object_pairs_hook=_reject_duplicate_json_keys,
                    )
                except (OSError, UnicodeError, json.JSONDecodeError, BootstrapError) as exc:
                    raise BootstrapError(
                        f"existing snapshot marker is unreadable for "
                        f"{component['component_id']}/{weight_id}"
                    ) from exc
                if (
                    existing.get("schema_version") == "1.0"
                    and existing.get("repository") == weight["repository"]
                    and existing.get("revision") == weight["revision"]
                ):
                    declared_inventory = existing.get("inventory_sha256")
                    if not isinstance(declared_inventory, str) or not re.fullmatch(
                        r"[0-9a-f]{64}", declared_inventory
                    ):
                        raise BootstrapError(
                            f"existing snapshot marker has no byte inventory for "
                            f"{component['component_id']}/{weight_id}; move the snapshot aside "
                            "and rerun bootstrap"
                        )
                    actual_inventory = _directory_inventory_sha256(destination)
                    if actual_inventory != declared_inventory:
                        raise BootstrapError(
                            f"existing snapshot files do not match the recorded inventory for "
                            f"{component['component_id']}/{weight_id}; move the snapshot aside "
                            "and rerun bootstrap"
                        )
                    results[weight_id] = {
                        "status": "downloaded",
                        "path": str(destination),
                        "revision": weight["revision"],
                    }
                    continue
                raise BootstrapError(
                    f"existing snapshot marker does not match the trusted repository/revision for "
                    f"{component['component_id']}/{weight_id}; move the directory aside and rerun"
                )
            bootstrap_python = self._ensure_huggingface_hub()
            script = (
                "import os,sys; from huggingface_hub import snapshot_download; "
                "snapshot_download(repo_id=sys.argv[1], revision=sys.argv[2], "
                "local_dir=sys.argv[3], token=os.getenv(sys.argv[4]) if sys.argv[4] else None)"
            )
            _run(
                [
                    str(bootstrap_python),
                    "-c",
                    script,
                    weight["repository"],
                    weight["revision"],
                    str(destination),
                    token_env or "",
                ],
                env=self._weight_download_env(token_env),
            )
            expected_size = weight.get("expected_size_bytes")
            if expected_size is not None:
                actual_size = _directory_size(destination, exclude={marker.name})
                if actual_size != expected_size:
                    raise BootstrapError(
                        f"weight snapshot size mismatch for {component['component_id']}/{weight_id}: "
                        f"expected {expected_size}, got {actual_size}"
                    )
            _atomic_json(
                marker,
                {
                    "schema_version": "1.0",
                    "repository": weight["repository"],
                    "revision": weight["revision"],
                    "inventory_sha256": _directory_inventory_sha256(destination),
                },
            )
            results[weight_id] = {
                "status": "downloaded",
                "path": str(destination),
                "revision": weight["revision"],
            }
        return results

    def _environment_state(
        self,
        component: dict[str, Any],
        environment: Path,
        *,
        source_path: Path | None = None,
        effective_accelerator: str | None,
    ) -> dict[str, Any]:
        return {
            "status": "installed",
            "environment": str(environment),
            "python": component["install"]["python"],
            "package": component["install"].get("package"),
            "version": component["install"].get("version"),
            "accelerator": effective_accelerator,
            "source_revision": (
                component["source"]["revision"] if component.get("source") else None
            ),
            "source_path": str(source_path) if source_path else None,
            "environment_inventory": str(
                self.root / "inventories" / f"{component['component_id']}.freeze.txt"
            ),
            "replayable_lock": False,
        }

    def _ensure_uv(self) -> list[str]:
        if self._uv is not None:
            return self._uv
        bootstrap = (self.root / "bootstrap").resolve()
        _assert_below(bootstrap, self.root)
        python_path = _environment_python(bootstrap)
        if not python_path.exists():
            _run(
                [sys.executable, "-m", "venv", str(bootstrap)],
                env=self._uv_env(),
            )
        uv_version = self.manifest["uv_version"]
        result = subprocess.run(
            [str(python_path), "-m", "pip", "show", "uv"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or f"Version: {uv_version}" not in result.stdout:
            _run(
                [
                    str(python_path),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    f"uv=={uv_version}",
                ],
                env=self._uv_env(),
            )
        self._bootstrap_python = python_path
        self._uv = [str(python_path), "-m", "uv"]
        return self._uv

    def _ensure_huggingface_hub(self) -> Path:
        self._ensure_uv()
        assert self._bootstrap_python is not None
        version = self.manifest["huggingface_hub_version"]
        result = subprocess.run(
            [str(self._bootstrap_python), "-m", "pip", "show", "huggingface-hub"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or f"Version: {version}" not in result.stdout:
            _run(
                [
                    str(self._bootstrap_python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    f"huggingface-hub=={version}",
                ],
                env=self._uv_env(),
            )
        return self._bootstrap_python

    def _uv_env(self) -> dict[str, str]:
        result = {
            name: value
            for name, value in os.environ.items()
            if not _sensitive_environment_name(name)
        }
        result["UV_CACHE_DIR"] = str(self.root / "cache" / "uv")
        result["PIP_CACHE_DIR"] = str(self.root / "cache" / "pip")
        result["HF_HOME"] = str(self.root / "cache" / "huggingface")
        result["BOLTZ_CACHE"] = str(self.root / "cache" / "boltz")
        result["UNIMOL_WEIGHT_DIR"] = str(self.root / "cache" / "unimol")
        temporary = str(self.root / "tmp")
        result["TMPDIR"] = temporary
        result["TMP"] = temporary
        result["TEMP"] = temporary
        return result

    def _weight_download_env(self, token_env: str | None) -> dict[str, str]:
        result = self._uv_env()
        if token_env and token_env in os.environ:
            result[token_env] = os.environ[token_env]
        return result


def doctor(
    manifest: dict[str, Any],
    root: Path,
    profile_name: str,
    *,
    allow_external_root: bool = False,
) -> dict[str, Any]:
    root = _confined_root(root, allow_external=allow_external_root)
    state_path = root / "state" / "install-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    state_revision = state.get("manifest_revision")
    state_components = state.get("components", {})
    results: list[dict[str, Any]] = []
    for component in resolve_profile(manifest, profile_name):
        component_id = component["component_id"]
        recorded = state_components.get(component_id, {})
        status = recorded.get("status", "not_installed")
        healthy = False
        detail = status
        if status == "installed":
            environment = root / "envs" / component_id
            python_path = _environment_python(environment)
            import_name = component["install"].get("import_name")
            if python_path.is_file() and import_name:
                result = subprocess.run(
                    [
                        str(python_path),
                        "-c",
                        f"import importlib; importlib.import_module({import_name!r})",
                    ],
                    capture_output=True,
                    text=True,
                )
                healthy = result.returncode == 0
                detail = "import_ok" if healthy else (result.stderr.strip()[-2_000:] or "import_failed")
        elif status == "source_ready":
            source_path = Path(recorded.get("source_path", ""))
            healthy = source_path.is_dir()
            detail = "source_present" if healthy else "source_missing"
        results.append(
            {
                "component_id": component_id,
                "recorded_status": status,
                "healthy": healthy,
                "detail": detail,
            }
        )
    return {
        "schema_version": "1.0",
        "manifest_revision": manifest["manifest_revision"],
        "state_manifest_revision": state_revision,
        "profile": profile_name,
        "healthy": state_revision == manifest["manifest_revision"]
        and all(item["healthy"] for item in results),
        "components": results,
    }


def _validate_manifest_revision(payload: dict[str, Any]) -> None:
    revision = payload.get("manifest_revision")
    if not isinstance(revision, str) or len(revision) != 64:
        raise BootstrapError("manifest_revision must be a SHA-256 digest")
    material = dict(payload)
    material.pop("manifest_revision", None)
    actual = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if actual != revision:
        raise BootstrapError("integration manifest revision does not match its content")


def _validate_manifest_shape(payload: dict[str, Any]) -> None:
    top_allowed = {
        "schema_version",
        "manifest_revision",
        "generated_at",
        "resolution_cutoff",
        "uv_version",
        "huggingface_hub_version",
        "profiles",
        "components",
    }
    _reject_unknown(payload, top_allowed, "manifest")
    if payload.get("schema_version") != "1.0":
        raise BootstrapError("unsupported manifest schema_version")
    if not isinstance(payload.get("profiles"), dict) or not isinstance(payload.get("components"), list):
        raise BootstrapError("manifest profiles/components have the wrong type")
    if not payload["profiles"] or not payload["components"]:
        raise BootstrapError("manifest profiles/components cannot be empty")
    _validate_manifest_timestamps(payload)
    for field in ("uv_version", "huggingface_hub_version"):
        value = payload.get(field)
        if not isinstance(value, str) or not value or value.startswith("-") or any(ch.isspace() for ch in value):
            raise BootstrapError(f"{field} must be a pinned version token")

    profile_allowed = {"schema_version", "description", "components"}
    for name, profile in payload["profiles"].items():
        if not isinstance(name, str) or not isinstance(profile, dict):
            raise BootstrapError("profile names and definitions must be JSON objects")
        _validate_safe_slug(name, "profile name")
        _reject_unknown(profile, profile_allowed, f"profile {name}")
        if profile.get("schema_version") != "1.0":
            raise BootstrapError(f"profile {name!r} has an unsupported schema_version")
        profile_components = profile.get("components")
        if not isinstance(profile_components, list) or not profile_components:
            raise BootstrapError(f"profile {name!r} components must be a non-empty list")
        for component_id in profile_components:
            _validate_safe_slug(component_id, f"profile {name} component")
        if len(profile_components) != len(set(profile_components)):
            raise BootstrapError(f"profile {name!r} contains duplicate components")

    component_allowed = {
        "schema_version",
        "component_id",
        "display_name",
        "role",
        "modalities",
        "platforms",
        "accelerators",
        "dependencies",
        "install",
        "source",
        "license",
        "weights",
        "api",
        "resources",
        "status",
        "notes",
    }
    install_allowed = {
        "schema_version",
        "kind",
        "python",
        "package",
        "version",
        "import_name",
        "local_path",
        "constraints",
        "cuda_extra",
        "extra_index_urls",
        "find_links",
        "archive_url",
        "archive_sha256",
        "archive_size_bytes",
        "archive_subdirectory",
        "install_local",
    }
    source_allowed = {"schema_version", "repository", "revision", "release"}
    license_allowed = {
        "schema_version",
        "code_license",
        "code_license_url",
        "model_license",
        "model_license_url",
        "requires_acceptance",
        "acceptance_env",
    }
    weight_allowed = {
        "schema_version",
        "weight_id",
        "kind",
        "repository",
        "revision",
        "download_url",
        "sha256",
        "expected_size_bytes",
        "gated",
        "token_env",
        "acceptance_env",
        "notes",
    }
    api_allowed = {"schema_version", "protocol", "base_url_env", "default_port"}
    resources_allowed = {"schema_version", "storage_gb", "memory_gb", "vram_gb"}
    ids: set[str] = set()
    components_by_id: dict[str, dict[str, Any]] = {}
    for index, component in enumerate(payload["components"]):
        if not isinstance(component, dict):
            raise BootstrapError(f"component {index} must be an object")
        _reject_unknown(component, component_allowed, f"component {index}")
        component_id = component.get("component_id")
        if not isinstance(component_id, str) or not component_id:
            raise BootstrapError("component_id must be a non-empty string")
        _validate_safe_slug(component_id, "component_id")
        if component.get("schema_version") != "1.0":
            raise BootstrapError(f"component {component_id!r} has an unsupported schema_version")
        if component_id in ids:
            raise BootstrapError(f"duplicate component_id {component_id!r}")
        ids.add(component_id)
        components_by_id[component_id] = component
        install = component.get("install")
        if not isinstance(install, dict):
            raise BootstrapError(f"component {component_id!r} needs an install object")
        _reject_unknown(install, install_allowed, f"install {component_id}")
        if install.get("schema_version") != "1.0":
            raise BootstrapError(f"install {component_id!r} has an unsupported schema_version")
        if install.get("kind") not in {"local_project", "pypi", "github_archive", "remote_api"}:
            raise BootstrapError(f"component {component_id!r} has unsupported install kind")
        _validate_install_spec(component_id, install)
        dependencies = component.get("dependencies")
        if not isinstance(dependencies, list):
            raise BootstrapError(f"component {component_id!r} dependencies must be a list")
        for dependency in dependencies:
            _validate_safe_slug(dependency, f"component {component_id} dependency")
        if len(dependencies) != len(set(dependencies)):
            raise BootstrapError(f"component {component_id!r} has duplicate dependencies")
        platforms = component.get("platforms")
        accelerators = component.get("accelerators")
        if (
            not isinstance(platforms, list)
            or not platforms
            or any(not isinstance(item, str) for item in platforms)
            or len(platforms) != len(set(platforms))
            or not set(platforms)
            <= {
            "windows",
            "linux",
            "darwin",
            }
        ):
            raise BootstrapError(f"component {component_id!r} has invalid platforms")
        if (
            not isinstance(accelerators, list)
            or not accelerators
            or any(not isinstance(item, str) for item in accelerators)
            or len(accelerators) != len(set(accelerators))
            or not set(accelerators)
            <= {
            "cpu",
            "cuda",
            "mps",
            "remote",
            }
        ):
            raise BootstrapError(f"component {component_id!r} has invalid accelerators")
        if component.get("source") is not None:
            if not isinstance(component["source"], dict):
                raise BootstrapError(f"source {component_id!r} must be an object")
            _reject_unknown(component["source"], source_allowed, f"source {component_id}")
            _validate_https_url(
                component["source"].get("repository"),
                f"source {component_id} repository",
            )
            revision = component["source"].get("revision", "")
            if (
                not isinstance(revision, str)
                or len(revision) != 40
                or any(ch not in "0123456789abcdef" for ch in revision)
            ):
                raise BootstrapError(f"component {component_id!r} source revision is not pinned")
        license_spec = component.get("license")
        if not isinstance(license_spec, dict):
            raise BootstrapError(f"license {component_id!r} must be an object")
        _reject_unknown(license_spec, license_allowed, f"license {component_id}")
        if license_spec.get("schema_version") != "1.0" or not isinstance(
            license_spec.get("code_license"), str
        ):
            raise BootstrapError(f"license {component_id!r} is incomplete")
        for field in ("code_license_url", "model_license_url"):
            if license_spec.get(field) is not None:
                _validate_https_url(license_spec[field], f"license {component_id} {field}")
        acceptance_env = license_spec.get("acceptance_env")
        if acceptance_env is not None and (
            not isinstance(acceptance_env, str)
            or re.fullmatch(r"[A-Z][A-Z0-9_]+", acceptance_env) is None
        ):
            raise BootstrapError(f"license {component_id!r} has an invalid acceptance_env")
        if license_spec.get("requires_acceptance") and acceptance_env is None:
            raise BootstrapError(f"license {component_id!r} needs an acceptance_env")
        weights = component.get("weights")
        if not isinstance(weights, list):
            raise BootstrapError(f"weights {component_id!r} must be a list")
        weight_ids: set[str] = set()
        for weight in weights:
            if not isinstance(weight, dict):
                raise BootstrapError(f"weight {component_id!r} must be an object")
            _reject_unknown(weight, weight_allowed, f"weight {component_id}")
            _validate_weight_spec(component_id, weight)
            weight_id = weight["weight_id"]
            if weight_id in weight_ids:
                raise BootstrapError(f"component {component_id!r} has duplicate weight ids")
            weight_ids.add(weight_id)
        if component.get("api") is not None:
            if not isinstance(component["api"], dict):
                raise BootstrapError(f"api {component_id!r} must be an object")
            _reject_unknown(component["api"], api_allowed, f"api {component_id}")
        resources = component.get("resources")
        if not isinstance(resources, dict):
            raise BootstrapError(f"resources {component_id!r} must be an object")
        _reject_unknown(resources, resources_allowed, f"resources {component_id}")
        for field in ("storage_gb", "memory_gb", "vram_gb"):
            value = resources.get(field)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0
            ):
                raise BootstrapError(f"resource {component_id!r}.{field} must be non-negative")
        if install.get("kind") == "github_archive":
            digest = install.get("archive_sha256", "")
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise BootstrapError(f"component {component_id!r} archive SHA-256 is missing")
        for key, value in install.items():
            if "command" in key.lower() or "script" in key.lower():
                raise BootstrapError("manifest cannot contain executable commands or scripts")

    for component in payload["components"]:
        unknown = set(component.get("dependencies", [])) - ids
        if unknown:
            raise BootstrapError(
                f"component {component['component_id']!r} has unknown dependencies {sorted(unknown)}"
            )
    for name, profile in payload["profiles"].items():
        unknown = set(profile.get("components", [])) - ids
        if unknown:
            raise BootstrapError(f"profile {name!r} has unknown components {sorted(unknown)}")
        resolve_profile(payload, name)


def _validate_manifest_timestamps(payload: dict[str, Any]) -> None:
    generated = _parse_aware_timestamp(payload.get("generated_at"), "generated_at")
    cutoff = _parse_aware_timestamp(payload.get("resolution_cutoff"), "resolution_cutoff")
    now = datetime.now(timezone.utc)
    if cutoff > generated:
        raise BootstrapError("resolution_cutoff must not be later than generated_at")
    if generated > now:
        raise BootstrapError("generated_at must not be in the future")


def _parse_aware_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise BootstrapError(f"{label} must be an RFC 3339 timestamp with a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BootstrapError(f"{label} must be an RFC 3339 timestamp with a timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise BootstrapError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _validate_safe_slug(value: Any, label: str) -> None:
    if not isinstance(value, str) or SAFE_SLUG.fullmatch(value) is None:
        raise BootstrapError(
            f"{label} must be a lowercase filesystem-safe slug of at most 64 characters"
        )
    windows_stem = value.split(".", 1)[0].lower()
    if windows_stem in WINDOWS_RESERVED_NAMES:
        raise BootstrapError(f"{label} uses a Windows reserved name: {value!r}")


def _validate_install_spec(component_id: str, install: dict[str, Any]) -> None:
    kind = install["kind"]
    python_version = install.get("python")
    if python_version is not None and (
        not isinstance(python_version, str) or PYTHON_MINOR.fullmatch(python_version) is None
    ):
        raise BootstrapError(f"install {component_id!r} must pin Python major.minor")
    constraints = install.get("constraints", [])
    if not isinstance(constraints, list) or any(
        not isinstance(item, str) or PACKAGE_PIN.fullmatch(item) is None
        for item in constraints
    ):
        raise BootstrapError(f"install {component_id!r} constraints must be package==version pins")
    for key in ("extra_index_urls", "find_links"):
        urls = install.get(key, [])
        if not isinstance(urls, list):
            raise BootstrapError(f"install {component_id!r} {key} must be a list")
        for url in urls:
            _validate_https_url(url, f"install {component_id} {key}")

    if kind == "local_project":
        local_path = install.get("local_path")
        if python_version is None or not isinstance(local_path, str):
            raise BootstrapError(f"local project {component_id!r} needs Python and local_path")
        _validate_relative_manifest_path(local_path, f"local project {component_id}")
    elif kind == "pypi":
        package = install.get("package")
        version = install.get("version")
        if python_version is None or not isinstance(package, str) or not isinstance(version, str):
            raise BootstrapError(f"PyPI component {component_id!r} needs Python/package/version")
        if re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", package) is None:
            raise BootstrapError(f"PyPI component {component_id!r} has an unsafe package name")
        if not version or version.startswith("-") or any(ch.isspace() for ch in version) or any(
            mark in version for mark in "<>=~*"
        ):
            raise BootstrapError(f"PyPI component {component_id!r} must use an exact version")
    elif kind == "github_archive":
        archive_url = install.get("archive_url")
        digest = install.get("archive_sha256")
        size = install.get("archive_size_bytes")
        if python_version is None:
            raise BootstrapError(f"archive component {component_id!r} needs Python")
        _validate_https_url(archive_url, f"archive {component_id}")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise BootstrapError(f"archive component {component_id!r} needs a SHA-256")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 < size <= MAX_ARCHIVE_COMPRESSED_BYTES:
            raise BootstrapError(
                f"archive component {component_id!r} size must be within the compressed archive limit"
            )
        subdirectory = install.get("archive_subdirectory")
        if subdirectory is not None:
            if not isinstance(subdirectory, str):
                raise BootstrapError(f"archive component {component_id!r} subdirectory is invalid")
            _validate_relative_manifest_path(subdirectory, f"archive {component_id} subdirectory")
    elif kind == "remote_api":
        forbidden = (
            "python",
            "package",
            "version",
            "import_name",
            "local_path",
            "archive_url",
            "archive_sha256",
        )
        if any(install.get(key) is not None for key in forbidden):
            raise BootstrapError(f"remote API {component_id!r} cannot contain install instructions")


def _validate_weight_spec(component_id: str, weight: dict[str, Any]) -> None:
    if weight.get("schema_version") != "1.0":
        raise BootstrapError(f"weight {component_id!r} has an unsupported schema_version")
    _validate_safe_slug(weight.get("weight_id"), f"weight {component_id} id")
    kind = weight.get("kind")
    if kind not in {"huggingface", "https", "managed", "manual"}:
        raise BootstrapError(f"weight {component_id!r} has an unsupported kind")
    if kind == "huggingface":
        repository = weight.get("repository")
        revision = weight.get("revision")
        if (
            not isinstance(repository, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None
            or not isinstance(revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", revision) is None
        ):
            raise BootstrapError(f"Hugging Face weight {component_id!r} is not commit-pinned")
    if kind == "manual":
        _validate_https_url(weight.get("download_url"), f"manual weight {component_id}")
    if kind == "https":
        _validate_https_url(weight.get("download_url"), f"HTTPS weight {component_id}")
        revision = weight.get("revision")
        digest = weight.get("sha256")
        size = weight.get("expected_size_bytes")
        if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise BootstrapError(
                f"HTTPS weight {component_id!r} must bind an immutable 40-character revision"
            )
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise BootstrapError(
                f"HTTPS weight {component_id!r} must publish an exact SHA-256"
            )
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or not 0 < size <= MAX_ARCHIVE_COMPRESSED_BYTES
        ):
            raise BootstrapError(
                f"HTTPS weight {component_id!r} must publish a bounded positive size"
            )
    expected_size = weight.get("expected_size_bytes")
    if expected_size is not None and (
        not isinstance(expected_size, int) or isinstance(expected_size, bool) or expected_size < 0
    ):
        raise BootstrapError(f"weight {component_id!r} expected_size_bytes is invalid")
    for key in ("token_env", "acceptance_env"):
        value = weight.get(key)
        if value is not None and (
            not isinstance(value, str) or re.fullmatch(r"[A-Z][A-Z0-9_]+", value) is None
        ):
            raise BootstrapError(f"weight {component_id!r} has an invalid {key}")
    if weight.get("gated") and not (weight.get("token_env") and weight.get("acceptance_env")):
        raise BootstrapError(f"gated weight {component_id!r} needs token and acceptance variables")


def _validate_https_url(value: Any, label: str) -> None:
    if not isinstance(value, str):
        raise BootstrapError(f"{label} must be an HTTPS URL")
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
        raise BootstrapError(f"{label} must be a credential-free HTTPS URL without fragments")


def _validate_relative_manifest_path(value: str, label: str) -> None:
    if value == ".":
        return
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        not value
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or ".." in windows.parts
    ):
        raise BootstrapError(f"{label} must be a confined relative path")


def _reject_unknown(value: dict[str, Any], allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} must be an object")
    unknown = set(value) - allowed
    if unknown:
        raise BootstrapError(f"{label} contains unknown fields: {sorted(unknown)}")


def _license_accepted(component: dict[str, Any], accepted: set[str]) -> bool:
    license_spec = component["license"]
    if not license_spec.get("requires_acceptance", False):
        return True
    if component["component_id"] in accepted:
        return True
    env_name = license_spec.get("acceptance_env")
    return bool(env_name and os.getenv(env_name, "").strip().lower() in TRUTHY)


def _installation_unresolved(
    states: dict[str, Any],
    *,
    include_weights: bool,
) -> list[dict[str, str]]:
    unresolved: list[dict[str, str]] = []
    for component_id, state in states.items():
        component_status = state.get("status", "unknown")
        if component_status not in {"installed", "source_ready"}:
            unresolved.append(
                {
                    "component_id": component_id,
                    "item": "component",
                    "status": str(component_status),
                }
            )
            continue
        if not include_weights:
            continue
        weights = state.get("weights", {})
        for weight_id, weight_state in weights.items():
            weight_status = weight_state.get("status", "unknown")
            if weight_status != "downloaded":
                unresolved.append(
                    {
                        "component_id": component_id,
                        "item": weight_id,
                        "status": str(weight_status),
                    }
                )
    return unresolved


def _index_url_applies(url: str, accelerator: str | None) -> bool:
    lowered = url.lower()
    is_cuda_specific = re.search(r"/cu\d+", lowered) is not None
    return accelerator == "cuda" or not is_cuda_specific


def _find_existing_python(python_version: str) -> Path | None:
    candidates: list[list[str]] = []
    candidates.append([sys.executable])
    for name in (f"python{python_version}", "python3", "python"):
        executable = shutil.which(name)
        if executable:
            candidates.append([executable])
    launcher = shutil.which("py")
    if launcher:
        candidates.append([launcher, f"-{python_version}"])

    seen: set[tuple[str, ...]] = set()
    for command in candidates:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        try:
            result = subprocess.run(
                [*command, "-c", "import sys; print(sys.executable); print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if result.returncode == 0 and len(lines) >= 2 and lines[-1] == python_version:
            path = Path(lines[-2]).resolve()
            if path.is_file():
                return path
    return None


def _sensitive_environment_name(name: str) -> bool:
    normalized = name.upper()
    markers = (
        "TOKEN",
        "SECRET",
        "PASSWORD",
        "PASSWD",
        "CREDENTIAL",
        "PRIVATE_KEY",
        "API_KEY",
        "ACCESS_KEY",
    )
    return any(marker in normalized for marker in markers) or normalized in {
        "SSH_AUTH_SOCK",
        "GPG_AGENT_INFO",
    }


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise BootstrapError(f"cannot find an existing ancestor for {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def _directory_size(path: Path, *, exclude: set[str] | None = None) -> int:
    excluded = exclude or set()
    total = 0
    for item in path.rglob("*"):
        if item.name in excluded or not item.is_file():
            continue
        total += item.stat().st_size
    return total


def _directory_inventory_sha256(
    path: Path,
    *,
    excluded_names: frozenset[str] = frozenset({".snapshot.json"}),
    excluded_directories: frozenset[str] = frozenset({".cache"}),
) -> str:
    """Hash every model byte and relative path, excluding download metadata."""

    selected = path.expanduser()
    if selected.is_symlink():
        raise BootstrapError(f"snapshot directory must not be a symlink: {selected}")
    root = selected.resolve(strict=True)
    if not root.is_dir():
        raise BootstrapError(f"snapshot path must be a directory: {root}")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if any(part in excluded_directories for part in relative.parts[:-1]):
            continue
        if candidate.is_symlink():
            raise BootstrapError(f"snapshot inventory contains a symlink: {candidate}")
        if candidate.is_file() and candidate.name not in excluded_names:
            files.append(candidate)
    if not files:
        raise BootstrapError(f"snapshot contains no model files: {root}")
    digest = hashlib.sha256()
    for candidate in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = candidate.relative_to(root).as_posix().encode("utf-8")
        size = candidate.stat().st_size
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(size.to_bytes(8, "big"))
        with candidate.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _host_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "darwin"
    return "linux"


def _detect_accelerator() -> str:
    cuda_visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip().lower() in {"", "none", "void", "-1"}:
        return "mps" if platform.system() == "Darwin" and platform.machine().lower() in {
            "arm64",
            "aarch64",
        } else "cpu"
    visible = os.getenv("NVIDIA_VISIBLE_DEVICES", "").strip().lower()
    if visible and visible not in {"none", "void", "-1"}:
        return "cuda"
    executable = shutil.which("nvidia-smi")
    if executable:
        result = subprocess.run(
            [executable, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return "cuda"
    return (
        "mps"
        if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}
        else "cpu"
    )


def _environment_python(environment: Path) -> Path:
    return (
        environment / "Scripts" / "python.exe"
        if os.name == "nt"
        else environment / "bin" / "python"
    )


def _download_verified(
    url: str,
    destination: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> None:
    if not 0 < expected_size <= MAX_ARCHIVE_COMPRESSED_BYTES:
        raise BootstrapError("download exceeds the configured compressed archive limit")
    if destination.is_file():
        if _file_sha256(destination) == expected_sha256 and destination.stat().st_size == expected_size:
            return
        raise BootstrapError(f"existing download failed checksum validation: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if partial.is_symlink() or (partial.exists() and not partial.is_file()):
        raise BootstrapError(f"unsafe partial download path: {partial}")
    request = urllib.request.Request(url, headers={"User-Agent": "discovery-os-bootstrap/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as handle:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    advertised_size = int(content_length)
                except ValueError as exc:
                    raise BootstrapError("download returned an invalid Content-Length") from exc
                if advertised_size > expected_size:
                    raise BootstrapError(
                        f"download advertises {advertised_size} bytes; expected at most {expected_size}"
                    )
            received = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > expected_size or received > MAX_ARCHIVE_COMPRESSED_BYTES:
                    raise BootstrapError("download exceeded its declared byte limit")
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception as exc:
        partial.unlink(missing_ok=True)
        raise BootstrapError(f"download failed for {url}: {exc}") from exc
    if partial.stat().st_size != expected_size or _file_sha256(partial) != expected_sha256:
        partial.unlink(missing_ok=True)
        raise BootstrapError(f"download checksum/size mismatch for {url}")
    os.replace(partial, destination)


def _extract_archive(
    archive: Path,
    destination: Path,
    sources_root: Path,
    *,
    marker_payload: dict[str, Any] | None = None,
) -> None:
    _assert_below(destination.resolve(), sources_root.resolve())
    compressed_size = archive.stat().st_size
    if not 0 < compressed_size <= MAX_ARCHIVE_COMPRESSED_BYTES:
        raise BootstrapError("archive exceeds the configured compressed size limit")
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=sources_root.resolve())
    ).resolve()
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            members: list[tarfile.TarInfo] = []
            for member in handle:
                if len(members) >= MAX_ARCHIVE_MEMBERS:
                    raise BootstrapError(
                        f"archive exceeds the {MAX_ARCHIVE_MEMBERS} members limit"
                    )
                members.append(member)
            uncompressed_size = 0
            for member in members:
                if member.size < 0:
                    raise BootstrapError(f"archive member has a negative size: {member.name}")
                if member.isfile():
                    uncompressed_size += member.size
            expanded_limit = min(
                MAX_ARCHIVE_UNCOMPRESSED_BYTES,
                compressed_size * MAX_ARCHIVE_EXPANSION_RATIO,
            )
            if uncompressed_size > expanded_limit:
                raise BootstrapError(
                    f"archive expands to {uncompressed_size} bytes; limit is {expanded_limit}"
                )
            roots = {
                PurePosixPath(member.name).parts[0]
                for member in members
                if PurePosixPath(member.name).parts
            }
            strip_root = next(iter(roots)) if len(roots) == 1 else None
            seen_targets: set[str] = set()
            for member in members:
                path = PurePosixPath(member.name)
                parts = list(path.parts)
                if strip_root and parts and parts[0] == strip_root:
                    parts = parts[1:]
                if not parts:
                    continue
                if path.is_absolute() or ".." in parts or member.issym() or member.islnk():
                    raise BootstrapError(f"unsafe archive member: {member.name}")
                if not (member.isdir() or member.isfile()):
                    raise BootstrapError(f"unsupported archive member type: {member.name}")
                if marker_payload is not None and parts == [".discovery-source.json"]:
                    raise BootstrapError("archive attempts to provide the trusted source marker")
                target = temporary.joinpath(*parts).resolve()
                _assert_below(target, temporary)
                relative_key = target.relative_to(temporary).as_posix()
                if os.name == "nt":
                    relative_key = relative_key.casefold()
                if relative_key in seen_targets:
                    raise BootstrapError(f"duplicate archive target: {member.name}")
                seen_targets.add(relative_key)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = handle.extractfile(member)
                if source is None:
                    raise BootstrapError(f"cannot extract archive member: {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
        if marker_payload is not None:
            completed_marker = dict(marker_payload)
            completed_marker["inventory_sha256"] = _directory_inventory_sha256(
                temporary,
                excluded_names=frozenset({".discovery-source.json"}),
                excluded_directories=frozenset(),
            )
            _atomic_json(temporary / ".discovery-source.json", completed_marker)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _confined_root(root: Path, *, allow_external: bool = False) -> Path:
    resolved = root.resolve()
    if resolved.exists() and not resolved.is_dir():
        raise BootstrapError("bootstrap root must be a directory")
    if resolved == Path(resolved.anchor):
        raise BootstrapError("bootstrap root cannot be a filesystem root")
    workspace = WORKSPACE.resolve()
    if not allow_external:
        _assert_below(resolved, workspace)
    if resolved == workspace:
        raise BootstrapError("bootstrap root must be a subdirectory of the workspace")
    return resolved


def _assert_below(path: Path, root: Path) -> None:
    if path != root and root not in path.parents:
        raise BootstrapError(f"path escapes configured root: {path}")


def _run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    display = " ".join(command[:5]) + (" ..." if len(command) > 5 else "")
    print(f"[bootstrap] run: {display}", flush=True)
    subprocess.run(command, check=True, env=env)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        Path(temporary_name).unlink(missing_ok=True)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install isolated Discovery OS scientific integrations from a pinned manifest."
    )
    parser.add_argument("command", choices=["plan", "install", "doctor", "verify-manifest"])
    parser.add_argument("--profile", default="core")
    parser.add_argument("--accelerator", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--allow-custom-manifest", action="store_true")
    parser.add_argument("--allow-external-root", action="store_true")
    parser.add_argument("--include-weights", action="store_true")
    parser.add_argument("--accept-license", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    try:
        manifest = load_manifest(
            args.manifest.resolve(),
            allow_custom_manifest=args.allow_custom_manifest,
        )
        accepted = set(args.accept_license)
        if args.command == "verify-manifest":
            result = {
                "schema_version": "1.0",
                "valid": True,
                "manifest_revision": manifest["manifest_revision"],
                "profiles": sorted(manifest["profiles"]),
                "component_count": len(manifest["components"]),
            }
        elif args.command == "plan":
            result = build_plan(
                manifest,
                args.profile,
                accelerator=args.accelerator,
                accepted_licenses=accepted,
                include_weights=args.include_weights,
            )
        elif args.command == "doctor":
            result = doctor(
                manifest,
                args.root,
                args.profile,
                allow_external_root=args.allow_external_root,
            )
        else:
            installer = Installer(
                manifest,
                args.root,
                accelerator=(
                    _detect_accelerator() if args.accelerator == "auto" else args.accelerator
                ),
                accepted_licenses=accepted,
                include_weights=args.include_weights,
                dry_run=args.dry_run,
                allow_external_root=args.allow_external_root,
            )
            result = installer.install(args.profile)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.require_all:
            if (
                result.get("status") in {"partial", "blocked"}
                or bool(result.get("unresolved"))
                or result.get("healthy") is False
            ):
                return 2
        return 0
    except (BootstrapError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"schema_version": "1.0", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
