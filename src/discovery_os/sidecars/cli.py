"""Command-line launcher for one-model-per-environment sidecars."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
from collections.abc import Mapping
import sys
from typing import Any

from discovery_os.integration_manifest import load_integration_manifest
from discovery_os.hashing import stable_hash

from .app import create_sidecar_app
from .base import runtime_provenance_parameters
from .experts import (
    BoltzExpert,
    CHGNetExpert,
    ChempropExpert,
    ESMExpert,
    MatterSimExpert,
    PySCFExpert,
    QHNetExpert,
    RNAFMExpert,
    ScGPTExpert,
    UMAExpert,
    UniMolExpert,
)
from .generators import MatterGenGenerator, ReinventGenerator
from .errors import SidecarError, UnsupportedModelError
from .qhnet import attest_qhnet_bundle
from .types import ModelIdentity, SidecarLimits
from .weight_binding import (
    WeightBindingError,
    attest_file_revision,
    directory_inventory_sha256,
    require_snapshot_member,
    verify_huggingface_snapshot,
)


DEFAULT_PORTS = {
    "mattergen": 8101,
    "unimol": 8102,
    "boltz": 8103,
    "esm": 8104,
    "rnafm": 8105,
    "scgpt": 8106,
    "qhnet": 8107,
    "pyscf": 8108,
    "uma": 8109,
    "mattersim": 8110,
    "chemprop": 8111,
    "reinvent4": 8112,
    "chgnet": 8113,
}

_QHNET_RUNTIME_VERSIONS = {
    "torch": "2.2.0",
    "torch-geometric": "2.5.3",
    "torch-scatter": "2.1.2",
    "torch-cluster": "1.6.3",
    "e3nn": "0.5.1",
}


# Only module/executable presence is checked here. Checkpoints remain lazy and
# are loaded by the adapter on its first real request.
_MODEL_MODULES = {
    "mattergen": "mattergen",
    "unimol": "unimol_tools",
    "boltz": "boltz",
    "esm": "esm",
    "rnafm": "fm",
    "scgpt": "scgpt",
    "pyscf": "pyscf",
    "uma": "fairchem",
    "mattersim": "mattersim",
    "chemprop": "chemprop",
    "chgnet": "chgnet",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one isolated Discovery OS model sidecar")
    parser.add_argument("--model", required=True, choices=sorted(DEFAULT_PORTS))
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate support, dependencies, and configuration without loading a checkpoint",
    )
    args = parser.parse_args(argv)

    values = os.environ
    try:
        identity, runtime, limits, host, port = _prepare_sidecar(
            args.model,
            values,
            host_override=args.host,
            port_override=args.port,
        )
    except (OSError, SidecarError, TypeError, ValueError) as exc:
        if not args.preflight:
            raise
        print(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "supported": False,
                    "model": args.model,
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    if args.preflight:
        print(
            json.dumps(
                _preflight_payload(identity, runtime, limits, host=host, port=port),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    app = create_sidecar_app(identity=identity, runtime=runtime, limits=limits)
    try:
        import uvicorn
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError("install uvicorn in this sidecar environment") from exc
    uvicorn.run(app, host=host, port=port, workers=1, log_level=args.log_level)
    return 0


def preflight_configuration(
    model: str,
    values: Mapping[str, str],
    *,
    host: str | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    """Validate one sidecar without importing or loading its checkpoint.

    Launch orchestration calls this in every isolated component environment
    before it starts the first web server.  It deliberately checks only static
    configuration, declared adapter support, package/CLI presence, and supplied
    path existence; heavyweight model construction remains request-lazy.
    """

    if model not in DEFAULT_PORTS:
        raise ValueError(f"unknown sidecar model {model!r}")
    identity, runtime, limits, resolved_host, resolved_port = _prepare_sidecar(
        model,
        values,
        host_override=host,
        port_override=port,
    )
    return _preflight_payload(
        identity,
        runtime,
        limits,
        host=resolved_host,
        port=resolved_port,
    )


def _prepare_sidecar(
    model: str,
    values: Mapping[str, str],
    *,
    host_override: str | None,
    port_override: int | None,
) -> tuple[ModelIdentity, Any, SidecarLimits, str, int]:
    effective_values = _bind_weight_configuration(model, values)
    identity = _identity(model, effective_values)
    runtime = _runtime(model, effective_values)
    identity = replace(
        identity,
        runtime_parameters_hash=stable_hash(runtime_provenance_parameters(runtime)),
    )
    if not bool(getattr(runtime, "supported", True)):
        reason = str(getattr(runtime, "reason", "adapter is marked unsupported"))
        action = str(getattr(runtime, "install_action", "configure a reviewed runtime"))
        raise UnsupportedModelError(
            f"{identity.model_id} cannot be launched: {reason}; {action}"
        )
    _validate_static_configuration(model, effective_values)
    _validate_runtime_dependency(model)
    limits = _limits(effective_values)
    host = (
        host_override
        if host_override is not None
        else effective_values.get("SIDECAR_HOST", "127.0.0.1")
    )
    if not isinstance(host, str) or not host.strip() or any(char in host for char in "\r\n\t"):
        raise ValueError("sidecar host must be a non-blank host without control characters")
    host = host.strip()
    port = (
        port_override
        if port_override is not None
        else _integer(effective_values, "SIDECAR_PORT", DEFAULT_PORTS[model])
    )
    if not 1 <= port <= 65_535:
        raise ValueError("sidecar port must be between 1 and 65535")
    return identity, runtime, limits, host, port


def _preflight_payload(
    identity: ModelIdentity,
    runtime: Any,
    limits: SidecarLimits,
    *,
    host: str,
    port: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "supported": True,
        "configuration_only": True,
        "checkpoint_loaded": bool(getattr(runtime, "loaded", False)),
        "model_id": identity.model_id,
        "model_version": identity.model_version,
        "code_revision": identity.code_revision,
        "weight_revision": identity.weight_revision,
        "runtime_parameters_hash": identity.runtime_parameters_hash,
        "host": host,
        "port": port,
        "limits": {
            "max_request_bytes": limits.max_request_bytes,
            "max_batch_size": limits.max_batch_size,
            "max_concurrency": limits.max_concurrency,
            "max_queue_size": limits.max_queue_size,
            "request_timeout_seconds": limits.request_timeout_seconds,
        },
    }


def _bind_weight_configuration(
    model: str,
    values: Mapping[str, str],
) -> dict[str, str]:
    """Resolve the selected local bytes before constructing a runtime.

    Fixed Hugging Face components must point at bootstrap's verified snapshot.
    Manual files are promoted to their measured SHA-256.  Managed checkpoints
    without a selectable local file are explicitly marked non-byte-attested.
    """

    effective = {str(key): str(value) for key, value in values.items()}
    component_id = "qhnet-source" if model == "qhnet" else model
    component = next(
        (
            item
            for item in load_integration_manifest().components
            if item.component_id == component_id
        ),
        None,
    )
    if component is None or not component.weights:
        return effective
    if len(component.weights) != 1:
        raise WeightBindingError(
            f"{component_id} requires an explicit multi-weight binding policy"
        )
    weight = component.weights[0]
    declared = effective.get("SIDECAR_WEIGHT_REVISION")
    if weight.kind == "huggingface":
        if weight.repository is None or weight.revision is None:
            raise WeightBindingError("fixed Hugging Face weight lacks repository/revision")
        snapshot_raw = effective.get("SIDECAR_WEIGHT_SNAPSHOT_PATH", "").strip()
        if not snapshot_raw:
            raise WeightBindingError(
                f"{component_id} requires SIDECAR_WEIGHT_SNAPSHOT_PATH from bootstrap"
            )
        snapshot = verify_huggingface_snapshot(
            snapshot_raw,
            repository=weight.repository,
            revision=weight.revision,
        )
        if declared and declared.strip() != weight.revision:
            raise WeightBindingError(
                f"{component_id} weight revision must equal verified snapshot {weight.revision}"
            )
        effective["SIDECAR_WEIGHT_REVISION"] = weight.revision
        effective["SIDECAR_WEIGHT_ATTESTATION"] = (
            f"huggingface:{weight.repository}@{weight.revision}"
        )
        if model == "mattergen":
            name = effective.get("MATTERGEN_PRETRAINED_NAME", "mattergen_base").strip()
            path = require_snapshot_member(
                snapshot,
                f"checkpoints/{name}",
                kind="directory",
            )
            effective["MATTERGEN_CHECKPOINT_PATH"] = str(path)
        elif model == "unimol":
            remove_hs = _boolean(effective, "UNIMOL_REMOVE_HS", False)
            checkpoint_name = (
                "mol_pre_no_h_220816.pt" if remove_hs else "mol_pre_all_h_220816.pt"
            )
            effective["UNIMOL_CHECKPOINT_PATH"] = str(
                require_snapshot_member(snapshot, checkpoint_name, kind="file")
            )
            effective["UNIMOL_DICTIONARY_PATH"] = str(
                require_snapshot_member(snapshot, "mol.dict.txt", kind="file")
            )
        elif model == "esm":
            require_snapshot_member(
                snapshot,
                "data/weights/esm3_sm_open_v1.pth",
                kind="file",
            )
            effective["ESM_SNAPSHOT_PATH"] = str(snapshot)
        elif model == "rnafm":
            effective["RNAFM_CHECKPOINT_PATH"] = str(
                require_snapshot_member(snapshot, "RNA-FM_pretrained.pth", kind="file")
            )
        elif model == "uma":
            model_name = effective.get("UMA_MODEL_NAME", "uma-s-1p2").strip()
            effective["UMA_CHECKPOINT_PATH"] = str(
                require_snapshot_member(
                    snapshot,
                    f"{model_name}.pt",
                    kind="file",
                )
            )
        elif model == "boltz":
            effective["BOLTZ_CHECKPOINT_PATH"] = str(
                require_snapshot_member(snapshot, "boltz2_conf.ckpt", kind="file")
            )
            effective["BOLTZ_AFFINITY_CHECKPOINT_PATH"] = str(
                require_snapshot_member(snapshot, "boltz2_aff.ckpt", kind="file")
            )
            effective["BOLTZ_MOLS_TAR_PATH"] = str(
                require_snapshot_member(snapshot, "mols.tar", kind="file")
            )
            effective.setdefault(
                "BOLTZ_CACHE",
                str(snapshot.parent / f"{snapshot.name}-runtime-cache"),
            )
        else:
            raise WeightBindingError(
                f"{component_id} has fixed weights but no reviewed local snapshot binding"
            )
        return effective

    if weight.kind == "manual":
        if model == "qhnet":
            bundle = attest_qhnet_bundle(
                _required(effective, "QHNET_CHECKPOINT_PATH"),
                _required(effective, "QHNET_CONFIG_PATH"),
                declared_revision=declared,
            )
            effective["QHNET_CHECKPOINT_PATH"] = str(bundle.checkpoint_path)
            effective["QHNET_CONFIG_PATH"] = str(bundle.config_path)
            effective["SIDECAR_WEIGHT_REVISION"] = bundle.revision
            effective["SIDECAR_WEIGHT_ATTESTATION"] = bundle.revision
            return effective
        if model == "scgpt":
            selected_bundle = Path(
                _required(effective, "SCGPT_CHECKPOINT_DIR")
            ).expanduser()
            if selected_bundle.is_symlink():
                raise WeightBindingError("SCGPT_CHECKPOINT_DIR must not be a symlink")
            bundle = selected_bundle.resolve(strict=True)
            if not bundle.is_dir():
                raise WeightBindingError(
                    "SCGPT_CHECKPOINT_DIR must be a regular non-symlink directory"
                )
            for name in ("args.json", "vocab.json", "best_model.pt"):
                require_snapshot_member(bundle, name, kind="file")
            digest = directory_inventory_sha256(bundle)
            attestation = f"sha256:{digest}"
            if declared and declared.strip().lower().startswith("sha256:") and (
                declared.strip().lower() != attestation
            ):
                raise WeightBindingError(
                    "scgpt declared weight revision conflicts with SCGPT_CHECKPOINT_DIR"
                )
            effective["SCGPT_CHECKPOINT_DIR"] = str(bundle)
            effective["SCGPT_BUNDLE_INVENTORY_SHA256"] = digest
            effective["SIDECAR_WEIGHT_REVISION"] = attestation
            effective["SIDECAR_WEIGHT_ATTESTATION"] = attestation
            return effective
        path_name = {
            "chemprop": "CHEMPROP_CHECKPOINT_PATH",
            "reinvent4": "REINVENT_MODEL_FILE",
        }.get(model)
        if path_name is None:
            raise WeightBindingError(
                f"{component_id} manual weight has no reviewed local file binding"
            )
        path = _required(effective, path_name)
        attestation = attest_file_revision(
            path,
            declared_revision=declared,
            label=component_id,
        )
        effective["SIDECAR_WEIGHT_REVISION"] = attestation
        effective["SIDECAR_WEIGHT_ATTESTATION"] = attestation
        return effective

    if weight.kind == "managed":
        if model == "mattersim" and effective.get("MATTERSIM_CHECKPOINT_PATH", "").strip():
            attestation = attest_file_revision(
                effective["MATTERSIM_CHECKPOINT_PATH"],
                declared_revision=declared,
                label="mattersim",
            )
            effective["SIDECAR_WEIGHT_REVISION"] = attestation
            effective["SIDECAR_WEIGHT_ATTESTATION"] = attestation
            return effective
        if declared is None or not declared.strip():
            raise WeightBindingError(
                f"SIDECAR_WEIGHT_REVISION is required for managed component {component_id}"
            )
        normalized = declared.strip()
        if normalized.lower().startswith("sha256:"):
            raise WeightBindingError(
                f"{component_id} cannot claim a SHA-256 without a selected local checkpoint file"
            )
        if not normalized.startswith("managed-unattested:"):
            normalized = f"managed-unattested:{normalized}"
        effective["SIDECAR_WEIGHT_REVISION"] = normalized
        effective["SIDECAR_WEIGHT_ATTESTATION"] = normalized
        return effective
    raise WeightBindingError(f"unsupported weight kind for {component_id}: {weight.kind}")


def _validate_static_configuration(model: str, values: Mapping[str, str]) -> None:
    device = values.get("SIDECAR_DEVICE", "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda", "mps"} and not device.startswith("cuda:"):
        raise ValueError("SIDECAR_DEVICE must be auto, cpu, cuda, cuda:N, or mps")
    if device.startswith("cuda:"):
        suffix = device.removeprefix("cuda:")
        if not suffix.isdigit():
            raise ValueError("SIDECAR_DEVICE cuda index must be a non-negative integer")

    path_rules: tuple[tuple[str, str], ...]
    if model == "mattergen":
        path_rules = (("MATTERGEN_CHECKPOINT_PATH", "exists"),)
    elif model == "unimol":
        path_rules = (
            ("UNIMOL_CHECKPOINT_PATH", "file"),
            ("UNIMOL_DICTIONARY_PATH", "file"),
        )
    elif model == "uma":
        path_rules = (("UMA_CHECKPOINT_PATH", "file"),)
    elif model == "esm":
        path_rules = (("ESM_SNAPSHOT_PATH", "exists"),)
    elif model == "rnafm":
        path_rules = (("RNAFM_CHECKPOINT_PATH", "file"),)
    elif model == "scgpt":
        path_rules = (("SCGPT_CHECKPOINT_DIR", "directory"),)
    elif model == "qhnet":
        path_rules = (
            ("QHNET_SOURCE_PATH", "directory"),
            ("QHNET_CHECKPOINT_PATH", "file"),
            ("QHNET_CONFIG_PATH", "file"),
        )
    elif model == "boltz":
        path_rules = (
            ("BOLTZ_CHECKPOINT_PATH", "file"),
            ("BOLTZ_AFFINITY_CHECKPOINT_PATH", "file"),
            ("BOLTZ_MOLS_TAR_PATH", "file"),
        )
    elif model == "mattersim":
        path_rules = (("MATTERSIM_CHECKPOINT_PATH", "file"),)
    elif model == "chemprop":
        path_rules = (("CHEMPROP_CHECKPOINT_PATH", "file"),)
    elif model == "reinvent4":
        path_rules = (("REINVENT_MODEL_FILE", "file"),)
    else:
        path_rules = ()
    for name, kind in path_rules:
        raw = values.get(name)
        if raw is None or not raw.strip():
            # Required paths were already rejected by _runtime; optional paths
            # remain absent without forcing a checkpoint choice.
            continue
        try:
            resolved = Path(raw.strip()).expanduser().resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{name} does not exist or cannot be resolved") from exc
        if kind == "file" and not resolved.is_file():
            raise ValueError(f"{name} must point to a file")
        if kind == "directory" and not resolved.is_dir():
            raise ValueError(f"{name} must point to a directory")

    nonblank_names = {
        "mattergen": ("MATTERGEN_PRETRAINED_NAME",),
        "uma": ("UMA_MODEL_NAME", "UMA_TASK_NAME"),
        "chgnet": ("CHGNET_MODEL_NAME",),
        "esm": ("ESM_MODEL_NAME",),
        "pyscf": ("PYSCF_BASIS",),
    }.get(model, ())
    for name in nonblank_names:
        if name in values and not values[name].strip():
            raise ValueError(f"{name} must not be blank")
    if model == "boltz":
        if device == "mps":
            raise ValueError("Boltz 2.2.1 supports cpu/cuda, not SIDECAR_DEVICE=mps")
        cache_raw = values.get("BOLTZ_CACHE", "~/.boltz").strip()
        if not cache_raw:
            raise ValueError("BOLTZ_CACHE must not be blank")
        cache = Path(cache_raw).expanduser().resolve()
        if cache.exists() and not cache.is_dir():
            raise ValueError("BOLTZ_CACHE must point to a directory or a creatable path")
        process_timeout = _float(values, "BOLTZ_PROCESS_TIMEOUT_SECONDS", 840.0)
        request_timeout = _float(values, "SIDECAR_TIMEOUT_SECONDS", 900.0)
        if process_timeout <= 0 or process_timeout >= request_timeout:
            raise ValueError(
                "BOLTZ_PROCESS_TIMEOUT_SECONDS must be positive and lower than SIDECAR_TIMEOUT_SECONDS"
            )
    if model == "qhnet" and device == "mps":
        raise ValueError("QHNet supports SIDECAR_DEVICE=cpu/cuda, not mps")


def _validate_runtime_dependency(model: str) -> None:
    # FastAPI and Uvicorn are part of every operational sidecar environment.
    for common in ("fastapi", "uvicorn"):
        if not _module_available(common):
            raise ValueError(f"sidecar runtime dependency {common!r} is not installed")
    if model == "reinvent4":
        # _runtime already resolved and bound an exact executable path.
        return
    if model == "qhnet":
        for module_name in ("torch", "torch_geometric", "torch_cluster", "torch_scatter", "e3nn"):
            if not _module_available(module_name):
                raise ValueError(
                    f"QHNet runtime dependency {module_name!r} is not installed in this sidecar environment"
                )
        for distribution, expected in _QHNET_RUNTIME_VERSIONS.items():
            try:
                actual = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError as exc:
                raise ValueError(
                    f"QHNet runtime distribution {distribution!r} is not installed"
                ) from exc
            comparable = actual.split("+", 1)[0] if distribution == "torch" else actual
            if comparable != expected:
                raise ValueError(
                    f"QHNet requires {distribution}=={expected}, found {actual}"
                )
        return
    module_name = _MODEL_MODULES.get(model)
    if module_name is not None and not _module_available(module_name):
        raise ValueError(
            f"model dependency {module_name!r} is not installed in this sidecar environment"
        )


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _required_reinvent_executable() -> str:
    """Resolve REINVENT from the active isolated environment before PATH."""

    # Do not resolve the Python executable symlink before taking its parent:
    # POSIX virtual environments commonly link bin/python to an interpreter
    # outside the environment, while their console scripts remain in bin/.
    environment_bin = Path(sys.executable).expanduser().absolute().parent
    names = ("reinvent.exe", "reinvent") if os.name == "nt" else ("reinvent", "reinvent.exe")
    for name in names:
        candidate = environment_bin / name
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate.resolve())
    discovered = shutil.which("reinvent")
    if discovered is not None:
        return str(Path(discovered).resolve())
    raise ValueError(
        "REINVENT executable was not found beside the active sidecar Python or on PATH"
    )


def _required_boltz_executable() -> str:
    """Resolve the pinned Boltz console script from the active environment."""

    environment_bin = Path(sys.executable).expanduser().absolute().parent
    names = ("boltz.exe", "boltz") if os.name == "nt" else ("boltz", "boltz.exe")
    for name in names:
        candidate = environment_bin / name
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return str(candidate.resolve())
    discovered = shutil.which("boltz")
    if discovered is not None:
        return str(Path(discovered).resolve())
    raise ValueError("Boltz executable was not found beside the active sidecar Python or on PATH")


def _identity(model: str, values: Mapping[str, str]) -> ModelIdentity:
    component_id = "qhnet-source" if model == "qhnet" else model
    component = next(
        (item for item in load_integration_manifest().components if item.component_id == component_id),
        None,
    )
    manifest_version = None
    manifest_code = None
    manifest_weight = None
    if component is not None:
        manifest_version = component.install.version or (
            component.source.release if component.source is not None else None
        )
        manifest_code = component.source.revision if component.source is not None else None
        exact = {item.revision for item in component.weights if item.revision is not None}
        unresolved = [item for item in component.weights if item.revision is None]
        if len(exact) == 1 and not unresolved:
            manifest_weight = next(iter(exact))
        elif not component.weights:
            manifest_weight = "no-external-weight"
    model_version = _required_or_default(values, "SIDECAR_MODEL_VERSION", manifest_version)
    code_revision = _required_or_default(values, "SIDECAR_CODE_REVISION", manifest_code)
    weight_revision = _required_or_default(values, "SIDECAR_WEIGHT_REVISION", manifest_weight)
    capability = "generate" if model in {"mattergen", "reinvent4"} else "features"
    return ModelIdentity(
        model_id=component_id,
        model_version=model_version,
        adapter_version=values.get("SIDECAR_ADAPTER_VERSION", "1.0.0").strip(),
        code_revision=code_revision,
        weight_revision=weight_revision,
        capabilities=frozenset({capability}),
        projection_version=values.get("SIDECAR_PROJECTION_VERSION") or None,
    )


def _runtime(model: str, values: Mapping[str, str]) -> Any:
    device = values.get("SIDECAR_DEVICE", "auto")
    if model == "mattergen":
        objective_map_raw = values.get("MATTERGEN_OBJECTIVE_MAP", "{}")
        try:
            objective_map = json.loads(objective_map_raw)
        except json.JSONDecodeError as exc:
            raise ValueError("MATTERGEN_OBJECTIVE_MAP must be a JSON object") from exc
        if not isinstance(objective_map, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in objective_map.items()
        ):
            raise ValueError("MATTERGEN_OBJECTIVE_MAP must map strings to strings")
        return MatterGenGenerator(
            pretrained_name=values.get("MATTERGEN_PRETRAINED_NAME", "mattergen_base"),
            checkpoint_path=values.get("MATTERGEN_CHECKPOINT_PATH") or None,
            objective_map=objective_map,
            device=device,
        )
    if model == "reinvent4":
        return ReinventGenerator(
            model_file=_required(values, "REINVENT_MODEL_FILE"),
            mode=values.get("REINVENT_MODE", "reinvent"),
            executable=_required_reinvent_executable(),
            device=device,
        )
    if model == "unimol":
        return UniMolExpert(
            checkpoint_path=values.get("UNIMOL_CHECKPOINT_PATH") or None,
            dictionary_path=values.get("UNIMOL_DICTIONARY_PATH") or None,
            remove_hs=_boolean(values, "UNIMOL_REMOVE_HS", False),
            device=device,
        )
    if model == "uma":
        return UMAExpert(
            model_name=values.get("UMA_MODEL_NAME", "uma-s-1p2"),
            task_name=values.get("UMA_TASK_NAME", "omat"),
            checkpoint_path=values.get("UMA_CHECKPOINT_PATH") or None,
            device=device,
        )
    if model == "mattersim":
        return MatterSimExpert(
            checkpoint_path=values.get("MATTERSIM_CHECKPOINT_PATH") or None,
            weight_attestation=values.get("SIDECAR_WEIGHT_ATTESTATION") or None,
            device=device,
        )
    if model == "chgnet":
        return CHGNetExpert(
            model_name=values.get("CHGNET_MODEL_NAME", "0.3.0"),
            weight_attestation=values.get("SIDECAR_WEIGHT_ATTESTATION") or None,
            device=device,
        )
    if model == "chemprop":
        names = _required_csv(values, "CHEMPROP_PROPERTY_NAMES")
        units = _required_csv(values, "CHEMPROP_PROPERTY_UNITS")
        return ChempropExpert(
            checkpoint_path=_required(values, "CHEMPROP_CHECKPOINT_PATH"),
            property_names=names,
            property_units=units,
            encoding_layer=_integer(values, "CHEMPROP_ENCODING_LAYER", 0),
            device=device,
        )
    if model == "esm":
        return ESMExpert(
            model_name=values.get("ESM_MODEL_NAME", "esm3_sm_open_v1"),
            snapshot_path=values.get("ESM_SNAPSHOT_PATH") or None,
            device=device,
        )
    if model == "rnafm":
        return RNAFMExpert(
            checkpoint_path=values.get("RNAFM_CHECKPOINT_PATH") or None,
            device=device,
        )
    if model == "pyscf":
        return PySCFExpert(basis=values.get("PYSCF_BASIS", "def2-svp"), device=device)
    if model == "boltz":
        return BoltzExpert(
            executable=_required_boltz_executable(),
            cache_path=values.get("BOLTZ_CACHE") or None,
            checkpoint_path=values.get("BOLTZ_CHECKPOINT_PATH") or None,
            affinity_checkpoint_path=values.get("BOLTZ_AFFINITY_CHECKPOINT_PATH") or None,
            mols_tar_path=values.get("BOLTZ_MOLS_TAR_PATH") or None,
            process_timeout_seconds=_float(values, "BOLTZ_PROCESS_TIMEOUT_SECONDS", 840.0),
            max_json_bytes=_integer(values, "BOLTZ_MAX_JSON_BYTES", 1024 * 1024),
            max_cif_bytes=_integer(values, "BOLTZ_MAX_CIF_BYTES", 8 * 1024 * 1024),
            max_sequence_length=_integer(values, "BOLTZ_MAX_SEQUENCE_LENGTH", 16_384),
            max_smiles_length=_integer(values, "BOLTZ_MAX_SMILES_LENGTH", 8_192),
            no_kernels=_boolean(values, "BOLTZ_NO_KERNELS", False),
            device=device,
        )
    if model == "scgpt":
        return ScGPTExpert(
            checkpoint_dir=_required(values, "SCGPT_CHECKPOINT_DIR"),
            max_genes=_integer(values, "SCGPT_MAX_GENES", 65_536),
            max_length=_integer(values, "SCGPT_MAX_LENGTH", 1_200),
            use_fast_transformer=_boolean(
                values,
                "SCGPT_USE_FAST_TRANSFORMER",
                False,
            ),
            bundle_inventory_sha256=(
                values.get("SCGPT_BUNDLE_INVENTORY_SHA256") or None
            ),
            device=device,
        )
    if model == "qhnet":
        return QHNetExpert(
            source_path=_required(values, "QHNET_SOURCE_PATH"),
            checkpoint_path=_required(values, "QHNET_CHECKPOINT_PATH"),
            config_path=_required(values, "QHNET_CONFIG_PATH"),
            weight_attestation=values.get("SIDECAR_WEIGHT_ATTESTATION") or None,
            device=device,
        )
    raise ValueError(f"unknown sidecar model {model!r}")


def _limits(values: Mapping[str, str]) -> SidecarLimits:
    return SidecarLimits(
        max_request_bytes=_integer(values, "SIDECAR_MAX_REQUEST_BYTES", 8 * 1024 * 1024),
        max_batch_size=_integer(values, "SIDECAR_MAX_BATCH_SIZE", 32),
        max_concurrency=_integer(values, "SIDECAR_MAX_CONCURRENCY", 1),
        max_queue_size=_integer(values, "SIDECAR_MAX_QUEUE_SIZE", 2),
        request_timeout_seconds=_float(values, "SIDECAR_TIMEOUT_SECONDS", 900.0),
    )


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _required_csv(values: Mapping[str, str], name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in _required(values, name).split(","))
    if any(not item for item in items):
        raise ValueError(f"{name} must be a comma-separated list without blank entries")
    return items


def _required_or_default(values: Mapping[str, str], name: str, default: str | None) -> str:
    value = values.get(name)
    if value is not None and value.strip():
        return value.strip()
    if default is not None and default.strip():
        return default.strip()
    raise ValueError(
        f"{name} is required because the integration manifest has no single exact value"
    )


def _integer(values: Mapping[str, str], name: str, default: int) -> int:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float(values: Mapping[str, str], name: str, default: float) -> float:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


def _boolean(values: Mapping[str, str], name: str, default: bool) -> bool:
    raw = values.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes"}:
        return True
    if normalized in {"0", "false", "no"}:
        return False
    raise ValueError(f"{name} must be true or false")


__all__ = ["DEFAULT_PORTS", "main", "preflight_configuration"]
