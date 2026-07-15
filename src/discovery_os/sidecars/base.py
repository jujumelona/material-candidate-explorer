"""Lazy loading and optional-dependency helpers for isolated sidecars."""

from __future__ import annotations

import importlib
import math
import os
import threading
from abc import ABC, abstractmethod
from types import ModuleType
from typing import Any, Generic, TypeVar

from .errors import ModelOutputError, OptionalDependencyError


ModelT = TypeVar("ModelT")


def runtime_provenance_parameters(runtime: Any) -> dict[str, Any]:
    """Return model parameters plus the process-level accelerator binding."""

    provider = getattr(runtime, "provenance_parameters", None)
    raw = provider() if callable(provider) else {"runtime_class": type(runtime).__name__}
    if not isinstance(raw, dict) or not all(isinstance(key, str) for key in raw):
        raise TypeError("provenance_parameters() must return a string-keyed dictionary")
    parameters = dict(raw)
    if "execution_environment" in parameters:
        raise TypeError("runtime provenance reserves the execution_environment key")
    parameters["execution_environment"] = {
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES"),
        "NVIDIA_VISIBLE_DEVICES": os.getenv("NVIDIA_VISIBLE_DEVICES"),
        "ROCR_VISIBLE_DEVICES": os.getenv("ROCR_VISIBLE_DEVICES"),
        "HIP_VISIBLE_DEVICES": os.getenv("HIP_VISIBLE_DEVICES"),
    }
    return parameters


def require_module(module_name: str, *, install_hint: str) -> ModuleType:
    """Import an optional package only after a request reaches its sidecar."""

    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise OptionalDependencyError(
            f"optional package {module_name!r} is unavailable; {install_hint}"
        ) from exc


def resolve_device(requested: str = "auto") -> str:
    """Resolve CUDA/MPS lazily, without making torch a core dependency."""

    normalized = requested.strip().lower()
    if normalized not in {"auto", "cpu", "cuda", "mps"} and not normalized.startswith("cuda:"):
        raise ValueError("device must be auto, cpu, cuda, cuda:N, or mps")
    if normalized != "auto":
        return normalized
    try:
        torch = importlib.import_module("torch")
    except (ImportError, ModuleNotFoundError):
        return "cpu"
    if bool(torch.cuda.is_available()):
        return "cuda"
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and bool(mps.is_available()):
        return "mps"
    return "cpu"


class LazyModelAdapter(ABC, Generic[ModelT]):
    """Thread-safe, one-time model/checkpoint loader.

    Failed loads are cached as well.  This prevents a missing checkpoint or
    rejected gated model from causing an import/download storm on every HTTP
    request.  Restarting the sidecar is the explicit retry boundary.
    """

    def __init__(self, *, device: str = "auto") -> None:
        self._requested_device = device
        self._resolved_device: str | None = None
        self._model: ModelT | None = None
        self._load_error: BaseException | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def load_failed(self) -> bool:
        return self._load_error is not None

    @property
    def device(self) -> str:
        return self._resolved_device or self._requested_device

    def _ensure_loaded(self) -> ModelT:
        if self._model is not None:
            return self._model
        if self._load_error is not None:
            raise self._load_error
        with self._load_lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise self._load_error
            try:
                self._resolved_device = resolve_device(self._requested_device)
                model = self._load_model(self._resolved_device)
                if model is None:
                    raise ModelOutputError("model loader returned no model")
                self._model = model
            except BaseException as exc:
                self._load_error = exc
                raise
        return self._model

    @abstractmethod
    def _load_model(self, device: str) -> ModelT:
        raise NotImplementedError

    def close(self) -> None:
        model = self._model
        close = getattr(model, "close", None)
        if callable(close):
            close()


def to_plain_data(value: Any) -> Any:
    """Detach common tensor/array wrappers without importing their packages."""

    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    method = getattr(current, "numpy", None)
    if callable(method):
        current = method()
    method = getattr(current, "tolist", None)
    if callable(method):
        current = method()
    return current


def numeric_tensor_data(value: Any, *, max_values: int = 65_536) -> tuple[list[int], list[float]]:
    """Return a rectangular finite tensor shape and flattened JSON values."""

    plain = to_plain_data(value)
    shape = _infer_shape(plain)
    if not shape:
        shape = [1]
        plain = [plain]
    flat: list[float] = []
    _flatten_numeric(plain, flat, max_values=max_values)
    if math.prod(shape) != len(flat):
        raise ModelOutputError("model tensor is ragged")
    if not flat:
        raise ModelOutputError("model tensor is empty")
    return shape, flat


def _infer_shape(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)):
        return []
    length = len(value)
    if length == 0:
        raise ModelOutputError("model tensor contains an empty axis")
    child_shapes = [_infer_shape(item) for item in value]
    if any(item != child_shapes[0] for item in child_shapes[1:]):
        raise ModelOutputError("model tensor is ragged")
    return [length, *child_shapes[0]]


def _flatten_numeric(value: Any, target: list[float], *, max_values: int) -> None:
    if isinstance(value, (list, tuple)):
        for item in value:
            _flatten_numeric(item, target, max_values=max_values)
        return
    if isinstance(value, bool):
        raise ModelOutputError("boolean values are not valid numeric tensor entries")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError("model tensor contains a non-numeric value") from exc
    if not math.isfinite(number):
        raise ModelOutputError("model tensor contains NaN or infinity")
    target.append(number)
    if len(target) > max_values:
        raise ModelOutputError(f"model tensor exceeds the {max_values} value wire limit")


__all__ = [
    "LazyModelAdapter",
    "numeric_tensor_data",
    "require_module",
    "resolve_device",
    "runtime_provenance_parameters",
    "to_plain_data",
]
