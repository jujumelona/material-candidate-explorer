"""Sidecar-specific failures with safe, actionable client messages."""

from __future__ import annotations


class SidecarError(RuntimeError):
    """Base class for errors that may cross the sidecar HTTP boundary."""

    error_code = "sidecar_error"
    status_code = 500

    def __init__(self, message: str) -> None:
        self.safe_message = message
        super().__init__(message)


class UnsupportedModelError(SidecarError):
    """The requested model integration is deliberately unavailable.

    This is preferable to returning a fabricated tensor or prediction when an
    upstream package has no stable, versioned inference entrypoint.
    """

    error_code = "unsupported_model"
    status_code = 503


class OptionalDependencyError(UnsupportedModelError):
    error_code = "optional_dependency_missing"


class CandidateConversionError(SidecarError):
    error_code = "candidate_conversion_failed"
    status_code = 422


class ModelExecutionError(SidecarError):
    error_code = "model_execution_failed"
    status_code = 502


class ModelOutputError(SidecarError):
    error_code = "invalid_model_output"
    status_code = 502


class RequestLimitError(SidecarError):
    error_code = "request_limit_exceeded"
    status_code = 413


class SidecarBusyError(SidecarError):
    error_code = "sidecar_busy"
    status_code = 429


class ModelTimeoutError(SidecarError):
    error_code = "model_timeout"
    status_code = 504


__all__ = [
    "CandidateConversionError",
    "ModelExecutionError",
    "ModelOutputError",
    "ModelTimeoutError",
    "OptionalDependencyError",
    "RequestLimitError",
    "SidecarBusyError",
    "SidecarError",
    "UnsupportedModelError",
]
