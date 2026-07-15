"""Optional, isolated model servers for Discovery OS fusion contracts."""

from .app import create_sidecar_app
from .errors import (
    CandidateConversionError,
    ModelExecutionError,
    ModelOutputError,
    ModelTimeoutError,
    OptionalDependencyError,
    SidecarBusyError,
    SidecarError,
    UnsupportedModelError,
)
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
from .types import (
    ExpertResult,
    GeneratedBatch,
    GeneratedCandidateData,
    ModelIdentity,
    PropertyResult,
    SidecarLimits,
)


__all__ = [
    "BoltzExpert",
    "CHGNetExpert",
    "CandidateConversionError",
    "ChempropExpert",
    "ESMExpert",
    "ExpertResult",
    "GeneratedBatch",
    "GeneratedCandidateData",
    "MatterGenGenerator",
    "MatterSimExpert",
    "ModelExecutionError",
    "ModelIdentity",
    "ModelOutputError",
    "ModelTimeoutError",
    "OptionalDependencyError",
    "PropertyResult",
    "PySCFExpert",
    "QHNetExpert",
    "RNAFMExpert",
    "ReinventGenerator",
    "ScGPTExpert",
    "SidecarBusyError",
    "SidecarError",
    "SidecarLimits",
    "UMAExpert",
    "UniMolExpert",
    "UnsupportedModelError",
    "create_sidecar_app",
]
