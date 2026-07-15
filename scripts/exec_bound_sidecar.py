"""Exec one launch-plan sidecar with only its reviewed runtime bindings."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan")
    parser.add_argument("component_id")
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    payload = json.loads(Path(arguments.plan).read_text(encoding="utf-8"))
    matches = [
        item for item in payload.get("sidecars", [])
        if item.get("component_id") == arguments.component_id
    ]
    if len(matches) != 1:
        raise SystemExit("launch plan does not contain exactly one requested component")
    item = matches[0]
    base_names = {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "TZ",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key in base_names or key.startswith("LC_")
    }
    environment.update(
        {
            "SIDECAR_MODEL_VERSION": item["model_version"],
            "SIDECAR_CODE_REVISION": item["code_revision"],
            "SIDECAR_WEIGHT_REVISION": item["weight_revision"],
            "PYTHONUNBUFFERED": "1",
        }
    )
    runtime_environment = item.get("runtime_environment") or {}
    if not isinstance(runtime_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in runtime_environment.items()
    ):
        raise SystemExit("launch plan contains an invalid runtime_environment")
    environment.update(runtime_environment)
    command = list(item["command"])
    if arguments.preflight:
        command.append("--preflight")
    os.execve(item["python"], command, environment)


if __name__ == "__main__":
    main()
