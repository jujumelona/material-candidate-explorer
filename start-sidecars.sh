#!/usr/bin/env sh
set -eu

WORKSPACE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROFILE=all-open
COMPONENTS=
MANIFEST="$WORKSPACE/integrations/manifest.v1.json"
MANIFEST_SET=0
ALLOW_CUSTOM_MANIFEST=0
INSTALL_ROOT="$WORKSPACE/.discovery"
ALLOW_EXTERNAL_ROOT=0
DRY_RUN=0
READY_TIMEOUT_SECONDS=120

case "$(uname -s 2>/dev/null || echo unknown)" in
    Linux) HOST_PLATFORM=linux ;;
    Darwin) HOST_PLATFORM=darwin ;;
    *) echo "This launcher supports Linux and macOS POSIX hosts only." >&2; exit 2 ;;
esac

usage() {
    echo "Usage: $0 [--profile NAME] [--component ID] [--install-root PATH] [--manifest PATH]" >&2
    echo "          [--allow-custom-manifest] [--allow-external-root] [--dry-run]" >&2
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --profile) [ "$#" -ge 2 ] || { usage; exit 2; }; PROFILE=$2; shift 2 ;;
        --component) [ "$#" -ge 2 ] || { usage; exit 2; }; COMPONENTS="${COMPONENTS}${COMPONENTS:+,}$2"; shift 2 ;;
        --install-root) [ "$#" -ge 2 ] || { usage; exit 2; }; INSTALL_ROOT=$2; shift 2 ;;
        --manifest) [ "$#" -ge 2 ] || { usage; exit 2; }; MANIFEST=$2; MANIFEST_SET=1; shift 2 ;;
        --allow-custom-manifest) ALLOW_CUSTOM_MANIFEST=1; shift ;;
        --allow-external-root) ALLOW_EXTERNAL_ROOT=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        --ready-timeout-seconds) [ "$#" -ge 2 ] || { usage; exit 2; }; READY_TIMEOUT_SECONDS=$2; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

case "$READY_TIMEOUT_SECONDS" in
    ''|*[!0-9]*) echo "--ready-timeout-seconds must be an integer." >&2; exit 2 ;;
esac
[ "$READY_TIMEOUT_SECONDS" -ge 1 ] && [ "$READY_TIMEOUT_SECONDS" -le 86400 ] || {
    echo "--ready-timeout-seconds must be between 1 and 86400." >&2; exit 2;
}

if [ -z "${PYTHON:-}" ]; then
    if command -v python3 >/dev/null 2>&1; then PYTHON=python3
    elif command -v python >/dev/null 2>&1; then PYTHON=python
    else echo "Python 3 is required to validate the manifest and launch plan." >&2; exit 2
    fi
fi

PLAN_FILE=$(mktemp "${TMPDIR:-/tmp}/discovery-sidecars-plan.XXXXXX")
ROWS_FILE=$(mktemp "${TMPDIR:-/tmp}/discovery-sidecars-rows.XXXXXX")
trap 'rm -f "$PLAN_FILE" "$ROWS_FILE"' EXIT HUP INT TERM

export DISCOVERY_SIDECAR_WORKSPACE=$WORKSPACE
export DISCOVERY_SIDECAR_PROFILE=$PROFILE
export DISCOVERY_SIDECAR_COMPONENTS=$COMPONENTS
export DISCOVERY_SIDECAR_MANIFEST=$MANIFEST
export DISCOVERY_SIDECAR_MANIFEST_SET=$MANIFEST_SET
export DISCOVERY_SIDECAR_ALLOW_CUSTOM=$ALLOW_CUSTOM_MANIFEST
export DISCOVERY_SIDECAR_INSTALL_ROOT=$INSTALL_ROOT
export DISCOVERY_SIDECAR_ALLOW_EXTERNAL=$ALLOW_EXTERNAL_ROOT
export DISCOVERY_SIDECAR_HOST_PLATFORM=$HOST_PLATFORM

"$PYTHON" - "$PLAN_FILE" <<'PY'
import json
import hashlib
import os
import pathlib
import re
import sys

def fail(message):
    raise SystemExit(message)

def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key in manifest: {key}")
        result[key] = value
    return result

workspace = pathlib.Path(os.environ["DISCOVERY_SIDECAR_WORKSPACE"]).resolve()
host_platform = os.environ["DISCOVERY_SIDECAR_HOST_PLATFORM"]
install_root = pathlib.Path(os.environ["DISCOVERY_SIDECAR_INSTALL_ROOT"])
if not install_root.is_absolute():
    install_root = workspace / install_root
install_root = install_root.resolve()
if install_root == pathlib.Path(install_root.anchor):
    fail("InstallRoot cannot be a filesystem root")
allow_external = os.environ["DISCOVERY_SIDECAR_ALLOW_EXTERNAL"] == "1"
try:
    install_root.relative_to(workspace)
except ValueError:
    if not allow_external:
        fail("InstallRoot is outside the workspace; pass --allow-external-root explicitly")

manifest_path = pathlib.Path(os.environ["DISCOVERY_SIDECAR_MANIFEST"])
if not manifest_path.is_absolute():
    manifest_path = workspace / manifest_path
manifest_path = manifest_path.resolve()
default_manifest = (workspace / "integrations" / "manifest.v1.json").resolve()
if manifest_path != default_manifest and os.environ["DISCOVERY_SIDECAR_ALLOW_CUSTOM"] != "1":
    fail("custom manifests are disabled; pass --allow-custom-manifest to trust this file")
try:
    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=unique_object)
except (OSError, json.JSONDecodeError) as exc:
    fail(f"cannot read integration manifest: {exc}")
if manifest.get("schema_version") != "1.0" or not isinstance(manifest.get("components"), list):
    fail("integration manifest does not satisfy schema version 1.0")

by_id = {}
for component in manifest["components"]:
    component_id = component.get("component_id")
    if not isinstance(component_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", component_id):
        fail("manifest contains an unsafe component_id")
    if component_id in by_id:
        fail(f"duplicate component_id: {component_id}")
    by_id[component_id] = component

raw_components = os.environ["DISCOVERY_SIDECAR_COMPONENTS"]
if raw_components:
    selected = []
    for item in raw_components.split(","):
        if item and item not in selected:
            selected.append(item)
else:
    profile = os.environ["DISCOVERY_SIDECAR_PROFILE"]
    profiles = manifest.get("profiles", {})
    if profile not in profiles:
        fail(f"unknown profile: {profile}")
    selected = profiles[profile].get("components", [])

safe_metadata = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+/@=-]{0,255}\Z")
plans = []
for component_id in selected:
    if component_id not in by_id:
        fail(f"unknown component: {component_id}")
    component = by_id[component_id]
    api = component.get("api")
    if api is None:
        continue
    if host_platform not in component.get("platforms", []):
        fail(f"component {component_id!r} does not support this {host_platform} host")
    api_env = api.get("base_url_env")
    if not isinstance(api_env, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*_API_URL", api_env):
        fail(f"component {component_id!r} has an unsafe API environment name")
    port = api.get("default_port")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        fail(f"component {component_id!r} has an invalid port")
    source = component.get("source") or {}
    install = component.get("install") or {}
    # Match the central registry's provenance contract exactly: a package
    # version wins, then a named source release, otherwise the model version is
    # explicitly remote while code_revision retains the pinned commit.
    model_version = install.get("version") or source.get("release") or "remote"
    code_revision = source.get("revision") or "workspace"
    if not isinstance(model_version, str) or not safe_metadata.fullmatch(model_version):
        fail(f"component {component_id!r} has an unsafe model version")
    if not isinstance(code_revision, str) or not safe_metadata.fullmatch(code_revision):
        fail(f"component {component_id!r} has an unsafe code revision")
    weights = component.get("weights") or []
    prefix = api_env.removesuffix("_API_URL")
    weight_env = f"{prefix}_WEIGHT_REVISION"
    runtime_environment = {}
    component_device = os.environ.get(f"{prefix}_DEVICE", "").strip()
    if not component_device:
        component_device = os.environ.get("SIDECAR_DEVICE", "").strip()
    if component_device:
        runtime_environment["SIDECAR_DEVICE"] = component_device
    component_visible_devices = os.environ.get(
        f"{prefix}_CUDA_VISIBLE_DEVICES", ""
    ).strip()
    if component_visible_devices:
        if any(char in component_visible_devices for char in "\x00\r\n\t"):
            fail(f"{prefix}_CUDA_VISIBLE_DEVICES contains control characters")
        runtime_environment["CUDA_VISIBLE_DEVICES"] = component_visible_devices
    declared_revision = os.environ.get(weight_env, "").strip()
    if not weights:
        weight_revision = "no-external-weight"
    else:
        if len(weights) != 1:
            fail(f"{weight_env} is required because weight revisions are ambiguous")
        weight = weights[0]
        kind = weight.get("kind")
        if kind == "huggingface":
            repository, revision = weight.get("repository"), weight.get("revision")
            snapshot = (install_root / "models" / component_id / weight["weight_id"] / revision).resolve()
            try:
                snapshot.relative_to(install_root)
            except ValueError:
                fail(f"weight snapshot for {component_id!r} escapes InstallRoot")
            marker_path = snapshot / ".snapshot.json"
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
            except (OSError, json.JSONDecodeError) as exc:
                fail(f"verified snapshot marker is missing or unreadable for {component_id!r}: {exc}")
            if (marker.get("schema_version"), marker.get("repository"), marker.get("revision")) != ("1.0", repository, revision):
                fail(f"snapshot marker repository/revision does not match the manifest for {component_id!r}")
            inventory = marker.get("inventory_sha256")
            if not isinstance(inventory, str) or not re.fullmatch(r"[0-9a-f]{64}", inventory):
                fail(f"snapshot marker inventory_sha256 is missing or invalid for {component_id!r}")
            weight_revision = revision
            runtime_environment["SIDECAR_WEIGHT_SNAPSHOT_PATH"] = str(snapshot)
            runtime_environment["SIDECAR_WEIGHT_ATTESTATION"] = f"huggingface:{repository}@{revision}"
        elif kind == "manual":
            if component_id == "scgpt":
                raw_path = os.environ.get("SCGPT_CHECKPOINT_DIR", "").strip()
                if not raw_path:
                    fail("SCGPT_CHECKPOINT_DIR is required to start the scgpt sidecar")
                selected_bundle = pathlib.Path(raw_path).expanduser()
                if selected_bundle.is_symlink():
                    fail("SCGPT_CHECKPOINT_DIR must not be a symlink")
                bundle = selected_bundle.resolve()
                if not bundle.is_dir():
                    fail("SCGPT_CHECKPOINT_DIR must be a regular non-symlink directory")
                for name in ("args.json", "vocab.json", "best_model.pt"):
                    member = bundle / name
                    if not member.is_file() or member.is_symlink():
                        fail(
                            "SCGPT_CHECKPOINT_DIR requires regular non-symlink file "
                            f"{name!r}"
                        )
                files = []
                for candidate in bundle.rglob("*"):
                    relative = candidate.relative_to(bundle)
                    if ".cache" in relative.parts[:-1]:
                        continue
                    if candidate.is_symlink():
                        fail(f"SCGPT_CHECKPOINT_DIR inventory contains a symlink: {candidate}")
                    if candidate.is_file() and candidate.name != ".snapshot.json":
                        files.append(candidate)
                if not files:
                    fail("SCGPT_CHECKPOINT_DIR contains no checkpoint files")
                digest = hashlib.sha256()
                for candidate in sorted(
                    files,
                    key=lambda item: item.relative_to(bundle).as_posix(),
                ):
                    relative = candidate.relative_to(bundle).as_posix().encode("utf-8")
                    size = candidate.stat().st_size
                    digest.update(len(relative).to_bytes(4, "big"))
                    digest.update(relative)
                    digest.update(size.to_bytes(8, "big"))
                    with candidate.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                bundle_digest = digest.hexdigest()
                weight_revision = f"sha256:{bundle_digest}"
                if (
                    declared_revision.lower().startswith("sha256:")
                    and declared_revision.lower() != weight_revision
                ):
                    fail(
                        f"{weight_env} conflicts with the selected SCGPT_CHECKPOINT_DIR"
                    )
                runtime_environment["SCGPT_CHECKPOINT_DIR"] = str(bundle)
                runtime_environment["SCGPT_BUNDLE_INVENTORY_SHA256"] = bundle_digest
                runtime_environment["SIDECAR_WEIGHT_ATTESTATION"] = weight_revision
                path_name = None
            elif component_id == "qhnet-source":
                selected_paths = {}
                selected_digests = {}
                for selected_name in ("QHNET_CHECKPOINT_PATH", "QHNET_CONFIG_PATH"):
                    raw_path = os.environ.get(selected_name, "").strip()
                    if not raw_path:
                        fail(f"{selected_name} is required to start the QHNet sidecar")
                    selected = pathlib.Path(raw_path).expanduser()
                    if selected.is_symlink():
                        fail(f"{selected_name} must not be a symlink")
                    selected = selected.resolve()
                    if not selected.is_file():
                        fail(f"{selected_name} must be a regular non-symlink file")
                    digest = hashlib.sha256()
                    with selected.open("rb") as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    selected_paths[selected_name] = selected
                    selected_digests[selected_name] = digest.hexdigest()
                material = json.dumps(
                    {
                        "checkpoint_sha256": selected_digests["QHNET_CHECKPOINT_PATH"],
                        "config_sha256": selected_digests["QHNET_CONFIG_PATH"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                weight_revision = f"bundle-sha256:{hashlib.sha256(material).hexdigest()}"
                if declared_revision and declared_revision.lower() != weight_revision:
                    fail(f"{weight_env} conflicts with the selected QHNet checkpoint/config bundle")
                source_root = (install_root / "sources" / "qhnet-source").resolve()
                try:
                    source_root.relative_to(install_root)
                except ValueError:
                    fail("QHNet source path escapes InstallRoot")
                if not source_root.is_dir() or source_root.is_symlink():
                    fail("verified QHNet source was not found under InstallRoot; run bootstrap first")
                runtime_environment["QHNET_SOURCE_PATH"] = str(source_root)
                runtime_environment["QHNET_CHECKPOINT_PATH"] = str(selected_paths["QHNET_CHECKPOINT_PATH"])
                runtime_environment["QHNET_CONFIG_PATH"] = str(selected_paths["QHNET_CONFIG_PATH"])
                runtime_environment["SIDECAR_WEIGHT_ATTESTATION"] = weight_revision
                path_name = None
            else:
                path_name = {
                    "chemprop": "CHEMPROP_CHECKPOINT_PATH",
                    "reinvent4": "REINVENT_MODEL_FILE",
                }.get(component_id)
            if component_id in {"scgpt", "qhnet-source"}:
                pass
            elif path_name is None:
                fail(f"manual weight binding is not implemented for {component_id!r}")
            else:
                raw_path = os.environ.get(path_name, "").strip()
                if not raw_path:
                    fail(f"{path_name} is required to start the {component_id} sidecar")
                selected_path = pathlib.Path(raw_path).expanduser()
                if selected_path.is_symlink():
                    fail(f"{path_name} must be a regular non-symlink file")
                local_path = selected_path.resolve()
                if not local_path.is_file():
                    fail(f"{path_name} must be a regular non-symlink file")
                digest = hashlib.sha256()
                with local_path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                weight_revision = f"sha256:{digest.hexdigest()}"
                if declared_revision.lower().startswith("sha256:") and declared_revision.lower() != weight_revision:
                    fail(f"{weight_env} conflicts with the selected {path_name}")
                runtime_environment[path_name] = str(local_path)
                runtime_environment["SIDECAR_WEIGHT_ATTESTATION"] = weight_revision
        elif kind == "managed":
            local_raw = os.environ.get("MATTERSIM_CHECKPOINT_PATH", "").strip() if component_id == "mattersim" else ""
            if local_raw:
                selected_path = pathlib.Path(local_raw).expanduser()
                if selected_path.is_symlink():
                    fail("MATTERSIM_CHECKPOINT_PATH must be a regular non-symlink file")
                local_path = selected_path.resolve()
                if not local_path.is_file():
                    fail("MATTERSIM_CHECKPOINT_PATH must be a regular non-symlink file")
                digest = hashlib.sha256()
                with local_path.open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                weight_revision = f"sha256:{digest.hexdigest()}"
                if declared_revision.lower().startswith("sha256:") and declared_revision.lower() != weight_revision:
                    fail(f"{weight_env} conflicts with MATTERSIM_CHECKPOINT_PATH")
                runtime_environment["MATTERSIM_CHECKPOINT_PATH"] = str(local_path)
                runtime_environment["SIDECAR_WEIGHT_ATTESTATION"] = weight_revision
            else:
                if not declared_revision:
                    fail(f"{weight_env} is required because {component_id!r} uses managed weights")
                if declared_revision.lower().startswith("sha256:"):
                    fail(f"{weight_env} cannot claim SHA-256 without a selected local checkpoint file")
                weight_revision = declared_revision if declared_revision.startswith("managed-unattested:") else f"managed-unattested:{declared_revision}"
                runtime_environment["SIDECAR_WEIGHT_ATTESTATION"] = weight_revision
        else:
            fail(f"unsupported weight kind for {component_id!r}: {kind!r}")
    allowed_configuration_names = (
        "SIDECAR_MAX_REQUEST_BYTES", "SIDECAR_MAX_BATCH_SIZE",
        "SIDECAR_MAX_CONCURRENCY", "SIDECAR_MAX_QUEUE_SIZE", "SIDECAR_TIMEOUT_SECONDS",
        "MATTERGEN_PRETRAINED_NAME", "MATTERGEN_OBJECTIVE_MAP", "UNIMOL_REMOVE_HS",
        "REINVENT_MODE", "UMA_MODEL_NAME", "UMA_TASK_NAME", "CHGNET_MODEL_NAME",
        "CHEMPROP_PROPERTY_NAMES", "CHEMPROP_PROPERTY_UNITS", "CHEMPROP_ENCODING_LAYER", "ESM_MODEL_NAME", "PYSCF_BASIS",
        "SCGPT_MAX_GENES", "SCGPT_MAX_LENGTH", "SCGPT_USE_FAST_TRANSFORMER", "BOLTZ_CACHE",
        "BOLTZ_PROCESS_TIMEOUT_SECONDS", "BOLTZ_MAX_JSON_BYTES", "BOLTZ_MAX_CIF_BYTES",
        "BOLTZ_MAX_SEQUENCE_LENGTH", "BOLTZ_MAX_SMILES_LENGTH", "BOLTZ_NO_KERNELS",
    )
    for configuration_name in allowed_configuration_names:
        configuration_value = os.environ.get(configuration_name, "").strip()
        if configuration_value:
            runtime_environment[configuration_name] = configuration_value
    if not isinstance(weight_revision, str) or not safe_metadata.fullmatch(weight_revision):
        fail(f"{weight_env} contains unsupported characters or is blank")
    if component_id == "chemprop" and not os.environ.get("CHEMPROP_CHECKPOINT_PATH", "").strip():
        fail("CHEMPROP_CHECKPOINT_PATH is required to start the Chemprop sidecar")
    if component_id == "chemprop" and not os.environ.get("CHEMPROP_PROPERTY_NAMES", "").strip():
        fail("CHEMPROP_PROPERTY_NAMES is required to start the Chemprop sidecar")
    if component_id == "chemprop" and not os.environ.get("CHEMPROP_PROPERTY_UNITS", "").strip():
        fail("CHEMPROP_PROPERTY_UNITS is required to start the Chemprop sidecar")
    if component_id == "reinvent4" and not os.environ.get("REINVENT_MODEL_FILE", "").strip():
        fail("REINVENT_MODEL_FILE is required to start the REINVENT sidecar")
    # uv/venv commonly makes bin/python a symlink to a managed interpreter.
    # Keep the launcher path lexically confined to the installed environment,
    # matching bootstrap's environment layout without rejecting that valid link.
    python = install_root / "envs" / component_id / "bin" / "python"
    try:
        python.relative_to(install_root)
    except ValueError:
        fail(f"environment for {component_id!r} escapes InstallRoot")
    if not python.is_file():
        fail(f"installed Python for {component_id!r} was not found at {python}; run bootstrap first")
    cli_id = "qhnet" if component_id == "qhnet-source" else component_id
    url = f"http://127.0.0.1:{port}"
    command = [str(python), "-m", "discovery_os.sidecars", "--model", cli_id,
               "--host", "127.0.0.1", "--port", str(port)]
    plans.append({
        "component_id": component_id, "cli_model_id": cli_id, "python": str(python),
        "port": port, "api_env": api_env, "url": url,
        "model_version": model_version, "code_revision": code_revision,
        "weight_revision": weight_revision, "weight_env": weight_env,
        "runtime_environment": runtime_environment,
        "state_path": str(install_root / "state" / "sidecars" / f"{component_id}.json"),
        "command": command,
    })
if not plans:
    fail("the selected profile has no launchable API sidecars")
if len({item["port"] for item in plans}) != len(plans):
    fail("selected sidecars contain duplicate loopback ports")
if len({item["api_env"] for item in plans}) != len(plans):
    fail("selected sidecars contain duplicate API environment variables")
payload = {
    "schema_version": "1.0", "profile": os.environ["DISCOVERY_SIDECAR_PROFILE"],
    "install_root": str(install_root), "manifest_revision": manifest.get("manifest_revision"),
    "env_files": [str(install_root / "sidecars.env.ps1"), str(install_root / "sidecars.env.sh")],
    "sidecars": plans,
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

INSTALL_ROOT=$(
    "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["install_root"])' "$PLAN_FILE"
)
STATE_DIR="$INSTALL_ROOT/state/sidecars"
LOG_DIR="$INSTALL_ROOT/logs/sidecars"
mkdir -p -- "$STATE_DIR" "$LOG_DIR"

# Write only endpoint and immutable revision bindings. Tokens, passwords, API
# keys, and model credentials are intentionally never serialized.
write_environment_files() {
"$PYTHON" - "$PLAN_FILE" <<'PY'
import json
import os
import pathlib
import sys

plan = json.load(open(sys.argv[1], encoding="utf-8"))
root = pathlib.Path(plan["install_root"])
ps = ["# Generated by start-sidecars.sh. Contains endpoints and revisions only; no credentials.",
      "$env:DISCOVERY_ALLOW_INSECURE_LOCAL_HTTP = '1'"]
sh = ["# Generated by start-sidecars.sh. Contains endpoints and revisions only; no credentials.",
      "export DISCOVERY_ALLOW_INSECURE_LOCAL_HTTP='1'"]
def ps_quote(value): return "'" + value.replace("'", "''") + "'"
def sh_quote(value): return "'" + value.replace("'", "'\"'\"'") + "'"
for item in plan["sidecars"]:
    ps.append(f"$env:{item['api_env']} = {ps_quote(item['url'])}")
    ps.append(f"$env:{item['weight_env']} = {ps_quote(item['weight_revision'])}")
    sh.append(f"export {item['api_env']}={sh_quote(item['url'])}")
    sh.append(f"export {item['weight_env']}={sh_quote(item['weight_revision'])}")
    runtime_hash = item.get("runtime_parameters_hash")
    if runtime_hash:
        runtime_hash_env = item["api_env"].removesuffix("_API_URL") + "_RUNTIME_PARAMETERS_HASH"
        ps.append(f"$env:{runtime_hash_env} = {ps_quote(runtime_hash)}")
        sh.append(f"export {runtime_hash_env}={sh_quote(runtime_hash)}")
for name, lines in (("sidecars.env.ps1", ps), ("sidecars.env.sh", sh)):
    target = root / name
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
PY
}
write_environment_files

if [ "$DRY_RUN" -eq 1 ]; then
    "$PYTHON" - "$PLAN_FILE" <<'PY'
import json, sys
p = json.load(open(sys.argv[1], encoding="utf-8"))
p["dry_run"] = True
print(json.dumps(p, ensure_ascii=False, indent=2))
PY
    exit 0
fi

"$PYTHON" - "$PLAN_FILE" > "$ROWS_FILE" <<'PY'
import json, sys
for p in json.load(open(sys.argv[1], encoding="utf-8"))["sidecars"]:
    fields = [p[k] for k in ("component_id", "cli_model_id", "python", "port", "url",
                              "model_version", "code_revision", "weight_revision", "state_path")]
    if any("\t" in str(v) or "\n" in str(v) for v in fields):
        raise SystemExit("launch plan contains unsafe control characters")
    print("\t".join(map(str, fields)))
PY

TAB=$(printf '\t')
# Preflight every selected sidecar before starting the first process.
while IFS="$TAB" read -r component_id cli_id model_python port url model_version code_revision weight_revision state_path; do
    if [ -f "$state_path" ]; then
        if ! old_pid=$("$PYTHON" -c 'import json,sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
pid = int(value.get("pid", 0))
print(pid if pid > 0 else 0)' "$state_path"); then
            echo "Refusing to replace unreadable sidecar state: $state_path" >&2
            exit 1
        fi
        if [ "$old_pid" -gt 0 ] 2>/dev/null && kill -0 "$old_pid" 2>/dev/null; then
            echo "Refusing to overwrite live sidecar process state for '$component_id'." >&2
            exit 1
        fi
    fi
    if "$PYTHON" - "$port" <<'PY'
import socket, sys
s = socket.socket()
s.settimeout(.25)
try: occupied = s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0
finally: s.close()
raise SystemExit(0 if occupied else 1)
PY
    then
        echo "Refusing to start '$component_id': loopback port $port is already in use." >&2
        exit 1
    fi
done < "$ROWS_FILE"

# Validate every isolated environment before starting the first server. The
# CLI preflight checks declared support, required configuration, path/package
# presence, and limits without importing or loading the model checkpoint.
while IFS="$TAB" read -r component_id cli_id model_python port url model_version code_revision weight_revision state_path; do
    if preflight_output=$(
        "$PYTHON" "$WORKSPACE/scripts/exec_bound_sidecar.py" \
            "$PLAN_FILE" "$component_id" --preflight 2>&1
    ); then
        :
    else
        echo "Sidecar configuration preflight failed for '$component_id' before any server was started: $preflight_output" >&2
        exit 1
    fi
    DISCOVERY_PREFLIGHT_OUTPUT=$preflight_output "$PYTHON" - "$PLAN_FILE" "$component_id" <<'PY'
import json, os, pathlib, sys
plan_path = pathlib.Path(sys.argv[1])
component_id = sys.argv[2]
result = json.loads(os.environ["DISCOVERY_PREFLIGHT_OUTPUT"])
runtime_hash = result.get("runtime_parameters_hash")
if not isinstance(runtime_hash, str) or len(runtime_hash) != 64 or any(c not in "0123456789abcdef" for c in runtime_hash):
    raise SystemExit(f"sidecar preflight omitted runtime_parameters_hash for {component_id!r}")
plan = json.loads(plan_path.read_text(encoding="utf-8"))
matches = [item for item in plan["sidecars"] if item["component_id"] == component_id]
if len(matches) != 1:
    raise SystemExit("preflight component is missing or duplicated in launch plan")
matches[0]["runtime_parameters_hash"] = runtime_hash
temporary = plan_path.with_name(plan_path.name + f".{os.getpid()}.tmp")
temporary.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, plan_path)
PY
done < "$ROWS_FILE"

write_environment_files

write_state() {
    "$PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$9" "${10}" "${11}" "${12}" "${13}" <<'PY'
import datetime, json, os, pathlib, sys
state_path, component_id, pid, status, url, out_log, err_log, model_version, code_revision, weight_revision, cli_id, model_python, plan_path = sys.argv[1:]
plan = json.loads(pathlib.Path(plan_path).read_text(encoding="utf-8"))
matches = [item for item in plan["sidecars"] if item["component_id"] == component_id]
if len(matches) != 1:
    raise SystemExit("state writer could not find a unique component in the launch plan")
runtime_parameters_hash = matches[0].get("runtime_parameters_hash")
if not isinstance(runtime_parameters_hash, str) or len(runtime_parameters_hash) != 64:
    raise SystemExit("state writer requires the attested runtime_parameters_hash")
payload = {
    "schema_version": "1.0", "component_id": component_id, "pid": int(pid), "status": status,
    "url": url, "command": [model_python, "-m", "discovery_os.sidecars", "--model", cli_id,
                                "--host", "127.0.0.1", "--port", url.rsplit(":", 1)[1]],
    "model_version": model_version, "code_revision": code_revision, "weight_revision": weight_revision,
    "runtime_parameters_hash": runtime_parameters_hash,
    "stdout_log": out_log, "stderr_log": err_log,
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
target = pathlib.Path(state_path)
temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
os.replace(temporary, target)
PY
}

while IFS="$TAB" read -r component_id cli_id model_python port url model_version code_revision weight_revision state_path; do
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    stdout_log="$LOG_DIR/$component_id-$stamp.out.log"
    stderr_log="$LOG_DIR/$component_id-$stamp.err.log"
    nohup "$PYTHON" "$WORKSPACE/scripts/exec_bound_sidecar.py" \
        "$PLAN_FILE" "$component_id" >"$stdout_log" 2>"$stderr_log" </dev/null &
    sidecar_pid=$!
    write_state "$state_path" "$component_id" "$sidecar_pid" starting "$url" "$stdout_log" "$stderr_log" \
        "$model_version" "$code_revision" "$weight_revision" "$cli_id" "$model_python" "$PLAN_FILE"
    deadline=$(($(date +%s) + READY_TIMEOUT_SECONDS))
    ready=0
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if ! kill -0 "$sidecar_pid" 2>/dev/null; then break; fi
        if "$PYTHON" - "$url/health" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
        value = json.load(response)
    ok = response.status == 200 and value.get("ready") is True
except Exception:
    ok = False
raise SystemExit(0 if ok else 1)
PY
        then ready=1; break
        fi
        sleep 1
    done
    if [ "$ready" -ne 1 ]; then
        if kill -0 "$sidecar_pid" 2>/dev/null; then status=readiness_timeout; else status=exited; fi
        write_state "$state_path" "$component_id" "$sidecar_pid" "$status" "$url" "$stdout_log" "$stderr_log" \
            "$model_version" "$code_revision" "$weight_revision" "$cli_id" "$model_python" "$PLAN_FILE"
        echo "Sidecar '$component_id' did not become ready in ${READY_TIMEOUT_SECONDS}s; process was left untouched. Inspect $stderr_log" >&2
        exit 1
    fi
    write_state "$state_path" "$component_id" "$sidecar_pid" ready "$url" "$stdout_log" "$stderr_log" \
        "$model_version" "$code_revision" "$weight_revision" "$cli_id" "$model_python" "$PLAN_FILE"
    echo "[sidecar] ready $component_id pid=$sidecar_pid $url"
done < "$ROWS_FILE"

echo "Source $INSTALL_ROOT/sidecars.env.sh to configure the central Fusion Core clients."
