from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from discovery_os.sidecars import cli as sidecar_cli
from discovery_os.sidecars.cli import main as sidecar_main
from discovery_os.sidecars.cli import preflight_configuration
from discovery_os.sidecars.weight_binding import directory_inventory_sha256


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL_LAUNCHER = ROOT / "start-sidecars.ps1"
POSIX_LAUNCHER = ROOT / "start-sidecars.sh"
POWERSHELL_BOOTSTRAP = ROOT / "bootstrap.ps1"
MANIFEST = ROOT / "integrations" / "manifest.v1.json"


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _fake_component_python(install_root: Path, component_id: str, *, windows: bool) -> Path:
    relative = (
        Path("envs") / component_id / "Scripts" / "python.exe"
        if windows
        else Path("envs") / component_id / "bin" / "python"
    )
    target = install_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"")
    if not windows:
        target.chmod(0o700)
    return target


def _fake_scgpt_bundle(root: Path) -> Path:
    bundle = root / "scgpt-bundle"
    bundle.mkdir(parents=True)
    (bundle / "args.json").write_text('{"fixture":true}', encoding="utf-8")
    (bundle / "vocab.json").write_text('{"<pad>":0}', encoding="utf-8")
    (bundle / "best_model.pt").write_bytes(b"scgpt-fixture")
    (bundle / ".cache").mkdir()
    (bundle / ".cache" / "runtime.tmp").write_bytes(b"excluded-runtime-cache")
    return bundle


def _run_powershell(*arguments: str, environ: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if executable is None:
        pytest.skip("Windows PowerShell is unavailable")
    return subprocess.run(
        [
            executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(POWERSHELL_LAUNCHER),
            *arguments,
        ],
        cwd=ROOT,
        env=environ,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )


def test_launchers_use_the_real_sidecar_cli_and_safe_process_primitives() -> None:
    ps = POWERSHELL_LAUNCHER.read_text(encoding="utf-8")
    sh = POSIX_LAUNCHER.read_text(encoding="utf-8")
    bound_exec = (ROOT / "scripts" / "exec_bound_sidecar.py").read_text(encoding="utf-8")

    for source in (ps, sh):
        assert "-m" in source
        assert "discovery_os.sidecars" in source
        assert "--model" in source
        assert "--host" in source
        assert "127.0.0.1" in source
        assert "--port" in source
        assert "/health" in source
        assert "sidecars.env.ps1" in source
        assert "sidecars.env.sh" in source

    assert "Start-Process" in ps
    assert "-WindowStyle Hidden" in ps
    assert "Stop-Process" not in ps
    assert "taskkill" not in ps.lower()
    assert "nohup" in sh
    assert 'kill -0 "$sidecar_pid"' in sh
    assert 'kill "$sidecar_pid"' not in sh
    for name in ("SIDECAR_MODEL_VERSION", "SIDECAR_CODE_REVISION", "SIDECAR_WEIGHT_REVISION"):
        assert name in ps
        assert name in bound_exec
    for source in (ps, sh):
        assert "SCGPT_CHECKPOINT_DIR" in source
        assert "SCGPT_BUNDLE_INVENTORY_SHA256" in source
        assert "SCGPT_MAX_LENGTH" in source
        assert "QHNET_CHECKPOINT_PATH" in source
        assert "QHNET_CONFIG_PATH" in source
        assert "QHNET_SOURCE_PATH" in source
        assert "bundle-sha256:" in source


@pytest.mark.skipif(os.name != "nt", reason="PowerShell native layout is Windows-specific")
def test_powershell_scgpt_manual_bundle_uses_content_inventory_revision(tmp_path: Path) -> None:
    install_root = tmp_path / "sidecars"
    _fake_component_python(install_root, "scgpt", windows=True)
    bundle = _fake_scgpt_bundle(tmp_path)
    environ = dict(os.environ)
    environ["SCGPT_CHECKPOINT_DIR"] = str(bundle)
    environ.pop("SCGPT_WEIGHT_REVISION", None)
    result = _run_powershell(
        "-Backend",
        "native",
        "-Component",
        "scgpt",
        "-InstallRoot",
        str(install_root),
        "-AllowExternalRoot",
        "-DryRun",
        environ=environ,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    sidecar = payload["sidecars"][0]
    expected = f"sha256:{directory_inventory_sha256(bundle)}"
    assert sidecar["weight_revision"] == expected
    assert sidecar["runtime_environment"]["SCGPT_CHECKPOINT_DIR"] == str(bundle.resolve())
    assert sidecar["runtime_environment"]["SCGPT_BUNDLE_INVENTORY_SHA256"] == expected.removeprefix(
        "sha256:"
    )


def test_readiness_is_bounded_and_unsupported_health_never_counts_as_ready() -> None:
    ps = POWERSHELL_LAUNCHER.read_text(encoding="utf-8")
    sh = POSIX_LAUNCHER.read_text(encoding="utf-8")

    assert "$health.ready -eq $true" in ps
    assert "AddSeconds($ReadyTimeoutSeconds)" in ps
    assert '"readiness_timeout"' in ps
    assert 'value.get("ready") is True' in sh
    assert "READY_TIMEOUT_SECONDS" in sh
    assert "status=readiness_timeout" in sh
    # A 200 response reporting unsupported/ready=false must time out; neither
    # launcher accepts a status string or HTTP success alone.
    assert 'health.status -eq "ready"' not in ps
    assert 'value.get("status")' not in sh


def test_configuration_preflight_rejects_missing_qhnet_bundle() -> None:
    with pytest.raises(ValueError, match="QHNET_CHECKPOINT_PATH is required"):
        preflight_configuration(
            "qhnet",
            {},
            host="127.0.0.1",
            port=8107,
        )


def test_configuration_preflight_is_static_and_keeps_checkpoint_lazy(monkeypatch) -> None:
    monkeypatch.setattr(
        "discovery_os.sidecars.cli._module_available",
        lambda _module_name: True,
    )
    report = preflight_configuration(
        "chgnet",
        {"SIDECAR_WEIGHT_REVISION": "managed-unattested:chgnet-0.3.0-fixture"},
        host="127.0.0.1",
        port=8113,
    )
    assert report["supported"] is True
    assert report["configuration_only"] is True
    assert report["checkpoint_loaded"] is False
    assert report["model_id"] == "chgnet"
    assert report["port"] == 8113


def test_configuration_preflight_rejects_missing_model_package(monkeypatch) -> None:
    monkeypatch.setattr(
        "discovery_os.sidecars.cli._module_available",
        lambda module_name: module_name != "chgnet",
    )
    with pytest.raises(ValueError, match="model dependency 'chgnet' is not installed"):
        preflight_configuration(
            "chgnet",
            {"SIDECAR_WEIGHT_REVISION": "managed-unattested:chgnet-0.3.0-fixture"},
            host="127.0.0.1",
            port=8113,
        )


def test_cli_preflight_emits_machine_readable_success_without_loading(monkeypatch, capsys) -> None:
    monkeypatch.setenv(
        "SIDECAR_WEIGHT_REVISION", "managed-unattested:chgnet-0.3.0-fixture"
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.cli._module_available",
        lambda _module_name: True,
    )
    exit_code = sidecar_main(
        ["--model", "chgnet", "--host", "127.0.0.1", "--port", "8113", "--preflight"]
    )
    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["supported"] is True
    assert payload["configuration_only"] is True
    assert payload["checkpoint_loaded"] is False


def test_configuration_preflight_rejects_missing_required_model_config() -> None:
    with pytest.raises(ValueError, match="CHEMPROP_CHECKPOINT_PATH is required"):
        preflight_configuration(
            "chemprop",
            {"SIDECAR_WEIGHT_REVISION": "sha256:chemprop-fixture"},
            host="127.0.0.1",
            port=8111,
        )


def test_chemprop_runtime_requires_property_names_and_units(tmp_path: Path) -> None:
    checkpoint = tmp_path / "task.ckpt"
    checkpoint.write_bytes(b"fixture")
    with pytest.raises(ValueError, match="CHEMPROP_PROPERTY_NAMES is required"):
        sidecar_cli._runtime(
            "chemprop",
            {"CHEMPROP_CHECKPOINT_PATH": str(checkpoint)},
        )
    with pytest.raises(ValueError, match="CHEMPROP_PROPERTY_UNITS is required"):
        sidecar_cli._runtime(
            "chemprop",
            {
                "CHEMPROP_CHECKPOINT_PATH": str(checkpoint),
                "CHEMPROP_PROPERTY_NAMES": "solubility",
            },
        )


def test_reinvent_runtime_prefers_entrypoint_beside_isolated_python(
    tmp_path: Path, monkeypatch
) -> None:
    bin_directory = tmp_path / ("Scripts" if os.name == "nt" else "bin")
    bin_directory.mkdir()
    python = bin_directory / ("python.exe" if os.name == "nt" else "python")
    python.write_bytes(b"")
    executable = bin_directory / ("reinvent.exe" if os.name == "nt" else "reinvent")
    executable.write_bytes(b"fixture")
    executable.chmod(0o700)
    model_file = tmp_path / "prior.model"
    model_file.write_bytes(b"fixture")
    path_fallback = tmp_path / "fallback" / "reinvent"

    monkeypatch.setattr(sidecar_cli.sys, "executable", str(python))
    monkeypatch.setattr(sidecar_cli.shutil, "which", lambda _name: str(path_fallback))

    runtime = sidecar_cli._runtime(
        "reinvent4",
        {"REINVENT_MODEL_FILE": str(model_file)},
    )
    assert runtime.executable == str(executable.resolve())
    assert runtime.executable != str(path_fallback)


def test_launchers_preflight_every_plan_before_the_first_server_start() -> None:
    ps = POWERSHELL_LAUNCHER.read_text(encoding="utf-8")
    sh = POSIX_LAUNCHER.read_text(encoding="utf-8")

    assert "--preflight" in ps
    assert "--preflight" in sh
    assert ps.index("--preflight") < ps.index("$process = Start-Process")
    assert sh.index("--preflight") < sh.index('nohup "$PYTHON"')
    assert "before any server was started" in ps
    assert "before any server was started" in sh
    assert "CHEMPROP_PROPERTY_UNITS is required" in ps
    assert "CHEMPROP_PROPERTY_UNITS is required" in sh


def test_manifest_keeps_chgnet_and_pyscf_on_the_required_ports() -> None:
    components = {item["component_id"]: item for item in _manifest()["components"]}
    assert components["chgnet"]["api"] == {
        "schema_version": "1.0",
        "protocol": "expert-feature-v1",
        "base_url_env": "CHGNET_API_URL",
        "default_port": 8113,
    }
    assert components["pyscf"]["api"] == {
        "schema_version": "1.0",
        "protocol": "expert-feature-v1",
        "base_url_env": "PYSCF_API_URL",
        "default_port": 8108,
    }


@pytest.mark.skipif(os.name != "nt", reason="PowerShell native layout is Windows-specific")
def test_powershell_dry_run_builds_a_pinned_command_and_secret_free_env(tmp_path: Path) -> None:
    install_root = tmp_path / "sidecars"
    python = _fake_component_python(install_root, "unimol", windows=True)
    snapshot = (
        install_root
        / "models"
        / "unimol"
        / "unimol-models"
        / "9f19c45c718192888a1c8a1c905f69f0755ea502"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "mol_pre_all_h_220816.pt").write_bytes(b"checkpoint")
    (snapshot / "mol.dict.txt").write_text("[PAD]\n", encoding="utf-8")
    (snapshot / ".snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "repository": "dptech/Uni-Mol-Models",
                "revision": "9f19c45c718192888a1c8a1c905f69f0755ea502",
                "inventory_sha256": directory_inventory_sha256(snapshot),
            }
        ),
        encoding="utf-8",
    )
    result = _run_powershell(
        "-Backend",
        "native",
        "-Component",
        "unimol",
        "-InstallRoot",
        str(install_root),
        "-AllowExternalRoot",
        "-DryRun",
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    sidecar = plan["sidecars"][0]
    assert sidecar["command"] == [
        str(python.resolve()),
        "-m",
        "discovery_os.sidecars",
        "--model",
        "unimol",
        "--host",
        "127.0.0.1",
        "--port",
        "8102",
    ]
    assert sidecar["model_version"] == "0.1.6"
    assert sidecar["code_revision"] == "4596596aa8f73eb462d5cc5a921d79966d0465da"
    assert sidecar["weight_revision"] == "9f19c45c718192888a1c8a1c905f69f0755ea502"

    ps_env = (install_root / "sidecars.env.ps1").read_text(encoding="utf-8")
    sh_env = (install_root / "sidecars.env.sh").read_text(encoding="utf-8")
    for content in (ps_env, sh_env):
        assert "UNIMOL_API_URL" in content
        assert "UNIMOL_WEIGHT_REVISION" in content
        assert "http://127.0.0.1:8102" in content
        assert "TOKEN" not in content.upper()
        assert "SECRET" not in content.upper()
        assert "PASSWORD" not in content.upper()


@pytest.mark.skipif(os.name != "nt", reason="PowerShell native layout is Windows-specific")
def test_powershell_managed_weight_revision_is_fail_closed(tmp_path: Path) -> None:
    install_root = tmp_path / "sidecars"
    _fake_component_python(install_root, "chgnet", windows=True)
    environ = dict(os.environ)
    environ.pop("CHGNET_WEIGHT_REVISION", None)
    blocked = _run_powershell(
        "-Backend",
        "native",
        "-Component",
        "chgnet",
        "-InstallRoot",
        str(install_root),
        "-AllowExternalRoot",
        "-DryRun",
        environ=environ,
    )
    assert blocked.returncode != 0
    assert "CHGNET_WEIGHT_REVISION is required" in (blocked.stdout + blocked.stderr)

    environ["CHGNET_WEIGHT_REVISION"] = "managed-unattested:chgnet-0.3.0-fixture"
    accepted = _run_powershell(
        "-Backend",
        "native",
        "-Component",
        "chgnet",
        "-InstallRoot",
        str(install_root),
        "-AllowExternalRoot",
        "-DryRun",
        environ=environ,
    )
    assert accepted.returncode == 0, accepted.stderr
    sidecar = json.loads(accepted.stdout)["sidecars"][0]
    assert sidecar["port"] == 8113
    assert sidecar["weight_revision"] == "managed-unattested:chgnet-0.3.0-fixture"


@pytest.mark.skipif(os.name != "nt", reason="PowerShell native layout is Windows-specific")
def test_powershell_rejects_external_install_root_without_opt_in(tmp_path: Path) -> None:
    install_root = tmp_path / "external"
    _fake_component_python(install_root, "unimol", windows=True)
    result = _run_powershell(
        "-Backend",
        "native",
        "-Component",
        "unimol",
        "-InstallRoot",
        str(install_root),
        "-DryRun",
    )
    assert result.returncode != 0
    assert "InstallRoot is outside the workspace" in (result.stdout + result.stderr)


def test_powershell_has_explicit_wsl_delegation_for_linux_profiles() -> None:
    source = POWERSHELL_LAUNCHER.read_text(encoding="utf-8")
    assert "$requiresLinux" in source
    assert "@($entry[0].platforms)" in source
    assert "wsl.exe" in source
    assert "wslpath -a" in source
    assert "start-sidecars.sh" in source
    assert "Component '$id' does not support native Windows; use -Backend wsl." in source


def test_wsl_bridges_only_reviewed_bootstrap_credentials_without_persisting_them() -> None:
    source = POWERSHELL_BOOTSTRAP.read_text(encoding="utf-8")
    assert 'foreach ($name in @("HF_TOKEN", "ACCEPT_ESM_LICENSE", "ACCEPT_UMA_LICENSE"))' in source
    assert "$savedWslenv = $env:WSLENV" in source
    assert "finally" in source
    assert "Remove-Item Env:WSLENV" in source
    # The credential value is inherited by one WSL process. It is not appended
    # to bootstrap argv or any generated endpoint/state file.
    assert '$arguments += @("--token"' not in source
    assert "sidecars.env" not in source


def test_boltz_runtime_controls_are_forwarded_by_both_launchers() -> None:
    ps = POWERSHELL_LAUNCHER.read_text(encoding="utf-8")
    sh = POSIX_LAUNCHER.read_text(encoding="utf-8")
    for name in (
        "BOLTZ_CACHE",
        "BOLTZ_PROCESS_TIMEOUT_SECONDS",
        "BOLTZ_MAX_JSON_BYTES",
        "BOLTZ_MAX_CIF_BYTES",
        "BOLTZ_MAX_SEQUENCE_LENGTH",
        "BOLTZ_MAX_SMILES_LENGTH",
        "BOLTZ_NO_KERNELS",
    ):
        assert name in ps
        assert name in sh
    assert '"QHNET_CONFIG_PATH", "BOLTZ_CACHE"' in ps


@pytest.mark.skipif(os.name == "nt", reason="POSIX execution is validated on POSIX hosts")
def test_posix_dry_run_uses_component_environment_and_no_external_weight(tmp_path: Path) -> None:
    install_root = tmp_path / "sidecars"
    python = _fake_component_python(install_root, "pyscf", windows=False)
    result = subprocess.run(
        [
            "/bin/sh",
            str(POSIX_LAUNCHER),
            "--component",
            "pyscf",
            "--install-root",
            str(install_root),
            "--allow-external-root",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    sidecar = json.loads(result.stdout)["sidecars"][0]
    assert sidecar["command"] == [
        str(python.resolve()),
        "-m",
        "discovery_os.sidecars",
        "--model",
        "pyscf",
        "--host",
        "127.0.0.1",
        "--port",
        "8108",
    ]
    assert sidecar["weight_revision"] == "no-external-weight"


def test_qhnet_manifest_id_maps_only_the_cli_alias() -> None:
    for source in (
        POWERSHELL_LAUNCHER.read_text(encoding="utf-8"),
        POSIX_LAUNCHER.read_text(encoding="utf-8"),
    ):
        assert "qhnet-source" in source
        assert "qhnet" in source
