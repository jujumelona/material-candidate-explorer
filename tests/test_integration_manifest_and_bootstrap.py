from __future__ import annotations

import importlib.util
import hashlib
import io
import json
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from discovery_os.configured_experts import build_expert_registry_from_environment
from discovery_os.integration_manifest import load_integration_manifest


ROOT = Path(__file__).resolve().parents[1]


def _bootstrap_module():
    spec = importlib.util.spec_from_file_location(
        "discovery_bootstrap_test",
        ROOT / "scripts" / "bootstrap.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _manifest_payload() -> dict:
    return json.loads((ROOT / "integrations" / "manifest.v1.json").read_text(encoding="utf-8"))


def _refresh_revision(payload: dict) -> None:
    material = dict(payload)
    material.pop("manifest_revision", None)
    payload["manifest_revision"] = hashlib.sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _write_manifest(path: Path, payload: dict) -> Path:
    _refresh_revision(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_manifest_pins_incompatible_model_stacks_in_separate_environments() -> None:
    manifest = load_integration_manifest()
    components = {item.component_id: item for item in manifest.components}

    assert len(components) == 14
    assert components["mattergen"].install.python == "3.10"
    assert "mattersim==1.1.2" in components["mattergen"].install.constraints
    assert components["mattersim"].install.python == "3.12"
    assert components["uma"].install.version == "2.21.0"
    assert components["reinvent4"].source.revision == "80a8d21aefd9c0d3ec806377522effb30cfca12a"
    assert components["qhnet-source"].install.install_local is False
    assert components["qhnet-source"].install.python == "3.10"
    assert "torch==2.2.0" in components["qhnet-source"].install.constraints
    assert "torch-geometric==2.5.3" in components["qhnet-source"].install.constraints
    assert components["qhnet-source"].status == "research"
    assert all(len(item.source.revision) == 40 for item in manifest.components if item.source)


def test_manifest_revision_detects_tampering(tmp_path) -> None:
    bootstrap = _bootstrap_module()
    payload = _manifest_payload()
    payload["uv_version"] = "999.0.0"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(bootstrap.BootstrapError, match="revision"):
        bootstrap.load_manifest(path, allow_custom_manifest=True)


def test_default_manifest_revision_is_trusted_and_custom_manifest_requires_opt_in(
    tmp_path, monkeypatch
) -> None:
    bootstrap = _bootstrap_module()
    payload = _manifest_payload()
    assert payload["manifest_revision"] == bootstrap.TRUSTED_DEFAULT_MANIFEST_REVISION

    custom = _write_manifest(tmp_path / "custom.json", payload)
    with pytest.raises(bootstrap.BootstrapError, match="custom manifests are disabled"):
        bootstrap.load_manifest(custom)
    assert bootstrap.load_manifest(custom, allow_custom_manifest=True)["schema_version"] == "1.0"

    changed = _manifest_payload()
    changed["uv_version"] = "999.0.0"
    changed_path = _write_manifest(tmp_path / "changed-default.json", changed)
    monkeypatch.setattr(bootstrap, "DEFAULT_MANIFEST", changed_path)
    with pytest.raises(bootstrap.BootstrapError, match="trusted revision"):
        bootstrap.load_manifest(changed_path)


@pytest.mark.parametrize(
    "unsafe_id",
    [".", "..", "a/b", r"a\b", "CON", "con.txt", "prn", "nul", "com1", "lpt9"],
)
def test_manifest_rejects_unsafe_component_ids(tmp_path, unsafe_id: str) -> None:
    bootstrap = _bootstrap_module()
    payload = _manifest_payload()
    payload["components"][0]["component_id"] = unsafe_id
    for component in payload["components"]:
        component["dependencies"] = [unsafe_id if item == "core" else item for item in component["dependencies"]]
    for profile in payload["profiles"].values():
        profile["components"] = [unsafe_id if item == "core" else item for item in profile["components"]]
    path = _write_manifest(tmp_path / "unsafe.json", payload)

    with pytest.raises(bootstrap.BootstrapError, match="slug|reserved"):
        bootstrap.load_manifest(path, allow_custom_manifest=True)


def test_manifest_rejects_unsafe_weight_id(tmp_path) -> None:
    bootstrap = _bootstrap_module()
    payload = _manifest_payload()
    payload["components"][1]["weights"][0]["weight_id"] = "../../escape"
    path = _write_manifest(tmp_path / "unsafe-weight.json", payload)
    with pytest.raises(bootstrap.BootstrapError, match="slug"):
        bootstrap.load_manifest(path, allow_custom_manifest=True)


def test_manifest_rejects_pip_option_in_constraint_field(tmp_path) -> None:
    bootstrap = _bootstrap_module()
    payload = _manifest_payload()
    payload["components"][0]["install"]["constraints"] = ["--no-deps"]
    path = _write_manifest(tmp_path / "pip-option.json", payload)
    with pytest.raises(bootstrap.BootstrapError, match="package==version"):
        bootstrap.load_manifest(path, allow_custom_manifest=True)


@pytest.mark.parametrize(
    ("generated_at", "cutoff"),
    [
        ("2026-01-01T00:00:00", "2025-01-01T00:00:00Z"),
        ("2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"),
    ],
)
def test_manifest_requires_aware_ordered_timestamps(
    tmp_path, generated_at: str, cutoff: str
) -> None:
    bootstrap = _bootstrap_module()
    payload = _manifest_payload()
    payload["generated_at"] = generated_at
    payload["resolution_cutoff"] = cutoff
    path = _write_manifest(tmp_path / "bad-time.json", payload)
    with pytest.raises(bootstrap.BootstrapError, match="timezone|later"):
        bootstrap.load_manifest(path, allow_custom_manifest=True)


def test_manifest_rejects_future_generation_time(tmp_path) -> None:
    bootstrap = _bootstrap_module()
    payload = _manifest_payload()
    future = datetime.now(timezone.utc) + timedelta(days=1)
    payload["generated_at"] = future.isoformat()
    payload["resolution_cutoff"] = future.isoformat()
    path = _write_manifest(tmp_path / "future.json", payload)
    with pytest.raises(bootstrap.BootstrapError, match="future"):
        bootstrap.load_manifest(path, allow_custom_manifest=True)


def test_bootstrap_dry_run_is_non_mutating(tmp_path, monkeypatch) -> None:
    bootstrap = _bootstrap_module()
    monkeypatch.setattr(
        bootstrap.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=0, free=100 * 1024**3),
    )
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    root = ROOT / ".bootstrap-test-must-not-exist"
    if root.exists():
        pytest.fail("unexpected test directory exists before dry run")
    installer = bootstrap.Installer(
        manifest,
        root,
        accelerator="cpu",
        accepted_licenses=set(),
        include_weights=False,
        dry_run=True,
    )

    result = installer.install("core")

    assert result["dry_run"] is True
    assert result["components"][0]["component_id"] == "core"
    assert result["status"] == "ready"
    assert not root.exists()


def test_bootstrap_rejects_root_outside_workspace(tmp_path) -> None:
    bootstrap = _bootstrap_module()
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    with pytest.raises(bootstrap.BootstrapError, match="escapes"):
        bootstrap.Installer(
            manifest,
            tmp_path,
            accelerator="cpu",
            accepted_licenses=set(),
            include_weights=False,
            dry_run=True,
        )


def test_bootstrap_allows_external_root_only_with_explicit_opt_in(tmp_path) -> None:
    bootstrap = _bootstrap_module()
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    installer = bootstrap.Installer(
        manifest,
        tmp_path / "external",
        accelerator="cpu",
        accepted_licenses=set(),
        include_weights=False,
        dry_run=True,
        allow_external_root=True,
    )

    result = installer.install("core")

    assert result["dry_run"] is True
    assert not (tmp_path / "external").exists()


def test_disk_preflight_is_non_mutating_and_fail_closed(tmp_path, monkeypatch) -> None:
    bootstrap = _bootstrap_module()
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    root = tmp_path / "external"
    monkeypatch.setattr(
        bootstrap.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=10, used=10, free=0),
    )
    installer = bootstrap.Installer(
        manifest,
        root,
        accelerator="cpu",
        accepted_licenses=set(),
        include_weights=False,
        dry_run=True,
        allow_external_root=True,
    )

    result = installer.install("core")

    assert result["status"] == "partial"
    assert result["disk_preflight"]["ok"] is False
    assert any(item["status"] == "insufficient_disk_space" for item in result["unresolved"])
    assert not root.exists()


def test_disk_preflight_source_api_requires_source_and_environment(
    tmp_path, monkeypatch
) -> None:
    bootstrap = _bootstrap_module()
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    monkeypatch.setattr(bootstrap, "_host_platform", lambda: "linux")
    monkeypatch.setattr(
        bootstrap.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(total=100 * 1024**3, used=0, free=100 * 1024**3),
    )
    plan = bootstrap.build_plan(
        manifest,
        "electronic-open",
        accelerator="cuda",
        accepted_licenses=set(),
    )
    qhnet_row = next(
        item for item in plan["components"] if item["component_id"] == "qhnet-source"
    )
    qhnet_plan = {"components": [qhnet_row]}
    root = tmp_path / "external"
    installer = bootstrap.Installer(
        manifest,
        root,
        accelerator="cuda",
        accepted_licenses=set(),
        include_weights=False,
        dry_run=True,
        allow_external_root=True,
    )
    source_marker = root / "sources" / "qhnet-source" / ".discovery-source.json"
    source_marker.parent.mkdir(parents=True)
    source_marker.write_text("{}", encoding="utf-8")

    source_only = installer._disk_preflight(qhnet_plan)
    assert source_only["required_gb"] == qhnet_row["storage_gb"]

    environment_python = bootstrap._environment_python(root / "envs" / "qhnet-source")
    environment_python.parent.mkdir(parents=True)
    environment_python.touch()
    source_marker.unlink()
    environment_only = installer._disk_preflight(qhnet_plan)
    assert environment_only["required_gb"] == qhnet_row["storage_gb"]

    source_marker.write_text("{}", encoding="utf-8")
    complete = installer._disk_preflight(qhnet_plan)
    assert complete["required_gb"] == 0.0


def test_license_acceptance_is_explicit(monkeypatch) -> None:
    bootstrap = _bootstrap_module()
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    monkeypatch.setattr(bootstrap, "_host_platform", lambda: "linux")
    monkeypatch.delenv("ACCEPT_ESM_LICENSE", raising=False)

    blocked = bootstrap.build_plan(
        manifest,
        "biology-open",
        accelerator="cpu",
        accepted_licenses=set(),
    )
    accepted = bootstrap.build_plan(
        manifest,
        "biology-open",
        accelerator="cpu",
        accepted_licenses={"esm"},
    )

    blocked_esm = next(item for item in blocked["components"] if item["component_id"] == "esm")
    accepted_esm = next(item for item in accepted["components"] if item["component_id"] == "esm")
    assert blocked_esm["action"] == "license_required"
    assert accepted_esm["action"] == "install"


def test_cuda_plan_falls_back_to_cpu_per_component(monkeypatch) -> None:
    bootstrap = _bootstrap_module()
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    monkeypatch.setattr(bootstrap, "_host_platform", lambda: "linux")

    plan = bootstrap.build_plan(
        manifest,
        "electronic-open",
        accelerator="cuda",
        accepted_licenses=set(),
    )
    rows = {item["component_id"]: item for item in plan["components"]}

    assert rows["pyscf"]["action"] == "install"
    assert rows["pyscf"]["accelerator"] == "cpu"
    assert rows["pyscf"]["accelerator_fallback_from"] == "cuda"
    assert rows["qhnet-source"]["action"] == "install"
    assert rows["qhnet-source"]["accelerator"] == "cuda"
    assert rows["qhnet-source"]["environment"] == "envs/qhnet-source"


def test_include_weights_marks_manual_and_gated_items_unresolved(monkeypatch) -> None:
    bootstrap = _bootstrap_module()
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    monkeypatch.setattr(bootstrap, "_host_platform", lambda: "linux")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    manual = bootstrap.build_plan(
        manifest,
        "molecule-generation",
        accelerator="cuda",
        accepted_licenses=set(),
        include_weights=True,
    )
    gated = bootstrap.build_plan(
        manifest,
        "uma",
        accelerator="cuda",
        accepted_licenses={"uma"},
        include_weights=True,
    )

    assert manual["status"] == "partial"
    assert any(item["status"] == "manual_download_required" for item in manual["unresolved"])
    assert gated["status"] == "partial"
    assert any(item["status"] == "credential_required" for item in gated["unresolved"])


def test_require_all_applies_to_plan_and_dry_run(monkeypatch, capsys) -> None:
    bootstrap = _bootstrap_module()
    monkeypatch.setattr(bootstrap, "_host_platform", lambda: "windows")

    plan_exit = bootstrap.main(["plan", "--profile", "all-open", "--require-all"])
    dry_run_exit = bootstrap.main(
        ["install", "--profile", "all-open", "--dry-run", "--require-all"]
    )

    assert plan_exit == 2
    assert dry_run_exit == 2
    capsys.readouterr()


def test_nested_weight_statuses_make_real_install_state_partial() -> None:
    bootstrap = _bootstrap_module()
    states = {
        "reinvent4": {
            "status": "installed",
            "weights": {"prior": {"status": "manual_download_required"}},
        }
    }
    unresolved = bootstrap._installation_unresolved(states, include_weights=True)
    assert unresolved == [
        {
            "component_id": "reinvent4",
            "item": "prior",
            "status": "manual_download_required",
        }
    ]


def test_cuda_specific_indexes_are_not_used_for_cpu_fallback() -> None:
    bootstrap = _bootstrap_module()
    assert bootstrap._index_url_applies("https://download.pytorch.org/whl/cu126", "cuda")
    assert not bootstrap._index_url_applies("https://download.pytorch.org/whl/cu126", "cpu")


def test_bootstrap_subprocess_environment_does_not_leak_unrelated_secrets(
    tmp_path, monkeypatch
) -> None:
    bootstrap = _bootstrap_module()
    manifest = bootstrap.load_manifest(ROOT / "integrations" / "manifest.v1.json")
    monkeypatch.setenv("UNRELATED_API_TOKEN", "do-not-forward")
    monkeypatch.setenv("DATABASE_PASSWORD", "do-not-forward")
    monkeypatch.setenv("HF_TOKEN", "requested-weight-token")
    monkeypatch.setenv("NORMAL_SETTING", "safe")
    installer = bootstrap.Installer(
        manifest,
        tmp_path / "install",
        accelerator="cpu",
        accepted_licenses=set(),
        include_weights=False,
        dry_run=True,
        allow_external_root=True,
    )

    ordinary = installer._uv_env()
    weight = installer._weight_download_env("HF_TOKEN")

    assert ordinary["NORMAL_SETTING"] == "safe"
    assert "UNRELATED_API_TOKEN" not in ordinary
    assert "DATABASE_PASSWORD" not in ordinary
    assert "HF_TOKEN" not in ordinary
    assert weight["HF_TOKEN"] == "requested-weight-token"
    assert "UNRELATED_API_TOKEN" not in weight


def test_download_stops_at_declared_size_and_removes_partial(tmp_path, monkeypatch) -> None:
    bootstrap = _bootstrap_module()

    class Response(io.BytesIO):
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        bootstrap.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: Response(b"12345"),
    )
    destination = tmp_path / "archive.tar.gz"

    with pytest.raises(bootstrap.BootstrapError, match="byte limit"):
        bootstrap._download_verified(
            "https://example.invalid/archive.tar.gz",
            destination,
            expected_sha256="0" * 64,
            expected_size=4,
        )

    assert not destination.exists()
    assert not destination.with_suffix(".gz.part").exists()


def test_archive_member_and_expansion_limits_cleanup_temporary_directory(
    tmp_path, monkeypatch
) -> None:
    bootstrap = _bootstrap_module()
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for name, data in (("root/a.txt", b"aaaa"), ("root/b.txt", b"bbbb")):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            handle.addfile(member, io.BytesIO(data))

    sources = tmp_path / "sources"
    sources.mkdir()
    destination = sources / "component"
    monkeypatch.setattr(bootstrap, "MAX_ARCHIVE_MEMBERS", 1)

    with pytest.raises(bootstrap.BootstrapError, match="members"):
        bootstrap._extract_archive(archive, destination, sources)

    assert not destination.exists()
    assert list(sources.iterdir()) == []


def test_archive_uncompressed_size_limit_is_enforced(tmp_path, monkeypatch) -> None:
    bootstrap = _bootstrap_module()
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("root/data.bin")
        member.size = 8
        handle.addfile(member, io.BytesIO(b"12345678"))

    sources = tmp_path / "sources"
    sources.mkdir()
    monkeypatch.setattr(bootstrap, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", 7)

    with pytest.raises(bootstrap.BootstrapError, match="expands"):
        bootstrap._extract_archive(archive, sources / "component", sources)

    assert list(sources.iterdir()) == []


def test_archive_source_marker_binds_extracted_byte_inventory(tmp_path, monkeypatch) -> None:
    bootstrap = _bootstrap_module()
    archive_bytes_path = tmp_path / "fixture.tar.gz"
    with tarfile.open(archive_bytes_path, "w:gz") as handle:
        data = b"reviewed source bytes"
        # GitHub archives have one repository root that bootstrap strips;
        # archive_subdirectory is resolved below that stripped root.
        member = tarfile.TarInfo("archive-root/root/module.py")
        member.size = len(data)
        handle.addfile(member, io.BytesIO(data))
    archive_bytes = archive_bytes_path.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    component = {
        "component_id": "fixture-source",
        "source": {"revision": "a" * 40},
        "install": {
            "archive_url": "https://example.invalid/fixture.tar.gz",
            "archive_sha256": archive_sha256,
            "archive_size_bytes": len(archive_bytes),
            "archive_subdirectory": "root",
        },
    }
    root = tmp_path / "install"
    (root / "sources").mkdir(parents=True)
    installer = SimpleNamespace(root=root)

    def fake_download(_url, destination, *, expected_sha256, expected_size):
        assert expected_sha256 == archive_sha256
        assert expected_size == len(archive_bytes)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive_bytes)

    monkeypatch.setattr(bootstrap, "_download_verified", fake_download)
    extracted = bootstrap.Installer._install_archive(installer, component)
    marker = json.loads(
        (root / "sources" / "fixture-source" / ".discovery-source.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(marker["inventory_sha256"]) == 64
    (extracted / "module.py").write_bytes(b"tampered")

    with pytest.raises(bootstrap.BootstrapError, match="byte inventory"):
        bootstrap.Installer._install_archive(installer, component)


def test_configured_expert_catalog_is_unavailable_until_url_is_set() -> None:
    registry = build_expert_registry_from_environment(environ={}, include_unconfigured=True)
    descriptors = {item.expert_id: item for item in registry.describe()}

    assert len(descriptors) == 11
    assert descriptors["unimol"].available is False
    assert descriptors["rnafm"].supported_representations == ["rna_sequence", "fasta"]

    configured = build_expert_registry_from_environment(
        environ={"UNIMOL_API_URL": "http://localhost:8102"},
        include_unconfigured=True,
    )
    assert next(item for item in configured.describe() if item.expert_id == "unimol").available
