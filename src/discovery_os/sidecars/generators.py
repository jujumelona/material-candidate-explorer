"""Production-oriented wrappers for the official MatterGen/REINVENT CLIs."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from discovery_os.fusion_schemas import FusionGenerationRequest
from discovery_os.schemas import CandidateRepresentation, RepresentationKind

from .base import LazyModelAdapter, require_module
from .conversions import (
    ase_chemical_system,
    candidate_smiles,
    candidate_to_ase,
    pymatgen_to_cif,
)
from .errors import ModelExecutionError, ModelOutputError, ModelTimeoutError, OptionalDependencyError
from .types import GeneratedBatch, GeneratedCandidateData
from .weight_binding import directory_inventory_sha256, sha256_file


class MatterGenGenerator(LazyModelAdapter[Any]):
    """MatterGen through its public ``CrystalGenerator`` Python entrypoint.

    The checkpoint is downloaded/resolved and loaded once, on the first HTTP
    request.  Later requests reuse the prepared diffusion model while updating
    only bounded conditioning and batch controls.
    """

    _CONDITION_NAMES = frozenset(
        {
            "chemical_system",
            "space_group",
            "dft_mag_density",
            "dft_band_gap",
            "ml_bulk_modulus",
            "hhi_score",
            "energy_above_hull",
        }
    )

    def __init__(
        self,
        *,
        pretrained_name: str = "mattergen_base",
        checkpoint_path: str | None = None,
        objective_map: dict[str, str] | None = None,
        guidance_max: float = 4.0,
        max_cif_bytes: int = 20_000,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        if not pretrained_name.strip():
            raise ValueError("pretrained_name must not be blank")
        if not 0.0 <= guidance_max <= 100.0:
            raise ValueError("guidance_max must be between 0 and 100")
        self.pretrained_name = pretrained_name
        self.checkpoint_path = checkpoint_path
        self.objective_map = dict(objective_map or {})
        self.guidance_max = guidance_max
        self.max_cif_bytes = max_cif_bytes
        self._inference_lock = threading.Lock()
        self.checkpoint_inventory_sha256 = (
            directory_inventory_sha256(checkpoint_path) if checkpoint_path else None
        )

    def _load_model(self, device: str) -> Any:
        generator_module = require_module(
            "mattergen.generator",
            install_hint="install the pinned MatterGen release in this isolated sidecar",
        )
        data_classes = require_module(
            "mattergen.common.utils.data_classes",
            install_hint="install the complete pinned MatterGen release and checkpoint metadata",
        )
        if self.checkpoint_path is not None:
            path = Path(self.checkpoint_path).expanduser().resolve(strict=True)
            if not path.exists():
                raise ModelExecutionError("configured MatterGen checkpoint path does not exist")
            if (
                self.checkpoint_inventory_sha256 is None
                or directory_inventory_sha256(path) != self.checkpoint_inventory_sha256
            ):
                raise ModelExecutionError(
                    "MatterGen checkpoint bytes changed after runtime attestation"
                )
            self.checkpoint_path = str(path)
        try:
            if self.checkpoint_path is not None:
                checkpoint = data_classes.MatterGenCheckpointInfo(
                    model_path=Path(self.checkpoint_path).resolve(),
                    load_epoch="last",
                    config_overrides=[],
                    strict_checkpoint_loading=True,
                )
            else:
                raise ModelExecutionError(
                    "MatterGen requires MATTERGEN_CHECKPOINT_PATH from a verified local snapshot; "
                    "Hugging Face download fallback is disabled"
                )
            generator = generator_module.CrystalGenerator(
                checkpoint_info=checkpoint,
                properties_to_condition_on={},
                batch_size=1,
                num_batches=1,
                record_trajectories=False,
                diffusion_guidance_factor=0.0,
            )
            generator.prepare()
            generator.model.to(device)
            return generator
        except Exception as exc:
            raise ModelExecutionError(
                f"MatterGen checkpoint could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc

    def generate(self, request: FusionGenerationRequest) -> GeneratedBatch:
        generator = self._ensure_loaded()
        count = request.run_config.candidate_count
        controls = request.run_config.generation_controls
        conditions, condition_warnings = self._conditions(request)
        warnings = [
            "MatterGen v1 has no parent-structure mutation operator; the parent is lineage "
            "and may only contribute an explicit chemical-system condition.",
            "MatterGen v1 does not expose temperature, mutation_strength, or "
            "diversity_strength; those controls were preserved in provenance but not applied.",
            *condition_warnings,
        ]
        with tempfile.TemporaryDirectory(prefix="discovery-mattergen-") as temporary:
            root = Path(temporary)
            try:
                with self._inference_lock:
                    _seed_mattergen(request.run_config.effective_generator_seed)
                    generator.properties_to_condition_on = conditions
                    generator.diffusion_guidance_factor = (
                        round(controls.alpha * self.guidance_max, 8) if conditions else 0.0
                    )
                    structures = generator.generate(
                        batch_size=count,
                        num_batches=1,
                        output_dir=str(root / "output"),
                    )
            except Exception as exc:
                raise ModelExecutionError(
                    f"MatterGen generation failed: {type(exc).__name__}: {exc}"
                ) from exc
        if len(structures) != count:
            raise ModelOutputError(
                f"MatterGen returned {len(structures)} structures, expected {count}"
            )
        cifs = [
            (f"generated-{index}.cif", pymatgen_to_cif(structure, max_bytes=self.max_cif_bytes))
            for index, structure in enumerate(structures)
        ]
        candidates = tuple(
            GeneratedCandidateData(
                name=f"MatterGen candidate {index + 1}",
                representations=(
                    CandidateRepresentation(
                        kind=RepresentationKind.CIF,
                        value=cif,
                        media_type="chemical/x-cif",
                        format_version="CIF",
                        canonical=False,
                        metadata={"source_entry": name},
                    ),
                ),
                attributes={
                    "mattergen_pretrained_name": self.pretrained_name,
                    "conditions": conditions,
                    "generation_controls": controls.model_dump(mode="json"),
                },
                provenance={"adapter": "mattergen-crystal-generator-v1"},
            )
            for index, (name, cif) in enumerate(cifs)
        )
        return GeneratedBatch(candidates=candidates, warnings=tuple(warnings))

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "pretrained_name": self.pretrained_name,
            "checkpoint_inventory_sha256": self.checkpoint_inventory_sha256,
            "objective_map": dict(sorted(self.objective_map.items())),
            "guidance_max": self.guidance_max,
            "max_cif_bytes": self.max_cif_bytes,
            "requested_device": self._requested_device,
        }

    def _conditions(
        self, request: FusionGenerationRequest
    ) -> tuple[dict[str, str | float | int], list[str]]:
        conditions: dict[str, str | float | int] = {}
        warnings: list[str] = []
        for objective in request.goal.objectives:
            mapped = self.objective_map.get(objective.property_name, objective.property_name)
            if mapped not in self._CONDITION_NAMES or objective.target_value is None:
                continue
            conditions[mapped] = _condition_value(mapped, objective.target_value)
        proposal = request.revision_proposal
        if proposal is not None:
            warnings.append(
                "MatterGen does not accept the raw unified latent; only explicit supported "
                "revision desired_changes with concrete target values were translated."
            )
            for change in proposal.desired_changes:
                if change.property_name is None:
                    continue
                mapped = self.objective_map.get(change.property_name, change.property_name)
                if change.target_value is None:
                    if mapped in self._CONDITION_NAMES:
                        warnings.append(
                            f"revision change {change.property_name!r} requested {change.direction} "
                            "without a target value; MatterGen target was not invented"
                        )
                    continue
                if mapped not in self._CONDITION_NAMES:
                    warnings.append(
                        f"revision property {change.property_name!r} is not a supported MatterGen "
                        "condition and was not applied"
                    )
                    continue
                value = _condition_value(mapped, change.target_value)
                if mapped in conditions and conditions[mapped] != value:
                    warnings.append(
                        f"revision target for {mapped!r} overrides the original goal target "
                        "for this iteration"
                    )
                conditions[mapped] = value
        if self.pretrained_name in {"chemical_system", "chemical_system_energy_above_hull"}:
            if "chemical_system" not in conditions:
                parent_atoms = candidate_to_ase(request.parent_candidate)
                conditions["chemical_system"] = ase_chemical_system(parent_atoms)
        return conditions, warnings


class ReinventGenerator(LazyModelAdapter[str]):
    """REINVENT4 sampling through its documented JSON/TOML CLI contract."""

    def __init__(
        self,
        *,
        model_file: str,
        mode: Literal["reinvent", "mol2mol"] = "reinvent",
        executable: str = "reinvent",
        process_timeout_seconds: float = 1_800.0,
        oversample_factor: int = 2,
        max_output_bytes: int = 64 * 1024 * 1024,
        pass_environment: tuple[str, ...] = (),
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        if mode not in {"reinvent", "mol2mol"}:
            raise ValueError("REINVENT mode must be reinvent or mol2mol")
        if not 1 <= oversample_factor <= 20:
            raise ValueError("oversample_factor must be between 1 and 20")
        self.model_file = model_file
        self.mode = mode
        self.executable = executable
        executable_path = Path(executable).expanduser()
        self.executable_sha256 = (
            sha256_file(executable_path)
            if executable_path.exists() and executable_path.is_file()
            else None
        )
        self.process_timeout_seconds = process_timeout_seconds
        self.oversample_factor = oversample_factor
        self.max_output_bytes = max_output_bytes
        self.pass_environment = pass_environment
        self.model_sha256 = sha256_file(model_file)

    def _load_model(self, device: str) -> str:
        # Check the selected prior before resolving any executable so checkpoint
        # replacement is reported as the primary fail-closed condition.
        self._verify_prior()
        resolved = shutil.which(self.executable)
        if resolved is None:
            raise OptionalDependencyError(
                "reinvent was not found on PATH; install the pinned REINVENT4 release in this "
                "sidecar environment and verify `reinvent --help`"
            )
        executable = str(Path(resolved).expanduser().resolve(strict=True))
        self._verify_invocation_artifacts(executable)
        return executable

    def _verify_invocation_artifacts(self, executable: str) -> None:
        """Re-attest files that the next external REINVENT process will open."""

        self._verify_prior()
        executable_path = Path(executable).expanduser().resolve(strict=True)
        if not executable_path.is_file():
            raise ModelExecutionError("configured REINVENT executable is not a file")
        if (
            self.executable_sha256 is not None
            and sha256_file(executable_path) != self.executable_sha256
        ):
            raise ModelExecutionError(
                "REINVENT executable bytes changed after runtime attestation"
            )

    def _verify_prior(self) -> None:
        model_path = Path(self.model_file).expanduser().resolve(strict=True)
        if not model_path.is_file():
            raise ModelExecutionError("configured REINVENT model_file is not a file")
        if sha256_file(model_path) != self.model_sha256:
            raise ModelExecutionError(
                "REINVENT prior bytes changed after runtime attestation"
            )
        self.model_file = str(model_path)

    def generate(self, request: FusionGenerationRequest) -> GeneratedBatch:
        executable = self._ensure_loaded()
        # Unlike an in-process model, REINVENT opens both its prior and console
        # entrypoint again for every request.  A successful first request must
        # not turn later file replacement into an unrecorded model change.
        self._verify_invocation_artifacts(executable)
        requested = request.run_config.candidate_count
        controls = request.run_config.generation_controls
        sample_count = requested * self.oversample_factor
        warnings = [
            "REINVENT sampling applies temperature directly; alpha has no sampling-mode CLI "
            "equivalent and is retained only in provenance.",
        ]
        if request.revision_proposal is not None or request.latent_state is not None:
            warnings.append(
                "REINVENT sampling mode does not consume the raw unified latent or property "
                "desired_changes. Scheduler temperature/mutation/diversity controls are applied; "
                "property optimization requires an explicitly configured staged-learning/scoring "
                "workflow, so no synthetic conditioning was invented."
            )
        with tempfile.TemporaryDirectory(prefix="discovery-reinvent-") as temporary:
            root = Path(temporary)
            output = root / "sampling.csv"
            parameters: dict[str, Any] = {
                "model_file": self.model_file,
                "output_file": str(output),
                "num_smiles": sample_count,
                "unique_molecules": controls.diversity_strength > 0.0,
                "randomize_smiles": controls.mutation_strength > 0.0,
                "sample_strategy": "multinomial",
                "temperature": controls.temperature,
            }
            if self.mode == "mol2mol":
                seed_file = root / "parent.smi"
                seed_file.write_text(candidate_smiles(request.parent_candidate) + "\n", encoding="utf-8")
                parameters["smiles_file"] = str(seed_file)
            else:
                warnings.append(
                    "The configured REINVENT prior is de-novo; the parent molecule is recorded as "
                    "lineage but is not a sampling seed. Use mode='mol2mol' with a Mol2Mol prior "
                    "for direct parent conditioning."
                )
            config = {
                "run_type": "sampling",
                "device": _reinvent_device(self.device),
                "parameters": parameters,
            }
            config_path = root / "sampling.json"
            config_path.write_text(
                json.dumps(config, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                encoding="utf-8",
            )
            result = _run_bounded_process(
                [
                    executable,
                    "-f",
                    "json",
                    "-d",
                    _reinvent_device(self.device),
                    "-s",
                    str(request.run_config.effective_generator_seed),
                    str(config_path),
                ],
                cwd=root,
                env=_subprocess_environment(self.pass_environment),
                timeout=self.process_timeout_seconds,
            )
            if result.returncode != 0:
                raise ModelExecutionError(
                    "REINVENT generation failed with exit code "
                    f"{result.returncode}: {result.stderr_text or result.stdout_text or 'no log output'}"
                )
            smiles = _read_reinvent_csv(
                output,
                requested_count=requested,
                max_output_bytes=self.max_output_bytes,
            )
        candidates = tuple(
            GeneratedCandidateData(
                name=f"REINVENT candidate {index + 1}",
                representations=(
                    CandidateRepresentation(
                        kind=RepresentationKind.SMILES,
                        value=value,
                        media_type="chemical/x-daylight-smiles",
                        canonical=True,
                    ),
                ),
                attributes={"generation_controls": controls.model_dump(mode="json")},
                provenance={"adapter": "reinvent4-cli-v1", "mode": self.mode},
            )
            for index, value in enumerate(smiles)
        )
        return GeneratedBatch(candidates=candidates, warnings=tuple(warnings))

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "model_sha256": self.model_sha256,
            "executable_sha256": self.executable_sha256,
            "mode": self.mode,
            "process_timeout_seconds": self.process_timeout_seconds,
            "oversample_factor": self.oversample_factor,
            "max_output_bytes": self.max_output_bytes,
            "pass_environment": list(self.pass_environment),
            "requested_device": self._requested_device,
        }


class _ProcessResult:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes, *, truncated: bool) -> None:
        self.returncode = returncode
        suffix = " [log truncated]" if truncated else ""
        self.stdout_text = stdout.decode("utf-8", errors="replace").strip() + suffix
        self.stderr_text = stderr.decode("utf-8", errors="replace").strip() + suffix


def _seed_mattergen(seed: int) -> None:
    """Apply the request's generator seed to every RNG MatterGen uses."""

    numpy = require_module("numpy", install_hint="install MatterGen's pinned NumPy build")
    torch = require_module("torch", install_hint="install MatterGen's pinned PyTorch build")
    random.seed(seed)
    numpy.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    cuda = getattr(torch, "cuda", None)
    manual_seed_all = getattr(cuda, "manual_seed_all", None)
    if callable(manual_seed_all):
        manual_seed_all(seed)


def _run_bounded_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
    log_limit: int = 256 * 1024,
) -> _ProcessResult:
    """Run a fixed argv without a shell while continuously draining bounded logs."""

    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
        )
    except OSError as exc:
        raise ModelExecutionError(f"could not start model CLI: {type(exc).__name__}: {exc}") from exc
    buffers = [bytearray(), bytearray()]
    truncated = [False, False]

    def drain(stream: Any, index: int) -> None:
        try:
            while True:
                chunk = stream.read(64 * 1024)
                if not chunk:
                    break
                remaining = log_limit - len(buffers[index])
                if remaining > 0:
                    buffers[index].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    truncated[index] = True
        finally:
            stream.close()

    assert process.stdout is not None and process.stderr is not None
    threads = [
        threading.Thread(target=drain, args=(process.stdout, 0), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, 1), daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        process.kill()
        process.wait()
        for thread in threads:
            thread.join(timeout=5)
        raise ModelTimeoutError(f"model CLI exceeded its {timeout:g} second process timeout") from exc
    for thread in threads:
        thread.join(timeout=5)
    return _ProcessResult(
        returncode,
        bytes(buffers[0]),
        bytes(buffers[1]),
        truncated=any(truncated),
    )


def _subprocess_environment(pass_names: tuple[str, ...]) -> dict[str, str]:
    names = {
        "PATH",
        "HOME",
        "USERPROFILE",
        "SYSTEMROOT",
        "WINDIR",
        "TMP",
        "TEMP",
        "CUDA_VISIBLE_DEVICES",
        "XDG_CACHE_HOME",
        *pass_names,
    }
    return {name: value for name, value in os.environ.items() if name in names}


def _read_cif_archive(
    path: Path,
    *,
    expected_count: int,
    max_archive_bytes: int,
    max_cif_bytes: int,
) -> list[tuple[str, str]]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ModelOutputError("MatterGen did not produce generated_crystals_cif.zip") from exc
    if size <= 0 or size > max_archive_bytes:
        raise ModelOutputError("MatterGen CIF archive is empty or exceeds the configured size limit")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = [item for item in archive.infolist() if not item.is_dir()]
            if len(entries) != expected_count:
                raise ModelOutputError(
                    f"MatterGen returned {len(entries)} CIF files, expected {expected_count}"
                )
            output: list[tuple[str, str]] = []
            total = 0
            for entry in sorted(entries, key=lambda item: item.filename):
                pure = PurePosixPath(entry.filename)
                if pure.is_absolute() or ".." in pure.parts or pure.suffix.lower() != ".cif":
                    raise ModelOutputError("MatterGen archive contains an unsafe or non-CIF entry")
                if entry.file_size <= 0 or entry.file_size > max_cif_bytes:
                    raise ModelOutputError("MatterGen CIF exceeds the per-representation size limit")
                if entry.compress_size == 0 or entry.file_size / entry.compress_size > 200:
                    raise ModelOutputError("MatterGen archive has a suspicious compression ratio")
                total += entry.file_size
                if total > expected_count * max_cif_bytes:
                    raise ModelOutputError("MatterGen archive exceeds the uncompressed size limit")
                raw = archive.read(entry)
                if len(raw) != entry.file_size:
                    raise ModelOutputError("MatterGen archive entry size changed while reading")
                output.append((entry.filename, raw.decode("utf-8")))
            return output
    except ModelOutputError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise ModelOutputError(f"MatterGen returned an invalid CIF archive: {type(exc).__name__}: {exc}") from exc


def _read_reinvent_csv(path: Path, *, requested_count: int, max_output_bytes: int) -> list[str]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ModelOutputError("REINVENT did not produce the configured sampling CSV") from exc
    if size <= 0 or size > max_output_bytes:
        raise ModelOutputError("REINVENT sampling CSV is empty or exceeds the size limit")
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ModelOutputError("REINVENT sampling CSV has no header")
            smiles_name = next(
                (name for name in reader.fieldnames if name.strip().lower() in {"smiles", "smile"}),
                None,
            )
            if smiles_name is None:
                raise ModelOutputError("REINVENT sampling CSV has no SMILES column")
            raw_values = [row.get(smiles_name, "").strip() for row in reader]
    except ModelOutputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ModelOutputError(f"REINVENT returned an invalid CSV: {type(exc).__name__}: {exc}") from exc
    chem = require_module(
        "rdkit.Chem",
        install_hint="install RDKit in the REINVENT sidecar to validate generated SMILES",
    )
    canonical: list[str] = []
    seen: set[str] = set()
    for value in raw_values:
        if not value or "\n" in value or "\r" in value:
            continue
        molecule = chem.MolFromSmiles(value)
        if molecule is None:
            continue
        normalized = str(chem.MolToSmiles(molecule, canonical=True))
        if normalized and normalized not in seen:
            canonical.append(normalized)
            seen.add(normalized)
        if len(canonical) == requested_count:
            return canonical
    raise ModelOutputError(
        f"REINVENT produced only {len(canonical)} unique valid molecules, expected {requested_count}"
    )


def _reinvent_device(device: str) -> str:
    if device == "mps":
        return "mps"
    if device.startswith("cuda"):
        return device
    return "cpu"


def _condition_value(name: str, value: Any) -> str | float | int:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ModelExecutionError(
            f"MatterGen condition {name!r} must be a string or finite scalar"
        )
    if isinstance(value, float) and not (-1e100 < value < 1e100):
        raise ModelExecutionError(f"MatterGen condition {name!r} is not finite")
    if isinstance(value, str) and not value.strip():
        raise ModelExecutionError(f"MatterGen condition {name!r} cannot be blank")
    return value


__all__ = ["MatterGenGenerator", "ReinventGenerator"]
