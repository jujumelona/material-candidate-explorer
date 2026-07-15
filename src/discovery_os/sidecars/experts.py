"""Lazy specialist adapters backed by upstream public inference entrypoints.

No optional scientific package is imported when this module is imported.
Every adapter either returns values produced by the actual upstream model or
fails closed; there are deliberately no random/zero-vector fallbacks.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
import os
import shlex
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from discovery_os.fusion_schemas import ExpertFeatureRequest, TensorRole
from discovery_os.fusion_schemas import ScientificModality
from discovery_os.schemas import CandidateType, RepresentationKind

from .base import LazyModelAdapter, require_module, to_plain_data
from .conversions import (
    atom_entity_ids,
    candidate_sequence,
    candidate_smiles,
    candidate_to_ase,
    candidate_to_pymatgen,
    cell_expression,
    representation,
)
from .errors import (
    CandidateConversionError,
    ModelExecutionError,
    ModelOutputError,
    SidecarError,
    UnsupportedModelError,
)
from .generators import _run_bounded_process, _subprocess_environment
from .qhnet import (
    QHNET_POSITION_SCALE_TO_BOHR,
    QHNetRuntimeConfig,
    QHNetSourceAttestation,
    attest_qhnet_bundle,
    load_qhnet_runtime_config,
    verify_qhnet_source_bundle,
)
from .types import ExpertResult, PropertyResult
from .weight_binding import directory_inventory_sha256, sha256_file


_PERIODIC_MATERIAL_TYPES = frozenset(
    {
        CandidateType.CRYSTAL,
        CandidateType.COMPOSITION,
        CandidateType.ALLOY,
        CandidateType.CATALYST,
        CandidateType.BATTERY_MATERIAL,
    }
)
_MOLECULAR_TYPES = frozenset({CandidateType.SMALL_MOLECULE, CandidateType.CUSTOM})


def _require_request_route(
    request: ExpertFeatureRequest,
    *,
    modality: ScientificModality,
    feature_space: str,
    candidate_types: frozenset[CandidateType],
    representation_kinds: tuple[RepresentationKind, ...],
) -> Any:
    """Fail before model loading when a direct sidecar call bypasses its descriptor."""

    if request.modality != modality:
        raise CandidateConversionError(
            f"adapter requires modality {modality}, got {request.modality}"
        )
    if request.feature_space != feature_space:
        raise CandidateConversionError(
            f"adapter requires feature space {feature_space!r}, got {request.feature_space!r}"
        )
    if request.candidate.candidate_type not in candidate_types:
        allowed = ", ".join(sorted(str(item) for item in candidate_types))
        raise CandidateConversionError(
            f"adapter does not support candidate type {request.candidate.candidate_type}; "
            f"allowed types: {allowed}"
        )
    return representation(request.candidate, representation_kinds)


def _require_atoms_periodicity(atoms: Any, *, periodic: bool, model_name: str) -> None:
    try:
        raw = to_plain_data(atoms.get_pbc())
        flags = [bool(raw)] * 3 if isinstance(raw, bool) else [bool(item) for item in raw]
    except Exception as exc:
        raise CandidateConversionError(f"{model_name} could not inspect structure periodicity") from exc
    if len(flags) != 3:
        raise CandidateConversionError(f"{model_name} requires exactly three periodic axes")
    if periodic and not all(flags):
        raise CandidateConversionError(f"{model_name} requires a fully periodic structure")
    if not periodic and any(flags):
        raise CandidateConversionError(f"{model_name} molecular task requires a non-periodic structure")


class UMAExpert(LazyModelAdapter[Any]):
    """FAIR-Chem UMA through ``pretrained_mlip`` and ``FAIRChemCalculator``."""

    def __init__(
        self,
        *,
        model_name: str = "uma-s-1p2",
        task_name: str = "omat",
        checkpoint_path: str | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.task_name = task_name.strip().lower()
        if self.task_name not in {"omat", "omol"}:
            raise ValueError(
                "UMA task must be 'omat' or 'omol'; other tasks require a reviewed route"
            )
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path) if checkpoint_path else None

    def _load_model(self, device: str) -> Any:
        if self.checkpoint_path is not None and (
            self.checkpoint_sha256 is None
            or sha256_file(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise ModelExecutionError("UMA checkpoint bytes changed after runtime attestation")
        core = require_module(
            "fairchem.core",
            install_hint=(
                "install fairchem-core, obtain access to the configured UMA weights, "
                "and authenticate with Hugging Face"
            ),
        )
        try:
            if self.checkpoint_path is None:
                raise ModelExecutionError(
                    "UMA requires UMA_CHECKPOINT_PATH from a verified local snapshot; "
                    "online model-name resolution is disabled"
                )
            calculator = core.FAIRChemCalculator.from_model_checkpoint(
                str(Path(self.checkpoint_path).expanduser().resolve(strict=True)),
                task_name=self.task_name,
                device=device,
            )
        except Exception as exc:
            raise ModelExecutionError(
                f"UMA checkpoint {self.model_name!r} could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        return calculator

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        if self.task_name == "omat":
            kinds = (RepresentationKind.CIF, RepresentationKind.POSCAR)
            _require_request_route(
                request,
                modality=ScientificModality.CRYSTAL_MATERIAL,
                feature_space="uma-atomic-v1",
                candidate_types=_PERIODIC_MATERIAL_TYPES,
                representation_kinds=kinds,
            )
            periodic = True
        else:
            kinds = (
                RepresentationKind.XYZ,
                RepresentationKind.EXTXYZ,
                RepresentationKind.SDF,
            )
            _require_request_route(
                request,
                modality=ScientificModality.MOLECULE_3D,
                feature_space="uma-atomic-v1",
                candidate_types=frozenset({CandidateType.SMALL_MOLECULE}),
                representation_kinds=kinds,
            )
            periodic = False
        atoms = candidate_to_ase(request.candidate, kinds=kinds)
        _require_atoms_periodicity(atoms, periodic=periodic, model_name=f"UMA {self.task_name}")
        calculator = self._ensure_loaded()
        try:
            atoms.calc = calculator
            return _ase_force_result(atoms, source=f"UMA:{self.model_name}")
        except SidecarError:
            raise
        except Exception as exc:
            raise ModelExecutionError(f"UMA inference failed: {type(exc).__name__}: {exc}") from exc

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "model_name": self.model_name,
            "task_name": self.task_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "requested_device": self._requested_device,
        }


class MatterSimExpert(LazyModelAdapter[Any]):
    """MatterSim via its public ASE ``MatterSimCalculator``."""

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        weight_attestation: str | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = sha256_file(checkpoint_path) if checkpoint_path else None
        self.weight_attestation = weight_attestation

    def _load_model(self, device: str) -> Any:
        if self.checkpoint_path is not None and (
            self.checkpoint_sha256 is None
            or sha256_file(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise ModelExecutionError(
                "MatterSim checkpoint bytes changed after runtime attestation"
            )
        forcefield = require_module(
            "mattersim.forcefield",
            install_hint="install the pinned mattersim package in its isolated Python environment",
        )
        kwargs: dict[str, Any] = {"device": device}
        if self.checkpoint_path is not None:
            path = Path(self.checkpoint_path).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ModelExecutionError("MatterSim checkpoint path is not a file")
            kwargs["load_path"] = str(path)
        try:
            return forcefield.MatterSimCalculator(**kwargs)
        except Exception as exc:
            raise ModelExecutionError(
                f"MatterSim calculator could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        kinds = (RepresentationKind.CIF, RepresentationKind.POSCAR)
        _require_request_route(
            request,
            modality=ScientificModality.CRYSTAL_MATERIAL,
            feature_space="mattersim-atomic-v1",
            candidate_types=_PERIODIC_MATERIAL_TYPES,
            representation_kinds=kinds,
        )
        atoms = candidate_to_ase(request.candidate, kinds=kinds)
        _require_atoms_periodicity(atoms, periodic=True, model_name="MatterSim")
        calculator = self._ensure_loaded()
        try:
            atoms.calc = calculator
            return _ase_force_result(atoms, source="MatterSim")
        except SidecarError:
            raise
        except Exception as exc:
            raise ModelExecutionError(
                f"MatterSim inference failed: {type(exc).__name__}: {exc}"
            ) from exc

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "checkpoint_sha256": self.checkpoint_sha256,
            "weight_attestation": self.weight_attestation,
            "requested_device": self._requested_device,
        }


class CHGNetExpert(LazyModelAdapter[Any]):
    """CHGNet direct structure inference with energy/forces/stress/magmoms."""

    def __init__(
        self,
        *,
        model_name: str = "0.3.0",
        weight_attestation: str | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.weight_attestation = weight_attestation

    def _load_model(self, device: str) -> Any:
        module = require_module(
            "chgnet.model.model",
            install_hint="install chgnet and pymatgen in this isolated sidecar environment",
        )
        try:
            model = module.CHGNet.load(model_name=self.model_name)
            move = getattr(model, "to", None)
            if callable(move):
                model = move(device)
            return model
        except Exception as exc:
            raise ModelExecutionError(
                f"CHGNet checkpoint {self.model_name!r} could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        _require_request_route(
            request,
            modality=ScientificModality.CRYSTAL_MATERIAL,
            feature_space="chgnet-atomic-v1",
            candidate_types=_PERIODIC_MATERIAL_TYPES,
            representation_kinds=(RepresentationKind.CIF, RepresentationKind.POSCAR),
        )
        structure = candidate_to_pymatgen(request.candidate)
        model = self._ensure_loaded()
        try:
            prediction = model.predict_structure(structure, task="efsm")
        except TypeError:
            # Older supported CHGNet releases infer the same task implicitly.
            prediction = model.predict_structure(structure)
        except Exception as exc:
            raise ModelExecutionError(f"CHGNet inference failed: {type(exc).__name__}: {exc}") from exc
        try:
            forces = _matrix(prediction.get("f", prediction.get("forces")), columns=3)
            magmom_raw = prediction.get("m", prediction.get("magmom"))
            magmoms = _vector(magmom_raw) if magmom_raw is not None else [0.0] * len(forces)
            if len(magmoms) != len(forces):
                raise ModelOutputError("CHGNet magmom length does not match the atom count")
            atom_features = [force + [magmom] for force, magmom in zip(forces, magmoms, strict=True)]
            energy = _scalar(prediction.get("e", prediction.get("energy")), "CHGNet energy")
            stress = prediction.get("s", prediction.get("stress"))
            properties = [
                PropertyResult("energy_per_atom", energy, "eV/atom", source="CHGNet"),
                PropertyResult("max_force", _max_row_norm(forces), "eV/angstrom", source="CHGNet"),
            ]
            if stress is not None:
                properties.append(
                    PropertyResult("stress_norm", _numeric_norm(stress), "GPa", source="CHGNet")
                )
            return ExpertResult(
                values=atom_features,
                tensor_role=TensorRole.CUSTOM,
                projection_id="chgnet-force-magmom-v1",
                entity_type="atom",
                entity_ids=atom_entity_ids(structure),
                normalization="none",
                coordinate_frame="Cartesian; force xyz then magnetic moment",
                unit_semantics={
                    "columns_0_2": "eV/angstrom",
                    "column_3": "mu_B",
                    "energy_per_atom": "eV/atom",
                    "stress_norm": "GPa",
                },
                properties=tuple(properties),
            )
        except SidecarError:
            raise
        except Exception as exc:
            raise ModelOutputError(
                f"CHGNet returned an invalid result: {type(exc).__name__}: {exc}"
            ) from exc

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "model_name": self.model_name,
            "weight_attestation": self.weight_attestation,
            "requested_device": self._requested_device,
        }


class UniMolExpert(LazyModelAdapter[Any]):
    """Uni-Mol CLS embeddings through ``unimol_tools.UniMolRepr``."""

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        dictionary_path: str | None = None,
        remove_hs: bool = False,
        batch_size: int = 16,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self.dictionary_path = dictionary_path
        self.remove_hs = remove_hs
        self.batch_size = batch_size
        self.checkpoint_sha256 = sha256_file(checkpoint_path) if checkpoint_path else None
        self.dictionary_sha256 = sha256_file(dictionary_path) if dictionary_path else None

    def _load_model(self, device: str) -> Any:
        if self.checkpoint_path and (
            self.checkpoint_sha256 is None
            or sha256_file(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise ModelExecutionError(
                "Uni-Mol checkpoint bytes changed after runtime attestation"
            )
        if self.dictionary_path and (
            self.dictionary_sha256 is None
            or sha256_file(self.dictionary_path) != self.dictionary_sha256
        ):
            raise ModelExecutionError(
                "Uni-Mol dictionary bytes changed after runtime attestation"
            )
        module = require_module(
            "unimol_tools",
            install_hint="install the pinned unimol_tools package in this sidecar environment",
        )
        kwargs: dict[str, Any] = {
            "data_type": "molecule",
            "remove_hs": self.remove_hs,
            "batch_size": self.batch_size,
            "use_gpu": device.startswith("cuda"),
        }
        if not self.checkpoint_path or not self.dictionary_path:
            raise ModelExecutionError(
                "Uni-Mol requires checkpoint and dictionary files from a verified local snapshot; "
                "online/package fallback is disabled"
            )
        if self.checkpoint_path:
            kwargs["pretrained_model_path"] = str(Path(self.checkpoint_path).resolve(strict=True))
        if self.dictionary_path:
            kwargs["pretrained_dict_path"] = str(Path(self.dictionary_path).resolve(strict=True))
        try:
            return module.UniMolRepr(**kwargs)
        except Exception as exc:
            raise ModelExecutionError(
                f"Uni-Mol checkpoint could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        if request.modality == ScientificModality.MOLECULE_3D:
            kinds = (RepresentationKind.SDF, RepresentationKind.XYZ)
            _require_request_route(
                request,
                modality=ScientificModality.MOLECULE_3D,
                feature_space="unimol-cls-v1",
                candidate_types=frozenset(
                    {CandidateType.SMALL_MOLECULE, CandidateType.CATALYST}
                ),
                representation_kinds=kinds,
            )
            atoms = candidate_to_ase(request.candidate, kinds=kinds)
            model_input: Any = {
                "atoms": [list(str(item) for item in atoms.get_chemical_symbols())],
                "coordinates": [
                    [[float(value) for value in row] for row in atoms.get_positions().tolist()]
                ],
            }
        elif request.modality == ScientificModality.MOLECULE_2D:
            _require_request_route(
                request,
                modality=ScientificModality.MOLECULE_2D,
                feature_space="unimol-cls-v1",
                candidate_types=frozenset(
                    {
                        CandidateType.SMALL_MOLECULE,
                        CandidateType.POLYMER,
                        CandidateType.CATALYST,
                    }
                ),
                representation_kinds=(RepresentationKind.SMILES,),
            )
            model_input = [
                candidate_smiles(
                    request.candidate,
                    kinds=(RepresentationKind.SMILES,),
                )
            ]
        else:
            raise CandidateConversionError(
                "Uni-Mol requires molecule_2d or molecule_3d modality"
            )
        model = self._ensure_loaded()
        try:
            output = model.get_repr(model_input, return_atomic_reprs=True)
            cls = output["cls_repr"]
        except Exception as exc:
            raise ModelExecutionError(f"Uni-Mol inference failed: {type(exc).__name__}: {exc}") from exc
        rows = _matrix(cls)
        if len(rows) != 1:
            raise ModelOutputError("Uni-Mol single-candidate call returned an unexpected batch size")
        return ExpertResult(
            values=rows,
            tensor_role=TensorRole.GLOBAL_EMBEDDING,
            projection_id="unimol-cls-v1",
            entity_type="molecule",
            entity_ids=("molecule",),
            pooling="cls",
            normalization="upstream Uni-Mol checkpoint normalization",
        )

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dictionary_sha256": self.dictionary_sha256,
            "remove_hs": self.remove_hs,
            "batch_size": self.batch_size,
            "requested_device": self._requested_device,
        }


class ChempropExpert(LazyModelAdapter[tuple[Any, Any, Any, Any]]):
    """Chemprop v2 MPN fingerprint plus configured prediction properties."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        property_names: tuple[str, ...],
        property_units: tuple[str, ...],
        encoding_layer: int = 0,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        if not property_names or any(not item.strip() for item in property_names):
            raise ValueError("Chemprop requires at least one non-blank property name")
        if len(set(property_names)) != len(property_names):
            raise ValueError("Chemprop property names must be unique")
        if len(property_units) != len(property_names) or any(
            not item.strip() for item in property_units
        ):
            raise ValueError(
                "Chemprop requires one non-blank property unit per property name"
            )
        self.checkpoint_path = checkpoint_path
        self.property_names = property_names
        self.property_units = property_units
        self.encoding_layer = encoding_layer
        self.checkpoint_sha256 = sha256_file(checkpoint_path)

    def _load_model(self, device: str) -> tuple[Any, Any, Any, Any]:
        path = Path(self.checkpoint_path).expanduser().resolve(strict=True)
        if sha256_file(path) != self.checkpoint_sha256:
            raise ModelExecutionError(
                "Chemprop checkpoint bytes changed after runtime attestation"
            )
        chemprop = require_module(
            "chemprop",
            install_hint="install chemprop>=2 and RDKit in this isolated sidecar environment",
        )
        torch = require_module("torch", install_hint="install the Chemprop-compatible PyTorch build")
        if not path.is_file():
            raise ModelExecutionError("Chemprop checkpoint path is not a file")
        try:
            model = chemprop.models.MPNN.load_from_checkpoint(path, map_location=device)
            model.eval()
            model.to(device)
            featurizer = chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer()
        except Exception as exc:
            raise ModelExecutionError(
                f"Chemprop checkpoint could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        return model, chemprop.data, featurizer, torch

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        _require_request_route(
            request,
            modality=ScientificModality.MOLECULE_2D,
            feature_space="chemprop-mpn-v1",
            candidate_types=frozenset(
                {CandidateType.SMALL_MOLECULE, CandidateType.CATALYST}
            ),
            representation_kinds=(RepresentationKind.SMILES,),
        )
        smiles = candidate_smiles(
            request.candidate,
            kinds=(RepresentationKind.SMILES,),
        )
        model, data, featurizer, torch = self._ensure_loaded()
        try:
            points = [data.MoleculeDatapoint.from_smi(smiles)]
            dataset = data.MoleculeDataset(points, featurizer=featurizer)
            batch = next(iter(data.build_dataloader(dataset, shuffle=False)))
            move = getattr(batch, "to", None)
            if callable(move):
                batch = move(self.device)
            with torch.inference_mode():
                fingerprint = model.encoding(
                    batch.bmg,
                    batch.V_d,
                    batch.X_d,
                    i=self.encoding_layer,
                )
                prediction = model.predict_step(batch, 0)
        except Exception as exc:
            raise ModelExecutionError(f"Chemprop inference failed: {type(exc).__name__}: {exc}") from exc
        rows = _matrix(fingerprint)
        if len(rows) != 1:
            raise ModelOutputError("Chemprop single-candidate call returned an unexpected batch size")
        predictions = _matrix(prediction)[0]
        if len(self.property_names) != len(predictions):
            raise ModelOutputError(
                "configured Chemprop property names do not match checkpoint output width"
            )
        properties = [
            PropertyResult(name, value, unit, source="Chemprop")
            for name, value, unit in zip(
                self.property_names,
                predictions,
                self.property_units,
                strict=True,
            )
        ]
        return ExpertResult(
            values=rows,
            tensor_role=TensorRole.GLOBAL_EMBEDDING,
            projection_id=f"chemprop-mpn-layer-{self.encoding_layer}-v1",
            entity_type="molecule",
            entity_ids=("molecule",),
            pooling="mean",
            normalization="upstream Chemprop checkpoint normalization",
            properties=tuple(properties),
        )

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "checkpoint_sha256": self.checkpoint_sha256,
            "property_names": list(self.property_names),
            "property_units": list(self.property_units),
            "encoding_layer": self.encoding_layer,
            "requested_device": self._requested_device,
        }


class ESMExpert(LazyModelAdapter[tuple[Any, Any, Any]]):
    """Pinned EvolutionaryScale ESM3 embeddings through the Biohub API."""

    def __init__(
        self,
        *,
        model_name: str = "esm3_sm_open_v1",
        snapshot_path: str | None = None,
        max_residues: int = 1_022,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.model_name = model_name
        self.snapshot_path = snapshot_path
        self.max_residues = max_residues
        main_weight = (
            Path(snapshot_path) / "data" / "weights" / "esm3_sm_open_v1.pth"
            if snapshot_path
            else None
        )
        self.checkpoint_sha256 = sha256_file(main_weight) if main_weight else None
        self.snapshot_inventory_sha256 = (
            directory_inventory_sha256(snapshot_path) if snapshot_path else None
        )

    def _load_model(self, device: str) -> tuple[Any, Any, Any]:
        if self.snapshot_path is not None and (
            self.snapshot_inventory_sha256 is None
            or directory_inventory_sha256(self.snapshot_path)
            != self.snapshot_inventory_sha256
        ):
            raise ModelExecutionError("ESM snapshot bytes changed after runtime attestation")
        esm3_module = require_module(
            "esm.models.esm3",
            install_hint=(
                "install the pinned EvolutionaryScale/Biohub esm==3.2.3 package and authenticate "
                "for the configured ESM3 weights"
            ),
        )
        api = require_module(
            "esm.sdk.api",
            install_hint="install the complete pinned EvolutionaryScale esm SDK",
        )
        try:
            if self.snapshot_path is None:
                raise ModelExecutionError(
                    "ESM3 requires ESM_SNAPSHOT_PATH from a verified local snapshot; "
                    "Hugging Face download fallback is disabled"
                )
            pretrained = require_module(
                "esm.pretrained",
                install_hint="install the complete pinned EvolutionaryScale esm SDK",
            )
            snapshot = Path(self.snapshot_path).expanduser().resolve(strict=True)
            # esm.pretrained imports data_root into module scope.  Rebinding that
            # exact symbol keeps every ESM3 builder on the verified snapshot and
            # prevents constants.data_root() from calling snapshot_download().
            pretrained.data_root = lambda _model: snapshot
            model = esm3_module.ESM3.from_pretrained(self.model_name).to(device)
            evaluate = getattr(model, "eval", None)
            if callable(evaluate):
                evaluate()
        except Exception as exc:
            raise ModelExecutionError(
                f"ESM3 checkpoint {self.model_name!r} could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return model, api.ESMProtein, api.LogitsConfig

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        _require_request_route(
            request,
            modality=ScientificModality.PROTEIN_SEQUENCE,
            feature_space="esm-sequence-v1",
            candidate_types=frozenset(
                {CandidateType.BIOLOGIC, CandidateType.PROTEIN}
            ),
            representation_kinds=(
                RepresentationKind.PROTEIN_SEQUENCE,
                RepresentationKind.FASTA,
                RepresentationKind.PDB,
            ),
        )
        sequence = candidate_sequence(request.candidate, molecule="protein")
        if len(sequence) > self.max_residues:
            raise CandidateConversionError(
                f"protein length {len(sequence)} exceeds this ESM checkpoint limit {self.max_residues}"
            )
        model, protein_class, logits_config_class = self._ensure_loaded()
        try:
            protein = protein_class(sequence=sequence)
            protein_tensor = model.encode(protein)
            output = model.logits(
                protein_tensor,
                logits_config_class(sequence=True, return_embeddings=True),
            )
            embeddings = output.embeddings
            if embeddings is None:
                raise ModelOutputError("ESM3 logits output omitted requested embeddings")
            dimensions = int(embeddings.dim())
            if dimensions == 3:
                if int(embeddings.shape[0]) != 1:
                    raise ModelOutputError("ESM3 returned an unexpected embedding batch size")
                embeddings = embeddings[0]
            if int(embeddings.dim()) != 2:
                raise ModelOutputError("ESM3 embeddings must have token and feature axes")
            token_count = int(embeddings.shape[0])
            if token_count == len(sequence) + 2:
                embeddings = embeddings[1:-1]
            elif token_count != len(sequence):
                raise ModelOutputError(
                    "ESM3 embedding token count does not match the protein sequence"
                )
            pooled = embeddings.mean(dim=0, keepdim=True)
        except SidecarError:
            raise
        except Exception as exc:
            raise ModelExecutionError(f"ESM3 inference failed: {type(exc).__name__}: {exc}") from exc
        return ExpertResult(
            values=pooled,
            tensor_role=TensorRole.GLOBAL_EMBEDDING,
            projection_id=f"esm3-{self.model_name}-mean-v1",
            entity_type="protein",
            entity_ids=("protein",),
            pooling="mean",
            normalization="mean of unmodified ESM3 residue embeddings",
            warnings=(
                "sequence was deterministically extracted from a single-chain PDB; "
                "this feature does not encode the supplied coordinates",
            )
            if any(item.kind == RepresentationKind.PDB for item in request.candidate.representations)
            else (),
        )

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "model_name": self.model_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "snapshot_inventory_sha256": self.snapshot_inventory_sha256,
            "max_residues": self.max_residues,
            "pooling": "mean_without_bos_eos",
            "requested_device": self._requested_device,
        }


class RNAFMExpert(LazyModelAdapter[tuple[Any, Any, int, Any]]):
    """RNA-FM mean sequence embedding through ``fm.pretrained.rna_fm_t12``."""

    def __init__(
        self,
        *,
        checkpoint_path: str | None = None,
        max_nucleotides: int = 1_022,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.checkpoint_path = checkpoint_path
        self.max_nucleotides = max_nucleotides
        self.checkpoint_sha256 = sha256_file(checkpoint_path) if checkpoint_path else None

    def _load_model(self, device: str) -> tuple[Any, Any, int, Any]:
        if self.checkpoint_path is not None and (
            self.checkpoint_sha256 is None
            or sha256_file(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise ModelExecutionError(
                "RNA-FM checkpoint bytes changed after runtime attestation"
            )
        fm = require_module("fm", install_hint="install rna-fm in this isolated sidecar")
        torch = require_module("torch", install_hint="install the RNA-FM-compatible PyTorch build")
        try:
            if self.checkpoint_path is None:
                raise ModelExecutionError(
                    "RNA-FM requires RNAFM_CHECKPOINT_PATH from a verified local snapshot; "
                    "network download fallback is disabled"
                )
            model, alphabet = fm.pretrained.rna_fm_t12(
                model_location=str(Path(self.checkpoint_path).expanduser().resolve(strict=True))
            )
            model.eval().to(device)
            converter = alphabet.get_batch_converter()
        except Exception as exc:
            raise ModelExecutionError(
                f"RNA-FM checkpoint could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        return model, converter, 12, torch

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        _require_request_route(
            request,
            modality=ScientificModality.RNA_SEQUENCE,
            feature_space="rnafm-t12-v1",
            candidate_types=frozenset({CandidateType.BIOLOGIC, CandidateType.RNA}),
            representation_kinds=(
                RepresentationKind.RNA_SEQUENCE,
                RepresentationKind.FASTA,
            ),
        )
        sequence = candidate_sequence(request.candidate, molecule="rna")
        if len(sequence) > self.max_nucleotides:
            raise CandidateConversionError(
                f"RNA length {len(sequence)} exceeds this RNA-FM limit {self.max_nucleotides}"
            )
        model, converter, layer, torch = self._ensure_loaded()
        try:
            _, _, tokens = converter([("candidate", sequence)])
            tokens = tokens.to(self.device)
            with torch.no_grad():
                residues = model(tokens, repr_layers=[layer])["representations"][layer][
                    0, 1 : len(sequence) + 1
                ]
                pooled = residues.mean(dim=0, keepdim=True)
        except Exception as exc:
            raise ModelExecutionError(f"RNA-FM inference failed: {type(exc).__name__}: {exc}") from exc
        return ExpertResult(
            values=pooled,
            tensor_role=TensorRole.GLOBAL_EMBEDDING,
            projection_id="rnafm-t12-layer-12-mean-v1",
            entity_type="rna",
            entity_ids=("rna",),
            pooling="mean",
            normalization="mean of unmodified upstream nucleotide representations",
            warnings=(),
        )

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "checkpoint_sha256": self.checkpoint_sha256,
            "max_nucleotides": self.max_nucleotides,
            "representation_layer": 12,
            "pooling": "mean_without_special_tokens",
            "requested_device": self._requested_device,
        }


class PySCFExpert(LazyModelAdapter[tuple[Any, Any]]):
    """Restricted/unrestricted Hartree-Fock through PySCF's public API."""

    def __init__(self, *, basis: str = "def2-svp", device: str = "cpu") -> None:
        if device not in {"auto", "cpu"}:
            raise ValueError("this PySCF adapter supports CPU only")
        super().__init__(device="cpu")
        self.basis = basis

    def _load_model(self, device: str) -> tuple[Any, Any]:
        gto = require_module("pyscf.gto", install_hint="install pyscf in its Linux/WSL sidecar")
        scf = require_module("pyscf.scf", install_hint="install pyscf in its Linux/WSL sidecar")
        return gto, scf

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        kinds = (RepresentationKind.XYZ, RepresentationKind.SDF)
        _require_request_route(
            request,
            modality=ScientificModality.ELECTRONIC_STRUCTURE,
            feature_space="pyscf-orbital-v1",
            candidate_types=_MOLECULAR_TYPES,
            representation_kinds=kinds,
        )
        atoms = candidate_to_ase(request.candidate, max_atoms=500, kinds=kinds)
        _require_atoms_periodicity(atoms, periodic=False, model_name="PySCF molecular")
        gto, scf = self._ensure_loaded()
        charge = _integer_attribute(request.candidate.attributes, "charge", default=0)
        spin = _integer_attribute(request.candidate.attributes, "spin", default=0)
        atom_spec = [
            (symbol, tuple(float(value) for value in position))
            for symbol, position in zip(
                atoms.get_chemical_symbols(), atoms.get_positions().tolist(), strict=True
            )
        ]
        try:
            molecule = gto.M(atom=atom_spec, basis=self.basis, charge=charge, spin=spin, unit="Angstrom")
            calculation = scf.RHF(molecule) if spin == 0 else scf.UHF(molecule)
            total_energy = float(calculation.kernel())
        except Exception as exc:
            raise ModelExecutionError(f"PySCF SCF calculation failed: {type(exc).__name__}: {exc}") from exc
        if not bool(getattr(calculation, "converged", False)):
            raise ModelExecutionError("PySCF SCF calculation did not converge; no feature was emitted")
        orbital = to_plain_data(calculation.mo_energy)
        if spin != 0:
            if not isinstance(orbital, (list, tuple)) or len(orbital) != 2:
                raise ModelOutputError("PySCF UHF did not return alpha and beta orbital channels")
            channels = [
                [float(value) for value in to_plain_data(channel)]
                for channel in orbital
            ]
            flattened = [value for channel in channels for value in channel]
        else:
            channels = [[float(value) for value in orbital]]
            flattened = list(channels[0])
        if not flattened or any(not math.isfinite(value) for value in flattened):
            raise ModelOutputError("PySCF returned invalid orbital energies")
        properties = [PropertyResult("total_energy", total_energy, "hartree", source="PySCF")]
        nelec = tuple(int(value) for value in getattr(molecule, "nelec", ()))
        if len(nelec) != 2 or any(value < 0 for value in nelec):
            raise ModelOutputError("PySCF molecule returned invalid alpha/beta electron counts")
        occupied_by_channel = nelec if spin != 0 else (sum(nelec) // 2,)
        occupied_energies: list[float] = []
        unoccupied_energies: list[float] = []
        for channel, occupied in zip(channels, occupied_by_channel, strict=True):
            if occupied > len(channel):
                raise ModelOutputError("PySCF electron count exceeds the orbital channel width")
            if occupied:
                occupied_energies.extend(channel[:occupied])
            if occupied < len(channel):
                unoccupied_energies.extend(channel[occupied:])
        if occupied_energies and unoccupied_energies:
            # UHF frontiers are selected across both spin channels, not from
            # an alpha-then-beta flattened array.
            homo = max(occupied_energies)
            lumo = min(unoccupied_energies)
            properties.extend(
                (
                    PropertyResult("homo_energy", homo, "hartree", source="PySCF"),
                    PropertyResult("lumo_energy", lumo, "hartree", source="PySCF"),
                    PropertyResult("homo_lumo_gap", lumo - homo, "hartree", source="PySCF"),
                )
            )
        return ExpertResult(
            values=[flattened],
            tensor_role=TensorRole.CUSTOM,
            projection_id="pyscf-mo-energies-v1",
            entity_type="electronic_structure",
            entity_ids=("molecular_orbitals",),
            pooling="none",
            normalization="none",
            basis=self.basis,
            unit_semantics={"tensor": "hartree"},
            properties=tuple(properties),
        )

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "basis": self.basis,
            "method": "RHF-if-spin-zero-else-UHF",
            "coordinate_unit": "angstrom",
            "device": "cpu",
        }


class _UnsupportedExpert:
    model_name = "model"
    reason = "no version-pinned public inference adapter is configured"
    install_action = "inject a reviewed runtime implementing encode(request)"
    loaded = False
    load_failed = False
    device = "unavailable"
    supported = False

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        raise UnsupportedModelError(
            f"{self.model_name} sidecar is fail-closed: {self.reason}; {self.install_action}. "
            "No synthetic tensor was returned."
        )

    def close(self) -> None:
        return None


class BoltzExpert(LazyModelAdapter[str]):
    """Boltz 2.2.1 through its official, file-oriented ``boltz predict`` CLI.

    The central feature request currently carries one candidate entity, not an
    entire related workspace.  This adapter therefore supports exactly one
    protein sequence, RNA sequence, or SMILES ligand.  It deliberately does
    not pretend that a ligand-only request is a protein-ligand affinity run.
    The returned tensor contains documented confidence observables and a small
    mmCIF structure summary; it is not a fabricated hidden embedding.
    """

    _CONFIDENCE_FIELDS = (
        ("confidence_score", "1"),
        ("ptm", "1"),
        ("iptm", "1"),
        ("ligand_iptm", "1"),
        ("protein_iptm", "1"),
        ("complex_plddt", "1"),
        ("complex_iplddt", "1"),
        ("complex_pde", "angstrom"),
        ("complex_ipde", "angstrom"),
    )
    _AFFINITY_FIELDS = (
        ("affinity_pred_value", "log10(micromolar_IC50)"),
        ("affinity_probability_binary", "1"),
        ("affinity_pred_value1", "log10(micromolar_IC50)"),
        ("affinity_probability_binary1", "1"),
        ("affinity_pred_value2", "log10(micromolar_IC50)"),
        ("affinity_probability_binary2", "1"),
    )

    def __init__(
        self,
        *,
        executable: str | None = None,
        executable_arguments: tuple[str, ...] = (),
        cache_path: str | None = None,
        checkpoint_path: str | None = None,
        affinity_checkpoint_path: str | None = None,
        mols_tar_path: str | None = None,
        process_timeout_seconds: float = 840.0,
        max_json_bytes: int = 1024 * 1024,
        max_cif_bytes: int = 8 * 1024 * 1024,
        max_sequence_length: int = 16_384,
        max_smiles_length: int = 8_192,
        no_kernels: bool = False,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.executable = executable
        self.executable_arguments = executable_arguments
        self.cache_path = cache_path
        self.checkpoint_path = checkpoint_path
        self.affinity_checkpoint_path = affinity_checkpoint_path
        self.mols_tar_path = mols_tar_path
        self.process_timeout_seconds = process_timeout_seconds
        self.max_json_bytes = max_json_bytes
        self.max_cif_bytes = max_cif_bytes
        self.max_sequence_length = max_sequence_length
        self.max_smiles_length = max_smiles_length
        self.no_kernels = no_kernels
        self.executable_sha256 = _optional_regular_file_sha256(executable)
        self.executable_argument_sha256 = [
            _optional_regular_file_sha256(item) for item in executable_arguments
        ]
        self.checkpoint_sha256 = (
            sha256_file(checkpoint_path) if checkpoint_path is not None else None
        )
        self.affinity_checkpoint_sha256 = (
            sha256_file(affinity_checkpoint_path)
            if affinity_checkpoint_path is not None
            else None
        )
        self.mols_tar_sha256 = sha256_file(mols_tar_path) if mols_tar_path is not None else None
        if process_timeout_seconds <= 0:
            raise ValueError("Boltz process timeout must be positive")
        if max_json_bytes <= 0 or max_cif_bytes <= 0:
            raise ValueError("Boltz output byte limits must be positive")
        if max_sequence_length <= 0 or max_smiles_length <= 0:
            raise ValueError("Boltz candidate length limits must be positive")
        for label, value in (
            ("executable", executable),
            *[("executable argument", item) for item in executable_arguments],
        ):
            if value is not None and (not value.strip() or any(char in value for char in "\r\n\x00")):
                raise ValueError(f"Boltz {label} contains an unsafe value")

    def _load_model(self, device: str) -> str:
        for label, path_value, expected_sha256 in (
            ("Boltz confidence checkpoint", self.checkpoint_path, self.checkpoint_sha256),
            (
                "Boltz affinity checkpoint",
                self.affinity_checkpoint_path,
                self.affinity_checkpoint_sha256,
            ),
            ("Boltz mols archive", self.mols_tar_path, self.mols_tar_sha256),
        ):
            if path_value is not None and (
                expected_sha256 is None
                or sha256_file(path_value) != expected_sha256
            ):
                raise ModelExecutionError(
                    f"{label} bytes changed after runtime attestation"
                )
        executable = self.executable
        if executable is None:
            executable = "boltz.exe" if sys.platform == "win32" else "boltz"
        path = Path(executable).expanduser()
        if path.is_absolute() or path.parent != Path("."):
            try:
                resolved = path.resolve(strict=True)
            except OSError as exc:
                raise ModelExecutionError("configured Boltz executable does not exist") from exc
            if not resolved.is_file():
                raise ModelExecutionError("configured Boltz executable is not a file")
            if (
                self.executable_sha256 is not None
                and sha256_file(resolved) != self.executable_sha256
            ):
                raise ModelExecutionError(
                    "Boltz executable bytes changed after runtime attestation"
                )
            for argument, expected_sha256 in zip(
                self.executable_arguments,
                self.executable_argument_sha256,
                strict=True,
            ):
                if expected_sha256 is not None and sha256_file(argument) != expected_sha256:
                    raise ModelExecutionError(
                        "Boltz executable argument bytes changed after runtime attestation"
                    )
            return str(resolved)
        return executable

    def _verify_invocation_artifacts(self, executable: str) -> None:
        """Re-attest every file the next external Boltz process will read."""

        for label, path_value, expected_sha256 in (
            ("Boltz confidence checkpoint", self.checkpoint_path, self.checkpoint_sha256),
            (
                "Boltz affinity checkpoint",
                self.affinity_checkpoint_path,
                self.affinity_checkpoint_sha256,
            ),
            ("Boltz mols archive", self.mols_tar_path, self.mols_tar_sha256),
        ):
            if path_value is not None and (
                expected_sha256 is None or sha256_file(path_value) != expected_sha256
            ):
                raise ModelExecutionError(
                    f"{label} bytes changed after runtime attestation"
                )
        if self.executable_sha256 is not None and (
            sha256_file(executable) != self.executable_sha256
        ):
            raise ModelExecutionError(
                "Boltz executable bytes changed after runtime attestation"
            )
        for argument, expected_sha256 in zip(
            self.executable_arguments,
            self.executable_argument_sha256,
            strict=True,
        ):
            if expected_sha256 is not None and sha256_file(argument) != expected_sha256:
                raise ModelExecutionError(
                    "Boltz executable argument bytes changed after runtime attestation"
                )

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "adapter": "boltz-2.2.1-cli-observables-v1",
            "model": "boltz2",
            "output_format": "mmcif",
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "protein_msa": "empty",
            "executable_sha256": self.executable_sha256,
            "executable_arguments": list(self.executable_arguments),
            "executable_argument_sha256": self.executable_argument_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "affinity_checkpoint_sha256": self.affinity_checkpoint_sha256,
            "mols_tar_sha256": self.mols_tar_sha256,
            "process_timeout_seconds": self.process_timeout_seconds,
            "max_json_bytes": self.max_json_bytes,
            "max_cif_bytes": self.max_cif_bytes,
            "max_sequence_length": self.max_sequence_length,
            "max_smiles_length": self.max_smiles_length,
            "no_kernels": self.no_kernels,
            "requested_device": self._requested_device,
        }

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        executable = self._ensure_loaded()
        # Boltz is an external process and reopens its executable/checkpoints on
        # every request, so the one-time lazy-load attestation is not sufficient.
        self._verify_invocation_artifacts(executable)
        entity_type, yaml_entity, route_warnings = _boltz_entity(
            request,
            max_sequence_length=self.max_sequence_length,
            max_smiles_length=self.max_smiles_length,
        )
        cache = (
            Path(self.cache_path).expanduser().resolve()
            if self.cache_path
            else (Path.home() / ".boltz").resolve()
        )
        if cache.exists() and not cache.is_dir():
            raise ModelExecutionError("configured Boltz cache path is not a directory")
        cache.mkdir(parents=True, exist_ok=True)
        _stage_boltz_cache_files(
            cache,
            checkpoint_path=self.checkpoint_path,
            affinity_checkpoint_path=self.affinity_checkpoint_path,
            mols_tar_path=self.mols_tar_path,
        )

        with tempfile.TemporaryDirectory(prefix="discovery-boltz-") as temporary:
            root = Path(temporary)
            input_path = root / "request.yaml"
            output_root = root / "output"
            # JSON is a strict YAML 1.2 subset.  Building this object ourselves
            # prevents requests from injecting arbitrary YAML tags or paths.
            document = {"version": 1, "sequences": [yaml_entity]}
            input_path.write_text(
                json.dumps(document, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
                encoding="utf-8",
            )
            command = [
                executable,
                *self.executable_arguments,
                "predict",
                str(input_path),
                "--out_dir",
                str(output_root),
                "--cache",
                str(cache),
                "--model",
                "boltz2",
                "--accelerator",
                _boltz_accelerator(self.device),
                "--devices",
                "1",
                "--diffusion_samples",
                "1",
                "--max_parallel_samples",
                "1",
                "--num_workers",
                "0",
                "--preprocessing-threads",
                "1",
                "--output_format",
                "mmcif",
                "--seed",
                str(request.seed),
                "--override",
            ]
            if self.checkpoint_path is not None:
                command.extend(("--checkpoint", self.checkpoint_path))
            if self.affinity_checkpoint_path is not None:
                command.extend(("--affinity_checkpoint", self.affinity_checkpoint_path))
            if self.no_kernels:
                command.append("--no_kernels")
            environment = _subprocess_environment(())
            if self.device.startswith("cuda:"):
                environment["CUDA_VISIBLE_DEVICES"] = self.device.split(":", 1)[1]
            result = _run_bounded_process(
                command,
                cwd=root,
                env=environment,
                timeout=self.process_timeout_seconds,
            )
            if result.returncode != 0:
                detail = result.stderr_text or result.stdout_text or "no log output"
                raise ModelExecutionError(
                    f"Boltz prediction failed with exit code {result.returncode}: {detail}"
                )
            parsed = _read_boltz_221_outputs(
                output_root,
                max_json_bytes=self.max_json_bytes,
                max_cif_bytes=self.max_cif_bytes,
            )

        confidence = parsed["confidence"]
        summary = parsed["structure"]
        tensor_names = [name for name, _ in self._CONFIDENCE_FIELDS] + [
            "predicted_atom_count",
            "predicted_chain_count",
            "predicted_residue_count",
        ]
        tensor_values = [confidence[name] for name, _ in self._CONFIDENCE_FIELDS] + [
            float(summary["atom_count"]),
            float(summary["chain_count"]),
            float(summary["residue_count"]),
        ]
        units = [unit for _, unit in self._CONFIDENCE_FIELDS] + ["count", "count", "count"]
        properties = [
            PropertyResult(name, confidence[name], unit, source="Boltz 2.2.1 confidence")
            for name, unit in self._CONFIDENCE_FIELDS
        ]
        properties.extend(
            PropertyResult(name, float(summary[name.removeprefix("predicted_")]), "count", source="Boltz 2.2.1 mmCIF")
            for name in (
                "predicted_atom_count",
                "predicted_chain_count",
                "predicted_residue_count",
            )
        )
        affinity = parsed["affinity"]
        if affinity is not None:
            properties.extend(
                PropertyResult(name, affinity[name], unit, source="Boltz 2.2.1 affinity")
                for name, unit in self._AFFINITY_FIELDS
            )
        warnings = [
            "Boltz tensor columns are documented confidence observables and mmCIF counts, not hidden embeddings.",
            *route_warnings,
        ]
        if entity_type == "ligand":
            warnings.append(
                "This is a ligand-only structure prediction; protein-ligand affinity requires a "
                "workspace-aware complex request and was not inferred from an unrelated target."
            )
        return ExpertResult(
            values=[tensor_values],
            tensor_role=TensorRole.CUSTOM,
            projection_id="boltz-2.2.1-confidence-structure-v1",
            entity_type="boltz_prediction",
            entity_ids=("model_0",),
            pooling="none",
            normalization="none",
            unit_semantics={
                f"column_{index}_{name}": unit
                for index, (name, unit) in enumerate(zip(tensor_names, units, strict=True))
            },
            properties=tuple(properties),
            quality_flags=(f"predicted_mmcif_sha256:{summary['sha256']}",),
            warnings=tuple(warnings),
        )


class ScGPTExpert(LazyModelAdapter[dict[str, Any]]):
    """scGPT 0.2.4 single-cell embedding using its official cell-embedding path.

    This adapter deliberately supports only one explicit cell-expression vector
    at a time.  It loads the inseparable ``args.json``/``vocab.json``/
    ``best_model.pt`` bundle and never substitutes a model-zoo checkpoint.
    """

    _SPECIAL_TOKENS = ("<pad>", "<cls>", "<eoc>")
    _CONFIG_MAX_BYTES = 2 * 1024 * 1024
    _VOCAB_MAX_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        *,
        checkpoint_dir: str,
        max_genes: int = 65_536,
        max_length: int = 1_200,
        use_fast_transformer: bool = False,
        bundle_inventory_sha256: str | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        if not 1 <= max_genes <= 65_536:
            raise ValueError("scGPT max_genes must be between 1 and 65536")
        if not 2 <= max_length <= 65_536:
            raise ValueError("scGPT max_length must be between 2 and 65536")
        selected_root = Path(checkpoint_dir).expanduser()
        if selected_root.is_symlink():
            raise ValueError("SCGPT_CHECKPOINT_DIR must not be a symlink")
        root = selected_root.resolve(strict=True)
        if not root.is_dir():
            raise ValueError("SCGPT_CHECKPOINT_DIR must be a regular non-symlink directory")
        self.checkpoint_dir = root
        self.config_path = _required_scgpt_bundle_file(root, "args.json")
        self.vocab_path = _required_scgpt_bundle_file(root, "vocab.json")
        self.checkpoint_path = _required_scgpt_bundle_file(root, "best_model.pt")
        measured_inventory = directory_inventory_sha256(root)
        if bundle_inventory_sha256 is not None and bundle_inventory_sha256 != measured_inventory:
            raise ValueError("scGPT bundle inventory changed after weight binding")
        self.bundle_inventory_sha256 = measured_inventory
        self.config_sha256 = sha256_file(self.config_path)
        self.vocab_sha256 = sha256_file(self.vocab_path)
        self.checkpoint_sha256 = sha256_file(self.checkpoint_path)
        self.model_config = _read_scgpt_json_object(
            self.config_path,
            max_bytes=self._CONFIG_MAX_BYTES,
            label="scGPT args.json",
        )
        self.vocab_mapping = _read_scgpt_json_object(
            self.vocab_path,
            max_bytes=self._VOCAB_MAX_BYTES,
            label="scGPT vocab.json",
        )
        self.architecture = _validate_scgpt_bundle_metadata(
            self.model_config,
            self.vocab_mapping,
        )
        self.max_genes = max_genes
        self.max_length = max_length
        self.use_fast_transformer = use_fast_transformer

    def _load_model(self, device: str) -> dict[str, Any]:
        if (
            directory_inventory_sha256(self.checkpoint_dir)
            != self.bundle_inventory_sha256
            or sha256_file(self.config_path) != self.config_sha256
            or sha256_file(self.vocab_path) != self.vocab_sha256
            or sha256_file(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise ModelExecutionError(
                "scGPT checkpoint bundle changed after runtime attestation"
            )
        torch = require_module(
            "torch",
            install_hint="install the PyTorch version resolved with pinned scgpt==0.2.4",
        )
        model_module = require_module(
            "scgpt.model",
            install_hint="install pinned scgpt==0.2.4 in this isolated sidecar",
        )
        tokenizer_module = require_module(
            "scgpt.tokenizer",
            install_hint="install the complete pinned scgpt==0.2.4 package",
        )
        collator_module = require_module(
            "scgpt.data_collator",
            install_hint="install the complete pinned scgpt==0.2.4 package",
        )
        utils_module = require_module(
            "scgpt.utils",
            install_hint="install the complete pinned scgpt==0.2.4 package",
        )
        preprocess_module = require_module(
            "scgpt.preprocess",
            install_hint="install the complete pinned scgpt==0.2.4 package",
        )
        try:
            vocab = tokenizer_module.GeneVocab.from_file(self.vocab_path)
            vocab.set_default_index(vocab[self.architecture["pad_token"]])
            model = model_module.TransformerModel(
                ntoken=len(vocab),
                d_model=self.architecture["embsize"],
                nhead=self.architecture["nheads"],
                d_hid=self.architecture["d_hid"],
                nlayers=self.architecture["nlayers"],
                nlayers_cls=self.architecture["n_layers_cls"],
                n_cls=1,
                vocab=vocab,
                dropout=self.architecture["dropout"],
                pad_token=self.architecture["pad_token"],
                pad_value=self.architecture["pad_value"],
                do_mvc=True,
                do_dab=False,
                use_batch_labels=False,
                domain_spec_batchnorm=False,
                input_emb_style="continuous",
                cell_emb_style="cls",
                mvc_decoder_style=self.architecture["mvc_decoder_style"],
                explicit_zero_prob=False,
                use_fast_transformer=self.use_fast_transformer,
                fast_transformer_backend="flash",
                pre_norm=self.architecture["pre_norm"],
            )
            if bool(getattr(model, "use_fast_transformer", False)) != self.use_fast_transformer:
                raise ModelExecutionError(
                    "scGPT changed the requested transformer backend during construction; "
                    "silent flash-attention fallback is disabled"
                )
            checkpoint = _safe_torch_load(torch, self.checkpoint_path, device=device)
            _validate_scgpt_encoder_checkpoint(
                model,
                checkpoint,
                translate_flash_weights=not self.use_fast_transformer,
            )
            utils_module.load_pretrained(model, checkpoint, strict=False, verbose=False)
            model.to(device)
            model.eval()
            collator = collator_module.DataCollator(
                do_padding=True,
                pad_token_id=vocab[self.architecture["pad_token"]],
                pad_value=self.architecture["pad_value"],
                do_mlm=False,
                # Binning is invoked explicitly with n_bins=51 below.  The
                # pinned 0.2.4 DataCollator hard-codes 51 and exposes no
                # n_bins argument, so leaving it enabled would hide that
                # output-affecting setting behind an upstream default.
                do_binning=False,
                max_length=self.max_length,
                sampling=False,
                keep_first_n_tokens=1,
            )
        except SidecarError:
            raise
        except Exception as exc:
            raise ModelExecutionError(
                f"scGPT checkpoint bundle could not be loaded: {type(exc).__name__}: {exc}"
            ) from exc
        return {
            "model": model,
            "vocab": vocab,
            "collator": collator,
            "preprocess": preprocess_module,
            "torch": torch,
        }

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        if request.modality != ScientificModality.CELL_STATE:
            raise CandidateConversionError("scGPT requires modality='cell_state'")
        if request.feature_space != "scgpt-cell-v1":
            raise CandidateConversionError("scGPT requires feature_space='scgpt-cell-v1'")
        if request.candidate.candidate_type not in {CandidateType.CELL_STATE, CandidateType.CUSTOM}:
            raise CandidateConversionError("scGPT requires a cell_state or custom candidate")
        genes, values, value_semantics = cell_expression(
            request.candidate,
            max_genes=self.max_genes,
        )
        if any(value < 0 for value in values):
            raise CandidateConversionError("scGPT expression values must be non-negative")
        if value_semantics == "raw_counts" and any(not value.is_integer() for value in values):
            raise CandidateConversionError("scGPT raw_counts values must be integers")

        known_nonzero_raw = [
            (gene, value)
            for gene, value in zip(genes, values, strict=True)
            if value != 0.0 and gene in self.vocab_mapping
        ]
        if not known_nonzero_raw:
            raise CandidateConversionError(
                "scGPT input has no non-zero genes present in the bound vocabulary"
            )

        bundle = self._ensure_loaded()
        model = bundle["model"]
        vocab = bundle["vocab"]
        collator = bundle["collator"]
        preprocess = bundle["preprocess"]
        torch = bundle["torch"]
        try:
            if value_semantics == "raw_counts":
                # Match the declared raw-count pipeline: library-size
                # normalization is defined on the complete caller vector.
                # Vocabulary filtering happens only after that transform, so
                # an out-of-vocabulary gene still contributes to cell depth.
                total = sum(values)
                if not math.isfinite(total) or total <= 0.0:
                    raise CandidateConversionError(
                        "scGPT raw_counts total expression must be finite and positive"
                    )
                prepared_values = [
                    math.log1p((value / total) * 10_000.0)
                    for _, value in known_nonzero_raw
                ]
                input_transform = "normalize_total_10000_then_log1p_then_vocab_filter"
            else:
                prepared_values = [value for _, value in known_nonzero_raw]
                input_transform = "caller_supplied_normalized_log1p"
            prepared_tensor = torch.tensor(
                prepared_values,
                dtype=torch.float32,
            )
            binned_values = _vector(
                to_plain_data(preprocess.binning(prepared_tensor, n_bins=51))
            )
            if len(binned_values) != len(known_nonzero_raw):
                raise ModelOutputError("scGPT binning changed the gene/value alignment")
            ranked = [
                (gene, prepared, binned)
                for (gene, _), prepared, binned in zip(
                    known_nonzero_raw,
                    prepared_values,
                    binned_values,
                    strict=True,
                )
            ]
            # Official scGPT sampling is stochastic when a cell exceeds
            # max_length.  For reproducible evidence, bin the complete filtered
            # vector first, then retain the highest-expression genes with a
            # lexical tie-break.
            ranked.sort(key=lambda item: (-item[1], item[0]))
            truncated = len(ranked) > self.max_length - 1
            selected = ranked[: self.max_length - 1]
            gene_ids = [vocab["<cls>"], *(vocab[gene] for gene, _, _ in selected)]
            expressions = [
                float(self.architecture["pad_value"]),
                *(binned for _, _, binned in selected),
            ]
            example = {
                "id": torch.tensor(0, dtype=torch.long),
                "genes": torch.tensor(gene_ids, dtype=torch.long),
                "expressions": torch.tensor(expressions, dtype=torch.float32),
            }
            batch = collator([example])
            input_gene_ids = batch["gene"].to(self.device)
            input_values = batch["expr"].to(self.device)
            if len(input_gene_ids.shape) != 2 or int(input_gene_ids.shape[0]) != 1:
                raise ModelOutputError("scGPT collator returned an invalid gene batch")
            if tuple(input_gene_ids.shape) != tuple(input_values.shape):
                raise ModelOutputError("scGPT gene and value tensors have different shapes")
            padding_mask = input_gene_ids.eq(vocab[self.architecture["pad_token"]])
            with torch.no_grad():
                encoded = model._encode(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=padding_mask,
                    batch_labels=None,
                )
            if len(encoded.shape) != 3 or int(encoded.shape[0]) != 1:
                raise ModelOutputError("scGPT encoder returned an invalid batch tensor")
            if int(encoded.shape[2]) != self.architecture["embsize"]:
                raise ModelOutputError("scGPT embedding width differs from args.json")
            row = _vector(to_plain_data(encoded[0, 0, :]))
            norm = math.sqrt(sum(value * value for value in row))
            if norm == 0.0:
                raise ModelOutputError("scGPT returned a zero-norm cell embedding")
            normalized = [[value / norm for value in row]]
        except SidecarError:
            raise
        except Exception as exc:
            raise ModelExecutionError(f"scGPT inference failed: {type(exc).__name__}: {exc}") from exc

        matched_count = len(known_nonzero_raw)
        input_nonzero_count = sum(value != 0.0 for value in values)
        flags = (
            f"input_gene_count:{len(genes)}",
            f"input_nonzero_gene_count:{input_nonzero_count}",
            f"vocab_nonzero_gene_count:{matched_count}",
            f"encoded_gene_count:{len(selected)}",
            f"value_semantics:{value_semantics}",
            f"input_transform:{input_transform}",
        )
        warnings: list[str] = []
        filtered = len(genes) - matched_count
        if filtered:
            warnings.append(
                f"{filtered} zero-valued or out-of-vocabulary genes were excluded before encoding"
            )
        if truncated:
            warnings.append(
                f"non-zero vocabulary genes were deterministically limited to {self.max_length - 1}"
            )
        return ExpertResult(
            values=normalized,
            tensor_role=TensorRole.CELL_EMBEDDING,
            projection_id="scgpt-0.2.4-cls-l2-v1",
            entity_type="cell",
            entity_ids=("cell:0",),
            pooling="cls",
            normalization="L2 normalization of the upstream <cls> encoder state",
            quality_flags=flags,
            warnings=tuple(warnings),
        )

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "bundle_inventory_sha256": self.bundle_inventory_sha256,
            "config_sha256": self.config_sha256,
            "vocab_sha256": self.vocab_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "bundle_members": ["args.json", "vocab.json", "best_model.pt"],
            "architecture": dict(self.architecture),
            "max_genes": self.max_genes,
            "max_length": self.max_length,
            "special_tokens": list(self._SPECIAL_TOKENS),
            "input_schema": "one_cell_genes_values_with_semantics_v1",
            "accepted_value_semantics": ["raw_counts", "normalized_log1p"],
            "vocabulary_filter": "exact_token_match_nonzero_only",
            "length_policy": "highest_expression_then_gene_name",
            "preprocessing": {
                "input_values": "explicit raw_counts or normalized_log1p selected in candidate JSON",
                "raw_counts_transform": "normalize_total_10000_then_log1p_then_vocab_filter",
                "normalized_log1p_transform": "identity",
                "zero_policy": "exclude before vocabulary filtering",
                "vocabulary_policy": "exact token match",
                "binning_api": "scgpt.preprocess.binning",
                "n_bins": 51,
            },
            "binning": {"enabled": True, "n_bins": 51, "explicit": True},
            "cell_embedding": "cls",
            "normalization": "l2",
            "use_fast_transformer": self.use_fast_transformer,
            "fast_transformer_backend": "flash" if self.use_fast_transformer else "pytorch",
            "requested_device": self._requested_device,
        }


def _required_scgpt_bundle_file(root: Path, name: str) -> Path:
    selected = root / name
    if selected.is_symlink():
        raise ValueError(f"scGPT bundle member {name!r} must not be a symlink")
    path = selected.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"scGPT bundle member {name!r} escapes SCGPT_CHECKPOINT_DIR") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"scGPT bundle requires regular non-symlink file {name!r}")
    return path


def _read_scgpt_json_object(path: Path, *, max_bytes: int, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    if not raw or len(raw) > max_bytes:
        raise ValueError(f"{label} is empty or exceeds {max_bytes} bytes")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _validate_scgpt_bundle_metadata(
    config: dict[str, Any],
    vocab: dict[str, Any],
) -> dict[str, Any]:
    if not vocab:
        raise ValueError("scGPT vocab.json must not be empty")
    if not all(isinstance(token, str) and token for token in vocab):
        raise ValueError("scGPT vocab.json contains an invalid token")
    if not all(isinstance(index, int) and not isinstance(index, bool) for index in vocab.values()):
        raise ValueError("scGPT vocab.json indices must be integers")
    indices = list(vocab.values())
    if len(indices) != len(set(indices)) or set(indices) != set(range(len(indices))):
        raise ValueError("scGPT vocab.json indices must be unique and contiguous from zero")
    for token in ScGPTExpert._SPECIAL_TOKENS:
        if token not in vocab:
            raise ValueError(
                f"scGPT vocab.json must already contain {token!r}; random special-token insertion is disabled"
            )

    def positive_integer(name: str) -> int:
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"scGPT args.json {name!r} must be a positive integer")
        return value

    embsize = positive_integer("embsize")
    nheads = positive_integer("nheads")
    if embsize % nheads != 0:
        raise ValueError("scGPT args.json embsize must be divisible by nheads")
    dropout_raw = config.get("dropout")
    if isinstance(dropout_raw, bool) or not isinstance(dropout_raw, (int, float)):
        raise ValueError("scGPT args.json 'dropout' must be numeric")
    dropout = float(dropout_raw)
    if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
        raise ValueError("scGPT args.json dropout must be in [0, 1)")
    pad_token = config.get("pad_token")
    if not isinstance(pad_token, str) or pad_token not in vocab:
        raise ValueError("scGPT args.json pad_token must name a token in vocab.json")
    if pad_token != "<pad>":
        raise ValueError("this reviewed scGPT cell-embedding codec requires pad_token='<pad>'")
    pad_value = config.get("pad_value")
    if isinstance(pad_value, bool) or not isinstance(pad_value, (int, float)):
        raise ValueError("scGPT args.json pad_value must be numeric")
    if not math.isfinite(float(pad_value)):
        raise ValueError("scGPT args.json pad_value must be finite")
    if config.get("input_emb_style", "continuous") != "continuous":
        raise ValueError("this scGPT adapter supports only continuous input embeddings")
    if config.get("cell_emb_style", "cls") != "cls":
        raise ValueError("this scGPT adapter supports only the upstream cls cell embedding")
    for name in ("use_batch_labels", "INPUT_BATCH_LABELS"):
        if config.get(name, False) is not False:
            raise ValueError("this scGPT adapter does not infer dataset-specific batch labels")
    for name in ("domain_spec_batchnorm", "DSBN"):
        if config.get(name, False) not in (False, None):
            raise ValueError("this scGPT adapter does not infer domain-specific batch labels")
    if "n_bins" in config and config["n_bins"] != 51:
        raise ValueError("scGPT 0.2.4 DataCollator cell embedding requires n_bins=51")
    pre_norm = config.get("pre_norm", False)
    if not isinstance(pre_norm, bool):
        raise ValueError("scGPT args.json pre_norm must be boolean when present")
    mvc_decoder_style = config.get("mvc_decoder_style", "inner product")
    if mvc_decoder_style not in {"inner product", "concat query", "sum query"}:
        raise ValueError("scGPT args.json contains an unsupported mvc_decoder_style")
    return {
        "embsize": embsize,
        "nheads": nheads,
        "d_hid": positive_integer("d_hid"),
        "nlayers": positive_integer("nlayers"),
        "n_layers_cls": positive_integer("n_layers_cls"),
        "dropout": dropout,
        "pad_token": pad_token,
        "pad_value": pad_value,
        "input_emb_style": "continuous",
        "cell_emb_style": "cls",
        "mvc_decoder_style": mvc_decoder_style,
        "pre_norm": pre_norm,
        "use_batch_labels": False,
        "domain_spec_batchnorm": False,
    }


def _safe_torch_load(torch: Any, path: Path, *, device: str) -> Mapping[str, Any]:
    load_parameters = inspect.signature(torch.load).parameters
    if "weights_only" not in load_parameters:
        raise ModelExecutionError(
            "scGPT requires a Torch version with weights_only checkpoint loading"
        )
    checkpoint = torch.load(str(path), map_location=device, weights_only=True)
    if not isinstance(checkpoint, Mapping) or not checkpoint:
        raise ModelExecutionError("scGPT best_model.pt must contain a non-empty state dictionary")
    if not all(isinstance(key, str) for key in checkpoint):
        raise ModelExecutionError("scGPT state dictionary keys must be strings")
    return checkpoint


def _validate_scgpt_encoder_checkpoint(
    model: Any,
    checkpoint: Mapping[str, Any],
    *,
    translate_flash_weights: bool,
) -> None:
    translated = {
        key.replace("Wqkv.", "in_proj_") if translate_flash_weights else key: value
        for key, value in checkpoint.items()
    }
    target = model.state_dict()
    required_prefixes = ("encoder.", "value_encoder.", "transformer_encoder.", "bn.", "dsbn.")
    required = [key for key in target if key.startswith(required_prefixes)]
    if not required:
        raise ModelExecutionError("scGPT model exposes no encoder state to validate")
    missing = [key for key in required if key not in translated]
    mismatched: list[str] = []
    for key in required:
        if key not in translated:
            continue
        expected_shape = tuple(getattr(target[key], "shape", ()))
        actual_shape = tuple(getattr(translated[key], "shape", ()))
        if not expected_shape or expected_shape != actual_shape:
            mismatched.append(key)
    if missing or mismatched:
        detail = []
        if missing:
            detail.append(f"missing {len(missing)} encoder tensors")
        if mismatched:
            detail.append(f"shape mismatch in {len(mismatched)} encoder tensors")
        raise ModelExecutionError(
            "scGPT checkpoint does not fully cover the configured embedding encoder: "
            + ", ".join(detail)
        )


class QHNetExpert(LazyModelAdapter[dict[str, Any]]):
    """AIRS/QHNet inference through its exact pinned source graph path.

    The released checkpoints are dataset-specific.  The selected config binds
    the ordered atomic-number sequences, basis, dtype, and Hamiltonian unit;
    candidates outside that declared scope fail closed.
    """

    supported = True

    def __init__(
        self,
        *,
        source_path: str,
        checkpoint_path: str,
        config_path: str,
        weight_attestation: str | None = None,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        self.source: QHNetSourceAttestation = verify_qhnet_source_bundle(source_path)
        self.config: QHNetRuntimeConfig = load_qhnet_runtime_config(config_path)
        bundle = attest_qhnet_bundle(
            checkpoint_path,
            config_path,
            declared_revision=weight_attestation,
        )
        self.checkpoint_path = bundle.checkpoint_path
        self.checkpoint_sha256 = bundle.checkpoint_sha256
        self.config_sha256 = bundle.config_sha256
        self.weight_attestation = bundle.revision

    def _load_model(self, device: str) -> dict[str, Any]:
        source = verify_qhnet_source_bundle(self.source.archive_root)
        config = load_qhnet_runtime_config(self.config.path)
        if (
            source != self.source
            or config.sha256 != self.config_sha256
            or sha256_file(self.checkpoint_path) != self.checkpoint_sha256
        ):
            raise ModelExecutionError(
                "QHNet source/checkpoint/config changed after runtime attestation"
            )
        if device == "mps":
            raise ModelExecutionError("the pinned QHNet runtime supports CPU or CUDA, not MPS")
        torch = require_module(
            "torch",
            install_hint="bootstrap the isolated qhnet-source environment",
        )
        pyg_data = require_module(
            "torch_geometric.data",
            install_hint="bootstrap the pinned QHNet torch-geometric dependencies",
        )
        models = _load_qhnet_models_package(self.source)
        get_model = getattr(models, "get_model", None)
        if not callable(get_model):
            raise ModelExecutionError("pinned QHNet source has no callable models.get_model")
        dtype = torch.float64 if self.config.dtype == "float64" else torch.float32
        get_default_dtype = getattr(torch, "get_default_dtype", None)
        set_default_dtype = getattr(torch, "set_default_dtype", None)
        if not callable(get_default_dtype) or not callable(set_default_dtype):
            raise ModelExecutionError("QHNet requires Torch default-dtype controls")
        previous_dtype = get_default_dtype()
        try:
            set_default_dtype(dtype)
            fork_rng = getattr(getattr(torch, "random", None), "fork_rng", None)
            if callable(fork_rng):
                with fork_rng(devices=[]):
                    torch.manual_seed(0)
                    model = get_model(SimpleNamespace(version=self.config.model_version))
            else:
                torch.manual_seed(0)
                model = get_model(SimpleNamespace(version=self.config.model_version))
        except Exception as exc:
            raise ModelExecutionError(
                f"QHNet {self.config.model_version} construction failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        finally:
            set_default_dtype(previous_dtype)

        load_parameters = inspect.signature(torch.load).parameters
        if "weights_only" not in load_parameters:
            raise ModelExecutionError(
                "QHNet requires a Torch version with weights_only checkpoint loading"
            )
        try:
            checkpoint = torch.load(
                str(self.checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ModelExecutionError(
                f"QHNet checkpoint could not be read safely: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(checkpoint, Mapping) or set(checkpoint) - {
            "state_dict",
            "eval",
            "batch_idx",
        }:
            raise ModelExecutionError(
                "QHNet checkpoint must be the official state_dict/eval/batch_idx mapping"
            )
        state_dict = checkpoint.get("state_dict")
        if not isinstance(state_dict, Mapping) or not state_dict or not all(
            isinstance(key, str) for key in state_dict
        ):
            raise ModelExecutionError("QHNet checkpoint contains no valid state_dict")
        try:
            model.load_state_dict(state_dict, strict=True)
            model.set(device)
            model.to(dtype=dtype)
            model.eval()
        except Exception as exc:
            raise ModelExecutionError(
                f"QHNet checkpoint/config mismatch: {type(exc).__name__}: {exc}"
            ) from exc
        data_class = getattr(pyg_data, "Data", None)
        if not callable(data_class):
            raise ModelExecutionError("torch_geometric.data.Data is unavailable")
        return {"model": model, "torch": torch, "data_class": data_class, "dtype": dtype}

    def encode(self, request: ExpertFeatureRequest) -> ExpertResult:
        if request.modality != ScientificModality.ELECTRONIC_STRUCTURE:
            raise CandidateConversionError("QHNet requires electronic_structure modality")
        if request.feature_space != "qhnet-hamiltonian-v1":
            raise CandidateConversionError("QHNet requires feature_space='qhnet-hamiltonian-v1'")
        if request.candidate.candidate_type not in {
            CandidateType.SMALL_MOLECULE,
            CandidateType.CUSTOM,
        }:
            raise CandidateConversionError("QHNet accepts small_molecule or custom candidates")
        selected = representation(
            request.candidate,
            (RepresentationKind.XYZ, RepresentationKind.SDF),
        )
        # Do not let an alternative canonical representation silently change
        # the model input after route selection.
        qhnet_candidate = request.candidate.model_copy(
            update={"representations": [selected], "candidate_ref": None}
        )
        atoms = candidate_to_ase(qhnet_candidate, max_atoms=64)
        try:
            if any(bool(value) for value in atoms.get_pbc()):
                raise CandidateConversionError("QHNet MD17 checkpoints accept non-periodic molecules only")
            atomic_numbers = tuple(int(value) for value in atoms.get_atomic_numbers())
            positions = atoms.get_positions().tolist()
        except SidecarError:
            raise
        except Exception as exc:
            raise CandidateConversionError(
                f"QHNet could not read atomic numbers/coordinates: {type(exc).__name__}: {exc}"
            ) from exc
        if atomic_numbers not in self.config.allowed_atomic_number_sequences:
            raise CandidateConversionError(
                "QHNet candidate atomic-number order is outside the configured checkpoint scope"
            )
        charge = _integer_attribute(request.candidate.attributes, "charge", default=0)
        multiplicity = _integer_attribute(
            request.candidate.attributes,
            "spin_multiplicity",
            default=1,
        )
        if charge != self.config.molecular_charge or multiplicity != self.config.spin_multiplicity:
            raise CandidateConversionError(
                "QHNet candidate charge/spin does not match the configured neutral-singlet checkpoint"
            )

        runtime = self._ensure_loaded()
        torch = runtime["torch"]
        try:
            position_tensor = torch.tensor(
                positions,
                dtype=runtime["dtype"],
                device=self.device,
            ) * QHNET_POSITION_SCALE_TO_BOHR
            atom_tensor = torch.tensor(
                atomic_numbers,
                dtype=torch.int64,
                device=self.device,
            ).view(-1, 1)
            data = runtime["data_class"](pos=position_tensor, atoms=atom_tensor)
            data.batch = torch.zeros(len(atomic_numbers), dtype=torch.int64, device=self.device)
            data.ptr = torch.tensor(
                [0, len(atomic_numbers)],
                dtype=torch.int64,
                device=self.device,
            )
            inference_context = getattr(torch, "inference_mode", None) or torch.no_grad
            with inference_context():
                output = runtime["model"](data)
        except SidecarError:
            raise
        except Exception as exc:
            raise ModelExecutionError(f"QHNet inference failed: {type(exc).__name__}: {exc}") from exc
        if not isinstance(output, Mapping) or "hamiltonian" not in output:
            raise ModelOutputError("QHNet output does not contain a Hamiltonian")
        plain = to_plain_data(output["hamiltonian"])
        if not isinstance(plain, (list, tuple)) or len(plain) != 1:
            raise ModelOutputError("QHNet single-molecule output must have one batch dimension")
        hamiltonian = _matrix(plain[0])
        expected_dimension = sum(5 if value <= 2 else 14 for value in atomic_numbers)
        if len(hamiltonian) != expected_dimension or any(
            len(row) != expected_dimension for row in hamiltonian
        ):
            raise ModelOutputError(
                "QHNet Hamiltonian shape does not match the pinned upstream orbital masks"
            )
        if expected_dimension * expected_dimension > 65_536:
            raise ModelOutputError("QHNet Hamiltonian exceeds the wire tensor limit")
        symmetry_tolerance = 1e-10 if self.config.dtype == "float64" else 1e-5
        if any(
            abs(hamiltonian[row][column] - hamiltonian[column][row]) > symmetry_tolerance
            for row in range(expected_dimension)
            for column in range(row)
        ):
            raise ModelOutputError("QHNet returned a non-symmetric Hamiltonian")
        return ExpertResult(
            values=hamiltonian,
            tensor_role=TensorRole.HAMILTONIAN,
            projection_id="qhnet-airspinned-full-hamiltonian-v1",
            entity_type="atomic_orbital_basis_function",
            entity_ids=_qhnet_orbital_entity_ids(atomic_numbers),
            pooling="none",
            normalization="none",
            coordinate_frame="input Cartesian xyz converted from angstrom to bohr",
            basis=self.config.basis,
            unit_semantics={"hamiltonian_matrix_element": self.config.hamiltonian_unit},
            quality_flags=(
                f"dataset_scope:{self.config.dataset_id}",
                f"checkpoint_sha256:{self.checkpoint_sha256}",
                f"config_sha256:{self.config_sha256}",
                f"source_inventory_sha256:{self.source.source_inventory_sha256}",
            ),
            warnings=(
                "QHNet is dataset-specific; the exact ordered atomic-number sequence was enforced.",
                "No overlap matrix was supplied, so orbital energies/coefficients were not fabricated.",
            ),
        )

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "source_archive_sha256": self.source.archive_sha256,
            "source_inventory_sha256": self.source.source_inventory_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "config_sha256": self.config_sha256,
            "weight_attestation": self.weight_attestation,
            "model_version": self.config.model_version,
            "dataset_id": self.config.dataset_id,
            "dtype": self.config.dtype,
            "basis": self.config.basis,
            "hamiltonian_unit": self.config.hamiltonian_unit,
            "position_unit": "angstrom",
            "position_scale_to_bohr": QHNET_POSITION_SCALE_TO_BOHR,
            "molecular_charge": self.config.molecular_charge,
            "spin_multiplicity": self.config.spin_multiplicity,
            "allowed_atomic_number_sequences": [
                list(sequence) for sequence in self.config.allowed_atomic_number_sequences
            ],
            "requested_device": self._requested_device,
        }


def _load_qhnet_models_package(source: QHNetSourceAttestation) -> Any:
    """Import only the attested QHNet ``models`` package under a private name."""

    package_name = f"_discovery_qhnet_models_{source.source_inventory_sha256[:20]}"
    existing = sys.modules.get(package_name)
    if existing is not None:
        return existing
    initializer = source.qhnet_root / "models" / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package_name,
        initializer,
        submodule_search_locations=[str(initializer.parent)],
    )
    if spec is None or spec.loader is None:
        raise ModelExecutionError("could not construct an import spec for pinned QHNet models")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        for name in tuple(sys.modules):
            if name == package_name or name.startswith(package_name + "."):
                sys.modules.pop(name, None)
        raise ModelExecutionError(
            f"pinned QHNet source import failed: {type(exc).__name__}: {exc}"
        ) from exc
    return module


def _qhnet_orbital_entity_ids(atomic_numbers: tuple[int, ...]) -> tuple[str, ...]:
    ids: list[str] = []
    for atom_index, atomic_number in enumerate(atomic_numbers):
        # This is exactly QHNet.get_orbital_mask(): H/He select five of the
        # padded 14 channels; period-2 elements use all fourteen channels.
        mask = (0, 1, 3, 4, 5) if atomic_number <= 2 else tuple(range(14))
        ids.extend(
            f"atom:{atom_index}:Z{atomic_number}:orbital_mask:{mask_index}"
            for mask_index in mask
        )
    return tuple(ids)


def _boltz_entity(
    request: ExpertFeatureRequest,
    *,
    max_sequence_length: int,
    max_smiles_length: int,
) -> tuple[str, dict[str, Any], tuple[str, ...]]:
    if request.feature_space != "boltz-structure-v1":
        raise CandidateConversionError("Boltz requires feature_space='boltz-structure-v1'")
    candidate = request.candidate
    kinds = {item.kind for item in candidate.representations}
    groups: set[str] = set()
    if RepresentationKind.PROTEIN_SEQUENCE in kinds:
        groups.add("protein")
    if RepresentationKind.RNA_SEQUENCE in kinds:
        groups.add("rna")
    if RepresentationKind.SMILES in kinds:
        groups.add("ligand")
    if RepresentationKind.FASTA in kinds:
        if candidate.candidate_type == CandidateType.PROTEIN:
            groups.add("protein")
        elif candidate.candidate_type == CandidateType.RNA:
            groups.add("rna")
        elif not groups:
            raise CandidateConversionError(
                "Boltz FASTA input is ambiguous unless candidate_type is protein or rna"
            )
    if len(groups) != 1:
        if not groups:
            raise CandidateConversionError(
                "Boltz accepts one protein_sequence, rna_sequence, FASTA, or SMILES entity"
            )
        raise CandidateConversionError(
            "one Boltz feature request cannot mix alternative protein, RNA, and ligand entities"
        )
    entity_type = next(iter(groups))
    if RepresentationKind.FASTA in kinds and not (
        RepresentationKind.PROTEIN_SEQUENCE in kinds or RepresentationKind.RNA_SEQUENCE in kinds
    ):
        fasta = representation(candidate, (RepresentationKind.FASTA,)).value
        headers = [line for line in fasta.splitlines() if line.lstrip().startswith(">")]
        if len(headers) != 1:
            raise CandidateConversionError("Boltz FASTA route requires exactly one sequence record")

    if entity_type == "protein":
        if candidate.candidate_type not in {CandidateType.PROTEIN, CandidateType.BIOLOGIC}:
            raise CandidateConversionError("protein sequence requires a protein or biologic candidate")
        if request.modality != ScientificModality.PROTEIN_STRUCTURE:
            raise CandidateConversionError("Boltz protein route requires protein_structure modality")
        sequence = candidate_sequence(candidate, molecule="protein")
        if len(sequence) > max_sequence_length:
            raise CandidateConversionError(
                f"protein sequence length {len(sequence)} exceeds Boltz sidecar limit {max_sequence_length}"
            )
        return (
            entity_type,
            {"protein": {"id": "A", "sequence": sequence, "msa": "empty"}},
            (
                "Protein inference uses Boltz's explicit single-sequence msa: empty mode; "
                "the official documentation warns that this can reduce accuracy.",
            ),
        )
    if entity_type == "rna":
        if candidate.candidate_type not in {CandidateType.RNA, CandidateType.BIOLOGIC}:
            raise CandidateConversionError("RNA sequence requires an rna or biologic candidate")
        if request.modality != ScientificModality.RNA_STRUCTURE:
            raise CandidateConversionError("Boltz RNA route requires rna_structure modality")
        sequence = candidate_sequence(candidate, molecule="rna")
        if len(sequence) > max_sequence_length:
            raise CandidateConversionError(
                f"RNA sequence length {len(sequence)} exceeds Boltz sidecar limit {max_sequence_length}"
            )
        return entity_type, {"rna": {"id": "A", "sequence": sequence}}, ()

    if candidate.candidate_type != CandidateType.SMALL_MOLECULE:
        raise CandidateConversionError("Boltz SMILES route requires a small_molecule candidate")
    if request.modality != ScientificModality.MOLECULE_3D:
        raise CandidateConversionError("Boltz ligand route requires molecule_3d modality")
    smiles = candidate_smiles(candidate)
    if len(smiles) > max_smiles_length:
        raise CandidateConversionError(
            f"SMILES length {len(smiles)} exceeds Boltz sidecar limit {max_smiles_length}"
        )
    return entity_type, {"ligand": {"id": "A", "smiles": smiles}}, ()


def _stage_boltz_cache_files(
    cache: Path,
    *,
    checkpoint_path: str | None,
    affinity_checkpoint_path: str | None,
    mols_tar_path: str | None,
) -> None:
    configured = (checkpoint_path, affinity_checkpoint_path, mols_tar_path)
    if not any(configured):
        return
    if not all(configured):
        raise ModelExecutionError(
            "snapshot-bound Boltz requires confidence, affinity, and mols.tar files together"
        )
    for raw_source, name in zip(
        configured,
        ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar"),
        strict=True,
    ):
        assert raw_source is not None
        selected_source = Path(raw_source).expanduser()
        if selected_source.is_symlink():
            raise ModelExecutionError(f"verified Boltz cache source {name} must not be a symlink")
        source = selected_source.resolve(strict=True)
        if not source.is_file():
            raise ModelExecutionError(f"verified Boltz cache source {name} is not a regular file")
        target = cache / name
        if target.exists() or target.is_symlink():
            try:
                if target.is_file() and os.path.samefile(source, target):
                    continue
            except OSError:
                pass
            raise ModelExecutionError(
                f"Boltz cache file {name} is not bound to the configured verified snapshot"
            )
        try:
            os.link(source, target)
        except FileExistsError:
            if not target.is_file() or not os.path.samefile(source, target):
                raise ModelExecutionError(f"Boltz cache race produced an unbound {name}")
        except OSError:
            try:
                target.symlink_to(source)
            except FileExistsError:
                try:
                    if target.is_file() and os.path.samefile(source, target):
                        continue
                except OSError:
                    pass
                raise ModelExecutionError(f"Boltz cache race produced an unbound {name}")
            except OSError as exc:
                raise ModelExecutionError(
                    f"could not bind verified Boltz snapshot file {name} into its cache"
                ) from exc


def _optional_regular_file_sha256(path: str | None) -> str | None:
    if path is None:
        return None
    selected = Path(path).expanduser()
    if not selected.exists() or not selected.is_file():
        return None
    return sha256_file(selected)


def _boltz_accelerator(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "mps":
        raise ModelExecutionError("Boltz 2.2.1 CLI does not declare an MPS accelerator")
    if device == "cuda" or device.startswith("cuda:"):
        return "gpu"
    raise ModelExecutionError(f"Boltz resolved an unsupported device {device!r}")


def _read_boltz_221_outputs(
    output_root: Path,
    *,
    max_json_bytes: int,
    max_cif_bytes: int,
) -> dict[str, Any]:
    """Read only the documented v2.2.1 single-input/model-0 files."""

    prediction_dir = output_root / "boltz_results_request" / "predictions" / "request"
    confidence_path = prediction_dir / "confidence_request_model_0.json"
    cif_path = prediction_dir / "request_model_0.cif"
    affinity_path = prediction_dir / "affinity_request.json"
    confidence = _bounded_json_object(
        confidence_path,
        root=output_root,
        max_bytes=max_json_bytes,
        label="Boltz confidence output",
    )
    confidence_values = _numeric_fields(
        confidence,
        BoltzExpert._CONFIDENCE_FIELDS,
        label="Boltz confidence output",
    )
    for name in (
        "confidence_score",
        "ptm",
        "iptm",
        "ligand_iptm",
        "protein_iptm",
        "complex_plddt",
        "complex_iplddt",
    ):
        if not 0.0 <= confidence_values[name] <= 1.0:
            raise ModelOutputError(f"Boltz confidence field {name!r} is outside [0, 1]")
    cif_bytes = _bounded_file_bytes(
        cif_path,
        root=output_root,
        max_bytes=max_cif_bytes,
        label="Boltz mmCIF output",
    )
    try:
        cif_text = cif_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ModelOutputError("Boltz mmCIF output is not UTF-8") from exc
    structure = _mmcif_structure_summary(cif_text)
    structure["sha256"] = hashlib.sha256(cif_bytes).hexdigest()

    affinity: dict[str, float] | None = None
    if affinity_path.exists() or affinity_path.is_symlink():
        raw_affinity = _bounded_json_object(
            affinity_path,
            root=output_root,
            max_bytes=max_json_bytes,
            label="Boltz affinity output",
        )
        affinity = _numeric_fields(
            raw_affinity,
            BoltzExpert._AFFINITY_FIELDS,
            label="Boltz affinity output",
        )
        for name in (
            "affinity_probability_binary",
            "affinity_probability_binary1",
            "affinity_probability_binary2",
        ):
            if not 0.0 <= affinity[name] <= 1.0:
                raise ModelOutputError(f"Boltz affinity field {name!r} is outside [0, 1]")
    return {"confidence": confidence_values, "structure": structure, "affinity": affinity}


def _bounded_json_object(
    path: Path,
    *,
    root: Path,
    max_bytes: int,
    label: str,
) -> dict[str, Any]:
    raw = _bounded_file_bytes(path, root=root, max_bytes=max_bytes, label=label)

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ModelOutputError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ModelOutputError(f"{label} contains non-finite JSON number {value}")

    try:
        decoded = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except ModelOutputError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelOutputError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ModelOutputError(f"{label} must be a JSON object")
    return decoded


def _bounded_file_bytes(path: Path, *, root: Path, max_bytes: int, label: str) -> bytes:
    cursor = path
    while cursor != root:
        if cursor.is_symlink():
            raise ModelOutputError(f"{label} cannot be a symbolic link")
        parent = cursor.parent
        if parent == cursor:
            raise ModelOutputError(f"{label} escapes its temporary output root")
        cursor = parent
    try:
        resolved_root = root.resolve(strict=True)
        resolved = path.resolve(strict=True)
        resolved.relative_to(resolved_root)
    except (OSError, ValueError) as exc:
        raise ModelOutputError(f"{label} is missing or outside its temporary output root") from exc
    if not resolved.is_file():
        raise ModelOutputError(f"{label} is not a regular file")
    size = resolved.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ModelOutputError(f"{label} is empty or exceeds the configured size limit")
    return resolved.read_bytes()


def _numeric_fields(
    payload: dict[str, Any],
    fields: tuple[tuple[str, str], ...],
    *,
    label: str,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, _unit in fields:
        if name not in payload:
            raise ModelOutputError(f"{label} is missing required field {name!r}")
        raw = payload[name]
        if isinstance(raw, bool):
            raise ModelOutputError(f"{label} field {name!r} is boolean, not numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ModelOutputError(f"{label} field {name!r} is not numeric") from exc
        if not math.isfinite(value):
            raise ModelOutputError(f"{label} field {name!r} is not finite")
        values[name] = value
    return values


def _mmcif_structure_summary(value: str) -> dict[str, int]:
    lines = value.splitlines()
    atom_rows: list[tuple[str, ...]] = []
    headers: list[str] = []
    index = 0
    while index < len(lines):
        if lines[index].strip().lower() != "loop_":
            index += 1
            continue
        cursor = index + 1
        candidate_headers: list[str] = []
        while cursor < len(lines) and lines[cursor].strip().startswith("_"):
            candidate_headers.append(lines[cursor].strip().split()[0].lower())
            cursor += 1
        if candidate_headers and all(item.startswith("_atom_site.") for item in candidate_headers):
            headers = candidate_headers
            while cursor < len(lines):
                stripped = lines[cursor].strip()
                lowered = stripped.lower()
                if not stripped:
                    cursor += 1
                    continue
                if stripped == "#" or lowered == "loop_" or lowered.startswith("data_") or stripped.startswith("_"):
                    break
                try:
                    tokens = tuple(shlex.split(stripped, comments=False, posix=True))
                except ValueError as exc:
                    raise ModelOutputError("Boltz mmCIF atom loop contains invalid quoting") from exc
                if len(tokens) != len(headers):
                    raise ModelOutputError("Boltz mmCIF atom loop has a wrapped or malformed row")
                atom_rows.append(tokens)
                cursor += 1
            break
        index = cursor
    if not headers or not atom_rows:
        raise ModelOutputError("Boltz mmCIF output contains no _atom_site loop rows")
    positions = {name: offset for offset, name in enumerate(headers)}
    required = ("_atom_site.group_pdb", "_atom_site.label_asym_id")
    if any(name not in positions for name in required):
        raise ModelOutputError("Boltz mmCIF atom loop lacks group_PDB or label_asym_id")
    sequence_name = next(
        (name for name in ("_atom_site.label_seq_id", "_atom_site.auth_seq_id") if name in positions),
        None,
    )
    component_name = next(
        (name for name in ("_atom_site.label_comp_id", "_atom_site.auth_comp_id") if name in positions),
        None,
    )
    chains: set[str] = set()
    residues: set[tuple[str, str, str]] = set()
    atoms = 0
    for row in atom_rows:
        group = row[positions["_atom_site.group_pdb"]].upper()
        if group not in {"ATOM", "HETATM"}:
            continue
        atoms += 1
        chain = row[positions["_atom_site.label_asym_id"]]
        chains.add(chain)
        sequence = row[positions[sequence_name]] if sequence_name is not None else "."
        component = row[positions[component_name]] if component_name is not None else "."
        residues.add((chain, sequence, component))
    if atoms <= 0 or not chains:
        raise ModelOutputError("Boltz mmCIF output contains no ATOM/HETATM records")
    return {"atom_count": atoms, "chain_count": len(chains), "residue_count": len(residues)}


def _ase_force_result(atoms: Any, *, source: str) -> ExpertResult:
    energy = float(atoms.get_potential_energy())
    forces = _matrix(atoms.get_forces(), columns=3)
    entity_ids = atom_entity_ids(atoms)
    atom_count = len(entity_ids)
    if atom_count != len(forces):
        raise ModelOutputError(
            "ASE calculator returned a force row count that does not match the atom count"
        )
    properties = [
        PropertyResult("energy", energy, "eV", source=source),
        # This is the calculator's total potential energy normalized by atom
        # count.  It is not a phase-stability energy-above-hull calculation.
        PropertyResult("energy_per_atom", energy / atom_count, "eV/atom", source=source),
        PropertyResult("max_force", _max_row_norm(forces), "eV/angstrom", source=source),
    ]
    warnings: list[str] = []
    try:
        stress = atoms.get_stress(voigt=False)
    except Exception:
        warnings.append("upstream calculator did not expose stress for this structure")
    else:
        properties.append(
            PropertyResult("stress_norm", _numeric_norm(stress), "eV/angstrom^3", source=source)
        )
    return ExpertResult(
        values=forces,
        tensor_role=TensorRole.CUSTOM,
        projection_id=f"{source.lower().replace(':', '-')}-force-v1",
        entity_type="atom",
        entity_ids=entity_ids,
        normalization="none",
        coordinate_frame="Cartesian xyz",
        unit_semantics={
            "tensor": "eV/angstrom",
            "energy": "eV",
            "energy_per_atom": "eV/atom",
            "stress_norm": "eV/angstrom^3",
        },
        properties=tuple(properties),
        warnings=tuple(warnings),
    )


def _matrix(value: Any, *, columns: int | None = None) -> list[list[float]]:
    plain = to_plain_data(value)
    if not isinstance(plain, (list, tuple)) or not plain:
        raise ModelOutputError("model output is not a non-empty matrix")
    rows: list[list[float]] = []
    width: int | None = None
    for raw_row in plain:
        if not isinstance(raw_row, (list, tuple)):
            raise ModelOutputError("model output matrix contains a scalar row")
        row = [_finite_float(item) for item in raw_row]
        if not row:
            raise ModelOutputError("model output matrix contains an empty row")
        width = len(row) if width is None else width
        if len(row) != width or (columns is not None and len(row) != columns):
            raise ModelOutputError("model output matrix has an unexpected or ragged width")
        rows.append(row)
    return rows


def _vector(value: Any) -> list[float]:
    plain = to_plain_data(value)
    if not isinstance(plain, (list, tuple)) or not plain:
        raise ModelOutputError("model output is not a non-empty vector")
    return [_finite_float(item) for item in plain]


def _scalar(value: Any, label: str) -> float:
    plain = to_plain_data(value)
    if isinstance(plain, (list, tuple)):
        if len(plain) != 1:
            raise ModelOutputError(f"{label} is not scalar")
        plain = plain[0]
    return _finite_float(plain)


def _finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelOutputError("model output contains a non-numeric value") from exc
    if not math.isfinite(number):
        raise ModelOutputError("model output contains NaN or infinity")
    return number


def _max_row_norm(rows: list[list[float]]) -> float:
    return max(math.sqrt(sum(value * value for value in row)) for row in rows)


def _numeric_norm(value: Any) -> float:
    plain = to_plain_data(value)
    numbers: list[float] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            numbers.append(_finite_float(item))

    visit(plain)
    if not numbers:
        raise ModelOutputError("model output contains an empty stress tensor")
    return math.sqrt(sum(item * item for item in numbers))


def _integer_attribute(attributes: dict[str, Any], name: str, *, default: int) -> int:
    value = attributes.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CandidateConversionError(f"candidate attribute {name!r} must be an integer")
    return value


__all__ = [
    "BoltzExpert",
    "CHGNetExpert",
    "ChempropExpert",
    "ESMExpert",
    "MatterSimExpert",
    "PySCFExpert",
    "QHNetExpert",
    "RNAFMExpert",
    "ScGPTExpert",
    "UMAExpert",
    "UniMolExpert",
]
