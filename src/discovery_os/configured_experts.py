"""Code-owned mapping from installed specialist services to fusion encoders."""

from __future__ import annotations

import os
from collections.abc import Mapping

from .fusion_adapters import HttpExpertEncoder
from .fusion_registry import ExpertRegistry
from .fusion_schemas import (
    ExpertDescriptor,
    ExpertFeaturePayload,
    ExpertFeatureRequest,
    ExpertRoute,
    ScientificModality,
)
from .integration_manifest import IntegrationComponent, load_integration_manifest
from .schemas import CandidateType, RepresentationKind


_EXPERT_CAPABILITIES: dict[str, dict] = {
    "unimol": {
        "modalities": [ScientificModality.MOLECULE_2D, ScientificModality.MOLECULE_3D],
        "candidate_types": [
            CandidateType.SMALL_MOLECULE,
            CandidateType.POLYMER,
            CandidateType.CATALYST,
        ],
        "representations": [
            RepresentationKind.SMILES,
            RepresentationKind.SDF,
            RepresentationKind.XYZ,
        ],
        "feature_spaces": ["unimol-cls-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.MOLECULE_2D,
                feature_space="unimol-cls-v1",
                representation_kinds=[RepresentationKind.SMILES],
                candidate_types=[
                    CandidateType.SMALL_MOLECULE,
                    CandidateType.POLYMER,
                    CandidateType.CATALYST,
                ],
            ),
            ExpertRoute(
                modality=ScientificModality.MOLECULE_3D,
                feature_space="unimol-cls-v1",
                representation_kinds=[RepresentationKind.SDF, RepresentationKind.XYZ],
                candidate_types=[CandidateType.SMALL_MOLECULE, CandidateType.CATALYST],
            ),
        ],
    },
    "boltz": {
        "modalities": [
            ScientificModality.PROTEIN_STRUCTURE,
            ScientificModality.RNA_STRUCTURE,
            ScientificModality.MOLECULE_3D,
        ],
        "candidate_types": [
            CandidateType.BIOLOGIC,
            CandidateType.PROTEIN,
            CandidateType.RNA,
            CandidateType.SMALL_MOLECULE,
        ],
        "representations": [
            RepresentationKind.PROTEIN_SEQUENCE,
            RepresentationKind.RNA_SEQUENCE,
            RepresentationKind.FASTA,
            RepresentationKind.SMILES,
        ],
        "feature_spaces": ["boltz-structure-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.PROTEIN_STRUCTURE,
                feature_space="boltz-structure-v1",
                representation_kinds=[
                    RepresentationKind.PROTEIN_SEQUENCE,
                ],
                candidate_types=[CandidateType.BIOLOGIC, CandidateType.PROTEIN],
            ),
            ExpertRoute(
                modality=ScientificModality.PROTEIN_STRUCTURE,
                feature_space="boltz-structure-v1",
                representation_kinds=[RepresentationKind.FASTA],
                candidate_types=[CandidateType.PROTEIN],
            ),
            ExpertRoute(
                modality=ScientificModality.RNA_STRUCTURE,
                feature_space="boltz-structure-v1",
                representation_kinds=[RepresentationKind.RNA_SEQUENCE],
                candidate_types=[CandidateType.BIOLOGIC, CandidateType.RNA],
            ),
            ExpertRoute(
                modality=ScientificModality.RNA_STRUCTURE,
                feature_space="boltz-structure-v1",
                representation_kinds=[RepresentationKind.FASTA],
                candidate_types=[CandidateType.RNA],
            ),
            ExpertRoute(
                modality=ScientificModality.MOLECULE_3D,
                feature_space="boltz-structure-v1",
                representation_kinds=[RepresentationKind.SMILES],
                candidate_types=[CandidateType.SMALL_MOLECULE],
            ),
        ],
    },
    "esm": {
        "modalities": [ScientificModality.PROTEIN_SEQUENCE],
        "candidate_types": [CandidateType.BIOLOGIC, CandidateType.PROTEIN],
        "representations": [
            RepresentationKind.PROTEIN_SEQUENCE,
            RepresentationKind.FASTA,
            RepresentationKind.PDB,
        ],
        "feature_spaces": ["esm-sequence-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.PROTEIN_SEQUENCE,
                feature_space="esm-sequence-v1",
                representation_kinds=[
                    RepresentationKind.PROTEIN_SEQUENCE,
                ],
                candidate_types=[CandidateType.BIOLOGIC, CandidateType.PROTEIN],
            ),
            ExpertRoute(
                modality=ScientificModality.PROTEIN_SEQUENCE,
                feature_space="esm-sequence-v1",
                representation_kinds=[RepresentationKind.FASTA],
                candidate_types=[CandidateType.PROTEIN],
            ),
            ExpertRoute(
                # Coordinates are deliberately ignored: the adapter extracts
                # one sequence from PDB and still returns a sequence feature.
                modality=ScientificModality.PROTEIN_SEQUENCE,
                feature_space="esm-sequence-v1",
                representation_kinds=[RepresentationKind.PDB],
                candidate_types=[CandidateType.BIOLOGIC, CandidateType.PROTEIN],
            ),
        ],
    },
    "rnafm": {
        "modalities": [ScientificModality.RNA_SEQUENCE],
        "candidate_types": [CandidateType.BIOLOGIC, CandidateType.RNA],
        "representations": [RepresentationKind.RNA_SEQUENCE, RepresentationKind.FASTA],
        "feature_spaces": ["rnafm-t12-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.RNA_SEQUENCE,
                feature_space="rnafm-t12-v1",
                representation_kinds=[
                    RepresentationKind.RNA_SEQUENCE,
                ],
                candidate_types=[CandidateType.BIOLOGIC, CandidateType.RNA],
            ),
            ExpertRoute(
                modality=ScientificModality.RNA_SEQUENCE,
                feature_space="rnafm-t12-v1",
                representation_kinds=[RepresentationKind.FASTA],
                candidate_types=[CandidateType.RNA],
            ),
        ],
    },
    "scgpt": {
        "modalities": [ScientificModality.CELL_STATE],
        "candidate_types": [CandidateType.CELL_STATE, CandidateType.CUSTOM],
        "representations": [RepresentationKind.CELL_EXPRESSION, RepresentationKind.CUSTOM],
        "feature_spaces": ["scgpt-cell-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.CELL_STATE,
                feature_space="scgpt-cell-v1",
                representation_kinds=[
                    RepresentationKind.CELL_EXPRESSION,
                    RepresentationKind.CUSTOM,
                ],
                candidate_types=[
                    CandidateType.CELL_STATE,
                    CandidateType.CUSTOM,
                ],
            ),
        ],
    },
    "qhnet-source": {
        "modalities": [ScientificModality.ELECTRONIC_STRUCTURE],
        "candidate_types": [CandidateType.SMALL_MOLECULE, CandidateType.CUSTOM],
        "representations": [
            RepresentationKind.SDF,
            RepresentationKind.XYZ,
        ],
        "feature_spaces": ["qhnet-hamiltonian-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.ELECTRONIC_STRUCTURE,
                feature_space="qhnet-hamiltonian-v1",
                representation_kinds=[
                    RepresentationKind.SDF,
                    RepresentationKind.XYZ,
                ],
                candidate_types=[CandidateType.SMALL_MOLECULE, CandidateType.CUSTOM],
            ),
        ],
    },
    "pyscf": {
        "modalities": [ScientificModality.ELECTRONIC_STRUCTURE],
        "candidate_types": [
            CandidateType.SMALL_MOLECULE,
            CandidateType.CUSTOM,
        ],
        "representations": [
            RepresentationKind.XYZ,
            RepresentationKind.SDF,
        ],
        "feature_spaces": ["pyscf-orbital-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.ELECTRONIC_STRUCTURE,
                feature_space="pyscf-orbital-v1",
                representation_kinds=[
                    RepresentationKind.XYZ,
                    RepresentationKind.SDF,
                ],
                candidate_types=[
                    CandidateType.SMALL_MOLECULE,
                    CandidateType.CUSTOM,
                ],
            )
        ],
    },
    "uma": {
        # The default ``omat`` task is a periodic materials potential.  A
        # separate, task-aware capability is constructed for ``omol`` below.
        "modalities": [ScientificModality.CRYSTAL_MATERIAL],
        "candidate_types": [
            CandidateType.CRYSTAL,
            CandidateType.COMPOSITION,
            CandidateType.ALLOY,
            CandidateType.CATALYST,
            CandidateType.BATTERY_MATERIAL,
        ],
        "representations": [
            RepresentationKind.CIF,
            RepresentationKind.POSCAR,
        ],
        "feature_spaces": ["uma-atomic-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.CRYSTAL_MATERIAL,
                feature_space="uma-atomic-v1",
                representation_kinds=[
                    RepresentationKind.CIF,
                    RepresentationKind.POSCAR,
                ],
                candidate_types=[
                    CandidateType.CRYSTAL,
                    CandidateType.COMPOSITION,
                    CandidateType.ALLOY,
                    CandidateType.CATALYST,
                    CandidateType.BATTERY_MATERIAL,
                ],
            ),
        ],
        "metadata": {"task_name": "omat"},
    },
    "mattersim": {
        "modalities": [ScientificModality.CRYSTAL_MATERIAL],
        "candidate_types": [
            CandidateType.CRYSTAL,
            CandidateType.COMPOSITION,
            CandidateType.ALLOY,
            CandidateType.CATALYST,
            CandidateType.BATTERY_MATERIAL,
        ],
        "representations": [
            RepresentationKind.CIF,
            RepresentationKind.POSCAR,
        ],
        "feature_spaces": ["mattersim-atomic-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.CRYSTAL_MATERIAL,
                feature_space="mattersim-atomic-v1",
                representation_kinds=[
                    RepresentationKind.CIF,
                    RepresentationKind.POSCAR,
                ],
                candidate_types=[
                    CandidateType.CRYSTAL,
                    CandidateType.COMPOSITION,
                    CandidateType.ALLOY,
                    CandidateType.CATALYST,
                    CandidateType.BATTERY_MATERIAL,
                ],
            ),
        ],
    },
    "chgnet": {
        "modalities": [ScientificModality.CRYSTAL_MATERIAL],
        "candidate_types": [
            CandidateType.CRYSTAL,
            CandidateType.COMPOSITION,
            CandidateType.ALLOY,
            CandidateType.CATALYST,
            CandidateType.BATTERY_MATERIAL,
        ],
        "representations": [
            RepresentationKind.CIF,
            RepresentationKind.POSCAR,
        ],
        "feature_spaces": ["chgnet-atomic-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.CRYSTAL_MATERIAL,
                feature_space="chgnet-atomic-v1",
                representation_kinds=[
                    RepresentationKind.CIF,
                    RepresentationKind.POSCAR,
                ],
                candidate_types=[
                    CandidateType.CRYSTAL,
                    CandidateType.COMPOSITION,
                    CandidateType.ALLOY,
                    CandidateType.CATALYST,
                    CandidateType.BATTERY_MATERIAL,
                ],
            )
        ],
    },
    "chemprop": {
        "modalities": [ScientificModality.MOLECULE_2D],
        "candidate_types": [
            CandidateType.SMALL_MOLECULE,
            CandidateType.CATALYST,
        ],
        "representations": [RepresentationKind.SMILES],
        "feature_spaces": ["chemprop-mpn-v1"],
        "routes": [
            ExpertRoute(
                modality=ScientificModality.MOLECULE_2D,
                feature_space="chemprop-mpn-v1",
                representation_kinds=[RepresentationKind.SMILES],
                candidate_types=[
                    CandidateType.SMALL_MOLECULE,
                    CandidateType.CATALYST,
                ],
            ),
        ],
    },
}


class UnavailableExpertEncoder:
    def __init__(self, descriptor: ExpertDescriptor) -> None:
        self._descriptor = descriptor

    @property
    def descriptor(self) -> ExpertDescriptor:
        return self._descriptor

    def encode(self, request: ExpertFeatureRequest) -> ExpertFeaturePayload:
        raise RuntimeError(f"expert service {self.descriptor.expert_id!r} is not configured")


def _capability_for_component(
    component_id: str,
    environ: Mapping[str, str],
) -> dict:
    capability = _EXPERT_CAPABILITIES[component_id]
    if component_id != "uma":
        return capability

    task_name = environ.get("UMA_TASK_NAME", "omat").strip().lower()
    if task_name == "omat":
        return capability
    if task_name == "omol":
        return {
            "modalities": [ScientificModality.MOLECULE_3D],
            "candidate_types": [CandidateType.SMALL_MOLECULE],
            "representations": [
                RepresentationKind.XYZ,
                RepresentationKind.EXTXYZ,
                RepresentationKind.SDF,
            ],
            "feature_spaces": ["uma-atomic-v1"],
            "routes": [
                ExpertRoute(
                    modality=ScientificModality.MOLECULE_3D,
                    feature_space="uma-atomic-v1",
                    representation_kinds=[
                        RepresentationKind.XYZ,
                        RepresentationKind.EXTXYZ,
                        RepresentationKind.SDF,
                    ],
                    candidate_types=[CandidateType.SMALL_MOLECULE],
                )
            ],
            "metadata": {"task_name": "omol"},
        }
    raise ValueError(
        "UMA_TASK_NAME must be 'omat' or 'omol'; other UMA tasks need an explicit reviewed route"
    )


def build_expert_registry_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    include_unconfigured: bool = True,
) -> ExpertRegistry:
    values = environ if environ is not None else os.environ
    manifest = load_integration_manifest()
    registry = ExpertRegistry()
    for component in manifest.components:
        if component.component_id not in _EXPERT_CAPABILITIES:
            continue
        if component.api is None or component.api.protocol != "expert-feature-v1":
            continue
        capability = _capability_for_component(component.component_id, values)
        base_url = values.get(component.api.base_url_env)
        weight_revision = _component_weight_revision(
            component,
            values,
            required=bool(base_url),
        )
        parameters_hash = _optional_sha256(
            values,
            f"{component.api.base_url_env.removesuffix('_API_URL')}_RUNTIME_PARAMETERS_HASH",
        )
        descriptor = _descriptor(
            component,
            capability,
            available=bool(base_url),
            weight_revision=weight_revision,
            parameters_hash=parameters_hash,
        )
        if base_url:
            token_name = f"{component.api.base_url_env[:-4]}_TOKEN"
            token = values.get(token_name)
            headers = {"Authorization": f"Bearer {token}"} if token else None
            registry.register(
                HttpExpertEncoder(
                    descriptor,
                    base_url,
                    headers=headers,
                    allow_insecure_http=_truthy(values.get("DISCOVERY_ALLOW_INSECURE_HTTP")),
                )
            )
        elif include_unconfigured:
            registry.register(UnavailableExpertEncoder(descriptor))
    return registry


def _descriptor(
    component: IntegrationComponent,
    capability: dict,
    *,
    available: bool,
    weight_revision: str | None,
    parameters_hash: str | None,
) -> ExpertDescriptor:
    source_revision = component.source.revision if component.source is not None else "remote"
    version = component.install.version or (
        component.source.release if component.source is not None else None
    ) or "remote"
    return ExpertDescriptor(
        expert_id=component.component_id,
        display_name=component.display_name,
        adapter_version="1.0.0",
        modalities=capability["modalities"],
        supported_candidate_types=capability["candidate_types"],
        supported_representations=capability["representations"],
        feature_spaces=capability["feature_spaces"],
        routes=capability["routes"],
        available=available,
        metadata={
            "component_version": version or "unversioned",
            "source_revision": source_revision,
            "model_version": version,
            "code_revision": source_revision,
            "weight_revision": weight_revision,
            "parameters_hash": parameters_hash,
            "base_url_env": component.api.base_url_env,
            **capability.get("metadata", {}),
        },
    )


def _component_weight_revision(
    component: IntegrationComponent,
    environ: Mapping[str, str],
    *,
    required: bool,
) -> str | None:
    if component.api is None:
        return None
    env_name = f"{component.api.base_url_env.removesuffix('_API_URL')}_WEIGHT_REVISION"
    configured = environ.get(env_name)
    if configured and configured.strip():
        return configured.strip()
    exact = {item.revision for item in component.weights if item.revision is not None}
    unresolved = [item for item in component.weights if item.revision is None]
    if len(exact) == 1 and not unresolved:
        return next(iter(exact))
    if not component.weights:
        return "no-external-weight"
    if required:
        raise ValueError(
            f"{env_name} is required because {component.component_id!r} uses managed/manual or ambiguous weights"
        )
    return None


def _truthy(value: str | None) -> bool:
    return bool(value and value.strip().lower() in {"1", "true", "yes"})


def _optional_sha256(environ: Mapping[str, str], name: str) -> str | None:
    raw = environ.get(name)
    if raw is None or not raw.strip():
        return None
    value = raw.strip()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


__all__ = ["UnavailableExpertEncoder", "build_expert_registry_from_environment"]
