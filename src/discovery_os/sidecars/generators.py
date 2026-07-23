"""Production-oriented wrappers for the official MatterGen/REINVENT CLIs."""

from __future__ import annotations

import csv
import json
import math
import os
import random
import shutil
import subprocess
import tempfile
import threading
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Literal, Mapping

from discovery_os.crystal_identity import (
    CRYSTAL_IDENTITY_CANONICALIZATION,
    CrystalIdentityError,
    InvalidCrystalGeometryError,
    PymatgenRequiredError,
    canonicalize_crystal_structure,
    classify_crystal_structure_relation,
    exact_file_hash,
    group_crystal_structures,
)
from discovery_os.fusion_schemas import FusionGenerationRequest
from discovery_os.schemas import CandidateRepresentation, RepresentationKind

from .base import LazyModelAdapter, require_module
from .conversions import (
    ase_chemical_system,
    candidate_smiles,
    candidate_to_ase,
    pymatgen_to_cif,
)
from .errors import (
    ModelExecutionError,
    ModelOutputError,
    ModelTimeoutError,
    OptionalDependencyError,
)
from .types import GeneratedBatch, GeneratedCandidateData
from .weight_binding import directory_inventory_sha256, sha256_file

_PERIODIC_TABLE_SYMBOLS = (
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
)
_ATOMIC_NUMBER_BY_SYMBOL = {
    symbol: atomic_number
    for atomic_number, symbol in enumerate(_PERIODIC_TABLE_SYMBOLS, start=1)
}
_NOBLE_GASES = frozenset({"He", "Ne", "Ar", "Kr", "Xe", "Rn", "Og"})
_EXPLICIT_MODEL_CARD_EXCLUSIONS = frozenset({"Tc", "Pm"})


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
    # Public MatterGen 1.0 checkpoint contract.  These names and condition
    # modules come from the released checkpoint inventory; do not infer a
    # condition from a similarly named local directory.
    _KNOWN_CHECKPOINT_CONDITIONS: Mapping[str, frozenset[str]] = {
        "mattergen_base": frozenset(),
        "mp_20_base": frozenset(),
        "chemical_system": frozenset({"chemical_system"}),
        "space_group": frozenset({"space_group"}),
        "dft_mag_density": frozenset({"dft_mag_density"}),
        "dft_band_gap": frozenset({"dft_band_gap"}),
        "ml_bulk_modulus": frozenset({"ml_bulk_modulus"}),
        "dft_mag_density_hhi_score": frozenset({"dft_mag_density", "hhi_score"}),
        "chemical_system_energy_above_hull": frozenset(
            {"chemical_system", "energy_above_hull"}
        ),
    }

    def __init__(
        self,
        *,
        pretrained_name: str = "mattergen_base",
        checkpoint_path: str | None = None,
        objective_map: dict[str, str] | None = None,
        supported_condition_names: Iterable[str] | None = None,
        guidance_max: float = 4.0,
        max_cif_bytes: int = 20_000,
        deduplication_max_generation_rounds: int = 4,
        minimum_distance_angstrom: float = 0.5,
        matcher_ltol: float = 0.02,
        matcher_stol: float = 0.05,
        matcher_angle_tol: float = 1.0,
        device: str = "auto",
    ) -> None:
        super().__init__(device=device)
        if not pretrained_name.strip():
            raise ValueError("pretrained_name must not be blank")
        if not 0.0 <= guidance_max <= 100.0:
            raise ValueError("guidance_max must be between 0 and 100")
        if not 1 <= deduplication_max_generation_rounds <= 16:
            raise ValueError(
                "deduplication_max_generation_rounds must be between 1 and 16"
            )
        if not 0.0 < minimum_distance_angstrom <= 10.0:
            raise ValueError("minimum_distance_angstrom must be between 0 and 10")
        if not 0.0 < matcher_ltol <= 1.0 or not 0.0 < matcher_stol <= 1.0:
            raise ValueError(
                "StructureMatcher length/site tolerances must be between 0 and 1"
            )
        if not 0.0 < matcher_angle_tol <= 180.0:
            raise ValueError("matcher_angle_tol must be between 0 and 180")
        self.pretrained_name = pretrained_name.strip()
        self.checkpoint_path = checkpoint_path
        self.objective_map = dict(objective_map or {})
        explicit_conditions = (
            None
            if supported_condition_names is None
            else frozenset(item.strip() for item in supported_condition_names)
        )
        if explicit_conditions is not None:
            if "" in explicit_conditions:
                raise ValueError(
                    "supported_condition_names cannot contain a blank name"
                )
            unknown_conditions = explicit_conditions - self._CONDITION_NAMES
            if unknown_conditions:
                raise ValueError(
                    "unsupported MatterGen condition declaration(s): "
                    + ", ".join(sorted(unknown_conditions))
                )
        known_conditions = self._KNOWN_CHECKPOINT_CONDITIONS.get(self.pretrained_name)
        if known_conditions is not None:
            if (
                explicit_conditions is not None
                and explicit_conditions != known_conditions
            ):
                raise ValueError(
                    f"official MatterGen checkpoint {self.pretrained_name!r} has the exact "
                    "condition allowlist "
                    f"{sorted(known_conditions)!r}; it cannot be widened or changed"
                )
            self.supported_condition_names = known_conditions
            self.condition_contract_source = "official-checkpoint-allowlist"
        else:
            # Unknown/custom checkpoints are unconditional unless their adapter
            # configuration explicitly declares the condition modules that were
            # trained and packaged with those exact weights.
            self.supported_condition_names = explicit_conditions or frozenset()
            self.condition_contract_source = (
                "explicit-custom-checkpoint-declaration"
                if explicit_conditions is not None
                else "undeclared-custom-checkpoint-unconditional-only"
            )
        self.guidance_max = guidance_max
        self.max_cif_bytes = max_cif_bytes
        self.deduplication_max_generation_rounds = deduplication_max_generation_rounds
        self.minimum_distance_angstrom = minimum_distance_angstrom
        self.matcher_ltol = matcher_ltol
        self.matcher_stol = matcher_stol
        self.matcher_angle_tol = matcher_angle_tol
        self._inference_lock = threading.Lock()
        self._session_identity_structures: dict[str, list[Any]] = {}
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
                raise ModelExecutionError(
                    "configured MatterGen checkpoint path does not exist"
                )
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
        count = request.run_config.candidate_count
        controls = request.run_config.generation_controls
        conditions, condition_warnings = self._conditions(request)
        # Reject an unsupported condition contract before loading a large
        # checkpoint or allocating model memory.
        generator = self._ensure_loaded()
        requested_controls = controls.model_dump(mode="json")
        diffusion_guidance_factor = (
            round(controls.alpha * self.guidance_max, 8) if conditions else 0.0
        )
        ignored_controls = ["temperature", "mutation_strength", "diversity_strength"]
        if not conditions:
            ignored_controls.append("alpha")
        applied_controls = {
            "conditions": dict(conditions),
            "diffusion_guidance_factor": diffusion_guidance_factor,
            "alpha_to_gamma_scale": self.guidance_max,
            "mapping": "gamma=alpha*alpha_to_gamma_scale when conditioned; otherwise gamma=0",
        }
        warnings = [
            "MatterGen v1 has no parent-structure mutation operator; the parent is lineage "
            "and may only contribute an explicit chemical-system condition.",
            "MatterGen v1 does not expose temperature, mutation_strength, or "
            "diversity_strength; those controls were preserved in provenance but not applied.",
            "MatterGen applied "
            f"diffusion_guidance_factor={diffusion_guidance_factor:g}; requested and applied "
            "controls are recorded separately.",
            *condition_warnings,
        ]
        raw_records: list[dict[str, Any]] = []
        rejected_details: list[str] = []
        raw_structure_count = 0
        parsed_structure_count = 0
        geometry_rejected_count = 0
        canonicalization_rejected_count = 0
        canonicalized_structure_count = 0
        applicability_rejected_count = 0
        condition_rejected_count = 0
        source_atom_count_rejected_count = 0
        primitive_atom_count_rejected_count = 0
        model_card_element_rejected_count = 0
        chemical_system_rejected_count = 0
        space_group_rejected_count = 0
        cross_call_duplicate_rejected_count = 0
        cross_call_ambiguous_comparison_count = 0
        exact_file_hashes: set[str] = set()
        generation_rounds = 0
        grouping = group_crystal_structures(())
        with tempfile.TemporaryDirectory(prefix="discovery-mattergen-") as temporary:
            root = Path(temporary)
            with self._inference_lock:
                search_session_id = getattr(
                    request.run_config,
                    "search_session_id",
                    None,
                )
                if (
                    search_session_id is not None
                    and search_session_id not in self._session_identity_structures
                    and len(self._session_identity_structures) >= 64
                ):
                    self._session_identity_structures.pop(
                        next(iter(self._session_identity_structures))
                    )
                prior_session_structures = list(
                    self._session_identity_structures.get(
                        search_session_id,
                        [],
                    )
                    if search_session_id is not None
                    else []
                )
                generator.properties_to_condition_on = conditions
                generator.diffusion_guidance_factor = diffusion_guidance_factor
                while (
                    len(grouping.groups) < count
                    and generation_rounds < self.deduplication_max_generation_rounds
                ):
                    missing = count - len(grouping.groups)
                    round_index = generation_rounds
                    round_seed = (
                        request.run_config.effective_generator_seed + round_index
                    )
                    try:
                        _seed_mattergen(round_seed)
                        structures = generator.generate(
                            batch_size=missing,
                            num_batches=1,
                            output_dir=str(root / f"output-{round_index}"),
                        )
                    except Exception as exc:
                        raise ModelExecutionError(
                            f"MatterGen generation failed: {type(exc).__name__}: {exc}"
                        ) from exc
                    generation_rounds += 1
                    if len(structures) != missing:
                        raise ModelOutputError(
                            f"MatterGen returned {len(structures)} structures, expected {missing} "
                            f"in generation round {generation_rounds}"
                        )
                    for structure in structures:
                        raw_index = raw_structure_count
                        raw_structure_count += 1
                        try:
                            raw_cif = pymatgen_to_cif(
                                structure,
                                max_bytes=self.max_cif_bytes,
                            )
                        except ModelOutputError as exc:
                            rejected_details.append(
                                f"raw structure {raw_index} could not be serialized/parsed: {exc}"
                            )
                            continue
                        parsed_structure_count += 1
                        source_exact_sha256 = exact_file_hash(raw_cif)
                        exact_file_hashes.add(source_exact_sha256)
                        try:
                            canonical = canonicalize_crystal_structure(
                                structure,
                                minimum_distance_angstrom=self.minimum_distance_angstrom,
                                max_cif_bytes=self.max_cif_bytes,
                            )
                        except PymatgenRequiredError as exc:
                            raise OptionalDependencyError(str(exc)) from exc
                        except InvalidCrystalGeometryError as exc:
                            geometry_rejected_count += 1
                            rejected_details.append(
                                f"raw structure {raw_index} rejected: {type(exc).__name__}: {exc}"
                            )
                            continue
                        except CrystalIdentityError as exc:
                            canonicalization_rejected_count += 1
                            rejected_details.append(
                                f"raw structure {raw_index} rejected: {type(exc).__name__}: {exc}"
                            )
                            continue
                        canonicalized_structure_count += 1
                        rejection = _mattergen_output_contract_rejection(
                            canonical,
                            conditions=conditions,
                        )
                        if rejection is not None:
                            category, reason, detail = rejection
                            if category == "applicability":
                                applicability_rejected_count += 1
                                if reason == "source_atom_count":
                                    source_atom_count_rejected_count += 1
                                elif reason == "primitive_atom_count":
                                    primitive_atom_count_rejected_count += 1
                                elif reason == "model_card_element_domain":
                                    model_card_element_rejected_count += 1
                            else:
                                condition_rejected_count += 1
                                if reason == "chemical_system":
                                    chemical_system_rejected_count += 1
                                elif reason == "space_group":
                                    space_group_rejected_count += 1
                            rejected_details.append(
                                f"raw structure {raw_index} rejected by MatterGen {category} "
                                f"contract ({reason}): {detail}"
                            )
                            continue
                        cross_call_duplicate = False
                        for previous in prior_session_structures:
                            assessment = classify_crystal_structure_relation(
                                previous,
                                canonical,
                                strict_ltol=self.matcher_ltol,
                                strict_stol=self.matcher_stol,
                                strict_angle_tol=self.matcher_angle_tol,
                            )
                            if assessment.hard_deduplication_allowed:
                                cross_call_duplicate = True
                                break
                            if assessment.relation.value == "ambiguous":
                                cross_call_ambiguous_comparison_count += 1
                                rejected_details.append(
                                    "cross-call identity comparison was ambiguous and both "
                                    "structures were preserved: "
                                    + (assessment.reason or "unspecified matcher failure")
                                )
                        if cross_call_duplicate:
                            cross_call_duplicate_rejected_count += 1
                            rejected_details.append(
                                f"raw structure {raw_index} duplicates a previously accepted "
                                f"candidate in search session {search_session_id!r}; replacement requested"
                            )
                            continue
                        raw_records.append(
                            {
                                "raw_index": raw_index,
                                "generation_round": generation_rounds,
                                "generation_seed": round_seed,
                                "raw_cif": raw_cif,
                                "source_exact_sha256": source_exact_sha256,
                                "canonical": canonical,
                                "composition_key": _canonical_composition_key(
                                    canonical
                                ),
                            }
                        )
                    try:
                        grouping = group_crystal_structures(
                            tuple(item["canonical"] for item in raw_records),
                            ltol=self.matcher_ltol,
                            stol=self.matcher_stol,
                            angle_tol=self.matcher_angle_tol,
                        )
                    except PymatgenRequiredError as exc:
                        raise OptionalDependencyError(str(exc)) from exc
                    except CrystalIdentityError as exc:
                        raise ModelOutputError(
                            f"MatterGen crystal deduplication failed: {exc}"
                        ) from exc
                if search_session_id is not None and len(grouping.groups) == count:
                    accepted = [
                        raw_records[item.representative_index]["canonical"]
                        for item in grouping.groups
                    ]
                    self._session_identity_structures.setdefault(
                        search_session_id,
                        [],
                    ).extend(accepted)
        unique_count = len(grouping.groups)
        raw_geometry_valid_count = canonicalized_structure_count
        # The public funnel reports geometry-valid candidates after the
        # crystallographic grouping stage.  The pre-dedup count is retained
        # separately for auditability.
        geometry_valid_count = unique_count
        duplicate_count = len(raw_records) - unique_count
        if unique_count != count:
            raise ModelOutputError(
                "MatterGen could not satisfy the requested crystallographically unique "
                f"candidate count after {generation_rounds} generation rounds: "
                f"requested_samples={count}, raw_model_structures={raw_structure_count}, "
                f"parsed_structures={parsed_structure_count}, "
                f"exact_file_unique={len(exact_file_hashes)}, "
                f"raw_geometry_valid={raw_geometry_valid_count}, "
                f"applicability_rejected={applicability_rejected_count}, "
                f"condition_rejected={condition_rejected_count}, "
                f"geometry_valid={geometry_valid_count}, "
                f"crystallographically_unique={unique_count}, duplicates_removed="
                f"{duplicate_count}"
            )
        funnel = {
            "requested_samples": count,
            "raw_model_structures": raw_structure_count,
            "parsed_structures": parsed_structure_count,
            "exact_file_unique": len(exact_file_hashes),
            "crystallographically_unique": unique_count,
            "geometry_valid": geometry_valid_count,
            "raw_geometry_valid": raw_geometry_valid_count,
            "requested_unique_candidates": count,
            "parse_rejected": raw_structure_count - parsed_structure_count,
            "geometry_rejected": geometry_rejected_count,
            "canonicalization_rejected": canonicalization_rejected_count,
            "applicability_rejected": applicability_rejected_count,
            "condition_rejected": condition_rejected_count,
            "source_atom_count_rejected": source_atom_count_rejected_count,
            "primitive_atom_count_rejected": primitive_atom_count_rejected_count,
            "model_card_element_rejected": model_card_element_rejected_count,
            "chemical_system_rejected": chemical_system_rejected_count,
            "space_group_rejected": space_group_rejected_count,
            "cross_call_duplicate_rejected": cross_call_duplicate_rejected_count,
            "cross_call_ambiguous_comparisons": (
                cross_call_ambiguous_comparison_count
            ),
            "duplicates_removed": duplicate_count,
            "generation_rounds": generation_rounds,
        }
        warnings.append(
            "MatterGen crystal identity funnel: "
            + ", ".join(f"{key}={value}" for key, value in funnel.items())
            + "."
        )
        if rejected_details:
            warnings.extend(rejected_details[:3])
            if len(rejected_details) > 3:
                warnings.append(
                    f"{len(rejected_details) - 3} additional invalid raw structures were rejected"
                )
        if duplicate_count:
            warnings.append(
                f"StructureMatcher removed {duplicate_count} crystallographic duplicate(s); "
                "deterministic replacement generation retained the requested unique count."
            )
        representatives = [
            raw_records[item.representative_index] for item in grouping.groups
        ]
        candidates = tuple(
            GeneratedCandidateData(
                name=f"MatterGen candidate {index + 1}",
                representations=(
                    CandidateRepresentation(
                        kind=RepresentationKind.CIF,
                        # Preserve the direct MatterGen structure serialization as
                        # the authoritative candidate.  Canonicalization is used
                        # for identity and deduplication, not to silently replace
                        # the generated geometry delivered to downstream experts.
                        value=record["raw_cif"],
                        media_type="chemical/x-cif",
                        format_version="CIF",
                        canonical=False,
                        metadata={
                            "source_entry": f"generated-{record['raw_index']}.cif",
                            "source_exact_sha256": record["source_exact_sha256"],
                            "identity_structure_sha256": record[
                                "canonical"
                            ].identity_structure_hash,
                            "identity_canonicalization": (
                                CRYSTAL_IDENTITY_CANONICALIZATION
                            ),
                        },
                    ),
                ),
                attributes={
                    "mattergen_pretrained_name": self.pretrained_name,
                    "conditions": conditions,
                    # ``generation_controls`` remains as a compatibility alias
                    # for consumers written before the requested/applied split.
                    "generation_controls": requested_controls,
                    "requested_generation_controls": requested_controls,
                    "applied_generation_controls": applied_controls,
                    "ignored_generation_controls": ignored_controls,
                    "composition_key": record["composition_key"],
                    "crystal_identity": {
                        "identity_structure_sha256": record[
                            "canonical"
                        ].identity_structure_hash,
                        "identity_canonicalization": (
                            CRYSTAL_IDENTITY_CANONICALIZATION
                        ),
                        "source_atom_count": record["canonical"].source_atom_count,
                        "primitive_atom_count": record[
                            "canonical"
                        ].primitive_atom_count,
                        "conventional_atom_count": record[
                            "canonical"
                        ].conventional_atom_count,
                        "space_group_symbol": record["canonical"].space_group_symbol,
                        "space_group_number": record["canonical"].space_group_number,
                    },
                    "generation_funnel": funnel,
                    "generation_funnel_hashes": {
                        # Structured hashes let an orchestrator compute exact
                        # uniqueness across independently generated profiles
                        # without reopening temporary MatterGen output files.
                        "exact_file_sha256s": sorted(exact_file_hashes),
                    },
                },
                provenance={
                    "adapter": "mattergen-crystal-generator-v1",
                    "raw_generation_stream_position": record["raw_index"],
                    "raw_generation_round": record["generation_round"],
                    "raw_generation_seed": record["generation_seed"],
                    "source_exact_sha256": record["source_exact_sha256"],
                    "identity_structure_sha256": record[
                        "canonical"
                    ].identity_structure_hash,
                    "identity_canonicalization": CRYSTAL_IDENTITY_CANONICALIZATION,
                    "deduplication": funnel,
                    "requested_generation_controls": requested_controls,
                    "applied_generation_controls": applied_controls,
                    "ignored_generation_controls": ignored_controls,
                },
            )
            for index, record in enumerate(representatives)
        )
        return GeneratedBatch(candidates=candidates, warnings=tuple(warnings))

    def provenance_parameters(self) -> dict[str, Any]:
        return {
            "runtime_class": type(self).__name__,
            "pretrained_name": self.pretrained_name,
            "checkpoint_inventory_sha256": self.checkpoint_inventory_sha256,
            "objective_map": dict(sorted(self.objective_map.items())),
            "supported_condition_names": sorted(self.supported_condition_names),
            "condition_contract_source": self.condition_contract_source,
            "guidance_max": self.guidance_max,
            "guidance_mapping": "gamma=alpha*guidance_max when conditioned; otherwise gamma=0",
            "max_cif_bytes": self.max_cif_bytes,
            "deduplication_max_generation_rounds": self.deduplication_max_generation_rounds,
            "minimum_distance_angstrom": self.minimum_distance_angstrom,
            "matcher_ltol": self.matcher_ltol,
            "matcher_stol": self.matcher_stol,
            "matcher_angle_tol": self.matcher_angle_tol,
            "requested_device": self._requested_device,
        }

    def _conditions(
        self, request: FusionGenerationRequest
    ) -> tuple[dict[str, str | float | int], list[str]]:
        conditions: dict[str, str | float | int] = {}
        warnings: list[str] = []
        objectives = list(request.goal.objectives)
        for objective in objectives:
            mapped = self.objective_map.get(
                objective.property_name, objective.property_name
            )
            if mapped not in self._CONDITION_NAMES or objective.target_value is None:
                continue
            conditions[mapped] = _condition_value(
                mapped,
                objective.target_value,
                unit=getattr(objective, "unit", None),
            )
        proposal = request.revision_proposal
        if proposal is not None:
            warnings.append(
                "MatterGen does not accept the raw unified latent; only explicit supported "
                "revision desired_changes with concrete target values were translated."
            )
            for change in proposal.desired_changes:
                if change.property_name is None:
                    continue
                mapped = self.objective_map.get(
                    change.property_name, change.property_name
                )
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
                matching_objective = next(
                    (
                        objective
                        for objective in objectives
                        if self.objective_map.get(
                            objective.property_name, objective.property_name
                        )
                        == mapped
                    ),
                    None,
                )
                value = _condition_value(
                    mapped,
                    change.target_value,
                    unit=(
                        change.unit
                        if getattr(change, "unit", None) is not None
                        else getattr(matching_objective, "unit", None)
                        if matching_objective is not None
                        else None
                    ),
                )
                if mapped in conditions and conditions[mapped] != value:
                    warnings.append(
                        f"revision target for {mapped!r} overrides the original goal target "
                        "for this iteration"
                    )
                conditions[mapped] = value
        if "chemical_system" in self.supported_condition_names:
            if "chemical_system" not in conditions:
                parent_atoms = candidate_to_ase(request.parent_candidate)
                conditions["chemical_system"] = ase_chemical_system(parent_atoms)
        unsupported = frozenset(conditions) - self.supported_condition_names
        if unsupported:
            if self.pretrained_name in self._KNOWN_CHECKPOINT_CONDITIONS:
                contract = f"official checkpoint allowlist={sorted(self.supported_condition_names)!r}"
            else:
                contract = (
                    "custom checkpoint requires explicit supported_condition_names; "
                    f"declared={sorted(self.supported_condition_names)!r}"
                )
            raise ModelExecutionError(
                f"MatterGen checkpoint {self.pretrained_name!r} cannot apply requested "
                f"condition(s) {sorted(unsupported)!r}; {contract}"
            )
        if "chemical_system" in conditions:
            # Parse before inference so requests outside the released model-card
            # domain never consume GPU time or masquerade as supported output.
            requested_system = _validated_chemical_system(
                conditions["chemical_system"]
            )
            allowed_system = _declared_goal_chemical_system(request.goal)
            if allowed_system is not None and requested_system != allowed_system:
                raise ModelExecutionError(
                    "MatterGen chemical_system differs from the immutable hard goal "
                    "constraint; expanded-system/dopant branches require explicit user opt-in"
                )
        if "space_group" in conditions:
            conditions["space_group"] = _validated_space_group(
                conditions["space_group"]
            )
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
                seed_file.write_text(
                    candidate_smiles(request.parent_candidate) + "\n", encoding="utf-8"
                )
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
                json.dumps(
                    config, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                ),
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
    def __init__(
        self, returncode: int, stdout: bytes, stderr: bytes, *, truncated: bool
    ) -> None:
        self.returncode = returncode
        suffix = " [log truncated]" if truncated else ""
        self.stdout_text = stdout.decode("utf-8", errors="replace").strip() + suffix
        self.stderr_text = stderr.decode("utf-8", errors="replace").strip() + suffix


def _seed_mattergen(seed: int) -> None:
    """Apply the request's generator seed to every RNG MatterGen uses."""

    numpy = require_module(
        "numpy", install_hint="install MatterGen's pinned NumPy build"
    )
    torch = require_module(
        "torch", install_hint="install MatterGen's pinned PyTorch build"
    )
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
        raise ModelExecutionError(
            f"could not start model CLI: {type(exc).__name__}: {exc}"
        ) from exc
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
        raise ModelTimeoutError(
            f"model CLI exceeded its {timeout:g} second process timeout"
        ) from exc
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
        raise ModelOutputError(
            "MatterGen did not produce generated_crystals_cif.zip"
        ) from exc
    if size <= 0 or size > max_archive_bytes:
        raise ModelOutputError(
            "MatterGen CIF archive is empty or exceeds the configured size limit"
        )
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
                if (
                    pure.is_absolute()
                    or ".." in pure.parts
                    or pure.suffix.lower() != ".cif"
                ):
                    raise ModelOutputError(
                        "MatterGen archive contains an unsafe or non-CIF entry"
                    )
                if entry.file_size <= 0 or entry.file_size > max_cif_bytes:
                    raise ModelOutputError(
                        "MatterGen CIF exceeds the per-representation size limit"
                    )
                if (
                    entry.compress_size == 0
                    or entry.file_size / entry.compress_size > 200
                ):
                    raise ModelOutputError(
                        "MatterGen archive has a suspicious compression ratio"
                    )
                total += entry.file_size
                if total > expected_count * max_cif_bytes:
                    raise ModelOutputError(
                        "MatterGen archive exceeds the uncompressed size limit"
                    )
                raw = archive.read(entry)
                if len(raw) != entry.file_size:
                    raise ModelOutputError(
                        "MatterGen archive entry size changed while reading"
                    )
                output.append((entry.filename, raw.decode("utf-8")))
            return output
    except ModelOutputError:
        raise
    except (OSError, UnicodeError, zipfile.BadZipFile) as exc:
        raise ModelOutputError(
            f"MatterGen returned an invalid CIF archive: {type(exc).__name__}: {exc}"
        ) from exc


def _read_reinvent_csv(
    path: Path, *, requested_count: int, max_output_bytes: int
) -> list[str]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ModelOutputError(
            "REINVENT did not produce the configured sampling CSV"
        ) from exc
    if size <= 0 or size > max_output_bytes:
        raise ModelOutputError(
            "REINVENT sampling CSV is empty or exceeds the size limit"
        )
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ModelOutputError("REINVENT sampling CSV has no header")
            smiles_name = next(
                (
                    name
                    for name in reader.fieldnames
                    if name.strip().lower() in {"smiles", "smile"}
                ),
                None,
            )
            if smiles_name is None:
                raise ModelOutputError("REINVENT sampling CSV has no SMILES column")
            raw_values = [row.get(smiles_name, "").strip() for row in reader]
    except ModelOutputError:
        raise
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ModelOutputError(
            f"REINVENT returned an invalid CSV: {type(exc).__name__}: {exc}"
        ) from exc
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


def _condition_value(
    name: str,
    value: Any,
    *,
    unit: str | None,
) -> str | float | int:
    """Validate and normalize one public MatterGen condition.

    Numeric strings are deliberately rejected.  A target such as ``1500`` is
    unsafe without its declared unit because passing meV as eV would silently
    move the diffusion condition by three orders of magnitude.
    """

    if name == "chemical_system":
        if unit not in {None, ""}:
            raise ModelExecutionError(
                "MatterGen chemical_system must not declare a numeric unit"
            )
        _validated_chemical_system(value)
        return str(value).strip()
    if name == "space_group":
        if not _dimensionless_unit(unit):
            raise ModelExecutionError(
                "MatterGen space_group must be dimensionless"
            )
        return _validated_space_group(value)

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelExecutionError(
            f"MatterGen numeric condition {name!r} requires a finite number, not a string"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ModelExecutionError(f"MatterGen condition {name!r} is not finite")

    normalized_unit = _normalized_condition_unit(unit)
    contracts: dict[str, dict[str, float]] = {
        "energy_above_hull": {"ev/atom": 1.0, "mev/atom": 0.001},
        "dft_band_gap": {"ev": 1.0, "mev": 0.001},
        "ml_bulk_modulus": {"gpa": 1.0, "mpa": 0.001},
        "dft_mag_density": {
            "ub/angstrom^3": 1.0,
            "u_b/angstrom^3": 1.0,
            "mub/angstrom^3": 1.0,
            "mu_b/angstrom^3": 1.0,
            "bohrmagneton/angstrom^3": 1.0,
        },
    }
    if name == "hhi_score":
        if not _dimensionless_unit(unit):
            raise ModelExecutionError("MatterGen hhi_score must be dimensionless")
        return numeric
    allowed = contracts.get(name)
    if allowed is None:
        raise ModelExecutionError(f"unknown MatterGen condition contract {name!r}")
    if normalized_unit not in allowed:
        expected = ", ".join(sorted(allowed))
        raise ModelExecutionError(
            f"MatterGen condition {name!r} has incompatible or missing unit {unit!r}; "
            f"accepted normalized units are: {expected}"
        )
    return numeric * allowed[normalized_unit]


def _normalized_condition_unit(unit: str | None) -> str | None:
    if unit is None:
        return None
    normalized = unit.strip().replace("Å", "angstrom").replace("Å", "angstrom")
    normalized = normalized.replace("μ", "u").replace("µ", "u")
    normalized = normalized.replace(" ", "").casefold()
    return normalized or None


def _dimensionless_unit(unit: str | None) -> bool:
    normalized = _normalized_condition_unit(unit)
    return normalized in {None, "1", "dimensionless", "unitless"}


def _validated_chemical_system(value: Any) -> frozenset[str]:
    """Validate a requested element set against the released model-card domain."""

    if not isinstance(value, str):
        raise ModelExecutionError(
            "MatterGen chemical_system must be a hyphen-separated string"
        )
    normalized = value.strip()
    tokens = normalized.split("-")
    if (
        not normalized
        or any(not token or token.strip() != token for token in tokens)
        or len(set(tokens)) != len(tokens)
    ):
        raise ModelExecutionError(
            "MatterGen chemical_system must contain unique element symbols separated by '-'"
        )
    unknown = sorted(
        symbol for symbol in tokens if symbol not in _ATOMIC_NUMBER_BY_SYMBOL
    )
    if unknown:
        raise ModelExecutionError(
            f"MatterGen chemical_system contains unknown element symbol(s): {unknown!r}"
        )
    excluded = _model_card_excluded_symbols(tokens)
    if excluded:
        raise ModelExecutionError(
            "MatterGen chemical_system is outside the released model-card domain; "
            "noble gases, Tc/Pm, and elements with atomic number greater than 84 are "
            f"excluded (requested={excluded!r})"
        )
    return frozenset(tokens)


def _declared_goal_chemical_system(goal: Any) -> frozenset[str] | None:
    declared: set[frozenset[str]] = set()
    for constraint in getattr(goal, "constraints", []):
        if (
            getattr(constraint, "hard", False)
            and getattr(constraint, "property_name", None)
            in {"chemical_system", "allowed_chemical_system"}
            and getattr(constraint, "operator", None) == "eq"
            and isinstance(getattr(constraint, "value", None), str)
        ):
            declared.add(_validated_chemical_system(constraint.value))
    if len(declared) > 1:
        raise ModelExecutionError(
            "goal contains conflicting hard chemical-system constraints"
        )
    return next(iter(declared)) if declared else None


def _model_card_excluded_symbols(symbols: Iterable[str]) -> list[str]:
    return sorted(
        symbol
        for symbol in symbols
        if symbol in _NOBLE_GASES
        or symbol in _EXPLICIT_MODEL_CARD_EXCLUSIONS
        or _ATOMIC_NUMBER_BY_SYMBOL[symbol] > 84
    )


def _validated_space_group(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 230:
        raise ModelExecutionError(
            "MatterGen space_group must be an integer from 1 through 230"
        )
    return value


def _canonical_element_symbols(canonical: Any) -> frozenset[str] | None:
    for attribute in ("primitive_structure", "canonical_structure"):
        structure = getattr(canonical, attribute, None)
        composition = getattr(structure, "composition", None)
        elements = getattr(composition, "elements", None)
        if elements is not None:
            symbols = frozenset(
                str(getattr(element, "symbol", element)).strip() for element in elements
            )
            if symbols and all(
                symbol in _ATOMIC_NUMBER_BY_SYMBOL for symbol in symbols
            ):
                return symbols
    fingerprint = getattr(canonical, "fingerprint", None)
    if isinstance(fingerprint, Mapping):
        composition = fingerprint.get("composition")
        if isinstance(composition, Mapping):
            symbols = frozenset(str(symbol).strip() for symbol in composition)
            if symbols and all(
                symbol in _ATOMIC_NUMBER_BY_SYMBOL for symbol in symbols
            ):
                return symbols
    return None


def _canonical_composition_key(canonical: Any) -> str | None:
    explicit = getattr(canonical, "composition_key", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    for attribute in ("primitive_structure", "canonical_structure"):
        structure = getattr(canonical, attribute, None)
        composition = getattr(structure, "composition", None)
        reduced_formula = getattr(composition, "reduced_formula", None)
        if isinstance(reduced_formula, str) and reduced_formula.strip():
            return reduced_formula.strip()
    return None


def _mattergen_output_contract_rejection(
    canonical: Any,
    *,
    conditions: Mapping[str, str | float | int],
) -> tuple[str, str, str] | None:
    """Return a fail-closed applicability/condition rejection, if any."""

    source_atom_count = getattr(canonical, "source_atom_count", None)
    if not isinstance(source_atom_count, int) or source_atom_count < 1:
        return (
            "applicability",
            "source_atom_count",
            "generated unit-cell atom count could not be verified",
        )
    if source_atom_count > 20:
        return (
            "applicability",
            "source_atom_count",
            f"generated unit-cell atom count {source_atom_count} exceeds the released limit of 20",
        )
    primitive_atom_count = getattr(canonical, "primitive_atom_count", None)
    if not isinstance(primitive_atom_count, int) or primitive_atom_count < 1:
        return (
            "applicability",
            "primitive_atom_count",
            "primitive atom count could not be verified",
        )
    if primitive_atom_count > 20:
        return (
            "applicability",
            "primitive_atom_count",
            f"primitive atom count {primitive_atom_count} exceeds the released limit of 20",
        )
    actual_symbols = _canonical_element_symbols(canonical)
    if actual_symbols is None:
        return (
            "applicability",
            "model_card_element_domain",
            "generated primitive composition could not be verified",
        )
    excluded_symbols = _model_card_excluded_symbols(actual_symbols)
    if excluded_symbols:
        return (
            "applicability",
            "model_card_element_domain",
            "generated structure contains element(s) outside the released model-card "
            f"domain: {excluded_symbols!r}",
        )
    requested_chemical_system = conditions.get("chemical_system")
    if requested_chemical_system is not None:
        requested_symbols = _validated_chemical_system(requested_chemical_system)
        if actual_symbols != requested_symbols:
            return (
                "condition",
                "chemical_system",
                f"requested={sorted(requested_symbols)!r}, generated={sorted(actual_symbols)!r}",
            )
    requested_space_group = conditions.get("space_group")
    if requested_space_group is not None:
        requested_number = _validated_space_group(requested_space_group)
        actual_number = getattr(canonical, "space_group_number", None)
        if actual_number != requested_number:
            return (
                "condition",
                "space_group",
                f"requested={requested_number}, generated={actual_number!r}",
            )
    return None


__all__ = ["MatterGenGenerator", "ReinventGenerator"]
