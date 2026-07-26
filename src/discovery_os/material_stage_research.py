"""Research-backed query and MCP contracts for the five material stages.

The bibliography in this module is not an implementation claim.  Each basis
entry records the specific executable constraint derived from a primary paper
or official specification.  Numerical decisions remain with the runtime
validator named by :mod:`discovery_os.validation_evidence`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from .literature_rag import (
    LiteratureQueryBlueprint,
    McpStructuredRecordContract,
)
from .material_domains import MaterialEvidenceStage
from .schemas import Identifier, MaterialField, NonEmptyText, StrictSchema


class StageResearchBasis(StrictSchema):
    basis_id: NonEmptyText
    source_url: NonEmptyText
    implementation_effect: NonEmptyText


class StageQueryIntent(StrictSchema):
    intent_id: Identifier
    objective: NonEmptyText
    query_terms: list[NonEmptyText] = Field(min_length=2)
    expected_record_types: list[Identifier] = Field(min_length=1)
    explicit_negative_or_null_evidence: bool = False

    @model_validator(mode="after")
    def _intent_is_unique(self) -> "StageQueryIntent":
        if len(self.query_terms) != len(set(self.query_terms)):
            raise ValueError("stage query terms must be unique")
        if len(self.expected_record_types) != len(
            set(self.expected_record_types)
        ):
            raise ValueError("stage query record types must be unique")
        return self


class StageMcpPolicy(StrictSchema):
    scope_argument: Literal[
        "generation_scope",
        "identity_scope",
        "mlip_scope",
        "relaxation_scope",
        "dft_scope",
    ]
    allowed_record_types: list[Identifier] = Field(min_length=1)
    required_record_fields: list[Identifier] = Field(
        default_factory=lambda: [
            "source_id",
            "title",
            "record_type",
            "support_text",
            "provenance",
            "stage_metadata",
        ]
    )
    required_provenance_fields: list[Identifier] = Field(
        default_factory=lambda: [
            "provider",
            "provider_version",
            "snapshot_id",
            "source_locator",
            "retrieved_at",
            "request_hash",
            "record_hash",
        ]
    )
    required_stage_metadata_fields: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def _mcp_policy_is_unique(self) -> "StageMcpPolicy":
        for label, values in (
            ("record types", self.allowed_record_types),
            ("record fields", self.required_record_fields),
            ("provenance fields", self.required_provenance_fields),
            ("stage metadata fields", self.required_stage_metadata_fields),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"stage MCP {label} must be unique")
        return self

    @property
    def accepted_arguments(self) -> list[str]:
        return [
            "query",
            "max_results",
            "from_date",
            "to_date",
            "stage",
            "intent_id",
            "chemical_system",
            "material_field",
            "application_subtype",
            "composition_keys",
            "candidate_refs",
            "record_types",
            self.scope_argument,
        ]

    def runtime_contract(self, stage: MaterialEvidenceStage) -> McpStructuredRecordContract:
        return McpStructuredRecordContract(
            stage=stage,
            allowed_record_types=list(self.allowed_record_types),
            required_record_fields=list(self.required_record_fields),
            required_provenance_fields=list(self.required_provenance_fields),
            required_stage_metadata_fields=list(
                self.required_stage_metadata_fields
            ),
        )


class StageResearchPolicy(StrictSchema):
    policy_id: Identifier
    policy_version: Literal["2.0.0"] = "2.0.0"
    stage: MaterialEvidenceStage
    query_intents: list[StageQueryIntent] = Field(min_length=5, max_length=6)
    mcp: StageMcpPolicy
    research_bases: list[StageResearchBasis] = Field(min_length=4)
    runtime_escalations: list[NonEmptyText] = Field(min_length=2)
    evidence_role: Literal["search-context-never-runtime-property-authority"] = (
        "search-context-never-runtime-property-authority"
    )

    @model_validator(mode="after")
    def _policy_is_closed(self) -> "StageResearchPolicy":
        intent_ids = [item.intent_id for item in self.query_intents]
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("stage research intent identifiers must be unique")
        record_types = {
            record_type
            for item in self.query_intents
            for record_type in item.expected_record_types
        }
        if record_types != set(self.mcp.allowed_record_types):
            raise ValueError(
                "stage MCP record types must exactly cover the query intents"
            )
        basis_ids = [item.basis_id for item in self.research_bases]
        if len(basis_ids) != len(set(basis_ids)):
            raise ValueError("stage research bases must be unique")
        return self


def _intent(
    intent_id: str,
    objective: str,
    query_terms: Sequence[str],
    record_types: Sequence[str],
    *,
    negative: bool = False,
) -> StageQueryIntent:
    return StageQueryIntent(
        intent_id=intent_id,
        objective=objective,
        query_terms=list(query_terms),
        expected_record_types=list(record_types),
        explicit_negative_or_null_evidence=negative,
    )


def _basis(basis_id: str, url: str, effect: str) -> StageResearchBasis:
    return StageResearchBasis(
        basis_id=basis_id,
        source_url=url,
        implementation_effect=effect,
    )


STAGE_RESEARCH_POLICIES: dict[MaterialEvidenceStage, StageResearchPolicy] = {
    "generation_prior": StageResearchPolicy(
        policy_id="material-generation-evidence-v2",
        stage="generation_prior",
        query_intents=[
            _intent(
                "successful_target",
                "Retrieve explicitly characterized target phases and successful recipes.",
                [
                    "synthesized prepared phase pure XRD Rietveld characterized",
                    "target phase precursor operation temperature atmosphere time",
                ],
                ["reported_phase", "synthesis_success"],
            ),
            _intent(
                "impurity_or_partial",
                "Retrieve explicit impurity, secondary-phase, yield, and partial-success evidence.",
                [
                    "impurity secondary phase byproduct phase purity yield",
                    "target obtained with impurities partial reaction",
                ],
                ["synthesis_partial"],
                negative=True,
            ),
            _intent(
                "failed_no_target",
                "Retrieve explicitly reported failed attempts; absence is not failure.",
                [
                    "failed synthesis target not obtained no crystalline product",
                    "amorphous decomposed unsuccessful experiment negative result",
                ],
                ["synthesis_failure"],
                negative=True,
            ),
            _intent(
                "condition_window",
                "Retrieve bounded composition and processing windows.",
                [
                    "composition dopant range precursor route temperature pressure",
                    "time atmosphere solvent pH processing window",
                ],
                ["composition_window", "synthesis_condition_window"],
            ),
            _intent(
                "generator_condition_limit",
                "Retrieve the released generator's supported conditions and scope limits.",
                [
                    "MatterGen model card checkpoint supported conditioning",
                    "chemical system space group band gap bulk modulus hull limitation",
                ],
                ["generator_condition_limit"],
            ),
        ],
        mcp=StageMcpPolicy(
            scope_argument="generation_scope",
            allowed_record_types=[
                "reported_phase",
                "synthesis_success",
                "synthesis_partial",
                "synthesis_failure",
                "composition_window",
                "synthesis_condition_window",
                "generator_condition_limit",
            ],
            required_stage_metadata_fields=[
                "chemical_system",
                "composition",
                "outcome",
                "conditions",
                "evidence_polarity",
            ],
        ),
        research_bases=[
            _basis(
                "doi:10.1038/s41586-025-08628-5",
                "https://doi.org/10.1038/s41586-025-08628-5",
                "Only released, checkpoint-supported MatterGen conditions may steer generation.",
            ),
            _basis(
                "doi:10.1038/s41597-019-0224-1",
                "https://doi.org/10.1038/s41597-019-0224-1",
                "Recipes retain targets, precursors, ordered operations, and heating conditions.",
            ),
            _basis(
                "doi:10.1038/s41597-022-01317-2",
                "https://doi.org/10.1038/s41597-022-01317-2",
                "Solution synthesis evidence retains quantities, operations, and conditions.",
            ),
            _basis(
                "doi:10.1038/nature17439",
                "https://doi.org/10.1038/nature17439",
                "Only explicitly recorded failed experiments become negative evidence.",
            ),
            _basis(
                "doi:10.1038/s41586-019-1540-5",
                "https://doi.org/10.1038/s41586-019-1540-5",
                "Publication frequency and record absence cannot become synthesis probability.",
            ),
        ],
        runtime_escalations=[
            "A title-only, uncertain, or outcome-free record cannot create a generator branch.",
            "Unsupported MatterGen condition names are discarded even when literature mentions them.",
        ],
    ),
    "identity_novelty": StageResearchPolicy(
        policy_id="material-identity-evidence-v2",
        stage="identity_novelty",
        query_intents=[
            _intent(
                "exact_formula_alias",
                "Retrieve exact and reduced-formula crystallographic aliases.",
                [
                    "exact formula reduced formula phase alias polymorph",
                    "space group prototype structure type CIF",
                ],
                ["crystallographic_entry", "structure_alias"],
            ),
            _intent(
                "polymorph_and_conditions",
                "Retrieve pressure, temperature, composition, and polymorph context.",
                [
                    "pressure phase temperature phase polymorph transition",
                    "solid solution dopant composition range",
                ],
                ["polymorph", "pressure_temperature_phase"],
            ),
            _intent(
                "disorder_and_occupancy",
                "Retrieve disorder, partial occupancy, vacancy, and site-mixing context.",
                [
                    "disorder partial occupancy site mixing vacancy",
                    "average structure ordered model crystallography",
                ],
                ["disordered_structure"],
            ),
            _intent(
                "federated_structure_records",
                "Retrieve versioned structures from configured databases.",
                [
                    "OPTIMADE structures Materials Project COD crystallography",
                    "database identifier lattice vectors species positions revision",
                ],
                ["database_structure_record"],
            ),
            _intent(
                "identity_method_scope",
                "Retrieve structure-comparison method and tolerance scope.",
                [
                    "Niggli reduction structure matching tolerance species preserving",
                    "primitive supercell scale false crystal identity",
                ],
                ["identity_method_limit"],
            ),
        ],
        mcp=StageMcpPolicy(
            scope_argument="identity_scope",
            allowed_record_types=[
                "crystallographic_entry",
                "structure_alias",
                "polymorph",
                "pressure_temperature_phase",
                "disordered_structure",
                "database_structure_record",
                "identity_method_limit",
            ],
            required_stage_metadata_fields=[
                "database_name",
                "database_entry_id",
                "formula",
                "structure_locator",
                "match_scope",
            ],
        ),
        research_bases=[
            _basis(
                "doi:10.1107/S010876730302186X",
                "https://doi.org/10.1107/S010876730302186X",
                "Identity canonicalization uses tolerance-aware Niggli reduction.",
            ),
            _basis(
                "doi:10.1016/j.commatsci.2012.10.028",
                "https://doi.org/10.1016/j.commatsci.2012.10.028",
                "Structure comparison is delegated to a versioned pymatgen matcher policy.",
            ),
            _basis(
                "doi:10.1038/s41597-021-00974-z",
                "https://doi.org/10.1038/s41597-021-00974-z",
                "Federated provider results retain OPTIMADE version and database metadata.",
            ),
            _basis(
                "doi:10.1093/nar/gkr900",
                "https://doi.org/10.1093/nar/gkr900",
                "COD records are scoped crystallographic references, not exhaustive novelty proof.",
            ),
            _basis(
                "doi:10.1038/s41524-020-00483-4",
                "https://doi.org/10.1038/s41524-020-00483-4",
                "Scaled prototype similarity remains distinct from full crystal identity.",
            ),
        ],
        runtime_escalations=[
            "Every external prefilter structure requires a local strict unscaled recheck.",
            "Provider failure, incomplete pagination, disorder, or absent coordinates yields unknown.",
        ],
    ),
    "mlip_disagreement": StageResearchPolicy(
        policy_id="material-mlip-evidence-v2",
        stage="mlip_disagreement",
        query_intents=[
            _intent(
                "model_training_domain",
                "Retrieve model training level, chemistry, structure, pressure, and temperature scope.",
                [
                    "MatterSim CHGNet model card training data PBE GGA U",
                    "elements temperature pressure structure domain limitation",
                ],
                ["model_card_limit"],
            ),
            _intent(
                "energy_alignment",
                "Retrieve compatible same-composition relative-energy benchmarks.",
                [
                    "same composition relative energy polymorph ranking benchmark",
                    "energy reference alignment eV per atom",
                ],
                ["same_composition_reference", "benchmark_result"],
            ),
            _intent(
                "force_stress_error",
                "Retrieve force and stress benchmark limits on relevant geometry classes.",
                [
                    "force error stress error high stress distorted structure benchmark",
                    "energy force stress unit test set",
                ],
                ["benchmark_result"],
            ),
            _intent(
                "electronic_state_caveat",
                "Retrieve magnetic, charge, correlation, radical, and surface caveats.",
                [
                    "magnetic charge state strongly correlated oxidation limitation",
                    "surface defect polymer molecule out of domain",
                ],
                ["magnetic_charge_caveat", "out_of_domain_case"],
                negative=True,
            ),
            _intent(
                "uncertainty_and_extrapolation",
                "Retrieve validated uncertainty or extrapolation diagnostics.",
                [
                    "interatomic potential uncertainty ensemble disagreement extrapolation grade",
                    "active learning out of distribution calibration",
                ],
                ["uncertainty_method", "out_of_domain_case"],
            ),
        ],
        mcp=StageMcpPolicy(
            scope_argument="mlip_scope",
            allowed_record_types=[
                "model_card_limit",
                "same_composition_reference",
                "benchmark_result",
                "magnetic_charge_caveat",
                "out_of_domain_case",
                "uncertainty_method",
            ],
            required_stage_metadata_fields=[
                "model_id",
                "model_version",
                "limitation_kind",
                "property_scope",
                "evaluation_scope",
            ],
        ),
        research_bases=[
            _basis(
                "arxiv:2405.04967",
                "https://arxiv.org/abs/2405.04967",
                "MatterSim evidence preserves checkpoint and documented evaluation domain.",
            ),
            _basis(
                "doi:10.1038/s42256-023-00716-3",
                "https://doi.org/10.1038/s42256-023-00716-3",
                "CHGNet evidence preserves its GGA/GGA+U training level and charge scope.",
            ),
            _basis(
                "doi:10.1038/s42256-025-01055-1",
                "https://doi.org/10.1038/s42256-025-01055-1",
                "Discovery benchmarking separates stability ranking from raw regression error.",
            ),
            _basis(
                "doi:10.1016/j.commatsci.2016.12.196",
                "https://doi.org/10.1016/j.commatsci.2016.12.196",
                "Extrapolation diagnostics are evidence for escalation, not a property value.",
            ),
        ],
        runtime_escalations=[
            "Raw absolute energies from different model reference conventions are audit-only.",
            "Missing unit, composition alignment, checkpoint attestation, or geometry match blocks disagreement scoring.",
        ],
    ),
    "relaxation_validation": StageResearchPolicy(
        policy_id="material-relaxation-evidence-v2",
        stage="relaxation_validation",
        query_intents=[
            _intent(
                "optimizer_convergence",
                "Retrieve optimizer, force, stress, cell, and restart convergence practice.",
                [
                    "periodic structure optimization BFGS force convergence stress cell",
                    "ASE optimizer trajectory restart convergence criterion",
                ],
                ["optimization_reference"],
            ),
            _intent(
                "phase_transformation",
                "Retrieve reported transformations and competing relaxed phases.",
                [
                    "phase transformation reconstruction relaxation polymorph",
                    "pressure temperature structural transition",
                ],
                ["phase_transformation"],
            ),
            _intent(
                "geometry_failure",
                "Retrieve collapse, overlap, volume, dimensionality, and cell-shape failures.",
                [
                    "structure collapse atom overlap volume collapse cell shear",
                    "relaxation failure reconstruction decomposition",
                ],
                ["geometry_failure"],
                negative=True,
            ),
            _intent(
                "phonon_instability",
                "Retrieve converged phonon and soft-mode evidence.",
                [
                    "phonon dispersion imaginary mode soft mode supercell convergence",
                    "finite displacement DFPT q mesh acoustic sum rule NAC",
                ],
                ["phonon_instability", "phonon_method_limit"],
            ),
            _intent(
                "finite_temperature_phase",
                "Retrieve anharmonic and finite-temperature stability context.",
                [
                    "finite temperature anharmonic free energy phase stability",
                    "molecular dynamics pressure temperature kinetic trap",
                ],
                ["finite_temperature_phase"],
            ),
        ],
        mcp=StageMcpPolicy(
            scope_argument="relaxation_scope",
            allowed_record_types=[
                "optimization_reference",
                "phase_transformation",
                "geometry_failure",
                "phonon_instability",
                "phonon_method_limit",
                "finite_temperature_phase",
            ],
            required_stage_metadata_fields=[
                "method",
                "convergence_criterion",
                "pressure",
                "temperature",
                "instability_kind",
            ],
        ),
        research_bases=[
            _basis(
                "doi:10.1088/1361-648X/aa680e",
                "https://doi.org/10.1088/1361-648X/aa680e",
                "ASE execution provenance includes optimizer and convergence settings.",
            ),
            _basis(
                "doi:10.1080/27660400.2024.2384822",
                "https://doi.org/10.1080/27660400.2024.2384822",
                "Symmetry changes are recorded with a versioned spglib tolerance policy.",
            ),
            _basis(
                "doi:10.1016/j.scriptamat.2015.07.021",
                "https://doi.org/10.1016/j.scriptamat.2015.07.021",
                "Phonon claims require method and supercell or q-grid provenance.",
            ),
            _basis(
                "doi:10.1039/D2DD00050D",
                "https://doi.org/10.1039/D2DD00050D",
                "Harmonic imaginary modes do not automatically decide finite-temperature stability.",
            ),
        ],
        runtime_escalations=[
            "Execution, optimizer convergence, geometry validity, and dynamical stability are separate states.",
            "Large model divergence, missing stress, collapse, or unconverged phonons escalates to DFT.",
        ],
    ),
    "dft_handoff": StageResearchPolicy(
        policy_id="material-dft-evidence-v2",
        stage="dft_handoff",
        query_intents=[
            _intent(
                "reference_phase_policy",
                "Retrieve compatible reference phases and correction/mixing policies.",
                [
                    "reference phase convex hull chemical potential correction scheme",
                    "same functional compatible energy reference set",
                ],
                ["reference_phase"],
            ),
            _intent(
                "electronic_method_policy",
                "Retrieve XC, U, spin, SOC, charge, dispersion, and occupation choices.",
                [
                    "exchange correlation Hubbard U spin orbit noncollinear magnetic order",
                    "charge smearing occupation dispersion method convergence",
                ],
                ["method_policy"],
            ),
            _intent(
                "pseudopotential_verification",
                "Retrieve verified pseudopotential identity and cutoff recommendations.",
                [
                    "SSSP PseudoDojo pseudopotential verification valence relativity",
                    "wavefunction density cutoff convergence checksum",
                ],
                ["pseudopotential_verification"],
            ),
            _intent(
                "numerical_convergence",
                "Retrieve k/q mesh, cutoff, cell, supercell, and observable convergence studies.",
                [
                    "k point q point cutoff supercell convergence sweep",
                    "energy force stress phonon band observable tolerance",
                ],
                ["convergence_study"],
            ),
            _intent(
                "specialized_workflow",
                "Retrieve the field-specific high-fidelity workflow and output contract.",
                [
                    "DFPT NEB EPW GW defect transport CALPHAD GCMC workflow",
                    "AiiDA NOMAD provenance parser immutable output",
                ],
                ["specialized_workflow", "provenance_reference"],
            ),
        ],
        mcp=StageMcpPolicy(
            scope_argument="dft_scope",
            allowed_record_types=[
                "reference_phase",
                "method_policy",
                "pseudopotential_verification",
                "convergence_study",
                "specialized_workflow",
                "provenance_reference",
            ],
            required_stage_metadata_fields=[
                "workflow_type",
                "code",
                "code_version",
                "method",
                "convergence_scope",
            ],
        ),
        research_bases=[
            _basis(
                "doi:10.1088/1361-648X/aa8f79",
                "https://doi.org/10.1088/1361-648X/aa8f79",
                "Periodic DFT handoffs preserve the exact code and method family.",
            ),
            _basis(
                "doi:10.1038/s41524-018-0127-2",
                "https://doi.org/10.1038/s41524-018-0127-2",
                "Pseudopotential and cutoff choices require verification and convergence evidence.",
            ),
            _basis(
                "doi:10.1038/s41597-020-00638-4",
                "https://doi.org/10.1038/s41597-020-00638-4",
                "Completed workflow results retain a traversable provenance graph.",
            ),
            _basis(
                "doi:10.21105/joss.05388",
                "https://doi.org/10.21105/joss.05388",
                "NOMAD identifiers are provenance locators, not numerical validation by themselves.",
            ),
        ],
        runtime_escalations=[
            "Prepared input is never a completed DFT result.",
            "Missing binary/container, pseudopotential, input/output, parser, or convergence hashes blocks every property claim.",
        ],
    ),
}


def stage_research_policy(stage: MaterialEvidenceStage | str) -> StageResearchPolicy:
    key = str(stage)
    if key not in STAGE_RESEARCH_POLICIES:
        raise ValueError(f"unknown material evidence stage: {stage}")
    policy = STAGE_RESEARCH_POLICIES[key]  # type: ignore[index]
    return StageResearchPolicy.model_validate_json(
        policy.model_dump_json(), strict=True
    )


def build_stage_query_blueprints(
    *,
    stage: MaterialEvidenceStage | str,
    chemical_system: str,
    material_field: MaterialField | str | None = None,
    application_subtype: str | None = None,
    problem_context: Mapping[str, JsonValue] | None = None,
    composition_keys: Sequence[str] = (),
    candidate_refs: Sequence[str] = (),
    focus_terms: Sequence[str] = (),
) -> list[LiteratureQueryBlueprint]:
    """Create deterministic, research-policy queries for exactly one stage."""

    policy = stage_research_policy(str(stage))
    clean_system = _bounded_text(chemical_system, "chemical system", 512)
    field_value = str(material_field) if material_field is not None else None
    subtype = (
        _bounded_text(application_subtype, "application subtype", 256)
        if application_subtype
        else None
    )
    compositions = _bounded_unique(composition_keys, "composition", 128, 256)
    candidates = _bounded_unique(candidate_refs, "candidate reference", 128, 256)
    focus = _bounded_unique(focus_terms, "focus term", 12, 1_000)
    context = dict(problem_context or {})
    _validate_context(context)
    base_parts = [
        clean_system,
        field_value or "",
        subtype or "",
        " ".join(compositions),
        " ".join(focus),
    ]
    base = " ".join(item for item in base_parts if item).strip()
    stage_scope: dict[str, JsonValue] = {
        "declared_context": context,
        "focus_terms": focus,
        "evidence_only": True,
        "property_score_authority": False,
    }
    rows: list[LiteratureQueryBlueprint] = []
    for intent in policy.query_intents:
        query = " ".join([base, *intent.query_terms]).strip()
        if len(query) > 4_000:
            query = query[:4_000].rsplit(" ", 1)[0]
        rows.append(
            LiteratureQueryBlueprint(
                intent_id=intent.intent_id,
                query=query,
                rationale=(
                    f"{intent.objective} Evidence must match the typed "
                    f"{policy.stage} record contract and cannot replace its runtime validator."
                ),
                expected_record_types=list(intent.expected_record_types),
                mcp_arguments={
                    "stage": policy.stage,
                    "intent_id": intent.intent_id,
                    "chemical_system": clean_system,
                    "material_field": field_value,
                    "application_subtype": subtype,
                    "composition_keys": compositions,
                    "candidate_refs": candidates,
                    "record_types": list(intent.expected_record_types),
                    policy.mcp.scope_argument: stage_scope,
                },
            )
        )
    return rows


def _bounded_text(value: object, label: str, max_length: int) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    if not text or len(text) > max_length:
        raise ValueError(f"{label} is missing or exceeds {max_length} characters")
    return text


def _bounded_unique(
    values: Sequence[object],
    label: str,
    max_items: int,
    max_length: int,
) -> list[str]:
    if len(values) > max_items:
        raise ValueError(f"{label} list exceeds {max_items} entries")
    rows = [_bounded_text(item, label, max_length) for item in values]
    if len(rows) != len(set(rows)):
        raise ValueError(f"{label} entries must be unique")
    return rows


def _validate_context(context: Mapping[str, JsonValue]) -> None:
    serialized = json.dumps(
        dict(context),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    if len(serialized) > 12_000:
        raise ValueError("stage research context exceeds 12000 characters")
    for key in context:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(key).casefold()).strip("_")
        if any(
            marker in normalized
            for marker in (
                "api_key",
                "access_key",
                "private_key",
                "client_secret",
                "token",
                "secret",
                "password",
                "credential",
                "authorization",
                "bearer",
            )
        ):
            raise ValueError("stage research context cannot contain secrets")


__all__ = [
    "STAGE_RESEARCH_POLICIES",
    "StageMcpPolicy",
    "StageQueryIntent",
    "StageResearchBasis",
    "StageResearchPolicy",
    "build_stage_query_blueprints",
    "stage_research_policy",
]
