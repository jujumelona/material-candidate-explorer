"""Built-in offline validators and explicit unavailable connector descriptors."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from .chemistry import FormulaError, molar_mass, parse_formula
from .hashing import candidate_content_hash, stable_hash
from .registry import ToolRegistry
from .schemas import (
    Candidate,
    CandidateType,
    ComputationalEvidenceDetails,
    DiscoveryDomain,
    EvidenceBatch,
    EvidenceKind,
    EvidenceRecord,
    EvidenceStatus,
    Fidelity,
    MethodClass,
    ParameterDescriptor,
    ParameterType,
    PropertyResult,
    RepresentationKind,
    ResourceBudget,
    ToolCall,
    ToolDescriptor,
    ToolOperationDescriptor,
    UncertaintyKind,
)


ALL_DOMAINS = list(DiscoveryDomain)
ALL_CANDIDATE_TYPES = list(CandidateType)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _representation(candidate: Candidate, kinds: set[str]) -> str | None:
    for representation in candidate.representations:
        if str(representation.kind) in kinds:
            return representation.value
    return None


def _normalize_rows(
    *,
    descriptor: ToolDescriptor,
    call: ToolCall,
    candidates: list[Candidate],
    rows: list[dict[str, Any]],
    runtime_seconds: float,
) -> EvidenceBatch:
    by_id = {row["candidate_id"]: row for row in rows}
    records: list[EvidenceRecord] = []
    for candidate in candidates:
        row = by_id.get(candidate.candidate_id)
        if row is None:
            row = {
                "candidate_id": candidate.candidate_id,
                "status": EvidenceStatus.FAILED,
                "properties": [],
                "failure_modes": ["adapter_returned_no_result"],
                "warnings": [],
            }
        output_hash = stable_hash(row)
        records.append(
            EvidenceRecord(
                evidence_id=f"EVD-{stable_hash([call.call_id, candidate.candidate_id, output_hash])[:20]}",
                call_id=call.call_id,
                candidate_id=candidate.candidate_id,
                candidate_ref=candidate.candidate_ref,
                tool_name=descriptor.tool_name,
                tool_version=descriptor.tool_version,
                operation=call.operation,
                method_class=call.method_class,
                status=row.get("status", EvidenceStatus.SUCCESS),
                evidence_kind=call.evidence_kind,
                fidelity=call.fidelity,
                properties=row.get("properties", []),
                failure_modes=row.get("failure_modes", []),
                warnings=row.get("warnings", []),
                runtime_seconds=runtime_seconds,
                input_hash=stable_hash(
                    {
                        "candidate_ref": candidate.candidate_ref,
                        "candidate": candidate if candidate.candidate_ref is None else None,
                        "operation": call.operation,
                        "conditions": call.conditions,
                        "tool_version": descriptor.tool_version,
                    }
                ),
                output_hash=output_hash,
                parameters_hash=stable_hash(call.conditions),
                convergence_checks={"operation_completed": True},
                computational_details=ComputationalEvidenceDetails(
                    method_name=descriptor.tool_name,
                    method_version=descriptor.tool_version,
                    parameters=call.conditions,
                    code_revision="discovery-os-0.1.0",
                ),
                observed_at=_now(),
            )
        )
    return EvidenceBatch(records=records)


class CommonRulesAdapter:
    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            tool_name="common_rules",
            tool_version="1.0",
            adapter_version="1.0",
            description="Dependency-free representation, content-hash, and lineage checks.",
            operations=[
                ToolOperationDescriptor(
                    operation="validate_candidate",
                    description="Validate candidate representation and immutable lineage metadata.",
                    supported_domains=ALL_DOMAINS,
                    supported_candidate_types=ALL_CANDIDATE_TYPES,
                    method_class=MethodClass.RULE_BASED,
                    produced_properties=[
                        "representation_valid",
                        "lineage_valid",
                        "content_hash_valid",
                    ],
                    evidence_kinds=[EvidenceKind.COMPUTATIONAL],
                    supported_fidelities=[Fidelity.CHEAP],
                    default_max_runtime_seconds=60,
                )
            ],
            deterministic=True,
            default_resource_budget=ResourceBudget(cpu_cores=0.1, memory_gb=0.1),
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def run(self, call: ToolCall, candidates: list[Candidate]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            representation_valid = bool(candidate.representations) and all(
                bool(item.value.strip()) for item in candidate.representations
            )
            lineage_valid = (
                candidate.candidate_id not in candidate.parent_candidate_ids
                and len(candidate.parent_candidate_ids) == len(set(candidate.parent_candidate_ids))
            )
            expected_hash = candidate_content_hash(candidate)
            content_hash_valid = (
                candidate.candidate_ref is not None
                and candidate.candidate_ref.content_hash == expected_hash
            )
            warnings: list[str] = []
            if candidate.candidate_ref is None:
                warnings.append("candidate_ref is missing; cross-version evidence reuse cannot be checked")
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": EvidenceStatus.SUCCESS,
                    "properties": [
                        PropertyResult(
                            property_name="representation_valid",
                            value=representation_valid,
                            meets_criterion=representation_valid,
                            criterion="at least one non-empty typed representation",
                        ),
                        PropertyResult(
                            property_name="lineage_valid",
                            value=lineage_valid,
                            meets_criterion=lineage_valid,
                            criterion="no self-parent and no duplicate parents",
                        ),
                        PropertyResult(
                            property_name="content_hash_valid",
                            value=content_hash_valid,
                            meets_criterion=content_hash_valid,
                            criterion="CandidateRef content hash matches immutable candidate content",
                        ),
                    ],
                    "warnings": warnings,
                    "failure_modes": [],
                }
            )
        return rows

    def normalize(
        self,
        call: ToolCall,
        candidates: list[Candidate],
        raw_result: Any,
        runtime_seconds: float,
    ) -> EvidenceBatch:
        return _normalize_rows(
            descriptor=self.descriptor,
            call=call,
            candidates=candidates,
            rows=raw_result,
            runtime_seconds=runtime_seconds,
        )


class CompositionRulesAdapter:
    def __init__(self) -> None:
        supported = [
            CandidateType.CRYSTAL,
            CandidateType.COMPOSITION,
            CandidateType.ALLOY,
            CandidateType.BATTERY_MATERIAL,
            CandidateType.CATALYST,
        ]
        self._descriptor = ToolDescriptor(
            tool_name="composition_rules",
            tool_version="1.0",
            adapter_version="1.0",
            description="Offline element-symbol, stoichiometry, and formula sanity checks.",
            operations=[
                ToolOperationDescriptor(
                    operation="validate_composition",
                    description="Parse a chemical formula without inferring phase stability or synthesis.",
                    supported_domains=ALL_DOMAINS,
                    supported_candidate_types=supported,
                    method_class=MethodClass.RULE_BASED,
                    produced_properties=[
                        "formula_validity",
                        "composition_validity",
                        "element_count",
                        "atom_count",
                        "molar_mass",
                        "elements",
                    ],
                    evidence_kinds=[EvidenceKind.COMPUTATIONAL],
                    supported_fidelities=[Fidelity.CHEAP],
                    default_max_runtime_seconds=60,
                )
            ],
            deterministic=True,
            default_resource_budget=ResourceBudget(cpu_cores=0.1, memory_gb=0.1),
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def run(self, call: ToolCall, candidates: list[Candidate]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            formula = _representation(candidate, {str(RepresentationKind.CHEMICAL_FORMULA)})
            if formula is None:
                formula_value = candidate.attributes.get("formula")
                formula = formula_value if isinstance(formula_value, str) else None
            if formula is None:
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": EvidenceStatus.SUCCESS,
                        "properties": [
                            PropertyResult(
                                property_name="formula_validity",
                                value=False,
                                meets_criterion=False,
                                criterion="a parseable chemical_formula representation is required",
                            ),
                            PropertyResult(
                                property_name="composition_validity",
                                value=False,
                                meets_criterion=False,
                                criterion="a parseable chemical formula is required",
                            ),
                        ],
                        "failure_modes": [],
                        "warnings": ["no chemical formula was supplied"],
                    }
                )
                continue
            try:
                composition = parse_formula(formula)
                mass = molar_mass(composition)
                properties = [
                    PropertyResult(
                        property_name="formula_validity",
                        value=True,
                        meets_criterion=True,
                        criterion="valid element symbols and positive finite stoichiometry",
                    ),
                    PropertyResult(
                        property_name="composition_validity",
                        value=True,
                        meets_criterion=True,
                        criterion="syntactic composition validity only; not stability or charge balance",
                    ),
                    PropertyResult(property_name="element_count", value=len(composition)),
                    PropertyResult(property_name="atom_count", value=sum(composition.values())),
                    PropertyResult(
                        property_name="elements",
                        value=",".join(composition),
                    ),
                ]
                if mass is not None and math.isfinite(mass):
                    properties.append(
                        PropertyResult(property_name="molar_mass", value=mass, unit="g/mol")
                    )
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": EvidenceStatus.SUCCESS,
                        "properties": properties,
                        "failure_modes": [],
                        "warnings": [
                            "formula parsing does not establish oxidation states, charge balance, crystal structure, phase stability, or synthesizability"
                        ],
                    }
                )
            except FormulaError as exc:
                rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "status": EvidenceStatus.SUCCESS,
                        "properties": [
                            PropertyResult(
                                property_name="formula_validity",
                                value=False,
                                meets_criterion=False,
                                criterion="valid element symbols and positive finite stoichiometry",
                            ),
                            PropertyResult(
                                property_name="composition_validity",
                                value=False,
                                meets_criterion=False,
                                criterion="parseable chemical formula",
                            ),
                        ],
                        "failure_modes": [],
                        "warnings": [str(exc)],
                    }
                )
        return rows

    def normalize(
        self,
        call: ToolCall,
        candidates: list[Candidate],
        raw_result: Any,
        runtime_seconds: float,
    ) -> EvidenceBatch:
        return _normalize_rows(
            descriptor=self.descriptor,
            call=call,
            candidates=candidates,
            rows=raw_result,
            runtime_seconds=runtime_seconds,
        )


class RDKitAdapter:
    def __init__(self) -> None:
        try:
            from rdkit import rdBase

            self._version = rdBase.rdkitVersion
            self._import_error: str | None = None
        except ImportError as exc:
            self._version = "unavailable"
            self._import_error = str(exc)

    @property
    def descriptor(self) -> ToolDescriptor:
        return ToolDescriptor(
            tool_name="rdkit",
            tool_version=self._version,
            adapter_version="1.0",
            description="RDKit sanitization, descriptors, structural alerts, and SA heuristic.",
            operations=[
                ToolOperationDescriptor(
                    operation="validate_molecule",
                    description="Sanitize a SMILES and calculate inexpensive molecular descriptors.",
                    supported_domains=ALL_DOMAINS,
                    supported_candidate_types=[CandidateType.SMALL_MOLECULE],
                    method_class=MethodClass.RULE_BASED,
                    produced_properties=[
                        "validity",
                        "chemical_validity",
                        "canonical_smiles",
                        "inchi_key",
                        "molecular_formula",
                        "molecular_weight",
                        "logp",
                        "tpsa",
                        "hbd",
                        "hba",
                        "rotatable_bonds",
                        "ring_count",
                        "fraction_csp3",
                        "qed",
                        "lipinski_violations",
                        "formal_charge",
                        "fragment_count",
                        "unassigned_stereocenters",
                        "radical_electrons",
                        "pains_alert_count",
                        "brenk_alert_count",
                        "synthetic_accessibility",
                    ],
                    evidence_kinds=[EvidenceKind.COMPUTATIONAL],
                    supported_fidelities=[Fidelity.CHEAP],
                    default_max_runtime_seconds=120,
                )
            ],
            available=self._import_error is None,
            deterministic=True,
            default_resource_budget=ResourceBudget(cpu_cores=1, memory_gb=1),
            metadata={"unavailable_reason": self._import_error or ""},
        )

    def run(self, call: ToolCall, candidates: list[Candidate]) -> list[dict[str, Any]]:
        if self._import_error is not None:
            raise RuntimeError(f"RDKit is unavailable: {self._import_error}")
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, FilterCatalog, Lipinski, QED, rdMolDescriptors

        pains_catalog = self._filter_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        brenk_catalog = self._filter_catalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        try:
            from rdkit.Contrib.SA_Score import sascorer
        except ImportError:
            sascorer = None

        rows: list[dict[str, Any]] = []
        for candidate in candidates:
            smiles = _representation(candidate, {str(RepresentationKind.SMILES)})
            if smiles is None:
                rows.append(self._invalid_row(candidate, "no SMILES representation was supplied"))
                continue
            try:
                molecule = Chem.MolFromSmiles(smiles, sanitize=True)
            except Exception as exc:  # RDKit exposes several C++ exception types.
                rows.append(self._invalid_row(candidate, f"RDKit sanitization error: {exc}"))
                continue
            if molecule is None:
                rows.append(self._invalid_row(candidate, "RDKit could not parse or sanitize the SMILES"))
                continue

            canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
            properties: list[PropertyResult] = [
                PropertyResult(property_name="validity", value=True, meets_criterion=True),
                PropertyResult(property_name="chemical_validity", value=True, meets_criterion=True),
                PropertyResult(property_name="canonical_smiles", value=canonical),
                PropertyResult(property_name="molecular_formula", value=rdMolDescriptors.CalcMolFormula(molecule)),
            ]
            try:
                properties.append(PropertyResult(property_name="inchi_key", value=Chem.MolToInchiKey(molecule)))
            except Exception:
                pass
            molecular_weight = Descriptors.MolWt(molecule)
            logp = Crippen.MolLogP(molecule)
            hbd = Lipinski.NumHDonors(molecule)
            hba = Lipinski.NumHAcceptors(molecule)
            rotors = Lipinski.NumRotatableBonds(molecule)
            tpsa = rdMolDescriptors.CalcTPSA(molecule)
            lipinski_violations = sum(
                [molecular_weight > 500, logp > 5, hbd > 5, hba > 10]
            )
            properties.extend(
                [
                    PropertyResult(
                        property_name="molecular_weight",
                        value=molecular_weight,
                        unit="g/mol",
                        meets_criterion=molecular_weight <= 500,
                        criterion="Lipinski heuristic MW <= 500; warning only",
                    ),
                    PropertyResult(
                        property_name="logp",
                        value=logp,
                        meets_criterion=logp <= 5,
                        criterion="Lipinski heuristic cLogP <= 5; warning only",
                    ),
                    PropertyResult(property_name="tpsa", value=tpsa, unit="angstrom^2"),
                    PropertyResult(property_name="hbd", value=hbd),
                    PropertyResult(property_name="hba", value=hba),
                    PropertyResult(property_name="rotatable_bonds", value=rotors),
                    PropertyResult(property_name="ring_count", value=Lipinski.RingCount(molecule)),
                    PropertyResult(property_name="fraction_csp3", value=rdMolDescriptors.CalcFractionCSP3(molecule)),
                    PropertyResult(property_name="qed", value=QED.qed(molecule)),
                    PropertyResult(property_name="lipinski_violations", value=lipinski_violations),
                    PropertyResult(property_name="formal_charge", value=Chem.GetFormalCharge(molecule)),
                    PropertyResult(property_name="fragment_count", value=len(Chem.GetMolFrags(molecule))),
                    PropertyResult(
                        property_name="unassigned_stereocenters",
                        value=sum(
                            1
                            for _, label in Chem.FindMolChiralCenters(
                                molecule, includeUnassigned=True, useLegacyImplementation=False
                            )
                            if label == "?"
                        ),
                    ),
                    PropertyResult(
                        property_name="radical_electrons",
                        value=sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()),
                    ),
                    PropertyResult(
                        property_name="pains_alert_count",
                        value=len(pains_catalog.GetMatches(molecule)),
                    ),
                    PropertyResult(
                        property_name="brenk_alert_count",
                        value=len(brenk_catalog.GetMatches(molecule)),
                    ),
                ]
            )
            warnings = [
                "RDKit validity and descriptors do not establish efficacy, safety, stability, or synthesizability"
            ]
            if sascorer is not None:
                sa_score = float(sascorer.calculateScore(molecule))
                properties.append(
                    PropertyResult(
                        property_name="synthetic_accessibility",
                        value=sa_score,
                        meets_criterion=sa_score <= 6.0,
                        criterion="RDKit SA heuristic <= 6; not a synthesis route or proof",
                    )
                )
            else:
                warnings.append("RDKit SA_Score contribution is unavailable")
            if len(Chem.GetMolFrags(molecule)) > 1:
                warnings.append("multiple disconnected fragments or salt components")
            if any(
                item.property_name in {"pains_alert_count", "brenk_alert_count"}
                and isinstance(item.value, int)
                and item.value > 0
                for item in properties
            ):
                warnings.append("structural alerts require expert review and are not automatic rejection rules")
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "status": EvidenceStatus.SUCCESS,
                    "properties": properties,
                    "failure_modes": [],
                    "warnings": warnings,
                }
            )
        return rows

    @staticmethod
    def _filter_catalog(catalog_kind: Any) -> Any:
        from rdkit.Chem import FilterCatalog

        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(catalog_kind)
        return FilterCatalog.FilterCatalog(params)

    @staticmethod
    def _invalid_row(candidate: Candidate, warning: str) -> dict[str, Any]:
        return {
            "candidate_id": candidate.candidate_id,
            "status": EvidenceStatus.SUCCESS,
            "properties": [
                PropertyResult(
                    property_name="validity",
                    value=False,
                    meets_criterion=False,
                    criterion="RDKit parsing and sanitization",
                ),
                PropertyResult(
                    property_name="chemical_validity",
                    value=False,
                    meets_criterion=False,
                    criterion="RDKit parsing and sanitization",
                ),
            ],
            "failure_modes": [],
            "warnings": [warning],
        }

    def normalize(
        self,
        call: ToolCall,
        candidates: list[Candidate],
        raw_result: Any,
        runtime_seconds: float,
    ) -> EvidenceBatch:
        return _normalize_rows(
            descriptor=self.descriptor,
            call=call,
            candidates=candidates,
            rows=raw_result,
            runtime_seconds=runtime_seconds,
        )


class DummySimulationTool:
    """Deterministic integration-test tool; policy disables it by default."""

    def __init__(self) -> None:
        self._descriptor = ToolDescriptor(
            tool_name="dummy_simulation",
            tool_version="1.0",
            adapter_version="1.0",
            description="Synthetic values for orchestration tests only; never scientific evidence.",
            operations=[
                ToolOperationDescriptor(
                    operation="simulate",
                    description="Return a deterministic synthetic target property.",
                    supported_domains=ALL_DOMAINS,
                    supported_candidate_types=ALL_CANDIDATE_TYPES,
                    method_class=MethodClass.PHYSICS_SIMULATION,
                    produced_properties=["target_property"],
                    evidence_kinds=[EvidenceKind.COMPUTATIONAL],
                    supported_fidelities=[Fidelity.CHEAP, Fidelity.MEDIUM, Fidelity.HIGH],
                    condition_parameters=[
                        ParameterDescriptor(
                            name="unit",
                            value_type=ParameterType.STRING,
                            required=False,
                        )
                    ],
                )
            ],
            deterministic=True,
            default_resource_budget=ResourceBudget(cpu_cores=0.1),
            metadata={"mock": True},
        )

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def run(self, call: ToolCall, candidates: list[Candidate]) -> list[dict[str, Any]]:
        return [
            {
                "candidate_id": candidate.candidate_id,
                "status": EvidenceStatus.SUCCESS,
                "properties": [
                    PropertyResult(
                        property_name="target_property",
                        value=int(stable_hash(candidate)[:8], 16) / 0xFFFFFFFF,
                        unit=str(call.conditions.get("unit", "arb")),
                        uncertainty=1.0,
                        uncertainty_kind=UncertaintyKind.UNKNOWN,
                        meets_criterion=False,
                        criterion="dummy output cannot satisfy a scientific criterion",
                    )
                ],
                "failure_modes": [],
                "warnings": ["synthetic integration-test output; not scientific evidence"],
            }
            for candidate in candidates
        ]

    def normalize(
        self,
        call: ToolCall,
        candidates: list[Candidate],
        raw_result: Any,
        runtime_seconds: float,
    ) -> EvidenceBatch:
        return _normalize_rows(
            descriptor=self.descriptor,
            call=call,
            candidates=candidates,
            rows=raw_result,
            runtime_seconds=runtime_seconds,
        )


class UnavailableToolAdapter:
    def __init__(self, descriptor: ToolDescriptor) -> None:
        self._descriptor = descriptor.model_copy(update={"available": False})

    @property
    def descriptor(self) -> ToolDescriptor:
        return self._descriptor

    def run(self, call: ToolCall, candidates: list[Candidate]) -> Any:
        raise RuntimeError(
            f"connector {self.descriptor.tool_name!r} requires an installed and configured backend"
        )

    def normalize(
        self,
        call: ToolCall,
        candidates: list[Candidate],
        raw_result: Any,
        runtime_seconds: float,
    ) -> EvidenceBatch:
        raise RuntimeError("unavailable connectors cannot normalize results")


def _placeholder(
    name: str,
    description: str,
    operations: list[tuple[str, list[str], MethodClass, list[CandidateType]]],
) -> UnavailableToolAdapter:
    return UnavailableToolAdapter(
        ToolDescriptor(
            tool_name=name,
            tool_version="not-configured",
            adapter_version="1.0-contract",
            description=description,
            operations=[
                ToolOperationDescriptor(
                    operation=operation,
                    description=description,
                    supported_domains=ALL_DOMAINS,
                    supported_candidate_types=candidate_types,
                    method_class=method_class,
                    produced_properties=properties,
                    evidence_kinds=[EvidenceKind.COMPUTATIONAL],
                    supported_fidelities=[Fidelity.MEDIUM, Fidelity.HIGH],
                    default_max_runtime_seconds=86_400,
                )
                for operation, properties, method_class, candidate_types in operations
            ],
            available=False,
            metadata={"connector_status": "backend_required"},
        )
    )


def build_default_tool_registry(*, include_placeholders: bool = True) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(CommonRulesAdapter())
    registry.register(CompositionRulesAdapter())
    registry.register(RDKitAdapter())
    registry.register(DummySimulationTool())
    if include_placeholders:
        small = [CandidateType.SMALL_MOLECULE]
        structures = [
            CandidateType.CRYSTAL,
            CandidateType.COMPOSITION,
            CandidateType.ALLOY,
            CandidateType.BATTERY_MATERIAL,
            CandidateType.CATALYST,
        ]
        registry.register(
            _placeholder(
                "boltz",
                "Boltz structure/affinity connector; model weights and runtime configuration required.",
                [("predict_affinity", ["binding_affinity", "target_activity"], MethodClass.MACHINE_LEARNING, small)],
            )
        )
        registry.register(
            _placeholder(
                "openmm",
                "OpenMM molecular simulation connector; force-field coverage and convergence checks required.",
                [("molecular_dynamics", ["structural_stability", "interaction_energy"], MethodClass.MOLECULAR_SIMULATION, small + [CandidateType.BIOLOGIC])],
            )
        )
        registry.register(
            _placeholder(
                "admet",
                "Calibrated ADMET/QSAR backend connector; endpoint and applicability-domain metadata required.",
                [("predict_admet", ["admet", "toxicity", "selectivity", "off_target_risk"], MethodClass.MACHINE_LEARNING, small)],
            )
        )
        registry.register(
            _placeholder(
                "retrosynthesis",
                "Retrosynthesis backend connector.",
                [("plan_synthesis", ["synthesis_feasibility", "synthetic_accessibility"], MethodClass.MACHINE_LEARNING, small)],
            )
        )
        registry.register(
            _placeholder(
                "pymatgen",
                "pymatgen structure, oxidation-state, and phase-analysis connector.",
                [("validate_structure", ["structure_validity", "composition_validity", "charge_balance"], MethodClass.RULE_BASED, structures)],
            )
        )
        registry.register(
            _placeholder(
                "spglib",
                "spglib symmetry and cell-standardization connector.",
                [("standardize_structure", ["structure_validity", "space_group", "standardized_structure"], MethodClass.RULE_BASED, structures)],
            )
        )
        registry.register(
            _placeholder(
                "materials_database",
                "Versioned reference-database connector for duplicate, novelty, and competing-phase searches.",
                [
                    ("novelty_search", ["database_novelty", "nearest_known_match"], MethodClass.RULE_BASED, structures),
                    ("competing_phases", ["competing_phases", "reference_snapshot"], MethodClass.RULE_BASED, structures),
                ],
            )
        )
        registry.register(
            _placeholder(
                "dft",
                "Configured DFT engine connector with fixed pseudopotential and convergence policies.",
                [
                    ("relax", ["formation_energy", "structure_validity"], MethodClass.QUANTUM_CHEMISTRY, structures),
                    ("electronic_structure", ["electronic_structure", "density_of_states", "target_property"], MethodClass.QUANTUM_CHEMISTRY, structures),
                ],
            )
        )
        registry.register(
            _placeholder(
                "phase_diagram",
                "Consistent competing-phase reference connector for convex-hull stability.",
                [("energy_above_hull", ["energy_above_hull", "thermodynamic_stability"], MethodClass.QUANTUM_CHEMISTRY, structures)],
            )
        )
        registry.register(
            _placeholder(
                "phonopy",
                "Phonon/dynamical-stability connector with convergence policy.",
                [("phonon_stability", ["phonon_stability", "dynamic_stability"], MethodClass.QUANTUM_CHEMISTRY, structures)],
            )
        )
        registry.register(
            _placeholder(
                "quantum_espresso",
                "Quantum ESPRESSO SCF, phonon, and electron-phonon connector with fixed convergence recipes.",
                [
                    ("scf", ["formation_energy", "electronic_structure", "density_of_states"], MethodClass.QUANTUM_CHEMISTRY, structures),
                    ("phonon", ["phonon_stability", "dynamic_stability"], MethodClass.QUANTUM_CHEMISTRY, structures),
                    ("electron_phonon", ["electron_phonon_coupling", "critical_temperature", "tc"], MethodClass.QUANTUM_CHEMISTRY, [CandidateType.CRYSTAL]),
                ],
            )
        )
        registry.register(
            _placeholder(
                "epw",
                "Electron-phonon coupling and Tc-range connector for applicable mechanisms.",
                [("electron_phonon", ["electron_phonon_coupling", "critical_temperature", "tc"], MethodClass.QUANTUM_CHEMISTRY, [CandidateType.CRYSTAL])],
            )
        )
    return registry


__all__ = [
    "CommonRulesAdapter",
    "CompositionRulesAdapter",
    "DummySimulationTool",
    "RDKitAdapter",
    "UnavailableToolAdapter",
    "build_default_tool_registry",
]
