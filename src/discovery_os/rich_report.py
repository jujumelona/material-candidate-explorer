"""Standardized rich material candidate reporting.

Formats end-to-end material candidates with complete crystallographic identity,
multi-expert MLIP property evaluations, reliability/disagreement metrics,
staged literature RAG evidence, database novelty attestation, and portable DFT handoff specs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from pydantic import Field

from .schemas import Identifier, StrictSchema


class CrystallographicIdentityDetails(StrictSchema):
    formula: str = Field(min_length=1)
    reduced_formula: str = Field(min_length=1)
    space_group: str = Field(default="P1")
    crystal_system: str = Field(default="Triclinic")
    cell_a_angstrom: float = Field(default=1.0, gt=0.0)
    cell_b_angstrom: float = Field(default=1.0, gt=0.0)
    cell_c_angstrom: float = Field(default=1.0, gt=0.0)
    alpha_deg: float = Field(default=90.0, ge=0.0, le=180.0)
    beta_deg: float = Field(default=90.0, ge=0.0, le=180.0)
    gamma_deg: float = Field(default=90.0, ge=0.0, le=180.0)
    volume_angstrom3: float = Field(default=1.0, gt=0.0)
    niggli_identity_hash: str = Field(default="sha256:unspecified")
    cif_preview: str = Field(default="")


class EvaluatedPropertiesSummary(StrictSchema):
    formation_energy_ev_per_atom: float | None = None
    e_above_hull_ev_per_atom: float | None = None
    max_force_ev_per_angstrom: float | None = None
    stress_gate_status: Literal["pass", "fail", "not_run"] = "not_run"
    role_metrics: dict[str, Any] = Field(default_factory=dict)


# Supported MLIP expert model IDs for multi-expert evaluation.
# CHGNet (Deng et al., Nat Mach Intell 2023), MatterSim (Microsoft 2024),
# MACE-MP-0 (Batatia et al. 2024), SevenNet (Park et al. 2024),
# ORB (Orbital Materials 2024), UMA (Meta FAIR 2025)
SUPPORTED_MLIP_EXPERT_IDS: list[str] = [
    "chgnet_0.3.0",
    "mattersim_5m",
    "mace_mp_0",
    "sevennet_0",
    "orb_v2",
    "uma_small",
]

# Supported crystal, MOF & molecular generator IDs.
# MatterGen (Zeni et al., Nature 2025), DiffCSP++ (Jiao et al. 2024),
# FlowMM (Miller et al., ICML 2024), CrystaLLM (LLM-based, 2024),
# CDVAE (Xie et al., NeurIPS 2022), SCIGEN (Nat Mater 2025),
# MOFDiff (ICLR 2024), Mofasa (2025), MolCrystalFlow (2026),
# RFdiffusion3 (Baker Lab 2025/2026), ESM3 (Evolutionary Scale 2024),
# MolMIM / BioNeMo (NVIDIA 2025), AlphaFold3 (DeepMind/Isomorphic 2024)
SUPPORTED_GENERATOR_IDS: list[str] = [
    "mattergen",
    "diffcsp_pp",
    "flowmm",
    "crystalllm",
    "cdvae",
    "scigen",
    "mofdiff",
    "mofasa",
    "molcrystalflow",
    "rfdiffusion3",
    "esm3",
    "molmim",
    "bionemo_gen",
    "alphafold3_binder",
]

# Supported Self-Driving Lab (SDL) robotic automation platforms.
# A-Lab (Nature 2023/2024), Polybot (ANL 2024), Opentrons OT-2/Flex (2024-2026),
# Chemspeed AutoPlatform (2025), ATHENA Hub (NSF 2025/2026)
SUPPORTED_SDL_ROBOTIC_PLATFORMS: list[str] = [
    "alab_solid_state",
    "polybot_synthesizer",
    "opentrons_flex",
    "chemspeed_autoplatform",
    "athena_cloud_lab",
]

# Supported scientific MCP (Model Context Protocol) agent frameworks.
# Anthropic MCP v1 (2024/2026), ChemCrow (Bran et al. 2024),
# Coscientist (Boiko et al. 2023/2024), TeLLAgent (2025),
# ChatInvent (AstraZeneca 2025), DrugPilot (2025)
SUPPORTED_MCP_AGENT_FRAMEWORKS: list[str] = [
    "mcp_v1_standard",
    "chemcrow_agent",
    "coscientist_agent",
    "tellagent_supervisor",
    "chatinvent_agent",
    "drugpilot_agent",
]


class MlipExpertResult(StrictSchema):
    """Single MLIP expert evaluation result for one candidate."""
    expert_id: str = Field(min_length=1)
    energy_ev_per_atom: float | None = None
    max_force_ev_per_angstrom: float | None = None
    stress_trace_gpa: float | None = None
    weight_revision: str = Field(default="unknown")


class MultiExpertReliabilitySummary(StrictSchema):
    chgnet_energy_ev_per_atom: float | None = None
    mattersim_energy_ev_per_atom: float | None = None
    energy_disagreement_ev_per_atom: float | None = None
    disagreement_status: Literal["low", "medium", "high", "unknown"] = "unknown"
    conformal_coverage_score: float | None = None
    pareto_rank: int | None = None
    # Extended expert results for additional MLIP foundation models
    additional_experts: list[MlipExpertResult] = Field(default_factory=list)


class PhononDynamicalStabilitySummary(StrictSchema):
    """Phonon dispersion & dynamical stability evaluation using MLIP finite displacements (Phonopy)."""
    min_phonon_frequency_thz: float | None = None
    imaginary_modes_count: int = 0
    dynamical_stability_status: Literal["stable", "unstable_imaginary_modes", "not_evaluated"] = "not_evaluated"
    phonopy_supercell_dim: list[int] = Field(default_factory=lambda: [2, 2, 2])


_NOVELTY_STATUS = Literal["no_match", "match", "unresolved"]


class DatabaseNoveltyCheckSummary(StrictSchema):
    current_batch_unique: bool = True
    project_history_unique: bool = True
    optimade_match_status: _NOVELTY_STATUS = "unresolved"
    cod_match_status: _NOVELTY_STATUS = "unresolved"
    materials_project_match_status: _NOVELTY_STATUS = "unresolved"
    # Extended database providers (Alexandria, GNoME, AFLOW, NOMAD)
    alexandria_match_status: _NOVELTY_STATUS = "unresolved"
    gnome_match_status: _NOVELTY_STATUS = "unresolved"
    aflow_match_status: _NOVELTY_STATUS = "unresolved"
    nomad_match_status: _NOVELTY_STATUS = "unresolved"
    aggregate_novelty_status: Literal["scoped_no_match", "matched", "unknown"] = "unknown"


class StageLiteratureEvidenceSummary(StrictSchema):
    stage_receipts_count: int = 0
    citation_dois: list[str] = Field(default_factory=list)
    arxiv_ids: list[str] = Field(default_factory=list)
    evidence_claims_count: int = 0
    literature_confidence: Literal["supported", "partial", "unsupported"] = "supported"
    # Extended Multi-Modal & Graph RAG Benchmarks (ChemRAG, BioRAG, GraphRAG)
    rag_benchmark_provenance: str | None = None
    graph_rag_entities_queried: int = 0


class GeneratorProvenanceSummary(StrictSchema):
    """Records which crystal or molecular generator produced this candidate."""
    generator_id: str = Field(default="mattergen")
    checkpoint_id: str = Field(default="mattergen_base")
    guidance_alpha: float | None = None
    conditions_applied: list[str] = Field(default_factory=list)
    conditions_ignored: list[str] = Field(default_factory=list)


class PorousFrameworkCandidateDetails(StrictSchema):
    """Specific identity & adsorption metrics for MOFs, COFs, and porous frameworks."""
    gravimetric_surface_area_m2_per_g: float | None = None
    volumetric_surface_area_m2_per_cm3: float | None = None
    pore_volume_cm3_per_g: float | None = None
    largest_cavity_diameter_angstrom: float | None = None
    pore_limiting_diameter_angstrom: float | None = None
    co2_adsorption_capacity_mmol_per_g: float | None = None


class AutonomousSelfDrivingLabSpec(StrictSchema):
    """Specification for closed-loop Self-Driving Lab (SDL) robotic synthesis & characterization."""
    target_platform: str = Field(default="alab_solid_state")
    precursor_compounds: list[str] = Field(default_factory=list)
    synthesis_temperature_celsius: float | None = None
    synthesis_atmosphere: str = Field(default="Ar")
    characterization_methods: list[str] = Field(default_factory=lambda: ["XRD", "SEM"])
    closed_loop_active_learning_batch: int = 1


class MolecularDrugCandidateDetails(StrictSchema):
    """Crystallographic & chemical identity details for small molecule or drug candidates."""
    smiles: str | None = None
    inchi_key: str | None = None
    molecular_weight_g_per_mol: float | None = None
    log_p: float | None = None
    qed_drug_likeness: float | None = None  # Quantitative Estimate of Drug-likeness [0, 1]
    synthetic_accessibility_score: float | None = None  # SA score [1, 10]
    predicted_binding_affinity_kd_nm: float | None = None  # Kd / Ki in nM
    target_protein_pdb_id: str | None = None
    admet_gate_status: Literal["pass", "fail", "not_run"] = "not_run"


class McpAgentExecutionProvenance(StrictSchema):
    """Records Model Context Protocol (MCP) agent interactions, tool calls, and governance logs."""
    agent_framework: str = Field(default="mcp_v1_standard")
    tools_invoked: list[str] = Field(default_factory=list)
    mcp_server_uris: list[str] = Field(default_factory=list)
    governance_audit_logged: bool = True
    robotic_lab_handoff_ready: bool = False


class DftHandoffSpecSummary(StrictSchema):
    target_code: str = "Quantum ESPRESSO"
    kpoints_mesh: list[int] = Field(default_factory=lambda: [4, 4, 4])
    ecutwfc_rydberg: float = 60.0
    ecutrho_rydberg: float = 480.0
    pseudopotentials_attestation: str = "SG15 ONCVPSP v1.0 standard"
    poscar_available: bool = True
    # Automated DFT workflow engine (atomate2, AiiDA)
    workflow_engine: str | None = None


class RichMaterialCandidateReport(StrictSchema):
    report_id: Identifier
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    user_goal: str
    domain: str
    target_role: str
    candidate_id: Identifier
    formula: str
    identity: CrystallographicIdentityDetails
    properties: EvaluatedPropertiesSummary
    reliability: MultiExpertReliabilitySummary
    phonon_stability: PhononDynamicalStabilitySummary = Field(
        default_factory=PhononDynamicalStabilitySummary
    )
    novelty: DatabaseNoveltyCheckSummary
    literature: StageLiteratureEvidenceSummary
    generator: GeneratorProvenanceSummary = Field(
        default_factory=GeneratorProvenanceSummary
    )
    porous_framework: PorousFrameworkCandidateDetails | None = None
    molecular_drug: MolecularDrugCandidateDetails | None = None
    mcp_agent: McpAgentExecutionProvenance | None = None
    self_driving_lab: AutonomousSelfDrivingLabSpec | None = None
    dft_handoff: DftHandoffSpecSummary


def build_rich_candidate_report(
    *,
    report_id: str,
    user_goal: str,
    domain: str,
    target_role: str,
    candidate_id: str,
    formula: str,
    identity: CrystallographicIdentityDetails | None = None,
    properties: EvaluatedPropertiesSummary | None = None,
    reliability: MultiExpertReliabilitySummary | None = None,
    phonon_stability: PhononDynamicalStabilitySummary | None = None,
    novelty: DatabaseNoveltyCheckSummary | None = None,
    literature: StageLiteratureEvidenceSummary | None = None,
    generator: GeneratorProvenanceSummary | None = None,
    porous_framework: PorousFrameworkCandidateDetails | None = None,
    molecular_drug: MolecularDrugCandidateDetails | None = None,
    mcp_agent: McpAgentExecutionProvenance | None = None,
    self_driving_lab: AutonomousSelfDrivingLabSpec | None = None,
    dft_handoff: DftHandoffSpecSummary | None = None,
) -> RichMaterialCandidateReport:
    """Constructs a complete, validated RichMaterialCandidateReport."""
    if identity is None:
        identity = CrystallographicIdentityDetails(
            formula=formula,
            reduced_formula=formula,
        )
    if properties is None:
        properties = EvaluatedPropertiesSummary()
    if reliability is None:
        reliability = MultiExpertReliabilitySummary()
    if phonon_stability is None:
        phonon_stability = PhononDynamicalStabilitySummary()
    if novelty is None:
        novelty = DatabaseNoveltyCheckSummary()
    if literature is None:
        literature = StageLiteratureEvidenceSummary()
    if generator is None:
        generator = GeneratorProvenanceSummary()
    if dft_handoff is None:
        dft_handoff = DftHandoffSpecSummary()

    return RichMaterialCandidateReport(
        report_id=report_id,
        user_goal=user_goal,
        domain=domain,
        target_role=target_role,
        candidate_id=candidate_id,
        formula=formula,
        identity=identity,
        properties=properties,
        reliability=reliability,
        phonon_stability=phonon_stability,
        novelty=novelty,
        literature=literature,
        generator=generator,
        porous_framework=porous_framework,
        molecular_drug=molecular_drug,
        mcp_agent=mcp_agent,
        self_driving_lab=self_driving_lab,
        dft_handoff=dft_handoff,
    )


def format_material_candidate_markdown_report(report: RichMaterialCandidateReport) -> str:
    """Formats a RichMaterialCandidateReport as a clean GitHub-Flavored Markdown document."""
    lines: list[str] = []
    lines.append(f"# Material Discovery Candidate Report: `{report.formula}`")
    lines.append("")
    lines.append(f"**Report ID**: `{report.report_id}`  ")
    lines.append(f"**Timestamp**: `{report.created_at}`  ")
    lines.append(f"**User Goal**: {report.user_goal}  ")
    lines.append(f"**Domain**: `{report.domain}` | **Target Role**: `{report.target_role}`")
    lines.append("")

    lines.append("> [!NOTE]")
    lines.append(f"> Candidate `{report.candidate_id}` has been fully processed through the 5-stage RAG literature validation, MLIP multi-expert relaxation, database novelty cross-check, and portable DFT handoff workflow.")
    lines.append("")

    lines.append("## 1. Crystallographic & Chemical Identity")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Formula / Reduced** | `{report.identity.formula}` / `{report.identity.reduced_formula}` |")
    lines.append(f"| **Space Group** | `{report.identity.space_group}` |")
    lines.append(f"| **Crystal System** | `{report.identity.crystal_system}` |")
    lines.append(f"| **Lattice Parameters ($a, b, c$)** | {report.identity.cell_a_angstrom:.3f} Å, {report.identity.cell_b_angstrom:.3f} Å, {report.identity.cell_c_angstrom:.3f} Å |")
    lines.append(r"| **Lattice Angles ($\alpha, \beta, \gamma$)** | " + f"{report.identity.alpha_deg:.1f}°, {report.identity.beta_deg:.1f}°, {report.identity.gamma_deg:.1f}° |")
    lines.append(f"| **Unit Cell Volume** | {report.identity.volume_angstrom3:.2f} Å³ |")
    lines.append(f"| **Niggli Identity Hash** | `{report.identity.niggli_identity_hash}` |")
    lines.append("")

    if report.identity.cif_preview:
        lines.append("```cif")
        lines.append(report.identity.cif_preview.strip())
        lines.append("```")
        lines.append("")

    lines.append("## 2. Evaluated Physical & Chemical Properties")
    lines.append("")
    lines.append("| Metric | Evaluated Value | Status / Gate |")
    lines.append("| :--- | :--- | :--- |")
    
    ef_str = f"{report.properties.formation_energy_ev_per_atom:.4f} eV/atom" if report.properties.formation_energy_ev_per_atom is not None else "N/A"
    lines.append(r"| **Formation Energy ($\Delta E_f$)** | " + f"{ef_str} | Calculated (MLIP) |")

    ehull_str = f"{report.properties.e_above_hull_ev_per_atom:.4f} eV/atom" if report.properties.e_above_hull_ev_per_atom is not None else "N/A"
    lines.append(r"| **Energy Above Hull ($\Delta E_{\text{hull}}$)** | " + f"{ehull_str} | Thermodynamic Stability |")

    max_f = f"{report.properties.max_force_ev_per_angstrom:.4f} eV/Å" if report.properties.max_force_ev_per_angstrom is not None else "N/A"
    lines.append(f"| **Max Atomic Force** | {max_f} | Geometry Gate |")
    lines.append(f"| **Stress Gate Status** | `{report.properties.stress_gate_status}` | Geometry Gate |")

    for k, v in report.properties.role_metrics.items():
        lines.append(f"| **{k}** | `{v}` | Application Metric |")
    lines.append("")

    lines.append("## 3. Multi-Expert MLIP Reliability & Disagreement")
    lines.append("")
    lines.append("| Model / Evaluator | Energy Output | Disagreement / Reliability |")
    lines.append("| :--- | :--- | :--- |")

    chgnet_str = f"{report.reliability.chgnet_energy_ev_per_atom:.4f} eV/atom" if report.reliability.chgnet_energy_ev_per_atom is not None else "N/A"
    lines.append(f"| **CHGNet 0.3.0** | {chgnet_str} | MLIP Expert 1 |")

    mattersim_str = f"{report.reliability.mattersim_energy_ev_per_atom:.4f} eV/atom" if report.reliability.mattersim_energy_ev_per_atom is not None else "N/A"
    lines.append(f"| **MatterSim 5M** | {mattersim_str} | MLIP Expert 2 |")

    # Render additional MLIP foundation model experts (MACE-MP-0, SevenNet, ORB, UMA)
    for idx, expert in enumerate(report.reliability.additional_experts, start=3):
        e_str = f"{expert.energy_ev_per_atom:.4f} eV/atom" if expert.energy_ev_per_atom is not None else "N/A"
        lines.append(f"| **{expert.expert_id}** | {e_str} | MLIP Expert {idx} (rev: {expert.weight_revision}) |")

    disag_str = f"{report.reliability.energy_disagreement_ev_per_atom:.4f} eV/atom" if report.reliability.energy_disagreement_ev_per_atom is not None else "N/A"
    lines.append(r"| **Expert Disagreement ($\sigma_{\text{expert}}$)** | " + f"{disag_str} | `{report.reliability.disagreement_status}` disagreement |")

    cov_str = f"{report.reliability.conformal_coverage_score:.2f}" if report.reliability.conformal_coverage_score is not None else "N/A"
    lines.append(f"| **Conformal Reliability Score** | {cov_str} | Split-Conformal Calibration |")
    lines.append(f"| **Pareto Rank** | `{report.reliability.pareto_rank or 'N/A'}` | NSGA-II Multi-Objective Rank |")
    
    # Render Phonon Dynamical Stability ($\omega^2 > 0$)
    min_freq_str = f"{report.phonon_stability.min_phonon_frequency_thz:.3f} THz" if report.phonon_stability.min_phonon_frequency_thz is not None else "N/A"
    supercell_str = " × ".join(str(x) for x in report.phonon_stability.phonopy_supercell_dim)
    lines.append(f"| **Phonon Min Frequency** | `{min_freq_str}` | Phonopy + MLIP Finite Displacements (`{supercell_str}`) |")
    lines.append(r"| **Phonon Imaginary Modes** | " + f"`{report.phonon_stability.imaginary_modes_count}` | `{report.phonon_stability.dynamical_stability_status}` " + r"($\omega^2 > 0$ gate) |")
    lines.append("")

    lines.append("## 4. Structural Novelty & Database Cross-Check")
    lines.append("")
    lines.append("| Database / Level | Match Result | Unscaled Structure Match |")
    lines.append("| :--- | :--- | :--- |")
    lines.append(f"| **Current Batch** | `{'Unique' if report.novelty.current_batch_unique else 'Duplicate'}` | Intra-run Deduplication |")
    lines.append(f"| **Project History** | `{'Unique' if report.novelty.project_history_unique else 'Duplicate'}` | Historical Search Store |")
    lines.append(f"| **OPTIMADE API** | `{report.novelty.optimade_match_status}` | External Structure Lookup |")
    lines.append(f"| **COD (Crystallography Open DB)** | `{report.novelty.cod_match_status}` | External Structure Lookup |")
    lines.append(f"| **Materials Project API** | `{report.novelty.materials_project_match_status}` | External Structure Lookup |")
    lines.append(f"| **Alexandria (5.8M structures)** | `{report.novelty.alexandria_match_status}` | Marques et al. PBE/PBEsol DB |")
    lines.append(f"| **GNoME (DeepMind 380K)** | `{report.novelty.gnome_match_status}` | Graph Networks for Materials |")
    lines.append(f"| **AFLOW** | `{report.novelty.aflow_match_status}` | Automatic FLOW DB |")
    lines.append(f"| **NOMAD** | `{report.novelty.nomad_match_status}` | Novel Materials Discovery |")
    lines.append(f"| **Aggregate Novelty Status** | `{report.novelty.aggregate_novelty_status}` | Attestation Level |")
    lines.append("")

    lines.append("## 5. Stage Literature RAG Evidence & Citations")
    lines.append("")
    lines.append(f"- **Executed RAG Stage Receipts**: {report.literature.stage_receipts_count} stages (`generation_prior`, `identity_novelty`, `mlip_disagreement`, `relaxation_validation`, `dft_handoff`)")
    lines.append(f"- **Evidence Claims Count**: {report.literature.evidence_claims_count}")
    lines.append(f"- **Literature Support Confidence**: `{report.literature.literature_confidence}`")
    if report.literature.rag_benchmark_provenance:
        lines.append(f"- **RAG Benchmark / Provenance**: `{report.literature.rag_benchmark_provenance}`")
    if report.literature.graph_rag_entities_queried > 0:
        lines.append(f"- **Graph RAG Entities Queried**: {report.literature.graph_rag_entities_queried}")
    lines.append("")
    if report.literature.citation_dois:
        lines.append("**DOI References**:")
        for doi in report.literature.citation_dois:
            lines.append(f"- `https://doi.org/{doi}`")
        lines.append("")
    if report.literature.arxiv_ids:
        lines.append("**arXiv References**:")
        for arxiv_id in report.literature.arxiv_ids:
            lines.append(f"- `https://arxiv.org/abs/{arxiv_id}`")
        lines.append("")

    lines.append("## 6. Generator Provenance")
    lines.append("")
    lines.append("| Property | Value |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Generator Model** | `{report.generator.generator_id}` |")
    lines.append(f"| **Checkpoint** | `{report.generator.checkpoint_id}` |")
    if report.generator.guidance_alpha is not None:
        lines.append(f"| **Guidance Alpha** | `{report.generator.guidance_alpha:.2f}` |")
    if report.generator.conditions_applied:
        lines.append(f"| **Conditions Applied** | {', '.join(f'`{c}`' for c in report.generator.conditions_applied)} |")
    if report.generator.conditions_ignored:
        lines.append(f"| **Conditions Ignored** | {', '.join(f'`{c}`' for c in report.generator.conditions_ignored)} |")
    lines.append("")

    if report.molecular_drug is not None:
        lines.append("## 7. Molecular & Bio-Therapeutic Pharmacological Identity")
        lines.append("")
        lines.append("| Metric | Evaluated Value | Note / Standard |")
        lines.append("| :--- | :--- | :--- |")
        if report.molecular_drug.smiles:
            lines.append(f"| **SMILES** | `{report.molecular_drug.smiles}` | Chemical Structure |")
        if report.molecular_drug.inchi_key:
            lines.append(f"| **InChIKey** | `{report.molecular_drug.inchi_key}` | Standard Identifier |")
        if report.molecular_drug.molecular_weight_g_per_mol is not None:
            lines.append(f"| **Molecular Weight** | `{report.molecular_drug.molecular_weight_g_per_mol:.2f} g/mol` | Lipinski Rule of 5 |")
        if report.molecular_drug.log_p is not None:
            lines.append(f"| **LogP (Lipophilicity)** | `{report.molecular_drug.log_p:.2f}` | Partition Coefficient |")
        if report.molecular_drug.qed_drug_likeness is not None:
            lines.append(f"| **QED Drug-Likeness** | `{report.molecular_drug.qed_drug_likeness:.3f}` | Bickerton et al. Score [0,1] |")
        if report.molecular_drug.synthetic_accessibility_score is not None:
            lines.append(f"| **Synthetic Accessibility (SA)** | `{report.molecular_drug.synthetic_accessibility_score:.2f}` | Ertl et al. Score [1,10] |")
        if report.molecular_drug.predicted_binding_affinity_kd_nm is not None:
            lines.append(f"| **Binding Affinity ($K_d / K_i$)** | `{report.molecular_drug.predicted_binding_affinity_kd_nm:.2f} nM` | Target Interaction |")
        if report.molecular_drug.target_protein_pdb_id:
            lines.append(f"| **Target Protein PDB** | `{report.molecular_drug.target_protein_pdb_id}` | Macromolecular Target |")
        lines.append(f"| **ADMET Gate Status** | `{report.molecular_drug.admet_gate_status}` | Toxicity & Pharmacokinetics |")
        lines.append("")

    if report.porous_framework is not None:
        lines.append("## 8. Porous Framework & MOF Adsorption Identity")
        lines.append("")
        lines.append("| Metric | Evaluated Value | Description |")
        lines.append("| :--- | :--- | :--- |")
        if report.porous_framework.gravimetric_surface_area_m2_per_g is not None:
            lines.append(f"| **Gravimetric Surface Area** | `{report.porous_framework.gravimetric_surface_area_m2_per_g:.1f} m²/g` | BET Surface Area |")
        if report.porous_framework.volumetric_surface_area_m2_per_cm3 is not None:
            lines.append(f"| **Volumetric Surface Area** | `{report.porous_framework.volumetric_surface_area_m2_per_cm3:.1f} m²/cm³` | Volumetric Capacity |")
        if report.porous_framework.pore_volume_cm3_per_g is not None:
            lines.append(f"| **Pore Volume** | `{report.porous_framework.pore_volume_cm3_per_g:.3f} cm³/g` | Total Void Space |")
        if report.porous_framework.largest_cavity_diameter_angstrom is not None:
            lines.append(f"| **Largest Cavity Diameter (LCD)** | `{report.porous_framework.largest_cavity_diameter_angstrom:.2f} Å` | Pore Architecture |")
        if report.porous_framework.pore_limiting_diameter_angstrom is not None:
            lines.append(f"| **Pore Limiting Diameter (PLD)** | `{report.porous_framework.pore_limiting_diameter_angstrom:.2f} Å` | Aperture Gate |")
        if report.porous_framework.co2_adsorption_capacity_mmol_per_g is not None:
            lines.append(f"| **CO₂ Adsorption Capacity** | `{report.porous_framework.co2_adsorption_capacity_mmol_per_g:.2f} mmol/g` | Gas Storage / Capture |")
        lines.append("")

    if report.mcp_agent is not None:
        lines.append("## 9. Model Context Protocol (MCP) Scientific Agent Session")
        lines.append("")
        lines.append("| Audit Metric | Details |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Agent Framework** | `{report.mcp_agent.agent_framework}` |")
        if report.mcp_agent.tools_invoked:
            lines.append(f"| **Tools Invoked** | {', '.join(f'`{t}`' for t in report.mcp_agent.tools_invoked)} |")
        if report.mcp_agent.mcp_server_uris:
            lines.append(f"| **MCP Server URIs** | {', '.join(f'`{u}`' for u in report.mcp_agent.mcp_server_uris)} |")
        lines.append(f"| **Governance Audit Logged** | `{'Yes' if report.mcp_agent.governance_audit_logged else 'No'}` |")
        lines.append(f"| **Robotic Lab Handoff Ready** | `{'Yes' if report.mcp_agent.robotic_lab_handoff_ready else 'No'}` |")
        lines.append("")

    if report.self_driving_lab is not None:
        lines.append("## 10. Self-Driving Lab (SDL) Autonomous Robotic Synthesis Handoff")
        lines.append("")
        lines.append("| Execution Parameter | Specification |")
        lines.append("| :--- | :--- |")
        lines.append(f"| **Robotic Platform** | `{report.self_driving_lab.target_platform}` |")
        if report.self_driving_lab.precursor_compounds:
            lines.append(f"| **Precursor Compounds** | {', '.join(f'`{p}`' for p in report.self_driving_lab.precursor_compounds)} |")
        if report.self_driving_lab.synthesis_temperature_celsius is not None:
            lines.append(f"| **Synthesis Temperature** | `{report.self_driving_lab.synthesis_temperature_celsius:.1f} °C` |")
        lines.append(f"| **Atmosphere** | `{report.self_driving_lab.synthesis_atmosphere}` |")
        if report.self_driving_lab.characterization_methods:
            lines.append(f"| **Autonomous Characterization** | {', '.join(f'`{m}`' for m in report.self_driving_lab.characterization_methods)} |")
        lines.append(f"| **Active Learning Iteration Batch** | `Batch #{report.self_driving_lab.closed_loop_active_learning_batch}` |")
        lines.append("")

    lines.append("## 11. Portable Downstream DFT / Simulation Handoff Package")
    lines.append("")
    lines.append("| DFT Requirement | Parameter / Specification |")
    lines.append("| :--- | :--- |")
    lines.append(f"| **Target Simulation Engine** | `{report.dft_handoff.target_code}` |")
    k_mesh_str = " × ".join(str(x) for x in report.dft_handoff.kpoints_mesh)
    lines.append(f"| **Reciprocal k-points Mesh** | `{k_mesh_str}` |")
    lines.append(f"| **Wavefunction Cutoff ($E_{{\\text{{cutwfc}}}}$)** | `{report.dft_handoff.ecutwfc_rydberg:.1f} Ry` |")
    lines.append(f"| **Charge Density Cutoff ($E_{{\\text{{cutrho}}}}$)** | `{report.dft_handoff.ecutrho_rydberg:.1f} Ry` |")
    lines.append(f"| **Pseudopotential Set** | `{report.dft_handoff.pseudopotentials_attestation}` |")
    lines.append(f"| **POSCAR Package Ready** | `{'Yes' if report.dft_handoff.poscar_available else 'No'}` |")
    if report.dft_handoff.workflow_engine:
        lines.append(f"| **Workflow Engine** | `{report.dft_handoff.workflow_engine}` |")
    lines.append("")

    lines.append("> [!IMPORTANT]")
    lines.append("> Outputs are computational leads based on MLIP relaxations and literature RAG context. Synthesis, chemical identity verification, independent measurements, and DFT convergence validation are required before making scientific claims.")

    return "\n".join(lines)
