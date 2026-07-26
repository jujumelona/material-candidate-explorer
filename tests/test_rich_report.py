"""Tests for standardized rich material candidate reporting and CLI material goal runner."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from discovery_os.rich_report import (
    CrystallographicIdentityDetails,
    DatabaseNoveltyCheckSummary,
    DftHandoffSpecSummary,
    EvaluatedPropertiesSummary,
    GeneratorProvenanceSummary,
    McpAgentExecutionProvenance,
    MlipExpertResult,
    MolecularDrugCandidateDetails,
    MultiExpertReliabilitySummary,
    RichMaterialCandidateReport,
    SUPPORTED_GENERATOR_IDS,
    SUPPORTED_MCP_AGENT_FRAMEWORKS,
    SUPPORTED_MLIP_EXPERT_IDS,
    StageLiteratureEvidenceSummary,
    build_rich_candidate_report,
    format_material_candidate_markdown_report,
)
from discovery_os.material_decision_runner import MaterialDecisionRunner
from discovery_os import cli


def test_rich_candidate_report_creation_and_markdown_formatting() -> None:
    report = build_rich_candidate_report(
        report_id="RICH-TEST-001",
        user_goal="High ionic conductivity solid electrolyte candidate for lithium batteries",
        domain="battery",
        target_role="solid_electrolyte",
        candidate_id="CAND-LI7LA3ZR2O12",
        formula="Li7La3Zr2O12",
        identity=CrystallographicIdentityDetails(
            formula="Li7La3Zr2O12",
            reduced_formula="Li7La3Zr2O12",
            space_group="Ia-3d",
            crystal_system="Cubic",
            cell_a_angstrom=12.96,
            cell_b_angstrom=12.96,
            cell_c_angstrom=12.96,
            alpha_deg=90.0,
            beta_deg=90.0,
            gamma_deg=90.0,
            volume_angstrom3=2176.6,
            niggli_identity_hash="sha256:11223344556677889900aabbccddeeff11223344556677889900aabbccddeeff",
            cif_preview="data_Li7La3Zr2O12\n_cell_length_a 12.96",
        ),
        properties=EvaluatedPropertiesSummary(
            formation_energy_ev_per_atom=-0.52,
            e_above_hull_ev_per_atom=0.005,
            max_force_ev_per_angstrom=0.004,
            stress_gate_status="pass",
            role_metrics={
                "estimated_ionic_conductivity_S_cm": 1.2e-3,
            },
        ),
        reliability=MultiExpertReliabilitySummary(
            chgnet_energy_ev_per_atom=-0.521,
            mattersim_energy_ev_per_atom=-0.519,
            energy_disagreement_ev_per_atom=0.002,
            disagreement_status="low",
            conformal_coverage_score=0.98,
            pareto_rank=1,
        ),
        novelty=DatabaseNoveltyCheckSummary(
            current_batch_unique=True,
            project_history_unique=True,
            optimade_match_status="no_match",
            cod_match_status="no_match",
            materials_project_match_status="no_match",
            aggregate_novelty_status="scoped_no_match",
        ),
        literature=StageLiteratureEvidenceSummary(
            stage_receipts_count=5,
            citation_dois=["10.1038/s41586-021-00000-0"],
            arxiv_ids=["2101.00000"],
            evidence_claims_count=3,
            literature_confidence="supported",
        ),
        dft_handoff=DftHandoffSpecSummary(
            target_code="Quantum ESPRESSO",
            kpoints_mesh=[4, 4, 4],
            ecutwfc_rydberg=60.0,
            ecutrho_rydberg=480.0,
            pseudopotentials_attestation="SG15 ONCVPSP v1.0 standard",
            poscar_available=True,
        ),
    )

    assert report.formula == "Li7La3Zr2O12"
    assert report.identity.space_group == "Ia-3d"
    assert report.reliability.disagreement_status == "low"

    md = format_material_candidate_markdown_report(report)
    assert "# Material Discovery Candidate Report: `Li7La3Zr2O12`" in md
    assert "Ia-3d" in md
    assert "CHGNet 0.3.0" in md
    assert "Quantum ESPRESSO" in md
    assert "https://doi.org/10.1038/s41586-021-00000-0" in md


def test_material_decision_runner_goal_discovery(tmp_path: Path) -> None:
    runner = MaterialDecisionRunner(artifact_root=tmp_path)
    run, reports = runner.run_material_goal_discovery(
        "Find high ionic conductivity solid electrolyte candidate for all-solid-state lithium batteries",
        run_rag=False,
    )

    assert run.run_id.startswith("MDRUN-")
    assert len(reports) > 0
    first_report = reports[0]
    assert isinstance(first_report, RichMaterialCandidateReport)
    assert first_report.user_goal == "Find high ionic conductivity solid electrolyte candidate for all-solid-state lithium batteries"

    # Check generated files in artifact root
    run_dir = tmp_path / run.run_id
    assert (run_dir / "material-decision-run.json").exists()
    assert (run_dir / "rich-candidate-report-1.json").exists()
    assert (run_dir / "rich-candidate-report-1.md").exists()


def test_cli_material_goal_run_command(tmp_path: Path) -> None:
    export_md = tmp_path / "output_report.md"
    export_json = tmp_path / "output_report.json"
    artifacts_dir = tmp_path / "artifacts"

    exit_code = cli.main(
        [
            "material-goal-run",
            "--goal",
            "Develop narrow bandgap perovskite candidate for solar cell",
            "--no-rag",
            "--artifacts",
            str(artifacts_dir),
            "--export-markdown",
            str(export_md),
            "--export-json",
            str(export_json),
        ]
    )

    assert exit_code == 0
    assert export_md.exists()
    assert export_json.exists()
    md_text = export_md.read_text(encoding="utf-8")
    assert "Material Discovery Candidate Report" in md_text


def test_supported_ids_lists() -> None:
    """Verify supported MLIP and generator ID lists contain expected entries."""
    assert "chgnet_0.3.0" in SUPPORTED_MLIP_EXPERT_IDS
    assert "mattersim_5m" in SUPPORTED_MLIP_EXPERT_IDS
    assert "mace_mp_0" in SUPPORTED_MLIP_EXPERT_IDS
    assert "sevennet_0" in SUPPORTED_MLIP_EXPERT_IDS
    assert "orb_v2" in SUPPORTED_MLIP_EXPERT_IDS
    assert "uma_small" in SUPPORTED_MLIP_EXPERT_IDS
    assert len(SUPPORTED_MLIP_EXPERT_IDS) == 6

    assert "mattergen" in SUPPORTED_GENERATOR_IDS
    assert "diffcsp_pp" in SUPPORTED_GENERATOR_IDS
    assert "flowmm" in SUPPORTED_GENERATOR_IDS
    assert "crystalllm" in SUPPORTED_GENERATOR_IDS
    assert "cdvae" in SUPPORTED_GENERATOR_IDS
    assert "scigen" in SUPPORTED_GENERATOR_IDS
    assert "rfdiffusion3" in SUPPORTED_GENERATOR_IDS
    assert "esm3" in SUPPORTED_GENERATOR_IDS
    assert "molmim" in SUPPORTED_GENERATOR_IDS
    assert "bionemo_gen" in SUPPORTED_GENERATOR_IDS
    assert "alphafold3_binder" in SUPPORTED_GENERATOR_IDS
    assert len(SUPPORTED_GENERATOR_IDS) == 11

    assert "mcp_v1_standard" in SUPPORTED_MCP_AGENT_FRAMEWORKS
    assert "chemcrow_agent" in SUPPORTED_MCP_AGENT_FRAMEWORKS
    assert "coscientist_agent" in SUPPORTED_MCP_AGENT_FRAMEWORKS
    assert "tellagent_supervisor" in SUPPORTED_MCP_AGENT_FRAMEWORKS
    assert "chatinvent_agent" in SUPPORTED_MCP_AGENT_FRAMEWORKS
    assert "drugpilot_agent" in SUPPORTED_MCP_AGENT_FRAMEWORKS
    assert len(SUPPORTED_MCP_AGENT_FRAMEWORKS) == 6


def test_mlip_expert_result_schema() -> None:
    """MlipExpertResult can be constructed and serialized."""
    expert = MlipExpertResult(
        expert_id="mace_mp_0",
        energy_ev_per_atom=-0.45,
        max_force_ev_per_angstrom=0.012,
        stress_trace_gpa=0.5,
        weight_revision="mace_mp_0_medium_2024",
    )
    assert expert.expert_id == "mace_mp_0"
    data = expert.model_dump()
    assert data["energy_ev_per_atom"] == pytest.approx(-0.45)


def test_generator_provenance_schema() -> None:
    """GeneratorProvenanceSummary records generator metadata."""
    gen = GeneratorProvenanceSummary(
        generator_id="diffcsp_pp",
        checkpoint_id="diffcsp_pp_mp20",
        guidance_alpha=0.7,
        conditions_applied=["space_group"],
        conditions_ignored=["dft_band_gap"],
    )
    assert gen.generator_id == "diffcsp_pp"
    assert gen.guidance_alpha == pytest.approx(0.7)
    assert "space_group" in gen.conditions_applied


def test_extended_novelty_database_fields() -> None:
    """DatabaseNoveltyCheckSummary includes Alexandria, GNoME, AFLOW, NOMAD."""
    novelty = DatabaseNoveltyCheckSummary(
        current_batch_unique=True,
        project_history_unique=True,
        optimade_match_status="no_match",
        cod_match_status="no_match",
        materials_project_match_status="no_match",
        alexandria_match_status="no_match",
        gnome_match_status="match",
        aflow_match_status="unresolved",
        nomad_match_status="no_match",
        aggregate_novelty_status="scoped_no_match",
    )
    assert novelty.gnome_match_status == "match"
    assert novelty.alexandria_match_status == "no_match"
    assert novelty.aflow_match_status == "unresolved"
    assert novelty.nomad_match_status == "no_match"


def test_rich_report_with_extended_fields_and_markdown() -> None:
    """Full report with additional experts, generator provenance, extended DBs renders correctly."""
    report = build_rich_candidate_report(
        report_id="RICH-EXT-001",
        user_goal="Find perovskite photovoltaic absorber",
        domain="semiconductor",
        target_role="photovoltaic_absorber",
        candidate_id="CAND-CSPBI3",
        formula="CsPbI3",
        reliability=MultiExpertReliabilitySummary(
            chgnet_energy_ev_per_atom=-0.30,
            mattersim_energy_ev_per_atom=-0.29,
            energy_disagreement_ev_per_atom=0.01,
            disagreement_status="low",
            additional_experts=[
                MlipExpertResult(
                    expert_id="mace_mp_0",
                    energy_ev_per_atom=-0.295,
                    weight_revision="mace_mp_0_medium",
                ),
                MlipExpertResult(
                    expert_id="sevennet_0",
                    energy_ev_per_atom=-0.305,
                    weight_revision="sevennet_0_11July2024",
                ),
            ],
        ),
        novelty=DatabaseNoveltyCheckSummary(
            alexandria_match_status="no_match",
            gnome_match_status="no_match",
            aflow_match_status="no_match",
            nomad_match_status="no_match",
            aggregate_novelty_status="scoped_no_match",
        ),
        generator=GeneratorProvenanceSummary(
            generator_id="diffcsp_pp",
            checkpoint_id="diffcsp_pp_mp20",
            guidance_alpha=0.5,
            conditions_applied=["space_group", "chemical_system"],
        ),
        dft_handoff=DftHandoffSpecSummary(
            workflow_engine="atomate2",
        ),
    )

    assert report.generator.generator_id == "diffcsp_pp"
    assert len(report.reliability.additional_experts) == 2
    assert report.dft_handoff.workflow_engine == "atomate2"

    md = format_material_candidate_markdown_report(report)
    # Additional experts rendered
    assert "mace_mp_0" in md
    assert "sevennet_0" in md
    # Extended DB novelty rendered
    assert "Alexandria" in md
    assert "GNoME" in md
    assert "AFLOW" in md
    assert "NOMAD" in md
    # Generator provenance section rendered
    assert "Generator Provenance" in md
    assert "diffcsp_pp" in md
    # DFT workflow engine rendered
    assert "atomate2" in md
    # Section numbering updated
    assert "## 6. Generator Provenance" in md
    assert "## 9. Portable Downstream DFT" in md


def test_molecular_drug_and_mcp_agent_report_rendering() -> None:
    """Verify drug candidate details and MCP agent sessions render cleanly in markdown."""
    report = build_rich_candidate_report(
        report_id="DRUG-MCP-001",
        user_goal="Discover novel small molecule inhibitor for SARS-CoV-2 main protease",
        domain="pharmaceutical",
        target_role="small_molecule_inhibitor",
        candidate_id="CAND-DRUG-99",
        formula="C21H23N5O4",
        generator=GeneratorProvenanceSummary(
            generator_id="rfdiffusion3",
            checkpoint_id="rfd3_all_atom_v1",
        ),
        molecular_drug=MolecularDrugCandidateDetails(
            smiles="CC(C)CC(NC(=O)C(F)(F)F)C(=O)NC(Cc1ccccc1)C=O",
            inchi_key="InChIKey=XYZ123456789",
            molecular_weight_g_per_mol=425.45,
            log_p=2.35,
            qed_drug_likeness=0.82,
            synthetic_accessibility_score=3.2,
            predicted_binding_affinity_kd_nm=12.5,
            target_protein_pdb_id="7TLL",
            admet_gate_status="pass",
        ),
        mcp_agent=McpAgentExecutionProvenance(
            agent_framework="chemcrow_agent",
            tools_invoked=["pubchem_lookup", "rdkit_sa_score", "opentrons_liquid_handler"],
            mcp_server_uris=["mcp://chem.lab.internal/v1"],
            governance_audit_logged=True,
            robotic_lab_handoff_ready=True,
        ),
    )

    assert report.molecular_drug is not None
    assert report.molecular_drug.qed_drug_likeness == pytest.approx(0.82)
    assert report.mcp_agent is not None
    assert report.mcp_agent.agent_framework == "chemcrow_agent"

    md = format_material_candidate_markdown_report(report)
    assert "Molecular & Bio-Therapeutic Pharmacological Identity" in md
    assert "CC(C)CC(NC(=O)C(F)(F)F)C(=O)NC(Cc1ccccc1)C=O" in md
    assert "7TLL" in md
    assert "0.820" in md
    assert "Model Context Protocol (MCP) Scientific Agent Session" in md
    assert "chemcrow_agent" in md
    assert "opentrons_liquid_handler" in md
    assert "mcp://chem.lab.internal/v1" in md
