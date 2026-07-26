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
    MultiExpertReliabilitySummary,
    RichMaterialCandidateReport,
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
