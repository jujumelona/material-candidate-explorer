from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[1] / "MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb"


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_t4_notebook_is_clean_and_compilable() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5
    cells = notebook["cells"]
    identifiers = [cell.get("id") for cell in cells]
    assert all(identifiers)
    assert len(identifiers) == len(set(identifiers))
    assert "colab.research.google.com/github/jujumelona/material-candidate-explorer" in "".join(
        cells[0]["source"]
    )
    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        assert "??" not in source
        assert not (
            source.startswith("__") and source.endswith("__")
        ), f"unreplaced placeholder in cell {index}"
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_t4_notebook_preserves_the_material_screening_contract() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for name in (
        "requested_samples",
        "raw_model_structures",
        "parsed_structures",
        "exact_file_unique",
        "crystallographically_unique",
        "geometry_valid",
        "mlip_evaluated",
        "relaxation_converged",
        "ranked_candidates",
    ):
        assert name in source
    for required in (
        "TOTAL_CANDIDATES = 16",
        "if not 8 <= TOTAL_CANDIDATES <= 32",
        "SEARCH_ROUNDS = 8",
        "MAX_GENERATION_CALLS = 32",
        "FRONTIER_WIDTH = 1",
        "BASE_GUIDANCE_ALPHA = 0.5",
        "BASE_CFG_GAMMA = 2.0",
        "GoalConstraint",
        'constraint_id="immutable-chemical-system"',
        'property_name="chemical_system"',
        'operator="eq"',
        "value=chemical_system",
        "hard=True",
        "target_energy_above_hull_eV_atom=None",
        '"fusion-search"',
        "PersistedFusionSearchReport",
        "ExpertEvidenceStore",
        "search_report.rounds_completed < 3",
        "generation_funnel_hashes",
        "group_crystal_structures",
        "crystal_matcher_settings = asdict(grouping.matcher_settings)",
        "grouping.ambiguous_comparisons",
        '"matcher_settings": asdict(relaxed_grouping.matcher_settings)',
        "relaxed_grouping.ambiguous_comparisons",
        '"deduplicated": False',
        '"mattersim"',
        '"chgnet"',
        '"/v1/relax"',
        "require_stress_comparison=True",
        "require_relaxed_structure_comparison=True",
        "rank_composition_scoped_pareto",
        "CompositionEnergyPair",
        "composition_relative_energy_disagreement",
        "COMPOSITION_ENERGY_ALIGNMENT_PATH",
        '"raw_cross_model_energy_offsets_used_for_risk": False',
        "payload.final_energy_eV / relaxed_atom_count",
        "max_force_eV_A=payload.final_max_force_eV_A",
        'relative_energy=relative_energy',
        'row["predictions"]["mattersim"]',
        'row["predictions"]["chgnet"]',
        'force_rmse_eV_A=row["force_rmse_eV_A"]',
        '"initial_common_geometry_force_rmse_eV_A"',
        '"initial_common_geometry_stress_diff_GPa"',
        '"energy_risk": "composition_relative_independently_relaxed_geometries"',
        '"relaxed_force_cross_model_comparison_performed": False',
        '"relaxed_stress_cross_model_comparison_performed": False',
        'common_geometry_mattersim=row["predictions"]["mattersim"]',
        'common_geometry_chgnet=row["predictions"]["chgnet"]',
        'common_geometry_alignment_id=row["initial_common_geometry_alignment_id"]',
        '"schema": "common-geometry-mlip-alignment-v1"',
        '"initial_common_geometry_alignment_id"',
        "singleton_composition_fraction",
        "SINGLETON_COMPOSITION_FRACTION_THRESHOLD = 0.5",
        '"insufficient_within_composition_replication"',
        '"triage_by_force_diversity_uncertainty_not_cross_formula_stability_ranking"',
        '"rag_numeric_stability_substitution_performed": False',
        '"initial_feature_predictions"',
        '"relaxed_ranking_predictions"',
        '"unranked_failed_relaxation_requires_operator_review"',
        "MaterialsProjectStructureLookup",
        "PortablePeriodicDFTInputBackend",
        "reserve_external_no_match_portfolio_slot",
        "novelty-dft-portfolio-receipt.json",
        '"unknown_external_novelty_receives_credit": False',
        "generation-profile-matrix.json",
        "applied_generation_profiles",
        "applied_target_profiles",
        "len(applied_alphas) < 2",
        "len(applied_gammas) < 2",
        "len(applied_target_profiles) < 3",
        'funnel["crystallographically_unique"] != TOTAL_CANDIDATES',
        '"mattersim": "sha256:e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5"',
        '"chgnet": "sha256:d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1"',
        '"cross_stoichiometry_energy_comparison_performed": False',
        '"dft_executed": False',
        '"zip_or_extxyz_merge_performed": False',
    ):
        assert required in source
    assert "ltol=0.2" not in source
    assert "stol=0.3" not in source
    assert "angle_tol=5.0" not in source
    assert 'property_name="energy_above_hull"' not in source
    assert "maximum_energy_abs_difference_eV_atom" not in source
    assert "predictions_for_disagreement" not in source
    assert "ranking_vector_disagreement" not in source
    assert "managed-unattested" not in source


def test_complete_common_geometry_disagreement_can_stay_low() -> None:
    from discovery_os.materials_screening import (
        MLIPScreeningPrediction,
        classify_model_disagreement,
    )
    from discovery_os.mlip_reliability import (
        CompositionEnergyPair,
        composition_relative_energy_disagreement,
    )

    relative_energy = composition_relative_energy_disagreement(
        [
            CompositionEnergyPair(
                candidate_id="candidate-a",
                reduced_composition="Li2O",
                first_model_id="mattersim",
                second_model_id="chgnet",
                first_energy_per_atom_eV=-2.0,
                second_energy_per_atom_eV=-20.0,
                alignment_artifact_id="aligned-low-panel",
            ),
            CompositionEnergyPair(
                candidate_id="candidate-b",
                reduced_composition="Li2O",
                first_model_id="mattersim",
                second_model_id="chgnet",
                first_energy_per_atom_eV=-1.0,
                second_energy_per_atom_eV=-19.0,
                alignment_artifact_id="aligned-low-panel",
            ),
        ]
    )[0]
    disagreement = classify_model_disagreement(
        MLIPScreeningPrediction(
            expert_id="mattersim",
            energy_per_atom_eV=-5.0,
            max_force_eV_A=0.03,
            stress_norm=1.0,
            stress_unit="GPa",
        ),
        MLIPScreeningPrediction(
            expert_id="chgnet",
            energy_per_atom_eV=7.0,
            max_force_eV_A=0.04,
            stress_norm=1.01,
            stress_unit="GPa",
        ),
        force_rmse_eV_A=0.01,
        relative_energy=relative_energy,
        relaxed_structure_match=True,
        require_stress_comparison=True,
        require_relaxed_structure_comparison=True,
    )

    assert disagreement.raw_energy_per_atom_abs_diff_eV == 12.0
    assert disagreement.energy_comparison_basis == "composition_relative_aligned"
    assert disagreement.composition_relative_energy_abs_diff_eV_atom == 0.0
    assert disagreement.stress_norm_abs_diff_GPa is not None
    assert abs(disagreement.stress_norm_abs_diff_GPa - 0.01) < 1e-12
    assert disagreement.risk == "low"
    assert disagreement.dft_escalation is False


def test_t4_notebook_runs_all_stage_specific_evidence_routes() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for stage in (
        "GENERATION_PRIOR",
        "IDENTITY_NOVELTY",
        "MLIP_DISAGREEMENT",
        "RELAXATION_VALIDATION",
        "DFT_HANDOFF",
    ):
        assert f"ValidationEvidenceStage.{stage}" in source
    for boundary in (
        "ENABLE_STAGE_EVIDENCE = True",
        "RAG_MODEL_API_URL",
        "RAG_MODEL_NAME",
        "MATERIAL_RAG_MCP_URL",
        "MATERIAL_RAG_MCP_TOOL",
        "MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR",
        "MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY",
        "MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT",
        "MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION",
        "MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF",
        "getpass.getpass",
        '"--rag-bundle"',
        '"property_score_created"',
        '"configured-tool-only"',
        '"administrator-configured-allowlist-only"',
        '"unknown_is_pass": False',
        '"operator-procedure-not-scientific-validator"',
        'PROJECT_SKILL_ID = "material-candidate-validation"',
        "report.handoff",
        "report.mcp_contract_status",
        "checkpoint-receipt.json",
        "EXPECTED_EVIDENCE_STAGES",
    ):
        assert boundary in source
    # The notebook owns one audited wrapper around the reusable router. Five
    # checkpoints call it; prompts or model output never bypass the wrapper.
    assert source.count("stage_evidence_router.run(") == 1
    assert source.count("run_stage_evidence_checkpoint(") == 6
    assert "stage-validation-evidence-index.json" in source

    assert source.index("generation_evidence_run = run_stage_evidence_checkpoint(") < source.index(
        '"fusion-search"'
    )
    assert source.index("identity_novelty_assessments = StagedNoveltyAssessor(") < source.index(
        "identity_evidence_run = run_stage_evidence_checkpoint("
    )
    assert source.index("disagreement_risk_counts =") < source.index(
        "mlip_evidence_run = run_stage_evidence_checkpoint("
    )
    assert source.index("RELAXATION_CHECKPOINT_PATH.write_text(") < source.index(
        "relaxation_evidence_run = run_stage_evidence_checkpoint("
    )
    assert source.index("dft_report = PortablePeriodicDFTInputBackend().prepare_inputs(") < source.index(
        "dft_evidence_run = run_stage_evidence_checkpoint("
    )


def test_t4_notebook_routes_one_audited_material_field_fail_closed() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert 'MATERIAL_FIELD = "AUTO"' in source
    assert 'MAIN_AI_FIELD_ROUTING = "AUTO"' in source
    for field in (
        "general_inorganic",
        "battery_electrode",
        "solid_electrolyte",
        "superconductor",
        "heterogeneous_catalyst",
        "semiconductor",
        "photovoltaic_absorber",
        "thermoelectric",
        "magnetic_material",
        "ferroelectric_piezoelectric",
        "structural_alloy",
        "porous_framework",
    ):
        assert field in source
    for contract in (
        "build_material_domain_plan(",
        "build_main_model_material_field_classifier_from_environment(",
        'MAIN_AI_FIELD_ROUTING == "REQUIRED"',
        'MATERIAL_FIELD_MODEL_API_URL = ""',
        'MATERIAL_FIELD_MODEL_NAME = ""',
        "MATERIAL_PROBLEM_CONTEXT_JSON",
        "main_field_classifier.classify(",
        "chemical_system=chemical_system",
        "problem_context=material_problem_context",
        "model_run=main_model_run",
        "domain_plan.main_model_run.model_dump",
        "domain_plan.main_model_run.endpoint_or_tool_selection_performed",
        "domain_plan.resolution.requires_operator_choice",
        "domain_plan.missing_required_context",
        "if not domain_plan.field_route_ready",
        "selected_validation_profile = get_validation_profile(",
        "domain=domain_plan.profile.discovery_domain",
        "validation_profile_id=selected_validation_profile.profile_id",
        'candidate_types=[CandidateType.CRYSTAL]',
        "MATERIAL_DOMAIN_PLAN_PATH",
        '"material-domain-plan.json"',
        "MATERIAL_DOMAIN_FINAL_AUDIT_PATH",
        '"material-domain-final-audit.json"',
        "domain_plan.unexecuted_required_properties",
        '"field_specific_property_calculation_executed": False',
        "GENERIC_T4_FIELD_PROPERTY_BOUNDARY",
    ):
        assert contract in source
    assert source.count(
        "material_field=domain_plan.profile.material_field"
    ) == 5
    assert source.count(
        "application_subtype=domain_plan.resolution.application_subtype"
    ) == 5
    assert source.count("problem_context=material_problem_context") >= 7
    assert source.index(
        "main_field_classifier.classify("
    ) < source.index("draft_parent = Candidate(")
    assert source.index(
        "domain_plan = build_material_domain_plan("
    ) < source.index("draft_parent = Candidate(")


def test_t4_notebook_uses_one_global_adaptive_fusion_search() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert source.count('"fusion-search"') == 1
    assert "fusion-iterate" not in source
    for contract in (
        "GENERATION_BATCH_SIZE = 1",
        "minimum_rounds_for_budget = 1 + (TOTAL_CANDIDATES - 1 + 4) // 5",
        '"--rounds", str(SEARCH_ROUNDS)',
        '"--frontier-width", str(FRONTIER_WIDTH)',
        '"--max-generation-calls", str(MAX_GENERATION_CALLS)',
        '"--max-generated-candidates", str(TOTAL_CANDIDATES)',
        '"--no-control-sweep"',
        '"--required-evaluator", "mattersim"',
        '"--required-evaluator", "chgnet"',
        '"pareto", "stability", "target_property", "novelty", "expert_disagreement"',
        "search_report.budget_usage.generated_candidates",
        "search_report.rounds_completed < 3",
        "record.generation_provenance is not None",
        "evidence_store.load(evidence_id)",
        '"feature_refs": feature_refs',
        'applied.get("diffusion_guidance_factor") != BASE_CFG_GAMMA',
        "GENERATION_PROFILE_MATRIX_PATH",
        "expected_gamma = round(requested_alpha * float(alpha_to_gamma_scale), 8)",
        "unique_target_energy_above_hull_eV_atom",
        'CREDENTIAL_ENV_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")',
        "if name not in credential_env_values",
        "for secret_name, secret in credential_env_values.items()",
    ):
        assert contract in source

    # A stage bundle is passed only after the typed generation handoff permits
    # steering; absent or failed evidence leaves the search on validator-only control.
    assert source.index("if generation_evidence_run.report.handoff.can_steer_generation:") < source.index(
        'command.extend(["--rag-bundle", str(generation_rag_bundle_path)])'
    )


def test_t4_notebook_uses_a_bounded_external_novelty_dft_portfolio() -> None:
    notebook = _load_notebook()
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])

    assert source.index("base_dft_refs = select_dft_handoff_refs(") < source.index(
        "novelty_portfolio = reserve_external_no_match_portfolio_slot("
    )
    assert source.index(
        "novelty_portfolio = reserve_external_no_match_portfolio_slot("
    ) < source.index(
        "dft_report = PortablePeriodicDFTInputBackend().prepare_inputs("
    )
    for contract in (
        "base_candidate_refs=base_dft_refs",
        "eligible_candidate_refs=eligible_dft_refs",
        "assessments=novelty_assessments",
        "max_novelty_slots=1",
        "selected_dft_refs = novelty_portfolio.selected_candidate_refs",
        "NOVELTY_PORTFOLIO_PATH",
        'external_database.status) != "no_match"',
    ):
        assert contract in source

    assert "property-space-diversity (the internal branch identifier is `novelty`)" in source
    assert "External database novelty is assessed later" in source
