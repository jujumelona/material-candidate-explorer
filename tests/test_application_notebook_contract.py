from __future__ import annotations

import json
from pathlib import Path


NOTEBOOK = (
    Path(__file__).resolve().parents[1] / "MATERIAL_APPLICATION_RECOMMENDER_T4.ipynb"
)


def _load_notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _notebook_source() -> str:
    notebook = _load_notebook()
    return "\n".join("".join(cell["source"]) for cell in notebook["cells"])


def test_application_notebook_is_clean_and_compilable() -> None:
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] >= 5

    cells = notebook["cells"]
    identifiers = [cell.get("id") for cell in cells]
    assert all(identifiers)
    assert len(identifiers) == len(set(identifiers))
    assert (
        "https://colab.research.google.com/github/jujumelona/"
        "material-candidate-explorer/blob/main/"
        "MATERIAL_APPLICATION_RECOMMENDER_T4.ipynb"
    ) in "".join(cells[0]["source"])

    for index, cell in enumerate(cells):
        source = "".join(cell["source"])
        assert "??" not in source
        if cell["cell_type"] == "code":
            assert cell.get("execution_count") is None
            assert cell.get("outputs") == []
            compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec")


def test_application_notebook_exposes_general_material_decision_settings() -> None:
    source = _notebook_source()

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

    for setting in (
        "APPLICATION_QUESTION",
        'MATERIAL_FIELD = "AUTO"',
        'MAIN_MODEL_ROUTING = "AUTO"',
        "REQUIRE_CONDITION_COMPLETE",
        "INCLUDE_RETRIEVAL_SEEDS",
        "RUN_RAG",
        "APPLICATION_CONTEXT_JSON",
        "EXPLICIT_ROLE_IDS",
        "CHEMICAL_SYSTEM",
        "RUN_SINGLE_ROLE_BULK_SEARCH",
        "BULK_SEARCH_ROLE_ID",
        "BULK_SEARCH_INPUT_MODE",
        "BULK_SEARCH_GOAL_FILE",
        "BULK_SEARCH_PARENT_FILE",
        "BULK_SEARCH_RUN_CONFIG_FILE",
        "BULK_SEARCH_ROUNDS",
        "BULK_SEARCH_TOTAL_CANDIDATES",
        "BULK_SEARCH_MAX_GENERATION_CALLS",
        "CANDIDATES_JSON",
        "OBSERVATIONS_JSON",
        "PREFERENCES_JSON",
        "RAG_MODEL_API_URL",
        "RAG_MODEL_NAME",
        "RAG_MODEL_TIMEOUT_SECONDS",
        "MATERIAL_FIELD_MODEL_API_URL",
        "MATERIAL_FIELD_MODEL_NAME",
        "MATERIAL_FIELD_MODEL_TIMEOUT_SECONDS",
        "MATTERGEN_API_URL",
        "MATTERSIM_API_URL",
        "CHGNET_API_URL",
        "CONTACT_EMAIL",
        "RAG_FROM_DATE",
        "RAG_TO_DATE",
        "RAG_MAX_RESULTS",
        "MATERIAL_RAG_MCP_URL",
        "MATERIAL_APPLICATION_RAG_MCP_TOOL",
        "MCP_TOOL_GENERATION_PRIOR",
        "MCP_TOOL_IDENTITY_NOVELTY",
        "MCP_TOOL_MLIP_DISAGREEMENT",
        "MCP_TOOL_RELAXATION_VALIDATION",
        "MCP_TOOL_DFT_HANDOFF",
        "MATERIAL_RAG_MCP_TIMEOUT_SECONDS",
        "MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP",
    ):
        assert setting in source

    assert 'MAIN_MODEL_ROUTING not in {"AUTO", "REQUIRED", "OFF"}' in source
    assert (
        "bool(RAG_MODEL_API_URL.strip()) != bool(RAG_MODEL_NAME.strip())"
        in source
    )
    assert (
        "bool(MATERIAL_RAG_MCP_URL.strip()) "
        "!= any(item.strip() for item in MCP_TOOL_VALUES)"
    ) in source
    for environment_name in (
        "MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR",
        "MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY",
        "MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT",
        "MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION",
        "MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF",
    ):
        assert environment_name in source
    assert 'hidden_environment("RAG_MODEL_API_KEY"' in source
    assert 'hidden_environment("MATERIAL_FIELD_MODEL_API_KEY"' in source
    assert 'hidden_environment("MATTERGEN_API_TOKEN"' in source
    assert 'hidden_environment("MATTERSIM_API_TOKEN"' in source
    assert 'hidden_environment("CHGNET_API_TOKEN"' in source
    assert 'hidden_environment("OPENALEX_API_KEY"' in source
    assert 'hidden_environment("MATERIAL_RAG_MCP_TOKEN"' in source
    assert "getpass(" in source


def test_application_notebook_runs_the_typed_decision_pipeline() -> None:
    source = _notebook_source()

    for contract in (
        "MaterialDecisionRunner",
        "MaterialApplicationCandidate",
        "MaterialApplicationObservation",
        "MaterialDecisionPreference",
        "MaterialApplicationCandidate.model_validate(item, strict=True)",
        "MaterialApplicationObservation.model_validate(item, strict=True)",
        "MaterialDecisionPreference.model_validate(item, strict=True)",
        "explicit_role_ids=role_ids",
        "require_condition_complete=REQUIRE_CONDITION_COMPLETE",
        "include_retrieval_seeds=INCLUDE_RETRIEVAL_SEEDS",
        "candidates=candidate_rows",
        "observations=observation_rows",
        "preferences=preference_rows",
        "run_rag=RUN_RAG",
        "rag_from_date=rag_from_date",
        "rag_to_date=rag_to_date",
        "rag_max_results_per_query=RAG_MAX_RESULTS",
        "decision_run.brief.material_field",
        "decision_run.brief.question_kind",
        "decision_run.report.cross_role_ranking_performed",
        "decision_run.generation_or_specialist_execution_performed",
        "decision_run.rag_bundle_id",
        "decision_run.rag_stage_receipts",
        "item.evidence_stage: item.status",
    ):
        assert contract in source

    assert "rag_sources=" not in source
    assert "LiteratureSource." not in source
    assert source.index("MaterialDecisionRunner(artifact_root=ARTIFACT_ROOT).run(") < (
        source.index("decision_run.report.role_recommendations")
    )
    assert "MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb" in source


def test_application_notebook_keeps_score_and_evidence_semantics_explicit() -> None:
    source = _notebook_source()

    for output_field in (
        '"role": portfolio.role_id',
        '"candidate": item.candidate.material_or_stack',
        '"condition_group": item.comparison_group_id or "UNKNOWN"',
        '"rank": item.rank_within_role_and_condition',
        '"pareto_front": item.pareto_front',
        '"hard_gates": item.hard_gate_status',
        '"decision_score": item.pool_relative_decision_score',
        '"score_semantics": item.score_semantics',
        '"evidence_complete_%": item.evidence_completeness_score',
        '"uncertainty": item.evidence_uncertainty_status',
        '"model_disagreement": item.candidate.model_disagreement',
        '"why": "; ".join(item.why_selected)',
        '"why_not_top": "; ".join(item.why_not_top)',
        '"tradeoffs": "; ".join(item.main_tradeoffs)',
        '"citations": "; ".join(c.doi or c.record_id for c in item.citations)',
        '"next_validation": item.next_validations[0]',
        '"mcp_tool": item.selected_mcp_tool or "UNCONFIGURED"',
        '"source_bundle": item.source_bundle_id or "NONE"',
    ):
        assert output_field in source

    for boundary in (
        "never ranks unlike roles together",
        "literature citations do not receive material-performance credit",
        "condition-complete named-validator observations and explicit weights",
        "Missing evidence stays `UNKNOWN`",
        "scores are role- and condition-local decision support",
        "not probabilities",
        "`UNKNOWN` is not zero or pass",
        "Retrieval seeds need exact ",
        "source closure and named-validator observations",
        "never executes a generator, specialist property model",
        "never automatically chooses among unlike roles",
    ):
        assert boundary in source

    assert "shutil.make_archive" in source
    assert "files.download(archive_path)" in source


def test_application_notebook_writes_only_a_manual_t4_handoff() -> None:
    source = _notebook_source()

    for contract in (
        '"automatic_role_selection_performed": False',
        '"automatic_transfer_or_execution_performed": False',
        '"application_decision_run_generation_or_specialist_execution_performed"',
        '"operator_context_required"',
        '"operator_chemical_system_required"',
        '"not_compatible_with_bulk_crystal_t4"',
        '"bulk_triage_only_interface_or_device_validation_required"',
        '"manual_bulk_crystal_triage_ready"',
        '"DISCOVERY_PROMPT"',
        '"MATERIAL_FIELD"',
        '"CHEMICAL_SYSTEM"',
        '"MATERIAL_PROBLEM_CONTEXT_JSON"',
        '"application-to-crystal-t4-handoff.json"',
        "Choose exactly one role",
    ):
        assert contract in source

    assert source.index("decision_run = MaterialDecisionRunner") < source.index(
        "downstream_handoff = {"
    )
    assert source.index("downstream_handoff = {") < source.index(
        "shutil.make_archive"
    )


def test_application_notebook_bulk_search_is_explicit_and_fail_closed() -> None:
    source = _notebook_source()

    for contract in (
        "if not RUN_SINGLE_ROLE_BULK_SEARCH:",
        "if len(selected_roles) != 1:",
        "EXPLICIT_ROLE_IDS to one role",
        'selected_role.role_id != BULK_SEARCH_ROLE_ID.strip()',
        '"bulk_crystal" not in selected_role.representation_scopes',
        "decision_run.brief.ready_for_condition_complete_scoring",
        "decision_run.brief.field_plan.resolution.requires_operator_choice",
        "DiscoveryGoal.model_validate_json",
        "Candidate.model_validate_json",
        "WorkspaceRunConfig.model_validate_json",
        '"energy_per_atom", "max_force"',
        'run_config_input.generator_id != "mattergen"',
        '"material-fusion-search"',
        '"--no-control-sweep"',
        '"--max-generation-calls"',
        '"--max-generated-candidates"',
        '"--expert", "mattersim"',
        '"--expert", "chgnet"',
        '"--required-evaluator", "mattersim"',
        '"--required-evaluator", "chgnet"',
        '"explicit-bulk-crystal-search-report.json"',
        '"application_property_scoring_performed": False',
        '"application_claim_created": False',
    ):
        assert contract in source

    assert "retrieval_seed_promoted_to_structure" not in source
    assert source.index("if len(selected_roles) != 1:") < source.index(
        '"material-fusion-search"'
    )
    assert source.index('"material-fusion-search"') < source.index(
        "shutil.make_archive"
    )
