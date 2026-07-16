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
        '"tight"',
        '"balanced"',
        '"broad"',
        '"explore"',
        "generation_funnel_hashes",
        "group_crystal_structures",
        '"mattersim"',
        '"chgnet"',
        '"/v1/relax"',
        "require_stress_comparison=True",
        "require_relaxed_structure_comparison=True",
        "rank_composition_scoped_pareto",
        "MaterialsProjectStructureLookup",
        "PortablePeriodicDFTInputBackend",
        '"cross_stoichiometry_energy_comparison_performed": False',
        '"dft_executed": False',
        '"zip_or_extxyz_merge_performed": False',
    ):
        assert required in source


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
        "getpass.getpass",
        '"--decision-context"',
        '"property_score_created"',
        '"configured-tool-only"',
        "EXPECTED_EVIDENCE_STAGES",
    ):
        assert boundary in source
    assert source.count("stage_evidence_router.run(") == 5
    assert "stage-validation-evidence-index.json" in source
