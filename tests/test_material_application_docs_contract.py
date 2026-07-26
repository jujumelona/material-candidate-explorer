from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_application_docs_do_not_claim_automatic_specialist_execution() -> None:
    readme = _read("README.md")
    guide = _read("docs/APPLICATION_MATERIAL_DECISIONS.md")

    collapsed_readme = " ".join(readme.split())
    collapsed_guide = " ".join(guide.split())
    assert "application decision stage does not run a generator" in collapsed_readme
    assert "decision run does not execute a generator" in collapsed_guide
    for collapsed in (collapsed_readme, collapsed_guide):
        assert "manual handoff" in collapsed
    assert "No notebook automatically converts a multi-role" in collapsed_guide
    assert "non-executed DFT-input preparation" in collapsed_guide


def test_docs_separate_the_explicit_bulk_bridge_from_application_routing() -> None:
    readme = " ".join(_read("README.md").split())
    guide = _read("docs/APPLICATION_MATERIAL_DECISIONS.md")
    guide_collapsed = " ".join(guide.split())

    for contract in (
        "## Explicit bulk-crystal execution bridge",
        "discovery-os material-fusion-search",
        "--generator mattergen",
        "--rounds 4",
        "--max-generated-candidates 16",
        "--expert mattersim",
        "--expert chgnet",
        "does not turn a retrieval seed into a structure",
        "does not create an application-property score or scientific claim",
        "`RUN_SINGLE_ROLE_BULK_SEARCH`",
        "Broad or multi-role portfolios stay on the manual",
    ):
        assert contract in guide_collapsed

    assert "`material-fusion-search` is the separate execution bridge" in readme
    assert "global 8-32 candidate budget" in readme
    assert "retrieval seeds are never promoted into structures" in readme
    assert "application RAG is not a runtime validator" in readme


def test_application_rag_usage_matches_stage_specific_source_contract() -> None:
    guide = _read("docs/APPLICATION_MATERIAL_DECISIONS.md")

    assert "Leave `--rag-source` unset for the code-owned stage policy" in guide
    assert "OpenAlex only at `generation_prior` and\n`identity_novelty`" in guide
    assert "--rag-source openalex" not in guide
    assert "application-rag/generation_prior.json" not in guide
    assert "application-rag/" in guide
    assert "generation_prior.json" in guide
    assert "material-decision-run.json        # includes all stage receipts" in guide


def test_research_citations_are_kept_in_a_non_capability_document() -> None:
    workflow = _read("docs/DOMAIN_MATERIAL_WORKFLOWS.md")
    foundations = _read("docs/RESEARCH_FOUNDATIONS.md")

    assert "## Research basis and official implementations" not in workflow
    assert "## External design references" in workflow
    assert "live in [Research foundations](RESEARCH_FOUNDATIONS.md)" in workflow
    assert "This is a bibliography and claim-boundary document, not a feature inventory." in foundations
    assert "A citation does not mean that the cited model" in foundations
