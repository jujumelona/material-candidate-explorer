from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_mcp_guide_documents_every_executable_stage_policy() -> None:
    guide = _read("docs/MCP_RAG.md")

    expected = {
        "generation_prior": (
            "material-generation-evidence-v2",
            "successful_target",
            "generation_scope",
        ),
        "identity_novelty": (
            "material-identity-evidence-v2",
            "federated_structure_records",
            "identity_scope",
        ),
        "mlip_disagreement": (
            "material-mlip-evidence-v2",
            "uncertainty_and_extrapolation",
            "mlip_scope",
        ),
        "relaxation_validation": (
            "material-relaxation-evidence-v2",
            "phonon_instability",
            "relaxation_scope",
        ),
        "dft_handoff": (
            "material-dft-evidence-v2",
            "pseudopotential_verification",
            "dft_scope",
        ),
    }
    for stage, values in expected.items():
        assert stage in guide
        for value in values:
            assert value in guide

    for field in (
        "source_id",
        "title",
        "record_type",
        "support_text",
        "provenance",
        "stage_metadata",
        "provider",
        "provider_version",
        "snapshot_id",
        "source_locator",
        "retrieved_at",
        "request_hash",
        "record_hash",
    ):
        assert field in guide


def test_stage_guides_preserve_missing_intent_and_authority_boundaries() -> None:
    combined = " ".join(
        "\n".join(
            [
                _read("README.md"),
                _read("docs/MCP_RAG.md"),
                _read("docs/STAGE_VALIDATION_EVIDENCE.md"),
                _read("docs/DOMAIN_MATERIAL_WORKFLOWS.md"),
            ]
        ).split()
    )

    for phrase in (
        "missing required intent",
        "`partial`",
        "`unknown`",
        "not runtime validation",
        "cannot steer",
        "title-only",
    ):
        assert phrase in combined

    assert "all papers are implemented" not in combined.casefold()
    assert "every paper is implemented" not in combined.casefold()


def test_readme_keeps_optional_mcp_configuration_fail_open_for_retrieval_only() -> None:
    readme = _read("README.md")
    mcp_guide = _read("docs/MCP_RAG.md")

    assert "Leave the paired RAG or MCP endpoint fields blank to skip that integration." in readme
    assert "Leaving the URL and every tool field blank" in mcp_guide
    assert "no-MCP configuration" in mcp_guide
    assert "never interpreted as novelty" in readme
