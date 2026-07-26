from __future__ import annotations

from datetime import datetime, timezone

import pytest

from discovery_os.literature_rag import (
    LiteratureQuery,
    LiteratureRagError,
    LiteratureSource,
    MultiSourceLiteratureRetriever,
    PromptSearchPlanner,
)
from discovery_os.hashing import stable_hash
from discovery_os.material_domains import MATERIAL_EVIDENCE_STAGES
from discovery_os.material_stage_research import (
    STAGE_RESEARCH_POLICIES,
    build_stage_query_blueprints,
    stage_research_policy,
)


def test_five_stages_have_distinct_closed_query_and_mcp_policies() -> None:
    assert set(STAGE_RESEARCH_POLICIES) == set(MATERIAL_EVIDENCE_STAGES)
    policy_ids = set()
    scope_arguments = set()
    intent_sets = set()
    for stage in MATERIAL_EVIDENCE_STAGES:
        policy = stage_research_policy(stage)
        policy_ids.add(policy.policy_id)
        scope_arguments.add(policy.mcp.scope_argument)
        intent_sets.add(tuple(item.intent_id for item in policy.query_intents))
        assert policy.stage == stage
        assert len(policy.query_intents) >= 5
        assert len(policy.research_bases) >= 4
        assert all(item.source_url.startswith("https://") for item in policy.research_bases)
        assert policy.mcp.required_record_fields == [
            "source_id",
            "title",
            "record_type",
            "support_text",
            "provenance",
            "stage_metadata",
        ]
        assert {
            record_type
            for intent in policy.query_intents
            for record_type in intent.expected_record_types
        } == set(policy.mcp.allowed_record_types)
    assert len(policy_ids) == 5
    assert len(scope_arguments) == 5
    assert len(intent_sets) == 5


def test_policy_blueprints_expand_every_intent_to_every_selected_source() -> None:
    policy = stage_research_policy("mlip_disagreement")
    blueprints = build_stage_query_blueprints(
        stage="mlip_disagreement",
        chemical_system="Li-Fe-P-O",
        material_field="battery_electrode",
        application_subtype="cathode",
        problem_context={"temperature_k": 300},
        composition_keys=["LiFePO4"],
        candidate_refs=["candidate-1:v1:" + "a" * 64],
        focus_terms=["same-composition relative energies only"],
    )
    plan = PromptSearchPlanner().plan(
        "Rank LiFePO4 cathode candidates",
        sources=[LiteratureSource.CROSSREF, LiteratureSource.MCP],
        query_blueprints=blueprints,
        query_policy_id=policy.policy_id,
        query_policy_version=policy.policy_version,
    )
    assert plan.required_intent_ids == [
        item.intent_id for item in policy.query_intents
    ]
    assert len(plan.queries) == len(policy.query_intents) * 2
    for source in (LiteratureSource.CROSSREF, LiteratureSource.MCP):
        assert {
            item.intent_id for item in plan.queries if item.source == source
        } == set(plan.required_intent_ids)
    assert all(
        item.mcp_arguments["mlip_scope"]["property_score_authority"] is False
        for item in plan.queries
    )


def test_stage_query_context_rejects_secret_fields() -> None:
    with pytest.raises(ValueError, match="cannot contain secrets"):
        build_stage_query_blueprints(
            stage="dft_handoff",
            chemical_system="Li-O",
            problem_context={"api_key": "must-not-leave-process"},
        )


def test_typed_generation_mcp_record_is_preserved_with_provenance() -> None:
    policy = stage_research_policy("generation_prior")
    calls = []

    class Client:
        endpoint = "https://mcp.example/evidence"

        def call_tool(self, name, arguments):
            calls.append((name, arguments))
            row = {
                "source_id": "recipe-1",
                "title": "Li2O target synthesis",
                "record_type": "synthesis_success",
                "support_text": (
                    "Li2O was synthesized at 800 K under argon and "
                    "characterized by XRD."
                ),
                "provenance": {
                    "provider": "fixture",
                    "provider_version": "1",
                    "snapshot_id": "snapshot-1",
                    "source_locator": "doi:10.1000/recipe",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "request_hash": stable_hash(arguments),
                },
                "stage_metadata": {
                    "chemical_system": "Li-O",
                    "composition": "Li2O",
                    "outcome": "successful_target",
                    "conditions": {
                        "temperature": "800 K",
                        "atmosphere": "argon",
                    },
                    "evidence_polarity": "supports",
                },
            }
            row["provenance"]["record_hash"] = stable_hash(row)
            return {"records": [row]}

    retriever = MultiSourceLiteratureRetriever(
        mcp_client=Client(),
        mcp_tool="search_generation_evidence",
        mcp_record_contract=policy.mcp.runtime_contract("generation_prior"),
    )
    blueprint = build_stage_query_blueprints(
        stage="generation_prior",
        chemical_system="Li-O",
    )[0]
    query = LiteratureQuery(
        query_id="generation-query",
        source=LiteratureSource.MCP,
        query=blueprint.query,
        rationale=blueprint.rationale,
        intent_id=blueprint.intent_id,
        expected_record_types=["synthesis_success"],
        mcp_arguments=blueprint.mcp_arguments,
    )
    records = retriever._search_mcp(query)
    assert records[0].raw_metadata["mcp_record_type"] == "synthesis_success"
    assert records[0].raw_metadata["mcp_provenance"]["snapshot_id"] == "snapshot-1"
    assert calls[0][1]["stage"] == "generation_prior"
    assert calls[0][1]["intent_id"] == blueprint.intent_id
    assert calls[0][1]["generation_scope"]["property_score_authority"] is False


def test_typed_mcp_record_missing_stage_provenance_fails_closed() -> None:
    policy = stage_research_policy("identity_novelty")

    class Client:
        endpoint = "https://mcp.example/evidence"

        def call_tool(self, name, arguments):
            return {
                "records": [
                    {
                        "source_id": "structure-1",
                        "title": "Li2O structure",
                        "record_type": "crystallographic_entry",
                        "support_text": "A crystallographic entry for Li2O.",
                        "provenance": {"provider": "fixture"},
                        "stage_metadata": {
                            "database_name": "fixture",
                            "database_entry_id": "1",
                            "formula": "Li2O",
                            "structure_locator": "fixture:1",
                            "match_scope": "formula-only",
                        },
                    }
                ]
            }

    retriever = MultiSourceLiteratureRetriever(
        mcp_client=Client(),
        mcp_tool="search_identity_evidence",
        mcp_record_contract=policy.mcp.runtime_contract("identity_novelty"),
    )
    blueprint = build_stage_query_blueprints(
        stage="identity_novelty",
        chemical_system="Li-O",
    )[0]
    with pytest.raises(LiteratureRagError, match="structured contract"):
        retriever._search_mcp(
            LiteratureQuery(
                query_id="identity-query",
                source=LiteratureSource.MCP,
                query=blueprint.query,
                rationale=blueprint.rationale,
                intent_id=blueprint.intent_id,
                expected_record_types=list(blueprint.expected_record_types),
                mcp_arguments=blueprint.mcp_arguments,
            )
        )
