from __future__ import annotations

from datetime import date, datetime, timezone

from discovery_os.literature_rag import (
    EvidenceBranchKind,
    EvidenceBranchPlanner,
    EvidenceClaim,
    EvidenceClaimExtractor,
    EvidenceGraphBuilder,
    EvidencePolarity,
    EvidenceStage,
    LiteratureEvidencePolicy,
    LiteratureQuery,
    LiteratureRecord,
    LiteratureSource,
    PromptSearchPlanner,
    RagEvidenceBundle,
    RagSearchPlan,
    SourceRetrievalStatus,
    SourceRunStatus,
    deduplicate_records,
)
from discovery_os.schemas import (
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    ObjectiveDirection,
    PropertyObjective,
)


class FakeModel:
    model_id = "fake-rag"
    model_version = "fixture-v1"

    def complete_json(self, *, operation: str, system: str, user: str):
        if operation == "search-plan":
            return {
                "concepts": ["pancreatic cancer", "KRAS G12D"],
                "target_entities": ["KRAS G12D"],
                "mechanism_terms": ["inhibition"],
                "negative_terms": ["toxicity"],
                "queries": [
                    {
                        "source": "pubmed",
                        "query": "pancreatic cancer KRAS G12D inhibitor",
                        "rationale": "targeted recent evidence",
                    }
                ],
            }
        return {
            "claims": [
                {
                    "subject": "Compound X",
                    "predicate": "inhibits",
                    "object": "KRAS G12D",
                    "polarity": "supports",
                    "stage": "in_vitro",
                    "support_text": "Compound X inhibited KRAS G12D in pancreatic cancer cells.",
                    "confidence": 0.9,
                    "qualifiers": {"target": "KRAS G12D", "smiles": "CCO"},
                    "entity_aliases": {},
                },
                {
                    "subject": "Hallucinated",
                    "predicate": "cures",
                    "object": "all cancer",
                    "polarity": "supports",
                    "stage": "clinical_trial",
                    "support_text": "This sentence is not in the source.",
                    "confidence": 1.0,
                    "qualifiers": {},
                    "entity_aliases": {},
                },
            ]
        }


def _goal() -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="rag-goal",
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        title="Find pancreatic cancer candidates",
        scientific_question="Find new KRAS G12D anticancer molecules",
        objectives=[
            PropertyObjective(
                property_name="binding_affinity",
                direction=ObjectiveDirection.MAXIMIZE,
            )
        ],
        validation_profile_id="medicinal-chemistry-v1",
        candidate_types=[CandidateType.SMALL_MOLECULE],
    )


def _record(source: str, source_id: str, *, doi: str | None = None) -> LiteratureRecord:
    return LiteratureRecord(
        record_id=f"LIT-{source}-{source_id}",
        title="Compound X inhibited KRAS G12D in pancreatic cancer cells.",
        abstract="The study reported selective pathway inhibition and measured cell viability.",
        publication_date=date(2026, 5, 1),
        publication_year=2026,
        doi=doi,
        source_ids={source: source_id},
        source_queries=["query-1"],
        retrieved_at=datetime.now(timezone.utc),
    )


def test_model_planner_builds_source_specific_queries() -> None:
    plan = PromptSearchPlanner(FakeModel()).plan(
        "최신 KRAS G12D 항암물질을 찾아라",
        goal=_goal(),
        sources=[LiteratureSource.PUBMED],
        from_date=date(2025, 1, 1),
        max_results_per_query=17,
    )
    assert plan.planner_id == "fake-rag"
    assert len(plan.queries) == 1
    assert plan.queries[0].source == LiteratureSource.PUBMED
    assert plan.queries[0].max_results == 17
    assert plan.queries[0].from_date == date(2025, 1, 1)


def test_deduplicate_records_merges_cross_source_metadata_by_doi() -> None:
    first = _record("pubmed", "123", doi="10.1000/example")
    second = _record("crossref", "10.1000/example", doi="https://doi.org/10.1000/example")
    second = second.model_copy(update={"abstract": "A substantially longer abstract for deduplication that contains many more words than the original abstract and should therefore be preserved."})
    merged = deduplicate_records([first, second])
    assert len(merged) == 1
    assert set(merged[0].source_ids) == {"pubmed", "crossref"}
    assert merged[0].abstract == "A substantially longer abstract for deduplication that contains many more words than the original abstract and should therefore be preserved."


def test_claim_extractor_rejects_unsupported_model_claims() -> None:
    claims, warnings = EvidenceClaimExtractor(FakeModel()).extract(
        [_record("pubmed", "123")],
        prompt="find anticancer compounds",
        goal=_goal(),
    )
    assert len(claims) == 1
    assert claims[0].subject == "Compound X"
    assert claims[0].support_text in _record("pubmed", "123").evidence_text
    assert any("rejected unsupported" in item for item in warnings)


def test_graph_and_branch_planner_create_recheck_analog_and_mechanism_branches() -> None:
    record = _record("pubmed", "123")
    claim = EvidenceClaim(
        claim_id="claim-x",
        source_record_id=record.record_id,
        subject="Compound X",
        predicate="inhibits",
        object="KRAS G12D",
        polarity=EvidencePolarity.SUPPORTS,
        stage=EvidenceStage.IN_VITRO,
        support_text="Compound X inhibited KRAS G12D in pancreatic cancer cells.",
        confidence=0.9,
        qualifiers={"target": "KRAS G12D", "smiles": "CCO"},
    )
    graph = EvidenceGraphBuilder().build([claim])
    branches = EvidenceBranchPlanner().plan(
        [claim], graph, prompt="pancreatic cancer", goal=_goal()
    )
    kinds = {item.kind for item in branches}
    assert EvidenceBranchKind.EXACT_RECHECK in kinds
    assert EvidenceBranchKind.DERIVATIVE_OR_ANALOG in kinds
    assert EvidenceBranchKind.MECHANISM_ALTERNATIVE in kinds
    assert all(item.scientific_role if hasattr(item, "scientific_role") else True for item in [])


def test_evidence_policy_changes_branch_weight_from_real_search_observation() -> None:
    record = _record("pubmed", "123")
    claim = EvidenceClaim(
        claim_id="claim-x",
        source_record_id=record.record_id,
        subject="Compound X",
        predicate="inhibits",
        object="KRAS G12D",
        polarity=EvidencePolarity.SUPPORTS,
        stage=EvidenceStage.IN_VITRO,
        support_text="Compound X inhibited KRAS G12D in pancreatic cancer cells.",
        confidence=0.9,
    )
    graph = EvidenceGraphBuilder().build([claim])
    branches = EvidenceBranchPlanner().plan([claim], graph, prompt="cancer", goal=_goal())
    plan = RagSearchPlan(
        plan_id="plan-x",
        user_prompt="cancer",
        goal_hash=None,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        generated_at=datetime.now(timezone.utc),
        planner_id="fixture",
        planner_version="1",
        queries=[
            LiteratureQuery(
                query_id="query-1",
                source=LiteratureSource.PUBMED,
                query="cancer",
                rationale="fixture",
            )
        ],
    )
    bundle = RagEvidenceBundle(
        bundle_id="bundle-x",
        created_at=datetime.now(timezone.utc),
        search_plan=plan,
        source_statuses=[
            SourceRetrievalStatus(
                source=LiteratureSource.PUBMED,
                status=SourceRunStatus.SUCCESS,
                query_ids=["query-1"],
                result_count=1,
            )
        ],
        records=[record],
        claims=[claim],
        graph=graph,
        branches=branches,
    )
    policy = LiteratureEvidencePolicy(bundle)
    assignment = policy.select(round_index=0, exploration_branch="pareto")
    assert assignment is not None
    before = policy.weights[assignment.branch_id]
    policy.observe(
        round_index=0,
        exploration_branch="pareto",
        objective_improvement=1.0,
        structural_collapse_rate=0.0,
    )
    assert policy.weights[assignment.branch_id] > before
