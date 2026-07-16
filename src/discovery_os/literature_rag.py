"""Live scientific-literature RAG and evidence-guided search branching.

This module deliberately separates three concerns:

* literature evidence proposes *where to search*;
* candidate generators propose structures;
* specialist evaluators decide whether generated candidates satisfy scientific
  objectives.

A paper's recency or citation count is never converted into a material or drug
property score. Every extracted claim keeps an exact supporting text span and
source identifiers, and every generated search branch cites the claims that
created it.
"""

from __future__ import annotations

import concurrent.futures
import html
import json
import math
import os
import re
import threading
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol
from urllib.parse import urljoin, urlparse

import requests
from pydantic import AwareDatetime, Field, JsonValue, model_validator

from ._compat import StrEnum
from .hashing import stable_hash
from .mcp_client import StreamableHttpMcpClient
from .schemas import (
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    Identifier,
    NonEmptyText,
    Probability,
    StrictSchema,
)


class LiteratureRagError(RuntimeError):
    """Raised when live evidence cannot satisfy the configured fail policy."""


class LiteratureSource(StrEnum):
    PUBMED = "pubmed"
    EUROPE_PMC = "europe_pmc"
    OPENALEX = "openalex"
    CROSSREF = "crossref"
    ARXIV = "arxiv"
    MCP = "mcp"


class SourceRunStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    FAILED = "failed"


class EvidencePolarity(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NULL = "null"
    UNCERTAIN = "uncertain"


class EvidenceStage(StrEnum):
    HYPOTHESIS = "hypothesis"
    COMPUTATIONAL = "computational"
    IN_VITRO = "in_vitro"
    ANIMAL = "animal"
    CLINICAL_OBSERVATIONAL = "clinical_observational"
    CLINICAL_TRIAL = "clinical_trial"
    META_ANALYSIS = "meta_analysis"
    MATERIAL_SYNTHESIS = "material_synthesis"
    MATERIAL_CHARACTERIZATION = "material_characterization"
    UNKNOWN = "unknown"


class EvidenceBranchKind(StrEnum):
    EXACT_RECHECK = "exact_recheck"
    DERIVATIVE_OR_ANALOG = "derivative_or_analog"
    MECHANISM_ALTERNATIVE = "mechanism_alternative"
    MATERIAL_COMPOSITION = "material_composition"
    MATERIAL_CONDITION = "material_condition"
    NEGATIVE_EVIDENCE_AVOIDANCE = "negative_evidence_avoidance"
    CONFLICT_RESOLUTION = "conflict_resolution"
    UNDEREXPLORED_RELATION = "underexplored_relation"


class LiteratureQuery(StrictSchema):
    query_id: Identifier
    source: LiteratureSource
    query: NonEmptyText
    rationale: NonEmptyText
    max_results: int = Field(default=25, gt=0, le=200)
    from_date: date | None = None
    to_date: date | None = None

    @model_validator(mode="after")
    def _dates_are_ordered(self) -> "LiteratureQuery":
        if self.from_date and self.to_date and self.from_date > self.to_date:
            raise ValueError("from_date must not be after to_date")
        return self


class RagSearchPlan(StrictSchema):
    plan_id: Identifier
    user_prompt: NonEmptyText
    goal_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    domain: DiscoveryDomain | None = None
    generated_at: AwareDatetime
    planner_id: Identifier
    planner_version: Identifier
    concepts: list[str] = Field(default_factory=list)
    target_entities: list[str] = Field(default_factory=list)
    mechanism_terms: list[str] = Field(default_factory=list)
    negative_terms: list[str] = Field(default_factory=list)
    queries: list[LiteratureQuery] = Field(min_length=1)

    @model_validator(mode="after")
    def _query_ids_are_unique(self) -> "RagSearchPlan":
        ids = [item.query_id for item in self.queries]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate literature query ids")
        return self


class SourceRetrievalStatus(StrictSchema):
    source: LiteratureSource
    status: SourceRunStatus
    query_ids: list[Identifier] = Field(default_factory=list)
    result_count: int = Field(default=0, ge=0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    error: str | None = Field(default=None, max_length=4_000)
    endpoint: str | None = Field(default=None, max_length=2_000)


class LiteratureRecord(StrictSchema):
    record_id: Identifier
    title: NonEmptyText
    abstract: str = Field(default="", max_length=100_000)
    publication_date: date | None = None
    publication_year: int | None = Field(default=None, ge=1600, le=3000)
    authors: list[str] = Field(default_factory=list)
    venue: str | None = Field(default=None, max_length=2_000)
    doi: str | None = Field(default=None, max_length=512)
    pmid: str | None = Field(default=None, max_length=128)
    pmcid: str | None = Field(default=None, max_length=128)
    arxiv_id: str | None = Field(default=None, max_length=128)
    source_ids: dict[str, str] = Field(default_factory=dict)
    source_queries: list[Identifier] = Field(default_factory=list)
    urls: list[str] = Field(default_factory=list)
    is_retracted: bool | None = None
    citation_count: int | None = Field(default=None, ge=0)
    open_access: bool | None = None
    retrieved_at: AwareDatetime
    raw_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _identifiers_are_consistent(self) -> "LiteratureRecord":
        if not self.source_ids:
            raise ValueError("literature record requires at least one source identifier")
        if len(self.source_queries) != len(set(self.source_queries)):
            raise ValueError("duplicate source query ids")
        return self

    @property
    def evidence_text(self) -> str:
        return "\n".join(item for item in (self.title, self.abstract) if item).strip()


class EvidenceClaim(StrictSchema):
    claim_id: Identifier
    source_record_id: Identifier
    subject: NonEmptyText
    predicate: NonEmptyText
    object: NonEmptyText
    polarity: EvidencePolarity = EvidencePolarity.UNCERTAIN
    stage: EvidenceStage = EvidenceStage.UNKNOWN
    support_text: NonEmptyText
    confidence: Probability
    qualifiers: dict[str, JsonValue] = Field(default_factory=dict)
    entity_aliases: dict[str, list[str]] = Field(default_factory=dict)


class EvidenceGraphNode(StrictSchema):
    node_id: Identifier
    canonical_name: NonEmptyText
    node_type: Literal[
        "compound",
        "scaffold",
        "target",
        "disease",
        "mechanism",
        "material",
        "element",
        "dopant",
        "property",
        "condition",
        "toxicity",
        "other",
    ]
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)


class EvidenceGraphEdge(StrictSchema):
    edge_id: Identifier
    subject_node_id: Identifier
    predicate: NonEmptyText
    object_node_id: Identifier
    claim_ids: list[Identifier] = Field(min_length=1)
    positive_count: int = Field(default=0, ge=0)
    negative_count: int = Field(default=0, ge=0)
    uncertain_count: int = Field(default=0, ge=0)
    conflict: bool = False


class EvidenceGraph(StrictSchema):
    graph_id: Identifier
    nodes: list[EvidenceGraphNode]
    edges: list[EvidenceGraphEdge]

    @model_validator(mode="after")
    def _graph_is_closed(self) -> "EvidenceGraph":
        nodes = {item.node_id for item in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError("duplicate evidence graph nodes")
        edges = [item.edge_id for item in self.edges]
        if len(edges) != len(set(edges)):
            raise ValueError("duplicate evidence graph edges")
        for edge in self.edges:
            if edge.subject_node_id not in nodes or edge.object_node_id not in nodes:
                raise ValueError("evidence graph edge cites an unknown node")
        return self


class EvidenceBranch(StrictSchema):
    branch_id: Identifier
    kind: EvidenceBranchKind
    title: NonEmptyText
    rationale: NonEmptyText
    source_claim_ids: list[Identifier] = Field(min_length=1)
    generator_hints: dict[str, JsonValue] = Field(default_factory=dict)
    exclusion_hints: list[str] = Field(default_factory=list)
    priority: Probability
    evidence_only: Literal[True] = True

    @model_validator(mode="after")
    def _claim_ids_are_unique(self) -> "EvidenceBranch":
        if len(self.source_claim_ids) != len(set(self.source_claim_ids)):
            raise ValueError("duplicate source claim ids")
        return self


class RagEvidenceBundle(StrictSchema):
    bundle_id: Identifier
    created_at: AwareDatetime
    search_plan: RagSearchPlan
    source_statuses: list[SourceRetrievalStatus]
    records: list[LiteratureRecord]
    claims: list[EvidenceClaim]
    graph: EvidenceGraph
    branches: list[EvidenceBranch]
    warnings: list[str] = Field(default_factory=list)
    scientific_role: Literal["search_prior_only"] = "search_prior_only"

    @model_validator(mode="after")
    def _bundle_is_closed(self) -> "RagEvidenceBundle":
        record_ids = {item.record_id for item in self.records}
        claim_ids = {item.claim_id for item in self.claims}
        if len(record_ids) != len(self.records):
            raise ValueError("duplicate literature records")
        if len(claim_ids) != len(self.claims):
            raise ValueError("duplicate evidence claims")
        if any(item.source_record_id not in record_ids for item in self.claims):
            raise ValueError("claim cites an unknown literature record")
        if any(
            claim_id not in claim_ids
            for branch in self.branches
            for claim_id in branch.source_claim_ids
        ):
            raise ValueError("evidence branch cites an unknown claim")
        return self


class RagModel(Protocol):
    model_id: str
    model_version: str

    def complete_json(self, *, operation: str, system: str, user: str) -> Any: ...


class OpenAICompatibleRagModel:
    """Configurable planner/extractor for OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        timeout: float = 180.0,
        session: requests.Session | None = None,
    ) -> None:
        if not base_url.strip() or not model.strip():
            raise ValueError("RAG model base_url and model are required")
        root = base_url.rstrip("/") + "/"
        self.endpoint = (
            base_url
            if base_url.rstrip("/").endswith("/chat/completions")
            else urljoin(root, "chat/completions")
        )
        self.model_id = model.strip()
        self.model_version = "openai-compatible-json-v1"
        self.api_key = api_key.strip() if api_key and api_key.strip() else None
        self.timeout = timeout
        self.session = session or requests.Session()

    def complete_json(self, *, operation: str, system: str, user: str) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model_id,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        response = self.session.post(
            self.endpoint,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = response.json()
        try:
            content = raw["choices"][0]["message"]["content"]
        except Exception as exc:
            raise LiteratureRagError(
                f"RAG model {operation} response has no message content"
            ) from exc
        if isinstance(content, list):
            content = "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item)
                for item in content
            )
        try:
            return json.loads(str(content))
        except json.JSONDecodeError as exc:
            raise LiteratureRagError(
                f"RAG model {operation} did not return valid JSON"
            ) from exc


class PromptSearchPlanner:
    def __init__(self, model: RagModel | None = None) -> None:
        self.model = model

    def plan(
        self,
        prompt: str,
        *,
        goal: DiscoveryGoal | None = None,
        sources: Sequence[LiteratureSource] | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        max_results_per_query: int = 25,
    ) -> RagSearchPlan:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("literature RAG prompt cannot be empty")
        selected_sources = list(sources or LiteratureSource)
        if self.model is None:
            return self._deterministic_plan(
                prompt,
                goal=goal,
                sources=selected_sources,
                from_date=from_date,
                to_date=to_date,
                max_results_per_query=max_results_per_query,
            )
        payload = self.model.complete_json(
            operation="search-plan",
            system=(
                "You are a scientific search planner. Return JSON only. Convert the user goal "
                "into multiple high-recall and high-precision scholarly search queries. Do not "
                "invent findings. Include synonyms, target/mechanism terms, negative evidence, "
                "toxicity or failed-trial terms for biomedicine, and composition/doping/pressure/"
                "synthesis/stability terms for materials."
            ),
            user=json.dumps(
                {
                    "prompt": prompt,
                    "goal": goal.model_dump(mode="json") if goal else None,
                    "allowed_sources": [item.value for item in selected_sources],
                    "from_date": from_date.isoformat() if from_date else None,
                    "to_date": to_date.isoformat() if to_date else None,
                    "max_results_per_query": max_results_per_query,
                    "required_output": {
                        "concepts": ["string"],
                        "target_entities": ["string"],
                        "mechanism_terms": ["string"],
                        "negative_terms": ["string"],
                        "queries": [
                            {
                                "source": "allowed source",
                                "query": "string",
                                "rationale": "string",
                            }
                        ],
                    },
                },
                ensure_ascii=False,
            ),
        )
        if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
            raise LiteratureRagError("RAG search planner returned an invalid plan object")
        query_rows: list[LiteratureQuery] = []
        allowed = {item.value: item for item in selected_sources}
        for index, row in enumerate(payload["queries"]):
            if not isinstance(row, dict):
                continue
            source = allowed.get(str(row.get("source", "")))
            query = str(row.get("query", "")).strip()
            rationale = str(row.get("rationale", "")).strip()
            if source is None or not query or not rationale:
                continue
            query_rows.append(
                LiteratureQuery(
                    query_id=f"LQ-{stable_hash([source.value, query, index])[:24]}",
                    source=source,
                    query=query,
                    rationale=rationale,
                    max_results=max_results_per_query,
                    from_date=from_date,
                    to_date=to_date,
                )
            )
        if not query_rows:
            raise LiteratureRagError("RAG search planner produced no usable queries")
        plan_payload = {
            "user_prompt": prompt,
            "goal_hash": stable_hash(goal) if goal else None,
            "domain": goal.domain if goal else None,
            "planner_id": self.model.model_id,
            "planner_version": self.model.model_version,
            "concepts": _clean_string_list(payload.get("concepts")),
            "target_entities": _clean_string_list(payload.get("target_entities")),
            "mechanism_terms": _clean_string_list(payload.get("mechanism_terms")),
            "negative_terms": _clean_string_list(payload.get("negative_terms")),
            "queries": query_rows,
        }
        return RagSearchPlan(
            plan_id=f"RPLAN-{stable_hash(plan_payload)[:24]}",
            generated_at=datetime.now(timezone.utc),
            **plan_payload,
        )

    def _deterministic_plan(
        self,
        prompt: str,
        *,
        goal: DiscoveryGoal | None,
        sources: Sequence[LiteratureSource],
        from_date: date | None,
        to_date: date | None,
        max_results_per_query: int,
    ) -> RagSearchPlan:
        domain = DiscoveryDomain(str(goal.domain)) if goal else _infer_domain(prompt)
        base_terms = [prompt]
        if goal:
            base_terms.extend([goal.title, goal.scientific_question])
            base_terms.extend(item.property_name for item in goal.objectives)
        core = " ".join(dict.fromkeys(_compact(item) for item in base_terms if item))
        if domain == DiscoveryDomain.MEDICINAL_CHEMISTRY:
            expansions = [
                f"({core}) compound target mechanism efficacy",
                f"({core}) scaffold analog derivative structure activity relationship",
                f"({core}) toxicity adverse effect resistance failed clinical trial",
                f"({core}) in vitro animal clinical trial biomarker",
            ]
            mechanism_terms = ["target", "mechanism", "pathway", "inhibition", "activation"]
            negative_terms = ["toxicity", "adverse", "resistance", "failed trial", "no effect"]
        else:
            expansions = [
                f"({core}) composition crystal structure property",
                f"({core}) doping substitution pressure temperature synthesis",
                f"({core}) stability formation energy phase diagram",
                f"({core}) experimental characterization mechanism",
            ]
            mechanism_terms = ["composition", "doping", "pressure", "synthesis", "phase"]
            negative_terms = ["unstable", "decomposition", "failed synthesis", "retracted"]
        rows: list[LiteratureQuery] = []
        for source in sources:
            for index, query in enumerate(expansions):
                rows.append(
                    LiteratureQuery(
                        query_id=f"LQ-{stable_hash([source.value, query, index])[:24]}",
                        source=source,
                        query=query,
                        rationale=f"deterministic {domain.value} evidence query {index + 1}",
                        max_results=max_results_per_query,
                        from_date=from_date,
                        to_date=to_date,
                    )
                )
        payload = {
            "user_prompt": prompt,
            "goal_hash": stable_hash(goal) if goal else None,
            "domain": domain,
            "planner_id": "deterministic-query-planner",
            "planner_version": "1.0.0",
            "concepts": _clean_string_list(base_terms),
            "target_entities": [],
            "mechanism_terms": mechanism_terms,
            "negative_terms": negative_terms,
            "queries": rows,
        }
        return RagSearchPlan(
            plan_id=f"RPLAN-{stable_hash(payload)[:24]}",
            generated_at=datetime.now(timezone.utc),
            **payload,
        )


class MultiSourceLiteratureRetriever:
    """Concurrent retrieval from independent scholarly metadata providers."""

    EUROPE_PMC_ENDPOINT = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    OPENALEX_ENDPOINT = "https://api.openalex.org/works"
    CROSSREF_ENDPOINT = "https://api.crossref.org/works"
    ARXIV_ENDPOINT = "https://export.arxiv.org/api/query"

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        timeout: tuple[float, float] = (10.0, 60.0),
        user_agent: str = "discovery-os-literature-rag/1.0",
        email: str | None = None,
        ncbi_api_key: str | None = None,
        openalex_api_key: str | None = None,
        mcp_client: StreamableHttpMcpClient | None = None,
        mcp_tool: str | None = None,
        max_workers: int = 5,
        arxiv_min_interval_seconds: float = 3.0,
    ) -> None:
        if not 0.0 <= arxiv_min_interval_seconds <= 60.0:
            raise ValueError("arxiv_min_interval_seconds must be between 0 and 60")
        self.session = session or requests.Session()
        self.timeout = timeout
        self.user_agent = user_agent
        self.email = email
        self.ncbi_api_key = ncbi_api_key
        self.openalex_api_key = openalex_api_key
        self.mcp_client = mcp_client
        self.mcp_tool = mcp_tool
        self.max_workers = max(1, min(max_workers, 10))
        self.arxiv_min_interval_seconds = float(arxiv_min_interval_seconds)
        self._arxiv_rate_lock = threading.Lock()
        self._arxiv_last_request: float | None = None

    def retrieve(
        self, plan: RagSearchPlan
    ) -> tuple[list[LiteratureRecord], list[SourceRetrievalStatus]]:
        grouped: dict[LiteratureSource, list[LiteratureQuery]] = {}
        for query in plan.queries:
            grouped.setdefault(LiteratureSource(str(query.source)), []).append(query)
        all_records: list[LiteratureRecord] = []
        statuses: list[SourceRetrievalStatus] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._retrieve_source, source, rows): source
                for source, rows in grouped.items()
            }
            for future in concurrent.futures.as_completed(futures):
                source = futures[future]
                try:
                    records, status = future.result()
                except Exception as exc:  # Defensive boundary around each provider.
                    records = []
                    status = SourceRetrievalStatus(
                        source=source,
                        status=SourceRunStatus.FAILED,
                        query_ids=[item.query_id for item in grouped[source]],
                        error=f"{type(exc).__name__}: {exc}",
                    )
                all_records.extend(records)
                statuses.append(status)
        return deduplicate_records(all_records), sorted(
            statuses, key=lambda item: str(item.source)
        )

    def _retrieve_source(
        self, source: LiteratureSource, queries: Sequence[LiteratureQuery]
    ) -> tuple[list[LiteratureRecord], SourceRetrievalStatus]:
        started = time.monotonic()
        if source == LiteratureSource.OPENALEX and not self.openalex_api_key:
            return [], SourceRetrievalStatus(
                source=source,
                status=SourceRunStatus.SKIPPED,
                query_ids=[item.query_id for item in queries],
                elapsed_seconds=time.monotonic() - started,
                error="OPENALEX_API_KEY is not configured",
                endpoint=self.OPENALEX_ENDPOINT,
            )
        if source == LiteratureSource.MCP and (self.mcp_client is None or not self.mcp_tool):
            return [], SourceRetrievalStatus(
                source=source,
                status=SourceRunStatus.SKIPPED,
                query_ids=[item.query_id for item in queries],
                elapsed_seconds=time.monotonic() - started,
                error="MATERIAL_RAG_MCP_URL and MATERIAL_RAG_MCP_TOOL are not configured",
                endpoint=None,
            )
        handlers = {
            LiteratureSource.EUROPE_PMC: self._search_europe_pmc,
            LiteratureSource.PUBMED: self._search_pubmed,
            LiteratureSource.OPENALEX: self._search_openalex,
            LiteratureSource.CROSSREF: self._search_crossref,
            LiteratureSource.ARXIV: self._search_arxiv,
            LiteratureSource.MCP: self._search_mcp,
        }
        records: list[LiteratureRecord] = []
        errors: list[str] = []
        endpoint = {
            LiteratureSource.EUROPE_PMC: self.EUROPE_PMC_ENDPOINT,
            LiteratureSource.PUBMED: self.NCBI_ESEARCH,
            LiteratureSource.OPENALEX: self.OPENALEX_ENDPOINT,
            LiteratureSource.CROSSREF: self.CROSSREF_ENDPOINT,
            LiteratureSource.ARXIV: self.ARXIV_ENDPOINT,
            LiteratureSource.MCP: self.mcp_client.endpoint if self.mcp_client else "mcp:unconfigured",
        }[source]
        for query in queries:
            try:
                records.extend(handlers[source](query))
            except Exception as exc:
                errors.append(f"{query.query_id}: {type(exc).__name__}: {exc}")
        if records and errors:
            status = SourceRunStatus.PARTIAL
        elif records:
            status = SourceRunStatus.SUCCESS
        else:
            status = SourceRunStatus.FAILED if errors else SourceRunStatus.SUCCESS
        return records, SourceRetrievalStatus(
            source=source,
            status=status,
            query_ids=[item.query_id for item in queries],
            result_count=len(records),
            elapsed_seconds=time.monotonic() - started,
            error="; ".join(errors)[:4000] or None,
            endpoint=endpoint,
        )

    def _search_mcp(self, query: LiteratureQuery) -> list[LiteratureRecord]:
        if self.mcp_client is None or not self.mcp_tool:
            return []
        payload = self.mcp_client.call_tool(
            self.mcp_tool,
            {
                "query": query.query,
                "max_results": query.max_results,
                "from_date": query.from_date.isoformat() if query.from_date else None,
                "to_date": query.to_date.isoformat() if query.to_date else None,
            },
        )
        rows = payload.get("records")
        if not isinstance(rows, list):
            raise LiteratureRagError("MCP evidence tool result requires a records array")
        records: list[LiteratureRecord] = []
        invalid_rows = 0
        for item in rows[: query.max_results]:
            if not isinstance(item, dict):
                invalid_rows += 1
                continue
            raw_title = item.get("title")
            raw_source_id = item.get("source_id")
            if not isinstance(raw_title, str) or not isinstance(raw_source_id, str):
                invalid_rows += 1
                continue
            title = _clean_markup(raw_title)
            source_id = raw_source_id.strip()
            if not title or not source_id or len(source_id) > 2_000:
                invalid_rows += 1
                continue
            raw_abstract = item.get("abstract")
            if raw_abstract is None:
                raw_abstract = item.get("support_text", "")
            abstract = _clean_markup(raw_abstract) if isinstance(raw_abstract, str) else ""
            pub_date = _parse_date(item.get("publication_date"))
            url = _optional_text(item.get("url")) if isinstance(item.get("url"), str) else None
            if url and urlparse(url).scheme.lower() not in {"http", "https"}:
                url = None
            records.append(
                LiteratureRecord(
                    record_id=f"LIT-{stable_hash(['mcp', source_id, title])[:24]}",
                    title=title,
                    abstract=abstract,
                    publication_date=pub_date,
                    publication_year=_year(pub_date, item.get("publication_year")),
                    authors=_clean_string_list(item.get("authors", [])),
                    venue=_optional_text(item.get("venue")),
                    doi=_normalize_doi(item.get("doi")),
                    pmid=_optional_text(item.get("pmid")),
                    pmcid=_optional_text(item.get("pmcid")),
                    source_ids={LiteratureSource.MCP.value: source_id},
                    source_queries=[query.query_id],
                    urls=[url] if url else [],
                    is_retracted=_bool_or_none(item.get("is_retracted")),
                    citation_count=_int_or_none(item.get("citation_count")),
                    open_access=_bool_or_none(item.get("open_access")),
                    retrieved_at=datetime.now(timezone.utc),
                    raw_metadata={"mcp_tool": self.mcp_tool},
                )
            )
        if invalid_rows and not records:
            raise LiteratureRagError(
                f"MCP evidence tool returned {invalid_rows} invalid record(s)"
            )
        return records

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self.user_agent, "Accept": "application/json, application/xml"}

    def _get(self, url: str, *, params: Mapping[str, Any]) -> requests.Response:
        last: Exception | None = None
        for attempt in range(3):
            try:
                response = self.session.get(
                    url,
                    params=dict(params),
                    headers=self._headers(),
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return response
            except Exception as exc:
                last = exc
                if attempt < 2:
                    time.sleep(1.5 * (attempt + 1))
        assert last is not None
        raise last

    def _search_europe_pmc(self, query: LiteratureQuery) -> list[LiteratureRecord]:
        q = query.query
        if query.from_date:
            q += f" AND FIRST_PDATE:[{query.from_date.isoformat()} TO 3000-12-31]"
        if query.to_date:
            q += f" AND FIRST_PDATE:[1600-01-01 TO {query.to_date.isoformat()}]"
        response = self._get(
            self.EUROPE_PMC_ENDPOINT,
            params={
                "query": q,
                "format": "json",
                "resultType": "core",
                "pageSize": query.max_results,
                "sort": "FIRST_PDATE_D desc",
            },
        )
        results = response.json().get("resultList", {}).get("result", [])
        return [
            _record_from_europe_pmc(item, query.query_id)
            for item in results
            if isinstance(item, dict) and str(item.get("title", "")).strip()
        ]

    def _search_pubmed(self, query: LiteratureQuery) -> list[LiteratureRecord]:
        term = query.query
        if query.from_date or query.to_date:
            start = query.from_date or date(1600, 1, 1)
            end = query.to_date or date.today()
            term += f' AND ("{start.isoformat()}"[Date - Publication] : "{end.isoformat()}"[Date - Publication])'
        params: dict[str, Any] = {
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": query.max_results,
            "sort": "pub date",
        }
        if self.email:
            params["email"] = self.email
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        ids = self._get(self.NCBI_ESEARCH, params=params).json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fetch_params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(str(item) for item in ids),
            "retmode": "xml",
        }
        if self.email:
            fetch_params["email"] = self.email
        if self.ncbi_api_key:
            fetch_params["api_key"] = self.ncbi_api_key
        xml_text = self._get(self.NCBI_EFETCH, params=fetch_params).text
        return _records_from_pubmed_xml(xml_text, query.query_id)

    def _search_openalex(self, query: LiteratureQuery) -> list[LiteratureRecord]:
        filters = ["is_retracted:false"]
        if query.from_date:
            filters.append(f"from_publication_date:{query.from_date.isoformat()}")
        if query.to_date:
            filters.append(f"to_publication_date:{query.to_date.isoformat()}")
        response = self._get(
            self.OPENALEX_ENDPOINT,
            params={
                "api_key": self.openalex_api_key,
                "search": query.query,
                "filter": ",".join(filters),
                "sort": "publication_date:desc",
                "per-page": query.max_results,
            },
        )
        return [
            _record_from_openalex(item, query.query_id)
            for item in response.json().get("results", [])
            if isinstance(item, dict) and str(item.get("title", "")).strip()
        ]

    def _search_crossref(self, query: LiteratureQuery) -> list[LiteratureRecord]:
        filters: list[str] = []
        if query.from_date:
            filters.append(f"from-pub-date:{query.from_date.isoformat()}")
        if query.to_date:
            filters.append(f"until-pub-date:{query.to_date.isoformat()}")
        params: dict[str, Any] = {
            "query.bibliographic": query.query,
            "rows": query.max_results,
            "sort": "published",
            "order": "desc",
        }
        if filters:
            params["filter"] = ",".join(filters)
        if self.email:
            params["mailto"] = self.email
        response = self._get(self.CROSSREF_ENDPOINT, params=params)
        return [
            _record_from_crossref(item, query.query_id)
            for item in response.json().get("message", {}).get("items", [])
            if isinstance(item, dict) and item.get("title")
        ]

    def _search_arxiv(self, query: LiteratureQuery) -> list[LiteratureRecord]:
        self._wait_for_arxiv_slot()
        response = self._get(
            self.ARXIV_ENDPOINT,
            params={
                "search_query": f"all:{query.query}",
                "start": 0,
                "max_results": query.max_results,
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            },
        )
        rows = _records_from_arxiv_xml(response.text, query.query_id)
        return [
            item
            for item in rows
            if (query.from_date is None or item.publication_date is None or item.publication_date >= query.from_date)
            and (query.to_date is None or item.publication_date is None or item.publication_date <= query.to_date)
        ]

    def _wait_for_arxiv_slot(self) -> None:
        """Apply arXiv's documented inter-request courtesy interval."""

        with self._arxiv_rate_lock:
            now = time.monotonic()
            if self._arxiv_last_request is not None:
                delay = self.arxiv_min_interval_seconds - (
                    now - self._arxiv_last_request
                )
                if delay > 0.0:
                    time.sleep(delay)
            self._arxiv_last_request = time.monotonic()


class EvidenceClaimExtractor:
    def __init__(self, model: RagModel | None = None, *, max_records: int = 120) -> None:
        self.model = model
        self.max_records = max_records

    def extract(
        self,
        records: Sequence[LiteratureRecord],
        *,
        prompt: str,
        goal: DiscoveryGoal | None = None,
    ) -> tuple[list[EvidenceClaim], list[str]]:
        selected = list(records)[: self.max_records]
        if self.model is None:
            return self._deterministic_extract(selected, prompt=prompt, goal=goal)
        claims: list[EvidenceClaim] = []
        warnings: list[str] = []
        for record in selected:
            if not record.evidence_text:
                continue
            try:
                raw = self.model.complete_json(
                    operation="claim-extraction",
                    system=(
                        "Extract only claims explicitly supported by the supplied title/abstract. "
                        "Return JSON only. Every support_text must be an exact contiguous substring "
                        "of the supplied text. Separate positive, negative/null, and uncertain claims. "
                        "For drugs capture compound, target, disease, mechanism, efficacy, toxicity, "
                        "trial stage, and failed combinations. For materials capture composition, "
                        "dopant, pressure, temperature, structure, property, synthesis, stability, "
                        "and failed synthesis. Do not infer a relation absent from the text."
                    ),
                    user=json.dumps(
                        {
                            "user_goal": prompt,
                            "structured_goal": goal.model_dump(mode="json") if goal else None,
                            "record_id": record.record_id,
                            "title_and_abstract": record.evidence_text,
                            "required_output": {
                                "claims": [
                                    {
                                        "subject": "string",
                                        "predicate": "string",
                                        "object": "string",
                                        "polarity": "supports|contradicts|null|uncertain",
                                        "stage": "EvidenceStage value",
                                        "support_text": "exact substring",
                                        "confidence": "0..1",
                                        "qualifiers": {},
                                        "entity_aliases": {},
                                    }
                                ]
                            },
                        },
                        ensure_ascii=False,
                    ),
                )
            except Exception as exc:
                warnings.append(f"{record.record_id}: model extraction failed: {type(exc).__name__}: {exc}")
                continue
            rows = raw.get("claims", []) if isinstance(raw, dict) else []
            for row in rows:
                claim = _validated_claim_from_model(record, row)
                if claim is None:
                    warnings.append(
                        f"{record.record_id}: rejected unsupported or malformed model claim"
                    )
                    continue
                claims.append(claim)
        return _deduplicate_claims(claims), warnings

    def _deterministic_extract(
        self,
        records: Sequence[LiteratureRecord],
        *,
        prompt: str,
        goal: DiscoveryGoal | None,
    ) -> tuple[list[EvidenceClaim], list[str]]:
        claims: list[EvidenceClaim] = []
        warnings = [
            "No RAG extraction model was configured; deterministic extraction keeps only "
            "conservative title-level search-prior claims."
        ]
        domain = DiscoveryDomain(str(goal.domain)) if goal else _infer_domain(prompt)
        for record in records:
            sentence = record.title.strip()
            if not sentence:
                continue
            polarity = (
                EvidencePolarity.CONTRADICTS
                if re.search(r"\b(fail(?:ed|ure)?|no effect|not associated|toxic|unstable|decompos)\b", sentence, re.I)
                else EvidencePolarity.UNCERTAIN
            )
            subject = _first_entity_phrase(sentence) or sentence[:200]
            predicate = "reported_in_recent_literature"
            object_value = _goal_object(goal, prompt, domain)
            payload = [record.record_id, subject, predicate, object_value, sentence]
            claims.append(
                EvidenceClaim(
                    claim_id=f"ECL-{stable_hash(payload)[:24]}",
                    source_record_id=record.record_id,
                    subject=subject,
                    predicate=predicate,
                    object=object_value,
                    polarity=polarity,
                    stage=_infer_stage(sentence),
                    support_text=sentence,
                    confidence=0.35,
                    qualifiers={"deterministic_fallback": True},
                )
            )
        return _deduplicate_claims(claims), warnings


class EvidenceGraphBuilder:
    def build(self, claims: Sequence[EvidenceClaim]) -> EvidenceGraph:
        node_map: dict[str, EvidenceGraphNode] = {}
        grouped: dict[tuple[str, str, str], list[EvidenceClaim]] = {}
        for claim in claims:
            subject = _node_for_entity(claim.subject, claim.predicate, position="subject")
            object_node = _node_for_entity(claim.object, claim.predicate, position="object")
            node_map.setdefault(subject.node_id, subject)
            node_map.setdefault(object_node.node_id, object_node)
            grouped.setdefault((subject.node_id, _normal_text(claim.predicate), object_node.node_id), []).append(claim)
        edges: list[EvidenceGraphEdge] = []
        for (subject_id, predicate, object_id), rows in sorted(grouped.items()):
            positive = sum(item.polarity == EvidencePolarity.SUPPORTS for item in rows)
            negative = sum(item.polarity in {EvidencePolarity.CONTRADICTS, EvidencePolarity.NULL} for item in rows)
            uncertain = len(rows) - positive - negative
            payload = [subject_id, predicate, object_id, sorted(item.claim_id for item in rows)]
            edges.append(
                EvidenceGraphEdge(
                    edge_id=f"EGE-{stable_hash(payload)[:24]}",
                    subject_node_id=subject_id,
                    predicate=predicate,
                    object_node_id=object_id,
                    claim_ids=sorted(item.claim_id for item in rows),
                    positive_count=positive,
                    negative_count=negative,
                    uncertain_count=uncertain,
                    conflict=positive > 0 and negative > 0,
                )
            )
        graph_payload = {
            "nodes": [item.model_dump(mode="json") for item in sorted(node_map.values(), key=lambda row: row.node_id)],
            "edges": [item.model_dump(mode="json") for item in edges],
        }
        return EvidenceGraph(
            graph_id=f"EGRAPH-{stable_hash(graph_payload)[:24]}",
            nodes=sorted(node_map.values(), key=lambda row: row.node_id),
            edges=edges,
        )


class EvidenceBranchPlanner:
    """Translate claims into generator search priors, never evaluator scores."""

    def plan(
        self,
        claims: Sequence[EvidenceClaim],
        graph: EvidenceGraph,
        *,
        prompt: str,
        goal: DiscoveryGoal | None = None,
        max_branches: int = 24,
    ) -> list[EvidenceBranch]:
        domain = DiscoveryDomain(str(goal.domain)) if goal else _infer_domain(prompt)
        branches: list[EvidenceBranch] = []
        by_pair: dict[tuple[str, str], list[EvidenceClaim]] = {}
        for claim in claims:
            by_pair.setdefault((_normal_text(claim.subject), _normal_text(claim.object)), []).append(claim)
        for (_subject_key, _object_key), rows in sorted(by_pair.items()):
            positive = [item for item in rows if item.polarity == EvidencePolarity.SUPPORTS]
            negative = [
                item
                for item in rows
                if item.polarity in {EvidencePolarity.CONTRADICTS, EvidencePolarity.NULL}
            ]
            selected = positive or rows
            representative = sorted(
                selected,
                key=lambda item: (-item.confidence, item.claim_id),
            )[0]
            claim_ids = sorted(item.claim_id for item in rows)
            if positive and negative:
                branches.append(
                    _make_branch(
                        EvidenceBranchKind.CONFLICT_RESOLUTION,
                        f"Resolve conflicting evidence for {representative.subject}",
                        "Preserve the disputed relation as a dedicated validation branch; do not "
                        "treat the literature relation as settled.",
                        claim_ids,
                        {"conflict_subject": representative.subject, "conflict_object": representative.object},
                        priority=0.95,
                    )
                )
            if negative and not positive:
                branches.append(
                    _make_branch(
                        EvidenceBranchKind.NEGATIVE_EVIDENCE_AVOIDANCE,
                        f"Avoid or challenge negative-evidence region: {representative.subject}",
                        "Negative, null, toxicity, instability, or failed-trial evidence becomes an "
                        "exclusion or counterfactual branch rather than a positive candidate score.",
                        claim_ids,
                        {"avoid_entities": sorted({item.subject for item in negative})},
                        exclusion_hints=[item.support_text for item in negative[:5]],
                        priority=0.9,
                    )
                )
                continue
            if domain == DiscoveryDomain.MEDICINAL_CHEMISTRY:
                hints = _medicinal_hints(rows)
                branches.extend(
                    [
                        _make_branch(
                            EvidenceBranchKind.EXACT_RECHECK,
                            f"Re-evaluate reported entity {representative.subject}",
                            "Retain the reported compound/entity as a positive-control or repurposing "
                            "branch and verify it with the current specialist panel.",
                            claim_ids,
                            hints | {"search_mode": "exact_recheck"},
                            priority=0.85,
                        ),
                        _make_branch(
                            EvidenceBranchKind.DERIVATIVE_OR_ANALOG,
                            f"Generate analogs around {representative.subject}",
                            "Preserve the evidence-linked core or seed while exploring derivatives; "
                            "the generated molecules must still pass binding, ADMET, toxicity, and "
                            "other specialist evaluation.",
                            claim_ids,
                            hints | {"search_mode": "analog_or_derivative"},
                            priority=0.8,
                        ),
                        _make_branch(
                            EvidenceBranchKind.MECHANISM_ALTERNATIVE,
                            f"Explore alternative structures for {representative.object}",
                            "Search structurally distinct candidates that address the same target or "
                            "mechanism, preventing the literature seed from collapsing diversity.",
                            claim_ids,
                            hints | {"search_mode": "mechanism_alternative"},
                            priority=0.75,
                        ),
                    ]
                )
            else:
                hints = _material_hints(rows)
                branches.append(
                    _make_branch(
                        EvidenceBranchKind.MATERIAL_COMPOSITION,
                        f"Explore reported composition relation for {representative.subject}",
                        "Translate evidence-linked elements, compositions, dopants, or phases into a "
                        "generation branch without treating the paper as a stability score.",
                        claim_ids,
                        hints | {"search_mode": "composition"},
                        priority=0.82,
                    )
                )
                if any(key in hints for key in ("pressure_gpa", "temperature_k", "space_group", "dopants")):
                    branches.append(
                        _make_branch(
                            EvidenceBranchKind.MATERIAL_CONDITION,
                            f"Explore reported conditions for {representative.subject}",
                            "Preserve pressure, temperature, dopant, phase, or synthesis context as a "
                            "separate search branch so unsupported ambient-condition claims are not made.",
                            claim_ids,
                            hints | {"search_mode": "condition"},
                            priority=0.8,
                        )
                    )
        # Graph paths of length two create explicitly labelled hypotheses, not claims.
        adjacency: dict[str, list[EvidenceGraphEdge]] = {}
        for edge in graph.edges:
            adjacency.setdefault(edge.subject_node_id, []).append(edge)
        nodes = {item.node_id: item for item in graph.nodes}
        for first in graph.edges:
            for second in adjacency.get(first.object_node_id, []):
                if first.subject_node_id == second.object_node_id:
                    continue
                claim_ids = sorted(set(first.claim_ids + second.claim_ids))
                subject = nodes[first.subject_node_id].canonical_name
                middle = nodes[first.object_node_id].canonical_name
                target = nodes[second.object_node_id].canonical_name
                branches.append(
                    _make_branch(
                        EvidenceBranchKind.UNDEREXPLORED_RELATION,
                        f"Test linked hypothesis: {subject} → {middle} → {target}",
                        "Two independently sourced graph relations create a hypothesis branch. The "
                        "composed relation is not treated as established evidence and requires direct "
                        "specialist or experimental validation.",
                        claim_ids,
                        {
                            "hypothesis_subject": subject,
                            "bridge_entity": middle,
                            "hypothesis_target": target,
                            "search_mode": "graph_path_hypothesis",
                        },
                        priority=0.65,
                    )
                )
        deduped: dict[str, EvidenceBranch] = {}
        for branch in branches:
            key = stable_hash([branch.kind, branch.generator_hints, branch.source_claim_ids])
            prior = deduped.get(key)
            if prior is None or branch.priority > prior.priority:
                deduped[key] = branch
        return sorted(
            deduped.values(), key=lambda item: (-item.priority, item.branch_id)
        )[:max_branches]


class JsonEvidenceIndex:
    """Append/update local evidence index with immutable record and claim files."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "records").mkdir(exist_ok=True)
        (self.root / "claims").mkdir(exist_ok=True)
        (self.root / "bundles").mkdir(exist_ok=True)

    def update(self, bundle: RagEvidenceBundle) -> Path:
        for record in bundle.records:
            _write_json_atomic(self.root / "records" / f"{record.record_id}.json", record)
        for claim in bundle.claims:
            _write_json_atomic(self.root / "claims" / f"{claim.claim_id}.json", claim)
        bundle_path = self.root / "bundles" / f"{bundle.bundle_id}.json"
        _write_json_atomic(bundle_path, bundle)
        manifest = {
            "schema_version": "1.0",
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "latest_bundle_id": bundle.bundle_id,
            "record_count": len(list((self.root / "records").glob("*.json"))),
            "claim_count": len(list((self.root / "claims").glob("*.json"))),
            "bundle_count": len(list((self.root / "bundles").glob("*.json"))),
        }
        _write_json_atomic(self.root / "manifest.json", manifest)
        return bundle_path


class LiteratureRagPipeline:
    def __init__(
        self,
        planner: PromptSearchPlanner,
        retriever: MultiSourceLiteratureRetriever,
        extractor: EvidenceClaimExtractor,
        *,
        graph_builder: EvidenceGraphBuilder | None = None,
        branch_planner: EvidenceBranchPlanner | None = None,
    ) -> None:
        self.planner = planner
        self.retriever = retriever
        self.extractor = extractor
        self.graph_builder = graph_builder or EvidenceGraphBuilder()
        self.branch_planner = branch_planner or EvidenceBranchPlanner()

    def run(
        self,
        prompt: str,
        *,
        goal: DiscoveryGoal | None = None,
        sources: Sequence[LiteratureSource] | None = None,
        from_date: date | None = None,
        to_date: date | None = None,
        max_results_per_query: int = 25,
        max_branches: int = 24,
        index: JsonEvidenceIndex | None = None,
    ) -> RagEvidenceBundle:
        plan = self.planner.plan(
            prompt,
            goal=goal,
            sources=sources,
            from_date=from_date,
            to_date=to_date,
            max_results_per_query=max_results_per_query,
        )
        records, statuses = self.retriever.retrieve(plan)
        claims, warnings = self.extractor.extract(records, prompt=prompt, goal=goal)
        graph = self.graph_builder.build(claims)
        branches = self.branch_planner.plan(
            claims,
            graph,
            prompt=prompt,
            goal=goal,
            max_branches=max_branches,
        )
        payload = {
            "search_plan": plan.model_dump(mode="json"),
            "records": [item.model_dump(mode="json") for item in records],
            "claims": [item.model_dump(mode="json") for item in claims],
            "graph": graph.model_dump(mode="json"),
            "branches": [item.model_dump(mode="json") for item in branches],
        }
        bundle = RagEvidenceBundle(
            bundle_id=f"RBUNDLE-{stable_hash(payload)[:24]}",
            created_at=datetime.now(timezone.utc),
            search_plan=plan,
            source_statuses=statuses,
            records=records,
            claims=claims,
            graph=graph,
            branches=branches,
            warnings=warnings,
        )
        if index is not None:
            index.update(bundle)
        return bundle


class EvidenceBranchAssignment(StrictSchema):
    branch_id: Identifier
    branch_kind: EvidenceBranchKind
    source_claim_ids: list[Identifier]
    generator_hints: dict[str, JsonValue]
    rationale: NonEmptyText


class LiteratureEvidencePolicy:
    """Stateful deterministic allocation of evidence branches to search workers."""

    def __init__(self, bundle: RagEvidenceBundle) -> None:
        self.bundle = RagEvidenceBundle.model_validate_json(bundle.model_dump_json(), strict=True)
        self._branches = list(self.bundle.branches)
        self._weights = {item.branch_id: float(item.priority) for item in self._branches}
        self._counts = {item.branch_id: 0 for item in self._branches}
        self._assignments: dict[tuple[int, str], str] = {}

    def select(
        self,
        *,
        round_index: int,
        exploration_branch: str,
    ) -> EvidenceBranchAssignment | None:
        if not self._branches:
            return None
        key = (round_index, exploration_branch)
        existing = self._assignments.get(key)
        if existing is None:
            total = 1 + sum(self._counts.values())
            ranked = sorted(
                self._branches,
                key=lambda item: (
                    -(
                        self._weights[item.branch_id]
                        + math.sqrt(2.0 * math.log(total + 1.0) / (self._counts[item.branch_id] + 1.0))
                    ),
                    item.branch_id,
                ),
            )
            selected = ranked[0]
            self._assignments[key] = selected.branch_id
            self._counts[selected.branch_id] += 1
        else:
            selected = next(item for item in self._branches if item.branch_id == existing)
        return EvidenceBranchAssignment(
            branch_id=selected.branch_id,
            branch_kind=EvidenceBranchKind(str(selected.kind)),
            source_claim_ids=selected.source_claim_ids,
            generator_hints=selected.generator_hints,
            rationale=selected.rationale,
        )

    def observe(
        self,
        *,
        round_index: int,
        exploration_branch: str,
        objective_improvement: float | None,
        structural_collapse_rate: float,
        failed: bool = False,
    ) -> None:
        branch_id = self._assignments.get((round_index, exploration_branch))
        if branch_id is None:
            return
        reward = 0.0
        if objective_improvement is not None:
            reward += max(-1.0, min(1.0, objective_improvement)) * 0.2
        reward -= max(0.0, min(1.0, structural_collapse_rate)) * 0.35
        if failed:
            reward -= 0.5
        self._weights[branch_id] = max(0.0, min(1.5, self._weights[branch_id] + reward))

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)


def build_literature_rag_from_environment(
    *,
    environ: Mapping[str, str] | None = None,
    require_model: bool = False,
) -> LiteratureRagPipeline:
    values = environ if environ is not None else os.environ
    model: RagModel | None = None
    base_url = str(values.get("RAG_MODEL_API_URL", "")).strip()
    model_name = str(values.get("RAG_MODEL_NAME", "")).strip()
    if base_url or model_name:
        if not base_url or not model_name:
            raise LiteratureRagError(
                "RAG_MODEL_API_URL and RAG_MODEL_NAME must be configured together"
            )
        model = OpenAICompatibleRagModel(
            base_url,
            model_name,
            api_key=values.get("RAG_MODEL_API_KEY"),
            timeout=float(values.get("RAG_MODEL_TIMEOUT_SECONDS", "180")),
        )
    elif require_model:
        raise LiteratureRagError(
            "A RAG planning/extraction model is required but RAG_MODEL_API_URL and "
            "RAG_MODEL_NAME are not configured"
        )
    mcp_url = str(values.get("MATERIAL_RAG_MCP_URL", "")).strip()
    mcp_tool = str(values.get("MATERIAL_RAG_MCP_TOOL", "")).strip()
    if bool(mcp_url) != bool(mcp_tool):
        raise LiteratureRagError(
            "MATERIAL_RAG_MCP_URL and MATERIAL_RAG_MCP_TOOL must be configured together"
        )
    mcp_client = None
    if mcp_url:
        try:
            StreamableHttpMcpClient.validate_tool_name(mcp_tool)
            mcp_client = StreamableHttpMcpClient(
                mcp_url,
                token=values.get("MATERIAL_RAG_MCP_TOKEN"),
                timeout=float(values.get("MATERIAL_RAG_MCP_TIMEOUT_SECONDS", "60")),
                allow_loopback_http=str(
                    values.get("MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP", "")
                ).strip()
                == "1",
            )
        except (TypeError, ValueError) as exc:
            raise LiteratureRagError(f"Invalid MCP RAG configuration: {exc}") from exc
    retriever = MultiSourceLiteratureRetriever(
        email=values.get("LITERATURE_CONTACT_EMAIL"),
        ncbi_api_key=values.get("NCBI_API_KEY"),
        openalex_api_key=values.get("OPENALEX_API_KEY"),
        mcp_client=mcp_client,
        mcp_tool=mcp_tool or None,
        user_agent=values.get(
            "LITERATURE_USER_AGENT", "discovery-os-literature-rag/1.0"
        ),
        arxiv_min_interval_seconds=float(
            values.get("LITERATURE_ARXIV_MIN_INTERVAL_SECONDS", "3")
        ),
    )
    return LiteratureRagPipeline(
        PromptSearchPlanner(model),
        retriever,
        EvidenceClaimExtractor(model),
    )


def load_evidence_bundle(path: Path | str) -> RagEvidenceBundle:
    return RagEvidenceBundle.model_validate_json(Path(path).read_text(encoding="utf-8"), strict=True)


def save_evidence_bundle(bundle: RagEvidenceBundle, path: Path | str) -> Path:
    target = Path(path)
    _write_json_atomic(target, bundle)
    return target


def deduplicate_records(records: Iterable[LiteratureRecord]) -> list[LiteratureRecord]:
    merged: dict[str, LiteratureRecord] = {}
    for original in records:
        record = LiteratureRecord.model_validate_json(original.model_dump_json(), strict=True)
        key = _record_identity_key(record)
        prior = merged.get(key)
        if prior is None:
            merged[key] = record
            continue
        identifiers = dict(prior.source_ids)
        identifiers.update(record.source_ids)
        urls = sorted(set(prior.urls + record.urls))
        queries = sorted(set(prior.source_queries + record.source_queries))
        authors = prior.authors if len(prior.authors) >= len(record.authors) else record.authors
        abstract = prior.abstract if len(prior.abstract) >= len(record.abstract) else record.abstract
        title = prior.title if len(prior.title) >= len(record.title) else record.title
        merged_record = LiteratureRecord(
            record_id=f"LIT-{stable_hash(key)[:24]}",
            title=title,
            abstract=abstract,
            publication_date=prior.publication_date or record.publication_date,
            publication_year=prior.publication_year or record.publication_year,
            authors=authors,
            venue=prior.venue or record.venue,
            doi=prior.doi or record.doi,
            pmid=prior.pmid or record.pmid,
            pmcid=prior.pmcid or record.pmcid,
            arxiv_id=prior.arxiv_id or record.arxiv_id,
            source_ids=identifiers,
            source_queries=queries,
            urls=urls,
            is_retracted=(
                True
                if prior.is_retracted is True or record.is_retracted is True
                else prior.is_retracted if prior.is_retracted is not None else record.is_retracted
            ),
            citation_count=max(
                item for item in (prior.citation_count, record.citation_count) if item is not None
            ) if any(item is not None for item in (prior.citation_count, record.citation_count)) else None,
            open_access=(prior.open_access if prior.open_access is not None else record.open_access),
            retrieved_at=max(prior.retrieved_at, record.retrieved_at),
            raw_metadata={"merged_sources": sorted(identifiers)},
        )
        merged[key] = merged_record
    return sorted(
        merged.values(),
        key=lambda item: (
            item.publication_date or date.min,
            item.record_id,
        ),
        reverse=True,
    )


# ---- Provider normalization helpers -------------------------------------------------


def _record_from_europe_pmc(item: Mapping[str, Any], query_id: str) -> LiteratureRecord:
    doi = _normalize_doi(item.get("doi"))
    pmid = _optional_text(item.get("pmid"))
    pmcid = _optional_text(item.get("pmcid"))
    source_id = pmid or pmcid or doi or _optional_text(item.get("id")) or stable_hash(item)[:16]
    pub_date = _parse_date(item.get("firstPublicationDate") or item.get("firstIndexDate"))
    title = _clean_markup(str(item.get("title", "")))
    abstract = _clean_markup(str(item.get("abstractText", "")))
    payload = ["europe_pmc", source_id, doi, pmid, title]
    return LiteratureRecord(
        record_id=f"LIT-{stable_hash(payload)[:24]}",
        title=title,
        abstract=abstract,
        publication_date=pub_date,
        publication_year=_year(pub_date, item.get("pubYear")),
        authors=_split_authors(item.get("authorString")),
        venue=_optional_text(item.get("journalTitle")),
        doi=doi,
        pmid=pmid,
        pmcid=pmcid,
        source_ids={LiteratureSource.EUROPE_PMC.value: str(source_id)},
        source_queries=[query_id],
        urls=[url for url in [_doi_url(doi), _pmid_url(pmid), _pmcid_url(pmcid)] if url],
        is_retracted=_bool_or_none(item.get("isRetracted")),
        citation_count=_int_or_none(item.get("citedByCount")),
        open_access=_bool_or_none(item.get("isOpenAccess")),
        retrieved_at=datetime.now(timezone.utc),
        raw_metadata={"source": "europe_pmc"},
    )


def _records_from_pubmed_xml(xml_text: str, query_id: str) -> list[LiteratureRecord]:
    root = ET.fromstring(xml_text)
    records: list[LiteratureRecord] = []
    for article in root.findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        data = article.find("PubmedData")
        if citation is None:
            continue
        pmid = _text(citation.find("PMID"))
        article_node = citation.find("Article")
        if article_node is None:
            continue
        title = _element_text(article_node.find("ArticleTitle"))
        if not title:
            continue
        abstract = " ".join(
            _element_text(item)
            for item in article_node.findall("Abstract/AbstractText")
            if _element_text(item)
        )
        doi = None
        pmcid = None
        if data is not None:
            for identifier in data.findall("ArticleIdList/ArticleId"):
                kind = identifier.attrib.get("IdType")
                value = _text(identifier)
                if kind == "doi":
                    doi = _normalize_doi(value)
                elif kind == "pmc":
                    pmcid = value
        pub_date = _pubmed_date(article_node)
        authors = []
        for author in article_node.findall("AuthorList/Author"):
            collective = _text(author.find("CollectiveName"))
            name = collective or " ".join(
                part
                for part in (_text(author.find("ForeName")), _text(author.find("LastName")))
                if part
            )
            if name:
                authors.append(name)
        venue = _text(article_node.find("Journal/Title"))
        payload = ["pubmed", pmid, doi, title]
        records.append(
            LiteratureRecord(
                record_id=f"LIT-{stable_hash(payload)[:24]}",
                title=_clean_markup(title),
                abstract=_clean_markup(abstract),
                publication_date=pub_date,
                publication_year=_year(pub_date, None),
                authors=authors,
                venue=venue,
                doi=doi,
                pmid=pmid,
                pmcid=pmcid,
                source_ids={LiteratureSource.PUBMED.value: pmid or stable_hash(payload)[:16]},
                source_queries=[query_id],
                urls=[url for url in [_doi_url(doi), _pmid_url(pmid), _pmcid_url(pmcid)] if url],
                retrieved_at=datetime.now(timezone.utc),
                raw_metadata={"source": "pubmed"},
            )
        )
    return records


def _record_from_openalex(item: Mapping[str, Any], query_id: str) -> LiteratureRecord:
    identifier = str(item.get("id") or stable_hash(item)[:16])
    doi = _normalize_doi(item.get("doi"))
    pub_date = _parse_date(item.get("publication_date"))
    abstract = _abstract_from_inverted_index(item.get("abstract_inverted_index"))
    authors = [
        str(row.get("author", {}).get("display_name", "")).strip()
        for row in item.get("authorships", []) or []
        if isinstance(row, dict) and str(row.get("author", {}).get("display_name", "")).strip()
    ]
    location = item.get("primary_location") or {}
    source = location.get("source") or {}
    ids = item.get("ids") or {}
    pmid = _optional_text(ids.get("pmid"))
    if pmid and pmid.startswith("https://pubmed.ncbi.nlm.nih.gov/"):
        pmid = pmid.rstrip("/").rsplit("/", 1)[-1]
    payload = ["openalex", identifier, doi, item.get("title")]
    return LiteratureRecord(
        record_id=f"LIT-{stable_hash(payload)[:24]}",
        title=_clean_markup(str(item.get("title", ""))),
        abstract=_clean_markup(abstract),
        publication_date=pub_date,
        publication_year=_year(pub_date, item.get("publication_year")),
        authors=authors,
        venue=_optional_text(source.get("display_name")),
        doi=doi,
        pmid=pmid,
        source_ids={LiteratureSource.OPENALEX.value: identifier},
        source_queries=[query_id],
        urls=[url for url in [_optional_text(item.get("doi")), _optional_text(item.get("id"))] if url],
        is_retracted=_bool_or_none(item.get("is_retracted")),
        citation_count=_int_or_none(item.get("cited_by_count")),
        open_access=_bool_or_none((item.get("open_access") or {}).get("is_oa")),
        retrieved_at=datetime.now(timezone.utc),
        raw_metadata={"source": "openalex"},
    )


def _record_from_crossref(item: Mapping[str, Any], query_id: str) -> LiteratureRecord:
    doi = _normalize_doi(item.get("DOI"))
    title_rows = item.get("title") or []
    title = _clean_markup(str(title_rows[0] if title_rows else ""))
    abstract = _clean_markup(str(item.get("abstract", "")))
    pub_date = _crossref_date(item)
    authors = []
    for author in item.get("author", []) or []:
        if not isinstance(author, dict):
            continue
        name = " ".join(
            part for part in (str(author.get("given", "")).strip(), str(author.get("family", "")).strip()) if part
        )
        if name:
            authors.append(name)
    identifier = doi or _optional_text(item.get("URL")) or stable_hash([title, pub_date])[:16]
    payload = ["crossref", identifier, title]
    return LiteratureRecord(
        record_id=f"LIT-{stable_hash(payload)[:24]}",
        title=title,
        abstract=abstract,
        publication_date=pub_date,
        publication_year=_year(pub_date, None),
        authors=authors,
        venue=_first_text(item.get("container-title")),
        doi=doi,
        source_ids={LiteratureSource.CROSSREF.value: str(identifier)},
        source_queries=[query_id],
        urls=[url for url in [_doi_url(doi), _optional_text(item.get("URL"))] if url],
        is_retracted=(
            True
            if str(item.get("type", "")).lower() in {"retraction", "correction"}
            else None
        ),
        citation_count=_int_or_none(item.get("is-referenced-by-count")),
        retrieved_at=datetime.now(timezone.utc),
        raw_metadata={"source": "crossref", "type": str(item.get("type", ""))},
    )


def _records_from_arxiv_xml(xml_text: str, query_id: str) -> list[LiteratureRecord]:
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    records: list[LiteratureRecord] = []
    for entry in root.findall("atom:entry", ns):
        identifier_url = _text(entry.find("atom:id", ns))
        arxiv_id = identifier_url.rstrip("/").rsplit("/", 1)[-1] if identifier_url else None
        title = _compact(_text(entry.find("atom:title", ns)))
        if not title:
            continue
        abstract = _compact(_text(entry.find("atom:summary", ns)))
        published = _parse_date(_text(entry.find("atom:published", ns)))
        authors = [
            _text(author.find("atom:name", ns))
            for author in entry.findall("atom:author", ns)
            if _text(author.find("atom:name", ns))
        ]
        doi = _normalize_doi(_text(entry.find("arxiv:doi", ns)))
        payload = ["arxiv", arxiv_id, doi, title]
        records.append(
            LiteratureRecord(
                record_id=f"LIT-{stable_hash(payload)[:24]}",
                title=title,
                abstract=abstract,
                publication_date=published,
                publication_year=_year(published, None),
                authors=authors,
                venue="arXiv",
                doi=doi,
                arxiv_id=arxiv_id,
                source_ids={LiteratureSource.ARXIV.value: arxiv_id or stable_hash(payload)[:16]},
                source_queries=[query_id],
                urls=[url for url in [identifier_url, _doi_url(doi)] if url],
                open_access=True,
                retrieved_at=datetime.now(timezone.utc),
                raw_metadata={"source": "arxiv"},
            )
        )
    return records


# ---- Validation, graph and branching helpers ---------------------------------------


def _validated_claim_from_model(
    record: LiteratureRecord, row: Any
) -> EvidenceClaim | None:
    if not isinstance(row, dict):
        return None
    subject = str(row.get("subject", "")).strip()
    predicate = str(row.get("predicate", "")).strip()
    object_value = str(row.get("object", "")).strip()
    support = str(row.get("support_text", "")).strip()
    if not subject or not predicate or not object_value or not support:
        return None
    if _normal_text(support) not in _normal_text(record.evidence_text):
        return None
    try:
        polarity = EvidencePolarity(str(row.get("polarity", "uncertain")))
        stage = EvidenceStage(str(row.get("stage", "unknown")))
        confidence = max(0.0, min(1.0, float(row.get("confidence", 0.5))))
    except Exception:
        return None
    qualifiers = row.get("qualifiers") if isinstance(row.get("qualifiers"), dict) else {}
    aliases_raw = row.get("entity_aliases") if isinstance(row.get("entity_aliases"), dict) else {}
    aliases = {
        str(key): _clean_string_list(value)
        for key, value in aliases_raw.items()
        if str(key).strip()
    }
    payload = [record.record_id, subject, predicate, object_value, support, polarity.value]
    return EvidenceClaim(
        claim_id=f"ECL-{stable_hash(payload)[:24]}",
        source_record_id=record.record_id,
        subject=subject,
        predicate=predicate,
        object=object_value,
        polarity=polarity,
        stage=stage,
        support_text=support,
        confidence=confidence,
        qualifiers=qualifiers,
        entity_aliases=aliases,
    )


def _deduplicate_claims(claims: Iterable[EvidenceClaim]) -> list[EvidenceClaim]:
    by_key: dict[str, EvidenceClaim] = {}
    for claim in claims:
        key = stable_hash(
            [
                claim.source_record_id,
                _normal_text(claim.subject),
                _normal_text(claim.predicate),
                _normal_text(claim.object),
                claim.polarity,
                _normal_text(claim.support_text),
            ]
        )
        prior = by_key.get(key)
        if prior is None or claim.confidence > prior.confidence:
            by_key[key] = claim
    return sorted(by_key.values(), key=lambda item: item.claim_id)


def _node_for_entity(name: str, predicate: str, *, position: str) -> EvidenceGraphNode:
    canonical = _compact(name)
    lower = f"{predicate} {canonical}".lower()
    if re.search(r"\b(cancer|disease|carcinoma|tumou?r|syndrome)\b", lower):
        node_type = "disease"
    elif re.search(r"\b(protein|gene|receptor|kinase|enzyme|target)\b", lower):
        node_type = "target"
    elif re.search(r"\b(scaffold|core|motif)\b", lower):
        node_type = "scaffold"
    elif re.search(r"\b(toxic|toxicity|adverse)\b", lower):
        node_type = "toxicity"
    elif re.search(r"\b(pressure|temperature|anneal|synthesis|condition)\b", lower):
        node_type = "condition"
    elif re.search(r"\b(dopant|doping|substitution)\b", lower):
        node_type = "dopant"
    elif re.search(r"\b(compound|drug|inhibitor|agonist|antagonist|molecule)\b", lower):
        node_type = "compound"
    elif re.search(r"\b(material|crystal|phase|alloy|oxide|hydride|formula)\b", lower):
        node_type = "material"
    elif re.fullmatch(r"[A-Z][a-z]?", canonical):
        node_type = "element"
    elif position == "object" and re.search(r"\b(gap|energy|conduct|modulus|capacity|efficacy)\b", lower):
        node_type = "property"
    else:
        node_type = "other"
    return EvidenceGraphNode(
        node_id=f"EGN-{stable_hash([node_type, _normal_text(canonical)])[:24]}",
        canonical_name=canonical,
        node_type=node_type,
    )


def _make_branch(
    kind: EvidenceBranchKind,
    title: str,
    rationale: str,
    claim_ids: Sequence[str],
    generator_hints: Mapping[str, JsonValue],
    *,
    exclusion_hints: Sequence[str] = (),
    priority: float,
) -> EvidenceBranch:
    payload = [kind.value, title, sorted(claim_ids), dict(generator_hints)]
    return EvidenceBranch(
        branch_id=f"EBR-{stable_hash(payload)[:24]}",
        kind=kind,
        title=title,
        rationale=rationale,
        source_claim_ids=sorted(set(claim_ids)),
        generator_hints=dict(generator_hints),
        exclusion_hints=list(dict.fromkeys(exclusion_hints)),
        priority=max(0.0, min(1.0, priority)),
    )


def _medicinal_hints(claims: Sequence[EvidenceClaim]) -> dict[str, JsonValue]:
    subjects = sorted({item.subject for item in claims})
    objects = sorted({item.object for item in claims})
    targets = sorted(
        {
            value
            for item in claims
            for value in [item.qualifiers.get("target"), item.qualifiers.get("protein")]
            if isinstance(value, str) and value.strip()
        }
    )
    mechanisms = sorted(
        {
            value
            for item in claims
            for value in [item.qualifiers.get("mechanism"), item.predicate]
            if isinstance(value, str) and value.strip()
        }
    )
    scaffolds = sorted(
        {
            value
            for item in claims
            for value in [item.qualifiers.get("scaffold_smiles"), item.qualifiers.get("smiles")]
            if isinstance(value, str) and value.strip()
        }
    )
    return {
        "seed_entities": subjects,
        "target_contexts": targets or objects,
        "mechanisms": mechanisms,
        "scaffold_smiles": scaffolds,
    }


def _material_hints(claims: Sequence[EvidenceClaim]) -> dict[str, JsonValue]:
    texts = [item.subject for item in claims] + [item.object for item in claims]
    chemical_systems: set[str] = set()
    elements: set[str] = set()
    dopants: set[str] = set()
    pressures: list[float] = []
    temperatures: list[float] = []
    space_groups: set[str] = set()
    for item in claims:
        for key in ("chemical_system", "composition", "formula"):
            value = item.qualifiers.get(key)
            if isinstance(value, str) and value.strip():
                system, found = _chemical_system(value)
                if system:
                    chemical_systems.add(system)
                elements.update(found)
        value = item.qualifiers.get("dopant")
        if isinstance(value, str) and value.strip():
            dopants.add(value.strip())
        value = item.qualifiers.get("pressure_gpa")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            pressures.append(float(value))
        value = item.qualifiers.get("temperature_k")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            temperatures.append(float(value))
        value = item.qualifiers.get("space_group")
        if isinstance(value, str) and value.strip():
            space_groups.add(value.strip())
    for text in texts:
        system, found = _chemical_system(text)
        if system:
            chemical_systems.add(system)
        elements.update(found)
        for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*GPa\b", text, re.I):
            pressures.append(float(match.group(1)))
        for match in re.finditer(r"(-?\d+(?:\.\d+)?)\s*K\b", text):
            temperatures.append(float(match.group(1)))
    hints: dict[str, JsonValue] = {
        "reported_entities": sorted(set(texts)),
        "elements": sorted(elements),
    }
    if chemical_systems:
        hints["chemical_system_candidates"] = sorted(chemical_systems)
        if len(chemical_systems) == 1:
            hints["chemical_system"] = next(iter(chemical_systems))
    if dopants:
        hints["dopants"] = sorted(dopants)
    if pressures:
        hints["pressure_gpa"] = sorted(set(pressures))
    if temperatures:
        hints["temperature_k"] = sorted(set(temperatures))
    if space_groups:
        hints["space_group"] = sorted(space_groups)
    return hints


# ---- General helpers ---------------------------------------------------------------


def _record_identity_key(record: LiteratureRecord) -> str:
    normalized_doi = _normalize_doi(record.doi)
    if normalized_doi:
        return f"doi:{normalized_doi}"
    if record.pmid:
        return f"pmid:{record.pmid}"
    if record.arxiv_id:
        return f"arxiv:{record.arxiv_id.lower()}"
    return "title:" + stable_hash([_normal_text(record.title), record.publication_year])


def _normalize_doi(value: Any) -> str | None:
    text = _optional_text(value)
    if not text:
        return None
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.I)
    text = re.sub(r"^doi:\s*", "", text, flags=re.I)
    return text.strip().lower() or None


def _normal_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip().casefold()


def _compact(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value))).strip()


def _clean_markup(value: str) -> str:
    return _compact(re.sub(r"<[^>]+>", " ", value))


def _clean_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(_compact(str(item)) for item in value if _compact(str(item))))


def _optional_text(value: Any) -> str | None:
    text = _compact(str(value)) if value is not None else ""
    return text or None


def _first_text(value: Any) -> str | None:
    if isinstance(value, list) and value:
        return _optional_text(value[0])
    return _optional_text(value)


def _split_authors(value: Any) -> list[str]:
    text = _optional_text(value)
    if not text:
        return []
    return [item.strip() for item in re.split(r",|;|\band\b", text) if item.strip()]


def _parse_date(value: Any) -> date | None:
    text = _optional_text(value)
    if not text:
        return None
    text = text[:10]
    for fmt in ("%Y-%m-%d", "%Y-%m", "%Y"):
        try:
            parsed = datetime.strptime(text[: len(datetime.now().strftime(fmt))], fmt)
            return parsed.date().replace(month=parsed.month or 1, day=parsed.day or 1)
        except ValueError:
            continue
    match = re.search(r"(19|20)\d{2}", text)
    return date(int(match.group(0)), 1, 1) if match else None


def _year(parsed: date | None, fallback: Any) -> int | None:
    if parsed:
        return parsed.year
    try:
        value = int(fallback)
    except (TypeError, ValueError):
        return None
    return value if 1600 <= value <= 3000 else None


def _int_or_none(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _bool_or_none(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1", "yes", "y"}:
            return True
        if value.strip().lower() in {"false", "0", "no", "n"}:
            return False
    return None


def _doi_url(doi: str | None) -> str | None:
    return f"https://doi.org/{doi}" if doi else None


def _pmid_url(pmid: str | None) -> str | None:
    return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None


def _pmcid_url(pmcid: str | None) -> str | None:
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/" if pmcid else None


def _text(element: ET.Element | None) -> str:
    return _compact(element.text or "") if element is not None else ""


def _element_text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return _compact("".join(element.itertext()))


def _pubmed_date(article: ET.Element) -> date | None:
    for path in ("Journal/JournalIssue/PubDate", "ArticleDate"):
        node = article.find(path)
        if node is None:
            continue
        year = _text(node.find("Year"))
        month = _text(node.find("Month")) or "1"
        day = _text(node.find("Day")) or "1"
        if year.isdigit():
            try:
                month_number = int(month) if month.isdigit() else datetime.strptime(month[:3], "%b").month
                return date(int(year), month_number, int(day) if day.isdigit() else 1)
            except ValueError:
                return date(int(year), 1, 1)
    medline = _text(article.find("Journal/JournalIssue/PubDate/MedlineDate"))
    return _parse_date(medline)


def _crossref_date(item: Mapping[str, Any]) -> date | None:
    for key in ("published-print", "published-online", "published", "created"):
        row = item.get(key)
        if not isinstance(row, dict):
            continue
        parts = row.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            values = parts[0]
            try:
                return date(int(values[0]), int(values[1]) if len(values) > 1 else 1, int(values[2]) if len(values) > 2 else 1)
            except (TypeError, ValueError):
                continue
    return None


def _abstract_from_inverted_index(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    positions: list[tuple[int, str]] = []
    for token, indexes in value.items():
        if not isinstance(indexes, list):
            continue
        for index in indexes:
            if isinstance(index, int):
                positions.append((index, str(token)))
    return " ".join(token for _index, token in sorted(positions))


def _infer_domain(prompt: str) -> DiscoveryDomain:
    lower = prompt.casefold()
    if any(term in lower for term in ("암", "항암", "drug", "compound", "cancer", "protein", "target", "약물")):
        return DiscoveryDomain.MEDICINAL_CHEMISTRY
    if any(term in lower for term in ("초전도", "superconduct")):
        return DiscoveryDomain.SUPERCONDUCTORS
    if any(term in lower for term in ("battery", "배터리", "전지")):
        return DiscoveryDomain.BATTERIES
    if any(term in lower for term in ("catalyst", "촉매")):
        return DiscoveryDomain.CATALYSTS
    return DiscoveryDomain.GENERAL_MATERIALS


def _goal_object(goal: DiscoveryGoal | None, prompt: str, domain: DiscoveryDomain) -> str:
    if goal:
        return goal.scientific_question
    return prompt if prompt else domain.value


def _first_entity_phrase(text: str) -> str | None:
    match = re.search(r"\b[A-Z][A-Za-z0-9+\-]{1,30}(?:\s+[A-Z][A-Za-z0-9+\-]{1,30}){0,3}\b", text)
    return match.group(0) if match else None


def _infer_stage(text: str) -> EvidenceStage:
    lower = text.casefold()
    if any(term in lower for term in ("randomized", "clinical trial", "phase i", "phase ii", "phase iii")):
        return EvidenceStage.CLINICAL_TRIAL
    if any(term in lower for term in ("patient", "cohort", "clinical")):
        return EvidenceStage.CLINICAL_OBSERVATIONAL
    if any(term in lower for term in ("mouse", "mice", "rat", "animal", "xenograft")):
        return EvidenceStage.ANIMAL
    if any(term in lower for term in ("cell line", "in vitro", "organoid")):
        return EvidenceStage.IN_VITRO
    if any(term in lower for term in ("synthesized", "synthesis", "prepared")):
        return EvidenceStage.MATERIAL_SYNTHESIS
    if any(term in lower for term in ("measured", "characterization", "xrd", "transport")):
        return EvidenceStage.MATERIAL_CHARACTERIZATION
    if any(term in lower for term in ("simulation", "computed", "dft", "predicted")):
        return EvidenceStage.COMPUTATIONAL
    return EvidenceStage.UNKNOWN


def _chemical_system(text: str) -> tuple[str | None, set[str]]:
    symbols = set(re.findall(r"(?<![a-z])[A-Z][a-z]?", text))
    valid = {
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
    }
    symbols &= valid
    return ("-".join(sorted(symbols)) if len(symbols) >= 2 else None), symbols


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


__all__ = [
    "EvidenceBranch",
    "EvidenceBranchAssignment",
    "EvidenceBranchKind",
    "EvidenceBranchPlanner",
    "EvidenceClaim",
    "EvidenceClaimExtractor",
    "EvidenceGraph",
    "EvidenceGraphBuilder",
    "JsonEvidenceIndex",
    "LiteratureEvidencePolicy",
    "LiteratureQuery",
    "LiteratureRagError",
    "LiteratureRagPipeline",
    "LiteratureRecord",
    "LiteratureSource",
    "MultiSourceLiteratureRetriever",
    "OpenAICompatibleRagModel",
    "PromptSearchPlanner",
    "RagEvidenceBundle",
    "RagSearchPlan",
    "SourceRetrievalStatus",
    "build_literature_rag_from_environment",
    "deduplicate_records",
    "load_evidence_bundle",
    "save_evidence_bundle",
]
