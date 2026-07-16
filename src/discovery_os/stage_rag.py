"""Stage-aware routing for evidence retrieval in the drug-development loop.

RAG stages have different data contracts and must not share one undifferentiated
vector index.  This module routes a query, invokes only configured stage
adapters, and records missing/failed evidence without fabricating a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

from ._compat import StrEnum as CompatStrEnum


class RagStage(CompatStrEnum):
    SCIENTIFIC_EVIDENCE = "scientific_evidence"
    BIOMEDICAL_KG = "biomedical_kg"
    GENOMICS_BIOMARKER = "genomics_biomarker"
    MOLECULE_ASSAY = "molecule_assay"
    PROTEIN_POCKET_3D = "protein_pocket_3d"
    GENERATION_RETRIEVAL = "generation_retrieval"
    MEDICINAL_TRANSFORMATION = "medicinal_transformation"
    REACTION_SYNTHESIS = "reaction_synthesis"
    CLINICAL_TRIAL = "clinical_trial"
    REGULATORY = "regulatory"
    FAILURE_MEMORY = "failure_memory"


class StageEvidenceStatus(CompatStrEnum):
    SUCCESS = "success"
    NOT_CONFIGURED = "not_configured"
    FAILED = "failed"
    EVIDENCE_INSUFFICIENT = "evidence_insufficient"
    CONFLICTING_EVIDENCE = "conflicting_evidence"


@dataclass(frozen=True)
class StageRoute:
    stage: RagStage
    retrieval_modes: tuple[str, ...]
    sources: tuple[str, ...]
    downstream_experts: tuple[str, ...]
    output_schema: str
    required: bool = False


@dataclass(frozen=True)
class StageRagQuery:
    user_query: str
    disease: str | None = None
    variant: str | None = None
    candidate_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StageEvidence:
    stage: RagStage
    status: StageEvidenceStatus
    payload: Mapping[str, Any] = field(default_factory=dict)
    source_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    error: str | None = None


ROUTES: dict[RagStage, StageRoute] = {
    RagStage.SCIENTIFIC_EVIDENCE: StageRoute(RagStage.SCIENTIFIC_EVIDENCE, ("bm25", "dense", "citation"), ("PubMed", "Europe PMC", "FDA", "EMA"), ("scientific_planner",), "grounded_claim"),
    RagStage.BIOMEDICAL_KG: StageRoute(RagStage.BIOMEDICAL_KG, ("entity_link", "1_3_hop_graph", "relation_filter"), ("Open Targets", "UniProt", "Reactome", "STRING", "ClinVar"), ("kg_validator",), "typed_graph_path"),
    RagStage.GENOMICS_BIOMARKER: StageRoute(RagStage.GENOMICS_BIOMARKER, ("variant_filter", "eQTL_lookup", "biomarker_link"), ("GWAS Catalog", "ClinVar", "GTEx", "Ensembl"), ("alphagenome", "GEARS", "CPA"), "variant_biomarker_evidence"),
    RagStage.MOLECULE_ASSAY: StageRoute(RagStage.MOLECULE_ASSAY, ("inchikey", "fingerprint", "assay_filter"), ("ChEMBL", "BindingDB", "PubChem BioAssay", "Tox21"), ("chemprop", "admet_ai"), "assay_measurement"),
    RagStage.PROTEIN_POCKET_3D: StageRoute(RagStage.PROTEIN_POCKET_3D, ("sequence_similarity", "pocket_shape", "interaction_fingerprint"), ("RCSB PDB", "AlphaFold DB"), ("boltz", "chai", "protenix", "docking"), "pocket_evidence"),
    RagStage.GENERATION_RETRIEVAL: StageRoute(RagStage.GENERATION_RETRIEVAL, ("exemplar", "fragment", "scaffold"), ("ChEMBL", "PubChem", "internal_successes"), ("reinvent", "retmol", "targetdiff"), "generation_constraints"),
    RagStage.MEDICINAL_TRANSFORMATION: StageRoute(RagStage.MEDICINAL_TRANSFORMATION, ("matched_molecular_pair", "local_environment"), ("ChEMBL", "BindingDB", "failure_memory"), ("reinvent", "transformation_agent"), "transformation_rule"),
    RagStage.REACTION_SYNTHESIS: StageRoute(RagStage.REACTION_SYNTHESIS, ("reaction_fingerprint", "condition_filter"), ("USPTO", "Open Reaction DB", "building_block_db"), ("aizynthfinder", "askcos", "chemformer"), "retrosynthesis_route"),
    RagStage.CLINICAL_TRIAL: StageRoute(RagStage.CLINICAL_TRIAL, ("metadata_filter", "endpoint_filter", "text_rerank"), ("ClinicalTrials.gov", "FDA", "EMA"), ("clinical_planner",), "trial_evidence"),
    RagStage.REGULATORY: StageRoute(RagStage.REGULATORY, ("version_filter", "page_retrieval", "rule_check"), ("FDA", "EMA", "ICH"), ("regulatory_checker",), "regulatory_evidence"),
    RagStage.FAILURE_MEMORY: StageRoute(RagStage.FAILURE_MEMORY, ("structured_failure_filter", "analog_lookup", "negative_evidence"), ("run_artifacts", "assay_failures", "clinical_failures"), ("failure_controller",), "failure_constraint"),
}


class StageAwareRagRouter:
    """Small deterministic router; it never chooses an adapter from model text."""

    def route(self, query: StageRagQuery) -> list[StageRoute]:
        text = f"{query.user_query} {query.disease or ''} {query.variant or ''}".lower()
        selected: list[RagStage] = [RagStage.SCIENTIFIC_EVIDENCE]
        rules = (
            (("variant", "mutation", "biomarker", "genomic", "gwas", "clinvar"), RagStage.GENOMICS_BIOMARKER),
            (("gene", "pathway", "disease", "target", "drug interaction"), RagStage.BIOMEDICAL_KG),
            (("smiles", "molecule", "assay", "ic50", "ki", "admet", "tox"), RagStage.MOLECULE_ASSAY),
            (("pocket", "binding", "pdb", "structure", "docking"), RagStage.PROTEIN_POCKET_3D),
            (("generate", "scaffold", "fragment", "analog"), RagStage.GENERATION_RETRIEVAL),
            (("synthesis", "retrosynthesis", "reaction", "yield"), RagStage.REACTION_SYNTHESIS),
            (("trial", "patient", "endpoint", "phase"), RagStage.CLINICAL_TRIAL),
            (("fda", "ema", "ich", "regulatory", "ind"), RagStage.REGULATORY),
            (("failed", "failure", "inactive", "terminated", "negative"), RagStage.FAILURE_MEMORY),
        )
        for words, stage in rules:
            if any(word in text for word in words) and stage not in selected:
                selected.append(stage)
        return [ROUTES[stage] for stage in selected]


StageAdapter = Callable[[StageRagQuery, StageRoute], Mapping[str, Any]]


class StageAwareMultiRag:
    def __init__(self, adapters: Mapping[RagStage | str, StageAdapter] | None = None, *, router: StageAwareRagRouter | None = None) -> None:
        self.adapters = {RagStage(key): value for key, value in (adapters or {}).items()}
        self.router = router or StageAwareRagRouter()

    def run(self, query: StageRagQuery) -> list[StageEvidence]:
        evidence: list[StageEvidence] = []
        for route in self.router.route(query):
            adapter = self.adapters.get(route.stage)
            if adapter is None:
                evidence.append(StageEvidence(route.stage, StageEvidenceStatus.NOT_CONFIGURED, warnings=("No stage adapter is configured; no evidence was fabricated.",)))
                continue
            try:
                payload = dict(adapter(query, route))
                status = StageEvidenceStatus(str(payload.pop("status", "success")))
                evidence.append(StageEvidence(route.stage, status, payload=payload, source_ids=tuple(str(item) for item in payload.pop("source_ids", ())), warnings=tuple(str(item) for item in payload.pop("warnings", ()))))
            except Exception as exc:
                evidence.append(StageEvidence(route.stage, StageEvidenceStatus.FAILED, error=f"{type(exc).__name__}: {exc}"))
        return evidence
