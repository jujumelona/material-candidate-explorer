# Stage-Aware Multi-RAG

The drug-development workflow does not use one universal RAG index. A query is routed to stage-specific retrieval contracts, each with its own sources, filters, output schema, and downstream expert models.

```text
user goal
  -> deterministic stage router
  -> Scientific Evidence / Biomedical KG / Genomics / Assay / Pocket / Generation /
     Transformation / Synthesis / Clinical / Regulatory / Failure Memory RAG
  -> expert evaluation
  -> evidence-aware controller
```

The router always includes Scientific Evidence RAG and adds stages based on explicit query terms and metadata. Adapters are injected by stage; an unconfigured stage returns `not_configured`, never a guessed result. Adapter failures return `failed`, while insufficient or conflicting evidence remains explicit in the evidence status.

## Adapter contract

```python
from discovery_os.stage_rag import StageAwareMultiRag, StageRagQuery, RagStage

def molecule_assay(query, route):
    return {
        "source_ids": ["CHEMBL..."],
        "query_candidate": query.candidate_ids[0],
        "measurement": "IC50",
        "unit": "nM",
    }

evidence = StageAwareMultiRag({RagStage.MOLECULE_ASSAY: molecule_assay}).run(
    StageRagQuery(
        user_query="KRAS G12D molecule assay and ADMET evidence",
        disease="pancreatic cancer",
        variant="KRAS G12D",
        candidate_ids=("candidate_001",),
    )
)
```

The existing literature pipeline can be exposed as the Scientific Evidence adapter. AlphaGenome belongs in the Genomics/Biomarker stage as an evaluator after variant candidates are proposed; it must not replace the stage router or generate clinical conclusions.

## Safety and evidence boundaries

- Preserve source IDs, publication dates, assay units, species, cell line, endpoint, and negative evidence.
- Apply hard metadata filters before semantic reranking for assays, trials, and regulations.
- Do not compare IC50, Ki, Kd, EC50, or assay contexts as one scalar.
- Regulatory and clinical outputs are planning evidence only; deterministic rule checks and qualified review remain required.
- A missing adapter is an explicit coverage gap, not a successful prediction.
