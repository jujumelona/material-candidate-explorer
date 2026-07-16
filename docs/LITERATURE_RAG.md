# Literature RAG for evidence-guided material search

Literature evidence controls where the search explores. It is never converted into a predicted material property or treated as proof that a candidate is stable, synthesizable, or novel.

The material-search path is:

```text
discovery prompt
  -> bounded queries for each configured source
  -> normalized records with source identifiers
  -> exact-span claim extraction
  -> conflict-preserving evidence graph
  -> generator search-prior branches
  -> MatterGen candidate generation
  -> MatterSim and CHGNet evaluation
  -> branch reward/collapse/failure observation
  -> evidence and generation priorities adapted for the next round
```

Every extracted claim keeps the record identifier and an exact contiguous `support_text` span from the retrieved title or abstract. Unsupported model output is rejected. Positive, negative, null, uncertain, and conflicting results remain separate.

## Sources

The built-in retriever supports:

- PubMed
- Europe PMC
- OpenAlex
- Crossref
- arXiv
- one optional configuration-owned MCP Streamable HTTP tool

Sources run independently. One failed or unconfigured source does not erase successful records from the others. The evidence bundle records each source status, endpoint, query identifiers, result count, elapsed time, and bounded error message.

OpenAlex requires `OPENALEX_API_KEY` in this project. PubMed can run without an NCBI key, but a contact email and key are recommended for identified, rate-limited use.

```bash
export LITERATURE_CONTACT_EMAIL="researcher@example.com"
export NCBI_API_KEY=""                       # optional
export OPENALEX_API_KEY=""                   # optional; OpenAlex is skipped when blank
export LITERATURE_USER_AGENT="DiscoveryOS/0.4 researcher@example.com"
```

## Optional planning and extraction model

The pipeline accepts an OpenAI-compatible `chat/completions` JSON endpoint:

```bash
export RAG_MODEL_API_URL="https://YOUR-ENDPOINT/v1"
export RAG_MODEL_NAME="YOUR-MODEL"
export RAG_MODEL_API_KEY=""                  # only when required
export RAG_MODEL_TIMEOUT_SECONDS="180"
```

The model only proposes bounded searches and extracts claims from supplied records. It does not generate expert property values. If no model is configured, the pipeline uses a conservative deterministic plan and title-level claim fallback. Use `--require-model` or `--rag-require-model` when a model-backed run is mandatory.

## Optional MCP evidence tool

Configure the endpoint and tool together:

```bash
export MATERIAL_RAG_MCP_URL="https://YOUR-MCP-SERVER/mcp"
export MATERIAL_RAG_MCP_TOOL="search_material_evidence"
export MATERIAL_RAG_MCP_TOKEN=""             # only when required
export MATERIAL_RAG_MCP_TIMEOUT_SECONDS="60"
```

If both endpoint and tool are blank, MCP is recorded as skipped. A partial configuration fails before retrieval. See [MCP_RAG.md](MCP_RAG.md) for the strict tool arguments, record schema, HTTPS policy, and local-loopback opt-in.

## Create one grounded evidence bundle

```bash
discovery-os rag-update \
  --prompt "Find experimentally characterized Li-O phases, stability evidence, and failed synthesis conditions" \
  --from-date 2024-01-01 \
  --max-results 25 \
  --max-branches 24 \
  --index .discovery/evidence-index \
  --output runs/li-o/literature-evidence.json
```

Omit `--source` to use all built-in sources. Repeat `--source` to select a subset, for example `--source europe_pmc --source crossref --source mcp`.

## Use RAG inside the real multi-round loop

The recommended reproducible path builds the evidence bundle once and passes it to `fusion-search`:

```bash
discovery-os fusion-search \
  --search-id li-o-001 \
  --goal goal.json \
  --parent parent.json \
  --run-config run-config.json \
  --generator mattergen \
  --rounds 4 \
  --frontier-width 2 \
  --expert mattersim \
  --expert chgnet \
  --required-evaluator mattersim \
  --required-evaluator chgnet \
  --rag-bundle runs/li-o/literature-evidence.json \
  --artifacts runs/li-o/search
```

You can replace `--rag-bundle` with `--rag-prompt` and the `--rag-*` retrieval options to retrieve immediately before the search.

`LiteratureEvidencePolicy` assigns an evidence branch to each exploration branch in every round. After expert evaluation, it updates that branch's weight from objective improvement, structural collapse, and execution failure. The same immutable evidence bundle is therefore used adaptively throughout the closed loop; the search does not repeatedly query the network and discard earlier results.

## Validation boundary

Search rank is diagnostic. Candidate properties come from the configured expert panel, and thermodynamic stability still requires reference phases, relaxation, a validated convex-hull workflow, and appropriate higher-fidelity or experimental checks. Literature recency, citation count, and model confidence are not material-property scores.
