# MCP evidence sources for material RAG

Discovery OS can supplement PubMed, Europe PMC, OpenAlex, Crossref, and arXiv with administrator-configured MCP Streamable HTTP evidence tools. Stage routing can use a different allow-listed tool on the same endpoint for each scientific question. A discovery prompt, planner, model response, observation, or MCP result cannot choose an endpoint or tool.

## Configuration

Configure the endpoint and only the stage tools that exist on that server. `MATERIAL_RAG_MCP_TOOL` is the generic fallback for a stage whose dedicated variable is blank.

```bash
export MATERIAL_RAG_MCP_URL="https://YOUR-MCP-SERVER/mcp"
export MATERIAL_RAG_MCP_TOOL="search_material_evidence"
export MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR="search_generation_prior"
export MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY="search_crystal_identity"
export MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT="search_mlip_limits"
export MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION="search_relaxation_instability"
export MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF="search_periodic_dft_methods"
export MATERIAL_RAG_MCP_TOKEN="..."   # only when required
export MATERIAL_RAG_MCP_TIMEOUT_SECONDS="60"
```

All five dedicated variables are optional. When any one is set, `MATERIAL_RAG_MCP_URL` is required. For each stage, the router selects its dedicated variable first, then the generic fallback; if neither names a tool, MCP is skipped for that stage while its scholarly providers continue.

Use `--rag-source mcp` for a non-staged literature request that selects only MCP, or omit `--rag-source` to use it with scholarly providers. The five-stage validation router owns its source allowlists and tool selection.

## Tool discovery and contract verification

The client implements the stable [MCP `2025-11-25` Streamable HTTP lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports): initialize, initialized notification, bounded `tools/list`, and `tools/call`. It preserves `Mcp-Session-Id`, accepts completed JSON or POST-SSE replies, limits decompressed responses to 16 MiB, refuses redirects, and rejects insecure non-loopback HTTP. Local HTTP requires both a loopback host and `MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP=1`.

Before a stage sends evidence arguments, the client requires `tools/list` to advertise the configured tool exactly once. Its `inputSchema` must describe an object and declare all adapter arguments:

```json
{
  "query": "Li-O stable oxide synthesis",
  "max_results": 25,
  "from_date": "2024-01-01",
  "to_date": null
}
```

The call must return a structured JSON object with a `records` array. Every record requires `source_id` and `title`. Optional fields are `abstract` or `support_text`, `publication_date`, `publication_year`, `authors`, `venue`, `doi`, `pmid`, `pmcid`, `url`, `is_retracted`, `citation_count`, and `open_access`. If the server publishes `outputSchema`, it must declare `records` as an array; the adapter still validates returned records at runtime when `outputSchema` is absent. Unstructured prose is rejected as evidence.

A missing tool, duplicate advertisement, incompatible schema, malformed output, or failed call removes MCP from that stage and records the contract status or source as failed/skipped. It does not fall back to model memory. Other allowed scholarly sources may still produce a `partial` report; no grounded records produce `unknown`.

Resumable SSE with `Last-Event-ID`, server-initiated requests, elicitation, sampling, and task-augmented tool calls are intentionally outside this bounded evidence client.

## Scientific boundary

MCP records are deduplicated with the other providers and pass through the same source-grounded claim extraction, conflict graph, and evidence-branch policy. They are search context, not runtime validation. They cannot become energy, force, stress, hull, novelty, relaxation, Pareto, or DFT values. Structure matching, Materials Project lookup, MLIP sidecars, relaxation gates, and the selected DFT backend remain authoritative for their respective outputs.

Use the repository-local Codex skill [`$material-candidate-validation`](../.codex/skills/material-candidate-validation/SKILL.md) when implementing or auditing this boundary. The skill is procedural guidance; it neither starts an MCP server nor replaces the typed runtime contracts.
