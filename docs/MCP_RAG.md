# MCP evidence source for material RAG

Discovery OS can supplement PubMed, Europe PMC, OpenAlex, Crossref, and arXiv with one configuration-owned MCP Streamable HTTP evidence tool. A prompt or model response cannot choose an arbitrary endpoint or tool.

```bash
export MATERIAL_RAG_MCP_URL="https://YOUR-MCP-SERVER/mcp"
export MATERIAL_RAG_MCP_TOOL="search_material_evidence"
export MATERIAL_RAG_MCP_TOKEN="..."   # only when required
export MATERIAL_RAG_MCP_TIMEOUT_SECONDS="60"
```

Use `--rag-source mcp` to select only MCP, or omit `--rag-source` to use it with the scholarly providers. Missing MCP configuration produces a `skipped` source status without stopping other providers.

The client implements the stable [MCP `2025-11-25` Streamable HTTP lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports): initialize, initialized notification, and tools/call. It preserves `Mcp-Session-Id`, accepts completed JSON or POST-SSE replies, limits decompressed responses to 16 MiB, refuses redirects, and rejects insecure non-loopback HTTP. Local HTTP requires both a loopback host and `MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP=1`.

Resumable SSE with `Last-Event-ID`, server-initiated requests, elicitation, sampling, and task-augmented tool calls are intentionally outside this bounded evidence client.

The tool arguments are:

```json
{
  "query": "Li-O stable oxide synthesis",
  "max_results": 25,
  "from_date": "2024-01-01",
  "to_date": null
}
```

The tool must return a JSON object with a `records` array. Every record requires `source_id` and `title`. Optional fields are `abstract` or `support_text`, `publication_date`, `publication_year`, `authors`, `venue`, `doi`, `pmid`, `pmcid`, `url`, `is_retracted`, `citation_count`, and `open_access`. Unstructured prose is rejected as evidence.

MCP evidence is deduplicated with the other providers and then passes through the same source-grounded claim extraction, conflict graph, and evidence-branch policy. Literature evidence guides where to search; expert models still determine candidate properties.
