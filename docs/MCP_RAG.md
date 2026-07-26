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

All five dedicated variables are optional. When any one is set,
`MATERIAL_RAG_MCP_URL` is required. For each stage, the router selects its
dedicated variable first, then the generic fallback. If neither names a tool,
MCP is recorded as unconfigured for that stage while its scholarly providers
continue. Leaving the URL and every tool field blank is therefore a supported
no-MCP configuration, not an error.

Use `--rag-source mcp` for a non-staged literature request that selects only MCP, or omit `--rag-source` to use it with scholarly providers. The five-stage validation router owns its source allowlists and tool selection.

## Five distinct research policies

The router does not reuse one generic deterministic query at all checkpoints.
It expands every required intent below for every scholarly source allowed at
that stage. The policy ID, version, required intent IDs, and per-intent coverage
are persisted with the bundle and report.

| Stage | Policy ID | Required query intents | Stage scope argument |
|---|---|---|---|
| `generation_prior` | `material-generation-evidence-v2` | `successful_target`, `impurity_or_partial`, `failed_no_target`, `condition_window`, `generator_condition_limit` | `generation_scope` |
| `identity_novelty` | `material-identity-evidence-v2` | `exact_formula_alias`, `polymorph_and_conditions`, `disorder_and_occupancy`, `federated_structure_records`, `identity_method_scope` | `identity_scope` |
| `mlip_disagreement` | `material-mlip-evidence-v2` | `model_training_domain`, `energy_alignment`, `force_stress_error`, `electronic_state_caveat`, `uncertainty_and_extrapolation` | `mlip_scope` |
| `relaxation_validation` | `material-relaxation-evidence-v2` | `optimizer_convergence`, `phase_transformation`, `geometry_failure`, `phonon_instability`, `finite_temperature_phase` | `relaxation_scope` |
| `dft_handoff` | `material-dft-evidence-v2` | `reference_phase_policy`, `electronic_method_policy`, `pseudopotential_verification`, `numerical_convergence`, `specialized_workflow` | `dft_scope` |

Every tool accepts the same common arguments: `query`, `max_results`,
`from_date`, `to_date`, `stage`, `intent_id`, `chemical_system`,
`material_field`, `application_subtype`, `composition_keys`,
`candidate_refs`, and `record_types`. It must additionally accept the one
stage-scope argument in the table.

An intent is `covered` only when at least one returned record is linked to a
query with that exact intent ID. A successful provider call with no record for
an intent is `no_records`; it is not converted into negative evidence. If any
required intent is not covered, the report cannot be `completed`: available
records produce `partial`, while no usable grounded records produce `unknown`.
This preserves publication bias and incomplete-index coverage instead of
treating search absence as novelty, stability, or synthesis failure.

The policies are code-owned. A RAG model may help extract source-closed claims,
but it cannot delete an intent, change the policy, introduce a tool, or choose
an endpoint.

## Tool discovery and input contract

The client implements the stable [MCP `2025-11-25` Streamable HTTP lifecycle](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports): initialize, initialized notification, bounded `tools/list`, and `tools/call`. It preserves `Mcp-Session-Id`, accepts completed JSON or POST-SSE replies, limits decompressed responses to 16 MiB, refuses redirects, and rejects insecure non-loopback HTTP. Local HTTP requires both a loopback host and `MATERIAL_RAG_MCP_ALLOW_LOOPBACK_HTTP=1`.

Before a stage sends evidence arguments, the client requires `tools/list` to
advertise the configured tool exactly once. Its `inputSchema` must describe an
object and declare all common arguments plus that stage's scope argument:

```json
{
  "query": "Li-O successful characterized target phase synthesis conditions",
  "max_results": 25,
  "from_date": "2024-01-01",
  "to_date": null,
  "stage": "generation_prior",
  "intent_id": "successful_target",
  "chemical_system": "Li-O",
  "material_field": "general_inorganic",
  "application_subtype": null,
  "composition_keys": ["Li2O"],
  "candidate_refs": [],
  "record_types": ["reported_phase", "synthesis_success"],
  "generation_scope": {
    "declared_context": {"temperature": 300, "pressure": 1.0},
    "focus_terms": ["low-energy oxide"],
    "evidence_only": true,
    "property_score_authority": false
  }
}
```

The client sends only bounded, non-secret request context. The discovery prompt,
planner output, candidate observation, or returned record cannot add arguments
or override `stage`, `intent_id`, `record_types`, or the stage scope.

## Structured record and provenance contract

The call must return a structured JSON object with a `records` array. Every
record requires all six top-level fields:

```json
{
  "records": [
    {
      "source_id": "provider:stable-record-id",
      "title": "Source title",
      "record_type": "synthesis_success",
      "support_text": "The bounded source passage that supports the record.",
      "provenance": {
        "provider": "configured-provider",
        "provider_version": "provider-version",
        "snapshot_id": "snapshot-or-index-version",
        "source_locator": "stable-source-locator",
        "retrieved_at": "2026-07-26T00:00:00Z",
        "request_hash": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "record_hash": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
      },
      "stage_metadata": {
        "chemical_system": "Li-O",
        "composition": "Li2O",
        "outcome": "success",
        "conditions": {"temperature_K": 973},
        "evidence_polarity": "supports"
      }
    }
  ]
}
```

The six fields are `source_id`, `title`, `record_type`, `support_text`,
`provenance`, and `stage_metadata`. Every stage requires the same seven
provenance fields shown above. Its `record_type` must be one expected by the
current intent and one allow-listed by the current policy.

Both hashes use the repository's canonical `stable_hash` JSON encoding.
`request_hash` must equal the hash of the exact tool argument object sent by
the client. `record_hash` must equal the hash of the complete returned record
with only `provenance.record_hash` omitted. The adapter recomputes both values;
arbitrary 64-character placeholders, stale request bindings, blank required
metadata, malformed retrieval timestamps, and modified records are rejected.

| Stage | Allowed `record_type` values |
|---|---|
| `generation_prior` | `reported_phase`, `synthesis_success`, `synthesis_partial`, `synthesis_failure`, `composition_window`, `synthesis_condition_window`, `generator_condition_limit` |
| `identity_novelty` | `crystallographic_entry`, `structure_alias`, `polymorph`, `pressure_temperature_phase`, `disordered_structure`, `database_structure_record`, `identity_method_limit` |
| `mlip_disagreement` | `model_card_limit`, `same_composition_reference`, `benchmark_result`, `magnetic_charge_caveat`, `out_of_domain_case`, `uncertainty_method` |
| `relaxation_validation` | `optimization_reference`, `phase_transformation`, `geometry_failure`, `phonon_instability`, `phonon_method_limit`, `finite_temperature_phase` |
| `dft_handoff` | `reference_phase`, `method_policy`, `pseudopotential_verification`, `convergence_study`, `specialized_workflow`, `provenance_reference` |

`stage_metadata` is stage-specific:

| Stage | Required metadata fields |
|---|---|
| `generation_prior` | `chemical_system`, `composition`, `outcome`, `conditions`, `evidence_polarity` |
| `identity_novelty` | `database_name`, `database_entry_id`, `formula`, `structure_locator`, `match_scope` |
| `mlip_disagreement` | `model_id`, `model_version`, `limitation_kind`, `property_scope`, `evaluation_scope` |
| `relaxation_validation` | `method`, `convergence_criterion`, `pressure`, `temperature`, `instability_kind` |
| `dft_handoff` | `workflow_type`, `code`, `code_version`, `method`, `convergence_scope` |

If the server publishes `outputSchema`, `records.items` must be an object whose
declared `properties` and `required` set include the six required record
fields. The adapter validates the complete returned record, provenance, stage
metadata, and record type even when `outputSchema` is absent. Unstructured
prose is rejected.

A missing tool, duplicate advertisement, incompatible schema, malformed output, or failed call removes MCP from that stage and records the contract status or source as failed/skipped. It does not fall back to model memory. Other allowed scholarly sources may still produce a `partial` report; no grounded records produce `unknown`.

Resumable SSE with `Last-Event-ID`, server-initiated requests, elicitation, sampling, and task-augmented tool calls are intentionally outside this bounded evidence client.

## Scientific boundary

MCP records are deduplicated with the other providers and pass through the same
source-grounded claim extraction, conflict graph, and evidence-branch policy.
They are search context, not runtime validation. They cannot become energy,
force, stress, hull, novelty, relaxation, Pareto, or DFT values. Structure
matching, database lookup, MLIP sidecars, relaxation gates, and the selected
DFT backend remain authoritative for their respective outputs.

Generation steering is narrower still. Only a `generation_prior` branch may
steer, and only when its claim is tied to a returned source record and literal
support passage, has explicit non-uncertain polarity and typed synthesis
outcome/conditions, and uses a MatterGen-supported condition. A title alone,
record absence, an uncertain claim, or a later-stage record cannot steer.

Use the repository-local Codex skill [`$material-candidate-validation`](../.codex/skills/material-candidate-validation/SKILL.md) when implementing or auditing this boundary. The skill is procedural guidance; it neither starts an MCP server nor replaces the typed runtime contracts.
