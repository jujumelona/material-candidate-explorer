# Stage-specific validation evidence

`discovery_os.validation_evidence` connects numerical and structural validation stages to bounded scholarly retrieval and administrator-configured, stage-specific MCP evidence tools. It is shared by the library, CLI, and `MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb`.

The evidence router supplements a validator; it never replaces one. Literature record counts, citation counts, model summaries, and MCP responses cannot become energies, forces, hull values, novelty booleans, relaxation convergence, Pareto utilities, or DFT results.

RAG and MCP are separate boundaries. RAG retrieves scholarly metadata and closes claims to source records. The optional MCP route supplies more structured evidence records through an administrator-selected, schema-checked tool. It does not authorize an action or establish that a generator, expert, matcher, database client, relaxation, or DFT calculation ran. Those actions remain with the runtime authority named in the route.

## Routes

| Stage value | Runtime authority | Scholarly sources | Stage MCP variable | Context retrieved |
|---|---|---|---|---|
| `generation_prior` | fixed allowlist for a named official MatterGen checkpoint and deterministic Fusion controller; a custom allowlist is recorded as operator attestation, not automatically verified training metadata | Crossref, arXiv, OpenAlex, optional MCP | `MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR` | phases, negative synthesis evidence, composition ranges, stability constraints |
| `identity_novelty` | source-Niggli non-symmetrized identity plus strict unscaled pymatgen `StructureMatcher`; optional Materials Project `find_structure` is a similarity prefilter followed by local strict recheck | Crossref, arXiv, OpenAlex, optional MCP | `MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY` | reported phases, crystallographic aliases, scoped database context |
| `mlip_disagreement` | separately executed MatterSim and CHGNet properties with explicit units and launcher-verified exact weight SHA-256 values; aligned same-composition relative energy for cross-model energy evidence | Crossref, arXiv, optional MCP | `MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT` | applicability limits, out-of-domain chemistry, magnetic and charge-state caveats |
| `relaxation_validation` | separate `/v1/relax` payloads, optimizer convergence, and strict geometry gates | Crossref, arXiv, optional MCP | `MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION` | transformations, mechanical/dynamical instability, pressure/temperature, phonons |
| `dft_handoff` | actual executing `PeriodicDFTBackend`; completed results require input-manifest, method-policy, immutable output/convergence evidence, and applicable reference-set or phonon provenance | Crossref, arXiv, optional MCP | `MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF` | reference phases, magnetic order, functional/U, pseudopotential and convergence review |

The route stores its official-validator identifiers as provenance. Actual candidate values still come from the corresponding structure matcher, database lookup, expert sidecar, relaxation endpoint, or DFT backend.

## Configuration

Credentials are never embedded in a request file. Crossref and arXiv can run without an API key; OpenAlex requires its free key and is recorded as skipped when that key is blank. Configure only the integrations you intend to use:

~~~bash
export VALIDATION_EVIDENCE_ENABLED=1
export VALIDATION_EVIDENCE_MAX_RESULTS=8
export VALIDATION_EVIDENCE_MAX_BRANCHES=12
export VALIDATION_EVIDENCE_FROM_DATE=""       # blank keeps older phase literature
export LITERATURE_CONTACT_EMAIL="researcher@example.org"
export OPENALEX_API_KEY=""                  # required only to enable OpenAlex
export LITERATURE_ARXIV_MIN_INTERVAL_SECONDS="3"

# Optional OpenAI-compatible planner and claim extractor; set both or neither.
export RAG_MODEL_API_URL="https://YOUR-ENDPOINT/v1"
export RAG_MODEL_NAME="YOUR-MODEL"
export RAG_MODEL_API_KEY=""                   # runtime secret when required

# Optional MCP evidence source. A dedicated stage tool overrides the fallback.
export MATERIAL_RAG_MCP_URL="https://YOUR-MCP-SERVER/mcp"
export MATERIAL_RAG_MCP_TOOL="search_material_evidence"
export MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR="search_generation_prior"
export MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY="search_crystal_identity"
export MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT="search_mlip_limits"
export MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION="search_relaxation_instability"
export MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF="search_periodic_dft_methods"
export MATERIAL_RAG_MCP_TOKEN=""              # runtime secret when required
~~~

The MCP endpoint and tool names are administrator configuration. They are not accepted from the discovery prompt, RAG-model output, stage observations, or an MCP response. Each route selects its dedicated tool first and `MATERIAL_RAG_MCP_TOOL` second. If any dedicated variable is set, the URL is required; a stage with neither a dedicated tool nor the fallback records MCP as unconfigured and continues its other sources. HTTPS is required outside an explicitly opted-in loopback development endpoint.

Before retrieval, the MCP client performs bounded `tools/list` discovery and requires the selected tool to be advertised exactly once. Its object `inputSchema` must declare `query`, `max_results`, `from_date`, and `to_date`. A published `outputSchema` must declare a `records` array; output is still runtime-validated when that optional schema is absent. Contract failure omits MCP for that stage and never falls back to model memory. See [MCP evidence sources for material RAG](MCP_RAG.md).

The evidence route accepts records only. A same-named MCP tool that claims to mutate candidates, run relaxation, submit DFT, write a database, or replace a validator is outside this contract and must not be invoked through this path.

## CLI

Example `validation-stage.json`:

~~~json
{
  "schema_version": "1.0",
  "stage": "mlip_disagreement",
  "chemical_system": "Li-O",
  "candidate_refs": [],
  "composition_keys": ["Li2O"],
  "observations": {
    "models": ["MatterSim", "CHGNet"],
    "high_disagreement_candidates": 2,
    "property_scores_created_from_literature": false
  },
  "focus": "Find published model limitations relevant to this disagreement."
}
~~~

Run it with:

~~~bash
discovery-os validation-evidence \
  --request validation-stage.json \
  --goal goal.json \
  --artifacts runs/material-run-001
~~~

The command prints a strict `ValidationEvidenceReport` and persists both the report and any source-grounded bundle under `validation-evidence/<stage>/`. The report records MCP contract verification, the selected tool name, validator authorities, and a typed evidence handoff. `validator_execution_state="not_executed"` is intentional: the evidence router describes the required validator but does not pretend to have run it.

## Typed handoffs

Every route emits a fail-closed `ValidationEvidenceHandoff`. It keeps candidate/composition identity, evidence status, bundle identity, the expected consumer and payload schema, and the required validator authority IDs together.

| Stage | Handoff kind | Consumer | Validator payload/result contract |
|---|---|---|---|
| `generation_prior` | `generation_constraint_context` | `FusionDecisionContext` | `FusionDecisionContext`; usable source-closed evidence may steer generation |
| `identity_novelty` | `identity_novelty_context` | `StagedNoveltyAssessor` | `ScientificNoveltyAssessment` |
| `mlip_disagreement` | `mlip_disagreement_context` | `classify_model_disagreement` | `ModelDisagreement` |
| `relaxation_validation` | `relaxation_gate_context` | `PeriodicRelaxationResult` | `PeriodicRelaxationPayload` |
| `dft_handoff` | `dft_preparation_context` | `PeriodicDFTBackend` | `DFTInputHandoffReport`, followed by separately recorded backend execution |

All handoffs require a validator result, fix `evidence_can_replace_validator` to `false`, and use `unknown-not-pass` failure semantics. Only a `generation_prior` handoff with persisted usable evidence can set `can_steer_generation=true`; all later-stage handoffs are context for their named validator, not generator instructions.

## Generation binding

Only `generation_prior` evidence can be converted to `FusionDecisionContext`. Source claim identifiers, branch identifiers, rationale, and generator hints remain closed together. `fusion_decision_contexts_from_stage_evidence()` shares one deterministic branch policy across several profile workers so available evidence branches are not reset for every worker.

Pass the resulting JSON to a normal iteration:

~~~bash
discovery-os fusion-iterate \
  --goal goal.json \
  --parent parent.json \
  --run-config run-config.json \
  --decision-context decision-context.json \
  --generator mattergen \
  --expert mattersim \
  --expert chgnet
~~~

Non-generation evidence cannot steer a generator. The Evidence Fusion backend also filters material hints through the MatterGen-supported condition allowlist.

## Failure and audit semantics

- disabled route: `skipped` with a reason
- missing pipeline or failed retrieval: `unknown` with a reason
- no source-grounded records: `unknown`
- failed or incompatible stage MCP tool contract: omit MCP, preserve a failed contract status, and do not use model memory
- one or more unavailable sources with some records: `partial`
- every requested source successful with records: `completed`
- validator unavailable or failed: `unknown-not-pass`, regardless of retrieved literature
- absence from literature or a structure database: not proof of novelty or validity
- the selector branch ID `novelty`: property-space diversity only; external structural novelty requires the staged assessor and scoped provider provenance
- strict external scoped no-match: eligible only for a bounded shortlist tie-break (at most one reserved DFT slot); `unknown` gets no credit and no-match is not a novelty proof

Every report keeps source statuses and hashes, sets `scientific_role` to `search_and_validation_context_only`, and fixes `property_score_created` to `false`. Requests reject secret-like observation keys. The T4 notebook separately scans all hidden credentials against every exported artifact before creating the result archive.

## Codex skill

Use the repository-local [`$material-candidate-validation`](../.codex/skills/material-candidate-validation/SKILL.md) skill when Codex implements, audits, or documents this workflow. It provides procedural routing guidance for the same five stage IDs. It is not a runtime expert, does not invoke MCP by itself, and cannot replace the schemas, sidecars, databases, or DFT backend above.

## Official references

- [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/)
- [arXiv API user manual](https://info.arxiv.org/help/api/user-manual.html)
- [OpenAlex developer documentation](https://developers.openalex.org/)
- [pymatgen `StructureMatcher`](https://pymatgen.org/pymatgen.core.html)
- [Materials Project `MPRester`](https://materialsproject.github.io/api/_autosummary/mp_api.client.mprester.MPRester.html)
- [MatterSim relaxation example](https://microsoft.github.io/mattersim/examples/relax_example.html)
- [CHGNet API](https://chgnet.lbl.gov/api)
- [ASE structure optimization](https://ase-lib.org/ase/optimize.html)
- [Quantum ESPRESSO `pw.x` input reference](https://www.quantum-espresso.org/Doc/INPUT_PW.html)
- [MCP 2025-11-25 Streamable HTTP transport](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)
