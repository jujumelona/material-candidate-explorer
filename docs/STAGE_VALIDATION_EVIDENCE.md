# Stage-specific validation evidence

`discovery_os.validation_evidence` connects numerical and structural validation stages to bounded scholarly retrieval and one configuration-owned MCP evidence tool. It is shared by the library, CLI, and `MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb`.

The evidence router supplements a validator; it never replaces one. Literature record counts, citation counts, model summaries, and MCP responses cannot become energies, forces, hull values, novelty booleans, relaxation convergence, Pareto utilities, or DFT results.

## Routes

| Stage value | Runtime authority | Scholarly sources | Context retrieved |
|---|---|---|---|
| `generation_prior` | MatterGen-supported condition allowlist and deterministic Fusion controller | Crossref, arXiv, OpenAlex, optional configured MCP | phases, negative synthesis evidence, composition ranges, stability constraints |
| `identity_novelty` | pymatgen `StructureMatcher` and optional Materials Project `find_structure` | Crossref, arXiv, OpenAlex, optional configured MCP | reported phases, crystallographic aliases, scoped database context |
| `mlip_disagreement` | separate MatterSim and CHGNet properties with explicit units | Crossref, arXiv, optional configured MCP | applicability limits, out-of-domain chemistry, magnetic and charge-state caveats |
| `relaxation_validation` | separate `/v1/relax` payloads, optimizer convergence, and strict geometry gates | Crossref, arXiv, optional configured MCP | transformations, mechanical/dynamical instability, pressure/temperature, phonons |
| `dft_handoff` | `PeriodicDFTBackend` input/output contract | Crossref, arXiv, optional configured MCP | reference phases, magnetic order, functional/U, pseudopotential and convergence review |

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

# Optional MCP evidence source; set URL and tool together.
export MATERIAL_RAG_MCP_URL="https://YOUR-MCP-SERVER/mcp"
export MATERIAL_RAG_MCP_TOOL="search_material_evidence"
export MATERIAL_RAG_MCP_TOKEN=""              # runtime secret when required
~~~

The MCP endpoint and tool name are administrator configuration. They are not accepted from the discovery prompt, RAG-model output, stage observations, or an MCP response. HTTPS is required outside an explicitly opted-in loopback development endpoint.

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

The command prints a strict `ValidationEvidenceReport` and persists both the report and any source-grounded bundle under `validation-evidence/<stage>/`.

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
- one or more unavailable sources with some records: `partial`
- every requested source successful with records: `completed`

Every report keeps source statuses and hashes, sets `scientific_role` to `search_and_validation_context_only`, and fixes `property_score_created` to `false`. Requests reject secret-like observation keys. The T4 notebook separately scans all hidden credentials against every exported artifact before creating the result archive.

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
