# Application-driven material decisions

`discovery-os material-recommend` is the natural-language front end for
questions such as:

- “Which materials should be used in each part of a semiconductor device?”
- “Which cathode and anode families fit a high-power sodium-ion cell?”
- “Which solid electrolyte and interface coating should be investigated?”
- “Which active phase and support fit CO2 electroreduction?”
- “Which n-type and p-type thermoelectric legs fit 500–800 K service?”
- “Which alloy family fits a hot, chloride-bearing load-bearing component?”

The command does not ask a language model to invent a material and a score. It
compiles the question into a code-owned selection problem:

```text
question
  -> field hypothesis with literal input evidence
  -> component/function roles
  -> required conditions, criteria, units, and validator allowlists
  -> five bounded evidence questions per role
  -> incumbent/emerging retrieval seeds plus supplied/generated candidates
  -> condition- and unit-closed observations
  -> hard gates and robust Pareto fronts inside one role/condition group
  -> optional operator-weighted pool-relative score
  -> reasons, citations, unknowns, conflicts, and next validators
```

Unlike roles are never cross-ranked. A gate dielectric is not “better” than a
channel, a positive electrode is not ranked against a negative electrode, and
an n-type thermoelectric leg is not ranked against a p-type leg.

## Implemented role registry

The existing 12 material fields remain unchanged. A separate application-role
registry decomposes them as follows:

| Field | Application roles |
|---|---|
| General inorganic | stable bulk phase, thermal management, optical window, refractory component, electrical insulator |
| Battery electrode | positive-electrode active material, negative-electrode active material |
| Solid electrolyte | bulk separator, interface buffer/coating |
| Superconductor | high-field magnet conductor, power conductor, RF resonator, Josephson device |
| Heterogeneous catalyst | active phase/site, support/interface |
| Semiconductor | logic channel, power switch, transparent electrode, source/drain contact, interconnect, gate dielectric, diffusion barrier, thermal stack, detector, emitter, modulator, substrate/buffer, passivation |
| Photovoltaic | single-junction absorber, tandem top absorber, transport/contact layer |
| Thermoelectric | n-type leg, p-type leg, contact/interconnect |
| Magnetic material | permanent magnet, soft core, spintronic layer, magnetocaloric refrigerant |
| Ferroelectric/piezoelectric | memory, actuator, sensor/transducer, dielectric energy storage |
| Structural alloy | lightweight load-bearing, high-temperature, corrosion-resistant, wear-resistant |
| Porous framework | gas storage, gas separation, carbon capture, atmospheric water harvesting |

Each role owns:

- English and Korean aliases;
- the minimum application, operating, geometry, processing, and service
  context;
- property names, exact units, objective direction, and required conditions;
- named numerical/experimental validator IDs;
- bulk/film/interface/device representation limits;
- incumbent and emerging material-family retrieval seeds;
- failure modes and a conservative claim boundary; and
- five ordered evidence tasks.

Retrieval seeds are not recommendations. They have no performance score and
exist so retrieval covers incumbent baselines, emerging alternatives, and
negative evidence rather than only generating unfamiliar formulas.

## Run a scenario map

No API key is required for deterministic routing and the unscored seed
portfolio:

```bash
discovery-os material-recommend \
  --prompt "Which materials fit the different parts of a semiconductor device?" \
  --field AUTO \
  --main-model-routing off \
  --artifacts runs/application-map
```

A broad question returns several role portfolios. Each seed has:

- `pool_relative_decision_score: null`;
- `evidence_completeness_score: 0`;
- explicit unknown criteria;
- failure modes and claim boundaries; and
- the validators needed before ranking.

That is intentional. A missing result is `UNKNOWN`, not zero and not a pass.

For a focused question, bind one role and its target conditions:

```bash
discovery-os material-recommend \
  --prompt "Compare gate dielectric candidates for this scaled nFET stack" \
  --field semiconductor \
  --role gate_dielectric \
  --context-json '{
    "channel": "strained Si",
    "target_eot": 0.7,
    "gate_architecture": "GAA",
    "electric_field": 5.0,
    "temperature": 398,
    "max_leakage": 1e-7,
    "process_temperature": 723,
    "lifetime_requirement": "10 years"
  }' \
  --candidates candidates.json \
  --observations observations.json \
  --preferences preferences.json \
  --artifacts runs/gate-dielectric
```

`--require-condition-complete` changes missing role context into a typed
clarification state. Without it, the system may still return several explicit
reference scenarios, but it groups and ranks them only when their exact
conditions match.

## Google Colab

Open
[MATERIAL_APPLICATION_RECOMMENDER_T4.ipynb](https://colab.research.google.com/github/jujumelona/material-candidate-explorer/blob/main/MATERIAL_APPLICATION_RECOMMENDER_T4.ipynb)
for the same workflow without a local installation. Its first cell exposes the
question, field mode, optional role IDs, context JSON, strict candidate,
observation, and preference JSON, RAG controls, and administrator-owned MCP
endpoint/tool settings. API keys and bearer tokens are requested with hidden
runtime prompts and are not written into the notebook or result archive.

The notebook's application-decision cell creates typed role portfolios and,
when enabled, attaches stage-bounded evidence. That decision run does not
execute a generator, specialist property model, relaxation, DFT, or experiment.
It writes
`application-to-crystal-t4-handoff.json` as a manual handoff receipt with one
status per role. The operator
must choose exactly one role and resolve its required context and chemical
system before copying the recorded input values into
[MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb](https://colab.research.google.com/github/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb)
for configured 8-32-candidate MatterGen, MatterSim, CHGNet,
duplicate-removal, multi-round Pareto, external identity, and non-executed
DFT-input preparation. No notebook automatically converts a multi-role
portfolio into one crystal-generation target.

## Explicit bulk-crystal execution bridge

`material-recommend` and `MaterialDecisionRunner` never execute generation or
specialist validation. A separate opt-in command invokes the real
`FusionSearchRunner` only after the operator supplies one selected
`bulk_crystal` role and complete typed runtime inputs:

```bash
export MATTERGEN_API_URL="https://YOUR-MATTERGEN-SIDECAR"
export MATTERSIM_API_URL="https://YOUR-MATTERSIM-SIDECAR"
export CHGNET_API_URL="https://YOUR-CHGNET-SIDECAR"
# Optional secrets use MATTERGEN_API_TOKEN, MATTERSIM_API_TOKEN,
# and CHGNET_API_TOKEN. Never put them in the JSON files.

discovery-os material-fusion-search \
  --brief runs/application/MDRUN-.../application-brief.json \
  --role stable_bulk_phase \
  --search-id stable-bulk-run-001 \
  --goal goal.json \
  --parent parent-candidate.json \
  --run-config run-config.json \
  --generator mattergen \
  --rounds 4 \
  --frontier-width 1 \
  --no-control-sweep \
  --max-generation-calls 32 \
  --max-generated-candidates 16 \
  --expert mattersim \
  --expert chgnet \
  --required-evaluator mattersim \
  --required-evaluator chgnet \
  --artifacts runs/application-fusion
```

The bridge rejects the request before sidecar execution unless:

- the brief contains the explicitly named role and its scope includes
  `bulk_crystal`;
- material-field routing is unambiguous;
- the parent is an immutable CIF, mmCIF, or POSCAR candidate with a current
  content hash;
- the goal and workspace-ON run configuration cite that parent and each other;
- goal objectives use only the MLIP diagnostics `energy_per_atom` in eV/atom
  and/or `max_force` in eV/angstrom, with `minimize` direction and the
  code-owned broad runtime validation profile;
- at least three rounds and a global 8-32 generated-candidate budget can reach
  the third adaptive round;
- the explicit panel contains at least two configured non-dummy experts; and
- the generator and expert sidecars are configured and return valid payloads.

The command does not turn a retrieval seed into a structure, does not pass
application RAG to a generator or runtime validator, and does not create an
application-property score or scientific claim. It persists the nested Fusion
search report, candidate lineage, budget usage, and validation-handoff
candidates. Missing sidecars, invalid provenance, runtime failure, or an
exhausted/incomplete search remains a failure or partial diagnostic result.

The application Colab exposes the same command behind
`RUN_SINGLE_ROLE_BULK_SEARCH`. It additionally requires exactly one selected
role, complete role context, the three API URLs, and uploaded goal, parent, and
run-configuration JSON. Broad or multi-role portfolios stay on the manual
handoff path.

## Optional main-AI routing

`--main-model-routing auto` uses the existing trusted OpenAI-compatible
reasoning endpoint when configured:

```bash
export MATERIAL_FIELD_MODEL_API_URL="https://YOUR-ENDPOINT/v1"
export MATERIAL_FIELD_MODEL_NAME="YOUR-MODEL"
export MATERIAL_FIELD_MODEL_API_KEY=""
```

If the dedicated pair is absent, the complete `RAG_MODEL_*` pair is reused.
`required` fails when no endpoint is configured. `off` is deterministic.

The model can propose only:

- question kind;
- allowlisted role IDs;
- an application subtype;
- objective IDs already declared by those roles;
- context copied exactly from the supplied object or a role-declared scalar
  condition whose value appears literally in the question;
- confidence and literal input evidence spans; and
- a clarification question.

Code rejects confidence below 0.70, an unregistered subtype, a stale model run,
a hallucinated role/objective, a non-literal or non-meaningful evidence span,
an unregistered context key/value, or any attempt to select an endpoint, MCP
tool, validator, score, or pass/fail result.

## Stage- and role-specific RAG/MCP

Use `--run-rag` to execute the source-grounded application evidence prompt:

```bash
export LITERATURE_CONTACT_EMAIL="researcher@example.org"
export OPENALEX_API_KEY=""
export RAG_MODEL_API_URL="https://YOUR-ENDPOINT/v1"
export RAG_MODEL_NAME="YOUR-MODEL"
export RAG_MODEL_API_KEY=""

discovery-os material-recommend \
  --prompt "Find cathode and anode options for a fast-charge sodium-ion cell" \
  --field battery_electrode \
  --run-rag \
  --artifacts runs/sodium-cell
```

Application retrieval executes five separate requests. Each request contains
exactly one validation stage and the corresponding task for every selected
role:

1. `generation_prior`: requirements, metrics, conditions, and known priors;
2. `identity_novelty`: incumbents, aliases, phases/stacks, and external
   identity evidence;
3. `mlip_disagreement`: model applicability and disagreement limitations;
4. `relaxation_validation`: instability, degradation, transformations, and
   negative evidence; and
5. `dft_handoff`: authoritative calculations, measurements, controls, and
   reproducibility requirements.

Leave `--rag-source` unset for the code-owned stage policy: Crossref and arXiv
are requested at all five stages, OpenAlex only at `generation_prior` and
`identity_novelty`, and a configured MCP tool is requested through the
administrator-owned precedence below. An explicit `--rag-source` selection
applies to all five stages, so only sources allowed by every selected stage are
valid; it cannot be used to force OpenAlex into a later stage.

The administrator-selected MCP tool precedence is: the matching stage-specific
variable, then `MATERIAL_APPLICATION_RAG_MCP_TOOL`, then the generic
`MATERIAL_RAG_MCP_TOOL`. Prompts and model output cannot choose the endpoint or
tool:

```text
MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR
MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY
MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT
MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION
MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF
```

OpenAlex is permitted only for `generation_prior` and `identity_novelty`;
later stages use Crossref and arXiv. Each stage receives its field profile's
own MCP capability subset. The result archive stores all five source bundles,
their source failures, and a citation-only composite bundle. The composite
discards generic generator branches and cannot create a property score.

Every role task declares read-only MCP *capabilities*. A capability is a
required evidence function, not a claim that an upstream project ships a
same-named MCP server.

RAG records should retain at least:

```text
material/phase_or_stack, component_role, property, value, unit,
complete_conditions, film_or_device_geometry, measured_or_calculated,
method, sample/process, uncertainty, negative_or_null_result,
DOI/stable_source_id, exact_support_span
```

Only an exact literal candidate mention can create a citation link. It cannot
create a property observation. Literature-derived numbers require a separate
trusted structured-database validator before they are eligible for scoring.

## Candidate and observation inputs

`candidates.json` is an array of `MaterialApplicationCandidate` objects:

```json
[
  {
    "schema_version": "1.0",
    "candidate_id": "GATE-HFO2",
    "role_id": "gate_dielectric",
    "material_or_stack": "Si / interfacial SiO2 / HfO2 / metal gate",
    "phase_or_stack": "deposited gate stack",
    "origin": "user_supplied",
    "candidate_ref": null,
    "structure_or_record_id": null,
    "evidence_claim_ids": [],
    "research_reference_ids": [],
    "provenance_id": "operator-gate-stack-v1",
    "triage_priority_score": null,
    "triage_score_semantics": "search-priority-only-not-application-fitness",
    "model_disagreement": "not_applicable",
    "external_identity_status": "not_checked"
  }
]
```

`observations.json` is an array of `MaterialApplicationObservation` objects.
Every successful row needs:

- one candidate and role;
- one role-declared property and validator;
- exact unit;
- every required condition;
- value and optional lower/upper uncertainty bounds;
- method and sample/model scope;
- numerical, experimental, or separately trusted structured-database
  authority;
- provenance ID; and
- raw-artifact SHA-256.

`literature_or_mcp_derived` is fixed to `false`. Wrong units, missing
conditions, a target-condition mismatch, or an unapproved validator is
`incomparable`. Different successful values under the same conditions are
`conflicting`; they are preserved and never averaged.

Inspect the exact JSON Schemas:

```bash
discovery-os schema MaterialApplicationCandidate
discovery-os schema MaterialApplicationObservation
discovery-os schema MaterialDecisionPreference
discovery-os schema MaterialRecommendationReport
```

## Score semantics

The report always exposes separate vectors:

```text
hard_gate_status
performance_vector
reliability_vector
integration_vector
resource_safety_vector
evidence_completeness_score
evidence_uncertainty_status
model_disagreement
pareto_front
pool_relative_decision_score_or_null
```

`evidence_completeness_score` is coverage, not performance. It is the
percentage of required criteria with condition-complete named-validator
results.

Hard gates use the full uncertainty interval:

- interval wholly satisfies the constraint: `pass`;
- interval wholly violates it: `fail`;
- interval crosses the boundary or evidence is missing: `unknown`.

Pareto dominance is conservative. Candidate A robustly dominates B only when
A's worst uncertainty bound is no worse than B's best bound on every role
criterion and strictly better on at least one.

The optional scalar score requires:

- explicit operator or source-closed non-zero weights;
- at least two eligible candidates;
- all weighted criteria available;
- resolved directions/targets; and
- the same role, units, and exact condition signature.

It min-max normalizes conservative utility inside that candidate pool and is
therefore labelled
`operator-weighted-pool-relative-decision-support-or-null`. It is not a
probability, scientific truth, discovery claim, or a score comparable to
another role or condition group. Citation counts, RAG record counts, search
priority, main-model confidence, and unknown values never enter it.

## Outputs

Every run writes:

```text
<artifacts>/<run_id>/
  application-brief.json
  application-rag-bundle.json       # citation-only composite, when available
  application-rag/
    generation_prior.json           # one file per completed stage
    identity_novelty.json
    mlip_disagreement.json
    relaxation_validation.json
    dft_handoff.json
  material-recommendation.json
  material-recommendation.csv
  material-recommendation.md
  material-decision-run.json        # includes all stage receipts and failures
```

The Colab notebook additionally writes
`application-to-crystal-t4-handoff.json`. It is a manual transfer receipt, not
a generated structure, queued job, specialist result, or automatic loop.

Every candidate row explains:

- why it was retained;
- why it is not top-ranked;
- its main trade-offs;
- exact observation IDs;
- source-closed citations;
- unknown, conflicting, or unquantified evidence;
- model disagreement and scoped identity state; and
- the next named calculation or experiment.

## Verification boundary

This document describes executable schemas, routing, retrieval, ranking, and
artifact behavior only. Research citations are kept separately in
[Research foundations](RESEARCH_FOUNDATIONS.md) and are not listed as
implemented features. A paper, database record, model card, or benchmark does
not establish that this repository ran the corresponding experiment or that a
candidate passed it.
