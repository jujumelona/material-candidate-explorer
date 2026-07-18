---
name: material-candidate-validation
description: Route material-candidate checks through the repository's five stage-specific evidence paths and their authoritative scientific validators. Use when Codex evaluates generated crystals, investigates novelty or model disagreement, prepares relaxation or DFT handoffs, configures material RAG/MCP, audits a T4 discovery run, or changes validation workflow documentation and tests.
---

# Material Candidate Validation

Use literature RAG and the administrator-configured MCP tool for each stage as bounded evidence context. Keep numerical and structural decisions with the runtime validator for the current stage.

## Run the workflow

1. Inspect the candidate artifacts, goal, current stage, source statuses, units, provenance, and missing values. Never infer a successful validator call from its name appearing in a route.
2. Select exactly one stage from the table below. Do not collapse several scientific questions into one generic RAG request.
3. Build a strict `ValidationEvidenceRequest`. Include chemical system, reduced compositions, candidate references, non-secret observations, and a narrow focus. Never place API keys, tokens, credentials, or private candidate data in observations.
4. Run `discovery-os validation-evidence --request <request.json> --goal <goal.json> --artifacts <run-dir>` when evidence retrieval is useful. Preserve the report, bundle, source statuses, hashes, conflicts, and negative/null findings.
5. Run the authoritative validator separately and preserve its raw result and provenance. Treat `skipped`, failed, absent, or unconfigured validators as unknown, never as passing.
6. Reconcile evidence with validator output. Use RAG to identify caveats, follow-up tests, and bounded generation priors; never turn record counts, citations, summaries, or MCP prose into energies, forces, hull values, novelty booleans, convergence flags, Pareto utilities, or DFT results.
7. Keep only source-closed `generation_prior` branches eligible for `FusionDecisionContext`. Do not allow later-stage evidence to steer generation.

## Route by stage

| Stage | Stage MCP tool variable | Retrieve with RAG and configured MCP | Runtime authority |
|---|---|---|---|
| `generation_prior` | `MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR` | Reported phases, failed synthesis, composition ranges, stability constraints | MatterGen-supported condition allowlist plus deterministic Evidence Fusion controller |
| `identity_novelty` | `MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY` | Reported phases, crystallographic aliases, scoped database context | pymatgen `StructureMatcher`; configured Materials Project `find_structure` for external identity lookup |
| `mlip_disagreement` | `MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT` | Model-domain limits, magnetic/charge-state caveats, relevant published calculations | Separate MatterSim and CHGNet sidecar results with explicit units and unit-normalized cross-model disagreement |
| `relaxation_validation` | `MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION` | Phase transformations, instability, pressure/temperature, relaxation and phonon context | Separate MatterSim and CHGNet `/v1/relax` payloads, optimizer convergence, and strict periodic geometry gates |
| `dft_handoff` | `MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF` | Reference phases, magnetic order, functional/U, pseudopotential and convergence considerations | Actual `PeriodicDFTBackend` execution and its recorded inputs/outputs; require explicit pseudopotential and convergence review |

OpenAlex is appropriate for `generation_prior` and `identity_novelty`; use Crossref and arXiv for all five routes. Use the configured MCP tool only as another structured evidence provider. A literature provider or MCP response is never the runtime authority.

## Enforce MCP boundaries

- Read the endpoint from `MATERIAL_RAG_MCP_URL`. Select the stage-specific tool variable in the table first, then use `MATERIAL_RAG_MCP_TOOL` only as its administrator-configured fallback. Read the token only from the runtime environment.
- Never let a discovery prompt, planner, model output, observation, or MCP response select or replace the endpoint or tool.
- Require the server's bounded `tools/list` catalog to advertise the selected tool exactly once. Verify its object `inputSchema` declares `query`, `max_results`, `from_date`, and `to_date`; when `outputSchema` is published, require a `records` array. Validate structured output and record fields at runtime even when `outputSchema` is absent.
- Require structured records with stable source identifiers and titles. Reject unstructured prose as evidence.
- Preserve `configured-tool-only`; require HTTPS except for explicitly opted-in loopback development.
- Record missing MCP configuration as skipped and continue other providers. Record retrieval failure or no grounded records as unknown.

## Preserve scientific integrity

- Keep `property_score_created` fixed to `false` and `scientific_role` fixed to `search_and_validation_context_only`.
- Treat absence from literature or Materials Project as unknown, not novel.
- Compare energies only within a reduced composition; never rank different stoichiometries by raw MLIP total energy.
- Require explicit unit normalization before comparing MatterSim and CHGNet.
- Separate execution success, optimizer convergence, structural validity, and scientific acceptance.
- Keep CIF/POSCAR/Quantum ESPRESSO input generation distinct from an executed DFT result. Never claim a DFT value from a prepared input package.
- Escalate missing stress, failed relaxation, large model disagreement, structural collapse, or incomplete provenance instead of silently scoring it.

## Verify changes

Read `docs/STAGE_VALIDATION_EVIDENCE.md` before changing the route contract. Run the focused validation-evidence tests and the notebook contract tests affected by the change, then run the repository test suite in proportion to risk. Scan exported artifacts for secrets and verify that every private or locally excluded path remains untracked before publishing.
