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
| `generation_prior` | `MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR` | Reported phases, failed synthesis, composition ranges, stability constraints | Fixed official MatterGen checkpoint allowlist plus deterministic Evidence Fusion controller; a custom allowlist is operator attestation, not training-config verification |
| `identity_novelty` | `MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY` | Reported phases, crystallographic aliases, scoped database context | Non-symmetrized source-Niggli identity plus strict unscaled pymatgen `StructureMatcher`; Materials Project `find_structure` is only a prefilter whose returned structures require local strict recheck |
| `mlip_disagreement` | `MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT` | Model-domain limits, magnetic/charge-state caveats, relevant published calculations | Separate MatterSim and CHGNet sidecar results with explicit units; cross-model energy evidence requires aligned same-composition relative energies, while raw absolute differences are audit-only |
| `relaxation_validation` | `MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION` | Phase transformations, instability, pressure/temperature, relaxation and phonon context | Separate MatterSim and CHGNet `/v1/relax` payloads, optimizer convergence, and strict periodic geometry gates |
| `dft_handoff` | `MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF` | Reference phases, magnetic order, functional/U, pseudopotential and convergence considerations | Actual executing `PeriodicDFTBackend`; completed results require input-manifest, method-policy, immutable output and convergence evidence, with reference-set/phonon fields when applicable |

OpenAlex is appropriate for `generation_prior` and `identity_novelty`; use Crossref and arXiv for all five routes. Use the configured MCP tool only as another structured evidence provider. A literature provider or MCP response is never the runtime authority.

RAG retrieves and closes scholarly evidence to records. The configured MCP tool is an evidence-provider interface, not an action validator. Do not use this evidence route to mutate a candidate, run a sidecar, relax a structure, submit DFT, or write an external database; invoke and attest the named runtime validator separately.

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
- Keep the selector branch named `novelty` separate from scientific novelty: it is property-space diversity only. Structural/database novelty comes only from the staged assessor and remains scoped to recorded providers and snapshots.
- Give `unknown`, skipped, failed, or partial external novelty no ranking credit. A DFT portfolio may reserve at most one slot for an otherwise eligible strict external scoped-no-match candidate; record that as a bounded prioritization rule, never a novelty proof.
- Preserve the three duplicate boundaries: MatterGen strict tolerance-aware within-call and search-session checks happen before expert evaluation; the coordinator's exact attested-hash check is only a post-evaluation fallback; a final notebook grouping is a separate audit. Do not describe one boundary as if it performs all three.
- Compare energies only within a reduced composition; never rank different stoichiometries by raw MLIP total energy.
- Require explicit unit normalization before comparing MatterSim and CHGNet.
- For the trusted default MatterSim/CHGNet panel, require bootstrap and launcher provenance to match the trusted manifest's exact HTTPS artifact size and SHA-256. The current defaults are MatterSim 5M `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5` and CHGNet 0.3.0 `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`. Reject `managed-unattested:*`, changed bytes, and missing attestation instead of treating them as comparable expert evidence.
- Separate execution success, optimizer convergence, structural validity, and scientific acceptance.
- Keep CIF/POSCAR/Quantum ESPRESSO input generation distinct from an executed DFT result. Never claim a DFT value from a prepared input package. A completed result must retain its input manifest, immutable output and convergence artifacts, method policy, explicit convergence, and any required reference-set or phonon provenance.
- Escalate missing stress, failed relaxation, large model disagreement, structural collapse, or incomplete provenance instead of silently scoring it.

## Verify changes

Read `docs/STAGE_VALIDATION_EVIDENCE.md` before changing the route contract. Run the focused validation-evidence tests and the notebook contract tests affected by the change, then run the repository test suite in proportion to risk. Scan exported artifacts for secrets and verify that every private or locally excluded path remains untracked before publishing.
