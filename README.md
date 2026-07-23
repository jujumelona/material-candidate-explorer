# Material Candidate Explorer

[![Run T4 Discovery in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb) [![View T4 Notebook](https://img.shields.io/badge/T4%20Discovery-GitHub-181717?logo=github)](https://github.com/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb) [![Materials validation](https://github.com/jujumelona/material-candidate-explorer/actions/workflows/materials-validation.yml/badge.svg)](https://github.com/jujumelona/material-candidate-explorer/actions/workflows/materials-validation.yml)

Material Candidate Explorer is a model-agnostic orchestration engine for exploring and prioritizing material candidates across chemistry, crystal materials, superconductors, batteries, catalysts, polymers, and medicinal chemistry.

It connects isolated expert and generator sidecars through strict JSON contracts. Expert properties, units, status, and provenance are preserved; incompatible embeddings are never averaged into a fake scientific score.

> Scientific boundary: outputs are computational leads, not proof of a new material. Synthesis, identity and purity checks, independent measurements, controls, repeated experiments, and reproducibility are required before making scientific or medical claims.

The Python package remains discovery_os and the CLI remains discovery-os for compatibility.

## Architecture

~~~text
goal + parent candidate
        |
        v
expert sidecars (MatterSim, CHGNet, UMA, Uni-Mol, Chemprop, ...)
        |
        v
Evidence Store: original properties, units, status, provenance, cache
        |
        v
local EvidenceDrivenFusionBackend or optional remote FusionBackend
        |
        v
MatterGen / REINVENT generator sidecar
        |
        v
candidate batch -> expert re-evaluation -> branch selection
~~~

The local Fusion backend never averages expert tensors. Its fixed eight-value latent is search-control state only:

~~~text
[cycle, successful experts, non-successful experts,
 worst objective utility, distinct-expert disagreement,
 improvement flag, structural collapse rate, guidance alpha]
~~~

Actual candidate selection is performed by the evidence store, deterministic exploration selector, adaptive scheduler, and Pareto branches.

The scientific choices and claim boundaries are mapped to primary sources in [Research foundations](docs/RESEARCH_FOUNDATIONS.md). Model or benchmark publications justify a workflow choice; they do not validate a newly generated candidate.

## Implemented behavior

- Candidate batches and paired OFF/ON runs support candidate_count from 1 to 1024.
- Fusion search keeps stability, target-property, expert-disagreement, Pareto, and a legacy `novelty` branch whose implemented score is property-space diversity. That branch is not a structural or database novelty result.
- Every generated candidate is evaluated again by the configured experts.
- MatterGen accepts only the fixed allowlist for a named official checkpoint. A custom checkpoint is unconditional unless an operator supplies an explicit condition allowlist; that declaration is recorded as operator attestation and is not automatic inspection or proof of how the custom weights were trained. The adapter records the actual classifier-free guidance factor, requested/applied/ignored controls, model-envelope rejections, checkpoint provenance, and generation funnel.
- MatterGen retains the direct generated CIF as the candidate. Hard identity uses a source-derived Niggli representation without symmetry refinement; the symmetry-standardized structure is retained only as prototype context. Only a strict, unscaled crystallographic match is hard-deduplicated, so a volume-scaled same-prototype candidate is retained. The sidecar removes strict duplicates within a call and, when `search_session_id` is present, across calls before expert evaluation while requesting replacements. The coordinator keeps a post-evaluation exact attested-identity-hash fallback for adapters that emit the compatible raw-identity receipt but do not perform session-aware replacement. Exact-file hashes remain separate from both identity receipts.
- MatterSim and CHGNet expose a separate `/v1/relax` operation, so model execution, optimizer convergence, and the strict geometry gate are recorded independently.
- The trusted manifest downloads the default MatterSim 5M and CHGNet 0.3.0 files from immutable upstream source revisions, checks their declared byte sizes and project-pinned SHA-256 digests, and writes an attestation marker. The launchers rehash the files before binding them; the defaults are no longer upstream-managed or `managed-unattested`.
- Staged structural novelty assessment is separate from the property-diversity branch. It distinguishes current-batch matches, project-history matches, external strict structure matches, scoped no-match, and `unknown`. A Materials Project `find_structure` result is only a similarity prefilter until the returned structure passes the local strict unscaled matcher. With several configured providers, external `no_match` requires all of them to return scoped no-match; one unresolved provider keeps the aggregate `unknown`.
- Composition-scoped Pareto fronts use NSGA-II crowding to preserve objective boundaries. The top-1-to-5 DFT portfolio covers distinct reduced compositions before filling remaining slots and may reserve at most one slot for an eligible strict external scoped-no-match candidate. `unknown` receives no such credit, and even scoped no-match is not proof of novelty.
- Optional split-conformal reliability applies only to an exact model/DFT/chemistry calibration scope. Missing or out-of-domain calibration returns unknown, exposes no coverage claim, and escalates to DFT.
- The portable DFT handoff writes reviewable CIF/POSCAR/Quantum ESPRESSO input packages with an auditable reciprocal-space starting grid and unresolved pseudopotential-specific cutoffs; it does not bundle pseudopotentials or claim a calculation ran. A separately executing backend can return `completed` only with the input-manifest hash, immutable output and convergence artifacts, a method-policy hash, and explicit convergence. Formation or hull values additionally require a reference-set hash, and phonon conclusions require the recorded q mesh, minimum frequency, tolerance, and consistent imaginary-mode classification.
- Goal-hashed feature caching prevents duplicate expert calculations.
- CIF, SMILES, sequences, original expert payloads, generator provenance, latent states, scheduler controls, and search history are persisted under the artifact root.
- If FUSION_API_URL is unset, the deterministic local EvidenceDrivenFusionBackend is selected automatically.
- Remote Fusion requests keep the legacy strict payload by default. Extended local search context requires FUSION_SEND_EXTENDED_REQUEST_CONTEXT=1.

## Interactive notebook

- [View the T4 material-discovery notebook on GitHub](https://github.com/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb)
- [Open the T4 material-discovery notebook in Google Colab](https://colab.research.google.com/github/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb)
- [View the legacy adaptive-loop notebook](https://github.com/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_EXPLORER_V11_PROMPT_RAG_REAL_GENERATION.ipynb)

`MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb` is the recommended material-candidate workflow. It runs a budgeted, multi-round `fusion-search`: the MatterGen sidecar performs raw-geometry, tolerance-aware within-call and search-session duplicate rejection before MatterSim and CHGNet evaluation; evaluated candidates are then inserted into persistent evidence and branch pools and used to schedule the next round. The coordinator's attested-hash check is a defensive post-evaluation fallback, not a replacement for `StructureMatcher`. Global generator-call and generated-candidate limits cover every round, branch, frontier, and control variant. A MAXIMUM run does not fabricate candidates or silently accept a shortfall: rejection, sidecar failure, deduplication, or frontier exhaustion is preserved in the funnel and the notebook fails closed if it cannot establish the requested 8-32 crystallographically unique candidates. It then runs explicit MLIP relaxations, composition-scoped Pareto ranking with crowding, staged structural novelty checks, and research-only DFT input preparation for the top 1-5 candidates.

~~~text
bounded multi-round MatterGen search -> pre-expert strict duplicate rejection
-> scaled-prototype relation retained separately
-> geometry gate -> separate MatterSim + CHGNet screening and relaxation
-> composition-scoped Pareto fronts + NSGA-II crowding
-> property-space diversity branch + separate staged structural novelty lookup
-> composition-diverse top 1-5 research-only DFT input packages
~~~

At five intermediate boundaries the notebook also runs source-grounded validation context:

| Stage ID | Numerical or structural authority | RAG/MCP role | Dedicated MCP tool variable |
|---|---|---|---|
| `generation_prior` | MatterGen-supported conditions only | reported phases, failed synthesis, and bounded condition priors | `MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR` |
| `identity_novelty` | pymatgen `StructureMatcher` plus optional Materials Project structure lookup | crystallographic reports and aliases | `MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY` |
| `mlip_disagreement` | separate MatterSim and CHGNet outputs with unit normalization | applicability limits and possible disagreement causes | `MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT` |
| `relaxation_validation` | separate MatterSim and CHGNet `/v1/relax` results and strict geometry gates | phase transformations, instability, pressure/temperature, and phonon context | `MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION` |
| `dft_handoff` | selected periodic DFT backend; a prepared input package is not an executed calculation | reference phases, magnetism, functional/U, pseudopotential, convergence, and phonon review | `MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF` |

Crossref, arXiv, and OpenAlex are bounded scholarly metadata sources. Optional MCP Streamable HTTP tools may be selected only from the stage variables above, with `MATERIAL_RAG_MCP_TOOL` as the administrator-configured generic fallback. The client verifies the selected tool through bounded `tools/list`, checks its input/output contract, and rejects unstructured evidence. A prompt or model output cannot select an endpoint or tool. Missing providers, missing credentials, empty retrieval, schema mismatch, or errors are recorded as `partial`, `skipped`, or `unknown`; they never become a candidate property score or a validator pass.

RAG and MCP have distinct jobs here. RAG retrieves and closes scholarly evidence into citable records. The optional MCP route is only a configured structured evidence-provider interface; it is not permission to execute a model, database write, relaxation, or DFT job. Runtime sidecars, the strict structure matcher, external structure clients, and an explicitly configured periodic DFT backend remain the action validators.

Each route persists a typed evidence handoff naming its consumer, payload schema, and still-required runtime validator. Only a source-grounded `generation_prior` handoff may guide a Fusion decision; identity, MLIP, relaxation, and DFT handoffs cannot steer generation. The evidence router never marks its listed validators as executed.

The search selector's legacy branch ID `novelty` means nearest-neighbor diversity in normalized expert-property space only. It neither queries a structure database nor emits a novelty claim. Structural/database novelty is produced only by `StagedNoveltyAssessor`, with its batch, project-history, external-provider, query, matcher, timestamp, and database-version provenance.

The notebook form exposes the non-secret RAG/MCP settings. OpenAlex requires its free API key when that source is used; pressing Enter skips OpenAlex while the other sources continue. OpenAlex, optional RAG-model, and optional MCP credentials appear as hidden `getpass` prompts in the Setup cell and are scanned out of the archive. Leave the paired RAG or MCP endpoint fields blank to skip that integration.

The notebook never merges official CIF output with an EXTXYZ conversion path. It reports `requested_samples`, `raw_model_structures`, `parsed_structures`, `exact_file_unique`, `crystallographically_unique`, `geometry_valid`, `mlip_evaluated`, `relaxation_converged`, and `ranked_candidates` separately. A blank Materials Project API key is allowed; external novelty then remains `unknown`, never `novel=true`.

The legacy V11 notebook remains available for the older adaptive-loop interface. Blank optional API fields are skipped.

Linux and Colab use the POSIX launchers; PowerShell is not required:

~~~bash
python -m pip install -e ".[dev,materials]"
./bootstrap.sh --profile materials-open --accelerator cuda --include-weights
./start-sidecars.sh --component mattergen --component mattersim --component chgnet
source .discovery/sidecars.env.sh
~~~

Use Python 3.11 or 3.12 on Windows PowerShell:

~~~powershell
py -3.11 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[chem,dev]"
~~~

For the minimum installation without RDKit:

~~~powershell
python -m pip install -e ".[dev]"
~~~

Run the deterministic demo without external model credentials:

~~~powershell
discovery-os demo --goal "Review small-molecule candidates with possible target-protein binding" --domain medicinal_chemistry --max-cycles 1 --output runs/demo
~~~

Inspect profiles and contracts:

~~~powershell
discovery-os profiles
discovery-os schema ValidationPlan
~~~

Run tests:

~~~powershell
pytest
~~~

## Material candidate search with MatterGen

~~~powershell
# 1. Verify the pinned integration manifest
python scripts/bootstrap.py verify-manifest

# 2. Install permitted isolated environments and weights
.\\bootstrap.ps1 -Profile all-open -Accelerator cuda -IncludeWeights -AcceptLicense esm

# 3. Start generator and expert sidecars
.\\start-sidecars.ps1 -Component mattergen,mattersim,chgnet -Backend auto
. .\\.discovery\\wsl\\sidecars.env.ps1

# 4. Inspect configured endpoints and expert availability
discovery-os integrations --profile materials-open
discovery-os experts

# 5. Run a multi-branch ON search
discovery-os fusion-search `
  --search-id material-run-001 `
  --goal goal.json `
  --parent parent.json `
  --run-config run-config.json `
  --generator mattergen `
  --rounds 4 `
  --frontier-width 2 `
  --expert mattersim `
  --expert chgnet `
  --required-evaluator mattersim `
  --required-evaluator chgnet `
  --max-generation-calls 24 `
  --max-generated-candidates 32 `
  --artifacts runs/material-run-001
~~~

goal.json defines objectives and units. parent.json contains the starting candidate, normally as CIF or a chemical formula. run-config.json binds workspace_mode=on, candidate_count, the evaluator panel, generator identity, checkpoint revision, and runtime parameters.

The client reads sidecar URLs from the integration manifest environment names, for example:

~~~powershell
$env:MATTERGEN_API_URL = "http://127.0.0.1:8101"
$env:MATTERSIM_API_URL = "http://127.0.0.1:8110"
$env:CHGNET_API_URL = "http://127.0.0.1:8113"
~~~

Prefer sourcing the environment file generated by the launcher instead of setting ports manually. Loopback HTTP requires the explicit development opt-in generated by the launcher. Production endpoints must use HTTPS and environment or secret-manager tokens.

For the trusted `materials-open` profile, `--include-weights` downloads and verifies these exact public evaluator artifacts:

| Evaluator | Immutable upstream revision | Bytes | SHA-256 |
|---|---|---:|---|
| MatterSim 5M | `40a1eb8f1189a53af310957b4f2c5dfbfe68d647` | `91,176,875` | `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5` |
| CHGNet 0.3.0 | `8d04abc467630b30cd8349c2fe12eeab10f62171` | `4,863,221` | `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1` |

Bootstrap rejects an unexpected size or digest and records `.artifact.json` beside the verified bytes. The launcher verifies that marker and rehashes the file before setting `MATTERSIM_CHECKPOINT_PATH` or `CHGNET_CHECKPOINT_PATH` and a `sha256:<digest>` weight revision. Do not set the old `managed-unattested:*` values for these defaults.

To add live RAG to the same search, configure any optional scholarly credentials, RAG model, or MCP evidence tool and add these arguments:

~~~text
--rag-prompt "Find experimentally supported Li-O phases and failure modes" \
--rag-from-date 2024-01-01 \
--rag-index .discovery/evidence-index
~~~

The evidence bundle is not converted into a material score. Only source-closed generation-prior evidence may allocate or adapt search branches, while MatterSim and CHGNet remain proxy evaluators. This loop is adaptive screening, not active learning: no generator or expert weights are retrained.

Run one stage route directly from code or automation with a strict request JSON:

~~~bash
export VALIDATION_EVIDENCE_ENABLED=1
export VALIDATION_EVIDENCE_MAX_RESULTS=8
export LITERATURE_CONTACT_EMAIL="researcher@example.org"
export MATERIAL_RAG_MCP_URL="https://YOUR-MCP-SERVER/mcp"
export MATERIAL_RAG_MCP_TOOL="search_material_evidence" # optional generic fallback
export MATERIAL_RAG_MCP_TOOL_GENERATION_PRIOR="search_generation_prior"
export MATERIAL_RAG_MCP_TOOL_IDENTITY_NOVELTY="search_crystal_identity"
export MATERIAL_RAG_MCP_TOOL_MLIP_DISAGREEMENT="search_mlip_limits"
export MATERIAL_RAG_MCP_TOOL_RELAXATION_VALIDATION="search_relaxation_instability"
export MATERIAL_RAG_MCP_TOOL_DFT_HANDOFF="search_periodic_dft_methods"
discovery-os validation-evidence \
  --request validation-stage.json \
  --goal goal.json \
  --artifacts runs/material-run-001
~~~

See [Stage-specific validation evidence](docs/STAGE_VALIDATION_EVIDENCE.md) for request examples, typed handoffs, environment variables, route semantics, and artifact layout. Codex contributors can invoke the repository-local [`$material-candidate-validation`](.codex/skills/material-candidate-validation/SKILL.md) skill to apply the same five-stage authority and MCP boundaries while changing or auditing the project. The skill is procedural guidance, not a runtime validator or MCP server.

## Genomic candidate evaluation with AlphaGenome

AlphaGenome is connected as an intermediate genomic expert, not as a sequence generator:

~~~text
candidate variants / regulatory sequences
        -> AlphaGenome evaluation
        -> expression, splicing, chromatin and contact-map deltas
        -> literature and independent-model cross-checks
        -> uncertainty/disagreement-aware ranking
        -> mutate or redesign the highest-priority candidates
~~~

The optional adapter supports `GENOMIC_VARIANT_DISCOVERY`, `REGULATORY_SEQUENCE_DESIGN`, `SPLICE_VARIANT_ANALYSIS`, `DISEASE_VARIANT_PRIORITIZATION`, and `ALPHAGENOME_EVALUATION_ONLY`. Install it only when needed:

~~~bash
python -m pip install -e ".[genomics,dev]"
export ALPHAGENOME_API_KEY="..."  # runtime secret; never commit this value
~~~

The AlphaGenome adapter is a separate Python/CLI genomic workflow and is not part of the material T4 notebook. AlphaGenome API use is subject to Google's terms and non-commercial restrictions; this project does not make clinical claims or redistribute model weights.

## Fusion and generator controls

The local controller emits only these MatterGen condition names:

~~~text
chemical_system
space_group
dft_mag_density
dft_band_gap
ml_bulk_modulus
hhi_score
energy_above_hull
~~~

It does not invent unsupported lattice or atom-movement instructions. Explicit goal targets take priority. Invalid types, invalid space groups, unknown elements, incompatible units, and unsupported evidence are rejected or left without a numeric target.

For MatterGen, `alpha` maps to classifier-free guidance as `gamma = alpha * guidance_max`; the default `guidance_max=4` makes `alpha=0.5` equivalent to `gamma=2`. Sustained improvement increases guidance for condition-focused exploitation. Stagnation or structural collapse lowers guidance to broaden or stabilize sampling. Higher guidance can improve condition adherence at the cost of diversity or realism, so this direction is not interchangeable with generic temperature semantics. MatterGen v1 applies only supported checkpoint conditions, guidance, seed, and batch controls; it does not expose parent mutation, temperature, mutation strength, or diversity strength, so unsupported values remain requested-but-ignored provenance. REINVENT applies its own supported sampling controls.

The released MatterGen model envelope represented by the adapter is a primitive cell of at most 20 atoms, excluding noble gases, Tc, Pm, and elements with atomic number above 84. Passing this gate is not a stability claim. Official checkpoint names have fixed condition allowlists. An unknown/custom checkpoint is unconditional unless the operator supplies `supported_condition_names`; the adapter validates only that the names are in its known condition vocabulary and records the declaration source. It does not inspect the custom training run or prove that those condition modules were trained and packaged with the weights. Treat the custom declaration as operator attestation and independently review the checkpoint inventory hash and training configuration. See the [MatterGen model card](https://github.com/microsoft/mattergen/blob/main/MODEL_CARD.md) and [Research foundations](docs/RESEARCH_FOUNDATIONS.md).

## Outputs and validation handoff

Each artifact root contains candidate records, CIF/SMILES/sequence representations, original expert payloads, feature cache entries, latent states, generator provenance, scheduler history, branch pools, and the final diagnostic report.

validation_handoff_candidate_refs is a bounded shortlist for a separately configured high-cost validator. It prioritizes Pareto candidates, then stability candidates, and removes exact scientific-representation duplicates. This coordinator shortlist check is not a tolerance-aware crystallographic identity decision; the MatterGen sidecar and T4 structure stages persist their separate matcher receipts. The handoff does not mean that DFT, relaxation, phonons, experiments, or synthesis have run.

MatterSim and UMA expose total energy (eV) and `energy_per_atom` (eV/atom), but raw MLIP energy is compared only among aligned candidates with the same reduced composition. Cross-stoichiometry ordering requires reference-consistent formation energies; `energy_per_atom` is not `energy_above_hull`. Convex-hull stability requires relaxed structures, reference phases, a declared correction/mixing policy, and an executed validated hull workflow.

MatterSim-CHGNet agreement is not automatically calibrated uncertainty. `discovery_os.mlip_reliability` can consume a split-conformal artifact bound to exact weight revisions, DFT method/reference hashes, units, held-out coverage metadata, and a declared chemistry/exchangeability scope. Any mismatch is `uncalibrated_or_ood`, with no interval or coverage claim and an explicit DFT escalation.

Symmetry context and deletion-safe identity are deliberately separate in `discovery_os.crystal_identity`. `candidate_content_hash` remains the immutable-record integrity hash. `canonical_structure_hash` describes the symmetry-standardized prototype context, while hard deletion uses `identity_structure_hash` from the non-symmetrized source-Niggli representation plus strict unscaled `StructureMatcher` checks and a per-atom volume guard.

## Sidecars and dependencies

Model dependencies are isolated because CUDA, Torch, NumPy, and model packages can conflict. Pinned versions, source revisions, checkpoint requirements, licenses, and environment variable names are documented in:

- Dependency and bootstrap matrix: docs/DEPENDENCIES.md
- Sidecar operation guide: docs/SIDECARS.md
- Integration manifest: integrations/manifest.v1.json
- Expert/Fusion/Generator API contract: docs/EXPERT_API_CONTRACT.md
- Unified Fusion Core specification: docs/FUSION_CORE.md

Available adapters include MatterGen, REINVENT4, Uni-Mol, UMA, MatterSim, CHGNet, Chemprop, Boltz, ESM, RNA-FM, scGPT, QHNet, and PySCF. A registered adapter name does not mean that its package, checkpoint, license, or GPU environment is installed; the launcher fails closed when required state is missing.

## Repository safety

The repository excludes local environments, caches, run artifacts, logs, credentials, private keys, checkpoints, model weights, and downloaded archives through .gitignore. Never commit real API tokens or gated-model credentials; use environment variables or a secret manager.

## Documentation

- Unified Scientific Fusion Core: docs/FUSION_CORE.md
- Expert/Fusion/Generator API contract: docs/EXPERT_API_CONTRACT.md
- Model connection contract: docs/MODEL_CONTRACT.md
- Dependency and bootstrap matrix: docs/DEPENDENCIES.md
- Sidecar operation guide: docs/SIDECARS.md
- Validation matrix: docs/VALIDATION_MATRIX.md
- Literature RAG workflow: docs/LITERATURE_RAG.md
- MCP evidence source for RAG: docs/MCP_RAG.md
- Stage-specific validation evidence: docs/STAGE_VALIDATION_EVIDENCE.md
- Research foundations and scientific claim boundaries: docs/RESEARCH_FOUNDATIONS.md
- Codex material validation skill: .codex/skills/material-candidate-validation/SKILL.md
- Genomic AlphaGenome evaluation branch: docs/GENOMIC_ALPHAGENOME.md

Material Candidate Explorer is an exploration and orchestration system. Scientific validity still comes from independent, domain-appropriate computation and experiment.
