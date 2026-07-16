# Material Candidate Explorer

[📓 Open the notebook on GitHub](https://github.com/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_EXPLORER_V11_PROMPT_RAG_REAL_GENERATION.ipynb) · [▶ Run it in Google Colab](https://colab.research.google.com/github/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_EXPLORER_V11_PROMPT_RAG_REAL_GENERATION.ipynb)

[![Open in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_EXPLORER_V11_PROMPT_RAG_REAL_GENERATION.ipynb) [![View Notebook](https://img.shields.io/badge/Notebook-GitHub-181717?logo=github)](https://github.com/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_EXPLORER_V11_PROMPT_RAG_REAL_GENERATION.ipynb)

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

## Implemented behavior

- Candidate batches and paired OFF/ON runs support candidate_count from 1 to 1024.
- Fusion search keeps stability, target-property, novelty, expert-disagreement, and Pareto branches.
- Every generated candidate is evaluated again by the configured experts.
- Goal-hashed feature caching prevents duplicate expert calculations.
- CIF, SMILES, sequences, original expert payloads, generator provenance, latent states, scheduler controls, and search history are persisted under the artifact root.
- If FUSION_API_URL is unset, the deterministic local EvidenceDrivenFusionBackend is selected automatically.
- Remote Fusion requests keep the legacy strict payload by default. Extended local search context requires FUSION_SEND_EXTENDED_REQUEST_CONTEXT=1.

## Quick start

## Interactive notebook

- [View the notebook on GitHub](https://github.com/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_EXPLORER_V11_PROMPT_RAG_REAL_GENERATION.ipynb)
- [Open the notebook in Google Colab](https://colab.research.google.com/github/jujumelona/material-candidate-explorer/blob/main/MATERIAL_CANDIDATE_EXPLORER_V11_PROMPT_RAG_REAL_GENERATION.ipynb)

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
discovery-os fusion-search --search-id material-run-001 --goal goal.json --parent parent.json --run-config run-config.json --generator mattergen --rounds 4 --frontier-width 6 --artifacts runs/material-run-001
~~~

goal.json defines objectives and units. parent.json contains the starting candidate, normally as CIF or a chemical formula. run-config.json binds workspace_mode=on, candidate_count, the evaluator panel, generator identity, checkpoint revision, and runtime parameters.

The client reads sidecar URLs from the integration manifest environment names, for example:

~~~powershell
$env:MATTERGEN_API_URL = "http://127.0.0.1:8101"
$env:MATTERSIM_API_URL = "http://127.0.0.1:8110"
$env:CHGNET_API_URL = "http://127.0.0.1:8111"
~~~

Loopback HTTP requires the explicit development opt-in generated by the launcher. Production endpoints must use HTTPS and environment or secret-manager tokens.

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

The Colab notebook includes a separate AlphaGenome batch-evaluation cell. It requests the key with hidden input, evaluates every candidate, and writes a ranked JSON table. AlphaGenome API use is subject to Google's terms and non-commercial restrictions; this project does not make clinical claims or redistribute model weights.

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

The scheduler lowers guidance alpha after sustained improvement, explores more broadly after stagnation, and reduces temperature or mutation when structural collapse rises. MatterGen v1 applies supported guidance, condition, seed, and batch controls; it does not expose temperature, mutation strength, or diversity strength, so those values remain provenance and warnings rather than being falsely reported as applied. REINVENT applies its supported sampling controls.

## Outputs and validation handoff

Each artifact root contains candidate records, CIF/SMILES/sequence representations, original expert payloads, feature cache entries, latent states, generator provenance, scheduler history, branch pools, and the final diagnostic report.

validation_handoff_candidate_refs is a bounded shortlist for a separately configured high-cost validator. It prioritizes Pareto candidates, then stability candidates, and removes exact scientific representation duplicates. It does not mean that DFT, relaxation, phonons, experiments, or synthesis have run.

MatterSim and UMA expose total energy (eV) and energy_per_atom (eV/atom); the latter is comparable as an expert property axis with CHGNet. energy_per_atom is not energy_above_hull. Convex-hull stability requires reference phases, relaxation, and a dedicated validated hull connector.

Symmetry- and tolerance-aware crystal matching is not part of exact-content deduplication yet; use a dedicated structure-standardization connector for that boundary.

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
- Stage-aware multi-RAG contracts: docs/STAGE_AWARE_MULTI_RAG.md
- Genomic AlphaGenome evaluation branch: docs/GENOMIC_ALPHAGENOME.md

Material Candidate Explorer is an exploration and orchestration system. Scientific validity still comes from independent, domain-appropriate computation and experiment.
