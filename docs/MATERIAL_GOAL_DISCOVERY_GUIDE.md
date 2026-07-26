# Material Goal Discovery Guide: End-to-End User & Developer Manual

This guide explains how to use **Material Candidate Explorer** (`discovery_os`) to execute natural-language material discovery requests, route queries across 5 evidence RAG stages and MCP skills, run multi-expert MLIP property screening, and export standardized 7-section scientific candidate reports (`RichMaterialCandidateReport`).

---

## 1. Quick Start: CLI Command (`discovery-os material-goal-run`)

To discover material candidates for a specific natural language material application goal, use the `material-goal-run` subcommand:

```bash
discovery-os material-goal-run \
  --goal "High ionic conductivity solid electrolyte candidate for all-solid-state lithium batteries" \
  --artifacts .discovery/goals \
  --export-markdown battery_electrolyte_report.md \
  --export-json battery_electrolyte_report.json
```

### CLI Command Options

| Parameter | Required | Description | Default |
| :--- | :--- | :--- | :--- |
| `--goal` | **Yes** | Natural language request specifying the material application & targets | None |
| `--artifacts` | No | Directory path to store JSON/MD run artifacts | `.discovery/goals` |
| `--no-rag` | No | Disable automatic literature RAG stage queries | `False` (RAG enabled) |
| `--export-markdown` | No | Export the top candidate's report as a Markdown file | None |
| `--export-json` | No | Export the top candidate's report as a JSON file | None |

---

## 2. Python API Usage

You can invoke the discovery pipeline directly in Python scripts or Jupyter Notebooks:

```python
from discovery_os.material_decision_runner import MaterialDecisionRunner
from discovery_os.rich_report import format_material_candidate_markdown_report

# 1. Initialize runner with artifact output directory
runner = MaterialDecisionRunner(artifact_root=".discovery/goals")

# 2. Execute natural language goal discovery
run, rich_reports = runner.run_material_goal_discovery(
    goal="High Tc superconductor candidate under ambient pressure",
    run_rag=True,
)

print(f"Run ID: {run.run_id}")
print(f"Detected Field: {run.brief.material_field}")
print(f"Generated Reports Count: {len(rich_reports)}")

# 3. Access top candidate report
top_report = rich_reports[0]
print(f"Formula: {top_report.formula}")
print(f"Space Group: {top_report.identity.space_group}")
print(f"CHGNet Energy: {top_report.reliability.chgnet_energy_ev_per_atom} eV/atom")
print(f"MatterSim Energy: {top_report.reliability.mattersim_energy_ev_per_atom} eV/atom")

# 4. Export Markdown format
markdown_text = format_material_candidate_markdown_report(top_report)
with open("superconductor_report.md", "w", encoding="utf-8") as f:
    f.write(markdown_text)
```

---

## 3. Understanding the 7-Section Rich Candidate Report

Every discovered candidate is output as a `RichMaterialCandidateReport` containing 7 standardized sections:

### Section 1: Executive Overview & Target Application
- **User Goal**: Natural language request prompt.
- **Domain & Role**: Automatically classified domain (e.g. `battery`, `catalyst`, `superconductor`, `medicinal`, `polymer`) and specific role (e.g. `solid_electrolyte`, `oer_catalyst`).

### Section 2: Crystallographic & Chemical Identity
- **Formula / Reduced Formula**: Standard chemical composition.
- **Space Group & Crystal System**: Symmetry group (e.g. `Ia-3d`, `P63/mmc`) and lattice family.
- **Lattice Parameters & Angles**: $a, b, c$ (in Å) and $\alpha, \beta, \gamma$ (in degrees).
- **Unit Cell Volume**: Volume in Å³.
- **Niggli Identity Hash**: Canonical Niggli-reduced species-preserving identity digest (`sha256:...`) used for strict crystallographic deduplication.
- **CIF Preview**: Unscaled raw Crystallographic Information File representation.

### Section 3: Evaluated Physical & Chemical Properties
- **Formation Energy ($\Delta E_f$)**: MLIP-calculated formation energy per atom.
- **Energy Above Hull ($\Delta E_{\text{hull}}$)**: Distance to the thermodynamic convex hull.
- **Max Atomic Force & Stress Gate Status**: Maximum residual force and stress tensor validation (`pass` / `fail`).
- **Role-Specific Application Metrics**: Estimated ionic conductivity, bandgap, overpotential, magnetic moment, etc.

### Section 4: Multi-Expert MLIP Reliability & Disagreement
- **CHGNet 0.3.0 & MatterSim 5M Energies**: Independent neural network interatomic potential evaluations.
- **Expert Disagreement ($\sigma_{\text{expert}}$)**: Inter-model energy prediction difference ($\text{eV/atom}$) and disagreement status (`low`, `medium`, `high`).
- **Conformal Reliability Score**: Split-conformal coverage calibration score.
- **Pareto Rank**: NSGA-II non-dominated sorting rank within the composition pool.

### Section 5: Structural Novelty & Database Cross-Check
- **Current Batch & Project History Uniqueness**: Deduplication results against current run and historical project stores.
- **External Database Match Status**: Strict unscaled structure match checks against:
  - **OPTIMADE API**
  - **Crystallography Open Database (COD)**
  - **Materials Project API**
- **Aggregate Novelty Status**: `scoped_no_match` (externally unique), `matched` (known material), or `unknown`.

### Section 6: Stage Literature RAG Evidence & Citations
- **5 Evidence Stages**:
  1. `generation_prior`: Literature background for seed selection.
  2. `identity_novelty`: External database & literature novelty evidence.
  3. `mlip_disagreement`: Model reliability & calibration references.
  4. `relaxation_validation`: Geometry gate & MLIP relaxation literature.
  5. `dft_handoff`: First-principles calculation parameters & pseudopotential references.
- **Citations**: Extracted DOI links (`https://doi.org/...`) and arXiv papers (`https://arxiv.org/abs/...`).

### Section 7: Portable Downstream DFT Handoff Package
- **Target Simulation Code**: Default `Quantum ESPRESSO` / `VASP`.
- **Reciprocal k-mesh**: Target spacing grid (e.g. $4 \times 4 \times 4$).
- **Energy Cutoffs**: Wavefunction cutoff ($E_{\text{cutwfc}}$) and charge density cutoff ($E_{\text{cutrho}}$) in Rydberg.
- **Pseudopotential Attestation**: e.g., `SG15 ONCVPSP v1.0 standard`.
- **POSCAR / QE Input**: Path to ready-to-run calculation input files.

---

## 4. Google Colab T4 GPU Execution

In Google Colab using a free T4 GPU instance:

1. Open `MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb`.
2. Run the environment setup cell to install dependencies (`pymatgen`, `chgnet`, `mattersim`, `discovery_os`).
3. Set your target material goal prompt in the notebook parameter cell:
   ```python
   TARGET_GOAL = "High-performance OER catalyst candidate for water splitting"
   ```
4. Run the execution cells to perform candidate generation, MLIP relaxation, Pareto ranking, and display the formatted Markdown candidate report directly in the notebook output.

---

## 5. Scientific Boundaries & Guidelines

> [!WARNING]
> Computational leads generated by `discovery_os` are MLIP-relaxed candidate hypotheses. Synthesis, chemical identity verification, independent physical measurements, and full DFT convergence validation are required before asserting scientific claims.
