# Material Goal Discovery Quick Usage Guide

## 1. CLI Usage (Command Line)

Run a natural-language material goal search directly in terminal:

```bash
discovery-os material-goal-run \
  --goal "High ionic conductivity solid electrolyte candidate for all-solid-state lithium batteries" \
  --export-markdown report.md \
  --export-json report.json
```

### CLI Parameters

| Option | Required | Description | Default |
| :--- | :--- | :--- | :--- |
| `--goal` | **Yes** | Natural language request prompt for target material | None |
| `--export-markdown` | No | Path to save candidate Markdown report | None |
| `--export-json` | No | Path to save candidate JSON report | None |
| `--artifacts` | No | Output folder for discovery run logs | `.discovery/goals` |
| `--no-rag` | No | Disable automatic literature RAG queries | `False` |

---

## 2. Python API Usage

Execute discovery directly in Python:

```python
from discovery_os.material_decision_runner import MaterialDecisionRunner
from discovery_os.rich_report import format_material_candidate_markdown_report

# 1. Run discovery
runner = MaterialDecisionRunner(artifact_root=".discovery/goals")
run, reports = runner.run_material_goal_discovery("High Tc superconductor candidate")

# 2. Access top candidate report
top = reports[0]
print(f"Formula: {top.formula} | Space Group: {top.identity.space_group}")
print(f"Formation Energy: {top.properties.formation_energy_ev_per_atom} eV/atom")

# 3. Export Markdown
with open("report.md", "w", encoding="utf-8") as f:
    f.write(format_material_candidate_markdown_report(top))
```

---

## 3. Google Colab T4 GPU Usage

1. Open `MATERIAL_CANDIDATE_DISCOVERY_T4.ipynb` in Colab.
2. Run Setup cell to install packages.
3. Set goal prompt & run:
   ```python
   TARGET_GOAL = "Narrow bandgap perovskite candidate for solar cell"
   ```

---

## 4. Output Report Quick Structure

Each generated `report.md` contains:
1. **Executive Overview**: Target domain & application role
2. **Crystallographic Identity**: Formula, space group, cell ($a,b,c,\alpha,\beta,\gamma$), volume, CIF
3. **Physical Properties**: Formation energy ($\Delta E_f$), $E_{\text{hull}}$, stress gate status
4. **Multi-Expert MLIP Reliability**: CHGNet & MatterSim energies, disagreement ($\sigma_{\text{expert}}$), Pareto rank
5. **Database Novelty**: OPTIMADE, COD, Materials Project match status
6. **Literature RAG Evidence**: 5-stage RAG citations & DOI/arXiv links
7. **Downstream DFT Handoff**: POSCAR / QE input specs & k-mesh
