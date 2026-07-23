# Field-specific material discovery workflows

The candidate explorer routes a crystal through a code-owned material-field
profile before it builds literature queries, MCP evidence requests, or
high-fidelity calculation handoffs. The route is a scientific plan, not a
property calculator. MatterGen, MatterSim, CHGNet, literature RAG, and a
database search can triage candidates, but they do not by themselves establish
that a candidate is a battery electrode, superconductor, catalyst, or other
functional material.

The executable source of truth is
[`src/discovery_os/material_domains.py`](../src/discovery_os/material_domains.py).
Each profile fixes:

- required problem context;
- property names and units;
- five ordered evidence stages;
- read-only MCP capabilities;
- authoritative numerical or experimental validators;
- an explicit boundary on the scientific claim.

## Resolve the field first

List every code-owned profile:

~~~bash
discovery-os material-fields
~~~

Inspect one complete profile:

~~~bash
discovery-os material-fields --field solid_electrolyte
~~~

Resolve an explicit field:

~~~bash
discovery-os material-route \
  --field thermoelectric \
  --prompt "Find an n-type thermoelectric at 700 K"
~~~

Use automatic routing only when the operator genuinely does not know the
field:

~~~bash
discovery-os material-route \
  --field AUTO \
  --prompt "Find a stable Li solid electrolyte for a lithium-metal cell" \
  --fail-on-ambiguous
~~~

Without `--use-main-model`, `AUTO` uses a code-owned weighted keyword table. No
match falls back conservatively to `general_inorganic`. A tie records
`auto-ambiguous`, `requires_operator_choice=true`, and restricts the plan to
general screening. Use `--fail-on-ambiguous` in automation so an ambiguous
prompt stops instead of silently acquiring a specialized scientific claim.

## Optional main-AI field classifier

The main reasoning model may propose a field before retrieval. It does not
replace the code-owned profiles or deterministic resolver. Its strict response
contains:

- `primary_field`;
- up to four `secondary_fields` that describe real cross-field requirements;
- one `application_subtype` from the selected profile's allowlist;
- confidence from 0 to 1;
- short literal `evidence_spans` copied from the prompt, chemical system, or
  supplied problem context;
- `needs_clarification` plus exactly one clarification question when needed;
- a short routing summary.

The client rejects an unknown field, an unlisted subtype, malformed
clarification, duplicate fields, or an evidence span that is not literally
present in the normalized input. It records prompt hashes, model identity,
decision identity, and the verified structured decision. Classification occurs
before literature or MCP retrieval.

Code-owned reconciliation then applies these rules:

1. An explicit `--field` always wins, regardless of model output.
2. In `AUTO`, agreement between a deterministic field and a verified model
   primary field is `auto-consensus`.
3. A verified model decision with confidence at least 0.70 may resolve a
   deterministic no-match or tie; it is recorded as `auto-model`.
4. Model confidence below 0.70, a clarification request, no primary field, or
   model/deterministic disagreement is fail-closed
   `auto-model-conflict`: the selected profile is `general_inorganic`,
   `requires_operator_choice=true`, and specialized routing is blocked.
5. Without a configured model, CLI and notebook `AUTO` fall back to the
   deterministic resolver. Notebook `REQUIRED` refuses to generate instead.
   Notebook `OFF` always uses deterministic routing.

The 0.70 threshold is a conservative orchestration gate over an uncalibrated
model-reported confidence. It is not a calibrated scientific probability,
property uncertainty, or evidence that the selected material function is true.

Configure a dedicated OpenAI-compatible classifier endpoint:

~~~bash
export MATERIAL_FIELD_MODEL_API_URL="https://YOUR-OPENAI-COMPATIBLE-ENDPOINT/v1"
export MATERIAL_FIELD_MODEL_NAME="YOUR-MODEL"
export MATERIAL_FIELD_MODEL_API_KEY="" # optional if authentication is not required
export MATERIAL_FIELD_MODEL_TIMEOUT_SECONDS="180"

discovery-os material-route \
  --field AUTO \
  --prompt "Design a crystalline lithium solid electrolyte" \
  --chemical-system "Li-P-S" \
  --context-json '{"mobile_ion":"Li","temperature":300,"electrode_pair":"Li|NMC"}' \
  --use-main-model \
  --fail-on-ambiguous
~~~

The dedicated URL and model name form one atomic configuration. If either
`MATERIAL_FIELD_MODEL_API_URL` or `MATERIAL_FIELD_MODEL_NAME` is supplied,
both are required and only the dedicated API key and timeout are used. If the
dedicated pair is entirely absent, the complete `RAG_MODEL_*` configuration is
reused. Individual fields from the two configurations are never cross-paired.
`--use-main-model` fails when the chosen pair is incomplete; omitting the flag
uses deterministic routing.

In the notebook, set `MAIN_AI_FIELD_ROUTING` to `AUTO` (model when configured,
otherwise deterministic), `REQUIRED` (model configuration and a reconciled
decision required), or `OFF` (deterministic only). API keys are runtime secrets,
not request context or saved routing evidence.

The classifier can describe the scientific application, but it cannot select
an API, MCP URL/tool, database action, sidecar, calculation engine, validator,
ranking score, or scientific pass/fail result. Those choices remain code-owned
and administrator-configured.

The command output is a routing plan only. Its
`scientific_status` is
`routing-plan-only-no-field-property-calculation`, and every required property
that has not been calculated remains in
`unexecuted_required_properties`.

## The five-stage authority boundary

Every field uses the same ordered stages, but the questions, MCP capabilities,
and final validators are extended by the selected profile.

Each stage request also preserves the code-validated `application_subtype` and
the declared non-secret problem context. These values become bounded search
constraints in the RAG/MCP prompt; they do not select a tool and do not become
property values. An invalid subtype is rejected before retrieval.

| Stage | RAG and read-only MCP role | Runtime authority | Can steer generation? |
|---|---|---|---|
| `generation_prior` | Reported phases, composition ranges, synthesis conditions, negative results, and field operating conditions | Bound MatterGen snapshot plus its supported-condition guard and deterministic Evidence Fusion controller | Yes, but only source-closed evidence and supported generator conditions |
| `identity_novelty` | Crystallographic aliases, polymorphs, database identifiers, disorder, pressure phases, and field databases | Source-Niggli identity, strict unscaled pymatgen `StructureMatcher`, then local strict recheck of any external prefilter match | No |
| `mlip_disagreement` | Model cards, benchmark domains, chemistry, charge, spin, pressure, and bonding failure modes | Separately executed, unit-normalized MatterSim and CHGNet results on the same geometry; same-composition relative energies only | No |
| `relaxation_validation` | Reconstructions, decomposition pathways, soft modes, kinetic traps, finite-temperature and pressure context | Independent MatterSim and CHGNet `/v1/relax` results, optimizer convergence, and periodic geometry gates | No |
| `dft_handoff` | Validated methods, reference phases, pseudopotentials, functionals, spin/U/SOC, convergence settings, and field-specific controls | A separately executed `PeriodicDFTBackend` and the specialized physics workflows named below | No |

Literature RAG retrieves source-grounded records and helps formulate bounded
questions. The configured MCP tool is another structured evidence provider. It
is **read-only in this workflow**: it may search scholarly, crystallographic,
provenance, or benchmark records, but it may not mutate a candidate, run a
sidecar, relax a structure, submit DFT, or write an external database.

The administrator fixes the MCP endpoint and tool names through environment
variables. The prompt, planner, RAG model, candidate, and MCP response cannot
select or replace them. The client checks `tools/list`, input and output
schemas, and the structured `records` result before accepting evidence. See
[Stage-specific validation evidence](STAGE_VALIDATION_EVIDENCE.md) and
[MCP evidence sources for material RAG](MCP_RAG.md).

The capability IDs in a field profile describe the records that the
administrator-selected stage tool must be able to supply. They are not tool
names chosen by the model and are not permission to invoke a calculation.
Every field inherits the common capabilities below.

This repository does not ship or claim the existence of official upstream MCP
servers for Materials Project, OPTIMADE, COD, OpenKIM, NOMAD, AiiDA, OCP, or
the other named sources. It ships a client contract. An administrator may bind
that contract to a separately deployed read-only MCP server or adapter; leaving
it unconfigured records MCP as skipped. Capability labels below specify
required evidence functions and are not names of official upstream tools.

| Stage | Common evidence capabilities |
|---|---|
| `generation_prior` | `scholarly-materials-search`, `synthesis-procedure-and-negative-result-search` |
| `identity_novelty` | `optimade-federated-structure-search`, `materials-project-structure-search`, `cod-crystallography-search` |
| `mlip_disagreement` | `mlip-model-card-and-benchmark-search`, `openkim-test-search` |
| `relaxation_validation` | `nomad-or-aiida-provenance-search`, `phase-and-phonon-reference-search` |
| `dft_handoff` | `validated-method-and-convergence-search`, `nomad-or-aiida-provenance-search` |

Specialized profiles add only the capabilities needed for their field:

| Field | Stage-specific MCP capability additions |
|---|---|
| `general_inorganic` | DFT: `materials-project-phase-diagram-search` |
| `battery_electrode` | Generation: `battery-literature-and-protocol-search`; identity: `materials-project-insertion-electrode-search`; DFT: `battery-electrode-data-and-voltage-search` |
| `solid_electrolyte` | Generation: `solid-electrolyte-protocol-search`; DFT: `ionic-conductor-and-interface-data-search` |
| `superconductor` | Generation: `superconducting-materials-and-pressure-search`; identity: `superconductor-database-search`; DFT: `electron-phonon-method-and-data-search` |
| `heterogeneous_catalyst` | Generation: `catalysis-reaction-and-condition-search`; identity: `open-catalyst-and-surface-data-search`; DFT: `ocp-oc20-oc22-and-catalysis-data-search` |
| `semiconductor` | Identity: `electronic-materials-database-search`; DFT: `gw-hybrid-defect-and-transport-method-search` |
| `photovoltaic_absorber` | Generation: `photovoltaic-material-and-device-search`; DFT: `optical-defect-interface-data-search` |
| `thermoelectric` | Generation: `thermoelectric-transport-data-search`; DFT: `phonon-and-boltzmann-transport-search` |
| `magnetic_material` | Identity: `magnetic-structure-database-search`; DFT: `magnetic-order-soc-and-exchange-search` |
| `ferroelectric_piezoelectric` | Identity: `polar-and-ferroelectric-structure-search`; DFT: `dfpt-polarization-and-piezoelectric-search` |
| `structural_alloy` | Generation: `alloy-processing-and-failure-search`; DFT: `calphad-elastic-defect-and-service-data-search` |
| `porous_framework` | Identity: `core-mof-csd-cod-and-zeolite-search`; DFT: `adsorption-isotherm-and-force-field-search` |

## The 12 profiles

The property identifiers and units below are single canonical contracts. A
validator may create a comparable score only for its explicitly named
properties. Context such as temperature, pressure, composition, state of
charge, carrier concentration, surface coverage, field, and microstructure is
not optional metadata: it defines what the value means.

| Field | Required properties and canonical units | Named score-producing authorities | Claim boundary |
|---|---|---|---|
| `general_inorganic` | `energy_above_hull` — eV/atom; `minimum_phonon_frequency` — THz | `energy_above_hull` → `reference-phase-dft-and-phase-diagram`; `minimum_phonon_frequency` → `phonon-stability-workflow` | Generated and MLIP-relaxed crystals are triage, not proof of thermodynamic, dynamic, synthetic, or functional validity |
| `battery_electrode` | `average_voltage` — V; `specific_capacity` — mAh/g; `ion_migration_barrier` — eV | `average_voltage`, `specific_capacity` → `battery-reaction-phase-diagram`; `ion_migration_barrier` → `working-ion-neb-or-aimd` | A stable host does not establish voltage, accessible capacity, ion kinetics, cycling, charged-state stability, or safety |
| `solid_electrolyte` | `ionic_conductivity` — S/cm; `migration_barrier` — eV; `electrochemical_stability_window` — V | `ionic_conductivity` → `finite-temperature-ion-transport`; `migration_barrier` → `mobile-ion-path-neb`; `electrochemical_stability_window` → `electrode-interface-grand-potential` | Static energies and force agreement do not establish conductivity, grain-boundary transport, or electrode compatibility |
| `superconductor` | `critical_temperature` — K; `electron_phonon_coupling` — dimensionless | Both required properties → `epw-or-eliashberg-workflow` | A predicted transition temperature alone is not superconductivity; phase identity, zero resistance, and bulk magnetic screening must coincide |
| `heterogeneous_catalyst` | `reaction_free_energy` — eV; `activation_barrier` — eV; `durability` — h | `reaction_free_energy` → `surface-adsorbate-free-energy-workflow`; `activation_barrier` → `transition-state-and-microkinetic-workflow`; `durability` → `operando-durability-validation` | Bulk stability or a generic catalyst score is not activity, selectivity, kinetics, or durability |
| `semiconductor` | `band_gap` — eV; `carrier_mobility` — cm^2/(V s); `minimum_native_defect_formation_energy` — eV | `band_gap` → `hybrid-or-gw-electronic-structure`; `carrier_mobility`, `minimum_native_defect_formation_energy` → `charged-defect-and-transport-workflow` | A semilocal or ML gap is triage and cannot establish mobility, defects, dopability, or device behavior |
| `photovoltaic_absorber` | `optical_absorption_coefficient` — cm^-1; `slme` — fraction; `nonradiative_recombination_rate` — s^-1 | `optical_absorption_coefficient`, `slme` → `quasiparticle-optics-and-slme`; `nonradiative_recombination_rate` → `photovoltaic-defect-interface-workflow` | A gap near an empirical optimum does not establish absorption, efficiency, native-defect energetics, contacts, recombination, or operational stability |
| `thermoelectric` | `power_factor` — W/(m K^2); `lattice_thermal_conductivity` — W/(m K); `zt` — dimensionless | `power_factor` → `electronic-boltzmann-transport`; `lattice_thermal_conductivity` → `anharmonic-phonon-transport`; `zt` → `thermoelectric-zt-integration` | Band features or harmonic phonons alone cannot establish power factor, thermal conductivity, or ZT |
| `magnetic_material` | `magnetic_ordering_energy` — eV/atom; `magnetocrystalline_anisotropy` — MJ/m^3; `ordering_temperature` — K | `magnetic_ordering_energy` → `magnetic-order-and-correlation-workflow`; `magnetocrystalline_anisotropy`, `ordering_temperature` → `soc-anisotropy-exchange-temperature-workflow` | One ferromagnetic initialization or predicted moment does not establish the ground state, anisotropy, coercivity, or ordering temperature |
| `ferroelectric_piezoelectric` | `spontaneous_polarization` — C/m^2; `switching_barrier` — eV per formula unit; `piezoelectric_strain_coefficient` — pm/V | `spontaneous_polarization`, `switching_barrier` → `berry-phase-switching-workflow`; `piezoelectric_strain_coefficient` → `dfpt-polar-response-workflow` | A polar space group is not proof of switchable ferroelectricity or a usable piezoelectric |
| `structural_alloy` | `mixing_gibbs_free_energy` — eV/atom; `youngs_modulus` — GPa; `service_degradation_rate` — s^-1 | `mixing_gibbs_free_energy` → `finite-temperature-alloy-thermodynamics`; `youngs_modulus`, `service_degradation_rate` → `elastic-defect-and-service-workflow` | 0 K stability and elastic moduli do not establish strength, toughness, creep, fatigue, oxidation, corrosion, or processability |
| `porous_framework` | `accessible_volume_fraction` — dimensionless; `adsorption_selectivity` — dimensionless; `framework_decomposition_free_energy` — eV/atom | `accessible_volume_fraction` → `probe-resolved-porosity-workflow`; `adsorption_selectivity` → `gcmc-mixture-adsorption-workflow`; `framework_decomposition_free_energy` → `framework-stability-workflow` | A geometric pore or rigid-framework GCMC result does not establish accessible capacity, mixture selectivity, activation, flexibility, or humidity stability |

### Conditions are part of every property value

A canonical unit does not make two values comparable when their operating
conditions differ. The assessment gate requires every context field below
before it accepts a result from the named authority. Computational-ranking
readiness also requires one explicit, complete `target_conditions` set; this
prevents individually valid results at different temperatures, pressures, or
other operating states from being combined across properties.

| Field | Required property conditions |
|---|---|
| `general_inorganic` | `energy_above_hull`, `minimum_phonon_frequency`: pressure and temperature |
| `battery_electrode` | `average_voltage`: working ion, reference electrode, and state of charge; `specific_capacity`: working ion and cycling protocol; `ion_migration_barrier`: working ion, state of charge, and temperature |
| `solid_electrolyte` | `ionic_conductivity`: mobile ion, temperature, and microstructure; `migration_barrier`: mobile ion and defect concentration; `electrochemical_stability_window`: electrode pair and temperature |
| `superconductor` | `critical_temperature`: pressure, magnetic field, and isotope; `electron_phonon_coupling`: pressure |
| `heterogeneous_catalyst` | `reaction_free_energy`: reaction, facet, coverage, temperature, pressure, electrode potential, and pH; `activation_barrier`: reaction, facet, coverage, and temperature; `durability`: reaction, temperature, pressure, electrode potential, and pH |
| `semiconductor` | `band_gap`: temperature and strain; `carrier_mobility`: carrier type, temperature, and doping; `minimum_native_defect_formation_energy`: Fermi level, chemical potentials, and charge state |
| `photovoltaic_absorber` | `optical_absorption_coefficient`: photon energy, polarization, and temperature; `slme`: absorber thickness and temperature; `nonradiative_recombination_rate`: chemical potentials, contacts, temperature, and carrier concentration |
| `thermoelectric` | `power_factor`: temperature, carrier concentration, and carrier type; `lattice_thermal_conductivity`: temperature and microstructure; `zt`: temperature, carrier concentration, and microstructure |
| `magnetic_material` | `magnetic_ordering_energy`: temperature and magnetic field; `magnetocrystalline_anisotropy`: temperature; `ordering_temperature`: magnetic field |
| `ferroelectric_piezoelectric` | `spontaneous_polarization`: orientation and temperature; `switching_barrier`: electric field and stress; `piezoelectric_strain_coefficient`: orientation, tensor component, temperature, and stress |
| `structural_alloy` | `mixing_gibbs_free_energy`: composition range, temperature, and processing history; `youngs_modulus`: temperature, orientation, and microstructure; `service_degradation_rate`: service environment, degradation mechanism, temperature, and time |
| `porous_framework` | `accessible_volume_fraction`: guest species and activation state; `adsorption_selectivity`: guest species, temperature, pressure, and humidity; `framework_decomposition_free_energy`: temperature, humidity, and activation state |

If a required condition is absent, the property is `incomparable`, not a
default-condition estimate. If the named authority did not run successfully,
the property is `unknown`. Successful values at different condition sets are
`incomparable`. Different successful normalized values at the same complete
condition set are `conflicting` and remain separate; the assessment never
averages them or chooses an automatic winner.

### `general_inorganic`

RAG asks for competing phases, synthesis windows, failed synthesis, known
polymorphs, pressure/temperature phases, and phonon or decomposition evidence.
MCP capabilities cover federated structure search, phase diagrams, provenance,
and phonon references. DFT formation energies must share a compatible reference
set before a convex-hull value is created. The phonon authority reports
`minimum_phonon_frequency` in THz at the stated pressure and temperature while
preserving the full spectrum; imaginary modes require convergence, supercell,
and non-analytical-correction review where applicable.

### `battery_electrode`

Generation and identity evidence add working-ion, state-of-charge, voltage,
capacity, volume-change, and insertion-electrode records. Numerical validation
enumerates accessible end members and intermediate orderings before voltage and
capacity are calculated. Migration barriers are pathway- and charge-state
specific. Cell chemistry, cycling protocol, first-cycle efficiency, retention,
impedance, and safety remain experimental authorities.

### `solid_electrolyte`

The route adds mobile-ion sublattice, defect concentration, temperature,
electrode pair, processing, grain-boundary, and interface evidence. NEB answers
a chosen pathway question; long-time AIMD or uncertainty-qualified MLIP MD
answers a finite-temperature diffusion question. Neither substitutes for
impedance measurements that separate bulk and grain-boundary response.

### `superconductor`

RAG and MCP must preserve pressure, isotope, field, phase identity, and pairing
assumption. Electron-phonon calculations are authoritative only for a justified
conventional mechanism and require converged electronic and phonon meshes.
Unconventional candidates require a mechanism-specific many-body route rather
than forcing an Allen-Dynes estimate.

### `heterogeneous_catalyst`

The object under test is an active surface or interface under a named reaction,
not just a bulk CIF. Retrieval includes facets, adsorbates, coverage, solvent,
potential, pH, pressure, reconstruction, poisoning, and negative results.
OC20/OC22 models are surface-adsorbate triage within their documented domain;
free-energy corrections, transition states, microkinetics, product-resolved
activity, and operando durability remain separate. `durability` is accepted in
hours only when `operando-durability-validation` preserves the reaction,
temperature, pressure, electrode potential, and pH.

### `semiconductor`

The route distinguishes quasiparticle or optical gap from a semilocal screening
gap. It searches validated treatments for SOC, electron-phonon scattering,
charged-defect corrections, chemical-potential bounds, Fermi level, doping,
temperature, strain, and dimensionality. Effective mass alone is not mobility,
and one neutral defect supercell is not a dopability assessment.
`minimum_native_defect_formation_energy` is an eV value tied to the stated
Fermi level, chemical potentials, and charge state.

### `photovoltaic_absorber`

Retrieval adds thickness, polarization, illumination, contacts, device
architecture, recombination, and operational degradation. SLME is computed
from a converged absorption spectrum and direct/indirect gap information at a
declared thickness; it is not inferred from the band gap. Certified device
efficiency is an experimental result, not an SLME synonym.
`optical_absorption_coefficient` is a cm^-1 value at a specified photon energy,
polarization, and temperature; `nonradiative_recombination_rate` is an s^-1
value conditioned on chemical potentials, contacts, temperature, and carrier
concentration.

### `thermoelectric`

All transport terms must share temperature, carrier type, carrier
concentration, and microstructure. Electronic transport needs a stated
scattering model; lattice thermal conductivity needs converged second- and
third-order force constants and a phonon BTE solution. ZT is accepted only when
Seebeck coefficient, electrical conductivity, electronic thermal conductivity,
and lattice thermal conductivity are combined under consistent conditions.

### `magnetic_material`

RAG searches oxidation states, known magnetic structures, correlation
treatments, non-collinear/SOC effects, temperature, and application-specific
requirements. Multiple magnetic orderings must be enumerated. An ordering
temperature requires exchange parameters and a declared statistical model;
permanent-magnet, soft-magnet, spintronic, and magnetocaloric claims need
different experimental controls. `magnetic_ordering_energy` is normalized to
eV/atom and remains tied to the stated temperature and magnetic field.

### `ferroelectric_piezoelectric`

The route searches parent and polar phases, symmetry, soft modes, domain states,
leakage, switching paths, temperature, stress, orientation, and fatigue.
Polarization is a Berry-phase difference along a justified insulating path, not
an absolute point value. Switchability requires a meaningful path and
experiment; a polar structure alone is insufficient.
`piezoelectric_strain_coefficient` is reported only in pm/V for a declared
orientation, tensor component, temperature, and stress.

### `structural_alloy`

Composition ranges, disorder, processing history, temperature, microstructure,
and service environment are first-class context. The route searches CALPHAD
assessments, thermodynamic databases, elastic/defect data, oxidation or
corrosion, creep, fatigue, fracture, and failed processing. A DFT or MLIP
crystal represents one state, not the processed alloy microstructure.
`mixing_gibbs_free_energy` (eV/atom), `youngs_modulus` (GPa), and
`service_degradation_rate` (s^-1) are meaningful only with their composition or
service, temperature, processing, orientation, microstructure, mechanism, and
time conditions preserved.

### `porous_framework`

Identity retrieval must account for solvent, disorder, activation state, and
framework aliases. Geometric accessibility records the probe; adsorption
records guest mixture, temperature, pressure, humidity, charges, force field,
and flexibility. Computation-ready database membership is not proof that a
sample can be activated or retains porosity after cycling.
`accessible_volume_fraction` and `adsorption_selectivity` are dimensionless
under their declared probe/guest and thermodynamic conditions;
`framework_decomposition_free_energy` is in eV/atom for the stated temperature,
humidity, and activation state.

## UNKNOWN and fail-closed rules

- `implemented` means the repository contains the named validator path; it
  does not mean that it ran successfully for this candidate.
- `sidecar_required`, `credential_required`, and `external_required` are
  explicit availability states. If the dependency is absent or the call fails,
  the result is `UNKNOWN`, not zero and not pass.
- `skipped`, partial, failed, unconverged, missing-unit, missing-context, and
  incomplete-provenance results are `UNKNOWN`.
- RAG record counts, citations, summaries, and MCP records never create a
  property value, novelty boolean, relaxation flag, Pareto score, or DFT result.
- Absence from literature, Materials Project, OPTIMADE, COD, or a specialist
  database is not proof of novelty.
- The selector's `novelty` branch is property-space diversity. Scientific
  structural/database novelty is a separate scoped assessment with provider and
  snapshot provenance.
- MatterSim and CHGNet absolute total energies are not compared across models or
  stoichiometries. Cross-model energy evidence requires aligned
  same-composition relative energies after unit normalization.
- Preparing CIF, POSCAR, Quantum ESPRESSO, NEB, phonon, EPW, defect, or transport
  inputs is a handoff, not an executed result.
- A field claim is allowed only after all properties marked
  `required_for_field_claim` have authoritative, condition-complete results and
  the profile's experimental boundary is stated.

## Research basis and official implementations

These sources ground the route design; they are not imported as numerical
results:

- General generation and stability: [MatterGen, Nature
  (2025)](https://www.nature.com/articles/s41586-025-08628-5),
  [Materials Project phase-diagram
  documentation](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds),
  and [Quantum ESPRESSO `pw.x`
  inputs](https://www.quantum-espresso.org/Doc/INPUT_PW.html).
- Battery electrodes: [Materials Project Battery Explorer
  tutorial](https://docs.materialsproject.org/apps/explorer-apps/battery-explorer/tutorial)
  and [pymatgen battery
  analysis](https://pymatgen.org/pymatgen.analysis.battery.html).
- Solid electrolytes: [Famprikis et al., Nature Materials
  (2019)](https://www.nature.com/articles/s41563-019-0431-3) and
  [Zhao et al., Nature Reviews Materials
  (2020)](https://www.nature.com/articles/s41578-019-0165-5).
- Superconductors: [EPW official
  documentation](https://docs.epw-code.org/), [EPW method
  paper](https://doi.org/10.1016/j.cpc.2016.07.028), and
  [Allen-Dynes strong-coupling
  analysis](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.12.905).
- Catalysts: [OC20, ACS Catalysis
  (2021)](https://pubs.acs.org/doi/10.1021/acscatal.0c04525),
  [OC22, ACS Catalysis
  (2023)](https://pubs.acs.org/doi/10.1021/acscatal.2c05426), and
  [CatMAP](https://doi.org/10.1002/cctc.201300825).
- Semiconductors: [Materials Project electronic-structure
  methodology](https://docs.materialsproject.org/methodology/materials-methodology/electronic-structure)
  and [AMSET first-principles scattering and
  transport](https://www.nature.com/articles/s41524-021-00529-1).
- Photovoltaics: [Yu and Zunger's SLME
  paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.108.068701).
- Thermoelectrics: [BoltzTraP2](https://doi.org/10.1016/j.cpc.2018.05.010),
  [phono3py](https://phonopy.github.io/phono3py/), and
  [ShengBTE](https://doi.org/10.1016/j.cpc.2014.02.015).
- Magnetic materials: [Materials Project magnetic-property
  methodology](https://docs.materialsproject.org/methodology/materials-methodology/magnetic-properties)
  and [pymatgen magnetic-order
  analysis](https://pymatgen.org/pymatgen.analysis.magnetism.html).
- Ferroelectrics and piezoelectrics: [modern polarization
  theory](https://journals.aps.org/rmp/abstract/10.1103/RevModPhys.66.899)
  and [Materials Project piezoelectric
  methodology](https://docs.materialsproject.org/methodology/materials-methodology/piezoelectric-constants).
- Structural alloys: [pycalphad](https://pycalphad.org/docs/latest/),
  [OpenCalphad](https://www.nist.gov/publications/open-calphad-free-thermodynamic-software),
  and [Materials Project elasticity
  methodology](https://docs.materialsproject.org/methodology/materials-methodology/elasticity).
- Porous frameworks: [CoRE MOF
  2019](https://pubs.acs.org/doi/10.1021/acs.jced.9b00835),
  [RASPA](https://www.tandfonline.com/doi/full/10.1080/08927022.2015.1010082),
  and [Zeo++](https://pubs.acs.org/doi/10.1021/acs.chemmater.7b01475).
- Evidence and interoperability: [OPTIMADE
  specification](https://www.optimade.org/optimade/),
  [Materials Project API](https://materialsproject.github.io/api/),
  [AiiDA provenance](https://www.aiida.net/), and the
  [MCP specification](https://modelcontextprotocol.io/specification/2025-11-25).

Use the sources to choose hypotheses, boundary conditions, convergence studies,
and experiments. Preserve the raw runtime outputs that actually determine a
candidate's scientific status.
