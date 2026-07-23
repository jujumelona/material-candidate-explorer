# Research foundations and scientific claim boundaries

This document maps the implemented material-candidate workflow to primary literature and official model or database documentation. The software is a screening and orchestration system: it generates hypotheses, preserves evidence and provenance, and prepares expensive validation. It does not establish synthesis, novelty, thermodynamic stability, superconductivity, medical efficacy, or any other experimental claim.

## What the pipeline implements

| Pipeline decision | Implemented contract | Research or official source |
|---|---|---|
| Conditional crystal generation | Use the fixed condition allowlist for a named official MatterGen checkpoint; treat any custom-checkpoint allowlist as operator attestation, not automatic training-config verification; record requested, applied, and ignored controls separately | [MatterGen paper](https://www.nature.com/articles/s41586-025-08628-5), [official repository](https://github.com/microsoft/mattergen), [official model card](https://github.com/microsoft/mattergen/blob/main/MODEL_CARD.md) |
| Crystal identity | Hard-delete only a strict, species-preserving, unscaled crystallographic duplicate; retain volume-scaled prototype similarity as a separate relation | [pymatgen `StructureMatcher`](https://pymatgen.org/pymatgen.core.html#pymatgen.core.structure_matcher.StructureMatcher), [AFLOW-XtalFinder](https://www.nature.com/articles/s41524-020-00483-4) |
| Proxy-energy ranking | Compare raw MLIP energies only inside one reduced-composition pool; preserve multi-objective front coverage with NSGA-II crowding | [NSGA-II](https://doi.org/10.1109/4235.996017), [Materials Project energy-mixing warning](https://docs.materialsproject.org/frequently-asked-questions/glossary-of-terms) |
| Evaluator artifact identity | Bind the default MatterSim and CHGNet evaluators to immutable upstream source revisions, exact byte counts, and project-pinned file SHA-256 values; rehash before launch | [MatterSim official repository](https://github.com/microsoft/mattersim), [CHGNet official repository](https://github.com/CederGroupHub/chgnet) |
| MLIP uncertainty | Apply calibrated intervals only when model weights, DFT labels, units, diagnostics, and chemistry/exchangeability scope match exactly; otherwise return unknown/OOD and escalate | [Conformal UQ for ML potentials](https://doi.org/10.1088/2632-2153/aca7b1), [calibrated energy/force ensembles](https://doi.org/10.1039/D3CP02143B), [misspecified MLIP uncertainty](https://www.nature.com/articles/s41524-025-01758-4), [flexible uncertainty calibration](https://www.nature.com/articles/s41524-026-02080-3) |
| Benchmark interpretation | Treat benchmark performance as evidence about a fixed dataset and task, not a universal guarantee for a new chemistry | [Matbench Discovery](https://doi.org/10.1038/s42256-025-01055-1), [benchmark repository](https://github.com/janosh/matbench-discovery) |
| DFT handoff | Prepare reviewable structures and input skeletons while leaving cutoffs, pseudopotentials, convergence, reference phases, and execution explicit | [Quantum ESPRESSO `pw.x` input reference](https://www.quantum-espresso.org/Doc/INPUT_PW.html), [SSSP](https://www.nature.com/articles/s41524-018-0127-2), [DFT reproducibility study](https://doi.org/10.1126/science.aad3000) |
| Literature and MCP context | Route five narrow evidence questions to scholarly retrieval and an optional administrator-configured MCP tool; never replace the stage validator | [Crossref REST API](https://www.crossref.org/documentation/retrieve-metadata/rest-api/), [arXiv API](https://info.arxiv.org/help/api/user-manual.html), [OpenAlex API](https://developers.openalex.org/) |

The code also leaves room for alternative or complementary generators such as [DiffCSP](https://proceedings.neurips.cc/paper_files/paper/2023/hash/38b787fc530d0b31825827e2cc306656-Abstract-Conference.html), [FlowMM](https://proceedings.mlr.press/v235/miller24a.html), and [SCIGEN](https://www.nature.com/articles/s41563-025-02355-y). They are research references, not bundled or silently selected runtime backends. Raw sample count is never reported as discovery success: the pipeline retains the generation, parsing, applicability, identity, geometry, MLIP, relaxation, and ranking funnel. [LeMat-GenBench](https://arxiv.org/abs/2512.04562) reports that no evaluated generator dominates stability, novelty, and diversity together and documents trade-offs among them. That evidence supports keeping stability, Pareto objectives, property-space diversity, and externally scoped structural novelty as separate signals instead of collapsing them into one score. It does not validate any candidate from this repository.

## 1. MatterGen contract and applicability envelope

The adapter follows the official MatterGen checkpoint inventory rather than assuming every checkpoint supports every condition. The released contracts represented in the code are:

- unconditional: `mattergen_base`, `mp_20_base`;
- one condition: `chemical_system`, `space_group`, `dft_mag_density`, `dft_band_gap`, or `ml_bulk_modulus`;
- released multi-condition checkpoints: `dft_mag_density_hhi_score` and `chemical_system_energy_above_hull`.

An unknown or custom checkpoint is unconditional unless the operator explicitly supplies `supported_condition_names`. The adapter checks only that this allowlist uses its known MatterGen condition vocabulary and records `condition_contract_source="explicit-custom-checkpoint-declaration"`. It does not inspect the training run or prove that the declared modules were trained and packaged with those weights. The declaration is therefore operator attestation and must be reviewed together with the checkpoint inventory hash and training/configuration artifacts. A similarly named directory is not evidence that a condition was trained.

For conditioned generation, the adapter maps the normalized scheduler value to MatterGen classifier-free guidance as

~~~text
gamma = alpha * guidance_max
~~~

The default `guidance_max=4`, so `alpha=0.5` means `gamma=2`, the scale used in the official examples. Higher guidance generally strengthens condition adherence while reducing sampling breadth or realism. The MatterGen scheduler therefore increases `alpha` for condition-focused exploitation after sustained improvement and decreases it to broaden sampling after stagnation or structural collapse. This interpretation is generator-specific; it must not be transferred to a backend whose control has different semantics.

The released model card defines an applicability envelope based on primitive cells with at most 20 atoms and excludes noble gases, Tc, Pm, and elements with atomic number above 84. The adapter rejects candidates outside this envelope and records rejection counters. This is an applicability gate, not evidence that every in-envelope output is physically stable.

MatterGen v1 does not expose a parent-structure mutation operator or the generic `temperature`, `mutation_strength`, and `diversity_strength` controls used by the common generator contract. Those values remain requested-but-ignored provenance. The applied condition dictionary, seed, batch controls, actual guidance factor, checkpoint inventory hash, and rejection funnel are retained separately.

## 2. Crystal identity is not one Boolean

The implementation keeps two representations with different purposes. A symmetry-standardized primitive structure and `canonical_structure_hash` provide prototype and symmetry context. Hard deletion instead uses `identity_structure_hash`, built by Niggli-reducing the parsed source structure without symmetry refinement, followed by a strict unscaled matcher. This prevents a loose symmetry tolerance from idealizing a genuine displacement before the duplicate decision.

The typed pair-comparison API exposes four relations: `strict_material_duplicate`, `scaled_same_prototype`, `ambiguous`, and `distinct`.

- Strict duplicates use species-preserving `StructureMatcher` comparison without lattice scaling and a symmetric relative-volume guard. Only this relation can remove a candidate from the hard-duplicate pool.
- A match that requires volume normalization is recorded as `scaled_same_prototype`. Both candidates are retained because a substantially different cell volume may encode a physically important difference or a failed generation/relaxation state.
- Direction-dependent or failed comparisons are `ambiguous` and fail closed: neither candidate is deleted.
- Exact-file hashes remain integrity identifiers, not crystallographic identity tests.

Supercell choices, atom ordering, and origin choices may still resolve to a strict duplicate when the matcher establishes equivalence. A typed pair assessment returns both matcher configurations and the relative-volume difference; hard grouping returns its strict matcher settings and any ambiguous pairs. Callers that publish the decision should persist those returned fields.

MatterGen applies that strict matcher within each generated call and, when a search session ID is supplied, against previously accepted structures before expert evaluation; it requests replacement samples for rejected duplicates. The coordinator also recognizes an exact repeated, adapter-attested raw identity hash after evaluation as a defensive fallback. That fallback preserves lineage but removes the duplicate from selection; it cannot replace the sidecar's tolerance-aware comparison. A final notebook grouping is a separately persisted audit over the collected structures.

Database absence is also not novelty. The selector branch whose compatibility ID is `novelty` computes only normalized expert-property-space diversity. Structural/database novelty is a separate staged assessment that must record the provider, client version, database snapshot or release, request/query identity, retrieval date, and structure-matching policy. Materials Project [`MPRester.find_structure`](https://materialsproject.github.io/api/_autosummary/mp_api.client.mprester.MPRester.html) uses relatively loose similarity tolerances, so returned IDs are only prefilter candidates; the adapter fetches them and requires the local strict unscaled relation before recording a hard match. With multiple providers, `match` is returned if any provider has a strict match, `no_match` only if every configured provider returns scoped no-match, and otherwise `unknown`. The [Materials Project database version](https://docs.materialsproject.org/changes/database-versions) can change independently of client code; the [Crystallography Open Database](https://academic.oup.com/nar/article/40/D1/D420/2903497) is another useful but non-exhaustive source. A failed, skipped, rate-limited, unresolved, or unconfigured lookup remains `unknown`. Even a completed scoped no-match is not proof of universal novelty.

## 3. Composition-scoped Pareto screening

Absolute MLIP energies contain model- and composition-dependent offsets. The selector therefore never lets a raw total energy or energy-per-atom value from one reduced composition dominate a candidate from another. If a periodic candidate has no unambiguous composition scope, raw energy is excluded from that comparison.

Inside each reduced-composition pool, Pareto dominance uses separate MatterSim energy, CHGNet energy, and both force-envelope values. Structural gates and disagreement remain later ordering and escalation evidence; they are not averaged into those axes. Non-dominated sorting is followed by normalized NSGA-II crowding within each composition and Pareto front. Boundary candidates are retained to preserve the envelope; crowding is a diversity tie-breaker, not a probability of success. The separate legacy `novelty` branch is also only a property-space diversity heuristic. Neither quantity is a crystallographic or database novelty score.

The DFT shortlist first covers distinct reduced compositions, then fills remaining slots by ranked evidence while reserving a disagreement-escalation candidate when possible. A later portfolio step may reserve at most one slot for an otherwise eligible candidate whose configured external providers all produced a strict scoped no-match. `unknown`, skipped, failed, or partially resolved novelty receives no portfolio credit, and a scoped no-match remains only a prioritization signal, never proof of universal novelty. This creates a portfolio of hypotheses without pretending that raw energies are comparable across stoichiometries. Formation energies and energy above hull require a consistent reference set and correction/mixing policy; see the [Materials Project phase-diagram method](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/phase-diagrams-pds) and [GGA/GGA+U/r2SCAN mixing documentation](https://docs.materialsproject.org/methodology/materials-methodology/thermodynamic-stability/thermodynamic-stability/gga-gga%2Bu-r2scan-mixing).

## 4. Calibrated reliability and fail-closed OOD handling

MatterSim and CHGNet are independent proxy models ([MatterSim model card](https://github.com/microsoft/mattersim/blob/main/MODEL_CARD.md), [CHGNet paper](https://www.nature.com/articles/s42256-023-00716-3)). Their agreement does not prove DFT accuracy, and their raw absolute-energy difference is not a calibrated uncertainty interval.

The trusted manifest fixes the default MatterSim 5M file to upstream revision `40a1eb8f1189a53af310957b4f2c5dfbfe68d647`, size `91,176,875`, SHA-256 `e3df9fa708725e3d453140646c7d1838324b347a3d1214cf1440522146f872b5`; it fixes CHGNet 0.3.0 to upstream revision `8d04abc467630b30cd8349c2fe12eeab10f62171`, size `4,863,221`, SHA-256 `d14ab7c0f093efe64b60a7bcd540bca10e74fb7f46c86108a079af60524659d1`. These are project-measured hashes of files served by immutable official source revisions, not upstream signatures. Bootstrap verifies size and digest, writes an artifact marker, and the launcher rehashes the bytes before it binds a `sha256:<digest>` weight revision. A changed file, marker, URL, revision, size, or digest fails closed. This artifact identity improves reproducibility; it does not establish model accuracy on a new chemistry.

`discovery_os.mlip_reliability` exposes a stricter optional contract:

1. A split-conformal artifact retains all calibration residuals and the finite-sample order statistic `ceil((n + 1) * (1 - alpha))`.
2. The artifact is bound to exact expert weight revisions, DFT method and reference hashes, units, held-out diagnostics, and a declared chemistry/exchangeability scope.
3. A candidate receives an interval only if every applicability key matches. Missing calibration, changed weights or labels, unmatched chemistry, a unit mismatch, or an uncalibrated property returns `uncalibrated_or_ood`, exposes no interval or coverage claim, and requests DFT escalation.
4. Cross-model energy disagreement removes each model's composition-local offset and compares relative energies and ranks only among aligned structures of the same reduced composition. A singleton pool or missing alignment attestation remains unknown.
5. Force disagreement retains component RMSE, per-atom vector RMS, and maximum atom-vector difference so localized force failures are not hidden by one global mean.

No calibration dataset or universal coverage guarantee is bundled. A project must build and validate its own versioned calibration artifact against the intended chemistry and reference calculation policy. Committee disagreement is useful for selecting follow-up calculations ([committee disagreement study](https://doi.org/10.1063/5.0016004)), but high or low disagreement is an escalation signal, not a physical property.

Recent [flexible uncertainty calibration work](https://www.nature.com/articles/s41524-026-02080-3) further motivates reporting calibration diagnostics and the applicable data/model scope rather than treating one calibration number as universal coverage. The repository therefore fails closed on weight, label, unit, method, chemistry, or exchangeability-scope mismatch.

## 5. DFT escalation and input preparation

The portable backend prepares CIF, POSCAR, a Quantum ESPRESSO `pw.in` skeleton, and a manifest. It does not run DFT and cannot emit an energy, convergence result, hull value, or phonon conclusion.

When no explicit grid is supplied, the starting grid is computed from reciprocal-vector lengths using a target spacing of `0.30 A^-1`:

~~~text
n_i = max(1, ceil(|b_i| / 0.30 A^-1))
~~~

This is an auditable initial sampling rule, not a converged k-point setting. The manifest requires k-point-spacing sweeps and target changes in energy, force, and stress. Plane-wave cutoffs remain unresolved placeholders unless both `ecutwfc` and `ecutrho` were explicitly reviewed and supplied for the selected external pseudopotentials. No pseudopotential file or credential is bundled.

An executed result can be marked completed only with the exact input-manifest hash, immutable output artifacts, immutable convergence-evidence artifacts, a method-policy hash, and explicit `converged=true`. A completed relaxation or static calculation must include energy per atom. Formation or hull values additionally require a reference-set hash, and hull requires formation energy. Phonon results require the q mesh, minimum frequency, imaginary-mode tolerance, and a classification consistent with those numbers. Failed results cannot expose completed scientific values or convergence artifacts. Magnetic order, electronic state, functional/U, pseudopotentials, reference phases, and convergence must be reviewed for each chemistry; see the [Materials Project convergence guidance](https://docs.materialsproject.org/methodology/materials-methodology/calculation-details/gga%2Bu-calculations/parameters-and-convergence) and [magnetic ground-state study](https://doi.org/10.1038/s41524-019-0199-7).

## 6. Five bounded RAG/MCP evidence routes

RAG and an optional MCP source provide stage-specific context only. They do not calculate or validate candidate properties. RAG is the scholarly retrieval and evidence-closing path. MCP is an administrator-configured structured evidence-provider interface with a bounded tool contract; it is not an implicit action executor. The actual actions remain with the named runtime validator: generator or expert sidecars, the structure matcher, external structure clients, relaxation endpoints, and an explicitly configured periodic DFT backend.

| Stage | Retrieved context | Runtime authority |
|---|---|---|
| `generation_prior` | reported phases, failed synthesis, composition ranges, stability constraints | exact MatterGen condition allowlist and deterministic Fusion controller |
| `identity_novelty` | reported phases, aliases, scoped database context | strict crystal matcher and configured external structure lookup |
| `mlip_disagreement` | applicability, magnetism, charge state, published calculations | separate, unit-normalized MatterSim and CHGNet results; optional calibrated reliability artifact |
| `relaxation_validation` | transformations, pressure/temperature, instability and phonon context | separately executed MLIP relaxations, optimizer convergence, and geometry gates |
| `dft_handoff` | reference phases, magnetism, functional/U, pseudopotential and convergence choices | an executing periodic DFT backend; an input package alone is not execution |

Each stage selects only its administrator-configured tool name and verifies the bounded MCP `tools/list` and schema contract. Prompts, model output, and retrieved records cannot choose an endpoint or tool. Failed, skipped, empty, or invalid sources remain unknown. Only a source-closed `generation_prior` branch may influence generation; later-stage evidence can recommend checks but cannot steer the generator. See [Stage-specific validation evidence](STAGE_VALIDATION_EVIDENCE.md) and [MCP evidence sources](MCP_RAG.md).

## 7. Multi-round search and Colab budget

The T4 notebook uses the same `fusion-search` engine as the CLI. It is a true closed screening loop:

~~~text
generate batch -> deduplicate -> evaluate experts -> update branch pools
-> adapt controls -> generate from retained frontiers -> repeat
~~~

A global `SearchBudget` caps both generator calls and generated candidates across all rounds, branches, frontiers, and control variants. The persisted report records requested limits, actual usage, round history, failures, scheduler decisions, candidate lineage, and whether a limit was exhausted. Rejection, sidecar failure, deduplication, or frontier exhaustion can produce fewer valid unique candidates than the requested budget; the report never fabricates replacements, and the T4 MAXIMUM workflow treats a final 8-32 unique-candidate shortfall as a failed run rather than a successful discovery result.

The loop is adaptive screening, not active learning: it does not retrain MatterGen, MatterSim, or CHGNet. Search controls change from observed screening evidence, while every generated candidate is independently re-evaluated by the configured experts. Literature context remains bounded generation prior or validation context, never a learned reward.

## Reproducibility requirements

Every publishable run should retain:

- source-code revision, Python/package environment, accelerator, random seeds, and complete run configuration;
- model ID, exact checkpoint/weight revision or inventory hash, license acceptance, requested/applied/ignored controls, and an explicit marker when custom condition support is only operator-attested;
- original expert payloads, units, failure states, cache keys, and candidate lineage;
- structure matcher settings and relation, not only a deduplicated count;
- RAG query bounds, source identifiers, retrieval timestamps, source failures, MCP contract result, and evidence-bundle hashes;
- API client and server version when available, plus the external database snapshot/release; and
- DFT method-policy, pseudopotential, convergence, magnetic/electronic state, and reference-set hashes for every executed result.

Credentials, private candidates, model weights, downloaded databases, and local run artifacts must stay outside Git. A moving API or database with no recorded snapshot cannot support an exact novelty or benchmark reproduction claim.

## Further primary references

- Large-scale stable-material discovery: [GNoME](https://www.nature.com/articles/s41586-023-06735-9)
- Autonomous synthesis and characterization: [A-Lab](https://www.nature.com/articles/s41586-023-06734-w)
- Thermodynamic metastability in experimentally reported inorganic crystals: [Sun et al.](https://doi.org/10.1126/sciadv.1600225)
- Critical analysis of computational stability prediction: [Bartel et al.](https://doi.org/10.1038/s41524-020-00362-y)
- Universal-MLIP softening during relaxation: [systematic study](https://doi.org/10.1038/s41524-024-01500-6)
- Batch multi-objective Bayesian optimization: [qNEHVI](https://proceedings.neurips.cc/paper/2021/hash/11704817e347269b7254e744b5e22dac-Abstract.html)
