"""Closed scientific workflow policies for field-specific property evidence.

An external process receipt is only evidence that a process ran.  It is not
automatically evidence for the scientific property named by that process.
This module owns the exact method, condition, output-role, and scientific-gate
contract that must be satisfied before a specialist result can enter ranking.

The registry is deliberately closed.  A new validator, method, or claimed
property must be added in code and reviewed; arbitrary receipt strings cannot
grant score authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import Field, model_validator

from .hashing import stable_hash
from .schemas import Identifier, MaterialField, NonEmptyText, StrictSchema


class SpecialistMethodSpec(StrictSchema):
    """One explicitly reviewed method-family and implementation pairing."""

    method_family: Identifier
    method_id: Identifier


class SpecialistWorkflowPolicy(StrictSchema):
    """Code-owned contract for one field/property/validator tuple."""

    policy_id: Identifier
    policy_version: Identifier
    validator_contract_version: Identifier
    material_field: MaterialField
    property_name: Identifier
    validator_id: Identifier
    unit: NonEmptyText
    execution_kind: str = Field(
        pattern=r"^(numerical_simulation|experimental_measurement)$"
    )
    allowed_methods: list[SpecialistMethodSpec] = Field(min_length=1)
    required_condition_fields: list[Identifier] = Field(min_length=1)
    required_output_evidence_labels: list[Identifier] = Field(min_length=1)
    required_scientific_gate_ids: list[Identifier] = Field(min_length=1)
    rejected_shortcuts: list[NonEmptyText] = Field(min_length=1)

    @model_validator(mode="after")
    def _policy_is_closed_and_unique(self) -> "SpecialistWorkflowPolicy":
        methods = [
            (item.method_family, item.method_id) for item in self.allowed_methods
        ]
        if len(methods) != len(set(methods)):
            raise ValueError("specialist policy methods must be unique")
        for values, label in (
            (self.required_condition_fields, "condition fields"),
            (self.required_output_evidence_labels, "output evidence labels"),
            (self.required_scientific_gate_ids, "scientific gates"),
            (self.rejected_shortcuts, "rejected shortcuts"),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"specialist policy {label} must be unique")
        return self

    def permits_method(self, *, method_family: str, method_id: str) -> bool:
        return any(
            item.method_family == method_family and item.method_id == method_id
            for item in self.allowed_methods
        )


def _methods(*pairs: tuple[str, str]) -> list[SpecialistMethodSpec]:
    return [
        SpecialistMethodSpec(method_family=family, method_id=method_id)
        for family, method_id in pairs
    ]


def _policy(
    field: MaterialField,
    property_name: str,
    validator_id: str,
    unit: str,
    *,
    methods: tuple[tuple[str, str], ...],
    conditions: tuple[str, ...],
    outputs: tuple[str, ...],
    gates: tuple[str, ...],
    shortcuts: tuple[str, ...],
    experimental: bool = False,
) -> SpecialistWorkflowPolicy:
    return SpecialistWorkflowPolicy(
        policy_id=f"{field.value}-{property_name}-workflow",
        policy_version="1.0.0",
        validator_contract_version="1.0.0",
        material_field=field,
        property_name=property_name,
        validator_id=validator_id,
        unit=unit,
        execution_kind=(
            "experimental_measurement"
            if experimental
            else "numerical_simulation"
        ),
        allowed_methods=_methods(*methods),
        required_condition_fields=list(conditions),
        required_output_evidence_labels=list(outputs),
        required_scientific_gate_ids=list(gates),
        rejected_shortcuts=list(shortcuts),
    )


_POLICIES: tuple[SpecialistWorkflowPolicy, ...] = (
    # General inorganic crystals.
    _policy(
        MaterialField.GENERAL_INORGANIC,
        "energy_above_hull",
        "reference-phase-dft-and-phase-diagram",
        "eV/atom",
        methods=(
            ("compatible_reference_phase_dft", "aiida-common-workflow-phase-diagram"),
            ("compatible_reference_phase_dft", "periodic-dft-reference-phase-set"),
        ),
        conditions=("pressure", "temperature"),
        outputs=(
            "candidate_formation_energy",
            "reference_phase_energy_table",
            "phase_diagram_result",
        ),
        gates=(
            "candidate_and_references_share_method",
            "energy_cutoff_and_kmesh_converged",
            "reference_phase_set_audited",
            "phase_diagram_compatibility_passed",
        ),
        shortcuts=(
            "A standalone candidate total energy is not energy above hull.",
            "Energies from incompatible methods or reference sets cannot be mixed.",
        ),
    ),
    _policy(
        MaterialField.GENERAL_INORGANIC,
        "minimum_phonon_frequency",
        "phonon-stability-workflow",
        "THz",
        methods=(
            ("finite_displacement_phonons", "phonopy-finite-displacement"),
            ("density_functional_perturbation_theory", "dfpt-phonon-dispersion"),
        ),
        conditions=("pressure", "temperature"),
        outputs=("force_constants", "phonon_dispersion", "minimum_signed_frequency"),
        gates=(
            "supercell_or_qmesh_converged",
            "forces_or_dfpt_response_converged",
            "acoustic_sum_rule_audited",
            "non_analytical_correction_assessed",
        ),
        shortcuts=(
            "A relaxed geometry is not evidence of dynamical stability.",
            "Gamma-only frequencies cannot establish full Brillouin-zone stability.",
        ),
    ),
    # Battery electrodes.
    _policy(
        MaterialField.BATTERY_ELECTRODE,
        "average_voltage",
        "battery-reaction-phase-diagram",
        "V",
        methods=(
            ("battery_reaction_dft", "pymatgen-battery-compatible-dft"),
            ("battery_reaction_dft", "periodic-dft-reaction-endpoints"),
        ),
        conditions=("working_ion", "reference_electrode", "state_of_charge"),
        outputs=("endpoint_energy_table", "balanced_reaction", "voltage_profile"),
        gates=(
            "charged_and_discharged_endpoints_enumerated",
            "endpoint_energies_share_method",
            "working_ion_reference_bound",
            "state_of_charge_range_bound",
        ),
        shortcuts=(
            "One host formation energy is not a cell voltage.",
            "Voltage endpoints cannot be inferred from an unenumerated state of charge.",
        ),
    ),
    _policy(
        MaterialField.BATTERY_ELECTRODE,
        "specific_capacity",
        "battery-reaction-phase-diagram",
        "mAh/g",
        methods=(
            ("stoichiometric_redox_capacity", "pymatgen-battery-capacity"),
            ("stoichiometric_redox_capacity", "audited-redox-electron-count"),
        ),
        conditions=("working_ion", "cycling_protocol"),
        outputs=(
            "accessible_endpoint_compositions",
            "redox_electron_count",
            "active_material_mass_basis",
            "specific_capacity_result",
        ),
        gates=(
            "accessible_redox_endpoints_justified",
            "reaction_charge_and_mass_balanced",
            "active_material_mass_basis_audited",
            "cycling_protocol_bound",
        ),
        shortcuts=(
            "Theoretical electron count without accessible endpoint structures is not reversible capacity.",
            "Capacity cannot use an undeclared active-mass basis.",
        ),
    ),
    _policy(
        MaterialField.BATTERY_ELECTRODE,
        "ion_migration_barrier",
        "working-ion-neb-or-aimd",
        "eV",
        methods=(
            ("nudged_elastic_band", "ci-neb-working-ion"),
            ("finite_temperature_free_energy", "aimd-free-energy-barrier"),
        ),
        conditions=("working_ion", "state_of_charge", "temperature"),
        outputs=("migration_path", "path_energy_profile", "migration_barrier_result"),
        gates=(
            "endpoint_sites_and_charge_state_bound",
            "path_images_and_forces_converged",
            "path_connectivity_verified",
            "finite_size_and_temperature_assessed",
        ),
        shortcuts=(
            "A geometric bottleneck is not an ion-migration barrier.",
            "A barrier from another state of charge is not transferable by default.",
        ),
    ),
    # Solid electrolytes and ionic conductors.
    _policy(
        MaterialField.SOLID_ELECTROLYTE,
        "ionic_conductivity",
        "finite-temperature-ion-transport",
        "S/cm",
        methods=(
            ("ab_initio_molecular_dynamics", "aimd-ionic-transport"),
            ("uncertainty_audited_mlip_md", "mlip-md-ionic-transport"),
        ),
        conditions=("mobile_ion", "temperature", "microstructure"),
        outputs=(
            "equilibrated_trajectory",
            "mean_squared_displacement",
            "conductivity_estimate",
            "statistical_uncertainty",
        ),
        gates=(
            "equilibration_and_diffusive_regime_verified",
            "sampling_time_converged",
            "finite_size_effect_assessed",
            "ion_correlation_or_haven_ratio_assessed",
            "mlip_extrapolation_audited_when_used",
        ),
        shortcuts=(
            "One migration barrier is not bulk ionic conductivity.",
            "Short non-diffusive trajectories cannot authorize conductivity.",
        ),
    ),
    _policy(
        MaterialField.SOLID_ELECTROLYTE,
        "migration_barrier",
        "mobile-ion-path-neb",
        "eV",
        methods=(("nudged_elastic_band", "ci-neb-mobile-ion-path"),),
        conditions=("mobile_ion", "defect_concentration"),
        outputs=("site_network", "neb_path_energy_profile", "migration_barrier_result"),
        gates=(
            "mobile_ion_sites_and_defects_enumerated",
            "neb_images_and_forces_converged",
            "path_connectivity_verified",
            "cell_size_and_charge_compensation_audited",
        ),
        shortcuts=(
            "A static pore radius is not a migration barrier.",
            "One path cannot establish network transport unless the site network is enumerated.",
        ),
    ),
    _policy(
        MaterialField.SOLID_ELECTROLYTE,
        "electrochemical_stability_window",
        "electrode-interface-grand-potential",
        "V",
        methods=(
            ("grand_potential_phase_diagram", "grand-potential-interface-dft"),
            ("grand_potential_phase_diagram", "compatible-electrochemical-window"),
        ),
        conditions=("electrode_pair", "temperature"),
        outputs=(
            "grand_potential_phase_diagram",
            "decomposition_reaction_set",
            "interface_reaction_energies",
            "stability_window_result",
        ),
        gates=(
            "reference_chemical_potentials_bound",
            "competing_phases_and_decomposition_enumerated",
            "both_electrode_interfaces_assessed",
            "finite_temperature_treatment_declared",
        ),
        shortcuts=(
            "A bulk band gap is not an electrochemical stability window.",
            "Thermodynamic window without electrode-interface reactions is incomplete.",
        ),
    ),
    # Superconductors.
    _policy(
        MaterialField.SUPERCONDUCTOR,
        "critical_temperature",
        "epw-or-eliashberg-workflow",
        "K",
        methods=(
            ("anisotropic_eliashberg", "epw-anisotropic-eliashberg"),
            ("conventional_epc_tc", "converged-alpha2f-tc"),
        ),
        conditions=("pressure", "magnetic_field", "isotope"),
        outputs=(
            "eliashberg_spectral_function",
            "coulomb_parameter_or_kernel",
            "gap_equation_solution",
            "critical_temperature_result",
        ),
        gates=(
            "conventional_mechanism_scope_justified",
            "electronic_kmesh_and_phonon_qmesh_converged",
            "dynamical_stability_at_pressure_passed",
            "eliashberg_or_epc_solver_converged",
            "field_and_isotope_conditions_bound",
        ),
        shortcuts=(
            "Metallicity or density of states alone is not a critical temperature.",
            "A regression-only Tc estimate is not an Eliashberg validation receipt.",
        ),
    ),
    _policy(
        MaterialField.SUPERCONDUCTOR,
        "electron_phonon_coupling",
        "epw-or-eliashberg-workflow",
        "dimensionless",
        methods=(
            ("wannier_interpolated_electron_phonon", "epw-electron-phonon-coupling"),
            ("density_functional_perturbation_theory", "dfpt-alpha2f-lambda"),
        ),
        conditions=("pressure",),
        outputs=(
            "phonon_dispersion",
            "eliashberg_spectral_function",
            "electron_phonon_coupling_result",
        ),
        gates=(
            "electronic_kmesh_and_phonon_qmesh_converged",
            "wannier_or_dfpt_interpolation_audited",
            "dynamical_stability_at_pressure_passed",
            "smearing_and_fine_grid_converged",
        ),
        shortcuts=(
            "A large density of states does not establish electron-phonon coupling.",
        ),
    ),
    # Heterogeneous catalysts.
    _policy(
        MaterialField.HETEROGENEOUS_CATALYST,
        "reaction_free_energy",
        "surface-adsorbate-free-energy-workflow",
        "eV",
        methods=(
            ("surface_thermochemistry", "converged-slab-free-energy"),
            ("electrocatalysis_thermochemistry", "computational-hydrogen-electrode"),
        ),
        conditions=(
            "reaction",
            "facet",
            "coverage",
            "temperature",
            "pressure",
            "electrode_potential",
            "ph",
        ),
        outputs=(
            "converged_slab_and_adsorbate_set",
            "thermochemical_corrections",
            "reaction_free_energy_diagram",
        ),
        gates=(
            "facet_slab_vacuum_and_kmesh_converged",
            "adsorption_sites_and_coverage_enumerated",
            "reference_states_and_stoichiometry_audited",
            "entropy_solvation_field_and_potential_treatment_declared",
        ),
        shortcuts=(
            "Bulk stability or an adsorbate geometry is not a reaction free energy.",
            "One adsorption energy is not a full reaction free-energy diagram.",
        ),
    ),
    _policy(
        MaterialField.HETEROGENEOUS_CATALYST,
        "activation_barrier",
        "transition-state-and-microkinetic-workflow",
        "eV",
        methods=(
            ("transition_state_search", "ci-neb-surface-reaction"),
            ("transition_state_search", "dimer-surface-transition-state"),
        ),
        conditions=("reaction", "facet", "coverage", "temperature"),
        outputs=(
            "reactant_and_product_states",
            "transition_state_structure",
            "minimum_energy_path",
            "activation_barrier_result",
        ),
        gates=(
            "transition_state_has_one_relevant_imaginary_mode",
            "path_connects_declared_endpoints",
            "transition_state_forces_converged",
            "facet_coverage_and_temperature_bound",
        ),
        shortcuts=(
            "Reaction free energy is not an activation barrier.",
            "An unconverged NEB maximum is not a verified transition state.",
        ),
    ),
    _policy(
        MaterialField.HETEROGENEOUS_CATALYST,
        "durability",
        "operando-durability-validation",
        "h",
        methods=(
            ("operando_durability_experiment", "application-specific-operando-aging"),
            ("accelerated_stress_test", "controlled-catalyst-stress-test"),
        ),
        conditions=(
            "reaction",
            "temperature",
            "pressure",
            "electrode_potential",
            "ph",
        ),
        outputs=(
            "time_resolved_activity",
            "degradation_endpoint_definition",
            "pre_and_post_characterization",
            "durability_result",
        ),
        gates=(
            "instrument_calibration_and_controls_passed",
            "exposure_protocol_and_duration_bound",
            "degradation_criterion_predeclared",
            "replicates_and_uncertainty_reported",
        ),
        shortcuts=(
            "A static surface calculation is not an operando durability lifetime.",
            "Activity at one time point is not durability.",
        ),
        experimental=True,
    ),
    # Semiconductors.
    _policy(
        MaterialField.SEMICONDUCTOR,
        "band_gap",
        "hybrid-or-gw-electronic-structure",
        "eV",
        methods=(
            ("hybrid_functional_electronic_structure", "converged-hybrid-band-structure"),
            ("gw_quasiparticle_electronic_structure", "converged-gw-band-gap"),
        ),
        conditions=("temperature", "strain"),
        outputs=("quasiparticle_or_hybrid_eigenvalues", "band_extrema", "band_gap_result"),
        gates=(
            "basis_kmesh_and_empty_bands_converged",
            "direct_or_indirect_gap_identified",
            "spin_orbit_coupling_assessed",
            "temperature_and_strain_treatment_bound",
        ),
        shortcuts=(
            "A semilocal-DFT screening gap cannot masquerade as a converged hybrid or GW gap.",
        ),
    ),
    _policy(
        MaterialField.SEMICONDUCTOR,
        "carrier_mobility",
        "charged-defect-and-transport-workflow",
        "cm^2/(V s)",
        methods=(
            ("electron_phonon_transport", "epw-carrier-mobility"),
            ("scattering_aware_boltzmann_transport", "amset-carrier-mobility"),
        ),
        conditions=("carrier_type", "temperature", "doping"),
        outputs=("scattering_rates", "transport_distribution", "mobility_tensor"),
        gates=(
            "electron_phonon_or_scattering_model_explicit",
            "kmesh_and_scattering_converged",
            "carrier_temperature_and_doping_bound",
            "mobility_not_effective_mass_only",
        ),
        shortcuts=(
            "Effective mass alone is not carrier mobility.",
            "A constant relaxation time cannot authorize absolute mobility.",
        ),
    ),
    _policy(
        MaterialField.SEMICONDUCTOR,
        "minimum_native_defect_formation_energy",
        "charged-defect-and-transport-workflow",
        "eV",
        methods=(
            ("charged_defect_supercell", "doped-charged-defect-workflow"),
            ("charged_defect_supercell", "pydefect-formation-energy"),
        ),
        conditions=("fermi_level", "chemical_potentials", "charge_state"),
        outputs=(
            "defect_configuration_set",
            "finite_size_corrections",
            "chemical_potential_region",
            "defect_formation_energy_diagram",
        ),
        gates=(
            "native_defects_and_charge_states_enumerated",
            "supercell_and_finite_size_correction_converged",
            "competing_phase_chemical_potentials_bound",
            "fermi_level_and_charge_state_bound",
        ),
        shortcuts=(
            "A neutral defect in one small cell is not a native-defect formation-energy landscape.",
        ),
    ),
    # Photovoltaic absorbers.
    _policy(
        MaterialField.PHOTOVOLTAIC_ABSORBER,
        "optical_absorption_coefficient",
        "quasiparticle-optics-and-slme",
        "cm^-1",
        methods=(
            ("many_body_optics", "gw-bse-absorption-spectrum"),
            ("calibrated_independent_particle_optics", "hybrid-optical-spectrum"),
        ),
        conditions=("photon_energy", "polarization", "temperature"),
        outputs=("electronic_structure", "dielectric_function", "absorption_spectrum"),
        gates=(
            "kmesh_bands_and_broadening_converged",
            "direct_indirect_and_soc_effects_assessed",
            "excitonic_treatment_justified",
            "photon_energy_polarization_temperature_bound",
        ),
        shortcuts=(
            "Band gap alone is not an absorption coefficient.",
        ),
    ),
    _policy(
        MaterialField.PHOTOVOLTAIC_ABSORBER,
        "slme",
        "quasiparticle-optics-and-slme",
        "fraction",
        methods=(
            ("spectroscopic_limited_maximum_efficiency", "yu-zunger-slme"),
            ("spectroscopic_limited_maximum_efficiency", "audited-slme-integration"),
        ),
        conditions=("absorber_thickness", "temperature"),
        outputs=(
            "converged_absorption_spectrum",
            "direct_and_fundamental_gap",
            "radiative_fraction_model",
            "slme_result",
        ),
        gates=(
            "absorption_spectrum_provenance_verified",
            "thickness_dependent_absorptivity_integrated",
            "direct_indirect_gap_and_radiative_fraction_bound",
            "temperature_bound",
            "band_gap_only_shortcut_rejected",
        ),
        shortcuts=(
            "Band gap alone is never sufficient evidence for SLME.",
            "Shockley-Queisser efficiency from a scalar gap is not SLME.",
        ),
    ),
    _policy(
        MaterialField.PHOTOVOLTAIC_ABSORBER,
        "nonradiative_recombination_rate",
        "photovoltaic-defect-interface-workflow",
        "s^-1",
        methods=(
            ("nonradiative_multiphonon_capture", "defect-capture-recombination"),
            ("interface_recombination", "explicit-interface-recombination"),
        ),
        conditions=(
            "chemical_potentials",
            "contacts",
            "temperature",
            "carrier_concentration",
        ),
        outputs=(
            "defect_or_interface_state_set",
            "electron_phonon_capture_parameters",
            "capture_coefficient",
            "recombination_rate_result",
        ),
        gates=(
            "defects_interfaces_and_charge_states_enumerated",
            "finite_size_and_configuration_coordinate_converged",
            "chemical_potentials_and_contacts_bound",
            "temperature_and_carrier_concentration_bound",
        ),
        shortcuts=(
            "A band gap or defect level alone is not a nonradiative recombination rate.",
        ),
    ),
    # Thermoelectrics.
    _policy(
        MaterialField.THERMOELECTRIC,
        "power_factor",
        "electronic-boltzmann-transport",
        "W/(m K^2)",
        methods=(
            ("scattering_aware_boltzmann_transport", "amset-power-factor"),
            ("electron_phonon_transport", "epw-power-factor"),
            ("calibrated_relaxation_time_transport", "experiment-calibrated-boltztrap2"),
        ),
        conditions=("temperature", "carrier_concentration", "carrier_type"),
        outputs=(
            "seebeck_coefficient",
            "absolute_electrical_conductivity",
            "scattering_time_or_rates",
            "power_factor_result",
        ),
        gates=(
            "transport_kmesh_converged",
            "temperature_carrier_density_and_type_bound",
            "absolute_scattering_model_calculated_or_calibrated",
            "constant_tau_only_shortcut_rejected",
        ),
        shortcuts=(
            "Constant-relaxation-time output sigma/tau is not absolute conductivity or power factor.",
            "A favorable band shape is not a power factor.",
        ),
    ),
    _policy(
        MaterialField.THERMOELECTRIC,
        "lattice_thermal_conductivity",
        "anharmonic-phonon-transport",
        "W/(m K)",
        methods=(
            ("phonon_boltzmann_transport", "phono3py-lattice-thermal-conductivity"),
            ("phonon_boltzmann_transport", "shengbte-lattice-thermal-conductivity"),
        ),
        conditions=("temperature", "microstructure"),
        outputs=(
            "second_order_force_constants",
            "third_order_force_constants",
            "phonon_lifetimes",
            "lattice_thermal_conductivity_tensor",
        ),
        gates=(
            "harmonic_and_anharmonic_supercells_converged",
            "phonon_qmesh_and_cutoff_converged",
            "dynamic_stability_passed",
            "isotope_boundary_and_microstructure_treatment_bound",
        ),
        shortcuts=(
            "Harmonic frequencies alone are not lattice thermal conductivity.",
        ),
    ),
    _policy(
        MaterialField.THERMOELECTRIC,
        "zt",
        "thermoelectric-zt-integration",
        "dimensionless",
        methods=(("audited_transport_integration", "co-conditioned-zt-integration"),),
        conditions=("temperature", "carrier_concentration", "microstructure"),
        outputs=(
            "seebeck_coefficient",
            "absolute_electrical_conductivity",
            "electronic_thermal_conductivity",
            "lattice_thermal_conductivity",
            "zt_result",
        ),
        gates=(
            "electronic_and_lattice_inputs_share_conditions",
            "absolute_scattering_model_verified",
            "thermal_conductivity_components_independently_validated",
            "units_and_zt_identity_audited",
        ),
        shortcuts=(
            "ZT cannot combine quantities from incompatible temperature, doping, or microstructure.",
            "Constant-tau transport cannot authorize absolute ZT.",
        ),
    ),
    # Magnetic materials.
    _policy(
        MaterialField.MAGNETIC_MATERIAL,
        "magnetic_ordering_energy",
        "magnetic-order-and-correlation-workflow",
        "eV/atom",
        methods=(
            ("magnetic_order_enumeration_dft", "spin-polarized-order-enumeration"),
            ("correlated_magnetic_dft", "dft-u-hybrid-order-enumeration"),
        ),
        conditions=("temperature", "magnetic_field"),
        outputs=("magnetic_configuration_set", "configuration_energies", "ground_order_result"),
        gates=(
            "multiple_symmetry_distinct_orders_enumerated",
            "spin_and_electronic_convergence_passed",
            "oxidation_spin_and_correlation_settings_justified",
            "all_orders_share_numerical_settings",
        ),
        shortcuts=(
            "One ferromagnetic initialization cannot establish the magnetic ground state.",
        ),
    ),
    _policy(
        MaterialField.MAGNETIC_MATERIAL,
        "magnetocrystalline_anisotropy",
        "soc-anisotropy-exchange-temperature-workflow",
        "MJ/m^3",
        methods=(
            ("spin_orbit_anisotropy_dft", "soc-total-energy-anisotropy"),
            ("spin_orbit_anisotropy_dft", "soc-force-theorem-anisotropy"),
        ),
        conditions=("temperature",),
        outputs=("orientation_energy_table", "magnetic_volume", "anisotropy_result"),
        gates=(
            "spin_orbit_coupling_enabled",
            "dense_kmesh_and_energy_difference_converged",
            "magnetization_orientations_enumerated",
            "temperature_interpretation_bound",
        ),
        shortcuts=(
            "A magnetic moment without spin-orbit directional energies is not anisotropy.",
        ),
    ),
    _policy(
        MaterialField.MAGNETIC_MATERIAL,
        "ordering_temperature",
        "soc-anisotropy-exchange-temperature-workflow",
        "K",
        methods=(
            ("finite_temperature_spin_model", "exchange-plus-monte-carlo"),
            ("finite_temperature_spin_model", "exchange-plus-spin-dynamics"),
        ),
        conditions=("magnetic_field",),
        outputs=(
            "exchange_parameter_set",
            "spin_hamiltonian",
            "finite_temperature_observables",
            "ordering_temperature_result",
        ),
        gates=(
            "exchange_parameters_fit_to_multiple_orders",
            "spin_model_and_anisotropy_declared",
            "finite_size_equilibration_and_sampling_converged",
            "ordering_transition_extracted_with_uncertainty",
        ),
        shortcuts=(
            "A 0 K ordering-energy difference is not an ordering temperature.",
            "Ordering temperature requires exchange extraction and finite-temperature statistics.",
        ),
    ),
    # Ferroelectrics and piezoelectrics.
    _policy(
        MaterialField.FERROELECTRIC_PIEZOELECTRIC,
        "spontaneous_polarization",
        "berry-phase-switching-workflow",
        "C/m^2",
        methods=(("berry_phase_polarization", "modern-polarization-difference"),),
        conditions=("orientation", "temperature"),
        outputs=(
            "nonpolar_reference_structure",
            "polarization_branch_path",
            "berry_phase_polarization_difference",
        ),
        gates=(
            "polar_and_reference_states_insulating",
            "nonpolar_reference_justified",
            "polarization_branch_continuity_resolved",
            "orientation_and_temperature_bound",
        ),
        shortcuts=(
            "A polar space group is not spontaneous or switchable polarization.",
            "One absolute Berry phase without a reference path is not polarization.",
        ),
    ),
    _policy(
        MaterialField.FERROELECTRIC_PIEZOELECTRIC,
        "switching_barrier",
        "berry-phase-switching-workflow",
        "eV per formula unit",
        methods=(
            ("polarization_switching_path", "neb-ferroelectric-switching"),
            ("polarization_switching_path", "constrained-polar-distortion-path"),
        ),
        conditions=("electric_field", "stress"),
        outputs=("opposite_polar_endpoints", "switching_path", "path_band_gaps", "switching_barrier_result"),
        gates=(
            "path_connects_symmetry_related_polar_states",
            "all_path_images_remain_insulating",
            "path_forces_or_constraint_converged",
            "electric_field_and_stress_bound",
        ),
        shortcuts=(
            "Structural polarity without a switching path is not a switching barrier.",
        ),
    ),
    _policy(
        MaterialField.FERROELECTRIC_PIEZOELECTRIC,
        "piezoelectric_strain_coefficient",
        "dfpt-polar-response-workflow",
        "pm/V",
        methods=(
            ("density_functional_perturbation_theory", "dfpt-piezoelectric-tensor"),
            ("finite_difference_polar_response", "finite-difference-piezoelectric-tensor"),
        ),
        conditions=("orientation", "tensor_component", "temperature", "stress"),
        outputs=(
            "dielectric_tensor",
            "born_effective_charges",
            "elastic_tensor",
            "piezoelectric_tensor",
            "selected_strain_coefficient",
        ),
        gates=(
            "response_cutoff_kmesh_and_qmesh_converged",
            "crystal_symmetry_and_tensor_component_resolved",
            "elastic_mechanical_stability_passed",
            "orientation_temperature_and_stress_bound",
        ),
        shortcuts=(
            "A polar structure alone is not a piezoelectric strain coefficient.",
        ),
    ),
    # Structural and high-temperature alloys.
    _policy(
        MaterialField.STRUCTURAL_ALLOY,
        "mixing_gibbs_free_energy",
        "finite-temperature-alloy-thermodynamics",
        "eV/atom",
        methods=(
            ("cluster_expansion_thermodynamics", "cluster-expansion-monte-carlo"),
            ("calphad_thermodynamics", "validated-calphad-assessment"),
        ),
        conditions=("composition_range", "temperature", "processing_history"),
        outputs=(
            "configuration_or_phase_model",
            "finite_temperature_free_energy_terms",
            "composition_temperature_phase_result",
            "mixing_gibbs_free_energy_result",
        ),
        gates=(
            "configuration_or_phase_space_enumerated",
            "cluster_expansion_cross_validation_or_calphad_assessment_passed",
            "configurational_vibrational_and_magnetic_terms_assessed",
            "composition_temperature_and_processing_bound",
        ),
        shortcuts=(
            "One ordered 0 K cell is not finite-temperature alloy Gibbs free energy.",
        ),
    ),
    _policy(
        MaterialField.STRUCTURAL_ALLOY,
        "youngs_modulus",
        "elastic-defect-and-service-workflow",
        "GPa",
        methods=(
            ("finite_strain_elasticity", "converged-elastic-tensor"),
            ("stress_strain_elasticity", "temperature-resolved-elastic-tensor"),
        ),
        conditions=("temperature", "orientation", "microstructure"),
        outputs=("stress_strain_set", "elastic_tensor", "orientation_resolved_youngs_modulus"),
        gates=(
            "strain_amplitude_and_numerical_settings_converged",
            "elastic_tensor_symmetry_and_stability_passed",
            "orientation_transformation_audited",
            "temperature_and_microstructure_interpretation_bound",
        ),
        shortcuts=(
            "Bulk modulus is not orientation-resolved Young's modulus.",
            "0 K elasticity cannot be reused as service degradation or lifetime.",
        ),
    ),
    _policy(
        MaterialField.STRUCTURAL_ALLOY,
        "service_degradation_rate",
        "elastic-defect-and-service-workflow",
        "s^-1",
        methods=(
            ("service_reaction_kinetics", "surface-reaction-degradation-kinetics"),
            ("multiscale_service_degradation", "atomistic-mesoscale-degradation-rate"),
        ),
        conditions=("service_environment", "degradation_mechanism", "temperature", "time"),
        outputs=(
            "service_mechanism_model",
            "environment_dependent_free_energies",
            "kinetic_or_mesoscale_trajectory",
            "degradation_rate_result",
        ),
        gates=(
            "service_environment_and_mechanism_explicit",
            "finite_temperature_kinetics_or_transport_included",
            "time_window_and_rate_definition_bound",
            "model_validation_and_uncertainty_reported",
        ),
        shortcuts=(
            "0 K elastic constants do not establish corrosion, creep, oxidation, or degradation rate.",
            "A static reaction energy is not a time-normalized degradation rate.",
        ),
    ),
    # Porous frameworks.
    _policy(
        MaterialField.POROUS_FRAMEWORK,
        "accessible_volume_fraction",
        "probe-resolved-porosity-workflow",
        "dimensionless",
        methods=(
            ("periodic_probe_geometry", "zeopp-accessible-volume"),
            ("periodic_probe_geometry", "poreblazer-accessible-volume"),
        ),
        conditions=("guest_species", "activation_state"),
        outputs=("activated_structure", "probe_definition", "accessible_volume_result"),
        gates=(
            "solvent_disorder_and_occupancy_resolved",
            "activation_state_explicit",
            "guest_probe_radius_and_connectivity_bound",
            "periodic_geometry_converged",
        ),
        shortcuts=(
            "Crystallographic void volume without probe and activation assumptions is not accessible volume.",
        ),
    ),
    _policy(
        MaterialField.POROUS_FRAMEWORK,
        "adsorption_selectivity",
        "gcmc-mixture-adsorption-workflow",
        "dimensionless",
        methods=(
            ("grand_canonical_monte_carlo", "raspa-mixture-gcmc"),
            ("grand_canonical_monte_carlo", "validated-mixture-gcmc"),
        ),
        conditions=("guest_species", "temperature", "pressure", "humidity"),
        outputs=(
            "mixture_definition",
            "force_field_charge_and_flexibility_model",
            "component_uptake_isotherms",
            "adsorption_selectivity_result",
        ),
        gates=(
            "mixture_composition_and_thermodynamic_conditions_bound",
            "force_field_and_charges_validated",
            "framework_flexibility_and_humidity_assessed",
            "equilibration_sampling_and_uncertainty_converged",
            "geometry_only_shortcut_rejected",
        ),
        shortcuts=(
            "Pore geometry alone is not adsorption selectivity.",
            "A ratio of unrelated pure-component uptakes is not validated mixture selectivity.",
        ),
    ),
    _policy(
        MaterialField.POROUS_FRAMEWORK,
        "framework_decomposition_free_energy",
        "framework-stability-workflow",
        "eV/atom",
        methods=(
            ("framework_reaction_thermodynamics", "periodic-dft-framework-decomposition"),
            ("finite_temperature_framework_stability", "phonon-flexible-framework-stability"),
        ),
        conditions=("temperature", "humidity", "activation_state"),
        outputs=(
            "competing_product_set",
            "framework_and_product_free_energies",
            "dynamic_or_flexible_framework_evidence",
            "decomposition_free_energy_result",
        ),
        gates=(
            "decomposition_products_enumerated",
            "framework_and_products_share_method",
            "finite_temperature_dynamic_and_mechanical_stability_assessed",
            "humidity_and_activation_state_bound",
        ),
        shortcuts=(
            "Geometric porosity is not framework chemical or humidity stability.",
        ),
    ),
)


SpecialistWorkflowKey = tuple[MaterialField, str, str]

SPECIALIST_WORKFLOW_POLICIES: dict[
    SpecialistWorkflowKey, SpecialistWorkflowPolicy
] = {
    (
        MaterialField(str(policy.material_field)),
        policy.property_name,
        policy.validator_id,
    ): policy
    for policy in _POLICIES
}

if len(SPECIALIST_WORKFLOW_POLICIES) != len(_POLICIES):
    raise RuntimeError("duplicate specialist workflow policy registry key")


def specialist_workflow_policy(
    material_field: MaterialField | str,
    property_name: str,
    validator_id: str,
) -> SpecialistWorkflowPolicy:
    """Return a defensive copy of the exact reviewed workflow policy."""

    key = (MaterialField(str(material_field)), property_name, validator_id)
    try:
        policy = SPECIALIST_WORKFLOW_POLICIES[key]
    except KeyError as exc:
        raise ValueError(
            "no code-owned specialist workflow policy for "
            f"{key[0].value}/{property_name}/{validator_id}"
        ) from exc
    return SpecialistWorkflowPolicy.model_validate_json(
        policy.model_dump_json(),
        strict=True,
    )


def specialist_workflow_policy_sha256(
    policy: SpecialistWorkflowPolicy,
) -> str:
    """Canonical policy hash bound into every external workflow receipt."""

    return stable_hash(policy.model_dump(mode="json"))


def missing_specialist_condition_fields(
    policy: SpecialistWorkflowPolicy,
    conditions: Mapping[str, object],
) -> list[str]:
    """Return required scientific condition names that are absent or empty."""

    def missing(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict)):
            return not value
        return False

    return [
        name
        for name in policy.required_condition_fields
        if name not in conditions or missing(conditions[name])
    ]


def validate_specialist_policy_coverage(
    entries: Iterable[tuple[MaterialField | str, str, str]],
) -> None:
    """Raise if any declared score authority lacks an exact workflow policy."""

    for material_field, property_name, validator_id in entries:
        specialist_workflow_policy(material_field, property_name, validator_id)


__all__ = [
    "SPECIALIST_WORKFLOW_POLICIES",
    "SpecialistMethodSpec",
    "SpecialistWorkflowPolicy",
    "missing_specialist_condition_fields",
    "specialist_workflow_policy",
    "specialist_workflow_policy_sha256",
    "validate_specialist_policy_coverage",
]
