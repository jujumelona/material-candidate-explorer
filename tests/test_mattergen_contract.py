from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from discovery_os.fusion_schemas import GenerationControls
from discovery_os.sidecars.errors import ModelExecutionError
from discovery_os.sidecars.generators import MatterGenGenerator


def _condition_request(**targets: object) -> SimpleNamespace:
    canonical_units = {
        "dft_mag_density": "µB/Å^3",
        "dft_band_gap": "eV",
        "ml_bulk_modulus": "GPa",
        "energy_above_hull": "eV/atom",
    }
    return SimpleNamespace(
        goal=SimpleNamespace(
            objectives=[
                SimpleNamespace(
                    property_name=name,
                    target_value=value,
                    unit=canonical_units.get(name),
                )
                for name, value in targets.items()
            ]
        ),
        revision_proposal=None,
        parent_candidate=object(),
    )


def test_official_checkpoint_condition_allowlists_are_exact() -> None:
    expected = {
        "mattergen_base": frozenset(),
        "mp_20_base": frozenset(),
        "chemical_system": frozenset({"chemical_system"}),
        "space_group": frozenset({"space_group"}),
        "dft_mag_density": frozenset({"dft_mag_density"}),
        "dft_band_gap": frozenset({"dft_band_gap"}),
        "ml_bulk_modulus": frozenset({"ml_bulk_modulus"}),
        "dft_mag_density_hhi_score": frozenset({"dft_mag_density", "hhi_score"}),
        "chemical_system_energy_above_hull": frozenset(
            {"chemical_system", "energy_above_hull"}
        ),
    }

    assert MatterGenGenerator._KNOWN_CHECKPOINT_CONDITIONS == expected
    for checkpoint_name, supported in expected.items():
        runtime = MatterGenGenerator(pretrained_name=checkpoint_name, device="cpu")
        assert runtime.supported_condition_names == supported
        assert runtime.condition_contract_source == "official-checkpoint-allowlist"
        assert runtime.matcher_ltol == pytest.approx(0.02)
        assert runtime.matcher_stol == pytest.approx(0.05)
        assert runtime.matcher_angle_tol == pytest.approx(1.0)

    with pytest.raises(ValueError, match="exact condition allowlist"):
        MatterGenGenerator(
            pretrained_name="dft_band_gap",
            supported_condition_names=("dft_band_gap", "chemical_system"),
            device="cpu",
        )


def test_custom_checkpoint_conditioning_requires_an_explicit_declaration() -> None:
    request = _condition_request(dft_band_gap=1.5)

    undeclared = MatterGenGenerator(pretrained_name="local_custom", device="cpu")
    with pytest.raises(ModelExecutionError, match="explicit supported_condition_names"):
        undeclared._conditions(request)

    declared = MatterGenGenerator(
        pretrained_name="local_custom",
        supported_condition_names=("dft_band_gap",),
        device="cpu",
    )
    conditions, _warnings = declared._conditions(request)
    assert conditions == {"dft_band_gap": 1.5}


def test_numeric_condition_units_are_normalized_and_strings_fail_closed() -> None:
    runtime = MatterGenGenerator(pretrained_name="dft_band_gap", device="cpu")
    request = _condition_request(dft_band_gap=1500.0)
    request.goal.objectives[0].unit = "meV"

    conditions, _warnings = runtime._conditions(request)

    assert conditions == {"dft_band_gap": pytest.approx(1.5)}

    request.goal.objectives[0].unit = "kJ/mol"
    with pytest.raises(ModelExecutionError, match="incompatible or missing unit"):
        runtime._conditions(request)

    request.goal.objectives[0].unit = "eV"
    request.goal.objectives[0].target_value = "1.5"
    with pytest.raises(ModelExecutionError, match="requires a finite number"):
        runtime._conditions(request)


@pytest.mark.parametrize("excluded", ["He", "Ne", "Tc", "Pm", "At", "U", "Og"])
def test_chemical_system_rejects_released_model_card_exclusions(excluded: str) -> None:
    runtime = MatterGenGenerator(pretrained_name="chemical_system", device="cpu")

    with pytest.raises(ModelExecutionError, match="released model-card domain"):
        runtime._conditions(_condition_request(chemical_system=f"Li-{excluded}"))


def test_chemical_system_cannot_override_hard_goal_scope() -> None:
    runtime = MatterGenGenerator(pretrained_name="chemical_system", device="cpu")
    request = _condition_request(chemical_system="Li-Fe-O")
    request.goal.constraints = [
        SimpleNamespace(
            hard=True,
            property_name="chemical_system",
            operator="eq",
            value="Li-O",
        )
    ]

    with pytest.raises(ModelExecutionError, match="immutable hard goal"):
        runtime._conditions(request)


def test_generation_replaces_contract_violations_and_audits_applied_gamma(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RawStructure:
        def __init__(self, identity: str) -> None:
            self.identity = identity

    class FakeGenerator:
        properties_to_condition_on: dict[str, object] = {}
        diffusion_guidance_factor: float = 0.0

        def __init__(self) -> None:
            self.identities = iter(
                (
                    "source-too-large",
                    "primitive-too-large",
                    "excluded-elements",
                    "wrong-elements",
                    "wrong-space-group",
                    "valid",
                )
            )
            self.batch_sizes: list[int] = []

        def generate(
            self, *, batch_size: int, num_batches: int, output_dir: str
        ) -> list[Any]:
            assert num_batches == 1
            assert output_dir
            self.batch_sizes.append(batch_size)
            return [RawStructure(next(self.identities)) for _index in range(batch_size)]

    def fake_canonicalize(structure: RawStructure, **_kwargs: Any) -> Any:
        symbols = (
            ("Li", "U")
            if structure.identity == "excluded-elements"
            else ("Li", "N")
            if structure.identity == "wrong-elements"
            else ("Li", "O")
        )
        space_group = 1 if structure.identity == "wrong-space-group" else 225
        source_atom_count = 21 if structure.identity == "source-too-large" else 3
        primitive_atom_count = (
            21 if structure.identity == "primitive-too-large" else 3
        )
        composition = SimpleNamespace(
            elements=tuple(SimpleNamespace(symbol=symbol) for symbol in symbols),
            reduced_formula="LiN" if symbols == ("Li", "N") else "Li2O",
        )
        return SimpleNamespace(
            primitive_structure=SimpleNamespace(composition=composition),
            canonical_structure=SimpleNamespace(composition=composition),
            structure_hash=f"hash-{structure.identity}",
            identity_structure_hash=f"identity-hash-{structure.identity}",
            source_atom_count=source_atom_count,
            primitive_atom_count=primitive_atom_count,
            conventional_atom_count=primitive_atom_count,
            space_group_symbol="Fm-3m" if space_group == 225 else "P1",
            space_group_number=space_group,
        )

    def fake_group(structures: tuple[Any, ...], **_kwargs: Any) -> Any:
        return SimpleNamespace(
            groups=tuple(
                SimpleNamespace(representative_index=index)
                for index, _structure in enumerate(structures)
            )
        )

    monkeypatch.setattr(
        "discovery_os.sidecars.generators.pymatgen_to_cif",
        lambda structure, *, max_bytes: f"data_{structure.identity}\n",
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.canonicalize_crystal_structure",
        fake_canonicalize,
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.group_crystal_structures",
        fake_group,
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators._seed_mattergen", lambda _seed: None
    )

    model = FakeGenerator()
    runtime = MatterGenGenerator(
        pretrained_name="local_joint_checkpoint",
        supported_condition_names=("chemical_system", "space_group"),
        deduplication_max_generation_rounds=6,
        device="cpu",
    )
    runtime._model = model
    runtime._resolved_device = "cpu"
    controls = GenerationControls(
        alpha=0.5,
        temperature=1.4,
        mutation_strength=0.7,
        diversity_strength=0.8,
        decision_reason="contract test",
    )
    request = _condition_request(chemical_system="Li-O", space_group=225)
    request.run_config = SimpleNamespace(
        candidate_count=1,
        generation_controls=controls,
        effective_generator_seed=17,
    )

    batch = runtime.generate(request)

    assert model.batch_sizes == [1, 1, 1, 1, 1, 1]
    assert model.properties_to_condition_on == {
        "chemical_system": "Li-O",
        "space_group": 225,
    }
    assert model.diffusion_guidance_factor == 2.0
    assert len(batch.candidates) == 1
    attributes = batch.candidates[0].attributes
    assert attributes["composition_key"] == "Li2O"
    assert attributes["requested_generation_controls"] == controls.model_dump(
        mode="json"
    )
    assert attributes["applied_generation_controls"]["diffusion_guidance_factor"] == 2.0
    assert attributes["ignored_generation_controls"] == [
        "temperature",
        "mutation_strength",
        "diversity_strength",
    ]
    assert attributes["generation_funnel"] | {} == {
        "requested_samples": 1,
        "raw_model_structures": 6,
        "parsed_structures": 6,
        "exact_file_unique": 6,
        "crystallographically_unique": 1,
        "geometry_valid": 1,
        "raw_geometry_valid": 6,
        "requested_unique_candidates": 1,
        "parse_rejected": 0,
        "geometry_rejected": 0,
        "canonicalization_rejected": 0,
        "applicability_rejected": 3,
        "condition_rejected": 2,
        "source_atom_count_rejected": 1,
        "primitive_atom_count_rejected": 1,
        "model_card_element_rejected": 1,
        "chemical_system_rejected": 1,
        "space_group_rejected": 1,
        "cross_call_duplicate_rejected": 0,
        "cross_call_ambiguous_comparisons": 0,
        "duplicates_removed": 0,
        "generation_rounds": 6,
    }


def test_search_session_replaces_cross_call_duplicate_before_expert_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RawStructure:
        def __init__(self, identity: str) -> None:
            self.identity = identity

    class FakeGenerator:
        properties_to_condition_on: dict[str, object] = {}
        diffusion_guidance_factor = 0.0

        def __init__(self) -> None:
            self.identities = iter(("first", "first", "replacement"))
            self.batch_sizes: list[int] = []

        def generate(
            self, *, batch_size: int, num_batches: int, output_dir: str
        ) -> list[RawStructure]:
            assert num_batches == 1
            assert output_dir
            self.batch_sizes.append(batch_size)
            return [RawStructure(next(self.identities)) for _ in range(batch_size)]

    def canonicalize(structure: RawStructure, **_kwargs: Any) -> Any:
        composition = SimpleNamespace(
            elements=(
                SimpleNamespace(symbol="Li"),
                SimpleNamespace(symbol="O"),
            ),
            reduced_formula="Li2O",
        )
        return SimpleNamespace(
            identity=structure.identity,
            primitive_structure=SimpleNamespace(composition=composition),
            canonical_structure=SimpleNamespace(composition=composition),
            structure_hash=f"prototype-{structure.identity}",
            identity_structure_hash=f"identity-{structure.identity}",
            source_atom_count=3,
            primitive_atom_count=3,
            conventional_atom_count=3,
            space_group_symbol="P1",
            space_group_number=1,
        )

    def grouping(structures: tuple[Any, ...], **_kwargs: Any) -> Any:
        return SimpleNamespace(
            groups=tuple(
                SimpleNamespace(representative_index=index)
                for index, _structure in enumerate(structures)
            )
        )

    def relation(first: Any, second: Any, **_kwargs: Any) -> Any:
        duplicate = first.identity == second.identity
        return SimpleNamespace(
            hard_deduplication_allowed=duplicate,
            relation=SimpleNamespace(
                value="strict_material_duplicate" if duplicate else "distinct"
            ),
            reason=None,
        )

    monkeypatch.setattr(
        "discovery_os.sidecars.generators.pymatgen_to_cif",
        lambda structure, *, max_bytes: f"data_{structure.identity}\n",
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.canonicalize_crystal_structure",
        canonicalize,
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.group_crystal_structures",
        grouping,
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators.classify_crystal_structure_relation",
        relation,
    )
    monkeypatch.setattr(
        "discovery_os.sidecars.generators._seed_mattergen",
        lambda _seed: None,
    )

    model = FakeGenerator()
    runtime = MatterGenGenerator(
        pretrained_name="mattergen_base",
        deduplication_max_generation_rounds=2,
        device="cpu",
    )
    runtime._model = model
    runtime._resolved_device = "cpu"
    request = _condition_request()
    request.run_config = SimpleNamespace(
        candidate_count=1,
        generation_controls=GenerationControls(),
        effective_generator_seed=31,
        search_session_id="search-session-a",
    )

    first = runtime.generate(request)
    second = runtime.generate(request)

    assert model.batch_sizes == [1, 1, 1]
    assert first.candidates[0].representations[0].value == "data_first"
    assert second.candidates[0].representations[0].value == "data_replacement"
    funnel = second.candidates[0].attributes["generation_funnel"]
    assert funnel["cross_call_duplicate_rejected"] == 1
    assert funnel["raw_model_structures"] == 2
    assert funnel["crystallographically_unique"] == 1
