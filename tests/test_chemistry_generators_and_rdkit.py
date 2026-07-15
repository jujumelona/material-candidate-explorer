from __future__ import annotations

import pytest

from discovery_os.artifacts import ArtifactStore
from discovery_os.chemistry import FormulaError, parse_formula
from discovery_os.generators import (
    DummyGenerator,
    GeneratorExecutionError,
    GeneratorRuntime,
    build_default_generator_registry,
)
from discovery_os.hashing import candidate_content_hash
from discovery_os.runtime import ToolRuntime
from discovery_os.schemas import (
    CandidatePlan,
    CandidateType,
    DiscoveryDomain,
    EvidenceStatus,
    GenerationTask,
    ValidationPlan,
)
from discovery_os.tool_adapters import build_default_tool_registry


@pytest.mark.parametrize(
    "formula, expected",
    [
        ("H2O", {"H": 2.0, "O": 1.0}),
        ("Ca(OH)2", {"H": 2.0, "O": 2.0, "Ca": 1.0}),
        ("Al2(SO4)3", {"O": 12.0, "Al": 2.0, "S": 3.0}),
        (
            "K4(ON(SO3)2)2",
            {"N": 2.0, "O": 14.0, "S": 4.0, "K": 4.0},
        ),
        (
            "La1.85Sr0.15CuO4",
            {"O": 4.0, "Cu": 1.0, "Sr": 0.15, "La": 1.85},
        ),
    ],
)
def test_formula_parser_accepts_supported_formulas(
    formula: str, expected: dict[str, float]
) -> None:
    assert parse_formula(formula) == pytest.approx(expected)


@pytest.mark.parametrize(
    "formula",
    ["", "2H", "H0", "Xx2", "H2O$", "Ca(OH2", "Ca)OH(", "Na..2Cl"],
)
def test_formula_parser_rejects_invalid_formulas(formula: str) -> None:
    with pytest.raises(FormulaError):
        parse_formula(formula)


def test_dummy_generator_emits_integrity_checked_candidate_refs() -> None:
    task = GenerationTask(
        task_id="GEN-INTEGRITY",
        generator_name="dummy_generator",
        candidate_type=CandidateType.SMALL_MOLECULE,
        requested_count=2,
        conditions={"domain": DiscoveryDomain.MEDICINAL_CHEMISTRY},
        reason="Generate known test fixtures.",
    )
    generator = DummyGenerator()

    first = generator.generate(task, [])
    second = generator.generate(task, [])

    assert first == second
    assert len(first.candidates) == 2
    for candidate in first.candidates:
        assert candidate.candidate_ref is not None
        assert candidate.candidate_ref.candidate_id == candidate.candidate_id
        assert candidate.candidate_ref.version == 1
        assert candidate.candidate_ref.content_hash == candidate_content_hash(candidate)


def test_generator_runtime_rechecks_candidate_ref_content_hash(monkeypatch) -> None:
    registry = build_default_generator_registry(include_placeholders=False)
    adapter = registry.get("dummy_generator")
    original_generate = adapter.generate

    def generate_with_bad_hash(task, parents):
        batch = original_generate(task, parents)
        candidate = batch.candidates[0]
        bad_ref = candidate.candidate_ref.model_copy(update={"content_hash": "0" * 64})
        return batch.model_copy(
            update={
                "candidates": [candidate.model_copy(update={"candidate_ref": bad_ref})]
            }
        )

    monkeypatch.setattr(adapter, "generate", generate_with_bad_hash)
    plan = CandidatePlan(
        tasks=[
            GenerationTask(
                task_id="GEN-TAMPER",
                generator_name="dummy_generator",
                candidate_type=CandidateType.SMALL_MOLECULE,
                requested_count=1,
                reason="Tampering test fixture.",
            )
        ],
        plan_reason="Verify runtime integrity enforcement.",
    )

    with pytest.raises(GeneratorExecutionError, match="content hash is invalid"):
        GeneratorRuntime(registry).execute(plan)


def test_rdkit_invalid_smiles_is_a_successful_execution_with_failed_validity(
    tmp_path, candidate_factory, tool_call_factory
) -> None:
    pytest.importorskip("rdkit")
    valid = candidate_factory(candidate_id="MOL-VALID", value="CCO")
    invalid = candidate_factory(candidate_id="MOL-INVALID", value="C1(CC")
    registry = build_default_tool_registry(include_placeholders=False)
    assert registry.get("rdkit").descriptor.available
    call = tool_call_factory(
        call_id="CALL-RDKIT",
        tool_name="rdkit",
        operation="validate_molecule",
        candidate_ids=[valid.candidate_id, invalid.candidate_id],
        requested_properties=["validity", "canonical_smiles", "molecular_weight"],
        max_runtime_seconds=30,
    )
    runtime = ToolRuntime(registry, ArtifactStore(tmp_path / "artifacts"))

    batch = runtime.execute_plan(
        ValidationPlan(
            calls=[call],
            expected_information_gain={},
            plan_reason="Exercise RDKit normalization.",
        ),
        candidates=[valid, invalid],
    )

    records = {record.candidate_id: record for record in batch.records}
    valid_properties = {
        item.property_name: item for item in records[valid.candidate_id].properties
    }
    invalid_properties = {
        item.property_name: item for item in records[invalid.candidate_id].properties
    }
    assert records[valid.candidate_id].status == EvidenceStatus.SUCCESS
    assert valid_properties["validity"].value is True
    assert valid_properties["validity"].meets_criterion is True
    assert "canonical_smiles" in valid_properties
    assert "molecular_weight" in valid_properties

    # Parsing completed normally: scientific invalidity is a property result,
    # not an adapter crash, timeout, or execution failure.
    assert records[invalid.candidate_id].status == EvidenceStatus.SUCCESS
    assert records[invalid.candidate_id].failure_modes == []
    assert invalid_properties["validity"].value is False
    assert invalid_properties["validity"].meets_criterion is False
    assert records[invalid.candidate_id].warnings
