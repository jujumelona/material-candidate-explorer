from types import SimpleNamespace

from discovery_os.alphagenome import (
    AlphaGenomeClient,
    GenomicMode,
    VariantCandidate,
)


def candidate(**overrides):
    values = dict(
        candidate_id="var-a",
        chromosome="chr22",
        position=36201698,
        reference_bases="A",
        alternate_bases="C",
        interval_start=35677410,
        interval_end=36725986,
        literature_support=0.8,
    )
    values.update(overrides)
    return VariantCandidate(**values)


def test_genomic_modes_and_candidate_validation():
    assert GenomicMode.DISEASE_VARIANT_PRIORITIZATION.value == "DISEASE_VARIANT_PRIORITIZATION"
    try:
        candidate(literature_support=2.0)
    except ValueError as exc:
        assert "literature_support" in str(exc)
    else:
        raise AssertionError("invalid literature support was accepted")


def test_fake_model_variant_evaluation_is_rankable():
    def track(values):
        return SimpleNamespace(reference=SimpleNamespace(values=values), alternate=SimpleNamespace(values=[x + 1 for x in values]))

    output = SimpleNamespace(rna_seq=track([0.0, 1.0]), splice=track([0.0, 2.0]), atac=track([1.0, 3.0]))
    model = SimpleNamespace(predict_variant=lambda **kwargs: output)
    result = AlphaGenomeClient(model=model).evaluate(candidate())
    assert result.status == "success"
    assert result.expression_change == 1.0
    assert result.splicing_change == 1.0
    assert result.chromatin_change == 1.0
    assert result.provenance["api_key"] == "runtime_only"

