from discovery_os.stage_rag import (
    RagStage,
    StageAwareMultiRag,
    StageAwareRagRouter,
    StageEvidenceStatus,
    StageRagQuery,
)


def test_router_selects_stage_specific_routes():
    routes = StageAwareRagRouter().route(
        StageRagQuery("KRAS G12D variant assay pocket and failed trial evidence")
    )
    stages = {route.stage for route in routes}
    assert RagStage.SCIENTIFIC_EVIDENCE in stages
    assert RagStage.GENOMICS_BIOMARKER in stages
    assert RagStage.MOLECULE_ASSAY in stages
    assert RagStage.PROTEIN_POCKET_3D in stages
    assert RagStage.CLINICAL_TRIAL in stages
    assert RagStage.FAILURE_MEMORY in stages


def test_unconfigured_stage_fails_closed_and_adapter_payload_is_preserved():
    query = StageRagQuery("KRAS G12D molecule assay")
    pipeline = StageAwareMultiRag({
        RagStage.MOLECULE_ASSAY: lambda query, route: {
            "source_ids": ["CHEMBL1"], "measurement": "IC50", "unit": "nM"
        }
    })
    rows = pipeline.run(query)
    by_stage = {row.stage: row for row in rows}
    assert by_stage[RagStage.MOLECULE_ASSAY].status == StageEvidenceStatus.SUCCESS
    assert by_stage[RagStage.MOLECULE_ASSAY].source_ids == ("CHEMBL1",)
    assert by_stage[RagStage.SCIENTIFIC_EVIDENCE].status == StageEvidenceStatus.NOT_CONFIGURED

