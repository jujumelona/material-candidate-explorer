from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from discovery_os.fusion_schemas import ExpertFeatureRequest, ScientificModality
from discovery_os.hashing import candidate_content_hash
from discovery_os.schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    DiscoveryGoal,
    ObjectiveDirection,
    PropertyObjective,
    RepresentationKind,
)
from discovery_os.sidecars import BoltzExpert, CandidateConversionError, ModelExecutionError
from discovery_os.sidecars.base import numeric_tensor_data
from discovery_os.sidecars import cli as sidecar_cli
from discovery_os.sidecars.weight_binding import directory_inventory_sha256


def _goal(candidate_type: CandidateType) -> DiscoveryGoal:
    return DiscoveryGoal(
        goal_id="boltz-goal",
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        title="Boltz feature fixture",
        scientific_question="What documented structure-confidence observables are predicted?",
        objectives=[
            PropertyObjective(
                property_name="confidence_score",
                direction=ObjectiveDirection.MAXIMIZE,
            )
        ],
        validation_profile_id="boltz-fixture-v1",
        candidate_types=[candidate_type],
    )


def _candidate(
    *,
    candidate_type: CandidateType,
    kind: RepresentationKind,
    value: str,
) -> Candidate:
    bare = Candidate(
        candidate_id="boltz-candidate",
        candidate_type=candidate_type,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[CandidateRepresentation(kind=kind, value=value, canonical=True)],
    )
    return bare.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=bare.candidate_id,
                version=1,
                content_hash=candidate_content_hash(bare),
            )
        }
    )


def _request(candidate: Candidate, modality: ScientificModality) -> ExpertFeatureRequest:
    return ExpertFeatureRequest(
        workspace_entity_id="primary",
        candidate=candidate,
        goal=_goal(candidate.candidate_type),
        modality=modality,
        feature_space="boltz-structure-v1",
        cycle=0,
        seed=19,
    )


def _fake_boltz_cli(path: Path) -> None:
    path.write_text(
        '''
import json
from pathlib import Path
import sys

args = sys.argv[1:]
assert args[0] == "predict"
input_path = Path(args[1])
out_dir = Path(args[args.index("--out_dir") + 1])
cache = Path(args[args.index("--cache") + 1])
document = json.loads(input_path.read_text(encoding="utf-8"))
cache.mkdir(parents=True, exist_ok=True)
(cache / "captured.json").write_text(
    json.dumps({"argv": args, "document": document}, sort_keys=True),
    encoding="utf-8",
)
prediction = out_dir / "boltz_results_request" / "predictions" / "request"
prediction.mkdir(parents=True, exist_ok=True)
(prediction / "confidence_request_model_0.json").write_text(
    json.dumps({
        "confidence_score": 0.8,
        "ptm": 0.7,
        "iptm": 0.6,
        "ligand_iptm": 0.5,
        "protein_iptm": 0.4,
        "complex_plddt": 0.9,
        "complex_iplddt": 0.85,
        "complex_pde": 1.1,
        "complex_ipde": 1.2,
        "chains_ptm": {"0": 0.7},
    }),
    encoding="utf-8",
)
(prediction / "affinity_request.json").write_text(
    json.dumps({
        "affinity_pred_value": -2.0,
        "affinity_probability_binary": 0.91,
        "affinity_pred_value1": -1.9,
        "affinity_probability_binary1": 0.90,
        "affinity_pred_value2": -2.1,
        "affinity_probability_binary2": 0.92,
    }),
    encoding="utf-8",
)
(prediction / "request_model_0.cif").write_text(
    """data_request
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
ATOM 1 C CA ALA A 1
ATOM 2 N N ALA A 1
ATOM 3 C CA GLY A 2
#
""",
    encoding="utf-8",
)
'''.strip()
        + "\n",
        encoding="utf-8",
    )


def test_boltz_221_fake_cli_uses_owned_yaml_and_parses_documented_outputs(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_boltz.py"
    cache = tmp_path / "cache"
    _fake_boltz_cli(script)
    runtime = BoltzExpert(
        executable=sys.executable,
        executable_arguments=(str(script),),
        cache_path=str(cache),
        process_timeout_seconds=10,
        device="cpu",
    )
    candidate = _candidate(
        candidate_type=CandidateType.PROTEIN,
        kind=RepresentationKind.PROTEIN_SEQUENCE,
        value="ACDE",
    )

    result = runtime.encode(_request(candidate, ScientificModality.PROTEIN_STRUCTURE))

    shape, values = numeric_tensor_data(result.values)
    assert shape == [1, 12]
    assert values == [0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.85, 1.1, 1.2, 3.0, 1.0, 2.0]
    assert result.projection_id == "boltz-2.2.1-confidence-structure-v1"
    properties = {item.property_name: item for item in result.properties}
    assert properties["confidence_score"].value == 0.8
    assert properties["predicted_atom_count"].value == 3.0
    assert properties["affinity_pred_value"].unit == "log10(micromolar_IC50)"
    assert properties["affinity_probability_binary"].value == 0.91
    assert len(result.quality_flags[0]) == len("predicted_mmcif_sha256:") + 64
    assert any("not hidden embeddings" in warning for warning in result.warnings)
    assert any("msa: empty" in warning for warning in result.warnings)

    captured = json.loads((cache / "captured.json").read_text(encoding="utf-8"))
    assert captured["document"] == {
        "version": 1,
        "sequences": [{"protein": {"id": "A", "sequence": "ACDE", "msa": "empty"}}],
    }
    argv = captured["argv"]
    assert argv[:2] == ["predict", argv[1]]
    assert argv[1].endswith("request.yaml")
    assert argv[argv.index("--model") + 1] == "boltz2"
    assert argv[argv.index("--accelerator") + 1] == "cpu"
    assert argv[argv.index("--diffusion_samples") + 1] == "1"
    assert argv[argv.index("--output_format") + 1] == "mmcif"
    assert argv[argv.index("--seed") + 1] == "19"
    assert "--use_msa_server" not in argv
    assert "--write_embeddings" not in argv


def test_boltz_rejects_ambiguous_multirecord_fasta_before_cli(tmp_path: Path) -> None:
    runtime = BoltzExpert(executable=sys.executable, cache_path=str(tmp_path / "cache"), device="cpu")
    candidate = _candidate(
        candidate_type=CandidateType.PROTEIN,
        kind=RepresentationKind.FASTA,
        value=">A\nAC\n>B\nDE\n",
    )
    with pytest.raises(CandidateConversionError, match="exactly one sequence record"):
        runtime.encode(_request(candidate, ScientificModality.PROTEIN_STRUCTURE))


def test_boltz_rejects_ligand_affinity_claim_without_complex_contract(tmp_path: Path) -> None:
    script = tmp_path / "fake_boltz.py"
    cache = tmp_path / "cache"
    _fake_boltz_cli(script)
    runtime = BoltzExpert(
        executable=sys.executable,
        executable_arguments=(str(script),),
        cache_path=str(cache),
        process_timeout_seconds=10,
        device="cpu",
    )
    candidate = _candidate(
        candidate_type=CandidateType.SMALL_MOLECULE,
        kind=RepresentationKind.SMILES,
        value="CCO",
    )
    result = runtime.encode(_request(candidate, ScientificModality.MOLECULE_3D))
    captured = json.loads((cache / "captured.json").read_text(encoding="utf-8"))
    assert captured["document"] == {
        "version": 1,
        "sequences": [{"ligand": {"id": "A", "smiles": "CCO"}}],
    }
    assert any("ligand-only" in warning for warning in result.warnings)


def test_boltz_preflight_binds_the_pinned_snapshot_without_loading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    for name in ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "mols.tar"):
        (snapshot / name).write_bytes((name + "\n").encode("ascii"))
    (snapshot / ".snapshot.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "repository": "boltz-community/boltz-2",
                "revision": "6fdef46d763fee7fbb83ca5501ccceff43b85607",
                "inventory_sha256": directory_inventory_sha256(snapshot),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(sidecar_cli, "_required_boltz_executable", lambda: sys.executable)
    monkeypatch.setattr(sidecar_cli, "_module_available", lambda _name: True)

    report = sidecar_cli.preflight_configuration(
        "boltz",
        {
            "SIDECAR_WEIGHT_SNAPSHOT_PATH": str(snapshot),
            "SIDECAR_WEIGHT_REVISION": "6fdef46d763fee7fbb83ca5501ccceff43b85607",
            "SIDECAR_DEVICE": "cpu",
        },
        host="127.0.0.1",
        port=8103,
    )

    assert report["supported"] is True
    assert report["configuration_only"] is True
    assert report["checkpoint_loaded"] is False
    assert report["model_id"] == "boltz"
    assert report["weight_revision"] == "6fdef46d763fee7fbb83ca5501ccceff43b85607"
    assert len(report["runtime_parameters_hash"]) == 64


@pytest.mark.parametrize(
    ("changed_artifact", "message"),
    (
        ("checkpoint", "confidence checkpoint bytes changed"),
        ("argument", "executable argument bytes changed"),
    ),
)
def test_boltz_rechecks_external_artifacts_before_every_invocation(
    tmp_path: Path,
    changed_artifact: str,
    message: str,
) -> None:
    script = tmp_path / "fake_boltz.py"
    _fake_boltz_cli(script)
    checkpoint = tmp_path / "boltz2_conf.ckpt"
    affinity = tmp_path / "boltz2_aff.ckpt"
    mols = tmp_path / "mols.tar"
    for selected in (checkpoint, affinity, mols):
        selected.write_bytes(b"attested")
    runtime = BoltzExpert(
        executable=sys.executable,
        executable_arguments=(str(script),),
        cache_path=str(tmp_path / "cache"),
        checkpoint_path=str(checkpoint),
        affinity_checkpoint_path=str(affinity),
        mols_tar_path=str(mols),
        process_timeout_seconds=10,
        device="cpu",
    )
    runtime._ensure_loaded()

    selected = checkpoint if changed_artifact == "checkpoint" else script
    selected.write_bytes(b"changed-after-first-load")

    # Verification happens before route parsing or subprocess creation.
    with pytest.raises(ModelExecutionError, match=message):
        runtime.encode(object())  # type: ignore[arg-type]
