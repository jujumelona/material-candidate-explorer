from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from discovery_os.fusion_schemas import ExpertFeatureRequest, ScientificModality, TensorRole
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
from discovery_os.sidecars.cli import preflight_configuration
from discovery_os.sidecars.conversions import cell_expression
from discovery_os.sidecars.errors import CandidateConversionError, ModelExecutionError
from discovery_os.sidecars.experts import ScGPTExpert
from discovery_os.sidecars.app import create_sidecar_app
from discovery_os.sidecars.types import ModelIdentity


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path / "scgpt-bundle"
    root.mkdir()
    (root / "args.json").write_text(
        json.dumps(
            {
                "embsize": 2,
                "nheads": 1,
                "d_hid": 2,
                "nlayers": 1,
                "n_layers_cls": 1,
                "dropout": 0.0,
                "pad_token": "<pad>",
                "pad_value": -2,
                "n_bins": 51,
                "input_emb_style": "continuous",
                "cell_emb_style": "cls",
                "pre_norm": False,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (root / "vocab.json").write_text(
        json.dumps({"<pad>": 0, "<cls>": 1, "<eoc>": 2, "G1": 3, "G2": 4}),
        encoding="utf-8",
    )
    (root / "best_model.pt").write_bytes(b"pinned-scgpt-state-fixture")
    return root


def _candidate(value: str, *, kind: RepresentationKind = RepresentationKind.CELL_EXPRESSION) -> Candidate:
    candidate = Candidate(
        candidate_id="cell-1",
        candidate_type=CandidateType.CELL_STATE,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[CandidateRepresentation(kind=kind, value=value, canonical=True)],
    )
    return candidate.model_copy(
        update={
            "candidate_ref": CandidateRef(
                candidate_id=candidate.candidate_id,
                version=1,
                content_hash=candidate_content_hash(candidate),
            )
        }
    )


def _request(value: str) -> ExpertFeatureRequest:
    return ExpertFeatureRequest(
        workspace_entity_id="cell",
        candidate=_candidate(value),
        goal=DiscoveryGoal(
            goal_id="cell-goal",
            domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
            title="Cell embedding",
            scientific_question="What is the candidate cell state?",
            objectives=[
                PropertyObjective(
                    property_name="cell_state_similarity",
                    direction=ObjectiveDirection.MAXIMIZE,
                )
            ],
            validation_profile_id="cell-v1",
            candidate_types=[CandidateType.CELL_STATE],
        ),
        modality=ScientificModality.CELL_STATE,
        feature_space="scgpt-cell-v1",
        cycle=0,
        seed=11,
    )


def test_scgpt_rechecks_bundle_immediately_before_lazy_load(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    runtime = ScGPTExpert(checkpoint_dir=str(bundle), device="cpu")
    (bundle / "best_model.pt").write_bytes(b"changed-after-attestation")

    with pytest.raises(ModelExecutionError, match="changed after runtime attestation"):
        runtime._load_model("cpu")


class _FakeTensor:
    def __init__(self, value: Any) -> None:
        self.value = value

    @property
    def shape(self) -> tuple[int, ...]:
        value = self.value
        shape: list[int] = []
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else None
        return tuple(shape)

    def to(self, _device: str) -> _FakeTensor:
        return self

    def eq(self, scalar: Any) -> _FakeTensor:
        def apply(value: Any) -> Any:
            return [apply(item) for item in value] if isinstance(value, list) else value == scalar

        return _FakeTensor(apply(self.value))

    def __getitem__(self, key: Any) -> _FakeTensor:
        value = self.value
        keys = key if isinstance(key, tuple) else (key,)
        for item in keys:
            value = value[item]
        return _FakeTensor(value)

    def detach(self) -> _FakeTensor:
        return self

    def cpu(self) -> _FakeTensor:
        return self

    def tolist(self) -> Any:
        return self.value


class _FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: Any) -> None:
        return None


class _FakeTorch:
    long = "long"
    float32 = "float32"

    def __init__(self, *, omit_encoder_weight: bool = False) -> None:
        self.omit_encoder_weight = omit_encoder_weight
        self.load_calls: list[dict[str, Any]] = []

    def tensor(self, value: Any, *, dtype: Any) -> _FakeTensor:
        assert dtype in {self.long, self.float32}
        return _FakeTensor(value)

    def no_grad(self) -> _FakeNoGrad:
        return _FakeNoGrad()

    def load(
        self,
        path: str,
        *,
        map_location: str | None = None,
        weights_only: bool = False,
    ) -> dict[str, _FakeTensor]:
        self.load_calls.append(
            {"path": path, "map_location": map_location, "weights_only": weights_only}
        )
        state = {
            "encoder.embedding.weight": _FakeTensor([[0.0, 0.0]] * 5),
            "value_encoder.linear.weight": _FakeTensor([[0.0, 0.0], [0.0, 0.0]]),
            "transformer_encoder.layer.weight": _FakeTensor([[0.0, 0.0], [0.0, 0.0]]),
        }
        if self.omit_encoder_weight:
            state.pop("encoder.embedding.weight")
        return state


class _FakeLegacyTorch(_FakeTorch):
    def load(
        self,
        path: str,
        *,
        map_location: str | None = None,
    ) -> dict[str, _FakeTensor]:
        raise AssertionError("unsafe legacy torch.load must not be called")


class _FakeVocab:
    def __init__(self, mapping: dict[str, int]) -> None:
        self.mapping = mapping
        self.default_index: int | None = None

    @classmethod
    def from_file(cls, path: Path) -> _FakeVocab:
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def set_default_index(self, value: int) -> None:
        self.default_index = value

    def __getitem__(self, key: str) -> int:
        return self.mapping[key]

    def __len__(self) -> int:
        return len(self.mapping)


class _FakeModel:
    last: _FakeModel | None = None

    def __init__(self, **kwargs: Any) -> None:
        _FakeModel.last = self
        self.kwargs = kwargs
        self.eval_called = False
        self.loaded = False
        self.encoded_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def state_dict(self) -> dict[str, _FakeTensor]:
        return {
            "encoder.embedding.weight": _FakeTensor([[0.0, 0.0]] * 5),
            "value_encoder.linear.weight": _FakeTensor([[0.0, 0.0], [0.0, 0.0]]),
            "transformer_encoder.layer.weight": _FakeTensor([[0.0, 0.0], [0.0, 0.0]]),
        }

    def to(self, _device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        self.eval_called = True
        return self

    def _encode(
        self,
        genes: _FakeTensor,
        values: _FakeTensor,
        *,
        src_key_padding_mask: _FakeTensor,
        batch_labels: None,
    ) -> _FakeTensor:
        assert src_key_padding_mask.shape == genes.shape
        assert batch_labels is None
        self.encoded_shapes.append((genes.shape, values.shape))
        return _FakeTensor([[[3.0, 4.0] for _ in range(genes.shape[1])]])


class _FakeCollator:
    last_kwargs: dict[str, Any] | None = None

    def __init__(self, **kwargs: Any) -> None:
        _FakeCollator.last_kwargs = kwargs

    def __call__(self, examples: list[dict[str, _FakeTensor]]) -> dict[str, _FakeTensor]:
        assert len(examples) == 1
        return {
            "gene": _FakeTensor([examples[0]["genes"].value]),
            "expr": _FakeTensor([examples[0]["expressions"].value]),
        }


def _install_fake_modules(
    monkeypatch: pytest.MonkeyPatch,
    torch: _FakeTorch,
) -> list[list[float]]:
    binning_calls: list[int] = []
    binning_inputs: list[list[float]] = []

    def binning(row: _FakeTensor, *, n_bins: int) -> _FakeTensor:
        assert n_bins == 51
        binning_calls.append(n_bins)
        binning_inputs.append(list(row.value))
        return row

    modules = {
        "torch": torch,
        "scgpt.model": SimpleNamespace(TransformerModel=_FakeModel),
        "scgpt.tokenizer": SimpleNamespace(GeneVocab=_FakeVocab),
        "scgpt.data_collator": SimpleNamespace(DataCollator=_FakeCollator),
        "scgpt.preprocess": SimpleNamespace(binning=binning, calls=binning_calls),
        "scgpt.utils": SimpleNamespace(
            load_pretrained=lambda model, _state, strict, verbose: (
                setattr(model, "loaded", True),
                strict is False,
                verbose is False,
            )
        ),
    }
    monkeypatch.setattr(
        "discovery_os.sidecars.experts.require_module",
        lambda name, **_kwargs: modules[name],
    )
    return binning_inputs


def test_scgpt_real_api_path_returns_only_actual_normalized_cell_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _FakeTorch()
    binning_inputs = _install_fake_modules(monkeypatch, torch)
    adapter = ScGPTExpert(checkpoint_dir=str(_bundle(tmp_path)), max_length=3, device="cpu")

    result = adapter.encode(
        _request(
            json.dumps(
                {
                    "genes": ["G1", "ZERO", "UNKNOWN", "G2"],
                    "values": [1.0, 0.0, 5.0, 3.0],
                    "value_semantics": "raw_counts",
                }
            )
        )
    )

    assert result.tensor_role == TensorRole.CELL_EMBEDDING
    assert result.values[0] == pytest.approx([0.6, 0.8])
    assert result.properties == ()
    assert result.entity_ids == ("cell:0",)
    assert "encoded_gene_count:2" in result.quality_flags
    assert torch.load_calls[0]["weights_only"] is True
    assert _FakeModel.last is not None and _FakeModel.last.eval_called is True
    assert _FakeModel.last.loaded is True
    assert _FakeModel.last.encoded_shapes == [((1, 3), (1, 3))]
    assert _FakeCollator.last_kwargs is not None
    assert _FakeCollator.last_kwargs["sampling"] is False
    assert _FakeCollator.last_kwargs["do_binning"] is False
    # UNKNOWN is out of vocabulary but still contributes to the raw cell's
    # library-size normalization denominator: total = 1 + 0 + 5 + 3 = 9.
    assert binning_inputs[0] == pytest.approx(
        [math.log1p(10_000.0 / 9.0), math.log1p(30_000.0 / 9.0)]
    )
    assert "value_semantics:raw_counts" in result.quality_flags
    assert (
        "input_transform:normalize_total_10000_then_log1p_then_vocab_filter"
        in result.quality_flags
    )
    provenance = adapter.provenance_parameters()
    assert provenance["binning"] == {"enabled": True, "n_bins": 51, "explicit": True}
    assert provenance["checkpoint_sha256"]
    assert provenance["length_policy"] == "highest_expression_then_gene_name"


def test_scgpt_fails_closed_when_checkpoint_does_not_cover_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_modules(monkeypatch, _FakeTorch(omit_encoder_weight=True))
    adapter = ScGPTExpert(checkpoint_dir=str(_bundle(tmp_path)), device="cpu")
    with pytest.raises(ModelExecutionError, match="fully cover"):
        adapter.encode(
            _request(
                '{"genes":["G1"],"values":[1.0],"value_semantics":"raw_counts"}'
            )
        )


def test_scgpt_rejects_torch_without_weights_only_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_modules(monkeypatch, _FakeLegacyTorch())
    adapter = ScGPTExpert(checkpoint_dir=str(_bundle(tmp_path)), device="cpu")

    with pytest.raises(ModelExecutionError, match="weights_only"):
        adapter.encode(
            _request(
                '{"genes":["G1"],"values":[1.0],"value_semantics":"raw_counts"}'
            )
        )


@pytest.mark.parametrize(
    "payload, message",
    [
        (
            '{"genes":["G1"],"genes":["G2"],"values":[1],"value_semantics":"raw_counts"}',
            "duplicate-free",
        ),
        (
            '{"genes":["G1","G1"],"values":[1,2],"value_semantics":"raw_counts"}',
            "duplicate genes",
        ),
        (
            '{"genes":["G1"],"values":[],"value_semantics":"raw_counts"}',
            "lengths must match",
        ),
        ('{"genes":["G1"],"values":[1]}', "value_semantics"),
        (
            '{"genes":["G1"],"values":["1"],"value_semantics":"raw_counts"}',
            "non-numeric",
        ),
        ('{"G1":1}', "exactly genes, values, and value_semantics"),
    ],
)
def test_cell_expression_schema_rejects_ambiguous_or_duplicate_input(
    payload: str,
    message: str,
) -> None:
    with pytest.raises(CandidateConversionError, match=message):
        cell_expression(_candidate(payload))


def test_scgpt_preflight_binds_manual_bundle_inventory_without_loading_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr(
        "discovery_os.sidecars.cli._module_available",
        lambda _module_name: True,
    )
    report = preflight_configuration(
        "scgpt",
        {"SCGPT_CHECKPOINT_DIR": str(bundle), "SIDECAR_DEVICE": "cpu"},
        host="127.0.0.1",
        port=8106,
    )
    assert report["supported"] is True
    assert report["checkpoint_loaded"] is False
    assert report["weight_revision"].startswith("sha256:")
    assert len(report["weight_revision"]) == len("sha256:") + 64


def test_scgpt_rejects_negative_expression_before_model_load(tmp_path: Path) -> None:
    adapter = ScGPTExpert(checkpoint_dir=str(_bundle(tmp_path)), device="cpu")
    with pytest.raises(CandidateConversionError, match="non-negative"):
        adapter.encode(
            _request(
                '{"genes":["G1"],"values":[-1],"value_semantics":"raw_counts"}'
            )
        )
    assert adapter.loaded is False


def test_scgpt_normalized_fixture_is_unit_length(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    binning_inputs = _install_fake_modules(monkeypatch, _FakeTorch())
    result = ScGPTExpert(checkpoint_dir=str(_bundle(tmp_path)), device="cpu").encode(
        _request(
            '{"genes":["G1"],"values":[2.25],"value_semantics":"normalized_log1p"}'
        )
    )
    assert binning_inputs == [[2.25]]
    assert "value_semantics:normalized_log1p" in result.quality_flags
    assert "input_transform:caller_supplied_normalized_log1p" in result.quality_flags
    assert math.sqrt(sum(value * value for value in result.values[0])) == pytest.approx(1.0)


def test_scgpt_http_features_contract_uses_real_runtime_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_modules(monkeypatch, _FakeTorch())
    adapter = ScGPTExpert(checkpoint_dir=str(_bundle(tmp_path)), device="cpu")
    app = create_sidecar_app(
        identity=ModelIdentity(
            model_id="scgpt",
            model_version="0.2.4",
            adapter_version="1.0.0",
            code_revision="0cd3c73779e93e999789d52b4412e6c23baaa02b",
            weight_revision=f"sha256:{adapter.bundle_inventory_sha256}",
            capabilities=frozenset({"features"}),
        ),
        runtime=adapter,
    )
    with TestClient(app) as client:
        response = client.post(
            "/v1/features",
            json=_request(
                '{"genes":["G1","G2"],"values":[1,3],"value_semantics":"raw_counts"}'
            ).model_dump(
                mode="json",
                exclude_none=False,
            ),
        )
    assert response.status_code == 200
    body = response.json()
    assert body["expert_id"] == "scgpt"
    assert body["tensor"]["shape"] == [1, 2]
    assert body["tensor"]["values"] == pytest.approx([0.6, 0.8])
    assert body["semantics"]["tensor_role"] == "cell_embedding"
    assert body["properties"] == []


def test_scgpt_raw_counts_reject_fractional_values_before_model_load(tmp_path: Path) -> None:
    adapter = ScGPTExpert(checkpoint_dir=str(_bundle(tmp_path)), device="cpu")
    with pytest.raises(CandidateConversionError, match="raw_counts values must be integers"):
        adapter.encode(
            _request(
                '{"genes":["G1"],"values":[1.5],"value_semantics":"raw_counts"}'
            )
        )
    assert adapter.loaded is False
