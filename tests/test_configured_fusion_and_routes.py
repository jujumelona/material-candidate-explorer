from __future__ import annotations

import pytest

from discovery_os.configured_experts import build_expert_registry_from_environment
from discovery_os.configured_fusion import (
    FusionConfigurationError,
    build_fusion_backend_from_environment,
    build_generator_from_environment,
    build_generators_from_environment,
)
from discovery_os.evidence_fusion import EvidenceDrivenFusionBackend
from discovery_os.fusion_runtime import FusionRuntime
from discovery_os.schemas import (
    Candidate,
    CandidateRepresentation,
    CandidateType,
    DiscoveryDomain,
    RepresentationKind,
)


def test_configured_experts_publish_explicit_routes() -> None:
    registry = build_expert_registry_from_environment(
        environ={},
        include_unconfigured=True,
    )
    descriptors = {item.expert_id: item for item in registry.describe()}

    assert all(descriptor.routes for descriptor in descriptors.values())
    unimol_routes = {
        route.modality: (route.feature_space, set(route.representation_kinds))
        for route in descriptors["unimol"].routes
    }
    assert unimol_routes["molecule_2d"] == ("unimol-cls-v1", {"smiles"})
    assert unimol_routes["molecule_3d"] == ("unimol-cls-v1", {"sdf", "xyz"})

    for descriptor in descriptors.values():
        for route in descriptor.routes:
            assert route.modality in descriptor.modalities
            assert route.feature_space in descriptor.feature_spaces
            assert set(route.representation_kinds) <= set(
                descriptor.supported_representations
            )


def test_configured_expert_binds_exported_runtime_parameters_hash() -> None:
    runtime_hash = "a" * 64
    registry = build_expert_registry_from_environment(
        environ={
            "UNIMOL_API_URL": "https://unimol.example.test",
            "UNIMOL_RUNTIME_PARAMETERS_HASH": runtime_hash,
        },
        include_unconfigured=True,
    )
    descriptor = registry.get("unimol").descriptor
    assert descriptor.metadata["parameters_hash"] == runtime_hash

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_expert_registry_from_environment(
            environ={
                "UNIMOL_API_URL": "https://unimol.example.test",
                "UNIMOL_RUNTIME_PARAMETERS_HASH": "A" * 64,
            }
        )


def test_fusion_backend_uses_fixed_url_and_bearer_token_environment() -> None:
    backend = build_fusion_backend_from_environment(
        environ={
            "FUSION_API_URL": "https://fusion.example.test/root/",
            "FUSION_API_TOKEN": " secret-token ",
        }
    )

    assert backend is not None
    assert backend.base_url == "https://fusion.example.test/root"
    assert backend.headers == {"Authorization": "Bearer secret-token"}
    assert backend.send_extended_request_context is False


def test_fusion_backend_extended_remote_context_is_explicitly_opted_in() -> None:
    backend = build_fusion_backend_from_environment(
        environ={
            "FUSION_API_URL": "https://fusion.example.test",
            "FUSION_SEND_EXTENDED_REQUEST_CONTEXT": "yes",
        }
    )

    assert backend.send_extended_request_context is True


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize(
    "environ",
    [
        {},
        {"FUSION_API_URL": "   "},
        {
            "FUSION_API_TOKEN": "must-not-be-forwarded",
            "FUSION_BACKEND_ID": "ignored-without-a-remote-url",
        },
    ],
)
def test_fusion_backend_missing_url_uses_local_evidence_controller(
    required: bool,
    environ: dict[str, str],
) -> None:
    backend = build_fusion_backend_from_environment(
        environ=environ,
        required=required,
    )

    assert isinstance(backend, EvidenceDrivenFusionBackend)


def test_plain_http_requires_explicit_loopback_opt_in() -> None:
    with pytest.raises(FusionConfigurationError, match="local HTTP"):
        build_fusion_backend_from_environment(
            environ={"FUSION_API_URL": "http://localhost:9000"}
        )

    backend = build_fusion_backend_from_environment(
        environ={
            "FUSION_API_URL": "http://127.0.0.1:9000",
            "FUSION_ALLOW_INSECURE_LOCAL_HTTP": "yes",
        }
    )
    assert backend is not None
    assert backend.base_url == "http://127.0.0.1:9000"

    with pytest.raises(FusionConfigurationError, match="must use HTTPS"):
        build_fusion_backend_from_environment(
            environ={
                "FUSION_API_URL": "http://fusion.example.test",
                "FUSION_ALLOW_INSECURE_LOCAL_HTTP": "1",
            }
        )


def test_generator_builder_is_manifest_allow_listed() -> None:
    generator = build_generator_from_environment(
        "mattergen",
        environ={
            "MATTERGEN_API_URL": "https://mattergen.example.test",
            "MATTERGEN_API_TOKEN": "mattergen-token",
        },
    )
    assert generator is not None
    assert generator.base_url == "https://mattergen.example.test"
    assert generator.headers == {"Authorization": "Bearer mattergen-token"}
    assert generator.expected_generator_id == "mattergen"
    assert generator.expected_generator_version == "1.0.3"
    assert generator.expected_code_revision == "842ffe735f7d06cec89d56aa23d9f001e1124b30"

    assert (
        build_generator_from_environment(
            "reinvent4",
            environ={},
            required=False,
        )
        is None
    )
    with pytest.raises(FusionConfigurationError, match="not an allow-listed generator"):
        build_generator_from_environment(
            "chemprop",
            environ={"CHEMPROP_API_URL": "https://chemprop.example.test"},
        )


def test_all_generator_builder_uses_each_manifest_url_and_token_name() -> None:
    generators = build_generators_from_environment(
        environ={
            "MATTERGEN_API_URL": "https://mattergen.example.test",
            "REINVENT_API_URL": "http://localhost:8112",
            "REINVENT_API_TOKEN": "reinvent-token",
            "REINVENT_WEIGHT_REVISION": "reinvent-prior-sha256-fixture",
            "REINVENT_ALLOW_INSECURE_LOCAL_HTTP": "1",
        }
    )

    assert set(generators) == {"mattergen", "reinvent4"}
    assert generators["reinvent4"].headers == {
        "Authorization": "Bearer reinvent-token"
    }


def test_overlapping_representations_do_not_duplicate_an_expert_route() -> None:
    registry = build_expert_registry_from_environment(
        environ={},
        include_unconfigured=True,
    )
    esm = registry.get("esm").descriptor
    candidate = Candidate(
        candidate_id="protein",
        candidate_type=CandidateType.PROTEIN,
        domain=DiscoveryDomain.MEDICINAL_CHEMISTRY,
        representations=[
            CandidateRepresentation(
                kind=RepresentationKind.PROTEIN_SEQUENCE,
                value="MKT",
            ),
            CandidateRepresentation(
                kind=RepresentationKind.FASTA,
                value=">protein\nMKT",
            ),
        ],
    )

    routes = FusionRuntime._matching_routes(esm, candidate, modality=None)

    assert routes.count(("protein_sequence", "esm-sequence-v1")) == 1
