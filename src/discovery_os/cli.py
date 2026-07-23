"""Command-line demo and contract inspection utilities."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Sequence

from .artifacts import ArtifactStore
from .cache import JsonCache
from .compiler import PlanCompiler, RuntimePolicy
from .engine import DiscoveryEngine
from .configured_experts import build_expert_registry_from_environment
from .configured_fusion import (
    build_fusion_backend_from_environment,
    build_generator_from_environment,
)
from .fusion_loop import FusionLoopRunner, WorkspaceBenchmarkRunner
from .fusion_metrics import compare_workspace_snapshots
from .fusion_exploration import ExpertEvidenceStore
from .fusion_runtime import FusionRuntime
from .fusion_search import (
    FusionSearchReport,
    FusionSearchRunner,
    PersistedFusionSearchReport,
    SearchBudget,
    SearchControlPoint,
    SearchControlSweep,
)
from .fusion_schemas import (
    ExpertDescriptor,
    ExpertFeaturePayload,
    FeatureSemantics,
    FusionBatchIterationReport,
    FusionDecisionContext,
    FusionGenerationRequest,
    FusionGenerationResponse,
    FusionIterationReport,
    FusionOutput,
    FusionRequest,
    FusionRevisionProposal,
    FusionWorkspaceSnapshot,
    ScientificModality,
    UnifiedLatentStateRef,
    WorkspaceEntityInput,
    WorkspacePairedRunReport,
    WorkspaceRelation,
    WorkspaceRunConfig,
    WorkspaceComparisonReport,
)
from .generators import GeneratorRuntime, build_default_generator_registry
from .hashing import stable_hash
from .integration_manifest import load_integration_manifest
from .literature_rag import (
    JsonEvidenceIndex,
    LiteratureEvidencePolicy,
    LiteratureSource,
    RagEvidenceBundle,
    RagSearchPlan,
    build_literature_rag_from_environment,
    load_evidence_bundle,
    save_evidence_bundle,
)
from .mock_model import MockDiscoveryModel
from .dft_handoff import DFTInputHandoffReport, DFTInputManifest
from .novelty import ScientificNoveltyAssessment
from .profiles import VALIDATION_PROFILES
from .relaxation import PeriodicRelaxationPayload, PeriodicRelaxationRequest
from .validation_evidence import (
    McpEvidenceContract,
    ValidationEvidenceHandoff,
    ValidationEvidenceReport,
    ValidationEvidenceRequest,
    ValidationEvidenceRoute,
    ValidationHandoffContract,
    ValidatorAuthority,
    build_validation_evidence_router_from_environment,
)
from .runtime import ToolRuntime
from .schemas import (
    Candidate,
    DiscoveryDomain,
    DiscoveryGoal,
    EvidenceBatch,
    FinalReport,
    ToolCall,
    ValidationPlan,
)
from .store import JsonDiscoveryStore
from .tool_adapters import build_default_tool_registry


SCHEMA_TYPES = {
    item.__name__: item
    for item in [
        Candidate,
        DFTInputHandoffReport,
        DFTInputManifest,
        DiscoveryGoal,
        RagSearchPlan,
        RagEvidenceBundle,
        ExpertDescriptor,
        ExpertFeaturePayload,
        FeatureSemantics,
        EvidenceBatch,
        FinalReport,
        FusionOutput,
        FusionGenerationRequest,
        FusionGenerationResponse,
        FusionBatchIterationReport,
        FusionDecisionContext,
        FusionIterationReport,
        FusionRequest,
        FusionRevisionProposal,
        FusionWorkspaceSnapshot,
        FusionSearchReport,
        PersistedFusionSearchReport,
        PeriodicRelaxationPayload,
        PeriodicRelaxationRequest,
        SearchControlPoint,
        SearchControlSweep,
        ScientificNoveltyAssessment,
        ToolCall,
        UnifiedLatentStateRef,
        ValidationPlan,
        ValidationEvidenceReport,
        ValidationEvidenceRequest,
        ValidationEvidenceHandoff,
        ValidationEvidenceRoute,
        ValidationHandoffContract,
        ValidatorAuthority,
        McpEvidenceContract,
        WorkspaceComparisonReport,
        WorkspaceEntityInput,
        WorkspacePairedRunReport,
        WorkspaceRelation,
        WorkspaceRunConfig,
    ]
}


def build_demo_engine(output: Path, *, run_id: str | None = None) -> DiscoveryEngine:
    output = output.resolve()
    store = (
        JsonDiscoveryStore.resume(output, run_id)
        if run_id is not None
        else JsonDiscoveryStore(output)
    )
    tool_registry = build_default_tool_registry(include_placeholders=True)
    generator_registry = build_default_generator_registry(include_placeholders=True)
    artifact_store = ArtifactStore(output / store.run_id / "artifacts")
    cache = JsonCache(output / ".discovery_cache")
    return DiscoveryEngine(
        model=MockDiscoveryModel(),
        generator_runtime=GeneratorRuntime(generator_registry, allow_mock=True),
        tool_runtime=ToolRuntime(tool_registry, artifact_store, cache),
        store=store,
        plan_compiler=PlanCompiler(
            tool_registry,
            RuntimePolicy(
                allow_mock_tools=False,
                inject_mandatory_sanity_checks=True,
            ),
        ),
    )


def _demo(args: argparse.Namespace) -> int:
    engine = build_demo_engine(Path(args.output))
    report = engine.run(
        args.goal,
        max_cycles=args.max_cycles,
        domain_hint=args.domain,
    )
    report_path = (
        Path(args.output).resolve()
        / report.run_id
        / "artifacts"
        / "reports"
        / f"{report.run_id}-{stable_hash(report.model_dump(mode='json'))}.json"
    )
    summary = {
        "run_id": report.run_id,
        "domain": report.goal.domain,
        "candidates": len(report.candidate_reports),
        "statuses": {
            item.candidate.candidate_id: item.validation.status
            for item in report.candidate_reports
        },
        "stop_reason": report.stop_decision.reason_code,
        "report": str(report_path),
        "notice": "Mock candidates are known fixtures; no discovery claim is made.",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _profiles(_args: argparse.Namespace) -> int:
    rows = []
    for domain, profile in sorted(VALIDATION_PROFILES.items(), key=lambda item: str(item[0])):
        rows.append(
            {
                "domain": str(domain),
                "profile_id": profile.profile_id,
                "gates": [gate.gate_id for gate in profile.gates],
                "final_claim_level": profile.final_claim_level,
            }
        )
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def _schema(args: argparse.Namespace) -> int:
    schema_type = SCHEMA_TYPES.get(args.name)
    if schema_type is None:
        choices = ", ".join(sorted(SCHEMA_TYPES))
        raise SystemExit(f"unknown schema {args.name!r}; choose one of: {choices}")
    print(json.dumps(schema_type.model_json_schema(), ensure_ascii=False, indent=2))
    return 0


def _integrations(args: argparse.Namespace) -> int:
    manifest = load_integration_manifest()
    if args.profile is None:
        payload = {
            "manifest_revision": manifest.manifest_revision,
            "profiles": {
                name: {
                    "description": profile.description,
                    "components": profile.components,
                }
                for name, profile in sorted(manifest.profiles.items())
            },
        }
    else:
        payload = {
            "manifest_revision": manifest.manifest_revision,
            "profile": args.profile,
            "components": [
                {
                    "component_id": item.component_id,
                    "display_name": item.display_name,
                    "role": item.role,
                    "python": item.install.python,
                    "install_kind": item.install.kind,
                    "package": item.install.package,
                    "version": item.install.version,
                    "source_revision": item.source.revision if item.source else None,
                    "platforms": item.platforms,
                    "accelerators": item.accelerators,
                    "api_url_env": item.api.base_url_env if item.api else None,
                    "weights": [
                        {
                            "weight_id": weight.weight_id,
                            "kind": weight.kind,
                            "revision": weight.revision,
                            "gated": weight.gated,
                        }
                        for weight in item.weights
                    ],
                    "license": item.license.code_license,
                    "requires_acceptance": item.license.requires_acceptance,
                    "notes": item.notes,
                }
                for item in manifest.resolve_profile(args.profile)
            ],
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _experts(_args: argparse.Namespace) -> int:
    registry = build_expert_registry_from_environment(include_unconfigured=True)
    print(
        json.dumps(
            [item.model_dump(mode="json") for item in registry.describe()],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _fusion_compare(args: argparse.Namespace) -> int:
    goal = DiscoveryGoal.model_validate_json(
        Path(args.goal).read_text(encoding="utf-8"),
        strict=True,
    )
    off = FusionWorkspaceSnapshot.model_validate_json(
        Path(args.off_snapshot).read_text(encoding="utf-8"),
        strict=True,
    )
    on = FusionWorkspaceSnapshot.model_validate_json(
        Path(args.on_snapshot).read_text(encoding="utf-8"),
        strict=True,
    )
    report = compare_workspace_snapshots(
        off,
        on,
        goal,
        artifact_store=ArtifactStore(Path(args.artifact_root).resolve()),
    )
    print(report.model_dump_json(indent=2))
    return 0


def _model_from_file(model_type, path: str):
    return model_type.model_validate_json(
        Path(path).read_text(encoding="utf-8"),
        strict=True,
    )


def _model_list_from_file(model_type, path: str | None) -> list:
    if path is None:
        return []
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"{path} must contain a top-level JSON array")
    return [model_type.model_validate(item, strict=True) for item in raw]


def _fusion_runtime_from_environment(artifacts: str) -> FusionRuntime:
    registry = build_expert_registry_from_environment(include_unconfigured=False)
    backend = build_fusion_backend_from_environment(required=True)
    return FusionRuntime(
        registry,
        backend,
        ArtifactStore(Path(artifacts).resolve()),
    )


def _fusion_iterate(args: argparse.Namespace) -> int:
    goal = _model_from_file(DiscoveryGoal, args.goal)
    parent = _model_from_file(Candidate, args.parent)
    config = _model_from_file(WorkspaceRunConfig, args.run_config)
    previous = (
        _model_from_file(UnifiedLatentStateRef, args.previous_state)
        if args.previous_state
        else None
    )
    decision_context = (
        _model_from_file(FusionDecisionContext, args.decision_context)
        if args.decision_context
        else None
    )
    runtime = _fusion_runtime_from_environment(args.artifacts)
    generator = build_generator_from_environment(args.generator, required=True)
    if generator is None:
        raise SystemExit(f"generator {args.generator!r} is not configured")
    report = FusionLoopRunner(runtime, generator).iterate(
        goal=goal,
        parent_candidate=parent,
        cycle=args.cycle,
        run_config=config,
        previous_state=previous,
        expert_ids=args.expert,
        context_entities=_model_list_from_file(WorkspaceEntityInput, args.context),
        relations=_model_list_from_file(WorkspaceRelation, args.relations),
        workspace_id=args.workspace_id,
        decision_context=decision_context,
    )
    print(report.model_dump_json(indent=2))
    return 0


def _validation_evidence(args: argparse.Namespace) -> int:
    request = _model_from_file(ValidationEvidenceRequest, args.request)
    goal = _model_from_file(DiscoveryGoal, args.goal) if args.goal else None
    run = build_validation_evidence_router_from_environment(
        artifact_root=args.artifacts,
    ).run(request, goal=goal)
    print(run.report.model_dump_json(indent=2))
    return 0


def _fusion_pair(args: argparse.Namespace) -> int:
    goal = _model_from_file(DiscoveryGoal, args.goal)
    parent = _model_from_file(Candidate, args.parent)
    off_config = _model_from_file(WorkspaceRunConfig, args.off_config)
    on_config = _model_from_file(WorkspaceRunConfig, args.on_config)
    runtime = _fusion_runtime_from_environment(args.artifacts)
    generator = build_generator_from_environment(args.generator, required=True)
    if generator is None:
        raise SystemExit(f"generator {args.generator!r} is not configured")
    report = WorkspaceBenchmarkRunner(runtime, generator).run_pair(
        goal=goal,
        parent_candidate=parent,
        cycle=args.cycle,
        off_config=off_config,
        on_config=on_config,
        expert_ids=args.expert,
        context_entities=_model_list_from_file(WorkspaceEntityInput, args.context),
        relations=_model_list_from_file(WorkspaceRelation, args.relations),
        workspace_id=args.workspace_id,
    )
    print(report.model_dump_json(indent=2))
    return 0


def _parse_iso_date(value: str | None) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"invalid ISO date {value!r}; expected YYYY-MM-DD") from exc


def _rag_update(args: argparse.Namespace) -> int:
    goal = _model_from_file(DiscoveryGoal, args.goal) if args.goal else None
    selected_sources = (
        [LiteratureSource(item) for item in args.source]
        if args.source
        else list(LiteratureSource)
    )
    pipeline = build_literature_rag_from_environment(require_model=args.require_model)
    index = JsonEvidenceIndex(args.index) if args.index else None
    bundle = pipeline.run(
        args.prompt,
        goal=goal,
        sources=selected_sources,
        from_date=_parse_iso_date(args.from_date),
        to_date=_parse_iso_date(args.to_date),
        max_results_per_query=args.max_results,
        max_branches=args.max_branches,
        index=index,
    )
    target = save_evidence_bundle(bundle, args.output)
    summary = {
        "bundle": str(target.resolve()),
        "bundle_id": bundle.bundle_id,
        "records": len(bundle.records),
        "claims": len(bundle.claims),
        "graph_nodes": len(bundle.graph.nodes),
        "graph_edges": len(bundle.graph.edges),
        "branches": len(bundle.branches),
        "source_statuses": [item.model_dump(mode="json") for item in bundle.source_statuses],
        "scientific_role": bundle.scientific_role,
        "warnings": bundle.warnings,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _parse_control_point(value: str, *, index: int) -> SearchControlPoint:
    raw = value.strip()
    separator = ":" if ":" in raw else ","
    parts = [item.strip() for item in raw.split(separator)]
    if len(parts) != 2:
        raise SystemExit(
            f"invalid --control-point {value!r}; expected ALPHA:TEMPERATURE"
        )
    try:
        alpha = float(parts[0])
        temperature = float(parts[1])
    except ValueError as exc:
        raise SystemExit(
            f"invalid --control-point {value!r}; alpha and temperature must be numbers"
        ) from exc
    return SearchControlPoint(
        alpha=alpha,
        temperature=temperature,
        label=f"grid-{index:02d}",
    )


def _fusion_search(args: argparse.Namespace) -> int:
    goal = _model_from_file(DiscoveryGoal, args.goal)
    parent = _model_from_file(Candidate, args.parent)
    config = _model_from_file(WorkspaceRunConfig, args.run_config)
    previous = (
        _model_from_file(UnifiedLatentStateRef, args.previous_state)
        if args.previous_state
        else None
    )
    runtime = _fusion_runtime_from_environment(args.artifacts)
    evidence_policy = None
    if args.rag_bundle:
        evidence_policy = LiteratureEvidencePolicy(load_evidence_bundle(args.rag_bundle))
    elif args.rag_prompt:
        pipeline = build_literature_rag_from_environment(
            require_model=args.rag_require_model
        )
        bundle = pipeline.run(
            args.rag_prompt,
            goal=goal,
            sources=(
                [LiteratureSource(item) for item in args.rag_source]
                if args.rag_source
                else list(LiteratureSource)
            ),
            from_date=_parse_iso_date(args.rag_from_date),
            to_date=_parse_iso_date(args.rag_to_date),
            max_results_per_query=args.rag_max_results,
            max_branches=args.rag_max_branches,
            index=(JsonEvidenceIndex(args.rag_index) if args.rag_index else None),
        )
        rag_output = Path(args.artifacts).resolve() / "literature" / f"{bundle.bundle_id}.json"
        save_evidence_bundle(bundle, rag_output)
        evidence_policy = LiteratureEvidencePolicy(bundle)
    control_sweep = None
    if not args.no_control_sweep:
        raw_points = args.control_point or ["0.25:1.40", "0.50:1.00", "0.75:0.70"]
        control_sweep = SearchControlSweep(
            points=[
                _parse_control_point(value, index=index)
                for index, value in enumerate(raw_points)
            ],
            include_adaptive_center=True,
            max_variants_per_parent=args.max_control_variants,
        )
    generator = build_generator_from_environment(args.generator, required=True)
    if generator is None:
        raise SystemExit(f"generator {args.generator!r} is not configured")
    if (args.max_generation_calls is None) != (
        args.max_generated_candidates is None
    ):
        raise SystemExit(
            "--max-generation-calls and --max-generated-candidates must be set together"
        )
    search_budget = (
        SearchBudget(
            max_generation_calls=args.max_generation_calls,
            max_generated_candidates=args.max_generated_candidates,
        )
        if args.max_generation_calls is not None
        else None
    )
    persisted = FusionSearchRunner(
        FusionLoopRunner(runtime, generator),
        ExpertEvidenceStore(runtime.artifact_store),
    ).run(
        search_id=args.search_id,
        goal=goal,
        initial_candidate=parent,
        base_run_config=config,
        rounds=args.rounds,
        initial_cycle=args.cycle,
        initial_state=previous,
        expert_ids=args.expert,
        required_primary_evaluator_ids=args.required_evaluator,
        modality=(ScientificModality(args.modality) if args.modality else None),
        context_entities=_model_list_from_file(WorkspaceEntityInput, args.context),
        relations=_model_list_from_file(WorkspaceRelation, args.relations),
        workspace_id=args.workspace_id,
        frontier_width=args.frontier_width,
        evidence_policy=evidence_policy,
        control_sweep=control_sweep,
        search_budget=search_budget,
        ranking_limit=args.ranking_limit,
    )
    print(persisted.model_dump_json(indent=2))
    return 0


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="discovery-os",
        description="Evidence-driven material discovery orchestration",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the deterministic mock loop")
    demo.add_argument("--goal", required=True, help="scientific screening goal")
    demo.add_argument(
        "--domain",
        choices=[str(item) for item in DiscoveryDomain],
        default=None,
        help="optional domain hint; otherwise the mock model infers it",
    )
    demo.add_argument("--max-cycles", type=int, default=1)
    demo.add_argument("--output", default="runs")
    demo.set_defaults(handler=_demo)

    profiles = subparsers.add_parser("profiles", help="list code-owned validation gates")
    profiles.set_defaults(handler=_profiles)

    schema = subparsers.add_parser("schema", help="print a model-connection JSON Schema")
    schema.add_argument("name")
    schema.set_defaults(handler=_schema)

    integrations = subparsers.add_parser(
        "integrations",
        help="show pinned specialist environments, sources, weights, and API bindings",
    )
    integrations.add_argument("--profile", default=None)
    integrations.set_defaults(handler=_integrations)

    experts = subparsers.add_parser(
        "experts",
        help="show expert feature services configured through environment variables",
    )
    experts.set_defaults(handler=_experts)

    rag_update = subparsers.add_parser(
        "rag-update",
        help=(
            "search live scholarly sources, extract source-grounded claims, build an "
            "evidence graph, and produce generator search-prior branches"
        ),
    )
    rag_update.add_argument("--prompt", required=True)
    rag_update.add_argument("--goal", help="optional DiscoveryGoal JSON")
    rag_update.add_argument("--output", required=True, help="output RagEvidenceBundle JSON")
    rag_update.add_argument("--index", help="optional persistent evidence-index directory")
    rag_update.add_argument("--from-date")
    rag_update.add_argument("--to-date")
    rag_update.add_argument("--max-results", type=int, default=25)
    rag_update.add_argument("--max-branches", type=int, default=24)
    rag_update.add_argument(
        "--source",
        action="append",
        choices=[item.value for item in LiteratureSource],
    )
    rag_update.add_argument(
        "--require-model",
        action="store_true",
        help="fail unless RAG_MODEL_API_URL and RAG_MODEL_NAME are configured",
    )
    rag_update.set_defaults(handler=_rag_update)

    fusion_compare = subparsers.add_parser(
        "fusion-compare",
        help="compare paired workspace OFF/ON snapshots without creating a scientific claim",
    )
    fusion_compare.add_argument("--goal", required=True)
    fusion_compare.add_argument("--off-snapshot", required=True)
    fusion_compare.add_argument("--on-snapshot", required=True)
    fusion_compare.add_argument(
        "--artifact-root",
        required=True,
        help="content-addressed artifact root used to verify both snapshots",
    )
    fusion_compare.set_defaults(handler=_fusion_compare)

    fusion_iterate = subparsers.add_parser(
        "fusion-iterate",
        help="run revision -> generator -> expert re-extraction -> latent update",
    )
    fusion_iterate.add_argument("--goal", required=True)
    fusion_iterate.add_argument("--parent", required=True)
    fusion_iterate.add_argument("--run-config", required=True)
    fusion_iterate.add_argument("--generator", required=True)
    fusion_iterate.add_argument("--cycle", type=int, default=0)
    fusion_iterate.add_argument("--previous-state")
    fusion_iterate.add_argument(
        "--decision-context",
        help="optional FusionDecisionContext JSON with source-closed evidence hints",
    )
    fusion_iterate.add_argument("--context", help="JSON array of WorkspaceEntityInput")
    fusion_iterate.add_argument("--relations", help="JSON array of WorkspaceRelation")
    fusion_iterate.add_argument("--workspace-id")
    fusion_iterate.add_argument("--expert", action="append")
    fusion_iterate.add_argument("--artifacts", default=".discovery/fusion")
    fusion_iterate.set_defaults(handler=_fusion_iterate)

    fusion_pair = subparsers.add_parser(
        "fusion-pair",
        help="generate and evaluate a fail-closed paired workspace OFF/ON run",
    )
    fusion_pair.add_argument("--goal", required=True)
    fusion_pair.add_argument("--parent", required=True)
    fusion_pair.add_argument("--off-config", required=True)
    fusion_pair.add_argument("--on-config", required=True)
    fusion_pair.add_argument("--generator", required=True)
    fusion_pair.add_argument("--cycle", type=int, default=0)
    fusion_pair.add_argument("--context", help="JSON array of WorkspaceEntityInput")
    fusion_pair.add_argument("--relations", help="JSON array of WorkspaceRelation")
    fusion_pair.add_argument("--workspace-id")
    fusion_pair.add_argument("--expert", action="append")
    fusion_pair.add_argument("--artifacts", default=".discovery/fusion")
    fusion_pair.set_defaults(handler=_fusion_pair)

    validation_evidence = subparsers.add_parser(
        "validation-evidence",
        help="run one stage-specific official/RAG/MCP evidence retrieval",
    )
    validation_evidence.add_argument("--request", required=True)
    validation_evidence.add_argument("--goal")
    validation_evidence.add_argument(
        "--artifacts",
        default=".discovery",
    )
    validation_evidence.set_defaults(handler=_validation_evidence)

    fusion_search = subparsers.add_parser(
        "fusion-search",
        help="run adaptive multi-round Pareto/novelty/disagreement fusion search",
    )
    fusion_search.add_argument("--search-id", required=True)
    fusion_search.add_argument("--goal", required=True)
    fusion_search.add_argument("--parent", required=True)
    fusion_search.add_argument("--run-config", required=True)
    fusion_search.add_argument("--generator", required=True)
    fusion_search.add_argument("--rounds", required=True, type=int)
    fusion_search.add_argument("--frontier-width", type=int, default=4)
    fusion_search.add_argument(
        "--control-point",
        action="append",
        help=(
            "alpha:temperature operating point; repeat to sweep several points. "
            "Points move with the adaptive scheduler in later rounds."
        ),
    )
    fusion_search.add_argument(
        "--max-control-variants",
        type=int,
        default=3,
        help="maximum alpha/temperature attempts per parent candidate",
    )
    fusion_search.add_argument(
        "--no-control-sweep",
        action="store_true",
        help="use only the adaptive scheduler's single alpha/temperature point",
    )
    fusion_search.add_argument(
        "--ranking-limit",
        type=int,
        default=50,
        help="maximum number of final candidates emitted in unified ranked order",
    )
    fusion_search.add_argument(
        "--max-generation-calls",
        type=int,
        help="global generation-call budget across all rounds, branches, and variants",
    )
    fusion_search.add_argument(
        "--max-generated-candidates",
        type=int,
        help="global generated-candidate budget across the complete search",
    )
    fusion_search.add_argument("--cycle", type=int, default=0)
    fusion_search.add_argument("--previous-state")
    fusion_search.add_argument("--context", help="JSON array of WorkspaceEntityInput")
    fusion_search.add_argument("--relations", help="JSON array of WorkspaceRelation")
    fusion_search.add_argument("--workspace-id")
    fusion_search.add_argument(
        "--modality",
        choices=[item.value for item in ScientificModality],
    )
    fusion_search.add_argument(
        "--expert",
        action="append",
        help="expert id to evaluate; repeat for a fixed panel",
    )
    fusion_search.add_argument(
        "--required-evaluator",
        action="append",
        help="primary evaluator required by branch selection; repeat as needed",
    )
    fusion_search.add_argument("--artifacts", default=".discovery/fusion")
    rag_group = fusion_search.add_mutually_exclusive_group()
    rag_group.add_argument(
        "--rag-bundle",
        help="existing RagEvidenceBundle JSON used to allocate literature-guided branches",
    )
    rag_group.add_argument(
        "--rag-prompt",
        help="run live literature retrieval from this prompt before generation",
    )
    fusion_search.add_argument("--rag-index")
    fusion_search.add_argument("--rag-from-date")
    fusion_search.add_argument("--rag-to-date")
    fusion_search.add_argument("--rag-max-results", type=int, default=25)
    fusion_search.add_argument("--rag-max-branches", type=int, default=24)
    fusion_search.add_argument(
        "--rag-source",
        action="append",
        choices=[item.value for item in LiteratureSource],
    )
    fusion_search.add_argument("--rag-require-model", action="store_true")
    fusion_search.set_defaults(handler=_fusion_search)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


__all__ = ["build_demo_engine", "main", "make_parser"]
