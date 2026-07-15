"""Deterministic evidence controller for the unified fusion contract.

The controller deliberately does *not* combine specialist embeddings.  Its
eight-value latent is a compact search-state record; scientific ranking stays
with the evidence store, deterministic exploration selector, and scheduler.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

from .fusion_schemas import (
    ChangeAxis,
    DesiredChange,
    DiagnosticProperty,
    FeatureStatus,
    FusionFeatureInput,
    FusionOutput,
    FusionRequest,
    FusionRevisionProposal,
    FusionRevisionRequest,
    NumericTensor,
    TensorDType,
    WorkspaceEntityRole,
)
from .schemas import (
    CandidateType,
    DiscoveryDomain,
    ObjectiveDirection,
    PropertyObjective,
    RepresentationKind,
)


MATTERGEN_SUPPORTED_CONDITIONS = frozenset(
    {
        "chemical_system",
        "space_group",
        "dft_mag_density",
        "dft_band_gap",
        "ml_bulk_modulus",
        "hhi_score",
        "energy_above_hull",
    }
)

_BACKEND_ID = "evidence-rule-fusion"
_BACKEND_VERSION = "1.0.0"
_CODE_REVISION = "deterministic-evidence-controller-v1"
_WEIGHT_REVISION = "no-learned-weights"
_HULL_STABLE_THRESHOLD = 0.03
_HULL_EXPLORATION_TARGET = 0.05
_DISAGREEMENT_THRESHOLD = 0.25
_HULL_UNIT = "eV/atom"
_HULL_MILLI_UNIT = "meV/atom"
_PERIODIC_SYMBOLS = frozenset(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni
    Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe
    Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg
    Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg
    Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og""".split()
)


class EvidenceDrivenFusionBackend:
    """Learning-free search controller backed only by structured evidence.

    Tensor values and feature spaces are never inspected.  The fixed latent
    layout is::

        [cycle, successful experts, non-successful experts, worst objective utility,
         expert disagreement, improved since previous round,
         structural collapse rate, guidance alpha]
    """

    def fuse(self, request: FusionRequest) -> FusionOutput:
        primary, ignored = _partition_primary_features(request)
        if not primary:
            raise ValueError(
                "evidence fusion requires at least one primary-candidate feature"
            )

        successful_experts = {
            item.payload.expert_id
            for item in primary
            if item.payload.status == FeatureStatus.SUCCESS
        }
        partial_experts = {
            item.payload.expert_id
            for item in primary
            if item.payload.status == FeatureStatus.PARTIAL
        }
        partial_only_experts = partial_experts - successful_experts
        non_successful_experts = (
            set(request.failed_expert_ids)
            | set(request.missing_expert_ids)
            | partial_only_experts
        ) - successful_experts

        objective_utilities = _objective_utilities(primary, request.goal.objectives)
        warnings = [
            "Expert tensor values were not combined; the latent records deterministic "
            "search-control metadata only.",
            "This rule-based state is not scientific evidence or a learned prediction.",
        ]
        if partial_only_experts:
            warnings.append(
                "Partial primary payloads were excluded from utilities, disagreement, "
                "and revision targets and counted as non-successful experts: "
                + ", ".join(sorted(partial_only_experts))
            )
        if not objective_utilities:
            warnings.append(
                "No unit-compatible primary expert property matched a numeric goal "
                "objective; worst objective utility uses the neutral 0.0 sentinel."
            )
        if request.missing_expert_ids:
            warnings.append(
                "Missing primary experts were counted in the non-successful evaluator "
                "axis (without claiming that they executed and failed): "
                + ", ".join(sorted(request.missing_expert_ids))
            )

        worst_utility = min(objective_utilities, default=0.0)
        disagreement = _expert_disagreement(primary, request.goal.objectives)
        previous_improvement = request.decision_context.previous_objective_improvement
        improved = 1.0 if previous_improvement is not None and previous_improvement > 0.0 else 0.0

        return FusionOutput(
            latent=NumericTensor(
                dtype=TensorDType.FLOAT64,
                shape=[8],
                values=[
                    float(request.cycle),
                    float(len(successful_experts)),
                    float(len(non_successful_experts)),
                    float(worst_utility),
                    float(disagreement),
                    improved,
                    float(request.decision_context.structural_collapse_rate),
                    float(request.decision_context.guidance_alpha),
                ],
            ),
            used_feature_ids=[item.feature_id for item in primary],
            ignored_feature_ids=[item.feature_id for item in ignored],
            backend_id=_BACKEND_ID,
            backend_version=_BACKEND_VERSION,
            code_revision=_CODE_REVISION,
            weight_revision=_WEIGHT_REVISION,
            warnings=warnings,
        )

    def propose_revision(
        self, request: FusionRevisionRequest
    ) -> FusionRevisionProposal:
        controller_values = _validated_controller_latent(request)
        primary = _primary_revision_features(request)
        properties = _properties_by_name(primary)
        objectives = {item.property_name: item for item in request.goal.objectives}
        disagreement = _expert_disagreement(primary, request.goal.objectives)

        changes, rule_rationale, has_concrete_target = _select_supported_changes(
            request,
            objectives=objectives,
            properties=properties,
            disagreement=disagreement,
        )
        evidence_changes, evidence_rationale = _evidence_guided_changes(request)
        if evidence_changes:
            evidence_keys = {
                (str(item.axis), item.property_name)
                for item in evidence_changes
            }
            changes = [
                item
                for item in changes
                if (str(item.axis), item.property_name) not in evidence_keys
            ]
            changes = [*evidence_changes, *changes]
            rule_rationale = f"{evidence_rationale} {rule_rationale}".strip()
            has_concrete_target = True

        material_route = request.goal.domain != DiscoveryDomain.MEDICINAL_CHEMISTRY
        if material_route and any(
            change.property_name not in MATTERGEN_SUPPORTED_CONDITIONS
            for change in changes
        ):
            raise RuntimeError("material evidence fusion produced an unsupported MatterGen condition")

        successful_count = controller_values[1]
        failed_count = controller_values[2]
        panel_size = successful_count + failed_count
        coverage = successful_count / panel_size if panel_size > 0.0 else 0.0
        confidence = min(0.95, max(0.0, 0.75 * coverage * (1.0 - disagreement)))
        if not has_concrete_target:
            confidence = 0.0
        if (
            disagreement >= _DISAGREEMENT_THRESHOLD
            or request.decision_context.exploration_branch == "expert_disagreement"
        ):
            confidence = min(confidence, 0.2)

        safety_notes = [
            "Learning-free rule controller; this proposal is not a scientific prediction.",
            (
                "Literature evidence only selects a search branch; specialist evaluators still "
                "determine scientific merit."
                if request.decision_context.evidence_branch_id is not None
                else "No live-literature branch was attached to this revision."
            ),
            (
                "Only explicit MatterGen-supported conditions may be emitted."
                if material_route
                else "Molecule hints are generator priors, not efficacy or safety scores."
            ),
        ]
        if (
            disagreement >= _DISAGREEMENT_THRESHOLD
            or request.decision_context.exploration_branch == "expert_disagreement"
        ):
            safety_notes.append(
                "High-disagreement candidate must remain available for additional expert "
                "evaluation instead of being treated as a settled elite."
            )
        if not has_concrete_target:
            safety_notes.append(
                "No supported evidence-backed numeric target was available; the preserve "
                "proposal intentionally supplies no target value."
            )

        preferred_generators = (
            ["reinvent4", "chemformer"]
            if request.goal.domain == DiscoveryDomain.MEDICINAL_CHEMISTRY
            or request.candidate.candidate_type == CandidateType.SMALL_MOLECULE
            else ["mattergen"]
        )
        branch_note = (
            f" Live-evidence branch {request.decision_context.evidence_branch_id!r} "
            f"({request.decision_context.evidence_branch_kind}) was used as a search prior."
            if request.decision_context.evidence_branch_id is not None
            else ""
        )
        return FusionRevisionProposal(
            parent_candidate_ref=request.candidate.candidate_ref,
            state_id=request.state.state_id,
            desired_changes=changes,
            preferred_generator_ids=preferred_generators,
            confidence=round(confidence, 12),
            rationale=(
                "Deterministic condition control from primary specialist properties. "
                f"{rule_rationale}{branch_note}"
            ),
            safety_notes=safety_notes,
        )


def _partition_primary_features(
    request: FusionRequest,
) -> tuple[list[FusionFeatureInput], list[FusionFeatureInput]]:
    primary_id = request.workspace.primary_entity_id
    primary = [item for item in request.features if item.workspace_entity_id == primary_id]
    ignored = [item for item in request.features if item.workspace_entity_id != primary_id]
    return primary, ignored


def _validated_controller_latent(request: FusionRevisionRequest) -> list[float]:
    expected_provenance = (
        _BACKEND_ID,
        _BACKEND_VERSION,
        _CODE_REVISION,
        _WEIGHT_REVISION,
    )
    actual_provenance = (
        request.state.backend_id,
        request.state.backend_version,
        request.state.code_revision,
        request.state.weight_revision,
    )
    if actual_provenance != expected_provenance:
        raise ValueError("revision state was not produced by evidence-rule-fusion v1")
    if request.latent.dtype != TensorDType.FLOAT64 or request.latent.shape != [8]:
        raise ValueError("evidence fusion revision requires a float64 latent with shape [8]")
    values = request.latent.values
    if len(values) != 8 or any(not math.isfinite(item) for item in values):
        raise ValueError("evidence fusion revision requires eight finite latent values")
    if values[0] != float(request.state.cycle):
        raise ValueError("evidence fusion latent cycle does not match its state")
    if any(value < 0.0 or not value.is_integer() for value in values[1:3]):
        raise ValueError("evidence fusion expert counts must be non-negative integers")
    if not 0.0 <= values[4] <= 1.0:
        raise ValueError("evidence fusion disagreement must be between zero and one")
    if values[5] not in {0.0, 1.0}:
        raise ValueError("evidence fusion improvement flag must be zero or one")
    if not 0.0 <= values[6] <= 1.0 or not 0.0 <= values[7] <= 1.0:
        raise ValueError("evidence fusion collapse rate and alpha must be bounded")
    expected_improvement = (
        1.0
        if request.decision_context.previous_objective_improvement is not None
        and request.decision_context.previous_objective_improvement > 0.0
        else 0.0
    )
    expected_controls = (
        expected_improvement,
        request.decision_context.structural_collapse_rate,
        request.decision_context.guidance_alpha,
    )
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
        for actual, expected in zip(values[5:8], expected_controls, strict=True)
    ):
        raise ValueError(
            "evidence fusion latent controls do not match the revision decision context"
        )
    return values


def _primary_revision_features(
    request: FusionRevisionRequest,
) -> list[FusionFeatureInput]:
    primary_id = next(
        item.entity_id
        for item in request.state.workspace_entities
        if item.role == WorkspaceEntityRole.PRIMARY_CANDIDATE
    )
    by_id = {item.feature_id: item for item in request.features}
    source_ids = set(request.state.source_feature_ids)
    if any(
        feature_id not in by_id
        or by_id[feature_id].workspace_entity_id != primary_id
        for feature_id in source_ids
    ):
        raise ValueError(
            "evidence fusion state source features must all belong to the primary entity"
        )
    return [
        item
        for item in request.features
        if item.feature_id in source_ids and item.workspace_entity_id == primary_id
    ]


def _objective_utilities(
    features: Iterable[FusionFeatureInput],
    objectives: Iterable[PropertyObjective],
) -> list[float]:
    objective_by_name = {item.property_name: item for item in objectives}
    values: list[float] = []
    rows = _unambiguous_success_properties(features)
    for property_name, objective in objective_by_name.items():
        scale = _objective_numeric_scale(property_name, objective)
        if scale is None:
            continue
        for value in _objective_expert_values(
            property_name,
            rows.get(property_name, {}),
            objective,
        ).values():
            utility = _utility(value, objective, objective_scale=scale)
            if utility is not None:
                values.append(utility)
    return values


def _utility(
    value: float,
    objective: PropertyObjective,
    *,
    objective_scale: float = 1.0,
) -> float | None:
    direction = ObjectiveDirection(objective.direction)
    if direction == ObjectiveDirection.MAXIMIZE:
        return value
    if direction == ObjectiveDirection.MINIMIZE:
        return -value
    if direction == ObjectiveDirection.TARGET:
        target = objective.target_value
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            return None
        return -abs(value - float(target) * objective_scale)
    if direction == ObjectiveDirection.RANGE:
        if objective.lower_bound is None or objective.upper_bound is None:
            return None
        lower = objective.lower_bound * objective_scale
        upper = objective.upper_bound * objective_scale
        if lower <= value <= upper:
            return 1.0
        distance = min(abs(value - lower), abs(value - upper))
        return -distance
    if direction == ObjectiveDirection.SATISFY:
        target = objective.target_value
        if not isinstance(target, bool) and isinstance(target, (int, float)):
            return 1.0 if value == float(target) * objective_scale else 0.0
        if (
            objective.lower_bound is not None
            and value < objective.lower_bound * objective_scale
        ):
            return 0.0
        if (
            objective.upper_bound is not None
            and value > objective.upper_bound * objective_scale
        ):
            return 0.0
        if objective.lower_bound is not None or objective.upper_bound is not None:
            return 1.0
    return None


def _expert_disagreement(
    features: Iterable[FusionFeatureInput],
    objectives: Iterable[PropertyObjective],
) -> float:
    objective_by_name = {item.property_name: item for item in objectives}
    grouped: dict[str, dict[str, float]] = {}
    rows = _unambiguous_success_properties(features)
    for name, objective in objective_by_name.items():
        scale = _objective_numeric_scale(name, objective)
        if scale is None:
            continue
        for expert_id, value in _objective_expert_values(
            name,
            rows.get(name, {}),
            objective,
        ).items():
            utility = _utility(value, objective, objective_scale=scale)
            if utility is None:
                continue
            grouped.setdefault(name, {})[expert_id] = utility

    spans: list[float] = []
    for name, expert_values in grouped.items():
        if len(expert_values) < 2:
            continue
        values = list(expert_values.values())
        low, high = min(values), max(values)
        scale = max(abs(low), abs(high), 1e-12)
        spans.append(min(1.0, abs(high - low) / scale))
    return round(max(spans, default=0.0), 12)


def _unit_matches(prop: DiagnosticProperty, objective: PropertyObjective) -> bool:
    return objective.unit is None or prop.unit == objective.unit


def _objective_numeric_scale(
    property_name: str,
    objective: PropertyObjective,
) -> float | None:
    if property_name != "energy_above_hull":
        return 1.0
    if objective.unit in {None, _HULL_UNIT}:
        return 1.0
    if objective.unit == _HULL_MILLI_UNIT:
        return 0.001
    return None


def _objective_expert_values(
    property_name: str,
    expert_rows: dict[str, DiagnosticProperty],
    objective: PropertyObjective,
) -> dict[str, float]:
    if property_name == "energy_above_hull":
        return _canonical_hull_expert_values(expert_rows, objective)
    compatible = {
        expert_id: prop
        for expert_id, prop in expert_rows.items()
        if _unit_matches(prop, objective)
    }
    if objective.unit is None and len({prop.unit for prop in compatible.values()}) > 1:
        return {}
    return {expert_id: prop.value for expert_id, prop in compatible.items()}


def _canonical_hull_expert_values(
    expert_rows: dict[str, DiagnosticProperty],
    objective: PropertyObjective | None,
) -> dict[str, float]:
    if objective is not None and _objective_numeric_scale(
        "energy_above_hull", objective
    ) is None:
        return {}
    units = {prop.unit for prop in expert_rows.values()}
    if objective is not None and objective.unit is None and len(units) > 1:
        return {}
    if not units.issubset({_HULL_UNIT, _HULL_MILLI_UNIT}):
        return {}
    return {
        expert_id: (
            prop.value * 0.001 if prop.unit == _HULL_MILLI_UNIT else prop.value
        )
        for expert_id, prop in expert_rows.items()
    }


def _properties_by_name(
    features: Iterable[FusionFeatureInput],
) -> dict[str, list[DiagnosticProperty]]:
    return {
        name: list(expert_rows.values())
        for name, expert_rows in _unambiguous_success_properties(features).items()
    }


def _unambiguous_success_properties(
    features: Iterable[FusionFeatureInput],
) -> dict[str, dict[str, DiagnosticProperty]]:
    grouped: dict[str, dict[str, list[DiagnosticProperty]]] = {}
    for feature in features:
        if feature.payload.status != FeatureStatus.SUCCESS:
            continue
        for prop in feature.payload.properties:
            if not prop.out_of_domain and math.isfinite(prop.value):
                grouped.setdefault(prop.property_name, {}).setdefault(
                    feature.payload.expert_id, []
                ).append(prop)
    result: dict[str, dict[str, DiagnosticProperty]] = {}
    for name, expert_rows in grouped.items():
        for expert_id, rows in expert_rows.items():
            unique = {(row.value, row.unit): row for row in rows}
            if len(unique) == 1:
                result.setdefault(name, {})[expert_id] = next(iter(unique.values()))
    return result



def _evidence_guided_changes(
    request: FusionRevisionRequest,
) -> tuple[list[DesiredChange], str]:
    """Convert validated live-literature branch hints into generator priors.

    The function never creates objective utilities.  It only emits explicit
    generation controls linked to claim identifiers already validated by the
    RAG evidence bundle.
    """

    context = request.decision_context
    if context.evidence_branch_id is None or not context.evidence_generator_hints:
        return [], ""
    hints = context.evidence_generator_hints
    claim_note = ", ".join(context.evidence_claim_ids)
    rationale_prefix = (
        f"Live evidence branch {context.evidence_branch_id} from claims {claim_note}: "
    )
    changes: list[DesiredChange] = []

    if request.goal.domain == DiscoveryDomain.MEDICINAL_CHEMISTRY or (
        request.candidate.candidate_type == CandidateType.SMALL_MOLECULE
    ):
        mapping = (
            "seed_entities",
            "scaffold_smiles",
            "target_contexts",
            "mechanisms",
            "avoid_entities",
            "hypothesis_subject",
            "bridge_entity",
            "hypothesis_target",
            "search_mode",
        )
        for key in mapping:
            value = hints.get(key)
            if value in (None, "", []):
                continue
            changes.append(
                DesiredChange(
                    axis=ChangeAxis.MOLECULAR_STRUCTURE,
                    direction="explore",
                    property_name=key,
                    target_value=value,
                    rationale=(
                        rationale_prefix
                        + f"use {key!r} only to define a molecule-generation branch."
                    ),
                )
            )
        return changes, (
            "Live biomedical evidence supplied molecule seed/scaffold/target/mechanism "
            "search priors; it did not alter evaluator scores."
            if changes
            else ""
        )

    material_keys = (
        "chemical_system",
        "space_group",
        "dft_mag_density",
        "dft_band_gap",
        "ml_bulk_modulus",
        "hhi_score",
        "energy_above_hull",
    )
    for key in material_keys:
        value = hints.get(key)
        if value in (None, "", []):
            continue
        if key == "space_group" and isinstance(value, list):
            value = value[0] if value else None
        value = _validated_condition_target(key, value)
        if value is None:
            continue
        changes.append(
            DesiredChange(
                axis=ChangeAxis.TARGET_PROPERTY,
                direction="target",
                property_name=key,
                target_value=value,
                rationale=(
                    rationale_prefix
                    + f"explore the evidence-linked MatterGen condition {key!r}."
                ),
            )
        )
    return changes, (
        "Live materials evidence supplied explicit supported generation conditions; "
        "it did not become a stability or property score."
        if changes
        else ""
    )

def _select_supported_changes(
    request: FusionRevisionRequest,
    *,
    objectives: dict[str, PropertyObjective],
    properties: dict[str, list[DiagnosticProperty]],
    disagreement: float,
) -> tuple[list[DesiredChange], str, bool]:
    changes: list[DesiredChange] = []
    rationales: list[str] = []
    emitted: set[str] = set()
    hull_objective = objectives.get("energy_above_hull")
    hull_rows = properties.get("energy_above_hull", [])
    hull_values = _unit_compatible_hull_values(hull_rows, hull_objective)
    if hull_objective is not None or hull_rows:
        target = _next_hull_target(
            hull_objective,
            hull_values,
            disagreement=disagreement,
            previous_improvement=(
                request.decision_context.previous_objective_improvement
            ),
            exploration_branch=request.decision_context.exploration_branch,
        )
        target = _validated_condition_target("energy_above_hull", target)
        if target is not None:
            changes.append(
                DesiredChange(
                    axis=ChangeAxis.TARGET_PROPERTY,
                    direction="target",
                    property_name="energy_above_hull",
                    target_value=target,
                    rationale=(
                        "Adjust the stability condition from the current primary expert "
                        "panel without interpreting embeddings."
                    ),
                )
            )
            emitted.add("energy_above_hull")
            rationales.append(
                f"energy_above_hull target selected as {target:.2f} eV/atom"
            )

    for objective in request.goal.objectives:
        if (
            objective.property_name not in MATTERGEN_SUPPORTED_CONDITIONS
            or objective.property_name in emitted
            or (
                objective.property_name == "energy_above_hull"
                and (hull_objective is not None or bool(hull_rows))
            )
        ):
            continue
        target = _objective_condition_target(
            objective,
            properties.get(objective.property_name, []),
        )
        target = _validated_condition_target(objective.property_name, target)
        if target is None:
            continue
        changes.append(
            DesiredChange(
                axis=ChangeAxis.TARGET_PROPERTY,
                direction="target",
                property_name=objective.property_name,
                target_value=target,
                rationale="Use only an explicit goal value or a current expert result.",
            )
        )
        emitted.add(objective.property_name)
        rationales.append(f"selected supported goal condition {objective.property_name!r}")

    if changes:
        return changes, "; ".join(rationales) + ".", True

    chemical_system = _chemical_system(request)
    if chemical_system is not None:
        return (
            [
                DesiredChange(
                    axis=ChangeAxis.TARGET_PROPERTY,
                    direction="target",
                    property_name="chemical_system",
                    target_value=chemical_system,
                    rationale="Preserve the parent candidate's explicit chemical system.",
                )
            ],
            "No supported numeric target was available; preserved the parent chemical system.",
            True,
        )

    return (
        [
            DesiredChange(
                axis=ChangeAxis.TARGET_PROPERTY,
                direction="preserve",
                property_name="chemical_system",
                target_value=None,
                rationale=(
                    "Fail safe because neither the goal nor primary evidence supplied a "
                    "concrete MatterGen-supported target."
                ),
            )
        ],
        "No concrete supported target was invented.",
        False,
    )


def _next_hull_target(
    objective: PropertyObjective | None,
    values: list[float],
    *,
    disagreement: float,
    previous_improvement: float | None,
    exploration_branch: str | None,
) -> float | None:
    if objective is not None:
        explicit = objective.target_value
        if not isinstance(explicit, bool) and isinstance(explicit, (int, float)):
            scale = _objective_numeric_scale("energy_above_hull", objective)
            if scale is None:
                return None
            return round(float(explicit) * scale, 8)
    if not values:
        return None

    stable_fraction = sum(value <= _HULL_STABLE_THRESHOLD for value in values) / len(values)
    if stable_fraction < 0.5:
        return 0.0
    if exploration_branch == "stability":
        return 0.0
    if (
        exploration_branch == "expert_disagreement"
        or disagreement >= _DISAGREEMENT_THRESHOLD
    ):
        return _HULL_STABLE_THRESHOLD
    if exploration_branch == "novelty":
        return 0.08
    if exploration_branch == "target_property":
        return _HULL_STABLE_THRESHOLD
    if exploration_branch == "pareto":
        return _HULL_EXPLORATION_TARGET
    if previous_improvement is not None and previous_improvement > 0.0:
        return 0.0
    return _HULL_EXPLORATION_TARGET


def _unit_compatible_hull_values(
    properties: list[DiagnosticProperty],
    objective: PropertyObjective | None,
) -> list[float]:
    if not properties:
        return []
    return list(
        _canonical_hull_expert_values(
            {str(index): item for index, item in enumerate(properties)},
            objective,
        ).values()
    )


def _objective_condition_target(
    objective: PropertyObjective,
    _properties: list[DiagnosticProperty],
) -> str | float | int | None:
    target = objective.target_value
    if isinstance(target, (str, int, float)) and not isinstance(target, bool):
        return target
    if (
        ObjectiveDirection(objective.direction) == ObjectiveDirection.RANGE
        and objective.lower_bound is not None
        and objective.upper_bound is not None
    ):
        return (objective.lower_bound + objective.upper_bound) / 2.0
    return None


def _chemical_system(request: FusionRevisionRequest) -> str | None:
    attribute = request.candidate.attributes.get("chemical_system")
    if isinstance(attribute, str):
        canonical = _canonical_chemical_system(attribute)
        if canonical is not None:
            return canonical
    for representation in request.candidate.representations:
        if representation.kind != RepresentationKind.CHEMICAL_FORMULA:
            continue
        if not re.fullmatch(r"(?:[A-Z][a-z]?(?:\d+(?:\.\d+)?)?)+", representation.value):
            continue
        elements = re.findall(r"[A-Z][a-z]?", representation.value)
        canonical = _canonical_chemical_system("-".join(elements))
        if canonical is not None:
            return canonical
    return None


def _validated_condition_target(
    name: str,
    target: str | float | int | None,
) -> str | float | int | None:
    if target is None or isinstance(target, bool):
        return None
    if name == "chemical_system":
        return _canonical_chemical_system(target) if isinstance(target, str) else None
    if name == "space_group":
        return target if isinstance(target, int) and 1 <= target <= 230 else None
    if name in MATTERGEN_SUPPORTED_CONDITIONS:
        if isinstance(target, (int, float)) and math.isfinite(float(target)):
            return float(target)
        return None
    return None


def _canonical_chemical_system(value: str) -> str | None:
    tokens = value.split("-")
    if (
        not tokens
        or any(not re.fullmatch(r"[A-Z][a-z]?", token) for token in tokens)
        or any(token not in _PERIODIC_SYMBOLS for token in tokens)
    ):
        return None
    return "-".join(sorted(set(tokens)))


__all__ = ["EvidenceDrivenFusionBackend", "MATTERGEN_SUPPORTED_CONDITIONS"]
