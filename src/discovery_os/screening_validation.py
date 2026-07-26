"""Executed MatterSim/CHGNet screening between generation and DFT handoff.

This module is deliberately orchestration code, not another scientific model.
It preserves same-input feature evidence, executes each MLIP's independent
periodic relaxation, normalizes units, computes composition-relative
disagreement, applies an optional calibration artifact, and calls the
composition-scoped Pareto selector.  Missing evidence remains unknown.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Literal, Protocol

from pydantic import Field, model_validator

from .crystal_identity import (
    CrystalMatchRelation,
    classify_crystal_structure_relation,
)
from .fusion_exploration import ExpertEvidenceStore
from .fusion_schemas import ExpertFeaturePayload, FeatureStatus
from .hashing import stable_hash
from .materials_screening import (
    CandidateScreeningVector,
    MLIPScreeningPrediction,
    ParetoRankedScreening,
    classify_model_disagreement,
    rank_composition_scoped_pareto,
)
from .mlip_reliability import (
    CandidateProxyReliabilityAssessment,
    CandidateProxyRequest,
    CompositionEnergyPair,
    CompositionRelativeEnergyDisagreement,
    ExpertWeightRevision,
    ForceDisagreementMetrics,
    SplitConformalCalibrationArtifact,
    assess_candidate_proxy,
    composition_relative_energy_disagreement,
    force_disagreement_metrics,
)
from .relaxation import (
    PeriodicRelaxationPayload,
    PeriodicRelaxationRequest,
    PeriodicRelaxationSettings,
)
from .schemas import (
    Candidate,
    CandidateRef,
    CandidateType,
    Identifier,
    RepresentationKind,
    StrictSchema,
)


_PERIODIC_TYPES = {
    CandidateType.CRYSTAL,
    CandidateType.ALLOY,
    CandidateType.BATTERY_MATERIAL,
    CandidateType.CATALYST,
}


class PeriodicRelaxationPort(Protocol):
    @property
    def expert_id(self) -> str: ...

    def relax(self, request: PeriodicRelaxationRequest) -> PeriodicRelaxationPayload: ...


class RelaxationAttemptReceipt(StrictSchema):
    """Execution receipt that does not confuse HTTP failure with non-convergence."""

    expert_id: Literal["mattersim", "chgnet"]
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_status: Literal["succeeded", "failed"]
    cache_hit: bool = False
    payload: PeriodicRelaxationPayload | None = None
    error_type: Identifier | None = None
    error: str | None = Field(default=None, max_length=4_000)

    @model_validator(mode="after")
    def _execution_contract(self) -> RelaxationAttemptReceipt:
        if self.execution_status == "succeeded":
            if self.payload is None or self.error_type is not None or self.error is not None:
                raise ValueError("successful relaxation requires a payload and no error")
            if self.payload.expert_id != self.expert_id:
                raise ValueError("relaxation payload belongs to another expert")
        elif self.payload is not None or self.error_type is None or self.error is None:
            raise ValueError("failed relaxation requires a typed error and no payload")
        if self.cache_hit and self.execution_status != "succeeded":
            raise ValueError("only successful relaxation payloads may be cache hits")
        return self


class CandidateMaterialScreeningReceipt(StrictSchema):
    """All model, relaxation, calibration, and Pareto evidence for one candidate."""

    candidate_ref: CandidateRef
    composition_key: Identifier | None = None
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["complete", "unknown", "invalid_geometry"]
    common_geometry_alignment_id: Identifier | None = None
    force_disagreement: ForceDisagreementMetrics | None = None
    composition_relative_energy: CompositionRelativeEnergyDisagreement | None = None
    mattersim_relaxation: RelaxationAttemptReceipt
    chgnet_relaxation: RelaxationAttemptReceipt
    proxy_request: CandidateProxyRequest | None = None
    proxy_reliability: CandidateProxyReliabilityAssessment | None = None
    screening_vector: CandidateScreeningVector | None = None
    pareto_rank: ParetoRankedScreening | None = None
    dft_escalation: bool
    reasons: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _receipt_is_consistent(self) -> CandidateMaterialScreeningReceipt:
        if (self.proxy_request is None) != (self.proxy_reliability is None):
            raise ValueError("proxy request and reliability result must be present together")
        if self.status in {"complete", "invalid_geometry"} and self.screening_vector is None:
            raise ValueError("executed screening requires a screening vector")
        if self.status == "unknown" and self.screening_vector is not None:
            raise ValueError("unknown screening cannot expose a screening vector")
        if self.pareto_rank is not None:
            if self.screening_vector is None:
                raise ValueError("Pareto rank requires a complete screening vector")
            if self.pareto_rank.candidate_ref != self.candidate_ref:
                raise ValueError("Pareto rank belongs to another candidate")
        if self.screening_vector is not None:
            if self.screening_vector.candidate_ref != self.candidate_ref:
                raise ValueError("screening vector belongs to another candidate")
            if self.composition_relative_energy is None:
                raise ValueError("complete screening must retain relative-energy evidence")
        if self.status in {"unknown", "invalid_geometry"} and not self.dft_escalation:
            raise ValueError("unknown or invalid screening must fail closed")
        return self


class MaterialScreeningBatchReceipt(StrictSchema):
    batch_id: Identifier
    receipts: list[CandidateMaterialScreeningReceipt] = Field(min_length=1)
    pareto_ranking: list[ParetoRankedScreening] = Field(default_factory=list)
    relaxations_executed: int = Field(ge=0)
    relaxation_cache_hits: int = Field(ge=0)
    scientific_claim: Literal["diagnostic_screening_only"] = (
        "diagnostic_screening_only"
    )

    @model_validator(mode="after")
    def _batch_is_consistent(self) -> MaterialScreeningBatchReceipt:
        refs = [stable_hash(item.candidate_ref) for item in self.receipts]
        if len(refs) != len(set(refs)):
            raise ValueError("screening batch contains duplicate candidate refs")
        ranked = {stable_hash(item.candidate_ref) for item in self.pareto_ranking}
        complete = {
            stable_hash(item.candidate_ref)
            for item in self.receipts
            if item.screening_vector is not None
        }
        if ranked != complete:
            raise ValueError("Pareto ranking must cover every complete screening vector")
        if self.relaxation_cache_hits > 2 * len(self.receipts):
            raise ValueError("relaxation cache hits exceed the two-expert batch bound")
        return self


class MaterialScreeningValidationRunner:
    """Execute and cache the bounded two-MLIP material validation panel."""

    def __init__(
        self,
        evidence_store: ExpertEvidenceStore,
        *,
        mattersim_client: PeriodicRelaxationPort,
        chgnet_client: PeriodicRelaxationPort,
        settings: PeriodicRelaxationSettings | None = None,
        calibration: SplitConformalCalibrationArtifact | None = None,
    ) -> None:
        if mattersim_client.expert_id != "mattersim":
            raise ValueError("mattersim_client must identify expert 'mattersim'")
        if chgnet_client.expert_id != "chgnet":
            raise ValueError("chgnet_client must identify expert 'chgnet'")
        self.evidence_store = evidence_store
        self.mattersim_client = mattersim_client
        self.chgnet_client = chgnet_client
        self.settings = PeriodicRelaxationSettings.model_validate_json(
            (settings or PeriodicRelaxationSettings()).model_dump_json(),
            strict=True,
        )
        self.calibration = (
            None
            if calibration is None
            else SplitConformalCalibrationArtifact.model_validate_json(
                calibration.model_dump_json(),
                strict=True,
            )
        )
        self._relaxation_cache: dict[str, PeriodicRelaxationPayload] = {}

    def evaluate(
        self,
        candidates: Sequence[Candidate],
        evidence_ids_by_candidate: dict[str, list[str]],
        *,
        seed: int = 0,
    ) -> MaterialScreeningBatchReceipt:
        if not candidates:
            raise ValueError("material screening requires at least one candidate")
        if isinstance(seed, bool) or seed < 0:
            raise ValueError("material screening seed must be a non-negative integer")
        validated = [
            Candidate.model_validate_json(item.model_dump_json(), strict=True)
            for item in candidates
        ]
        keys = [_ref_key(_required_ref(item)) for item in validated]
        if len(keys) != len(set(keys)):
            raise ValueError("material screening candidates must have unique refs")

        prepared: list[dict[str, object]] = []
        executed = 0
        cache_hits = 0
        for candidate in validated:
            ref = _required_ref(candidate)
            evidence_ids = evidence_ids_by_candidate.get(_ref_key(ref), [])
            row, row_executed, row_hits = self._prepare_candidate(
                candidate,
                evidence_ids,
                seed=seed,
            )
            prepared.append(row)
            executed += row_executed
            cache_hits += row_hits

        relative_by_ref = self._relative_energy_panel(prepared)
        receipts: list[CandidateMaterialScreeningReceipt] = []
        vectors: list[CandidateScreeningVector] = []
        for row in prepared:
            receipt = self._candidate_receipt(
                row,
                relative_by_ref.get(
                    _relative_key(
                        row["candidate_ref"].candidate_id,  # type: ignore[union-attr]
                        str(row["composition"]),
                    )
                ),
            )
            receipts.append(receipt)
            if receipt.screening_vector is not None:
                vectors.append(receipt.screening_vector)

        # Always invoke the reviewed Pareto implementation, including an empty
        # vector panel, so the central path cannot silently substitute a local
        # ranking heuristic.
        ranking = rank_composition_scoped_pareto(vectors)
        rank_by_ref = {_ref_key(item.candidate_ref): item for item in ranking}
        receipts = [
            CandidateMaterialScreeningReceipt.model_validate_json(
                item.model_copy(
                    update={"pareto_rank": rank_by_ref.get(_ref_key(item.candidate_ref))}
                ).model_dump_json(),
                strict=True,
            )
            for item in receipts
        ]
        batch_id = "MSCREEN-" + stable_hash(
            {
                "candidate_cache_keys": [item.cache_key for item in receipts],
                "pareto_ranking": ranking,
                "settings": self.settings,
                "calibration": (
                    self.calibration.artifact_id
                    if self.calibration is not None
                    else None
                ),
            }
        )[:32]
        return MaterialScreeningBatchReceipt(
            batch_id=batch_id,
            receipts=receipts,
            pareto_ranking=ranking,
            relaxations_executed=executed,
            relaxation_cache_hits=cache_hits,
        )

    def _prepare_candidate(
        self,
        candidate: Candidate,
        evidence_ids: Sequence[str],
        *,
        seed: int,
    ) -> tuple[dict[str, object], int, int]:
        ref = _required_ref(candidate)
        composition = _composition_key(candidate)
        cache_key = stable_hash(
            {
                "candidate_ref": ref,
                "evidence_ids": sorted(evidence_ids),
                "settings": self.settings,
                "calibration": (
                    self.calibration.artifact_id
                    if self.calibration is not None
                    else None
                ),
            }
        )
        base: dict[str, object] = {
            "candidate": candidate,
            "candidate_ref": ref,
            "composition": composition,
            "cache_key": cache_key,
            "reasons": [],
        }
        if candidate.candidate_type not in _PERIODIC_TYPES:
            reason = "candidate_type_is_not_supported_by_periodic_relaxation"
            base["reasons"] = [reason]
            base["mattersim_attempt"] = _failed_attempt("mattersim", cache_key, reason)
            base["chgnet_attempt"] = _failed_attempt("chgnet", cache_key, reason)
            return base, 0, 0
        if composition is None:
            base["reasons"] = ["reduced_composition_is_missing"]

        try:
            mattersim_payload, chgnet_payload = self._common_payloads(
                ref,
                evidence_ids,
            )
            (
                common_mattersim,
                mattersim_forces,
            ) = _common_geometry_prediction(mattersim_payload)
            common_chgnet, chgnet_forces = _common_geometry_prediction(chgnet_payload)
            entity_ids = mattersim_payload.semantics.entity_ids  # type: ignore[union-attr]
            chgnet_entity_ids = chgnet_payload.semantics.entity_ids  # type: ignore[union-attr]
            if not entity_ids or entity_ids != chgnet_entity_ids:
                raise ValueError(
                    "MatterSim and CHGNet atom entity_ids are absent or misaligned"
                )
            force_metrics = force_disagreement_metrics(
                mattersim_forces,
                chgnet_forces,
            )
            common_alignment = "CGALIGN-" + stable_hash(
                {
                    "candidate_ref": ref,
                    "entity_ids": entity_ids,
                    "mattersim_feature": mattersim_payload.provenance,
                    "chgnet_feature": chgnet_payload.provenance,
                }
            )[:32]
            base.update(
                {
                    "mattersim_payload": mattersim_payload,
                    "chgnet_payload": chgnet_payload,
                    "common_mattersim": common_mattersim,
                    "common_chgnet": common_chgnet,
                    "force_metrics": force_metrics,
                    "common_alignment": common_alignment,
                }
            )
        except Exception as exc:
            reason = (
                "common_geometry_evidence_unavailable:"
                f"{type(exc).__name__}:{_bounded_error(exc)}"
            )
            base["reasons"] = [*base["reasons"], reason]  # type: ignore[list-item]
            base["mattersim_attempt"] = _failed_attempt("mattersim", cache_key, reason)
            base["chgnet_attempt"] = _failed_attempt("chgnet", cache_key, reason)
            return base, 0, 0

        attempts: dict[str, RelaxationAttemptReceipt] = {}
        executed = 0
        hits = 0
        for expert_id, client in (
            ("mattersim", self.mattersim_client),
            ("chgnet", self.chgnet_client),
        ):
            attempt, called, hit = self._relax(
                candidate,
                client,
                expert_id=expert_id,
                seed=seed,
            )
            attempts[expert_id] = attempt
            executed += int(called)
            hits += int(hit)
        base["mattersim_attempt"] = attempts["mattersim"]
        base["chgnet_attempt"] = attempts["chgnet"]
        return base, executed, hits

    def _common_payloads(
        self,
        candidate_ref: CandidateRef,
        evidence_ids: Sequence[str],
    ) -> tuple[ExpertFeaturePayload, ExpertFeaturePayload]:
        selected: dict[str, ExpertFeaturePayload] = {}
        for evidence_id in sorted(set(evidence_ids)):
            envelope = self.evidence_store.load(evidence_id)
            payload = envelope.payload
            if payload.candidate_ref != candidate_ref:
                raise ValueError("expert evidence belongs to another candidate")
            if payload.expert_id not in {"mattersim", "chgnet"}:
                continue
            if payload.expert_id in selected:
                raise ValueError(f"duplicate {payload.expert_id} evidence")
            if payload.status != FeatureStatus.SUCCESS:
                raise ValueError(f"{payload.expert_id} feature status is not success")
            selected[payload.expert_id] = payload
        missing = {"mattersim", "chgnet"} - set(selected)
        if missing:
            raise ValueError("missing required common-geometry evidence: " + ", ".join(sorted(missing)))
        return selected["mattersim"], selected["chgnet"]

    def _relax(
        self,
        candidate: Candidate,
        client: PeriodicRelaxationPort,
        *,
        expert_id: str,
        seed: int,
    ) -> tuple[RelaxationAttemptReceipt, bool, bool]:
        expert_seed = int(
            stable_hash(
                {
                    "base_seed": seed,
                    "candidate_ref": _required_ref(candidate),
                    "expert_id": expert_id,
                    "operation": "periodic-relaxation-v1",
                }
            )[:8],
            16,
        )
        request = PeriodicRelaxationRequest(
            candidate=candidate,
            settings=self.settings,
            seed=expert_seed,
        )
        request_hash = stable_hash(request)
        cached = self._relaxation_cache.get(request_hash)
        if cached is not None:
            return (
                RelaxationAttemptReceipt(
                    expert_id=expert_id,
                    request_hash=request_hash,
                    execution_status="succeeded",
                    cache_hit=True,
                    payload=cached,
                ),
                False,
                True,
            )
        try:
            payload = client.relax(request)
            payload = PeriodicRelaxationPayload.model_validate_json(
                payload.model_dump_json(),
                strict=True,
            )
            if payload.candidate_ref != _required_ref(candidate):
                raise ValueError("relaxation response belongs to another candidate")
            if payload.expert_id != expert_id:
                raise ValueError("relaxation response belongs to another expert")
        except Exception as exc:
            return (
                RelaxationAttemptReceipt(
                    expert_id=expert_id,
                    request_hash=request_hash,
                    execution_status="failed",
                    error_type=type(exc).__name__,
                    error=_bounded_error(exc),
                ),
                True,
                False,
            )
        self._relaxation_cache[request_hash] = payload
        return (
            RelaxationAttemptReceipt(
                expert_id=expert_id,
                request_hash=request_hash,
                execution_status="succeeded",
                payload=payload,
            ),
            True,
            False,
        )

    @staticmethod
    def _relative_energy_panel(
        rows: Sequence[dict[str, object]],
    ) -> dict[str, CompositionRelativeEnergyDisagreement]:
        complete_by_composition: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            composition = row["composition"]
            first = row.get("mattersim_attempt")
            second = row.get("chgnet_attempt")
            if (
                isinstance(composition, str)
                and isinstance(first, RelaxationAttemptReceipt)
                and isinstance(second, RelaxationAttemptReceipt)
                and first.payload is not None
                and second.payload is not None
            ):
                complete_by_composition[composition].append(row)
        pairs: list[CompositionEnergyPair] = []
        for composition, members in sorted(complete_by_composition.items()):
            alignment_id = "RELAXPANEL-" + stable_hash(
                {
                    "composition": composition,
                    "candidate_refs": sorted(
                        (_ref_key(item["candidate_ref"]) for item in members)
                    ),
                    "contract": "independently-relaxed-energy-per-atom-v1",
                }
            )[:32]
            for row in members:
                ref = row["candidate_ref"]
                first = row["mattersim_attempt"]
                second = row["chgnet_attempt"]
                assert isinstance(ref, CandidateRef)
                assert isinstance(first, RelaxationAttemptReceipt)
                assert isinstance(second, RelaxationAttemptReceipt)
                assert first.payload is not None and second.payload is not None
                pairs.append(
                    CompositionEnergyPair(
                        candidate_id=ref.candidate_id,
                        reduced_composition=composition,
                        first_model_id="mattersim",
                        second_model_id="chgnet",
                        first_energy_per_atom_eV=(
                            first.payload.final_energy_eV / first.payload.atom_count
                        ),
                        second_energy_per_atom_eV=(
                            second.payload.final_energy_eV / second.payload.atom_count
                        ),
                        alignment_artifact_id=alignment_id,
                    )
                )
        return {
            _relative_key(item.candidate_id, item.reduced_composition): item
            for item in composition_relative_energy_disagreement(pairs)
        }

    def _candidate_receipt(
        self,
        row: dict[str, object],
        relative_energy: CompositionRelativeEnergyDisagreement | None,
    ) -> CandidateMaterialScreeningReceipt:
        candidate = row["candidate"]
        ref = row["candidate_ref"]
        composition = row["composition"]
        cache_key = row["cache_key"]
        first = row["mattersim_attempt"]
        second = row["chgnet_attempt"]
        assert isinstance(candidate, Candidate)
        assert isinstance(ref, CandidateRef)
        assert isinstance(cache_key, str)
        assert isinstance(first, RelaxationAttemptReceipt)
        assert isinstance(second, RelaxationAttemptReceipt)
        reasons = list(row["reasons"])  # type: ignore[arg-type]
        if first.payload is None:
            reasons.append("mattersim_relaxation_execution_failed")
        if second.payload is None:
            reasons.append("chgnet_relaxation_execution_failed")
        if (
            not isinstance(composition, str)
            or first.payload is None
            or second.payload is None
            or "common_mattersim" not in row
            or "common_chgnet" not in row
            or "force_metrics" not in row
        ):
            return CandidateMaterialScreeningReceipt(
                candidate_ref=ref,
                composition_key=composition if isinstance(composition, str) else None,
                cache_key=cache_key,
                status="unknown",
                mattersim_relaxation=first,
                chgnet_relaxation=second,
                dft_escalation=True,
                reasons=sorted(set(reasons or ["screening_evidence_incomplete"])),
            )

        common_first = row["common_mattersim"]
        common_second = row["common_chgnet"]
        force_metrics = row["force_metrics"]
        common_alignment = row["common_alignment"]
        assert isinstance(common_first, MLIPScreeningPrediction)
        assert isinstance(common_second, MLIPScreeningPrediction)
        assert isinstance(force_metrics, ForceDisagreementMetrics)
        assert isinstance(common_alignment, str)
        mattersim_prediction = _relaxed_prediction(first.payload)
        chgnet_prediction = _relaxed_prediction(second.payload)
        relaxed_match = _relaxed_structure_match(first.payload, second.payload)
        disagreement = classify_model_disagreement(
            common_first,
            common_second,
            force_rmse_eV_A=force_metrics.component_rmse_eV_A,
            relative_energy=relative_energy,
            relaxed_structure_match=relaxed_match,
            require_stress_comparison=True,
            require_relaxed_structure_comparison=True,
        )
        geometry_valid = (
            first.payload.geometry_gate.is_valid
            and second.payload.geometry_gate.is_valid
        )
        relaxation_gate_passed = (
            first.payload.strict_gate_passed
            and second.payload.strict_gate_passed
        )
        if not geometry_valid:
            reasons.append("one_or_more_relaxed_geometries_failed_validation")
        if not relaxation_gate_passed:
            reasons.append("one_or_more_strict_relaxation_gates_failed")
        proxy_request = self._proxy_request(
            candidate,
            mattersim_prediction,
            chgnet_prediction,
            row["mattersim_payload"],
            row["chgnet_payload"],
        )
        # This is intentionally called even without a calibration artifact:
        # assess_candidate_proxy then returns a typed, fail-closed OOD receipt.
        proxy_reliability = assess_candidate_proxy(
            proxy_request,
            self.calibration,
        )
        vector = CandidateScreeningVector(
            candidate_ref=ref,
            composition_key=composition,
            mattersim=mattersim_prediction,
            chgnet=chgnet_prediction,
            common_geometry_mattersim=common_first,
            common_geometry_chgnet=common_second,
            common_geometry_alignment_id=common_alignment,
            disagreement=disagreement,
            geometry_valid=geometry_valid,
            relaxation_gate_passed=relaxation_gate_passed,
        )
        status: Literal["complete", "invalid_geometry"] = (
            "complete" if geometry_valid else "invalid_geometry"
        )
        dft_escalation = (
            disagreement.dft_escalation
            or proxy_reliability.dft_escalation
            or not relaxation_gate_passed
            or not geometry_valid
        )
        reasons.extend(disagreement.uncertainty_reasons)
        reasons.extend(proxy_reliability.reasons)
        if not reasons:
            reasons.append("complete_two_mlip_screening_receipt")
        return CandidateMaterialScreeningReceipt(
            candidate_ref=ref,
            composition_key=composition,
            cache_key=cache_key,
            status=status,
            common_geometry_alignment_id=common_alignment,
            force_disagreement=force_metrics,
            composition_relative_energy=relative_energy,
            mattersim_relaxation=first,
            chgnet_relaxation=second,
            proxy_request=proxy_request,
            proxy_reliability=proxy_reliability,
            screening_vector=vector,
            dft_escalation=dft_escalation,
            reasons=sorted(set(reasons)),
        )

    def _proxy_request(
        self,
        candidate: Candidate,
        mattersim: MLIPScreeningPrediction,
        chgnet: MLIPScreeningPrediction,
        mattersim_payload: object,
        chgnet_payload: object,
    ) -> CandidateProxyRequest:
        if not isinstance(mattersim_payload, ExpertFeaturePayload) or not isinstance(
            chgnet_payload,
            ExpertFeaturePayload,
        ):
            raise TypeError("proxy reliability requires original expert payloads")
        composition = _composition_key(candidate)
        assert composition is not None
        calibration = self.calibration
        return CandidateProxyRequest(
            candidate_id=candidate.candidate_id,
            proxy_name="relaxed_force_envelope",
            unit="eV/angstrom",
            proxy_value=max(mattersim.max_force_eV_A, chgnet.max_force_eV_A),
            scope_id=(
                calibration.chemistry_scope.scope_id
                if calibration is not None
                else "unconfigured-calibration-scope"
            ),
            elements=_candidate_elements(candidate, composition),
            reduced_composition=composition,
            expert_weight_revisions=[
                ExpertWeightRevision(
                    expert_id=payload.expert_id,
                    weight_revision=payload.provenance.weight_revision,
                )
                for payload in sorted(
                    (mattersim_payload, chgnet_payload),
                    key=lambda item: item.expert_id,
                )
            ],
            dft_method=(
                calibration.dft_method
                if calibration is not None
                else "no-reference-dft-calibration-configured"
            ),
            dft_reference_hash=(
                calibration.dft_reference_hash
                if calibration is not None
                else stable_hash("no-reference-dft-calibration-configured")
            ),
        )


def build_http_material_screening_runner(
    *,
    registry: object,
    evidence_store: ExpertEvidenceStore,
    settings: PeriodicRelaxationSettings | None = None,
    calibration: SplitConformalCalibrationArtifact | None = None,
) -> MaterialScreeningValidationRunner | None:
    """Auto-bind relax clients only when both registered experts are HTTP sidecars."""

    from .fusion_adapters import HttpExpertEncoder
    from .fusion_registry import ExpertRegistry

    if not isinstance(registry, ExpertRegistry):
        return None
    try:
        mattersim = registry.get("mattersim")
        chgnet = registry.get("chgnet")
    except KeyError:
        return None
    if not isinstance(mattersim, HttpExpertEncoder) or not isinstance(
        chgnet,
        HttpExpertEncoder,
    ):
        return None
    return MaterialScreeningValidationRunner(
        evidence_store,
        mattersim_client=mattersim.periodic_relaxation_client(),
        chgnet_client=chgnet.periodic_relaxation_client(),
        settings=settings,
        calibration=calibration,
    )


def _common_geometry_prediction(
    payload: ExpertFeaturePayload,
) -> tuple[MLIPScreeningPrediction, list[list[float]]]:
    tensor = payload.tensor
    semantics = payload.semantics
    if tensor is None or semantics is None or len(tensor.shape) != 2:
        raise ValueError(f"{payload.expert_id} force tensor is missing or not a matrix")
    rows, columns = tensor.shape
    if columns < 3 or rows != len(semantics.entity_ids):
        raise ValueError(f"{payload.expert_id} force tensor/entity shape is invalid")
    unit_key = "tensor" if payload.expert_id == "mattersim" else "columns_0_2"
    if _normalized_unit(semantics.unit_semantics.get(unit_key)) != "ev/angstrom":
        raise ValueError(f"{payload.expert_id} force tensor unit is not eV/angstrom")
    forces = [
        [
            float(tensor.values[index * columns + offset])
            for offset in range(3)
        ]
        for index in range(rows)
    ]
    energy = _property(payload, "energy_per_atom", {"ev/atom"})
    max_force = _property(payload, "max_force", {"ev/angstrom"})
    stress = _optional_property(
        payload,
        "stress_norm",
        {"gpa", "ev/angstrom^3"},
    )
    return (
        MLIPScreeningPrediction(
            expert_id=payload.expert_id,
            energy_per_atom_eV=energy[0],
            max_force_eV_A=max_force[0],
            stress_norm=None if stress is None else stress[0],
            stress_unit=None if stress is None else stress[1],
        ),
        forces,
    )


def _relaxed_prediction(payload: PeriodicRelaxationPayload) -> MLIPScreeningPrediction:
    stress = payload.final_stress
    return MLIPScreeningPrediction(
        expert_id=payload.expert_id,
        energy_per_atom_eV=payload.final_energy_eV / payload.atom_count,
        max_force_eV_A=payload.final_max_force_eV_A,
        stress_norm=(
            None if stress is None else stress.frobenius_norm_eV_A3
        ),
        stress_unit=None if stress is None else "eV/angstrom^3",
    )


def _relaxed_structure_match(
    first: PeriodicRelaxationPayload,
    second: PeriodicRelaxationPayload,
) -> bool | None:
    try:
        assessment = classify_crystal_structure_relation(
            first.relaxed_structure.value,
            second.relaxed_structure.value,
        )
    except Exception:
        return None
    if assessment.relation == CrystalMatchRelation.AMBIGUOUS:
        return None
    return assessment.relation == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE


def _property(
    payload: ExpertFeaturePayload,
    name: str,
    units: set[str],
) -> tuple[float, str]:
    matches = [item for item in payload.properties if item.property_name == name]
    if len(matches) != 1:
        raise ValueError(f"{payload.expert_id} requires exactly one {name} property")
    normalized = _normalized_unit(matches[0].unit)
    if normalized not in units:
        raise ValueError(f"{payload.expert_id} {name} has unsupported unit")
    return float(matches[0].value), normalized


def _optional_property(
    payload: ExpertFeaturePayload,
    name: str,
    units: set[str],
) -> tuple[float, str] | None:
    matches = [item for item in payload.properties if item.property_name == name]
    if not matches:
        return None
    return _property(payload, name, units)


def _normalized_unit(value: str | None) -> str:
    return (
        (value or "")
        .strip()
        .casefold()
        .replace(" ", "")
        .replace("å", "angstrom")
        .replace("Å", "angstrom")
        .replace("å", "angstrom")
    )


def _composition_key(candidate: Candidate) -> str | None:
    for key in ("composition_key", "reduced_formula"):
        value = candidate.attributes.get(key)
        if isinstance(value, str) and value.strip():
            return "".join(value.split())
    formulas = [
        item
        for item in candidate.representations
        if item.kind == RepresentationKind.CHEMICAL_FORMULA
    ]
    canonical = [item for item in formulas if item.canonical]
    selected = canonical[0] if len(canonical) == 1 else formulas[0] if formulas else None
    if selected is None:
        return None
    return "".join(selected.value.split()) or None


def _candidate_elements(candidate: Candidate, composition: str) -> list[str]:
    import re

    raw = candidate.attributes.get("elements")
    if isinstance(raw, list):
        values = [str(item).strip() for item in raw if str(item).strip()]
    else:
        chemical_system = candidate.attributes.get("chemical_system")
        values = (
            [item.strip() for item in chemical_system.split("-") if item.strip()]
            if isinstance(chemical_system, str)
            else re.findall(r"[A-Z][a-z]?", composition)
        )
    unique = list(dict.fromkeys(values))
    if not unique:
        raise ValueError("candidate elements are unavailable for reliability scope")
    return unique


def _failed_attempt(
    expert_id: Literal["mattersim", "chgnet"],
    cache_key: str,
    reason: str,
) -> RelaxationAttemptReceipt:
    return RelaxationAttemptReceipt(
        expert_id=expert_id,
        request_hash=stable_hash({"cache_key": cache_key, "expert_id": expert_id}),
        execution_status="failed",
        error_type="ScreeningPreconditionError",
        error=reason,
    )


def _required_ref(candidate: Candidate) -> CandidateRef:
    if candidate.candidate_ref is None:
        raise ValueError("material screening requires immutable candidate_ref")
    return candidate.candidate_ref


def _ref_key(reference: object) -> str:
    if not isinstance(reference, CandidateRef):
        raise TypeError("candidate reference is invalid")
    return stable_hash(reference)


def _relative_key(candidate_id: str, composition: str) -> str:
    return stable_hash({"candidate_id": candidate_id, "composition": composition})


def _bounded_error(exc: Exception) -> str:
    return " ".join(str(exc).split())[:4_000] or "exception provided no message"


__all__ = [
    "CandidateMaterialScreeningReceipt",
    "MaterialScreeningBatchReceipt",
    "MaterialScreeningValidationRunner",
    "PeriodicRelaxationPort",
    "RelaxationAttemptReceipt",
    "build_http_material_screening_runner",
]
