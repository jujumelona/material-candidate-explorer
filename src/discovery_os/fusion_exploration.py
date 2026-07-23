"""Deterministic, evidence-preserving exploration utilities for Fusion Core.

This module deliberately operates on the original per-expert property vectors.
It never averages expert payloads or consumes a fused latent as scientific
evidence.  Failed, partial, out-of-domain, stale, or incomparable evaluations
are excluded rather than imputed.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from ._compat import StrEnum
from typing import Iterable, Literal

from pydantic import Field, model_validator

from .artifacts import ArtifactStore
from .fusion_schemas import (
    ContentArtifactRef,
    DiagnosticProperty,
    ExpertFeaturePayload,
    ExpertFeatureRef,
    FeatureStatus,
    GenerationControls,
)
from .hashing import bytes_hash, candidate_content_hash, canonical_json, stable_hash
from .schemas import (
    Candidate,
    CandidateRef,
    CandidateType,
    DiscoveryGoal,
    GoalConstraint,
    Identifier,
    NonEmptyText,
    ObjectiveDirection,
    PropertyObjective,
    RepresentationKind,
    StrictSchema,
)


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ExplorationError(RuntimeError):
    """Base error for deterministic exploration failures."""


class ExpertEvidenceIntegrityError(ExplorationError):
    """Raised when a stored payload, reference, or content hash is inconsistent."""


class ExpertEvidenceConflict(ExplorationError):
    """Raised when one cache key resolves to two different expert outputs."""


def _expert_evidence_cache_key(
    payload: ExpertFeaturePayload,
    goal_hash: str,
) -> str:
    """Mirror the immutable request semantics used by FusionRuntime's cache."""

    provenance = payload.provenance
    return stable_hash(
        {
            "contract": "expert-evidence-cache-v2",
            "workspace_entity_id": payload.workspace_entity_id,
            "candidate_ref": payload.candidate_ref,
            "goal_hash": goal_hash,
            "expert_id": payload.expert_id,
            "modality": payload.modality,
            "feature_space": payload.feature_space,
            "adapter_version": provenance.adapter_version,
            "model_version": provenance.model_version,
            "code_revision": provenance.code_revision,
            "weight_revision": provenance.weight_revision,
            "dataset_revision": provenance.dataset_revision,
            "projection_version": provenance.projection_version,
            "parameters_hash": provenance.parameters_hash,
            "seed": provenance.seed,
        }
    )


class ExplorationBranch(StrEnum):
    STABILITY = "stability"
    TARGET_PROPERTY = "target_property"
    NOVELTY = "novelty"
    EXPERT_DISAGREEMENT = "expert_disagreement"
    PARETO = "pareto"


class ExpertEvidenceEnvelope(StrictSchema):
    """Exact original payload and reference stored as one immutable object."""

    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload: ExpertFeaturePayload
    feature_ref: ExpertFeatureRef

    @model_validator(mode="after")
    def _payload_matches_reference(self) -> ExpertEvidenceEnvelope:
        payload = self.payload
        ref = self.feature_ref
        pairs = (
            (payload.workspace_entity_id, ref.workspace_entity_id, "workspace entity"),
            (payload.candidate_ref, ref.candidate_ref, "candidate"),
            (payload.expert_id, ref.expert_id, "expert"),
            (payload.modality, ref.modality, "modality"),
            (payload.feature_space, ref.feature_space, "feature space"),
            (payload.status, ref.status, "status"),
            (payload.semantics, ref.semantics, "semantics"),
            (payload.properties, ref.properties, "properties"),
            (payload.quality_flags, ref.quality_flags, "quality flags"),
            (payload.warnings, ref.warnings, "warnings"),
            (payload.provenance, ref.provenance, "provenance"),
        )
        for actual, expected, label in pairs:
            if actual != expected:
                raise ValueError(f"expert payload and reference {label} differ")
        if payload.tensor is None:
            if ref.tensor_dtype is not None or ref.tensor_shape:
                raise ValueError("tensor-free payload has tensor metadata in its reference")
        elif (
            ref.tensor_dtype != payload.tensor.dtype
            or ref.tensor_shape != payload.tensor.shape
        ):
            raise ValueError("expert payload and reference tensor metadata differ")
        if self.cache_key != _expert_evidence_cache_key(payload, ref.goal_hash):
            raise ValueError("expert evidence cache key does not match its request semantics")
        return self


class StoredExpertEvidence(StrictSchema):
    evidence_id: Identifier
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_ref: CandidateRef
    goal_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    workspace_entity_id: Identifier
    evaluator_id: Identifier
    feature_id: Identifier
    artifact: ContentArtifactRef


class ExpertEvidenceStore:
    """Content-addressed store with deterministic lookup indexes.

    Indexes are reconstructed from immutable envelopes on disk, so a fresh
    process can query evidence without trusting an auxiliary mutable database.
    """

    PREFIX = "fusion/expert-evidence/objects"
    MEDIA_TYPE = "application/vnd.discovery-os.expert-evidence+json"

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self._index_loaded = False
        self._records_by_id: dict[str, StoredExpertEvidence] = {}
        self._records_by_cache_key: dict[str, StoredExpertEvidence] = {}
        self._record_ids_by_candidate_ref: dict[str, set[str]] = {}
        self._record_ids_by_candidate_id: dict[str, set[str]] = {}
        self._record_ids_by_evaluator: dict[str, set[str]] = {}
        self._index_lock = threading.RLock()

    @staticmethod
    def cache_key_for(
        payload: ExpertFeaturePayload,
        *,
        goal_hash: str,
    ) -> str:
        """Build a cache key from immutable evaluation inputs/provenance.

        Output tensor/property values are intentionally excluded.  If a service
        emits different outputs for the same key, :meth:`put` reports a conflict
        rather than silently replacing or averaging them.
        """

        if not _SHA256.fullmatch(goal_hash):
            raise ValueError("expert evidence goal_hash must be a lowercase SHA-256")
        return _expert_evidence_cache_key(payload, goal_hash)

    def put(
        self,
        payload: ExpertFeaturePayload,
        feature_ref: ExpertFeatureRef,
        *,
        cache_key: str | None = None,
    ) -> StoredExpertEvidence:
        payload_copy = ExpertFeaturePayload.model_validate_json(
            payload.model_dump_json(), strict=True
        )
        ref_copy = ExpertFeatureRef.model_validate_json(
            feature_ref.model_dump_json(), strict=True
        )
        expected_key = self.cache_key_for(
            payload_copy,
            goal_hash=ref_copy.goal_hash,
        )
        resolved_key = cache_key or expected_key
        if not _SHA256.fullmatch(resolved_key):
            raise ValueError("expert evidence cache_key must be a lowercase SHA-256")
        if resolved_key != expected_key:
            raise ValueError("expert evidence cache_key does not match its request semantics")

        self._verify_source_payload(payload_copy, ref_copy)
        envelope = ExpertEvidenceEnvelope(
            cache_key=resolved_key,
            payload=payload_copy,
            feature_ref=ref_copy,
        )
        encoded = canonical_json(envelope).encode("utf-8")
        digest = bytes_hash(encoded)

        with self._index_lock:
            existing = self.by_cache_key(resolved_key)
            if existing:
                if len(existing) != 1:
                    raise ExpertEvidenceIntegrityError(
                        "one expert cache key resolved to multiple stored objects"
                    )
                prior = self.load(existing[0])
                if canonical_json(prior) != canonical_json(envelope):
                    raise ExpertEvidenceConflict(
                        "expert returned different original output for an existing cache key"
                    )
                return existing[0]

            relative_path = f"{self.PREFIX}/{digest[:2]}/{digest}.json"
            written_path, written_digest = self.artifact_store.write_bytes(
                relative_path, encoded
            )
            if written_digest != digest:
                raise ExpertEvidenceIntegrityError(
                    "artifact store returned a different digest"
                )
            record = self._record_from(
                envelope,
                relative_path=written_path,
                digest=digest,
                byte_size=len(encoded),
            )
            self._index_record(record)
            return record

    def get(self, evidence_id: str) -> StoredExpertEvidence:
        self._ensure_index()
        try:
            return self._records_by_id[evidence_id]
        except KeyError:
            raise KeyError(evidence_id) from None

    def load(
        self, record: StoredExpertEvidence | str
    ) -> ExpertEvidenceEnvelope:
        resolved = self.get(record) if isinstance(record, str) else record
        raw = self.artifact_store.read_bytes(
            resolved.artifact.relative_path,
            expected_sha256=resolved.artifact.sha256,
        )
        if len(raw) != resolved.artifact.byte_size:
            raise ExpertEvidenceIntegrityError("expert evidence byte size changed")
        envelope = ExpertEvidenceEnvelope.model_validate_json(raw, strict=True)
        expected = self._record_from(
            envelope,
            relative_path=resolved.artifact.relative_path,
            digest=bytes_hash(raw),
            byte_size=len(raw),
        )
        if expected != resolved:
            raise ExpertEvidenceIntegrityError(
                "expert evidence envelope does not match its lookup record"
            )
        return envelope

    def query(
        self,
        *,
        candidate: Candidate | CandidateRef | str | None = None,
        candidate_ref: CandidateRef | None = None,
        evaluator_id: str | None = None,
        cache_key: str | None = None,
    ) -> list[StoredExpertEvidence]:
        if candidate is not None and candidate_ref is not None:
            raise ValueError("provide candidate or candidate_ref, not both")
        resolved_candidate_ref: CandidateRef | None = candidate_ref
        candidate_id: str | None = None
        if isinstance(candidate, Candidate):
            resolved_candidate_ref = candidate.candidate_ref
            if resolved_candidate_ref is None:
                raise ValueError("candidate lookup requires candidate_ref")
        elif isinstance(candidate, CandidateRef):
            resolved_candidate_ref = candidate
        elif isinstance(candidate, str):
            candidate_id = candidate
        elif candidate is not None:
            raise TypeError("candidate lookup must be Candidate, CandidateRef, or id")
        if cache_key is not None and not _SHA256.fullmatch(cache_key):
            raise ValueError("cache_key must be a lowercase SHA-256")

        self._ensure_index()
        candidate_ref_key = (
            stable_hash(resolved_candidate_ref)
            if resolved_candidate_ref is not None
            else None
        )
        indexed_sets: list[set[str]] = []
        if candidate_ref_key is not None:
            indexed_sets.append(
                self._record_ids_by_candidate_ref.get(candidate_ref_key, set())
            )
        if candidate_id is not None:
            indexed_sets.append(
                self._record_ids_by_candidate_id.get(candidate_id, set())
            )
        if evaluator_id is not None:
            indexed_sets.append(
                self._record_ids_by_evaluator.get(evaluator_id, set())
            )
        if cache_key is not None:
            cache_record = self._records_by_cache_key.get(cache_key)
            indexed_sets.append(
                set() if cache_record is None else {cache_record.evidence_id}
            )
        if indexed_sets:
            candidate_ids = set.intersection(*indexed_sets)
        else:
            candidate_ids = set(self._records_by_id)

        output = []
        for evidence_id in candidate_ids:
            item = self._records_by_id[evidence_id]
            if (
                resolved_candidate_ref is not None
                and item.candidate_ref != resolved_candidate_ref
            ):
                continue
            if candidate_id is not None and item.candidate_ref.candidate_id != candidate_id:
                continue
            if evaluator_id is not None and item.evaluator_id != evaluator_id:
                continue
            if cache_key is not None and item.cache_key != cache_key:
                continue
            output.append(item)
        return sorted(output, key=lambda item: item.evidence_id)

    def by_candidate(
        self, candidate: Candidate | CandidateRef | str
    ) -> list[StoredExpertEvidence]:
        return self.query(candidate=candidate)

    def by_evaluator(self, evaluator_id: str) -> list[StoredExpertEvidence]:
        return self.query(evaluator_id=evaluator_id)

    def by_cache_key(self, cache_key: str) -> list[StoredExpertEvidence]:
        return self.query(cache_key=cache_key)

    get_by_candidate = by_candidate
    get_by_evaluator = by_evaluator
    get_by_cache_key = by_cache_key

    def _verify_source_payload(
        self,
        payload: ExpertFeaturePayload,
        ref: ExpertFeatureRef,
    ) -> None:
        try:
            raw = self.artifact_store.read_json(
                ref.artifact.relative_path,
                expected_sha256=ref.artifact.sha256,
            )
        except (OSError, ValueError) as exc:
            raise ExpertEvidenceIntegrityError(
                "original expert payload artifact is missing or corrupt"
            ) from exc
        source = ExpertFeaturePayload.model_validate(raw)
        if source != payload:
            raise ExpertEvidenceIntegrityError(
                "original expert payload does not match its artifact reference"
            )

    def _records(self) -> list[StoredExpertEvidence]:
        self._ensure_index()
        return [self._records_by_id[key] for key in sorted(self._records_by_id)]

    def _ensure_index(self) -> None:
        """Validate immutable objects once and build process-local lookup indexes.

        A restarted process still reconstructs its indexes from the
        content-addressed envelopes.  Within one search process, subsequent
        ``get``/``query`` calls are O(1) or proportional to the matching set
        instead of recursively re-reading every evidence object.
        """

        if self._index_loaded:
            return
        with self._index_lock:
            if self._index_loaded:
                return
            self._records_by_id.clear()
            self._records_by_cache_key.clear()
            self._record_ids_by_candidate_ref.clear()
            self._record_ids_by_candidate_id.clear()
            self._record_ids_by_evaluator.clear()
            object_root = self.artifact_store.resolve(self.PREFIX)
            if not object_root.exists():
                self._index_loaded = True
                return
            try:
                for path in sorted(object_root.rglob("*.json")):
                    try:
                        relative_path = path.relative_to(
                            self.artifact_store.root
                        ).as_posix()
                    except ValueError as exc:
                        raise ExpertEvidenceIntegrityError(
                            "expert evidence index escaped the artifact root"
                        ) from exc
                    raw = self.artifact_store.read_bytes(relative_path)
                    digest = bytes_hash(raw)
                    if path.stem != digest:
                        raise ExpertEvidenceIntegrityError(
                            "expert evidence object path is not content-addressed: "
                            f"{relative_path}"
                        )
                    try:
                        envelope = ExpertEvidenceEnvelope.model_validate_json(
                            raw, strict=True
                        )
                    except Exception as exc:
                        raise ExpertEvidenceIntegrityError(
                            f"invalid expert evidence envelope: {relative_path}"
                        ) from exc
                    self._index_record(
                        self._record_from(
                            envelope,
                            relative_path=relative_path,
                            digest=digest,
                            byte_size=len(raw),
                        )
                    )
            except Exception:
                self._records_by_id.clear()
                self._records_by_cache_key.clear()
                self._record_ids_by_candidate_ref.clear()
                self._record_ids_by_candidate_id.clear()
                self._record_ids_by_evaluator.clear()
                raise
            self._index_loaded = True

    def _index_record(self, record: StoredExpertEvidence) -> None:
        prior_id = self._records_by_id.get(record.evidence_id)
        if prior_id is not None:
            if prior_id != record:
                raise ExpertEvidenceIntegrityError("duplicate expert evidence ids")
            return
        prior_cache = self._records_by_cache_key.get(record.cache_key)
        if prior_cache is not None and prior_cache.evidence_id != record.evidence_id:
            raise ExpertEvidenceIntegrityError(
                "one expert cache key resolved to multiple stored objects"
            )
        self._records_by_id[record.evidence_id] = record
        self._records_by_cache_key[record.cache_key] = record
        candidate_ref_key = stable_hash(record.candidate_ref)
        self._record_ids_by_candidate_ref.setdefault(candidate_ref_key, set()).add(
            record.evidence_id
        )
        self._record_ids_by_candidate_id.setdefault(
            record.candidate_ref.candidate_id, set()
        ).add(record.evidence_id)
        self._record_ids_by_evaluator.setdefault(record.evaluator_id, set()).add(
            record.evidence_id
        )

    @staticmethod
    def _record_from(
        envelope: ExpertEvidenceEnvelope,
        *,
        relative_path: str,
        digest: str,
        byte_size: int,
    ) -> StoredExpertEvidence:
        evidence_id = f"XEVID-{digest}"
        return StoredExpertEvidence(
            evidence_id=evidence_id,
            cache_key=envelope.cache_key,
            candidate_ref=envelope.payload.candidate_ref,
            goal_hash=envelope.feature_ref.goal_hash,
            workspace_entity_id=envelope.payload.workspace_entity_id,
            evaluator_id=envelope.payload.expert_id,
            feature_id=envelope.feature_ref.feature_id,
            artifact=ContentArtifactRef(
                artifact_id=evidence_id,
                relative_path=relative_path,
                sha256=digest,
                media_type=ExpertEvidenceStore.MEDIA_TYPE,
                byte_size=byte_size,
            ),
        )


class CandidatePoolEntry(StrictSchema):
    candidate: Candidate
    evidence_ids: list[Identifier] = Field(min_length=1)

    @model_validator(mode="after")
    def _candidate_and_evidence_are_unique(self) -> CandidatePoolEntry:
        if self.candidate.candidate_ref is None:
            raise ValueError("candidate pool entries require candidate_ref")
        if candidate_content_hash(self.candidate) != self.candidate.candidate_ref.content_hash:
            raise ValueError("candidate pool entry has a stale candidate_ref")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence ids are not allowed in a pool entry")
        return self


class CandidatePool(StrictSchema):
    pool_id: Identifier = "fusion-exploration-pool"
    entries: list[CandidatePoolEntry] = Field(min_length=1)
    required_evaluator_ids: list[Identifier] = Field(default_factory=list)

    @model_validator(mode="after")
    def _candidate_refs_are_unique(self) -> CandidatePool:
        keys = [
            stable_hash(item.candidate.candidate_ref)
            for item in self.entries
        ]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate candidate refs are not allowed in a pool")
        if len(self.required_evaluator_ids) != len(set(self.required_evaluator_ids)):
            raise ValueError("duplicate required evaluator ids are not allowed")
        return self


class SelectedCandidate(StrictSchema):
    candidate_ref: CandidateRef
    score: float | None
    score_status: Literal["available", "unknown"] = "available"
    evidence_reasons: list[NonEmptyText] = Field(default_factory=list)
    rationale: NonEmptyText
    expert_property_vectors: dict[str, list[DiagnosticProperty]]

    @model_validator(mode="after")
    def _score_availability_is_explicit(self) -> SelectedCandidate:
        if self.score_status == "unknown":
            if self.score is not None:
                raise ValueError("unknown branch scores must remain null")
            if not self.evidence_reasons:
                raise ValueError("unknown branch scores require explicit evidence reasons")
        elif self.score is None:
            raise ValueError("available branch scores require a numeric value")
        return self


class ExplorationBranchResult(StrictSchema):
    branch: ExplorationBranch
    rationale: NonEmptyText = "deterministic evidence-preserving branch ranking"
    candidates: list[SelectedCandidate] = Field(default_factory=list)


class ExcludedCandidate(StrictSchema):
    candidate_ref: CandidateRef
    reasons: list[NonEmptyText] = Field(min_length=1)


class ExplorationSelection(StrictSchema):
    pool_id: Identifier
    property_dimensions: list[str]
    branches: list[ExplorationBranchResult]
    excluded_candidates: list[ExcludedCandidate] = Field(default_factory=list)
    scientific_claim: Literal["diagnostic_only"] = "diagnostic_only"


@dataclass(frozen=True)
class _CandidateVector:
    entry: CandidatePoolEntry
    utilities: dict[tuple[str, str], float]
    original: dict[str, list[DiagnosticProperty]]
    composition_scope: str | None


@dataclass(frozen=True)
class _DisagreementAssessment:
    status: Literal["available", "unknown"]
    score: float | None
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class _NumericConstraint:
    constraint_id: str
    property_name: str
    operator: str
    values: tuple[float, ...]


class DeterministicExplorationSelector:
    """Select independent exploration branches from original expert vectors."""

    def __init__(
        self,
        evidence_store: ExpertEvidenceStore,
        *,
        disagreement_threshold: float = 0.25,
    ) -> None:
        if not 0.0 <= disagreement_threshold <= 1.0:
            raise ValueError("disagreement_threshold must be between zero and one")
        self.evidence_store = evidence_store
        self.disagreement_threshold = disagreement_threshold

    def select(
        self,
        pool: CandidatePool,
        goal: DiscoveryGoal,
        *,
        limit_per_branch: int = 4,
    ) -> ExplorationSelection:
        if isinstance(limit_per_branch, bool) or limit_per_branch <= 0:
            raise ValueError("limit_per_branch must be a positive integer")
        objective_by_name = {item.property_name: item for item in goal.objectives}
        if len(objective_by_name) != len(goal.objectives):
            raise ExplorationError("goal contains duplicate property objectives")
        numeric_constraints = _numeric_constraints(goal.constraints)

        provisional: list[_CandidateVector] = []
        excluded: dict[str, tuple[CandidateRef, list[str]]] = {}
        for entry in pool.entries:
            vector, reasons = self._build_vector(
                entry,
                objective_by_name,
                numeric_constraints=numeric_constraints,
                required_evaluators=set(pool.required_evaluator_ids),
            )
            if reasons:
                ref = entry.candidate.candidate_ref
                assert ref is not None
                excluded[stable_hash(ref)] = (ref, sorted(set(reasons)))
            elif vector is not None:
                provisional.append(vector)

        expected_dimensions = sorted(
            {dimension for item in provisional for dimension in item.utilities}
        )
        if not expected_dimensions:
            for item in provisional:
                ref = item.entry.candidate.candidate_ref
                assert ref is not None
                excluded[stable_hash(ref)] = (
                    ref,
                    ["no comparable goal property was returned by an expert"],
                )
            provisional = []

        comparable: list[_CandidateVector] = []
        for item in provisional:
            missing = [dim for dim in expected_dimensions if dim not in item.utilities]
            if missing:
                ref = item.entry.candidate.candidate_ref
                assert ref is not None
                excluded[stable_hash(ref)] = (
                    ref,
                    [
                        "missing evaluator/property dimensions: "
                        + ", ".join(_dimension_name(dim) for dim in missing)
                    ],
                )
            else:
                comparable.append(item)

        unit_error = self._unit_panel_error(comparable, objective_by_name)
        if unit_error:
            for item in comparable:
                ref = item.entry.candidate.candidate_ref
                assert ref is not None
                excluded[stable_hash(ref)] = (ref, [unit_error])
            comparable = []

        composition_scoped_dimensions = {
            dimension
            for dimension in expected_dimensions
            if _is_raw_energy_property(dimension[1])
            and any(item.composition_scope is not None for item in comparable)
        }
        normalized = self._normalize(
            comparable,
            expected_dimensions,
            composition_scoped_dimensions=composition_scoped_dimensions,
        )
        pareto = [
            item
            for item in comparable
            if not any(
                _dominates_candidate(
                    other,
                    item,
                    expected_dimensions,
                    composition_scoped_dimensions=composition_scoped_dimensions,
                )
                for other in comparable
                if other is not item
            )
        ]
        novelty = self._novelty_scores(comparable, normalized, expected_dimensions)
        disagreement = self._disagreement_scores(
            comparable,
            normalized,
            expected_dimensions,
            composition_scoped_dimensions=composition_scoped_dimensions,
        )
        target = self._weighted_objective_scores(
            comparable,
            normalized,
            expected_dimensions,
            objective_by_name,
        )
        stability_objectives = {
            name: objective
            for name, objective in objective_by_name.items()
            if _is_stability_property(name)
        }
        stability = self._weighted_objective_scores(
            comparable,
            normalized,
            expected_dimensions,
            stability_objectives,
        )
        has_raw_energy_objective = any(
            _is_raw_energy_property(name) for name in objective_by_name
        )
        stability_rationale = (
            "thermodynamic-stability utility uses only explicit formation-energy, "
            "energy-above-hull, or decomposition-energy evidence; raw total or "
            "per-atom energy is excluded"
            if stability_objectives
            else (
                "unpopulated because no formation-energy, energy-above-hull, or "
                "decomposition-energy evidence is available; raw energy supports "
                "only within-composition low-energy triage, not thermodynamic stability"
            )
        )
        target_rationale = (
            "goal-weighted utility using each objective's worst expert value"
            + (
                "; raw energy supports only within-composition low-energy triage, "
                "not thermodynamic stability"
                if has_raw_energy_objective
                else ""
            )
        )
        disagreement_rationale = (
            "available cross-expert property spans are ranked independently; at most "
            "one candidate with unknown disagreement evidence is retained as a null-score "
            "fallback with explicit reasons"
        )
        disagreement_candidates = self._rank_disagreement(
            comparable,
            disagreement,
            limit=limit_per_branch,
            rationale=disagreement_rationale,
        )
        if stability_objectives:
            stability_candidates = self._rank(
                comparable,
                stability,
                limit=limit_per_branch,
                rationale=stability_rationale,
            )
        else:
            stability_seed_rows = self._rank(
                pareto or comparable,
                target,
                limit=1,
                rationale=stability_rationale,
            )
            stability_candidates = [
                SelectedCandidate(
                    candidate_ref=row.candidate_ref,
                    score=None,
                    score_status="unknown",
                    evidence_reasons=[
                        "generation_prior_only_no_stability_evidence"
                    ],
                    rationale=(
                        stability_rationale
                        + "; deterministic Pareto parent retained only to keep the "
                        "stability generation prior reachable"
                    ),
                    expert_property_vectors=row.expert_property_vectors,
                )
                for row in stability_seed_rows
            ]

        branch_rows = [
            ExplorationBranchResult(
                branch=ExplorationBranch.STABILITY,
                rationale=stability_rationale,
                candidates=stability_candidates,
            ),
            ExplorationBranchResult(
                branch=ExplorationBranch.TARGET_PROPERTY,
                rationale=target_rationale,
                candidates=self._rank(
                    comparable,
                    target,
                    limit=limit_per_branch,
                    rationale=target_rationale,
                ),
            ),
            ExplorationBranchResult(
                branch=ExplorationBranch.NOVELTY,
                rationale=(
                    "property-space diversity from nearest-neighbor Chebyshev distance; "
                    "this is not structural or database novelty"
                ),
                candidates=self._rank(
                    comparable,
                    novelty,
                    limit=limit_per_branch,
                    rationale=(
                        "property-space diversity from nearest-neighbor Chebyshev distance; "
                        "this is not structural or database novelty"
                    ),
                ),
            ),
            ExplorationBranchResult(
                branch=ExplorationBranch.EXPERT_DISAGREEMENT,
                rationale=disagreement_rationale,
                candidates=disagreement_candidates,
            ),
            ExplorationBranchResult(
                branch=ExplorationBranch.PARETO,
                rationale=(
                    "non-dominated original per-expert objective vector; raw energies "
                    "are compared only within one reduced composition and support "
                    "only within-composition low-energy triage, not "
                    "thermodynamic stability"
                ),
                candidates=self._rank(
                    pareto,
                    {key: value for key, value in target.items()},
                    limit=limit_per_branch,
                    rationale=(
                        "non-dominated original per-expert objective vector; raw energies "
                        "are compared only within one reduced composition and support "
                        "only within-composition low-energy triage, not "
                        "thermodynamic stability"
                    ),
                ),
            ),
        ]
        excluded_rows = [
            ExcludedCandidate(candidate_ref=ref, reasons=reasons)
            for _, (ref, reasons) in sorted(excluded.items())
        ]
        return ExplorationSelection(
            pool_id=pool.pool_id,
            property_dimensions=[_dimension_name(item) for item in expected_dimensions],
            branches=branch_rows,
            excluded_candidates=excluded_rows,
        )

    def _build_vector(
        self,
        entry: CandidatePoolEntry,
        objectives: dict[str, PropertyObjective],
        *,
        numeric_constraints: tuple[_NumericConstraint, ...],
        required_evaluators: set[str],
    ) -> tuple[_CandidateVector | None, list[str]]:
        candidate_ref = entry.candidate.candidate_ref
        assert candidate_ref is not None
        reasons: list[str] = []
        utilities: dict[tuple[str, str], float] = {}
        originals: dict[str, list[DiagnosticProperty]] = {}
        for evidence_id in sorted(entry.evidence_ids):
            try:
                record = self.evidence_store.get(evidence_id)
                envelope = self.evidence_store.load(record)
            except (KeyError, ExplorationError, OSError, ValueError) as exc:
                reasons.append(
                    f"evidence {evidence_id!r} is unavailable or corrupt: {type(exc).__name__}"
                )
                continue
            payload = envelope.payload
            if record.candidate_ref != candidate_ref:
                reasons.append(f"evidence {evidence_id!r} belongs to another candidate")
                continue
            if payload.status != FeatureStatus.SUCCESS:
                reasons.append(
                    f"evaluator {payload.expert_id!r} status is {payload.status!s}, not success"
                )
                continue
            if any(item.out_of_domain for item in payload.properties):
                reasons.append(f"evaluator {payload.expert_id!r} returned out-of-domain data")
                continue
            expert_rows = originals.setdefault(payload.expert_id, [])
            existing_names = {item.property_name for item in expert_rows}
            for prop in payload.properties:
                if prop.property_name in existing_names:
                    reasons.append(
                        f"evaluator {payload.expert_id!r} returned duplicate property "
                        f"{prop.property_name!r}"
                    )
                    continue
                existing_names.add(prop.property_name)
                expert_rows.append(prop)
                objective = objectives.get(prop.property_name)
                if objective is None:
                    continue
                try:
                    utilities[(payload.expert_id, prop.property_name)] = _utility(
                        prop.value, objective
                    )
                except ExplorationError as exc:
                    reasons.append(str(exc))
        for rows in originals.values():
            rows.sort(key=lambda item: item.property_name)
        missing_evaluators = sorted(required_evaluators - set(originals))
        if missing_evaluators:
            reasons.append(
                "missing required evaluators: " + ", ".join(missing_evaluators)
            )
        returned_properties = {
            item.property_name
            for rows in originals.values()
            for item in rows
        }
        missing_required = sorted(
            item.property_name
            for item in objectives.values()
            if item.required and item.property_name not in returned_properties
        )
        if missing_required:
            reasons.append(
                "missing required goal objectives: " + ", ".join(missing_required)
            )
        reasons.extend(
            _numeric_constraint_reasons(originals, numeric_constraints)
        )
        composition_scope = _candidate_composition_scope(entry.candidate)
        if any(_is_raw_energy_property(item[1]) for item in utilities) and (
            _is_periodic_candidate(entry.candidate) and composition_scope is None
        ):
            reasons.append(
                "raw material energy requires an explicit reduced-composition scope"
            )
        if reasons:
            return None, reasons
        return _CandidateVector(
            entry=entry,
            utilities=utilities,
            original=originals,
            composition_scope=composition_scope,
        ), []

    @staticmethod
    def _weighted_objective_scores(
        vectors: list[_CandidateVector],
        normalized: dict[str, dict[tuple[str, str], float]],
        dimensions: list[tuple[str, str]],
        objectives: dict[str, PropertyObjective],
    ) -> dict[str, float]:
        if not objectives:
            return {}
        dimensions_by_property: dict[str, list[tuple[str, str]]] = {}
        for dimension in dimensions:
            if dimension[1] in objectives:
                dimensions_by_property.setdefault(dimension[1], []).append(dimension)
        if not dimensions_by_property:
            return {}
        output: dict[str, float] = {}
        for vector in vectors:
            key = _candidate_key(vector)
            weighted_total = 0.0
            total_weight = 0.0
            for property_name in sorted(dimensions_by_property):
                objective = objectives[property_name]
                weight = float(objective.weight)
                if weight <= 0.0:
                    continue
                worst_expert_utility = min(
                    normalized[key][dimension]
                    for dimension in dimensions_by_property[property_name]
                )
                weighted_total += weight * worst_expert_utility
                total_weight += weight
            output[key] = (
                weighted_total / total_weight if total_weight > 0.0 else 0.0
            )
        return output

    @staticmethod
    def _unit_panel_error(
        vectors: list[_CandidateVector],
        objectives: dict[str, PropertyObjective],
    ) -> str | None:
        units: dict[str, set[str | None]] = {}
        for vector in vectors:
            for rows in vector.original.values():
                for row in rows:
                    if row.property_name in objectives:
                        units.setdefault(row.property_name, set()).add(row.unit)
        for name, values in sorted(units.items()):
            expected = objectives[name].unit
            if len(values) != 1 or (expected is not None and values != {expected}):
                return (
                    f"incompatible units for property {name}: "
                    f"{sorted(str(item) for item in values)}"
                )
        return None

    @staticmethod
    def _normalize(
        vectors: list[_CandidateVector],
        dimensions: list[tuple[str, str]],
        *,
        composition_scoped_dimensions: set[tuple[str, str]] | None = None,
    ) -> dict[str, dict[tuple[str, str], float]]:
        scoped = composition_scoped_dimensions or set()
        ranges: dict[tuple[tuple[str, str], str | None], tuple[float, float]] = {}
        for dim in dimensions:
            if dim in scoped:
                composition_keys = sorted(
                    {item.composition_scope for item in vectors},
                    key=lambda item: "" if item is None else item,
                )
                for composition in composition_keys:
                    values = [
                        item.utilities[dim]
                        for item in vectors
                        if item.composition_scope == composition
                    ]
                    ranges[(dim, composition)] = (min(values), max(values))
            else:
                values = [item.utilities[dim] for item in vectors]
                ranges[(dim, None)] = (min(values), max(values))
        result: dict[str, dict[tuple[str, str], float]] = {}
        for item in vectors:
            row: dict[tuple[str, str], float] = {}
            for dim in dimensions:
                range_key = (
                    (dim, item.composition_scope) if dim in scoped else (dim, None)
                )
                low, high = ranges[range_key]
                row[dim] = 1.0 if high == low else (item.utilities[dim] - low) / (high - low)
            result[_candidate_key(item)] = row
        return result

    @staticmethod
    def _novelty_scores(
        vectors: list[_CandidateVector],
        normalized: dict[str, dict[tuple[str, str], float]],
        dimensions: list[tuple[str, str]],
    ) -> dict[str, float]:
        if len(vectors) == 1:
            return {_candidate_key(vectors[0]): 1.0}
        result: dict[str, float] = {}
        for item in vectors:
            key = _candidate_key(item)
            distances = []
            for other in vectors:
                other_key = _candidate_key(other)
                if key == other_key:
                    continue
                distances.append(
                    max(
                        abs(normalized[key][dim] - normalized[other_key][dim])
                        for dim in dimensions
                    )
                )
            result[key] = min(distances)
        return result

    @staticmethod
    def _disagreement_scores(
        vectors: list[_CandidateVector],
        normalized: dict[str, dict[tuple[str, str], float]],
        dimensions: list[tuple[str, str]],
        *,
        composition_scoped_dimensions: set[tuple[str, str]],
    ) -> dict[str, _DisagreementAssessment]:
        """Return typed diagnostic spans, never a calibrated error bar.

        Using the normalized panel avoids interpreting arbitrary absolute energy
        gauges from two MLIPs as a quantitative uncertainty.  A separate
        reliability calibration is still required before making an error-bound
        claim.  Incomparable panels remain null/unknown; in particular, a raw
        energy panel containing one candidate cannot manufacture numeric zero
        disagreement because it has no within-composition ordering information.
        """

        global_values: dict[str, list[float]] = {}
        by_candidate: dict[str, dict[str, list[tuple[tuple[str, str], float]]]] = {}
        composition_panel_sizes: dict[str, int] = {}
        for item in vectors:
            key = _candidate_key(item)
            if item.composition_scope is not None:
                composition_panel_sizes[item.composition_scope] = (
                    composition_panel_sizes.get(item.composition_scope, 0) + 1
                )
            grouped: dict[str, list[tuple[tuple[str, str], float]]] = {}
            for dimension in dimensions:
                property_name = dimension[1]
                raw_value = item.utilities[dimension]
                grouped.setdefault(property_name, []).append(
                    (dimension, raw_value)
                )
                global_values.setdefault(property_name, []).append(raw_value)
            by_candidate[key] = grouped

        output: dict[str, _DisagreementAssessment] = {}
        vectors_by_key = {_candidate_key(item): item for item in vectors}
        for key, groups in by_candidate.items():
            item = vectors_by_key[key]
            spans: list[float] = []
            reasons: list[str] = []
            for property_name, rows in groups.items():
                if len(rows) < 2:
                    reasons.append(
                        f"property {property_name!r} disagreement is unknown: "
                        "fewer than two successful expert values"
                    )
                    continue
                if any(
                    dimension in composition_scoped_dimensions
                    for dimension, _value in rows
                ):
                    composition = item.composition_scope
                    if composition is None:
                        reasons.append(
                            f"raw energy property {property_name!r} disagreement is "
                            "unknown: reduced-composition scope is missing"
                        )
                        continue
                    if composition_panel_sizes.get(composition, 0) < 2:
                        reasons.append(
                            f"raw energy property {property_name!r} disagreement is "
                            f"unknown for reduced composition {composition!r}: singleton "
                            "composition panel has no within-composition ordering"
                        )
                        continue
                    values = [normalized[key][dimension] for dimension, _value in rows]
                    spans.append(max(values) - min(values))
                    continue
                values = [value for _dimension, value in rows]
                population = global_values[property_name]
                scale = max(population) - min(population)
                spans.append(
                    0.0 if scale == 0.0 else (max(values) - min(values)) / scale
                )
            if spans:
                output[key] = _DisagreementAssessment(
                    status="available",
                    score=max(spans),
                    reasons=tuple(sorted(set(reasons))),
                )
            else:
                output[key] = _DisagreementAssessment(
                    status="unknown",
                    score=None,
                    reasons=tuple(
                        sorted(
                            set(reasons)
                            or {
                                "cross-expert disagreement is unknown: no property "
                                "has two comparable expert values"
                            }
                        )
                    ),
                )
        return output

    def _rank_disagreement(
        self,
        vectors: list[_CandidateVector],
        assessments: dict[str, _DisagreementAssessment],
        *,
        limit: int,
        rationale: str,
    ) -> list[SelectedCandidate]:
        """Rank available spans first and retain at most one unknown fallback."""

        unknown = [
            item
            for item in vectors
            if assessments[_candidate_key(item)].status == "unknown"
        ]
        unknown.sort(
            key=lambda item: (
                item.entry.candidate.candidate_id,
                item.entry.candidate.candidate_ref.version,
                item.entry.candidate.candidate_ref.content_hash,
            )
        )
        reserve_unknown = bool(unknown) and limit > 1
        available_limit = limit - 1 if reserve_unknown else limit
        available_scores = {
            key: assessment.score
            for key, assessment in assessments.items()
            if assessment.status == "available"
            and assessment.score is not None
            and assessment.score >= self.disagreement_threshold
        }
        available_reasons = {
            key: assessment.reasons
            for key, assessment in assessments.items()
            if assessment.status == "available" and assessment.reasons
        }
        selected = self._rank(
            vectors,
            available_scores,
            limit=available_limit,
            rationale=rationale,
            evidence_reasons=available_reasons,
        )
        if unknown and len(selected) < limit:
            fallback = unknown[0]
            assessment = assessments[_candidate_key(fallback)]
            selected.append(
                SelectedCandidate(
                    candidate_ref=fallback.entry.candidate.candidate_ref,
                    score=None,
                    score_status="unknown",
                    evidence_reasons=list(assessment.reasons),
                    rationale=(
                        "unknown cross-expert disagreement retained for follow-up; "
                        "null is not numeric agreement"
                    ),
                    expert_property_vectors=fallback.original,
                )
            )
        return selected

    @staticmethod
    def _rank(
        vectors: list[_CandidateVector],
        scores: dict[str, float],
        *,
        limit: int | None,
        rationale: str,
        evidence_reasons: dict[str, tuple[str, ...]] | None = None,
    ) -> list[SelectedCandidate]:
        eligible = [item for item in vectors if _candidate_key(item) in scores]
        eligible.sort(
            key=lambda item: (
                -scores[_candidate_key(item)],
                item.entry.candidate.candidate_id,
                item.entry.candidate.candidate_ref.version,
                item.entry.candidate.candidate_ref.content_hash,
            )
        )
        if limit is not None:
            eligible = eligible[:limit]
        return [
            SelectedCandidate(
                candidate_ref=item.entry.candidate.candidate_ref,
                score=scores[_candidate_key(item)],
                score_status="available",
                evidence_reasons=list(
                    (evidence_reasons or {}).get(_candidate_key(item), ())
                ),
                rationale=rationale,
                expert_property_vectors=item.original,
            )
            for item in eligible
        ]


class SchedulerObservation(StrictSchema):
    objective_improvement: float
    structural_collapse_rate: float = Field(ge=0.0, le=1.0)
    high_disagreement_candidates: list[CandidateRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _candidate_refs_are_unique(self) -> SchedulerObservation:
        keys = [stable_hash(item) for item in self.high_disagreement_candidates]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate high-disagreement candidate refs")
        return self


class SchedulerDecision(StrictSchema):
    observation: SchedulerObservation
    controls: GenerationControls
    reasons: list[NonEmptyText] = Field(min_length=1)


class AdaptiveGenerationScheduler:
    """Bounded controller for exploitation, exploration, and collapse recovery."""

    def __init__(
        self,
        initial: GenerationControls | None = None,
        *,
        trend_window: int = 2,
        improvement_epsilon: float = 1e-9,
        collapse_epsilon: float = 1e-6,
        alpha_step: float = 0.10,
        temperature_step: float = 0.20,
        mutation_step: float = 0.10,
        diversity_step: float = 0.05,
        max_history: int = 1_024,
        alpha_semantics: Literal["generic", "classifier_free_guidance"] = "generic",
    ) -> None:
        if isinstance(trend_window, bool) or trend_window < 2:
            raise ValueError("trend_window must be at least two")
        if isinstance(max_history, bool) or max_history <= 0:
            raise ValueError("max_history must be positive")
        if alpha_semantics not in {"generic", "classifier_free_guidance"}:
            raise ValueError("alpha_semantics is not supported")
        for label, value in (
            ("improvement_epsilon", improvement_epsilon),
            ("collapse_epsilon", collapse_epsilon),
            ("alpha_step", alpha_step),
            ("temperature_step", temperature_step),
            ("mutation_step", mutation_step),
            ("diversity_step", diversity_step),
        ):
            if value < 0.0:
                raise ValueError(f"{label} cannot be negative")
        self._controls = GenerationControls.model_validate_json(
            (initial or GenerationControls()).model_dump_json(), strict=True
        )
        self.trend_window = trend_window
        self.improvement_epsilon = improvement_epsilon
        self.collapse_epsilon = collapse_epsilon
        self.alpha_step = alpha_step
        self.temperature_step = temperature_step
        self.mutation_step = mutation_step
        self.diversity_step = diversity_step
        self.max_history = max_history
        self.alpha_semantics = alpha_semantics
        self._history: list[SchedulerDecision] = []

    @property
    def controls(self) -> GenerationControls:
        return GenerationControls.model_validate_json(
            self._controls.model_dump_json(), strict=True
        )

    @property
    def history(self) -> list[SchedulerDecision]:
        return [
            SchedulerDecision.model_validate_json(item.model_dump_json(), strict=True)
            for item in self._history
        ]

    def update(
        self,
        *,
        improvement: float,
        structural_collapse_rate: float,
        high_disagreement_candidates: Iterable[CandidateRef] = (),
    ) -> GenerationControls:
        observation = SchedulerObservation(
            objective_improvement=improvement,
            structural_collapse_rate=structural_collapse_rate,
            high_disagreement_candidates=list(high_disagreement_candidates),
        )
        return self.observe(observation)

    def observe(self, observation: SchedulerObservation) -> GenerationControls:
        observation = SchedulerObservation.model_validate_json(
            observation.model_dump_json(), strict=True
        )
        improvements = [
            item.observation.objective_improvement for item in self._history
        ] + [observation.objective_improvement]
        recent = improvements[-self.trend_window :]
        previous_collapse = (
            self._history[-1].observation.structural_collapse_rate
            if self._history
            else None
        )

        alpha = self._controls.alpha
        temperature = self._controls.temperature
        mutation = self._controls.mutation_strength
        diversity = self._controls.diversity_strength
        reasons: list[str] = []

        collapse_increased = (
            previous_collapse is not None
            and observation.structural_collapse_rate
            > previous_collapse + self.collapse_epsilon
        )
        sustained_improvement = (
            len(recent) == self.trend_window
            and all(value > self.improvement_epsilon for value in recent)
        )
        stagnated = (
            len(recent) == self.trend_window
            and all(abs(value) <= self.improvement_epsilon for value in recent)
        )

        if collapse_increased:
            temperature -= self.temperature_step
            mutation -= self.mutation_step
            if self.alpha_semantics == "classifier_free_guidance":
                alpha -= self.alpha_step
            reasons.append(
                "structural collapse increased; reduced temperature and mutation strength"
                + (
                    " and classifier-free guidance"
                    if self.alpha_semantics == "classifier_free_guidance"
                    else ""
                )
            )
        elif sustained_improvement:
            if self.alpha_semantics == "classifier_free_guidance":
                alpha += self.alpha_step
                reasons.append(
                    "improvement persisted; increased classifier-free guidance for "
                    "condition-focused exploitation"
                )
            else:
                alpha -= self.alpha_step
                reasons.append("improvement persisted; reduced alpha for precision")
        elif stagnated:
            temperature += self.temperature_step
            mutation += self.mutation_step
            diversity += self.diversity_step
            if self.alpha_semantics == "classifier_free_guidance":
                alpha -= self.alpha_step
            reasons.append(
                "objective stagnated; increased temperature, mutation, and diversity"
                + (
                    " while reducing classifier-free guidance to broaden sampling"
                    if self.alpha_semantics == "classifier_free_guidance"
                    else ""
                )
            )
        else:
            reasons.append("no sustained trend; retained the current exploration controls")

        if observation.high_disagreement_candidates:
            diversity += self.diversity_step
            reasons.append(
                "high-disagreement candidates retained in an independent exploration branch"
            )

        reason = "; ".join(reasons)
        self._controls = GenerationControls(
            alpha=_clamp(alpha, 0.0, 1.0),
            temperature=_clamp(temperature, 0.01, 5.0),
            mutation_strength=_clamp(mutation, 0.0, 1.0),
            diversity_strength=_clamp(diversity, 0.0, 1.0),
            schedule_step=self._controls.schedule_step + 1,
            decision_reason=reason,
        )
        self._history.append(
            SchedulerDecision(
                observation=observation,
                controls=self._controls,
                reasons=reasons,
            )
        )
        if len(self._history) > self.max_history:
            del self._history[: len(self._history) - self.max_history]
        return self.controls


GenerationControlScheduler = AdaptiveGenerationScheduler


def _candidate_key(vector: _CandidateVector) -> str:
    ref = vector.entry.candidate.candidate_ref
    assert ref is not None
    return stable_hash(ref)


_RAW_ENERGY_PROPERTIES = frozenset(
    {
        "energy",
        "energy_per_atom",
        "potential_energy",
        "total_energy",
    }
)


def _is_raw_energy_property(name: str) -> bool:
    """Return whether a property has an arbitrary method/composition energy gauge."""

    return name.strip().casefold().replace("-", "_") in _RAW_ENERGY_PROPERTIES


def _is_periodic_candidate(candidate: Candidate) -> bool:
    return candidate.candidate_type in {
        CandidateType.CRYSTAL,
        CandidateType.COMPOSITION,
        CandidateType.ALLOY,
        CandidateType.BATTERY_MATERIAL,
        CandidateType.CATALYST,
    }


def _candidate_composition_scope(candidate: Candidate) -> str | None:
    """Read an explicit reduced-composition key without inventing chemistry.

    Generator adapters should persist ``composition_key`` after parsing their
    authoritative structure.  A canonical chemical-formula representation is a
    compatible fallback for externally supplied candidates.  CIF text is not
    reparsed here because the model-neutral coordinator must remain usable
    without a crystallography dependency.
    """

    for key in ("composition_key", "reduced_formula"):
        value = candidate.attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    formulas = [
        item
        for item in candidate.representations
        if item.kind == RepresentationKind.CHEMICAL_FORMULA
    ]
    canonical = [item for item in formulas if item.canonical]
    selected = canonical[0] if len(canonical) == 1 else formulas[0] if formulas else None
    if selected is None:
        return None
    value = "".join(selected.value.split())
    return value or None


def _dimension_name(dimension: tuple[str, str]) -> str:
    return f"{dimension[0]}/{dimension[1]}"


def _dominates(
    left: dict[tuple[str, str], float],
    right: dict[tuple[str, str], float],
    dimensions: list[tuple[str, str]],
) -> bool:
    return all(left[item] >= right[item] for item in dimensions) and any(
        left[item] > right[item] for item in dimensions
    )


def _dominates_candidate(
    left: _CandidateVector,
    right: _CandidateVector,
    dimensions: list[tuple[str, str]],
    *,
    composition_scoped_dimensions: set[tuple[str, str]],
) -> bool:
    """Compare only scientifically compatible objective axes.

    Within one composition all axes are available.  Across compositions, raw
    total/per-atom energies are omitted because their absolute gauges do not
    define a thermodynamic ordering.  Reference-consistent formation energy or
    energy-above-hull objectives are deliberately not in the raw-energy set.
    """

    comparable = (
        dimensions
        if left.composition_scope == right.composition_scope
        else [
            item
            for item in dimensions
            if item not in composition_scoped_dimensions
        ]
    )
    if not comparable:
        return False
    return _dominates(left.utilities, right.utilities, comparable)


def _numeric_constraints(
    constraints: list[GoalConstraint],
) -> tuple[_NumericConstraint, ...]:
    """Compile only explicit hard numeric constraints; never infer a threshold."""

    compiled: list[_NumericConstraint] = []
    scalar_operators = {"eq", "ne", "gt", "gte", "lt", "lte"}
    for constraint in sorted(constraints, key=lambda item: item.constraint_id):
        if not constraint.hard or constraint.operator not in {
            *scalar_operators,
            "between",
        }:
            continue
        if constraint.property_name is None or not constraint.property_name.strip():
            raise ExplorationError(
                f"hard numeric constraint {constraint.constraint_id!r} has no property_name"
            )
        if constraint.operator in scalar_operators:
            value = constraint.value
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ExplorationError(
                    f"hard numeric constraint {constraint.constraint_id!r} requires one numeric value"
                )
            values = (float(value),)
        else:
            value = constraint.value
            if (
                not isinstance(value, list)
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in value
                )
            ):
                raise ExplorationError(
                    f"hard numeric constraint {constraint.constraint_id!r} requires [lower, upper]"
                )
            values = (float(value[0]), float(value[1]))
            if values[0] > values[1]:
                raise ExplorationError(
                    f"hard numeric constraint {constraint.constraint_id!r} has reversed bounds"
                )
        compiled.append(
            _NumericConstraint(
                constraint_id=constraint.constraint_id,
                property_name=constraint.property_name.strip(),
                operator=constraint.operator,
                values=values,
            )
        )
    return tuple(compiled)


def _numeric_constraint_reasons(
    originals: dict[str, list[DiagnosticProperty]],
    constraints: tuple[_NumericConstraint, ...],
) -> list[str]:
    reasons: list[str] = []
    rows_by_property: dict[str, list[tuple[str, DiagnosticProperty]]] = {}
    for expert_id, rows in originals.items():
        for row in rows:
            rows_by_property.setdefault(row.property_name, []).append((expert_id, row))
    for constraint in constraints:
        rows = rows_by_property.get(constraint.property_name, [])
        if not rows:
            reasons.append(
                f"missing property {constraint.property_name!r} required by hard numeric "
                f"constraint {constraint.constraint_id!r}"
            )
            continue
        units = {row.unit for _expert_id, row in rows}
        if len(units) != 1:
            reasons.append(
                f"incompatible units for hard numeric constraint {constraint.constraint_id!r}: "
                + ", ".join(sorted(str(item) for item in units))
            )
            continue
        for expert_id, row in sorted(rows, key=lambda item: item[0]):
            if not _numeric_constraint_satisfied(row.value, constraint):
                reasons.append(
                    f"hard numeric constraint {constraint.constraint_id!r} failed for "
                    f"evaluator {expert_id!r} property {constraint.property_name!r}"
                )
    return reasons


def _numeric_constraint_satisfied(
    value: float,
    constraint: _NumericConstraint,
) -> bool:
    threshold = constraint.values[0]
    if constraint.operator == "eq":
        return value == threshold
    if constraint.operator == "ne":
        return value != threshold
    if constraint.operator == "gt":
        return value > threshold
    if constraint.operator == "gte":
        return value >= threshold
    if constraint.operator == "lt":
        return value < threshold
    if constraint.operator == "lte":
        return value <= threshold
    if constraint.operator == "between":
        return threshold <= value <= constraint.values[1]
    raise ExplorationError(
        f"unsupported numeric constraint operator {constraint.operator!r}"
    )


def _utility(value: float, objective: PropertyObjective) -> float:
    if objective.lower_bound is not None and value < objective.lower_bound:
        raise ExplorationError(
            f"objective {objective.property_name!r} violates lower_bound"
        )
    if objective.upper_bound is not None and value > objective.upper_bound:
        raise ExplorationError(
            f"objective {objective.property_name!r} violates upper_bound"
        )
    if objective.direction == ObjectiveDirection.MAXIMIZE:
        return value
    if objective.direction == ObjectiveDirection.MINIMIZE:
        return -value
    if objective.direction == ObjectiveDirection.TARGET:
        target = objective.target_value
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise ExplorationError(
                f"target objective {objective.property_name!r} is not numeric"
            )
        return -abs(value - float(target))
    if objective.direction == ObjectiveDirection.RANGE:
        if objective.lower_bound is None or objective.upper_bound is None:
            raise ExplorationError(
                f"range objective {objective.property_name!r} has no bounds"
            )
        return 1.0
    if objective.direction == ObjectiveDirection.SATISFY:
        has_bound = (
            objective.lower_bound is not None or objective.upper_bound is not None
        )
        target = objective.target_value
        has_numeric_target = not isinstance(target, bool) and isinstance(
            target, (int, float)
        )
        if not has_bound and not has_numeric_target:
            raise ExplorationError(
                f"satisfy objective {objective.property_name!r} has no explicit numeric criterion"
            )
        if has_numeric_target and value != float(target):
            raise ExplorationError(
                f"satisfy objective {objective.property_name!r} does not equal its target"
            )
        return 1.0
    raise ExplorationError(
        f"objective {objective.property_name!r} has no numeric ordering"
    )


def _is_stability_property(name: str) -> bool:
    normalized = name.lower().replace("-", "_")
    return (
        "stability" in normalized
        or normalized in {"energy_above_hull", "formation_energy", "decomposition_energy"}
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return round(min(maximum, max(minimum, value)), 12)


__all__ = [
    "AdaptiveGenerationScheduler",
    "CandidatePool",
    "CandidatePoolEntry",
    "DeterministicExplorationSelector",
    "ExcludedCandidate",
    "ExpertEvidenceConflict",
    "ExpertEvidenceEnvelope",
    "ExpertEvidenceIntegrityError",
    "ExpertEvidenceStore",
    "ExplorationBranch",
    "ExplorationBranchResult",
    "ExplorationError",
    "ExplorationSelection",
    "GenerationControlScheduler",
    "SchedulerDecision",
    "SchedulerObservation",
    "SelectedCandidate",
    "StoredExpertEvidence",
]
