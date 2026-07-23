"""Fail-closed, staged scientific novelty assessment.

Crystal candidates are canonicalized and compared with pymatgen
``StructureMatcher`` so reordered, primitive, and supercell-equivalent inputs
are grouped scientifically.  Exact representation matching is retained only
for non-crystal modalities.  Missing credentials, optional dependencies, and
provider failures remain ``unknown`` instead of being promoted to novelty.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Callable, Protocol

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from ._compat import StrEnum
from .crystal_identity import (
    CRYSTAL_IDENTITY_CANONICALIZATION,
    CanonicalCrystalStructure,
    CrystalIdentityError,
    CrystalMatchRelation,
    PymatgenRequiredError,
    canonical_structure_hash,
    canonicalize_crystal_structure,
    classify_crystal_structure_relation,
    group_crystal_structures,
)
from .hashing import stable_hash
from .schemas import (
    Candidate,
    CandidateRef,
    CandidateRepresentation,
    CandidateType,
    Identifier,
    NonEmptyText,
    RepresentationKind,
    StrictSchema,
)


class NoveltyStage(StrEnum):
    WITHIN_BATCH = "within_batch"
    PROJECT_HISTORY = "project_history"
    EXTERNAL_DATABASE = "external_database"


class NoveltyStatus(StrEnum):
    """Scope-aware result; ``no_match`` is not a universal novelty claim."""

    MATCH = "match"
    NO_MATCH = "no_match"
    UNKNOWN = "unknown"


LIVE_MOVING_SNAPSHOT_UNPINNED = "live_moving_snapshot_unpinned"


class NoveltyMatch(StrictSchema):
    source_id: Identifier
    record_id: NonEmptyText
    match_kind: Identifier
    candidate_ref: CandidateRef | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class NoveltyStageResult(StrictSchema):
    stage: NoveltyStage
    status: NoveltyStatus
    method: Identifier
    query_count: int = Field(default=0, ge=0)
    matches: list[NoveltyMatch] = Field(default_factory=list)
    reason: str | None = Field(default=None, max_length=2_000)
    composition_match_count: int | None = Field(default=None, ge=0)
    structure_match_count: int | None = Field(default=None, ge=0)
    closest_match_id: str | None = Field(default=None, max_length=512)
    closest_distance: float | None = Field(default=None, ge=0.0)
    provider_id: Identifier | None = None
    client_version: NonEmptyText | None = None
    database_version_or_release: NonEmptyText | None = None
    retrieved_at: AwareDatetime | None = None
    query_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    matcher_policy: NonEmptyText | None = None
    matcher_settings: dict[str, JsonValue] = Field(default_factory=dict)
    provider_results: list["ExternalNoveltyOutcome"] = Field(default_factory=list)
    similarity_findings: list[NoveltyMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def _status_matches_evidence(self) -> "NoveltyStageResult":
        if self.status == NoveltyStatus.MATCH and not self.matches:
            raise ValueError("match status requires at least one match record")
        if self.status != NoveltyStatus.MATCH and self.matches:
            raise ValueError("only match status may contain match records")
        if self.status == NoveltyStatus.UNKNOWN and not self.reason:
            raise ValueError("unknown novelty status requires a reason")
        if len({(item.source_id, item.record_id) for item in self.matches}) != len(
            self.matches
        ):
            raise ValueError("duplicate novelty match records are not allowed")
        if len(
            {(item.source_id, item.record_id) for item in self.similarity_findings}
        ) != len(self.similarity_findings):
            raise ValueError("duplicate external similarity findings are not allowed")
        if (
            self.structure_match_count is not None
            and self.structure_match_count != len(self.matches)
        ):
            raise ValueError("structure_match_count must equal the preserved match records")
        if self.closest_match_id is not None and not self.matches:
            raise ValueError("closest_match_id requires a structure match")
        provenance = (
            self.provider_id,
            self.client_version,
            self.database_version_or_release,
            self.retrieved_at,
            self.query_sha256,
            self.matcher_policy,
        )
        if self.stage == NoveltyStage.EXTERNAL_DATABASE:
            if any(value is None for value in provenance) or not self.matcher_settings:
                raise ValueError(
                    "external novelty stages require complete provider, snapshot, query, "
                    "and matcher provenance"
                )
            if (
                self.status == NoveltyStatus.NO_MATCH
                and self.database_version_or_release
                == LIVE_MOVING_SNAPSHOT_UNPINNED
                and LIVE_MOVING_SNAPSHOT_UNPINNED not in (self.reason or "")
            ):
                raise ValueError(
                    "an unpinned live-snapshot no-match requires an explicit scope warning"
                )
            if len({item.provider_id for item in self.provider_results}) != len(
                self.provider_results
            ):
                raise ValueError("external provider results must have unique provider IDs")
            if self.provider_results and self.query_count != sum(
                item.query_count for item in self.provider_results
            ):
                raise ValueError(
                    "external aggregate query_count must equal provider result query counts"
                )
        elif any(value is not None for value in provenance) or self.matcher_settings:
            raise ValueError("external lookup provenance is only valid for the external stage")
        elif self.provider_results or self.similarity_findings:
            raise ValueError(
                "provider_results and similarity findings are only valid for the external stage"
            )
        return self


class ScientificNoveltyAssessment(StrictSchema):
    candidate_ref: CandidateRef
    scientific_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    within_batch: NoveltyStageResult
    project_history: NoveltyStageResult
    external_database: NoveltyStageResult
    overall_status: NoveltyStatus
    scope_note: NonEmptyText = (
        "no_match means no match in the configured stages; it is not proof of universal novelty"
    )

    @model_validator(mode="after")
    def _stages_and_overall_status_are_consistent(self) -> "ScientificNoveltyAssessment":
        expected_stages = (
            (self.within_batch, NoveltyStage.WITHIN_BATCH),
            (self.project_history, NoveltyStage.PROJECT_HISTORY),
            (self.external_database, NoveltyStage.EXTERNAL_DATABASE),
        )
        if any(result.stage != stage for result, stage in expected_stages):
            raise ValueError("novelty assessment stages are mislabelled")
        statuses = [item.status for item, _stage in expected_stages]
        expected = (
            NoveltyStatus.MATCH
            if NoveltyStatus.MATCH in statuses
            else (
                NoveltyStatus.NO_MATCH
                if all(item == NoveltyStatus.NO_MATCH for item in statuses)
                else NoveltyStatus.UNKNOWN
            )
        )
        if self.overall_status != expected:
            raise ValueError("overall novelty status does not match staged results")
        return self


class NoveltyPortfolioSelection(StrictSchema):
    """Bounded DFT portfolio receipt for one strict external no-match slot."""

    selected_candidate_refs: list[CandidateRef]
    reserved_external_no_match_ref: CandidateRef | None = None
    max_novelty_slots: int = Field(default=1, ge=0, le=1)
    policy: str = (
        "reserve-at-most-one-completed-strict-external-no-match; "
        "unknown-receives-no-novelty-credit"
    )

    @model_validator(mode="after")
    def _references_are_unique(self) -> "NoveltyPortfolioSelection":
        keys = [stable_hash(item) for item in self.selected_candidate_refs]
        if len(keys) != len(set(keys)):
            raise ValueError("novelty portfolio candidate references must be unique")
        if (
            self.reserved_external_no_match_ref is not None
            and stable_hash(self.reserved_external_no_match_ref) not in set(keys)
        ):
            raise ValueError("reserved novelty reference must be in the selected portfolio")
        return self


def reserve_external_no_match_portfolio_slot(
    *,
    base_candidate_refs: Sequence[CandidateRef],
    eligible_candidate_refs: Sequence[CandidateRef],
    assessments: Sequence[ScientificNoveltyAssessment],
    top_k: int,
    max_novelty_slots: int = 1,
) -> NoveltyPortfolioSelection:
    """Reserve at most one DFT slot for a completed strict database no-match.

    The caller supplies an already-science-gated priority order.  This function
    never makes an ineligible candidate eligible and never treats ``unknown`` as
    novelty.  At least one base/Pareto slot is preserved, so a one-slot handoff
    is never replaced solely for database absence.
    """

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if max_novelty_slots not in {0, 1}:
        raise ValueError("max_novelty_slots must be zero or one")

    def key(reference: CandidateRef) -> str:
        return stable_hash(reference)

    eligible_by_key: dict[str, CandidateRef] = {}
    for reference in eligible_candidate_refs:
        eligible_by_key.setdefault(key(reference), reference)
    assessment_by_key: dict[str, ScientificNoveltyAssessment] = {}
    for assessment in assessments:
        assessment_key = key(assessment.candidate_ref)
        if assessment_key in assessment_by_key:
            raise ValueError("duplicate scientific novelty assessment reference")
        assessment_by_key[assessment_key] = assessment

    selected: list[CandidateRef] = []
    selected_keys: set[str] = set()
    for reference in base_candidate_refs:
        reference_key = key(reference)
        if reference_key not in eligible_by_key or reference_key in selected_keys:
            continue
        selected.append(eligible_by_key[reference_key])
        selected_keys.add(reference_key)
        if len(selected) == top_k:
            break
    # A truncated or stale base ranking must not leave DFT capacity unused when
    # the caller supplied additional science-gated candidates.  Preserve the
    # eligible order and only fill; this does not award novelty credit.
    for reference_key, reference in eligible_by_key.items():
        if len(selected) == top_k:
            break
        if reference_key in selected_keys:
            continue
        selected.append(reference)
        selected_keys.add(reference_key)

    reserved: CandidateRef | None = None
    if max_novelty_slots == 1 and top_k >= 2:
        for reference_key, reference in eligible_by_key.items():
            assessment = assessment_by_key.get(reference_key)
            if (
                assessment is None
                or assessment.external_database.status != NoveltyStatus.NO_MATCH
            ):
                continue
            reserved = reference
            if reference_key not in selected_keys:
                if len(selected) < top_k:
                    selected.append(reference)
                else:
                    selected[-1] = reference
                selected_keys = {key(item) for item in selected}
            break

    return NoveltyPortfolioSelection(
        selected_candidate_refs=selected,
        reserved_external_no_match_ref=reserved,
        max_novelty_slots=max_novelty_slots,
    )


class ExternalNoveltyOutcome(StrictSchema):
    provider_id: Identifier
    client_version: NonEmptyText
    database_version_or_release: NonEmptyText
    retrieved_at: AwareDatetime
    query_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    matcher_policy: NonEmptyText
    matcher_settings: dict[str, JsonValue]
    status: NoveltyStatus
    method: Identifier
    query_count: int = Field(default=0, ge=0)
    matches: list[NoveltyMatch] = Field(default_factory=list)
    reason: str | None = Field(default=None, max_length=2_000)
    composition_match_count: int | None = Field(default=None, ge=0)
    structure_match_count: int | None = Field(default=None, ge=0)
    closest_match_id: str | None = Field(default=None, max_length=512)
    closest_distance: float | None = Field(default=None, ge=0.0)
    similarity_findings: list[NoveltyMatch] = Field(default_factory=list)

    @model_validator(mode="after")
    def _outcome_is_consistent(self) -> "ExternalNoveltyOutcome":
        # Reuse the exact stage-level invariants at the provider boundary.
        NoveltyStageResult(
            stage=NoveltyStage.EXTERNAL_DATABASE,
            status=self.status,
            method=self.method,
            query_count=self.query_count,
            matches=self.matches,
            reason=self.reason,
            composition_match_count=self.composition_match_count,
            structure_match_count=self.structure_match_count,
            closest_match_id=self.closest_match_id,
            closest_distance=self.closest_distance,
            provider_id=self.provider_id,
            client_version=self.client_version,
            database_version_or_release=self.database_version_or_release,
            retrieved_at=self.retrieved_at,
            query_sha256=self.query_sha256,
            matcher_policy=self.matcher_policy,
            matcher_settings=self.matcher_settings,
            similarity_findings=self.similarity_findings,
        )
        return self


class ExternalNoveltyLookup(Protocol):
    provider_id: str
    client_version: str
    database_version_or_release: str
    matcher_policy: str
    matcher_settings: Mapping[str, JsonValue]

    def lookup(self, candidate: Candidate) -> ExternalNoveltyOutcome: ...


class ProjectNoveltyIndex:
    """Project history with canonical crystals and matcher-compatible structures."""

    def __init__(self, candidates: Iterable[Candidate] = ()) -> None:
        self._exact_by_fingerprint: dict[str, dict[str, CandidateRef]] = {}
        self._crystal_records: list[
            tuple[CandidateRef, str, CanonicalCrystalStructure]
        ] = []
        for candidate in candidates:
            self.add(candidate)

    def __len__(self) -> int:
        return len(self._crystal_records) + sum(
            len(rows) for rows in self._exact_by_fingerprint.values()
        )

    def add(self, candidate: Candidate) -> None:
        reference = _required_candidate_ref(candidate)
        if _is_crystal_candidate(candidate):
            canonical = _canonical_candidate_crystal(candidate)
            reference_key = stable_hash(reference)
            if any(stable_hash(item[0]) == reference_key for item in self._crystal_records):
                return
            self._crystal_records.append(
                (reference, canonical.structure_hash, canonical)
            )
            return
        fingerprint = scientific_fingerprint(candidate)
        self._exact_by_fingerprint.setdefault(fingerprint, {})[
            stable_hash(reference)
        ] = reference

    def matches(self, candidate: Candidate) -> list[CandidateRef]:
        if _is_crystal_candidate(candidate):
            if not self._crystal_records:
                return []
            current = _canonical_candidate_crystal(candidate)
            grouping = group_crystal_structures(
                (current, *(item[2] for item in self._crystal_records))
            )
            matching_indices = next(
                group.member_indices
                for group in grouping.groups
                if 0 in group.member_indices
            )
            return [
                self._crystal_records[index - 1][0]
                for index in matching_indices
                if index > 0
            ]
        rows = self._exact_by_fingerprint.get(scientific_fingerprint(candidate), {})
        return [rows[key] for key in sorted(rows)]


class StagedNoveltyAssessor:
    """Assess matcher-based crystal duplicates and exact non-crystal duplicates."""

    def __init__(
        self,
        external_lookup: ExternalNoveltyLookup
        | Sequence[ExternalNoveltyLookup]
        | None = None,
    ) -> None:
        if external_lookup is None:
            lookups: tuple[ExternalNoveltyLookup, ...] = ()
        elif isinstance(external_lookup, Sequence) and not isinstance(
            external_lookup, (str, bytes, bytearray)
        ):
            lookups = tuple(external_lookup)
        else:
            lookups = (external_lookup,)
        provider_ids = [str(getattr(item, "provider_id", "")).strip() for item in lookups]
        if any(not item for item in provider_ids):
            raise ValueError("every external novelty lookup requires a provider_id")
        if len(set(provider_ids)) != len(provider_ids):
            raise ValueError("external novelty lookup provider IDs must be unique")
        self.external_lookups = lookups
        # Preserve the original public attribute for single-provider callers.
        self.external_lookup = lookups[0] if len(lookups) == 1 else None

    def assess(
        self,
        candidates: Sequence[Candidate],
        *,
        project_history: ProjectNoveltyIndex | Iterable[Candidate] | None = None,
    ) -> list[ScientificNoveltyAssessment]:
        if not candidates:
            return []
        fingerprints: dict[str, str | None] = {}
        peers_by_reference: dict[str, list[Candidate]] = {}
        internal_failures: dict[str, str] = {}
        for candidate in candidates:
            reference = _required_candidate_ref(candidate)
            peers_by_reference[stable_hash(reference)] = []

        crystal_rows = [
            (index, item)
            for index, item in enumerate(candidates)
            if _is_crystal_candidate(item)
        ]
        if crystal_rows:
            canonical_rows: list[
                tuple[int, Candidate, CanonicalCrystalStructure]
            ] = []
            for source_index, candidate in crystal_rows:
                reference_key = stable_hash(_required_candidate_ref(candidate))
                try:
                    canonical = _canonical_candidate_crystal(candidate)
                except PymatgenRequiredError:
                    fingerprints[reference_key] = None
                    internal_failures[reference_key] = (
                        "crystal_identity_dependency_not_installed"
                    )
                except CrystalIdentityError as exc:
                    fingerprints[reference_key] = None
                    internal_failures[reference_key] = (
                        f"crystal_identity_failed:{type(exc).__name__}"
                    )
                else:
                    fingerprints[reference_key] = canonical.structure_hash
                    canonical_rows.append((source_index, candidate, canonical))
            if canonical_rows:
                try:
                    grouped = group_crystal_structures(
                        tuple(item[2] for item in canonical_rows)
                    )
                except CrystalIdentityError as exc:
                    for _source_index, candidate, _canonical in canonical_rows:
                        reference_key = stable_hash(_required_candidate_ref(candidate))
                        internal_failures[reference_key] = (
                            f"crystal_structure_match_failed:{type(exc).__name__}"
                        )
                else:
                    for group in grouped.groups:
                        members = [
                            canonical_rows[index][1]
                            for index in group.member_indices
                        ]
                        for candidate in members:
                            key = stable_hash(_required_candidate_ref(candidate))
                            peers_by_reference[key] = [
                                item
                                for item in members
                                if item.candidate_ref != candidate.candidate_ref
                            ]

        exact_groups: dict[str, list[Candidate]] = {}
        for candidate in candidates:
            if _is_crystal_candidate(candidate):
                continue
            fingerprint = scientific_fingerprint(candidate)
            reference_key = stable_hash(_required_candidate_ref(candidate))
            fingerprints[reference_key] = fingerprint
            exact_groups.setdefault(fingerprint, []).append(candidate)
        for members in exact_groups.values():
            for candidate in members:
                key = stable_hash(_required_candidate_ref(candidate))
                peers_by_reference[key] = [
                    item
                    for item in members
                    if item.candidate_ref != candidate.candidate_ref
                ]

        history_failure: str | None = None
        try:
            history = (
                project_history
                if isinstance(project_history, ProjectNoveltyIndex)
                else (
                    ProjectNoveltyIndex(project_history)
                    if project_history is not None
                    else None
                )
            )
        except PymatgenRequiredError:
            history = None
            history_failure = "crystal_identity_dependency_not_installed"
        except CrystalIdentityError as exc:
            history = None
            history_failure = f"project_history_identity_failed:{type(exc).__name__}"
        assessments: list[ScientificNoveltyAssessment] = []
        for candidate in candidates:
            reference = _required_candidate_ref(candidate)
            reference_key = stable_hash(reference)
            fingerprint = fingerprints.get(reference_key)
            peers = peers_by_reference[reference_key]
            internal_method, match_kind = _internal_matching_contract(candidate)
            if reference_key in internal_failures:
                within = NoveltyStageResult(
                    stage=NoveltyStage.WITHIN_BATCH,
                    status=NoveltyStatus.UNKNOWN,
                    method=internal_method,
                    query_count=max(0, len(candidates) - 1),
                    reason=internal_failures[reference_key],
                )
            else:
                within = _internal_stage(
                    stage=NoveltyStage.WITHIN_BATCH,
                    method=internal_method,
                    match_kind=match_kind,
                    source_id="current-batch",
                    matches=[_required_candidate_ref(item) for item in peers],
                    query_count=max(0, len(candidates) - 1),
                )
            if history_failure is not None:
                project = NoveltyStageResult(
                    stage=NoveltyStage.PROJECT_HISTORY,
                    status=NoveltyStatus.UNKNOWN,
                    method=internal_method,
                    reason=history_failure,
                )
            elif history is None:
                project = NoveltyStageResult(
                    stage=NoveltyStage.PROJECT_HISTORY,
                    status=NoveltyStatus.UNKNOWN,
                    method=internal_method,
                    reason="project_history_not_provided",
                )
            else:
                try:
                    history_matches = history.matches(candidate)
                except PymatgenRequiredError:
                    project = NoveltyStageResult(
                        stage=NoveltyStage.PROJECT_HISTORY,
                        status=NoveltyStatus.UNKNOWN,
                        method=internal_method,
                        reason="crystal_identity_dependency_not_installed",
                    )
                except CrystalIdentityError as exc:
                    project = NoveltyStageResult(
                        stage=NoveltyStage.PROJECT_HISTORY,
                        status=NoveltyStatus.UNKNOWN,
                        method=internal_method,
                        reason=f"project_history_match_failed:{type(exc).__name__}",
                    )
                else:
                    project = _internal_stage(
                        stage=NoveltyStage.PROJECT_HISTORY,
                        method=internal_method,
                        match_kind=match_kind,
                        source_id="project-history",
                        matches=history_matches,
                        query_count=len(history),
                    )
            external = self._external_stage(candidate)
            statuses = [within.status, project.status, external.status]
            overall = (
                NoveltyStatus.MATCH
                if NoveltyStatus.MATCH in statuses
                else (
                    NoveltyStatus.NO_MATCH
                    if all(item == NoveltyStatus.NO_MATCH for item in statuses)
                    else NoveltyStatus.UNKNOWN
                )
            )
            assessments.append(
                ScientificNoveltyAssessment(
                    candidate_ref=reference,
                    scientific_fingerprint=fingerprint,
                    within_batch=within,
                    project_history=project,
                    external_database=external,
                    overall_status=overall,
                )
            )
        return assessments

    def _external_stage(self, candidate: Candidate) -> NoveltyStageResult:
        if not self.external_lookups:
            return NoveltyStageResult(
                stage=NoveltyStage.EXTERNAL_DATABASE,
                status=NoveltyStatus.UNKNOWN,
                method="external-structure-lookup-v1",
                reason="external_lookup_not_configured",
                provider_id="external-lookup-unconfigured",
                client_version="not-applicable",
                database_version_or_release="not-configured",
                retrieved_at=datetime.now(timezone.utc),
                query_sha256=_fallback_external_query_sha256(
                    candidate,
                    provider_id="external-lookup-unconfigured",
                ),
                matcher_policy="no-external-matcher-configured",
                matcher_settings={"configured_provider_count": 0},
            )
        outcomes: list[ExternalNoveltyOutcome] = []
        for lookup in self.external_lookups:
            try:
                outcome = lookup.lookup(candidate)
                if outcome.provider_id != lookup.provider_id:
                    raise ValueError("external lookup returned a mismatched provider_id")
            except Exception as exc:  # provider failures must never become no-match
                outcome = _failed_external_outcome(lookup, candidate, exc)
            outcomes.append(outcome)

        statuses = [item.status for item in outcomes]
        status = (
            NoveltyStatus.MATCH
            if NoveltyStatus.MATCH in statuses
            else (
                NoveltyStatus.NO_MATCH
                if all(item == NoveltyStatus.NO_MATCH for item in statuses)
                else NoveltyStatus.UNKNOWN
            )
        )
        matches_by_key: dict[tuple[str, str], NoveltyMatch] = {}
        similarities_by_key: dict[tuple[str, str], NoveltyMatch] = {}
        for outcome in outcomes:
            for item in outcome.matches:
                matches_by_key[(item.source_id, item.record_id)] = item
            for item in outcome.similarity_findings:
                similarities_by_key[(item.source_id, item.record_id)] = item
        matches = [matches_by_key[key] for key in sorted(matches_by_key)]
        similarities = [
            similarities_by_key[key] for key in sorted(similarities_by_key)
        ]
        moving_snapshot = any(
            item.database_version_or_release == LIVE_MOVING_SNAPSHOT_UNPINNED
            for item in outcomes
        )
        if status == NoveltyStatus.NO_MATCH:
            reason = "no_strict_structure_match_in_all_configured_providers"
            if moving_snapshot:
                reason += (
                    f":{LIVE_MOVING_SNAPSHOT_UNPINNED}:"
                    "not_reproducible_against_a_pinned_database_release"
                )
        elif status == NoveltyStatus.UNKNOWN:
            unresolved = ",".join(
                f"{item.provider_id}={item.status}:{item.reason or 'unspecified'}"
                for item in outcomes
                if item.status == NoveltyStatus.UNKNOWN
            )
            reason = f"one_or_more_external_providers_unresolved:{unresolved}"
        else:
            reason = None

        single = outcomes[0] if len(outcomes) == 1 else None
        provider_id = single.provider_id if single else "multi-provider-aggregate"
        client_version = single.client_version if single else "see-provider-results"
        database_release = (
            single.database_version_or_release
            if single
            else (
                LIVE_MOVING_SNAPSHOT_UNPINNED
                if moving_snapshot
                else "see-provider-results"
            )
        )
        matcher_policy = (
            single.matcher_policy
            if single
            else "all-configured-providers-required-for-no-match-v1"
        )
        matcher_settings: dict[str, JsonValue] = (
            dict(single.matcher_settings)
            if single
            else {
                "aggregation": "match-if-any;no-match-only-if-all;otherwise-unknown",
                "required_provider_ids": [item.provider_id for item in outcomes],
            }
        )
        query_sha256 = (
            single.query_sha256
            if single
            else stable_hash(
                {
                    "aggregation_policy": matcher_policy,
                    "provider_queries": [
                        {
                            "provider_id": item.provider_id,
                            "query_sha256": item.query_sha256,
                        }
                        for item in outcomes
                    ],
                }
            )
        )
        return NoveltyStageResult(
            stage=NoveltyStage.EXTERNAL_DATABASE,
            status=status,
            method=(
                single.method if single else "required-external-provider-aggregate-v1"
            ),
            query_count=sum(item.query_count for item in outcomes),
            matches=matches,
            reason=reason if reason is not None else (single.reason if single else None),
            composition_match_count=(
                sum(item.composition_match_count or 0 for item in outcomes)
                if all(item.composition_match_count is not None for item in outcomes)
                else None
            ),
            structure_match_count=len(matches),
            closest_match_id=matches[0].record_id if matches else None,
            closest_distance=None,
            provider_id=provider_id,
            client_version=client_version,
            database_version_or_release=database_release,
            retrieved_at=max(item.retrieved_at for item in outcomes),
            query_sha256=query_sha256,
            matcher_policy=matcher_policy,
            matcher_settings=matcher_settings,
            provider_results=outcomes,
            similarity_findings=similarities,
        )


class MaterialsProjectStructureLookup:
    """Optional ``mp-api`` structure matcher with an injected runtime credential.

    The official client's ``find_structure`` route uses scale-normalized,
    relatively loose matching.  Its raw IDs are therefore similarity candidates,
    never hard identity evidence.  This adapter fetches each returned structure
    and applies the local deletion-safe classifier.  A formula query is retained
    only as an audited coverage count.
    """

    provider_id = "materials-project"
    matcher_policy = "materials-project-scaled-prefilter-local-strict-recheck-v1"

    def __init__(
        self,
        api_key: str | None,
        *,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
        client_version: str | None = None,
        database_version_or_release: str = LIVE_MOVING_SNAPSHOT_UNPINNED,
        rester_factory: Callable[[str], object] | None = None,
    ) -> None:
        if ltol <= 0 or stol <= 0 or angle_tol <= 0:
            raise ValueError("structure-match tolerances must be positive")
        self._api_key = api_key.strip() if api_key else ""
        self.ltol = float(ltol)
        self.stol = float(stol)
        self.angle_tol = float(angle_tol)
        self.client_version = (
            client_version.strip()
            if client_version and client_version.strip()
            else _installed_package_version("mp-api")
        )
        release = database_version_or_release.strip()
        if not release:
            raise ValueError("database_version_or_release must not be blank")
        self.database_version_or_release = release
        self.matcher_settings: dict[str, JsonValue] = {
            "remote_scaled_prefilter": {
                "endpoint": "MPRester.find_structure",
                "ltol": self.ltol,
                "stol": self.stol,
                "angle_tol": self.angle_tol,
                "primitive_cell": True,
                "scale": True,
                "attempt_supercell": False,
                "allow_subset": False,
                "comparator": "ElementComparator",
                "allow_multiple_results": True,
                "search_scope": "same-reduced-formula-material-documents",
            },
            "local_strict_recheck": {
                "implementation": "classify_crystal_structure_relation",
                "canonicalization": CRYSTAL_IDENTITY_CANONICALIZATION,
                "ltol": 0.02,
                "stol": 0.05,
                "angle_tol": 1.0,
                "primitive_cell": True,
                "scale": False,
                "attempt_supercell": True,
                "allow_subset": False,
                "comparator": "StructureMatcher-default-species-comparator",
                "symmetric_fit": "native-symmetric-or-required-bidirectional-fallback",
                "max_relative_volume_difference": 0.03,
            },
            "local_scaled_classification": {
                "canonicalization": CRYSTAL_IDENTITY_CANONICALIZATION,
                "ltol": 0.2,
                "stol": 0.3,
                "angle_tol": 5.0,
                "primitive_cell": True,
                "scale": True,
                "attempt_supercell": True,
                "allow_subset": False,
                "comparator": "StructureMatcher-default-species-comparator",
                "symmetric_fit": "native-symmetric-or-required-bidirectional-fallback",
            },
            "hard_identity_relation": CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE.value,
            "composition_lookup_role": "coverage_count_only",
        }
        self._rester_factory = rester_factory

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        **kwargs: object,
    ) -> "MaterialsProjectStructureLookup":
        values = os.environ if environ is None else environ
        kwargs.setdefault(
            "database_version_or_release",
            values.get(
                "MP_DATABASE_VERSION_OR_RELEASE",
                LIVE_MOVING_SNAPSHOT_UNPINNED,
            ),
        )
        return cls(values.get("MP_API_KEY"), **kwargs)

    def lookup(self, candidate: Candidate) -> ExternalNoveltyOutcome:
        retrieved_at = datetime.now(timezone.utc)
        cif = _representation(candidate, RepresentationKind.CIF)
        if not self._api_key:
            return self._unknown(
                "materials_project_api_key_not_configured",
                candidate=candidate,
                cif=cif,
                query_count=0,
                retrieved_at=retrieved_at,
            )
        if cif is None:
            return self._unknown(
                "candidate_has_no_cif_representation",
                candidate=candidate,
                cif=None,
                query_count=0,
                retrieved_at=retrieved_at,
            )
        try:
            factory = self._rester_factory or _materials_project_rester_factory()
        except (ImportError, ModuleNotFoundError):
            return self._unknown(
                "materials_project_client_not_installed",
                candidate=candidate,
                cif=cif,
                query_count=0,
                retrieved_at=retrieved_at,
            )

        client: object | None = None
        database_release = self.database_version_or_release
        try:
            client = factory(self._api_key)
            database_release = (
                _materials_project_database_release(client) or database_release
            )
            find_structure = getattr(client, "find_structure")
            with tempfile.TemporaryDirectory(prefix="discovery-mp-lookup-") as root:
                path = Path(root) / "candidate.cif"
                path.write_text(cif.value, encoding="utf-8")
                raw = find_structure(
                    str(path),
                    ltol=self.ltol,
                    stol=self.stol,
                    angle_tol=self.angle_tol,
                    allow_multiple_results=True,
                )
                composition_ids = self._composition_matches(client, candidate)
                identifiers = _external_identifiers(raw)
                matches, similarities, unresolved = self._strict_recheck(
                    client,
                    cif.value,
                    identifiers,
                )
        except Exception as exc:
            return self._unknown(
                f"materials_project_lookup_failed:{type(exc).__name__}",
                candidate=candidate,
                cif=cif,
                query_count=1,
                retrieved_at=retrieved_at,
                database_version_or_release=database_release,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        if not identifiers:
            return ExternalNoveltyOutcome(
                **self._provenance(
                    candidate,
                    cif,
                    retrieved_at=retrieved_at,
                    database_version_or_release=database_release,
                ),
                status=NoveltyStatus.NO_MATCH,
                method="materials-project-find-structure-v1",
                query_count=1,
                reason=_scoped_no_match_reason(database_release),
                composition_match_count=(
                    len(composition_ids) if composition_ids is not None else None
                ),
                structure_match_count=0,
            )
        if not matches and unresolved:
            return ExternalNoveltyOutcome(
                **self._provenance(
                    candidate,
                    cif,
                    retrieved_at=retrieved_at,
                    database_version_or_release=database_release,
                ),
                status=NoveltyStatus.UNKNOWN,
                method="materials-project-find-structure-v1",
                query_count=1,
                reason=(
                    "materials_project_scaled_similarities_could_not_all_be_"
                    "strictly_rechecked"
                ),
                composition_match_count=(
                    len(composition_ids) if composition_ids is not None else None
                ),
                structure_match_count=0,
                similarity_findings=similarities,
            )
        return ExternalNoveltyOutcome(
            **self._provenance(
                candidate,
                cif,
                retrieved_at=retrieved_at,
                database_version_or_release=database_release,
            ),
            status=(NoveltyStatus.MATCH if matches else NoveltyStatus.NO_MATCH),
            method="materials-project-find-structure-v1",
            query_count=1,
            reason=(
                None
                if matches
                else (
                    _scoped_no_match_reason(database_release)
                    + ":remote_scaled_similarities_rejected_by_local_strict_policy"
                )
            ),
            composition_match_count=(
                len(composition_ids) if composition_ids is not None else None
            ),
            structure_match_count=len(matches),
            closest_match_id=matches[0].record_id if matches else None,
            closest_distance=None,
            matches=matches,
            similarity_findings=similarities,
        )

    def _unknown(
        self,
        reason: str,
        *,
        candidate: Candidate,
        cif: CandidateRepresentation | None,
        query_count: int,
        retrieved_at: datetime,
        database_version_or_release: str | None = None,
    ) -> ExternalNoveltyOutcome:
        return ExternalNoveltyOutcome(
            **self._provenance(
                candidate,
                cif,
                retrieved_at=retrieved_at,
                database_version_or_release=(
                    database_version_or_release or self.database_version_or_release
                ),
            ),
            status=NoveltyStatus.UNKNOWN,
            method="materials-project-find-structure-v1",
            query_count=query_count,
            reason=reason,
        )

    def _provenance(
        self,
        candidate: Candidate,
        cif: CandidateRepresentation | None,
        *,
        retrieved_at: datetime,
        database_version_or_release: str,
    ) -> dict[str, object]:
        query = {
            "provider_id": self.provider_id,
            "method": "materials-project-find-structure-v1",
            "candidate_ref": _required_candidate_ref(candidate),
            "cif_sha256": (
                stable_hash(_normalized_representation_value(cif)) if cif else None
            ),
            "database_version_or_release": database_version_or_release,
            "matcher_policy": self.matcher_policy,
            "matcher_settings": self.matcher_settings,
        }
        return {
            "provider_id": self.provider_id,
            "client_version": self.client_version,
            "database_version_or_release": database_version_or_release,
            "retrieved_at": retrieved_at,
            "query_sha256": stable_hash(query),
            "matcher_policy": self.matcher_policy,
            "matcher_settings": self.matcher_settings,
        }

    def _strict_recheck(
        self,
        client: object,
        candidate_cif: str,
        identifiers: Sequence[str],
    ) -> tuple[list[NoveltyMatch], list[NoveltyMatch], bool]:
        matches: list[NoveltyMatch] = []
        similarities: list[NoveltyMatch] = []
        unresolved = False
        fetch_structure = _materials_project_structure_fetcher(client)
        for material_id in identifiers:
            if fetch_structure is None:
                unresolved = True
                similarities.append(
                    NoveltyMatch(
                        source_id=self.provider_id,
                        record_id=material_id,
                        match_kind="provider-scaled-similarity-unverified",
                        metadata={
                            "strict_recheck": "structure_fetch_api_unavailable",
                            "hard_identity": "false",
                        },
                    )
                )
                continue
            try:
                remote_structure = fetch_structure(material_id)
                assessment = classify_crystal_structure_relation(
                    candidate_cif,
                    remote_structure,
                )
                metadata = {
                    "strict_match": str(assessment.strict_match).lower(),
                    "scaled_match": str(assessment.scaled_match).lower(),
                    "relative_volume_difference": str(
                        assessment.relative_volume_difference
                    ),
                    "strict_settings_sha256": stable_hash(
                        asdict(assessment.strict_settings)
                    ),
                    "scaled_settings_sha256": stable_hash(
                        asdict(assessment.scaled_settings)
                    ),
                    "reason": assessment.reason or "none",
                }
                finding = NoveltyMatch(
                    source_id=self.provider_id,
                    record_id=material_id,
                    match_kind=assessment.relation.value,
                    metadata=metadata,
                )
                if assessment.relation == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE:
                    matches.append(finding)
                else:
                    similarities.append(finding)
                if assessment.relation == CrystalMatchRelation.AMBIGUOUS:
                    unresolved = True
            except Exception as exc:
                unresolved = True
                similarities.append(
                    NoveltyMatch(
                        source_id=self.provider_id,
                        record_id=material_id,
                        match_kind="provider-scaled-similarity-unverified",
                        metadata={
                            "strict_recheck": f"failed:{type(exc).__name__}",
                            "hard_identity": "false",
                        },
                    )
                )
        return matches, similarities, unresolved

    @staticmethod
    def _composition_matches(client: object, candidate: Candidate) -> list[str] | None:
        """Best-effort composition count; structure matching stays independent.

        The remote ``find_structure`` call is intentionally performed first, so
        an absent local pymatgen extra cannot prevent an injected or official MP
        client from doing its own parsing and tolerance-aware comparison.
        """

        try:
            formula = _representation(candidate, RepresentationKind.CHEMICAL_FORMULA)
            if formula is not None:
                reduced_formula = _normalized_representation_value(formula)
            else:
                canonical = _canonical_candidate_crystal(candidate)
                reduced_formula = str(
                    canonical.canonical_structure.composition.reduced_formula
                )
            get_material_ids = getattr(client, "get_material_ids")
            return _external_identifiers(get_material_ids(reduced_formula))
        except Exception:
            return None


def scientific_fingerprint(candidate: Candidate) -> str:
    """Return canonical crystal identity or exact primary content for non-crystals."""

    if _is_crystal_candidate(candidate):
        # Deliberately no exact-text fallback: a reordered or supercell CIF must
        # retain one scientific identity, and missing pymatgen is actionable.
        cif = _representation(candidate, RepresentationKind.CIF)
        if cif is not None:
            return canonical_structure_hash(cif.value, fmt="cif")
        poscar = _representation(candidate, RepresentationKind.POSCAR)
        if poscar is None:
            raise ValueError("periodic novelty assessment requires CIF or POSCAR")
        return canonical_structure_hash(poscar.value, fmt="poscar")

    representation = _primary_representation(candidate)
    return stable_hash(
        {
            "candidate_type": candidate.candidate_type,
            "domain": candidate.domain,
            "representation_kind": representation.kind,
            "representation_value": _normalized_representation_value(representation),
        }
    )


def _is_crystal_candidate(candidate: Candidate) -> bool:
    periodic_types = {
        CandidateType.CRYSTAL,
        CandidateType.COMPOSITION,
        CandidateType.ALLOY,
        CandidateType.BATTERY_MATERIAL,
        CandidateType.CATALYST,
    }
    return candidate.candidate_type in periodic_types and any(
        item.kind in {RepresentationKind.CIF, RepresentationKind.POSCAR}
        for item in candidate.representations
    )


def _canonical_candidate_crystal(candidate: Candidate) -> CanonicalCrystalStructure:
    cif = _representation(candidate, RepresentationKind.CIF)
    if cif is not None:
        return canonicalize_crystal_structure(cif.value, fmt="cif")
    poscar = _representation(candidate, RepresentationKind.POSCAR)
    if poscar is None:
        raise ValueError("periodic novelty assessment requires CIF or POSCAR")
    return canonicalize_crystal_structure(poscar.value, fmt="poscar")


def _primary_representation(candidate: Candidate) -> CandidateRepresentation:
    priority = (
        RepresentationKind.CIF,
        RepresentationKind.POSCAR,
        RepresentationKind.SMILES,
        RepresentationKind.SDF,
        RepresentationKind.XYZ,
        RepresentationKind.EXTXYZ,
        RepresentationKind.CHEMICAL_FORMULA,
        RepresentationKind.PROTEIN_SEQUENCE,
        RepresentationKind.RNA_SEQUENCE,
        RepresentationKind.FASTA,
        RepresentationKind.CUSTOM,
    )
    for kind in priority:
        rows = [item for item in candidate.representations if item.kind == kind]
        if rows:
            canonical = [item for item in rows if item.canonical]
            return canonical[0] if len(canonical) == 1 else rows[0]
    return candidate.representations[0]


def _normalized_representation_value(representation: CandidateRepresentation) -> str:
    return representation.value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _internal_stage(
    *,
    stage: NoveltyStage,
    method: str,
    match_kind: str,
    source_id: str,
    matches: Sequence[CandidateRef],
    query_count: int,
) -> NoveltyStageResult:
    rows = [
        NoveltyMatch(
            source_id=source_id,
            record_id=f"{item.candidate_id}@{item.version}:{item.content_hash}",
            match_kind=match_kind,
            candidate_ref=item,
        )
        for item in matches
    ]
    return NoveltyStageResult(
        stage=stage,
        status=NoveltyStatus.MATCH if rows else NoveltyStatus.NO_MATCH,
        method=method,
        query_count=query_count,
        matches=rows,
    )


def _internal_matching_contract(candidate: Candidate) -> tuple[str, str]:
    if _is_crystal_candidate(candidate):
        return (
            "pymatgen-structure-matcher-v1",
            "canonical-tolerance-aware-structure-match",
        )
    return (
        "exact-scientific-representation-v1",
        "exact-scientific-representation",
    )


def _required_candidate_ref(candidate: Candidate) -> CandidateRef:
    if candidate.candidate_ref is None:
        raise ValueError("novelty assessment requires immutable candidate_ref values")
    return candidate.candidate_ref


def _representation(
    candidate: Candidate, kind: RepresentationKind
) -> CandidateRepresentation | None:
    rows = [item for item in candidate.representations if item.kind == kind]
    if not rows:
        return None
    canonical = [item for item in rows if item.canonical]
    return canonical[0] if len(canonical) == 1 else rows[0]


def _external_identifiers(value: object) -> list[str]:
    if value is None:
        return []
    rows = [value] if isinstance(value, str) else value
    if not isinstance(rows, (list, tuple, set)):
        return []
    return sorted(
        {
            str(item).strip()
            for item in rows
            if str(item).strip() and len(str(item).strip()) <= 512
        }
    )


def _failed_external_outcome(
    lookup: ExternalNoveltyLookup,
    candidate: Candidate,
    exc: Exception,
) -> ExternalNoveltyOutcome:
    """Return a redacted fail-closed result while retaining provider identity."""

    provider_id = str(getattr(lookup, "provider_id", "external-provider")).strip()
    raw_settings = getattr(lookup, "matcher_settings", None)
    matcher_settings: dict[str, JsonValue]
    if isinstance(raw_settings, Mapping) and raw_settings:
        matcher_settings = {
            str(key): _bounded_json_value(value) for key, value in raw_settings.items()
        }
    else:
        matcher_settings = {
            "provenance_status": "unavailable_due_to_provider_exception"
        }
    return ExternalNoveltyOutcome(
        provider_id=provider_id,
        client_version=(
            str(getattr(lookup, "client_version", "provider-version-unavailable"))
            or "provider-version-unavailable"
        ),
        database_version_or_release=(
            str(
                getattr(
                    lookup,
                    "database_version_or_release",
                    LIVE_MOVING_SNAPSHOT_UNPINNED,
                )
            )
            or LIVE_MOVING_SNAPSHOT_UNPINNED
        ),
        retrieved_at=datetime.now(timezone.utc),
        query_sha256=_fallback_external_query_sha256(
            candidate,
            provider_id=provider_id,
        ),
        matcher_policy=(
            str(getattr(lookup, "matcher_policy", "provider-defined-external-lookup"))
            or "provider-defined-external-lookup"
        ),
        matcher_settings=matcher_settings,
        status=NoveltyStatus.UNKNOWN,
        method="external-structure-lookup-v1",
        query_count=1,
        reason=f"external_lookup_failed:{type(exc).__name__}",
    )


def _fallback_external_query_sha256(
    candidate: Candidate,
    *,
    provider_id: str,
) -> str:
    return stable_hash(
        {
            "provider_id": provider_id,
            "candidate_ref": _required_candidate_ref(candidate),
            "query_contract": "provider-query-unavailable-v1",
        }
    )


def _bounded_json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _bounded_json_value(item)
            for key, item in list(value.items())[:100]
        }
    if isinstance(value, (list, tuple)):
        return [_bounded_json_value(item) for item in value[:100]]
    return str(value)[:1_000]


def _installed_package_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "not-installed"
    except Exception:
        return "version-unavailable"


def _materials_project_database_release(client: object) -> str | None:
    for owner in (client, getattr(client, "materials", None)):
        if owner is None:
            continue
        for name in ("db_version", "get_database_version", "database_version"):
            value = getattr(owner, name, None)
            try:
                resolved = value() if callable(value) else value
            except Exception:
                continue
            text = str(resolved).strip() if resolved is not None else ""
            if text:
                return text[:2_000]
    return None


def _materials_project_structure_fetcher(
    client: object,
) -> Callable[[str], object] | None:
    for owner in (client, getattr(client, "materials", None)):
        if owner is None:
            continue
        for name in ("get_structure_by_material_id", "get_structure"):
            method = getattr(owner, name, None)
            if not callable(method):
                continue

            def fetch(material_id: str, *, _method: Callable[..., object] = method) -> object:
                try:
                    return _method(material_id)
                except TypeError:
                    return _method(material_id=material_id)

            return fetch
    return None


def _scoped_no_match_reason(database_version_or_release: str) -> str:
    base = f"no_strict_structure_match_in_database_scope:{database_version_or_release}"
    if database_version_or_release == LIVE_MOVING_SNAPSHOT_UNPINNED:
        return (
            base
            + ":not_reproducible_against_a_pinned_database_release;"
            "absence_is_not_proof_of_universal_novelty"
        )
    return base + ":absence_is_not_proof_of_universal_novelty"


def _materials_project_rester_factory() -> Callable[[str], object]:
    from mp_api.client import MPRester

    return lambda api_key: MPRester(api_key, mute_progress_bars=True)


__all__ = [
    "ExternalNoveltyLookup",
    "ExternalNoveltyOutcome",
    "LIVE_MOVING_SNAPSHOT_UNPINNED",
    "MaterialsProjectStructureLookup",
    "NoveltyMatch",
    "NoveltyPortfolioSelection",
    "NoveltyStage",
    "NoveltyStageResult",
    "NoveltyStatus",
    "ProjectNoveltyIndex",
    "ScientificNoveltyAssessment",
    "StagedNoveltyAssessor",
    "reserve_external_no_match_portfolio_slot",
    "scientific_fingerprint",
]
