"""Fail-closed, staged scientific novelty assessment.

Crystal candidates are canonicalized and compared with pymatgen
``StructureMatcher`` so reordered, primitive, and supercell-equivalent inputs
are grouped scientifically.  Exact representation matching is retained only
for non-crystal modalities.  Missing credentials, optional dependencies, and
provider failures remain ``unknown`` instead of being promoted to novelty.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import urljoin, urlsplit

import requests
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
    inspect_crystal_occupancy,
    parse_crystal_structure,
    validate_crystal_geometry,
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
        _context, candidate_issue = _external_candidate_query_context(candidate)
        if candidate_issue is not None:
            return self._unknown(
                candidate_issue,
                candidate=candidate,
                cif=cif,
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
                validate_crystal_geometry(remote_structure)
                occupancy = inspect_crystal_occupancy(remote_structure)
                if not occupancy.is_fully_occupied_ordered:
                    unresolved = True
                    similarities.append(
                        NoveltyMatch(
                            source_id=self.provider_id,
                            record_id=material_id,
                            match_kind="provider-scaled-similarity-unverified",
                            metadata={
                                "strict_recheck": (
                                    "remote_disorder_or_partial_occupancy_unsupported:"
                                    + ",".join(occupancy.reason_codes)
                                ),
                                "hard_identity": "false",
                            },
                        )
                    )
                    continue
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


@dataclass(frozen=True, slots=True)
class _ExternalCandidateQueryContext:
    representation: CandidateRepresentation
    fmt: str
    structure: Any
    structure_sha256: str
    optimade_reduced_formula: str
    cod_hill_formula: str


@dataclass(frozen=True, slots=True)
class _ProviderRetrieval:
    records: tuple[Mapping[str, Any], ...]
    query_count: int
    database_version_or_release: str
    complete: bool
    issues: tuple[str, ...]
    receipt: dict[str, JsonValue]


class _ExternalRetrievalFailure(RuntimeError):
    def __init__(self, code: str, *, query_count: int) -> None:
        super().__init__(code)
        self.code = code
        self.query_count = query_count


class OptimadeStructureLookup:
    """Read-only OPTIMADE structure lookup followed by local strict identity.

    OPTIMADE filtering supplies bounded formula candidates; it never supplies a
    novelty Boolean.  Every ordered, complete structure payload is reconstructed
    locally and passed to :func:`classify_crystal_structure_relation`.  Missing
    provider/version/pagination metadata, truncated pages, disorder, and absent
    structure payloads prevent a provider ``no_match`` result.
    """

    matcher_policy = "optimade-formula-prefilter-local-strict-recheck-v1"

    def __init__(
        self,
        base_url: str,
        *,
        provider_id: str | None = None,
        database_version_or_release: str = LIVE_MOVING_SNAPSHOT_UNPINNED,
        page_limit: int = 100,
        max_pages: int = 20,
        max_records: int = 500,
        timeout_seconds: float = 30.0,
        max_response_bytes: int = 4 * 1024 * 1024,
        client_version: str | None = None,
        session: object | None = None,
        allow_loopback_http: bool = False,
    ) -> None:
        if not 1 <= page_limit <= 1_000:
            raise ValueError("page_limit must be between 1 and 1000")
        if not 1 <= max_pages <= 100:
            raise ValueError("max_pages must be between 1 and 100")
        if not 1 <= max_records <= 10_000:
            raise ValueError("max_records must be between 1 and 10000")
        if not 0 < timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between zero and 120")
        if not 1_024 <= max_response_bytes <= 64 * 1024 * 1024:
            raise ValueError("max_response_bytes must be between 1 KiB and 64 MiB")
        self.base_url = _validated_readonly_base_url(
            base_url,
            allow_loopback_http=allow_loopback_http,
            require_optimade_version=True,
        )
        self.provider_id = (
            provider_id.strip()
            if provider_id and provider_id.strip()
            else _provider_id_from_url("optimade", self.base_url)
        )
        release = database_version_or_release.strip()
        if not release:
            raise ValueError("database_version_or_release must not be blank")
        self.database_version_or_release = release
        self.page_limit = int(page_limit)
        self.max_pages = int(max_pages)
        self.max_records = int(max_records)
        self.timeout_seconds = float(timeout_seconds)
        self.max_response_bytes = int(max_response_bytes)
        self.client_version = (
            client_version.strip()
            if client_version and client_version.strip()
            else f"requests-{_installed_package_version('requests')}"
        )
        self._http = session if session is not None else requests.Session()
        self.matcher_settings: dict[str, JsonValue] = {
            "remote_prefilter": {
                "endpoint": f"{self.base_url}/structures",
                "filter_field": "chemical_formula_reduced",
                "response_fields": list(_OPTIMADE_STRUCTURE_FIELDS),
                "page_limit": self.page_limit,
                "max_pages": self.max_pages,
                "max_records": self.max_records,
                "prefilter_only": True,
            },
            "local_strict_recheck": _local_strict_matcher_settings(),
            "unsupported_structure_features": [
                "disorder",
                "implicit_atoms",
                "assemblies",
                "partial_occupancy",
            ],
            "no_match_requires": [
                "provider_metadata",
                "api_version",
                "database_version_or_release",
                "complete_pagination",
                "all_structure_payloads_locally_resolved",
            ],
        }

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        **kwargs: object,
    ) -> "OptimadeStructureLookup":
        values = os.environ if environ is None else environ
        kwargs.setdefault(
            "provider_id",
            values.get("OPTIMADE_PROVIDER_ID") or None,
        )
        kwargs.setdefault(
            "database_version_or_release",
            values.get(
                "OPTIMADE_DATABASE_VERSION_OR_RELEASE",
                LIVE_MOVING_SNAPSHOT_UNPINNED,
            ),
        )
        return cls(values.get("OPTIMADE_API_URL", ""), **kwargs)

    def lookup(self, candidate: Candidate) -> ExternalNoveltyOutcome:
        retrieved_at = datetime.now(timezone.utc)
        context, candidate_issue = _external_candidate_query_context(candidate)
        if candidate_issue is not None or context is None:
            return self._unknown(
                candidate_issue or "candidate_structure_context_unavailable",
                candidate=candidate,
                context=context,
                retrieved_at=retrieved_at,
                query_count=0,
            )
        try:
            retrieval = self._retrieve(context)
        except _ExternalRetrievalFailure as exc:
            return self._unknown(
                exc.code,
                candidate=candidate,
                context=context,
                retrieved_at=retrieved_at,
                query_count=exc.query_count,
            )

        matches: list[NoveltyMatch] = []
        similarities: list[NoveltyMatch] = []
        unresolved = False
        for resource in retrieval.records:
            finding, is_unresolved = _optimade_local_recheck(
                provider_id=self.provider_id,
                candidate_structure=context.representation.value,
                resource=resource,
            )
            if finding is None:
                unresolved = True
                continue
            if finding.match_kind == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE.value:
                matches.append(finding)
            else:
                similarities.append(finding)
            unresolved = unresolved or is_unresolved

        global_issues = list(retrieval.issues)
        if (
            retrieval.database_version_or_release
            == LIVE_MOVING_SNAPSHOT_UNPINNED
        ):
            global_issues.append("optimade_database_snapshot_unavailable")
        if not retrieval.complete:
            global_issues.append("optimade_pagination_incomplete")
        global_issues = sorted(set(global_issues))
        if global_issues:
            similarities.extend(
                _downgrade_untrusted_matches(
                    matches,
                    reason="optimade_provider_receipt_incomplete",
                )
            )
            matches = []
            status = NoveltyStatus.UNKNOWN
            reason = "optimade_provider_receipt_incomplete:" + ",".join(global_issues)
        elif matches:
            status = NoveltyStatus.MATCH
            reason = None
        elif unresolved:
            status = NoveltyStatus.UNKNOWN
            reason = "optimade_structure_payloads_not_all_strictly_resolved"
        else:
            status = NoveltyStatus.NO_MATCH
            reason = _scoped_no_match_reason(
                retrieval.database_version_or_release
            )

        provenance = self._provenance(
            candidate,
            context,
            retrieved_at=retrieved_at,
            retrieval=retrieval,
        )
        return ExternalNoveltyOutcome(
            **provenance,
            status=status,
            method="optimade-structures-local-strict-v1",
            query_count=retrieval.query_count,
            matches=matches,
            reason=reason,
            composition_match_count=len(retrieval.records),
            structure_match_count=len(matches),
            closest_match_id=matches[0].record_id if matches else None,
            similarity_findings=_unique_findings(similarities),
        )

    def _retrieve(
        self,
        context: _ExternalCandidateQueryContext,
    ) -> _ProviderRetrieval:
        endpoint = f"{self.base_url}/structures"
        query_filter = (
            'chemical_formula_reduced="'
            + context.optimade_reduced_formula
            + '"'
        )
        params: Mapping[str, object] | None = {
            "filter": query_filter,
            "response_fields": ",".join(_OPTIMADE_STRUCTURE_FIELDS),
            "page_limit": self.page_limit,
        }
        current_url = endpoint
        seen_urls: set[str] = set()
        records: list[Mapping[str, Any]] = []
        record_ids: set[str] = set()
        issues: list[str] = []
        page_receipts: list[dict[str, JsonValue]] = []
        api_version: str | None = None
        provider_prefix: str | None = None
        response_database_release: str | None = None
        complete = False
        query_count = 0

        for _page_index in range(self.max_pages):
            if current_url in seen_urls:
                issues.append("optimade_pagination_cycle")
                break
            seen_urls.add(current_url)
            query_count += 1
            try:
                response = self._http.get(
                    current_url,
                    params=params,
                    headers={"Accept": "application/vnd.api+json"},
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
                payload = _bounded_json_response(
                    response,
                    max_bytes=self.max_response_bytes,
                )
            except Exception as exc:
                issues.append(
                    f"optimade_page_lookup_failed:{type(exc).__name__}"
                )
                break
            params = None
            if not isinstance(payload, Mapping):
                issues.append("optimade_response_not_an_object")
                break
            if payload.get("errors"):
                issues.append("optimade_response_contains_errors")
                break
            data = payload.get("data")
            meta = payload.get("meta")
            links = payload.get("links", {})
            if not isinstance(data, list):
                issues.append("optimade_data_not_an_array")
                break
            if not isinstance(meta, Mapping):
                issues.append("optimade_meta_missing")
                break
            if not isinstance(links, Mapping):
                issues.append("optimade_links_not_an_object")
                break

            page_api_version = _bounded_text(meta.get("api_version"))
            query_meta = meta.get("query")
            query_representation = (
                _bounded_text(query_meta.get("representation"))
                if isinstance(query_meta, Mapping)
                else None
            )
            provider_meta = meta.get("provider")
            page_provider_prefix = (
                _bounded_text(provider_meta.get("prefix"))
                if isinstance(provider_meta, Mapping)
                else None
            )
            database_meta = meta.get("database")
            page_database_release = (
                _bounded_text(database_meta.get("version"))
                if isinstance(database_meta, Mapping)
                else None
            )
            implementation_meta = meta.get("implementation")
            implementation_version = (
                _bounded_text(implementation_meta.get("version"))
                if isinstance(implementation_meta, Mapping)
                else None
            )
            more_data = meta.get("more_data_available")
            data_returned = meta.get("data_returned")
            if page_api_version is None:
                issues.append("optimade_api_version_missing")
            elif not re.fullmatch(r"1(?:\.\d+){1,2}", page_api_version):
                issues.append("optimade_api_version_unsupported")
            elif api_version is None:
                api_version = page_api_version
            elif api_version != page_api_version:
                issues.append("optimade_api_version_changed_between_pages")
            if query_representation is None:
                issues.append("optimade_query_representation_missing")
            if page_provider_prefix is None:
                issues.append("optimade_provider_metadata_missing")
            elif provider_prefix is None:
                provider_prefix = page_provider_prefix
            elif provider_prefix != page_provider_prefix:
                issues.append("optimade_provider_changed_between_pages")
            if page_database_release is not None:
                if response_database_release is None:
                    response_database_release = page_database_release
                elif response_database_release != page_database_release:
                    issues.append("optimade_database_version_changed_between_pages")
                if (
                    self.database_version_or_release
                    != LIVE_MOVING_SNAPSHOT_UNPINNED
                    and self.database_version_or_release != page_database_release
                ):
                    issues.append("optimade_database_version_mismatch")
            if not isinstance(more_data, bool):
                issues.append("optimade_more_data_available_missing")
            if data_returned is not None and (
                isinstance(data_returned, bool)
                or not isinstance(data_returned, int)
                or data_returned < len(data)
            ):
                issues.append("optimade_data_returned_invalid")

            for resource in data:
                if not isinstance(resource, Mapping):
                    issues.append("optimade_structure_record_not_an_object")
                    continue
                record_id = _optimade_record_id(resource)
                if record_id is None:
                    issues.append("optimade_structure_record_id_missing")
                    continue
                if record_id in record_ids:
                    issues.append("optimade_duplicate_structure_record_id")
                    continue
                if len(records) >= self.max_records:
                    issues.append("optimade_record_limit_exceeded")
                    break
                record_ids.add(record_id)
                records.append(resource)

            next_url = _jsonapi_link_href(links.get("next"))
            page_receipts.append(
                {
                    "api_version": page_api_version,
                    "query_representation_sha256": (
                        stable_hash(query_representation)
                        if query_representation is not None
                        else None
                    ),
                    "provider_prefix": page_provider_prefix,
                    "database_version": page_database_release,
                    "implementation_version": implementation_version,
                    "time_stamp": _bounded_text(meta.get("time_stamp")),
                    "data_returned": (
                        data_returned
                        if isinstance(data_returned, int)
                        and not isinstance(data_returned, bool)
                        else None
                    ),
                    "data_available": (
                        meta.get("data_available")
                        if isinstance(meta.get("data_available"), int)
                        and not isinstance(meta.get("data_available"), bool)
                        else None
                    ),
                    "more_data_available": (
                        more_data if isinstance(more_data, bool) else None
                    ),
                    "next_present": next_url is not None,
                }
            )
            if issues:
                break
            if more_data is False:
                if next_url is not None:
                    issues.append("optimade_final_page_has_next_link")
                else:
                    complete = True
                break
            if next_url is None:
                issues.append("optimade_next_link_missing")
                break
            try:
                current_url = _validated_provider_next_url(
                    self.base_url,
                    next_url,
                )
            except ValueError:
                issues.append("optimade_next_link_outside_provider")
                break
        else:
            issues.append("optimade_page_limit_exceeded")

        database_release = (
            response_database_release
            or self.database_version_or_release
        )
        if (
            response_database_release is None
            and self.database_version_or_release
            == LIVE_MOVING_SNAPSHOT_UNPINNED
        ):
            issues.append("optimade_database_version_missing")
        receipt: dict[str, JsonValue] = {
            "endpoint": endpoint,
            "filter": query_filter,
            "api_version": api_version,
            "provider_prefix": provider_prefix,
            "database_version": response_database_release,
            "configured_database_version_or_release": (
                self.database_version_or_release
            ),
            "pages_retrieved": query_count,
            "records_returned": len(records),
            "pagination_complete": complete,
            "page_receipts": page_receipts,
            "issues": sorted(set(issues)),
        }
        return _ProviderRetrieval(
            records=tuple(records),
            query_count=query_count,
            database_version_or_release=database_release,
            complete=complete and not issues,
            issues=tuple(sorted(set(issues))),
            receipt=receipt,
        )

    def _unknown(
        self,
        reason: str,
        *,
        candidate: Candidate,
        context: _ExternalCandidateQueryContext | None,
        retrieved_at: datetime,
        query_count: int,
    ) -> ExternalNoveltyOutcome:
        receipt: dict[str, JsonValue] = {
            "endpoint": f"{self.base_url}/structures",
            "pagination_complete": False,
            "issues": [reason],
        }
        retrieval = _ProviderRetrieval(
            records=(),
            query_count=query_count,
            database_version_or_release=self.database_version_or_release,
            complete=False,
            issues=(reason,),
            receipt=receipt,
        )
        return ExternalNoveltyOutcome(
            **self._provenance(
                candidate,
                context,
                retrieved_at=retrieved_at,
                retrieval=retrieval,
            ),
            status=NoveltyStatus.UNKNOWN,
            method="optimade-structures-local-strict-v1",
            query_count=query_count,
            reason=reason,
        )

    def _provenance(
        self,
        candidate: Candidate,
        context: _ExternalCandidateQueryContext | None,
        *,
        retrieved_at: datetime,
        retrieval: _ProviderRetrieval,
    ) -> dict[str, object]:
        settings = dict(self.matcher_settings)
        settings["provider_receipt"] = retrieval.receipt
        query = {
            "provider_id": self.provider_id,
            "method": "optimade-structures-local-strict-v1",
            "endpoint": f"{self.base_url}/structures",
            "candidate_ref": _required_candidate_ref(candidate),
            "structure_sha256": context.structure_sha256 if context else None,
            "formula": (
                context.optimade_reduced_formula if context else None
            ),
            "database_version_or_release": retrieval.database_version_or_release,
            "response_fields": list(_OPTIMADE_STRUCTURE_FIELDS),
            "page_limit": self.page_limit,
            "max_pages": self.max_pages,
            "max_records": self.max_records,
            "matcher_policy": self.matcher_policy,
        }
        return {
            "provider_id": self.provider_id,
            "client_version": self.client_version,
            "database_version_or_release": (
                retrieval.database_version_or_release
            ),
            "retrieved_at": retrieved_at,
            "query_sha256": stable_hash(query),
            "matcher_policy": self.matcher_policy,
            "matcher_settings": settings,
        }


class CodStructureLookup:
    """Read-only COD formula search with revision-pinned CIF strict rechecks.

    The COD search endpoint has no documented paged-result contract.  A bounded
    full JSON result is therefore required.  Each result must expose a COD ID
    and revision so the exact ``.cif@REVISION`` artifact can be fetched.  A
    configured database release is required before an all-clear ``no_match`` is
    accepted; live unpinned searches remain ``unknown``.
    """

    provider_id = "cod"
    matcher_policy = "cod-formula-prefilter-revision-cif-local-strict-v1"

    def __init__(
        self,
        *,
        base_url: str = "https://www.crystallography.net/cod",
        database_version_or_release: str = LIVE_MOVING_SNAPSHOT_UNPINNED,
        max_records: int = 500,
        timeout_seconds: float = 30.0,
        max_search_response_bytes: int = 8 * 1024 * 1024,
        max_cif_bytes: int = 4 * 1024 * 1024,
        include_theoretical: bool = True,
        client_version: str | None = None,
        session: object | None = None,
        allow_loopback_http: bool = False,
    ) -> None:
        if not 1 <= max_records <= 10_000:
            raise ValueError("max_records must be between 1 and 10000")
        if not 0 < timeout_seconds <= 120:
            raise ValueError("timeout_seconds must be between zero and 120")
        if not 1_024 <= max_search_response_bytes <= 64 * 1024 * 1024:
            raise ValueError(
                "max_search_response_bytes must be between 1 KiB and 64 MiB"
            )
        if not 1_024 <= max_cif_bytes <= 16 * 1024 * 1024:
            raise ValueError("max_cif_bytes must be between 1 KiB and 16 MiB")
        self.base_url = _validated_readonly_base_url(
            base_url,
            allow_loopback_http=allow_loopback_http,
            require_optimade_version=False,
        )
        release = database_version_or_release.strip()
        if not release:
            raise ValueError("database_version_or_release must not be blank")
        self.database_version_or_release = release
        self.max_records = int(max_records)
        self.timeout_seconds = float(timeout_seconds)
        self.max_search_response_bytes = int(max_search_response_bytes)
        self.max_cif_bytes = int(max_cif_bytes)
        self.include_theoretical = bool(include_theoretical)
        self.client_version = (
            client_version.strip()
            if client_version and client_version.strip()
            else f"requests-{_installed_package_version('requests')}"
        )
        self._http = session if session is not None else requests.Session()
        self.matcher_settings: dict[str, JsonValue] = {
            "remote_prefilter": {
                "endpoint": f"{self.base_url}/result",
                "filter_field": "formula",
                "formula_notation": "Hill-separated",
                "include_duplicates": True,
                "include_errors": False,
                "include_theoretical": self.include_theoretical,
                "max_records": self.max_records,
                "prefilter_only": True,
            },
            "structure_fetch": {
                "format": "revision-pinned-cif",
                "url_contract": f"{self.base_url}/COD_ID.cif@REVISION",
            },
            "local_strict_recheck": _local_strict_matcher_settings(),
            "no_match_requires": [
                "configured_database_version_or_release",
                "complete_bounded_search_response",
                "revision_for_every_record",
                "all_cif_payloads_locally_resolved",
            ],
        }

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        **kwargs: object,
    ) -> "CodStructureLookup":
        values = os.environ if environ is None else environ
        kwargs.setdefault(
            "base_url",
            values.get(
                "COD_API_URL",
                "https://www.crystallography.net/cod",
            ),
        )
        kwargs.setdefault(
            "database_version_or_release",
            values.get(
                "COD_DATABASE_VERSION_OR_RELEASE",
                LIVE_MOVING_SNAPSHOT_UNPINNED,
            ),
        )
        return cls(**kwargs)

    def lookup(self, candidate: Candidate) -> ExternalNoveltyOutcome:
        retrieved_at = datetime.now(timezone.utc)
        context, candidate_issue = _external_candidate_query_context(candidate)
        if candidate_issue is not None or context is None:
            return self._unknown(
                candidate_issue or "candidate_structure_context_unavailable",
                candidate=candidate,
                context=context,
                retrieved_at=retrieved_at,
                query_count=0,
            )
        try:
            retrieval = self._search(context)
        except _ExternalRetrievalFailure as exc:
            return self._unknown(
                exc.code,
                candidate=candidate,
                context=context,
                retrieved_at=retrieved_at,
                query_count=exc.query_count,
            )

        matches: list[NoveltyMatch] = []
        similarities: list[NoveltyMatch] = []
        unresolved = False
        query_count = retrieval.query_count
        for row in retrieval.records:
            record_id, revision = _cod_record_identity(row)
            if record_id is None or revision is None:
                unresolved = True
                similarities.append(
                    NoveltyMatch(
                        source_id=self.provider_id,
                        record_id=record_id or "cod-record-without-id",
                        match_kind="cod-structure-unverified",
                        metadata={
                            "strict_recheck": (
                                "record_id_missing"
                                if record_id is None
                                else "record_revision_missing"
                            ),
                            "hard_identity": "false",
                        },
                    )
                )
                continue
            query_count += 1
            cif_url = f"{self.base_url}/{record_id}.cif@{revision}"
            try:
                response = self._http.get(
                    cif_url,
                    headers={"Accept": "chemical/x-cif,text/plain"},
                    timeout=self.timeout_seconds,
                    allow_redirects=False,
                    stream=True,
                )
                remote_cif = _bounded_text_response(
                    response,
                    max_bytes=self.max_cif_bytes,
                )
            except Exception as exc:
                unresolved = True
                similarities.append(
                    NoveltyMatch(
                        source_id=self.provider_id,
                        record_id=f"{record_id}@{revision}",
                        match_kind="cod-structure-unverified",
                        metadata={
                            "strict_recheck": f"cif_fetch_failed:{type(exc).__name__}",
                            "hard_identity": "false",
                            "revision": revision,
                        },
                    )
                )
                continue
            finding, is_unresolved = _cod_local_recheck(
                candidate_structure=context.representation.value,
                record_id=record_id,
                revision=revision,
                row=row,
                remote_cif=remote_cif,
                cif_url=cif_url,
            )
            if finding is None:
                unresolved = True
                continue
            if finding.match_kind == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE.value:
                matches.append(finding)
            else:
                similarities.append(finding)
            unresolved = unresolved or is_unresolved

        global_issues = list(retrieval.issues)
        if self.database_version_or_release == LIVE_MOVING_SNAPSHOT_UNPINNED:
            global_issues.append("cod_database_snapshot_unavailable")
        if not retrieval.complete:
            global_issues.append("cod_search_completeness_unverified")
        global_issues = sorted(set(global_issues))
        if global_issues:
            similarities.extend(
                _downgrade_untrusted_matches(
                    matches,
                    reason="cod_provider_receipt_incomplete",
                )
            )
            matches = []
            status = NoveltyStatus.UNKNOWN
            reason = "cod_provider_receipt_incomplete:" + ",".join(global_issues)
        elif matches:
            status = NoveltyStatus.MATCH
            reason = None
        elif unresolved:
            status = NoveltyStatus.UNKNOWN
            reason = "cod_records_not_all_revision_pinned_and_strictly_resolved"
        else:
            status = NoveltyStatus.NO_MATCH
            reason = _scoped_no_match_reason(self.database_version_or_release)

        dynamic_receipt = dict(retrieval.receipt)
        dynamic_receipt["cif_fetch_count"] = query_count - retrieval.query_count
        dynamic_receipt["all_structure_payloads_resolved"] = not unresolved
        retrieval = _ProviderRetrieval(
            records=retrieval.records,
            query_count=query_count,
            database_version_or_release=retrieval.database_version_or_release,
            complete=retrieval.complete and not unresolved,
            issues=tuple(sorted(set([*retrieval.issues]))),
            receipt=dynamic_receipt,
        )
        return ExternalNoveltyOutcome(
            **self._provenance(
                candidate,
                context,
                retrieved_at=retrieved_at,
                retrieval=retrieval,
            ),
            status=status,
            method="cod-result-revision-cif-local-strict-v1",
            query_count=query_count,
            matches=matches,
            reason=reason,
            composition_match_count=len(retrieval.records),
            structure_match_count=len(matches),
            closest_match_id=matches[0].record_id if matches else None,
            similarity_findings=_unique_findings(similarities),
        )

    def _search(
        self,
        context: _ExternalCandidateQueryContext,
    ) -> _ProviderRetrieval:
        endpoint = f"{self.base_url}/result"
        params: dict[str, object] = {
            "formula": context.cod_hill_formula,
            "format": "json",
            "include_duplicates": "1",
        }
        if self.include_theoretical:
            params["include_theoretical"] = "1"
        try:
            response = self._http.get(
                endpoint,
                params=params,
                headers={"Accept": "application/json"},
                timeout=self.timeout_seconds,
                allow_redirects=False,
                stream=True,
            )
            payload = _bounded_json_response(
                response,
                max_bytes=self.max_search_response_bytes,
            )
        except Exception as exc:
            raise _ExternalRetrievalFailure(
                f"cod_lookup_failed:{type(exc).__name__}",
                query_count=1,
            ) from None
        issues: list[str] = []
        if isinstance(payload, list):
            raw_records = payload
        elif isinstance(payload, Mapping) and isinstance(payload.get("data"), list):
            raw_records = payload["data"]
        elif isinstance(payload, Mapping) and isinstance(payload.get("records"), list):
            raw_records = payload["records"]
        else:
            raw_records = []
            issues.append("cod_search_response_not_a_record_array")
        records: list[Mapping[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                issues.append("cod_search_record_not_an_object")
                continue
            record_id, revision = _cod_record_identity(raw)
            key = (record_id or "", revision or "")
            if key in seen:
                issues.append("cod_duplicate_search_record")
                continue
            seen.add(key)
            if len(records) >= self.max_records:
                issues.append("cod_record_limit_exceeded")
                break
            records.append(raw)
        headers = getattr(response, "headers", {})
        receipt: dict[str, JsonValue] = {
            "endpoint": endpoint,
            "query": {
                "formula": context.cod_hill_formula,
                "format": "json",
                "include_duplicates": True,
                "include_theoretical": self.include_theoretical,
                "include_errors": False,
            },
            "configured_database_version_or_release": (
                self.database_version_or_release
            ),
            "pagination_mode": "single-complete-json-result",
            "pagination_complete": not issues,
            "records_returned": len(records),
            "etag": (
                _bounded_text(headers.get("ETag"))
                if isinstance(headers, Mapping)
                else None
            ),
            "last_modified": (
                _bounded_text(headers.get("Last-Modified"))
                if isinstance(headers, Mapping)
                else None
            ),
            "issues": sorted(set(issues)),
        }
        return _ProviderRetrieval(
            records=tuple(records),
            query_count=1,
            database_version_or_release=self.database_version_or_release,
            complete=not issues,
            issues=tuple(sorted(set(issues))),
            receipt=receipt,
        )

    def _unknown(
        self,
        reason: str,
        *,
        candidate: Candidate,
        context: _ExternalCandidateQueryContext | None,
        retrieved_at: datetime,
        query_count: int,
    ) -> ExternalNoveltyOutcome:
        retrieval = _ProviderRetrieval(
            records=(),
            query_count=query_count,
            database_version_or_release=self.database_version_or_release,
            complete=False,
            issues=(reason,),
            receipt={
                "endpoint": f"{self.base_url}/result",
                "pagination_complete": False,
                "issues": [reason],
            },
        )
        return ExternalNoveltyOutcome(
            **self._provenance(
                candidate,
                context,
                retrieved_at=retrieved_at,
                retrieval=retrieval,
            ),
            status=NoveltyStatus.UNKNOWN,
            method="cod-result-revision-cif-local-strict-v1",
            query_count=query_count,
            reason=reason,
        )

    def _provenance(
        self,
        candidate: Candidate,
        context: _ExternalCandidateQueryContext | None,
        *,
        retrieved_at: datetime,
        retrieval: _ProviderRetrieval,
    ) -> dict[str, object]:
        settings = dict(self.matcher_settings)
        settings["provider_receipt"] = retrieval.receipt
        query = {
            "provider_id": self.provider_id,
            "method": "cod-result-revision-cif-local-strict-v1",
            "endpoint": f"{self.base_url}/result",
            "candidate_ref": _required_candidate_ref(candidate),
            "structure_sha256": context.structure_sha256 if context else None,
            "formula": context.cod_hill_formula if context else None,
            "database_version_or_release": retrieval.database_version_or_release,
            "include_duplicates": True,
            "include_errors": False,
            "include_theoretical": self.include_theoretical,
            "max_records": self.max_records,
            "matcher_policy": self.matcher_policy,
        }
        return {
            "provider_id": self.provider_id,
            "client_version": self.client_version,
            "database_version_or_release": (
                retrieval.database_version_or_release
            ),
            "retrieved_at": retrieved_at,
            "query_sha256": stable_hash(query),
            "matcher_policy": self.matcher_policy,
            "matcher_settings": settings,
        }


def build_external_novelty_lookups_from_environment(
    environ: Mapping[str, str] | None = None,
    *,
    mp_rester_factory: Callable[[str], object] | None = None,
    optimade_session: object | None = None,
    cod_session: object | None = None,
) -> list[ExternalNoveltyLookup]:
    """Build the explicitly configured read-only external structure panel.

    Providers are ordered deterministically as Materials Project, OPTIMADE, and
    COD.  A blank provider credential/URL disables only that provider.  Once a
    provider is configured, malformed URLs, numeric limits, or switches raise
    immediately rather than silently weakening external novelty coverage.

    Environment variables:

    - ``MP_API_KEY`` and optional ``MP_DATABASE_VERSION_OR_RELEASE``;
    - ``OPTIMADE_API_URL`` (explicit versioned v1 base), optional
      ``OPTIMADE_PROVIDER_ID`` and ``OPTIMADE_DATABASE_VERSION_OR_RELEASE``;
    - ``COD_API_URL`` and optional ``COD_DATABASE_VERSION_OR_RELEASE``;
    - optional bounded controls ``OPTIMADE_PAGE_LIMIT``,
      ``OPTIMADE_MAX_PAGES``, ``OPTIMADE_MAX_RECORDS``, ``COD_MAX_RECORDS``,
      ``EXTERNAL_NOVELTY_TIMEOUT_SECONDS``, ``COD_INCLUDE_THEORETICAL``, and
      ``EXTERNAL_NOVELTY_ALLOW_LOOPBACK_HTTP``.
    """

    values = os.environ if environ is None else environ
    lookups: list[ExternalNoveltyLookup] = []
    mp_api_key = str(values.get("MP_API_KEY", "")).strip()
    optimade_url = str(values.get("OPTIMADE_API_URL", "")).strip()
    cod_url = str(values.get("COD_API_URL", "")).strip()
    if not any((mp_api_key, optimade_url, cod_url)):
        return lookups
    if optimade_url or cod_url:
        allow_loopback_http = _environment_switch(
            values,
            "EXTERNAL_NOVELTY_ALLOW_LOOPBACK_HTTP",
            default=False,
        )
        timeout_seconds = _environment_float(
            values,
            "EXTERNAL_NOVELTY_TIMEOUT_SECONDS",
            default=30.0,
        )
    else:
        allow_loopback_http = False
        timeout_seconds = 30.0

    if mp_api_key:
        mp_kwargs: dict[str, object] = {
            "database_version_or_release": _environment_optional_text(
                values,
                "MP_DATABASE_VERSION_OR_RELEASE",
            )
            or LIVE_MOVING_SNAPSHOT_UNPINNED,
        }
        if mp_rester_factory is not None:
            mp_kwargs["rester_factory"] = mp_rester_factory
        lookups.append(
            MaterialsProjectStructureLookup(
                mp_api_key,
                **mp_kwargs,
            )
        )

    if optimade_url:
        lookups.append(
            OptimadeStructureLookup(
                optimade_url,
                provider_id=(
                    _environment_optional_text(values, "OPTIMADE_PROVIDER_ID")
                ),
                database_version_or_release=(
                    _environment_optional_text(
                        values,
                        "OPTIMADE_DATABASE_VERSION_OR_RELEASE",
                    )
                    or LIVE_MOVING_SNAPSHOT_UNPINNED
                ),
                page_limit=_environment_int(
                    values,
                    "OPTIMADE_PAGE_LIMIT",
                    default=100,
                ),
                max_pages=_environment_int(
                    values,
                    "OPTIMADE_MAX_PAGES",
                    default=20,
                ),
                max_records=_environment_int(
                    values,
                    "OPTIMADE_MAX_RECORDS",
                    default=500,
                ),
                timeout_seconds=timeout_seconds,
                session=optimade_session,
                allow_loopback_http=allow_loopback_http,
            )
        )

    if cod_url:
        lookups.append(
            CodStructureLookup(
                base_url=cod_url,
                database_version_or_release=(
                    _environment_optional_text(
                        values,
                        "COD_DATABASE_VERSION_OR_RELEASE",
                    )
                    or LIVE_MOVING_SNAPSHOT_UNPINNED
                ),
                max_records=_environment_int(
                    values,
                    "COD_MAX_RECORDS",
                    default=500,
                ),
                timeout_seconds=timeout_seconds,
                include_theoretical=_environment_switch(
                    values,
                    "COD_INCLUDE_THEORETICAL",
                    default=True,
                ),
                session=cod_session,
                allow_loopback_http=allow_loopback_http,
            )
        )
    return lookups


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


_OPTIMADE_STRUCTURE_FIELDS = (
    "immutable_id",
    "last_modified",
    "chemical_formula_reduced",
    "elements",
    "nelements",
    "nsites",
    "lattice_vectors",
    "cartesian_site_positions",
    "species_at_sites",
    "species",
    "dimension_types",
    "nperiodic_dimensions",
    "structure_features",
    "assemblies",
)


def _local_strict_matcher_settings() -> dict[str, JsonValue]:
    return {
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
        "hard_identity_relation": (
            CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE.value
        ),
    }


def _validated_readonly_base_url(
    value: str,
    *,
    allow_loopback_http: bool,
    require_optimade_version: bool,
) -> str:
    text = str(value).strip().rstrip("/")
    if not text or len(text) > 2_000:
        raise ValueError("external structure provider base URL is required")
    parsed = urlsplit(text)
    if parsed.username or parsed.password:
        raise ValueError("external structure provider URLs must not contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("external structure provider base URLs cannot contain query/fragment")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise ValueError("external structure provider URL requires a host")
    if parsed.scheme != "https":
        if not (
            allow_loopback_http
            and parsed.scheme == "http"
            and _is_loopback_host(host)
        ):
            raise ValueError(
                "external structure provider URLs require HTTPS; "
                "loopback HTTP requires explicit opt-in"
            )
    if require_optimade_version and not re.search(
        r"/v1(?:\.\d+(?:\.\d+)?)?$",
        parsed.path.rstrip("/"),
    ):
        raise ValueError("OPTIMADE lookup requires an explicit versioned v1 base URL")
    return text


def _is_loopback_host(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _provider_id_from_url(prefix: str, url: str) -> str:
    host = (urlsplit(url).hostname or "provider").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", host).strip("-") or "provider"
    return f"{prefix}-{slug}"[:256]


def _validated_provider_next_url(base_url: str, next_link: str) -> str:
    if not next_link or len(next_link) > 4_096:
        raise ValueError("provider next link is blank or too long")
    absolute = urljoin(base_url.rstrip("/") + "/", next_link)
    base = urlsplit(base_url)
    target = urlsplit(absolute)
    if (
        target.scheme != base.scheme
        or target.netloc != base.netloc
        or target.username
        or target.password
        or target.fragment
    ):
        raise ValueError("provider next link leaves the configured origin")
    base_path = base.path.rstrip("/")
    if not (
        target.path == base_path
        or target.path.startswith(base_path + "/")
    ):
        raise ValueError("provider next link leaves the configured versioned base path")
    return absolute


def _bounded_json_response(response: object, *, max_bytes: int) -> object:
    status_code = getattr(response, "status_code", 200)
    if (
        isinstance(status_code, bool)
        or not isinstance(status_code, int)
        or not 200 <= status_code < 300
    ):
        raise ValueError("provider returned a non-success HTTP status")
    raise_for_status = getattr(response, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()
    raw = _read_bounded_response_bytes(response, max_bytes=max_bytes)
    if raw:
        try:
            payload = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("provider response is not valid UTF-8 JSON") from exc
    else:
        loader = getattr(response, "json", None)
        if not callable(loader):
            raise TypeError("provider response has no JSON decoder")
        payload = loader()
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise ValueError("provider decoded JSON exceeds the configured byte limit")
    return payload


def _bounded_text_response(response: object, *, max_bytes: int) -> str:
    status_code = getattr(response, "status_code", 200)
    if (
        isinstance(status_code, bool)
        or not isinstance(status_code, int)
        or not 200 <= status_code < 300
    ):
        raise ValueError("provider returned a non-success HTTP status")
    raise_for_status = getattr(response, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()
    content = _read_bounded_response_bytes(response, max_bytes=max_bytes)
    if content:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("provider text response is not UTF-8") from exc
    else:
        text = str(getattr(response, "text", ""))
        if len(text.encode("utf-8")) > max_bytes:
            raise ValueError("provider text response exceeds the configured byte limit")
    if not text.strip() or "\x00" in text:
        raise ValueError("provider text response is empty or invalid")
    return text


def _read_bounded_response_bytes(response: object, *, max_bytes: int) -> bytes:
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        chunks: list[bytes] = []
        size = 0
        try:
            for chunk in iterator(chunk_size=min(64 * 1024, max_bytes + 1)):
                if not chunk:
                    continue
                raw = bytes(chunk)
                size += len(raw)
                if size > max_bytes:
                    raise ValueError(
                        "provider response exceeds the configured byte limit"
                    )
                chunks.append(raw)
            return b"".join(chunks)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)):
        raw = bytes(content)
        if len(raw) > max_bytes:
            raise ValueError("provider response exceeds the configured byte limit")
        return raw
    return b""


def _bounded_text(value: object, *, limit: int = 2_000) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:limit]


def _external_candidate_query_context(
    candidate: Candidate,
) -> tuple[_ExternalCandidateQueryContext | None, str | None]:
    representation = _representation(candidate, RepresentationKind.CIF)
    fmt = "cif"
    if representation is None:
        representation = _representation(candidate, RepresentationKind.POSCAR)
        fmt = "poscar"
    if representation is None:
        return None, "candidate_has_no_cif_or_poscar_representation"
    try:
        structure = parse_crystal_structure(representation.value, fmt=fmt)
        validate_crystal_geometry(structure)
        occupancy = inspect_crystal_occupancy(structure)
    except PymatgenRequiredError:
        return None, "crystal_identity_dependency_not_installed"
    except CrystalIdentityError as exc:
        return None, f"candidate_crystal_identity_failed:{type(exc).__name__}"
    if not occupancy.is_fully_occupied_ordered:
        suffix = ",".join(occupancy.reason_codes) or "unsupported_occupancy"
        return (
            None,
            "candidate_disorder_or_partial_occupancy_unsupported:" + suffix,
        )
    try:
        counts = _ordered_structure_element_counts(structure)
    except ValueError:
        return None, "candidate_species_not_supported_for_external_identity"
    normalized = _normalized_representation_value(representation)
    return (
        _ExternalCandidateQueryContext(
            representation=representation,
            fmt=fmt,
            structure=structure,
            structure_sha256=stable_hash(
                {
                    "format": fmt,
                    "representation": normalized,
                }
            ),
            optimade_reduced_formula=_formula_from_counts(
                counts,
                order="alphabetical",
                separator="",
            ),
            cod_hill_formula=_formula_from_counts(
                counts,
                order="hill",
                separator=" ",
            ),
        ),
        None,
    )


def _ordered_structure_element_counts(structure: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for site in structure:
        species = list(site.species.items())
        if len(species) != 1 or abs(float(species[0][1]) - 1.0) > 1e-8:
            raise ValueError("strict external identity requires one full species per site")
        specie = species[0][0]
        symbol = getattr(specie, "symbol", None)
        if symbol is None:
            element = getattr(specie, "element", None)
            symbol = getattr(element, "symbol", None)
        text = str(symbol).strip() if symbol is not None else ""
        if not re.fullmatch(r"[A-Z][a-z]?", text):
            raise ValueError("strict external identity requires chemical element species")
        counts[text] = counts.get(text, 0) + 1
    if not counts:
        raise ValueError("crystal contains no supported elements")
    divisor = 0
    for amount in counts.values():
        divisor = amount if divisor == 0 else _greatest_common_divisor(divisor, amount)
    return {key: value // max(1, divisor) for key, value in counts.items()}


def _greatest_common_divisor(first: int, second: int) -> int:
    left, right = abs(int(first)), abs(int(second))
    while right:
        left, right = right, left % right
    return max(1, left)


def _formula_from_counts(
    counts: Mapping[str, int],
    *,
    order: str,
    separator: str,
) -> str:
    if order == "alphabetical":
        symbols = sorted(counts)
    elif order == "hill":
        if "C" in counts:
            symbols = [
                "C",
                *(["H"] if "H" in counts else []),
                *sorted(key for key in counts if key not in {"C", "H"}),
            ]
        else:
            symbols = sorted(counts)
    else:
        raise ValueError("unknown formula ordering")
    return separator.join(
        symbol + (str(counts[symbol]) if counts[symbol] != 1 else "")
        for symbol in symbols
    )


def _optimade_record_id(resource: Mapping[str, Any]) -> str | None:
    record_id = _bounded_text(resource.get("id"), limit=512)
    record_type = _bounded_text(resource.get("type"), limit=128)
    if record_id is None or record_type != "structures":
        return None
    return record_id


def _jsonapi_link_href(value: object) -> str | None:
    if isinstance(value, str):
        return _bounded_text(value, limit=4_096)
    if isinstance(value, Mapping):
        return _bounded_text(value.get("href"), limit=4_096)
    return None


def _optimade_local_recheck(
    *,
    provider_id: str,
    candidate_structure: str,
    resource: Mapping[str, Any],
) -> tuple[NoveltyMatch | None, bool]:
    record_id = _optimade_record_id(resource)
    if record_id is None:
        return None, True
    remote, payload_metadata, issue = _optimade_resource_to_structure(resource)
    if issue is not None or remote is None:
        return (
            NoveltyMatch(
                source_id=provider_id,
                record_id=record_id,
                match_kind="optimade-structure-unverified",
                metadata={
                    **payload_metadata,
                    "strict_recheck": issue or "structure_payload_unavailable",
                    "hard_identity": "false",
                },
            ),
            True,
        )
    try:
        assessment = classify_crystal_structure_relation(
            candidate_structure,
            remote,
        )
    except Exception as exc:
        return (
            NoveltyMatch(
                source_id=provider_id,
                record_id=record_id,
                match_kind="optimade-structure-unverified",
                metadata={
                    **payload_metadata,
                    "strict_recheck": f"failed:{type(exc).__name__}",
                    "hard_identity": "false",
                },
            ),
            True,
        )
    finding = _assessment_finding(
        provider_id=provider_id,
        record_id=record_id,
        assessment=assessment,
        metadata=payload_metadata,
    )
    return finding, assessment.relation == CrystalMatchRelation.AMBIGUOUS


def _optimade_resource_to_structure(
    resource: Mapping[str, Any],
) -> tuple[Any | None, dict[str, str], str | None]:
    attributes = resource.get("attributes")
    if not isinstance(attributes, Mapping):
        return None, {}, "attributes_missing"
    immutable_id = _bounded_text(attributes.get("immutable_id"), limit=1_000)
    last_modified = _bounded_text(attributes.get("last_modified"), limit=128)
    formula = _bounded_text(
        attributes.get("chemical_formula_reduced"),
        limit=512,
    )
    features_raw = attributes.get("structure_features")
    features = (
        sorted(
            {
                str(item).strip().lower()
                for item in features_raw
                if str(item).strip()
            }
        )
        if isinstance(features_raw, list)
        else []
    )
    metadata = {
        "immutable_id": immutable_id or "not-reported",
        "last_modified": last_modified or "not-reported",
        "chemical_formula_reduced": formula or "not-reported",
        "structure_features": ",".join(features) or "none",
    }
    unsupported = {"disorder", "implicit_atoms", "assemblies"}
    if unsupported.intersection(features):
        return None, metadata, "unsupported_structure_features"
    assemblies = attributes.get("assemblies")
    if assemblies not in (None, []):
        return None, metadata, "assemblies_not_supported"
    dimensions = attributes.get("dimension_types")
    if dimensions is not None and dimensions != [1, 1, 1]:
        return None, metadata, "non_three_dimensional_structure"
    periodic_dimensions = attributes.get("nperiodic_dimensions")
    if periodic_dimensions is not None and periodic_dimensions != 3:
        return None, metadata, "non_three_dimensional_structure"

    lattice = attributes.get("lattice_vectors")
    positions = attributes.get("cartesian_site_positions")
    species_at_sites = attributes.get("species_at_sites")
    species_rows = attributes.get("species")
    nsites = attributes.get("nsites")
    if not (
        isinstance(lattice, list)
        and len(lattice) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in lattice)
    ):
        return None, metadata, "lattice_vectors_missing_or_invalid"
    if not isinstance(positions, list) or not isinstance(species_at_sites, list):
        return None, metadata, "site_positions_or_species_at_sites_missing"
    if (
        len(positions) != len(species_at_sites)
        or not positions
        or not isinstance(nsites, int)
        or isinstance(nsites, bool)
        or nsites != len(positions)
    ):
        return None, metadata, "site_count_or_site_arrays_inconsistent"
    if not isinstance(species_rows, list):
        return None, metadata, "species_definitions_missing"

    species_by_name: dict[str, str] = {}
    for raw in species_rows:
        if not isinstance(raw, Mapping):
            return None, metadata, "species_definition_not_an_object"
        name = _bounded_text(raw.get("name"), limit=256)
        symbols = raw.get("chemical_symbols")
        concentrations = raw.get("concentration")
        if (
            name is None
            or name in species_by_name
            or not isinstance(symbols, list)
            or not isinstance(concentrations, list)
            or len(symbols) != 1
            or len(concentrations) != 1
        ):
            return None, metadata, "disordered_or_invalid_species_definition"
        symbol = str(symbols[0]).strip()
        try:
            concentration = float(concentrations[0])
        except (TypeError, ValueError):
            return None, metadata, "invalid_species_concentration"
        if (
            not re.fullmatch(r"[A-Z][a-z]?", symbol)
            or abs(concentration - 1.0) > 1e-8
        ):
            return None, metadata, "partial_or_non_element_species_definition"
        species_by_name[name] = symbol
    try:
        site_species = [species_by_name[str(item)] for item in species_at_sites]
    except (KeyError, TypeError):
        return None, metadata, "species_at_sites_reference_unknown_definition"
    try:
        lattice_values = [[float(item) for item in row] for row in lattice]
        position_values = [[float(item) for item in row] for row in positions]
    except (TypeError, ValueError):
        return None, metadata, "non_numeric_lattice_or_site_position"
    if any(
        not all(_finite_number(item) for item in row)
        for row in [*lattice_values, *position_values]
    ):
        return None, metadata, "non_finite_lattice_or_site_position"
    if any(len(row) != 3 for row in position_values):
        return None, metadata, "cartesian_site_position_shape_invalid"
    try:
        from pymatgen.core import Lattice, Structure

        remote = Structure(
            Lattice(lattice_values),
            site_species,
            position_values,
            coords_are_cartesian=True,
            to_unit_cell=True,
        )
        validate_crystal_geometry(remote)
        occupancy = inspect_crystal_occupancy(remote)
    except Exception as exc:
        return (
            None,
            metadata,
            f"local_structure_reconstruction_failed:{type(exc).__name__}",
        )
    if not occupancy.is_fully_occupied_ordered:
        return None, metadata, "partial_or_disordered_reconstructed_structure"
    payload = {
        "lattice_vectors": lattice_values,
        "cartesian_site_positions": position_values,
        "species_at_sites": site_species,
    }
    metadata["structure_payload_sha256"] = stable_hash(payload)
    return remote, metadata, None


def _finite_number(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def _cod_record_identity(
    row: Mapping[str, Any],
) -> tuple[str | None, str | None]:
    raw_id = (
        row.get("file")
        or row.get("id")
        or row.get("cod_id")
        or row.get("codid")
    )
    record_id = str(raw_id).strip() if raw_id is not None else ""
    record_id = re.sub(r"\.cif(?:@\d+)?$", "", record_id, flags=re.IGNORECASE)
    if not re.fullmatch(r"\d{5,12}", record_id):
        record_id = ""
    raw_revision = (
        row.get("svnrevision")
        or row.get("revision")
        or row.get("rev")
    )
    revision = str(raw_revision).strip() if raw_revision is not None else ""
    if not re.fullmatch(r"\d{1,20}", revision):
        revision = ""
    return record_id or None, revision or None


def _cod_local_recheck(
    *,
    candidate_structure: str,
    record_id: str,
    revision: str,
    row: Mapping[str, Any],
    remote_cif: str,
    cif_url: str,
) -> tuple[NoveltyMatch | None, bool]:
    stable_record_id = f"{record_id}@{revision}"
    metadata = {
        "cod_id": record_id,
        "revision": revision,
        "source_url": cif_url,
        "theoretical": _bounded_text(
            row.get("theoretical")
            or row.get("is_theoretical"),
            limit=64,
        )
        or "not-reported",
        "duplicate_of": _bounded_text(
            row.get("duplicateof") or row.get("duplicate_of"),
            limit=512,
        )
        or "none",
        "structure_payload_sha256": stable_hash(
            remote_cif.replace("\r\n", "\n").replace("\r", "\n").strip()
        ),
    }
    if len(re.findall(r"(?im)^\s*data_", remote_cif)) != 1:
        return (
            NoveltyMatch(
                source_id="cod",
                record_id=stable_record_id,
                match_kind="cod-structure-unverified",
                metadata={
                    **metadata,
                    "strict_recheck": "cif_requires_exactly_one_data_block",
                    "hard_identity": "false",
                },
            ),
            True,
        )
    try:
        validate_crystal_geometry(remote_cif, fmt="cif")
        occupancy = inspect_crystal_occupancy(remote_cif, fmt="cif")
    except Exception as exc:
        return (
            NoveltyMatch(
                source_id="cod",
                record_id=stable_record_id,
                match_kind="cod-structure-unverified",
                metadata={
                    **metadata,
                    "strict_recheck": f"cif_parse_failed:{type(exc).__name__}",
                    "hard_identity": "false",
                },
            ),
            True,
        )
    if not occupancy.is_fully_occupied_ordered:
        return (
            NoveltyMatch(
                source_id="cod",
                record_id=stable_record_id,
                match_kind="cod-structure-unverified",
                metadata={
                    **metadata,
                    "strict_recheck": (
                        "partial_or_disordered_structure:"
                        + ",".join(occupancy.reason_codes)
                    ),
                    "hard_identity": "false",
                },
            ),
            True,
        )
    try:
        assessment = classify_crystal_structure_relation(
            candidate_structure,
            remote_cif,
        )
    except Exception as exc:
        return (
            NoveltyMatch(
                source_id="cod",
                record_id=stable_record_id,
                match_kind="cod-structure-unverified",
                metadata={
                    **metadata,
                    "strict_recheck": f"failed:{type(exc).__name__}",
                    "hard_identity": "false",
                },
            ),
            True,
        )
    finding = _assessment_finding(
        provider_id="cod",
        record_id=stable_record_id,
        assessment=assessment,
        metadata=metadata,
    )
    return finding, assessment.relation == CrystalMatchRelation.AMBIGUOUS


def _assessment_finding(
    *,
    provider_id: str,
    record_id: str,
    assessment: Any,
    metadata: Mapping[str, str],
) -> NoveltyMatch:
    return NoveltyMatch(
        source_id=provider_id,
        record_id=record_id,
        match_kind=assessment.relation.value,
        metadata={
            **dict(metadata),
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
            "hard_identity": str(
                assessment.relation
                == CrystalMatchRelation.STRICT_MATERIAL_DUPLICATE
            ).lower(),
        },
    )


def _downgrade_untrusted_matches(
    matches: Sequence[NoveltyMatch],
    *,
    reason: str,
) -> list[NoveltyMatch]:
    return [
        item.model_copy(
            update={
                "match_kind": "strict-local-match-provider-receipt-incomplete",
                "metadata": {
                    **item.metadata,
                    "hard_identity": "false",
                    "provider_receipt": reason,
                },
            }
        )
        for item in matches
    ]


def _unique_findings(
    findings: Sequence[NoveltyMatch],
) -> list[NoveltyMatch]:
    by_key: dict[tuple[str, str], NoveltyMatch] = {}
    for item in findings:
        by_key[(item.source_id, item.record_id)] = item
    return [by_key[key] for key in sorted(by_key)]


def _environment_optional_text(
    values: Mapping[str, str],
    key: str,
) -> str | None:
    text = str(values.get(key, "")).strip()
    return text or None


def _environment_int(
    values: Mapping[str, str],
    key: str,
    *,
    default: int,
) -> int:
    text = str(values.get(key, "")).strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc


def _environment_float(
    values: Mapping[str, str],
    key: str,
    *,
    default: float,
) -> float:
    text = str(values.get(key, "")).strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{key} must be a number") from exc


def _environment_switch(
    values: Mapping[str, str],
    key: str,
    *,
    default: bool,
) -> bool:
    text = str(values.get(key, "")).strip()
    if not text:
        return default
    if text == "1":
        return True
    if text == "0":
        return False
    raise ValueError(f"{key} must be '0' or '1'")


__all__ = [
    "CodStructureLookup",
    "ExternalNoveltyLookup",
    "ExternalNoveltyOutcome",
    "LIVE_MOVING_SNAPSHOT_UNPINNED",
    "MaterialsProjectStructureLookup",
    "NoveltyMatch",
    "NoveltyPortfolioSelection",
    "NoveltyStage",
    "NoveltyStageResult",
    "NoveltyStatus",
    "OptimadeStructureLookup",
    "ProjectNoveltyIndex",
    "ScientificNoveltyAssessment",
    "StagedNoveltyAssessor",
    "build_external_novelty_lookups_from_environment",
    "reserve_external_no_match_portfolio_slot",
    "scientific_fingerprint",
]
