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
from pathlib import Path
from typing import Callable, Protocol

from pydantic import Field, model_validator

from ._compat import StrEnum
from .crystal_identity import (
    CanonicalCrystalStructure,
    CrystalIdentityError,
    PymatgenRequiredError,
    canonical_structure_hash,
    canonicalize_crystal_structure,
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
        if (
            self.structure_match_count is not None
            and self.structure_match_count != len(self.matches)
        ):
            raise ValueError("structure_match_count must equal the preserved match records")
        if self.closest_match_id is not None and not self.matches:
            raise ValueError("closest_match_id requires a structure match")
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


class ExternalNoveltyOutcome(StrictSchema):
    provider_id: Identifier
    status: NoveltyStatus
    method: Identifier
    query_count: int = Field(default=0, ge=0)
    matches: list[NoveltyMatch] = Field(default_factory=list)
    reason: str | None = Field(default=None, max_length=2_000)
    composition_match_count: int | None = Field(default=None, ge=0)
    structure_match_count: int | None = Field(default=None, ge=0)
    closest_match_id: str | None = Field(default=None, max_length=512)
    closest_distance: float | None = Field(default=None, ge=0.0)

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
        )
        return self


class ExternalNoveltyLookup(Protocol):
    provider_id: str

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

    def __init__(self, external_lookup: ExternalNoveltyLookup | None = None) -> None:
        self.external_lookup = external_lookup

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
        if self.external_lookup is None:
            return NoveltyStageResult(
                stage=NoveltyStage.EXTERNAL_DATABASE,
                status=NoveltyStatus.UNKNOWN,
                method="external-structure-lookup-v1",
                reason="external_lookup_not_configured",
            )
        try:
            outcome = self.external_lookup.lookup(candidate)
        except Exception as exc:  # provider failures must never become no-match
            return NoveltyStageResult(
                stage=NoveltyStage.EXTERNAL_DATABASE,
                status=NoveltyStatus.UNKNOWN,
                method="external-structure-lookup-v1",
                query_count=1,
                reason=f"external_lookup_failed:{type(exc).__name__}",
            )
        return NoveltyStageResult(
            stage=NoveltyStage.EXTERNAL_DATABASE,
            status=outcome.status,
            method=outcome.method,
            query_count=outcome.query_count,
            matches=outcome.matches,
            reason=outcome.reason,
            composition_match_count=outcome.composition_match_count,
            structure_match_count=outcome.structure_match_count,
            closest_match_id=outcome.closest_match_id,
            closest_distance=outcome.closest_distance,
        )


class MaterialsProjectStructureLookup:
    """Optional ``mp-api`` structure matcher with an injected runtime credential.

    The official client's ``find_structure`` route performs tolerance-aware
    structure matching.  This adapter never falls back to formula-only matching,
    because composition equality is not sufficient evidence of a duplicate
    crystal structure.
    """

    provider_id = "materials-project"

    def __init__(
        self,
        api_key: str | None,
        *,
        ltol: float = 0.2,
        stol: float = 0.3,
        angle_tol: float = 5.0,
        rester_factory: Callable[[str], object] | None = None,
    ) -> None:
        if ltol <= 0 or stol <= 0 or angle_tol <= 0:
            raise ValueError("structure-match tolerances must be positive")
        self._api_key = api_key.strip() if api_key else ""
        self.ltol = float(ltol)
        self.stol = float(stol)
        self.angle_tol = float(angle_tol)
        self._rester_factory = rester_factory

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        **kwargs: object,
    ) -> "MaterialsProjectStructureLookup":
        values = os.environ if environ is None else environ
        return cls(values.get("MP_API_KEY"), **kwargs)

    def lookup(self, candidate: Candidate) -> ExternalNoveltyOutcome:
        if not self._api_key:
            return self._unknown("materials_project_api_key_not_configured", query_count=0)
        cif = _representation(candidate, RepresentationKind.CIF)
        if cif is None:
            return self._unknown("candidate_has_no_cif_representation", query_count=0)
        try:
            factory = self._rester_factory or _materials_project_rester_factory()
        except (ImportError, ModuleNotFoundError):
            return self._unknown("materials_project_client_not_installed", query_count=0)

        client: object | None = None
        try:
            client = factory(self._api_key)
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
        except Exception as exc:
            return self._unknown(
                f"materials_project_lookup_failed:{type(exc).__name__}",
                query_count=1,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        identifiers = _external_identifiers(raw)
        if not identifiers:
            return ExternalNoveltyOutcome(
                provider_id=self.provider_id,
                status=NoveltyStatus.NO_MATCH,
                method="materials-project-find-structure-v1",
                query_count=1,
                composition_match_count=(
                    len(composition_ids) if composition_ids is not None else None
                ),
                structure_match_count=0,
            )
        return ExternalNoveltyOutcome(
            provider_id=self.provider_id,
            status=NoveltyStatus.MATCH,
            method="materials-project-find-structure-v1",
            query_count=1,
            composition_match_count=(
                len(composition_ids) if composition_ids is not None else None
            ),
            structure_match_count=len(identifiers),
            closest_match_id=identifiers[0],
            closest_distance=None,
            matches=[
                NoveltyMatch(
                    source_id=self.provider_id,
                    record_id=item,
                    match_kind="tolerance-aware-structure-match",
                    metadata={
                        "ltol": str(self.ltol),
                        "stol": str(self.stol),
                        "angle_tol": str(self.angle_tol),
                        "distance": "not_returned_by_materials_project_find_structure",
                    },
                )
                for item in identifiers
            ],
        )

    def _unknown(self, reason: str, *, query_count: int) -> ExternalNoveltyOutcome:
        return ExternalNoveltyOutcome(
            provider_id=self.provider_id,
            status=NoveltyStatus.UNKNOWN,
            method="materials-project-find-structure-v1",
            query_count=query_count,
            reason=reason,
        )

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


def _materials_project_rester_factory() -> Callable[[str], object]:
    from mp_api.client import MPRester

    return lambda api_key: MPRester(api_key, mute_progress_bars=True)


__all__ = [
    "ExternalNoveltyLookup",
    "ExternalNoveltyOutcome",
    "MaterialsProjectStructureLookup",
    "NoveltyMatch",
    "NoveltyStage",
    "NoveltyStageResult",
    "NoveltyStatus",
    "ProjectNoveltyIndex",
    "ScientificNoveltyAssessment",
    "StagedNoveltyAssessor",
    "scientific_fingerprint",
]
