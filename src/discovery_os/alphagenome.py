"""Optional AlphaGenome API integration for genomic candidate exploration.

The adapter is deliberately client-only: no AlphaGenome weights or genomic
data are bundled.  The API key is read at runtime and is never included in
provenance or serialized results.  Outputs are normalized into small,
JSON-safe summaries so the orchestration layer can rank many candidates while
retaining the original model response only in the caller's private store.
"""

from __future__ import annotations

import importlib
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from ._compat import StrEnum


class GenomicMode(StrEnum):
    GENOMIC_VARIANT_DISCOVERY = "GENOMIC_VARIANT_DISCOVERY"
    REGULATORY_SEQUENCE_DESIGN = "REGULATORY_SEQUENCE_DESIGN"
    SPLICE_VARIANT_ANALYSIS = "SPLICE_VARIANT_ANALYSIS"
    DISEASE_VARIANT_PRIORITIZATION = "DISEASE_VARIANT_PRIORITIZATION"
    ALPHAGENOME_EVALUATION_ONLY = "ALPHAGENOME_EVALUATION_ONLY"


@dataclass(frozen=True)
class VariantCandidate:
    candidate_id: str
    chromosome: str
    position: int
    reference_bases: str
    alternate_bases: str
    interval_start: int
    interval_end: int
    ontology_terms: tuple[str, ...] = ()
    literature_support: float = 0.0
    source: str = "generated"

    def __post_init__(self) -> None:
        if self.position < 0 or self.interval_start < 0 or self.interval_end <= self.interval_start:
            raise ValueError("genomic coordinates must be non-negative and interval_end > interval_start")
        if not self.reference_bases or not self.alternate_bases:
            raise ValueError("reference_bases and alternate_bases are required")
        if not 0.0 <= self.literature_support <= 1.0:
            raise ValueError("literature_support must be between 0 and 1")


@dataclass(frozen=True)
class AlphaGenomeEvaluation:
    candidate_id: str
    status: str
    expression_change: float | None = None
    splicing_change: float | None = None
    chromatin_change: float | None = None
    contact_map_change: float | None = None
    model_disagreement: float = 0.0
    uncertainty: float = 1.0
    literature_support: float = 0.0
    priority_score: float = 0.0
    output_names: tuple[str, ...] = ()
    error: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "status": self.status,
            "expression_change": self.expression_change,
            "splicing_change": self.splicing_change,
            "chromatin_change": self.chromatin_change,
            "contact_map_change": self.contact_map_change,
            "model_disagreement": self.model_disagreement,
            "uncertainty": self.uncertainty,
            "literature_support": self.literature_support,
            "priority_score": self.priority_score,
            "output_names": list(self.output_names),
            "error": self.error,
            "provenance": dict(self.provenance),
        }


def _finite_numbers(value: Any, *, limit: int = 20_000) -> list[float]:
    """Extract a bounded numeric sample from numpy/pyarrow-like API tracks."""
    if value is None or len(numbers := []) >= limit:
        return numbers
    if isinstance(value, (str, bytes, Mapping)):
        if isinstance(value, Mapping):
            for item in value.values():
                numbers.extend(_finite_numbers(item, limit=limit - len(numbers)))
                if len(numbers) >= limit:
                    break
        return numbers[:limit]
    if isinstance(value, (int, float)):
        return [float(value)] if math.isfinite(float(value)) else []
    try:
        for item in value:
            numbers.extend(_finite_numbers(item, limit=limit - len(numbers)))
            if len(numbers) >= limit:
                break
    except (TypeError, ValueError):
        raw = getattr(value, "values", None)
        if raw is not None and raw is not value:
            numbers.extend(_finite_numbers(raw, limit=limit))
    return numbers[:limit]


def _track_delta(reference: Any, alternate: Any) -> float | None:
    ref = _finite_numbers(getattr(reference, "values", reference))
    alt = _finite_numbers(getattr(alternate, "values", alternate))
    if not ref or not alt:
        return None
    size = min(len(ref), len(alt))
    return float(sum(abs(alt[i] - ref[i]) for i in range(size)) / size)


class AlphaGenomeClient:
    """Thin wrapper around the official ``alphagenome`` client package."""

    def __init__(self, api_key: str | None = None, *, model: Any | None = None) -> None:
        self._api_key = (api_key or os.environ.get("ALPHAGENOME_API_KEY", "")).strip()
        if model is None:
            if not self._api_key:
                raise ValueError("ALPHAGENOME_API_KEY is required at runtime")
            try:
                dna_client = importlib.import_module("alphagenome.models.dna_client")
            except ImportError as exc:
                raise RuntimeError("install the official AlphaGenome client before using genomic mode") from exc
            model = dna_client.create(self._api_key)
        self._model = model

    def evaluate(self, candidate: VariantCandidate) -> AlphaGenomeEvaluation:
        try:
            try:
                genome = importlib.import_module("alphagenome.data.genome")
                interval = genome.Interval(candidate.chromosome, candidate.interval_start, candidate.interval_end)
                variant = genome.Variant(
                    chromosome=candidate.chromosome,
                    position=candidate.position,
                    reference_bases=candidate.reference_bases,
                    alternate_bases=candidate.alternate_bases,
                )
            except ImportError:
                # Enables contract tests and offline adapters without bundling
                # the optional official client package.
                interval = SimpleNamespace(chromosome=candidate.chromosome, start=candidate.interval_start, end=candidate.interval_end)
                variant = SimpleNamespace(chromosome=candidate.chromosome, position=candidate.position, reference_bases=candidate.reference_bases, alternate_bases=candidate.alternate_bases)
            kwargs: dict[str, Any] = {"interval": interval, "variant": variant}
            if candidate.ontology_terms:
                kwargs["ontology_terms"] = list(candidate.ontology_terms)
            outputs = self._model.predict_variant(**kwargs)
            values: dict[str, float | None] = {}
            names: list[str] = []
            for name, output in self._output_pairs(outputs):
                names.append(name)
                values[name] = _track_delta(getattr(output, "reference", None), getattr(output, "alternate", None))
            expression = self._first(values, "rna_seq", "gene_expression", "cage")
            splicing = self._first(values, "splice", "splicing")
            chromatin = self._first(values, "atac", "dnase", "chromatin")
            contact = self._first(values, "contact")
            observed = [x for x in (expression, splicing, chromatin, contact) if x is not None]
            disagreement = (max(observed) - min(observed)) / (max(observed) + 1e-8) if len(observed) > 1 else 0.0
            uncertainty = min(1.0, disagreement + (0.5 if not observed else 0.0))
            score = self.rank_score(expression, splicing, chromatin, disagreement, uncertainty, candidate.literature_support)
            return AlphaGenomeEvaluation(candidate.candidate_id, "success", expression, splicing, chromatin, contact, disagreement, uncertainty, candidate.literature_support, score, tuple(names), provenance={"expert_id": "alphagenome", "api_key": "runtime_only"})
        except Exception as exc:  # API failures are candidate-local and fail closed.
            return AlphaGenomeEvaluation(candidate.candidate_id, "failed", error=f"{type(exc).__name__}: {exc}", literature_support=candidate.literature_support, provenance={"expert_id": "alphagenome"})

    def evaluate_many(self, candidates: Iterable[VariantCandidate]) -> list[AlphaGenomeEvaluation]:
        return sorted((self.evaluate(candidate) for candidate in candidates), key=lambda result: result.priority_score, reverse=True)

    @staticmethod
    def rank_score(expression: float | None, splicing: float | None, chromatin: float | None, disagreement: float, uncertainty: float, literature: float) -> float:
        effects = [x for x in (expression, splicing, chromatin) if x is not None]
        effect = sum(effects) / len(effects) if effects else 0.0
        return float(effect * (1.0 - 0.5 * uncertainty) + 0.2 * literature - 0.1 * disagreement)

    @staticmethod
    def _first(values: Mapping[str, float | None], *names: str) -> float | None:
        return next((value for key, value in values.items() if any(name in key for name in names) and value is not None), None)

    @staticmethod
    def _output_pairs(outputs: Any) -> list[tuple[str, Any]]:
        names = vars(outputs).keys() if hasattr(outputs, "__dict__") else ()
        return [(name, value) for name in names if not name.startswith("_")
                for value in [getattr(outputs, name)]
                if hasattr(value, "reference") and hasattr(value, "alternate")]


def evaluate_genomic_candidates(candidates: Iterable[VariantCandidate], *, api_key: str | None = None) -> list[AlphaGenomeEvaluation]:
    """Evaluate and rank a batch without ever persisting the API key."""
    return AlphaGenomeClient(api_key).evaluate_many(candidates)
