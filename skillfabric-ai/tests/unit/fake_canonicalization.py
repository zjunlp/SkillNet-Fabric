from __future__ import annotations

from skillfabric.compiled_graph.canonicalization.candidates import (
    contract_object_type,
    normalized_candidate_text,
)
from skillfabric.compiled_graph.canonicalization.models import CanonicalizationCluster


class FixtureCanonicalizationProvider:
    """Test-only provider that maps fixture interface terms by normalized name."""

    model_id = "fixture-canonicalizer"

    def __init__(self) -> None:
        self.calls: list[CanonicalizationCluster] = []

    def canonicalize(self, cluster: CanonicalizationCluster) -> dict[str, object]:
        self.calls.append(cluster)
        return {
            "canonical_objects": [
                {
                    "name": _canonical_name(term.name),
                    "type": contract_object_type(term.kind),
                    "term_ids": [term.term_id],
                    "confidence": 0.9,
                }
                for term in cluster.terms
                if _canonical_name(term.name)
            ],
            "omitted_term_ids": [
                term.term_id for term in cluster.terms if not _canonical_name(term.name)
            ],
        }


def _canonical_name(value: str) -> str:
    return "_".join(normalized_candidate_text(value).split())
