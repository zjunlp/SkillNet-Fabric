"""Prompt construction for interface term canonicalization."""

from __future__ import annotations

import json
from typing import Any

from skillfabric.compiled_graph.canonicalization.candidates import normalized_candidate_text
from skillfabric.compiled_graph.canonicalization.models import CanonicalizationCluster

CANONICALIZATION_PROMPT_ID = "interface_term_canonicalization_v2"


def build_canonicalization_messages(cluster: CanonicalizationCluster) -> list[dict[str, str]]:
    """Build structured messages for resolving one candidate term group."""

    payload: dict[str, Any] = {
        "prompt_id": CANONICALIZATION_PROMPT_ID,
        "task": "Canonicalize short SkillFabric interface terms that name the same reusable interface object.",
        "context": {
            "interface_term": "A requires or produces field extracted from a skill interface.",
            "canonical_object": "A stable snake_case name shared by terms that refer to the same reusable object.",
            "quality_bar": "Prefer small, precise groups. Put unresolved or weakly supported terms in omitted_term_ids.",
        },
        "input": {
            "cluster_id": cluster.cluster_id,
            "term_count": len(cluster.terms),
        },
        "output_schema": {
            "canonical_objects": [
                {
                    "name": "stable_snake_case_name",
                    "type": "artifact|data|text|report|state|belief_state|planning_state|credential|environment",
                    "term_ids": ["term:id"],
                    "confidence": 0.0,
                }
            ],
            "omitted_term_ids": ["term:id"],
        },
        "terms": [
            {
                "term_id": term.term_id,
                "name": term.name,
                "normalized_name": normalized_candidate_text(term.name),
                "role": term.role,
                "kind": term.kind,
                "description": term.description,
            }
            for term in cluster.terms
        ],
    }
    return [
        {
            "role": "system",
            "content": "You canonicalize short interface terms into stable object names and return JSON only.",
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
