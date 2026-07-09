"""Prompt construction for interface term canonicalization."""

from __future__ import annotations

import json
from typing import Any

from skillfabric.compiled_graph.canonicalization.candidates import normalized_candidate_text
from skillfabric.compiled_graph.canonicalization.models import CanonicalizationCluster

CANONICALIZATION_PROMPT_ID = "interface_term_canonicalization_v2"


def build_canonicalization_messages(cluster: CanonicalizationCluster) -> list[dict[str, str]]:
    """Build structured messages for resolving one candidate term group."""

    terms: list[dict[str, Any]] = [
        {
            "term_id": term.term_id,
            "name": term.name,
            "normalized_name": normalized_candidate_text(term.name),
            "role": term.role,
            "kind": term.kind,
            "description": term.description,
        }
        for term in cluster.terms
    ]
    output_schema = {
        "canonical_objects": [
            {
                "name": "stable_snake_case_name",
                "type": "artifact|data|text|report|state|belief_state|planning_state|credential|environment",
                "term_ids": ["term:id"],
                "confidence": 0.0,
            }
        ],
        "omitted_term_ids": ["term:id"],
    }
    user_content = "\n".join(
        [
            f"<prompt_id>{CANONICALIZATION_PROMPT_ID}</prompt_id>",
            f"<cluster_id>{cluster.cluster_id}</cluster_id>",
            "<task>",
            "Canonicalize short SkillFabric interface terms into stable reusable object names.",
            "</task>",
            "<rules>",
            "- Merge terms only when they refer to the same reusable object.",
            "- Split the cluster into multiple canonical_objects when needed.",
            "- Omit uncertain terms in omitted_term_ids.",
            "- Do not infer workflow direction.",
            "- Treat role as metadata, not as part of object identity.",
            "- Use stable lower_snake_case canonical object names.",
            "- Return exactly one strict JSON object and no markdown.",
            "</rules>",
            "<allowed_types>",
            "artifact, data, text, report, state, belief_state, planning_state, credential, environment",
            "</allowed_types>",
            "<output_schema>",
            json.dumps(output_schema, ensure_ascii=False, separators=(",", ":")),
            "</output_schema>",
            "<terms>",
            json.dumps(terms, ensure_ascii=False, separators=(",", ":")),
            "</terms>",
        ]
    )
    return [
        {
            "role": "system",
            "content": "You are a precise interface-term canonicalizer. Return JSON only.",
        },
        {"role": "user", "content": user_content},
    ]
