"""Prompt construction for pool-level canonicalization."""

from __future__ import annotations

import json

from skillfabric.compiled_graph.canonicalization.models import CanonicalizationCluster

CANONICALIZATION_PROMPT_ID = "canonicalization_operational_objects"


def build_canonicalization_messages(cluster: CanonicalizationCluster) -> list[dict[str, str]]:
    """Build LiteLLM messages for canonicalizing one raw-term cluster."""

    payload = {
        "todo": (
            "Normalize one candidate component of raw SkillContract terms into reusable operational objects for routing "
            "and workflow planning. Keep only objects that improve cross-skill matching or coverage explanation."
        ),
        "task": (
            "Canonicalize a candidate-graph component of SkillContract requires/produces terms into operational objects "
            "that are useful for routing and workflow planning. This is a pool-level normalization step, "
            "not a per-skill summary. Candidate edges are lexical or semantic evidence, not final decisions."
        ),
        "prompt_id": CANONICALIZATION_PROMPT_ID,
        "input": {
            "cluster": "One connected candidate component containing raw requires/produces terms, candidate edges, ambiguity flags, and evidence.",
            "raw_terms": "Candidate terms may be noisy, generic, local-only, example-only, or semantically different despite lexical similarity.",
            "candidate_edges": "Lexical or semantic neighbor evidence. Use it as merge evidence, not as a final decision.",
        },
        "output": {
            "format": "Return one strict JSON object, with no markdown, comments, or extra keys.",
            "required_top_level_keys": ["canonical_objects", "assignments", "rejected_terms"],
            "purpose": (
                "Canonical objects support cross-skill routing and orchestration. Rejected terms document why noisy terms should not be promoted."
            ),
        },
        "workflow": [
            "Step 1: Group terms by the operational thing a downstream agent can require, produce, inspect, verify, or hand off.",
            "Step 2: Separate final deliverables, intermediate artifacts, data objects, reports, text, environment prerequisites, credentials, world states, belief states, and planning states.",
            "Step 3: Use candidate_edges as evidence, but split terms that are only topically or lexically similar.",
            "Step 4: Merge aliases only when they are substitutable in routing or workflow planning.",
            "Step 5: Reject generic, local-only, placeholder, section-heading, example-only, and unsupported terms.",
            "Step 6: Decide promotion by cross-skill utility, not by whether a phrase appears in a skill.",
            "Step 7: Assign each raw term exactly once: either to one canonical object or to rejected_terms.",
            "Step 8: Return a small, high-precision registry that favors useful workflow gates over exhaustive preservation.",
        ],
        "decision_workflow": [
            "Group raw terms by the operational thing a downstream agent can require, produce, inspect, verify, or hand off.",
            "Separate final deliverables, intermediate artifacts, data objects, environment prerequisites, credentials, world states, belief states, and planning states.",
            "Merge aliases only when they are substitutable in routing or workflow planning.",
            "Reject terms that are too generic, local-only, example-only, or unsupported by reusable operational semantics.",
            "Promote only objects that can help connect skills or explain coverage; do not preserve every phrase from the source contracts.",
        ],
        "output_schema": {
            "canonical_objects": [
                {
                    "canonical_name": "stable_snake_case_name",
                    "type": "artifact|data|world_state|belief_state|planning_state|credential|environment|text|report",
                    "description": "",
                    "aliases": [],
                    "promoted": True,
                    "confidence": 0.0,
                    "reason": "",
                }
            ],
            "assignments": [
                {
                    "raw_name": "",
                    "canonical_name": "",
                    "confidence": 0.0,
                    "reason": "",
                }
            ],
            "rejected_terms": [
                {
                    "raw_name": "",
                    "reason": "generic|local_only|ambiguous|unsupported",
                }
            ],
        },
        "rules": [
            "Return JSON only.",
            "Use stable snake_case canonical names. Do not include spaces, slashes, punctuation, or skill names.",
            "Every assignment raw_name must exactly match one input term name.",
            "A raw term must appear in exactly one of assignments or rejected_terms.",
            "Use candidate_edges as merge evidence, but reject weak candidate edges when the raw terms name different operational objects.",
            "If the component is marked ambiguous, be conservative and split or reject instead of forcing one canonical object.",
            "Merge aliases aggressively when the terms describe the same operational object, even if their raw kind differs across artifact, data, text, state, report, or environment.",
            "Do not merge terms that merely belong to the same broad domain. A report, chart, slide deck, spreadsheet, validation log, and source dataset are different operational objects.",
            "Choose the canonical type by operational semantics, not by the raw kind. For example, an inventory-held condition is state even if one raw term is text.",
            "Never merge belief_state or planning_state into world_state. A remembered, observed, inferred, or planned fact is not the same as a physical environment state.",
            "object_permanence_state is belief_state unless the skill actually performs and confirms a take/pickup action. Do not canonicalize object_permanence_state to object_in_inventory.",
            "structured_task_parse, sequential_sub_objective_plan, parsed_goal, and routing decisions are planning_state or data; they are not workflow-enabling world_state objects.",
            "Only world_state objects may represent physical gates such as object_in_inventory, receptacle_open, cleaned_object_state, heated_object_state, cooled_object_state, or agent_at_target_location.",
            "Prefer reusable state/data names for workflow gates: object_in_inventory, agent_at_target_location, receptacle_open, target_object_located, appliance_ready, cleaned_object_state, heated_object_state, cooled_object_state, task_verified.",
            "Do not keep separate canonical objects for spelling variants such as object in inventory vs object_in_inventory, target receptacle vs target_receptacle_identifier, current observation vs environment_observation.",
            "Reject or demote generic parameters that cannot connect producer to consumer by themselves: object, data, result, output, content, file, text, target, item, command.",
            "Reject terms that only name an example, placeholder, section heading, or local implementation detail unless they identify a reusable artifact or state.",
            "Promote=true only when the object can support cross-skill routing or workflow planning. Strong cases: at least one producing skill and one requiring skill; or a state/credential/environment shared by multiple skills as a real workflow gate.",
            "Promote=false for local-only outputs, one-off reports, isolated action strings, examples, and final task-completion descriptions that no other skill can consume.",
            "Do not merge semantically different objects just because they share common words. A target object, target receptacle, target tool, and task target are different.",
            "If all terms are local-only or generic, return no canonical_objects, no assignments, and put each raw term in rejected_terms with a short reason.",
            "Use confidence >= 0.9 only for evidence-backed exact or near-exact aliases; use lower confidence when an assignment is inferred from context.",
        ],
        "constraints": [
            "Do not create canonical objects merely to preserve every raw phrase.",
            "Do not merge terms only because they share a broad domain, file family, or common word.",
            "Do not merge belief_state or planning_state into world_state.",
            "Do not promote object, data, result, output, content, file, text, target, item, or command unless the cluster provides a specific reusable concept.",
            "Do not include skill names in canonical object names.",
            "If the component is ambiguous, prefer split or reject over a forced merge.",
        ],
        "cluster": cluster.to_dict(),
    }
    return [
        {
            "role": "system",
            "content": (
                "You are the SkillFabric pool-level canonicalizer. Your job is to reduce noisy "
                "requires/produces terms into a small set of reusable operational objects. "
                "Prioritize downstream routing and workflow planning quality over preserving every raw phrase."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]
