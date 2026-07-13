# Selection-Only Explorer And Prompt-Only Planner

## Goal

Keep graph-assisted discovery while letting task-time reasoning decide which skills and relations matter. Produce one self-contained execution prompt without exposing or persisting an intermediate workflow DAG.

## Architecture

The route path is:

```text
compiled graph -> bounded query wiki -> SDK selection explorer -> route evidence -> one LLM planner call -> execution_prompt.md
```

The Explorer selects task-relevant skills after reading wiki cards, sources, and graph evidence. Its structured result contains only selected skills, near misses, coverage gaps, pages read, and a rationale. It does not declare dependencies, composition links, or a total skill order.

The Route projects graph edges whose endpoints are both selected into relation evidence. These edges inform the Planner but do not expand the selection, force prerequisites, or become validation constraints.

The Planner receives the original task, selected skill contracts and full sources, relation evidence, and coverage gaps. One ordinary LLM call returns exactly `{"execution_prompt": "..."}`. The prompt may describe serial, parallel, synthesis, and verification behavior in natural language, but there is no workflow schema or DAG artifact.

## Validation

Selection validation remains strict about trust boundaries: schema shape, selection limit, manifest membership, selectable status, evidence paths, evidence ownership, pages actually read, duplicate ids, near-miss conflicts, and explicit coverage gaps for an empty selection.

Validation does not require graph prerequisite closure, graph-consistent relation declarations, or a total ordering. Planner validation checks only the exact output schema and a non-empty execution prompt.

## Context Budget

The Planner must receive every selected skill source. Before the external call, SkillFabric estimates the complete message size and fails explicitly when it exceeds a configurable `planner_context_max_tokens`. It does not truncate, summarize, retry with less context, or use a fallback planner.

## Prompt Contract

Each LLM prompt separates fixed policy from untrusted data, defines relation or field semantics, gives a short decision process, states the exact output schema, and prohibits hidden reasoning or surrounding prose. Data is serialized inside explicit XML sections. Prompts avoid unsupported claims, redundant instructions, and task-specific heuristics.

## Artifacts And Public API

The finalized planner package contains `route.json`, selected skill cards and sources, `planner_request.json`, `planner_output.json`, `planner_validation.json`, and `execution_prompt.md`. `workflow_plan.json` and all workflow-step validators are removed.

`SkillFabric.plan()` performs Route when needed, prepares context, invokes the Planner once, and returns the finalized execution package. Prepare/finalize remain explicit building blocks for callers that execute the same strict planner contract externally; they do not support legacy workflow output.

## Failure Semantics

Missing skills, invalid evidence, malformed planner output, context overflow, missing credentials, and provider errors fail visibly. There is no deterministic selection, prompt renderer, schema compatibility branch, silent truncation, or fallback output.
