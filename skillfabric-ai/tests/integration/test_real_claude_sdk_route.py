from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.builder import BuildConfig, build_graph
from skillfabric.compiled_graph.execution.validation import DeterministicExecutionFlowValidator
from skillfabric.compiled_graph.interface.extraction import DeterministicInterfaceExtractor
from skillfabric.compiled_graph.relations.validation import StaticPairValidator
from skillfabric.router.routing import RouterConfig, route_task
from skillfabric.wiki.explorer.prompting import EXPLORER_PROMPT_ID
from skillfabric.wiki.materializer import build_wiki
from skillfabric.wiki.models import WikiBuildConfig
from tests.unit.fake_embeddings import FakeEmbeddingProvider


@unittest.skipUnless(
    os.environ.get("SKILLFABRIC_REAL_CC_SDK") == "1",
    "set SKILLFABRIC_REAL_CC_SDK=1 to run real Claude Agent SDK smoke tests",
)
class RealClaudeSdkRouteTests(unittest.TestCase):
    def test_real_claude_sdk_routes_from_materialized_query_wiki(self) -> None:
        with TemporaryDirectory(prefix="skillfabric-real-sdk-") as tmp:
            skill_root = Path(tmp) / "skills"
            workspace = Path(tmp) / ".skillfabric"
            _write_synthetic_skills(skill_root)
            build_graph(
                BuildConfig(
                    skill_root=skill_root,
                    workspace=workspace,
                    similar_top_k=2,
                    candidate_top_k=4,
                    validator=StaticPairValidator(
                        {
                            ("skill:synthetic-kpi-extractor", "skill:synthetic-pdf-table-parser"): {
                                "edge_type": "depend_on",
                                "direction": "A->B",
                                "confidence": 0.94,
                                "evidence": [
                                    {
                                        "skill": "skill:synthetic-kpi-extractor",
                                        "line": 6,
                                        "text": "Use this after synthetic-pdf-table-parser has produced CSV tables.",
                                    }
                                ],
                                "reason": "KPI extraction consumes CSV tables produced by PDF parsing.",
                            }
                        }
                    ),
                    interface_extractor=DeterministicInterfaceExtractor(),
                    execution_validator=DeterministicExecutionFlowValidator(),
                    embedding_provider=FakeEmbeddingProvider(),
                    build_id="real-cc-sdk-synthetic",
                )
            )
            build_wiki(WikiBuildConfig(workspace=workspace, use_llm_summaries=False))

            result = route_task(
                RouterConfig(
                    workspace=workspace,
                    query="extract financial KPIs from a PDF report",
                    trace_id="real-cc-sdk-route",
                    explorer_backend="claude-code",
                    explorer_model=os.environ.get("ANTHROPIC_MODEL") or None,
                    strict_explorer=True,
                    max_selected_skills=4,
                )
            )

            trace_dir = workspace / "runs" / "real-cc-sdk-route"
            self.assertEqual(result.provenance, "claude_code")
            self.assertIn("skill:synthetic-pdf-table-parser", result.selected_skill_ids)
            self.assertIn("skill:synthetic-kpi-extractor", result.selected_skill_ids)
            self.assertFalse(any("wiki explorer failed" in warning for warning in result.warnings), result.warnings)
            self.assertFalse(any("skill package validation error" in warning for warning in result.warnings), result.warnings)
            self.assertTrue(
                any(edge.before_skill == "skill:synthetic-pdf-table-parser" for edge in result.required_edges),
                [edge.to_dict() for edge in result.required_edges],
            )
            self.assertTrue((trace_dir / "query_wiki" / "manifest.json").exists())
            self.assertTrue((trace_dir / "cc_explorer" / "skill_package.json").exists())
            self.assertTrue((trace_dir / "cc_explorer" / "validation.json").exists())
            prompt_contract = json.loads((trace_dir / "cc_explorer" / "prompt_contract.json").read_text(encoding="utf-8"))
            prompt_context = json.loads((trace_dir / "cc_explorer" / "prompt_context.json").read_text(encoding="utf-8"))
            self.assertEqual(prompt_contract["prompt_id"], EXPLORER_PROMPT_ID)
            self.assertEqual(prompt_context["prompt_id"], EXPLORER_PROMPT_ID)
            self.assertNotIn("query", prompt_context)


def _write_synthetic_skills(skill_root: Path) -> None:
    _write_skill(
        skill_root / "synthetic-pdf-table-parser" / "SKILL.md",
        """---
name: synthetic-pdf-table-parser
description: Extract synthetic PDF report tables and save normalized CSV output.
---

# Synthetic PDF Table Parser

Use this skill for PDF report table extraction tasks.
It reads `.pdf` reports and produces `.csv` tables for downstream KPI extraction.

## Workflow

1. Parse PDF pages.
2. Extract tabular rows.
3. Write normalized CSV tables.
""",
    )
    _write_skill(
        skill_root / "synthetic-kpi-extractor" / "SKILL.md",
        """---
name: synthetic-kpi-extractor
description: Extract financial KPI values from normalized CSV report tables.
---

# Synthetic KPI Extractor

Use this after `synthetic-pdf-table-parser` has produced CSV tables.
It extracts revenue, margin, cash flow, and other financial KPIs, then writes `kpi.json`.

## Workflow

1. Load CSV tables.
2. Extract KPI values.
3. Write structured JSON metrics.
""",
    )


def _write_skill(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
