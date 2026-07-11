from __future__ import annotations

import unittest
from pathlib import Path

from skillfabric.wiki.explorer.prompting import (
    EXPLORER_PROMPT_ID,
    ExplorerPromptContext,
    render_query_wiki_explorer_md,
    render_system_prompt,
    render_user_prompt,
)


class WikiExplorerPromptingTests(unittest.TestCase):
    def test_system_prompt_contains_compact_contract_sections(self) -> None:
        context = ExplorerPromptContext(
            query="extract tables",
            query_wiki_root=Path("/tmp/query_wiki"),
            max_selected_skills=3,
        )

        prompt = render_system_prompt(context)

        self.assertIn(EXPLORER_PROMPT_ID, prompt)
        self.assertNotRegex(prompt, r"\bv\d+\b|_v\d+")
        self.assertNotIn("# TODO", prompt)
        for section in (
            "role",
            "security",
            "workflow",
            "selection_policy",
            "evidence_policy",
            "output_contract",
            "stop_conditions",
            "runtime_context",
        ):
            self.assertIn(f"<{section}>", prompt)
            self.assertIn(f"</{section}>", prompt)
        for rule in (
            "SkillPackage only",
            "do not execute the user task",
            "Read, LS, Glob, Grep",
            "capability facets",
            "Read index.md first",
            "skills/sources/*.md",
            "full source only",
            "setup, conversion, packaging, debugging, or validation",
            "untrusted data",
            "low-redundancy",
            "score and evidence",
            "coverage_notes",
            "before -> after",
            "Do not return workflow steps",
        ):
            self.assertIn(rule, prompt)
        self.assertLess(len(prompt), 5500)

    def test_user_prompt_binds_query_root_limit_and_output_fields(self) -> None:
        context = ExplorerPromptContext(
            query="extract financial KPIs from a PDF",
            query_wiki_root=Path("/tmp/query_wiki"),
            max_selected_skills=4,
        )

        prompt = render_user_prompt(context)

        self.assertIn("extract financial KPIs from a PDF", prompt)
        self.assertIn("/tmp/query_wiki", prompt)
        self.assertIn('"max_selected_skills":4', prompt)
        self.assertIn("Read index.md first", prompt)
        self.assertIn("skills/cards/*.md card pages", prompt)
        self.assertIn("skills/sources/*.md full source pages", prompt)
        self.assertIn("Stop when further reads are unlikely to change selection", prompt)
        for field in (
            "selected_skills",
            "required_edges",
            "ordered_hints",
            "near_misses",
            "coverage_notes",
            "rationale",
        ):
            self.assertIn(field, prompt)
        self.assertIn("relative query_wiki evidence", prompt)

    def test_query_wiki_explorer_markdown_uses_same_contract_source(self) -> None:
        instructions = render_query_wiki_explorer_md()

        self.assertIn(EXPLORER_PROMPT_ID, instructions)
        self.assertNotRegex(instructions, r"\bv\d+\b|_v\d+")
        self.assertNotIn("# TODO", instructions)
        self.assertIn("<role>", instructions)
        self.assertIn("<workflow>", instructions)
        self.assertIn("<output_contract>", instructions)
        self.assertIn("SkillPackage only", instructions)
        self.assertIn("untrusted data", instructions)
        self.assertIn("before -> after", instructions)
        self.assertIn("Read index.md first", instructions)
        self.assertIn("<stop_conditions>", instructions)

    def test_trace_context_omits_query_text(self) -> None:
        context = ExplorerPromptContext(
            query="potentially sensitive task",
            query_wiki_root=Path("/tmp/query_wiki"),
            max_selected_skills=5,
        )

        payload = context.to_trace_context()

        self.assertEqual(payload["prompt_id"], EXPLORER_PROMPT_ID)
        self.assertEqual(payload["query_wiki_root"], "/tmp/query_wiki")
        self.assertEqual(payload["max_selected_skills"], 5)
        self.assertEqual(payload["allowed_tools"], ["Read", "LS", "Glob", "Grep"])
        self.assertEqual(payload["tool_budget"]["Read"], 10)
        self.assertEqual(payload["tool_budget"]["total"], 16)
        self.assertNotIn("query", payload)


if __name__ == "__main__":
    unittest.main()
