from __future__ import annotations

import unittest

from skillfabric.compiled_graph.relations.mentions import extract_skill_mentions
from tests.unit.relation_helpers import make_skill


class RelationMentionTests(unittest.TestCase):
    def test_extracts_name_id_wikilink_and_alias_mentions(self) -> None:
        parser = make_skill("skill:pdf-table-parser", "pdf-table-parser", "Parse PDF tables.")
        writer = make_skill(
            "skill:report-writer",
            "report-writer",
            "\n".join(
                [
                    "Use after `skill:pdf-table-parser` has produced CSV tables.",
                    "This also works with [[pdf-table-parser]].",
                    "The PDF table parser alias should match too.",
                ]
            ),
        )

        mentions = extract_skill_mentions([parser, writer])
        mention_types = {item.mention_type for item in mentions}

        self.assertEqual({item.to_skill for item in mentions}, {"skill:pdf-table-parser"})
        self.assertIn("id", mention_types)
        self.assertIn("wikilink", mention_types)
        self.assertIn("alias", mention_types)
        self.assertTrue(all(item.line > 0 and item.text for item in mentions))

    def test_does_not_match_substrings_or_self_mentions(self) -> None:
        alpha = make_skill("skill:alpha", "alpha", "alpha should not mention itself.")
        alphabet = make_skill("skill:alphabet", "alphabet", "The word alphabet soup should not match the shorter skill.")

        mentions = extract_skill_mentions([alpha, alphabet])

        self.assertEqual(mentions, [])

    def test_infers_direction_hints_from_relation_phrases(self) -> None:
        parser = make_skill("skill:pdf-table-parser", "pdf-table-parser", "Parse PDF tables.")
        kpi = make_skill(
            "skill:financial-kpi-extractor",
            "financial-kpi-extractor",
            "Use after pdf-table-parser has produced CSV tables.",
        )
        tester = make_skill(
            "skill:testing-python",
            "testing-python",
            "This skill composes with analyze-ci when CI fails.",
        )
        ci = make_skill("skill:analyze-ci", "analyze-ci", "Analyze CI logs.")

        mentions = extract_skill_mentions([parser, kpi, tester, ci])
        by_pair = {(item.from_skill, item.to_skill): item.direction_hint for item in mentions}

        self.assertEqual(by_pair[("skill:financial-kpi-extractor", "skill:pdf-table-parser")], "A->B")
        self.assertEqual(by_pair[("skill:testing-python", "skill:analyze-ci")], "undirected")


if __name__ == "__main__":
    unittest.main()
