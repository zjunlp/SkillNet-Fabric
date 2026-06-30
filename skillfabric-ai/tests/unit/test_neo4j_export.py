from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.builder import BuildConfig, build_graph
from skillfabric.exporters.neo4j import Neo4jExportConfig, export_neo4j
from tests.unit.fake_embeddings import FakeEmbeddingProvider

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SKILLS = ROOT / "fixtures" / "skills"


class Neo4jExportTests(unittest.TestCase):
    def test_export_neo4j_writes_clean_skill_level_cypher(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            build_graph(
                BuildConfig(
                    skill_root=FIXTURE_SKILLS,
                    workspace=workspace,
                    skip_llm_validation=True,
                    embedding_provider=FakeEmbeddingProvider(),
                    similar_top_k=2,
                    candidate_top_k=4,
                    build_id="neo4j-test",
                )
            )

            result = export_neo4j(Neo4jExportConfig(workspace=workspace, batch_size=3))
            text = result.output_path.read_text(encoding="utf-8")

            self.assertGreater(result.node_count, 0)
            self.assertGreater(result.relationship_count, 0)
            self.assertIn("CREATE CONSTRAINT skillfabric_entity_id", text)
            self.assertIn(":SkillFabricEntity:Skill", text)
            self.assertIn(":SkillFabricEntity:Community", text)
            self.assertIn("SIMILAR_TO", text)
            self.assertIn("MEMBER_OF", text)
            self.assertNotIn(":SkillFabricEntity:SkillInterface", text)
            self.assertNotIn(":SkillFabricEntity:InterfaceField", text)
            self.assertNotIn(":SkillFabricEntity:Artifact", text)
            self.assertNotIn(":SkillFabricEntity:Scenario", text)
            self.assertNotIn("HAS_INTERFACE", text)
            self.assertNotIn("PRODUCES_ARTIFACT", text)

    def test_export_neo4j_can_write_custom_output_path(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Path(tmp) / ".skillfabric"
            output_path = Path(tmp) / "graph.cypher"
            build_graph(
                BuildConfig(
                    skill_root=FIXTURE_SKILLS,
                    workspace=workspace,
                    skip_llm_validation=True,
                    embedding_provider=FakeEmbeddingProvider(),
                    similar_top_k=1,
                    candidate_top_k=2,
                    build_id="neo4j-cli-test",
                )
            )

            result = export_neo4j(
                Neo4jExportConfig(workspace=workspace, output_path=output_path, batch_size=2)
            )

            self.assertEqual(result.output_path, output_path)
            self.assertGreater(result.node_count, 0)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
