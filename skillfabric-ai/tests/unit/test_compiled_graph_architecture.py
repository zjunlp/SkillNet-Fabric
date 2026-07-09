from __future__ import annotations

import importlib
import unittest


class CompiledGraphArchitectureTests(unittest.TestCase):
    def test_compiled_graph_is_the_primary_build_entrypoint(self) -> None:
        builder = importlib.import_module("skillfabric.compiled_graph.builder")
        models = importlib.import_module("skillfabric.compiled_graph.models")
        relations = importlib.import_module("skillfabric.compiled_graph.relations")

        self.assertTrue(hasattr(builder, "build_graph"))
        self.assertTrue(hasattr(builder, "BuildConfig"))
        self.assertTrue(hasattr(models, "GraphDocument"))
        self.assertTrue(hasattr(relations, "build_similar_edges"))
        self.assertTrue(hasattr(relations, "enforce_depend_on_acyclicity"))
        self.assertFalse(hasattr(relations, "generate_relation_candidates"))

    def test_removed_graph_modules_are_not_public_entrypoints(self) -> None:
        for module_name in [
            "skillfabric.graph.builder",
            "skillfabric.graph.candidates",
            "skillfabric.graph.validation",
        ]:
            with self.subTest(module_name=module_name):
                with self.assertRaises(ModuleNotFoundError):
                    importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
