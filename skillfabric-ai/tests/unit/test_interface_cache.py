from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.compiled_graph.interface.extraction import extract_skill_interfaces
from skillfabric.llm_jobs import LLMJobOptions
from tests.unit.relation_helpers import make_skill


class CountingFallbackExtractor:
    model_id = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def extract(self, skill):
        self.calls += 1
        return {
            "capability_summary": f"summary {self.calls}",
            "uses_tools": [{"name": "python", "kind": "tool", "confidence": 0.8}],
        }


class FlakyExtractor:
    model_id = "flaky"

    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    def extract(self, skill):
        self.calls[skill.id] = self.calls.get(skill.id, 0) + 1
        if self.calls[skill.id] == 1:
            raise RuntimeError("transient")
        return {
            "capability_summary": f"summary {skill.name}",
            "uses_tools": [{"name": "python", "kind": "tool", "confidence": 0.8}],
        }


class InterfaceCacheTests(unittest.TestCase):
    def test_content_hash_unchanged_reuses_cached_interface(self) -> None:
        skill = make_skill("skill:python", "python", "Run Python.")
        extractor = CountingFallbackExtractor()

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "interface_cache.json"
            first = extract_skill_interfaces([skill], extractor=extractor, cache_path=cache_path)
            second = extract_skill_interfaces([skill], extractor=extractor, cache_path=cache_path)

        self.assertEqual(extractor.calls, 1)
        self.assertEqual(first[0].interface.capability_summary, "summary 1")
        self.assertEqual(second[0].interface.capability_summary, "summary 1")
        self.assertEqual(second[0].interface.provenance, "cache")

    def test_interface_extraction_retries_and_writes_cache(self) -> None:
        skills = [
            make_skill("skill:python", "python", "Run Python."),
            make_skill("skill:report", "report", "Write reports."),
        ]
        extractor = FlakyExtractor()

        with TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "interface_cache.json"
            records = extract_skill_interfaces(
                skills,
                extractor=extractor,
                cache_path=cache_path,
                job_options=LLMJobOptions(concurrency=2, max_retries=1, progress_every=0),
            )
            self.assertTrue(cache_path.exists())

        self.assertTrue(all(record.accepted for record in records))
        self.assertEqual(extractor.calls["skill:python"], 2)
        self.assertEqual(extractor.calls["skill:report"], 2)


if __name__ == "__main__":
    unittest.main()
