from __future__ import annotations

import os
import unittest
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SRC = PACKAGE_ROOT / "src" / "skillfabric"


class ExperimentalParityTests(unittest.TestCase):
    def test_core_route_and_package_modules_match_experimental_source(self) -> None:
        experimental_root = os.environ.get("SKILLFABRIC_EXPERIMENTAL_ROOT")
        if not experimental_root:
            self.skipTest("set SKILLFABRIC_EXPERIMENTAL_ROOT to run local experimental parity checks")
        experimental_src = Path(experimental_root) / "src" / "skillfabric"
        if not experimental_src.exists():
            self.skipTest(f"experimental source tree not found: {experimental_src}")

        equivalent_modules = (
            "wiki/query_wiki.py",
            "orchestrator/package.py",
            "orchestrator/agent_run_spec.py",
            "orchestrator/renderers/codex.py",
        )
        for module in equivalent_modules:
            with self.subTest(module=module):
                self.assertEqual(
                    (PUBLIC_SRC / module).read_text(encoding="utf-8"),
                    (experimental_src / module).read_text(encoding="utf-8"),
                )


if __name__ == "__main__":
    unittest.main()
