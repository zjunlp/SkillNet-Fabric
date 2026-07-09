from __future__ import annotations

import unittest

from skillfabric.storage import Workspace
from skillfabric.wiki.models import WikiBuildConfig


class WorkspaceLayoutTests(unittest.TestCase):
    def test_workspace_exposes_only_current_artifact_directories(self) -> None:
        workspace = Workspace(".skillfabric")

        for removed_name in ("registry_dir", "index_dir", "interfaces_dir", "execution_dir"):
            self.assertFalse(hasattr(workspace, removed_name), removed_name)

    def test_wiki_build_config_has_no_debug_page_option(self) -> None:
        with self.assertRaises(TypeError):
            WikiBuildConfig(include_debug_pages=True)


if __name__ == "__main__":
    unittest.main()
