from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from skillfabric.storage import Workspace


class WorkspaceLockTests(unittest.TestCase):
    def test_stale_lock_file_is_reclaimed(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.root.mkdir(parents=True)
            workspace.lock_path.write_text("999999999", encoding="utf-8")

            with workspace.lock():
                self.assertEqual(workspace.lock_path.read_text(encoding="utf-8"), str(os.getpid()))

            self.assertFalse(workspace.lock_path.exists())

    def test_active_lock_file_is_preserved(self) -> None:
        with TemporaryDirectory() as tmp:
            workspace = Workspace(Path(tmp) / ".skillfabric")
            workspace.root.mkdir(parents=True)
            workspace.lock_path.write_text(str(os.getpid()), encoding="utf-8")

            with (
                self.assertRaisesRegex(RuntimeError, "build lock already exists"),
                workspace.lock(),
            ):
                pass

            self.assertEqual(workspace.lock_path.read_text(encoding="utf-8"), str(os.getpid()))


if __name__ == "__main__":
    unittest.main()
