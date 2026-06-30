from __future__ import annotations

import unittest

from skillfabric.indexing.bm25 import _fts_query


class BM25IndexTests(unittest.TestCase):
    def test_fts_query_filters_stopwords(self) -> None:
        query = _fts_query("heat the potato then put it in a receptacle")

        self.assertEqual(query, "heat OR potato OR put OR receptacle")


if __name__ == "__main__":
    unittest.main()
