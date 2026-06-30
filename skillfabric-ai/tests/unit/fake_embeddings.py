from __future__ import annotations

import hashlib
import math
import re


class FakeEmbeddingProvider:
    """Small stable embedding provider for unit tests."""

    model_id = "test-fake-embedding"

    def __init__(self, dimension: int = 96) -> None:
        self.dimension = dimension

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in re.findall(r"[a-z0-9][a-z0-9_.+-]*", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0:
            return vector
        return [value / norm for value in vector]

    def embed_query(self, query: str) -> list[float]:
        return self.embed(query)
