"""SkillFabric public package."""

from importlib.metadata import PackageNotFoundError, version

from skillfabric.api import SkillFabric

try:
    __version__ = version("skillfabric-ai")
except PackageNotFoundError:  # pragma: no cover - source tree without installation metadata.
    __version__ = "0.2.0"

__all__ = ["SkillFabric", "__version__"]
