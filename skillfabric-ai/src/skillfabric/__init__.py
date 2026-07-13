"""SkillFabric public package."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("skillfabric-ai")
except PackageNotFoundError:  # pragma: no cover - source tree without installation metadata.
    __version__ = "0.1.0"

__all__ = ["SkillFabric", "__version__"]


def __getattr__(name: str):
    if name == "SkillFabric":
        from skillfabric.api import SkillFabric

        return SkillFabric
    raise AttributeError(name)
