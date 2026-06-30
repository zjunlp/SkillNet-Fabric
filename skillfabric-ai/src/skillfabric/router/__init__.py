"""Router package for selecting skill packages from tasks."""

from skillfabric.router.bundle import RouterBundle, RouterBundleConfig, build_router_bundle
from skillfabric.router.models import RouteEdge, RouteResult, RouterQuery, RouteSelectedSkill

__all__ = [
    "RouteEdge",
    "RouteResult",
    "RouteSelectedSkill",
    "RouterBundle",
    "RouterBundleConfig",
    "RouterConfig",
    "RouterQuery",
    "build_router_bundle",
    "route_task",
]


def __getattr__(name: str):
    if name in {"RouterConfig", "route_task"}:
        from skillfabric.router.routing import RouterConfig, route_task

        return {"RouterConfig": RouterConfig, "route_task": route_task}[name]
    raise AttributeError(name)
