from contextlib import contextmanager

from langchain_core.runnables import RunnableConfig

from app.deps import Deps
from app.events import log_event


def get_deps(config: RunnableConfig) -> Deps:
    return config["configurable"]["deps"]


@contextmanager
def node_events(deps: Deps, run_id: str, node: str, **payload):
    """Bracket a node with node_started / node_finished events for the audit trail."""
    with deps.session_factory() as session:
        log_event(session, run_id, node, "node_started", payload)
    yield
    with deps.session_factory() as session:
        log_event(session, run_id, node, "node_finished", payload)
