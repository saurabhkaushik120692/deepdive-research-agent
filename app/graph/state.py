import operator
from typing import Annotated, TypedDict


class ResearchState(TypedDict):
    """Deliberately thin: content lives in SQLite/Chroma, the graph only carries ids."""

    run_id: str
    question: str
    round: int
    pending_sq_ids: list[int]
    researched: Annotated[list[int], operator.add]
